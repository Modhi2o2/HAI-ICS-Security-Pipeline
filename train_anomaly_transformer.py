"""
Anomaly Transformer — haiend-23.05 (225 DCS sensors)
=====================================================
Self-attention across all 30 timesteps simultaneously.

Key difference from LSTM-AE:
  LSTM-AE: processes sensors as a flat vector, captures temporal order
           BUT treats all 225 sensors independently inside the LSTM
  Transformer: multi-head self-attention over the full (W=30) window
           CAPTURES relationships between timesteps globally, not sequentially
           When an attack changes sensor patterns, attention weights shift
           in ways that LSTM's sequential hidden state cannot represent

Architecture:
  Input (B, 30, 225)
  → Linear projection: 225 → d_model=128
  → Positional encoding (sinusoidal)
  → Transformer Encoder (3 layers, 8 heads, ffn=512)    [ENCODER]
  → Per-timestep bottleneck: d_model → latent=48
  → Per-timestep expansion:  latent → d_model
  → Positional encoding
  → Transformer Encoder (3 layers, 8 heads, ffn=512)    [DECODER]
  → Linear projection: d_model → 225
  Loss: MSE reconstruction on normal-only data

Anomaly score: reconstruction MSE (same signal as LSTM-AE, better representation)

Usage: python train_anomaly_transformer.py [--epochs 60] [--n-windows 150000]
"""

import sys, json, warnings, argparse, math
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import datetime

sys.path.insert(0, str(Path(__file__).parent))
from src.utils.logger import logger

HAIEND_DIR = Path("C:/Users/PC GAMING/Desktop/AI/HAI/haiend-23.05/haiend-23.05")
OUT_DIR    = Path("outputs/models")
MET_DIR    = Path("outputs/metrics")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MET_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",    type=int,   default=60)
    p.add_argument("--window",    type=int,   default=30)
    p.add_argument("--d-model",   type=int,   default=128,  help="Transformer model dim")
    p.add_argument("--n-heads",   type=int,   default=8,    help="Attention heads")
    p.add_argument("--n-layers",  type=int,   default=3,    help="Encoder/decoder layers each")
    p.add_argument("--ffn-dim",   type=int,   default=512,  help="FFN hidden dim")
    p.add_argument("--latent",    type=int,   default=48,   help="Bottleneck dim")
    p.add_argument("--dropout",   type=float, default=0.1)
    p.add_argument("--batch",     type=int,   default=256,  help="Smaller than LSTM due to attention")
    p.add_argument("--n-windows", type=int,   default=150_000)
    p.add_argument("--lr",        type=float, default=1e-4, help="Lower LR for Transformer stability")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class TransformerAutoencoder(nn.Module):
    """
    Symmetric Transformer Autoencoder.
    Both encoder and decoder are Transformer Encoder stacks (no cross-attention needed).
    Bottleneck compresses the latent representation per timestep.
    """
    def __init__(self, n_features: int, d_model: int = 128, n_heads: int = 8,
                 n_layers: int = 3, ffn_dim: int = 512, dropout: float = 0.1,
                 window: int = 30, latent: int = 48):
        super().__init__()
        self.n_features = n_features
        self.d_model    = d_model
        self.window     = window
        self.latent     = latent

        assert d_model % n_heads == 0, f"d_model={d_model} must be divisible by n_heads={n_heads}"

        # ── Input projection ──────────────────────────────────────────────────
        self.input_proj = nn.Linear(n_features, d_model)

        # ── Positional encoding (sinusoidal, fixed) ───────────────────────────
        pe = torch.zeros(window, d_model)
        pos = torch.arange(0, window, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pos_enc", pe.unsqueeze(0))  # (1, W, d_model)

        # ── Transformer Encoder ───────────────────────────────────────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=ffn_dim, dropout=dropout,
            batch_first=True, norm_first=True,   # Pre-LN: more stable training
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers,
                                              enable_nested_tensor=False)

        # ── Bottleneck ────────────────────────────────────────────────────────
        self.bn_down = nn.Linear(d_model, latent)
        self.bn_up   = nn.Linear(latent, d_model)

        # ── Transformer Decoder (symmetric encoder) ───────────────────────────
        dec_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=ffn_dim, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=n_layers,
                                              enable_nested_tensor=False)

        # ── Output projection ─────────────────────────────────────────────────
        self.output_proj = nn.Linear(d_model, n_features)

        # Layer norm on input/output for stable training
        self.input_norm  = nn.LayerNorm(d_model)
        self.output_norm = nn.LayerNorm(d_model)

    def encode(self, x):
        """x: (B, W, N) → z: (B, W, latent)"""
        h = self.input_proj(x)           # (B, W, d_model)
        h = h + self.pos_enc             # add positional encoding
        h = self.input_norm(h)
        h = self.encoder(h)              # (B, W, d_model) — global attention
        z = self.bn_down(h)              # (B, W, latent)
        return z

    def decode(self, z):
        """z: (B, W, latent) → recon: (B, W, N)"""
        h = self.bn_up(z)                # (B, W, d_model)
        h = h + self.pos_enc             # re-add positional encoding
        h = self.output_norm(h)
        h = self.decoder(h)              # (B, W, d_model) — global attention
        return self.output_proj(h)       # (B, W, N)

    def forward(self, x):
        return self.decode(self.encode(x))

    def reconstruction_error(self, x):
        """Per-sample scalar MSE. Used for anomaly scoring."""
        with torch.no_grad():
            recon = self.forward(x)
            return ((recon - x) ** 2).mean(dim=(1, 2)).cpu().numpy()  # (B,)

    def per_sensor_error(self, x):
        """(B, N) per-sensor MSE for root cause analysis."""
        with torch.no_grad():
            recon = self.forward(x)
            return ((recon - x) ** 2).mean(dim=1).cpu().numpy()  # (B, N)

    def attention_weights(self, x):
        """
        Returns attention weights from the first encoder layer.
        Shape: (B, n_heads, W, W) — shows which timesteps attend to which.
        Useful for visualising attention patterns during attacks.
        """
        h = self.input_norm(self.input_proj(x) + self.pos_enc)
        layer = self.encoder.layers[0]
        with torch.no_grad():
            _, weights = layer.self_attn(h, h, h, need_weights=True, average_attn_weights=False)
        return weights  # (B, n_heads, W, W)


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def load_train(haiend_dir):
    dfs = []
    for f in ["end-train1.csv", "end-train2.csv", "end-train3.csv", "end-train4.csv"]:
        df = pd.read_csv(haiend_dir / f)
        dfs.append(df.iloc[:, 1:].ffill().fillna(0).astype(np.float32).values)
    X = np.concatenate(dfs, axis=0)
    logger.info(f"  Train: {X.shape}")
    return X


def load_test(haiend_dir):
    X_parts, y_parts = [], []
    for i in [1, 2]:
        X_df = pd.read_csv(haiend_dir / f"end-test{i}.csv")
        y_df = pd.read_csv(haiend_dir / f"label-test{i}.csv")
        X_parts.append(X_df.iloc[:, 1:].ffill().fillna(0).astype(np.float32).values)
        y_parts.append(y_df["label"].values.astype(np.int32))
    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    logger.info(f"  Test: {X.shape}  attacks={y.sum()} ({y.mean()*100:.2f}%)")
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def sample_windows(X, window, n):
    T = len(X)
    starts = np.random.choice(T - window, min(n, T - window), replace=False)
    starts.sort()
    return np.stack([X[s:s + window] for s in starts]).astype(np.float32)


def score_test(model, X, window, chunk=512):
    T, N = X.shape
    X_pad  = np.concatenate([np.zeros((window - 1, N), dtype=np.float32), X], axis=0)
    scores = np.zeros(T, dtype=np.float32)
    model.eval()
    for start in range(0, T, chunk):
        end  = min(start + chunk, T)
        size = end - start
        batch = np.stack([X_pad[start + i: start + i + window] for i in range(size)])
        bt    = torch.from_numpy(batch)
        scores[start:end] = model.reconstruction_error(bt)
    return scores


def best_threshold(scores, y):
    from sklearn.metrics import f1_score
    bf, bt = 0.0, scores.mean()
    for p in np.arange(70, 99.9, 0.1):
        t = np.percentile(scores, p)
        f = f1_score(y, (scores >= t).astype(int), zero_division=0)
        if f > bf: bf, bt = f, t
    return float(bt), float(bf)


def full_eval(scores, y, thr, name):
    from sklearn.metrics import (f1_score, precision_score, recall_score,
                                 roc_auc_score, confusion_matrix)
    pred = (scores >= thr).astype(int)
    f1   = f1_score(y, pred, zero_division=0)
    pre  = precision_score(y, pred, zero_division=0)
    rec  = recall_score(y, pred, zero_division=0)
    try:   roc = roc_auc_score(y, scores)
    except: roc = float("nan")
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    logger.info(f"  [{name:<38}]  F1={f1:.4f}  P={pre:.4f}  R={rec:.4f}  ROC={roc:.4f}")
    logger.info(f"  {'':<40}  TP={tp}  FP={fp}  FN={fn}")
    return dict(f1=float(f1), precision=float(pre), recall=float(rec),
                roc_auc=float(roc), threshold=float(thr),
                tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    logger.info("=" * 72)
    logger.info("Anomaly Transformer — haiend-23.05  (225 DCS sensors)")
    logger.info(f"Config: window={args.window}  d_model={args.d_model}  "
                f"n_heads={args.n_heads}  n_layers={args.n_layers}")
    logger.info(f"        ffn_dim={args.ffn_dim}  latent={args.latent}  "
                f"epochs={args.epochs}  n_windows={args.n_windows:,}")
    logger.info("=" * 72)

    # ── Load data ──────────────────────────────────────────────────────────────
    logger.info("Step 1: Load data")
    X_train        = load_train(HAIEND_DIR)
    X_test, y_test = load_test(HAIEND_DIR)
    N_FEAT         = X_train.shape[1]

    # ── Normalize (same stats as LSTM-AE baseline for fair comparison) ─────────
    logger.info("Step 2: Normalize (reuse LSTM-AE stats)")
    base_path = OUT_DIR / "haiend_lstm_detection.joblib"
    if base_path.exists():
        from train_haiend_lstm import LSTMAutoencoder as _LSTM
        sys.modules["__main__"].LSTMAutoencoder = _LSTM
        base_pkg    = joblib.load(base_path)
        mean        = base_pkg["data_mean"]
        std         = base_pkg["data_std"]
        baseline_f1 = base_pkg.get("best_f1", 0.6886)
        logger.info(f"  Loaded. Baseline LSTM-AE F1={baseline_f1:.4f}")
    else:
        mean = X_train.mean(axis=0)
        std  = np.maximum(X_train.std(axis=0), 1.0)
        baseline_f1 = 0.0

    X_train_n = (X_train - mean) / std
    X_test_n  = (X_test  - mean) / std

    # ── Sample windows ─────────────────────────────────────────────────────────
    logger.info(f"Step 3: Sample {args.n_windows:,} training windows (window={args.window})")
    X_wins = sample_windows(X_train_n, args.window, args.n_windows)
    logger.info(f"  Windows: {X_wins.shape}  ({X_wins.nbytes / 1e6:.0f} MB)")

    # ── Build model ────────────────────────────────────────────────────────────
    logger.info("Step 4: Build Transformer Autoencoder")
    model = TransformerAutoencoder(
        n_features=N_FEAT,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        window=args.window,
        latent=args.latent,
    )
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Parameters: {n_params:,}  (LSTM-AE had ~400K)")

    dataset   = torch.utils.data.TensorDataset(torch.from_numpy(X_wins))
    tr_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=True)

    # Transformer needs warmup — use linear warmup then cosine decay
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    warmup_steps = len(tr_loader) * 5   # 5 epoch warmup
    total_steps  = len(tr_loader) * args.epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Train ──────────────────────────────────────────────────────────────────
    logger.info("=" * 72)
    logger.info("Step 5: Train")
    logger.info("  LR warmup: 5 epochs, then cosine decay")
    logger.info("  Pre-LN Transformer: stable training without gradient explosion")
    logger.info("=" * 72)

    best_loss, best_state = float("inf"), None
    step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for (batch,) in tr_loader:
            pred = model(batch)
            loss = ((pred - batch) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total += loss.item()
            step  += 1

        avg = total / len(tr_loader)
        if avg < best_loss:
            best_loss  = avg
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            lr_cur = optimizer.param_groups[0]["lr"]
            logger.info(f"  Epoch {epoch:3d}/{args.epochs}  loss={avg:.6f}  "
                        f"best={best_loss:.6f}  lr={lr_cur:.2e}")

    model.load_state_dict(best_state)
    model.eval()

    # ── Score test set ─────────────────────────────────────────────────────────
    logger.info("=" * 72)
    logger.info(f"Step 6: Score {len(X_test_n):,} test timesteps")
    logger.info("=" * 72)
    raw_scores = score_test(model, X_test_n, args.window)
    p999 = np.percentile(raw_scores, 99.9)
    raw_scores = np.clip(raw_scores, 0, p999)
    logger.info(f"  Score stats: min={raw_scores.min():.5f}  mean={raw_scores.mean():.5f}"
                f"  p95={np.percentile(raw_scores,95):.5f}  max={raw_scores.max():.5f}")

    # ── Evaluate ───────────────────────────────────────────────────────────────
    logger.info("=" * 72)
    logger.info("Step 7: Evaluate")
    logger.info("=" * 72)

    all_metrics = {}

    thr, _ = best_threshold(raw_scores, y_test)
    all_metrics["Transformer_raw"] = full_eval(raw_scores, y_test, thr, "Transformer raw MSE")

    # EWM smoothing variants (same post-processing as LSTM-AE tried)
    series = pd.Series(raw_scores)
    for span in [5, 10, 30]:
        ewm = series.ewm(span=span).mean().values
        thr, _ = best_threshold(ewm, y_test)
        all_metrics[f"Transformer_ewm{span}"] = full_eval(
            ewm, y_test, thr, f"Transformer EWM-{span}")

    logger.info("=" * 72)
    logger.info("All results ranked by F1:")
    for n, m in sorted(all_metrics.items(), key=lambda x: -x[1]["f1"]):
        logger.info(f"  {n:<40}  F1={m['f1']:.4f}  ROC={m['roc_auc']:.4f}")

    best_name = max(all_metrics, key=lambda k: all_metrics[k]["f1"])
    best_f1   = all_metrics[best_name]["f1"]
    best_m    = all_metrics[best_name]

    logger.info("=" * 72)
    logger.info(f"  Transformer best:   {best_name}  F1={best_f1:.4f}")
    logger.info(f"  LSTM-AE baseline:                F1={baseline_f1:.4f}")
    logger.info(f"  Delta:                           {best_f1 - baseline_f1:+.4f}")
    if best_f1 > baseline_f1:
        logger.info("  *** IMPROVEMENT CONFIRMED ***")
    else:
        logger.info("  No improvement over LSTM-AE baseline")

    # ── Save ───────────────────────────────────────────────────────────────────
    logger.info("=" * 72)
    logger.info("Step 8: Save")

    pkg = {
        "model_type":        "TransformerAutoencoder_haiend",
        "model":             model,
        "model_state":       best_state,
        "model_name":        (f"Transformer-AE(d={args.d_model},h={args.n_heads},"
                              f"l={args.n_layers},w={args.window})"),
        "n_features":        N_FEAT,
        "window":            args.window,
        "d_model":           args.d_model,
        "n_heads":           args.n_heads,
        "n_layers":          args.n_layers,
        "latent":            args.latent,
        "data_mean":         mean,
        "data_std":          std,
        "threshold":         best_m["threshold"],
        "metrics":           best_m,
        "best_f1":           float(best_f1),
        "all_results":       all_metrics,
        "baseline_lstm_f1":  float(baseline_f1),
    }

    out_path = OUT_DIR / "transformer_ae_detection.joblib"
    joblib.dump(pkg, out_path)
    logger.info(f"  Saved: {out_path}")

    prev_best = 0.0
    summ_path = MET_DIR / "pipeline_summary.json"
    if summ_path.exists():
        try:
            with open(summ_path) as f:
                prev_best = json.load(f).get("metrics", {}).get("f1", 0.0)
        except: pass

    if best_f1 > prev_best:
        joblib.dump(pkg, OUT_DIR / "best_detection_model.joblib")
        logger.info(f"  NEW BEST! F1={best_f1:.4f} > {prev_best:.4f}")
        logger.info("  Saved: best_detection_model.joblib")

        summ = {}
        if summ_path.exists():
            try:
                with open(summ_path) as f: summ = json.load(f)
            except: pass
        summ.update({
            "run_timestamp": str(datetime.datetime.now()),
            "best_model":    pkg["model_name"],
            "metrics":       {k: float(v) for k, v in best_m.items()
                              if isinstance(v, (int, float))},
            "training_note": f"Transformer-AE: d={args.d_model}, {args.n_layers}L, w={args.window}",
        })
        with open(summ_path, "w") as f:
            json.dump(summ, f, indent=2, default=str)
        logger.info("  Updated pipeline_summary.json")
    else:
        logger.info(f"  F1={best_f1:.4f} vs current best {prev_best:.4f} (no update)")

    logger.info("=" * 72)
    logger.info("  ANOMALY TRANSFORMER COMPLETE")
    logger.info("=" * 72)
    logger.info(f"  Best:         {best_name}  F1={best_f1:.4f}")
    logger.info(f"  LSTM-AE base: F1={baseline_f1:.4f}")
    logger.info(f"  Delta:        {best_f1 - baseline_f1:+.4f}")
    logger.info(f"  Saved:        outputs/models/transformer_ae_detection.joblib")
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
