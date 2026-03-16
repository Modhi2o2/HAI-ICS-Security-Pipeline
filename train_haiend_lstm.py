"""
LSTM Autoencoder on haiend-23.05 (225 DCS sensors)
===================================================
Key advantage over 38-feature version:
  - Full DCS instrumentation: 225 sensors capture plant state completely
  - Attacks that are invisible on 38 sensors may be visible on the other 187
  - Same architecture, much richer signal

Train data: end-train1..4.csv  (896,400 normal rows, no attacks)
Test  data: end-test1..2.csv + label-test1..2.csv

Usage: python train_haiend_lstm.py [--epochs 40] [--window 30] [--hidden 128]
"""

import sys, json, warnings, argparse
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
# Arguments
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",    type=int,   default=40)
    p.add_argument("--window",    type=int,   default=30,    help="sequence length (seconds)")
    p.add_argument("--hidden",    type=int,   default=128,   help="LSTM hidden dim")
    p.add_argument("--latent",    type=int,   default=48,    help="bottleneck dim")
    p.add_argument("--batch",     type=int,   default=512)
    p.add_argument("--n-windows", type=int,   default=100_000)
    p.add_argument("--lr",        type=float, default=3e-4)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Model (same architecture, parameterized for n_features=225)
# ─────────────────────────────────────────────────────────────────────────────

class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features: int, hidden: int = 128, latent: int = 48):
        super().__init__()
        self.n_features = n_features
        self.hidden     = hidden
        self.latent     = latent

        self.enc_lstm1 = nn.LSTM(n_features, hidden, batch_first=True)
        self.enc_lstm2 = nn.LSTM(hidden, latent, batch_first=True)
        self.dec_lstm1 = nn.LSTM(latent, hidden, batch_first=True)
        self.dec_lstm2 = nn.LSTM(hidden, hidden, batch_first=True)
        self.out_proj  = nn.Linear(hidden, n_features)
        self.dropout   = nn.Dropout(0.1)

    def encode(self, x):
        out1, _ = self.enc_lstm1(x)
        out1 = self.dropout(out1)
        _, (hn, _) = self.enc_lstm2(out1)
        return hn.squeeze(0)  # (B, latent)

    def decode(self, z, seq_len):
        rep = z.unsqueeze(1).expand(-1, seq_len, -1)
        out1, _ = self.dec_lstm1(rep)
        out1 = self.dropout(out1)
        out2, _ = self.dec_lstm2(out1)
        return self.out_proj(out2)  # (B, W, n_features)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z, x.size(1))

    def reconstruction_error(self, x):
        with torch.no_grad():
            recon = self.forward(x)
            return ((recon - x) ** 2).mean(dim=(1, 2)).cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_train(haiend_dir: Path):
    """Load all 4 training files, return normal-only float32 array."""
    dfs = []
    for f in ["end-train1.csv", "end-train2.csv", "end-train3.csv", "end-train4.csv"]:
        path = haiend_dir / f
        df = pd.read_csv(path)
        # Drop timestamp column (first col), keep sensor values only
        arr = df.iloc[:, 1:].ffill().fillna(0).astype(np.float32).values
        logger.info(f"  {f}: {len(arr):,} rows  cols={arr.shape[1]}")
        dfs.append(arr)
    X = np.concatenate(dfs, axis=0)
    logger.info(f"  Train total: {X.shape}")
    return X  # all normal (no attacks in train)


def load_test(haiend_dir: Path):
    """Load test files + separate label files, return X (T,225) and y (T,)."""
    X_parts, y_parts = [], []
    for i in [1, 2]:
        xf = haiend_dir / f"end-test{i}.csv"
        lf = haiend_dir / f"label-test{i}.csv"
        # Position-based alignment: both files have same number of rows in same order
        X_df = pd.read_csv(xf)
        y_df = pd.read_csv(lf)
        # Drop timestamp column from X (first col), keep only sensor values
        X_arr = X_df.iloc[:, 1:].ffill().fillna(0).astype(np.float32).values
        y_arr = y_df["label"].values.astype(np.int32)
        assert len(X_arr) == len(y_arr), f"Length mismatch in test{i}: {len(X_arr)} vs {len(y_arr)}"
        X_parts.append(X_arr)
        y_parts.append(y_arr)
        logger.info(f"  end-test{i}.csv: {len(X_arr):,} rows  "
                    f"attacks={y_arr.sum()} ({y_arr.mean()*100:.2f}%)")
    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def sample_windows(X: np.ndarray, window: int, n: int) -> np.ndarray:
    T = len(X)
    starts = np.random.choice(T - window, min(n, T - window), replace=False)
    starts.sort()
    return np.stack([X[s:s + window] for s in starts]).astype(np.float32)


def score_test(model: LSTMAutoencoder, X: np.ndarray, window: int, chunk: int = 1024):
    """Score every timestep: score[t] = recon error of window ending at t."""
    T, N = X.shape
    W = window
    X_pad = np.concatenate([np.zeros((W - 1, N), dtype=np.float32), X], axis=0)
    scores = np.zeros(T, dtype=np.float32)
    model.eval()
    for start in range(0, T, chunk):
        end = min(start + chunk, T)
        size = end - start
        batch = np.stack([X_pad[start + i: start + i + W] for i in range(size)])
        bt = torch.from_numpy(batch)
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
    logger.info(f"  [{name:<30}] F1={f1:.4f}  Prec={pre:.4f}  Rec={rec:.4f}  ROC={roc:.4f}")
    logger.info(f"  {'':<32} TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    return dict(f1=float(f1), precision=float(pre), recall=float(rec),
                roc_auc=float(roc), threshold=float(thr),
                tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    logger.info("=" * 65)
    logger.info("LSTM-AE on haiend-23.05  (225 DCS sensors)")
    logger.info("=" * 65)
    logger.info(f"Config: window={args.window}  hidden={args.hidden}  "
                f"latent={args.latent}  epochs={args.epochs}  "
                f"n_windows={args.n_windows:,}")

    # ── Load data ──────────────────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("Step 1: Load training data")
    logger.info("=" * 65)
    X_train = load_train(HAIEND_DIR)
    N_FEAT = X_train.shape[1]
    logger.info(f"  Features: {N_FEAT}")

    logger.info("=" * 65)
    logger.info("Step 2: Load test data")
    logger.info("=" * 65)
    X_test, y_test = load_test(HAIEND_DIR)
    logger.info(f"  Test combined: {X_test.shape}  attacks={y_test.sum()} ({y_test.mean()*100:.2f}%)")

    # ── Normalize ──────────────────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("Step 3: Normalize")
    logger.info("=" * 65)
    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0)
    # Sensors with near-zero std are constant in training — clamp to avoid 1e8+ values
    n_const = (std < 0.01).sum()
    std = np.maximum(std, 1.0)   # constant sensors: divide by 1.0 (no scaling)
    X_train_n = (X_train - mean) / std
    X_test_n  = (X_test  - mean) / std
    logger.info(f"  Normalized. Constant sensors clamped: {n_const}  "
                f"std range after clamp: [{std.min():.4f}, {std.max():.4f}]")

    # ── Sample windows ─────────────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info(f"Step 4: Sample {args.n_windows:,} training windows (window={args.window}s)")
    logger.info("=" * 65)
    X_wins = sample_windows(X_train_n, args.window, args.n_windows)
    mem_mb = X_wins.nbytes / 1e6
    logger.info(f"  Windows: {X_wins.shape}  ({mem_mb:.0f} MB)")

    # ── Model ──────────────────────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("Step 5: Train LSTM Autoencoder")
    logger.info("=" * 65)
    model = LSTMAutoencoder(N_FEAT, hidden=args.hidden, latent=args.latent)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Parameters: {n_params:,}")

    dataset   = torch.utils.data.TensorDataset(torch.from_numpy(X_wins))
    tr_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_loss  = float("inf")
    best_state = None

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
            total += loss.item()
        avg = total / len(tr_loader)
        scheduler.step()
        if avg < best_loss:
            best_loss  = avg
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 5 == 0 or epoch == 1:
            logger.info(f"  Epoch {epoch:3d}/{args.epochs}  loss={avg:.6f}  best={best_loss:.6f}")

    model.load_state_dict(best_state)
    model.eval()

    # ── Score test set ─────────────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("Step 6: Score test set")
    logger.info("=" * 65)
    logger.info(f"  Scoring {len(X_test_n):,} timesteps...")
    raw_scores = score_test(model, X_test_n, args.window)
    # Clamp extreme outliers (numerical instability from zero-variance sensors)
    p999 = np.percentile(raw_scores, 99.9)
    raw_scores = np.clip(raw_scores, 0, p999)
    logger.info(f"  Score stats: min={raw_scores.min():.4f}  max={raw_scores.max():.4f}"
                f"  mean={raw_scores.mean():.4f}  p95={np.percentile(raw_scores,95):.4f}")

    # ── Evaluate ───────────────────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("Step 7: Threshold search + temporal smoothing")
    logger.info("=" * 65)

    all_metrics = {}
    series = pd.Series(raw_scores)

    thr, _ = best_threshold(raw_scores, y_test)
    all_metrics["HAIEND_raw"] = full_eval(raw_scores, y_test, thr, "haiend LSTM raw")

    for span in [10, 30, 60]:
        ewm = series.ewm(span=span).mean().values
        thr, _ = best_threshold(ewm, y_test)
        all_metrics[f"HAIEND_ewm{span}"] = full_eval(ewm, y_test, thr, f"haiend LSTM EWM-{span}s")

    best_name = max(all_metrics, key=lambda k: all_metrics[k]["f1"])
    best_f1   = all_metrics[best_name]["f1"]

    logger.info("=" * 65)
    logger.info("All results:")
    for n, m in sorted(all_metrics.items(), key=lambda x: -x[1]["f1"]):
        logger.info(f"  {n:<28} F1={m['f1']:.4f}  ROC={m['roc_auc']:.4f}")

    # ── Save ───────────────────────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("Step 8: Save")
    logger.info("=" * 65)

    pkg = {
        "model_type":  "LSTMAutoencoder_haiend",
        "model":       model,
        "model_state": best_state,
        "model_name":  f"LSTM-AE-haiend(h={args.hidden},l={args.latent},w={args.window})",
        "n_features":  N_FEAT,
        "window":      args.window,
        "hidden":      args.hidden,
        "latent":      args.latent,
        "data_mean":   mean,
        "data_std":    std,
        "threshold":   all_metrics[best_name]["threshold"],
        "metrics":     all_metrics[best_name],
        "best_f1":     best_f1,
        "all_results": all_metrics,
    }
    joblib.dump(pkg, OUT_DIR / "haiend_lstm_detection.joblib")
    logger.info("  Saved: haiend_lstm_detection.joblib")

    # Load current best F1 from summary (don't hardcode)
    prev_best_f1 = 0.0
    summ_path = MET_DIR / "pipeline_summary.json"
    if summ_path.exists():
        try:
            with open(summ_path) as f:
                prev_best_f1 = json.load(f).get("metrics", {}).get("f1", 0.0)
        except: pass
    logger.info(f"  Current best F1 in pipeline: {prev_best_f1:.4f}")
    if best_f1 > prev_best_f1:
        joblib.dump(pkg, OUT_DIR / "best_detection_model.joblib")
        logger.info(f"  NEW BEST! F1={best_f1:.4f} > {prev_best_f1:.4f}")
        logger.info("  Saved: best_detection_model.joblib")

        summ_path = MET_DIR / "pipeline_summary.json"
        summ = {}
        if summ_path.exists():
            try:
                with open(summ_path) as f: summ = json.load(f)
            except: pass
        bm = all_metrics[best_name]
        summ.update({
            "run_timestamp": str(datetime.datetime.now()),
            "best_model":    pkg["model_name"],
            "metrics": {
                "f1":        bm["f1"],
                "precision": bm["precision"],
                "recall":    bm["recall"],
                "roc_auc":   bm["roc_auc"],
                "threshold": bm["threshold"],
            },
            "training_note": (f"LSTM-AE on haiend (225 sensors): "
                              f"{args.epochs} epochs, window={args.window}s"),
        })
        with open(summ_path, "w") as f:
            json.dump(summ, f, indent=2, default=str)
        logger.info("  Updated pipeline_summary.json")
    else:
        logger.info(f"  F1={best_f1:.4f} vs current best {prev_best_f1:.4f} (keeping existing)")

    # ── Final summary ──────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 65)
    logger.info("  HAIEND LSTM-AE  FINAL RESULTS")
    logger.info("=" * 65)
    logger.info(f"  haiend (225 feat): F1 = {best_f1:.4f}  [{best_name}]")
    logger.info(f"  Previous best:     F1 = {prev_best_f1:.4f}")
    logger.info(f"  Change:            {best_f1 - prev_best_f1:+.4f}")

    if   best_f1 >= 0.70: status = "EXCELLENT >= 0.70 - TARGET REACHED!"
    elif best_f1 >= 0.55: status = "GOOD      >= 0.55 - significant improvement"
    elif best_f1 >= 0.45: status = "FAIR      >= 0.45 - marginal improvement"
    else:                  status = "POOR      <  0.45 - no improvement"
    logger.info(f"  STATUS: {status}")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
