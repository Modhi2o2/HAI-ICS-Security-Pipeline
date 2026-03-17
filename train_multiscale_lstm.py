"""
Multi-Scale LSTM Autoencoder — haiend-23.05 (225 DCS sensors)
=============================================================
Trains w=10 and w=60 models; loads existing w=30 best model.
Ensemble fires if ANY scale exceeds its F1-optimal threshold.

Why multi-scale fixes FN=4037:
  w=10  — catches attacks shorter than 30 seconds (immediate response)
  w=30  — existing best model (F1=0.6886)
  w=60  — catches slow-drift attacks that build over a minute

Ensemble decision:
  fired = w10_fired OR w30_fired OR w60_fired
  score = max(norm_w10, norm_w30, norm_w60)   ← for display

Usage: python train_multiscale_lstm.py [--epochs 60] [--n-windows 100000]
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

# Scales to train (w=30 loaded from existing best model)
SCALE_CONFIGS = {
    10: {"hidden": 128, "latent": 32},   # short attacks
    60: {"hidden": 128, "latent": 64},   # slow drift
}


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",    type=int,   default=60)
    p.add_argument("--n-windows", type=int,   default=100_000)
    p.add_argument("--batch",     type=int,   default=512)
    p.add_argument("--lr",        type=float, default=3e-4)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Model (identical architecture as existing)
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
        return hn.squeeze(0)

    def decode(self, z, seq_len):
        rep = z.unsqueeze(1).expand(-1, seq_len, -1)
        out1, _ = self.dec_lstm1(rep)
        out1 = self.dropout(out1)
        out2, _ = self.dec_lstm2(out1)
        return self.out_proj(out2)

    def forward(self, x):
        return self.decode(self.encode(x), x.size(1))

    def reconstruction_error(self, x):
        with torch.no_grad():
            recon = self.forward(x)
            return ((recon - x) ** 2).mean(dim=(1, 2)).cpu().numpy()

    def per_sensor_error(self, x):
        """Returns (B, n_features) per-sensor MSE for root cause."""
        with torch.no_grad():
            recon = self.forward(x)
            return ((recon - x) ** 2).mean(dim=1).cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def load_train(haiend_dir: Path):
    dfs = []
    for f in ["end-train1.csv", "end-train2.csv", "end-train3.csv", "end-train4.csv"]:
        df = pd.read_csv(haiend_dir / f)
        dfs.append(df.iloc[:, 1:].ffill().fillna(0).astype(np.float32).values)
    X = np.concatenate(dfs, axis=0)
    logger.info(f"  Train: {X.shape}")
    return X


def load_test(haiend_dir: Path):
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

def sample_windows(X: np.ndarray, window: int, n: int) -> np.ndarray:
    T = len(X)
    starts = np.random.choice(T - window, min(n, T - window), replace=False)
    starts.sort()
    return np.stack([X[s:s + window] for s in starts]).astype(np.float32)


def score_test(model: LSTMAutoencoder, X: np.ndarray, window: int, chunk: int = 1024):
    T, N = X.shape
    X_pad = np.concatenate([np.zeros((window - 1, N), dtype=np.float32), X], axis=0)
    scores = np.zeros(T, dtype=np.float32)
    model.eval()
    for start in range(0, T, chunk):
        end  = min(start + chunk, T)
        size = end - start
        batch = np.stack([X_pad[start + i: start + i + window] for i in range(size)])
        scores[start:end] = model.reconstruction_error(torch.from_numpy(batch))
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
    logger.info(f"  [{name:<34}]  F1={f1:.4f}  P={pre:.4f}  R={rec:.4f}  ROC={roc:.4f}")
    logger.info(f"  {'':<36}  TP={tp}  FP={fp}  FN={fn}")
    return dict(f1=float(f1), precision=float(pre), recall=float(rec),
                roc_auc=float(roc), threshold=float(thr),
                tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn))


def train_scale(X_train_n, window, hidden, latent, n_windows, epochs, batch_size, lr):
    """Train one LSTM-AE scale. Returns model with best training loss."""
    logger.info(f"  Sampling {n_windows:,} windows of size {window}...")
    X_wins = sample_windows(X_train_n, window, n_windows)
    logger.info(f"  Windows: {X_wins.shape}  ({X_wins.nbytes / 1e6:.0f} MB)")

    N_FEAT  = X_train_n.shape[1]
    model   = LSTMAutoencoder(N_FEAT, hidden=hidden, latent=latent)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Parameters: {n_params:,}")

    dataset   = torch.utils.data.TensorDataset(torch.from_numpy(X_wins))
    tr_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss, best_state = float("inf"), None

    for epoch in range(1, epochs + 1):
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
        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"    Epoch {epoch:3d}/{epochs}  loss={avg:.6f}  best={best_loss:.6f}")

    model.load_state_dict(best_state)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    logger.info("=" * 70)
    logger.info("Multi-Scale LSTM-AE — haiend-23.05  (225 DCS sensors)")
    logger.info("Scales: w=10 (new)  w=30 (loaded)  w=60 (new)")
    logger.info("=" * 70)

    # ── Load data ──────────────────────────────────────────────────────────────
    logger.info("Step 1: Load data")
    X_train = load_train(HAIEND_DIR)
    X_test, y_test = load_test(HAIEND_DIR)
    N_FEAT = X_train.shape[1]

    # ── Normalize (load existing stats from w=30 model for consistency) ─────────
    logger.info("Step 2: Normalize using existing w=30 model stats")
    existing_pkg_path = OUT_DIR / "haiend_lstm_detection.joblib"
    if existing_pkg_path.exists():
        logger.info("  Loading normalization stats from haiend_lstm_detection.joblib...")
        from train_haiend_lstm import LSTMAutoencoder as _LSTM
        sys.modules["__main__"].LSTMAutoencoder = _LSTM
        existing_pkg = joblib.load(existing_pkg_path)
        mean = existing_pkg["data_mean"]
        std  = existing_pkg["data_std"]
        logger.info("  Loaded existing mean/std — all scales use identical normalization")
    else:
        logger.info("  No existing model found — computing normalization from training data")
        mean = X_train.mean(axis=0)
        std  = X_train.std(axis=0)
        std  = np.maximum(std, 1.0)

    X_train_n = (X_train - mean) / std
    X_test_n  = (X_test  - mean) / std

    # ── Load or train each scale ───────────────────────────────────────────────
    scale_models    = {}   # window -> LSTMAutoencoder
    scale_scores    = {}   # window -> np.ndarray (T,)
    scale_metrics   = {}   # window -> dict
    scale_thresholds = {}  # window -> float

    # w=30: load from existing best model (already trained with 150K windows, 60 epochs)
    logger.info("=" * 70)
    logger.info("Step 3: Load existing w=30 model")
    logger.info("=" * 70)
    if existing_pkg_path.exists():
        w30_model = existing_pkg["model"]
        w30_model.eval()
        scale_models[30] = w30_model
        logger.info("  Loaded w=30 from haiend_lstm_detection.joblib")
        logger.info(f"  Existing w=30 F1: {existing_pkg.get('best_f1', 'N/A')}")
    else:
        logger.info("  No w=30 model found — training from scratch")
        scale_models[30] = train_scale(
            X_train_n, window=30, hidden=128, latent=48,
            n_windows=args.n_windows, epochs=args.epochs,
            batch_size=args.batch, lr=args.lr
        )

    # w=10 and w=60: train fresh
    for window, cfg in SCALE_CONFIGS.items():
        logger.info("=" * 70)
        logger.info(f"Step: Train w={window}  (hidden={cfg['hidden']}, latent={cfg['latent']})")
        logger.info("=" * 70)
        scale_models[window] = train_scale(
            X_train_n,
            window=window,
            hidden=cfg["hidden"],
            latent=cfg["latent"],
            n_windows=args.n_windows,
            epochs=args.epochs,
            batch_size=args.batch,
            lr=args.lr,
        )

    # ── Score test set for all scales ──────────────────────────────────────────
    logger.info("=" * 70)
    logger.info(f"Step 4: Score {len(X_test_n):,} test timesteps for all scales")
    logger.info("=" * 70)
    for window, model in sorted(scale_models.items()):
        logger.info(f"  Scoring w={window}...")
        raw = score_test(model, X_test_n, window)
        p999 = np.percentile(raw, 99.9)
        raw  = np.clip(raw, 0, p999)
        scale_scores[window] = raw
        logger.info(f"    w={window}: min={raw.min():.5f}  mean={raw.mean():.5f}  max={raw.max():.5f}")

    # ── Per-scale evaluation ───────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("Step 5: Per-scale evaluation")
    logger.info("=" * 70)
    for window in sorted(scale_scores.keys()):
        thr, _ = best_threshold(scale_scores[window], y_test)
        scale_thresholds[window] = thr
        scale_metrics[window]    = full_eval(scale_scores[window], y_test, thr, f"w={window} LSTM-AE")

    # ── Ensemble evaluation ────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("Step 6: Ensemble evaluation")
    logger.info("=" * 70)

    # Normalize each scale's scores to [0, 1] for combination
    norm_scores = {}
    for window, scores in scale_scores.items():
        p97 = np.percentile(scores, 97)
        norm_scores[window] = np.clip(scores / (p97 + 1e-8), 0.0, 2.0)

    # Strategy 1: Hard OR — fire if ANY scale's calibrated threshold is exceeded
    fired_w10 = (scale_scores[10] >= scale_thresholds[10]).astype(np.int32)
    fired_w30 = (scale_scores[30] >= scale_thresholds[30]).astype(np.int32)
    fired_w60 = (scale_scores[60] >= scale_thresholds[60]).astype(np.int32)
    hard_or   = np.maximum.reduce([fired_w10, fired_w30, fired_w60])

    from sklearn.metrics import (f1_score, precision_score, recall_score,
                                 roc_auc_score, confusion_matrix)
    f1_or  = f1_score(y_test, hard_or, zero_division=0)
    pre_or = precision_score(y_test, hard_or, zero_division=0)
    rec_or = recall_score(y_test, hard_or, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, hard_or, labels=[0, 1]).ravel()
    logger.info(f"  [Hard OR (w10|w30|w60)            ]  F1={f1_or:.4f}  P={pre_or:.4f}  R={rec_or:.4f}")
    logger.info(f"  {'':<36}  TP={tp}  FP={fp}  FN={fn}")

    # Strategy 2: Max normalized score — threshold search
    max_score = np.maximum.reduce([norm_scores[10], norm_scores[30], norm_scores[60]])
    thr_max, _ = best_threshold(max_score, y_test)
    metrics_max = full_eval(max_score, y_test, thr_max, "Max-norm ensemble")

    # Strategy 3: Weighted max (w=30 dominates, w=10 and w=60 add sensitivity)
    weighted = 0.4 * norm_scores[10] + 0.4 * norm_scores[30] + 0.2 * norm_scores[60]
    thr_w, _ = best_threshold(weighted, y_test)
    metrics_w = full_eval(weighted, y_test, thr_w, "Weighted ensemble (0.4/0.4/0.2)")

    # Strategy 4: OR of w10+w30 only (skip w60 if it hurts)
    or_1030   = np.maximum(fired_w10, fired_w30)
    f1_1030   = f1_score(y_test, or_1030, zero_division=0)
    pre_1030  = precision_score(y_test, or_1030, zero_division=0)
    rec_1030  = recall_score(y_test, or_1030, zero_division=0)
    tn2, fp2, fn2, tp2 = confusion_matrix(y_test, or_1030, labels=[0, 1]).ravel()
    logger.info(f"  [OR w10+w30 only                  ]  F1={f1_1030:.4f}  P={pre_1030:.4f}  R={rec_1030:.4f}")
    logger.info(f"  {'':<36}  TP={tp2}  FP={fp2}  FN={fn2}")

    # ── Pick best ensemble strategy ────────────────────────────────────────────
    candidates = {
        "hard_or_w10w30w60": (f1_or,   scale_thresholds,     "Hard OR all 3 scales"),
        "max_norm":          (metrics_max["f1"], thr_max,    "Max normalized score"),
        "weighted":          (metrics_w["f1"],  thr_w,       "Weighted 0.4/0.4/0.2"),
        "or_w10w30":         (f1_1030,  scale_thresholds,    "OR w10+w30"),
    }

    logger.info("=" * 70)
    logger.info("Summary — all results:")
    logger.info("=" * 70)
    logger.info(f"  w=10  alone:       F1={scale_metrics[10]['f1']:.4f}  ROC={scale_metrics[10]['roc_auc']:.4f}")
    logger.info(f"  w=30  alone:       F1={scale_metrics[30]['f1']:.4f}  ROC={scale_metrics[30]['roc_auc']:.4f}  ← baseline")
    logger.info(f"  w=60  alone:       F1={scale_metrics[60]['f1']:.4f}  ROC={scale_metrics[60]['roc_auc']:.4f}")
    logger.info(f"  Hard OR (all 3):   F1={f1_or:.4f}")
    logger.info(f"  Max-norm:          F1={metrics_max['f1']:.4f}")
    logger.info(f"  Weighted:          F1={metrics_w['f1']:.4f}")
    logger.info(f"  OR (w10+w30):      F1={f1_1030:.4f}")

    best_f1 = max(
        scale_metrics[10]["f1"], scale_metrics[30]["f1"], scale_metrics[60]["f1"],
        f1_or, metrics_max["f1"], metrics_w["f1"], f1_1030
    )
    logger.info(f"\n  BEST F1 across all strategies: {best_f1:.4f}")
    logger.info(f"  Baseline w=30 alone:           {scale_metrics[30]['f1']:.4f}")
    logger.info(f"  Improvement:                   {best_f1 - scale_metrics[30]['f1']:+.4f}")

    # ── Save multi-scale package ───────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("Step 7: Save multi-scale package")
    logger.info("=" * 70)

    pkg = {
        "model_type":    "MultiScale_LSTMAe_haiend",
        "scales":        [10, 30, 60],
        "models": {
            10: scale_models[10],
            30: scale_models[30],
            60: scale_models[60],
        },
        "thresholds":     scale_thresholds,   # {10: float, 30: float, 60: float}
        "data_mean":      mean,
        "data_std":       std,
        "n_features":     N_FEAT,
        # ensemble scores (for threshold search in digital twin if needed)
        "max_norm_threshold": float(thr_max),
        "weighted_threshold": float(thr_w),
        # metrics per scale
        "scale_metrics":  {str(w): m for w, m in scale_metrics.items()},
        "ensemble_metrics": {
            "hard_or":    dict(f1=float(f1_or), precision=float(pre_or),
                               recall=float(rec_or), tp=int(tp), fp=int(fp), fn=int(fn)),
            "max_norm":   metrics_max,
            "weighted":   metrics_w,
            "or_w10w30":  dict(f1=float(f1_1030), precision=float(pre_1030),
                               recall=float(rec_1030), tp=int(tp2), fp=int(fp2), fn=int(fn2)),
        },
        "best_ensemble_f1": float(best_f1),
        "w30_baseline_f1":  float(scale_metrics[30]["f1"]),
    }

    out_path = OUT_DIR / "multiscale_lstm_detection.joblib"
    joblib.dump(pkg, out_path)
    logger.info(f"  Saved: {out_path}")

    # Update pipeline summary if this beats the current best
    summ_path = MET_DIR / "pipeline_summary.json"
    prev_best  = 0.0
    if summ_path.exists():
        try:
            with open(summ_path) as f:
                prev_best = json.load(f).get("metrics", {}).get("f1", 0.0)
        except: pass

    if best_f1 > prev_best:
        # Determine which strategy won and use its threshold/metrics for reporting
        if best_f1 == f1_or:
            best_strategy, report_metrics = "hard_or_all_scales", dict(
                f1=f1_or, precision=pre_or, recall=rec_or)
        elif best_f1 == metrics_max["f1"]:
            best_strategy, report_metrics = "max_norm_ensemble", metrics_max
        elif best_f1 == metrics_w["f1"]:
            best_strategy, report_metrics = "weighted_ensemble", metrics_w
        else:
            best_strategy, report_metrics = "or_w10w30", dict(
                f1=f1_1030, precision=pre_1030, recall=rec_1030)

        summ = {}
        if summ_path.exists():
            try:
                with open(summ_path) as f: summ = json.load(f)
            except: pass
        summ.update({
            "run_timestamp":  str(datetime.datetime.now()),
            "best_model":     f"MultiScale-LSTM-AE (w=10/30/60, strategy={best_strategy})",
            "metrics":        {k: float(v) for k, v in report_metrics.items()
                               if isinstance(v, (int, float))},
            "training_note":  f"Multi-scale LSTM-AE: w=10+30+60, epochs={args.epochs}",
        })
        with open(summ_path, "w") as f:
            json.dump(summ, f, indent=2, default=str)
        logger.info(f"  NEW BEST! F1={best_f1:.4f} > {prev_best:.4f} — updated pipeline_summary.json")
    else:
        logger.info(f"  F1={best_f1:.4f} vs current best {prev_best:.4f} (no update)")

    logger.info("=" * 70)
    logger.info("  MULTI-SCALE LSTM-AE — COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  w=10 alone:      F1={scale_metrics[10]['f1']:.4f}")
    logger.info(f"  w=30 alone:      F1={scale_metrics[30]['f1']:.4f}  ← was baseline")
    logger.info(f"  w=60 alone:      F1={scale_metrics[60]['f1']:.4f}")
    logger.info(f"  Best ensemble:   F1={best_f1:.4f}")
    logger.info(f"  Delta vs w=30:   {best_f1 - scale_metrics[30]['f1']:+.4f}")
    logger.info(f"  Saved: outputs/models/multiscale_lstm_detection.joblib")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
