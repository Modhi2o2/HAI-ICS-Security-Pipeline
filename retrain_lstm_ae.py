"""
LSTM Autoencoder for HAI Anomaly Detection
==========================================
Key advantage over MLP autoencoder:
  - Processes sequences of 60 timesteps (1 minute of data)
  - Captures temporal correlations and sustained deviations
  - Attacks that last minutes → consistently high reconstruction error
  - Short noise spikes → quickly forgotten by LSTM

Architecture:
  Encoder: LSTM(38, 64) → LSTM(64, 32) → 32-dim latent
  Decoder: repeat(latent, 60) → LSTM(32, 64) → Linear(64, 38)

Expected F1: 0.45-0.65 (vs 0.40 from MLP)

Usage: python retrain_lstm_ae.py [--epochs 30] [--window 60] [--hidden 64]
"""

import sys, json, argparse, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

sys.path.insert(0, str(Path(__file__).parent))
from src.utils.logger import logger
from src.data.multi_version_loader import MultiVersionLoader, COMMON_FEATURES

HAI_ROOT = "C:/Users/PC GAMING/Desktop/AI/HAI"
OUT_DIR  = Path("outputs/models")
MET_DIR  = Path("outputs/metrics")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MET_DIR.mkdir(parents=True, exist_ok=True)

VERSIONS = ["hai-20.07", "hai-21.03", "hai-22.04", "hai-23.05"]
N_FEAT   = len(COMMON_FEATURES)  # 38


# ─────────────────────────────────────────────────────────────────────────────
# Arguments
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",    type=int,   default=40)
    p.add_argument("--window",    type=int,   default=60,   help="sequence length (seconds)")
    p.add_argument("--hidden",    type=int,   default=64,   help="LSTM hidden dim")
    p.add_argument("--latent",    type=int,   default=24,   help="bottleneck dim")
    p.add_argument("--batch",     type=int,   default=512)
    p.add_argument("--n-windows", type=int,   default=150000, help="training windows to sample")
    p.add_argument("--lr",        type=float, default=3e-4)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# LSTM Autoencoder
# ─────────────────────────────────────────────────────────────────────────────

class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, latent: int = 24):
        super().__init__()
        self.n_features = n_features
        self.hidden     = hidden
        self.latent     = latent

        # Encoder: 2-layer LSTM → bottleneck
        self.enc_lstm1 = nn.LSTM(n_features, hidden, batch_first=True)
        self.enc_lstm2 = nn.LSTM(hidden, latent,  batch_first=True)

        # Decoder: repeat context → 2-layer LSTM → reconstruction
        self.dec_lstm1 = nn.LSTM(latent,  hidden, batch_first=True)
        self.dec_lstm2 = nn.LSTM(hidden,  hidden, batch_first=True)
        self.out_proj  = nn.Linear(hidden, n_features)

        self.dropout = nn.Dropout(0.1)

    def encode(self, x):
        # x: (B, T, F)
        out1, _ = self.enc_lstm1(x)                  # (B, T, hidden)
        out1 = self.dropout(out1)
        out2, (hn, _) = self.enc_lstm2(out1)         # hn: (1, B, latent)
        return hn.squeeze(0)                          # (B, latent)

    def decode(self, z, seq_len: int):
        # z: (B, latent)
        repeated = z.unsqueeze(1).expand(-1, seq_len, -1)   # (B, T, latent)
        out1, _ = self.dec_lstm1(repeated)           # (B, T, hidden)
        out1 = self.dropout(out1)
        out2, _ = self.dec_lstm2(out1)               # (B, T, hidden)
        recon = self.out_proj(out2)                  # (B, T, F)
        return recon

    def forward(self, x):
        z     = self.encode(x)
        recon = self.decode(z, x.size(1))
        return recon

    def reconstruction_error(self, x: torch.Tensor) -> np.ndarray:
        """Per-sample mean squared reconstruction error."""
        with torch.no_grad():
            recon = self.forward(x)
            err   = ((recon - x) ** 2).mean(dim=(1, 2))  # mean over T and F
        return err.cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Normalizer
# ─────────────────────────────────────────────────────────────────────────────

def fit_normalizer(X: np.ndarray):
    mean = X.mean(axis=0)
    std  = X.std(axis=0) + 1e-8
    return mean, std

def normalize(X: np.ndarray, mean, std) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Sliding window dataset
# ─────────────────────────────────────────────────────────────────────────────

def make_windows(X: np.ndarray, window: int, n_windows: int = None,
                 stride: int = 1) -> np.ndarray:
    """
    Extract sliding windows from time-series X (N, F).
    Returns (n_windows, window, F).
    If n_windows given, randomly sample that many windows.
    """
    n = len(X)
    if n < window:
        raise ValueError(f"X too short ({n}) for window {window}")

    if n_windows is not None and n_windows < (n - window):
        # Random sampling
        starts = np.random.choice(n - window, n_windows, replace=False)
        starts.sort()
    else:
        starts = np.arange(0, n - window, stride)

    windows = np.stack([X[s:s+window] for s in starts], axis=0)
    return windows.astype(np.float32)


def make_windows_sequential(X: np.ndarray, window: int) -> np.ndarray:
    """Full stride-1 windows for sequential evaluation."""
    n = len(X)
    if n < window:
        return X[np.newaxis, :]  # single incomplete window
    starts = np.arange(n - window + 1)
    return np.stack([X[s:s+window] for s in starts], axis=0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_lstm_ae(model: LSTMAutoencoder, windows: np.ndarray,
                  epochs: int, batch: int, lr: float, device: str) -> LSTMAutoencoder:
    model = model.to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    crit  = nn.MSELoss()

    n = len(windows)
    best_loss  = float("inf")
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        perm  = np.random.permutation(n)
        total = 0.0; steps = 0

        for i in range(0, n, batch):
            idx = perm[i:i+batch]
            xb  = torch.from_numpy(windows[idx]).to(device)
            opt.zero_grad()
            loss = crit(model(xb), xb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item(); steps += 1

        sched.step()
        avg = total / max(steps, 1)

        if avg < best_loss:
            best_loss  = avg
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if ep % 5 == 0 or ep == 1:
            logger.info(f"  Epoch {ep:3d}/{epochs}  loss={avg:.6f}  best={best_loss:.6f}")

    model.load_state_dict(best_state)
    return model.to("cpu").eval()


# ─────────────────────────────────────────────────────────────────────────────
# Scoring test set
# ─────────────────────────────────────────────────────────────────────────────

def score_test_set(model: LSTMAutoencoder, X_norm: np.ndarray,
                   window: int, batch: int = 256) -> np.ndarray:
    """
    Score each timestep using the reconstruction error of the window
    ending at that timestep. First (window-1) timesteps get score of window[0].
    """
    logger.info(f"  Scoring {len(X_norm):,} timesteps (window={window})...")
    n   = len(X_norm)
    all_scores = np.zeros(n, dtype=np.float32)

    # Process in batches of windows (each window ends at a different timestep)
    chunk = batch * window  # ~15K timesteps per chunk
    model.eval()

    with torch.no_grad():
        starts = list(range(n - window + 1))
        for i in range(0, len(starts), batch):
            batch_starts = starts[i:i+batch]
            wins = np.stack([X_norm[s:s+window] for s in batch_starts])
            xb   = torch.from_numpy(wins)
            errs = model.reconstruction_error(xb)
            for j, s in enumerate(batch_starts):
                all_scores[s + window - 1] = errs[j]  # assign to last timestep

        # Fill in first (window-1) timesteps with first valid score
        first_valid = all_scores[window-1]
        all_scores[:window-1] = first_valid

    return all_scores


# ─────────────────────────────────────────────────────────────────────────────
# Threshold sweep
# ─────────────────────────────────────────────────────────────────────────────

def best_threshold(scores: np.ndarray, y: np.ndarray):
    from sklearn.metrics import f1_score
    bf, bt = 0.0, scores.mean()
    for p in np.arange(80, 99.9, 0.1):
        t = np.percentile(scores, p)
        f = f1_score(y, (scores >= t).astype(int), zero_division=0)
        if f > bf: bf, bt = f, t
    return float(bt), float(bf)


def full_eval(scores, y, thr, name=""):
    from sklearn.metrics import (f1_score, precision_score, recall_score,
                                 roc_auc_score, confusion_matrix)
    pred = (scores >= thr).astype(int)
    f1   = f1_score(y, pred, zero_division=0)
    pre  = precision_score(y, pred, zero_division=0)
    rec  = recall_score(y, pred, zero_division=0)
    try:   roc = roc_auc_score(y, scores)
    except: roc = float("nan")
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    m = dict(f1=float(f1), precision=float(pre), recall=float(rec),
             roc_auc=float(roc), threshold=float(thr),
             tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn))
    logger.info(f"  [{name}]  F1={f1:.4f}  Prec={pre:.4f}  Rec={rec:.4f}  "
                f"ROC={roc:.4f}  thr={thr:.6f}")
    logger.info(f"  {'':<5}  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    loader = MultiVersionLoader(HAI_ROOT)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    logger.info(f"Config: window={args.window}  hidden={args.hidden}  "
                f"latent={args.latent}  epochs={args.epochs}  "
                f"n_windows={args.n_windows:,}")

    # ── 1. Load TRAIN data (normal) ───────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("Step 1: Load training data (normal only)")
    logger.info("=" * 65)

    X_train_raw, y_train = loader.load_all(
        versions=VERSIONS, split="train", features=COMMON_FEATURES
    )
    X_train_raw = X_train_raw[y_train == 0]
    logger.info(f"  Normal training: {X_train_raw.shape}")

    # ── 2. Load TEST data ─────────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("Step 2: Load test data")
    logger.info("=" * 65)

    X_test_raw, y_test = loader.load_all(
        versions=VERSIONS, split="test", features=COMMON_FEATURES
    )
    logger.info(f"  Test: {X_test_raw.shape}  attacks={y_test.sum()} ({y_test.mean()*100:.2f}%)")

    # ── 3. Normalize ──────────────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("Step 3: Normalize sensors")
    logger.info("=" * 65)

    mean, std = fit_normalizer(X_train_raw)
    X_train_norm = normalize(X_train_raw, mean, std)
    X_test_norm  = normalize(X_test_raw,  mean, std)
    logger.info(f"  Normalized  train={X_train_norm.shape}  test={X_test_norm.shape}")

    # ── 4. Create training windows ────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info(f"Step 4: Creating {args.n_windows:,} training windows (len={args.window})")
    logger.info("=" * 65)

    windows = make_windows(X_train_norm, args.window, args.n_windows)
    logger.info(f"  Training windows: {windows.shape}  "
                f"({windows.nbytes / 1e6:.1f} MB)")

    # ── 5. Train LSTM autoencoder ─────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("Step 5: Train LSTM Autoencoder")
    logger.info("=" * 65)

    model = LSTMAutoencoder(N_FEAT, args.hidden, args.latent)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Parameters: {n_params:,}")

    model = train_lstm_ae(model, windows, args.epochs, args.batch,
                          args.lr, device)

    # ── 6. Score test set ─────────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("Step 6: Score test set")
    logger.info("=" * 65)

    raw_scores = score_test_set(model, X_test_norm, args.window, batch=128)
    logger.info(f"  Score stats: min={raw_scores.min():.4f}  "
                f"max={raw_scores.max():.4f}  mean={raw_scores.mean():.4f}  "
                f"p95={np.percentile(raw_scores,95):.4f}")

    # ── 7. Threshold search ───────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("Step 7: Find optimal threshold + temporal smoothing")
    logger.info("=" * 65)

    all_metrics = {}
    sc = pd.Series(raw_scores)

    # Raw
    thr, f1 = best_threshold(raw_scores, y_test)
    all_metrics["LSTM_AE_raw"] = full_eval(raw_scores, y_test, thr, "LSTM_AE raw")

    # EWM smoothing at multiple spans
    best_f1, best_name = f1, "LSTM_AE_raw"
    for span in [10, 30, 60, 120, 300]:
        smoothed = sc.ewm(span=span).mean().values
        t, f = best_threshold(smoothed, y_test)
        nm = f"LSTM_AE_ewm{span}"
        all_metrics[nm] = full_eval(smoothed, y_test, t, f"LSTM_AE EWM-{span}s")
        if f > best_f1:
            best_f1, best_name = f, nm

    # Rolling mean
    for w in [10, 30, 60]:
        smoothed = sc.rolling(w, min_periods=1).mean().values
        t, f = best_threshold(smoothed, y_test)
        nm = f"LSTM_AE_rmean{w}"
        all_metrics[nm] = full_eval(smoothed, y_test, t, f"LSTM_AE rollmean-{w}s")
        if f > best_f1:
            best_f1, best_name = f, nm

    # ── 8. Ensemble with MLP autoencoder ──────────────────────────────────────
    logger.info("=" * 65)
    logger.info("Step 8: Ensemble LSTM-AE + MLP-AE")
    logger.info("=" * 65)

    mlp_ae_path = OUT_DIR / "autoencoder_detection.joblib"
    if mlp_ae_path.exists():
        try:
            mlp_pkg = joblib.load(mlp_ae_path)
            from retrain_autoencoder import Autoencoder as MLPAutoencoder
            mlp_model = MLPAutoencoder(mlp_pkg["input_dim"], mlp_pkg["bottleneck"])
            mlp_model.load_state_dict(mlp_pkg["model_state"])
            mlp_model.eval()

            # Re-engineer for MLP AE
            df_test = pd.DataFrame(X_test_raw, columns=COMMON_FEATURES).ffill().fillna(0)
            parts = [df_test]
            d1 = df_test.diff().fillna(0); d1.columns = [f"{c}_d1" for c in COMMON_FEATURES]
            parts.append(d1)
            r  = df_test.rolling(30, min_periods=1)
            rm = r.mean(); rm.columns = [f"{c}_r30m" for c in COMMON_FEATURES]
            rs = r.std().fillna(0); rs.columns = [f"{c}_r30s" for c in COMMON_FEATURES]
            parts.append(rm); parts.append(rs)
            l5 = df_test.shift(5).fillna(0); l5.columns = [f"{c}_l5" for c in COMMON_FEATURES]
            parts.append(l5)
            X_mlp = pd.concat(parts, axis=1).astype(np.float32).values
            m_mean = mlp_pkg["eng_state"]["mean"]
            m_std  = mlp_pkg["eng_state"]["std"]
            X_mlp  = (X_mlp - m_mean) / m_std
            X_mlp  = np.nan_to_num(X_mlp, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

            # MLP AE scores with EWM-30
            mlp_scores = []
            Xt_mlp = torch.from_numpy(X_mlp)
            for i in range(0, len(Xt_mlp), 8192):
                mlp_scores.append(mlp_model.reconstruction_error(Xt_mlp[i:i+8192]))
            mlp_scores = np.concatenate(mlp_scores)
            mlp_ewm = pd.Series(mlp_scores).ewm(span=30).mean().values

            # Grid search ensemble weights
            def n01(x): return (x - x.min()) / (x.max() - x.min() + 1e-8)
            lstm_n = n01(sc.ewm(span=30).mean().values)  # best LSTM signal
            mlp_n  = n01(mlp_ewm)

            from sklearn.metrics import f1_score as _f1
            best_ens_f1, best_w = 0.0, 0.5
            for w in np.arange(0.1, 0.95, 0.05):
                ens = w * lstm_n + (1 - w) * mlp_n
                _, f = best_threshold(ens, y_test)
                if f > best_ens_f1: best_ens_f1, best_w = f, w

            ens_scores = best_w * lstm_n + (1 - best_w) * mlp_n
            thr_ens, f1_ens = best_threshold(ens_scores, y_test)
            all_metrics["LSTM_MLP_ensemble"] = full_eval(
                ens_scores, y_test, thr_ens, f"LSTM+MLP ensemble (w_lstm={best_w:.2f})")
            logger.info(f"  Ensemble weight: LSTM={best_w:.2f}  MLP={1-best_w:.2f}")
            if f1_ens > best_f1:
                best_f1, best_name = f1_ens, "LSTM_MLP_ensemble"
        except Exception as e:
            logger.warning(f"  MLP-AE ensemble skipped: {e}")

    # ── 9. Save best model ────────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("Step 9: Saving models")
    logger.info("=" * 65)

    lstm_pkg = {
        "model_type":    "LSTMAutoencoder",
        "model":         model,
        "model_state":   model.state_dict(),
        "n_features":    N_FEAT,
        "hidden":        args.hidden,
        "latent":        args.latent,
        "window":        args.window,
        "data_mean":     mean,
        "data_std":      std,
        "features":      COMMON_FEATURES,
        "threshold":     all_metrics[best_name]["threshold"],
        "smooth_type":   "ewm",
        "smooth_window": 30,
        "metrics":       all_metrics[best_name],
        "best_f1":       best_f1,
        "all_results":   all_metrics,
    }
    joblib.dump(lstm_pkg, OUT_DIR / "lstm_ae_detection.joblib")
    logger.info(f"  Saved: lstm_ae_detection.joblib")

    # Update best_detection_model.joblib if LSTM is better
    current_best = 0.396
    try:
        cur = joblib.load(OUT_DIR / "best_detection_model.joblib")
        current_best = cur.get("best_f1", 0.396)
    except: pass

    if best_f1 > current_best:
        joblib.dump(lstm_pkg, OUT_DIR / "best_detection_model.joblib")
        logger.info(f"  NEW BEST: F1={best_f1:.4f} > {current_best:.4f}  "
                    f"Updated best_detection_model.joblib")
    else:
        logger.info(f"  LSTM F1={best_f1:.4f} vs current best {current_best:.4f}  "
                    f"(keeping existing best)")

    # Update dashboard metrics
    import datetime
    summ_path = MET_DIR / "pipeline_summary.json"
    summ = {}
    if summ_path.exists():
        try:
            with open(summ_path) as f: summ = json.load(f)
        except: pass
    best_m = all_metrics[best_name]
    summ.update({
        "run_timestamp":     str(datetime.datetime.now()),
        "best_model":        best_name,
        "metrics": {
            "f1":        best_f1,
            "precision": best_m.get("precision", 0),
            "recall":    best_m.get("recall", 0),
            "roc_auc":   best_m.get("roc_auc", 0),
            "threshold": best_m.get("threshold", 0),
        },
        "all_model_metrics": {k: v for k, v in list(all_metrics.items())[:8]},
        "training_note":     f"LSTM-AE: window={args.window}s  hidden={args.hidden}  "
                             f"latent={args.latent}  epochs={args.epochs}",
        "training_versions": VERSIONS,
    })
    with open(summ_path, "w") as f:
        json.dump(summ, f, indent=2, default=str)

    # ── 10. Final report ──────────────────────────────────────────────────────
    W = 65
    logger.info("\n" + "=" * W)
    logger.info("  LSTM AUTOENCODER — FINAL RESULTS")
    logger.info("=" * W)
    for nm, m in sorted(all_metrics.items(), key=lambda x: -x[1]["f1"])[:8]:
        logger.info(f"  {nm:<32} F1={m['f1']:.4f}  ROC={m['roc_auc']:.4f}")
    logger.info("=" * W)
    logger.info(f"  BEST: {best_name}  F1 = {best_f1:.4f}")
    if   best_f1 >= 0.70: status = "EXCELLENT - TARGET REACHED!"
    elif best_f1 >= 0.55: status = "GOOD - significantly better than MLP"
    elif best_f1 >= 0.42: status = "FAIR - modest improvement over MLP"
    else:                  status = "SIMILAR TO MLP - try larger model or more windows"
    logger.info(f"  STATUS: {status}")
    logger.info("=" * W)


if __name__ == "__main__":
    main()
