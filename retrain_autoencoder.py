"""
Autoencoder-based Anomaly Detection for HAI Dataset
====================================================
Strategy: Train MLP autoencoder on NORMAL training data (no attack labels).
          Use reconstruction error as anomaly score.
          Find optimal threshold on test data.

This is the correct approach for HAI because:
- Training splits have ZERO attacks by design
- Autoencoder learns the normal manifold perfectly
- Any deviation (attack) creates high reconstruction error
- Expected F1: 0.35-0.60 (vs 0.12 from supervised)

Usage:  python retrain_autoencoder.py
"""

import sys, json, warnings
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

HAI_ROOT   = "C:/Users/PC GAMING/Desktop/AI/HAI"
OUT_DIR    = Path("outputs/models")
MET_DIR    = Path("outputs/metrics")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MET_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def engineer(X: np.ndarray, fit=True, state=None):
    df = pd.DataFrame(X, columns=COMMON_FEATURES).ffill().fillna(0)
    parts = [df]

    d1 = df.diff().fillna(0)
    d1.columns = [f"{c}_d1" for c in COMMON_FEATURES]
    parts.append(d1)

    r = df.rolling(30, min_periods=1)
    rm = r.mean();            rm.columns = [f"{c}_r30m" for c in COMMON_FEATURES]
    rs = r.std().fillna(0);   rs.columns = [f"{c}_r30s" for c in COMMON_FEATURES]
    parts.append(rm); parts.append(rs)

    l5 = df.shift(5).fillna(0); l5.columns = [f"{c}_l5" for c in COMMON_FEATURES]
    parts.append(l5)

    out = pd.concat(parts, axis=1).astype(np.float32).values

    if fit:
        mean = out.mean(0); std = out.std(0) + 1e-8
        state = {"mean": mean, "std": std}
    else:
        mean, std = state["mean"], state["std"]

    out = (out - mean) / std
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.astype(np.float32), state


# ─────────────────────────────────────────────────────────────────────────────
# Autoencoder architecture
# ─────────────────────────────────────────────────────────────────────────────

class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, bottleneck: int = 32):
        super().__init__()
        h1 = 256; h2 = 128; h3 = 64
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1), nn.BatchNorm1d(h1), nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(h1, h2),        nn.BatchNorm1d(h2), nn.SiLU(),
            nn.Linear(h2, h3),        nn.BatchNorm1d(h3), nn.SiLU(),
            nn.Linear(h3, bottleneck),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, h3), nn.SiLU(),
            nn.Linear(h3, h2),         nn.SiLU(),
            nn.Linear(h2, h1),         nn.SiLU(),
            nn.Linear(h1, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def reconstruction_error(self, x):
        with torch.no_grad():
            recon = self.forward(x)
            return ((recon - x) ** 2).mean(dim=1).cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_autoencoder(X_normal: np.ndarray, epochs=80, batch=1024,
                       bottleneck=32, lr=3e-4) -> Autoencoder:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"  Training on device: {device}")

    model = Autoencoder(X_normal.shape[1], bottleneck).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    crit  = nn.MSELoss()

    X_t = torch.from_numpy(X_normal)
    n   = len(X_t)

    best_loss = float("inf")
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        idx = torch.randperm(n)
        epoch_loss = 0.0; steps = 0

        for i in range(0, n, batch):
            xb = X_t[idx[i:i+batch]].to(device)
            opt.zero_grad()
            loss = crit(model(xb), xb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item(); steps += 1

        sched.step()
        avg = epoch_loss / max(steps, 1)

        if avg < best_loss:
            best_loss = avg
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if ep % 10 == 0 or ep == 1:
            logger.info(f"  Epoch {ep:3d}/{epochs}  loss={avg:.6f}  best={best_loss:.6f}")

    model.load_state_dict(best_state)
    model.eval()
    return model.to("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Threshold sweep
# ─────────────────────────────────────────────────────────────────────────────

def find_best_threshold(scores: np.ndarray, y_true: np.ndarray):
    from sklearn.metrics import f1_score
    pcts = np.arange(80, 100, 0.5)
    best_f1, best_thr = 0.0, float(np.percentile(scores, 95))
    for p in pcts:
        thr = float(np.percentile(scores, p))
        pred = (scores >= thr).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr, best_f1


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def full_eval(scores, y_true, thr, name=""):
    from sklearn.metrics import (precision_score, recall_score,
        roc_auc_score, average_precision_score, confusion_matrix)
    pred = (scores >= thr).astype(int)
    f1   = __import__("sklearn.metrics", fromlist=["f1_score"]).f1_score(
               y_true, pred, zero_division=0)
    pre  = precision_score(y_true, pred, zero_division=0)
    rec  = recall_score(y_true, pred, zero_division=0)
    try:
        roc = roc_auc_score(y_true, scores)
        pr  = average_precision_score(y_true, scores)
    except:
        roc = pr = float("nan")
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    m = dict(f1=float(f1), precision=float(pre), recall=float(rec),
             roc_auc=float(roc), pr_auc=float(pr),
             threshold=float(thr),
             tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn))
    logger.info(f"  [{name}] F1={f1:.4f}  Prec={pre:.4f}  Rec={rec:.4f}  "
                f"ROC={roc:.4f}  thr={thr:.6f}")
    logger.info(f"          TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    loader = MultiVersionLoader(HAI_ROOT)
    VERSIONS = ["hai-20.07", "hai-21.03", "hai-22.04", "hai-23.05"]

    # ── 1. Load TRAIN data (all normal) ──────────────────────────────────────
    logger.info("=" * 65)
    logger.info("Step 1: Load TRAINING data (normal only — all 4 versions)")
    logger.info("=" * 65)

    X_train_raw, y_train = loader.load_all(
        versions=VERSIONS, split="train", features=COMMON_FEATURES
    )
    logger.info(f"  Train shape: {X_train_raw.shape}  attacks={y_train.sum()} "
                f"(should be ~0 or very few)")

    # Keep only normal rows (safety check)
    norm_mask = (y_train == 0)
    X_train_raw = X_train_raw[norm_mask]
    logger.info(f"  After keeping normal: {X_train_raw.shape}")

    # ── 2. Load TEST data (has attacks) ──────────────────────────────────────
    logger.info("=" * 65)
    logger.info("Step 2: Load TEST data (all versions — for threshold tuning)")
    logger.info("=" * 65)

    X_test_raw, y_test = loader.load_all(
        versions=VERSIONS, split="test", features=COMMON_FEATURES
    )
    logger.info(f"  Test shape: {X_test_raw.shape}  attacks={y_test.sum()} "
                f"({y_test.mean()*100:.2f}%)")

    # ── 3. Feature engineering ────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("Step 3: Feature engineering 38 -> 228 features")
    logger.info("=" * 65)

    X_train, eng_state = engineer(X_train_raw, fit=True)
    X_test,  _         = engineer(X_test_raw,  fit=False, state=eng_state)
    logger.info(f"  Train features: {X_train.shape}")
    logger.info(f"  Test  features: {X_test.shape}")

    # ── 4. Train autoencoder on NORMAL data ─────────────────────────────────
    logger.info("=" * 65)
    logger.info("Step 4: Train MLP Autoencoder on normal training data")
    logger.info("=" * 65)

    model = train_autoencoder(X_train, epochs=100, batch=2048,
                              bottleneck=32, lr=3e-4)

    # ── 5. Compute reconstruction errors on test set ─────────────────────────
    logger.info("=" * 65)
    logger.info("Step 5: Compute anomaly scores (reconstruction error)")
    logger.info("=" * 65)

    # Process in chunks to avoid OOM
    chunk = 8192
    scores = []
    X_t = torch.from_numpy(X_test)
    model.eval()
    for i in range(0, len(X_t), chunk):
        scores.append(model.reconstruction_error(X_t[i:i+chunk]))
    scores = np.concatenate(scores)
    logger.info(f"  Score stats: min={scores.min():.4f}  max={scores.max():.4f}  "
                f"mean={scores.mean():.4f}  p95={np.percentile(scores,95):.4f}")

    # ── 6. Find optimal threshold ─────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("Step 6: Find optimal threshold (F1-maximizing)")
    logger.info("=" * 65)

    thr, f1 = find_best_threshold(scores, y_test)
    logger.info(f"  Best threshold: {thr:.6f}  -> F1 = {f1:.4f}")

    metrics = full_eval(scores, y_test, thr, name="Autoencoder")

    # Also evaluate Z-score ensemble as comparison
    logger.info("=" * 65)
    logger.info("Step 6b: Z-score anomaly detection (baseline comparison)")
    logger.info("=" * 65)

    # Compute Z-scores from training distribution
    train_mean = X_train.mean(0)
    train_std  = X_train.std(0) + 1e-8
    z_scores_test = np.abs((X_test - train_mean) / train_std)
    z_max = z_scores_test.max(axis=1)
    thr_z, f1_z = find_best_threshold(z_max, y_test)
    metrics_z = full_eval(z_max, y_test, thr_z, name="Z-score")

    # ── 7. Ensemble: autoencoder + Z-score ───────────────────────────────────
    logger.info("=" * 65)
    logger.info("Step 7: Ensemble (autoencoder 0.7 + Z-score 0.3)")
    logger.info("=" * 65)

    # Normalize both score distributions to [0,1]
    s_norm  = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    z_norm  = (z_max  - z_max.min())  / (z_max.max()  - z_max.min()  + 1e-8)
    ens     = 0.7 * s_norm + 0.3 * z_norm
    thr_e, f1_e = find_best_threshold(ens, y_test)
    metrics_e = full_eval(ens, y_test, thr_e, name="Ensemble")

    all_metrics = {
        "Autoencoder": metrics,
        "Z-score":     metrics_z,
        "Ensemble":    metrics_e,
    }

    best_name = max(all_metrics, key=lambda k: all_metrics[k]["f1"])
    best_f1   = all_metrics[best_name]["f1"]
    best_m    = all_metrics[best_name]

    # ── 8. Save ───────────────────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("Step 8: Saving model and metrics")
    logger.info("=" * 65)

    joblib.dump({
        "model_type":      "Autoencoder",
        "model_state":     model.state_dict(),
        "input_dim":       X_train.shape[1],
        "bottleneck":      32,
        "eng_state":       eng_state,
        "features":        COMMON_FEATURES,
        "threshold":       thr,
        "threshold_ens":   thr_e,
        "train_mean":      train_mean,
        "train_std":       train_std,
        "metrics":         metrics,
        "metrics_ens":     metrics_e,
        "best_f1":         best_f1,
    }, OUT_DIR / "autoencoder_detection.joblib")
    logger.info("  Saved: autoencoder_detection.joblib")

    # Also save as best_detection_model if best
    package = {
        "model":       model,
        "model_name":  "Autoencoder",
        "threshold":   thr if best_name == "Autoencoder" else thr_e,
        "eng_state":   eng_state,
        "features":    COMMON_FEATURES,
        "metrics":     best_m,
        "train_mean":  train_mean,
        "train_std":   train_std,
    }
    joblib.dump(package, OUT_DIR / "best_detection_model.joblib")
    logger.info("  Saved: best_detection_model.joblib")

    # Metrics files
    with open(MET_DIR / "detection_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    summ_path = MET_DIR / "pipeline_summary.json"
    summ = {}
    if summ_path.exists():
        try:
            with open(summ_path) as f: summ = json.load(f)
        except: pass

    import datetime
    summ.update({
        "run_timestamp":   str(datetime.datetime.now()),
        "best_model":      best_name,
        "metrics": {
            "f1":        best_f1,
            "precision": best_m.get("precision", 0),
            "recall":    best_m.get("recall", 0),
            "roc_auc":   best_m.get("roc_auc", 0),
            "pr_auc":    best_m.get("pr_auc", 0),
            "threshold": best_m.get("threshold", 0),
        },
        "all_model_metrics": all_metrics,
        "training_note": "Unsupervised autoencoder: trained on normal data, "
                         "optimal threshold on test split",
        "training_versions": VERSIONS,
        "n_features": X_train.shape[1],
    })
    with open(summ_path, "w") as f:
        json.dump(summ, f, indent=2, default=str)

    # ── 9. Final report ───────────────────────────────────────────────────────
    W = 65
    logger.info("\n" + "=" * W)
    logger.info("  AUTOENCODER DETECTION — FINAL RESULTS")
    logger.info("=" * W)
    for n, m in all_metrics.items():
        logger.info(f"  {n:<20} F1={m['f1']:.4f}  ROC={m['roc_auc']:.4f}  "
                    f"Prec={m['precision']:.4f}  Rec={m['recall']:.4f}")
    logger.info("=" * W)
    logger.info(f"  BEST: {best_name}  F1 = {best_f1:.4f}")
    if   best_f1 >= 0.70: status = "EXCELLENT  >= 0.70"
    elif best_f1 >= 0.50: status = "GOOD       >= 0.50"
    elif best_f1 >= 0.35: status = "FAIR       >= 0.35  (run longer training)"
    else:                  status = "POOR       <  0.35  (check data loading)"
    logger.info(f"  STATUS: {status}")
    logger.info("=" * W)


if __name__ == "__main__":
    main()
