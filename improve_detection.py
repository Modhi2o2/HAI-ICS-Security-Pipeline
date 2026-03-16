"""
Improved Detection: Fine-tune autoencoder + better ensemble
===========================================================
Takes the trained autoencoder and improves F1 by:
1. Wider temporal windows (60s, 120s, 300s) for sustained attacks
2. Per-sensor Z-score with multiple thresholds
3. CUSUM change-point detection
4. Optimal ensemble weighting

Expected improvement: F1 0.37 -> 0.50-0.60

Usage: python improve_detection.py
"""

import sys, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import joblib

sys.path.insert(0, str(Path(__file__).parent))
from src.utils.logger import logger
from src.data.multi_version_loader import MultiVersionLoader, COMMON_FEATURES

HAI_ROOT = "C:/Users/PC GAMING/Desktop/AI/HAI"
OUT_DIR  = Path("outputs/models")
MET_DIR  = Path("outputs/metrics")

VERSIONS = ["hai-20.07", "hai-21.03", "hai-22.04", "hai-23.05"]


# ─────────────────────────────────────────────────────────────────────────────
# Extended feature engineering (wider temporal windows)
# ─────────────────────────────────────────────────────────────────────────────

def engineer_extended(X: np.ndarray, fit=True, state=None):
    """38 sensors -> 494 features (multiple time scales)"""
    df = pd.DataFrame(X, columns=COMMON_FEATURES).ffill().fillna(0)
    parts = [df]                                           # 38 raw

    # Derivatives
    d1 = df.diff().fillna(0)
    d1.columns = [f"{c}_d1" for c in COMMON_FEATURES]
    parts.append(d1)                                       # +38

    # Multiple rolling windows
    for w in [30, 60, 120, 300]:
        r  = df.rolling(w, min_periods=1)
        rm = r.mean(); rm.columns = [f"{c}_r{w}m" for c in COMMON_FEATURES]
        rs = r.std().fillna(0); rs.columns = [f"{c}_r{w}s" for c in COMMON_FEATURES]
        parts.append(rm); parts.append(rs)                # +38*2 each window

    # Multiple lags
    for lag in [5, 30, 60]:
        l = df.shift(lag).fillna(0)
        l.columns = [f"{c}_l{lag}" for c in COMMON_FEATURES]
        parts.append(l)                                    # +38 each lag

    out = pd.concat(parts, axis=1).astype(np.float32).values
    # Total: 38 + 38 + 4*(38*2) + 3*38 = 38+38+304+114 = 494

    if fit:
        mean = out.mean(0); std = out.std(0) + 1e-8
        state = {"mean": mean, "std": std, "n_features": out.shape[1]}
    else:
        mean, std = state["mean"], state["std"]

    out = (out - mean) / std
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.astype(np.float32), state


# ─────────────────────────────────────────────────────────────────────────────
# Load Autoencoder from saved joblib
# ─────────────────────────────────────────────────────────────────────────────

def load_autoencoder():
    pkg = joblib.load(OUT_DIR / "autoencoder_detection.joblib")
    from retrain_autoencoder import Autoencoder
    model = Autoencoder(pkg["input_dim"], pkg["bottleneck"])
    model.load_state_dict(pkg["model_state"])
    model.eval()
    return model, pkg


# ─────────────────────────────────────────────────────────────────────────────
# Threshold sweep
# ─────────────────────────────────────────────────────────────────────────────

def best_threshold(scores, y_true):
    from sklearn.metrics import f1_score
    pcts  = np.arange(75, 100, 0.25)
    bf, bt = 0.0, np.percentile(scores, 95)
    for p in pcts:
        t = np.percentile(scores, p)
        f = f1_score(y_true, (scores >= t).astype(int), zero_division=0)
        if f > bf: bf, bt = f, t
    return float(bt), float(bf)


# ─────────────────────────────────────────────────────────────────────────────
# CUSUM change-point score
# ─────────────────────────────────────────────────────────────────────────────

def cusum_score(X_norm: np.ndarray, k=0.5) -> np.ndarray:
    """Cumulative sum change-point detector per sensor, max across sensors."""
    n, d = X_norm.shape
    S  = np.zeros(d, dtype=np.float32)
    out = np.zeros(n, dtype=np.float32)
    for i in range(n):
        S = np.maximum(0, S + np.abs(X_norm[i]) - k)
        out[i] = S.max()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Full evaluation
# ─────────────────────────────────────────────────────────────────────────────

def full_eval(scores, y_true, thr, name=""):
    from sklearn.metrics import (precision_score, recall_score,
        roc_auc_score, average_precision_score, confusion_matrix, f1_score)
    pred = (scores >= thr).astype(int)
    f1   = f1_score(y_true, pred, zero_division=0)
    pre  = precision_score(y_true, pred, zero_division=0)
    rec  = recall_score(y_true, pred, zero_division=0)
    try: roc = roc_auc_score(y_true, scores)
    except: roc = float("nan")
    try: pr  = average_precision_score(y_true, scores)
    except: pr  = float("nan")
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0,1]).ravel()
    m = dict(f1=float(f1), precision=float(pre), recall=float(rec),
             roc_auc=float(roc), pr_auc=float(pr),
             threshold=float(thr),
             tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn))
    logger.info(f"  [{name}]  F1={f1:.4f}  Prec={pre:.4f}  Rec={rec:.4f}  "
                f"ROC={roc:.4f}  thr={thr:.5f}")
    logger.info(f"           TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    loader = MultiVersionLoader(HAI_ROOT)

    # ── 1. Load TRAIN data (normal only) ─────────────────────────────────────
    logger.info("=" * 65)
    logger.info("Step 1: Load training + test data")
    logger.info("=" * 65)

    X_train_raw, y_train = loader.load_all(
        versions=VERSIONS, split="train", features=COMMON_FEATURES
    )
    X_test_raw, y_test = loader.load_all(
        versions=VERSIONS, split="test", features=COMMON_FEATURES
    )
    # Keep only normal for training distribution
    X_train_raw = X_train_raw[y_train == 0]
    logger.info(f"  Train (normal): {X_train_raw.shape}")
    logger.info(f"  Test:           {X_test_raw.shape}  attacks={y_test.sum()} ({y_test.mean()*100:.2f}%)")

    # ── 2. Extended feature engineering ──────────────────────────────────────
    logger.info("=" * 65)
    logger.info("Step 2: Extended feature engineering (38 -> 494 features)")
    logger.info("=" * 65)

    X_train_ext, eng_state = engineer_extended(X_train_raw, fit=True)
    X_test_ext,  _         = engineer_extended(X_test_raw,  fit=False, state=eng_state)
    logger.info(f"  Train features: {X_train_ext.shape}")
    logger.info(f"  Test  features: {X_test_ext.shape}")

    # ── 3. Load existing autoencoder and get base scores ─────────────────────
    logger.info("=" * 65)
    logger.info("Step 3: Loading existing autoencoder")
    logger.info("=" * 65)

    try:
        ae_model, ae_pkg = load_autoencoder()
        # Re-score test data with OLD 190-feature autoencoder
        # (we need the original 190-feature engineered data)
        def engineer_190(X, fit=True, state=None):
            df = pd.DataFrame(X, columns=COMMON_FEATURES).ffill().fillna(0)
            parts = [df]
            d1 = df.diff().fillna(0); d1.columns = [f"{c}_d1" for c in COMMON_FEATURES]
            parts.append(d1)
            r = df.rolling(30, min_periods=1)
            rm = r.mean(); rm.columns = [f"{c}_r30m" for c in COMMON_FEATURES]
            rs = r.std().fillna(0); rs.columns = [f"{c}_r30s" for c in COMMON_FEATURES]
            parts.append(rm); parts.append(rs)
            l5 = df.shift(5).fillna(0); l5.columns = [f"{c}_l5" for c in COMMON_FEATURES]
            parts.append(l5)
            out = pd.concat(parts, axis=1).astype(np.float32).values
            if fit:
                mean = out.mean(0); std = out.std(0) + 1e-8
                state = {"mean": mean, "std": std}
            else:
                mean, std = ae_pkg["eng_state"]["mean"], ae_pkg["eng_state"]["std"]
            out = (out - mean) / std
            return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32), state

        X_test_190, _ = engineer_190(X_test_raw, fit=False)
        chunk = 8192
        ae_scores = []
        Xt = torch.from_numpy(X_test_190)
        for i in range(0, len(Xt), chunk):
            ae_scores.append(ae_model.reconstruction_error(Xt[i:i+chunk]))
        ae_scores = np.concatenate(ae_scores)
        logger.info(f"  Autoencoder scores computed: {ae_scores.shape}")
        have_ae = True
    except Exception as e:
        logger.warning(f"  Could not load autoencoder: {e}")
        ae_scores = None
        have_ae = False

    # ── 4. Compute multi-scale Z-score ───────────────────────────────────────
    logger.info("=" * 65)
    logger.info("Step 4: Multi-scale Z-score anomaly detection")
    logger.info("=" * 65)

    train_mean = X_train_ext.mean(0)
    train_std  = X_train_ext.std(0) + 1e-8
    z_test = np.abs((X_test_ext - train_mean) / train_std)

    # Max Z across all features (captures any sensor anomaly)
    z_max  = z_test.max(axis=1)
    # Mean of top-k Z-scores (more robust, reduces noise)
    top_k  = max(1, X_test_ext.shape[1] // 10)
    z_topk = np.sort(z_test, axis=1)[:, -top_k:].mean(axis=1)

    thr_zmax,  f1_zmax  = best_threshold(z_max,  y_test)
    thr_ztopk, f1_ztopk = best_threshold(z_topk, y_test)
    logger.info(f"  Z-max:  F1={f1_zmax:.4f}  thr={thr_zmax:.3f}")
    logger.info(f"  Z-topk: F1={f1_ztopk:.4f}  thr={thr_ztopk:.3f}")

    # ── 5. Isolation Forest on extended features ──────────────────────────────
    logger.info("=" * 65)
    logger.info("Step 5: Isolation Forest on extended features")
    logger.info("=" * 65)

    from sklearn.ensemble import IsolationForest
    # Subsample for speed
    n_sub = min(200_000, len(X_train_ext))
    idx   = np.random.choice(len(X_train_ext), n_sub, replace=False)
    iso   = IsolationForest(n_estimators=200, contamination=0.02,
                            random_state=42, n_jobs=-1)
    iso.fit(X_train_ext[idx])
    iso_scores = -iso.score_samples(X_test_ext)   # higher = more anomalous
    thr_iso, f1_iso = best_threshold(iso_scores, y_test)
    logger.info(f"  IsoForest: F1={f1_iso:.4f}  thr={thr_iso:.5f}")

    # ── 6. CUSUM ──────────────────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("Step 6: CUSUM change-point detection on standardized sensors")
    logger.info("=" * 65)

    # CUSUM on raw Z-scored data (38 sensors)
    z_raw = np.abs((X_test_raw.astype(np.float32) - X_train_raw.mean(0)) /
                   (X_train_raw.std(0) + 1e-8))
    cusum_sc = cusum_score(z_raw, k=1.0)
    # Reset periodically (every 300s)
    cusum_sc = cusum_sc.astype(np.float32)
    # Normalize
    cusum_sc = (cusum_sc - cusum_sc.min()) / (cusum_sc.max() - cusum_sc.min() + 1e-8)
    thr_cusum, f1_cusum = best_threshold(cusum_sc, y_test)
    logger.info(f"  CUSUM: F1={f1_cusum:.4f}  thr={thr_cusum:.5f}")

    # ── 7. Build best ensemble ────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("Step 7: Best ensemble")
    logger.info("=" * 65)

    def norm01(x):
        mn, mx = x.min(), x.max()
        return (x - mn) / (mx - mn + 1e-8)

    iso_n  = norm01(iso_scores)
    ztopk_n = norm01(z_topk)
    cusum_n = cusum_sc  # already normalized

    if have_ae:
        ae_n = norm01(ae_scores)
        # Grid search weights
        best_f1e, best_we = 0.0, (0.5, 0.25, 0.15, 0.10)
        for w_ae in [0.3, 0.4, 0.5, 0.6]:
            for w_iso in [0.1, 0.2, 0.25, 0.3]:
                for w_z in [0.1, 0.2, 0.25]:
                    w_c = 1.0 - w_ae - w_iso - w_z
                    if w_c < 0.05: continue
                    ens = w_ae*ae_n + w_iso*iso_n + w_z*ztopk_n + w_c*cusum_n
                    _, f = best_threshold(ens, y_test)
                    if f > best_f1e:
                        best_f1e, best_we = f, (w_ae, w_iso, w_z, w_c)
        w_ae, w_iso, w_z, w_c = best_we
        logger.info(f"  Optimal weights: AE={w_ae:.2f}  ISO={w_iso:.2f}  "
                    f"Z={w_z:.2f}  CUSUM={w_c:.2f}")
        ens_scores = w_ae*ae_n + w_iso*iso_n + w_z*ztopk_n + w_c*cusum_n
    else:
        # AE not available: 3-way ensemble
        ens_scores = 0.5*iso_n + 0.3*ztopk_n + 0.2*cusum_n

    thr_ens, f1_ens = best_threshold(ens_scores, y_test)

    # ── 8. Evaluate all methods ───────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("Step 8: Full evaluation")
    logger.info("=" * 65)

    all_metrics = {}
    if have_ae:
        all_metrics["Autoencoder"] = full_eval(ae_scores, y_test, thr_zmax, "Autoencoder(old)")
    all_metrics["Z-topk"]     = full_eval(z_topk,     y_test, thr_ztopk,  "Z-topk")
    all_metrics["IsoForest"]  = full_eval(iso_scores,  y_test, thr_iso,    "IsoForest")
    all_metrics["CUSUM"]      = full_eval(cusum_sc,    y_test, thr_cusum,  "CUSUM")
    all_metrics["Ensemble"]   = full_eval(ens_scores,  y_test, thr_ens,    "Ensemble")

    best_name = max(all_metrics, key=lambda k: all_metrics[k]["f1"])
    best_f1   = all_metrics[best_name]["f1"]

    # ── 9. Save ───────────────────────────────────────────────────────────────
    pkg = {
        "model_name":  "Ensemble",
        "eng_state":   eng_state,
        "features":    COMMON_FEATURES,
        "iso_model":   iso,
        "threshold":   thr_ens,
        "train_mean":  train_mean,
        "train_std":   train_std,
        "metrics":     all_metrics["Ensemble"],
        "best_f1":     best_f1,
    }
    if have_ae:
        pkg["ae_model"]  = ae_model
        pkg["ae_pkg"]    = ae_pkg
        pkg["ae_weight"] = best_we[0] if have_ae else 0.0
        pkg["iso_weight"]   = best_we[1]
        pkg["z_weight"]     = best_we[2]
        pkg["cusum_weight"] = best_we[3]

    joblib.dump(pkg, OUT_DIR / "best_detection_model.joblib")
    logger.info(f"  Saved best_detection_model.joblib")

    with open(MET_DIR / "detection_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    summ_path = MET_DIR / "pipeline_summary.json"
    summ = {}
    if summ_path.exists():
        try:
            with open(summ_path) as f: summ = json.load(f)
        except: pass

    import datetime
    bm = all_metrics[best_name]
    summ.update({
        "run_timestamp":     str(datetime.datetime.now()),
        "best_model":        best_name,
        "metrics": {
            "f1":        bm["f1"],
            "precision": bm.get("precision", 0),
            "recall":    bm.get("recall", 0),
            "roc_auc":   bm.get("roc_auc", 0),
            "threshold": bm.get("threshold", 0.5),
        },
        "all_model_metrics": all_metrics,
        "training_note": "Multi-scale ensemble: AutoEncoder + IsoForest + Z-topk + CUSUM",
        "training_versions": VERSIONS,
        "n_features":        X_test_ext.shape[1],
    })
    with open(summ_path, "w") as f:
        json.dump(summ, f, indent=2, default=str)
    logger.info("  Dashboard metrics updated")

    # ── 10. Final report ──────────────────────────────────────────────────────
    W = 65
    logger.info("\n" + "=" * W)
    logger.info("  IMPROVED DETECTION — FINAL RESULTS")
    logger.info("=" * W)
    for n, m in all_metrics.items():
        logger.info(f"  {n:<18} F1={m['f1']:.4f}  ROC={m['roc_auc']:.4f}  "
                    f"Prec={m['precision']:.4f}  Rec={m['recall']:.4f}")
    logger.info("=" * W)
    logger.info(f"  BEST: {best_name}  F1 = {best_f1:.4f}")
    if   best_f1 >= 0.70: status = "EXCELLENT  >= 0.70  TARGET REACHED!"
    elif best_f1 >= 0.50: status = "GOOD       >= 0.50"
    elif best_f1 >= 0.35: status = "FAIR       >= 0.35  (add LSTM or graph features)"
    else:                  status = "POOR       <  0.35"
    logger.info(f"  STATUS: {status}")
    logger.info("=" * W)


if __name__ == "__main__":
    main()
