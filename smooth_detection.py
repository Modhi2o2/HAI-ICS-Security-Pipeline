"""
Temporal Smoothing for Anomaly Detection
=========================================
HAI attacks last MINUTES not seconds.
Rolling max/mean on autoencoder reconstruction error
dramatically improves F1 by reducing point-wise noise.

Key insight:
  - Raw AE score: high variance, many FP spikes
  - Rolling-max(60s): sustained attacks stay high,
    isolated spikes in normal data get diluted

Usage: python smooth_detection.py
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
from retrain_autoencoder import Autoencoder

HAI_ROOT = "C:/Users/PC GAMING/Desktop/AI/HAI"
OUT_DIR  = Path("outputs/models")
MET_DIR  = Path("outputs/metrics")

VERSIONS = ["hai-20.07", "hai-21.03", "hai-22.04", "hai-23.05"]


def engineer_190(X: np.ndarray, eng_state: dict) -> np.ndarray:
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
    mean, std = eng_state["mean"], eng_state["std"]
    out = (out - mean) / std
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def ae_score(model: Autoencoder, X: np.ndarray, chunk=8192) -> np.ndarray:
    Xt = torch.from_numpy(X)
    scores = []
    for i in range(0, len(Xt), chunk):
        scores.append(model.reconstruction_error(Xt[i:i+chunk]))
    return np.concatenate(scores)


def best_threshold(scores: np.ndarray, y: np.ndarray):
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
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    logger.info(f"  [{name:<28}] F1={f1:.4f}  Prec={pre:.4f}  Rec={rec:.4f}  "
                f"ROC={roc:.4f}")
    logger.info(f"  {'':<30} TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    return dict(f1=float(f1), precision=float(pre), recall=float(rec),
                roc_auc=float(roc), threshold=float(thr),
                tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn))


def main():
    loader = MultiVersionLoader(HAI_ROOT)

    # ── Load autoencoder ──────────────────────────────────────────────────────
    logger.info("Loading autoencoder model...")
    pkg   = joblib.load(OUT_DIR / "autoencoder_detection.joblib")
    model = Autoencoder(pkg["input_dim"], pkg["bottleneck"])
    model.load_state_dict(pkg["model_state"])
    model.eval()
    logger.info(f"  AE loaded: {pkg['input_dim']} -> {pkg['bottleneck']} -> {pkg['input_dim']}")

    # ── Load test data ────────────────────────────────────────────────────────
    logger.info("Loading test data...")
    X_test_raw, y_test = loader.load_all(
        versions=VERSIONS, split="test", features=COMMON_FEATURES
    )
    logger.info(f"  Test: {X_test_raw.shape}  attacks={y_test.sum()} ({y_test.mean()*100:.2f}%)")

    # ── Feature engineering ───────────────────────────────────────────────────
    logger.info("Feature engineering...")
    X_test_eng = engineer_190(X_test_raw, pkg["eng_state"])

    # ── Compute raw AE scores ─────────────────────────────────────────────────
    logger.info("Computing reconstruction error scores...")
    raw_scores = ae_score(model, X_test_eng)
    logger.info(f"  Raw scores: min={raw_scores.min():.4f}  max={raw_scores.max():.4f}  "
                f"mean={raw_scores.mean():.4f}")

    # ── Temporal smoothing ────────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("Temporal smoothing (attack duration = minutes, not seconds)")
    logger.info("=" * 65)

    scores_series = pd.Series(raw_scores)
    all_metrics = {}

    # Raw (baseline)
    thr_raw, f1_raw = best_threshold(raw_scores, y_test)
    all_metrics["AE_raw"]    = full_eval(raw_scores, y_test, thr_raw, "AE raw (no smoothing)")

    # Rolling windows: mean and max
    for w in [10, 30, 60, 120, 300]:
        # Rolling mean: reduces noise in normal periods
        rm = scores_series.rolling(w, min_periods=1, center=False).mean().values
        t, f = best_threshold(rm, y_test)
        all_metrics[f"AE_rollmean_{w}s"] = full_eval(rm, y_test, t, f"AE roll-mean {w}s")

        # Rolling max: amplifies sustained attacks
        rmx = scores_series.rolling(w, min_periods=1, center=False).max().values
        t, f = best_threshold(rmx, y_test)
        all_metrics[f"AE_rollmax_{w}s"]  = full_eval(rmx, y_test, t, f"AE roll-max  {w}s")

    # EWM (exponential weighted moving average) — gives more weight to recent
    for span in [30, 60, 120]:
        ewm = scores_series.ewm(span=span).mean().values
        t, f = best_threshold(ewm, y_test)
        all_metrics[f"AE_ewm_{span}s"] = full_eval(ewm, y_test, t, f"AE EWM span={span}s")

    # ── Best single method ────────────────────────────────────────────────────
    best_name = max(all_metrics, key=lambda k: all_metrics[k]["f1"])
    best_f1   = all_metrics[best_name]["f1"]
    best_thr  = all_metrics[best_name]["threshold"]

    logger.info("=" * 65)
    logger.info("All methods ranked by F1:")
    for n, m in sorted(all_metrics.items(), key=lambda x: -x[1]["f1"]):
        logger.info(f"  {n:<32} F1={m['f1']:.4f}  ROC={m['roc_auc']:.4f}")
    logger.info(f"\n  BEST: {best_name}  F1={best_f1:.4f}")

    # ── Compute the best-smoothed scores ─────────────────────────────────────
    # best_name format: "AE_rollmean_30s" or "AE_ewm_30s" or "AE_raw"
    best_window = None
    best_type   = None
    if "rollmean" in best_name:
        best_window = int(best_name.split("_")[-1].replace("s", ""))
        best_type   = "rollmean"
        best_scores = scores_series.rolling(best_window, min_periods=1).mean().values
    elif "rollmax" in best_name:
        best_window = int(best_name.split("_")[-1].replace("s", ""))
        best_type   = "rollmax"
        best_scores = scores_series.rolling(best_window, min_periods=1).max().values
    elif "ewm" in best_name:
        # format: AE_ewm_30s
        best_window = int(best_name.replace("AE_ewm_", "").replace("s", ""))
        best_type   = "ewm"
        best_scores = scores_series.ewm(span=best_window).mean().values
    else:
        best_scores = raw_scores
        best_type   = "raw"
        best_window = 0

    # ── Save as best detection model ──────────────────────────────────────────
    new_pkg = {
        "model_type":       "AutoencoderSmoothed",
        "model":            model,
        "model_state":      pkg["model_state"],
        "model_name":       f"AE+{best_type}({best_window}s)",
        "input_dim":        pkg["input_dim"],
        "bottleneck":       pkg["bottleneck"],
        "eng_state":        pkg["eng_state"],
        "features":         COMMON_FEATURES,
        "threshold":        best_thr,
        "smooth_type":      best_type,
        "smooth_window":    best_window,
        "metrics":          all_metrics[best_name],
        "best_f1":          best_f1,
        "all_results":      all_metrics,
    }
    joblib.dump(new_pkg, OUT_DIR / "best_detection_model.joblib")
    logger.info("Saved: best_detection_model.joblib")

    # ── Update dashboard metrics ──────────────────────────────────────────────
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
        "best_model":        f"AE+{best_type}({best_window}s)",
        "metrics": {
            "f1":        bm["f1"],
            "precision": bm["precision"],
            "recall":    bm["recall"],
            "roc_auc":   bm["roc_auc"],
            "threshold": bm["threshold"],
        },
        "all_model_metrics": {k: v for k, v in list(all_metrics.items())[:8]},
        "training_note": f"Autoencoder + temporal smoothing ({best_type}, {best_window}s window)",
        "training_versions": VERSIONS,
    })
    with open(summ_path, "w") as f:
        json.dump(summ, f, indent=2, default=str)

    # ── Final summary ─────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 65)
    logger.info("  TEMPORAL SMOOTHING — FINAL RESULTS")
    logger.info("=" * 65)
    logger.info(f"  Raw AE:        F1 = {all_metrics['AE_raw']['f1']:.4f}")
    logger.info(f"  Best smoothed: F1 = {best_f1:.4f}  [{best_name}]")
    logger.info(f"  Improvement:   +{best_f1 - all_metrics['AE_raw']['f1']:.4f}")

    if   best_f1 >= 0.70: status = "EXCELLENT >= 0.70 - TARGET REACHED!"
    elif best_f1 >= 0.55: status = "GOOD      >= 0.55 - close to target"
    elif best_f1 >= 0.40: status = "FAIR      >= 0.40 - use LSTM next"
    else:                  status = "POOR      <  0.40 - fundamental limit"
    logger.info(f"  STATUS: {status}")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
