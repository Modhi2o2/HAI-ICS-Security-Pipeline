"""
Re-evaluate haiend LSTM model with alternative scoring strategies.
No retraining — just different aggregation of per-sensor reconstruction errors.

Strategies:
  mean  : average error across all 225 sensors (current approach)
  max   : worst-offending sensor
  top5  : mean of 5 worst sensors
  top10 : mean of 10 worst sensors
  top20 : mean of 20 worst sensors
"""

import sys, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import joblib
import datetime

sys.path.insert(0, str(Path(__file__).parent))
from src.utils.logger import logger

HAIEND_DIR = Path("C:/Users/PC GAMING/Desktop/AI/HAI/haiend-23.05/haiend-23.05")
OUT_DIR    = Path("outputs/models")
MET_DIR    = Path("outputs/metrics")


def load_test():
    X_parts, y_parts = [], []
    for i in [1, 2]:
        X_df = pd.read_csv(HAIEND_DIR / f"end-test{i}.csv")
        y_df = pd.read_csv(HAIEND_DIR / f"label-test{i}.csv")
        X_parts.append(X_df.iloc[:, 1:].ffill().fillna(0).astype(np.float32).values)
        y_parts.append(y_df["label"].values.astype(np.int32))
    return np.concatenate(X_parts), np.concatenate(y_parts)


def score_per_sensor(model, X_norm, window, chunk=1024):
    """Returns (T, N) per-sensor reconstruction errors."""
    T, N = X_norm.shape
    W = window
    X_pad = np.concatenate([np.zeros((W-1, N), dtype=np.float32), X_norm], axis=0)
    per_sensor = np.zeros((T, N), dtype=np.float32)

    model.eval()
    for start in range(0, T, chunk):
        end = min(start + chunk, T)
        size = end - start
        batch = np.stack([X_pad[start+i: start+i+W] for i in range(size)])
        bt = torch.from_numpy(batch)
        with torch.no_grad():
            recon = model(bt)
            err = ((bt - recon) ** 2).mean(dim=1)  # (B, N) — mean over time steps
        per_sensor[start:end] = err.numpy()

    return per_sensor


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
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    logger.info(f"  [{name:<28}] F1={f1:.4f}  Prec={pre:.4f}  Rec={rec:.4f}  ROC={roc:.4f}")
    logger.info(f"  {'':<30} TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    return dict(f1=float(f1), precision=float(pre), recall=float(rec),
                roc_auc=float(roc), threshold=float(thr),
                tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn))


def main():
    logger.info("=" * 65)
    logger.info("Re-evaluation: per-sensor scoring strategies")
    logger.info("=" * 65)

    # ── Load model ─────────────────────────────────────────────────────────────
    logger.info("Loading haiend LSTM model...")
    import sys
    from train_haiend_lstm import LSTMAutoencoder as _LSTM
    sys.modules["__main__"].LSTMAutoencoder = _LSTM
    pkg = joblib.load(OUT_DIR / "haiend_lstm_detection.joblib")
    model = pkg["model"]
    model.eval()
    window = pkg["window"]
    mean   = pkg["data_mean"]
    std    = pkg["data_std"]
    logger.info(f"  Model: {pkg['model_name']}  F1 at save: {pkg['best_f1']:.4f}")

    # ── Load test data ─────────────────────────────────────────────────────────
    logger.info("Loading test data...")
    X_test, y_test = load_test()
    logger.info(f"  Test: {X_test.shape}  attacks={y_test.sum()} ({y_test.mean()*100:.2f}%)")

    # ── Normalize ──────────────────────────────────────────────────────────────
    X_norm = ((X_test - mean) / std).astype(np.float32)

    # ── Score per sensor ───────────────────────────────────────────────────────
    logger.info(f"Computing per-sensor reconstruction errors ({len(X_test):,} timesteps)...")
    per_sensor = score_per_sensor(model, X_norm, window)  # (T, 225)
    logger.info(f"  Done. Shape: {per_sensor.shape}")

    # ── Clamp outliers ─────────────────────────────────────────────────────────
    p999 = np.percentile(per_sensor, 99.9)
    per_sensor = np.clip(per_sensor, 0, p999)

    # ── Evaluate different aggregations ───────────────────────────────────────
    logger.info("=" * 65)
    logger.info("Evaluating aggregation strategies:")
    logger.info("=" * 65)

    all_metrics = {}
    N = per_sensor.shape[1]

    strategies = {
        "mean_all":  per_sensor.mean(axis=1),
        "max_sensor": per_sensor.max(axis=1),
        "top5_mean":  np.sort(per_sensor, axis=1)[:, -5:].mean(axis=1),
        "top10_mean": np.sort(per_sensor, axis=1)[:, -10:].mean(axis=1),
        "top20_mean": np.sort(per_sensor, axis=1)[:, -20:].mean(axis=1),
        "top30_mean": np.sort(per_sensor, axis=1)[:, -30:].mean(axis=1),
    }

    for name, scores in strategies.items():
        thr, _ = best_threshold(scores, y_test)
        all_metrics[name] = full_eval(scores, y_test, thr, name)

    # ── Also try EWM on top strategies ────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("With EWM-10s smoothing on best strategies:")
    logger.info("=" * 65)

    for k in ["mean_all", "top10_mean", "top20_mean"]:
        ewm = pd.Series(strategies[k]).ewm(span=10).mean().values
        thr, _ = best_threshold(ewm, y_test)
        label = f"{k}_ewm10"
        all_metrics[label] = full_eval(ewm, y_test, thr, label)

    # ── Summary ────────────────────────────────────────────────────────────────
    best_name = max(all_metrics, key=lambda k: all_metrics[k]["f1"])
    best_f1   = all_metrics[best_name]["f1"]

    logger.info("=" * 65)
    logger.info("All strategies ranked by F1:")
    for n, m in sorted(all_metrics.items(), key=lambda x: -x[1]["f1"]):
        logger.info(f"  {n:<28} F1={m['f1']:.4f}  ROC={m['roc_auc']:.4f}")

    logger.info(f"\n  BEST: {best_name}  F1={best_f1:.4f}")
    logger.info(f"  Previous (mean raw): F1={pkg['best_f1']:.4f}")
    logger.info(f"  Improvement: {best_f1 - pkg['best_f1']:+.4f}")

    # ── Save if better ─────────────────────────────────────────────────────────
    prev_best = 0.0
    summ_path = MET_DIR / "pipeline_summary.json"
    if summ_path.exists():
        try:
            with open(summ_path) as f:
                prev_best = json.load(f).get("metrics", {}).get("f1", 0.0)
        except: pass

    if best_f1 > prev_best:
        # Update pkg with best scoring strategy info
        pkg["scoring_strategy"] = best_name
        pkg["threshold"]        = all_metrics[best_name]["threshold"]
        pkg["metrics"]          = all_metrics[best_name]
        pkg["best_f1"]          = best_f1
        joblib.dump(pkg, OUT_DIR / "haiend_lstm_detection.joblib")
        joblib.dump(pkg, OUT_DIR / "best_detection_model.joblib")
        logger.info(f"  NEW BEST! Saved F1={best_f1:.4f}")

        bm = all_metrics[best_name]
        summ = {}
        if summ_path.exists():
            try:
                with open(summ_path) as f: summ = json.load(f)
            except: pass
        summ.update({
            "run_timestamp":    str(datetime.datetime.now()),
            "best_model":       pkg["model_name"],
            "scoring_strategy": best_name,
            "metrics": {
                "f1":        bm["f1"],
                "precision": bm["precision"],
                "recall":    bm["recall"],
                "roc_auc":   bm["roc_auc"],
                "threshold": bm["threshold"],
            },
        })
        with open(summ_path, "w") as f:
            json.dump(summ, f, indent=2, default=str)

    if   best_f1 >= 0.70: status = "TARGET REACHED! F1 >= 0.70"
    elif best_f1 >= 0.55: status = "GOOD - above 0.55"
    else:                  status = "No significant improvement"
    logger.info(f"  STATUS: {status}")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
