"""
Evaluate Digital Twin detection performance on haiend test set.

Runs the full digital twin (all layers) on the test data and reports:
  - F1, Precision, Recall, ROC-AUC per layer
  - Combined final score metrics
  - Physics violation statistics

Usage: python eval_digital_twin.py
"""

import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import json

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
        X_parts.append(X_df.iloc[:, 1:].ffill().fillna(0).astype(np.float32))
        y_parts.append(y_df["label"].values.astype(np.int32))
    X = pd.concat(X_parts, axis=0, ignore_index=True)
    y = np.concatenate(y_parts)
    return X, y


def best_threshold(scores, y):
    from sklearn.metrics import f1_score
    bf, bt = 0.0, scores.mean()
    for p in np.arange(70, 99.9, 0.1):
        t = np.percentile(scores, p)
        f = f1_score(y, (scores >= t).astype(int), zero_division=0)
        if f > bf: bf, bt = f, t
    return float(bt), float(bf)


def full_eval(scores, y, name, thr=None):
    from sklearn.metrics import (f1_score, precision_score, recall_score,
                                 roc_auc_score, confusion_matrix)
    if thr is None:
        thr, _ = best_threshold(scores, y)
    pred = (scores >= thr).astype(int)
    f1   = f1_score(y, pred, zero_division=0)
    pre  = precision_score(y, pred, zero_division=0)
    rec  = recall_score(y, pred, zero_division=0)
    try:   roc = roc_auc_score(y, scores)
    except: roc = float("nan")
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    logger.info(f"  [{name:<32}] F1={f1:.4f}  P={pre:.4f}  R={rec:.4f}  ROC={roc:.4f}"
                f"   TP={tp}  FP={fp}  FN={fn}")
    return dict(f1=float(f1), precision=float(pre), recall=float(rec),
                roc_auc=float(roc), threshold=float(thr),
                tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn))


def main():
    logger.info("=" * 65)
    logger.info("Digital Twin — Full Evaluation on haiend test set")
    logger.info("=" * 65)

    # ── Load test data ─────────────────────────────────────────────────────────
    logger.info("Loading test data...")
    X_df, y_test = load_test()
    logger.info(f"  Test: {X_df.shape}  attacks={y_test.sum()} ({y_test.mean()*100:.2f}%)")

    col_names = list(X_df.columns)
    X_test    = X_df.values.astype(np.float32)
    T         = len(X_test)

    # ── Build a minimal config for DigitalTwin ─────────────────────────────────
    config = {
        "paths":        {"outputs": "outputs"},
        "digital_twin": {"anomaly_threshold": 0.5, "alert_cooldown": 0},
        "data":         {"label_col": "label"},
    }

    # ── Initialise Digital Twin ────────────────────────────────────────────────
    from src.digital_twin.digital_twin import DigitalTwin

    logger.info("Initialising Digital Twin...")
    twin = DigitalTwin(config, feature_names=col_names)

    # Baseline from training data (first train file as proxy)
    logger.info("Fitting baseline from training data...")
    train_df = pd.read_csv(HAIEND_DIR / "end-train1.csv")
    X_base   = train_df.iloc[:, 1:].ffill().fillna(0).astype(np.float32).values
    # Align columns: baseline may differ if train has same cols
    twin.fit_baseline(X_base[:, :len(col_names)])

    # Load all models
    logger.info("Loading models...")
    loaded = twin.load_best_model(str(OUT_DIR))
    if not loaded:
        logger.error("No models loaded — check outputs/models/")
        return

    # ── Run digital twin on full test set ──────────────────────────────────────
    logger.info(f"Running digital twin on {T:,} test timesteps...")
    logger.info("  (this may take a minute — full window-based LSTM inference)")

    layer_scores = {
        "lstm_haiend_raw": np.zeros(T, dtype=np.float32),  # absolute MSE (same as batch)
        "lstm_haiend_ns":  np.zeros(T, dtype=np.float32),  # normalised (display)
        "physics":         np.zeros(T, dtype=np.float32),
        "zscore":          np.zeros(T, dtype=np.float32),
        "final":           np.zeros(T, dtype=np.float32),
    }
    predictions = np.zeros(T, dtype=np.int32)

    for t in range(T):
        if t % 50_000 == 0 and t > 0:
            logger.info(f"  Progress: {t:,}/{T:,}  ({t/T*100:.0f}%)")
        result = twin.ingest(X_test[t])
        scores = result.get("layer_scores", {})
        layer_scores["lstm_haiend_raw"][t] = scores.get("lstm_haiend_raw", 0.0)
        layer_scores["lstm_haiend_ns"][t]  = scores.get("lstm_haiend", 0.0)
        layer_scores["physics"][t]         = scores.get("physics", 0.0)
        layer_scores["zscore"][t]          = scores.get("zscore", 0.0)
        layer_scores["final"][t]           = result["anomaly_score"]
        predictions[t]                     = int(result["is_anomalous"])

    logger.info("  Done.")

    # ── Evaluate each layer ────────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("Per-layer detection metrics:")
    logger.info("=" * 65)

    all_metrics = {}

    if twin._haiend_model is not None:
        # Use absolute MSE scores — same ranking as batch mode
        all_metrics["LSTM_haiend_raw"] = full_eval(
            layer_scores["lstm_haiend_raw"], y_test, "LSTM-haiend (abs MSE)")
        # Also report normalised for reference
        all_metrics["LSTM_haiend_ns"]  = full_eval(
            layer_scores["lstm_haiend_ns"],  y_test, "LSTM-haiend (normalised)")

    if twin._physics_models:
        all_metrics["Physics_raw"] = full_eval(
            layer_scores["physics"], y_test, "Physics residual (raw)")

    all_metrics["Zscore_raw"] = full_eval(
        layer_scores["zscore"], y_test, "Z-score (raw)")

    # Final combined score
    logger.info("=" * 65)
    logger.info("Combined Digital Twin score:")
    logger.info("=" * 65)
    all_metrics["DT_combined"] = full_eval(
        layer_scores["final"], y_test, "Digital Twin (combined)")

    # Digital twin's actual binary prediction (uses LSTM pre-calibrated threshold)
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
    f1_fixed  = f1_score(y_test, predictions, zero_division=0)
    pre_fixed = precision_score(y_test, predictions, zero_division=0)
    rec_fixed = recall_score(y_test, predictions, zero_division=0)
    try:    roc_fixed = roc_auc_score(y_test, layer_scores["lstm_haiend_raw"])
    except: roc_fixed = float("nan")
    logger.info(f"\n  [DT binary prediction] F1={f1_fixed:.4f}  P={pre_fixed:.4f}  R={rec_fixed:.4f}  "
                f"ROC(LSTM)={roc_fixed:.4f}")
    all_metrics["DT_binary"] = dict(f1=float(f1_fixed), precision=float(pre_fixed),
                                    recall=float(rec_fixed), roc_auc=float(roc_fixed))

    # ── Summary ────────────────────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("All results ranked by F1:")
    for n, m in sorted(all_metrics.items(), key=lambda x: -x[1]["f1"]):
        logger.info(f"  {n:<36}  F1={m['f1']:.4f}  ROC={m['roc_auc']:.4f}")

    best_name = max(all_metrics, key=lambda k: all_metrics[k]["f1"])
    best_f1   = all_metrics[best_name]["f1"]
    logger.info(f"\n  BEST: {best_name}  F1={best_f1:.4f}")

    # ── Save metrics ───────────────────────────────────────────────────────────
    out_path = MET_DIR / "digital_twin_eval.json"
    with open(out_path, "w") as f:
        json.dump({
            "timestamp":   str(pd.Timestamp.now()),
            "test_shape":  list(X_df.shape),
            "n_attacks":   int(y_test.sum()),
            "attack_rate": float(y_test.mean()),
            "models_active": twin.get_state().get("models_active", []),
            "metrics":     all_metrics,
            "best_layer":  best_name,
            "best_f1":     best_f1,
        }, f, indent=2, default=str)
    logger.info(f"\n  Saved: {out_path}")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
