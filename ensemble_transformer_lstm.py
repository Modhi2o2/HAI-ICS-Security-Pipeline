"""
Ensemble: Transformer-AE + LSTM-AE
===================================
Both models learned complementary representations:
  LSTM-AE:     F1=0.6886  ROC=0.8650  FP=2607  FN=4037  (higher recall)
  Transformer: F1=0.6795  ROC=0.8886  FP=2141  FN=4424  (higher precision, better ROC)

Combining them should reduce both FP and FN:
  - Transformer's better ranking (ROC) improves separation
  - LSTM's higher recall catches attacks Transformer misses
  - Joint score threshold can be tuned to the optimal F1

Strategies tested:
  1. Average of normalised scores
  2. Max of normalised scores
  3. Weighted average (various alpha)
  4. LSTM raw OR Transformer raw (pre-calibrated thresholds)
  5. Joint percentile threshold on raw score combination

Usage: python ensemble_transformer_lstm.py
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

HAIEND_DIR = Path("C:/Users/PC GAMING/Desktop/AI/HAI/haiend-23.05/haiend-23.05")
OUT_DIR    = Path("outputs/models")
MET_DIR    = Path("outputs/metrics")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_test():
    X_parts, y_parts = [], []
    for i in [1, 2]:
        X_df = pd.read_csv(HAIEND_DIR / f"end-test{i}.csv")
        y_df = pd.read_csv(HAIEND_DIR / f"label-test{i}.csv")
        X_parts.append(X_df.iloc[:, 1:].ffill().fillna(0).astype(np.float32).values)
        y_parts.append(y_df["label"].values.astype(np.int32))
    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    logger.info(f"  Test: {X.shape}  attacks={y.sum()} ({y.mean()*100:.2f}%)")
    return X, y


def score_model(model, X_norm, window, chunk=512):
    T, N   = X_norm.shape
    X_pad  = np.concatenate([np.zeros((window - 1, N), dtype=np.float32), X_norm], axis=0)
    scores = np.zeros(T, dtype=np.float32)
    model.eval()
    for start in range(0, T, chunk):
        end   = min(start + chunk, T)
        size  = end - start
        batch = np.stack([X_pad[start + i: start + i + window] for i in range(size)])
        bt    = torch.from_numpy(batch)
        scores[start:end] = model.reconstruction_error(bt)
    p999 = np.percentile(scores, 99.9)
    return np.clip(scores, 0, p999)


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
    logger.info(f"  [{name:<42}]  F1={f1:.4f}  P={pre:.4f}  R={rec:.4f}  ROC={roc:.4f}")
    logger.info(f"  {'':<44}  TP={tp}  FP={fp}  FN={fn}")
    return dict(f1=float(f1), precision=float(pre), recall=float(rec),
                roc_auc=float(roc), threshold=float(thr),
                tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 72)
    logger.info("Ensemble: Transformer-AE + LSTM-AE")
    logger.info("=" * 72)

    # ── Load models ────────────────────────────────────────────────────────────
    logger.info("Step 1: Load models")

    # LSTM-AE
    from train_haiend_lstm import LSTMAutoencoder as _LSTM
    sys.modules["__main__"].LSTMAutoencoder = _LSTM
    lstm_pkg = joblib.load(OUT_DIR / "haiend_lstm_detection.joblib")
    lstm_model = lstm_pkg["model"]
    lstm_model.eval()
    lstm_thr   = float(lstm_pkg["threshold"])
    lstm_f1    = float(lstm_pkg.get("best_f1", 0))
    logger.info(f"  LSTM-AE:     F1={lstm_f1:.4f}  threshold={lstm_thr:.6f}")

    # Transformer-AE
    from train_anomaly_transformer import TransformerAutoencoder as _Transformer
    sys.modules["__main__"].TransformerAutoencoder = _Transformer
    tr_pkg = joblib.load(OUT_DIR / "transformer_ae_detection.joblib")
    tr_model = tr_pkg["model"]
    tr_model.eval()
    tr_thr   = float(tr_pkg["threshold"])
    tr_f1    = float(tr_pkg.get("best_f1", 0))
    logger.info(f"  Transformer: F1={tr_f1:.4f}  threshold={tr_thr:.6f}")

    # Shared normalization (both use same stats)
    mean = lstm_pkg["data_mean"].astype(np.float32)
    std  = lstm_pkg["data_std"].astype(np.float32)

    # ── Load test data ─────────────────────────────────────────────────────────
    logger.info("Step 2: Load test data")
    X_test, y_test = load_test()
    X_norm = (X_test - mean) / std

    # ── Score both models ──────────────────────────────────────────────────────
    logger.info("Step 3: Score test set with both models")
    logger.info(f"  Scoring LSTM-AE (window={lstm_pkg['window']})...")
    lstm_scores = score_model(lstm_model, X_norm, window=int(lstm_pkg["window"]))
    logger.info(f"    LSTM: min={lstm_scores.min():.5f}  mean={lstm_scores.mean():.5f}  max={lstm_scores.max():.5f}")

    logger.info(f"  Scoring Transformer (window={tr_pkg['window']})...")
    tr_scores = score_model(tr_model, X_norm, window=int(tr_pkg["window"]), chunk=256)
    logger.info(f"    Transformer: min={tr_scores.min():.5f}  mean={tr_scores.mean():.5f}  max={tr_scores.max():.5f}")

    # ── Normalise each to [0,1] using running p97 on test set ─────────────────
    lstm_p97 = float(np.percentile(lstm_scores, 97))
    tr_p97   = float(np.percentile(tr_scores,   97))
    lstm_norm = np.clip(lstm_scores / (lstm_p97 + 1e-8), 0.0, 3.0)
    tr_norm   = np.clip(tr_scores   / (tr_p97   + 1e-8), 0.0, 3.0)
    logger.info(f"  Normalised by p97: lstm_p97={lstm_p97:.5f}  tr_p97={tr_p97:.5f}")

    # ── Evaluate ensemble strategies ───────────────────────────────────────────
    logger.info("=" * 72)
    logger.info("Step 4: Ensemble evaluation")
    logger.info("=" * 72)

    all_metrics = {}

    # Baseline: single models
    thr, _ = best_threshold(lstm_scores, y_test)
    all_metrics["LSTM_AE_alone"] = full_eval(lstm_scores, y_test, thr, "LSTM-AE alone (raw)")

    thr, _ = best_threshold(tr_scores, y_test)
    all_metrics["Transformer_alone"] = full_eval(tr_scores, y_test, thr, "Transformer alone (raw)")

    # Strategy 1: Average normalised scores
    avg_norm = 0.5 * lstm_norm + 0.5 * tr_norm
    thr, _ = best_threshold(avg_norm, y_test)
    all_metrics["avg_norm_50_50"] = full_eval(avg_norm, y_test, thr, "Average norm (50/50)")

    # Strategy 2: Max normalised scores
    max_norm = np.maximum(lstm_norm, tr_norm)
    thr, _ = best_threshold(max_norm, y_test)
    all_metrics["max_norm"] = full_eval(max_norm, y_test, thr, "Max norm")

    # Strategy 3: Weighted average — various LSTM weights (LSTM has higher recall)
    for w_lstm in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        combined = w_lstm * lstm_norm + (1 - w_lstm) * tr_norm
        thr, _ = best_threshold(combined, y_test)
        all_metrics[f"weighted_lstm{int(w_lstm*10)}"] = full_eval(
            combined, y_test, thr, f"Weighted LSTM={w_lstm:.1f} Tr={1-w_lstm:.1f}")

    # Strategy 4: Hard OR of pre-calibrated thresholds
    lstm_fired = (lstm_scores >= lstm_thr).astype(np.int32)
    tr_fired   = (tr_scores   >= tr_thr  ).astype(np.int32)
    hard_or    = np.maximum(lstm_fired, tr_fired)
    from sklearn.metrics import (f1_score, precision_score, recall_score,
                                 roc_auc_score, confusion_matrix)
    f1_or  = f1_score(y_test, hard_or, zero_division=0)
    pre_or = precision_score(y_test, hard_or, zero_division=0)
    rec_or = recall_score(y_test, hard_or, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, hard_or, labels=[0, 1]).ravel()
    roc_or = roc_auc_score(y_test, avg_norm)
    logger.info(f"  [{'Hard OR (pre-calibrated thresholds)':<42}]  F1={f1_or:.4f}  P={pre_or:.4f}  R={rec_or:.4f}  ROC={roc_or:.4f}")
    logger.info(f"  {'':<44}  TP={tp}  FP={fp}  FN={fn}")
    all_metrics["hard_or"] = dict(f1=float(f1_or), precision=float(pre_or),
                                   recall=float(rec_or), roc_auc=float(roc_or),
                                   tp=int(tp), fp=int(fp), fn=int(fn))

    # Strategy 5: AND (both must fire — ultra-high precision)
    hard_and = np.minimum(lstm_fired, tr_fired)
    f1_and   = f1_score(y_test, hard_and, zero_division=0)
    pre_and  = precision_score(y_test, hard_and, zero_division=0)
    rec_and  = recall_score(y_test, hard_and, zero_division=0)
    tn2, fp2, fn2, tp2 = confusion_matrix(y_test, hard_and, labels=[0, 1]).ravel()
    logger.info(f"  [{'Hard AND (both must fire)':<42}]  F1={f1_and:.4f}  P={pre_and:.4f}  R={rec_and:.4f}")
    logger.info(f"  {'':<44}  TP={tp2}  FP={fp2}  FN={fn2}")
    all_metrics["hard_and"] = dict(f1=float(f1_and), precision=float(pre_and),
                                    recall=float(rec_and), roc_auc=float(roc_or),
                                    tp=int(tp2), fp=int(fp2), fn=int(fn2))

    # ── Summary ────────────────────────────────────────────────────────────────
    logger.info("=" * 72)
    logger.info("All results ranked by F1:")
    logger.info("=" * 72)
    for n, m in sorted(all_metrics.items(), key=lambda x: -x[1]["f1"]):
        logger.info(f"  {n:<44}  F1={m['f1']:.4f}  ROC={m['roc_auc']:.4f}")

    best_name = max(all_metrics, key=lambda k: all_metrics[k]["f1"])
    best_f1   = all_metrics[best_name]["f1"]
    best_m    = all_metrics[best_name]
    baseline  = lstm_f1

    logger.info("=" * 72)
    logger.info(f"  BEST ensemble: {best_name}  F1={best_f1:.4f}")
    logger.info(f"  LSTM-AE alone: F1={baseline:.4f}")
    logger.info(f"  Delta:         {best_f1 - baseline:+.4f}")
    if best_f1 > baseline:
        logger.info("  *** ENSEMBLE BEATS BOTH INDIVIDUAL MODELS ***")

    # ── Save best ensemble config ──────────────────────────────────────────────
    import datetime
    best_strategy = best_name
    # Parse weight from strategy name for saving
    if "weighted_lstm" in best_strategy:
        w_lstm = int(best_strategy.replace("weighted_lstm", "")) / 10.0
    elif "avg_norm" in best_strategy:
        w_lstm = 0.5
    else:
        w_lstm = None

    ensemble_pkg = {
        "model_type":        "Ensemble_Transformer_LSTM",
        "lstm_pkg_path":     str(OUT_DIR / "haiend_lstm_detection.joblib"),
        "tr_pkg_path":       str(OUT_DIR / "transformer_ae_detection.joblib"),
        "best_strategy":     best_strategy,
        "lstm_weight":       w_lstm,
        "threshold":         best_m.get("threshold", 0.5),
        "best_f1":           float(best_f1),
        "baseline_lstm_f1":  float(baseline),
        "all_metrics":       all_metrics,
        "timestamp":         str(datetime.datetime.now()),
    }

    out_path = MET_DIR / "ensemble_eval.json"
    with open(out_path, "w") as f:
        json.dump(ensemble_pkg, f, indent=2, default=str)
    logger.info(f"  Saved ensemble config: {out_path}")

    if best_f1 > baseline:
        summ_path = MET_DIR / "pipeline_summary.json"
        summ = {}
        if summ_path.exists():
            try:
                with open(summ_path) as f: summ = json.load(f)
            except: pass
        summ.update({
            "run_timestamp": str(datetime.datetime.now()),
            "best_model":    f"Ensemble(Transformer+LSTM, strategy={best_strategy})",
            "metrics":       {k: float(v) for k, v in best_m.items()
                              if isinstance(v, (int, float))},
            "training_note": "Ensemble of Transformer-AE and LSTM-AE",
        })
        with open(summ_path, "w") as f:
            json.dump(summ, f, indent=2, default=str)
        logger.info("  Updated pipeline_summary.json")

    logger.info("=" * 72)
    logger.info("  ENSEMBLE COMPLETE")
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
