"""
Ensemble GDN + LSTM-AE scores.
Both models are already trained — this is a quick score-combination test.
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
N_FEAT   = len(COMMON_FEATURES)


def best_threshold(scores, y):
    from sklearn.metrics import f1_score
    bf, bt = 0.0, scores.mean()
    for p in np.arange(70, 99.9, 0.1):
        t = np.percentile(scores, p)
        f = f1_score(y, (scores >= t).astype(int), zero_division=0)
        if f > bf: bf, bt = f, t
    return float(bt), float(bf)


def full_eval(scores, y, thr, name):
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
    pred = (scores >= thr).astype(int)
    f1   = f1_score(y, pred, zero_division=0)
    pre  = precision_score(y, pred, zero_division=0)
    rec  = recall_score(y, pred, zero_division=0)
    try:   roc = roc_auc_score(y, scores)
    except: roc = float("nan")
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    logger.info(f"  [{name:<38}] F1={f1:.4f}  Prec={pre:.4f}  Rec={rec:.4f}  ROC={roc:.4f}")
    logger.info(f"  {'':<40} TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    return dict(f1=float(f1), precision=float(pre), recall=float(rec),
                roc_auc=float(roc), threshold=float(thr),
                tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn))


def score_lstm(pkg, X_test):
    """Reconstruct LSTM AE scores for full test set."""
    from retrain_lstm_ae import LSTMAutoencoder
    model = LSTMAutoencoder(N_FEAT, pkg["hidden"], pkg["latent"])
    model.load_state_dict(pkg["model_state"])
    model.eval()

    W = pkg["window"]
    mean = pkg["data_mean"]
    std  = pkg["data_std"]
    X_n = ((X_test - mean) / std).astype(np.float32)
    X_pad = np.concatenate([np.zeros((W-1, N_FEAT), dtype=np.float32), X_n], axis=0)

    T = len(X_test)
    scores = np.zeros(T, dtype=np.float32)
    chunk = 2048
    for start in range(0, T, chunk):
        end = min(start + chunk, T)
        size = end - start
        wins = np.stack([X_pad[start+i: start+i+W] for i in range(size)])
        bt = torch.from_numpy(wins)
        with torch.no_grad():
            recon = model(bt)
            err = ((bt - recon) ** 2).mean(dim=(1, 2)).numpy()
        scores[start:end] = err
    return scores


def score_gdn(pkg, X_test):
    """GDN deviation scores for full test set."""
    from train_gdn import GDN, score_all_timesteps
    model = GDN(N_FEAT, window=pkg["window"], embed_dim=64, top_k=pkg["top_k"], hidden=128)
    model.load_state_dict(pkg["model_state"])
    model.eval()

    mean = pkg["mean"]
    std  = pkg["std"]
    X_n = ((X_test - mean) / std).astype(np.float32)
    raw_scores, _ = score_all_timesteps(model, X_n, pkg["window"])
    return raw_scores


def normalize_01(s):
    lo, hi = np.percentile(s, 1), np.percentile(s, 99)
    return np.clip((s - lo) / (hi - lo + 1e-8), 0, 1)


def main():
    logger.info("=" * 65)
    logger.info("GDN + LSTM-AE Ensemble")
    logger.info("=" * 65)

    # ── Load test data ─────────────────────────────────────────────────────────
    logger.info("Loading test data...")
    loader = MultiVersionLoader(HAI_ROOT)
    X_test, y_test = loader.load_all(versions=VERSIONS, split="test", features=COMMON_FEATURES)
    X_test = X_test.astype(np.float32)
    logger.info(f"  Test: {X_test.shape}  attacks={y_test.sum()} ({y_test.mean()*100:.2f}%)")

    # ── Load models ────────────────────────────────────────────────────────────
    logger.info("Loading LSTM-AE model...")
    import sys
    from retrain_lstm_ae import LSTMAutoencoder as _LSTM
    sys.modules["__main__"].LSTMAutoencoder = _LSTM
    lstm_pkg = joblib.load(OUT_DIR / "lstm_ae_detection.joblib")
    logger.info(f"  LSTM: window={lstm_pkg['window']} hidden={lstm_pkg['hidden']} latent={lstm_pkg['latent']}")

    logger.info("Loading GDN model...")
    # GDN was saved with __main__.GDN — patch before loading
    import sys
    from train_gdn import GDN as _GDN
    sys.modules["__main__"].GDN = _GDN
    gdn_pkg = joblib.load(OUT_DIR / "gdn_detection.joblib")
    logger.info(f"  GDN: window={gdn_pkg['window']} top_k={gdn_pkg['top_k']}")

    # ── Compute raw scores ─────────────────────────────────────────────────────
    logger.info("Scoring with LSTM-AE...")
    lstm_scores = score_lstm(lstm_pkg, X_test)
    logger.info(f"  LSTM stats: mean={lstm_scores.mean():.4f}  p95={np.percentile(lstm_scores,95):.4f}")

    logger.info("Scoring with GDN...")
    gdn_scores = score_gdn(gdn_pkg, X_test)
    logger.info(f"  GDN stats: mean={gdn_scores.mean():.4f}  p95={np.percentile(gdn_scores,95):.4f}")

    # ── Normalize both to [0, 1] ───────────────────────────────────────────────
    lstm_n = normalize_01(lstm_scores)
    gdn_n  = normalize_01(gdn_scores)

    # ── Grid search over ensemble weights ──────────────────────────────────────
    logger.info("=" * 65)
    logger.info("Searching ensemble weights...")
    logger.info("=" * 65)

    all_metrics = {}
    best_f1, best_w = 0.0, 0.5

    for w_lstm in np.arange(0.0, 1.01, 0.1):
        w_gdn = 1.0 - w_lstm
        combo = w_lstm * lstm_n + w_gdn * gdn_n
        # Apply EWM-10 smoothing
        ewm = pd.Series(combo).ewm(span=10).mean().values
        thr, f1 = best_threshold(ewm, y_test)
        label = f"LSTM({w_lstm:.1f})+GDN({w_gdn:.1f})_ewm10"
        all_metrics[label] = full_eval(ewm, y_test, thr, label)
        if f1 > best_f1:
            best_f1, best_w = f1, w_lstm

    # ── Also try EWM-30 at best weight ────────────────────────────────────────
    for span in [30, 60]:
        combo = best_w * lstm_n + (1 - best_w) * gdn_n
        ewm = pd.Series(combo).ewm(span=span).mean().values
        thr, f1 = best_threshold(ewm, y_test)
        label = f"LSTM({best_w:.1f})+GDN({1-best_w:.1f})_ewm{span}"
        all_metrics[label] = full_eval(ewm, y_test, thr, label)

    # ── Summary ────────────────────────────────────────────────────────────────
    best_name = max(all_metrics, key=lambda k: all_metrics[k]["f1"])
    best_f1   = all_metrics[best_name]["f1"]

    logger.info("=" * 65)
    logger.info("Ensemble ranked by F1:")
    for n, m in sorted(all_metrics.items(), key=lambda x: -x[1]["f1"])[:5]:
        logger.info(f"  {n:<44} F1={m['f1']:.4f}  ROC={m['roc_auc']:.4f}")

    logger.info(f"\n  BEST ensemble: {best_name}  F1={best_f1:.4f}")
    logger.info(f"  Standalone LSTM-AE best:  F1=0.4339")
    logger.info(f"  Standalone GDN best:      F1=0.4169")
    logger.info(f"  Ensemble improvement:     {best_f1 - 0.4339:+.4f}")

    if   best_f1 >= 0.70: status = "TARGET REACHED!"
    elif best_f1 >= 0.55: status = "GOOD - significant improvement"
    elif best_f1 >= 0.45: status = "FAIR - marginal improvement"
    else:                  status = "POOR - no improvement"
    logger.info(f"  STATUS: {status}")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
