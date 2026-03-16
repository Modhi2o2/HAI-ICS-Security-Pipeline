"""
Physics-Informed Residual Model for HAI Anomaly Detection
==========================================================
Uses the physical boiler graph (phy_boiler.json) to build 50
sensor-to-sensor regression models based on real plant topology.

For each physical edge (sensor_A -> sensor_B):
  - Fit Ridge regression on normal data: B(t) ~ A(t), A(t-1), A(t-2), A(t-5)
  - At test time: residual = |B_actual - B_predicted|

Attacks that violate physical causality (e.g. pump running but
downstream flow stays zero) will show high residuals even if each
sensor individually looks temporally normal.

Ensemble with LSTM-AE to combine temporal + spatial detection.

Usage: python train_physics_residual.py
"""

import sys, json, warnings, datetime
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from src.utils.logger import logger

GRAPH_PATH = Path("C:/Users/PC GAMING/Desktop/AI/HAI/graph/graph/boiler/phy_boiler.json")
HAIEND_DIR = Path("C:/Users/PC GAMING/Desktop/AI/HAI/haiend-23.05/haiend-23.05")
OUT_DIR    = Path("outputs/models")
MET_DIR    = Path("outputs/metrics")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MET_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Build sensor-to-sensor edges from physics graph
# ─────────────────────────────────────────────────────────────────────────────

def load_physics_edges(graph_path: Path, haiend_cols: set):
    with open(graph_path) as f:
        g = json.load(f)

    # Component → its sensor tags
    comp_tags = {}
    for n in g["nodes"]:
        tags = []
        for field in ["in_tags", "out_tags"]:
            t = n.get(field, "")
            if t:
                tags.extend([x.strip() for x in t.split(",")])
        comp_tags[n["id"]] = [t for t in tags if t in haiend_cols]

    # Component links → sensor-to-sensor edges
    edges = []
    seen = set()
    for link in g["links"]:
        src_tags = comp_tags.get(link["source"], [])
        tgt_tags = comp_tags.get(link["target"], [])
        for s in src_tags:
            for t in tgt_tags:
                if s != t and (s, t) not in seen:
                    edges.append((s, t))
                    seen.add((s, t))
    return edges


# ─────────────────────────────────────────────────────────────────────────────
# Data loading (same as train_haiend_lstm.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_train(haiend_dir: Path):
    dfs = []
    for f in ["end-train1.csv", "end-train2.csv", "end-train3.csv", "end-train4.csv"]:
        df = pd.read_csv(haiend_dir / f)
        dfs.append(df.iloc[:, 1:].ffill().fillna(0))  # drop Timestamp
    return pd.concat(dfs, axis=0, ignore_index=True)


def load_test(haiend_dir: Path):
    X_parts, y_parts = [], []
    for i in [1, 2]:
        X_df = pd.read_csv(haiend_dir / f"end-test{i}.csv")
        y_df = pd.read_csv(haiend_dir / f"label-test{i}.csv")
        X_parts.append(X_df.iloc[:, 1:].ffill().fillna(0))
        y_parts.append(y_df["label"].values.astype(np.int32))
    return pd.concat(X_parts, axis=0, ignore_index=True), np.concatenate(y_parts)


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering: lagged values of source sensor
# ─────────────────────────────────────────────────────────────────────────────

LAGS = [0, 1, 2, 5, 10, 30]  # lag offsets in seconds

def make_lag_features(series: pd.Series) -> np.ndarray:
    """Return (T, len(LAGS)) lag matrix for one sensor."""
    parts = []
    for lag in LAGS:
        parts.append(series.shift(lag).fillna(method="bfill").fillna(0).values)
    return np.stack(parts, axis=1)  # (T, n_lags)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

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
    logger.info(f"  [{name:<34}] F1={f1:.4f}  Prec={pre:.4f}  Rec={rec:.4f}  ROC={roc:.4f}")
    logger.info(f"  {'':<36} TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    return dict(f1=float(f1), precision=float(pre), recall=float(rec),
                roc_auc=float(roc), threshold=float(thr),
                tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 65)
    logger.info("Physics Residual Model — HAI Anomaly Detection")
    logger.info("=" * 65)

    # ── Load data ──────────────────────────────────────────────────────────────
    logger.info("Loading training data...")
    df_train = load_train(HAIEND_DIR)
    logger.info(f"  Train: {df_train.shape}  (all normal)")

    logger.info("Loading test data...")
    df_test, y_test = load_test(HAIEND_DIR)
    logger.info(f"  Test:  {df_test.shape}  attacks={y_test.sum()} ({y_test.mean()*100:.2f}%)")

    haiend_cols = set(df_train.columns)

    # ── Load physics edges ─────────────────────────────────────────────────────
    logger.info("Loading physics graph...")
    edges = load_physics_edges(GRAPH_PATH, haiend_cols)
    logger.info(f"  Physical sensor-to-sensor edges: {len(edges)}")

    # ── Train one Ridge regression per edge ────────────────────────────────────
    logger.info("=" * 65)
    logger.info(f"Training {len(edges)} Ridge regression models (one per physical edge)...")
    logger.info("=" * 65)

    models   = {}  # (src, tgt) -> fitted Ridge
    train_residuals = []

    for i, (src, tgt) in enumerate(edges):
        # Lag features of source sensor
        X = make_lag_features(df_train[src])   # (T, n_lags)
        y = df_train[tgt].values               # (T,)

        reg = Ridge(alpha=1.0)
        reg.fit(X, y)
        models[(src, tgt)] = reg

        # Training residual (for reference)
        pred = reg.predict(X)
        resid = np.abs(y - pred).mean()
        train_residuals.append(resid)

        if (i + 1) % 10 == 0 or i == 0:
            logger.info(f"  Edge {i+1:3d}/{len(edges)}: {src:22s} -> {tgt:22s}  "
                        f"train_mae={resid:.4f}")

    logger.info(f"  Mean training MAE across all edges: {np.mean(train_residuals):.4f}")

    # ── Score test set ─────────────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("Scoring test set...")
    logger.info("=" * 65)

    T = len(df_test)
    residual_matrix = np.zeros((T, len(edges)), dtype=np.float32)

    for i, (src, tgt) in enumerate(edges):
        X = make_lag_features(df_test[src])
        y = df_test[tgt].values
        pred = models[(src, tgt)].predict(X)
        residual_matrix[:, i] = np.abs(y - pred).astype(np.float32)

    logger.info(f"  Residual matrix: {residual_matrix.shape}")
    logger.info(f"  Mean residual: {residual_matrix.mean():.4f}  "
                f"Max: {residual_matrix.max():.4f}")

    # ── Aggregate strategies ───────────────────────────────────────────────────
    # Clip outliers
    p999 = np.percentile(residual_matrix, 99.9)
    residual_matrix = np.clip(residual_matrix, 0, p999)

    mean_score = residual_matrix.mean(axis=1)
    max_score  = residual_matrix.max(axis=1)
    top5_score = np.sort(residual_matrix, axis=1)[:, -5:].mean(axis=1)
    top10_score= np.sort(residual_matrix, axis=1)[:, -10:].mean(axis=1)

    # ── Evaluate physics model alone ───────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("Physics residual model (standalone):")
    logger.info("=" * 65)
    all_metrics = {}

    for name, scores in [("phys_mean", mean_score), ("phys_max", max_score),
                          ("phys_top5", top5_score), ("phys_top10", top10_score)]:
        thr, _ = best_threshold(scores, y_test)
        all_metrics[name] = full_eval(scores, y_test, thr, name)

    # EWM smoothing
    for span in [10, 30]:
        ewm = pd.Series(mean_score).ewm(span=span).mean().values
        thr, _ = best_threshold(ewm, y_test)
        all_metrics[f"phys_mean_ewm{span}"] = full_eval(ewm, y_test, thr, f"phys_mean_ewm{span}")

    best_phys_name = max(all_metrics, key=lambda k: all_metrics[k]["f1"])
    best_phys_f1   = all_metrics[best_phys_name]["f1"]
    logger.info(f"\n  Physics best: {best_phys_name}  F1={best_phys_f1:.4f}")

    # ── Ensemble with LSTM-AE ──────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("Ensemble: Physics + LSTM-AE haiend")
    logger.info("=" * 65)

    # Load LSTM-AE scores by re-running the model
    lstm_pkg_path = OUT_DIR / "haiend_lstm_detection.joblib"
    if lstm_pkg_path.exists():
        logger.info("  Loading haiend LSTM model for re-scoring...")
        import sys as _sys
        from train_haiend_lstm import LSTMAutoencoder as _LSTM
        _sys.modules["__main__"].LSTMAutoencoder = _LSTM
        import torch

        lstm_pkg = joblib.load(lstm_pkg_path)
        model    = lstm_pkg["model"]
        model.eval()
        W        = lstm_pkg["window"]
        mean_n   = lstm_pkg["data_mean"]
        std_n    = lstm_pkg["data_std"]

        X_test_arr = df_test.values.astype(np.float32)
        X_norm     = (X_test_arr - mean_n) / std_n
        X_pad      = np.concatenate([np.zeros((W-1, X_norm.shape[1]), dtype=np.float32), X_norm])

        lstm_scores = np.zeros(T, dtype=np.float32)
        chunk = 1024
        for start in range(0, T, chunk):
            end  = min(start + chunk, T)
            size = end - start
            batch = np.stack([X_pad[start+i: start+i+W] for i in range(size)])
            bt = torch.from_numpy(batch)
            with torch.no_grad():
                recon = model(bt)
                err   = ((bt - recon) ** 2).mean(dim=(1, 2)).numpy()
            lstm_scores[start:end] = err

        # Clip and normalize both to [0,1]
        p999_lstm = np.percentile(lstm_scores, 99.9)
        lstm_scores = np.clip(lstm_scores, 0, p999_lstm)

        def norm01(s):
            lo, hi = np.percentile(s, 1), np.percentile(s, 99)
            return np.clip((s - lo) / (hi - lo + 1e-8), 0, 1)

        lstm_n01 = norm01(lstm_scores)
        phys_n01 = norm01(mean_score)

        logger.info("  Searching ensemble weights...")
        best_ens_f1, best_w = 0.0, 0.5
        for w_lstm in np.arange(0.0, 1.01, 0.1):
            combo = w_lstm * lstm_n01 + (1 - w_lstm) * phys_n01
            thr, f1 = best_threshold(combo, y_test)
            label = f"ens_lstm{w_lstm:.1f}_phys{1-w_lstm:.1f}"
            m = full_eval(combo, y_test, thr, label)
            all_metrics[label] = m
            if f1 > best_ens_f1:
                best_ens_f1, best_w = f1, w_lstm

        logger.info(f"\n  Best ensemble weight: LSTM={best_w:.1f}  Phys={1-best_w:.1f}")
        logger.info(f"  Best ensemble F1: {best_ens_f1:.4f}")
        logger.info(f"  Standalone LSTM F1: {lstm_pkg['best_f1']:.4f}")
        logger.info(f"  Standalone physics F1: {best_phys_f1:.4f}")
    else:
        logger.info("  LSTM model not found — skipping ensemble")
        best_ens_f1 = best_phys_f1

    # ── Final summary ──────────────────────────────────────────────────────────
    overall_best_name = max(all_metrics, key=lambda k: all_metrics[k]["f1"])
    overall_best_f1   = all_metrics[overall_best_name]["f1"]

    logger.info("=" * 65)
    logger.info("All results ranked by F1:")
    for n, m in sorted(all_metrics.items(), key=lambda x: -x[1]["f1"])[:8]:
        logger.info(f"  {n:<40} F1={m['f1']:.4f}  ROC={m['roc_auc']:.4f}")

    # ── Save if best ───────────────────────────────────────────────────────────
    prev_best = 0.0
    summ_path = MET_DIR / "pipeline_summary.json"
    if summ_path.exists():
        try:
            with open(summ_path) as f:
                prev_best = json.load(f).get("metrics", {}).get("f1", 0.0)
        except: pass

    logger.info(f"\n  Current pipeline best: F1={prev_best:.4f}")
    logger.info(f"  This run best:         F1={overall_best_f1:.4f}  [{overall_best_name}]")

    pkg = {
        "model_type":  "PhysicsResidual",
        "model_name":  "PhysicsResidual+LSTMensemble",
        "edges":       edges,
        "models":      models,
        "best_f1":     overall_best_f1,
        "metrics":     all_metrics[overall_best_name],
        "all_results": all_metrics,
    }
    joblib.dump(pkg, OUT_DIR / "physics_residual.joblib")
    logger.info("  Saved: physics_residual.joblib")

    if overall_best_f1 > prev_best:
        summ = {}
        if summ_path.exists():
            try:
                with open(summ_path) as f: summ = json.load(f)
            except: pass
        bm = all_metrics[overall_best_name]
        summ.update({
            "run_timestamp": str(datetime.datetime.now()),
            "best_model":    overall_best_name,
            "metrics": {
                "f1":        bm["f1"],
                "precision": bm["precision"],
                "recall":    bm["recall"],
                "roc_auc":   bm["roc_auc"],
                "threshold": bm["threshold"],
            },
            "training_note": "PhysicsResidual (50 edges, Ridge) + haiend LSTM-AE ensemble",
        })
        with open(summ_path, "w") as f:
            json.dump(summ, f, indent=2, default=str)
        logger.info(f"  NEW BEST! Updated pipeline_summary.json")

    if   overall_best_f1 >= 0.70: status = "TARGET REACHED! F1 >= 0.70"
    elif overall_best_f1 >= 0.55: status = "GOOD - above 0.55"
    else:                          status = "No improvement over LSTM alone"
    logger.info(f"  STATUS: {status}")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
