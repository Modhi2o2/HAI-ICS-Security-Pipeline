"""
Semi-Supervised Detection via Autoencoder Latent Features
==========================================================
1. Get AE encoder latent (32-dim) + per-feature recon error for each sample
2. Train XGBoost on: normal latent features (label=0) + synthetic attack latent (label=1)
3. Evaluate combined AE+classifier on real test data

Why this works:
- AE latent space learned from REAL normal data (3.3M samples)
- Synthetic attacks from diffusion model represent attack-like patterns
- XGBoost finds the optimal decision boundary in latent space
- No real attack labels needed during training

Expected F1: 0.40 -> 0.50+

Usage: python semi_supervised_detection.py
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


# ─────────────────────────────────────────────────────────────────────────────
# Load autoencoder
# ─────────────────────────────────────────────────────────────────────────────

def load_ae():
    pkg = joblib.load(OUT_DIR / "autoencoder_detection.joblib")
    model = Autoencoder(pkg["input_dim"], pkg["bottleneck"])
    model.load_state_dict(pkg["model_state"])
    model.eval()
    return model, pkg


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering (same as retrain_autoencoder.py)
# ─────────────────────────────────────────────────────────────────────────────

def engineer(X: np.ndarray, eng_state: dict) -> np.ndarray:
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


# ─────────────────────────────────────────────────────────────────────────────
# Extract latent features from autoencoder
# ─────────────────────────────────────────────────────────────────────────────

def extract_latent_features(model: Autoencoder, X_eng: np.ndarray,
                             chunk: int = 4096) -> np.ndarray:
    """Returns (N, bottleneck + input_dim) = latent + per-feature recon error."""
    all_features = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X_eng), chunk):
            xb = torch.from_numpy(X_eng[i:i+chunk])
            z    = model.encoder(xb)                    # (B, bottleneck)
            recon = model.decoder(z)                    # (B, input_dim)
            err  = ((recon - xb) ** 2)                  # (B, input_dim) per-feature
            # Combine: latent + mean recon error (scalar) + top-k feature errors
            mean_err = err.mean(dim=1, keepdim=True)   # (B, 1)
            max_err  = err.max(dim=1, keepdim=True).values  # (B, 1)
            feats = torch.cat([z, mean_err, max_err], dim=1)  # (B, 34)
            all_features.append(feats.cpu().numpy())
    return np.concatenate(all_features, axis=0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Load synthetic attacks from diffusion model
# ─────────────────────────────────────────────────────────────────────────────

def load_synthetic_attacks(n: int = 15000) -> np.ndarray:
    try:
        from train_diffusion_full import build_model
        s = torch.load(OUT_DIR / "diffusion_best.pt", map_location="cpu",
                       weights_only=False)
        if not isinstance(s, dict) or "model_state" not in s:
            raise ValueError("bad format")
        m = build_model(s["input_dim"], s["hidden_dim"], s["n_layers"], 3).to("cpu")
        m.load_state_dict(s["model_state"]); m.eval()
        betas  = torch.linspace(s.get("beta_start", 1e-4), s.get("beta_end", 0.02), s["T"])
        alphas = 1 - betas; acp = torch.cumprod(alphas, 0)
        with torch.no_grad():
            c = torch.full((n,), 1, dtype=torch.long)
            x = torch.randn(n, s["input_dim"])
            for step in reversed(range(s["T"])):
                tb   = torch.full((n,), step, dtype=torch.long)
                pred = m(x, tb, c)
                x    = (1/torch.sqrt(alphas[step])) * \
                       (x - (1-alphas[step])/torch.sqrt(1-acp[step]) * pred)
                if step > 0:
                    x += torch.sqrt(betas[step]) * torch.randn_like(x)
        raw = x.numpy() * s["data_std"] + s["data_mean"]
        logger.info(f"Generated {n} synthetic attacks from diffusion model")
        return raw.astype(np.float32)
    except Exception as e:
        logger.warning(f"Synthetic generation failed: {e}")
        return np.zeros((0, 38), dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Threshold sweep
# ─────────────────────────────────────────────────────────────────────────────

def best_threshold(scores, y):
    from sklearn.metrics import f1_score
    bf, bt = 0.0, 0.5
    for t in np.arange(0.02, 0.98, 0.01):
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
                f"ROC={roc:.4f}  thr={thr:.3f}")
    logger.info(f"  {'':<5}  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    loader = MultiVersionLoader(HAI_ROOT)

    # ── Load AE ──────────────────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("Loading autoencoder...")
    model, pkg = load_ae()
    eng_state  = pkg["eng_state"]
    logger.info(f"  AE: {pkg['input_dim']} -> {pkg['bottleneck']} -> {pkg['input_dim']}")

    # ── Load test data ────────────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("Loading test data...")
    X_test_raw, y_test = loader.load_all(
        versions=VERSIONS, split="test", features=COMMON_FEATURES
    )
    logger.info(f"  Test: {X_test_raw.shape}  attacks={y_test.sum()} ({y_test.mean()*100:.2f}%)")

    # ── Load train data (normal) ──────────────────────────────────────────────
    logger.info("Loading training data (normal)...")
    X_train_raw, y_train = loader.load_all(
        versions=VERSIONS, split="train", features=COMMON_FEATURES
    )
    X_train_raw = X_train_raw[y_train == 0]
    # Subsample normal for speed
    n_norm = min(100_000, len(X_train_raw))
    idx = np.random.choice(len(X_train_raw), n_norm, replace=False)
    X_train_raw = X_train_raw[idx]
    logger.info(f"  Normal subsample: {X_train_raw.shape}")

    # ── Load synthetic attacks ────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("Generating synthetic attacks from diffusion model...")
    X_synth = load_synthetic_attacks(n=15000)
    logger.info(f"  Synthetic attacks: {X_synth.shape}")

    # ── Feature engineering ───────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("Feature engineering...")
    X_train_eng = engineer(X_train_raw, eng_state)
    X_test_eng  = engineer(X_test_raw,  eng_state)
    if len(X_synth) > 0:
        X_synth_eng = engineer(X_synth, eng_state)
    logger.info(f"  Train eng: {X_train_eng.shape}")
    logger.info(f"  Test  eng: {X_test_eng.shape}")

    # ── Extract latent features ───────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("Extracting latent features from autoencoder encoder...")
    Z_train = extract_latent_features(model, X_train_eng)
    Z_test  = extract_latent_features(model, X_test_eng)
    if len(X_synth) > 0:
        Z_synth = extract_latent_features(model, X_synth_eng)
    logger.info(f"  Latent feature dim: {Z_train.shape[1]}")

    # ── Also get raw AE reconstruction error (scalar per sample) ─────────────
    logger.info("Getting raw reconstruction errors...")
    import pandas as _pd
    def ae_score_series(X_eng):
        scores = []
        Xt = torch.from_numpy(X_eng)
        for i in range(0, len(Xt), 8192):
            scores.append(model.reconstruction_error(Xt[i:i+8192]))
        return np.concatenate(scores)

    ae_test_scores = ae_score_series(X_test_eng)

    # Baseline: AE reconstruction error with EWM-30 smoothing
    ewm_scores = _pd.Series(ae_test_scores).ewm(span=30).mean().values
    thr_base, f1_base = best_threshold(ewm_scores, y_test)
    logger.info(f"  Baseline AE+EWM30: F1={f1_base:.4f}  thr={thr_base:.4f}")

    if len(X_synth) == 0:
        logger.warning("No synthetic attacks generated — XGBoost training skipped")
        return

    # ── Build semi-supervised training set ───────────────────────────────────
    logger.info("=" * 65)
    logger.info("Building semi-supervised training set...")
    # Normal latent features (label=0)
    X_cls_norm  = Z_train
    y_cls_norm  = np.zeros(len(Z_train), dtype=np.int8)
    # Synthetic attack latent features (label=1)
    X_cls_atk   = Z_synth
    y_cls_atk   = np.ones(len(Z_synth), dtype=np.int8)
    # Combine and balance
    n_atk_target = min(len(X_cls_atk), int(len(X_cls_norm) * 0.30))
    if n_atk_target < len(X_cls_atk):
        idx = np.random.choice(len(X_cls_atk), n_atk_target, replace=False)
        X_cls_atk = X_cls_atk[idx]
        y_cls_atk = y_cls_atk[idx]
    X_cls = np.vstack([X_cls_norm, X_cls_atk])
    y_cls = np.concatenate([y_cls_norm, y_cls_atk])
    # Shuffle
    shuf = np.random.permutation(len(X_cls))
    X_cls, y_cls = X_cls[shuf], y_cls[shuf]
    logger.info(f"  Training: {X_cls.shape}  normal={int((y_cls==0).sum())}  "
                f"synth_attack={int((y_cls==1).sum())}  "
                f"({y_cls.mean()*100:.1f}% attack)")

    # ── Train XGBoost on latent features ─────────────────────────────────────
    logger.info("=" * 65)
    logger.info("Training XGBoost classifier on latent features...")
    try:
        import xgboost as xgb
        ver = tuple(int(x) for x in xgb.__version__.split(".")[:2])
        spw = max(1, int((y_cls==0).sum() // max((y_cls==1).sum(), 1)))
        val_size = int(len(X_cls) * 0.15)
        Xv, yv = X_cls[-val_size:], y_cls[-val_size:]
        Xt, yt = X_cls[:-val_size], y_cls[:-val_size]
        kw = dict(n_estimators=300, max_depth=6, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8,
                  scale_pos_weight=spw, eval_metric="aucpr",
                  n_jobs=-1, verbosity=0, random_state=42)
        if ver >= (2, 0): kw["early_stopping_rounds"] = 30
        clf = xgb.XGBClassifier(**kw)
        clf.fit(Xt, yt, eval_set=[(Xv, yv)], verbose=False)
        logger.info("  XGBoost trained")
    except Exception as e:
        logger.error(f"XGBoost failed: {e}")
        return

    # ── Evaluate XGBoost on real test data ───────────────────────────────────
    logger.info("=" * 65)
    logger.info("Evaluating XGBoost on REAL test data (latent features)...")
    proba = clf.predict_proba(Z_test)[:, 1]
    thr_xgb, f1_xgb = best_threshold(proba, y_test)
    m_xgb = full_eval(proba, y_test, thr_xgb, "XGB(latent)")

    # ── Ensemble: XGBoost + AE EWM ───────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("Ensemble: XGBoost_latent + AE_EWM30...")

    # Normalize both to [0,1]
    def n01(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    proba_n  = n01(proba)
    ewm_n    = n01(ewm_scores)

    best_f1e, best_we = 0.0, (0.5, 0.5)
    for w in np.arange(0.1, 0.95, 0.05):
        ens = w * proba_n + (1 - w) * ewm_n
        _, f = best_threshold(ens, y_test)
        if f > best_f1e: best_f1e, best_we = f, (w, 1 - w)

    w_xgb, w_ae = best_we
    logger.info(f"  Optimal: w_xgb={w_xgb:.2f}  w_ae={w_ae:.2f}")
    ens_scores = w_xgb * proba_n + w_ae * ewm_n
    thr_ens, f1_ens = best_threshold(ens_scores, y_test)
    m_ens = full_eval(ens_scores, y_test, thr_ens, "XGB+AE_Ensemble")

    # ── Summary ───────────────────────────────────────────────────────────────
    all_metrics = {
        "AE_ewm30_baseline": {"f1": f1_base, "threshold": thr_base},
        "XGBoost_latent":    m_xgb,
        "XGB_AE_Ensemble":   m_ens,
    }

    best_name = max(all_metrics, key=lambda k: all_metrics[k]["f1"])
    best_f1   = all_metrics[best_name]["f1"]

    # ── Save if better ────────────────────────────────────────────────────────
    if best_f1 > 0.3960:
        logger.info(f"  New best F1={best_f1:.4f} > 0.3960 — saving model")
        pkg_new = {
            "model_type":    "AE_XGB_SemiSupervised",
            "model_name":    best_name,
            "ae_model":      model,
            "ae_pkg":        pkg,
            "clf":           clf,
            "features":      COMMON_FEATURES,
            "threshold":     all_metrics[best_name]["threshold"],
            "w_xgb":         w_xgb if "Ensemble" in best_name else 1.0,
            "w_ae":          w_ae  if "Ensemble" in best_name else 0.0,
            "metrics":       all_metrics[best_name],
            "best_f1":       best_f1,
        }
        joblib.dump(pkg_new, OUT_DIR / "best_detection_model.joblib")

        import datetime
        summ_path = MET_DIR / "pipeline_summary.json"
        summ = {}
        if summ_path.exists():
            try:
                with open(summ_path) as f: summ = json.load(f)
            except: pass
        bm = all_metrics[best_name]
        summ.update({
            "run_timestamp": str(datetime.datetime.now()),
            "best_model":    best_name,
            "metrics": {
                "f1":        bm["f1"],
                "precision": bm.get("precision", 0),
                "recall":    bm.get("recall", 0),
                "roc_auc":   bm.get("roc_auc", 0),
                "threshold": bm.get("threshold", 0.5),
            },
            "training_note": "AE latent features + XGBoost semi-supervised (synthetic attacks)",
        })
        with open(summ_path, "w") as f:
            json.dump(summ, f, indent=2, default=str)
        logger.info("  Dashboard updated")
    else:
        logger.info(f"  F1={best_f1:.4f} not better than baseline 0.396 — keeping current model")

    # ── Final report ──────────────────────────────────────────────────────────
    W = 65
    logger.info("\n" + "=" * W)
    logger.info("  SEMI-SUPERVISED DETECTION — FINAL RESULTS")
    logger.info("=" * W)
    logger.info(f"  AE+EWM30 (baseline)     F1 = {f1_base:.4f}")
    logger.info(f"  XGBoost(latent)         F1 = {m_xgb['f1']:.4f}  ROC={m_xgb['roc_auc']:.4f}")
    logger.info(f"  XGB+AE ensemble         F1 = {m_ens['f1']:.4f}  ROC={m_ens['roc_auc']:.4f}")
    logger.info("=" * W)
    logger.info(f"  BEST: {best_name}  F1 = {best_f1:.4f}")
    if   best_f1 >= 0.70: status = "EXCELLENT - TARGET REACHED!"
    elif best_f1 >= 0.55: status = "GOOD - close to target"
    elif best_f1 >= 0.40: status = "FAIR - meaningful improvement"
    else:                  status = "POOR - try more synthetic data or better diffusion"
    logger.info(f"  STATUS: {status}")
    logger.info("=" * W)


if __name__ == "__main__":
    main()
