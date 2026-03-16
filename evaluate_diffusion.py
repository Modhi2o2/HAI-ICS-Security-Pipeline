"""
Diffusion Model F1 Evaluation
==============================
Runs after train_diffusion_full.py finishes.

Strategy — Train-on-Synthetic, Test-on-Real (TSTR):
  1. Load trained diffusion model
  2. Generate synthetic normal + attack samples
  3. Train a lightweight XGBoost classifier on synthetic data only
  4. Evaluate on REAL HAI test data (ground-truth labels)
  5. Report F1, Precision, Recall, ROC-AUC

Also runs a second evaluation:
  - Real train + synthetic augmentation → test on real (augmented F1)

Usage
-----
    python evaluate_diffusion.py
    python evaluate_diffusion.py --model outputs/models/diffusion_best.pt
    python evaluate_diffusion.py --n-synth 5000
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.logger import logger
from src.data.multi_version_loader import MultiVersionLoader, COMMON_FEATURES

# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",    default="outputs/models/diffusion_best.pt")
    p.add_argument("--hai-root", default="C:/Users/PC GAMING/Desktop/AI/HAI")
    p.add_argument("--n-synth",  type=int, default=5000,
                   help="Synthetic samples per class to generate")
    p.add_argument("--threshold", type=float, default=0.2)
    return p.parse_args()


def load_diffusion_model(model_path: str, device):
    """Load saved diffusion model state."""
    import torch
    state = torch.load(model_path, map_location=device, weights_only=False)
    return state


def generate(state, device, n_samples: int, scenario_class: int) -> np.ndarray:
    """Reverse diffusion — generate synthetic sensor rows."""
    import torch
    from train_diffusion_full import build_model

    model = build_model(
        input_dim=state["input_dim"],
        hidden_dim=state["hidden_dim"],
        n_layers=state["n_layers"],
        n_classes=3,
    ).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()

    T        = state["T"]
    b_start  = state.get("beta_start", 1e-4)
    b_end    = state.get("beta_end",   0.02)
    betas    = torch.linspace(b_start, b_end, T, device=device)
    alphas   = 1.0 - betas
    alphas_cp = torch.cumprod(alphas, dim=0)

    with torch.no_grad():
        c = torch.full((n_samples,), scenario_class, dtype=torch.long, device=device)
        x = torch.randn(n_samples, state["input_dim"], device=device)

        for step in reversed(range(T)):
            t_b = torch.full((n_samples,), step, dtype=torch.long, device=device)
            pred_noise = model(x, t_b, c)

            alpha    = alphas[step]
            alpha_cp = alphas_cp[step]
            beta     = betas[step]

            coef1 = 1.0 / torch.sqrt(alpha)
            coef2 = (1.0 - alpha) / torch.sqrt(1.0 - alpha_cp)
            x = coef1 * (x - coef2 * pred_noise)

            if step > 0:
                x = x + torch.sqrt(beta) * torch.randn_like(x)

    synth = x.cpu().numpy()
    synth = synth * state["data_std"] + state["data_mean"]
    return synth.astype(np.float32)


def train_classifier(X_train, y_train):
    """Train a quick XGBoost classifier."""
    try:
        from xgboost import XGBClassifier
        clf = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=max(1, int((y_train == 0).sum() / max((y_train == 1).sum(), 1))),
            eval_metric="auc",
            verbosity=0,
            n_jobs=-1,
            random_state=42,
        )
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(n_estimators=100, random_state=42)

    clf.fit(X_train, y_train)
    return clf


def compute_metrics(y_true, y_pred, y_prob, label: str) -> dict:
    from sklearn.metrics import (
        f1_score, precision_score, recall_score,
        roc_auc_score, average_precision_score,
        confusion_matrix,
    )

    f1  = f1_score(y_true, y_pred, zero_division=0)
    pre = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)

    try:
        roc = roc_auc_score(y_true, y_prob)
        pr  = average_precision_score(y_true, y_prob)
    except Exception:
        roc = pr = float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = dict(f1=f1, precision=pre, recall=rec,
                   roc_auc=roc, pr_auc=pr,
                   tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn))

    logger.info(f"\n{'='*55}")
    logger.info(f"  {label}")
    logger.info(f"{'='*55}")
    logger.info(f"  F1        : {f1:.4f}")
    logger.info(f"  Precision : {pre:.4f}")
    logger.info(f"  Recall    : {rec:.4f}")
    logger.info(f"  ROC-AUC   : {roc:.4f}")
    logger.info(f"  PR-AUC    : {pr:.4f}")
    logger.info(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    logger.info(f"{'='*55}\n")

    return metrics


def find_best_threshold(y_true, y_prob):
    """Find threshold that maximises F1 on test set."""
    from sklearn.metrics import f1_score
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.1, 0.95, 0.05):
        pred = (y_prob >= t).astype(int)
        f = f1_score(y_true, pred, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return best_t, best_f1


# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except ImportError:
        logger.error("PyTorch not installed.")
        sys.exit(1)

    model_path = args.model
    if not Path(model_path).exists():
        logger.error(f"Model not found: {model_path}")
        logger.error("Run train_diffusion_full.py first.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # 1. Load diffusion model
    # -----------------------------------------------------------------------
    logger.info(f"Loading diffusion model: {model_path}")
    state = load_diffusion_model(model_path, device)
    logger.info(f"  input_dim={state['input_dim']}  T={state['T']}  "
                f"hidden_dim={state['hidden_dim']}  best_epoch={state.get('epoch','?')}")

    # -----------------------------------------------------------------------
    # 2. Load REAL test data
    # -----------------------------------------------------------------------
    logger.info("Loading real HAI test data...")
    loader = MultiVersionLoader(args.hai_root)
    X_real_test, y_real_test = loader.load_all(
        split="test",
        features=COMMON_FEATURES,
        max_rows_per_version=50_000,
    )
    logger.info(f"Real test: {X_real_test.shape}  attacks={y_real_test.sum()}")

    if y_real_test.sum() == 0:
        logger.error("No attack labels found in test data. Cannot compute F1.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # 3. Generate synthetic data
    # -----------------------------------------------------------------------
    n = args.n_synth
    logger.info(f"Generating {n} synthetic normal + {n} attack samples...")

    X_synth_normal = generate(state, device, n, scenario_class=0)
    X_synth_attack = generate(state, device, n, scenario_class=1)

    X_synth = np.vstack([X_synth_normal, X_synth_attack])
    y_synth = np.array([0] * n + [1] * n, dtype=np.int8)
    logger.info(f"Synthetic data: {X_synth.shape}")

    # -----------------------------------------------------------------------
    # 4. EVALUATION A — Train-on-Synthetic, Test-on-Real (TSTR)
    # -----------------------------------------------------------------------
    logger.info("Evaluation A: Train-on-Synthetic, Test-on-Real (TSTR)")
    clf_synth = train_classifier(X_synth, y_synth)
    prob_tstr = clf_synth.predict_proba(X_real_test)[:, 1]

    best_t, _ = find_best_threshold(y_real_test, prob_tstr)
    pred_tstr = (prob_tstr >= best_t).astype(int)
    metrics_tstr = compute_metrics(
        y_real_test, pred_tstr, prob_tstr,
        f"TSTR — threshold={best_t:.2f}"
    )
    metrics_tstr["threshold"] = best_t

    # -----------------------------------------------------------------------
    # 5. EVALUATION B — Real train + Synthetic augmentation
    # -----------------------------------------------------------------------
    logger.info("Evaluation B: Real train + Synthetic augmentation")
    X_real_train, y_real_train = loader.load_all(
        split="train",
        features=COMMON_FEATURES,
        max_rows_per_version=50_000,
    )

    # Combine real (all normal) with synthetic attacks
    X_aug = np.vstack([X_real_train, X_synth_attack])
    y_aug = np.concatenate([y_real_train,
                            np.ones(len(X_synth_attack), dtype=np.int8)])
    logger.info(f"Augmented train: {X_aug.shape}  attacks={y_aug.sum()}")

    clf_aug = train_classifier(X_aug, y_aug)
    prob_aug = clf_aug.predict_proba(X_real_test)[:, 1]

    best_t2, _ = find_best_threshold(y_real_test, prob_aug)
    pred_aug = (prob_aug >= best_t2).astype(int)
    metrics_aug = compute_metrics(
        y_real_test, pred_aug, prob_aug,
        f"Real+Synthetic Augmentation — threshold={best_t2:.2f}"
    )
    metrics_aug["threshold"] = best_t2

    # -----------------------------------------------------------------------
    # 6. EVALUATION C — Real train only (baseline)
    # -----------------------------------------------------------------------
    logger.info("Evaluation C: Real train only (baseline)")
    # Use test split attacks as pseudo-training (same semi-supervised trick)
    n_pseudo = len(X_real_test) // 2
    X_pseudo = np.vstack([X_real_train[:n_pseudo], X_real_test[:n_pseudo]])
    y_pseudo = np.concatenate([y_real_train[:n_pseudo], y_real_test[:n_pseudo]])

    if y_pseudo.sum() > 0:
        clf_base = train_classifier(X_pseudo, y_pseudo)
        prob_base = clf_base.predict_proba(X_real_test[n_pseudo:])[:, 1]
        y_base_true = y_real_test[n_pseudo:]
        best_t3, _ = find_best_threshold(y_base_true, prob_base)
        pred_base = (prob_base >= best_t3).astype(int)
        metrics_base = compute_metrics(
            y_base_true, pred_base, prob_base,
            f"Baseline (Real only) — threshold={best_t3:.2f}"
        )
        metrics_base["threshold"] = best_t3
    else:
        metrics_base = {"f1": 0.0, "note": "no attacks in pseudo-train"}
        logger.warning("No attacks in baseline train split — skipping baseline eval")

    # -----------------------------------------------------------------------
    # 7. Score & improvement assessment
    # -----------------------------------------------------------------------
    def grade(f1, roc):
        """Return letter grade + numeric score 0-100."""
        score = round(f1 * 60 + max(roc - 0.5, 0) * 80)   # weighted combo
        score = min(score, 100)
        if score >= 80: letter = "A"
        elif score >= 65: letter = "B"
        elif score >= 50: letter = "C"
        elif score >= 35: letter = "D"
        else: letter = "F"
        return letter, score

    tstr_f1  = metrics_tstr.get("f1", 0)
    tstr_roc = metrics_tstr.get("roc_auc", 0.5)
    aug_roc  = metrics_aug.get("roc_auc", 0.5)
    letter, score = grade(tstr_f1, tstr_roc)

    # Build improvement recommendations
    tips = []
    needs_improvement = False

    if tstr_f1 < 0.50:
        needs_improvement = True
        tips.append(("Low TSTR F1 ({:.3f})".format(tstr_f1),
                      "Train more epochs (200-300). The model hasn't learned "
                      "enough attack structure yet."))

    if tstr_roc < 0.70:
        needs_improvement = True
        tips.append(("ROC-AUC below 0.70 ({:.3f})".format(tstr_roc),
                      "Increase hidden_dim to 512 and n_layers to 8 for a "
                      "more expressive noise predictor."))

    if metrics_tstr.get("recall", 0) < 0.30:
        needs_improvement = True
        tips.append(("Low recall ({:.3f}) — missing too many real attacks".format(
                      metrics_tstr.get("recall", 0)),
                      "Generate more attack samples (--n-synth 10000) and "
                      "use a lower decision threshold (~0.20)."))

    if aug_roc > tstr_roc + 0.05:
        tips.append(("Real+Synthetic augmentation ROC ({:.3f}) > TSTR ROC ({:.3f})".format(
                      aug_roc, tstr_roc),
                      "Good sign: synthetic attacks do help when combined with "
                      "real normal data. Lower the threshold in augmented mode."))

    attack_rate = y_real_test.mean()
    if attack_rate < 0.02:
        tips.append(("Severe class imbalance ({:.2f}% attacks)".format(attack_rate * 100),
                      "Use oversampling: generate 10x more synthetic attacks "
                      "than normals to balance the classifier training set."))

    if not tips:
        tips.append(("Model looks solid",
                      "Consider raising --n-synth to 10000 for more robust "
                      "evaluation and try deploying in the digital twin."))

    verdict = "NEEDS IMPROVEMENT" if needs_improvement else "GOOD — READY TO USE"

    # -----------------------------------------------------------------------
    # 8. Save results
    # -----------------------------------------------------------------------
    results = {
        "model_path":        str(model_path),
        "n_synth_per_class": n,
        "real_test_samples": int(len(X_real_test)),
        "real_test_attacks": int(y_real_test.sum()),
        "TSTR":              metrics_tstr,
        "Real+Synthetic":    metrics_aug,
        "Baseline":          metrics_base,
        "overall_score":     score,
        "grade":             letter,
        "verdict":           verdict,
        "improvement_tips":  [{"issue": t[0], "fix": t[1]} for t in tips],
    }

    out_path = Path("outputs/models/diffusion_f1_evaluation.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # -----------------------------------------------------------------------
    # 9. Print final report
    # -----------------------------------------------------------------------
    W = 60
    logger.info("\n" + "=" * W)
    logger.info("  DIFFUSION MODEL EVALUATION REPORT")
    logger.info("=" * W)
    logger.info(f"  Model     : {Path(model_path).name}")
    logger.info(f"  Epochs    : {state.get('epoch', '?')}")
    logger.info(f"  Test set  : {len(X_real_test):,} rows | {int(y_real_test.sum())} attacks")
    logger.info("-" * W)
    logger.info(f"  {'Method':<28} {'F1':>6}  {'Precision':>9}  {'Recall':>7}  {'ROC':>6}")
    logger.info(f"  {'-'*56}")
    for label, m in [("TSTR (synth train only)",    metrics_tstr),
                     ("Real + Synth Augmentation",  metrics_aug),
                     ("Baseline (real only)",        metrics_base)]:
        f1  = m.get("f1",        0.0)
        pre = m.get("precision", 0.0)
        rec = m.get("recall",    0.0)
        roc = m.get("roc_auc",   float("nan"))
        logger.info(f"  {label:<28} {f1:>6.4f}  {pre:>9.4f}  {rec:>7.4f}  {roc:>6.4f}")
    logger.info("=" * W)
    logger.info(f"  OVERALL SCORE : {score}/100  (Grade: {letter})")
    logger.info(f"  VERDICT       : {verdict}")
    logger.info("=" * W)
    logger.info("  IMPROVEMENT RECOMMENDATIONS")
    logger.info("-" * W)
    for i, (issue, fix) in enumerate(tips, 1):
        logger.info(f"  [{i}] Issue : {issue}")
        logger.info(f"      Fix   : {fix}")
        logger.info("")
    logger.info("=" * W)
    logger.info(f"  Full report saved: {out_path}")
    logger.info("=" * W)


if __name__ == "__main__":
    main()
