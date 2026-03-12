"""Evaluation metrics for anomaly/attack detection."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score, f1_score, precision_score, recall_score,
    precision_recall_curve, roc_curve
)


def compute_detection_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    threshold: float = 0.5,
    model_name: str = "model"
) -> Dict[str, Any]:
    """
    Compute comprehensive detection metrics.

    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        y_prob: Predicted probabilities (for AUC metrics)
        threshold: Decision threshold
        model_name: Name for reporting

    Returns:
        Dictionary of metrics
    """
    metrics = {
        "model": model_name,
        "threshold": threshold,
        "n_samples": len(y_true),
        "n_attacks": int(y_true.sum()),
        "n_predicted_attacks": int(y_pred.sum()),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    if y_prob is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))

        # Find optimal threshold by F1
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
        f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
        best_idx = np.argmax(f1_scores[:-1])
        metrics["optimal_threshold"] = float(thresholds[best_idx])
        metrics["optimal_f1"] = float(f1_scores[best_idx])

    # Detection delay (time to first detection after attack onset)
    # Compute per-attack-segment detection delay
    metrics["detection_delay_seconds"] = compute_detection_delay(y_true, y_pred)

    return metrics


def compute_detection_delay(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute average detection delay in seconds (samples at 1 Hz)."""
    delays = []
    in_attack = False
    attack_start = 0
    detected = False

    for i, (true, pred) in enumerate(zip(y_true, y_pred)):
        if true == 1 and not in_attack:
            in_attack = True
            attack_start = i
            detected = False
        elif true == 0 and in_attack:
            in_attack = False
            if not detected:
                delays.append(np.nan)  # missed

        if in_attack and pred == 1 and not detected:
            delays.append(i - attack_start)
            detected = True

    valid_delays = [d for d in delays if not np.isnan(d)]
    return float(np.mean(valid_delays)) if valid_delays else float('nan')


def compute_anomaly_scores_stats(scores: np.ndarray, labels: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Compute statistics about anomaly scores."""
    stats = {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "p50": float(np.percentile(scores, 50)),
        "p90": float(np.percentile(scores, 90)),
        "p95": float(np.percentile(scores, 95)),
        "p99": float(np.percentile(scores, 99)),
    }

    if labels is not None:
        normal_scores = scores[labels == 0]
        attack_scores = scores[labels == 1]
        stats["mean_normal"] = float(np.mean(normal_scores)) if len(normal_scores) > 0 else 0.0
        stats["mean_attack"] = float(np.mean(attack_scores)) if len(attack_scores) > 0 else 0.0
        stats["separation_ratio"] = stats["mean_attack"] / (stats["mean_normal"] + 1e-8)

    return stats


def print_metrics_report(metrics: Dict[str, Any]) -> None:
    """Print a formatted metrics report."""
    print(f"\n{'='*60}")
    print(f"  DETECTION METRICS: {metrics.get('model', 'Unknown')}")
    print(f"{'='*60}")
    print(f"  Samples: {metrics.get('n_samples', 'N/A')}")
    print(f"  True attacks: {metrics.get('n_attacks', 'N/A')}")
    print(f"  Predicted attacks: {metrics.get('n_predicted_attacks', 'N/A')}")
    print(f"  Precision:  {metrics.get('precision', 0):.4f}")
    print(f"  Recall:     {metrics.get('recall', 0):.4f}")
    print(f"  F1 Score:   {metrics.get('f1', 0):.4f}")
    if "roc_auc" in metrics:
        print(f"  ROC-AUC:    {metrics['roc_auc']:.4f}")
    if "pr_auc" in metrics:
        print(f"  PR-AUC:     {metrics['pr_auc']:.4f}")
    if "detection_delay_seconds" in metrics:
        delay = metrics['detection_delay_seconds']
        if not np.isnan(delay):
            print(f"  Avg Detection Delay: {delay:.1f}s")
    print(f"{'='*60}\n")
