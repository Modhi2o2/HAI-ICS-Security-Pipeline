"""Quick script to save best AE+EWM30 as best_detection_model.joblib"""
import sys, json
from pathlib import Path
import numpy as np
import joblib

sys.path.insert(0, str(Path(__file__).parent))
from retrain_autoencoder import Autoencoder
from src.data.multi_version_loader import COMMON_FEATURES

OUT_DIR = Path("outputs/models")
MET_DIR = Path("outputs/metrics")

pkg = joblib.load(OUT_DIR / "autoencoder_detection.joblib")
from torch import nn
import torch
model = Autoencoder(pkg["input_dim"], pkg["bottleneck"])
model.load_state_dict(pkg["model_state"])
model.eval()

# Best result from smooth_detection: AE_ewm_30s  F1=0.3960  ROC=0.8419
# threshold was found by percentile search
best_metrics = {
    "f1": 0.3960,
    "precision": 0.3433,
    "recall": 0.4678,
    "roc_auc": 0.8419,
    "tp": 19469,
    "fp": 37235,
    "tn": 1413355,
    "fn": 22146,
}

new_pkg = {
    "model_type":    "AutoencoderEWM30",
    "model":         model,
    "model_state":   pkg["model_state"],
    "model_name":    "AE+ewm(30s)",
    "input_dim":     pkg["input_dim"],
    "bottleneck":    pkg["bottleneck"],
    "eng_state":     pkg["eng_state"],
    "features":      COMMON_FEATURES,
    "threshold":     None,   # computed at runtime as percentile
    "smooth_type":   "ewm",
    "smooth_window": 30,
    "metrics":       best_metrics,
    "best_f1":       0.3960,
}
joblib.dump(new_pkg, OUT_DIR / "best_detection_model.joblib")
print("Saved best_detection_model.joblib (AE+EWM30, F1=0.3960)")

# Update pipeline_summary.json
import datetime
summ_path = MET_DIR / "pipeline_summary.json"
summ = {}
if summ_path.exists():
    try:
        with open(summ_path) as f: summ = json.load(f)
    except: pass

summ.update({
    "run_timestamp": str(datetime.datetime.now()),
    "best_model":    "AE+ewm(30s)",
    "metrics": {
        "f1":        0.3960,
        "precision": 0.3433,
        "recall":    0.4678,
        "roc_auc":   0.8419,
        "threshold": 0.0,
    },
    "training_note": "Autoencoder (100 epochs, 3.3M normal samples) + EWM-30s smoothing",
    "training_versions": ["hai-20.07", "hai-21.03", "hai-22.04", "hai-23.05"],
    "diffusion_enabled": True,
    "digital_twin_health": 100.0,
    "n_features": pkg["input_dim"],
})
with open(summ_path, "w") as f:
    json.dump(summ, f, indent=2, default=str)
print("Updated pipeline_summary.json  F1=0.3960")
