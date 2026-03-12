#!/usr/bin/env python3
"""
HAI ICS Security Pipeline — Main Training and Evaluation Script

Runs the complete end-to-end pipeline:
1. Data loading
2. EDA (optional)
3. Feature engineering + preprocessing
4. Model training and evaluation
5. Diffusion model training
6. Digital twin initialization
7. Synthetic scenario generation
8. Save all outputs

Usage:
    python run_pipeline.py                    # Full pipeline
    python run_pipeline.py --skip-eda         # Skip EDA
    python run_pipeline.py --skip-diffusion   # Skip diffusion model
    python run_pipeline.py --fast             # Fast mode: limit data size
"""

import os
import sys
import json
import time
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.data_loader import HAIDataLoader, load_config
from src.models.model_trainer import ModelTrainer
from src.digital_twin.digital_twin import DigitalTwin
from src.diffusion.scenario_generator import ScenarioGenerator
from src.utils.logger import setup_logger
from src.utils.visualization import (
    plot_confusion_matrix, plot_roc_pr_curves, plot_feature_importance,
    plot_synthetic_vs_real, ensure_figures_dir
)
from src.utils.metrics import print_metrics_report

logger = setup_logger("hai_pipeline", log_file="outputs/logs/pipeline.log")

# Ensure PyTorch-dependent imports are optional
try:
    from src.diffusion.diffusion_model import HAIDiffusionModel
    HAS_DIFFUSION = True
except ImportError:
    HAS_DIFFUSION = False
    logger.warning("PyTorch not available — diffusion model will be skipped")


def parse_args():
    parser = argparse.ArgumentParser(description="HAI ICS Security Pipeline")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--skip-eda", action="store_true", help="Skip EDA step")
    parser.add_argument("--skip-diffusion", action="store_true", help="Skip diffusion model training")
    parser.add_argument("--skip-lstm", action="store_true", help="Skip LSTM model training")
    parser.add_argument("--fast", action="store_true",
                        help="Fast mode: limit rows for quick validation")
    return parser.parse_args()


def create_output_dirs(config: dict) -> None:
    """Create all required output directories."""
    dirs = [
        "outputs/logs", "outputs/models", "outputs/metrics",
        "outputs/predictions", "outputs/synthetic",
        "reports/figures", "data/processed",
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def run_eda_step(config: dict) -> dict:
    """Run EDA analysis."""
    logger.info("=" * 50)
    logger.info("PHASE 1: EDA & DATA QUALITY ANALYSIS")
    logger.info("=" * 50)

    try:
        from run_eda import run_eda
        return run_eda()
    except Exception as e:
        logger.error(f"EDA failed: {e}")
        return {}


def run_training_step(
    config: dict,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict:
    """Run model training and evaluation."""
    logger.info("=" * 50)
    logger.info("PHASE 2: MODEL TRAINING & EVALUATION")
    logger.info("=" * 50)

    trainer = ModelTrainer(config)
    results = trainer.train_and_evaluate(train_df, test_df)

    # Generate plots
    from numpy import ndarray
    label_col = config["data"]["label_col"]

    for model_name, metrics in results["metrics"].items():
        if "confusion_matrix" in metrics:
            import numpy as np
            cm = np.array(metrics["confusion_matrix"])
            plot_confusion_matrix(cm, model_name=model_name)

    # ROC/PR curves
    if "y_test" in results and "y_prob" in results:
        y_test = results["y_test"]
        # For multi-model curves, we only have the best model probability here
        # In a full pipeline, collect probs from all models
        y_probs_dict = {"Best Model": results["y_prob"]}
        plot_roc_pr_curves(y_test, y_probs_dict)

    # Feature importance
    for model_name, fi_df in results["feature_importances"].items():
        if len(fi_df) > 0:
            plot_feature_importance(fi_df, model_name=model_name)

    return results


def run_diffusion_step(
    config: dict,
    train_df: pd.DataFrame,
    feature_cols: list,
) -> object:
    """Train diffusion model and generate scenarios."""
    logger.info("=" * 50)
    logger.info("PHASE 3: DIFFUSION MODEL TRAINING")
    logger.info("=" * 50)

    if not HAS_DIFFUSION:
        logger.warning("Skipping diffusion — PyTorch not available")
        return None

    label_col = config["data"]["label_col"]

    # Prepare training data (normal samples only for unconditional training)
    # Also include attack samples with labels for conditional generation
    numeric_cols = [c for c in train_df.select_dtypes(include=['number']).columns
                   if c != label_col]

    # Limit features to available ones
    use_cols = [c for c in feature_cols if c in numeric_cols][:100]
    if not use_cols:
        use_cols = numeric_cols[:50]

    X_train = train_df[use_cols].fillna(0).values.astype(np.float32)
    y_train = train_df[label_col].values if label_col in train_df.columns else np.zeros(len(train_df))

    # Limit size for training speed
    max_diffusion_samples = min(50000, len(X_train))
    idx = np.random.choice(len(X_train), size=max_diffusion_samples, replace=False)
    X_diff = X_train[idx]
    y_diff = y_train[idx]

    try:
        diffusion_model = HAIDiffusionModel(config)
        diffusion_model.fit(X_diff, y_diff)
        diffusion_model.save()

        # Generate synthetic scenarios
        scenario_gen = ScenarioGenerator(config, diffusion_model=diffusion_model)
    except Exception as e:
        logger.warning(f"Diffusion training failed: {e}. Using rule-based generator only.")
        scenario_gen = ScenarioGenerator(config, diffusion_model=None)

    # Generate all scenario types
    logger.info("Generating synthetic scenarios...")
    scenarios = scenario_gen.generate_all_scenarios(
        baseline_data=X_train[:10000],
        n_per_scenario=500,
        feature_names=use_cols,
    )

    # Compare real vs synthetic distributions
    if scenarios:
        first_scenario = next(iter(scenarios.values()))
        real_sample = pd.DataFrame(X_train[:1000], columns=use_cols[:X_train.shape[1]])
        synth_sample = pd.DataFrame(first_scenario[:1000, :len(use_cols)], columns=use_cols[:first_scenario.shape[1]])
        plot_synthetic_vs_real(real_sample, synth_sample, cols=use_cols[:8])

    # Save quality metrics
    if scenarios:
        try:
            quality_metrics = {}
            for scenario_name, synth_data in scenarios.items():
                n_feats = min(X_train.shape[1], synth_data.shape[1])
                try:
                    from src.diffusion.diffusion_model import HAIDiffusionModel as DM
                    dm_temp = HAIDiffusionModel(config)
                    dm_temp.data_mean = X_train[:, :n_feats].mean(axis=0)
                    dm_temp.data_std = X_train[:, :n_feats].std(axis=0)
                    quality_metrics[scenario_name] = dm_temp.evaluate_quality(
                        X_train[:5000, :n_feats], synth_data[:5000, :n_feats]
                    )
                except Exception:
                    pass

            with open("outputs/metrics/synthetic_quality.json", "w") as f:
                json.dump(quality_metrics, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Quality metric computation failed: {e}")

    return scenario_gen


def run_digital_twin_step(
    config: dict,
    train_df: pd.DataFrame,
    detection_model,
    feature_cols: list,
    scenario_gen: object = None,
) -> DigitalTwin:
    """Initialize and test digital twin."""
    logger.info("=" * 50)
    logger.info("PHASE 4: DIGITAL TWIN INITIALIZATION")
    logger.info("=" * 50)

    label_col = config["data"]["label_col"]

    # Get normal training data for baseline
    if label_col in train_df.columns:
        normal_train = train_df[train_df[label_col] == 0]
    else:
        normal_train = train_df

    numeric_cols = [c for c in train_df.select_dtypes(include=['number']).columns
                   if c != label_col]
    use_cols = [c for c in feature_cols if c in numeric_cols][:100]
    if not use_cols:
        use_cols = numeric_cols[:50]

    X_normal = normal_train[use_cols].fillna(0).values.astype(np.float32)

    # Initialize digital twin
    twin = DigitalTwin(config, feature_names=use_cols)
    twin.fit_baseline(X_normal)

    if detection_model is not None:
        twin.set_detection_model(detection_model)

    # Run a test batch through the twin
    logger.info("Running test batch through digital twin...")
    test_data = X_normal[:1000]
    batch_results = twin.process_batch(test_data)
    logger.info(f"Digital Twin test batch complete: {len(batch_results)} results")

    # Test scenario injection
    if scenario_gen is not None:
        logger.info("Testing scenario injection...")
        try:
            scenarios_all = scenario_gen.generate_all_scenarios(X_normal[:2000], n_per_scenario=100, feature_names=use_cols)
            for scenario_name, scenario_data in list(scenarios_all.items())[:2]:
                twin.inject_scenario(scenario_name, scenario_data[:, :len(use_cols)])
                twin.stop_scenario()
        except Exception as e:
            logger.warning(f"Scenario injection test failed: {e}")

    # Save twin state
    twin.save_state()

    # Save alert log
    alert_log = twin.get_alert_log()
    if len(alert_log) > 0:
        alert_log.to_csv("outputs/metrics/alert_log.csv", index=False)
        logger.info(f"Alert log saved: {len(alert_log)} alerts")

    logger.info(f"Digital Twin initialized. Health score: {twin.health_score}")
    return twin


def main():
    args = parse_args()

    start_time = time.time()
    logger.info("=" * 60)
    logger.info("  HAI ICS Security Pipeline — Starting")
    logger.info(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # Load config
    config = load_config(args.config)

    # Override max rows for fast mode
    if args.fast:
        logger.info("FAST MODE: Limiting data size")
        config["data"]["max_train_rows"] = 30000
        config["data"]["max_test_rows"] = 10000
        config["models"]["detection"]["xgboost"]["n_estimators"] = 100
        config["diffusion"]["epochs"] = 5

    create_output_dirs(config)

    # ── EDA ──────────────────────────────────────────────────────
    eda_summary = {}
    if not args.skip_eda:
        eda_summary = run_eda_step(config)

    # ── Load Data ─────────────────────────────────────────────────
    logger.info("Loading dataset...")
    loader = HAIDataLoader(config)
    train_df, test_df = loader.load_all()

    logger.info(f"Train: {train_df.shape}, Test: {test_df.shape}")

    # ── Model Training ────────────────────────────────────────────
    training_results = run_training_step(config, train_df, test_df)

    # Extract best model
    best_model = training_results.get("best_model", None)
    best_model_name = training_results.get("best_model_name", "Unknown")

    # Get feature columns used
    feature_cols = list(training_results.get("feature_importances", {}).get(
        best_model_name, pd.DataFrame(columns=["feature"])
    ).get("feature", pd.Series()).values) if best_model_name in training_results.get("feature_importances", {}) else []

    if not feature_cols:
        label_col = config["data"]["label_col"]
        ts_col = config["data"]["timestamp_col"]
        feature_cols = [c for c in train_df.columns
                       if c not in [label_col, ts_col] and train_df[c].dtype in [np.float64, np.float32, np.int64, np.int32]][:50]

    # ── Diffusion Model ───────────────────────────────────────────
    scenario_gen = None
    if not args.skip_diffusion:
        scenario_gen = run_diffusion_step(config, train_df, feature_cols)
    else:
        # Rule-based only
        scenario_gen = ScenarioGenerator(config, diffusion_model=None)

    # ── Digital Twin ──────────────────────────────────────────────
    # Load best model for digital twin
    from src.models.detection_model import AttackDetectionModel
    det_model = AttackDetectionModel(config)

    best_model_path = f"outputs/models/best_model_{best_model_name.lower()}.joblib"
    dt_model = None
    if Path(best_model_path).exists():
        try:
            det_model.load_best_model(best_model_path)
            dt_model = det_model.best_model
        except Exception as e:
            logger.warning(f"Could not load model for digital twin: {e}")

    twin = run_digital_twin_step(config, train_df, dt_model, feature_cols, scenario_gen)

    # ── Final Summary ─────────────────────────────────────────────
    elapsed = time.time() - start_time

    # Compile pipeline summary
    best_metrics = training_results.get("metrics", {}).get(best_model_name, {})

    pipeline_summary = {
        "run_timestamp": str(datetime.now()),
        "elapsed_seconds": round(elapsed, 1),
        "dataset_version": config["data"]["version"],
        "train_shape": list(train_df.shape),
        "test_shape": list(test_df.shape),
        "best_model": best_model_name,
        "metrics": {
            "precision": best_metrics.get("precision", 0),
            "recall": best_metrics.get("recall", 0),
            "f1": best_metrics.get("f1", 0),
            "roc_auc": best_metrics.get("roc_auc", 0),
            "pr_auc": best_metrics.get("pr_auc", 0),
        },
        "all_model_metrics": {
            name: {k: v for k, v in m.items() if isinstance(v, (int, float, str))}
            for name, m in training_results.get("metrics", {}).items()
        },
        "diffusion_enabled": not args.skip_diffusion and HAS_DIFFUSION,
        "digital_twin_health": twin.health_score,
        "digital_twin_alerts": len(twin.alert_log),
        "eda_completed": not args.skip_eda,
    }

    with open("outputs/metrics/pipeline_summary.json", "w") as f:
        json.dump(pipeline_summary, f, indent=2, default=str)

    logger.info("\n" + "=" * 60)
    logger.info("  PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Best Model:    {best_model_name}")
    logger.info(f"  F1 Score:      {best_metrics.get('f1', 0):.4f}")
    logger.info(f"  ROC-AUC:       {best_metrics.get('roc_auc', 0):.4f}")
    logger.info(f"  PR-AUC:        {best_metrics.get('pr_auc', 0):.4f}")
    logger.info(f"  Precision:     {best_metrics.get('precision', 0):.4f}")
    logger.info(f"  Recall:        {best_metrics.get('recall', 0):.4f}")
    logger.info(f"  Elapsed:       {elapsed:.1f}s")
    logger.info("=" * 60)
    logger.info("  Outputs:")
    logger.info("    Models:      outputs/models/")
    logger.info("    Metrics:     outputs/metrics/")
    logger.info("    Predictions: outputs/predictions/")
    logger.info("    Synthetic:   outputs/synthetic/")
    logger.info("    Figures:     reports/figures/")
    logger.info("=" * 60)

    return pipeline_summary


if __name__ == "__main__":
    main()
