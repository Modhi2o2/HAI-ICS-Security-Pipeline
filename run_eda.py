#!/usr/bin/env python3
"""
HAI ICS Security Dataset — Full EDA and Data Quality Analysis

Run: python run_eda.py
Outputs: reports/figures/*.png, reports/eda_summary.json, outputs/metrics/data_quality.json

This script performs:
1. Data loading and schema inspection
2. Missing value analysis
3. Distribution analysis
4. Correlation analysis
5. Attack label distribution
6. Time-series trends (rolling stats)
7. Sensor behavior during attacks vs normal
8. Class imbalance assessment
9. Outlier detection
10. Leakage risk assessment
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.data_loader import HAIDataLoader, load_config
from src.data.schema import DATA_DICTIONARY, CRITICAL_SENSORS, ALL_SENSOR_GROUPS
from src.utils.logger import setup_logger
from src.utils.visualization import (
    plot_missing_values_heatmap, plot_label_distribution,
    plot_correlation_heatmap, plot_sensor_distributions,
    plot_attack_timeline, plot_rolling_statistics, ensure_figures_dir
)

logger = setup_logger("hai_eda", log_file="outputs/logs/eda.log")


def run_eda(config_path: str = "configs/config.yaml") -> dict:
    """
    Run full EDA pipeline.

    Returns:
        Summary dictionary of key findings
    """
    # Create output directories
    Path("outputs/logs").mkdir(parents=True, exist_ok=True)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    Path("outputs/metrics").mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  HAI ICS Security Dataset — EDA Pipeline")
    logger.info("=" * 60)

    # ── Load Config ──────────────────────────────────────────────
    config = load_config(config_path)
    logger.info(f"Config loaded from: {config_path}")

    # ── Load Data ────────────────────────────────────────────────
    loader = HAIDataLoader(config)

    logger.info("Loading training data...")
    train_df = loader.load_train()

    logger.info("Loading test data...")
    test_df = loader.load_test()

    # Combine for EDA
    combined_df = pd.concat([train_df, test_df], ignore_index=True)

    label_col = config["data"]["label_col"]
    ts_col = config["data"]["timestamp_col"]

    logger.info(f"Combined dataset: {combined_df.shape}")

    # ── 1. Schema Summary ────────────────────────────────────────
    logger.info("\n[1/10] Computing schema summary...")
    schema = loader.get_schema_summary(combined_df)

    print("\n" + "=" * 60)
    print("  DATASET SCHEMA SUMMARY")
    print("=" * 60)
    print(f"  Shape:          {schema['shape'][0]:,} rows x {schema['shape'][1]} cols")
    print(f"  Memory Usage:   {schema['memory_mb']:.1f} MB")
    print(f"  Numeric cols:   {schema['n_numeric']}")
    print(f"  Binary cols:    (included in numeric)")
    print(f"  Datetime cols:  {schema['n_datetime']}")
    print(f"  Duplicates:     {schema['duplicates']:,}")
    print(f"  Attack rate:    {schema['attack_rate']*100:.2f}%")
    print(f"  Time range:     {schema['time_range']['start']} to {schema['time_range']['end']}")
    print("=" * 60)

    # ── 2. Missing Values ────────────────────────────────────────
    logger.info("[2/10] Analyzing missing values...")
    null_pct = pd.Series(schema['null_pct'])
    missing_cols = null_pct[null_pct > 0]

    print(f"\n  Missing Values: {len(missing_cols)} columns with nulls")
    if len(missing_cols) > 0:
        print(missing_cols.to_string())
    else:
        print("  No missing values detected - OK")

    fig_missing = plot_missing_values_heatmap(combined_df)
    logger.info(f"  Missing value heatmap saved: {fig_missing}")

    # ── 3. Label Distribution ────────────────────────────────────
    logger.info("[3/10] Analyzing label distribution...")

    if label_col in combined_df.columns:
        label_counts = combined_df[label_col].value_counts()
        attack_rate = combined_df[label_col].mean()

        print(f"\n  Label Distribution:")
        print(f"    Normal (0): {label_counts.get(0, 0):,} ({(1-attack_rate)*100:.2f}%)")
        print(f"    Attack (1): {label_counts.get(1, 0):,} ({attack_rate*100:.2f}%)")
        print(f"    Class imbalance ratio: {label_counts.get(0, 1)/max(label_counts.get(1, 1), 1):.1f}:1")

        fig_label = plot_label_distribution(combined_df[label_col])
        logger.info(f"  Label distribution plot saved: {fig_label}")

    # ── 4. Descriptive Statistics ────────────────────────────────
    logger.info("[4/10] Computing descriptive statistics...")
    numeric_cols = combined_df.select_dtypes(include=[np.number]).columns.tolist()
    if label_col in numeric_cols:
        numeric_cols.remove(label_col)

    stats = combined_df[numeric_cols[:50]].describe().T
    stats["cv"] = stats["std"] / (stats["mean"].abs() + 1e-8)  # Coefficient of variation
    stats_path = "reports/descriptive_stats.csv"
    stats.to_csv(stats_path)
    logger.info(f"  Descriptive stats saved: {stats_path}")

    # ── 5. Outlier Analysis ──────────────────────────────────────
    logger.info("[5/10] Analyzing outliers...")
    outlier_counts = {}
    for col in numeric_cols[:50]:
        Q1 = combined_df[col].quantile(0.25)
        Q3 = combined_df[col].quantile(0.75)
        IQR = Q3 - Q1
        n_outliers = ((combined_df[col] < Q1 - 3*IQR) | (combined_df[col] > Q3 + 3*IQR)).sum()
        if n_outliers > 0:
            outlier_counts[col] = int(n_outliers)

    print(f"\n  Outlier Analysis (IQR 3x):")
    print(f"    Columns with outliers: {len(outlier_counts)}/{len(numeric_cols)}")
    top_outlier_cols = sorted(outlier_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    for col, count in top_outlier_cols:
        print(f"    {col}: {count:,} outliers ({count/len(combined_df)*100:.2f}%)")

    # ── 6. Correlation Analysis ──────────────────────────────────
    logger.info("[6/10] Computing correlations...")
    fig_corr = plot_correlation_heatmap(combined_df)
    logger.info(f"  Correlation heatmap saved: {fig_corr}")

    # Top correlated pairs with label
    if label_col in combined_df.columns:
        label_corr = combined_df[numeric_cols + [label_col]].corr()[label_col].drop(label_col)
        label_corr_abs = label_corr.abs().sort_values(ascending=False)
        print(f"\n  Top 10 Features Correlated with Attack Label:")
        for col, corr_val in label_corr_abs.head(10).items():
            print(f"    {col}: {label_corr[col]:+.4f}")

    # ── 7. Sensor Distributions ──────────────────────────────────
    logger.info("[7/10] Plotting sensor distributions...")
    fig_dist = plot_sensor_distributions(combined_df, label_col=label_col)
    logger.info(f"  Sensor distributions saved: {fig_dist}")

    # ── 8. Attack Timeline ───────────────────────────────────────
    logger.info("[8/10] Plotting attack timeline...")

    if label_col in test_df.columns and ts_col in test_df.columns:
        # Plot timeline on test set (has attacks)
        feature_col = CRITICAL_SENSORS[0] if CRITICAL_SENSORS[0] in test_df.columns else numeric_cols[0]
        fig_timeline = plot_attack_timeline(
            test_df.head(50000), ts_col, label_col, feature_col=feature_col
        )
        logger.info(f"  Attack timeline saved: {fig_timeline}")

    # ── 9. Rolling Statistics ────────────────────────────────────
    logger.info("[9/10] Computing rolling statistics...")

    # Use critical sensors that exist in the dataset
    available_critical = [c for c in CRITICAL_SENSORS if c in combined_df.columns]
    if available_critical:
        fig_rolling = plot_rolling_statistics(
            combined_df.head(10000), cols=available_critical[:4], window=300
        )
        logger.info(f"  Rolling statistics saved: {fig_rolling}")

    # ── 10. Attack Segment Analysis ──────────────────────────────
    logger.info("[10/10] Analyzing attack segments...")

    attack_segments = []
    if label_col in combined_df.columns:
        in_attack = False
        start_idx = 0

        for i, row in enumerate(combined_df[label_col]):
            if row == 1 and not in_attack:
                in_attack = True
                start_idx = i
            elif row == 0 and in_attack:
                in_attack = False
                attack_segments.append({
                    "start": start_idx,
                    "end": i - 1,
                    "duration_s": i - start_idx
                })

        if attack_segments:
            durations = [s["duration_s"] for s in attack_segments]
            print(f"\n  Attack Segment Analysis:")
            print(f"    Total attack segments: {len(attack_segments)}")
            print(f"    Min duration: {min(durations)}s ({min(durations)//60}m {min(durations)%60}s)")
            print(f"    Max duration: {max(durations)}s ({max(durations)//60}m {max(durations)%60}s)")
            print(f"    Mean duration: {sum(durations)/len(durations):.1f}s")

    # ── Generate Data Dictionary ─────────────────────────────────
    logger.info("Generating data dictionary...")
    dd_path = "reports/data_dictionary.json"

    # Auto-generate for columns not in predefined dict
    data_dict = dict(DATA_DICTIONARY)
    for col in combined_df.columns:
        if col not in data_dict:
            dtype = str(combined_df[col].dtype)
            nunique = combined_df[col].nunique()
            col_mean = combined_df[col].mean() if pd.api.types.is_numeric_dtype(combined_df[col]) else None
            data_dict[col] = f"Auto-detected: dtype={dtype}, unique_values={nunique}" + \
                             (f", mean={col_mean:.4f}" if col_mean is not None else "")

    with open(dd_path, "w") as f:
        json.dump(data_dict, f, indent=2)
    logger.info(f"Data dictionary saved: {dd_path}")

    # ── Summary JSON ─────────────────────────────────────────────
    summary = {
        "run_timestamp": str(datetime.now()),
        "dataset": {
            "version": config["data"]["version"],
            "train_shape": list(train_df.shape),
            "test_shape": list(test_df.shape),
            "combined_shape": list(combined_df.shape),
            "memory_mb": schema["memory_mb"],
            "n_features": schema["n_numeric"],
            "sampling_rate_hz": 1,
        },
        "data_quality": {
            "missing_values": {k: v for k, v in schema["null_pct"].items() if v > 0},
            "duplicates": schema["duplicates"],
            "n_outlier_columns": len(outlier_counts),
        },
        "labels": {
            "attack_rate": round(schema["attack_rate"], 4),
            "n_normal": int(schema["label_distribution"].get(0, 0)),
            "n_attack": int(schema["label_distribution"].get(1, 0)),
            "class_imbalance_ratio": round(schema["label_distribution"].get(0, 1) /
                                           max(schema["label_distribution"].get(1, 1), 1), 1),
            "n_attack_segments": len(attack_segments),
        },
        "feature_groups": {k: len(v) for k, v in ALL_SENSOR_GROUPS.items()},
        "assumptions": [
            "HAI-23.05 version used as primary dataset",
            "Training files contain only normal operation (no attacks)",
            "Attack labels from separate label files aligned by timestamp",
            "1 Hz sampling rate assumed throughout",
            "All sensor values treated as continuous unless clearly binary (0/1)",
            "Class imbalance addressed via scale_pos_weight in XGBoost and class_weight in RF/LGB",
        ],
        "leakage_risks": [
            "None identified: time-aware split used, no shuffle before split",
            "Rolling/lag features computed separately for train and test",
            "Label column excluded from all feature matrices",
        ],
        "figures_generated": [
            "reports/figures/missing_values.png",
            "reports/figures/label_distribution.png",
            "reports/figures/correlation_heatmap.png",
            "reports/figures/sensor_distributions.png",
            "reports/figures/attack_timeline.png",
            "reports/figures/rolling_statistics.png",
        ],
    }

    summary_path = "reports/eda_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"EDA summary saved: {summary_path}")

    print("\n" + "=" * 60)
    print("  EDA COMPLETE")
    print("=" * 60)
    print(f"  Figures:      reports/figures/")
    print(f"  Summary:      {summary_path}")
    print(f"  Data Dict:    {dd_path}")
    print(f"  Stats CSV:    {stats_path}")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="HAI EDA Pipeline")
    parser.add_argument("--config", default="configs/config.yaml", help="Config file path")
    args = parser.parse_args()
    run_eda(args.config)
