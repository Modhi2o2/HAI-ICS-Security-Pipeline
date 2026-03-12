"""Visualization utilities for HAI pipeline."""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

FIGURES_DIR = Path("reports/figures")


def ensure_figures_dir(subdir: str = "") -> Path:
    """Create figures directory if it doesn't exist."""
    path = FIGURES_DIR / subdir if subdir else FIGURES_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_missing_values_heatmap(df: pd.DataFrame, save_path: str = None, max_cols: int = 50) -> str:
    """Plot missing values heatmap."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Missing value counts
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0]

    if len(missing_pct) == 0:
        axes[0].text(0.5, 0.5, 'No missing values!', ha='center', va='center',
                    transform=axes[0].transAxes, fontsize=14, color='green')
    else:
        missing_pct.head(max_cols).plot(kind='barh', ax=axes[0], color='coral')
        axes[0].set_title('Missing Value % by Column')
        axes[0].set_xlabel('Missing %')

    # Missing value heatmap (sample)
    sample = df.sample(min(1000, len(df))).isnull()
    if sample.shape[1] > max_cols:
        sample = sample.iloc[:, :max_cols]
    sns.heatmap(sample.T, cmap='YlOrRd', ax=axes[1], cbar=False, yticklabels=True)
    axes[1].set_title('Missing Value Pattern (sample)')
    axes[1].set_xlabel('Sample index')

    plt.tight_layout()
    save_path = save_path or str(ensure_figures_dir() / "missing_values.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path


def plot_label_distribution(labels: pd.Series, save_path: str = None) -> str:
    """Plot attack label distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    counts = labels.value_counts()
    colors = ['#2ecc71', '#e74c3c']

    # Bar chart
    axes[0].bar(['Normal (0)', 'Attack (1)'], counts.values, color=colors, edgecolor='black')
    axes[0].set_title('Label Distribution')
    axes[0].set_ylabel('Count')
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + counts.max() * 0.01, f'{v:,}\n({v/len(labels)*100:.1f}%)',
                    ha='center', fontweight='bold')

    # Pie chart
    axes[1].pie(counts.values, labels=['Normal', 'Attack'], colors=colors, autopct='%1.2f%%',
               startangle=90, explode=[0, 0.1])
    axes[1].set_title('Attack Ratio')

    plt.suptitle('Attack Label Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = save_path or str(ensure_figures_dir() / "label_distribution.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path


def plot_correlation_heatmap(df: pd.DataFrame, save_path: str = None, max_cols: int = 40) -> str:
    """Plot feature correlation heatmap."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > max_cols:
        # Select top features by variance
        variances = df[numeric_cols].var().sort_values(ascending=False)
        numeric_cols = variances.head(max_cols).index.tolist()

    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(min(20, len(numeric_cols) * 0.5 + 2),
                                    min(18, len(numeric_cols) * 0.4 + 2)))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
               ax=ax, square=True, linewidths=0.5,
               cbar_kws={"shrink": 0.5},
               xticklabels=True, yticklabels=True)
    ax.set_title(f'Feature Correlation Heatmap (top {len(numeric_cols)} features by variance)',
                fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    save_path = save_path or str(ensure_figures_dir() / "correlation_heatmap.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path


def plot_sensor_distributions(df: pd.DataFrame, label_col: str = None,
                               save_path: str = None, max_cols: int = 20) -> str:
    """Plot distributions for each sensor."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if label_col in numeric_cols:
        numeric_cols.remove(label_col)
    numeric_cols = numeric_cols[:max_cols]

    n_cols = 4
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        if label_col is not None and label_col in df.columns:
            for label_val, color, name in [(0, '#2ecc71', 'Normal'), (1, '#e74c3c', 'Attack')]:
                subset = df[df[label_col] == label_val][col].dropna()
                ax.hist(subset, bins=50, alpha=0.6, color=color, label=name, density=True)
            ax.legend(fontsize=7)
        else:
            ax.hist(df[col].dropna(), bins=50, alpha=0.8, color='steelblue')
        ax.set_title(col, fontsize=9, fontweight='bold')
        ax.set_xlabel('')
        ax.tick_params(labelsize=7)

    # Hide unused axes
    for i in range(len(numeric_cols), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('Sensor Distributions (Normal vs Attack)', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_path = save_path or str(ensure_figures_dir() / "sensor_distributions.png")
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    return save_path


def plot_attack_timeline(df: pd.DataFrame, timestamp_col: str, label_col: str,
                          feature_col: str = None, save_path: str = None) -> str:
    """Plot attack events over time."""
    fig, axes = plt.subplots(2, 1, figsize=(18, 8), sharex=True)

    # Plot attack labels as timeline
    ax1 = axes[0]
    ax1.fill_between(range(len(df)), df[label_col], alpha=0.7, color='#e74c3c', label='Attack')
    ax1.set_ylabel('Attack Label', fontsize=11)
    ax1.set_title('Attack Timeline', fontsize=12, fontweight='bold')
    ax1.set_ylim(-0.1, 1.2)
    ax1.legend()

    # Plot one feature signal
    if feature_col is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c != label_col]
        feature_col = numeric_cols[0] if len(numeric_cols) > 0 else None

    ax2 = axes[1]
    if feature_col and feature_col in df.columns:
        ax2.plot(df[feature_col].values, color='steelblue', linewidth=0.5, label=feature_col)
        # Shade attack regions
        attack_mask = df[label_col].values == 1
        ax2.fill_between(range(len(df)), df[feature_col].min(), df[feature_col].max(),
                        where=attack_mask, alpha=0.3, color='#e74c3c', label='Attack period')
        ax2.set_ylabel(feature_col, fontsize=11)
        ax2.set_title(f'{feature_col} Signal with Attack Periods', fontsize=11)
        ax2.legend()

    ax2.set_xlabel('Time (seconds)', fontsize=11)
    plt.tight_layout()
    save_path = save_path or str(ensure_figures_dir() / "attack_timeline.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path


def plot_rolling_statistics(df: pd.DataFrame, cols: List[str],
                             window: int = 300, save_path: str = None) -> str:
    """Plot rolling mean and std for key sensors."""
    n_cols_plot = min(len(cols), 4)
    fig, axes = plt.subplots(n_cols_plot, 2, figsize=(16, n_cols_plot * 3))

    if n_cols_plot == 1:
        axes = axes.reshape(1, -1)

    for i, col in enumerate(cols[:n_cols_plot]):
        series = df[col].rolling(window=window)
        rolling_mean = series.mean()
        rolling_std = series.std()

        axes[i, 0].plot(rolling_mean.values, color='steelblue', linewidth=0.8)
        axes[i, 0].set_title(f'{col} - Rolling Mean (w={window})', fontsize=9)
        axes[i, 0].tick_params(labelsize=7)

        axes[i, 1].plot(rolling_std.values, color='coral', linewidth=0.8)
        axes[i, 1].set_title(f'{col} - Rolling Std (w={window})', fontsize=9)
        axes[i, 1].tick_params(labelsize=7)

    plt.suptitle(f'Rolling Statistics (window={window}s)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    save_path = save_path or str(ensure_figures_dir() / "rolling_statistics.png")
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    return save_path


def plot_confusion_matrix(cm: np.ndarray, model_name: str = "Model",
                           save_path: str = None) -> str:
    """Plot confusion matrix."""
    fig, ax = plt.subplots(figsize=(7, 6))

    labels = ['Normal', 'Attack']
    cm_normalized = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
               xticklabels=labels, yticklabels=labels,
               linewidths=2, linecolor='white')

    # Add percentages
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j + 0.5, i + 0.7, f'({cm_normalized[i,j]:.1%})',
                   ha='center', va='center', fontsize=9, color='gray')

    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)

    plt.tight_layout()
    save_path = save_path or str(ensure_figures_dir() / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path


def plot_roc_pr_curves(y_true: np.ndarray, y_probs: Dict[str, np.ndarray],
                        save_path: str = None) -> str:
    """Plot ROC and PR curves for multiple models."""
    from sklearn.metrics import roc_curve, precision_recall_curve, auc

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

    for (model_name, y_prob), color in zip(y_probs.items(), colors):
        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, color=color, lw=2,
                    label=f'{model_name} (AUC={roc_auc:.3f})')

        # PR
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
        axes[1].plot(recall, precision, color=color, lw=2,
                    label=f'{model_name} (AUC={pr_auc:.3f})')

    # ROC plot formatting
    axes[0].plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    axes[0].set_xlabel('False Positive Rate', fontsize=11)
    axes[0].set_ylabel('True Positive Rate', fontsize=11)
    axes[0].set_title('ROC Curves', fontsize=12, fontweight='bold')
    axes[0].legend(loc='lower right', fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # PR plot formatting
    baseline = y_true.mean()
    axes[1].axhline(y=baseline, color='k', linestyle='--', lw=1, label=f'Baseline ({baseline:.3f})')
    axes[1].set_xlabel('Recall', fontsize=11)
    axes[1].set_ylabel('Precision', fontsize=11)
    axes[1].set_title('Precision-Recall Curves', fontsize=12, fontweight='bold')
    axes[1].legend(loc='upper right', fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = save_path or str(ensure_figures_dir() / "roc_pr_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path


def plot_feature_importance(importance_df: pd.DataFrame, model_name: str = "Model",
                             top_n: int = 30, save_path: str = None) -> str:
    """Plot feature importance bar chart."""
    top_features = importance_df.head(top_n)

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_features)))[::-1]

    ax.barh(range(len(top_features)), top_features['importance'].values, color=colors)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'].values, fontsize=9)
    ax.set_xlabel('Importance Score', fontsize=11)
    ax.set_title(f'Top {top_n} Feature Importances - {model_name}',
                fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    save_path = save_path or str(ensure_figures_dir() / f"feature_importance_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path


def plot_synthetic_vs_real(real_data: pd.DataFrame, synthetic_data: pd.DataFrame,
                            cols: List[str] = None, save_path: str = None) -> str:
    """Compare real vs synthetic data distributions."""
    if cols is None:
        cols = real_data.select_dtypes(include=[np.number]).columns.tolist()[:8]

    n_cols = min(len(cols), 8)
    n_rows = (n_cols + 3) // 4
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, n_rows * 4))
    axes = axes.flatten()

    for i, col in enumerate(cols[:n_cols]):
        if col in real_data.columns and col in synthetic_data.columns:
            axes[i].hist(real_data[col].dropna(), bins=50, alpha=0.6,
                        color='steelblue', label='Real', density=True)
            axes[i].hist(synthetic_data[col].dropna(), bins=50, alpha=0.6,
                        color='coral', label='Synthetic', density=True)
            axes[i].set_title(col, fontsize=9, fontweight='bold')
            axes[i].legend(fontsize=7)
            axes[i].tick_params(labelsize=7)

    for i in range(n_cols, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('Real vs Synthetic Data Distributions', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_path = save_path or str(ensure_figures_dir() / "real_vs_synthetic.png")
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    return save_path
