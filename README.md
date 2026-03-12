# HAI ICS Security Pipeline

A production-grade end-to-end machine learning pipeline for **Industrial Control System (ICS) security** — attack detection, anomaly scoring, digital twin simulation, and synthetic scenario generation using the **HAI (Hardware-In-the-Loop Augmented ICS)** dataset.

## System Overview

| Component | Description |
|-----------|-------------|
| **Dataset** | HAI-23.05 — Boiler/Steam ICS with binary attack labels at 1 Hz |
| **Detection** | XGBoost, LightGBM, Random Forest, BiLSTM Ensemble |
| **Anomaly** | Isolation Forest, LSTM Autoencoder |
| **Diffusion** | DDPM (class-conditioned time-series diffusion) |
| **Digital Twin** | State estimator, health scoring, root cause analysis |
| **Dashboard** | 8-page Streamlit interactive app |

## Quick Start

### 1. Prerequisites

```bash
# Python 3.9+
python --version

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Setup

The dataset is expected at:
```
C:/Users/PC GAMING/Desktop/AI/HAI/hai-23.05/
```

This folder should contain:
```
hai-train1.csv  hai-train2.csv  hai-train3.csv  hai-train4.csv
hai-test1.csv   hai-test2.csv
label-test1.csv label-test2.csv
```

If your dataset is at a different location, update `configs/config.yaml`:
```yaml
paths:
  raw_data: "YOUR_PATH_HERE/hai-23.05"
```

### 3. Run Full Pipeline

```bash
cd C:/Users/PC\ GAMING/Desktop/AI/HAI_Pipeline

# Full pipeline (recommended first run)
python run_pipeline.py

# Fast mode (quick test, limits data size)
python run_pipeline.py --fast

# Skip specific steps
python run_pipeline.py --skip-eda
python run_pipeline.py --skip-diffusion
python run_pipeline.py --skip-eda --skip-diffusion
```

### 4. Run EDA Only

```bash
python run_eda.py
```

### 5. Launch Dashboard

```bash
streamlit run app/streamlit_app.py
```

Open browser at: **http://localhost:8501**

---

## Pipeline Phases

### Phase 1 — EDA (`run_eda.py`)
- Dataset shape, memory, column types
- Missing value analysis
- Label distribution and class imbalance
- Correlation heatmap
- Sensor distribution (Normal vs Attack overlay)
- Attack timeline and rolling statistics
- Attack segment analysis
- Data dictionary generation

**Outputs:**
- `reports/figures/*.png` — all EDA charts
- `reports/eda_summary.json` — structured summary
- `reports/data_dictionary.json` — column descriptions
- `reports/descriptive_stats.csv` — per-column statistics

### Phase 2 — Model Training (`run_pipeline.py`)
- Feature engineering: lag features, rolling stats, derivatives, cross-sensor ratios
- Preprocessing: StandardScaler, forward-fill imputation, IQR outlier clipping
- Time-aware train/val split (no data leakage)
- Models: XGBoost, LightGBM, Random Forest, Ensemble
- Evaluation: F1, ROC-AUC, PR-AUC, confusion matrix, detection delay

**Outputs:**
- `outputs/models/best_model_*.joblib` — best trained model
- `outputs/models/model_*.joblib` — all trained models
- `outputs/models/preprocessor.joblib` — fitted preprocessor
- `outputs/metrics/detection_metrics.json` — all metrics
- `outputs/predictions/test_predictions.csv` — test predictions
- `reports/figures/confusion_matrix_*.png`
- `reports/figures/roc_pr_curves.png`
- `reports/figures/feature_importance_*.png`

### Phase 3 — Diffusion Model
- Lightweight DDPM trained on sensor windows
- Class-conditioned generation (Normal / Cyberattack / Fault)
- 8 rule-based scenario fallbacks (always available)

**Outputs:**
- `outputs/models/diffusion_model.pt`
- `outputs/synthetic/scenario_*.npy` — generated scenarios
- `outputs/metrics/synthetic_quality.json` — quality metrics
- `reports/figures/real_vs_synthetic.png`

### Phase 4 — Digital Twin
- Baseline statistics from normal training data
- Per-sample anomaly detection + health scoring
- Root cause analysis (top deviating sensors + subsystem scores)
- Alert engine with severity levels and cooldown

**Outputs:**
- `outputs/digital_twin_state.json` — state snapshot
- `outputs/metrics/alert_log.csv` — alert history

---

## Project Structure

```
HAI_Pipeline/
├── app/
│   └── streamlit_app.py          # 8-page Streamlit dashboard
├── configs/
│   └── config.yaml               # Pipeline configuration
├── data/
│   ├── raw/                      # (symlink to HAI dataset)
│   └── processed/                # Preprocessed data (auto-generated)
├── notebooks/                    # Jupyter exploration notebooks
├── outputs/
│   ├── models/                   # Trained model files
│   ├── metrics/                  # JSON metrics and logs
│   ├── predictions/              # Test predictions CSV
│   └── synthetic/                # Generated scenario arrays
├── reports/
│   ├── figures/                  # All EDA and evaluation plots
│   ├── FINAL_ANALYSIS_REPORT.md  # Complete methodology report
│   ├── eda_summary.json          # EDA findings
│   ├── data_dictionary.json      # Column descriptions
│   └── descriptive_stats.csv     # Descriptive statistics
├── src/
│   ├── data/
│   │   ├── data_loader.py        # HAI multi-file loader + label alignment
│   │   └── schema.py             # Column taxonomy, data dictionary
│   ├── features/
│   │   ├── preprocessing.py      # Stateful scaler/imputer/outlier handler
│   │   └── feature_engineering.py # Lag, rolling, derivative, cross-sensor features
│   ├── models/
│   │   ├── detection_model.py    # XGBoost/LightGBM/RF/Ensemble trainer
│   │   ├── anomaly_detection.py  # Isolation Forest + LSTM Autoencoder
│   │   ├── lstm_model.py         # BiLSTM with attention for sequence detection
│   │   └── model_trainer.py      # Training orchestrator
│   ├── diffusion/
│   │   ├── diffusion_model.py    # DDPM implementation (PyTorch)
│   │   └── scenario_generator.py # Rule-based + diffusion scenario generator
│   ├── digital_twin/
│   │   └── digital_twin.py       # Digital twin with health scoring + root cause
│   └── utils/
│       ├── logger.py             # Centralized logging
│       ├── metrics.py            # Detection metrics suite
│       └── visualization.py      # Matplotlib/seaborn plotting utilities
├── run_eda.py                    # EDA entry point
├── run_pipeline.py               # Main pipeline entry point
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## Configuration

All settings are in `configs/config.yaml`. Key sections:

```yaml
data:
  version: "hai-23.05"
  max_train_rows: 200000    # Set to null for full dataset
  max_test_rows: 50000

preprocessing:
  window_size: 60           # 60-second windows
  scaler: "standard"        # standard, minmax, robust
  lag_features: [1, 5, 10, 30, 60]
  rolling_windows: [10, 30, 60, 300]

models:
  detection:
    xgboost:
      n_estimators: 300
      scale_pos_weight: 20  # handles 20:1 class imbalance

diffusion:
  timesteps: 200
  epochs: 50
```

---

## Dashboard Pages

| Page | Description |
|------|-------------|
| **Overview** | KPIs, dataset summary, model comparison table |
| **Data Explorer** | Interactive time-series viewer with attack highlighting |
| **Detection** | Upload CSV → get attack predictions with confidence scores |
| **Digital Twin** | Real-time health score, deviating sensors, subsystem status |
| **Scenario Simulation** | Choose scenario type + intensity → visualize system response |
| **Synthetic Generation** | DDPM-generated or rule-based attack/fault scenarios |
| **Explainability** | Feature importances + SHAP plots |
| **Alerts** | Alert feed with severity and recommended actions |

---

## Dataset

**HAI (Hardware-In-the-Loop Augmented ICS Security Dataset)**

| Version | Columns | Files | Notes |
|---------|---------|-------|-------|
| hai-20.07 | 64 | 4 | Original; semicolon-delimited |
| hai-21.03 | 84 | 8 | Expanded; 5 test scenarios |
| hai-22.04 | 88 | 10 | Multi-attack summaries |
| **hai-23.05** | **87** | **6 + 2 labels** | **← Used here** |
| haiend-23.05 | 226 | 6 + 2 labels | Extended DCS features |

**System:** Simulated petrochemical boiler with pressure, temperature, flow, and level control loops.

**Attacks:** Binary (0=Normal, 1=Attack), ~3-5% prevalence, 14-40 attack events per test set.

---

## Key Assumptions

1. **Dataset version**: HAI-23.05 used (most recent standard version)
2. **Training data**: All-normal operation (attack labels only in test files)
3. **Supervised learning**: Binary classification (attack label available in test)
4. **Temporal integrity**: Chronological split only — no time-window shuffling
5. **Class imbalance**: ~20:1 ratio → addressed via scale_pos_weight and class_weight
6. **Sampling rate**: 1 Hz throughout (1 second intervals)
7. **Feature engineering**: Applied after split to prevent leakage

---

## Troubleshooting

**"File not found" error:**
- Verify dataset path in `configs/config.yaml`
- Check that HAI-23.05 CSV files exist at the specified path

**Out of memory:**
- Reduce `max_train_rows` in config.yaml (try 50000)
- Use `python run_pipeline.py --fast`

**PyTorch not installed:**
- LSTM and diffusion models will be skipped automatically
- Core XGBoost/LightGBM pipeline still works without PyTorch
- Install: `pip install torch torchvision`

**Streamlit won't start:**
- `pip install streamlit plotly`
- Run from project root: `streamlit run app/streamlit_app.py`

---

## Hardware Requirements

| Mode | RAM | GPU | Time (approx.) |
|------|-----|-----|------|
| Fast (`--fast`) | 4 GB | Not required | ~5 min |
| Full (no LSTM) | 8 GB | Not required | ~20 min |
| Full (with LSTM+Diffusion) | 16 GB | Recommended | ~60 min |

---

*HAI ICS Security Pipeline — Built for HAI Dataset v23.05 (Boiler/Steam Control System)*
