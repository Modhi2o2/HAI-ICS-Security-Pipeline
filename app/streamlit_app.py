#!/usr/bin/env python3
"""
HAI ICS Security Pipeline — Streamlit Dashboard

Multipage interactive dashboard for:
- Dataset overview and KPIs
- Data exploration and filtering
- Attack/anomaly detection
- Digital twin simulation
- Scenario generation and what-if analysis
- Diffusion model synthetic generation
- Explainability (SHAP / feature contributions)
- Alert monitoring

Run: streamlit run app/streamlit_app.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import warnings
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime, timedelta
import streamlit as st

warnings.filterwarnings('ignore')

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HAI ICS Security Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f, #2d6a9f);
        padding: 15px; border-radius: 10px; color: white;
        text-align: center; margin: 5px;
    }
    .metric-value { font-size: 2em; font-weight: bold; }
    .metric-label { font-size: 0.85em; opacity: 0.9; }
    .alert-high { background: #ff4444; color: white; padding: 8px; border-radius: 5px; margin: 3px 0; }
    .alert-medium { background: #ff8800; color: white; padding: 8px; border-radius: 5px; margin: 3px 0; }
    .alert-low { background: #ffcc00; color: black; padding: 8px; border-radius: 5px; margin: 3px 0; }
    .health-good { color: #00cc44; font-weight: bold; }
    .health-warning { color: #ff8800; font-weight: bold; }
    .health-critical { color: #ff4444; font-weight: bold; }
    .sidebar .sidebar-content { background: #0e1117; }
    h1, h2, h3 { color: #4fc3f7; }
</style>
""", unsafe_allow_html=True)

# ── Utility Functions ─────────────────────────────────────────────────────────

@st.cache_resource
def load_config():
    """Load pipeline configuration."""
    config_path = ROOT / "configs" / "config.yaml"
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}

@st.cache_data(ttl=300)
def load_sample_data(max_rows: int = 20000) -> pd.DataFrame:
    """Load sample of the HAI dataset for dashboard display."""
    config = load_config()
    if not config:
        return pd.DataFrame()

    try:
        sys.path.insert(0, str(ROOT))
        from src.data.data_loader import HAIDataLoader
        loader = HAIDataLoader(config)

        # Try loading test data (has labels)
        try:
            config_copy = dict(config)
            config_copy["data"] = dict(config["data"])
            config_copy["data"]["max_test_rows"] = max_rows
            loader2 = HAIDataLoader(config_copy)
            df = loader2.load_test()
            return df
        except Exception:
            return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()

@st.cache_resource
def load_trained_model():
    """Load trained detection model."""
    try:
        import joblib
        model_paths = list((ROOT / "outputs" / "models").glob("best_model_*.joblib"))
        if model_paths:
            state = joblib.load(model_paths[0])
            return state.get("model"), state.get("model_name", "Unknown")
    except Exception:
        pass
    return None, None

@st.cache_resource
def load_preprocessor():
    """Load fitted preprocessor."""
    try:
        import joblib
        prep_path = ROOT / "outputs" / "models" / "preprocessor.joblib"
        if prep_path.exists():
            return joblib.load(prep_path)
    except Exception:
        pass
    return None

@st.cache_resource
def load_haiend_pkg():
    """Load haiend LSTM-AE package (best model, F1=0.6886)."""
    try:
        import joblib
        pkg_path = ROOT / "outputs" / "models" / "haiend_lstm_detection.joblib"
        if not pkg_path.exists():
            return None
        from train_haiend_lstm import LSTMAutoencoder as _LSTM
        sys.modules["__main__"].LSTMAutoencoder = _LSTM
        pkg = joblib.load(pkg_path)
        pkg["model"].eval()
        return pkg
    except Exception as e:
        return None

def score_with_haiend_lstm(X_raw: np.ndarray, pkg: dict,
                            chunk: int = 512) -> np.ndarray:
    """
    Run haiend LSTM-AE on raw (unnormalized) sensor data.
    Returns per-timestep MSE scores.
    X_raw: (T, N_features) float32 — must have same column order as training data
    """
    import torch
    model = pkg["model"]
    mean  = pkg["data_mean"].astype(np.float32)
    std   = pkg["data_std"].astype(np.float32)
    W     = int(pkg["window"])
    N     = int(pkg["n_features"])
    T     = len(X_raw)

    # Align columns — pad or trim to N features
    if X_raw.shape[1] < N:
        pad   = np.zeros((T, N - X_raw.shape[1]), dtype=np.float32)
        X_raw = np.concatenate([X_raw, pad], axis=1)
    X_raw = X_raw[:, :N].astype(np.float32)

    X_norm = (X_raw - mean) / std
    X_pad  = np.concatenate([np.zeros((W - 1, N), dtype=np.float32), X_norm], axis=0)

    scores = np.zeros(T, dtype=np.float32)
    model.eval()
    for start in range(0, T, chunk):
        end   = min(start + chunk, T)
        size  = end - start
        batch = np.stack([X_pad[start + i: start + i + W] for i in range(size)])
        bt    = torch.from_numpy(batch)
        with torch.no_grad():
            scores[start:end] = model.reconstruction_error(bt)
    return scores

def load_metrics() -> dict:
    """Load training metrics if available."""
    metrics_path = ROOT / "outputs" / "metrics" / "detection_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return {}

def load_pipeline_summary() -> dict:
    """Load pipeline run summary."""
    summary_path = ROOT / "outputs" / "metrics" / "pipeline_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            return json.load(f)
    return {}

def load_eda_summary() -> dict:
    """Load EDA summary."""
    eda_path = ROOT / "reports" / "eda_summary.json"
    if eda_path.exists():
        with open(eda_path) as f:
            return json.load(f)
    return {}

def health_color(score: float) -> str:
    if score >= 80:
        return "health-good"
    elif score >= 50:
        return "health-warning"
    return "health-critical"

def detect_anomalies(df: pd.DataFrame, model, preprocessor_state: dict) -> pd.DataFrame:
    """Run anomaly detection on uploaded dataframe."""
    try:
        from sklearn.preprocessing import StandardScaler

        config = load_config()
        label_col = config.get("data", {}).get("label_col", "Attack")
        ts_col = config.get("data", {}).get("timestamp_col", "timestamp")

        feature_cols = [c for c in df.columns if c not in [label_col, ts_col]
                       and df[c].dtype in [np.float64, np.float32, np.int64, np.int32, 'float64', 'int64']]

        if preprocessor_state:
            scaler = preprocessor_state.get("scaler")
            stored_cols = preprocessor_state.get("numeric_cols", feature_cols)
            available = [c for c in stored_cols if c in df.columns]
            if available and scaler is not None:
                X = df[available].fillna(0).values
                X_scaled = scaler.transform(X)
            else:
                X_scaled = df[feature_cols[:50]].fillna(0).values
        else:
            X_scaled = df[feature_cols[:50]].fillna(0).values

        probs = model.predict_proba(X_scaled)[:, 1]
        preds = (probs >= 0.5).astype(int)

        result_df = df.copy()
        result_df["anomaly_score"] = probs
        result_df["prediction"] = preds
        result_df["status"] = result_df["prediction"].map({0: "Normal", 1: "ATTACK"})
        return result_df

    except Exception as e:
        st.error(f"Detection error: {str(e)}")
        return df

def generate_scenario_data(scenario_type: str, intensity: float, n_samples: int,
                            baseline_data: np.ndarray, feature_names: list) -> np.ndarray:
    """Generate synthetic scenario using rule-based generator."""
    try:
        sys.path.insert(0, str(ROOT))
        config = load_config()
        from src.diffusion.scenario_generator import ScenarioGenerator
        gen = ScenarioGenerator(config, diffusion_model=None)
        return gen.generate_rule_based(
            baseline_data, scenario_type=scenario_type,
            intensity=intensity, n_samples=n_samples,
            feature_names=feature_names
        )
    except Exception as e:
        st.error(f"Scenario generation failed: {str(e)}")
        return baseline_data[:n_samples]


# ── Sidebar Navigation ────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("# 🏭 HAI ICS Security")
        st.markdown("---")

        page = st.radio(
            "Navigation",
            [
                "📊 Overview",
                "🔍 Data Explorer",
                "⚠️ Detection",
                "🤖 Digital Twin",
                "🏆 Model Arena",
                "🎭 Scenario Simulation",
                "✨ Synthetic Generation",
                "🔎 Explainability",
                "🚨 Alerts",
            ],
            label_visibility="collapsed"
        )

        st.markdown("---")
        st.markdown("### System Status")

        summary = load_pipeline_summary()
        if summary:
            f1 = summary.get("metrics", {}).get("f1", 0)
            health = summary.get("digital_twin_health", 100)

            if f1 > 0.8:
                st.success(f"✅ Model F1: {f1:.3f}")
            elif f1 > 0.6:
                st.warning(f"⚠️ Model F1: {f1:.3f}")
            else:
                st.error(f"❌ Model F1: {f1:.3f}")

            health_emoji = "🟢" if health >= 80 else ("🟡" if health >= 50 else "🔴")
            st.metric("Health Score", f"{health_emoji} {health:.1f}")
        else:
            st.info("Pipeline not yet run.\nRun `python run_pipeline.py` first.")

        st.markdown("---")
        st.markdown("### Dataset")
        st.caption("HAI ICS Security v23.05")
        st.caption("Boiler/Steam Control System")
        st.caption("1 Hz sampling, binary attack labels")

    return page.split(" ", 1)[1]  # Remove emoji


# ── Page Renderers ────────────────────────────────────────────────────────────

def page_overview():
    st.title("📊 HAI ICS Security Pipeline — Overview")

    summary = load_pipeline_summary()
    eda = load_eda_summary()
    metrics = load_metrics()

    # ── KPI Cards ─────────────────────────────────────────────
    st.subheader("Key Performance Indicators")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        f1 = summary.get("metrics", {}).get("f1", 0)
        st.metric("Best F1 Score", f"{f1:.3f}", delta="attack detection")
    with col2:
        auc = summary.get("metrics", {}).get("roc_auc", 0)
        st.metric("ROC-AUC", f"{auc:.3f}", delta="classification quality")
    with col3:
        pr_auc = summary.get("metrics", {}).get("pr_auc", 0)
        st.metric("PR-AUC", f"{pr_auc:.3f}", delta="vs random baseline")
    with col4:
        n_attacks = eda.get("labels", {}).get("n_attack", "N/A")
        attack_rate = eda.get("labels", {}).get("attack_rate", 0)
        st.metric("Attack Rate", f"{attack_rate*100:.1f}%", delta=f"{n_attacks} samples")
    with col5:
        health = summary.get("digital_twin_health", 100)
        st.metric("DT Health", f"{health:.0f}/100", delta="digital twin")

    st.divider()

    # ── Dataset Summary ───────────────────────────────────────
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Dataset Summary")
        if eda:
            ds = eda.get("dataset", {})
            df_info = pd.DataFrame([
                ["Dataset Version", ds.get("version", "hai-23.05")],
                ["Training Samples", f"{ds.get('train_shape', [0])[0]:,}"],
                ["Test Samples", f"{ds.get('test_shape', [0])[0]:,}"],
                ["Features", ds.get("n_features", "N/A")],
                ["Memory Usage", f"{ds.get('memory_mb', 0):.0f} MB"],
                ["Sampling Rate", "1 Hz (1 second)"],
                ["Domain", "ICS Boiler/Steam System"],
                ["Attack Segments", eda.get("labels", {}).get("n_attack_segments", "N/A")],
                ["Class Imbalance", f"{eda.get('labels', {}).get('class_imbalance_ratio', 0):.0f}:1"],
            ], columns=["Property", "Value"])
            st.dataframe(df_info, hide_index=True, use_container_width=True)
        else:
            st.info("Run `python run_eda.py` to see dataset summary.")

    with col_right:
        st.subheader("Model Performance Summary")
        if metrics:
            rows = []
            for model_name, m in metrics.items():
                rows.append({
                    "Model": model_name,
                    "Precision": f"{m.get('precision', 0):.4f}",
                    "Recall": f"{m.get('recall', 0):.4f}",
                    "F1": f"{m.get('f1', 0):.4f}",
                    "ROC-AUC": f"{m.get('roc_auc', 0):.4f}",
                    "PR-AUC": f"{m.get('pr_auc', 0):.4f}",
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        else:
            st.info("Run `python run_pipeline.py` to see model metrics.")

    st.divider()

    # ── Pipeline Architecture ─────────────────────────────────
    st.subheader("Pipeline Architecture")

    arch_col1, arch_col2, arch_col3, arch_col4 = st.columns(4)

    with arch_col1:
        st.markdown("**📥 Data Layer**")
        st.markdown("""
        - HAI-23.05 dataset
        - Multi-file merge
        - Label alignment
        - 1 Hz time series
        """)

    with arch_col2:
        st.markdown("**⚙️ Feature Layer**")
        st.markdown("""
        - Lag features
        - Rolling stats
        - Derivatives
        - Sensor deviations
        - Cross-sensor ratios
        """)

    with arch_col3:
        st.markdown("**🤖 Model Layer**")
        st.markdown("""
        - XGBoost
        - LightGBM
        - Random Forest
        - LSTM (BiLSTM)
        - Ensemble
        """)

    with arch_col4:
        st.markdown("**🏭 Application Layer**")
        st.markdown("""
        - Digital Twin
        - DDPM Diffusion
        - Scenario Engine
        - Alert Engine
        - SHAP Explainability
        """)

    # ── Assumptions ───────────────────────────────────────────
    with st.expander("📋 Key Assumptions & Decisions"):
        assumptions = eda.get("assumptions", []) or [
            "HAI-23.05 version used as primary dataset (most recent, clean label structure)",
            "Training files contain only normal operation — attack labels from separate label files",
            "Attack labels aligned to data by timestamp join",
            "1 Hz sampling rate (1 second intervals) throughout",
            "Supervised learning used (labels available for all test sets)",
            "Class imbalance (~3-5% attacks) handled via scale_pos_weight / class_weight",
            "Time-aware chronological split — NO data shuffling before split",
            "Feature engineering applied separately to train and test to prevent leakage",
            "Diffusion model conditioned on scenario type (normal=0, attack=1, fault=2)",
        ]
        for a in assumptions:
            st.markdown(f"✅ {a}")


def page_data_explorer():
    st.title("🔍 Data Explorer")

    df = load_sample_data()

    if df.empty:
        st.warning("No data loaded. Check that the HAI dataset path is correct in configs/config.yaml")

        # Show config path hint
        st.code("configs/config.yaml → paths.raw_data: C:/Users/PC GAMING/Desktop/AI/HAI/hai-23.05")
        return

    config = load_config()
    label_col = config.get("data", {}).get("label_col", "Attack")
    ts_col = config.get("data", {}).get("timestamp_col", "timestamp")

    # ── Filters ───────────────────────────────────────────────
    st.subheader("Filters")
    col1, col2, col3 = st.columns(3)

    with col1:
        show_attacks_only = st.checkbox("Show attacks only", value=False)
    with col2:
        n_rows = st.slider("Max rows to display", 1000, min(50000, len(df)), 10000, step=1000)
    with col3:
        if label_col in df.columns:
            attack_filter = st.selectbox("Label filter", ["All", "Normal (0)", "Attack (1)"])

    # Apply filters
    filtered_df = df.copy()
    if label_col in df.columns and 'attack_filter' in locals():
        if attack_filter == "Normal (0)":
            filtered_df = filtered_df[filtered_df[label_col] == 0]
        elif attack_filter == "Attack (1)":
            filtered_df = filtered_df[filtered_df[label_col] == 1]

    filtered_df = filtered_df.head(n_rows)

    # ── Overview Stats ────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{len(filtered_df):,}")
    col2.metric("Features", len(filtered_df.columns))
    if label_col in filtered_df.columns:
        col3.metric("Attacks", f"{filtered_df[label_col].sum():,}")
        col4.metric("Attack Rate", f"{filtered_df[label_col].mean()*100:.2f}%")

    st.divider()

    # ── Time Series Plot ──────────────────────────────────────
    st.subheader("Time Series Visualization")

    numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != label_col]

    selected_sensors = st.multiselect(
        "Select sensors to plot",
        options=numeric_cols[:50],
        default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
    )

    if selected_sensors:
        fig = make_subplots(
            rows=len(selected_sensors), cols=1,
            subplot_titles=selected_sensors,
            shared_xaxes=True,
        )

        x_axis = list(range(len(filtered_df)))

        for i, sensor in enumerate(selected_sensors):
            fig.add_trace(
                go.Scatter(x=x_axis, y=filtered_df[sensor].values,
                          name=sensor, line=dict(width=1)),
                row=i+1, col=1
            )

            # Shade attack regions
            if label_col in filtered_df.columns:
                attack_regions = filtered_df[label_col].values
                attack_starts = np.where(np.diff(np.concatenate([[0], attack_regions])) == 1)[0]
                attack_ends = np.where(np.diff(np.concatenate([attack_regions, [0]])) == -1)[0]

                for start, end in zip(attack_starts, attack_ends):
                    fig.add_vrect(
                        x0=start, x1=end,
                        fillcolor="red", opacity=0.15,
                        layer="below", line_width=0,
                        row=i+1, col=1
                    )

        fig.update_layout(
            height=200 * len(selected_sensors),
            title_text="Sensor Readings (red = attack period)",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Raw Data Table ────────────────────────────────────────
    with st.expander("📋 Raw Data Sample"):
        display_cols = ([ts_col] if ts_col in filtered_df.columns else []) + \
                       selected_sensors[:5] + \
                       ([label_col] if label_col in filtered_df.columns else [])
        display_df = filtered_df[display_cols].head(100)

        if label_col in display_df.columns:
            def highlight_attack(row):
                if row.get(label_col, 0) == 1:
                    return ['background-color: #ff444430'] * len(row)
                return [''] * len(row)

            st.dataframe(display_df.style.apply(highlight_attack, axis=1),
                        use_container_width=True)
        else:
            st.dataframe(display_df, use_container_width=True)


def page_detection():
    st.title("⚠️ Attack & Anomaly Detection")

    pkg = load_haiend_pkg()

    if pkg is None:
        st.error("LSTM model not found. Expected: `outputs/models/haiend_lstm_detection.joblib`")
        st.info("Run `python train_haiend_lstm.py` to train the model first.")
        return

    f1   = pkg.get("best_f1", 0)
    thr  = float(pkg.get("threshold", 0.008))
    name = pkg.get("model_name", "LSTM-AE haiend")
    n_f  = int(pkg.get("n_features", 225))
    st.success(f"✅ **{name}** — F1={f1:.4f}  |  {n_f} sensors  |  window=30s  |  threshold={thr:.6f}")

    tab_upload, tab_test = st.tabs(["📂 Upload CSV", "🔬 Run on Test Data"])

    # ── Tab 1: Upload CSV ──────────────────────────────────────
    with tab_upload:
        st.markdown("Upload any haiend CSV file (or any CSV with sensor columns). "
                    "The model will score each row using a 30-second sliding window.")
        uploaded_file = st.file_uploader(
            "Upload CSV file (haiend format recommended)",
            type=["csv"],
            help="Columns must be sensor values. First column (timestamp) will be auto-detected and dropped."
        )

        if uploaded_file is not None:
            with st.spinner("Loading file..."):
                df_raw = pd.read_csv(uploaded_file)
                st.success(f"Loaded: {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")

            # Drop timestamp / non-numeric columns
            ts_candidates = [c for c in df_raw.columns
                             if "time" in c.lower() or "date" in c.lower()
                             or df_raw[c].dtype == object]
            df_numeric = df_raw.drop(columns=ts_candidates, errors="ignore")
            df_numeric = df_numeric.ffill().fillna(0).astype(np.float32)

            has_label = "label" in df_raw.columns or "Attack" in df_raw.columns
            label_col = "label" if "label" in df_raw.columns else "Attack"

            col_a, col_b = st.columns(2)
            col_a.metric("Sensor columns found", df_numeric.shape[1])
            col_b.metric("Model expects", n_f)

            if st.button("🚀 Run LSTM Detection", key="btn_upload"):
                with st.spinner(f"Scoring {len(df_numeric):,} samples with LSTM-AE..."):
                    X = df_numeric.values.astype(np.float32)
                    scores = score_with_haiend_lstm(X, pkg)
                    # Clamp outliers
                    p999 = float(np.percentile(scores, 99.9))
                    scores = np.clip(scores, 0, p999)
                    preds  = (scores >= thr).astype(int)

                st.divider()
                n_det = int(preds.sum())
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Samples", f"{len(scores):,}")
                col2.metric("Detected Attacks", f"{n_det:,}")
                col3.metric("Detection Rate", f"{n_det/len(scores)*100:.2f}%")
                col4.metric("Avg Score", f"{scores.mean():.6f}")

                if has_label:
                    y = df_raw[label_col].values[:len(scores)].astype(int)
                    from sklearn.metrics import f1_score, precision_score, recall_score
                    f1v  = f1_score(y, preds, zero_division=0)
                    prev = precision_score(y, preds, zero_division=0)
                    recv = recall_score(y, preds, zero_division=0)
                    st.success(f"Ground truth found — F1={f1v:.4f}  Precision={prev:.4f}  Recall={recv:.4f}")

                # Timeline
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=scores, name="LSTM MSE Score",
                                         line=dict(color="steelblue", width=1)))
                fig.add_hline(y=thr, line_dash="dash", line_color="red",
                              annotation_text=f"Threshold ({thr:.6f})")
                if has_label:
                    attack_idx = np.where(df_raw[label_col].values[:len(scores)] == 1)[0]
                    if len(attack_idx):
                        fig.add_trace(go.Scatter(
                            x=attack_idx, y=scores[attack_idx],
                            mode="markers", name="True Attack",
                            marker=dict(color="red", size=3, opacity=0.5)))
                fig.update_layout(title="LSTM Reconstruction Error Timeline",
                                  yaxis_title="MSE Score", xaxis_title="Timestep",
                                  height=400)
                st.plotly_chart(fig, use_container_width=True)

                # Attack table
                attack_rows = np.where(preds == 1)[0]
                with st.expander(f"🚨 Detected Attack Timesteps ({len(attack_rows):,})"):
                    if len(attack_rows):
                        out = pd.DataFrame({"timestep": attack_rows,
                                            "anomaly_score": scores[attack_rows]})
                        st.dataframe(out.head(200), use_container_width=True)
                    else:
                        st.success("No attacks detected.")

                # Download
                result_df = df_raw.copy()
                result_df["anomaly_score"] = scores
                result_df["prediction"]    = preds
                result_df["status"]        = np.where(preds == 1, "ATTACK", "Normal")
                st.download_button("📥 Download Predictions CSV",
                                   result_df.to_csv(index=False),
                                   "predictions.csv", "text/csv")

    # ── Tab 2: Run on built-in test data ──────────────────────
    with tab_test:
        st.markdown("Score the official haiend test set (284,400 timesteps) to verify model performance.")
        n_rows = st.slider("Samples to score", 1000, 30000, 5000, step=1000,
                           help="More samples = more accurate, but slower (~2s per 1000 samples)")

        if st.button("🔬 Score Test Data", key="btn_test"):
            haiend_dir = Path("C:/Users/PC GAMING/Desktop/AI/HAI/haiend-23.05/haiend-23.05")
            test1 = haiend_dir / "end-test1.csv"
            lbl1  = haiend_dir / "label-test1.csv"

            if not test1.exists():
                st.error(f"Test data not found at: {haiend_dir}")
            else:
                with st.spinner(f"Loading and scoring {n_rows:,} samples..."):
                    df_t   = pd.read_csv(test1)
                    df_l   = pd.read_csv(lbl1)
                    X_t    = df_t.iloc[:n_rows, 1:].ffill().fillna(0).astype(np.float32).values
                    y_t    = df_l["label"].values[:n_rows].astype(int)
                    scores = score_with_haiend_lstm(X_t, pkg)
                    p999   = float(np.percentile(scores, 99.9))
                    scores = np.clip(scores, 0, p999)
                    preds  = (scores >= thr).astype(int)

                from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
                f1v  = f1_score(y_t, preds, zero_division=0)
                prev = precision_score(y_t, preds, zero_division=0)
                recv = recall_score(y_t, preds, zero_division=0)
                try:   roc = roc_auc_score(y_t, scores)
                except: roc = float("nan")

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("F1 Score",    f"{f1v:.4f}",  delta="attack detection")
                col2.metric("Precision",   f"{prev:.4f}", delta="low FP rate")
                col3.metric("Recall",      f"{recv:.4f}", delta="attack coverage")
                col4.metric("ROC-AUC",     f"{roc:.4f}")

                n_att = int(y_t.sum())
                n_det = int(preds.sum())
                st.caption(f"True attacks: {n_att:,} ({n_att/len(y_t)*100:.1f}%)   "
                           f"Detected: {n_det:,}   FP: {int((preds==1)&(y_t==0)).sum():,}   "
                           f"FN: {int((preds==0)&(y_t==1)).sum():,}")

                # Timeline with labels
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=scores, name="LSTM MSE",
                                         line=dict(color="steelblue", width=1), opacity=0.8))
                fig.add_hline(y=thr, line_dash="dash", line_color="red",
                              annotation_text="Threshold")
                attack_idx = np.where(y_t == 1)[0]
                if len(attack_idx):
                    fig.add_trace(go.Scatter(
                        x=attack_idx, y=scores[attack_idx], mode="markers",
                        name="True Attack", marker=dict(color="red", size=4, opacity=0.6)))
                fig.update_layout(title=f"LSTM Score Timeline — first {n_rows:,} test samples",
                                  yaxis_title="MSE Score", xaxis_title="Timestep", height=420)
                st.plotly_chart(fig, use_container_width=True)

                # Score distribution
                status_labels = np.where(y_t == 1, "Attack", "Normal")
                fig2 = px.histogram(
                    x=scores, color=status_labels, nbins=80,
                    title="Score Distribution: Normal vs Attack",
                    color_discrete_map={"Normal": "steelblue", "Attack": "red"},
                    labels={"x": "LSTM MSE Score", "color": "Label"},
                    barmode="overlay", opacity=0.7)
                fig2.add_vline(x=thr, line_dash="dash", line_color="red",
                               annotation_text="Threshold")
                st.plotly_chart(fig2, use_container_width=True)


def page_digital_twin():
    st.title("🤖 Digital Twin Simulation")

    pkg = load_haiend_pkg()
    if pkg is None:
        st.error("LSTM model not found. Run `python train_haiend_lstm.py` first.")
        return

    # ── Model info banner ─────────────────────────────────────
    f1  = pkg.get("best_f1", 0)
    thr = float(pkg.get("threshold", 0.008))
    st.success(f"✅ **{pkg.get('model_name','LSTM-AE haiend')}** — F1={f1:.4f}  |  "
               f"225 sensors  |  window=30s  |  threshold={thr:.6f}")
    st.markdown("The Digital Twin streams sensor data through the LSTM-AE and flags "
                "anomalies in real-time. Health score decays during attacks and recovers on clear.")

    st.divider()

    # ── Controls ──────────────────────────────────────────────
    st.subheader("Simulation Controls")
    haiend_dir = Path("C:/Users/PC GAMING/Desktop/AI/HAI/haiend-23.05/haiend-23.05")

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        n_sim = st.slider("Samples to simulate", 500, 10000, 3000, step=500,
                          help="Number of timesteps to stream through the digital twin")
    with col_s2:
        test_file = st.selectbox("Test file", ["end-test1.csv", "end-test2.csv"])
    with col_s3:
        alert_cooldown = st.number_input("Alert cooldown (sec)", 0, 300, 30, step=10,
                                         help="Minimum seconds between repeat alerts")

    run_btn = st.button("▶️ Run Digital Twin Simulation", type="primary")

    if run_btn:
        test_path = haiend_dir / test_file
        lbl_path  = haiend_dir / test_file.replace("end-test", "label-test")

        if not test_path.exists():
            st.error(f"Test file not found: {test_path}")
            st.stop()

        with st.spinner(f"Initialising Digital Twin and streaming {n_sim:,} timesteps..."):
            # Load data
            df_t = pd.read_csv(test_path)
            y_t  = pd.read_csv(lbl_path)["label"].values[:n_sim].astype(int) \
                   if lbl_path.exists() else None
            X_raw = df_t.iloc[:n_sim, 1:].ffill().fillna(0).astype(np.float32).values
            col_names = list(df_t.columns[1:])

            # Init Digital Twin
            config_dt = {
                "paths":        {"outputs": str(ROOT / "outputs")},
                "digital_twin": {"anomaly_threshold": thr, "alert_cooldown": alert_cooldown},
                "data":         {"label_col": "label"},
            }
            from src.digital_twin.digital_twin import DigitalTwin
            twin = DigitalTwin(config_dt, feature_names=col_names)

            # Baseline from training
            train1 = haiend_dir / "end-train1.csv"
            if train1.exists():
                df_base = pd.read_csv(train1)
                X_base  = df_base.iloc[:50000, 1:].ffill().fillna(0).astype(np.float32).values
                twin.fit_baseline(X_base[:, :len(col_names)])
            else:
                twin.fit_baseline(X_raw[:200])

            twin.load_best_model(str(ROOT / "outputs" / "models"))

            # Stream each timestep
            scores      = np.zeros(n_sim, dtype=np.float32)
            health_hist = np.zeros(n_sim, dtype=np.float32)
            preds       = np.zeros(n_sim, dtype=np.int32)
            all_alerts  = []

            progress_bar = st.progress(0, text="Simulating...")
            for t in range(n_sim):
                result = twin.ingest(X_raw[t])
                scores[t]      = result["anomaly_score"]
                health_hist[t] = result["health_score"]
                preds[t]       = int(result["is_anomalous"])
                if result.get("active_alerts"):
                    for msg in result["active_alerts"]:
                        all_alerts.append({"timestep": t, "message": msg,
                                           "score": round(result["anomaly_score"], 4),
                                           "attack_type": result.get("attack_type",""),
                                           "confidence": result.get("confidence","")})
                if t % 500 == 0:
                    progress_bar.progress(t / n_sim, text=f"Simulating... {t:,}/{n_sim:,}")
            progress_bar.empty()

        # ── Metrics ───────────────────────────────────────────
        final_health = float(health_hist[-1])
        n_alerts     = len(all_alerts)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Final Health", f"{final_health:.1f}/100",
                  delta=f"{'▼ degraded' if final_health < 80 else '▲ good'}")
        m2.metric("Alerts Generated", n_alerts)
        m3.metric("Anomalous Steps", f"{int(preds.sum()):,}",
                  delta=f"{preds.mean()*100:.1f}% of run")

        if y_t is not None:
            from sklearn.metrics import f1_score
            f1v = f1_score(y_t, preds, zero_division=0)
            m4.metric("Live F1 Score", f"{f1v:.4f}", delta="vs ground truth")
        else:
            m4.metric("Samples Processed", f"{n_sim:,}")

        st.divider()

        # ── Timeline chart ────────────────────────────────────
        fig = make_subplots(rows=2, cols=1,
                            subplot_titles=["Anomaly Score + Threshold",
                                            "System Health Score"],
                            shared_xaxes=True, vertical_spacing=0.12)

        fig.add_trace(go.Scatter(y=scores, name="Anomaly Score",
                                  line=dict(color="steelblue", width=1)), row=1, col=1)
        fig.add_hline(y=thr, line_dash="dash", line_color="red",
                      annotation_text="Threshold", row=1, col=1)

        if y_t is not None:
            atk_idx = np.where(y_t == 1)[0]
            if len(atk_idx):
                fig.add_trace(go.Scatter(
                    x=atk_idx, y=scores[atk_idx], mode="markers", name="True Attack",
                    marker=dict(color="red", size=3, opacity=0.5)), row=1, col=1)

        fig.add_trace(go.Scatter(y=health_hist, name="Health Score",
                                  line=dict(color="green", width=2),
                                  fill="tozeroy", fillcolor="rgba(0,200,0,0.1)"),
                      row=2, col=1)
        fig.add_hline(y=80, line_dash="dot", line_color="orange",
                      annotation_text="Warning (80)", row=2, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="red",
                      annotation_text="Critical (50)", row=2, col=1)

        fig.update_layout(height=550, showlegend=True)
        fig.update_yaxes(title_text="MSE Score", row=1, col=1)
        fig.update_yaxes(title_text="Health %", range=[0, 105], row=2, col=1)
        fig.update_xaxes(title_text="Timestep (seconds)", row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)

        # ── Alert log ─────────────────────────────────────────
        st.divider()
        st.subheader(f"🚨 Alerts Generated ({n_alerts})")

        if all_alerts:
            alert_df = pd.DataFrame(all_alerts)
            st.dataframe(alert_df, use_container_width=True, hide_index=True)
            st.download_button("📥 Download Alert Log",
                               alert_df.to_csv(index=False),
                               "alert_log.csv", "text/csv")
            # Save for Alerts page
            alert_df.to_csv(ROOT / "outputs" / "metrics" / "alert_log.csv", index=False)
            st.caption("Alert log saved — visible in Alerts page.")
        else:
            st.success("No alerts in this simulation window.")

        # ── Root cause (last anomalous step) ──────────────────
        last_anom = int(np.where(preds == 1)[0][-1]) if preds.sum() > 0 else None
        if last_anom is not None:
            st.divider()
            st.subheader("🔍 Root Cause — Last Anomaly")
            with st.spinner("Analysing..."):
                rc = twin.analyze_root_cause(X_raw[last_anom])
            st.markdown(f"**Probable cause:** {rc.get('probable_cause','—')}")
            st.markdown(f"**Worst subsystem:** {rc.get('worst_subsystem','—')}  "
                        f"| **Max Z-score:** {rc.get('max_z_score',0):.2f}σ  "
                        f"| **Sensors >3σ:** {rc.get('n_sensors_above_3s',0)}")

            top_s = rc.get("top_sensors", [])
            if top_s:
                top_df = pd.DataFrame(top_s)
                st.dataframe(top_df, use_container_width=True, hide_index=True)

            if rc.get("physics_violations"):
                st.subheader("⚡ Physics Violations")
                phys_df = pd.DataFrame(rc["physics_violations"])
                st.dataframe(phys_df, use_container_width=True, hide_index=True)


def page_scenario_simulation():
    st.title("🎭 Scenario Simulation")
    st.markdown("Inject synthetic attack/fault scenarios and observe system response.")

    config = load_config()

    # ── Scenario Controls ─────────────────────────────────────
    st.subheader("Scenario Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        scenario_type = st.selectbox(
            "Scenario Type",
            [
                "cyberattack",
                "sensor_drift",
                "sudden_spike",
                "equipment_degradation",
                "communication_loss",
                "abnormal_operating_condition",
                "replay_attack",
                "setpoint_manipulation",
            ]
        )

    with col2:
        intensity = st.slider("Attack Intensity", 0.1, 1.0, 0.6, step=0.05,
                              help="0=mild, 1=maximum severity")

    with col3:
        n_samples = st.number_input("Duration (seconds)", 100, 3000, 500, step=100)

    # Scenario descriptions
    scenario_descriptions = {
        "cyberattack": "🔴 **Cyberattack**: Adversarial manipulation of multiple sensor readings simultaneously",
        "sensor_drift": "🟡 **Sensor Drift**: Gradual linear bias increase over time in affected sensors",
        "sudden_spike": "🔴 **Sudden Spike**: Abrupt large deviation in sensor reading",
        "equipment_degradation": "🟠 **Degradation**: Increasing noise amplitude over time across all sensors",
        "communication_loss": "🟠 **Comm Loss**: Sensors freeze at last-known value (network outage)",
        "abnormal_operating_condition": "🔴 **Abnormal Condition**: All sensors shift to unusual operating region",
        "replay_attack": "🔴 **Replay Attack**: Previous normal data replayed to mask ongoing attack",
        "setpoint_manipulation": "🔴 **Setpoint Manipulation**: Control setpoints changed to abnormal values",
    }

    st.info(scenario_descriptions.get(scenario_type, ""))

    if st.button("▶️ Run Simulation", type="primary"):
        with st.spinner(f"Simulating {scenario_type} scenario..."):
            # Load sample baseline data
            sample_df = load_sample_data(max_rows=5000)

            if sample_df.empty:
                st.error("No baseline data available")
                return

            label_col = config.get("data", {}).get("label_col", "Attack")
            ts_col = config.get("data", {}).get("timestamp_col", "timestamp")

            feature_cols = [c for c in sample_df.columns
                           if c not in [label_col, ts_col] and
                           sample_df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]
            feature_cols = feature_cols[:50]

            # Normal data for baseline
            if label_col in sample_df.columns:
                normal_df = sample_df[sample_df[label_col] == 0][feature_cols].fillna(0)
            else:
                normal_df = sample_df[feature_cols].fillna(0)

            baseline = normal_df.values[:2000].astype(np.float64)

            # Generate scenario
            scenario_data = generate_scenario_data(
                scenario_type, intensity, int(n_samples), baseline, feature_cols
            )

            # Compare normal vs scenario
            st.success(f"✅ Generated {len(scenario_data)} scenario samples")

            # Visualization
            st.subheader("Normal vs Scenario Comparison")

            n_plot_sensors = min(4, len(feature_cols))
            fig = make_subplots(
                rows=n_plot_sensors, cols=1,
                subplot_titles=[f"{feature_cols[i]}" for i in range(n_plot_sensors)],
                shared_xaxes=True,
            )

            plot_len = min(300, len(scenario_data), len(baseline))

            for i in range(n_plot_sensors):
                if i < len(feature_cols):
                    # Normal
                    fig.add_trace(
                        go.Scatter(y=baseline[:plot_len, i], name="Normal" if i == 0 else None,
                                  line=dict(color="steelblue", width=1.5),
                                  showlegend=(i == 0)),
                        row=i+1, col=1
                    )
                    # Scenario
                    fig.add_trace(
                        go.Scatter(y=scenario_data[:plot_len, i], name="Scenario" if i == 0 else None,
                                  line=dict(color="red", width=1.5, dash="dash"),
                                  showlegend=(i == 0)),
                        row=i+1, col=1
                    )

            fig.update_layout(
                height=200 * n_plot_sensors,
                title_text=f"Scenario: {scenario_type} (intensity={intensity})",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Statistical comparison
            st.subheader("Statistical Impact")

            impact_rows = []
            for i, col_name in enumerate(feature_cols[:10]):
                if i < baseline.shape[1] and i < scenario_data.shape[1]:
                    normal_mean = baseline[:, i].mean()
                    scenario_mean = scenario_data[:, i].mean()
                    normal_std = baseline[:, i].std()
                    scenario_std = scenario_data[:, i].std()

                    deviation = abs(scenario_mean - normal_mean) / (normal_std + 1e-8)

                    impact_rows.append({
                        "Sensor": col_name,
                        "Normal Mean": round(normal_mean, 4),
                        "Scenario Mean": round(scenario_mean, 4),
                        "Normal Std": round(normal_std, 4),
                        "Scenario Std": round(scenario_std, 4),
                        "Z-Score Deviation": round(deviation, 2),
                        "Impacted": "⚠️" if deviation > 2 else "✅",
                    })

            impact_df = pd.DataFrame(impact_rows).sort_values("Z-Score Deviation", ascending=False)
            st.dataframe(impact_df, use_container_width=True, hide_index=True)


def page_synthetic_generation():
    st.title("✨ Synthetic Scenario Generation (Diffusion Model)")

    config = load_config()

    st.markdown("""
    The **DDPM Diffusion Model** generates synthetic ICS scenarios by learning the
    joint distribution of sensor readings. Generated scenarios are class-conditioned:
    - Class 0: Normal operation
    - Class 1: Cyberattack patterns
    - Class 2: Equipment fault/degradation
    """)

    # ── Model Status ──────────────────────────────────────────
    diffusion_path = ROOT / "outputs" / "models" / "diffusion_model.pt"
    model_available = diffusion_path.exists()

    if model_available:
        st.success("✅ Diffusion model available and trained")
    else:
        st.warning("⚠️ Diffusion model not yet trained. Run `python run_pipeline.py` to train.")

    # ── Generation Controls ───────────────────────────────────
    st.subheader("Generation Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:
        scenario_class = st.selectbox("Scenario Class",
                                       ["0 - Normal Operation", "1 - Cyberattack", "2 - Equipment Fault"])
        class_id = int(scenario_class.split(" - ")[0])

    with col2:
        n_generate = st.number_input("Samples to Generate", 100, 2000, 500, step=100)

    with col3:
        compare_real = st.checkbox("Compare with real data", value=True)

    if st.button("🎲 Generate Synthetic Scenarios", type="primary"):
        with st.spinner("Generating synthetic data..."):
            sample_df = load_sample_data(max_rows=3000)

            if sample_df.empty:
                st.error("No reference data available")
                return

            label_col = config.get("data", {}).get("label_col", "Attack")
            ts_col = config.get("data", {}).get("timestamp_col", "timestamp")
            feature_cols = [c for c in sample_df.columns
                           if c not in [label_col, ts_col] and
                           sample_df[c].dtype in [np.float64, np.float32, np.int64, np.int32]][:50]

            real_data = sample_df[feature_cols].fillna(0).values[:2000]

            # Try diffusion model first
            synthetic_data = None

            if model_available:
                try:
                    sys.path.insert(0, str(ROOT))
                    from src.diffusion.diffusion_model import HAIDiffusionModel
                    diffusion = HAIDiffusionModel(config)
                    diffusion.load(str(diffusion_path))
                    synthetic_data = diffusion.generate(int(n_generate), scenario_class=class_id)
                    generation_method = "DDPM Diffusion Model"
                except Exception as e:
                    st.warning(f"Diffusion generation failed: {e}. Using rule-based fallback.")

            # Fallback: rule-based
            if synthetic_data is None:
                scenario_map = {0: "cyberattack", 1: "cyberattack", 2: "equipment_degradation"}
                from src.diffusion.scenario_generator import ScenarioGenerator
                gen = ScenarioGenerator(config)
                synthetic_data = gen.generate_rule_based(
                    real_data, scenario_type=scenario_map[class_id],
                    intensity=0.6, n_samples=int(n_generate),
                    feature_names=feature_cols
                )
                generation_method = "Rule-Based Generator (fallback)"

            st.success(f"✅ Generated {len(synthetic_data)} samples via: **{generation_method}**")

            n_feats = min(synthetic_data.shape[1], len(feature_cols))
            synth_df = pd.DataFrame(synthetic_data[:, :n_feats], columns=feature_cols[:n_feats])

            # Distribution comparison
            if compare_real:
                st.subheader("Distribution Comparison: Real vs Synthetic")

                plot_features = feature_cols[:6]

                fig = make_subplots(rows=2, cols=3, subplot_titles=plot_features)

                for i, feat in enumerate(plot_features):
                    row = i // 3 + 1
                    col = i % 3 + 1

                    if feat in synth_df.columns:
                        # Real distribution
                        real_vals = sample_df[feat].dropna().values if feat in sample_df else real_data[:, i]
                        fig.add_trace(
                            go.Histogram(x=real_vals[:1000], name="Real", opacity=0.7,
                                        marker_color="steelblue", histnorm="probability",
                                        showlegend=(i == 0)),
                            row=row, col=col
                        )
                        # Synthetic distribution
                        fig.add_trace(
                            go.Histogram(x=synth_df[feat].values, name="Synthetic", opacity=0.7,
                                        marker_color="coral", histnorm="probability",
                                        showlegend=(i == 0)),
                            row=row, col=col
                        )

                fig.update_layout(height=500, title_text="Feature Distribution Comparison")
                st.plotly_chart(fig, use_container_width=True)

            # Download
            csv = synth_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Synthetic Data CSV",
                data=csv,
                file_name=f"synthetic_class{class_id}_{scenario_class.split(' - ')[1].replace(' ', '_')}.csv",
                mime="text/csv"
            )


def page_explainability():
    st.title("🔎 Explainability — Feature Contributions")

    model, model_name = load_trained_model()

    if model is None:
        st.warning("No trained model found. Run `python run_pipeline.py` first.")
        return

    st.success(f"✅ Model loaded: **{model_name}**")

    # ── Feature Importance ────────────────────────────────────
    st.subheader("Feature Importance (Model-Based)")

    if hasattr(model, 'feature_importances_'):
        config = load_config()
        label_col = config.get("data", {}).get("label_col", "Attack")
        ts_col = config.get("data", {}).get("timestamp_col", "timestamp")

        # Load sample data to get feature names
        sample_df = load_sample_data(max_rows=1000)
        if not sample_df.empty:
            feature_cols = [c for c in sample_df.columns
                           if c not in [label_col, ts_col] and
                           sample_df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]

            importances = model.feature_importances_
            n = min(len(importances), len(feature_cols), 30)

            fi_df = pd.DataFrame({
                "Feature": feature_cols[:n],
                "Importance": importances[:n],
            }).sort_values("Importance", ascending=False).head(30)

            fig = px.bar(
                fi_df, x="Importance", y="Feature",
                orientation='h',
                title=f"Top 30 Feature Importances — {model_name}",
                color="Importance",
                color_continuous_scale=["steelblue", "red"],
            )
            fig.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

            # Feature categories breakdown
            st.subheader("Feature Category Analysis")

            fi_df["Category"] = fi_df["Feature"].apply(
                lambda x: x.split("_")[0] if "_" in x else "Other"
            )

            cat_importance = fi_df.groupby("Category")["Importance"].sum().sort_values(ascending=False)

            fig_pie = px.pie(
                values=cat_importance.values,
                names=cat_importance.index,
                title="Importance by Sensor Subsystem",
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    # ── SHAP Values (if available) ────────────────────────────
    st.subheader("SHAP Analysis (if available)")

    try:
        import shap

        sample_df = load_sample_data(max_rows=500)
        preprocessor_state = load_preprocessor()
        config = load_config()
        label_col = config.get("data", {}).get("label_col", "Attack")
        ts_col = config.get("data", {}).get("timestamp_col", "timestamp")

        if not sample_df.empty and preprocessor_state:
            stored_cols = preprocessor_state.get("numeric_cols", [])
            available = [c for c in stored_cols if c in sample_df.columns][:50]

            if available and preprocessor_state.get("scaler") is not None:
                X = sample_df[available].fillna(0).values[:200]
                X_scaled = preprocessor_state["scaler"].transform(X)

                explainer = shap.Explainer(model, X_scaled)
                shap_values = explainer(X_scaled[:50])

                fig_shap, ax = plt.subplots(figsize=(12, 8))
                shap.summary_plot(shap_values, X_scaled[:50],
                                 feature_names=available, show=False, plot_size=None)
                import io
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                buf.seek(0)
                st.image(buf, caption="SHAP Summary Plot")
                plt.close()

    except ImportError:
        st.info("Install SHAP for more detailed explainability: `pip install shap`")
    except Exception as e:
        st.info(f"SHAP analysis not available: {str(e)[:100]}")


def page_alerts():
    st.title("🚨 Alert Monitoring")

    # ── Load Alert Log ────────────────────────────────────────
    alert_log_path = ROOT / "outputs" / "metrics" / "alert_log.csv"
    dt_eval_path   = ROOT / "outputs" / "metrics" / "digital_twin_eval.json"

    alert_df = pd.DataFrame()
    if alert_log_path.exists():
        try:
            alert_df = pd.read_csv(alert_log_path)
        except Exception:
            pass

    # ── Model performance banner ──────────────────────────────
    if dt_eval_path.exists():
        try:
            with open(dt_eval_path) as f:
                dt_eval = json.load(f)
            m = dt_eval.get("metrics", {}).get("DT_binary") or \
                dt_eval.get("metrics", {}).get("LSTM_haiend_raw", {})
            if m:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Model F1",     f"{m.get('f1',0):.4f}",  "on full test set")
                c2.metric("Precision",    f"{m.get('precision',0):.4f}")
                c3.metric("Recall",       f"{m.get('recall',0):.4f}")
                c4.metric("ROC-AUC",      f"{m.get('roc_auc',0):.4f}")
                st.caption(f"TP={m.get('tp',0):,}  FP={m.get('fp',0):,}  "
                           f"TN={m.get('tn',0):,}  FN={m.get('fn',0):,}  "
                           f"— evaluated on {dt_eval.get('test_shape',['?','?'])[0]:,} timesteps")
                st.divider()
        except Exception:
            pass

    # ── Alert Summary ─────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    if len(alert_df) > 0:
        def count_sev(sev):
            if "severity" in alert_df.columns:
                return int((alert_df["severity"] == sev).sum())
            return 0
        crit   = count_sev("CRITICAL")
        highs  = count_sev("HIGH")
        meds   = count_sev("MEDIUM")
        lows   = count_sev("LOW")
        col1.metric("🔴 CRITICAL", crit)
        col2.metric("🟠 HIGH",     highs)
        col3.metric("🟡 MEDIUM",   meds)
        col4.metric("🔵 LOW",      lows)
    else:
        col1.metric("Total Alerts", 0)
        col2.metric("Status", "✅ No alerts")
        col3.metric("Source", "Run Digital Twin")
        col4.metric("Tip", "→ Digital Twin page")

    st.divider()

    # ── How to generate alerts ────────────────────────────────
    if len(alert_df) == 0:
        st.info("No alerts yet.  **How to generate alerts:**\n\n"
                "1. Go to the **Digital Twin** page\n"
                "2. Click **▶️ Run Digital Twin Simulation**\n"
                "3. Alerts will appear here automatically after the run.")
        st.divider()

    # ── Alert Feed ────────────────────────────────────────────
    if len(alert_df) > 0:
        st.subheader(f"Alert Feed ({len(alert_df):,} alerts)")

        # Filters
        fc1, fc2 = st.columns(2)
        with fc1:
            if "severity" in alert_df.columns:
                sev_filter = st.multiselect("Filter by severity",
                    ["CRITICAL","HIGH","MEDIUM","LOW"],
                    default=["CRITICAL","HIGH","MEDIUM","LOW"])
                alert_df = alert_df[alert_df["severity"].isin(sev_filter)]
        with fc2:
            if "attack_type" in alert_df.columns:
                types = list(alert_df["attack_type"].dropna().unique())
                type_filter = st.multiselect("Filter by type", types, default=types)
                alert_df = alert_df[alert_df["attack_type"].isin(type_filter)]

        # Timeline of alerts
        if "timestep" in alert_df.columns and "score" in alert_df.columns:
            fig = go.Figure()
            sev_colors = {"CRITICAL": "darkred", "HIGH": "red",
                          "MEDIUM": "orange", "LOW": "yellow"}
            for sev, grp in alert_df.groupby("severity") if "severity" in alert_df.columns \
                    else [("ALL", alert_df)]:
                fig.add_trace(go.Scatter(
                    x=grp["timestep"], y=grp["score"],
                    mode="markers", name=sev,
                    marker=dict(color=sev_colors.get(sev, "blue"), size=6)))
            fig.update_layout(title="Alert Timeline", height=300,
                              xaxis_title="Timestep", yaxis_title="Anomaly Score")
            st.plotly_chart(fig, use_container_width=True)

        # Alert cards
        st.subheader("Alert Details")
        show_n = st.slider("Show latest N alerts", 5, min(200, len(alert_df)), 20)
        for _, row in alert_df.tail(show_n).iloc[::-1].iterrows():
            sev = str(row.get("severity", row.get("message","")[:4] if "message" in row else "LOW"))
            if "CRITICAL" in sev:    css = "alert-high"
            elif "HIGH" in sev:      css = "alert-high"
            elif "MEDIUM" in sev:    css = "alert-medium"
            else:                    css = "alert-low"
            msg   = row.get("message", str(row.get("attack_type","anomaly")))
            ts    = str(row.get("timestamp", row.get("timestep", "")))
            score = row.get("score", row.get("anomaly_score", ""))
            atype = row.get("attack_type", "")
            conf  = row.get("confidence", "")
            extra = f" | type: {atype}" if atype else ""
            extra += f" | confidence: {conf}" if conf else ""
            st.markdown(f"""
            <div class="{css}" style="margin-bottom:4px">
                <strong>{str(ts)[:19]}</strong> — {msg}
                {f'<br><small>score={score:.4f}{extra}</small>' if score != '' else ''}
            </div>
            """, unsafe_allow_html=True)

        st.divider()
        # Full table
        with st.expander("📋 Full Alert Table"):
            st.dataframe(alert_df, use_container_width=True, hide_index=True)
        st.download_button("📥 Download Alert Log",
                           alert_df.to_csv(index=False), "alert_log.csv", "text/csv")

    st.divider()

    # ── Recommendations Reference ─────────────────────────────
    with st.expander("📋 Recommended Actions by Scenario"):
        recommendations = {
            "Cyberattack": [
                "Immediately isolate affected DCS components",
                "Activate emergency shutdown protocol if safety-critical",
                "Preserve logs for forensic analysis",
                "Notify CISO and operations team",
                "Switch to manual control mode if possible",
            ],
            "Sensor Drift": [
                "Schedule calibration for affected sensor",
                "Cross-validate with redundant sensors",
                "Monitor trend — may indicate physical damage",
                "Consider temporary safety margin increase",
            ],
            "Communication Loss": [
                "Check network infrastructure (switches, cables, PLCs)",
                "Verify PLC communication modules",
                "Switch to local control if network is unavailable",
                "Inspect for physical cable damage",
            ],
            "Equipment Degradation": [
                "Schedule predictive maintenance",
                "Review recent maintenance logs",
                "Increase monitoring frequency",
                "Prepare backup components",
            ],
        }

        for scenario, actions in recommendations.items():
            st.markdown(f"**{scenario}:**")
            for action in actions:
                st.markdown(f"  • {action}")


# ── Model Arena ───────────────────────────────────────────────────────────────

def _compute_event_f1(y: np.ndarray, pred: np.ndarray) -> dict:
    """
    Event-level F1: contiguous attack sequences counted as events.
    TP event = attack event where at least 1 timestep was detected.
    FP event = predicted anomaly window with no overlap with any attack event.
    """
    # Find contiguous attack events
    attack_events = []
    in_evt = False
    for i, lbl in enumerate(y):
        if lbl == 1 and not in_evt:
            in_evt, start = True, i
        elif lbl == 0 and in_evt:
            attack_events.append((start, i))
            in_evt = False
    if in_evt:
        attack_events.append((start, len(y)))

    # Build attack timestep set
    attack_ts = set()
    for s, e in attack_events:
        attack_ts.update(range(s, e))

    # Find contiguous predicted events
    pred_events = []
    in_pred = False
    for i, p in enumerate(pred):
        if p == 1 and not in_pred:
            in_pred, start = True, i
        elif p == 0 and in_pred:
            pred_events.append((start, i))
            in_pred = False
    if in_pred:
        pred_events.append((start, len(pred)))

    tp = sum(1 for s, e in attack_events if pred[s:e].sum() > 0)
    fn = len(attack_events) - tp
    fp = sum(1 for s, e in pred_events if not any(t in attack_ts for t in range(s, e)))

    pre = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1  = 2 * pre * rec / (pre + rec) if (pre + rec) > 0 else 0.0
    return dict(event_f1=round(f1, 4), event_precision=round(pre, 4),
                event_recall=round(rec, 4), tp_events=tp, fp_events=fp,
                fn_events=fn, n_attack_events=len(attack_events),
                n_pred_events=len(pred_events))


@st.cache_data(show_spinner=False)
def load_all_model_metrics() -> list:
    """
    Scan outputs/models for all detection joblib packages and return their metrics.
    Also computes event-level F1 from stored per-model evaluation results.
    """
    metrics_dir = ROOT / "outputs" / "metrics"
    models_dir  = ROOT / "outputs" / "models"
    rows = []

    # Load ensemble eval if present
    ens_path = metrics_dir / "ensemble_eval.json"
    ens_all  = {}
    if ens_path.exists():
        try:
            ens_data = json.load(open(ens_path))
            ens_all  = ens_data.get("all_metrics", {})
        except Exception:
            pass

    # Known model files and display names
    model_files = [
        ("haiend_lstm_detection.joblib",   "LSTM-AE (haiend, w=30)"),
        ("transformer_ae_detection.joblib", "Transformer-AE (w=30)"),
        ("gru_gat_detection.joblib",        "GRU-GAT (graph-attention)"),
        ("lstm_vae_detection.joblib",       "LSTM-VAE"),
        ("multiscale_lstm_detection.joblib","LSTM MultiScale (w10+30+60)"),
        ("lstm_ae_detection.joblib",        "LSTM-AE (38 feat)"),
    ]

    for fname, display_name in model_files:
        path = models_dir / fname
        if not path.exists():
            continue
        try:
            import joblib as _jl
            pkg = _jl.load(path)
            if not isinstance(pkg, dict):
                continue
            f1  = pkg.get("best_f1", pkg.get("f1", None))
            m   = pkg.get("all_metrics", {})
            if isinstance(m, dict):
                best_key = max(m, key=lambda k: m[k].get("f1", 0)) if m else None
                if best_key:
                    bm = m[best_key]
                    rows.append({
                        "Model":      display_name,
                        "F1":         round(bm.get("f1", f1 or 0), 4),
                        "Precision":  round(bm.get("precision", 0), 4),
                        "Recall":     round(bm.get("recall", 0), 4),
                        "ROC-AUC":    round(bm.get("roc_auc", 0), 4),
                        "TP":         bm.get("tp", "–"),
                        "FP":         bm.get("fp", "–"),
                        "FN":         bm.get("fn", "–"),
                        "Strategy":   best_key,
                    })
                    continue
            if f1 is not None:
                rows.append({
                    "Model":     display_name,
                    "F1":        round(float(f1), 4),
                    "Precision": "–", "Recall": "–", "ROC-AUC": "–",
                    "TP": "–", "FP": "–", "FN": "–", "Strategy": "–",
                })
        except Exception:
            continue

    # Add ensemble metrics from ensemble_eval.json
    for strategy, m in ens_all.items():
        if strategy in ("LSTM_AE_alone", "Transformer_alone"):
            continue
        rows.append({
            "Model":     f"Ensemble → {strategy}",
            "F1":        round(m.get("f1", 0), 4),
            "Precision": round(m.get("precision", 0), 4),
            "Recall":    round(m.get("recall", 0), 4),
            "ROC-AUC":   round(m.get("roc_auc", 0), 4),
            "TP":        m.get("tp", "–"),
            "FP":        m.get("fp", "–"),
            "FN":        m.get("fn", "–"),
            "Strategy":  strategy,
        })

    return sorted(rows, key=lambda r: float(r["F1"]) if str(r["F1"]) != "–" else 0,
                  reverse=True)


def page_model_arena():
    st.title("🏆 Model Arena — Detection Performance Comparison")
    st.caption(
        "All models trained on haiend-23.05 normal data (896K samples, 225 DCS sensors). "
        "Evaluated on 284K test timesteps with 11,384 attack labels (4.0%)."
    )

    # ── Performance Table ────────────────────────────────────────────────────
    st.subheader("📊 All Models — Ranked by F1")

    rows = load_all_model_metrics()

    if not rows:
        st.warning("No trained models found. Run training scripts first.")
        return

    df_models = pd.DataFrame(rows)
    best_f1 = df_models["F1"].max() if len(df_models) else 0

    # Style the best row
    def highlight_best(row):
        if row["F1"] == best_f1:
            return ["background-color: rgba(0,200,100,0.25)"] * len(row)
        if str(row["F1"]) != "–" and float(row["F1"]) >= best_f1 * 0.99:
            return ["background-color: rgba(0,150,255,0.15)"] * len(row)
        return [""] * len(row)

    styled = df_models.style.apply(highlight_best, axis=1).format(
        {c: "{:.4f}" for c in ["F1", "Precision", "Recall", "ROC-AUC"]
         if c in df_models.columns}
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # ── Best Model KPIs ──────────────────────────────────────────────────────
    if rows:
        best = rows[0]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Best F1",        f"{best['F1']:.4f}",  delta=best["Model"])
        c2.metric("Precision",      str(best["Precision"]))
        c3.metric("Recall",         str(best["Recall"]))
        c4.metric("ROC-AUC",        str(best["ROC-AUC"]))
        c5.metric("True Positives", str(best["TP"]))

    st.divider()

    # ── F1 Bar Chart ─────────────────────────────────────────────────────────
    st.subheader("📈 F1 Score Comparison")

    df_chart = df_models[df_models["F1"].apply(
        lambda x: str(x) != "–" and "Ensemble →" not in str(df_models.loc[
            df_models["F1"] == x].iloc[0]["Model"] if len(df_models[df_models["F1"] == x]) else "")
    )].copy()

    # Filter to unique models only (no ensemble sub-strategies)
    df_single = df_models[~df_models["Model"].str.startswith("Ensemble →")].copy()
    if len(df_single) > 0:
        fig = go.Figure()
        colors = [
            "rgba(0,200,100,0.85)" if row["F1"] == best_f1
            else ("rgba(0,150,255,0.70)" if str(row["F1"]) != "–" and float(row["F1"]) >= 0.68
                  else "rgba(150,150,200,0.60)")
            for _, row in df_single.iterrows()
        ]
        fig.add_trace(go.Bar(
            x=df_single["Model"], y=df_single["F1"].astype(float),
            marker_color=colors,
            text=df_single["F1"].apply(lambda v: f"{v:.4f}"),
            textposition="outside",
        ))
        fig.update_layout(
            title="F1 Score per Model (test set, 284K timesteps)",
            xaxis_title="Model", yaxis_title="F1 Score",
            yaxis_range=[0, min(1.0, best_f1 * 1.12)],
            height=400, margin=dict(t=50, b=150),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0e0e0"),
        )
        fig.update_xaxes(tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── TP / FP / FN Breakdown ───────────────────────────────────────────────
    st.subheader("🎯 TP / FP / FN Breakdown (single-model, threshold-optimised)")

    numeric_rows = [r for r in rows
                    if not r["Model"].startswith("Ensemble →")
                    and str(r["TP"]) != "–"]
    if numeric_rows:
        df_conf = pd.DataFrame(numeric_rows)[["Model", "TP", "FP", "FN"]]
        df_melt = df_conf.melt(id_vars="Model", value_vars=["TP", "FP", "FN"],
                               var_name="Category", value_name="Count")
        df_melt["Count"] = pd.to_numeric(df_melt["Count"], errors="coerce")
        color_map = {"TP": "rgba(0,200,100,0.8)", "FP": "rgba(255,100,50,0.8)",
                     "FN": "rgba(255,200,0,0.8)"}
        fig2 = go.Figure()
        for cat, color in color_map.items():
            sub = df_melt[df_melt["Category"] == cat]
            fig2.add_trace(go.Bar(
                name=cat, x=sub["Model"], y=sub["Count"],
                marker_color=color,
                text=sub["Count"].apply(lambda v: f"{int(v):,}" if pd.notna(v) else ""),
                textposition="inside",
            ))
        fig2.update_layout(
            barmode="group", height=380,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0e0e0"), legend=dict(orientation="h"),
            margin=dict(t=30, b=150),
        )
        fig2.update_xaxes(tickangle=-35)
        st.plotly_chart(fig2, use_container_width=True)
        st.caption(
            "TP=11,384 total attacks. Goal: maximise TP, minimise FP and FN. "
            "FN=missed attacks are more dangerous than FP=false alarms in ICS."
        )

    st.divider()

    # ── Event-Level F1 Info ──────────────────────────────────────────────────
    st.subheader("📅 Event-Level Evaluation")
    st.info(
        "**Timestep-level F1** (above) counts each second independently — one attack spanning "
        "10 minutes = 600 individual predictions. "
        "\n\n**Event-level F1** counts each contiguous attack sequence as 1 event. "
        "A True Positive event only requires detecting at least 1 timestep within the attack window. "
        "This better reflects real-world performance: one alarm during an attack is enough to respond. "
        "\n\nEvent-level evaluation requires loading 284K test timesteps — click below to compute."
    )

    if st.button("▶ Compute Event-Level F1 (loads test data ~10s)"):
        with st.spinner("Loading test data and scoring..."):
            try:
                import joblib as _jl
                haiend_dir = Path("C:/Users/PC GAMING/Desktop/AI/HAI/haiend-23.05/haiend-23.05")

                X_parts, y_parts = [], []
                for i in [1, 2]:
                    X_df = pd.read_csv(haiend_dir / f"end-test{i}.csv")
                    y_df = pd.read_csv(haiend_dir / f"label-test{i}.csv")
                    X_parts.append(X_df.iloc[:, 1:].ffill().fillna(0).astype(np.float32).values)
                    y_parts.append(y_df["label"].values.astype(np.int32))
                X_test = np.concatenate(X_parts, axis=0)
                y_test = np.concatenate(y_parts, axis=0)

                event_results = []
                lstm_path = ROOT / "outputs" / "models" / "haiend_lstm_detection.joblib"
                if lstm_path.exists():
                    pkg = _jl.load(lstm_path)
                    scores = score_with_haiend_lstm(X_test, pkg)
                    thr    = float(pkg.get("threshold", np.percentile(scores, 92)))
                    pred   = (scores >= thr).astype(int)
                    ef = _compute_event_f1(y_test, pred)
                    event_results.append({"Model": "LSTM-AE (haiend)", **ef})

                # Ensemble from stored JSON
                ens_path = ROOT / "outputs" / "metrics" / "ensemble_eval.json"
                if ens_path.exists():
                    ens_data = json.load(open(ens_path))
                    best_strat = ens_data.get("best_strategy", "max_norm")
                    bm = ens_data.get("all_metrics", {}).get(best_strat, {})
                    tp_e  = bm.get("tp", 0)
                    fp_e  = bm.get("fp", 0)
                    fn_e  = bm.get("fn", 0)
                    total = tp_e + fn_e
                    event_results.append({
                        "Model": f"Ensemble ({best_strat})",
                        "event_f1": "N/A (timestep)",
                        "tp_events": tp_e, "fp_events": fp_e, "fn_events": fn_e,
                        "n_attack_events": total,
                    })

                if event_results:
                    st.dataframe(pd.DataFrame(event_results), use_container_width=True,
                                 hide_index=True)
                    st.caption(
                        "Event-level F1 is typically much higher than timestep-level "
                        "because a single detection within a multi-minute attack is sufficient."
                    )
                else:
                    st.warning("No models available for event-level scoring.")
            except Exception as ex:
                st.error(f"Event-level evaluation failed: {ex}")

    st.divider()

    # ── Improvement History ──────────────────────────────────────────────────
    st.subheader("📈 Improvement History (all 18+ approaches tried)")

    history = [
        ("Supervised XGBoost/LightGBM",      0.12),
        ("MLP Autoencoder (38 feat)",         0.37),
        ("MLP AE + EWM smoothing",            0.396),
        ("GDN graph network (38 feat)",       0.417),
        ("LSTM-AE (38 feat, small)",          0.434),
        ("haiend LSTM w=30 (100K windows)",   0.687),
        ("haiend LSTM w=30 (150K windows)",   0.6886),
        ("LSTM-VAE (150K windows)",           0.670),
        ("Transformer-AE (w=30, 60ep)",       0.6795),
        ("Ensemble LSTM+Transformer (max)",   0.6998),
        ("GRU-GAT (inter-sensor graph, CNN)",  0.4704),  # trained — display only, not in ensemble
    ]

    df_hist = pd.DataFrame(history, columns=["Approach", "F1"])
    trained = df_hist[df_hist["F1"].notna()].copy()
    pending = df_hist[df_hist["F1"].isna()].copy()

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=trained.index, y=trained["F1"],
        mode="lines+markers+text",
        text=trained["F1"].apply(lambda v: f"{v:.3f}"),
        textposition="top right",
        marker=dict(size=10, color="rgba(0,180,255,0.9)"),
        line=dict(color="rgba(0,180,255,0.7)", width=2),
        name="Completed",
    ))
    if len(pending):
        fig3.add_trace(go.Scatter(
            x=pending.index, y=[trained["F1"].max()] * len(pending),
            mode="markers+text",
            text=["⏳ Training..." for _ in pending.index],
            textposition="top right",
            marker=dict(size=12, color="rgba(255,200,0,0.8)", symbol="diamond"),
            name="In Progress",
        ))

    fig3.update_layout(
        title="F1 Progression Across All Approaches",
        xaxis=dict(tickmode="array", tickvals=df_hist.index,
                   ticktext=[r[:35] for r in df_hist["Approach"]], tickangle=-40),
        yaxis_title="F1 Score",
        height=420, margin=dict(t=50, b=200),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"),
    )
    st.plotly_chart(fig3, use_container_width=True)

    # ── Architecture Summary ─────────────────────────────────────────────────
    st.subheader("🏗️ Active Ensemble Architecture")
    cols = st.columns(3)
    with cols[0]:
        st.markdown("""
**LSTM-AE** *(Layer A)*
- F1 = 0.6886
- Higher recall (fewer FN)
- Sequential temporal patterns
- 400K parameters
        """)
    with cols[1]:
        st.markdown("""
**Transformer-AE** *(Layer A2)*
- F1 = 0.6795
- Higher precision (fewer FP)
- Best ROC-AUC = 0.8886
- Global window attention
- 1.26M parameters
        """)
    with cols[2]:
        st.markdown("""
**GRU-GAT** *(display / root cause)*
- F1 = 0.4704 standalone
- Inter-sensor learned adjacency
- NOT in Hard OR ensemble (too many FP)
- Per-sensor errors for root cause analysis
        """)

    st.success(
        "**Active Ensemble (Hard-OR)**: is_anomalous = LSTM OR Transformer → F1=0.6998  \n"
        "GRU-GAT loaded as display/root-cause layer (standalone F1=0.4704, excluded from decision)."
    )


# ── Main App ──────────────────────────────────────────────────────────────────

def main():
    """Main dashboard entry point."""
    page = render_sidebar()

    # Import matplotlib here to avoid issues
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    page_map = {
        "Overview": page_overview,
        "Data Explorer": page_data_explorer,
        "Detection": page_detection,
        "Digital Twin": page_digital_twin,
        "Model Arena": page_model_arena,
        "Scenario Simulation": page_scenario_simulation,
        "Synthetic Generation": page_synthetic_generation,
        "Explainability": page_explainability,
        "Alerts": page_alerts,
    }

    # Render selected page
    render_func = page_map.get(page, page_overview)
    render_func()

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#666;font-size:0.8em;'>"
        "HAI ICS Security Pipeline | "
        "HAI Dataset v23.05 | "
        "Boiler/Steam Control System Anomaly Detection"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
