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

    model, model_name = load_trained_model()
    preprocessor_state = load_preprocessor()

    if model is None:
        st.warning("No trained model found. Run `python run_pipeline.py` first.")
        st.info("The pipeline will train XGBoost, LightGBM, Random Forest, and Ensemble models.")
        return

    st.success(f"✅ Model loaded: **{model_name}**")

    # ── File Upload ───────────────────────────────────────────
    st.subheader("Upload Data for Detection")
    uploaded_file = st.file_uploader(
        "Upload CSV file (HAI format, comma-separated)",
        type=["csv"],
        help="Upload a CSV file with HAI sensor columns"
    )

    if uploaded_file is not None:
        with st.spinner("Processing..."):
            df = pd.read_csv(uploaded_file)
            st.success(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

            # Run detection
            result_df = detect_anomalies(df, model, preprocessor_state)

            if "anomaly_score" in result_df.columns:
                # Summary metrics
                n_attacks = result_df["prediction"].sum() if "prediction" in result_df.columns else 0
                st.divider()

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Samples", f"{len(result_df):,}")
                col2.metric("Detected Attacks", f"{n_attacks:,}")
                col3.metric("Detection Rate", f"{n_attacks/len(result_df)*100:.2f}%")
                col4.metric("Avg Anomaly Score", f"{result_df['anomaly_score'].mean():.4f}")

                # Anomaly score timeline
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=result_df["anomaly_score"].values,
                    name="Anomaly Score",
                    line=dict(color="steelblue", width=1),
                ))
                fig.add_hline(y=0.5, line_dash="dash", line_color="red",
                             annotation_text="Detection Threshold (0.5)")
                fig.update_layout(
                    title="Anomaly Score Timeline",
                    yaxis_title="Anomaly Score",
                    xaxis_title="Sample",
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)

                # Download predictions
                csv = result_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions CSV",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv"
                )

            # Show detected attacks
            if "status" in result_df.columns:
                attacks = result_df[result_df["status"] == "ATTACK"]
                with st.expander(f"🚨 Detected Attack Samples ({len(attacks):,})"):
                    if len(attacks) > 0:
                        display_cols = [c for c in ["anomaly_score", "status"] +
                                       list(df.columns[:5]) if c in attacks.columns]
                        st.dataframe(attacks[display_cols].head(100), use_container_width=True)
                    else:
                        st.success("No attacks detected in uploaded data!")

    else:
        # Show detection on sample data
        st.subheader("Detection on Sample Data")

        if st.button("🔍 Run Detection on Sample Test Data"):
            with st.spinner("Loading and analyzing sample data..."):
                sample_df = load_sample_data(max_rows=5000)

                if not sample_df.empty:
                    result_df = detect_anomalies(sample_df, model, preprocessor_state)

                    if "anomaly_score" in result_df.columns:
                        config = load_config()
                        label_col = config.get("data", {}).get("label_col", "Attack")

                        n_true_attacks = sample_df[label_col].sum() if label_col in sample_df.columns else 0
                        n_detected = result_df["prediction"].sum() if "prediction" in result_df.columns else 0

                        col1, col2, col3 = st.columns(3)
                        col1.metric("True Attacks", f"{n_true_attacks:,}")
                        col2.metric("Detected", f"{n_detected:,}")
                        col3.metric("Mean Score", f"{result_df['anomaly_score'].mean():.4f}")

                        # Score distribution
                        fig = px.histogram(
                            result_df, x="anomaly_score",
                            color="status" if "status" in result_df.columns else None,
                            nbins=50,
                            title="Anomaly Score Distribution",
                            color_discrete_map={"Normal": "steelblue", "ATTACK": "red"}
                        )
                        st.plotly_chart(fig, use_container_width=True)


def page_digital_twin():
    st.title("🤖 Digital Twin Simulation")

    config = load_config()

    # Load twin state if available
    twin_state_path = ROOT / "outputs" / "digital_twin_state.json"
    twin_state = {}
    if twin_state_path.exists():
        with open(twin_state_path) as f:
            twin_state = json.load(f)

    # ── Current State ─────────────────────────────────────────
    st.subheader("Current System State")

    health = twin_state.get("health_score", twin_state.get("state", {}).get("health_score", 100))
    is_anomalous = twin_state.get("state", {}).get("is_anomalous", False)
    anomaly_score = twin_state.get("state", {}).get("anomaly_score", 0.0)
    alerts = twin_state.get("state", {}).get("active_alerts", [])

    col1, col2, col3, col4 = st.columns(4)

    health_color_val = "#00cc44" if health >= 80 else ("#ff8800" if health >= 50 else "#ff4444")

    with col1:
        st.metric("Health Score", f"{health:.1f}/100")
        health_bar = health / 100.0
        st.progress(health_bar, text=f"System Health: {'Good' if health >= 80 else 'Warning' if health >= 50 else 'Critical'}")

    with col2:
        status_text = "⚠️ ANOMALOUS" if is_anomalous else "✅ NORMAL"
        st.metric("Status", status_text)
        st.metric("Anomaly Score", f"{anomaly_score:.4f}")

    with col3:
        alert_count = twin_state.get("alert_count", 0)
        st.metric("Total Alerts", alert_count)
        active_scenario = twin_state.get("active_scenario", "None")
        st.metric("Active Scenario", active_scenario or "None")

    with col4:
        history_len = twin_state.get("history_length", 0)
        st.metric("History Buffer", f"{history_len} samples")
        if twin_state.get("state", {}).get("timestamp"):
            st.caption(f"Last update: {twin_state['state']['timestamp'][:19]}")

    st.divider()

    # ── Feature Trends ────────────────────────────────────────
    st.subheader("Top Deviating Sensors")

    top_sensors = twin_state.get("state", {}).get("top_deviating_sensors", [])

    if top_sensors:
        sensor_df = pd.DataFrame(top_sensors)

        fig = px.bar(
            sensor_df, x="sensor", y="z_score",
            color="z_score",
            color_continuous_scale=["blue", "yellow", "red"],
            title="Top Deviating Sensors (Z-Score from Baseline)",
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(sensor_df, use_container_width=True, hide_index=True)
    else:
        st.info("No digital twin state available. Run `python run_pipeline.py` first.")

    # ── Subsystem Health ──────────────────────────────────────
    st.divider()
    st.subheader("Subsystem Status")

    # Placeholder subsystem visualization
    subsystems_mock = {
        "P1 - Water Treatment": np.random.uniform(70, 100),
        "P2 - Secondary System": np.random.uniform(60, 100),
        "P3 - Feed Water": np.random.uniform(75, 100),
        "P4 - Steam Generation": np.random.uniform(65, 100),
    }

    if twin_state:
        subsys = twin_state.get("state", {}).get("top_deviating_sensors", [])

    sub_col1, sub_col2, sub_col3, sub_col4 = st.columns(4)

    for (subsystem, score), col in zip(subsystems_mock.items(), [sub_col1, sub_col2, sub_col3, sub_col4]):
        with col:
            status = "🟢" if score > 80 else ("🟡" if score > 60 else "🔴")
            st.metric(subsystem, f"{status} {score:.0f}%")


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

    if alert_log_path.exists():
        alert_df = pd.read_csv(alert_log_path)
    else:
        # Create example alerts
        alert_df = pd.DataFrame()

    # ── Alert Summary ─────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    if len(alert_df) > 0:
        high_alerts = len(alert_df[alert_df.get("severity", pd.Series()).eq("HIGH")]) if "severity" in alert_df.columns else 0
        med_alerts = len(alert_df[alert_df.get("severity", pd.Series()).eq("MEDIUM")]) if "severity" in alert_df.columns else 0
        low_alerts = len(alert_df[alert_df.get("severity", pd.Series()).eq("LOW")]) if "severity" in alert_df.columns else 0

        col1.metric("🔴 HIGH Alerts", high_alerts)
        col2.metric("🟠 MEDIUM Alerts", med_alerts)
        col3.metric("🟡 LOW Alerts", low_alerts)
    else:
        col1.metric("Total Alerts", 0)
        col2.metric("Status", "✅ No alerts")
        col3.metric("System", "Normal")

    st.divider()

    # ── Alert Feed ────────────────────────────────────────────
    st.subheader("Alert Feed")

    if len(alert_df) > 0 and "message" in alert_df.columns:
        for _, row in alert_df.iterrows():
            severity = row.get("severity", "LOW")
            message = row.get("message", "Unknown alert")
            timestamp = row.get("timestamp", "")
            recommendation = row.get("recommendation", "")

            css_class = f"alert-{severity.lower()}"
            st.markdown(f"""
            <div class="{css_class}">
                <strong>{timestamp[:19] if timestamp else ''}</strong> — {message}
                <br><small>💡 {recommendation}</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No alerts generated yet. Run the pipeline or upload data for detection.")

        # Show example alert format
        st.subheader("Alert Format Preview")
        example_alerts = [
            {"severity": "HIGH", "message": "[HIGH] Anomaly detected (score=0.923)",
             "recommendation": "Immediate inspection required. Consider emergency shutdown.",
             "scenario": "cyberattack"},
            {"severity": "MEDIUM", "message": "[MEDIUM] Anomaly detected (score=0.712)",
             "recommendation": "Investigate anomalous sensors. Verify readings manually.",
             "scenario": "sensor_drift"},
            {"severity": "LOW", "message": "[LOW] Anomaly detected (score=0.543)",
             "recommendation": "Monitor closely. Check sensor calibration.",
             "scenario": None},
        ]

        for alert in example_alerts:
            severity = alert["severity"]
            css_class = f"alert-{severity.lower()}"
            st.markdown(f"""
            <div class="{css_class}">
                <strong>2026-03-12 10:34:15</strong> — {alert['message']}
                <br><small>💡 {alert['recommendation']}</small>
            </div>
            """, unsafe_allow_html=True)

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
