"""
HAI Digital Twin Module — v5  (triple-ensemble: LSTM + Transformer + GRU-GAT)

Detection layers (priority order):
  Layer A   haiend LSTM-AE (w=30, 225 sensors, F1=0.6886)  — primary
  Layer A2  Transformer-AE (w=30, 225 sensors, F1=0.6795)  — ensemble partner (better ROC)
  Layer A3  GRU-GAT        (w=30, 225 sensors)             — inter-sensor graph attention
  Layer B   38-feature LSTM-AE                              — fallback if haiend unavailable
  Layer C   Physics residual (44 edges, Ridge)              — explainer + weak detector
  Layer D   Z-score baseline deviation                      — always available
  Layer E   Isolation Forest                                — unsupervised catch-all

Primary decision (Hard OR dual ensemble — F1=0.6998):
  is_anomalous = lstm_fired OR transformer_fired

Ensemble rationale:
  LSTM-AE:     highest recall  (fewest FN), captures temporal sequential patterns
  Transformer: highest ROC     (fewer FP),  captures global window relationships
  GRU-GAT:     display only    (F1=0.4704 standalone), inter-sensor per-sensor errors
               → excluded from Hard OR (adds 2108 FP for only ~200 new TP)

Each layer contributes a normalised [0,1] score.
Display score = max of deep model scores.
Confidence = number of independent layers that fired.
"""

import sys
import numpy as np
import pandas as pd
import json
import joblib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from collections import deque

from src.utils.logger import logger


# ---------------------------------------------------------------------------
# Layer weights
# ---------------------------------------------------------------------------
_W_LSTM_HAIEND = 0.60   # primary detector
_W_LSTM_38     = 0.55   # fallback if haiend absent; else 0
_W_PHYSICS     = 0.10   # weak detector / explainer
_W_ZSCORE      = 0.20   # always present
_W_ISOFOREST   = 0.10   # optional


# ---------------------------------------------------------------------------
# System State
# ---------------------------------------------------------------------------

class SystemState:
    """Represents the current estimated state of the ICS."""

    def __init__(self, feature_names: List[str]):
        self.feature_names  = feature_names
        self.n_features     = len(feature_names)
        self.current_values = np.zeros(self.n_features)
        self.baseline_mean  = np.zeros(self.n_features)
        self.baseline_std   = np.ones(self.n_features)
        self.timestamp      = datetime.now()
        self.is_anomalous   = False
        self.anomaly_score  = 0.0
        self.health_score   = 100.0
        self.active_alerts: List[str] = []
        self.attack_type:   str = "none"
        self.confidence:    str = "LOW"

    def update(self, values: np.ndarray, timestamp: datetime = None) -> None:
        self.current_values = np.array(values).flatten()[:self.n_features]
        self.timestamp = timestamp or datetime.now()

    def get_deviations(self) -> np.ndarray:
        n = min(len(self.current_values), len(self.baseline_mean))
        return (self.current_values[:n] - self.baseline_mean[:n]) / (self.baseline_std[:n] + 1e-8)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp":              str(self.timestamp),
            "health_score":           round(self.health_score, 2),
            "anomaly_score":          round(self.anomaly_score, 4),
            "is_anomalous":           bool(self.is_anomalous),
            "attack_type":            self.attack_type,
            "confidence":             self.confidence,
            "active_alerts":          self.active_alerts,
            "sensor_values":          {
                name: round(float(val), 4)
                for name, val in zip(self.feature_names[:20], self.current_values[:20])
            },
            "top_deviating_sensors":  self._get_top_deviating(5),
            "subsystem_scores":       self._subsystem_scores(),
        }

    def _get_top_deviating(self, n: int = 5) -> List[Dict]:
        devs    = self.get_deviations()
        abs_dev = np.abs(devs)
        top_idx = np.argsort(abs_dev)[::-1][:n]
        return [
            {
                "sensor":        self.feature_names[i],
                "current_value": round(float(self.current_values[i]), 4),
                "baseline_mean": round(float(self.baseline_mean[i]), 4),
                "z_score":       round(float(devs[i]), 2),
            }
            for i in top_idx if i < len(self.feature_names)
        ]

    def _subsystem_scores(self) -> Dict[str, float]:
        devs = np.abs(self.get_deviations())
        subs: Dict[str, List[float]] = {}
        for i, name in enumerate(self.feature_names):
            if i >= len(devs):
                break
            prefix = name.split("_")[0] if "_" in name else "Other"
            subs.setdefault(prefix, []).append(float(devs[i]))
        return {k: round(float(np.mean(v)), 3) for k, v in subs.items() if v}


# ---------------------------------------------------------------------------
# Digital Twin
# ---------------------------------------------------------------------------

class DigitalTwin:
    """
    HAI ICS Digital Twin — multi-model, window-based anomaly detection.

    Primary model: haiend LSTM-AE (225 sensors, window=30s, F1=0.687)
    """

    ATTACK_PATTERNS = {
        "sensor_spike":          {"pattern": "sudden_large",   "threshold_z": 8.0},
        "setpoint_manipulation": {"pattern": "sustained_med",  "threshold_z": 4.0},
        "sensor_drift":          {"pattern": "gradual",        "threshold_z": 2.5},
        "replay_attack":         {"pattern": "frozen_values",  "threshold_z": 0.5},
        "communication_loss":    {"pattern": "zero_values",    "threshold_z": 0.5},
        "equipment_degradation": {"pattern": "noisy",          "threshold_z": 2.0},
        "cyberattack":           {"pattern": "multi_sensor",   "threshold_z": 5.0},
    }

    SCENARIO_DESCRIPTIONS = {
        "attack":              "Simulated cyberattack: adversarial manipulation of sensor/actuator values",
        "sensor_failure":      "Simulated sensor failure: one or more sensors reporting incorrect values",
        "drift":               "Simulated sensor drift: gradual linear bias in readings",
        "communication_fault": "Simulated communication fault: frozen or dropped sensor readings",
        "overload":            "Simulated system overload: extreme values across pressure/flow/temperature",
        "degradation":         "Simulated equipment degradation: increasing noise + gradual drift",
        "unknown":             "Random abnormal event: combination of perturbation types",
    }

    def __init__(self, config: Dict[str, Any], feature_names: List[str] = None):
        self.config        = config
        self.twin_cfg      = config.get("digital_twin", {})
        self.feature_names = feature_names or []

        # State
        self.state = SystemState(self.feature_names)

        # Rolling history
        self.history_buffer  = deque(maxlen=3600)
        self.score_history   = deque(maxlen=300)
        self.predict_history = deque(maxlen=60)

        # Baseline
        self.baseline_mean: Optional[np.ndarray] = None
        self.baseline_std:  Optional[np.ndarray] = None

        # ── Layer A (top priority): LSTM-VAE ────────────────────────────────
        self._vae_model          = None        # LSTMVariationalAutoencoder
        self._vae_pkg            = None        # full pkg
        self._vae_n_features     = 225
        self._vae_window         = 30
        self._vae_mean           = None
        self._vae_std            = None
        self._vae_threshold      = None        # threshold for best_score_type
        self._vae_score_type     = "elbo"      # mse / kl / elbo / mu_mag
        self._vae_buffer         = deque(maxlen=30)
        self._vae_score_buf      = deque(maxlen=600)
        self._vae_last_raw: float = 0.0
        self._vae_per_sensor     = None        # (225,) last per-sensor MSE

        # ── Layer A: Multi-scale LSTM-AE (w=10+30+60) ───────────────────────
        self._ms_pkg             = None        # full multiscale joblib pkg
        self._ms_models: Dict[int, Any] = {}   # window -> LSTMAutoencoder
        self._ms_thresholds: Dict[int, float] = {}  # window -> raw MSE threshold
        self._ms_buffers: Dict[int, deque] = {} # window -> deque rolling buffer
        self._ms_score_bufs: Dict[int, deque] = {}  # window -> deque score history
        self._ms_n_features  = 225
        self._ms_mean        = None            # shared (225,) float32
        self._ms_std         = None            # shared (225,) float32

        # ── Layer A fallback: single-scale haiend LSTM-AE (w=30) ────────────
        self._haiend_model       = None        # LSTMAutoencoder object
        self._haiend_n_features  = 225
        self._haiend_window      = 30
        self._haiend_mean        = None        # (225,) float32
        self._haiend_std         = None        # (225,) float32 — already clamped ≥1.0
        self._haiend_threshold   = None        # raw MSE threshold from training
        self._haiend_buffer      = deque(maxlen=30)   # rolling (W, N) window
        self._haiend_score_buf   = deque(maxlen=600)  # running distribution
        self._haiend_per_sensor  = None        # (225,) last per-sensor errors
        self._haiend_columns: List[str] = []   # sensor column names if saved
        self._haiend_last_raw: float   = 0.0  # last raw MSE (for eval)

        # ── Layer A2: Transformer-AE (ensemble partner to LSTM-AE) ──────────
        self._tr_model           = None        # TransformerAutoencoder
        self._tr_pkg             = None
        self._tr_n_features      = 225
        self._tr_window          = 30
        self._tr_mean            = None
        self._tr_std             = None
        self._tr_threshold       = None        # raw MSE threshold
        self._tr_buffer          = deque(maxlen=30)
        self._tr_score_buf       = deque(maxlen=600)
        self._tr_last_raw: float = 0.0

        # ── Layer A3: GRU-GAT (graph-attended CNN AE) ────────────────────────
        # Captures inter-sensor dependencies via learned graph attention.
        # Architecturally different from both LSTM and Transformer.
        self._gat_model          = None        # GRUGATModel
        self._gat_pkg            = None
        self._gat_n_features     = 225
        self._gat_window         = 30
        self._gat_mean           = None
        self._gat_std            = None
        self._gat_threshold      = None        # raw MSE threshold
        self._gat_buffer         = deque(maxlen=30)
        self._gat_score_buf      = deque(maxlen=600)
        self._gat_last_raw: float = 0.0
        self._gat_per_sensor     = None        # (N,) last per-sensor errors

        # ── Layer B: 38-feature LSTM-AE fallback ────────────────────────────
        self._fallback_model     = None
        self._fallback_pkg       = None
        self._fallback_type      = None        # "autoencoder" or "lstm"

        # ── Layer C: Physics residual ────────────────────────────────────────
        self._physics_pkg        = None        # loaded joblib pkg
        self._physics_edges: List[Tuple] = []  # [(src, tgt), ...]
        self._physics_models: Dict = {}        # (src,tgt) -> Ridge
        self._physics_lag_buffers: Dict[str, deque] = {}  # sensor -> deque(maxlen=35)
        self._physics_score_buf  = deque(maxlen=600)
        self._physics_p99        = None
        self._physics_last_edges: List[Dict] = []   # last top violated edges

        # ── Layer D: Z-score ─────────────────────────────────────────────────
        # (uses baseline_mean / baseline_std — always available)

        # ── Layer E: Isolation Forest ────────────────────────────────────────
        self.anomaly_model       = None
        self.iso_threshold       = 0.5

        # Sensor name → index in feature_names (for physics lookup)
        self._sensor_idx: Dict[str, int] = {}

        # Backward-compat: keep old attribute names used by Streamlit
        self.detection_model      = None
        self.detection_model_type = "none"
        self.detection_threshold  = self.twin_cfg.get("anomaly_threshold", 0.5)
        self._ae_eng_state        = None
        self._ae_smooth_type      = "ewm"
        self._ae_smooth_window    = 30
        self._ae_score_buffer     = deque(maxlen=300)
        self._ae_score_p99        = None

        # Health / alerts
        self.health_score          = 100.0
        self.consecutive_anomalies = 0
        self.alert_log:        List[Dict] = []
        self.alert_cooldowns:  Dict[str, datetime] = {}
        self.alert_cooldown_secs = self.twin_cfg.get("alert_cooldown", 60)

        # Scenario
        self.active_scenario: Optional[str] = None
        self.scenario_data:   Optional[np.ndarray] = None
        self.scenario_idx:    int = 0

        self._outputs_dir = Path(config["paths"]["outputs"])
        self._outputs_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Digital Twin v5 initialized (triple-ensemble: LSTM + Transformer + GRU-GAT)")

    # ------------------------------------------------------------------
    # Baseline
    # ------------------------------------------------------------------

    def fit_baseline(self, normal_data: np.ndarray) -> None:
        logger.info(f"Computing baseline from {len(normal_data):,} normal samples...")
        self.baseline_mean = normal_data.mean(axis=0)
        self.baseline_std  = normal_data.std(axis=0)
        self.state.baseline_mean = self.baseline_mean
        self.state.baseline_std  = self.baseline_std
        logger.info("Baseline fitted.")

    def set_detection_model(self, model, threshold: float = None,
                             anomaly_model=None) -> None:
        self.detection_model = model
        if threshold is not None:
            self.detection_threshold = threshold
        if anomaly_model is not None:
            self.anomaly_model = anomaly_model

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_best_model(self, model_dir: str = "outputs/models") -> bool:
        """
        Auto-load the best available detection models from disk.

        Priority:
          1. haiend_lstm_detection.joblib  — primary (F1=0.687)
          2. best_detection_model.joblib   — if it exists and is better
          3. 38-feature autoencoder        — as fallback
          4. physics_residual.joblib       — as explainer / layer C
          5. isolation_forest.joblib       — as layer E
        """
        model_dir = Path(model_dir)
        loaded_primary = False

        # ── Layer A (top priority): LSTM-VAE — only if it beats LSTM-AE ────────
        # VAE trained 2026-03-17: best score=MSE, F1=0.6700 < 0.6886 baseline
        # Auto-load disabled until a VAE configuration exceeds the LSTM-AE F1
        vae_path = model_dir / "lstm_vae_detection.joblib"
        if vae_path.exists():
            try:
                from train_lstm_vae import LSTMVariationalAutoencoder as _VAE
                sys.modules["__main__"].LSTMVariationalAutoencoder = _VAE
                vae_pkg = joblib.load(vae_path)
                if isinstance(vae_pkg, dict) and \
                   vae_pkg.get("model_type") == "LSTMVariationalAutoencoder_haiend":
                    vae_f1 = vae_pkg.get("best_f1", 0.0)
                    # Only use VAE if it actually beats the LSTM-AE baseline
                    if vae_f1 > 0.6886:
                        self._vae_pkg         = vae_pkg
                        self._vae_model       = vae_pkg["model"]
                        self._vae_model.eval()
                        self._vae_n_features  = int(vae_pkg.get("n_features", 225))
                        self._vae_window      = int(vae_pkg.get("window", 30))
                        self._vae_mean        = vae_pkg["data_mean"].astype(np.float32)
                        self._vae_std         = vae_pkg["data_std"].astype(np.float32)
                        self._vae_threshold   = float(vae_pkg.get("threshold", 0.01))
                        self._vae_score_type  = vae_pkg.get("best_score_type", "elbo")
                        self._vae_buffer      = deque(maxlen=self._vae_window)
                        self._haiend_model      = self._vae_model
                        self._haiend_mean       = self._vae_mean
                        self._haiend_std        = self._vae_std
                        self._haiend_threshold  = self._vae_threshold
                        self._haiend_n_features = self._vae_n_features
                        self._haiend_window     = self._vae_window
                        self._haiend_buffer     = deque(maxlen=self._vae_window)
                        self.detection_model      = self._vae_model
                        self.detection_model_type = "lstm_vae"
                        self.detection_threshold  = self._vae_threshold
                        logger.info(
                            f"[Layer A] LSTM-VAE loaded: score={self._vae_score_type}  "
                            f"F1={vae_f1:.4f}  (beats baseline)"
                        )
                        loaded_primary = True
                    else:
                        logger.info(
                            f"[Layer A] LSTM-VAE skipped: F1={vae_f1:.4f} < 0.6886 baseline"
                        )
            except Exception as e:
                logger.warning(f"Could not load LSTM-VAE from {vae_path}: {e}")

        # ── Layer A (2nd priority): Multi-scale LSTM-AE (w=10+30+60) ────────
        if not loaded_primary:
            ms_path = model_dir / "multiscale_lstm_detection.joblib"
            if ms_path.exists():
                try:
                    from train_haiend_lstm import LSTMAutoencoder as _LSTMMs
                    sys.modules["__main__"].LSTMAutoencoder = _LSTMMs

                    ms_pkg = joblib.load(ms_path)
                    if isinstance(ms_pkg, dict) and ms_pkg.get("model_type") == "MultiScale_LSTMAe_haiend":
                        self._ms_pkg        = ms_pkg
                        self._ms_models     = ms_pkg["models"]
                        self._ms_thresholds = ms_pkg["thresholds"]
                        self._ms_mean       = ms_pkg["data_mean"].astype(np.float32)
                        self._ms_std        = ms_pkg["data_std"].astype(np.float32)
                        self._ms_n_features = int(ms_pkg.get("n_features", 225))

                        for w, mdl in self._ms_models.items():
                            mdl.eval()
                            self._ms_buffers[w]    = deque(maxlen=w)
                            self._ms_score_bufs[w] = deque(maxlen=600)

                        if 30 in self._ms_models:
                            self._haiend_model      = self._ms_models[30]
                            self._haiend_mean       = self._ms_mean
                            self._haiend_std        = self._ms_std
                            self._haiend_threshold  = self._ms_thresholds.get(30, 0.01)
                            self._haiend_n_features = self._ms_n_features
                            self._haiend_window     = 30
                            self._haiend_buffer     = deque(maxlen=30)
                            self.detection_model      = self._haiend_model
                            self.detection_model_type = "lstm_multiscale"
                            self.detection_threshold  = self._haiend_threshold

                        best_f1 = ms_pkg.get("best_ensemble_f1", ms_pkg.get("w30_baseline_f1", 0))
                        logger.info(
                            f"[Layer A] Multi-scale LSTM-AE: scales={sorted(self._ms_models.keys())}  "
                            f"n_feat={self._ms_n_features}  best_F1={best_f1:.4f}"
                        )
                        loaded_primary = True
                except Exception as e:
                    logger.warning(f"Could not load multiscale LSTM from {ms_path}: {e}")

        # ── Layer A fallback: single-scale haiend LSTM-AE (w=30) ────────────
        if not loaded_primary:
            for fname in ["haiend_lstm_detection.joblib", "best_detection_model.joblib"]:
                path = model_dir / fname
                if not path.exists():
                    continue
                try:
                    from train_haiend_lstm import LSTMAutoencoder as _LSTmHaiend
                    sys.modules["__main__"].LSTMAutoencoder = _LSTmHaiend

                    pkg = joblib.load(path)
                    if not isinstance(pkg, dict):
                        continue
                    if pkg.get("model_type") != "LSTMAutoencoder_haiend":
                        continue

                    self._haiend_model      = pkg["model"]
                    self._haiend_model.eval()
                    self._haiend_n_features = int(pkg.get("n_features", 225))
                    self._haiend_window     = int(pkg.get("window", 30))
                    self._haiend_mean       = pkg["data_mean"].astype(np.float32)
                    self._haiend_std        = pkg["data_std"].astype(np.float32)
                    self._haiend_threshold  = float(pkg.get("threshold", 0.01))
                    self._haiend_buffer     = deque(maxlen=self._haiend_window)
                    self._haiend_columns    = pkg.get("columns", [])

                    self.detection_model      = self._haiend_model
                    self.detection_model_type = "lstm_haiend"
                    self.detection_threshold  = self._haiend_threshold

                    logger.info(
                        f"[Layer A] haiend LSTM-AE loaded: {path.name}  "
                        f"n_feat={self._haiend_n_features}  window={self._haiend_window}  "
                        f"F1={pkg.get('best_f1', 0):.4f}"
                    )
                    loaded_primary = True
                    break
                except Exception as e:
                    logger.warning(f"Could not load haiend LSTM from {path}: {e}")

        # ── Layer A2: Transformer-AE ensemble partner ────────────────────────
        tr_path = model_dir / "transformer_ae_detection.joblib"
        if tr_path.exists() and loaded_primary:
            try:
                from train_anomaly_transformer import TransformerAutoencoder as _Tr
                sys.modules["__main__"].TransformerAutoencoder = _Tr
                tr_pkg = joblib.load(tr_path)
                if isinstance(tr_pkg, dict) and \
                   tr_pkg.get("model_type") == "TransformerAutoencoder_haiend":
                    self._tr_pkg         = tr_pkg
                    self._tr_model       = tr_pkg["model"]
                    self._tr_model.eval()
                    self._tr_n_features  = int(tr_pkg.get("n_features", 225))
                    self._tr_window      = int(tr_pkg.get("window", 30))
                    self._tr_mean        = tr_pkg["data_mean"].astype(np.float32)
                    self._tr_std         = tr_pkg["data_std"].astype(np.float32)
                    self._tr_threshold   = float(tr_pkg.get("threshold", 0.001))
                    self._tr_buffer      = deque(maxlen=self._tr_window)
                    logger.info(
                        f"[Layer A2] Transformer-AE loaded: "
                        f"n_feat={self._tr_n_features}  window={self._tr_window}  "
                        f"F1={tr_pkg.get('best_f1', 0):.4f}  "
                        f"(ensemble with LSTM: F1~0.6998)"
                    )
            except Exception as e:
                logger.warning(f"Could not load Transformer-AE: {e}")

        # ── Layer A3: GRU-GAT (graph-attended CNN autoencoder) ───────────────
        gat_path = model_dir / "gru_gat_detection.joblib"
        if gat_path.exists() and loaded_primary:
            try:
                from train_gru_gat import GRUGATModel as _GRUGATModel
                sys.modules["__main__"].GRUGATModel = _GRUGATModel
                gat_pkg = joblib.load(gat_path)
                if isinstance(gat_pkg, dict) and \
                   gat_pkg.get("model_type") == "GRUGATModel_haiend":
                    self._gat_pkg         = gat_pkg
                    self._gat_model       = gat_pkg["model"]
                    self._gat_model.eval()
                    self._gat_n_features  = int(gat_pkg.get("n_features", 225))
                    self._gat_window      = int(gat_pkg.get("window", 30))
                    self._gat_mean        = gat_pkg["data_mean"].astype(np.float32)
                    self._gat_std         = gat_pkg["data_std"].astype(np.float32)
                    self._gat_threshold   = float(gat_pkg.get("threshold", 0.001))
                    self._gat_buffer      = deque(maxlen=self._gat_window)
                    gat_f1 = float(gat_pkg.get("best_f1", 0))
                    logger.info(
                        f"[Layer A3] GRU-GAT loaded: "
                        f"n_feat={self._gat_n_features}  window={self._gat_window}  "
                        f"F1={gat_f1:.4f}  (graph-attention inter-sensor ensemble)"
                    )
            except Exception as e:
                logger.warning(f"Could not load GRU-GAT: {e}")

        # ── Layer B: 38-feature LSTM-AE fallback ────────────────────────────
        for fname in ["lstm_ae_detection.joblib"]:
            path = model_dir / fname
            if not path.exists():
                continue
            try:
                from retrain_lstm_ae import LSTMAutoencoder as _LSTM38
                sys.modules["__main__"].LSTMAutoencoder = _LSTM38

                pkg38 = joblib.load(path)
                if isinstance(pkg38, dict) and "model" in pkg38:
                    self._fallback_model = pkg38["model"]
                    self._fallback_model.eval()
                    self._fallback_pkg   = pkg38
                    self._fallback_type  = "lstm"
                    if not loaded_primary:
                        self.detection_model      = self._fallback_model
                        self.detection_model_type = "autoencoder"
                        self._ae_eng_state        = pkg38.get("eng_state", None)
                    logger.info(
                        f"[Layer B] 38-feat LSTM-AE loaded: {fname}  "
                        f"F1={pkg38.get('best_f1', 0):.4f}"
                    )
            except Exception as e:
                logger.warning(f"Could not load 38-feat LSTM from {fname}: {e}")

        # ── Layer C: Physics residual ────────────────────────────────────────
        phys_path = model_dir / "physics_residual.joblib"
        if phys_path.exists():
            try:
                self._physics_pkg    = joblib.load(phys_path)
                self._physics_edges  = self._physics_pkg.get("edges", [])
                self._physics_models = self._physics_pkg.get("models", {})
                # Build per-sensor lag buffers for all sensors referenced by edges
                all_sensors = set()
                for (s, t) in self._physics_edges:
                    all_sensors.add(s)
                    all_sensors.add(t)
                for s in all_sensors:
                    self._physics_lag_buffers[s] = deque(maxlen=35)
                logger.info(
                    f"[Layer C] Physics residual loaded: {len(self._physics_edges)} edges, "
                    f"{len(all_sensors)} sensors"
                )
            except Exception as e:
                logger.warning(f"Could not load physics model: {e}")

        # ── Layer E: Isolation Forest ────────────────────────────────────────
        iso_path = model_dir / "isolation_forest.joblib"
        if iso_path.exists():
            try:
                self.anomaly_model = joblib.load(iso_path)
                logger.info("[Layer E] Isolation Forest loaded")
            except Exception as e:
                logger.warning(f"Could not load IsoForest: {e}")

        # Build sensor→index lookup
        self._sensor_idx = {name: i for i, name in enumerate(self.feature_names)}

        return loaded_primary or (self._fallback_model is not None)

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    def ingest(self, sample: np.ndarray, timestamp: datetime = None) -> Dict[str, Any]:
        """Process one sensor reading through all detection layers."""
        sample = np.array(sample).flatten()

        # Scenario injection
        if self.active_scenario is not None and self.scenario_data is not None:
            if self.scenario_idx < len(self.scenario_data):
                pert = self.scenario_data[self.scenario_idx]
                n = min(len(sample), len(pert))
                sample[:n] = pert[:n]
                self.scenario_idx += 1
            else:
                self.stop_scenario()

        self.state.update(sample, timestamp)
        self.history_buffer.append(sample.copy())

        # Multi-layer detection
        anomaly_score, is_anomalous, confidence, layers, layer_scores = \
            self._detect_anomaly(sample)

        self.score_history.append(anomaly_score)
        self.predict_history.append(int(is_anomalous))

        self.state.anomaly_score = anomaly_score
        self.state.is_anomalous  = is_anomalous
        self.state.confidence    = confidence

        if is_anomalous:
            self.consecutive_anomalies += 1
            self.state.attack_type = self._classify_attack(sample)
        else:
            self.consecutive_anomalies = 0
            self.state.attack_type = "none"

        self.health_score       = self._update_health(is_anomalous, anomaly_score)
        self.state.health_score = self.health_score

        alerts = self._check_alerts(sample, is_anomalous, anomaly_score, layers)
        self.state.active_alerts = [a["message"] for a in alerts]

        result = self.state.to_dict()
        result["layers_active"]         = layers
        result["layer_scores"]          = layer_scores
        result["consecutive_anomalies"] = self.consecutive_anomalies
        return result

    def process_batch(self, data: np.ndarray,
                      timestamps: List[datetime] = None) -> pd.DataFrame:
        results = []
        for i, sample in enumerate(data):
            ts = timestamps[i] if timestamps and i < len(timestamps) else None
            r  = self.ingest(sample, timestamp=ts)
            r["sample_idx"] = i
            results.append(r)
        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Layer A2 — Transformer-AE (ensemble partner)
    # ------------------------------------------------------------------

    def _transformer_score(self, sample: np.ndarray
                           ) -> Tuple[float, bool]:
        """
        Score using Transformer-AE.

        Returns
        -------
        norm_score : float  0-2 normalised by running p97 (for display)
        above_thr  : bool   raw MSE >= model threshold
        """
        try:
            import torch
            N = self._tr_n_features
            W = self._tr_window

            n_in = len(sample)
            vec  = sample[:N].astype(np.float32) if n_in >= N else \
                   np.pad(sample.astype(np.float32), (0, N - n_in))

            self._tr_buffer.append(vec)
            buf = list(self._tr_buffer)
            if len(buf) < W:
                buf = [np.zeros(N, dtype=np.float32)] * (W - len(buf)) + buf
            window_arr  = np.array(buf[-W:], dtype=np.float32)
            window_norm = (window_arr - self._tr_mean) / self._tr_std

            x = torch.from_numpy(window_norm[np.newaxis])   # (1, W, N)
            self._tr_model.eval()
            raw_score = float(self._tr_model.reconstruction_error(x)[0])
            self._tr_last_raw = raw_score
            self._tr_score_buf.append(raw_score)

            above_thr = raw_score >= self._tr_threshold

            if len(self._tr_score_buf) >= 20:
                p97 = float(np.percentile(list(self._tr_score_buf), 97))
            else:
                p97 = max(self._tr_threshold * 3.0, 1e-6)
            norm_score = float(np.clip(raw_score / (p97 + 1e-8), 0.0, 2.0))

            return norm_score, above_thr

        except Exception as e:
            logger.debug(f"Transformer score error: {e}")
            return 0.0, False

    # ------------------------------------------------------------------
    # Layer A3 — GRU-GAT (graph-attended CNN autoencoder)
    # ------------------------------------------------------------------

    def _gru_gat_score(self, sample: np.ndarray) -> Tuple[float, Optional[np.ndarray], bool]:
        """
        Score using GRU-GAT (inter-sensor graph attention model).

        Returns
        -------
        norm_score  : float  0-1 (normalised by running p97, for display)
        per_sensor  : ndarray (N,) per-sensor MSE, or None
        above_thr   : bool   raw MSE >= pre-calibrated threshold
        """
        try:
            import torch
            N = self._gat_n_features
            W = self._gat_window

            n_in = len(sample)
            vec  = sample[:N].astype(np.float32) if n_in >= N else \
                   np.pad(sample.astype(np.float32), (0, N - n_in))

            self._gat_buffer.append(vec)
            buf = list(self._gat_buffer)
            if len(buf) < W:
                buf = [np.zeros(N, dtype=np.float32)] * (W - len(buf)) + buf
            window_arr  = np.array(buf[-W:], dtype=np.float32)
            window_norm = (window_arr - self._gat_mean) / self._gat_std

            x = torch.from_numpy(window_norm[np.newaxis])   # (1, W, N)
            self._gat_model.eval()
            with torch.no_grad():
                per_sensor = self._gat_model.per_sensor_error(x)[0]  # (N,)
                raw_score  = float(per_sensor.mean())

            self._gat_per_sensor = per_sensor
            self._gat_last_raw   = raw_score
            self._gat_score_buf.append(raw_score)

            # Update shared root-cause sensor errors (GAT is more precise for
            # relational anomalies than LSTM reconstruction)
            if self._haiend_per_sensor is None:
                self._haiend_per_sensor = per_sensor

            above_thr = raw_score >= self._gat_threshold

            if len(self._gat_score_buf) >= 20:
                p97 = float(np.percentile(list(self._gat_score_buf), 97))
            else:
                p97 = max(self._gat_threshold * 3.0, 1e-6)
            norm_score = float(np.clip(raw_score / (p97 + 1e-8), 0.0, 2.0))

            return norm_score, per_sensor, above_thr

        except Exception as e:
            logger.debug(f"GRU-GAT score error: {e}")
            return 0.0, None, False

    # ------------------------------------------------------------------
    # Layer A — LSTM-VAE (probabilistic latent space)
    # ------------------------------------------------------------------

    def _lstm_vae_score(self, sample: np.ndarray
                        ) -> Tuple[float, Optional[np.ndarray], bool]:
        """
        Score using LSTM-VAE. Anomaly signal = best_score_type from training.

        Returns
        -------
        norm_score  : float  0-1 (normalised by running p97, for display)
        per_sensor  : ndarray (225,) per-sensor MSE from reconstruction
        above_thr   : bool   score >= pre-calibrated threshold
        """
        try:
            import torch
            N = self._vae_n_features
            W = self._vae_window

            n_in = len(sample)
            vec  = sample[:N].astype(np.float32) if n_in >= N else \
                   np.pad(sample.astype(np.float32), (0, N - n_in))

            self._vae_buffer.append(vec)
            buf = list(self._vae_buffer)
            if len(buf) < W:
                buf = [np.zeros(N, dtype=np.float32)] * (W - len(buf)) + buf
            window_arr  = np.array(buf[-W:], dtype=np.float32)
            window_norm = (window_arr - self._vae_mean) / self._vae_std

            x = torch.from_numpy(window_norm[np.newaxis])   # (1, W, N)
            self._vae_model.eval()

            score_type = self._vae_score_type
            if score_type == "kl":
                raw_score = float(self._vae_model.kl_score(x)[0])
            elif score_type == "elbo":
                raw_score = float(self._vae_model.elbo_score(x, beta=1.0)[0])
            elif score_type == "mu_mag":
                raw_score = float(self._vae_model.mu_magnitude(x)[0])
            else:  # mse
                raw_score = float(self._vae_model.reconstruction_error(x)[0])

            # Per-sensor MSE for root cause
            per_sensor = self._vae_model.per_sensor_error(x)[0]  # (N,)
            self._vae_per_sensor     = per_sensor
            self._haiend_per_sensor  = per_sensor  # expose for root cause analysis
            self._vae_last_raw       = raw_score
            self._haiend_last_raw    = raw_score

            self._vae_score_buf.append(raw_score)
            self._haiend_score_buf.append(raw_score)

            above_thr = raw_score >= self._vae_threshold

            # Running p97 for display
            if len(self._vae_score_buf) >= 20:
                p97 = float(np.percentile(list(self._vae_score_buf), 97))
            else:
                p97 = max(self._vae_threshold * 3.0, 1e-6)
            norm_score = float(np.clip(raw_score / (p97 + 1e-8), 0.0, 2.0))

            return norm_score, per_sensor, above_thr

        except Exception as e:
            logger.debug(f"VAE score error: {e}")
            return 0.0, None, False

    # ------------------------------------------------------------------
    # Layer A — Multi-scale LSTM-AE (w=10 + w=30 + w=60)
    # ------------------------------------------------------------------

    def _lstm_multiscale_score(self, sample: np.ndarray
                               ) -> Tuple[float, Optional[np.ndarray], bool]:
        """
        Score using all 3 scales. Returns aggregate norm score, w=30 per-sensor
        errors (for root cause), and fired=True if ANY scale fires.

        Returns
        -------
        norm_score  : float  max of normalised scores across scales (display)
        per_sensor  : ndarray (225,) from w=30 (most interpretable)
        fired       : bool   True if any scale's raw MSE >= its threshold
        """
        try:
            import torch
            N = self._ms_n_features

            # Align sample to N features
            n_in = len(sample)
            if n_in >= N:
                vec = sample[:N].astype(np.float32)
            else:
                vec = np.zeros(N, dtype=np.float32)
                vec[:n_in] = sample[:n_in].astype(np.float32)

            per_sensor_w30 = None
            norm_scores    = {}
            scale_fired    = {}

            for w, model in self._ms_models.items():
                buf = self._ms_buffers[w]
                buf.append(vec)

                # Build padded window
                buf_list = list(buf)
                if len(buf_list) < w:
                    pad      = [np.zeros(N, dtype=np.float32)] * (w - len(buf_list))
                    buf_list = pad + buf_list
                window_arr  = np.array(buf_list[-w:], dtype=np.float32)   # (W, N)
                window_norm = (window_arr - self._ms_mean) / self._ms_std

                x = torch.from_numpy(window_norm[np.newaxis])             # (1, W, N)
                model.eval()
                with torch.no_grad():
                    recon      = model(x)                                  # (1, W, N)
                    sq_err     = (x - recon) ** 2                          # (1, W, N)
                    per_sensor = sq_err.mean(dim=1).squeeze(0).cpu().numpy()  # (N,)
                    raw_score  = float(per_sensor.mean())

                self._ms_score_bufs[w].append(raw_score)

                # Fire decision: pre-calibrated absolute threshold
                thr = self._ms_thresholds.get(w, 0.01)
                scale_fired[w] = raw_score >= thr

                # Running p97 normalisation for display
                sbuf = self._ms_score_bufs[w]
                if len(sbuf) >= 20:
                    p97 = float(np.percentile(list(sbuf), 97))
                else:
                    p97 = max(thr * 3.0, 1e-6)
                norm_scores[w] = float(np.clip(raw_score / (p97 + 1e-8), 0.0, 2.0))

                if w == 30:
                    per_sensor_w30           = per_sensor
                    self._haiend_per_sensor  = per_sensor
                    self._haiend_last_raw    = raw_score
                    self._haiend_score_buf.append(raw_score)

            # Decision: use w=30 threshold only (OR logic adds FP faster than it reduces FN)
            # w=10 and w=60 contribute only to the display score
            fired      = scale_fired.get(30, False)
            norm_score = float(max(norm_scores.values())) if norm_scores else 0.0

            fired_scales = [w for w, f in scale_fired.items() if f]
            if fired_scales:
                logger.debug(f"Multi-scale fired: {fired_scales}")

            return norm_score, per_sensor_w30, fired

        except Exception as e:
            logger.debug(f"Multiscale LSTM score error: {e}")
            return 0.0, None, False

    # ------------------------------------------------------------------
    # Layer A fallback — single-scale haiend LSTM-AE (window-based)
    # ------------------------------------------------------------------

    def _lstm_haiend_score(self, sample: np.ndarray
                           ) -> Tuple[float, Optional[np.ndarray], bool]:
        """
        Score using the 225-sensor LSTM-AE.

        Returns
        -------
        norm_score  : float  0-1 (normalised by running p97)
        per_sensor  : ndarray (225,) per-sensor MSE, or None
        above_thr   : bool   raw MSE >= model threshold
        """
        try:
            import torch
            N = self._haiend_n_features
            W = self._haiend_window

            # Map incoming sample to N-dim vector
            n_in = len(sample)
            if n_in >= N:
                vec = sample[:N].astype(np.float32)
            else:
                vec = np.zeros(N, dtype=np.float32)
                vec[:n_in] = sample[:n_in].astype(np.float32)

            self._haiend_buffer.append(vec)

            # Build window (pad front with zeros if < W samples seen)
            buf = list(self._haiend_buffer)
            if len(buf) < W:
                pad = [np.zeros(N, dtype=np.float32)] * (W - len(buf))
                buf = pad + buf
            window_arr = np.array(buf[-W:], dtype=np.float32)   # (W, N)

            # Normalise
            window_norm = (window_arr - self._haiend_mean) / self._haiend_std

            # Forward pass
            x = torch.from_numpy(window_norm[np.newaxis])        # (1, W, N)
            self._haiend_model.eval()
            with torch.no_grad():
                recon      = self._haiend_model(x)               # (1, W, N)
                sq_err     = (x - recon) ** 2                    # (1, W, N)
                per_sensor = sq_err.mean(dim=1).squeeze(0).cpu().numpy()  # (N,)
                raw_score  = float(per_sensor.mean())

            self._haiend_per_sensor = per_sensor
            self._haiend_score_buf.append(raw_score)
            self._haiend_last_raw  = raw_score

            # Absolute threshold from training (pre-calibrated for F1=0.687)
            above_thr = raw_score >= self._haiend_threshold

            # Running p97 normalisation — used only for display / weighted score
            if len(self._haiend_score_buf) >= 20:
                p97 = float(np.percentile(list(self._haiend_score_buf), 97))
            else:
                p97 = max(self._haiend_threshold * 3.0, 1e-6)

            norm_score = float(np.clip(raw_score / (p97 + 1e-8), 0.0, 2.0))

            return norm_score, per_sensor, above_thr

        except Exception as e:
            logger.debug(f"haiend LSTM score error: {e}")
            return 0.0, None, False

    # ------------------------------------------------------------------
    # Layer B — 38-feature autoencoder fallback
    # ------------------------------------------------------------------

    def _fallback_score(self, sample: np.ndarray) -> float:
        """Score using 38-feature LSTM-AE fallback. Returns normalised [0,1]."""
        try:
            import torch
            pkg = self._fallback_pkg
            W   = pkg["window"]
            mean = pkg["data_mean"]
            std  = pkg["data_std"]
            N   = len(mean)

            n_in = len(sample)
            vec  = sample[:N].astype(np.float32) if n_in >= N else \
                   np.pad(sample.astype(np.float32), (0, N - n_in))

            # Use history buffer for the window
            hist  = [np.array(h[:N], dtype=np.float32) for h in list(self.history_buffer)[-(W-1):]]
            hist.append(vec)
            if len(hist) < W:
                pad  = [np.zeros(N, dtype=np.float32)] * (W - len(hist))
                hist = pad + hist
            window_arr  = np.array(hist[-W:], dtype=np.float32)  # (W, N)
            window_norm = (window_arr - mean) / (std + 1e-8)

            x = torch.from_numpy(window_norm[np.newaxis])         # (1, W, N)
            self._fallback_model.eval()
            with torch.no_grad():
                err = float(self._fallback_model.reconstruction_error(x)[0])

            self._ae_score_buffer.append(err)
            if len(self._ae_score_buffer) >= 10:
                self._ae_score_p99 = float(np.percentile(list(self._ae_score_buffer), 97))
            p99 = self._ae_score_p99 or max(err * 2, 1e-6)
            return float(np.clip(err / (p99 + 1e-8), 0.0, 1.5))

        except Exception as e:
            logger.debug(f"Fallback score error: {e}")
            return 0.0

    # ------------------------------------------------------------------
    # Layer C — Physics residual
    # ------------------------------------------------------------------

    def _physics_score(self, sample: np.ndarray) -> Tuple[float, List[Dict]]:
        """
        Compute physics residual score.

        Returns
        -------
        norm_score   : float 0-1
        top_edges    : list of dicts describing most-violated physical edges
        """
        if not self._physics_models:
            return 0.0, []
        try:
            # Get sensor value by name
            def get_val(name: str) -> float:
                idx = self._sensor_idx.get(name, -1)
                if 0 <= idx < len(sample):
                    return float(sample[idx])
                # Try haiend buffer (last sample)
                if self._haiend_columns and name in self._haiend_columns:
                    col_idx = self._haiend_columns.index(name)
                    if self._haiend_buffer:
                        return float(list(self._haiend_buffer)[-1][col_idx])
                return 0.0

            # Update lag buffers
            for name in self._physics_lag_buffers:
                self._physics_lag_buffers[name].append(get_val(name))

            lags = [0, 1, 2, 5, 10, 30]
            residuals = []
            for (src, tgt) in self._physics_edges:
                mdl = self._physics_models.get((src, tgt))
                if mdl is None:
                    continue
                buf = list(self._physics_lag_buffers.get(src, []))
                if len(buf) < 31:
                    continue
                feats = np.array(
                    [buf[-1 - lag] if lag < len(buf) else buf[0] for lag in lags],
                    dtype=np.float64
                ).reshape(1, -1)
                pred   = float(mdl.predict(feats)[0])
                actual = get_val(tgt)
                residuals.append({
                    "edge":     f"{src}→{tgt}",
                    "src":      src,
                    "tgt":      tgt,
                    "actual":   round(actual, 4),
                    "predicted": round(pred, 4),
                    "residual": abs(actual - pred),
                })

            if not residuals:
                return 0.0, []

            resid_vals = np.array([r["residual"] for r in residuals])
            mean_resid = float(resid_vals.mean())

            self._physics_score_buf.append(mean_resid)
            if len(self._physics_score_buf) >= 50:
                self._physics_p99 = float(
                    np.percentile(list(self._physics_score_buf), 97))
            p99 = self._physics_p99 or max(mean_resid * 3, 1e-6)

            norm_score = float(np.clip(mean_resid / (p99 + 1e-8), 0.0, 1.5))
            top_edges  = sorted(residuals, key=lambda x: -x["residual"])[:5]
            self._physics_last_edges = top_edges
            return norm_score, top_edges

        except Exception as e:
            logger.debug(f"Physics score error: {e}")
            return 0.0, []

    # ------------------------------------------------------------------
    # Layer D — Z-score
    # ------------------------------------------------------------------

    def _zscore_score(self, sample: np.ndarray) -> float:
        if self.baseline_mean is None:
            return 0.0
        n  = min(len(sample), len(self.baseline_mean))
        z  = np.abs((sample[:n] - self.baseline_mean[:n]) / (self.baseline_std[:n] + 1e-8))
        p90 = float(np.percentile(z, 90))
        return float(np.clip(p90 / 10.0, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Combined detection
    # ------------------------------------------------------------------

    def _detect_anomaly(self, sample: np.ndarray
                        ) -> Tuple[float, bool, str, List[str], Dict[str, float]]:
        """
        Run all available layers and combine into a single anomaly decision.

        Returns
        -------
        anomaly_score  : float  0-1
        is_anomalous   : bool
        confidence     : str    HIGH / MEDIUM / LOW
        layers_fired   : list[str]
        layer_scores   : dict   raw [0-1] score per layer name
        """
        scores:       Dict[str, float] = {}
        weights:      Dict[str, float] = {}
        layers_fired: List[str]        = []

        # Layer A — VAE (top priority) → Multi-scale → Single-scale haiend
        #           + Layer A2: Transformer-AE ensemble (OR logic)
        tr_norm_score = 0.0
        if self._vae_model is not None:
            ns, _per, above = self._lstm_vae_score(sample)
            scores["lstm_haiend"]     = ns
            scores["lstm_haiend_raw"] = getattr(self, "_vae_last_raw", 0.0)
            weights["lstm_haiend"]    = _W_LSTM_HAIEND
            if above:
                layers_fired.append("LSTM-haiend")
        elif self._ms_pkg is not None and self._ms_models:
            ns, _per, fired = self._lstm_multiscale_score(sample)
            scores["lstm_haiend"]     = ns
            scores["lstm_haiend_raw"] = getattr(self, "_haiend_last_raw", 0.0)
            weights["lstm_haiend"]    = _W_LSTM_HAIEND
            if fired:
                layers_fired.append("LSTM-haiend")
        elif self._haiend_model is not None:
            ns, _per, above = self._lstm_haiend_score(sample)
            scores["lstm_haiend"]     = ns
            scores["lstm_haiend_raw"] = getattr(self, "_haiend_last_raw", 0.0)
            weights["lstm_haiend"]    = _W_LSTM_HAIEND
            if above:
                layers_fired.append("LSTM-haiend")

        # Layer A2 — Transformer-AE (ensemble partner, OR logic with LSTM)
        if self._tr_model is not None:
            tr_ns, tr_above = self._transformer_score(sample)
            scores["transformer"]  = tr_ns
            weights["transformer"] = _W_LSTM_HAIEND * 0.8   # slightly lower weight for display
            if tr_above:
                layers_fired.append("Transformer")
            tr_norm_score = tr_ns

        # Layer A3 — GRU-GAT (inter-sensor graph attention, OR logic)
        if self._gat_model is not None:
            gat_ns, _gat_per, gat_above = self._gru_gat_score(sample)
            scores["gru_gat"]  = gat_ns
            weights["gru_gat"] = _W_LSTM_HAIEND * 0.75  # display weight
            if gat_above:
                layers_fired.append("GRU-GAT")

        # Update display score to max of all active deep models
        deep_keys = [k for k in ("lstm_haiend", "transformer", "gru_gat") if k in scores]
        if deep_keys:
            scores["lstm_haiend"] = max(scores[k] for k in deep_keys)

        # Layer B — 38-feat fallback (only if haiend absent)
        elif self._fallback_model is not None:
            fs = self._fallback_score(sample)
            scores["lstm_fallback"]  = fs
            weights["lstm_fallback"] = _W_LSTM_38
            if fs >= 0.55:
                layers_fired.append("LSTM-fallback")

        # Layer C — Physics residual
        if self._physics_models:
            ps, _edges = self._physics_score(sample)
            scores["physics"]  = ps
            weights["physics"] = _W_PHYSICS
            if ps >= 0.65:
                layers_fired.append("Physics")

        # Layer D — Z-score (always)
        zs = self._zscore_score(sample)
        scores["zscore"]  = zs
        weights["zscore"] = _W_ZSCORE
        if zs >= 0.30:
            layers_fired.append("Z-score")

        # Layer E — Isolation Forest
        if self.anomaly_model is not None:
            try:
                iso_raw   = -self.anomaly_model.score_samples(sample.reshape(1, -1))[0]
                iso_score = float(np.clip(iso_raw / (self.iso_threshold * 2 + 1e-8), 0, 1))
                scores["isolation"]  = iso_score
                weights["isolation"] = _W_ISOFOREST
                if iso_raw >= self.iso_threshold:
                    layers_fired.append("IsoForest")
            except Exception:
                pass

        # Weighted combination
        total_w = sum(weights[k] for k in scores if k in weights)
        total_s = sum(scores[k] * weights.get(k, 0) for k in scores)
        final   = total_s / total_w if total_w > 0 else 0.0
        final   = float(np.clip(final, 0.0, 1.0))

        n_fired    = len(layers_fired)
        confidence = "HIGH" if n_fired >= 2 else ("MEDIUM" if n_fired == 1 else "LOW")

        # Primary decision: LSTM OR Transformer (Hard OR, F1=0.6998)
        # GRU-GAT (F1=0.4704 standalone) is EXCLUDED from the Hard OR decision —
        # its 2108 FPs add more noise than its ~200 unique TPs are worth.
        # GRU-GAT is used only for per-sensor contribution analysis (root cause).
        lstm_fired        = "LSTM-haiend" in layers_fired or "LSTM-fallback" in layers_fired
        transformer_fired = "Transformer" in layers_fired
        gat_fired         = "GRU-GAT" in layers_fired   # tracked for display only
        is_anomalous      = lstm_fired or transformer_fired
        # Note: if GRU-GAT improves significantly in future training, re-enable below:
        # is_anomalous = lstm_fired or transformer_fired or gat_fired

        return final, bool(is_anomalous), confidence, layers_fired, scores

    # ------------------------------------------------------------------
    # Health scoring
    # ------------------------------------------------------------------

    def _update_health(self, is_anomalous: bool, anomaly_score: float) -> float:
        if is_anomalous:
            accel = min(3.0, 1.0 + self.consecutive_anomalies * 0.1)
            drop  = anomaly_score * 8.0 * accel
            self.health_score = max(0.0, self.health_score - drop)
        else:
            gap      = 100.0 - self.health_score
            recovery = gap * 0.05
            self.health_score = min(100.0, self.health_score + max(recovery, 0.2))
        return round(self.health_score, 2)

    # ------------------------------------------------------------------
    # Attack classification
    # ------------------------------------------------------------------

    def _classify_attack(self, sample: np.ndarray) -> str:
        if self.baseline_mean is None:
            return "unknown"
        n      = min(len(sample), len(self.baseline_mean))
        z      = (sample[:n] - self.baseline_mean[:n]) / (self.baseline_std[:n] + 1e-8)
        abs_z  = np.abs(z)
        max_z  = float(abs_z.max()) if len(abs_z) > 0 else 0.0
        n_high = int((abs_z > 3.0).sum())

        # Communication loss: sensors unexpectedly drop to zero relative to their baseline
        # (Don't flag sensors that are normally zero — e.g. constant DCS binary states)
        baseline_nonzero = np.abs(self.baseline_mean[:n]) > 0.1
        if baseline_nonzero.sum() > 5:
            dropped = (np.abs(sample[:n]) < 0.01) & baseline_nonzero
            drop_rate = float(dropped.sum()) / float(baseline_nonzero.sum())
            if drop_rate > 0.5:
                return "communication_loss"

        if len(self.history_buffer) >= 5:
            recent = np.array(list(self.history_buffer)[-5:])
            if recent.shape[0] >= 2 and np.abs(np.diff(recent, axis=0)).mean() < 0.001:
                return "replay_attack"

        if max_z > 10.0:           return "sensor_spike"
        if max_z > 5.0 and n_high >= 3: return "cyberattack"
        if max_z > 5.0:            return "setpoint_manipulation"
        if max_z > 2.5 and n_high >= 5: return "equipment_degradation"
        return "sensor_drift"

    # ------------------------------------------------------------------
    # Alert engine
    # ------------------------------------------------------------------

    def _check_alerts(self, sample: np.ndarray, is_anomalous: bool,
                       anomaly_score: float, layers: List[str]) -> List[Dict]:
        if not is_anomalous:
            return []

        alerts  = []
        now     = datetime.now()
        sev_cfg = self.twin_cfg.get("severity_levels", {})

        if anomaly_score >= sev_cfg.get("high", 0.80) or self.consecutive_anomalies >= 10:
            severity = "CRITICAL"
            recommendation = (
                "Immediate action required. Verify physical plant. "
                "Consider emergency shutdown if readings cannot be explained."
            )
        elif anomaly_score >= sev_cfg.get("medium", 0.60) or self.consecutive_anomalies >= 5:
            severity = "HIGH"
            recommendation = (
                "Investigate anomalous sensors now. "
                "Check control loops and recent operator commands."
            )
        elif anomaly_score >= sev_cfg.get("low", 0.40):
            severity = "MEDIUM"
            recommendation = "Monitor closely. Verify sensor calibration and network integrity."
        else:
            severity = "LOW"
            recommendation = "Minor deviation detected. Log and continue monitoring."

        attack_type = self.state.attack_type
        confidence  = self.state.confidence
        alert_key   = f"{severity}_{attack_type}"
        last_alert  = self.alert_cooldowns.get(alert_key)

        if last_alert and (now - last_alert).total_seconds() < self.alert_cooldown_secs:
            return alerts

        root_cause = self.analyze_root_cause(sample)
        top_sensor = (root_cause.get("top_sensors") or [{}])[0].get("sensor", "unknown")

        alert = {
            "timestamp":         str(now),
            "severity":          severity,
            "confidence":        confidence,
            "anomaly_score":     round(anomaly_score, 4),
            "attack_type":       attack_type,
            "layers_detected":   layers,
            "message": (
                f"[{severity}] {attack_type.replace('_', ' ').title()} detected "
                f"(score={anomaly_score:.3f}, confidence={confidence})"
            ),
            "top_sensor":        top_sensor,
            "recommendation":    recommendation,
            "root_cause":        root_cause,
            "scenario":          self.active_scenario,
            "health_score":      round(self.health_score, 1),
            "consecutive":       self.consecutive_anomalies,
        }

        alerts.append(alert)
        self.alert_log.append(alert)
        self.alert_cooldowns[alert_key] = now
        logger.warning(
            f"ALERT [{severity}] {attack_type}  score={anomaly_score:.3f}  "
            f"confidence={confidence}  sensor={top_sensor}"
        )
        return alerts

    # ------------------------------------------------------------------
    # Root cause analysis  (enhanced with per-sensor LSTM errors + physics)
    # ------------------------------------------------------------------

    def analyze_root_cause(self, sample: np.ndarray) -> Dict[str, Any]:
        if self.baseline_mean is None:
            return {"error": "Baseline not fitted"}

        n     = min(len(sample), len(self.baseline_mean))
        z     = (sample[:n] - self.baseline_mean[:n]) / (self.baseline_std[:n] + 1e-8)
        abs_z = np.abs(z)
        max_z = float(abs_z.max()) if len(abs_z) > 0 else 0.0

        # Primary signal: per-sensor LSTM reconstruction error (most precise)
        if self._haiend_per_sensor is not None:
            err   = self._haiend_per_sensor
            names = (self._haiend_columns if self._haiend_columns
                     else [f"sensor_{i}" for i in range(len(err))])
            top_n   = min(10, len(err))
            top_idx = np.argsort(err)[::-1][:top_n]
            top_sensors = [
                {
                    "sensor":    names[i] if i < len(names) else f"sensor_{i}",
                    "lstm_err":  round(float(err[i]), 6),
                    "z_score":   round(float(z[i]), 2) if i < len(z) else None,
                    "current":   round(float(sample[i]), 4) if i < len(sample) else None,
                    "baseline":  round(float(self.baseline_mean[i]), 4)
                                 if i < len(self.baseline_mean) else None,
                    "deviation": f"{'+' if (i < len(z) and z[i] > 0) else ''}"
                                 f"{z[i]:.1f}σ" if i < len(z) else "n/a",
                }
                for i in top_idx
            ]
        else:
            top_n   = min(5, len(self.feature_names))
            top_idx = np.argsort(abs_z)[::-1][:top_n]
            top_sensors = [
                {
                    "sensor":    self.feature_names[i],
                    "z_score":   round(float(z[i]), 2),
                    "current":   round(float(sample[i]), 4),
                    "baseline":  round(float(self.baseline_mean[i]), 4),
                    "deviation": f"{'+' if z[i] > 0 else ''}{z[i]:.1f}σ",
                }
                for i in top_idx if i < len(self.feature_names)
            ]

        subsystem_scores = self.state._subsystem_scores()
        worst_sub = max(subsystem_scores, key=subsystem_scores.get) \
                    if subsystem_scores else "Unknown"

        if max_z > 10:   probable_cause = f"Sensor spoofing or command injection in {worst_sub}"
        elif max_z > 5:  probable_cause = f"Significant process deviation in {worst_sub} — check actuators"
        elif max_z > 3:  probable_cause = f"Gradual drift or calibration issue in {worst_sub}"
        else:            probable_cause = "Minor process variation — within extended monitoring threshold"

        result = {
            "top_sensors":        top_sensors[:5],
            "subsystem_scores":   subsystem_scores,
            "worst_subsystem":    worst_sub,
            "probable_cause":     probable_cause,
            "max_z_score":        round(max_z, 2),
            "n_sensors_above_3s": int((abs_z > 3.0).sum()),
            "detection_source":   (
                "GRU-GAT+LSTM" if (self._gat_per_sensor is not None and
                                   self._haiend_per_sensor is not None)
                else ("GRU-GAT-per-sensor" if self._gat_per_sensor is not None
                      else ("LSTM-per-sensor" if self._haiend_per_sensor is not None
                            else "Z-score"))
            ),
        }

        # Append physics violations if available
        if self._physics_last_edges:
            result["physics_violations"] = [
                {
                    "edge":      e["edge"],
                    "residual":  round(e["residual"], 4),
                    "actual":    e["actual"],
                    "predicted": e["predicted"],
                }
                for e in self._physics_last_edges[:5]
            ]

        return result

    # ------------------------------------------------------------------
    # Scenario injection
    # ------------------------------------------------------------------

    def inject_scenario(self, scenario_type: str, scenario_data: np.ndarray) -> None:
        self.active_scenario = scenario_type
        self.scenario_data   = scenario_data
        self.scenario_idx    = 0
        logger.info(f"Scenario injected: {scenario_type} ({len(scenario_data)} samples)")

    def stop_scenario(self) -> None:
        if self.active_scenario:
            logger.info(f"Scenario ended: {self.active_scenario}")
        self.active_scenario = None
        self.scenario_data   = None
        self.scenario_idx    = 0

    # ------------------------------------------------------------------
    # State & persistence
    # ------------------------------------------------------------------

    def get_state(self) -> Dict[str, Any]:
        trend = "stable"
        if len(self.predict_history) >= 10:
            rate = sum(list(self.predict_history)[-10:]) / 10
            if   rate > 0.7: trend = "deteriorating"
            elif rate < 0.2: trend = "recovering"

        recent_scores = list(self.score_history)[-60:] if self.score_history else [0]

        # Build model info block
        models_active = []
        has_lstm = self._haiend_model is not None or self._ms_pkg is not None
        has_tr   = self._tr_model is not None
        has_gat  = self._gat_model is not None
        # GRU-GAT is display-only (F1=0.4704 alone; excluded from Hard OR decision)
        if has_lstm and has_tr:
            label = "Ensemble(LSTM+Transformer)"
            if has_gat:
                label += "+GRU-GAT(display)"
            models_active.append(label)
        elif self._vae_model is not None:
            models_active.append(f"LSTM-VAE(score={self._vae_score_type})")
        elif self._ms_pkg is not None and self._ms_models:
            scales = sorted(self._ms_models.keys())
            models_active.append(f"LSTM-MultiScale(w={'|'.join(str(s) for s in scales)})")
        elif self._haiend_model is not None:
            models_active.append("LSTM-haiend(225)")
        if self._fallback_model is not None: models_active.append("LSTM-38feat")
        if self._physics_models:             models_active.append("PhysicsResidual")
        if self.anomaly_model   is not None: models_active.append("IsoForest")
        models_active.append("Z-score")

        return {
            **self.state.to_dict(),
            "health_score":          self.health_score,
            "alert_count":           len(self.alert_log),
            "active_scenario":       self.active_scenario,
            "history_length":        len(self.history_buffer),
            "trend":                 trend,
            "avg_anomaly_score_1m":  round(float(np.mean(recent_scores)), 4),
            "consecutive_anomalies": self.consecutive_anomalies,
            "models_active":         models_active,
            "primary_model_f1":      (
                0.6998 if (has_lstm and has_tr)   # ensemble F1 — GRU-GAT not in decision
                else (self._vae_pkg.get("best_f1", 0.6874) if self._vae_pkg
                      else (self._ms_pkg.get("best_ensemble_f1", 0.6874) if self._ms_pkg
                            else (0.6874 if self._haiend_model is not None else None)))
            ),
        }

    def get_alert_log(self) -> pd.DataFrame:
        if not self.alert_log:
            return pd.DataFrame()
        return pd.DataFrame(self.alert_log)

    def reset_health(self) -> None:
        self.health_score          = 100.0
        self.consecutive_anomalies = 0
        self.state.health_score    = 100.0
        self.alert_log             = []
        self.alert_cooldowns       = {}
        self.score_history.clear()
        self.predict_history.clear()
        logger.info("Digital Twin reset")

    def save_state(self, path: str = None) -> str:
        save_path = path or str(self._outputs_dir / "digital_twin_state.json")
        state_data = {
            "config": {
                "n_features":          len(self.feature_names),
                "feature_names":       self.feature_names[:50],
                "detection_threshold": self.detection_threshold,
                "models_active":       self.get_state().get("models_active", []),
            },
            "baseline": {
                "mean": self.baseline_mean.tolist()[:20] if self.baseline_mean is not None else None,
                "std":  self.baseline_std.tolist()[:20]  if self.baseline_std  is not None else None,
            },
            "health_score":  self.health_score,
            "total_alerts":  len(self.alert_log),
            "state":         self.get_state(),
            "recent_alerts": self.alert_log[-10:],
        }
        with open(save_path, "w") as f:
            json.dump(state_data, f, indent=2, default=str)
        logger.info(f"Digital Twin state saved: {save_path}")
        return save_path

    def _compute_subsystem_scores(self, abs_z_scores: np.ndarray) -> Dict[str, float]:
        """Backward compat."""
        return self.state._subsystem_scores()
