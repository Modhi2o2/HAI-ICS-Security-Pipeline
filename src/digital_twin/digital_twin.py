"""
HAI Digital Twin Module

Simulates the HAI boiler/steam ICS system behavior.
Detects deviations from baseline, estimates health, and generates alerts.

Components:
- StateEstimator: tracks current system state
- AnomalyDetector: detects deviations from baseline
- RootCauseAnalyzer: identifies which sensors/subsystems are responsible
- HealthScorer: computes overall system health (0-100)
- AlertEngine: generates prioritized alerts with recommendations
- ScenarioEngine: injects synthetic scenarios for what-if analysis
"""

import numpy as np
import pandas as pd
import json
import joblib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque

from src.utils.logger import logger


class SystemState:
    """Represents the current estimated state of the ICS."""

    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.n_features = len(feature_names)
        self.current_values = np.zeros(self.n_features)
        self.baseline_mean = np.zeros(self.n_features)
        self.baseline_std = np.ones(self.n_features)
        self.timestamp = datetime.now()
        self.is_anomalous = False
        self.anomaly_score = 0.0
        self.health_score = 100.0
        self.active_alerts: List[str] = []

    def update(self, values: np.ndarray, timestamp: datetime = None) -> None:
        """Update current state with new sensor readings."""
        self.current_values = np.array(values).flatten()[:self.n_features]
        self.timestamp = timestamp or datetime.now()

    def get_deviations(self) -> np.ndarray:
        """Compute Z-score deviation from baseline."""
        return (self.current_values - self.baseline_mean) / (self.baseline_std + 1e-8)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dictionary."""
        return {
            "timestamp": str(self.timestamp),
            "health_score": round(self.health_score, 2),
            "anomaly_score": round(self.anomaly_score, 4),
            "is_anomalous": bool(self.is_anomalous),
            "active_alerts": self.active_alerts,
            "sensor_values": {
                name: round(float(val), 4)
                for name, val in zip(self.feature_names[:20], self.current_values[:20])
            },
            "top_deviating_sensors": self._get_top_deviating(5),
        }

    def _get_top_deviating(self, n: int = 5) -> List[Dict]:
        """Return top N most deviating sensors."""
        deviations = self.get_deviations()
        abs_devs = np.abs(deviations)
        top_idx = np.argsort(abs_devs)[::-1][:n]
        return [
            {
                "sensor": self.feature_names[i],
                "current_value": round(float(self.current_values[i]), 4),
                "baseline_mean": round(float(self.baseline_mean[i]), 4),
                "z_score": round(float(deviations[i]), 2),
            }
            for i in top_idx if i < len(self.feature_names)
        ]


class DigitalTwin:
    """
    Main Digital Twin for HAI ICS Boiler System.

    Provides:
    - Real-time (batch) anomaly detection
    - System health scoring
    - Root cause analysis
    - Scenario injection (attack/fault simulation)
    - Alert generation with recommendations
    """

    SCENARIO_DESCRIPTIONS = {
        "attack": "Simulated cyberattack: adversarial manipulation of sensor/actuator values",
        "sensor_failure": "Simulated sensor failure: one or more sensors reporting incorrect values",
        "drift": "Simulated sensor drift: gradual linear bias in readings",
        "communication_fault": "Simulated communication fault: frozen or dropped sensor readings",
        "overload": "Simulated system overload: extreme values across pressure/flow/temperature",
        "degradation": "Simulated equipment degradation: increasing noise + gradual drift",
        "unknown": "Random abnormal event: combination of perturbation types",
    }

    def __init__(self, config: Dict[str, Any], feature_names: List[str] = None):
        self.config = config
        self.twin_cfg = config.get("digital_twin", {})
        self.feature_names = feature_names or []

        # State
        self.state = SystemState(self.feature_names)

        # History buffer (rolling window)
        self.history_buffer = deque(maxlen=3600)  # 1 hour of 1-Hz data

        # Baseline stats (computed from training data)
        self.baseline_mean: Optional[np.ndarray] = None
        self.baseline_std: Optional[np.ndarray] = None
        self.baseline_corr: Optional[np.ndarray] = None

        # Detection model (optional, set after training)
        self.detection_model = None
        self.anomaly_model = None
        self.detection_threshold = self.twin_cfg.get("anomaly_threshold", 0.5)

        # Health scoring
        self.health_score = 100.0
        self.health_decay = self.twin_cfg.get("health_decay_rate", 0.95)

        # Alert history
        self.alert_log: List[Dict] = []
        self.alert_cooldowns: Dict[str, datetime] = {}
        self.alert_cooldown_secs = self.twin_cfg.get("alert_cooldown", 60)

        # Scenario injection state
        self.active_scenario: Optional[str] = None
        self.scenario_data: Optional[np.ndarray] = None
        self.scenario_idx: int = 0

        self._outputs_dir = Path(config["paths"]["outputs"])
        self._outputs_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Digital Twin initialized")

    def fit_baseline(self, normal_data: np.ndarray) -> None:
        """
        Compute baseline statistics from normal operation data.

        Args:
            normal_data: Normal operation samples (n_samples, n_features)
        """
        logger.info(f"Computing baseline from {len(normal_data):,} normal samples...")

        self.baseline_mean = normal_data.mean(axis=0)
        self.baseline_std = normal_data.std(axis=0)

        # Correlation matrix for relationship-based anomaly detection
        n_cols = min(50, normal_data.shape[1])
        self.baseline_corr = np.corrcoef(normal_data[:, :n_cols].T)
        self.baseline_corr = np.nan_to_num(self.baseline_corr, nan=0.0)

        # Update state baseline
        self.state.baseline_mean = self.baseline_mean
        self.state.baseline_std = self.baseline_std

        logger.info(f"Baseline fitted: mean range [{self.baseline_mean.min():.3f}, {self.baseline_mean.max():.3f}]")

    def set_detection_model(self, model, anomaly_model=None) -> None:
        """Attach trained detection model to the digital twin."""
        self.detection_model = model
        self.anomaly_model = anomaly_model
        logger.info("Detection model attached to Digital Twin")

    def ingest(self, sample: np.ndarray, timestamp: datetime = None) -> Dict[str, Any]:
        """
        Process a single sensor reading.

        Args:
            sample: Sensor values (1D array of shape n_features)
            timestamp: Optional timestamp

        Returns:
            State update dict with anomaly info and health score
        """
        sample = np.array(sample).flatten()

        # If scenario is active, inject perturbation
        if self.active_scenario is not None and self.scenario_data is not None:
            if self.scenario_idx < len(self.scenario_data):
                perturbation = self.scenario_data[self.scenario_idx]
                n = min(len(sample), len(perturbation))
                sample[:n] = perturbation[:n]
                self.scenario_idx += 1
            else:
                self.stop_scenario()

        # Update state
        self.state.update(sample, timestamp)
        self.history_buffer.append(sample.copy())

        # Compute anomaly score
        anomaly_score, is_anomalous = self._detect_anomaly(sample)
        self.state.anomaly_score = anomaly_score
        self.state.is_anomalous = is_anomalous

        # Update health score
        self.health_score = self._update_health(is_anomalous, anomaly_score)
        self.state.health_score = self.health_score

        # Generate alerts if needed
        alerts = self._check_alerts(sample, is_anomalous, anomaly_score)
        self.state.active_alerts = [a["message"] for a in alerts]

        return self.state.to_dict()

    def process_batch(self, data: np.ndarray) -> pd.DataFrame:
        """
        Process a batch of sensor readings.

        Args:
            data: Batch of readings (n_samples, n_features)

        Returns:
            DataFrame with anomaly scores, predictions, health scores
        """
        results = []

        for i, sample in enumerate(data):
            result = self.ingest(sample)
            result["sample_idx"] = i
            results.append(result)

        return pd.DataFrame(results)

    def _detect_anomaly(self, sample: np.ndarray) -> Tuple[float, bool]:
        """
        Compute anomaly score for a sample.

        Uses ML model if available, falls back to Z-score baseline deviation.
        """
        # Strategy 1: ML model (most accurate)
        if self.detection_model is not None:
            try:
                sample_2d = sample.reshape(1, -1)
                prob = self.detection_model.predict_proba(sample_2d)[0][1]
                is_anomalous = prob >= self.detection_threshold
                return float(prob), bool(is_anomalous)
            except Exception:
                pass

        # Strategy 2: Z-score deviation from baseline
        if self.baseline_mean is not None and self.baseline_std is not None:
            n = min(len(sample), len(self.baseline_mean))
            z_scores = np.abs((sample[:n] - self.baseline_mean[:n]) / (self.baseline_std[:n] + 1e-8))
            anomaly_score = float(np.percentile(z_scores, 90))  # 90th percentile Z-score

            is_anomalous = anomaly_score > 3.0  # 3-sigma rule

            # Normalize to [0, 1]
            normalized_score = min(1.0, anomaly_score / 10.0)
            return normalized_score, is_anomalous

        return 0.0, False

    def _update_health(self, is_anomalous: bool, anomaly_score: float) -> float:
        """Update system health score based on anomaly detection."""
        if is_anomalous:
            # Decay health proportional to anomaly severity
            severity = min(1.0, anomaly_score)
            self.health_score = max(0.0, self.health_score - severity * 5.0)
        else:
            # Slowly recover toward 100 when normal
            self.health_score = min(100.0, self.health_score + 0.1)

        return round(self.health_score, 2)

    def _check_alerts(self, sample: np.ndarray, is_anomalous: bool,
                       anomaly_score: float) -> List[Dict]:
        """Generate alerts based on detection results."""
        alerts = []
        now = datetime.now()

        severity_cfg = self.twin_cfg.get("severity_levels", {})

        if is_anomalous:
            # Determine severity level
            if anomaly_score >= severity_cfg.get("high", 0.85):
                severity = "HIGH"
                recommendation = "Immediate inspection required. Consider emergency shutdown."
            elif anomaly_score >= severity_cfg.get("medium", 0.6):
                severity = "MEDIUM"
                recommendation = "Investigate anomalous sensors. Verify readings manually."
            else:
                severity = "LOW"
                recommendation = "Monitor closely. Check sensor calibration."

            alert_key = f"anomaly_{severity}"
            last_alert = self.alert_cooldowns.get(alert_key)

            if last_alert is None or (now - last_alert).total_seconds() > self.alert_cooldown_secs:
                # Root cause analysis
                root_cause = self.analyze_root_cause(sample)

                alert = {
                    "timestamp": str(now),
                    "severity": severity,
                    "anomaly_score": round(anomaly_score, 4),
                    "message": f"[{severity}] Anomaly detected (score={anomaly_score:.3f})",
                    "recommendation": recommendation,
                    "root_cause": root_cause,
                    "scenario": self.active_scenario,
                }
                alerts.append(alert)
                self.alert_log.append(alert)
                self.alert_cooldowns[alert_key] = now

                logger.warning(f"ALERT [{severity}]: anomaly_score={anomaly_score:.3f}")

        return alerts

    def analyze_root_cause(self, sample: np.ndarray) -> Dict[str, Any]:
        """
        Identify root cause of anomaly.

        Returns top deviating sensors, estimated subsystem impact,
        and probable cause description.
        """
        if self.baseline_mean is None:
            return {"error": "Baseline not fitted"}

        n = min(len(sample), len(self.baseline_mean))
        z_scores = (sample[:n] - self.baseline_mean[:n]) / (self.baseline_std[:n] + 1e-8)
        abs_z = np.abs(z_scores)

        # Top contributing sensors
        top_n = min(5, len(self.feature_names))
        top_idx = np.argsort(abs_z)[::-1][:top_n]

        top_sensors = []
        for idx in top_idx:
            if idx < len(self.feature_names):
                top_sensors.append({
                    "sensor": self.feature_names[idx],
                    "z_score": round(float(z_scores[idx]), 2),
                    "current": round(float(sample[idx]), 4),
                    "baseline": round(float(self.baseline_mean[idx]), 4),
                })

        # Subsystem analysis
        subsystem_scores = self._compute_subsystem_scores(abs_z)

        # Probable cause inference
        max_z = float(abs_z.max()) if len(abs_z) > 0 else 0.0
        if max_z > 10:
            probable_cause = "Sensor spoofing or command injection attack"
        elif max_z > 5:
            probable_cause = "Significant process deviation — check actuators and control loops"
        elif max_z > 3:
            probable_cause = "Gradual drift or calibration issue"
        else:
            probable_cause = "Minor process variation — within monitoring threshold"

        return {
            "top_sensors": top_sensors,
            "subsystem_scores": subsystem_scores,
            "probable_cause": probable_cause,
            "max_z_score": round(max_z, 2),
        }

    def _compute_subsystem_scores(self, abs_z_scores: np.ndarray) -> Dict[str, float]:
        """Compute anomaly contribution per subsystem (P1, P2, P3, P4)."""
        subsystems = {}

        for i, name in enumerate(self.feature_names):
            if i >= len(abs_z_scores):
                break
            prefix = name.split("_")[0] if "_" in name else "Unknown"
            if prefix not in subsystems:
                subsystems[prefix] = []
            subsystems[prefix].append(float(abs_z_scores[i]))

        return {
            k: round(float(np.mean(v)), 3)
            for k, v in subsystems.items()
            if v
        }

    def inject_scenario(self, scenario_type: str, scenario_data: np.ndarray) -> None:
        """
        Inject a synthetic scenario into the digital twin stream.

        Args:
            scenario_type: Type label (e.g., "attack", "sensor_failure")
            scenario_data: Array of scenario samples (n_samples, n_features)
        """
        self.active_scenario = scenario_type
        self.scenario_data = scenario_data
        self.scenario_idx = 0
        logger.info(f"Scenario injected: {scenario_type} ({len(scenario_data)} samples)")

    def stop_scenario(self) -> None:
        """Stop the active scenario injection."""
        if self.active_scenario:
            logger.info(f"Scenario stopped: {self.active_scenario}")
        self.active_scenario = None
        self.scenario_data = None
        self.scenario_idx = 0

    def get_state(self) -> Dict[str, Any]:
        """Get current full state summary."""
        return {
            **self.state.to_dict(),
            "health_score": self.health_score,
            "alert_count": len(self.alert_log),
            "active_scenario": self.active_scenario,
            "history_length": len(self.history_buffer),
        }

    def get_alert_log(self) -> pd.DataFrame:
        """Return alert log as DataFrame."""
        if not self.alert_log:
            return pd.DataFrame()
        return pd.DataFrame(self.alert_log)

    def reset_health(self) -> None:
        """Reset health score and alerts."""
        self.health_score = 100.0
        self.state.health_score = 100.0
        self.alert_log = []
        self.alert_cooldowns = {}
        logger.info("Digital Twin health and alerts reset")

    def save_state(self, path: str = None) -> str:
        """Save current twin state and configuration."""
        save_path = path or str(self._outputs_dir / "digital_twin_state.json")
        state_data = {
            "config": {
                "n_features": len(self.feature_names),
                "feature_names": self.feature_names[:50],  # truncate for readability
                "detection_threshold": self.detection_threshold,
            },
            "baseline": {
                "mean": self.baseline_mean.tolist()[:20] if self.baseline_mean is not None else None,
                "std": self.baseline_std.tolist()[:20] if self.baseline_std is not None else None,
            },
            "health_score": self.health_score,
            "total_alerts": len(self.alert_log),
            "state": self.get_state(),
        }
        with open(save_path, "w") as f:
            json.dump(state_data, f, indent=2, default=str)
        logger.info(f"Digital Twin state saved: {save_path}")
        return save_path
