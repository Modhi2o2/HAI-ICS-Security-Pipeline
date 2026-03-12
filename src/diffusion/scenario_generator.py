"""
Scenario Generator — creates synthetic ICS attack/fault scenarios.

Uses the trained diffusion model + rule-based perturbations to generate
a variety of realistic abnormal scenarios for testing and augmentation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from pathlib import Path

from src.utils.logger import logger


class ScenarioGenerator:
    """
    Generates synthetic attack and fault scenarios for ICS digital twin.

    Strategy:
    1. Diffusion-based: use generative model (when trained)
    2. Rule-based: physics-informed perturbations of baseline data
    Both are always available; diffusion adds statistical realism.
    """

    SCENARIO_TYPES = [
        "cyberattack",
        "sensor_drift",
        "sudden_spike",
        "equipment_degradation",
        "communication_loss",
        "abnormal_operating_condition",
        "replay_attack",
        "setpoint_manipulation",
    ]

    def __init__(self, config: Dict[str, Any], diffusion_model=None):
        """
        Args:
            config: Pipeline config
            diffusion_model: Optional trained HAIDiffusionModel instance
        """
        self.config = config
        self.diffusion_model = diffusion_model
        self.outputs_dir = Path(config["paths"]["synthetic"])
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

    def generate_rule_based(
        self,
        baseline_data: np.ndarray,
        scenario_type: str = "sensor_drift",
        intensity: float = 0.5,
        n_samples: int = 500,
        feature_names: List[str] = None,
    ) -> np.ndarray:
        """
        Generate synthetic scenario by perturbing baseline data.

        Args:
            baseline_data: Normal operation data (n_samples, n_features)
            scenario_type: Type of scenario to generate
            intensity: 0.0-1.0, how extreme the perturbation is
            n_samples: Number of samples to generate
            feature_names: Column names (for targeted perturbations)

        Returns:
            Perturbed synthetic data (n_samples, n_features)
        """
        # Sample from baseline
        idx = np.random.choice(len(baseline_data), size=n_samples, replace=True)
        synthetic = baseline_data[idx].copy().astype(np.float64)

        if scenario_type == "cyberattack":
            synthetic = self._simulate_cyberattack(synthetic, intensity, feature_names)
        elif scenario_type == "sensor_drift":
            synthetic = self._simulate_sensor_drift(synthetic, intensity, feature_names)
        elif scenario_type == "sudden_spike":
            synthetic = self._simulate_sudden_spike(synthetic, intensity, feature_names)
        elif scenario_type == "equipment_degradation":
            synthetic = self._simulate_degradation(synthetic, intensity, feature_names)
        elif scenario_type == "communication_loss":
            synthetic = self._simulate_communication_loss(synthetic, intensity, feature_names)
        elif scenario_type == "abnormal_operating_condition":
            synthetic = self._simulate_abnormal_condition(synthetic, intensity, feature_names)
        elif scenario_type == "replay_attack":
            synthetic = self._simulate_replay_attack(synthetic, intensity)
        elif scenario_type == "setpoint_manipulation":
            synthetic = self._simulate_setpoint_manipulation(synthetic, intensity, feature_names)
        else:
            logger.warning(f"Unknown scenario type: {scenario_type}, returning unmodified baseline")

        return synthetic

    def _simulate_cyberattack(self, data: np.ndarray, intensity: float,
                               feature_names: List[str] = None) -> np.ndarray:
        """Simulate cyberattack: randomly manipulate multiple sensor readings."""
        n_features = data.shape[1]
        n_affected = max(1, int(n_features * intensity * 0.4))
        affected_cols = np.random.choice(n_features, size=n_affected, replace=False)

        for col in affected_cols:
            col_range = data[:, col].max() - data[:, col].min()
            if col_range < 1e-6:
                continue
            # Inject constant fake value (flat-line attack) or random spike
            attack_type = np.random.choice(["flatline", "spike", "constant_offset"])
            if attack_type == "flatline":
                fake_val = data[:, col].mean() + np.random.choice([-1, 1]) * col_range * intensity
                data[:, col] = fake_val
            elif attack_type == "spike":
                spike_idx = np.random.choice(len(data))
                data[spike_idx, col] += col_range * intensity * 3
            else:
                offset = col_range * intensity * np.random.choice([-1, 1])
                data[:, col] += offset

        return data

    def _simulate_sensor_drift(self, data: np.ndarray, intensity: float,
                                feature_names: List[str] = None) -> np.ndarray:
        """Simulate gradual sensor drift: linear bias increasing over time."""
        n_features = data.shape[1]
        n_affected = max(1, int(n_features * 0.3))
        affected_cols = np.random.choice(n_features, size=n_affected, replace=False)

        t = np.linspace(0, 1, len(data))

        for col in affected_cols:
            col_std = data[:, col].std()
            drift_magnitude = col_std * intensity * 2
            drift = drift_magnitude * t * np.random.choice([-1, 1])
            data[:, col] += drift

        return data

    def _simulate_sudden_spike(self, data: np.ndarray, intensity: float,
                                feature_names: List[str] = None) -> np.ndarray:
        """Simulate sudden spike/drop in sensor readings."""
        n_features = data.shape[1]
        spike_col = np.random.randint(0, n_features)

        spike_start = len(data) // 3
        spike_end = spike_start + max(5, int(len(data) * 0.1))

        col_range = data[:, spike_col].max() - data[:, spike_col].min()
        spike_magnitude = col_range * intensity * 2

        data[spike_start:spike_end, spike_col] += spike_magnitude * np.random.choice([-1, 1])
        return data

    def _simulate_degradation(self, data: np.ndarray, intensity: float,
                               feature_names: List[str] = None) -> np.ndarray:
        """Simulate equipment degradation: increasing noise + drift."""
        n_features = data.shape[1]
        t = np.linspace(0, 1, len(data))

        for col in range(n_features):
            col_std = data[:, col].std()
            if col_std < 1e-6:
                continue
            # Increasing noise amplitude
            noise_amplitude = col_std * intensity * t
            noise = np.random.normal(0, 1, len(data)) * noise_amplitude
            data[:, col] += noise

        return data

    def _simulate_communication_loss(self, data: np.ndarray, intensity: float,
                                      feature_names: List[str] = None) -> np.ndarray:
        """Simulate network loss: sensors report last-known or zero values."""
        n_features = data.shape[1]
        n_affected = max(1, int(n_features * intensity * 0.5))
        affected_cols = np.random.choice(n_features, size=n_affected, replace=False)

        loss_start = int(len(data) * 0.3)

        for col in affected_cols:
            # Hold last value (simulating frozen sensor)
            last_val = data[loss_start - 1, col]
            data[loss_start:, col] = last_val + np.random.normal(0, 0.001 * abs(last_val),
                                                                   len(data) - loss_start)
        return data

    def _simulate_abnormal_condition(self, data: np.ndarray, intensity: float,
                                      feature_names: List[str] = None) -> np.ndarray:
        """Simulate abnormal operating condition: shift all sensors to unusual region."""
        data_means = data.mean(axis=0)
        data_stds = data.std(axis=0) + 1e-8

        # Shift all features in a consistent direction (overpressure, overtemp, etc.)
        direction = np.random.choice([-1, 1])
        shift = data_stds * intensity * 3 * direction
        data += shift
        return data

    def _simulate_replay_attack(self, data: np.ndarray, intensity: float) -> np.ndarray:
        """Simulate replay attack: copy a segment and repeat it, masking anomalies."""
        n = len(data)
        segment_len = int(n * 0.2)
        start_idx = int(n * 0.1)

        # Replay segment starting at ~1/3 of the data
        replay_start = int(n * 0.5)
        replay_end = min(replay_start + segment_len, n)
        actual_len = replay_end - replay_start

        # Replace with earlier segment
        data[replay_start:replay_end] = data[start_idx:start_idx + actual_len]
        return data

    def _simulate_setpoint_manipulation(self, data: np.ndarray, intensity: float,
                                         feature_names: List[str] = None) -> np.ndarray:
        """Simulate setpoint manipulation: change control valve setpoints."""
        # Target columns ending in 'Z' (setpoint columns in HAI)
        if feature_names:
            setpoint_idxs = [i for i, n in enumerate(feature_names) if n.endswith('Z') or 'SP' in n]
        else:
            setpoint_idxs = []

        if not setpoint_idxs:
            # Fall back to first 20% of columns
            setpoint_idxs = list(range(min(10, data.shape[1] // 5)))

        for col_idx in setpoint_idxs:
            col_range = data[:, col_idx].max() - data[:, col_idx].min()
            if col_range < 1e-6:
                continue
            # Abrupt setpoint change
            change_point = int(len(data) * 0.4)
            new_setpoint = data[change_point, col_idx] + col_range * intensity * np.random.choice([-1, 1])
            data[change_point:, col_idx] = new_setpoint

        return data

    def generate_all_scenarios(
        self,
        baseline_data: np.ndarray,
        n_per_scenario: int = 200,
        feature_names: List[str] = None,
    ) -> Dict[str, np.ndarray]:
        """Generate all scenario types and save them."""
        results = {}

        for scenario_type in self.SCENARIO_TYPES:
            logger.info(f"Generating scenario: {scenario_type}...")

            # Try diffusion first
            if self.diffusion_model is not None:
                try:
                    class_id = 1 if "attack" in scenario_type or "spike" in scenario_type else 2
                    synthetic = self.diffusion_model.generate(n_per_scenario, scenario_class=class_id)
                    method = "diffusion"
                except Exception as e:
                    logger.warning(f"Diffusion failed for {scenario_type}: {e}, using rule-based")
                    synthetic = self.generate_rule_based(
                        baseline_data, scenario_type, intensity=0.6, n_samples=n_per_scenario,
                        feature_names=feature_names
                    )
                    method = "rule_based"
            else:
                synthetic = self.generate_rule_based(
                    baseline_data, scenario_type, intensity=0.6, n_samples=n_per_scenario,
                    feature_names=feature_names
                )
                method = "rule_based"

            results[scenario_type] = synthetic

            # Save scenario
            save_path = self.outputs_dir / f"scenario_{scenario_type}.npy"
            np.save(save_path, synthetic)
            logger.info(f"  {scenario_type}: {len(synthetic)} samples generated ({method})")

        logger.info(f"All {len(results)} scenarios generated")
        return results

    def scenarios_to_dataframe(
        self,
        scenarios: Dict[str, np.ndarray],
        feature_names: List[str],
    ) -> pd.DataFrame:
        """Convert scenario arrays to a labeled DataFrame."""
        dfs = []
        for scenario_name, data in scenarios.items():
            n_feats = min(len(feature_names), data.shape[1])
            df = pd.DataFrame(data[:, :n_feats], columns=feature_names[:n_feats])
            df["scenario_type"] = scenario_name
            df["is_attack"] = 1 if "attack" in scenario_name or "spike" in scenario_name else 0
            dfs.append(df)

        return pd.concat(dfs, ignore_index=True)
