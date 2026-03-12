"""
HAI Feature Engineering Module

Generates:
- Lag features (past N seconds)
- Rolling statistics (mean, std, min, max)
- Rate-of-change (derivative) features
- Process deviation features (sensor vs setpoint)
- Alarm/threshold crossing features
- Interaction features between critical sensors
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from src.utils.logger import logger


class HAIFeatureEngineer:
    """
    Generates domain-informed features from HAI sensor data.

    All feature generation is based on physical process understanding:
    - ICS attacks often manipulate sensor readings or setpoints
    - Deviations between demand and actual values are key attack signals
    - Rapid changes in normally stable processes indicate anomalies
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.feat_cfg = config.get("features", {})
        self.prep_cfg = config.get("preprocessing", {})
        self.data_cfg = config.get("data", {})

        self.timestamp_col = self.data_cfg.get("timestamp_col", "timestamp")
        self.label_col = self.data_cfg.get("label_col", "Attack")

        self.lag_windows = self.prep_cfg.get("lag_features", [1, 5, 10, 30, 60])
        self.rolling_windows = self.prep_cfg.get("rolling_windows", [10, 30, 60, 300])

        self._generated_feature_names: List[str] = []

    def _get_sensor_cols(self, df: pd.DataFrame) -> List[str]:
        """Get continuous sensor columns (exclude timestamp, label, binary)."""
        exclude = [self.timestamp_col, self.label_col]
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return [c for c in numeric_cols if c not in exclude]

    def add_lag_features(self, df: pd.DataFrame, cols: List[str] = None) -> pd.DataFrame:
        """
        Add lagged values for sensor columns.

        Lag features capture how current readings compare to past values,
        useful for detecting sudden changes (spike attacks, sensor spoofing).
        """
        if not self.feat_cfg.get("use_lag_features", True):
            return df

        sensor_cols = cols or self._get_sensor_cols(df)
        lag_windows = self.lag_windows

        new_cols = {}
        for lag in lag_windows:
            for col in sensor_cols:
                feat_name = f"{col}_lag{lag}"
                new_cols[feat_name] = df[col].shift(lag)
                self._generated_feature_names.append(feat_name)

        lag_df = pd.DataFrame(new_cols, index=df.index)
        df = pd.concat([df, lag_df], axis=1)

        logger.info(f"Added {len(new_cols)} lag features (windows: {lag_windows})")
        return df

    def add_rolling_statistics(self, df: pd.DataFrame, cols: List[str] = None) -> pd.DataFrame:
        """
        Add rolling mean, std, min, max for sensor columns.

        Rolling statistics detect:
        - Persistent sensor drift (gradual attacks)
        - Increased variance (noisy injection attacks)
        - Sustained abnormal values (setpoint manipulation)
        """
        if not self.feat_cfg.get("use_rolling_stats", True):
            return df

        sensor_cols = cols or self._get_sensor_cols(df)
        rolling_windows = self.rolling_windows

        new_cols = {}
        for window in rolling_windows:
            for col in sensor_cols:
                rolling = df[col].rolling(window=window, min_periods=1)

                mean_name = f"{col}_rmean{window}"
                std_name = f"{col}_rstd{window}"

                new_cols[mean_name] = rolling.mean()
                new_cols[std_name] = rolling.std().fillna(0)

                self._generated_feature_names.extend([mean_name, std_name])

        roll_df = pd.DataFrame(new_cols, index=df.index)
        df = pd.concat([df, roll_df], axis=1)

        logger.info(f"Added {len(new_cols)} rolling stats features (windows: {rolling_windows})")
        return df

    def add_derivative_features(self, df: pd.DataFrame, cols: List[str] = None) -> pd.DataFrame:
        """
        Add rate-of-change (first derivative) features.

        Derivative features are excellent for detecting:
        - Sudden jumps (command injection attacks)
        - Abnormally fast changes in controlled variables
        """
        if not self.feat_cfg.get("use_derivative_features", True):
            return df

        sensor_cols = cols or self._get_sensor_cols(df)

        new_cols = {}
        for col in sensor_cols:
            diff_name = f"{col}_diff1"
            diff2_name = f"{col}_diff2"
            new_cols[diff_name] = df[col].diff().fillna(0)
            new_cols[diff2_name] = df[col].diff().diff().fillna(0)
            self._generated_feature_names.extend([diff_name, diff2_name])

        diff_df = pd.DataFrame(new_cols, index=df.index)
        df = pd.concat([df, diff_df], axis=1)

        logger.info(f"Added {len(new_cols)} derivative features")
        return df

    def add_process_deviation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add deviation between demand/setpoint and actual values.

        In ICS attacks, attackers often send false setpoints or manipulate
        actuator demands. The gap between demand and actual reading is a
        strong attack indicator.
        """
        # Known demand-actual pairs in HAI dataset
        demand_actual_pairs = [
            ("P1_FCV01D", "P1_FCV01Z"),  # Valve demand vs setpoint
            ("P1_FCV02D", "P1_FCV02Z"),
            ("P1_FCV03D", "P1_FCV03Z"),
            ("P3_LCV01D", "P3_LCV01Z"),
        ]

        new_cols = {}
        for demand_col, setpoint_col in demand_actual_pairs:
            if demand_col in df.columns and setpoint_col in df.columns:
                dev_name = f"dev_{demand_col}_vs_{setpoint_col}"
                new_cols[dev_name] = df[demand_col] - df[setpoint_col]
                self._generated_feature_names.append(dev_name)

        if new_cols:
            df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
            logger.info(f"Added {len(new_cols)} process deviation features")

        return df

    def add_cross_sensor_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add interaction features between physically related sensors.

        Process physics: flow in = flow out (conservation); pressure and
        flow are inversely related in some conditions. Violations of these
        relationships indicate sensor spoofing or process manipulation.
        """
        new_cols = {}

        # Flow balance features (P1)
        if "P1_FT01" in df.columns and "P1_FT02" in df.columns:
            new_cols["flow_balance_P1"] = df["P1_FT01"] - df["P1_FT02"]

        if "P1_FT01" in df.columns and "P3_FIT01" in df.columns:
            new_cols["flow_ratio_p1_p3"] = df["P1_FT01"] / (df["P3_FIT01"].abs() + 1e-6)

        # Pressure-flow relationship (P1)
        if "P1_PIT01" in df.columns and "P1_FT01" in df.columns:
            new_cols["pressure_flow_ratio_P1"] = df["P1_PIT01"] / (df["P1_FT01"].abs() + 1e-6)

        # Temperature consistency (P1)
        if "P1_TIT01" in df.columns and "P1_TIT02" in df.columns:
            new_cols["temp_diff_P1"] = df["P1_TIT01"] - df["P1_TIT02"]

        # Steam process consistency (P4)
        if "P4_ST_PT01" in df.columns and "P4_ST_TT01" in df.columns:
            new_cols["steam_pt_ratio"] = df["P4_ST_PT01"] / (df["P4_ST_TT01"].abs() + 1e-6)

        # Level vs flow (P3)
        if "P3_LIT01" in df.columns and "P3_FIT01" in df.columns:
            new_cols["level_flow_P3"] = df["P3_LIT01"] * df["P3_FIT01"]

        if new_cols:
            df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
            self._generated_feature_names.extend(list(new_cols.keys()))
            logger.info(f"Added {len(new_cols)} cross-sensor interaction features")

        return df

    def drop_low_variance_features(self, df: pd.DataFrame, threshold: float = None) -> pd.DataFrame:
        """Remove features with near-zero variance (constant or nearly constant)."""
        if not self.feat_cfg.get("drop_near_zero_variance", True):
            return df

        threshold = threshold or self.feat_cfg.get("variance_threshold", 0.001)
        exclude = [self.timestamp_col, self.label_col]
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c not in exclude]

        variances = df[feature_cols].var()
        low_var_cols = variances[variances < threshold].index.tolist()

        if low_var_cols:
            df = df.drop(columns=low_var_cols, errors='ignore')
            logger.info(f"Dropped {len(low_var_cols)} low-variance features")

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps."""
        logger.info("Applying feature engineering...")
        self._generated_feature_names = []

        sensor_cols = self._get_sensor_cols(df)

        # Core features
        df = self.add_derivative_features(df, sensor_cols)
        df = self.add_rolling_statistics(df, sensor_cols)
        df = self.add_lag_features(df, sensor_cols)
        df = self.add_process_deviation_features(df)
        df = self.add_cross_sensor_features(df)

        # Clean up NaN from lag/diff operations
        df = df.fillna(method='bfill').fillna(0)

        # Drop constant features
        df = self.drop_low_variance_features(df)

        total_features = len(df.select_dtypes(include=[np.number]).columns) - 1  # exclude label
        logger.info(f"Feature engineering complete: {total_features} total features")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering to test/inference data."""
        return self.fit_transform(df)  # Stateless operations, same for train/test

    def get_generated_features(self) -> List[str]:
        """Return names of all generated feature columns."""
        return self._generated_feature_names.copy()


def create_sequence_windows(
    X: np.ndarray,
    y: np.ndarray,
    window_size: int = 60,
    step_size: int = 10,
    label_strategy: str = "any"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create overlapping time-series windows for LSTM/sequence models.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Label array (n_samples,)
        window_size: Number of timesteps per window
        step_size: Step between windows (overlap control)
        label_strategy: 'any' = window is attack if any timestep is attack
                        'last' = use label of last timestep
                        'majority' = majority vote

    Returns:
        (X_windows, y_windows) arrays
    """
    n_samples, n_features = X.shape
    windows_X = []
    windows_y = []

    for start in range(0, n_samples - window_size + 1, step_size):
        end = start + window_size
        windows_X.append(X[start:end])

        window_labels = y[start:end]
        if label_strategy == "any":
            label = int(window_labels.max())
        elif label_strategy == "last":
            label = int(window_labels[-1])
        elif label_strategy == "majority":
            label = int(window_labels.mean() >= 0.5)
        else:
            label = int(window_labels[-1])

        windows_y.append(label)

    return np.array(windows_X, dtype=np.float32), np.array(windows_y, dtype=np.int64)
