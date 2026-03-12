"""
HAI Dataset Preprocessing Pipeline

Handles:
- Timestamp parsing and time-series sorting
- Missing value imputation
- Outlier handling
- Feature scaling/normalization
- Windowing for time-series models
- Train/test splitting (time-aware, no leakage)
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from src.utils.logger import logger


class HAIPreprocessor:
    """
    Stateful preprocessing pipeline for HAI ICS dataset.

    Fit on training data, transform both train and test.
    Saves fitted objects for reproducibility.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.prep_cfg = config.get("preprocessing", {})
        self.feat_cfg = config.get("features", {})
        self.data_cfg = config.get("data", {})

        self.timestamp_col = self.data_cfg.get("timestamp_col", "timestamp")
        self.label_col = self.data_cfg.get("label_col", "Attack")

        self.scaler_type = self.prep_cfg.get("scaler", "standard")
        self.impute_strategy = self.prep_cfg.get("impute_strategy", "forward_fill")
        self.outlier_method = self.prep_cfg.get("outlier_method", "iqr")
        self.outlier_threshold = self.prep_cfg.get("outlier_threshold", 3.0)

        # Fitted objects (set during fit())
        self._scaler = None
        self._imputer = None
        self._feature_cols: List[str] = []
        self._binary_cols: List[str] = []
        self._numeric_cols: List[str] = []
        self._is_fitted = False

        # Output paths
        self.models_dir = Path(config["paths"]["models"])
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def _detect_column_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Detect binary vs continuous numeric columns."""
        exclude = [self.timestamp_col, self.label_col]
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in exclude]

        binary_cols = []
        continuous_cols = []

        for col in numeric_cols:
            unique_vals = df[col].dropna().unique()
            if set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                binary_cols.append(col)
            else:
                continuous_cols.append(col)

        logger.info(f"Detected {len(continuous_cols)} continuous, {len(binary_cols)} binary columns")
        return continuous_cols, binary_cols

    def _impute(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Impute missing values in specified columns."""
        if not cols:
            return df

        strategy = self.impute_strategy

        if strategy == "forward_fill":
            df[cols] = df[cols].fillna(method='ffill').fillna(method='bfill')
        elif strategy == "mean":
            if self._imputer is None:
                self._imputer = SimpleImputer(strategy='mean')
                df[cols] = self._imputer.fit_transform(df[cols])
            else:
                df[cols] = self._imputer.transform(df[cols])
        elif strategy == "median":
            if self._imputer is None:
                self._imputer = SimpleImputer(strategy='median')
                df[cols] = self._imputer.fit_transform(df[cols])
            else:
                df[cols] = self._imputer.transform(df[cols])
        else:
            df[cols] = df[cols].fillna(0)

        # Final fallback for any remaining NaNs
        df[cols] = df[cols].fillna(0)
        return df

    def _handle_outliers(self, df: pd.DataFrame, cols: List[str], fit: bool = True) -> pd.DataFrame:
        """Clip outliers using IQR or Z-score method."""
        if self.outlier_method == "none" or not cols:
            return df

        method = self.outlier_method
        threshold = self.outlier_threshold

        if not hasattr(self, '_outlier_bounds') or fit:
            self._outlier_bounds = {}

        for col in cols:
            if fit:
                if method == "iqr":
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - threshold * IQR
                    upper = Q3 + threshold * IQR
                elif method == "zscore":
                    mean = df[col].mean()
                    std = df[col].std()
                    lower = mean - threshold * std
                    upper = mean + threshold * std
                else:
                    continue
                self._outlier_bounds[col] = (lower, upper)

            if col in self._outlier_bounds:
                lower, upper = self._outlier_bounds[col]
                df[col] = df[col].clip(lower=lower, upper=upper)

        return df

    def _build_scaler(self) -> object:
        """Return a scaler instance based on config."""
        scaler_type = self.scaler_type
        if scaler_type == "standard":
            return StandardScaler()
        elif scaler_type == "minmax":
            return MinMaxScaler()
        elif scaler_type == "robust":
            return RobustScaler()
        else:
            return StandardScaler()

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit preprocessor on training data and transform it.

        Args:
            df: Training DataFrame with timestamp and label columns

        Returns:
            Preprocessed DataFrame
        """
        logger.info("Fitting preprocessor on training data...")
        df = df.copy()

        # Detect column types
        self._numeric_cols, self._binary_cols = self._detect_column_types(df)
        self._feature_cols = self._numeric_cols + self._binary_cols

        # Impute
        df = self._impute(df, self._numeric_cols + self._binary_cols)

        # Handle outliers (continuous cols only)
        df = self._handle_outliers(df, self._numeric_cols, fit=True)

        # Scale continuous columns
        if self._numeric_cols:
            self._scaler = self._build_scaler()
            df[self._numeric_cols] = self._scaler.fit_transform(df[self._numeric_cols])

        self._is_fitted = True
        logger.info(f"Preprocessor fitted on {len(df):,} samples, {len(self._feature_cols)} features")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform test/inference data using fitted preprocessor.

        Args:
            df: DataFrame to transform

        Returns:
            Preprocessed DataFrame
        """
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted before calling transform()")

        df = df.copy()

        # Impute
        available_numeric = [c for c in self._numeric_cols if c in df.columns]
        available_binary = [c for c in self._binary_cols if c in df.columns]
        df = self._impute(df, available_numeric + available_binary)

        # Handle outliers
        df = self._handle_outliers(df, available_numeric, fit=False)

        # Scale
        if self._scaler is not None and available_numeric:
            df[available_numeric] = self._scaler.transform(df[available_numeric])

        return df

    def get_feature_columns(self) -> List[str]:
        """Return list of feature column names."""
        return self._feature_cols.copy()

    def save(self, path: str = None) -> str:
        """Save fitted preprocessor to disk."""
        save_path = path or str(self.models_dir / "preprocessor.joblib")
        joblib.dump({
            "scaler": self._scaler,
            "imputer": self._imputer,
            "feature_cols": self._feature_cols,
            "numeric_cols": self._numeric_cols,
            "binary_cols": self._binary_cols,
            "outlier_bounds": getattr(self, '_outlier_bounds', {}),
            "config": {
                "scaler_type": self.scaler_type,
                "impute_strategy": self.impute_strategy,
                "outlier_method": self.outlier_method,
                "outlier_threshold": self.outlier_threshold,
            }
        }, save_path)
        logger.info(f"Preprocessor saved to: {save_path}")
        return save_path

    def load(self, path: str) -> None:
        """Load a fitted preprocessor from disk."""
        state = joblib.load(path)
        self._scaler = state["scaler"]
        self._imputer = state["imputer"]
        self._feature_cols = state["feature_cols"]
        self._numeric_cols = state["numeric_cols"]
        self._binary_cols = state["binary_cols"]
        self._outlier_bounds = state.get("outlier_bounds", {})
        self._is_fitted = True
        logger.info(f"Preprocessor loaded from: {path}")


def time_aware_train_test_split(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    test_ratio: float = 0.2,
    label_col: str = "Attack"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset chronologically (no shuffling) to avoid data leakage.

    The split point is chosen such that any attack segments are not
    split mid-event.

    Args:
        df: Time-sorted DataFrame
        timestamp_col: Timestamp column name
        test_ratio: Fraction for test set
        label_col: Attack label column

    Returns:
        (train_df, test_df) tuple
    """
    n = len(df)
    split_idx = int(n * (1 - test_ratio))

    # Don't split in the middle of an attack segment
    if label_col in df.columns:
        # Find the nearest non-attack row to split_idx
        window = min(300, split_idx // 10)  # search within 5-min window
        search_start = max(0, split_idx - window)
        search_end = min(n - 1, split_idx + window)
        search_region = df.iloc[search_start:search_end][label_col]
        normal_indices = search_region[search_region == 0].index
        if len(normal_indices) > 0:
            # Find index closest to split_idx
            split_idx = min(normal_indices, key=lambda x: abs(x - split_idx))
            split_idx = df.index.get_loc(split_idx)

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    logger.info(f"Time-aware split: train={len(train_df):,}, test={len(test_df):,}")
    if label_col in df.columns:
        logger.info(f"  Train attack rate: {train_df[label_col].mean()*100:.2f}%")
        logger.info(f"  Test attack rate:  {test_df[label_col].mean()*100:.2f}%")

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
