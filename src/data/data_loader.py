"""
HAI Dataset Loader and Merger

Loads and merges multiple CSV files from any HAI dataset version into
a unified, training-ready DataFrame.
"""

import os
import gc
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from src.utils.logger import logger


class HAIDataLoader:
    """
    Loads HAI ICS Security Dataset files.

    Handles:
    - Multiple CSV file merging (train1, train2, ...)
    - Version-specific parsing (semicolon vs comma delimiters)
    - Separate label file alignment (hai-23.05+)
    - Memory-efficient loading with optional row limits
    - Timestamp parsing and sorting
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Pipeline configuration dict (from config.yaml)
        """
        self.config = config
        self.data_cfg = config.get("data", {})
        self.raw_data_path = Path(config["paths"]["raw_data"])
        self.version = self.data_cfg.get("version", "hai-23.05")
        self.timestamp_col = self.data_cfg.get("timestamp_col", "timestamp")
        self.label_col = self.data_cfg.get("label_col", "Attack")

        # Version-specific delimiter
        self._delimiter = ";" if self.version == "hai-20.07" else ","

        logger.info(f"HAIDataLoader initialized for version: {self.version}")
        logger.info(f"Data path: {self.raw_data_path}")

    def _load_single_file(self, filepath: Path, max_rows: int = None) -> pd.DataFrame:
        """Load a single CSV file with proper parsing."""
        logger.info(f"Loading: {filepath.name} ...")

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        # Read with type inference
        df = pd.read_csv(
            filepath,
            sep=self._delimiter,
            parse_dates=[self.timestamp_col] if self.timestamp_col else False,
            nrows=max_rows,
            low_memory=False,
        )

        # Clean column names
        df.columns = [c.strip() for c in df.columns]

        logger.info(f"  Loaded {len(df):,} rows x {len(df.columns)} cols")
        return df

    def _align_labels(self, data_df: pd.DataFrame, label_df: pd.DataFrame) -> pd.DataFrame:
        """
        Align separate label file with data by timestamp.

        HAI-23.05 uses separate label-test*.csv files.
        """
        label_col_name = self.label_col
        ts_col = self.timestamp_col

        # Ensure timestamps are parsed
        if not pd.api.types.is_datetime64_any_dtype(data_df[ts_col]):
            data_df[ts_col] = pd.to_datetime(data_df[ts_col])
        if not pd.api.types.is_datetime64_any_dtype(label_df[ts_col]):
            label_df[ts_col] = pd.to_datetime(label_df[ts_col])

        # Find the label column in label file
        label_file_cols = [c for c in label_df.columns if c.lower() not in [ts_col.lower()]]
        if not label_file_cols:
            raise ValueError("No label column found in label file")

        actual_label_col = label_file_cols[0]  # e.g., 'Attack'

        # Merge on timestamp
        merged = data_df.merge(
            label_df[[ts_col, actual_label_col]].rename(columns={actual_label_col: label_col_name}),
            on=ts_col,
            how='left'
        )

        # Fill missing labels with 0 (assume normal)
        merged[label_col_name] = merged[label_col_name].fillna(0).astype(int)

        logger.info(f"  Labels aligned: {merged[label_col_name].sum():,} attack samples "
                   f"({merged[label_col_name].mean()*100:.2f}%)")
        return merged

    def load_train(self, max_rows: int = None) -> pd.DataFrame:
        """
        Load and merge all training files.

        Returns:
            Merged training DataFrame sorted by timestamp
        """
        train_files = self.data_cfg.get("train_files", [])
        max_train_rows = max_rows or self.data_cfg.get("max_train_rows", None)

        if not train_files:
            raise ValueError("No train files specified in config")

        dfs = []
        rows_loaded = 0

        for fname in train_files:
            fpath = self.raw_data_path / fname
            if not fpath.exists():
                logger.warning(f"Train file not found: {fpath}, skipping")
                continue

            # Per-file row limit for balanced loading
            per_file_limit = None
            if max_train_rows:
                remaining = max_train_rows - rows_loaded
                if remaining <= 0:
                    break
                per_file_limit = remaining

            df = self._load_single_file(fpath, max_rows=per_file_limit)

            # Train files in HAI-23.05 do NOT have attack labels
            # They are all normal operation data
            if self.label_col not in df.columns:
                df[self.label_col] = 0

            dfs.append(df)
            rows_loaded += len(df)
            del df
            gc.collect()

        if not dfs:
            raise ValueError("No training files could be loaded")

        train_df = pd.concat(dfs, ignore_index=True)
        del dfs
        gc.collect()

        train_df = self._post_process(train_df)
        logger.info(f"Training data loaded: {len(train_df):,} rows, "
                   f"{train_df[self.label_col].mean()*100:.2f}% attack")
        return train_df

    def load_test(self, max_rows: int = None) -> pd.DataFrame:
        """
        Load and merge all test files with their labels.

        Returns:
            Merged test DataFrame with labels, sorted by timestamp
        """
        test_files = self.data_cfg.get("test_files", [])
        label_files = self.data_cfg.get("label_files", [])
        max_test_rows = max_rows or self.data_cfg.get("max_test_rows", None)

        if not test_files:
            raise ValueError("No test files specified in config")

        dfs = []

        for i, (test_fname, label_fname) in enumerate(zip(test_files, label_files)):
            test_path = self.raw_data_path / test_fname
            label_path = self.raw_data_path / label_fname

            if not test_path.exists():
                logger.warning(f"Test file not found: {test_path}, skipping")
                continue

            per_file_limit = None
            if max_test_rows:
                remaining = max_test_rows - sum(len(d) for d in dfs)
                if remaining <= 0:
                    break
                per_file_limit = remaining

            test_df = self._load_single_file(test_path, max_rows=per_file_limit)

            # Attach labels from separate file
            if label_path.exists() and self.label_col not in test_df.columns:
                label_df = pd.read_csv(label_path, sep=self._delimiter)
                label_df.columns = [c.strip() for c in label_df.columns]
                test_df = self._align_labels(test_df, label_df)
            elif self.label_col not in test_df.columns:
                logger.warning(f"Label file not found: {label_path}, setting labels to 0")
                test_df[self.label_col] = 0

            dfs.append(test_df)

        if not dfs:
            raise ValueError("No test files could be loaded")

        test_df = pd.concat(dfs, ignore_index=True)
        test_df = self._post_process(test_df)

        logger.info(f"Test data loaded: {len(test_df):,} rows, "
                   f"{test_df[self.label_col].mean()*100:.2f}% attack")
        return test_df

    def load_all(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load both train and test datasets."""
        logger.info("Loading training data...")
        train_df = self.load_train()

        logger.info("Loading test data...")
        test_df = self.load_test()

        return train_df, test_df

    def _post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply common post-processing steps."""
        # Parse timestamps
        ts_col = self.timestamp_col
        if ts_col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[ts_col]):
            df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')

        # Sort by timestamp
        if ts_col in df.columns:
            df = df.sort_values(ts_col).reset_index(drop=True)

        # Remove duplicate timestamps (keep last)
        if ts_col in df.columns:
            n_dupes = df.duplicated(subset=[ts_col]).sum()
            if n_dupes > 0:
                logger.warning(f"Removing {n_dupes:,} duplicate timestamps")
                df = df.drop_duplicates(subset=[ts_col], keep='last')

        # Cast label to int
        if self.label_col in df.columns:
            df[self.label_col] = df[self.label_col].fillna(0).astype(int)

        return df

    def get_schema_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate schema summary of a DataFrame."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        dt_cols = df.select_dtypes(include=['datetime']).columns.tolist()

        if self.label_col in numeric_cols:
            numeric_cols.remove(self.label_col)

        return {
            "shape": df.shape,
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
            "n_numeric": len(numeric_cols),
            "n_categorical": len(cat_cols),
            "n_datetime": len(dt_cols),
            "numeric_cols": numeric_cols,
            "categorical_cols": cat_cols,
            "datetime_cols": dt_cols,
            "null_counts": df.isnull().sum().to_dict(),
            "null_pct": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
            "duplicates": int(df.duplicated().sum()),
            "label_distribution": df[self.label_col].value_counts().to_dict() if self.label_col in df.columns else {},
            "attack_rate": float(df[self.label_col].mean()) if self.label_col in df.columns else 0.0,
            "time_range": {
                "start": str(df[self.timestamp_col].min()) if self.timestamp_col in df.columns else None,
                "end": str(df[self.timestamp_col].max()) if self.timestamp_col in df.columns else None,
                "duration_hours": None
            }
        }


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
