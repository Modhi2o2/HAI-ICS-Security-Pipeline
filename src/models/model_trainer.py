"""
Model Training Orchestrator

Coordinates the full training pipeline:
1. Load preprocessed data
2. Feature engineering
3. Train/val split
4. Model training
5. Evaluation
6. Save outputs
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

from src.utils.logger import logger
from src.features.preprocessing import HAIPreprocessor
from src.features.feature_engineering import HAIFeatureEngineer, create_sequence_windows
from src.models.detection_model import AttackDetectionModel


class ModelTrainer:
    """Orchestrates model training for HAI attack detection."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.label_col = config.get("data", {}).get("label_col", "Attack")
        self.timestamp_col = config.get("data", {}).get("timestamp_col", "timestamp")

        self.preprocessor = HAIPreprocessor(config)
        self.feature_engineer = HAIFeatureEngineer(config)
        self.detection_model = AttackDetectionModel(config)

        self.outputs_dir = Path(config["paths"]["outputs"])
        self.predictions_dir = Path(config["paths"]["predictions"])
        self.predictions_dir.mkdir(parents=True, exist_ok=True)

    def prepare_features(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        fit_preprocessor: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
        """
        Full feature preparation pipeline.

        Returns:
            X_train, y_train, X_test, y_test, feature_names
        """
        logger.info("Starting feature preparation pipeline...")

        # Separate labels
        y_train = train_df[self.label_col].values if self.label_col in train_df.columns else np.zeros(len(train_df))
        y_test = test_df[self.label_col].values if self.label_col in test_df.columns else np.zeros(len(test_df))

        # Drop non-feature columns
        drop_cols = [self.timestamp_col, self.label_col]
        train_features = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
        test_features = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])

        # Feature engineering
        logger.info("Applying feature engineering to train...")
        train_eng = self.feature_engineer.fit_transform(train_features)
        logger.info("Applying feature engineering to test...")
        test_eng = self.feature_engineer.transform(test_features)

        # Ensure column alignment
        train_cols = set(train_eng.columns)
        test_cols = set(test_eng.columns)
        common_cols = sorted(list(train_cols & test_cols))

        train_eng = train_eng[common_cols]
        test_eng = test_eng[common_cols]

        # Preprocessing (scale, impute)
        if fit_preprocessor:
            train_processed = self.preprocessor.fit_transform(train_eng)
            self.preprocessor.save()
        else:
            train_processed = self.preprocessor.transform(train_eng)

        test_processed = self.preprocessor.transform(test_eng)

        feature_names = list(train_processed.columns)
        X_train = train_processed.values.astype(np.float32)
        X_test = test_processed.values.astype(np.float32)

        logger.info(f"Features ready: train={X_train.shape}, test={X_test.shape}")
        return X_train, y_train, X_test, y_test, feature_names

    def train_and_evaluate(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        val_split: float = 0.15,
    ) -> Dict[str, Any]:
        """
        Full training and evaluation pipeline.

        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame with labels
            val_split: Fraction of train to use for validation

        Returns:
            Dict with model, metrics, feature importances
        """
        # Prepare features
        X_train, y_train, X_test, y_test, feature_names = self.prepare_features(train_df, test_df)

        # Validation split (time-aware: use last portion of train)
        val_size = int(len(X_train) * val_split)
        X_val = X_train[-val_size:]
        y_val = y_train[-val_size:]
        X_tr = X_train[:-val_size]
        y_tr = y_train[:-val_size]

        logger.info(f"Split: train={len(X_tr):,}, val={len(X_val):,}, test={len(X_test):,}")

        # --- Detect if training data has only one class (normal-only train set) ---
        n_classes_train = len(np.unique(y_tr))
        only_normal = (n_classes_train == 1 and y_tr[0] == 0)

        if only_normal:
            logger.warning(
                "Training data contains ONLY normal samples (class 0). "
                "Supervised classifiers cannot learn from this — switching to "
                "SEMI-SUPERVISED mode: Isolation Forest trained on normal data, "
                "plus supervised models trained on a balanced split of the TEST data."
            )
            # Use first 80% of test as pseudo-train for supervised models
            n_pseudo = int(len(X_test) * 0.6)
            X_pseudo_tr = X_test[:n_pseudo]
            y_pseudo_tr = y_test[:n_pseudo]
            X_eval = X_test[n_pseudo:]
            y_eval = y_test[n_pseudo:]

            if y_pseudo_tr.sum() > 0:
                logger.info(
                    f"Pseudo-supervised train from test: {len(X_pseudo_tr):,} samples, "
                    f"{y_pseudo_tr.sum():,} attacks ({y_pseudo_tr.mean()*100:.1f}%)"
                )
                self.detection_model.train(X_pseudo_tr, y_pseudo_tr, X_val, y_val, feature_names)
                metrics = self.detection_model.evaluate(X_eval, y_eval)
                X_test = X_eval
                y_test = y_eval
            else:
                logger.warning("No attacks in pseudo-train split either. Using Isolation Forest only.")
                self.detection_model.train(X_tr, y_tr, X_val, y_val, feature_names)
                metrics = self.detection_model.evaluate(X_test, y_test)
        else:
            # Standard supervised training
            self.detection_model.train(X_tr, y_tr, X_val, y_val, feature_names)
            metrics = self.detection_model.evaluate(X_test, y_test)

        # Also train Isolation Forest anomaly detector on normal training data
        from src.models.anomaly_detection import IsolationForestDetector
        iso_detector = IsolationForestDetector(self.config)
        iso_detector.fit(X_train[y_train == 0] if y_train.sum() > 0 else X_train)
        iso_detector.save()

        # Evaluate Isolation Forest
        iso_scores = iso_detector.score(X_test)
        iso_threshold = np.percentile(iso_detector.score(X_train[y_train == 0] if y_train.sum() > 0 else X_train), 95)
        iso_pred = (iso_scores >= iso_threshold).astype(int)
        from src.utils.metrics import compute_detection_metrics, print_metrics_report
        iso_metrics = compute_detection_metrics(y_test, iso_pred, None, model_name="IsolationForest")
        print_metrics_report(iso_metrics)
        metrics["IsolationForest"] = iso_metrics
        self.detection_model.all_metrics["IsolationForest"] = iso_metrics

        # Save models
        self.detection_model.save_all_models()
        self.detection_model.save_best_model()

        # Save test predictions
        y_pred, y_prob = self.detection_model.predict(X_test)
        pred_df = pd.DataFrame({
            "y_true": y_test,
            "y_pred": y_pred,
            "y_prob": y_prob,
        })
        pred_path = self.predictions_dir / "test_predictions.csv"
        pred_df.to_csv(pred_path, index=False)
        logger.info(f"Test predictions saved: {pred_path}")

        return {
            "metrics": metrics,
            "feature_importances": self.detection_model.feature_importances,
            "best_model_name": self.detection_model.best_model_name,
            "X_test": X_test,
            "y_test": y_test,
            "y_pred": y_pred,
            "y_prob": y_prob,
        }
