"""
HAI Attack Detection Model

Trains and evaluates supervised classification models for ICS cyberattack detection:
- XGBoost (primary model)
- LightGBM
- Random Forest
- Ensemble (voting)
"""

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

from src.utils.logger import logger
from src.utils.metrics import compute_detection_metrics, print_metrics_report

# Optional imports with graceful fallback
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    logger.warning("XGBoost not installed — skipping XGB model")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    logger.warning("LightGBM not installed — skipping LGB model")


class AttackDetectionModel:
    """
    Trains multiple classifiers for attack detection on HAI data.

    Designed for highly imbalanced data (~3-5% attack rate).
    Uses time-aware evaluation (no shuffled CV to prevent leakage).
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_cfg = config.get("models", {}).get("detection", {})
        self.outputs_dir = Path(config["paths"]["models"])
        self.metrics_dir = Path(config["paths"]["metrics"])
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.label_col = config.get("data", {}).get("label_col", "Attack")

        self.models: Dict[str, Any] = {}
        self.best_model_name: str = None
        self.best_model = None
        self.feature_importances: Dict[str, pd.DataFrame] = {}
        self.all_metrics: Dict[str, Dict] = {}

    def _build_xgboost(self) -> object:
        """Build XGBoost classifier (compatible with v1.x, 2.x, and 3.x)."""
        cfg = self.model_cfg.get("xgboost", {})
        rounds = cfg.get("early_stopping_rounds", 30)
        kwargs = dict(
            n_estimators=cfg.get("n_estimators", 300),
            max_depth=cfg.get("max_depth", 6),
            learning_rate=cfg.get("learning_rate", 0.05),
            subsample=cfg.get("subsample", 0.8),
            colsample_bytree=cfg.get("colsample_bytree", 0.8),
            scale_pos_weight=cfg.get("scale_pos_weight", 20),
            eval_metric="auc",
            random_state=self.config.get("project", {}).get("seed", 42),
            n_jobs=-1,
            verbosity=0,
        )
        # XGBoost 2.0+: early_stopping_rounds is a constructor param, not fit() param
        try:
            import xgboost as _xgb
            _xgb_ver = tuple(int(x) for x in _xgb.__version__.split(".")[:2])
            if _xgb_ver >= (2, 0):
                kwargs["early_stopping_rounds"] = rounds
        except Exception:
            pass
        try:
            kwargs.pop("use_label_encoder", None)  # removed in v2+
        except Exception:
            pass
        return xgb.XGBClassifier(**kwargs)

    def _build_lightgbm(self) -> object:
        """Build LightGBM classifier."""
        cfg = self.model_cfg.get("lightgbm", {})
        return lgb.LGBMClassifier(
            n_estimators=cfg.get("n_estimators", 300),
            max_depth=cfg.get("max_depth", 6),
            learning_rate=cfg.get("learning_rate", 0.05),
            num_leaves=cfg.get("num_leaves", 63),
            class_weight=cfg.get("class_weight", "balanced"),
            random_state=self.config.get("project", {}).get("seed", 42),
            n_jobs=-1,
            verbose=-1,
        )

    def _build_random_forest(self) -> object:
        """Build Random Forest classifier."""
        cfg = self.model_cfg.get("random_forest", {})
        return RandomForestClassifier(
            n_estimators=cfg.get("n_estimators", 200),
            max_depth=cfg.get("max_depth", 10),
            class_weight=cfg.get("class_weight", "balanced"),
            random_state=self.config.get("project", {}).get("seed", 42),
            n_jobs=-1,
        )

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        feature_names: List[str] = None,
    ) -> None:
        """
        Train all available models.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional, for early stopping)
            y_val: Validation labels
            feature_names: Column names for feature importance
        """
        logger.info(f"Training detection models on {len(X_train):,} samples "
                   f"({y_train.sum():,} attacks = {y_train.mean()*100:.1f}%)")

        # Build model dict
        builders = []
        if HAS_XGB:
            builders.append(("XGBoost", self._build_xgboost))
        if HAS_LGB:
            builders.append(("LightGBM", self._build_lightgbm))
        builders.append(("RandomForest", self._build_random_forest))

        for name, builder in builders:
            logger.info(f"Training {name}...")
            model = builder()

            try:
                if name == "XGBoost" and X_val is not None:
                    # early_stopping_rounds now lives on the constructor (v2+), not fit()
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        verbose=False,
                    )
                elif name == "LightGBM" and X_val is not None:
                    callbacks = [lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)]
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        callbacks=callbacks,
                    )
                else:
                    model.fit(X_train, y_train)

                self.models[name] = model
                logger.info(f"  {name} trained successfully")

                # Extract feature importances
                if feature_names and hasattr(model, 'feature_importances_'):
                    fi = pd.DataFrame({
                        'feature': feature_names,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    self.feature_importances[name] = fi

            except Exception as e:
                logger.error(f"  Failed to train {name}: {e}")

        # Build ensemble if multiple models trained
        if len(self.models) >= 2:
            try:
                estimators = [(name, model) for name, model in self.models.items()]
                ensemble = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
                ensemble.fit(X_train, y_train)
                self.models["Ensemble"] = ensemble
                logger.info("Ensemble model trained")
            except Exception as e:
                logger.warning(f"Ensemble training failed: {e}")

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict[str, Dict]:
        """
        Evaluate all trained models on test set.

        Args:
            X_test: Test features
            y_test: True labels
            threshold: Decision threshold for classification

        Returns:
            Dict of metrics per model
        """
        logger.info(f"Evaluating {len(self.models)} models on {len(X_test):,} test samples...")

        best_f1 = -1.0

        for name, model in self.models.items():
            try:
                proba = model.predict_proba(X_test)
                # Handle case where model only saw one class during training
                if proba.shape[1] == 1:
                    logger.warning(f"{name}: predict_proba returned 1 column (only 1 class seen in train). "
                                   "Skipping — use anomaly detection instead.")
                    continue
                y_prob = proba[:, 1]
                y_pred = (y_prob >= threshold).astype(int)

                metrics = compute_detection_metrics(
                    y_test, y_pred, y_prob,
                    threshold=threshold,
                    model_name=name
                )
                self.all_metrics[name] = metrics
                print_metrics_report(metrics)

                # Track best model by F1
                if metrics["f1"] > best_f1:
                    best_f1 = metrics["f1"]
                    self.best_model_name = name
                    self.best_model = model

            except Exception as e:
                logger.error(f"Evaluation failed for {name}: {e}")

        logger.info(f"Best model: {self.best_model_name} (F1={best_f1:.4f})")

        # Save metrics
        self._save_metrics()
        return self.all_metrics

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict using the best model.

        Returns:
            (predictions, probabilities) tuple
        """
        if self.best_model is None:
            raise RuntimeError("No model trained or selected. Call train() and evaluate() first.")

        y_prob = self.best_model.predict_proba(X)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
        return y_pred, y_prob

    def _save_metrics(self) -> None:
        """Save metrics to JSON file."""
        metrics_path = self.metrics_dir / "detection_metrics.json"

        # Convert to JSON-serializable format
        serializable = {}
        for model_name, metrics in self.all_metrics.items():
            serializable[model_name] = {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in metrics.items()
            }

        with open(metrics_path, "w") as f:
            json.dump(serializable, f, indent=2, default=str)
        logger.info(f"Metrics saved to: {metrics_path}")

    def save_best_model(self, path: str = None) -> str:
        """Save best model to disk."""
        if self.best_model is None:
            raise RuntimeError("No best model to save")

        save_path = path or str(self.outputs_dir / f"best_model_{self.best_model_name.lower()}.joblib")
        joblib.dump({
            "model": self.best_model,
            "model_name": self.best_model_name,
            "metrics": self.all_metrics.get(self.best_model_name, {}),
        }, save_path)
        logger.info(f"Best model saved: {save_path}")
        return save_path

    def save_all_models(self) -> Dict[str, str]:
        """Save all trained models."""
        saved = {}
        for name, model in self.models.items():
            path = str(self.outputs_dir / f"model_{name.lower()}.joblib")
            joblib.dump(model, path)
            saved[name] = path
        logger.info(f"Saved {len(saved)} models to {self.outputs_dir}")
        return saved

    def load_best_model(self, path: str) -> None:
        """Load best model from disk."""
        state = joblib.load(path)
        self.best_model = state["model"]
        self.best_model_name = state["model_name"]
        logger.info(f"Loaded model: {self.best_model_name} from {path}")

    def get_feature_importance(self, model_name: str = None) -> pd.DataFrame:
        """Get feature importance DataFrame for a model."""
        name = model_name or self.best_model_name
        if name in self.feature_importances:
            return self.feature_importances[name]
        return pd.DataFrame()
