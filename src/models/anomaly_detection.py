"""
Unsupervised Anomaly Detection Models

Used as:
1. Baseline comparison for supervised models
2. Anomaly scoring for digital twin
3. Fallback when labels are unavailable

Models:
- Isolation Forest
- LSTM Autoencoder (reconstruction error)
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

from src.utils.logger import logger

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not installed — autoencoder model unavailable")


class IsolationForestDetector:
    """Isolation Forest anomaly detector."""

    def __init__(self, config: Dict[str, Any]):
        cfg = config.get("models", {}).get("anomaly", {}).get("isolation_forest", {})
        self.model = IsolationForest(
            n_estimators=cfg.get("n_estimators", 200),
            contamination=cfg.get("contamination", 0.05),
            random_state=config.get("project", {}).get("seed", 42),
            n_jobs=-1,
        )
        self.threshold = None
        self._outputs_dir = Path(config["paths"]["models"])
        self._outputs_dir.mkdir(parents=True, exist_ok=True)

    def fit(self, X: np.ndarray) -> None:
        """Fit Isolation Forest on normal (training) data."""
        logger.info(f"Fitting Isolation Forest on {len(X):,} samples...")
        self.model.fit(X)
        logger.info("Isolation Forest fitted")

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores (higher = more anomalous)."""
        # decision_function returns negative anomaly scores (lower = more anomalous)
        # We negate to make higher = more anomalous
        raw_scores = self.model.decision_function(X)
        return -raw_scores  # now higher = more anomalous

    def predict(self, X: np.ndarray, threshold: float = None) -> np.ndarray:
        """Return binary predictions (1=anomaly)."""
        scores = self.score(X)
        if threshold is not None:
            return (scores >= threshold).astype(int)
        # Use built-in threshold
        raw_pred = self.model.predict(X)
        return (raw_pred == -1).astype(int)

    def save(self, path: str = None) -> str:
        save_path = path or str(self._outputs_dir / "isolation_forest.joblib")
        joblib.dump(self.model, save_path)
        logger.info(f"Isolation Forest saved: {save_path}")
        return save_path

    def load(self, path: str) -> None:
        self.model = joblib.load(path)
        logger.info(f"Isolation Forest loaded from: {path}")


class LSTMAutoencoder(nn.Module if HAS_TORCH else object):
    """
    LSTM-based autoencoder for time-series anomaly detection.

    Anomaly score = reconstruction error (MSE).
    High error -> sequence pattern doesn't match normal behavior -> anomaly.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for LSTMAutoencoder")
        super().__init__()

        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,
                               dropout=0.2 if num_layers > 1 else 0.0)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True,
                               dropout=0.2 if num_layers > 1 else 0.0)
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
        """Encode then decode sequence."""
        batch_size, seq_len, _ = x.size()

        # Encode
        _, (hidden, cell) = self.encoder(x)

        # Use last hidden state as context, repeat for decoder
        context = hidden[-1].unsqueeze(1).repeat(1, seq_len, 1)

        # Decode
        decoded, _ = self.decoder(context)
        output = self.output_layer(decoded)
        return output


class AutoencoderDetector:
    """Wrapper for LSTM Autoencoder anomaly detection."""

    def __init__(self, config: Dict[str, Any]):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required")

        cfg = config.get("models", {}).get("anomaly", {}).get("autoencoder", {})
        self.cfg = cfg
        self.epochs = cfg.get("epochs", 50)
        self.batch_size = cfg.get("batch_size", 256)
        self.lr = cfg.get("learning_rate", 0.001)
        self.threshold_pct = cfg.get("threshold_percentile", 95)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model: Optional[LSTMAutoencoder] = None
        self.threshold: float = None
        self.scaler = MinMaxScaler()

        self._outputs_dir = Path(config["paths"]["models"])
        self._outputs_dir.mkdir(parents=True, exist_ok=True)

    def _to_tensor(self, X: np.ndarray) -> 'torch.Tensor':
        return torch.FloatTensor(X).to(self.device)

    def fit(self, X: np.ndarray, sequence_length: int = 60) -> None:
        """
        Train autoencoder on normal data windows.

        Args:
            X: Normal training data (n_samples, n_features)
            sequence_length: Window length for LSTM input
        """
        logger.info(f"Training LSTM Autoencoder: {X.shape}, device={self.device}")

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Build sequences
        n_features = X.shape[1]
        sequences = []
        for i in range(0, len(X_scaled) - sequence_length, sequence_length):
            sequences.append(X_scaled[i:i+sequence_length])

        if len(sequences) == 0:
            logger.warning("Not enough data for autoencoder sequences")
            return

        X_seq = torch.FloatTensor(np.array(sequences)).to(self.device)
        dataset = TensorDataset(X_seq, X_seq)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Initialize model
        self.model = LSTMAutoencoder(n_features, hidden_dim=64, num_layers=2).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                output = self.model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(loader)
                logger.info(f"  Epoch [{epoch+1}/{self.epochs}] Loss: {avg_loss:.6f}")

        # Compute threshold from training reconstruction errors
        train_errors = self._compute_errors(X_scaled, sequence_length)
        self.threshold = np.percentile(train_errors, self.threshold_pct)
        logger.info(f"Autoencoder trained. Threshold (p{self.threshold_pct}): {self.threshold:.6f}")

    def _compute_errors(self, X_scaled: np.ndarray, sequence_length: int = 60) -> np.ndarray:
        """Compute per-sample reconstruction errors."""
        if self.model is None:
            raise RuntimeError("Model not trained")

        self.model.eval()
        errors = []

        with torch.no_grad():
            for i in range(0, len(X_scaled) - sequence_length, sequence_length):
                seq = torch.FloatTensor(X_scaled[i:i+sequence_length]).unsqueeze(0).to(self.device)
                reconstruction = self.model(seq)
                error = ((seq - reconstruction) ** 2).mean(dim=(1, 2)).cpu().numpy()
                errors.extend([float(error[0])] * sequence_length)

        return np.array(errors[:len(X_scaled)])

    def score(self, X: np.ndarray, sequence_length: int = 60) -> np.ndarray:
        """Return reconstruction error scores."""
        X_scaled = self.scaler.transform(X)
        return self._compute_errors(X_scaled, sequence_length)

    def predict(self, X: np.ndarray, sequence_length: int = 60) -> np.ndarray:
        """Return binary anomaly predictions."""
        scores = self.score(X, sequence_length)
        if self.threshold is not None:
            return (scores >= self.threshold).astype(int)
        return (scores >= np.percentile(scores, 95)).astype(int)

    def save(self, path: str = None) -> str:
        if self.model is None:
            raise RuntimeError("No model to save")
        save_path = path or str(self._outputs_dir / "autoencoder.pt")
        torch.save({
            "model_state": self.model.state_dict(),
            "threshold": self.threshold,
            "scaler": self.scaler,
        }, save_path)
        logger.info(f"Autoencoder saved: {save_path}")
        return save_path
