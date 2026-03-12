"""
LSTM-based Attack Detection Model for HAI time-series data.

Uses sequence windowing to capture temporal patterns that simple
tabular models may miss, especially for slow/gradual attacks.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from src.utils.logger import logger

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class LSTMDetector(nn.Module if HAS_TORCH else object):
    """
    Bidirectional LSTM for attack sequence classification.

    Architecture:
    - BiLSTM encoder -> captures forward and backward temporal patterns
    - Attention pooling -> focuses on most anomalous timesteps
    - FC layers -> classification head
    """

    def __init__(self, input_dim: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.3):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for LSTMDetector")
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Attention layer for sequence pooling
        self.attention = nn.Linear(hidden_size * 2, 1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
        """Forward pass returning probability of attack."""
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden*2)

        # Attention weights
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)  # (batch, seq, 1)
        context = (attn_weights * lstm_out).sum(dim=1)  # (batch, hidden*2)

        logit = self.classifier(context).squeeze(-1)
        return torch.sigmoid(logit)


class LSTMAttackDetector:
    """Training/inference wrapper for LSTM attack detector."""

    def __init__(self, config: Dict[str, Any]):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required")

        self.config = config
        cfg = config.get("models", {}).get("detection", {}).get("lstm", {})
        self.seq_len = cfg.get("sequence_length", 60)
        self.hidden_size = cfg.get("hidden_size", 128)
        self.num_layers = cfg.get("num_layers", 2)
        self.dropout = cfg.get("dropout", 0.3)
        self.batch_size = cfg.get("batch_size", 256)
        self.epochs = cfg.get("epochs", 30)
        self.lr = cfg.get("learning_rate", 0.001)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[LSTMDetector] = None
        self.best_threshold: float = 0.5

        self._outputs_dir = Path(config["paths"]["models"])
        self._outputs_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"LSTM detector using device: {self.device}")

    def _make_dataset(self, X: np.ndarray, y: np.ndarray) -> TensorDataset:
        """Convert arrays to PyTorch dataset."""
        X_t = torch.FloatTensor(X).to(self.device)
        y_t = torch.FloatTensor(y).to(self.device)
        return TensorDataset(X_t, y_t)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None) -> None:
        """
        Train LSTM on windowed sequences.

        Args:
            X_train: (n_windows, seq_len, n_features)
            y_train: (n_windows,) binary labels
            X_val: Validation windows
            y_val: Validation labels
        """
        input_dim = X_train.shape[2] if X_train.ndim == 3 else X_train.shape[1]
        self.model = LSTMDetector(input_dim, self.hidden_size, self.num_layers, self.dropout).to(self.device)

        # Class weight for imbalanced data
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)]).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Re-build without sigmoid for BCEWithLogitsLoss
        # Use BCE directly on sigmoid output
        criterion = nn.BCELoss()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        train_dataset = self._make_dataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        best_val_f1 = -1.0
        best_state = None

        logger.info(f"Training LSTM: {X_train.shape}, epochs={self.epochs}")

        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            # Validation phase
            if X_val is not None and y_val is not None:
                val_preds, val_probs = self.predict(X_val)
                from sklearn.metrics import f1_score
                val_f1 = f1_score(y_val, val_preds, zero_division=0)
                scheduler.step(1 - val_f1)

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

                if (epoch + 1) % 5 == 0:
                    logger.info(f"  Epoch [{epoch+1}/{self.epochs}] "
                               f"Loss: {train_loss/len(train_loader):.4f} "
                               f"Val F1: {val_f1:.4f}")
            else:
                if (epoch + 1) % 5 == 0:
                    logger.info(f"  Epoch [{epoch+1}/{self.epochs}] "
                               f"Loss: {train_loss/len(train_loader):.4f}")

        # Restore best checkpoint
        if best_state is not None:
            self.model.load_state_dict(best_state)
            logger.info(f"LSTM training complete. Best val F1: {best_val_f1:.4f}")
        else:
            logger.info("LSTM training complete")

    def predict(self, X: np.ndarray, threshold: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """Return (predictions, probabilities)."""
        if self.model is None:
            raise RuntimeError("Model not trained")

        self.model.eval()
        threshold = threshold or self.best_threshold

        X_t = torch.FloatTensor(X).to(self.device)
        all_probs = []

        with torch.no_grad():
            for i in range(0, len(X_t), self.batch_size):
                batch = X_t[i:i+self.batch_size]
                probs = self.model(batch).cpu().numpy()
                all_probs.extend(probs)

        y_prob = np.array(all_probs)
        y_pred = (y_prob >= threshold).astype(int)
        return y_pred, y_prob

    def save(self, path: str = None) -> str:
        save_path = path or str(self._outputs_dir / "lstm_detector.pt")
        torch.save({
            "model_state": self.model.state_dict(),
            "config": {
                "input_dim": next(self.model.parameters()).shape[-1] if self.model else 0,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
            },
            "threshold": self.best_threshold,
        }, save_path)
        logger.info(f"LSTM model saved: {save_path}")
        return save_path
