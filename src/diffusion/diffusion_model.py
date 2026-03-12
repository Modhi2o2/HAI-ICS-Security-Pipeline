"""
Lightweight DDPM (Denoising Diffusion Probabilistic Model) for HAI Time-Series

Architecture:
- Forward process: gradually add Gaussian noise over T steps
- Reverse process: learn to denoise step by step using a simple MLP/Transformer
- Conditioned generation: can condition on scenario type (normal, attack, fault)

For the HAI dataset, we use a compact UNet-style MLP as the noise predictor.
This is much lighter than pixel-space diffusion but captures temporal structure.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

from src.utils.logger import logger

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not installed — diffusion model unavailable")


class SinusoidalEmbedding(nn.Module if HAS_TORCH else object):
    """Sinusoidal timestep embeddings (same as in DDPM paper)."""

    def __init__(self, dim: int):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required")
        super().__init__()
        self.dim = dim

    def forward(self, t: 'torch.Tensor') -> 'torch.Tensor':
        device = t.device
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class NoisePredictor(nn.Module if HAS_TORCH else object):
    """
    MLP-based noise predictor for tabular/time-series diffusion.

    Input: noisy sample x_t + timestep embedding + optional conditioning
    Output: predicted noise epsilon
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256,
                 n_layers: int = 4, n_classes: int = 3):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required")
        super().__init__()

        self.time_embed = SinusoidalEmbedding(hidden_dim)
        self.class_embed = nn.Embedding(n_classes, hidden_dim)  # conditioning: 0=normal, 1=attack, 2=fault

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Residual blocks
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Dropout(0.1),
            ))

        # Output head
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, input_dim),
        )

        self.n_classes = n_classes

    def forward(self, x: 'torch.Tensor', t: 'torch.Tensor',
                c: 'torch.Tensor' = None) -> 'torch.Tensor':
        """
        Args:
            x: Noisy input (batch, input_dim)
            t: Timestep indices (batch,)
            c: Conditioning class (batch,) — 0=normal, 1=attack, 2=fault
        Returns:
            Predicted noise (batch, input_dim)
        """
        h = self.input_proj(x)
        t_emb = self.time_embed(t)
        h = h + t_emb

        if c is not None:
            c_emb = self.class_embed(c)
            h = h + c_emb

        for layer in self.layers:
            h = h + layer(h)  # residual connection

        return self.output_head(h)


class HAIDiffusionModel:
    """
    DDPM-based scenario generator for HAI ICS data.

    Generates:
    - Normal operation scenarios (class 0)
    - Cyberattack scenarios (class 1)
    - Equipment fault scenarios (class 2)

    Training strategy:
    - Train on real HAI sensor windows
    - Condition on attack label
    - At inference: generate class-conditioned synthetic windows
    """

    def __init__(self, config: Dict[str, Any]):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for diffusion model")

        self.config = config
        diff_cfg = config.get("diffusion", {})

        self.T = diff_cfg.get("timesteps", 200)
        self.hidden_dim = diff_cfg.get("hidden_dim", 256)
        self.num_layers = diff_cfg.get("num_layers", 4)
        self.batch_size = diff_cfg.get("batch_size", 128)
        self.epochs = diff_cfg.get("epochs", 50)
        self.lr = diff_cfg.get("learning_rate", 0.0002)
        self.beta_start = diff_cfg.get("beta_start", 0.0001)
        self.beta_end = diff_cfg.get("beta_end", 0.02)
        self.seq_len = diff_cfg.get("sequence_length", 60)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Precompute diffusion schedule
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.T).to(self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        self.model: Optional[NoisePredictor] = None
        self.input_dim: Optional[int] = None

        # For data normalization
        self.data_mean: Optional[np.ndarray] = None
        self.data_std: Optional[np.ndarray] = None

        self._outputs_dir = Path(config["paths"]["synthetic"])
        self._outputs_dir.mkdir(parents=True, exist_ok=True)
        self._models_dir = Path(config["paths"]["models"])
        self._models_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"HAIDiffusionModel initialized: T={self.T}, device={self.device}")

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        """Z-score normalize using stored statistics."""
        return (X - self.data_mean) / (self.data_std + 1e-8)

    def _denormalize(self, X: np.ndarray) -> np.ndarray:
        """Reverse normalization."""
        return X * (self.data_std + 1e-8) + self.data_mean

    def _q_sample(self, x0: 'torch.Tensor', t: 'torch.Tensor') -> Tuple['torch.Tensor', 'torch.Tensor']:
        """Forward diffusion: add noise at timestep t."""
        noise = torch.randn_like(x0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        x_t = sqrt_alpha * x0 + sqrt_one_minus * noise
        return x_t, noise

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> None:
        """
        Train diffusion model on sensor windows.

        Args:
            X: Sensor data (n_samples, n_features) — can be raw rows or pre-windowed
            y: Labels (0=normal, 1=attack) — used for conditioning
        """
        logger.info(f"Training HAI Diffusion Model on {len(X):,} samples...")

        # Store normalization stats
        self.data_mean = X.mean(axis=0)
        self.data_std = X.std(axis=0)

        X_norm = self._normalize(X)
        self.input_dim = X_norm.shape[1]

        # Build model
        self.model = NoisePredictor(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.num_layers,
            n_classes=3,
        ).to(self.device)

        # Prepare conditioning labels
        if y is None:
            y_cond = np.zeros(len(X), dtype=np.int64)
        else:
            y_cond = y.astype(np.int64).clip(0, 1)  # 0=normal, 1=attack

        # Dataset
        X_tensor = torch.FloatTensor(X_norm).to(self.device)
        y_tensor = torch.LongTensor(y_cond).to(self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_x, batch_c in loader:
                # Sample random timesteps
                t = torch.randint(0, self.T, (len(batch_x),), device=self.device)

                # Forward diffusion
                x_t, noise = self._q_sample(batch_x, t)

                # Predict noise (with class conditioning)
                pred_noise = self.model(x_t, t, batch_c)

                # Simple MSE loss on noise prediction
                loss = F.mse_loss(pred_noise, noise)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()

            scheduler.step()

            if (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / len(loader)
                logger.info(f"  Diffusion Epoch [{epoch+1}/{self.epochs}] Loss: {avg_loss:.6f}")

        logger.info("Diffusion model training complete")

    @torch.no_grad()
    def generate(self, n_samples: int = 100, scenario_class: int = 1) -> np.ndarray:
        """
        Generate synthetic sensor scenarios using reverse diffusion.

        Args:
            n_samples: Number of synthetic samples to generate
            scenario_class: 0=normal, 1=attack, 2=fault/degradation

        Returns:
            Synthetic samples (n_samples, n_features) in original scale
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        logger.info(f"Generating {n_samples} synthetic samples (class={scenario_class})...")

        self.model.eval()
        c = torch.full((n_samples,), scenario_class, dtype=torch.long, device=self.device)

        # Start from pure noise
        x = torch.randn(n_samples, self.input_dim, device=self.device)

        # Reverse diffusion: denoise step by step
        for t_step in reversed(range(self.T)):
            t_batch = torch.full((n_samples,), t_step, dtype=torch.long, device=self.device)

            # Predict noise
            pred_noise = self.model(x, t_batch, c)

            # Compute reverse step
            alpha = self.alphas[t_step]
            alpha_cumprod = self.alphas_cumprod[t_step]
            beta = self.betas[t_step]

            # DDPM reverse formula
            coef1 = 1.0 / torch.sqrt(alpha)
            coef2 = (1.0 - alpha) / torch.sqrt(1.0 - alpha_cumprod)
            x = coef1 * (x - coef2 * pred_noise)

            # Add noise for t > 0
            if t_step > 0:
                noise = torch.randn_like(x)
                x = x + torch.sqrt(beta) * noise

        synthetic = x.cpu().numpy()
        synthetic = self._denormalize(synthetic)

        logger.info(f"Generated {len(synthetic)} synthetic samples")
        return synthetic

    def generate_attack_scenario(self, n_samples: int = 500,
                                  attack_type: str = "cyberattack") -> np.ndarray:
        """
        Generate specific attack scenario type.

        Args:
            n_samples: Number of samples
            attack_type: 'cyberattack', 'sensor_drift', 'spike',
                         'degradation', 'communication_loss', 'abnormal'
        Returns:
            Synthetic attack samples (original scale)
        """
        if attack_type in ("cyberattack", "spike"):
            synthetic = self.generate(n_samples, scenario_class=1)
        elif attack_type in ("degradation", "sensor_drift"):
            synthetic = self.generate(n_samples, scenario_class=2)
        elif attack_type == "communication_loss":
            # Communication loss: many sensors go to zero or last known value
            synthetic = self.generate(n_samples, scenario_class=2)
            if self.data_mean is not None:
                # Zero out random sensor groups (communication blackout effect)
                n_features = synthetic.shape[1]
                blackout_cols = np.random.choice(n_features, size=n_features // 3, replace=False)
                synthetic[:, blackout_cols] = self.data_mean[blackout_cols] * 0.0
        elif attack_type == "abnormal":
            # Mix of attack and fault classes
            n_attack = n_samples // 2
            n_fault = n_samples - n_attack
            synthetic = np.vstack([
                self.generate(n_attack, scenario_class=1),
                self.generate(n_fault, scenario_class=2)
            ])
        else:
            synthetic = self.generate(n_samples, scenario_class=1)

        # Save generated samples
        save_path = self._outputs_dir / f"synthetic_{attack_type}.npy"
        np.save(save_path, synthetic)
        logger.info(f"Saved {attack_type} synthetic scenario: {save_path}")

        return synthetic

    def evaluate_quality(self, real_data: np.ndarray, synthetic_data: np.ndarray) -> Dict[str, float]:
        """
        Evaluate synthetic data quality vs real data.

        Metrics:
        - Mean absolute difference of feature means
        - Mean absolute difference of feature stds
        - Correlation matrix similarity (Frobenius norm)
        """
        metrics = {}

        # Statistical similarity
        real_means = real_data.mean(axis=0)
        synth_means = synthetic_data.mean(axis=0)
        metrics["mean_diff_abs"] = float(np.abs(real_means - synth_means).mean())

        real_stds = real_data.std(axis=0)
        synth_stds = synthetic_data.std(axis=0)
        metrics["std_diff_abs"] = float(np.abs(real_stds - synth_stds).mean())

        # Correlation similarity
        min_samples = min(len(real_data), len(synthetic_data), 5000)
        real_sample = real_data[:min_samples]
        synth_sample = synthetic_data[:min_samples]

        # Only use finite features
        n_feats = min(50, real_data.shape[1])  # limit for efficiency
        real_corr = np.corrcoef(real_sample[:, :n_feats].T)
        synth_corr = np.corrcoef(synth_sample[:, :n_feats].T)

        # Handle NaN in correlation
        real_corr = np.nan_to_num(real_corr, nan=0.0)
        synth_corr = np.nan_to_num(synth_corr, nan=0.0)

        frob_diff = np.linalg.norm(real_corr - synth_corr, 'fro')
        max_frob = np.linalg.norm(real_corr, 'fro') + 1e-8
        metrics["correlation_similarity"] = float(1.0 - frob_diff / max_frob)

        logger.info(f"Synthetic quality — mean_diff: {metrics['mean_diff_abs']:.4f}, "
                   f"std_diff: {metrics['std_diff_abs']:.4f}, "
                   f"corr_sim: {metrics['correlation_similarity']:.4f}")
        return metrics

    def save(self, path: str = None) -> str:
        """Save diffusion model."""
        if self.model is None:
            raise RuntimeError("No model to save")
        save_path = path or str(self._models_dir / "diffusion_model.pt")
        torch.save({
            "model_state": self.model.state_dict(),
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "T": self.T,
            "data_mean": self.data_mean,
            "data_std": self.data_std,
        }, save_path)
        logger.info(f"Diffusion model saved: {save_path}")
        return save_path

    def load(self, path: str) -> None:
        """Load saved diffusion model."""
        state = torch.load(path, map_location=self.device)
        self.input_dim = state["input_dim"]
        self.hidden_dim = state["hidden_dim"]
        self.num_layers = state["num_layers"]
        self.T = state["T"]
        self.data_mean = state["data_mean"]
        self.data_std = state["data_std"]

        self.model = NoisePredictor(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.num_layers,
        ).to(self.device)
        self.model.load_state_dict(state["model_state"])
        self.model.eval()
        logger.info(f"Diffusion model loaded from: {path}")
