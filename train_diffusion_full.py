"""
Full Production Diffusion Model Training Script
================================================
Trains a class-conditioned DDPM on ALL HAI dataset versions (20.07 → 23.05)
using the 38 common sensor features, graph-informed signal groupings, and
proper training settings (100 epochs, cosine LR schedule, EMA).

Usage
-----
    python train_diffusion_full.py [options]

Options
-------
    --hai-root    PATH   Root of HAI dataset folder
                         (default: C:/Users/PC GAMING/Desktop/AI/HAI)
    --graph-dir   PATH   Graph JSON directory
                         (default: <hai-root>/graph/graph/boiler)
    --epochs      INT    Training epochs (default: 100)
    --batch-size  INT    Batch size (default: 256)
    --hidden-dim  INT    Model hidden dim (default: 256)
    --timesteps   INT    Diffusion timesteps T (default: 300)
    --lr          FLOAT  Learning rate (default: 2e-4)
    --max-rows    INT    Max rows per version (default: 300_000)
    --output-dir  PATH   Where to save model/results
                         (default: outputs/models)
    --no-haiend         Skip haiend-23.05 (it has different columns)
    --fast               Quick smoke-test: 5 epochs, 50k rows/version
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# ---- Project root on path ---------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.logger import logger
from src.data.multi_version_loader import (
    MultiVersionLoader, COMMON_FEATURES,
    load_graph_topology, graph_feature_groups,
)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="HAI Full Diffusion Training")
    p.add_argument("--hai-root", default="C:/Users/PC GAMING/Desktop/AI/HAI")
    p.add_argument("--graph-dir", default=None,
                   help="Graph JSON dir (default: <hai-root>/graph/graph/boiler)")
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--batch-size", type=int,   default=256)
    p.add_argument("--hidden-dim", type=int,   default=256)
    p.add_argument("--timesteps",  type=int,   default=300)
    p.add_argument("--lr",         type=float, default=2e-4)
    p.add_argument("--max-rows",   type=int,   default=300_000)
    p.add_argument("--output-dir", default="outputs/models")
    p.add_argument("--no-haiend", action="store_true")
    p.add_argument("--fast",      action="store_true",
                   help="Quick test: 5 epochs, 50k rows/version")
    return p.parse_args()


# ---------------------------------------------------------------------------
# EMA helper (Exponential Moving Average for model weights)
# ---------------------------------------------------------------------------

class EMA:
    """Simple EMA wrapper for a PyTorch model."""

    def __init__(self, model, decay: float = 0.9999):
        import torch
        self.decay = decay
        self.shadow = {k: v.clone().detach()
                       for k, v in model.state_dict().items()}

    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v.detach()

    def apply(self, model):
        model.load_state_dict(self.shadow)


# ---------------------------------------------------------------------------
# Improved NoisePredictor with wider architecture
# ---------------------------------------------------------------------------

def build_model(input_dim: int, hidden_dim: int, n_layers: int, n_classes: int = 3):
    """Build an improved DDPM noise predictor."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        raise RuntimeError("PyTorch is required for diffusion training.")

    class SinEmb(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim

        def forward(self, t):
            device = t.device
            half = self.dim // 2
            freq = torch.exp(
                torch.arange(half, device=device) * -(np.log(10000) / (half - 1))
            )
            emb = t.float()[:, None] * freq[None, :]
            return torch.cat([emb.sin(), emb.cos()], dim=-1)

    class ResBlock(nn.Module):
        def __init__(self, dim, cond_dim):
            super().__init__()
            self.norm = nn.LayerNorm(dim)
            self.fc1  = nn.Linear(dim, dim * 2)
            self.fc2  = nn.Linear(dim * 2, dim)
            self.act  = nn.GELU()
            self.drop = nn.Dropout(0.1)
            self.cond_proj = nn.Linear(cond_dim, dim)

        def forward(self, x, cond):
            h = self.norm(x)
            h = h + self.cond_proj(cond)   # inject conditioning
            h = self.fc2(self.drop(self.act(self.fc1(h))))
            return x + h

    class DiffNet(nn.Module):
        def __init__(self):
            super().__init__()
            cond_dim = hidden_dim
            self.time_emb  = SinEmb(hidden_dim)
            self.class_emb = nn.Embedding(n_classes, hidden_dim)
            self.time_proj  = nn.Sequential(nn.Linear(hidden_dim, cond_dim), nn.SiLU(),
                                            nn.Linear(cond_dim, cond_dim))
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            self.blocks     = nn.ModuleList([
                ResBlock(hidden_dim, cond_dim) for _ in range(n_layers)
            ])
            self.out = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, input_dim),
            )

        def forward(self, x, t, c=None):
            t_emb = self.time_proj(self.time_emb(t))
            if c is not None:
                t_emb = t_emb + self.class_emb(c)
            h = self.input_proj(x)
            for blk in self.blocks:
                h = blk(h, t_emb)
            return self.out(h)

    return DiffNet()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_diffusion(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int,
    batch_size: int,
    hidden_dim: int,
    T: int,
    lr: float,
    output_dir: Path,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
):
    try:
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        logger.error("PyTorch not installed. Install with: pip install torch")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training device: {device}")

    # Normalize data
    data_mean = X.mean(axis=0).astype(np.float32)
    data_std  = X.std(axis=0).astype(np.float32) + 1e-8
    X_norm = ((X - data_mean) / data_std).astype(np.float32)

    input_dim = X_norm.shape[1]

    # Diffusion schedule
    betas = torch.linspace(beta_start, beta_end, T, device=device)
    alphas = 1.0 - betas
    alphas_cp = torch.cumprod(alphas, dim=0)
    sqrt_alpha_cp = torch.sqrt(alphas_cp)
    sqrt_one_minus = torch.sqrt(1.0 - alphas_cp)

    # Model
    model = build_model(input_dim, hidden_dim, n_layers=6, n_classes=3).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    ema = EMA(model, decay=0.9999)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.1)

    # Dataset — clip y to [0,2] for class conditioning
    y_cond = np.clip(y, 0, 2).astype(np.int64)
    X_t  = torch.FloatTensor(X_norm)
    y_t  = torch.LongTensor(y_cond)
    ds   = TensorDataset(X_t, y_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        drop_last=True, num_workers=0, pin_memory=(device.type == "cuda"))

    history = []
    best_loss = float("inf")
    best_epoch = 0

    logger.info(f"Starting diffusion training: {epochs} epochs, "
                f"{len(ds):,} samples, batch={batch_size}")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for bx, bc in loader:
            bx = bx.to(device)
            bc = bc.to(device)

            # Random timestep per sample
            ts = torch.randint(0, T, (len(bx),), device=device)

            # Forward diffusion: q(x_t | x_0)
            noise = torch.randn_like(bx)
            sqrt_a  = sqrt_alpha_cp[ts].unsqueeze(1)
            sqrt_om = sqrt_one_minus[ts].unsqueeze(1)
            x_t = sqrt_a * bx + sqrt_om * noise

            # Predict noise
            pred = model(x_t, ts, bc)

            # SNR-weighted loss (prioritise harder timesteps)
            snr = alphas_cp[ts] / (1.0 - alphas_cp[ts])
            weight = (snr / snr.max()).unsqueeze(1)
            loss = (weight * (pred - noise) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema.update(model)

            epoch_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        history.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            # Save EMA checkpoint
            ema_state = dict(ema.shadow)
            torch.save({
                "model_state":  ema_state,
                "input_dim":    input_dim,
                "hidden_dim":   hidden_dim,
                "n_layers":     6,
                "T":            T,
                "data_mean":    data_mean,
                "data_std":     data_std,
                "beta_start":   beta_start,
                "beta_end":     beta_end,
                "features":     COMMON_FEATURES,
                "epoch":        epoch,
            }, output_dir / "diffusion_best.pt")

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t0
            logger.info(
                f"  Epoch {epoch:4d}/{epochs} | loss={avg_loss:.6f} "
                f"| best={best_loss:.6f} (ep {best_epoch}) "
                f"| lr={scheduler.get_last_lr()[0]:.2e} "
                f"| elapsed={elapsed/60:.1f}m"
            )

    # Save final model (with EMA weights)
    ema.apply(model)
    final_path = output_dir / "diffusion_final.pt"
    torch.save({
        "model_state":  model.state_dict(),
        "input_dim":    input_dim,
        "hidden_dim":   hidden_dim,
        "n_layers":     6,
        "T":            T,
        "data_mean":    data_mean,
        "data_std":     data_std,
        "beta_start":   beta_start,
        "beta_end":     beta_end,
        "features":     COMMON_FEATURES,
        "history":      history,
        "best_epoch":   best_epoch,
        "best_loss":    best_loss,
    }, final_path)
    logger.info(f"Final model saved: {final_path}")

    return {
        "model": model,
        "data_mean": data_mean,
        "data_std": data_std,
        "T": T,
        "betas": betas,
        "alphas": alphas,
        "alphas_cp": alphas_cp,
        "input_dim": input_dim,
        "device": device,
        "history": history,
        "final_path": str(final_path),
        "best_loss": best_loss,
    }


# ---------------------------------------------------------------------------
# Generation & evaluation
# ---------------------------------------------------------------------------

def generate_samples(state: dict, n_samples: int = 1000,
                     scenario_class: int = 1) -> np.ndarray:
    """Run reverse diffusion to generate synthetic sensor scenarios."""
    try:
        import torch
    except ImportError:
        return np.zeros((0, state["input_dim"]))

    model     = state["model"]
    T         = state["T"]
    device    = state["device"]
    betas     = state["betas"]
    alphas    = state["alphas"]
    alphas_cp = state["alphas_cp"]

    model.eval()

    with torch.no_grad():
        c = torch.full((n_samples,), scenario_class, dtype=torch.long, device=device)
        x = torch.randn(n_samples, state["input_dim"], device=device)

        for step in reversed(range(T)):
            t_batch = torch.full((n_samples,), step, dtype=torch.long, device=device)
            pred_noise = model(x, t_batch, c)

            alpha    = alphas[step]
            alpha_cp = alphas_cp[step]
            beta     = betas[step]

            coef1 = 1.0 / torch.sqrt(alpha)
            coef2 = (1.0 - alpha) / torch.sqrt(1.0 - alpha_cp)
            x = coef1 * (x - coef2 * pred_noise)

            if step > 0:
                x = x + torch.sqrt(beta) * torch.randn_like(x)

    synthetic = x.cpu().numpy()
    # Denormalise
    synthetic = synthetic * state["data_std"] + state["data_mean"]
    return synthetic


def evaluate_quality(real: np.ndarray, synth: np.ndarray) -> dict:
    """Statistical fidelity metrics."""
    metrics = {}
    metrics["mean_mae"]  = float(np.abs(real.mean(0) - synth.mean(0)).mean())
    metrics["std_mae"]   = float(np.abs(real.std(0)  - synth.std(0)).mean())

    n = min(len(real), len(synth), 5000)
    f = min(real.shape[1], 50)
    rc = np.corrcoef(real[:n, :f].T)
    sc = np.corrcoef(synth[:n, :f].T)
    rc, sc = np.nan_to_num(rc), np.nan_to_num(sc)
    diff = np.linalg.norm(rc - sc, "fro")
    metrics["corr_sim"] = float(1.0 - diff / (np.linalg.norm(rc, "fro") + 1e-8))

    logger.info(
        f"Quality — mean_mae={metrics['mean_mae']:.4f} "
        f"std_mae={metrics['std_mae']:.4f} "
        f"corr_sim={metrics['corr_sim']:.4f}"
    )
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.fast:
        args.epochs   = 5
        args.max_rows = 50_000
        logger.info("FAST mode: 5 epochs, 50k rows/version")

    hai_root  = args.hai_root
    graph_dir = args.graph_dir or str(Path(hai_root) / "graph" / "graph" / "boiler")
    out_dir   = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Load graph topology
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 1: Loading graph topology")
    logger.info("=" * 60)
    try:
        topology = load_graph_topology(graph_dir)
        feat_groups = graph_feature_groups(topology, COMMON_FEATURES)
        logger.info(f"  DCS modules: {list(topology['dcs_nodes'].keys())}")
        logger.info(f"  Physical nodes: {len(topology['phy_nodes'])}")
        logger.info(f"  Feature groups from graph: {len(feat_groups)}")
        # Save topology summary
        topo_summary = {
            "dcs_modules": {k: len(v) for k, v in topology["dcs_nodes"].items()},
            "physical_nodes": len(topology["phy_nodes"]),
            "unique_signals": len(topology["signal_names"]),
            "feature_groups": {k: v for k, v in feat_groups.items()},
        }
        with open(out_dir / "graph_topology_summary.json", "w") as f:
            json.dump(topo_summary, f, indent=2)
    except Exception as e:
        logger.warning(f"Graph loading failed: {e} — proceeding without graph info")
        topology = None

    # -----------------------------------------------------------------------
    # 2. Load data from all versions
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 2: Loading multi-version HAI dataset")
    logger.info("=" * 60)

    loader = MultiVersionLoader(hai_root)

    # Load training splits
    X_train, y_train = loader.load_all(
        versions=["hai-20.07", "hai-21.03", "hai-22.04", "hai-23.05"],
        split="train",
        features=COMMON_FEATURES,
        max_rows_per_version=args.max_rows,
    )

    # Load test splits (have attack labels → needed for conditioning)
    X_test, y_test = loader.load_all(
        versions=["hai-20.07", "hai-21.03", "hai-22.04", "hai-23.05"],
        split="test",
        features=COMMON_FEATURES,
        max_rows_per_version=args.max_rows // 4,
    )

    # Combine train + test
    X_all = np.vstack([X_train, X_test]).astype(np.float32)
    y_all = np.concatenate([y_train, y_test]).astype(np.int8)

    logger.info(f"Total combined: {len(X_all):,} samples x {X_all.shape[1]} features")
    logger.info(f"  Normal:  {int((y_all == 0).sum()):,} ({(y_all==0).mean()*100:.1f}%)")
    logger.info(f"  Attack:  {int((y_all == 1).sum()):,} ({(y_all==1).mean()*100:.1f}%)")

    # Save dataset statistics
    stats = {
        "total_samples": int(len(X_all)),
        "n_features": int(X_all.shape[1]),
        "n_normal":  int((y_all == 0).sum()),
        "n_attack":  int((y_all == 1).sum()),
        "attack_rate_pct": float(y_all.mean() * 100),
        "feature_names": COMMON_FEATURES,
        "versions_loaded": ["hai-20.07", "hai-21.03", "hai-22.04", "hai-23.05"],
    }
    with open(out_dir / "diffusion_dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # -----------------------------------------------------------------------
    # 3. Train diffusion model
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 3: Training DDPM diffusion model")
    logger.info("=" * 60)

    state = train_diffusion(
        X=X_all,
        y=y_all,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        T=args.timesteps,
        lr=args.lr,
        output_dir=out_dir,
    )

    if state is None:
        logger.error("Training failed (PyTorch not available).")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # 4. Generate synthetic samples & evaluate quality
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 4: Generating synthetic samples and evaluating quality")
    logger.info("=" * 60)

    synth_dir = Path("outputs/synthetic")
    synth_dir.mkdir(parents=True, exist_ok=True)

    quality_results = {}
    for scenario_class, name in [(0, "normal"), (1, "attack"), (2, "fault")]:
        synth = generate_samples(state, n_samples=2000, scenario_class=scenario_class)
        np.save(synth_dir / f"synthetic_{name}_full.npy", synth)
        logger.info(f"  Generated {len(synth)} {name} samples -> outputs/synthetic/")

        # Compare against real data of same class
        real_class = X_all[y_all == min(scenario_class, 1)]
        if len(real_class) > 100:
            quality = evaluate_quality(real_class, synth)
            quality_results[name] = quality

    # Save quality metrics
    with open(out_dir / "diffusion_quality_metrics.json", "w") as f:
        json.dump(quality_results, f, indent=2)

    # -----------------------------------------------------------------------
    # 5. Summary
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"  Best loss:       {state['best_loss']:.6f}")
    logger.info(f"  Model saved:     {state['final_path']}")
    logger.info(f"  Best checkpoint: {out_dir}/diffusion_best.pt")
    logger.info(f"  Synthetic data:  outputs/synthetic/")
    logger.info(f"  Quality metrics: {out_dir}/diffusion_quality_metrics.json")
    logger.info("=" * 60)

    return state


if __name__ == "__main__":
    main()
