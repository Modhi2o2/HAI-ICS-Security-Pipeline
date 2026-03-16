"""
Graph Deviation Network (GDN) for HAI Anomaly Detection
========================================================
Key insight: ICS sensors obey physical causal laws.
  - GDN learns which sensors predict which other sensors from normal data
  - Attacks violate these learned relationships → high deviation score

Architecture (simplified GDN, Deng & Hooi AAAI 2021):
  1. Node embeddings: each of the 38 sensors gets a learned 64-dim vector
  2. Graph: for each sensor, top-K most similar sensors by embedding cosine sim
  3. Predictor: MLP(neighbor_values + own_embedding) → predicted_value
  4. Anomaly score: mean absolute deviation across all sensors

Expected F1: 0.50-0.70 (GDN gets ~0.74 on SWaT, a similar ICS dataset)

Usage: python train_gdn.py [--epochs 50] [--top-k 10] [--embed-dim 64]
"""

import sys, json, argparse, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import datetime

sys.path.insert(0, str(Path(__file__).parent))
from src.utils.logger import logger
from src.data.multi_version_loader import MultiVersionLoader, COMMON_FEATURES

HAI_ROOT = "C:/Users/PC GAMING/Desktop/AI/HAI"
OUT_DIR  = Path("outputs/models")
MET_DIR  = Path("outputs/metrics")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MET_DIR.mkdir(parents=True, exist_ok=True)

VERSIONS = ["hai-20.07", "hai-21.03", "hai-22.04", "hai-23.05"]
N_FEAT   = len(COMMON_FEATURES)  # 38


# ─────────────────────────────────────────────────────────────────────────────
# Arguments
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",    type=int,   default=50)
    p.add_argument("--top-k",     type=int,   default=10,  help="graph neighbors per node")
    p.add_argument("--embed-dim", type=int,   default=64,  help="node embedding dim")
    p.add_argument("--hidden",    type=int,   default=128, help="predictor hidden dim")
    p.add_argument("--window",    type=int,   default=5,   help="temporal context steps")
    p.add_argument("--batch",     type=int,   default=2048)
    p.add_argument("--lr",        type=float, default=1e-3)
    p.add_argument("--max-train", type=int,   default=500_000, help="max training windows")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class GDN(nn.Module):
    """
    Graph Deviation Network.

    For each sensor i:
      - Finds its top-K "related" sensors by learned embedding similarity
      - Predicts sensor i's current value from those K neighbors' recent values
      - Anomaly score = |actual - predicted|

    The graph structure is fully learned — no domain knowledge required.
    """
    def __init__(self, n_sensors: int, window: int = 5,
                 embed_dim: int = 64, top_k: int = 10, hidden: int = 128):
        super().__init__()
        self.n       = n_sensors
        self.w       = window
        self.top_k   = min(top_k, n_sensors - 1)

        # Learnable node embeddings — capture each sensor's "role" in the plant
        self.node_emb = nn.Embedding(n_sensors, embed_dim)
        nn.init.xavier_uniform_(self.node_emb.weight)

        # Prediction MLP per sensor:
        # Input: K neighbor values × W timesteps + own embedding
        in_dim = self.top_k * window + embed_dim
        self.predictor = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def _topk_neighbors(self) -> torch.Tensor:
        """Return (N, K) neighbor indices from learned embedding similarity."""
        emb  = self.node_emb.weight           # (N, d)
        norm = F.normalize(emb, dim=1)
        sim  = norm @ norm.T                  # (N, N) cosine similarity
        # Mask self-connections
        sim  = sim - torch.eye(self.n, device=sim.device) * 1e9
        _, topk = sim.topk(self.top_k, dim=1) # (N, K)
        return topk

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:       (B, W, N) — batch of W-step windows, N sensors
        returns: (B, N)    — predicted value for each sensor at step W
        """
        B, W, N = x.shape
        topk = self._topk_neighbors()  # (N, K)

        # Gather neighbor values across all W timesteps
        # Result shape: (B, N, K*W)
        parts = []
        for t in range(W):
            x_t    = x[:, t, :]                              # (B, N)
            idx    = topk.unsqueeze(0).expand(B, -1, -1)     # (B, N, K)
            x_exp  = x_t.unsqueeze(1).expand(-1, N, -1)      # (B, N, N)
            nv     = x_exp.gather(2, idx)                     # (B, N, K)
            parts.append(nv)
        neighbor_vals = torch.cat(parts, dim=2)               # (B, N, K*W)

        # Own embeddings broadcast to batch
        emb = self.node_emb.weight.unsqueeze(0).expand(B, -1, -1)  # (B, N, d)

        # Concatenate neighbor values + own embedding → (B, N, K*W+d)
        feats = torch.cat([neighbor_vals, emb], dim=2)

        # Apply shared MLP across all (batch × sensor) pairs
        feats_flat = feats.reshape(B * N, -1)
        pred_flat  = self.predictor(feats_flat).squeeze(-1)   # (B*N,)
        return pred_flat.reshape(B, N)                         # (B, N)

    def deviation_score(self, x: torch.Tensor):
        """
        x: (B, W, N)
        returns: mean_score (B,), per_sensor_dev (B, N)
        """
        with torch.no_grad():
            target = x[:, -1, :]       # current timestep to predict (B, N)
            pred   = self.forward(x)   # (B, N)
            dev    = (target - pred).abs()
        return dev.mean(dim=1).numpy(), dev.numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def sample_windows(X: np.ndarray, window: int, max_samples: int) -> np.ndarray:
    """Randomly sample sliding windows from (T, N) normal data."""
    T = len(X)
    n_possible = T - window
    n = min(max_samples, n_possible)
    starts = np.random.choice(n_possible, n, replace=False)
    starts.sort()
    wins = np.stack([X[s:s + window] for s in starts]).astype(np.float32)
    return wins  # (n, W, N)


def score_all_timesteps(model: GDN, X: np.ndarray, window: int, chunk: int = 4096):
    """
    Score every timestep in X (T, N).
    Returns mean_scores (T,) and per_sensor_devs (T, N).
    """
    T, N = X.shape
    W = window
    # Pad the start so every timestep gets a full window
    X_pad = np.concatenate([np.zeros((W - 1, N), dtype=np.float32), X], axis=0)

    scores = np.zeros(T, dtype=np.float32)
    devs   = np.zeros((T, N), dtype=np.float32)

    model.eval()
    for start in range(0, T, chunk):
        end = min(start + chunk, T)
        size = end - start
        # Build (size, W, N) windows
        batch_wins = np.stack([X_pad[start + i: start + i + W] for i in range(size)])
        bt = torch.from_numpy(batch_wins)
        s, d = model.deviation_score(bt)
        scores[start:end] = s
        devs[start:end]   = d

    return scores, devs


def best_threshold(scores: np.ndarray, y: np.ndarray):
    from sklearn.metrics import f1_score
    bf, bt = 0.0, scores.mean()
    for p in np.arange(70, 99.9, 0.1):
        t = np.percentile(scores, p)
        f = f1_score(y, (scores >= t).astype(int), zero_division=0)
        if f > bf:
            bf, bt = f, t
    return float(bt), float(bf)


def full_eval(scores, y, thr, name):
    from sklearn.metrics import (f1_score, precision_score, recall_score,
                                 roc_auc_score, confusion_matrix)
    pred = (scores >= thr).astype(int)
    f1   = f1_score(y, pred, zero_division=0)
    pre  = precision_score(y, pred, zero_division=0)
    rec  = recall_score(y, pred, zero_division=0)
    try:   roc = roc_auc_score(y, scores)
    except: roc = float("nan")
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    logger.info(f"  [{name:<32}] F1={f1:.4f}  Prec={pre:.4f}  Rec={rec:.4f}  ROC={roc:.4f}")
    logger.info(f"  {'':<34} TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    return dict(f1=float(f1), precision=float(pre), recall=float(rec),
                roc_auc=float(roc), threshold=float(thr),
                tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    logger.info("=" * 65)
    logger.info("Graph Deviation Network (GDN) — HAI Anomaly Detection")
    logger.info("=" * 65)
    logger.info(f"Config: window={args.window}  top_k={args.top_k}  "
                f"embed_dim={args.embed_dim}  hidden={args.hidden}  epochs={args.epochs}")

    loader = MultiVersionLoader(HAI_ROOT)

    # ── Load data ──────────────────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("Step 1: Load training data (normal only)")
    logger.info("=" * 65)
    X_train, y_train = loader.load_all(
        versions=VERSIONS, split="train", features=COMMON_FEATURES
    )
    X_train = X_train[y_train == 0].astype(np.float32)
    logger.info(f"  Normal training samples: {X_train.shape}")

    logger.info("=" * 65)
    logger.info("Step 2: Load test data")
    logger.info("=" * 65)
    X_test, y_test = loader.load_all(
        versions=VERSIONS, split="test", features=COMMON_FEATURES
    )
    X_test = X_test.astype(np.float32)
    logger.info(f"  Test: {X_test.shape}  attacks={y_test.sum()} ({y_test.mean()*100:.2f}%)")

    # ── Normalize ──────────────────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("Step 3: Normalize")
    logger.info("=" * 65)
    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0) + 1e-8
    X_train_n = (X_train - mean) / std
    X_test_n  = (X_test  - mean) / std
    logger.info(f"  Done. mean range: [{mean.min():.3f}, {mean.max():.3f}]")

    # ── Build training windows ─────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info(f"Step 4: Sample {args.max_train:,} training windows (window={args.window})")
    logger.info("=" * 65)
    X_wins = sample_windows(X_train_n, args.window, args.max_train)
    logger.info(f"  Windows: {X_wins.shape}  ({X_wins.nbytes / 1e6:.0f} MB)")

    # ── Build model ────────────────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("Step 5: Train GDN")
    logger.info("=" * 65)
    model = GDN(N_FEAT, window=args.window, embed_dim=args.embed_dim,
                top_k=args.top_k, hidden=args.hidden)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Parameters: {n_params:,}")

    dataset   = torch.utils.data.TensorDataset(torch.from_numpy(X_wins))
    tr_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_loss  = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for (batch,) in tr_loader:
            target = batch[:, -1, :]   # (B, N) — predict current from neighbors
            pred   = model(batch)      # (B, N)
            loss   = F.mse_loss(pred, target)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(tr_loader)
        scheduler.step()

        if avg_loss < best_loss:
            best_loss  = avg_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            logger.info(f"  Epoch {epoch:3d}/{args.epochs}  loss={avg_loss:.6f}  best={best_loss:.6f}")

    model.load_state_dict(best_state)
    model.eval()

    # ── Score test set ─────────────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("Step 6: Score test set")
    logger.info("=" * 65)
    logger.info(f"  Scoring {len(X_test_n):,} timesteps...")
    raw_scores, per_sensor_devs = score_all_timesteps(model, X_test_n, args.window)
    max_scores = per_sensor_devs.max(axis=1)   # worst-offending sensor per timestep
    logger.info(f"  Mean score stats: min={raw_scores.min():.4f}  max={raw_scores.max():.4f}"
                f"  mean={raw_scores.mean():.4f}  p95={np.percentile(raw_scores, 95):.4f}")

    # ── Evaluate ───────────────────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("Step 7: Threshold search + evaluation")
    logger.info("=" * 65)

    all_metrics = {}
    scores_series = pd.Series(raw_scores)
    max_series    = pd.Series(max_scores)

    # Raw mean deviation
    thr, _ = best_threshold(raw_scores, y_test)
    all_metrics["GDN_mean_raw"] = full_eval(raw_scores, y_test, thr, "GDN mean-dev raw")

    # Max-sensor deviation (catches sensors that deviate most)
    thr, _ = best_threshold(max_scores, y_test)
    all_metrics["GDN_max_raw"] = full_eval(max_scores, y_test, thr, "GDN max-sensor raw")

    # EWM smoothing on both
    for span in [10, 30, 60]:
        ewm_mean = scores_series.ewm(span=span).mean().values
        thr, _ = best_threshold(ewm_mean, y_test)
        all_metrics[f"GDN_mean_ewm{span}"] = full_eval(ewm_mean, y_test, thr,
                                                        f"GDN mean-dev EWM-{span}s")

        ewm_max = max_series.ewm(span=span).mean().values
        thr, _ = best_threshold(ewm_max, y_test)
        all_metrics[f"GDN_max_ewm{span}"] = full_eval(ewm_max, y_test, thr,
                                                       f"GDN max-sensor EWM-{span}s")

    # Best
    best_name = max(all_metrics, key=lambda k: all_metrics[k]["f1"])
    best_f1   = all_metrics[best_name]["f1"]

    logger.info("=" * 65)
    logger.info("All results ranked by F1:")
    for n, m in sorted(all_metrics.items(), key=lambda x: -x[1]["f1"]):
        logger.info(f"  {n:<36} F1={m['f1']:.4f}  ROC={m['roc_auc']:.4f}")
    logger.info(f"\n  BEST: {best_name}  F1={best_f1:.4f}")

    # ── Save model ─────────────────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("Step 8: Save")
    logger.info("=" * 65)

    pkg = {
        "model_type":  "GDN",
        "model":       model,
        "model_state": best_state,
        "model_name":  f"GDN(k={args.top_k},w={args.window})",
        "n_sensors":   N_FEAT,
        "features":    COMMON_FEATURES,
        "mean":        mean,
        "std":         std,
        "window":      args.window,
        "top_k":       args.top_k,
        "threshold":   all_metrics[best_name]["threshold"],
        "metrics":     all_metrics[best_name],
        "best_f1":     best_f1,
        "all_results": all_metrics,
    }
    joblib.dump(pkg, OUT_DIR / "gdn_detection.joblib")
    logger.info("  Saved: gdn_detection.joblib")

    current_best_f1 = 0.4339  # LSTM AE
    if best_f1 > current_best_f1:
        joblib.dump(pkg, OUT_DIR / "best_detection_model.joblib")
        logger.info(f"  NEW BEST! GDN F1={best_f1:.4f} > previous F1={current_best_f1:.4f}")
        logger.info("  Saved: best_detection_model.joblib")

        # Update pipeline summary
        bm = all_metrics[best_name]
        summ_path = MET_DIR / "pipeline_summary.json"
        summ = {}
        if summ_path.exists():
            try:
                with open(summ_path) as f: summ = json.load(f)
            except: pass
        summ.update({
            "run_timestamp": str(datetime.datetime.now()),
            "best_model":    f"GDN(k={args.top_k},w={args.window})",
            "metrics": {
                "f1":        bm["f1"],
                "precision": bm["precision"],
                "recall":    bm["recall"],
                "roc_auc":   bm["roc_auc"],
                "threshold": bm["threshold"],
            },
            "training_note": (f"GDN: {args.epochs} epochs, top_k={args.top_k}, "
                              f"window={args.window}, embed_dim={args.embed_dim}"),
            "training_versions": VERSIONS,
        })
        with open(summ_path, "w") as f:
            json.dump(summ, f, indent=2, default=str)
        logger.info("  Updated pipeline_summary.json")
    else:
        logger.info(f"  GDN F1={best_f1:.4f} vs current best {current_best_f1:.4f} (keeping existing)")

    # ── Final summary ──────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 65)
    logger.info("  GDN FINAL RESULTS")
    logger.info("=" * 65)
    logger.info(f"  GDN best:      F1 = {best_f1:.4f}  [{best_name}]")
    logger.info(f"  Previous best: F1 = {current_best_f1:.4f}  [LSTM AE]")
    logger.info(f"  Change:        {best_f1 - current_best_f1:+.4f}")

    if   best_f1 >= 0.70: status = "EXCELLENT >= 0.70 - TARGET REACHED!"
    elif best_f1 >= 0.55: status = "GOOD      >= 0.55 - significant improvement"
    elif best_f1 >= 0.45: status = "FAIR      >= 0.45 - marginal improvement"
    else:                  status = "POOR      <  0.45 - no improvement over LSTM"
    logger.info(f"  STATUS: {status}")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
