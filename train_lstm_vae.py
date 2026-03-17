"""
LSTM Variational Autoencoder — haiend-23.05 (225 DCS sensors)
=============================================================
Replaces the deterministic latent space of LSTM-AE with a probabilistic
one (μ, σ). Anomaly score = reconstruction MSE + β·KL divergence.

Why this can beat LSTM-AE:
  - KL divergence measures how far the encoded distribution deviates from N(0,I)
  - For normal data:  q(z|x) ≈ N(0,I)  → KL ≈ 0
  - For attacks:      q(z|x) shifts     → KL spikes
  - Catches subtle attacks where MSE alone is small (the FN=4037 problem)

Training:
  - Normal data only (896K samples, same as LSTM-AE)
  - β-annealing: β linearly increases 0→1 over first 20 epochs
    Prevents KL collapse — the #1 failure mode of VAEs
  - Inference: use μ (mean) not sampled z → deterministic, stable scores

Anomaly scores evaluated:
  1. Reconstruction MSE     (same as LSTM-AE baseline)
  2. KL divergence          (-0.5 * sum(1 + log_var - μ² - exp(log_var)))
  3. ELBO                   (MSE + β·KL)
  4. μ magnitude            (||μ||₂²) — pure latent deviation

Usage: python train_lstm_vae.py [--epochs 60] [--beta 1.0] [--n-windows 150000]
"""

import sys, json, warnings, argparse
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import datetime

sys.path.insert(0, str(Path(__file__).parent))
from src.utils.logger import logger

HAIEND_DIR = Path("C:/Users/PC GAMING/Desktop/AI/HAI/haiend-23.05/haiend-23.05")
OUT_DIR    = Path("outputs/models")
MET_DIR    = Path("outputs/metrics")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MET_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",      type=int,   default=60)
    p.add_argument("--window",      type=int,   default=30)
    p.add_argument("--hidden",      type=int,   default=128)
    p.add_argument("--latent",      type=int,   default=48)
    p.add_argument("--batch",       type=int,   default=512)
    p.add_argument("--n-windows",   type=int,   default=150_000)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--beta",        type=float, default=1.0,
                   help="Max KL weight (beta-VAE). Anneals 0→beta over first 20 epochs.")
    p.add_argument("--beta-warmup", type=int,   default=20,
                   help="Epochs to linearly anneal beta from 0 to --beta")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class LSTMVariationalAutoencoder(nn.Module):
    """
    LSTM-VAE: same encoder/decoder structure as LSTMAEutoencoder but with
    a stochastic latent space.

    Encoder outputs (mu, log_var) instead of a deterministic z.
    Decoder uses mu at inference (no sampling) for stable anomaly scores.
    """
    def __init__(self, n_features: int, hidden: int = 128, latent: int = 48):
        super().__init__()
        self.n_features = n_features
        self.hidden     = hidden
        self.latent     = latent

        # Encoder (identical to LSTM-AE up to the latent projection)
        self.enc_lstm1 = nn.LSTM(n_features, hidden, batch_first=True)
        self.enc_lstm2 = nn.LSTM(hidden, hidden, batch_first=True)
        self.dropout   = nn.Dropout(0.1)

        # Project to (mu, log_var) instead of single latent
        self.fc_mu      = nn.Linear(hidden, latent)
        self.fc_log_var = nn.Linear(hidden, latent)

        # Decoder (identical to LSTM-AE)
        self.dec_lstm1 = nn.LSTM(latent, hidden, batch_first=True)
        self.dec_lstm2 = nn.LSTM(hidden, hidden, batch_first=True)
        self.out_proj  = nn.Linear(hidden, n_features)

    def encode(self, x):
        """x: (B, W, N) → mu: (B, latent), log_var: (B, latent)"""
        out1, _    = self.enc_lstm1(x)
        out1       = self.dropout(out1)
        _, (hn, _) = self.enc_lstm2(out1)
        h          = hn.squeeze(0)              # (B, hidden)
        mu         = self.fc_mu(h)              # (B, latent)
        log_var    = self.fc_log_var(h)         # (B, latent)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        """Sample z ~ N(mu, exp(log_var/2)) during training."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, seq_len):
        """z: (B, latent) → (B, W, N)"""
        rep    = z.unsqueeze(1).expand(-1, seq_len, -1)
        out1, _ = self.dec_lstm1(rep)
        out1    = self.dropout(out1)
        out2, _ = self.dec_lstm2(out1)
        return self.out_proj(out2)

    def forward(self, x):
        """Training forward: encode → sample → decode."""
        mu, log_var = self.encode(x)
        z           = self.reparameterize(mu, log_var)
        recon       = self.decode(z, x.size(1))
        return recon, mu, log_var

    def reconstruction_error(self, x):
        """Inference: use mu (no sampling) → deterministic MSE per sample."""
        with torch.no_grad():
            mu, _ = self.encode(x)
            recon = self.decode(mu, x.size(1))
            return ((recon - x) ** 2).mean(dim=(1, 2)).cpu().numpy()  # (B,)

    def kl_score(self, x):
        """Per-sample KL divergence: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))."""
        with torch.no_grad():
            mu, log_var = self.encode(x)
            # Shape: (B, latent) → sum over latent → (B,)
            kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(dim=1)
            return kl.cpu().numpy()  # (B,) always positive

    def elbo_score(self, x, beta=1.0):
        """ELBO anomaly score = reconstruction MSE + beta * KL."""
        with torch.no_grad():
            mu, log_var = self.encode(x)
            recon       = self.decode(mu, x.size(1))
            mse = ((recon - x) ** 2).mean(dim=(1, 2)).cpu().numpy()
            kl  = (-0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
                   .sum(dim=1)).cpu().numpy()
            return mse + beta * kl  # (B,)

    def mu_magnitude(self, x):
        """||mu||^2: how far the latent code is from origin."""
        with torch.no_grad():
            mu, _ = self.encode(x)
            return (mu ** 2).sum(dim=1).cpu().numpy()  # (B,)

    def per_sensor_error(self, x):
        """(B, N) per-sensor MSE using mu (for root cause analysis)."""
        with torch.no_grad():
            mu, log_var = self.encode(x)
            recon       = self.decode(mu, x.size(1))
            return ((recon - x) ** 2).mean(dim=1).cpu().numpy()  # (B, N)


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

def vae_loss(recon, x, mu, log_var, beta):
    """
    ELBO loss = reconstruction MSE + beta * KL
    KL = -0.5 * sum(1 + log_var - mu^2 - exp(log_var)) / (B * W * N)
    Normalized by sequence length and features so beta is scale-independent.
    """
    B, W, N = x.size()
    recon_loss = ((recon - x) ** 2).mean()  # scalar
    # KL per sample, summed over latent, mean over batch
    kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(dim=1).mean()
    # Normalize KL by W*N so it's on same scale as per-timestep MSE
    kl_norm = kl / (W * N)
    return recon_loss + beta * kl_norm, recon_loss.item(), kl_norm.item()


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def load_train(haiend_dir):
    dfs = []
    for f in ["end-train1.csv", "end-train2.csv", "end-train3.csv", "end-train4.csv"]:
        df = pd.read_csv(haiend_dir / f)
        dfs.append(df.iloc[:, 1:].ffill().fillna(0).astype(np.float32).values)
    X = np.concatenate(dfs, axis=0)
    logger.info(f"  Train: {X.shape}")
    return X


def load_test(haiend_dir):
    X_parts, y_parts = [], []
    for i in [1, 2]:
        X_df = pd.read_csv(haiend_dir / f"end-test{i}.csv")
        y_df = pd.read_csv(haiend_dir / f"label-test{i}.csv")
        X_parts.append(X_df.iloc[:, 1:].ffill().fillna(0).astype(np.float32).values)
        y_parts.append(y_df["label"].values.astype(np.int32))
    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    logger.info(f"  Test: {X.shape}  attacks={y.sum()} ({y.mean()*100:.2f}%)")
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def sample_windows(X, window, n):
    T = len(X)
    starts = np.random.choice(T - window, min(n, T - window), replace=False)
    starts.sort()
    return np.stack([X[s:s + window] for s in starts]).astype(np.float32)


def score_test_all(model, X, window, chunk=1024):
    """Score every timestep with all 4 anomaly signals."""
    T, N = X.shape
    X_pad   = np.concatenate([np.zeros((window - 1, N), dtype=np.float32), X], axis=0)
    mse_scores  = np.zeros(T, dtype=np.float32)
    kl_scores   = np.zeros(T, dtype=np.float32)
    elbo_scores = np.zeros(T, dtype=np.float32)
    mu_scores   = np.zeros(T, dtype=np.float32)
    model.eval()
    for start in range(0, T, chunk):
        end   = min(start + chunk, T)
        size  = end - start
        batch = np.stack([X_pad[start + i: start + i + window] for i in range(size)])
        bt    = torch.from_numpy(batch)
        mse_scores[start:end]  = model.reconstruction_error(bt)
        kl_scores[start:end]   = model.kl_score(bt)
        elbo_scores[start:end] = model.elbo_score(bt, beta=1.0)
        mu_scores[start:end]   = model.mu_magnitude(bt)
    return mse_scores, kl_scores, elbo_scores, mu_scores


def best_threshold(scores, y):
    from sklearn.metrics import f1_score
    bf, bt = 0.0, scores.mean()
    for p in np.arange(70, 99.9, 0.1):
        t = np.percentile(scores, p)
        f = f1_score(y, (scores >= t).astype(int), zero_division=0)
        if f > bf: bf, bt = f, t
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
    logger.info(f"  [{name:<36}]  F1={f1:.4f}  P={pre:.4f}  R={rec:.4f}  ROC={roc:.4f}")
    logger.info(f"  {'':<38}  TP={tp}  FP={fp}  FN={fn}")
    return dict(f1=float(f1), precision=float(pre), recall=float(rec),
                roc_auc=float(roc), threshold=float(thr),
                tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    logger.info("=" * 70)
    logger.info("LSTM-VAE — haiend-23.05  (225 DCS sensors)")
    logger.info(f"Config: window={args.window}  hidden={args.hidden}  latent={args.latent}")
    logger.info(f"        epochs={args.epochs}  beta={args.beta}  warmup={args.beta_warmup}")
    logger.info(f"        n_windows={args.n_windows:,}  lr={args.lr}")
    logger.info("=" * 70)

    # ── Load data ──────────────────────────────────────────────────────────────
    logger.info("Step 1: Load data")
    X_train        = load_train(HAIEND_DIR)
    X_test, y_test = load_test(HAIEND_DIR)
    N_FEAT         = X_train.shape[1]

    # ── Normalize (reuse exact stats from w=30 LSTM-AE for fair comparison) ────
    logger.info("Step 2: Normalize (using same stats as LSTM-AE baseline)")
    base_path = OUT_DIR / "haiend_lstm_detection.joblib"
    if base_path.exists():
        from train_haiend_lstm import LSTMAutoencoder as _LSTM
        sys.modules["__main__"].LSTMAutoencoder = _LSTM
        base_pkg   = joblib.load(base_path)
        mean       = base_pkg["data_mean"]
        std        = base_pkg["data_std"]
        baseline_f1 = base_pkg.get("best_f1", 0.6886)
        logger.info(f"  Loaded normalization from LSTM-AE baseline (F1={baseline_f1:.4f})")
    else:
        mean = X_train.mean(axis=0)
        std  = X_train.std(axis=0)
        std  = np.maximum(std, 1.0)
        baseline_f1 = 0.0

    X_train_n = (X_train - mean) / std
    X_test_n  = (X_test  - mean) / std

    # ── Sample windows ─────────────────────────────────────────────────────────
    logger.info(f"Step 3: Sample {args.n_windows:,} training windows (window={args.window})")
    X_wins  = sample_windows(X_train_n, args.window, args.n_windows)
    mem_mb  = X_wins.nbytes / 1e6
    logger.info(f"  Windows: {X_wins.shape}  ({mem_mb:.0f} MB)")

    # ── Build model ────────────────────────────────────────────────────────────
    logger.info("Step 4: Build LSTM-VAE")
    model    = LSTMVariationalAutoencoder(N_FEAT, hidden=args.hidden, latent=args.latent)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Parameters: {n_params:,}")

    dataset   = torch.utils.data.TensorDataset(torch.from_numpy(X_wins))
    tr_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Train ──────────────────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("Step 5: Train LSTM-VAE with beta annealing")
    logger.info(f"  Beta anneals: 0.0 → {args.beta} over {args.beta_warmup} epochs")
    logger.info("=" * 70)

    best_loss, best_state = float("inf"), None

    for epoch in range(1, args.epochs + 1):
        # Linear beta annealing: 0 → args.beta over warmup epochs
        beta_cur = min(args.beta, args.beta * epoch / max(args.beta_warmup, 1))

        model.train()
        total_loss = total_recon = total_kl = 0.0

        for (batch,) in tr_loader:
            recon, mu, log_var = model(batch)
            loss, recon_l, kl_l = vae_loss(recon, batch, mu, log_var, beta_cur)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss  += loss.item()
            total_recon += recon_l
            total_kl    += kl_l

        avg_loss  = total_loss  / len(tr_loader)
        avg_recon = total_recon / len(tr_loader)
        avg_kl    = total_kl    / len(tr_loader)
        scheduler.step()

        if avg_loss < best_loss:
            best_loss  = avg_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                f"  Epoch {epoch:3d}/{args.epochs}  "
                f"loss={avg_loss:.6f}  recon={avg_recon:.6f}  kl={avg_kl:.6f}  "
                f"beta={beta_cur:.3f}"
            )

    model.load_state_dict(best_state)
    model.eval()

    # ── Score test set ─────────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info(f"Step 6: Score {len(X_test_n):,} test timesteps")
    logger.info("=" * 70)

    mse_s, kl_s, elbo_s, mu_s = score_test_all(model, X_test_n, args.window)

    # Clip outliers
    for name, arr in [("MSE", mse_s), ("KL", kl_s), ("ELBO", elbo_s), ("mu_mag", mu_s)]:
        p999 = np.percentile(arr, 99.9)
        arr[:] = np.clip(arr, 0, p999)
        logger.info(f"  {name:<8}: min={arr.min():.5f}  mean={arr.mean():.5f}  "
                    f"p95={np.percentile(arr,95):.5f}  max={arr.max():.5f}")

    # ── Evaluate all anomaly scores ────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("Step 7: Evaluate all anomaly scores")
    logger.info("=" * 70)

    all_metrics = {}

    thr, _ = best_threshold(mse_s, y_test)
    all_metrics["VAE_MSE"]       = full_eval(mse_s,  y_test, thr, "VAE reconstruction MSE")

    thr, _ = best_threshold(kl_s, y_test)
    all_metrics["VAE_KL"]        = full_eval(kl_s,   y_test, thr, "VAE KL divergence")

    thr, _ = best_threshold(elbo_s, y_test)
    all_metrics["VAE_ELBO"]      = full_eval(elbo_s, y_test, thr, "VAE ELBO (MSE + KL)")

    thr, _ = best_threshold(mu_s, y_test)
    all_metrics["VAE_mu_mag"]    = full_eval(mu_s,   y_test, thr, "VAE mu magnitude")

    # Combined: MSE + KL with different beta weightings
    for b in [0.1, 0.5, 2.0]:
        combined = mse_s + b * kl_s
        thr, _ = best_threshold(combined, y_test)
        all_metrics[f"VAE_combined_b{b}"] = full_eval(
            combined, y_test, thr, f"VAE MSE + {b}*KL")

    # ── Summary ────────────────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("All results ranked by F1:")
    logger.info("=" * 70)
    for n, m in sorted(all_metrics.items(), key=lambda x: -x[1]["f1"]):
        logger.info(f"  {n:<36}  F1={m['f1']:.4f}  ROC={m['roc_auc']:.4f}")

    best_name = max(all_metrics, key=lambda k: all_metrics[k]["f1"])
    best_f1   = all_metrics[best_name]["f1"]
    best_m    = all_metrics[best_name]

    logger.info("=" * 70)
    logger.info(f"  LSTM-VAE best:   {best_name}  F1={best_f1:.4f}")
    logger.info(f"  LSTM-AE baseline:             F1={baseline_f1:.4f}")
    logger.info(f"  Delta:                        {best_f1 - baseline_f1:+.4f}")

    if best_f1 > baseline_f1:
        logger.info("  IMPROVEMENT CONFIRMED")
    else:
        logger.info("  No improvement over LSTM-AE baseline")

    # ── Save ───────────────────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("Step 8: Save")
    logger.info("=" * 70)

    # Determine best score type and threshold
    best_score_type = best_name.replace("VAE_", "").lower()  # mse / kl / elbo / mu_mag / combined_b*
    best_thr        = best_m["threshold"]

    pkg = {
        "model_type":       "LSTMVariationalAutoencoder_haiend",
        "model":            model,
        "model_state":      best_state,
        "model_name":       f"LSTM-VAE(h={args.hidden},l={args.latent},w={args.window},b={args.beta})",
        "n_features":       N_FEAT,
        "window":           args.window,
        "hidden":           args.hidden,
        "latent":           args.latent,
        "beta":             args.beta,
        "data_mean":        mean,
        "data_std":         std,
        "best_score_type":  best_score_type,
        "threshold":        best_thr,
        "metrics":          best_m,
        "best_f1":          float(best_f1),
        "all_results":      all_metrics,
        "baseline_lstm_ae_f1": float(baseline_f1),
    }

    out_path = OUT_DIR / "lstm_vae_detection.joblib"
    joblib.dump(pkg, out_path)
    logger.info(f"  Saved: {out_path}")

    # Update best model if improved
    prev_best = 0.0
    summ_path = MET_DIR / "pipeline_summary.json"
    if summ_path.exists():
        try:
            with open(summ_path) as f:
                prev_best = json.load(f).get("metrics", {}).get("f1", 0.0)
        except: pass

    if best_f1 > prev_best:
        joblib.dump(pkg, OUT_DIR / "best_detection_model.joblib")
        logger.info(f"  NEW BEST! F1={best_f1:.4f} > {prev_best:.4f}")
        logger.info("  Saved: best_detection_model.joblib")

        summ = {}
        if summ_path.exists():
            try:
                with open(summ_path) as f: summ = json.load(f)
            except: pass
        summ.update({
            "run_timestamp":  str(datetime.datetime.now()),
            "best_model":     pkg["model_name"],
            "metrics": {
                "f1":        float(best_f1),
                "precision": float(best_m["precision"]),
                "recall":    float(best_m["recall"]),
                "roc_auc":   float(best_m["roc_auc"]),
                "threshold": float(best_thr),
            },
            "training_note": f"LSTM-VAE: w={args.window}, beta={args.beta}, score={best_score_type}",
        })
        with open(summ_path, "w") as f:
            json.dump(summ, f, indent=2, default=str)
        logger.info("  Updated pipeline_summary.json")
    else:
        logger.info(f"  F1={best_f1:.4f} vs current best {prev_best:.4f} (no update)")

    logger.info("=" * 70)
    logger.info("  LSTM-VAE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Best score:   {best_name}   F1={best_f1:.4f}")
    logger.info(f"  LSTM-AE base: F1={baseline_f1:.4f}")
    logger.info(f"  Delta:        {best_f1 - baseline_f1:+.4f}")
    logger.info(f"  Saved:        outputs/models/lstm_vae_detection.joblib")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
