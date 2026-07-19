#!/usr/bin/env python3
"""
Compare MethylGPT 19k vs MethylGPT 21k runs.

19k run: methylGPT_medium_19k / yny3obvg   (cached)
21k run: methylGPT_medium_21k_altumage / xzrw1qwr

Goal: identify which run is stronger, decide which dataset to use for k-fold CV.
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import wandb

warnings.filterwarnings("ignore")

ENTITY    = "netanelazran11-hebrew-university-of-jerusalem"
CACHE_DIR = Path("wandb_run_comparison_methylgpt_v5_v7b")   # 19k already cached here
OUT_DIR   = Path("wandb_methylgpt_19k_vs_21k")
OUT_DIR.mkdir(exist_ok=True)

STREAM_CHUNK = 10_000

RUNS = {
    "gpt_19k": ("methylGPT_medium_19k",          "yny3obvg"),
    "gpt_21k": ("methylGPT_medium_21k_altumage",  "xzrw1qwr"),
}

METRIC_CANDIDATES = {
    "val_mae":   ["valid_mae", "val/mae", "validation/mae", "val_mae",
                  "valid_mae_loss/dataloader_idx_0"],
    "val_medae": ["valid_medae", "val/medae", "validation/medae", "val_medae"],
    "val_r2":    ["valid_r2",   "val/r2",   "validation/r2",   "val_r2"],
    "val_loss":  ["valid_mse_loss/dataloader_idx_0", "val/loss", "validation/loss",
                  "valid_loss_norm/dataloader_idx_0"],
    "epoch":     ["epoch"],
}


# ── Download / load from cache ────────────────────────────────────────────────
def load_run(label: str, project: str, run_id: str) -> pd.DataFrame:
    # Check 19k cache first
    cached_19k = CACHE_DIR / "raw_history_methylgpt.csv"
    if label == "gpt_19k" and cached_19k.exists():
        print(f"[{label}] Loading from cache: {cached_19k}")
        return pd.read_csv(cached_19k)

    # Check own cache
    cache = OUT_DIR / f"raw_history_{label}.csv"
    if cache.exists():
        print(f"[{label}] Loading from cache: {cache}")
        return pd.read_csv(cache)

    print(f"[{label}] Downloading from WandB ...")
    api = wandb.Api(timeout=120)
    run = api.run(f"{ENTITY}/{project}/{run_id}")
    print(f"  run: {run.name}  steps: {run.lastHistoryStep:,}")

    header_written = False
    chunk = []
    n = 0
    for row in run.scan_history():
        chunk.append(row)
        if len(chunk) >= STREAM_CHUNK:
            pd.DataFrame(chunk).to_csv(cache, mode="a", header=not header_written, index=False)
            header_written = True
            n += len(chunk)
            chunk = []
            print(f"  ... {n:,} rows", end="\r")
    if chunk:
        pd.DataFrame(chunk).to_csv(cache, mode="a", header=not header_written, index=False)
        n += len(chunk)
    print(f"  [{label}] {n:,} rows downloaded")
    return pd.read_csv(cache)


# ── Find first matching metric column ─────────────────────────────────────────
def find_col(df: pd.DataFrame, candidates: list) -> str | None:
    for c in candidates:
        if c in df.columns and df[c].notna().sum() > 0:
            return c
    return None


# ── Build epoch-level summary ─────────────────────────────────────────────────
def epoch_summary(df: pd.DataFrame, label: str) -> pd.DataFrame:
    epoch_col = find_col(df, METRIC_CANDIDATES["epoch"])
    if epoch_col is None:
        # estimate epoch from step if no epoch column
        max_steps = df["_step"].max() if "_step" in df.columns else len(df)
        df = df.copy()
        df["epoch"] = (df.get("_step", range(len(df))) / max_steps * 300).astype(int)
        epoch_col = "epoch"

    cols = {"epoch": df[epoch_col]}
    for key, candidates in METRIC_CANDIDATES.items():
        if key == "epoch":
            continue
        c = find_col(df, candidates)
        if c:
            cols[key] = df[c]
        else:
            print(f"  [{label}] metric '{key}' not found — tried: {candidates}")

    out = pd.DataFrame(cols)
    out = out.dropna(subset=["epoch"])
    out["epoch"] = out["epoch"].astype(int)
    # aggregate per epoch (take last logged value per epoch)
    out = out.groupby("epoch").last().reset_index()
    return out


# ── Main ──────────────────────────────────────────────────────────────────────
dfs_raw  = {}
dfs_epoch = {}
for label, (project, run_id) in RUNS.items():
    raw = load_run(label, project, run_id)
    print(f"  [{label}] columns: {[c for c in raw.columns if not c.startswith('_')][:15]}")
    ep = epoch_summary(raw, label)
    dfs_raw[label]   = raw
    dfs_epoch[label] = ep
    print(f"  [{label}] {len(ep)} epochs, metrics: {[c for c in ep.columns if c != 'epoch']}")

print("\n" + "=" * 65)
print("METHYLGPT 19k vs 21k — BEST CHECKPOINT COMPARISON")
print("=" * 65)

results = {}
for label, ep in dfs_epoch.items():
    r = {"label": label, "n_epochs": len(ep)}

    for metric, col in [("val_medae", "val_medae"), ("val_r2", "val_r2"), ("val_mae", "val_mae")]:
        if col not in ep.columns:
            continue
        series = ep[col].dropna()
        if metric in ("val_medae", "val_mae"):
            best_val = series.min()
            best_ep  = ep.loc[series.idxmin(), "epoch"]
        else:
            best_val = series.max()
            best_ep  = ep.loc[series.idxmax(), "epoch"]
        r[f"best_{metric}"]    = best_val
        r[f"best_{metric}_ep"] = best_ep

    # Convergence: epoch when val_r2 first exceeds 0.90
    if "val_r2" in ep.columns:
        above = ep[ep["val_r2"] > 0.90]
        r["epoch_r2_090"] = int(above["epoch"].min()) if len(above) else None

    # Stability: last 20 epochs mean val_medae
    if "val_medae" in ep.columns:
        last20 = ep.tail(20)["val_medae"].dropna()
        r["last20_medae_mean"] = last20.mean()
        r["last20_medae_std"]  = last20.std()

    results[label] = r
    print(f"\n{label.upper()}:")
    print(f"  Epochs logged       : {r['n_epochs']}")
    if "best_val_medae" in r:
        print(f"  Best val MedAE      : {r['best_val_medae']:.3f} yr  (ep {r['best_val_medae_ep']})")
    if "best_val_r2" in r:
        print(f"  Best val R²         : {r['best_val_r2']:.4f}      (ep {r['best_val_r2_ep']})")
    if "best_val_mae" in r:
        print(f"  Best val MAE        : {r['best_val_mae']:.3f} yr  (ep {r['best_val_mae_ep']})")
    if r.get("epoch_r2_090"):
        print(f"  Epoch to R²>0.90    : {r['epoch_r2_090']}")
    if "last20_medae_mean" in r:
        print(f"  Last-20ep MedAE     : {r['last20_medae_mean']:.3f} ± {r['last20_medae_std']:.3f} yr")

print("\n" + "=" * 65)
print("VERDICT")
print("=" * 65)

r19 = results.get("gpt_19k", {})
r21 = results.get("gpt_21k", {})

if "best_val_medae" in r19 and "best_val_medae" in r21:
    m19 = r19["best_val_medae"]
    m21 = r21["best_val_medae"]
    winner = "19k" if m19 < m21 else "21k"
    diff   = abs(m19 - m21)
    print(f"\n  MethylGPT 19k best val MedAE: {m19:.3f} yr")
    print(f"  MethylGPT 21k best val MedAE: {m21:.3f} yr")
    print(f"  Winner (lower MedAE): {winner}  (Δ = {diff:.3f} yr)")
    print(f"\n  → For conservative comparison: use MethylGPT {winner} as baseline")
    print(f"    (conservative = harder baseline = more credible result for V7b)")
    print(f"\n  → Run V7b k-fold on {'19k' if winner == '19k' else '21k'} h5ad")
    print(f"    so V7b and MethylGPT are trained on EXACTLY the same samples")

# Save summary
pd.DataFrame(results).T.to_csv(OUT_DIR / "methylgpt_19k_vs_21k_summary.csv")
print(f"\nSummary saved: {OUT_DIR}/methylgpt_19k_vs_21k_summary.csv")
