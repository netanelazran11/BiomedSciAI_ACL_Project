# MethylGPT 21k — Fair Baseline for MethylLlama V7b

Companion notes to `MethylLlama_vs_MethylGPT_21k.html`. Records the data/split
verification and how the MethylGPT 21k baseline number was chosen.

## 1. Same dataset & split (verified)

Both models use the same underlying **21k AltumAge** data.

| | MethylGPT 21k | MethylLlama V7b |
|--|--|--|
| Source | `MethylGPT/data/21k_altumage/finetuning_data/{train,valid,test}.parquet` | `data/data_methyl_21k_h5ad/altumage_21k_3way.h5ad` |
| Config | `train_methylgpt_21k_altumage.yml` | `finetune_llama_small_v7b_kfold.sh` |
| Index | integer row index | `GSM_ID` (10,988 total) |
| Split counts | 7416 / 1308 / 2264 | 7416 / 1308 / 2264 |

Per-split age distributions match **exactly** (sizes + `np.allclose`):

| Split | Size | Age range | Mean |
|-------|------|-----------|------|
| Train | 7,416 | [-0.7, 144.0] | 38.60 |
| Valid | 1,308 | [-0.6, 114.0] | 39.22 |
| Test  | 2,264 | [-0.7, 103.0] | 43.48 |

Verified by `scripts/utils/compare_splits_21k.py` and
`scripts/utils/verify_21k_comparison.py`.

**Note on row-level match:** the exact GSM-ID match reports a low count only
because MethylGPT stores plain integer indices that don't preserve h5ad row
order (re-indexing artifact). The order-independent distribution check is
authoritative → **SAME DATASET & SPLIT ✓**.

**Filter caveat:** MethylGPT evaluates on the full raw **2,264** test rows.
V7b's pipeline additionally drops **115** (5.1%) = 99 age outliers (age<0 or
>120) + 16 duplicates → **2,149**. Same split, minor filtering difference.

## 2. MethylGPT 21k result — which number is the baseline

WandB run `xzrw1qwr`, 300 epochs, test logged every epoch.

| Selection view | Epoch | Test R² | Test MedAE | Test MAE | Saved? |
|----------------|-------|---------|-----------|----------|--------|
| **Best valid_medae (fair)** | **253** | **0.9044** | **3.839** | **5.521** | ✓ on disk |
| Best valid_r2 | 151 | 0.9078 | 3.430 | 5.313 | ✗ |
| Final epoch (WandB headline) | 299 | 0.9083 | 3.731 | 5.405 | ✗ not saved |
| Oracle best test (peeking) | 233/151 | 0.9104 | 3.430 | 5.269 | — |

Saved checkpoints on disk (`checkpoints_21k_altumage/`), selected by
`valid_medae`:
- **epoch 253** — valid_medae=3.021, test_medae=3.839, test_mae=5.521, test_s_r(Spearman)=0.9100
- epoch 26 — valid_medae=3.584, test_medae=4.260, test_mae=5.807

Convergence: valid R² > 0.90 by epoch 22, then noisy plateau (test R²
0.887–0.910). Last-20-epoch stability: test R² 0.9021 ± 0.0057, test MedAE
3.885 ± 0.238. The 0.9083 final-epoch headline sits at the high end of the noise
band and has **no saved checkpoint**.

## 3. The fair baseline

V7b's k-fold selects the best-`val_medae` checkpoint per fold, so MethylGPT is
represented by its best-`val_medae` checkpoint (epoch 253) for symmetry:

> **MethylGPT 21k baseline: R² = 0.904 · MedAE = 3.84 yr · MAE = 5.52 yr · Spearman = 0.910**

This is the reproducible target for V7b to match/beat with 5-fold confidence
intervals. Do **not** use the 3.731 headline (lucky un-saved final epoch).

Trajectory data: `methylgpt_21k_trajectory.csv` (300 epochs).
