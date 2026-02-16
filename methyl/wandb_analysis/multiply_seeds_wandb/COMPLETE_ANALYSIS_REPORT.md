# Complete Multi-Seed Analysis Report

## Overview

- **Project:** netanelazran11-hebrew-university-of-jerusalem/finetune-bmfm-multiseed
- **Number of Seeds:** 5 (seeds 40, 41, 42, 43, 44)
- **Model:** BMFM-RNA adapted for methylation
- **Task:** Age prediction from DNA methylation

---

## Summary Statistics (n=5 seeds)

| Metric | Mean ± Std | Min | Max |
|--------|------------|-----|-----|
| **Test MAE (years)** | **4.79 ± 0.14** | 4.64 | 4.97 |
| **Test R²** | **0.9273 ± 0.0043** | 0.9208 | 0.9316 |
| **Val MAE Best** | 4.95 ± 0.06 | 4.85 | 4.99 |
| **Train MAE Final** | 2.34 ± 0.14 | 2.18 | 2.55 |
| **Best Epoch** | 132 ± 49 | 85 | 214 |

---

## Individual Seed Results

| Seed | Test MAE (years) | Test R² | Val MAE Best | Best Epoch | Status |
|------|------------------|---------|--------------|------------|--------|
| 40 | 4.8806 | 0.9260 | 4.9590 | 116 | finished  |
| 41 | 4.9746 | 0.9208 | 4.9874 | 85 | finished  |
| 42 | 4.7634 | 0.9277 | 4.9720 | 214 | finished  |
| 43 | 4.6694 | 0.9306 | 4.9796 | 118 | finished  |
| 44 | 4.6406 | 0.9316 | 4.8528 | 125 | finished **Best** |

---

## Best Model

- **Seed:** 44
- **Test MAE:** 4.6406 years
- **Test R²:** 0.9316
- **Best Epoch:** 125
- **Run ID:** llznkqvj

---

## Key Findings

1. **Excellent Consistency:** Low variance across seeds (MAE std = 0.14 years)
2. **Strong Performance:** Mean R² = 0.9273 (explains 92.7% of age variance)
3. **Best MAE:** 4.64 years (Seed 44)
4. **Convergence:** Models converge around epoch 132 on average

---

## Comparison with Baseline

| Model | Test MAE (years) | Test R² |
|-------|------------------|---------|
| Mean Prediction | 22.82 | 0.00 |
| MethylGPT | 4.95 | 0.911 |
| **BMFM-RNA (Mean ± Std)** | **4.79 ± 0.14** | **0.9273 ± 0.0043** |
| **BMFM-RNA (Best)** | **4.64** | **0.9316** |

**Improvement over MethylGPT:**
- MAE: 3.3% better (mean), 6.3% better (best)
- R²: 1.8% better (mean), 2.3% better (best)

---

## Output Files

- `runs_table.csv` - All runs with metrics
- `summary_statistics.csv` - Statistical summary
- `test_metrics_comparison.png` - Bar charts comparing seeds
- `mae_convergence_all_seeds.png` - MAE training curves
- `r2_convergence_all_seeds.png` - R² training curves
- `metrics_distribution.png` - Box plots
