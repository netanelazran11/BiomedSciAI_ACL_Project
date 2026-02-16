# Methylation-Based Age Prediction: BMFM-RNA vs MethylGPT

## Project Overview

This project explores **DNA methylation-based biological age prediction** using Transformer foundation models. DNA methylation patterns at CpG sites change systematically with age, making them powerful biomarkers. We adapt IBM's **BMFM-RNA** architecture (originally for single-cell RNA-seq) to methylation data and compare it against the **MethylGPT** baseline.

**Key idea:** Use multi-field tokenization to separately encode CpG site identity and methylation values, then pretrain with masked language modeling (MLM) before fine-tuning for age regression.

**Paper Reference:** [MethylGPT: a foundation model for the DNA methylome (bioRxiv 2024)](https://www.biorxiv.org/content/10.1101/2024.10.30.621013v2)

---

## Latest Results (Multi-Seed Analysis)

**Dataset:** 8,000 CpG sites from Altumage dataset
**Samples:** 11,500 (Train: 5,483 | Valid: 1,371 | Test: 4,646)

### Multi-Seed Performance (n=5 seeds: 40, 41, 42, 43, 44)

| Metric | Mean ± Std | Min | Max |
|--------|-----------|-----|-----|
| **MAE (years)** | **4.79 ± 0.14** | 4.64 | 4.97 |
| **R²** | **0.927 ± 0.004** | 0.921 | 0.932 |

### Individual Seed Results

| Seed | Test MAE (years) | Test R² | Best Epoch |
|------|------------------|---------|------------|
| 40 | 4.88 | 0.926 | 116 |
| 41 | 4.97 | 0.921 | 85 |
| 42 | 4.76 | 0.928 | 214 |
| 43 | 4.67 | 0.931 | 118 |
| **44** | **4.64** | **0.932** | 125 |

### Best Individual Model
- **Seed 44:** MAE = 4.64 years, R² = 0.932
- **Checkpoint:** Epoch 125 (best validation MAE: 4.85)

---

## Key Files

| File | Location | Description |
|------|----------|-------------|
| `tokenizer.py` | `bmfm_methylation/tokenizer.py` | Multi-field tokenizer that builds vocabulary from CpG site names and encodes samples as (cpg_ids, beta_values) pairs |
| `model.py` | `bmfm_methylation/model.py` | SCBert model architecture with CpG embeddings, continuous value encoder, and Transformer layers |
| `pretrain.py` | `bmfm_methylation/pretrain.py` | MLM/WCED pretraining script - trains encoder to reconstruct β-values |
| `finetune.py` | `bmfm_methylation/finetune.py` | Age regression fine-tuning with freeze/unfreeze strategy |
| `data_module.py` | `bmfm_methylation/data_module.py` | PyTorch Lightning DataModule for loading h5ad methylation data |
| `dataset.py` | `bmfm_methylation/dataset.py` | Dataset class that tokenizes samples and applies masking |
| `config.py` | `bmfm_methylation/config.py` | Dataclass configurations for model, training, and pretraining modes |
| `wced_module.py` | `bmfm_methylation/wced_module.py` | WCED (Whole Cell Expression Decoder) pretraining module |
| `decoders/` | `bmfm_methylation/decoders/` | WCED decoder implementations |

### Scripts

| Script | Location | Description |
|--------|----------|-------------|
| `finetune_multiseed.sh` | `scripts/finetune_multiseed.sh` | SLURM script for fine-tuning with a specific seed |
| `launch_multiseed.sh` | `scripts/launch_multiseed.sh` | Launches 5 parallel jobs with seeds 40-44 |
| `baseline_ridge.py` | `scripts/baseline_ridge.py` | Ridge regression baseline for comparison |

---

## 1. Introduction

DNA methylation provides strong signals for age prediction, but the data are high-dimensional and noisy. We use Transformer-based foundation models to learn sample-level representations from CpG tokens. This project:

1. Establishes a stable **8K-CpG baseline** using MethylGPT
2. Adapts the **BMFM-RNA** (IBM's SCBert) architecture for methylation data
3. Supports two pretraining modes: **MLM** and **WCED**
4. Compares both approaches on the same dataset and evaluation protocol

---

## 2. Pretraining Modes

### MLM (Masked Language Modeling)
- Mask 30% of β-values
- Predict only the masked positions
- Learns per-token representations

### WCED (Whole Cell Expression Decoder)
- No masking - use all β-values
- Reconstruct ALL positions from [CLS] token
- Creates global bottleneck forcing [CLS] to aggregate full profile
- [CLS] can be directly used for downstream tasks

**Usage:**
```bash
# MLM pretraining (default)
python -m bmfm_methylation.pretrain \
    data_path=/path/to/methylation.h5ad \
    pretraining_mode=mlm

# WCED pretraining
python -m bmfm_methylation.pretrain \
    --config-name=pretrain_wced_config \
    data_path=/path/to/methylation.h5ad
```

---

## 3. Data Description

| Item | Description |
|------|-------------|
| **Pretraining sources** | EWAS Data Hub; ClockBase |
| **Collected profiles** | 226,555 |
| **After QC + deduplication** | 154,063 |
| **Platforms** | Illumina 27K / 450K / EPIC |
| **CpG harmonization** | 49,156 sites (≥5 EWAS traits and/or ≥95% presence) |
| **Value range** | β-values in [0,1] |
| **Fine-tuning samples** | 11,453 (train/valid/test = 48/12/40 split) |
| **CpG sites used** | 8,000 per sample |
| **Age range** | 0–100 years |
| **Tissue composition** | Blood ~47.2%, Brain ~34.5%, Others ~18.3% |

---

## 4. Architecture

### Model Configuration
```
BMFM-RNA Encoder:
  - Hidden size: 512
  - Attention heads: 8
  - Layers: 6
  - Intermediate size: 2,048
  - Max position embeddings: 8,002
  - Total encoder params: ~23M

Regression Head (Fine-tuning):
  - Input: 512 (from encoder)
  - Hidden: 256 (with dropout=0.2)
  - Hidden: 128
  - Output: 1 (predicted age)
```

### Architecture Comparison

| Parameter | BMFM-RNA (original) | BMFM-RNA (ours) | MethylGPT (scGPT) |
|-----------|---------------------|-----------------|-------------------|
| **Layers** | 12 | 6 | 6 |
| **Attention heads** | 12 | 8 | 4 |
| **Hidden size d** | 768 | 512 | 64 |
| **FFN size** | 3072 | 2048 | 256 |
| **Head dimension** | 64 | 64 | 16 |
| **Parameters (encoder)** | ~110M | ~23M | ~2M |
| **CpG subset** | — | 8,000 | 8,000 |

---

## 5. Results

### Pretraining Results (MLM)

| Split | Loss | MSE | MAE | PCC |
|-------|------|-----|-----|-----|
| **Train** | 0.00132 | 0.00132 | 0.0221 | 0.994 |
| **Validation** | 0.00147 | 0.00147 | 0.0234 | 0.994 |
| **Test** | 0.00133 | 0.00087 | 0.0195 | **0.997** |

**PCC = 0.997** indicates the model accurately predicts masked β-values from context.

### Fine-tuning Results (n=5 seeds)

| Split | MAE (years) | R² |
|-------|-------------|-----|
| **Train** | 2.34 ± 0.14 | — |
| **Validation** | 4.95 ± 0.06 | — |
| **Test (Mean)** | **4.79 ± 0.14** | **0.927 ± 0.004** |
| **Test (Best)** | **4.64** | **0.932** |

### Final Comparison

| Model | Test MAE (years) | Test R² | CpG sites |
|-------|------------------|---------|-----------|
| Mean prediction | 22.82 | 0.00 | — |
| MethylGPT baseline | 4.95 | 0.911 | 8,000 |
| **BMFM-RNA (Mean, n=5)** | **4.79 ± 0.14** | **0.927** | 8,000 |
| **BMFM-RNA (Best)** | **4.64** | **0.932** | 8,000 |

- **79% error reduction** compared to mean prediction baseline
- **3.3% improvement** in MAE over MethylGPT (mean: 4.79 vs 4.95 years)
- **6.3% improvement** for best model (4.64 vs 4.95 years)
- **1.8% improvement** in R² (0.927 vs 0.911)

---

## 6. Experimental Setup

| Hyperparameter | Value |
|----------------|-------|
| **Optimizer** | AdamW |
| **Learning rate** | 5×10⁻⁴ |
| **Weight decay** | 0.01 |
| **LR schedule** | Cosine decay with 200-step linear warmup |
| **Batch size** | 32 (effective 64 with gradient accumulation ×2) |
| **Max epochs** | 300 |
| **Precision** | 16-bit mixed |
| **Early stopping** | Patience = 60 epochs on val/MAE |
| **Checkpoint** | Best validation MAE |

---

## 7. Project Structure

```
methyl/
├── bmfm_methylation/
│   ├── configs/              # Hydra configuration files
│   │   ├── pretrain_config.yaml
│   │   ├── pretrain_wced_config.yaml
│   │   ├── finetune_config.yaml
│   │   └── model/
│   ├── decoders/             # WCED decoder implementations
│   │   ├── __init__.py
│   │   └── wced_decoder.py
│   ├── data_module.py        # Data loading and preprocessing
│   ├── dataset.py            # PyTorch Dataset classes
│   ├── tokenizer.py          # Multi-field tokenizer for CpG sites
│   ├── model.py              # Model architecture definitions
│   ├── config.py             # Configuration classes
│   ├── pretrain.py           # MLM/WCED pretraining script
│   ├── finetune.py           # Age regression fine-tuning
│   ├── wced_module.py        # WCED training module
│   └── lightning_module.py   # PyTorch Lightning modules
├── scripts/
│   ├── pretrain_*.sh         # SLURM pretraining scripts
│   ├── finetune_*.sh         # SLURM fine-tuning scripts
│   ├── baseline_ridge.py     # Ridge regression baseline
│   └── analyze_*_wandb.py    # W&B analysis scripts
├── wandb_analysis/           # Training curves and analysis
│   ├── finetune/
│   └── multiply_seeds_wandb/
├── docs/
│   └── BMFM_Methylation_Adaptation_Presentation.html
└── README.md
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/netanelazran11/BiomedSciAI_ACL_Project.git
cd BiomedSciAI_ACL_Project

# Create environment
uv venv .venv -p3.12
source .venv/bin/activate

# Install base package
uv pip install -e .

# Install methylation-specific dependencies
cd methyl
pip install -r requirements.txt
```

---

## Usage

### Pretraining

```bash
# MLM pretraining (default)
python -m bmfm_methylation.pretrain \
    data_path=/path/to/methylation.h5ad \
    output_directory=./outputs/pretrain

# WCED pretraining
python -m bmfm_methylation.pretrain \
    --config-name=pretrain_wced_config \
    data_path=/path/to/methylation.h5ad \
    output_directory=./outputs/pretrain_wced
```

### Fine-tuning

```bash
# Single-seed
python -m bmfm_methylation.finetune \
    data_path=/path/to/methylation.h5ad \
    checkpoint_path=/path/to/pretrained.ckpt \
    output_directory=./outputs/finetune

# Multi-seed training
for seed in 40 41 42 43 44; do
    python -m bmfm_methylation.finetune \
        data_path=/path/to/methylation.h5ad \
        checkpoint_path=/path/to/pretrained.ckpt \
        seed.seed_value=$seed
done
```

---

## WandB Dashboards

- [BMFM-RNA Fine-tuning](https://wandb.ai/netanelazran11-hebrew-university-of-jerusalem/finetune-bmfm-rna-methylation-8k)
- [Multi-Seed Project](https://wandb.ai/netanelazran11-hebrew-university-of-jerusalem/finetune-bmfm-multiseed)

---

## References

1. Ying, K., et al., "MethylGPT: a foundation model for the DNA methylome," *bioRxiv*, 2024.
2. Li, D., et al., "BMFM-RNA: A biomedical foundation model for single-cell transcriptomics," IBM Research, 2024.
3. Horvath, S., "DNA methylation age of human tissues and cell types," *Genome Biology*, 14(10):R115, 2013.
4. Hannum, G., et al., "Genome-wide methylation profiles reveal quantitative views of human aging rates," *Molecular Cell*, 49(2):359–367, 2013.
5. de Lima Camillo, L.P., et al., "AltumAge: A pan-tissue DNA methylation epigenetic clock based on deep learning," *npj Aging*, 8(1):1–15, 2022.

---

## Citation

```bibtex
@misc{azran2025methylation,
  title={Methylation Age Prediction with BMFM-RNA Architecture},
  author={Azran, Netanel},
  year={2025},
  howpublished={\url{https://github.com/netanelazran11/BiomedSciAI_ACL_Project}}
}
```

---

## Contact

**Author:** Netanel Azran
**Institution:** Hebrew University of Jerusalem
**Email:** netanelazran11@gmail.com
**GitHub:** [@netanelazran11](https://github.com/netanelazran11)
