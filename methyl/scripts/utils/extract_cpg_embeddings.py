"""
One-time script: extract BMFM-DNA embeddings for all AltumAge CpG sites.

For each CpG in probe_ids_type3_21k.csv:
  1. Look up genomic coordinates in HM450 hg38 manifest
  2. Extract ±512bp DNA window from hg38.fa
  3. Run BMFM-DNA → mean-pool hidden states → 768-dim embedding

Output: cpg_embeddings_bmfdna_21k.npy  shape [21368, 768]
        cpg_ids_order.txt              CpG IDs in same row order as embeddings

Usage (via SLURM):
    sbatch scripts/llama/extract_cpg_embeddings.sh
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE       = "/sci/labs/benjamin.yakir/netanel.azran"
PROBE_CSV  = f"{BASE}/data/data_methyl_21k_h5ad/probe_ids_type3_21k.csv"
MANIFEST   = f"{BASE}/data/manifests/HM450.hg38.manifest.tsv"
GENOME_FA  = f"{BASE}/data/genomes/hg38/hg38.fa"
HF_CACHE   = f"{BASE}/data/hf_cache"
OUT_DIR    = f"{BASE}/data/cpg_embeddings"
OUT_NPY    = f"{OUT_DIR}/cpg_embeddings_bmfdna_21k.npy"
OUT_IDS    = f"{OUT_DIR}/cpg_ids_order.txt"

MODEL_ID   = "ibm-research/biomed.dna.ref.modernbert.113m.v1"
# Local path to the downloaded last.ckpt (PyTorch Lightning checkpoint)
LOCAL_CKPT = f"{HF_CACHE}/bmfdna_last.ckpt"
WINDOW     = 512      # ±512 bp around CpG position
BATCH_SIZE = 64       # sequences per forward pass
# ─────────────────────────────────────────────────────────────────────────────

os.environ["HF_HOME"] = HF_CACHE
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("BMFM-DNA CpG Embedding Extraction")
print("=" * 60)

# ── 1. Load CpG IDs ───────────────────────────────────────────────────────────
print(f"\n[1] Loading CpG IDs from {PROBE_CSV}")
probe_df = pd.read_csv(PROBE_CSV)
cpg_ids = probe_df["illumina_probe_id"].tolist()
print(f"    {len(cpg_ids)} CpG IDs loaded")

# ── 2. Load manifest → coordinates ───────────────────────────────────────────
print(f"\n[2] Loading manifest from {MANIFEST}")
manifest = pd.read_csv(MANIFEST, sep="\t", low_memory=False)
print(f"    Manifest columns: {list(manifest.columns[:8])}")

# Identify key columns (different manifests use different names)
id_col  = "probeID"   if "probeID"  in manifest.columns else manifest.columns[0]
chr_col = "CpG_chrm"  if "CpG_chrm" in manifest.columns else "CHR"
pos_col = "CpG_beg"   if "CpG_beg"  in manifest.columns else "MAPINFO"

manifest = manifest.set_index(id_col)
print(f"    Using columns: id={id_col}, chr={chr_col}, pos={pos_col}")
print(f"    Manifest probes: {len(manifest)}")

# ── 3. Load BMFM-DNA ──────────────────────────────────────────────────────────
# SCModernBertModel is the base encoder (no MLM head) — exactly what we need.
# PreTrainedTokenizerFast loads BMFM-DNA's own k-mer BPE tokenizer for DNA sequences.
# NOTE: this is BMFM-DNA's tokenizer for DNA text, completely separate from
#       the methylation tokenizer (cg_id → integer) used by MethylLlama.
#
# Loading strategy (SCModernBert is NOT registered with HF AutoModel/AutoConfig):
#   1. Find HF snapshot dir → has config.json + tokenizer files
#   2. Load tokenizer from snapshot dir
#   3. Instantiate model from config, then load weights from local last.ckpt
print(f"\n[3] Loading BMFM-DNA from local files")
from bmfm_targets.models.predictive.scmodernbert.modeling_scmodernbert import SCModernBertModel
from transformers import PreTrainedTokenizerFast

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"    Device: {device}")

# Find snapshot directory in HF cache
hub_dir = Path(HF_CACHE) / "hub"
model_cache_name = "models--" + MODEL_ID.replace("/", "--")
snapshots_dir = hub_dir / model_cache_name / "snapshots"
if not snapshots_dir.exists():
    raise FileNotFoundError(
        f"HF snapshot dir not found: {snapshots_dir}\n"
        f"Run: huggingface-cli download {MODEL_ID} --cache-dir {HF_CACHE}"
    )
# Pick the first (and usually only) snapshot hash
snapshot_dirs = list(snapshots_dir.iterdir())
if not snapshot_dirs:
    raise FileNotFoundError(f"No snapshots found in {snapshots_dir}")
snapshot_dir = snapshot_dirs[0]
print(f"    Snapshot dir: {snapshot_dir}")

# Load tokenizer from snapshot dir
tokenizer = PreTrainedTokenizerFast.from_pretrained(str(snapshot_dir))
print(f"    Tokenizer loaded: vocab_size={tokenizer.vocab_size}")

# Load model config then instantiate
# config_class is set on the model class itself (no separate import needed)
config = SCModernBertModel.config_class.from_pretrained(str(snapshot_dir))
model = SCModernBertModel(config)

# Load weights from local PyTorch Lightning checkpoint
print(f"    Loading weights from {LOCAL_CKPT}")
if not Path(LOCAL_CKPT).exists():
    raise FileNotFoundError(f"Checkpoint not found: {LOCAL_CKPT}")

ckpt = torch.load(LOCAL_CKPT, map_location="cpu", weights_only=False)
# PL checkpoints store weights under "state_dict" key
# Keys may have a prefix like "model." — strip it to match bare model keys
if "state_dict" in ckpt:
    raw_sd = ckpt["state_dict"]
else:
    raw_sd = ckpt  # already a plain state_dict

# Determine prefix (e.g. "model.") by checking first key
first_key = next(iter(raw_sd))
prefix = ""
if first_key.startswith("model."):
    prefix = "model."
elif "." in first_key:
    # Try to detect any common prefix
    candidate = first_key.split(".")[0] + "."
    if all(k.startswith(candidate) for k in list(raw_sd.keys())[:10]):
        prefix = candidate

if prefix:
    print(f"    Stripping state_dict prefix: '{prefix}'")
    sd = {k[len(prefix):]: v for k, v in raw_sd.items() if k.startswith(prefix)}
else:
    sd = raw_sd

missing, unexpected = model.load_state_dict(sd, strict=False)
if missing:
    print(f"    WARNING: {len(missing)} missing keys (first 5: {missing[:5]})")
if unexpected:
    print(f"    WARNING: {len(unexpected)} unexpected keys (first 5: {unexpected[:5]})")

model = model.to(device).eval()
n_params = sum(p.numel() for p in model.parameters())
print(f"    Model loaded: {n_params/1e6:.1f}M params")

# ── 4. Load genome ────────────────────────────────────────────────────────────
print(f"\n[4] Opening genome {GENOME_FA}")
from pyfaidx import Fasta
genome = Fasta(GENOME_FA)
print(f"    Chromosomes available: {len(genome.keys())}")

# ── 5. Extract embeddings ─────────────────────────────────────────────────────
print(f"\n[5] Extracting embeddings (window=±{WINDOW}bp, batch={BATCH_SIZE})")

embeddings = np.zeros((len(cpg_ids), 768), dtype=np.float32)
missing = []

def get_dna_sequence(cpg_id):
    """Get DNA window around CpG site. Returns None if not in manifest."""
    if cpg_id not in manifest.index:
        return None
    row = manifest.loc[cpg_id]
    chrom = str(row[chr_col])
    pos   = int(row[pos_col])

    # Ensure chromosome name matches genome (chr1 vs 1)
    if chrom not in genome and f"chr{chrom}" in genome:
        chrom = f"chr{chrom}"
    if chrom not in genome:
        return None

    chrom_len = len(genome[chrom])
    start = max(0, pos - WINDOW)
    end   = min(chrom_len, pos + WINDOW)
    seq   = str(genome[chrom][start:end])
    return seq.upper()

def embed_batch(sequences):
    """Run BMFM-DNA on a batch of DNA sequences → mean-pooled embeddings."""
    inputs = tokenizer(
        sequences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    ).to(device)
    with torch.no_grad():
        # SCModernBertForMaskedLM: use the base model's hidden states
        outputs = model(**inputs)  # SCModernBertModel → last_hidden_state
    # Mean pool over sequence length (exclude padding)
    hidden = outputs.last_hidden_state          # [B, L, 768]
    mask   = inputs["attention_mask"].unsqueeze(-1).float()  # [B, L, 1]
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)    # [B, 768]
    return pooled.cpu().float().numpy()

batch_seqs  = []
batch_idxs  = []

for i, cpg_id in enumerate(cpg_ids):
    seq = get_dna_sequence(cpg_id)
    if seq is None or len(seq) < 10:
        missing.append(cpg_id)
        batch_seqs.append("ACGT" * 10)   # dummy — will be overwritten with zeros
        batch_idxs.append((i, False))
    else:
        batch_seqs.append(seq)
        batch_idxs.append((i, True))

    if len(batch_seqs) == BATCH_SIZE or i == len(cpg_ids) - 1:
        embs = embed_batch(batch_seqs)
        for (idx, valid), emb in zip(batch_idxs, embs):
            if valid:
                embeddings[idx] = emb
        batch_seqs  = []
        batch_idxs  = []

        if (i + 1) % 1000 == 0 or i == len(cpg_ids) - 1:
            print(f"    {i+1}/{len(cpg_ids)} CpGs processed, {len(missing)} missing")

print(f"\n    Done. Missing/skipped: {len(missing)} CpGs")
if missing:
    print(f"    First 5 missing: {missing[:5]}")

# ── 6. Save ───────────────────────────────────────────────────────────────────
print(f"\n[6] Saving embeddings to {OUT_NPY}")
np.save(OUT_NPY, embeddings)
with open(OUT_IDS, "w") as f:
    f.write("\n".join(cpg_ids))

print(f"    Saved: {embeddings.shape} float32 ({embeddings.nbytes/1e6:.1f} MB)")
print(f"    CpG ID order saved to {OUT_IDS}")
print("\n" + "=" * 60)
print("Extraction complete.")
print("=" * 60)
