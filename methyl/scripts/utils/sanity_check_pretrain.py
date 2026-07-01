#!/usr/bin/env python3
"""
Sanity check for the full pretrain pipeline before the big run.

Checks (in order):
  A. Environment & paths
  B. Data loading (20 real samples from train split)
  C. WCEDCollator batch structure
  D. Genomic RoPE position_ids correctness
  E. NaN handling
  F. Contrastive views (v1 vs v2)
  G. Model forward pass (full architecture)
  H. Loss computation (recon + InfoNCE)
  I. Backward pass & gradients
  J. 3 simulated training steps (loss changes)

Usage (as sbatch job):
  sbatch scripts/utils/run_sanity_check.sh
"""

import sys
import os
import traceback

REPO = "/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
sys.path.insert(0, REPO)
os.chdir(REPO)

# ─── Counters ────────────────────────────────────────────────────────────────
_passed = 0
_failed = 0
_failures = []

def ok(name):
    global _passed
    _passed += 1
    print(f"  PASS  {name}")

def fail(name, reason=""):
    global _failed
    _failed += 1
    msg = f"  FAIL  {name}" + (f" — {reason}" if reason else "")
    print(msg)
    _failures.append(msg)

def section(title):
    print(f"\n=== {title} ===")

# ─── Config (mirrors pretrain_llama_small_6L_contrastive.sh) ─────────────────
DATA_PATH        = "/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_pretrain_type3_h5ad/methylgpt_pretrain_type3.h5ad"
PROBE_IDS_CSV    = "/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_pretrain_type3_h5ad/probe_ids_type3_pretrain.csv"
GENOMIC_RANK_NPY = f"{REPO}/outputs/cpg_genomic_sort/cpg_genomic_rank.npy"
TOKENIZER_PATH   = f"{REPO}/tokenizer_llama_pretrain49k"

HIDDEN_SIZE      = 256
NUM_LAYERS       = 6
NUM_HEADS        = 4
INTERMEDIATE_SIZE= 512
ROPE_THETA       = 10000.0
N_SIN_BASIS      = 48
BASIS_SCALE      = 2.0

SUBSET_K         = 49156
INPUT_RATIO      = 0.5
CONTRASTIVE      = True
CONTRASTIVE_WT   = 0.05
CONTRASTIVE_TEMP = 0.1
BATCH_SIZE       = 4     # small for sanity check
N_SAMPLES        = 20    # load only 20 real samples

# =============================================================================
# A. Environment & Paths
# =============================================================================
section("A  Environment & paths")

import torch
import numpy as np

print(f"  torch: {torch.__version__}  cuda: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
    print("  Running on CPU")

for label, path in [
    ("data h5ad",        DATA_PATH),
    ("probe_ids csv",    PROBE_IDS_CSV),
    ("genomic rank npy", GENOMIC_RANK_NPY),
    ("tokenizer dir",    TOKENIZER_PATH),
]:
    if os.path.exists(path):
        ok(f"{label} exists")
    else:
        fail(f"{label} exists", f"not found: {path}")

# =============================================================================
# B. Data Loading
# =============================================================================
section("B  Data loading (20 real train samples)")

try:
    from bmfm_methylation.shared.tokenizer import (
        extract_cpg_sites_from_h5ad,
        create_methylation_multifield_tokenizer,
    )
    from bmfm_methylation.shared.data_module import MethylationDataset, WCEDCollator
    from bmfm_targets.tokenization import MultiFieldTokenizer
    ok("imports succeeded")
except Exception as e:
    fail("imports", str(e))
    print(traceback.format_exc())
    sys.exit(1)

try:
    tokenizer = MultiFieldTokenizer.from_pretrained(TOKENIZER_PATH)
    ok("tokenizer loaded")
except Exception as e:
    fail("tokenizer loaded", str(e))
    sys.exit(1)

try:
    cpg_sites = extract_cpg_sites_from_h5ad(DATA_PATH, probe_ids_csv=PROBE_IDS_CSV)
    n_cpgs = len(cpg_sites)
    print(f"  {n_cpgs} CpG sites")
    if n_cpgs == SUBSET_K:
        ok(f"n_cpgs == {SUBSET_K}")
    else:
        fail(f"n_cpgs == {SUBSET_K}", f"got {n_cpgs}")
except Exception as e:
    fail("cpg_sites extraction", str(e))
    sys.exit(1)

try:
    dataset = MethylationDataset(
        h5ad_path=DATA_PATH,
        split="train",
        tokenizer=tokenizer,
        cpg_sites=cpg_sites,
        split_column="split",
        bmfm_style=False,
        probe_ids_csv=PROBE_IDS_CSV,
    )
    n_train = len(dataset)
    print(f"  train dataset: {n_train} samples")
    ok(f"dataset loaded ({n_train} samples)")
except Exception as e:
    fail("dataset loaded", str(e))
    print(traceback.format_exc())
    sys.exit(1)

# Sample N_SAMPLES evenly
sample_indices = np.linspace(0, n_train - 1, N_SAMPLES, dtype=int).tolist()
try:
    samples = [dataset[i] for i in sample_indices]
    ok(f"{N_SAMPLES} samples fetched")
except Exception as e:
    fail("sample fetch", str(e))
    sys.exit(1)

# =============================================================================
# C. WCEDCollator — batch structure
# =============================================================================
section("C  WCEDCollator batch structure")

try:
    genomic_rank = np.load(GENOMIC_RANK_NPY)
    ok(f"genomic_rank loaded (shape={genomic_rank.shape})")
    if len(genomic_rank) == n_cpgs:
        ok("genomic_rank length == n_cpgs")
    else:
        fail("genomic_rank length", f"got {len(genomic_rank)}, expected {n_cpgs}")
except Exception as e:
    fail("genomic_rank load", str(e))
    sys.exit(1)

try:
    collator = WCEDCollator(
        tokenizer=tokenizer,
        cpg_sites=cpg_sites,
        vocab_size=SUBSET_K,
        input_ratio=INPUT_RATIO,
        contrastive=CONTRASTIVE,
        fixed_subset_seed=42,
        genomic_rank_path=GENOMIC_RANK_NPY,
    )
    ok("WCEDCollator created")
except Exception as e:
    fail("WCEDCollator created", str(e))
    sys.exit(1)

try:
    batch = collator(samples[:BATCH_SIZE])
    ok("collator called successfully")
except Exception as e:
    fail("collator call", str(e))
    print(traceback.format_exc())
    sys.exit(1)

REQUIRED_KEYS = ["cpg_ids", "beta_values", "attention_mask", "input_mask",
                 "all_betas", "valid_mask", "position_ids",
                 "cpg_ids_v2", "beta_values_v2", "attention_mask_v2",
                 "input_mask_v2", "position_ids_v2"]

for key in REQUIRED_KEYS:
    if key in batch:
        ok(f"batch has '{key}'")
    else:
        fail(f"batch has '{key}'", "key missing")

B = BATCH_SIZE
max_len = int(SUBSET_K * INPUT_RATIO) + 1
print(f"  expected max_input_len={max_len}")

for key, expected_shape in [
    ("cpg_ids",        (B, max_len)),
    ("beta_values",    (B, max_len)),
    ("attention_mask", (B, max_len)),
    ("input_mask",     (B, SUBSET_K)),
    ("all_betas",      (B, SUBSET_K)),
    ("valid_mask",     (B, SUBSET_K)),
    ("position_ids",   (B, max_len)),
]:
    if key not in batch:
        continue
    shape = tuple(batch[key].shape)
    if shape == expected_shape:
        ok(f"shape {key}: {shape}")
    else:
        fail(f"shape {key}", f"got {shape}, expected {expected_shape}")

# =============================================================================
# D. Genomic RoPE position_ids correctness
# =============================================================================
section("D  Genomic RoPE position_ids")

pid = batch["position_ids"]   # [B, max_len]
attn = batch["attention_mask"]  # [B, max_len]

# CLS at position 0
cls_positions = pid[:, 0]
if (cls_positions == 0).all():
    ok("CLS position_id == 0 (all samples)")
else:
    fail("CLS position_id == 0", f"got {cls_positions.tolist()}")

# PAD positions == 0
for i in range(B):
    seq_len = attn[i].sum().item()
    pad_positions = pid[i, int(seq_len):]
    if (pad_positions == 0).all():
        ok(f"  sample {i}: PAD positions == 0")
    else:
        fail(f"  sample {i}: PAD positions == 0", f"max={pad_positions.max()}")

# CpG positions >= 1
cpg_positions = pid[:, 1:]  # skip CLS
for i in range(B):
    seq_len = attn[i].sum().item()
    real_cpg_pos = pid[i, 1:int(seq_len)]
    if len(real_cpg_pos) == 0:
        fail(f"  sample {i}: has CpG positions", "empty")
        continue
    if (real_cpg_pos >= 1).all():
        ok(f"  sample {i}: all CpG positions >= 1 (min={real_cpg_pos.min().item()})")
    else:
        fail(f"  sample {i}: CpG positions >= 1", f"min={real_cpg_pos.min().item()}")
    # Monotone increasing (genomic order)
    diffs = real_cpg_pos[1:].float() - real_cpg_pos[:-1].float()
    if (diffs > 0).all():
        ok(f"  sample {i}: positions strictly monotone increasing (genomic order)")
    else:
        fail(f"  sample {i}: positions monotone increasing", f"min_diff={diffs.min().item()}")

# position_ids <= SUBSET_K
max_pos = pid.max().item()
if max_pos <= SUBSET_K:
    ok(f"max position_id={max_pos} <= {SUBSET_K}")
else:
    fail("max position_id <= SUBSET_K", f"got {max_pos}")

# position_ids NOT sequential (0,1,2,...) — proves genomic, not slot-order
pid_v1 = batch["position_ids"]
pid_v2 = batch["position_ids_v2"]
sequential = torch.arange(max_len).unsqueeze(0).expand(B, -1)
if not torch.equal(pid_v1, sequential):
    ok("position_ids are genomic (not sequential 0,1,2,...)")
else:
    fail("position_ids are genomic", "looks sequential — RoPE might not be using genomic ranks")

# v1 != v2 (independent random views)
if not torch.equal(pid_v1, pid_v2):
    ok("position_ids_v1 != position_ids_v2 (independent views)")
else:
    fail("position_ids_v1 != position_ids_v2", "views are identical — contrastive won't work")

# =============================================================================
# E. NaN handling
# =============================================================================
section("E  NaN handling")

all_betas = batch["all_betas"]
valid_mask = batch["valid_mask"]
input_mask = batch["input_mask"]

# No NaN in all_betas (should have been replaced by 0)
if not torch.isnan(all_betas).any():
    ok("all_betas has no NaN (replaced by 0)")
else:
    fail("all_betas has no NaN", f"found {torch.isnan(all_betas).sum()} NaN values")

# valid_mask is False where NaN was
for i in range(B):
    n_valid = valid_mask[i].sum().item()
    pct_valid = 100 * n_valid / SUBSET_K
    print(f"  sample {i}: {n_valid}/{SUBSET_K} valid CpGs ({pct_valid:.1f}%)")
ok("valid_mask computed per sample")

# Input only samples from valid positions
for i in range(B):
    input_i = input_mask[i]    # True = in input
    valid_i = valid_mask[i]    # True = non-NaN
    invalid_in_input = (input_i & ~valid_i).sum().item()
    if invalid_in_input == 0:
        ok(f"  sample {i}: no NaN CpGs in input")
    else:
        fail(f"  sample {i}: no NaN CpGs in input", f"{invalid_in_input} NaN positions in input")

# Reconstruction mask = non_input AND valid
non_input = ~input_mask
recon_mask = non_input & valid_mask
n_recon = recon_mask.float().sum(dim=1)
for i in range(B):
    n = int(n_recon[i].item())
    if n > 0:
        ok(f"  sample {i}: recon_mask has {n} positions (held-out non-NaN CpGs)")
    else:
        fail(f"  sample {i}: recon_mask has positions", "0 positions to reconstruct")

# =============================================================================
# F. Contrastive views
# =============================================================================
section("F  Contrastive views (v1 vs v2)")

cpg_ids_v1 = batch["cpg_ids"]
cpg_ids_v2 = batch["cpg_ids_v2"]
betas_v1   = batch["beta_values"]
betas_v2   = batch["beta_values_v2"]
attn_v1    = batch["attention_mask"]
attn_v2    = batch["attention_mask_v2"]

if not torch.equal(cpg_ids_v1, cpg_ids_v2):
    ok("cpg_ids_v1 != cpg_ids_v2 (independent subsets)")
else:
    fail("cpg_ids_v1 != cpg_ids_v2", "views are identical")

for i in range(B):
    len1 = attn_v1[i].sum().item()
    len2 = attn_v2[i].sum().item()
    print(f"  sample {i}: v1 len={int(len1)}, v2 len={int(len2)}")
ok("both views have non-zero lengths")

# =============================================================================
# G. Model forward pass
# =============================================================================
section("G  Model forward pass")

try:
    from bmfm_methylation.llama.model import MethylLlamaConfig, MethylLlamaModel
    ok("model imports")
except Exception as e:
    fail("model imports", str(e))
    sys.exit(1)

n_special = 5
model_vocab_size = n_cpgs + n_special

try:
    config = MethylLlamaConfig(
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        intermediate_size=INTERMEDIATE_SIZE,
        vocab_size=model_vocab_size,
        rope_theta=ROPE_THETA,
        n_sin_basis=N_SIN_BASIS,
        basis_scale=BASIS_SCALE,
    )
    model = MethylLlamaModel(config).to(DEVICE).eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  model: {total_params/1e6:.2f}M params")
    ok("model created")
except Exception as e:
    fail("model created", str(e))
    print(traceback.format_exc())
    sys.exit(1)

def to_dev(x):
    return x.to(DEVICE) if x is not None else None

cpg_ids = batch["cpg_ids"].to(DEVICE)
beta_values = batch["beta_values"].to(DEVICE)
attn_mask = batch["attention_mask"].to(DEVICE)
pos_ids_v1 = batch["position_ids"].to(DEVICE)

input_ids = torch.stack([cpg_ids.float(), beta_values], dim=1)  # [B, 2, L]

try:
    with torch.no_grad():
        # Test 1: with position_ids (Level 2 Genomic RoPE)
        out_genomic = model(input_ids=input_ids, attention_mask=attn_mask, position_ids=pos_ids_v1)
        lhs_genomic = out_genomic.last_hidden_state   # [B, L, D]
        cls_genomic = out_genomic.pooler_output       # [B, D]

    if tuple(lhs_genomic.shape) == (B, max_len, HIDDEN_SIZE):
        ok(f"last_hidden_state shape: {tuple(lhs_genomic.shape)}")
    else:
        fail("last_hidden_state shape", f"got {tuple(lhs_genomic.shape)}")

    if tuple(cls_genomic.shape) == (B, HIDDEN_SIZE):
        ok(f"pooler_output shape: {tuple(cls_genomic.shape)}")
    else:
        fail("pooler_output shape", f"got {tuple(cls_genomic.shape)}")

    if not torch.isnan(lhs_genomic).any():
        ok("last_hidden_state has no NaN")
    else:
        fail("last_hidden_state no NaN", f"{torch.isnan(lhs_genomic).sum()} NaN values")

    if not torch.isnan(cls_genomic).any():
        ok("pooler_output has no NaN")
    else:
        fail("pooler_output no NaN", f"{torch.isnan(cls_genomic).sum()} NaN values")

    # Test 2: without position_ids (Level 1 sequential RoPE)
    out_seq = model(input_ids=input_ids, attention_mask=attn_mask, position_ids=None)
    cls_seq = out_seq.pooler_output

    # Genomic RoPE must produce different output than sequential
    if not torch.allclose(cls_genomic, cls_seq, atol=1e-4):
        ok("Genomic RoPE changes output vs sequential (Level 2 is active)")
    else:
        fail("Genomic RoPE changes output", "outputs identical — position_ids may not be used")

except Exception as e:
    fail("model forward pass", str(e))
    print(traceback.format_exc())

# =============================================================================
# H. Loss computation (recon + InfoNCE)
# =============================================================================
section("H  Loss computation")

try:
    from bmfm_methylation.llama.wced_llama import WCEDLlamaModule
    ok("WCEDLlamaModule import")
except Exception as e:
    fail("WCEDLlamaModule import", str(e))
    sys.exit(1)

try:
    module = WCEDLlamaModule(
        model_config=config,
        learning_rate=3e-4,
        weight_decay=0.01,
        warmup_steps=3000,
        vocab_size=SUBSET_K,
        contrastive_weight=CONTRASTIVE_WT,
        contrastive_temp=CONTRASTIVE_TEMP,
        normalize_loss=False,
        age_weight=0.0,
        decoder_dropout=0.1,
    ).to(DEVICE)
    ok("WCEDLlamaModule created")
except Exception as e:
    fail("WCEDLlamaModule created", str(e))
    print(traceback.format_exc())
    sys.exit(1)

try:
    batch_dev = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

    module.train()
    out = module._shared_step(batch_dev, "train")

    recon_loss = out["recon_loss"].item()
    contrastive_loss = out["contrastive_loss"].item()
    total_loss = out["loss"].item()

    print(f"  recon_loss      = {recon_loss:.6f}")
    print(f"  contrastive_loss= {contrastive_loss:.6f}")
    print(f"  total_loss      = {total_loss:.6f}")

    if np.isfinite(recon_loss) and recon_loss > 0:
        ok(f"recon_loss finite and > 0 ({recon_loss:.4f})")
    else:
        fail("recon_loss", f"got {recon_loss}")

    if np.isfinite(contrastive_loss) and contrastive_loss > 0:
        ok(f"contrastive_loss (InfoNCE) finite and > 0 ({contrastive_loss:.4f})")
    else:
        fail("contrastive_loss", f"got {contrastive_loss} — may be 0 if contrastive not wired")

    if np.isfinite(total_loss):
        ok(f"total_loss finite ({total_loss:.4f})")
    else:
        fail("total_loss finite", f"got {total_loss}")

    expected_total = recon_loss + CONTRASTIVE_WT * contrastive_loss
    if abs(total_loss - expected_total) < 1e-4:
        ok(f"total = recon + {CONTRASTIVE_WT}*InfoNCE (formula correct)")
    else:
        fail("loss formula", f"got {total_loss:.6f}, expected {expected_total:.6f}")

except Exception as e:
    fail("loss computation", str(e))
    print(traceback.format_exc())

# =============================================================================
# I. Backward pass & gradients
# =============================================================================
section("I  Backward pass & gradients")

try:
    module.zero_grad()
    out2 = module._shared_step(batch_dev, "train")
    out2["loss"].backward()

    grad_norms = []
    nan_grads = []
    for name, param in module.named_parameters():
        if param.grad is not None:
            gn = param.grad.norm().item()
            grad_norms.append(gn)
            if not np.isfinite(gn):
                nan_grads.append(name)

    if len(grad_norms) > 0:
        ok(f"{len(grad_norms)} parameters have gradients")
    else:
        fail("gradients exist", "no parameters have gradients")

    if len(nan_grads) == 0:
        ok(f"all gradients finite (max_norm={max(grad_norms):.4f})")
    else:
        fail("all gradients finite", f"NaN grad in: {nan_grads[:3]}")

    total_grad_norm = np.sqrt(sum(g**2 for g in grad_norms))
    print(f"  total grad norm = {total_grad_norm:.4f}")
    if total_grad_norm > 0:
        ok(f"total grad norm > 0 ({total_grad_norm:.4f})")
    else:
        fail("total grad norm > 0", "zero gradients — nothing is learning")

except Exception as e:
    fail("backward pass", str(e))
    print(traceback.format_exc())

# =============================================================================
# J. 3 training steps — loss changes
# =============================================================================
section("J  3 training steps")

try:
    optim = module.configure_optimizers()
    if isinstance(optim, tuple):
        optimizer = optim[0][0]
    elif isinstance(optim, dict):
        optimizer = optim["optimizer"]
    else:
        optimizer = optim

    losses = []
    module.train()
    # Create two batches for variety
    batch2 = collator(samples[BATCH_SIZE:2*BATCH_SIZE])
    batch2_dev = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                  for k, v in batch2.items()}

    for step, b in enumerate([batch_dev, batch2_dev, batch_dev]):
        optimizer.zero_grad()
        o = module._shared_step(b, "train")
        o["loss"].backward()
        torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
        optimizer.step()
        l = o["loss"].item()
        losses.append(l)
        print(f"  step {step+1}: loss={l:.6f}")

    if all(np.isfinite(l) for l in losses):
        ok("all 3 steps produced finite loss")
    else:
        fail("all steps finite", f"losses: {losses}")

    if len(set(f"{l:.6f}" for l in losses)) > 1:
        ok("loss changes across steps (model is updating)")
    else:
        fail("loss changes", "all steps identical — optimizer may not be working")

except Exception as e:
    fail("training steps", str(e))
    print(traceback.format_exc())

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print(f"  Results: {_passed}/{_passed+_failed} passed,  {_failed} failed")
if _failures:
    print("\n  FAILURES:")
    for f in _failures:
        print(f"  {f}")
    print("\n  DO NOT submit the pretrain job — fix failures first.")
else:
    print("  All checks passed — safe to submit pretrain job.")
print("=" * 60)
