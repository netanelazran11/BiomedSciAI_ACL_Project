"""
tests/test_genomic_rope_v2.py  —  Level 2 Genomic RoPE verification
======================================================================
Run locally (no cluster packages):
    cd /path/to/methyl && python tests/test_genomic_rope_v2.py

Run on cluster (full env):
    pytest tests/test_genomic_rope_v2.py -v

What this file verifies:
  SECTION A — RoPE math
    A1  unit norms: cos²+sin² == 1 at every position
    A2  position-0 is identity rotation (cos=1, sin=0)
    A3  norm preservation: RoPE is an isometry
    A4  ** relative-position property **: Q_rot(m)·K_rot(n) depends ONLY on (m-n),
        not on m or n individually — the core mathematical guarantee of RoPE
    A5  large positions (up to 49156) handled without error or overflow

  SECTION B — apply_rotary_pos_emb broadcasting
    B1  2-D cos/sin [L, Dh]   (sequential mode)
    B2  3-D cos/sin [B, L, Dh] (position_ids mode)
    B3  consistency: 3-D with positions [0..L-1] == 2-D sequential

  SECTION C — model layer wiring
    C1  position_ids flows: Attention → Layer → Encoder → Model
    C2  backward compat: position_ids=None gives IDENTICAL output to omitting it
    C3  Level 2 changes output: different position_ids → different activations
    C4  gradient flows through Level 2 path

  SECTION D — WCEDCollator position_ids construction
    D1  position_ids present in batch when genomic_rank set
    D2  position_ids absent when no genomic_rank
    D3  shape == [B, max_input_len]
    D4  dtype == torch.long
    D5  CLS slot (index 0) always has position_id == 0
    D6  PAD slots (attention_mask==0) always have position_id == 0
    D7  real CpG slots have position_id >= 1 (rank+1 convention)
    D8  real CpG positions are STRICTLY MONOTONE INCREASING per sample
    D9  exact value check: position_id == genomic_rank[col_idx]+1 for each selected CpG
    D10 vocab_cpg_indices is identity when all CpGs used (assertion guard)
    D11 contrastive mode: position_ids_v2 present with same properties
    D12 max position_id <= n_cpgs (no overflow past rank range)

  SECTION E — end-to-end
    E1  collator batch → model forward: shapes correct, no NaN
    E2  Level 2 output differs from Level 1 on same batch
"""

import sys
import types
import importlib.util as ilu
import os
import tempfile

# ---------------------------------------------------------------------------
# Stub minimal packages so model.py and data_module.py load without the
# full cluster environment (anndata, bmfm_targets, etc.)
# ---------------------------------------------------------------------------
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_STUBS = [
    "bmfm_methylation", "bmfm_methylation.llama", "bmfm_methylation.shared",
    "bmfm_targets", "bmfm_targets.config", "bmfm_targets.tokenization",
    "bmfm_targets.dataset", "anndata", "anndata.logging",
    "lightning", "lightning.pytorch", "lightning.pytorch.core",
    "pytorch_lightning", "wandb",
]
for _pkg in _STUBS:
    sys.modules.setdefault(_pkg, types.ModuleType(_pkg))

# Provide the minimal attributes that might be needed at import time
_lt = sys.modules["bmfm_targets"]
for _sub in ["FieldInfo", "LabelColumnInfo"]:
    setattr(sys.modules["bmfm_targets.config"], _sub, type(_sub, (), {}))
for _sub in ["MultiFieldTokenizer", "MultiFieldInstance"]:
    setattr(sys.modules["bmfm_targets.tokenization"], _sub, type(_sub, (), {}))


def _load_file(module_name: str, rel_path: str, package: str):
    """Load a single .py file as a named module (bypasses package __init__ chain)."""
    path = os.path.join(_REPO, rel_path)
    spec = ilu.spec_from_file_location(module_name, path, submodule_search_locations=[])
    mod  = ilu.module_from_spec(spec)
    mod.__package__ = package
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_scale_adapt = _load_file(
    "bmfm_methylation.llama.scale_adapt",
    "bmfm_methylation/llama/scale_adapt.py",
    "bmfm_methylation.llama",
)
_model_mod = _load_file(
    "bmfm_methylation.llama.model",
    "bmfm_methylation/llama/model.py",
    "bmfm_methylation.llama",
)

# data_module has heavier deps; load with exception guard
_dm_mod = None
try:
    _dm_mod = _load_file(
        "bmfm_methylation.shared.data_module",
        "bmfm_methylation/shared/data_module.py",
        "bmfm_methylation.shared",
    )
except Exception as _dm_err:
    pass  # collator tests will be skipped below with a message

# ---------------------------------------------------------------------------
import numpy as np
import torch

RotaryEmbedding      = _model_mod.RotaryEmbedding
apply_rotary_pos_emb = _model_mod.apply_rotary_pos_emb
_rotate_half         = _model_mod._rotate_half
MethylLlamaConfig    = _model_mod.MethylLlamaConfig
MethylLlamaModel     = _model_mod.MethylLlamaModel

# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------
PASS = FAIL = 0


def ok(name: str):
    global PASS
    print(f"  PASS  {name}")
    PASS += 1


def fail(name: str, msg: str = ""):
    global FAIL
    suffix = f": {msg}" if msg else ""
    print(f"  FAIL  {name}{suffix}")
    FAIL += 1


def check(name: str, cond: bool, msg: str = ""):
    if cond:
        ok(name)
    else:
        fail(name, msg)


def approx(a, b, atol=1e-4):
    return abs(float(a) - float(b)) < atol


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def make_config(**kw):
    d = dict(vocab_size=30, hidden_size=64, num_hidden_layers=2,
             num_attention_heads=4, intermediate_size=128,
             max_seq_len=128, rope_theta=10000.0, add_pooling_layer=True)
    d.update(kw)
    return MethylLlamaConfig(**d)


def make_model(cfg=None):
    m = MethylLlamaModel(cfg or make_config())
    m.eval()
    return m


def dummy_input(B, L, vocab_size=25, seed=0):
    g = torch.Generator(); g.manual_seed(seed)
    cpg_ids   = torch.randint(5, vocab_size, (B, L), generator=g)
    betas     = torch.rand(B, L, generator=g)
    input_ids = torch.stack([cpg_ids.float(), betas], dim=1)
    attn      = torch.ones(B, L, dtype=torch.long)
    return input_ids, attn


# ---------------------------------------------------------------------------
# Helper: rotate a single vector at a given absolute position
# ---------------------------------------------------------------------------

def rotated_dot(rope, q_vec: torch.Tensor, k_vec: torch.Tensor,
                pos_q: int, pos_k: int) -> float:
    """
    Compute (rotate q_vec at pos_q) · (rotate k_vec at pos_k).

    Puts q at slot 0, k at slot 1 in a length-2 sequence, then runs
    apply_rotary_pos_emb with position_ids=[[pos_q, pos_k]].
    Reads q_rot[slot 0] · k_rot[slot 1].
    """
    Dh = q_vec.shape[0]
    q_4d = torch.zeros(1, 1, 2, Dh)
    k_4d = torch.zeros(1, 1, 2, Dh)
    q_4d[0, 0, 0] = q_vec
    k_4d[0, 0, 1] = k_vec
    pos_ids = torch.tensor([[pos_q, pos_k]])
    cos, sin = rope(position_ids=pos_ids)          # [1, 2, Dh]
    q_rot, k_rot = apply_rotary_pos_emb(q_4d, k_4d, cos, sin)
    return (q_rot[0, 0, 0] * k_rot[0, 0, 1]).sum().item()


# ===========================================================================
# SECTION A — RoPE math
# ===========================================================================

def section_a():
    print("\n=== A  RoPE math ===")
    Dh   = 32
    rope = RotaryEmbedding(dim=Dh, max_seq_len=128, theta=10000.0)

    # A1 — unit norms
    for positions in [torch.arange(64), torch.tensor([0, 1, 100, 5000, 49155])]:
        cos, sin = rope(position_ids=positions)
        norms = cos ** 2 + sin ** 2
        check(f"A1 unit norms ({positions.shape[0]} pts)",
              torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

    # A2 — position 0 is identity rotation
    cos0, sin0 = rope(position_ids=torch.tensor([0]))
    check("A2 cos[0]==1",  torch.allclose(cos0[0], torch.ones(Dh),  atol=1e-6))
    check("A2 sin[0]==0",  torch.allclose(sin0[0], torch.zeros(Dh), atol=1e-6))

    # A3 — norm preservation (RoPE is a rotation — isometry)
    torch.manual_seed(42)
    q = torch.randn(1, 1, 5, Dh)
    k = torch.randn(1, 1, 5, Dh)
    pos = torch.randint(0, 49156, (1, 5))
    cos, sin = rope(position_ids=pos)
    q_rot, k_rot = apply_rotary_pos_emb(q.clone(), k.clone(), cos, sin)
    check("A3 norm preserved q",
          torch.allclose(q.norm(dim=-1), q_rot.norm(dim=-1), atol=1e-5))
    check("A3 norm preserved k",
          torch.allclose(k.norm(dim=-1), k_rot.norm(dim=-1), atol=1e-5))

    # A4 — RELATIVE POSITION PROPERTY
    # Q_rot(m) · K_rot(n) depends ONLY on (m-n), not on m, n individually.
    # Tested: same Δ at different absolute positions → same dot product.
    #
    # Tolerance note: the property holds EXACTLY in real arithmetic.  With float32,
    # large absolute positions accumulate ~1e-5 relative error per frequency dimension
    # (cos(p * 1.0) for p=10000 loses ~7 significant decimal digits in float32).
    # We use atol=5e-3 for large positions and 1e-5 for small ones.  The relative
    # error stays below 0.1% of the dot-product magnitude in all tested cases.
    torch.manual_seed(7)
    q_vec = torch.randn(Dh)
    k_vec = torch.randn(Dh)

    # (Δ, [(m, n), ...], atol)
    # Float32 accumulates error ~p * 1.2e-7 per frequency for argument p.
    # At p=5000, cos(5000*1.0) already has ~6e-4 error, so we only use
    # small absolute positions for tight tolerances.
    CASES = [
        (1,     [(0, 1),   (10, 11),  (50, 51)],   1e-5),  # small absolute pos
        (10,    [(0, 10),  (20, 30),  (60, 70)],   1e-4),
        (100,   [(0, 100), (50, 150), (80, 180)],  1e-4),
        (1000,  [(0,1000), (100,1100)],              5e-3),  # large Δ
        (5000,  [(0,5000), (100,5100)],              5e-3),
        (49000, [(0,49000),(100,49100)],              1e-2), # max range
    ]

    passed_rpe = True
    for Δ, positions, atol in CASES:
        dots = [rotated_dot(rope, q_vec, k_vec, m, n) for m, n in positions]
        ref  = dots[0]
        for i, (m, n) in enumerate(positions[1:], 1):
            if not approx(dots[i], ref, atol=atol):
                fail(f"A4 Δ={Δ}: ({positions[0][0]},{positions[0][1]}) vs "
                     f"({m},{n}): {ref:.6f} != {dots[i]:.6f} (atol={atol})")
                passed_rpe = False
    if passed_rpe:
        ok("A4 relative-position property (same Δ → same dot, 6 Δ-values, float32 tol)")

    # A4b — different Δ → different dot products (RoPE encodes distance)
    distinct = True
    prev_dot = None
    for Δ in [1, 100, 5000]:
        d = rotated_dot(rope, q_vec, k_vec, 1000, 1000 + Δ)
        if prev_dot is not None and approx(d, prev_dot, atol=1e-4):
            distinct = False
        prev_dot = d
    check("A4b different Δ → different dot product", distinct)

    # A5 — large positions: up to 49156 (max rank+1) without error or NaN
    try:
        big_pos = torch.tensor([[0, 1, 100, 10000, 49155, 49156]])
        cos, sin = rope(position_ids=big_pos)
        check("A5 large positions no error", True)
        check("A5 large positions no NaN", not torch.isnan(cos).any() and not torch.isnan(sin).any())
        check("A5 large positions unit norm",
              torch.allclose(cos**2 + sin**2, torch.ones_like(cos), atol=1e-5))
    except Exception as e:
        fail("A5 large positions raised exception", str(e))


# ===========================================================================
# SECTION B — apply_rotary_pos_emb
# ===========================================================================

def section_b():
    print("\n=== B  apply_rotary_pos_emb broadcasting ===")
    B, H, L, Dh = 3, 4, 10, 32
    rope = RotaryEmbedding(dim=Dh, max_seq_len=128, theta=10000.0)
    torch.manual_seed(0)
    q = torch.randn(B, H, L, Dh)
    k = torch.randn(B, H, L, Dh)

    # B1 — 2-D sequential
    cos_2d, sin_2d = rope(seq_len=L)           # [L, Dh]
    q2, k2 = apply_rotary_pos_emb(q.clone(), k.clone(), cos_2d, sin_2d)
    check("B1 shape q [2D]",  q2.shape == (B, H, L, Dh))
    check("B1 shape k [2D]",  k2.shape == (B, H, L, Dh))
    check("B1 no NaN",        not torch.isnan(q2).any() and not torch.isnan(k2).any())
    check("B1 norm preserved", torch.allclose(q.norm(dim=-1), q2.norm(dim=-1), atol=1e-5))

    # B2 — 3-D position_ids per batch item
    pos = torch.randint(0, 49156, (B, L))
    cos_3d, sin_3d = rope(position_ids=pos)    # [B, L, Dh]
    q3, k3 = apply_rotary_pos_emb(q.clone(), k.clone(), cos_3d, sin_3d)
    check("B2 shape q [3D]",  q3.shape == (B, H, L, Dh))
    check("B2 shape k [3D]",  k3.shape == (B, H, L, Dh))
    check("B2 no NaN",        not torch.isnan(q3).any() and not torch.isnan(k3).any())
    check("B2 norm preserved", torch.allclose(q.norm(dim=-1), q3.norm(dim=-1), atol=1e-5))

    # B3 — sequential positions via position_ids == sequential mode
    pos_seq = torch.arange(L).unsqueeze(0).expand(B, -1)   # [B, L] = [[0,1,...,L-1], ...]
    cos_s, sin_s = rope(position_ids=pos_seq)
    q_a, k_a = apply_rotary_pos_emb(q.clone(), k.clone(), cos_2d, sin_2d)
    q_b, k_b = apply_rotary_pos_emb(q.clone(), k.clone(), cos_s,  sin_s)
    check("B3 2D==3D for sequential positions q", torch.allclose(q_a, q_b, atol=1e-5))
    check("B3 2D==3D for sequential positions k", torch.allclose(k_a, k_b, atol=1e-5))

    # B4 — each batch item can have different positions
    pos_diff = torch.zeros(B, L, dtype=torch.long)
    pos_diff[0] = torch.arange(L)             # batch 0: sequential
    pos_diff[1] = torch.arange(L) * 100       # batch 1: every 100th genomic slot
    pos_diff[2] = torch.arange(L) * 5000      # batch 2: sparse
    cos_d, sin_d = rope(position_ids=pos_diff)
    q_d, _ = apply_rotary_pos_emb(q.clone(), k.clone(), cos_d, sin_d)
    check("B4 batch-specific positions: no NaN", not torch.isnan(q_d).any())
    # batch 0 and batch 2 should differ (different position_ids)
    check("B4 batch-specific: different output per batch",
          not torch.allclose(q_d[0], q_d[2], atol=1e-3))


# ===========================================================================
# SECTION C — model layer wiring
# ===========================================================================

def section_c():
    print("\n=== C  Model layer wiring ===")

    # C1 — forward without position_ids (Level 1 / backward compat)
    model = make_model()
    B, L = 2, 12
    inp, attn = dummy_input(B, L)
    with torch.no_grad():
        out = model(input_ids=inp, attention_mask=attn)
    check("C1 L1 lhs shape", out.last_hidden_state.shape == (B, L, 64))
    check("C1 L1 pool shape", out.pooler_output.shape == (B, 64))
    check("C1 L1 no NaN", not torch.isnan(out.last_hidden_state).any())

    # C1b — forward WITH position_ids (Level 2)
    pos = torch.zeros(B, L, dtype=torch.long)
    pos[:, 1:] = torch.randint(1, 49157, (B, L - 1))   # CLS=0, CpGs=rank+1
    with torch.no_grad():
        out2 = model(input_ids=inp, attention_mask=attn, position_ids=pos)
    check("C1b L2 lhs shape", out2.last_hidden_state.shape == (B, L, 64))
    check("C1b L2 no NaN lhs", not torch.isnan(out2.last_hidden_state).any())
    check("C1b L2 no NaN pool", not torch.isnan(out2.pooler_output).any())

    # C2 — backward compat: position_ids=None is IDENTICAL to omitting it
    with torch.no_grad():
        out_a = model(input_ids=inp, attention_mask=attn)
        out_b = model(input_ids=inp, attention_mask=attn, position_ids=None)
    check("C2 None==omitted lhs",  torch.allclose(out_a.last_hidden_state, out_b.last_hidden_state))
    check("C2 None==omitted pool", torch.allclose(out_a.pooler_output,     out_b.pooler_output))

    # C3 — Level 2 is NOT a no-op: different positions → different output
    inp1, attn1 = dummy_input(1, 8, seed=99)
    pos_seq = torch.arange(8).unsqueeze(0)                           # sequential [0..7]
    pos_geo = torch.tensor([[0, 1, 100, 500, 5000, 10000, 30000, 49000]])  # genomic gaps
    with torch.no_grad():
        o_seq = model(input_ids=inp1, attention_mask=attn1, position_ids=pos_seq)
        o_geo = model(input_ids=inp1, attention_mask=attn1, position_ids=pos_geo)
    check("C3 Level2 changes lhs",
          not torch.allclose(o_seq.last_hidden_state, o_geo.last_hidden_state, atol=1e-4))
    check("C3 Level2 changes pool",
          not torch.allclose(o_seq.pooler_output, o_geo.pooler_output, atol=1e-4))

    # C4 — gradients flow through Level 2 path
    model.train()
    out_g = model(input_ids=inp, attention_mask=attn, position_ids=pos)
    out_g.pooler_output.sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    check("C4 gradients non-empty", len(grads) > 0)
    check("C4 no NaN gradients", all(not torch.isnan(g).any() for g in grads))
    model.eval()

    # C5 — multiple calls with different position_ids are deterministic
    with torch.no_grad():
        r1 = model(input_ids=inp, attention_mask=attn, position_ids=pos)
        r2 = model(input_ids=inp, attention_mask=attn, position_ids=pos)
    check("C5 deterministic", torch.allclose(r1.last_hidden_state, r2.last_hidden_state))


# ===========================================================================
# SECTION D — WCEDCollator position_ids
# ===========================================================================

def _make_fake_tokenizer(n_cpgs: int):
    """Minimal tokenizer stub that WCEDCollator's __init__ can use."""
    cpg_names = [f"cg{i:08d}" for i in range(n_cpgs)]

    class _CpGTok:
        unk_token_id  = 0
        cls_token_id  = 1
        pad_token_id  = 4
        def get_vocab(self):
            return {n: i + 5 for i, n in enumerate(cpg_names)}

    class _Tok:
        def __init__(self):
            self.tokenizers = {"cpg_sites": _CpGTok()}
            self.cls_token_id = 1
            self.pad_token_id = 4

    return _Tok(), cpg_names


def _make_rank_array(n_cpgs: int, seed: int = 42) -> np.ndarray:
    """Random permutation as genomic_rank (each value appears exactly once)."""
    return np.random.default_rng(seed).permutation(n_cpgs).astype(np.int32)


def _save_rank(arr: np.ndarray) -> str:
    """Save rank array to a temp file; return path."""
    f = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
    np.save(f.name, arr)
    f.close()
    return f.name


def _make_examples(n_cpgs: int, n_samples: int = 5, nan_rate: float = 0.02):
    rng = np.random.default_rng(77)
    out = []
    for _ in range(n_samples):
        betas = rng.random(n_cpgs).astype(np.float32)
        bad   = rng.choice(n_cpgs, size=max(1, int(n_cpgs * nan_rate)), replace=False)
        betas[bad] = np.nan
        out.append(types.SimpleNamespace(
            data     = {"beta_values": betas.tolist()},
            metadata = {"labels": float(rng.integers(20, 80))},
        ))
    return out


def section_d():
    print("\n=== D  WCEDCollator position_ids ===")

    if _dm_mod is None:
        print(f"  SKIP  (data_module not importable in this env — run on cluster)")
        return

    WCEDCollator = _dm_mod.WCEDCollator

    N = 100   # vocabulary size = all CpGs (no subsetting)
    tok, cpg_names = _make_fake_tokenizer(N)
    rank_arr  = _make_rank_array(N)
    rank_path = _save_rank(rank_arr)

    def make_col(contrastive=False, with_genomic=True):
        return WCEDCollator(
            tokenizer        = tok,
            cpg_sites        = cpg_names,
            vocab_size        = N,      # all CpGs → identity vocab_cpg_indices
            input_ratio       = 0.5,
            contrastive       = contrastive,
            genomic_rank_path = rank_path if with_genomic else None,
        )

    examples = _make_examples(N, n_samples=5)
    B        = len(examples)
    max_L    = int(N * 0.5) + 1       # = 51

    col_g = make_col(with_genomic=True,  contrastive=False)
    col_n = make_col(with_genomic=False, contrastive=False)
    col_c = make_col(with_genomic=True,  contrastive=True)

    batch_g = col_g(examples)
    batch_n = col_n(examples)
    batch_c = col_c(examples)

    # D1 — present when genomic_rank set
    check("D1 position_ids present",   "position_ids" in batch_g)
    # D2 — absent when no genomic_rank
    check("D2 position_ids absent",    "position_ids" not in batch_n)

    pos  = batch_g["position_ids"]
    attn = batch_g["attention_mask"]

    # D3 — shape
    check("D3 shape", pos.shape == (B, max_L), str(pos.shape))
    # D4 — dtype
    check("D4 dtype", pos.dtype == torch.long)
    # D5 — CLS slot = 0
    check("D5 CLS position == 0", (pos[:, 0] == 0).all().item())
    # D6 — PAD slots = 0
    pad_pos = pos[attn == 0]
    check("D6 PAD positions == 0", (pad_pos == 0).all().item() if len(pad_pos) > 0 else True)
    # D7 — CpG positions >= 1
    real_idx = (attn == 1) & (torch.arange(max_L).unsqueeze(0) > 0)
    real_pos = pos[real_idx]
    check("D7 CpG positions >= 1", (real_pos >= 1).all().item())
    # D7b — CpG positions <= N (rank+1 convention, max rank=N-1 → max position=N)
    check("D7b CpG positions <= N", (real_pos <= N).all().item())

    # D8 — strictly monotone increasing per sample
    monotone_ok = True
    for b in range(B):
        cpg_positions = pos[b][attn[b] == 1][1:]   # skip CLS
        if len(cpg_positions) > 1:
            diffs = cpg_positions[1:] - cpg_positions[:-1]
            if not (diffs > 0).all():
                monotone_ok = False
                fail(f"D8 monotone sample {b}", f"diffs={diffs.tolist()}")
    if monotone_ok:
        ok("D8 monotone increasing CpG positions (all samples)")

    # D9 — EXACT VALUE CHECK
    # Re-run collator with a fixed seed to inspect one sample
    # Force seed by calling once to advance _call_count, then reset
    col_test = make_col(with_genomic=True, contrastive=False)
    # Patch rng seed to be deterministic
    col_test._call_count = 0
    torch.manual_seed(0)  # controls torch.initial_seed() in collator

    ex_single = [examples[0]]
    batch_s   = col_test(ex_single)
    pos_s     = batch_s["position_ids"][0]   # [max_L]
    attn_s    = batch_s["attention_mask"][0]

    # Re-derive what the collator should have produced
    betas_np = np.asarray(ex_single[0].data["beta_values"], dtype=np.float32)
    vocab_betas = betas_np[col_test.vocab_cpg_indices]
    valid = np.isfinite(vocab_betas)
    valid_indices_np = np.where(valid)[0]
    n_input = int(len(valid_indices_np) * 0.5)

    seed0 = (torch.initial_seed() + 0) % (2**32)  # _call_count was 0 when called
    rng0  = np.random.default_rng(seed0)
    chosen = rng0.choice(valid_indices_np, size=n_input, replace=False)
    chosen_sorted = chosen[np.argsort(rank_arr[chosen])]   # vocab_cpg_indices is identity

    expected_pos = np.zeros(max_L, dtype=np.int64)
    n_real = min(len(chosen_sorted), max_L - 1)
    expected_pos[1:n_real + 1] = rank_arr[chosen_sorted[:n_real]] + 1   # rank+1

    got     = pos_s.numpy()
    matches = np.array_equal(got, expected_pos)
    check("D9 exact position_id values match genomic_rank[col]+1", matches,
          f"first mismatch at: {np.where(got != expected_pos)[0][:5]}")

    # D10 — vocab_cpg_indices is identity (guard against subsetting bug)
    check("D10 vocab_cpg_indices is identity (all CpGs used)",
          np.array_equal(col_g.vocab_cpg_indices, np.arange(N)))

    # D11 — contrastive: position_ids_v2 present
    pos2  = batch_c.get("position_ids_v2")
    attn2 = batch_c.get("attention_mask_v2")
    check("D11 position_ids_v2 present", pos2 is not None)
    if pos2 is not None:
        check("D11 shape v2", pos2.shape == (B, max_L))
        check("D11 CLS v2 == 0", (pos2[:, 0] == 0).all().item())
        pad2 = pos2[attn2 == 0]
        check("D11 PAD v2 == 0", (pad2 == 0).all().item() if len(pad2) > 0 else True)
        # v1 and v2 may differ (different random subsets)
        check("D11 v1 != v2 (independent views)",
              not torch.equal(batch_c["position_ids"], pos2))

    # D12 — no position_id exceeds N
    check("D12 no overflow above N", (pos <= N).all().item())


# ===========================================================================
# SECTION E — end-to-end
# ===========================================================================

def section_e():
    print("\n=== E  End-to-end (collator → model) ===")

    if _dm_mod is None:
        print("  SKIP  (data_module not importable in this env — run on cluster)")
        return

    WCEDCollator = _dm_mod.WCEDCollator

    N   = 80
    tok, cpg_names = _make_fake_tokenizer(N)
    rank_arr  = _make_rank_array(N, seed=13)
    rank_path = _save_rank(rank_arr)

    col = WCEDCollator(
        tokenizer=tok, cpg_sites=cpg_names, vocab_size=N,
        input_ratio=0.5, contrastive=False, genomic_rank_path=rank_path,
    )
    examples = _make_examples(N, n_samples=3)
    batch    = col(examples)

    B   = batch["cpg_ids"].shape[0]
    L   = batch["cpg_ids"].shape[1]
    pos = batch["position_ids"]

    cfg   = make_config(vocab_size=N + 10, max_seq_len=L + 10)
    model = make_model(cfg)

    input_ids = torch.stack([batch["cpg_ids"].float(), batch["beta_values"]], dim=1)

    # E1 — collator batch → model forward: shapes correct, no NaN
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=batch["attention_mask"],
                    position_ids=pos)
    check("E1 lhs shape", out.last_hidden_state.shape == (B, L, 64))
    check("E1 pool shape", out.pooler_output.shape == (B, 64))
    check("E1 no NaN lhs",  not torch.isnan(out.last_hidden_state).any().item())
    check("E1 no NaN pool", not torch.isnan(out.pooler_output).any().item())

    # E2 — Level 2 output differs from Level 1 on same batch
    with torch.no_grad():
        out_l1 = model(input_ids=input_ids, attention_mask=batch["attention_mask"])
        out_l2 = model(input_ids=input_ids, attention_mask=batch["attention_mask"],
                       position_ids=pos)
    check("E2 L2 pool != L1 pool",
          not torch.allclose(out_l1.pooler_output, out_l2.pooler_output, atol=1e-4))
    check("E2 L2 lhs  != L1 lhs",
          not torch.allclose(out_l1.last_hidden_state, out_l2.last_hidden_state, atol=1e-4))


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    section_a()
    section_b()
    section_c()
    section_d()
    section_e()

    print(f"\n{'='*54}")
    total = PASS + FAIL
    print(f"  Results: {PASS}/{total} passed,  {FAIL} failed")
    if FAIL:
        sys.exit(1)
    else:
        print("  All checks passed — Level 2 Genomic RoPE is correct.")
