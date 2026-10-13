# Public release plan — MethylLlama code, checkpoints and reproduction bundle

Written 2026-10-13 from the state of four locations: this working tree, `origin/main` of
`netanelazran11/BiomedSciAI_ACL_Project` (= the cluster clone that ran every job), the cluster-only
artifacts, and the public repo `netanelazran11/MethylLlama`. Nothing has been moved, committed or
pushed while writing it.

## 1. Where things are today

| Location | What it holds | Problem |
|---|---|---|
| This tree, `BMFM-RNA_thesis/methyl` (branch `paper-source-of-truth`) | The most complete copy: model code, every canonical script, the manuscript | 29 canonical scripts are **untracked** (exist only on this laptop), incl. 6 figure scripts the manuscript uses. Branch is 11 commits ahead and 10 behind `origin/main` |
| `origin/main` of the fork = cluster `repos/BMFM-RNA/methyl` | The code that actually ran (pretraining 45468861, folds 45586010–14, random init 45683236, held-out evaluations) | Diverged from the local branch; some figure scripts differ from the local copies |
| Cluster only | Canonical tokenizer `tokenizer_llama_pretrain49k` (49,161), `cpg_genomic_rank.npy`, `cpg_genomic_rank_finetune.npy`, all checkpoints, the two h5ad files, run logs and `hparams.yaml`, the Python environment | Not backed up anywhere public. The local `methyl/tokenizer/` is the **legacy 8,005-entry** vocabulary |
| Public `MethylLlama` | Remote last commit 2026-07-01 (old 4-layer model). Local working copy holds an uncommitted 2026-09-15 sync | The sync is itself out of date (no held-out evaluation, audit, or new figure scripts) |

Facts that shape the plan:

- **Model code has no hard-coded paths** (`bmfm_methylation/llama`, `shared`: 0 of 15 files). Scripts do:
  `scripts/llama` 45 of 46, `scripts/utils` 73 of 94, `scripts/repr_analysis_v7b` 13 of 32.
- **`bmfm-targets` is not on PyPI** (HTTP 404). The code imports `MultiFieldTokenizer`, `FieldInfo`,
  `SCBertConfig` (in `shared/config.py`). It is Apache-2.0 (`BiomedSciAI/biomed-multi-omic`).
- A pattern scan for API keys and tokens found nothing; a full scanner must still run before release.
- Internal codenames (`v7b`, `MethylLlamaV7b`) appear in folder names and prediction files.
- `paper/sections/05_data_availability.tex` and `06_code_availability.tex` are empty skeletons.

## 2. Decision: export from a manifest, do not copy folders or filter history

Three ways to build the public repo:

| Approach | Verdict |
|---|---|
| Copy `methyl/` or `bmfm_methylation/` wholesale | No. Ships legacy code (SCBert-WCED, MLM, V5/V6 scripts), the wrong tokenizer and cluster paths |
| `git filter-repo` of `methyl/` from the fork | No. Carries legacy files, large outputs and old history that has never been audited for secrets |
| **Allowlist manifest + export script into a release branch** | **Yes.** Every published file is named, its source is recorded, and each copy can be checked against the version that ran |

The public repo keeps its history: tag the current `main` as `v0-legacy`, build the release on a
branch `release/v1`, merge when every gate in section 5 passes, tag `v1.0.0`.

## 3. Target layout

```
MethylLlama/
  README.md            model summary, install, quickstart, reproduce-the-paper, data/checkpoint access, citation
  LICENSE              Apache-2.0
  NOTICE               attribution: WCED adapted from BMFM-RNA (biomed-multi-omic, Apache-2.0)
  CITATION.cff
  pyproject.toml       pinned dependencies; bmfm-targets pinned to an upstream git commit
  requirements-lock.txt  exact versions, from `pip freeze` of the cluster env
  bmfm_methylation/    package (name kept for checkpoint compatibility in v1)
    llama/  model.py scale_adapt.py wced_llama.py finetune_llama.py pretrain_llama.py configs/
    shared/ data_module.py dataset.py tokenizer.py config.py
  tokenizer/           the 49,161-entry vocabulary used by run 45468861
  configs/paths.example.yaml   DATA_DIR, OUT_DIR, CKPT_DIR … (scripts read these, never absolute paths)
  scripts/
    data/        genomic rank, fine-tune rank, duplicate detection, k-fold splits
    pretrain/    pretraining (SLURM template)
    finetune/    k-fold fine-tuning, test evaluation, test predictions, random initialisation
    baselines/   ElasticNetCV; MethylGPT comparison wrapper + instructions
    evaluation/  reconstruction, cross-view (held-out), frozen probes, LODO, paired bootstrap, RoPE pilot
    figures/     only the figure scripts the final manuscript uses
    slurm/       cluster templates with placeholders
  results/       small derived files that regenerate every figure and table without a GPU (CSV/JSON)
  tutorials/     quickstart, embeddings, age prediction (6 layers, FFN 512, genomic position ids)
  tests/         CPU smoke tests + checkpoint-reproduction tests
```

Outside git:

| Where | What |
|---|---|
| Hugging Face model repo | epoch-85 pretrained checkpoint, five fold fine-tuned checkpoints (and random-init fold 0 if reported), tokenizer, both genomic-rank files, model card |
| Zenodo (code) | automatic DOI for the `v1.0.0` GitHub release |
| Zenodo (reproduction bundle) | per-profile predictions, embeddings, fold and test ids, duplicate list, cross-view per-run outputs (~250 MB) |
| Not redistributed | the pretraining matrix (MethylGPT / EWAS Data Hub / ClockBase) and the AltuMAge matrix: download instructions + preprocessing scripts only, after their terms are checked |

## 4. Phases

**P0 — Secure what exists (before anything else).** Commit the 29 untracked canonical files in this
tree; merge `origin/main` into the thesis branch so one branch holds everything that ran plus the
paper. Needs the user's go-ahead (commit/push).

**P1 — Freeze the inventory (after the Codex redline).** Update `CANONICAL_FILES.md` to the final
manuscript (which figure scripts, which analyses: held-out cross-view, audit scripts, InfoNCE
ablation and CNS task only if they are in the submitted paper). Turn it into `release/MANIFEST.tsv`:
`source_path  destination_path  role  produces`.

**P2 — Collect cluster-only artifacts.** Tokenizer, rank files, checkpoints, `hparams.yaml` and logs of
the six runs, `pip freeze` of `bmfm_methyl_env`, Python/CUDA/torch versions, W&B run ids. Exact commands
to be prepared; the user runs them.

**P3 — Export.** A script reads the manifest, copies each file into a clean clone of `MethylLlama` on
`release/v1`, and writes `release/provenance.json` (source path, source commit, sha256).

**P4 — Parameterise scripts.** In the exported copies only, replace absolute paths with values from
`configs/paths.yaml` / environment variables. A per-file diff must show that only path lines changed.
Rename user-facing `v7b` names (mapping table kept in `release/`).

**P5 — Dependencies.** Pin `bmfm-targets @ git+https://github.com/BiomedSciAI/biomed-multi-omic@<commit>`
(the commit matching 0.15.0), plus the locked versions from P2. Decide whether `shared/config.py`
(SCBertConfig) is still needed by the canonical path; drop it only if the tests in P7 still pass.

**P6 — Artifacts and tutorials.** Upload to Hugging Face and Zenodo (user's accounts). Update the
three notebooks: 6 layers, FFN 512, new checkpoint filename, and `position_ids` from the genomic-rank
file (without them the sequential fallback runs and the representation is not the manuscript's).

**P7 — Verification gates** (section 5).

**P8 — Release.** Merge, tag `v1.0.0`, mint the DOIs, then fill `05_data_availability.tex` and
`06_code_availability.tex` with the URLs and DOIs.

## 5. Gates — the release is not published until every one passes

| Gate | Check |
|---|---|
| G1 Code identity | Every exported model/data file is byte-identical to its `origin/main` version, or its diff is path-only and recorded |
| G2 Clean install | Fresh environment from `pyproject.toml` + lock installs and passes the CPU tests |
| G3 Pretrained checkpoint | Released checkpoint + tokenizer + rank file reproduce the stored CLS embeddings (`embeddings_cls.npy`) for the demo profiles within float tolerance |
| G4 Fine-tuned checkpoints | Each fold checkpoint reproduces its stored test predictions for the demo profiles |
| G5 Figures | Every manuscript figure regenerates from `results/` using only the public repo; provenance numbers match |
| G6 Table 1 | The paired bootstrap on the released predictions reproduces Table 1 exactly |
| G7 Hygiene | Secret scanner (e.g. gitleaks) clean; no `/sci/labs` or home paths; no internal codenames in user-facing names |
| G8 Licences | Apache-2.0 + NOTICE; third-party data and MethylGPT terms checked; nothing redistributed without permission |

## 6. Decisions for the user

| # | Question | Recommendation |
|---|---|---|
| D1 | Package name | Keep `bmfm_methylation` in v1 (checkpoints and loaders reference it); a `methyllama` alias can come later |
| D2 | Scope | The repo contains what the submitted paper reports, plus tutorials. InfoNCE ablation and CNS task only if they enter the paper |
| D3 | MethylGPT comparator | Publish our wrapper/conversion scripts and instructions; include none of MethylGPT's code unless its licence allows |
| D4 | When public | With the preprint. For journal review before that, a private repo or reviewer link |
| D5 | History of the public repo | Keep it; tag `v0-legacy`; release on `release/v1` |
