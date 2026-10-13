# Canonical vs legacy files — MethylLlama thesis repo (`methyl/`)

Written 2026-09-14 from the paper's provenance JSONs (`figures/paper/*_provenance.json`),
the run scripts that produced every reported number, and file modification dates.
"Canonical" = the manuscript depends on it. "Legacy" = superseded by a canonical file and
not referenced by the manuscript; safe to move to `deprecated/` (do not delete: the git
history and W&B runs still point at these names). Nothing here has been moved.

## 1. Manuscript

| Path | Role |
|---|---|
| `paper/` | Git-tracked authoring copy (Claude's tree). `main.tex`, `supplementary.tex`, `sections/*.tex`, `figures/` (copies), `references.bib`, `SOURCES.md`, `diff_sources.sh`, `drafts/` (review diffs) |
| `outputs/overleaf/MethylLlama_Publication/` | Local mirror of the live Overleaf (ChatGPT's tree). Sync target only. |
| `outputs/overleaf/MethylLlama_Abstract_Introduction/` | **Legacy** — partial early package. |

## 2. Model code (the only implementation the paper describes)

| Path | Role |
|---|---|
| `bmfm_methylation/llama/model.py` | MethylLlamaModel: ScaleAdapt token encoder, genomic-rank RoPE on Q/K, RMSNorm, SwiGLU, CLS pooler (linear+tanh) |
| `bmfm_methylation/llama/scale_adapt.py` | 48-pair sinusoidal β encoder |
| `bmfm_methylation/llama/wced_llama.py` | WCED Lightning module: two views, decoder, projection head, symmetric InfoNCE, per-sample-standardised reconstruction loss |
| `bmfm_methylation/llama/finetune_llama.py` | Age fine-tuning: 256→128→64→1 head, Huber δ=5 yr/SD, freeze/unfreeze at epoch 10 |
| `bmfm_methylation/llama/pretrain_llama.py` | Pretraining entry point + diagnostics |
| `bmfm_methylation/shared/data_module.py`, `dataset.py`, `tokenizer.py`, `config.py` | WCEDCollator (view sampling, genomic ordering, position ids), dedup exclusion, validity mask |
| `bmfm_methylation/llama/configs/pretrain_llama.yaml`, `finetune_llama.yaml` | Defaults; the run scripts override them |
| `tokenizer/tokenizers/cpg_sites/`, `tokenizer/tokenizers/beta_values/` | 49,161-entry identity vocabulary |
| `bmfm_methylation/wced/`, `bmfm_methylation/mlm/`, `bmfm_methylation/downstream/` | **Legacy** — SCBert-WCED, MLM and multitask/smoking variants; not used by any reported result |

## 3. Run scripts that produced reported numbers

| Path | Produces |
|---|---|
| `scripts/llama/pretrain_llama_small_6L_contrastive.sh` | The pretraining run (job 45468861, checkpoint epoch 85) |
| `scripts/utils/sort_probes_genomic.py` (+ `.sh`) | `cpg_genomic_rank.npy` from the HM450 hg38 manifest |
| `scripts/llama/create_finetune_genomic_rank.py` | `cpg_genomic_rank_finetune.npy` (21,368-CpG ranks; unknown CpGs → column index) |
| `scripts/utils/analyze_full_dataset.py` | `dataset_fingerprint_outputs/duplicate_pairs.csv` (cosine ≥ 0.9999) |
| `scripts/utils/create_kfold_splits.py` | `outputs/kfold_splits/` (StratifiedKFold, age-bin × tissue, seed 42) |
| `scripts/llama/finetune_llama_small_v7b_kfold.sh`, `launch_kfold.sh`, `testeval_kfold.sh` | Five fold-trained models + test evaluation |
| `scripts/llama/finetune_llama_v7b_random_init.sh`, `testeval_random_init.sh` | Fold-0 random-initialisation comparison (Fig. 4c) |
| `scripts/llama/run_kfold_test_predictions.sh` + `scripts/repr_analysis/extract_kfold_test_predictions.py`, `validate_kfold_test_predictions.py` | `outputs/bootstrap_predictions/methyllama/` |
| `scripts/repr_analysis/paired_bootstrap_comparison.py` | `outputs/bootstrap_predictions/paired_*.json/csv` (Table 1, Fig. 4d–e) |
| `scripts/baselines/elasticnet_age.py` (+ `.sh`), `scripts/utils/verify_elasticnet_test_set.*` | `outputs/baselines/elasticnet/elasticnetcv-45888710/` |
| `scripts/repr_analysis_v7b/extract_pretrain_cls.py`, `run_pretrain_cls.sh`, `run_full_analysis.sh` | `figures/v7b_pretrain_cls/{embeddings_cls,embeddings_mean}.npy`, `analysis_summary.json`, `metadata.csv` (Fig. 3c–d, Table S2) |
| `scripts/repr_analysis_v7b/leave_one_dataset_out_probe.py` | `lodo_age_probe.csv`, `lodo_reference.json` (Fig. 3e) |
| `scripts/repr_analysis_v7b/visualize_cls.py` | `cls_umap50_coords.npy` (Fig. 3a–b) |
| `scripts/repr_analysis_v7b/two_view_consistency.py`, `run_two_view_consistency.sh` | `two_view_*.npy`, `two_view_consistency.json` (Fig. 2a) |
| `scripts/repr_analysis_v7b/view_design_eval.py`, `run_view_design_eval.sh` | `figures/v7b_pretrain_cls/view_design/` (Fig. 2b, Fig. S2) |
| `scripts/repr_analysis_v7b/reconstruction_withheld_eval.py`, `run_reconstruction_withheld.sh` | `figures/v7b_pretrain_cls/reconstruction_withheld/` (Fig. 2c–d) |
| `scripts/repr_analysis_v7b/extract_finetune_cls.py`, `run_finetune_cls.sh` | `figures/v7b_finetuned_cls/` (Fig. S4) |
| `scripts/repr_analysis/kfold_full_history_analysis.py` | `kfold_full_history_analysis/*.csv` (Fig. 4c curves) |
| `scripts/llama/run_ablation_rope_step0_subset.sh`, `pretrain_ablation_ropeA_genomic.sh`, `pretrain_ablation_ropeB_nogenomic.sh`; `scripts/repr_analysis_v7b/run_cpg_context_ablation_run{A,B}.sh`, `extract_contextual_cpg.py`, `analyze_cpg_genomic_locality.py`, `analyze_rope_ablation_comparison.py` | Supplementary Note 1 (RoPE pilot) |
| `scripts/paper_figures/*.py` | Every figure in `figures/paper/` — the only figure scripts the manuscript uses |
| `scripts/utils/compare_splits_21k.py`, `verify_21k_comparison.py` | Same-dataset verification between the two pipelines |

**Legacy scripts** (superseded, not referenced): everything else in `scripts/llama/` dated before 2026-07-12 (`finetune_llama_small_v5/v6/v7.sh`, `pretrain_llama_small.sh`, `sweep_arch.sh`, `*_domain*.sh`, `run_figure3/4.sh`, `run_cls_*.sh`, `run_reconstruction_baselines.sh`, `run_attention_analysis.sh`, `run_robustness.sh`, `run_cpg_ablation.sh`, `run_avg_bio.sh`, `run_raw_umap.sh`, `run_age_predictions.sh`, `submit_repr_analysis.sh`, `extract_*_embeddings.sh`, `analyze_cpg_coverage.sh`, `align_cpg_manifest.sh`, `eval_llama_checkpoint.*`); all of `scripts/repr_analysis/` except the five files named above (`reconstruction_baselines.py` in particular is explicitly superseded — wrong checkpoint and objective); `scripts/wced/`, `scripts/mlm/`, `scripts/downstream/`; `scripts/utils/` files dated before 2026-07-01 except the fingerprint/kfold ones above; `scripts/repr_analysis_v7b/attention_analysis_v7b.py`, `run_attention_v7b.sh`, `analyze_cpg_embeddings.py`, `visualize_cpg_context.py`, `visualize_cls_publication.py`, `raw_vs_model_umap.py`, `run_raw_vs_model.sh` (results not in the paper).

## 4. Artifacts

| Path | Status |
|---|---|
| `figures/paper/` | Canonical figures + provenance + captions |
| `figures/v7b_pretrain_cls/`, `figures/v7b_finetuned_cls/` | Canonical analysis artifacts (see above) |
| `outputs/bootstrap_predictions/`, `outputs/baselines/`, `outputs/wandb_curves/`, `outputs/ablation_rope/` | Canonical |
| `kfold_full_history_analysis/` | Canonical (Fig. 4c) |
| `dataset_fingerprint_outputs/` | Canonical (duplicates; `duplicate_pairs.csv` itself lives on the cluster) |
| `data/manifests/HM450.hg38.manifest.tsv` | Canonical input (164 MB, untracked — add to `.gitignore`) |
| `figures/v7b_attention/`, `figures/v7b_cpg_context/`, `figures/mockups/` | Not in the paper; keep for now |
| `figures/{smoke,arch_sweep,attention,attention_v5,cls_probing,robustness_v5,cpg_ablation,figure3,figure4,reconstruction_baselines,cls_attention}/` | **Legacy** — earlier model versions (V4/V5) |
| `kfold_paper_figure/`, `kfold_presentation_figures/`, `wandb_analysis/`, `wandb_run_comparison_*`, `outputs/comparisons/`, `tests/`, `tmp/` | **Legacy** — presentation-era outputs |

## 5. Documentation

| Path | Status |
|---|---|
| `docs/CANONICAL_FILES.md` (this file), `paper/SOURCES.md` | Current |
| `docs/presentations/MethylLlama_vs_MethylGPT_21k.html` | Latest deck (results now superseded by `paper/` for numbers) |
| `docs/presentations/WCED_ablation_{methods,results,discussion}.md` | Draft text later merged into `paper/` |
| `README.md` | **Stale** — describes a 6-layer/8-head/512-d mean-pooling model, epoch-190 checkpoint, 8k–19k CpGs. Must be rewritten from `paper/sections/04_methods.tex`. |
| Everything else under `docs/` dated ≤ 2026-07-14 | **Legacy** — V4/V5/19k-era presentations and reports |
| `deprecated/` | Already archived 21k-pipeline code |
