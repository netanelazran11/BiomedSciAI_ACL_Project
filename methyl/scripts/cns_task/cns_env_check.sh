#!/bin/bash
# cns_env_check.sh — read-only inventory for the CNS classification task, both models.
# Safe on the login node: no torch import, no writes outside outputs/cns_task/preflight/.
#   bash scripts/cns_task/cns_env_check.sh 2>&1 | tee outputs/cns_task/preflight/env_check.txt
LAB=/sci/labs/benjamin.yakir/netanel.azran
LL=$LAB/repos/BMFM-RNA/methyl
GPT=$LAB/repos/MethylGPT-Thesis
mkdir -p $LL/outputs/cns_task/preflight

echo "=== 1. Repos: HEAD and local changes"
for r in $LL $GPT $LAB/repos/MethylGPT; do
  echo "--- $r"; git -C $r log --oneline -1 2>&1; git -C $r status -s 2>&1 | head -15
done
git -C $GPT/external/MethylGPT log --oneline -1 2>&1
git -C $GPT/external/MethylGPT diff --stat 2>&1 | tail -3

echo "=== 2. Model inputs"
ls -la $LL/tokenizer_llama_pretrain49k/tokenizers/*/ 2>&1
for v in $(find $LL/tokenizer_llama_pretrain49k -name vocab.txt); do
  echo "$v: $(wc -l < $v) lines; first: $(head -6 $v | tr '\n' ' ')"
done
ls -la $LL/outputs/cpg_genomic_sort/ 2>&1
ls -la $LL/outputs/pretrain-llama-wced/llama-6L-all49k-r0.5-w0.05-genomic-45468861/checkpoints/ 2>&1 | head
ls -la $LAB/MethylGPT/data/cpg_ids/ 2>&1
grep -n "probe_id_dir\|n_hvg\|pad_value\|layer_size\|nlayers" $GPT/scripts/finetuning_age_prediction_medium/args.json 2>&1
ls -la $LAB/repos/MethylGPT/pretrained_models/large/methylGPT-medium/ 2>&1
ls -la $LL/data/manifests/ 2>&1

echo "=== 3. Data already on the cluster"
ls -la $LAB/data/data_methyl_pretrain_type3_h5ad/ 2>&1
find $LAB -maxdepth 4 \( -iname "*GSE90496*" -o -iname "*GSE109379*" -o -iname "*capper*" -o -iname "*cns*" \) 2>/dev/null | head -20

echo "=== 4. Python environments (versions only, no import of torch)"
for env in $LL/bmfm_methyl_env $LAB/venv_torch22; do
  echo "--- $env"
  $env/bin/python - <<'EOF' 2>&1
import importlib.metadata as m, sys
print("python", sys.version.split()[0])
for p in ["torch","lightning","pytorch-lightning","anndata","h5py","numpy","pandas","scikit-learn","torchmetrics","pyarrow","flash-attn","wandb","scipy"]:
    try: print(f"  {p} {m.version(p)}")
    except Exception: print(f"  {p} -")
EOF
done

echo "=== 5. Storage and network"
df -h $LAB 2>&1 | tail -1
lfs quota -h -g $(id -gn) $LAB 2>/dev/null | head -5 || true
du -sh $LAB/data 2>/dev/null
curl -sS -m 20 -o /dev/null -w "GEO ftp over https from $(hostname): HTTP %{http_code}\n" \
  https://ftp.ncbi.nlm.nih.gov/geo/series/GSE109nnn/GSE109379/matrix/ 2>&1
which wget curl 2>&1
