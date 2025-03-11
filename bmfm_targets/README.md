# bmfm_targets

Package to pretrain foundational AI models using single-cell data or genomic sequences

## Source code and issue tracker

Available on Github Enterprise, [BiomedSciAI-Innersource/
bmfm-targets](https://github.ibm.com/BiomedSciAI-Innersource/bmfm-targets).
Please report bugs, issues and feature extensions there.

## Training

After installing the package run

```bash
bmfm-targets-scbert -cn scbert_train_panglaodb # to use panglaodb
bmfm-targets-scbert -cn scbert_train_zheng68k # to use zheng68k in mlm mode
bmfm-targets-scbert -cn scbert_train_humancellatlas # to use human cell atlas in mlm mode
bmfm-targets-scbert -cn scbert_train_cellxgene # to use cellXgene in mlm mode
bmfm-targets-scbert -cn scbert_nystromformer_train_panglaodb  # to use nystromformer model with panglaodb
bmfm-targets-scbert -cn scbert_train_streaming_panglaodb # to use scbert with streamiing panglaodb data
bmfm-targets-scbert -cn scbert_nystromformer_streaming_snpdb # to use nystromformer model with streaming snp genomic sequences
```

For finetuning you will probably want to select a checkpoint and provide it to the `model.checkpoint` field of the
config and then set `mlm` to false in the data module:

```bash
bmfm-targets-scbert -cn scbert_train_zheng68k data_module.mlm=false # to use zheng68k in fine-tune mode
```

or to use the T5 model

```bash
bmfm-targets-t5
```

to execute pretraining with the scBERT or T5 models on the panglaodb datasets respectively.
See [tasks](./tasks/) for more details.

## Downstream tasks

Currently cell-type annotation on pretrained models is implemented. To execute it please run

```bash
bmfm-targets-scbert -cn scbert_predict_zheng68k
```

This task is only defined on existing checkpoints.
Make sure you have updated paths to pretrained models in your config.

## Running dna seq finetuning tasks

Currently there are 10 finetuning data that we are running on our pre-trained dnaseq model.


## Configuration

The configuration is structured into different sections to define various aspects of the experiment.

By default the app makes use of the following environment variables. Below are their settings for CCC.
To use the scripts either copy these lines into your bashrc or override them in your config yaml.

```bash
export BMFM_TARGETS_PANGLAO_DATA=/dccstor/bmfm-targets/data/omics/transcriptome/scRNA/pretrain/panglao/
export BMFM_TARGETS_PANGLAO_METADATA=/dccstor/bmfm-targets/data/omics/transcriptome/scRNA/pretrain/panglao/metadata/metadata.csv
export BMFM_TARGETS_ZHENG68K_DATA=/dccstor/bmfm-targets/data/omics/transcriptome/scRNA/finetune/Zheng68K/
export BMFM_TARGETS_HUMANCELLATLAS_DATA=/dccstor/bmfm-targets/data/omics/transcriptome/scRNA/finetune/HumanCellAtlas/
export BMFM_TARGETS_CELLXGENE_DATA=/dccstor/bmfm-targets/data/omics/transcriptome/scRNA/pretrain/cellxgene/
export BMFM_TARGETS_SCIPLEX3_DATA=/dccstor/bmfm-targets/data/omics/transcriptome/scRNA/finetune/SciPlex3/
export BMFM_TARGETS_PANCREAS_CROSS_DATA=/dccstor/bmfm-targets/data/omics/transcriptome/scRNA/finetune/PancreasCross/
export BMFM_TARGETS_HBONES_DATA=/dccstor/bmfm-targets/data/omics/transcriptome/scRNA/finetune/hBones/
export BMFM_TARGETS_TCGA_KIRC_data=/dccstor/bmfm-targets/data/omics/transcriptome/bulkRNA/finetune/TCGA-KIRC/

export BMFM_TARGETS_DNASEQ_LENTI_MPRA_DATA=/dccstor/bmfm-targets1/data/omics/genome/finetune_datasets/lenti_mpra_regression
export BMFM_TARGETS_DNASEQ_CHROMATIN_PROFILE_DATA=/dccstor/bmfm-targets1/data/omics/genome/finetune_datasets/chromatin_profile_prediction
export BMFM_TARGETS_DNASEQ_CORE_PROMOTER_DATA=/dccstor/bmfm-targets1/data/omics/genome/finetune_datasets/core_promoter_prediction
export BMFM_TARGETS_DNASEQ_COVID_PREDICTION_DATA=/dccstor/bmfm-targets1/data/omics/genome/finetune_datasets/covid_prediction
export BMFM_TARGETS_DNASEQ_DROSOPHILA_ENHANCER_DATA=/dccstor/bmfm-targets1/data/omics/genome/finetune_datasets/drosophila_enhancer_prediction
export BMFM_TARGETS_DNASEQ_EPIGENETIC_MARK_PREDICTION_DATA=/dccstor/bmfm-targets1/data/omics/genome/finetune_datasets/epigenetic_marks_prediction
export BMFM_TARGETS_DNASEQ_PROMOTER_DATA=/dccstor/bmfm-targets1/data/omics/genome/finetune_datasets/promoter_prediction
export BMFM_TARGETS_DNASEQ_SNV_MPRA_CAGI_DATA=/dccstor/bmfm-targets1/data/omics/genome/finetune_datasets/snv_mpra_cagi_regression
export BMFM_TARGETS_DNASEQ_SPLICE_SITE_DATA=/dccstor/bmfm-targets1/data/omics/genome/finetune_datasets/splice_site_prediction
export BMFM_TARGETS_DNASEQ_TF_PREDICTION_DATA=/dccstor/bmfm-targets1/data/omics/genome/finetune_datasets/tf_prediction


export BMFM_TARGETS_SCIBD_DATA=/dccstor/bmfm-targets/data/omics/transcriptome/scRNA/finetune/scIBD/
export BMFM_TARGETS_SCIBD300K_DATA=/dccstor/bmfm-targets/data/omics/transcriptome/scRNA/finetune/scIBD300K/
export BMFM_TARGETS_TIL_DATA=/dccstor/bmfm-targets/data/omics/transcriptome/scRNA/finetune/TIL/
export BMFM_TARGETS_SCP1884_DATA=/dccstor/bmfm-targets/data/omics/transcriptome/scRNA/finetune/SCP1884/
export BMFM_TARGETS_ADAMSON_DATA=/dccstor/bmfm-targets/data/omics/transcriptome/scRNA/finetune/Perturbation/scPerturb/
export BMFM_TARGETS_ENSEMBL_DATA=/dccstor/bmfm-targets/data/omics/proteins/ensembl-release-112/data/
export BMFM_TARGETS_DNA_DATA_REF_1KB=/dccstor/bmfm-targets/data/omics/genome/snpdb/litdata/ref_1kb/
export BMFM_TARGETS_DNA_DATA_REF_RC_1KB=/dccstor/bmfm-targets/data/omics/genome/snpdb/litdata/ref_rc_1kb/
export BMFM_TARGETS_DNA_DATA_REF_RC_1KB_10KB=/dccstor/bmfm-targets/data/omics/genome/snpdb/litdata/ref_rc_1kb_10kb/
export BMFM_TARGETS_DNA_DATA_REF_RC_1KB_10KB_10X=/dccstor/bmfm-targets/data/omics/genome/snpdb/litdata/ref_rc_1kb_10kb_10x/

export BMFM_TARGETS_DNA_DATA_SNP_1KB=/dccstor/bmfm-targets/data/omics/genome/snpdb/litdata/biallele_1kb/
export BMFM_TARGETS_DNA_DATA_SNP_1KB_10KB=/dccstor/bmfm-targets/data/omics/genome/snpdb/litdata/biallele_1kb_10kb/
export BMFM_TARGETS_DNA_DATA_SNP_RC_1KB_10KB_10X=/dccstor/bmfm-targets/data/omics/genome/snpdb/litdata/biallele_rc_1kb_10kb_10x/

export BMFM_TARGETS_TRAINING_DIR=/dccstor/bmfm-targets/users/$USER/training_runs
```

### Seed

The seed value used to ensure reproducibility across runs of the experiment.

```yaml
seed:
  seed_value: 1234
```

### Fields

The settings shared between data module and module. In order for a loss to be valid for a field in language_modeling/multitask mode, the field must have `is_masked=True`. For sequence labeling, the field with the loss should have `is_input=False`.

```yaml
fields:
  - _target_: bmfm_targets.config.FieldInfo
    field_name: "genes"
    pretrained_embedding: null
    is_masked: false
  - _target_: bmfm_targets.config.FieldInfo
    field_name: "expressions"
    pretrained_embedding: null
    is_masked: true
```

### Data Module

The data module settings selects which data to use and also what kind of task
will be executed. The dataset can refer to one of the packaged datasets or a generic h5ad file based dataset.

By selecting a `collation_strategy`, the same dataset can be used for different tasks.
The recommended option is `"multitask"` which supports masked language modeling and sequence classification.
To perform just MLM `"language_modeling"` can be used and `"sequence_classification"` for just sequence classification.
In "multitask" mode, any combination of field and label_column losses is supported.

The `"sequence_labeling"` option is only supported for datasets with paired "control" perturbed cells.

Full documentation can be found in [data_module.py](./training/data_module.py)

#### Package datasets

```yaml
data_module:
  _target_: bmfm_targets.datasets.zheng68k.Zheng68kDataModule
  _partial_: true
  num_workers: 8
  collation_strategy: "multitask"
  batch_size: 20
  max_length: 512
  pad_to_multiple_of: 16
  change_ratio: 0.15
  mask_ratio: 0.5
  switch_ratio: 0.5
  limit_dataset_samples: null
  shuffle: true
  sequence_order: "random"
  mlm: True
  data_dir: ${oc.env:BMFM_TARGETS_ZHENG68K_DATA}
```


#### Panglao data module
The Panglao data module behaves differently than the others because it is a dataset of datasets and has transforms defined within the dataset object:

```yaml
data_module:
  _target_: bmfm_targets.datasets.panglaodb.PanglaoDBDataModule
  _partial_: true
  num_workers: 8
  mlm: true
  collation_strategy: "language_modeling"
  batch_size: 2
  transform_datasets: false
  dataset_kwargs:
    data_dir: ${oc.env:BMFM_TARGETS_PANGLAO_DATA}
    data_info_path: ${oc.env:BMFM_TARGETS_PANGLAO_METADATA}
    filter_query: 'Species == "Homo sapiens"'
    num_workers: 8
    convert_rdata_to_h5ad: false
```

#### Dataset without dedicated data module

The package has many datasets that have been processed and have default settings.
If you want to use a new dataset you can either add it in a PR to the package or access
it directly as an h5ad file:

```yaml
data_module:
  _target_: bmfm_targets.training.data_module.DataModule
  _partial_: true
  num_workers: 8
  collation_strategy: "sequence_classification"
  batch_size: 2
  transform_datasets: true
  num_workers: 8
  transform_kwargs:
    source_h5ad_file_name: /path/to/raw_data.h5ad # you must supply this
    processed_data_source: /path/to/processed.h5ad # will be created if transform_datasets=True
    stratifying_label: "celltype"
  dataset_kwargs:
    processed_data_source: /path/to/processed.h5ad # must be identical to transform_kwargs.processed_data_source
    label_columns: ["celltype"]
    label_dict_path: /path/to/label_dict.json #will be created if absent, should be persistent for a given dataset
```

After running once with `transform_datasets=True`, future runs can run with `transform_datasets=False` and omit `transform_kwargs`.

### Model

Configuration options for the model.
Most of the parameters are standard HuggingFace Transformers settings.
Note that passing a checkpoint in the model config is necessary if beginning a training job based on a pretrained MLM checkpoint, or otherwise constructing a new model on top of another model. For test or predict tasks, this is not necessary or desirable.

```yaml
model:
  _target_: bmfm_targets.config.SCBertConfig
  _partial_: true
  num_hidden_layers: 12
  num_attention_heads: 12
  intermediate_size: 3072
  hidden_act: "gelu"
  hidden_size: 768
  hidden_dropout_prob: 0.1
  attention_probs_dropout_prob: 0.1
  initializer_range: 0.02
  layer_norm_eps: 1e-12
  pad_token_id: 0
  use_cache: true
  classifier_dropout: 0.1
  attention: torch
  checkpoint: null
```

### Trainer
Multiple losses can be used simultaneously with different weights. If there are masked fields in the `fields` section, or if the task is `sequence_labeling` then field losses (identified by `field_name`) can be used.
If the dataset has labels and the `label_columns` section is set, then `label_column` losses can be used.

For futher details see [training_config.py](./config/training_config.py) and [loss_handling.py](./training/metrics/loss_handling.py)

```yaml
trainer:
  _target_: bmfm_targets.config.TrainerConfig
  warmup_steps: 1000
  weight_decay: 0.01
  losses:
    - label_column_name: celltype
      name: focal
    - field_name: expressions
      name: mse
      ignore_zero: true
      # link_function: exp
    - field_name: expressions
      name: is_zero_bce
    - field_name: genes
      name: cross_entropy
```

### Task

Task-specific parameters for training the model:

```yaml
task:
  _target_: bmfm_targets.config.TrainingTaskConfig
  default_root_dir: ${oc.env:BMFM_TARGETS_TRAINING_DIR}
  max_epochs: 20
  max_steps: 1000000
  precision: "16-mixed"
  val_check_interval: 0.1
  accelerator: "gpu"
  accumulate_grad_batches: 4

```

Task-specific parameters for testing the model:

```yaml
task:
  _target_: bmfm_targets.config.TestTaskConfig
  default_root_dir: ${oc.env:BMFM_TARGETS_TRAINING_DIR}
  precision: "16-mixed"
  accelerator: "gpu"
  tf32_mode: "medium" # permitted nonNone values of tf32_mode:  "highest","high","medium"
  checkpoint: path/to/trained/model/checkpoint.ckpt
  num_bootstrap_runs: 0
  ci_method: binomial

```

During testing, three types of CI for the evaluation metrics are available thorugh the ci_method and num_bootstrap_runs arguments.
    - bootstrap_quantiles: takes the 2.5% and 97.5% quantiles of the bootstrap sample.
    - bootstrap_t_interval: based on the bootstrap sample distribution around the sample's mean.
        If len(data)=1 returns None values for the CI's bounds.
    - binomial: CI based on the binomial distribution for proportion metrics. Does not require
        repeated samplings from the data, thus num_bootstrap_runs can be set to 0. In case num_bootstrap_runs >=1
        The binomial CI uses the metric's average value across the bootstrap samples as the estimate.

### Running `scbert.py` on CCC

```jbsub -mem 32g -cores 1+1 -q x86_6h -require a100 bmfm-targets-scbert -cn scbert_train_panglaodb```

Note that the environment variables (as detailed in the Configuration section above) need to be exported within the user .bashrc file (located in the CCC user root directory `/u/{user_name}/.bashrc`).

### Running `scbert_train.py` on CCC with SessionManager

[SessionManager](https://github.ibm.com/BiomedSciAI-Innersource/bmfm-core/blob/main/bmfm_core/infra/session_manager/README.md) allows running a task in a clean "container" like environment while resubmitting it to queue when needed, overcoming the 24h queue limitation.
See [scbert_train_session_manager](../session_manager/scbert_train_session_manager.yaml) for running instructions.


## Working with olded checkpoints

The code for the MultiFieldTokenizer has been refactored and checkpoints that where saved prior to 15th July 2024 will need to convert the tokenizer to the new format.  To do this, use the
`bmfm-targets/bmfm_targets/tokenization/convert_multifiled_tokenizer.py`
script.  Give the script the path to the old tokenizer and either a path to save the new one, or use `--write-back` to save it back to the input directory.  Note: saving back does not override old tokenizers' files.
