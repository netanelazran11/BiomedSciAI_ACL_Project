# biomed-multi-omic entrypoint: `bmfm-targets-run`

## scRNA zero shot inference

To get embeddings for an h5ad file from the checkpoints discussed in the manuscript ( <https://arxiv.org/abs/2506.14861> ) run the following code snippets, after installing the package.

The only thing you need is an h5ad file with raw gene counts to run inference, and a writable directory `working_dir` for output. For convenience, this page assumes that the location of the file is stored to an environment variable. Checkpoints will be downloaded automatically from HuggingFace.

```bash
export MY_DATA_FILE=# h5ad file with raw counts and genes identified by gene symbol
```

The program will produce embeddings in `working_dir/embeddings.csv` and predictions in `working_dir/predictions.csv` as csv files indexed with the same `obs` index as the initial h5ad file.

### MLM+RDA

```bash
bmfm-targets-run -cn predict input_file=$MY_DATA_FILE working_dir=/tmp data_module.collation_strategy=language_modeling ++data_module.rda_transform=auto_align data_module.log_normalize_transform=false data_module.max_length=4096 checkpoint=ibm-research/biomed.rna.bert.110m.mlm.rda.v1
```

### MLM+Multitask

```bash
bmfm-targets-run -cn predict input_file=$MY_DATA_FILE working_dir=/tmp data_module.max_length=4096 checkpoint=ibm-research/biomed.rna.bert.110m.mlm.multitask.v1
```

### WCED+Multitask

```bash
bmfm-targets-run -cn predict input_file=$MY_DATA_FILE working_dir=/tmp checkpoint=ibm-research/biomed.rna.bert.110m.wced.multitask.v1
```

### WCED 10 pct

```bash
bmfm-targets-run -cn predict input_file=$MY_DATA_FILE working_dir=/tmp data_module.collation_strategy=language_modeling checkpoint=ibm-research/biomed.rna.bert.110m.wced.v1
```

## scRNA fine-tuning

The requirements are the same as for inference, except that to do fine-tuning you need a split defined, and you need to supply the name of the column that you would like to predict.
You can either add one manually to a column of your choice of the `obs` DataFrame, or you can get a quick random split with this utility:

```python
from os import environ
import scanpy as sc
import bmfm_targets.datasets.datasets_utils as du

ad = sc.read_h5ad(environ["MY_DATA_FILE"])
ad.obs["split_random"] = du.get_random_split(ad.obs, {"train":0.8, "dev": 0.1, "test": 0.1},random_state=42)
ad.write_h5ad(environ["MY_DATA_FILE"])
```

If you use your own split column modify `split_column_name=null` below to your chosen column name, otherwise it will look for a `split_random` column, as created above.

### MLM+RDA

```bash
bmfm-targets-run -cn finetune label_column_name=celltype split_column_name=null input_file=$MY_DATA_FILE working_dir=/tmp ++data_module.rda_transform=auto_align data_module.log_normalize_transform=false data_module.max_length=4096 checkpoint=ibm-research/biomed.rna.bert.110m.mlm.rda.v1
```

### MLM+Multitask

```bash
bmfm-targets-run -cn finetune label_column_name=celltype split_column_name=null input_file=$MY_DATA_FILE working_dir=/tmp data_module.max_length=4096 checkpoint=ibm-research/biomed.rna.bert.110m.mlm.multitask.v1
```

### WCED+Multitask

```bash
bmfm-targets-run  -cn finetune label_column_name=celltype split_column_name=null input_file=$MY_DATA_FILE working_dir=/tmp checkpoint=ibm-research/biomed.rna.bert.110m.wced.multitask.v1
```

### WCED 10 pct

```bash
bmfm-targets-run -cn finetune label_column_name=celltype split_column_name=null input_file=$MY_DATA_FILE working_dir=/tmp checkpoint=ibm-research/biomed.rna.bert.110m.wced.v1
```
