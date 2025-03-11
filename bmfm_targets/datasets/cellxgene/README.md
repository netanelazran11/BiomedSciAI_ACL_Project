### CellXGene dataset

The cellXgene dataset contains over 33M samples from different tissues, suspension types, developmental stages and disease states.
For details on the dataset and the metadata it holds, visit: https://github.com/chanzuckerberg/single-cell-curation/blob/main/schema/3.0.0/schema.md#cell_type_ontology_term_id.
In the [review_cellxgene_metadata.ipynb](bmfm_targets/datasets/cellxgene/review_cellxgene_metadata.ipynb) notebook there is a detailed print of the whole dataset metadata and the unique values it holds in each field.

The cellXgene dataset was partitioned to 167 files (166 out of them contains 200K samples) and is saved on CCC at:
[/dccstor/bmfm-targets/data/omics/transcriptome/scRNA/pretrain/cellxgene/anndata_all/all_unique_cells/](/dccstor/bmfm-targets/data/omics/transcriptome/scRNA/pretrain/cellxgene/anndata_all/all_unique_cells/)

In order to use this dataset for training/prediction, the steps are:

1. First, the user need to select how many partition files he want to use.
Then, make a copy of a concatenated file to the h5ad subfolder by running the preprocessing script:

    - [bmfm_targets/datasets/cellxgene/process_cellxgene_partition_files.py](./process_cellxgene_partition_files.py)
    and adjusting the `n_files` parameter at `line 82`.
    Preprocessing includes editing the obs object indices (it shouldn't include only numbers) and fix gene names from number coding to their names.
    - In this script the user also need to decide on filter terms to apply on the cellxgene dataset. The default is to filter all records of 'suspension_type' equals 'cell' (leaving out all nucleus records).

2. Given the h5ad processed file, the user should create the cellXgene cell type labels dictionary by running the following file:
[bmfm_targets/datasets/cellxgene/create_cellxgene_labels_json_file.py](bmfm_targets/datasets/cellxgene/create_cellxgene_labels_json_file.py).
The cell type labels dictionary can also be generated on the fly by setting in [scbert_train_cellxgene.yaml](bmfm_targets/tasks/scbert/scbert_train_cellxgene.yaml) file the following data_module arg:
```
data_module:
  dataset_kwargs:
    label_columns: ["cell_type_ontology_term_id"]
```
