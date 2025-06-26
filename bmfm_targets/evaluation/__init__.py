"""Evaluation utilities for trained models."""

from .embeddings import (
    evaluate_clusters,
    generate_clusters,
    load_prediction_data_to_anndata,
    calculate_metrics_embedding,
)
from .clearml_utils import get_folder_of_checkpoints, get_model_path, get_root_folder
from .utils import (
    check_gpu,
    convert_ids_to_label_names,
    create_results_obs,
    get_general_cell_type,
    get_label_dict,
    get_label_map,
    merge_bmfm_adata,
    plot_cell_type_counts,
    plot_logits,
    plot_embeddings,
)


__all__ = [
    "calculate_metrics_embedding",
    "check_gpu",
    "convert_ids_to_label_names",
    "create_results_obs",
    "evaluate_clusters",
    "generate_clusters",
    "get_folder_of_checkpoints",
    "get_general_cell_type",
    "get_label_dict",
    "get_label_map",
    "get_model_path",
    "get_root_folder",
    "load_prediction_data_to_anndata",
    "merge_bmfm_adata",
    "plot_cell_type_counts",
    "plot_logits",
    "plot_embeddings",
]
