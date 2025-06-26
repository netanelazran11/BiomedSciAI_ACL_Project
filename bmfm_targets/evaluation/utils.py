from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import requests
import scanpy as sc
import seaborn as sns
import torch
from anndata import AnnData
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans


def _get_ibm_color_palette():
    import zipfile

    import swatch
    from matplotlib.colors import ListedColormap

    url = "https://github.com/carbon-design-system/carbon/raw/refs/heads/main/packages/colors/artifacts/IBM_Colors.zip"
    ibm_pal_path = Path(__file__).parent / "IBM_Colors.zip"

    if not ibm_pal_path.exists():
        response = requests.get(url)
        response.raise_for_status()
        if response.status_code == 200:
            with open(ibm_pal_path, "wb") as f:
                f.write(response.content)
        else:
            raise (f"Failed to download ZIP: {response.status_code}")

        with zipfile.ZipFile(ibm_pal_path, "r") as zip_ref:
            zip_ref.extractall(Path(__file__))

    ase_file = ibm_pal_path.parent / ibm_pal_path.stem / "IBM_Colors_RGB_HEX_v2.1.ase"
    ibm_color_map = swatch.parse(ase_file)
    ibm_color_map = [
        s["data"]["values"]
        for c in ibm_color_map
        for s in c["swatches"]
        if len(s["name"].split("_")) > 1
        and 40 < int(s["name"].split("_")[-2])
        and 90 > int(s["name"].split("_")[-2])
        and "Gray" not in s["name"]
    ]

    ibm_color_map = ListedColormap(ibm_color_map, name="ibm_carbon")
    mpl.colormaps.register(name="ibm_carbon", cmap=ibm_color_map)

    return ibm_color_map


def plot_logits(
    cell_name: str,
    label: str,
    results: dict,
    label_dict: dict,
    adata: AnnData | None = None,
):
    import numpy as np

    logits = dict(zip(results["cell_names"], results[f"{label}_logits"]))
    logits = pd.DataFrame(logits[cell_name]).rename(
        index={v: k for k, v in label_dict[label].items()},
        columns={0: f"{label}_logits"},
    )
    logits = logits.reset_index().rename(columns={"index": label})
    logits = logits.sort_values(by="cell_type_logits", ascending=False).head(10)
    color = mpl.colormaps["ibm_carbon"].colors[13]

    if adata:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        adata = adata.copy()
        adata.obs["highlight"] = adata.obs_names == cell_name
        idx = np.where(adata.obs_names == cell_name)[0][0]
        other_indices = np.setdiff1d(np.arange(adata.n_obs), idx)
        new_order = np.concatenate([other_indices, [idx]])
        adata = adata[new_order, :]
        sc.pl.umap(
            adata,
            color="highlight",
            palette=["lightgray", color],
            title=cell_name,
            sort_order="highlight",
            show=False,
            ax=axes[0],
        )

        logits = sns.barplot(
            data=logits, x="cell_type_logits", y="cell_type", color=color, ax=axes[1]
        )
    else:
        sns.barplot(data=logits, x="cell_type_logits", y="cell_type", color=color)

    plt.tight_layout()
    plt.show()


def _create_general_cell_type_mapping(label_dict: dict, level: str = "sub_class"):
    from cellxgene_ontology_guide.curated_ontology_term_lists import (
        get_curated_ontology_term_list,
    )
    from cellxgene_ontology_guide.entities import CuratedOntologyTermList, Ontology
    from cellxgene_ontology_guide.ontology_parser import OntologyParser
    from cellxgene_ontology_guide.supported_versions import (
        CXGSchema,
        load_supported_versions,
    )

    load_supported_versions()
    cxg_schema = CXGSchema(version="5.3.0")
    cxg_schema.get_ontology_download_url(Ontology.CL)
    ontology_parser = OntologyParser(schema_version="v5.3.0")

    if level == "sub_class":
        cell_terms = get_curated_ontology_term_list(
            CuratedOntologyTermList.CELL_SUBCLASS,
        )
    elif level == "class":
        cell_terms = get_curated_ontology_term_list(
            CuratedOntologyTermList.CELL_CLASS,
        )
    else:
        raise ValueError("Must be class or sub_class")

    no_cl_labels = {}
    specific_cell_types = {}

    for c in label_dict["cell_type"]:
        term = ontology_parser.get_term_id_by_label(c, "CL")
        if term is not None:
            specific_cell_types[c] = term
        else:
            no_cl_labels[c] = c

    mapping_results = ontology_parser.map_high_level_terms(
        term_ids=list(specific_cell_types.values()),
        high_level_terms=cell_terms,
    )

    sub_label_names = {
        t: ontology_parser.get_term_label(mapping_results[t][0])
        if len(mapping_results[t]) > 0
        else ontology_parser.get_term_label(t)
        for t in mapping_results
    }

    general_labels = no_cl_labels
    general_labels.update(
        {l: sub_label_names[t] for l, t in specific_cell_types.items()}
    )

    return level, general_labels


def get_general_cell_type(
    cell_labels: list | AnnData,
    label_dict: dict,
    key: str = "cell_type_predictions",
    level: str = "cell_type_descendants",
) -> dict:
    level, general_mapping = _create_general_cell_type_mapping(label_dict, level)

    if isinstance(cell_labels, AnnData):
        cell_labels = cell_labels.copy()
        cell_type_general = [general_mapping[c] for c in cell_labels.obs[key].tolist()]
        cell_labels.obs[f"{level}_{key}"] = cell_type_general
        return cell_labels
    elif isinstance(cell_labels, list):
        cell_type_general = [general_mapping[c] for c in cell_labels]
        return cell_type_general
    else:
        raise ValueError("Must provide a list or adata object with a key.")


def convert_ids_to_label_names(label_dict: dict, label_ids: dict) -> list:
    label_dict = {value: key for key, value in label_dict.items()}
    label_names = [label_dict[c] for c in label_ids]
    return label_names


def get_label_map(key: str, predictions: dict, label_dict: dict) -> dict:
    return {
        v: k
        for k, v in label_dict[key].items()
        if v in predictions[f"{key}_predictions"]
    }


def check_gpu(set_gpu: str | None = None) -> str:
    if set_gpu is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS")
        else:
            device = torch.device("cpu")
            print("Using CPU")
    else:
        device = torch.device(set_gpu)

    return device.type


def create_results_obs(results: dict, label_dict: dict | None = None) -> sc.AnnData:
    df_embed = pd.DataFrame(index=results["cell_names"], data=results["embeddings"])
    cell_barcodes = results["cell_names"]
    preds = {
        y: convert_ids_to_label_names(
            label_dict[f"{y.replace('_predictions', '')}"], results[f"{y}"]
        )
        for y in results.keys()
        if "prediction" in y
    }

    obs = pd.DataFrame(preds, index=cell_barcodes)
    adata = sc.AnnData(X=df_embed, obs=obs)

    return adata


def get_label_dict(ckpt_path: Path | str) -> dict:
    device = check_gpu()

    ckpt = torch.load(
        ckpt_path,
        map_location=torch.device(device),
        weights_only=False,
    )

    label_dict = ckpt["hyper_parameters"]["label_dict"]
    return label_dict


def plot_embeddings(adata, labels: list | str, title: str | None = None) -> None:
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)

    if "ibm_carbon" not in mpl.colormaps:
        _ = _get_ibm_color_palette()

    title = (f"UMAP - {labels}",)

    sc.tl.umap(adata, n_components=2)
    sc.pl.embedding(
        adata,
        basis="umap",
        title=title,
        color=labels,
        show=False,
        palette="ibm_carbon",
    )

    plt.show()


def merge_bmfm_adata(bmfm_adata: AnnData, reference_adata: AnnData) -> AnnData:
    bmfm_adata = bmfm_adata.copy()
    reference_adata = reference_adata.copy()
    bmfm_adata.obsm["X_bmfm"] = bmfm_adata.X
    reference_adata.obsm["X_bmfm"] = bmfm_adata.obsm["X_bmfm"]
    reference_adata.obsm["X_umap"] = bmfm_adata.obsm["X_umap"]
    reference_adata.obs = reference_adata.obs.join(
        bmfm_adata.obs, how="left", lsuffix="_bmfm"
    )
    return reference_adata


def plot_cell_type_counts(adata: AnnData, key: str, ax=None, show: bool = False):
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)

    cell_counts = adata.obs[key].value_counts().reset_index()
    bar = sns.barplot(
        cell_counts,
        x=key,
        y="count",
        hue=key,
        palette="ibm_carbon",
        order=cell_counts[key],
        ax=ax,
    )
    ax.set_title(f"N Counts - {key}")
    ax.tick_params(axis="x", labelrotation=90)

    if show:
        plt.show()
    else:
        return bar


def generate_clusters(
    adata,
    n_components=2,
    label="CellType",
    clustering_method="kmeans",
    **kwargs,
):
    adata_copy = sc.tl.pca(adata, n_comps=n_components, copy=True)
    sc.pp.neighbors(adata_copy, use_rep="X_pca")
    n_classes_in_data = adata.obs[label].nunique()
    clusters = get_clusters(clustering_method, adata_copy, n_classes_in_data, **kwargs)

    return clusters


def get_clusters(
    clustering_method: str,
    adata_dim_reduced: sc.AnnData,
    n_classes_in_data: int,
    **kwargs,
):
    if clustering_method == "kmeans":
        if not "n_clusters" in kwargs:
            kwargs["n_clusters"] = n_classes_in_data
        kmeans = KMeans(**kwargs).fit(adata_dim_reduced.obsm["X_pca"])
        clusters = adata_dim_reduced.copy()
        clusters.obs["kmeans"] = pd.Categorical(kmeans.labels_)
    elif clustering_method == "dbscan":
        db = DBSCAN(**kwargs).fit(adata_dim_reduced.obsm["X_pca"])
        clusters = adata_dim_reduced.copy()
        clusters.obs["dbscan"] = pd.Categorical(db.labels_)
    elif clustering_method == "hierarchical":
        if not "n_clusters" in kwargs:
            kwargs["n_clusters"] = n_classes_in_data
        clusterer = AgglomerativeClustering(**kwargs)
        hierarchical = clusterer.fit(adata_dim_reduced.obsm["X_pca"])
        clusters = adata_dim_reduced.copy()
        clusters.obs["hierarchical"] = pd.Categorical(hierarchical.labels_)
    elif clustering_method in ["leiden", "louvain"]:
        raise NotImplementedError(
            "Both leiden and louvain are currently not implemented due to GPL license."
        )
    else:
        raise ValueError(
            "clustering_method is not 'kmeans' or 'dbscan' or 'hierarchical'"
        )

    return clusters
