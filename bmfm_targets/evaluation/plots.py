from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import requests
import scanpy as sc
import seaborn as sns
from anndata import AnnData


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
