import json
from pathlib import Path

import pandas as pd


def get_hgnc_df(filter_query=None):
    fname = Path(__file__).parent / "hgnc_complete_set_2024-08-23.tsv"

    hgnc = pd.read_csv(fname, sep="\t")
    if filter_query:
        return hgnc.query(filter_query)
    return hgnc


def get_protein_coding_genes():
    fname = Path(__file__).parent / "protein_coding_genes.json"
    with open(fname) as f:
        return json.load(f)
