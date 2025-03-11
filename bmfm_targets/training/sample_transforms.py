"""
Transforms operating on MultiFieldInstance objects.

Used for modifying training logic or augmenting data.
"""

import random
import warnings
from operator import itemgetter
from typing import Literal

import numpy as np
import scipy.stats as stats

from bmfm_targets.tokenization import MultiFieldInstance


def randomize(mfi: MultiFieldInstance, *args, **kwargs) -> MultiFieldInstance:
    """
    Randomize the order of the tokens in each field.

    Args:
    ----
        mfi (MultiFieldInstance): mfi

    Returns:
    -------
        MultiFieldInstance: mfi

    """
    shuffled_indices = random.sample(range(mfi.seq_length), k=mfi.seq_length)
    randomized_data = {}
    for field, values in mfi.data.items():
        randomized_data[field] = [values[i] for i in shuffled_indices]
    return MultiFieldInstance(data=randomized_data, metadata=mfi.metadata)


def sort_by_field(
    mfi: MultiFieldInstance, field: str, reverse=True, *args, **kwargs
) -> MultiFieldInstance:
    """
    Sort all fields by floating point values of requested field.

    Args:
    ----
        mfi (MultiFieldInstance): mfi
        field (str): name of field
        reverse (bool, optional): reverse sort (True means large to small). Defaults to True.

    Returns:
    -------
        MultiFieldInstance: mfi

    """
    try:
        float_vals = [float(i) for i in mfi.data[field]]
        is_float_field = True
    except ValueError:
        is_float_field = False

    if is_float_field:
        sorted_indices = sorted(
            range(len(mfi.data[field])),
            key=lambda i: float(mfi.data[field][i]),
            reverse=reverse,
        )
    else:
        sorted_indices = sorted(
            range(len(mfi.data[field])),
            key=lambda i: mfi.data[field][i],
            reverse=reverse,
        )
    sorted_data = {}
    for field, values in mfi.data.items():
        sorted_data[field] = [values[i] for i in sorted_indices]
    return MultiFieldInstance(data=sorted_data, metadata=mfi.metadata)


def rda_downsample(
    mfi: MultiFieldInstance,
    max_length: int | None,
    normalized_sum,
    downsample_threshold,
    *args,
    **kwargs,
) -> MultiFieldInstance:
    """
    Implement read-depth aware downsampling of multifield instance.

    Downsample the gene expression values based on the criteria described in
    Hao, Minsheng, et al. "Large-scale foundation model on single-cell transcriptomics."
    Nature Methods (2024): 1-11.

    Note that the norm/log transforms need to be performed in this function
    because they must operate only on the sequence, not the S and T values.

    Args:
    ----
        max_length (int): max length of the sequence
        normalized_sum (int): sum to use for normalizing
        downsample_threshold (int): read count below which not to downsample

    Returns:
    -------
        MultiFieldInstance: RDA transformed MultiFieldInstance with
            additional field "label_expressions" and additional tokens "S" and "T"

    """
    if max_length is not None:
        for key, val in mfi.data.items():
            mfi.data[key] = val[: max_length - 2]
    raw_expressions = np.array([float(value) for value in mfi.data["expressions"]])
    raw_expressions_sum = sum(raw_expressions)
    gamma = (
        0 if raw_expressions_sum < downsample_threshold else np.random.binomial(1, 0.5)
    )
    if gamma == 1:
        beta_distribution = stats.beta(a=2, b=2)
        b = beta_distribution.rvs()
        input_expressions = np.random.binomial(n=raw_expressions.astype(int), p=b)
    else:
        input_expressions = raw_expressions.copy()
    input_expressions_sum = input_expressions.sum()
    if input_expressions_sum == 0:
        input_expressions = raw_expressions.copy()
        input_expressions_sum = raw_expressions_sum
    raw_expressions = np.log1p((raw_expressions / raw_expressions_sum) * normalized_sum)
    input_expressions = np.log1p(
        (input_expressions / input_expressions_sum) * normalized_sum
    )
    genes = ["[S]", "[T]"] + mfi.data["genes"]
    ST_list = [np.log1p(input_expressions_sum), np.log1p(raw_expressions_sum)]
    input_expressions = ST_list + input_expressions.tolist()
    raw_expressions = ST_list + raw_expressions.tolist()

    updated_data = {}
    for field, vals in mfi.data.items():
        if field == "genes":
            updated_data[field] = genes
        elif field == "expressions":
            updated_data[field] = input_expressions
            updated_data["label_expressions"] = raw_expressions
        else:
            updated_data[field] = vals

    return MultiFieldInstance(
        data=updated_data,
        metadata=mfi.metadata,
    )


def log_normalize(
    mfi: MultiFieldInstance,
    max_length: int | None,
    normalized_sum=10000,
    *args,
    **kwargs,
) -> MultiFieldInstance:
    """

    Implement log normalization.

    ----
        max_length (int): max length of the sequence
        normalized_sum (int): sum to use for normalizing

    Returns
    -------
        MultiFieldInstance: Log normalized MutifieldInstance

    """
    if max_length is not None:
        for key, val in mfi.data.items():
            mfi.data[key] = val[: max_length - 2]
    expressions = np.array([float(value) for value in mfi.data["expressions"]])
    expressions_sum = sum(expressions) + 1
    expressions = np.log1p((expressions / expressions_sum) * normalized_sum)

    updated_data = {}
    for field, vals in mfi.data.items():
        if field == "expressions":
            updated_data[field] = expressions
        else:
            updated_data[field] = vals

    return MultiFieldInstance(
        data=updated_data,
        metadata=mfi.metadata,
    )


def dropout_random(
    mfi: MultiFieldInstance, dropout_ratio: float, *args, **kwargs
) -> MultiFieldInstance:
    """
    Dropout elements at random across all fields.

    Args:
    ----
        mfi (MultiFieldInstance): mfi
        dropout_ratio (float): probability of being dropped out

    Returns:
    -------
        MultiFieldInstance: mfi

    """
    assert 0 <= dropout_ratio < 1
    sequence_len = len(next(iter(mfi.data.values())))
    dropout_indices = [
        i for i in range(sequence_len) if dropout_ratio > random.random()
    ]
    data_after_dropout = {}
    for field, values in mfi.data.items():
        data_after_dropout[field] = [
            elem for i, elem in enumerate(values) if i not in dropout_indices
        ]
    return MultiFieldInstance(data=data_after_dropout, metadata=mfi.metadata)


def dropout_chunk_in_range(
    mfi: MultiFieldInstance,
    chunk_size: int,
    drop_range: tuple | None = None,
    *args,
    **kwargs,
) -> MultiFieldInstance:
    """
    Dropout contiguous block from MultiFieldInstance.

    Args:
    ----
        mfi (MultiFieldInstance): mfi
        chunk_size (int): size of chunk to drop
        drop_range (tuple | None, optional): range within which to drop. Defaults to None.

    Returns:
    -------
        MultiFieldInstance: mfi

    """
    if drop_range is None:
        drop_range = (0, mfi.seq_length - 1)

    range_len = drop_range[1] - drop_range[0] + 1
    if mfi.seq_length < range_len:
        drop_range = (drop_range[0], drop_range[0] + mfi.seq_length - 1)
        range_len = drop_range[1] - drop_range[0] + 1

    if chunk_size > range_len:
        warnings.warn(
            f"Requested to drop chunk of size {chunk_size} in range of length {range_len}. Ignoring."
        )
        return mfi

    start_index = random.randint(0, range_len - chunk_size)
    stop_index = start_index + chunk_size
    data_after_dropout = {}
    for field, values in mfi.data.items():
        data_after_dropout[field] = values[:start_index] + values[stop_index:]
    return MultiFieldInstance(data=data_after_dropout, metadata=mfi.metadata)


def rda_align(
    mfi: MultiFieldInstance,
    target_read_resolution: int,
    normalized_sum: int,
    max_length: int | None = None,
    *args,
    **kwargs,
) -> MultiFieldInstance:
    """
    Read-Depth align a sample for inference.

    Insert target read resolution in the [T] position and input read resolution in the [S]
    position, for aligning read depth in a model that was trained with rda_downsampling.

    Args:
    ----
        mfi (MultiFieldInstance): MultiFieldInstance with raw integer read counts
        target_read_resolution (int): target read resolution - sample will be embedded
          as if it were being upsampled to this resolution
        normalized_sum (int): sum for normalization
        max_length (int | None, optional): max length so that the sums of reads are
          calculated correctly. Defaults to None.

    Returns:
    -------
        MultiFieldInstance: _description_

    """
    if max_length is not None:
        for key, val in mfi.data.items():
            mfi.data[key] = val[: max_length - 2]

    raw_expressions = np.array([float(value) for value in mfi.data["expressions"]])
    raw_expressions_sum = sum(raw_expressions)
    normed_expressions = np.log1p(
        (raw_expressions / raw_expressions_sum) * normalized_sum
    )

    genes = ["[S]", "[T]"] + mfi.data["genes"]
    ST_list = [np.log1p(raw_expressions_sum), np.log1p(target_read_resolution)]
    rda_aligned_expressions = ST_list + normed_expressions.tolist()

    updated_data = {}
    for field, vals in mfi.data.items():
        if field == "genes":
            updated_data[field] = genes
        elif field == "expressions":
            updated_data[field] = rda_aligned_expressions
        else:
            updated_data[field] = vals

    return MultiFieldInstance(
        data=updated_data,
        metadata=mfi.metadata,
    )


def pad_zero_expressed_genes(
    mfi: MultiFieldInstance,
    pad_zero_expression_strategy: str,
    max_length: int,
    *args,
    **kwargs,
):
    if pad_zero_expression_strategy == "batch_wise":
        assert kwargs["expressed_genes_in_batch"] is not None
        updated_data = {}
        expressed_indices = []
        batch_expressed_indices = []
        non_expressed_indices = []
        for i, (gene, expression) in enumerate(
            zip(mfi.data["genes"], mfi.data["expressions"])
        ):
            if expression > 0.0:
                expressed_indices.append(i)
            elif gene in kwargs["expressed_genes_in_batch"]:
                batch_expressed_indices.append(i)
            else:
                non_expressed_indices.append(i)
        ordered_indices = (
            expressed_indices + batch_expressed_indices + non_expressed_indices
        )
        for field, values in mfi.data.items():
            updated_data[field] = list(
                itemgetter(*ordered_indices)(values)[:max_length]
            )
        updated_mfi = MultiFieldInstance(data=updated_data, metadata=mfi.metadata)

    elif pad_zero_expression_strategy == "random":
        data_to_keep = {}
        zero_indices = []
        keep_indices = []
        for i, expression in enumerate(mfi.data["expressions"]):
            if expression == 0.0:
                zero_indices.append(i)
            else:
                keep_indices.append(i)
        if len(keep_indices) < max_length:
            keep_indices.extend(
                random.sample(zero_indices, max_length - len(keep_indices))
            )
        for field, values in mfi.data.items():
            data_to_keep[field] = list(itemgetter(*keep_indices)(values))
        updated_mfi = MultiFieldInstance(data=data_to_keep, metadata=mfi.metadata)

    return updated_mfi


def downcast_numeric_fields(
    mfi: MultiFieldInstance,
    fields_to_downcast: list[str],
    casting_strategy: Literal["round"] | Literal["ceil"] | Literal["floor"] = "round",
    *args,
    **kwargs,
):
    """
    Downcast floating point fields to integers for tokenization.

    Args:
    ----
        mfi (MultiFieldInstance): input mfi
        fields_to_downcast (list[str]): list of fields to downcast
        casting_strategy ("round","ceil","floor"):
            Whether to downcast via round, floor or ceil. Defaults to "round".

    Returns:
    -------
        _type_: mfi with integer fields in place of floats

    """
    np_op = {"round": np.round, "floor": np.floor, "ceil": np.ceil}

    data = {
        k: (
            np_op[casting_strategy](v).astype(int).tolist()
            if k in fields_to_downcast
            else v
        )
        for k, v in mfi
    }
    return MultiFieldInstance(data=data, metadata=mfi.metadata)
