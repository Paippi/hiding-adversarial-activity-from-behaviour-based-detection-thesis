#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Samuli Piippo https://github.com/Paippi/

"""Dataset related functionality."""

# Standard imports
from dataclasses import dataclass
from typing import List, Callable

# 3rd party imports
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.model_selection import train_test_split

# Local application/library specific imports
from explain_nids.typing import NDArrayOfIntsOrFloats


@dataclass
class Dataset:
    """Convenience class made for easier usage of data and labels."""

    data: NDArrayOfIntsOrFloats
    labels: npt.NDArray[np.str_]


def get_benign_indices(
    labels: npt.NDArray[np.str_],
) -> npt.NDArray[np.bool_]:
    """Get indices of benign labels from `labels`."""
    return labels == "benign"


def rename_columns(df: pd.DataFrame) -> None:
    """Converts new column names to the older one, in place.

    This is done just so that the transformer doesn't complain that it was fitted
    with different feature names.

    Args:
        df: Dataframe which columns are renamed.
    """
    mapping = {
        "Protocol": "Protocol",
        "Flow Duration": "Flow Duration",
        "Total Fwd Packets": "Total Fwd Packets",
        "Total Backward Packets": "Total Backward Packets",
        "Total Length of Fwd Packets": "Fwd Packets Length Total",
        "Total Length of Bwd Packets": "Bwd Packets Length Total",
        "Fwd Packet Length Max": "Fwd Packet Length Max",
        "Fwd Packet Length Min": "Fwd Packet Length Min",
        "Fwd Packet Length Mean": "Fwd Packet Length Mean",
        "Fwd Packet Length Std": "Fwd Packet Length Std",
        "Bwd Packet Length Max": "Bwd Packet Length Max",
        "Bwd Packet Length Min": "Bwd Packet Length Min",
        "Bwd Packet Length Mean": "Bwd Packet Length Mean",
        "Bwd Packet Length Std": "Bwd Packet Length Std",
        "Flow Bytes/s": "Flow Bytes/s",
        "Flow Packets/s": "Flow Packets/s",
        "Flow IAT Mean": "Flow IAT Mean",
        "Flow IAT Std": "Flow IAT Std",
        "Flow IAT Max": "Flow IAT Max",
        "Flow IAT Min": "Flow IAT Min",
        "Fwd IAT Total": "Fwd IAT Total",
        "Fwd IAT Mean": "Fwd IAT Mean",
        "Fwd IAT Std": "Fwd IAT Std",
        "Fwd IAT Max": "Fwd IAT Max",
        "Fwd IAT Min": "Fwd IAT Min",
        "Bwd IAT Total": "Bwd IAT Total",
        "Bwd IAT Mean": "Bwd IAT Mean",
        "Bwd IAT Std": "Bwd IAT Std",
        "Bwd IAT Max": "Bwd IAT Max",
        "Bwd IAT Min": "Bwd IAT Min",
        "Fwd PSH Flags": "Fwd PSH Flags",
        "Fwd Header Length": "Fwd Header Length",
        "Bwd Header Length": "Bwd Header Length",
        "Fwd Packets/s": "Fwd Packets/s",
        "Bwd Packets/s": "Bwd Packets/s",
        "Min Packet Length": "Packet Length Min",
        "Max Packet Length": "Packet Length Max",
        "Packet Length Mean": "Packet Length Mean",
        "Packet Length Std": "Packet Length Std",
        "Packet Length Variance": "Packet Length Variance",
        "FIN Flag Count": "FIN Flag Count",
        "SYN Flag Count": "SYN Flag Count",
        "RST Flag Count": "RST Flag Count",
        "PSH Flag Count": "PSH Flag Count",
        "ACK Flag Count": "ACK Flag Count",
        "URG Flag Count": "URG Flag Count",
        "ECE Flag Count": "ECE Flag Count",
        "Down/Up Ratio": "Down/Up Ratio",
        "Average Packet Size": "Avg Packet Size",
        "Avg Fwd Segment Size": "Avg Fwd Segment Size",
        "Avg Bwd Segment Size": "Avg Bwd Segment Size",
        "Subflow Fwd Packets": "Subflow Fwd Packets",
        "Subflow Fwd Bytes": "Subflow Fwd Bytes",
        "Subflow Bwd Packets": "Subflow Bwd Packets",
        "Subflow Bwd Bytes": "Subflow Bwd Bytes",
        "Init_Win_bytes_forward": "Init Fwd Win Bytes",
        "Init_Win_bytes_backward": "Init Bwd Win Bytes",
        "act_data_pkt_fwd": "Fwd Act Data Packets",
        "min_seg_size_forward": "Fwd Seg Size Min",
        "Active Mean": "Active Mean",
        "Active Std": "Active Std",
        "Active Max": "Active Max",
        "Active Min": "Active Min",
        "Idle Mean": "Idle Mean",
        "Idle Std": "Idle Std",
        "Idle Max": "Idle Max",
        "Idle Min": "Idle Min",
    }
    df.rename(columns=mapping, inplace=True)


def split_dataset(
    dataset: Dataset, test_size: float, random_state: int = None
):
    (
        train_data,
        validation_data,
        train_labels,
        validation_labels,
    ) = train_test_split(
        dataset.data,
        dataset.labels,
        stratify=dataset.labels,
        test_size=test_size,
        random_state=random_state,
    )
    return Dataset(train_data, train_labels), Dataset(
        validation_data, validation_labels
    )


def filter_negative_rows(dataset: pd.DataFrame):
    """Remove rows containing negative values from the dataset"""
    return dataset[dataset.select_dtypes(include=[np.number]).ge(0).all(1)]


def get_dataset(
    path,
    label_column="Y",
    filters: List[Callable[pd.DataFrame, pd.DataFrame]] = None,
) -> Dataset:
    """Loads given dataset and removes values less than zero from it (impossible values).

    Args:
        path: Path to the dataset.
        label_column: Column name containing the sample's label/target.
        filters: Filters to apply to the dataset after reading it.

    Returns:
        Dataset and labels converted to lower characters.
    """
    dataset = pd.read_csv(path)

    if filters:
        # Apply filters
        for filter_ in filters:
            dataset = filter_(dataset)

    labels = dataset.pop(label_column)
    rename_columns(dataset)
    return Dataset(dataset, labels.str.lower())
