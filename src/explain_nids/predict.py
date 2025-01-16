#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Samuli Piippo https://github.com/Paippi/

"""Module for anomaly detection."""

# Standard imports
from typing import Tuple, Optional, Callable
import pickle
from functools import partial

# 3rd party imports
import numpy as np
import numpy.typing as npt

# Local application/library specific imports
from explain_nids.typing import (
    NDArrayOrDataFrame,
    NDArrayOfIntsOrFloats,
)


def get_anomaly_detector(
    path: str, classification: bool, anomaly_threshold: Optional[float] = None
) -> Callable[NDArrayOrDataFrame, NDArrayOfIntsOrFloats]:
    """Chooses the predict function depending on the task.

    Args:
        path: Path to the anomaly detector model.
        classification: Determine whether the task is classification or
            regression. True if classification, False if regression.
        anomaly_threshold: Threshold that needs to be crossed over to
            consider prediction as anomaly.

    Returns:
        If `classification`, returns classifier,
        else returns decision function.
    """
    if classification and anomaly_threshold is None:
        raise ValueError(
            "`anomaly_threshold cannot be None if the task is classification.`"
        )
    with open(path, "rb") as f:
        anomaly_detector = pickle.load(f)

    if classification:
        return partial(
            predict_classification,
            model=anomaly_detector,
            threshold=anomaly_threshold,
        )
    return partial(distance_from_hyperplane, model=anomaly_detector)


def predict_classification(
    x: NDArrayOrDataFrame, model, threshold: float
) -> npt.NDArray[Tuple[np.int_, np.int_]]:
    """Classifies `x` using `threshold`.

    Counts distances (anomaly score) from hyperplane using `-model.decision_function` distances
    above threshold will be classified as anomaly (1), other's will be classified as benign (0).

    Args:
        x: Data to predict.
        model: Anomaly detector. Any model that gives anomaly score, using
            function `decision_function`.
        threshold: Threshold that needs to be crossed over to
            consider sample as anomalous.
    """
    res = -model.decision_function(x)
    pred = np.where(res < threshold, 0, 1)
    return np.eye(2)[pred]


def distance_from_hyperplane(
    x: NDArrayOrDataFrame, model
) -> np.ndarray[float]:
    """Converts negative values to zero and calculates the distance from hyperplane.

    Args:
        x: Data to predict.
        model: Anomaly detector. Any model that gives anomaly score, using
            function `decision_function`.

    Returns:
        Distances from hyperplane for `x`.
    """
    # Fix autoencoder generated negative numbers.
    # TODO: fix in a better way.
    x = x.copy()  # Supressing warnings...
    x[x < 0] = 0
    res = -model.decision_function(x)
    return res.reshape(-1, 1)
