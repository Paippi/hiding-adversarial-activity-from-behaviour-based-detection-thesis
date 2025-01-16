#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Samuli Piippo https://github.com/Paippi/

"""<module docstring>"""

# Standard imports
from typing import Callable, Tuple
from functools import partial

# 3rd party imports
import numpy as np
import numpy.typing as npt
from alibi.explainers.backends.cfrl_base import get_classification_reward


# Local application/library specific imports
from explain_nids.typing import NDArrayOfIntsOrFloats
from explain_nids.dataset import Dataset, get_benign_indices


def get_reward_fn(
    classification: bool, train: None = Dataset
) -> Callable[
    Tuple[NDArrayOfIntsOrFloats, NDArrayOfIntsOrFloats],
    npt.NDArray[np.int_],
]:
    """Chooses the reward function depending on the task.

    Args:
        classification: Determine whether the task is classification or
            regression. True if classification, False if regression.
        train: Dataset used for training.

    Returns:
        If `classification`, returns classification reward function,
        else returns regression based reward function.
    """
    if classification:
        return get_classification_reward

    benign_indices = get_benign_indices(train.labels)
    # Tolerance works as a standard measure and is model agnostic.
    # Using benign traffic as it is what one would gather from real world.
    # Malicious traffic can be too unpredictable.
    tolerance = np.quantile(train.data[benign_indices], 0.95)

    return partial(reward_fn, tolerance=tolerance)


def reward_fn(
    cf_prediction: npt.NDArray[np.float64],
    target: npt.NDArray[np.float64],
    tolerance: float,
) -> npt.NDArray[int]:
    """Reward function that uses tolerance as a standard measure (for anomaly_score).

    Calculates reward based on tolerance, if the distance between the `cf_prediction`
    and the `target` deviates by this standard measure (tolerance) by:
        * 2.5% = 4 points
        * 5% = 2 points
        * 10% = 1 point.

    Args:
        cf_prediction: The prediction given to counterfactual.
        target: The target where the counterfactual aimed for.
        tolerance: Some value intended to work as a standard measure, so that each
            reward is equally counted.

    Returns:
        Array of scores.
    """
    distance = cf_prediction - target
    return np.vectorize(_count_reward)(distance, tolerance)


def _count_reward(distance, tolerance):
    if distance <= tolerance * 0.025:
        return 4
    elif distance <= tolerance * 0.05:
        return 2
    elif distance <= tolerance * 0.1:
        return 1
    return 0
