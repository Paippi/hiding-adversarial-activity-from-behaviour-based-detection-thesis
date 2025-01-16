#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Samuli Piippo https://github.com/Paippi/

"""Callbacks for CFRL."""

# Standard imports
from typing import Dict

# 3rd party imports
from alibi.explainers.cfrl_base import Callback
import numpy as np
import tensorflow as tf

# Local application/library specific imports


class RewardCallback(Callback):
    def __init__(self, log_writer):
        self.log_writer = log_writer

    def __call__(
        self,
        step: int,
        update: int,
        model,
        sample: Dict[str, np.ndarray],
        losses: Dict[str, float],
    ):
        if (step + update) % 100 != 0:
            return

        target = sample["Y_t"]
        X_cf = model.params["decoder_inv_preprocessor"](sample["X_cf"])

        prediction = model.params["predictor"](X_cf)

        reward = np.mean(model.params["reward_func"](prediction, target))
        with self.log_writer.as_default():
            tf.summary.scalar("reward", reward, step=step)


class LossCallback(Callback):
    """Callback that reports losses during training."""

    def __init__(self, log_writer: tf.summary.SummaryWriter):
        self.log_writer = log_writer

    def __call__(
        self,
        step: int,
        update: int,
        model,
        sample: Dict[str, np.ndarray],
        losses: Dict[str, float],
    ):
        if (step + update) % 100 != 0:
            return

        with self.log_writer.as_default():
            for loss_name, loss in losses.items():
                tf.summary.scalar(loss_name, loss, step=step)
