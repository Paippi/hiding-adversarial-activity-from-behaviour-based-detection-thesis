#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Samuli Piippo https://github.com/Paippi/

"""Counterfactual reinforcement learning hyperparatemer optimization."""

# Standard imports
import warnings
import os
import argparse
import tempfile
import shutil
from typing import Union, Callable, Tuple, Dict, Any, List
from functools import partial
import pickle
import sys

# 3rd party imports
import numpy as np
import numpy.typing as npt
from alibi.explainers import CounterfactualRL
from alibi.explainers.backends.cfrl_tabular import (
    get_he_preprocessor,
    get_conditional_vector,
    apply_category_mapping,
)
from alibi.models.tensorflow import ADULTEncoder, ADULTDecoder
from alibi.explainers.cfrl_base import Postprocessing, Callback
from alibi.saving import load_explainer
import tensorflow as tf
from ray import tune, train as ray_train
from ray.train import Checkpoint, RunConfig, CheckpointConfig
from ray.tune.search.bayesopt import BayesOptSearch
from sklearn.model_selection import train_test_split


# Local application/library specific imports
from explain_nids.dataset import (
    filter_negative_rows,
    Dataset,
    get_dataset,
    filter_negative_rows,
    split_dataset,
)
from explain_nids.encoder import get_autoencoder
from explain_nids.cf.callback import RewardCallback, LossCallback
from explain_nids.cf.reward import get_reward_fn
from explain_nids.typing import NDArrayOrDataFrame
from explain_nids.predict import get_anomaly_detector

# No explanation given why this is set...
os.environ["TF_USE_LEGACY_KERAS"] = "1"

RANDOM_STATE = 42


def get_preprocessor(
    path: str, dataset: Dataset
) -> Tuple[
    Callable[NDArrayOrDataFrame, npt.NDArray[np.float64]],
    Callable[NDArrayOrDataFrame, npt.NDArray[np.float64]],
]:
    """Gets preprocessor for the dataset.

    Loads a preprocessor which is extended to disallow changes to the protocols column.

    Args:
        path: Path to a pickle file containing the preprocessor.
        dataset: Dataset that will be transformed (This is only used to gather
            the values of "Protocol" column, currently only supports one protocol).

    Returns:
        Preprocessor and inverse preprocessor.
    """
    with open(path, "rb") as f:
        preprocessor = pickle.load(f)

    dataset_t = preprocessor.transform(dataset.data)
    protocols_t = dataset_t[:, :1]

    preprocessor_ = partial(_preprocess, preprocessor=preprocessor.transform)

    inv_preprocessor = partial(
        _inv_preprocess,
        protocols_transformed=protocols_t,
        inv_preprocessor=preprocessor.inverse_transform,
    )

    return preprocessor_, inv_preprocessor


def _preprocess(x, preprocessor):
    x = preprocessor(x).astype(np.float32)
    x = x[:, 1:]
    return x


def _inv_preprocess(x, inv_preprocessor, protocols_transformed):
    if np.unique(protocols_transformed).shape[0] != 1:
        raise NotImplementedError(
            "inv_preprocess doesn't yet support more than one protocol at a time."
        )
    x = np.concatenate([protocols_transformed[: len(x)], x], axis=1)
    return inv_preprocessor(x)


def train_explainer(
    train: Dataset,
    anomaly_detector: Callable,
    preprocessor: Callable,
    inv_preprocessor: Callable,
    autoencoder: Callable,
    reward_fn: Callable,
    latent_dim: int,
    coeff_sparsity: float,
    coeff_consistency: float,
    steps: int,
    batch_size: int,
    callbacks: List[Callback],
) -> CounterfactualRL:
    """Train the CFRL model.

    Args:
        train: Train dataset.
        anomaly_detector: Anomaly detection function. See `get_anomaly_detector`.
        preprocessor: Preprocessor for the dataset.
        inv_preprocessor: Inverse preprocessor for the dataset.
        autoencoder: Autoencoder for the dataset.
        reward_fn: Reward function for the predictions.
        latent_dim: Autoencoder latent dimension.
        coeff_sparsity: Sparsity loss coefficient.
        coeff_consistency: Consisteny loss coefficient.
        steps: Number of training steps.
        batch_size: Training batch size.
        callbacks: List of callback functions to be applied after each training step.

    Returns:
        Fitted explainer.
    """
    explainer = CounterfactualRL(
        predictor=anomaly_detector,
        encoder=autoencoder.encoder,
        decoder=autoencoder.decoder,
        latent_dim=latent_dim,
        encoder_preprocessor=preprocessor,
        decoder_inv_preprocessor=inv_preprocessor,
        coeff_sparsity=coeff_sparsity,
        coeff_consistency=coeff_consistency,
        train_steps=steps,
        batch_size=batch_size,
        backend="tensorflow",
        reward_func=reward_fn,
        callbacks=callbacks,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer.fit(train.data)

    return explainer


def setup(
    config: Dict[str, Any], train: Dataset
) -> Tuple[Callable, Callable, Callable, Callable, Callable]:
    """Load models and setup functions for CFRL model.

    Args:
        config: Sampled search space. See `ray_main`.
        train: Train dataset.

    Returns:
        autoencoder, preprocessor, inverse preprocessor, anomaly detector,
        reward function.
    """
    autoencoder = get_autoencoder(config["autoencoder_path"])
    preprocessor, inv_preprocessor = get_preprocessor(
        config["preprocessor_path"], train
    )
    anomaly_detector = get_anomaly_detector(
        config["anomaly_detector_path"],
        config["classification"],
        config["anomaly_threshold"],
    )
    reward_fn = get_reward_fn(config["classification"], train)

    return (
        autoencoder,
        preprocessor,
        inv_preprocessor,
        anomaly_detector,
        reward_fn,
    )


def choose_explainer(
    config: Dict[str, Any],
    anomaly_threshold: float,
    train: Dataset,
    validation: Dataset,
):
    """Objective for `ray.tune`

    Trains the CFRL model, validates it, and reports the results to `ray`.

    Args:
        config: Sampled search space. See `ray_main`.
        anomaly_threshold: Threshold that needs to be crossed over to
            consider prediction as anomaly.
        train: Train dataset.
        validation: Validation dataset.
    """
    steps = config["steps"]
    coeff_sparsity = config["coeff_sparsity"]
    coeff_consistency = config["coeff_consistency"]
    latent_dim = config["latent_dim"]
    batch_size = config["batch_size"]

    (
        autoencoder,
        preprocessor,
        inv_preprocessor,
        anomaly_detector,
        reward_fn,
    ) = setup(config, train)

    with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
        explainer_path = os.path.join(temp_checkpoint_dir, "explainer")

        train_log_dir = os.path.join(temp_checkpoint_dir, "logs/train")
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        explainer = train_explainer(
            train,
            anomaly_detector,
            preprocessor,
            inv_preprocessor,
            autoencoder,
            reward_fn,
            latent_dim,
            coeff_sparsity,
            coeff_consistency,
            steps,
            batch_size,
            callbacks=[
                RewardCallback(train_summary_writer),
                LossCallback(train_summary_writer),
            ],
        )

        (
            reward_mean,
            reward_benign_mean,
            reward_attack_mean,
        ) = validate_explainer(
            explainer,
            validation,
            anomaly_threshold,
            reward_fn,
            config["classification"],
        )

        explainer.save(explainer_path)

        checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)

        ray_train.report(
            {
                "reward_mean": reward_mean,
                "reward_benign_mean": reward_benign_mean,
                "reward_attack_mean": reward_attack_mean,
            },
            checkpoint=checkpoint,
        )


def _choose_anomaly_targets_classification():
    return np.array([1]), np.array([0])


def _choose_anomaly_targets_regression(anomaly_threshold):
    benign_target = np.array([anomaly_threshold])

    # Target for malicious traffic, i.e., target is to be labeled benign
    # Anything that's below anomaly threshold.
    attack_target = np.array([anomaly_threshold * 0.999])
    return benign_target, attack_target


def validate_explainer(
    explainer: CounterfactualRL,
    validation: Dataset,
    anomaly_threshold: float,
    reward_fn: Callable,
    classification: bool,
) -> Tuple[float, float, float]:
    """Validates explainer.

    Args:
        explainer: Explainer fitted to generate counterfactuals.
        validation: Validation data.
        anomaly_threshold: Threshold that needs to be crossed over to
            consider prediction as anomaly.
        reward_fn: Function calculating reward (see `get_reward_fn`).
        classification: Determine whether the task is classification or
            regression. True if classification, False if regression.

    Returns:
        Mean scores for total, benign samples and attack samples.
    """
    benign_indices = get_benign_indices(validation.labels)

    benign_samples = validation.data[benign_indices]
    attack_samples = validation.data[~benign_indices]

    # Target for benign traffic, i.e., target is to be labeled malicious
    if classification:
        benign_target, attack_target = _choose_anomaly_targets_classification()
    else:
        benign_target, attack_target = _choose_anomaly_targets_regression(
            anomaly_threshold
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explanation = explainer.explain(benign_samples, benign_target)

    reward_benign = reward_fn(
        explanation["data"]["cf"]["class"], explanation["data"]["target"]
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explanation = explainer.explain(attack_samples, attack_target)

    reward_attack = reward_fn(
        explanation["data"]["cf"]["class"], explanation["data"]["target"]
    )

    reward_mean = np.concatenate([reward_benign, reward_attack]).mean()
    reward_benign_mean = reward_benign.mean()
    reward_attack_mean = reward_attack.mean()

    return reward_mean, reward_benign_mean, reward_attack_mean


def main(argv=sys.argv):
    """Optimize hyperparameters for CFRL.

    Optimizes hyperparameters using `ray.tune` hyperparameter optimization library.
    See `python train_counterfactual_rl.py --help` for more info.
    """
    parser = _create_argparser()
    args = parser.parse_args(argv[1:])
    _main(args)


def _main(args) -> tune.ResultGrid:
    dataset = get_dataset(
        args.dataset_path, args.label_column, filters=[filter_negative_rows]
    )
    train, validation = split_dataset(
        dataset,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    return ray_main(
        train,
        validation,
        args.steps,
        os.path.abspath(args.anomaly_detector_path),
        args.anomaly_threshold,
        os.path.abspath(args.preprocessor_path),
        os.path.abspath(args.autoencoder_path),
        args.latent_dim,
        args.batch_size,
        args.num_samples,
        args.classification,
        args.experiment_name,
        args.coeff_sparsity,
        args.coeff_consistency,
    )


def ray_main(
    train: Dataset,
    validation: Dataset,
    steps: int,
    anomaly_detector_path: str,
    anomaly_threshold: float,
    preprocessor_path: str,
    autoencoder_path: str,
    latent_dim: int,
    batch_size: int,
    num_samples: int,
    classification: bool,
    experiment_name: str,
    coeff_sparsity: Union[float, None] = None,
    coeff_consistency: Union[float, None] = None,
) -> tune.ResultGrid:
    """Optimize the CFRL model's hyperparameters.

    Args:
        train: Train dataset.
        validation: Validation dataset.
        steps: Number of steps.
        anomaly_detector_path: Path to the anomaly detector pickle file
            containing any model that gives anomaly score, using function
            `decision_function`.
        anomaly_threshold: Threshold that needs to be crossed over to
            consider prediction as anomaly.
        preprocessor_path: Path to the preprocessor pickle file.
        autoencoder_path: Path to the Keras autoencoder.
        latent_dim: Latent dimension (the same as with autoencoder)
        batch_size: Batch size.
        num_samples: Number of times to sample from the hyperparameter search space.
        classification: Determine whether the task is classification or
            regression. True if classification, False if regression.
        experiment_name: Name of the experiment. Depending on the task
            "-classification" or "-regression" will be added to the name.
        coeff_sparsity: If set will force the tuner to use given coeff sparsity,
            otherwise will search for the best coeff sparsity.
        coeff_consistency: If set will force the tuner to use given coeff consistency,
            otherwise will search for the best coefff consistency.

    Returns:
        Result grid containing each experiment's results.
    """
    metric = "reward_mean"
    search_space = {
        "steps": steps,
        "coeff_sparsity": coeff_sparsity
        if coeff_sparsity
        else tune.quniform(0.01, 1 - 0.01, 0.01),
        "coeff_consistency": coeff_consistency
        if coeff_consistency
        else tune.quniform(0.01, 1 - 0.01, 0.01),
        "latent_dim": latent_dim,
        "batch_size": batch_size,
        "anomaly_detector_path": anomaly_detector_path,
        "anomaly_threshold": anomaly_threshold,
        "preprocessor_path": preprocessor_path,
        "autoencoder_path": autoencoder_path,
        "classification": classification,
    }
    bayesopt = BayesOptSearch(
        metric=metric,
        mode="max",
        random_state=RANDOM_STATE,
    )
    tuner = tune.Tuner(
        tune.with_parameters(
            choose_explainer,
            anomaly_threshold=anomaly_threshold,
            train=train,
            validation=validation,
        ),
        run_config=RunConfig(
            name=f"{experiment_name}-{'classification' if classification else 'regression'}",
            checkpoint_config=CheckpointConfig(
                num_to_keep=3,
                checkpoint_score_attribute=metric,
                checkpoint_score_order="max",
            ),
        ),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            num_samples=num_samples,
            metric=metric,
            mode="max",
            # search_alg=bayesopt,
        ),
    )

    results = tuner.fit()
    return results


def _create_argparser():
    parser = argparse.ArgumentParser(
        prog="train-cfrl",
        description="Train counterfactual reinforcement learning model.",
    )
    parser.add_argument(
        "anomaly_detector_path",
        help="Path to the anomaly detector pickle file containing any model that "
        "gives anomaly score, using function `decision_function`.",
    )
    parser.add_argument(
        "dataset_path",
        help="Path to the dataset used to train and validate the CFRL model.",
    )
    parser.add_argument(
        "autoencoder_path", help="Path to the Keras autoencoder"
    )
    parser.add_argument(
        "preprocessor_path", help="Path to the preprocessor pickle file."
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        help="Forces the tuning to use certain latent dimension. Needs to be the "
        "same as for the autoencoder.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100_000,
        help="Set number of steps (default 100_000)",
    )
    parser.add_argument(
        "--coeff-sparsity",
        type=float,
        default=None,
        help="If set will force model's to use given coeff sparsity (if not "
        "specified will search for best coeff sparsity)",
    )
    parser.add_argument(
        "--coeff-consistency",
        type=float,
        default=None,
        help="If set will force model's to use given coeff consistency (if not "
        "specified will search for best coeff consistency)",
    )
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument(
        "--anomaly-threshold",
        type=float,
        default=-0.0002196942507948895,
        # By default the given threshold is from the multi-stage hierarchical
        # IDS paper for the best F1 score.
        help="Set the anomaly threshold, ignored if --classification flag set.",
    )
    parser.add_argument(
        "--classification",
        action="store_true",
        help="Flag that if set, will use classification reward system instead of regression",
    )
    parser.add_argument(
        "--num-samples",
        default=1000,
        type=int,
        help="Number of times to sample from the hyperparameter search space.",
    )
    parser.add_argument(
        "--label-column",
        default="Y",
        help="Column name containing the sample's label/target.",
    )
    parser.add_argument(
        "--experiment-name",
        default="CIC-IDS-2017-explainer",
        help="Name for the experiment (default CIC-IDS-2017-explainer)",
    )
    parser.add_argument(
        "--output_dir",
        default="output",
        help="Path where to save the best result.",
    )
    return parser


if __name__ == "__main__":
    parser = _create_argparser()
    args = parser.parse_args()
    results = _main(args)

    anomaly_detector = get_anomaly_detector(
        args.anomaly_detector_path, args.classification, args.anomaly_threshold
    )
    best = results.get_best_result()
    with best.checkpoint.as_directory() as result_dir:
        output_dir = os.path.join(
            args.output_dir,
            args.experiment_name + "-" + "classification"
            if args.classification
            else "regression",
        )
        shutil.copytree(result_dir, output_dir, dirs_exist_ok=True)
        explainer = load_explainer(
            os.path.join(result_dir, "explainer"),
            anomaly_detector,
        )
