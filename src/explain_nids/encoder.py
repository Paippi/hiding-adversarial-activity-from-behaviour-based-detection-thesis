#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Samuli Piippo https://github.com/Paippi/

"""Autoencoder specification and hyperparatemer optimization."""
# Standard imports
import sys
import os
from typing import List, Dict, Callable, Tuple, Any
import argparse
import shutil
import tempfile
import pickle

# 3rd party imports
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from ray import train as ray_train, tune
from ray.train import RunConfig, CheckpointConfig
from ray.air.integrations.keras import ReportCheckpointCallback
from ray.train.tensorflow import TensorflowCheckpoint
from ray.util.annotations import PublicAPI

# Local application/library specific imports
from explain_nids.typing import OptionalInt

# No explanation given why this is set...
os.environ["TF_USE_LEGACY_KERAS"] = "1"


class Encoder(keras.Model):
    """Encoder for CIC-IDS-2017."""

    def __init__(self, hidden_dim, latent_dim, **kwargs):
        """Constructor.

        Args:
            hidden_dim: Hidden dimension.
            latent_dim: Latent dimension.
        """
        super().__init__(**kwargs)

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.fc1 = keras.layers.Dense(hidden_dim, activation="relu")
        self.fc2 = keras.layers.Dense(latent_dim, activation="tanh")

    def call(self, x: tf.Tensor, **kwargs) -> tf.Tensor:
        """Forward pass.

        Args:
            x: Input tensor.
            **kwargs: Ignored other arguments.

        Returns:
            Encoded representation of values between [-1, 1].
        """
        return self.fc2(self.fc1(x))

    def get_config(self):
        base_config = super().get_config()
        config = {
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        hidden_dim = config.pop("hidden_dim")
        latent_dim = config.pop("latent_dim")

        return cls(hidden_dim, latent_dim, **config)


class Decoder(keras.Model):
    """Decoder for CIC-IDS-2017."""

    def __init__(self, hidden_dim: int, output_dim: int, **kwargs):
        """Constructor.

        Args:
            hidden_dim: Hidden dimension.
            output_dim: Output dimension.
        """
        super().__init__(**kwargs)

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.fc1 = keras.layers.Dense(hidden_dim, activation="relu")
        self.fc2 = keras.layers.Dense(output_dim)

    def call(self, x: tf.Tensor, **kwargs) -> List[tf.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor.
            **kwargs: Ingored other arguments.

        Returns:
            Reconstructed values.
        """
        return self.fc2(self.fc1(x))

    def get_config(self):
        base_config = super().get_config()
        config = {
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        hidden_dim = config.pop("hidden_dim")
        output_dim = config.pop("output_dim")

        return cls(hidden_dim, output_dim, **config)


class AutoEncoder(keras.Model):
    """Simple autoencoder for CIC-IDS-2017."""

    def __init__(self, encoder: keras.Model, decoder: keras.Model, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, x: tf.Tensor, **kwargs):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def get_config(self):
        base_config = super().get_config()
        config = {
            "encoder": keras.saving.serialize_keras_object(self.encoder),
            "decoder": keras.saving.serialize_keras_object(self.decoder),
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        encoder_config = config.pop("encoder")
        encoder = keras.saving.deserialize_keras_object(encoder_config)

        decoder_config = config.pop("decoder")
        decoder = keras.saving.deserialize_keras_object(decoder_config)

        return cls(encoder, decoder, **config)


def train_autoencoder(
    config: Dict[str, Any],
    preprocessor: Callable,
    train: np.ndarray,
):
    """Trains autoencoder, checkpoints model and reports results to ray.

    Args:
        config: Sampled search space. See `ray_main`.
        preprocessor: Preprocessor for the dataset.
        train: trainset for autoencoder.
    """
    epochs = config["epochs"]
    hidden_dim = config["hidden_dim"]
    latent_dim = config["latent_dim"]
    # In a real application we would probably use the training data to teach
    # the autoencoder. But here we don't have access to the original training
    # data.

    train_transformed = preprocessor(train).astype(np.float32)

    # Removing the protocol column from the trainset since it shouldn't ever
    # change.
    train_transformed = train_transformed[:, 1:]

    trainset = tf.data.Dataset.from_tensor_slices(train_transformed)
    trainset = trainset.map(lambda x: (x, x))
    trainset = trainset.shuffle(1024).batch(128, drop_remainder=True)

    autoencoder = AutoEncoder(
        encoder=Encoder(hidden_dim=hidden_dim, latent_dim=latent_dim),
        decoder=Decoder(
            hidden_dim=hidden_dim, output_dim=train_transformed.shape[1]
        ),
    )

    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.MeanSquaredError(),
        loss_weights=1.0,
        metrics=keras.metrics.MeanSquaredError(name="mean_squared_error"),
    )

    autoencoder.fit(
        trainset,
        epochs=epochs,
        callbacks=[
            ReportCheckpointSubclassCallback(
                metrics={"mse": "mean_squared_error"},
            )
        ],
    )


def main(argv=sys.argv):
    """Optimize hyperparameters for the autoencoder.

    Optimizes hyperparameters using `ray.tune` hyperparameter optimization library.
    See `python encoder.py --help` for more info.
    """
    parser = _create_argparser()
    args = parser.parse_args(argv[1:])
    _main(args)


def _main(args):
    output_path = args.output_path
    overwrite = args.overwrite
    epochs = args.epochs
    hidden_dim = args.hidden_dim
    latent_dim = args.latent_dim
    num_samples = args.num_samples
    preprocessor_path = args.preprocessor_path

    with open(preprocessor_path, "rb") as f:
        preprocessor = pickle.load(f)

    train = pd.read_csv(args.trainset_path)
    # Remove rows with negative values.
    train = train[train.select_dtypes(include=[np.number]).ge(0).all(1)]

    # Remove labels
    train.pop("Y")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if len(os.listdir(output_path)) == 0 or overwrite:
        results = ray_main(
            train,
            epochs,
            preprocessor.transform,
            num_samples,
            hidden_dim,
            latent_dim,
        )
        res = results.get_best_result()
        autoencoder = get_autoencoder(res.checkpoint.path)
        autoencoder.save(output_path, save_format="tf")
    else:
        autoencoder = get_autoencoder(output_path)

    return {
        "encoder": autoencoder.encoder,
        "decoder": autoencoder.decoder,
        "preprocessor": preprocessor.transform,
        "inv_preprocessor": preprocessor.inverse_transform,
    }


def ray_main(
    train: pd.DataFrame,
    epochs: int,
    preprocessor: Callable,
    num_samples: int,
    hidden_dim: OptionalInt = None,
    latent_dim: OptionalInt = None,
):
    """Optimize the `AutoEncoder` model's hyperparameters.

    Args:
        train: Train dataset.
        epochs: Number of epochs.
        preprocessor: Preprocessor function.
        num_samples: Number of times to sample from the hyperparameter search space.
        hidden_dim: If set will force the tuner to use given hidden dimension
            for the autoencoder, otherwise will search for the best hidden dimension.
        latent_dim: If set will force the tuner to use given latent dimension
            for the autoencoder, otherwise will search for the best latent dimension.
    """
    metric = "mse"
    search_space = {
        "epochs": epochs,
        "hidden_dim": hidden_dim if hidden_dim else tune.randint(50, 200),
        "latent_dim": latent_dim if latent_dim else tune.randint(1, 50),
    }

    tuner = tune.Tuner(
        tune.with_parameters(
            train_autoencoder,
            preprocessor=preprocessor,
            train=train,
        ),
        run_config=RunConfig(
            # storage_path="./ray_results",
            name="CIC-IDS-2017-AE",
            checkpoint_config=CheckpointConfig(
                num_to_keep=3,
                checkpoint_score_attribute=metric,
                checkpoint_score_order="min",
            )
            # storage_filesystem=LocalFileSystem(),
        ),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            num_samples=num_samples,
            metric=metric,
            mode="min",
        ),
    )

    results = tuner.fit()
    return results


@PublicAPI(stability="alpha")
class ReportCheckpointSubclassCallback(ReportCheckpointCallback):
    """Slightly modified `ReportCheckpointCallback` to save extended `keras.Model`s."""

    def _handle(self, logs: dict, when: str):
        assert when in self._checkpoint_on or when in self._report_metrics_on

        metrics = self._get_reported_metrics(logs)

        should_checkpoint = when in self._checkpoint_on
        if should_checkpoint:
            checkpoint = TensorflowSubclassCheckpoint.from_model(self.model)
            ray_train.report(metrics, checkpoint=checkpoint)
            # Clean up temporary checkpoint
            shutil.rmtree(checkpoint.path, ignore_errors=True)
        else:
            ray_train.report(metrics, checkpoint=None)


class TensorflowSubclassCheckpoint(TensorflowCheckpoint):
    """Class that allows saving extended `keras.Model`s."""

    @classmethod
    def from_model(cls, model: keras.Model, *, preprocessor=None):
        tempdir = tempfile.mkdtemp()
        model.save(tempdir, save_format="tf")

        checkpoint = cls.from_directory(tempdir)
        if preprocessor:
            checkpoint.set_preprocessor(preprocessor)
        checkpoint.update_metadata({cls.MODEL_FILENAME_KEY: tempdir})
        return checkpoint


def preprocess(
    x, preprocessor: Callable[np.ndarray, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocess `x`

    Args:
        x: CIC-IDS-2017 data as is (no modifications).
        preprocessor: Preprocessor function.

    Returns:
        Preprocessed `x` and protocol column separately (separated from `x`).
    """
    x = preprocessor(x).astype(np.float32)
    protocol = x[:, :1]
    x = x[:, 1:]
    return x, protocol


def inv_preprocess(
    x: np.ndarray,
    inv_preprocessor: Callable[np.ndarray, np.ndarray],
    protocol: np.ndarray,
) -> np.ndarray:
    """Inverse preprocess `x`.

    Args:
        x: Autoencoded data.
        inv_preprocessor: Inverse preprocessor function.
        protocol: Preprocessed protocol column.

    Returns:
        Inverse preprocessed data.
    """
    x = np.concatenate([protocol, x], axis=1)
    return inv_preprocessor(x)


def test_autoencoder(x: np.ndarray, autoencoder: AutoEncoder) -> np.ndarray:
    """Utility function for manually testing the autoencoder.

    Args:
        x: Data to be processed.
        autoencoder: Trained autoencoder to test.

    Returns:
        Preprocessed, autoencoded, and inverse preprocessed `x`.
    """
    x, protocol = preprocess(x, autoencoder["preprocessor"])
    x_autoencoder = autoencoder["decoder"](autoencoder["encoder"](x))
    return inv_preprocess(
        x_autoencoder, autoencoder["inv_preprocessor"], protocol
    )


def get_autoencoder(path) -> keras.Model:
    """Loads autoencoder from `path`."""
    return keras.models.load_model(path, compile=False)


def _create_argparser():
    parser = argparse.ArgumentParser(
        prog="train-autoencoder",
        description="Train autoencoder for cic-ids-2017 dataset",
    )
    parser.add_argument("trainset_path")
    parser.add_argument("preprocessor_path")
    parser.add_argument(
        "-o",
        "--output-path",
        default="output/CIC-IDS-autoencoder",
        help="Path where the autoencoder will be saved. (Default: output/CIC-IDS-autoencoder)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set will overwrite existing autoencoder.",
    )
    parser.add_argument(
        "--epochs",
        default=200,
        type=int,
        help="Define the amount of epochs for the autoencoders.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        help="Forces the tuning to use certain hidden dimension. If not set "
        "will search for the best hidden dim from range (50-200)",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        help="Forces the tuning to use certain latent dimension. If not set "
        "will search for the best latent dim from range (1-50)",
    )
    parser.add_argument(
        "--num-samples",
        default=1000,
        type=int,
        help="Number of times to sample from the hyperparameter search space.",
    )
    return parser


if __name__ == "__main__":
    parser = _create_argparser()
    args = parser.parse_args()
    autoencoder = _main(args)
