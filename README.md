# Hiding Adversarial Activity from Behaviour-Based Detection

This repository contains code for the Thesis "Hiding Adversarial Activity from Behaviour-Based Detection"

## Install

The package uses pipenv+pyenv to manage requirements. See pipenv and pyenv installation guides.

```
$ pipenv install .
```

## Development Installation

```
$ pipenv install --dev -e .
```

## Usage

### train-autoencoder

```
$ train-autoencoder --help
usage: train-autoencoder [-h] [-o OUTPUT_PATH] [--overwrite] [--epochs EPOCHS] [--hidden-dim HIDDEN_DIM] [--latent-dim LATENT_DIM] [--num-samples NUM_SAMPLES] trainset_path preprocessor_path

Train autoencoder for cic-ids-2017 dataset

positional arguments:
  trainset_path
  preprocessor_path

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT_PATH, --output-path OUTPUT_PATH
                        Path where the autoencoder will be saved. (Default: output/CIC-IDS-autoencoder)
  --overwrite           If set will overwrite existing autoencoder.
  --epochs EPOCHS       Define the amount of epochs for the autoencoders.
  --hidden-dim HIDDEN_DIM
                        Forces the tuning to use certain hidden dimension. If not set will search for the best hidden dim from range (50-200)
  --latent-dim LATENT_DIM
                        Forces the tuning to use certain latent dimension. If not set will search for the best latent dim from range (1-50)
  --num-samples NUM_SAMPLES
                        Number of times to sample from the hyperparameter search space.
```

#### Example

```
$ train-autoencoder data/test.csv models/stage1_ocsvm_scaler.p -o output/CIC-IDS-autoencoder --epochs 200 --hidden-dim 187 --latent-dim 49 --num-samples 1
```

### train-cfrl

```
usage: train-cfrl [-h] [--latent-dim LATENT_DIM] [--steps STEPS] [--coeff-sparsity COEFF_SPARSITY] [--coeff-consistency COEFF_CONSISTENCY] [--batch-size BATCH_SIZE] [--anomaly-threshold ANOMALY_THRESHOLD]
                  [--classification] [--num-samples NUM_SAMPLES] [--label-column LABEL_COLUMN] [--experiment-name EXPERIMENT_NAME] [--output_dir OUTPUT_DIR]
                  anomaly_detector_path dataset_path autoencoder_path preprocessor_path

Train counterfactual reinforcement learning model.

positional arguments:
  anomaly_detector_path
                        Path to the anomaly detector pickle file containing any model that gives anomaly score, using function `decision_function`.
  dataset_path          Path to the dataset used to train and validate the CFRL model.
  autoencoder_path      Path to the Keras autoencoder
  preprocessor_path     Path to the preprocessor pickle file.

optional arguments:
  -h, --help            show this help message and exit
  --latent-dim LATENT_DIM
                        Forces the tuning to use certain latent dimension. Needs to be the same as for the autoencoder.
  --steps STEPS         Set number of steps (default 100_000)
  --coeff-sparsity COEFF_SPARSITY
                        If set will force model's to use given coeff sparsity (if not specified will search for best coeff sparsity)
  --coeff-consistency COEFF_CONSISTENCY
                        If set will force model's to use given coeff consistency (if not specified will search for best coeff consistency)
  --batch-size BATCH_SIZE
  --anomaly-threshold ANOMALY_THRESHOLD
                        Set the anomaly threshold, ignored if --classification flag set.
  --classification      Flag that if set, will use classification reward system instead of regression
  --num-samples NUM_SAMPLES
                        Number of times to sample from the hyperparameter search space.
  --label-column LABEL_COLUMN
                        Column name containing the sample's label/target.
  --experiment-name EXPERIMENT_NAME
                        Name for the experiment (default CIC-IDS-2017-explainer)
  --output_dir OUTPUT_DIR
                        Path where to save the best result. The final path will be <output dir>/<experiment name>-<regression/classification depending on the task>. (default: output).
```

#### Example

```
$ train-cfrl models/stage1_ocsvm.p data/all.csv output/CIC-IDS-autoencoder models/stage1_ocsvm_scaler.p --latent-dim 49 --num-samples 1 --coeff-consistency 0.18 --coeff-sparsity 0.1 --label-column="Label" --classification
```

### Generating Counterfactuals

```python
>>> from alibi.saving import load_explainer
>>> from explain_nids.predict import get_anomaly_detector
>>> import pandas as pd
>>> import numpy as np
>>> import warnings
>>> from explain_nids.compare import highlight_differences
>>> anomaly_detector_path = "models/stage1_ocsvm.p"
>>> classification = True
>>> anomaly_threshold = -0.0002196942507948895
>>> explainer_path = "output/CIC-IDS-2017-explainer-classification/explainer"
>>> anomaly_detector = get_anomaly_detector(
...     anomaly_detector_path,
...     classification,
...     anomaly_threshold,
... )
>>> explainer = load_explainer(explainer_path, anomaly_detector)
>>> samples = pd.read_csv("data/test.csv")
>>> # Separate benign and malicious samples
>>> benign_mask = samples["Y"] == "Benign"
>>> samples.pop("Y")
>>> benign_samples = samples[benign_mask]
>>> attack_samples = samples[~benign_mask]
>>> benign_target = np.array([1])
>>> attack_target = np.array([0])
>>> # Ignore feature name warnings...
>>> with warnings.catch_warnings():
...     warnings.simplefilter("ignore")
...     benign_cf = explainer.explain(benign_samples, benign_target)
>>> with warnings.catch_warnings():
...     warnings.simplefilter("ignore")
...     attack_cf = explainer.explain(attack_samples, attack_target)
>>> # Highlight differences between CF and the original sample (not required but
>>> # visually more appeling)
>>> sample_index = 42
>>> diffs = highlight_differences(attack_cf.orig["X"][sample_index], attack_cf.cf["X"][sample_index], samples.columns)
>>> diffs.to_html("cf.html")
```

Now open the cf.html in a web browser to observe the differences.

### Shap Values

First calculate the shap values for the model (this takes a long time).

```
$ shap-calculate data/test.csv models/stage1_ocsvm.p
```

Then create different graphs based on needs (see help for more options).

```
$ shap-draw --waterfall --global --beeswarm
```

### Testing Different MTUs

First create the PCAPs.

```
$ cd scripts
$ ./mtu_test.sh
```

Then convert the PCAPs to flows.

```
$ ./convert_to_flow.sh
```

To visualize the difference between the flows use `diff` subcommand (optional)

```
$ flow-diff output/flows/curl_mtu_1500.pcap_Flow.csv output/flows/curl_mtu_68.pcap_Flow.csv diff --remove-no-diff
```

To compare the anomaly scores between the two flows use score `score` subcommand

```
$ flow-diff output/flows/curl_mtu_1500.pcap_Flow.csv output/flows/curl_mtu_68.pcap_Flow.csv score
```


## TODOs

### Setup Documentation

from package root directory (e.g. /libraries/package-name/) change

1. from docs/index.md change the title

        Package Name Documentation

2. from docs/conf.py change package name & project

        from <your_package> import (
        ...
        project = "Package Name"

3. Generate api documentation pages

        sphinx-apidoc --tocfile api_ref src/package_name -M -o docs

    NOTE: you might have to run the above command in case of you package structure changes.

    This command should create api_ref.rst and file from each package within your src/ dir.

4. Exclude Possible Imports

    For example your package might have class like foo.bar.Bar. But you have imported the Bar to be within foo's context to improve the api.

    * open `<package>`.rst where import happens.

            vim foo.rst

    * Exclude the member

            .. automodule:: foo.bar
            :members:
            :undoc-members:
            :show-inheritance:
            :exclude-members: Bar

---

#### Test Document Generation

cd to docs and generate documentation

    cd docs/
    make html

Open the webpage in your browser docs/_build/html/index.html

Navigate around and assure that you can find api documentation and it doesn't have duplicates. (Duplicates can happen when you import modules/functions/classes from your package)

See Exclude Possible Imports for guide to how to exclude.

---

### Testing 

Testing is done by simply running pytest ([documentation](https://docs.pytest.org/en/6.2.x/contents.html))

    $ pytest

Tests are usually run while developing code. If that's the case, it is encouraged to install the package in development (editable) mode.

    pip install -e <path to package root (usually just ".")>

To actually test anything you need to have tests in a folder `tests/`.

#### Setup testing of multiple python versions

Add the following lines in the `pyproject.toml`.

    [tool.tox]
    legacy_tox_ini = """
    [tox]
    # envlist = <tested versions>
    envlist = py36, py37, py38, py39
    [testenv]
    deps = -rdev-requirements.txt
    commands = pytest
    """

At the time being, tox prioritizes pyproject.toml, but doesn't completely support toml format. Thus usage of legacy_tox_ini.

### Finally

Replace `README.md` with description of your package

---
