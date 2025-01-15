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
