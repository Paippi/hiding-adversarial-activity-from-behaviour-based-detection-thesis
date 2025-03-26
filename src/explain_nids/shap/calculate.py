#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Samuli Piippo https://github.com/Paippi/

import pickle
import argparse
from functools import partial
import sys

import shap
import pandas as pd
import numpy as np


def predict(array, columns, model):
    df = pd.DataFrame(array, columns=columns)
    return -model.decision_function(df)


def pick_first_n(dataset, label, amount):
    return dataset[dataset["Y"] == label][:amount]


def pick_first_n_each(dataset, amount):
    for label in np.unique(dataset["Y"]):
        yield pick_first_n(dataset, label, amount)


def main(argv=sys.argv):
    parser = _create_argparser()
    args = parser.parse_args(argv[1:])
    _main(args)


def _main(args):
    with open(args.anomaly_detector, "rb") as f:
        anomaly_detector = pickle.load(f)

    flows = pd.read_csv(args.filename)

    flow_samples = pd.concat(pick_first_n_each(flows, 200))

    labels = flow_samples.pop("Y")

    predict_ = partial(
        predict, columns=flow_samples.columns, model=anomaly_detector
    )
    explainer = shap.Explainer(predict_, flow_samples)

    shap_values = explainer(flow_samples)

    with open(args.output_path, "wb") as f:
        pickle.dump(shap_values, f)
        print(f"shapley values written in {args.output_path}")

    return shap_values


def _create_argparser():
    parser = argparse.ArgumentParser(
        prog="shap-calculate",
        description="Calculate shapley values.",
    )
    parser.add_argument("filename", help="Path to the flow file.")
    parser.add_argument(
        "anomaly_detector", help="Path to the anomaly detector pickle file."
    )
    parser.add_argument("--output-path", default="output/shap_values.p")

    return parser


if __name__ == "__main__":
    parser = _create_argparser()
    args = parser.parse_args()
    shap_values = _main(args)
