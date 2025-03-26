#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Samuli Piippo https://github.com/Paippi/

import argparse
import pickle
import sys
import os

import shap
from matplotlib import pyplot as plt


def main(argv=sys.argv):
    parser = _create_argparser()
    args = parser.parse_args(argv[1:])
    _main(args)


def _main(args):
    output_dir = args.output_dir
    max_display = args.show_top
    with open(args.filename, "rb") as f:
        shap_values = pickle.load(f)

    shap_values *= args.coefficient

    if args.global_feature_importance:
        fig = plt.figure()
        axis = shap.plots.bar(shap_values, show=False, max_display=max_display)
        axis.set_xlabel("Shapley values x 100,000")

        fname = os.path.join(
            output_dir,
            f"global-feature-importance-plot-top-{max_display}.png",
        )
        fig.savefig(
            fname,
            bbox_inches="tight",
            dpi=68,
        )
        print(f"Global feature importance plot saved in {fname}")

    if args.beeswarm:
        fig = plt.figure()
        axis = shap.plots.beeswarm(
            shap_values, show=False, max_display=max_display
        )
        axis.set_xlabel("Shapley values x 100,000")

        fname = os.path.join(
            output_dir,
            f"beeswarm-top-{max_display}.png",
        )

        fig.savefig(
            fname,
            bbox_inches="tight",
            dpi=68,
        )
        print(f"Beeswarm plot saved in {fname}")

    if args.waterfall:
        fig = plt.figure()
        axis = shap.plots.waterfall(
            shap_values[args.sample_index], show=False, max_display=max_display
        )
        axis.set_xlabel("Shapley values x 100,000")

        fname = os.path.join(output_dir, f"waterfall-top-{max_display}.png")
        fig.savefig(
            fname,
            bbox_inches="tight",
            dpi=68,
        )
        print(f"Waterfall plot saved in {fname}")


def _create_argparser():
    parser = argparse.ArgumentParser(
        prog="shap-draw",
        description="Draw plots from shapley values.",
    )
    parser.add_argument("--filename", default="output/shap_values.p")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--coefficient", type=int, default=100_000)
    parser.add_argument(
        "-g", "--global-feature-importance", action="store_true"
    )
    parser.add_argument("-b", "--beeswarm", action="store_true")
    parser.add_argument("-w", "--waterfall", action="store_true")
    parser.add_argument("--show-top", type=int, default=5)
    parser.add_argument("--output-dir", default="images")

    return parser


if __name__ == "__main__":
    parser = _create_argparser()
    args = parser.parse_args()
    _main(args)
