#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Samuli Piippo https://github.com/Paippi/

"""Module for explaining counterfactuals."""

# Standard imports
import argparse
from typing import List
import sys
import pickle

# 3rd party imports
import pandas as pd
from pandas.io.formats.style import Styler

# Local application/library specific imports
from explain_nids.typing import NDArrayOfIntsOrFloats


def _choose_diff_color(column):
    styles = ["", ""]
    diff_ = column["Diff"]
    if diff_ == 0:
        styles.append("")
    elif diff_ > 0:
        styles.append("background: #99ff99")
    else:
        styles.append("background: #99ccff")
    return styles


def _remove_no_diff(df):
    return df[df["Diff"] != 0]


def highlight_differences(
    flow_a: NDArrayOfIntsOrFloats,
    flow_b: NDArrayOfIntsOrFloats,
    columns: List[str],
    first_flow_name="Original",
    second_flow_name="Counterfactual",
    remove_no_diff=False,
) -> Styler:
    """Creates a styler that highlights differences between `flow_a` and `flow_b`.

    Args:
        flow_a: First flow.
        flow_b: Second flow.
        columns: Column names.

    Returns:
        A Pandas styler that can be used to visualize the differences between
        the `flow_a` and `flow_b`.
    """
    diffs = flow_b - flow_a
    df = pd.DataFrame(
        [flow_a, flow_b, diffs],
        columns=columns,
        index=[first_flow_name, second_flow_name, "Diff"],
    ).T
    if remove_no_diff:
        df = _remove_no_diff(df)

    styler = df.style.apply(_choose_diff_color, axis=1)
    styler.set_properties(
        **{
            "text-align": "right",
            "padding-left": "15px",
            "padding-right": "15px",
        }
    )
    styler.set_table_styles(
        [
            {
                "selector": "th",
                "props": [("text-align", "right"), ("padding-right", "15px")],
            }
        ]
    )
    return styler.format("{:.2f}")


def _remove_flow_specific_information(flow):
    flow.pop("Src IP")
    flow.pop("Src Port")
    flow.pop("Dst IP")
    flow.pop("Dst Port")
    flow.pop("Protocol")
    flow.pop("Timestamp")


def main(argv=sys.argv):
    parser = _create_argparser()
    args = parser.parse_args(argv[1:])
    _main(args)


def _main(args):
    file_a = pd.read_csv(args.file_a)
    file_b = pd.read_csv(args.file_b)

    return args.func(file_a, file_b, args)


def _diff(file_a, file_b, args):
    # Remove Labels from the comparison if they exist
    file_a.pop("Label")
    file_b.pop("Label")

    # Separate ids from comparison
    flow_id_a = file_a.pop("Flow ID")
    flow_id_b = file_b.pop("Flow ID")

    # Remove flow specific information
    _remove_flow_specific_information(file_a)
    _remove_flow_specific_information(file_b)

    diffs = highlight_differences(
        file_a.iloc[args.index_a].to_numpy(),
        file_b.iloc[args.index_b].to_numpy(),
        file_a.columns,
        flow_id_a.iloc[args.index_a],
        flow_id_b.iloc[args.index_b],
        args.remove_no_diff,
    )
    diffs.to_html(args.output)
    print(f"Diffs saved to {args.output}")
    return diffs


def _score_diff(file_a, file_b, args):
    with open(args.anomaly_detector, "rb") as f:
        anomaly_detector = pickle.load(f)

    flows_a = convert_columns_for_anomaly_detector(file_a)[
        anomaly_detector.feature_names_in_
    ]
    flows_b = convert_columns_for_anomaly_detector(file_b)[
        anomaly_detector.feature_names_in_
    ]

    sample_a = flows_a.iloc[args.index_a]
    sample_b = flows_b.iloc[args.index_b]

    result_a = -anomaly_detector.decision_function([sample_a])[0]
    result_b = -anomaly_detector.decision_function([sample_b])[0]

    diff = result_a - result_b
    print(diff)
    return diff


def convert_columns_for_anomaly_detector(df):
    mapping = {
        "Protocol": "Protocol",
        "Flow Duration": "Flow Duration",
        "Total Fwd Packet": "Total Fwd Packets",
        "Total Bwd packets": "Total Backward Packets",
        "Total Length of Fwd Packet": "Fwd Packets Length Total",
        "Total Length of Bwd Packet": "Bwd Packets Length Total",
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
        "Packet Length Min": "Packet Length Min",
        "Packet Length Max": "Packet Length Max",
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
        "Fwd Segment Size Avg": "Avg Fwd Segment Size",
        "Bwd Segment Size Avg": "Avg Bwd Segment Size",
        "Subflow Fwd Packets": "Subflow Fwd Packets",
        "Subflow Fwd Bytes": "Subflow Fwd Bytes",
        "Subflow Bwd Packets": "Subflow Bwd Packets",
        "Subflow Bwd Bytes": "Subflow Bwd Bytes",
        "FWD Init Win Bytes": "Init Fwd Win Bytes",
        "Bwd Init Win Bytes": "Init Bwd Win Bytes",
        "Fwd Act Data Pkts": "Fwd Act Data Packets",
        "Fwd Seg Size Min": "Fwd Seg Size Min",
        "Active Mean": "Active Mean",
        "Active Std": "Active Std",
        "Active Max": "Active Max",
        "Active Min": "Active Min",
        "Idle Mean": "Idle Mean",
        "Idle Std": "Idle Std",
        "Idle Max": "Idle Max",
        "Idle Min": "Idle Min",
    }
    return df.rename(columns=mapping)


def _create_argparser():
    parser = argparse.ArgumentParser(
        prog="compare-flows",
        description="Draw plots from shapley values.",
    )
    parser.add_argument("file_a", help="First file to compare")
    parser.add_argument("file_b")
    parser.add_argument(
        "--index-a",
        default=0,
        type=int,
        help="index of the flow in `file_a` default 0",
    )
    parser.add_argument(
        "--index-b",
        default=0,
        type=int,
        help="index of the flow in `file_b` default 0",
    )
    subparsers = parser.add_subparsers(help="Choose subcommand")

    diff_parser = subparsers.add_parser(
        "diff", help="Show diff between flows."
    )
    diff_parser.add_argument("--remove-no-diff", "-r", action="store_true")
    diff_parser.add_argument("--output", default="output/diff.html")
    diff_parser.add_argument("--label-column-name", default="Label")
    diff_parser.set_defaults(func=_diff)

    score_parser = subparsers.add_parser(
        "score", help="Calculate difference between flows' anomaly score."
    )
    score_parser.add_argument(
        "--anomaly-detector",
        default="models/stage1_ocsvm.p",
        help="Path to the anomaly detector .pickle file.",
    )
    score_parser.set_defaults(func=_score_diff)

    return parser


if __name__ == "__main__":
    parser = _create_argparser()
    args = parser.parse_args()
    res = _main(args)
