#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Samuli Piippo https://github.com/Paippi/

"""Module for explaining counterfactuals."""

# Standard imports
from typing import List

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


def highlight_differences(
    original: NDArrayOfIntsOrFloats,
    counterfactual: NDArrayOfIntsOrFloats,
    columns: List[str],
) -> Styler:
    """Creates a styler that highlights differences between original and the counterfactual.

    Args:
        original: Original sample.
        counterfactual: Generated counterfactual.
        columns: Column names.

    Returns:
        A Pandas styler that can be used to visualize the differences between
        the original sample and the counterfactual sample.
    """
    diffs = counterfactual - original
    df = pd.DataFrame(
        [original, counterfactual, diffs],
        columns=columns,
        index=["Original", "Destination", "Diff"],
    ).T
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
