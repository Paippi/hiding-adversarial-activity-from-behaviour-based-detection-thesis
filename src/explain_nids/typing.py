#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Samuli Piippo https://github.com/Paippi/

"""Module containing typing help."""
# Standard imports
from typing import Union

# 3rd party imports
import numpy.typing as npt
import numpy as np
import pandas as pd

# Local application/library specific imports

NPFloatOrInt = Union[np.float64, np.int_]
NDArrayOfIntsOrFloats = npt.NDArray[NPFloatOrInt]
NDArrayOrDataFrame = Union[NDArrayOfIntsOrFloats, pd.DataFrame]

OptionalInt = Union[None, int]
