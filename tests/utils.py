# Copyright 2025 InstaDeep Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from loguru import logger
from numpy.typing import NDArray


def compare_dicts_with_tolerance(
    dict1: dict[str, NDArray[np.floating] | float | bool | None],
    dict2: dict[str, NDArray[np.floating] | float | bool | None],
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> bool:
    """
    Compare two dictionaries with values of types (NDArray[float, float, bool, None).

    Args:
        dict1: The first dictionary.
        dict2: The second dictionary.
        rtol: Relative tolerance for numerical comparison.
        atol: Absolute tolerance for numerical comparison.

    Returns:
        bool: True if the dictionaries are equivalent (up to tolerance), False otherwise.
    """
    # Check if keys are the same
    if dict1.keys() != dict2.keys():
        return False

    for key in dict1:
        value1 = dict1[key]
        value2 = dict2[key]

        if (not (isinstance(value1, (np.ndarray, float, bool)) or value1 is None)
                or not (isinstance(value2, (np.ndarray, float, bool)) or value2 is None)):
            raise ValueError(
                'Dict values must be of type (np.ndarray, float, bool, None). '
                f'Invalid types ({type(value1)}, {type(value2)}) for key = \'{key}\'.'
            )

        # Catch if types are not the same
        if type(value1) != type(value2):
            logger.info(
                f"Dict value types not equal for key = '{key}': ({type(value1)}, {type(value2)})."
            )
            return False

        # Types are same and are one of (array, float, bool, None). Handle None case:
        if value1 is None and value2 is None:
            continue

        # Types are same and are one of (array, float, bool). Handle all cases:
        if not np.allclose(value1, value2, atol=atol, rtol=rtol):
            logger.info(
                f"Dict values not equal for key = '{key}': ({value1}, {value2})."
            )
            return False

    return True
