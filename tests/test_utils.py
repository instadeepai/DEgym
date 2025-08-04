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

import pytest
from typing import Any
import numpy as np

from tests.utils import compare_dicts_with_tolerance


@pytest.mark.parametrize(
    "dict1, dict2, expected",
    [
        ({"a": 1.0}, {"a": 1.0}, True),
        ({"a": 1.0}, {"b": 1.0}, False),
        ({"a": 1.0}, {"a": 1.0, "b": None}, False),
        ({"b": None, "a": 1.0}, {"a": 1.0, "b": None}, True),
        ({"a": np.array([0.0, 1.0])}, {"a": np.array([0.0 - 1e-8, 1.0 + 1e-8])}, True),
    ]
)
def test_compare_dicts_with_tolerance(
    dict1: dict[str, Any],
    dict2: dict[str, Any],
    expected: bool
) -> None:
    output = compare_dicts_with_tolerance(dict1, dict2)
    assert output == expected
