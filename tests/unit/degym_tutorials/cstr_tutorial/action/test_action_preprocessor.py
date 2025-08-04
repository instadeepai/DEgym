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

import gymnasium as gym
from degym_tutorials.cstr_tutorial.action_concrete_classes import (
    CSTRActionConvertor,
    CSTRActionPreprocessor,
    CSTRActionRegulator,
)
from degym_tutorials.cstr_tutorial.physical_parameters import (
    CSTRPhysicalParameters,
)
from degym_tutorials.cstr_tutorial.state_concrete_classes import CSTRState


def test_action_preprocessor_does_not_smoke(
    physical_parameters: CSTRPhysicalParameters,
) -> None:
    action_preprocessor = CSTRActionPreprocessor(
        action_convertor=CSTRActionConvertor(), action_regulator=CSTRActionRegulator()
    )

    assert isinstance(action_preprocessor.action_space, gym.spaces.Box)
    assert action_preprocessor.action_space.low == -1
    assert action_preprocessor.action_space.high == 1


@pytest.mark.parametrize("q_normalized", [1.0, 2.0])
def test_preprocess_action(
    cstr_state: CSTRState,
    physical_parameters: CSTRPhysicalParameters,
    q_normalized: float,
) -> None:
    action_regulator = CSTRActionRegulator()
    action_convertor = CSTRActionConvertor()

    action_preprocessor = CSTRActionPreprocessor(
        action_convertor=action_convertor, action_regulator=action_regulator
    )

    if q_normalized == 1.0:  # legal action
        preprocessed_action = action_preprocessor.preprocess_action(
            q_normalized, cstr_state
        )
        assert preprocessed_action.q == physical_parameters.q_max
    else:  # illegal action
        preprocessed_action = action_preprocessor.preprocess_action(
            q_normalized, cstr_state
        )
        assert preprocessed_action.q == physical_parameters.q_max
