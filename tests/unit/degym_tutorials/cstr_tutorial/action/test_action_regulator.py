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

from degym_tutorials.cstr_tutorial.action_concrete_classes import (
    CSTRActionRegulator,
    CSTRDAEAction,
)
from degym_tutorials.cstr_tutorial.physical_parameters import (
    CSTRPhysicalParameters,
)
from degym_tutorials.cstr_tutorial.state_concrete_classes import CSTRState


def test_is_legal(
    cstr_state: CSTRState,
    physical_parameters: CSTRPhysicalParameters,
) -> None:
    action_regulator = CSTRActionRegulator()

    legal_action = CSTRDAEAction(q=physical_parameters.q_max)
    assert action_regulator.is_legal(legal_action, cstr_state)

    illegal_action = CSTRDAEAction(q=physical_parameters.q_max + 1)
    assert not action_regulator.is_legal(illegal_action, cstr_state)

    illegal_action = CSTRDAEAction(q=-1)
    assert not action_regulator.is_legal(illegal_action, cstr_state)


def test_convert_to_legal_action(physical_parameters: CSTRPhysicalParameters, cstr_state: CSTRState) -> None:
    action_regulator = CSTRActionRegulator()
    action = CSTRDAEAction(q=physical_parameters.q_max + 1)
    legal_action = action_regulator.convert_to_legal_action(action, state=cstr_state)
    assert legal_action.q == physical_parameters.q_max
