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
    CSTRAction,
    CSTRActionConverter,
    CSTRDAEAction,
)

from degym_tutorials.cstr_tutorial.state_concrete_classes import CSTRState


def test_action_to_dae_action(cstr_state: CSTRState) -> None:
    action_converter = CSTRActionConverter()

    action = CSTRAction(q_normalized=1.0)
    dae_action = action_converter.action_to_dae_action(action, cstr_state)

    assert isinstance(dae_action, CSTRDAEAction)
    assert dae_action.q == action.q_normalized * cstr_state.non_dae_params.q_max


def test_dae_action_to_action(cstr_state: CSTRState) -> None:
    action_converter = CSTRActionConverter()

    dae_action = CSTRDAEAction(q=cstr_state.non_dae_params.q_max)
    action = action_converter.dae_action_to_action(dae_action, cstr_state)

    assert isinstance(action, CSTRAction)
    assert action.q_normalized == +1
