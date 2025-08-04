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

from typing import Any

import pytest

from degym_tutorials.cstr_tutorial.physical_parameters import CSTRPhysicalParameters
from degym_tutorials.cstr_tutorial.state_concrete_classes import CSTRState, CSTRDAEState, \
    CSTRDAEParameters, CSTRNonDAEParameters


@pytest.fixture
def physical_parameters_config() -> dict[str, Any]:
    return {
        "fixed_values": {
            "f": 6,
            "v": 7,
            "c_a_0": 8,
            "p": 9,
            "c_p": 10,
            "t_0": 11,
            "dh": 13,
            "k_0_a": 14,
            "k_0_b": 15,
            "e_a": 16,
            "e_b": 17,
            "r": 18,
            "q_max": 19,
            "max_timestep": 3,
            },
        "sampled_values": {},
    }


@pytest.fixture
def physical_parameters(physical_parameters_config: dict) -> CSTRPhysicalParameters:
    fixed_values = physical_parameters_config["fixed_values"]
    return CSTRPhysicalParameters(
        p=fixed_values["p"],
        c_a_0=fixed_values["c_a_0"],
        c_p=fixed_values["c_p"],
        e_a=fixed_values["e_a"],
        e_b=fixed_values["e_b"],
        F=fixed_values["f"],
        dh=fixed_values["dh"],
        k_0_a=fixed_values["k_0_a"],
        k_0_b=fixed_values["k_0_b"],
        R=fixed_values["r"],
        T_0=fixed_values["t_0"],
        V=fixed_values["v"],
        q_max=fixed_values["q_max"],
        max_timestep=fixed_values["max_timestep"],
    )


@pytest.fixture
def cstr_dae_state() -> CSTRDAEState:
    return CSTRDAEState(c_a=1, c_b=2, T=3)

@pytest.fixture
def cstr_dae_params() -> CSTRDAEParameters:
    return CSTRDAEParameters(
        F=6,
        V=7,
        c_a_0=8,
        p=9,
        c_p=10,
        T_0=11,
        dh=12,
        k_0_a=13,
        k_0_b=14,
        E_a_A=15,
        E_a_B=16,
        R=17
    )


@pytest.fixture
def cstr_non_dae_params() -> CSTRNonDAEParameters:
    return CSTRNonDAEParameters(q_max=19, max_timestep=20, timestep=21)

@pytest.fixture
def cstr_state(
        cstr_dae_state: CSTRDAEState,
        cstr_dae_params: CSTRDAEParameters,
        cstr_non_dae_params: CSTRNonDAEParameters
) -> CSTRState:
    return CSTRState(
        dae_state=cstr_dae_state,
        dae_params=cstr_dae_params,
        non_dae_params=cstr_non_dae_params
    )
