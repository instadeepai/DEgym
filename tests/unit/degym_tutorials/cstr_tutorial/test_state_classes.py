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

from copy import deepcopy

import numpy as np

from degym_tutorials.cstr_tutorial.cstr_utils import reaction_rate
from degym_tutorials.cstr_tutorial.physical_parameters import CSTRPhysicalParameters
from degym_tutorials.cstr_tutorial.state_concrete_classes import (
    CSTRInitialStateGenerator,
    CSTRState,
    CSTRStatePostprocessor,
    CSTRStatePreprocessor,
    CSTRDAEState,
    CSTRDAEParameters,
    CSTRNonDAEParameters,
)


def test_cstrdaestate_to_numpy_array(cstr_dae_state: CSTRDAEState) -> None:
    array = cstr_dae_state.to_np_array()
    np.testing.assert_array_equal(array, [1, 2, 3])

def test_cstrdaestate_from_numpy_array(cstr_dae_state: CSTRDAEState) -> None:
    array = np.array([1, 2, 3])
    assert CSTRDAEState.from_np_array(array) == cstr_dae_state


def test_cstrdaeparams_to_numpy_array(cstr_dae_params: CSTRDAEParameters) -> None:
    array = cstr_dae_params.to_np_array()
    np.testing.assert_array_equal(array, [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])

def test_cstrdaeparams_from_numpy_array(cstr_dae_params: CSTRDAEParameters) -> None:
    array = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
    assert CSTRDAEParameters.from_np_array(array) == cstr_dae_params


def test_cstrnondaeparams_to_numpy_array(cstr_non_dae_params: CSTRNonDAEParameters) -> None:
    array = cstr_non_dae_params.to_np_array()
    np.testing.assert_array_equal(array, [19, 20, 21])

def test_cstrnondaeparams_from_numpy_array(cstr_non_dae_params: CSTRNonDAEParameters) -> None:
    array = np.array([19, 20, 21])
    assert CSTRNonDAEParameters.from_np_array(array) == cstr_non_dae_params

def test_deepcopy(cstr_state: CSTRState) -> None:
    new_state = deepcopy(cstr_state)
    assert id(cstr_state) != id(new_state)
    assert cstr_state == new_state


def test_state_preprocessor(cstr_state: CSTRState) -> None:
    preprocessor = CSTRStatePreprocessor()  # smoke test
    assert preprocessor.preprocess_state(cstr_state) == cstr_state


def test_state_postprocessor(cstr_state: CSTRState) -> None:
    postprocessor = CSTRStatePostprocessor()  # smoke test
    assert postprocessor.postprocess_state(cstr_state) == cstr_state



def test_initial_state_sampler() -> None:
    physical_parameters = CSTRPhysicalParameters()

    # Create the expected state structure that matches what CSTRInitialStateGenerator produces
    expected_dae_state = CSTRDAEState(
        c_a=physical_parameters.c_a_0,
        c_b=0.0,
        T=physical_parameters.T_0,
    )

    expected_dae_params = CSTRDAEParameters(
        F=physical_parameters.F,
        V=physical_parameters.V,
        c_a_0=physical_parameters.c_a_0,
        p=physical_parameters.p,
        c_p=physical_parameters.c_p,
        T_0=physical_parameters.T_0,
        dh=physical_parameters.dh,
        k_0_a=physical_parameters.k_0_a,
        k_0_b=physical_parameters.k_0_b,
        E_a_A=physical_parameters.e_a,
        E_a_B=physical_parameters.e_b,
        R=physical_parameters.R,
    )

    expected_non_dae_params = CSTRNonDAEParameters(
        q_max=physical_parameters.q_max,
        max_timestep=physical_parameters.max_timestep,
        timestep=0,
    )

    expected_state = CSTRState(
        dae_state=expected_dae_state,
        dae_params=expected_dae_params,
        non_dae_params=expected_non_dae_params
    )

    generator = CSTRInitialStateGenerator()
    for _ in range(3):
        sampled_state = generator.generate(physical_parameters)
        np.testing.assert_array_equal(
            expected_state.to_np_array(), sampled_state.to_np_array()
        )
