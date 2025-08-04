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

from degym_tutorials.cstr_tutorial.action_concrete_classes import CSTRDAEAction
from degym_tutorials.cstr_tutorial.physical_parameters import (
    CSTRPhysicalParameters,
)
from degym_tutorials.cstr_tutorial.state_concrete_classes import CSTRDAEState, CSTRDAEParameters
from degym_tutorials.cstr_tutorial.system_dynamics.diffeqpy_dynamics import (
    CSTRDiffeqpySystemDynamics,
)
from degym.integrators import DiffeqpyIntegrator, DiffeqpyIntegratorConfig, TimeSpan

from tests import skip_if_not_diffeqpy


@skip_if_not_diffeqpy
def test_diffeqpy_integrate() -> None:
    start_time, action_duration = (1.0, 15.0)
    time_span = TimeSpan(start_time=start_time, end_time=start_time + action_duration)
    physical_parameters = CSTRPhysicalParameters()
    system_dynamics = CSTRDiffeqpySystemDynamics()
    integrator_config = DiffeqpyIntegratorConfig(action_duration=action_duration)
    integrator = DiffeqpyIntegrator(
        system_dynamics=system_dynamics,
        integrator_config=integrator_config,
    )

    dae_state = CSTRDAEState(
        c_a=physical_parameters.c_a_0,
        c_b=0.0,
        T=physical_parameters.T_0,
    )

    dae_params = CSTRDAEParameters(
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

    dae_action = CSTRDAEAction(q=1500.)
    next_dae_state = integrator.integrate(
        input_values=dae_state.to_np_array(),
        parameters=dae_params.to_np_array(),
        action=dae_action.to_np_array(),
        time_span=time_span,
    )
    next_dae_state = CSTRDAEState.from_np_array(next_dae_state)

    assert 0.0 <= next_dae_state.c_a < physical_parameters.c_a_0
    assert 0.0 < next_dae_state.c_b < physical_parameters.c_a_0
    np.testing.assert_allclose(
        next_dae_state.c_a + next_dae_state.c_b,
        physical_parameters.c_a_0,
        atol=1e-6,
        rtol=0.0,
    )
    assert next_dae_state.T > physical_parameters.T_0
