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
from degym_tutorials.cstr_tutorial.cstr_utils import reaction_rate
from degym_tutorials.cstr_tutorial.physical_parameters import (
    CSTRPhysicalParameters,
)
from degym_tutorials.cstr_tutorial.state_concrete_classes import (
    CSTRDAEParameters,
    CSTRDAEState,
)
from degym_tutorials.cstr_tutorial.system_dynamics.diffeqpy_dynamics import (
    CSTRDiffeqpySystemDynamics,
)



def test_system_dynamics(physical_parameters: CSTRPhysicalParameters) -> None:
    """
    Check that derivates correctly computed and stored by CSTRDiffeqpySystemDynamics.

    For timestep 0 (t=0), with Q=0, we have: c_a = c_a_0, c_b=0., q=0, T=T0.
    Then:
        dc_a / dt = − k_a(t=0) *  c_a_0
        dc_b / dt = k_a(t=0) *  c_a_0
        dT / dt = (- dh * V * (k_a(t=0) *  c_a_0)) / (p * c_p * V)
    where
        k_a = k_0_a * exp(-e_a / r * T0)
        k_b = k_0_b * exp(-e_b / r * T0)
    """
    system_dynamics = CSTRDiffeqpySystemDynamics()

    k_a = reaction_rate(
        k_0=physical_parameters.k_0_a,
        e=physical_parameters.e_a,
        r=physical_parameters.R,
        t=physical_parameters.T_0
    )

    k_b = reaction_rate(
        k_0=physical_parameters.k_0_b,
        e=physical_parameters.e_b,
        r=physical_parameters.R,
        t=physical_parameters.T_0
    )

    initial_dae_state = CSTRDAEState(
        c_a=physical_parameters.c_a_0,
        c_b=0.0,
        T=physical_parameters.T_0,
    )
    initial_dae_parameters = CSTRDAEParameters(
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

    action = CSTRDAEAction(q=0.0)

    derivative = [0.0, 0.0, 0.0]
    time = np.array([])

    dc_a = -k_a * physical_parameters.c_a_0
    dc_b = +k_a * physical_parameters.c_a_0
    dt = (
        -physical_parameters.dh
        * physical_parameters.V
        * k_a
        * physical_parameters.c_a_0
        / (physical_parameters.p * physical_parameters.c_p * physical_parameters.V)
    )

    # Complete new derivative values
    system_dynamics(
        input_values=initial_dae_state.to_np_array(),
        parameters=np.concatenate(
            [initial_dae_parameters.to_np_array(), action.to_np_array()]
        ),
        derivative=derivative,
        time=time,
    )

    # derivative list should have been updated in-place
    np.testing.assert_array_almost_equal(
        derivative, [dc_a, dc_b, dt], decimal=16
    )
