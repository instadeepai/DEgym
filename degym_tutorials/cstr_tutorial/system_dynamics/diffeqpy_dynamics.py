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
from numpy.typing import NDArray
from degym.system_dynamics import DiffeqpySystemDynamicsFn

from degym_tutorials.cstr_tutorial.action_concrete_classes import CSTRDAEAction
from degym_tutorials.cstr_tutorial.cstr_utils import reaction_rate
from degym_tutorials.cstr_tutorial.state_concrete_classes import (
    CSTRDAEParameters,
    CSTRDAEState,
)


class CSTRDiffeqpySystemDynamics(DiffeqpySystemDynamicsFn):  # noqa: D101
    # state has 3 values: {c_a, c_b, t}
    # We have differential equations for all 3 values => Identity matrix
    mass_matrix = np.eye(3, 3)

    @staticmethod
    def __call__(
        derivative: list[float],
        input_values: NDArray[np.floating],
        parameters: NDArray[np.floating],
        time: NDArray[np.floating],
    ) -> None:
        """
        We implement the DAE describing the dynamics of the CSTR problem.

        With:
            k_a = k_0_a * exp(-e_a / rt)
            k_b = k_0_b * exp(-e_b / rt)

        Differential equations:
            (1): dc_a/dt = (F / V) * (c_a0 − c_a)− k_a c_a  + k_b * c_b
            (2): dc_b/dt = (F / V) * (− c_b) + k_a c_a  - k_b * c_b
            (3): dT/dt = (F * p * c_p (T_0 - T) + q - dh * V
                            * (k_a * c_a - k_b * c_b)) / (p * c_p * V)
        """
        s = CSTRDAEState.from_np_array(input_values)  # dae stae
        pm = CSTRDAEParameters.from_np_array(parameters[:-1])  # dae parameters
        a = CSTRDAEAction.from_np_array(parameters[-1:])  # dae action

        k_a = reaction_rate(k_0=pm.k_0_a, e=pm.E_a_A, r=pm.R, t=s.T)
        k_b = reaction_rate(k_0=pm.k_0_b, e=pm.E_a_B, r=pm.R, t=s.T)

        # Differential equations.
        derivative[0] = (
            (pm.F / pm.V) * (pm.c_a_0 - s.c_a) - (k_a * s.c_a) + (k_b * s.c_b)
        )  # d[A]/dt
        derivative[1] = (pm.F / pm.V) * (-s.c_b) + (k_a * s.c_a) - (k_b * s.c_b)  # d[B]/dt
        derivative[2] = (
            pm.F * pm.p * pm.c_p * (pm.T_0 - s.T) + a.q - pm.dh * pm.V * (k_a * s.c_a - k_b * s.c_b)
        ) / (pm.p * pm.c_p * pm.V)  # dT/dt
