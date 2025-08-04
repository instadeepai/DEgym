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

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from degym.system_dynamics.base import SystemDynamicsFn


class DiffeqpySystemDynamicsFn(SystemDynamicsFn, ABC):
    """
    Implements the system dynamics equations in a way that is suitable for use
    with the diffeqpy package for solving differential equations.
        https://github.com/SciML/diffeqpy

    For setting the mass_matrix variable:
        Equations describing DAE can be written using mass matrix, M, as:
            M(x(t), t) x'(t) = f(x(t), t)

        where (with N = num equations, D = num system variables):
            M = (N, D) mass matrix for terms containing time derivatives
            x = (D, 1) vector containing system variables, x_i(t)
            x' = (D, 1) vector of time derivatives of each variable, dx_i(t) / dt
            f = (N, 1) vector of functions for completing each equation
    """

    mass_matrix: NDArray[np.floating]

    @staticmethod
    @abstractmethod
    def __call__(  # NOTE: This is arg order assumed by diffeqpy - do not change
        derivative: list[float],
        input_values: NDArray[np.floating],
        parameters: NDArray[np.floating],
        time: NDArray[np.floating],
    ) -> None:
        """
        Signature of a diffeqpy system dynamics function.

        Uses state and parameters to update the values stored in 'derivative'.
        Modifications are done in-place: derivative must be a list and not a numpy array,
        or copies will be made instead of in-place modifications.
        """
