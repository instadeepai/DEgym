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


class ScipySystemDynamicsFn(SystemDynamicsFn, ABC):
    """
    Implements the system dynamics equations in a way that is suitable for use
    with the scipy package for solving differential equations.
        https://github.com/scipy/scipy
    """

    @staticmethod
    @abstractmethod
    def __call__(  # NOTE: state and time are required by scipy - do not change
        state: NDArray[np.floating],
        parameters: NDArray[np.floating],
        action: NDArray[np.floating],
        time: float,
    ) -> NDArray[np.floating]:
        """
        Signature of a scipy system dynamics function.

        Uses state, action and parameters to return the derivative of
        each state variable with respect to time.

        Args:
            state: Numpy array of shape (n_states,).
            parameters: Numpy array of shape (n_parameters,).
            action: Numpy array of shape (n_actions,).
            time: Time.

        The return value is a numpy array of shape (n_states,).
        """
