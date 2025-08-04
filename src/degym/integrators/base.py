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
from typing import Any

import numpy as np
from numpy.typing import NDArray

from degym.system_dynamics import SystemDynamicsFn
from degym.utils import PydanticBaseModel


class TimeSpan(PydanticBaseModel):
    """A class for storing the start and end times of an integration step."""

    start_time: float
    end_time: float


class IntegratorConfig(ABC):
    """A config for storing the parameters of the integrator."""


class Integrator(ABC):
    """Class responsible for integrating a dynamical system."""

    def __init__(self, system_dynamics: SystemDynamicsFn, integrator_config: IntegratorConfig):
        self._system_dynamics = system_dynamics
        self._config = integrator_config

    @abstractmethod
    def integrate(self, *args: Any, **kwargs: Any) -> NDArray[np.floating]:
        """
        Integrate the dynamical system.

        Args:
            *args: The arguments to pass to the integrator, which may vary between integrators.
            **kwargs: The keyword arguments to pass to the integrator.
        Returns:
            The results of the integration, i.e. the updated values of the dae_state.
        """

    @property
    def system_dynamics(self) -> SystemDynamicsFn:
        """Return the system dynamics function."""
        return self._system_dynamics

    @property
    def config(self) -> IntegratorConfig:
        """Return the integrator config."""
        return self._config
