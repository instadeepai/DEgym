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

import importlib.util

import numpy as np

if importlib.util.find_spec("diffeqpy") is not None:
    from diffeqpy import de

from numpy.typing import NDArray
from pydantic.dataclasses import dataclass

from degym.integrators.base import Integrator, IntegratorConfig, TimeSpan
from degym.system_dynamics import DiffeqpySystemDynamicsFn


@dataclass
class DiffeqpyIntegratorConfig(IntegratorConfig):
    """Config for DiffeqpyIntegrator."""

    action_duration: float


class DiffeqpyIntegrator(Integrator):
    """Class responsible for integrating batches of DAEs with diffeqpy."""

    def __init__(
        self, system_dynamics: DiffeqpySystemDynamicsFn, integrator_config: DiffeqpyIntegratorConfig
    ):
        if importlib.util.find_spec("diffeqpy") is None:
            raise ImportError("diffeqpy is not installed")

        super().__init__(system_dynamics, integrator_config)
        self.ode_function = de.ODEFunction(system_dynamics, mass_matrix=system_dynamics.mass_matrix)

    def integrate(
        self,
        input_values: NDArray[np.floating],
        parameters: NDArray[np.floating],
        action: NDArray[np.floating],
        time_span: TimeSpan,
    ) -> NDArray[np.floating]:
        """
        Integrate DAE over one timespan, to get updated values of time-dependent variables.

        Args:
            input_values: 1D array containing current values of time-dependent variables.
            parameters: 1D array containing parameters used in the DAE equations.
            action: 1D array containing action values used in the DAE equations
            time_span: DiffeqpyTimeSpan object containing start and end times for the integration.
        Returns:
            next_values: 1D array of updated values of time-dependent variables.
        """
        # Setup ODE to be solved
        problem = de.ODEProblem(
            self.ode_function,
            input_values,
            (time_span.start_time, time_span.end_time),
            np.concatenate([parameters, action]),
            saveat=time_span.end_time,
        )

        # Use integrator to solve for updated state
        solution = de.solve(
            problem,
            de.Rodas5(autodiff=False),
        )

        next_values = np.array(de.stack(solution.u))[:, -1]
        return next_values
