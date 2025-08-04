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
from pydantic.dataclasses import dataclass
from scipy.integrate import solve_ivp

from degym.integrators.base import Integrator, IntegratorConfig, TimeSpan


@dataclass
class ScipyIntegratorConfig(IntegratorConfig):
    """Config for ScipyIntegrator."""

    action_duration: float
    method: str = "RK45"  # 4th order Runge-Kutta method (explicit Runge-Kutta of order 4(5))
    rtol: float = 1e-6  # Relative tolerance
    atol: float = 1e-8  # Absolute tolerance


class ScipyIntegrator(Integrator):
    """Class responsible for integrating batches of ODEs."""

    def integrate(
        self,
        input_values: NDArray[np.floating],
        parameters: NDArray[np.floating],
        action: NDArray[np.floating],
        time_span: TimeSpan,
    ) -> NDArray[np.floating]:
        """
        Integrate system over one timespan, to get updated values of time-dependent variables.

        Args:
            input_values: 1D array containing current values of time-dependent variables.
            parameters: 1D array containing parameters used in the DAE equations.
            action: 1D array containing action values used in the DAE equations
            time_span: DiffeqpyTimeSpan object containing start and end times for the integration.
        Returns:
            next_values: 1D array of updated values of time-dependent variables.
        """
        # Solve the ODE using the method specified in the config
        solution = solve_ivp(
            fun=lambda time, state: self.system_dynamics(state, parameters, action, time),
            t_span=(time_span.start_time, time_span.end_time),
            y0=input_values,
            method=self.config.method,
            rtol=self.config.rtol,
            atol=self.config.atol,
        )

        next_values = solution.y[:, -1]
        return next_values
