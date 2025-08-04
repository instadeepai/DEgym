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

from typing import Optional

from numpy.typing import NDArray
from degym.environment import Environment
from degym.integrators import TimeSpan

from degym_tutorials.cstr_tutorial.rendering import draw_cstr_with_heater_and_circles
from degym_tutorials.cstr_tutorial.state_concrete_classes import (
    CSTRDAEParameters,
    CSTRNonDAEParameters,
    CSTRState,
)


class CSTREnvironment(Environment):
    """Environment for the CSTR problem."""

    def _return_next_dae_params(self, state: CSTRState) -> CSTRDAEParameters:
        """Return the current DAE parameters as the next DAE parameters."""
        return state.dae_params

    def _return_next_non_dae_params(self, state: CSTRState) -> CSTRNonDAEParameters:
        """Return the next non-DAE parameters by incrementing the timestep."""
        state.non_dae_params.timestep += 1
        return state.non_dae_params

    def _calculate_time_span(self) -> TimeSpan:
        """Calculate the time span of step function."""
        time_span = TimeSpan(
            start_time=self._current_time,
            end_time=self._current_time + self._integrator.config.action_duration,
        )
        return time_span

    def render(self, last_action: Optional[float] = None) -> NDArray:
        """
        Render the current state of the environment.

        Args:
            last_action (float, optional): The last action taken by the agent.

        Returns:
            np.ndarray: A numpy array representation of the rendered image with shape with
            (height, width, channels).
        """
        return draw_cstr_with_heater_and_circles(
            self.state.dae_state, self._physical_parameters, last_action
        )
