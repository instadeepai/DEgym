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

from degym.action import DAEAction
from degym.state import State


class TruncatedExtractor(ABC):
    """
    Abstract base class for determining episode truncation conditions.

    The TruncatedExtractor is responsible for evaluating whether an RL episode should
    end due to artificial stopping conditions that are outside the natural termination
    conditions of the Markov Decision Process (MDP). Truncation typically occurs due to time
    limits, computational constraints, or external factors unrelated to the system's dynamics.

    Purpose and Role:
        - Handles episode ending conditions not related to MDP dynamics
        - Enforces time limits and computational constraints

    Common Truncation Conditions:
        - Time limits: Maximum episode length or simulation time reached
        - Computational limits: Maximum integration steps or CPU time exceeded

    Integration with Environment Flow:
        The TruncatedExtractor is called during each environment step:
        1. Agent takes action and environment integrates to new state
        2. TruncatedExtractor evaluates external stopping conditions:
            - If truncated=True, episode should end but environment should not reset immediately
            - If truncated=False, episode should continue with next step

    Implementation Strategies:
        - Step counting: Track timesteps against maximum episode length
        - Time tracking: Monitor simulation time against time limits
        - Resource monitoring: Check computational resource usage

    Design Considerations:
        - Truncation should not affect the value function estimation in RL
        - Time limits should be long enough to allow meaningful learning
        - Consider varying truncation conditions for training robustness
        - Truncation is independent of the system's physical state

    Example:
        ```python
        class CSTRTruncatedExtractor(TruncatedExtractor):
            def extract_truncated(self, state: CSTRState, action: CSTRDAEAction,
                                next_state: CSTRState) -> bool:
                # Truncate after maximum number of timesteps
                max_timesteps = next_state.non_dae_params.max_timestep
                current_timestep = next_state.non_dae_params.timestep

                return current_timestep >= max_timesteps

        class TimeLimitedTruncatedExtractor(TruncatedExtractor):
            def __init__(self, max_time: float):
                self.max_time = max_time

            def extract_truncated(self, state: CSTRState, action: CSTRDAEAction,
                                next_state: CSTRState) -> bool:
                # Truncate after maximum simulation time
                current_time = next_state.non_dae_params.timestep * 0.1  # dt = 0.1 min
                return current_time >= self.max_time
        ```

    Note:
        - Truncation is distinct from termination (MDP-based episode ending)
        - The extractor has access to the full state transition (s, a, s')
        - Truncation conditions are independent of system dynamics
    """

    @abstractmethod
    def extract_truncated(self, state: State, action: DAEAction, next_state: State) -> bool:
        """
        Determine whether episode truncation conditions are met.

        This method evaluates external stopping conditions to decide if the episode
        should end due to artificial constraints rather than natural MDP termination.
        Truncation typically occurs due to time limits, computational constraints,
        or other factors external to the system dynamics.

        Args:
            state: The environment state at the beginning of the step (s). Contains
                dae_state, dae_params, and non_dae_params representing the system
                configuration before the action was applied.
            action: The DAE action taken during the step (a). Generally not directly
                relevant for truncation decisions, but included for consistency
                with other extractors.
            next_state: The environment state at the end of the step (s'). Contains
                timing information and counters used to evaluate truncation
                conditions such as timestep limits.

        Returns:
            bool: True if the episode should truncate, False if the episode should continue.
            Truncation indicates an artificial endpoint of the episode.

        Example:
            Truncation evaluation might involve:
            - Checking if maximum timesteps have been reached
            - Verifying that simulation time limits are not exceeded


        Note:
            - Truncation should reflect external constraints, not system state
            - Most commonly based on timestep or time limits
            - Truncation doesn't affect value function estimation in RL algorithms
            - Distinguish from termination which handles MDP-based episode ending
        """
