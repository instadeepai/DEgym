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


class TerminatedExtractor(ABC):
    """
    Abstract base class for determining episode termination conditions.

    The TerminatedExtractor is responsible for evaluating whether an RL episode should
    end due to reaching terminal states defined by the underlying Markov Decision Process (MDP).
    Termination occurs when the system reaches states that naturally conclude the episode, such
    as achieving objectives, reaching absorbing states, or violating safety constraints.

    Purpose and Role:
        - Identifies natural ending conditions for episodes
        - Enforces MDP termination conditions based on state transitions
        - Ensures episodes end when objectives are achieved or constraints violated
        - Distinguishes between natural termination and artificial truncation

    Common Termination Conditions:
        - Objective achievement: Target concentrations, temperatures, or conversions reached
        - Safety violations: Operating outside safe temperature, pressure, or concentration limits
        - Equipment failures: Simulated component failures or degradation beyond limits
        - Economic targets: Cost thresholds or profit targets met
        - Physical constraints: Material balance violations or thermodynamic impossibilities, e.g.
        negative concentrations

    Integration with Environment Flow:
        The TerminatedExtractor is called during each environment step:
        1. Agent takes action and environment integrates to new state
        2. TerminatedExtractor evaluates the state transition:
            - If terminated=True, episode should  end and environment should reset
            - If terminated=False, episode should continue with next step

    Implementation Strategies:
        - Simple thresholds: Binary checks against state variable limits
        - Multi-condition logic: Combination of multiple termination criteria
        - Probabilistic termination: Stochastic ending based on failure models

    Design Considerations:
        - Termination should reflect natural episode endings, not time limits
        - Termination is not the best way to ensure safe operation. Nevertheless, safety-critical
        behavior could be encouraged by using conservative termination criteria
        - Avoid premature termination that prevents learning
        - Consider both upper and lower bounds for continuous variables

    Example:
        ```python
        class CSTRTerminatedExtractor(TerminatedExtractor):
            def extract_terminated(self, state: CSTRState, action: CSTRDAEAction,
                                 next_state: CSTRState) -> bool:
                # Terminate if temperature exceeds safe operating limits
                max_temp = 400.0  # K
                min_temp = 250.0  # K
                temp = next_state.dae_state.T

                # Terminate if concentrations become negative (unphysical)
                c_a = next_state.dae_state.c_a
                c_b = next_state.dae_state.c_b

                return (c_a < 0.0 or c_b < 0.0)
        ```

    Note:
        - Termination is distinct from truncation (time-based episode ending)
        - The extractor has access to the full state transition (s, a, s')
        - Termination conditions should align with the problem's natural endpoints
    """

    @abstractmethod
    def extract_terminated(self, state: State, action: DAEAction, next_state: State) -> bool:
        """
        Determine whether episode termination conditions are met.

        This method evaluates the state transition to decide if the episode should
        end due to natural termination conditions of the MDP. Termination occurs
        when the system reaches absorbing states, achieves objectives, or violates
        fundamental constraints.

        Args:
            state: The environment state at the beginning of the step (s). Contains
                dae_state, dae_params, and non_dae_params representing the system
                configuration before the action was applied.
            action: The DAE action taken during the step (a). Represents the control
                inputs applied to the system, which may influence termination
                conditions through their effects on the state.
            next_state: The environment state at the end of the step (s'). Contains
                the system configuration after applying the action and integrating
                the dynamics, which is evaluated for termination conditions.

        Returns:
            bool: True if the episode should terminate due to MDP termination
                conditions, False if the episode should continue. Termination
                indicates a natural endpoint of the episode.

        Example:
            Termination evaluation might involve:
            - Checking if state variables exceed safety limits
            - Verifying that concentrations remain physically meaningful
            - Detecting equipment failure or degradation conditions
            - Confirming achievement of process objectives or targets

        Note:
            - Termination should reflect natural episode endings, not time limits
            - Consider both the action taken and the resulting state change
            - Termination is not the best way to ensure safe operation. Nevertheless,
            safety-critical behavior could be encouraged by using conservative termination criteria
            - Distinguish from truncation which handles for exampletime-based episode ending
        """
