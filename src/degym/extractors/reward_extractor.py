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


class RewardExtractor(ABC):
    """
    Abstract base class for extracting reward signals from transitions.

    The RewardExtractor is responsible for computing reward signals that guide RL agent
    learning by encoding the optimization objective. It evaluates state transitions
    (s, a, s') to provide immediate feedback about the desirability of actions taken
    in specific states.

    Purpose and Role:
        - Encodes the optimization objective as numerical reward signals
        - Provides immediate feedback for RL agent learning
        - Evaluates the quality of state transitions and actions
        - Bridges domain objectives with RL learning algorithms

    Common Reward Formulations:
        - Objective maximization: Reward proportional to desired quantity (e.g., product
        concentration)
        - Cost minimization: Negative reward for undesirable outcomes (energy consumption, waste)
        - Sparse rewards: Non-zero only when specific conditions are met (target reached)
        - Dense rewards: Continuous feedback based on progress toward goals
        - Multi-objective: Weighted combination of multiple performance criteria

    Integration with Environment Flow:
        The RewardExtractor is called during each environment step:
        1. Agent takes action in current state (s)
        2. Environment integrates to compute next state (s')
        3. RewardExtractor evaluates the transition (s, a, s')
        4. A float is returned to agent as the reward for the transition

    Implementation Strategies:
        - Direct objective: Reward equals the increase in quantity to be optimized
        - Shaped rewards: Include intermediate progress signals to aid learning
        - Constraint penalties: Negative rewards for violating operating (soft) constraints
        - Efficiency metrics: Rewards based on resource utilization or time

    Design Considerations:
        - Reward scale affects learning dynamics (typically normalize to [-1, 1] or [0, 1])
        - Sparse vs dense rewards trade off exploration vs sample efficiency
        - Multi-objective rewards require careful weighting of competing objectives

    Example:
        ```python
        class CSTRRewardExtractor(RewardExtractor):
            def extract_reward(self, state: CSTRState, action: CSTRDAEAction,
                             next_state: CSTRState) -> float:
                # Reward is the concentration of desired product B
                return float(next_state.dae_state.c_b)

        class MultiObjectiveCSTRRewardExtractor(RewardExtractor):
            def extract_reward(self, state: CSTRState, action: CSTRDAEAction,
                             next_state: CSTRState) -> float:
                # Multi-objective reward: maximize product, minimize energy
                product_reward = next_state.dae_state.c_b  # Maximize
                energy_penalty = -0.1 * action.q / state.non_dae_params.q_max  # Minimize
                return product_reward + energy_penalty
        ```

    Note:
        - Reward design significantly impacts RL performance and learned behavior
        - Rewards are not the best way to ensure safe operation. Constraints should be incorporated
        in other places, e.g. action preprocessing, to ensure safe operation
        - The extractor has access to the full transition (s, a, s')
        - Rewards should be bounded and scaled appropriately for the RL algorithm
    """

    @abstractmethod
    def extract_reward(self, state: State, action: DAEAction, next_state: State) -> float:
        """
        Compute the reward for a state transition in the environment.

        This method evaluates the quality of taking a specific action in a given state
        by analyzing the resulting state transition. The reward signal guides RL agent
        learning by providing immediate feedback about action desirability.

        Args:
            state: The environment state at the beginning of the step (s). Contains
                dae_state, dae_params, and non_dae_params representing the system
                configuration before the action was applied.
            action: The DAE action taken during the step (a). Represents the control
                inputs applied to the system, already converted and regulated by
                the action preprocessing pipeline.
            next_state: The environment state at the end of the step (s'). Contains
                the system configuration after applying the action, integrating
                the dynamics over the time step, and postprocessing.

        Returns:
            float: The immediate reward for this state transition.

        Example:
            Reward computation might involve:
            - Evaluating objective function value (concentration, temperature, efficiency)
            - Computing progress toward goals or setpoints
            - Applying penalties for constraint violations or unsafe operation
            - Including action costs (energy, material consumption)

        Notes:
            - The reward is immediate feedback for the specific transition (s, a, s')
            - Reward design directly affects what behavior the agent learns
            - Consider reward shaping to provide denser learning signals
            - Ensure rewards are bounded and scaled appropriately for the RL algorithm
        """
