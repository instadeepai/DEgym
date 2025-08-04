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

from degym.action.action import DAEAction
from degym.state import State


class ActionRegulator(ABC):
    """
    Abstract base class for enforcing constraints on DAE actions.

    The ActionRegulator is responsible for ensuring that DAEActions comply with
    physical, safety, and operational constraints before being used in differential
    equation integration. It provides validation and correction mechanisms to
    prevent unsafe or infeasible control inputs from reaching the system dynamics.

    Purpose and Role:
        - Enforces safety constraints on control actions
        - Validates action feasibility before integration
        - Provides correction mechanisms for constraint violations
        - Ensures actions remain within physically meaningful ranges

    Common Constraint Types:
        - Physical limits: Maximum heat rates, flow capacities, pressure bounds
        - Safety constraints: Temperature limits, concentration ranges, equipment ratings
        - Operational bounds: Economic limits, efficiency thresholds, resource availability
        - Dynamic constraints: Rate limits, state-dependent bounds, time-varying limits

    Integration with Environment Flow:
        The ActionRegulator is used in the action preprocessing pipeline:
        1. Agent Action is converted to DAEAction by ActionConvertor
        2. ActionRegulator validates and corrects the DAEAction
        3. Corrected DAEAction is used in numerical integration
        4. Regulation prevents unsafe or infeasible system operation

    Implementation Strategies:
        - Hard clipping: Clip actions to predefined bounds
        - Soft constraints: Apply penalty-based corrections
        - State-dependent limits: Constraints that vary with system state
        - Rate limiting: Restrict how quickly actions can change

    Constraint Enforcement Approaches:
        - Validation only: Check constraints but don't modify actions
        - Automatic correction: Modify violating actions to nearest feasible values
        - Projection: Map infeasible actions to feasible action space
        - Barrier methods: Use smooth functions to discourage constraint violations

    Example:
        ```python
        class CSTRActionRegulator(ActionRegulator):
            def is_legal(self, dae_action: CSTRDAEAction, state: CSTRState) -> bool:
                # Check if heat input is within physical limits
                q_max = state.non_dae_params.q_max
                return 0.0 <= dae_action.q <= q_max

            def convert_to_legal_action(self, dae_action: CSTRDAEAction,
                                      state: CSTRState) -> CSTRDAEAction:
                # Clip heat input to feasible range
                q_max = state.non_dae_params.q_max
                q_clipped = max(0.0, min(dae_action.q, q_max))
                return CSTRDAEAction(q=q_clipped)
        ```

    Design Considerations:
        - Constraint violations should be handled gracefully without crashing
        - The logic behind correcting actions should be aligned with use-case
        - State-dependent constraints enable adaptive safety mechanisms

    Note:
        - ActionRegulator operates on DAEActions, before the integration step
        - Constraints should reflect real physical and safety limitations
        - Regulation provides the final safety check before numerical integration
    """

    @abstractmethod
    def is_legal(self, dae: DAEAction, state: State) -> bool:
        """
        Check if a DAE action satisfies all constraints.

        This method validates whether the given DAEAction complies with all
        physical, safety, and operational constraints for the current system
        state. It provides a boolean check without modifying the action.

        Args:
            dae: The DAEAction to be validated. This represents control inputs
                in their physical units and ranges as required by the
                differential equation solver.
            state: The current environment state that may influence constraint
                evaluation. State-dependent constraints enable adaptive
                safety and operational limits.

        Returns:
            bool: True if the action satisfies all constraints and is safe
                to use in numerical integration, False if any constraints
                are violated and the action requires correction.

        Example:
            Checking heat input constraints for a CSTR:
            ```python
            action = CSTRDAEAction(q=6000.0)  # High heat input
            is_safe = regulator.is_legal(action, current_state)
            # is_safe might be False if q_max = 5000.0
            ```

        Note:
            This method should be consistent with convert_to_legal_action() -
            actions that pass this check should not be modified by the
            conversion method.
        """

    @abstractmethod
    def convert_to_legal_action(self, dae_action: DAEAction, state: State) -> DAEAction:
        """
        Convert a potentially illegal DAE action to a legal one.

        This method takes a DAEAction that may violate constraints and returns
        a corrected version that satisfies all constraints while remaining as
        close as possible to the original action. This ensures safe operation
        while preserving the agent's control intentions.

        Args:
            dae_action: The potentially constraint-violating DAEAction to be
                corrected. This represents the raw control inputs after
                conversion from RL actions but before constraint enforcement.
            state: The current environment state that influences constraint
                evaluation and correction strategies. State-dependent
                constraints enable context-aware action regulation.

        Returns:
            DAEAction: A corrected action that satisfies all constraints and
                is safe for use in numerical integration. The corrected action
                should be as close as possible to the original while ensuring
                feasibility and safety.

        Example:
            Correcting an excessive heat input for a CSTR:
            ```python
            unsafe_action = CSTRDAEAction(q=6000.0)  # Above q_max
            safe_action = regulator.convert_to_legal_action(unsafe_action, state)
            # safe_action.q might be 5000.0 if q_max = 5000.0 (clipped)
            ```

        Note:
            The corrected action should pass the is_legal() check for the
            same state. Common correction strategies include clipping,
            projection, and constraint-aware optimization.
        """
