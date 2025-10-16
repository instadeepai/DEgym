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

from degym.action.action import Action, DAEAction
from degym.state import State


class ActionConverter(ABC):
    """
    Abstract base class for converting semantic actions (Action) to DAE actions (DAEAction) and
    vice versa.

    The ActionConverter transforms semantically meaningful Action representations into
    the DAE actions (DAEActions) that appear directly in the differential
    equation system. It bridges the gap between interpretable action structures and the
    numerical values required by the DAE solver.

    Purpose and Role:
        - Converts semantic action representations to DAE actions
        - Enables separation between interpretable actions and numerical DAE variables
        - Provides bidirectional mapping for analysis and debugging
        - Handles scaling, unit conversion, and complex control logic transformations

    Conversion Process:
        Raw Step Input → Action (semantic) → DAEAction (DAE actions)

        Example flow:
        1. Raw: np.array([1, 0.5]) passed to step()
        2. Action: CSTRAction(heater_on=True, flow_rate_normalized=0.5)
        3. DAEAction: CSTRDAEAction(q_dot=5000.0, volumetric_flow=2.5), where q_dot and
        volumetric_flow appear in differential equations

    Common Conversion Operations:
        - Boolean to physical: heater_on=True → q_dot=5000 kJ/h
        - Normalized to physical: flow_rate=0.5 → volumetric_flow=2.5 m³/h
        - Categorical to parameters: recipe_id=2 → specific temperature/pressure setpoints
        - Complex mappings: control_mode="heating" → multiple coordinated DAE parameters

    Integration with Environment Flow:
        The ActionConverter operates in the action preprocessing pipeline:
        1. Raw step input is wrapped into semantic Action
        2. ActionConverter transforms Action to DAEAction (DAE actions)
        3. ActionRegulator applies constraints to DAEAction
        4. DAEAction parameters are used directly in numerical integration

    Implementation Strategies:
        - Direct scaling: Linear transformations from normalized to physical ranges
        - Lookup tables: Categorical actions mapped to predefined parameter sets
        - State-dependent: Conversions that adapt based on current system conditions
        - Complex logic: Multi-step transformations for sophisticated control strategies

    Design Considerations:
        - It is recommended that conversions preserve the interpretability of semantic actions
        - DAE actions must match the variables expected by the DAE formulation
        - State dependency enables adaptive and context-aware control strategies
        - Bidirectional conversion supports analysis and action space exploration

    Example:
        ```python
        class CSTRActionConverter(ActionConverter):
            def _action_to_dae_action(self, action: CSTRAction,
                                    state: CSTRState) -> CSTRDAEAction:
                # Convert semantic action to DAE actions
                if action.heater_on:
                    q_dot = state.non_dae_params.q_max  # Full heating power
                else:
                    q_dot = 0.0  # No heating

                # Scale normalized flow to physical range
                flow_max = state.non_dae_params.flow_max
                volumetric_flow = action.flow_rate_normalized * flow_max

                return CSTRDAEAction(q_dot=q_dot, volumetric_flow=volumetric_flow)

            def _dae_action_to_action(self, dae_action: CSTRDAEAction,
                                    state: CSTRState) -> CSTRAction:
                # Convert DAE actions back to semantic representation
                heater_on = dae_action.q_dot > 0.0
                flow_normalized = dae_action.volumetric_flow / state.non_dae_params.flow_max
                return CSTRAction(heater_on=heater_on, flow_rate_normalized=flow_normalized)
        ```

    Note:
        - ActionConverter bridges semantic meaning and DAE actions
        - Conversions may involve complex logic, not just simple scaling
        - State dependency allows for adaptive and intelligent control transformations
        - Both conversion directions enable comprehensive action space analysis
    """

    def action_to_dae_action(self, action: Action, state: State) -> DAEAction:
        """
        Convert a semantic Action to DAE actions.

        This public method transforms semantically meaningful Action representations
        into the DAE actions (DAEAction) that appear directly in
        the differential equation system. It delegates to the protected method
        for the actual conversion implementation.

        Args:
            action: The semantic action representation that gives meaning to
                raw step inputs. This provides interpretable structure with
                named fields that clarify control intentions.
            state: The current environment state that may influence the conversion.
                State-dependent conversions enable adaptive control strategies
                and context-aware parameter selection.

        Returns:
            DAEAction: The DAE actions ready for use in the
                differential equation system. These values appear directly
                as variables in the DAE formulation and numerical integration.

        Example:
            Converting semantic heating control to DAE actions:
            ```python
            semantic_action = CSTRAction(heater_on=True, flow_rate_normalized=0.5)
            dae_params = converter.action_to_dae_action(semantic_action, current_state)
            # dae_params.q_dot = 5000.0 kJ/h (appears in energy balance equation)
            # dae_params.volumetric_flow = 2.5 m³/h (appears in material balance)
            ```

        Note:
            This method provides the main interface for semantic-to-DAE
            conversion while allowing subclasses to implement domain-specific
            transformation logic in the protected method.
        """
        return self._action_to_dae_action(action, state)

    @abstractmethod
    def _action_to_dae_action(self, action: Action, state: State) -> DAEAction:
        """
        Implement the conversion from semantic Action to DAEAction.

        This protected method contains the actual conversion logic that transforms
        semantically meaningful actions into DAE actions for the
        DAE system. Subclasses must implement this method to define how semantic
        actions map to the specific DAE variables in their system.

        Args:
            action: The semantic action representation to be converted.
            state: The current environment state that may influence the conversion.

        Returns:
            DAEAction: DAE actions for the DAE system.

        Example:
            Converting semantic controls to DAE actions:
            ```python
            def _action_to_dae_action(self, action: CSTRAction,
                                    state: CSTRState) -> CSTRDAEAction:
                # Boolean heater control → DAE action heat rate
                q_dot = state.non_dae_params.q_max if action.heater_on else 0.0

                # Normalized flow → DAE action volumetric flow rate
                volumetric_flow = action.flow_rate_normalized * state.non_dae_params.flow_max

                return CSTRDAEAction(q_dot=q_dot, volumetric_flow=volumetric_flow)
            ```

        Note:
            This method should create DAEAction parameters that match exactly
            the variables expected by the DAE formulation and numerical solver.
        """
        raise NotImplementedError

    def dae_action_to_action(self, dae_action: DAEAction, state: State) -> Action:
        """
        Convert physical DAE actions back to semantic Action representation.

        This public method provides the inverse conversion, transforming
        DAEAction back into semantically meaningful Action representations.
        This capability supports analysis, debugging, and understanding of the
        relationship between semantic actions and DAE actions.

        Args:
            dae_action: The DAE actions to be converted back
                to semantic format. These represent the actual variables that
                appear in the differential equation system.
            state: The current environment state that may influence the conversion.
                State-dependent conversions should be consistent with the
                forward conversion direction.

        Returns:
            Action: The semantic action representation that corresponds to the
                given DAE actions. This should provide interpretable
                meaning to the control decisions.

        Example:
            Converting DAE actions back to semantic representation:
            ```python
            dae_params = CSTRDAEAction(q_dot=5000.0, volumetric_flow=2.5)
            semantic_action = converter.dae_action_to_action(dae_params, current_state)
            # semantic_action.heater_on = True (since q_dot > 0)
            # semantic_action.flow_rate_normalized = 0.5 (2.5 / 5.0 max flow)
            ```

        Note:
            This method enables bidirectional conversion for analysis and
            should be mathematically consistent with action_to_dae_action.
        """
        return self._dae_action_to_action(dae_action, state)

    @abstractmethod
    def _dae_action_to_action(self, dae_action: DAEAction, state: State) -> Action:
        """
        Implement the conversion from DAEAction to semantic Action.

        This protected method contains the actual inverse conversion logic that
        transforms DAE actions back into semantically meaningful
        Action representations. Subclasses must implement this method to provide
        bidirectional conversion capability.

        Args:
            dae_action: The DAE actions to convert to semantic format.
            state: The current environment state that may influence the conversion.

        Returns:
            Action: The semantic action representation.

        Example:
            Converting DAE actions back to semantic actions:
            ```python
            def _dae_action_to_action(self, dae_action: CSTRDAEAction,
                                    state: CSTRState) -> CSTRAction:
                # DAE action heat rate → boolean heater control
                heater_on = dae_action.q_dot > 0.0

                # DAE action flow → normalized flow rate
                flow_max = state.non_dae_params.flow_max
                flow_normalized = dae_action.volumetric_flow / flow_max

                return CSTRAction(heater_on=heater_on, flow_rate_normalized=flow_normalized)
            ```

        Note:
            This method should be the mathematical inverse of _action_to_dae_action
            to ensure consistent bidirectional conversion and enable proper
            round-trip transformations.
        """
