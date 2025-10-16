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

from degym.utils import PydanticBaseModel


class Action(ABC):
    """
    Abstract base class for semantic action representations.

    The Action class represents an intermediate layer that gives semantic meaning to
    raw actions, i.e. the actions passed to the step function. The raw action is sometimes also
    referred to as the RL-agent action. Action is a transformation of raw actions into a
    structured, interpretable action representation before conversion to control parameters for
    the differential equation system.

    Purpose and Role:
        - Provides semantic meaning to raw actions (e.g., numpy arrays)
        - Creates interpretable action structure with named fields and clear meanings
        - Serves as intermediate representation between raw actions and physical controls
        - Enables clear mapping from agent decisions to control intentions

    Raw Agent Output → Action → DAEAction Flow:
        1. Agent produces raw action (e.g., np.array([1, 0.5]))
        2. The raw action is converted to Action by the ActionPreprocessor. The Action
        gives semantic meaning (e.g., heater_on=True, flow_rate=0.5)
        3. ActionConverter transforms the Action to DAEAction (e.g., Q_dot=5000 kJ/h,
        F=2.5 m³/h). These are the actual control parameters that form the DAE.

    Common Action Representations:
        - Structured actions: Named fields for different control components
        - Categorical + continuous: Discrete choices combined with continuous parameters
        - High-level commands: Abstract control decisions that map to complex control strategies
        - Normalized parameters: Scaled values that represent control intensities

    Semantic Interpretation Examples:
        - Raw [1, 0.5] → CSTRAction(heater_on=True, flow_rate_normalized=0.5)
        - Raw 0.7 → CSTRAction(heat_intensity=0.7)
        - Raw [2, 0.3, 0.8] → BatchAction(recipe_id=2, feed_rate=0.3, temperature_setpoint=0.8)

    Usage in Environment Pipeline:
        Actions bridge the gap between RL agent outputs and actions as they appear in the DAE:
        1. Raw agent output is wrapped into structured Action
        2. ActionConverter transforms Action to DAEAction with physical meaning
        3. ActionRegulator applies constraints to DAEAction
        4. DAEAction is used in numerical integration

    Implementation Strategies:
        - Direct mapping: Simple one-to-one correspondence with raw actions
        - Semantic structuring: Named fields that clarify control intentions
        - High-level abstraction: Actions that represent complex control policies
        - Domain-specific interpretation: Actions tailored to specific reactor types

    Example:
        ```python
        # Raw agent output: np.array([1, 0.5])
        # Semantic Action representation:
        @dataclass(frozen=True)
        class CSTRAction(Action):
            heater_on: bool      # 1 → True (heater is on)
            flow_rate: float     # 0.5 → normalized flow rate [0,1]

        # Later converted to physical DAEAction:
        # CSTRDAEAction(q_dot=5000.0, volumetric_flow=2.5)
        ```

    Note:
        - Actions provide semantic meaning, not physical units or direct control values
        - The Action class is a marker interface with no required methods
        - Conversion to physical parameters is handled by ActionConverter implementations
        - Actions make the control logic more interpretable and debuggable
    """


class DAEAction(PydanticBaseModel):
    """
    Abstract base class for actions as they appear in differential equation systems.

    The DAEAction class represents control parameters in the format required by the
    differential equation solver. These are domain-specific, physically meaningful
    control inputs that directly affect the system dynamics through the DAE formulation.

    Purpose and Role:
        - Encapsulates control parameters for differential equation integration
        - Provides interface between action processing and numerical solvers
        - Ensures actions are in physically meaningful units and ranges
        - Enables direct use in system dynamics calculations

    Common DAEAction Components:
        - Physical control inputs: Heat rates, flow rates, pressures, temperatures
        - Actuator commands: Valve positions, pump speeds, heater power levels
        - Setpoints: Target values for controlled variables
        - Operating parameters: Concentrations, residence times, reaction conditions

    Integration with Environment Flow:
        DAEActions are used in the numerical integration process:
        1. Agent Action is converted to DAEAction by ActionConverter
        2. ActionRegulator applies constraints to ensure feasible DAEAction
        3. DAEAction is passed to integrator for dynamics computation
        4. DAEAction values appear directly in differential equation formulations

    Implementation Strategies:
        - Direct mapping: One-to-one correspondence with control actuators
        - Derived parameters: Computed from multiple action components
        - Rate-based actions: Control rates of change rather than absolute values
        - Constraint-aware: Actions that inherently respect physical limitations

    Numerical Array Interface:
        DAEActions must be convertible to/from numpy arrays for integrator compatibility:
        - to_np_array(): Converts action to flat array for numerical computation
        - from_np_array(): Reconstructs action from numerical solver output

    Example:
        ```python
        class CSTRDAEAction(DAEAction):
            q: float  # Heat input rate [kJ/min]

            def to_np_array(self) -> NDArray[np.floating]:
                return np.array([self.q])

            @classmethod
            def from_np_array(cls, np_array: NDArray[np.floating]) -> "CSTRDAEAction":
                return cls(q=np_array[0])

        class MultiInputDAEAction(DAEAction):
            heat_rate: float       # Heat input [kJ/min]
            flow_rate: float       # Volumetric flow [m³/min]
            inlet_temp: float      # Inlet temperature [K]
        ```

    Note:
        - DAEActions represent control parameters in physical units and realistic ranges
        - Values should be ready for direct use in differential equation calculations
        - Conversion between Action and DAEAction is handled by ActionConverter
        - Constraints and safety limits are enforced by ActionRegulator
    """

    @abstractmethod
    def to_np_array(self) -> NDArray[np.floating]:
        """
        Convert the DAEAction to a numpy array for numerical computation.

        This method transforms the DAEAction data structure into a flat numpy array
        that can be used by numerical integrators and system dynamics functions.
        The conversion should be deterministic and maintain consistent ordering.

        Returns:
            NDArray[np.floating]: A 1D numpy array containing all action values in a
                consistent order. The array is used directly in differential
                equation calculations and must match the expected format of
                the system dynamics function.

        Example:
            For a CSTR with heat input control:
            ```python
            action = CSTRDAEAction(q=1500.0)
            array = action.to_np_array()  # Returns [1500.0]
            ```

        Note:
            The ordering and units of values should match the expectations of
            the system dynamics function and remain consistent across all
            instances of the same DAEAction type.
        """

    @classmethod
    @abstractmethod
    def from_np_array(cls, np_array: NDArray[np.floating]) -> "DAEAction":
        """
        Create a DAEAction instance from a numpy array.

        This method reconstructs a DAEAction from a flat numpy array, typically
        after numerical computation or optimization. It serves as the inverse
        operation to to_np_array() and enables round-trip conversion.

        Args:
            np_array: A 1D numpy array containing action values in the same
                order as produced by to_np_array(). The array length and
                structure should match the DAEAction's expected format.

        Returns:
            DAEAction: A new instance of the DAEAction class initialized with
                values from the numpy array. The instance should be equivalent
                to the original that produced the array.

        Example:
            For a CSTR with heat input control:
            ```python
            array = np.array([1500.0])
            action = CSTRDAEAction.from_np_array(array)
            # action.q == 1500.0
            ```

        Note:
            - This method should validate array dimensions and handle any
            necessary type conversions to ensure proper DAEAction construction.
            - The method must be consistent with the to_np_array() implementation.
        """
