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
from typing import TypeAlias

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from degym.action.action import DAEAction
from degym.action.action_converter import ActionConverter
from degym.action.action_regulator import ActionRegulator
from degym.state import State

RawActionType: TypeAlias = NDArray[np.floating] | float


class ActionPreprocessor(ABC):
    """
    Abstract base class for preprocessing RL agent actions before numerical integration.

    The ActionPreprocessor coordinates the complete action processing pipeline that
    transforms raw RL agent outputs into constrained, physically meaningful DAEActions
    ready for differential equation integration. It combines action conversion and
    constraint enforcement to ensure safe and feasible control inputs.

    Purpose and Role:
        - Coordinates complete action preprocessing pipeline
        - Transforms raw agent outputs to physics-ready control inputs
        - Defines the action space visible to RL agents
        - Ensures actions are converted, constrained, and validated before integration

    Action Processing Pipeline:
        The ActionPreprocessor orchestrates several steps:
        1. Wrap raw action output in Action class structure
        2. Convert Action to DAEAction using ActionConverter
        3. Apply constraints using ActionRegulator (safety, feasibility)
        4. Return validated DAEAction ready for numerical integration

    Component Integration:
        - action_converter: Handles transformation between action representations
        - action_regulator: Enforces constraints and safety limits
        - action_space: Defines bounds and structure for RL algorithms

    Common Preprocessing Operations:
        - Action space definition: Specify bounds, dimensions, and data types
        - Raw action wrapping: Convert numpy arrays or scalars to Action objects
        - Conversion coordination: Manage Action to DAEAction transformation
        - Constraint enforcement: Apply safety and feasibility checks
        - Validation: Ensure final actions are ready for integration

    Implementation Strategies:
        - Simple pipeline: Direct conversion and regulation in sequence
        - Adaptive preprocessing: State-dependent conversion and constraints
        - Multi-step validation: Multiple constraint checking phases
        - Error handling: Graceful handling of invalid or extreme actions

    Example:
        ```python
        class CSTRActionPreprocessor(ActionPreprocessor):
            def __init__(self, converter: CSTRActionConverter,
                        regulator: CSTRActionRegulator):
                self.action_converter = converter
                self.action_regulator = regulator

            @property
            def action_space(self) -> gym.spaces.Box:
                # Normalized heat input in [0, 1]
                return gym.spaces.Box(low=0.0, high=1.0, shape=(1,))

            def preprocess_action(self, raw_action: float,
                                state: CSTRState) -> CSTRDAEAction:
                # Wrap raw action
                rl_action = CSTRAction(q_normalized=raw_action)
                # Convert to physical units
                dae_action = self.action_converter.action_to_dae_action(rl_action, state)
                # Apply constraints
                safe_action = self.action_regulator.convert_to_legal_action(dae_action, state)
                return safe_action
        ```

    Design Considerations:
        - Action space design affects RL algorithm performance and stability
        - Preprocessing should preserve important properties for learning
        - Error handling should be robust for extreme or invalid actions
        - Performance is important as preprocessing occurs every environment step

    Note:
        - ActionPreprocessor is the main interface between RL agents and action processing
        - It coordinates but doesn't implement conversion or regulation logic
        - The action_space property is crucial for RL algorithm compatibility
    """

    action_converter: ActionConverter
    action_regulator: ActionRegulator

    @property
    @abstractmethod
    def action_space(self) -> gym.spaces.Space:
        """
        Define the action space for RL agents.

        This property specifies the structure, bounds, and data types of actions
        that RL agents should produce. The action space definition is crucial for
        RL algorithms to understand the expected action format and ranges.

        Returns:
            gym.spaces.Space: A gymnasium Space object defining the action
                structure. Commonly a Box space for continuous control, but
                can be Discrete, MultiDiscrete, or other space types depending
                on the control paradigm.

        Example:
            For a CSTR with normalized heat input control:
            ```python
            @property
            def action_space(self) -> gym.spaces.Box:
                return gym.spaces.Box(
                    low=np.array([0.0]),     # Minimum normalized heat
                    high=np.array([1.0]),    # Maximum normalized heat
                    shape=(1,),              # Single control input
                    dtype=np.float32
                )
            ```

        Note:
            The action space should match the format expected by preprocess_action()
            and be compatible with the RL algorithm being used. Mismatched action
            spaces can cause training failures or suboptimal performance.
        """

    @abstractmethod
    def preprocess_action(self, action: RawActionType, state: State) -> DAEAction:
        """
        Preprocess a raw RL agent action (i.e. the action passed to the step function)
        for use in numerical integration.

        This method orchestrates the complete action preprocessing pipeline,
        transforming raw agent outputs through conversion and constraint enforcement
        to produce safe, feasible DAEActions ready for differential equation solving.

        The preprocessing should follow these steps:
        1. Wrap the raw action in the appropriate Action class structure
        2. Use ActionConverter to transform Action to DAEAction (scaling, units)
        3. Apply ActionRegulator to enforce constraints and ensure safety
        4. Return the validated DAEAction ready for integration

        Args:
            action: Raw action output from the RL agent, typically a numpy array
                or scalar value. The format should match the action_space
                specification and represent the agent's control decision.
            state: Current environment state that may influence action processing.
                State-dependent conversion and constraints enable adaptive
                control strategies and context-aware safety mechanisms.

        Returns:
            DAEAction: Processed action ready for numerical integration. The
                action should be in physical units, satisfy all constraints,
                and be safe for use in differential equation calculations.

        Example:
            Processing a normalized heat input for a CSTR:
            ```python
            def preprocess_action(self, action: float, state: CSTRState) -> CSTRDAEAction:
                # Step 1: Wrap raw action
                rl_action = CSTRAction(q_normalized=action)

                # Step 2: Convert to physical units
                dae_action = self.action_converter.action_to_dae_action(rl_action, state)

                # Step 3: Apply constraints
                safe_action = self.action_regulator.convert_to_legal_action(dae_action, state)

                return safe_action
            ```

        Note:
            This method is called every environment step and should be efficient.
            Error handling should be robust for edge cases and invalid inputs.
        """
