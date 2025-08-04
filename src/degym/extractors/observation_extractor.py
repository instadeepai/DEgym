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

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from degym.state import State


class Observation(ABC):
    """
    Abstract base class for observations that are provided to RL agents.

    The Observation class represents the visible information that an RL agent receives
    from the environment at each step. It encapsulates the subset of environment state
    that is observable to the agent, typically after processing and transformation from
    the full internal state.

    Purpose and Role:
        - Encapsulates agent-visible information from environment state
        - Provides interface for converting observations to numerical arrays
        - Defines the structure of what agents can perceive

    Common Observation Components:
        - Process variables (concentrations, temperatures, pressures)
        - Normalized or scaled state measurements
        - Derived quantities (ratios, differences, trends)
        - Setpoints and operational targets
        - Time-based information (timestep, elapsed time)

    Usage in Environment:
        Observations are created by ObservationExtractor from the environment state
        and returned to the agent through the step() and reset() methods. They must
        be convertible to numpy arrays for compatibility with RL algorithms.

    Example:
        ```python
        @dataclass(frozen=True)
        class CSTRObservation(Observation):
            c_a: float  # Normalized concentration A
            c_b: float  # Normalized concentration B
            t: float    # Normalized temperature

            def to_np_array(self) -> NDArray[np.floating]:
                return np.array([self.c_a, self.c_b, self.t])
        ```

    Note:
        - The observation space should be bounded and well-defined
        - Observations must be serializable to numpy arrays
    """

    @abstractmethod
    def to_np_array(self) -> NDArray[np.floating]:
        """
        Convert the observation to a numpy array.

        This method transforms the observation data structure into a flat numpy array
        (that can be consumed by RL agents).

        Returns:
            NDArray[np.floating]: A 1D numpy array containing all observation values
                in a consistent order. The array length should match the
                observation space defined by the ObservationExtractor.

        Example:
            For a CSTR observation with concentrations and temperature:
            ```python
            observation = CSTRObservation(c_a=0.5, c_b=0.3, t=0.8)
            array = observation.to_np_array()  # Returns [0.5, 0.3, 0.8]
            ```

        Note:
            The ordering and scaling of values in the array should match
            the observation space bounds defined in the corresponding
            ObservationExtractor.
        """


class ObservationExtractor(ABC):
    """
    Abstract base class for extracting observations from environment states.

    The ObservationExtractor is responsible for transforming the complete environment
    state into observations that are provided to RL agents. It defines what information
    is visible to the agent and how it should be processed and scaled.

    Purpose and Role:
        - Converts internal environment state to agent-visible observations
        - Defines the observation space for RL algorithms
        - Enables scaling, normalization, and filtering to raw state data
        - Enables encapsulation of domain knowledge about relevant information for
        decision-making

    Common Extraction Operations:
        - Selecting relevant state variables for observation
        - Normalizing values to standard ranges (e.g., [0, 1] or [-1, 1])
        - Computing derived quantities (ratios, differences, trends)
        - Applying noise or measurement uncertainty models
        - Filtering or smoothing noisy measurements

    Integration with Environment Flow:
        The ObservationExtractor is called in both step() and reset() methods:
        1. Environment state is updated (via integration or initialization)
        2. ObservationExtractor extracts the observation from the state
        3. (After this class) Observation is returned to the agent and agent uses observation to
        select next action

    Implementation Strategies:
        - Direct mapping: Use raw state values with minimal processing
        - Normalized mapping: Scale all values to consistent ranges
        - Engineered features: Compute domain-specific derived quantities
        - Partial observability: Expose only subset of full state information

    Usage in Environment:
        The extractor is called after each integration step and during environment
        reset to provide the agent with current observations. It works in conjunction
        with the observation space to ensure consistent data formatting.

    Example:
        ```python
        class CSTRObservationExtractor(ObservationExtractor):
            @property
            def observation_space(self) -> gym.spaces.Box:
                return gym.spaces.Box(
                    low=np.array([0.0, 0.0, 0.0]),
                    high=np.array([1.0, 1.0, 2.0]),
                    shape=(3,)
                )

            def extract_observation(self, next_state: CSTRState) -> CSTRObservation:
                # Normalize concentrations by initial concentration
                c_a_norm = next_state.dae_state.c_a / next_state.dae_params.c_a_0
                c_b_norm = next_state.dae_state.c_b / next_state.dae_params.c_a_0
                # Normalize temperature by initial temperature
                t_norm = next_state.dae_state.T / next_state.dae_params.T_0
                return CSTRObservation(c_a=c_a_norm, c_b=c_b_norm, t=t_norm)
        ```

    Note:
        - The observation extractor only has access to the next_state (s'), not the
          previous state or action, unlike other extractors
        - Observations should be designed to provide sufficient information for
          optimal decision-making while remaining computationally efficient
    """

    @property
    @abstractmethod
    def observation_space(self) -> gym.spaces.Space:
        """
        Define the observation space for the RL environment.

        This property specifies the structure, bounds, and data types of observations
        that will be provided to RL agents. The space definition is crucial for
        RL algorithms to understand the expected input format and ranges.

        Returns:
            gym.spaces.Space: A gymnasium Space object defining the observation
                structure. Commonly a Box space for continuous observations,
                but can be Discrete, MultiDiscrete, or other space types.

        Example:
            For a CSTR with normalized concentrations and temperature:
            ```python
            return gym.spaces.Box(
                low=np.array([0.0, 0.0, 0.0]),    # Min values
                high=np.array([1.0, 1.0, 2.0]),   # Max values
                shape=(3,),                        # 3D observation
                dtype=np.float32
            )
            ```

        Note:
            The observation space bounds should match the actual range of values
            returned by extract_observation(). Violations can cause RL algorithm
            failures or suboptimal performance.
        """

    @abstractmethod
    def extract_observation(self, next_state: State) -> Observation:
        """
        Extract an observation from the environment state.

        This method processes the environment state to create an observation that
        will be provided to the RL agent. The extraction should focus on information
        relevant for decision-making while maintaining consistency with the defined
        observation space.

        Args:
            next_state: The complete environment state containing dae_state,
                dae_params, and non_dae_params. This represents the state
                after the most recent state update (s') due to the most recent
                action (a) or resetting.

        Returns:
            Observation: A concrete observation instance containing the processed
                information from the state. The observation should conform to
                the bounds and structure defined by observation_space.

        Example:
            Observation extraction might involve:
            - Extracting key state variables (concentrations, temperature)
            - Normalizing values by reference quantities
            - Computing derived features (conversion rates, efficiency metrics)
            - Adding time-based information if relevant

        Note:
            - This extractor only receives next_state, not the previous state or action
            - The returned observation should be within the bounds of observation_space
            - Extraction should be deterministic for reproducible behavior
        """
