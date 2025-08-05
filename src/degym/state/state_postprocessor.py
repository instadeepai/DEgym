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

from degym.state.state import State


class StatePostprocessor(ABC):
    """
    Abstract base class for postprocessing states after numerical integration.

    The StatePostprocessor is responsible for transforming the state that emerges from
    numerical integration before it is used for observation extraction, reward computation,
    and other downstream operations. This postprocessing step occurs after the integrator
    has computed the next state from the differential equations.

    Purpose and Role:
        - Corrects or adjusts state data after numerical integration
        - Applies necessary transformations to ensure state validity
        - Reverses any preprocessing transformations applied earlier

    Common Postprocessing Operations:
        - Reverse scaling and normalization applied in preprocessing
        - Unit conversions back to original measurement systems
        - State variable clamping to enforce physical constraints
        - Validation and correction of integration artifacts

    Integration with Environment Flow:
        The StatePostprocessor is called after integration in the environment step() method:
        1. (before this class) Current state is preprocessed
        2. (before this class) Preprocessed state is passed to the integrator
        3. (before this class) Integration produces the raw next state
        4. StatePostprocessor transforms the integrated state
        5. Postprocessed state is used for observation and reward extraction

    Implementation Strategies:
        - Identity operation: Return state unchanged (most common case)
        - Reverse scaling: Undo normalization applied in preprocessing
        - Clamping: Ensure state variables remain within physical bounds
        - Validation: Check and correct numerical integration artifacts

    Usage in Environment:
        The postprocessor is called directly by the Environment class after integration
        to ensure the state is in the correct format for subsequent operations like
        observation extraction and reward computation.

    Example:
        ```python
        # Simple identity postprocessor (most common)
        class CSTRStatePostprocessor(StatePostprocessor):
            def postprocess_state(self, state: CSTRState) -> CSTRState:
                return state

        # Postprocessor that reverses preprocessing scaling
        class ScaledStatePostprocessor(StatePostprocessor):
            def postprocess_state(self, state: ReactorState) -> ReactorState:
                # Reverse the scaling applied in preprocessing
                unscaled_dae_state = ReactorDAEState(
                    c_a=state.dae_state.c_a * 1000.0,  # Convert back from kmol/L to mol/L
                    c_b=state.dae_state.c_b * 1000.0,
                    T=state.dae_state.T
                )
                return ReactorState(
                    dae_state=unscaled_dae_state,
                    dae_params=state.dae_params,
                    non_dae_params=state.non_dae_params
                )
        ```

    Note:
        - State postprocessing returns a State object; Hence, its scope is limited to that.
        - Any transformations applied should reverse preprocessing operations and ensure
        the State object is in the expected format for downstream components.
    """

    @abstractmethod
    def postprocess_state(self, state: State) -> State:
        """
        Postprocess the state after numerical integration.

        This method transforms the state that has been computed by numerical integration
        to prepare it for downstream operations like observation extraction and reward
        computation. The postprocessing should ensure the state is in the correct format
        and range expected by other environment components.

        Common postprocessing operations include:
        - Reversing scaling or normalization applied in preprocessing
        - Clamping values to physically realistic ranges
        - Converting units back to original coordinate systems
        - Validating and correcting numerical integration artifacts
        - Ensuring state consistency and physical constraints

        Args:
            state: The state produced by numerical integration containing dae_state,
                dae_params, and non_dae_params. This represents the raw output from
                the integrator before any corrections or transformations.

        Returns:
            State: The postprocessed state ready for downstream operations. Should
                maintain the same structure as the input state but with corrected
                or transformed values as needed for observation and reward extraction.

        Example:
            Postprocessing might involve:
            - Ensuring concentrations remain non-negative
            - Reverting temperature scaling applied in preprocessing
            - Converting pressure units back to standard values

        Note:
            - The returned state is a State object; Hence, its scope is limited to that.
            - Any transformations applied should reverse preprocessing operations and ensure
            the State object is in the expected format for downstream components.
        """
