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


class StatePreprocessor(ABC):
    """
    Abstract base class for preprocessing states before numerical integration.

    The StatePreprocessor is responsible for transforming the current environment state
    before it is passed to the integrator for solving the differential equations. This
    preprocessing step occurs at the beginning of each environment step, before any
    numerical computation takes place.

    Purpose and Role:
        - Prepares state data for optimal numerical integration
        - Applies necessary transformations or corrections to state variables

    Common Preprocessing Operations:
        - Variable scaling and normalization for numerical stability
        - Unit conversions between different measurement systems
        - State variable clamping to physically realistic bounds
        - Smoothing or filtering of noisy state measurements

    Integration with Environment Flow:
        The StatePreprocessor is called early in the environment step() method:
        1. Current state is retrieved
        2. StatePreprocessor transforms the state
        3. Preprocessed state is passed to the integrator
        4. Integration produces the next state
        5. (beyond the scope of this class) StatePostprocessor may apply final corrections

    Implementation Strategies:
        - Identity operation: Return state unchanged (most common case)
        - Scaling: Normalize variables to improve numerical conditioning
        - Clamping: Ensure state variables remain within physical bounds
        - Validation: Check and correct invalid state configurations

    Usage in Environment:
        The preprocessor works in conjunction with the main Preprocessor class, which
        coordinates both state and action preprocessing before integration. It ensures
        that the state is in the optimal format for numerical computation.

    Example:
        ```python
        # Simple identity preprocessor (most common)
        class CSTRStatePreprocessor(StatePreprocessor):
            def preprocess_state(self, state: CSTRState) -> CSTRState:
                return state

        # Preprocessor with normalization
        class ScaledStatePreprocessor(StatePreprocessor):
            def preprocess_state(self, state: ReactorState) -> ReactorState:
                # Scale concentrations for better numerical stability
                scaled_dae_state = ReactorDAEState(
                    c_a=state.dae_state.c_a / 1000.0,  # Convert mol/L to kmol/L
                    c_b=state.dae_state.c_b / 1000.0,
                    T=state.dae_state.T
                )
                return ReactorState(
                    dae_state=scaled_dae_state,
                    dae_params=state.dae_params,
                    non_dae_params=state.non_dae_params
                )
        ```

    Note:
        - State preprocessing returns a State object; Hence, its scope is limited to that.
        - Any transformations applied should be mathematically consistent with the format of
        the State object and reversible if needed (in postprocessing this is reversed).
    """

    @abstractmethod
    def preprocess_state(self, state: State) -> State:
        """
        Preprocess the state before numerical integration.

        This method transforms the current environment state to prepare it for
        optimal numerical integration. The preprocessing should maintain the
        physical validity of the state while potentially improving numerical
        properties or ensuring compatibility with the integrator.

        Common preprocessing operations include:
        - Scaling variables for better numerical conditioning
        - Clamping values to physically realistic ranges
        - Converting units or coordinate systems
        - Computing derived variables needed for integration
        - Applying smoothing or filtering to reduce numerical noise

        Args:
            state: The current environment state containing dae_state, dae_params,
                and non_dae_params. This represents the complete system state
                before any numerical computation.

        Returns:
            State: The preprocessed state ready for integration. Should maintain
                the same structure as the input state but with potentially
                transformed values optimized for numerical computation.

        Example:
            For a CSTR system, preprocessing might involve:
            - Ensuring concentrations are non-negative
            - Normalizing temperature to a reference value
            - Scaling flow rates to improve integration stability
            - Converting pressure units if needed

        Note:
            The returned state should be compatible with the system dynamics
            function and integrator. Any transformations should preserve the
            mathematical relationships described by the differential equations.
            Most implementations simply return the state unchanged (identity operation).
        """
