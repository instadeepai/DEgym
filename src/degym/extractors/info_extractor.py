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
from typing import Optional

from degym.action import DAEAction
from degym.state import State


class InfoExtractor(ABC):
    """
    Abstract base class for extracting diagnostic information from state transitions.

    The InfoExtractor is responsible for collecting additional information that is not
    used by the agent for optimization (not used in learning) but is valuable for
    monitoring, debugging, analysis, and understanding agent behavior. This information
    is returned in the 'info' dictionary of the environment's step and reset methods.

    Purpose and Role:
        - Provides diagnostic and monitoring information for analysis
        - Enables debugging and understanding of environment behavior
        - Collects metrics for performance evaluation and system monitoring
        - Supports logging and visualization of training progress

    Common Information Types:
        - Performance metrics: Efficiency, productivity, energy consumption
        - Intermediate calculations: Reaction rates, heat transfer rates, etc.
        - Constraint violations: Safety limits exceeded, operating bounds violated
        - System health: Equipment status, degradation measures, fault indicators
        - Debugging data: Solver statistics, integration errors, convergence flags

    Integration with Environment Flow:
        The InfoExtractor is called in both step() and reset() methods:
        1. For step(): Called after state is updated, and hence (s, a, s') information
        is available
        2. For reset(): Called after environment reset where state is available
        3. Info dictionary is returned to the agent alongside other step outputs
        4. Agent or training infrastructure can use info for logging and analysis

    Implementation Strategies:
        - Minimal info: Return empty dictionary for production environments
        - Rich diagnostics: Include detailed metrics for development and debugging
        - Configurable verbosity: Adjust information level based on use case
        - Performance monitoring: Track key performance indicators over time

    Design Considerations:
        - Info collection should not significantly impact performance
        - Include information useful for debugging and analysis
        - Consider memory usage for long episodes or large info dictionaries

    Example:
        ```python
        class CSTRInfoExtractor(InfoExtractor):
            def extract_info(self, state: Optional[CSTRState],
                           action: Optional[CSTRDAEAction],
                           next_state: CSTRState) -> dict:
                info = {}

                # Performance metrics
                info['conversion'] = next_state.dae_state.c_b / next_state.dae_params.c_a_0
                info['temperature'] = next_state.dae_state.T

                # Economic indicators
                if action is not None:
                    info['energy_cost'] = action.q * 0.01  # Cost per KJ

                # Safety monitoring
                info['temp_violation'] = next_state.dae_state.T > 400.0

                return info
        ```

    Note:
        - InfoExtractor is the only extractor used in both step() and reset() methods
        - For reset(), state and action are None (only next_state is available)
        - Info is not used by the agent for optimization (not used in learning)
    """

    @abstractmethod
    def extract_info(
        self, state: Optional[State], action: Optional[DAEAction], next_state: Optional[State]
    ) -> dict:
        """
        Extract diagnostic information from the environment state transition.

        This method collects additional information that is useful for monitoring,
        debugging, and analysis but is not part of the core RL loop. The information
        is returned as a dictionary that can be used by training infrastructure,
        logging systems, or analysis tools.

        Args:
            state: The environment state at the beginning of the step (s), or None
                during reset. Contains dae_state, dae_params, and non_dae_params
                representing the system configuration before action application.
            action: The DAE action taken during the step (a), or None during reset.
                Represents the control inputs applied to the system.
            next_state: The environment state at the end of the step or after reset (s').
                Contains the system configuration after applying the action and
                integrating dynamics, or the initial state after reset.

        Returns:
            dict: A dictionary containing diagnostic information.

        Example:
            Info extraction might include:
            - System performance metrics (efficiency, productivity, yields)
            - Safety and constraint monitoring (distance to violation, alarms)
            - Economic indicators (breakdown of costs/energy consumption/material usage)
            - Debugging information (solver statistics, integration errors)

        Note:
            - This method is called in both step() and reset() functions
            - During reset(), state and action are None (only next_state is available)
            - Return empty dict if no diagnostic information is needed
        """
