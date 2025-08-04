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

from degym.physical_parameters import PhysicalParameters
from degym.state.state import State


class InitialStateGenerator(ABC):
    """
    Abstract base class for generating initial states at the start of episodes.

    The InitialStateGenerator is responsible for creating complete State instances that define
    the starting conditions for a reactor system. This includes
    setting up all three components of the state: dae_state, dae_params, and non_dae_params.

    Purpose and Role:
        - Converts high-level PhysicalParameters into concrete State instances
        - Ensures consistent initialization between different system components
        - Enables diverse starting conditions for robust RL training
        - Bridges the gap between reactor configuration and simulation state

    State Generation Process:
        1. Takes PhysicalParameters as input (reactor configuration)
        2. Creates DAEState with initial process variables (concentrations, temperature, etc.)
        3. Generates DAEParameters for the differential equations (flow rates, volumes, etc.)
        4. Sets up NonDAEParameters for simulation control (timestep limits, constraints, etc.)
        5. Combines all components into a complete State instance

    Examples of Initial State Components:
        - DAEState: Initial concentrations (c_a=c_a_0, c_b=0.0), initial temperature (T=T_0)
        - DAEParameters: Reactor volume (V), flow rate (F), kinetic constants (k_0_a, k_0_b)
        - NonDAEParameters: Maximum heat input (q_max), episode length (max_timestep)

    Usage in Environment:
        The generator is called during environment initialization and reset() to establish
        the starting state. It works in conjunction with PhysicalParametersGenerator to
        create diverse training scenarios while maintaining physical consistency.

    Example:
        ```python
        # Generate initial state for a CSTR episode
        physical_params = physical_param_generator.generate(rng)
        initial_state = state_generator.generate(physical_params)

        # State contains all necessary information to start simulation
        print(f"Initial concentration A: {initial_state.dae_state.c_a}")
        print(f"Reactor volume: {initial_state.dae_params.V}")
        print(f"Max timesteps: {initial_state.non_dae_params.max_timestep}")
        ```
    """

    @abstractmethod
    def generate(self, physical_parameters: PhysicalParameters) -> State:
        """
        Generate a complete initial state from physical parameters.

        This method creates a State instance that represents the starting conditions
        for a reactor system episode. It transforms high-level physical parameters
        into the specific state components needed for simulation.

        The method must create all three state components:
        1. DAEState: Initial values for differentiated variables (e.g., concentrations,
        temperature)
        2. DAEParameters: Parameters used in differential equations (e.g., flow rates,
        volumes)
        3. NonDAEParameters: Simulation control parameters (e.g., limits, timestep info)

        Args:
            physical_parameters: Physical configuration of the reactor system containing
                all the constants and properties needed to define the initial state.
                Examples include reactor geometry, material properties, initial
                compositions, and operating constraints.

        Returns:
            State: A complete state instance containing dae_state, dae_params, and
                non_dae_params, ready for use in environment initialization and
                simulation.

        Example:
            For a CSTR system, this method would:
            - Set initial concentrations based on feed composition
            - Initialize temperature to feed temperature
            - Configure flow rates, volumes, and kinetic parameters
            - Set up simulation limits and control parameters
        """
