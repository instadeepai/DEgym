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
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PhysicalParameters(ABC):
    """
    Abstract base class for physical parameters; PhysicalParameters are all the parameters one
    needs to setup an environment and start an episde.

    Physical parameters represent the fundamental constants and properties that characterize a
    specific reactor configuration.

    Examples of physical parameters include:
    - Reactor's make/model or more detailed parameters like reactor geometry (volume,
    surface area, pipe dimensions, thermal conductivity, etc.)
    - Operating constraints (maximum flow rates, pressure limits, temperature ranges)
    - Reaction kinetics (activation energies, pre-exponential factors)
    - Feed composition (inlet concentrations, flow rates)


    Notes:
        - Physical parameters serve as the foundation for generating both DAE parameters and
        non-DAE parameters that are used in the simulation. They bridge the gap between
        real-world reactor specifications and the mathematical model parameters.
        For example, the model of the reactor could be a physical parameter which specifies the
        volume of the reactor which is used in the dynamical equations.
        - Physical parameters are immutable within an episode but are regenerated at each
        environment reset, allowing for domain randomization and robust policy training
        across different reactor configurations.
        - These parameters are created **once** at the start of an episode and remain constant
        during a single episode.
        - Although these parameters are immutable, they define dae_params and non_dae_params
        which can have time-dependent behaviour (See `return_dae_params` and
        `return_non_dae_params` in `environment.py`)
    """


class PhysicalParametersGenerator(ABC):
    """
    Abstract base class for generating physical parameters for reactor systems.

    The generator is responsible for creating instances of PhysicalParameters that define
    the characteristics of a reactor system for each episode. This enables domain
    randomization by sampling different reactor configurations, operating conditions,
    and system properties across training episodes.

    The generator can implement various sampling strategies:
    - Fixed parameters for consistent environments
    - Random sampling from specified distributions for robustness
    - Curriculum learning with gradually increasing complexity
    - Scenario-based sampling for specific operating conditions

    Usage:
        The generator is called at environment initialization and reset to create
        new physical parameters. These parameters are then used by other components
        to generate initial states, configure the dynamics, and set up extractors.

    Example:
        ```python
        # Generate parameters for a new episode
        rng = np.random.default_rng(seed=42)
        physical_params = generator.generate(rng)

        # Use parameters to create initial state
        initial_state = state_generator.generate(physical_params)
        ```
    """

    @abstractmethod
    def generate(self, rng: np.random.Generator) -> PhysicalParameters:
        """
        Generate a set of physical parameters describing the system.

        This method creates a new set of physical parameters that define the characteristics
        of a reactor system. The parameters should be consistent with
        each other and represent a realistic or desired reactor configuration.

        Args:
            rng: Random number generator for reproducible parameter sampling. Used when
                implementing stochastic parameter generation (e.g., sampling from
                distributions, adding noise to nominal values).

        Returns:
            PhysicalParameters: A concrete instance containing all the physical constants
                and properties needed to define the reactor system for this episode.

        Example:
            For a CSTR system, this might return parameters including reactor volume,
            flow rates, density, heat capacity, activation energies, and maximum heat
            input rates.

        Note:
            The generated parameters should be validated to ensure they represent a
            physically reasonable and safe operating configuration. Invalid parameter
            combinations could lead to numerical instabilities or unrealistic behavior.
        """
