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

from dataclasses import asdict
from typing import Any, Optional, Union

import numpy as np
from numpy.typing import NDArray
from pydantic import model_validator
from pydantic.dataclasses import dataclass
from degym.physical_parameters import PhysicalParameters, PhysicalParametersGenerator
from degym.utils import PydanticBaseModel

from degym_tutorials.cstr_tutorial.sampling import sampling_constructors


@dataclass(frozen=True)
class CSTRPhysicalParameters(PhysicalParameters):
    """
    Physical constants for the CSTR problem.

    Attributes:
        p: liquid density.
        c_a_0: Initial concentration of A.
        c_p: Heat capacity of the tank.
        e_a: Activation energy of A -> B
        e_b : Activation energy of B -> A
        F: Flow rate in and out of the reactor.
        dh: Heat of the reaction.
        k_0_a: Pre-exponential factor for A -> B
        k_0_b: Pre-exponential factor for B -> A
        R: Ideal gas constant
        T_0: Initial temperature in the reactor
        V: Tank volume.
        q_max: Max heat that can be applied at each timestep.
        max_timestep: Number of minutes to run.
    """

    p: float = 780  # kg/m³
    c_a_0: float = 0.3  # kmol/m³
    c_p: float = 3.25  # kJ/K/kg
    e_a: float = 41570  # kJ / kmol
    e_b: float = 45727  # kJ / kmol
    F: float = 0.0025  # m³/min
    dh: float = 4157  # kJ / kmol
    k_0_a: float = 50_000  # min-1
    k_0_b: float = 100_000  # min-1
    R: float = 8.314  # kJ / kmol −K
    T_0: float = 300  # K
    V: float = 0.2  # m³
    q_max: float = 5000  # kJ/min
    max_timestep: int = 600

    def to_np_array(self) -> NDArray[np.floating]:
        """Concatenate all the attributes."""
        return np.asarray(list(asdict(self).values()))


class CSTRPhysicalParametersGeneratorConfig(PydanticBaseModel):
    """
    Configuration for the CSTRPhysicalParametersGenerator, which allows
    for generating physical parameters with both fixed and sampled values.

    Args:
        fixed_values: Dict like {param_name1: value1, param_name2: value2, ...}.
        sampled_values: Dict specifying configs for each sampled value, like
                        { param_name1: {"distribution": "choice", "size": 1, "choices": _},
                          param_name2: {"distribution": "normal", "size": 1, "loc": _, "scale": _},
                          param_name3: {"distribution": "uniform", "size": 1, "low": _, "high": _},
                          ... }.
    """

    fixed_values: dict[str, Any]
    sampled_values: Optional[dict[str, dict[str, Any]]]

    @model_validator(mode="before")
    @classmethod
    def check_overlap(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Check for overlapping keys in fixed_values and sampled_values."""
        if values["fixed_values"] is not None and values["sampled_values"] is not None:
            overlap = set(values["fixed_values"].keys()).intersection(
                set(values["sampled_values"].keys())
            )
            if overlap:
                raise ValueError(
                    "The provided keys in fixed_values and sampled_values overlap in "
                    f"the following keys: {overlap}. Please provide unique keys for "
                    f"each dictionary."
                )
        return values


class CSTRPhysicalParametersGenerator(PhysicalParametersGenerator):
    """PhysicalParametersGenerator for the CSTR problem."""

    def __init__(self, config: CSTRPhysicalParametersGeneratorConfig):
        """
        Initialise CSTRPhysicalParametersGenerator with configs for generating
        physical parameters with fixed and sampled values.

        Args:
            config: Configuration for the physical parameters generator.
        """
        self._fixed_parameters = config.fixed_values
        self._sampled_parameters = config.sampled_values

    def _sample_variable_parameters(self, rng: np.random.Generator) -> dict[str, Any]:
        """
        Sample and return a value for each physical parameter variable specified
        in self.sample_parameters_config.

        Args:
            rng: Random number generator to use for sampling.
        """
        sampled_parameters: dict[str, Union[int, float, np.array]] = {}
        if self._sampled_parameters is None:
            return sampled_parameters

        for param_name, sampling_config in self._sampled_parameters.items():
            sampling_strategy = sampling_constructors.get_strategy(sampling_config["distribution"])
            sampling_strategy_instance = sampling_strategy(rng, sampling_config)
            sampled_parameters[param_name] = sampling_strategy_instance.sample()
        return sampled_parameters

    def generate(self, rng: np.random.Generator) -> CSTRPhysicalParameters:
        """
        Generate and return a set of physical parameters sampled for this episode.
        If self.sample_parameters_config is empty, returns the base parameters, else
        samples a value for these parameters and replaces the base value before returning.

        Args:
            rng: Random number generator to use for sampling.
        """
        values = self._fixed_parameters.copy()
        values.update(self._sample_variable_parameters(rng))
        return CSTRPhysicalParameters(
            p=values["p"],
            c_a_0=values["c_a_0"],
            c_p=values["c_p"],
            e_a=values["e_a"],
            e_b=values["e_b"],
            F=values["f"],
            dh=values["dh"],
            k_0_a=values["k_0_a"],
            k_0_b=values["k_0_b"],
            R=values["r"],
            T_0=values["t_0"],
            V=values["v"],
            q_max=values["q_max"],
            max_timestep=values["max_timestep"],
        )
