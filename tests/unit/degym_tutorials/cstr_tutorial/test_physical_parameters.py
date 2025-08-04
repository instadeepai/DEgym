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

import numpy as np
import pytest
from pydantic import ValidationError

from degym_tutorials.cstr_tutorial.physical_parameters import (
    CSTRPhysicalParameters,
    CSTRPhysicalParametersGenerator,
    CSTRPhysicalParametersGeneratorConfig,
)


def test_cstr_physical_parameters_does_not_smoke() -> None:
    physical_parameters = CSTRPhysicalParameters()
    np.testing.assert_array_equal(
        physical_parameters.to_np_array(),
        [
            physical_parameters.p,
            physical_parameters.c_a_0,
            physical_parameters.c_p,
            physical_parameters.e_a,
            physical_parameters.e_b,
            physical_parameters.F,
            physical_parameters.dh,
            physical_parameters.k_0_a,
            physical_parameters.k_0_b,
            physical_parameters.R,
            physical_parameters.T_0,
            physical_parameters.V,
            physical_parameters.q_max,
            physical_parameters.max_timestep,
        ],
    )

    # Test pydantic catches unwanted types
    with pytest.raises(ValidationError):
        CSTRPhysicalParameters(p=None)


def test_init_generator_config(physical_parameters_config: dict) -> None:
    fixed_values = physical_parameters_config["fixed_values"]
    sampled_values = physical_parameters_config["sampled_values"]
    CSTRPhysicalParametersGeneratorConfig(fixed_values=fixed_values, sampled_values=sampled_values)

    with pytest.raises(ValueError):
        # Create overlap between fixed and sampled values
        fixed_values["p"] = 780
        sampled_values["p"] = {"distribution": "choice", "size": 1, "choices": [780]}
        CSTRPhysicalParametersGeneratorConfig(
            fixed_values=fixed_values, sampled_values=sampled_values
        )


class TestCSTRPhysicalParametersGenerator:
    def test_init(self, physical_parameters_config: dict) -> None:
        generator_config = CSTRPhysicalParametersGeneratorConfig(**physical_parameters_config)
        CSTRPhysicalParametersGenerator(config=generator_config)

    def test_generate_deterministic(self, physical_parameters_config: dict) -> None:
        fixed_values = physical_parameters_config["fixed_values"]
        sampled_values = physical_parameters_config["sampled_values"]
        generator_config = CSTRPhysicalParametersGeneratorConfig(
            fixed_values=fixed_values, sampled_values=sampled_values
        )
        generator = CSTRPhysicalParametersGenerator(config=generator_config)

        physical_parameters = generator.generate(rng=np.random.default_rng(0))
        np.testing.assert_array_equal(
            physical_parameters.to_np_array(),
            [
                fixed_values["p"],
                fixed_values["c_a_0"],
                fixed_values["c_p"],
                fixed_values["e_a"],
                fixed_values["e_b"],
                fixed_values["f"],
                fixed_values["dh"],
                fixed_values["k_0_a"],
                fixed_values["k_0_b"],
                fixed_values["r"],
                fixed_values["t_0"],
                fixed_values["v"],
                fixed_values["q_max"],
                fixed_values["max_timestep"],
            ],
        )

        # Test pydantic catches unwanted types
        with (pytest.raises(ValidationError)):
            fixed_values["p"] = None
            config = CSTRPhysicalParametersGeneratorConfig(
                fixed_values=fixed_values, sampled_values=sampled_values
            )
            CSTRPhysicalParametersGenerator(config=config).generate(rng=np.random.default_rng(0))

    def test_generate_stochastic(self, physical_parameters_config: dict) -> None:
        # Add sampling configs for parameters (p, c_a_0, c_p)
        fixed_values = physical_parameters_config["fixed_values"].copy()
        del fixed_values["p"], fixed_values["c_a_0"], fixed_values["c_p"]
        sample_values_config = {
            "p": {"distribution": "choice", "choices": np.arange(10), "size": 1},
            "c_a_0": {"distribution": "normal", "loc": 0.1, "scale": 0.1, "size": 1},
            "c_p": {"distribution": "uniform", "low": 0.1, "high": 0.2, "size": 1},
        }

        # Instantiate generator and sample 10 times
        generator_config = CSTRPhysicalParametersGeneratorConfig(
            fixed_values=fixed_values,
            sampled_values=sample_values_config
        )
        generator = CSTRPhysicalParametersGenerator(config=generator_config)
        rng = np.random.default_rng(0)  # Test should always pass with this random seed
        sampled_params = [generator.generate(rng) for _ in range(10)]

        # Check that different values of (p, c_a_0, c_p) have been sampled
        for param_name in sample_values_config.keys():
            sampled_values = [getattr(sampled, param_name) for sampled in sampled_params]
            assert not np.all(np.isclose(sampled_values, sampled_values[0]))

        # Check that values of e_a are kept constant
        sampled_ea = [sampled.e_a for sampled in sampled_params]
        assert np.all(np.isclose(sampled_ea, physical_parameters_config["fixed_values"]["e_a"]))

    def test_generate_stochastic_wrong_keys(self, physical_parameters_config: dict) -> None:
        fixed_values = physical_parameters_config["fixed_values"]
        del fixed_values["p"]
        sampled_values_config = {"p": {"distribution": "choice", "size": 1}}
        generator_config = CSTRPhysicalParametersGeneratorConfig(
            fixed_values=fixed_values, sampled_values=sampled_values_config
        )
        generator = CSTRPhysicalParametersGenerator(config=generator_config)
        rng = np.random.default_rng(0)  # Test should always pass with this random seed
        with pytest.raises(ValueError):
            generator.generate(rng)
