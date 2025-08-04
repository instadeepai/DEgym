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
from degym_tutorials.cstr_tutorial.make_env import make_cstr_environment
from numpy.typing import NDArray
from tests import skip_if_not_diffeqpy


@pytest.fixture
def action_sequence() -> NDArray[np.floating]:
    return np.random.uniform(0, 1, size=100)


@skip_if_not_diffeqpy
def test_compare_integrators(cstr_tutorial_env_config: dict, action_sequence: list[float]) -> None:
    diffeqpy_env_config = cstr_tutorial_env_config["env_config"].copy()
    diffeqpy_env_config["integrator"] = "diffeqpy"
    diffeqpy_environment = make_cstr_environment(diffeqpy_env_config)

    scipy_env_config = cstr_tutorial_env_config["env_config"].copy()
    scipy_env_config["integrator"] = "scipy"
    scipy_environment = make_cstr_environment(scipy_env_config)

    diffeqpy_environment.reset()
    scipy_environment.reset()

    assert diffeqpy_environment.state == scipy_environment.state

    for action in action_sequence:
        diffeqpy_environment.step(action)
        scipy_environment.step(action)

        np.testing.assert_allclose(
            diffeqpy_environment.state.to_np_array(),
            scipy_environment.state.to_np_array(),
            atol=1e-6,
        )
