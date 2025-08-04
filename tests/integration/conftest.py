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

from typing import Any

import pytest


@pytest.fixture(scope="function")
def cstr_tutorial_env_config() -> dict[str, Any]:
    return {
        "env": "cstr_tutorial",
        "env_config": {
            "integrator": "scipy",
            "integrator_config": {
                "action_duration": 1,
                "method": "RK45",
                "rtol": 1e-6,
                "atol": 1e-8,
            },
            "random_seed": 0,
            "physical_parameters": {
                "fixed_values": {
                    "c_a_0": 0.3,
                    "c_p": 3.25,
                    "e_a": 41570,
                    "e_b": 45727,
                    "f": 0.0025,
                    "dh": 4157,
                    "k_0_a": 50_000,
                    "k_0_b": 100_000,
                    "r": 8.314,
                    "t_0": 300,
                    "v": 0.2,
                    "q_max": 5000,
                    "max_timestep": 10,
                },
                "sampled_values": {
                    "p": {"distribution": "choice", "choices": [780, 790], "size": 1}
                },
            },
        }
    }
