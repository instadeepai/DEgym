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

from degym_tutorials.cstr_tutorial.environment import CSTREnvironment
from degym_tutorials.cstr_tutorial.make_env import make_cstr_environment

from tests import skip_if_not_diffeqpy


@skip_if_not_diffeqpy
def test_make_cstr_environment_diffeqpy(cstr_tutorial_env_config: dict) -> None:
    cstr_tutorial_env_config["env_config"]["integrator"] = "diffeqpy"
    env = make_cstr_environment(cstr_tutorial_env_config["env_config"])
    assert isinstance(env, CSTREnvironment)


def test_make_cstr_environment_scipy(cstr_tutorial_env_config: dict) -> None:
    cstr_tutorial_env_config["env_config"]["integrator"] = "scipy"
    env = make_cstr_environment(cstr_tutorial_env_config["env_config"])
    assert isinstance(env, CSTREnvironment)
