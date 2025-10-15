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

from degym_tutorials.cstr_tutorial.action_concrete_classes import (
    CSTRActionConverter,
    CSTRActionPreprocessor,
    CSTRActionRegulator,
)
from degym_tutorials.cstr_tutorial.environment import CSTREnvironment
from degym_tutorials.cstr_tutorial.extractors import (
    CSTRInfoExtractor,
    CSTRObservationExtractor,
    CSTRRewardExtractor,
    CSTRTerminatedExtractor,
    CSTRTruncatedExtractor,
)
from degym_tutorials.cstr_tutorial.physical_parameters import (
    CSTRPhysicalParametersGenerator,
    CSTRPhysicalParametersGeneratorConfig,
)
from degym_tutorials.cstr_tutorial.state_concrete_classes import (
    CSTRInitialStateGenerator,
    CSTRStatePostprocessor,
    CSTRStatePreprocessor,
)
from degym_tutorials.cstr_tutorial.system_dynamics.diffeqpy_dynamics import (
    CSTRDiffeqpySystemDynamics,
)
from degym.integrators import DiffeqpyIntegrator, DiffeqpyIntegratorConfig
from tests import skip_if_not_diffeqpy

@pytest.fixture(scope="function")
def physical_parameters_generator(
    physical_parameters_config: dict
) -> CSTRPhysicalParametersGenerator:
    config = CSTRPhysicalParametersGeneratorConfig(**physical_parameters_config)
    return CSTRPhysicalParametersGenerator(config=config)


@pytest.fixture(scope="function")
def initial_state_generator() -> CSTRInitialStateGenerator:
    return CSTRInitialStateGenerator()


@pytest.fixture(scope="function")
def state_preprocessor() -> CSTRStatePreprocessor:
    return CSTRStatePreprocessor()

@pytest.fixture(scope="function")
def action_preprocessor() -> CSTRActionPreprocessor:
    action_regulator = CSTRActionRegulator()
    action_converter = CSTRActionConverter()
    return CSTRActionPreprocessor(action_converter=action_converter, action_regulator=action_regulator)

@pytest.fixture(scope="function")
def integrator(cstr_system_dynamics: CSTRDiffeqpySystemDynamics) -> DiffeqpyIntegrator:
    integrator_config = DiffeqpyIntegratorConfig(action_duration=1.0)
    integrator = DiffeqpyIntegrator(
        system_dynamics=cstr_system_dynamics,
        integrator_config=integrator_config,
    )
    return integrator


@pytest.fixture(scope="function")
def cstr_system_dynamics() -> CSTRDiffeqpySystemDynamics:
    return CSTRDiffeqpySystemDynamics()


@pytest.fixture()
def max_timestep() -> int:
    return 1


@pytest.fixture()
def observation_extractor() -> CSTRObservationExtractor:
    return CSTRObservationExtractor()


@pytest.fixture()
def reward_extractor() -> CSTRRewardExtractor:
    return CSTRRewardExtractor()


@pytest.fixture()
def terminated_extractor(max_timestep: int) -> CSTRTerminatedExtractor:
    return CSTRTerminatedExtractor()


@pytest.fixture()
def truncated_extractor() -> CSTRTruncatedExtractor:
    return CSTRTruncatedExtractor()


@pytest.fixture()
def info_extractor() -> CSTRInfoExtractor:
    return CSTRInfoExtractor()


@pytest.fixture()
def state_postprocessor() -> CSTRStatePostprocessor:
    return CSTRStatePostprocessor()


@skip_if_not_diffeqpy
def test_reset_environment(
    physical_parameters_generator: CSTRPhysicalParametersGenerator,
    initial_state_generator: CSTRInitialStateGenerator,
    state_preprocessor: CSTRStatePreprocessor,
    action_preprocessor: CSTRActionPreprocessor,
    integrator: DiffeqpyIntegrator,
    info_extractor: CSTRInfoExtractor,
    observation_extractor: CSTRObservationExtractor,
    reward_extractor: CSTRRewardExtractor,
    truncated_extractor: CSTRTruncatedExtractor,
    terminated_extractor: CSTRTerminatedExtractor,
    state_postprocessor: CSTRStatePostprocessor,
) -> None:
    env = CSTREnvironment(
        physical_parameters_generator=physical_parameters_generator,
        initial_state_generator=initial_state_generator,
        state_preprocessor=state_preprocessor,
        action_preprocessor=action_preprocessor,
        integrator=integrator,
        info_extractor=info_extractor,
        reward_extractor=reward_extractor,
        observation_extractor=observation_extractor,
        truncated_extractor=truncated_extractor,
        terminated_extractor=terminated_extractor,
        seed=0,
        state_postprocessor=state_postprocessor,
    )
    obs, info = env.reset()
    # Extract observation values from output array
    c_a, c_b, t = obs

    physical_parameters = env._physical_parameters
    assert c_a == env.state.dae_state.c_a / physical_parameters.c_a_0
    assert c_b == 0.0
    assert t == 1.0
    assert info == {}


@skip_if_not_diffeqpy
def test_step_environment(
    physical_parameters_generator: CSTRPhysicalParametersGenerator,
    initial_state_generator: CSTRInitialStateGenerator,
    state_preprocessor: CSTRStatePreprocessor,
    action_preprocessor: CSTRActionPreprocessor,
    integrator: DiffeqpyIntegrator,
    info_extractor: CSTRInfoExtractor,
    observation_extractor: CSTRObservationExtractor,
    reward_extractor: CSTRRewardExtractor,
    truncated_extractor: CSTRTruncatedExtractor,
    terminated_extractor: CSTRTerminatedExtractor,
    state_postprocessor: CSTRStatePostprocessor,
) -> None:
    env = CSTREnvironment(
        physical_parameters_generator=physical_parameters_generator,
        initial_state_generator=initial_state_generator,
        state_preprocessor=state_preprocessor,
        action_preprocessor=action_preprocessor,
        integrator=integrator,
        info_extractor=info_extractor,
        reward_extractor=reward_extractor,
        observation_extractor=observation_extractor,
        truncated_extractor=truncated_extractor,
        terminated_extractor=terminated_extractor,
        seed=0,
        state_postprocessor=state_postprocessor,
    )
    physical_parameters = env._physical_parameters
    obs, reward, terminated, truncated, info = env.step(action=1.0)
    # Extract observation values from output array
    c_a, c_b, t = obs

    assert 0.0 <= c_a < 1.0
    assert 0.0 < c_b < 1.0
    np.testing.assert_allclose(
        c_a + c_b,
        1.0,
        atol=1e-6,
        rtol=0.0,
    )

    # observation is normalized, reward is not
    np.testing.assert_allclose(reward, c_b * physical_parameters.c_a_0)

    while not (terminated or truncated):
        _, _, terminated, truncated, _ = env.step(action=1.0)
    assert terminated  # max_timestep = 1 reached
    assert not truncated  # Never truncated
