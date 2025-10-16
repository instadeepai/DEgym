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

from degym.integrators import (
    DiffeqpyIntegrator,
    DiffeqpyIntegratorConfig,
    ScipyIntegrator,
    ScipyIntegratorConfig,
)

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
from degym_tutorials.cstr_tutorial.system_dynamics.scipy_dynamics import (
    CSTRScipySystemDynamics,
)


def make_cstr_environment(env_config: dict) -> CSTREnvironment:  # pylint: disable=too-many-locals
    """
    Instantiate a CSTR environment.

    NOTE: Passed to RLLib via the ray.tune.registry.register_env function,
          for which env_config is a required argument.
    """
    physical_parameter_generator_config = CSTRPhysicalParametersGeneratorConfig(
        **env_config["physical_parameters"]
    )
    physical_parameters_generator = CSTRPhysicalParametersGenerator(
        config=physical_parameter_generator_config
    )

    # Prepare preprocessing classes
    action_regulator = CSTRActionRegulator()
    action_converter = CSTRActionConverter()
    action_preprocessor = CSTRActionPreprocessor(
        action_converter=action_converter, action_regulator=action_regulator
    )
    state_preprocessor = CSTRStatePreprocessor()

    # Prepare integrator
    if env_config["integrator"] == "diffeqpy":
        system_dynamics = CSTRDiffeqpySystemDynamics()
        integrator_config = DiffeqpyIntegratorConfig(**env_config["integrator_config"])
        integrator = DiffeqpyIntegrator(
            system_dynamics=system_dynamics, integrator_config=integrator_config
        )
    elif env_config["integrator"] == "scipy":
        system_dynamics = CSTRScipySystemDynamics()
        integrator_config = ScipyIntegratorConfig(**env_config["integrator_config"])
        integrator = ScipyIntegrator(
            system_dynamics=system_dynamics, integrator_config=integrator_config
        )
    else:
        raise NotImplementedError(f"Integrator {env_config['integrator']} not implemented")

    # Prepare extractors + state postprocessor
    observation_extractor = CSTRObservationExtractor()
    reward_extractor = CSTRRewardExtractor()
    terminated_extractor = CSTRTerminatedExtractor()
    truncated_extractor = CSTRTruncatedExtractor()
    info_extractor = CSTRInfoExtractor()
    state_postprocessor = CSTRStatePostprocessor()

    # Instantiate CSTR Environment
    env = CSTREnvironment(
        physical_parameters_generator=physical_parameters_generator,
        initial_state_generator=CSTRInitialStateGenerator(),
        action_preprocessor=action_preprocessor,
        state_preprocessor=state_preprocessor,
        integrator=integrator,
        info_extractor=info_extractor,
        reward_extractor=reward_extractor,
        observation_extractor=observation_extractor,
        truncated_extractor=truncated_extractor,
        terminated_extractor=terminated_extractor,
        seed=env_config["random_seed"],
        state_postprocessor=state_postprocessor,
    )

    return env
