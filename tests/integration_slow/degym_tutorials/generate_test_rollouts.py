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

"""
Script to generate and save environment rollouts, to use in test_reproduce_rollout_behaviour tests.

Requires a Hydra config YAML file to be saved in the directory degym/tests/integration_slow/configs.
The name of this YAML file must be specified via `--config-name` when running this script.
Additionally, the integrator (either 'diffeqpy' or 'scipy') must be specified via `+env_config.integrator`.
e.g.
    > uv run ./generate_test_rollouts.py --config-name 'cstr_tutorial' +env_config.integrator=diffeqpy
"""

import warnings
import numpy as np
import pickle
import os
import hydra
from hydra.core.hydra_config import HydraConfig
from pathlib import Path

from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf
from typing import Optional

from degym.environment import Environment
from degym_tutorials.cstr_tutorial.make_env import make_cstr_environment

CONFIG_PATH = Path(__file__).parent.resolve() / "configs"
DATA_PATH = Path(__file__).parent.resolve() / "data"


def get_transition_dict(
    dae_state_array: NDArray[np.floating],
    observation_array: NDArray[np.floating],
    info: dict[str, NDArray[np.floating] | float],
    action: Optional[NDArray[np.floating] | float] = None,
    reward: Optional[float] = None,
    terminated: Optional[bool] = None,
    truncated: Optional[bool] = None,
) -> dict[str, Optional[NDArray[np.floating] | float]]:
    """Return a dictionary containing all data related to this environment transition."""
    data_dict = {
        "dae_state": dae_state_array,
        "observation": observation_array,
        "action": action,
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
    }
    data_dict.update({f"info::{key}": value for key, value in info.items()})
    return data_dict


def generate_random_policy_rollout(environment: Environment) -> list[dict]:
    """
    Generate and return a full episode using a uniformly random policy.

    Args:
        environment: Environment object to use to generate a rollout.
    Returns:
        List containing one dictionary for each environment step, with each dictionary
            containing the action, the environment state, and all step outputs.
    """
    terminated = truncated = False
    observation_array, info = environment.reset()
    dae_state_array = environment.state.dae_state.to_np_array()
    transitions = [get_transition_dict(dae_state_array, observation_array, info=info)]

    # Retrieve action_space object from environment property; fix random seed
    action_space = environment.action_space
    action_space.seed(0)

    # For each step, sample an action from action_space; store data dictionary
    while not terminated and not truncated:
        action = action_space.sample()
        observation_array, reward, terminated, truncated, info = environment.step(action)
        dae_state_array = environment.state.dae_state.to_np_array()

        transition_dict = get_transition_dict(
            dae_state_array,
            observation_array,
            info=info,
            action=action,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
        )
        transitions.append(transition_dict)
    return transitions


@hydra.main(version_base=None, config_path=str(CONFIG_PATH), config_name=None)
def generate_test_rollouts(config: DictConfig) -> None:
    """
    Generate test rollouts for an environment.

    Note:
        - Environment configs specified by Hydra config file via the '--config-name' CLI argument.
        - Integrator specified via `+env_config.integrator` CLI argument.
    """
    # Create output directory for the 'config_name' provided
    config_name = HydraConfig.get().job.config_name
    output_dir = str(DATA_PATH / config_name)
    os.makedirs(output_dir, exist_ok=True)

    integrator_name = config.env_config.integrator
    if integrator_name not in ["diffeqpy", "scipy"]:
        raise ValueError(f"Integrator {integrator_name} not supported")

    # Instantiate environment using configs
    if OmegaConf.select(config, "env_config.random_seed", default=None) is None:
        raise ValueError(
            "env_config.random_seed must be specified in YAML file to ensure reproducibility."
        )

    environment = make_cstr_environment(config.env_config)

    # Generate and save random policy rollout
    random_rollout = generate_random_policy_rollout(environment)
    pickle.dump(random_rollout, open(f"{output_dir}/random_rollout_{integrator_name}.pkl", "wb"))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    generate_test_rollouts()
