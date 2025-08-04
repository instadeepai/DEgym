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
import gymnasium as gym

from degym_tutorials.cstr_tutorial.extractors import (
    CSTRInfoExtractor,
    CSTRObservationExtractor,
    CSTRRewardExtractor,
    CSTRTerminatedExtractor,
    CSTRTruncatedExtractor,
)
from degym_tutorials.cstr_tutorial.physical_parameters import (
    CSTRPhysicalParameters,
)
from degym_tutorials.cstr_tutorial.state_concrete_classes import CSTRState


def test_observation_extractor(
    cstr_state: CSTRState, physical_parameters: CSTRPhysicalParameters
) -> None:
    obs_extractor = CSTRObservationExtractor()
    obs = obs_extractor.extract_observation(
        next_state=cstr_state,
    )
    assert obs.c_a == cstr_state.dae_state.c_a / physical_parameters.c_a_0
    assert obs.c_b == cstr_state.dae_state.c_b / physical_parameters.c_a_0
    assert obs.t == cstr_state.dae_state.T / physical_parameters.T_0
    np.testing.assert_array_equal(
        obs.to_np_array(),
        [
            cstr_state.dae_state.c_a / physical_parameters.c_a_0,
            cstr_state.dae_state.c_b / physical_parameters.c_a_0,
            cstr_state.dae_state.T / physical_parameters.T_0,
        ],
    )

    assert isinstance(obs_extractor.observation_space, gym.spaces.Box)
    np.testing.assert_array_equal(obs_extractor.observation_space.low, np.array([0.0, 0.0, 0.0]))
    np.testing.assert_array_equal(
        obs_extractor.observation_space.high,
        np.array([1.0, 1.0, np.infty])
    )


def test_reward_extractor(cstr_state: CSTRState) -> None:
    reward_extractor = CSTRRewardExtractor()
    reward = reward_extractor.extract_reward(
        state=None,  # type: ignore[arg-type, unused-ignore]
        action=None,  # type: ignore[arg-type, unused-ignore]
        next_state=cstr_state
    )

    assert reward == cstr_state.dae_state.c_b


def test_terminated_extractor(cstr_state: CSTRState) -> None:
    terminated_extractor = CSTRTerminatedExtractor()
    terminated = terminated_extractor.extract_terminated(
        state=None,  # type: ignore[arg-type, unused-ignore]
        action=None,  # type: ignore[arg-type, unused-ignore]
        next_state=cstr_state
    )

    assert terminated

    cstr_state.non_dae_params.timestep = cstr_state.non_dae_params.max_timestep - 1

    terminated = terminated_extractor.extract_terminated(
    state=None,  # type: ignore[arg-type, unused-ignore]
    action=None,  # type: ignore[arg-type, unused-ignore]
    next_state=cstr_state
    )
    assert  not terminated


def test_truncated_extractor(cstr_state: CSTRState) -> None:
    truncated_extractor = CSTRTruncatedExtractor()
    truncated = truncated_extractor.extract_truncated(
        state=None,  # type: ignore[arg-type, unused-ignore]
        action=None,  # type: ignore[arg-type, unused-ignore]
        next_state=cstr_state
    )
    assert not truncated


def test_info_extractor(cstr_state: CSTRState) -> None:
    info_extractor = CSTRInfoExtractor()
    info = info_extractor.extract_info(
        state=None,  # type: ignore[arg-type, unused-ignore]
        action=None,  # type: ignore[arg-type, unused-ignore]
        next_state=cstr_state
    )
    assert info == {}
