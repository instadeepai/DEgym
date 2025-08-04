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
from typing import Optional

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray
from pydantic.dataclasses import dataclass
from degym.extractors import (
    InfoExtractor,
    Observation,
    ObservationExtractor,
    RewardExtractor,
    TerminatedExtractor,
    TruncatedExtractor,
)

from degym_tutorials.cstr_tutorial.action_concrete_classes import CSTRDAEAction
from degym_tutorials.cstr_tutorial.state_concrete_classes import CSTRState


@dataclass(frozen=True)
class CSTRObservation(Observation):
    """In the 'CSTR Simple Reaction' example, the agent observes measurements {[A], [B], T}."""

    c_a: float
    c_b: float
    t: float

    def to_np_array(self) -> NDArray[np.floating]:
        """Concatenate all the attributes."""
        return np.asarray(list(asdict(self).values()))


class CSTRObservationExtractor(ObservationExtractor):
    """ObservationExtractor for the 'CSTR Simple Reaction' example."""

    @property
    def observation_space(self) -> gym.spaces.Box:
        """Return observation space for the 'CSTR Simple Reaction' example."""
        return gym.spaces.Box(
            low=np.array([0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, np.infty]),
            shape=(3,),
        )

    def extract_observation(self, next_state: CSTRState) -> CSTRObservation:
        """Return the observation returned by the environment, from a given state."""
        c_0 = next_state.dae_params.c_a_0
        t_0 = next_state.dae_params.T_0
        return CSTRObservation(
            c_a=next_state.dae_state.c_a / c_0,
            c_b=next_state.dae_state.c_b / c_0,
            t=next_state.dae_state.T / t_0,
        )


class CSTRRewardExtractor(RewardExtractor):  # noqa: D101
    def extract_reward(
        self, state: CSTRState, action: CSTRDAEAction, next_state: CSTRState
    ) -> float:
        """Immediate reward is the concentration of the product, [B]."""
        return float(next_state.dae_state.c_b)


class CSTRTerminatedExtractor(TerminatedExtractor):  # noqa: D101
    def extract_terminated(
        self, state: CSTRState, action: CSTRDAEAction, next_state: CSTRState
    ) -> bool:
        """
        The state is terminal only if maximum timestep reached.
        Useful for the agent to be aware of this terminal condition,
        hence we treat it as part of the MDP (i.e. not truncation).
        """
        is_terminal: bool = (
            next_state.non_dae_params.timestep >= next_state.non_dae_params.max_timestep
        )
        return is_terminal


class CSTRTruncatedExtractor(TruncatedExtractor):  # noqa: D101
    def extract_truncated(
        self, state: CSTRState, action: CSTRDAEAction, next_state: CSTRState
    ) -> bool:
        """No truncation."""
        return False


class CSTRInfoExtractor(InfoExtractor):  # noqa: D101
    def extract_info(
        self,
        state: Optional[CSTRState],
        action: Optional[CSTRDAEAction],
        next_state: CSTRState,
    ) -> dict:
        """Return empty dict."""
        return {}
