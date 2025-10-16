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

from typing import TypeAlias

import gymnasium as gym
import numpy as np
from degym.action import Action, ActionConverter, ActionPreprocessor, ActionRegulator, DAEAction
from numpy.typing import NDArray
from pydantic.dataclasses import dataclass

from degym_tutorials.cstr_tutorial.state_concrete_classes import CSTRState

CSTRRawActionType: TypeAlias = float


@dataclass(frozen=True)
class CSTRAction(Action):
    """
    Action class for the CSTR problem.


    The agent selects a continuous action in the range [0, 1],
    which relates to heat being applied in the range [0, Q_max].
    """

    q_normalized: CSTRRawActionType


class CSTRDAEAction(DAEAction):
    """
    Action as it appears in the DAE system.

    Attributes:
        q: Heat being applied to the system (we use q and q_dot interchangeably!).
    """

    q: float

    def to_np_array(self) -> NDArray[np.floating]:
        """Return the action as a numpy array."""
        return np.array([self.q])

    @classmethod
    def from_np_array(cls, np_array: NDArray[np.floating]) -> "CSTRDAEAction":
        """
        Return a new instance of the class from a numpy array.

        Args:
            np_array: Numpy array with shape (1,).

        Returns:
            CSTRDAEAction: Instance of the class.

        Raises:
            ValueError: If the shape of the numpy array is not (1,).
        """
        if np_array.shape != (1,):
            raise ValueError(f"Expected shape (1,) but got {np_array.shape}")
        return CSTRDAEAction(q=np_array[0])


class CSTRActionConverter(ActionConverter):
    """Action converter for the CSTR problem."""

    def _action_to_dae_action(self, action: CSTRAction, state: CSTRState) -> CSTRDAEAction:
        """Multiply the RL action by q_max to denormalize."""
        q = action.q_normalized * state.non_dae_params.q_max
        return CSTRDAEAction(q=q)

    def _dae_action_to_action(self, dae_action: CSTRDAEAction, state: CSTRState) -> CSTRAction:
        """Divide the RL action by q_max to normalize."""
        q_normalized = dae_action.q / state.non_dae_params.q_max
        return CSTRAction(q_normalized=q_normalized)


class CSTRActionRegulator(ActionRegulator):
    """Action regulator for the CSTR problem."""

    def is_legal(
        self,
        dae_action: CSTRDAEAction,
        state: CSTRState,
    ) -> bool:
        """The action is legal if the heat stays within the bounds."""
        is_legal: bool = 0 <= dae_action.q <= state.non_dae_params.q_max
        return is_legal

    def convert_to_legal_action(self, dae_action: CSTRDAEAction, state: CSTRState) -> CSTRDAEAction:
        """If illegal action outside [0, q_max] then clamp it."""
        if dae_action.q <= 0:
            return CSTRDAEAction(q=0.0)
        if dae_action.q >= state.non_dae_params.q_max:
            return CSTRDAEAction(q=state.non_dae_params.q_max)
        return dae_action


class CSTRActionPreprocessor(ActionPreprocessor):
    """Action preprocessor for the CSTR problem."""

    def __init__(
        self,
        action_converter: CSTRActionConverter,
        action_regulator: CSTRActionRegulator,
    ):
        self.action_converter = action_converter
        self.action_regulator = action_regulator

    @property
    def action_space(self) -> gym.spaces.Box:
        """CSTR RL action space: add normalized heat in [0, 1]"""
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(1,))

    def preprocess_action(
        self,
        action: CSTRRawActionType,
        state: CSTRState,
    ) -> CSTRDAEAction:
        """
        Preprocess an action provided by an agent to the environment.

        Wraps in the CSTRAction class, then uses CSTRActionConverter to
        convert to a CSTRDAEAction, then uses CSTRActionRegulator to
        preprocess the dae action for applying to environment dynamics.

        Args:
            action: Raw action from the agent.
            state: Current state of the environment.

        Returns:
            CSTRDAEAction: Preprocessed action to be applied to the environment.
        """
        action = CSTRAction(q_normalized=action)

        dae_action: CSTRDAEAction = self.action_converter.action_to_dae_action(
            action=action, state=state
        )

        if self.action_regulator.is_legal(dae_action, state):
            return dae_action

        regulated_dae_action: CSTRDAEAction = self.action_regulator.convert_to_legal_action(
            dae_action, state=state
        )
        return regulated_dae_action
