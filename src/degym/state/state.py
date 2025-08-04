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

from abc import abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import model_validator

from degym.utils import PydanticBaseModel


class DAEState(PydanticBaseModel):
    """State of the DAE model."""

    @abstractmethod
    def to_np_array(self) -> NDArray[np.floating]:
        """Return state values as a concatenated numpy array."""

    @classmethod
    @abstractmethod
    def from_np_array(cls, np_array: NDArray[np.floating]) -> "DAEState":
        """Return an instance of the class initialized by a numpy array."""


class DAEParameters(PydanticBaseModel):
    """Parameters of the DAE model."""

    @abstractmethod
    def to_np_array(self) -> NDArray[np.floating]:
        """Return parameter values as a concatenated numpy array."""

    @classmethod
    @abstractmethod
    def from_np_array(cls, np_array: NDArray[np.floating]) -> "DAEParameters":
        """Return an instance of the class initialized by a numpy array."""


class NonDAEParameters(PydanticBaseModel):
    """Parameters that are not part of the DAE model."""

    @abstractmethod
    def to_np_array(self) -> NDArray[np.floating]:
        """Return parameter values as a concatenated numpy array."""

    @classmethod
    @abstractmethod
    def from_np_array(cls, np_array: NDArray[np.floating]) -> "NonDAEParameters":
        """Return an instance of the class initialized by a numpy array."""


class State(PydanticBaseModel):
    """
    State of the environment, where the state is all one needs to describe the environment at a
    given time step.

    The state of the environment is represented by the State class in DEgym which includes
      * dae_state,
      * dae_params, and
      * non_dae_params.

    To identify which parameters belong to which category, consider the following DAE:

    d/dt dae_state = f(dae_state, dae_params, dae_action)
    In the above equation:
      * dae_state: The variables that appear on the left-hand side of the DAE, e.g. in case of an
        ODE, these are the variables that are differentiated,
      * dae_action: The action as it appears in the DAE formulation, and
      * dae_params: The parameters/variables that are part of calculation of f which are not
        dae_state or dae_action.


    For further details on what the stats is (and what it is not!), refer to the
    [Terminology](./degym/docs/terminology.md) and
    [Step-by-step guide](./degym/docs/how_to_build_new_env.md).
    """

    dae_state: DAEState
    dae_params: DAEParameters
    non_dae_params: NonDAEParameters

    @model_validator(mode="before")
    @classmethod
    def check_attributes(cls, values: dict) -> dict:
        """
        Raise an error if the pairwise union of DAEState, DAEParameters, and NonDAEParameters is
        not empty.

        Args:
            values: Dictionary containing the values of the attributes.

        Raises:
            ValueError: If the pairwise union of DAEState, DAEParameters, and NonDAEParameters is
              not empty.
        """
        overlapping_attributes = find_all_common_keys(
            values.get("dae_state"), values.get("dae_params"), values.get("non_dae_params")
        )
        if overlapping_attributes:
            raise ValueError(
                "Attributes of DAEState, DAEParameters, and NonDAEParameters should not overlap."
                f"Here are the overlapping attributes: {overlapping_attributes}"
            )
        return values

    def to_np_array(self) -> NDArray[np.floating]:
        """Return state values as a concatenated numpy array."""
        return np.concatenate(
            [
                self.dae_state.to_np_array(),
                self.dae_params.to_np_array(),
                self.non_dae_params.to_np_array(),
            ]
        )

    @classmethod
    def from_np_array(cls, np_array: NDArray[np.floating]) -> "State":
        # from_np_arrays of DAEState, DAEParameters, and NonDAEParameters.
        """Return an instance of the class initialized"""
        raise NotImplementedError("This method is not implemented yet.")


def find_all_common_keys(*pydantic_dataclasses: Any) -> set[str]:
    """
    Find all the common keys between any pairs of the provided pydantic dataclasses.

    Args:
        *pydantic_dataclasses: List of instances of pydantic dataclasses.
    """
    common_keys = set()
    num_dicts = len(pydantic_dataclasses)

    for i in range(num_dicts):
        for j in range(i + 1, num_dicts):
            common_keys.update(
                pydantic_dataclasses[i].model_fields_set & pydantic_dataclasses[j].model_fields_set
            )

    return common_keys
