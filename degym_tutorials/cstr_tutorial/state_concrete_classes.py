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

from __future__ import annotations

import numpy as np
from degym.state import (
    DAEParameters,
    DAEState,
    InitialStateGenerator,
    State,
    StatePostprocessor,
    StatePreprocessor,
)
from numpy.typing import NDArray

from degym_tutorials.cstr_tutorial.physical_parameters import CSTRPhysicalParameters


class CSTRState(State):
    """The state of the CSTR problem."""

    dae_state: CSTRDAEState
    dae_params: CSTRDAEParameters
    non_dae_params: CSTRNonDAEParameters


class CSTRDAEState(DAEState):
    """
    State class for the CSTR problem.

    Attributes:
    c_a: Concentration of species A, from Eq. (1)
    c_b: Concentration of species B, from Eq. (2)
    T: Current temperature of system, from Eq. (3)
    k_a: Reaction rate of A -> B, from Eq. (4)
    k_b: Reaction rate of B -> A, from Eq. (5)
    """

    c_a: float
    c_b: float
    T: float

    def to_np_array(self) -> NDArray:
        """Concatenate all the attributes."""
        return np.asarray(
            [
                self.c_a,
                self.c_b,
                self.T,
            ]
        )

    @classmethod
    def from_np_array(cls, np_array: NDArray) -> "DAEState":
        """Return a new instance of the class from a numpy array."""
        return CSTRDAEState(
            c_a=np_array[0],
            c_b=np_array[1],
            T=np_array[2],
        )


class CSTRDAEParameters(DAEParameters):
    """
    DAE Parameters of the CSTR problem.

    Attributes:
        F: Flow rate in and out of the reactor, from Eq. (1).
        V: Tank volume, from Eq. (1).
        c_a_0: Initial concentration of A, from Eq. (1).
        p: Density of the fluid in the reactor, from Eq. (3).
        c_p: Heat capacity of the tank, from Eq. (3).
        T_0: Initial temperature of the reactor and the temperature of inflow, from Eq. (3).
        dh: Heat of the reaction, from Eq. (3).
        k_0_a: Pre-exponential factor for reaction A -> B, from Eq. (4).
        k_0_b: Pre-exponential factor for reaction B -> A, from Eq. (4).
        E_a_A: Activation energy for reaction A -> B, from Eq. (4).
        E_a_B: Activation energy for reaction B -> A, from Eq. (4).
        R: Ideal gas constant, from Eq. (4).
    """

    F: float  # m³/min from Eq. (1)
    V: float  # m³ from Eq. (1)
    c_a_0: float  # kmol/m³ from Eq. (1)
    p: float  # kg/m³ from Eq. (3)
    c_p: float  # kJ/K/kg from Eq. (3)
    T_0: float  # K from Eq. (3)
    dh: float  # kJ / kmol from Eq. (3)
    k_0_a: float  # Pre-exponential factor for reaction A -> B, from Eq. (4)
    k_0_b: float  # Pre-exponential factor for reaction B -> A, from Eq. (4)
    E_a_A: float  # Activation energy for reaction A -> B, from Eq. (4)
    E_a_B: float  # Activation energy for reaction B -> A, from Eq. (4)
    R: float  # Ideal gas constant, from Eq. (2)

    def to_np_array(self) -> NDArray:
        """Concatenate all the attributes."""
        return np.asarray(
            [
                self.F,
                self.V,
                self.c_a_0,
                self.p,
                self.c_p,
                self.T_0,
                self.dh,
                self.k_0_a,
                self.k_0_b,
                self.E_a_A,
                self.E_a_B,
                self.R,
            ]
        )

    @classmethod
    def from_np_array(cls, np_array: NDArray) -> "DAEParameters":
        """Return a new instance of the class from a numpy array."""
        return CSTRDAEParameters(
            F=np_array[0],
            V=np_array[1],
            c_a_0=np_array[2],
            p=np_array[3],
            c_p=np_array[4],
            T_0=np_array[5],
            dh=np_array[6],
            k_0_a=np_array[7],
            k_0_b=np_array[8],
            E_a_A=np_array[9],
            E_a_B=np_array[10],
            R=np_array[11],
        )


class CSTRNonDAEParameters(DAEParameters):
    """
    Non-DAE parameters for the CSTR problem.

    Attributes:
        q_max: Maximum heat that can be applied at each timestep.
        max_timestep: Number of minutes to run (default = 600).
        timestep: Current timestep.
    """

    q_max: float
    max_timestep: int  # Number of minutes to run (default = 600)
    timestep: int = 0  # Current timestep
    # counters etc.

    def to_np_array(self) -> NDArray:
        """Concatenate all the attributes."""
        return np.asarray(
            [
                self.q_max,
                self.max_timestep,
                self.timestep,
            ]
        )

    @classmethod
    def from_np_array(cls, np_array: NDArray) -> "DAEParameters":
        """Return a new instance of the class from a numpy array."""
        return CSTRNonDAEParameters(
            q_max=np_array[0],
            max_timestep=np_array[1],
            timestep=np_array[2],
        )


class CSTRStatePreprocessor(StatePreprocessor):  # noqa: D101
    def preprocess_state(self, state: CSTRState) -> CSTRState:
        """Preprocess the state."""
        return state


class CSTRStatePostprocessor(StatePostprocessor):  # noqa: D101
    def postprocess_state(self, state: CSTRState) -> CSTRState:
        """No preprocessing for the state in the CSTR problem."""
        return state


class CSTRInitialStateGenerator(InitialStateGenerator):
    """InitialStateGenerator class for the CSTR problem."""

    def generate(self, physical_parameters: CSTRPhysicalParameters) -> CSTRState:
        """Return the initial state given a set of physical parameters for the system."""
        dae_state = CSTRDAEState(
            c_a=physical_parameters.c_a_0,
            c_b=0.0,
            T=physical_parameters.T_0,
        )

        dae_params = CSTRDAEParameters(
            F=physical_parameters.F,
            V=physical_parameters.V,
            c_a_0=physical_parameters.c_a_0,
            p=physical_parameters.p,
            c_p=physical_parameters.c_p,
            T_0=physical_parameters.T_0,
            dh=physical_parameters.dh,
            k_0_a=physical_parameters.k_0_a,
            k_0_b=physical_parameters.k_0_b,
            E_a_A=physical_parameters.e_a,
            E_a_B=physical_parameters.e_b,
            R=physical_parameters.R,
        )
        non_dae_params = CSTRNonDAEParameters(
            q_max=physical_parameters.q_max,
            max_timestep=physical_parameters.max_timestep,
            timestep=0,
        )
        return CSTRState(dae_state=dae_state, dae_params=dae_params, non_dae_params=non_dae_params)
