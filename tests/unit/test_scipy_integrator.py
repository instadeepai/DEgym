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

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from degym.integrators import TimeSpan, ScipyIntegrator, ScipyIntegratorConfig
from degym.state import State
from degym.system_dynamics import ScipySystemDynamicsFn


@pytest.fixture
def resistance() -> float:
    return 5.0


@pytest.fixture
def capacity() -> float:
    return 0.1


@pytest.fixture
def true_solution_rc(
    resistance: float, capacity: float
) -> Callable[[float, float], list[float]]:
    def system_dynamics(time: float, u_0: float) -> list[float]:
        tau = resistance * capacity

        u = u_0 * np.exp(-time / tau)
        return [u]

    return system_dynamics


class RCSciPySystemDynamicsFn(ScipySystemDynamicsFn):
    def __init__(self, resistance: float, capacity: float):
        self.resistance = resistance
        self.capacity = capacity

    @staticmethod
    def __call__(
        state: NDArray[np.floating],
        parameters: NDArray[np.floating],
        action: NDArray[np.floating],
        time: float,
    ) -> NDArray[np.floating]:
        """
        The goal is to define a simple system whose true solution is known.

        We use an edited version of an RC circuit system.
        An RC system is defined by the following equations:

            (1): du/dt = -u / tau with tau = RC
            (2): u(t) = R * i(t)

        However in our case we drop the 2nd (algebraic) equation, to get:
            (1): du/dt = -u / RC
        """
        resistance, capacity = parameters
        u = state
        derivative = np.array([-u / (resistance * capacity)])
        return derivative


@pytest.fixture
def rc_scipy_dynamics_fn(
    resistance: float, capacity: float
) -> RCSciPySystemDynamicsFn:
    return RCSciPySystemDynamicsFn(capacity=capacity, resistance=resistance)


@dataclass(frozen=True)
class RCState(State):
    u: float

    def to_np_array(self) -> NDArray[np.floating]:
        return np.asarray([self.u])

    def __deepcopy__(self, memodict: dict) -> "RCState":
        state_copy = RCState(self.u)
        memodict[id(self)] = state_copy
        return state_copy


@pytest.mark.parametrize("start_time, duration, u_0", [(0, 5, 1.5)])
def test_scipy_integrate(
    true_solution_rc: Callable[[float, float], list[float]],
    start_time: float,
    duration: float,
    u_0: float,
    resistance: float,
    capacity: float,
    rc_scipy_dynamics_fn: ScipySystemDynamicsFn,
) -> None:
    integrator = ScipyIntegrator(
        system_dynamics=rc_scipy_dynamics_fn,
        integrator_config=ScipyIntegratorConfig(action_duration=duration),
    )

    initial_state = RCState(u=u_0)

    next_values = integrator.integrate(
        input_values=initial_state.to_np_array(),
        parameters=np.array([resistance, capacity]),
        action=np.array([]),
        time_span=TimeSpan(start_time=start_time, end_time=start_time + duration),
    )

    true_next_values = np.asarray(true_solution_rc(duration, u_0))

    np.testing.assert_allclose(
        next_values,
        true_next_values,
        atol=1e-6,
        rtol=0.0,
    )
