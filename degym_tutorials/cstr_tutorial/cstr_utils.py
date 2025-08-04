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

"""Utility functions for the CSTR Simple Reaction problem."""
import numpy as np


def reaction_rate(k_0: float, e: float, r: float, t: float) -> float:
    """
    Compute initial reaction rate for the CSTR Simple Reaction problem.

    The reaction rate is a function of temperature based on the Arrhenius equation:
        k = k_0 * exp{−E_a / RT}

    Args:
        k_0: Pre-exponential factor.
        e: E_a, the activation energy of the reaction.
        r: R, Ideal gas constant
        t: T, Temperature in the reactor.
    Returns:
        k, the reaction rate.
    """
    rate: float = k_0 * np.exp(-e / (r * t))
    return rate
