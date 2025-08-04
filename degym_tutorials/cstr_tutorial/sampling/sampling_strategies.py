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

"""Defines sampling strategies and a factory for sampling parameters based on distributions."""
from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np
from numpy.typing import NDArray


class SamplingStrategy(ABC):
    """
    Base class for all sampling strategies.
    Defines an interface for sampling a parameter based on a specific distribution.

    Methods:
        sample(rng: np.random.Generator, config: Dict[str, Any]) -> NDArray:
            Abstract method to be implemented by subclasses to sample based on the configuration.
    """

    _required_keys: list[str]

    def __init__(self, random_generator: np.random.Generator, config: Dict[str, Any]):
        """
        Initialize the sampling strategy with a random number generator and configuration.

        Args:
            random_generator (np.random.Generator): The random number generator to use for sampling.
            config (Dict[str, Any]): The configuration dictionary containing parameters
                needed for sampling.
        """
        self._random_generator = random_generator
        self._config = config

        for key in self._required_keys:
            if key not in config:
                raise ValueError(
                    f"Provided config = {config} does not contain all required keys"
                    f"for this SamplingStrategy: {self._required_keys}. Please update config."
                )

    @abstractmethod
    def sample(self) -> NDArray:
        """
        Sample a value based on the given random number generator and configuration.

        Returns:
            NDArray: Sampled values as an array.
        """


class ChoiceSamplingStrategy(SamplingStrategy):
    """
    Sampling strategy for 'choice' distribution.

    Samples values by randomly selecting from the specified choices.
    """

    _required_keys = ["choices", "size"]

    def sample(self) -> NDArray:
        """
        Sample values from the provided choices.

        Returns:
            NDArray: Randomly selected choices.
        """
        return self._random_generator.choice(self._config["choices"], size=self._config["size"])


class NormalSamplingStrategy(SamplingStrategy):
    """Sampling strategy for 'normal' (Gaussian) distribution."""

    _required_keys = ["loc", "scale", "size"]

    def sample(self) -> NDArray:
        """
        Sample values from a normal distribution.

        Returns:
            NDArray: Sampled values from the normal distribution.
        """
        return self._random_generator.normal(
            self._config["loc"], self._config["scale"], size=self._config["size"]
        )


class UniformSamplingStrategy(SamplingStrategy):
    """Sampling strategy for 'uniform' distribution."""

    _required_keys = ["low", "high", "size"]

    def sample(self) -> NDArray:
        """
        Sample values from a uniform distribution.

        Returns:
            NDArray: Sampled values from the uniform distribution.
        """
        return self._random_generator.uniform(
            self._config["low"], self._config["high"], size=self._config["size"]
        )
