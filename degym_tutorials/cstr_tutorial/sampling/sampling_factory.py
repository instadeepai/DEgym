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

"""A registry for sampling methods."""
from degym_tutorials.cstr_tutorial.sampling.sampling_strategies import SamplingStrategy


class SamplingStrategyFactory:
    """
    Factory class for creating sampling strategies.
    Allows dynamic registration of strategies for different distribution types.
    """

    _strategies: dict[str, type[SamplingStrategy]] = {}

    @classmethod
    def register_strategy(cls, distribution: str, strategy: type[SamplingStrategy]) -> None:
        """
        Register a new sampling strategy for a distribution.

        Args:
            distribution (str): The type of distribution (e.g., "choice", "normal", "uniform").
            strategy (SamplingStrategy): The strategy instance to handle the distribution.
        """
        cls._strategies[distribution] = strategy

    @classmethod
    def get_strategy(cls, distribution: str) -> type[SamplingStrategy]:
        """
        Retrieve the sampling strategy corresponding to the given distribution type.

        Args:
            distribution (str): The type of distribution.

        Returns:
            SamplingStrategy: An instance of the corresponding strategy.
        """
        if distribution not in cls._strategies:
            raise NotImplementedError(
                f"The configured sampling strategy '{distribution}' is not "
                f"found degym/sampling/sampling_factory.py. "
                f"Either implement your own or use and register it with the "
                f"factory or use one of the following\n:"
                f" {cls._strategies.keys()}"
            )
        return cls._strategies[distribution]
