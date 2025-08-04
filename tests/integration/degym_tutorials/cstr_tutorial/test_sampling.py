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

from typing import Any

import numpy as np
import pytest

from degym_tutorials.cstr_tutorial.sampling import sampling_constructors
from degym_tutorials.cstr_tutorial.sampling.sampling_strategies import (
    ChoiceSamplingStrategy,
    NormalSamplingStrategy,
    SamplingStrategy,
    UniformSamplingStrategy,
)


@pytest.mark.parametrize("distribution", ["choice", "normal", "uniform", "dummy"])
def test_registered_sampling_constructors(distribution: str) -> None:
    if distribution == "dummy":
        with pytest.raises(NotImplementedError):
            sampling_constructors.get_strategy(distribution)
    else:
        sampling_constructors.get_strategy(distribution)


@pytest.mark.parametrize(
    "sampling_constructor, config",
    [
        (ChoiceSamplingStrategy, {"choices": [5, 10]}),
        (NormalSamplingStrategy, {"loc": 5, "scale": 1}),
        (UniformSamplingStrategy, {"low": 5, "high": 10}),
    ])
def test_sampling_strategies(
    sampling_constructor: type[SamplingStrategy], config: dict[str, Any]
) -> None:
    # Prepare configs, instantiate, and sample from sampler
    rng = np.random.default_rng(0)
    config.update({"size": 2})
    sampler = sampling_constructor(rng, config)
    sampler.sample()

    # Check raises ValueError due to missing one of _required_keys
    with pytest.raises(ValueError):
        del config["size"]
        sampling_constructor(rng, config)
