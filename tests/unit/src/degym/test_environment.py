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

import pytest

from degym.environment import Environment


def test_non_overridability_of_init() -> None:
    with pytest.raises(TypeError):
        class SubClass(Environment):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

def test_non_overridability_of_step() -> None:
    with pytest.raises(TypeError):
        class SubClass(Environment):
            def step(self, *args: Any, **kwargs: Any) -> Any:
                pass


def test_non_overridability_of_reset() -> None:
    with pytest.raises(TypeError):
        class SubClass(Environment):
            def reset(self) -> None:
                pass

def test_non_overridability_of__extract_step_outputs() -> None:
    with pytest.raises(TypeError):
        class SubClass(Environment):
            def _extract_step_outputs(self, *args: Any, **kwargs: Any) -> Any:
                pass


def test_non_overridability_of_state() -> None:
    with pytest.raises(TypeError):
        class SubClass(Environment):
            @property
            def state(self) -> Any:
                pass


def test_non_overridability_of_current_time() -> None:
    with pytest.raises(TypeError):
        class SubClass(Environment):
            @property
            def current_time(self) -> Any:
                pass


def test_non_overridability_of_step_counter() -> None:
    with pytest.raises(TypeError):
        class SubClass(Environment):
            @property
            def step_counter(self) -> Any:
                pass


def test_non_overridability_of_observation_space() -> None:
    with pytest.raises(TypeError):
        class SubClass(Environment):
            @property
            def observation_space(self) -> Any:
                pass


def test_non_overridability_of_action_space() -> None:
    with pytest.raises(TypeError):
        class SubClass(Environment):
            @property
            def action_space(self) -> Any:
                pass
