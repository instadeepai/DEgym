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

import pytest
from pydantic import ValidationError

from degym_tutorials.cstr_tutorial.action_concrete_classes import (
    CSTRAction,
    CSTRDAEAction,
)


def test_action_class_does_not_smoke() -> None:
    CSTRAction(q_normalized=0.0)
    CSTRAction(q_normalized=0.5)
    CSTRAction(q_normalized=2)

    # Test pydantic catches unwanted types
    with pytest.raises(ValidationError):
        CSTRAction(q_normalized=None)
    with pytest.raises(ValidationError):
        CSTRAction(q_normalized=CSTRAction(0.0))



def test_dae_action_class_does_not_smoke() -> None:

    CSTRDAEAction(q=500)
    CSTRDAEAction(q=5.0)
    # Test pydantic catches unwanted types
    with pytest.raises(ValidationError):
        CSTRDAEAction(q=None)
