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

from degym.utils import NoOverrideMeta, no_override

class BaseClass(metaclass=NoOverrideMeta):
    @no_override
    def method_non_overridable(self) -> None:
        """This is the method which is decorated with no_override. In case this method is overridden
         in a subclass, it should raise an error."""
        pass

    def method_overridable(self) -> None:
        """This is the method which is not decorated with no_override. In case this method is overridden
         in a subclass, it should not raise an error."""
        pass

def test_no_override() -> None:
    """
    Test the no_override decorator.

    If the method decorated with no_override is overridden in a subclass, it should raise an error.
    If the method not decorated with no_override is overridden in a subclass, it should not raise
    an error.

    """
    # Test 1: Overriding a method which is decorated with no_override.
    with pytest.raises(TypeError):
        class ASubClass(BaseClass):
            def method_non_overridable(self) -> None:
                pass
    # Test 2: Overriding a method which is not decorated with no_override.
    class AnotherSubClass(BaseClass):
        def method_overridable(self) -> None:
            pass
