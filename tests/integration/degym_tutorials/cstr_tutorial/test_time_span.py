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

"""Testing the time span of a CSTR reaction system."""

import numpy as np

from degym.integrators import TimeSpan


def test_time_span() -> None:
    """Test the time span defined for CSTR reaction system.

    In particular, test whether the start_time and the end_time of the time span are calculated
    correctly.
    """
    # Initialize the time span with random start_time and action_duration values.
    start_time, action_duration = np.random.uniform(0, 100, 2)
    time_span = TimeSpan(start_time=start_time, end_time=start_time + action_duration)

    assert time_span.start_time == start_time
    assert time_span.end_time == start_time + action_duration
