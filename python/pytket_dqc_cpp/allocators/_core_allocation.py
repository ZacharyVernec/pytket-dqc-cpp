# Copyright 2023 Quantinuum and The University of Tokyo
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Part of the `pytket_dqc_cpp` adapter.

"""Helpers shared by the core-backed allocators."""

from __future__ import annotations

from typing import Any

from ..circuits.distribution import Distribution
from ..placement.placement import Placement


def distribution_from_core(
    core_distribution: Any, dist_circ: Any, network: Any
) -> Distribution:
    """Wrap a core allocation result in the Python ``Distribution``.

    The core allocators only choose a placement (the hypergraph is untouched),
    so the Python circuit object is reused rather than rebuilt from the core.
    """
    placement = Placement(
        {
            int(vertex): int(server)
            for vertex, server in core_distribution.placement.placement.items()
        }
    )
    return Distribution(dist_circ, placement, network)
