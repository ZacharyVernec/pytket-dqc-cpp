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
# Adapted from `pytket_dqc` (commit 1ac175e) for the `pytket_dqc_cpp` adapter.

from __future__ import annotations

from ..circuits.distribution import Distribution
from .boundary_reallocation import BoundaryReallocation
from .refiner import Refiner


class DetachedGates(Refiner):
    """An alias for ``BoundaryReallocation`` with ``fixed_vertices`` set to
    the list of qubit-vertices and embedded gate-vertices. This refiner can
    optimise gate distribution so that detached gates may be used.

    The fixed vertex list is computed in Python exactly as upstream does, and
    the greedy rounds then run in the compiled core through
    :class:`.BoundaryReallocation`.  (The core's dedicated
    ``detached_gates_refine`` entry point is avoided because it crashes on
    degenerate hyperedges that contain no gate vertices, where upstream raises
    ``ValueError``.)
    """

    def refine(self, distribution: Distribution, **kwargs) -> bool:
        """An alias for boundary reallocation with no movements of
        qubit vertices or embedded gates. Key arguments (kwargs) are
        passed directly to ``BoundaryReallocation``
        the arguments it accepts are described in its documentation.

        :param distribution: Distribution to refine.
        :type distribution: Distribution

        :key seed: Seed for randomness. Default is None: unseeded calls
            continue Python's ambient ``random`` stream and return the advanced
            state to it, exactly as upstream ``pytket_dqc`` does. For
            reproducible results or cross-library parity, specify an explicit
            integer seed.

        :return: Distribution where the placement updated.
        :rtype: Distribution
        """

        fixed_vertices = (
            distribution.circuit.get_qubit_vertices()
            + distribution.circuit.get_all_h_embedded_gate_vertices()
        )

        return BoundaryReallocation().refine(
            distribution, fixed_vertices=fixed_vertices, **kwargs
        )
