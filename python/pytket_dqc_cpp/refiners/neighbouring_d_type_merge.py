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

from .._bridge import core, core_distribution, sync_from_core
from ..circuits.distribution import Distribution
from .refiner import Refiner


class NeighbouringDTypeMerge(Refiner):
    """:class:`.Refiner` merging neighbouring packets when no Hadamard act
    between them. For each qubit in the circuit, and given the packets
    ordered as they appear in the corresponding hypergraph, this refiner
    will attempt to merge packets appearing consecutively in that ordering.

    The merge scan runs in the compiled core; the resulting hypergraph is
    synced back onto ``distribution`` immediately after the call.
    """

    def refine(self, distribution: Distribution, **kwargs) -> bool:
        """Merges neighbouring packets when no Hadamard acts between them.

        :param distribution: Distribution whose packets should be merged.
        :type distribution: Distribution
        :return: True if a refinement has been performed. False otherwise.
        :rtype: bool
        """

        core_dist = core_distribution(distribution)
        refinement_made = bool(core.neighbouring_d_type_merge_refine(core_dist))
        sync_from_core(distribution, core_dist)
        return refinement_made
