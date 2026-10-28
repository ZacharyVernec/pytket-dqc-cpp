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

from .._bridge import (
    call_with_python_rng,
    core,
    core_hypergraph_circuit,
    core_network,
    normalize_seed,
)
from ..circuits.distribution import Distribution
from ..circuits.hypergraph_circuit import HypergraphCircuit
from ._core_allocation import distribution_from_core
from .allocator import Allocator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pytket import Circuit
    from pytket_dqc_cpp.networks import NISQNetwork


class Random(Allocator):
    """Distribute hypergraph vertices onto servers at random. The resulting
    placement is valid, which is to say vertices will not be placed on servers
    once that are full.
    """

    def __init__(self) -> None:
        pass

    def allocate(self, circ: Circuit, network: NISQNetwork, **kwargs) -> Distribution:
        """Distribute ``circ`` onto ``network`` by randomly placing
        vertices onto servers. Qubit vertices are placed onto servers
        until the server is full. Gate vertices are placed on servers at random
        without restriction.

        :param circ: Circuit to distribute.
        :type circ: pytket.Circuit
        :param network: Network onto which ``circ`` should be distributed.
        :type network: NISQNetwork
        :return: Distribution of ``circ`` onto ``network``.
        :rtype: Distribution

        :key seed: Seed for randomness. Default is None, which reseeds Python's
            global ``random`` generator from OS entropy exactly as the
            reference does; the advanced state is installed back into
            ``random`` when the call returns.
        """

        dist_circ = HypergraphCircuit(circ)
        if not network.can_implement(dist_circ):
            raise Exception("This circuit cannot be implemented on this network.")

        # The core binding takes an unsigned integer; CPython's random.seed
        # uses abs(seed), including for negative integers.
        seed = normalize_seed(kwargs.get("seed", None))

        core_distribution = call_with_python_rng(
            core.random_allocate_with_state,
            core_hypergraph_circuit(dist_circ),
            core_network(network),
            seed,
            seed=seed,
        )
        return distribution_from_core(core_distribution, dist_circ, network)
