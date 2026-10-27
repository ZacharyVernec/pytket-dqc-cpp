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
    core,
    core_hypergraph_circuit,
    core_network,
)
from ..circuits.distribution import Distribution
from ..circuits.hypergraph_circuit import HypergraphCircuit
from ._core_allocation import distribution_from_core
from .allocator import Allocator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pytket import Circuit
    from pytket_dqc_cpp.networks import NISQNetwork


def order_reducing_size(server_qubits: dict[int, list[int]]) -> dict[int, list[int]]:
    """Reorder ``server_qubits`` dictionary so that servers are in
    reducing size.

    :param server_qubits: Dictionary mapping servers to the qubits
        they contain.
    :type server_qubits: dict[int, list[int]]
    :return: Dictionary mapping servers to the qubits they contain,
        in order from largest to smallest server.
    :rtype: dict[int, list[int]]
    """

    # List of servers whose position in a ranking by size is unknown
    position_unknown = list(server_qubits.keys())

    # The order of the servers by their size
    order = []

    # Look through the list of servers n-1 times, where n is the number of
    # servers, for the largest. Add the largest found to the order list.
    for _ in range(len(server_qubits) - 1):
        # Initialise with largest at start of list of servers with unknown
        # position in the order by size.
        largest_i = 0
        largest = position_unknown[largest_i]

        # For each server with unknown position, check if it is the largest.
        for i, server in enumerate(position_unknown[1:]):
            if len(server_qubits[largest]) < len(server_qubits[server]):
                largest = server
                largest_i = i + 1

        # Append the largest found to the order list, and remove it from the
        # list of servers with unknown position.
        order.append(largest)
        position_unknown.pop(largest_i)

    # Add the remaining server to the order. Only one will remain.
    order.append(position_unknown[0])

    return {server: server_qubits[server] for server in order}


class Ordered(Allocator):
    """Distribute hypergraph vertices onto servers, populating the largest
    servers first until all vertices are assigned.
    """

    def __init__(self) -> None:
        pass

    def allocate(self, circ: Circuit, network: NISQNetwork, **kwargs) -> Distribution:
        """Distribute ``circ`` onto ``network`` by placing quibts onto
        servers, in decreasing order of size, until they are full.

        :param circ: Circuit to distribute.
        :type circ: pytket.Circuit
        :param network: Network onto which ``circ`` should be distributed.
        :type network: NISQNetwork
        :return: Distribution of ``circ`` onto ``network``.
        :rtype: Distribution
        """

        dist_circ = HypergraphCircuit(circ)
        if not network.can_implement(dist_circ):
            raise Exception("This circuit cannot be implemented on this network.")

        core_distribution = core.ordered_allocate(
            core_hypergraph_circuit(dist_circ), core_network(network)
        )
        return distribution_from_core(core_distribution, dist_circ, network)
