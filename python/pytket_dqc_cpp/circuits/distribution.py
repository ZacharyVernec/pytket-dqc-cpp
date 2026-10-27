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

"""A hypergraph circuit together with a placement onto a NISQ network."""

from __future__ import annotations

from .hypergraph import Hyperedge, Vertex
from .hypergraph_circuit import HypergraphCircuit
from .._bridge import (
    core,
    core_distribution,
    core_graph,
    core_hypergraph_circuit,
    core_placement,
    sync_from_core,
    translate_core_error,
)
from ..networks import NISQNetwork
from ..placement import Placement
from ..utils import steiner_tree
from pytket import Circuit, OpType, Qubit
import networkx as nx  # type: ignore
from numpy import isclose


class Distribution:
    """Class containing all information required to generate pytket circuit."""

    def __init__(
        self,
        circuit: HypergraphCircuit,
        placement: Placement,
        network: NISQNetwork,
    ):
        """Initialisation function for Distribution

        :param circuit: Circuit to be distributed, including its hypergraph.
        :type circuit: HypergraphCircuit
        :param placement: A placement of the qubits and gates onto servers.
        :type placement: Placement
        :param network: Network onto which circuit is distributed.
        :type network: NISQNetwork
        :raises Exception: Raised if the placement is not valid for the
            circuit.
        :raises Exception: Raised if the placement is not valid for the
            packets.
        """

        self.circuit = circuit
        self.placement = placement
        self.network = network

    def __eq__(self, other) -> bool:
        """Check equality based on equality of components"""
        if isinstance(other, Distribution):
            return (
                self.circuit == other.circuit
                and self.placement == other.placement
                and self.network == other.network
            )
        return False

    def to_dict(self) -> dict[str, dict]:
        """Generate JSON serialisable dictionary representation of
        the Distribution.

        :return: JSON serialisable dictionary representation of
            the Distribution.
        :rtype: dict[str, dict]
        """
        return {
            "circuit": self.circuit.to_dict(),
            "placement": self.placement.to_dict(),
            "network": self.network.to_dict(),
        }

    @classmethod
    def from_dict(cls, distribution_dict: dict[str, dict]) -> Distribution:
        """Construct ``Distribution`` instance from JSON serialisable
        dictionary representation of the Distribution.

        :param distribution_dict: JSON serialisable dictionary
            representation of the Distribution
        :type distribution_dict: dict[str, dict]
        :return: Distribution instance constructed from distribution_dict.
        :rtype: Distribution
        """
        return cls(
            circuit=HypergraphCircuit.from_dict(distribution_dict["circuit"]),
            placement=Placement.from_dict(distribution_dict["placement"]),
            network=NISQNetwork.from_dict(distribution_dict["network"]),
        )

    def is_valid(self) -> bool:
        """Check that this distribution can be implemented."""

        # TODO: There may be some other checks that we want to do here to check
        # that the hypergraph is not totally nonsensical.
        return self.placement.is_valid(self.circuit, self.network)

    def cost(self) -> int:
        """Return the number of ebits required to implement this distribution.

        NOTE: this function does not guarantee that the distribution being
        analysed satisfies the bound to the link qubit registers. If you wish
        to take the bound into account, call ``to_pytket_circuit`` before
        calling this function. Alternatively, call ``ebit_cost`` on the
        distributed circuit.

        :return: The number of ebits used in this distribution.
        :rtype: int
        """
        if not self.is_valid():
            raise Exception("This is not a valid distribution")

        return core.distribution_cost(
            core_hypergraph_circuit(self.circuit),
            core_placement(self.placement),
            core_graph(self.network.get_server_nx()),
        )

    def non_local_gate_count(self) -> int:
        """Scan the distribution and return the number of non-local gates.
        A non-local gate is a 2-qubit gate that acts in a server not
        containing one of the two qubits it acts on.

        :return: A list of non-local gates
        :rtype: int
        """

        return len(self.non_local_gate_list())

    def non_local_gate_list(self) -> list[Vertex]:
        """Scan the distribution and return a list of non-local gates.
        A non-local gate is a 2-qubit gate that acts in a server not
        containing one of the two qubits it acts on.

        :return: The number of non-local gates
        :rtype: list[Vertex]
        """

        non_local_list = []

        for vertex in self.circuit.vertex_list:
            if not self.circuit.is_qubit_vertex(vertex):
                # Qubits gate acts on in original circuit
                q_1, q_2 = self.circuit.get_gate_of_vertex(vertex).qubits

                # Vertices of these qubits
                v_1 = self.circuit.get_vertex_of_qubit(q_1)
                v_2 = self.circuit.get_vertex_of_qubit(q_2)

                # Servers to which the qubits have been assigned
                s_1 = self.placement.placement[v_1]
                s_2 = self.placement.placement[v_2]

                # Server to which the gate has been assigned
                s_g = self.placement.placement[vertex]

                # Append if non-local
                if not ((s_1 == s_g) and (s_2 == s_g)):
                    non_local_list.append(vertex)

        return non_local_list

    def detached_gate_list(self) -> list[Vertex]:
        """Scan the distribution and return a list of detached gates in it.
        A detached gate is a 2-qubit gate that acts on link qubits on both
        ends; i.e. it is implemented away from both of its home servers.

        :return: A list of detached gates
        :rtype: list[Vertex]
        """

        detached_list = []

        for vertex in self.circuit.vertex_list:
            if not self.circuit.is_qubit_vertex(vertex):
                # Qubits gate acts on in original circuit
                q_1, q_2 = self.circuit.get_gate_of_vertex(vertex).qubits

                # Vertices of these qubits
                v_1 = self.circuit.get_vertex_of_qubit(q_1)
                v_2 = self.circuit.get_vertex_of_qubit(q_2)

                # Servers to which the qubits have been assigned
                s_1 = self.placement.placement[v_1]
                s_2 = self.placement.placement[v_2]

                # Server to which the gate has been assigned
                s_g = self.placement.placement[vertex]

                # Append if detached
                if not ((s_1 == s_g) or (s_2 == s_g)):
                    detached_list.append(vertex)

        return detached_list

    def detached_gate_count(self) -> int:
        """Scan the distribution and return the number of detached gates in it.
        A detached gate is a 2-qubit gate that acts on link qubits on both
        ends; i.e. it is implemented away from both of its home servers.

        :return: The number of detached gates
        :rtype: int
        """

        return len(self.detached_gate_list())

    def hyperedge_cost(self, hyperedge: Hyperedge, **kwargs) -> int:
        """First, we check whether the hyperedge requires H-embeddings to be
        implemented. If not, we calculate its cost by counting the number of
        edges in the Steiner tree connecting all required servers. Otherwise,
        the cost of implementing the hyperedge is calculated using an "as lazy
        as possible" (ALAP) algorithm which we expect not to be optimal, but
        decent enough. In the latter case, both reduction of ebit cost via
        Steiner trees and embedding are considered.

        Note: the cost only takes into account the ebits required to
        distribute the gates in the hyperedge; it does not consider the ebit
        cost of distributing the embedded gates, since that will be calculated
        by calling ``hyperedge_cost`` that the embedded gates belong to.
        However, we do guarantee that the correction gates added during the
        embedding will not require extra ebits when distributed.

        :param hyperedge: The hyperedge whose cost is to be calculated.
        :type hyperedge: Hyperedge

        :key server_tree: The connectivity tree that should be used.
            If not provided, this function will find a Steiner tree.
        :type server_tree: nx.Graph
        :key requires_h_embedded_cu1: If not provided, it is checked.
        :type requires_h_embedded_cu1: bool

        :return: The cost of the hyperedge.
        :rtype: int
        """

        tree = kwargs.get("server_tree", None)
        requires_h_embedded_cu1 = kwargs.get("requires_h_embedded_cu1", None)

        if hyperedge.weight != 1:
            raise Exception(
                "Hyperedges with weight other than 1 \
                 are not currently supported"
            )

        dist_circ = self.circuit
        placement_map = self.placement.placement

        # Extract hyperedge data
        shared_qubit = dist_circ.get_qubit_vertex(hyperedge)
        home_server = placement_map[shared_qubit]
        servers = [placement_map[v] for v in hyperedge.vertices]
        # Obtain the Steiner tree or check that the one given is valid
        if tree is None:
            tree = steiner_tree(self.network.get_server_nx(), servers)
        else:
            assert all(s in tree.nodes for s in servers)

        # If not known, check if H-embedding of CU1 is required
        if requires_h_embedded_cu1 is None:
            requires_h_embedded_cu1 = dist_circ.requires_h_embedded_cu1(hyperedge)

        # If not required, we can easily calculate the cost
        if not requires_h_embedded_cu1:
            return len(tree.edges)

        # Otherwise, we need to run ALAP
        else:
            # Collect all of the commands between the first and last gates in
            # the hyperedge. Ignore those that do not act on the shared qubit.
            # The gates in between embedded CU1 gates are guaranteed to be of
            # the form [Rz, H, Rz, H]; i.e. an explicit Euler decomposition.
            # Furthermore, the Rz gates at either side of an embedded CU1 gate
            # are squashed together (that's why there's no Rz at the end of
            # the Euler decomposition).
            commands = dist_circ.get_hyperedge_subcircuit(hyperedge)

            # We will use the fact that, by construction, the index of the
            # vertices is ordered (qubits first, then gates left to right)
            vertices = sorted(hyperedge.vertices)
            assert vertices.pop(0) == shared_qubit

            cost = 0
            currently_h_embedding = False  # Switched when finding a Hadamard
            connected_servers = {home_server}  # With access to shared_qubit
            for command in commands:
                if command.op.type == OpType.H:
                    currently_h_embedding = not currently_h_embedding

                elif command.op.type == OpType.Rz:
                    assert (
                        not currently_h_embedding
                        # Gate has a phase multiple of pi (i.e. I or Z)
                        or isclose(command.op.params[0] % 1, 0)
                        or isclose(command.op.params[0] % 1, 1)
                    )

                elif command.op.type == OpType.CU1:
                    if currently_h_embedding:  # The gate is to be H-embedded
                        assert isclose(command.op.params[0] % 2, 1)  # CZ gate

                        q_vertices = [
                            dist_circ.get_vertex_of_qubit(q) for q in command.qubits
                        ]
                        remote_vertex = [q for q in q_vertices if q != shared_qubit][0]
                        remote_server = placement_map[remote_vertex]

                        # The only two servers from ``connected_servers`` that
                        # are left intact are the ``home_server`` and the
                        # ``remote_server``. All other servers must be
                        # disconnected.
                        #
                        # NOTE: Done for the sake of simplicity. If we don't,
                        # certain correction gates wouldn't be free or be free
                        # only if the path of the EJPP process distributing
                        # the embedded gate could be reused to implement the
                        # correction gates. This is not trivial and not
                        # taken into account on Junyi's approach since it
                        # assumes only two servers / all-to-all connectivity
                        #
                        # NOTE: by the condition of H-embeddability, all gates
                        # that are being embedded simultaneously act on the
                        # same two distinct servers.
                        connected_servers = connected_servers.intersection(
                            {home_server, remote_server}
                        )
                        assert home_server != remote_server

                    # If the command does not match the vertex, then this
                    # CU1 gate is meant to be D-embedded
                    elif command != dist_circ.get_gate_of_vertex(vertices[0]):
                        pass  # Nothing needs to be done
                    else:  # Gate to be distributed (or already local)
                        # Get the server where the gate is to be implemented
                        gate_vertex = vertices.pop(0)
                        gate_server = placement_map[gate_vertex]
                        # If gate_server doesn't have access to shared_qubit
                        # update the cost, adding the necessary ebits
                        if gate_server not in connected_servers:
                            # For each server in ``connected_servers`` find
                            # the shortest path to ``gate_server`` and use
                            # the one that is shortest among them
                            best_path = None
                            for c_server in connected_servers:
                                connection_path = nx.shortest_path(
                                    tree, c_server, gate_server
                                )
                                # fmt: off
                                if (
                                    best_path is None
                                    or len(connection_path) < len(best_path)
                                ):
                                    best_path = connection_path
                                # fmt: on
                            assert best_path is not None
                            # The first element of the path is a ``c_server``
                            # so the actual cost is the length minus one
                            #
                            # NOTE: the ``best_path`` will only contain
                            # one server from ``connected_servers``. If
                            # it had two, the shortest path from the second
                            # would be shorter => contradiction
                            connected_servers.update(best_path)
                            cost += len(best_path) - 1
            # Sanity check: all gate vertices have been considered
            assert not vertices
            return cost

    def get_qubit_mapping(self) -> dict[Qubit, Qubit]:
        """Mapping from the names of the original circuit's qubits to the
        names of the hardware qubits, including whether they are link qubits
        or computation qubits and the module they belong to.

        :return: The mapping of circuit qubits to hardware qubits.
        :rtype: dict[Qubit, Qubit]
        """
        if not self.is_valid():
            raise Exception("The distribution of the circuit is not valid!")

        qubit_map = {}
        current_in_server = {s: 0 for s in self.network.get_server_list()}
        for v in self.circuit.get_qubit_vertices():
            server = self.placement.placement[v]
            circ_qubit = self.circuit.get_qubit_of_vertex(v)
            hw_qubit = Qubit(f"server_{server}", current_in_server[server])
            current_in_server[server] += 1
            qubit_map[circ_qubit] = hw_qubit

        return qubit_map

    def to_pytket_circuit(
        self, satisfy_bound: bool = True, allow_update: bool = False
    ) -> Circuit:
        """Generate the circuit corresponding to this `Distribution`.

        :param satisfy_bound: Whether the output circuit is only allowed
            to use as many link qubits per module as establisehd by the
            network's communication capacity (if provided). Optional
            parameter, defaults to True.
        :param allow_update: Whether the `Distribution` may be altered by
            this method, in order to make it satisfy the network's
            communication capacity (in case it has been bounded). Optional
            parameter, defaults to False.
        :raise ``ConstraintException``: If a server's communication capacity
            is exceeded, `satisfy_bound` was set to True and `allow_update`
            was set to False.
        """
        if not self.is_valid():
            raise Exception("The distribution of the circuit is not valid!")

        # The circuit generation (including the ALAP embedding, the
        # communication-memory checks and the cost sanity checks) lives in the
        # compiled core. When ``allow_update`` is set, the core splits
        # hyperedges in place, so the Python state is refreshed afterwards.
        core_dist = core_distribution(self)

        try:
            circuit = core_dist.to_pytket_circuit(satisfy_bound, allow_update)
        except RuntimeError as exc:
            raise translate_core_error(exc) from exc

        if allow_update:
            sync_from_core(self, core_dist)

        return circuit
