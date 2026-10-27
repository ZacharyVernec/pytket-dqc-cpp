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

"""Tree and routing-graph helpers backed by the compiled core."""

from graphlib import CycleError, TopologicalSorter

import networkx as nx  # type: ignore
from typing import Any, List, Tuple

from .._bridge import core, core_graph, nx_graph_from_core


def _core_graph_preserving_adjacency(graph: nx.Graph) -> Any:
    """Build a ``_core.Graph`` that reproduces networkx's adjacency order.

    ``_bridge.core_graph`` feeds edges to the core in ``graph.edges()`` order,
    which is grouped by node order.  That can reorder the neighbours of a node
    relative to the graph's actual insertion order, and the core's weighted
    tie-breaking depends on neighbour order (exactly like NetworkX's).  A
    topological ordering of the per-node adjacency constraints reproduces the
    original neighbour order for every node.
    """
    if graph.is_multigraph():
        return core_graph(graph)

    edges = list(graph.edges(data=True))
    edge_by_key = {}
    edge_by_index = {}
    for index, (u, v, data) in enumerate(edges):
        key = (min(u, v), max(u, v))
        edge_by_key[key] = index
        edge_by_index[index] = (u, v, data)

    sorter = TopologicalSorter()
    for node in graph.nodes():
        previous = None
        for neighbour in graph.neighbors(node):
            index = edge_by_key[(min(node, neighbour), max(node, neighbour))]
            sorter.add(index)
            if previous is not None and previous != index:
                sorter.add(index, previous)
            previous = index

    try:
        order = list(sorter.static_order())
    except CycleError:  # pragma: no cover - only for manually reordered graphs
        return core_graph(graph)

    core_graph_obj = core.Graph()
    for node in graph.nodes():
        core_graph_obj.add_node(int(node))
    for index in order:
        u, v, data = edge_by_index[index]
        core_graph_obj.add_edge(int(u), int(v), float(data.get("weight", 1.0)))
    return core_graph_obj


def direct_from_origin(G: nx.Graph, origin: int) -> List[Tuple[int, int]]:
    """Generates list of edges in the tree G. These edges are ordered and
    directed so as to expand from an origin node.

    :param G: The original undirected graph. Note that nodes of this graph
        should be integers. G should be a tree.
    :type G: nx.Graph
    :param origin: Node from which edges should spread.
    :type origin: int
    :raises Exception: Raised if the graph G is not a tree.
    :raises Exception: Raised if the graph G is not connected.
    :return: List of edges.
    :rtype: List[Tuple[int]]
    """

    try:
        edges = core.direct_from_origin(core_graph(G), int(origin))
    except RuntimeError as exc:
        # The core reports validation failures as ``RuntimeError``; upstream
        # raises a plain ``Exception`` with the same message.
        raise Exception(str(exc)) from None
    except IndexError:
        # The core raises ``IndexError`` when ``origin`` is not in the graph,
        # while upstream lets NetworkX raise ``NetworkXError``.
        raise nx.NetworkXError(f"The node {origin} is not in the graph.") from None
    return [(int(u), int(v)) for u, v in edges]


def steiner_tree(graph: nx.Graph, nodes: list[int]) -> nx.Graph:
    """Calls NetworkX's steiner_tree but manually manages the case of
    ``nodes`` being a singleton set, so that the singleton graph is
    returned, instead of the empty graph that NetworkX returns.
    """
    if len(nodes) == 0:
        raise Exception("No nodes have been provided")
    elif len(set(nodes)) == 1:
        # The core already returns the singleton graph for a single terminal.
        terminals = [int(n) for n in nodes]
    else:
        # NetworkX validates the terminal nodes before running the algorithm;
        # mirror its error instead of letting the core's ``IndexError`` leak.
        for node in nodes:
            if node not in graph:
                raise nx.NodeNotFound(f"Node {node} not found in graph")
        terminals = [int(n) for n in nodes]

    tree = nx_graph_from_core(
        core.steiner_tree(_core_graph_preserving_adjacency(graph), terminals)
    )

    # ``nx_graph_from_core`` always records a ``weight`` attribute. Upstream
    # returns a subgraph view of ``graph``, so restore the original edge data
    # (which may be empty, or hold ``weight`` plus any other attributes).
    for u, v in tree.edges():
        original = graph.get_edge_data(u, v)
        if original is not None:
            tree[u][v].clear()
            tree[u][v].update(original)

    return tree
