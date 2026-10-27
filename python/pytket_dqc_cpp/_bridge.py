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
# Part of the `pytket_dqc` adapter.

"""Conversions between the pure-Python API objects and the compiled core.

Only plain data crosses the pybind11 boundary, so every call into ``_core`` goes
through one of the helpers below.  All conversions preserve insertion order of
dicts, lists and networkx nodes/edges, because that order feeds the
combinatorial decisions of the ported algorithms.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import networkx as nx
from pytket import Circuit, OpType
from pytket.circuit import Command, Op, Qubit

from . import _core as _core_module

# In the build-directory layout `_core` is the pure-Python loader, which exposes
# the extension as `_core.core`.  When the extension is installed *inside* the
# package, the shared object shadows `_core.py`, so `_core` is the extension
# itself and has no `core` attribute.
core = getattr(_core_module, "core", _core_module)

__all__ = [
    "OP_TYPE_BY_CODE",
    "command_from_core",
    "commands_from_core",
    "core_distribution",
    "core_graph",
    "core_hypergraph_circuit",
    "core_network",
    "core_placement",
    "nx_graph_from_core",
    "op_from_core",
    "op_type_from_code",
    "qubit_from_string",
    "qubit_lookup",
    "serialize_circuit",
    "sync_from_core",
    "to_nx_graph",
    "translate_core_error",
]

# ---------------------------------------------------------------------------
# OpType <-> integer code
# ---------------------------------------------------------------------------

#: Map from the integer code used by the core to the pytket ``OpType``.
OP_TYPE_BY_CODE: Dict[int, OpType] = {}
for _member in OpType.__members__.values():
    OP_TYPE_BY_CODE.setdefault(int(_member), _member)
del _member


def op_type_from_code(code: int) -> OpType:
    """Return the ``OpType`` whose integer value is ``code``.

    :raises ValueError: If ``code`` does not correspond to an ``OpType``.
    """
    try:
        return OP_TYPE_BY_CODE[int(code)]
    except KeyError:  # pragma: no cover - defensive
        raise ValueError(f"Unknown OpType code {code!r}.") from None


def op_from_core(op_type: Any, params: Sequence[float]) -> Op:
    """Rebuild a pytket ``Op`` from a core ``(op_type, params)`` pair.

    :param op_type: ``OpType`` instance or its integer code.
    :param params: Gate parameters, as floats.
    """
    optype = op_type if isinstance(op_type, OpType) else op_type_from_code(op_type)
    values = [float(p) for p in params]
    if not values:
        return Op.create(optype)
    if len(values) == 1:
        return Op.create(optype, values[0])
    return Op.create(optype, values)


# ---------------------------------------------------------------------------
# Circuit serialisation
# ---------------------------------------------------------------------------


def serialize_circuit(circuit: Circuit) -> Tuple[List[str], List[Tuple[str, int, List[str], List[float]]]]:
    """Convert a pytket ``Circuit`` into the core's plain-data form.

    :return: ``(qubits, commands)`` where ``qubits`` are the circuit's qubits as
        strings and each command is ``(op name, int(op.type), qubit strings,
        parameters)``.
    """
    qubits = [str(q) for q in circuit.qubits]
    commands = [
        (
            command.op.get_name(),
            int(command.op.type),
            [str(q) for q in command.qubits],
            [float(p) for p in command.op.params],
        )
        for command in circuit.get_commands()
    ]
    return qubits, commands


def qubit_lookup(circuit: Circuit) -> Dict[str, Qubit]:
    """Map ``str(qubit)`` to the original ``Qubit`` objects of ``circuit``.

    Used to rebuild core-produced ``Command`` objects without inventing new
    ``Qubit`` identities.
    """
    return {str(q): q for q in circuit.qubits}


def qubit_from_string(name: str, lookup: Optional[Mapping[str, Qubit]] = None) -> Qubit:
    """Rebuild a ``Qubit`` from its string form (``"q[0]"``, ``"node[1]"``...).

    ``lookup`` (see :func:`qubit_lookup`) takes precedence so that qubits keep
    their original identity; the parse is only a fallback.
    """
    if lookup is not None and name in lookup:
        return lookup[name]
    register, sep, index = name.rpartition("[")
    if sep and register and index.endswith("]"):
        return Qubit(register, int(index[:-1]))
    return Qubit(int(name))


def command_from_core(
    op_type: Any,
    params: Sequence[float],
    qubit_strings: Sequence[str],
    lookup: Optional[Mapping[str, Qubit]] = None,
) -> Command:
    """Rebuild a pytket ``Command`` from a core ``(op_type, params, qubits)``."""
    qubits = [qubit_from_string(q, lookup) for q in qubit_strings]
    return Command(op_from_core(op_type, params), qubits)


def commands_from_core(
    rows: Sequence[Sequence[Any]],
    lookup: Optional[Mapping[str, Qubit]] = None,
) -> List[Command]:
    """Rebuild a list of ``Command`` objects from core subcircuit rows."""
    return [
        command_from_core(op_type, params, qubits, lookup)
        for op_type, params, qubits in rows
    ]


# ---------------------------------------------------------------------------
# Object conversions
# ---------------------------------------------------------------------------


def _circuit_of(hypergraph_circuit: Any) -> Circuit:
    """Return the pytket circuit behind a ``HypergraphCircuit``."""
    circuit = getattr(hypergraph_circuit, "_circuit", None)
    if circuit is None:
        circuit = hypergraph_circuit.get_circuit()
    return circuit


def core_hypergraph_circuit(hypergraph_circuit: Any) -> Any:
    """Build a ``_core.HypergraphCircuit`` from a Python ``HypergraphCircuit``.

    The hypergraph is rebuilt from the Python ``hyperedge_list`` (order
    preserved) rather than from the core's own reconstruction, so any Python-side
    refinement is carried across faithfully.
    """
    core_circuit = core.HypergraphCircuit()
    qubits, commands = serialize_circuit(_circuit_of(hypergraph_circuit))
    core_circuit.from_circuit(qubits, commands)
    core_circuit.reset_hyperedges()
    for hyperedge in hypergraph_circuit.hyperedge_list:
        core_circuit.add_hyperedge(
            [int(v) for v in hyperedge.vertices], int(hyperedge.weight)
        )
    return core_circuit


def core_placement(placement: Any) -> Any:
    """Build a ``_core.Placement`` from a Python ``Placement``.

    Vertices are inserted in the Python dict's insertion order, which the core
    mirrors when iterating its own placement.
    """
    core_placement_obj = core.Placement()
    for vertex, server in placement.placement.items():
        core_placement_obj.set(int(vertex), int(server))
    return core_placement_obj


def core_graph(graph: nx.Graph) -> Any:
    """Build a ``_core.Graph`` from a networkx graph.

    Node and edge insertion order are preserved, as is the ``weight`` attribute
    (defaulting to ``1.0``).  Both properties affect the core's tie-breaking.
    """
    core_graph_obj = core.Graph()
    for node in graph.nodes():
        core_graph_obj.add_node(int(node))
    for u, v, data in graph.edges(data=True):
        core_graph_obj.add_edge(int(u), int(v), float(data.get("weight", 1.0)))
    return core_graph_obj


def nx_graph_from_core(graph: Any) -> nx.Graph:
    """Convert a ``_core.Graph`` back into a networkx ``Graph``."""
    nx_graph = nx.Graph()
    for node in graph.nodes():
        nx_graph.add_node(int(node))
    for u, v, weight in graph.edges():
        nx_graph.add_edge(int(u), int(v), weight=float(weight))
    return nx_graph


#: Backwards-friendly alias used by the graph-tools helpers.
to_nx_graph = nx_graph_from_core


def _server_qubits(network: Any) -> Dict[int, List[int]]:
    return {int(s): [int(q) for q in qs] for s, qs in network.server_qubits.items()}


def _server_ebit_mem(network: Any) -> Dict[int, int]:
    return {int(s): int(bound) for s, bound in network.server_ebit_mem.items()}


def core_network(network: Any) -> Any:
    """Build a ``_core.NISQNetwork`` from a Python ``NISQNetwork``."""
    return core.NISQNetwork(
        [[int(u), int(v)] for u, v in network.server_coupling],
        _server_qubits(network),
        _server_ebit_mem(network),
    )


def core_distribution(distribution: Any) -> Any:
    """Build a ``_core.Distribution`` from a Python ``Distribution``."""
    return core.Distribution(
        core_hypergraph_circuit(distribution.circuit),
        core_placement(distribution.placement),
        core_graph(distribution.network.get_server_nx()),
        _server_qubits(distribution.network),
        _server_ebit_mem(distribution.network),
    )


def sync_from_core(distribution: Any, core_distribution_obj: Any) -> None:
    """Copy hyperedges and placement from the core back onto ``distribution``.

    Refiners run entirely inside the core, so their effect must be mirrored onto
    the Python objects before anything else observes them.  The circuit itself
    (``_circuit``, ``_vertex_circuit_map``, ``_commands``) is unchanged by
    refinement and is left alone.
    """
    hypergraph_circuit = distribution.circuit
    core_circuit = core_distribution_obj.circuit

    hypergraph_circuit.hyperedge_list = []
    hypergraph_circuit.hyperedge_dict = {v: [] for v in hypergraph_circuit.vertex_list}
    hypergraph_circuit.vertex_neighbours = {
        v: set() for v in hypergraph_circuit.vertex_list
    }
    for hyperedge in core_circuit.hyperedge_list:
        hypergraph_circuit.add_hyperedge(
            vertices=[int(v) for v in hyperedge.vertices],
            weight=int(hyperedge.weight),
        )

    # Refinement only ever splits hyperedges, never moves vertices, so the
    # existing key order can (and must) be preserved: the core stores the
    # placement in a sorted map, while the Python dict order is the order in
    # which the vertices were placed.
    core_placement_map = {
        int(vertex): int(server)
        for vertex, server in core_distribution_obj.placement.placement.items()
    }
    placement_map = distribution.placement.placement
    distribution.placement.placement = {
        vertex: core_placement_map[vertex]
        for vertex in placement_map
        if vertex in core_placement_map
    }
    for vertex, server in core_placement_map.items():
        distribution.placement.placement.setdefault(vertex, server)


# ---------------------------------------------------------------------------
# Error translation
# ---------------------------------------------------------------------------

#: Messages the core raises when the communication-memory constraint is hit.
_CONSTRAINT_PREFIXES = (
    "Communication memory capacity of server ",
    "Could not implement the gate ",
    "The distribution could not be amended to satisfy the bound",
)

_SERVER_PATTERNS = (
    re.compile(r"Communication memory capacity of server (\d+)"),
    re.compile(r"non-home server (\d+)"),
)


def _constraint_server(message: str) -> Optional[int]:
    for pattern in _SERVER_PATTERNS:
        match = pattern.search(message)
        if match:
            return int(match.group(1))
    return None


def translate_core_error(exc: BaseException) -> BaseException:
    """Translate a core error into its ``pytket_dqc_cpp`` counterpart.

    The core raises its internal ``ConstraintException`` as a plain
    ``RuntimeError`` (pybind11 does not register the C++ type).  Those errors are
    converted back into :class:`pytket_dqc_cpp.utils.circuit_analysis.ConstraintException`,
    preserving the offending ``server`` when the message allows it.  Any other
    error is returned unchanged, so callers can write
    ``raise translate_core_error(exc) from exc``.
    """
    if not isinstance(exc, RuntimeError):
        return exc
    message = str(exc)
    if not message.startswith(_CONSTRAINT_PREFIXES):
        return exc

    # Imported lazily: `utils.circuit_analysis` imports this module's siblings.
    from .utils.circuit_analysis import ConstraintException

    return ConstraintException(message, _constraint_server(message))
