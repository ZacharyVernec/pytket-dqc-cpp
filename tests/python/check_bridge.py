#!/usr/bin/env python3
"""Gate 1 check: the package scaffold and the ``_core`` bridge.

Run from the repository root with the project virtualenv (the file matches the
``check_*.py`` pattern, so pytest never collects it):

    PYTHONPATH=python /home/ubuntu/pytket-dqc-cpp/.venv/bin/python \
        tests/python/check_bridge.py

The reference ``pytket_dqc`` objects are used as the "Python objects" fed to the
bridge, which is exactly how the Phase 2 classes will use it.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "python"))

import networkx as nx  # noqa: F401,E402
from pytket import Circuit, OpType  # noqa: E402

import pytket_dqc_cpp  # noqa: E402
from pytket_dqc_cpp import _bridge  # noqa: E402

core = _bridge.core

from pytket_dqc_cpp._delegation import (  # noqa: E402
    from_reference,
    reference_module,
    reset_warnings,
    to_reference,
    warn_once,
)
from pytket_dqc import HypergraphCircuit as PyHypergraphCircuit  # noqa: E402
from pytket_dqc import NISQNetwork as PyNISQNetwork  # noqa: E402
from pytket_dqc import Placement as PyPlacement  # noqa: E402
from pytket_dqc.circuits import Distribution as PyDistribution  # noqa: E402


def check_loader():
    """1.2: the loader exposes the extension both as `core` and mirrored."""
    assert hasattr(core, "HypergraphCircuit")
    assert pytket_dqc_cpp._core.HypergraphCircuit is core.HypergraphCircuit
    # The build-directory layout mirrors the extension as `_core.core`; the
    # installed layout has the shared object *as* `_core` and no `core` alias.
    assert getattr(pytket_dqc_cpp._core, "core", core) is core
    assert os.path.basename(_bridge.__file__) == "_bridge.py"
    print("loader: OK")


def check_serialisation():
    """1.3: circuit serialisation matches the core's expectations."""
    circ = Circuit(2).H(0).CU1(0.5, 0, 1).Rz(0.25, 1)
    qubits, commands = _bridge.serialize_circuit(circ)
    assert qubits == [str(q) for q in circ.qubits]
    assert commands[0][:3] == ("H", int(OpType.H), [str(circ.qubits[0])])
    # pytket 2.16 includes the parameters in `Op.get_name()`.
    assert commands[1][0].startswith("CU1") and commands[1][3] == [0.5]
    assert commands[2][0].startswith("Rz") and commands[2][3] == [0.25]

    # OpType <-> code is a bijection over the whole enum.
    for code, op_type in _bridge.OP_TYPE_BY_CODE.items():
        assert _bridge.op_type_from_code(code) is op_type

    # Rebuilding an Op reproduces the original name, type and parameters.
    for command in circ.get_commands():
        rebuilt = _bridge.op_from_core(int(command.op.type), command.op.params)
        assert rebuilt.type == command.op.type
        assert rebuilt.get_name() == command.op.get_name()
        assert [round(p, 10) for p in rebuilt.params] == [
            round(p, 10) for p in command.op.params
        ]
    print("serialisation: OK")


def check_hypergraph_circuit(hc):
    """1.3: `core_hypergraph_circuit` reproduces the Python hypergraph."""
    core_hc = _bridge.core_hypergraph_circuit(hc)
    assert list(core_hc.vertex_list) == list(hc.vertex_list)
    assert [
        (list(h.vertices), h.weight) for h in core_hc.hyperedge_list
    ] == [(list(h.vertices), h.weight) for h in hc.hyperedge_list]
    assert {
        v: [(list(h.vertices), h.weight) for h in hedges]
        for v, hedges in core_hc.hyperedge_dict.items()
    } == {
        v: [(list(h.vertices), h.weight) for h in hedges]
        for v, hedges in hc.hyperedge_dict.items()
    }

    # Embedded gates are rebuilt into equivalent `Command` objects.
    lookup = _bridge.qubit_lookup(hc.get_circuit())
    for hyperedge in hc.hyperedge_list:
        rebuilt = _bridge.commands_from_core(
            core_hc.get_hyperedge_subcircuit(
                [int(v) for v in hyperedge.vertices], int(hyperedge.weight)
            ),
            lookup,
        )
        expected = hc.get_hyperedge_subcircuit(hyperedge)
        assert [(c.op.get_name(), [str(q) for q in c.qubits]) for c in rebuilt] == [
            (c.op.get_name(), [str(q) for q in c.qubits]) for c in expected
        ], hyperedge
        assert [
            [round(p, 10) for p in c.op.params] for c in rebuilt
        ] == [[round(p, 10) for p in c.op.params] for c in expected], hyperedge


def check_placement_and_network(placement, network):
    """1.3: `core_placement` / `core_network` / `core_graph` preserve data."""
    core_pl = _bridge.core_placement(placement)
    assert dict(core_pl.placement) == dict(placement.placement)
    assert list(core_pl.placement.keys()) == list(placement.placement.keys())

    core_nw = _bridge.core_network(network)
    assert [list(pair) for pair in core_nw.server_coupling] == [
        list(pair) for pair in network.server_coupling
    ]
    assert dict(core_nw.server_qubits) == dict(network.server_qubits)
    assert list(core_nw.server_qubits.keys()) == list(network.server_qubits.keys())
    assert dict(core_nw.server_ebit_mem) == dict(network.server_ebit_mem)

    nx_graph = network.get_server_nx()
    core_g = _bridge.core_graph(nx_graph)
    assert list(core_g.nodes()) == list(nx_graph.nodes())
    assert [(u, v) for u, v, _ in core_g.edges()] == [
        (u, v) for u, v in nx_graph.edges()
    ]
    # Round trip back to networkx keeps nodes, edges and weights.
    back = _bridge.nx_graph_from_core(core_g)
    assert list(back.nodes()) == list(nx_graph.nodes())
    assert [(u, v) for u, v in back.edges()] == [
        (u, v) for u, v in nx_graph.edges()
    ]
    assert [d.get("weight", 1.0) for _, _, d in back.edges(data=True)] == [
        d.get("weight", 1.0) for _, _, d in nx_graph.edges(data=True)
    ]


def check_distribution_cost(cases):
    """1.3: cost read back from the core matches the reference."""
    for circ, network, placement_dict in cases:
        py_hc = PyHypergraphCircuit(circ)
        py_pl = PyPlacement(placement_dict)
        py_dist = PyDistribution(py_hc, py_pl, network)

        core_hc = _bridge.core_hypergraph_circuit(py_hc)
        core_pl = _bridge.core_placement(py_pl)
        core_g = _bridge.core_graph(network.get_server_nx())
        assert core.distribution_cost(core_hc, core_pl, core_g) == py_dist.cost()

        core_dist = _bridge.core_distribution(py_dist)
        assert core_dist.cost() == py_dist.cost()
        assert core_dist.is_valid() == py_dist.is_valid()

        for hyperedge in py_hc.hyperedge_list:
            assert core.hyperedge_cost(
                core_hc,
                list(hyperedge.vertices),
                hyperedge.weight,
                core_pl,
                core_g,
            ) == py_dist.hyperedge_cost(hyperedge)
    print("distribution cost: OK")


def check_sync_from_core(case):
    """1.3: refiners running in the core are mirrored back onto Python."""
    circ, network, placement_dict = case
    py_hc = PyHypergraphCircuit(circ)
    py_dist = PyDistribution(py_hc, PyPlacement(dict(placement_dict)), network)
    core_dist = _bridge.core_distribution(py_dist)

    changed = core.eager_h_type_merge_refine(core_dist)
    assert isinstance(changed, bool)
    assert _bridge.sync_from_core(py_dist, core_dist) is None

    assert [
        (list(h.vertices), h.weight) for h in py_hc.hyperedge_list
    ] == [
        (list(h.vertices), h.weight) for h in core_dist.circuit.hyperedge_list
    ]
    assert py_dist.placement.placement == dict(core_dist.placement.placement)
    # The Python cost recomputed from the synced hyperedges matches the core's.
    assert py_dist.cost() == core_dist.cost()
    return py_dist.cost()


def check_delegation():
    """1.4: lazy reference import, dict round trip and one warning per feature."""
    reset_warnings()
    import warnings as warnings_module

    with warnings_module.catch_warnings(record=True) as caught:
        warnings_module.simplefilter("always")
        warn_once("Routing")
        warn_once("Routing")
        warn_once("check_equivalence")
    assert len(caught) == 2, [str(w.message) for w in caught]
    assert all(issubclass(w.category, UserWarning) for w in caught)
    texts = sorted(str(w.message) for w in caught)
    assert any("Routing" in text for text in texts)
    assert any("check_equivalence" in text for text in texts)

    reference_module("utils.graph_tools")  # lazy import works

    circ = Circuit(3).add_gate(OpType.CU1, 1.0, [0, 1]).add_gate(
        OpType.CU1, 1.0, [0, 2]
    )
    network = PyNISQNetwork([[0, 1], [1, 2]], {0: [0], 1: [1], 2: [2]})
    placement = PyPlacement({0: 0, 1: 1, 2: 2, 3: 1, 4: 2})
    py_dist = PyDistribution(PyHypergraphCircuit(circ), placement, network)

    converted = to_reference(py_dist)
    assert type(converted).__module__.startswith("pytket_dqc")
    assert converted.cost() == py_dist.cost()

    try:
        back = from_reference(converted, kind="Distribution")
    except ModuleNotFoundError:  # pragma: no cover - Phase 1 only
        # The adapter classes themselves land in Phase 2/3.
        print("delegation: OK (from_reference skipped: adapter classes absent)")
        return
    assert type(back).__module__.startswith("pytket_dqc_cpp")
    print("delegation: OK")


def main():
    check_loader()
    check_serialisation()

    no_embedding = (
        Circuit(3)
        .add_gate(OpType.CU1, 1.0, [0, 1])
        .add_gate(OpType.CU1, 1.0, [0, 2]),
        PyNISQNetwork(
            [[0, 1], [1, 2], [1, 3], [2, 4]],
            {0: [0], 1: [1], 2: [2], 3: [3], 4: [4]},
        ),
        {0: 0, 1: 1, 2: 2, 3: 1, 4: 2},
    )

    euler = Circuit(5)
    euler.add_gate(OpType.CU1, [0.3], [0, 1])
    euler.H(0)
    euler.Rz(0.5, 0)
    euler.add_gate(OpType.CU1, [1.0], [0, 3])
    euler.H(0)
    euler.add_gate(OpType.CU1, [1.0], [0, 4])
    euler.Rz(0.5, 0)
    euler.H(0)
    euler.add_gate(OpType.CU1, [0.8], [0, 2])
    euler_case = (
        euler,
        PyNISQNetwork([[0, 1]], {0: [0], 1: [1, 2, 3, 4]}),
        {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1},
    )

    cases = [no_embedding, euler_case]

    py_hc = PyHypergraphCircuit(no_embedding[0])
    check_hypergraph_circuit(py_hc)
    check_hypergraph_circuit(PyHypergraphCircuit(euler))
    check_placement_and_network(PyPlacement(no_embedding[2]), no_embedding[1])
    check_distribution_cost(cases)

    merged_cost = check_sync_from_core(euler_case)
    assert isinstance(merged_cost, int)
    print(f"sync from core: OK (refined cost {merged_cost})")

    check_delegation()

    print("ALL BRIDGE CHECKS PASSED")


if __name__ == "__main__":
    main()
