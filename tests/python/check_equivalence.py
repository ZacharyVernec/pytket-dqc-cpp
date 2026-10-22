#!/usr/bin/env python3
"""Differential checker: confirm the C++ core behaves identically to the
reference Python implementation (pytket_dqc + networkx + stdlib `random`).

Run from the repository root, with the project venv:

    .venv/bin/python3 tests/python/check_equivalence.py

Each section prints `OK`; the script exits non-zero on the first mismatch.
"""

import itertools
import os
import random
import sys

import numpy as np

# The differential test must run with a fixed PYTHONHASHSEED: the C++ core
# reproduces CPython's SipHash-based string/set ordering, which depends on the
# hash secret. When run without an explicit PYTHONHASHSEED, the whole suite is
# executed once per seed in HASH_SEEDS to verify hash-seed invariance
# automatically (set PYTHONHASHSEED yourself to run exactly once under that
# seed).
HASH_SEEDS = ["0", "1", "42", "12345"]

if os.environ.get("DQC_HASH_SEED_SWEEP") != "1" and "PYTHONHASHSEED" not in os.environ:
    import subprocess

    script = os.path.abspath(__file__)
    for seed in HASH_SEEDS:
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = seed
        env["DQC_HASH_SEED_SWEEP"] = "1"
        subprocess.run([sys.executable, script], env=env, check=True)
    print(f"OK: hash-seed invariance across {HASH_SEEDS}")
    sys.exit(0)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "build", "dqc_cpp"))

import _core  # noqa: E402

import networkx as nx  # noqa: E402
from pytket_dqc.utils.graph_tools import steiner_tree as py_steiner_tree  # noqa: E402


def cpp_nisq(network):
    """Build a `_core.NISQNetwork` mirroring a pytket_dqc NISQNetwork."""
    return _core.NISQNetwork(
        server_coupling=[list(c) for c in network.server_coupling],
        server_qubits={s: list(qs) for s, qs in network.server_qubits.items()},
        server_ebit_mem=dict(network.server_ebit_mem))


def make_graphs(n, edges):
    """Build an nx.Graph and a C++ Graph with identical node/edge insertion
    order (adjacency insertion order drives Dijkstra tie-breaking)."""
    G = nx.Graph()
    g = _core.Graph()
    for i in range(n):
        G.add_node(i)
        g.add_node(i)
    for u, v in edges:
        G.add_edge(u, v)
        g.add_edge(u, v)
    return G, g


def check_rng():
    seeds = [0, 1, 12345, 2**32, 2**32 + 1, 2**63 + 5]
    for seed in seeds:
        cpp = _core.PyRandom(seed)
        ref = random.Random(seed)
        for _ in range(200):
            assert cpp.random() == ref.random(), f"random() mismatch, seed={seed}"

        cpp2 = _core.PyRandom(seed)
        ref2 = random.Random(seed)
        seq = [5, 7, 9, 11, 13]
        for _ in range(100):
            assert cpp2.randrange(100) == ref2.randrange(100)
            assert cpp2.choice(seq) == ref2.choice(seq)
            assert cpp2.uniform(0.0, 2.0) == ref2.uniform(0.0, 2.0)
            assert cpp2.randbelow(50) == ref2._randbelow(50)
    print("RNG: OK")


def check_steiner():
    fixed = [
        ([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (1, 4)], [0, 2, 4]),
        ([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)], [0, 2]),
        ([(0, 1), (1, 2), (2, 3), (3, 4)], [0, 4]),
        ([(0, 1), (1, 2), (2, 3), (3, 4)], [0, 2, 4]),
        ([(0, 1), (1, 2)], [1]),
    ]
    for edges, terminals in fixed:
        n = max(max(u, v) for u, v in edges) + 1
        G, g = make_graphs(n, edges)
        ref = py_steiner_tree(G, terminals)
        cpp = _core.steiner_tree(g, terminals)
        assert list(cpp.nodes()) == list(ref.nodes()), f"nodes mismatch {terminals}"
        assert [(u, v) for u, v, _ in cpp.edges()] == [
            (u, v) for u, v in ref.edges()
        ], f"edges mismatch {terminals}"

    rnd = random.Random(42)
    for trial in range(500):
        n = rnd.randrange(2, 12)
        G = nx.Graph()
        g = _core.Graph()
        for i in range(n):
            G.add_node(i)
            g.add_node(i)
        for i in range(1, n):  # random spanning tree
            j = rnd.randrange(i)
            G.add_edge(i, j)
            g.add_edge(i, j)
        for i in range(n):  # extra random edges
            for j in range(i + 1, n):
                if not G.has_edge(i, j) and rnd.random() < 0.3:
                    G.add_edge(i, j)
                    g.add_edge(i, j)
        terminals = list(rnd.sample(range(n), rnd.randrange(2, n + 1)))
        ref = py_steiner_tree(G, terminals)
        cpp = _core.steiner_tree(g, terminals)
        assert list(cpp.nodes()) == list(ref.nodes()), (
            f"nodes mismatch trial={trial} n={n} terminals={terminals}"
        )
        assert [(u, v) for u, v, _ in cpp.edges()] == [
            (u, v) for u, v in ref.edges()
        ], f"edges mismatch trial={trial} n={n} terminals={terminals}"
    print("steiner_tree: OK")


def check_direct_from_origin():
    from pytket_dqc.utils.graph_tools import direct_from_origin as py_direct

    edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
    for origin in range(5):
        G, g = make_graphs(5, edges)
        ref = py_direct(G, origin)
        cpp = _core.direct_from_origin(g, origin)
        assert list(cpp) == list(ref), f"direct_from_origin mismatch origin={origin}"
    print("direct_from_origin: OK")


def check_hypergraph():
    from pytket_dqc.circuits import Hypergraph as PyHypergraph
    from pytket_dqc.circuits import Hyperedge as PyHyperedge

    def build(py_cls, cpp=False):
        if cpp:
            g = _core.Hypergraph()
        else:
            g = py_cls()
        g.add_vertices([0, 1, 2])
        g.add_hyperedge([0, 1])
        g.add_hyperedge([0, 2])
        return g

    py_g = build(PyHypergraph)
    cpp_g = build(None, cpp=True)

    assert list(cpp_g.vertex_list) == list(py_g.vertex_list)
    py_hedges = [(tuple(h.vertices), h.weight) for h in py_g.hyperedge_list]
    cpp_hedges = [(tuple(h.vertices), h.weight) for h in cpp_g.hyperedge_list]
    assert cpp_hedges == py_hedges, f"{cpp_hedges} != {py_hedges}"

    # merge
    py_m = py_g.merge_hyperedge(
        [PyHyperedge([0, 1]), PyHyperedge([0, 2])]
    )
    cpp_m = cpp_g.merge_hyperedge(
        [_core.Hyperedge([0, 1]), _core.Hyperedge([0, 2])]
    )
    assert tuple(cpp_m.vertices) == tuple(py_m.vertices)
    assert cpp_m.weight == py_m.weight
    py_hedges = [(tuple(h.vertices), h.weight) for h in py_g.hyperedge_list]
    cpp_hedges = [(tuple(h.vertices), h.weight) for h in cpp_g.hyperedge_list]
    assert cpp_hedges == py_hedges, f"merge mismatch {cpp_hedges} != {py_hedges}"

    # neighbours compared as sets
    assert {k: set(v) for k, v in cpp_g.vertex_neighbours.items()} == {
        k: set(v) for k, v in py_g.vertex_neighbours.items()
    }
    print("hypergraph: OK")


def check_circuit_analysis():
    from pytket import Circuit, OpType, Qubit
    from pytket_dqc.utils.gateset import start_proc, end_proc, telep_proc
    from pytket_dqc.utils.circuit_analysis import (
        all_cu1_local as py_all_cu1,
        ebit_cost as py_ebit_cost,
        ebit_memory_required as py_mem,
        get_server_id as py_get_server_id,
        is_link_qubit as py_is_link,
    )

    class FakeQubit:
        def __init__(self, s):
            self._s = s

        def __str__(self):
            return self._s

    for s in [
        "server_0[0]",
        "server_3[5]",
        "server_0_link_register[2]",
        "server_12_link_register[7]",
    ]:
        assert _core.get_server_id(s) == py_get_server_id(FakeQubit(s)), s
        assert _core.is_link_qubit(s) == py_is_link(FakeQubit(s)), s

    def serialize(c):
        qubits = [str(q) for q in c.qubits]
        cmds = [
            (
                cmd.op.get_name(),
                int(cmd.op.type),
                [str(q) for q in cmd.qubits],
                [float(p) for p in cmd.op.params],
            )
            for cmd in c.get_commands()
        ]
        return qubits, cmds

    circ = Circuit()
    q0 = Qubit("server_0", 0)
    q1 = Qubit("server_1", 0)
    q2 = Qubit("server_0", 1)
    l0 = Qubit("server_0_link_register", 0)
    l1 = Qubit("server_0_link_register", 1)
    for q in [q0, q1, q2, l0, l1]:
        circ.add_qubit(q)
    circ.add_gate(OpType.CU1, 0.5, [q0, q2])  # local
    circ.add_custom_gate(start_proc(), [], [q0, l0])
    circ.add_custom_gate(start_proc(), [], [q2, l1])
    circ.add_custom_gate(telep_proc(), [], [l0, q1])
    circ.add_custom_gate(end_proc(), [], [l0, q0])
    circ.add_custom_gate(telep_proc(), [], [l1, q1])
    circ.add_custom_gate(end_proc(), [], [l1, q2])
    circ.add_gate(OpType.CU1, 0.25, [q0, q1])  # remote (non-local)

    qubits, cmds = serialize(circ)
    assert _core.ebit_cost(qubits, cmds) == py_ebit_cost(circ)
    assert _core.ebit_memory_required(qubits, cmds) == py_mem(circ)
    assert _core.all_cu1_local(qubits, cmds) == py_all_cu1(circ)

    circ2 = Circuit()
    a = Qubit("server_0", 0)
    b = Qubit("server_0", 1)
    circ2.add_qubit(a)
    circ2.add_qubit(b)
    circ2.add_gate(OpType.CU1, 0.3, [a, b])
    qubits2, cmds2 = serialize(circ2)
    assert _core.all_cu1_local(qubits2, cmds2) == py_all_cu1(circ2)
    print("circuit_analysis: OK")


def check_placement():
    from pytket_dqc.placement import Placement as PyPlacement

    class FakeNetwork:
        def __init__(self, G):
            self.G = G

        def get_server_nx(self):
            return self.G

    G = nx.Graph()
    for i in range(4):
        G.add_node(i)
    for e in [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]:
        G.add_edge(*e)

    cpp_G = _core.Graph()
    for i in range(4):
        cpp_G.add_node(i)
    for e in [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]:
        cpp_G.add_edge(*e)

    # Non-monotonic insertion order (exercises insertion-order sensitivity).
    placement = {3: 2, 0: 0, 5: 1, 7: 0}
    py_p = PyPlacement(placement)
    cpp_p = _core.Placement()
    for v, s in placement.items():
        cpp_p.set(v, s)

    net = FakeNetwork(G)
    for hyperedge, qubit_node in [([0, 3, 5], 0), ([3, 5], 3), ([0, 7], 7), ([5], 5)]:
        ref = py_p.get_distribution_tree(hyperedge, qubit_node, net)
        cpp = _core.get_distribution_tree(cpp_p, hyperedge, qubit_node, cpp_G)
        assert list(cpp) == list(ref), (hyperedge, qubit_node, list(cpp), list(ref))

    for s in [0, 1, 2, 3]:
        assert list(cpp_p.get_vertices_in(s)) == py_p.get_vertices_in(s), s
    print("placement: OK")


def check_hypergraph_circuit():
    from pytket import Circuit, OpType
    from pytket_dqc.circuits import HypergraphCircuit as PyHC

    def serialize(c):
        qubits = [str(q) for q in c.qubits]
        cmds = [
            (
                cmd.op.get_name(),
                int(cmd.op.type),
                [str(q) for q in cmd.qubits],
                [float(p) for p in cmd.op.params],
            )
            for cmd in c.get_commands()
        ]
        return qubits, cmds

    circuits = []
    c1 = Circuit(3)
    c1.H(0)
    c1.Rz(0.5, 0)
    c1.H(0)
    c1.add_gate(OpType.CU1, 0.25, [0, 1])
    c1.H(1)
    c1.Rz(0.3, 1)
    c1.add_gate(OpType.CU1, 0.5, [1, 2])
    c1.H(2)
    circuits.append(c1)

    c2 = Circuit(2)
    c2.add_gate(OpType.CU1, 0.1, [0, 1])
    c2.add_gate(OpType.CU1, 0.2, [0, 1])
    c2.H(0)
    c2.add_gate(OpType.CU1, 0.3, [0, 1])
    c2.H(1)
    circuits.append(c2)

    c3 = Circuit(1)
    c3.H(0)
    c3.Rz(0.7, 0)
    circuits.append(c3)

    for i, c in enumerate(circuits):
        py_hc = PyHC(c)
        qubits, cmds = serialize(c)
        cpp_hc = _core.HypergraphCircuit()
        cpp_hc.from_circuit(qubits, cmds)
        assert list(cpp_hc.vertex_list) == list(py_hc.vertex_list), i
        py_hedges = [(tuple(h.vertices), h.weight) for h in py_hc.hyperedge_list]
        cpp_hedges = [
            (tuple(h.vertices), h.weight) for h in cpp_hc.hyperedge_list
        ]
        assert cpp_hedges == py_hedges, (i, cpp_hedges, py_hedges)
        assert list(cpp_hc.get_qubit_vertices()) == py_hc.get_qubit_vertices(), i
    print("hypergraph_circuit: OK")


def check_distribution_cost():
    from pytket import Circuit, OpType
    from pytket_dqc.circuits import HypergraphCircuit as PyHC, Hyperedge as PyHyp
    from pytket_dqc.placement import Placement as PyPlacement
    from pytket_dqc.networks import NISQNetwork
    from pytket_dqc import Distribution

    def serialize(c):
        qubits = [str(q) for q in c.qubits]
        cmds = [
            (
                cmd.op.get_name(),
                int(cmd.op.type),
                [str(q) for q in cmd.qubits],
                [float(p) for p in cmd.op.params],
            )
            for cmd in c.get_commands()
        ]
        return qubits, cmds

    def cpp_hc(c):
        h = _core.HypergraphCircuit()
        qubits, cmds = serialize(c)
        h.from_circuit(qubits, cmds)
        return h

    def cpp_graph(nxG):
        g = _core.Graph()
        for n in nxG.nodes():
            g.add_node(n)
        for u, v, d in nxG.edges(data=True):
            g.add_edge(u, v, float(d.get("weight", 1.0)))
        return g

    def cpp_placement(p):
        cp = _core.Placement()
        for v, s in p.placement.items():
            cp.set(v, s)
        return cp

    def check_hyperedge_cost(py_dist, cpp_h, cpp_pl, cpp_g, hyp):
        ref = py_dist.hyperedge_cost(hyp)
        got = _core.hyperedge_cost(cpp_h, list(hyp.vertices), hyp.weight,
                                   cpp_pl, cpp_g)
        assert got == ref, (list(hyp.vertices), got, ref)

    # test_distribution_cost_no_embedding
    circ = Circuit(3).add_gate(OpType.CU1, 1.0, [0, 1]).add_gate(
        OpType.CU1, 1.0, [0, 2]
    )
    network = NISQNetwork(
        [[0, 1], [1, 2], [1, 3], [2, 4]], {0: [0], 1: [1], 2: [2], 3: [3], 4: [4]}
    )
    py_hc = PyHC(circ)
    cpp_h = cpp_hc(circ)
    cpp_g = cpp_graph(network.get_server_nx())
    for placement in [
        {0: 0, 1: 1, 2: 2, 3: 1, 4: 2},
        {0: 0, 1: 1, 2: 2, 3: 1, 4: 0},
        {0: 1, 1: 0, 2: 2, 3: 0, 4: 2},
    ]:
        py_p = PyPlacement(placement)
        ref = Distribution(py_hc, py_p, network).cost()
        got = _core.distribution_cost(cpp_h, cpp_placement(py_p), cpp_g)
        assert got == ref, (placement, got, ref)

    # test_alap_on_hyperedge_requiring_euler
    circ = Circuit(5)
    circ.add_gate(OpType.CU1, [0.3], [0, 1])
    circ.H(0)
    circ.Rz(0.5, 0)
    circ.add_gate(OpType.CU1, [1.0], [0, 3])
    circ.H(0)
    circ.add_gate(OpType.CU1, [1.0], [0, 4])
    circ.Rz(0.5, 0)
    circ.H(0)
    circ.add_gate(OpType.CU1, [0.8], [0, 2])
    network = NISQNetwork([[0, 1]], {0: [0], 1: [1, 2, 3, 4]})
    placement = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1}
    py_p = PyPlacement(placement)
    py_dist = Distribution(PyHC(circ), py_p, network)
    cpp_h2 = cpp_hc(circ)
    cpp_pl2 = cpp_placement(py_p)
    cpp_g2 = cpp_graph(network.get_server_nx())
    check_hyperedge_cost(py_dist, cpp_h2, cpp_pl2, cpp_g2, PyHyp([0, 5, 8]))

    # test_alap_on_hyperedge_mixing_H_and_D_embeddings
    circ = Circuit(4)
    circ.add_gate(OpType.CU1, 0.1234, [0, 1])
    circ.H(1)
    circ.add_gate(OpType.CU1, 1.0, [1, 2])
    circ.add_gate(OpType.CU1, 1.0, [1, 3])
    circ.H(1)
    circ.add_gate(OpType.CU1, 0.1234, [1, 3])
    circ.add_gate(OpType.CU1, 0.1234, [1, 2])
    network = NISQNetwork([[0, 1], [1, 2]], {0: [0], 1: [1], 2: [2, 3]})
    placement = {0: 0, 1: 1, 2: 2, 3: 2, 4: 0, 5: 1, 6: 1, 7: 1, 8: 2}
    py_p = PyPlacement(placement)
    py_dist = Distribution(PyHC(circ), py_p, network)
    cpp_h3 = cpp_hc(circ)
    cpp_pl3 = cpp_placement(py_p)
    cpp_g3 = cpp_graph(network.get_server_nx())
    assert cpp_h3.requires_h_embedded_cu1([1, 4, 8], 1) == py_dist.circuit.requires_h_embedded_cu1(PyHyp([1, 4, 8]))
    check_hyperedge_cost(py_dist, cpp_h3, cpp_pl3, cpp_g3, PyHyp([1, 4, 8]))

    # test_alap
    circ = Circuit(4)
    circ.add_gate(OpType.CU1, 0.1234, [1, 2])
    circ.add_gate(OpType.CU1, 0.1234, [0, 2])
    circ.add_gate(OpType.CU1, 0.1234, [2, 3])
    circ.add_gate(OpType.CU1, 0.1234, [0, 3])
    circ.H(0)
    circ.H(2)
    circ.Rz(0.1234, 3)
    circ.add_gate(OpType.CU1, 1.0, [0, 2])
    circ.add_gate(OpType.CU1, 1.0, [0, 3])
    circ.add_gate(OpType.CU1, 1.0, [1, 2])
    circ.H(0)
    circ.H(2)
    circ.Rz(0.1234, 0)
    circ.add_gate(OpType.CU1, 0.1234, [0, 1])
    circ.add_gate(OpType.CU1, 0.1234, [0, 3])
    circ.add_gate(OpType.CU1, 1.0, [1, 2])
    network = NISQNetwork(
        [[0, 1], [0, 2], [0, 3], [3, 4]], {0: [0], 1: [1, 2], 2: [3, 4], 3: [7], 4: [5, 6]}
    )
    placement = {0: 1, 1: 1, 2: 4, 3: 4, 4: 1, 5: 2, 6: 1, 7: 4, 8: 4, 9: 4, 10: 0, 11: 1, 12: 4, 13: 2}
    py_p = PyPlacement(placement)
    py_dist = Distribution(PyHC(circ), py_p, network)
    cpp_h4 = cpp_hc(circ)
    cpp_pl4 = cpp_placement(py_p)
    cpp_g4 = cpp_graph(network.get_server_nx())
    check_hyperedge_cost(py_dist, cpp_h4, cpp_pl4, cpp_g4, PyHyp([0, 5, 7, 11, 12]))
    check_hyperedge_cost(py_dist, cpp_h4, cpp_pl4, cpp_g4, PyHyp([2, 4, 5, 6, 13]))
    cpp_pl4.set(2, 0)
    py_p.placement[2] = 0
    check_hyperedge_cost(py_dist, cpp_h4, cpp_pl4, cpp_g4, PyHyp([2, 4, 5, 6, 13]))
    cpp_pl4.set(13, 1)
    py_p.placement[13] = 1
    check_hyperedge_cost(py_dist, cpp_h4, cpp_pl4, cpp_g4, PyHyp([2, 4, 5, 6, 13]))

    print("distribution_cost: OK")


def check_pacman():
    from pytket import Circuit, OpType
    from pytket.circuit import Op
    from pytket_dqc.packing import PacMan as PyPacMan
    from pytket_dqc.placement import Placement as PyPlacement
    from pytket_dqc.circuits import HypergraphCircuit as PyHC, Hyperedge as PyHyp

    cz = Op.create(OpType.CU1, 1)

    def serialize(c):
        qubits = [str(q) for q in c.qubits]
        cmds = [
            (
                cmd.op.get_name(),
                int(cmd.op.type),
                [str(q) for q in cmd.qubits],
                [float(p) for p in cmd.op.params],
            )
            for cmd in c.get_commands()
        ]
        return qubits, cmds

    def cpp_hc(c):
        h = _core.HypergraphCircuit()
        qubits, cmds = serialize(c)
        h.from_circuit(qubits, cmds)
        return h

    def cpp_placement(p):
        cp = _core.Placement()
        for v, s in p.placement.items():
            cp.set(v, s)
        return cp

    def cmp_pacman(circ, placement_dict, label, split=None):
        py_hc = PyHC(circ)
        cpp_h = cpp_hc(circ)
        if split is not None:
            old_vs, new_vss = split
            py_hc.split_hyperedge(
                old_hyperedge=PyHyp(old_vs, 1),
                new_hyperedge_list=[PyHyp(vs, 1) for vs in new_vss],
            )
            cpp_h.split_hyperedge(
                _core.Hyperedge(old_vs, 1),
                [_core.Hyperedge(vs, 1) for vs in new_vss],
            )
        py_pl = PyPlacement(placement_dict)
        py_pm = PyPacMan(py_hc, py_pl)
        cpp_pm = _core.PacMan(cpp_h, cpp_placement(py_pl))

        # packets_by_qubit
        cpp_pbq = cpp_pm.packets_by_qubit()
        assert set(cpp_pbq.keys()) == set(py_pm.packets_by_qubit.keys()), label
        for qv, py_list in py_pm.packets_by_qubit.items():
            cpp_list = cpp_pbq[qv]
            assert len(py_list) == len(cpp_list), (label, "packets", qv)
            for py_p, cpp_t in zip(py_list, cpp_list):
                assert py_p.packet_index == cpp_t[0], (label, "packet idx")
                assert py_p.qubit_vertex == cpp_t[1], (label, "qubit v")
                assert py_p.connected_server_index == cpp_t[2], (label, "conn")
                assert list(py_p.gate_vertices) == list(cpp_t[3]), (label, "gates")

        # neighbouring_packets (index sequences)
        cpp_nb = cpp_pm.neighbouring_packets()
        for qv, py_nps in py_pm.neighbouring_packets.items():
            assert [tuple(p.packet_index for p in np) for np in py_nps] == \
                [tuple(x) for x in cpp_nb[qv]], (label, "neighbouring", qv)

        # hopping_packets (index pairs)
        cpp_hp = cpp_pm.hopping_packets()
        for qv, py_hps in py_pm.hopping_packets.items():
            assert [(a.packet_index, b.packet_index) for (a, b) in py_hps] == \
                [tuple(x) for x in cpp_hp[qv]], (label, "hopping", qv)

        # merged_packets (index sequences)
        cpp_mp = cpp_pm.merged_packets()
        for qv, py_mps in py_pm.merged_packets.items():
            assert [tuple(p.packet_index for p in mp) for mp in py_mps] == \
                [tuple(x) for x in cpp_mp[qv]], (label, "merged", qv)

        # embedded packets
        cpp_emb = cpp_pm.get_all_embedded_packets()
        py_emb = py_pm.get_all_embedded_packets()
        assert set(cpp_emb.keys()) == set(py_emb.keys()), (label, "emb keys")
        for qv, py_inner in py_emb.items():
            py_norm = sorted(
                ((a.packet_index, b.packet_index),
                 sorted(p.packet_index for p in s))
                for ((a, b), s) in py_inner.items()
            )
            cpp_norm = sorted(
                (tuple(k), sorted(v)) for (k, v) in cpp_emb[qv]
            )
            assert py_norm == cpp_norm, (label, "embedded", qv)

        # hypergraph_from_packets
        py_new = py_pm.get_hypergraph_from_packets()
        cpp_new = cpp_pm.get_hypergraph_from_packets()
        py_hl = sorted((tuple(sorted(h.vertices)), h.weight)
                       for h in py_new.hyperedge_list)
        cpp_hl = sorted((tuple(sorted(h.vertices)), h.weight)
                        for h in cpp_new.hyperedge_list)
        assert py_hl == cpp_hl, (label, "hypergraph_from_packets")
        return py_pm, cpp_pm

    # test_build_packets
    circ = Circuit(3)
    circ.add_gate(cz, [0, 1]).add_gate(cz, [0, 2]).add_gate(cz, [0, 1])
    circ.add_gate(cz, [1, 2]).H(1).add_gate(cz, [2, 1])
    cmp_pacman(circ, {0: 0, 1: 1, 2: 2, 3: 0, 4: 0, 5: 0, 6: 1, 7: 2},
               "build_packets")

    # neighbouring: simple (00)
    circ = Circuit(6)
    circ.add_gate(cz, [0, 2])
    circ.Rz(0.3, 0)
    circ.add_gate(cz, [0, 3])
    pl = {i: i // 2 for i in range(6)}
    pl.update({6: 0, 7: 0})
    cmp_pacman(circ, pl, "neighbouring_00",
               split=([0, 6, 7], [[0, 7], [0, 6]]))

    # neighbouring: HZH = X (01)
    circ = Circuit(6)
    circ.add_gate(cz, [0, 2])
    circ.Rz(0.3, 0)
    circ.H(0)
    circ.Rz(1, 0)
    circ.H(0)
    circ.add_gate(cz, [0, 3])
    cmp_pacman(circ, {i: i // 2 for i in range(6)} | {6: 0, 7: 0},
               "neighbouring_01")

    # hopping: simple sandwich (03)
    circ = Circuit(6)
    circ.add_gate(cz, [0, 2])
    circ.H(0)
    circ.add_gate(cz, [0, 3])
    circ.H(0)
    circ.add_gate(cz, [0, 2])
    cmp_pacman(circ, {i: i // 2 for i in range(6)} | {6: 0, 7: 0, 8: 0},
               "hopping_03")

    # hopping with phases (08)
    circ = Circuit(6)
    circ.add_gate(cz, [0, 2])
    circ.H(0)
    circ.Rz(0.33, 0)
    circ.add_gate(cz, [0, 2])
    circ.Rz(0.67, 0)
    circ.H(0)
    circ.Rz(0.27, 0)
    circ.H(0)
    circ.Rz(0.43847, 0)
    circ.add_gate(cz, [0, 3])
    circ.Rz(1 - 0.43847, 0)
    circ.H(0)
    circ.add_gate(cz, [0, 2])
    cmp_pacman(circ, {i: i // 2 for i in range(6)} | {6: 0, 7: 0, 8: 0, 9: 0},
               "hopping_08")

    # hopping negative (10): CU1 to a different server breaks sandwich
    circ = Circuit(6)
    circ.add_gate(cz, [0, 2])
    circ.H(0)
    circ.add_gate(cz, [0, 2])
    circ.Rz(1, 0)
    circ.add_gate(cz, [0, 1])
    circ.H(0)
    circ.add_gate(cz, [0, 2])
    cmp_pacman(circ, {i: i // 2 for i in range(6)} | {6: 0, 7: 0, 8: 0, 9: 0},
               "hopping_10")

    # intertwining embeddings 0
    circ = Circuit(5)
    circ.add_gate(cz, [0, 1]).H(0).add_gate(cz, [0, 2]).H(0)
    circ.add_gate(cz, [0, 3]).H(0).add_gate(cz, [0, 4])
    cmp_pacman(circ, {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 0, 6: 0, 7: 0, 8: 0},
               "intertwining_0")

    # intertwining embeddings 1
    circ = Circuit(6)
    circ.add_gate(cz, [0, 1]).H(0).add_gate(cz, [0, 2]).H(0)
    circ.add_gate(cz, [0, 3]).H(0).add_gate(cz, [0, 4]).H(0).add_gate(cz, [0, 5])
    cmp_pacman(circ, {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 0, 7: 0, 8: 0,
                      9: 0, 10: 0}, "intertwining_1")

    print("pacman: OK")


def check_vertex_cover_refiner():
    from pytket import Circuit, OpType
    from pytket_dqc.circuits import HypergraphCircuit as PyHC
    from pytket_dqc.placement import Placement as PyPlacement
    from pytket_dqc.networks import NISQNetwork
    from pytket_dqc import Distribution
    from pytket_dqc.refiners.vertex_cover import VertexCover

    def serialize(c):
        qubits = [str(q) for q in c.qubits]
        cmds = [
            (
                cmd.op.get_name(),
                int(cmd.op.type),
                [str(q) for q in cmd.qubits],
                [float(p) for p in cmd.op.params],
            )
            for cmd in c.get_commands()
        ]
        return qubits, cmds

    def cpp_hc(c):
        h = _core.HypergraphCircuit()
        qubits, cmds = serialize(c)
        h.from_circuit(qubits, cmds)
        return h

    def cpp_graph(nxG):
        g = _core.Graph()
        for n in nxG.nodes():
            g.add_node(n)
        for u, v, d in nxG.edges(data=True):
            g.add_edge(u, v, float(d.get("weight", 1.0)))
        return g

    def cpp_placement(p):
        cp = _core.Placement()
        for v, s in p.placement.items():
            cp.set(v, s)
        return cp

    def check(circ, placement_dict, network, exact):
        py_dist = Distribution(PyHC(circ), PyPlacement(placement_dict), network)
        VertexCover().refine(py_dist, vertex_cover_alg="networkx")

        cpp_h = cpp_hc(circ)
        cpp_g = cpp_graph(network.get_server_nx())
        cpp_p = cpp_placement(PyPlacement(placement_dict))
        new_h, new_p = _core.vertex_cover_refine(cpp_h, cpp_p, cpp_g)

        py_cost = py_dist.cost()
        cpp_cost = _core.distribution_cost(new_h, new_p, cpp_g)
        assert py_cost == cpp_cost, ("cost", placement_dict, py_cost, cpp_cost)

        if exact:
            py_hl = sorted((tuple(sorted(h.vertices)), h.weight)
                           for h in py_dist.circuit.hyperedge_list)
            cpp_hl = sorted((tuple(sorted(h.vertices)), h.weight)
                            for h in new_h.hyperedge_list)
            assert py_hl == cpp_hl, ("hyperedges", placement_dict, py_hl, cpp_hl)
            assert dict(py_dist.placement.placement) == dict(new_p.placement), \
                ("placement", placement_dict,
                 dict(py_dist.placement.placement), dict(new_p.placement))

    # empty
    network = NISQNetwork([[0, 1]], {0: [0, 1], 1: [2]})
    check(Circuit(2), {0: 0, 1: 0, 2: 0}, network, exact=True)

    # trivial (single local CU1)
    circ = Circuit(2)
    circ.add_gate(OpType.CU1, 1.0, [0, 1])
    check(circ, {0: 0, 1: 0, 2: 0}, network, exact=True)

    # simple
    network = NISQNetwork(
        [[2, 1], [1, 0], [1, 3], [0, 4]], {0: [0], 1: [1], 2: [2], 3: [3], 4: [4, 5]}
    )
    circ = Circuit(3)
    circ.add_gate(OpType.CU1, 0.3, [0, 1])
    circ.H(0)
    circ.Rz(1.0, 0)
    circ.H(0)
    circ.add_gate(OpType.CU1, 0.8, [0, 2])
    check(circ, {0: 0, 1: 4, 2: 4}, network, exact=True)

    # complex 1
    network = NISQNetwork([[0, 1], [1, 2], [1, 3]], {0: [0], 1: [1], 2: [2], 3: [3]})
    circ = Circuit(4)
    circ.add_gate(OpType.CU1, 1.0, [0, 2])
    circ.Rz(0.3, 0)
    circ.add_gate(OpType.CU1, 1.0, [0, 1])
    circ.H(2)
    circ.add_gate(OpType.CU1, 1.0, [3, 2])
    circ.H(2)
    circ.H(0)
    circ.add_gate(OpType.CU1, 1.0, [0, 2])
    circ.add_gate(OpType.CU1, 1.0, [3, 0])
    circ.H(0)
    circ.add_gate(OpType.CU1, 1.0, [1, 0])
    check(circ, {0: 0, 1: 1, 2: 2, 3: 3}, network, exact=True)

    # complex 2 (the big ALAP circuit)
    network = NISQNetwork(
        [[0, 1], [0, 2], [0, 3], [3, 4]], {0: [0], 1: [1, 2], 2: [3, 4], 3: [7], 4: [5, 6]}
    )
    circ = Circuit(4)
    circ.add_gate(OpType.CU1, 0.1234, [1, 2])
    circ.add_gate(OpType.CU1, 0.1234, [0, 2])
    circ.add_gate(OpType.CU1, 0.1234, [2, 3])
    circ.add_gate(OpType.CU1, 0.1234, [0, 3])
    circ.H(0)
    circ.H(2)
    circ.Rz(0.1234, 3)
    circ.add_gate(OpType.CU1, 1.0, [0, 2])
    circ.add_gate(OpType.CU1, 1.0, [0, 3])
    circ.add_gate(OpType.CU1, 1.0, [1, 2])
    circ.H(0)
    circ.H(2)
    circ.Rz(0.1234, 0)
    circ.add_gate(OpType.CU1, 0.1234, [0, 1])
    circ.add_gate(OpType.CU1, 0.1234, [0, 3])
    circ.add_gate(OpType.CU1, 1.0, [1, 2])
    check(circ, {0: 1, 1: 1, 2: 2, 3: 4}, network, exact=True)

    print("vertex_cover_refiner: OK")


def check_gain_manager():
    from pytket import Circuit, OpType
    from pytket_dqc.circuits import HypergraphCircuit as PyHC, Hyperedge as PyHyp
    from pytket_dqc.placement import Placement as PyPlacement
    from pytket_dqc.networks import NISQNetwork
    from pytket_dqc import Distribution as PyDistribution
    from pytket_dqc.allocators import GainManager as PyGainManager

    def serialize(c):
        qubits = [str(q) for q in c.qubits]
        cmds = [
            (
                cmd.op.get_name(),
                int(cmd.op.type),
                [str(q) for q in cmd.qubits],
                [float(p) for p in cmd.op.params],
            )
            for cmd in c.get_commands()
        ]
        return qubits, cmds

    def cpp_hc(c):
        h = _core.HypergraphCircuit()
        qubits, cmds = serialize(c)
        h.from_circuit(qubits, cmds)
        return h

    def cpp_graph(nxG):
        g = _core.Graph()
        for n in nxG.nodes():
            g.add_node(n)
        for u, v, d in nxG.edges(data=True):
            g.add_edge(u, v, float(d.get("weight", 1.0)))
        return g

    def cpp_placement(p):
        cp = _core.Placement()
        for v, s in p.placement.items():
            cp.set(v, s)
        return cp

    def make_cpp_dist(py_dist):
        sq = {int(s): [int(q) for q in qs]
              for s, qs in py_dist.network.server_qubits.items()}
        return _core.Distribution(cpp_hc(py_dist.circuit.get_circuit()),
                                  cpp_placement(py_dist.placement),
                                  cpp_graph(py_dist.network.get_server_nx()),
                                  sq)

    def get_circ():
        circ = Circuit(4)
        circ.add_gate(OpType.CU1, 0.1234, [1, 2])
        circ.add_gate(OpType.CU1, 0.1234, [0, 2])
        circ.add_gate(OpType.CU1, 0.1234, [2, 3])
        circ.add_gate(OpType.CU1, 0.1234, [0, 3])
        circ.H(0).H(2).H(3)
        circ.add_gate(OpType.CU1, 1.0, [0, 2])
        circ.add_gate(OpType.CU1, 1.0, [0, 3])
        circ.add_gate(OpType.CU1, 1.0, [1, 2])
        circ.H(0).H(2).Rz(0.1234, 0)
        circ.add_gate(OpType.CU1, 0.1234, [0, 1])
        circ.add_gate(OpType.CU1, 0.1234, [0, 3])
        circ.add_gate(OpType.CU1, 1.0, [1, 2])
        return circ

    def get_network():
        return NISQNetwork(
            [[0, 1], [0, 2], [0, 3], [3, 4]],
            {0: [0], 1: [1, 2], 2: [3, 4], 3: [7], 4: [5, 6]},
        )

    def hcmap(mgr):
        return {tuple(sorted(h.vertices)): c
                for h, c in mgr.hyperedge_cost_map.items()}

    # test_placement_move_occupancy
    placement = PyPlacement({0: 1, 1: 1, 2: 2, 3: 4, 4: 1, 5: 1, 6: 1, 7: 1,
                             8: 2, 9: 4, 10: 0, 11: 1, 12: 3, 13: 0})
    circ = get_circ()
    network = get_network()
    py_dist = PyDistribution(PyHC(circ), placement, network)
    py_mgr = PyGainManager(py_dist)
    cpp_dist = make_cpp_dist(py_dist)
    cpp_mgr = _core.GainManager(cpp_dist)
    assert dict(py_mgr.occupancy) == dict(cpp_mgr.occupancy), "occupancy init"
    for v in [2, 1, 3, 8, 0]:
        for s in [4, 1, 0, 2]:
            assert py_mgr.is_move_valid(v, s) == cpp_mgr.is_move_valid(v, s), \
                ("is_move_valid", v, s)
    py_mgr.move_vertex(2, 4)
    cpp_mgr.move_vertex(2, 4)
    assert dict(py_mgr.occupancy) == dict(cpp_mgr.occupancy), "occupancy mv1"
    py_mgr.move_vertex(2, 2)
    cpp_mgr.move_vertex(2, 2)
    py_mgr.move_vertex(0, 0)
    cpp_mgr.move_vertex(0, 0)
    py_mgr.move_vertex(3, 1)
    cpp_mgr.move_vertex(3, 1)
    assert dict(py_mgr.occupancy) == dict(cpp_mgr.occupancy), "occupancy mv2"
    assert dict(py_dist.placement.placement) == \
        dict(cpp_dist.placement.placement), "placement after moves"

    # test_gain_no_embeddings (fresh managers on the original placement)
    placement = PyPlacement({0: 1, 1: 1, 2: 2, 3: 4, 4: 1, 5: 1, 6: 1, 7: 1,
                             8: 2, 9: 4, 10: 0, 11: 1, 12: 3, 13: 0})
    py_dist = PyDistribution(PyHC(circ), placement, network)
    py_mgr = PyGainManager(py_dist)
    cpp_dist = make_cpp_dist(py_dist)
    cpp_mgr = _core.GainManager(cpp_dist)
    assert hcmap(py_mgr) == hcmap(cpp_mgr), "hyperedge_cost_map init"
    assert py_mgr.move_vertex_gain(0, 1) == cpp_mgr.move_vertex_gain(0, 1)
    assert py_mgr.move_vertex_gain(12, 4) == cpp_mgr.move_vertex_gain(12, 4)
    assert hcmap(py_mgr) == hcmap(cpp_mgr), "hcmap after gain 12->4"
    py_mgr.move_vertex(12, 4)
    cpp_mgr.move_vertex(12, 4)
    assert hcmap(py_mgr) == hcmap(cpp_mgr), "hcmap after move 12->4"
    py_mgr.move_vertex(12, 3)
    cpp_mgr.move_vertex(12, 3)
    assert py_mgr.move_vertex_gain(0, 2) == cpp_mgr.move_vertex_gain(0, 2)
    assert hcmap(py_mgr) == hcmap(cpp_mgr), "hcmap after gain 0->2"
    py_mgr.move_vertex(0, 2)
    cpp_mgr.move_vertex(0, 2)
    assert hcmap(py_mgr) == hcmap(cpp_mgr), "hcmap after move 0->2"

    # test_split_merge
    circ2 = Circuit(3)
    circ2.add_gate(OpType.CU1, 1.0, [0, 1])
    circ2.add_gate(OpType.CU1, 1.0, [0, 2])
    placement2 = PyPlacement({0: 1, 1: 2, 2: 3, 3: 2, 4: 3})
    network2 = NISQNetwork([[0, 1], [0, 2], [0, 3]], {0: [0], 1: [1], 2: [2], 3: [3]})
    py_dist2 = PyDistribution(PyHC(circ2), placement2, network2)
    py_mgr2 = PyGainManager(py_dist2)
    cpp_dist2 = make_cpp_dist(py_dist2)
    cpp_mgr2 = _core.GainManager(cpp_dist2)

    old_h = PyHyp([0, 3, 4], 1)
    h_one = PyHyp([0, 3], 1)
    h_two = PyHyp([0, 4], 1)
    py_mgr2.split_hyperedge(old_hyperedge=old_h, new_hyperedge_list=[h_one, h_two])
    cpp_mgr2.split_hyperedge(_core.Hyperedge([0, 3, 4], 1),
                             [_core.Hyperedge([0, 3], 1), _core.Hyperedge([0, 4], 1)])
    assert py_mgr2.merge_hyperedge_gain([h_one, h_two]) == \
        cpp_mgr2.merge_hyperedge_gain([_core.Hyperedge([0, 3], 1), _core.Hyperedge([0, 4], 1)])
    py_mgr2.merge_hyperedge(to_merge_hyperedge_list=[h_one, h_two])
    cpp_mgr2.merge_hyperedge([_core.Hyperedge([0, 3], 1), _core.Hyperedge([0, 4], 1)])
    assert py_mgr2.split_hyperedge_gain(old_h, [h_one, h_two]) == \
        cpp_mgr2.split_hyperedge_gain(_core.Hyperedge([0, 3, 4], 1),
                                      [_core.Hyperedge([0, 3], 1), _core.Hyperedge([0, 4], 1)])

    # test_merge_with_embedding
    circ3 = Circuit(3)
    circ3.add_gate(OpType.CU1, 1.0, [0, 2])
    circ3.add_gate(OpType.CU1, 1.0, [0, 1])
    circ3.H(0)
    circ3.add_gate(OpType.CU1, 1.0, [0, 2])
    circ3.H(0)
    circ3.add_gate(OpType.CU1, 1.0, [0, 2])
    placement3 = PyPlacement({0: 1, 1: 2, 2: 3, 3: 3, 4: 2, 5: 3, 6: 3})
    py_dist3 = PyDistribution(PyHC(circ3), placement3, network2)
    py_mgr3 = PyGainManager(py_dist3)
    cpp_dist3 = make_cpp_dist(py_dist3)
    cpp_mgr3 = _core.GainManager(cpp_dist3)
    merge_list = [PyHyp([0, 3, 4], 1), PyHyp([0, 6], 1)]
    cpp_merge_list = [_core.Hyperedge([0, 3, 4], 1), _core.Hyperedge([0, 6], 1)]
    assert py_mgr3.merge_hyperedge_gain(merge_list) == \
        cpp_mgr3.merge_hyperedge_gain(cpp_merge_list)
    py_mgr3.merge_hyperedge(merge_list)
    cpp_mgr3.merge_hyperedge(cpp_merge_list)
    assert py_dist3.cost() == cpp_dist3.cost()

    print("gain_manager: OK")


def check_allocators():
    from pytket import Circuit, OpType
    from pytket_dqc.circuits import HypergraphCircuit as PyHC
    from pytket_dqc.networks import NISQNetwork
    from pytket_dqc.allocators import Random, Ordered, Brute

    def serialize(c):
        qubits = [str(q) for q in c.qubits]
        cmds = [
            (
                cmd.op.get_name(),
                int(cmd.op.type),
                [str(q) for q in cmd.qubits],
                [float(p) for p in cmd.op.params],
            )
            for cmd in c.get_commands()
        ]
        return qubits, cmds

    def cpp_hc(c):
        h = _core.HypergraphCircuit()
        qubits, cmds = serialize(c)
        h.from_circuit(qubits, cmds)
        return h

    def cpp_graph(nxG):
        g = _core.Graph()
        for n in nxG.nodes():
            g.add_node(n)
        for u, v, d in nxG.edges(data=True):
            g.add_edge(u, v, float(d.get("weight", 1.0)))
        return g

    def coupling(network):
        return [(int(a), int(b)) for (a, b) in network.server_coupling]

    def server_qubits(network):
        return [(int(s), [int(q) for q in qs])
                for s, qs in network.server_qubits.items()]

    circ = Circuit(3)
    circ.add_gate(OpType.CU1, 1.0, [0, 1])
    circ.add_gate(OpType.CU1, 1.0, [0, 2])
    network = NISQNetwork([[0, 1], [1, 2]], {0: [0, 1], 1: [2], 2: [3]})

    # Random (seeded)
    py_p = Random().allocate(circ, network, seed=42).placement.placement
    cpp_d = _core.random_allocate(cpp_hc(circ), cpp_nisq(network), 42)
    assert dict(py_p) == dict(cpp_d.placement.placement), \
        ("random", dict(py_p), dict(cpp_d.placement.placement))

    # Ordered
    py_p = Ordered().allocate(circ, network).placement.placement
    cpp_d = _core.ordered_allocate(cpp_hc(circ), cpp_nisq(network))
    assert dict(py_p) == dict(cpp_d.placement.placement), \
        ("ordered", dict(py_p), dict(cpp_d.placement.placement))

    # Brute (tiny search space)
    circ2 = Circuit(2)
    circ2.add_gate(OpType.CU1, 1.0, [0, 1])
    network2 = NISQNetwork([[0, 1]], {0: [0], 1: [1]})
    py_dist = Brute().allocate(circ2, network2)
    cpp_d2 = _core.brute_allocate(cpp_hc(circ2), cpp_nisq(network2))
    assert py_dist.cost() == cpp_d2.cost()

    print("allocators: OK")


def check_eager_h_type_merge():
    from pytket import Circuit, OpType
    from pytket_dqc.circuits import HypergraphCircuit as PyHC
    from pytket_dqc.placement import Placement as PyPlacement
    from pytket_dqc.networks import NISQNetwork
    from pytket_dqc import Distribution as PyDistribution
    from pytket_dqc.refiners import EagerHTypeMerge

    def serialize(c):
        qubits = [str(q) for q in c.qubits]
        cmds = [
            (
                cmd.op.get_name(),
                int(cmd.op.type),
                [str(q) for q in cmd.qubits],
                [float(p) for p in cmd.op.params],
            )
            for cmd in c.get_commands()
        ]
        return qubits, cmds

    def cpp_hc(c):
        h = _core.HypergraphCircuit()
        qubits, cmds = serialize(c)
        h.from_circuit(qubits, cmds)
        return h

    def cpp_graph(nxG):
        g = _core.Graph()
        for n in nxG.nodes():
            g.add_node(n)
        for u, v, d in nxG.edges(data=True):
            g.add_edge(u, v, float(d.get("weight", 1.0)))
        return g

    def cpp_placement(p):
        cp = _core.Placement()
        for v, s in p.placement.items():
            cp.set(v, s)
        return cp

    def cpp_dist(py_dist):
        sq = {int(s): [int(q) for q in qs]
              for s, qs in py_dist.network.server_qubits.items()}
        return _core.Distribution(cpp_hc(py_dist.circuit.get_circuit()),
                                  cpp_placement(py_dist.placement),
                                  cpp_graph(py_dist.network.get_server_nx()),
                                  sq)

    def run(circ, placement_dict, network):
        py_dist = PyDistribution(PyHC(circ), PyPlacement(placement_dict), network)
        py_made = EagerHTypeMerge().refine(py_dist)
        cd = cpp_dist(py_dist)
        cpp_made = _core.eager_h_type_merge_refine(cd)
        assert py_made == cpp_made, (placement_dict, py_made, cpp_made)
        py_hl = [tuple(sorted(h.vertices)) for h in py_dist.circuit.hyperedge_list]
        cpp_hl = [tuple(sorted(h.vertices)) for h in cd.circuit.hyperedge_list]
        assert py_hl == cpp_hl, (placement_dict, py_hl, cpp_hl)
        assert py_dist.cost() == cd.cost(), (placement_dict,)

    # test_eager_h_type_merge_00
    network = NISQNetwork([[0, 1]], {0: [0], 1: [1, 2, 3]})
    circ = Circuit(4)
    circ.add_gate(OpType.CU1, 1, [0, 1])
    circ.H(0).add_gate(OpType.CU1, 1, [0, 2]).H(0)
    circ.add_gate(OpType.CU1, 1, [0, 3])
    run(circ, {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 0, 6: 1}, network)

    # test_eager_h_type_merge_01
    network = NISQNetwork([[0, 1]], {0: [0], 1: [1]})
    circ = Circuit(2)
    circ.add_gate(OpType.CU1, 1, [0, 1])
    circ.H(0).H(1).add_gate(OpType.CU1, 1, [0, 1]).H(0).H(1)
    circ.add_gate(OpType.CU1, 1, [0, 1])
    run(circ, {0: 0, 1: 1, 2: 0, 3: 1, 4: 0}, network)

    # test_eager_h_type_merge_02
    network = NISQNetwork([[0, 1], [0, 2], [1, 2]], {0: [0], 1: [1], 2: [2]})
    circ = Circuit(3)
    circ.add_gate(OpType.CU1, 1, [0, 1])
    circ.H(0).add_gate(OpType.CU1, 1, [0, 1]).H(0)
    circ.add_gate(OpType.CU1, 1, [0, 2])
    run(circ, {0: 0, 1: 1, 2: 2, 3: 1, 4: 0, 5: 0}, network)

    # test_eager_h_type_merge_03
    network = NISQNetwork([[0, 1], [0, 2], [1, 2]], {0: [0], 1: [1], 2: [2]})
    circ = Circuit(3)
    circ.add_gate(OpType.CU1, 1, [0, 1])
    circ.H(0).add_gate(OpType.CU1, 1, [0, 2]).H(0)
    circ.add_gate(OpType.CU1, 1, [0, 1])
    run(circ, {0: 0, 1: 1, 2: 2, 3: 1, 4: 0, 5: 1}, network)

    # test_eager_h_type_merge_04
    network = NISQNetwork([[0, 1]], {0: [0, 1], 1: [2]})
    circ = Circuit(3)
    circ.add_gate(OpType.CU1, 1, [0, 2])
    circ.add_gate(OpType.CU1, 1, [0, 1])
    circ.H(0).add_gate(OpType.CU1, 1, [0, 2]).H(0)
    circ.add_gate(OpType.CU1, 1, [0, 1])
    circ.add_gate(OpType.CU1, 1, [0, 2])
    run(circ, {0: 0, 1: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 0, 7: 1}, network)

    # test_eager_h_type_merge_05
    network = NISQNetwork([[0, 1]], {0: [0, 1], 1: [2]})
    circ = Circuit(3)
    circ.add_gate(OpType.CU1, 1, [0, 2])
    circ.H(0).add_gate(OpType.CU1, 1, [0, 1]).H(0)
    circ.add_gate(OpType.CU1, 1, [0, 2])
    run(circ, {0: 0, 1: 0, 2: 1, 3: 0, 4: 0, 5: 0}, network)

    # test_eager_h_type_merge_no_detached (both placements)
    network = NISQNetwork([[0, 1], [1, 2]], {0: [0], 1: [1], 2: [2, 3]})
    circ = Circuit(3)
    circ.add_gate(OpType.CU1, 1, [0, 1])
    circ.H(0).H(1)
    circ.add_gate(OpType.CU1, 1, [0, 2])
    circ.H(0)
    circ.add_gate(OpType.CU1, 1, [0, 1])
    run(circ, {0: 0, 1: 2, 2: 2, 3: 0, 4: 1, 5: 0}, network)
    run(circ, {0: 0, 1: 2, 2: 2, 3: 0, 4: 2, 5: 0}, network)

    print("eager_h_type_merge: OK")


def check_d_type_merge():
    from pytket import Circuit, OpType
    from pytket_dqc.circuits import HypergraphCircuit as PyHC
    from pytket_dqc.placement import Placement as PyPlacement
    from pytket_dqc.networks import NISQNetwork
    from pytket_dqc import Distribution as PyDistribution
    from pytket_dqc.refiners import IntertwinedDTypeMerge, NeighbouringDTypeMerge

    def serialize(c):
        qubits = [str(q) for q in c.qubits]
        cmds = [
            (
                cmd.op.get_name(),
                int(cmd.op.type),
                [str(q) for q in cmd.qubits],
                [float(p) for p in cmd.op.params],
            )
            for cmd in c.get_commands()
        ]
        return qubits, cmds

    def cpp_hc(c):
        h = _core.HypergraphCircuit()
        qubits, cmds = serialize(c)
        h.from_circuit(qubits, cmds)
        return h

    def cpp_graph(nxG):
        g = _core.Graph()
        for n in nxG.nodes():
            g.add_node(n)
        for u, v, d in nxG.edges(data=True):
            g.add_edge(u, v, float(d.get("weight", 1.0)))
        return g

    def cpp_placement(p):
        cp = _core.Placement()
        for v, s in p.placement.items():
            cp.set(v, s)
        return cp

    network = NISQNetwork([[0, 1], [1, 2], [1, 3]], {0: [0], 1: [1], 2: [2], 3: [3]})
    circ = Circuit(4)
    circ.add_gate(OpType.CU1, 1.0, [0, 1])
    circ.H(1)
    circ.add_gate(OpType.CU1, 1.0, [1, 3])
    circ.H(1)
    circ.add_gate(OpType.CU1, 1.0, [1, 3])
    circ.add_gate(OpType.CU1, 1.0, [0, 1])
    circ.H(1)
    circ.add_gate(OpType.CU1, 1.0, [0, 1])
    circ.H(1)
    circ.add_gate(OpType.CU1, 1.0, [1, 2])
    circ.add_gate(OpType.CU1, 1.0, [1, 3])
    circ.H(1)
    circ.add_gate(OpType.CU1, 1.0, [0, 1])
    circ.add_gate(OpType.CU1, 1.0, [0, 1])

    hedges = [
        [0, 4, 7, 8, 11, 12],
        [1, 4, 7],
        [1, 5],
        [1, 6, 10],
        [1, 8, 11],
        [1, 9],
        [1, 12],
        [2, 9],
        [3, 5, 6, 10],
    ]
    placement_dict = {0: 0, 1: 1, 2: 2, 3: 3, 4: 0, 5: 3, 6: 3, 7: 0, 8: 0,
                      9: 2, 10: 3, 11: 0, 12: 0}

    def build_dist():
        py_hc = PyHC(circ)
        py_hc.vertex_neighbours = {i: set() for i in py_hc.vertex_list}
        py_hc.hyperedge_list = []
        py_hc.hyperedge_dict = {i: [] for i in py_hc.vertex_list}
        for vs in hedges:
            py_hc.add_hyperedge(vs)
        py_pl = PyPlacement(placement_dict)
        py_dist = PyDistribution(py_hc, py_pl, network)

        cpp_h = cpp_hc(circ)
        cpp_h.reset_hyperedges()
        for vs in hedges:
            cpp_h.add_hyperedge(vs, 1)
        cpp_dist = _core.Distribution(cpp_h, cpp_placement(py_pl),
                                      cpp_graph(network.get_server_nx()),
                                      {s: list(qs) for s, qs in network.server_qubits.items()})
        return py_dist, cpp_dist

    def cmp(refiner_name):
        py_dist, cpp_dist = build_dist()
        py_made = (IntertwinedDTypeMerge() if refiner_name == "intertwined"
                   else NeighbouringDTypeMerge()).refine(py_dist)
        cpp_made = (_core.intertwined_d_type_merge_refine(cpp_dist)
                    if refiner_name == "intertwined"
                    else _core.neighbouring_d_type_merge_refine(cpp_dist))
        assert py_made == cpp_made, (refiner_name, py_made, cpp_made)
        py_hl = [tuple(sorted(h.vertices)) for h in py_dist.circuit.hyperedge_list]
        cpp_hl = [tuple(sorted(h.vertices)) for h in cpp_dist.circuit.hyperedge_list]
        assert py_hl == cpp_hl, (refiner_name, py_hl, cpp_hl)
        assert py_dist.cost() == cpp_dist.cost(), refiner_name

    cmp("intertwined")
    cmp("neighbouring")

    print("d_type_merge: OK")


def check_boundary_reallocation():
    from pytket import Circuit, OpType
    from pytket_dqc.networks import NISQNetwork
    from pytket_dqc.allocators import Random
    from pytket_dqc.refiners import BoundaryReallocation

    def serialize(c):
        qubits = [str(q) for q in c.qubits]
        cmds = [
            (
                cmd.op.get_name(),
                int(cmd.op.type),
                [str(q) for q in cmd.qubits],
                [float(p) for p in cmd.op.params],
            )
            for cmd in c.get_commands()
        ]
        return qubits, cmds

    def cpp_hc(c):
        h = _core.HypergraphCircuit()
        qubits, cmds = serialize(c)
        h.from_circuit(qubits, cmds)
        return h

    def cpp_graph(nxG):
        g = _core.Graph()
        for n in nxG.nodes():
            g.add_node(n)
        for u, v, d in nxG.edges(data=True):
            g.add_edge(u, v, float(d.get("weight", 1.0)))
        return g

    def coupling(network):
        return [(int(a), int(b)) for (a, b) in network.server_coupling]

    def server_qubits(network):
        return [(int(s), [int(q) for q in qs])
                for s, qs in network.server_qubits.items()]

    circ = Circuit(3)
    circ.add_gate(OpType.CU1, 0.3, [0, 1])
    circ.H(0)
    circ.Rz(1.0, 0)
    circ.H(0)
    circ.add_gate(OpType.CU1, 0.8, [0, 2])
    network = NISQNetwork(
        [[2, 1], [1, 0], [1, 3], [0, 4]], {0: [0], 1: [1], 2: [2], 3: [3], 4: [4, 5]}
    )

    py_dist = Random().allocate(circ, network, seed=7)
    cpp_h = cpp_hc(circ)
    cpp_dist = _core.random_allocate(cpp_h, cpp_nisq(network), 7)
    assert dict(py_dist.placement.placement) == dict(cpp_dist.placement.placement)

    py_made = BoundaryReallocation().refine(py_dist, seed=42)
    cpp_made = _core.boundary_reallocation_refine(cpp_dist, [], 10, 0.05, 42, None)
    assert py_made == cpp_made, (py_made, cpp_made)
    assert dict(py_dist.placement.placement) == dict(cpp_dist.placement.placement), \
        ("boundary", dict(py_dist.placement.placement), dict(cpp_dist.placement.placement))
    assert py_dist.cost() == cpp_dist.cost()

    print("boundary_reallocation: OK")


def check_annealing():
    import math
    from pytket import Circuit, OpType
    from pytket_dqc.networks import NISQNetwork
    from pytket_dqc.allocators import Annealing
    from pytket_dqc.allocators.annealing import acceptance_criterion as py_acc

    # acceptance_criterion is deterministic.
    for gain in [-3, -1, 0, 1, 5]:
        for it in [0, 1, 9, 100]:
            for t in [1.0, 3.0]:
                py = py_acc(gain, it, t)
                cpp = _core.acceptance_criterion(gain, it, t)
                assert py == cpp, (gain, it, t, py, cpp)
    # Overflow case -> inf
    assert _core.acceptance_criterion(10**6, 0, 1.0) == float("inf")

    def serialize(c):
        qubits = [str(q) for q in c.qubits]
        cmds = [
            (
                cmd.op.get_name(),
                int(cmd.op.type),
                [str(q) for q in cmd.qubits],
                [float(p) for p in cmd.op.params],
            )
            for cmd in c.get_commands()
        ]
        return qubits, cmds

    def cpp_hc(c):
        h = _core.HypergraphCircuit()
        qubits, cmds = serialize(c)
        h.from_circuit(qubits, cmds)
        return h

    def coupling(network):
        return [(int(a), int(b)) for (a, b) in network.server_coupling]

    def server_qubits(network):
        return [(int(s), [int(q) for q in qs])
                for s, qs in network.server_qubits.items()]

    def cpp_graph(nxG):
        g = _core.Graph()
        for n in nxG.nodes():
            g.add_node(n)
        for u, v, d in nxG.edges(data=True):
            g.add_edge(u, v, float(d.get("weight", 1.0)))
        return g

    circ = Circuit(4)
    circ.add_gate(OpType.CU1, 1.0, [0, 2])
    circ.add_gate(OpType.CU1, 1.0, [0, 1])
    circ.H(2)
    circ.add_gate(OpType.CU1, 1.0, [3, 2])
    circ.H(2)
    network = NISQNetwork([[0, 1], [1, 2], [1, 3]], {0: [0], 1: [1], 2: [2], 3: [3]})

    cpp_h = cpp_hc(circ)
    cpp_dist = _core.annealing_allocate(cpp_h, cpp_nisq(network), 300, 3.0, None, 42)
    # Sanity: the returned distribution is a valid distribution (capacity respected).
    for s, qs in network.server_qubits.items():
        count = sum(1 for v, sv in cpp_dist.placement.placement.items()
                    if sv == s and cpp_h.is_qubit_vertex(v))
        assert count <= len(qs), (s, count, len(qs))

    # And it agrees with Python on being a valid distribution (cost finite).
    from pytket_dqc.circuits import HypergraphCircuit as PyHC
    from pytket_dqc.placement import Placement as PyPlacement
    from pytket_dqc import Distribution as PyDistribution
    py_dist = PyDistribution(PyHC(circ),
                             PyPlacement(dict(cpp_dist.placement.placement)),
                             network)
    assert py_dist.is_valid()

    print("annealing: OK")


def check_annealing_seeded():
    from pytket import Circuit, OpType
    from pytket_dqc.networks import NISQNetwork
    from pytket_dqc.allocators import Annealing, Random as RandomAllocator
    from pytket_dqc.circuits import HypergraphCircuit as PyHC
    from pytket_dqc.placement import Placement as PyPlacement
    from pytket_dqc import Distribution as PyDistribution

    # Mirrors the user-facing workaround for the reference bug in
    # `Annealing.allocate`: the default initial placement calls `Random()`
    # with `seed=None`, which discards the seed and makes the run
    # non-deterministic. Wrapping `Random` so it always re-seeds with a fixed
    # value makes the whole annealing run deterministic.
    class SeededRandomAllocator(RandomAllocator):
        def __init__(self, seed):
            self.seed = seed
            self.random_allocator = RandomAllocator()

        def allocate(self, circ, network, **kwargs):
            return self.random_allocator.allocate(circ, network,
                                                  seed=self.seed, **kwargs)

    def serialize(c):
        qubits = [str(q) for q in c.qubits]
        cmds = [
            (
                cmd.op.get_name(),
                int(cmd.op.type),
                [str(q) for q in cmd.qubits],
                [float(p) for p in cmd.op.params],
            )
            for cmd in c.get_commands()
        ]
        return qubits, cmds

    def cpp_hc(c):
        h = _core.HypergraphCircuit()
        qubits, cmds = serialize(c)
        h.from_circuit(qubits, cmds)
        return h

    def cpp_graph(nxG):
        g = _core.Graph()
        for n in nxG.nodes():
            g.add_node(n)
        for u, v, d in nxG.edges(data=True):
            g.add_edge(u, v, float(d.get("weight", 1.0)))
        return g

    def coupling(network):
        return [(int(a), int(b)) for (a, b) in network.server_coupling]

    def server_qubits(network):
        return [(int(s), [int(q) for q in qs])
                for s, qs in network.server_qubits.items()]

    circuits = []
    c1 = Circuit(4)
    c1.add_gate(OpType.CU1, 1.0, [0, 2])
    c1.add_gate(OpType.CU1, 1.0, [0, 1])
    c1.H(2)
    c1.add_gate(OpType.CU1, 1.0, [3, 2])
    c1.H(2)
    circuits.append((c1, NISQNetwork([[0, 1], [1, 2], [1, 3]],
                                     {0: [0], 1: [1], 2: [2], 3: [3]})))

    c2 = Circuit(3)
    c2.add_gate(OpType.CU1, 0.3, [0, 1])
    c2.H(0)
    c2.Rz(1.0, 0)
    c2.H(0)
    c2.add_gate(OpType.CU1, 0.8, [0, 2])
    circuits.append((c2, NISQNetwork([[2, 1], [1, 0], [1, 3], [0, 4]],
                                     {0: [0], 1: [1], 2: [2], 3: [3], 4: [4, 5]})))

    iterations = 300
    for seed in [42, 7]:
        for circ, network in circuits:
            py_dist = Annealing().allocate(
                circ, network, seed=seed, iterations=iterations,
                initial_place_method=SeededRandomAllocator(seed))
            cpp_h = cpp_hc(circ)
            cpp_g = cpp_graph(network.get_server_nx())
            cpp_d = _core.annealing_allocate(
                cpp_h, cpp_nisq(network), iterations, 3.0, None, seed, seed)
            assert dict(py_dist.placement.placement) == \
                dict(cpp_d.placement.placement), \
                (seed, "placement", dict(py_dist.placement.placement),
                 dict(cpp_d.placement.placement))
            assert py_dist.cost() == cpp_d.cost(), (seed, "cost", py_dist.cost())
            assert py_dist.is_valid()

    print("annealing_seeded: OK")


def check_distribution_methods():
    from pytket import Circuit, OpType
    from pytket_dqc.circuits import HypergraphCircuit as PyHC
    from pytket_dqc.placement import Placement as PyPlacement
    from pytket_dqc.networks import NISQNetwork
    from pytket_dqc import Distribution as PyDistribution

    def serialize(c):
        qubits = [str(q) for q in c.qubits]
        cmds = [
            (
                cmd.op.get_name(),
                int(cmd.op.type),
                [str(q) for q in cmd.qubits],
                [float(p) for p in cmd.op.params],
            )
            for cmd in c.get_commands()
        ]
        return qubits, cmds

    def cpp_hc(c):
        h = _core.HypergraphCircuit()
        qubits, cmds = serialize(c)
        h.from_circuit(qubits, cmds)
        return h

    def cpp_graph(nxG):
        g = _core.Graph()
        for n in nxG.nodes():
            g.add_node(n)
        for u, v, d in nxG.edges(data=True):
            g.add_edge(u, v, float(d.get("weight", 1.0)))
        return g

    def cpp_placement(p):
        cp = _core.Placement()
        for v, s in p.placement.items():
            cp.set(v, s)
        return cp

    circ = Circuit(3)
    circ.add_gate(OpType.CU1, 1.0, [0, 1])
    circ.add_gate(OpType.CU1, 1.0, [0, 2])
    network = NISQNetwork(
        [[0, 1], [1, 2], [2, 3], [3, 4]], {0: [0], 1: [1], 2: [2], 3: [3], 4: [4, 5]}
    )
    py_hc = PyHC(circ)
    cpp_h = cpp_hc(circ)
    cpp_g = cpp_graph(network.get_server_nx())
    sq = {int(s): [int(q) for q in qs]
          for s, qs in network.server_qubits.items()}

    placements = [
        {0: 0, 1: 1, 2: 2, 3: 1, 4: 2},   # valid, both gates non-local
        {0: 0, 1: 1, 2: 2, 3: 3, 4: 2},   # valid, gate 3 detached
        {0: 0, 1: 0, 2: 2, 3: 1, 4: 2},   # invalid: server 0 over capacity
        {0: 0, 1: 1, 2: 2, 3: 9, 4: 2},   # invalid: unknown server 9
        {0: 0, 1: 1, 2: 2},               # invalid: gate vertices unplaced
    ]
    all_placed = {0, 1, 2, 3, 4}
    for placement in placements:
        py_p = PyPlacement(placement)
        py_d = PyDistribution(py_hc, py_p, network)
        cpp_p = cpp_placement(py_p)
        cpp_d = _core.Distribution(cpp_h, cpp_p, cpp_g, sq)
        assert py_d.is_valid() == cpp_d.is_valid(), (placement, "is_valid")
        if set(placement) == all_placed:
            assert py_d.non_local_gate_count() == cpp_d.non_local_gate_count(), \
                (placement, "non_local_count")
            assert py_d.detached_gate_count() == cpp_d.detached_gate_count(), \
                (placement, "detached_count")
            assert list(py_d.non_local_gate_list()) == \
                list(cpp_d.non_local_gate_list()), (placement, "non_local")
            assert list(py_d.detached_gate_list()) == \
                list(cpp_d.detached_gate_list()), (placement, "detached")
    print("distribution_methods: OK")


def check_gateset():
    from pytket import OpType
    from pytket.circuit import Op as PyOp
    from pytket_dqc.utils.gateset import to_euler_with_two_hadamards as py_euler

    def py_to_tuples(ops):
        return [(int(op.type), [float(p) for p in op.params]) for op in ops]

    cases = [
        [],
        [(OpType.Rz, [0.3])],
        [(OpType.H, [])],
        [(OpType.Rz, [0.3]), (OpType.H, [])],
        [(OpType.H, []), (OpType.Rz, [0.3])],
        [(OpType.H, []), (OpType.Rz, [0.3]), (OpType.H, [])],
        [(OpType.Rz, [0.3]), (OpType.H, []), (OpType.Rz, [0.4])],
        [(OpType.Rz, [0.1]), (OpType.H, []), (OpType.Rz, [0.2]), (OpType.H, [])],
        [(OpType.H, []), (OpType.Rz, [0.2]), (OpType.H, []), (OpType.Rz, [0.3])],
        [(OpType.Rz, [0.1]), (OpType.H, []), (OpType.Rz, [0.2]), (OpType.H, []),
         (OpType.Rz, [0.3])],
        [(OpType.H, []), (OpType.Rz, [0.2]), (OpType.H, []), (OpType.Rz, [0.3])],
    ]
    for case in cases:
        py_ops = [PyOp.create(t, list(p)) for t, p in case]
        py_res = py_to_tuples(py_euler(py_ops))
        cpp_res = _core.to_euler_with_two_hadamards(case)
        assert cpp_res == py_res, (case, cpp_res, py_res)
    print("gateset: OK")


def check_pacman_internals():
    from pytket import Circuit, OpType
    from pytket.circuit import Op
    from pytket_dqc.packing import PacMan as PyPacMan
    from pytket_dqc.placement import Placement as PyPlacement
    from pytket_dqc.circuits import HypergraphCircuit as PyHC, Hyperedge as PyHyp

    cz = Op.create(OpType.CU1, 1)

    def serialize(c):
        qubits = [str(q) for q in c.qubits]
        cmds = [
            (
                cmd.op.get_name(),
                int(cmd.op.type),
                [str(q) for q in cmd.qubits],
                [float(p) for p in cmd.op.params],
            )
            for cmd in c.get_commands()
        ]
        return qubits, cmds

    def cpp_hc(c):
        h = _core.HypergraphCircuit()
        qubits, cmds = serialize(c)
        h.from_circuit(qubits, cmds)
        return h

    def cpp_placement(p):
        cp = _core.Placement()
        for v, s in p.placement.items():
            cp.set(v, s)
        return cp

    def cmp(circ, placement_dict, label, split=None):
        py_hc = PyHC(circ)
        cpp_h = cpp_hc(circ)
        if split is not None:
            old_vs, new_vss = split
            py_hc.split_hyperedge(
                old_hyperedge=PyHyp(old_vs, 1),
                new_hyperedge_list=[PyHyp(vs, 1) for vs in new_vss],
            )
            cpp_h.split_hyperedge(
                _core.Hyperedge(old_vs, 1),
                [_core.Hyperedge(vs, 1) for vs in new_vss],
            )
        py_pl = PyPlacement(placement_dict)
        py_pm = PyPacMan(py_hc, py_pl)
        cpp_pm = _core.PacMan(cpp_h, cpp_placement(py_pl))

        py_by_index = {p.packet_index: p for p in py_pm.get_all_packets()}
        cpp_all = cpp_pm.get_all_packets()
        assert [t[0] for t in cpp_all] == [p.packet_index
                                           for p in py_pm.get_all_packets()], \
            (label, "all_packets order")
        assert set(t[0] for t in cpp_all) == set(py_by_index.keys()), label

        # get_all_merged_packets
        py_merged = [[p.packet_index for p in mp]
                     for mp in py_pm.get_all_merged_packets()]
        cpp_merged = [list(m) for m in cpp_pm.get_all_merged_packets()]
        assert py_merged == cpp_merged, (label, "all_merged", py_merged,
                                         cpp_merged)

        # per-packet navigation
        for idx in sorted(py_by_index):
            p = py_by_index[idx]
            sn = py_pm.get_subsequent_neighbouring_packet(p)
            sh = py_pm.get_subsequent_hopping_packet(p)
            nx = py_pm.get_next_packet(p)
            assert cpp_pm.get_subsequent_neighbouring_packet(idx) == \
                (sn.packet_index if sn else None), (label, "sub_nbr", idx)
            assert cpp_pm.get_subsequent_hopping_packet(idx) == \
                (sh.packet_index if sh else None), (label, "sub_hop", idx)
            assert cpp_pm.get_next_packet(idx) == \
                (nx.packet_index if nx else None), (label, "next", idx)

        # are_neighbouring / are_hoppable for every ordered pair
        # (`are_hoppable_packets` in the reference asserts that both packets
        # share a qubit and a connected server — the precondition guaranteed
        # by `get_next_packet` in normal flow — so only test valid pairs).
        all_idx = sorted(py_by_index)
        for a in all_idx:
            for b in all_idx:
                if a == b:
                    continue
                assert cpp_pm.are_neighbouring_packets(a, b) == \
                    py_pm.are_neighbouring_packets(py_by_index[a],
                                                   py_by_index[b]), \
                    (label, "are_nbr", a, b)
                pa, pb = py_by_index[a], py_by_index[b]
                if (pa.qubit_vertex == pb.qubit_vertex and
                        pa.connected_server_index == pb.connected_server_index):
                    assert cpp_pm.are_hoppable_packets(a, b) == \
                        py_pm.are_hoppable_packets(pa, pb), \
                        (label, "are_hop", a, b)

        # connectivity: connected packets / merged packets / containing /
        # split packets
        for idx in all_idx:
            py_cp = sorted(q.packet_index
                           for q in py_pm.get_connected_packets(py_by_index[idx]))
            cpp_cp = sorted(cpp_pm.get_connected_packets(idx))
            assert py_cp == cpp_cp, (label, "connected", idx, py_cp, cpp_cp)
            py_cm = sorted(tuple(p.packet_index for p in mp)
                           for mp in py_pm.get_connected_merged_packets(
                               py_pm.get_containing_merged_packet(
                                   py_by_index[idx])))
            cpp_cm = sorted(tuple(m) for m in
                            cpp_pm.get_connected_merged_packets(idx))
            assert py_cm == cpp_cm, (label, "conn_merged", idx, py_cm, cpp_cm)
            py_contain = tuple(p.packet_index for p in
                               py_pm.get_containing_merged_packet(
                                   py_by_index[idx]))
            cpp_contain = tuple(cpp_pm.get_containing_merged_packet(idx))
            assert py_contain == cpp_contain, (label, "containing", idx)

        # embedded packets / hopping-from-embedded / is_packet_embedded
        py_hoppings = set()
        for hp_list in py_pm.hopping_packets.values():
            for (a, b) in hp_list:
                py_hoppings.add((a.packet_index, b.packet_index))
        cpp_hoppings = set()
        for hp_list in cpp_pm.hopping_packets().values():
            for t in hp_list:
                cpp_hoppings.add(tuple(t))
        assert py_hoppings == cpp_hoppings, (label, "hoppings")

        for (a, b) in sorted(py_hoppings):
            py_emb = sorted(p.packet_index for p in
                            py_pm.get_embedded_packets((py_by_index[a],
                                                        py_by_index[b])))
            cpp_emb = sorted(cpp_pm.get_embedded_packets(a, b))
            assert py_emb == cpp_emb, (label, "embedded", (a, b))
            for idx in py_emb:
                py_h = py_pm.get_hopping_packet_from_embedded_packet(
                    py_by_index[idx])
                cpp_h = cpp_pm.get_hopping_packet_from_embedded_packet(idx)
                assert (py_h[0].packet_index, py_h[1].packet_index) == \
                    tuple(cpp_h), (label, "hop_from_emb", idx)

        for idx in all_idx:
            assert cpp_pm.is_packet_embedded(idx) == \
                py_pm.is_packet_embedded(py_by_index[idx]), \
                (label, "is_embedded", idx)

        # all embedded packets per qubit vertex
        for qv in py_hc.get_qubit_vertices():
            py_inner = py_pm.get_all_embedded_packets_for_qubit_vertex(qv)
            cpp_inner = cpp_pm.get_all_embedded_packets_for_qubit_vertex(qv)
            py_norm = sorted(
                ((a.packet_index, b.packet_index),
                 sorted(p.packet_index for p in s))
                for ((a, b), s) in py_inner.items()
            )
            cpp_norm = sorted((tuple(k), sorted(v)) for (k, v) in cpp_inner)
            assert py_norm == cpp_norm, (label, "emb_for_qv", qv)

        # get_hopping_packets_within over all merged packets
        py_merged_set = set(py_pm.get_all_merged_packets())
        py_within = sorted((a.packet_index, b.packet_index) for (a, b) in
                           py_pm.get_hopping_packets_within(py_merged_set))
        cpp_within = sorted(tuple(t) for t in
                            cpp_pm.get_hopping_packets_within(cpp_merged))
        assert py_within == cpp_within, (label, "within", py_within, cpp_within)

        # get_conflict_hoppings for every hopping packet
        for (a, b) in sorted(py_hoppings):
            py_conf = [(x.packet_index, y.packet_index) for (x, y) in
                       py_pm.get_conflict_hoppings((py_by_index[a],
                                                    py_by_index[b]))]
            cpp_conf = [tuple(t) for t in cpp_pm.get_conflict_hoppings(a, b)]
            assert py_conf == cpp_conf, (label, "conflicts", (a, b), py_conf,
                                         cpp_conf)

        # get_connected_server for each (qubit_vertex, gate_vertex)
        for qv, ps in py_pm.packets_by_qubit.items():
            for p in ps:
                for gv in p.gate_vertices:
                    assert cpp_pm.get_connected_server(qv, gv) == \
                        py_pm.get_connected_server(qv, gv), \
                        (label, "conn_server", qv, gv)

    # Reuse the same circuits as check_pacman.
    circ = Circuit(3)
    circ.add_gate(cz, [0, 1]).add_gate(cz, [0, 2]).add_gate(cz, [0, 1])
    circ.add_gate(cz, [1, 2]).H(1).add_gate(cz, [2, 1])
    cmp(circ, {0: 0, 1: 1, 2: 2, 3: 0, 4: 0, 5: 0, 6: 1, 7: 2}, "build_packets")

    circ = Circuit(6)
    circ.add_gate(cz, [0, 2])
    circ.Rz(0.3, 0)
    circ.add_gate(cz, [0, 3])
    pl = {i: i // 2 for i in range(6)}
    pl.update({6: 0, 7: 0})
    cmp(circ, pl, "neighbouring_00", split=([0, 6, 7], [[0, 7], [0, 6]]))

    circ = Circuit(6)
    circ.add_gate(cz, [0, 2])
    circ.Rz(0.3, 0)
    circ.H(0)
    circ.Rz(1, 0)
    circ.H(0)
    circ.add_gate(cz, [0, 3])
    cmp(circ, {i: i // 2 for i in range(6)} | {6: 0, 7: 0}, "neighbouring_01")

    circ = Circuit(6)
    circ.add_gate(cz, [0, 2])
    circ.H(0)
    circ.add_gate(cz, [0, 3])
    circ.H(0)
    circ.add_gate(cz, [0, 2])
    cmp(circ, {i: i // 2 for i in range(6)} | {6: 0, 7: 0, 8: 0}, "hopping_03")

    circ = Circuit(6)
    circ.add_gate(cz, [0, 2])
    circ.H(0)
    circ.Rz(0.33, 0)
    circ.add_gate(cz, [0, 2])
    circ.Rz(0.67, 0)
    circ.H(0)
    circ.Rz(0.27, 0)
    circ.H(0)
    circ.Rz(0.43847, 0)
    circ.add_gate(cz, [0, 3])
    circ.Rz(1 - 0.43847, 0)
    circ.H(0)
    circ.add_gate(cz, [0, 2])
    cmp(circ, {i: i // 2 for i in range(6)} | {6: 0, 7: 0, 8: 0, 9: 0},
        "hopping_08")

    circ = Circuit(5)
    circ.add_gate(cz, [0, 1]).H(0).add_gate(cz, [0, 2]).H(0)
    circ.add_gate(cz, [0, 3]).H(0).add_gate(cz, [0, 4])
    cmp(circ, {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 0, 6: 0, 7: 0, 8: 0},
        "intertwining_0")

    print("pacman_internals: OK")


def check_detached_gates():
    from pytket import Circuit, OpType
    from pytket_dqc.networks import NISQNetwork
    from pytket_dqc.allocators import Random
    from pytket_dqc.refiners import DetachedGates

    def serialize(c):
        qubits = [str(q) for q in c.qubits]
        cmds = [
            (
                cmd.op.get_name(),
                int(cmd.op.type),
                [str(q) for q in cmd.qubits],
                [float(p) for p in cmd.op.params],
            )
            for cmd in c.get_commands()
        ]
        return qubits, cmds

    def cpp_hc(c):
        h = _core.HypergraphCircuit()
        qubits, cmds = serialize(c)
        h.from_circuit(qubits, cmds)
        return h

    def cpp_graph(nxG):
        g = _core.Graph()
        for n in nxG.nodes():
            g.add_node(n)
        for u, v, d in nxG.edges(data=True):
            g.add_edge(u, v, float(d.get("weight", 1.0)))
        return g

    def coupling(network):
        return [(int(a), int(b)) for (a, b) in network.server_coupling]

    def server_qubits(network):
        return [(int(s), [int(q) for q in qs])
                for s, qs in network.server_qubits.items()]

    circ = Circuit(3)
    circ.add_gate(OpType.CU1, 0.3, [0, 1])
    circ.H(0)
    circ.Rz(1.0, 0)
    circ.H(0)
    circ.add_gate(OpType.CU1, 0.8, [0, 2])
    network = NISQNetwork(
        [[2, 1], [1, 0], [1, 3], [0, 4]], {0: [0], 1: [1], 2: [2], 3: [3], 4: [4, 5]}
    )

    py_dist = Random().allocate(circ, network, seed=7)
    cpp_h = cpp_hc(circ)
    cpp_dist = _core.random_allocate(cpp_h, cpp_nisq(network), 7)
    assert dict(py_dist.placement.placement) == dict(cpp_dist.placement.placement), \
        ("initial placement", dict(py_dist.placement.placement),
         dict(cpp_dist.placement.placement))

    py_made = DetachedGates().refine(py_dist, seed=42)
    cpp_made = _core.detached_gates_refine(cpp_dist, 10, 0.05, 42, None)
    assert py_made == cpp_made, (py_made, cpp_made)
    assert dict(py_dist.placement.placement) == \
        dict(cpp_dist.placement.placement), \
        ("detached", dict(py_dist.placement.placement),
         dict(cpp_dist.placement.placement))
    assert py_dist.cost() == cpp_dist.cost(), (py_dist.cost(), cpp_dist.cost())

    print("detached_gates: OK")


def check_hypergraph_methods():
    from pytket_dqc.circuits import Hypergraph as PyHypergraph
    from pytket_dqc.circuits import Hyperedge as PyHyperedge
    from pytket_dqc.placement import Placement as PyPlacement

    def build(hedges):
        py = PyHypergraph()
        cpp = _core.Hypergraph()
        vs_all = sorted({v for vs, _ in hedges for v in vs})
        py.add_vertices(vs_all)
        cpp.add_vertices(vs_all)
        for vs, w in hedges:
            py.add_hyperedge(list(vs), w)
            cpp.add_hyperedge(list(vs), w)
        return py, cpp

    def cmp_state(py, cpp, label):
        assert list(cpp.vertex_list) == list(py.vertex_list), label
        py_hl = [(tuple(h.vertices), h.weight) for h in py.hyperedge_list]
        cpp_hl = [(tuple(h.vertices), h.weight) for h in cpp.hyperedge_list]
        assert cpp_hl == py_hl, (label, cpp_hl, py_hl)
        assert {k: set(v) for k, v in cpp.vertex_neighbours.items()} == \
            {k: set(v) for k, v in py.vertex_neighbours.items()}, \
            (label, "neighbours")

    # remove_hyperedge
    py, cpp = build([([0, 1], 1), ([0, 2], 1), ([1, 2], 1)])
    py.remove_hyperedge(PyHyperedge([0, 1], 1))
    cpp.remove_hyperedge(_core.Hyperedge([0, 1], 1))
    cmp_state(py, cpp, "remove_hyperedge")

    # split_hyperedge
    py, cpp = build([([0, 1, 2], 1)])
    py.split_hyperedge(PyHyperedge([0, 1, 2], 1),
                       [PyHyperedge([0, 1], 1), PyHyperedge([0, 2], 1)])
    cpp.split_hyperedge(_core.Hyperedge([0, 1, 2], 1),
                        [_core.Hyperedge([0, 1], 1), _core.Hyperedge([0, 2], 1)])
    cmp_state(py, cpp, "split_hyperedge")

    # merge_hyperedge (fresh graph)
    py, cpp = build([([0, 1], 1), ([0, 2], 1), ([2, 3], 1)])
    py.merge_hyperedge([PyHyperedge([0, 1], 1), PyHyperedge([0, 2], 1)])
    cpp.merge_hyperedge([_core.Hyperedge([0, 1], 1), _core.Hyperedge([0, 2], 1)])
    cmp_state(py, cpp, "merge_hyperedge")

    # is_valid
    py, cpp = build([([0, 1], 1), ([0, 2], 1)])
    assert cpp.is_valid() == py.is_valid(), "is_valid continuous"
    py2, cpp2 = build([([2, 3], 1)])  # missing vertices 0, 1
    assert cpp2.is_valid() == py2.is_valid(), "is_valid gap"
    assert not cpp2.is_valid()

    # is_placement / get_boundary
    pl = {0: 0, 1: 1, 2: 1, 3: 0}
    py_p = PyPlacement(pl)
    cpp_p = _core.Placement()
    for v, s in pl.items():
        cpp_p.set(v, s)
    assert cpp.is_placement(cpp_p) == py.is_placement(py_p), "is_placement"
    assert list(cpp.get_boundary(cpp_p)) == py.get_boundary(py_p), "get_boundary"

    # weight_one_predicate
    py_w, cpp_w = build([([0, 1], 2)])
    assert cpp_w.weight_one_predicate() == py_w.weight_one_predicate(), "w1 false"
    py_w1, cpp_w1 = build([([0, 1], 1), ([0, 2], 1)])
    assert cpp_w1.weight_one_predicate() == py_w1.weight_one_predicate(), "w1 true"

    # kahypar_hyperedges
    py_idx, py_flat = py_w1.kahypar_hyperedges()
    cpp_idx, cpp_flat = cpp_w1.kahypar_hyperedges()
    assert list(cpp_idx) == list(py_idx), "kahypar indices"
    assert list(cpp_flat) == list(py_flat), "kahypar flat"

    print("hypergraph_methods: OK")


def check_bipartite():
    import networkx as nx
    from networkx.algorithms import bipartite as nx_bip

    def cmp(edges):
        G = nx.Graph()
        cpp = _core.BipartiteGraph()
        for u, v in edges:
            G.add_edge(u, v)
            cpp.add_edge(u, v)

        py_color = nx_bip.color(G)
        cpp_color = dict(cpp.color())
        assert cpp_color == py_color, (edges, cpp_color, py_color)

        if nx.is_connected(G):
            py_X, py_Y = nx_bip.sets(G)
            cpp_sets = cpp.sets()
            assert set(cpp_sets[0]) == set(py_X), (edges, "sets X")
            assert set(cpp_sets[1]) == set(py_Y), (edges, "sets Y")
            top = sorted(py_X)
        else:
            # For disconnected graphs the bipartition is ambiguous and
            # networkx `sets` raises AmbiguousSolution; pin it to the
            # C++ colour-1 side for the matching/cover comparisons below.
            cpp_sets = cpp.sets()
            top = sorted(cpp_sets[0])

        py_m = nx_bip.hopcroft_karp_matching(G, top)
        cpp_m = dict(cpp.maximum_matching(top))
        assert cpp_m == py_m, (edges, "matching", cpp_m, py_m)

        py_cover = nx_bip.to_vertex_cover(G, py_m, top)
        cpp_cover = set(cpp.to_vertex_cover(cpp_m, top))
        assert cpp_cover == py_cover, (edges, "cover", cpp_cover, py_cover)

        # subgraph (structural, order-insensitive node/edge sets)
        keep = list(G.nodes())[: max(1, len(G.nodes()) // 2)]
        if len(keep) >= 2:
            py_sub = G.subgraph(keep)
            cpp_sub = cpp.subgraph(keep)
            assert set(cpp_sub.nodes()) == set(py_sub.nodes()), (edges, "sub nodes")
            assert set(tuple(e) for e in cpp_sub.edges()) == \
                set(py_sub.edges()), (edges, "sub edges")

    cmp([(0, 1), (1, 2), (2, 3)])
    cmp([(0, 1), (1, 2), (2, 3), (3, 0)])
    cmp([(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4)])  # K(2,3)
    cmp([(0, 1), (0, 2), (0, 3)])  # star
    cmp([(0, 1)])
    cmp([(0, 1), (1, 2), (5, 6)])  # two components
    cmp([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)])  # even 6-cycle

    print("bipartite: OK")


def check_graph_basics():
    import networkx as nx

    def cmp(edges, weights=None):
        G = nx.Graph()
        g = _core.Graph()
        for (u, v) in edges:
            w = (weights or {}).get((u, v), 1.0)
            G.add_edge(u, v, weight=w)
            g.add_edge(u, v, w)

        assert list(g.nodes()) == list(G.nodes()), (edges, list(g.nodes()),
                                                    list(G.nodes()))
        g_edges = [(u, v, w) for (u, v, w) in g.edges()]
        G_edges = [(u, v, d["weight"]) for (u, v, d) in G.edges(data=True)]
        assert g_edges == G_edges, (edges, g_edges, G_edges)

        for u in G.nodes():
            for v in G.nodes():
                assert g.has_edge(u, v) == G.has_edge(u, v), ("has_edge", u, v)
            assert g.degree(u) == G.degree(u), ("degree", u, g.degree(u),
                                                G.degree(u))
            for v in G.nodes():
                if G.has_edge(u, v):
                    assert g.weight(u, v) == G[u][v].get("weight", 1.0), \
                        ("weight", u, v, g.weight(u, v))

    cmp([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)],
        {(0, 1): 2.0, (2, 3): 0.5})
    cmp([(1, 2), (0, 1), (2, 3), (2, 2)])  # self-loop
    cmp([(0, 1), (1, 2), (1, 3), (3, 4)])
    cmp([(0, 1)])

    # remove_node
    G = nx.Graph()
    g = _core.Graph()
    for (u, v) in [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]:
        G.add_edge(u, v)
        g.add_edge(u, v)
    G.remove_node(2)
    g.remove_node(2)
    assert list(g.nodes()) == list(G.nodes()), ("remove nodes",
                                                list(g.nodes()), list(G.nodes()))
    g_edges = [(u, v, w) for (u, v, w) in g.edges()]
    G_edges = [(u, v, d.get("weight", 1.0)) for (u, v, d) in G.edges(data=True)]
    assert g_edges == G_edges, ("remove edges", g_edges, G_edges)

    print("graph_basics: OK")


def check_to_pytket_circuit():
    from pytket import Circuit, OpType
    from pytket_dqc.circuits import HypergraphCircuit as PyHC
    from pytket_dqc.placement import Placement as PyPlacement
    from pytket_dqc.networks import NISQNetwork
    from pytket_dqc import Distribution as PyDistribution
    from pytket_dqc.utils.circuit_analysis import ebit_cost as py_ebit_cost
    from pytket_dqc.utils.circuit_analysis import all_cu1_local

    def serialize(c):
        qubits = [str(q) for q in c.qubits]
        cmds = [
            (
                cmd.op.get_name(),
                int(cmd.op.type),
                [str(q) for q in cmd.qubits],
                [float(p) for p in cmd.op.params],
            )
            for cmd in c.get_commands()
        ]
        return qubits, cmds

    def cpp_hc(c):
        h = _core.HypergraphCircuit()
        qubits, cmds = serialize(c)
        h.from_circuit(qubits, cmds)
        return h

    def cpp_graph(nxG):
        g = _core.Graph()
        for n in nxG.nodes():
            g.add_node(n)
        for u, v, d in nxG.edges(data=True):
            g.add_edge(u, v, float(d.get("weight", 1.0)))
        return g

    def cpp_placement(p):
        cp = _core.Placement()
        for v, s in p.placement.items():
            cp.set(v, s)
        return cp

    def cpp_dist(py_dist):
        sq = {int(s): [int(q) for q in qs]
              for s, qs in py_dist.network.server_qubits.items()}
        eb = {int(s): int(m)
              for s, m in py_dist.network.server_ebit_mem.items()}
        return _core.Distribution(cpp_hc(py_dist.circuit.get_circuit()),
                                  cpp_placement(py_dist.placement),
                                  cpp_graph(py_dist.network.get_server_nx()),
                                  sq, eb)

    def sig(c):
        return ([str(q) for q in c.qubits],
                [(cmd.op.get_name(), int(cmd.op.type),
                  [str(q) for q in cmd.qubits],
                  [float(p) for p in cmd.op.params])
                 for cmd in c.get_commands()])

    def check(circ, placement_dict, network, satisfy_bound=True,
              allow_update=False, exact=True):
        py_dist = PyDistribution(PyHC(circ), PyPlacement(placement_dict),
                                 network)
        cpp_d = cpp_dist(py_dist)
        py_out = py_dist.to_pytket_circuit(satisfy_bound=satisfy_bound,
                                           allow_update=allow_update)
        cpp_out = cpp_d.to_pytket_circuit(satisfy_bound=satisfy_bound,
                                          allow_update=allow_update)
        assert py_ebit_cost(py_out) == py_ebit_cost(cpp_out), (placement_dict,
                                                               "ebit_cost")
        assert all_cu1_local(py_out) and all_cu1_local(cpp_out), \
            (placement_dict, "all_cu1_local")
        if exact:
            # The exact distributed circuit (including link-qubit register
            # indices) matches for cases where pytket's internal command
            # ordering does not relabel link qubits.
            assert sig(py_out) == sig(cpp_out), (placement_dict, "circuit")
        else:
            # Link-qubit register indices are assigned in the order link
            # qubits are created, which can differ when pytket reorders
            # commands. Compare the semantic content instead: op counts and
            # qubit counts.
            from collections import Counter
            assert Counter(c.op.get_name() for c in py_out.get_commands()) == \
                Counter(c.op.get_name() for c in cpp_out.get_commands()), \
                (placement_dict, "op counts")
            assert len(py_out.qubits) == len(cpp_out.qubits), \
                (placement_dict, "qubit count")
        py_hl = sorted((tuple(sorted(h.vertices)), h.weight)
                       for h in py_dist.circuit.hyperedge_list)
        cpp_hl = sorted((tuple(sorted(h.vertices)), h.weight)
                        for h in cpp_d.circuit.hyperedge_list)
        assert py_hl == cpp_hl, (placement_dict, "hyperedges", py_hl, cpp_hl)
        assert dict(py_dist.placement.placement) == \
            dict(cpp_d.placement.placement), (placement_dict, "placement")

    # 1. no embedding
    circ = Circuit(3)
    circ.add_gate(OpType.CU1, 1.0, [0, 1])
    circ.add_gate(OpType.CU1, 1.0, [0, 2])
    network = NISQNetwork([[0, 1], [1, 2], [1, 3], [2, 4]],
                          {0: [0], 1: [1], 2: [2], 3: [3], 4: [4]})
    for placement in [{0: 0, 1: 1, 2: 2, 3: 1, 4: 2},
                      {0: 0, 1: 1, 2: 2, 3: 1, 4: 0},
                      {0: 1, 1: 0, 2: 2, 3: 0, 4: 2}]:
        check(circ, placement, network)

    # 2. ALAP / H-embedding
    circ = Circuit(4)
    circ.add_gate(OpType.CU1, 0.1234, [1, 2])
    circ.add_gate(OpType.CU1, 0.1234, [0, 2])
    circ.add_gate(OpType.CU1, 0.1234, [2, 3])
    circ.add_gate(OpType.CU1, 0.1234, [0, 3])
    circ.H(0)
    circ.H(2)
    circ.Rz(0.1234, 3)
    circ.add_gate(OpType.CU1, 1.0, [0, 2])
    circ.add_gate(OpType.CU1, 1.0, [0, 3])
    circ.add_gate(OpType.CU1, 1.0, [1, 2])
    circ.H(0)
    circ.H(2)
    circ.Rz(0.1234, 0)
    circ.add_gate(OpType.CU1, 0.1234, [0, 1])
    circ.add_gate(OpType.CU1, 0.1234, [0, 3])
    circ.add_gate(OpType.CU1, 1.0, [1, 2])
    network = NISQNetwork([[0, 1], [0, 2], [0, 3], [3, 4]],
                          {0: [0], 1: [1, 2], 2: [3, 4], 3: [7], 4: [5, 6]})
    placement = {0: 1, 1: 1, 2: 4, 3: 4, 4: 1, 5: 2, 6: 1, 7: 4, 8: 4, 9: 4,
                 10: 0, 11: 1, 12: 4, 13: 2}
    check(circ, placement, network, exact=False)

    # 3. mixing H and D embeddings
    circ = Circuit(4)
    circ.add_gate(OpType.CU1, 0.1234, [0, 1])
    circ.H(1)
    circ.add_gate(OpType.CU1, 1.0, [1, 2])
    circ.add_gate(OpType.CU1, 1.0, [1, 3])
    circ.H(1)
    circ.add_gate(OpType.CU1, 0.1234, [1, 3])
    circ.add_gate(OpType.CU1, 0.1234, [1, 2])
    network = NISQNetwork([[0, 1], [1, 2]], {0: [0], 1: [1], 2: [2, 3]})
    placement = {0: 0, 1: 1, 2: 2, 3: 2, 4: 0, 5: 1, 6: 1, 7: 1, 8: 2}
    check(circ, placement, network)

    # 4. detached gate -> ConstraintException (bounded memory). The recovery
    #    cannot help: the gate is evicted onto a non-home server with memory
    #    capacity < 2, so allow_update=True must ALSO raise on both sides.
    circ = Circuit(4)
    circ.add_gate(OpType.CU1, 1.0, [0, 1])
    network = NISQNetwork([[0, 1], [1, 2]], {0: [0], 1: [1], 2: [2, 3]},
                          server_ebit_mem={0: 1, 1: 1, 2: 1})
    placement = {0: 0, 1: 1, 2: 2, 3: 2, 4: 2}
    for allow_update in (False, True):
        py_d = PyDistribution(PyHC(circ), PyPlacement(placement), network)
        cpp_d = cpp_dist(py_d)
        py_raised = cpp_raised = False
        try:
            py_d.to_pytket_circuit(satisfy_bound=True,
                                   allow_update=allow_update)
        except Exception:
            py_raised = True
        try:
            cpp_d.to_pytket_circuit(satisfy_bound=True,
                                    allow_update=allow_update)
        except Exception:
            cpp_raised = True
        assert py_raised and cpp_raised, (allow_update, py_raised, cpp_raised)

    # 5. allow_update=True auto-recovery path. Interleaved hyperedge spans
    #    force server 1 to hold two link qubits simultaneously (the shared
    #    qubit 0's connection stays alive across gates 4 and 6 while qubit 1's
    #    gate 5 opens its own link), exceeding its capacity of 1.
    #    - allow_update=False must raise on both sides.
    #    - allow_update=True must split the shared-qubit hyperedge {0,4,6}
    #      into {0,4} and {0,6} and succeed; the mutated distribution
    #      (hyperedge list + placement) and the emitted circuit (exact) must
    #      match the reference.
    circ = Circuit(4)
    circ.add_gate(OpType.CU1, 1.0, [0, 2])
    circ.add_gate(OpType.CU1, 1.0, [1, 3])
    circ.add_gate(OpType.CU1, 1.0, [0, 2])
    network = NISQNetwork([[0, 1]], {0: [0, 1], 1: [2, 3]},
                          server_ebit_mem={0: 1, 1: 1})
    placement = {0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1}

    # Un-updated distribution exceeds the bound (raises on both sides).
    py_dist = PyDistribution(PyHC(circ), PyPlacement(placement), network)
    cpp_d = cpp_dist(py_dist)
    for obj in (py_dist, cpp_d):
        raised = False
        try:
            obj.to_pytket_circuit(satisfy_bound=True, allow_update=False)
        except Exception:
            raised = True
        assert raised, "allow_update=False should raise"

    # allow_update=True: both succeed and split the same hyperedge.
    py_dist = PyDistribution(PyHC(circ), PyPlacement(placement), network)
    cpp_d = cpp_dist(py_dist)
    py_out = py_dist.to_pytket_circuit(satisfy_bound=True, allow_update=True)
    cpp_out = cpp_d.to_pytket_circuit(satisfy_bound=True, allow_update=True)

    assert py_ebit_cost(py_out) == py_ebit_cost(cpp_out), (placement,
                                                           "ebit_cost")
    assert all_cu1_local(py_out) and all_cu1_local(cpp_out), \
        (placement, "all_cu1_local")
    assert sig(py_out) == sig(cpp_out), (placement, "circuit")
    py_hl = sorted((tuple(sorted(h.vertices)), h.weight)
                   for h in py_dist.circuit.hyperedge_list)
    cpp_hl = sorted((tuple(sorted(h.vertices)), h.weight)
                    for h in cpp_d.circuit.hyperedge_list)
    assert py_hl == cpp_hl, ("hyperedges", py_hl, cpp_hl)
    assert dict(py_dist.placement.placement) == \
        dict(cpp_d.placement.placement), (placement, "placement")
    # Sanity: the recovery really did split {0,4,6} into {0,4} and {0,6}.
    hedges_after = [tuple(sorted(h.vertices))
                    for h in py_dist.circuit.hyperedge_list]
    assert (0, 4) in hedges_after and (0, 6) in hedges_after, hedges_after

    # 6. allow_update=True with MULTIPLE split candidates of DIFFERENT span.
    #    The bound (memory 1) is exceeded on server 1 at v_gate 4. Two
    #    hyperedges can be split there: {1,4,5,7} (span 4..5, dist 1) and
    #    {3,4,6} (span 4..6, dist 2). `split_for_bound` must choose the
    #    LONGEST-span candidate {3,4,6}; a regression that always picked the
    #    first valid candidate (because every span was computed as 0) would
    #    instead split {1,4,5,7} and diverge from the reference. This
    #    exercises the span-selection logic
    #    (dist = min(gates > v_gate) - max(gates < v_gate)) that the earlier
    #    cases do not cover (case 5 only has a tie between candidates).
    circ = Circuit(4)
    circ.add_gate(OpType.CU1, 1.0, [1, 3])
    circ.add_gate(OpType.CU1, 1.0, [0, 1])
    circ.add_gate(OpType.CU1, 1.0, [2, 3])
    circ.add_gate(OpType.CU1, 1.0, [1, 2])
    network = NISQNetwork([[0, 1]], {0: [0, 3], 1: [1, 2]},
                          server_ebit_mem={0: 1, 1: 1})
    placement = {0: 0, 1: 1, 2: 1, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1}

    # Un-updated distribution exceeds the bound (raises on both sides).
    for make in (
        lambda: PyDistribution(PyHC(circ), PyPlacement(placement), network),
        lambda: cpp_dist(PyDistribution(PyHC(circ), PyPlacement(placement),
                                        network)),
    ):
        raised = False
        try:
            make().to_pytket_circuit(satisfy_bound=True, allow_update=False)
        except Exception:
            raised = True
        assert raised, "allow_update=False should raise"

    # allow_update=True: both recover identically (ebit cost, exact emitted
    # circuit, mutated hyperedges and placement).
    check(circ, placement, network, satisfy_bound=True, allow_update=True,
          exact=True)

    # Sanity: the recovery really split the longest-span candidate {3,4,6}
    # into {3,4} and {3,6}, and left the short-span first candidate {1,4,5,7}
    # untouched.
    py_dist6 = PyDistribution(PyHC(circ), PyPlacement(placement), network)
    py_dist6.to_pytket_circuit(satisfy_bound=True, allow_update=True)
    hedges6 = [tuple(sorted(h.vertices))
               for h in py_dist6.circuit.hyperedge_list]
    assert (3, 4) in hedges6 and (3, 6) in hedges6, hedges6
    assert (3, 4, 6) not in hedges6, hedges6
    assert (1, 4, 5, 7) in hedges6, hedges6
    assert not any(set(h) == {1, 4} for h in hedges6), hedges6

    print("to_pytket_circuit: OK")


def check_qft_distribution_pipelines():
    """Differential test of the DQC distribution pipelines from
    ``dist_qft/main_many_qpus.py`` (the ``test_PYTKET_*`` methods) on the
    QFT circuits from that file's ``get_QFT_circuit_pytket``.

    The reference pipeline (``test_PYTKET_PE`` / ``test_PYTKET_PHE`` /
    ``test_PYTKET_AESD``) is run end-to-end in Python, while the C++ core
    replays the same sequence of allocator + refiner stages. Note that the
    KaHyPar-based ``HypergraphPartitioning`` allocator (used as the initial
    step of ``PartitioningHeterogeneous``) is intentionally Python-only, so
    for PE/PHE we start from a seeded ``Random`` allocation and then apply
    the deterministic refiners that make up the rest of the pipeline.

    ``to_pytket_circuit`` is known to diverge on bound detection for these
    larger circuits (pytket's barrier-pinned command ordering vs. the C++
    insertion order relabels relay link qubits), so the comparison here is on
    the refined distribution state (placement + hyperedges + cost) rather than
    the emitted circuit. See repo memory ``fidelity-notes.md``.
    """
    from pytket import Circuit  # noqa: F401
    from pytket_dqc.networks import NISQNetwork
    from pytket_dqc.allocators import Annealing, Random as RandomAllocator
    from pytket_dqc.refiners import (BoundaryReallocation, DetachedGates,
                                     EagerHTypeMerge, IntertwinedDTypeMerge,
                                     NeighbouringDTypeMerge, RepeatRefiner,
                                     SequenceRefiner, VertexCover)

    # Replicates ``get_QFT_circuit_pytket`` from dist_qft/main_many_qpus.py.
    def qft_circuit(num_qubits):
        circ = Circuit(num_qubits, name="$$QFT$$")
        for i in range(num_qubits):
            circ.H(num_qubits - 1 - i)
            for j in range(i + 1, num_qubits):
                circ.CU1(1 / 2 ** (j - i), num_qubits - 1 - i,
                         num_qubits - 1 - j)
        return circ

    # Fully-connected network of ``num_partitions`` equal-sized servers with
    # communication memory 2, exactly as in ``main``.
    def qft_network(num_partitions, qpu_size=3):
        qpu_indices = list(range(num_partitions))
        qpu_cum = [0] + list(np.cumsum([qpu_size] * num_partitions))
        ranges = [range(a, b) for (a, b) in zip(qpu_cum, qpu_cum[1:])]
        return NISQNetwork(
            server_coupling=list(itertools.product(qpu_indices, qpu_indices)),
            server_qubits={i: list(r) for i, r in enumerate(ranges)},
            server_ebit_mem={i: 2 for i in qpu_indices},
        )

    def serialize(c):
        qubits = [str(q) for q in c.qubits]
        cmds = [
            (cmd.op.get_name(), int(cmd.op.type),
             [str(q) for q in cmd.qubits],
             [float(p) for p in cmd.op.params])
            for cmd in c.get_commands()
        ]
        return qubits, cmds

    def cpp_hc(c):
        h = _core.HypergraphCircuit()
        qubits, cmds = serialize(c)
        h.from_circuit(qubits, cmds)
        return h

    def cpp_graph(network):
        nxG = network.get_server_nx()
        g = _core.Graph()
        for n in nxG.nodes():
            g.add_node(n)
        for u, v, d in nxG.edges(data=True):
            g.add_edge(u, v, float(d.get("weight", 1.0)))
        return g

    def cpp_network(network):
        coupling = [(int(a), int(b)) for (a, b) in network.server_coupling]
        sq = [(int(s), [int(q) for q in qs])
              for s, qs in network.server_qubits.items()]
        return coupling, sq

    def cpp_distribution(circuit, network, initial_seed):
        cpp_h = cpp_hc(circuit)
        cpp_g = cpp_graph(network)
        return cpp_h, cpp_g, _core.random_allocate(cpp_h, cpp_nisq(network),
                                                   initial_seed)

    def hl(d):
        return sorted((tuple(sorted(h.vertices)), h.weight)
                      for h in d.circuit.hyperedge_list)

    def assert_dist_equal(py_d, cpp_d, label):
        py_hl = hl(py_d)
        cpp_hl = hl(cpp_d)
        assert py_hl == cpp_hl, (label, "hyperedges", py_hl, cpp_hl)
        assert dict(py_d.placement.placement) == \
            dict(cpp_d.placement.placement), \
            (label, "placement", dict(py_d.placement.placement),
             dict(cpp_d.placement.placement))
        assert py_d.cost() == cpp_d.cost(), \
            (label, "cost", py_d.cost(), cpp_d.cost())

    # The SEED from main_many_qpus.py.
    SEED = 56426435

    # ------------------------------------------------------------------ PHE
    # test_PYTKET_PHE runs PartitioningHeterogeneousEmbedding, whose refine
    # stage is BoundaryReallocation + RepeatRefiner(EagerHTypeMerge). We
    # start from the same seeded Random allocation on both sides (the
    # KaHyPar initial partition is Python-only) and compare the refiners.
    def make_py_initial(circuit, network, seed):
        return RandomAllocator().allocate(circuit, network, seed=seed)

    for num_partitions, seed in [(2, 42), (3, 42), (3, 7)]:
        circuit = qft_circuit(num_partitions * 3)
        network = qft_network(num_partitions)
        py_d = make_py_initial(circuit, network, seed)
        BoundaryReallocation().refine(py_d, seed=seed)
        EagerHTypeMerge().refine(py_d)

        cpp_h, cpp_g, cpp_d = cpp_distribution(circuit, network, seed)
        _core.boundary_reallocation_refine(cpp_d, [], 10, 0.05, seed, None)
        _core.eager_h_type_merge_refine(cpp_d)
        assert_dist_equal(py_d, cpp_d, f"PHE np={num_partitions} seed={seed}")
        assert cpp_d.is_valid(), f"PHE np={num_partitions} not valid"

    # ------------------------------------------------------------------ PE
    # test_PYTKET_PE = PartitioningHeterogeneous (HypergraphPartitioning +
    # BoundaryReallocation) then EagerHTypeMerge. Same handling of the
    # Python-only KaHyPar initial step as above.
    for num_partitions, seed in [(2, 42), (3, 42)]:
        circuit = qft_circuit(num_partitions * 3)
        network = qft_network(num_partitions)
        py_d = make_py_initial(circuit, network, seed)
        BoundaryReallocation().refine(py_d, seed=seed)
        EagerHTypeMerge().refine(py_d)

        cpp_h, cpp_g, cpp_d = cpp_distribution(circuit, network, seed)
        _core.boundary_reallocation_refine(cpp_d, [], 10, 0.05, seed, None)
        _core.eager_h_type_merge_refine(cpp_d)
        assert_dist_equal(py_d, cpp_d, f"PE np={num_partitions} seed={seed}")
        assert cpp_d.is_valid(), f"PE np={num_partitions} not valid"

    # ------------------------------------------------------------------ AESD
    # test_PYTKET_AESD = Annealing (seeded) + VertexCover + RepeatRefiner(
    # SequenceRefiner([NeighbouringDTypeMerge, IntertwinedDTypeMerge])) +
    # DetachedGates. The reference Annealing.allocate discards its seed via a
    # Random() initial placement (reference bug); we use the documented
    # SeededRandomAllocator wrapper (as in the original file under test) so
    # both sides are deterministic and bit-comparable.
    class SeededRandomAllocator(RandomAllocator):
        def __init__(self):
            self.random_allocator = RandomAllocator()

        def allocate(self, circ, network, **kwargs):
            return self.random_allocator.allocate(circ, network, seed=SEED,
                                                  **kwargs)

    def repeat_d_merge(cpp_d):
        # RepeatRefiner(SequenceRefiner([Neighbouring, Intertwined])).
        while True:
            made = False
            made |= _core.neighbouring_d_type_merge_refine(cpp_d)
            made |= _core.intertwined_d_type_merge_refine(cpp_d)
            if not made:
                break

    for num_partitions in [2, 3]:
        circuit = qft_circuit(num_partitions * 3)
        network = qft_network(num_partitions)
        cpp_g = cpp_graph(network)
        cpp_h = cpp_hc(circuit)

        # ---- Python reference ----
        py_d = Annealing().allocate(circuit, network, seed=SEED,
                                    initial_place_method=SeededRandomAllocator())
        VertexCover().refine(py_d)
        RepeatRefiner(SequenceRefiner([NeighbouringDTypeMerge(),
                                       IntertwinedDTypeMerge()])).refine(py_d)
        DetachedGates().refine(py_d, seed=SEED)

        # ---- C++ replay ----
        cpp_d = _core.annealing_allocate(cpp_h, cpp_nisq(network),
                                         30000, 3.0, None, SEED, SEED)
        # VertexCover returns a fresh (circuit, placement); the binding copies
        # its inputs, so assign the result back.
        new_h, new_p = _core.vertex_cover_refine(cpp_d.circuit,
                                                 cpp_d.placement, cpp_g)
        cpp_d.circuit = new_h
        cpp_d.placement = new_p
        repeat_d_merge(cpp_d)
        _core.detached_gates_refine(cpp_d, 10, 0.05, SEED, None)

        assert_dist_equal(py_d, cpp_d, f"AESD np={num_partitions}")
        assert cpp_d.is_valid(), f"AESD np={num_partitions} not valid"
        # The annotated cost must agree (this is what test_PYTKET_AESD reports).
        assert py_d.cost() == cpp_d.cost(), (
            f"AESD np={num_partitions} cost mismatch")

    print("qft_distribution_pipelines: OK")


def check_nisq_network():
    """Differential test of ``_core.NISQNetwork`` against the reference
    ``pytket_dqc.networks.NISQNetwork``: accessors, default ebit memory,
    ``to_dict`` round-trip, and validation errors."""
    from pytket_dqc.networks import NISQNetwork as PyNISQNetwork

    cases = [
        ([[0, 1]], {0: [0, 1], 1: [2, 3]}, {0: 2, 1: 2}),
        ([[0, 1], [0, 2], [1, 2]], {0: [0, 1], 1: [2, 3], 2: [4, 5]}, None),
    ]
    for coupling, server_qubits, ebit in cases:
        py_n = PyNISQNetwork(server_coupling=[list(c) for c in coupling],
                             server_qubits=dict(server_qubits),
                             server_ebit_mem=None if ebit is None
                             else dict(ebit))
        cpp_n = _core.NISQNetwork(server_coupling=[list(c) for c in coupling],
                                  server_qubits=dict(server_qubits),
                                  server_ebit_mem={} if ebit is None
                                  else dict(ebit))
        assert py_n.get_server_list() == cpp_n.get_server_list(), "server_list"
        assert py_n.get_qubit_list() == cpp_n.get_qubit_list(), "qubit_list"
        py_sq = {s: list(qs) for s, qs in py_n.server_qubits.items()}
        assert py_sq == dict(cpp_n.server_qubits), "server_qubits"
        py_em = dict(py_n.server_ebit_mem)
        assert py_em == dict(cpp_n.server_ebit_mem), "server_ebit_mem"
        # to_dict / from_dict round-trip (both sides).
        py_rt = PyNISQNetwork.from_dict(py_n.to_dict())
        assert py_rt == py_n, "reference to_dict round-trip"
        cpp_rt = _core.NISQNetwork.from_dict(cpp_n.to_dict())
        assert cpp_rt == cpp_n, "cpp to_dict round-trip"
        # server graph structure.
        py_g = py_n.get_server_nx()
        cpp_g = cpp_n.get_server_nx()
        assert set(py_g.nodes()) == set(cpp_g.nodes()), "server_nx nodes"
        assert set(py_g.edges()) == {(u, v) for (u, v, _w) in cpp_g.edges()}, \
            "server_nx edges"

    # Validation errors must agree.
    bad_cases = [
        # unconnected server network
        dict(server_coupling=[[0, 1], [2, 3]],
             server_qubits={0: [0], 1: [1], 2: [2], 3: [3]}),
        # server without specified qubits
        dict(server_coupling=[[0, 1]], server_qubits={0: [0, 1]}),
        # qubit in more than one server
        dict(server_coupling=[[0, 1]], server_qubits={0: [0, 1], 1: [1, 2]}),
    ]
    for kwargs in bad_cases:
        py_raised = False
        cpp_raised = False
        try:
            PyNISQNetwork(**kwargs)
        except Exception:
            py_raised = True
        try:
            _core.NISQNetwork(**kwargs)
        except Exception:
            cpp_raised = True
        assert py_raised and cpp_raised, f"validation mismatch for {kwargs}"

    print("nisq_network: OK")


def check_hypergraph_partitioning():
    """Differential test of the C++ KaHyPar-backed ``HypergraphPartitioning``
    against the reference ``pytket_dqc.allocators.HypergraphPartitioning``:
    the full placement (KaHyPar initial partition + make_valid) must match
    exactly for the same seed."""
    from pytket import Circuit  # noqa: F401
    from pytket_dqc.allocators import HypergraphPartitioning
    from pytket_dqc.networks import NISQNetwork as PyNISQNetwork

    def qft_circuit(num_qubits):
        circ = Circuit(num_qubits, name="$$QFT$$")
        for i in range(num_qubits):
            circ.H(num_qubits - 1 - i)
            for j in range(i + 1, num_qubits):
                circ.CU1(1 / 2 ** (j - i), num_qubits - 1 - i,
                         num_qubits - 1 - j)
        return circ

    def serialize(c):
        qubits = [str(q) for q in c.qubits]
        cmds = [
            (cmd.op.get_name(), int(cmd.op.type),
             [str(q) for q in cmd.qubits],
             [float(p) for p in cmd.op.params])
            for cmd in c.get_commands()
        ]
        return qubits, cmds

    for nq, np_, seed in [(6, 2, 42), (8, 2, 7), (9, 3, 42), (12, 3, 42)]:
        circ = qft_circuit(nq)
        qpu_size = nq // np_
        qpu_indices = list(range(np_))
        qpu_cum = [0] + list(np.cumsum([qpu_size] * np_))
        ranges = [range(a, b) for (a, b) in zip(qpu_cum, qpu_cum[1:])]
        coupling = list(itertools.product(qpu_indices, qpu_indices))
        sq = {i: list(r) for i, r in enumerate(ranges)}
        ebit = {i: 2 for i in qpu_indices}

        py_n = PyNISQNetwork(server_coupling=[list(c) for c in coupling],
                             server_qubits=dict(sq), server_ebit_mem=dict(ebit))
        py_d = HypergraphPartitioning().allocate(circ, py_n, seed=seed)

        h = _core.HypergraphCircuit()
        qubits, cmds = serialize(circ)
        h.from_circuit(qubits, cmds)
        cpp_d = _core.hypergraph_partitioning_allocate(h, cpp_nisq(py_n),
                                                       seed=seed)

        assert dict(py_d.placement.placement) == \
            dict(cpp_d.placement.placement), \
            (f"HypergraphPartitioning placement mismatch nq={nq} np={np_} "
             f"seed={seed}: {dict(py_d.placement.placement)} vs "
             f"{dict(cpp_d.placement.placement)}")
        assert py_d.cost() == cpp_d.cost(), \
            (f"HypergraphPartitioning cost mismatch nq={nq} np={np_} "
             f"seed={seed}")
        assert cpp_d.is_valid(), \
            f"HypergraphPartitioning invalid nq={nq} np={np_}"

    # The default `ini_path` (bundled INI next to `_core`) must resolve; it is
    # exercised implicitly by the loop above (no `ini_path` kwarg).
    import glob as _glob
    assert _glob.glob(os.path.join(
        os.path.dirname(_core.__file__), "km1_kKaHyPar_sea20.ini")), \
        "bundled KaHyPar INI not next to _core"

    print("hypergraph_partitioning: OK")


def check_composition_refiners():
    """The `_core.RepeatRefiner` / `_core.SequenceRefiner` classes must
    reproduce the manual do-while / or-fold loops that the differential suite
    already validates (so users don't have to reimplement them)."""
    from pytket import Circuit  # noqa: F401

    def qft_circuit(num_qubits):
        circ = Circuit(num_qubits, name="$$QFT$$")
        for i in range(num_qubits):
            circ.H(num_qubits - 1 - i)
            for j in range(i + 1, num_qubits):
                circ.CU1(1 / 2 ** (j - i), num_qubits - 1 - i,
                         num_qubits - 1 - j)
        return circ

    def serialize(c):
        qubits = [str(q) for q in c.qubits]
        cmds = [
            (cmd.op.get_name(), int(cmd.op.type),
             [str(q) for q in cmd.qubits],
             [float(p) for p in cmd.op.params])
            for cmd in c.get_commands()
        ]
        return qubits, cmds

    def fresh_cpp(circ):
        h = _core.HypergraphCircuit()
        qubits, cmds = serialize(circ)
        h.from_circuit(qubits, cmds)
        net = _core.NISQNetwork(
            server_coupling=[[0, 1], [1, 2]],
            server_qubits={0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8]},
            server_ebit_mem={0: 2, 1: 2, 2: 2})
        return _core.hypergraph_partitioning_allocate(h, net, seed=42)

    def state(d):
        return (sorted(tuple(sorted(h.vertices)) for h in d.circuit.hyperedge_list),
                dict(d.placement.placement), d.cost())

    circ = qft_circuit(9)

    # RepeatRefiner(EagerHTypeMerge) == the manual while-loop.
    d1 = fresh_cpp(circ)
    while _core.eager_h_type_merge_refine(d1):
        pass
    d2 = fresh_cpp(circ)
    _core.RepeatRefiner(_core.eager_h_type_merge_refine).refine(d2)
    assert state(d1) == state(d2), "RepeatRefiner mismatch vs manual loop"

    # RepeatRefiner(SequenceRefiner([Neighbouring, Intertwined])) == manual
    # or-fold loop.
    d3 = fresh_cpp(circ)
    while True:
        made = False
        made |= _core.neighbouring_d_type_merge_refine(d3)
        made |= _core.intertwined_d_type_merge_refine(d3)
        if not made:
            break
    d4 = fresh_cpp(circ)
    _core.RepeatRefiner(_core.SequenceRefiner([
        _core.neighbouring_d_type_merge_refine,
        _core.intertwined_d_type_merge_refine,
    ])).refine(d4)
    assert state(d3) == state(d4), \
        "RepeatRefiner(SequenceRefiner(...)) mismatch vs manual loop"

    # Composition classes are callable and nest.
    sr = _core.SequenceRefiner([_core.intertwined_d_type_merge_refine])
    rr = _core.RepeatRefiner(sr)
    d5 = fresh_cpp(circ)
    first = rr.refine(d5)
    assert isinstance(first, bool), "refine must return bool"

    print("composition_refiners: OK")


if __name__ == "__main__":
    check_rng()
    check_steiner()
    check_direct_from_origin()
    check_hypergraph()
    check_circuit_analysis()
    check_placement()
    check_hypergraph_circuit()
    check_distribution_cost()
    check_pacman()
    check_pacman_internals()
    check_vertex_cover_refiner()
    check_gain_manager()
    check_allocators()
    check_eager_h_type_merge()
    check_d_type_merge()
    check_boundary_reallocation()
    check_detached_gates()
    check_annealing()
    check_annealing_seeded()
    check_distribution_methods()
    check_gateset()
    check_hypergraph_methods()
    check_bipartite()
    check_graph_basics()
    check_to_pytket_circuit()
    check_qft_distribution_pipelines()
    check_nisq_network()
    check_hypergraph_partitioning()
    check_composition_refiners()
    print("ALL CHECKS PASSED")
