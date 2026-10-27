#!/usr/bin/env python3
"""Adapter equivalence check: ``pytket_dqc_cpp`` vs the reference ``pytket_dqc``.

This is the permanent replacement for the scratch differential harnesses used in
Phases 3 and 4.  It builds the same inputs twice -- once with the adapter and
once with the reference package -- and compares the observable results:

* ``HypergraphCircuit`` vertices / hyperedges / serialisation
* ``Distribution.cost``
* ``Random`` / ``Ordered`` / ``Brute`` placements
* seeded ``Annealing`` placement
* refiner booleans, hyperedge lists and placements
* ``PacMan`` packet dictionaries
* ``GainManager`` occupancy and cost maps
* ``steiner_tree`` node sets and edges *including* their attribute dicts
* ``to_pytket_circuit`` command names / qubits / parameters

The file matches the ``check_*.py`` pattern, so pytest never collects it.  Run it
from the repository root with the project virtualenv:

    PYTHONPATH=python /home/ubuntu/pytket-dqc-cpp/.venv/bin/python \
        tests/python/check_adapter_equivalence.py

Order-sensitive paths (set iteration, ``hyperedge_dict`` traversal, vertex-cover
cover selection) are compared under several ``PYTHONHASHSEED`` values: without an
explicit ``PYTHONHASHSEED`` the script re-executes itself once per seed in
``HASH_SEEDS``.
"""

import os
import random
import subprocess
import sys
import warnings

HASH_SEEDS = ["0", "1", "42", "12345"]

if os.environ.get("DQC_ADAPTER_HASH_SWEEP") != "1" and "PYTHONHASHSEED" not in os.environ:
    script = os.path.abspath(__file__)
    for _seed in HASH_SEEDS:
        _env = dict(os.environ)
        _env["PYTHONHASHSEED"] = _seed
        _env["DQC_ADAPTER_HASH_SWEEP"] = "1"
        subprocess.run([sys.executable, script], env=_env, check=True)
    print(f"OK: adapter differential hash-seed invariance across {HASH_SEEDS}")
    sys.exit(0)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "python"))

import pytket_dqc as ref  # noqa: E402
import pytket_dqc.allocators  # noqa: E402,F401 - exposed as attributes upstream
import pytket_dqc.packing  # noqa: E402,F401
import pytket_dqc.refiners  # noqa: E402,F401

import networkx as nx  # noqa: E402
import pytket_dqc_cpp as new  # noqa: E402
from pytket import Circuit, OpType  # noqa: E402
from pytket_dqc.utils import graph_tools as ref_graph_tools  # noqa: E402
from pytket_dqc_cpp._delegation import reset_warnings  # noqa: E402
from pytket_dqc_cpp.utils import graph_tools as new_graph_tools  # noqa: E402

# One ``UserWarning`` per delegated feature per process is expected; the
# differential does not exercise delegated generators, so keep output clean.
warnings.simplefilter("ignore", UserWarning)

FAILURES: list[str] = []
CHECKS = [0]


def progress(section):
    print(f"[seed={os.environ.get('PYTHONHASHSEED', 'random')}] {section}", file=sys.stderr, flush=True)


def check(name, condition, detail=""):
    CHECKS[0] += 1
    if not condition:
        FAILURES.append(f"{name}{': ' + detail if detail else ''}")
        print(f"FAIL {name} {detail}", flush=True)


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------


def hedge_key(hyperedge):
    return (tuple(hyperedge.vertices), hyperedge.weight)


def hedge_rows(hypergraph):
    return [hedge_key(h) for h in hypergraph.hyperedge_list]


def hedge_dict_rows(hypergraph):
    return {
        vertex: [hedge_key(h) for h in hedges]
        for vertex, hedges in hypergraph.hyperedge_dict.items()
    }


def packet_key(packet):
    return (
        packet.packet_index,
        packet.qubit_vertex,
        packet.connected_server_index,
        tuple(packet.gate_vertices),
        hedge_key(packet.parent_hedge),
    )


def packets_rows(packet_dict):
    return {
        vertex: [packet_key(packet) for packet in packets]
        for vertex, packets in packet_dict.items()
    }


def packet_groups(packet_dict):
    return {
        vertex: [tuple(packet_key(p) for p in group) for group in groups]
        for vertex, groups in packet_dict.items()
    }


def cost_map_rows(cost_map):
    return {hedge_key(hedge): cost for hedge, cost in cost_map.items()}


def command_rows(commands):
    return [
        (
            cmd.op.get_name(),
            tuple(str(q) for q in cmd.qubits),
            tuple(round(p, 10) for p in cmd.op.params),
        )
        for cmd in commands
    ]


def circuit_rows(circuit):
    return command_rows(circuit.get_commands())


def steiner_rows(tree):
    """Edges with their full attribute dicts, normalised for comparison."""
    return sorted(
        (min(u, v), max(u, v), tuple(sorted(data.items())))
        for u, v, data in tree.edges(data=True)
    )


def outcome(fn):
    """Return the value, or ``(exception type, message)`` if the call raised."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - failure-mode parity matters
        return ("raised", type(exc).__name__, str(exc))


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def make_intertwined_circuit():
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
    return circ


def make_random_circuit(n_qubits, depth, seed):
    rng = random.Random(seed)
    circ = Circuit(n_qubits)
    for _ in range(depth):
        kind = rng.randrange(3)
        if kind == 0:
            circ.H(rng.randrange(n_qubits))
        elif kind == 1:
            q0, q1 = rng.sample(range(n_qubits), 2)
            circ.add_gate(OpType.CU1, rng.choice([1.0, 0.5]), [q0, q1])
        else:
            q0, q1 = rng.sample(range(n_qubits), 2)
            circ.H(q1)
            circ.add_gate(OpType.CU1, 0.25, [q0, q1])
            circ.H(q1)
    return circ


def make_layered_circuit():
    circ = Circuit(5)
    for layer in range(3):
        for q in range(4):
            circ.CZ(q, q + 1)
        circ.Rz(0.1 * (layer + 1), 0)
        circ.H(4)
    return circ


CIRCUITS = {
    "intertwined": make_intertwined_circuit,
    "layered": make_layered_circuit,
    **{f"random{i}": (lambda i=i: make_random_circuit(5, 12, 100 + i)) for i in range(3)},
}


def network_blueprints():
    """Return ``[(coupling, server_qubits), ...]`` shared by both packages."""
    return [
        (
            [[0, 1], [1, 2], [1, 3]],
            {0: [0], 1: [1], 2: [2], 3: [3]},
        ),
        (
            [[0, 1], [1, 2], [2, 3], [3, 0]],
            {0: [0, 1], 1: [2], 2: [3], 3: [4]},
        ),
        (
            [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]],
            {0: [0, 1], 1: [2, 3], 2: [4], 3: [5], 4: [6]},
        ),
    ]


def make_network(package, coupling, server_qubits):
    return package.NISQNetwork(server_coupling=coupling, server_qubits=server_qubits)


def compile_circuit(circuit):
    """Run ``DQCPass`` so both packages see the same DQC gateset circuit."""
    ref.DQCPass().apply(circuit)
    return circuit


# ---------------------------------------------------------------------------
# 1. HypergraphCircuit
# ---------------------------------------------------------------------------


def check_hypergraph_circuit(name):
    circuit = compile_circuit(CIRCUITS[name]())
    ref_hc = ref.HypergraphCircuit(circuit)
    hc = new.HypergraphCircuit(circuit)

    check(f"{name}: vertex_list", hc.vertex_list == ref_hc.vertex_list)
    check(f"{name}: hyperedge_list", hedge_rows(hc) == hedge_rows(ref_hc))
    check(f"{name}: hyperedge_dict", hedge_dict_rows(hc) == hedge_dict_rows(ref_hc))
    check(f"{name}: get_qubit_vertices", hc.get_qubit_vertices() == ref_hc.get_qubit_vertices())
    check(f"{name}: to_dict", hc.to_dict() == ref_hc.to_dict())
    check(
        f"{name}: from_dict round trip",
        new.HypergraphCircuit.from_dict(hc.to_dict()).to_dict() == hc.to_dict(),
    )

    for ref_hedge, hedge in zip(ref_hc.hyperedge_list, hc.hyperedge_list):
        check(
            f"{name}: subcircuit {ref_hedge.vertices}",
            command_rows(hc.get_hyperedge_subcircuit(hedge))
            == command_rows(ref_hc.get_hyperedge_subcircuit(ref_hedge)),
        )
        check(
            f"{name}: requires_h_embedded_cu1 {ref_hedge.vertices}",
            hc.requires_h_embedded_cu1(hedge) == ref_hc.requires_h_embedded_cu1(ref_hedge),
        )


# ---------------------------------------------------------------------------
# 2/3/4. Allocation and cost
# ---------------------------------------------------------------------------


def allocation_cases():
    """Yield ``(tag, ref_dist, new_dist)`` for each allocator/network case."""
    for name, maker in CIRCUITS.items():
        for index, (coupling, server_qubits) in enumerate(network_blueprints()):
            circuit = compile_circuit(maker())
            ref_network = make_network(ref, coupling, server_qubits)
            new_network = make_network(new, coupling, server_qubits)
            tag = f"{name}/net{index}"

            try:
                ref_dist = ref.allocators.Ordered().allocate(circuit, ref_network)
            except Exception as exc:  # noqa: BLE001 - network too small for circuit
                print(f"skip {tag}: {type(exc).__name__}: {exc}", flush=True)
                continue
            new_dist = new.allocators.Ordered().allocate(circuit, new_network)

            check(
                f"{tag}: Ordered placement",
                ref_dist.placement.placement == new_dist.placement.placement,
            )
            check(f"{tag}: cost", ref_dist.cost() == new_dist.cost())
            yield tag, circuit, ref_network, new_network, ref_dist, new_dist


def check_random_allocator(tag, circuit, ref_network, new_network):
    if len(circuit.qubits) > 5:
        return
    for seed in (0, 7):
        ref_dist = outcome(lambda: ref.allocators.Random().allocate(circuit, ref_network, seed=seed))
        new_dist = outcome(lambda: new.allocators.Random().allocate(circuit, new_network, seed=seed))
        if isinstance(ref_dist, tuple) or isinstance(new_dist, tuple):
            check(
                f"{tag}: Random(seed={seed}) outcome",
                ref_dist == new_dist,
                f"ref={ref_dist} new={new_dist}",
            )
            continue
        check(
            f"{tag}: Random(seed={seed}) placement",
            ref_dist.placement.placement == new_dist.placement.placement,
        )
        check(f"{tag}: Random(seed={seed}) cost", ref_dist.cost() == new_dist.cost())


def check_brute_allocator(case, circuit, coupling, server_qubits):
    """Brute force is exponential: only tiny dedicated inputs are used."""
    ref_network = make_network(ref, coupling, server_qubits)
    new_network = make_network(new, coupling, server_qubits)

    ref_dist = outcome(lambda: ref.allocators.Brute().allocate(circuit, ref_network))
    new_dist = outcome(lambda: new.allocators.Brute().allocate(circuit, new_network))
    if isinstance(ref_dist, tuple) or isinstance(new_dist, tuple):
        check(f"{case}: Brute outcome", ref_dist == new_dist, f"ref={ref_dist} new={new_dist}")
        return
    check(f"{case}: Brute placement", ref_dist.placement.placement == new_dist.placement.placement)
    check(f"{case}: Brute cost", ref_dist.cost() == new_dist.cost())


def check_annealing_allocator(tag, circuit, ref_network, new_network):
    if len(circuit.qubits) > 5:
        return
    kwargs = dict(
        seed=11,
        iterations=200,
        initial_place_method=None,  # filled per package below
        cache_limit=2,
    )
    ref_dist = ref.allocators.Annealing().allocate(
        circuit, ref_network, initial_place_method=ref.allocators.Ordered(), **{
            k: v for k, v in kwargs.items() if k != "initial_place_method"
        }
    )
    new_dist = new.allocators.Annealing().allocate(
        circuit, new_network, initial_place_method=new.allocators.Ordered(), **{
            k: v for k, v in kwargs.items() if k != "initial_place_method"
        }
    )
    check(
        f"{tag}: Annealing placement",
        ref_dist.placement.placement == new_dist.placement.placement,
    )
    check(f"{tag}: Annealing cost", ref_dist.cost() == new_dist.cost())


def check_to_pytket_circuit(tag, ref_dist, new_dist):
    ref_circ = outcome(ref_dist.to_pytket_circuit)
    new_circ = outcome(new_dist.to_pytket_circuit)
    if isinstance(ref_circ, tuple) or isinstance(new_circ, tuple):
        check(f"{tag}: to_pytket_circuit outcome", ref_circ == new_circ, f"{ref_circ} / {new_circ}")
        return
    check(
        f"{tag}: to_pytket_circuit commands",
        circuit_rows(ref_circ) == circuit_rows(new_circ),
    )
    check(
        f"{tag}: to_pytket_circuit qubit_mapping",
        ref_dist.get_qubit_mapping() == new_dist.get_qubit_mapping(),
    )


# ---------------------------------------------------------------------------
# 5. Refiners
# ---------------------------------------------------------------------------


def refiner_cases(package):
    return [
        ("EagerHTypeMerge", lambda: package.refiners.EagerHTypeMerge(), {}),
        ("NeighbouringDTypeMerge", lambda: package.refiners.NeighbouringDTypeMerge(), {}),
        ("IntertwinedDTypeMerge", lambda: package.refiners.IntertwinedDTypeMerge(), {}),
        ("BoundaryReallocation", lambda: package.refiners.BoundaryReallocation(), {"seed": 7}),
        (
            "BoundaryReallocationR3",
            lambda: package.refiners.BoundaryReallocation(),
            {"seed": 3, "num_rounds": 3, "cache_limit": 2},
        ),
        ("DetachedGates", lambda: package.refiners.DetachedGates(), {"seed": 5}),
        (
            "RepeatSequence",
            lambda: package.refiners.RepeatRefiner(
                package.refiners.SequenceRefiner(
                    [
                        package.refiners.NeighbouringDTypeMerge(),
                        package.refiners.IntertwinedDTypeMerge(),
                    ]
                )
            ),
            {},
        ),
    ]


def check_refiners(tag, ref_dist, new_dist):
    for (ref_name, ref_factory, kwargs), (new_name, new_factory, _) in zip(
        refiner_cases(ref), refiner_cases(new)
    ):
        check(f"{tag}: refiner name {ref_name}", ref_name == new_name)
        ref_clone = ref.Distribution.from_dict(ref_dist.to_dict())
        new_clone = new.Distribution.from_dict(new_dist.to_dict())

        ref_result = outcome(lambda: ref_factory().refine(ref_clone, **kwargs))
        new_result = outcome(lambda: new_factory().refine(new_clone, **kwargs))
        check(
            f"{tag}/{ref_name}: refine outcome",
            ref_result == new_result,
            f"ref={ref_result} new={new_result}",
        )
        if isinstance(ref_result, tuple) or isinstance(new_result, tuple):
            continue
        check(
            f"{tag}/{ref_name}: hyperedges",
            hedge_rows(ref_clone.circuit) == hedge_rows(new_clone.circuit),
        )
        check(
            f"{tag}/{ref_name}: placement",
            ref_clone.placement.placement == new_clone.placement.placement,
        )
        check(f"{tag}/{ref_name}: cost", ref_clone.cost() == new_clone.cost())

    # VertexCover runs in Python on both sides (networkx cover selection only).
    if len(new_dist.circuit.vertex_list) <= 40:
        ref_clone = ref.Distribution.from_dict(ref_dist.to_dict())
        new_clone = new.Distribution.from_dict(new_dist.to_dict())
        ref_result = outcome(lambda: ref.refiners.VertexCover().refine(ref_clone))
        new_result = outcome(lambda: new.refiners.VertexCover().refine(new_clone))
        check(
            f"{tag}/VertexCover: refine outcome",
            ref_result == new_result,
            f"ref={ref_result} new={new_result}",
        )
        if not (isinstance(ref_result, tuple) or isinstance(new_result, tuple)):
            check(
                f"{tag}/VertexCover: hyperedges",
                hedge_rows(ref_clone.circuit) == hedge_rows(new_clone.circuit),
            )
            check(
                f"{tag}/VertexCover: placement",
                ref_clone.placement.placement == new_clone.placement.placement,
            )


# ---------------------------------------------------------------------------
# 6. PacMan
# ---------------------------------------------------------------------------


def check_pacman(tag, circuit, ref_network, new_network, placement_dict):
    ref_pacman = ref.packing.PacMan(
        ref.HypergraphCircuit(circuit), ref.placement.Placement(dict(placement_dict))
    )
    new_pacman = new.packing.PacMan(
        new.HypergraphCircuit(circuit), new.placement.Placement(dict(placement_dict))
    )

    check(
        f"{tag}: packets_by_qubit",
        packets_rows(ref_pacman.packets_by_qubit) == packets_rows(new_pacman.packets_by_qubit),
    )
    check(
        f"{tag}: neighbouring_packets",
        packet_groups(ref_pacman.neighbouring_packets)
        == packet_groups(new_pacman.neighbouring_packets),
    )
    check(
        f"{tag}: hopping_packets",
        packet_groups(ref_pacman.hopping_packets) == packet_groups(new_pacman.hopping_packets),
    )
    check(
        f"{tag}: merged_packets",
        packet_groups(ref_pacman.merged_packets) == packet_groups(new_pacman.merged_packets),
    )


# ---------------------------------------------------------------------------
# 7. GainManager
# ---------------------------------------------------------------------------


def check_gain_manager(tag, ref_dist, new_dist):
    ref_manager = ref.allocators.GainManager(ref.Distribution.from_dict(ref_dist.to_dict()))
    new_manager = new.allocators.GainManager(new.Distribution.from_dict(new_dist.to_dict()))

    check(f"{tag}: GainManager occupancy", ref_manager.occupancy == new_manager.occupancy)
    check(
        f"{tag}: GainManager hyperedge_cost_map",
        cost_map_rows(ref_manager.hyperedge_cost_map)
        == cost_map_rows(new_manager.hyperedge_cost_map),
    )
    check(
        f"{tag}: GainManager requires_h_embedded_cu1",
        cost_map_rows(ref_manager.requires_h_embedded_cu1)
        == cost_map_rows(new_manager.requires_h_embedded_cu1),
    )

    # Exercise a move on each side and compare the gain and resulting state.
    for vertex in ref_dist.circuit.get_qubit_vertices():
        target = None
        for server in ref_dist.network.get_server_list():
            if server != ref_dist.placement.placement[vertex]:
                target = server
                break
        if target is None:
            continue
        ref_gain = outcome(lambda: ref_manager.move_vertex_gain(vertex, target))
        new_gain = outcome(lambda: new_manager.move_vertex_gain(vertex, target))
        check(
            f"{tag}: move_vertex_gain q{vertex}->s{target}",
            ref_gain == new_gain,
            f"ref={ref_gain} new={new_gain}",
        )
        break


# ---------------------------------------------------------------------------
# 8. Graph tools
# ---------------------------------------------------------------------------


def _graph_blueprints():
    """Return ``[(name, graph_factory, terminals), ...]``.

    Each factory must build a fresh graph, because the reference returns a
    *view* of its input: sharing one graph would hide attribute differences.
    """

    def plain():
        graph = nx.Graph()
        graph.add_nodes_from(range(5))
        for u, v in [(0, 1), (1, 2), (2, 3), (3, 4), (1, 3)]:
            graph.add_edge(u, v)
        return graph, [1, 4]

    def attributed():
        graph = nx.Graph()
        graph.add_nodes_from(range(4))
        graph.add_edge(0, 1, weight=1.0, latency=5)
        graph.add_edge(1, 2, weight=2.0, latency=7)
        graph.add_edge(2, 3, weight=1.0, latency=9)
        graph.add_edge(1, 3, weight=4.0, latency=1)
        return graph, [0, 3]

    def tie_break():
        # Zero-weight edges make the choice between equal-cost trees depend on
        # neighbour/episode order, which the adapter must replay faithfully.
        graph = nx.Graph()
        graph.add_nodes_from(range(5))
        graph.add_edge(0, 1, weight=0.0)
        graph.add_edge(0, 2, weight=0.0)
        graph.add_edge(0, 4, weight=1.0)
        graph.add_edge(0, 3, weight=1.0)
        graph.add_edge(1, 2, weight=0.0)
        graph.add_edge(2, 3, weight=1.0)
        graph.add_edge(2, 4, weight=1.0)
        return graph, [1, 2, 3]

    return [("plain", plain), ("attributed", attributed), ("tie_break", tie_break)]


def check_graph_tools():
    """``steiner_tree`` must reproduce the reference's subgraph *view*.

    Upstream returns a subgraph view of the input, so every edge keeps its
    original attribute dict -- which may be empty or hold keys the core does
    not know about.  The adapter rebuilds the tree from the core, whose graph
    always carries ``weight``; this section guards the step that restores the
    original edge data.  Costs and placements are insensitive to it, so
    without this check a regression here is invisible.
    """
    for name, factory in _graph_blueprints():
        ref_graph, terminals = factory()
        new_graph, _ = factory()
        ref_tree = ref_graph_tools.steiner_tree(ref_graph, terminals)
        new_tree = new_graph_tools.steiner_tree(new_graph, terminals)

        check(
            f"steiner_tree/{name}: nodes",
            sorted(ref_tree.nodes) == sorted(new_tree.nodes),
            f"ref={sorted(ref_tree.nodes)} new={sorted(new_tree.nodes)}",
        )
        check(
            f"steiner_tree/{name}: edges(data=True)",
            steiner_rows(ref_tree) == steiner_rows(new_tree),
            f"ref={steiner_rows(ref_tree)} new={steiner_rows(new_tree)}",
        )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def check_import_hygiene():
    """``pytket_dqc`` must not be imported eagerly by the adapter package."""
    script = (
        "import sys;"
        "import pytket_dqc_cpp;"
        "bad=[m for m in sys.modules if m=='pytket_dqc' or m.startswith('pytket_dqc.')];"
        "print(bad)"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": os.path.join(ROOT, "python")},
    )
    check(
        "import hygiene: no eager reference imports",
        result.returncode == 0 and result.stdout.strip() == "[]",
        result.stdout.strip() + result.stderr.strip()[:200],
    )


def main():
    reset_warnings()
    progress("hypergraph circuits")
    for name in CIRCUITS:
        check_hypergraph_circuit(name)

    progress("brute force allocator")
    check_brute_allocator(
        "brute/small",
        Circuit(2).add_gate(OpType.CU1, 1.0, [0, 1]),
        [[0, 1]],
        {0: [0, 1], 1: [2]},
    )
    check_brute_allocator(
        "brute/med",
        Circuit(4)
        .add_gate(OpType.CU1, 1.0, [0, 1])
        .add_gate(OpType.CU1, 1.0, [1, 2])
        .add_gate(OpType.CU1, 1.0, [2, 3]),
        [[0, 1], [0, 2]],
        {0: [0], 1: [1], 2: [2, 3]},
    )

    for tag, circuit, _ref_network, _new_network, ref_dist, new_dist in allocation_cases():
        progress(f"{tag}")
        check_random_allocator(tag, circuit, _ref_network, _new_network)
        check_annealing_allocator(tag, circuit, _ref_network, _new_network)
        check_to_pytket_circuit(tag, ref_dist, new_dist)
        check_refiners(tag, ref_dist, new_dist)
        check_gain_manager(tag, ref_dist, new_dist)
        check_pacman(tag, circuit, _ref_network, _new_network, new_dist.placement.placement)

    progress("graph tools")
    check_graph_tools()

    progress("import hygiene")
    check_import_hygiene()

    print()
    print(f"{CHECKS[0]} checks, {len(FAILURES)} failures")
    if FAILURES:
        for failure in FAILURES:
            print("  -", failure)
        return 1
    print("ALL ADAPTER EQUIVALENCE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
