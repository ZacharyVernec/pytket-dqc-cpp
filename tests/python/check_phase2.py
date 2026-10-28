#!/usr/bin/env python3
"""Gate 2 check: the data and utility layer against the reference package.

Run from the repository root with the project virtualenv (the file matches the
``check_*.py`` pattern, so pytest never collects it):

    PYTHONPATH=python /home/ubuntu/pytket-dqc-cpp/.venv/bin/python \
        tests/python/check_phase2.py

Every section builds the same object with ``pytket_dqc_cpp`` and with the
reference ``pytket_dqc``, then compares the observable data.  ``pytket_dqc`` is
imported here on purpose; the adapter itself must never import it eagerly.
"""

import os
import subprocess
import sys
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "python"))

from pytket import Circuit, OpType  # noqa: E402
import pytket_dqc as ref  # noqa: E402
from pytket_dqc.circuits import Distribution as RefDistribution  # noqa: E402
from pytket_dqc.circuits import Hyperedge as RefHyperedge  # noqa: E402
from pytket_dqc.circuits.hypergraph_circuit import (  # noqa: E402
    CyclicHypergraphCircuit as RefCyclic,
    HypergraphCircuit as RefHypergraphCircuit,
    RegularGraphHypergraphCircuit as RefRegular,
)
from pytket_dqc.networks import AllToAll as RefAllToAll  # noqa: E402
from pytket_dqc.placement import Placement as RefPlacement  # noqa: E402
from pytket_dqc.utils import DQCPass as RefDQCPass  # noqa: E402
from pytket_dqc.utils.circuit_analysis import (  # noqa: E402
    ConstraintException as RefConstraintException,
)

import pytket_dqc_cpp  # noqa: E402
from pytket_dqc_cpp.circuits import (  # noqa: E402
    Distribution,
    HypergraphCircuit,
    RandomHypergraphCircuit,
    CyclicHypergraphCircuit,
    RegularGraphHypergraphCircuit,
)
from pytket_dqc_cpp.networks import AllToAll, NISQNetwork  # noqa: E402
from pytket_dqc_cpp.placement import Placement  # noqa: E402
from pytket_dqc_cpp.utils import DQCPass  # noqa: E402
from pytket_dqc_cpp.utils.circuit_analysis import (  # noqa: E402
    ConstraintException,
)
from pytket_dqc_cpp._delegation import reset_warnings  # noqa: E402

FAILURES = []
CHECKS = [0]


def check(name, condition, detail=""):
    CHECKS[0] += 1
    if not condition:
        FAILURES.append(f"{name}{': ' + detail if detail else ''}")
        print(f"FAIL {name} {detail}")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def dqc(circ):
    RefDQCPass().apply(circ)
    return circ


def plain_circuit():
    return dqc(Circuit(4).H(0).CU1(0.5, 0, 1).Rz(0.3, 1).H(2).CU1(0.7, 1, 2))


def h_embedding_circuit():
    """A CU1 framed by two Hadamards on the same qubit (requires H-embedding)."""
    return dqc(
        Circuit(3)
        .H(0)
        .CU1(1.0, 0, 1)
        .H(0)
        .CU1(0.5, 0, 2)
        .Rz(0.25, 2)
        .H(2)
        .CU1(0.75, 1, 2)
        .H(2)
    )


def layered_circuit():
    circ = Circuit(5)
    for layer in range(3):
        for q in range(4):
            circ.CZ(q, q + 1)
        circ.Rz(0.1 * (layer + 1), 0)
        circ.H(4)
    return dqc(circ)


CIRCUITS = {
    "plain": plain_circuit,
    "h_embedding": h_embedding_circuit,
    "layered": layered_circuit,
}


def closed_network():
    return RefAllToAll(3, 2), AllToAll(3, 2)


def line_network():
    return (
        ref.NISQNetwork([[0, 1], [1, 2]], {0: [0], 1: [1], 2: [2]}),
        NISQNetwork([[0, 1], [1, 2]], {0: [0], 1: [1], 2: [2]}),
    )


def placements(hc, ref_hc, network, ref_network, scheme):
    """Return ``(ref_placement, placement)`` for the given scheme."""
    if scheme == "home":
        mapping = {v: min(v, len(network.get_server_list()) - 1) for v in hc.vertex_list}
    elif scheme == "round_robin":
        servers = network.get_server_list()
        mapping = {v: servers[i % len(servers)] for i, v in enumerate(hc.vertex_list)}
    elif scheme == "gates_with_qubit":
        mapping = {}
        for v in hc.vertex_list:
            if hc.is_qubit_vertex(v):
                mapping[v] = min(v, len(network.get_server_list()) - 1)
        for hedge in hc.hyperedge_list:
            q = hc.get_qubit_vertex(hedge)
            for v in hc.get_gate_vertices(hedge):
                mapping[v] = mapping[q]
    else:  # pragma: no cover - defensive
        raise ValueError(scheme)
    return RefPlacement(dict(mapping)), Placement(dict(mapping))


def edge_rows(hypergraph):
    return [(list(h.vertices), h.weight) for h in hypergraph.hyperedge_list]


def edge_dict_rows(hypergraph):
    return {
        v: [(list(h.vertices), h.weight) for h in hedges]
        for v, hedges in hypergraph.hyperedge_dict.items()
    }


def outcome(fn):
    """Return either the call's value or the type of the exception it raises."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - parity of failure mode matters
        return (type(exc), str(exc))


def command_rows(commands):
    return [
        (c.op.get_name(), [str(q) for q in c.qubits], [round(p, 10) for p in c.op.params])
        for c in commands
    ]


def circuit_rows(circ):
    return [(c.op.get_name(), [str(q) for q in c.qubits], [round(p, 10) for p in c.op.params])
            for c in circ.get_commands()]


# ---------------------------------------------------------------------------
# 2.1 / 2.5 Hypergraph and HypergraphCircuit
# ---------------------------------------------------------------------------


def check_hypergraph(name):
    circuit = CIRCUITS[name]()
    ref_hc = RefHypergraphCircuit(circuit)
    hc = HypergraphCircuit(circuit)

    check(f"{name}: vertex_list", hc.vertex_list == ref_hc.vertex_list, str(hc.vertex_list))
    check(f"{name}: hyperedge_list", edge_rows(hc) == edge_rows(ref_hc))
    check(f"{name}: hyperedge_dict", edge_dict_rows(hc) == edge_dict_rows(ref_hc))
    check(
        f"{name}: vertex_neighbours",
        {v: sorted(n) for v, n in hc.vertex_neighbours.items()}
        == {v: sorted(n) for v, n in ref_hc.vertex_neighbours.items()},
    )
    check(f"{name}: is_valid", hc.is_valid() == ref_hc.is_valid())
    check(
        f"{name}: weight_one_predicate",
        hc.weight_one_predicate() == ref_hc.weight_one_predicate(),
    )
    check(f"{name}: kahypar_hyperedges", hc.kahypar_hyperedges() == ref_hc.kahypar_hyperedges())
    check(f"{name}: get_qubit_vertices", hc.get_qubit_vertices() == ref_hc.get_qubit_vertices())

    for hedge in ref_hc.hyperedge_list:
        adapter_hedge = hc.hyperedge_list[ref_hc.hyperedge_list.index(hedge)]
        check(
            f"{name}: get_qubit_vertex {hedge.vertices}",
            hc.get_qubit_vertex(adapter_hedge) == ref_hc.get_qubit_vertex(hedge),
        )
        check(
            f"{name}: get_gate_vertices {hedge.vertices}",
            hc.get_gate_vertices(adapter_hedge) == ref_hc.get_gate_vertices(hedge),
        )
        check(
            f"{name}: subcircuit {hedge.vertices}",
            command_rows(hc.get_hyperedge_subcircuit(adapter_hedge))
            == command_rows(ref_hc.get_hyperedge_subcircuit(hedge)),
        )
        check(
            f"{name}: requires_h_embedded_cu1 {hedge.vertices}",
            hc.requires_h_embedded_cu1(adapter_hedge)
            == ref_hc.requires_h_embedded_cu1(hedge),
        )
        check(
            f"{name}: get_h_embedded_gate_vertices {hedge.vertices}",
            outcome(lambda: hc.get_h_embedded_gate_vertices(adapter_hedge))
            == outcome(lambda: ref_hc.get_h_embedded_gate_vertices(hedge)),
        )

    check(
        f"{name}: get_all_h_embedded_gate_vertices",
        outcome(hc.get_all_h_embedded_gate_vertices)
        == outcome(ref_hc.get_all_h_embedded_gate_vertices),
    )
    check(
        f"{name}: get_vertex_to_command_index_map",
        hc.get_vertex_to_command_index_map() == ref_hc.get_vertex_to_command_index_map(),
    )
    for vertex in ref_hc.get_qubit_vertices():
        check(
            f"{name}: get_qubit_of_vertex {vertex}",
            hc.get_qubit_of_vertex(vertex) == ref_hc.get_qubit_of_vertex(vertex),
        )
        q = ref_hc.get_qubit_of_vertex(vertex)
        check(
            f"{name}: get_vertex_of_qubit {q}",
            hc.get_vertex_of_qubit(q) == ref_hc.get_vertex_of_qubit(q),
        )
    for vertex in ref_hc.vertex_list:
        if ref_hc.is_qubit_vertex(vertex):
            continue
        check(
            f"{name}: gate {vertex}",
            command_rows([hc.get_gate_of_vertex(vertex)])
            == command_rows([ref_hc.get_gate_of_vertex(vertex)]),
        )

    # Serialisation round trip
    check(
        f"{name}: to_dict",
        hc.to_dict() == ref_hc.to_dict(),
        str(hc.to_dict())[:200],
    )
    check(f"{name}: from_dict", HypergraphCircuit.from_dict(hc.to_dict()) == hc)
    check(f"{name}: get_circuit", circuit_rows(hc.get_circuit()) == circuit_rows(ref_hc.get_circuit()))
    check(f"{name}: __eq__", (hc == HypergraphCircuit(circuit)) and not (hc == 0))

    # Placement-dependent helpers
    ref_network, network = line_network()
    for scheme in ("home", "round_robin", "gates_with_qubit"):
        ref_pl, pl = placements(hc, ref_hc, network, ref_network, scheme)
        if not pl.is_valid(hc, network):
            continue
        check(
            f"{name}/{scheme}: is_placement",
            hc.is_placement(pl) == ref_hc.is_placement(ref_pl),
        )
        check(
            f"{name}/{scheme}: get_boundary",
            hc.get_boundary(pl) == ref_hc.get_boundary(ref_pl),
        )
        check(
            f"{name}/{scheme}: _get_server_to_qubit_vertex",
            hc._get_server_to_qubit_vertex(pl) == ref_hc._get_server_to_qubit_vertex(ref_pl),
        )

    # Merge / split behaviour
    multi = [h for h in ref_hc.hyperedge_list
             if len(h.vertices) > 2 and len(hc.get_hyperedges_containing(
                 [ref_hc.get_qubit_vertex(h)])) > 0]
    if multi:
        hedge = multi[0]
        adapter_hedge = hc.hyperedge_list[ref_hc.hyperedge_list.index(hedge)]
        q = ref_hc.get_qubit_vertex(hedge)
        gates = ref_hc.get_gate_vertices(hedge)
        split_points = [gates[: len(gates) // 2], gates[len(gates) // 2:]]
        if all(split_points):
            new = [RefHyperedge([q] + part) for part in split_points]
            hc.split_hyperedge(adapter_hedge, [type(adapter_hedge)(list(h.vertices), h.weight) for h in new])
            ref_hc.split_hyperedge(hedge, new)
            check(f"{name}: split_hyperedge", edge_rows(hc) == edge_rows(ref_hc))
            merged = hc.merge_hyperedge(
                [type(adapter_hedge)(list(h.vertices), h.weight) for h in new]
            )
            ref_merged = ref_hc.merge_hyperedge(new)
            check(
                f"{name}: merge_hyperedge",
                edge_rows(hc) == edge_rows(ref_hc)
                and list(merged.vertices) == list(ref_merged.vertices)
                and merged.weight == ref_merged.weight,
            )


def check_hypergraph_errors():
    circuit = plain_circuit()
    ref_hc = RefHypergraphCircuit(circuit)
    hc = HypergraphCircuit(circuit)

    bad_circuit = Circuit(2).Rx(0.5, 0).CX(0, 1)
    messages = []
    for cls in (RefHypergraphCircuit, HypergraphCircuit):
        try:
            cls(bad_circuit)
        except Exception as exc:  # noqa: BLE001 - message parity is the point
            messages.append(str(exc))
    check("construction error message", messages[0] == messages[1], str(messages))

    cases = [
        ("first vertex not qubit", lambda h: h.add_hyperedge([h.vertex_list[-1], 0])),
        ("two qubits", lambda h: h.add_hyperedge([0, 1])),
        ("decreasing", lambda h: h.add_hyperedge([1, 0])),
    ]
    for label, action in cases:
        messages = []
        for target in (ref_hc, hc):
            try:
                action(target)
            except Exception as exc:  # noqa: BLE001
                messages.append(str(exc))
            else:  # pragma: no cover - defensive
                messages.append("no error")
        check(f"add_hyperedge error: {label}", messages[0] == messages[1], str(messages))

    for label, action in (
        ("missing hyperedge", lambda h: h.merge_hyperedge([RefHyperedge([0, 999])])),
        ("bad split", lambda h: h.split_hyperedge(
            h.hyperedge_list[0], [RefHyperedge([h.hyperedge_list[0].vertices[0]])])),
    ):
        messages = []
        for target in (ref_hc, hc):
            try:
                action(target)
            except Exception as exc:  # noqa: BLE001
                messages.append(str(exc))
            else:  # pragma: no cover - defensive
                messages.append("no error")
        check(f"hyperedge error: {label}", messages[0] == messages[1], str(messages))


# ---------------------------------------------------------------------------
# 2.2 Placement
# ---------------------------------------------------------------------------


def check_placement(name):
    circuit = CIRCUITS[name]()
    ref_hc = RefHypergraphCircuit(circuit)
    hc = HypergraphCircuit(circuit)
    ref_network, network = closed_network()
    ref_pl, pl = placements(hc, ref_hc, network, ref_network, "round_robin")

    check(f"{name}: placement __eq__/str", str(pl) == str(ref_pl) and pl == Placement(dict(pl.placement)) and not (pl == 0))
    check(f"{name}: placement to_dict", pl.to_dict() == ref_pl.to_dict())
    check(f"{name}: placement from_dict", Placement.from_dict(pl.to_dict()) == pl)
    check(f"{name}: placement is_valid", pl.is_valid(hc, network) == ref_pl.is_valid(ref_hc, ref_network))
    for server in network.get_server_list():
        check(
            f"{name}: get_vertices_in({server})",
            pl.get_vertices_in(server) == ref_pl.get_vertices_in(server),
        )
    for hedge in hc.hyperedge_list:
        q = hc.get_qubit_vertex(hedge)
        check(
            f"{name}: get_distribution_tree {hedge.vertices}",
            pl.get_distribution_tree(hedge.vertices, q, network)
            == ref_pl.get_distribution_tree(
                list(hedge.vertices), q, ref_network
            ),
        )


# ---------------------------------------------------------------------------
# 2.6 Distribution
# ---------------------------------------------------------------------------


def check_distribution(name, scheme="round_robin"):
    circuit = CIRCUITS[name]()
    ref_hc = RefHypergraphCircuit(circuit)
    hc = HypergraphCircuit(circuit)
    ref_network, network = line_network()
    ref_pl, pl = placements(hc, ref_hc, network, ref_network, scheme)
    ref_dist = RefDistribution(ref_hc, ref_pl, ref_network)
    dist = Distribution(hc, pl, network)

    check(f"{name}/{scheme}: is_valid", dist.is_valid() == ref_dist.is_valid())
    if not dist.is_valid():
        return
    check(f"{name}/{scheme}: cost", dist.cost() == ref_dist.cost(), f"{dist.cost()} vs {ref_dist.cost()}")
    check(
        f"{name}/{scheme}: non_local_gate_list",
        dist.non_local_gate_list() == ref_dist.non_local_gate_list(),
    )
    check(
        f"{name}/{scheme}: detached_gate_list",
        dist.detached_gate_list() == ref_dist.detached_gate_list(),
    )
    check(
        f"{name}/{scheme}: counts",
        (dist.non_local_gate_count(), dist.detached_gate_count())
        == (ref_dist.non_local_gate_count(), ref_dist.detached_gate_count()),
    )
    check(
        f"{name}/{scheme}: get_qubit_mapping",
        {str(k): str(v) for k, v in dist.get_qubit_mapping().items()}
        == {str(k): str(v) for k, v in ref_dist.get_qubit_mapping().items()},
    )

    for hedge in hc.hyperedge_list:
        adapter_hedge = hedge
        ref_hedge = ref_hc.hyperedge_list[hc.hyperedge_list.index(hedge)]
        check(
            f"{name}/{scheme}: hyperedge_cost {hedge.vertices}",
            dist.hyperedge_cost(adapter_hedge) == ref_dist.hyperedge_cost(ref_hedge),
        )
        tree_ref = ref_pl.get_distribution_tree(
            list(ref_hedge.vertices), ref_hc.get_qubit_vertex(ref_hedge), ref_network
        )
        import networkx as nx  # local import: only needed here

        servers = [pl.placement[v] for v in hedge.vertices]
        tree = nx.Graph()
        tree.add_nodes_from(servers)
        tree.add_edges_from(tree_ref)
        check(
            f"{name}/{scheme}: hyperedge_cost server_tree {hedge.vertices}",
            dist.hyperedge_cost(adapter_hedge, server_tree=tree)
            == ref_dist.hyperedge_cost(ref_hedge, server_tree=tree),
        )
        check(
            f"{name}/{scheme}: hyperedge_cost requires_h kwarg {hedge.vertices}",
            dist.hyperedge_cost(adapter_hedge, requires_h_embedded_cu1=False)
            == ref_dist.hyperedge_cost(ref_hedge, requires_h_embedded_cu1=False),
        )

    # Serialisation
    check(f"{name}/{scheme}: to_dict", dist.to_dict() == ref_dist.to_dict())
    check(f"{name}/{scheme}: from_dict", Distribution.from_dict(dist.to_dict()) == dist)
    check(f"{name}/{scheme}: __eq__", (dist == Distribution(hc, pl, network)) and not (dist == 0))

    # Circuit generation
    ref_circ = ref_dist.to_pytket_circuit()
    circ = dist.to_pytket_circuit()
    check(f"{name}/{scheme}: to_pytket_circuit qubits", [str(q) for q in circ.qubits] == [str(q) for q in ref_circ.qubits])
    check(
        f"{name}/{scheme}: to_pytket_circuit commands",
        circuit_rows(circ) == circuit_rows(ref_circ),
        f"{circuit_rows(circ)[:3]} vs {circuit_rows(ref_circ)[:3]}",
    )


def check_constraint_and_update():
    circuit = Circuit(4).CU1(0.5, 0, 2).CU1(0.75, 1, 3).CU1(0.25, 0, 3)
    RefDQCPass().apply(circuit)
    ref_hc = RefHypergraphCircuit(circuit)
    hc = HypergraphCircuit(circuit)
    # Qubits 0 and 1 live in server 0; qubits 2 and 3 in server 1.  Every gate
    # is placed in server 1, so qubit 0's hyperedge holds a link qubit there
    # while qubit 1's nested hyperedge then needs a second one.
    mapping = {
        v: (
            1
            if not hc.is_qubit_vertex(v)
            else (0 if hc.get_qubit_of_vertex(v).index[0] in (0, 1) else 1)
        )
        for v in hc.vertex_list
    }
    ref_network = ref.NISQNetwork([[0, 1]], {0: [0, 1], 1: [2, 3]}, {0: 4, 1: 1})
    network = NISQNetwork([[0, 1]], {0: [0, 1], 1: [2, 3]}, {0: 4, 1: 1})
    reference_dist = RefDistribution(ref_hc, RefPlacement(dict(mapping)), ref_network)
    dist = Distribution(hc, Placement(dict(mapping)), network)

    ref_error = adapter_error = None
    try:
        reference_dist.to_pytket_circuit()
    except Exception as exc:  # noqa: BLE001
        ref_error = exc
    try:
        dist.to_pytket_circuit()
    except Exception as exc:  # noqa: BLE001
        adapter_error = exc

    check(
        "constraint: raised on both",
        ref_error is not None and adapter_error is not None,
        f"{ref_error!r} / {adapter_error!r}",
    )
    if ref_error is not None and adapter_error is not None:
        check(
            "constraint: adapter type",
            isinstance(adapter_error, ConstraintException),
            type(adapter_error).__name__,
        )
        check(
            "constraint: reference type",
            isinstance(ref_error, RefConstraintException),
        )
        check(
            "constraint: message prefix",
            str(adapter_error).startswith(
                "Communication memory capacity of server 1 exceeded."
            ),
            str(adapter_error),
        )
        # ConstraintException attribute parity: pybind11 translates
        # dqc::ConstraintException with .server, .v_gate, and the
        # "to_pytket_circuit(allow_update=True)" hint preserved.
        check(
            "constraint: adapter hint",
            "to_pytket_circuit(allow_update=True)" in str(adapter_error),
        )
        check(
            "constraint: reference hint",
            "to_pytket_circuit(allow_update=True)" in str(ref_error),
        )
        check("constraint: server", ref_error.server == adapter_error.server)
        check("constraint: v_gate", ref_error.v_gate == adapter_error.v_gate)

    # allow_update=True must split hyperedges identically on both sides.
    ref_updated = reference_dist.to_pytket_circuit(allow_update=True)
    updated = dist.to_pytket_circuit(allow_update=True)
    check(
        "allow_update: hyperedges",
        edge_rows(dist.circuit) == edge_rows(reference_dist.circuit),
        f"{edge_rows(dist.circuit)} vs {edge_rows(reference_dist.circuit)}",
    )
    check(
        "allow_update: placement",
        list(dist.placement.placement.items())
        == list(reference_dist.placement.placement.items()),
        f"{list(dist.placement.placement.items())} vs "
        f"{list(reference_dist.placement.placement.items())}",
    )
    check(
        "allow_update: circuit",
        circuit_rows(updated) == circuit_rows(ref_updated),
    )
    check("allow_update: cost", dist.cost() == reference_dist.cost())


# ---------------------------------------------------------------------------
# 2.5 Delegated circuit generators
# ---------------------------------------------------------------------------


def check_delegated_generators():
    reset_warnings()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cyclic = CyclicHypergraphCircuit(4, 2)
        cyclic_again = CyclicHypergraphCircuit(4, 2)
    ref_cyclic = RefCyclic(4, 2)
    check(
        "Cyclic delegation: circuit",
        circuit_rows(cyclic.get_circuit()) == circuit_rows(ref_cyclic.get_circuit()),
    )
    check(
        "Cyclic delegation: hyperedges",
        edge_rows(cyclic) == edge_rows(ref_cyclic),
    )
    check(
        "Cyclic delegation: single warning",
        len([w for w in caught if "CyclicHypergraphCircuit" in str(w.message)]) == 1,
        str([str(w.message) for w in caught]),
    )
    check("Cyclic delegation: warn once", cyclic_again == cyclic)

    import random

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        random.seed(4)
        regular = RegularGraphHypergraphCircuit(6, 3, 2, seed=5)
        random.seed(4)
        regular_again = RegularGraphHypergraphCircuit(6, 3, 2, seed=5)
    random.seed(4)
    ref_regular = RefRegular(6, 3, 2, seed=5)
    check(
        "Regular delegation: circuit",
        circuit_rows(regular.get_circuit()) == circuit_rows(ref_regular.get_circuit()),
    )
    check("Regular delegation: hyperedges", edge_rows(regular) == edge_rows(ref_regular))
    check(
        "Regular delegation: single warning",
        len([w for w in caught if "RegularGraphHypergraphCircuit" in str(w.message)]) == 1,
    )
    check("Regular delegation: deterministic", regular_again == regular)

    import numpy as np

    np.random.seed(11)
    random_circ = RandomHypergraphCircuit(3, 1)
    np.random.seed(11)
    ref_random_circ = ref.circuits.RandomHypergraphCircuit(3, 1)
    check(
        "Random delegation: circuit",
        circuit_rows(random_circ.get_circuit())
        == circuit_rows(ref_random_circ.get_circuit()),
    )
    check("Random delegation: hyperedges", edge_rows(random_circ) == edge_rows(ref_random_circ))


# ---------------------------------------------------------------------------
# Import hygiene
# ---------------------------------------------------------------------------


def check_import_hygiene():
    script = (
        "import sys;"
        "import pytket_dqc_cpp.circuits, pytket_dqc_cpp.placement,"
        "pytket_dqc_cpp.networks, pytket_dqc_cpp.utils;"
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
        "import hygiene: no reference imports",
        result.returncode == 0 and result.stdout.strip() == "[]",
        result.stdout.strip() + result.stderr.strip()[:200],
    )


def main():
    for name in CIRCUITS:
        check_hypergraph(name)
    check_hypergraph_errors()
    for name in CIRCUITS:
        check_placement(name)
    for name in CIRCUITS:
        check_distribution(name)
    check_distribution("h_embedding", scheme="gates_with_qubit")
    check_constraint_and_update()
    check_delegated_generators()
    check_import_hygiene()

    print()
    print(f"{CHECKS[0]} checks, {len(FAILURES)} failures")
    if FAILURES:
        for failure in FAILURES:
            print("  -", failure)
        return 1
    print("ALL PHASE 2 CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
