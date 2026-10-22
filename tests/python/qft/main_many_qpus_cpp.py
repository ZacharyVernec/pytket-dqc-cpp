"""C++ (``_core``) variant of ``main_many_qpus_py.py``: the Python ``pytket_dqc``
compute is replaced by the C++ core (the ``_core`` pybind11 module built by
CMake into ``build/dqc_cpp``).  ``pytket`` is only used to build and serialize
the QFT circuit (the core has no circuit builder and depends on pytket's exact
``get_commands()`` ordering); the DQC pipeline itself runs in ``_core``.

The ``_core`` mirror of the ``pytket_dqc`` API includes ``NISQNetwork``, the
``RepeatRefiner``/``SequenceRefiner`` composition classes, and the
KaHyPar-backed ``HypergraphPartitioning`` allocator
(``_core.hypergraph_partitioning_allocate``), so the PE / PHE workflows here
use the real KaHyPar initial partition.

Results are written to ``tests/python/qft/results_many_qpus_cpp/`` so they can
be diffed against the reference ``tests/python/qft/results_many_qpus_py/`` (see
``tests/python/check_qft_equivalence.py``).

Run: PYTHONHASHSEED=42 .venv/bin/python3 tests/python/qft/main_many_qpus_cpp.py
"""

import itertools
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
from pytket.circuit import Circuit as PytketCircuit

# <repo>/tests/python/qft/<this file> -> <repo>
QFT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = QFT_DIR / "results_many_qpus_cpp"
ROOT = QFT_DIR.parents[2]
sys.path.insert(0, str(ROOT / "build" / "dqc_cpp"))
from _core import (  # noqa: E402
    NISQNetwork,
    HypergraphCircuit,
    RepeatRefiner,
    SequenceRefiner,
    eager_h_type_merge_refine,
    boundary_reallocation_refine,
    vertex_cover_refine,
    neighbouring_d_type_merge_refine,
    intertwined_d_type_merge_refine,
    detached_gates_refine,
    annealing_allocate,
    hypergraph_partitioning_allocate,
    ebit_cost,
)


SEED = 56426435


def serialize(circ):
    qubits = [str(q) for q in circ.qubits]
    cmds = [
        (cmd.op.get_name(), int(cmd.op.type), [str(q) for q in cmd.qubits],
         [float(p) for p in cmd.op.params])
        for cmd in circ.get_commands()
    ]
    return qubits, cmds


def get_QFT_circuit_core(num_qubits):
    """Build the QFT as a ``pytket`` circuit and return the ``_core``
    ``HypergraphCircuit`` the DQC pipeline consumes (the core has no circuit
    builder). Mirrors ``get_QFT_circuit_pytket`` + ``HypergraphCircuit(circ)``."""
    circ = PytketCircuit(num_qubits, name="$$QFT$$")
    for i in range(num_qubits):
        circ.H(num_qubits-1-i)
        for j in range(i + 1, num_qubits):
            circ.CU1(1 / 2 ** (j - i), num_qubits-1-i, num_qubits-1-j)
    dist_circ = HypergraphCircuit()
    qubits, cmds = serialize(circ)
    dist_circ.from_circuit(qubits, cmds)
    return dist_circ


def test_PYTKET_PE(dist_circ, qpu_sizes, num_partitions):
    qpu_indices = list(range(num_partitions))
    qpu_cum_sizes = [0] + list(np.cumsum(qpu_sizes))
    qpu_qubits_ranges = [range(start, stop) for start,stop in zip(qpu_cum_sizes, qpu_cum_sizes[1:])]

    network = NISQNetwork(
        server_coupling=[list(p) for p in itertools.product(qpu_indices, qpu_indices)], # fully connected
        server_qubits={qpu_idx:list(qubits_range) for qpu_idx,qubits_range in enumerate(qpu_qubits_ranges)},
        server_ebit_mem={qpu_idx:2 for qpu_idx in qpu_indices}
    )

    start = time.time()
    """Equivalent to the workflow PartitionEmbed 
    cited in [Andres-Matrinez et al. 2024] and [Burt et al. 2025]

    Implemented by merging the pytket-dqc distributors
    - PartitioningHeterogeneous, which includes the HypergraphPartitioning
    - PartitioningHeterogeneousEmbedding, that includes embedding
    but removing the boundary reallocation of the latter (i.e. removing the 'Heterogeneous' part).
    """
    # HypergraphPartitioning (KaHyPar) + make_valid, in C++. Mirrors
    # `HypergraphPartitioning().allocate(circuit, network, seed=SEED)`.
    distribution = hypergraph_partitioning_allocate(dist_circ, network, seed=SEED)
    refiner = RepeatRefiner(eager_h_type_merge_refine)
    refiner.refine(distribution)

    stop = time.time()
    duration = stop - start

    cost = distribution.cost()
    start = time.time()
    distributed_circuit = distribution.to_pytket_circuit(allow_update=True)
    qubits, cmds = serialize(distributed_circuit)
    cost = ebit_cost(qubits, cmds)
    stop = time.time()
    duration_fixing = stop - start
    detached_count = distribution.detached_gate_count()

    return cost, duration, duration_fixing, detached_count


def test_PYTKET_PHE(dist_circ, qpu_sizes, num_partitions):
    qpu_indices = list(range(num_partitions))
    qpu_cum_sizes = [0] + list(np.cumsum(qpu_sizes))
    qpu_qubits_ranges = [range(start, stop) for start,stop in zip(qpu_cum_sizes, qpu_cum_sizes[1:])]

    network = NISQNetwork(
        server_coupling=[list(p) for p in itertools.product(qpu_indices, qpu_indices)], # fully connected
        server_qubits={qpu_idx:list(qubits_range) for qpu_idx,qubits_range in enumerate(qpu_qubits_ranges)},
        server_ebit_mem={qpu_idx:2 for qpu_idx in qpu_indices}
    )

    start = time.time()
    """Equivalent to the workflow PartitionHeterogeneousEmbed 
    cited in [Andres-Matrinez et al. 2024] and recommended by Burt
    """
    # HypergraphPartitioning (KaHyPar) + make_valid, in C++.
    distribution = hypergraph_partitioning_allocate(dist_circ, network, seed=SEED)
    boundary_reallocation_refine(distribution, seed=SEED)
    refiner = RepeatRefiner(eager_h_type_merge_refine)
    refiner.refine(distribution)

    stop = time.time()
    duration = stop - start

    cost = distribution.cost()
    start = time.time()
    distributed_circuit = distribution.to_pytket_circuit(allow_update=True)
    qubits, cmds = serialize(distributed_circuit)
    cost = ebit_cost(qubits, cmds)
    stop = time.time()
    duration_fixing = stop - start
    detached_count = distribution.detached_gate_count()

    return cost, duration, duration_fixing, detached_count


def test_PYTKET_AESD(dist_circ, qpu_sizes, num_partitions):
    qpu_indices = list(range(num_partitions))
    qpu_cum_sizes = [0] + list(np.cumsum(qpu_sizes))
    qpu_qubits_ranges = [range(start, stop) for start,stop in zip(qpu_cum_sizes, qpu_cum_sizes[1:])]

    network = NISQNetwork(
        server_coupling=[list(p) for p in itertools.product(qpu_indices, qpu_indices)], # fully connected
        server_qubits={qpu_idx:list(qubits_range) for qpu_idx,qubits_range in enumerate(qpu_qubits_ranges)},
        server_ebit_mem={qpu_idx:2 for qpu_idx in qpu_indices}
    )

    start = time.time()
    """Equivalent to the workflow EmbedSteinerDetach 
    cited in [Andres-Matrinez et al. 2024] 
    but with Annealer instead of KaHyPar, as in [Burt et al. 2024]

    Implemented by merging the pytket-dqc distributors
    - PartitioningAnnealing, which includes the annealing
    - CoverEmbedding, which includes the embedding
    - CoverEmbeddingSteiner, which includes the Steiner tree merging
    - CoverEmbeddingSteinerDetached, which includes the reallocation using detached gates
    """
    # PartitioningAnnealing: mirrors
    # `Annealing().allocate(circuit, network, seed=SEED,
    #                       initial_place_method=SeededRandomAllocator())`.
    distribution = annealing_allocate(dist_circ, network, seed=SEED,
                                      initial_place_seed=SEED)
    # CoverEmbedding
    # VertexCover rebuilds circuit + placement; the binding returns fresh
    # copies, so assign them back onto the distribution.
    new_h, new_p = vertex_cover_refine(distribution.circuit,
                                       distribution.placement,
                                       network.get_server_nx())
    distribution.circuit = new_h
    distribution.placement = new_p
    # Steiner
    refiner_list = [
        neighbouring_d_type_merge_refine,
        intertwined_d_type_merge_refine,
    ]
    refiner = RepeatRefiner(SequenceRefiner(refiner_list))
    refiner.refine(distribution)
    # Detached
    detached_gates_refine(distribution, seed=SEED)

    stop = time.time()
    duration = stop - start

    cost = distribution.cost()
    start = time.time()
    distributed_circuit = distribution.to_pytket_circuit(allow_update=True)
    qubits, cmds = serialize(distributed_circuit)
    cost = ebit_cost(qubits, cmds)
    stop = time.time()
    duration_fixing = stop - start
    detached_count = distribution.detached_gate_count()

    return cost, duration, duration_fixing, detached_count


def main():
    np.random.seed(SEED)
    random.seed(SEED)

    result_filenames = [
        "results-data-pytket-pe.txt",
        "results-data-pytket-aesd.txt",
        "results-data-pytket-phe.txt",
    ]
    method_names = [
        "PYTKET_PE",
        "PYTKET_AESD",
        "PYTKET_PHE",
    ]
    test_methods = [
        test_PYTKET_PE,
        test_PYTKET_AESD,
        test_PYTKET_PHE,
    ]
    qft_methods = [
        get_QFT_circuit_core,
        get_QFT_circuit_core,
        get_QFT_circuit_core,
    ]

    for get_QFT_circuit, test_method, method_name, result_filename in zip(
        qft_methods, test_methods, method_names, result_filenames
    ):
        print(f"Testing {method_name}")

        qpu_size = 3

        output_string = ""

        for num_partitions in range(3, 4+1):
            num_qubits = num_partitions * qpu_size

            print(f"Testing {num_partitions=}")

            circuit = get_QFT_circuit(num_qubits)

            qpu_sizes = [qpu_size] * num_partitions  # Equal sized QPUs

            output_string += f"{qpu_size=}, {num_qubits=}, {num_partitions=}\n"

            output_string += f'Number of qubits in circuit {num_qubits}\n'
            best_score, time, time_fixing, detached_count = test_method(
                circuit, qpu_sizes, num_partitions
            )
            output_string += f"Min e-bit count: {best_score}\n"
            output_string += f"Time taken for {method_name}: {time} seconds\n"
            output_string += f"Time_fixing taken for {method_name}: {time_fixing} seconds\n"
            output_string += f"Detached count for {method_name}: {detached_count} gates\n\n"

        filepath = RESULTS_DIR / result_filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with filepath.open('w') as f:
            print(output_string, file=f)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.WARNING)
    main()
