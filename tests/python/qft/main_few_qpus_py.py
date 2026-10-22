import itertools
import logging
import random
import time
from pathlib import Path

import numpy as np
from pytket.circuit import Circuit as PytketCircuit
from pytket_dqc.allocators import Annealing, HypergraphPartitioning
from pytket_dqc.allocators import Random as RandomAllocator
from pytket_dqc.distributors import (Distributor,
                                     PartitioningHeterogeneousEmbedding)
from pytket_dqc.networks import NISQNetwork
from pytket_dqc.refiners import (DetachedGates, EagerHTypeMerge,
                                 IntertwinedDTypeMerge, NeighbouringDTypeMerge,
                                 RepeatRefiner, SequenceRefiner, VertexCover)
from pytket_dqc.utils import DQCPass, ebit_cost



SEED = 56426435

QFT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = QFT_DIR / "results_few_qpus_py"


def get_QFT_circuit_pytket(num_qubits):
    circ = PytketCircuit(num_qubits, name="$$QFT$$")
    for i in range(num_qubits):
        circ.H(num_qubits-1-i)
        for j in range(i + 1, num_qubits):
            circ.CU1(1 / 2 ** (j - i), num_qubits-1-i, num_qubits-1-j)
    return circ

def test_PYTKET_PE(circuit, qpu_sizes, num_partitions):
    qpu_indices = list(range(num_partitions))
    qpu_cum_sizes = [0] + list(np.cumsum(qpu_sizes))
    qpu_qubits_ranges = [range(start, stop) for start,stop in zip(qpu_cum_sizes, qpu_cum_sizes[1:])]

    network = NISQNetwork(
        server_coupling=list(itertools.product(qpu_indices, qpu_indices)), # fully connected
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
    distribution = HypergraphPartitioning().allocate(circuit, network, seed=SEED)
    refiner = RepeatRefiner(EagerHTypeMerge())
    refiner.refine(distribution)
    
    stop = time.time()
    duration = stop - start

    cost = distribution.cost()
    start = time.time()
    distributed_circuit = distribution.to_pytket_circuit(allow_update=True)
    cost = ebit_cost(distributed_circuit)
    stop = time.time()
    duration_fixing = stop - start
    detached_count = distribution.detached_gate_count()

    return cost, duration, duration_fixing, detached_count

def test_PYTKET_PHE(circuit, qpu_sizes, num_partitions):
    qpu_indices = list(range(num_partitions))
    qpu_cum_sizes = [0] + list(np.cumsum(qpu_sizes))
    qpu_qubits_ranges = [range(start, stop) for start,stop in zip(qpu_cum_sizes, qpu_cum_sizes[1:])]

    network = NISQNetwork(
        server_coupling=list(itertools.product(qpu_indices, qpu_indices)), # fully connected
        server_qubits={qpu_idx:list(qubits_range) for qpu_idx,qubits_range in enumerate(qpu_qubits_ranges)},
        server_ebit_mem={qpu_idx:2 for qpu_idx in qpu_indices}
    )

    start = time.time()
    """Equivalent to the workflow PartitionHeterogeneousEmbed 
    cited in [Andres-Matrinez et al. 2024] and recommended by Burt
    """

    distribution = PartitioningHeterogeneousEmbedding().distribute(circuit, network, seed=SEED)

    stop = time.time()
    duration = stop - start

    cost = distribution.cost()
    start = time.time()
    distributed_circuit = distribution.to_pytket_circuit(allow_update=True)
    cost = ebit_cost(distributed_circuit)
    stop = time.time()
    duration_fixing = stop - start
    detached_count = distribution.detached_gate_count()

    return cost, duration, duration_fixing, detached_count

def test_PYTKET_AESD(circuit, qpu_sizes, num_partitions):
    qpu_indices = list(range(num_partitions))
    qpu_cum_sizes = [0] + list(np.cumsum(qpu_sizes))
    qpu_qubits_ranges = [range(start, stop) for start,stop in zip(qpu_cum_sizes, qpu_cum_sizes[1:])]

    network = NISQNetwork(
        server_coupling=list(itertools.product(qpu_indices, qpu_indices)), # fully connected
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
    # PartitioningAnnealing
    # Redefine random allocator to have allocate be called with a seed, since Annealing().allocate calls its initial_place_method without passing its seed
    class SeededRandomAllocator(RandomAllocator):
        def __init__(self):
            self.random_allocator = RandomAllocator()
        def allocate(self, circ, network, **kwargs):
            return self.random_allocator.allocate(circ, network, seed=SEED, **kwargs)
    distribution = Annealing().allocate(circuit, network, seed=SEED, initial_place_method=SeededRandomAllocator())
    # CoverEmbedding
    VertexCover().refine(distribution)
    # Steiner
    refiner_list = [
        NeighbouringDTypeMerge(),
        IntertwinedDTypeMerge(),
    ]
    refiner = RepeatRefiner(SequenceRefiner(refiner_list))
    refiner.refine(distribution)
    # Detached
    DetachedGates().refine(distribution, seed=SEED)

    stop = time.time()
    duration = stop - start

    cost = distribution.cost()
    start = time.time()
    distributed_circuit = distribution.to_pytket_circuit(allow_update=True)
    cost = ebit_cost(distributed_circuit)
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
        get_QFT_circuit_pytket, 
        get_QFT_circuit_pytket,
        get_QFT_circuit_pytket, 
    ]

    for get_QFT_circuit, test_method, method_name, result_filename in zip(qft_methods, test_methods, method_names, result_filenames):
        print(f"Testing {method_name}")

        for num_partitions in [2,4]:
            print(f"Testing {num_partitions=}")

            output_string = ""

            for num_qubits in range(8, 12+1, 4):

                print(f"Testing {num_qubits=}")

                circuit = get_QFT_circuit(num_qubits)

                qpu_size = num_qubits // num_partitions
                qpu_sizes = [qpu_size] * num_partitions # Equal sized QPUs

                output_string += f"{qpu_size=}, {num_qubits=}, {num_partitions=}\n"

                output_string += f'Number of qubits in circuit {num_qubits}\n'
                best_score, time, time_fixing, detached_count = test_method(circuit, qpu_sizes, num_partitions)
                output_string += f"Min e-bit count: {best_score}\n"
                output_string += f"Time taken for {method_name}: {time} seconds\n"
                output_string += f"Time_fixing taken for {method_name}: {time_fixing} seconds\n"
                output_string += f"Detached count for {method_name}: {detached_count} gates\n\n"

            filepath = RESULTS_DIR / f"{num_partitions}_qpus" / result_filename
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with filepath.open('w') as f:
                print(output_string, file=f)

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.WARNING)
    main()