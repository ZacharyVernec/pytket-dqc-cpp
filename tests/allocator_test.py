from pytket_dqc_cpp.allocators import (
    Annealing,
    Random,
    Ordered,
    Routing,
    HypergraphPartitioning,
    Brute,
)
import pytket_dqc_cpp
from pytket_dqc_cpp import HypergraphCircuit, Distribution
from pytket import Circuit
from pytket_dqc_cpp.networks import NISQNetwork
from pytket_dqc_cpp.allocators.ordered import order_reducing_size
from pytket_dqc_cpp.placement import Placement
import kahypar as kahypar  # type:ignore
from pytket.circuit import QControlBox, Op, OpType
import pytest
import json


# TODO: Test that the placement returned by routing does not
# have any cost two edges


def get_H_ladder_circ():
    circ = (
        Circuit(4)
        .add_gate(OpType.CU1, 1.0, [0, 3])
        .H(3)
        .add_gate(OpType.CU1, 1.0, [1, 3])
        .H(3)
        .add_gate(OpType.CU1, 1.0, [2, 3])
        .H(3)
    )

    return circ


def get_Rz_ladder_circ():
    circ = (
        Circuit(4)
        .add_gate(OpType.CU1, 1.0, [0, 3])
        .Rz(0.5, 3)
        .add_gate(OpType.CU1, 1.0, [1, 3])
        .Rz(0.5, 3)
        .add_gate(OpType.CU1, 1.0, [2, 3])
        .Rz(0.5, 3)
    )

    return circ


def get_line_network():
    return NISQNetwork([[0, 1], [0, 2]], {0: [0], 1: [1, 2], 2: [3, 4]})


def test_annealing_distribute():
    network = get_line_network()

    circ = get_H_ladder_circ()
    allocator = Annealing()

    distribution = allocator.allocate(
        circ,
        network,
        seed=2,
        iterations=1,
        initial_place_method=Ordered(),
    )

    assert distribution.placement == Placement(
        {0: 1, 1: 1, 2: 2, 3: 0, 4: 1, 5: 1, 6: 1}
    )
    assert distribution.cost() == 5

    distribution = allocator.allocate(
        circ,
        network,
        seed=1,
        iterations=1,
        initial_place_method=Ordered(),
    )

    assert distribution.placement == Placement(
        {0: 1, 1: 1, 2: 2, 3: 2, 4: 0, 5: 1, 6: 1}
    )
    assert distribution.cost() == 8

    circ = get_Rz_ladder_circ()
    allocator = Annealing()

    distribution = allocator.allocate(
        circ,
        network,
        seed=2,
        iterations=1,
        initial_place_method=Ordered(),
    )

    assert distribution.placement == Placement(
        {0: 1, 1: 1, 2: 2, 3: 0, 4: 1, 5: 1, 6: 1}
    )
    assert distribution.cost() == 3


def test_graph_initial_partitioning():
    network = get_line_network()
    circ = get_H_ladder_circ()

    allocator = HypergraphPartitioning()

    initial_distribution = allocator.allocate(circ, network, seed=1)

    assert initial_distribution.placement == Placement(
        {0: 2, 1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1}
    ) or initial_distribution.placement == Placement(
        {0: 2, 1: 1, 2: 2, 3: 1, 4: 1, 5: 1, 6: 1}
    )


def test_graph_partitioning_make_valid():
    network = NISQNetwork([[0, 1], [0, 2]], {0: [0], 1: [1, 2], 2: [3, 4, 5]})

    circ = (
        Circuit(4)
        .add_gate(OpType.CU1, 1.0, [0, 3])
        .H(0)
        .add_gate(OpType.CU1, 1.0, [1, 3])
        .H(1)
        .add_gate(OpType.CU1, 1.0, [2, 3])
        .H(2)
        .add_gate(OpType.CU1, 1.0, [0, 3])
        .add_gate(OpType.CU1, 1.0, [1, 3])
        .add_gate(OpType.CU1, 1.0, [2, 3])
    )
    dist_circ = HypergraphCircuit(circ)

    allocator = HypergraphPartitioning()
    bad_placement = Placement({v: 2 for v in dist_circ.vertex_list})
    assert not bad_placement.is_valid(dist_circ, network)

    distribution = Distribution(dist_circ, bad_placement, network)
    allocator.make_valid(distribution, seed=1)
    good_placement = Placement(
        {0: 0, 1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2}
    )

    assert distribution.is_valid()
    assert distribution.placement == good_placement


@pytest.mark.skip(reason="Circuit contains CX gates that are not supported.")
# NOTE: Moreover, when we do support them again (after including Junyi's work)
# the hypergraph will likely be different, so there is no guarantee the inital
# solution by KaHyPar will fail to be valid. Perhaps we should just remove
# this test, since validity is asserted at the end of refinement anyway.
def test_refinement_makes_valid():
    """In the case of an initial partition using KaHyPar this test fails
    since KaHyPar returns an invalid placement. Refinement fixes this.
    """
    server_coupling = [[0, 1], [1, 2]]
    server_qubits = {
        0: [0, 1],
        1: [2],
        2: [3, 4, 5],
    }
    network = NISQNetwork(server_coupling, server_qubits)

    with open("tests/test_circuits/not_valid_circ.json", "r") as fp:
        circuit = Circuit().from_dict(json.load(fp))

    allocator = HypergraphPartitioning()

    distribution = allocator.allocate(circuit, network, seed=0)
    assert distribution.is_valid()


def test_graph_partitioning_unused_qubits():
    network = get_line_network()
    allocator = HypergraphPartitioning()

    circ = Circuit(2)

    distribution = allocator.allocate(circ, network, seed=1)
    assert distribution.placement == Placement({0: 1, 1: 0})

    circ = Circuit(0)

    distribution = allocator.allocate(circ, network, seed=1)
    assert distribution.placement == Placement(dict())

    circ = Circuit(3).add_gate(OpType.CU1, 1.0, [1, 2])

    distribution = allocator.allocate(circ, network, seed=1)
    assert (distribution.placement == Placement({0: 2, 1: 1, 2: 1, 3: 1})) or (
        distribution.placement == Placement({0: 1, 1: 2, 2: 2, 3: 2})
    )


def test_kahypar_install():
    num_nodes = 4
    num_nets = 3

    hyperedge_indices = [0, 2, 4, 6]
    hyperedges = [0, 1, 1, 2, 2, 3]

    k = 2

    hypergraph = kahypar.Hypergraph(
        num_nodes, num_nets, hyperedge_indices, hyperedges, k
    )

    context = kahypar.Context()

    # Adapted: the packaged INI ships with pytket_dqc_cpp, not pytket_dqc.
    from pathlib import Path

    package_path = Path(pytket_dqc_cpp.__file__).resolve().parent
    ini_path = package_path / "allocators" / "km1_kKaHyPar_sea20.ini"
    assert ini_path.is_file()
    context.loadINIconfiguration(str(ini_path))

    context.setK(k)
    context.setEpsilon(0.03)
    context.suppressOutput(True)

    kahypar.partition(hypergraph, context)
    placement = [hypergraph.blockID(i) for i in range(hypergraph.numNodes())]

    assert placement == [0, 0, 1, 1] or placement == [1, 1, 0, 0]


def test_order_reducing_size():
    my_dict = {0: [0, 1], 2: [5, 6, 7, 8], 1: [2, 3, 4]}
    assert order_reducing_size(my_dict) == {
        2: [5, 6, 7, 8],
        1: [2, 3, 4],
        0: [0, 1],
    }


def test_random_allocator():
    circ = (
        Circuit(3).add_gate(OpType.CU1, 1.0, [0, 2]).add_gate(OpType.CU1, 1.0, [1, 2])
    )

    network = NISQNetwork([[0, 1], [0, 2]], {0: [0, 1], 1: [2, 3], 2: [4]})

    allocator = Random()
    distribution = allocator.allocate(circ, network, seed=0)

    assert distribution.placement == Placement({0: 1, 3: 1, 1: 0, 4: 1, 2: 2})
    assert distribution.cost() == 3


def test_ordered_allocator():
    small_circ = Circuit(2).add_gate(OpType.CU1, 1.0, [0, 1])

    med_circ = (
        Circuit(4)
        .add_gate(OpType.CU1, 1.0, [0, 1])
        .add_gate(OpType.CU1, 1.0, [1, 2])
        .add_gate(OpType.CU1, 1.0, [2, 3])
    )

    small_network = NISQNetwork([[0, 1]], {0: [0, 1], 1: [2, 3, 4]})

    large_network = NISQNetwork(
        [[0, 1], [0, 2]], {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8, 9]}
    )

    allocator = Ordered()

    placement_one = Placement({0: 2, 1: 2, 2: 2})
    alloc_distribution_one = allocator.allocate(small_circ, large_network)
    placement_two = Placement({0: 1, 1: 1, 2: 1, 3: 0, 4: 1, 5: 1, 6: 1})
    alloc_distribution_two = allocator.allocate(med_circ, small_network)

    assert alloc_distribution_one.placement == placement_one
    assert alloc_distribution_two.placement == placement_two


def test_brute_distribute_small_hyperedge():
    network = get_line_network()
    circ = get_H_ladder_circ()

    allocator = Brute()
    distribution = allocator.allocate(circ, network)

    assert distribution.cost() == 3
    assert distribution.placement == Placement(
        {0: 0, 4: 0, 1: 1, 5: 0, 2: 2, 6: 2, 3: 2}
    )


def test_brute_distribute():
    small_network = NISQNetwork([[0, 1]], {0: [0, 1], 1: [2]})
    small_circ = Circuit(2).add_gate(OpType.CU1, 1.0, [0, 1])

    med_network = NISQNetwork([[0, 1], [0, 2]], {0: [0], 1: [1], 2: [2, 3]})
    med_circ = (
        Circuit(4)
        .add_gate(OpType.CU1, 1.0, [0, 1])
        .add_gate(OpType.CU1, 1.0, [1, 2])
        .add_gate(OpType.CU1, 1.0, [2, 3])
    )

    allocator = Brute()

    distribution_small = allocator.allocate(small_circ, small_network)
    assert distribution_small.placement == Placement({0: 0, 2: 0, 1: 0})
    assert distribution_small.cost() == 0

    distribution_med = allocator.allocate(med_circ, med_network)
    assert distribution_med.placement == Placement(
        {0: 0, 4: 0, 1: 1, 5: 0, 2: 2, 6: 2, 3: 2}
    )
    assert distribution_med.cost() == 2


def test_routing_allocator():
    small_network = NISQNetwork([[0, 1], [0, 2]], {0: [0], 1: [1], 2: [2, 3]})
    small_circ = (
        Circuit(4)
        .add_gate(OpType.CU1, 1.0, [0, 1])
        .add_gate(OpType.CU1, 1.0, [1, 2])
        .add_gate(OpType.CU1, 1.0, [2, 3])
    )

    allocator = Routing()
    routing_distribution = allocator.allocate(small_circ, small_network)
    ideal_placement = Placement({0: 0, 4: 1, 5: 0, 1: 1, 2: 2, 6: 2, 3: 2})
    assert routing_distribution.placement == ideal_placement
    assert routing_distribution.cost() == 2

    med_network = NISQNetwork([[0, 1]], {0: [0, 1], 1: [2, 3, 4]})
    med_circ = (
        Circuit(5)
        .add_gate(OpType.CU1, 1.0, [0, 1])
        .add_gate(OpType.CU1, 1.0, [1, 2])
        .add_gate(OpType.CU1, 1.0, [0, 2])
        .add_gate(OpType.CU1, 1.0, [2, 3])
        .add_gate(OpType.CU1, 1.0, [3, 4])
        .add_gate(OpType.CU1, 1.0, [3, 2])
    )
    routing_distribution = allocator.allocate(med_circ, med_network)
    ideal_placement = Placement(
        {0: 0, 8: 1, 9: 0, 10: 0, 1: 0, 2: 1, 6: 1, 7: 1, 3: 1, 5: 1, 4: 1}
    )

    assert routing_distribution.placement == ideal_placement
    assert routing_distribution.cost() == 2

    med_circ_flipped = (
        Circuit(5)
        .add_gate(OpType.CU1, 1.0, [0, 1])
        .add_gate(OpType.CU1, 1.0, [1, 2])
        .add_gate(OpType.CU1, 1.0, [0, 2])
        .add_gate(OpType.CU1, 1.0, [2, 3])
        .add_gate(OpType.CU1, 1.0, [3, 4])
        .add_gate(OpType.CU1, 1.0, [2, 3])
    )

    routing_distribution = allocator.allocate(med_circ_flipped, med_network)
    ideal_placement = Placement(
        {0: 0, 8: 1, 9: 0, 10: 1, 1: 0, 2: 1, 6: 1, 7: 1, 3: 1, 5: 1, 4: 1}
    )

    assert routing_distribution.placement == ideal_placement
    assert routing_distribution.cost() == 1


@pytest.mark.skip(reason="QControlBox are not supported for now")
def test_q_control_box_circuits():
    network = NISQNetwork([[0, 1]], {0: [0], 1: [1]})

    op = Op.create(OpType.V)
    cv = QControlBox(op, 1)

    circ = Circuit(2)
    circ.add_qcontrolbox(cv, [0, 1])
    circ.add_qcontrolbox(cv, [0, 1])
    circ.Rz(0.3, 0)
    circ.add_qcontrolbox(cv, [0, 1])
    circ.H(0)
    circ.add_qcontrolbox(cv, [0, 1])
    circ.add_qcontrolbox(cv, [0, 1])

    allocator = Brute()

    distribution = allocator.allocate(circ, network)
    ideal_placement = Placement({0: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 1: 1})

    assert distribution.placement == ideal_placement
    assert distribution.cost() == 3


def test_CRz_circuits():
    network = NISQNetwork([[0, 1]], {0: [0], 1: [1]})

    circ = Circuit(2)
    circ.add_gate(OpType.CU1, 0.3, [1, 0])
    circ.add_gate(OpType.CU1, 0.1, [0, 1])
    circ.Rz(0.3, 0)
    circ.add_gate(OpType.CU1, 0.4, [0, 1])
    circ.H(0)
    circ.add_gate(OpType.CU1, 0.5, [1, 0])
    circ.add_gate(OpType.CU1, 0.2, [0, 1])

    allocator = Brute()

    distribution = allocator.allocate(circ, network)

    assert distribution.placement == Placement(
        {0: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 1: 1}
    )
    assert distribution.cost() == 1


def test_annealing_allocate_exception_type_mismatch_on_single_server():
    """Bug 10: Exception type mismatch in Annealing.allocate on 1-server network.

    In cpp/include/dqc/rng.hpp:55:
      std::invalid_argument("choice() on empty sequence") is mapped by pybind11
      to ValueError.
    Reference oracle (pytket_dqc.allocators.Annealing.allocate):
      Python's random.choice([]) raises IndexError: Cannot choose from an empty sequence.
    pytket_dqc_cpp:
      Raises ValueError instead of IndexError.
    """
    import pytket_dqc_cpp as cpp

    c = Circuit(1)
    net_cpp = cpp.networks.NISQNetwork([[0, 0]], {0: [0]})

    with pytest.raises(IndexError) as exc_info:
        cpp.allocators.Annealing().allocate(c, net_cpp, iterations=1)

    assert "Cannot choose from an empty sequence" in str(exc_info.value)


def test_hypergraph_partitioning_key_error_on_non_contiguous_servers():
    """Bug 11: Exception type mismatch in HypergraphPartitioning on non-contiguous server IDs.

    In cpp/include/dqc/gain_manager.hpp:
      occupancy_.at(server) throws std::out_of_range, mapped by pybind11 to
      IndexError: unordered_map::at.
    Reference oracle (pytket_dqc.allocators.HypergraphPartitioning.allocate):
      Dict lookup occupancy[server] raises KeyError: 0.
    pytket_dqc_cpp:
      Raises IndexError instead of KeyError.
    """
    import pytket_dqc_cpp as cpp
    from pytket.circuit import Op

    cz = Op.create(OpType.CU1, 1.0)
    circ = Circuit(2).add_gate(cz, [0, 1])
    net_cpp = cpp.networks.NISQNetwork([[10, 20]], {10: [0], 20: [1]})

    with pytest.raises(KeyError):
        cpp.allocators.HypergraphPartitioning().allocate(circ, net_cpp)


def test_hypergraph_partitioning_silent_overflow_on_large_seed():
    """Bug 13: Silent integer truncation and missing validation on large seeds.

    In cpp/include/dqc/hypergraph_partitioning.hpp:145:
      kahypar_set_seed(context, static_cast<int>(*seed));
      Silently casts uint64 to signed 32-bit int, overflowing seeds > 2^31 - 1.
    Reference oracle (pytket_dqc.allocators.HypergraphPartitioning):
      KaHyPar context raises TypeError: setSeed(): incompatible function arguments.
    pytket_dqc_cpp:
      Silently truncates to negative seed and passes without raising TypeError.
    """
    import pytket_dqc_cpp as cpp
    from pytket.circuit import Op

    cz = Op.create(OpType.CU1, 1.0)
    circ = Circuit(2).add_gate(cz, [0, 1])
    net_cpp = cpp.networks.NISQNetwork([[0, 1]], {0: [0], 1: [1]})
    large_seed = 3000000000  # > 2**31 - 1

    with pytest.raises(TypeError):
        cpp.allocators.HypergraphPartitioning().allocate(circ, net_cpp, seed=large_seed)

