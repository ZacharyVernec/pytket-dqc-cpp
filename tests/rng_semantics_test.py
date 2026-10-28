from __future__ import annotations

import pytest
import warnings
from pytket import Circuit, OpType
from pytket_dqc_cpp._delegation import reset_warnings
from pytket_dqc_cpp.allocators import (
    Annealing,
    Brute,
    HypergraphPartitioning,
    Ordered,
    Random,
)
from pytket_dqc_cpp.distributors import PartitioningAnnealing
from pytket_dqc_cpp.networks import NISQNetwork


def get_H_ladder_circ() -> Circuit:
    circ = (
        Circuit(4)
        .add_gate(OpType.CU1, 1.0, [0, 1])
        .H(1)
        .add_gate(OpType.CU1, 1.0, [1, 2])
        .H(2)
        .add_gate(OpType.CU1, 1.0, [2, 3])
        .H(3)
    )
    return circ


def get_line_network() -> NISQNetwork:
    return NISQNetwork([[0, 1], [0, 2]], {0: [0], 1: [1, 2], 2: [3, 4]})


def test_annealing_warns_about_default_initializer():
    """Verify that a warning is emitted once when seed is passed with default initializer."""
    reset_warnings()
    circ = get_H_ladder_circ()
    network = get_line_network()

    with pytest.warns(
        UserWarning,
        match="such as any from the following set: Ordered, HypergraphPartitioning, Brute",
    ):
        Annealing().allocate(circ, network, seed=1, iterations=1)

    # Calling it a second time should NOT emit a warning due to warn_once
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        Annealing().allocate(circ, network, seed=1, iterations=1)
        assert len([w for w in record if "Annealing: the default" in str(w.message)]) == 0

    # After reset_warnings(), it should warn again
    reset_warnings()
    with pytest.warns(
        UserWarning,
        match="such as any from the following set: Ordered, HypergraphPartitioning, Brute",
    ):
        Annealing().allocate(circ, network, seed=1, iterations=1)


def test_annealing_no_warning_when_unseeded():
    """Verify that unseeded Annealing does not trigger the default initializer warning."""
    reset_warnings()
    circ = get_H_ladder_circ()
    network = get_line_network()

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        Annealing().allocate(circ, network, iterations=1)
        assert len([w for w in record if "Annealing: the default" in str(w.message)]) == 0


def test_annealing_no_warning_with_explicit_initial_place_method():
    """Verify that explicitly passing an initial_place_method emits no warning."""
    reset_warnings()
    circ = get_H_ladder_circ()
    network = get_line_network()

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        Annealing().allocate(
            circ, network, seed=1, iterations=1, initial_place_method=Ordered()
        )
        assert len([w for w in record if "Annealing: the default" in str(w.message)]) == 0


@pytest.mark.parametrize("allocator_cls", [Ordered, Brute, HypergraphPartitioning])
def test_annealing_deterministic_allocators_match(allocator_cls):
    """Verify that all deterministic initializers produce identical results across runs."""
    circ = get_H_ladder_circ()
    network = get_line_network()

    allocator = Annealing()
    d1 = allocator.allocate(
        circ,
        network,
        seed=42,
        iterations=50,
        initial_place_method=allocator_cls(),
    )
    d2 = allocator.allocate(
        circ,
        network,
        seed=42,
        iterations=50,
        initial_place_method=allocator_cls(),
    )

    assert d1.placement == d2.placement
    assert d1.cost() == d2.cost()


def test_annealing_seeded_random_wrapper():
    """Verify that a custom seeded Random wrapper produces reproducible results."""
    circ = get_H_ladder_circ()
    network = get_line_network()

    class SeededRandomAllocator(Random):
        def __init__(self, seed: int):
            super().__init__()
            self.seed = seed

        def allocate(self, circ, network, **kwargs):
            kwargs["seed"] = self.seed
            return super().allocate(circ, network, **kwargs)

    allocator = Annealing()
    d1 = allocator.allocate(
        circ,
        network,
        seed=123,
        iterations=50,
        initial_place_method=SeededRandomAllocator(seed=123),
    )
    d2 = allocator.allocate(
        circ,
        network,
        seed=123,
        iterations=50,
        initial_place_method=SeededRandomAllocator(seed=123),
    )

    assert d1.placement == d2.placement
    assert d1.cost() == d2.cost()


def test_partitioning_annealing_warns_with_default_initializer():
    """Verify that PartitioningAnnealing emits a warning when seeded with default initializer."""
    reset_warnings()
    circ = get_H_ladder_circ()
    network = get_line_network()

    with pytest.warns(
        UserWarning,
        match="such as any from the following set: Ordered, HypergraphPartitioning, Brute",
    ):
        PartitioningAnnealing().distribute(circ, network, seed=1, iterations=1)


def test_partitioning_annealing_deterministic_with_explicit_initializer():
    """Verify that PartitioningAnnealing is reproducible when given a deterministic initializer."""
    circ = get_H_ladder_circ()
    network = get_line_network()

    distributor = PartitioningAnnealing()
    d1 = distributor.distribute(
        circ,
        network,
        seed=42,
        iterations=50,
        initial_place_method=Ordered(),
    )
    d2 = distributor.distribute(
        circ,
        network,
        seed=42,
        iterations=50,
        initial_place_method=Ordered(),
    )

    assert d1.placement == d2.placement
    assert d1.cost() == d2.cost()


def test_pyrandom_unseeded_is_nondeterministic():
    """Verify that unseeded PyRandom draws from OS entropy non-deterministically."""
    from pytket_dqc_cpp._bridge import core

    r1 = core.PyRandom()
    r2 = core.PyRandom()
    v1 = [r1.random() for _ in range(10)]
    v2 = [r2.random() for _ in range(10)]
    assert v1 != v2


def test_pyrandom_seeded_is_deterministic():
    """Verify that seeded PyRandom produces identical sequences."""
    from pytket_dqc_cpp._bridge import core

    r1 = core.PyRandom(42)
    r2 = core.PyRandom(42)
    v1 = [r1.random() for _ in range(10)]
    v2 = [r2.random() for _ in range(10)]
    assert v1 == v2


def test_random_allocate_consumes_global_random():
    """Random.allocate reseeds and advances Python's generator like upstream."""
    import random

    circ = get_H_ladder_circ()
    network = get_line_network()

    random.seed(123)
    before = random.getstate()
    Random().allocate(circ, network, seed=7)
    assert random.getstate() != before

    random.seed(123)
    Random().allocate(circ, network, seed=7)
    first = random.getstate()
    random.seed(999)
    Random().allocate(circ, network, seed=7)
    assert random.getstate() == first


def test_refiners_use_global_random_like_reference():
    """Refiners continue the ambient stream unseeded and reseed when seeded."""
    import random
    from pytket_dqc_cpp.refiners import BoundaryReallocation, DetachedGates

    circ = get_H_ladder_circ()
    network = get_line_network()

    for refiner in (BoundaryReallocation(), DetachedGates()):
        random.seed(123)
        before = random.getstate()
        refiner.refine(Ordered().allocate(circ, network))
        assert random.getstate() != before

        random.seed(123)
        refiner.refine(Ordered().allocate(circ, network))
        first = random.getstate()
        random.seed(123)
        refiner.refine(Ordered().allocate(circ, network))
        assert random.getstate() == first

        random.seed(123)
        refiner.refine(Ordered().allocate(circ, network), seed=42)
        seeded_from_123 = random.getstate()
        random.seed(999)
        refiner.refine(Ordered().allocate(circ, network), seed=42)
        assert random.getstate() == seeded_from_123


def test_seeded_refiners_are_deterministic():
    """Verify that refiners with explicit seed produce identical placements."""
    from pytket_dqc_cpp.refiners import BoundaryReallocation

    circ = get_H_ladder_circ()
    network = get_line_network()

    d1 = Ordered().allocate(circ, network)
    d2 = Ordered().allocate(circ, network)

    BoundaryReallocation().refine(d1, seed=42)
    BoundaryReallocation().refine(d2, seed=42)

    assert d1.placement == d2.placement
    assert d1.cost() == d2.cost()


def test_annealing_returns_core_state_to_python():
    """The annealing loop's draws are visible in Python's generator afterwards."""
    import random

    circ = get_H_ladder_circ()
    network = get_line_network()

    random.seed(11)
    after_seed = random.getstate()
    Annealing().allocate(
        circ, network, seed=11, iterations=200, initial_place_method=Ordered()
    )
    assert random.getstate() != after_seed

    random.seed(999)
    Annealing().allocate(
        circ, network, seed=11, iterations=200, initial_place_method=Ordered()
    )
    first = random.getstate()
    random.seed(999)
    Annealing().allocate(
        circ, network, seed=11, iterations=200, initial_place_method=Ordered()
    )
    assert random.getstate() == first


