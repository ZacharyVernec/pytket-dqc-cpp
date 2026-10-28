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

from __future__ import annotations

import random

from .._bridge import (
    call_with_python_rng,
    core,
    core_hypergraph_circuit,
    core_network,
    core_placement,
    normalize_seed,
)
from .._delegation import warn_once
from ..circuits.distribution import Distribution
from ..circuits.hypergraph_circuit import HypergraphCircuit
from ._core_allocation import distribution_from_core
from .allocator import Allocator
from .random import Random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pytket import Circuit
    from pytket_dqc_cpp.networks import NISQNetwork


class Annealing(Allocator):
    """Allocator taking a simulated annealing approach to quantum circuit
    distribution.
    """

    def __init__(self) -> None:
        pass

    def allocate(self, circ: Circuit, network: NISQNetwork, **kwargs) -> Distribution:
        """Distribute quantum circuit using simulated annealing approach.

        :param circ: Circuit to distribute.
        :type circ: pytket.Circuit
        :param network: Network onto which circuit is to be distributed.
        :type network: NISQNetwork
        :return: Distribution of ``circ`` onto ``network``.
        :rtype: Distribution

        :key seed: Seed for randomness. Default is None.
        :key iterations: The number of iterations in the annealing procedure.
            Default is 10000.
        :key initial_place_method: Allocator to use to find the initial
            placement. Default is Random().
            Note: The default Random() initializer does not consume user seeds
            deterministically. For guaranteed cross-platform reproducibility, pass
            a deterministic initial placement method such as any from the following
            set: Ordered, HypergraphPartitioning, Brute, or pass an explicitly
            seeded wrapper around Random.
        :key cache_limit: The maximum size of the set of servers whose cost is
            stored in cache; see GainManager. Default value is 5.
        :key initial_temperature: Initial temperature of annealing procedure.
            Default value of 3.
        """

        dist_circ = HypergraphCircuit(circ)
        if not network.can_implement(dist_circ):
            raise Exception("This circuit cannot be implemented on this network.")

        iterations = kwargs.get("iterations", 30000)
        initial_aloc = kwargs.get("initial_place_method", Random())
        seed = kwargs.get("seed", None)
        cache_limit = kwargs.get("cache_limit", None)
        initial_temperature = kwargs.get("initial_temperature", 3)

        if "initial_place_method" not in kwargs and seed is not None:
            warn_once(
                "Annealing_default_random_seed",
                message=(
                    "Annealing: the default Random() initializer does not consume the "
                    "seed deterministically. For guaranteed cross-platform reproducibility, "
                    "pass a deterministic initial placement method such as any from the "
                    "following set: Ordered, HypergraphPartitioning, Brute, or pass an "
                    "explicitly seeded wrapper around Random."
                ),
                stacklevel=2,
            )

        # Stage 1: The annealing procedure requires an initial placement to work with.
        # Run the initial placement allocator (Ordered, HypergraphPartitioning,
        # Brute, Random, or any custom allocator).
        random.seed(seed)
        distribution = initial_aloc.allocate(circ, network)

        # Stage 2: Simulated Annealing from the initial placement, executed in
        # the compiled core.  Python's ambient RNG state is handed over and the
        # post-loop state is installed again, matching the reference's use of
        # the module-level generator.  ``reseed=False`` keeps any draws Stage 1
        # already consumed.
        core_dist = call_with_python_rng(
            core.annealing_from_placement_with_state,
            core_hypergraph_circuit(dist_circ),
            core_placement(distribution.placement),
            core_network(network),
            int(iterations),
            float(initial_temperature),
            cache_limit,
            normalize_seed(seed),
            reseed=False,
        )
        return distribution_from_core(core_dist, dist_circ, network)
