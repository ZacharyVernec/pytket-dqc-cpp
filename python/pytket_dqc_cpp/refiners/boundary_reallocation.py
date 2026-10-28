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

import math

from .._bridge import (
    call_with_python_rng,
    core,
    core_distribution,
    normalize_seed,
    sync_from_core,
)
from ..circuits.distribution import Distribution
from .refiner import Refiner


class BoundaryReallocation(Refiner):
    """Refiner that considers reallocations of the vertices on the boundary of
    the partition, attempting to improve the solution. It is greedy and it
    is not capable of escaping local optima. The justification
    is that we assume that the Allocator used already grouped qubits
    efficiently and that our network has short average distance, so that we
    only need to refine the allocation of vertices in the boundary.

    The greedy rounds run in the compiled core; the resulting placement is
    synced back onto ``distribution`` immediately after the call.
    """

    def refine(self, distribution: Distribution, **kwargs) -> bool:
        """The refinement algorithm proceeds in rounds. In each round, all of
        the vertices in the boundary are visited in random order and we
        we calculate the gain achieved by moving the vertex to other servers.
        The best move is applied, with ties broken randomly. If all possible
        moves have negative gains, the vertex is not moved.

        The algorithm continues until the proportion of vertices
        moved in a round (i.e. #moved / #boundary) is smaller than
        ``stop_parameter`` or the maximum ``num_rounds`` is reached.

        This refinement algorithm is taken from the discussion of the "label
        propagation" algorithm in https://arxiv.org/abs/1402.3281. However,
        we do not consider any coarsening.

        :param distribution: Distribution to refine.
        :type distribution: Distribution

        :key fixed_vertices: A list of vertices that cannot be reallocated.
            Default is [].
        :key num_rounds: Max number of refinement rounds. Default is 10.
        :key stop_parameter: Real number in [0,1]. If proportion of moves
            in a round is smaller than this number, do no more rounds. Default
            is 0.05.
        :key seed: Seed for randomness. Default is None: unseeded calls
            continue Python's ambient ``random`` stream and return the advanced
            state to it, exactly as upstream ``pytket_dqc`` does. For
            reproducible results or cross-library parity, specify an explicit
            integer seed.
        :key cache_limit: The maximum size of the set of servers whose cost is
            stored in cache. Default value is 5.

        :return: Distribution where the placement updated.
        :rtype: Distribution
        """

        fixed_vertices = kwargs.get("fixed_vertices", [])
        num_rounds = kwargs.get("num_rounds", 10)
        stop_parameter = kwargs.get("stop_parameter", 0.05)
        # The core binding takes an unsigned integer; CPython's random.seed
        # uses abs(seed), including for negative integers.
        seed = normalize_seed(kwargs.get("seed", None))
        cache_limit = kwargs.get("cache_limit", None)

        core_dist = core_distribution(distribution)
        refinement_made = bool(
            call_with_python_rng(
                core.boundary_reallocation_refine_with_state,
                core_dist,
                [int(vertex) for vertex in fixed_vertices],
                # The reference loops ``while round_id < num_rounds``, so a
                # fractional ``num_rounds`` still runs ``ceil(num_rounds)``
                # rounds; the core takes an int.
                math.ceil(num_rounds),
                stop_parameter,
                seed,
                cache_limit,
                seed=seed,
                # The reference only reseeds when an explicit seed is given;
                # otherwise it continues the ambient stream.
                reseed=seed is not None,
            )
        )
        sync_from_core(distribution, core_dist)
        return refinement_made
