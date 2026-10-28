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

from pytket import Circuit

from ..circuits.distribution import Distribution
from ..networks.nisq_network import NISQNetwork
from ..refiners.h_type_merge import EagerHTypeMerge
from ..refiners.boundary_reallocation import BoundaryReallocation
from ..refiners.repeat_refiner import RepeatRefiner
from ..allocators.hypergraph_partitioning import HypergraphPartitioning
from ..allocators.annealing import Annealing
from .distributor import Distributor


class PartitioningAnnealing(Distributor):
    """Distributor using the :class:`.Annealing` allocator."""

    def distribute(self, circ: Circuit, network: NISQNetwork, **kwargs) -> Distribution:
        """Method producing a distribution of the given circuit
        onto the given network.

        Note that kwargs are passed on to the allocate method of
        :class:`.Annealing`.

        :param circ: Circuit to be distributed
        :type circ: pytket.Circuit
        :param network: Network onto which circuit should be distributed
        :type network: NISQNetwork
        :key seed: Seed for randomness. Default is None.
            Note: the annealing iterations run in the compiled core, but Stage 1
            seeds Python's global ``random`` module (``random.seed(seed)``), and
            any Python initial placement allocator may draw from it. The default
            initializer (``initial_place_method=Random()``) does not consume user
            seeds deterministically. For guaranteed reproducibility, pass both an
            explicit integer ``seed`` and a deterministic initial placement method
            such as any from: ``Ordered``, ``HypergraphPartitioning``, ``Brute``, or
            pass an explicitly seeded wrapper around ``Random``.
        :key iterations: The number of iterations in the annealing procedure.
            Default is 30000.
        :key initial_place_method: Allocator to use to find the initial
            placement. Default is Random().
        :key cache_limit: The maximum size of the set of servers whose cost is
            stored in cache; see GainManager. Default is None.
        :key initial_temperature: Initial temperature of annealing procedure.
            Default is 3.
        :return: Distribution of circ onto network.
        :rtype: Distribution
        """

        return Annealing().allocate(circ, network, **kwargs)


class PartitioningHeterogeneous(Distributor):
    """Distributor refining the output of :class:`.HypergraphPartitioning`
    to adapt the result to heterogeneous networks.
    """

    def distribute(self, circ: Circuit, network: NISQNetwork, **kwargs) -> Distribution:
        """Method producing a distribution of the given circuit
        onto the given network.

        Note that kwargs are passed on to the allocate method of
        :class:`.HypergraphPartitioning` and the `refine` method of
        :class:`.BoundaryReallocation`.

        :param circ: Circuit to be distributed
        :type circ: pytket.Circuit
        :param network: Network onto which circuit should be distributed
        :type network: NISQNetwork
        :key seed: Seed for randomness. Default is None: unseeded calls
            continue Python's ambient ``random`` stream and return the advanced
            state to it, exactly as upstream ``pytket_dqc`` does. For
            reproducible results or cross-library parity, specify an explicit
            integer seed.
        :return: Distribution of circ onto network.
        :rtype: Distribution
        """

        distribution = HypergraphPartitioning().allocate(circ, network, **kwargs)
        refiner = BoundaryReallocation()
        refiner.refine(distribution, **kwargs)

        return distribution


class PartitioningHeterogeneousEmbedding(Distributor):
    """Distributor refining the output of :class:`.PartitioningHeterogeneous`
    to make use of embedding.
    """

    def distribute(self, circ: Circuit, network: NISQNetwork, **kwargs) -> Distribution:
        """Method producing a distribution of the given circuit
        onto the given network.

        Note that kwargs are passed on to the distribute method of
        :class:`.PartitioningHeterogeneousEmbedding`.

        :param circ: Circuit to be distributed
        :type circ: pytket.Circuit
        :param network: Network onto which circuit should be distributed
        :type network: NISQNetwork
        :key seed: Seed for randomness. Default is None: unseeded calls
            continue Python's ambient ``random`` stream and return the advanced
            state to it, exactly as upstream ``pytket_dqc`` does. For
            reproducible results or cross-library parity, specify an explicit
            integer seed.
        :return: Distribution of circ onto network.
        :rtype: Distribution
        """

        initial_distributor = kwargs.get(
            "initial_distributor", PartitioningHeterogeneous()
        )

        distribution = initial_distributor.distribute(circ, network, **kwargs)
        refiner = RepeatRefiner(EagerHTypeMerge())
        refiner.refine(distribution)

        return distribution
