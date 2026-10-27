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

import importlib.resources
import random
from typing import TYPE_CHECKING

from .._bridge import (
    core,
    core_hypergraph_circuit,
    core_network,
)
from ..circuits.distribution import Distribution
from ..circuits.hypergraph_circuit import HypergraphCircuit
from ..placement.placement import Placement
from ._core_allocation import distribution_from_core
from .allocator import Allocator
from .gain_manager import GainManager

if TYPE_CHECKING:
    from pytket import Circuit
    from pytket_dqc_cpp.networks import NISQNetwork


def _default_ini_path() -> str:
    """Path to the KaHyPar INI configuration shipped with this package."""
    package_path = importlib.resources.files("pytket_dqc_cpp")
    return str(package_path / "allocators" / "km1_kKaHyPar_sea20.ini")


def _require_kahypar():
    """Raise an actionable error when the core was built without KaHyPar."""
    if not hasattr(core, "hypergraph_partitioning_allocate"):
        raise RuntimeError(
            "This build of the compiled core does not include KaHyPar. "
            "Rebuild with `-DDQC_BUILD_KAHYPAR=ON` (see README.md) or install "
            "KaHyPar, then reinstall the extension."
        )


class HypergraphPartitioning(Allocator):
    """Distribution technique, making use of existing tools for hypergraph
    partitioning available through the `KaHyPar <https://kahypar.org/>`_
    package. This allocator will ignore weights on hyperedges and
    assume all hyperedges have weight 1. This allocator will ignore the
    connectivity of the NISQNetwork.
    """

    def allocate(self, circ: Circuit, network: NISQNetwork, **kwargs) -> Distribution:
        """Distribute ``circ`` onto ``network``. The distribution
        is found by KaHyPar using the connectivity metric. All-to-all
        connectivity of the network of modules is assumed; you may wish
        to run the refiner ``BoundaryReallocation`` on the output
        ``Distribution`` to take the network topology into account.

        :param circ: Circuit to distribute.
        :type circ: pytket.Circuit
        :param network: Network onto which ``circ`` should be placed.
        :type network: NISQNetwork

        :key ini_path: Path to kahypar ini file. Default points to the
            ini file within the pytket-dqc repository.
        :key seed: Seed for randomness. Default is None

        :return: Distribution of ``circ`` onto ``network``.
        :rtype: Distribution
        """

        _require_kahypar()

        dist_circ = HypergraphCircuit(circ)
        if not network.can_implement(dist_circ):
            raise Exception("This circuit cannot be implemented on this network.")

        ini_path = kwargs.get("ini_path", _default_ini_path())
        seed = kwargs.get("seed", None)

        # KaHyPar's initial placement followed by ``make_valid`` both run in
        # the core; the result is a fully-populated placement.
        core_distribution = core.hypergraph_partitioning_allocate(
            core_hypergraph_circuit(dist_circ),
            core_network(network),
            ini_path,
            seed,
        )
        return distribution_from_core(core_distribution, dist_circ, network)

    def make_valid(self, distribution: Distribution, **kwargs):
        """Since KaHyPar does not guarantee that the requirement on server
        capacity will be satisfied, we enforce this ourselves.
        However, it usually does satisfy the requirement and the following
        code often does nothing or it moves very few vertices.
        """
        seed = kwargs.get("seed", None)
        if seed is not None:
            random.seed(seed)

        # We will use a ``GainManager`` to manage the calculation of gains
        # (and management of pre-computed values) in a transparent way
        gain_manager = GainManager(distribution)

        for vertex in gain_manager.qubit_vertices:
            # If moving in place is not valid then the current server is full
            if not gain_manager.is_move_valid(
                vertex, gain_manager.current_server(vertex)
            ):
                # Then, find the first server where the move would be valid
                for server in gain_manager.occupancy.keys():
                    if gain_manager.is_move_valid(vertex, server):
                        # Move ``vertex`` to a server with free spaces and
                        # return control to the outer loop
                        gain_manager.move_vertex(vertex, server)
                        break
        # Notice that the moves have been arbitrary, i.e. we have not
        # calculated gains. This is fine since the vertices we moved will
        # likely be boundary vertices; the following rounds of the
        # refinement algorithm will move them around to optimise gains.
        # At the end of the previous subroutine, no server should be
        # overpopulated.
        assert gain_manager.distribution.is_valid()
        # GainManager has updated ``distribution`` in place:
        assert gain_manager.distribution is distribution

    def initial_distribute(
        self,
        dist_circ: HypergraphCircuit,
        network: NISQNetwork,
        ini_path: str,
        **kwargs,
    ) -> Placement:
        """Distribute ``dist_circ`` onto ``network`` using graph partitioning
        tools available in `kahypar <https://kahypar.org/>`_ package. The
        placement returned is not taking into account network topology.
        However, it does take into account server sizes.

        :param dist_circ: Circuit to distribute.
        :type dist_circ: HypergraphCircuit
        :param network: Network onto which ``dist_circ`` should be placed.
        :type network: NISQNetwork
        :param ini_path: Path to kahypar ini file.
        :type ini_path: str

        :key seed: Seed for randomness. Default is None

        :return: Placement of ``dist_circ`` onto ``network``.
        :rtype: Placement
        """

        _require_kahypar()

        if not dist_circ.is_valid():
            raise Exception("This hypergraph is not valid.")

        seed = kwargs.get("seed", None)

        # `ini_path` is honoured here (unlike the reference, which silently
        # replaces it with its packaged default).
        core_placement_obj = core.hypergraph_partitioning_initial_placement(
            core_hypergraph_circuit(dist_circ),
            core_network(network),
            str(ini_path),
            seed,
        )
        return Placement(
            {
                int(vertex): int(server)
                for vertex, server in core_placement_obj.placement.items()
            }
        )
