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

from typing import TYPE_CHECKING

from .._delegation import from_reference, reference_module, to_reference, warn_once
from ..circuits.distribution import Distribution
from .allocator import Allocator

if TYPE_CHECKING:
    from pytket import Circuit
    from pytket_dqc_cpp.networks import NISQNetwork


class Routing(Allocator):
    """Distribute quantum circuits using routing tools available in
    `tket <https://cqcl.github.io/tket/pytket/api/routing.html>`_. Note that
    this allocator will alter the initial circuit.

    Routing is not ported to the C++ core, so this class delegates to the
    reference ``pytket_dqc`` implementation (with a one-off ``UserWarning``).
    """

    def __init__(self) -> None:
        pass

    def allocate(self, circ: Circuit, network: NISQNetwork, **kwargs) -> Distribution:
        """Distribute quantum circuits using routing tools available in
        `tket <https://cqcl.github.io/tket/pytket/api/routing.html>`_. Note
        that this allocator will alter the initial circuit.

        :param circ: Circuit to distribute.
        :type circ: pytket.Circuit
        :param network: Network onto which ``circ`` should be distributed.
        :type network: NISQNetwork
        :return: Distribution of ``circ`` onto ``network``.
        :rtype: Distribution
        """
        warn_once("Routing")
        reference_allocators = reference_module("allocators")
        reference_distribution = reference_allocators.Routing().allocate(
            circ, to_reference(network, kind="NISQNetwork"), **kwargs
        )
        return from_reference(reference_distribution, kind="Distribution")
