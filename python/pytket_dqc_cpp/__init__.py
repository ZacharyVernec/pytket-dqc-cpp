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

"""Drop-in Python API for the C++ port of ``pytket_dqc``.

The public exports mirror the reference package, so that

.. code-block:: python

    import pytket_dqc_cpp as pytket_dqc
    pytket_dqc.allocators.Random

works as expected.  Subpackages are imported eagerly (in the reference's
dependency order) so that attribute access on the top-level module always
resolves.
"""

from .circuits import HypergraphCircuit, Distribution  # noqa:F401
from .placement import Placement  # noqa:F401
from .networks import NISQNetwork  # noqa:F401
from .utils import DQCPass  # noqa:F401

from . import circuits  # noqa:F401
from . import placement  # noqa:F401
from . import networks  # noqa:F401
from . import utils  # noqa:F401
from . import packing  # noqa:F401
from . import allocators  # noqa:F401
from . import refiners  # noqa:F401
from . import distributors  # noqa:F401
