"""C++ core of pytket-dqc (pybind11 bindings).

The compiled `_core` extension is built to `<repo>/build/dqc_cpp/_core*.so` by
CMake (see `-DDQC_BUILD_BINDINGS=ON`). This package re-exports it.
"""

import os as _os
import sys as _sys

_BUILD_DIR = _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(__file__))),
                  "build", "dqc_cpp")
)
if _os.path.isdir(_BUILD_DIR) and _BUILD_DIR not in _sys.path:
    _sys.path.insert(0, _BUILD_DIR)

from dqc_cpp._core import *  # noqa: F401,F403,E402

# Default KaHyPar configuration used by `_core.hypergraph_partitioning_allocate`
# (mirrors the reference's `importlib_resources` default INI path). Overridable
# via the `ini_path` keyword argument.
KAHYPAR_INI_PATH = _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "km1_kKaHyPar_sea20.ini"
)
