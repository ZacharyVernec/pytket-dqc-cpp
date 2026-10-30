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
# Part of the `pytket_dqc_cpp` adapter.

"""Loader for the compiled ``_core`` extension module.

Two layouts are supported, in this order:

1. *Installed extension*: the shared object is imported as
   ``pytket_dqc_cpp._core`` (``from . import _core``).
2. *Build directory*: ``<repo>/build/dqc_cpp`` is added to ``sys.path`` and the
   **top-level** ``_core`` module is imported.  The legacy
   ``python/dqc_cpp/__init__.py`` does ``from dqc_cpp._core import *``, which
   only works when the shared object is copied next to it; from the build
   directory alone the extension is importable solely as ``_core``.  Note that
   this module is itself called ``_core``: in the build-directory layout the
   import below resolves back to *this* module, which is why the self-reference
   is detected explicitly.

The resolved extension module is exposed as :data:`core`, and its public
attributes are mirrored onto this module so that ``pytket_dqc_cpp._core.<name>``
also works in the build-directory layout.
"""

from __future__ import annotations

import importlib
import os
import sys
from types import ModuleType

__all__ = ["core", "load_core", "BUILD_COMMAND"]

#: Subdirectories holding the compiled extension produced by CMake.
_BUILD_SUBDIR = ("build", "dqc_cpp")

#: Attribute used to recognise a genuine ``_core`` extension module.
_REQUIRED_ATTRIBUTE = "HypergraphCircuit"

#: Build command quoted in the error raised when the extension is missing.
BUILD_COMMAND = (
    "KAHYPAR_ROOT=/path/to/kahypar "
    "cmake -S . -B build -DCMAKE_BUILD_TYPE=Release "
    "-DDQC_BUILD_BINDINGS=ON && cmake --build build -j"
)


def _package_dir() -> str:
    """Directory containing this file (``python/pytket_dqc_cpp``)."""
    return os.path.dirname(os.path.realpath(__file__))


def _repo_root() -> str:
    """Repository root, assuming the ``python/<package>`` source layout."""
    return os.path.dirname(os.path.dirname(_package_dir()))


def build_dir() -> str:
    """Absolute path of the CMake build directory holding ``_core``.

    ``DQC_CORE_DIR`` overrides the default ``<repo>/build/dqc_cpp`` so that
    alternate CMake build directories (or a staged install) can be used
    without editing the package.
    """
    override = os.environ.get("DQC_CORE_DIR")
    if override:
        return os.path.abspath(os.path.expanduser(override))
    return os.path.join(_repo_root(), *_BUILD_SUBDIR)


def _is_core_module(module: ModuleType) -> bool:
    return hasattr(module, _REQUIRED_ATTRIBUTE)


def _import_installed_extension() -> ModuleType | None:
    """Import ``pytket_dqc_cpp._core`` when the extension is installed."""
    if not __package__:
        return None
    try:
        module = importlib.import_module(f"{__package__}._core")
    except ImportError:
        return None
    # In the build-directory layout this import resolves to this very module
    # (`pytket_dqc_cpp._core` is this pure-Python loader), so it must be
    # rejected rather than accepted as the extension.
    if module is sys.modules.get(__name__):
        return None
    if not _is_core_module(module):
        return None
    return module


def _import_from_build_dir() -> ModuleType | None:
    """Import the top-level ``_core`` from ``<repo>/build/dqc_cpp``."""
    directory = build_dir()
    if not os.path.isdir(directory):
        return None
    if directory not in sys.path:
        sys.path.insert(0, directory)
    try:
        module = importlib.import_module("_core")
    except ImportError:
        return None
    if not _is_core_module(module):
        return None
    return module


def load_core() -> ModuleType:
    """Return the compiled ``_core`` extension module.

    :raises ImportError: If the extension cannot be found or imported.
    """
    module = _import_installed_extension()
    if module is None:
        module = _import_from_build_dir()
    if module is None:
        raise ImportError(
            "Could not import the compiled `_core` extension module.\n"
            "Build it first with:\n"
            f"    {BUILD_COMMAND}\n"
            "and either install the project (`uv sync` / `pip install .`) or "
            f"run with `{build_dir()}` present (or on `PYTHONPATH`)."
        )
    return module


#: The single module object every other adapter module imports.
core: ModuleType = load_core()


def _mirror_public_attributes() -> None:
    """Expose ``pytket_dqc_cpp._core.<name>`` in the build-directory layout."""
    for name in dir(core):
        if not name.startswith("__"):
            globals().setdefault(name, getattr(core, name))


_mirror_public_attributes()
del _mirror_public_attributes
