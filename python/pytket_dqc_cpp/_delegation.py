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

"""Helpers for the features that stay in the reference ``pytket_dqc`` package.

A handful of features are not ported to the C++ core (see ``README.md``).  They
are executed by the reference implementation instead, after a single
``UserWarning`` per feature and process.  ``pytket_dqc`` is a runtime
dependency, but it is only imported *inside* the delegated functions: importing
this module never imports ``pytket_dqc``, which keeps the adapter free of
circular imports and of module-scope coupling to the reference package.
"""

from __future__ import annotations

import importlib
import warnings
from types import ModuleType
from typing import Any, Dict, Tuple

__all__ = [
    "from_reference",
    "reference_module",
    "reset_warnings",
    "to_reference",
    "warn_once",
]

#: Features delegated to the reference implementation, warned about once each.
DEFAULT_MESSAGE = (
    "{feature} is not ported to the C++ core; delegating to the reference "
    "`pytket_dqc` implementation, which is slower."
)

#: Features already warned about in this process.
_WARNED: set = set()

#: ``kind`` -> (adapter module, reference submodule, class name)
_CLASS_MAP: Dict[str, Tuple[str, str, str]] = {
    "Distribution": (
        "pytket_dqc_cpp.circuits.distribution",
        "circuits",
        "Distribution",
    ),
    "HypergraphCircuit": (
        "pytket_dqc_cpp.circuits.hypergraph_circuit",
        "circuits",
        "HypergraphCircuit",
    ),
    "Placement": (
        "pytket_dqc_cpp.placement.placement",
        "placement",
        "Placement",
    ),
    "NISQNetwork": (
        "pytket_dqc_cpp.networks.nisq_network",
        "networks",
        "NISQNetwork",
    ),
}


def warn_once(feature: str, message: str | None = None, stacklevel: int = 3) -> None:
    """Emit one ``UserWarning`` per feature and process.

    :param feature: Name of the delegated feature, also used as the key.
    :param message: Warning text; defaults to :data:`DEFAULT_MESSAGE`.
    :param stacklevel: Stack level used to attribute the warning to the caller
        of the delegating API.
    """
    if feature in _WARNED:
        return
    _WARNED.add(feature)
    warnings.warn(
        message if message is not None else DEFAULT_MESSAGE.format(feature=feature),
        UserWarning,
        stacklevel=stacklevel,
    )


def reset_warnings() -> None:
    """Forget which features have been warned about (used by tests)."""
    _WARNED.clear()


def reference_module(name: str = "") -> ModuleType:
    """Import a module of the reference ``pytket_dqc`` package, lazily.

    :param name: Dotted module path relative to ``pytket_dqc`` (e.g.
        ``"allocators.routing"``).  Empty for the package itself.
    :raises ImportError: If ``pytket_dqc`` is not installed.
    """
    module_name = "pytket_dqc" if not name else f"pytket_dqc.{name}"
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "This feature is delegated to the reference `pytket_dqc` package, "
            "which could not be imported. Install it with "
            "`pip install pytket-dqc`."
        ) from exc


def _kind_of(obj: Any, kind: str | None) -> str:
    resolved = kind if kind is not None else type(obj).__name__
    if resolved not in _CLASS_MAP:
        raise TypeError(
            f"Cannot convert objects of kind {resolved!r}; expected one of "
            f"{sorted(_CLASS_MAP)}."
        )
    return resolved


def to_reference(obj: Any, kind: str | None = None) -> Any:
    """Convert an adapter object into its reference ``pytket_dqc`` counterpart.

    The conversion goes through ``to_dict()`` / ``from_dict()`` so that only the
    documented serialisation format is relied upon.

    :param obj: Adapter object (``Distribution``, ``HypergraphCircuit``,
        ``Placement`` or ``NISQNetwork``).
    :param kind: Overrides the class-name based dispatch.
    """
    _, reference_path, class_name = _CLASS_MAP[_kind_of(obj, kind)]
    reference_class = getattr(reference_module(reference_path), class_name)
    return reference_class.from_dict(obj.to_dict())


def from_reference(obj: Any, kind: str | None = None) -> Any:
    """Convert a reference ``pytket_dqc`` object into its adapter counterpart."""
    adapter_path, _, class_name = _CLASS_MAP[_kind_of(obj, kind)]
    adapter_class = getattr(importlib.import_module(adapter_path), class_name)
    return adapter_class.from_dict(obj.to_dict())
