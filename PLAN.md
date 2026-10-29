# Implementation Plan: Packaging & Build System

## Goal

Implement a standard Python build step and packaging configuration for `pytket_dqc_cpp` so the package builds and installs without requiring manual `cmake` commands and `sys.path` hacks.

## Context & Requirements

- **Current state:** Development requires manually compiling the C++ core via CMake (`cmake -S . -B build -DDQC_BUILD_BINDINGS=ON && cmake --build build`) and running with `PYTHONPATH=python` while `_core.py` falls back to loading `build/dqc_cpp/_core*.so`. `pyproject.toml` currently lacks a `[build-system]` definition.
- **Target state:** Standard `pip install .`, `uv pip install -e .`, or `uv build` / `pip wheel .` builds the C++ pybind11 extension and installs it directly inside the `pytket_dqc_cpp` package layout (`pytket_dqc_cpp/_core*.so`), eliminating the build-directory fallback.
- **Package data:** Ensure the KaHyPar configuration file (`km1_kKaHyPar_sea20.ini` in `pytket_dqc_cpp/allocators/`) is properly packaged into wheels and editable installs.
- **Scope constraint:** Build configuration changes in `pyproject.toml` and build integration in `CMakeLists.txt` are explicitly permitted for this packaging milestone.

## Tasks

- [ ] Select and configure a Python C++ build backend in `pyproject.toml` (e.g. `scikit-build-core` or `setuptools` with CMake integration).
- [ ] Configure extension building to place the compiled `_core` module directly within `pytket_dqc_cpp`.
- [ ] Ensure package data (`km1_kKaHyPar_sea20.ini`) is included in wheel distributions and installed package trees.
- [ ] Verify that `_core.py` successfully loads via its installed layout (`from . import _core`) without relying on the `build/dqc_cpp` fallback.
- [ ] Verify wheel building (`uv build`) and test suite execution against an installed wheel environment.
