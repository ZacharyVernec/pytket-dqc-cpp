# Pytket_DQC_CPP

Re-implementation of Pytket-DQC in C++ for speed. Mostly vibe-coded with human supervision.

Tested with AI-created tests to ensure method outputs are identical across implementations. Also tested on the QFT with human-created tests relevant to the [distributed QFT](https://github.com/MSRG/dist_qft) project.
We are very confident in the accuracy of this port and plan to use it for our research benchmarks as a drop-in replacement to pytket_dqc. However, other researchers are still advised to first do a comparison between versions on their own dataset, in case an unnoticed untested code path is present.

The reference implementation is the upstream Python package [`pytket_dqc`](https://github.com/CQCL/pytket-dqc) (Apache-2.0, commit `1ac175e`), installed in the project virtualenv at `.venv/lib/python3.11/site-packages/pytket_dqc/`. The C++ port is verified against it; the original Python code is **never modified**.

# Build and install

## Prerequisites

- CMake >= 3.20 and a C++20 compiler (GCC >= 12 or Clang >= 16).
- [uv](https://github.com/astral-sh/uv) for Python virtual environment management.
- Network access on the **first** configure only (CMake fetches Catch2 and pybind11).

First, install Boost and KaHyPar, following [these instructions steps 3 & 4](https://github.com/MSRG/dist_qft).

## Build and install the Python package

`pyproject.toml` uses `scikit-build-core`, which runs CMake for you.

```bash
export KAHYPAR_ROOT="$HOME/.local/kahypar"   # source tree or install prefix
uv sync                                       # editable install into .venv
```

`uv sync` installs the project editably: Python edits take effect immediately;
after C++ changes, rebuild with `uv sync --reinstall-package pytket-dqc-cpp`.

## Manual development build (C++ tests)

Use this when working on the C++ core or running `ctest`:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DDQC_BUILD_BINDINGS=ON -DDQC_BUILD_TESTS=ON \
      -DKAHYPAR_ROOT="$KAHYPAR_ROOT"
cmake --build build -j
```

This produces `build/dqc_cpp/_core*.so`, and the KaHyPar INI is copied next to
it so the `ini_path` default resolves automatically. The test suite still
expects this `build/` directory.

To install a manual build without pip, use
`cmake --install build --prefix <venv>/lib/python3.11/site-packages`.


## Using this package from another project

Add a path dependency in `<project-dir>/pyproject.toml`:

```toml
[project]
dependencies = ["pytket-dqc-cpp", ...]

[tool.uv.sources]
pytket-dqc-cpp = { path = "../pytket-dqc-cpp" }
```

Then:

```bash
cd /path/to/project-dir
KAHYPAR_ROOT="$HOME/.local/kahypar" uv sync
```

# Test

## Quickstart: Unified Verification Runner

The runner expects the manual development build in `build/` (see
[Manual development build](#manual-development-build-c-tests)) and the Python
`kahypar` extension installed in the environment.

To run the complete verification suite across all active tiers in a single command, run from the repository root:

```bash
./tests/python/run_all_checks.sh
```

This gate script automatically:

1. Runs all 89 native C++ Catch2 tests (`ctest --test-dir build --output-on-failure`).
2. Runs the fast Python test suite via pytest (`152 passed, 7 skipped`).
3. Runs the core differential equivalence sweep across 4 hash seeds (`check_equivalence.py`).
4. Runs the adapter differential equivalence sweep across 4 hash seeds (`check_adapter_equivalence.py`, 894 checks/seed, 0 skips).
5. Runs the QFT benchmark differential test (`check_qft_equivalence.py`), including both the Python and C++ benchmark drivers, the plots, and the field-by-field result comparison.
---

## 1. Run the C++ unit tests

These are the native Catch2 tests in `cpp/tests/` (see [`cpp/tests/README.md`](cpp/tests/README.md) for a map of all 17 test suites and 89 test cases), which test the C++ accelerator core:

```bash
ctest --test-dir build --output-on-failure
```

## 2. Confirm Python and C++ behave identically

### Generic test

Run the generic differential checker **with the project venv** to compare the C++ core algorithms (graph, hypergraph, PacMan, allocators, and refiners) directly against `pytket_dqc` and `networkx`.

```bash
.venv/bin/python3 tests/python/check_equivalence.py
```

In both pytket-dqc and pytket-dqc-cpp, set order is determined by PYTHONHASHSEED, which needs to be set to ensure reproducibility. The above test suite sweeps across PYTHONHASHSEEDS values `(0, 1, 42, 12345)` and prints `OK: hash-seed invariance across ...`. To run exactly once under a specific seed instead, set `PYTHONHASHSEED` yourself:

```bash
PYTHONHASHSEED=7 .venv/bin/python3 tests/python/check_equivalence.py
```

Each section prints `OK` and the script exits non-zero on the first mismatch.

### QFT benchmark differential test

As this project is mainly intended to be used to replace pytket-dqc for the QFT circuit, there is an additional differential test:

```bash
.venv/bin/python3 tests/python/check_qft_equivalence.py
```

It runs two benchmarks `tests/python/qft/main_*_qpus_*.py` through the reference Python package and this C++ accelerated package. They write their results to `tests/python/qft/results_*_qpus_*/` and the two `plot_*.py` scripts turn each variant's results into bar charts for easy human visualization (pass `--cpp` to plot the C++ variant).

The checker runs all four drivers under `PYTHONHASHSEED=12345` (override with `QFT_PYTHONHASHSEED`), compares the result files field by field **ignoring the two timing lines**, regenerates the plots for both variants, and exits non-zero if anything differs. Use `--skip-computation` to reuse the existing result files (the plots and the comparison still run), and `--skip-plots` to skip plotting. The generated `.txt`/`.png` files are gitignored.

## 3. Adapter suite: vendored regression tests

The upstream tests and fixtures from commit `1ac175e` of `pytket_dqc` are vendored into `tests/` (`tests/*_test.py`, `tests/test_circuits/`) with only the import root rewritten to `pytket_dqc_cpp`. In addition, `tests/` contains adapter-specific regression suites:

- `tests/rng_semantics_test.py` — validates `PyRandom` seeding/entropy behaviour, that `Random`, the refiners and `Annealing` leave Python's ambient `random` stream where the reference does (625-word state round-trip), and seeded parity.
- `tests/error_parity_test.py` — validates strict `ConstraintException` API parity (`.server`, `.v_gate`, guidance hints) against upstream.

Run the test suite **from the repository root** (fixtures are opened relative to the root) with the project venv:

```bash
.venv/bin/python -m pytest tests -m "not high_compute"
.venv/bin/python -m pytest tests -m high_compute   # slow, optional
```

The fast run executes 159 tests: 152 pass, 7 are skipped (as in the reference repository), and 7 are deselected (as they are slower; see [`tests/README.md`](tests/README.md)). 

Differences from upstream: `test_nisq_draw` is removed (drawing is out of scope)
and `test_kahypar_install` loads the INI from `pytket_dqc_cpp/allocators/`
(`importlib_resources` is no longer needed).

### (Optional) Upstream pytest suite to verify environment health

The original, unmodified upstream tests can also be run against the installed reference package (`pytket_dqc`) to verify that the environment is healthy (151 passed, 7 skipped, 7 deselected):

```bash
.venv/bin/python3 -m pytest <path-to-upstream-tests> -m "not high_compute"
```

## 4. Adapter differential check

`tests/python/check_adapter_equivalence.py` compares `pytket_dqc_cpp` against
`pytket_dqc` object by object. Run it from the repository root:

```bash
PYTHONPATH=python .venv/bin/python tests/python/check_adapter_equivalence.py
```

The script automatically executes a 4-seed sweep across `HASH_SEEDS` (`0, 1, 42, 12345`), running **894 checks per seed** (3,576 checks total) with a strict zero-skip gate (`SKIPPED[0] == 0`). Ambient `PYTHONHASHSEED` will not bypass the sweep; to run a single seed explicitly, set `DQC_ADAPTER_SINGLE_SEED=1`:

```bash
DQC_ADAPTER_SINGLE_SEED=1 PYTHONHASHSEED=7 PYTHONPATH=python .venv/bin/python tests/python/check_adapter_equivalence.py
```


# Design decisions (written by AI agent, based on human+agent planning and iteration)


## Architecture: hybrid (C++ core + Python pytket API)

- The compute-heavy DQC algorithms (hypergraph construction, distribution cost via Steiner + ALAP, PacMan packing, placement, allocators, refiners, graph tools) are ported to a self-contained C++ core library (`dqc_core`), which has **no** dependency on pytket / numpy / networkx.
- The core is exposed to Python through pybind11 with a narrow, data-only interface: only plain structs (`Hyperedge`, `Hypergraph`, serialized gates, placements, network descriptors) cross the boundary.
- A thin Python package mirrors the `pytket_dqc` public API and delegates compute to the C++ core. `pytket` is kept in Python for `Circuit`, gates, passes, and routing.

For deep-dive architectural design, bridge conversion mechanics, RNG round-tripping, and testing gate designs, see [adapter_details.md](file:///home/ubuntu/pytket-dqc-cpp-packaged/adapter_details.md).

## Exactness

- **Combinatorial results**: Given the same seed, the C++ version produces the **same combinatorial results** as Python: same hyperedges, same placements, same distributed circuits. Graph and hypergraph results are structurally identical (same nodes, edges, hyperedges in the same order).
- **Floating-point**: Compared with a tolerance; floating-point results are not guaranteed bit-identical due to standard numeric-library boundary behavior.
- **Set iteration order**: CPython's deterministic `set`/`dict` iteration order for `int`/`tuple`/`frozenset` and string keys is replicated in C++ (`PySet`, SipHash) wherever it affects combinatorial results (Steiner-tree terminal order, bipartition assignment, boundary detection, and packet ordering). Thus, when `PYTHONHASHSEED` is set, results match upstream bit-for-bit.
- **Randomness**: The RNG is **bit-exact**: Python's Mersenne Twister (MT19937 with CPython array seeding and `genrand_res53`) is ported as `dqc::PyRandom`. Algorithms are moved to C++ in their entirety to prevent splitting RNG consumption across the language boundary. Ambient RNG state is forwarded and restored around core operations (see `RNG semantics` below and [`adapter_details.md`](adapter_details.md)).

## Out of scope

- Drawing / visualisation (`draw()`, `draw_nisq_network`, `draw_server_network`; hypernetx, matplotlib, graphviz). Cosmetic only.
- PyZX equivalence checking (`utils/verification.py`). The `assert check_equivalence(...)` in `Distribution.to_pytket_circuit()` is **not** ported; it does not affect outputs.
- Logging / `__str__` output fidelity (only function inputs/outputs matter).

# Python adapter package: `pytket_dqc_cpp`

The adapter lives in `python/pytket_dqc_cpp/` and mirrors the public Python API of
`pytket_dqc`: the same module tree (`circuits`, `placement`, `networks`, `packing`,
`allocators`, `refiners`, `distributors`, `utils`), class names, signatures,
defaults, attributes, `to_dict`/`from_dict` JSON formats and error messages. All
subpackages are eagerly imported from the top-level `__init__`, so a single import
gives you the whole API.

```python
import pytket_dqc_cpp as pytket_dqc
from pytket import Circuit

circ = Circuit(2).H(0).CU1(0.5, 0, 1)
pytket_dqc.DQCPass().apply(circ)       # transpiles in place; returns bool
network = pytket_dqc.NISQNetwork([[0, 1]], {0: [0], 1: [1]})
distribution = pytket_dqc.allocators.Random().allocate(circ, network)

pytket_dqc.HypergraphCircuit          # top-level names as upstream
pytket_dqc.allocators.Random          # subpackages are pre-imported
pytket_dqc_cpp.circuits.Hyperedge     # full package path also works
```

There is **no global `pytket_dqc` alias**: the reference package stays installed
side by side (it is still the oracle, and a runtime dependency for the delegated
features below). Existing code that imports the reference must change its import
root, e.g.

```python
from pytket_dqc.circuits import HypergraphCircuit   # before
from pytket_dqc_cpp.circuits import HypergraphCircuit  # after
```

Install it with `uv sync` (editable) or `uv pip install .`; the compiled
extension is installed inside the package as `pytket_dqc_cpp/_core*.so`, so
`import pytket_dqc_cpp` works from any directory. For an uninstalled source
tree, put `python/` on `sys.path`; `pytket_dqc_cpp._core` then locates the
extension in `build/dqc_cpp/` (or `DQC_CORE_DIR`), or raises an `ImportError`
quoting the CMake command if it has not been built.

## Runtime dependency on the reference package

`pytket-dqc` remains a runtime dependency. It is imported **lazily** (inside the
delegated functions only, never at module scope) so that plain imports and the
core-backed code paths do not touch it.

These features are not ported to the C++ core and delegate to `pytket_dqc` at
call time, emitting **one `UserWarning` per feature per process**:

| Delegated feature                         | Why                                                                      |
| ----------------------------------------- | ------------------------------------------------------------------------ |
| `allocators.Routing`                    | needs tket `PlacementPass` / `RoutingPass` / `DecomposeSwapsToCXs` |
| `RandomHypergraphCircuit`               | needs `scipy.stats.unitary_group.rvs(4)`                               |
| `CyclicHypergraphCircuit`               | upstream circuit generator                                               |
| `RegularGraphHypergraphCircuit`         | needs `networkx.random_regular_graph`                                  |
| `utils.check_equivalence` (`to_pyzx`) | PyZX equivalence checking                                                |

They are correct but slower than the core-backed paths.

## Core-backed vs Python

| Area                                                                                                                                                                        | Implementation                                                                                                                                                 |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `HypergraphCircuit` construction, subcircuits, H-embedding queries, `distribution_cost`                                                                                 | C++ core (`_core.HypergraphCircuit`)                                                                                                                         |
| `Distribution.cost`, `hyperedge_cost`, `to_pytket_circuit`, validation helpers                                                                                        | C++ core (`_core.Distribution`)                                                                                                                              |
| `steiner_tree`, `direct_from_origin`, `to_euler_with_two_hadamards`, `ebit_cost`, `ebit_memory_required`, `all_cu1_local`, `get_server_id`, `is_link_qubit` | C++ core                                                                                                                                                       |
| `PacMan` (packets, neighbouring/hopping/merged, conflict graph)                                                                                                           | C++ core (`_core.PacMan`)                                                                                                                                    |
| `Random`, `Ordered`, `Brute`, `Annealing`, `HypergraphPartitioning`                                                                     | C++ core allocators                                                                                                                                            |
| `EagerHTypeMerge`, `NeighbouringDTypeMerge`, `IntertwinedDTypeMerge`, `BoundaryReallocation`                                                                        | C++ core refiners                                                                                                                                              |
| `DetachedGates`                                                                                                                                                           | upstream Python `fixed_vertices` + `BoundaryReallocation` (the core entry point segfaults on hyperedges with no gate vertices; see "Known approximations") |
| `GainManager`, `VertexCover` / `get_min_covers`, `DQCPass`, placement/network/hypergraph data structures                                                           | pure Python (hot primitives go to the core)                                                                                                                    |

`Annealing.allocate` runs the initial placement allocator in Python, then hands that
placement to `_core.annealing_from_placement_with_state`, which runs the annealing
iterations in the compiled core. Python's ambient RNG state is forwarded
(`random.getstate()`) and the post-loop state is installed back with
`random.setstate`, so the stream after the call matches the reference.
Its default initial placement method is `Random()`. Because the default `Random()`
initializer calls `random.seed(None)` (OS entropy) when unseeded, calling
`Annealing(seed=...)` with the default initializer reproduces upstream's
non-determinism and emits a warning. For guaranteed bit-for-bit cross-library reproducibility against the
reference, pass any deterministic initial placement method from the set
`{Ordered(), HypergraphPartitioning(), Brute()}`. For reproducible randomized
initial placements, pass an explicitly seeded wrapper around `Random`.

### RNG semantics (ambient parity with upstream)

Randomness crosses the Python/C++ boundary as the 625-word MT19937 state, and the
core returns the post-call state, which `_bridge.call_with_python_rng` installs with
`random.setstate`. Python-visible RNG semantics therefore match the reference:

- **`Random.allocate` / `Annealing.allocate`** call `random.seed(seed)` first, so
  `seed=None` reseeds the ambient generator from OS entropy (upstream's behaviour,
  including the default-`Random()` annealing caveat above) and an explicit seed
  resets the stream.
- **`BoundaryReallocation` / `DetachedGates`** reseed only when an explicit seed is
  passed; unseeded calls continue whatever ambient stream the caller left, exactly
  as upstream does.
- **After every such call**, `random.getstate()` equals the reference's state for the
  same seed and inputs, so code that mixes adapter calls with module-level `random`
  observes identical streams.
- **Differential tests** compare both the placement and the post-call RNG state.
  Unseeded runs reseed from independent OS entropy in each library, so result
  comparisons still use explicit seeds.

`HypergraphPartitioning` is excluded: KaHyPar uses its own RNG and upstream never
touches Python's `random`.

## Frozen state

After a core refiner call the Python state is synchronised immediately
(`_bridge.sync_from_core`), so `distribution.circuit.hyperedge_list`,
`distribution.placement.placement` and the placement key order stay consistent
with what the core computed.

## KaHyPar (hypergraph partitioning backend)

The `HypergraphPartitioning` allocator is backed by KaHyPar (GPL-3.0), called
through its C facade (`libkahypar.h`). Unlike upstream `pytket-dqc`, which binds
against the Python `kahypar` package, `pytket_dqc_cpp` links directly against
KaHyPar's C++ facade (`lib/libkahypar.cc`) or a prebuilt `libkahypar.so`.

The build takes one path input, `KAHYPAR_ROOT` (or the legacy
`-DDQC_KAHYPAR_SOURCE_DIR`), and auto-detects the layout:

- **SOURCE**: `<root>/lib/libkahypar.cc` exists. CMake compiles the
  single-translation-unit facade into `dqc_kahypar` (C++17, since KaHyPar uses
  `requires` as an identifier) and links it into the core. Requires Boost
  headers and `libboost_program_options`.
- **INSTALLED**: `<root>/include/libkahypar.h` exists. CMake links the prebuilt
  `<root>/lib/libkahypar.so`; Boost headers are not needed at build time, but
  `libboost_program_options.so` is needed at runtime.
- **PKGCONFIG**: no root given, but `pkg-config` finds `libkahypar`.
- **OFF**: no KaHyPar. The allocator is unavailable; configure fails unless
  `-DDQC_REQUIRE_KAHYPAR=OFF` is passed.

The bundled INI config is shipped as
`python/dqc_cpp/km1_kKaHyPar_sea20.ini`; CMake copies it next to the compiled
`_core` module, and the install rules place it inside `pytket_dqc_cpp/`. The
`ini_path` argument of `_core.hypergraph_partitioning_allocate` defaults to it;
pass your own to override.

The **Python** `kahypar` extension is a separate artifact. It is required by the
reference `pytket_dqc` (which imports it from `pytket_dqc.allocators`) and must
be installed in every venv that runs the reference, the differential tests, or
`dist_qft`. `pytket_dqc_cpp` itself does not import it.

The reference Python `kahypar` module and the C facade issue the same
`PartitionerFacade().partition` call with the same `Context`, so placements are
bit-identical for the same seed (verified: `check_hypergraph_partitioning`).
`suppressOutput(True)` is mirrored by silencing KaHyPar's stdout during the
partition call (`detail::ScopedStdoutSilencer`).

## Known approximations

- `Hypergraph.vertex_neighbours` uses `std::set` (ascending) as a stand-in for CPython int-set iteration order. This is **not observable**: `vertex_neighbours` is only ever used for membership and in `get_boundary` (where iteration order does not affect the returned vertex list), never iterated in an order-sensitive way.
- `PacMan.are_hoppable_packets` faithfully reproduces a reference bug: the "reverse the second 1q-op list and retry" fallback is dead code because the guard compares `op.params` (a list) to `0`, which is always `False`.
- `PyRandom::seed` covers integer seeds `<= 2^64-1`; larger seeds (and arbitrary-int keys) are expressible via `seed_words`.
- `DetachedGates` in the Python adapter delegates `fixed_vertices` calculation to Python and routes to `BoundaryReallocation`, because the core `detached_gates_refine` entry point segfaults on hyperedges with no gate vertices (such as degenerate hyperedges containing only qubit vertices).
- `ConstraintException` parity: when hardware communication memory capacity or placement constraints are violated during `Distribution.to_pytket_circuit()`, the pybind11 layer directly translates `dqc::ConstraintException` into `pytket_dqc_cpp.utils.circuit_analysis.ConstraintException`, attaching `.server`, `.v_gate`, and the remediation hint advising `to_pytket_circuit(allow_update=True)`.


