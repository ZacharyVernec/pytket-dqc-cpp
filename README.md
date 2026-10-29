# Pytket_DQC_CPP

Re-implementation of Pytket-DQC in C++ for speed. Mostly vibe-coded with human supervision.

Tested with AI-created tests to ensure method outputs are identical across implementations. Also tested on the QFT with human-created tests relevant to the [distributed QFT](https://github.com/MSRG/dist_qft) project.

The reference implementation is the upstream Python package `pytket_dqc` (Apache-2.0), installed in the project virtualenv at `.venv/lib/python3.11/site-packages/pytket_dqc/`. The C++ port is verified against it; the original Python code is **never modified**.

For deep-dive architectural design, bridge conversion mechanics, RNG round-tripping, and testing gate designs, see [adapter_details.md](file:///home/ubuntu/pytket-dqc-cpp-packaged/adapter_details.md).

# Design decisions (written by AI agent, based on human+agent planning and iteration)

## Architecture: hybrid (C++ core + Python pytket API)

- The compute-heavy DQC algorithms (hypergraph construction, distribution cost via Steiner + ALAP, PacMan packing, placement, allocators, refiners, graph tools) are ported to a self-contained C++ core library (`dqc_core`), which has **no** dependency on pytket / numpy / networkx.
- The core is exposed to Python through pybind11 with a narrow, data-only interface: only plain structs (`Hyperedge`, `Hypergraph`, serialized gates, placements, network descriptors) cross the boundary.
- A thin Python package mirrors the `pytket_dqc` public API and delegates compute to the C++ core. `pytket` is kept in Python for `Circuit`, gates, passes, and routing.

## Fidelity bar: structural exactness (NOT bit-identical)

- Given the same seed, the C++ version must produce the **same combinatorial results** as Python: same hyperedges, same placements, same distributed circuits.
- Floating-point is compared with a tolerance; floating-point results are **not** guaranteed bit-identical. The known numeric-library boundary is `scipy.stats.unitary_group.rvs` / LAPACK QR (see deferrals).
- CPython's deterministic `set`/`dict` iteration order for `int`/`tuple`/`frozenset` keys must be replicated where it affects combinatorial results (Steiner-tree terminal order, bipartition assignment, boundary detection, etc.).

## Randomness

- Reproducibility requires reimplementing the exact RNG used by Python:
  - Python `random` module = Mersenne Twister (MT19937) with CPython seeding (`init_by_array` over the little-endian 32-bit words of `|seed|`) and `random()` = `genrand_res53`. Ported as `dqc::PyRandom` (bit-exact).
  - `numpy.random.RandomState` (legacy MT19937) and `scipy.stats.unitary_group` are **not** reproduced in v1 (see deferrals).
- Rule: move **whole** algorithms to C++; never split one algorithm's RNG consumption across the Python/C++ boundary.
- `seed=None` is intentionally non-deterministic (no cross-language match is defined; unseeded `dqc::PyRandom` initializes from OS entropy `std::random_device`). In Python refiner calls, unseeded execution continues whatever ambient stream the caller left, matching upstream behavior. Negative seeds map to their absolute value (CPython semantics).

## Implemented (C++ core, v1)

- `PyRandom` (bit-exact MT19937), CPython int/tuple/frozenset hashing + `PySet` (CPython set iteration order).
- `Graph` + networkx algorithms: `dijkstra_multisource`, `bidirectional_dijkstra_path`, `minimum_spanning_edges`, `steiner_tree` (Mehlhorn), `direct_from_origin`, bipartite `color/sets/subgraph/maximum_matching/to_vertex_cover`.
- `Hypergraph`/`Hyperedge`, `HypergraphCircuit` (`from_circuit`, subcircuit + H-embedding helpers), `Placement`, gateset (`to_euler_with_two_hadamards`).
- `Distribution`: `hyperedge_cost`, `distribution_cost`, `is_valid`, `non_local_gate_list/count`, `detached_gate_list`. Constructor overload `Distribution(circuit, placement, network)` mirrors the reference `Distribution(dist_circ, placement, network)` (server graph = `get_server_nx()`, qubits + ebit memory taken from the `NISQNetwork`).
- `PacMan` (packets, neighbouring/hopping/merged, embedded/conflict hoppings, hypergraph reconstruction, merged/conflict graphs).
- `GainManager` (steiner cache, hyperedge cost map, split/merge/move gains, occupancy).
- Allocators: `Random`, `Ordered`, `Brute`, `Annealing`, `HypergraphPartitioning` (KaHyPar, via the C facade `libkahypar.h` — see `cpp/include/dqc/hypergraph_partitioning.hpp`). Mirrors the reference `Allocator().allocate(circ, network)`: each `_core` allocator takes a `_core.NISQNetwork` and returns a fully-populated `Distribution` (e.g. `_core.annealing_allocate(dist_circ, network, seed=...)`).
- Refiners: `VertexCover` (networkx path), `EagerHTypeMerge`, `NeighbouringDTypeMerge`, `IntertwinedDTypeMerge`, `BoundaryReallocation`, `DetachedGates`, plus the `RepeatRefiner` / `SequenceRefiner` composition classes (`cpp/include/dqc/refiners/composition.hpp`), which wrap any refiner callable.
- `NISQNetwork` / `ServerNetwork` (`cpp/include/dqc/network.hpp`): network container with validation, `get_server_list`, `get_qubit_list`, `can_implement`, `is_placement`, `get_server_nx` / `get_nisq_nx`, `to_dict` / `from_dict` (the data + validation are ported; pytket `Architecture`/`Routing` and drawing stay in Python).
- `circuit_analysis`: `ebit_cost`, `ebit_memory_required`, `all_cu1_local`, `get_server_id`, `is_link_qubit`.

## Out of scope

- Drawing / visualisation (`draw()`, `draw_nisq_network`, `draw_server_network`; hypernetx, matplotlib, graphviz). Cosmetic only.
- PyZX equivalence checking (`utils/verification.py`). The `assert check_equivalence(...)` in `Distribution.to_pytket_circuit()` is **not** ported; it does not affect outputs. `pytket_dqc_cpp.utils.check_equivalence` / `to_pyzx` delegate to the reference implementation and emit one `UserWarning`; PyZX remains a Python-side test oracle only.
- Logging / `__str__` output fidelity (only function inputs/outputs matter).

## Deferred (later phases)

- `RandomHypergraphCircuit` — requires `scipy.stats.unitary_group.rvs(4)` (numpy `RandomState.normal` ziggurat + LAPACK QR). `CyclicHypergraphCircuit` and `RegularGraphHypergraphCircuit` are circuit *generators* that construct a `pytket.Circuit` and apply `DQCPass`; they stay in the Python adapter (their hypergraph construction from the resulting circuit is ported). `RegularGraphHypergraphCircuit` additionally needs `networkx.random_regular_graph`.
- `Routing` allocator (uses tket `PlacementPass`/`RoutingPass`/`DecomposeSwapsToCXs`) — stays in Python for v1.
- `numpy.random.RandomState` reproduction — only needed by the deferred pieces above.

All of the deferred pieces above are still available through the Python adapter: the generator classes and the `Routing` allocator delegate to the reference `pytket_dqc` (with a `UserWarning`) instead of using the core.

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

circ = pytket_dqc.DQCPass().apply(Circuit(2).H(0).CU1(0.5, 0, 1))
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

Installing/using it means putting `python/` on `sys.path`;
`pytket_dqc_cpp._core` then locates the compiled extension in
`build/dqc_cpp/` automatically, or raises an `ImportError` quoting the CMake
command if it has not been built.

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

## Drawing

Drawing is **out of scope**. `draw`, `draw_nisq_network`, `draw_server_network`
(and the hypernetx / matplotlib / graphviz dependencies) are not implemented, and
upstream `test_nisq_draw` is not part of the vendored suite.

## KaHyPar (hypergraph partitioning backend)

The `HypergraphPartitioning` allocator is backed by KaHyPar (GPL-3.0), called
through its C facade (`libkahypar.h`). CMake compiles the single-translation-unit
facade (`lib/libkahypar.cc`) into `dqc_kahypar` (C++17, since KaHyPar uses
`requires` as an identifier) and links it into the core. It needs:

- the KaHyPar source tree (default `/home/ubuntu/kahypar`, override with
  `-DDQC_KAHYPAR_SOURCE_DIR=...`),
- Boost headers + `libboost_program_options` (for `parseIniToContext`),
- the bundled INI config shipped as `python/dqc_cpp/km1_kKaHyPar_sea20.ini` and
  copied next to the compiled `_core` module. The `ini_path` argument of
  `_core.hypergraph_partitioning_allocate` defaults to it; pass your own to
  override.

The reference Python `kahypar` module and the C facade issue the same
`PartitionerFacade().partition` call with the same `Context`, so placements are
bit-identical for the same seed (verified: `check_hypergraph_partitioning`).
`suppressOutput(True)` is mirrored by silencing KaHyPar's stdout during the
partition call (`detail::ScopedStdoutSilencer`).

## Testing strategy

- **Unit (C++)**: Catch2 tests mirroring the upstream `tests/` suite, feeding data directly.
- **Unit (adapter)**: pytest for the thin Python layer (pytket ↔ C++ data mapping).
- **Integration / differential**: run the unmodified upstream pytest suite against both `pytket_dqc` and the C++-backed package and assert structural equality.
- **Fuzzing**: random circuits + networks + seeds; diff placements, costs, output circuits.
- **Coverage**: gcov/llvm-cov for C++, coverage.py for the adapter, plus a Python→C++ line-mapping table.

## Environment pins (fidelity targets)

- pytket 2.16.0, networkx 3.6.1, numpy 2.4.6, scipy 1.17.1, Python 3.11, KaHyPar 1.3.2 (upstream CI pin).

## Known approximations

- `Hypergraph.vertex_neighbours` uses `std::set` (ascending) as a stand-in for CPython int-set iteration order. This is **not observable**: `vertex_neighbours` is only ever used for membership and in `get_boundary` (where iteration order does not affect the returned vertex list), never iterated in an order-sensitive way.
- `PacMan.are_hoppable_packets` faithfully reproduces a reference bug: the "reverse the second 1q-op list and retry" fallback is dead code because the guard compares `op.params` (a list) to `0`, which is always `False`.
- `PyRandom::seed` covers integer seeds `<= 2^64-1`; larger seeds (and arbitrary-int keys) are expressible via `seed_words`.
- `DetachedGates` in the Python adapter delegates `fixed_vertices` calculation to Python and routes to `BoundaryReallocation`, because the core `detached_gates_refine` entry point segfaults on hyperedges with no gate vertices (such as degenerate hyperedges containing only qubit vertices).
- `ConstraintException` parity: when hardware communication memory capacity or placement constraints are violated during `Distribution.to_pytket_circuit()`, the pybind11 layer directly translates `dqc::ConstraintException` into `pytket_dqc_cpp.utils.circuit_analysis.ConstraintException`, attaching `.server`, `.v_gate`, and the remediation hint advising `to_pytket_circuit(allow_update=True)`.

### Hash-seed exactness (PYTHONHASHSEED)

The reference (`pytket_dqc` + `networkx`) orders sets of `Packet` / `MergedPacket` / `HoppingPacket` objects by their CPython hash, where `Packet.__hash__ == hash("P<i>")` is a SipHash keyed by the `PYTHONHASHSEED`-derived secret. The C++ core reproduces this exactly:

- `dqc::hash.hpp` implements SipHash-1-3 plus the CPython `PYTHONHASHSEED` secret derivation (`lcg_urandom`), and CPython `int`/`tuple`/`frozenset`/`str` hashing (including the `-1 → -2` normalisation).
- `dqc::py_set.hpp` (`PySet`) reproduces CPython `set`'s hash-table layout — probe sequences, dummy-slot reuse ("last dummy wins"), and resize policy — so set iteration order matches CPython bit-for-bit.
- `dqc::bipartite.hpp` reproduces `networkx`'s `connected_components` (BFS), `bipartite.color`/`sets`, `hopcroft_karp_matching`, `to_vertex_cover` (König), and `G.subgraph` view node ordering, so the vertex-cover refiner chooses the *same* minimum cover as the reference.

As a result, when `PYTHONHASHSEED` is set (e.g. `0`, `1`, `42`), the C++ core and the reference produce **identical** results, including which minimum vertex cover is selected and the resulting placement/hypergraph. When `PYTHONHASHSEED` is unset, both the reference and the C++ core draw independent random hash secrets, so their (individually deterministic-per-process) set orderings differ — exactly as two separate runs of the reference would.

# Build & test

## Prerequisites

- CMake >= 3.20 and a C++20 compiler (GCC >= 12 or Clang >= 16).
- Python 3.11 with [uv](https://github.com/astral-sh/uv). The project virtual environment at `.venv/` can be synced using `uv sync` (locked in `uv.lock`). It contains `pytest`, `kahypar`, `pyzx`, and the reference `pytket_dqc` and `networkx` packages (the test oracles).
- Network access on the **first** configure only (CMake fetches Catch2 and pybind11).
- For the KaHyPar-backed `HypergraphPartitioning` allocator: the KaHyPar source tree (default `/home/ubuntu/kahypar`) and Boost (`libboost_program_options`). If `DQC_KAHYPAR_SOURCE_DIR` does not contain `lib/libkahypar.cc`, the allocator is skipped (the `_core.hypergraph_partitioning_*` functions and the KaHyPar tests are compiled out).

## Quickstart: Unified Verification Runner

To run the complete verification suite across all active tiers in a single command, run from the repository root:

```bash
./tests/python/run_gates.sh
```

This gate script automatically:

1. Builds the C++ core with Python bindings enabled (`-DDQC_BUILD_BINDINGS=ON`).
2. Runs all 89 native C++ Catch2 tests (`ctest --test-dir build --output-on-failure`).
3. Runs the fast Python test suite via pytest (`151 passed, 7 skipped`).
4. Runs the core differential equivalence sweep across 4 hash seeds (`check_equivalence.py`).
5. Runs the adapter differential equivalence sweep across 4 hash seeds (`check_adapter_equivalence.py`, 894 checks/seed, 0 skips).

---

## 1. Build and run the C++ unit tests

From the repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

These are the native Catch2 tests in `cpp/tests/`:

- `test_rng.cpp` — `PyRandom` is bit-exact against CPython `random` (golden vectors for seeds `0, 1, 12345, 2^32, 2^32+1, 2^63+5, 2^64`), verifies non-deterministic unseeded initialization via OS entropy, and verifies exact 625-word MT19937 `get_state` / `set_state` round-tripping.
- `test_hypergraph.cpp` — hypergraph add/remove/merge/split/boundary, `is_valid`, `is_placement`.
- `test_graph.cpp` — `steiner_tree`, `direct_from_origin`, Dijkstra vs networkx 3.6.1 golden vectors; node/edge iteration order, weights, self-loops, `degree`, `remove_node`, `subgraph`, `edge_subgraph`.
- `test_gateset.cpp` — `to_euler_with_two_hadamards` (all branch cases + error cases).
- `test_pacman.cpp` — packet construction, neighbouring/hopping/merged identification, packet-navigation helpers, connectivity, embedding, split packets.
- `test_allocators.cpp` — `Random` (seeded determinism), `Ordered`, `Brute`, `Annealing` (validity + seeded-initial-placement determinism), and `can_implement` errors.
- `test_refiners.cpp` — `DetachedGates` (golden seeded result), `BoundaryReallocation` (determinism), `VertexCover` (valid refined distribution).
- `test_distribution.cpp` — `cost`, `is_valid`, `non_local_gate_list/count`, `detached_gate_list/count`.
- `test_network.cpp` — `NISQNetwork` construction, accessors, validation errors, graph structure, equality.
- `test_composition.cpp` — `RepeatRefiner` / `SequenceRefiner` do-while / or-fold semantics and nesting.
- `test_to_pytket_circuit.cpp` — circuit emission to pytket `Circuit`, gate translation, ebit memory validation, and constraint error triggers.
- `test_hypergraph_partitioning.cpp` — KaHyPar placement validity, capacity, determinism, `can_implement` error (only built when KaHyPar is available).
- `test_bipartite.cpp` — bipartite graph coloring, independent sets, Hopcroft-Karp matching, and minimum vertex cover.
- `test_pyset.cpp` — `PySet` probe sequences, dummy-slot reuse, resize policy, reproducing CPython `set` iteration order.
- `test_circuit_analysis.cpp` — command inspection, ebit cost/memory requirements, CU1 locality checks.
- `test_placement.cpp` — `Placement` mapping validity, server assignment, serialization.
- `test_hypergraph_circuit.cpp` — hypergraph circuit construction, gate indexing, boundary tracking.

## 2. Build the Python bindings

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DDQC_BUILD_BINDINGS=ON
cmake --build build -j
```

This produces `build/dqc_cpp/_core*.so`, a pybind11 module exposing `PyRandom`, `Hypergraph`, `HypergraphCircuit`, `Graph`, `BipartiteGraph`, `Placement`, `PacMan`, `Distribution`, `GainManager`, `NISQNetwork`, `RepeatRefiner`, `SequenceRefiner`, `steiner_tree`, `direct_from_origin`, `to_euler_with_two_hadamards`, allocators (`Random`, `Ordered`, `Brute`, `Annealing`, `HypergraphPartitioning`) and the refiner functions. The KaHyPar INI is copied next to the module, so the `ini_path` default resolves automatically.

## 3. Confirm Python and C++ behave identically

Run the differential checker **with the project venv** (it compares the C++ core against the reference `pytket_dqc` / `networkx` / stdlib `random` in a single process):

```bash
.venv/bin/python3 tests/python/check_equivalence.py
```

Hash-seed invariance is verified **automatically**: when run without an explicit `PYTHONHASHSEED`, the script re-executes the entire suite once per seed in `HASH_SEEDS` (`0, 1, 42, 12345`) and prints `OK: hash-seed invariance across ...`. To run exactly once under a specific seed instead, set `PYTHONHASHSEED` yourself:

```bash
PYTHONHASHSEED=7 .venv/bin/python3 tests/python/check_equivalence.py
```

Each section prints `OK` and the script exits non-zero on the first mismatch. It checks:

- `PyRandom` vs `random` — bit-exact `random()`, `randrange`, `choice`, `uniform` for several seeds.
- `steiner_tree` vs `networkx.algorithms.approximation.steiner_tree` — exact node/edge order, on fixed and random graphs.
- `direct_from_origin` vs `pytket_dqc.utils.graph_tools.direct_from_origin`.
- `Hypergraph` / `HypergraphCircuit` vs the reference — vertex/hyperedge lists; plus `split_hyperedge`, `remove_hyperedge`, `merge_hyperedge`, `is_valid`, `is_placement`, `get_boundary`, `weight_one_predicate`, `kahypar_hyperedges`.
- `Graph` vs `networkx.Graph` — nodes/edges/weights/`has_edge`/`degree`, including self-loops and `remove_node`.
- `BipartiteGraph` vs `networkx.algorithms.bipartite` — `color`, `sets`, `hopcroft_karp_matching`, `to_vertex_cover`, `subgraph`.
- `to_euler_with_two_hadamards` vs `pytket_dqc.utils.gateset` for every decomposition branch.
- `Distribution` — `cost`, `is_valid`, `non_local_gate_list/count`, `detached_gate_list/count`.
- `PacMan` — all packet internals: navigation helpers, `are_neighbouring/hoppable`, connected/containing/split packets, embedded packets, `get_hopping_packets_within`, conflicts.
- Allocators (`Random` seeded, `Ordered`, `Brute`) and all refiners (`VertexCover`, `EagerHTypeMerge`, `Neighbouring/IntertwinedDTypeMerge`, `BoundaryReallocation`, `DetachedGates`). Each `_core` allocator returns a full `Distribution` (placement + cost + validity) compared against the reference `Distribution`.
- `Annealing` — the core's `acceptance_criterion` matches the reference helper exactly, and `annealing_allocate` reproduces the **full reference `allocate` loop bit-exactly** via the seeded-initial-placement workaround: passing an `initial_place_method` that wraps `Random` with a fixed seed (so the initial placement no longer discards the seed) makes the reference deterministic, and the C++ `annealing_allocate(dist_circ, network, ..., initial_place_seed=SEED)` reproduces it bit-for-bit (placement + final cost).
- `NISQNetwork` vs `pytket_dqc.networks.NISQNetwork` — accessors, `to_dict`/`from_dict` round-trip, graph structure, validation errors.
- `HypergraphPartitioning` vs `pytket_dqc.allocators.HypergraphPartitioning` — **bit-identical placements** (KaHyPar initial partition + `make_valid`) for QFT circuits across partition counts and seeds.
- `RepeatRefiner` / `SequenceRefiner` — the `_core` classes reproduce the manual do-while / or-fold loops already validated by the other sections.

## What "identical" means here

- The RNG is **bit-exact**: same doubles, same choices, same randranges.
- Graph/hypergraph results are **structurally identical**: same nodes, same edges, same hyperedges, in the same order.
- Set iteration order is **replicated** (CPython `set` layout and SipHash string hashes), so order-sensitive paths — including the vertex-cover refiner's choice of minimum cover — match the reference whenever `PYTHONHASHSEED` is set (the sweep above confirms this for multiple seeds).

## 4. QFT benchmark differential test

The four benchmark drivers in `tests/python/qft/` run the same QFT workloads through the reference Python package (`main_few_qpus_py.py`, `main_many_qpus_py.py`) and through the C++ adapter package (`main_few_qpus_cpp.py`, `main_many_qpus_cpp.py` using `pytket_dqc_cpp`). They write their results to `tests/python/qft/results_few_qpus_py/`, `results_few_qpus_cpp/`, `results_many_qpus_py/` and `results_many_qpus_cpp/`; the two `plot_*.py` scripts turn each variant's results into bar charts (pass `--cpp` to plot the C++ variant).

```bash
.venv/bin/python3 tests/python/check_qft_equivalence.py
```

The checker runs all four drivers under `PYTHONHASHSEED=12345` (override with `QFT_PYTHONHASHSEED`), compares the result files field by field **ignoring the two timing lines**, regenerates the plots for both variants, and exits non-zero if anything differs. Use `--skip-computation` to reuse the existing result files (the plots and the comparison still run), and `--skip-plots` to skip plotting. The generated `.txt`/`.png` files are gitignored.

## 5. Adapter suite: vendored & regression tests

The upstream tests and fixtures from commit `1ac175e` are vendored into `tests/`
(`tests/*_test.py`, `tests/test_circuits/`) with only the import root rewritten to
`pytket_dqc_cpp`. In addition, `tests/` contains adapter-specific regression suites:

- `tests/rng_semantics_test.py` — validates `PyRandom` seeding/entropy behaviour, that `Random`, the refiners and `Annealing` leave Python's ambient `random` stream where the reference does (625-word state round-trip), and seeded parity.
- `tests/error_parity_test.py` — validates strict `ConstraintException` API parity (`.server`, `.v_gate`, guidance hints) against upstream.

Run the test suite **from the repository root** (fixtures are opened relative to the root) with the project venv:

```bash
PYTHONPATH=python .venv/bin/python -m pytest tests -m "not high_compute"
PYTHONPATH=python .venv/bin/python -m pytest tests -m high_compute   # slow, optional
```

The fast run executes 158 tests: 151 pass and 7 are skipped (matching the exact 7 upstream skips documented in `tests/README.md`).

### Skipped vs deselected (the `151 passed, 7 skipped, 7 deselected` line)

`pytest tests -m "not high_compute"` collects 165 tests and leaves 14 of them unrun,
in two **disjoint** groups:

- **7 skipped** — unconditional `@pytest.mark.skip(reason=...)` decorators copied
  verbatim from upstream (CX / `QControlBox` / teleportation / removed-function /
  drawing-era features that are disabled upstream too). They are listed in
  `tests/README.md` and can never run without editing the source.
- **7 deselected** — tests marked `@pytest.mark.high_compute` and filtered out by
  the `-m "not high_compute"` expression *before* selection, so they are not run at
  all: `test_to_pytket_circuit_with_pauli_circ`,
  `test_to_pytket_circuit_pr78_bug`, `test_embedding_detached`
  (`tests/circuits_test.py`) and `test_vertex_cover_refiner_pauli_circ`,
  `test_vertex_cover_refiner_random_circ`, `test_vertex_cover_embedding_boundary_failure`,
  `test_boundary_reallocation_refiner_pauli_circ` (`tests/refiner_test.py`). These are
  valid heavyweight integration tests (10-qubit fixtures, KaHyPar + refiners, and a
  PyZX `check_equivalence` round-trip); run them with `-m high_compute`.

So `151 + 7 + 7 = 165`: the fast gate never exercises the `high_compute` group, and
a regression there is only caught by the optional slow run.

`pytest.ini` pins `testpaths = tests`, `python_files = *_test.py test_*.py` (so the
`tests/python/check_*.py` differential scripts are never collected) and declares the
`high_compute` marker. `tests/conftest.py` puts `python/` on `sys.path`, so the
`PYTHONPATH=python` prefix is optional in practice.

Differences from upstream: `test_nisq_draw` is removed (drawing is out of scope)
and `test_kahypar_install` loads the INI from `pytket_dqc_cpp/allocators/`
(`importlib_resources` is no longer needed).

## 6. Adapter differential check

`tests/python/check_adapter_equivalence.py` compares `pytket_dqc_cpp` against
`pytket_dqc` object by object. Run it from the repository root:

```bash
PYTHONPATH=python .venv/bin/python tests/python/check_adapter_equivalence.py
```

The script automatically executes a 4-seed sweep across `HASH_SEEDS` (`0, 1, 42, 12345`), running **894 checks per seed** (3,576 checks total) with a strict zero-skip gate (`SKIPPED[0] == 0`). Ambient `PYTHONHASHSEED` will not bypass the sweep; to run a single seed explicitly, set `DQC_ADAPTER_SINGLE_SEED=1`:

```bash
DQC_ADAPTER_SINGLE_SEED=1 PYTHONHASHSEED=7 PYTHONPATH=python .venv/bin/python tests/python/check_adapter_equivalence.py
```

It covers `HypergraphCircuit` vertices/hyperedges/serialisation, `Distribution.cost`,
`Random`/`Ordered`/`Brute` placements, seeded `Annealing` (`initial_place_method=Ordered()`),
refiner booleans + hyperedge lists + placements + costs, `PacMan` packet dictionaries,
`GainManager` occupancy/cost maps/move gains, graph tools (`steiner_tree` edge metadata),
`to_pytket_circuit` commands/qubits/params, ambient RNG parity (`random.getstate()` after each random-consuming call), and a lazy-import check for `pytket_dqc`.

## 7. (Optional) Upstream pytest suite as an oracle

The original, unmodified upstream tests can also be run against the reference
package to sanity-check the environment:151 passed, 7 skipped, 7 deselected

```bash
.venv/bin/python3 -m pytest <path-to-upstream-tests> -m "not high_compute"
```
