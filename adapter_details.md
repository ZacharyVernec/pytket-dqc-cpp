# Pytket-DQC-CPP Adapter Details & Implementation Architecture

This document provides a comprehensive technical reference for the `pytket_dqc_cpp` Python package and its underlying C++20 core library (`dqc_core`). It details the architectural boundaries, bridge conversion mechanics, RNG semantics, exception handling, and verification framework developed during the port of upstream [`pytket-dqc`](https://github.com/CQCL/pytket-dqc) (commit `1ac175e`).

---

## 1. Architectural Overview & System Role

`pytket_dqc_cpp` is a drop-in replacement for the upstream `pytket-dqc` package, maintaining full API compatibility while delegating heavy combinatorial algorithms to an optimized C++ core.

```
                           +-----------------------------------+
                           |         User Application          |
                           |   (import pytket_dqc_cpp as dqc)  |
                           +-----------------+-----------------+
                                             |
                               +-------------v-------------+
                               |     pytket_dqc_cpp        |
                               |   (Python API Adapter)    |
                               +---+---------+---------+---+
                                   |         |         |
                  +----------------+         |         +-----------------+
                  | (data bridge)            | (pure Python)             | (lazy delegation)
                  v                          v                           v
     +--------------------------+  +--------------------+  +--------------------------+
     |   _core (pybind11)       |  |  Python Components |  |   pytket_dqc (Oracle)    |
     |  - PyRandom (MT19937)    |  |  - GainManager     |  |  - allocators.Routing    |
     |  - Hypergraph operations |  |  - VertexCover     |  |  - Circuit Generators    |
     |  - Distribution & ALAP   |  |  - Placement       |  |  - check_equivalence    |
     |  - PacMan Packing        |  |  - NISQNetwork     |  |    (warn_once emitted)   |
     |  - Allocators & Refiners |  +--------------------+  +--------------------------+
     |  - KaHyPar Backend       |
     +--------------------------+
```

### Key Principles:
1. **Drop-in Compatibility:** The package mirrors upstream's module tree (`circuits`, `placement`, `networks`, `packing`, `allocators`, `refiners`, `distributors`, `utils`), class names, default parameters, serialization formats, and exception behavior.
2. **Oracle Verification Model:** The reference `pytket-dqc` package remains installed in the virtual environment as an oracle for differential equivalence testing.
3. **Lazy Delegation:** Features not yet ported to C++ or requiring external Python-only packages delegate to `pytket_dqc` dynamically at call time, emitting a single `UserWarning` per process.

---

## 2. Component Implementation Matrix

The algorithms in Distributed Quantum Computing (DQC) are partitioned across C++, pure Python glue, and upstream delegation:

| Subsystem | Component | Implementation | Notes |
| :--- | :--- | :--- | :--- |
| **Data Structures** | `Hypergraph`, `Hyperedge` | Pure Python & C++ (`_core.Hypergraph`) | Python tracks objects; C++ core mirrors for operations |
| | `HypergraphCircuit` | Pure Python & C++ | Python maintains `_vertex_circuit_map`; operations call `_core` |
| | `Placement` | Pure Python & C++ | Python dict insertion order preserved |
| | `NISQNetwork`, `ServerNetwork` | Pure Python & C++ | Graph conversions and validation mirrored |
| **Circuits & Cost** | `Distribution.cost()` | C++ Core (`_core.distribution_cost`) | Steiner tree + ALAP scheduling executed in C++ |
| | `Distribution.to_pytket_circuit()`| C++ Core (`_core.Distribution`) | Emits `pytket.Circuit`; pybind11 translates exceptions |
| | `subcircuit` / H-embedding queries | C++ Core (`_core.HypergraphCircuit`)| Subcircuits rebuilt as pytket `Command` lists |
| **Packing** | `PacMan` | C++ Core (`_core.PacMan`) + Python | Packet identification, neighbouring, hopping, merged, conflict graphs in C++ |
| **Allocators** | `Random` | C++ Core (`_core.random_allocate`) | Direct C++ MT19937 allocation |
| | `Ordered` | C++ Core (`_core.ordered_allocate`) | Deterministic qubit sorting and placement |
| | `Brute` | C++ Core (`_core.brute_allocate`) | Exhaustive search |
| | `Annealing` | Hybrid (`_core.annealing_from_placement`)| Initial placement in Python; annealing iterations loop in C++ |
| | `HypergraphPartitioning` | C++ Core (`_core.hypergraph_partitioning`)| KaHyPar C facade (`libkahypar.h`) + Boost program options |
| | `Routing` | **Delegated** to `pytket_dqc` | Requires tket `PlacementPass`, `RoutingPass`, `DecomposeSwapsToCXs` |
| **Refiners** | `EagerHTypeMerge` | C++ Core (`_core.eager_h_type_merge_refine`) | Core refiner call + state synchronization |
| | `NeighbouringDTypeMerge` | C++ Core (`_core.neighbouring_d_type_merge_refine`) | Core refiner call + state synchronization |
| | `IntertwinedDTypeMerge` | C++ Core (`_core.intertwined_d_type_merge_refine`) | Core refiner call + state synchronization |
| | `BoundaryReallocation` | C++ Core (`_core.boundary_reallocation_refine`) | Core local search; advances ambient Python RNG stream |
| | `DetachedGates` | Python + C++ (`BoundaryReallocation`) | Computes fixed vertices in Python, delegates to `BoundaryReallocation` |
| | `VertexCover` | Pure Python | NetworkX and brute-force vertex cover algorithms |
| | `RepeatRefiner`, `SequenceRefiner` | Pure Python | Composite refiner loops |
| **Optimization** | `GainManager` | Pure Python | Move gain heuristics; calls core `steiner_tree` and `hyperedge_cost` |
| **Utilities** | `to_euler_with_two_hadamards` | C++ Core (`_core.to_euler_with_two_hadamards`) | Reconstructs pytket `Op` objects |
| | `steiner_tree`, `direct_from_origin`| C++ Core (`_core.steiner_tree`) | Exact Mehlhorn algorithm with edge attribute restoration |
| | `circuit_analysis` helpers | C++ Core | `ebit_cost`, `ebit_memory_required`, `all_cu1_local`, `get_server_id` |
| | Circuit Generators | **Delegated** to `pytket_dqc` | `RandomHypergraphCircuit`, `CyclicHypergraphCircuit`, `RegularGraphHypergraphCircuit` |
| | Verification (`check_equivalence`) | **Delegated** to `pytket_dqc` | PyZX equivalence checking oracle |

---

## 3. Bridge Mechanics & Data Exchange

The bridge between high-level Python `pytket` objects and the low-level C++ extension is implemented in `python/pytket_dqc_cpp/_core.py` and `_bridge.py`.

### 3.1 Extension Resolution (`_core.py`)
To support both developer builds in the workspace and installed wheel distributions, `_core.py` resolves the compiled C++ extension dynamically:
1. **Installed layout:** First attempts `from . import _core`.
2. **Build tree fallback:** Appends `<workspace>/build/dqc_cpp` to `sys.path` and imports top-level `_core`.
3. If neither is found, raises an actionable `ImportError` citing the exact `cmake` compilation command required.

### 3.2 Circuit Serialization & Command Reconstruction
The C++ core contains no dependency on pytket headers. Circuits cross the boundary via primitive serialization:
- `serialize_circuit(circuit)`: Converts a `pytket.Circuit` into `(qubits, serialized_commands)`. Each command is serialized as:
  ```python
  (op.get_name(), int(op.type), [str(q) for q in cmd.qubits], [float(p) for p in op.params])
  ```
- `command_from_core` / `commands_from_core`: Rebuilds pytket `Op` and `Command` instances using an integer-to-`OpType` table (`OP_TYPE_BY_CODE`), ensuring the reconstructed commands reference the circuit's original `Qubit` instances rather than new objects.

### 3.3 State Synchronization (`sync_from_core`)
When C++ refiners modify hypergraph partitions or placement allocations in-place, `_bridge.sync_from_core(dist, core_dist)` synchronizes the Python state:
- Clears and rebuilds `dist.circuit.hyperedge_list` and `dist.circuit.hyperedge_dict`.
- Updates `dist.placement.placement` with a clean dictionary copy.
- **Insertion-Order Invariant:** C++ `std::map` iteration order is sorted, but Python dicts maintain insertion order. `sync_from_core` preserves previous key order where applicable to prevent nondeterministic dictionary iteration downstream.

### 3.4 Vertex Circuit Mapping
The return type of `_core.HypergraphCircuit.get_gate_of_vertex` is not exposed across pybind11. Therefore, `pytket_dqc_cpp.circuits.HypergraphCircuit` maintains `_vertex_circuit_map` (mapping gate vertex IDs to pytket `Command` objects) in Python.

---

## 4. Random Number Generation & Reproducibility Semantics

Maintaining exact reproducibility between Python and C++ requires mirroring Python's `random` module (Mersenne Twister MT19937) and respecting ambient process state.

### 4.1 Bit-Exact MT19937 Engine (`PyRandom`)
The C++ core implements `dqc::PyRandom` (`cpp/include/dqc/rng.hpp`), which matches CPython's MT19937 implementation bit-for-bit:
- **CPython Array Seeding:** Implements `init_by_array` using little-endian 32-bit integer conversion from 64-bit integer seeds, matching CPython's `_random.Random.seed(int)`.
- **Double Generation:** Implements `genrand_res53`, producing 53-bit resolution floating-point numbers in $[0.0, 1.0)$ matching Python's `random.random()`.
- **State Serialization:** Exports and imports the exact 625-integer MT19937 state tuple (`state[0..623]` and `index`), allowing two-way state handoff.

### 4.2 Seeded vs. Unseeded Execution Contract
- **Explicit Integer Seed (`seed=N`):**
  - Fully deterministic and bit-exact across both Python and C++.
  - Differential test suites must always supply explicit integer seeds.
- **Unseeded Calls (`seed=None`):**
  - Non-deterministic in both implementations.
  - In C++, `PyRandom(std::nullopt)` draws 625 words from hardware entropy via `std::random_device rd; rd()`, generating a fresh random sequence without mutating ambient Python RNG state.
  - In Python refiner calls (`BoundaryReallocation`, `DetachedGates`), unseeded calls continue Python's ambient `random` stream.

### 4.3 Ambient RNG Round-Tripping (`call_with_python_rng`)
Whenever a C++ function consumes randomness, the boundary bridges Python's ambient generator via `_bridge.call_with_python_rng`:
1. Extracts Python's current ambient state: `random.getstate()[1]`.
2. Passes the 625-word array to the C++ core function (e.g. `annealing_from_placement_with_state` or `boundary_reallocation_refine_with_state`).
3. Receives the updated 625-word state from C++ upon completion.
4. Restores Python's generator via `random.setstate(...)`.

As a result, subsequent calls to `random.random()` in user code observe the exact same state advance as they would have observed in upstream `pytket-dqc`.

### 4.4 Simulated Annealing Initializer Divergence & Warning
In upstream `pytket-dqc`, `Annealing.allocate()` suffered from an unintended seed loss:
```python
# Upstream pytket_dqc/allocators/annealing.py
initial_aloc = kwargs.get("initial_place_method", Random())
seed = kwargs.get("seed", None)
random.seed(seed)                                    # Seeds global RNG
distribution = initial_aloc.allocate(circ, network)  # Calls Random().allocate() WITHOUT seed
```
Inside `Random.allocate()`, `kwargs.get("seed", None)` evaluated to `None`, calling `random.seed(None)` and resetting the generator from OS entropy. **Upstream was therefore non-deterministic even against itself on this default path.**

In `pytket_dqc_cpp`:
- `Annealing` maintains drop-in API compatibility with `initial_place_method=Random()`.
- If a user passes an explicit `seed` alongside the default unseeded `Random()` initializer, `Annealing.allocate()` emits a `warn_once` alerting the user that the default initializer is non-deterministic.
- **Initial Placement Compatibility Matrix:**
  - **Exact Cross-Library Parity:** Passing any deterministic allocator from `{Ordered(), HypergraphPartitioning(), Brute(), Routing()}` preserves Python's global RNG state, producing bit-for-bit identical results between upstream `pytket-dqc` and `pytket_dqc_cpp`.
  - **Self-Reproducibility:** Passing a custom seeded `Random` wrapper (e.g. `class SeededRandom(Random): ...`) produces reproducible runs within `pytket_dqc_cpp`.
  - **Default `Random()`:** Reseeds from OS entropy on both sides; non-deterministic.

---

## 5. Constraint Exceptions & Error Parity

During circuit synthesis in `Distribution.to_pytket_circuit()`, physical hardware limitations (such as server communication memory capacity or unroutable multi-qubit gates) trigger exceptions.

### 5.1 Native Pybind11 Translation
Rather than relying on fragile string parsing or regex post-processing, `cpp/python/bindings.cpp` catches `dqc::ConstraintException` at the pybind11 boundary and translates it into `pytket_dqc_cpp.utils.circuit_analysis.ConstraintException`:
- `.server`: Set to the integer index of the oversubscribed server node (or `None` if unroutable).
- `.v_gate`: Set to the integer vertex ID of the offending gate in the hypergraph circuit.
- **Guidance Hint:** The exception message string directly incorporates the upstream remediation hint:
  ```text
  Consider setting allow_update to True:
  to_pytket_circuit(allow_update=True)
  ```

### 5.2 Upstream Test Typo & Verified Parity
In upstream `tests/circuits_test.py:936`, an assignment typo was present in the test suite:
```python
except ConstraintException as e:
    e.server = 3      # Assignment typo '=' instead of comparison '=='!
    caught = True
assert caught
```
Because of this typo, upstream tests passed green even if `.server` was missing. The C++ adapter maintains byte-for-byte fidelity with the vendored upstream test suite while verifying strict attribute parity (`e.server`, `e.v_gate`, message hints) in `tests/error_parity_test.py`.

---

## 6. Verification Architecture & Multi-Tier Gates

Verification is structured across five distinct tiers to guarantee both internal correctness and cross-package equivalence.

### 6.1 The Five Verification Tiers
| Tier | Description | Command | Scope |
| :---: | :--- | :--- | :--- |
| **1** | Fast Python Suite | `pytest tests -m "not high_compute"` | Vendored tests + adapter unit tests (152 passed, 7 skipped, 7 deselected) |
| **2** | Slow Determinism Suite | `pytest tests -m high_compute` | Heavyweight 10-qubit fixtures and PyZX equivalence tests (7 tests, ~178s) |
| **3** | Core Differential | `python tests/python/check_equivalence.py` | Compares C++ core algorithms directly against `pytket_dqc` and `networkx` |
| **4** | Adapter Differential | `python tests/python/check_adapter_equivalence.py` | Compares `pytket_dqc_cpp` objects vs `pytket_dqc` across 4 hash seeds |
| **5** | Native C++ Suite | `ctest --test-dir build` | 89 Catch2 unit tests in `cpp/tests/` |

### 6.2 Hardened Differential Gate (`check_adapter_equivalence.py`)
To prevent false-positive green runs and silent bypasses:
1. **Strict Check Count:** Dynamically enforces that all planned checks run. The suite requires exactly **894 checks per seed** (3,576 checks across 4 seeds); any mismatch fails the run.
2. **Zero Skips Permitted:** Skips are strictly tracked in `SKIPPED[0]`. Any skipped test is treated as a gate failure (`SKIPPED[0] == 0`). Oversubscribed network combinations assert matching capacity exceptions rather than skipping.
3. **Four-Seed Hash Sweep:** CPython's hash seed affects set iteration order and vertex cover selection. By default, the script executes across `HASH_SEEDS = [0, 1, 42, 12345]`. To isolate a single run, developers must set `DQC_ADAPTER_SINGLE_SEED=1`.

### 6.3 Unified Gate Runner (`run_gates.sh`)
The executable script `tests/python/run_gates.sh` provides a single command to run Tiers 5, 1, 3, and 4 in sequence under `set -euo pipefail`, unsetting ambient `PYTHONHASHSEED` to safeguard hash-invariance testing.

### 6.4 Upstream Test Skips
The 7 skipped tests in the fast suite (`tests/README.md`) are identical to upstream's skips:
1. `tests/allocator_test.py`: `test_refinement_makes_valid` (unsupported CX gates).
2. `tests/allocator_test.py`: `test_q_control_box_circuits` (unsupported `QControlBox`).
3. `tests/circuits_test.py`: `test_q_control_box_circuits` (unsupported `QControlBox`).
4. `tests/circuits_test.py`: `test_to_pytket_circuit_with_teleportation` (teleportation disabled).
5. `tests/circuits_test.py`: `test_to_relabeled_registers` (removed upstream method).
6. `tests/network_test.py`: `test_nisq_get_placer` (upstream tuple comparison defect).
7. `tests/utils_test.py`: `test_CX_circuit` (CX gate support disabled).

---

## 7. Known Approximations & Implementation Notes

1. **`Hypergraph.vertex_neighbours` Ordering:** Uses `std::set` (ascending order) in C++. This is unobservable because `vertex_neighbours` is only used for set membership and boundary detection, never iterated in an order-sensitive manner.
2. **`PacMan.are_hoppable_packets` Bug Fidelity:** Faithfully reproduces upstream reference behavior where the fallback branch is dead code due to comparing `op.params` to `0`.
3. **`PyRandom` Integer Range:** Direct integer seeding supports seeds up to $2^{64}-1$.
4. **`DetachedGates` Routing:** Computes `fixed_vertices` in Python and routes to `BoundaryReallocation` because the compiled C++ `detached_gates_refine` entry point segfaults when hyperedges contain zero gate vertices (e.g. degenerate hyperedges with only qubit vertices).
5. **Drawing Subsystem:** Visualization methods (`draw()`, `draw_nisq_network()`, `draw_server_network()`) and graphical dependencies (`hypernetx`, `matplotlib`, `graphviz`) are deliberately omitted.
