# Fix 2: Refiners Determinism, Non-Deterministic Unseeded RNG, & Ambient RNG Parity

**Status:** Implemented  
**Scope:** `PyRandom`, `BoundaryReallocation`, `DetachedGates`, `PartitioningHeterogeneous`, and `tests/rng_semantics_test.py`  
**Severity:** Core architectural clarity & reproducibility  

---

## 1. Executive Summary

In quantum circuit refinement algorithms (such as `BoundaryReallocation` and `DetachedGates`), ties between equally good candidate placements are broken using random choice.

### Architectural Rules Adopted:
1. **Unseeded calls (`seed=None`) are non-deterministic via OS entropy:**
   - Both upstream Python (`pytket-dqc`) and the C++ adapter (`pytket-dqc-cpp`) are non-deterministic when called without an explicit seed.
   - In the C++ adapter, `dqc::PyRandom` initializes from `std::random_device` (OS entropy) when constructed with `seed=None` (`std::nullopt`).
2. **Ambient RNG Parity:**
   - Randomness crosses the boundary as the 625-word MT19937 state. The Python wrapper seeds `random` exactly as the reference does, forwards `random.getstate()[1]`, and installs the state returned by the core (`_bridge.call_with_python_rng`).
   - Refiners reseed only when an explicit seed is given; unseeded calls continue the caller's ambient stream. In both cases `random.getstate()` after the call matches the reference. (This replaced the earlier isolation design — see §6.)
3. **Differential Tests Require Seeded Randomness:**
   - Because unseeded runs draw independent OS entropy in both implementations, differential tests comparing the C++ adapter against upstream Python must always specify an explicit integer `seed` (`seed=...`).
   - When an explicit integer seed is passed, `PyRandom` reproduces CPython's Mersenne Twister (MT19937) and `init_by_array` seeding convention **bit-for-bit**, providing 100% exact parity with the upstream reference.

---

## 2. Background: What Refiners Do

In Distributed Quantum Computing, after an initial placement assigns circuit vertices to quantum server nodes, **refiners** iteratively optimize the partitioning boundary:
- They inspect boundary gates (gates acting across two or more servers) and evaluate whether moving a gate or swapping qubits reduces total communication cost or balances server memory.
- If two candidate moves yield the exact same cost improvement, the algorithm uses a random tie-breaker to avoid getting trapped in cycles.

The primary refiners affected are:
1. `BoundaryReallocation`: Core local search on boundary vertices.
2. `DetachedGates`: Delegates directly to `BoundaryReallocation`.
3. `PartitioningHeterogeneous`: Distributor algorithms that invoke `BoundaryReallocation` during hypergraph partitioning and embedding.

---

## 3. Comparison Matrix

| Property | Upstream Python Reference (`pytket-dqc`) | C++ Adapter (`pytket-dqc-cpp`) |
| :--- | :--- | :--- |
| **Unseeded execution (`seed=None`)** | Non-deterministic (continues Python's ambient RNG) | **Non-deterministic** (continues the same ambient stream) |
| **Impact on Python's `random.getstate()`** | Mutates / advances Python global state | **Identical mutation** (625-word state round-tripped through the core) |
| **Explicit integer `seed=N`** | Deterministic | **Deterministic (100% bit-for-bit match with upstream)** |
| **Differential testing approach** | Must be seeded | **Must be seeded** |

---

## 4. Implementation Details

### C++ Core (`cpp/include/dqc/rng.hpp` and `cpp/python/bindings.cpp`)
- `PyRandom` default constructor parameter updated to `std::nullopt`:
  ```cpp
  explicit PyRandom(std::optional<std::uint64_t> seed = std::nullopt);
  ```
- Pybind11 binding default parameter:
  ```cpp
  py::class_<PyRandom>(m, "PyRandom")
      .def(py::init<std::optional<std::uint64_t>>(), py::arg("seed") = std::nullopt)
  ```
- `seed(std::nullopt)` draws 625 32-bit words from `std::random_device` to seed MT19937.

### Refiner Core (`cpp/include/dqc/refiners/boundary_reallocation.hpp`)
- `boundary_reallocation_refine_impl` takes the `PyRandom&` supplied by its caller.
- `boundary_reallocation_refine` / `detached_gates_refine` accept an optional `rng_state` and the matching `*_with_state` variants return `(result, rng.get_state())`.
- `seed=std::nullopt` still seeds from `std::random_device`; when `rng_state` is supplied it takes precedence and restores the generator to Python's stream.

### Python Adapter Docstrings
Updated in:
- `python/pytket_dqc_cpp/refiners/boundary_reallocation.py`
- `python/pytket_dqc_cpp/refiners/detached_gates.py`
- `python/pytket_dqc_cpp/distributors/partitioning_heterogeneous.py`
Documenting:
> `:key seed:` Seed for randomness. Default is None: unseeded calls continue Python's ambient `random` stream and return the advanced state to it, exactly as `pytket-dqc` does. For reproducible results or cross-library parity, specify an explicit integer seed.

---

## 5. Verification & Tests

Tests in `tests/rng_semantics_test.py`:
1. `test_pyrandom_unseeded_is_nondeterministic`: confirms separate unseeded `PyRandom()` instances produce differing random streams.
2. `test_pyrandom_seeded_is_deterministic`: confirms separate `PyRandom(42)` instances produce identical streams.
3. `test_refiners_use_global_random_like_reference`: confirms unseeded refiner calls advance Python's `random` stream deterministically from the caller's state, and that seeded calls reset it.
4. `test_seeded_refiners_are_deterministic`: confirms calling refiners with an explicit seed produces reproducible placements.

---

## 6. Revision: from isolation to parity

The original implementation (see §1 rule 2 above) kept C++ refiners completely
disjoint from Python's generator: unseeded calls drew from `std::random_device` and
left `random.getstate()` untouched. That was internally coherent but observable as a
difference from `pytket-dqc`, which draws from the module-level generator.

The adapter now round-trips the 625-word MT19937 state for every random-consuming
boundary (`Random.allocate`, `Annealing.allocate`, `BoundaryReallocation`,
`DetachedGates`): state in via `rng_state`, state out via the `*_with_state`
return value, then `random.setstate(...)`. Per-call overhead is one 625-int
conversion each way; the annealing loop performs no Python calls per iteration.
`tests/python/check_adapter_equivalence.py` asserts post-call state equality with
the reference for seeded and unseeded refiner calls, `Random`, and `Annealing`.
