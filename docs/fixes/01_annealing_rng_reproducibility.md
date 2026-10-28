# Fix 1: Simulated Annealing Default Initializer & Seed Handling

**Status:** Implemented  
**Scope:** `python/pytket_dqc_cpp/allocators/annealing.py`, documentation, and new tests  
**Severity:** Low operational risk, High reproducibility clarification  

---

## 1. Executive Summary

When running Simulated Annealing with default settings, the original Python library (`pytket-dqc`) had an unintentional quirk: **it was not reproducible even with itself**. Supplying `seed=11` to two consecutive runs in upstream produced two completely different circuit placements.

Our C++ adapter (`pytket-dqc-cpp`) is fully deterministic because its random engine is isolated in C++. However, because the upstream reference was inherently non-deterministic here, **the C++ adapter cannot produce identical placement numbers to the Python reference on this default path**.

To resolve this cleanly:
1. We will **not** attempt to match the random numbers of a non-deterministic upstream reference.
2. We will **emit a clear warning** (using `warn_once` so loops aren't spammed) when a user provides a `seed` alongside the default initializer, advising them to pass a deterministic initial placement method such as any from the following set: `Ordered`, `HypergraphPartitioning`, `Brute`, or to pass an explicitly seeded wrapper around `Random`.
3. We will **lock down the C++ adapter's deterministic output** with tests to ensure future internal refactoring does not alter the draw sequence.

---

## 2. Background: How Annealing Works in DQC

In Distributed Quantum Computing (DQC), the `Annealing` allocator decides which quantum servers will execute which qubits in a quantum circuit. Its goal is to minimize expensive inter-server quantum communication (EPR pairs / teleportation).

Simulated Annealing is a local-search heuristic:
1. **Initial Placement:** It first generates a starting layout (mapping qubits to servers). By default, it uses `Random()`.
2. **Annealing Loop:** It iteratively proposes random qubit swaps. If a swap reduces communication cost, it is accepted. If a swap increases cost, it may still be accepted with a temperature-dependent probability.

---

## 3. The Root Cause: Why Upstream Wasn't Reproducible

In the upstream reference (`pytket_dqc/allocators/annealing.py`):

```python
# Annealing.allocate():
initial_aloc = kwargs.get("initial_place_method", Random())
seed = kwargs.get("seed", None)
...
random.seed(seed)                                    # Line 101: Seeds Python's global RNG
distribution = initial_aloc.allocate(circ, network)  # Line 104: Calls Random().allocate()
```

Notice that `Annealing.allocate()` calls `initial_aloc.allocate(circ, network)` **without passing `seed` along**.

Inside `Random.allocate()` (`pytket_dqc/allocators/random.py`):

```python
# Random.allocate():
seed = kwargs.get("seed", None)  # seed is None because Annealing didn't pass it!
random.seed(seed)                # random.seed(None) -> RESEEDS FROM SYSTEM ENTROPY!
```

### The Chain Reaction:
1. The user passes `seed=11` to `Annealing.allocate()`.
2. Line 101 correctly seeds Python's global random generator with `11`.
3. Line 104 invokes `Random.allocate()`, which receives `seed=None`.
4. In Python, calling `random.seed(None)` resets the global generator using operating system entropy (`/dev/urandom` or the system clock).
5. **The initial `seed=11` is immediately obliterated.**
6. The subsequent annealing loop runs on an unpredictable, entropy-seeded state.

### Empirical Audit Proof:
Running the upstream reference twice with the exact same seed:
```
Run 1 (seed=11): {0: 0, 4: 0, 7: 0, 1: 0, 5: 0, 8: 3, 2: 1, 6: 1, 3: 3}
Run 2 (seed=11): {0: 2, 4: 1, 7: 3, 1: 0, 5: 2, 8: 1, 2: 0, 6: 2, 3: 1}
```

---

## 4. How the C++ Adapter and Upstream Interact

To understand the behavior, we have to look at the two layers:

### A. The Compiled C++ Core (`_core.random_allocate`, `_core.annealing_allocate` and `_core.annealing_from_placement`)
The C++ core implements a bit-exact Mersenne Twister MT19937 (`PyRandom`).
- When seeded explicitly (`PyRandom(seed)` or `rng.seed(seed)`), it is bit-exact with CPython's RNG initialized with that same integer seed.
- When unseeded (`seed=None` / `std::nullopt`), `PyRandom::seed(std::nullopt)` draws 625 32-bit words from hardware entropy via `std::random_device rd; rd()`.
- Therefore, each unseeded call creates a freshly randomized stream from OS entropy; it is not a fixed seed and does not just continue a shared previous stream.
- `annealing_from_placement` additionally accepts an `rng_state` (the 625-word CPython MT19937 state), letting the core continue an existing Python stream instead of starting a fresh one.

### B. The Python Layer (`Annealing.allocate`)
In `pytket-dqc-cpp`, `Annealing.allocate()` keeps the upstream control flow for the initial placement, then runs the annealing iterations in the C++ core:
```python
random.seed(seed)                                    # (1) Seeds Python's global RNG with user seed
distribution = initial_aloc.allocate(circ, network)  # (2) Calls initial allocator (Random by default)
core.annealing_from_placement_with_state(..., rng_state)  # (3) Core continues from Python's state
random.setstate(...)                                 # (4) Python continues from the core's state
```
There is no Python-side annealing loop. The state of Python's global generator (`random.getstate()[1]`) is forwarded as `rng_state`, and the 625-word state the core returns is installed again with `random.setstate`, so `random.getstate()` after the call matches the reference exactly.

The hand-off is two-way via `_bridge.call_with_python_rng`: the core accepts Python's ambient 625-word MT19937 state (`rng_state`), executes the annealing loop in C++, and returns its final RNG state, which is re-installed into Python's global generator via `random.setstate`. This matches upstream's consumption of module-level randomness, ensuring that any code calling `random` afterwards continues from the exact state produced by the annealing iterations. `tests/rng_semantics_test.py::test_annealing_returns_core_state_to_python` pins this behaviour.

When `initial_place_method` is left at the default `Random()`:
1. `Annealing.allocate` seeds Python's global `random` with `seed` (e.g. `11`).
2. It calls `Random().allocate(circ, network)` without forwarding `seed`.
3. In `pytket-dqc-cpp`, `Random.allocate()` delegates to `core.random_allocate(..., seed=None)`. The C++ core seeds its internal `PyRandom` from `std::random_device` (OS entropy).
4. `core.random_allocate` returns a non-deterministic initial placement (Run 1 != Run 2).
5. The annealing iterations in the core resume from Python's global `random` state (still at `seed=11`), but because step 2 produced different initial placements, the overall annealing trajectories diverge.

Hence, **neither upstream nor the adapter is reproducible with itself on `Annealing(seed=...)` when using the default unseeded `Random()`**.

---

## 5. Compatibility Analysis of All Initial Placement Allocators

Is `Ordered()` the only initial placement allocator that works for exact cross-library parity, or do others work as well?

To answer this conclusively, we tested every available allocator in `pytket_dqc`:

| Allocator Passed as `initial_place_method` | Preserves Python Global RNG State? | Matches Between Upstream & Adapter? | Reproducible with Itself? | Status |
| :--- | :---: | :---: | :---: | :--- |
| **`Ordered()`** | Yes (0 RNG draws) | **YES (Exact match)** | **YES** | **Works** — Fully deterministic. |
| **`HypergraphPartitioning()`** | Yes (0 Python RNG draws; KaHyPar uses C RNG) | **YES (Exact match)** | **YES** | **Works** — Deterministic graph partitioning. |
| **`Brute()`** | Yes (0 RNG draws) | **YES (Exact match)** | **YES** | **Works** — Fully deterministic exhaustive search. |
| **`Routing()`** | Yes (delegates to upstream reference) | **YES (Exact match)** | **YES** | **Works** — (Slow, but exact match). |
| **`Random(seed=s)` (custom wrapper)** | Upstream advances Python RNG; C++ adapter uses C++ RNG | **NO (Cross-library)** | **YES (Within each library)** | **Self-reproducible** — Can pass an explicitly seeded wrapper (as done in `tests/python/qft/main_*`). Produces reproducible runs within `pytket-dqc-cpp`. |
| **`Random()` (default)** | No (C++ uses `std::random_device`, upstream resets global RNG with `/dev/urandom`) | **NO** | **NO** | **Fails** — Reseeds from OS entropy on both sides; non-deterministic. |

### Summary of What Works vs What Doesn't:
- **What Works for Exact Cross-Library Parity:** **Any deterministic, non-random allocator**, such as any from the following set: `Ordered`, `HypergraphPartitioning`, `Brute` (or `Routing`).
  - None of these allocators consume Python `random` draws during `initial_aloc.allocate(circ, network)`.
  - Python's global `random` state remains at the exact state initialized by `random.seed(seed)`.
  - Upstream and the C++ adapter produce the exact same initial placement and then run the exact same sequence of annealing draws, achieving **bit-for-bit equivalence**.
- **What Works for Single-Library Self-Reproducibility:** A **custom seeded `Random` wrapper** (as used in `tests/python/qft/main_few_qpus_py.py` and `main_many_qpus_py.py`):
  ```python
  class SeededRandomAllocator(Random):
      def __init__(self, seed: int):
          self.seed = seed
      def allocate(self, circ, network, **kwargs):
          kwargs["seed"] = self.seed
          return super().allocate(circ, network, **kwargs)
  ```
  Passing `initial_place_method=SeededRandomAllocator(seed)` ensures that the initial placement step is deterministic across repeated runs within `pytket-dqc-cpp`.
- **What Doesn't Work:** The default unseeded `Random()` allocator.
  - Default `Random()` reseeds from system entropy (non-deterministic).
  - The initial placement varies on each call, causing the entire annealing trajectory to diverge.

---

## 6. Proposed Solution & Implementation Steps

### Step 1: Add a User Warning for Default Random Initializer
When a user explicitly passes a `seed` to `Annealing.allocate()`, they expect reproducible results. We will alert them when using the default unseeded `Random()` and suggest passing a deterministic initial placement method (such as any from the following set: `Ordered`, `HypergraphPartitioning`, `Brute`), or an explicitly seeded `Random` wrapper.

To prevent noisy stdout during benchmark loops, we use `warn_once` (imported from `pytket_dqc_cpp._delegation`):

In `python/pytket_dqc_cpp/allocators/annealing.py`:

```python
from .._delegation import warn_once

def allocate(self, circ, network, **kwargs):
    seed = kwargs.get("seed", None)
    if "initial_place_method" not in kwargs and seed is not None:
        warn_once(
            "Annealing_default_random_seed",
            message=(
                "Annealing: the default Random() initializer does not consume the "
                "seed deterministically. For guaranteed cross-platform reproducibility, "
                "pass a deterministic initial placement method such as any from the "
                "following set: Ordered, HypergraphPartitioning, Brute, or pass an "
                "explicitly seeded wrapper around Random."
            ),
            stacklevel=2,
        )
    ...
```

### Step 2: Update Documentation
Update `README.md` and docstrings in `python/pytket_dqc_cpp/allocators/annealing.py`:
- Clarify that passing any deterministic allocator from `{Ordered, HypergraphPartitioning, Brute}` provides exact, bit-for-bit equivalence between upstream Python and the C++ adapter.
- Clarify how to pass an explicitly seeded `Random` wrapper for reproducible randomized initial placements.
- Document that `initial_place_method=Random()` (the default) creates a non-deterministic initial placement (reseeding from OS entropy), which diverges from upstream and between runs.

### Step 3: Pin Parity and Determinism with Tests
Create a new test file `tests/rng_semantics_test.py` collected by `pytest`:

```python
@pytest.mark.parametrize("allocator_cls", [Ordered, Brute, HypergraphPartitioning])
def test_annealing_deterministic_allocators_match(allocator_cls):
    """Verify that all deterministic initializers produce identical results."""
    dist_ad = Annealing().allocate(circ, network, seed=11, iterations=100, initial_place_method=allocator_cls())
    ...

def test_annealing_warns_about_default_initializer():
    """Verify that a warning is emitted once when seed is passed with default initializer."""
    with pytest.warns(UserWarning, match="such as any from the following set"):
        Annealing().allocate(circ, network, seed=1, iterations=1)
```

---

## 7. Additional Details

1. **API Parity Decision (Option A):**
   - We will keep `initial_place_method=Random()` (unseeded) as the default parameter to preserve 100% drop-in API signature parity with upstream `pytket-dqc`.
   - Callers who pass `seed=...` without specifying `initial_place_method` will receive the informative `warn_once` pointing them to the deterministic options (`Ordered`, `HypergraphPartitioning`, `Brute`) or a seeded `Random` wrapper.

2. **Warning De-duplication (`warn_once`):**
   - In production and benchmark scripts, `Annealing.allocate()` may be called hundreds of times in tight loops.
   - Emitting a warning on every invocation would create massive log clutter. Using `warn_once("Annealing_default_random_seed", ...)` ensures the warning is emitted at most once per process run.
   - The warning tracker can be reset in unit tests using `_delegation.reset_warnings()`.
