# Remaining fixes — `pytket_dqc_cpp` adapter

Status: open work items after the 2026-10-26/27 audit.
Scope: Python adapter + tests only. No changes under `cpp/` or `CMakeLists.txt`
(the plan forbids them).

This document covers the three items that remain after the audit:

1. **Fix 3** — RNG semantics of the *default* code paths (`Annealing` default
   initializer; unseeded refiners).
2. **Fix 4** — error attribute parity (`ConstraintException.server` /
   `.v_gate`), which the vendored suite currently masks.
3. **Gates** — the verification gates are manual, partially excluded from
   pytest, and can report success while covering very little.

Background for reference (already applied): the `steiner_tree` edge-metadata
check (`check_graph_tools` / `steiner_rows` in
`tests/python/check_adapter_equivalence.py`) was added after the audit. It
addresses the earlier "mutation survives both gates" finding. The differential
should now report **739 checks** (733 + 3 blueprints × 2 checks) instead of 733.

---

## 0. Current gate state (baseline for this document)

| Gate | Command | Result at audit | Collected by pytest? |
| --- | --- | --- | --- |
| Vendored suite | `pytest tests -m "not high_compute"` | 135 passed, 7 skipped, 7 deselected | yes |
| Slow suite | `pytest tests -m high_compute` | 7 passed (~178 s) | yes (deselected by default) |
| Core differential | `tests/python/check_equivalence.py` | all checks passed + hash sweep | no |
| Adapter differential | `tests/python/check_adapter_equivalence.py` | 733 checks / 0 failures (+ hash sweep) | no |
| C++ | `ctest --test-dir build` | 87/87 | no |
| Phase gates | `check_phase2.py` (268/0), `check_bridge.py` | passed | no |

Runtime facts relied on below:

- Adapter loads the compiled extension from
  `build/dqc_cpp/_core.cpython-311-x86_64-linux-gnu.so`;
  `pytket_dqc_cpp.HypergraphCircuit is not pytket_dqc.HypergraphCircuit`.
- Vendored tests are byte-identical to `pytket-dqc@1ac175e` after
  `s/pytket_dqc_cpp/pytket_dqc/g`, with only the documented deviations
  (removed `test_nisq_draw`, KaHyPar INI path, comments).
- `tests/test_circuits/` fixtures are identical to upstream.

---

## 1. Fix 3 — pin the RNG semantics of the default paths

Two distinct issues with two different root causes and two different fixes.

### 3a. `Annealing()` with the default (`Random`) initializer

#### Causal chain

`python/pytket_dqc_cpp/allocators/annealing.py`:

```python
initial_aloc = kwargs.get("initial_place_method", Random())   # ~line 96
seed = kwargs.get("seed", None)
...
random.seed(seed)                                             # ~line 101
distribution = initial_aloc.allocate(circ, network)           # ~line 104
```

The reference `pytket_dqc/allocators/random.py::Random.allocate` then does:

```python
seed = kwargs.get("seed", None)      # None — Annealing passes no seed
random.seed(seed)                    # random.seed(None) -> OS entropy
```

So upstream **re-seeds the global stream from entropy inside the initial
placement**, and the annealing loop afterwards draws from that
entropy-derived state.

Consequences:

- `pytket_dqc.allocators.Annealing().allocate(circ, net, seed=11)` is **not
  reproducible even against itself** (two reference runs differ).
- The adapter's `Random` is core-backed and never touches the global stream, so
  the adapter *is* deterministic here — and therefore differs from any given
  reference run.
- Therefore "adapter == reference" is **undefined** for this path. It is not a
  parity bug that can be fixed by matching numbers.

Confirmed empirically at audit time:

```
Annealing default-init placements equal: False
  ref: {0: 0, 4: 0, 7: 0, 1: 0, 5: 0, 8: 3, 2: 1, 6: 1, 3: 3}
  new: {0: 1, 1: 0, 2: 3, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}
```

#### Actions

1. **Document the real reason.** The README currently says only "pass
   `initial_place_method=Ordered()` for a trajectory that is byte-for-byte
   reproducible against the reference". Add that the reference's default
   `Random` initializer reseeds from entropy, so the reference is itself
   nondeterministic on this path, while the adapter is deterministic but not
   comparable.
2. **Warn when a seed is passed together with the default initializer** —
   callers almost certainly assume reproducibility. Either reuse the
   `warn_once` style in `_delegation.py` or emit a plain `UserWarning`:

   ```python
   if "initial_place_method" not in kwargs and seed is not None:
       warnings.warn(
           "Annealing: the default Random initializer does not consume the "
           "seed deterministically; pass initial_place_method=Ordered() for "
           "a reproducible placement.",
           UserWarning,
           stacklevel=2,
       )
   ```

3. **Pin the adapter's own contract with tests** (see below), including a
   golden placement, so any accidental change in RNG plumbing fails.

#### Test sketch (new `tests/rng_semantics_test.py`, pytest-collected)

```python
def test_annealing_default_init_is_seeded_deterministic():
    """Documented divergence: reference reseeds from entropy inside Random."""
    a = Annealing().allocate(circ, network, seed=11, iterations=200, cache_limit=2)
    b = Annealing().allocate(circ, network, seed=11, iterations=200, cache_limit=2)
    assert a.placement.placement == b.placement.placement


def test_annealing_default_init_golden_placement():
    """Guards the RNG plumbing: change the draw order and this fails."""
    dist = Annealing().allocate(circ, network, seed=11, iterations=200)
    assert dist.placement.placement == GOLDEN_SEED_11


def test_annealing_warns_about_default_initializer():
    with pytest.warns(UserWarning, match="initial_place_method=Ordered"):
        Annealing().allocate(circ, network, seed=1, iterations=1)
```

The golden-value test is the important one: "two calls agree" would not catch a
changed draw order.

#### Note

`Annealing.allocate` **does** mutate the global RNG (`random.seed(seed)` plus
draws), unlike the core-backed refiners (3b). That asymmetry is upstream
behaviour for `Annealing` but not for the refiners; document it once, in the
same place.

---

### 3b. Unseeded refiners (`BoundaryReallocation`, `DetachedGates`, `PartitioningHeterogeneous`)

#### Causal chain

Reference `pytket_dqc/refiners/boundary_reallocation.py::refine`:

```python
seed = kwargs.get("seed", None)
if seed is not None:
    random.seed(seed)
...
best_swap_vertex = ... else random.choice([True, False])   # ambient stream
```

Adapter `python/pytket_dqc_cpp/refiners/boundary_reallocation.py` passes the
seed straight to the core:

```python
refinement_made = bool(
    core.boundary_reallocation_refine(
        core_dist,
        [int(vertex) for vertex in fixed_vertices],
        math.ceil(num_rounds),
        stop_parameter,
        seed,            # None -> core uses PyRandom(0)
        cache_limit,
    )
)
```

The core uses `PyRandom(0)` when `seed is None`, so with `seed=None`:

| | reference | adapter |
| --- | --- | --- |
| Result | depends on the process's ambient `random` state (reproducible if the caller called `random.seed(...)` first) | always the same |
| Global RNG after the call | advanced (and reseeded, if a seed was given) | untouched |

Confirmed empirically at audit time:

```
ref deterministic across two calls : False
adapter deterministic across calls : True
ref == adapter (call 1)            : False
```

`DetachedGates` inherits this (it routes through `BoundaryReallocation`), and so
does `distributors/partitioning_heterogeneous.py` (it calls
`BoundaryReallocation()` in both `PartitioningHeterogeneous` and
`PartitioningHeterogeneousEmbedding`).

#### Options, ranked

| Option | Effect | Verdict |
| --- | --- | --- |
| **A. Document + pin determinism** | unseeded calls are deterministic here; parity requires an explicit `seed=` | **recommended**, zero risk |
| B. Pass `random.getrandbits(64)` when `seed is None` | becomes nondeterministic like upstream, still not bit-equal, loses the determinism property | not recommended (half-measure) |
| C. Reproduce the refiner's draws in Python | requires reimplementing the tie-breaking loop | defeats the port |
| D. Accept a callable RNG in the core binding | exact parity | C++ change, out of scope |

#### Test sketch

```python
def test_unseeded_boundary_reallocation_is_deterministic():
    d1 = Distribution.from_dict(dist.to_dict())
    d2 = Distribution.from_dict(dist.to_dict())
    BoundaryReallocation().refine(d1)
    BoundaryReallocation().refine(d2)
    assert d1.placement.placement == d2.placement.placement


def test_refiners_do_not_touch_global_random():
    state = random.getstate()
    BoundaryReallocation().refine(Distribution.from_dict(dist.to_dict()))
    assert random.getstate() == state     # documents: no ambient-RNG coupling


def test_detached_gates_inherits_boundary_semantics():
    # DetachedGates delegates to BoundaryReallocation -> same seed policy
    ...
```

#### Audit required

Every binding that accepts a `seed` must share one documented policy for
`seed=None`:

- `random_allocate(hc, network, seed)`
- `ordered_allocate`
- `brute_allocate`
- `boundary_reallocation_refine(...)`
- `detached_gates_refine(...)` (avoided in practice — see below)
- `hypergraph_partitioning_allocate(...)`
- `hypergraph_partitioning_initial_placement(...)`

If they disagree, centralise the policy in one helper and state it once, rather
than per class.

---

### Documentation to update for Fix 3

- `README.md`, "Compatibility contract": the sentence "`seed=None` means
  OS-entropy seeding in both languages and is intentionally non-deterministic"
  is **wrong** for the refiner bindings.
- `README.md`, "Core-backed vs Python": extend the `Annealing` paragraph with
  the entropy-reseed explanation.
- `python/pytket_dqc_cpp/refiners/boundary_reallocation.py` docstring
  (`:key seed:`).
- `python/pytket_dqc_cpp/refiners/detached_gates.py` docstring.
- `python/pytket_dqc_cpp/allocators/annealing.py` docstring
  (`:key initial_place_method:` and `:key seed:`).

### Acceptance criteria for Fix 3

- [ ] `tests/rng_semantics_test.py` exists and passes; it is collected by
      `pytest tests -m "not high_compute"`.
- [ ] A golden placement pins default-init `Annealing(seed=11)`.
- [ ] Unseeded `BoundaryReallocation` determinism and "no global RNG mutation"
      are asserted.
- [ ] The README/docstring statements match observed behaviour (verified by the
      tests above).
- [ ] The differential's seeded paths still match the reference (no regression).

---

## 2. Fix 4 — assert `.server` / `.v_gate` instead of overwriting them

### What is actually lost

Reference `pytket_dqc/circuits/distribution.py` annotates the exception:

```python
except ConstraintException as e:
    e.v_gate = v_gate      # index of the offending gate vertex
    raise e
```

and its message ends with
`"... to_pytket_circuit(allow_update=True)"`.

The adapter receives a plain `RuntimeError` from pybind11, so
`python/pytket_dqc_cpp/_bridge.py::translate_core_error` can only:

- recover `.server` by regex (`_SERVER_PATTERNS`), which works for
  "Communication memory capacity of server N" and "non-home server N" but
  yields `None` for the third message ("could not be amended");
- leave `.v_gate = None` (hard-coded in
  `python/pytket_dqc_cpp/utils/circuit_analysis.py::ConstraintException`);
- propagate the message without the hint.

This is a stated plan divergence ("Known divergence"), but it is currently
**untested** by the acceptance suite.

### Why the suite cannot see it

`tests/circuits_test.py` (upstream verbatim, ~line 936):

```python
caught = False
try:  # Evicted gate 13 causing problems
    distribution.to_pytket_circuit()
except ConstraintException as e:
    e.server = 3          # assigns the expected value — asserts nothing
    caught = True
assert caught
```

A regression where `.server` becomes `None` would still pass.

### Recommended fix — adapter-owned test file

Do **not** edit the vendored test: keeping it byte-identical to upstream is one
of the project's strengths. Add `tests/error_parity_test.py`:

```python
def test_bounded_memory_error_parity(circ, placement, bounded_network):
    ref_err = raises(RefConstraintException, ref_dist.to_pytket_circuit)
    new_err = raises(ConstraintException, new_dist.to_pytket_circuit)

    assert new_err.server == ref_err.server            # e.g. 3 == 3
    assert str(new_err).startswith(str(ref_err)[:40])  # shared prefix
    assert "allow_update=True" in str(ref_err)         # reference-only hint

    # Documented divergence (PLAN.md, "Known divergence"):
    # v_gate does not survive pybind11.
    assert ref_err.v_gate is not None
    assert new_err.v_gate is None


@pytest.mark.parametrize(
    "message",
    [
        "Communication memory capacity of server 12 exceeded. ...",
        "non-home server 7 ...",
        "The distribution could not be amended ...",
    ],
)
def test_translate_core_error_server_parsing(message):
    err = translate_core_error(RuntimeError(message))
    assert isinstance(err, ConstraintException)
    assert err.v_gate is None
    if "server 12" in message:
        assert err.server == 12                       # multi-digit server
    if "could not be amended" in message:
        assert err.server is None


def test_translate_core_error_passthrough():
    plain = RuntimeError("something else")
    assert translate_core_error(plain) is plain       # guards the startswith list
    value = ValueError("nope")
    assert translate_core_error(value) is value
```

This converts the known divergence from "silently tolerated" into "explicitly
asserted", and covers the fragile parts of `translate_core_error`: the three
`startswith` prefixes, the server regexes (including multi-digit servers), and
pass-through for unrelated errors.

### Proper long-term fix (blocked)

Register the C++ exception with pybind11 (`py::register_exception<
dqc::ConstraintException>`) and expose `server` / `v_gate` members. Then
`translate_core_error` becomes a no-op and `.v_gate` parity is exact. Blocked
by the "no C++ changes" rule; record it in `PLAN.md` as a forfeit.

### Provenance note

Update `tests/README.md`: its "Modifications" list currently claims only the
import rename, `test_nisq_draw` removal and the KaHyPar INI path. List every
adapter-owned test file added, otherwise the "tests are verbatim upstream"
guarantee becomes ambiguous.

### Acceptance criteria for Fix 4

- [ ] `tests/error_parity_test.py` covers: type, `.server` equality,
      message prefix, documented `.v_gate is None`, regex parsing for all three
      message shapes, and pass-through for non-constraint errors.
- [ ] No vendored test is modified.
- [ ] `tests/README.md` lists the new file.

---

## 3. Gates are manual and partly skipped

### What runs where

Only one of the five gates is pytest:

| Gate | Command | Collected by pytest? |
| --- | --- | --- |
| Vendored suite | `pytest tests -m "not high_compute"` | yes |
| Slow suite | `pytest tests -m high_compute` | yes (deselected by default) |
| Core differential | `tests/python/check_equivalence.py` | no |
| Adapter differential | `tests/python/check_adapter_equivalence.py` | no |
| C++ | `ctest --test-dir build` | no |

`pytest.ini` sets `testpaths = tests` and
`python_files = *_test.py test_*.py`, so `pytest` at the repo root runs **only**
the vendored suite. There is no `.github/workflows`, `Makefile`, `nox`/`tox`, or
pre-commit hook — nothing fails when the differentials regress. The plan's
final gate (§5.5) is therefore a human checklist.

### Five specific hazards

**1. `PYTHONHASHSEED` silently disables the sweep.**

```python
if os.environ.get("DQC_ADAPTER_HASH_SWEEP") != "1" and "PYTHONHASHSEED" not in os.environ:
```

A CI image or `.env` that exports `PYTHONHASHSEED=0` gets **1/4 of the
coverage** while still printing `ALL ADAPTER EQUIVALENCE CHECKS PASSED`.

Fix: key the opt-out on an explicit variable only (e.g.
`DQC_ADAPTER_SINGLE_SEED=1`), never on an inherited env var.

**2. Zero checks still exits 0.**

```python
print(f"{CHECKS[0]} checks, {len(FAILURES)} failures")
if FAILURES:
    return 1
print("ALL ADAPTER EQUIVALENCE CHECKS PASSED")
return 0
```

An early return (or an exception path that stops the generator) yields
`0 checks, 0 failures` -> exit 0 -> "PASSED". Add a floor and a skip ceiling:

```python
MIN_CHECKS = 735
MAX_SKIPS = 4
...
if CHECKS[0] < MIN_CHECKS:
    FAILURES.append(f"only {CHECKS[0]} checks ran")
if SKIPPED[0] > MAX_SKIPS:
    FAILURES.append(f"{SKIPPED[0]} cases skipped")
```

(Update `MIN_CHECKS` after the `check_graph_tools` addition: the expected value
is ~739.)

**3. Skips are silent and unbounded.**

- `allocation_cases()` prints `skip <tag>: ...` and `continue`s for 4 of the 15
  circuit × network cases per seed ("This circuit cannot be implemented on this
  network" — a genuine reference limitation).
- `check_random_allocator` / `check_annealing_allocator` silently `return` for
  circuits with more than 5 qubits.
- `check_refiners` skips `VertexCover` above 40 vertices.
- `check_brute_allocator` runs on two tiny dedicated inputs only.

If a future change made *every* case skip, the script would still be green.
Count the skips (see `MAX_SKIPS`) and print the ratio
(`11/15 cases compared`).

**4. The default gate excludes the slow determinism checks.**

`high_compute` is deselected by the documented everyday command and is
"optional" in the plan, so the every-day green run never covers those 7 tests
(~178 s).

**5. The installed layout is never exercised.**

`tests/conftest.py` does `sys.path.insert(0, python/)`, so pytest always imports
the source tree, never an installed `pytket_dqc_cpp._core`. Combined with the
packaging gap (no `[build-system]`, an empty wheel), the "installed-extension"
branch of `_core.py` has zero coverage.

Related, tracked separately: `pyproject.toml` has no `[build-system]` or package
discovery, so `uv build` / `pip install .` produces a wheel containing only
`dist-info`. The README claim "or installing the directory" is false; only
`PYTHONPATH=python` works.

### Remediation

**One entry point:**

```bash
# tests/python/run_gates.sh
set -euo pipefail
unset PYTHONHASHSEED                 # keep the hash sweep honest
python -m pytest tests -m "not high_compute"
python tests/python/check_equivalence.py
python tests/python/check_adapter_equivalence.py
ctest --test-dir build --output-on-failure
```

**CI:** a workflow that builds (`cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
-DQC_BUILD_BINDINGS=ON`, then `cmake --build build -j`) and runs
`tests/python/run_gates.sh`, plus a nightly job running `-m high_compute`.

**Optionally, a single `pytest` entry point:**

```ini
# pytest.ini
markers =
    high_compute: ...
    differential: cross-package equivalence harnesses (slow)
addopts = -m "not high_compute and not differential"
```

### Acceptance criteria for the gates work

- [ ] `tests/python/run_gates.sh` exists, is executable, and fails on the first
      failing gate (`set -euo pipefail`).
- [ ] CI runs that script; the build step is part of the same workflow.
- [ ] The differential asserts a minimum check count and a maximum skip count.
- [ ] The hash-seed sweep can only be disabled explicitly
      (`DQC_ADAPTER_SINGLE_SEED=1`), never by an inherited `PYTHONHASHSEED`.
- [ ] The 7 upstream skips are listed in `tests/README.md` with their reasons.

---

## 4. Suggested order of work

| # | Item | File(s) | Effort | Risk |
| --- | --- | --- | --- | --- |
| 1 | Skip ceiling + min-check guard | `tests/python/check_adapter_equivalence.py` | S | none |
| 2 | Sweep opt-out keyed on explicit var | same | S | none |
| 3 | `tests/error_parity_test.py` (Fix 4) | new test file | S | none |
| 4 | `tests/rng_semantics_test.py` (Fix 3) | new test file | S | none |
| 5 | Docs: README + docstrings (Fix 3) | `README.md`, refiners/annealing modules | S | none |
| 6 | `Annealing` default-initializer warning | `allocators/annealing.py` | S | low (changes warning output only) |
| 7 | `run_gates.sh` + CI workflow | `tests/python/`, `.github/workflows/` | M | none |
| 8 | `pyproject.toml` packaging (separate item) | `pyproject.toml` | S | low |

Do items 1–4 first: they convert untested claims into assertions without
touching runtime behaviour.

---

## 5. Verification commands

```bash
# 1. Reproduce the current gate state
PYTHONPATH=python .venv/bin/python -m pytest tests -m "not high_compute" -q
PYTHONPATH=python .venv/bin/python tests/python/check_adapter_equivalence.py
ctest --test-dir build --output-on-failure

# 2. Confirm the new graph-tools section is live (expect ~739 checks)
PYTHONPATH=python .venv/bin/python tests/python/check_adapter_equivalence.py | tail -2

# 3. Re-run the mutation that previously survived (must now FAIL)
mkdir -p /tmp/mut && cp -r python tests /tmp/mut/ && ln -s "$PWD/build" /tmp/mut/build
#   edit /tmp/mut/python/pytket_dqc_cpp/utils/graph_tools.py and delete the
#   `for u, v in tree.edges(): ...` restoration loop
cd /tmp/mut && PYTHONPATH=python /home/ubuntu/pytket-dqc-cpp/.venv/bin/python \
    tests/python/check_adapter_equivalence.py | tail -3

# 4. Sanity-check the RNG claims after Fix 3 lands
PYTHONPATH=python .venv/bin/python -m pytest tests/rng_semantics_test.py -q

# 5. Sanity-check error parity after Fix 4 lands
PYTHONPATH=python .venv/bin/python -m pytest tests/error_parity_test.py -q
```

Quality bar for mutation testing: a semantic change to any adapter module should
fail at least one gate. The two mutations that previously failed as intended:
`Distribution.cost()` returning `...+ 1` (20 failures) and
`get_qubit_vertices()` returning the reversed list (17 failures).
