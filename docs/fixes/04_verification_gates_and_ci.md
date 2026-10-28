# Fix 4: Verification Gates Hardening & Elimination of Silent Skips

**Status:** Implemented  
**Scope:** `tests/python/check_adapter_equivalence.py`, `tests/python/run_gates.sh`  
**Severity:** High (Prevents silent test bypasses and false-positive runs)  

---

## 1. Executive Summary

We have built extensive testing harnesses to verify that our C++ adapter (`pytket-dqc-cpp`) matches the upstream Python library (`pytket-dqc`). Between unit tests, C++ tests, and differential equivalence sweeps, hundreds of checks exist across the codebase.

However, an audit of the testing infrastructure revealed critical blind spots:
1. **The False-Positive Trap:** The differential test script can report `ALL ADAPTER EQUIVALENCE CHECKS PASSED` and exit with code `0` (success) even if checks terminate prematurely.
2. **Silent & Unbounded Skips:** Cases in `check_adapter_equivalence.py` currently skip silently without asserting identical behavior (e.g., when circuit qubits exceed network capacity, or on circuit size thresholds).
3. **The Environment Trap:** If an ambient environment variable like `PYTHONHASHSEED` is present, 75% of the differential hash-invariance sweep is silently bypassed.
4. **Fragmented Execution:** Verification is split across five distinct tiers (C++ unit tests, Python fast unit tests, slow determinism tests, core differential, adapter differential), requiring disparate manual commands.

**Scope Boundaries:**
- **Automated CI (GitHub Actions):** Excluded from this fix.
- **Packaging (`pyproject.toml` / wheels):** Deferred to future work.
- **Test Integration:** Differential suites remain standalone executable scripts (no integration into `pytest`).
- **Core Directive:** **All planned checks must run.** Skips must not be silently bypassed; any skipped case is tracked by count and treated as a gate failure (`SKIPPED[0] == 0`).

---

## 2. Background: The Five Verification Tiers

The project currently depends on five verification suites:

| Tier | Gate | Command | Execution Time | Captured by `pytest`? |
| :--- | :--- | :--- | :--- | :--- |
| **1** | Vendored Python Suite | `pytest tests -m "not high_compute"` | ~15 s | Yes |
| **2** | Slow Determinism Suite | `pytest tests -m high_compute` | ~178 s | Deselected by default |
| **3** | Core Differential | `python tests/python/check_equivalence.py` | ~10 s | No (standalone script) |
| **4** | Adapter Differential | `python tests/python/check_adapter_equivalence.py` | ~45 s | No (standalone script) |
| **5** | C++ Unit Tests | `ctest --test-dir build` | ~5 s | No (standalone CMake) |

Because developers typically run only `pytest`, regressions in Tiers 3, 4, and 5 can go unnoticed without a single unified gate entry point.

---

## 3. The Hazards & Required Fixes

### Hazard 1: Zero Checks Still Reports Green
In `tests/python/check_adapter_equivalence.py`:

```python
print(f"{CHECKS[0]} checks, {len(FAILURES)} failures")
if FAILURES:
    return 1
print("ALL ADAPTER EQUIVALENCE CHECKS PASSED")
return 0
```

If an unhandled exception or bug prematurely terminates test loops before running checks, `CHECKS[0]` might be low or zero, yet `FAILURES` remains empty. The script prints `0 failures` and exits with code `0`.

### Hazard 2: `PYTHONHASHSEED` Silently Drops Coverage
In `check_adapter_equivalence.py`:

```python
if os.environ.get("DQC_ADAPTER_HASH_SWEEP") != "1" and "PYTHONHASHSEED" not in os.environ:
    # Run loop over 4 different hash seeds (0, 1, 42, 12345)
```

If the developer's shell or container environment has `PYTHONHASHSEED` set, the script silently executes only a single seed instead of sweeping across all four seeds, silently dropping **75% of differential coverage**.

### Hazard 3: Skips Are Silent and Unbounded (Must Be Fixed)
Currently, multiple testing paths silently skip execution rather than executing and verifying checks:
1. **Network Capacity in `allocation_cases()`:**
   - 4 of the 15 circuit × network pairings (`layered/net0`, `random0/net0`, `random1/net0`, `random2/net0`) fail because the circuit requires 5 qubits while `net0` provides only 4 qubits.
   - The script catches the exception with `except Exception as exc:` and immediately executes `continue`, skipping the entire test suite for those combinations.
   - *Fix:* Test inputs must be aligned so valid combinations execute completely. Furthermore, for invalid/oversubscribed combinations, the harness must explicitly assert that **both** `ref` and `new` raise identical capacity exceptions (`outcome(ref) == outcome(new)`), rather than skipping.
2. **Qubit Limits in `check_random_allocator`:**
   - Contains a silent `if len(circuit.qubits) > 5: return`.
   - *Fix:* Eliminate the bypass so that all circuits allocated to compatible networks are checked.
3. **Vertex Limits in `check_refiners`:**
   - Contains a silent `if len(new_dist.circuit.vertex_list) <= 40:` for `VertexCover`.
   - *Fix:* Remove arbitrary bypasses so that refiner checks execute deterministically across all test cases.

### Hazard 4: Slow Determinism Tests Are Bypassed in Daily Dev
Because `pytest tests -m "not high_compute"` is the default command, the 7 slow determinism tests (~178 s) are not executed regularly.

---

## 4. Implementation Steps

### Step 1: Eliminate Skips & Enforce Complete Execution in `check_adapter_equivalence.py`
Instead of using loose minimum check floors or maximum skip ceilings:
1. **Fix All Skip Sites:**
   - Convert silent `continue` blocks in `allocation_cases()` into explicit exception parity checks (`check(f"{tag}: capacity exception parity", ref_outcome == new_outcome)`).
   - Remove silent size-based `return` guards in `check_random_allocator` and `check_refiners`.
2. **Enforce That All Planned Checks Run:**
   - Maintain an exact expectation of planned checks: `EXPECTED_CHECKS`.
   - Require `CHECKS[0] == EXPECTED_CHECKS` at script completion. Any mismatch fails the run.
3. **Track and Enforce Zero Skips:**
   - Track every skipped case explicitly in a counter (`SKIPPED[0]`).
   - If `SKIPPED[0] > 0`, report the exact skip count and treat it as a verification failure. Zero skips permitted.
4. **Enforce Hash Sweep by Default:**
   - Always run the 4-seed sweep (`0`, `1`, `42`, `12345`) regardless of ambient `PYTHONHASHSEED`.
   - Only allow a single-seed run if an explicit opt-out variable is set (`DQC_ADAPTER_SINGLE_SEED=1`).

```python
# Completion assertion in check_adapter_equivalence.py:
if SKIPPED[0] > 0:
    FAILURES.append(
        f"Verification failure: {SKIPPED[0]} cases were skipped. All checks are required to run."
    )

if CHECKS[0] != EXPECTED_CHECKS:
    FAILURES.append(
        f"Verification failure: ran {CHECKS[0]} checks, expected exactly {EXPECTED_CHECKS}."
    )
```

### Step 2: Create Unified Local Gate Runner (`tests/python/run_gates.sh`)
Create an executable script that runs all active verification tiers in sequence with immediate failure on error:

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "=== 1. Building C++ Core (Release) ==="
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DDQC_BUILD_BINDINGS=ON
cmake --build build -j

echo "=== 2. Running C++ Unit Tests (Tier 5) ==="
ctest --test-dir build --output-on-failure

echo "=== 3. Running Python Fast Suite (Tier 1) ==="
PYTHONPATH=python python3 -m pytest tests -m "not high_compute"

echo "=== 4. Running Core Differential Equivalence (Tier 3) ==="
PYTHONPATH=python python3 tests/python/check_equivalence.py

echo "=== 5. Running Adapter Differential Equivalence (Tier 4) ==="
PYTHONPATH=python python3 tests/python/check_adapter_equivalence.py

echo "=== ALL VERIFICATION GATES PASSED SUCCESSFULLY ==="
```

---

## 5. Verification Commands

Run the full unified verification sequence locally:

```bash
chmod +x tests/python/run_gates.sh
./tests/python/run_gates.sh
```

To test the safety guardrails:
1. **Artificial Skip:** Inject a skip condition; the runner must abort, report that cases were skipped (`Verification failure: X cases were skipped`), and exit non-zero.
2. **Missing Check:** Artificially break a loop; the runner must abort due to `CHECKS[0] != EXPECTED_CHECKS` and exit non-zero.
3. **Single Seed Override:** Verify that `DQC_ADAPTER_SINGLE_SEED=1 python tests/python/check_adapter_equivalence.py` runs only the single pass when explicitly requested.

---

## 6. Summary of Decisions

1. **All Checks Must Run:** Rather than permitting a ceiling of skipped cases, all test paths are required to run and test parity (including exception parity on capacity-constrained networks).
2. **Strict Skip Count Enforcement:** Any skipped case is tracked by count (`SKIPPED[0] > 0`) and immediately fails the verification gate.
3. **Standalone Gate Execution:** Differential tests remain clean standalone scripts orchestrated by `run_gates.sh` without being forced into `pytest`.
4. **CI & Packaging Excluded:** Automated CI workflows and `pyproject.toml` packaging setup are omitted from this work item.
