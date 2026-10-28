# Fix 3: ConstraintException Attribute Parity & Error Details

**Status:** Implemented  
**Scope:** `cpp/python/bindings.cpp`, `cpp/include/dqc/to_pytket_circuit.hpp`, `python/pytket_dqc_cpp/_bridge.py`, and `tests/error_parity_test.py`  
**Severity:** Medium (Diagnostic clarity & test integrity)  

---

## 1. Executive Summary

When a distributed quantum circuit violates hardware topology or memory constraints (e.g., exceeding the communication qubit capacity of a server node), the system raises a `ConstraintException`.

In the original Python library, this exception included valuable debugging attributes:
- `e.server`: The index of the server node whose capacity was exceeded.
- `e.v_gate`: The vertex index in the hypergraph circuit corresponding to the offending quantum gate.
- A helpful hint in the message advising the user to try `to_pytket_circuit(allow_update=True)`.

### Resolution Implemented (Approved Decision):
By user decision, the scope restriction on C++ modifications was lifted to implement exact exception attribute parity directly at the pybind11 boundary:
1. `cpp/python/bindings.cpp` catches `dqc::ConstraintException` during `to_pytket_circuit()` and translates it into `pytket_dqc_cpp.utils.circuit_analysis.ConstraintException`, directly binding `e.server` and `e.v_gate`.
2. `cpp/include/dqc/to_pytket_circuit.hpp` includes the exact upstream remediation hint string (`to_pytket_circuit(allow_update=True)`).
3. `tests/error_parity_test.py` verifies exception type, `e.server`, `e.v_gate`, and message guidance hints directly from `to_pytket_circuit()`.
4. The vestigial `translate_core_error` regex parser and try/except conversion in `_bridge.py` and `distribution.py` have been removed since `ConstraintException` propagates directly out of pybind11.

---

## 2. Background: Constraint Checks in DQC

Converting a high-level distributed circuit into physical server operations happens in `distribution.to_pytket_circuit()`. During this process, the compiler checks:
1. **Communication Memory Limits:** Each QPU server node has a limited number of communication qubits available to store shared EPR entangled pairs.
2. **Gate Routing Constraints:** Non-home servers executing distributed gates must have available communication channels.

If a gate cannot be scheduled without violating these physical limits, execution aborts with a `ConstraintException`.

---

## 3. The Divergence Between Python and C++

### Upstream Python Handling:
In `pytket_dqc/circuits/distribution.py`:

```python
except ConstraintException as e:
    e.v_gate = v_gate  # Annotates exception with the specific gate index
    # Error string ends with:
    # "... The distribution could not be amended. Re-distribute the circuit or try to_pytket_circuit(allow_update=True)."
    raise e
```

### C++ Adapter Handling:
In `python/pytket_dqc_cpp/_bridge.py`:

```python
_SERVER_PATTERNS = [
    re.compile(r"Communication memory capacity of server (\d+)"),
    re.compile(r"non-home server (\d+)"),
]

def translate_core_error(exc: Exception) -> Exception:
    msg = str(exc)
    if any(msg.startswith(p) for p in _CONSTRAINT_PREFIXES):
        server = None
        for pattern in _SERVER_PATTERNS:
            m = pattern.search(msg)
            if m:
                server = int(m.group(1))
                break
        return ConstraintException(msg, server=server, v_gate=None)
    return exc
```

### The Gap Analysis:

| Attribute / Field | Upstream Python | C++ Adapter | Status |
| :--- | :--- | :--- | :--- |
| **Exception Type** | `ConstraintException` | `ConstraintException` | Matching |
| **`e.server` (capacity exceed)** | Integer (e.g., `0`) | Integer | Matching |
| **`e.server` (multi-digit server)** | Integer (e.g., `12`) | Integer | Matching |
| **`e.server` (unroutable gate)** | `None` | `None` | Matching |
| **`e.v_gate`** | Integer (gate vertex ID) | Integer (e.g. `11`) | **Matching (via pybind11 translation)** |
| **Hint in error message** | `"try ... allow_update=True"` | Present in C++ message | Matching |

---

## 4. The Upstream Test Flaw

In upstream `tests/circuits_test.py` (around line 936):

```python
caught = False
try:
    distribution.to_pytket_circuit()
except ConstraintException as e:
    e.server = 3      # <--- BUG IN TEST! This assigns 3, it does NOT assert 3!
    caught = True
assert caught
```

Because of this typo in upstream's test, if our adapter returned `e.server = None`, the test would still pass! The test verified that an exception was caught, but completely failed to verify the attribute values. Furthermore, running on that network causes server 0's capacity to be exceeded first, so an assertion `assert e.server == 3` would fail even in the reference implementation.

---

## 5. Implementation Steps & Verification

### Step 1: Preserve Vendored Upstream Tests
We maintain byte-level fidelity with the upstream test suite (`tests/circuits_test.py`) so we can continue to verify that we introduce zero regressions against upstream test fixtures.

### Step 2: C++ Core and pybind11 Update
In `cpp/include/dqc/to_pytket_circuit.hpp`, the communication capacity error message includes the upstream guidance hint:
```cpp
throw ConstraintException(
    "Communication memory capacity of server " +
        std::to_string(server) +
        " exceeded. \n\tConsider setting allow_update to True: "
        "\n\tto_pytket_circuit(allow_update=True)",
    server);
```
In `cpp/python/bindings.cpp`, `to_pytket_circuit` translates `ConstraintException` directly to the Python exception class, populating `v_gate` and `server`:
```cpp
catch (const ConstraintException& e) {
  py::object mod = py::module_::import("pytket_dqc_cpp.utils.circuit_analysis");
  py::object py_exc_cls = mod.attr("ConstraintException");
  py::object py_exc = py_exc_cls(e.what(), e.server);
  if (e.v_gate >= 0) {
    py_exc.attr("v_gate") = e.v_gate;
  }
  py::set_error(py_exc_cls, py_exc);
  throw py::error_already_set();
}
```

### Step 3: Adapter-Owned Test File (`tests/error_parity_test.py`)
`tests/error_parity_test.py` validates that:
- `test_constraint_exception_v_gate_parity`: `e.server == 0` and `e.v_gate == 11`.
- `test_constraint_exception_hint_parity`: `"to_pytket_circuit(allow_update=True)" in str(e)`.

### Step 4: Removal of Vestigial Code
`translate_core_error` and regex pattern matchers were removed from `_bridge.py` and `circuits/distribution.py` as `ConstraintException` is now handled natively by the extension.

### Step 5: Update `tests/README.md`
The provenance log in `tests/README.md` documents `tests/error_parity_test.py` as an adapter-specific parity suite.
