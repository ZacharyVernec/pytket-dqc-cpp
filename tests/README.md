# Vendored upstream test suite

These tests are copied from the reference implementation
[`Quantinuum/pytket-dqc`](https://github.com/Quantinuum/pytket-dqc) at commit
`1ac175ed820ac02366b2ed71b653d5b96fc72ff7` (the commit pinned by the installed
`pytket_dqc` oracle, see `direct_url.json`), and are licensed under the Apache
License 2.0, same as the upstream project.

They are the interface acceptance gate for the `pytket_dqc_cpp` adapter.

## Modifications

- Every `pytket_dqc` import root is rewritten to `pytket_dqc_cpp` (the test
  bodies are otherwise unchanged, including expected exception messages).
- `network_test.py::test_nisq_draw` is removed: drawing is out of scope.
- `allocator_test.py::test_kahypar_install` loads the INI file from
  `pytket_dqc_cpp/allocators/km1_kKaHyPar_sea20.ini` via `pathlib` instead of
  `importlib_resources.files("pytket_dqc")`.
- `tests/test_circuits/` (JSON and pickle fixtures) is copied verbatim. The tests
  open them through paths relative to the repository root, so run pytest from the
  repository root.
- `conftest.py` is adapter-only (adds `python/` to `sys.path`); there is no
  `tests/__init__.py` (rootdir-relative import mode is enough).
- `rng_semantics_test.py` and `error_parity_test.py` are adapter-specific test suites
  covering RNG reproducibility semantics and strict `ConstraintException` API parity
  against the upstream reference.

## Upstream Skips in Vendored Suite

When running `pytest tests -m "not high_compute"` (which executes 159 tests: 152 passed, 7 skipped), 7 tests are skipped upstream.
These are identical to the skips present in the upstream `pytket-dqc@1ac175e` repository:

1. `tests/allocator_test.py:152` (`test_refinement_makes_valid`): "Circuit contains CX gates that are not supported."
2. `tests/allocator_test.py:373` (`test_q_control_box_circuits`): "QControlBox are not supported for now"
3. `tests/circuits_test.py:439` (`test_q_control_box_circuits`): "QControlBox are not supported for now"
4. `tests/circuits_test.py:1693` (`test_to_pytket_circuit_with_teleportation`): "Support for teleportation has been disabled"
5. `tests/circuits_test.py:1791` (`test_to_relabeled_registers`): "Tests a function that has been removed"
6. `tests/network_test.py:122` (`test_nisq_get_placer`): "This test needs to be fixed."
7. `tests/utils_test.py:56` (`test_CX_circuit`): "Support for CX gates temporarily disabled"

## Deselected High-Compute Tests

When running `pytest tests -m "not high_compute"`, 7 tests marked `@pytest.mark.high_compute` are deselected before execution. These are valid, heavyweight integration tests (10-qubit fixtures, KaHyPar + refiners, and a PyZX `check_equivalence` round-trip) that take substantial execution time (~180s total). Run them with `pytest tests -m high_compute`:

1. `tests/circuits_test.py:1574` (`test_to_pytket_circuit_with_pauli_circ`)
2. `tests/circuits_test.py:1684` (`test_to_pytket_circuit_pr78_bug`)
3. `tests/circuits_test.py:2015` (`test_embedding_detached`)
4. `tests/refiner_test.py:865` (`test_vertex_cover_refiner_pauli_circ`)
5. `tests/refiner_test.py:922` (`test_vertex_cover_refiner_random_circ`)
6. `tests/refiner_test.py:1031` (`test_vertex_cover_embedding_boundary_failure`)
7. `tests/refiner_test.py:1443` (`test_boundary_reallocation_refiner_pauli_circ`)

## Commands

Run the full unified verification runner:

```bash
./tests/python/run_all_checks.sh
```

Or run individual tiers:

```bash
PYTHONPATH=python .venv/bin/python -m pytest tests -m "not high_compute"
PYTHONPATH=python .venv/bin/python -m pytest tests -m high_compute        # slow, optional
PYTHONPATH=python .venv/bin/python tests/python/check_equivalence.py       # Core differential
PYTHONPATH=python .venv/bin/python tests/python/check_adapter_equivalence.py # Adapter differential
.venv/bin/python3 tests/python/check_qft_equivalence.py                    # QFT benchmark differential
ctest --test-dir build --output-on-failure                                 # C++ Catch2 suite
```

`tests/python/check_*.py` are differential scripts, not pytest modules
(`pytest.ini` restricts `python_files` to `*_test.py` / `test_*.py`).
