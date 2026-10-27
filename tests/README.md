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

## Commands

```bash
PYTHONPATH=python .venv/bin/python -m pytest tests -m "not high_compute"
PYTHONPATH=python .venv/bin/python -m pytest tests -m high_compute        # slow, optional
PYTHONPATH=python .venv/bin/python tests/python/check_adapter_equivalence.py
```

`tests/python/check_*.py` are differential scripts, not pytest modules
(`pytest.ini` restricts `python_files` to `*_test.py` / `test_*.py`).
