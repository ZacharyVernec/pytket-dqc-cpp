"""Test bootstrap for the vendored upstream suite.

Adds the in-repo adapter package (``python/``) to ``sys.path`` so that
``import pytket_dqc_cpp`` works without setting ``PYTHONPATH``. The compiled
extension is located by ``pytket_dqc_cpp._core`` itself (it falls back to
``build/dqc_cpp``).

The vendored tests read fixtures through paths relative to the repository root
(``tests/test_circuits/...``), so run pytest from the repository root.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PYTHON_DIR = _REPO_ROOT / "python"

if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))
