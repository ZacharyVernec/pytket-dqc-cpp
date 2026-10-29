#!/usr/bin/env bash
set -euo pipefail

# Ensure execution from the repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# Activate virtual environment if present
if [ -f ".venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

# Ensure ambient hash seed doesn't bypass multi-seed sweeps
unset PYTHONHASHSEED

echo "=== 1. Running C++ Unit Tests (Tier 5) ==="
ctest --test-dir build --output-on-failure

echo "=== 2. Running Python Fast Suite (Tier 1) ==="
PYTHONPATH=python python3 -m pytest tests -m "not high_compute"

echo "=== 3. Running Core Differential Equivalence (Tier 3) ==="
PYTHONPATH=python python3 tests/python/check_equivalence.py

echo "=== 4. Running Adapter Differential Equivalence (Tier 4) ==="
PYTHONPATH=python python3 tests/python/check_adapter_equivalence.py

echo "=== 5. Running QFT Benchmark Differential Equivalence ==="
python3 tests/python/check_qft_equivalence.py

echo "=== ALL VERIFICATION CHECKS PASSED SUCCESSFULLY ==="
