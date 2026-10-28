#!/usr/bin/env python3
"""Differential test for the QFT distribution benchmarks.

The four benchmark drivers in ``tests/python/qft/`` run the same QFT workloads
through two independent implementations:

* ``main_few_qpus_py.py`` / ``main_many_qpus_py.py`` -- the reference Python
  ``pytket_dqc`` package (results in ``results_few_qpus_py/`` /
  ``results_many_qpus_py/``),
* ``main_few_qpus_cpp.py`` / ``main_many_qpus_cpp.py`` -- the C++ adapter
  ``pytket_dqc_cpp`` package (results in ``results_few_qpus_cpp/`` /
  ``results_many_qpus_cpp/``).

This checker runs all four, then compares the result files field by field with
the same line parsing the plot scripts use, ignoring the two timing lines (the
C++ port is expected to be faster).  The plot scripts are then run for both
variants so the generated charts can be compared by eye in the two result
directories.

Run with the project venv (from anywhere):

    .venv/bin/python3 tests/python/check_qft_equivalence.py

Options:
    --skip-computation   do not re-run the benchmark drivers; still regenerate
                         the plots and compare the existing result files
    --skip-plots         do not regenerate the plots

Exits non-zero on a failing benchmark script or on any mismatch.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

QFT_DIR = Path(__file__).resolve().parent / "qft"

# Both variants must hash strings with the same secret: the set iteration order
# of `Packet` objects (SipHash of "P<i>") feeds the refiner results.  Override
# with QFT_PYTHONHASHSEED if a different seed is wanted.
HASH_SEED = os.environ.get("QFT_PYTHONHASHSEED", "12345")

METHOD_RESULT_FILENAMES = (
    "results-data-pytket-pe.txt",
    "results-data-pytket-aesd.txt",
    "results-data-pytket-phe.txt",
)

BENCHMARKS = ("few", "many")
VARIANTS = ("py", "cpp")


def run_script(script, *args):
    cmd = [sys.executable, str(script), *args]
    print(f"$ {' '.join(cmd)}", flush=True)

    env = dict(os.environ)
    env["PYTHONHASHSEED"] = HASH_SEED
    env["MPLBACKEND"] = "Agg"
    repo_python = str(QFT_DIR.parents[2] / "python")
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{repo_python}:{existing_pp}" if existing_pp else repo_python

    subprocess.run(cmd, cwd=QFT_DIR, env=env, check=True)


def result_paths(benchmark, variant):
    """Result files written by one variant of a benchmark."""
    root = QFT_DIR / f"results_{benchmark}_qpus_{variant}" 
    if benchmark == "many":
        dirs = [root]
    else:
        dirs = [root / f"{n}_qpus" for n in (2, 4)]
    return [directory / method_file for directory in dirs for method_file in METHOD_RESULT_FILENAMES]


def parse_results(path):
    """Parse a benchmark result file.

    Mirrors the line handling of ``plot_few_qpus.py`` / ``plot_many_qpus.py``,
    but keys each record by its ``(qpu_size, num_qubits, num_partitions)``
    header instead of relying on list positions.

    Returns ``{key: {"min_ebit", "detached", "time", "time_fixing"}}``.
    """
    records = {}
    key = None
    with path.open(encoding="utf-8") as f:
        for line in f:
            if "qpu_size" in line:
                fields = {
                    k.strip(): v.strip()
                    for k, v in (part.split("=") for part in line.strip().split(","))
                }
                key = (
                    int(fields["qpu_size"]),
                    int(fields["num_qubits"]),
                    int(fields["num_partitions"]),
                )
                records[key] = {}
            elif key is None:
                continue
            elif "Min e-bit count" in line:
                records[key]["min_ebit"] = int(line.split(":")[1].strip())
            elif "Time taken" in line:
                records[key]["time"] = float(line.split(":")[1].split()[0])
            elif "Time_fixing taken" in line:
                records[key]["time_fixing"] = float(line.split(":")[1].split()[0])
            elif "Detached count" in line:
                records[key]["detached"] = int(line.split(":")[1].split()[0])
    return records


def compare_benchmark(benchmark, mismatches):
    """Compare every result file pair; timing fields are ignored."""
    for py_path, cpp_path in zip(result_paths(benchmark, "py"),
                                 result_paths(benchmark, "cpp")):
        label = str(py_path.relative_to(QFT_DIR))

        python_records = parse_results(py_path)
        cpp_records = parse_results(cpp_path)

        print(f"{label}:")
        for key in sorted(set(python_records) | set(cpp_records)): # key == qpu_size, num_qubits, num_partitions
            qpu_size, num_qubits, num_partitions = key

            p = python_records.get(key)
            c = cpp_records.get(key)

            for field in ("min_ebit", "detached"):
                if p.get(field) != c.get(field):
                    mismatches.append(
                        f"{label} ({qpu_size=}, {num_qubits=}, {num_partitions=}): {field} "
                        f"py={p.get(field)} cpp={c.get(field)}")

            print(f"  ({qpu_size=}, {num_qubits=}, {num_partitions=}): "
                  f"e-bit {p.get('min_ebit')}/{c.get('min_ebit')}, "
                  f"detached {p.get('detached')}/{c.get('detached')}, "
                  f"(py/cpp)")


def main(argv=None):
    # Args
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--skip-computation", action="store_true",
        help="skip the benchmark drivers; still regenerate the plots and "
             "compare the existing result files")
    parser.add_argument(
        "--skip-plots", action="store_true",
        help="do not regenerate the plots")
    args = parser.parse_args(argv)


    # Intro print statement
    print(f"QFT benchmark differential test (PYTHONHASHSEED={HASH_SEED})")
    print(f"scripts and results: {QFT_DIR}\n")

    # Getting results
    if args.skip_computation:
        print("--skip-computation: reusing the existing result files\n")
    else:
        for benchmark in BENCHMARKS:
            for variant in VARIANTS:
                run_script(QFT_DIR / f"main_{benchmark}_qpus_{variant}.py")

    if not args.skip_plots:
        print()
        for benchmark in BENCHMARKS:
            for variant in VARIANTS:
                plot_args = ["--cpp"] if variant == "cpp" else []
                run_script(QFT_DIR / f"plot_{benchmark}_qpus.py", *plot_args)

        print("\nplots for visual comparison (same chart names in both dirs):")
        for benchmark in BENCHMARKS:
            for variant in VARIANTS:
                path = QFT_DIR / f"results_{benchmark}_qpus_{variant}"
                print(f"  {benchmark}: {path}")

    # Checking results
    mismatches = []
    print()
    for benchmark in BENCHMARKS:
        compare_benchmark(benchmark, mismatches)

    if mismatches:
        print("\nMISMATCHES (timing lines are ignored):")
        for mismatch in mismatches:
            print(f"  {mismatch}")
        print(f"\nFAIL: {len(mismatches)} mismatch(es)")
        return 1

    print("\nOK: all QFT benchmark results match between python and cpp "
          "(timing excluded)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
