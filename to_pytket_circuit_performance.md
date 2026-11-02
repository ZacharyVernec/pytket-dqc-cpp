# `Distribution.to_pytket_circuit`: algorithmic changes and measured runtime

This note documents the performance work on the C++ port of
`Distribution.to_pytket_circuit` (2026-11), in particular the
`allow_update=True` path, which is the slowest part of the pipeline for
distributed QFT benchmarks. The observable behaviour is unchanged: the emitted
circuit, the estimated/actual e-bit cost and the (possibly split) hypergraph are
identical to the reference Python implementation.

All changes are in [`cpp/include/dqc/to_pytket_circuit.hpp`](cpp/include/dqc/to_pytket_circuit.hpp);
no Python reference code is touched.

## Background: what the reference does, and why it is slow

`to_pytket_circuit(satisfy_bound=True, allow_update=False)` walks the
distribution's hyperedges, and for each one re-emits the whole circuit so far,
inserting `starting_process` / `ending_process` teleportation commands and the
CU1 gates for the hyperedge's servers, while tracking link-qubit occupation in
`LinkManager`. When a server's communication memory bound is exceeded the
reference raises `ConstraintException`; with `allow_update=True` it splits **one**
hyperedge (`split_for_bound`) and then **restarts the whole construction from
the initial circuit** (`return self.to_pytket_circuit(allow_update=True)`).

That restart is the dominant cost. Profiling the port on a 16-qubit QFT
distribution (30 hyperedges, 136 source commands, memory bound 2):

| before the change | value |
| --- | --- |
| construction passes (restarts) | **138** |
| `one_hyperedge` invocations | **10 891** (for a final hypergraph of 167 hyperedges) |
| command copies (`BuiltCommand`, strings included) | 3.6 M |
| barrier commands pushed / barrier qubit entries | 1.3 M / 22.4 M |
| scratch-array cells zeroed in `get_commands_order()` | **8.6 × 10⁹** |
| wall time (core only) | **10.0 s** |

The same pattern at 20 qubits: 258 passes, 36 768 `one_hyperedge` calls,
8.9 × 10¹⁰ scratch cells, 79 s.

Because each pass redoes `O(K · C)` work for `K` hyperedges (and both `K` and the
command count `C` grow as hyperedges are split), the cost is roughly `O(K² · C)`
— i.e. the runtime grows much faster than the circuit itself.

## Change 1 — resume instead of restarting the build

The restart is *unobservable*, which makes it removable:

1. `Hypergraph::split_hyperedge(old, {h1, h2})` inserts the two halves at the
   **old hyperedge's index** and removes the old one. Every hyperedge before that
   index is untouched.
2. `one_hyperedge(hedge, circuit, ctx)` is a pure function of the hyperedge and
   the circuit built so far: `LinkManager` is constructed per hyperedge and
   `ctx` (placement, qubit mapping, server list, memory bounds) is not modified
   by a split.

So after a split at index `p` while processing hyperedge `j`, the circuit state
before `min(j, p)` is still valid: only hyperedges from that index onward need to
be rebuilt. The loop now does exactly that:

```cpp
std::size_t split_index = j;                       // failing hyperedge
tpc_internal::split_for_bound(dist, e, &split_index);   // splits + reports p
const std::size_t resume = std::min(j, split_index);
if (resume < j) { /* roll back to a checkpoint, replay up to `resume` */ }
start = resume;
```

Supporting pieces:

- `split_for_bound(dist, e, std::size_t* out_index)` reports
  `hyp_circ.index_in_list(chosen)` **before** splitting — the same index
  `split_hyperedge` uses, so "earlier hyperedges are untouched" is guaranteed
  even if two hyperedges compare equal.
- `BuiltCircuitSnapshot` stores only `qubits` (first-seen order) and `commands`
  (add order). `BuiltCircuit::restore()` reinstates them and calls
  `rebuild_dag()`; the DAG, `qubit_ids` and wire/frontier bookkeeping are pure
  functions of that state, so nothing derived is stored.
- `kResumeStride = 8`: a snapshot is taken every 8 hyperedges, bounding rollback
  memory to `K/8` copies of the command list while adding at most a few replayed
  hyperedges per split. (Stride 1 and 4 were measured and are not meaningfully
  faster; 16 is slightly slower.)
- The common case `p >= j` needs no rollback at all: the failing call threw
  before assigning its result, so the caller already holds the state before `j`.

The **sequence of splits is unchanged**, hence the final hypergraph, placement
and circuit are bit-identical to the reference. Each pass either completes or
strictly grows the hypergraph (a split adds one hyperedge), so the loop still
terminates.

## Change 2 — one Steiner tree per hyperedge

`LinkManager::start_link` recomputed
`steiner_tree(server_graph, servers_of(hyperedge))` on **every** distributed CU1
gate, even though the tree depends only on the hyperedge. It is now computed
lazily once per `LinkManager` (`hyperedge_tree()`) and reused — a pure cache, so
the tree is identical. On the 20-qubit case the pre-change run issued 56 414
`steiner_tree` calls; the tree is now computed at most once per hyperedge pass.

## Change 3 — skip size-1 hyperedges

`to_pytket_circuit_one_hyperedge` returns the circuit unchanged when the
hyperedge contains only its qubit vertex. The C++ builder implemented that as
`return circ;`, i.e. a **deep copy of the entire `BuiltCircuit`** (commands,
vertices, wires) for every such hyperedge. The caller now skips them.

## Measured runtime

Configuration: QFT on `P = 4` fully connected QPUs (`server_ebit_mem = 2`),
`HypergraphPartitioning(seed=56426435)` followed by `EagerHTypeMerge`, then
`to_pytket_circuit(satisfy_bound=True, allow_update=True)`.
`PYTHONHASHSEED=12345`.

**Core only** — a standalone driver linked against the header-only library
(`-O2`), consuming *the same* distribution before and after the change, so the
numbers isolate the algorithm:

| circuit | source cmds | hyperedges (initial → after splits) | before | after | speed-up | emitted circuit |
| --- | --- | --- | --- | --- | --- | --- |
| QFT 16 | 136 | 30 → 167 | 10.0 s | **2.3 s** | 4.3× | 352 cmds, 21 qubits, ebit 108 |
| QFT 20 | 210 | 38 → 295 | 79.2 s | **17.0 s** | 4.7× | 584 cmds, 24 qubits, ebit 187 |
| QFT 24 | 300 | 46 → 411 | 304.4 s | **41.4 s** | 7.3× | 818 cmds, 29 qubits, ebit 259 |

**End to end through the Python adapter** (`pytket_dqc_cpp.Distribution.to_pytket_circuit`,
including the Python↔C++ conversion and the pytket circuit rebuild):

| circuit | before | after |
| --- | --- | --- |
| QFT 16 | 11.2 s | **4.1 s** |
| QFT 20 | 69.5 s | **17.6 s** |
| QFT 24 | not measured end to end (core alone: 304.4 s) | **38.7 s** |

The speed-up grows with size because the removed work scales as the number of
restarts × suffix length; both grow with the circuit.

## Equivalence with the reference

The split path was compared directly against the unmodified `pytket_dqc`
package: one distribution object is handed to both implementations, and both
call `to_pytket_circuit(satisfy_bound=True, allow_update=True)`.

| QFT | hyperedges after splits | commands | e-bit cost | result |
| --- | --- | --- | --- | --- |
| 8 | 25 | 60 / 60 | 12 / 12 | exact circuit + identical hyperedge set |
| 12 | 91 | 190 / 190 | 56 / 56 | exact circuit + identical hyperedge set |
| 16 | 167 | 352 / 352 | 108 / 108 | exact circuit + identical hyperedge set |
| 20 | 295 | 584 / 584 | 187 / 187 | exact circuit + identical hyperedge set |

"Exact circuit" means identical qubit list and identical command sequence (op
name, op type, qubits, parameters), including the `server_X_link_register[i]`
labels. "Identical hyperedge set" means the resume logic performed exactly the
reference's sequence of splits.

Regression suites also pass unchanged: `ctest` (89/89), `pytest -m "not
high_compute"` (152 passed, 7 skipped) and `tests/python/check_equivalence.py`
(`ALL CHECKS PASSED`, which includes the exact-circuit `to_pytket_circuit`
cases and `qft_distribution_pipelines`).

## What is still slow (next steps)

The remaining cost is the per-hyperedge full rebuild, which the resume change
does not address:

- `get_commands_order()` is `O(#slices × #wires)`: it re-allocates and zeroes
  scanner arrays for every slice. A barrier is inserted after every CU1 and
  `push_barrier_over_all()` spans *all* qubits (link registers included), so
  there is roughly one slice per CU1 and the wire count grows with the number of
  link qubits.
- `rebuild_dag()` re-creates every vertex and two wires per (command, qubit),
  with `qubit_index()` doing a linear `std::string` scan per lookup.
- `BuiltCommand` copies carry `std::string` and two `std::vector`s; each
  hyperedge re-emits the whole command list.

Removing those (epoch-stamped scanners, integer qubit ids, barrier
version-counters, an interned command arena) is the next step to make the cost
`O(K · C)` instead of `O(K · C · Q)` with a much smaller constant.

## Note for developers

`python/pytket_dqc_cpp/_core.py` prefers the *installed* extension
(`.venv/lib/python3.11/site-packages/pytket_dqc_cpp/_core*.so`) over
`build/dqc_cpp`. After `cmake --build build`, copy the new `.so` into
site-packages (or run `uv sync --reinstall-package pytket-dqc-cpp`), otherwise
benchmarks and tests silently exercise the previous build.
