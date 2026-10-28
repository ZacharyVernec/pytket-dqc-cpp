# Implementation Plan: `pytket_dqc_cpp` Drop-in Adapter

Scope: Python API adapter (completed) and approved targeted C++ fixes for RNG semantics and exception bindings (see `REMAINING_FIXES.md` and `docs/fixes/`).

## Context for the implementing agent

- Workspace: `/home/ubuntu/pytket-dqc-cpp-packaged`
- Current Python side: only `python/dqc_cpp/__init__.py`, which re-exports the compiled `_core`. Leave it unchanged.
- Compiled core: `build/dqc_cpp/_core*.so` (built with `-DDQC_BUILD_BINDINGS=ON`). It contains the ported algorithms listed in `README.md`.
- Reference package source: `/home/ubuntu/pytket-dqc-cpp/.venv/lib/python3.11/site-packages/pytket_dqc`
- Reference commit: `1ac175ed820ac02366b2ed71b653d5b96fc72ff7` (matches `direct_url.json` in the installed `pytket_dqc-0.0.1.dist-info`).
- Reference test suite: fetch from the same commit (`https://github.com/Quantinuum/pytket-dqc`).
- Test venv: `/home/ubuntu/pytket-dqc-cpp/.venv` (pytket 2.16.0, networkx 3.6.1, KahyPar, PyZX, pytest).
- New package goes in `python/pytket_dqc_cpp/`.

## How to use this plan

- Work top-to-bottom through phases; respect dependency notes.
- Mark each checkbox when its acceptance criterion is met.
- Do not edit `cpp/` or `CMakeLists.txt` (note: relaxed only for approved C++ fixes recorded in `docs/fixes/`).
- Do not add drawing methods.
- Keep the reference package as a runtime dependency, but import it lazily only in delegated paths.
- Run the phase gate commands before moving to the next phase.
- If a fidelity risk fires, use the fallback documented in "Risks and fallbacks".

## Goal

Provide a `pytket_dqc_cpp` package that mirrors the public Python API of `pytket_dqc`: same module tree, class names, signatures, defaults, attributes, serialization formats and error behaviour. Ported compute must delegate to `_core`. Thin Python glue may be adapted from the Apache-2.0 reference with attribution. Users must be able to run `import pytket_dqc_cpp as pytket_dqc` and use `pytket_dqc.HypergraphCircuit`, and access submodules via the mirrored package paths (`pytket_dqc_cpp.allocators`, etc.).

## Non-negotiable decisions

- Adapt reference Python glue with attribution; do not clean-room rewrite it.
- Keep `pytket-dqc` as a runtime dependency.
- Out-of-core features call the original `pytket_dqc` code and emit one `UserWarning` per feature: `Routing`, `RandomHypergraphCircuit`, `CyclicHypergraphCircuit`, `RegularGraphHypergraphCircuit`, `check_equivalence`.
- Mirror all subpackages under `pytket_dqc_cpp`; no global `pytket_dqc` alias. Eagerly import subpackages from the top-level `__init__` so `import pytket_dqc_cpp as pytket_dqc; pytket_dqc.allocators.Random` works.
- Drawing stays out of scope.
- `VertexCover` (both `networkx` and `all_brute_force`) and `get_min_covers` are Python ports.
- `GainManager` is a Python port; its `steiner_tree` and `Distribution.hyperedge_cost` calls go through `_core`.
- `Annealing.allocate` runs the initial placement allocator in Python and then calls `_core.annealing_from_placement_with_state`; `Random`, `Ordered`, `Brute`, `HypergraphPartitioning` use `_core`.
- Every random-consuming boundary mirrors the reference's use of Python's ambient `random`: state in via `rng_state`, state out via `*_with_state`, then `random.setstate` (`Random`, `Annealing`, `BoundaryReallocation`, `DetachedGates`). KaHyPar is excluded (own RNG).
- Other refiners use `_core` refiner functions and immediately sync Python state.
- Vendored upstream tests are the interface acceptance gate, with `high_compute` excluded by default.

## Phase 0 — Preflight

- [X] Confirm `import _core` works with `build/dqc_cpp` on `sys.path`; confirm `hasattr(_core, "hypergraph_partitioning_allocate")` for KaHyPar.
- [X] Build bindings: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DDQC_BUILD_BINDINGS=ON` then `cmake --build build -j`.
- [X] Run baseline checks: `.venv/bin/python tests/python/check_equivalence.py` and `ctest --test-dir build --output-on-failure`.
- [X] Record runtime versions (`pytket`, `networkx`, `numpy`, `scipy`, `kahypar`, `pyzx`).
- [X] Create a working branch and keep `python/dqc_cpp/` unchanged.

Gate 0: baseline core differential and C++ tests pass before any Python changes.

### Phase 0 record (2026-10-26)

Interpreter: `/home/ubuntu/pytket-dqc-cpp/.venv/bin/python` (CPython 3.11.15).

| Component                  | Version                                   |
| -------------------------- | ----------------------------------------- |
| Python                     | 3.11.15                                   |
| pytket                     | 2.16.0                                    |
| networkx                   | 3.6.1                                     |
| numpy                      | 2.4.6                                     |
| scipy                      | 1.17.1                                    |
| kahypar (reference module) | dev                                       |
| pyzx                       | 0.10.5                                    |
| KaHyPar C facade           | enabled (source `/home/ubuntu/kahypar`) |
| Boost                      | 1.83.0 (`program_options`)              |
| Compiler                   | GCC 13.3.0                                |

Results:

- `_core` imports from `build/dqc_cpp/_core.cpython-311-x86_64-linux-gnu.so`; `hypergraph_partitioning_allocate` and `hypergraph_partitioning_initial_placement` both present (39 public names).
- `tests/python/check_equivalence.py`: `ALL CHECKS PASSED` for every section, and hash-seed invariance verified across `PYTHONHASHSEED` in `['0', '1', '42', '12345']`.
- `ctest --test-dir build --output-on-failure`: 87/87 tests passed (0.61 s).
- Build branch: `feature/pytket-dqc-cpp-adapter` (branched from `main`). `python/dqc_cpp/` untouched.

Note for Phase 1.2 (legacy loader quirk): the compiled module produced by CMake is a
**top-level** `_core` at `build/dqc_cpp/_core*.so`, so putting `build/dqc_cpp` on `sys.path`
makes `import _core` work but does **not** make `dqc_cpp._core` importable. The existing
`python/dqc_cpp/__init__.py` does `from dqc_cpp._core import *`, so it only imports if the
`.so` is copied into `python/dqc_cpp/`; run from the build directory alone it fails with
`ModuleNotFoundError: No module named 'dqc_cpp._core'`. That file stays unchanged per the
plan, but `pytket_dqc_cpp/_core.py` must therefore import the **top-level** `_core` in its
build-dir fallback (not `pytket_dqc_cpp._core`), and its `from . import _core` fast path
only succeeds when the extension is later installed inside the package.

## Phase 1 — Package scaffold and core bridge

### 1.1 Package skeleton

- [X] Create `python/pytket_dqc_cpp/` with subpackages: `circuits`, `placement`, `networks`, `packing`, `allocators`, `refiners`, `distributors`, `utils`.
- [X] Create empty `__init__.py` files in every directory (filled in Phase 4).
- [X] Add Apache-2.0 attribution headers to every adapted file.

### 1.2 `_core.py` loader

- [X] Try `from . import _core` first (installed-extension layout).
- [X] Fall back to adding `<repo>/build/dqc_cpp` to `sys.path` and importing top-level `_core`
  (`import _core`, **not** `import pytket_dqc_cpp._core` — see the Phase 0 note on the legacy
  `dqc_cpp` loader).
- [X] Raise a clear `ImportError` with the build command if the extension is missing.
- [X] Export a single module object for `_bridge.py` to import.

### 1.3 `_bridge.py` conversions

- [X] `serialize_circuit(circuit)` -> `(qubits, commands)` where each command is `(op.get_name(), int(op.type), [str(q) for q in cmd.qubits], [float(p) for p in op.params])`.
- [X] Reverse `OpType` map and helpers to rebuild `pytket.circuit.Op` and `pytket.circuit.Command` from core `(op_type, params, qubit_strings)`.
- [X] `core_hypergraph_circuit(hc)`: build `_core.HypergraphCircuit` from `hc._circuit` via `from_circuit`, then `reset_hyperedges()` and add every current Python hyperedge `(vertices, weight)`.
- [X] `core_placement(placement)`: create `_core.Placement` and `set(v, s)` in Python dict insertion order.
- [X] `core_network(network)`: `_core.NISQNetwork` with coupling, insertion-ordered `server_qubits`, and `server_ebit_mem`.
- [X] `core_graph(nx_graph)`: `_core.Graph` preserving node insertion order, edge insertion order and `weight` attributes.
- [X] `core_distribution(dist)`: `_core.Distribution(core_hc, core_pl, core_graph, server_qubits, server_ebit_mem)`.
- [X] `sync_from_core(dist, core_dist)`: clear and re-add Python hyperedges; replace `dist.placement.placement` with a plain dict copy.
- [X] `translate_core_error(exc)`: convert `RuntimeError` for communication-memory violations into `pytket_dqc_cpp.utils.circuit_analysis.ConstraintException` (note: superseded in Fix 4 by direct pybind11 exception translation; helper removed).
- [X] Do not rely on `_core.HypergraphCircuit.get_gate_of_vertex` (return type is not bound); use the Python `_vertex_circuit_map`.

### 1.4 `_delegation.py`

- [X] `warn_once(feature, message)`: `warnings.warn(..., UserWarning, stacklevel=...)`, once per feature per process.
- [X] `reference_module(name)`: lazy import with a clear `ImportError` if `pytket_dqc` is absent.
- [X] `to_reference(obj)` / `from_reference(obj)`: convert via `to_dict()` / `from_dict()` for `Distribution`, `HypergraphCircuit`, `Placement`, `NISQNetwork`.
- [X] No module-scope `import pytket_dqc`.

Gate 1: a scratch script builds a core `HypergraphCircuit`, `Placement`, `NISQNetwork` and `Distribution` from Python objects and reads back a cost successfully.

### Phase 1 record (2026-10-26)

Files added (all under `python/pytket_dqc_cpp/`, plus the gate script):

- `_core.py`: resolves `core` from (1) the installed `pytket_dqc_cpp._core` extension and
  (2) `<repo>/build/dqc_cpp` on `sys.path` + top-level `import _core`. Detects the
  self-reference case (this file *is* `pytket_dqc_cpp._core` in layout 2), mirrors the
  extension's public names onto itself, and raises an `ImportError` quoting the CMake command.
- `_bridge.py`: `serialize_circuit`, `OP_TYPE_BY_CODE` / `op_type_from_code` / `op_from_core` /
  `command_from_core` / `commands_from_core` / `qubit_lookup` / `qubit_from_string`,
  `core_hypergraph_circuit`, `core_placement`, `core_network`, `core_graph`,
  `nx_graph_from_core` (alias `to_nx_graph`), `core_distribution`, `sync_from_core`,
  `translate_core_error`.
- `_delegation.py`: `warn_once` / `reset_warnings`, `reference_module`, `to_reference`,
  `from_reference` (dict round trip, lazily imported on both sides).
- `tests/python/check_bridge.py`: Gate 1 script (`check_*.py`, never collected by pytest).

Notes:

- `_bridge.py` resolves the extension as `getattr(pytket_dqc_cpp._core, "core", pytket_dqc_cpp._core)`: in the installed layout the shared object **shadows** `_core.py`, so
  the bridge must accept the extension module itself (verified in a temp tree with the `.so`
  copied next to the package).
- pytket 2.16's `Op.get_name()` embeds parameters (`"CU1(0.5)"`); only the integer `op.type` and
  the float params are semantically meaningful across the boundary (matches `check_equivalence.py`).
- `str.startswith` on the three known core messages drives `translate_core_error`; the third
  message ("could not be amended") has no server in its text, so `.server` is `None` there.
- Gate 1 results: `ALL BRIDGE CHECKS PASSED` (hypergraph/placement/network/graph conversions,
  `distribution_cost` equal to the reference for a plain and an H-embedding circuit, subcircuit
  `Command` rebuild, refiner-then-`sync_from_core` with matching costs, error translation,
  lazy delegation). Baseline re-checked: `check_equivalence.py` (all sections + hash-seed sweep)
  and `ctest` 87/87 still pass; no changes under `cpp/` or `CMakeLists.txt`.

## Phase 2 — Data and utility layer

### 2.1 `circuits/hypergraph.py`

- [X] `Vertex = int`.
- [X] `Hyperedge` NamedTuple with `vertices` and `weight=1`; `__hash__ = hash((frozenset(vertices), weight))`; `to_dict`/`from_dict`.
- [X] `Hypergraph` with `vertex_list`, `hyperedge_list`, `hyperedge_dict`, `vertex_neighbours`.
- [X] `add_vertex`, `add_vertices`, `add_hyperedge` (including `hyperedge_list_index` and `hyperedge_dict_index`).
- [X] `remove_hyperedge`, `merge_hyperedge` with upstream rollback, `split_hyperedge`.
- [X] `is_valid`, `is_placement`, `get_boundary`, `weight_one_predicate`, `kahypar_hyperedges`, `to_dict`/`from_dict`, `__eq__`, `__str__`.
- [X] Omit `draw`.

### 2.2 `placement/placement.py`

- [X] Exact upstream `Placement`: `placement` dict, `__eq__`, `__str__`, `to_dict`, `from_dict`, `is_valid`, `get_distribution_tree`, `get_vertices_in`.
- [X] `get_distribution_tree` uses `utils.steiner_tree` and `utils.direct_from_origin`.

### 2.3 `networks/`

- [X] `server_network.py`: `ServerNetwork` with pair validation, connectivity check, `__eq__`, `is_placement`, `get_server_list` (use `list(set(...))`), `get_server_nx`; omit drawing.
- [X] `nisq_network.py`: `NISQNetwork` with exact validation/messages, `server_coupling`, `server_qubits`, `server_ebit_mem`, `__eq__`, `to_dict`/`from_dict`, `can_implement`, `get_qubit_list`, `get_architecture`, `get_placer`, `get_nisq_nx`; omit drawing; add `AllToAll`.
- [X] `random_networks.py`: copy `RandomNISQNetwork`, `ScaleFreeNISQNetwork`, `SmallWorldNISQNetwork` exactly, including seeds and RNG calls.
- [X] Preserve `server_qubits` insertion order.

### 2.4 `utils/`

- [X] `gateset.py`: `dqc_1_qubit`, `dqc_2_qubit`, `dqc_gateset`, `check_function`, `dqc_gateset_predicate`, `cz_to_cu1`, `tk2_to_cu1`, `tk1_to_euler`, `DQCPass`, `start_proc`, `end_proc`, `telep_proc`, `is_start_proc`, `is_end_proc`, `is_telep_proc`, `origin_of_start_proc`.
- [X] `gateset.to_euler_with_two_hadamards` calls `_core.to_euler_with_two_hadamards` and rebuilds `Op`s.
- [X] `op_analysis.py`: copy the four helpers and two op-type lists.
- [X] `graph_tools.py`: `steiner_tree` and `direct_from_origin` convert networkx <-> `_core.Graph` and call the core. Preserve node/edge order and weights.
- [X] `circuit_analysis.py`: `ConstraintException` (with `server`, `v_gate`), `all_cu1_local`, `ebit_cost`, `ebit_memory_required`, `is_link_qubit`, `get_server_id` (delegate to `_core` with serialized circuits/strings).
- [X] `qasm.py`: copy `to_qasm_str`.
- [X] `verification.py`: lazy `check_equivalence` (and `to_pyzx` if exposed) delegating to `pytket_dqc.utils.verification`; warn once.
- [X] `utils/__init__.py`: exact upstream exports.

### 2.5 `circuits/hypergraph_circuit.py`

- [X] Exact upstream `__init__(circuit)` with `reset`, `_circuit`, `_vertex_circuit_map`, `_commands`, `from_circuit`, `get_circuit`, `to_dict`/`from_dict`, `__eq__`, `__str__`.
- [X] `add_qubit_vertex`, `add_gate_vertex`, `get_qubit_vertices`, `get_qubit_vertex`, `get_vertex_of_qubit`, `get_gate_vertices`, `get_qubit_of_vertex`, `get_gate_of_vertex`, `get_hyperedges_containing`.
- [X] `add_hyperedge` validation: first vertex is a qubit, exactly one qubit, strictly increasing indices; preserve exact error messages.
- [X] `merge_hyperedge`, `split_hyperedge`, `place`, `_vertex_id_predicate`, `_sorted_hedges_predicate`, `_get_server_to_qubit_vertex`.
- [X] `get_vertex_to_command_index_map`, `get_last_gate_vertex`, `get_first_gate_vertex`, `get_intermediate_commands`, `is_h_embeddable_CU1`.
- [X] `get_hyperedge_subcircuit`: call core and rebuild `Command`s.
- [X] `requires_h_embedded_cu1`: call core.
- [X] `get_h_embedded_gate_vertices` and `get_all_h_embedded_gate_vertices`: upstream Python algorithm on top of the above.
- [X] Gate-set check at construction; raise the upstream message when not in `DQC` gateset.
- [X] `RandomHypergraphCircuit`, `CyclicHypergraphCircuit`, `RegularGraphHypergraphCircuit`: delegate to reference, warn once, wrap `ref.get_circuit()`.

### 2.6 `circuits/distribution.py`

- [X] Exact upstream state: `circuit`, `placement`, `network`; `__eq__`, `to_dict`/`from_dict`.
- [X] `is_valid`, `non_local_gate_list/count`, `detached_gate_list/count`, `get_qubit_mapping` in Python.
- [X] `cost()` validates then calls `_core.distribution_cost`.
- [X] `hyperedge_cost(hyperedge, **kwargs)`: validate weight; use `server_tree` if given, else `utils.steiner_tree`; use `requires_h_embedded_cu1` if given, else ask the circuit; if no H-embedding return `len(tree.edges)`; else run the upstream ALAP loop using `circuit.get_hyperedge_subcircuit` and `nx.shortest_path`.
- [X] `to_pytket_circuit(satisfy_bound=True, allow_update=False)`: validate; build core distribution; call binding; translate `ConstraintException`; sync `circuit` and `placement` when `allow_update=True`; preserve the cost-mismatch warning.

Gate 2: a scratch script reproduces upstream `HypergraphCircuit` hyperedges/vertices and `Distribution.cost()` for several fixed circuits.

### Phase 2 record (2026-10-26)

Files added under `python/pytket_dqc_cpp/` (all with the Apache-2.0 header plus an
"Adapted from `pytket_dqc` (commit 1ac175e)" note):

- `circuits/{hypergraph,hypergraph_circuit,distribution}.py`, `circuits/__init__.py`
- `placement/placement.py`, `placement/__init__.py`
- `networks/{server_network,nisq_network,random_networks}.py`, `networks/__init__.py`
- `utils/{gateset,op_analysis,graph_tools,circuit_analysis,qasm,verification}.py`,
  `utils/__init__.py`
- `tests/python/check_phase2.py`: Gate 2 differential script (`check_*.py`, never collected).

Core-backed paths:

- `HypergraphCircuit.get_hyperedge_subcircuit` / `requires_h_embedded_cu1` -> `_core`
  (results rebuilt into `Command`s with the circuit's own `Qubit` objects).
- `Distribution.cost` -> `_core.distribution_cost`; `to_pytket_circuit` ->
  `_core.Distribution.to_pytket_circuit` (the whole ALAP/embedding/bound machinery).
- `utils`: `to_euler_with_two_hadamards`, `steiner_tree`, `direct_from_origin`,
  `all_cu1_local`, `ebit_cost`, `ebit_memory_required`, `is_link_qubit`, `get_server_id`.
- Delegated to the reference with one `UserWarning` per process:
  `RandomHypergraphCircuit`, `CyclicHypergraphCircuit`, `RegularGraphHypergraphCircuit`,
  `check_equivalence` (and `to_pyzx`).

Notes and deviations:

- Drawing is absent: `Hypergraph.draw`, `ServerNetwork.draw_server_network` and
  `NISQNetwork.draw_nisq_network` are removed together with the `hypernetx`/`matplotlib`
  imports. `nisq_network` imports `HypergraphCircuit` only under `TYPE_CHECKING`.
- `graph_tools.steiner_tree` replays the graph into the core in an order that reproduces
  networkx's per-node neighbour order (the core's weighted tie-breaking depends on it) and
  restores the original edge data on the returned tree, so `edges(data=True)` matches the
  reference's subgraph view. Errors are normalised to upstream's types/messages
  (`Exception`, `nx.NodeNotFound`, `nx.NetworkXError`).
- `gateset.to_euler_with_two_hadamards` pre-rejects the two 3-op shapes the core aborts on
  (`[H,Rz,Rz]`, `[Rz,Rz,H]`) by raising upstream's `IndexError`; other malformed shapes
  raise the core's `RuntimeError` where upstream raises `AssertionError` (same messages).
- `_bridge.sync_from_core` keeps the previous `placement` key order (the core stores it in a
  sorted map), so dicts after `allow_update=True` match the reference exactly.
- `Distribution.to_pytket_circuit` necessarily imports `pytket_dqc.utils.gateset` inside the
  C++ binding, which rebuilds the `start_proc`/`end_proc` custom gates.
- Known core divergence (superseded in Fix 4): during the initial adapter phase (no C++ changes),
  the communication-memory `RuntimeError` message omitted upstream's trailing `to_pytket_circuit(allow_update=True)`
  hint and carried no `v_gate`. In follow-on Fix 4, C++ pybind11 translation was implemented, attaching
  `.v_gate` and the guidance hint directly.
- Gate 2 results: `ALL PHASE 2 CHECKS PASSED` (268 checks, repeated under
  `PYTHONHASHSEED` 0/1/42/12345) and a 150-case random fuzz over circuits x networks x
  placements with 0 mismatches (hyperedges, subcircuits, costs, gate lists, qubit maps,
  generated circuits). Baselines re-checked: `check_bridge.py`, `check_equivalence.py`
  (all sections + hash-seed sweep) and `ctest` 87/87 still pass; nothing under `cpp/` or
  `CMakeLists.txt` changed.

## Phase 3 — Packing, GainManager and allocators

### 3.1 `packing/pacman.py`

- [X] `Packet` NamedTuple fields in order: `packet_index`, `qubit_vertex`, `connected_server_index`, `gate_vertices`, `parent_hedge`.
- [X] `Packet.__str__`/`__repr__` = `P<i>`; `__hash__ = hash(repr(self))`.
- [X] `MergedPacket`, `NeighbouringPacket`, `HoppingPacket` aliases.
- [X] `PacMan.__init__` builds `_core.PacMan` and fills `packets_by_qubit`, `neighbouring_packets`, `hopping_packets`, `merged_packets`.
- [X] Reconstruct `Packet` objects with `parent_hedge`: the unique hyperedge in `hyperedge_dict[qubit_vertex]` containing all packet gate vertices; assert uniqueness, fall back to first containing hyperedge.
- [X] Delegate to core: `build_packets`, `identify_neighbouring_packets`, `identify_hopping_packets`, `merge_all_packets`, `are_neighbouring_packets`, `are_hoppable_packets`.
- [X] Copy upstream Python helpers: `hyperedge_to_packets`, `get_subsequent_*`, `get_next_packet`, `get_all_packets`, `get_all_merged_packets`, `get_connected_server`, `get_embedded_packets`, `get_all_embedded_packets`, `get_hopping_packet_from_embedded_packet`, `get_hopping_packets_within`, `is_packet_embedded`, `get_containing_merged_packet`, `get_connected_merged_packets`, `get_split_packets`, `get_intermediate_commands`, `are_1q_op_phases_npi`, `get_nx_graph_merged`, `get_nx_graph_neighbouring`, `get_nx_graph_conflict`, `assign_bipartitions`, `is_bipartite_predicate`, `get_mvc_*`, `get_conflict_edges_given_mvc`, `get_hypergraph_from_packets`.
- [X] `get_hypergraph_from_packets` returns a Python `HypergraphCircuit` built from the original circuit with the packet-derived hyperedges.

### 3.2 `allocators/gain_manager.py`

- [X] Python port of upstream `GainManager` with all attributes: `distribution`, `max_key_size`, `qubit_vertices`, `server_graph` (nx.Graph), `occupancy`, `hyperedge_cost_map`, `requires_h_embedded_cu1`, `steiner_cache`.
- [X] `update_cost`, `split_hyperedge_gain`, `split_hyperedge`, `merge_hyperedge_gain`, `merge_hyperedge`, `move_vertex_gain`, `move_vertex`, `is_move_valid`, `current_server`, `set_max_key_size`.
- [X] Use `utils.steiner_tree` and `Distribution.hyperedge_cost`; no direct reference imports.
- [X] Preserve upstream move restrictions after embedding.

### 3.3 Allocators

- [X] `allocator.py`: `Allocator` ABC with abstract `allocate(circ, network, **kwargs)`.
- [X] `random.py`: `Random.allocate` calls `_core.random_allocate` and adapts circuit/placement/network.
- [X] `ordered.py`: `order_reducing_size` copy; `Ordered.allocate` calls `_core.ordered_allocate`.
- [X] `brute.py`: `Brute.allocate` calls `_core.brute_allocate`.
- [X] `annealing.py`: `Annealing.allocate` runs the initial placement allocator in Python (`iterations`, `initial_place_method`, `seed`, `cache_limit`, `initial_temperature`), then calls `_core.annealing_from_placement_with_state` with the ambient Python RNG state (`random.getstate()`), installing the returned state with `random.setstate`. The earlier Python loop and `acceptance_criterion` copy were removed: the binding is unconditional, so the loop was unreachable dead code and the helper is internal to the core.
- [X] `hypergraph_partitioning.py`: `allocate` calls `_core.hypergraph_partitioning_allocate`; `initial_distribute` calls `_core.hypergraph_partitioning_initial_placement`; `make_valid` is the Python `GainManager` loop; raise a clear error if `_core` lacks KaHyPar.
- [X] `routing.py`: delegate to reference `Routing` with warning; convert network and result via dicts.
- [X] Copy `km1_kKaHyPar_sea20.ini` into `python/pytket_dqc_cpp/allocators/`.
- [X] `allocators/__init__.py`: exact upstream exports.

Gate 3: `tests/pacman_test.py`, `tests/gain_manager_test.py` and `tests/allocator_test.py` pass.

### Phase 3 record (2026-10-26)

Files added under `python/pytket_dqc_cpp/`:

- `packing/pacman.py`, `packing/__init__.py`.
- `allocators/{allocator,gain_manager,random,ordered,brute,annealing,hypergraph_partitioning,routing,__init__}.py`,
  `allocators/_core_allocation.py` (core `Distribution` -> Python `Distribution`) and
  `allocators/km1_kKaHyPar_sea20.ini` (copied from the reference package).

Deviations and notes:

- Phase-3 modules import phase-2 objects from their concrete modules
  (`..circuits.hypergraph`, `..circuits.hypergraph_circuit`, `..circuits.distribution`,
  `..placement.placement`) so they do not depend on the Phase 4 package re-exports.
- `PacMan` builds `_core.PacMan` once in `__init__`; `build_packets` and the identification
  passes re-read the core dictionaries, recovering `parent_hedge` from
  `hyperedge_dict[qubit_vertex]` (first qualifying hyperedge if several match).
- `HypergraphPartitioning` always passes an explicit INI path (the packaged one by default) and
  honours a caller-provided `ini_path`; the reference silently replaces it with its default.
- `Random` is core-backed, so it does not advance Python's global `random` state the way the
  reference does. Consequence: `Annealing(seed=...)` with the default `Random` initializer has a
  different seeded trajectory from the reference; with an explicit `initial_place_method`
  (e.g. `Ordered`) it matches exactly. All reference tests pass `Ordered`.

Validation (against real phase-2 modules):

- Adapted upstream `tests/{pacman,gain_manager,allocator}_test.py` (imports rewritten to
  `pytket_dqc_cpp`; run from a scratch copy with the package on `PYTHONPATH`):
  **32 passed, 2 skipped** (`high_compute`; `test_nisq_draw` not in this set), including the
  delegated `Routing` warning test.
- Additional differential run vs `pytket_dqc` for PacMan dictionaries/graphs,
  `GainManager` occupancy/costs/gains, and `Random`/`Ordered`/`Brute`/`Annealing`/HP
  placements: all equal.
- Two phase-2 integration bugs were fixed during gate 3 (`HypergraphCircuit` called
  `core.requires_h_embedded_cu1` / `core.get_hyperedge_subcircuit`, which live on
  `_core.HypergraphCircuit`; both now use `core_hypergraph_circuit(self)`).

## Phase 4 — Refiners, distributors, package exports

### 4.1 Refiners

- [X] `refiner.py`: `Refiner` ABC with `refine(distribution, **kwargs) -> bool`.
- [X] `repeat_refiner.py`, `sequence_refiner.py`: upstream copies.
- [X] `h_type_merge.py`, `neighbouring_d_type_merge.py`, `intertwined_d_type_merge.py`: build core distribution, call `_core.*_refine`, sync, return bool.
- [X] `boundary_reallocation.py`: map `fixed_vertices`, `num_rounds`, `stop_parameter`, `seed`, `cache_limit` to the core call; sync; return bool.
- [X] `detached_gates.py`: compute `fixed_vertices` in Python and call `BoundaryReallocation` (the core `detached_gates_refine` entry point segfaults on hyperedges with no gate vertices; see Phase 4 record); sync; return bool.
- [X] `vertex_cover.py`: full upstream Python port including `refine`, `exhaustive_refine`, `networkx_refine`, `get_min_covers`; no reference delegation.
- [X] `refiners/__init__.py`: exact upstream exports.

### 4.2 Distributors

- [X] `distributor.py`: `Distributor` ABC.
- [X] `cover_embedding.py`: `CoverEmbedding`, `CoverEmbeddingSteiner`, `CoverEmbeddingSteinerDetached`.
- [X] `partitioning_heterogeneous.py`: `PartitioningHeterogeneous`, `PartitioningHeterogeneousEmbedding`, `PartitioningAnnealing`.
- [X] `distributors/__init__.py`: exact upstream exports.

### 4.3 Package exports

- [X] Top-level `__init__.py`: `HypergraphCircuit`, `Distribution`, `Placement`, `NISQNetwork`, `DQCPass`; then eager-import `circuits`, `placement`, `networks`, `utils`, `packing`, `allocators`, `refiners`, `distributors`.
- [X] `circuits/__init__.py`: `HypergraphCircuit`, `RandomHypergraphCircuit`, `CyclicHypergraphCircuit`, `RegularGraphHypergraphCircuit`, `Hypergraph`, `Hyperedge`, `Vertex`, `Distribution`.
- [X] `placement/__init__.py`: `Placement`.
- [X] `networks/__init__.py`: `NISQNetwork`, `AllToAll`, `RandomNISQNetwork`, `ScaleFreeNISQNetwork`, `SmallWorldNISQNetwork`, `ServerNetwork`.
- [X] `packing/__init__.py`: `PacMan`, `Packet`, `MergedPacket`, `HoppingPacket`, `NeighbouringPacket`.
- [X] `allocators/__init__.py`: `Allocator`, `GainManager`, `Annealing`, `Brute`, `Routing`, `HypergraphPartitioning`, `Random`, `Ordered`.
- [X] `utils/__init__.py`: exact upstream exports listed in Phase 2.4.
- [X] Import order mirrors upstream to avoid circular-import failures.

Gate 4: `import pytket_dqc_cpp as pytket_dqc` succeeds; `pytket_dqc.allocators.Random` and `pytket_dqc_cpp.circuits.Hyperedge` are available; `python -c` smoke test passes.

## Phase 5 — Tests, docs and final verification

### 5.1 Vendor and adapt upstream tests

- [X] Fetch tests and fixtures from commit `1ac175e`.
- [X] Copy `tests/*_test.py` into repo `tests/` keeping original filenames.
- [X] Replace all `pytket_dqc` imports with `pytket_dqc_cpp`.
- [X] Copy `tests/test_circuits/` to repo `tests/test_circuits/` so relative fixture paths resolve from the repo root.
- [X] Skip/remove `test_nisq_draw` (drawing out of scope).
- [X] Adapt `test_kahypar_install` to locate `pytket_dqc_cpp/allocators/km1_kKaHyPar_sea20.ini`.
- [X] Keep `high_compute` markers from upstream.
- [X] Do not copy `tests/__init__.py` unless pytest import mode requires it.

### 5.2 Pytest configuration

- [X] Add `pytest.ini` with `testpaths = tests`, `python_files = *_test.py test_*.py`, and the `high_compute` marker description.
- [X] Ensure `tests/python/check_*.py` are not collected as tests.

### 5.3 Adapter differential test

- [X] Add `tests/python/check_adapter_equivalence.py`.
- [X] Compare `pytket_dqc_cpp` vs `pytket_dqc` for: `HypergraphCircuit` vertices/hyperedges, `Distribution.cost`, `Random`/`Ordered`/`Brute` placements, `Annealing` seeded placement, refiner hyperedge lists, `PacMan` packet dicts, `GainManager` cost maps, `steiner_tree` node/edge sets including edge attributes, and `to_pytket_circuit` command names/qubits.
- [X] Run under multiple `PYTHONHASHSEED` values where order-sensitive paths are compared.

### 5.4 Documentation

- [X] Update `README.md`: package/import usage, runtime reference dependency, delegated slow paths and warnings, core-backed vs Python paths, test commands, drawing exclusion.
- [X] Add a short note that separate `from pytket_dqc...` imports must change their root to `pytket_dqc_cpp`.

### 5.5 Final gates

- [X] `PYTHONPATH=python .venv/bin/python -m pytest tests -m "not high_compute"` passes.
- [X] `.venv/bin/python tests/python/check_equivalence.py` passes.
- [X] `.venv/bin/python tests/python/check_adapter_equivalence.py` passes.
- [X] `ctest --test-dir build --output-on-failure` passes.
- [X] Optional: `PYTHONPATH=python .venv/bin/python -m pytest tests -m high_compute` passes or documented skips.
- [X] Initial adapter phase completed without C++ edits; follow-on approved fixes updated C++ RNG semantics and pybind11 exception translation (see `docs/fixes/`).

## Next plans (after this adapter plan)

This plan (the adapter) is now complete. The follow-on work status:

1. **`REMAINING_FIXES.md`** — **Implemented** in accordance with user decisions documented in `docs/fixes/`:
   - Fix 3 (RNG semantics: `Annealing` default-initializer warning, non-deterministic OS entropy seeding via `PyRandom(std::nullopt)` with zero Python global RNG mutation, documented in `docs/fixes/01_*.md` and `02_*.md`).
   - Fix 4 (Error attribute parity: C++ pybind11 translation populates `.v_gate` and remediation hint directly, documented in `docs/fixes/03_*.md`).
   - Gates (Unified gate runner `tests/python/run_gates.sh`, strict zero-skip check count enforcement in `check_adapter_equivalence.py`, hash sweep enforcement, documented in `docs/fixes/04_*.md`).
2. **Implement a build step** — package the project so it builds without the
   current manual `cmake` + `sys.path` workflow (compiled extension installed
   inside `pytket_dqc_cpp`, no `build/dqc_cpp` fallback needed, KaHyPar INI
   shipped with the package).

The "no changes under `cpp/` or `CMakeLists.txt`" constraint applied to the
initial adapter plan, and was lifted for approved fixes in `REMAINING_FIXES.md`
(C++ RNG and pybind11 exception translation) and future build step plan 2.

## API surface checklist

Top-level: `HypergraphCircuit`, `Distribution`, `Placement`, `NISQNetwork`, `DQCPass`.

`circuits`: `HypergraphCircuit`, `RandomHypergraphCircuit`, `CyclicHypergraphCircuit`, `RegularGraphHypergraphCircuit`, `Hypergraph`, `Hyperedge`, `Vertex`, `Distribution`.

`placement`: `Placement`.

`networks`: `NISQNetwork`, `AllToAll`, `RandomNISQNetwork`, `ScaleFreeNISQNetwork`, `SmallWorldNISQNetwork`, `ServerNetwork`.

`packing`: `PacMan`, `Packet`, `MergedPacket`, `HoppingPacket`, `NeighbouringPacket`.

`allocators`: `Allocator`, `GainManager`, `Annealing`, `Brute`, `Routing`, `HypergraphPartitioning`, `Random`, `Ordered`.

`refiners`: `Refiner`, `VertexCover`, `NeighbouringDTypeMerge`, `IntertwinedDTypeMerge`, `EagerHTypeMerge`, `BoundaryReallocation`, `DetachedGates`, `RepeatRefiner`, `SequenceRefiner`.

`distributors`: `Distributor`, `CoverEmbedding`, `CoverEmbeddingSteiner`, `CoverEmbeddingSteinerDetached`, `PartitioningHeterogeneous`, `PartitioningHeterogeneousEmbedding`, `PartitioningAnnealing`.

`utils`: `dqc_gateset`, `dqc_gateset_predicate`, `DQCPass`, `to_euler_with_two_hadamards`, `is_antidiagonal`, `is_diagonal`, `is_distributable`, `distributable_1q_op_types`, `distributable_op_types`, `direct_from_origin`, `steiner_tree`, `ConstraintException`, `ebit_cost`, `ebit_memory_required`, `check_equivalence`, `to_qasm_str`.

Module-level extras used by tests: `pytket_dqc_cpp.circuits.hypergraph.Vertex`, `pytket_dqc_cpp.utils.gateset.start_proc`, `end_proc`, `telep_proc`, `pytket_dqc_cpp.utils.qasm.to_qasm_str`, `pytket_dqc_cpp.allocators.ordered.order_reducing_size`, `pytket_dqc_cpp.refiners.vertex_cover.get_min_covers`.

## Fidelity invariants

- `Hyperedge.__hash__`, `Packet.__hash__` and `Packet.__repr__` must match upstream exactly; set iteration order feeds vertex-cover and refiner decisions.
- Preserve dict and list insertion order everywhere (`server_qubits`, `placement`, `hyperedge_list`, `hyperedge_dict`, `_commands`).
- Preserve upstream error messages where tests assert them.
- `to_dict`/`from_dict` formats must round-trip reference JSON fixtures.
- No module-scope `import pytket_dqc`; delegated imports are lazy.
- One `UserWarning` per delegated feature per process.
- Core refiner calls must sync Python state immediately.
- `Distribution.hyperedge_cost` must honour `server_tree` and `requires_h_embedded_cu1` kwargs.
- Do not call `_core.HypergraphCircuit.get_gate_of_vertex`; use Python bookkeeping.
- Do not add drawing APIs.

## Verification commands

- Build: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DDQC_BUILD_BINDINGS=ON` and `cmake --build build -j`.
- Import: `PYTHONPATH=python .venv/bin/python -c "import pytket_dqc_cpp as pytket_dqc; print(pytket_dqc.HypergraphCircuit, pytket_dqc.allocators.Random)"`.
- Adapted suite: `PYTHONPATH=python .venv/bin/python -m pytest tests -m "not high_compute"`.
- Core differential: `.venv/bin/python tests/python/check_equivalence.py`.
- Adapter differential: `.venv/bin/python tests/python/check_adapter_equivalence.py`.
- C++: `ctest --test-dir build --output-on-failure`.
- Optional high compute: `PYTHONPATH=python .venv/bin/python -m pytest tests -m high_compute`.

## Definition of done

- All checkboxes above are complete.
- Every public name in the API surface checklist imports successfully.
- Adapted upstream non-high-compute tests pass against `pytket_dqc_cpp`.
- Core and adapter differential checks pass.
- No unauthorized changes under `cpp/` or `CMakeLists.txt` (only approved RNG and exception fixes per `docs/fixes/`).
- README documents usage, delegated slow paths and test commands.
- Drawing remains absent.

## Risks and fallbacks

1. `PacMan.parent_hedge` reconstruction diverges.
   - First fallback: verify uniqueness of the containing hyperedge; if multiple, choose the first in `hyperedge_dict[qubit_vertex]` order.
   - Second fallback: replace the core-backed packet identification with the upstream Python `build_packets`/`hyperedge_to_packets` implementation while keeping `are_neighbouring_packets`/`are_hoppable_packets` in the core.
2. `GainManager`/`Annealing` exactness diverges.
   - Compare against reference `GainManager` step by step, then fall back to delegating the Python `GainManager` to the reference with conversion and warning.
3. `to_pytket_circuit` exception translation misses a case.
   - Handled directly in `cpp/python/bindings.cpp` via pybind11 `ConstraintException` translation; regression tested in `tests/error_parity_test.py`.
4. KaHyPar unavailable in the local `_core`.
   - Skip KaHyPar tests with a clear reason; keep `HypergraphPartitioning` raising an actionable error.
5. Reference test fixtures fail to unpickle under pytket 2.16.
   - Prefer JSON fixtures; if a pickle fails, document the skip and add an equivalent JSON-based test.
6. Reference delegation creates circular imports.
   - Keep all `pytket_dqc` imports inside delegated functions/constructors; never at module scope.
