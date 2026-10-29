# C++ Core Unit Tests (Catch2)

The native Catch2 test suite in `cpp/tests/` verifies the C++ core algorithms (`dqc_core`) independently of Python, testing graph algorithms against NetworkX 3.6.1 golden vectors, exact MT19937 RNG sequences, CPython set iteration order (`PySet`), PacMan packet packing, allocators, and refiners.

## Running Tests

From the repository root (after building):

```bash
ctest --test-dir build --output-on-failure
```

Or run the Catch2 test runner directly:

```bash
./build/dqc_tests
```

## Test Suite Map (17 Test Suites, 89 Test Cases)

- `test_rng.cpp` — `PyRandom` is bit-exact against CPython `random` (golden vectors for seeds `0, 1, 12345, 2^32, 2^32+1, 2^63+5, 2^64`), verifies non-deterministic unseeded initialization via OS entropy, and verifies exact 625-word MT19937 `get_state` / `set_state` round-tripping.
- `test_hypergraph.cpp` — hypergraph add/remove/merge/split/boundary, `is_valid`, `is_placement`.
- `test_graph.cpp` — `steiner_tree`, `direct_from_origin`, Dijkstra vs networkx 3.6.1 golden vectors; node/edge iteration order, weights, self-loops, `degree`, `remove_node`, `subgraph`, `edge_subgraph`.
- `test_gateset.cpp` — `to_euler_with_two_hadamards` (all branch cases + error cases).
- `test_pacman.cpp` — packet construction, neighbouring/hopping/merged identification, packet-navigation helpers, connectivity, embedding, split packets.
- `test_allocators.cpp` — `Random` (seeded determinism), `Ordered`, `Brute`, `Annealing` (validity + seeded-initial-placement determinism), and `can_implement` errors.
- `test_refiners.cpp` — `DetachedGates` (golden seeded result), `BoundaryReallocation` (determinism), `VertexCover` (valid refined distribution).
- `test_distribution.cpp` — `cost`, `is_valid`, `non_local_gate_list/count`, `detached_gate_list/count`.
- `test_network.cpp` — `NISQNetwork` construction, accessors, validation errors, graph structure, equality.
- `test_composition.cpp` — `RepeatRefiner` / `SequenceRefiner` do-while / or-fold semantics and nesting.
- `test_to_pytket_circuit.cpp` — circuit emission to pytket `Circuit`, gate translation, ebit memory validation, and constraint error triggers.
- `test_hypergraph_partitioning.cpp` — KaHyPar placement validity, capacity, determinism, `can_implement` error (only built when KaHyPar is available).
- `test_bipartite.cpp` — bipartite graph coloring, independent sets, Hopcroft-Karp matching, and minimum vertex cover.
- `test_pyset.cpp` — `PySet` probe sequences, dummy-slot reuse, resize policy, reproducing CPython `set` iteration order.
- `test_circuit_analysis.cpp` — command inspection, ebit cost/memory requirements, CU1 locality checks.
- `test_placement.cpp` — `Placement` mapping validity, server assignment, serialization.
- `test_hypergraph_circuit.cpp` — hypergraph circuit construction, gate indexing, boundary tracking.
