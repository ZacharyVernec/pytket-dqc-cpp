#include <catch2/catch_test_macros.hpp>

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "dqc/distribution.hpp"

using dqc::Distribution;
using dqc::Graph;
using dqc::HypergraphCircuit;
using dqc::Placement;
using dqc::SerializedCircuit;
using dqc::Vertex;

// Two CU1 gates on qubit 0, no H embedding. Mirrors
// test_distribution_cost_no_embedding from the upstream test suite.
static HypergraphCircuit make_circuit() {
  SerializedCircuit c;
  c.qubits = {"q0", "q1", "q2"};
  c.commands = {
      {"CU1", {"q0", "q1"}, dqc::kOpCU1, {1.0}},
      {"CU1", {"q0", "q2"}, dqc::kOpCU1, {1.0}},
  };
  HypergraphCircuit h;
  h.from_circuit(c);
  return h;
}

static Graph make_server_graph() {
  Graph g;
  for (int i = 0; i < 5; ++i) g.add_node(i);
  g.add_edge(0, 1);
  g.add_edge(1, 2);
  g.add_edge(1, 3);
  g.add_edge(2, 4);
  return g;
}

static Placement make_placement(const std::vector<std::pair<int, int>>& entries) {
  Placement p;
  for (const auto& [v, s] : entries) p.set(v, s);
  return p;
}

TEST_CASE("distribution_cost without embedding") {
  const HypergraphCircuit h = make_circuit();
  const Graph g = make_server_graph();

  const std::vector<std::pair<std::vector<std::pair<int, int>>, int>> cases = {
      {{{0, 0}, {1, 1}, {2, 2}, {3, 1}, {4, 2}}, 2},
      {{{0, 0}, {1, 1}, {2, 2}, {3, 1}, {4, 0}}, 3},
      {{{0, 1}, {1, 0}, {2, 2}, {3, 0}, {4, 2}}, 2},
  };

  for (const auto& [entries, expected] : cases) {
    const Placement p = make_placement(entries);
    REQUIRE(dqc::distribution_cost(h, p, g) == expected);
  }
}

// Mirrors the upstream tests for `Distribution.is_valid`,
// `non_local_gate_list/count` and `detached_gate_list/count`. The circuit is
// two CU1 gates on qubit 0 (vertices 0..2 qubits, 3..4 gates); the network
// has servers 0..4 each holding a single qubit.
static std::map<int, std::vector<int>> server_qubits_5() {
  return {{0, {0}}, {1, {1}}, {2, {2}}, {3, {3}}, {4, {4}}};
}

TEST_CASE("distribution is_valid checks placement and capacity") {
  const HypergraphCircuit h = make_circuit();
  const Graph g = make_server_graph();

  // Valid: every vertex placed on a known server within capacity.
  Distribution d{h, make_placement({{0, 0}, {1, 1}, {2, 2}, {3, 1}, {4, 2}}),
                 g, server_qubits_5()};
  REQUIRE(d.is_valid());

  // Exceeding a server's qubit capacity is invalid.
  Distribution d_over{h,
                      make_placement({{0, 0}, {1, 1}, {2, 1}, {3, 1}, {4, 1}}),
                      g, server_qubits_5()};
  REQUIRE(!d_over.is_valid());

  // Placing a vertex on an unknown server is invalid.
  Distribution d_unknown{h,
                         make_placement({{0, 0}, {1, 1}, {2, 2}, {3, 9}, {4, 2}}),
                         g, server_qubits_5()};
  REQUIRE(!d_unknown.is_valid());
}

TEST_CASE("distribution non-local and detached gate lists") {
  const HypergraphCircuit h = make_circuit();
  const Graph g = make_server_graph();

  // Both gates on a qubit's home server: non-local, not detached.
  Distribution d1{h, make_placement({{0, 0}, {1, 1}, {2, 2}, {3, 1}, {4, 2}}),
                  g, server_qubits_5()};
  REQUIRE(d1.non_local_gate_list() == std::vector<Vertex>({3, 4}));
  REQUIRE(d1.non_local_gate_count() == 2);
  REQUIRE(d1.detached_gate_list() == std::vector<Vertex>({}));
  REQUIRE(d1.detached_gate_count() == 0);

  // Gate 3 moved to a server owning neither qubit: now detached.
  Distribution d2{h, make_placement({{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 2}}),
                  g, server_qubits_5()};
  REQUIRE(d2.non_local_gate_list() == std::vector<Vertex>({3, 4}));
  REQUIRE(d2.non_local_gate_count() == 2);
  REQUIRE(d2.detached_gate_list() == std::vector<Vertex>({3}));
  REQUIRE(d2.detached_gate_count() == 1);
}
