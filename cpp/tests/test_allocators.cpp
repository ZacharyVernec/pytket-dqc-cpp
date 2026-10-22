#include <catch2/catch_test_macros.hpp>

#include <map>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#include "dqc/allocators.hpp"
#include "dqc/distribution.hpp"
#include "dqc/graph.hpp"
#include "dqc/hypergraph_circuit.hpp"
#include "dqc/op_type.hpp"
#include "dqc/placement.hpp"

using dqc::Distribution;
using dqc::Graph;
using dqc::HypergraphCircuit;
using dqc::Placement;
using dqc::SerializedCircuit;
using dqc::Vertex;

namespace {

using ServerQubits = std::vector<std::pair<int, std::vector<int>>>;

HypergraphCircuit three_qubit_circuit() {
  SerializedCircuit c;
  c.qubits = {"q[0]", "q[1]", "q[2]"};
  c.commands = {
      {"CU1", {"q[0]", "q[1]"}, dqc::kOpCU1, {1.0}},
      {"CU1", {"q[0]", "q[2]"}, dqc::kOpCU1, {1.0}},
  };
  HypergraphCircuit h;
  h.from_circuit(c);
  return h;
}

HypergraphCircuit two_qubit_circuit() {
  SerializedCircuit c;
  c.qubits = {"q[0]", "q[1]"};
  c.commands = {
      {"CU1", {"q[0]", "q[1]"}, dqc::kOpCU1, {1.0}},
  };
  HypergraphCircuit h;
  h.from_circuit(c);
  return h;
}

Graph server_graph_3() {
  Graph g;
  for (int i = 0; i < 3; ++i) g.add_node(i);
  g.add_edge(0, 1);
  g.add_edge(1, 2);
  return g;
}

Graph server_graph_2() {
  Graph g;
  g.add_node(0);
  g.add_node(1);
  g.add_edge(0, 1);
  return g;
}

int qubit_count_on(const Placement& p, const HypergraphCircuit& h, int server) {
  int count = 0;
  for (const auto& [v, s] : p.entries) {
    if (s == server && h.is_qubit_vertex(v)) ++count;
  }
  return count;
}

bool respects_capacity(const Placement& p, const HypergraphCircuit& h,
                       const ServerQubits& sq) {
  for (const auto& [server, qubits] : sq) {
    if (qubit_count_on(p, h, server) > static_cast<int>(qubits.size()))
      return false;
  }
  return true;
}

}  // namespace

TEST_CASE("random allocator is seeded-deterministic and valid") {
  const HypergraphCircuit h = three_qubit_circuit();
  const ServerQubits sq = {{0, {0, 1}}, {1, {2}}, {2, {3}}};
  const std::vector<std::pair<int, int>> coupling = {{0, 1}, {1, 2}};

  const Distribution d1 = dqc::random_allocate(h, coupling, sq, server_graph_3(),
                                               {}, 42);
  const Distribution d2 = dqc::random_allocate(h, coupling, sq, server_graph_3(),
                                               {}, 42);
  REQUIRE(d1.placement.placement == d2.placement.placement);
  REQUIRE(dqc::placement_is_valid(h, d1.placement, sq));
  REQUIRE(respects_capacity(d1.placement, h, sq));
  REQUIRE(h.is_placement(d1.placement));
  REQUIRE(d1.is_valid());
}

TEST_CASE("random allocator differs across seeds") {
  const HypergraphCircuit h = three_qubit_circuit();
  const ServerQubits sq = {{0, {0, 1}}, {1, {2}}, {2, {3}}};
  const std::vector<std::pair<int, int>> coupling = {{0, 1}, {1, 2}};

  const Distribution d1 = dqc::random_allocate(h, coupling, sq, server_graph_3(),
                                               {}, 1);
  const Distribution d2 = dqc::random_allocate(h, coupling, sq, server_graph_3(),
                                               {}, 999);
  REQUIRE(d1.placement.placement != d2.placement.placement);
}

TEST_CASE("ordered allocator fills largest servers first") {
  const HypergraphCircuit h = three_qubit_circuit();
  const ServerQubits sq = {{0, {0, 1}}, {1, {2}}, {2, {3}}};

  const Distribution d = dqc::ordered_allocate(h, sq, server_graph_3(), {});
  // Server 0 (2 slots) gets qubits 0,1; server 1 gets qubit 2; gate vertex 3
  // goes to the largest (first) server.
  REQUIRE(d.placement.at(0) == 0);
  REQUIRE(d.placement.at(1) == 0);
  REQUIRE(d.placement.at(2) == 1);
  REQUIRE(d.placement.at(3) == 0);
  REQUIRE(dqc::placement_is_valid(h, d.placement, sq));
}

TEST_CASE("brute allocator finds a minimal-cost valid placement") {
  const HypergraphCircuit h = two_qubit_circuit();
  const ServerQubits sq = {{0, {0}}, {1, {1}}};
  const std::vector<std::pair<int, int>> coupling = {{0, 1}};

  const Distribution d = dqc::brute_allocate(h, coupling, sq, server_graph_2(), {});
  REQUIRE(dqc::placement_is_valid(h, d.placement, sq));
  // With two qubits on the two ends of a single link, the CU1 must be
  // implemented on one of the two servers (cost 1).
  REQUIRE(d.cost() == 1);
}

TEST_CASE("allocators raise when the network cannot implement the circuit") {
  const HypergraphCircuit h = three_qubit_circuit();
  const ServerQubits sq = {{0, {0}}, {1, {1}}};  // capacity 2 < 3 qubits
  const std::vector<std::pair<int, int>> coupling = {{0, 1}};

  REQUIRE_THROWS_AS(dqc::random_allocate(h, coupling, sq, server_graph_2(), {}, 1),
                    std::runtime_error);
  REQUIRE_THROWS_AS(dqc::ordered_allocate(h, sq, server_graph_2(), {}),
                    std::runtime_error);
}

TEST_CASE("annealing allocator returns a valid distribution") {
  const HypergraphCircuit h = three_qubit_circuit();
  const ServerQubits sq = {{0, {0, 1}}, {1, {2}}, {2, {3}}};
  const std::vector<std::pair<int, int>> coupling = {{0, 1}, {1, 2}};

  const Distribution d = dqc::annealing_allocate(h, coupling, sq, server_graph_3(),
                                                 {}, 300, 3.0, std::nullopt, 42);
  REQUIRE(respects_capacity(d.placement, h, sq));
  REQUIRE(h.is_placement(d.placement));
  // Every vertex is placed.
  for (const Vertex v : h.vertex_list()) REQUIRE(d.placement.contains(v));
}

TEST_CASE("annealing with a seeded initial placement is deterministic") {
  const HypergraphCircuit h = three_qubit_circuit();
  const ServerQubits sq = {{0, {0, 1}}, {1, {2}}, {2, {3}}};
  const std::vector<std::pair<int, int>> coupling = {{0, 1}, {1, 2}};

  // `initial_place_seed` mirrors passing an `initial_place_method` that wraps
  // Random with a fixed seed: the whole run becomes deterministic.
  const Distribution d1 = dqc::annealing_allocate(h, coupling, sq, server_graph_3(),
                                                  {}, 300, 3.0, std::nullopt, 42, 42);
  const Distribution d2 = dqc::annealing_allocate(h, coupling, sq, server_graph_3(),
                                                  {}, 300, 3.0, std::nullopt, 42, 42);
  REQUIRE(d1.placement.placement == d2.placement.placement);
  REQUIRE(respects_capacity(d1.placement, h, sq));
  REQUIRE(h.is_placement(d1.placement));
}
