#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "dqc/circuit_analysis.hpp"
#include "dqc/distribution.hpp"
#include "dqc/graph.hpp"
#include "dqc/hypergraph_circuit.hpp"
#include "dqc/op_type.hpp"
#include "dqc/placement.hpp"
#include "dqc/to_pytket_circuit.hpp"

using dqc::Distribution;
using dqc::Graph;
using dqc::HypergraphCircuit;
using dqc::Placement;
using dqc::SerializedCircuit;
using dqc::ToPytketCircuitResult;
using dqc::Vertex;

namespace {

Graph path_graph_5() {
  Graph g;
  for (int i = 0; i < 5; ++i) g.add_node(i);
  g.add_edge(0, 1);
  g.add_edge(1, 2);
  g.add_edge(1, 3);
  g.add_edge(2, 4);
  return g;
}

std::map<int, std::vector<int>> sq5() {
  return {{0, {0}}, {1, {1}}, {2, {2}}, {3, {3}}, {4, {4}}};
}

// Two CU1 gates on qubit 0, no H embedding (upstream
// `test_distribution_cost_no_embedding`).
HypergraphCircuit no_embedding_circuit() {
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

// The big ALAP / H-embedding circuit from `test_alap`.
HypergraphCircuit alap_circuit() {
  SerializedCircuit c;
  c.qubits = {"q[0]", "q[1]", "q[2]", "q[3]"};
  auto cu1 = [&](double p, const char* a, const char* b) {
    c.commands.push_back({"CU1", {a, b}, dqc::kOpCU1, {p}});
  };
  auto H = [&](const char* a) { c.commands.push_back({"H", {a}, dqc::kOpH}); };
  auto Rz = [&](double p, const char* a) {
    c.commands.push_back({"Rz", {a}, dqc::kOpRz, {p}});
  };
  cu1(0.1234, "q[1]", "q[2]");
  cu1(0.1234, "q[0]", "q[2]");
  cu1(0.1234, "q[2]", "q[3]");
  cu1(0.1234, "q[0]", "q[3]");
  H("q[0]");
  H("q[2]");
  Rz(0.1234, "q[3]");
  cu1(1.0, "q[0]", "q[2]");
  cu1(1.0, "q[0]", "q[3]");
  cu1(1.0, "q[1]", "q[2]");
  H("q[0]");
  H("q[2]");
  Rz(0.1234, "q[0]");
  cu1(0.1234, "q[0]", "q[1]");
  cu1(0.1234, "q[0]", "q[3]");
  cu1(1.0, "q[1]", "q[2]");
  HypergraphCircuit h;
  h.from_circuit(c);
  return h;
}

Placement placement_of(const std::vector<std::pair<int, int>>& entries) {
  Placement p;
  for (const auto& [v, s] : entries) p.set(v, s);
  return p;
}

}  // namespace

TEST_CASE("to_pytket_circuit: no embedding matches cost") {
  const HypergraphCircuit h = no_embedding_circuit();
  const Graph g = path_graph_5();
  const Placement p = placement_of({{0, 0}, {1, 1}, {2, 2}, {3, 1}, {4, 2}});
  Distribution dist{h, p, g, sq5()};

  const ToPytketCircuitResult res = dqc::to_pytket_circuit(dist);
  REQUIRE(res.real_ebit_cost == res.estimated_cost);
  REQUIRE(!res.cost_mismatch_detached_embedded);
  REQUIRE(dqc::all_cu1_local(res.circuit));
  // No teleportation: real cost equals the Steiner-tree cost.
  REQUIRE(res.real_ebit_cost == dist.cost());
}

TEST_CASE("to_pytket_circuit: ALAP / H-embedding matches cost") {
  const HypergraphCircuit h = alap_circuit();
  Graph g;
  for (int i = 0; i < 5; ++i) g.add_node(i);
  g.add_edge(0, 1);
  g.add_edge(0, 2);
  g.add_edge(0, 3);
  g.add_edge(3, 4);
  const int pl[14] = {1, 1, 4, 4, 1, 2, 1, 4, 4, 4, 0, 1, 4, 2};
  Placement p;
  for (int v = 0; v < 14; ++v) p.set(v, pl[v]);
  Distribution dist{h, p, g,
                    {{0, {0}}, {1, {1, 2}}, {2, {3, 4}}, {3, {7}}, {4, {5, 6}}}};

  const ToPytketCircuitResult res = dqc::to_pytket_circuit(dist);
  REQUIRE(res.real_ebit_cost == res.estimated_cost);
  REQUIRE(dqc::all_cu1_local(res.circuit));
  REQUIRE(res.real_ebit_cost == dist.cost());
}

TEST_CASE("to_pytket_circuit: bounded memory raises ConstraintException") {
  // A detached gate (gate on server 2, neither qubit home) needs two link
  // qubits on server 2, exceeding the bound of 1.
  SerializedCircuit c;
  c.qubits = {"q[0]", "q[1]", "q[2]", "q[3]"};
  c.commands = {{"CU1", {"q[0]", "q[1]"}, dqc::kOpCU1, {1.0}}};
  HypergraphCircuit h;
  h.from_circuit(c);

  Graph g;
  g.add_node(0);
  g.add_node(1);
  g.add_node(2);
  g.add_edge(0, 1);
  g.add_edge(1, 2);

  const Placement p = placement_of({{0, 0}, {1, 1}, {2, 2}, {3, 2}, {4, 2}});
  Distribution dist{h, p, g, {{0, {0}}, {1, {1}}, {2, {2, 3}}}};
  dist.server_ebit_mem = {{0, 1}, {1, 1}, {2, 1}};

  REQUIRE_THROWS_AS(dqc::to_pytket_circuit(dist, true, false),
                    dqc::ConstraintException);
}

// Interleaved hyperedge spans force server 1 to hold two link qubits
// simultaneously, exceeding the bound of 1: qubit 0's hyperedge {0,4,6} keeps
// its connection alive across gates 4 and 6 while qubit 1's gate 5 opens its
// own. `allow_update=True` must split the shared-qubit hyperedge {0,4,6} into
// {0,4} and {0,6} and then build the circuit.
TEST_CASE("to_pytket_circuit: allow_update=True splits hyperedge") {
  SerializedCircuit c;
  c.qubits = {"q[0]", "q[1]", "q[2]", "q[3]"};
  c.commands = {
      {"CU1", {"q[0]", "q[2]"}, dqc::kOpCU1, {1.0}},
      {"CU1", {"q[1]", "q[3]"}, dqc::kOpCU1, {1.0}},
      {"CU1", {"q[0]", "q[2]"}, dqc::kOpCU1, {1.0}},
  };
  HypergraphCircuit h;
  h.from_circuit(c);

  Graph g;
  g.add_node(0);
  g.add_node(1);
  g.add_edge(0, 1);

  const Placement p = placement_of({{0, 0}, {1, 0}, {2, 1}, {3, 1},
                                    {4, 1}, {5, 1}, {6, 1}});

  // Without update the bound is exceeded.
  Distribution dist{h, p, g, {{0, {0, 1}}, {1, {2, 3}}}};
  dist.server_ebit_mem = {{0, 1}, {1, 1}};
  REQUIRE_THROWS_AS(dqc::to_pytket_circuit(dist, true, false),
                    dqc::ConstraintException);

  // With update, the shared-qubit hyperedge {0,4,6} is split into {0,4} and
  // {0,6}, and the circuit builds successfully.
  Distribution dist2{h, p, g, {{0, {0, 1}}, {1, {2, 3}}}};
  dist2.server_ebit_mem = {{0, 1}, {1, 1}};
  const ToPytketCircuitResult res = dqc::to_pytket_circuit(dist2, true, true);

  const auto has_hedge = [&](std::initializer_list<Vertex> vs) {
    const std::set<Vertex> target(vs);
    for (const auto& hedge : dist2.circuit.hyperedge_list()) {
      if (std::set<Vertex>(hedge.vertices.begin(), hedge.vertices.end()) ==
          target) {
        return true;
      }
    }
    return false;
  };
  REQUIRE(has_hedge({0, 4}));
  REQUIRE(has_hedge({0, 6}));
  REQUIRE(dqc::all_cu1_local(res.circuit));
  REQUIRE(res.real_ebit_cost == 3);
}

// Two hyperedges can be split at the violation point (server 1, v_gate 4):
// {1,4,5,7} has span 4..5 (dist 1) and {3,4,6} has span 4..6 (dist 2). The
// `split_for_bound` selection must choose the LONGEST-span candidate {3,4,6}.
// A regression that always picked the first valid candidate (because every
// span was computed as 0) would instead split {1,4,5,7} and diverge.
TEST_CASE("to_pytket_circuit: allow_update=True splits longest-span candidate") {
  SerializedCircuit c;
  c.qubits = {"q[0]", "q[1]", "q[2]", "q[3]"};
  // Gate vertices: CU1(q1,q3)->4, CU1(q0,q1)->5, CU1(q2,q3)->6, CU1(q1,q2)->7.
  c.commands = {
      {"CU1", {"q[1]", "q[3]"}, dqc::kOpCU1, {1.0}},
      {"CU1", {"q[0]", "q[1]"}, dqc::kOpCU1, {1.0}},
      {"CU1", {"q[2]", "q[3]"}, dqc::kOpCU1, {1.0}},
      {"CU1", {"q[1]", "q[2]"}, dqc::kOpCU1, {1.0}},
  };
  HypergraphCircuit h;
  h.from_circuit(c);

  Graph g;
  g.add_node(0);
  g.add_node(1);
  g.add_edge(0, 1);

  // Qubits 0 and 3 live on server 0; qubits 1 and 2 on server 1; all gates on
  // server 1.
  const Placement p = placement_of({{0, 0}, {3, 0}, {1, 1}, {2, 1},
                                    {4, 1}, {5, 1}, {6, 1}, {7, 1}});

  // Without update the bound (memory 1) is exceeded on server 1.
  Distribution dist{h, p, g, {{0, {0, 3}}, {1, {1, 2}}}};
  dist.server_ebit_mem = {{0, 1}, {1, 1}};
  REQUIRE_THROWS_AS(dqc::to_pytket_circuit(dist, true, false),
                    dqc::ConstraintException);

  // With update, the longest-span candidate {3,4,6} (dist 2) is split into
  // {3,4} and {3,6}, NOT the first candidate {1,4,5,7} (dist 1).
  Distribution dist2{h, p, g, {{0, {0, 3}}, {1, {1, 2}}}};
  dist2.server_ebit_mem = {{0, 1}, {1, 1}};
  const ToPytketCircuitResult res = dqc::to_pytket_circuit(dist2, true, true);

  const auto has_hedge = [&](std::initializer_list<Vertex> vs) {
    const std::set<Vertex> target(vs);
    for (const auto& hedge : dist2.circuit.hyperedge_list()) {
      if (std::set<Vertex>(hedge.vertices.begin(), hedge.vertices.end()) ==
          target) {
        return true;
      }
    }
    return false;
  };
  // The correct (longest-span) split: {3,4,6} -> {3,4} + {3,6}.
  REQUIRE(has_hedge({3, 4}));
  REQUIRE(has_hedge({3, 6}));
  REQUIRE(!has_hedge({3, 4, 6}));
  // The short-span first candidate {1,4,5,7} must NOT have been split.
  REQUIRE(has_hedge({1, 4, 5, 7}));
  REQUIRE(!has_hedge({1, 4}));
  // The remaining hyperedges are untouched.
  REQUIRE(has_hedge({0, 5}));
  REQUIRE(has_hedge({2, 6, 7}));
  REQUIRE(dqc::all_cu1_local(res.circuit));
  REQUIRE(res.real_ebit_cost == 3);
}
