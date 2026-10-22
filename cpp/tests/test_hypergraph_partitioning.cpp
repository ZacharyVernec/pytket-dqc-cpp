#include <catch2/catch_test_macros.hpp>

#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#include "dqc/graph.hpp"
#include "dqc/hypergraph_circuit.hpp"
#include "dqc/hypergraph_partitioning.hpp"
#include "dqc/op_type.hpp"
#include "dqc/placement.hpp"

#ifndef DQC_KAHYPAR_INI
#error "DQC_KAHYPAR_INI must point at the bundled km1_kKaHyPar_sea20.ini"
#endif

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

int qubit_count_on(const Placement& p, const HypergraphCircuit& h, int server) {
  int count = 0;
  for (const auto& [v, s] : p.entries) {
    if (s == server && h.is_qubit_vertex(v)) ++count;
  }
  return count;
}

bool covers_all_vertices(const Placement& p, const HypergraphCircuit& h) {
  std::set<Vertex> placed;
  for (const auto& [v, s] : p.entries) {
    (void)s;
    placed.insert(v);
  }
  return placed == std::set<Vertex>(h.vertex_list().begin(), h.vertex_list().end());
}

}  // namespace

TEST_CASE("HypergraphPartitioning allocates a valid placement", "[kahypar]") {
  const HypergraphCircuit h = three_qubit_circuit();
  const ServerQubits sq = {{0, {0, 1}}, {1, {2, 3}}};
  Graph sg;
  sg.add_edge(0, 1);

  const Distribution d = dqc::hypergraph_partitioning_allocate(
      h, {{0, 1}}, sq, sg, {}, DQC_KAHYPAR_INI, 42);

  REQUIRE(covers_all_vertices(d.placement, h));
  // Per-server qubit capacity (2 each) must be respected after make_valid.
  REQUIRE(qubit_count_on(d.placement, h, 0) <= 2);
  REQUIRE(qubit_count_on(d.placement, h, 1) <= 2);
  REQUIRE(d.is_valid());
}

TEST_CASE("HypergraphPartitioning is deterministic for a fixed seed", "[kahypar]") {
  const HypergraphCircuit h = three_qubit_circuit();
  const ServerQubits sq = {{0, {0, 1}}, {1, {2, 3}}};
  Graph sg;
  sg.add_edge(0, 1);

  const Distribution d1 = dqc::hypergraph_partitioning_allocate(
      h, {{0, 1}}, sq, sg, {}, DQC_KAHYPAR_INI, 7);
  const Distribution d2 = dqc::hypergraph_partitioning_allocate(
      h, {{0, 1}}, sq, sg, {}, DQC_KAHYPAR_INI, 7);
  REQUIRE(d1.placement == d2.placement);
}

TEST_CASE("HypergraphPartitioning initial placement covers every vertex", "[kahypar]") {
  const HypergraphCircuit h = three_qubit_circuit();
  const ServerQubits sq = {{0, {0, 1}}, {1, {2, 3}}};

  const Placement p = dqc::hypergraph_partitioning_initial_placement(
      h, {{0, 1}}, sq, DQC_KAHYPAR_INI, 42);
  REQUIRE(covers_all_vertices(p, h));
}

TEST_CASE("HypergraphPartitioning rejects an un-implementable circuit", "[kahypar]") {
  SerializedCircuit c;
  c.qubits = {"q[0]", "q[1]", "q[2]", "q[3]", "q[4]"};
  c.commands = {{"CU1", {"q[0]", "q[1]"}, dqc::kOpCU1, {1.0}}};
  HypergraphCircuit h;
  h.from_circuit(c);

  const ServerQubits sq = {{0, {0}}, {1, {1}}};
  Graph sg;
  sg.add_edge(0, 1);
  REQUIRE_THROWS_AS(dqc::hypergraph_partitioning_allocate(
                        h, {{0, 1}}, sq, sg, {}, DQC_KAHYPAR_INI, 42),
                    std::runtime_error);
}
