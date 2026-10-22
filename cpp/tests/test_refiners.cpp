#include <catch2/catch_test_macros.hpp>

#include <map>
#include <optional>
#include <utility>
#include <vector>

#include "dqc/allocators.hpp"
#include "dqc/distribution.hpp"
#include "dqc/graph.hpp"
#include "dqc/hypergraph_circuit.hpp"
#include "dqc/op_type.hpp"
#include "dqc/placement.hpp"
#include "dqc/refiners/boundary_reallocation.hpp"
#include "dqc/refiners/vertex_cover.hpp"

using dqc::Distribution;
using dqc::Graph;
using dqc::HypergraphCircuit;
using dqc::Placement;
using dqc::SerializedCircuit;
using dqc::Vertex;

namespace {

HypergraphCircuit detached_circuit() {
  SerializedCircuit c;
  c.qubits = {"q[0]", "q[1]", "q[2]"};
  c.commands = {
      {"CU1", {"q[0]", "q[1]"}, dqc::kOpCU1, {0.3}},
      {"H", {"q[0]"}, dqc::kOpH},
      {"Rz", {"q[0]"}, dqc::kOpRz, {1.0}},
      {"H", {"q[0]"}, dqc::kOpH},
      {"CU1", {"q[0]", "q[2]"}, dqc::kOpCU1, {0.8}},
  };
  HypergraphCircuit h;
  h.from_circuit(c);
  return h;
}

Graph detached_server_graph() {
  Graph g;
  for (int i = 0; i < 5; ++i) g.add_node(i);
  g.add_edge(2, 1);
  g.add_edge(1, 0);
  g.add_edge(1, 3);
  g.add_edge(0, 4);
  return g;
}

std::vector<std::pair<int, std::vector<int>>> detached_server_qubits() {
  return {{0, {0}}, {1, {1}}, {2, {2}}, {3, {3}}, {4, {4, 5}}};
}

std::map<int, std::vector<int>> as_map(
    const std::vector<std::pair<int, std::vector<int>>>& sq) {
  return std::map<int, std::vector<int>>(sq.begin(), sq.end());
}

std::vector<std::pair<int, int>> detached_coupling() {
  return {{2, 1}, {1, 0}, {1, 3}, {0, 4}};
}

bool placement_has(const Placement& p,
                   const std::vector<std::pair<int, int>>& entries) {
  for (const auto& [v, s] : entries) {
    if (!p.contains(v) || p.at(v) != s) return false;
  }
  return p.placement.size() == entries.size();
}

}  // namespace

TEST_CASE("detached gates refiner matches reference (seeded)") {
  const HypergraphCircuit h = detached_circuit();
  const Placement init =
      dqc::random_allocate(h, detached_coupling(), detached_server_qubits(),
                           detached_server_graph(), {}, 7)
          .placement;

  // Reference initial placement from `Random().allocate(circ, network, seed=7)`.
  REQUIRE(placement_has(init, {{0, 2}, {1, 0}, {2, 4}, {3, 1}, {4, 3}}));

  Distribution dist{h, init, detached_server_graph(),
                    as_map(detached_server_qubits())};
  REQUIRE(dist.is_valid());

  const bool made =
      dqc::detached_gates_refine(dist, 10, 0.05, std::uint64_t{42});
  REQUIRE(made);
  // Reference: after `DetachedGates().refine(dist, seed=42)`.
  REQUIRE(placement_has(dist.placement,
                        {{0, 2}, {1, 0}, {2, 4}, {3, 2}, {4, 4}}));
  REQUIRE(dist.cost() == 5);
  REQUIRE(dist.is_valid());
}

TEST_CASE("boundary reallocation refiner is deterministic and cost-consistent") {
  const HypergraphCircuit h = detached_circuit();
  const Placement init =
      dqc::random_allocate(h, detached_coupling(), detached_server_qubits(),
                           detached_server_graph(), {}, 7)
          .placement;

  Distribution dist1{h, init, detached_server_graph(),
                     as_map(detached_server_qubits())};
  Distribution dist2{h, init, detached_server_graph(),
                     as_map(detached_server_qubits())};
  const bool made1 = dqc::boundary_reallocation_refine(
      dist1, {}, 10, 0.05, std::uint64_t{42});
  const bool made2 = dqc::boundary_reallocation_refine(
      dist2, {}, 10, 0.05, std::uint64_t{42});
  REQUIRE(made1 == made2);
  REQUIRE(dist1.placement.placement == dist2.placement.placement);
  REQUIRE(dist1.cost() == dist2.cost());
  // The recomputed cost equals the refined cost.
  REQUIRE(dist1.cost() == dqc::distribution_cost(dist1.circuit,
                                                 dist1.placement,
                                                 dist1.server_graph));
  REQUIRE(dist1.is_valid());
}

TEST_CASE("vertex cover refiner produces a valid refined distribution") {
  const HypergraphCircuit h = detached_circuit();
  const Placement init =
      dqc::random_allocate(h, detached_coupling(), detached_server_qubits(),
                           detached_server_graph(), {}, 7)
          .placement;

  Distribution dist{h, init, detached_server_graph(),
                    as_map(detached_server_qubits())};
  REQUIRE(dqc::vertex_cover_refine(dist));
  REQUIRE(dist.is_valid());
  REQUIRE(dist.cost() == dqc::distribution_cost(dist.circuit, dist.placement,
                                                dist.server_graph));
  // Every vertex must remain placed.
  for (const Vertex v : dist.circuit.vertex_list()) {
    REQUIRE(dist.placement.contains(v));
  }
  // Refining twice more is idempotent in cost (a fixed point in cost terms).
  const int cost_before = dist.cost();
  dqc::vertex_cover_refine(dist);
  REQUIRE(dist.cost() == cost_before);
}
