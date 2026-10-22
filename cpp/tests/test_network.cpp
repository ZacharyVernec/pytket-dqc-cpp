#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <map>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#include "dqc/graph.hpp"
#include "dqc/hypergraph_circuit.hpp"
#include "dqc/network.hpp"
#include "dqc/op_type.hpp"
#include "dqc/placement.hpp"

using dqc::Graph;
using dqc::HypergraphCircuit;
using dqc::NISQNetwork;
using dqc::Placement;
using dqc::SerializedCircuit;

namespace {

using ServerQubits = std::vector<std::pair<int, std::vector<int>>>;

NISQNetwork two_server_network() {
  return NISQNetwork(
      {{0, 1}},
      {{0, {0, 1}}, {1, {2, 3}}},
      {{0, 2}, {1, 2}});
}

NISQNetwork three_server_all_to_all() {
  return NISQNetwork(
      {{0, 1}, {0, 2}, {1, 2}},
      {{0, {0, 1}}, {1, {2, 3}}, {2, {4, 5}}},
      {{0, 2}, {1, 2}, {2, 2}});
}

HypergraphCircuit two_qubit_circuit() {
  SerializedCircuit c;
  c.qubits = {"q[0]", "q[1]"};
  c.commands = {{"CU1", {"q[0]", "q[1]"}, dqc::kOpCU1, {1.0}}};
  HypergraphCircuit h;
  h.from_circuit(c);
  return h;
}

}  // namespace

TEST_CASE("NISQNetwork basic accessors", "[network]") {
  NISQNetwork n = two_server_network();
  REQUIRE(n.server_coupling() == std::vector<std::pair<int, int>>({{0, 1}}));
  REQUIRE(n.server_qubits() == ServerQubits({{0, {0, 1}}, {1, {2, 3}}}));
  REQUIRE(n.server_ebit_mem() == std::map<int, int>({{0, 2}, {1, 2}}));
  // get_server_list is CPython int-set order of the coupling server ids.
  const std::vector<int> sl = n.get_server_list();
  REQUIRE(std::set<int>(sl.begin(), sl.end()) == std::set<int>({0, 1}));
  REQUIRE(n.get_qubit_list() == std::vector<int>({0, 1, 2, 3}));
}

TEST_CASE("NISQNetwork defaults ebit memory to unbounded (-1)", "[network]") {
  NISQNetwork n({{0, 1}}, {{0, {0}}, {1, {1}}});
  REQUIRE(n.server_ebit_mem() == std::map<int, int>({{0, -1}, {1, -1}}));
}

TEST_CASE("NISQNetwork can_implement", "[network]") {
  NISQNetwork n = two_server_network();
  REQUIRE(n.can_implement(two_qubit_circuit()));
  SerializedCircuit c;
  c.qubits = {"q[0]", "q[1]", "q[2]", "q[3]", "q[4]"};
  c.commands = {{"CU1", {"q[0]", "q[1]"}, dqc::kOpCU1, {1.0}}};
  HypergraphCircuit too_big;
  too_big.from_circuit(c);
  REQUIRE(!n.can_implement(too_big));
}

TEST_CASE("NISQNetwork is_placement", "[network]") {
  NISQNetwork n = two_server_network();
  Placement ok;
  ok.set(0, 0);
  ok.set(1, 1);
  REQUIRE(n.is_placement(ok));
  Placement bad;
  bad.set(0, 7);  // server 7 not in the network
  REQUIRE(!n.is_placement(bad));
}

TEST_CASE("NISQNetwork rejects an unconnected server network", "[network]") {
  REQUIRE_THROWS_WITH(NISQNetwork({{0, 1}, {2, 3}}, {{0, {0}}, {1, {1}}, {2, {2}}, {3, {3}}}),
                      Catch::Matchers::ContainsSubstring("unconnected"));
}

TEST_CASE("NISQNetwork rejects servers without specified qubits", "[network]") {
  REQUIRE_THROWS_WITH(NISQNetwork({{0, 1}}, {{0, {0, 1}}}),
                      Catch::Matchers::ContainsSubstring("have not been specified"));
}

TEST_CASE("NISQNetwork rejects qubits in more than one server", "[network]") {
  REQUIRE_THROWS_WITH(
      NISQNetwork({{0, 1}}, {{0, {0, 1}}, {1, {1, 2}}}),
      Catch::Matchers::ContainsSubstring("may belong to only one server"));
}

TEST_CASE("NISQNetwork server graph and nisq graph structure", "[network]") {
  NISQNetwork n = three_server_all_to_all();
  const Graph sg = n.get_server_nx();
  REQUIRE(sg.nodes() == std::vector<int>({0, 1, 2}));
  REQUIRE(sg.has_edge(0, 1));
  REQUIRE(sg.has_edge(0, 2));
  REQUIRE(sg.has_edge(1, 2));

  const Graph ng = n.get_nisq_nx();
  // Intra-server edges (weight 0): (0,1), (2,3), (4,5).
  REQUIRE(ng.has_edge(0, 1));
  REQUIRE(ng.has_edge(2, 3));
  REQUIRE(ng.has_edge(4, 5));
  // Inter-server edges (weight 1): first qubit of each pair.
  REQUIRE(ng.has_edge(0, 2));
  REQUIRE(ng.has_edge(0, 4));
  REQUIRE(ng.has_edge(2, 4));
  REQUIRE(ng.weight(0, 1) == 0.0);
  REQUIRE(ng.weight(0, 2) == 1.0);
}

TEST_CASE("NISQNetwork equality", "[network]") {
  NISQNetwork a = two_server_network();
  NISQNetwork b = two_server_network();
  REQUIRE(a == b);
  REQUIRE(!(a != b));
}
