#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

#include "dqc/hypergraph_circuit.hpp"
#include "dqc/op_type.hpp"

using dqc::HypergraphCircuit;
using dqc::SerializedCircuit;
using dqc::Command;

TEST_CASE("from_circuit builds hypergraph (H/Rz/CU1)") {
  SerializedCircuit circ;
  circ.qubits = {"q[0]", "q[1]"};
  circ.commands = {
      {"H", {"q[0]"}, dqc::kOpH},
      {"Rz", {"q[0]"}, dqc::kOpRz},
      {"CU1", {"q[0]", "q[1]"}, dqc::kOpCU1},
      {"H", {"q[0]"}, dqc::kOpH},
  };

  HypergraphCircuit h;
  h.from_circuit(circ);

  // Vertex order follows insertion: qubit 0, then its gate vertex 2,
  // then qubit 1.
  REQUIRE(h.vertex_list() == std::vector<int>{0, 2, 1});
  REQUIRE(h.hyperedge_list().size() == 2);
  REQUIRE(h.hyperedge_list()[0].vertices == std::vector<int>{0, 2});
  REQUIRE(h.hyperedge_list()[0].weight == 1);
  REQUIRE(h.hyperedge_list()[1].vertices == std::vector<int>{1, 2});
  REQUIRE(h.hyperedge_list()[1].weight == 1);
  REQUIRE(h.get_qubit_vertices() == std::vector<int>{0, 1});
  REQUIRE(h.is_qubit_vertex(0));
  REQUIRE(h.is_qubit_vertex(1));
  REQUIRE(!h.is_qubit_vertex(2));
}

TEST_CASE("from_circuit accumulates consecutive CU1 into one hyperedge") {
  SerializedCircuit circ;
  circ.qubits = {"q[0]", "q[1]"};
  circ.commands = {
      {"CU1", {"q[0]", "q[1]"}, dqc::kOpCU1},
      {"CU1", {"q[0]", "q[1]"}, dqc::kOpCU1},
      {"H", {"q[0]"}, dqc::kOpH},
  };

  HypergraphCircuit h;
  h.from_circuit(circ);

  // Vertex order: qubit 0, gate 2, gate 3, qubit 1.
  REQUIRE(h.vertex_list() == std::vector<int>{0, 2, 3, 1});
  REQUIRE(h.hyperedge_list().size() == 2);
  REQUIRE(h.hyperedge_list()[0].vertices == std::vector<int>{0, 2, 3});
  REQUIRE(h.hyperedge_list()[1].vertices == std::vector<int>{1, 2, 3});
}
