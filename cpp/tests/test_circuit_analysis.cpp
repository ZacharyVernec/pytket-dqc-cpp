#include <catch2/catch_test_macros.hpp>

#include <map>
#include <string>
#include <vector>

#include "dqc/circuit_analysis.hpp"

using dqc::SerializedCircuit;
using dqc::Command;

TEST_CASE("get_server_id and is_link_qubit") {
  REQUIRE(dqc::get_server_id("server_3[5]") == 3);
  REQUIRE(dqc::get_server_id("server_0[0]") == 0);
  REQUIRE(dqc::get_server_id("server_0_link_register[2]") == 0);
  REQUIRE(dqc::get_server_id("server_12_link_register[7]") == 12);

  REQUIRE(dqc::is_link_qubit("server_3[5]") == false);
  REQUIRE(dqc::is_link_qubit("server_0_link_register[2]") == true);
}

TEST_CASE("ebit_cost counts starting and teleportation") {
  SerializedCircuit circ;
  circ.qubits = {"server_0[0]", "server_1[0]", "server_0_link_register[0]"};
  circ.commands = {
      {"starting_process", {"server_0[0]", "server_0_link_register[0]"}},
      {"teleportation", {"server_0_link_register[0]", "server_1[0]"}},
      {"ending_process", {"server_0_link_register[0]", "server_0[0]"}},
      {"starting_process", {"server_0[0]", "server_0_link_register[0]"}},
      {"teleportation", {"server_0_link_register[0]", "server_1[0]"}},
      {"ending_process", {"server_0_link_register[0]", "server_0[0]"}},
  };
  REQUIRE(dqc::ebit_cost(circ) == 4);  // 2 starting + 2 teleportation
}

TEST_CASE("ebit_cost asserts starting == ending") {
  SerializedCircuit circ;
  circ.commands = {
      {"starting_process", {"server_0[0]", "server_0_link_register[0]"}},
  };
  REQUIRE_THROWS_AS(dqc::ebit_cost(circ), std::runtime_error);
}

TEST_CASE("ebit_memory_required tracks concurrent ebits per server") {
  SerializedCircuit circ;
  circ.qubits = {"server_0[0]", "server_1[0]", "server_0_link_register[0]",
                 "server_0_link_register[1]"};
  circ.commands = {
      {"starting_process", {"server_0[0]", "server_0_link_register[0]"}},
      {"starting_process", {"server_0[0]", "server_0_link_register[1]"}},
      {"ending_process", {"server_0_link_register[0]", "server_0[0]"}},
      {"ending_process", {"server_0_link_register[1]", "server_0[0]"}},
  };
  const std::map<int, int> mem = dqc::ebit_memory_required(circ);
  REQUIRE(mem.at(0) == 2);
  REQUIRE(mem.at(1) == 0);
}

TEST_CASE("all_cu1_local checks server ids of CU1 qubits") {
  SerializedCircuit local;
  local.qubits = {"server_0[0]", "server_0[1]"};
  local.commands = {{"CU1", {"server_0[0]", "server_0[1]"}, dqc::kOpCU1},
                    {"H", {"server_0[0]"}, dqc::kOpH}};
  REQUIRE(dqc::all_cu1_local(local));

  SerializedCircuit remote;
  remote.qubits = {"server_0[0]", "server_1[0]"};
  remote.commands = {{"CU1", {"server_0[0]", "server_1[0]"}, dqc::kOpCU1}};
  REQUIRE(!dqc::all_cu1_local(remote));
}
