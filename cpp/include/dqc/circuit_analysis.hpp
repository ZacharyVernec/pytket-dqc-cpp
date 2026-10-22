#pragma once

#include <cstddef>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "dqc/circuit.hpp"
#include "dqc/op_type.hpp"

namespace dqc {

// Mirrors `pytket_dqc.utils.circuit_analysis.ConstraintException`.
struct ConstraintException : std::runtime_error {
  int server;
  int v_gate = -1;  // Python initialises v_gate to None.

  ConstraintException(const std::string& message, int server)
      : std::runtime_error(message), server(server) {}
};

// Faithful `str.split(delim)` (keeps empty segments).
inline std::vector<std::string> split_str(const std::string& s, char delim) {
  std::vector<std::string> out;
  std::size_t start = 0;
  for (std::size_t i = 0; i <= s.size(); ++i) {
    if (i == s.size() || s[i] == delim) {
      out.push_back(s.substr(start, i - start));
      start = i + 1;
    }
  }
  return out;
}

// Mirrors `is_link_qubit`: link qubits have a name of the form
// `server_<id>_link_register[<n>]` (so `str(qubit).split("_")` has > 2 parts).
inline bool is_link_qubit(const std::string& qubit_str) {
  return split_str(qubit_str, '_').size() > 2;
}

// Mirrors `get_server_id`: parse the server id from the qubit's string form.
inline int get_server_id(const std::string& qubit_str) {
  const std::vector<std::string> parts = split_str(qubit_str, '_');
  if (parts.empty() || parts[0] != "server") {
    throw std::runtime_error("qubit name must start with 'server'");
  }
  if (parts.size() > 2) {
    return std::stoi(parts[1]);
  }
  return std::stoi(split_str(parts[1], '[')[0]);
}

// Mirrors `all_cu1_local` (which checks `op.type == OpType.CU1`).
inline bool all_cu1_local(const SerializedCircuit& circ) {
  for (const Command& cmd : circ.commands) {
    if (cmd.op_type == kOpCU1) {
      if (get_server_id(cmd.qubits[0]) != get_server_id(cmd.qubits[1])) {
        return false;
      }
    }
  }
  return true;
}

// Mirrors `ebit_memory_required`.
inline std::map<int, int> ebit_memory_required(const SerializedCircuit& circ) {
  std::map<int, int> mem;
  std::map<int, int> current;
  for (const std::string& q : circ.qubits) {
    const int sid = get_server_id(q);
    mem[sid] = 0;
    current[sid] = 0;
  }
  for (const Command& cmd : circ.commands) {
    if (cmd.op_name == "starting_process") {
      const std::string& link_qubit = cmd.qubits[1];
      const int sid = get_server_id(link_qubit);
      current[sid] += 1;
      if (current[sid] > mem[sid]) {
        mem[sid] += 1;
      }
    } else if (cmd.op_name == "ending_process") {
      const std::string& link_qubit = cmd.qubits[0];
      const int sid = get_server_id(link_qubit);
      current[sid] -= 1;
    }
  }
  return mem;
}

// Mirrors `ebit_cost`.
inline int ebit_cost(const SerializedCircuit& circ) {
  int starting = 0;
  int ending = 0;
  int telep = 0;
  for (const Command& cmd : circ.commands) {
    if (cmd.op_name == "starting_process") {
      ++starting;
    } else if (cmd.op_name == "ending_process") {
      ++ending;
    } else if (cmd.op_name == "teleportation") {
      ++telep;
    }
  }
  if (starting != ending) {
    throw std::runtime_error("starting_process count != ending_process count");
  }
  return starting + telep;
}

}  // namespace dqc
