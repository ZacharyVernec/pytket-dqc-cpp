#pragma once

#include <algorithm>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "dqc/circuit.hpp"
#include "dqc/gateset.hpp"
#include "dqc/hypergraph.hpp"
#include "dqc/numeric.hpp"
#include "dqc/op_type.hpp"

namespace dqc {

// Annotated command produced by `from_circuit` (mirrors `_commands` entries).
struct CircuitCommand {
  int op_type = -1;
  std::vector<double> params;
  std::vector<std::string> qubits;  // qubit strings (str(qubit))
  int two_q_gate_count = -1;        // set for CZ/CU1/CX commands
  int vertex = -1;                  // gate vertex (distributed gates)
  std::string kind;                 // "distributed gate" or "1q local gate"

  Command to_command() const { return Command{"", qubits, op_type, params}; }
};

// Mirrors `pytket_dqc.circuits.hypergraph_circuit.HypergraphCircuit`'s
// hypergraph construction (`from_circuit`) and subcircuit/H-embedding helpers.
// The gateset predicate is checked by the Python adapter before serialization.
class HypergraphCircuit : public Hypergraph {
 public:
  void from_circuit(const SerializedCircuit& circuit);

  const std::vector<CircuitCommand>& commands() const { return commands_; }

  // Mirrors `get_circuit()`: returns the serialized source circuit.
  const SerializedCircuit& get_circuit() const { return circuit_; }

  bool is_qubit_vertex(Vertex v) const {
    return qubit_vertex_map_.count(v) != 0;
  }

  std::vector<Vertex> get_qubit_vertices() const {
    std::vector<Vertex> out;
    for (const Vertex v : vertex_list()) {
      if (is_qubit_vertex(v)) out.push_back(v);
    }
    return out;
  }

  Vertex get_qubit_vertex(const Hyperedge& hyperedge) const {
    std::vector<Vertex> qubits;
    for (const Vertex v : hyperedge.vertices) {
      if (is_qubit_vertex(v)) qubits.push_back(v);
    }
    return qubits[0];
  }

  std::vector<Vertex> get_gate_vertices(const Hyperedge& hyperedge) const {
    std::vector<Vertex> gates;
    for (const Vertex v : hyperedge.vertices) {
      if (!is_qubit_vertex(v)) gates.push_back(v);
    }
    return gates;
  }

  const std::string& qubit_of_vertex(Vertex v) const {
    return qubits_.at(qubit_vertex_map_.at(v));
  }

  Vertex get_vertex_of_qubit(const std::string& qubit) const {
    for (const auto& [v, qi] : qubit_vertex_map_) {
      if (qubits_[qi] == qubit) return v;
    }
    throw std::out_of_range("qubit not found");
  }

  Command get_gate_of_vertex(Vertex v) const {
    const int ci = gate_vertex_map_.at(v);
    return commands_[ci].to_command();
  }

  const std::map<Vertex, int>& get_vertex_to_command_index_map() const {
    return gate_vertex_map_;
  }

  // Mirrors `get_last_gate_vertex`: max gate vertex = last gate in circuit.
  Vertex get_last_gate_vertex(const std::vector<Vertex>& gate_vertex_list) const {
    return *std::max_element(gate_vertex_list.begin(), gate_vertex_list.end());
  }

  // Mirrors `get_first_gate_vertex`: min gate vertex = first gate in circuit.
  Vertex get_first_gate_vertex(
      const std::vector<Vertex>& gate_vertex_list) const {
    return *std::min_element(gate_vertex_list.begin(), gate_vertex_list.end());
  }

  // Mirrors `is_h_embeddable_CU1`.
  bool is_h_embeddable_CU1(const Command& command,
                           const std::set<int>& servers,
                           const Placement& placement) const {
    std::set<int> this_command_servers;
    for (const std::string& qubit : command.qubits) {
      this_command_servers.insert(placement.at(get_vertex_of_qubit(qubit)));
    }
    if (servers != this_command_servers) return false;
    const double p = py_mod(command.params[0], 1.0);
    return isclose(p, 0.0) || isclose(p, 1.0);
  }

  std::vector<Command> get_intermediate_commands(Vertex first_vertex,
                                                 Vertex second_vertex,
                                                 Vertex qubit_vertex) const {
    const std::string& qubit = qubit_of_vertex(qubit_vertex);
    const int first_ci = gate_vertex_map_.at(first_vertex);
    const int second_ci = gate_vertex_map_.at(second_vertex);
    std::vector<Command> out;
    for (int i = first_ci + 1; i < second_ci; ++i) {
      const CircuitCommand& cc = commands_[i];
      if (std::find(cc.qubits.begin(), cc.qubits.end(), qubit) !=
          cc.qubits.end()) {
        out.push_back(cc.to_command());
      }
    }
    return out;
  }

  std::vector<Command> get_hyperedge_subcircuit(
      const Hyperedge& hyperedge) const;

  // Mirrors `get_hyperedges_containing`: hyperedges containing ALL of
  // `vertices` as a subset of their own vertices.
  std::vector<Hyperedge> get_hyperedges_containing(
      const std::vector<Vertex>& vertices) const {
    std::vector<Hyperedge> out;
    for (const Hyperedge& hedge : hyperedge_list()) {
      bool contains_all = true;
      for (const Vertex v : vertices) {
        if (std::find(hedge.vertices.begin(), hedge.vertices.end(), v) ==
            hedge.vertices.end()) {
          contains_all = false;
          break;
        }
      }
      if (contains_all) out.push_back(hedge);
    }
    return out;
  }

  // Mirrors `get_h_embedded_gate_vertices`.
  std::vector<Vertex> get_h_embedded_gate_vertices(
      const Hyperedge& hyperedge) const {
    const std::vector<Command> subcircuit =
        get_hyperedge_subcircuit(hyperedge);
    std::vector<Vertex> gate_vertices_in_subcircuit;
    const std::vector<Vertex> gates = get_gate_vertices(hyperedge);
    const Vertex min_g = *std::min_element(gates.begin(), gates.end());
    const Vertex max_g = *std::max_element(gates.begin(), gates.end());
    const std::string& q = qubit_of_vertex(get_qubit_vertex(hyperedge));
    for (Vertex v_g = min_g; v_g <= max_g; ++v_g) {
      const Command gate = get_gate_of_vertex(v_g);
      if (std::find(gate.qubits.begin(), gate.qubits.end(), q) !=
          gate.qubits.end()) {
        gate_vertices_in_subcircuit.push_back(v_g);
      }
    }
    std::vector<Vertex> h_embedded;
    bool currently_embedding = false;
    std::size_t gi = 0;
    for (const Command& cmd : subcircuit) {
      if (cmd.op_type == kOpH) {
        currently_embedding = !currently_embedding;
      } else if (cmd.op_type == kOpCU1) {
        const Vertex v_g = gate_vertices_in_subcircuit[gi++];
        if (currently_embedding) h_embedded.push_back(v_g);
      }
    }
    return h_embedded;
  }

  // Mirrors `get_all_h_embedded_gate_vertices`. Iteration order of the
  // underlying CPython set only affects downstream membership tests, so an
  // ascending `std::set` is equivalent here.
  std::vector<Vertex> get_all_h_embedded_gate_vertices() const {
    std::set<Vertex> h_embedded;
    for (const Hyperedge& hedge : hyperedge_list()) {
      for (const Vertex v : get_h_embedded_gate_vertices(hedge)) {
        h_embedded.insert(v);
      }
    }
    return std::vector<Vertex>(h_embedded.begin(), h_embedded.end());
  }

  bool requires_h_embedded_cu1(const Hyperedge& hyperedge) const {
    const std::vector<Command> commands = get_hyperedge_subcircuit(hyperedge);
    bool currently_embedding = false;
    for (const Command& cmd : commands) {
      if (cmd.op_type == kOpH) {
        currently_embedding = !currently_embedding;
      } else if (currently_embedding && cmd.op_type == kOpCU1) {
        return true;
      }
    }
    return false;
  }

 private:
  void add_qubit_vertex(Vertex v, int qubit_index) {
    add_vertex(v);
    qubit_vertex_map_[v] = qubit_index;
  }
  void add_gate_vertex(Vertex v, int command_index) {
    add_vertex(v);
    gate_vertex_map_[v] = command_index;
  }

  std::vector<std::string> qubits_;
  std::vector<CircuitCommand> commands_;
  std::map<Vertex, int> qubit_vertex_map_;  // vertex -> qubit index
  std::map<Vertex, int> gate_vertex_map_;   // vertex -> command index
  SerializedCircuit circuit_;               // serialized source circuit
};

inline void HypergraphCircuit::from_circuit(const SerializedCircuit& circuit) {
  const int n_qubits = static_cast<int>(circuit.qubits.size());
  qubits_ = circuit.qubits;
  circuit_ = circuit;

  commands_.clear();
  qubit_vertex_map_.clear();
  gate_vertex_map_.clear();

  int two_q_gate_count = 0;
  for (const Command& cmd : circuit.commands) {
    CircuitCommand cc;
    cc.op_type = cmd.op_type;
    cc.params = cmd.params;
    cc.qubits = cmd.qubits;
    if (cmd.op_type == kOpCZ || cmd.op_type == kOpCU1 || cmd.op_type == kOpCX) {
      cc.two_q_gate_count = two_q_gate_count++;
    }
    commands_.push_back(cc);
  }

  for (int qubit_index = 0; qubit_index < n_qubits; ++qubit_index) {
    const std::string& qubit = circuit.qubits[qubit_index];
    add_qubit_vertex(qubit_index, qubit_index);
    std::vector<Vertex> hyperedge = {qubit_index};

    std::vector<int> qubit_command_indices;
    for (int ci = 0; ci < static_cast<int>(commands_.size()); ++ci) {
      const auto& qs = commands_[ci].qubits;
      if (std::find(qs.begin(), qs.end(), qubit) != qs.end()) {
        qubit_command_indices.push_back(ci);
      }
    }

    bool two_qubit_gate_found = false;
    for (const int ci : qubit_command_indices) {
      CircuitCommand& command = commands_[ci];
      const int op_type = command.op_type;
      if (op_type == kOpCZ || op_type == kOpCU1) {
        two_qubit_gate_found = true;
        const int vertex = command.two_q_gate_count + n_qubits;
        add_gate_vertex(vertex, ci);
        hyperedge.push_back(vertex);
        command.vertex = vertex;
        command.kind = "distributed gate";
      } else if (op_type == kOpCX) {
        two_qubit_gate_found = true;
        int vertex;
        if (qubit == command.qubits[0]) {  // control
          vertex = command.two_q_gate_count + n_qubits;
          add_gate_vertex(vertex, ci);
          hyperedge.push_back(vertex);
        } else {  // target
          if (hyperedge.size() > 1) {
            Hypergraph::add_hyperedge(hyperedge);
          }
          vertex = command.two_q_gate_count + n_qubits;
          add_gate_vertex(vertex, ci);
          hyperedge = {qubit_index, vertex};
          Hypergraph::add_hyperedge(hyperedge, 2);
          hyperedge = {qubit_index};
        }
        command.vertex = vertex;
        command.kind = "distributed gate";
      } else if (op_type == kOpRz) {
        command.kind = "1q local gate";
      } else {  // H
        if (hyperedge.size() > 1) {
          Hypergraph::add_hyperedge(hyperedge);
          hyperedge = {qubit_index};
        }
        command.kind = "1q local gate";
      }
    }

    if (hyperedge.size() > 1 || !two_qubit_gate_found) {
      Hypergraph::add_hyperedge(hyperedge);
    }
  }

  if (!weight_one_predicate()) {
    throw std::runtime_error(
        "weight_one_predicate failed (teleportation not supported)");
  }
}

inline std::vector<Command> HypergraphCircuit::get_hyperedge_subcircuit(
    const Hyperedge& hyperedge) const {
  const Vertex hyp_q_vertex = get_qubit_vertex(hyperedge);
  const std::string& hyp_qubit = qubit_of_vertex(hyp_q_vertex);
  std::vector<Vertex> gate_vertices = get_gate_vertices(hyperedge);
  std::sort(gate_vertices.begin(), gate_vertices.end());
  if (gate_vertices.empty()) return {};

  auto make_1q = [&](int op_type, double phase) {
    return Command{"", {hyp_qubit}, op_type, {phase}};
  };

  auto prepare_h_embedding = [&](const std::vector<Command>& embed_cmds) {
    std::vector<Command> prepared;
    std::vector<int> cu1_indices;
    for (int i = 0; i < static_cast<int>(embed_cmds.size()); ++i) {
      if (embed_cmds[i].op_type == kOpCU1) cu1_indices.push_back(i);
    }
    if (cu1_indices.empty()) {
      prepared = embed_cmds;
    } else {
      prepared.insert(prepared.end(), embed_cmds.begin(),
                      embed_cmds.begin() + cu1_indices[0]);
      double prev_phase = 0.0;
      if (prepared.back().op_type == kOpRz) {
        prev_phase = prepared.back().params[0];
        prepared.pop_back();
      }
      prepared.push_back(embed_cmds[cu1_indices[0]]);

      int prev_cu1_idx = cu1_indices[0];
      for (std::size_t k = 1; k < cu1_indices.size(); ++k) {
        const int next_cu1_idx = cu1_indices[k];
        std::vector<Op> current_1q_ops;
        for (int i = prev_cu1_idx + 1; i < next_cu1_idx; ++i) {
          current_1q_ops.push_back(
              Op{embed_cmds[i].op_type, embed_cmds[i].params});
        }
        std::vector<Op> new_ops = to_euler_with_two_hadamards(current_1q_ops);
        double squashed_phase = prev_phase + new_ops[0].params[0];
        const double mid_phase = new_ops[2].params[0];
        const double last_phase = new_ops[4].params[0];
        if (isclose(py_mod(mid_phase, 2.0), 0.0) ||
            isclose(py_mod(mid_phase, 2.0), 2.0)) {
          prev_phase = squashed_phase + mid_phase + last_phase;
          squashed_phase = 0.0;
        } else {
          prev_phase = last_phase;
        }
        new_ops[0] = Op::create(kOpRz, {squashed_phase});
        std::vector<Command> current_1q_cmds;
        for (const Op& op : new_ops) {
          current_1q_cmds.push_back(
              Command{"", {hyp_qubit}, op.op_type, op.params});
        }
        current_1q_cmds.pop_back();
        prepared.insert(prepared.end(), current_1q_cmds.begin(),
                        current_1q_cmds.end());
        prepared.push_back(embed_cmds[next_cu1_idx]);
        prev_cu1_idx = next_cu1_idx;
      }

      std::vector<Command> last_1q_cmds(
          embed_cmds.begin() + prev_cu1_idx + 1, embed_cmds.end());
      if (!last_1q_cmds.empty() && last_1q_cmds[0].op_type == kOpRz) {
        prev_phase += last_1q_cmds[0].params[0];
        last_1q_cmds.erase(last_1q_cmds.begin());
      }
      prepared.push_back(make_1q(kOpRz, prev_phase));
      prepared.insert(prepared.end(), last_1q_cmds.begin(), last_1q_cmds.end());
    }
    return prepared;
  };

  std::vector<Command> subcirc_commands;
  subcirc_commands.push_back(get_gate_of_vertex(gate_vertices[0]));
  Vertex prev_hyp_gate_vertex = gate_vertices[0];

  for (std::size_t gi = 1; gi < gate_vertices.size(); ++gi) {
    const Vertex next_hyp_gate_vertex = gate_vertices[gi];
    const std::vector<Command> embedded_commands =
        get_intermediate_commands(prev_hyp_gate_vertex, next_hyp_gate_vertex,
                                  hyp_q_vertex);

    std::vector<int> h_indices;
    for (int i = 0; i < static_cast<int>(embedded_commands.size()); ++i) {
      if (embedded_commands[i].op_type == kOpH) h_indices.push_back(i);
    }

    if (h_indices.empty()) {
      subcirc_commands.insert(subcirc_commands.end(), embedded_commands.begin(),
                              embedded_commands.end());
    } else {
      subcirc_commands.insert(subcirc_commands.end(), embedded_commands.begin(),
                              embedded_commands.begin() + h_indices[0]);
      const std::vector<Command> middle(
          embedded_commands.begin() + h_indices[0],
          embedded_commands.begin() + h_indices.back() + 1);
      const std::vector<Command> prepared = prepare_h_embedding(middle);
      subcirc_commands.insert(subcirc_commands.end(), prepared.begin(),
                              prepared.end());
      subcirc_commands.insert(subcirc_commands.end(),
                              embedded_commands.begin() + h_indices.back() + 1,
                              embedded_commands.end());
    }

    subcirc_commands.push_back(get_gate_of_vertex(next_hyp_gate_vertex));
    prev_hyp_gate_vertex = next_hyp_gate_vertex;
  }
  return subcirc_commands;
}

}  // namespace dqc
