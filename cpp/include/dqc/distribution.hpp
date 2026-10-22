#pragma once

#include <map>
#include <set>
#include <stdexcept>
#include <vector>

#include "dqc/graph.hpp"
#include "dqc/graph_tools.hpp"
#include "dqc/hypergraph_circuit.hpp"
#include "dqc/numeric.hpp"
#include "dqc/op_type.hpp"
#include "dqc/placement.hpp"
#include "dqc/py_set.hpp"

namespace dqc {

// Mirrors `Distribution.hyperedge_cost(hyperedge, server_tree=..., \
// requires_h_embedded_cu1=...)` with both optional arguments supplied
// (used by GainManager).
inline int hyperedge_cost_with_tree(const HypergraphCircuit& dist_circ,
                                    const Hyperedge& hyperedge,
                                    const Placement& placement,
                                    const Graph& tree,
                                    bool requires_h) {
  if (hyperedge.weight != 1) {
    throw std::runtime_error(
        "Hyperedges with weight other than 1 are not currently supported");
  }

  const Vertex shared_qubit = dist_circ.get_qubit_vertex(hyperedge);
  const int home_server = placement.at(shared_qubit);

  if (!requires_h) {
    return static_cast<int>(tree.edges().size());
  }

  const std::vector<Command> commands =
      dist_circ.get_hyperedge_subcircuit(hyperedge);
  std::vector<Vertex> vertices = hyperedge.vertices;
  std::sort(vertices.begin(), vertices.end());
  vertices.erase(vertices.begin());  // pop the shared qubit

  int cost = 0;
  bool currently_h_embedding = false;
  PySet<int> connected_servers(&py_hash_key, &py_eq_key);
  connected_servers.add(home_server);

  for (const Command& command : commands) {
    if (command.op_type == kOpH) {
      currently_h_embedding = !currently_h_embedding;
    } else if (command.op_type == kOpRz) {
      const double p = py_mod(command.params[0], 1.0);
      (void)p;  // assert (!currently_h_embedding or isclose(p,0) or isclose(p,1))
    } else if (command.op_type == kOpCU1) {
      if (currently_h_embedding) {
        // assert isclose(py_mod(param, 2), 1)
        std::vector<Vertex> q_vertices;
        for (const std::string& q : command.qubits) {
          q_vertices.push_back(dist_circ.get_vertex_of_qubit(q));
        }
        const Vertex remote_vertex =
            (q_vertices[0] != shared_qubit) ? q_vertices[0] : q_vertices[1];
        const int remote_server = placement.at(remote_vertex);
        PySet<int> new_connected(&py_hash_key, &py_eq_key);
        for (const int s : connected_servers.iteration_order()) {
          if (s == home_server || s == remote_server) new_connected.add(s);
        }
        connected_servers = std::move(new_connected);
        // assert home_server != remote_server
      } else if (!(command == dist_circ.get_gate_of_vertex(vertices[0]))) {
        // Nothing to do.
      } else {
        const Vertex gate_vertex = vertices[0];
        vertices.erase(vertices.begin());
        const int gate_server = placement.at(gate_vertex);
        if (!connected_servers.contains(gate_server)) {
          std::vector<int> best_path;
          for (const int c_server : connected_servers.iteration_order()) {
            const std::vector<int> path =
                bidirectional_dijkstra_path(tree, c_server, gate_server);
            if (best_path.empty() || path.size() < best_path.size()) {
              best_path = path;
            }
          }
          for (const int n : best_path) connected_servers.add(n);
          cost += static_cast<int>(best_path.size()) - 1;
        }
      }
    }
  }
  return cost;
}

// Mirrors `Distribution.hyperedge_cost`. `server_graph` is the network's
// server-coupling graph (`network.get_server_nx()`).
//
// NOTE (fidelity): `connected_servers` is a CPython `set` in the reference.
// Its iteration order is replicated only approximately here (std::set
// ascending) rather than with CPython set order; see the README "Known
// approximations". This matters only for shortest-path tie-breaking inside
// the ALAP loop.
inline int hyperedge_cost(const HypergraphCircuit& dist_circ,
                          const Hyperedge& hyperedge,
                          const Placement& placement,
                          const Graph& server_graph) {
  if (hyperedge.weight != 1) {
    throw std::runtime_error(
        "Hyperedges with weight other than 1 are not currently supported");
  }
  std::vector<int> servers;
  for (const Vertex v : hyperedge.vertices) {
    servers.push_back(placement.at(v));
  }
  const Graph tree = steiner_tree(server_graph, servers);
  const bool requires_h = dist_circ.requires_h_embedded_cu1(hyperedge);
  return hyperedge_cost_with_tree(dist_circ, hyperedge, placement, tree,
                                  requires_h);
}

// Mirrors `Distribution.cost`.
inline int distribution_cost(const HypergraphCircuit& dist_circ,
                             const Placement& placement,
                             const Graph& server_graph) {
  int cost = 0;
  for (const Hyperedge& hyperedge : dist_circ.hyperedge_list()) {
    cost += hyperedge_cost(dist_circ, hyperedge, placement, server_graph);
  }
  return cost;
}

// Mirrors `pytket_dqc.circuits.distribution.Distribution`: a hypergraph
// circuit, its placement, and the network's server-coupling graph
// (`network.get_server_nx()`). `server_qubits` is `network.server_qubits`
// (server id -> list of qubits), needed for occupancy/capacity checks.
struct Distribution {
  HypergraphCircuit circuit;
  Placement placement;
  Graph server_graph;
  std::map<int, std::vector<int>> server_qubits;
  // Mirrors `NISQNetwork.server_ebit_mem` (server id -> communication
  // capacity). `-1` means unbounded (the `NISQNetwork` default). Populated by
  // the bindings; consumed by `to_pytket_circuit` for `satisfy_bound`.
  std::map<int, int> server_ebit_mem;

  // Mirrors `ServerNetwork.get_server_list()`. Order is irrelevant to
  // `to_pytket_circuit` (only per-server lists are iterated there), so the
  // ascending `std::map` key order of `server_qubits` is used.
  std::vector<int> get_server_list() const {
    std::vector<int> servers;
    for (const auto& [s, qs] : server_qubits) {
      (void)qs;
      servers.push_back(s);
    }
    return servers;
  }

  int cost() const { return distribution_cost(circuit, placement, server_graph); }

  // Mirrors `Distribution.is_valid`.
  bool is_valid() const {
    if (!circuit.is_placement(placement)) return false;
    for (const auto& [v, s] : placement.entries) {
      (void)v;
      if (server_qubits.count(s) == 0) return false;
    }
    for (const auto& [server, qubits] : server_qubits) {
      int count = 0;
      for (const auto& [v, s] : placement.entries) {
        if (s == server && circuit.is_qubit_vertex(v)) ++count;
      }
      if (count > static_cast<int>(qubits.size())) return false;
    }
    return true;
  }

  // Mirrors `Distribution.non_local_gate_list`.
  std::vector<Vertex> non_local_gate_list() const {
    std::vector<Vertex> non_local;
    for (const Vertex vertex : circuit.vertex_list()) {
      if (circuit.is_qubit_vertex(vertex)) continue;
      const Command gate = circuit.get_gate_of_vertex(vertex);
      const Vertex v_1 = circuit.get_vertex_of_qubit(gate.qubits[0]);
      const Vertex v_2 = circuit.get_vertex_of_qubit(gate.qubits[1]);
      const int s_1 = placement.at(v_1);
      const int s_2 = placement.at(v_2);
      const int s_g = placement.at(vertex);
      if (!((s_1 == s_g) && (s_2 == s_g))) non_local.push_back(vertex);
    }
    return non_local;
  }

  // Mirrors `Distribution.non_local_gate_count`.
  int non_local_gate_count() const {
    return static_cast<int>(non_local_gate_list().size());
  }

  // Mirrors `Distribution.detached_gate_list`.
  std::vector<Vertex> detached_gate_list() const {
    std::vector<Vertex> detached;
    for (const Vertex vertex : circuit.vertex_list()) {
      if (circuit.is_qubit_vertex(vertex)) continue;
      const Command gate = circuit.get_gate_of_vertex(vertex);
      const Vertex v_1 = circuit.get_vertex_of_qubit(gate.qubits[0]);
      const Vertex v_2 = circuit.get_vertex_of_qubit(gate.qubits[1]);
      const int s_1 = placement.at(v_1);
      const int s_2 = placement.at(v_2);
      const int s_g = placement.at(vertex);
      if (!((s_1 == s_g) || (s_2 == s_g))) detached.push_back(vertex);
    }
    return detached;
  }

  // Mirrors `Distribution.detached_gate_count`.
  int detached_gate_count() const {
    return static_cast<int>(detached_gate_list().size());
  }
};

// Builds a `Distribution` from a placement and the network descriptors the
// allocators receive. `server_qubits` is the insertion-ordered list of
// (server, qubits) pairs; an empty `server_ebit_mem` means unbounded (`-1`,
// the `NISQNetwork` default). Mirrors the reference's
// `Distribution(dist_circ, placement, network)` construction.
inline Distribution make_distribution(
    const HypergraphCircuit& circuit, const Placement& placement,
    const Graph& server_graph,
    const std::vector<std::pair<int, std::vector<int>>>& server_qubits,
    const std::map<int, int>& server_ebit_mem) {
  Distribution d{circuit, placement, server_graph};
  for (const auto& [s, qs] : server_qubits) d.server_qubits[s] = qs;
  if (server_ebit_mem.empty()) {
    for (const auto& [s, qs] : server_qubits) {
      (void)qs;
      d.server_ebit_mem[s] = -1;
    }
  } else {
    d.server_ebit_mem = server_ebit_mem;
  }
  return d;
}

}  // namespace dqc
