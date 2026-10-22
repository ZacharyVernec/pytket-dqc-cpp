#pragma once

#include <algorithm>
#include <map>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

#include "dqc/distribution.hpp"
#include "dqc/graph.hpp"
#include "dqc/hypergraph_circuit.hpp"
#include "dqc/placement.hpp"
#include "dqc/py_set.hpp"

namespace dqc {

namespace detail {

// Look up the qubit list of a server in an insertion-ordered server_qubits
// (mirrors `network.server_qubits[server]`).
inline std::vector<int> network_qubit_list_of(
    const std::vector<std::pair<int, std::vector<int>>>& server_qubits,
    int server) {
  for (const auto& [s, q] : server_qubits) {
    if (s == server) return q;
  }
  return {};
}

inline bool network_has_server(
    const std::vector<std::pair<int, std::vector<int>>>& server_qubits,
    int server) {
  for (const auto& [s, q] : server_qubits) {
    (void)q;
    if (s == server) return true;
  }
  return false;
}

// `nx.is_connected` for a `Graph`: every node reachable from the first node,
// and the graph has at least one node. (An empty graph is not connected.)
inline bool graph_is_connected(const Graph& g) {
  if (g.nodes().empty()) return false;
  std::vector<bool> seen(g.nodes().size(), false);
  std::queue<int> q;
  const int start = g.nodes().front();
  const std::size_t start_pos =
      static_cast<std::size_t>(std::find(g.nodes().begin(), g.nodes().end(), start) -
                               g.nodes().begin());
  seen[start_pos] = true;
  q.push(start);
  std::size_t visited = 0;
  while (!q.empty()) {
    const int n = q.front();
    q.pop();
    ++visited;
    for (const Graph::Neighbor& nb : g.nbrs(n)) {
      const int m = nb.node;
      const auto it = std::find(g.nodes().begin(), g.nodes().end(), m);
      const std::size_t pos = static_cast<std::size_t>(it - g.nodes().begin());
      if (!seen[pos]) {
        seen[pos] = true;
        q.push(m);
      }
    }
  }
  return visited == g.nodes().size();
}

}  // namespace detail

// Mirrors `pytket_dqc.networks.server_network.ServerNetwork`: a container for
// `server_coupling` (an insertion-ordered list of server-index pairs) with
// connectivity validation and `get_server_list()`. Only the data + validation
// are ported; networkx drawing and pytket `Architecture`/`Routing` helpers stay
// in Python (the core has no networkx/pytket dependency).
class ServerNetwork {
 public:
  explicit ServerNetwork(const std::vector<std::pair<int, int>>& server_coupling)
      : server_coupling_(server_coupling) {
    // `ServerNetwork.__init__` validates each element is a pair
    // (`len(server_list) == 2`) and that the coupling graph is connected.
    if (!detail::graph_is_connected(get_server_nx())) {
      throw std::runtime_error("This server network is unconnected.");
    }
  }

  const std::vector<std::pair<int, int>>& server_coupling() const {
    return server_coupling_;
  }

  // Mirrors `ServerNetwork.get_server_list()`: CPython int-set iteration order
  // of the server ids appearing in the coupling edges.
  std::vector<int> get_server_list() const {
    PySet<int> s(&py_hash_key, &py_eq_key);
    for (const auto& [a, b] : server_coupling_) {
      s.add(a);
      s.add(b);
    }
    return s.iteration_order();
  }

  // Mirrors `ServerNetwork.get_server_nx()`: graph with one edge per coupling
  // pair.
  Graph get_server_nx() const {
    Graph g;
    for (const auto& [u, v] : server_coupling_) g.add_edge(u, v);
    return g;
  }

  // Mirrors `ServerNetwork.is_placement`: every server used by the placement
  // must be in the network.
  bool is_placement(const Placement& placement) const {
    const std::vector<int> server_list = get_server_list();
    for (const auto& [v, s] : placement.entries) {
      (void)v;
      if (std::find(server_list.begin(), server_list.end(), s) ==
          server_list.end()) {
        return false;
      }
    }
    return true;
  }

  bool operator==(const ServerNetwork& other) const {
    return server_coupling_ == other.server_coupling_;
  }
  bool operator!=(const ServerNetwork& other) const {
    return !(*this == other);
  }

 private:
  std::vector<std::pair<int, int>> server_coupling_;
};

// Mirrors `pytket_dqc.networks.nisq_network.NISQNetwork`: adds per-server qubit
// lists and communication-memory capacities (with validation) on top of the
// server coupling. `server_qubits` is an insertion-ordered list of
// (server, qubit-list) pairs; `server_ebit_mem` maps server id -> capacity
// (empty means unbounded, mirrored as `-1`).
class NISQNetwork : public ServerNetwork {
 public:
  NISQNetwork(
      const std::vector<std::pair<int, int>>& server_coupling,
      const std::vector<std::pair<int, std::vector<int>>>& server_qubits,
      const std::map<int, int>& server_ebit_mem = {})
      : ServerNetwork(server_coupling),
        server_qubits_(server_qubits),
        server_ebit_mem_(server_ebit_mem.empty()
                             ? default_unbounded(server_qubits)
                             : server_ebit_mem) {
    const std::vector<int> server_list = get_server_list();
    for (const int server : server_list) {
      if (!detail::network_has_server(server_qubits_, server)) {
        throw std::runtime_error("The qubits in server " + std::to_string(server) +
                                 " have not been specified.");
      }
      if (server_ebit_mem_.count(server) == 0) {
        throw std::runtime_error(
            "The communication memory capacity of server " +
            std::to_string(server) + " has not been specified.");
      }
    }
    if (server_qubits_.size() > server_list.size()) {
      throw std::runtime_error(
          "The network defined does not cover all servers with qubits "
          "allocated to them.");
    }
    if (server_ebit_mem_.size() > server_list.size()) {
      throw std::runtime_error(
          "The network defined does not cover all servers that have a bound "
          "to communication memory assigned to them.");
    }
    const std::vector<int> qubit_list = get_qubit_list();
    std::set<int> unique(qubit_list.begin(), qubit_list.end());
    if (unique.size() != qubit_list.size()) {
      throw std::runtime_error(
          "Qubits may belong to only one server, and should feature only "
          "once per server.");
    }
    // The reference asserts connectivity of the full NISQ graph (qubits +
    // servers). Replicate the check; an invalid network raises here.
    if (!detail::graph_is_connected(get_nisq_nx())) {
      throw std::runtime_error("The NISQ network is unconnected.");
    }
  }

  const std::vector<std::pair<int, std::vector<int>>>& server_qubits() const {
    return server_qubits_;
  }

  const std::map<int, int>& server_ebit_mem() const { return server_ebit_mem_; }

  // Mirrors `NISQNetwork.can_implement`.
  bool can_implement(const HypergraphCircuit& dist_circ) const {
    return static_cast<int>(get_qubit_list().size()) >=
           static_cast<int>(dist_circ.get_qubit_vertices().size());
  }

  // Mirrors `NISQNetwork.get_qubit_list`: qubits of each server concatenated
  // in server_qubits insertion order.
  std::vector<int> get_qubit_list() const {
    std::vector<int> out;
    for (const auto& [s, qs] : server_qubits_) {
      (void)s;
      out.insert(out.end(), qs.begin(), qs.end());
    }
    return out;
  }

  // Mirrors `NISQNetwork.get_nisq_nx()`: intra-server qubit-qubit edges
  // (weight 0) and one inter-server edge per coupling pair (weight 1, between
  // the first qubit of each server). The `color` attribute is cosmetic
  // (drawing only) and is not carried over.
  Graph get_nisq_nx() const {
    Graph g;
    for (const auto& [s, qs] : server_qubits_) {
      (void)s;
      for (std::size_t i = 0; i < qs.size(); ++i) {
        for (std::size_t j = i + 1; j < qs.size(); ++j) {
          g.add_edge(qs[i], qs[j], {{"weight", 0.0}});
        }
      }
    }
    for (const auto& [u, v] : server_coupling()) {
      const std::vector<int> qu = detail::network_qubit_list_of(server_qubits_, u);
      const std::vector<int> qv = detail::network_qubit_list_of(server_qubits_, v);
      if (!qu.empty() && !qv.empty()) {
        g.add_edge(qu[0], qv[0], {{"weight", 1.0}});
      }
    }
    return g;
  }

  // Mirrors `NISQNetwork.to_dict`.
  std::map<std::string, std::vector<std::pair<int, int>>> to_dict() const {
    // server_qubits / server_ebit_mem are exposed separately; the coupling is
    // the only part that fits the flat map. Kept for API parity.
    return {{"server_coupling", server_coupling()}};
  }

  bool operator==(const NISQNetwork& other) const {
    return server_qubits_ == other.server_qubits_ &&
           server_ebit_mem_ == other.server_ebit_mem_ &&
           ServerNetwork::operator==(other);
  }
  bool operator!=(const NISQNetwork& other) const { return !(*this == other); }

 private:
  static std::map<int, int> default_unbounded(
      const std::vector<std::pair<int, std::vector<int>>>& server_qubits) {
    std::map<int, int> out;
    for (const auto& [s, qs] : server_qubits) {
      (void)qs;
      out[s] = -1;
    }
    return out;
  }

  std::vector<std::pair<int, std::vector<int>>> server_qubits_;
  std::map<int, int> server_ebit_mem_;
};

// Mirrors `pytket_dqc.circuits.distribution.Distribution(circ, placement,
// network)`: builds a Distribution from a placement and a `NISQNetwork`
// (server graph = `get_server_nx()`, qubits + ebit memory from the network).
inline Distribution make_distribution(const HypergraphCircuit& circuit,
                                      const Placement& placement,
                                      const NISQNetwork& network) {
  return make_distribution(circuit, placement, network.get_server_nx(),
                           network.server_qubits(), network.server_ebit_mem());
}

}  // namespace dqc
