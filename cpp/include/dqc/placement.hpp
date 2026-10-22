#pragma once

#include <algorithm>
#include <map>
#include <utility>
#include <vector>

#include "dqc/graph.hpp"
#include "dqc/graph_tools.hpp"
#include "dqc/types.hpp"

namespace dqc {

// Mirrors `pytket_dqc.placement.Placement`: a mapping from hypergraph vertices
// to server ids. Insertion order is preserved (CPython dict semantics), which
// matters for `get_distribution_tree`/`get_vertices_in` (they iterate
// `placement.items()` in insertion order).
struct Placement {
  std::vector<std::pair<int, int>> entries;  // insertion order
  std::map<Vertex, int> placement;           // vertex -> server lookup

  bool operator==(const Placement& o) const { return placement == o.placement; }
  bool operator!=(const Placement& o) const { return !(*this == o); }

  bool contains(Vertex v) const { return placement.count(v) != 0; }
  int at(Vertex v) const { return placement.at(v); }

  void set(Vertex v, int server) {
    auto it = placement.find(v);
    if (it == placement.end()) {
      placement[v] = server;
      entries.push_back({v, server});
    } else {
      it->second = server;
      for (auto& e : entries) {
        if (e.first == v) e.second = server;
      }
    }
  }

  // Mirrors `get_vertices_in`.
  std::vector<Vertex> get_vertices_in(int server) const {
    std::vector<Vertex> out;
    for (const auto& [v, s] : entries) {
      if (s == server) out.push_back(v);
    }
    return out;
  }
};

// Mirrors `Placement.get_distribution_tree(hyperedge, qubit_node, network)`.
// `server_graph` is the network's server-coupling graph (`get_server_nx()`).
inline std::vector<std::pair<int, int>> get_distribution_tree(
    const Placement& placement, const std::vector<int>& hyperedge,
    int qubit_node, const Graph& server_graph) {
  std::vector<int> servers_used;
  for (const auto& [v, s] : placement.entries) {
    if (std::find(hyperedge.begin(), hyperedge.end(), v) != hyperedge.end()) {
      servers_used.push_back(s);
    }
  }
  const Graph steiner_server_graph = steiner_tree(server_graph, servers_used);
  const int qubit_server = placement.at(qubit_node);
  return direct_from_origin(steiner_server_graph, qubit_server);
}

}  // namespace dqc
