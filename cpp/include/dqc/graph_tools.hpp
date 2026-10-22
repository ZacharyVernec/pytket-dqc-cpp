#pragma once

#include <algorithm>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#include "dqc/graph.hpp"

namespace dqc {

// Faithful port of `pytket_dqc.utils.graph_tools.steiner_tree` (networkx 3.6.1
// `steiner_tree` with the default "mehlhorn" method, plus the singleton
// special case).
inline Graph steiner_tree(const Graph& graph, const std::vector<int>& nodes) {
  if (nodes.empty()) {
    throw std::runtime_error("No nodes have been provided");
  }

  const std::set<int> unique(nodes.begin(), nodes.end());
  if (unique.size() == 1) {
    Graph tree;
    tree.add_nodes_from(nodes);
    return tree;
  }

  // Mehlhorn 1988.
  const auto [distances, paths] = dijkstra_multisource(graph, nodes);

  std::map<int, int> s;
  std::map<std::pair<int, int>, double> d1;
  for (const int v : graph.nodes()) {
    s[v] = paths.at(v)[0];
    d1[{v, s[v]}] = distances.at(v);
  }

  Graph g1_prime;
  for (const Edge& e : graph.edges()) {
    const int su = s.at(e.u);
    const int sv = s.at(e.v);
    const double weight_here =
        d1.at({e.u, su}) + e.weight("weight") + d1.at({e.v, sv});
    if (!g1_prime.has_edge(su, sv)) {
      g1_prime.add_edge(su, sv, {{"weight_d1", weight_here}});
    } else {
      const double existing = g1_prime.weight(su, sv, "weight_d1");
      g1_prime.add_edge(su, sv, {{"weight_d1", std::min(weight_here, existing)}});
    }
  }

  const std::vector<Edge> g2 = minimum_spanning_edges(g1_prime, "weight_d1");

  Graph g3;
  for (const Edge& mst_edge : g2) {
    const std::vector<int> path =
        bidirectional_dijkstra_path(graph, mst_edge.u, mst_edge.v);
    for (std::size_t i = 0; i + 1 < path.size(); ++i) {
      const int n1 = path[i];
      const int n2 = path[i + 1];
      g3.add_edge(n1, n2, {{"weight", graph.weight(n1, n2)}});
    }
  }

  const std::vector<Edge> g3_mst = minimum_spanning_edges(g3, "weight");
  std::vector<std::pair<int, int>> g3_mst_pairs;
  for (const Edge& e : g3_mst) g3_mst_pairs.push_back({e.u, e.v});

  Graph g4 = graph.edge_subgraph(g3_mst_pairs);
  remove_nonterminal_leaves(g4, nodes);

  std::vector<std::pair<int, int>> final_edges;
  for (const Edge& e : g4.edges()) final_edges.push_back({e.u, e.v});

  // `steiner_tree` returns `G.edge_subgraph(edges)` of the ORIGINAL graph.
  return graph.edge_subgraph(final_edges);
}

// Faithful port of `pytket_dqc.utils.graph_tools.direct_from_origin`.
inline std::vector<std::pair<int, int>> direct_from_origin(const Graph& G,
                                                           int origin) {
  if (G.nodes().empty()) {
    return {};
  }
  if (!G.is_tree()) {
    throw std::runtime_error("The graph to redirect must be a tree.");
  }
  if (!G.is_connected()) {
    throw std::runtime_error("The graph is not connected.");
  }

  Graph g_reduced = G;
  g_reduced.remove_node(origin);

  std::vector<int> nbrs;
  for (const auto& nb : G.nbrs(origin)) nbrs.push_back(nb.node);
  std::sort(nbrs.begin(), nbrs.end());

  std::vector<std::pair<int, int>> edge_list;
  for (const int n : nbrs) edge_list.push_back({origin, n});

  for (const int n : nbrs) {
    const std::set<int> c = g_reduced.node_connected_component(n);
    if (c.size() != 1) {
      const Graph sub = g_reduced.subgraph(c);
      const auto rest = direct_from_origin(sub, n);
      edge_list.insert(edge_list.end(), rest.begin(), rest.end());
    }
  }
  return edge_list;
}

}  // namespace dqc
