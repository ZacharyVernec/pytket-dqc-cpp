#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "dqc/py_set.hpp"

namespace dqc {

// An undirected edge with named numeric attributes. Mirrors the edge data
// dictionary of networkx's `nx.Graph`.
struct Edge {
  int u = 0;
  int v = 0;
  std::map<std::string, double> attrs;

  double weight(const std::string& key, double default_value = 1.0) const {
    auto it = attrs.find(key);
    return it == attrs.end() ? default_value : it->second;
  }
};

// A minimal faithful port of `networkx.Graph` (undirected, insertion-ordered
// adjacency). Node iteration order and neighbour iteration order follow
// insertion order exactly, matching CPython's dict insertion semantics, which
// networkx relies on for deterministic algorithm tie-breaking.
//
// Weights are kept as named edge attributes; `weight("weight")` defaults to 1
// (matching `data.get("weight", 1)` in networkx).
class Graph {
 public:
  struct Neighbor {
    int node = 0;
    std::map<std::string, double> attrs;
  };

  Graph() = default;

  const std::vector<int>& nodes() const { return nodes_; }
  std::size_t num_nodes() const { return nodes_.size(); }

  bool has_node(int n) const { return pos_.count(n) != 0; }
  bool has_edge(int u, int v) const {
    auto iu = pos_.find(u);
    return iu != pos_.end() && find_nbr(iu->second, v) != nullptr;
  }

  void add_node(int n) {
    if (pos_.count(n)) return;
    pos_[n] = nodes_.size();
    nodes_.push_back(n);
    adj_.push_back({});
  }

  void add_nodes_from(const std::vector<int>& ns) {
    for (const int n : ns) add_node(n);
  }

  // Mirrors `Graph.add_edge(u, v, **attr)`: updates attrs if the edge exists.
  void add_edge(int u, int v, const std::map<std::string, double>& attrs = {}) {
    add_node(u);
    add_node(v);
    if (u == v) {
      std::size_t i = pos_.at(u);
      Neighbor* e = find_nbr(i, u);
      if (e == nullptr) {
        adj_[i].push_back({u, attrs});
      } else {
        merge_attrs(e->attrs, attrs);
      }
      return;
    }
    std::size_t iu = pos_.at(u);
    std::size_t iv = pos_.at(v);
    Neighbor* e1 = find_nbr(iu, v);
    Neighbor* e2 = find_nbr(iv, u);
    if (e1 != nullptr && e2 != nullptr) {
      merge_attrs(e1->attrs, attrs);
      merge_attrs(e2->attrs, attrs);
      return;
    }
    adj_[iu].push_back({v, attrs});
    adj_[iv].push_back({u, attrs});
  }

  // Neighbours of `n` in insertion order (self-loops appear once).
  const std::vector<Neighbor>& nbrs(int n) const { return adj_.at(pos_.at(n)); }

  // Mirrors `Graph.degree`: a self-loop is counted twice (networkx).
  std::size_t degree(int n) const {
    std::size_t deg = nbrs(n).size();
    if (find_nbr(pos_.at(n), n) != nullptr) ++deg;
    return deg;
  }

  // Edge attributes of edge (u, v), or an empty map if absent.
  std::map<std::string, double> edge_attrs(int u, int v) const {
    auto iu = pos_.find(u);
    if (iu == pos_.end()) return {};
    const Neighbor* e = find_nbr(iu->second, v);
    return e == nullptr ? std::map<std::string, double>{} : e->attrs;
  }

  double weight(int u, int v, const std::string& key = "weight",
                double default_value = 1.0) const {
    const std::map<std::string, double> attrs = edge_attrs(u, v);
    auto it = attrs.find(key);
    return it == attrs.end() ? default_value : it->second;
  }

  // Iterates edges exactly like networkx `Graph.edges(data=True)`.
  std::vector<Edge> edges() const {
    std::vector<Edge> out;
    std::map<int, bool> seen;
    for (std::size_t i = 0; i < nodes_.size(); ++i) {
      const int n = nodes_[i];
      for (const Neighbor& nb : adj_[i]) {
        if (!seen.count(nb.node)) {
          out.push_back(Edge{n, nb.node, nb.attrs});
        }
      }
      seen[n] = true;
    }
    return out;
  }

  void remove_node(int n) {
    auto it = pos_.find(n);
    if (it == pos_.end()) return;
    const std::size_t idx = it->second;
    nodes_.erase(nodes_.begin() + static_cast<std::ptrdiff_t>(idx));
    adj_.erase(adj_.begin() + static_cast<std::ptrdiff_t>(idx));
    pos_.erase(it);
    for (auto& nbrs : adj_) {
      nbrs.erase(std::remove_if(nbrs.begin(), nbrs.end(),
                                [&](const Neighbor& nb) { return nb.node == n; }),
                 nbrs.end());
    }
    rebuild_pos();
  }

  void remove_nodes(const std::set<int>& ns) {
    for (const int n : ns) remove_node(n);
  }

  // Mirrors `G.subgraph(nodes)`: node order = parent order filtered by `nodes`,
  // edge order = parent edge order filtered to pairs inside `nodes`.
  Graph subgraph(const std::set<int>& c) const {
    Graph sub;
    for (const int n : nodes_) {
      if (c.count(n)) sub.add_node(n);
    }
    for (const Edge& e : edges()) {
      if (c.count(e.u) && c.count(e.v)) sub.add_edge(e.u, e.v, e.attrs);
    }
    return sub;
  }

  // Mirrors `nx.edge_subgraph(G, edges)`. networkx converts `edges` to a set
  // (so membership is orientation-insensitive via `show_edges`) and builds the
  // node list through a `subgraph_view` whose node iteration uses CPython set
  // order when `2*|nodes| < |G|` (FilterAtlas fast path). Both behaviours are
  // replicated for structural exactness.
  Graph edge_subgraph(const std::vector<std::pair<int, int>>& es) const {
    PySet<std::pair<int, int>> edges_set(&py_hash_key, &py_eq_key);
    for (const auto& [u, v] : es) {
      edges_set.add({u, v});
    }

    // Build the node set by iterating the edge SET (CPython set order).
    PySet<int> nodes_set(&py_hash_key, &py_eq_key);
    for (const auto& [u, v] : edges_set.iteration_order()) {
      nodes_set.add(u);
      nodes_set.add(v);
    }

    // Node list for the view (FilterAtlas fast path).
    std::vector<int> node_list;
    if (2 * nodes_set.size() < nodes_.size()) {
      node_list = nodes_set.iteration_order();
    } else {
      for (const int n : nodes_) {
        if (nodes_set.contains(n)) node_list.push_back(n);
      }
    }

    // Edge membership, both orientations (networkx `show_edges`).
    std::set<std::pair<int, int>> alledges;
    for (const auto& [u, v] : es) {
      alledges.insert({u, v});
      alledges.insert({v, u});
    }

    Graph sub;
    for (const int n : node_list) sub.add_node(n);

    std::set<int> seen;
    for (const int n : node_list) {
      for (const Neighbor& nb : nbrs(n)) {
        const int v = nb.node;
        if (!nodes_set.contains(v)) continue;  // node filter
        if (!alledges.count({n, v})) continue;  // edge filter
        if (!seen.count(v)) {
          sub.add_edge(n, v, nb.attrs);
        }
      }
      seen.insert(n);
    }
    return sub;
  }

  bool is_connected() const {
    if (nodes_.empty()) return true;
    std::set<int> visited;
    std::queue<int> q;
    q.push(nodes_[0]);
    visited.insert(nodes_[0]);
    while (!q.empty()) {
      const int v = q.front();
      q.pop();
      for (const Neighbor& nb : nbrs(v)) {
        if (!visited.count(nb.node)) {
          visited.insert(nb.node);
          q.push(nb.node);
        }
      }
    }
    return visited.size() == nodes_.size();
  }

  bool is_tree() const {
    return is_connected() && edges().size() == nodes_.size() - 1;
  }

  // Mirrors `nx.node_connected_component(G, n)`: BFS from `n`.
  std::set<int> node_connected_component(int n) const {
    std::set<int> visited;
    std::queue<int> q;
    q.push(n);
    visited.insert(n);
    while (!q.empty()) {
      const int v = q.front();
      q.pop();
      for (const Neighbor& nb : nbrs(v)) {
        if (!visited.count(nb.node)) {
          visited.insert(nb.node);
          q.push(nb.node);
        }
      }
    }
    return visited;
  }

 private:
  static void merge_attrs(std::map<std::string, double>& target,
                          const std::map<std::string, double>& src) {
    for (const auto& [k, v] : src) target[k] = v;
  }

  Neighbor* find_nbr(std::size_t idx, int node) {
    for (auto& nb : adj_[idx]) {
      if (nb.node == node) return &nb;
    }
    return nullptr;
  }
  const Neighbor* find_nbr(std::size_t idx, int node) const {
    for (const auto& nb : adj_[idx]) {
      if (nb.node == node) return &nb;
    }
    return nullptr;
  }

  void rebuild_pos() {
    pos_.clear();
    for (std::size_t i = 0; i < nodes_.size(); ++i) pos_[nodes_[i]] = i;
  }

  std::vector<int> nodes_;
  std::vector<std::vector<Neighbor>> adj_;
  std::unordered_map<int, std::size_t> pos_;
};

// Result of Dijkstra: distances keyed by node, and paths from the closest
// source to each node.
struct DijkstraResult {
  std::map<int, double> dist;
  std::map<int, std::vector<int>> paths;
};

// Faithful port of networkx `_dijkstra_multisource` (with `paths` provided,
// `target` = none). Sources are an ordered, possibly duplicated list.
inline DijkstraResult dijkstra_multisource(const Graph& G,
                                           const std::vector<int>& sources) {
  // Min-heap of (distance, count, node).
  using HeapEntry = std::tuple<double, std::uint64_t, int>;
  std::priority_queue<HeapEntry, std::vector<HeapEntry>, std::greater<>> fringe;

  std::uint64_t c = 0;
  std::map<int, double> seen;
  std::vector<int> unique_sources;
  std::set<int> source_set;
  for (const int s : sources) {
    seen[s] = 0.0;
    if (!source_set.count(s)) {
      source_set.insert(s);
      unique_sources.push_back(s);
    }
    fringe.push({0.0, c++, s});
  }
  const std::size_t number_of_sources = unique_sources.size();

  std::map<int, std::vector<int>> paths;
  for (const int s : unique_sources) paths[s] = {s};

  std::map<int, int> pred;  // first predecessor per node (pred_dict[v][0])
  std::map<int, double> dist;
  std::vector<int> pop_order;

  while (!fringe.empty()) {
    auto [dist_v, cnt, v] = fringe.top();
    (void)cnt;
    fringe.pop();
    if (dist.count(v)) continue;  // already finalized
    dist[v] = dist_v;
    pop_order.push_back(v);
    for (const auto& nb : G.nbrs(v)) {
      const int u = nb.node;
      const double cost = nb.attrs.count("weight") ? nb.attrs.at("weight") : 1.0;
      const double vu_dist = dist_v + cost;
      if (dist.count(u)) {
        // already finalized; no negative weights, pred is None -> no-op
      } else if (!seen.count(u) || vu_dist < seen[u]) {
        seen[u] = vu_dist;
        fringe.push({vu_dist, c++, u});
        pred[u] = v;
      }
      // pred != None case is never triggered (no predecessor dict requested).
    }
  }

  // Reconstruct paths in pop order (mirrors the `islice(dist, ...)` loop).
  for (std::size_t i = number_of_sources; i < pop_order.size(); ++i) {
    const int v = pop_order[i];
    const int p = pred.at(v);
    paths[v] = paths.at(p);
    paths[v].push_back(v);
  }

  return {dist, paths};
}

// Faithful port of networkx `bidirectional_dijkstra` (which is what
// `nx.shortest_path(G, u, v, weight)` uses for a single source/target pair).
// Returns the path list.
inline std::vector<int> bidirectional_dijkstra_path(const Graph& G, int source,
                                                    int target) {
  constexpr int kNone = -1;  // Python None sentinel for predecessors.

  auto build_path = [&](const std::map<int, int>& preds, int curr,
                        int direction) {
    std::vector<int> ret;
    if (curr == kNone) return ret;
    while (true) {
      ret.push_back(curr);
      auto it = preds.find(curr);
      if (it == preds.end() || it->second == kNone) break;
      curr = it->second;
    }
    if (direction == 0) std::reverse(ret.begin(), ret.end());
    return ret;
  };

  if (source == target) return {source};

  using HeapEntry = std::tuple<double, std::uint64_t, int>;
  std::priority_queue<HeapEntry, std::vector<HeapEntry>, std::greater<>> fringe[2];

  std::map<int, double> dists[2];
  std::map<int, int> preds[2];
  preds[0][source] = kNone;
  preds[1][target] = kNone;

  std::map<int, double> seen[2];
  seen[0][source] = 0.0;
  seen[1][target] = 0.0;

  std::uint64_t c = 0;
  fringe[0].push({0.0, c++, source});
  fringe[1].push({0.0, c++, target});

  bool has_finaldist = false;
  double finaldist = 0.0;
  int meetnode = kNone;

  int direction = 1;
  while (!fringe[0].empty() && !fringe[1].empty()) {
    direction = 1 - direction;
    auto [dist, cnt, v] = fringe[direction].top();
    (void)cnt;
    fringe[direction].pop();
    if (dists[direction].count(v)) continue;
    dists[direction][v] = dist;

    if (dists[1 - direction].count(v)) {
      int m_pred = kNone;
      if (preds[1].count(meetnode)) m_pred = preds[1].at(meetnode);
      std::vector<int> p0 = build_path(preds[0], meetnode, 0);
      std::vector<int> p1 = build_path(preds[1], m_pred, 1);
      p0.insert(p0.end(), p1.begin(), p1.end());
      return p0;
    }

    for (const auto& nb : G.nbrs(v)) {
      const int w = nb.node;
      const double cost =
          nb.attrs.count("weight") ? nb.attrs.at("weight") : 1.0;
      const double vw = dist + cost;
      if (dists[direction].count(w)) {
        if (vw < dists[direction].at(w)) {
          throw std::runtime_error(
              "Contradictory paths found: negative weights?");
        }
      } else if (!seen[direction].count(w) || vw < seen[direction].at(w)) {
        seen[direction][w] = vw;
        fringe[direction].push({vw, c++, w});
        preds[direction][w] = v;
        if (seen[1 - direction].count(w)) {
          const double finaldist_w = vw + seen[1 - direction].at(w);
          if (!has_finaldist || finaldist > finaldist_w) {
            finaldist = finaldist_w;
            meetnode = w;
            has_finaldist = true;
          }
        }
      }
    }
  }
  throw std::runtime_error("No path between source and target");
}

// Faithful port of networkx Kruskal `minimum_spanning_edges` (simple graph,
// no NaN weights, no partitions). Returns selected edges in Kruskal order.
inline std::vector<Edge> minimum_spanning_edges(const Graph& G,
                                                const std::string& weight) {
  std::vector<Edge> sorted_edges = G.edges();
  std::stable_sort(sorted_edges.begin(), sorted_edges.end(),
                   [&](const Edge& a, const Edge& b) {
                     return a.weight(weight) < b.weight(weight);
                   });

  // UnionFind keyed by node id.
  std::unordered_map<int, int> parent;
  std::function<int(int)> find = [&](int x) {
    if (!parent.count(x)) return x;
    if (parent[x] != x) parent[x] = find(parent[x]);
    return parent[x];
  };

  std::vector<Edge> result;
  for (const Edge& e : sorted_edges) {
    const int ru = find(e.u);
    const int rv = find(e.v);
    if (ru != rv) {
      parent[ru] = rv;
      result.push_back(e);
    }
  }
  return result;
}

// Faithful port of networkx `_remove_nonterminal_leaves`.
inline void remove_nonterminal_leaves(Graph& G,
                                      const std::vector<int>& terminals) {
  const std::set<int> terminal_set(terminals.begin(), terminals.end());
  std::set<int> leaves;
  for (const int n : G.nodes()) {
    if (G.degree(n) == 1) leaves.insert(n);
  }
  std::set<int> nonterminal_leaves;
  for (const int n : leaves) {
    if (!terminal_set.count(n)) nonterminal_leaves.insert(n);
  }

  while (!nonterminal_leaves.empty()) {
    std::set<int> candidate_leaves;
    for (const int n : nonterminal_leaves) {
      for (const auto& nb : G.nbrs(n)) candidate_leaves.insert(nb.node);
    }
    for (const int n : nonterminal_leaves) candidate_leaves.erase(n);
    for (const int n : terminal_set) candidate_leaves.erase(n);

    G.remove_nodes(nonterminal_leaves);

    leaves.clear();
    for (const int n : candidate_leaves) {
      if (G.degree(n) == 1) leaves.insert(n);
    }
    nonterminal_leaves.clear();
    for (const int n : leaves) {
      if (!terminal_set.count(n)) nonterminal_leaves.insert(n);
    }
  }
}

}  // namespace dqc
