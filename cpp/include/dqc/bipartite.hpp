#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#include "dqc/py_set.hpp"

namespace dqc {

// Faithful port of the relevant parts of
// `networkx.algorithms.bipartite.{basic,matching}` (networkx 3.6.1).
//
// `T` is the node type; `HashFn`/`EqFn` reproduce its CPython `__hash__` /
// `__eq__`, so that set iteration order (which drives bipartition assignment
// and matching tie-breaking) matches CPython set order, including
// PYTHONHASHSEED-derived string hashes.
template <typename T, py_hash_t (*HashFn)(const T&), bool (*EqFn)(const T&, const T&)>
struct BipartiteGraph {
  std::vector<T> nodes_;             // insertion order (mirrors nx.Graph)
  std::map<T, std::vector<T>> adj_;  // adjacency in insertion order
  std::set<T> node_set_;

  bool has_node(const T& n) const { return node_set_.count(n) != 0; }
  const std::vector<T>& nodes() const { return nodes_; }
  std::size_t num_nodes() const { return nodes_.size(); }

  void add_node(const T& n) {
    if (!has_node(n)) {
      nodes_.push_back(n);
      adj_[n] = {};
      node_set_.insert(n);
    }
  }

  void add_edge(const T& u, const T& v) {
    add_node(u);
    add_node(v);
    if (u == v) return;
    if (std::find(adj_[u].begin(), adj_[u].end(), v) == adj_[u].end()) {
      adj_[u].push_back(v);
      adj_[v].push_back(u);
    }
  }

  const std::vector<T>& nbrs(const T& n) const { return adj_.at(n); }
  std::size_t degree(const T& n) const { return nbrs(n).size(); }

  // Mirrors `Graph.edges()` (data omitted).
  std::vector<std::pair<T, T>> edges() const {
    std::set<T> seen;
    std::vector<std::pair<T, T>> out;
    for (const T& u : nodes_) {
      for (const T& v : adj_.at(u)) {
        if (seen.count(v) == 0) out.push_back({u, v});
      }
      seen.insert(u);
    }
    return out;
  }

  // Mirrors `nx.connected_components` (`_plain_bfs`): component sets are
  // built by inserting nodes in BFS level order, and components are yielded
  // in order of their first node in G.
  std::vector<PySet<T>> connected_components() const {
    std::set<T> visited;
    std::vector<PySet<T>> comps;
    for (const T& n : nodes_) {
      if (visited.count(n)) continue;
      PySet<T> comp(HashFn, EqFn);
      std::vector<T> nextlevel = {n};
      visited.insert(n);
      comp.add(n);
      while (!nextlevel.empty()) {
        const std::vector<T> thislevel = nextlevel;
        nextlevel.clear();
        for (const T& v : thislevel) {
          for (const T& w : adj_.at(v)) {
            if (!visited.count(w)) {
              visited.insert(w);
              nextlevel.push_back(w);
              comp.add(w);
            }
          }
        }
      }
      comps.push_back(std::move(comp));
    }
    return comps;
  }

  // Mirrors `bipartite.color`. First node of each connected component is
  // coloured 1; isolated nodes are coloured 0.
  std::map<T, int> color() const {
    std::map<T, int> c;
    for (const auto& [n, is_top] : color_items()) {
      c[n] = is_top;
    }
    return c;
  }

  // `bipartite.color` in CPython dict insertion order.
  std::vector<std::pair<T, int>> color_items() const {
    std::map<T, int> c;
    std::vector<std::pair<T, int>> order;
    for (const T& n : nodes_) {
      if (c.count(n) || degree(n) == 0) continue;
      c[n] = 1;
      order.push_back({n, 1});
      std::vector<T> stack = {n};
      while (!stack.empty()) {
        const T v = stack.back();
        stack.pop_back();
        const int cc = 1 - c.at(v);
        for (const T& w : adj_.at(v)) {
          if (c.count(w)) {
            if (c.at(w) == c.at(v)) {
              throw std::runtime_error("Graph is not bipartite.");
            }
          } else {
            c[w] = cc;
            order.push_back({w, cc});
            stack.push_back(w);
          }
        }
      }
    }
    for (const T& n : nodes_) {
      if (degree(n) == 0) {
        c[n] = 0;
        order.push_back({n, 0});
      }
    }
    return order;
  }

  // Mirrors `bipartite.sets`: returns (X, Y). With `top_nodes` given, X is
  // exactly `set(top_nodes)`; Y = `set(G) - X` (set(G) iterated in table
  // order). Without `top_nodes`, X = colour-1 nodes, Y = colour-0 nodes.
  std::pair<PySet<T>, PySet<T>> sets(const PySet<T>* top_nodes = nullptr) const {
    PySet<T> X(HashFn, EqFn);
    PySet<T> Y(HashFn, EqFn);
    if (top_nodes != nullptr) {
      for (const T& n : top_nodes->iteration_order()) X.add(n);
      PySet<T> G(HashFn, EqFn);
      for (const T& n : nodes_) G.add(n);
      for (const T& n : G.iteration_order()) {
        if (!X.contains(n)) Y.add(n);
      }
    } else {
      for (const auto& [n, is_top] : color_items()) {
        if (is_top)
          X.add(n);
        else
          Y.add(n);
      }
    }
    return {std::move(X), std::move(Y)};
  }

  // Mirrors `G.subgraph(nbunch)` (a filtered view): the node order is either
  // the iteration order of `set(nbunch)` (rebuilt via set_add_entry) or the
  // parent node order filtered, depending on `FilterAdjacency.__iter__`
  // (`2 * len(nbunch_set) < len(parent)`). Each node's neighbour order is the
  // parent graph's adjacency order filtered to the induced nodes.
  BipartiteGraph subgraph(const PySet<T>& keep) const {
    BipartiteGraph g;
    PySet<T> keep_set(HashFn, EqFn);
    for (const T& n : keep.iteration_order()) keep_set.add(n);
    if (2 * keep_set.size() < nodes_.size()) {
      for (const T& n : keep_set.iteration_order()) {
        if (has_node(n)) g.add_node(n);
      }
    } else {
      for (const T& n : nodes_) {
        if (keep_set.contains(n)) g.add_node(n);
      }
    }
    for (const T& u : g.nodes()) {
      for (const T& v : adj_.at(u)) {
        if (keep_set.contains(v)) g.add_edge(u, v);
      }
    }
    return g;
  }

  // Mirrors `bipartite.maximum_matching` (Hopcroft--Karp). `left` is iterated
  // in CPython set order (the order of `top_nodes`). Returns a map with
  // entries in BOTH directions for matched nodes; unmatched nodes are absent.
  std::map<T, T> maximum_matching(const PySet<T>& top_nodes) const {
    const auto [left_set, right_set] = sets(&top_nodes);
    const std::vector<T> left = left_set.iteration_order();
    const std::vector<T> right = right_set.iteration_order();

    const double INF = std::numeric_limits<double>::infinity();
    std::map<T, std::optional<T>> leftmatches;   // left node -> right node
    std::map<T, std::optional<T>> rightmatches;  // right node -> left node
    for (const T& v : left) leftmatches[v] = std::nullopt;
    for (const T& v : right) rightmatches[v] = std::nullopt;

    std::map<std::optional<T>, double> distances;  // key: left node or None

    auto rm_lookup = [&](const T& right_node) -> std::optional<T> {
      auto it = rightmatches.find(right_node);
      if (it == rightmatches.end()) return std::nullopt;
      return it->second;
    };

    auto bfs = [&]() -> bool {
      std::vector<std::optional<T>> queue;
      for (const T& v : left) {
        if (!leftmatches.at(v).has_value()) {
          distances[v] = 0;
          queue.push_back(v);
        } else {
          distances[v] = INF;
        }
      }
      distances[std::nullopt] = INF;
      while (!queue.empty()) {
        const std::optional<T> v = queue.front();
        queue.erase(queue.begin());
        const T* vv = v.has_value() ? &*v : nullptr;
        const double dv = vv ? distances.at(v) : distances.at(std::nullopt);
        if (vv == nullptr) {
          // distances[None] < distances[None] is false; nothing to expand.
          continue;
        }
        if (dv < distances.at(std::nullopt)) {
          for (const T& u : adj_.at(*vv)) {
            const std::optional<T> rm = rm_lookup(u);
            const T* rm_node = rm.has_value() ? &*rm : nullptr;
            const double drm =
                rm_node ? distances.at(rm) : distances.at(std::nullopt);
            if (std::isinf(drm)) {
              if (rm_node) {
                distances[rm] = dv + 1;
                queue.push_back(rm);
              } else {
                distances[std::nullopt] = dv + 1;
                queue.push_back(std::nullopt);
              }
            }
          }
        }
      }
      return !std::isinf(distances.at(std::nullopt));
    };

    std::function<bool(std::optional<T>)> dfs = [&](std::optional<T> v) {
      if (!v.has_value()) return true;
      const T& vv = *v;
      for (const T& u : adj_.at(vv)) {
        const std::optional<T> rm = rm_lookup(u);
        const T* rm_node = rm.has_value() ? &*rm : nullptr;
        const double drm =
            rm_node ? distances.at(rm) : distances.at(std::nullopt);
        if (drm == distances.at(v) + 1) {
          if (dfs(rm)) {
            rightmatches[u] = vv;
            leftmatches[vv] = u;
            return true;
          }
        }
      }
      distances[v] = INF;
      return false;
    };

    while (bfs()) {
      for (const T& v : left) {
        if (!leftmatches.at(v).has_value()) {
          dfs(v);
        }
      }
    }

    std::map<T, T> out;
    for (const auto& [k, v] : leftmatches) {
      if (v.has_value()) out[k] = *v;
    }
    for (const auto& [k, v] : rightmatches) {
      if (v.has_value()) out[k] = *v;
    }
    return out;
  }

  // Mirrors `bipartite.to_vertex_cover` (Konig's theorem). The returned cover
  // iterates in CPython set order.
  PySet<T> to_vertex_cover(const std::map<T, T>& matching,
                           const PySet<T>& top_nodes) const {
    const auto [L, R] = sets(&top_nodes);
    // unmatched_vertices = set(G) - set(matching): iterate set(G) in table
    // order, keeping nodes absent from the matching dict.
    PySet<T> G(HashFn, EqFn);
    for (const T& n : nodes_) G.add(n);
    PySet<T> unmatched_vertices(HashFn, EqFn);
    for (const T& n : G.iteration_order()) {
      if (matching.count(n) == 0) unmatched_vertices.add(n);
    }
    // U = unmatched_vertices & L: CPython set_intersection iterates the
    // smaller of the two sets.
    PySet<T> U(HashFn, EqFn);
    if (unmatched_vertices.size() <= L.size()) {
      for (const T& n : unmatched_vertices.iteration_order()) {
        if (L.contains(n)) U.add(n);
      }
    } else {
      for (const T& n : L.iteration_order()) {
        if (unmatched_vertices.contains(n)) U.add(n);
      }
    }
    const PySet<T> Z = connected_by_alternating_paths(matching, U);
    PySet<T> cover(HashFn, EqFn);
    // cover = (L - Z) | (R & Z). set_difference iterates L; set_intersection
    // iterates the smaller of R and Z; set_union copies the left operand then
    // updates with the right operand (disjoint here).
    for (const T& n : L.iteration_order()) {
      if (!Z.contains(n)) cover.add(n);
    }
    if (R.size() <= Z.size()) {
      for (const T& n : R.iteration_order()) {
        if (Z.contains(n)) cover.add(n);
      }
    } else {
      for (const T& n : Z.iteration_order()) {
        if (R.contains(n)) cover.add(n);
      }
    }
    return cover;
  }

 private:
  // Mirrors `_connected_by_alternating_paths`.
  PySet<T> connected_by_alternating_paths(const std::map<T, T>& matching,
                                          const PySet<T>& targets) const {
    // edge_sets: frozensets of matched edges; matched_edges: tuples.
    std::set<std::pair<T, T>> matched_edges;
    for (const auto& [u, v] : matching) {
      if (u < v)
        matched_edges.insert({u, v});
      else if (v < u)
        matched_edges.insert({v, u});
      // ignore self-loops (u == v): frozenset has a single element, the
      // tuple form would be (u, u); networkx matching never self-matches.
    }
    std::set<std::pair<T, T>> unmatched_edges;
    for (const auto& [u, v] : edges()) {
      std::pair<T, T> canonical = (u < v) ? std::make_pair(u, v)
                                          : std::make_pair(v, u);
      if (matched_edges.count(canonical) == 0) {
        unmatched_edges.insert({u, v});
      }
    }
    PySet<T> result(HashFn, EqFn);
    for (const T& v : nodes_) {
      if (targets.contains(v) ||
          is_connected_by_alternating_path(v, matched_edges, unmatched_edges,
                                           targets)) {
        result.add(v);
      }
    }
    return result;
  }

  // Mirrors `_is_connected_by_alternating_path`.
  bool is_connected_by_alternating_path(
      const T& v, const std::set<std::pair<T, T>>& matched_edges,
      const std::set<std::pair<T, T>>& unmatched_edges,
      const PySet<T>& targets) const {
    std::function<bool(const T&, bool)> alternating_dfs =
        [&](const T& u, bool along_matched) {
          std::set<T> visited;
          const int initial_depth = along_matched ? 0 : 1;
          // stack of (parent, iterator index, depth)
          std::vector<std::tuple<T, std::size_t, int>> stack;
          stack.push_back({u, 0, initial_depth});
          while (!stack.empty()) {
            auto& [parent, idx, depth] = stack.back();
            // Python: valid_edges = matched_edges if depth % 2 else
            // unmatched_edges (odd depth -> matched).
            const bool use_matched = (depth % 2 != 0);
            const std::vector<T>& children = adj_.at(parent);
            if (idx >= children.size()) {
              stack.pop_back();
              continue;
            }
            const T child = children[idx];
            ++idx;
            if (visited.count(child)) continue;
            const auto& valid = use_matched ? matched_edges : unmatched_edges;
            // Python checks BOTH orientations: (parent, child) or
            // (child, parent) in valid_edges.
            if (valid.count({parent, child}) || valid.count({child, parent})) {
              if (targets.contains(child)) return true;
              visited.insert(child);
              stack.push_back({child, 0, depth + 1});
            }
          }
          return false;
        };
    return alternating_dfs(v, true) || alternating_dfs(v, false);
  }
};

}  // namespace dqc
