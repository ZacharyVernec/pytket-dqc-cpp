#pragma once

#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#include "dqc/distribution.hpp"
#include "dqc/graph.hpp"
#include "dqc/graph_tools.hpp"
#include "dqc/hypergraph.hpp"
#include "dqc/py_set.hpp"

namespace dqc {

// Mirrors `pytket_dqc.allocators.gain_manager.GainManager`.
class GainManager {
 public:
  // Non-owning alias constructor (mirrors the reference semantics where
  // `GainManager(distribution)` holds a reference to the caller's object).
  GainManager(Distribution& distribution, int max_key_size = 5)
      : distribution_(std::shared_ptr<Distribution>(std::shared_ptr<void>(),
                                                    &distribution)),
        max_key_size_(max_key_size) {
    init();
  }

  // Owning constructor (used by bindings).
  GainManager(std::shared_ptr<Distribution> distribution, int max_key_size = 5)
      : distribution_(std::move(distribution)), max_key_size_(max_key_size) {
    init();
  }

  Distribution& distribution() const { return *distribution_; }
  int max_key_size() const { return max_key_size_; }
  void set_max_key_size(int max_key_size) { max_key_size_ = max_key_size; }

  const std::map<int, int>& occupancy() const { return occupancy_; }
  const std::map<Hyperedge, int>& hyperedge_cost_map() const {
    return hyperedge_cost_map_;
  }
  const std::map<Hyperedge, bool>& requires_h_embedded_cu1() const {
    return requires_h_embedded_cu1_;
  }

  // Mirrors `update_cost`.
  void update_cost(const Hyperedge& hyperedge) {
    const Distribution& d = *distribution_;
    // servers = frozenset(placement[v] for v in hyperedge.vertices);
    // iterate in CPython frozenset order.
    PySet<int> sset(&py_hash_key, &py_eq_key);
    for (const Vertex v : hyperedge.vertices) sset.add(d.placement.at(v));
    std::vector<int> servers = sset.iteration_order();

    Graph tree;
    if (static_cast<int>(servers.size()) <= max_key_size_) {
      const std::set<int> key(servers.begin(), servers.end());
      const auto it = steiner_cache_.find(key);
      if (it == steiner_cache_.end()) {
        tree = steiner_tree(d.server_graph, servers);
        steiner_cache_[key] = tree;
      } else {
        tree = it->second;
      }
    } else {
      tree = steiner_tree(d.server_graph, servers);
    }

    if (requires_h_embedded_cu1_.count(hyperedge) == 0) {
      requires_h_embedded_cu1_[hyperedge] =
          d.circuit.requires_h_embedded_cu1(hyperedge);
    }

    hyperedge_cost_map_[hyperedge] = hyperedge_cost_with_tree(
        d.circuit, hyperedge, d.placement, tree,
        requires_h_embedded_cu1_[hyperedge]);
  }

  int split_hyperedge_gain(const Hyperedge& old_hyperedge,
                           const std::vector<Hyperedge>& new_hyperedge_list) {
    const int current_cost = hyperedge_cost_map_.at(old_hyperedge);
    int new_cost = 0;
    for (const Hyperedge& hyperedge : new_hyperedge_list) {
      if (hyperedge_cost_map_.count(hyperedge) == 0) update_cost(hyperedge);
      new_cost += hyperedge_cost_map_[hyperedge];
    }
    return current_cost - new_cost;
  }

  void split_hyperedge(const Hyperedge& old_hyperedge,
                       const std::vector<Hyperedge>& new_hyperedge_list,
                       bool recalculate_cost = true) {
    distribution_->circuit.split_hyperedge(old_hyperedge, new_hyperedge_list);
    if (recalculate_cost) {
      for (const Hyperedge& h : new_hyperedge_list) update_cost(h);
    }
  }

  int merge_hyperedge_gain(
      const std::vector<Hyperedge>& to_merge_hyperedge_list) {
    std::set<Hyperedge> unique(to_merge_hyperedge_list.begin(),
                               to_merge_hyperedge_list.end());
    if (unique.size() != to_merge_hyperedge_list.size()) {
      throw std::runtime_error("The hyperedges to be merged must be unique.");
    }
    int current_cost = 0;
    for (const Hyperedge& h : to_merge_hyperedge_list) {
      current_cost += hyperedge_cost_map_.at(h);
    }
    std::set<Vertex> vs;
    for (const Hyperedge& h : to_merge_hyperedge_list) {
      vs.insert(h.vertices.begin(), h.vertices.end());
    }
    const Hyperedge new_hyperedge(std::vector<Vertex>(vs.begin(), vs.end()),
                                  to_merge_hyperedge_list[0].weight);
    if (hyperedge_cost_map_.count(new_hyperedge) == 0) {
      update_cost(new_hyperedge);
    }
    const int new_cost = hyperedge_cost_map_[new_hyperedge];
    return current_cost - new_cost;
  }

  Hyperedge merge_hyperedge(
      const std::vector<Hyperedge>& to_merge_hyperedge_list,
      bool recalculate_cost = true) {
    const Hyperedge new_hyperedge =
        distribution_->circuit.merge_hyperedge(to_merge_hyperedge_list);
    if (recalculate_cost) update_cost(new_hyperedge);
    return new_hyperedge;
  }

  int move_vertex_gain(Vertex vertex, int new_server) {
    const int prev_server = distribution_->placement.at(vertex);
    if (prev_server == new_server) return 0;

    std::vector<Hyperedge> incident =
        distribution_->circuit.hyperedge_dict().at(vertex);
    std::map<Hyperedge, int> prev_cost_map;
    int prev_cost = 0;
    for (const Hyperedge& h : incident) {
      prev_cost_map[h] = hyperedge_cost_map_[h];
      prev_cost += hyperedge_cost_map_[h];
    }

    move_vertex(vertex, new_server);  // recalculates costs

    int new_cost = 0;
    for (const Hyperedge& h : distribution_->circuit.hyperedge_dict().at(vertex)) {
      new_cost += hyperedge_cost_map_[h];
    }

    move_vertex(vertex, prev_server, false);
    for (const auto& [h, cost] : prev_cost_map) {
      hyperedge_cost_map_[h] = cost;
    }
    return prev_cost - new_cost;
  }

  void move_vertex(Vertex vertex, int server, bool recalculate_cost = true) {
    Distribution& d = *distribution_;
    if (d.circuit.is_qubit_vertex(vertex)) {
      for (const auto& [h, b] : requires_h_embedded_cu1_) {
        (void)h;
        if (b) {
          throw std::runtime_error(
              "Changing the placement after gates are embedded is not "
              "allowed.");
        }
      }
    }

    const int prev_server = d.placement.at(vertex);
    if (prev_server != server) {
      if (qubit_vertices_.count(vertex)) {
        occupancy_[server] += 1;
        occupancy_[prev_server] -= 1;
      }
      d.placement.set(vertex, server);
      if (recalculate_cost) {
        for (const Hyperedge& h : d.circuit.hyperedge_dict().at(vertex)) {
          update_cost(h);
        }
      }
    }
  }

  bool is_move_valid(Vertex vertex, int server) const {
    if (qubit_vertices_.count(vertex)) {
      const int capacity =
          static_cast<int>(distribution_->server_qubits.at(server).size());
      if (server == current_server(vertex)) {
        return occupancy_.at(server) <= capacity;
      }
      return occupancy_.at(server) < capacity;
    }
    return true;
  }

  int current_server(Vertex vertex) const {
    return distribution_->placement.at(vertex);
  }

 private:
  void init() {
    const HypergraphCircuit& dist_circ = distribution_->circuit;
    for (const Vertex v : dist_circ.vertex_list()) {
      if (dist_circ.is_qubit_vertex(v)) qubit_vertices_.insert(v);
    }
    for (const auto& [server, qubits] : distribution_->server_qubits) {
      (void)qubits;
      occupancy_[server] = 0;
    }
    for (const auto& [vertex, server] : distribution_->placement.placement) {
      if (qubit_vertices_.count(vertex)) occupancy_[server] += 1;
    }
    for (const Hyperedge& hyperedge : dist_circ.hyperedge_list()) {
      update_cost(hyperedge);
    }
  }

  std::shared_ptr<Distribution> distribution_;
  int max_key_size_;
  std::set<Vertex> qubit_vertices_;
  std::map<int, int> occupancy_;
  std::map<Hyperedge, int> hyperedge_cost_map_;
  std::map<Hyperedge, bool> requires_h_embedded_cu1_;
  std::map<std::set<int>, Graph> steiner_cache_;
};

}  // namespace dqc
