#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "dqc/hash.hpp"
#include "dqc/placement.hpp"
#include "dqc/types.hpp"

namespace dqc {

// Faithful port of `pytket_dqc.circuits.hypergraph.Hyperedge` (NamedTuple).
struct Hyperedge {
  std::vector<Vertex> vertices;
  int weight = 1;

  Hyperedge() = default;
  Hyperedge(std::vector<Vertex> vertices, int weight = 1)
      : vertices(std::move(vertices)), weight(weight) {}

  bool operator==(const Hyperedge& o) const {
    return weight == o.weight && vertices == o.vertices;
  }
  bool operator!=(const Hyperedge& o) const { return !(*this == o); }

  // Lexicographic ordering (used only as a deterministic map key; CPython
  // dicts keyed by Hyperedge use `__hash__`, but order is never relied upon
  // for hyperedge-keyed maps).
  bool operator<(const Hyperedge& o) const {
    if (weight != o.weight) return weight < o.weight;
    return std::lexicographical_compare(vertices.begin(), vertices.end(),
                                        o.vertices.begin(), o.vertices.end());
  }

  // Mirrors `Hyperedge.__hash__ = hash((frozenset(vertices), weight))`.
  py_hash_t py_hash() const {
    const py_hash_t h0 = py_hash_frozenset_int(vertices.begin(), vertices.end());
    const py_hash_t h1 = py_hash_int(weight);
    return py_hash_tuple2(h0, h1);
  }

  std::string to_string() const {
    std::ostringstream os;
    os << "Hyperedge(vertices=[";
    for (std::size_t i = 0; i < vertices.size(); ++i) {
      if (i) os << ", ";
      os << vertices[i];
    }
    os << "], weight=" << weight << ")";
    return os.str();
  }
};

// Faithful port of `pytket_dqc.circuits.hypergraph.Hypergraph`.
//
// NOTE (fidelity): `vertex_neighbours` is a Python `set` of ints. CPython's
// iteration order for an int set is NOT ascending in general; it follows the
// hash-table layout. `std::set` (ascending) is used here as a first
// approximation. A faithful `PyOrderedSet` replicating CPython int-set
// iteration order will replace it (needed for structural exactness of
// `get_boundary` and similar order-sensitive paths).
class Hypergraph {
 public:
  Hypergraph() = default;

  // --- accessors (mirror Python attribute names) ---
  const std::vector<Vertex>& vertex_list() const { return vertex_list_; }
  const std::vector<Hyperedge>& hyperedge_list() const {
    return hyperedge_list_;
  }
  const std::map<Vertex, std::vector<Hyperedge>>& hyperedge_dict() const {
    return hyperedge_dict_;
  }
  const std::map<Vertex, std::set<Vertex>>& vertex_neighbours() const {
    return vertex_neighbours_;
  }

  bool operator==(const Hypergraph& o) const {
    return vertex_list_ == o.vertex_list_ &&
           hyperedge_list_ == o.hyperedge_list_ &&
           hyperedge_dict_ == o.hyperedge_dict_ &&
           vertex_neighbours_ == o.vertex_neighbours_;
  }
  bool operator!=(const Hypergraph& o) const { return !(*this == o); }

  bool contains_vertex(Vertex v) const {
    return std::find(vertex_list_.begin(), vertex_list_.end(), v) !=
           vertex_list_.end();
  }
  bool contains_hyperedge(const Hyperedge& e) const {
    return std::find(hyperedge_list_.begin(), hyperedge_list_.end(), e) !=
           hyperedge_list_.end();
  }

  // Clears all hyperedges (list/dict/neighbours) while keeping the vertex
  // set and vertex ordering intact. Mirrors the manual clearing done in
  // `PacMan.get_hypergraph_from_packets`.
  void reset_hyperedges() {
    hyperedge_list_.clear();
    for (auto& [v, list] : hyperedge_dict_) list.clear();
    for (auto& [v, nbrs] : vertex_neighbours_) nbrs.clear();
  }

  void add_vertex(Vertex v) {
    if (!contains_vertex(v)) {
      vertex_list_.push_back(v);
      hyperedge_dict_[v] = {};
      vertex_neighbours_[v] = {};
    }
  }

  void add_vertices(const std::vector<Vertex>& vertices) {
    for (const Vertex v : vertices) {
      add_vertex(v);
    }
  }

  void add_hyperedge(const std::vector<Vertex>& vertices, int weight = 1,
                     std::optional<std::size_t> hyperedge_list_index =
                         std::nullopt,
                     const std::vector<std::size_t>* hyperedge_dict_index =
                         nullptr) {
    if (vertices.size() < 1) {
      throw std::runtime_error("Hyperedges must contain at least 1 vertex.");
    }
    const Hyperedge hyperedge(vertices, weight);
    for (std::size_t i = 0; i < vertices.size(); ++i) {
      const Vertex v = vertices[i];
      if (!contains_vertex(v)) {
        throw std::runtime_error(
            "An element of the hyperedge is not a vertex in the hypergraph.");
      }
      std::vector<Hyperedge>& incident = hyperedge_dict_.at(v);
      if (hyperedge_dict_index == nullptr) {
        incident.push_back(hyperedge);
      } else {
        incident.insert(incident.begin() + (*hyperedge_dict_index)[i],
                        hyperedge);
      }
      std::set<Vertex>& nbrs = vertex_neighbours_.at(v);
      nbrs.insert(vertices.begin(), vertices.end());
      nbrs.erase(v);
    }
    if (!hyperedge_list_index.has_value()) {
      hyperedge_list_.push_back(hyperedge);
    } else {
      hyperedge_list_.insert(hyperedge_list_.begin() + *hyperedge_list_index,
                             hyperedge);
    }
  }

  // Index of the first hyperedge equal to `e` in `hyperedge_list`. Throws
  // std::out_of_range if absent.
  std::size_t index_in_list(const Hyperedge& e) const {
    for (std::size_t i = 0; i < hyperedge_list_.size(); ++i) {
      if (hyperedge_list_[i] == e) return i;
    }
    throw std::out_of_range("hyperedge not in hyperedge_list");
  }

  // Index of the first hyperedge equal to `e` in `hyperedge_dict[v]`. Throws
  // std::out_of_range if absent.
  std::size_t index_in_dict(Vertex v, const Hyperedge& e) const {
    const std::vector<Hyperedge>& incident = hyperedge_dict_.at(v);
    for (std::size_t i = 0; i < incident.size(); ++i) {
      if (incident[i] == e) return i;
    }
    throw std::out_of_range("hyperedge not in hyperedge_dict[vertex]");
  }

  void remove_hyperedge(const Hyperedge& old_hyperedge) {
    auto list_it =
        std::find(hyperedge_list_.begin(), hyperedge_list_.end(), old_hyperedge);
    if (list_it == hyperedge_list_.end()) {
      throw std::out_of_range("The hyperedge is not in this hypergraph.");
    }
    hyperedge_list_.erase(list_it);

    for (const Vertex vertex : old_hyperedge.vertices) {
      // old_neighbour_list = set(old_hyperedge.vertices) - {vertex}
      std::set<Vertex> old_neighbours(old_hyperedge.vertices.begin(),
                                      old_hyperedge.vertices.end());
      old_neighbours.erase(vertex);
      for (const Vertex old_neighbour : old_neighbours) {
        // paired_elsewhere = any({vertex, old_neighbour} <= set(h.vertices)
        //                       for h in hyperedge_list)
        bool paired_elsewhere = false;
        for (const Hyperedge& h : hyperedge_list_) {
          const bool has_v = std::find(h.vertices.begin(), h.vertices.end(),
                                       vertex) != h.vertices.end();
          const bool has_n = std::find(h.vertices.begin(), h.vertices.end(),
                                       old_neighbour) != h.vertices.end();
          if (has_v && has_n) {
            paired_elsewhere = true;
            break;
          }
        }
        if (!paired_elsewhere) {
          vertex_neighbours_.at(vertex).erase(old_neighbour);
        }
      }
      std::vector<Hyperedge>& incident = hyperedge_dict_.at(vertex);
      auto dit = std::find(incident.begin(), incident.end(), old_hyperedge);
      if (dit != incident.end()) {
        incident.erase(dit);
      }
    }
  }

  Hyperedge merge_hyperedge(const std::vector<Hyperedge>& to_merge) {
    for (const Hyperedge& e : to_merge) {
      if (!contains_hyperedge(e)) {
        throw std::runtime_error(
            "At least one hyperedge in to_merge does not belong to this "
            "hypergraph.");
      }
    }
    for (const Hyperedge& e : to_merge) {
      if (e.weight != to_merge[0].weight) {
        throw std::runtime_error("Weights of hyperedges to merge should be "
                                 "equal.");
      }
    }
    for (std::size_t i = 0; i < to_merge.size(); ++i) {
      for (std::size_t j = i + 1; j < to_merge.size(); ++j) {
        if (to_merge[i] == to_merge[j]) {
          throw std::runtime_error("The hyperedges to be merged must be "
                                   "unique.");
        }
      }
    }

    std::vector<Vertex> vertices;
    for (const Hyperedge& e : to_merge) {
      for (const Vertex v : e.vertices) {
        if (std::find(vertices.begin(), vertices.end(), v) == vertices.end()) {
          vertices.push_back(v);
        }
      }
    }
    std::sort(vertices.begin(), vertices.end());
    const int weight = to_merge[0].weight;

    std::size_t list_index = index_in_list(to_merge[0]);
    for (const Hyperedge& e : to_merge) {
      list_index = std::min(list_index, index_in_list(e));
    }

    std::vector<std::size_t> dict_index(vertices.size());
    for (std::size_t i = 0; i < vertices.size(); ++i) {
      const Vertex v = vertices[i];
      std::size_t best = std::numeric_limits<std::size_t>::max();
      for (const Hyperedge& e : to_merge) {
        if (std::find(e.vertices.begin(), e.vertices.end(), v) !=
            e.vertices.end()) {
          best = std::min(best, index_in_dict(v, e));
        }
      }
      dict_index[i] = best;
    }

    const Hyperedge new_hyperedge(vertices, weight);
    add_hyperedge(new_hyperedge.vertices, new_hyperedge.weight, list_index,
                  &dict_index);

    // Remove the originals left-to-right (matches Python order of operations).
    for (const Hyperedge& e : to_merge) {
      remove_hyperedge(e);
    }
    return new_hyperedge;
  }

  void split_hyperedge(const Hyperedge& old_hyperedge,
                       const std::vector<Hyperedge>& new_hyperedge_list) {
    std::vector<Vertex> flat;
    for (const Hyperedge& h : new_hyperedge_list) {
      flat.insert(flat.end(), h.vertices.begin(), h.vertices.end());
    }
    const std::set<Vertex> flat_set(flat.begin(), flat.end());
    const std::set<Vertex> old_set(old_hyperedge.vertices.begin(),
                                   old_hyperedge.vertices.end());
    if (flat_set != old_set) {
      throw std::runtime_error(
          "new_hyperedge_list does not match the vertices in old_hyperedge");
    }

    const std::size_t list_index = index_in_list(old_hyperedge);

    // Precompute the dict insertion index for each position of the OLD
    // hyperedge's vertices, once, before inserting anything (mirrors Python).
    // NOTE (fidelity): Python then passes this full list to `add_hyperedge`,
    // which indexes it by the POSITION in the NEW hyperedge's vertices rather
    // than by vertex identity — a misalignment bug that is replicated here.
    std::vector<std::size_t> hyperedge_dict_index(
        old_hyperedge.vertices.size());
    for (std::size_t k = 0; k < old_hyperedge.vertices.size(); ++k) {
      hyperedge_dict_index[k] =
          index_in_dict(old_hyperedge.vertices[k], old_hyperedge);
    }

    // Insert new hyperedges in REVERSE order at the same position, so the
    // final order matches `new_hyperedge_list` (mirrors Python).
    for (std::size_t i = new_hyperedge_list.size(); i-- > 0;) {
      const Hyperedge& h = new_hyperedge_list[i];
      add_hyperedge(h.vertices, h.weight, list_index, &hyperedge_dict_index);
    }

    remove_hyperedge(old_hyperedge);
  }

  bool is_placement(const Placement& placement) const {
    for (const Vertex v : vertex_list_) {
      if (!placement.contains(v)) return false;
    }
    for (const auto& [v, server] : placement.entries) {
      (void)server;
      if (!contains_vertex(v)) return false;
    }
    return true;
  }

  bool is_valid() const {
    std::vector<Vertex> sorted_vertices = vertex_list_;
    std::sort(sorted_vertices.begin(), sorted_vertices.end());
    std::vector<Vertex> unique;
    for (const Vertex v : sorted_vertices) {
      if (unique.empty() || unique.back() != v) unique.push_back(v);
    }

    std::vector<Vertex> ideal;
    if (!vertex_list_.empty()) {
      const Vertex maxv =
          *std::max_element(vertex_list_.begin(), vertex_list_.end());
      for (Vertex i = 0; i <= maxv; ++i) ideal.push_back(i);
    }
    return unique == ideal;
  }

  // Mirrors `kahypar_hyperedges`: returns (hyperedge_indices, flat_vertices).
  std::pair<std::vector<int>, std::vector<Vertex>> kahypar_hyperedges() const {
    std::vector<Vertex> flat;
    for (const Hyperedge& h : hyperedge_list_) {
      flat.insert(flat.end(), h.vertices.begin(), h.vertices.end());
    }
    std::vector<int> indices;
    if (hyperedge_list_.empty()) {
      return {indices, flat};
    }
    indices.push_back(0);
    for (const Hyperedge& h : hyperedge_list_) {
      indices.push_back(static_cast<int>(h.vertices.size()) + indices.back());
    }
    return {indices, flat};
  }

  std::vector<Vertex> get_boundary(const Placement& placement) const {
    std::vector<Vertex> boundary;
    for (const Vertex v : vertex_list_) {
      const int my_block = placement.at(v);
      for (const Vertex neighbour : vertex_neighbours_.at(v)) {
        if (my_block != placement.at(neighbour)) {
          boundary.push_back(v);
          break;
        }
      }
    }
    return boundary;
  }

  bool weight_one_predicate() const {
    for (const Hyperedge& h : hyperedge_list_) {
      if (h.weight != 1) return false;
    }
    return true;
  }

  std::string to_string() const {
    std::ostringstream os;
    os << "Hyperedges: [";
    for (std::size_t i = 0; i < hyperedge_list_.size(); ++i) {
      if (i) os << ", ";
      os << hyperedge_list_[i].to_string();
    }
    os << "]\nVertices: [";
    for (std::size_t i = 0; i < vertex_list_.size(); ++i) {
      if (i) os << ", ";
      os << vertex_list_[i];
    }
    os << "]";
    return os.str();
  }

 private:
  std::vector<Vertex> vertex_list_;
  std::vector<Hyperedge> hyperedge_list_;
  std::map<Vertex, std::vector<Hyperedge>> hyperedge_dict_;
  std::map<Vertex, std::set<Vertex>> vertex_neighbours_;
};

}  // namespace dqc
