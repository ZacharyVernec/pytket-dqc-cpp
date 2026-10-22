#pragma once

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "dqc/bipartite.hpp"
#include "dqc/circuit.hpp"
#include "dqc/gateset.hpp"
#include "dqc/hypergraph_circuit.hpp"
#include "dqc/numeric.hpp"
#include "dqc/op_type.hpp"
#include "dqc/placement.hpp"
#include "dqc/py_set.hpp"

namespace dqc {

// Mirrors `pytket_dqc.utils.op_analysis.is_distributable`, restricted to the
// DQC gateset (Rz, X, Z, CU1) that can appear after DQCPass. The reference
// computes `op.get_unitary()` and tests diagonal/antidiagonal; the reduced
// gateset makes the enumeration equivalent.
inline bool is_distributable_op(int op_type) {
  return op_type == kOpRz || op_type == kOpX || op_type == kOpZ ||
         op_type == kOpCU1;
}

// Mirrors `pytket_dqc.packing.pacman.Packet` (NamedTuple).
struct Packet {
  int packet_index = 0;
  Vertex qubit_vertex = 0;
  int connected_server_index = 0;
  std::vector<Vertex> gate_vertices;
  Hyperedge parent_hedge;

  bool operator==(const Packet& o) const {
    return packet_index == o.packet_index && qubit_vertex == o.qubit_vertex &&
           connected_server_index == o.connected_server_index &&
           gate_vertices == o.gate_vertices && parent_hedge == o.parent_hedge;
  }
  bool operator!=(const Packet& o) const { return !(*this == o); }
  // packet_index is globally unique, so this is a strict weak ordering
  // consistent with `operator==`.
  bool operator<(const Packet& o) const {
    return packet_index < o.packet_index;
  }
};

using NeighbouringPacket = std::vector<Packet>;
using HoppingPacket = std::pair<Packet, Packet>;

// Mirrors `MergedPacket = tuple[Packet, ...]`.
struct MergedPacket {
  std::vector<Packet> packets;

  bool operator==(const MergedPacket& o) const {
    return packets == o.packets;
  }
  bool operator!=(const MergedPacket& o) const { return !(*this == o); }
  bool operator<(const MergedPacket& o) const {
    return std::lexicographical_compare(
        packets.begin(), packets.end(), o.packets.begin(), o.packets.end(),
        [](const Packet& a, const Packet& b) {
          return a.packet_index < b.packet_index;
        });
  }
  bool empty() const { return packets.empty(); }
  std::size_t size() const { return packets.size(); }
  const Packet& operator[](std::size_t i) const { return packets[i]; }
  bool contains(const Packet& p) const {
    return std::find(packets.begin(), packets.end(), p) != packets.end();
  }
};

// CPython hash/equality for PacMan's set keys. `Packet.__hash__` is
// `hash("P<i>")` (SipHash keyed by PYTHONHASHSEED); `MergedPacket`,
// `HoppingPacket`, `NeighbouringPacket` and the graph edges are tuples of
// Packets (xxHash tuplehash).
inline py_hash_t py_hash_packet(const Packet& p) {
  return py_hash_str("P" + std::to_string(p.packet_index));
}
inline bool py_eq_packet(const Packet& a, const Packet& b) { return a == b; }

inline py_hash_t py_hash_merged(const MergedPacket& mp) {
  std::vector<py_hash_t> hs;
  hs.reserve(mp.packets.size());
  for (const Packet& p : mp.packets) hs.push_back(py_hash_packet(p));
  return py_hash_tuple(hs.data(), hs.size());
}
inline bool py_eq_merged(const MergedPacket& a, const MergedPacket& b) {
  return a == b;
}

inline py_hash_t py_hash_hopping(const HoppingPacket& hp) {
  const py_hash_t h[2] = {py_hash_packet(hp.first), py_hash_packet(hp.second)};
  return py_hash_tuple(h, 2);
}
inline bool py_eq_hopping(const HoppingPacket& a, const HoppingPacket& b) {
  return a == b;
}

inline py_hash_t py_hash_merged_edge(
    const std::pair<MergedPacket, MergedPacket>& e) {
  const py_hash_t h[2] = {py_hash_merged(e.first), py_hash_merged(e.second)};
  return py_hash_tuple(h, 2);
}
inline bool py_eq_merged_edge(const std::pair<MergedPacket, MergedPacket>& a,
                              const std::pair<MergedPacket, MergedPacket>& b) {
  return a == b;
}

inline py_hash_t py_hash_hopping_edge(
    const std::pair<HoppingPacket, HoppingPacket>& e) {
  const py_hash_t h[2] = {py_hash_hopping(e.first), py_hash_hopping(e.second)};
  return py_hash_tuple(h, 2);
}
inline bool py_eq_hopping_edge(
    const std::pair<HoppingPacket, HoppingPacket>& a,
    const std::pair<HoppingPacket, HoppingPacket>& b) {
  return a == b;
}

using MergedBipartiteGraph =
    BipartiteGraph<MergedPacket, &py_hash_merged, &py_eq_merged>;
using ConflictBipartiteGraph =
    BipartiteGraph<HoppingPacket, &py_hash_hopping, &py_eq_hopping>;

// Mirrors `assign_bipartitions` (returns the "top" side, index 1), with the
// returned set iterating in CPython set order.
template <typename T, py_hash_t (*HashFn)(const T&), bool (*EqFn)(const T&, const T&)>
PySet<T> assign_top_nodes(const BipartiteGraph<T, HashFn, EqFn>& graph) {
  PySet<T> top(HashFn, EqFn);
  for (const PySet<T>& comp : graph.connected_components()) {
    const BipartiteGraph<T, HashFn, EqFn> sub = graph.subgraph(comp);
    const auto [X, Y] = sub.sets();
    for (const T& n : Y.iteration_order()) top.add(n);
  }
  return top;
}

// Mirrors `pytket_dqc.packing.pacman.PacMan`.
class PacMan {
 public:
  PacMan(const HypergraphCircuit& hypergraph_circuit,
         const Placement& placement)
      : hypergraph_circuit_(hypergraph_circuit), placement_(placement) {
    build_packets();
    identify_neighbouring_packets();
    identify_hopping_packets();
    merge_all_packets();
  }

  const std::map<Vertex, std::vector<Packet>>& packets_by_qubit() const {
    return packets_by_qubit_;
  }
  const std::map<Vertex, std::vector<NeighbouringPacket>>&
  neighbouring_packets() const {
    return neighbouring_packets_;
  }
  const std::map<Vertex, std::vector<HoppingPacket>>& hopping_packets() const {
    return hopping_packets_;
  }
  const std::map<Vertex, std::vector<MergedPacket>>& merged_packets() const {
    return merged_packets_;
  }

  // ---- basic construction ------------------------------------------------

  void build_packets() {
    int current_index = 0;
    for (const Vertex qubit_vertex : hypergraph_circuit_.get_qubit_vertices()) {
      packets_by_qubit_[qubit_vertex] = {};
      for (const Hyperedge& hyperedge :
           hypergraph_circuit_.hyperedge_dict().at(qubit_vertex)) {
        int next_index;
        std::vector<Packet> packets;
        hyperedge_to_packets(hyperedge, current_index, &next_index, &packets);
        current_index = next_index;
        for (const Packet& p : packets) {
          packets_by_qubit_[qubit_vertex].push_back(p);
        }
      }
    }
  }

  void hyperedge_to_packets(const Hyperedge& hyperedge, int current_index,
                            int* next_index,
                            std::vector<Packet>* packets) const {
    const Vertex hyperedge_qubit_vertex =
        hypergraph_circuit_.get_qubit_vertex(hyperedge);
    // Insertion-ordered map server -> gate vertices (CPython dict semantics).
    std::vector<std::pair<int, std::vector<Vertex>>> cstdg;
    for (const Vertex gate_vertex :
         hypergraph_circuit_.get_gate_vertices(hyperedge)) {
      const int connected_server =
          get_connected_server(hyperedge_qubit_vertex, gate_vertex);
      if (connected_server == placement_.at(hyperedge_qubit_vertex)) {
        continue;
      }
      bool found = false;
      for (auto& [server, gates] : cstdg) {
        if (server == connected_server) {
          gates.push_back(gate_vertex);
          found = true;
          break;
        }
      }
      if (!found) cstdg.push_back({connected_server, {gate_vertex}});
    }
    int idx = current_index;
    for (const auto& [connected_server, dist_gates] : cstdg) {
      packets->push_back(Packet{idx, hyperedge_qubit_vertex, connected_server,
                                dist_gates, hyperedge});
      idx += 1;
    }
    *next_index = idx;
  }

  int get_connected_server(Vertex qubit_vertex, Vertex gate_vertex) const {
    const int command_index =
        hypergraph_circuit_.get_vertex_to_command_index_map().at(gate_vertex);
    const std::vector<std::string>& qubits =
        hypergraph_circuit_.commands().at(command_index).qubits;
    std::vector<Vertex> candidates;
    for (const std::string& q : qubits) {
      const Vertex qv = hypergraph_circuit_.get_vertex_of_qubit(q);
      if (qv != qubit_vertex) candidates.push_back(qv);
    }
    return placement_.at(candidates[0]);
  }

  // ---- neighbouring / hopping / merging ----------------------------------

  void identify_neighbouring_packets() {
    for (const Vertex qubit_vertex : hypergraph_circuit_.get_qubit_vertices()) {
      std::vector<NeighbouringPacket> neighbouring;
      std::vector<Packet> considered;
      const std::vector<Packet>& packets = packets_by_qubit_.at(qubit_vertex);
      for (std::size_t i = 0; i + 1 < packets.size(); ++i) {
        const Packet& packet = packets[i];
        if (std::find(considered.begin(), considered.end(), packet) !=
            considered.end()) {
          continue;
        }
        NeighbouringPacket group = {packet};
        std::optional<Packet> next = get_next_packet(packet);
        while (next.has_value() &&
               are_neighbouring_packets(packet, *next)) {
          group.push_back(*next);
          considered.push_back(*next);
          next = get_next_packet(*next);
        }
        if (group.size() > 1) neighbouring.push_back(group);
      }
      neighbouring_packets_[qubit_vertex] = neighbouring;
    }
  }

  void identify_hopping_packets() {
    for (const Vertex qubit_vertex : hypergraph_circuit_.get_qubit_vertices()) {
      hopping_packets_[qubit_vertex] = {};
      const std::vector<Packet>& packets = packets_by_qubit_.at(qubit_vertex);
      for (std::size_t i = 0; i + 1 < packets.size(); ++i) {
        const Packet& packet = packets[i];
        std::optional<Packet> packet_to_compare =
            get_next_packet(get_next_packet(packet));
        while (packet_to_compare.has_value()) {
          if (are_hoppable_packets(packet, *packet_to_compare)) {
            hopping_packets_[qubit_vertex].push_back(
                {packet, *packet_to_compare});
            break;
          }
          packet_to_compare = get_next_packet(*packet_to_compare);
        }
      }
    }
  }

  void merge_all_packets() {
    for (const Vertex qubit_vertex : hypergraph_circuit_.get_qubit_vertices()) {
      merged_packets_[qubit_vertex] = {};
      std::vector<Packet> considered;
      for (const Packet& packet : packets_by_qubit_.at(qubit_vertex)) {
        if (std::find(considered.begin(), considered.end(), packet) !=
            considered.end()) {
          continue;
        }
        MergedPacket mergeable;
        mergeable.packets = {packet};
        Packet current = packet;
        bool continue_merging = true;
        while (continue_merging) {
          std::optional<Packet> sub_nbr =
              get_subsequent_neighbouring_packet(current);
          if (sub_nbr.has_value()) {
            current = *sub_nbr;
            mergeable.packets.push_back(current);
            considered.push_back(current);
          } else {
            std::optional<Packet> sub_hop =
                get_subsequent_hopping_packet(current);
            if (sub_hop.has_value()) {
              current = *sub_hop;
              mergeable.packets.push_back(current);
              considered.push_back(current);
            } else {
              continue_merging = false;
            }
          }
        }
        merged_packets_[qubit_vertex].push_back(mergeable);
      }
    }
  }

  std::optional<Packet> get_subsequent_neighbouring_packet(
      const Packet& packet) const {
    std::optional<NeighbouringPacket> found;
    for (const NeighbouringPacket& np :
         neighbouring_packets_.at(packet.qubit_vertex)) {
      if (std::find(np.begin(), np.end(), packet) != np.end()) {
        found = np;
        break;
      }
    }
    if (!found.has_value()) return std::nullopt;
    const NeighbouringPacket& np = *found;
    std::size_t idx = 0;
    for (std::size_t i = 0; i < np.size(); ++i) {
      if (np[i] == packet) {
        idx = i;
        break;
      }
    }
    if (idx + 1 == np.size()) return std::nullopt;
    return np[idx + 1];
  }

  std::optional<Packet> get_subsequent_hopping_packet(
      const Packet& packet) const {
    for (const HoppingPacket& hp : hopping_packets_.at(packet.qubit_vertex)) {
      if (hp.first == packet) return hp.second;
    }
    return std::nullopt;
  }

  std::optional<Packet> get_next_packet(const Packet& packet) const {
    for (const Packet& potential : packets_by_qubit_.at(packet.qubit_vertex)) {
      if (potential.packet_index <= packet.packet_index) continue;
      if (potential.connected_server_index == packet.connected_server_index) {
        return potential;
      }
    }
    return std::nullopt;
  }

  std::optional<Packet> get_next_packet(
      const std::optional<Packet>& packet) const {
    if (!packet.has_value()) return std::nullopt;
    return get_next_packet(*packet);
  }

  std::vector<Packet> get_all_packets() const {
    std::vector<Packet> all;
    for (const auto& [qv, packets] : packets_by_qubit_) {
      (void)qv;
      all.insert(all.end(), packets.begin(), packets.end());
    }
    std::sort(all.begin(), all.end(),
              [](const Packet& a, const Packet& b) {
                return a.packet_index < b.packet_index;
              });
    return all;
  }

  std::vector<MergedPacket> get_all_merged_packets() const {
    std::vector<MergedPacket> out;
    for (const auto& [qv, packets] : merged_packets_) {
      (void)qv;
      out.insert(out.end(), packets.begin(), packets.end());
    }
    return out;
  }

  // ---- packet predicates --------------------------------------------------

  bool are_neighbouring_packets(const Packet& first_packet,
                                const Packet& second_packet) const {
    if (first_packet.qubit_vertex != second_packet.qubit_vertex ||
        first_packet.connected_server_index !=
            second_packet.connected_server_index) {
      return false;
    }
    const std::vector<Command> intermediate =
        get_intermediate_commands(first_packet, second_packet);
    bool searching_for_Z = false;
    bool searching_for_H = false;
    for (const Command& command : intermediate) {
      if (searching_for_Z) {
        if (command.op_type == kOpRz &&
            std::abs(command.params[0]) == 1.0) {
          searching_for_Z = false;
          searching_for_H = true;
        } else {
          return false;
        }
      } else if (searching_for_H) {
        if (command.op_type == kOpH) {
          searching_for_H = false;
        } else {
          return false;
        }
      } else if (command.op_type == kOpH) {
        searching_for_Z = true;
      } else if (!is_distributable_op(command.op_type)) {
        return false;
      }
    }
    return !(searching_for_H || searching_for_Z);
  }

  bool are_hoppable_packets(const Packet& first_packet,
                            const Packet& second_packet) const {
    const std::vector<Command> intermediate =
        get_intermediate_commands(first_packet, second_packet);

    std::vector<int> h_indices;
    for (int i = 0; i < static_cast<int>(intermediate.size()); ++i) {
      if (intermediate[i].op_type == kOpH) h_indices.push_back(i);
    }
    if (h_indices.size() < 2) return false;

    const std::set<int> servers = {
        placement_.at(first_packet.qubit_vertex),
        first_packet.connected_server_index};

    std::vector<int> cu1_indices;
    for (int i = 0; i < static_cast<int>(intermediate.size()); ++i) {
      if (intermediate[i].op_type == kOpCU1) {
        if (i < h_indices[0] || i > h_indices.back()) {
          continue;  // ignored
        }
        if (!hypergraph_circuit_.is_h_embeddable_CU1(intermediate[i], servers,
                                                     placement_)) {
          return false;
        }
        cu1_indices.push_back(i);
      }
    }
    if (cu1_indices.empty()) return false;

    std::vector<std::vector<Op>> ops_1q_list;
    {
      std::vector<Op> first;
      for (int i = h_indices[0]; i < cu1_indices[0]; ++i) {
        first.push_back(Op{intermediate[i].op_type, intermediate[i].params});
      }
      ops_1q_list.push_back(first);
    }
    int prev_cu1_index = cu1_indices[0];
    for (std::size_t k = 1; k < cu1_indices.size(); ++k) {
      const int cu1_index = cu1_indices[k];
      std::vector<Op> ops;
      for (int i = prev_cu1_index + 1; i < cu1_index; ++i) {
        ops.push_back(Op{intermediate[i].op_type, intermediate[i].params});
      }
      ops_1q_list.push_back(to_euler_with_two_hadamards(ops));
      prev_cu1_index = cu1_index;
    }
    {
      std::vector<Op> last;
      for (int i = cu1_indices.back() + 1; i <= h_indices.back(); ++i) {
        last.push_back(Op{intermediate[i].op_type, intermediate[i].params});
      }
      ops_1q_list.push_back(last);
    }

    int n_hadamards_start = 0;
    for (const Op& op : ops_1q_list[0]) {
      if (op.op_type == kOpH) ++n_hadamards_start;
    }
    if (n_hadamards_start != 1) return false;
    int n_hadamards_end = 0;
    for (const Op& op : ops_1q_list.back()) {
      if (op.op_type == kOpH) ++n_hadamards_end;
    }
    if (n_hadamards_end != 1) return false;

    bool is_reversed = false;
    for (std::size_t i = 0; i + 1 < ops_1q_list.size(); ++i) {
      std::vector<Op> first_ops_1q =
          is_reversed ? reversed(ops_1q_list[i]) : ops_1q_list[i];
      is_reversed = false;
      const std::vector<Op>& second_ops_1q = ops_1q_list[i + 1];
      if (!are_1q_op_phases_npi(first_ops_1q, second_ops_1q)) {
        // NOTE (fidelity): the reference checks
        // `second_ops_1q[2].params == 0` — a list compared to an int, which
        // is always False — so the "reversed" fallback below is dead code
        // and the function always returns False here. Replicated faithfully.
        if (second_ops_1q.size() == 5 &&
            second_ops_1q[2].params == std::vector<double>{0.0} &&
            second_ops_1q[4].params == std::vector<double>{0.0}) {
          // Unreachable in the reference (condition is always False).
          if (are_1q_op_phases_npi(first_ops_1q,
                                   reversed(second_ops_1q))) {
            is_reversed = true;
          }
          return false;
        }
        return false;
      }
    }
    return true;
  }

  // ---- packet / hypergraph interfaces ------------------------------------

  std::vector<Command> get_intermediate_commands(
      const Packet& first_packet, const Packet& second_packet) const {
    const Vertex qubit_vertex = first_packet.qubit_vertex;
    const Vertex first_vertex = hypergraph_circuit_.get_last_gate_vertex(
        first_packet.gate_vertices);
    const Vertex second_vertex = hypergraph_circuit_.get_first_gate_vertex(
        second_packet.gate_vertices);
    return hypergraph_circuit_.get_intermediate_commands(first_vertex,
                                                         second_vertex,
                                                         qubit_vertex);
  }

  bool are_1q_op_phases_npi(const std::vector<Op>& prior_1q_ops,
                            const std::vector<Op>& post_1q_ops) const {
    double prior_phase;
    if (prior_1q_ops.size() != 5 &&
        prior_1q_ops.back().op_type == kOpH) {
      prior_phase = 0.0;
    } else {
      prior_phase = prior_1q_ops.back().params[0];
    }
    double post_phase;
    if (post_1q_ops.size() != 5 && post_1q_ops[0].op_type == kOpH) {
      post_phase = 0.0;
    } else {
      post_phase = post_1q_ops[0].params[0];
    }
    const double m = py_mod(prior_phase + post_phase, 1.0);
    return isclose(m, 0.0) || isclose(m, 1.0);
  }

  std::vector<Packet> get_connected_packets(const Packet& packet) const {
    PySet<Packet> connected(&py_hash_packet, &py_eq_packet);
    for (const Vertex gate_vertex : packet.gate_vertices) {
      const std::vector<std::string>& gate_qubits =
          hypergraph_circuit_.get_gate_of_vertex(gate_vertex).qubits;
      std::vector<Vertex> other_candidates;
      for (const std::string& q : gate_qubits) {
        const Vertex qv = hypergraph_circuit_.get_vertex_of_qubit(q);
        if (qv != packet.qubit_vertex) other_candidates.push_back(qv);
      }
      const Vertex other_qubit_vertex = other_candidates[0];
      for (const Packet& potential :
           packets_by_qubit_.at(other_qubit_vertex)) {
        if (std::find(potential.gate_vertices.begin(),
                      potential.gate_vertices.end(),
                      gate_vertex) != potential.gate_vertices.end()) {
          connected.add(potential);
          break;
        }
      }
    }
    return connected.iteration_order();
  }

  MergedPacket get_containing_merged_packet(const Packet& packet) const {
    for (const MergedPacket& mp : merged_packets_.at(packet.qubit_vertex)) {
      if (mp.contains(packet)) return mp;
    }
    throw std::runtime_error("packet not found in any merged packet");
  }

  std::vector<MergedPacket> get_connected_merged_packets(
      const MergedPacket& merged_packet) const {
    PySet<MergedPacket> connected(&py_hash_merged, &py_eq_merged);
    for (const Packet& packet : merged_packet.packets) {
      for (const Packet& cp : get_connected_packets(packet)) {
        connected.add(get_containing_merged_packet(cp));
      }
    }
    return connected.iteration_order();
  }

  std::pair<MergedPacket, MergedPacket> get_split_packets(
      const MergedPacket& merged_packet,
      const HoppingPacket& conflict_packet) const {
    const Packet& c_pac_a = conflict_packet.first;
    const Packet& c_pac_b = conflict_packet.second;
    MergedPacket a, b;
    for (const Packet& p : merged_packet.packets) {
      if (p.packet_index <= c_pac_a.packet_index) a.packets.push_back(p);
      if (p.packet_index >= c_pac_b.packet_index) b.packets.push_back(p);
    }
    return {a, b};
  }

  std::vector<Packet> get_embedded_packets(
      const HoppingPacket& hopping_packet) const {
    const int initial_index = hopping_packet.first.packet_index;
    const int final_index = hopping_packet.second.packet_index;
    PySet<Packet> embedded(&py_hash_packet, &py_eq_packet);
    for (const Packet& packet :
         packets_by_qubit_.at(hopping_packet.first.qubit_vertex)) {
      if (packet.packet_index > initial_index &&
          packet.packet_index < final_index) {
        embedded.add(packet);
      }
    }
    return embedded.iteration_order();
  }

  std::map<HoppingPacket, std::vector<Packet>>
  get_all_embedded_packets_for_qubit_vertex(Vertex qubit_vertex) const {
    std::map<HoppingPacket, std::vector<Packet>> embedded;
    for (const HoppingPacket& hp : hopping_packets_.at(qubit_vertex)) {
      embedded[hp] = get_embedded_packets(hp);
    }
    return embedded;
  }

  std::map<Vertex, std::map<HoppingPacket, std::vector<Packet>>>
  get_all_embedded_packets() const {
    std::map<Vertex, std::map<HoppingPacket, std::vector<Packet>>> embedded;
    for (const Vertex qubit_vertex : hypergraph_circuit_.get_qubit_vertices()) {
      auto per_qubit = get_all_embedded_packets_for_qubit_vertex(qubit_vertex);
      if (!per_qubit.empty()) embedded[qubit_vertex] = per_qubit;
    }
    return embedded;
  }

  HoppingPacket get_hopping_packet_from_embedded_packet(
      const Packet& embedded_packet) const {
    for (const HoppingPacket& hp :
         hopping_packets_.at(embedded_packet.qubit_vertex)) {
      if (hp.first.packet_index < embedded_packet.packet_index &&
          hp.second.packet_index > embedded_packet.packet_index) {
        return hp;
      }
    }
    throw std::runtime_error("embedded packet not in any hopping packet");
  }

  PySet<HoppingPacket> get_hopping_packets_within(
      const PySet<MergedPacket>& merged_packets_set) const {
    PySet<HoppingPacket> hoppings_within(&py_hash_hopping, &py_eq_hopping);
    for (const auto& [qv, hop_list] : hopping_packets_) {
      (void)qv;
      for (const HoppingPacket& hp : hop_list) {
        for (const MergedPacket& mp : merged_packets_set.iteration_order()) {
          if (mp.contains(hp.first) && mp.contains(hp.second)) {
            hoppings_within.add(hp);
          }
        }
      }
    }
    return hoppings_within;
  }

  bool is_packet_embedded(const Packet& packet) const {
    const auto all = get_all_embedded_packets();
    auto qit = all.find(packet.qubit_vertex);
    if (qit == all.end()) return false;
    for (const auto& [hp, embedded] : qit->second) {
      (void)hp;
      if (std::find(embedded.begin(), embedded.end(), packet) !=
          embedded.end()) {
        return true;
      }
    }
    return false;
  }

  std::vector<HoppingPacket> get_conflict_hoppings(
      const HoppingPacket& hopping_packet) const {
    std::vector<HoppingPacket> conflicts;
    for (const Packet& embedded : get_embedded_packets(hopping_packet)) {
      for (const Packet& connected : get_connected_packets(embedded)) {
        if (is_packet_embedded(connected)) {
          conflicts.push_back(
              get_hopping_packet_from_embedded_packet(connected));
        }
      }
    }
    return conflicts;
  }

  // ---- hypergraph reconstruction ------------------------------------------

  HypergraphCircuit get_hypergraph_from_packets() const {
    HypergraphCircuit hyp_circ;
    hyp_circ.from_circuit(hypergraph_circuit_.get_circuit());
    const std::vector<Vertex> qubit_vertices = hyp_circ.get_qubit_vertices();
    hyp_circ.reset_hyperedges();

    std::vector<std::vector<Vertex>> hedges_to_add;
    for (const Vertex qubit_vertex : qubit_vertices) {
      for (const Packet& packet : packets_by_qubit_.at(qubit_vertex)) {
        std::vector<Vertex> hedge = {qubit_vertex};
        hedge.insert(hedge.end(), packet.gate_vertices.begin(),
                     packet.gate_vertices.end());
        hedges_to_add.push_back(hedge);
      }
    }
    for (const Vertex vertex : hyp_circ.vertex_list()) {
      if (hyp_circ.is_qubit_vertex(vertex)) continue;
      int count = 0;
      for (const auto& hedge : hedges_to_add) {
        if (std::find(hedge.begin(), hedge.end(), vertex) != hedge.end()) {
          ++count;
        }
      }
      if (count == 0) {
        const Command gate = hyp_circ.get_gate_of_vertex(vertex);
        for (const std::string& q : gate.qubits) {
          hedges_to_add.push_back(
              {hyp_circ.get_vertex_of_qubit(q), vertex});
        }
      }
    }

    std::stable_sort(
        hedges_to_add.begin(), hedges_to_add.end(),
        [&](const std::vector<Vertex>& a, const std::vector<Vertex>& b) {
          Vertex min_a = -1, min_b = -1;
          for (const Vertex v : a) {
            if (!hyp_circ.is_qubit_vertex(v)) {
              min_a = v;
              break;
            }
          }
          for (const Vertex v : b) {
            if (!hyp_circ.is_qubit_vertex(v)) {
              min_b = v;
              break;
            }
          }
          return min_a < min_b;
        });

    for (const auto& hedge : hedges_to_add) {
      hyp_circ.add_hyperedge(hedge);
    }
    return hyp_circ;
  }

  // ---- graph methods -------------------------------------------------------

  // Full bipartite merged graph (nodes = MergedPacket). Node insertion order
  // follows the CPython set order of the edge set (nx `graph.add_edges_from`).
  MergedBipartiteGraph merged_graph() const {
    PySet<std::pair<MergedPacket, MergedPacket>> edges(&py_hash_merged_edge,
                                                       &py_eq_merged_edge);
    for (const Vertex qubit_vertex :
         hypergraph_circuit_.get_qubit_vertices()) {
      for (const MergedPacket& mp : merged_packets_.at(qubit_vertex)) {
        for (const MergedPacket& cmp : get_connected_merged_packets(mp)) {
          edges.add({mp, cmp});
        }
      }
    }
    MergedBipartiteGraph graph;
    for (const auto& [a, b] : edges.iteration_order()) {
      graph.add_edge(a, b);
    }
    return graph;
  }

  // Full bipartite conflict graph (nodes = HoppingPacket).
  ConflictBipartiteGraph conflict_graph() const {
    PySet<std::pair<HoppingPacket, HoppingPacket>> edges(&py_hash_hopping_edge,
                                                         &py_eq_hopping_edge);
    std::vector<HoppingPacket> checked;
    for (const Vertex qubit_vertex :
         hypergraph_circuit_.get_qubit_vertices()) {
      for (const HoppingPacket& hopping : hopping_packets_.at(qubit_vertex)) {
        for (const HoppingPacket& conflict :
             get_conflict_hoppings(hopping)) {
          if (std::find(checked.begin(), checked.end(), conflict) !=
              checked.end()) {
            continue;
          }
          edges.add({hopping, conflict});
        }
        checked.push_back(hopping);
      }
    }
    ConflictBipartiteGraph graph;
    for (const auto& [a, b] : edges.iteration_order()) {
      graph.add_edge(a, b);
    }
    return graph;
  }

  PySet<MergedPacket> merged_top_nodes() const {
    return assign_top_nodes(merged_graph());
  }
  PySet<HoppingPacket> conflict_top_nodes() const {
    return assign_top_nodes(conflict_graph());
  }

  // Mirrors `get_nx_graph_merged`: returns (edges, top_nodes) where top_nodes
  // is `bipartitions[1]` (the colour-0 side of each connected component).
  std::pair<std::vector<std::pair<MergedPacket, MergedPacket>>,
            PySet<MergedPacket>>
  get_nx_graph_merged() const {
    return {merged_graph().edges(), merged_top_nodes()};
  }

  // Mirrors `get_nx_graph_conflict`.
  std::pair<std::vector<std::pair<HoppingPacket, HoppingPacket>>,
            PySet<HoppingPacket>>
  get_nx_graph_conflict() const {
    return {conflict_graph().edges(), conflict_top_nodes()};
  }

 private:
  static std::vector<Op> reversed(const std::vector<Op>& ops) {
    std::vector<Op> r(ops.rbegin(), ops.rend());
    return r;
  }

  HypergraphCircuit hypergraph_circuit_;
  Placement placement_;
  std::map<Vertex, std::vector<Packet>> packets_by_qubit_;
  std::map<Vertex, std::vector<NeighbouringPacket>> neighbouring_packets_;
  std::map<Vertex, std::vector<HoppingPacket>> hopping_packets_;
  std::map<Vertex, std::vector<MergedPacket>> merged_packets_;
};

}  // namespace dqc
