#pragma once

#include <optional>
#include <set>
#include <utility>
#include <vector>

#include "dqc/distribution.hpp"
#include "dqc/gain_manager.hpp"
#include "dqc/hypergraph_circuit.hpp"
#include "dqc/pacman.hpp"

namespace dqc {

// Mirrors `pytket_dqc.refiners.h_type_merge.EagerHTypeMerge.refine`: merges
// hopping packets greedily (first-found hopping wins on conflicts), returning
// true if any merge was performed.
inline bool eager_h_type_merge_refine(Distribution& distribution) {
  GainManager gain_mgr(distribution);
  const std::vector<Vertex> detached = distribution.detached_gate_list();
  bool refinement_made = false;
  const PacMan pacman(distribution.circuit, distribution.placement);
  const HypergraphCircuit& hyp_circ = distribution.circuit;

  std::vector<HoppingPacket> already_done_hoppings;
  std::vector<std::set<Hyperedge>> all_hedges_to_merge;
  std::size_t current_idx = 0;

  for (const Vertex qubit_vertex :
       gain_mgr.distribution().circuit.get_qubit_vertices()) {
    std::vector<Packet> qubit_packets =
        pacman.packets_by_qubit().at(qubit_vertex);
    if (qubit_packets.empty()) continue;

    Packet current_packet = qubit_packets[0];
    all_hedges_to_merge.push_back({current_packet.parent_hedge});
    current_idx = all_hedges_to_merge.size() - 1;
    bool end_merging_hedges = false;

    while (!qubit_packets.empty()) {
      const auto it =
          std::find(qubit_packets.begin(), qubit_packets.end(), current_packet);
      if (it == qubit_packets.end()) break;
      qubit_packets.erase(it);

      const std::optional<Packet> next_hopper =
          pacman.get_subsequent_hopping_packet(current_packet);
      if (next_hopper.has_value()) {
        const HoppingPacket hopping_packet = {current_packet, *next_hopper};

        bool no_conflict = true;
        for (const HoppingPacket& conflict :
             pacman.get_conflict_hoppings(hopping_packet)) {
          if (std::find(already_done_hoppings.begin(),
                        already_done_hoppings.end(),
                        conflict) != already_done_hoppings.end()) {
            no_conflict = false;
            break;
          }
        }

        std::vector<Vertex> hedge_vertices = {qubit_vertex};
        hedge_vertices.insert(hedge_vertices.end(),
                              current_packet.gate_vertices.begin(),
                              current_packet.gate_vertices.end());
        hedge_vertices.insert(hedge_vertices.end(),
                              next_hopper->gate_vertices.begin(),
                              next_hopper->gate_vertices.end());
        bool no_detached = true;
        for (const Vertex gate :
             hyp_circ.get_h_embedded_gate_vertices(Hyperedge(hedge_vertices))) {
          if (std::find(detached.begin(), detached.end(), gate) !=
              detached.end()) {
            no_detached = false;
            break;
          }
        }

        if (no_conflict && no_detached) {
          already_done_hoppings.push_back(hopping_packet);
          current_packet = *next_hopper;
          all_hedges_to_merge[current_idx].insert(
              current_packet.parent_hedge);
        } else {
          end_merging_hedges = true;
        }
      } else {
        end_merging_hedges = true;
      }

      if (end_merging_hedges && !qubit_packets.empty()) {
        current_packet = qubit_packets[0];
        bool found = false;
        for (std::size_t k = 0; k < all_hedges_to_merge.size(); ++k) {
          if (all_hedges_to_merge[k].count(current_packet.parent_hedge)) {
            current_idx = k;
            found = true;
            break;
          }
        }
        if (!found) {
          all_hedges_to_merge.push_back({current_packet.parent_hedge});
          current_idx = all_hedges_to_merge.size() - 1;
        }
        end_merging_hedges = false;
      }
    }
  }

  for (std::set<Hyperedge>& merging_hedges : all_hedges_to_merge) {
    if (merging_hedges.size() > 1) {
      const std::vector<Hyperedge> list(merging_hedges.begin(),
                                        merging_hedges.end());
      if (gain_mgr.merge_hyperedge_gain(list) >= 0) {
        gain_mgr.merge_hyperedge(list);
        refinement_made = true;
      }
    }
  }
  return refinement_made;
}

}  // namespace dqc
