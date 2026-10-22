#pragma once

#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "dqc/bipartite.hpp"
#include "dqc/distribution.hpp"
#include "dqc/pacman.hpp"

namespace dqc {

// Mirrors `pytket_dqc.refiners.vertex_cover.VertexCover.networkx_refine`:
// returns the chosen cover (a set of merged packets, iterating in CPython
// set order).
inline PySet<MergedPacket> vertex_cover_networkx_refine(const PacMan& pacman) {
  const MergedBipartiteGraph merged_graph = pacman.merged_graph();
  const ConflictBipartiteGraph conflict_graph = pacman.conflict_graph();
  const PySet<MergedPacket> m_topnodes = pacman.merged_top_nodes();
  const PySet<HoppingPacket> c_topnodes = pacman.conflict_top_nodes();

  const auto matching = merged_graph.maximum_matching(m_topnodes);
  PySet<MergedPacket> cover =
      merged_graph.to_vertex_cover(matching, m_topnodes);

  const PySet<HoppingPacket> hop_packets =
      pacman.get_hopping_packets_within(cover);

  const ConflictBipartiteGraph true_conflict_graph =
      conflict_graph.subgraph(hop_packets);
  PySet<HoppingPacket> tc_topnodes(&py_hash_hopping, &py_eq_hopping);
  for (const HoppingPacket& node : true_conflict_graph.nodes()) {
    if (c_topnodes.contains(node)) tc_topnodes.add(node);
  }
  const auto tc_matching = true_conflict_graph.maximum_matching(tc_topnodes);
  const PySet<HoppingPacket> conflict_removal =
      true_conflict_graph.to_vertex_cover(tc_matching, tc_topnodes);

  for (const HoppingPacket& conflict : conflict_removal.iteration_order()) {
    const Packet p0 = conflict.first;
    const Packet p1 = conflict.second;
    const MergedPacket merged_packet = pacman.get_containing_merged_packet(p0);
    const auto [packet_a, packet_b] =
        pacman.get_split_packets(merged_packet, conflict);
    cover.erase(merged_packet);
    cover.add(packet_a);
    cover.add(packet_b);
  }
  return cover;
}

// Mirrors `pytket_dqc.refiners.vertex_cover.VertexCover.refine` (the
// "networkx" path; "all_brute_force" is not yet ported). Updates
// `distribution` in place and returns true.
inline bool vertex_cover_refine(Distribution& distribution,
                                const std::string& vertex_cover_alg =
                                    "networkx") {
  if (vertex_cover_alg != "networkx") {
    throw std::runtime_error(
        "Only vertex_cover_alg=\"networkx\" is currently supported.");
  }

  const PacMan pacman(distribution.circuit, distribution.placement);
  const PySet<MergedPacket> cover = vertex_cover_networkx_refine(pacman);
  const std::vector<MergedPacket> cover_order = cover.iteration_order();

  HypergraphCircuit new_hyp_circ = pacman.get_hypergraph_from_packets();
  for (const MergedPacket& merged_packet : cover_order) {
    if (merged_packet.empty()) continue;
    const Vertex qubit_vertex = merged_packet[0].qubit_vertex;
    std::vector<Hyperedge> hyperedges;
    for (const Packet& packet : merged_packet.packets) {
      std::vector<Vertex> vs = {qubit_vertex};
      vs.insert(vs.end(), packet.gate_vertices.begin(),
                packet.gate_vertices.end());
      hyperedges.push_back(Hyperedge(vs));
    }
    new_hyp_circ.merge_hyperedge(hyperedges);
  }
  distribution.circuit = new_hyp_circ;

  Placement new_placement;
  for (const Vertex q : new_hyp_circ.get_qubit_vertices()) {
    new_placement.set(q, distribution.placement.at(q));
  }
  for (const MergedPacket& merged_packet : cover_order) {
    for (const Packet& packet : merged_packet.packets) {
      const int server = packet.connected_server_index;
      for (const Vertex vertex : packet.gate_vertices) {
        new_placement.set(vertex, server);
      }
    }
  }
  for (const Vertex vertex : new_hyp_circ.vertex_list()) {
    if (!new_placement.contains(vertex)) {
      const Command gate = new_hyp_circ.get_gate_of_vertex(vertex);
      const Vertex q0 = new_hyp_circ.get_vertex_of_qubit(gate.qubits[0]);
      new_placement.set(vertex, new_placement.at(q0));
    }
  }
  distribution.placement = new_placement;
  return true;
}

}  // namespace dqc
