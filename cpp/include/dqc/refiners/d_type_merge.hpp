#pragma once

#include <algorithm>
#include <utility>
#include <vector>

#include "dqc/distribution.hpp"
#include "dqc/gain_manager.hpp"
#include "dqc/hypergraph_circuit.hpp"
#include "dqc/op_type.hpp"

namespace dqc {

// Mirrors `pytket_dqc.refiners.neighbouring_d_type_merge.NeighbouringDTypeMerge`.
inline bool neighbouring_d_type_merge_refine(Distribution& distribution) {
  GainManager gain_mgr(distribution);
  bool refinement_made = false;
  const HypergraphCircuit& circuit = gain_mgr.distribution().circuit;

  for (const Vertex qubit_vertex : circuit.get_qubit_vertices()) {
    std::vector<Hyperedge> hedge_list =
        circuit.hyperedge_dict().at(qubit_vertex);

    while (hedge_list.size() >= 2) {
      const Hyperedge hedge_one = hedge_list[0];
      const Hyperedge hedge_two = hedge_list[1];
      const std::vector<Vertex> hedge_one_gates =
          circuit.get_gate_vertices(hedge_one);
      const std::vector<Vertex> hedge_two_gates =
          circuit.get_gate_vertices(hedge_two);

      const Vertex min_two =
          *std::min_element(hedge_two_gates.begin(), hedge_two_gates.end());
      Vertex first_vertex = -1;
      for (const Vertex v : hedge_one_gates) {
        if (v < min_two) first_vertex = std::max(first_vertex, v);
      }

      const std::vector<Command> intermediate =
          circuit.get_intermediate_commands(first_vertex, min_two, qubit_vertex);
      bool has_H = false;
      for (const Command& c : intermediate) {
        if (c.op_type == kOpH) {
          has_H = true;
          break;
        }
      }

      if (!has_H && gain_mgr.merge_hyperedge_gain({hedge_one, hedge_two}) >= 0) {
        const Hyperedge new_hyperedge =
            gain_mgr.merge_hyperedge({hedge_one, hedge_two});
        hedge_list.erase(hedge_list.begin(), hedge_list.begin() + 2);
        hedge_list.insert(hedge_list.begin(), new_hyperedge);
        refinement_made = true;
      } else if (*std::max_element(hedge_one_gates.begin(),
                                   hedge_one_gates.end()) <
                 *std::max_element(hedge_two_gates.begin(),
                                   hedge_two_gates.end())) {
        hedge_list.erase(hedge_list.begin());
      } else {
        hedge_list.erase(hedge_list.begin() + 1);
      }
    }
  }
  return refinement_made;
}

// Mirrors `pytket_dqc.refiners.intertwined_d_type_merge.IntertwinedDTypeMerge`.
inline bool intertwined_d_type_merge_refine(Distribution& distribution) {
  GainManager gain_mgr(distribution);
  bool refinement_made = false;
  const HypergraphCircuit& circuit = gain_mgr.distribution().circuit;

  for (const Vertex qubit_vertex : circuit.get_qubit_vertices()) {
    std::vector<Hyperedge> hedge_list =
        circuit.hyperedge_dict().at(qubit_vertex);
    std::vector<std::vector<Vertex>> gate_vertices_list;
    for (const Hyperedge& h : hedge_list) {
      gate_vertices_list.push_back(circuit.get_gate_vertices(h));
    }

    while (hedge_list.size() >= 2) {
      Hyperedge first_hedge = hedge_list[0];
      const std::vector<Vertex> first_hedge_gate_vertices =
          gate_vertices_list[0];
      hedge_list.erase(hedge_list.begin());
      gate_vertices_list.erase(gate_vertices_list.begin());

      std::vector<std::pair<Hyperedge, std::vector<Vertex>>> intertwined;
      const Vertex max_first =
          *std::max_element(first_hedge_gate_vertices.begin(),
                            first_hedge_gate_vertices.end());
      for (std::size_t i = 0; i < hedge_list.size(); ++i) {
        const Vertex min_g =
            *std::min_element(gate_vertices_list[i].begin(),
                              gate_vertices_list[i].end());
        if (min_g <= max_first) {
          intertwined.push_back({hedge_list[i], gate_vertices_list[i]});
        }
      }

      for (const auto& [intertwined_hedge, gate_vertices] : intertwined) {
        std::vector<std::pair<Vertex, Vertex>> neighbour_gates;
        for (std::size_t k = 0; k + 1 < first_hedge_gate_vertices.size(); ++k) {
          const Vertex i = first_hedge_gate_vertices[k];
          const Vertex j = first_hedge_gate_vertices[k + 1];
          std::vector<Vertex> intermittent;
          for (const Vertex g : gate_vertices) {
            if (i < g && g < j) intermittent.push_back(g);
          }
          if (!intermittent.empty()) {
            neighbour_gates.push_back(
                {i, *std::min_element(intermittent.begin(),
                                      intermittent.end())});
            neighbour_gates.push_back(
                {*std::max_element(intermittent.begin(), intermittent.end()),
                 j});
          }
        }

        bool can_merge = false;
        for (const auto& [gate_one, gate_two] : neighbour_gates) {
          const std::vector<Command> commands =
              circuit.get_intermediate_commands(gate_one, gate_two,
                                                qubit_vertex);
          bool all_no_H = true;
          for (const Command& c : commands) {
            if (c.op_type == kOpH) {
              all_no_H = false;
              break;
            }
          }
          if (all_no_H) {
            can_merge = true;
            break;
          }
        }

        if (can_merge) {
          first_hedge = gain_mgr.merge_hyperedge(
              {first_hedge, intertwined_hedge});
          const auto idx =
              std::find(hedge_list.begin(), hedge_list.end(),
                        intertwined_hedge) -
              hedge_list.begin();
          if (idx < static_cast<long>(hedge_list.size())) {
            hedge_list.erase(hedge_list.begin() + idx);
            gate_vertices_list.erase(gate_vertices_list.begin() + idx);
          }
          refinement_made = true;
        }
      }
    }
  }
  return refinement_made;
}

}  // namespace dqc
