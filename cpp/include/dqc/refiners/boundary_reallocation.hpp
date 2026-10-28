#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "dqc/distribution.hpp"
#include "dqc/gain_manager.hpp"
#include "dqc/hypergraph_circuit.hpp"
#include "dqc/py_set.hpp"
#include "dqc/rng.hpp"

namespace dqc {

// Core of `BoundaryReallocation.refine`, using a caller-provided generator so
// that the surrounding wrapper can observe the final RNG state.
inline bool boundary_reallocation_refine_impl(
    Distribution& distribution, const std::vector<Vertex>& fixed_vertices,
    int num_rounds, double stop_parameter, std::optional<int> cache_limit,
    PyRandom& rng) {
  GainManager gain_manager(distribution);
  if (cache_limit.has_value()) gain_manager.set_max_key_size(*cache_limit);

  int round_id = 0;
  double proportion_moved = 1.0;
  bool refinement_made = false;
  const HypergraphCircuit& dist_circ = gain_manager.distribution().circuit;
  Placement& placement = gain_manager.distribution().placement;

  while (round_id < num_rounds && proportion_moved > stop_parameter) {
    std::vector<Vertex> active_vertices = dist_circ.get_boundary(placement);
    active_vertices.erase(
        std::remove_if(
            active_vertices.begin(), active_vertices.end(),
            [&](const Vertex v) {
              return std::find(fixed_vertices.begin(), fixed_vertices.end(),
                               v) != fixed_vertices.end();
            }),
        active_vertices.end());

    int moves = 0;
    for (const Vertex vertex : active_vertices) {
      const int current_server = gain_manager.current_server(vertex);

      PySet<int> sset(&py_hash_key, &py_eq_key);
      for (const Vertex v : dist_circ.vertex_neighbours().at(vertex)) {
        sset.add(gain_manager.current_server(v));
      }
      sset.add(current_server);
      const std::vector<int> potential_servers = sset.iteration_order();

      int best_server = -1;
      double best_gain = -std::numeric_limits<double>::infinity();
      int best_best_swap = -1;
      for (const int server : potential_servers) {
        int gain = gain_manager.move_vertex_gain(vertex, server);
        int best_swap_vertex = -1;
        if (!gain_manager.is_move_valid(vertex, server)) {
          const std::vector<Vertex> vs = placement.get_vertices_in(server);
          std::vector<Vertex> valid_swaps;
          for (const Vertex v : vs) {
            if (dist_circ.is_qubit_vertex(v)) valid_swaps.push_back(v);
          }

          gain_manager.move_vertex(vertex, server, false);
          double best_swap_gain =
              -std::numeric_limits<double>::infinity();
          for (const Vertex swap_vertex : valid_swaps) {
            const int swap_gain =
                gain_manager.move_vertex_gain(swap_vertex, current_server);
            if (best_swap_vertex == -1 || swap_gain > best_swap_gain ||
                (swap_gain == best_swap_gain && rng.randbelow(2) == 0)) {
              best_swap_gain = swap_gain;
              best_swap_vertex = swap_vertex;
            }
          }
          gain_manager.move_vertex(vertex, current_server, false);
          gain = gain + static_cast<int>(best_swap_gain);
        }
        if (best_server == -1 || gain > best_gain ||
            (gain == best_gain && rng.randbelow(2) == 0)) {
          best_gain = gain;
          best_server = server;
          best_best_swap = best_swap_vertex;
        }
      }

      if (best_server != current_server) {
        gain_manager.move_vertex(vertex, best_server);
        if (best_best_swap != -1) {
          gain_manager.move_vertex(best_best_swap, current_server);
        }
        refinement_made = true;
        moves += 1;
      }
    }
    round_id += 1;
    proportion_moved =
        active_vertices.empty()
            ? 0.0
            : static_cast<double>(moves) / active_vertices.size();
  }
  return refinement_made;
}

// Mirrors `pytket_dqc.refiners.boundary_reallocation.BoundaryReallocation`.
//
// `rng_state` (the 625-word CPython MT19937 state) continues an existing
// Python stream instead of starting a fresh one; the `*_with_state` variant
// also returns the final state so the caller can restore Python's ambient
// `random` module.
inline bool boundary_reallocation_refine(
    Distribution& distribution, const std::vector<Vertex>& fixed_vertices = {},
    int num_rounds = 10, double stop_parameter = 0.05,
    std::optional<std::uint64_t> seed = std::nullopt,
    std::optional<int> cache_limit = std::nullopt,
    const std::optional<std::vector<std::uint32_t>>& rng_state =
        std::nullopt) {
  PyRandom rng(seed);
  if (rng_state.has_value()) rng.set_state(*rng_state);
  return boundary_reallocation_refine_impl(distribution, fixed_vertices,
                                           num_rounds, stop_parameter,
                                           cache_limit, rng);
}

inline std::pair<bool, std::vector<std::uint32_t>>
boundary_reallocation_refine_with_state(
    Distribution& distribution, const std::vector<Vertex>& fixed_vertices,
    int num_rounds, double stop_parameter, std::optional<std::uint64_t> seed,
    std::optional<int> cache_limit,
    const std::optional<std::vector<std::uint32_t>>& rng_state) {
  PyRandom rng(seed);
  if (rng_state.has_value()) rng.set_state(*rng_state);
  const bool refinement_made = boundary_reallocation_refine_impl(
      distribution, fixed_vertices, num_rounds, stop_parameter, cache_limit,
      rng);
  return {refinement_made, rng.get_state()};
}

// Mirrors `pytket_dqc.refiners.detached_gates.DetachedGates`.
inline bool detached_gates_refine(
    Distribution& distribution, int num_rounds = 10,
    double stop_parameter = 0.05,
    std::optional<std::uint64_t> seed = std::nullopt,
    std::optional<int> cache_limit = std::nullopt,
    const std::optional<std::vector<std::uint32_t>>& rng_state =
        std::nullopt) {
  std::vector<Vertex> fixed = distribution.circuit.get_qubit_vertices();
  const std::vector<Vertex> embedded =
      distribution.circuit.get_all_h_embedded_gate_vertices();
  fixed.insert(fixed.end(), embedded.begin(), embedded.end());
  return boundary_reallocation_refine(distribution, fixed, num_rounds,
                                      stop_parameter, seed, cache_limit,
                                      rng_state);
}

inline std::pair<bool, std::vector<std::uint32_t>>
detached_gates_refine_with_state(
    Distribution& distribution, int num_rounds, double stop_parameter,
    std::optional<std::uint64_t> seed, std::optional<int> cache_limit,
    const std::optional<std::vector<std::uint32_t>>& rng_state) {
  std::vector<Vertex> fixed = distribution.circuit.get_qubit_vertices();
  const std::vector<Vertex> embedded =
      distribution.circuit.get_all_h_embedded_gate_vertices();
  fixed.insert(fixed.end(), embedded.begin(), embedded.end());
  return boundary_reallocation_refine_with_state(
      distribution, fixed, num_rounds, stop_parameter, seed, cache_limit,
      rng_state);
}

}  // namespace dqc
