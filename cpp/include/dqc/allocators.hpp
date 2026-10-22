#pragma once

#include <cmath>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#include "dqc/distribution.hpp"
#include "dqc/gain_manager.hpp"
#include "dqc/graph.hpp"
#include "dqc/hypergraph_circuit.hpp"
#include "dqc/placement.hpp"
#include "dqc/py_set.hpp"
#include "dqc/rng.hpp"

namespace dqc {

// `network.get_server_list()`: the set of servers appearing in the coupling
// edges, iterated in CPython int-set order.
inline std::vector<int> server_list_order(
    const std::vector<std::pair<int, int>>& server_coupling) {
  PySet<int> s(&py_hash_key, &py_eq_key);
  for (const auto& [a, b] : server_coupling) {
    s.add(a);
    s.add(b);
  }
  return s.iteration_order();
}

// Look up the qubit list of a server in an insertion-ordered server_qubits.
inline std::vector<int> qubit_list_of(
    const std::vector<std::pair<int, std::vector<int>>>& server_qubits,
    int server) {
  for (const auto& [s, q] : server_qubits) {
    if (s == server) return q;
  }
  return {};
}

// Mirrors `order_reducing_size`: stable selection sort, largest first.
inline std::vector<std::pair<int, std::vector<int>>> order_reducing_size(
    const std::vector<std::pair<int, std::vector<int>>>& server_qubits) {
  auto qubit_count = [&](int server) -> int {
    for (const auto& [s, q] : server_qubits) {
      if (s == server) return static_cast<int>(q.size());
    }
    return 0;
  };
  std::vector<std::pair<int, std::vector<int>>> order;
  std::vector<int> position_unknown;
  for (const auto& [s, q] : server_qubits) {
    (void)q;
    position_unknown.push_back(s);
  }
  for (std::size_t it = 0; it + 1 < server_qubits.size(); ++it) {
    std::size_t largest_i = 0;
    int largest = position_unknown[largest_i];
    for (std::size_t i = 1; i < position_unknown.size(); ++i) {
      const int server = position_unknown[i];
      if (qubit_count(largest) < qubit_count(server)) {
        largest = server;
        largest_i = i;
      }
    }
    order.push_back({largest, qubit_list_of(server_qubits, largest)});
    position_unknown.erase(position_unknown.begin() + largest_i);
  }
  order.push_back({position_unknown[0],
                   qubit_list_of(server_qubits, position_unknown[0])});
  return order;
}

// Mirrors `Placement.is_valid(dist_circ, network)` for the allocators. All
// placements produced here place every vertex, so `circuit.is_placement`
// holds trivially; only the per-server qubit-capacity check remains.
inline bool placement_is_valid(
    const HypergraphCircuit& dist_circ, const Placement& placement,
    const std::vector<std::pair<int, std::vector<int>>>& server_qubits) {
  for (const auto& [server, qubits] : server_qubits) {
    int count = 0;
    for (const auto& [v, s] : placement.entries) {
      if (s == server && dist_circ.is_qubit_vertex(v)) ++count;
    }
    if (count > static_cast<int>(qubits.size())) return false;
  }
  return true;
}

inline void check_can_implement(
    const HypergraphCircuit& dist_circ,
    const std::vector<std::pair<int, std::vector<int>>>& server_qubits) {
  int capacity = 0;
  for (const auto& [s, q] : server_qubits) {
    (void)s;
    capacity += static_cast<int>(q.size());
  }
  if (capacity <
      static_cast<int>(dist_circ.get_qubit_vertices().size())) {
    throw std::runtime_error(
        "This circuit cannot be implemented on this network.");
  }
}

// Mirrors `Random.allocate`: returns a Distribution whose placement is the
// random allocation.
inline Distribution random_allocate(
    const HypergraphCircuit& dist_circ,
    const std::vector<std::pair<int, int>>& server_coupling,
    const std::vector<std::pair<int, std::vector<int>>>& server_qubits,
    const Graph& server_graph, const std::map<int, int>& server_ebit_mem,
    std::optional<std::uint64_t> seed) {
  check_can_implement(dist_circ, server_qubits);
  const std::vector<int> server_list = server_list_order(server_coupling);

  PyRandom rng(std::uint64_t{0});
  rng.seed(seed);

  Placement placement;
  for (const Vertex vertex : dist_circ.vertex_list()) {
    bool server_full = true;
    int random_server = 0;
    while (server_full) {
      random_server = rng.choice(server_list);
      int n_qubits_placed = 0;
      for (const auto& [v, s] : placement.entries) {
        if (s == random_server && dist_circ.is_qubit_vertex(v)) {
          ++n_qubits_placed;
        }
      }
      const int capacity =
          static_cast<int>(qubit_list_of(server_qubits, random_server).size());
      server_full = (n_qubits_placed == capacity);
    }
    placement.set(vertex, random_server);
  }
  return make_distribution(dist_circ, placement, server_graph, server_qubits,
                           server_ebit_mem);
}

// Mirrors `Ordered.allocate`: returns a Distribution whose placement fills the
// largest servers first.
inline Distribution ordered_allocate(
    const HypergraphCircuit& dist_circ,
    const std::vector<std::pair<int, std::vector<int>>>& server_qubits,
    const Graph& server_graph, const std::map<int, int>& server_ebit_mem) {
  check_can_implement(dist_circ, server_qubits);
  const std::vector<std::pair<int, std::vector<int>>> ordered =
      order_reducing_size(server_qubits);

  std::vector<Vertex> qubit_vertex_list = dist_circ.get_qubit_vertices();
  Placement placement;
  for (const auto& [server, qubit_list] : ordered) {
    const int n = static_cast<int>(qubit_list.size());
    for (int i = 0; i < n && !qubit_vertex_list.empty(); ++i) {
      const Vertex v = qubit_vertex_list.front();
      qubit_vertex_list.erase(qubit_vertex_list.begin());
      placement.set(v, server);
    }
  }
  std::vector<Vertex> gate_vertex_list;
  for (const Vertex v : dist_circ.vertex_list()) {
    if (!dist_circ.is_qubit_vertex(v)) gate_vertex_list.push_back(v);
  }
  const int first_server = ordered.front().first;
  for (const Vertex v : gate_vertex_list) placement.set(v, first_server);
  return make_distribution(dist_circ, placement, server_graph, server_qubits,
                           server_ebit_mem);
}

// Mirrors `Brute.allocate`: returns the lowest-cost valid Distribution
// (first in `itertools.product` order on ties).
inline Distribution brute_allocate(
    const HypergraphCircuit& dist_circ,
    const std::vector<std::pair<int, int>>& server_coupling,
    const std::vector<std::pair<int, std::vector<int>>>& server_qubits,
    const Graph& server_graph, const std::map<int, int>& server_ebit_mem) {
  check_can_implement(dist_circ, server_qubits);
  const std::vector<int> server_list = server_list_order(server_coupling);
  const std::vector<Vertex> vertex_list = dist_circ.vertex_list();
  const std::size_t n = vertex_list.size();

  std::vector<std::size_t> digits(n, 0);
  bool found = false;
  Placement best;
  int best_cost = 0;
  bool exhausted = false;

  while (!exhausted) {
    Placement placement;
    for (std::size_t i = 0; i < n; ++i) {
      placement.set(vertex_list[i], server_list[digits[i]]);
    }
    if (placement_is_valid(dist_circ, placement, server_qubits)) {
      const int cost = distribution_cost(dist_circ, placement, server_graph);
      if (!found || cost < best_cost) {
        best = placement;
        best_cost = cost;
        found = true;
      }
      if (cost == 0) break;
    }
    // Advance the odometer (rightmost digit fastest, like itertools.product).
    std::size_t i = n;
    while (i > 0) {
      --i;
      digits[i] += 1;
      if (digits[i] < server_list.size()) break;
      digits[i] = 0;
      if (i == 0) exhausted = true;
    }
  }
  if (!found) {
    throw std::runtime_error("No valid placement could be found.");
  }
  return make_distribution(dist_circ, best, server_graph, server_qubits,
                           server_ebit_mem);
}

// Mirrors `Annealing.acceptance_criterion`. Overflow (math.exp raising
// OverflowError and being caught) yields +inf here too.
inline double acceptance_criterion(int gain, int iteration,
                                   double initial_temperature = 1.0) {
  const double temperature = initial_temperature / (iteration + 1);
  return std::exp(gain / temperature);
}

// Mirrors `Annealing.allocate`: returns the resulting Distribution.
//
// The reference calls `random.seed(seed)` and then `Random().allocate(...)`,
// which re-seeds with `random.seed(None)` (OS entropy), so the default run is
// non-deterministic — the seed passed to `Annealing.allocate` is discarded
// (a reference bug). This is replicated when `initial_place_seed` is nullopt.
//
// When `initial_place_seed` is set (the differential-test setup that passes
// an `initial_place_method` wrapping `Random` with a fixed seed), the RNG is
// re-seeded with that value instead of `None`, so the whole run is
// deterministic and bit-comparable with the reference.
inline Distribution annealing_allocate(
    const HypergraphCircuit& dist_circ,
    const std::vector<std::pair<int, int>>& server_coupling,
    const std::vector<std::pair<int, std::vector<int>>>& server_qubits,
    const Graph& server_graph, const std::map<int, int>& server_ebit_mem,
    int iterations, double initial_temperature,
    std::optional<int> cache_limit, std::optional<std::uint64_t> seed,
    std::optional<std::uint64_t> initial_place_seed = std::nullopt) {
  check_can_implement(dist_circ, server_qubits);
  const std::vector<int> server_list = server_list_order(server_coupling);

  PyRandom rng(std::uint64_t{0});
  rng.seed(seed);  // `random.seed(seed)` in Annealing.allocate
  if (initial_place_seed.has_value()) {
    // A seeded initial-placement allocator (e.g. the test helper wrapping
    // Random with a fixed seed): `random.seed(SEED)`, then place.
    rng.seed(initial_place_seed);
  } else {
    // Default `Random()` -> `random.seed(None)` (OS entropy): the reference
    // bug (the seed passed to Annealing is discarded), replicated.
    rng.seed(std::nullopt);
  }

  // Initial placement (Random.allocate with the seeding chosen above).
  Placement placement;
  for (const Vertex vertex : dist_circ.vertex_list()) {
    bool server_full = true;
    int random_server = 0;
    while (server_full) {
      random_server = rng.choice(server_list);
      int n_qubits_placed = 0;
      for (const auto& [v, s] : placement.entries) {
        if (s == random_server && dist_circ.is_qubit_vertex(v)) {
          ++n_qubits_placed;
        }
      }
      server_full =
          (n_qubits_placed ==
           static_cast<int>(qubit_list_of(server_qubits, random_server).size()));
    }
    placement.set(vertex, random_server);
  }

  std::map<int, std::vector<int>> sq_map(server_qubits.begin(),
                                         server_qubits.end());
  Distribution distribution{dist_circ, placement, server_graph, sq_map};
  GainManager gain_manager(distribution);
  if (cache_limit.has_value()) gain_manager.set_max_key_size(*cache_limit);

  for (int i = 0; i < iterations; ++i) {
    const Vertex vertex_to_move = rng.choice(dist_circ.vertex_list());
    const int home_server = gain_manager.current_server(vertex_to_move);
    std::vector<int> possible_servers = server_list;
    possible_servers.erase(
        std::find(possible_servers.begin(), possible_servers.end(),
                  home_server));
    const int destination_server = rng.choice(possible_servers);

    int swap_vertex = -1;
    if (dist_circ.is_qubit_vertex(vertex_to_move)) {
      std::vector<Vertex> dest_qubit_list;
      for (const Vertex v : dist_circ.get_qubit_vertices()) {
        if (gain_manager.current_server(v) == destination_server) {
          dest_qubit_list.push_back(v);
        }
      }
      const int q_in_dest = static_cast<int>(dest_qubit_list.size());
      const int size_dest =
          static_cast<int>(qubit_list_of(server_qubits, destination_server).size());
      if (q_in_dest == size_dest) {
        swap_vertex = rng.choice(dest_qubit_list);
      }
    }

    int gain = gain_manager.move_vertex_gain(vertex_to_move, destination_server);
    if (swap_vertex != -1) {
      gain_manager.move_vertex(vertex_to_move, destination_server, false);
      gain += gain_manager.move_vertex_gain(swap_vertex, home_server);
      gain_manager.move_vertex(vertex_to_move, home_server, false);
    }

    const double acceptance_prob =
        acceptance_criterion(gain, i, initial_temperature);
    if (acceptance_prob > rng.uniform(0.0, 1.0)) {
      gain_manager.move_vertex(vertex_to_move, destination_server);
      if (swap_vertex != -1) {
        gain_manager.move_vertex(swap_vertex, home_server);
      }
    }
  }
  return make_distribution(dist_circ, distribution.placement, server_graph,
                           server_qubits, server_ebit_mem);
}

}  // namespace dqc
