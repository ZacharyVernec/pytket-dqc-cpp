#pragma once

#include <cstdint>
#include <cstdio>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <unistd.h>
#endif

#include "libkahypar.h"

#include "dqc/allocators.hpp"
#include "dqc/distribution.hpp"
#include "dqc/gain_manager.hpp"
#include "dqc/hypergraph_circuit.hpp"
#include "dqc/py_set.hpp"

namespace dqc {

namespace detail {

// The reference calls `context.suppressOutput(True)` before partitioning; the
// C facade has no equivalent, so silence KaHyPar's stdout/stderr chatter for
// the duration of the partition call. This only suppresses output and cannot
// change the computed partition. (POSIX-only; a no-op elsewhere.)
class ScopedStdoutSilencer {
 public:
  ScopedStdoutSilencer() {
#if defined(__unix__) || defined(__APPLE__)
    std::fflush(stdout);
    std::fflush(stderr);
    saved_out_ = ::dup(STDOUT_FILENO);
    saved_err_ = ::dup(STDERR_FILENO);
    devnull_ = ::open("/dev/null", O_WRONLY);
    if (devnull_ >= 0) {
      if (saved_out_ >= 0) ::dup2(devnull_, STDOUT_FILENO);
      if (saved_err_ >= 0) ::dup2(devnull_, STDERR_FILENO);
    }
#endif
  }
  ~ScopedStdoutSilencer() {
#if defined(__unix__) || defined(__APPLE__)
    std::fflush(stdout);
    std::fflush(stderr);
    if (saved_out_ >= 0) {
      ::dup2(saved_out_, STDOUT_FILENO);
      ::close(saved_out_);
    }
    if (saved_err_ >= 0) {
      ::dup2(saved_err_, STDERR_FILENO);
      ::close(saved_err_);
    }
    if (devnull_ >= 0) ::close(devnull_);
#endif
  }
  ScopedStdoutSilencer(const ScopedStdoutSilencer&) = delete;
  ScopedStdoutSilencer& operator=(const ScopedStdoutSilencer&) = delete;

 private:
  int saved_out_ = -1;
  int saved_err_ = -1;
  int devnull_ = -1;
};

}  // namespace detail

// Mirrors `HypergraphPartitioning.initial_distribute`: builds the KaHyPar
// hypergraph from `dist_circ.kahypar_hyperedges()` (unit hyperedge weights,
// weight-1 qubit vertices, weight-0 gate vertices) and partitions it into
// `|get_server_list()|` blocks with per-server custom target weights, using
// the given INI configuration. Returns a Placement mapping vertex i to its
// KaHyPar block id (the reference uses block ids directly as server ids).
//
// Fidelity: this is the exact sequence of calls made by the reference through
// the Python `kahypar` module (`Context.loadINIconfiguration` /
// `setK` / `setCustomTargetBlockWeights` / `setSeed` / `partition` /
// `blockID`), via the C facade `libkahypar`. `epsilon` is left at the Context
// default (`std::numeric_limits<double>::max()`), matching the reference which
// never sets it; the INI's `suppressOutput` is cosmetic and has no effect on
// the result.
inline Placement hypergraph_partitioning_initial_placement(
    const HypergraphCircuit& dist_circ,
    const std::vector<std::pair<int, int>>& server_coupling,
    const std::vector<std::pair<int, std::vector<int>>>& server_qubits,
    const std::string& ini_path,
    std::optional<std::int64_t> seed) {
  if (!dist_circ.is_valid()) {
    throw std::runtime_error("This hypergraph is not valid.");
  }

  // Empty circuit: the reference returns an empty placement (and asserts the
  // underlying circuit is the empty circuit).
  if (dist_circ.hyperedge_list().empty()) {
    return Placement();
  }

  const auto [hyperedge_indices, hyperedges] = dist_circ.kahypar_hyperedges();
  const int num_hyperedges = static_cast<int>(hyperedge_indices.size()) - 1;
  const std::set<Vertex> unique_vertices(hyperedges.begin(), hyperedges.end());
  const int num_vertices = static_cast<int>(unique_vertices.size());

  // The KaHyPar facade expects `size_t` hyperedge indices; the core's
  // `kahypar_hyperedges` produces `int` indices (the reference's Python
  // wrapper converts its list of ints to `HyperedgeIndexVector` = size_t).
  std::vector<size_t> indices(hyperedge_indices.begin(), hyperedge_indices.end());

  const std::vector<int> server_list = server_list_order(server_coupling);
  const int num_servers = static_cast<int>(server_list.size());

  std::vector<kahypar_hypernode_weight_t> server_sizes;
  server_sizes.reserve(server_list.size());
  for (const int s : server_list) {
    server_sizes.push_back(static_cast<kahypar_hypernode_weight_t>(
        qubit_list_of(server_qubits, s).size()));
  }

  // Qubit vertices are given weight 1, gate vertices weight 0.
  const int num_qubits =
      static_cast<int>(dist_circ.get_qubit_vertices().size());
  std::vector<kahypar_hypernode_weight_t> vertex_weights(num_vertices, 0);
  for (int i = 0; i < num_vertices && i < num_qubits; ++i) {
    vertex_weights[static_cast<std::size_t>(i)] = 1;
  }
  // All hyperedges are assumed to have weight 1.
  std::vector<kahypar_hyperedge_weight_t> hyperedge_weights(num_hyperedges, 1);

  kahypar_context_t* context = kahypar_context_new();
  if (context == nullptr) {
    throw std::runtime_error("KaHyPar: could not allocate a context.");
  }
  kahypar_configure_context_from_file(context, ini_path.c_str());
  // setK(num_servers) is applied by kahypar_partition_hypergraph below.
  kahypar_set_custom_target_block_weights(
      num_servers, server_sizes.data(), context);
  if (seed.has_value()) {
    kahypar_set_seed(context, static_cast<int>(*seed));
  }

  kahypar_hypergraph_t* hg = kahypar_create_hypergraph(
      num_servers, static_cast<kahypar_hypernode_id_t>(num_vertices),
      static_cast<kahypar_hyperedge_id_t>(num_hyperedges),
      indices.data(),
      reinterpret_cast<const kahypar_hyperedge_id_t*>(hyperedges.data()),
      hyperedge_weights.data(), vertex_weights.data());
  if (hg == nullptr) {
    kahypar_context_free(context);
    throw std::runtime_error("KaHyPar: could not create the hypergraph.");
  }

  std::vector<kahypar_partition_id_t> partition(num_vertices, 0);
  kahypar_hyperedge_weight_t objective = 0;
  {
    detail::ScopedStdoutSilencer silence;
    kahypar_partition_hypergraph(
        hg, num_servers, std::numeric_limits<double>::max(), &objective, context,
        partition.data());
  }

  Placement placement;
  for (int i = 0; i < num_vertices; ++i) {
    placement.set(i, partition[static_cast<std::size_t>(i)]);
  }

  kahypar_hypergraph_free(hg);
  kahypar_context_free(context);
  return placement;
}

// Mirrors `HypergraphPartitioning.make_valid`: KaHyPar does not guarantee the
// per-server capacity constraint, so any qubit vertex on an over-capacity
// server is moved to the first server (in `server_qubits` insertion order,
// matching the reference's `occupancy.keys()` order) where the move is valid.
// Qubit vertices are visited in CPython frozenset order (the reference's
// `GainManager.qubit_vertices`), reproduced via `PySet<int>`.
inline void hypergraph_partitioning_make_valid(
    Distribution& distribution,
    const std::vector<std::pair<int, std::vector<int>>>& server_qubits) {
  GainManager gain_manager(distribution);

  std::vector<int> server_order;
  server_order.reserve(server_qubits.size());
  for (const auto& [s, qs] : server_qubits) {
    (void)qs;
    server_order.push_back(s);
  }

  PySet<Vertex> qubit_vertices(&py_hash_key, &py_eq_key);
  for (const Vertex v : distribution.circuit.get_qubit_vertices()) {
    qubit_vertices.add(v);
  }

  for (const Vertex vertex : qubit_vertices.iteration_order()) {
    if (!gain_manager.is_move_valid(vertex,
                                    gain_manager.current_server(vertex))) {
      for (const int server : server_order) {
        if (gain_manager.is_move_valid(vertex, server)) {
          gain_manager.move_vertex(vertex, server);
          break;
        }
      }
    }
  }

  // The reference asserts `distribution.is_valid()` afterwards.
  if (!distribution.is_valid()) {
    throw std::runtime_error(
        "HypergraphPartitioning.make_valid left the distribution invalid.");
  }
}

// Mirrors `HypergraphPartitioning.allocate`: KaHyPar initial placement
// followed by `make_valid`, returning the resulting Distribution.
// `server_graph` is the server-coupling graph (used by the GainManager inside
// `make_valid`).
inline Distribution hypergraph_partitioning_allocate(
    const HypergraphCircuit& dist_circ,
    const std::vector<std::pair<int, int>>& server_coupling,
    const std::vector<std::pair<int, std::vector<int>>>& server_qubits,
    const Graph& server_graph, const std::map<int, int>& server_ebit_mem,
    const std::string& ini_path, std::optional<std::int64_t> seed) {
  check_can_implement(dist_circ, server_qubits);

  Placement placement = hypergraph_partitioning_initial_placement(
      dist_circ, server_coupling, server_qubits, ini_path, seed);

  Distribution distribution = make_distribution(
      dist_circ, placement, server_graph, server_qubits, server_ebit_mem);
  hypergraph_partitioning_make_valid(distribution, server_qubits);
  return distribution;
}

}  // namespace dqc
