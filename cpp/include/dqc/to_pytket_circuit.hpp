#pragma once

#include <algorithm>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "dqc/circuit.hpp"
#include "dqc/circuit_analysis.hpp"
#include "dqc/distribution.hpp"
#include "dqc/graph.hpp"
#include "dqc/graph_tools.hpp"
#include "dqc/hypergraph_circuit.hpp"
#include "dqc/numeric.hpp"
#include "dqc/op_type.hpp"
#include "dqc/placement.hpp"

namespace dqc {

// Result of `to_pytket_circuit`. The core emits a serialized circuit (the
// data-only form the bindings turn back into a `pytket.Circuit`), together
// with the estimated cost (`Distribution.cost()`) and the *accurate* cost
// (`ebit_cost` of the final circuit).
struct ToPytketCircuitResult {
  SerializedCircuit circuit;
  int estimated_cost = 0;
  int real_ebit_cost = 0;
  // Set when `real_ebit_cost != estimated_cost` but the discrepancy is
  // explained by a gate that is both detached and embedded (the upstream
  // `warnings.warn` path, as opposed to the hard raise).
  bool cost_mismatch_detached_embedded = false;
};

namespace tpc_internal {

// Structural tags carried by `BuiltCommand`. The op *name* is only needed when
// a command is emitted or copied verbatim, so these cases are tagged once
// instead of being re-detected with string prefixes on every scan.
enum : std::uint8_t {
  kCmdStartProc = 1,
  kCmdEndProc = 2,
  kCmdBarrier = 4,
};

// A command in the circuit being incrementally built, in *interned* form.
// Qubit ids index the shared `TPCTables::qubits` table and `op` indexes
// `TPCTables::ops`, so the struct is trivially copyable: copying the command
// list per hyperedge no longer allocates or deep-copies strings.
// Commands of arity > 2 are stored either as a prefix of the circuit's qubit
// order (`kCmdBarrier` + `barrier_n`) or in `TPCTables::wide_qubits`
// (`wide`); parameter lists longer than two values live in
// `TPCTables::param_lists` (`params_idx`).
struct BuiltCommand {
  int op = -1;            // interned op name (only read when emitting)
  int op_type = -1;
  int q0 = -1;            // qubit id
  int q1 = -1;            // qubit id
  int start_origin = -1;  // qubit id; only meaningful for starting_process
  int wide = -1;          // TPCTables::wide_qubits entry when arity > 2
  int params_idx = -1;    // TPCTables::param_lists entry when n_params > 2
  int barrier_n = 0;      // kCmdBarrier: number of leading circuit qubits
  double p0 = 0.0;
  double p1 = 0.0;
  int n_params = 0;
  std::uint8_t arity = 0;
  std::uint8_t flags = 0;
};

// A tket `UnitID`: register name + index vector, compared by (name, index).
// This drives the within-slice ordering of pytket's `get_commands()`.
struct UnitID {
  std::string name;
  std::vector<unsigned> index;
};

inline bool operator<(const UnitID& a, const UnitID& b) {
  const int n = a.name.compare(b.name);
  if (n != 0) return n < 0;
  return a.index < b.index;
}

// Parse a qubit's string form `server_0[0]` (or `server_0_link_register[0]`)
// into its register name and index vector.
inline UnitID parse_unit_id(const std::string& q) {
  UnitID id;
  const std::size_t br = q.find('[');
  id.name = (br == std::string::npos) ? q : q.substr(0, br);
  if (br != std::string::npos && q.size() >= br + 2 && q.back() == ']') {
    const std::string inner = q.substr(br + 1, q.size() - br - 2);
    const std::vector<std::string> parts = split_str(inner, ',');
    for (const std::string& p : parts) {
      if (!p.empty()) id.index.push_back(static_cast<unsigned>(std::stoul(p)));
    }
  }
  return id;
}

// Shared, append-only interning tables for one `to_pytket_circuit` attempt.
// Interning qubit names means ids stay valid across every `BuiltCircuit`
// built during the attempt (so commands can be moved between circuits without
// touching strings), and per-qubit metadata is parsed once at intern time
// instead of on every use.
struct QubitTable {
  std::vector<std::string> names;
  std::vector<UnitID> unit_ids;
  std::vector<int> server_ids;
  std::vector<unsigned char> is_link;
  std::unordered_map<std::string, int> index;

  int intern(const std::string& q) {
    const auto it = index.find(q);
    if (it != index.end()) return it->second;
    const int id = static_cast<int>(names.size());
    index.emplace(q, id);
    names.push_back(q);
    unit_ids.push_back(parse_unit_id(q));
    server_ids.push_back(get_server_id(q));
    is_link.push_back(is_link_qubit(q) ? 1 : 0);
    return id;
  }
  int find(const std::string& q) const {
    const auto it = index.find(q);
    return it == index.end() ? -1 : it->second;
  }
};

struct OpTable {
  std::vector<std::string> names;
  std::unordered_map<std::string, int> index;

  int intern(const std::string& s) {
    const auto it = index.find(s);
    if (it != index.end()) return it->second;
    const int id = static_cast<int>(names.size());
    index.emplace(s, id);
    names.push_back(s);
    return id;
  }
};

struct TPCTables {
  QubitTable qubits;
  OpTable ops;
  // Rare escape hatches: commands with more than two qubits (other than
  // barriers) or more than two parameters.
  std::vector<std::vector<int>> wide_qubits;
  std::vector<std::vector<double>> param_lists;
  // Cached ids of the op names the builder emits itself.
  int op_h = -1;
  int op_rz = -1;
  int op_cu1 = -1;
  int op_cz = -1;
  int op_start = -1;
  int op_end = -1;

  void init_std() {
    if (op_h >= 0) return;
    op_h = ops.intern("H");
    op_rz = ops.intern("Rz");
    op_cu1 = ops.intern("CU1");
    op_cz = ops.intern("CZ");
    op_start = ops.intern("starting_process");
    op_end = ops.intern("ending_process");
  }
  int intern_wide(std::vector<int> v) {
    wide_qubits.push_back(std::move(v));
    return static_cast<int>(wide_qubits.size()) - 1;
  }
  int intern_params(std::vector<double> v) {
    param_lists.push_back(std::move(v));
    return static_cast<int>(param_lists.size()) - 1;
  }
};

// A wire on a qubit's path: source/target vertex index, or -1 for the
// input/output boundary.
struct Wire {
  int id;
  int qubit;
  int source;
  int target;
};

struct BuiltVertex {
  BuiltCommand cmd;
  std::vector<int> in_wires;   // port order
  std::vector<int> out_wires;  // port order
};

// Compact, derived-state-free copy of a `BuiltCircuit` (qubit ids in first-seen
// order plus the add-order command list). The DAG (vertices/wires/frontiers)
// is recomputed by `BuiltCircuit::restore`, so nothing derived is stored. The
// interning tables are append-only for the whole attempt, so the ids stay
// valid after a rollback.
struct BuiltCircuitSnapshot {
  std::vector<int> qubit_order;
  std::vector<BuiltCommand> commands;
};

// Resume checkpoint interval for the `allow_update` path: the builder state is
// snapshotted every this-many hyperedges so that a hyperedge split can resume
// from the closest checkpoint instead of restarting the whole pass.
inline constexpr std::size_t kResumeStride = 8;

// Circuit builder used by `to_pytket_circuit`. `commands` records the add
// order (mirroring the sequence of pytket `add_gate`/`add_barrier` calls in
// the reference); `rebuild_dag()` materialises the DAG that lets
// `get_commands_order()` reproduce pytket's `Circuit.get_commands()` order
// (slices in temporal order, within a slice by UnitID).
struct BuiltCircuit {
  TPCTables* tables = nullptr;         // shared, append-only for the attempt
  std::vector<int> qubit_order;        // qubit ids, first-seen order
  std::vector<unsigned char> present;  // id -> present in this circuit
  std::vector<int> qubit_first_wire;   // id -> input wire
  std::vector<int> frontier_wire;      // id -> current tip wire
  std::vector<BuiltCommand> commands;  // add order (barriers included)
  std::vector<BuiltVertex> vertices;
  std::vector<Wire> wires;
  int wire_seq = 0;
  // Scratch stamps for `get_commands_order` (see there). Reused across calls
  // instead of being reallocated/cleared once per slice.
  mutable std::vector<int> in_edge_stamp_;
  mutable std::vector<int> in_slice_stamp_;
  mutable std::vector<int> bad_stamp_;
  mutable int epoch_ = 0;

  explicit BuiltCircuit(TPCTables* t = nullptr) : tables(t) {}

  const std::string& qubit_name(int id) const {
    return tables->qubits.names[id];
  }
  int qubit_server(int id) const { return tables->qubits.server_ids[id]; }
  bool qubit_is_link(int id) const { return tables->qubits.is_link[id] != 0; }
  const std::string& op_name(int id) const { return tables->ops.names[id]; }
  std::vector<double> params_of(const BuiltCommand& c) const {
    if (c.params_idx >= 0) return tables->param_lists[c.params_idx];
    std::vector<double> out;
    if (c.n_params > 0) out.push_back(c.p0);
    if (c.n_params > 1) out.push_back(c.p1);
    return out;
  }
  void add_qubit_id(int id) {
    if (static_cast<int>(present.size()) <= id) present.resize(id + 1, 0);
    if (present[id] == 0) {
      present[id] = 1;
      qubit_order.push_back(id);
    }
  }
  int add_qubit(const std::string& name) {
    const int id = tables->qubits.intern(name);
    add_qubit_id(id);
    return id;
  }
  // Number of qubits a command acts on.
  int cmd_arity(const BuiltCommand& c) const {
    if ((c.flags & kCmdBarrier) != 0) return c.barrier_n;
    if (c.wide >= 0) {
      return static_cast<int>(tables->wide_qubits[c.wide].size());
    }
    return c.arity;
  }
  // Qubit id of port `p` of a command, in port order.
  int cmd_qubit(const BuiltCommand& c, int p) const {
    if ((c.flags & kCmdBarrier) != 0) return qubit_order[p];
    if (c.wide >= 0) return tables->wide_qubits[c.wide][p];
    return p == 0 ? c.q0 : c.q1;
  }
  // Mirror `new_circ.add_barrier(new_circ.qubits)`: a barrier over every qubit
  // currently in the circuit. `qubit_order` is append-only, so it is enough to
  // record the current length instead of copying every qubit.
  void push_barrier_over_all() {
    BuiltCommand b;
    b.op_type = kOpBarrier;
    b.flags = kCmdBarrier;
    b.barrier_n = static_cast<int>(qubit_order.size());
    commands.push_back(b);
  }
  // Rebuild the DAG (vertices/wires/frontiers) from `qubit_order` + `commands`
  // in add order, exactly as pytket's `add_gate`/`add_barrier` wire up the DAG.
  void rebuild_dag() {
    vertices.clear();
    wires.clear();
    wire_seq = 0;
    const std::size_t n_qubits = tables->qubits.names.size();
    qubit_first_wire.assign(n_qubits, -1);
    frontier_wire.assign(n_qubits, -1);
    for (const int qi : qubit_order) {
      const int w = wire_seq++;
      wires.push_back({w, qi, -1, -1});
      qubit_first_wire[qi] = w;
      frontier_wire[qi] = w;
    }
    for (const BuiltCommand& cmd : commands) {
      const int vidx = static_cast<int>(vertices.size());
      const int arity = cmd_arity(cmd);
      std::vector<int> in;
      std::vector<int> out;
      in.reserve(arity);
      out.reserve(arity);
      for (int p = 0; p < arity; ++p) {
        const int qi = cmd_qubit(cmd, p);
        const int w = frontier_wire[qi];
        if (w < 0) {
          throw std::runtime_error("rebuild_dag: unknown qubit " +
                                   qubit_name(qi));
        }
        in.push_back(w);
        wires[w].target = vidx;
      }
      vertices.push_back({cmd, in, {}});
      for (int p = 0; p < arity; ++p) {
        const int qi = cmd_qubit(cmd, p);
        const int w = wire_seq++;
        wires.push_back({w, qi, vidx, -1});
        out.push_back(w);
        frontier_wire[qi] = w;
      }
      vertices[vidx].out_wires = std::move(out);
    }
  }
  // State snapshot used by the `allow_update` resume path (see
  // `to_pytket_circuit`). Only the source data is stored: restoring rebuilds
  // the DAG, which is a pure function of `qubit_order` + `commands`.
  BuiltCircuitSnapshot snapshot() const { return {qubit_order, commands}; }
  void restore(const BuiltCircuitSnapshot& s) {
    qubit_order = s.qubit_order;
    present.assign(tables->qubits.names.size(), 0);
    for (const int id : qubit_order) present[id] = 1;
    commands = s.commands;
    rebuild_dag();
  }
  // Reproduce pytket's `Circuit.get_commands()` order: iterate slices
  // (frontier cuts) and, within each slice, order vertices by the first
  // reaching frontier edge, which is in UnitID order. Returns vertex indices.
  std::vector<int> get_commands_order() const {
    std::vector<int> qorder = qubit_order;
    std::sort(qorder.begin(), qorder.end(), [&](int a, int b) {
      return tables->qubits.unit_ids[a] < tables->qubits.unit_ids[b];
    });
    std::vector<int> frontier = qubit_first_wire;
    std::vector<int> order;
    // Per-slice bookkeeping is done with monotonic epoch stamps against these
    // persistent buffers instead of clearing whole O(wires)/O(vertices) arrays
    // once per slice: the slice loop runs once per barrier (~#CU1) and each
    // clear was O(circuit size), which dominated the whole build.
    if (in_edge_stamp_.size() < wires.size()) {
      in_edge_stamp_.resize(wires.size(), 0);
    }
    if (in_slice_stamp_.size() < vertices.size()) {
      in_slice_stamp_.resize(vertices.size(), 0);
    }
    if (bad_stamp_.size() < vertices.size()) {
      bad_stamp_.resize(vertices.size(), 0);
    }
    int& epoch = epoch_;
    for (;;) {
      ++epoch;
      for (const int qi : qorder) in_edge_stamp_[frontier[qi]] = epoch;
      std::vector<int> slice;
      for (const int qi : qorder) {
        const int v = wires[frontier[qi]].target;
        if (v < 0 || in_slice_stamp_[v] == epoch || bad_stamp_[v] == epoch) {
          continue;
        }
        bool good = true;
        for (const int iw : vertices[v].in_wires) {
          if (in_edge_stamp_[iw] != epoch) {
            good = false;
            break;
          }
        }
        if (!good) {
          bad_stamp_[v] = epoch;
          continue;
        }
        in_slice_stamp_[v] = epoch;
        slice.push_back(v);
      }
      if (slice.empty()) break;
      for (const int v : slice) order.push_back(v);
      for (const int qi : qorder) {
        const int v = wires[frontier[qi]].target;
        if (v >= 0 && in_slice_stamp_[v] == epoch) {
          const BuiltCommand& c = vertices[v].cmd;
          const int arity = cmd_arity(c);
          for (int p = 0; p < arity; ++p) {
            if (cmd_qubit(c, p) == qi) {
              frontier[qi] = vertices[v].out_wires[p];
              break;
            }
          }
        }
      }
    }
    return order;
  }
};

inline bool is_start_proc_cmd(const BuiltCommand& c) {
  return (c.flags & kCmdStartProc) != 0;
}

inline bool is_end_proc_cmd(const BuiltCommand& c) {
  return (c.flags & kCmdEndProc) != 0;
}

inline bool cmd_has_qubit(const BuiltCircuit& circ, const BuiltCommand& c,
                          int qubit) {
  const int arity = circ.cmd_arity(c);
  for (int p = 0; p < arity; ++p) {
    if (circ.cmd_qubit(c, p) == qubit) return true;
  }
  return false;
}

// Mirrors `origin_of_start_proc`: the origin is stored alongside the command.
inline int origin_of_start_proc(const BuiltCommand& c) {
  return c.start_origin;
}

// Mirrors `requires_embedding_start_proc(commands)` where `commands` is the
// slice `built_commands[begin:]`.
inline bool requires_embedding_start_proc(
    const BuiltCircuit& circ, const std::vector<BuiltCommand>& commands,
    std::size_t begin) {
  const int target_link = commands[begin].q1;
  for (std::size_t i = begin; i < commands.size(); ++i) {
    const BuiltCommand& g = commands[i];
    if (g.op_type == kOpCU1 && cmd_has_qubit(circ, g, target_link)) return true;
    if (is_end_proc_cmd(g) && g.q0 == target_link) return false;
  }
  throw std::runtime_error(
      "requires_embedding_start_proc: no conclusion reached");
}

// Mirrors `requires_embedding_end_proc(commands)` where `commands` is the
// slice `built_commands[:end]`.
inline bool requires_embedding_end_proc(
    const BuiltCircuit& circ, const std::vector<BuiltCommand>& commands,
    std::size_t end) {
  const int source_link = commands[end - 1].q0;
  for (std::size_t i = end; i-- > 0;) {
    const BuiltCommand& g = commands[i];
    if (g.op_type == kOpCU1 && cmd_has_qubit(circ, g, source_link)) return true;
    if (is_start_proc_cmd(g) && g.q1 == source_link) return false;
  }
  throw std::runtime_error(
      "requires_embedding_end_proc: no conclusion reached");
}

// Immutable context shared by the builder (captures the distribution's pieces
// and the qubit mapping, which is rebuilt before each retry).
struct TPCContext {
  const HypergraphCircuit& hyp_circ;
  const Placement& placement;
  const Graph& server_graph;
  const std::map<std::string, std::string>& qubit_mapping;
  const std::map<int, int>& server_ebit_mem;
  bool satisfy_bound;
  std::vector<int> server_list;
  TPCTables* tables;  // shared interning tables for this attempt

  const std::string& circuit_qubit_of_vertex(Vertex v) const {
    return hyp_circ.qubit_of_vertex(v);
  }
};

// Mirrors `Distribution.get_qubit_mapping`.
inline std::map<std::string, std::string> build_qubit_mapping(
    const Distribution& dist) {
  std::map<std::string, std::string> m;
  std::map<int, int> current_in_server;
  for (const Vertex v : dist.circuit.get_qubit_vertices()) {
    const int server = dist.placement.at(v);
    const int idx = current_in_server[server];
    m[dist.circuit.qubit_of_vertex(v)] =
        "server_" + std::to_string(server) + "[" + std::to_string(idx) + "]";
    current_in_server[server] += 1;
  }
  return m;
}

// Mirrors the nested `LinkManager` class, operating on interned qubit ids.
// `link_qubit_dict` is insertion ordered (CPython dict semantics) — NOT a
// std::map — because `connected_servers()` iterates its keys and drives
// `start_link` tie-breaks.
struct LinkManager {
  const TPCContext& ctx;
  BuiltCircuit& out;  // circuit being built; owns qubit interning + presence
  Hyperedge hyperedge;
  int home_id = -1;  // hardware qubit id of the hyperedge's circuit qubit
  std::vector<std::pair<int, int>> link_qubit_dict;  // (server, qubit id)
  std::map<int, std::vector<int>> available;
  std::map<int, std::vector<int>> occupied;
  std::map<int, int> link_qubit_id_update;
  // `steiner_tree(server_graph, servers_of(hyperedge))` is the same for every
  // CU1 gate of this hyperedge, so compute it lazily once instead of once per
  // `start_link` call. Pure caching: the value is identical.
  std::optional<Graph> hyperedge_tree_cache;

  LinkManager(const TPCContext& ctx, const Hyperedge& h,
              const std::vector<int>& servers, BuiltCircuit& out)
      : ctx(ctx), out(out), hyperedge(h) {
    for (const int s : servers) {
      available[s] = {};
      occupied[s] = {};
    }
    home_id = out.add_qubit(
        ctx.qubit_mapping.at(ctx.circuit_qubit_of_vertex(qubit_vertex())));
  }

  const Graph& hyperedge_tree() {
    if (!hyperedge_tree_cache.has_value()) {
      std::vector<int> hyp_servers;
      hyp_servers.reserve(hyperedge.vertices.size());
      for (const Vertex v : hyperedge.vertices) {
        hyp_servers.push_back(ctx.placement.at(v));
      }
      hyperedge_tree_cache = steiner_tree(ctx.server_graph, hyp_servers);
    }
    return *hyperedge_tree_cache;
  }

  Vertex qubit_vertex() const {
    return ctx.hyp_circ.get_qubit_vertex(hyperedge);
  }
  int home_server() const { return ctx.placement.at(qubit_vertex()); }
  const std::string& circuit_qubit() const {
    return ctx.circuit_qubit_of_vertex(qubit_vertex());
  }

  // Mirrors `_request_link_qubit`. Returns the qubit id (interning it into the
  // circuit if a fresh link qubit is needed).
  int request_link_qubit(int server) {
    int capacity = -1;
    auto it = ctx.server_ebit_mem.find(server);
    if (it != ctx.server_ebit_mem.end()) capacity = it->second;
    if (ctx.satisfy_bound && capacity > 0) {
      if (capacity <= static_cast<int>(occupied[server].size())) {
        throw ConstraintException(
            "Communication memory capacity of server " +
                std::to_string(server) +
                " exceeded. \n\tConsider setting allow_update to True: "
                "\n\tto_pytket_circuit(allow_update=True)",
            server);
      }
    }

    int qubit;
    if (available[server].empty()) {
      const int next_id = static_cast<int>(occupied[server].size());
      qubit = out.add_qubit("server_" + std::to_string(server) +
                            "_link_register[" + std::to_string(next_id) + "]");
    } else {
      qubit = available[server].back();
      available[server].pop_back();
    }
    occupied[server].push_back(qubit);
    return qubit;
  }

  // Mirrors `_release_link_qubit`.
  void release_link_qubit(int link_qubit) {
    const int server = out.qubit_server(link_qubit);
    auto& occ = occupied[server];
    occ.erase(std::find(occ.begin(), occ.end(), link_qubit));
    available[server].push_back(link_qubit);
  }

  // Mirrors `connected_servers()`: keys of link_qubit_dict in insertion order.
  std::vector<int> connected_servers() const {
    std::vector<int> out_servers;
    for (const auto& [s, q] : link_qubit_dict) {
      (void)q;
      out_servers.push_back(s);
    }
    return out_servers;
  }

  void set_link_qubit(int server, int q) {
    for (auto& [s, v] : link_qubit_dict) {
      if (s == server) {
        v = q;
        return;
      }
    }
    link_qubit_dict.push_back({server, q});
  }

  void erase_link_qubit(int server) {
    for (auto it = link_qubit_dict.begin(); it != link_qubit_dict.end(); ++it) {
      if (it->first == server) {
        link_qubit_dict.erase(it);
        return;
      }
    }
  }

  // Mirrors `get_link_qubit`.
  int get_link_qubit(int server) const {
    if (home_server() == server) {
      return home_id;
    }
    for (const auto& [s, q] : link_qubit_dict) {
      if (s == server) return q;
    }
    throw std::out_of_range("server not in link_qubit_dict");
  }

  // Mirrors `start_link`: returns the list of EjppAction (from, to) pairs.
  std::vector<std::pair<int, int>> start_link(int target) {
    const int q_vertex = qubit_vertex();
    const int home = ctx.placement.at(q_vertex);
    // Only depends on the hyperedge, so it is cached across `start_link` calls.
    const Graph& tree = hyperedge_tree();
    // assert target in hyp_servers (guaranteed by construction)

    std::vector<int> connected = connected_servers();
    connected.push_back(home);

    std::vector<int> best_path;
    for (const int c_server : connected) {
      const std::vector<int> path =
          bidirectional_dijkstra_path(tree, c_server, target);
      if (best_path.empty() || path.size() < best_path.size()) {
        best_path = path;
      }
    }
    // assert best_path is not None

    std::vector<std::pair<int, int>> actions;
    int last_link_qubit = get_link_qubit(best_path[0]);
    for (std::size_t i = 1; i < best_path.size(); ++i) {
      const int next_server = best_path[i];
      const int this_link_qubit = request_link_qubit(next_server);
      set_link_qubit(next_server, this_link_qubit);
      actions.push_back({last_link_qubit, this_link_qubit});
      last_link_qubit = this_link_qubit;
    }
    return actions;
  }

  // Mirrors `end_links`.
  std::vector<std::pair<int, int>> end_links(
      const std::vector<int>& targets) {
    const int home_link = home_id;
    std::vector<std::pair<int, int>> actions;
    for (const int target : targets) {
      const int target_link = get_link_qubit(target);
      actions.push_back({target_link, home_link});
      release_link_qubit(target_link);
      erase_link_qubit(target);
    }
    return actions;
  }

  // Mirrors `update_occupation`.
  void update_occupation(int link_qubit, bool starting) {
    if (starting) {
      const int server = out.qubit_server(link_qubit);
      const int new_link_qubit = request_link_qubit(server);
      link_qubit_id_update[link_qubit] = new_link_qubit;
    } else {
      const int new_link_qubit = link_qubit_id_update.at(link_qubit);
      release_link_qubit(new_link_qubit);
    }
  }

  // Mirrors `get_updated_name`.
  int get_updated_id(int q) const {
    auto it = link_qubit_id_update.find(q);
    if (it != link_qubit_id_update.end()) return it->second;
    return q;
  }
};

// Append a one-qubit gate command (`H`, `Rz`, ...).
inline void push1(BuiltCircuit& c, int op, int op_type, int q,
                  double param = 0.0, int n_params = 0) {
  BuiltCommand cmd;
  cmd.op = op;
  cmd.op_type = op_type;
  cmd.arity = 1;
  cmd.q0 = q;
  cmd.p0 = param;
  cmd.n_params = n_params;
  c.commands.push_back(cmd);
}

// Append a two-qubit gate command (`CU1`, `CZ`).
inline void push2(BuiltCircuit& c, int op, int op_type, int q0, int q1,
                  double param = 0.0, int n_params = 0) {
  BuiltCommand cmd;
  cmd.op = op;
  cmd.op_type = op_type;
  cmd.arity = 2;
  cmd.q0 = q0;
  cmd.q1 = q1;
  cmd.p0 = param;
  cmd.n_params = n_params;
  c.commands.push_back(cmd);
}

// `starting_process(from, to)` with the compute qubit it copies.
inline void push_start_proc(BuiltCircuit& c, int from, int to, int origin) {
  BuiltCommand cmd;
  cmd.op = c.tables->op_start;
  cmd.op_type = kOpCustomGate;
  cmd.flags = kCmdStartProc;
  cmd.arity = 2;
  cmd.q0 = from;
  cmd.q1 = to;
  cmd.start_origin = origin;
  c.commands.push_back(cmd);
}

inline void push_end_proc(BuiltCircuit& c, int from, int to) {
  BuiltCommand cmd;
  cmd.op = c.tables->op_end;
  cmd.op_type = kOpCustomGate;
  cmd.flags = kCmdEndProc;
  cmd.arity = 2;
  cmd.q0 = from;
  cmd.q1 = to;
  c.commands.push_back(cmd);
}

// Copy a command verbatim, renaming its qubits through the LinkManager.
inline BuiltCommand copy_cmd(BuiltCircuit& c, const BuiltCommand& src,
                             const LinkManager& lm) {
  BuiltCommand cmd = src;
  if ((src.flags & kCmdBarrier) == 0) {
    if (src.wide >= 0) {
      std::vector<int> qs = c.tables->wide_qubits[src.wide];
      for (int& q : qs) q = lm.get_updated_id(q);
      cmd.wide = c.tables->intern_wide(std::move(qs));
    } else {
      if (cmd.arity > 0) cmd.q0 = lm.get_updated_id(src.q0);
      if (cmd.arity > 1) cmd.q1 = lm.get_updated_id(src.q1);
    }
  }
  return cmd;
}

// Mirrors `to_pytket_circuit_one_hyperedge`.
inline BuiltCircuit one_hyperedge(const Hyperedge& hyperedge,
                                  const BuiltCircuit& circ,
                                  const TPCContext& ctx) {
  if (hyperedge.vertices.size() == 1) {
    return circ;
  }

  TPCTables& T = *ctx.tables;
  BuiltCircuit new_circ(ctx.tables);
  for (const auto& [orig, hw] : ctx.qubit_mapping) {
    (void)orig;
    new_circ.add_qubit(hw);
  }

  const Vertex q_vertex = ctx.hyp_circ.get_qubit_vertex(hyperedge);
  // `add_qubit` is idempotent, so this also yields the id of an already-added
  // hardware qubit.
  const int src_qubit = new_circ.add_qubit(
      ctx.qubit_mapping.at(ctx.circuit_qubit_of_vertex(q_vertex)));

  const std::vector<Vertex> gate_vertices =
      ctx.hyp_circ.get_gate_vertices(hyperedge);
  const Vertex min_gv =
      *std::min_element(gate_vertices.begin(), gate_vertices.end());
  const Vertex max_gv =
      *std::max_element(gate_vertices.begin(), gate_vertices.end());

  int v_gate = static_cast<int>(ctx.qubit_mapping.size()) - 1;
  bool currently_h_embedding = false;
  double carry_phase = 0.0;
  LinkManager lm(ctx, hyperedge, ctx.server_list, new_circ);

  // Scan in pytket's `get_commands()` order (slices, within a slice by
  // UnitID), NOT insertion order. The reference iterates `circ.get_commands()`
  // of the intermediate circuit, whose CU1 barriers pin the topological order;
  // the DAG in `circ` reproduces that exact order.
  std::vector<BuiltCommand> commands;
  for (const int vi : circ.get_commands_order()) {
    commands.push_back(circ.vertices[vi].cmd);
  }

  for (std::size_t cmd_idx = 0; cmd_idx < commands.size(); ++cmd_idx) {
    const BuiltCommand& cmd = commands[cmd_idx];

    if (cmd.op_type == kOpCU1) v_gate += 1;

    // Track link-qubit occupation across start/end processes (done for every
    // command, before the hyperedge-span early exit).
    if (is_start_proc_cmd(cmd)) {
      try {
        lm.update_occupation(cmd.q1, true);
      } catch (ConstraintException& e) {
        e.v_gate = v_gate;
        throw;
      }
      new_circ.add_qubit_id(lm.get_updated_id(cmd.q1));
    }
    if (is_end_proc_cmd(cmd)) {
      try {
        lm.update_occupation(cmd.q0, false);
      } catch (ConstraintException& e) {
        e.v_gate = v_gate;
        throw;
      }
    }

    // Early exit: commands before/after this hyperedge's gate span (and any
    // non-CU1 command at/after the last gate) are copied verbatim. Barriers
    // are re-added over ALL current qubits, mirroring the reference.
    if (v_gate < min_gv || (v_gate >= max_gv && cmd.op_type != kOpCU1)) {
      if (cmd.op_type == kOpBarrier) {
        new_circ.push_barrier_over_all();
      } else {
        new_circ.commands.push_back(copy_cmd(new_circ, cmd, lm));
      }
      continue;
    }

    if (cmd.op_type == kOpRz) {
      if (!cmd_has_qubit(circ, cmd, src_qubit)) {
        new_circ.commands.push_back(copy_cmd(new_circ, cmd, lm));
        continue;
      }
      const int q = cmd.q0;
      const double phase = cmd.p0;
      push1(new_circ, T.op_rz, kOpRz, q, phase, 1);
      if (currently_h_embedding) {
        const double p1 = py_mod(phase, 1.0);
        if (isclose(p1, 0.0) || isclose(p1, 1.0)) {
          for (const int server : lm.connected_servers()) {
            push1(new_circ, T.op_rz, kOpRz, lm.get_link_qubit(server), phase,
                  1);
          }
        } else {
          carry_phase += phase;
        }
      }
    } else if (cmd.op_type == kOpH) {
      if (!cmd_has_qubit(circ, cmd, src_qubit)) {
        new_circ.commands.push_back(copy_cmd(new_circ, cmd, lm));
        continue;
      }
      const int q = cmd.q0;

      if (!currently_h_embedding) {
        // Detect whether this H opens an embedding unit with content.
        int found_idx = -1;
        const BuiltCommand* found_cmd = nullptr;
        for (std::size_t i = cmd_idx + 1; i < commands.size(); ++i) {
          const BuiltCommand& g = commands[i];
          if (g.op_type == kOpCU1 && cmd_has_qubit(circ, g, q)) {
            found_idx = static_cast<int>(i);
            found_cmd = &g;
            break;
          } else if (is_start_proc_cmd(g) && g.start_origin == q &&
                     requires_embedding_start_proc(circ, commands, i)) {
            found_idx = static_cast<int>(i);
            found_cmd = &g;
            break;
          } else if (is_end_proc_cmd(g) && g.q1 == q &&
                     requires_embedding_end_proc(circ, commands, i + 1)) {
            found_idx = static_cast<int>(i);
            found_cmd = &g;
            break;
          } else if (g.op_type == kOpH && cmd_has_qubit(circ, g, q)) {
            break;
          }
        }

        if (found_cmd == nullptr) {
          // Case 1: empty embedding unit.
          for (const int server : lm.connected_servers()) {
            push1(new_circ, T.op_h, kOpH, lm.get_link_qubit(server));
          }
          currently_h_embedding = true;
        } else {
          // Case 2: embedding unit with content; keep only the remote server.
          int remote_qubit = -1;
          if (is_start_proc_cmd(*found_cmd)) {
            remote_qubit = found_cmd->q1;
          } else {
            const int arity = circ.cmd_arity(*found_cmd);
            for (int p = 0; p < arity; ++p) {
              const int rq = circ.cmd_qubit(*found_cmd, p);
              if (rq != q) remote_qubit = rq;
            }
          }
          if (found_cmd->op_type == kOpCU1 &&
              new_circ.qubit_is_link(remote_qubit)) {
            int original_remote = -1;
            for (std::size_t j = static_cast<std::size_t>(found_idx);
                 j < commands.size(); ++j) {
              const BuiltCommand& g = commands[j];
              if (is_end_proc_cmd(g) && g.q0 == remote_qubit) {
                original_remote = g.q1;
                break;
              }
            }
            if (original_remote < 0) {
              throw std::runtime_error(
                  "implementation of hyperedge failed: ending_process for a "
                  "link qubit was not found");
            }
            remote_qubit = original_remote;
          }
          const int remote_server = new_circ.qubit_server(remote_qubit);
          std::vector<int> end_servers;
          for (const int s : lm.connected_servers()) {
            if (s != remote_server) end_servers.push_back(s);
          }
          for (const auto& [from, to] : lm.end_links(end_servers)) {
            push_end_proc(new_circ, from, to);
          }
          for (const int server : lm.connected_servers()) {
            push1(new_circ, T.op_h, kOpH, lm.get_link_qubit(server));
          }
          currently_h_embedding = true;
        }
      } else {
        // Currently embedding.
        const double m2 = py_mod(carry_phase, 2.0);
        const double m1 = py_mod(carry_phase, 1.0);
        if (isclose(1.0 + m2, 1.0)) {
          // I gate: end embedding with H.
          currently_h_embedding = false;
          for (const int server : lm.connected_servers()) {
            push1(new_circ, T.op_h, kOpH, lm.get_link_qubit(server));
          }
        } else if (isclose(m2, 1.0)) {
          // Z gate: end embedding with Rz + H.
          currently_h_embedding = false;
          for (const int server : lm.connected_servers()) {
            const int lq = lm.get_link_qubit(server);
            push1(new_circ, T.op_rz, kOpRz, lq, carry_phase, 1);
            push1(new_circ, T.op_h, kOpH, lq);
          }
          carry_phase = 0.0;
        } else if (isclose(m1, 0.5)) {
          // S or S': still embedding.
          carry_phase = -carry_phase;
          currently_h_embedding = true;
        } else {
          throw std::runtime_error(
              "Implementation of hyperedge failed. It contains an H-type "
              "embedding with an internal phase that cannot be cancelled.");
        }
      }
      push1(new_circ, T.op_h, kOpH, q);
    } else if (cmd.op_type == kOpCU1) {
      if (!cmd_has_qubit(circ, cmd, src_qubit)) {
        new_circ.commands.push_back(copy_cmd(new_circ, cmd, lm));
        continue;
      }
      const double phase = cmd.p0;
      int rmt_qubit = -1;
      {
        const int arity = circ.cmd_arity(cmd);
        for (int p = 0; p < arity; ++p) {
          const int q = circ.cmd_qubit(cmd, p);
          if (q != src_qubit) rmt_qubit = q;
        }
      }

      if (std::find(hyperedge.vertices.begin(), hyperedge.vertices.end(),
                    v_gate) != hyperedge.vertices.end()) {
        // Distribute this gate.
        // assert !currently_h_embedding
        const int target_server = ctx.placement.at(v_gate);

        std::vector<std::pair<int, int>> start_actions;
        try {
          start_actions = lm.start_link(target_server);
        } catch (ConstraintException& e) {
          e.v_gate = v_gate;
          throw;
        }
        for (const auto& [from, to] : start_actions) {
          push_start_proc(new_circ, from, to, src_qubit);
        }
        push2(new_circ, T.op_cu1, kOpCU1, lm.get_link_qubit(target_server),
              lm.get_updated_id(rmt_qubit), phase, 1);

        if (v_gate == max_gv) {
          for (const auto& [from, to] :
               lm.end_links(lm.connected_servers())) {
            push_end_proc(new_circ, from, to);
          }
        }
      } else {
        new_circ.commands.push_back(copy_cmd(new_circ, cmd, lm));
        if (currently_h_embedding) {
          const double p1 = py_mod(phase, 1.0);
          (void)p1;  // assert isclose(p1,0) or isclose(p1,1)
          if (new_circ.qubit_is_link(rmt_qubit)) {
            int original_remote = -1;
            for (std::size_t j = cmd_idx; j < commands.size(); ++j) {
              const BuiltCommand& g = commands[j];
              if (is_end_proc_cmd(g) && g.q0 == rmt_qubit) {
                original_remote = g.q1;
                break;
              }
            }
            if (original_remote < 0) {
              throw std::runtime_error(
                  "implementation of hyperedge failed: ending_process for a "
                  "link qubit was not found");
            }
            rmt_qubit = original_remote;
          }
          for (const int server : lm.connected_servers()) {
            push2(new_circ, T.op_cz, kOpCZ, lm.get_link_qubit(server),
                  rmt_qubit);
          }
        }
      }
    } else if (is_start_proc_cmd(cmd)) {
      const int orig_qubit = cmd.start_origin;
      const int remote_qubit = lm.get_updated_id(cmd.q1);
      new_circ.commands.push_back(copy_cmd(new_circ, cmd, lm));
      if (currently_h_embedding && src_qubit == orig_qubit) {
        if (requires_embedding_start_proc(circ, commands, cmd_idx)) {
          for (const int server : lm.connected_servers()) {
            const int lq = lm.get_link_qubit(server);
            push1(new_circ, T.op_h, kOpH, remote_qubit);
            push2(new_circ, T.op_cz, kOpCZ, remote_qubit, lq);
            push1(new_circ, T.op_h, kOpH, remote_qubit);
          }
        }
      }
    } else if (is_end_proc_cmd(cmd)) {
      const int orig_qubit = lm.get_updated_id(cmd.q1);
      const int remote_qubit = lm.get_updated_id(cmd.q0);
      if (currently_h_embedding && src_qubit == orig_qubit) {
        if (requires_embedding_end_proc(circ, commands, cmd_idx + 1)) {
          for (const int server : lm.connected_servers()) {
            const int lq = lm.get_link_qubit(server);
            push1(new_circ, T.op_h, kOpH, remote_qubit);
            push2(new_circ, T.op_cz, kOpCZ, remote_qubit, lq);
            push1(new_circ, T.op_h, kOpH, remote_qubit);
          }
        }
      }
      new_circ.commands.push_back(copy_cmd(new_circ, cmd, lm));
    } else if (cmd.op_type == kOpCZ) {
      new_circ.commands.push_back(copy_cmd(new_circ, cmd, lm));
    } else if (cmd.op_type == kOpBarrier) {
      // Mirror the reference: re-add the barrier over all qubits currently in
      // the circuit. Barriers are scaffolding only (removed from the emitted
      // circuit), but they pin the topological order of the intermediate
      // circuits, exactly like pytket's barriers.
      new_circ.push_barrier_over_all();
    } else {
      throw std::runtime_error("Command not supported in to_pytket_circuit.");
    }
  }

  // Materialise the DAG of the rebuilt circuit so the next hyperedge can scan
  // it in pytket's `get_commands()` order.
  new_circ.rebuild_dag();
  return new_circ;
}

// Mirrors the `allow_update` fallback: choose a hyperedge to split and mutate
// `dist.circuit` in place. Returns the chosen hyperedge (informational; the
// production caller in `to_pytket_circuit` ignores it). When `out_index` is
// given it receives the chosen hyperedge's index in `hyperedge_list()` *before*
// the split, which is exactly the position where the two halves are inserted,
// i.e. the earliest index whose build result can change.
inline std::optional<Hyperedge> split_for_bound(Distribution& dist,
                                                const ConstraintException& e,
                                                std::size_t* out_index =
                                                    nullptr) {
  const HypergraphCircuit& hyp_circ = dist.circuit;
  const Placement& placement = dist.placement;
  const Graph& server_graph = dist.server_graph;

  std::vector<Hyperedge> hedges;
  for (const Hyperedge& h : hyp_circ.hyperedge_list()) {
    if (h.vertices.size() <= 2) continue;
    const std::vector<Vertex> gates = hyp_circ.get_gate_vertices(h);
    bool before = false;
    bool after = false;
    for (const Vertex v : gates) {
      if (v <= e.v_gate) before = true;
      if (v >= e.v_gate) after = true;
    }
    if (before && after) hedges.push_back(h);
  }

  std::optional<Hyperedge> chosen;
  int longest_dist = -1;
  for (const Hyperedge& h : hedges) {
    const std::vector<Vertex> gates = hyp_circ.get_gate_vertices(h);
    // Mirrors the reference: `v_prev = max([v for v in gates if v < e.v_gate])`
    // (default `e.v_gate` if none) and `v_post = min([v for v in gates if
    // v > e.v_gate])` (default `e.v_gate` if none). NOTE: the reference
    // filters first and takes the max/min of the survivors; a per-element
    // `max(v_prev, v)` starting from `e.v_gate` would never move because every
    // `v` is on the wrong side of the sentinel.
    Vertex v_prev = e.v_gate;
    bool have_prev = false;
    for (const Vertex v : gates) {
      if (v < e.v_gate && (!have_prev || v > v_prev)) {
        v_prev = v;
        have_prev = true;
      }
    }
    Vertex v_post = e.v_gate;
    bool have_post = false;
    for (const Vertex v : gates) {
      // `gates` need not be sorted (e.g. a hyperedge built via `merge_hyperedge`).
      if (v > e.v_gate && (!have_post || v < v_post)) {
        v_post = v;
        have_post = true;
      }
    }
    const int d = v_post - v_prev;
    if (longest_dist < d) {
      std::vector<int> h_servers;
      for (const Vertex v : h.vertices) h_servers.push_back(placement.at(v));
      const Graph tree = steiner_tree(server_graph, h_servers);
      if (tree.has_node(e.server)) {
        longest_dist = d;
        chosen = h;
      }
    }
  }

  if (!chosen.has_value()) {
    std::vector<int> home_servers;
    for (const Hyperedge& h :
         hyp_circ.get_hyperedges_containing({e.v_gate})) {
      home_servers.push_back(
          placement.at(hyp_circ.get_qubit_vertex(h)));
    }
    const int placed_server = placement.at(e.v_gate);
    if (std::find(home_servers.begin(), home_servers.end(), placed_server) ==
        home_servers.end()) {
      const int cap = dist.server_ebit_mem.at(placed_server);
      if (cap < 2) {
        throw ConstraintException(
            "Could not implement the gate " + std::to_string(e.v_gate) +
                " sinceit is placed on a non-home server " +
                std::to_string(placed_server) +
                " and said server has memory capacity less than two. "
                "Consider using an approach that does not produce evicted gates, "
                "or increase the communication memory of server " +
                std::to_string(placed_server) + ".",
            placed_server);
      }
    }
    throw ConstraintException(
        "The distribution could not be amended to satisfy the bound on "
        "communication capacity.",
        e.server);
  }

  const Hyperedge& chosen_hedge = *chosen;
  if (out_index != nullptr) {
    // Same index `split_hyperedge` will use (first match in the list), so the
    // hyperedges before it are provably untouched by the split.
    *out_index = hyp_circ.index_in_list(chosen_hedge);
  }
  const Vertex v_q = hyp_circ.get_qubit_vertex(chosen_hedge);
  std::vector<Vertex> gates = hyp_circ.get_gate_vertices(chosen_hedge);

  std::vector<Vertex> prev_vertices;
  std::vector<Vertex> post_vertices;
  for (const Vertex v : gates) {
    if (v <= e.v_gate) prev_vertices.push_back(v);
  }
  for (const Vertex v : gates) {
    if (v > e.v_gate) post_vertices.push_back(v);
  }
  if (post_vertices.empty()) {
    const Vertex last_prev = prev_vertices.back();
    prev_vertices.pop_back();
    // assert last_prev == e.v_gate
    post_vertices.push_back(last_prev);
  }

  std::vector<Vertex> prev_hedge_v = {v_q};
  prev_hedge_v.insert(prev_hedge_v.end(), prev_vertices.begin(),
                      prev_vertices.end());
  std::vector<Vertex> post_hedge_v = {v_q};
  post_hedge_v.insert(post_hedge_v.end(), post_vertices.begin(),
                      post_vertices.end());

  dist.circuit.split_hyperedge(
      chosen_hedge,
      {Hyperedge(prev_hedge_v, 1), Hyperedge(post_hedge_v, 1)});
  return *chosen;
}

}  // namespace tpc_internal

// Mirrors `Distribution.to_pytket_circuit(satisfy_bound, allow_update)`.
inline ToPytketCircuitResult to_pytket_circuit(Distribution& dist,
                                               bool satisfy_bound = true,
                                               bool allow_update = false) {
  if (!dist.is_valid()) {
    throw std::runtime_error("The distribution of the circuit is not valid!");
  }

  while (true) {
    std::map<std::string, std::string> qubit_mapping =
        tpc_internal::build_qubit_mapping(dist);
    // Interning tables for this attempt: `new_circ`, every per-hyperedge
    // circuit derived from it and the checkpoints all share them, so ids stay
    // valid across copies and rollbacks.
    tpc_internal::TPCTables tables;
    tables.init_std();
    tpc_internal::TPCContext ctx{
        dist.circuit,      dist.placement,
        dist.server_graph, qubit_mapping,
        dist.server_ebit_mem, satisfy_bound,
        dist.get_server_list(), &tables};

    tpc_internal::BuiltCircuit new_circ(&tables);
    for (const auto& [orig, hw] : qubit_mapping) {
      (void)orig;
      new_circ.add_qubit(hw);
    }
    // Mirror the reference: build the initial circuit from the serialized
    // commands and insert a full barrier after every CU1 gate. The barriers
    // pin the CU1 ordering so pytket's `get_commands()` is deterministic.
    for (const Command& cmd : dist.circuit.get_circuit().commands) {
      tpc_internal::BuiltCommand out;
      out.op = tables.ops.intern(cmd.op_name);
      out.op_type = cmd.op_type;
      if (cmd.op_name.rfind("starting_process", 0) == 0) {
        out.flags |= tpc_internal::kCmdStartProc;
      } else if (cmd.op_name.rfind("ending_process", 0) == 0) {
        out.flags |= tpc_internal::kCmdEndProc;
      }
      std::vector<int> qids;
      qids.reserve(cmd.qubits.size());
      for (const std::string& q : cmd.qubits) {
        qids.push_back(new_circ.add_qubit(qubit_mapping.at(q)));
      }
      if (qids.size() <= 2) {
        out.arity = static_cast<std::uint8_t>(qids.size());
        if (!qids.empty()) out.q0 = qids[0];
        if (qids.size() > 1) out.q1 = qids[1];
      } else {
        out.arity = 0;
        out.wide = tables.intern_wide(std::move(qids));
      }
      out.n_params = static_cast<int>(cmd.params.size());
      if (!cmd.params.empty()) out.p0 = cmd.params[0];
      if (cmd.params.size() > 1) out.p1 = cmd.params[1];
      if (cmd.params.size() > 2) {
        out.params_idx = tables.intern_params(cmd.params);
      }
      new_circ.commands.push_back(out);
      if (cmd.op_type == kOpCU1) {
        new_circ.push_barrier_over_all();
      }
    }
    new_circ.rebuild_dag();

    // Walk the hyperedges one at a time. The reference splits *one* hyperedge
    // and then restarts the whole build from the initial circuit. That restart
    // is unobservable, though: splitting a hyperedge only rewrites positions
    // >= its own index in `hyperedge_list()`, the builder is deterministic, and
    // every hyperedge's work depends only on the circuit built so far. So on a
    // constraint violation we resume from the earliest index whose result can
    // have changed (replaying from the closest checkpoint) instead of redoing
    // the entire pass. The sequence of splits — and therefore the final
    // hypergraph and circuit — is exactly the reference's.
    std::vector<tpc_internal::BuiltCircuitSnapshot> checkpoints;
    std::size_t start = 0;
    for (;;) {
      const std::vector<Hyperedge>& hedges = dist.circuit.hyperedge_list();
      bool split_happened = false;
      for (std::size_t j = start; j < hedges.size(); ++j) {
        if (j % tpc_internal::kResumeStride == 0 &&
            j / tpc_internal::kResumeStride == checkpoints.size()) {
          checkpoints.push_back(new_circ.snapshot());
        }
        // A hyperedge holding only its qubit vertex is a no-op: the reference
        // returns the circuit unchanged. Skip the call rather than copying the
        // whole circuit out and back in.
        if (hedges[j].vertices.size() == 1) continue;
        try {
          new_circ = tpc_internal::one_hyperedge(hedges[j], new_circ, ctx);
        } catch (ConstraintException& e) {
          if (!allow_update) throw;
          std::size_t split_index = j;
          tpc_internal::split_for_bound(dist, e, &split_index);
          // Everything before the split (and before the failing hyperedge,
          // whose partial work was discarded) is still valid.
          const std::size_t resume = std::min(j, split_index);
          if (resume < j) {
            // Roll back to the nearest checkpoint and replay up to `resume`.
            // The replayed hyperedges are the same ones that just succeeded
            // with the same prefix, so this cannot fail.
            const std::size_t cp = resume / tpc_internal::kResumeStride;
            checkpoints.resize(cp + 1);
            new_circ.restore(checkpoints[cp]);
            const std::vector<Hyperedge>& after =
                dist.circuit.hyperedge_list();
            for (std::size_t r = cp * tpc_internal::kResumeStride; r < resume;
                 ++r) {
              if (after[r].vertices.size() == 1) continue;
              new_circ = tpc_internal::one_hyperedge(after[r], new_circ, ctx);
            }
          }
          start = resume;
          split_happened = true;
          break;
        }
      }
      if (!split_happened) break;  // pass completed
    }

    SerializedCircuit out;
    out.qubits.reserve(new_circ.qubit_order.size());
    for (const int id : new_circ.qubit_order) {
      out.qubits.push_back(new_circ.qubit_name(id));
    }
    for (const tpc_internal::BuiltCommand& c : new_circ.commands) {
      if (c.op_type == kOpBarrier) continue;  // barriers are scaffolding only
      std::vector<std::string> qs;
      const int arity = new_circ.cmd_arity(c);
      qs.reserve(arity);
      for (int p = 0; p < arity; ++p) {
        qs.push_back(new_circ.qubit_name(new_circ.cmd_qubit(c, p)));
      }
      Command oc;
      if (c.op_type == kOpCZ) {
        oc = Command{"CU1", std::move(qs), kOpCU1, {1.0}};
      } else if (tpc_internal::is_start_proc_cmd(c)) {
        oc = Command{"starting_process", std::move(qs), kOpCustomGate, {}};
      } else if (tpc_internal::is_end_proc_cmd(c)) {
        oc = Command{"ending_process", std::move(qs), kOpCustomGate, {}};
      } else {
        oc = Command{new_circ.op_name(c.op), std::move(qs), c.op_type,
                     new_circ.params_of(c)};
      }
      out.commands.push_back(std::move(oc));
    }

    ToPytketCircuitResult res;
    res.circuit = std::move(out);
    res.real_ebit_cost = ebit_cost(res.circuit);
    res.estimated_cost = dist.cost();
    if (res.real_ebit_cost != res.estimated_cost) {
      const std::vector<Vertex> detached = dist.detached_gate_list();
      const std::vector<Vertex> embedded =
          dist.circuit.get_all_h_embedded_gate_vertices();
      bool overlap = false;
      for (const Vertex d : detached) {
        if (std::find(embedded.begin(), embedded.end(), d) != embedded.end()) {
          overlap = true;
          break;
        }
      }
      if (overlap) {
        res.cost_mismatch_detached_embedded = true;
      } else {
        throw std::runtime_error(
            "Estimated cost does not match actual cost. Estimate: " +
            std::to_string(res.estimated_cost) +
            ", real: " + std::to_string(res.real_ebit_cost) + ".");
      }
    }
    return res;
  }
}

}  // namespace dqc
