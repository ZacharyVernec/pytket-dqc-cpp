#pragma once

#include <algorithm>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
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

// A command in the circuit being incrementally built. Standard gates carry
// their pytket op name; custom gates carry "starting_process"/"ending_process".
// A starting process records its origin qubit (the compute qubit it copies)
// explicitly, mirroring `start_proc(origin=...)` + `origin_of_start_proc`.
struct BuiltCommand {
  std::string op_name;
  int op_type = -1;
  std::vector<double> params;
  std::vector<std::string> qubits;
  std::string start_origin;  // only meaningful for starting_process
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

// Circuit builder used by `to_pytket_circuit`. `commands` records the add
// order (mirroring the sequence of pytket `add_gate`/`add_barrier` calls in
// the reference); `rebuild_dag()` materialises the DAG that lets
// `get_commands_order()` reproduce pytket's `Circuit.get_commands()` order
// (slices in temporal order, within a slice by UnitID).
struct BuiltCircuit {
  std::vector<std::string> qubits;  // first-seen order
  std::vector<UnitID> qubit_ids;
  std::vector<int> qubit_first_wire;  // wire id of the input wire per qubit
  std::vector<BuiltCommand> commands; // add order (barriers included)
  std::vector<BuiltVertex> vertices;
  std::vector<Wire> wires;
  std::vector<int> frontier_wire;     // current tip wire per qubit
  int wire_seq = 0;

  int qubit_index(const std::string& q) const {
    for (int i = 0; i < static_cast<int>(qubits.size()); ++i) {
      if (qubits[i] == q) return i;
    }
    return -1;
  }
  bool has_qubit(const std::string& q) const { return qubit_index(q) >= 0; }
  void add_qubit(const std::string& q) {
    if (has_qubit(q)) return;
    qubits.push_back(q);
    qubit_ids.push_back(parse_unit_id(q));
  }
  // Append a command to the add-order list (the DAG is materialised later by
  // `rebuild_dag`). Returns its index.
  int push_command(const BuiltCommand& cmd) {
    commands.push_back(cmd);
    return static_cast<int>(commands.size()) - 1;
  }
  // Mirror `new_circ.add_barrier(new_circ.qubits)`: a barrier over every
  // qubit currently in the circuit.
  void push_barrier_over_all() {
    BuiltCommand b;
    b.op_name = "Barrier";
    b.op_type = kOpBarrier;
    for (const std::string& q : qubits) b.qubits.push_back(q);
    commands.push_back(std::move(b));
  }
  // Rebuild the DAG (vertices/wires/frontiers) from `qubits` + `commands` in
  // add order, exactly as pytket's `add_gate`/`add_barrier` wire up the DAG.
  void rebuild_dag() {
    vertices.clear();
    wires.clear();
    frontier_wire.clear();
    wire_seq = 0;
    qubit_first_wire.assign(qubits.size(), -1);
    for (int qi = 0; qi < static_cast<int>(qubits.size()); ++qi) {
      const int w = wire_seq++;
      wires.push_back({w, qi, -1, -1});
      qubit_first_wire[qi] = w;
      frontier_wire.push_back(w);
    }
    for (const BuiltCommand& cmd : commands) {
      const int vidx = static_cast<int>(vertices.size());
      std::vector<int> in;
      std::vector<int> out;
      for (const std::string& q : cmd.qubits) {
        const int qi = qubit_index(q);
        if (qi < 0) throw std::runtime_error("rebuild_dag: unknown qubit " + q);
        const int w = frontier_wire[qi];
        in.push_back(w);
        wires[w].target = vidx;
      }
      vertices.push_back({cmd, in, {}});
      for (const std::string& q : cmd.qubits) {
        const int qi = qubit_index(q);
        const int w = wire_seq++;
        wires.push_back({w, qi, vidx, -1});
        out.push_back(w);
        frontier_wire[qi] = w;
      }
      vertices[vidx].out_wires = out;
    }
  }
  // Reproduce pytket's `Circuit.get_commands()` order: iterate slices
  // (frontier cuts) and, within each slice, order vertices by the first
  // reaching frontier edge, which is in UnitID order. Returns vertex indices.
  std::vector<int> get_commands_order() const {
    std::vector<int> qorder(qubits.size());
    for (int i = 0; i < static_cast<int>(qubits.size()); ++i) qorder[i] = i;
    std::sort(qorder.begin(), qorder.end(),
              [&](int a, int b) { return qubit_ids[a] < qubit_ids[b]; });
    std::vector<int> frontier = qubit_first_wire;
    std::vector<int> order;
    std::vector<char> in_slice(vertices.size(), 0);
    std::vector<char> bad(vertices.size(), 0);
    std::vector<char> in_edge;
    for (;;) {
      std::vector<int> all_edges;
      all_edges.reserve(qubits.size());
      for (const int qi : qorder) all_edges.push_back(frontier[qi]);
      in_edge.assign(wires.size(), 0);
      for (const int w : all_edges) in_edge[w] = 1;
      std::fill(in_slice.begin(), in_slice.end(), 0);
      std::fill(bad.begin(), bad.end(), 0);
      std::vector<int> slice;
      for (const int w : all_edges) {
        const int v = wires[w].target;
        if (v < 0 || in_slice[v] || bad[v]) continue;
        bool good = true;
        for (const int iw : vertices[v].in_wires) {
          if (!in_edge[iw]) {
            good = false;
            break;
          }
        }
        if (!good) {
          bad[v] = 1;
          continue;
        }
        in_slice[v] = 1;
        slice.push_back(v);
      }
      if (slice.empty()) break;
      for (const int v : slice) order.push_back(v);
      for (const int qi : qorder) {
        const int w = frontier[qi];
        const int v = wires[w].target;
        if (v >= 0 && in_slice[v]) {
          const BuiltCommand& c = vertices[v].cmd;
          for (int p = 0; p < static_cast<int>(c.qubits.size()); ++p) {
            if (qubit_index(c.qubits[p]) == qi) {
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
  return c.op_type == kOpCustomGate &&
         c.op_name.rfind("starting_process", 0) == 0;
}

inline bool is_end_proc_cmd(const BuiltCommand& c) {
  return c.op_type == kOpCustomGate &&
         c.op_name.rfind("ending_process", 0) == 0;
}

inline bool qubit_in(const BuiltCommand& c, const std::string& q) {
  return std::find(c.qubits.begin(), c.qubits.end(), q) != c.qubits.end();
}

// Mirrors `origin_of_start_proc`: the origin is stored alongside the command.
inline const std::string& origin_of_start_proc(const BuiltCommand& c) {
  return c.start_origin;
}

// Mirrors `requires_embedding_start_proc(commands)` where `commands` is the
// slice `built_commands[begin:]`.
inline bool requires_embedding_start_proc(
    const std::vector<BuiltCommand>& commands, std::size_t begin) {
  const std::string& target_link = commands[begin].qubits[1];
  for (std::size_t i = begin; i < commands.size(); ++i) {
    const BuiltCommand& g = commands[i];
    if (g.op_type == kOpCU1 && qubit_in(g, target_link)) return true;
    if (is_end_proc_cmd(g) && g.qubits[0] == target_link) return false;
  }
  throw std::runtime_error(
      "requires_embedding_start_proc: no conclusion reached");
}

// Mirrors `requires_embedding_end_proc(commands)` where `commands` is the
// slice `built_commands[:end]`.
inline bool requires_embedding_end_proc(
    const std::vector<BuiltCommand>& commands, std::size_t end) {
  const std::string& source_link = commands[end - 1].qubits[0];
  for (std::size_t i = end; i-- > 0;) {
    const BuiltCommand& g = commands[i];
    if (g.op_type == kOpCU1 && qubit_in(g, source_link)) return true;
    if (is_start_proc_cmd(g) && g.qubits.size() > 1 &&
        g.qubits[1] == source_link)
      return false;
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

// Mirrors the nested `LinkManager` class. `link_qubit_dict` is insertion
// ordered (CPython dict semantics) — NOT a std::map — because
// `connected_servers()` iterates its keys and drives `start_link` tie-breaks.
struct LinkManager {
  const TPCContext& ctx;
  Hyperedge hyperedge;
  std::vector<std::pair<int, std::string>> link_qubit_dict;
  std::map<int, std::vector<std::string>> available;
  std::map<int, std::vector<std::string>> occupied;
  std::map<std::string, std::string> link_qubit_id_update;

  LinkManager(const TPCContext& ctx, const Hyperedge& h,
              const std::vector<int>& servers)
      : ctx(ctx), hyperedge(h) {
    for (const int s : servers) {
      available[s] = {};
      occupied[s] = {};
    }
  }

  Vertex qubit_vertex() const {
    return ctx.hyp_circ.get_qubit_vertex(hyperedge);
  }
  int home_server() const { return ctx.placement.at(qubit_vertex()); }
  const std::string& circuit_qubit() const {
    return ctx.circuit_qubit_of_vertex(qubit_vertex());
  }

  // Mirrors `_request_link_qubit`.
  std::string request_link_qubit(int server) {
    int capacity = -1;
    auto it = ctx.server_ebit_mem.find(server);
    if (it != ctx.server_ebit_mem.end()) capacity = it->second;
    if (ctx.satisfy_bound && capacity > 0) {
      if (capacity <= static_cast<int>(occupied[server].size())) {
        throw ConstraintException(
            "Communication memory capacity of server " +
                std::to_string(server) +
                " exceeded.\n\tConsider setting allow_update to True.",
            server);
      }
    }

    std::string qubit;
    if (available[server].empty()) {
      const int next_id = static_cast<int>(occupied[server].size());
      qubit = "server_" + std::to_string(server) + "_link_register[" +
              std::to_string(next_id) + "]";
    } else {
      qubit = available[server].back();
      available[server].pop_back();
    }
    occupied[server].push_back(qubit);
    return qubit;
  }

  // Mirrors `_release_link_qubit`.
  void release_link_qubit(const std::string& link_qubit) {
    const int server = get_server_id(link_qubit);
    auto& occ = occupied[server];
    occ.erase(std::find(occ.begin(), occ.end(), link_qubit));
    available[server].push_back(link_qubit);
  }

  // Mirrors `connected_servers()`: keys of link_qubit_dict in insertion order.
  std::vector<int> connected_servers() const {
    std::vector<int> out;
    for (const auto& [s, q] : link_qubit_dict) {
      (void)q;
      out.push_back(s);
    }
    return out;
  }

  void set_link_qubit(int server, const std::string& q) {
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
  std::string get_link_qubit(int server) const {
    if (home_server() == server) {
      return ctx.qubit_mapping.at(circuit_qubit());
    }
    for (const auto& [s, q] : link_qubit_dict) {
      if (s == server) return q;
    }
    throw std::out_of_range("server not in link_qubit_dict");
  }

  // Mirrors `start_link`: returns the list of EjppAction (from, to) pairs.
  std::vector<std::pair<std::string, std::string>> start_link(int target) {
    const int q_vertex = qubit_vertex();
    const int home = ctx.placement.at(q_vertex);
    std::vector<int> hyp_servers;
    for (const Vertex v : hyperedge.vertices) {
      hyp_servers.push_back(ctx.placement.at(v));
    }
    const Graph tree = steiner_tree(ctx.server_graph, hyp_servers);
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

    std::vector<std::pair<std::string, std::string>> actions;
    std::string last_link_qubit = get_link_qubit(best_path[0]);
    for (std::size_t i = 1; i < best_path.size(); ++i) {
      const int next_server = best_path[i];
      const std::string this_link_qubit = request_link_qubit(next_server);
      set_link_qubit(next_server, this_link_qubit);
      actions.push_back({last_link_qubit, this_link_qubit});
      last_link_qubit = this_link_qubit;
    }
    return actions;
  }

  // Mirrors `end_links`.
  std::vector<std::pair<std::string, std::string>> end_links(
      const std::vector<int>& targets) {
    const std::string home_link = ctx.qubit_mapping.at(circuit_qubit());
    std::vector<std::pair<std::string, std::string>> actions;
    for (const int target : targets) {
      const std::string target_link = get_link_qubit(target);
      actions.push_back({target_link, home_link});
      release_link_qubit(target_link);
      erase_link_qubit(target);
    }
    return actions;
  }

  // Mirrors `update_occupation`.
  void update_occupation(const std::string& link_qubit, bool starting) {
    if (starting) {
      const int server = get_server_id(link_qubit);
      const std::string new_link_qubit = request_link_qubit(server);
      link_qubit_id_update[link_qubit] = new_link_qubit;
    } else {
      const std::string new_link_qubit = link_qubit_id_update.at(link_qubit);
      release_link_qubit(new_link_qubit);
    }
  }

  // Mirrors `get_updated_name`.
  std::string get_updated_name(const std::string& q) const {
    auto it = link_qubit_id_update.find(q);
    if (it != link_qubit_id_update.end()) return it->second;
    return q;
  }
};

inline BuiltCommand mk_cmd(const std::string& name, int op_type,
                           const std::vector<double>& params,
                           const std::vector<std::string>& qubits,
                           const std::string& origin = "") {
  return BuiltCommand{name, op_type, params, qubits, origin};
}

// Copy a command verbatim, renaming its qubits through the LinkManager.
inline BuiltCommand copy_cmd(const BuiltCommand& src, const LinkManager& lm) {
  BuiltCommand out = src;
  out.qubits.clear();
  for (const std::string& q : src.qubits) {
    out.qubits.push_back(lm.get_updated_name(q));
  }
  return out;
}

// Mirrors `to_pytket_circuit_one_hyperedge`.
inline BuiltCircuit one_hyperedge(const Hyperedge& hyperedge,
                                  const BuiltCircuit& circ,
                                  const TPCContext& ctx) {
  if (hyperedge.vertices.size() == 1) {
    return circ;
  }

  BuiltCircuit new_circ;
  for (const auto& [orig, hw] : ctx.qubit_mapping) {
    (void)orig;
    new_circ.add_qubit(hw);
  }

  const Vertex q_vertex = ctx.hyp_circ.get_qubit_vertex(hyperedge);
  const std::string src_qubit =
      ctx.qubit_mapping.at(ctx.circuit_qubit_of_vertex(q_vertex));

  const std::vector<Vertex> gate_vertices =
      ctx.hyp_circ.get_gate_vertices(hyperedge);
  const Vertex min_gv =
      *std::min_element(gate_vertices.begin(), gate_vertices.end());
  const Vertex max_gv =
      *std::max_element(gate_vertices.begin(), gate_vertices.end());

  int v_gate = static_cast<int>(ctx.qubit_mapping.size()) - 1;
  bool currently_h_embedding = false;
  double carry_phase = 0.0;
  LinkManager lm(ctx, hyperedge, ctx.server_list);

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
        lm.update_occupation(cmd.qubits[1], true);
      } catch (ConstraintException& e) {
        e.v_gate = v_gate;
        throw;
      }
      const std::string remote_qubit = lm.get_updated_name(cmd.qubits[1]);
      new_circ.add_qubit(remote_qubit);
    }
    if (is_end_proc_cmd(cmd)) {
      try {
        lm.update_occupation(cmd.qubits[0], false);
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
        new_circ.commands.push_back(copy_cmd(cmd, lm));
      }
      continue;
    }

    if (cmd.op_type == kOpRz) {
      if (!qubit_in(cmd, src_qubit)) {
        new_circ.commands.push_back(copy_cmd(cmd, lm));
        continue;
      }
      const std::string q = cmd.qubits[0];
      const double phase = cmd.params[0];
      new_circ.commands.push_back(mk_cmd("Rz", kOpRz, {phase}, {q}));
      if (currently_h_embedding) {
        const double p1 = py_mod(phase, 1.0);
        if (isclose(p1, 0.0) || isclose(p1, 1.0)) {
          for (const int server : lm.connected_servers()) {
            const std::string lq = lm.get_link_qubit(server);
            new_circ.commands.push_back(mk_cmd("Rz", kOpRz, {phase}, {lq}));
          }
        } else {
          carry_phase += phase;
        }
      }
    } else if (cmd.op_type == kOpH) {
      if (!qubit_in(cmd, src_qubit)) {
        new_circ.commands.push_back(copy_cmd(cmd, lm));
        continue;
      }
      const std::string q = cmd.qubits[0];

      if (!currently_h_embedding) {
        // Detect whether this H opens an embedding unit with content.
        int found_idx = -1;
        const BuiltCommand* found_cmd = nullptr;
        for (std::size_t i = cmd_idx + 1; i < commands.size(); ++i) {
          const BuiltCommand& g = commands[i];
          if (g.op_type == kOpCU1 && qubit_in(g, q)) {
            found_idx = static_cast<int>(i);
            found_cmd = &g;
            break;
          } else if (is_start_proc_cmd(g) && g.start_origin == q &&
                     requires_embedding_start_proc(commands, i)) {
            found_idx = static_cast<int>(i);
            found_cmd = &g;
            break;
          } else if (is_end_proc_cmd(g) && g.qubits.size() > 1 &&
                     g.qubits[1] == q &&
                     requires_embedding_end_proc(commands, i + 1)) {
            found_idx = static_cast<int>(i);
            found_cmd = &g;
            break;
          } else if (g.op_type == kOpH && qubit_in(g, q)) {
            break;
          }
        }

        if (found_cmd == nullptr) {
          // Case 1: empty embedding unit.
          for (const int server : lm.connected_servers()) {
            const std::string lq = lm.get_link_qubit(server);
            new_circ.commands.push_back(mk_cmd("H", kOpH, {}, {lq}));
          }
          currently_h_embedding = true;
        } else {
          // Case 2: embedding unit with content; keep only the remote server.
          std::string remote_qubit;
          if (is_start_proc_cmd(*found_cmd)) {
            remote_qubit = found_cmd->qubits[1];
          } else {
            for (const std::string& rq : found_cmd->qubits) {
              if (rq != q) remote_qubit = rq;
            }
          }
          if (found_cmd->op_type == kOpCU1 && is_link_qubit(remote_qubit)) {
            std::string original_remote;
            for (std::size_t j = static_cast<std::size_t>(found_idx);
                 j < commands.size(); ++j) {
              const BuiltCommand& g = commands[j];
              if (is_end_proc_cmd(g) && g.qubits[0] == remote_qubit) {
                original_remote = g.qubits[1];
                break;
              }
            }
            remote_qubit = original_remote;
          }
          const int remote_server = get_server_id(remote_qubit);
          std::vector<int> end_servers;
          for (const int s : lm.connected_servers()) {
            if (s != remote_server) end_servers.push_back(s);
          }
          for (const auto& [from, to] : lm.end_links(end_servers)) {
            new_circ.commands.push_back(
                mk_cmd("ending_process", kOpCustomGate, {}, {from, to}));
          }
          for (const int server : lm.connected_servers()) {
            const std::string lq = lm.get_link_qubit(server);
            new_circ.commands.push_back(mk_cmd("H", kOpH, {}, {lq}));
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
            const std::string lq = lm.get_link_qubit(server);
            new_circ.commands.push_back(mk_cmd("H", kOpH, {}, {lq}));
          }
        } else if (isclose(m2, 1.0)) {
          // Z gate: end embedding with Rz + H.
          currently_h_embedding = false;
          for (const int server : lm.connected_servers()) {
            const std::string lq = lm.get_link_qubit(server);
            new_circ.commands.push_back(
                mk_cmd("Rz", kOpRz, {carry_phase}, {lq}));
            new_circ.commands.push_back(mk_cmd("H", kOpH, {}, {lq}));
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
      new_circ.commands.push_back(mk_cmd("H", kOpH, {}, {q}));
    } else if (cmd.op_type == kOpCU1) {
      if (!qubit_in(cmd, src_qubit)) {
        new_circ.commands.push_back(copy_cmd(cmd, lm));
        continue;
      }
      const double phase = cmd.params[0];
      std::string rmt_qubit;
      for (const std::string& q : cmd.qubits) {
        if (q != src_qubit) rmt_qubit = q;
      }

      if (std::find(hyperedge.vertices.begin(), hyperedge.vertices.end(),
                    v_gate) != hyperedge.vertices.end()) {
        // Distribute this gate.
        // assert !currently_h_embedding
        const int target_server = ctx.placement.at(v_gate);

        std::vector<std::pair<std::string, std::string>> start_actions;
        try {
          start_actions = lm.start_link(target_server);
        } catch (ConstraintException& e) {
          e.v_gate = v_gate;
          throw;
        }
        for (const auto& [from, to] : start_actions) {
          new_circ.add_qubit(to);
          new_circ.commands.push_back(
              mk_cmd("starting_process_" + src_qubit, kOpCustomGate, {},
                     {from, to}, src_qubit));
        }
        new_circ.commands.push_back(
            mk_cmd("CU1", kOpCU1, {phase},
                   {lm.get_link_qubit(target_server),
                    lm.get_updated_name(rmt_qubit)}));

        if (v_gate == max_gv) {
          for (const auto& [from, to] :
               lm.end_links(lm.connected_servers())) {
            new_circ.commands.push_back(
                mk_cmd("ending_process", kOpCustomGate, {}, {from, to}));
          }
        }
      } else {
        new_circ.commands.push_back(copy_cmd(cmd, lm));
        if (currently_h_embedding) {
          const double p1 = py_mod(phase, 1.0);
          (void)p1;  // assert isclose(p1,0) or isclose(p1,1)
          if (is_link_qubit(rmt_qubit)) {
            std::string original_remote;
            for (std::size_t j = cmd_idx; j < commands.size(); ++j) {
              const BuiltCommand& g = commands[j];
              if (is_end_proc_cmd(g) && g.qubits[0] == rmt_qubit) {
                original_remote = g.qubits[1];
                break;
              }
            }
            rmt_qubit = original_remote;
          }
          for (const int server : lm.connected_servers()) {
            const std::string lq = lm.get_link_qubit(server);
            new_circ.commands.push_back(
                mk_cmd("CZ", kOpCZ, {}, {lq, rmt_qubit}));
          }
        }
      }
    } else if (is_start_proc_cmd(cmd)) {
      const std::string& orig_qubit = cmd.start_origin;
      const std::string remote_qubit = lm.get_updated_name(cmd.qubits[1]);
      new_circ.commands.push_back(copy_cmd(cmd, lm));
      if (currently_h_embedding && src_qubit == orig_qubit) {
        if (requires_embedding_start_proc(commands, cmd_idx)) {
          for (const int server : lm.connected_servers()) {
            const std::string lq = lm.get_link_qubit(server);
            new_circ.commands.push_back(mk_cmd("H", kOpH, {}, {remote_qubit}));
            new_circ.commands.push_back(
                mk_cmd("CZ", kOpCZ, {}, {remote_qubit, lq}));
            new_circ.commands.push_back(mk_cmd("H", kOpH, {}, {remote_qubit}));
          }
        }
      }
    } else if (is_end_proc_cmd(cmd)) {
      const std::string orig_qubit = lm.get_updated_name(cmd.qubits[1]);
      const std::string remote_qubit = lm.get_updated_name(cmd.qubits[0]);
      if (currently_h_embedding && src_qubit == orig_qubit) {
        if (requires_embedding_end_proc(commands, cmd_idx + 1)) {
          for (const int server : lm.connected_servers()) {
            const std::string lq = lm.get_link_qubit(server);
            new_circ.commands.push_back(mk_cmd("H", kOpH, {}, {remote_qubit}));
            new_circ.commands.push_back(
                mk_cmd("CZ", kOpCZ, {}, {remote_qubit, lq}));
            new_circ.commands.push_back(mk_cmd("H", kOpH, {}, {remote_qubit}));
          }
        }
      }
      new_circ.commands.push_back(copy_cmd(cmd, lm));
    } else if (cmd.op_type == kOpCZ) {
      new_circ.commands.push_back(copy_cmd(cmd, lm));
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
// production caller in `to_pytket_circuit` ignores it).
inline std::optional<Hyperedge> split_for_bound(Distribution& dist,
                                                const ConstraintException& e) {
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
                " since it is placed on a non-home server " +
                std::to_string(placed_server) +
                " and said server has memory capacity less than two.",
            placed_server);
      }
    }
    throw ConstraintException(
        "The distribution could not be amended to satisfy the bound on "
        "communication capacity.",
        e.server);
  }

  const Hyperedge& chosen_hedge = *chosen;
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
    tpc_internal::TPCContext ctx{
        dist.circuit,      dist.placement,
        dist.server_graph, qubit_mapping,
        dist.server_ebit_mem, satisfy_bound,
        dist.get_server_list()};

    tpc_internal::BuiltCircuit new_circ;
    for (const auto& [orig, hw] : qubit_mapping) {
      (void)orig;
      new_circ.add_qubit(hw);
    }
    // Mirror the reference: build the initial circuit from the serialized
    // commands and insert a full barrier after every CU1 gate. The barriers
    // pin the CU1 ordering so pytket's `get_commands()` is deterministic.
    for (const Command& cmd : dist.circuit.get_circuit().commands) {
      tpc_internal::BuiltCommand out;
      out.op_name = cmd.op_name;
      out.op_type = cmd.op_type;
      out.params = cmd.params;
      for (const std::string& q : cmd.qubits) {
        out.qubits.push_back(qubit_mapping.at(q));
      }
      new_circ.commands.push_back(std::move(out));
      if (cmd.op_type == kOpCU1) {
        new_circ.push_barrier_over_all();
      }
    }
    new_circ.rebuild_dag();

    try {
      for (const Hyperedge& hedge : dist.circuit.hyperedge_list()) {
        new_circ = tpc_internal::one_hyperedge(hedge, new_circ, ctx);
      }
    } catch (ConstraintException& e) {
      if (!allow_update) throw;
      tpc_internal::split_for_bound(dist, e);
      continue;  // retry the whole body with the updated hypergraph
    }

    SerializedCircuit out;
    out.qubits = new_circ.qubits;
    for (const tpc_internal::BuiltCommand& c : new_circ.commands) {
      if (c.op_type == kOpBarrier) continue;  // barriers are scaffolding only
      Command oc;
      if (c.op_type == kOpCZ) {
        oc = Command{"CU1", c.qubits, kOpCU1, {1.0}};
      } else if (tpc_internal::is_start_proc_cmd(c)) {
        oc = Command{"starting_process", c.qubits, kOpCustomGate, {}};
      } else if (tpc_internal::is_end_proc_cmd(c)) {
        oc = Command{"ending_process", c.qubits, kOpCustomGate, {}};
      } else {
        oc = Command{c.op_name, c.qubits, c.op_type, c.params};
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
