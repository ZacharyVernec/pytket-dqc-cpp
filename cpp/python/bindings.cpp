#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <optional>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "dqc/allocators.hpp"
#include "dqc/bipartite.hpp"
#include "dqc/circuit_analysis.hpp"
#include "dqc/distribution.hpp"
#include "dqc/gain_manager.hpp"
#include "dqc/gateset.hpp"
#include "dqc/graph.hpp"
#include "dqc/graph_tools.hpp"
#include "dqc/hypergraph.hpp"
#include "dqc/hypergraph_circuit.hpp"
#ifdef DQC_HAS_KAHYPAR
#include "dqc/hypergraph_partitioning.hpp"
#endif
#include "dqc/network.hpp"
#include "dqc/numeric.hpp"
#include "dqc/pacman.hpp"
#include "dqc/refiners/boundary_reallocation.hpp"
#include "dqc/refiners/composition.hpp"
#include "dqc/refiners/d_type_merge.hpp"
#include "dqc/refiners/h_type_merge.hpp"
#include "dqc/refiners/vertex_cover.hpp"
#include "dqc/rng.hpp"
#include "dqc/to_pytket_circuit.hpp"

namespace py = pybind11;
using namespace dqc;

namespace {

// Rebuild a `pytket.Circuit` from the serialized form emitted by the core's
// `to_pytket_circuit`. The core stays pytket-free; only this binding layer
// touches pytket (lazily, at call time).
py::object rebuild_pytket_circuit(const SerializedCircuit& sc) {
  py::module_ pytket = py::module_::import("pytket");
  py::object Qubit = pytket.attr("circuit").attr("Qubit");
  py::object Op = pytket.attr("circuit").attr("Op");
  py::object OpType = pytket.attr("OpType");

  py::object circ = pytket.attr("Circuit")();

  auto parse_qubit = [&](const std::string& q) -> py::object {
    const std::size_t br = q.find('[');
    const std::string name = q.substr(0, br);
    const std::string idx = q.substr(br + 1, q.size() - br - 2);
    return Qubit(name, std::stoi(idx));
  };

  for (const std::string& q : sc.qubits) {
    circ.attr("add_qubit")(parse_qubit(q));
  }

  py::module_ gateset = py::module_::import("pytket_dqc.utils.gateset");

  for (const Command& c : sc.commands) {
    py::list qs;
    for (const std::string& q : c.qubits) qs.append(parse_qubit(q));

    if (c.op_type == kOpCustomGate) {
      py::object defn;
      if (c.op_name == "starting_process") {
        defn = gateset.attr("start_proc")();
      } else if (c.op_name == "ending_process") {
        defn = gateset.attr("end_proc")();
      } else {
        throw std::runtime_error("unknown custom gate: " + c.op_name);
      }
      circ.attr("add_custom_gate")(defn, py::list(), qs);
    } else {
      py::object opt;
      if (c.op_type == kOpH) opt = OpType.attr("H");
      else if (c.op_type == kOpRz) opt = OpType.attr("Rz");
      else if (c.op_type == kOpCU1) opt = OpType.attr("CU1");
      else if (c.op_type == kOpCZ) opt = OpType.attr("CZ");
      else
        throw std::runtime_error("unknown op_type: " +
                                 std::to_string(c.op_type));

      const py::object op = Op.attr("create")(opt, py::cast(c.params));
      circ.attr("add_gate")(op, qs);
    }
  }
  return circ;
}

#ifdef DQC_HAS_KAHYPAR
// Path to the bundled KaHyPar INI configuration. Prefers the `dqc_cpp`
// package attribute (`dqc_cpp.KAHYPAR_INI_PATH`, mirroring the reference's
// `importlib_resources` default); falls back to the copy of the INI shipped
// next to the compiled `_core` module (CMake copies it into the build dir).
std::string default_ini_path() {
  try {
    py::module_ dqc_cpp = py::module_::import("dqc_cpp");
    return py::cast<std::string>(dqc_cpp.attr("KAHYPAR_INI_PATH"));
  } catch (py::error_already_set&) {
    py::object file = py::module_::import("_core").attr("__file__");
    std::string path = py::cast<std::string>(file);
    const std::size_t slash = path.find_last_of("/\\");
    const std::string dir =
        slash == std::string::npos ? "." : path.substr(0, slash);
    return dir + "/km1_kKaHyPar_sea20.ini";
  }
}
#endif  // DQC_HAS_KAHYPAR

// Binding-layer composition refiners. The core primitive refiners are free
// functions (`bool(Distribution&)`); `RepeatRefiner`/`SequenceRefiner` wrap a
// Python callable and reproduce the reference's do-while / or-fold logic here
// (the C++-only `dqc::RepeatRefiner`/`dqc::SequenceRefiner` in
// `dqc/refiners/composition.hpp` provide the same logic for native callers).
struct PyRepeatRefiner {
  py::function refine_fn;
  explicit PyRepeatRefiner(py::function f) : refine_fn(std::move(f)) {}
  bool refine(std::shared_ptr<Distribution> d) {
    bool refinement_made = refine_fn(d).cast<bool>();
    const bool one_refinement_made = refinement_made;
    while (refinement_made) {
      refinement_made = refine_fn(d).cast<bool>();
    }
    return one_refinement_made;
  }
};

struct PySequenceRefiner {
  std::vector<py::function> refine_fns;
  explicit PySequenceRefiner(std::vector<py::function> fns)
      : refine_fns(std::move(fns)) {}
  bool refine(std::shared_ptr<Distribution> d) {
    bool refinement_made = false;
    for (const auto& fn : refine_fns) {
      refinement_made = fn(d).cast<bool>() || refinement_made;
    }
    return refinement_made;
  }
};

// Accepts either a plain callable (e.g. a `_core` refiner function or a lambda)
// or a duck-typed `Refiner` object with a `.refine(distribution)` method,
// mirroring the reference's `Refiner` ABC. Composition classes are themselves
// callable (`__call__` -> `refine`), so they can be nested.
py::function to_callable(py::object obj) {
  if (PyCallable_Check(obj.ptr())) {
    return py::reinterpret_borrow<py::function>(obj);
  }
  py::object refine = obj.attr("refine");
  return py::reinterpret_borrow<py::function>(refine);
}

// Mirrors `NISQNetwork.from_dict`. Preserves the insertion order of the
// `server_qubits` / `server_ebit_mem` dicts (the reference relies on dict
// insertion order for the GainManager occupancy keys).
NISQNetwork nq_from_dict(const py::dict& d) {
  std::vector<std::pair<int, int>> coupling;
  for (const auto& item : d["server_coupling"].cast<py::list>()) {
    const py::list pair = item.cast<py::list>();
    coupling.emplace_back(pair[0].cast<int>(), pair[1].cast<int>());
  }
  std::vector<std::pair<int, std::vector<int>>> server_qubits;
  const py::dict sq = d["server_qubits"].cast<py::dict>();
  for (const auto& [k, v] : sq) {
    server_qubits.emplace_back(k.cast<int>(), v.cast<std::vector<int>>());
  }
  std::map<int, int> ebit;
  if (d.contains("server_ebit_mem")) {
    const py::dict em = d["server_ebit_mem"].cast<py::dict>();
    for (const auto& [k, v] : em) {
      ebit[k.cast<int>()] = v.cast<int>();
    }
  }
  return NISQNetwork(coupling, server_qubits, ebit);
}

}  // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() = "C++ core of pytket-dqc (faithful port).";

  m.def(
      "isclose",
      [](double a, double b, double rtol, double atol) {
        return isclose(a, b, rtol, atol);
      },
      py::arg("a"), py::arg("b"), py::arg("rtol") = 1e-05, py::arg("atol") = 1e-08);

  // ---- gateset -------------------------------------------------------------
  m.def("to_euler_with_two_hadamards",
        [](const std::vector<std::pair<int, std::vector<double>>>& ops) {
          std::vector<Op> in;
          in.reserve(ops.size());
          for (const auto& [t, ps] : ops) in.push_back(Op{t, ps});
          const std::vector<Op> out = to_euler_with_two_hadamards(in);
          std::vector<std::pair<int, std::vector<double>>> res;
          res.reserve(out.size());
          for (const Op& op : out) res.push_back({op.op_type, op.params});
          return res;
        },
        py::arg("ops"));

  // ---- PyRandom ----------------------------------------------------------
  py::class_<PyRandom>(m, "PyRandom")
      .def(py::init<std::optional<std::uint64_t>>(), py::arg("seed") = std::nullopt)
      .def("seed", &PyRandom::seed, py::arg("a"))
      .def("random", &PyRandom::random)
      .def("uniform", &PyRandom::uniform, py::arg("a"), py::arg("b"))
      .def("randrange", py::overload_cast<std::uint64_t>(&PyRandom::randrange),
           py::arg("stop"))
      .def("randrange",
           py::overload_cast<std::uint64_t, std::uint64_t>(&PyRandom::randrange),
           py::arg("start"), py::arg("stop"))
      .def("randbelow", &PyRandom::randbelow, py::arg("n"))
      .def("choice",
           [](PyRandom& r, const std::vector<int>& seq) { return r.choice(seq); },
           py::arg("seq"))
      .def("getrandbits", &PyRandom::getrandbits, py::arg("k"))
      .def("get_state", &PyRandom::get_state)
      .def("set_state", &PyRandom::set_state, py::arg("state"));

  // ---- Hyperedge / Hypergraph / Placement ---------------------------------
  py::class_<Hyperedge>(m, "Hyperedge")
      .def(py::init<std::vector<Vertex>, int>(), py::arg("vertices"),
           py::arg("weight") = 1)
      .def_readwrite("vertices", &Hyperedge::vertices)
      .def_readwrite("weight", &Hyperedge::weight)
      .def("__eq__",
           [](const Hyperedge& a, const Hyperedge& b) { return a == b; })
      .def("__hash__",
           [](const Hyperedge& h) { return h.py_hash(); })
      .def("__repr__", &Hyperedge::to_string);

  py::class_<Placement>(m, "Placement")
      .def(py::init<>())
      .def("set", &Placement::set, py::arg("vertex"), py::arg("server"))
      .def_readonly("entries", &Placement::entries)
      .def_readonly("placement", &Placement::placement)
      .def("get_vertices_in", &Placement::get_vertices_in, py::arg("server"));

  py::class_<Hypergraph>(m, "Hypergraph")
      .def(py::init<>())
      .def("add_vertex", &Hypergraph::add_vertex)
      .def("add_vertices", &Hypergraph::add_vertices)
      .def("add_hyperedge",
           [](Hypergraph& g, const std::vector<Vertex>& vs, int w) {
             g.add_hyperedge(vs, w);
           },
           py::arg("vertices"), py::arg("weight") = 1)
      .def("remove_hyperedge", &Hypergraph::remove_hyperedge)
      .def("merge_hyperedge", &Hypergraph::merge_hyperedge)
      .def("split_hyperedge", &Hypergraph::split_hyperedge)
      .def_property_readonly("vertex_list", &Hypergraph::vertex_list)
      .def_property_readonly("hyperedge_list", &Hypergraph::hyperedge_list)
      .def_property_readonly("hyperedge_dict", &Hypergraph::hyperedge_dict)
      .def_property_readonly("vertex_neighbours", &Hypergraph::vertex_neighbours)
      .def("is_valid", &Hypergraph::is_valid)
      .def("is_placement", &Hypergraph::is_placement)
      .def("get_boundary", &Hypergraph::get_boundary)
      .def("weight_one_predicate", &Hypergraph::weight_one_predicate)
      .def("kahypar_hyperedges", &Hypergraph::kahypar_hyperedges)
      .def("__repr__", &Hypergraph::to_string);

  // ---- Graph ---------------------------------------------------------------
  py::class_<Graph>(m, "Graph")
      .def(py::init<>())
      .def("add_node", &Graph::add_node)
      .def("add_edge",
           [](Graph& g, int u, int v, double w) {
             g.add_edge(u, v, {{"weight", w}});
           },
           py::arg("u"), py::arg("v"), py::arg("weight") = 1.0)
      .def("has_edge", &Graph::has_edge)
      .def("degree", &Graph::degree, py::arg("n"))
      .def("remove_node", &Graph::remove_node, py::arg("n"))
      .def("weight", &Graph::weight, py::arg("u"), py::arg("v"),
           py::arg("key") = "weight", py::arg("default_value") = 1.0)
      .def("nodes", &Graph::nodes)
      .def("edges",
           [](const Graph& g) {
             std::vector<std::tuple<int, int, double>> out;
             for (const Edge& e : g.edges()) {
               out.emplace_back(e.u, e.v, e.weight("weight"));
             }
             return out;
           });

  // ---- bipartite (BipartiteGraph<int> exposed for differential tests) -------
  using IntBip = BipartiteGraph<int, &py_hash_key, &py_eq_key>;
  py::class_<IntBip>(m, "BipartiteGraph")
      .def(py::init<>())
      .def("add_edge", &IntBip::add_edge, py::arg("u"), py::arg("v"))
      .def("nodes", &IntBip::nodes)
      .def("edges",
           [](const IntBip& g) {
             py::list out;
             for (const auto& [u, v] : g.edges()) {
               out.append(py::make_tuple(u, v));
             }
             return out;
           })
      .def("color",
           [](const IntBip& g) {
             py::dict d;
             for (const auto& [n, c] : g.color()) d[py::int_(n)] = c;
             return d;
           })
      .def("sets",
           [](const IntBip& g, std::optional<std::vector<int>> top_nodes) {
             const PySet<int>* top = nullptr;
             PySet<int> tmp(&py_hash_key, &py_eq_key);
             if (top_nodes.has_value()) {
               for (const int n : *top_nodes) tmp.add(n);
               top = &tmp;
             }
             const auto [X, Y] = g.sets(top);
             return py::make_tuple(X.iteration_order(), Y.iteration_order());
           },
           py::arg("top_nodes") = std::nullopt)
      .def("maximum_matching",
           [](const IntBip& g, const std::vector<int>& top_nodes) {
             PySet<int> top(&py_hash_key, &py_eq_key);
             for (const int n : top_nodes) top.add(n);
             const std::map<int, int> m = g.maximum_matching(top);
             py::dict d;
             for (const auto& [k, v] : m) d[py::int_(k)] = v;
             return d;
           },
           py::arg("top_nodes"))
      .def("to_vertex_cover",
           [](const IntBip& g, const std::map<int, int>& matching,
              const std::vector<int>& top_nodes) {
             PySet<int> top(&py_hash_key, &py_eq_key);
             for (const int n : top_nodes) top.add(n);
             return g.to_vertex_cover(matching, top).iteration_order();
           },
           py::arg("matching"), py::arg("top_nodes"))
      .def("subgraph",
           [](const IntBip& g, const std::vector<int>& nodes) {
             PySet<int> keep(&py_hash_key, &py_eq_key);
             for (const int n : nodes) keep.add(n);
             return g.subgraph(keep);
           },
           py::arg("nodes"))
      .def("connected_components",
           [](const IntBip& g) {
             py::list out;
             for (const auto& comp : g.connected_components()) {
               py::list c;
               for (const int n : comp.iteration_order()) c.append(n);
               out.append(c);
             }
             return out;
           });

  // ---- graph_tools ---------------------------------------------------------
  m.def("steiner_tree", &steiner_tree, py::arg("graph"), py::arg("nodes"));
  m.def("direct_from_origin", &direct_from_origin, py::arg("graph"),
        py::arg("origin"));
  m.def("get_distribution_tree", &get_distribution_tree, py::arg("placement"),
        py::arg("hyperedge"), py::arg("qubit_node"), py::arg("server_graph"));

  // ---- circuit_analysis ----------------------------------------------------
  using Cmd = std::tuple<std::string, int, std::vector<std::string>,
                         std::vector<double>>;
  using CmdList = std::vector<Cmd>;

  auto to_serialized = [](const std::vector<std::string>& qubits,
                          const CmdList& commands) {
    SerializedCircuit c;
    c.qubits = qubits;
    for (const auto& [name, optype, qs, params] : commands) {
      c.commands.push_back({name, qs, optype, params});
    }
    return c;
  };

  m.def("is_link_qubit",
        [](const std::string& s) { return is_link_qubit(s); });
  m.def("get_server_id",
        [](const std::string& s) { return get_server_id(s); });
  m.def("all_cu1_local",
        [&](const std::vector<std::string>& qubits, const CmdList& commands) {
          return all_cu1_local(to_serialized(qubits, commands));
        },
        py::arg("qubits"), py::arg("commands"));
  m.def("ebit_cost",
        [&](const std::vector<std::string>& qubits, const CmdList& commands) {
          return ebit_cost(to_serialized(qubits, commands));
        },
        py::arg("qubits"), py::arg("commands"));
  m.def("ebit_memory_required",
        [&](const std::vector<std::string>& qubits, const CmdList& commands) {
          return ebit_memory_required(to_serialized(qubits, commands));
        },
        py::arg("qubits"), py::arg("commands"));

  // ---- hypergraph_circuit --------------------------------------------------
  py::class_<HypergraphCircuit>(m, "HypergraphCircuit")
      .def(py::init<>())
      .def("from_circuit",
           [&](HypergraphCircuit& h, const std::vector<std::string>& qubits,
               const CmdList& commands) {
             h.from_circuit(to_serialized(qubits, commands));
           },
           py::arg("qubits"), py::arg("commands"))
      .def_property_readonly("vertex_list", &HypergraphCircuit::vertex_list)
      .def_property_readonly("hyperedge_list",
                             &HypergraphCircuit::hyperedge_list)
      .def_property_readonly("hyperedge_dict",
                             &HypergraphCircuit::hyperedge_dict)
      .def("get_qubit_vertices", &HypergraphCircuit::get_qubit_vertices)
      .def("is_qubit_vertex", &HypergraphCircuit::is_qubit_vertex)
      .def("add_hyperedge",
           [](HypergraphCircuit& h, const std::vector<Vertex>& vs, int w) {
             h.add_hyperedge(vs, w);
           },
           py::arg("vertices"), py::arg("weight") = 1)
      .def("reset_hyperedges", &HypergraphCircuit::reset_hyperedges)
      .def("get_gate_of_vertex", &HypergraphCircuit::get_gate_of_vertex,
           py::arg("vertex"))
      .def("merge_hyperedge",
           [](HypergraphCircuit& h, const std::vector<Hyperedge>& es) {
             return h.merge_hyperedge(es);
           },
           py::arg("hyperedges"))
      .def("split_hyperedge",
           [](HypergraphCircuit& h, const Hyperedge& old,
              const std::vector<Hyperedge>& news) {
             h.split_hyperedge(old, news);
           },
           py::arg("old_hyperedge"), py::arg("new_hyperedge_list"))
      .def("requires_h_embedded_cu1",
           [](const HypergraphCircuit& h, const std::vector<int>& vertices,
              int weight) {
             return h.requires_h_embedded_cu1(Hyperedge(vertices, weight));
           },
           py::arg("vertices"), py::arg("weight") = 1)
      .def("get_hyperedge_subcircuit",
           [](const HypergraphCircuit& h, const std::vector<int>& vertices,
              int weight) {
             std::vector<std::tuple<int, std::vector<double>,
                                    std::vector<std::string>>>
                 out;
             for (const Command& c :
                  h.get_hyperedge_subcircuit(Hyperedge(vertices, weight))) {
               out.emplace_back(c.op_type, c.params, c.qubits);
             }
             return out;
           },
           py::arg("vertices"), py::arg("weight") = 1);

  m.def("hyperedge_cost",
        [](const HypergraphCircuit& h, const std::vector<int>& vertices,
           int weight, const Placement& placement, const Graph& server_graph) {
          return hyperedge_cost(h, Hyperedge(vertices, weight), placement,
                                server_graph);
        },
        py::arg("circuit"), py::arg("vertices"), py::arg("weight"),
        py::arg("placement"), py::arg("server_graph"));
  m.def("distribution_cost",
        [](const HypergraphCircuit& h, const Placement& placement,
           const Graph& server_graph) {
          return distribution_cost(h, placement, server_graph);
        },
        py::arg("circuit"), py::arg("placement"), py::arg("server_graph"));

  // ---- pacman -------------------------------------------------------------
  py::class_<Packet>(m, "Packet")
      .def_readonly("packet_index", &Packet::packet_index)
      .def_readonly("qubit_vertex", &Packet::qubit_vertex)
      .def_readonly("connected_server_index",
                    &Packet::connected_server_index)
      .def_readonly("gate_vertices", &Packet::gate_vertices);

  auto pkt = [](const Packet& p) {
    return py::make_tuple(p.packet_index, p.qubit_vertex,
                          p.connected_server_index, py::cast(p.gate_vertices));
  };

  auto packet_with_index = [](const PacMan& pm, int idx) -> Packet {
    for (const auto& [qv, ps] : pm.packets_by_qubit()) {
      (void)qv;
      for (const auto& p : ps) {
        if (p.packet_index == idx) return p;
      }
    }
    throw std::runtime_error("packet index not found");
  };

  py::class_<PacMan>(m, "PacMan")
      .def(py::init<const HypergraphCircuit&, const Placement&>())
      .def("packets_by_qubit",
           [&](const PacMan& pm) {
             py::dict d;
             for (const auto& [qv, ps] : pm.packets_by_qubit()) {
               py::list lst;
               for (const auto& p : ps) lst.append(pkt(p));
               d[py::int_(qv)] = lst;
             }
             return d;
           })
      .def("neighbouring_packets",
           [](const PacMan& pm) {
             py::dict d;
             for (const auto& [qv, nps] : pm.neighbouring_packets()) {
               py::list lst;
               for (const auto& np : nps) {
                 py::list inner;
                 for (const auto& p : np) inner.append(p.packet_index);
                 lst.append(inner);
               }
               d[py::int_(qv)] = lst;
             }
             return d;
           })
      .def("hopping_packets",
           [](const PacMan& pm) {
             py::dict d;
             for (const auto& [qv, hps] : pm.hopping_packets()) {
               py::list lst;
               for (const auto& hp : hps) {
                 lst.append(py::make_tuple(hp.first.packet_index,
                                           hp.second.packet_index));
               }
               d[py::int_(qv)] = lst;
             }
             return d;
           })
      .def("merged_packets",
           [](const PacMan& pm) {
             py::dict d;
             for (const auto& [qv, mps] : pm.merged_packets()) {
               py::list lst;
               for (const auto& mp : mps) {
                 py::list inner;
                 for (const auto& p : mp.packets) inner.append(p.packet_index);
                 lst.append(inner);
               }
               d[py::int_(qv)] = lst;
             }
             return d;
           })
      .def("get_all_embedded_packets",
           [](const PacMan& pm) {
             py::dict d;
             for (const auto& [qv, inner] : pm.get_all_embedded_packets()) {
               py::list lst;
               for (const auto& [hp, embedded] : inner) {
                 py::list emb;
                 for (const auto& p : embedded) emb.append(p.packet_index);
                 lst.append(py::make_tuple(
                     py::make_tuple(hp.first.packet_index,
                                    hp.second.packet_index),
                     emb));
               }
               d[py::int_(qv)] = lst;
             }
             return d;
           })
      .def("get_hypergraph_from_packets", &PacMan::get_hypergraph_from_packets)
      .def("get_nx_graph_merged",
           [](const PacMan& pm) {
             auto [edges, top] = pm.get_nx_graph_merged();
             py::list e;
             for (const auto& [a, b] : edges) {
               py::list la, lb;
               for (const auto& p : a.packets) la.append(p.packet_index);
               for (const auto& p : b.packets) lb.append(p.packet_index);
               e.append(py::make_tuple(la, lb));
             }
             py::set t;
             for (const auto& mp : top.iteration_order()) {
               py::list lm;
               for (const auto& p : mp.packets) lm.append(p.packet_index);
               t.add(py::tuple(lm));
             }
             return py::make_tuple(e, t);
           })
      .def("get_nx_graph_conflict",
           [](const PacMan& pm) {
             auto [edges, top] = pm.get_nx_graph_conflict();
             py::list e;
             for (const auto& [a, b] : edges) {
               e.append(py::make_tuple(
                   py::make_tuple(a.first.packet_index, a.second.packet_index),
                   py::make_tuple(b.first.packet_index, b.second.packet_index)));
             }
             py::set t;
             for (const auto& hp : top.iteration_order()) {
               t.add(py::make_tuple(hp.first.packet_index, hp.second.packet_index));
             }
             return py::make_tuple(e, t);
           })
      .def("get_conflict_hoppings",
           [&](const PacMan& pm, int a, int b) {
             std::vector<HoppingPacket> conflicts =
                 pm.get_conflict_hoppings({packet_with_index(pm, a),
                                           packet_with_index(pm, b)});
             py::list out;
             for (const auto& hp : conflicts) {
               out.append(py::make_tuple(hp.first.packet_index,
                                         hp.second.packet_index));
             }
             return out;
           })
      .def("get_containing_merged_packet",
           [&](const PacMan& pm, int idx) {
             const MergedPacket mp =
                 pm.get_containing_merged_packet(packet_with_index(pm, idx));
             py::list out;
             for (const auto& p : mp.packets) out.append(p.packet_index);
             return out;
           })
      .def("get_split_packets",
           [&](const PacMan& pm, int a, int b) {
             const Packet pa = packet_with_index(pm, a);
             const Packet pb = packet_with_index(pm, b);
             const MergedPacket mp = pm.get_containing_merged_packet(pa);
             auto [left, right] = pm.get_split_packets(mp, {pa, pb});
             py::list l, r;
             for (const auto& p : left.packets) l.append(p.packet_index);
             for (const auto& p : right.packets) r.append(p.packet_index);
             return py::make_tuple(l, r);
           })
      .def("get_connected_packets",
           [&](const PacMan& pm, int idx) {
             const auto cps =
                 pm.get_connected_packets(packet_with_index(pm, idx));
             py::list out;
             for (const auto& p : cps) out.append(p.packet_index);
             return out;
           })
      .def("get_connected_merged_packets",
           [&](const PacMan& pm, int idx) {
             const auto cmps = pm.get_connected_merged_packets(
                 pm.get_containing_merged_packet(packet_with_index(pm, idx)));
             py::list out;
             for (const auto& mp : cmps) {
               py::list l;
               for (const auto& p : mp.packets) l.append(p.packet_index);
               out.append(l);
             }
             return out;
           })
      .def("get_subsequent_neighbouring_packet",
           [&](const PacMan& pm, int idx) -> std::optional<int> {
             const std::optional<Packet> p =
                 pm.get_subsequent_neighbouring_packet(
                     packet_with_index(pm, idx));
             if (!p.has_value()) return std::nullopt;
             return p->packet_index;
           },
           py::arg("packet_index"))
      .def("get_subsequent_hopping_packet",
           [&](const PacMan& pm, int idx) -> std::optional<int> {
             const std::optional<Packet> p =
                 pm.get_subsequent_hopping_packet(packet_with_index(pm, idx));
             if (!p.has_value()) return std::nullopt;
             return p->packet_index;
           },
           py::arg("packet_index"))
      .def("get_next_packet",
           [&](const PacMan& pm, int idx) -> std::optional<int> {
             const std::optional<Packet> p =
                 pm.get_next_packet(packet_with_index(pm, idx));
             if (!p.has_value()) return std::nullopt;
             return p->packet_index;
           },
           py::arg("packet_index"))
      .def("get_all_packets",
           [&](const PacMan& pm) {
             py::list out;
             for (const Packet& p : pm.get_all_packets()) out.append(pkt(p));
             return out;
           })
      .def("get_all_merged_packets",
           [](const PacMan& pm) {
             py::list out;
             for (const MergedPacket& mp : pm.get_all_merged_packets()) {
               py::list l;
               for (const Packet& p : mp.packets) l.append(p.packet_index);
               out.append(l);
             }
             return out;
           })
      .def("get_connected_server", &PacMan::get_connected_server,
           py::arg("qubit_vertex"), py::arg("gate_vertex"))
      .def("are_neighbouring_packets",
           [&](const PacMan& pm, int a, int b) {
             return pm.are_neighbouring_packets(packet_with_index(pm, a),
                                                packet_with_index(pm, b));
           },
           py::arg("a"), py::arg("b"))
      .def("are_hoppable_packets",
           [&](const PacMan& pm, int a, int b) {
             return pm.are_hoppable_packets(packet_with_index(pm, a),
                                            packet_with_index(pm, b));
           },
           py::arg("a"), py::arg("b"))
      .def("get_embedded_packets",
           [&](const PacMan& pm, int a, int b) {
             const std::vector<Packet> embedded = pm.get_embedded_packets(
                 {packet_with_index(pm, a), packet_with_index(pm, b)});
             py::list out;
             for (const Packet& p : embedded) out.append(p.packet_index);
             return out;
           },
           py::arg("a"), py::arg("b"))
      .def("get_all_embedded_packets_for_qubit_vertex",
           [&](const PacMan& pm, int qv) {
             const auto inner =
                 pm.get_all_embedded_packets_for_qubit_vertex(qv);
             py::list out;
             for (const auto& [hp, embedded] : inner) {
               py::list emb;
               for (const Packet& p : embedded) emb.append(p.packet_index);
               out.append(py::make_tuple(
                   py::make_tuple(hp.first.packet_index,
                                  hp.second.packet_index),
                   emb));
             }
             return out;
           },
           py::arg("qubit_vertex"))
      .def("get_hopping_packet_from_embedded_packet",
           [&](const PacMan& pm, int idx) {
             const HoppingPacket hp =
                 pm.get_hopping_packet_from_embedded_packet(
                     packet_with_index(pm, idx));
             return py::make_tuple(hp.first.packet_index,
                                   hp.second.packet_index);
           },
           py::arg("packet_index"))
      .def("get_hopping_packets_within",
           [&](const PacMan& pm,
               const std::vector<std::vector<int>>& merged_index_lists) {
             PySet<MergedPacket> mps(&py_hash_merged, &py_eq_merged);
             for (const auto& idxs : merged_index_lists) {
               MergedPacket mp;
               for (const int idx : idxs) {
                 mp.packets.push_back(packet_with_index(pm, idx));
               }
               mps.add(mp);
             }
             const PySet<HoppingPacket> hps =
                 pm.get_hopping_packets_within(mps);
             py::list out;
             for (const HoppingPacket& hp : hps.iteration_order()) {
               out.append(py::make_tuple(hp.first.packet_index,
                                         hp.second.packet_index));
             }
             return out;
           },
           py::arg("merged_packets"))
      .def("is_packet_embedded",
           [&](const PacMan& pm, int idx) {
             return pm.is_packet_embedded(packet_with_index(pm, idx));
           },
           py::arg("packet_index"));

  // ---- refiners ------------------------------------------------------------
  m.def("vertex_cover_refine",
        [](const HypergraphCircuit& h, const Placement& p, const Graph& g) {
          Distribution d{h, p, g};
          vertex_cover_refine(d);
          return py::make_tuple(d.circuit, d.placement);
        },
        py::arg("circuit"), py::arg("placement"), py::arg("server_graph"));

  // ---- distribution / gain manager -----------------------------------------
  py::class_<Distribution, std::shared_ptr<Distribution>>(m, "Distribution")
      .def(py::init([](const HypergraphCircuit& h, const Placement& p,
                       const Graph& g,
                       const std::map<int, std::vector<int>>& server_qubits,
                       const std::map<int, int>& server_ebit_mem) {
             Distribution d{h, p, g};
             d.server_qubits = server_qubits;
             if (server_ebit_mem.empty()) {
               for (const auto& [s, qs] : server_qubits) {
                 (void)qs;
                 d.server_ebit_mem[s] = -1;
               }
             } else {
               d.server_ebit_mem = server_ebit_mem;
             }
             return d;
           }),
           py::arg("circuit"), py::arg("placement"), py::arg("server_graph"),
           py::arg("server_qubits"),
           py::arg("server_ebit_mem") = std::map<int, int>{})
      // Mirrors `Distribution(dist_circ, placement, network)`.
      .def(py::init([](const HypergraphCircuit& h, const Placement& p,
                       const NISQNetwork& network) {
             return make_distribution(h, p, network);
           }),
           py::arg("circuit"), py::arg("placement"), py::arg("network"))
      .def_readwrite("circuit", &Distribution::circuit)
      .def_readwrite("placement", &Distribution::placement)
      .def_readonly("server_graph", &Distribution::server_graph)
      .def_readonly("server_qubits", &Distribution::server_qubits)
      .def_readonly("server_ebit_mem", &Distribution::server_ebit_mem)
      .def("cost", &Distribution::cost)
      .def("is_valid", &Distribution::is_valid)
      .def("non_local_gate_list", &Distribution::non_local_gate_list)
      .def("non_local_gate_count", &Distribution::non_local_gate_count)
      .def("detached_gate_list", &Distribution::detached_gate_list)
      .def("detached_gate_count", &Distribution::detached_gate_count)
      .def("to_pytket_circuit",
           [](std::shared_ptr<Distribution> d, bool satisfy_bound,
              bool allow_update) {
             ToPytketCircuitResult res;
             try {
               res = to_pytket_circuit(*d, satisfy_bound, allow_update);
             } catch (const ConstraintException& e) {
               py::object mod =
                   py::module_::import("pytket_dqc_cpp.utils.circuit_analysis");
               py::object py_exc_cls = mod.attr("ConstraintException");
               py::object py_exc = py_exc_cls(e.what(), e.server);
               if (e.v_gate >= 0) {
                 py_exc.attr("v_gate") = e.v_gate;
               }
               py::set_error(py_exc_cls, py_exc);
               throw py::error_already_set();
             }
             if (res.cost_mismatch_detached_embedded) {
               const std::string msg =
                   "Detected a gate that is both detached and embedded. As a "
                   "consequence, the estimated cost did not match actual "
                   "cost. Estimate: " +
                   std::to_string(res.estimated_cost) +
                   ", real: " + std::to_string(res.real_ebit_cost) + ".";
               py::module_::import("warnings").attr("warn")(msg);
             }
             return rebuild_pytket_circuit(res.circuit);
           },
           py::arg("satisfy_bound") = true, py::arg("allow_update") = false);

  m.def("eager_h_type_merge_refine",
        [](std::shared_ptr<Distribution> d) {
          return eager_h_type_merge_refine(*d);
        },
        py::arg("distribution"));

  m.def("neighbouring_d_type_merge_refine",
        [](std::shared_ptr<Distribution> d) {
          return neighbouring_d_type_merge_refine(*d);
        },
        py::arg("distribution"));

  m.def("intertwined_d_type_merge_refine",
        [](std::shared_ptr<Distribution> d) {
          return intertwined_d_type_merge_refine(*d);
        },
        py::arg("distribution"));

  m.def("boundary_reallocation_refine",
        [](std::shared_ptr<Distribution> d,
           const std::vector<Vertex>& fixed_vertices, int num_rounds,
           double stop_parameter, std::optional<std::uint64_t> seed,
           std::optional<int> cache_limit) {
          return boundary_reallocation_refine(*d, fixed_vertices, num_rounds,
                                              stop_parameter, seed,
                                              cache_limit);
        },
        py::arg("distribution"), py::arg("fixed_vertices") = std::vector<Vertex>{},
        py::arg("num_rounds") = 10, py::arg("stop_parameter") = 0.05,
        py::arg("seed") = std::nullopt,
        py::arg("cache_limit") = std::nullopt);

  m.def("detached_gates_refine",
        [](std::shared_ptr<Distribution> d, int num_rounds,
           double stop_parameter, std::optional<std::uint64_t> seed,
           std::optional<int> cache_limit) {
          return detached_gates_refine(*d, num_rounds, stop_parameter, seed,
                                       cache_limit);
        },
        py::arg("distribution"), py::arg("num_rounds") = 10,
        py::arg("stop_parameter") = 0.05, py::arg("seed") = std::nullopt,
        py::arg("cache_limit") = std::nullopt);

  m.def("boundary_reallocation_refine_with_state",
        [](std::shared_ptr<Distribution> d,
           const std::vector<Vertex>& fixed_vertices, int num_rounds,
           double stop_parameter, std::optional<std::uint64_t> seed,
           std::optional<int> cache_limit,
           const std::optional<std::vector<std::uint32_t>>& rng_state) {
          return boundary_reallocation_refine_with_state(
              *d, fixed_vertices, num_rounds, stop_parameter, seed, cache_limit,
              rng_state);
        },
        py::arg("distribution"),
        py::arg("fixed_vertices") = std::vector<Vertex>{},
        py::arg("num_rounds") = 10, py::arg("stop_parameter") = 0.05,
        py::arg("seed") = std::nullopt,
        py::arg("cache_limit") = std::nullopt,
        py::arg("rng_state") = std::nullopt);

  m.def("detached_gates_refine_with_state",
        [](std::shared_ptr<Distribution> d, int num_rounds,
           double stop_parameter, std::optional<std::uint64_t> seed,
           std::optional<int> cache_limit,
           const std::optional<std::vector<std::uint32_t>>& rng_state) {
          return detached_gates_refine_with_state(
              *d, num_rounds, stop_parameter, seed, cache_limit, rng_state);
        },
        py::arg("distribution"), py::arg("num_rounds") = 10,
        py::arg("stop_parameter") = 0.05, py::arg("seed") = std::nullopt,
        py::arg("cache_limit") = std::nullopt,
        py::arg("rng_state") = std::nullopt);

  py::class_<GainManager>(m, "GainManager")
      .def(py::init(
               [](std::shared_ptr<Distribution> d, int max_key_size) {
                 return GainManager(d, max_key_size);
               }),
           py::arg("distribution"), py::arg("max_key_size") = 5)
      .def("update_cost", &GainManager::update_cost, py::arg("hyperedge"))
      .def("split_hyperedge_gain", &GainManager::split_hyperedge_gain,
           py::arg("old_hyperedge"), py::arg("new_hyperedge_list"))
      .def("split_hyperedge", &GainManager::split_hyperedge,
           py::arg("old_hyperedge"), py::arg("new_hyperedge_list"),
           py::arg("recalculate_cost") = true)
      .def("merge_hyperedge_gain", &GainManager::merge_hyperedge_gain,
           py::arg("to_merge_hyperedge_list"))
      .def("merge_hyperedge", &GainManager::merge_hyperedge,
           py::arg("to_merge_hyperedge_list"),
           py::arg("recalculate_cost") = true)
      .def("move_vertex_gain", &GainManager::move_vertex_gain,
           py::arg("vertex"), py::arg("new_server"))
      .def("move_vertex", &GainManager::move_vertex, py::arg("vertex"),
           py::arg("server"), py::arg("recalculate_cost") = true)
      .def("is_move_valid", &GainManager::is_move_valid, py::arg("vertex"),
           py::arg("server"))
      .def("current_server", &GainManager::current_server, py::arg("vertex"))
      .def("set_max_key_size", &GainManager::set_max_key_size,
           py::arg("max_key_size"))
      .def_property_readonly("occupancy", &GainManager::occupancy)
      .def_property_readonly("hyperedge_cost_map",
                             &GainManager::hyperedge_cost_map)
      .def_property_readonly("requires_h_embedded_cu1",
                             &GainManager::requires_h_embedded_cu1);

  // ---- allocators ----------------------------------------------------------
  // Mirrors `Allocator.allocate(circ, network)`: each allocator takes a
  // `NISQNetwork` and returns a fully-populated `Distribution`.
  m.def("random_allocate",
        [](const HypergraphCircuit& h, const NISQNetwork& network,
           std::optional<std::uint64_t> seed) {
          return random_allocate(h, network.server_coupling(),
                                 network.server_qubits(), network.get_server_nx(),
                                 network.server_ebit_mem(), seed);
        },
        py::arg("circuit"), py::arg("network"), py::arg("seed") = std::nullopt);
  m.def("random_allocate_with_state",
        [](const HypergraphCircuit& h, const NISQNetwork& network,
           std::optional<std::uint64_t> seed,
           const std::optional<std::vector<std::uint32_t>>& rng_state) {
          return random_allocate_with_state(
              h, network.server_coupling(), network.server_qubits(),
              network.get_server_nx(), network.server_ebit_mem(), seed,
              rng_state);
        },
        py::arg("circuit"), py::arg("network"), py::arg("seed") = std::nullopt,
        py::arg("rng_state") = std::nullopt);
  m.def("ordered_allocate",
        [](const HypergraphCircuit& h, const NISQNetwork& network) {
          return ordered_allocate(h, network.server_qubits(),
                                  network.get_server_nx(),
                                  network.server_ebit_mem());
        },
        py::arg("circuit"), py::arg("network"));
  m.def("brute_allocate",
        [](const HypergraphCircuit& h, const NISQNetwork& network) {
          return brute_allocate(h, network.server_coupling(),
                                network.server_qubits(), network.get_server_nx(),
                                network.server_ebit_mem());
        },
        py::arg("circuit"), py::arg("network"));

  m.def("acceptance_criterion", &acceptance_criterion, py::arg("gain"),
        py::arg("iteration"), py::arg("initial_temperature") = 1.0);

  m.def("annealing_allocate",
        [](const HypergraphCircuit& h, const NISQNetwork& network,
           int iterations, double initial_temperature,
           std::optional<int> cache_limit, std::optional<std::uint64_t> seed,
           std::optional<std::uint64_t> initial_place_seed) {
          return annealing_allocate(h, network.server_coupling(),
                                    network.server_qubits(),
                                    network.get_server_nx(),
                                    network.server_ebit_mem(), iterations,
                                    initial_temperature, cache_limit, seed,
                                    initial_place_seed);
        },
        py::arg("circuit"), py::arg("network"),
        py::arg("iterations") = 30000, py::arg("initial_temperature") = 3.0,
        py::arg("cache_limit") = std::nullopt,
        py::arg("seed") = std::nullopt,
        py::arg("initial_place_seed") = std::nullopt);

  m.def("annealing_from_placement",
        [](const HypergraphCircuit& h, const Placement& p,
           const NISQNetwork& network, int iterations,
           double initial_temperature, std::optional<int> cache_limit,
           std::optional<std::uint64_t> seed,
           const std::optional<std::vector<std::uint32_t>>& rng_state) {
          return annealing_from_placement(
              h, p, network.server_coupling(), network.server_qubits(),
              network.get_server_nx(), network.server_ebit_mem(), iterations,
              initial_temperature, cache_limit, seed, rng_state);
        },
        py::arg("circuit"), py::arg("placement"), py::arg("network"),
        py::arg("iterations") = 30000, py::arg("initial_temperature") = 3.0,
        py::arg("cache_limit") = std::nullopt,
        py::arg("seed") = std::nullopt,
        py::arg("rng_state") = std::nullopt);

  m.def("annealing_from_placement_with_state",
        [](const HypergraphCircuit& h, const Placement& p,
           const NISQNetwork& network, int iterations,
           double initial_temperature, std::optional<int> cache_limit,
           std::optional<std::uint64_t> seed,
           const std::optional<std::vector<std::uint32_t>>& rng_state) {
          return annealing_from_placement_with_state(
              h, p, network.server_coupling(), network.server_qubits(),
              network.get_server_nx(), network.server_ebit_mem(), iterations,
              initial_temperature, cache_limit, seed, rng_state);
        },
        py::arg("circuit"), py::arg("placement"), py::arg("network"),
        py::arg("iterations") = 30000, py::arg("initial_temperature") = 3.0,
        py::arg("cache_limit") = std::nullopt,
        py::arg("seed") = std::nullopt,
        py::arg("rng_state") = std::nullopt);

  m.def("annealing_allocate_with_state",
        [](const HypergraphCircuit& h, const NISQNetwork& network,
           int iterations, double initial_temperature,
           std::optional<int> cache_limit, std::optional<std::uint64_t> seed,
           std::optional<std::uint64_t> initial_place_seed) {
          return annealing_allocate_with_state(
              h, network.server_coupling(), network.server_qubits(),
              network.get_server_nx(), network.server_ebit_mem(), iterations,
              initial_temperature, cache_limit, seed, initial_place_seed);
        },
        py::arg("circuit"), py::arg("network"),
        py::arg("iterations") = 30000, py::arg("initial_temperature") = 3.0,
        py::arg("cache_limit") = std::nullopt,
        py::arg("seed") = std::nullopt,
        py::arg("initial_place_seed") = std::nullopt);

  // ---- HypergraphPartitioning (KaHyPar) ------------------------------------
#ifdef DQC_HAS_KAHYPAR
  m.def("hypergraph_partitioning_allocate",
        [](const HypergraphCircuit& h, const NISQNetwork& network,
           std::optional<std::string> ini_path,
           std::optional<std::int64_t> seed) {
          const std::string path =
              ini_path.has_value() ? *ini_path : default_ini_path();
          return hypergraph_partitioning_allocate(
              h, network.server_coupling(), network.server_qubits(),
              network.get_server_nx(), network.server_ebit_mem(), path, seed);
        },
        py::arg("circuit"), py::arg("network"),
        py::arg("ini_path") = std::nullopt, py::arg("seed") = std::nullopt);
  m.def("hypergraph_partitioning_initial_placement",
        [](const HypergraphCircuit& h, const NISQNetwork& network,
           std::optional<std::string> ini_path,
           std::optional<std::int64_t> seed) {
          const std::string path =
              ini_path.has_value() ? *ini_path : default_ini_path();
          return hypergraph_partitioning_initial_placement(
              h, network.server_coupling(), network.server_qubits(), path,
              seed);
        },
        py::arg("circuit"), py::arg("network"),
        py::arg("ini_path") = std::nullopt, py::arg("seed") = std::nullopt);
#endif  // DQC_HAS_KAHYPAR

  // ---- NISQNetwork ---------------------------------------------------------
  py::class_<NISQNetwork>(m, "NISQNetwork")
      .def(py::init([](const py::list& server_coupling,
                       const py::dict& server_qubits,
                       const py::dict& server_ebit_mem) {
             std::vector<std::pair<int, int>> coupling;
             coupling.reserve(server_coupling.size());
             for (const auto& item : server_coupling) {
               const py::sequence pair = item.cast<py::sequence>();
               coupling.emplace_back(pair[0].cast<int>(), pair[1].cast<int>());
             }
             // Preserve dict insertion order (the reference's occupancy keys
             // follow `server_qubits` insertion order).
             std::vector<std::pair<int, std::vector<int>>> sq;
             sq.reserve(server_qubits.size());
             for (const auto& [k, v] : server_qubits) {
               sq.emplace_back(k.cast<int>(), v.cast<std::vector<int>>());
             }
             std::map<int, int> em;
             for (const auto& [k, v] : server_ebit_mem) {
               em[k.cast<int>()] = v.cast<int>();
             }
             return NISQNetwork(coupling, sq, em);
           }),
           py::arg("server_coupling"), py::arg("server_qubits"),
           py::arg("server_ebit_mem") = py::dict{})
      .def_property_readonly(
          "server_coupling",
          [](const NISQNetwork& n) {
            py::list out;
            for (const auto& [u, v] : n.server_coupling()) {
              py::list pair;
              pair.append(u);
              pair.append(v);
              out.append(pair);
            }
            return out;
          })
      .def_property_readonly(
          "server_qubits",
          [](const NISQNetwork& n) {
            py::dict out;
            for (const auto& [s, qs] : n.server_qubits()) {
              out[py::int_(s)] = py::cast(qs);
            }
            return out;
          })
      .def_property_readonly(
          "server_ebit_mem",
          [](const NISQNetwork& n) {
            py::dict out;
            for (const auto& [s, bound] : n.server_ebit_mem()) {
              out[py::int_(s)] = bound;
            }
            return out;
          })
      .def("get_server_list", &NISQNetwork::get_server_list)
      .def("get_qubit_list", &NISQNetwork::get_qubit_list)
      .def("can_implement", &NISQNetwork::can_implement, py::arg("dist_circ"))
      .def("is_placement", &NISQNetwork::is_placement, py::arg("placement"))
      .def("get_server_nx", &NISQNetwork::get_server_nx)
      .def("get_nisq_nx", &NISQNetwork::get_nisq_nx)
      .def("to_dict",
           [](const NISQNetwork& n) {
             py::dict d;
             py::list coupling;
             for (const auto& [u, v] : n.server_coupling()) {
               py::list pair;
               pair.append(u);
               pair.append(v);
               coupling.append(pair);
             }
             d["server_coupling"] = coupling;
             py::dict sq;
             for (const auto& [s, qs] : n.server_qubits()) {
               sq[py::int_(s)] = py::cast(qs);
             }
             d["server_qubits"] = sq;
             py::dict em;
             for (const auto& [s, bound] : n.server_ebit_mem()) {
               em[py::int_(s)] = bound;
             }
             d["server_ebit_mem"] = em;
             return d;
           })
      .def_static("from_dict", &nq_from_dict, py::arg("network_dict"))
      .def("__eq__",
           [](const NISQNetwork& a, const NISQNetwork& b) { return a == b; });

  // ---- RepeatRefiner / SequenceRefiner -------------------------------------
  py::class_<PyRepeatRefiner>(m, "RepeatRefiner")
      .def(py::init([](py::object refiner) {
             return PyRepeatRefiner(to_callable(std::move(refiner)));
           }),
           py::arg("refiner"))
      .def("refine", &PyRepeatRefiner::refine, py::arg("distribution"))
      .def("__call__", &PyRepeatRefiner::refine, py::arg("distribution"));
  py::class_<PySequenceRefiner>(m, "SequenceRefiner")
      .def(py::init([](const py::list& refiner_list) {
             std::vector<py::function> fns;
             fns.reserve(refiner_list.size());
             for (const auto& item : refiner_list) {
               fns.push_back(to_callable(py::reinterpret_borrow<py::object>(item)));
             }
             return PySequenceRefiner(std::move(fns));
           }),
           py::arg("refiner_list"))
      .def("refine", &PySequenceRefiner::refine, py::arg("distribution"))
      .def("__call__", &PySequenceRefiner::refine, py::arg("distribution"));
}
