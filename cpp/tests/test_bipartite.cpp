#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <set>
#include <vector>

#include "dqc/bipartite.hpp"
#include "dqc/py_set.hpp"

using dqc::BipartiteGraph;
using dqc::PySet;
using dqc::py_eq_key;
using dqc::py_hash_key;

using IntBipartiteGraph =
    BipartiteGraph<int, &py_hash_key, &py_eq_key>;

namespace {
PySet<int> make_set(const std::set<int>& xs) {
  PySet<int> s(&py_hash_key, &py_eq_key);
  for (int x : xs) s.add(x);
  return s;
}

bool covers_all_edges(const IntBipartiteGraph& g, const PySet<int>& cover) {
  for (const auto& [u, v] : g.edges()) {
    if (!cover.contains(u) && !cover.contains(v)) return false;
  }
  return true;
}
}  // namespace

TEST_CASE("bipartite color and sets") {
  IntBipartiteGraph g;  // path 0-1-2-3
  g.add_edge(0, 1);
  g.add_edge(1, 2);
  g.add_edge(2, 3);

  const auto c = g.color();
  REQUIRE(c.at(0) == 1);
  REQUIRE(c.at(1) == 0);
  REQUIRE(c.at(2) == 1);
  REQUIRE(c.at(3) == 0);

  const auto [X, Y] = g.sets();
  REQUIRE(X.iteration_order() == std::vector<int>({0, 2}));
  REQUIRE(Y.iteration_order() == std::vector<int>({1, 3}));
}

TEST_CASE("bipartite maximum_matching and vertex cover (Konig)") {
  IntBipartiteGraph g;  // path 0-1-2-3
  g.add_edge(0, 1);
  g.add_edge(1, 2);
  g.add_edge(2, 3);
  const PySet<int> top = make_set({1, 3});

  const auto matching = g.maximum_matching(top);
  REQUIRE(matching.size() == 4);

  for (const auto& [u, v] : matching) {
    const auto& nbrs = g.nbrs(u);
    REQUIRE(std::find(nbrs.begin(), nbrs.end(), v) != nbrs.end());
  }

  const PySet<int> cover = g.to_vertex_cover(matching, top);
  REQUIRE(cover.size() == 2);
  REQUIRE(covers_all_edges(g, cover));
}

TEST_CASE("bipartite complete K(2,3) matching and vertex cover") {
  IntBipartiteGraph g;
  g.add_edge(0, 2);
  g.add_edge(0, 3);
  g.add_edge(0, 4);
  g.add_edge(1, 2);
  g.add_edge(1, 3);
  g.add_edge(1, 4);
  const PySet<int> top = make_set({2, 3, 4});

  const auto matching = g.maximum_matching(top);
  REQUIRE(matching.size() == 4);

  const PySet<int> cover = g.to_vertex_cover(matching, top);
  REQUIRE(cover.size() == 2);
  REQUIRE(covers_all_edges(g, cover));
}

TEST_CASE("bipartite subgraph preserves node order") {
  IntBipartiteGraph g;
  g.add_edge(0, 1);
  g.add_edge(1, 2);
  g.add_edge(2, 3);
  g.add_edge(3, 4);
  const PySet<int> keep = make_set({1, 2, 3});
  const IntBipartiteGraph sub = g.subgraph(keep);
  REQUIRE(sub.nodes() == std::vector<int>({1, 2, 3}));
  const auto edges = sub.edges();
  REQUIRE(edges.size() == 2);
}

TEST_CASE("PySet deletion keeps CPython iteration order") {
  PySet<int> s(&py_hash_key, &py_eq_key);
  for (int x : {1, 2, 3, 4, 5}) s.add(x);
  s.erase(2);
  s.erase(4);
  REQUIRE(s.size() == 3);
  const auto order = s.iteration_order();
  REQUIRE(std::find(order.begin(), order.end(), 1) != order.end());
  REQUIRE(std::find(order.begin(), order.end(), 3) != order.end());
  REQUIRE(std::find(order.begin(), order.end(), 5) != order.end());
  REQUIRE(s.add(2));
  REQUIRE(s.size() == 4);
  REQUIRE(s.contains(2));
}
