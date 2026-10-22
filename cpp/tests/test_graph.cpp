#include <catch2/catch_test_macros.hpp>

#include <map>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "dqc/graph.hpp"
#include "dqc/graph_tools.hpp"

using dqc::Graph;

namespace {

Graph path_graph(const std::vector<std::pair<int, int>>& edges) {
  Graph g;
  for (const auto& [u, v] : edges) g.add_edge(u, v);
  return g;
}

std::vector<std::pair<int, int>> as_pairs(const std::vector<dqc::Edge>& edges) {
  std::vector<std::pair<int, int>> out;
  for (const auto& e : edges) out.push_back({e.u, e.v});
  return out;
}

}  // namespace

TEST_CASE("steiner_tree matches networkx (5-cycle + chord)") {
  Graph g = path_graph({{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 0}, {1, 4}});
  Graph t = dqc::steiner_tree(g, {0, 2, 4});

  REQUIRE(t.nodes() == std::vector<int>{0, 1, 2, 4});
  REQUIRE(as_pairs(t.edges()) ==
          (std::vector<std::pair<int, int>>{{0, 1}, {0, 4}, {1, 2}}));
  REQUIRE(dqc::direct_from_origin(t, 0) ==
          (std::vector<std::pair<int, int>>{{0, 1}, {0, 4}, {1, 2}}));
}

TEST_CASE("steiner_tree matches networkx (4-cycle + chord)") {
  Graph g = path_graph({{0, 1}, {1, 2}, {2, 3}, {3, 0}, {0, 2}});
  Graph t = dqc::steiner_tree(g, {0, 2});

  REQUIRE(t.nodes() == std::vector<int>{0, 2});
  REQUIRE(as_pairs(t.edges()) ==
          (std::vector<std::pair<int, int>>{{0, 2}}));
  REQUIRE(dqc::direct_from_origin(t, 2) ==
          (std::vector<std::pair<int, int>>{{2, 0}}));
}

TEST_CASE("steiner_tree singleton terminal") {
  Graph g = path_graph({{0, 1}, {1, 2}});
  Graph t = dqc::steiner_tree(g, {1});

  REQUIRE(t.nodes() == std::vector<int>{1});
  REQUIRE(t.edges().empty());
}

TEST_CASE("steiner_tree on a path (terminals 0,4)") {
  Graph g = path_graph({{0, 1}, {1, 2}, {2, 3}, {3, 4}});
  Graph t = dqc::steiner_tree(g, {0, 4});

  REQUIRE(t.nodes() == std::vector<int>{0, 1, 2, 3, 4});
  REQUIRE(as_pairs(t.edges()) ==
          (std::vector<std::pair<int, int>>{{0, 1}, {1, 2}, {2, 3}, {3, 4}}));
  REQUIRE(dqc::direct_from_origin(t, 0) ==
          (std::vector<std::pair<int, int>>{{0, 1}, {1, 2}, {2, 3}, {3, 4}}));
}

TEST_CASE("steiner_tree on a path (terminals 0,2,4)") {
  Graph g = path_graph({{0, 1}, {1, 2}, {2, 3}, {3, 4}});
  Graph t = dqc::steiner_tree(g, {0, 2, 4});

  REQUIRE(t.nodes() == std::vector<int>{0, 1, 2, 3, 4});
  REQUIRE(as_pairs(t.edges()) ==
          (std::vector<std::pair<int, int>>{{0, 1}, {1, 2}, {2, 3}, {3, 4}}));
  REQUIRE(dqc::direct_from_origin(t, 4) ==
          (std::vector<std::pair<int, int>>{{4, 3}, {3, 2}, {2, 1}, {1, 0}}));
}

TEST_CASE("multi-source dijkstra matches networkx") {
  Graph g;
  g.add_edge(0, 1, {{"weight", 2.0}});
  g.add_edge(1, 2, {{"weight", 1.0}});
  g.add_edge(0, 2, {{"weight", 3.0}});

  auto [dist, paths] = dqc::dijkstra_multisource(g, {0, 2});

  REQUIRE(dist.at(0) == 0.0);
  REQUIRE(dist.at(2) == 0.0);
  REQUIRE(dist.at(1) == 1.0);
  REQUIRE(paths.at(0) == std::vector<int>{0});
  REQUIRE(paths.at(2) == std::vector<int>{2});
  REQUIRE(paths.at(1) == std::vector<int>{2, 1});
}

TEST_CASE("graph basics: nodes, edges, weights, self-loops") {
  Graph g;
  g.add_edge(1, 2, {{"weight", 3.0}});
  g.add_edge(0, 1);
  g.add_edge(2, 3, {{"weight", 0.5}});
  g.add_edge(2, 2, {{"weight", 7.0}});  // self-loop

  // Nodes follow insertion order.
  REQUIRE(g.nodes() == std::vector<int>({1, 2, 0, 3}));
  REQUIRE(g.num_nodes() == 4);
  REQUIRE(g.has_node(3));
  REQUIRE(!g.has_node(9));

  // Edges follow networkx `Graph.edges()` order: for each node in node order,
  // emit (n, nbr) for neighbours not yet seen (so the first endpoint is the
  // earlier node in node order); self-loops appear once.
  const std::vector<dqc::Edge> es = g.edges();
  REQUIRE(as_pairs(es) ==
          (std::vector<std::pair<int, int>>{{1, 2}, {1, 0}, {2, 3}, {2, 2}}));

  // Weights default to 1; explicit weights are preserved.
  REQUIRE(g.has_edge(0, 1));
  REQUIRE(g.has_edge(1, 0));
  REQUIRE(!g.has_edge(0, 3));
  REQUIRE(g.degree(2) == 4);  // neighbours 1, 3 and self-loop counted twice
  REQUIRE(g.weight(1, 2) == 3.0);
  REQUIRE(g.weight(0, 1) == 1.0);
  REQUIRE(g.weight(1, 2, "missing", 9.0) == 9.0);
  REQUIRE(g.weight(2, 2) == 7.0);
}

TEST_CASE("graph remove_node and subgraph") {
  Graph g;
  g.add_edge(0, 1);
  g.add_edge(1, 2);
  g.add_edge(2, 3);
  g.add_edge(3, 0);
  g.add_edge(0, 2);

  Graph sub = g.subgraph({0, 1, 2});
  REQUIRE(sub.nodes() == std::vector<int>({0, 1, 2}));
  REQUIRE(as_pairs(sub.edges()) ==
          (std::vector<std::pair<int, int>>{{0, 1}, {0, 2}, {1, 2}}));

  g.remove_node(2);
  REQUIRE(!g.has_node(2));
  REQUIRE(g.nodes() == std::vector<int>({0, 1, 3}));
  REQUIRE(as_pairs(g.edges()) ==
          (std::vector<std::pair<int, int>>{{0, 1}, {0, 3}}));
  REQUIRE(g.degree(1) == 1);
}

TEST_CASE("graph edge_subgraph keeps the requested edges only") {
  Graph g;
  g.add_edge(0, 1);
  g.add_edge(1, 2);
  g.add_edge(2, 3);

  const Graph sub = g.edge_subgraph({{1, 2}});
  REQUIRE(as_pairs(sub.edges()) ==
          (std::vector<std::pair<int, int>>{{1, 2}}));
  REQUIRE(sub.nodes() == std::vector<int>({1, 2}));
}

