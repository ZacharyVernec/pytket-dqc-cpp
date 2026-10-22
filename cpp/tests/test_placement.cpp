#include <catch2/catch_test_macros.hpp>

#include <utility>
#include <vector>

#include "dqc/placement.hpp"

using dqc::Graph;
using dqc::Placement;

namespace {

Graph path4() {
  Graph g;
  for (int i = 0; i < 4; ++i) g.add_node(i);
  g.add_edge(0, 1);
  g.add_edge(1, 2);
  g.add_edge(2, 3);
  return g;
}

}  // namespace

TEST_CASE("placement get_vertices_in preserves insertion order") {
  Placement p;
  p.set(3, 2);
  p.set(0, 0);
  p.set(5, 1);
  p.set(7, 0);

  REQUIRE(p.get_vertices_in(0) == std::vector<int>{0, 7});
  REQUIRE(p.get_vertices_in(1) == std::vector<int>{5});
  REQUIRE(p.get_vertices_in(2) == std::vector<int>{3});
  REQUIRE(p.get_vertices_in(3).empty());
}

TEST_CASE("get_distribution_tree matches reference behaviour") {
  Placement p;
  p.set(3, 2);
  p.set(0, 0);
  p.set(5, 1);
  p.set(7, 0);

  const Graph g = path4();

  // servers_used = [2, 0, 1] (insertion order); tree over {0,1,2};
  // qubit_node 0 -> server 0.
  REQUIRE(dqc::get_distribution_tree(p, {0, 3, 5}, 0, g) ==
          (std::vector<std::pair<int, int>>{{0, 1}, {1, 2}}));

  // Singleton hyperedge -> empty distribution tree.
  REQUIRE(dqc::get_distribution_tree(p, {7}, 7, g) ==
          (std::vector<std::pair<int, int>>{}));
}
