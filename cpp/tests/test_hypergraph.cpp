#include <catch2/catch_test_macros.hpp>

#include <set>
#include <vector>

#include "dqc/hypergraph.hpp"

using dqc::Hyperedge;
using dqc::Hypergraph;
using dqc::Placement;
using dqc::Vertex;

TEST_CASE("add vertices and hyperedges") {
  Hypergraph g;
  g.add_vertices({0, 1, 2});
  g.add_hyperedge({0, 1});
  g.add_hyperedge({0, 2});

  REQUIRE(g.hyperedge_list().size() == 2);
  REQUIRE(g.vertex_neighbours().at(0) == std::set<Vertex>{1, 2});
  REQUIRE(g.vertex_neighbours().at(1) == std::set<Vertex>{0});
  REQUIRE(g.vertex_neighbours().at(2) == std::set<Vertex>{0});
  REQUIRE(g.hyperedge_dict().at(0).size() == 2);
  REQUIRE(g.hyperedge_dict().at(1).size() == 1);
  REQUIRE(g.hyperedge_dict().at(2).size() == 1);
  REQUIRE(g.is_valid());
}

TEST_CASE("remove hyperedge updates neighbours") {
  Hypergraph g;
  g.add_vertices({0, 1, 2});
  g.add_hyperedge({0, 1});
  g.add_hyperedge({0, 2});

  g.remove_hyperedge(Hyperedge({0, 1}));

  REQUIRE(g.hyperedge_list().size() == 1);
  REQUIRE(g.vertex_neighbours().at(0) == std::set<Vertex>{2});
  REQUIRE(g.vertex_neighbours().at(1) == std::set<Vertex>{});
  REQUIRE(g.vertex_neighbours().at(2) == std::set<Vertex>{0});
}

TEST_CASE("merge hyperedges") {
  Hypergraph g;
  g.add_vertices({0, 1, 2});
  g.add_hyperedge({0, 1});
  g.add_hyperedge({0, 2});

  const Hyperedge m =
      g.merge_hyperedge({Hyperedge({0, 1}), Hyperedge({0, 2})});

  REQUIRE(m == Hyperedge({0, 1, 2}));
  REQUIRE(g.hyperedge_list().size() == 1);
  REQUIRE(g.hyperedge_list()[0] == Hyperedge({0, 1, 2}));
  REQUIRE(g.vertex_neighbours().at(0) == std::set<Vertex>{1, 2});
  REQUIRE(g.vertex_neighbours().at(1) == std::set<Vertex>{0, 2});
  REQUIRE(g.vertex_neighbours().at(2) == std::set<Vertex>{0, 1});
}

TEST_CASE("split hyperedge") {
  Hypergraph g;
  g.add_vertices({0, 1, 2});
  g.add_hyperedge({0, 1, 2});

  g.split_hyperedge(Hyperedge({0, 1, 2}),
                    {Hyperedge({0, 1}), Hyperedge({0, 2})});

  REQUIRE(g.hyperedge_list().size() == 2);
  REQUIRE(g.hyperedge_list()[0] == Hyperedge({0, 1}));
  REQUIRE(g.hyperedge_list()[1] == Hyperedge({0, 2}));
}

TEST_CASE("get boundary") {
  Hypergraph g;
  g.add_vertices({0, 1, 2});
  g.add_hyperedge({0, 1});
  g.add_hyperedge({0, 2});

  Placement p;
  p.set(0, 0);
  p.set(1, 1);
  p.set(2, 0);

  REQUIRE(g.is_placement(p));
  REQUIRE(g.get_boundary(p) == std::vector<Vertex>{0, 1});
}

TEST_CASE("weight one predicate") {
  Hypergraph g;
  g.add_vertices({0, 1});
  g.add_hyperedge({0, 1});
  REQUIRE(g.weight_one_predicate());

  g.add_hyperedge({0, 1}, 2);
  REQUIRE(!g.weight_one_predicate());
}

TEST_CASE("kahypar hyperedges format") {
  Hypergraph g;
  g.add_vertices({0, 1, 2});
  g.add_hyperedge({0, 1});
  g.add_hyperedge({0, 2});

  const auto [indices, flat] = g.kahypar_hyperedges();
  REQUIRE(indices == std::vector<int>{0, 2, 4});
  REQUIRE(flat == std::vector<Vertex>{0, 1, 0, 2});
}

TEST_CASE("is_valid requires a continuous vertex list") {
  Hypergraph g;
  g.add_vertices({0, 1, 2});
  REQUIRE(g.is_valid());

  // Missing vertex 1 -> not continuous.
  Hypergraph g2;
  g2.add_vertices({0, 2});
  REQUIRE(!g2.is_valid());

  // Duplicate-free continuous but starting at 1 -> not continuous.
  Hypergraph g3;
  g3.add_vertices({1, 2});
  REQUIRE(!g3.is_valid());
}

TEST_CASE("is_placement checks placed vertices match the hypergraph") {
  Hypergraph g;
  g.add_vertices({0, 1, 2});
  g.add_hyperedge({0, 1});

  Placement p;
  p.set(0, 0);
  p.set(1, 1);
  p.set(2, 0);
  REQUIRE(g.is_placement(p));

  // A hypergraph vertex that is not placed.
  Placement p_missing;
  p_missing.set(0, 0);
  p_missing.set(1, 1);
  REQUIRE(!g.is_placement(p_missing));

  // A placed vertex that is not in the hypergraph.
  Placement p_extra;
  p_extra.set(0, 0);
  p_extra.set(1, 1);
  p_extra.set(2, 0);
  p_extra.set(9, 0);
  REQUIRE(!g.is_placement(p_extra));
}
