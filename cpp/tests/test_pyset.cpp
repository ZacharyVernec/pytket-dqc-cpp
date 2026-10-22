#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <utility>
#include <vector>

#include "dqc/hash.hpp"
#include "dqc/py_set.hpp"

using dqc::PySet;
using dqc::py_hash_frozenset_int;
using dqc::py_hash_int;
using dqc::py_hash_tuple2;

TEST_CASE("CPython hash helpers") {
  REQUIRE(py_hash_int(8) == 8u);
  REQUIRE(static_cast<std::int64_t>(py_hash_int(-1)) == -2);
  REQUIRE(static_cast<std::int64_t>(py_hash_int(-2)) == -2);

  REQUIRE(static_cast<std::int64_t>(py_hash_tuple2(py_hash_int(1), py_hash_int(2))) ==
          -3550055125485641917LL);
  REQUIRE(static_cast<std::int64_t>(py_hash_tuple2(py_hash_int(2), py_hash_int(8))) ==
          -4698950346736940769LL);
  REQUIRE(static_cast<std::int64_t>(py_hash_tuple2(py_hash_int(8), py_hash_int(2))) ==
          3535990567544810553LL);
  REQUIRE(static_cast<std::int64_t>(py_hash_tuple2(py_hash_int(0), py_hash_int(1))) ==
          -1950498447580522560LL);
  REQUIRE(static_cast<std::int64_t>(py_hash_tuple2(py_hash_int(10), py_hash_int(8))) ==
          5680612697361088710LL);

  const std::vector<int> a{0, 1};
  const std::vector<int> b{0, 2};
  const std::vector<int> c{0, 1, 2};
  REQUIRE(static_cast<std::int64_t>(py_hash_frozenset_int(a.begin(), a.end())) ==
          9000494312501845833LL);
  REQUIRE(static_cast<std::int64_t>(py_hash_frozenset_int(b.begin(), b.end())) ==
          -3804352684191256534LL);
  REQUIRE(static_cast<std::int64_t>(py_hash_frozenset_int(c.begin(), c.end())) ==
          -4850097147344926394LL);
}

TEST_CASE("PySet<int> matches CPython set iteration order") {
  PySet<int> s1(&dqc::py_hash_key, &dqc::py_eq_key);
  for (int x : {1, 2, 8}) s1.add(x);
  REQUIRE(s1.iteration_order() == std::vector<int>{8, 1, 2});

  PySet<int> s2(&dqc::py_hash_key, &dqc::py_eq_key);
  for (int x : {0, 8, 16}) s2.add(x);
  REQUIRE(s2.iteration_order() == std::vector<int>{0, 8, 16});

  PySet<int> s3(&dqc::py_hash_key, &dqc::py_eq_key);
  for (int x = 0; x < 8; ++x) s3.add(x);
  REQUIRE(s3.iteration_order() == std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7});

  PySet<int> s4(&dqc::py_hash_key, &dqc::py_eq_key);
  for (int x = 0; x <= 12; ++x) s4.add(x);
  REQUIRE(s4.iteration_order() ==
          std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
}

TEST_CASE("PySet<pair> matches CPython set iteration order") {
  PySet<std::pair<int, int>> s1(&dqc::py_hash_key, &dqc::py_eq_key);
  s1.add({1, 2});
  s1.add({2, 8});
  s1.add({8, 2});
  REQUIRE(s1.iteration_order() ==
          (std::vector<std::pair<int, int>>{{8, 2}, {1, 2}, {2, 8}}));

  PySet<std::pair<int, int>> s2(&dqc::py_hash_key, &dqc::py_eq_key);
  s2.add({8, 2});
  s2.add({1, 2});
  REQUIRE(s2.iteration_order() ==
          (std::vector<std::pair<int, int>>{{8, 2}, {1, 2}}));
}
