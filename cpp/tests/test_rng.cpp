#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <vector>

#include "dqc/rng.hpp"

using dqc::PyRandom;

namespace {

std::vector<double> draw_random(PyRandom& r, int n) {
  std::vector<double> out;
  for (int i = 0; i < n; ++i) {
    out.push_back(r.random());
  }
  return out;
}

}  // namespace

TEST_CASE("PyRandom.random matches CPython for known seeds") {
  struct Case {
    std::uint64_t seed;
    std::vector<double> expected;
  };
  const std::vector<Case> cases = {
      {0ULL,
       {0.8444218515250481, 0.7579544029403025, 0.420571580830845,
        0.25891675029296335, 0.5112747213686085, 0.4049341374504143,
        0.7837985890347726, 0.30331272607892745}},
      {1ULL,  // matches CPython random.seed(-1) (abs semantics)
       {0.13436424411240122, 0.8474337369372327, 0.763774618976614,
        0.2550690257394217, 0.49543508709194095, 0.4494910647887381,
        0.651592972722763, 0.7887233511355132}},
      {12345ULL,
       {0.41661987254534116, 0.010169169457068361, 0.8252065092537432,
        0.2986398551995928, 0.3684116894884757, 0.19366134904507426,
        0.5660081687288613, 0.1616878239293682}},
      {4294967296ULL,  // 2**32
       {0.11299430095636409, 0.41782886486292836, 0.0166763664992291,
        0.6117214625782752, 0.8056470212514185, 0.33151119826617415,
        0.5456080212796112, 0.4518156201690309}},
      {4294967297ULL,  // 2**32 + 1
       {0.2309331037176915, 0.5124118614342635, 0.4145826832477072,
        0.09166228226078477, 0.6052024383005872, 0.6859959008046673}},
      {9223372036854775813ULL,  // 2**63 + 5
       {0.3442386103531446, 0.4613981665936734, 0.10248021807011609,
        0.20359573722295332, 0.2739464947584218, 0.24218189926299227}},
  };
  for (const auto& c : cases) {
    PyRandom r(c.seed);
    const auto got = draw_random(r, static_cast<int>(c.expected.size()));
    REQUIRE(got == c.expected);
  }
}

TEST_CASE("PyRandom.seed_words matches CPython seed(2**64)") {
  // random.seed(2**64) -> key words [0, 0, 1] (little-endian).
  PyRandom r(0);
  r.seed_words({0, 0, 1});
  const std::vector<double> expected = {
      0.9776462743838505, 0.3711105964481608, 0.49965891230531634,
      0.6245596866262333, 0.6223913544058562, 0.09715059259060921,
      0.4362673638154494, 0.6040318759166183};
  REQUIRE(draw_random(r, 8) == expected);
}

TEST_CASE("PyRandom.uniform matches CPython") {
  PyRandom r(0);
  const std::vector<double> expected = {
      1.6888437030500962, 1.515908805880605, 0.84114316166169,
      0.5178335005859267, 1.022549442737217, 0.8098682749008286};
  for (const double e : expected) {
    REQUIRE(r.uniform(0.0, 2.0) == e);
  }
}

TEST_CASE("PyRandom.randrange matches CPython") {
  PyRandom r1(0);
  const std::vector<std::uint64_t> rr10 = {6, 6, 0, 4, 8, 7, 6, 4, 7, 5, 9, 3};
  for (const auto e : rr10) {
    REQUIRE(r1.randrange(10) == e);
  }

  PyRandom r2(0);
  const std::vector<std::uint64_t> rb5 = {3, 3, 0, 2, 4, 3, 3, 2, 3, 2, 4, 1};
  for (const auto e : rb5) {
    REQUIRE(r2.randrange(5) == e);
  }

  PyRandom r3(12345);
  const std::vector<std::uint64_t> rr317 = {9, 14, 3, 16, 16, 15, 7, 16, 8, 6};
  for (const auto e : rr317) {
    REQUIRE(r3.randrange(3, 17) == e);
  }
}

TEST_CASE("PyRandom.choice matches CPython") {
  PyRandom r(0);
  const std::vector<int> seq = {5, 7, 9, 11, 13};
  const std::vector<int> expected = {11, 11, 5, 9, 13, 11, 11, 9, 11, 9};
  for (const int e : expected) {
    REQUIRE(r.choice(seq) == e);
  }
}
