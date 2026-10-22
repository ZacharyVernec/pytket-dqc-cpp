#include <catch2/catch_test_macros.hpp>

#include <functional>
#include <vector>

#include "dqc/distribution.hpp"
#include "dqc/refiners/composition.hpp"

using dqc::Distribution;
using dqc::RefineFn;
using dqc::RepeatRefiner;
using dqc::SequenceRefiner;

namespace {

Distribution make_dist() { return Distribution{}; }

}  // namespace

TEST_CASE("RepeatRefiner repeats until no refinement is made",
          "[composition]") {
  int calls = 0;
  RefineFn f = [&](Distribution&) -> bool {
    ++calls;
    return calls < 5;  // true for calls 1..4, false at call 5
  };
  RepeatRefiner rr(f);
  Distribution d = make_dist();
  const bool made = rr.refine(d);
  REQUIRE(made == true);
  REQUIRE(calls == 5);  // one initial + 4 repeats
}

TEST_CASE("RepeatRefiner returns false when the first refine makes nothing",
          "[composition]") {
  int calls = 0;
  RefineFn f = [&](Distribution&) -> bool {
    ++calls;
    return false;
  };
  RepeatRefiner rr(f);
  Distribution d = make_dist();
  REQUIRE(rr.refine(d) == false);
  REQUIRE(calls == 1);
}

TEST_CASE("SequenceRefiner applies every refiner in order and ORs the results",
          "[composition]") {
  std::vector<int> order;
  RefineFn a = [&](Distribution&) -> bool {
    order.push_back(1);
    return false;
  };
  RefineFn b = [&](Distribution&) -> bool {
    order.push_back(2);
    return true;
  };
  RefineFn c = [&](Distribution&) -> bool {
    order.push_back(3);
    return false;
  };
  SequenceRefiner sr({a, b, c});
  Distribution d = make_dist();
  REQUIRE(sr.refine(d) == true);
  REQUIRE(order == std::vector<int>({1, 2, 3}));
}

TEST_CASE("SequenceRefiner returns false when nothing is refined",
          "[composition]") {
  RefineFn f = [&](Distribution&) -> bool { return false; };
  SequenceRefiner sr({f, f});
  Distribution d = make_dist();
  REQUIRE(sr.refine(d) == false);
}

TEST_CASE("RepeatRefiner(SequenceRefiner(...)) composes", "[composition]") {
  int calls = 0;
  RefineFn f = [&](Distribution&) -> bool {
    ++calls;
    return calls < 3;  // true for the first two SequenceRefiner passes
  };
  SequenceRefiner seq({f});
  RepeatRefiner rep(seq);
  Distribution d = make_dist();
  REQUIRE(rep.refine(d) == true);
  // Sequence pass 1: calls 1..2 -> true; Sequence pass 2: call 3 -> false.
  REQUIRE(calls == 3);
}
