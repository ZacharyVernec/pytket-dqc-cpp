#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <vector>

#include "dqc/gateset.hpp"
#include "dqc/op_type.hpp"

using dqc::Op;
using dqc::to_euler_with_two_hadamards;

namespace {
Op rz(double p) { return Op::create(dqc::kOpRz, {p}); }
Op H() { return Op::create(dqc::kOpH, {}); }

bool same_ops(const std::vector<Op>& a, const std::vector<Op>& b) {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (a[i].op_type != b[i].op_type) return false;
    if (a[i].params != b[i].params) return false;
  }
  return true;
}
}  // namespace

TEST_CASE("to_euler with two hadamards leaves Rz-H-Rz-H-Rz unchanged") {
  const std::vector<Op> ops = {rz(0.1), H(), rz(0.2), H(), rz(0.3)};
  REQUIRE(same_ops(to_euler_with_two_hadamards(ops), ops));
}

TEST_CASE("to_euler with two hadamards inserts identity Rz at the ends") {
  // [H, Rz, H] -> [Rz(0), H, Rz, H, Rz(0)]
  const std::vector<Op> ops = {H(), rz(0.2), H()};
  const std::vector<Op> expected = {rz(0.0), H(), rz(0.2), H(), rz(0.0)};
  REQUIRE(same_ops(to_euler_with_two_hadamards(ops), expected));

  // [H, Rz, H, Rz] -> [Rz(0), H, Rz, H, Rz] (leading H only)
  const std::vector<Op> ops2 = {H(), rz(0.2), H(), rz(0.3)};
  const std::vector<Op> expected2 = {rz(0.0), H(), rz(0.2), H(), rz(0.3)};
  REQUIRE(same_ops(to_euler_with_two_hadamards(ops2), expected2));

  // [Rz, H, Rz, H] -> [Rz, H, Rz, H, Rz(0)] (trailing H only)
  const std::vector<Op> ops3 = {rz(0.1), H(), rz(0.2), H()};
  const std::vector<Op> expected3 = {rz(0.1), H(), rz(0.2), H(), rz(0.0)};
  REQUIRE(same_ops(to_euler_with_two_hadamards(ops3), expected3));
}

TEST_CASE("to_euler with no hadamards appends H Rz H Rz") {
  // A single Rz -> [Rz(a), H, Rz(0), H, Rz(0)].
  const std::vector<Op> ops = {rz(0.1)};
  const std::vector<Op> expected = {rz(0.1), H(), rz(0.0), H(), rz(0.0)};
  REQUIRE(same_ops(to_euler_with_two_hadamards(ops), expected));

  // Empty input -> [Rz(0), H, Rz(0), H, Rz(0)].
  const std::vector<Op> empty_expected = {rz(0.0), H(), rz(0.0), H(), rz(0.0)};
  REQUIRE(same_ops(to_euler_with_two_hadamards({}), empty_expected));
}

TEST_CASE("to_euler rejects a 0-H list of more than one op") {
  // Mirrors the reference: `assert len(new_ops) == 5` fails when a 0-H list
  // of length > 1 would produce 6 ops.
  const std::vector<Op> ops = {rz(0.1), rz(0.2)};
  REQUIRE_THROWS(to_euler_with_two_hadamards(ops));
}

TEST_CASE("to_euler with a single hadamard") {
  // [H] -> [Rz(0.5), H, Rz(0.5), H, Rz(0.5)]
  REQUIRE(same_ops(to_euler_with_two_hadamards({H()}),
                   (std::vector<Op>{rz(0.5), H(), rz(0.5), H(), rz(0.5)})));

  // [Rz(a), H] -> [Rz(a+0.5), H, Rz(0.5), H, Rz(0.5)]
  REQUIRE(same_ops(to_euler_with_two_hadamards({rz(0.3), H()}),
                   (std::vector<Op>{rz(0.8), H(), rz(0.5), H(), rz(0.5)})));

  // [H, Rz(a)] -> [Rz(0.5), H, Rz(0.5), H, Rz(a+0.5)] (list reversed)
  REQUIRE(same_ops(to_euler_with_two_hadamards({H(), rz(0.3)}),
                   (std::vector<Op>{rz(0.5), H(), rz(0.5), H(), rz(0.8)})));

  // [Rz(a), H, Rz(b)] -> [Rz(a+0.5), H, Rz(0.5), H, Rz(b+0.5)]
  REQUIRE(same_ops(to_euler_with_two_hadamards({rz(0.3), H(), rz(0.4)}),
                   (std::vector<Op>{rz(0.8), H(), rz(0.5), H(), rz(0.9)})));
}

TEST_CASE("to_euler throws on too many hadamards") {
  const std::vector<Op> ops = {H(), H(), H()};
  REQUIRE_THROWS(to_euler_with_two_hadamards(ops));
}
