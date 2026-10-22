#pragma once

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "dqc/op_type.hpp"

namespace dqc {

// A single-qubit gate operation (mirrors `pytket.Op`: type + params).
struct Op {
  int op_type = -1;
  std::vector<double> params;

  static Op create(int op_type, const std::vector<double>& params) {
    return Op{op_type, params};
  }
};

// Mirrors `pytket_dqc.utils.gateset.to_euler_with_two_hadamards`.
inline std::vector<Op> to_euler_with_two_hadamards(const std::vector<Op>& ops) {
  std::vector<int> hadamard_indices;
  for (int i = 0; i < static_cast<int>(ops.size()); ++i) {
    if (ops[i].op_type == kOpH) hadamard_indices.push_back(i);
  }
  const Op id_rz = Op::create(kOpRz, {0.0});
  const Op hadamard = Op::create(kOpH, {});
  const int hadamard_count = static_cast<int>(hadamard_indices.size());
  if (hadamard_count > 2) {
    throw std::runtime_error("There should not be more than 2 Hadamards.");
  }

  std::vector<Op> new_ops;
  if (hadamard_count == 2) {
    if (ops[0].op_type == kOpH) new_ops.push_back(id_rz);
    new_ops.insert(new_ops.end(), ops.begin(), ops.end());
    if (ops.back().op_type == kOpH) new_ops.push_back(id_rz);
  } else if (hadamard_count == 0) {
    if (ops.empty()) new_ops.push_back(id_rz);
    new_ops.insert(new_ops.end(), ops.begin(), ops.end());
    new_ops.push_back(hadamard);
    new_ops.push_back(id_rz);
    new_ops.push_back(hadamard);
    new_ops.push_back(id_rz);
  } else {  // exactly one Hadamard
    if (ops.size() > 3) {
      throw std::runtime_error(
          "There can only be up to 3 ops in this decomposition.");
    }
    const Op s_op = Op::create(kOpRz, {0.5});
    if (ops.size() == 1) {  // [H]
      new_ops = {s_op, hadamard, s_op, hadamard, s_op};
    } else if (ops.size() == 2) {  // [H, Op] or [Op, H]
      const int phase_op_index = (hadamard_indices[0] == 0) ? 1 : 0;
      const double phase = ops[phase_op_index].params[0];
      const Op new_phase_op = Op::create(kOpRz, {phase + 0.5});
      new_ops = {new_phase_op, hadamard, s_op, hadamard, s_op};
      if (phase_op_index) std::reverse(new_ops.begin(), new_ops.end());
    } else {  // [Op, H, Op]
      const double first_phase = ops[0].params[0];
      const double second_phase = ops[2].params[0];
      new_ops = {Op::create(kOpRz, {first_phase + 0.5}), hadamard, s_op,
                 hadamard, Op::create(kOpRz, {second_phase + 0.5})};
    }
  }
  if (new_ops.size() != 5) {
    throw std::runtime_error("Euler decomposition must produce 5 ops.");
  }
  return new_ops;
}

}  // namespace dqc
