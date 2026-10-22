#pragma once

#include <cmath>

namespace dqc {

// Mirrors `numpy.isclose(a, b)` with default tolerances
// (rtol=1e-05, atol=1e-08): |a - b| <= atol + rtol * |b|.
inline bool isclose(double a, double b, double rtol = 1e-05,
                    double atol = 1e-08) {
  return std::abs(a - b) <= (atol + rtol * std::abs(b));
}

// Mirrors Python's `%` operator for floats (result in [0, b) for b > 0).
inline double py_mod(double a, double b) {
  return a - std::floor(a / b) * b;
}

}  // namespace dqc
