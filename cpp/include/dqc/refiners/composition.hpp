#pragma once

#include <functional>
#include <utility>
#include <vector>

#include "dqc/distribution.hpp"

namespace dqc {

// A refiner in the C++ core is any callable `bool(Distribution&)`; the
// reference's `Refiner` abstract base becomes `std::function`, so the
// composition classes below can wrap any of the primitive refiner functions
// (or other composition classes).
using RefineFn = std::function<bool(Distribution&)>;

// Mirrors `pytket_dqc.refiners.repeat_refiner.RepeatRefiner`: repeats the
// given refiner until it makes no further refinement, returning whether the
// first application made a refinement.
class RepeatRefiner {
 public:
  explicit RepeatRefiner(RefineFn refiner) : refiner_(std::move(refiner)) {}

  bool refine(Distribution& distribution) {
    bool refinement_made = refiner_(distribution);
    const bool one_refinement_made = refinement_made;
    while (refinement_made) {
      refinement_made = refiner_(distribution);
    }
    return one_refinement_made;
  }

  // Composition classes are callable so they can nest (mirrors the reference's
  // duck-typed `Refiner.refine`).
  bool operator()(Distribution& distribution) { return refine(distribution); }

 private:
  RefineFn refiner_;
};

// Mirrors `pytket_dqc.refiners.sequence_refiner.SequenceRefiner`: applies a
// sequence of refiners in order, returning whether any of them made a
// refinement (each is always applied; `|=` in the reference evaluates every
// element).
class SequenceRefiner {
 public:
  explicit SequenceRefiner(std::vector<RefineFn> refiner_list)
      : refiner_list_(std::move(refiner_list)) {}

  bool refine(Distribution& distribution) {
    bool refinement_made = false;
    for (const RefineFn& refiner : refiner_list_) {
      const bool made = refiner(distribution);
      refinement_made = refinement_made || made;
    }
    return refinement_made;
  }

  bool operator()(Distribution& distribution) { return refine(distribution); }

 private:
  std::vector<RefineFn> refiner_list_;
};

}  // namespace dqc
