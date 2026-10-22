#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "dqc/hash.hpp"

namespace dqc {

// Faithful reproduction of CPython's `set` hash-table layout (setobject.c),
// restricted to the key types used in pytket-dqc. Iteration order over a set
// is ascending table-slot order, so reproducing the layout (including dummy
// slots after deletion) reproduces the order.
//
// Deletions mark slots as "dummy" (matching CPython's `set_discard_entry`),
// and insertions reuse the first dummy slot seen while probing (matching
// `set_add_entry`). `fill` (used + dummy) drives resizing exactly as in
// CPython: resize when `fill*5 >= mask*3` to the smallest power of 2 greater
// than `used*4` (used*2 when used > 50000).
template <typename Key>
class PySet {
 public:
  using HashFn = py_hash_t (*)(const Key&);
  using EqFn = bool (*)(const Key&, const Key&);

  PySet(HashFn hash, EqFn eq) : hash_(hash), eq_(eq) {
    table_.assign(kMinSize, Entry{});
  }

  bool contains(const Key& k) const {
    return find_slot(k, normalize(hash_(k))).has_value();
  }

  // Returns true if inserted (false if already present). Mirrors set_add_entry.
  bool add(const Key& k) {
    const py_hash_t h = normalize(hash_(k));
    const std::size_t mask = table_.size() - 1;
    std::size_t perturb = h;
    std::size_t i = h & mask;
    std::optional<std::size_t> freeslot;

    while (true) {
      Entry* entry = &table_[i];
      std::size_t probes = (i + kLinearProbes <= mask) ? kLinearProbes : 0;
      do {
        if (entry->state == State::kUnused) {
          if (freeslot.has_value()) {  // found_unused_or_dummy
            table_[*freeslot] = Entry{State::kUsed, h, k};
            used_ += 1;
          } else {  // found_unused
            *entry = Entry{State::kUsed, h, k};
            fill_ += 1;
            used_ += 1;
            if (fill_ * 5 >= mask * 3) {
              resize(used_ > 50000 ? used_ * 2 : used_ * 4);
            }
          }
          return true;
        }
        if (entry->state == State::kUsed && entry->hash == h &&
            eq_(entry->key, k)) {
          return false;  // found_active
        }
        if (entry->state == State::kDummy) {
          // CPython overwrites freeslot on every dummy, so the LAST dummy
          // probed before an unused slot is the one reused.
          freeslot = static_cast<std::size_t>(entry - table_.data());
        }
        ++entry;
      } while (probes--);
      perturb >>= kPerturbShift;
      i = (i * 5 + 1 + perturb) & mask;
    }
  }

  // Returns true if erased (false if absent). Mirrors set_discard_entry.
  bool erase(const Key& k) {
    const py_hash_t h = normalize(hash_(k));
    const std::optional<std::size_t> slot = find_slot(k, h);
    if (!slot.has_value()) return false;
    Entry& e = table_[*slot];
    e.state = State::kDummy;
    e.hash = 0;
    e.key = Key{};
    used_ -= 1;
    return true;
  }

  std::size_t size() const { return used_; }
  bool empty() const { return used_ == 0; }

  // Iteration order == ascending table-slot order (CPython set iteration).
  std::vector<Key> iteration_order() const {
    std::vector<Key> out;
    out.reserve(used_);
    for (const Entry& e : table_) {
      if (e.state == State::kUsed) out.push_back(e.key);
    }
    return out;
  }

 private:
  enum class State : std::uint8_t { kUnused = 0, kDummy = 1, kUsed = 2 };
  struct Entry {
    State state = State::kUnused;
    py_hash_t hash = 0;
    Key key{};
  };

  static py_hash_t normalize(py_hash_t h) {
    return (h == static_cast<py_hash_t>(0xFFFFFFFFFFFFFFFFULL))
               ? static_cast<py_hash_t>(0xFFFFFFFFFFFFFFFEULL)
               : h;
  }

  // Mirrors set_lookkey: returns the slot holding `k`, or nullopt if absent
  // (probing stops at the first UNUSED slot; dummies are skipped).
  std::optional<std::size_t> find_slot(const Key& k, py_hash_t h) const {
    const std::size_t mask = table_.size() - 1;
    std::size_t perturb = h;
    std::size_t i = h & mask;

    while (true) {
      const Entry* entry = &table_[i];
      std::size_t probes = (i + kLinearProbes <= mask) ? kLinearProbes : 0;
      do {
        if (entry->state == State::kUnused) return std::nullopt;
        if (entry->state == State::kUsed && entry->hash == h &&
            eq_(entry->key, k)) {
          return static_cast<std::size_t>(entry - table_.data());
        }
        ++entry;
      } while (probes--);
      perturb >>= kPerturbShift;
      i = (i * 5 + 1 + perturb) & mask;
    }
  }

  // Mirrors set_table_resize + set_insert_clean: rebuild into the smallest
  // power-of-two table > minused, re-inserting USED entries in old slot order
  // with their stored hashes.
  void resize(std::size_t minused) {
    std::size_t newsize = kMinSize;
    while (newsize <= minused) newsize <<= 1;
    std::vector<Entry> old = std::move(table_);
    table_.assign(newsize, Entry{});
    fill_ = 0;
    used_ = 0;
    for (const Entry& e : old) {
      if (e.state == State::kUsed) insert_clean(e.key, e.hash);
    }
  }

  // Mirrors set_insert_clean (no dummy entries present after a resize).
  void insert_clean(const Key& k, py_hash_t h) {
    const std::size_t mask = table_.size() - 1;
    std::size_t perturb = h;
    std::size_t i = h & mask;
    while (true) {
      Entry* entry = &table_[i];
      if (entry->state == State::kUnused) {
        *entry = Entry{State::kUsed, h, k};
        fill_ += 1;
        used_ += 1;
        return;
      }
      if (i + kLinearProbes <= mask) {
        for (std::size_t j = 0; j < kLinearProbes; ++j) {
          ++entry;
          if (entry->state == State::kUnused) {
            *entry = Entry{State::kUsed, h, k};
            fill_ += 1;
            used_ += 1;
            return;
          }
        }
      }
      perturb >>= kPerturbShift;
      i = (i * 5 + 1 + perturb) & mask;
    }
  }

  static constexpr std::size_t kMinSize = 8;
  static constexpr std::size_t kLinearProbes = 9;
  static constexpr std::size_t kPerturbShift = 5;

  HashFn hash_;
  EqFn eq_;
  std::vector<Entry> table_;
  std::size_t used_ = 0;
  std::size_t fill_ = 0;
};

// Key hash/equality specialisations used by the graph layer.
inline py_hash_t py_hash_key(const int& x) { return py_hash_int(x); }
inline bool py_eq_key(const int& a, const int& b) { return a == b; }

inline py_hash_t py_hash_key(const std::pair<int, int>& p) {
  return py_hash_tuple2(py_hash_int(p.first), py_hash_int(p.second));
}
inline bool py_eq_key(const std::pair<int, int>& a, const std::pair<int, int>& b) {
  return a == b;
}

}  // namespace dqc
