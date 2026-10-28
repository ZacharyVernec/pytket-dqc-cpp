// Faithful reproduction of CPython's `random` module.
//
// The Python `random` module implements the Mersenne Twister (MT19937) of
// Matsumoto & Nishimura (mt19937ar.c, public domain) together with CPython's
// seeding convention from `Modules/_randommodule.c` (PSF-2.0):
//   - `random.seed(int)` feeds `init_by_array` with the little-endian 32-bit
//     words of |seed| (minimal representation, trailing zero words dropped);
//   - `random.random()` returns `genrand_res53()`.
//
// This class must reproduce CPython bit-for-bit so that a fixed seed yields
// the same sequence as the reference Python implementation.
#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

namespace dqc {

class PyRandom {
 public:
  // Mirrors `random.Random(seed)`. Defaults to non-deterministic OS entropy (std::nullopt).
  explicit PyRandom(std::optional<std::uint64_t> seed = std::nullopt);

  // Mirrors `random.seed(a)` for an integer seed (magnitude <= 2^64-1).
  // `std::nullopt` mirrors `random.seed(None)` (OS entropy, intentionally
  // non-deterministic). Negative seeds map to their absolute value; pass the
  // magnitude here.
  void seed(std::optional<std::uint64_t> a);

  // Seeds from an explicit key: the little-endian 32-bit words of the seed,
  // exactly as CPython passes them to `init_by_array`. Use for seeds larger
  // than 2^64-1 (or for direct validation of the seeding path).
  void seed_words(const std::vector<std::uint32_t>& key);

  // Mirrors `random.random()`.
  double random();

  // Mirrors `random.uniform(a, b)`.
  double uniform(double a, double b);

  // Mirrors `random.randrange(stop)` (equivalently `random._randbelow(stop)`).
  std::uint64_t randrange(std::uint64_t stop);
  // Mirrors `random.randrange(start, stop)`.
  std::uint64_t randrange(std::uint64_t start, std::uint64_t stop);

  // Mirrors `random._randbelow(n)`.
  std::uint64_t randbelow(std::uint64_t n);

  // Mirrors `random.choice(seq)`.
  template <typename T>
  const T& choice(const std::vector<T>& seq) {
    if (seq.empty()) {
      throw std::invalid_argument("choice() on empty sequence");
    }
    return seq[randbelow(seq.size())];
  }

  // Mirrors `random.getrandbits(k)`. Values are restricted to k <= 64, which
  // covers every use in pytket-dqc (randrange/choice over small ranges).
  std::uint64_t getrandbits(int k);

  // Set the 625-word Mersenne Twister internal state (mt_[0..623] and mti_).
  // Matches Python's random.getstate()[1].
  void set_state(const std::vector<std::uint32_t>& state625);

  // Return the 625-word Mersenne Twister internal state.
  std::vector<std::uint32_t> get_state() const;

 private:
  std::uint32_t genrand_uint32();
  void init_genrand(std::uint32_t s);
  void init_by_array(const std::uint32_t* key, std::size_t key_length);

  static constexpr int kN = 624;
  static constexpr int kM = 397;
  static constexpr std::uint32_t kMatrixA = 0x9908b0dfUL;
  static constexpr std::uint32_t kUpperMask = 0x80000000UL;
  static constexpr std::uint32_t kLowerMask = 0x7fffffffUL;

  std::uint32_t mt_[kN];
  int mti_;
};

}  // namespace dqc
