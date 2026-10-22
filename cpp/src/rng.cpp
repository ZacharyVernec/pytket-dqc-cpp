#include "dqc/rng.hpp"

#include <random>

namespace dqc {

PyRandom::PyRandom(std::optional<std::uint64_t> seed) { this->seed(seed); }

void PyRandom::init_genrand(std::uint32_t s) {
  mt_[0] = s;
  for (mti_ = 1; mti_ < kN; ++mti_) {
    mt_[mti_] = (1812433253UL * (mt_[mti_ - 1] ^ (mt_[mti_ - 1] >> 30)) +
                 static_cast<std::uint32_t>(mti_));
  }
}

void PyRandom::init_by_array(const std::uint32_t* key, std::size_t key_length) {
  init_genrand(19650218UL);
  std::size_t i = 1;
  std::size_t j = 0;
  std::size_t k = (static_cast<std::size_t>(kN) > key_length)
                      ? static_cast<std::size_t>(kN)
                      : key_length;
  for (; k; --k) {
    mt_[i] = (mt_[i] ^ ((mt_[i - 1] ^ (mt_[i - 1] >> 30)) * 1664525UL)) +
             key[j] + static_cast<std::uint32_t>(j);
    ++i;
    ++j;
    if (i >= kN) {
      mt_[0] = mt_[kN - 1];
      i = 1;
    }
    if (j >= key_length) {
      j = 0;
    }
  }
  for (k = kN - 1; k; --k) {
    mt_[i] = (mt_[i] ^ ((mt_[i - 1] ^ (mt_[i - 1] >> 30)) * 1566083941UL)) -
             static_cast<std::uint32_t>(i);
    ++i;
    if (i >= kN) {
      mt_[0] = mt_[kN - 1];
      i = 1;
    }
  }
  mt_[0] = 0x80000000UL;
}

std::uint32_t PyRandom::genrand_uint32() {
  std::uint32_t y;
  static const std::uint32_t mag01[2] = {0x0UL, kMatrixA};
  if (mti_ >= kN) {
    int kk;
    if (mti_ == kN + 1) {
      init_genrand(5489UL);
    }
    for (kk = 0; kk < kN - kM; ++kk) {
      y = (mt_[kk] & kUpperMask) | (mt_[kk + 1] & kLowerMask);
      mt_[kk] = mt_[kk + kM] ^ (y >> 1) ^ mag01[y & 0x1UL];
    }
    for (; kk < kN - 1; ++kk) {
      y = (mt_[kk] & kUpperMask) | (mt_[kk + 1] & kLowerMask);
      mt_[kk] = mt_[kk + (kM - kN)] ^ (y >> 1) ^ mag01[y & 0x1UL];
    }
    y = (mt_[kN - 1] & kUpperMask) | (mt_[0] & kLowerMask);
    mt_[kN - 1] = mt_[kM - 1] ^ (y >> 1) ^ mag01[y & 0x1UL];
    mti_ = 0;
  }
  y = mt_[mti_++];
  y ^= (y >> 11);
  y ^= (y << 7) & 0x9d2c5680UL;
  y ^= (y << 15) & 0xefc60000UL;
  y ^= (y >> 18);
  return y;
}

void PyRandom::seed(std::optional<std::uint64_t> a) {
  if (!a.has_value()) {
    // `random.seed(None)` draws from OS entropy (non-deterministic). There is
    // no defined cross-language equivalence for seed=None.
    std::random_device rd;
    std::vector<std::uint32_t> key(625);
    for (auto& w : key) {
      w = rd();
    }
    init_by_array(key.data(), key.size());
    return;
  }

  // Little-endian 32-bit words of |seed|, minimal representation. Matches
  // CPython `random_seed` + `_PyLong_AsByteArray` (little_endian=1, unsigned):
  // n = minimal byte length, i = (n-1)/4 + 1 words.
  std::uint64_t v = *a;
  std::vector<std::uint32_t> key;
  do {
    key.push_back(static_cast<std::uint32_t>(v & 0xFFFFFFFFULL));
    v >>= 32;
  } while (v != 0);
  init_by_array(key.data(), key.size());
}

void PyRandom::seed_words(const std::vector<std::uint32_t>& key) {
  init_by_array(key.data(), key.size());
}

double PyRandom::random() {
  const std::uint32_t a = genrand_uint32() >> 5;  // 27 bits
  const std::uint32_t b = genrand_uint32() >> 6;  // 26 bits
  return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0);
}

double PyRandom::uniform(double a, double b) { return a + (b - a) * random(); }

std::uint64_t PyRandom::getrandbits(int k) {
  if (k < 0) {
    throw std::invalid_argument("number of bits must be non-negative");
  }
  if (k == 0) {
    return 0;
  }
  if (k > 64) {
    throw std::invalid_argument("getrandbits(k>64) not supported");
  }
  const int words = (k - 1) / 32 + 1;
  std::uint64_t result = 0;
  for (int i = 0; i < words; ++i) {
    std::uint32_t r = genrand_uint32();
    const int remaining = k - 32 * i;
    if (remaining < 32) {
      r >>= (32 - remaining);  // keep the top `remaining` bits
    }
    result |= (static_cast<std::uint64_t>(r) << (32 * i));
  }
  return result;
}

std::uint64_t PyRandom::randbelow(std::uint64_t n) {
  if (n == 0) {
    return 0;
  }
  int k = 0;
  std::uint64_t tmp = n;
  while (tmp) {
    ++k;
    tmp >>= 1;
  }
  std::uint64_t r = getrandbits(k);
  while (r >= n) {
    r = getrandbits(k);
  }
  return r;
}

std::uint64_t PyRandom::randrange(std::uint64_t stop) { return randbelow(stop); }

std::uint64_t PyRandom::randrange(std::uint64_t start, std::uint64_t stop) {
  if (start >= stop) {
    throw std::invalid_argument("empty range for randrange");
  }
  return start + randbelow(stop - start);
}

}  // namespace dqc
