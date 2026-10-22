#pragma once

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>

namespace dqc {

// CPython 3.11-compatible hashing for the key types used in pytket-dqc
// (int, tuple, frozenset, str). int/tuple/frozenset hashes are deterministic
// (CPython does not apply hash randomization to these types); str hashes use
// SipHash-1-3 keyed by the PYTHONHASHSEED-derived secret.

using py_hash_t = std::uint64_t;

inline py_hash_t py_hash_int(std::int64_t x) {
  py_hash_t h = static_cast<py_hash_t>(static_cast<std::uint64_t>(x));
  return (h == static_cast<py_hash_t>(0xFFFFFFFFFFFFFFFFULL))
             ? static_cast<py_hash_t>(0xFFFFFFFFFFFFFFFEULL)
             : h;
}

// CPython `tuplehash` (xxHash, CPython >= 3.8), general length.
inline py_hash_t py_hash_tuple(const py_hash_t* hashes, std::size_t n) {
  constexpr py_hash_t kXxprime1 = 11400714785074694791ULL;
  constexpr py_hash_t kXxprime2 = 14029467366897019727ULL;
  constexpr py_hash_t kXxprime5 = 2870177450012600261ULL;

  py_hash_t acc = kXxprime5;
  for (std::size_t i = 0; i < n; ++i) {
    acc += hashes[i] * kXxprime2;
    acc = (acc << 31) | (acc >> 33);  // _PyHASH_XXROTATE (rotate left 31)
    acc *= kXxprime1;
  }
  acc += static_cast<py_hash_t>(n) ^ (kXxprime5 ^ 3527539ULL);
  if (acc == static_cast<py_hash_t>(0xFFFFFFFFFFFFFFFFULL)) {
    acc = 1546275796ULL;
  }
  return acc;
}

// CPython `tuplehash` specialised to length 2 (xxHash, CPython >= 3.8).
inline py_hash_t py_hash_tuple2(py_hash_t h0, py_hash_t h1) {
  const py_hash_t pair[2] = {h0, h1};
  return py_hash_tuple(pair, 2);
}

// CPython `_shuffle_bits` (setobject.c).
inline py_hash_t py_shuffle_bits(py_hash_t h) {
  return ((h ^ 89869747ULL) ^ (h << 16)) * 3644798167ULL;
}

// CPython `frozenset_hash`, general element hashes.
inline py_hash_t py_hash_frozenset(const py_hash_t* hashes, std::size_t n) {
  py_hash_t h = 0;
  for (std::size_t i = 0; i < n; ++i) {
    h ^= py_shuffle_bits(hashes[i]);
  }
  h ^= (static_cast<py_hash_t>(n) + 1) * 1927868237ULL;
  h ^= (h >> 11) ^ (h >> 25);
  h = h * 69069ULL + 907133923ULL;
  if (h == static_cast<py_hash_t>(0xFFFFFFFFFFFFFFFFULL)) {
    h = 590923713ULL;
  }
  return h;
}

// CPython `frozenset_hash` for a set of non-negative ints.
template <typename It>
py_hash_t py_hash_frozenset_int(It first, It last) {
  py_hash_t h = 0;
  std::size_t size = 0;
  for (It it = first; it != last; ++it) {
    h ^= py_shuffle_bits(py_hash_int(static_cast<std::int64_t>(*it)));
    ++size;
  }
  h ^= (static_cast<py_hash_t>(size) + 1) * 1927868237ULL;
  h ^= (h >> 11) ^ (h >> 25);
  h = h * 69069ULL + 907133923ULL;
  if (h == static_cast<py_hash_t>(0xFFFFFFFFFFFFFFFFULL)) {
    h = 590923713ULL;
  }
  return h;
}

// ---- CPython `str` hashing (SipHash-1-3, keyed by PYTHONHASHSEED) ----------

inline std::uint64_t rotl64(std::uint64_t x, int b) {
  return (x << b) | (x >> (64 - b));
}

inline std::uint64_t read_le64(const std::uint8_t* p) {
  std::uint64_t v = 0;
  for (int i = 0; i < 8; ++i) v |= static_cast<std::uint64_t>(p[i]) << (8 * i);
  return v;
}

// CPython `siphash13` (SipHash-1-3 with the standard finalisation).
inline std::uint64_t siphash13(std::uint64_t k0, std::uint64_t k1,
                               const std::uint8_t* src, std::size_t src_sz) {
  std::uint64_t b = static_cast<std::uint64_t>(src_sz) << 56;
  const std::uint8_t* in = src;

  std::uint64_t v0 = k0 ^ 0x736f6d6570736575ULL;
  std::uint64_t v1 = k1 ^ 0x646f72616e646f6dULL;
  std::uint64_t v2 = k0 ^ 0x6c7967656e657261ULL;
  std::uint64_t v3 = k1 ^ 0x7465646279746573ULL;

  auto half_round = [](std::uint64_t& a, std::uint64_t& b, std::uint64_t& c,
                       std::uint64_t& d, int s, int t) {
    a += b;
    c += d;
    b = rotl64(b, s) ^ a;
    d = rotl64(d, t) ^ c;
    a = rotl64(a, 32);
  };
  auto single_round = [&](std::uint64_t& a, std::uint64_t& b,
                          std::uint64_t& c, std::uint64_t& d) {
    half_round(a, b, c, d, 13, 16);
    half_round(c, b, a, d, 17, 21);
  };

  while (src_sz >= 8) {
    std::uint64_t mi = read_le64(in);
    in += 8;
    src_sz -= 8;
    v3 ^= mi;
    single_round(v0, v1, v2, v3);
    v0 ^= mi;
  }

  std::uint64_t t = 0;
  std::uint8_t* pt = reinterpret_cast<std::uint8_t*>(&t);
  switch (src_sz) {
    case 7: pt[6] = in[6]; [[fallthrough]];
    case 6: pt[5] = in[5]; [[fallthrough]];
    case 5: pt[4] = in[4]; [[fallthrough]];
    case 4: std::memcpy(pt, in, sizeof(std::uint32_t)); break;
    case 3: pt[2] = in[2]; [[fallthrough]];
    case 2: pt[1] = in[1]; [[fallthrough]];
    case 1: pt[0] = in[0]; [[fallthrough]];
    default: break;
  }
  b |= t;  // t is already little-endian on x86-64

  v3 ^= b;
  single_round(v0, v1, v2, v3);
  v0 ^= b;
  v2 ^= 0xff;
  single_round(v0, v1, v2, v3);
  single_round(v0, v1, v2, v3);
  single_round(v0, v1, v2, v3);

  return (v0 ^ v1) ^ (v2 ^ v3);
}

// CPython `lcg_urandom`: x(n+1) = (x(n)*214013 + 2531011) mod 2^32; byte n is
// bits 23..16 of x(n).
inline void lcg_urandom(std::uint32_t x0, std::uint8_t* buffer, std::size_t size) {
  std::uint32_t x = x0;
  for (std::size_t i = 0; i < size; ++i) {
    x *= 214013u;
    x += 2531011u;
    buffer[i] = static_cast<std::uint8_t>((x >> 16) & 0xffu);
  }
}

struct HashSecret {
  std::uint64_t k0 = 0;
  std::uint64_t k1 = 0;
};

// Derives the SipHash key from PYTHONHASHSEED, mirroring CPython's
// `_Py_HashRandomization_Init`:
//   - unset / "random" / invalid -> OS entropy (non-deterministic);
//   - 0                          -> zero key;
//   - integer in [1, 4294967295] -> lcg_urandom(seed, 24 bytes), key = LE64 of
//     the first two 8-byte words.
inline HashSecret py_hash_secret() {
  static const HashSecret secret = []() {
    HashSecret s;
    const char* env = std::getenv("PYTHONHASHSEED");
    bool deterministic = false;
    std::uint64_t seed = 0;
    if (env != nullptr && std::strcmp(env, "random") != 0) {
      char* end = nullptr;
      errno = 0;
      const unsigned long long v = std::strtoull(env, &end, 10);
      if (end != env && *end == '\0' && errno == 0 && v <= 4294967295ULL) {
        seed = v;
        deterministic = true;
      }
    }
    if (deterministic) {
      if (seed == 0) {
        s.k0 = 0;
        s.k1 = 0;
      } else {
        std::uint8_t bytes[24];
        lcg_urandom(static_cast<std::uint32_t>(seed), bytes, 24);
        s.k0 = read_le64(bytes);
        s.k1 = read_le64(bytes + 8);
      }
    } else {
      std::random_device rd;
      std::uint8_t bytes[24];
      for (auto& b : bytes) b = static_cast<std::uint8_t>(rd());
      s.k0 = read_le64(bytes);
      s.k1 = read_le64(bytes + 8);
    }
    return s;
  }();
  return secret;
}

// CPython `str.__hash__` for ASCII strings (maps -1 -> -2 like `_Py_HashBytes`).
inline py_hash_t py_hash_str(const std::string& s) {
  if (s.empty()) return 0;
  const HashSecret sec = py_hash_secret();
  const std::uint64_t t = siphash13(
      sec.k0, sec.k1, reinterpret_cast<const std::uint8_t*>(s.data()),
      s.size());
  py_hash_t h = static_cast<py_hash_t>(t);
  if (h == static_cast<py_hash_t>(0xFFFFFFFFFFFFFFFFULL)) {
    h = static_cast<py_hash_t>(0xFFFFFFFFFFFFFFFEULL);
  }
  return h;
}

}  // namespace dqc
