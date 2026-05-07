// Plain reference implementations of the BMI/LZCNT/POPCNT primitives,
// used as ground truth for the shim tests. Independent loop-based
// implementations chosen for obvious correctness over speed.
#pragma once

#include <cstdint>

namespace ref {

inline unsigned popcount32(uint32_t x) {
    unsigned n = 0;
    while (x) { n += x & 1u; x >>= 1; }
    return n;
}
inline unsigned popcount64(uint64_t x) {
    unsigned n = 0;
    while (x) { n += (unsigned)(x & 1u); x >>= 1; }
    return n;
}

inline unsigned tzcnt32(uint32_t x) {
    if (x == 0) return 32;
    unsigned n = 0;
    while ((x & 1u) == 0) { x >>= 1; ++n; }
    return n;
}
inline unsigned tzcnt64(uint64_t x) {
    if (x == 0) return 64;
    unsigned n = 0;
    while ((x & 1u) == 0) { x >>= 1; ++n; }
    return n;
}

inline unsigned lzcnt32(uint32_t x) {
    if (x == 0) return 32;
    unsigned n = 0;
    while ((x & 0x80000000u) == 0) { x <<= 1; ++n; }
    return n;
}
inline unsigned lzcnt64(uint64_t x) {
    if (x == 0) return 64;
    unsigned n = 0;
    while ((x & (1ULL << 63)) == 0) { x <<= 1; ++n; }
    return n;
}

inline uint64_t pdep64(uint64_t src, uint64_t mask) {
    uint64_t res = 0;
    for (uint64_t bb = 1; mask != 0; bb += bb) {
        if (src & bb) res |= mask & (~mask + 1ULL); // lowest set bit of mask
        mask &= mask - 1;
    }
    return res;
}
inline uint64_t pext64(uint64_t src, uint64_t mask) {
    uint64_t res = 0;
    for (uint64_t bb = 1; mask != 0; bb += bb) {
        if (src & mask & (~mask + 1ULL)) res |= bb;
        mask &= mask - 1;
    }
    return res;
}

// Tiny xorshift PRNG so tests are deterministic without depending on stdlib.
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed ? seed : 0xDEADBEEFCAFEBABEULL) {}
    uint64_t next() {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        return s;
    }
};

} // namespace ref
