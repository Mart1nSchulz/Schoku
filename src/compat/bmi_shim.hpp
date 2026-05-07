// Scalar fallbacks for x86 BMI1/BMI2/LZCNT/POPCNT intrinsics on non-x86 hosts.
//
// Only included on architectures that lack the real intrinsics (currently
// aarch64 / Apple Silicon). Semantics match the Intel reference exactly,
// including the 0-input behavior of TZCNT/LZCNT (return the operand width).
//
// Hot intrinsics live in inlined functions; PDEP/PEXT use a portable
// loop because Arm has no equivalent. Their use in schoku is bounded
// (<= a handful of call sites per round) so the overhead is acceptable.
#pragma once

#include <cstdint>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#error "compat/bmi_shim.hpp must not be included on x86 targets"
#endif

namespace schoku_compat {

inline unsigned tzcnt32(unsigned x) noexcept { return x ? __builtin_ctz(x) : 32u; }
inline unsigned tzcnt64(unsigned long long x) noexcept { return x ? (unsigned)__builtin_ctzll(x) : 64u; }
inline unsigned lzcnt32(unsigned x) noexcept { return x ? __builtin_clz(x) : 32u; }
inline unsigned lzcnt64(unsigned long long x) noexcept { return x ? (unsigned)__builtin_clzll(x) : 64u; }

template <class T>
inline unsigned long long pdep64_impl(T src, unsigned long long mask) noexcept {
    unsigned long long res = 0;
    for (unsigned long long bb = 1; mask; bb += bb) {
        if (src & bb) res |= mask & -mask;
        mask &= mask - 1;
    }
    return res;
}

template <class T>
inline unsigned long long pext64_impl(T src, unsigned long long mask) noexcept {
    unsigned long long res = 0;
    for (unsigned long long bb = 1; mask; bb += bb) {
        if (src & mask & -mask) res |= bb;
        mask &= mask - 1;
    }
    return res;
}

} // namespace schoku_compat

// TZCNT / LZCNT
static inline unsigned short        _tzcnt_u16(unsigned short x)        { return (unsigned short)schoku_compat::tzcnt32((unsigned)x | 0x10000u); /* width 16 → cap at 16 */ }
static inline unsigned int          _tzcnt_u32(unsigned int x)          { return schoku_compat::tzcnt32(x); }
static inline unsigned long long    _tzcnt_u64(unsigned long long x)    { return schoku_compat::tzcnt64(x); }
static inline unsigned int          _lzcnt_u32(unsigned int x)          { return schoku_compat::lzcnt32(x); }
static inline unsigned long long    _lzcnt_u64(unsigned long long x)    { return schoku_compat::lzcnt64(x); }

// POPCNT
static inline int                   _popcnt16(unsigned short x)         { return __builtin_popcount((unsigned)x); }
static inline int                   _popcnt32(unsigned int x)           { return __builtin_popcount(x); }
static inline long long             _popcnt64(unsigned long long x)     { return __builtin_popcountll(x); }

// BMI1. Negation is done in unsigned arithmetic so INT_MIN/LLONG_MIN inputs
// don't invoke signed-overflow UB (`-(int)0x80000000` would be UB).
static inline unsigned int          _blsi_u32(unsigned int x)           { return x & (0u - x); }
static inline unsigned long long    _blsi_u64(unsigned long long x)     { return x & (0ull - x); }
static inline unsigned int          _blsr_u32(unsigned int x)           { return x & (x - 1); }
static inline unsigned long long    _blsr_u64(unsigned long long x)     { return x & (x - 1); }
static inline unsigned int          _andn_u32(unsigned int a, unsigned int b)               { return ~a & b; }
static inline unsigned long long    _andn_u64(unsigned long long a, unsigned long long b)   { return ~a & b; }
static inline unsigned long long    _bextr_u64(unsigned long long src, unsigned start, unsigned len) {
    if (start >= 64) return 0ULL;          // native BEXTR returns 0 here
    if (len >= 64)   return src >> start;  // mask = all-ones
    return (src >> start) & ((1ULL << len) - 1ULL);
}

// BMI2
static inline unsigned int          _bzhi_u32(unsigned int src, unsigned n) { return n >= 32 ? src : (src & ((1u << n) - 1u)); }
static inline unsigned long long    _bzhi_u64(unsigned long long src, unsigned n) { return n >= 64 ? src : (src & ((1ULL << n) - 1ULL)); }
static inline unsigned int          _pdep_u32(unsigned int src, unsigned int mask)        { return (unsigned int)schoku_compat::pdep64_impl<unsigned int>(src, mask); }
static inline unsigned long long    _pdep_u64(unsigned long long src, unsigned long long mask) { return schoku_compat::pdep64_impl<unsigned long long>(src, mask); }
static inline unsigned int          _pext_u32(unsigned int src, unsigned int mask)        { return (unsigned int)schoku_compat::pext64_impl<unsigned int>(src, mask); }
static inline unsigned long long    _pext_u64(unsigned long long src, unsigned long long mask) { return schoku_compat::pext64_impl<unsigned long long>(src, mask); }

// MSVC double-underscore aliases. MSVC and Cygwin's <intrin.h> expose both
// the BMI single-underscore names and these MSVC-flavoured ones; the solver
// uses both spellings interchangeably.
static inline unsigned short        __tzcnt_u16(unsigned short x)       { return (unsigned short)_tzcnt_u16(x); }
static inline unsigned int          __tzcnt_u32(unsigned int x)         { return _tzcnt_u32(x); }
static inline unsigned long long    __tzcnt_u64(unsigned long long x)   { return _tzcnt_u64(x); }
static inline unsigned short        __lzcnt16(unsigned short x)         { return x ? (unsigned short)(__builtin_clz((unsigned)x) - 16) : (unsigned short)16; }
static inline unsigned int          __lzcnt(unsigned int x)             { return _lzcnt_u32(x); }
static inline unsigned int          __lzcnt32(unsigned int x)           { return _lzcnt_u32(x); }
static inline unsigned long long    __lzcnt64(unsigned long long x)     { return _lzcnt_u64(x); }
static inline unsigned int          __blsi_u32(unsigned int x)          { return _blsi_u32(x); }
static inline unsigned long long    __blsi_u64(unsigned long long x)    { return _blsi_u64(x); }
static inline unsigned int          __blsr_u32(unsigned int x)          { return _blsr_u32(x); }
static inline unsigned long long    __blsr_u64(unsigned long long x)    { return _blsr_u64(x); }

// __builtin_cpu_supports / __builtin_cpu_is are x86-only in gcc/clang.
// On non-x86 the solver uses them to (a) gate the AVX2 startup check and
// (b) decide whether to take the PDEP/PEXT fast path inside the solver.
//
// We claim AVX2 + BMI1 are present (provided by simde + bmi_shim) but say
// BMI2 is missing, which steers the solver to the AVX2-only fallback for
// _pext/_pdep — that is faster on NEON than calling our scalar emulation
// in a hot loop.
#if !defined(__x86_64__) && !defined(__i386__)
namespace schoku_compat {
constexpr int cpu_supports(const char *feature) {
    // Match the few feature names the solver queries.
    return (feature[0]=='a' && feature[1]=='v' && feature[2]=='x' && feature[3]=='2' && feature[4]==0) ? 1
         : (feature[0]=='b' && feature[1]=='m' && feature[2]=='i' && feature[3]==0) ? 1
         : (feature[0]=='b' && feature[1]=='m' && feature[2]=='i' && feature[3]=='2' && feature[4]==0) ? 0
         : 0;
}
} // namespace schoku_compat
#define __builtin_cpu_supports(feature) (schoku_compat::cpu_supports(feature))
#define __builtin_cpu_is(name)          (0)
#endif
