// MSVC-style intrinsic shims used by schoku.cpp, mapped onto gcc/clang builtins.
//
// The original sources targeted Cygwin/MSVC and called these names directly.
// We use them on Linux/macOS where neither gcc nor clang ships them. On
// Cygwin and on real MSVC the names already exist (via <intrin.h>); we
// include the platform header there and skip our shims so we never shadow
// the natives.
//
// This header has no SIMD dependencies and is safe to include on any
// architecture; it intentionally does NOT pull in <x86intrin.h>.
#pragma once

#include <cstdint>

#if defined(_MSC_VER) && !defined(__clang__)
// Native MSVC: the real intrinsics already exist via <intrin.h>.
#include <intrin.h>
#define SCHOKU_HAVE_NATIVE_MSVC_INTRIN 1
#elif defined(__CYGWIN__)
// Cygwin's gcc ships its own <intrin.h> that defines the MSVC-flavoured
// names (_bittestandreset*, __popcnt*, etc.). Use it directly — staying on
// the Cygwin-native path preserves the original schoku build there.
#include <intrin.h>
#define SCHOKU_HAVE_NATIVE_MSVC_INTRIN 1
#endif

#ifndef SCHOKU_HAVE_NATIVE_MSVC_INTRIN

// Minimal scalar shims for non-Cygwin/non-MSVC builds.
// Each is guarded so we never redefine a name a future toolchain provides.

// Use unsigned shifts to avoid signed-overflow UB at bit 31 / bit 63.
// NOTE: native MSVC's _bittestandreset is a locked (atomic) bit op. This
// scalar shim is non-atomic and matches the *non*-atomic bittest variant.
// The schoku call sites use it on per-thread state (no concurrent access),
// so the contract difference is benign here, but a future caller that needs
// atomicity must NOT use this shim.
#ifndef _bittestandreset
static inline unsigned char _bittestandreset(int *p, int bit) {
    const unsigned int mask = 1u << bit;
    const unsigned int old  = ((unsigned int)(*p)) & mask;
    *p = (int)(((unsigned int)(*p)) & ~mask);
    return old != 0 ? 1 : 0;
}
#endif

#ifndef _bittestandreset64
static inline unsigned char _bittestandreset64(long long *p, int bit) {
    const unsigned long long mask = 1ULL << bit;
    const unsigned long long old  = ((unsigned long long)(*p)) & mask;
    *p = (long long)(((unsigned long long)(*p)) & ~mask);
    return old != 0 ? 1 : 0;
}
#endif

#ifndef __popcnt16
#define __popcnt16(x) ((unsigned short)__builtin_popcount((unsigned int)(unsigned short)(x)))
#endif
#ifndef __popcnt
#define __popcnt(x)   ((unsigned int)__builtin_popcount((unsigned int)(x)))
#endif
#ifndef __popcnt64
#define __popcnt64(x) ((unsigned long long)__builtin_popcountll((unsigned long long)(x)))
#endif

#endif // !SCHOKU_HAVE_NATIVE_MSVC_INTRIN
