// Toolchain entry point for x86 intrinsics. On x86 targets (gcc/clang) this
// includes <x86intrin.h> directly. On aarch64 (e.g. Apple Silicon) AVX2/SSE
// are provided through simde; the BMI/ABM scalar intrinsics not covered there
// are supplied by compat/bmi_shim.hpp.
//
// The MSVC-style names (_bittestandreset*, __popcnt*) are defined in
// msvc_intrin.hpp and are always available.
#pragma once

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
  #include <x86intrin.h>
#else
  // libc++ 17+ refuses to include <math.h>/<stdlib.h>/etc. transitively from
  // inside a system-include context (which is how simde gets pulled in via
  // -isystem). Force-include the C++ wrappers first so the include guards
  // are already set when simde transitively includes them.
  #include <cmath>
  #include <cstdlib>
  #include <cstring>
  #include <cstdio>
  #include <cstddef>
  #include <cstdint>
  #include <cerrno>

  // Use Intel-flavored names ("__m256i", "_mm256_*") backed by simde.
  #ifndef SIMDE_ENABLE_NATIVE_ALIASES
  #define SIMDE_ENABLE_NATIVE_ALIASES
  #endif
  #include <simde/x86/avx2.h>
  #include <simde/x86/sse4.2.h>
  #include "bmi_shim.hpp"

  // gcc/clang(x86) define these short names in <emmintrin.h>/<immintrin.h>;
  // on aarch64 clang only some are present, so we provide them all uniformly.
  // They are pure type aliases for vector_size attributes and only used for
  // lane-by-lane access in debug printers.
  typedef unsigned char        __v32qu_compat __attribute__((__vector_size__(32)));
  typedef unsigned short       __v16hu_compat __attribute__((__vector_size__(32)));
  typedef unsigned int         __v8su_compat  __attribute__((__vector_size__(32)));
  typedef unsigned long long   __v4du_compat  __attribute__((__vector_size__(32)));
  typedef unsigned char        __v16qu_compat __attribute__((__vector_size__(16)));
  typedef unsigned short       __v8hu_compat  __attribute__((__vector_size__(16)));
  typedef unsigned int         __v4su_compat  __attribute__((__vector_size__(16)));
  typedef unsigned long long   __v2du_compat  __attribute__((__vector_size__(16)));
  #ifndef __v32qu
  #define __v32qu __v32qu_compat
  #endif
  #ifndef __v16hu
  #define __v16hu __v16hu_compat
  #endif
  #ifndef __v8su
  #define __v8su __v8su_compat
  #endif
  #ifndef __v4du
  #define __v4du __v4du_compat
  #endif
  #ifndef __v16qu
  #define __v16qu __v16qu_compat
  #endif
  #ifndef __v8hu
  #define __v8hu __v8hu_compat
  #endif
  #ifndef __v4su
  #define __v4su __v4su_compat
  #endif
  #ifndef __v2du
  #define __v2du __v2du_compat
  #endif

  // gcc/clang(x86) provide these "_u"-suffixed aliases for unaligned access
  // (__m128i / __m256i with aligned(1)). simde doesn't expose them, so define
  // here. Used by the puzzle parser for misaligned 16-byte loads.
  typedef __m128i __m128i_u __attribute__((__aligned__(1)));
  typedef __m256i __m256i_u __attribute__((__aligned__(1)));
#endif

#include "msvc_intrin.hpp"
