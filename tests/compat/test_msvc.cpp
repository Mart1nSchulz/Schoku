// Verify MSVC-style helpers: _bittestandreset[64] and __popcnt*/__lzcnt*.
// These are exposed by Cygwin's <intrin.h> natively and shimmed everywhere
// else through compat/{msvc_intrin,bmi_shim}.hpp.
#include "../../src/compat/x86_intrin.hpp"
#include "testlib.hpp"
#include "ref_bits.hpp"

int main() {
    // _bittestandreset: returns old bit value, then clears the bit.
    {
        int v = 0b1010;
        REQUIRE_EQ_U64(_bittestandreset(&v, 1), 1);
        REQUIRE_EQ_U64(v, 0b1000);
        REQUIRE_EQ_U64(_bittestandreset(&v, 1), 0);  // already clear
        REQUIRE_EQ_U64(v, 0b1000);
        REQUIRE_EQ_U64(_bittestandreset(&v, 3), 1);
        REQUIRE_EQ_U64(v, 0);
        // Boundary: bit 31 must not invoke signed-overflow UB.
        int hi = (int)0x80000000u;
        REQUIRE_EQ_U64(_bittestandreset(&hi, 31), 1);
        REQUIRE_EQ_U64((unsigned)hi, 0u);
        REQUIRE_EQ_U64(_bittestandreset(&hi, 0), 0);
    }
    {
        long long v = (1LL << 63) | 1LL;
        REQUIRE_EQ_U64(_bittestandreset64(&v, 0), 1);
        REQUIRE_EQ_U64((unsigned long long)v, (1ULL << 63));
        REQUIRE_EQ_U64(_bittestandreset64(&v, 63), 1);
        REQUIRE_EQ_U64(v, 0);
    }

    // MSVC popcount
    REQUIRE_EQ_U64(__popcnt16((unsigned short)0xFFFFu), 16);
    REQUIRE_EQ_U64(__popcnt16((unsigned short)0),       0);
    REQUIRE_EQ_U64(__popcnt(~0u),                       32);
    REQUIRE_EQ_U64(__popcnt64(~0ULL),                   64);

    // MSVC lzcnt (only available via shim; on Cygwin via <intrin.h>)
    REQUIRE_EQ_U64(__lzcnt16((unsigned short)1u), 15);
    REQUIRE_EQ_U64(__lzcnt16((unsigned short)0), 16);
    // gcc/x86 only ships __lzcnt32/64; __lzcnt (no suffix) is a Cygwin/MSVC
    // alias. The shim provides both, but the test pins to the suffixed names
    // so it builds on stock gcc too.
    REQUIRE_EQ_U64(__lzcnt32(1u),                31);
    REQUIRE_EQ_U64(__lzcnt32(0u),                32);
    REQUIRE_EQ_U64(__lzcnt64(1ULL),              63);
    REQUIRE_EQ_U64(__lzcnt64(0ULL),              64);

    // MSVC tzcnt double-underscore variants
    REQUIRE_EQ_U64(__tzcnt_u16((unsigned short)0), 16);
    REQUIRE_EQ_U64(__tzcnt_u32(0u),                32);
    REQUIRE_EQ_U64(__tzcnt_u64(0ULL),              64);
    REQUIRE_EQ_U64(__tzcnt_u32(1u << 17),          17);
    REQUIRE_EQ_U64(__tzcnt_u64(1ULL << 50),        50);

    // __builtin_cpu_supports shim — ALWAYS true for avx2/bmi (we provide
    // them via simde/shim) on aarch64; bmi2 is intentionally false there
    // so the solver picks the AVX2-only fallback over scalar pdep/pext.
#if !defined(__x86_64__) && !defined(__i386__)
    REQUIRE(__builtin_cpu_supports("avx2") == 1);
    REQUIRE(__builtin_cpu_supports("bmi")  == 1);
    REQUIRE(__builtin_cpu_supports("bmi2") == 0);
#endif

    return tt::report_summary("test_msvc");
}
