// Verify _tzcnt_u16/32/64 and _lzcnt_u32/64 against a reference loop impl.
// Pays special attention to the zero-input case (Intel SDM: TZCNT(0) and
// LZCNT(0) return the operand width, not undefined like x86 BSF/BSR).
#include "../../src/compat/x86_intrin.hpp"
#include "testlib.hpp"
#include "ref_bits.hpp"

int main() {
    // Zero-input contract.
    REQUIRE_EQ_U64(_tzcnt_u16(0), 16);
    REQUIRE_EQ_U64(_tzcnt_u32(0), 32);
    REQUIRE_EQ_U64(_tzcnt_u64(0), 64);
    REQUIRE_EQ_U64(_lzcnt_u32(0), 32);
    REQUIRE_EQ_U64(_lzcnt_u64(0), 64);

    // Single-bit walks.
    for (unsigned n = 0; n < 16; ++n) {
        REQUIRE_EQ_U64(_tzcnt_u16((unsigned short)(1u << n)), n);
    }
    for (unsigned n = 0; n < 32; ++n) {
        REQUIRE_EQ_U64(_tzcnt_u32(1u << n), n);
        REQUIRE_EQ_U64(_lzcnt_u32(1u << n), 31u - n);
    }
    for (unsigned n = 0; n < 64; ++n) {
        REQUIRE_EQ_U64(_tzcnt_u64(1ULL << n), n);
        REQUIRE_EQ_U64(_lzcnt_u64(1ULL << n), 63u - n);
    }

    // All-ones.
    REQUIRE_EQ_U64(_tzcnt_u16((unsigned short)0xFFFFu), 0);
    REQUIRE_EQ_U64(_tzcnt_u32(~0u), 0);
    REQUIRE_EQ_U64(_tzcnt_u64(~0ULL), 0);
    REQUIRE_EQ_U64(_lzcnt_u32(~0u), 0);
    REQUIRE_EQ_U64(_lzcnt_u64(~0ULL), 0);

    // Random vs reference impl.
    ref::Rng rng(0xCAFEBABEDEADBEEFULL);
    for (int i = 0; i < 4096; ++i) {
        uint64_t v = rng.next();
        REQUIRE_EQ_U64(_tzcnt_u32((uint32_t)v),  ref::tzcnt32((uint32_t)v));
        REQUIRE_EQ_U64(_tzcnt_u64(v),            ref::tzcnt64(v));
        REQUIRE_EQ_U64(_lzcnt_u32((uint32_t)v),  ref::lzcnt32((uint32_t)v));
        REQUIRE_EQ_U64(_lzcnt_u64(v),            ref::lzcnt64(v));
    }

    return tt::report_summary("test_tzcnt_lzcnt");
}
