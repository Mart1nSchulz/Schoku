// Verify POPCNT and BMI1 helpers (BLSI/BLSR/ANDN/BZHI/BEXTR) and BMI2 BZHI.
// All are simple bit twiddles; tests pin them to single-bit walks plus
// random vs an independent reference, with explicit zero/all-ones cases.
#include "../../src/compat/x86_intrin.hpp"
#include "testlib.hpp"
#include "ref_bits.hpp"

int main() {
    // _popcnt32/64 are BMI names available on Linux gcc and macOS shim alike.
    // (16-bit popcount is exercised via __popcnt16 in test_msvc.cpp.)
    REQUIRE_EQ_U64(_popcnt32(0),                 0);
    REQUIRE_EQ_U64(_popcnt32(~0u),               32);
    REQUIRE_EQ_U64(_popcnt64(0ULL),              0);
    REQUIRE_EQ_U64(_popcnt64(~0ULL),             64);

    // _blsi: x & -x (lowest set bit)
    REQUIRE_EQ_U64(_blsi_u32(0),     0);
    REQUIRE_EQ_U64(_blsi_u64(0),     0);
    REQUIRE_EQ_U64(_blsi_u32(0b1010u),       0b0010u);
    REQUIRE_EQ_U64(_blsi_u64(0xF0F0F0F0F0F0F0F0ULL), 0x0000000000000010ULL);

    // _blsr: x & (x-1) (clear lowest set)
    REQUIRE_EQ_U64(_blsr_u32(0),     0);
    REQUIRE_EQ_U64(_blsr_u64(0),     0);
    REQUIRE_EQ_U64(_blsr_u32(0b1010u),       0b1000u);
    REQUIRE_EQ_U64(_blsr_u64(0x80000001ULL), 0x80000000ULL);

    // _andn: ~a & b
    REQUIRE_EQ_U64(_andn_u32(0xFu, 0xFFu),   0xF0u);
    REQUIRE_EQ_U64(_andn_u64(0xFULL, 0xFFFULL), 0xFF0ULL);

    // _bzhi: keep low n bits
    REQUIRE_EQ_U64(_bzhi_u32(0xFFFFFFFFu, 0),  0u);
    REQUIRE_EQ_U64(_bzhi_u32(0xFFFFFFFFu, 8),  0xFFu);
    REQUIRE_EQ_U64(_bzhi_u64(~0ULL, 40),       (1ULL << 40) - 1);
    REQUIRE_EQ_U64(_bzhi_u32(0xDEADBEEF, 32),  0xDEADBEEF);
    REQUIRE_EQ_U64(_bzhi_u64(~0ULL, 64),       ~0ULL);

    // _bextr_u64(src, start, len)
    REQUIRE_EQ_U64(_bextr_u64(0xABCDEF12u, 0,   8),  0x12);
    REQUIRE_EQ_U64(_bextr_u64(0xABCDEF12u, 8,   8),  0xEF);
    REQUIRE_EQ_U64(_bextr_u64(0xABCDEF12u, 16, 16),  0xABCD);
    REQUIRE_EQ_U64(_bextr_u64(~0ULL,       0,  64),  ~0ULL);

    // Random vs reference popcount.
    ref::Rng rng(0x9E3779B97F4A7C15ULL);
    for (int i = 0; i < 2048; ++i) {
        uint64_t v = rng.next();
        REQUIRE_EQ_U64((uint64_t)_popcnt32((uint32_t)v), ref::popcount32((uint32_t)v));
        REQUIRE_EQ_U64((uint64_t)_popcnt64(v),           ref::popcount64(v));
    }

    return tt::report_summary("test_popcnt_blsi");
}
