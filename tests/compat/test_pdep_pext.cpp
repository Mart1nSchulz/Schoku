// Verify _pdep_u32/64 and _pext_u32/64 (BMI2 shims on aarch64; native on x86).
//
// On x86 these compile to PDEP/PEXT and the test pins them to the
// reference loop impl; on aarch64 they call into compat/bmi_shim.hpp,
// which itself uses the same loop algorithm — this still serves as a
// sanity check on edge cases (zero masks, single-bit, fully-saturated).
#include "../../src/compat/x86_intrin.hpp"
#include "testlib.hpp"
#include "ref_bits.hpp"

int main() {
    // Edge cases per Intel SDM.
    REQUIRE_EQ_U64(_pdep_u64(0,        0),                    0);
    REQUIRE_EQ_U64(_pdep_u64(~0ULL,    0),                    0);
    REQUIRE_EQ_U64(_pdep_u64(~0ULL,    ~0ULL),                ~0ULL);
    REQUIRE_EQ_U64(_pdep_u64(0xABCDull,0xFFFFull),            0xABCDull);

    REQUIRE_EQ_U64(_pext_u64(0,        0),                    0);
    REQUIRE_EQ_U64(_pext_u64(~0ULL,    0),                    0);
    REQUIRE_EQ_U64(_pext_u64(~0ULL,    ~0ULL),                ~0ULL);
    REQUIRE_EQ_U64(_pext_u64(0xABCDull,0xFFFFull),            0xABCDull);

    // Roundtrip identity: pext(pdep(x, m), m) == x  &  ((1<<popcnt(m)) - 1).
    // pdep(pext(y, m), m) == y & m.
    ref::Rng rng(0x1234567890ABCDEFULL);
    for (int i = 0; i < 4096; ++i) {
        uint64_t src  = rng.next();
        uint64_t mask = rng.next();
        uint64_t pop  = ref::popcount64(mask);
        uint64_t lo   = (pop >= 64) ? ~0ULL : ((1ULL << pop) - 1ULL);
        REQUIRE_EQ_U64(_pext_u64(_pdep_u64(src, mask), mask), src & lo);
        REQUIRE_EQ_U64(_pdep_u64(_pext_u64(src, mask), mask), src & mask);
    }

    // Compare both shim/native call against an independent reference impl
    // on a wide spread of (src, mask) pairs.
    for (int i = 0; i < 4096; ++i) {
        uint64_t src  = rng.next();
        uint64_t mask = rng.next();
        REQUIRE_EQ_U64(_pdep_u64(src, mask), ref::pdep64(src, mask));
        REQUIRE_EQ_U64(_pext_u64(src, mask), ref::pext64(src, mask));
    }

    // 32-bit variants: same identities, narrower domain.
    for (int i = 0; i < 2048; ++i) {
        uint32_t src  = (uint32_t)rng.next();
        uint32_t mask = (uint32_t)rng.next();
        REQUIRE_EQ_U64(_pdep_u32(src, mask),
                       (uint32_t)ref::pdep64(src, (uint64_t)mask));
        REQUIRE_EQ_U64(_pext_u32(src, mask),
                       (uint32_t)ref::pext64(src, (uint64_t)mask));
    }

    return tt::report_summary("test_pdep_pext");
}
