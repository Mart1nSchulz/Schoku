// Tiny header-only test harness for compat shim tests.
// Single-TU friendly. Each TU includes this and defines a TEST_MAIN
// of REQUIRE() calls; the main() at the bottom counts pass/fail and
// prints a summary.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>

namespace tt {

struct Stats { int passed = 0; int failed = 0; };
inline Stats& stats() { static Stats s; return s; }

inline void report_fail(const char *file, int line, const char *expr, const char *extra) {
    std::fprintf(stderr, "FAIL %s:%d  %s  %s\n", file, line, expr, extra ? extra : "");
    stats().failed++;
}

inline int report_summary(const char *suite) {
    std::printf("%s: %d passed, %d failed\n", suite, stats().passed, stats().failed);
    return stats().failed == 0 ? 0 : 1;
}

} // namespace tt

#define REQUIRE(expr)                                                         \
    do {                                                                      \
        if (!(expr)) ::tt::report_fail(__FILE__, __LINE__, #expr, nullptr);   \
        else ::tt::stats().passed++;                                          \
    } while (0)

#define REQUIRE_EQ_U64(a, b)                                                  \
    do {                                                                      \
        unsigned long long _a = (unsigned long long)(a);                      \
        unsigned long long _b = (unsigned long long)(b);                      \
        if (_a != _b) {                                                       \
            char buf[160];                                                    \
            std::snprintf(buf, sizeof(buf), "(%s = 0x%llx) != (%s = 0x%llx)", \
                          #a, _a, #b, _b);                                    \
            ::tt::report_fail(__FILE__, __LINE__, "REQUIRE_EQ_U64", buf);     \
        } else ::tt::stats().passed++;                                        \
    } while (0)
