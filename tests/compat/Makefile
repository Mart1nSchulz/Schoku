# Build & run compat shim unit tests.
# Usage:
#   make            # macOS / Apple Silicon (default, NEON via simde)
#   make linux      # Linux x86 with native intrinsics (no simde)
#
# Each test is its own tiny TU built with the same compat header schoku
# uses. Tests print a per-suite summary; `make check` exits non-zero on
# any failure.
TESTS    := test_pdep_pext test_tzcnt_lzcnt test_popcnt_blsi test_msvc

CXX      ?= /usr/bin/clang++
BREW     ?= /opt/homebrew
SIMDE    ?= $(BREW)/opt/simde/include
SDK      ?= $(shell xcrun --show-sdk-path 2>/dev/null)

WFLAGS   := -Wall -Wextra -Wno-unused-parameter
CXXFLAGS := -std=gnu++17 -O2 $(WFLAGS) -I../../src
ifneq ($(SDK),)
CXXFLAGS += -isysroot $(SDK) -stdlib=libc++ -isystem $(SIMDE)
endif

# Linux x86 build — uses native <x86intrin.h>, no simde, no SDK pinning.
linux: CXX := g++
linux: CXXFLAGS := -std=gnu++17 -O2 $(WFLAGS) -I../../src -mavx2 -mbmi -mbmi2 -mlzcnt
linux: $(TESTS)

.PHONY: all check linux clean
all: $(TESTS)

check: $(TESTS)
	@fails=0; for t in $(TESTS); do \
	    ./$$t || fails=$$((fails + 1)); \
	done; \
	[ $$fails -eq 0 ] || { echo "$$fails compat suite(s) failed" >&2; exit 1; }

%: %.cpp testlib.hpp ref_bits.hpp ../../src/compat/x86_intrin.hpp \
        ../../src/compat/msvc_intrin.hpp ../../src/compat/bmi_shim.hpp
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	$(RM) $(TESTS)
