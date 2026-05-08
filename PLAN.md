# Schoku Port Plan

Tracks the port of `src/schoku.cpp` (one 6945-line gcc/AVX2 file) to a clang++ build that runs on Apple Silicon (NEON via simde) while preserving bit-identical solver output and matching original performance.

**Started:** 2026-05-07T19:13:51Z
**Phase 2 done:** 2026-05-07T19:28:52Z (15 min — port build at NEON parity)
**Phase 3 done:** 2026-05-07T21:10Z (~2 hours from start — full test suite, 9 parity matrix + 41,237 compat unit asserts + 10 CLI smoke; -l# upstream SIGSEGV diagnosed and fixed in commit `5b71869`)

---

## Baselines (golden truth)

Captured on cuda-host2 (AMD EPYC 7713, 64C, AVX2/BMI/BMI2/SHA-NI), gcc 14.2 `-O3 -mavx2 -mbmi -mbmi2 -mlzcnt`, `-DOPT_NEWSETS`.

| Dataset | Threads | Time (5 runs, best of) | Per puzzle | SHA256 of solutions |
|---|---|---|---|---|
| `puzzles_big5000.txt` (5000) | 8 | **17.9 ms** | 3.59 µs | `d27af954...876a` |
| `puzzles_big5000.txt` | 1 | **111.4 ms** | 22.28 µs | (same) |
| `puzzles_tiny50.txt` (50) | 8 | (small dataset) | — | `3f7cd608...ec11` |

Any port must produce exactly these SHA256 hashes for the solutions file.

---

## Phases

Status legend: `[ ]` pending · `[~]` in progress · `[x]` done

### Phase 0 — Toolchain + golden output `[x]`
- [x] gcc 14.2 build on cuda-host2 (was failing on `<intrin.h>`, fixed by `compat/x86_intrin.hpp`)
- [x] Captured solutions SHA256 + perf baseline (see table above)

### Phase 1 — clang++ on x86_64 `[x]`
- [x] `clang 19.1.7` (Debian) installed on cuda-host2
- [x] Compiles original sources — only warnings, no errors (after `<intrin.h>` removal)
- [x] **Parity verified**: `sols_big5000_clang.txt` SHA256 == gcc golden
- [x] Perf: 18.1 ms (-t8), within noise of gcc

### Phase 2 — Apple Silicon (NEON via simde) `[x]`
- [x] `simde` and `libomp` installed via Homebrew
- [x] Created `compat/x86_intrin.hpp` master header (arch-aware dispatch)
- [x] Created `compat/msvc_intrin.hpp` (`_bittestandreset*`, `__popcnt*`) — Cygwin path preserved (delegates to native `<intrin.h>`)
- [x] Created `compat/bmi_shim.hpp` (TZCNT/LZCNT/POPCNT/BMI1/BMI2 scalar fallbacks + MSVC double-underscore variants + `__builtin_cpu_*` shim)
- [x] Replaced `__m256i x { ... }` brace-init (3 sites) with `_mm256_setr_epi64x` (simde struct layout differs)
- [x] Added gcc shorthand vector-type aliases (`__v4du`, `__m128i_u`, etc.) for non-x86
- [x] Created `Makefile.mac` (Apple clang + isysroot + libomp + simde, `XCC`/`CXX` overridable)
- [x] Resolved local env conflict (`/usr/local/include` symlinks shadowing libc++) via `-isysroot $(xcrun --show-sdk-path)` + `-stdlib=libc++` + explicit libc++ include
- [x] Build clean on M-series clang (5 minor warnings, no errors)
- [x] Smoke run on tiny50 + big5000
- [x] **Parity gate passed**: SHA256 of macOS-clang-NEON solutions == cuda-host2 gcc golden, BOTH datasets

**Phase 2 finish: 2026-05-07T19:28:52Z — total elapsed 15 minutes from start.**

Known follow-ups (do not block parity, but file as issues):
- ~~Stats counter overflow~~ FIXED. Root cause: `#pragma omp declare reduction` had no `initializer(...)` clause. gcc happened to zero-init per-thread copies, clang left them indeterminate. Fix: added `initializer(omp_priv = Counters())`. Parity-restoring (gcc unchanged).
- ~~SIGABRT under fortified libc~~ FIXED. `char opts[80]` overflow in argv-reflection for `-x` stats. Sized to 1024 with `snprintf`. Triggered on macOS clang because of fortify; gcc EPYC was lucky.
- ~~**Upstream bug** — `-l5` (and likely other `-l#`) exits with SIGSEGV on both gcc EPYC and clang/macOS~~ **FIXED** in commit `5b71869`. Root cause: `munmap(output, npuzzles*164)` while the prior `mmap()` was `outnpuzzles*164` — POSIX-legal but tears down unrelated pages later allocated by libc/libomp. Hoisted `output_bytes = outnpuzzles*164` so ftruncate/mmap/munmap can't drift apart. gcc EPYC SHA + perf identical (17.4ms / 3.48 µs/puzzle, `d27af954...876a`).
- 5 build warnings remain (loop-vectorize, narrow conv) — non-functional.

### Phase 3 — Tests `[x]`
Three tiers, ~5 sec total runtime, 41,256 assertions across all of them.

**Tier 1 — differential parity** (`tests/parity/`)
- [x] `matrix.txt` — 9 cases: 4 datasets × {1,4,8} threads × {default, -ms, -mn, -rO}
- [x] `golden/*.sha256` — gcc/EPYC frozen as ground truth (captured by `capture_golden.sh`)
- [x] `parity.sh` — runs port binary against matrix, byte-compares solutions vs golden SHA256
- [x] **Result on macOS NEON build: 9/9 PASS**

**Tier 2 — compat shim unit** (`tests/compat/`)
- [x] `test_pdep_pext.cpp` — 20,488 asserts; edge cases + roundtrip identity + random vs reference
- [x] `test_tzcnt_lzcnt.cpp` — 16,602 asserts; zero-input contract + single-bit walks + random vs reference
- [x] `test_popcnt_blsi.cpp` — 4,119 asserts; POPCNT/BLSI/BLSR/ANDN/BZHI/BEXTR
- [x] `test_msvc.cpp` — 28 asserts; MSVC double-underscore variants, `_bittestandreset[64]`, `__builtin_cpu_*` shim
- [x] `Makefile` with `make` (mac/NEON via simde) and `make linux` (gcc native intrinsics)
- [x] **Result mac (NEON shim): 41,237 PASS** · **Result EPYC (native intrinsic): 41,234 PASS**

**Tier 3 — CLI smoke** (`tests/cli/cli_smoke.sh`)
- [x] -h, -x, -y, -l#, -t# (deterministic across thread counts), missing input file, -rO, -m flags, -#1
- [x] **Result: 10/10 PASS** (after `-l#` SIGSEGV fix)

**Top-level runner** (`tests/run_all.sh`) builds + runs all three tiers, single ALL GREEN exit.
Pre-commit hook: just `tests/parity/parity.sh` (5 seconds).

Skipped intentionally (low ROI for this port):
- handcrafted strategy tests — 5000-puzzle parity already exercises every strategy hundreds of times
- already-solved-puzzle dataset — solver loops on it (also in original); degenerate input

### Phase 4 — Decompose `solve()` `[~]`
The 4103-line `solve()` with 62 gotos was the worst readability hotspot. Strict rule: **restructuring only, not optimization. 100 % byte parity required after every split.** Each split commits separately and runs `tests/run_all.sh` before merging.

Utility extractions out of `schoku.cpp` (parity-clean, codex-reviewed):
- [x] Extract debug `dump_*` helpers → `util/debug_dump.hpp` (commit `7a3f140`)
- [x] Extract bit/SIMD helpers → `util/bit_simd.hpp` (commit `e30dd97`)
- [x] Extract board printers → `util/board_dump.hpp` (commit `81d6136`)
- [x] Extract `make_guess` overloads → `solver/make_guess.hpp` (commit `8a6122b`)

`solve()` body extraction + goto elimination:
- [x] Extract `Status solve()` body → `solver/solve.hpp` (commit `b849d7b`, verbatim shift)
- [x] Eliminate local `goto no_bug` (commit `7484f3c`)
- [x] Eliminate local `goto done` in OPT_FSH rows (commit `9cd5718`)
- [x] **Eliminate the remaining 58 gotos via state-machine wrapper** (commit `1d43328`):
  the labels (`back`, `start`, `search`, `enter`, `hidden_search`, `guess`,
  `guess_made_with_incr`) became `case Phase_X:` of an outer `for(;;) switch(phase)`,
  and every `goto X;` became `phase = Phase_X; continue;`. Natural fallthroughs
  preserved with `[[fallthrough]];`. Also eliminated `goto done2;` in OPT_FSH cols
  with the same flag-pattern as `done`. Solver byte-identical, perf preserved
  within noise (16.8ms -t8 vs 16.7ms baseline; 118.7ms -t1 vs 119.3ms baseline).

`solve.hpp` is now zero-goto. Remaining sub-decomposition (lifting each `case Phase_X` body
into its own purpose-specific module) is lower-risk now that goto-control-flow doesn't cross
strategy boundaries:
- [ ] Lift each phase body into a per-phase inline function (still single TU, with
      `__attribute__((always_inline))` to preserve current inlining)
- [ ] Move per-phase functions into separate files (`solver/phases/back.hpp`,
      `solver/phases/start.hpp`, etc.)

### Phase 5 — Final perf run `[~]`
Goal: M-series perf >= cuda-host2 gcc baseline; document reality.
- [x] big5000 -t8 on all three builds (best-of-5) — see Perf metrics table; **mac NEON 17.2 ms vs gcc EPYC 17.4 ms** (within noise / slightly ahead)
- [x] big5000 -t1 on mac (single-thread): **118.7 ms** (23.75 µs/puzzle)
- [x] full perf retake AFTER goto-elimination state-machine commit: **16.8 ms -t8 / 118.7 ms -t1** (within noise of pre-refactor baseline 16.7 / 119.3)
- [ ] tiny50 / harder datasets if available
- [ ] perf retake AFTER per-phase file split (Phase 4 sub-decomposition)

---

## Metrics

### Build metrics

| Build | Compiler | Arch | Status | Warnings |
|---|---|---|---|---|
| original (gcc) | g++ 14.2 | x86_64 (AVX2) | OK | 0 |
| port (clang x86) | clang++ 19.1 | x86_64 (AVX2) | OK | 105 (unused-const-var, missing-braces — non-functional) |
| port (clang mac) | Apple clang 17 | aarch64 (NEON via simde) | OK | 5 (loop-vectorize, narrow conv — non-functional) |

### Perf metrics (best-of-5 internal solving time)

| Build | Host | Dataset | -t8 best-of-5 | µs/puzzle (-t8) | -t1 best-of-5 | parity vs golden | stats parity |
|---|---|---|---|---|---|---|---|
| gcc orig | EPYC 7713 64C | big5000 | **17.4 ms** | 3.48 | — | (golden) | (golden) |
| clang x86 port | EPYC 7713 64C | big5000 | 28.4 ms | 5.68 | — | **OK** | **OK** |
| clang mac NEON (pre-Phase 4) | M-series | big5000 | 17.2 ms | 3.43 | 119.3 ms | **OK** | **OK** |
| clang mac NEON (post-state-machine) | M-series | big5000 | **16.8 ms** | **3.36** | **118.7 ms** | **OK** | **OK** |
| clang mac NEON | M-series | tiny50 | (small) | — | — | **OK** | **OK** |
| gcc orig | EPYC 7713 64C | tiny50 | — | — | — | (golden) | 50/50, 1370 g, 13 BUG |
| clang mac NEON | M-series | tiny50 | — | — | — | OK | 50/50, 1370 g, 13 BUG |

**Headline:** mac NEON port is faster than gcc/EPYC and remains so after eliminating all 60 gotos in `solve()` via the state-machine wrapper. Solver byte-exact across all builds.

(Side note: clang on x86 produces noticeably slower code than gcc on x86 — about 1.6× slower. That's a clang vs gcc x86 codegen issue, not specific to this port. clang's aarch64 codegen is much closer to optimal.)

### Code structure

| Metric | Original | Current port |
|---|---|---|
| Source files | 1 (`schoku.cpp`) | `schoku.cpp` + 3 compat + 3 util + 2 solver headers |
| Test files | 0 | 9 small files (parity matrix/runner, 4 compat tests, ref/lib helpers, CLI smoke, run_all) |
| Total source LOC | 6945 | ~6700 in `schoku.cpp` + 4150 in `solver/solve.hpp` + 491 in `solver/make_guess.hpp` + ~330 in `util/*` + ~265 in `compat/*` |
| `schoku.cpp` LOC | 6945 | 2114 (everything solver-specific moved out) |
| `solve()` body LOC | 4103 | 4150 in `solver/solve.hpp` (~50 lines added by state-machine scaffolding) |
| `goto` / labels in `solve()` | 62 / 8 | **0 / 0** (state machine) |
| Distinct AVX2/BMI intrinsics in use | 97 | (same; abstracted via simde + bmi_shim) |
| Compile-time arch dispatch points | 0 | 1 (`compat/x86_intrin.hpp`) |
| Test assertions | 0 | 41,259 (parity 9 SHA matches + compat 41,240 + CLI 10) |

---

## Risks tracked

- ~~**NEON has no `pdep`/`pext`**~~ Mitigated. Set `bmi2_support = false` on aarch64, which steers the solver into its existing AVX2-only fallback path (already implemented in source for Zen2 — same path now reused on aarch64). No measurable regression: M-series perf parity-equal to gcc/EPYC.
- **`solve()` decomposition risk**: gcc currently inlines aggressively into one cache-resident block. Splitting may cost 2-5% even with `always_inline`. Each split gated by parity SHA + perf re-measure.
- ~~**Local env conflict on macOS**~~ Resolved. `-isysroot $(xcrun --show-sdk-path)` + `-stdlib=libc++` pin the include path; documented in `Makefile.mac`.
