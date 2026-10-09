/// This code uses AVX2 instructions...
/*
 * Schoku
 *
 * A high speed sudoku solver by M. Schulz
 *
 * Copyright 2024-2026 Martin Schulz
 *
 * This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 * You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
 *
 * Based on the sudoku solver by Mirage ( https://codegolf.stackexchange.com/users/106606/mirage )
 * at https://codegolf.stackexchange.com/questions/190727/the-fastest-sudoku-solver
 * on Sep 22, 2021
 *
 * Version 1.0.2 speed
 *
 * Compile using g++, Linux or Cygwin
 *
 * Performance changes:
 * - digit-based templates reduces guesses significantly.
 *   for very hard puzzles there is a speed gain.
 *   This option is further tunable on the commandline.
 * - the hidden singles checks columns exclusively (!) and relies on triad processing
 *   to turn those in column hidden singles.
 *
 * Functional changes:
 * - added OPT_DGT single digit processing (built-in by default, option -md)
 * - added an option -a for 'self-tuning'.  Based on initial sampling with the single
 * - digit option, this option is then reset if its benefits are very small.
 * - added compilation option to create a library version only, as well as an adaptor
 *   for benchmaring with the tdoku framework.
 * - the library version has a tuning mechanism in anology to -a option described above.  
 *
 * Performance measurement and statistics:
 * - added two items for the single digit option (checks and found incompatibilities)
 *
 * data: 17-clue sudoku (49151 puzzles)
 * CPU:  Ryzen 7 4700U
 *
 * schoku version: 1.0.2 speed
 * command options: -x puzzles.txt
 * compile options:
 *      49151  puzzles entered
 *      49151  2673793/s  puzzles solved
 *     18.4ms   374ns/puzzle  solving time
 *      38596   78.53%  puzzles solved without guessing
 *      21584    0.43/puzzle  guesses
 *      13497    0.27/puzzle  back tracks
 *     167845    3.41/puzzle  digits entered and retracted
 *      22431    0.46/puzzle  'rounds'
 *     371352    7.56/puzzle  triads resolved
 *     962785   19.59/puzzle  triad updates
 *        709  bi-value universal graves avoided (BUG+1)
 *        138  bi-value universal graves detected
 *        134  board states without bivalues
 *
 * schoku version: 1.0.2 speed
 * command options: -x -md puzzles.txt
 * compile options: OPT_DGT
 *      49151    17.0/puzzle  puzzles entered and presets
 *      49151  2432360/s  puzzles solved
 *     20.2ms   411ns/puzzle  solving time
 *      41184   83.79%  puzzles solved without guessing
 *      16845    0.34/puzzle  guesses
 *      10748    0.22/puzzle  back tracks
 *     135587    2.76/puzzle  digits entered and retracted
 *      24357    0.50/puzzle  'rounds'
 *     360849    7.34/puzzle  triads resolved
 *     949492   19.32/puzzle  triad updates
 *      31447    0.64/puzzle  single digit placement compatibility tests
 *       6716    0.14/puzzle  single digit placement incompatibilities found
 *        727  bi-value universal graves avoided (BUG+1)
 *         43  bi-value universal graves detected
 *         80  board states without bivalues
 *
 */
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cctype>
#include <cstdarg>
#include <cassert>
#include <unistd.h>
#include <fcntl.h>
#include <omp.h>
#include <stdbool.h>
#include <x86intrin.h>
#include "schoku_intrin.h"
#include <sys/stat.h>
#include <sys/mman.h>

namespace Schoku {

const char *version_string = "1.0.2 speed";

const char *compilation_options =
#ifdef OPT_DGT
// Global checking of digit candidates.  Covers the same space as OPT_FSH before
// Uses an innovative algorithm, which has no corresponding human solving strategy.
//
"OPT_DGT "
#endif
""
;

// using type v16us for the built-in vector of unsigned short[16]
using v16us  = __v16hu;
using v8us   = __v8hu;
using v16usb = __v16hu;	// for boolean representation
using v8usb  = __v8hu;	// for boolean representation

// a complete configuration of candidates for a single digit.
// the array u8 contains 3 horizontal bands' applicable placement patterns
// a bit pattern of six bits in canonical order of the bands.
// the second group of 4 unsigned chars contains 3 vertical bands' applicable placement patterns.
union BandPlacements {
    unsigned long long u64;
    unsigned char u8[8];
};

#ifdef OPT_DGT

// Band Reduction lookup tables (for a single digit)  See OPT_DGT
// The placement bitmaps are precombined for two bands (horizontal and vertical),
// so this takes just two lookups.
// The intersection of the 2x2 bands for 2x2 boxes can be preevaluated for validity easily.
// The last bands' lookup tables are prepared for the 3rd band horizontally and vertically.
// Once combined, we have a testable placement, which is valid if it is a subset of the candidates.

typedef struct {
    unsigned short eliminations;  // possible eliminations of locked candidates
                                  // however, the algorithm already in place is still
                                  // a bit faster than using this information. 
    unsigned short config;        // the covering configs as bit set.
} BandReduction;

alignas(64)
BandReduction bandredux[512] {
    // Beware: numbers are given in octal notation!
    // A value of 0177777 aka -1 indicates an invalid input value.
    // A config value of 0 also indicates an invalid input value as
    // all candidates lock each other out and no possible
    // placement pattern exists.
    /* 00 combo = 0 */ { 0177777, 0177777},
    /* 01 combo = 0 */ { 0177777, 0177777},
    /* 02 combo = 0 */ { 0177777, 0177777},
    /* 03 combo = 0 */ { 0177777, 0177777},
    /* 04 combo = 0 */ { 0177777, 0177777},
    /* 05 combo = 0 */ { 0177777, 0177777},
    /* 06 combo = 0 */ { 0177777, 0177777},
    /* 07 combo = 0 */ { 0177777, 0177777},
    /* 010 combo = 0 */ { 0177777, 0177777},
    /* 011 combo = 0 */ { 0177777, 0177777},
    /* 012 combo = 0 */ { 0177777, 0177777},
    /* 013 combo = 0 */ { 0177777, 0177777},
    /* 014 combo = 0 */ { 0177777, 0177777},
    /* 015 combo = 0 */ { 0177777, 0177777},
    /* 016 combo = 0 */ { 0177777, 0177777},
    /* 017 combo = 0 */ { 0177777, 0177777},
    /* 020 combo = 0 */ { 0177777, 0177777},
    /* 021 combo = 0 */ { 0177777, 0177777},
    /* 022 combo = 0 */ { 0177777, 0177777},
    /* 023 combo = 0 */ { 0177777, 0177777},
    /* 024 combo = 0 */ { 0177777, 0177777},
    /* 025 combo = 0 */ { 0177777, 0177777},
    /* 026 combo = 0 */ { 0177777, 0177777},
    /* 027 combo = 0 */ { 0177777, 0177777},
    /* 030 combo = 0 */ { 0177777, 0177777},
    /* 031 combo = 0 */ { 0177777, 0177777},
    /* 032 combo = 0 */ { 0177777, 0177777},
    /* 033 combo = 0 */ { 0177777, 0177777},
    /* 034 combo = 0 */ { 0177777, 0177777},
    /* 035 combo = 0 */ { 0177777, 0177777},
    /* 036 combo = 0 */ { 0177777, 0177777},
    /* 037 combo = 0 */ { 0177777, 0177777},
    /* 040 combo = 0 */ { 0177777, 0177777},
    /* 041 combo = 0 */ { 0177777, 0177777},
    /* 042 combo = 0 */ { 0177777, 0177777},
    /* 043 combo = 0 */ { 0177777, 0177777},
    /* 044 combo = 0 */ { 0177777, 0177777},
    /* 045 combo = 0 */ { 0177777, 0177777},
    /* 046 combo = 0 */ { 0177777, 0177777},
    /* 047 combo = 0 */ { 0177777, 0177777},
    /* 050 combo = 0 */ { 0177777, 0177777},
    /* 051 combo = 0 */ { 0177777, 0177777},
    /* 052 combo = 0 */ { 0177777, 0177777},
    /* 053 combo = 0 */ { 0177777, 0177777},
    /* 054 combo = 0 */ { 0177777, 0177777},
    /* 055 combo = 0 */ { 0177777, 0177777},
    /* 056 combo = 0 */ { 0177777, 0177777},
    /* 057 combo = 0 */ { 0177777, 0177777},
    /* 060 combo = 0 */ { 0177777, 0177777},
    /* 061 combo = 0 */ { 0177777, 0177777},
    /* 062 combo = 0 */ { 0177777, 0177777},
    /* 063 combo = 0 */ { 0177777, 0177777},
    /* 064 combo = 0 */ { 0177777, 0177777},
    /* 065 combo = 0 */ { 0177777, 0177777},
    /* 066 combo = 0 */ { 0177777, 0177777},
    /* 067 combo = 0 */ { 0177777, 0177777},
    /* 070 combo = 0 */ { 0177777, 0177777},
    /* 071 combo = 0 */ { 0177777, 0177777},
    /* 072 combo = 0 */ { 0177777, 0177777},
    /* 073 combo = 0 */ { 0177777, 0177777},
    /* 074 combo = 0 */ { 0177777, 0177777},
    /* 075 combo = 0 */ { 0177777, 0177777},
    /* 076 combo = 0 */ { 0177777, 0177777},
    /* 077 combo = 0 */ { 0177777, 0177777},
    /* 0100 combo = 0 */ { 0177777, 0177777},
    /* 0101 combo = 0 */ { 0177777, 0177777},
    /* 0102 combo = 0 */ { 0177777, 0177777},
    /* 0103 combo = 0 */ { 0177777, 0177777},
    /* 0104 combo = 0 */ { 0177777, 0177777},
    /* 0105 combo = 0 */ { 0177777, 0177777},
    /* 0106 combo = 0 */ { 0177777, 0177777},
    /* 0107 combo = 0 */ { 0177777, 0177777},
    /* 0110 combo = 0 */ { 0177777, 0177777},
    /* 0111 combo = 0 */ { 0177777, 0177777},
    /* 0112 combo = 0 */ { 0177777, 0177777},
    /* 0113 combo = 0 */ { 0177777, 0177777},
    /* 0114 combo = 0 */ { 0177777, 0177777},
    /* 0115 combo = 0 */ { 0177777, 0177777},
    /* 0116 combo = 0 */ { 0116, 00},
    /* 0117 combo = 0 */ { 0117, 00},
    /* 0120 combo = 0 */ { 0177777, 0177777},
    /* 0121 combo = 0 */ { 0177777, 0177777},
    /* 0122 combo = 0 */ { 0177777, 0177777},
    /* 0123 combo = 0 */ { 0177777, 0177777},
    /* 0124 combo = 124 */ { 00, 040},
    /* 0125 combo = 124 */ { 01, 040},
    /* 0126 combo = 124 */ { 02, 040},
    /* 0127 combo = 124 */ { 03, 040},
    /* 0130 combo = 0 */ { 0177777, 0177777},
    /* 0131 combo = 0 */ { 0177777, 0177777},
    /* 0132 combo = 0 */ { 0177777, 0177777},
    /* 0133 combo = 0 */ { 0177777, 0177777},
    /* 0134 combo = 124 */ { 010, 040},
    /* 0135 combo = 124 */ { 011, 040},
    /* 0136 combo = 124 */ { 012, 040},
    /* 0137 combo = 124 */ { 013, 040},
    /* 0140 combo = 0 */ { 0177777, 0177777},
    /* 0141 combo = 0 */ { 0177777, 0177777},
    /* 0142 combo = 142 */ { 00, 010},
    /* 0143 combo = 142 */ { 01, 010},
    /* 0144 combo = 0 */ { 0177777, 0177777},
    /* 0145 combo = 0 */ { 0177777, 0177777},
    /* 0146 combo = 142 */ { 04, 010},
    /* 0147 combo = 142 */ { 05, 010},
    /* 0150 combo = 0 */ { 0177777, 0177777},
    /* 0151 combo = 0 */ { 0177777, 0177777},
    /* 0152 combo = 142 */ { 010, 010},
    /* 0153 combo = 142 */ { 011, 010},
    /* 0154 combo = 0 */ { 0177777, 0177777},
    /* 0155 combo = 0 */ { 0177777, 0177777},
    /* 0156 combo = 142 */ { 014, 010},
    /* 0157 combo = 142 */ { 015, 010},
    /* 0160 combo = 0 */ { 0177777, 0177777},
    /* 0161 combo = 0 */ { 0161, 00},
    /* 0162 combo = 142 */ { 020, 010},
    /* 0163 combo = 142 */ { 021, 010},
    /* 0164 combo = 124 */ { 040, 040},
    /* 0165 combo = 124 */ { 041, 040},
    /* 0166 combo = 166 */ { 00, 050},
    /* 0167 combo = 166 */ { 01, 050},
    /* 0170 combo = 0 */ { 0177777, 0177777},
    /* 0171 combo = 0 */ { 0171, 00},
    /* 0172 combo = 142 */ { 030, 010},
    /* 0173 combo = 142 */ { 031, 010},
    /* 0174 combo = 124 */ { 050, 040},
    /* 0175 combo = 124 */ { 051, 040},
    /* 0176 combo = 166 */ { 010, 050},
    /* 0177 combo = 166 */ { 011, 050},
    /* 0200 combo = 0 */ { 0177777, 0177777},
    /* 0201 combo = 0 */ { 0177777, 0177777},
    /* 0202 combo = 0 */ { 0177777, 0177777},
    /* 0203 combo = 0 */ { 0177777, 0177777},
    /* 0204 combo = 0 */ { 0177777, 0177777},
    /* 0205 combo = 0 */ { 0177777, 0177777},
    /* 0206 combo = 0 */ { 0177777, 0177777},
    /* 0207 combo = 0 */ { 0177777, 0177777},
    /* 0210 combo = 0 */ { 0177777, 0177777},
    /* 0211 combo = 0 */ { 0177777, 0177777},
    /* 0212 combo = 0 */ { 0177777, 0177777},
    /* 0213 combo = 0 */ { 0177777, 0177777},
    /* 0214 combo = 214 */ { 00, 020},
    /* 0215 combo = 214 */ { 01, 020},
    /* 0216 combo = 214 */ { 02, 020},
    /* 0217 combo = 214 */ { 03, 020},
    /* 0220 combo = 0 */ { 0177777, 0177777},
    /* 0221 combo = 0 */ { 0177777, 0177777},
    /* 0222 combo = 0 */ { 0177777, 0177777},
    /* 0223 combo = 0 */ { 0177777, 0177777},
    /* 0224 combo = 0 */ { 0177777, 0177777},
    /* 0225 combo = 0 */ { 0225, 00},
    /* 0226 combo = 0 */ { 0177777, 0177777},
    /* 0227 combo = 0 */ { 0227, 00},
    /* 0230 combo = 0 */ { 0177777, 0177777},
    /* 0231 combo = 0 */ { 0177777, 0177777},
    /* 0232 combo = 0 */ { 0177777, 0177777},
    /* 0233 combo = 0 */ { 0177777, 0177777},
    /* 0234 combo = 214 */ { 020, 020},
    /* 0235 combo = 214 */ { 021, 020},
    /* 0236 combo = 214 */ { 022, 020},
    /* 0237 combo = 214 */ { 023, 020},
    /* 0240 combo = 0 */ { 0177777, 0177777},
    /* 0241 combo = 241 */ { 00, 02},
    /* 0242 combo = 0 */ { 0177777, 0177777},
    /* 0243 combo = 241 */ { 02, 02},
    /* 0244 combo = 0 */ { 0177777, 0177777},
    /* 0245 combo = 241 */ { 04, 02},
    /* 0246 combo = 0 */ { 0177777, 0177777},
    /* 0247 combo = 241 */ { 06, 02},
    /* 0250 combo = 0 */ { 0177777, 0177777},
    /* 0251 combo = 241 */ { 010, 02},
    /* 0252 combo = 0 */ { 0252, 00},
    /* 0253 combo = 241 */ { 012, 02},
    /* 0254 combo = 214 */ { 040, 020},
    /* 0255 combo = 255 */ { 00, 022},
    /* 0256 combo = 214 */ { 042, 020},
    /* 0257 combo = 255 */ { 02, 022},
    /* 0260 combo = 0 */ { 0177777, 0177777},
    /* 0261 combo = 241 */ { 020, 02},
    /* 0262 combo = 0 */ { 0177777, 0177777},
    /* 0263 combo = 241 */ { 022, 02},
    /* 0264 combo = 0 */ { 0177777, 0177777},
    /* 0265 combo = 241 */ { 024, 02},
    /* 0266 combo = 0 */ { 0177777, 0177777},
    /* 0267 combo = 241 */ { 026, 02},
    /* 0270 combo = 0 */ { 0177777, 0177777},
    /* 0271 combo = 241 */ { 030, 02},
    /* 0272 combo = 0 */ { 0272, 00},
    /* 0273 combo = 241 */ { 032, 02},
    /* 0274 combo = 214 */ { 060, 020},
    /* 0275 combo = 255 */ { 020, 022},
    /* 0276 combo = 214 */ { 062, 020},
    /* 0277 combo = 255 */ { 022, 022},
    /* 0300 combo = 0 */ { 0177777, 0177777},
    /* 0301 combo = 0 */ { 0177777, 0177777},
    /* 0302 combo = 0 */ { 0177777, 0177777},
    /* 0303 combo = 0 */ { 0177777, 0177777},
    /* 0304 combo = 0 */ { 0177777, 0177777},
    /* 0305 combo = 0 */ { 0177777, 0177777},
    /* 0306 combo = 0 */ { 0177777, 0177777},
    /* 0307 combo = 0 */ { 0177777, 0177777},
    /* 0310 combo = 0 */ { 0177777, 0177777},
    /* 0311 combo = 0 */ { 0177777, 0177777},
    /* 0312 combo = 0 */ { 0177777, 0177777},
    /* 0313 combo = 0 */ { 0177777, 0177777},
    /* 0314 combo = 214 */ { 0100, 020},
    /* 0315 combo = 214 */ { 0101, 020},
    /* 0316 combo = 214 */ { 0102, 020},
    /* 0317 combo = 214 */ { 0103, 020},
    /* 0320 combo = 0 */ { 0177777, 0177777},
    /* 0321 combo = 0 */ { 0177777, 0177777},
    /* 0322 combo = 0 */ { 0177777, 0177777},
    /* 0323 combo = 0 */ { 0177777, 0177777},
    /* 0324 combo = 124 */ { 0200, 040},
    /* 0325 combo = 124 */ { 0201, 040},
    /* 0326 combo = 124 */ { 0202, 040},
    /* 0327 combo = 124 */ { 0203, 040},
    /* 0330 combo = 0 */ { 0177777, 0177777},
    /* 0331 combo = 0 */ { 0177777, 0177777},
    /* 0332 combo = 0 */ { 0177777, 0177777},
    /* 0333 combo = 0 */ { 0177777, 0177777},
    /* 0334 combo = 334 */ { 00, 060},
    /* 0335 combo = 334 */ { 01, 060},
    /* 0336 combo = 334 */ { 02, 060},
    /* 0337 combo = 334 */ { 03, 060},
    /* 0340 combo = 0 */ { 0177777, 0177777},
    /* 0341 combo = 241 */ { 0100, 02},
    /* 0342 combo = 142 */ { 0200, 010},
    /* 0343 combo = 343 */ { 00, 012},
    /* 0344 combo = 0 */ { 0344, 00},
    /* 0345 combo = 241 */ { 0104, 02},
    /* 0346 combo = 142 */ { 0204, 010},
    /* 0347 combo = 343 */ { 04, 012},
    /* 0350 combo = 0 */ { 0177777, 0177777},
    /* 0351 combo = 241 */ { 0110, 02},
    /* 0352 combo = 142 */ { 0210, 010},
    /* 0353 combo = 343 */ { 010, 012},
    /* 0354 combo = 214 */ { 0140, 020},
    /* 0355 combo = 255 */ { 0100, 022},
    /* 0356 combo = 356 */ { 00, 030},
    /* 0357 combo = 357 */ { 00, 032},
    /* 0360 combo = 0 */ { 0177777, 0177777},
    /* 0361 combo = 241 */ { 0120, 02},
    /* 0362 combo = 142 */ { 0220, 010},
    /* 0363 combo = 343 */ { 020, 012},
    /* 0364 combo = 124 */ { 0240, 040},
    /* 0365 combo = 365 */ { 00, 042},
    /* 0366 combo = 166 */ { 0200, 050},
    /* 0367 combo = 367 */ { 00, 052},
    /* 0370 combo = 0 */ { 0177777, 0177777},
    /* 0371 combo = 241 */ { 0130, 02},
    /* 0372 combo = 142 */ { 0230, 010},
    /* 0373 combo = 343 */ { 030, 012},
    /* 0374 combo = 334 */ { 040, 060},
    /* 0375 combo = 375 */ { 00, 062},
    /* 0376 combo = 376 */ { 00, 070},
    /* 0377 combo = 377 */ { 00, 072},
    /* 0400 combo = 0 */ { 0177777, 0177777},
    /* 0401 combo = 0 */ { 0177777, 0177777},
    /* 0402 combo = 0 */ { 0177777, 0177777},
    /* 0403 combo = 0 */ { 0177777, 0177777},
    /* 0404 combo = 0 */ { 0177777, 0177777},
    /* 0405 combo = 0 */ { 0177777, 0177777},
    /* 0406 combo = 0 */ { 0177777, 0177777},
    /* 0407 combo = 0 */ { 0177777, 0177777},
    /* 0410 combo = 0 */ { 0177777, 0177777},
    /* 0411 combo = 0 */ { 0177777, 0177777},
    /* 0412 combo = 412 */ { 00, 04},
    /* 0413 combo = 412 */ { 01, 04},
    /* 0414 combo = 0 */ { 0177777, 0177777},
    /* 0415 combo = 0 */ { 0177777, 0177777},
    /* 0416 combo = 412 */ { 04, 04},
    /* 0417 combo = 412 */ { 05, 04},
    /* 0420 combo = 0 */ { 0177777, 0177777},
    /* 0421 combo = 421 */ { 00, 01},
    /* 0422 combo = 0 */ { 0177777, 0177777},
    /* 0423 combo = 421 */ { 02, 01},
    /* 0424 combo = 0 */ { 0177777, 0177777},
    /* 0425 combo = 421 */ { 04, 01},
    /* 0426 combo = 0 */ { 0177777, 0177777},
    /* 0427 combo = 421 */ { 06, 01},
    /* 0430 combo = 0 */ { 0177777, 0177777},
    /* 0431 combo = 421 */ { 010, 01},
    /* 0432 combo = 412 */ { 020, 04},
    /* 0433 combo = 433 */ { 00, 05},
    /* 0434 combo = 0 */ { 0434, 00},
    /* 0435 combo = 421 */ { 014, 01},
    /* 0436 combo = 412 */ { 024, 04},
    /* 0437 combo = 433 */ { 04, 05},
    /* 0440 combo = 0 */ { 0177777, 0177777},
    /* 0441 combo = 0 */ { 0177777, 0177777},
    /* 0442 combo = 0 */ { 0177777, 0177777},
    /* 0443 combo = 0 */ { 0443, 00},
    /* 0444 combo = 0 */ { 0177777, 0177777},
    /* 0445 combo = 0 */ { 0177777, 0177777},
    /* 0446 combo = 0 */ { 0177777, 0177777},
    /* 0447 combo = 0 */ { 0447, 00},
    /* 0450 combo = 0 */ { 0177777, 0177777},
    /* 0451 combo = 0 */ { 0177777, 0177777},
    /* 0452 combo = 412 */ { 040, 04},
    /* 0453 combo = 412 */ { 041, 04},
    /* 0454 combo = 0 */ { 0177777, 0177777},
    /* 0455 combo = 0 */ { 0177777, 0177777},
    /* 0456 combo = 412 */ { 044, 04},
    /* 0457 combo = 412 */ { 045, 04},
    /* 0460 combo = 0 */ { 0177777, 0177777},
    /* 0461 combo = 421 */ { 040, 01},
    /* 0462 combo = 0 */ { 0177777, 0177777},
    /* 0463 combo = 421 */ { 042, 01},
    /* 0464 combo = 0 */ { 0177777, 0177777},
    /* 0465 combo = 421 */ { 044, 01},
    /* 0466 combo = 0 */ { 0177777, 0177777},
    /* 0467 combo = 421 */ { 046, 01},
    /* 0470 combo = 0 */ { 0177777, 0177777},
    /* 0471 combo = 421 */ { 050, 01},
    /* 0472 combo = 412 */ { 060, 04},
    /* 0473 combo = 433 */ { 040, 05},
    /* 0474 combo = 0 */ { 0474, 00},
    /* 0475 combo = 421 */ { 054, 01},
    /* 0476 combo = 412 */ { 064, 04},
    /* 0477 combo = 433 */ { 044, 05},
    /* 0500 combo = 0 */ { 0177777, 0177777},
    /* 0501 combo = 0 */ { 0177777, 0177777},
    /* 0502 combo = 0 */ { 0177777, 0177777},
    /* 0503 combo = 0 */ { 0177777, 0177777},
    /* 0504 combo = 0 */ { 0177777, 0177777},
    /* 0505 combo = 0 */ { 0177777, 0177777},
    /* 0506 combo = 0 */ { 0177777, 0177777},
    /* 0507 combo = 0 */ { 0177777, 0177777},
    /* 0510 combo = 0 */ { 0177777, 0177777},
    /* 0511 combo = 0 */ { 0177777, 0177777},
    /* 0512 combo = 412 */ { 0100, 04},
    /* 0513 combo = 412 */ { 0101, 04},
    /* 0514 combo = 0 */ { 0177777, 0177777},
    /* 0515 combo = 0 */ { 0177777, 0177777},
    /* 0516 combo = 412 */ { 0104, 04},
    /* 0517 combo = 412 */ { 0105, 04},
    /* 0520 combo = 0 */ { 0177777, 0177777},
    /* 0521 combo = 421 */ { 0100, 01},
    /* 0522 combo = 0 */ { 0522, 00},
    /* 0523 combo = 421 */ { 0102, 01},
    /* 0524 combo = 124 */ { 0400, 040},
    /* 0525 combo = 525 */ { 00, 041},
    /* 0526 combo = 124 */ { 0402, 040},
    /* 0527 combo = 525 */ { 02, 041},
    /* 0530 combo = 0 */ { 0177777, 0177777},
    /* 0531 combo = 421 */ { 0110, 01},
    /* 0532 combo = 412 */ { 0120, 04},
    /* 0533 combo = 433 */ { 0100, 05},
    /* 0534 combo = 124 */ { 0410, 040},
    /* 0535 combo = 525 */ { 010, 041},
    /* 0536 combo = 536 */ { 00, 044},
    /* 0537 combo = 537 */ { 00, 045},
    /* 0540 combo = 0 */ { 0177777, 0177777},
    /* 0541 combo = 0 */ { 0177777, 0177777},
    /* 0542 combo = 142 */ { 0400, 010},
    /* 0543 combo = 142 */ { 0401, 010},
    /* 0544 combo = 0 */ { 0177777, 0177777},
    /* 0545 combo = 0 */ { 0177777, 0177777},
    /* 0546 combo = 142 */ { 0404, 010},
    /* 0547 combo = 142 */ { 0405, 010},
    /* 0550 combo = 0 */ { 0177777, 0177777},
    /* 0551 combo = 0 */ { 0177777, 0177777},
    /* 0552 combo = 552 */ { 00, 014},
    /* 0553 combo = 552 */ { 01, 014},
    /* 0554 combo = 0 */ { 0177777, 0177777},
    /* 0555 combo = 0 */ { 0177777, 0177777},
    /* 0556 combo = 552 */ { 04, 014},
    /* 0557 combo = 552 */ { 05, 014},
    /* 0560 combo = 0 */ { 0177777, 0177777},
    /* 0561 combo = 421 */ { 0140, 01},
    /* 0562 combo = 142 */ { 0420, 010},
    /* 0563 combo = 563 */ { 00, 011},
    /* 0564 combo = 124 */ { 0440, 040},
    /* 0565 combo = 525 */ { 040, 041},
    /* 0566 combo = 166 */ { 0400, 050},
    /* 0567 combo = 567 */ { 00, 051},
    /* 0570 combo = 0 */ { 0177777, 0177777},
    /* 0571 combo = 421 */ { 0150, 01},
    /* 0572 combo = 552 */ { 020, 014},
    /* 0573 combo = 573 */ { 00, 015},
    /* 0574 combo = 124 */ { 0450, 040},
    /* 0575 combo = 525 */ { 050, 041},
    /* 0576 combo = 576 */ { 00, 054},
    /* 0577 combo = 577 */ { 00, 055},
    /* 0600 combo = 0 */ { 0177777, 0177777},
    /* 0601 combo = 0 */ { 0177777, 0177777},
    /* 0602 combo = 0 */ { 0177777, 0177777},
    /* 0603 combo = 0 */ { 0177777, 0177777},
    /* 0604 combo = 0 */ { 0177777, 0177777},
    /* 0605 combo = 0 */ { 0177777, 0177777},
    /* 0606 combo = 0 */ { 0177777, 0177777},
    /* 0607 combo = 0 */ { 0177777, 0177777},
    /* 0610 combo = 0 */ { 0177777, 0177777},
    /* 0611 combo = 0 */ { 0611, 00},
    /* 0612 combo = 412 */ { 0200, 04},
    /* 0613 combo = 412 */ { 0201, 04},
    /* 0614 combo = 214 */ { 0400, 020},
    /* 0615 combo = 214 */ { 0401, 020},
    /* 0616 combo = 616 */ { 00, 024},
    /* 0617 combo = 616 */ { 01, 024},
    /* 0620 combo = 0 */ { 0177777, 0177777},
    /* 0621 combo = 421 */ { 0200, 01},
    /* 0622 combo = 0 */ { 0177777, 0177777},
    /* 0623 combo = 421 */ { 0202, 01},
    /* 0624 combo = 0 */ { 0177777, 0177777},
    /* 0625 combo = 421 */ { 0204, 01},
    /* 0626 combo = 0 */ { 0177777, 0177777},
    /* 0627 combo = 421 */ { 0206, 01},
    /* 0630 combo = 0 */ { 0177777, 0177777},
    /* 0631 combo = 421 */ { 0210, 01},
    /* 0632 combo = 412 */ { 0220, 04},
    /* 0633 combo = 433 */ { 0200, 05},
    /* 0634 combo = 214 */ { 0420, 020},
    /* 0635 combo = 635 */ { 00, 021},
    /* 0636 combo = 616 */ { 020, 024},
    /* 0637 combo = 637 */ { 00, 025},
    /* 0640 combo = 0 */ { 0177777, 0177777},
    /* 0641 combo = 241 */ { 0400, 02},
    /* 0642 combo = 0 */ { 0177777, 0177777},
    /* 0643 combo = 241 */ { 0402, 02},
    /* 0644 combo = 0 */ { 0177777, 0177777},
    /* 0645 combo = 241 */ { 0404, 02},
    /* 0646 combo = 0 */ { 0177777, 0177777},
    /* 0647 combo = 241 */ { 0406, 02},
    /* 0650 combo = 0 */ { 0177777, 0177777},
    /* 0651 combo = 241 */ { 0410, 02},
    /* 0652 combo = 412 */ { 0240, 04},
    /* 0653 combo = 653 */ { 00, 06},
    /* 0654 combo = 214 */ { 0440, 020},
    /* 0655 combo = 255 */ { 0400, 022},
    /* 0656 combo = 616 */ { 040, 024},
    /* 0657 combo = 657 */ { 00, 026},
    /* 0660 combo = 0 */ { 0177777, 0177777},
    /* 0661 combo = 661 */ { 00, 03},
    /* 0662 combo = 0 */ { 0177777, 0177777},
    /* 0663 combo = 661 */ { 02, 03},
    /* 0664 combo = 0 */ { 0177777, 0177777},
    /* 0665 combo = 661 */ { 04, 03},
    /* 0666 combo = 0 */ { 0177777, 0177777},
    /* 0667 combo = 661 */ { 06, 03},
    /* 0670 combo = 0 */ { 0177777, 0177777},
    /* 0671 combo = 661 */ { 010, 03},
    /* 0672 combo = 412 */ { 0260, 04},
    /* 0673 combo = 673 */ { 00, 07},
    /* 0674 combo = 214 */ { 0460, 020},
    /* 0675 combo = 675 */ { 00, 023},
    /* 0676 combo = 616 */ { 060, 024},
    /* 0677 combo = 677 */ { 00, 027},
    /* 0700 combo = 0 */ { 0177777, 0177777},
    /* 0701 combo = 0 */ { 0177777, 0177777},
    /* 0702 combo = 0 */ { 0177777, 0177777},
    /* 0703 combo = 0 */ { 0177777, 0177777},
    /* 0704 combo = 0 */ { 0177777, 0177777},
    /* 0705 combo = 0 */ { 0177777, 0177777},
    /* 0706 combo = 0 */ { 0177777, 0177777},
    /* 0707 combo = 0 */ { 0177777, 0177777},
    /* 0710 combo = 0 */ { 0177777, 0177777},
    /* 0711 combo = 0 */ { 0711, 00},
    /* 0712 combo = 412 */ { 0300, 04},
    /* 0713 combo = 412 */ { 0301, 04},
    /* 0714 combo = 214 */ { 0500, 020},
    /* 0715 combo = 214 */ { 0501, 020},
    /* 0716 combo = 616 */ { 0100, 024},
    /* 0717 combo = 616 */ { 0101, 024},
    /* 0720 combo = 0 */ { 0177777, 0177777},
    /* 0721 combo = 421 */ { 0300, 01},
    /* 0722 combo = 0 */ { 0722, 00},
    /* 0723 combo = 421 */ { 0302, 01},
    /* 0724 combo = 124 */ { 0600, 040},
    /* 0725 combo = 525 */ { 0200, 041},
    /* 0726 combo = 124 */ { 0602, 040},
    /* 0727 combo = 525 */ { 0202, 041},
    /* 0730 combo = 0 */ { 0177777, 0177777},
    /* 0731 combo = 421 */ { 0310, 01},
    /* 0732 combo = 412 */ { 0320, 04},
    /* 0733 combo = 433 */ { 0300, 05},
    /* 0734 combo = 334 */ { 0400, 060},
    /* 0735 combo = 735 */ { 00, 061},
    /* 0736 combo = 736 */ { 00, 064},
    /* 0737 combo = 737 */ { 00, 065},
    /* 0740 combo = 0 */ { 0177777, 0177777},
    /* 0741 combo = 241 */ { 0500, 02},
    /* 0742 combo = 142 */ { 0600, 010},
    /* 0743 combo = 343 */ { 0400, 012},
    /* 0744 combo = 0 */ { 0744, 00},
    /* 0745 combo = 241 */ { 0504, 02},
    /* 0746 combo = 142 */ { 0604, 010},
    /* 0747 combo = 343 */ { 0404, 012},
    /* 0750 combo = 0 */ { 0177777, 0177777},
    /* 0751 combo = 241 */ { 0510, 02},
    /* 0752 combo = 552 */ { 0200, 014},
    /* 0753 combo = 753 */ { 00, 016},
    /* 0754 combo = 214 */ { 0540, 020},
    /* 0755 combo = 255 */ { 0500, 022},
    /* 0756 combo = 756 */ { 00, 034},
    /* 0757 combo = 757 */ { 00, 036},
    /* 0760 combo = 0 */ { 0177777, 0177777},
    /* 0761 combo = 661 */ { 0100, 03},
    /* 0762 combo = 142 */ { 0620, 010},
    /* 0763 combo = 763 */ { 00, 013},
    /* 0764 combo = 124 */ { 0640, 040},
    /* 0765 combo = 765 */ { 00, 043},
    /* 0766 combo = 166 */ { 0600, 050},
    /* 0767 combo = 767 */ { 00, 053},
    /* 0770 combo = 0 */ { 0177777, 0177777},
    /* 0771 combo = 661 */ { 0110, 03},
    /* 0772 combo = 552 */ { 0220, 014},
    /* 0773 combo = 773 */ { 00, 017},
    /* 0774 combo = 334 */ { 0440, 060},
    /* 0775 combo = 775 */ { 00, 063},
    /* 0776 combo = 776 */ { 00, 074},
    /* 0777 combo = 777 */ { 00, 077}
};

alignas(32)
const __m128i h2bandplacement[36] {
        _mm_set_epi64x(0,0700070007700070007),
        _mm_set_epi64x(0,0700070007070700007),
        _mm_set_epi64x(0,0700070007700007070),
        _mm_set_epi64x(0,0700070007007700070),
        _mm_set_epi64x(0,0700070007070007700),
        _mm_set_epi64x(0,0700070007007070700),
        _mm_set_epi64x(0,0070700007700070007),
        _mm_set_epi64x(0,0070700007070700007),
        _mm_set_epi64x(0,0070700007700007070),
        _mm_set_epi64x(0,0070700007007700070),
        _mm_set_epi64x(0,0070700007070007700),
        _mm_set_epi64x(0,0070700007007070700),
        _mm_set_epi64x(0,0700007070700070007),
        _mm_set_epi64x(0,0700007070070700007),
        _mm_set_epi64x(0,0700007070700007070),
        _mm_set_epi64x(0,0700007070007700070),
        _mm_set_epi64x(0,0700007070070007700),
        _mm_set_epi64x(0,0700007070007070700),
        _mm_set_epi64x(0,0007700070700070007),
        _mm_set_epi64x(0,0007700070070700007),
        _mm_set_epi64x(0,0007700070700007070),
        _mm_set_epi64x(0,0007700070007700070),
        _mm_set_epi64x(0,0007700070070007700),
        _mm_set_epi64x(0,0007700070007070700),
        _mm_set_epi64x(0,0070007700700070007),
        _mm_set_epi64x(0,0070007700070700007),
        _mm_set_epi64x(0,0070007700700007070),
        _mm_set_epi64x(0,0070007700007700070),
        _mm_set_epi64x(0,0070007700070007700),
        _mm_set_epi64x(0,0070007700007070700),
        _mm_set_epi64x(0,0007070700700070007),
        _mm_set_epi64x(0,0007070700070700007),
        _mm_set_epi64x(0,0007070700700007070),
        _mm_set_epi64x(0,0007070700007700070),
        _mm_set_epi64x(0,0007070700070007700),
        _mm_set_epi64x(0,0007070700007070700)
};

alignas(64)
__m128i h3bandplacement[6] {
        _mm_set_epi64x(00000000000000000340034,00007000000000000000000),
        _mm_set_epi64x(00000000000000000034340,00007000000000000000000),
        _mm_set_epi64x(00000000000000000340003,01070000000000000000000),
        _mm_set_epi64x(00000000000000000003740,00070000000000000000000),
        _mm_set_epi64x(00000000000000000034003,01700000000000000000000),
        _mm_set_epi64x(00000000000000000003434,00700000000000000000000),
};

__m128i v3bandplacement[6] {
        _mm_set_epi64x(00000000000000000200200,00400200200200100100100),
        _mm_set_epi64x(00000000000000000100100,00200400400400100100100),
        _mm_set_epi64x(00000000000000000200200,00400100100100200200200),
        _mm_set_epi64x(00000000000000000040040,00100400400400200200200),
        _mm_set_epi64x(00000000000000000100100,00200100100100400400400),
        _mm_set_epi64x(00000000000000000040040,00100200200200400400400),
};

__m128i v2bandplacement[36] {
        _mm_set_epi64x(00000000000000000022022,00044022022022011011011),
        _mm_set_epi64x(00000000000000000021021,00042024024024011011011),
        _mm_set_epi64x(00000000000000000022022,00044021021021012012012),
        _mm_set_epi64x(00000000000000000020420,01041024024024012012012),
        _mm_set_epi64x(00000000000000000021021,00042021021021014014014),
        _mm_set_epi64x(00000000000000000020420,01041022022022014014014),
        _mm_set_epi64x(00000000000000000012012,00024042042042011011011),
        _mm_set_epi64x(00000000000000000011011,00022044044044011011011),
        _mm_set_epi64x(00000000000000000012012,00024041041041012012012),
        _mm_set_epi64x(00000000000000000010410,01021044044044012012012),
        _mm_set_epi64x(00000000000000000011011,00022041041041014014014),
        _mm_set_epi64x(00000000000000000010410,01021042042042014014014),
        _mm_set_epi64x(00000000000000000022022,00044012012012021021021),
        _mm_set_epi64x(00000000000000000021021,00042014014014021021021),
        _mm_set_epi64x(00000000000000000022022,00044011011011022022022),
        _mm_set_epi64x(00000000000000000020420,01041014014014022022022),
        _mm_set_epi64x(00000000000000000021021,00042011011011024024024),
        _mm_set_epi64x(00000000000000000020420,01041012012012024024024),
        _mm_set_epi64x(00000000000000000006006,00014042042042021021021),
        _mm_set_epi64x(00000000000000000005005,00012044044044021021021),
        _mm_set_epi64x(00000000000000000006006,00014041041041022022022),
        _mm_set_epi64x(00000000000000000004404,01011044044044022022022),
        _mm_set_epi64x(00000000000000000005005,00012041041041024024024),
        _mm_set_epi64x(00000000000000000004404,01011042042042024024024),
        _mm_set_epi64x(00000000000000000012012,00024012012012041041041),
        _mm_set_epi64x(00000000000000000011011,00022014014014041041041),
        _mm_set_epi64x(00000000000000000012012,00024011011011042042042),
        _mm_set_epi64x(00000000000000000010410,01021014014014042042042),
        _mm_set_epi64x(00000000000000000011011,00022011011011044044044),
        _mm_set_epi64x(00000000000000000010410,01021012012012044044044),
        _mm_set_epi64x(00000000000000000006006,00014022022022041041041),
        _mm_set_epi64x(00000000000000000005005,00012024024024041041041),
        _mm_set_epi64x(00000000000000000006006,00014021021021042042042),
        _mm_set_epi64x(00000000000000000004404,01011024024024042042042),
        _mm_set_epi64x(00000000000000000005005,00012021021021044044044),
        _mm_set_epi64x(00000000000000000004404,01011022022022044044044),
};
#endif

// Kind enum
// used for access to data tables and for templates depending on the Kind of section
//
typedef
enum Kind {
   Row = 0,
   Col = 1,
   Box = 2,
   All = 3, // special case for look up table.
} Kind;

const char *kinds[4] = { "row", "col", "box", "all" };
// return status
typedef struct {
   bool solved = false;
   bool unique = true;
   bool verified = false;
   bool used_assumed_uniqueness = false;
} Status;

enum Verbosity {
   VNone  = 0,
   VStats = 1,
   VDebug = 2
};

// bit128_t aggregate type
// used for all 81-bit fields to support different access patterns.
typedef
union {
    __uint128_t    u128;
    __m128i        m128;
    unsigned long long u64[2];
    unsigned int   u32[4];
    unsigned short u16[8];
    unsigned char  u8[16];

    inline operator __uint128_t () {
        return this->u128;
    }

    inline operator __m128i () {
        return this->m128;
    }

    inline __uint128_t operator | (const __uint128_t b) {
        return this->u128 | b;
    }
    inline __uint128_t operator & (const __uint128_t b) {
        return this->u128 & b;
    }

    template<unsigned int off=0>
    inline bool check_indexbit(unsigned int idx) {
        if ( off ) {
            idx += off;
        }
        // slightly faster than _bt_u32
        return this->u32[idx>>5] & (1<<(idx & 0x1f));
        // return _bt_u32(this->u32[idx>>5], idx & 0x1f);
    }

    template<unsigned int off=0>
    inline void set_indexbit(unsigned int idx) {
        if ( off ) {
            idx += off;
        }
        // slightly faster than _bts_u32
        this->u32[idx>>5] |= 1<<(idx & 0x1f);
        // _bts_u32(this->u32[idx>>5], idx & 0x1f);
    }
    inline void set_indexbits(unsigned long long mask, unsigned char pos, unsigned char bitcount) {
        mask &= (unsigned long long)((1LL<<bitcount)-1);  // since the bit count is specific, enforce it
        if ( pos < 64 ) {
            u64[0] |= mask << pos;
            if ( pos + bitcount >= 64 ) {
                u64[1] |= mask >> (64 - pos);
            }
        } else {
            u64[1] |= mask << (pos-64);
        }
    }

    // The following 'immediate' right shift works across the 64 bit boundary and
    // should translate into one unaligned load plus shift plus an 'and' (which can be optimized away as needed)
    // pos must not be less than 32.
    // 32 <= pos < 64, pos+bitcount <= 64
    // able to cover bits 32 through 95.
    // This is ideal for all accesses to the third band, or 8th row.
    template<unsigned int pos, unsigned int bitcount=64>
    inline unsigned long long get_rshfti() {
        unsigned long long ret = (*((unsigned long long *)&u8[4])>>(pos-32));
        if ( bitcount == 64 ) {
            return ret;
        }
        if ( bitcount < 64 ) {
            return ret & ((1LL<<bitcount)-1);
        }
    }

    // requires bitcount <= 64
    inline unsigned long long get_indexbits(unsigned int pos, unsigned int bitcount) {
        // need to use u8 to avoid aliasing error, but in reality, it's u16 we want.
        // Anything larger and the u64 access would go beyond the data.
        unsigned long long *ppos = (unsigned long long*)(&u8[(pos>>3)&~1]);
        pos &= 0xf;
        unsigned long long res = *ppos >> pos;
        if ( pos+bitcount <= 64 ) {
            // clip result
            return _bzhi_u64(res, bitcount);
        }
        return _bzhi_u64(res | (ppos[1] << (64 - pos)), bitcount);
    }

    inline unsigned char popcount() {
        return _popcnt64(u64[0]) + _popcnt32(u32[2]);
    }
} bit128_t;

// the following table serves to quickly identify if two positions of the same digit can view
// each other:
// the first byte represents 0..7 -> 0x1..0x80 of rows, then 0..7 cols and 0..7 boxes.
// the last byte represents the nineth row, col or box, 0b0..0b111.
//

typedef
union {
    unsigned char u8[4];
    unsigned int  u32;
    // assume a single kind is indicated, return it.
    // otherwise, the first kind will be returned.
    inline Kind kindof() {
        unsigned char k = _tzcnt_u32(u32)/8;
        if ( k == 4 ) {
            k = _tzcnt_u32(u8[3]);
        }
        return (Kind)k;   // 32 indicating the class is 0.
    }
} viewbits_t;

alignas(64)
const viewbits_t viewbits_by_i[81] = {
  {  0x1, 0x1,  0x1,   0},  {  0x1, 0x2,  0x1,   0},  {  0x1, 0x4,  0x1,   0},  {  0x1, 0x8,  0x2,   0},  {  0x1, 0x10,  0x2,   0},  {  0x1, 0x20,  0x2,   0},  {  0x1, 0x40,  0x4,   0},  {  0x1, 0x80,  0x4,   0},  {  0x1, 0,  0x4, 0x2},
  {  0x2, 0x1,  0x1,   0},  {  0x2, 0x2,  0x1,   0},  {  0x2, 0x4,  0x1,   0},  {  0x2, 0x8,  0x2,   0},  {  0x2, 0x10,  0x2,   0},  {  0x2, 0x20,  0x2,   0},  {  0x2, 0x40,  0x4,   0},  {  0x2, 0x80,  0x4,   0},  {  0x2, 0,  0x4, 0x2},
  {  0x4, 0x1,  0x1,   0},  {  0x4, 0x2,  0x1,   0},  {  0x4, 0x4,  0x1,   0},  {  0x4, 0x8,  0x2,   0},  {  0x4, 0x10,  0x2,   0},  {  0x4, 0x20,  0x2,   0},  {  0x4, 0x40,  0x4,   0},  {  0x4, 0x80,  0x4,   0},  {  0x4, 0,  0x4, 0x2},
  {  0x8, 0x1,  0x8,   0},  {  0x8, 0x2,  0x8,   0},  {  0x8, 0x4,  0x8,   0},  {  0x8, 0x8, 0x10,   0},  {  0x8, 0x10, 0x10,   0},  {  0x8, 0x20, 0x10,   0},  {  0x8, 0x40, 0x20,   0},  {  0x8, 0x80, 0x20,   0},  {  0x8, 0, 0x20, 0x2},
  { 0x10, 0x1,  0x8,   0},  { 0x10, 0x2,  0x8,   0},  { 0x10, 0x4,  0x8,   0},  { 0x10, 0x8, 0x10,   0},  { 0x10, 0x10, 0x10,   0},  { 0x10, 0x20, 0x10,   0},  { 0x10, 0x40, 0x20,   0},  { 0x10, 0x80, 0x20,   0},  { 0x10, 0, 0x20, 0x2},
  { 0x20, 0x1,  0x8,   0},  { 0x20, 0x2,  0x8,   0},  { 0x20, 0x4,  0x8,   0},  { 0x20, 0x8, 0x10,   0},  { 0x20, 0x10, 0x10,   0},  { 0x20, 0x20, 0x10,   0},  { 0x20, 0x40, 0x20,   0},  { 0x20, 0x80, 0x20,   0},  { 0x20, 0, 0x20, 0x2},
  { 0x40, 0x1, 0x40,   0},  { 0x40, 0x2, 0x40,   0},  { 0x40, 0x4, 0x40,   0},  { 0x40, 0x8, 0x80,   0},  { 0x40, 0x10, 0x80,   0},  { 0x40, 0x20, 0x80,   0},  { 0x40, 0x40,    0, 0x4},  { 0x40, 0x80,    0, 0x4},  { 0x40, 0,    0, 0x6},
  { 0x80, 0x1, 0x40,   0},  { 0x80, 0x2, 0x40,   0},  { 0x80, 0x4, 0x40,   0},  { 0x80, 0x8, 0x80,   0},  { 0x80, 0x10, 0x80,   0},  { 0x80, 0x20, 0x80,   0},  { 0x80, 0x40,    0, 0x4},  { 0x80, 0x80,    0, 0x4},  { 0x80, 0,    0, 0x6},
  {    0, 0x1, 0x40, 0x1},  {    0, 0x2, 0x40, 0x1},  {    0, 0x4, 0x40, 0x1},  {    0, 0x8, 0x80, 0x1},  {    0, 0x10, 0x80, 0x1},  {    0, 0x20, 0x80, 0x1},  {    0, 0x40,    0, 0x5},  {    0, 0x80,    0, 0x5},  {    0, 0,    0, 0x7},
};

// not heavily used
const unsigned char index_by_i[81][3] = {
  { 0, 0, 0},  { 0, 1, 0},  { 0, 2, 0},  { 0, 3, 1},  { 0, 4, 1},  { 0, 5, 1},  { 0, 6, 2},  { 0, 7, 2},  { 0, 8, 2},
  { 1, 0, 0},  { 1, 1, 0},  { 1, 2, 0},  { 1, 3, 1},  { 1, 4, 1},  { 1, 5, 1},  { 1, 6, 2},  { 1, 7, 2},  { 1, 8, 2},
  { 2, 0, 0},  { 2, 1, 0},  { 2, 2, 0},  { 2, 3, 1},  { 2, 4, 1},  { 2, 5, 1},  { 2, 6, 2},  { 2, 7, 2},  { 2, 8, 2},
  { 3, 0, 3},  { 3, 1, 3},  { 3, 2, 3},  { 3, 3, 4},  { 3, 4, 4},  { 3, 5, 4},  { 3, 6, 5},  { 3, 7, 5},  { 3, 8, 5},
  { 4, 0, 3},  { 4, 1, 3},  { 4, 2, 3},  { 4, 3, 4},  { 4, 4, 4},  { 4, 5, 4},  { 4, 6, 5},  { 4, 7, 5},  { 4, 8, 5},
  { 5, 0, 3},  { 5, 1, 3},  { 5, 2, 3},  { 5, 3, 4},  { 5, 4, 4},  { 5, 5, 4},  { 5, 6, 5},  { 5, 7, 5},  { 5, 8, 5},
  { 6, 0, 6},  { 6, 1, 6},  { 6, 2, 6},  { 6, 3, 7},  { 6, 4, 7},  { 6, 5, 7},  { 6, 6, 8},  { 6, 7, 8},  { 6, 8, 8},
  { 7, 0, 6},  { 7, 1, 6},  { 7, 2, 6},  { 7, 3, 7},  { 7, 4, 7},  { 7, 5, 7},  { 7, 6, 8},  { 7, 7, 8},  { 7, 8, 8},
  { 8, 0, 6},  { 8, 1, 6},  { 8, 2, 6},  { 8, 3, 7},  { 8, 4, 7},  { 8, 5, 7},  { 8, 6, 8},  { 8, 7, 8},  { 8, 8, 8},
};

alignas(64)
// some box related indices
//
const unsigned char box_start[81] = {
    0, 0, 0, 3, 3, 3, 6, 6, 6,
    0, 0, 0, 3, 3, 3, 6, 6, 6,
    0, 0, 0, 3, 3, 3, 6, 6, 6,
    27, 27, 27, 30, 30, 30, 33, 33, 33,
    27, 27, 27, 30, 30, 30, 33, 33, 33,
    27, 27, 27, 30, 30, 30, 33, 33, 33,
    54, 54, 54, 57, 57, 57, 60, 60, 60,
    54, 54, 54, 57, 57, 57, 60, 60, 60,
    54, 54, 54, 57, 57, 57, 60, 60, 60
};

const unsigned char box_start_by_boxindex[9] = {
    0, 3, 6, 27, 30, 33, 54, 57, 60
};

const unsigned char box_offset[9] = {
    0, 1, 2, 9, 10, 11, 18, 19, 20
};

alignas(64)
// this table provides the bit masks corresponding to each section index and each Kind of section.
const unsigned long long small_index_lut[9][3][2] = {
{{              0x1ff,        0x0 }, { 0x8040201008040201,      0x100 }, {           0x1c0e07,        0x0 }},
{{            0x3fe00,        0x0 }, {   0x80402010080402,      0x201 }, {           0xe07038,        0x0 }},
{{          0x7fc0000,        0x0 }, {  0x100804020100804,      0x402 }, {          0x70381c0,        0x0 }},
{{        0xff8000000,        0x0 }, {  0x201008040201008,      0x804 }, {     0xe07038000000,        0x0 }},
{{     0x1ff000000000,        0x0 }, {  0x402010080402010,     0x1008 }, {    0x70381c0000000,        0x0 }},
{{   0x3fe00000000000,        0x0 }, {  0x804020100804020,     0x2010 }, {   0x381c0e00000000,        0x0 }},
{{ 0x7fc0000000000000,        0x0 }, { 0x1008040201008040,     0x4020 }, { 0x81c0000000000000,      0x703 }},
{{ 0x8000000000000000,       0xff }, { 0x2010080402010080,     0x8040 }, {  0xe00000000000000,     0x381c }},
{{                0x0,    0x1ff00 }, { 0x4020100804020100,    0x10080 }, { 0x7000000000000000,    0x1c0e0 }},
};

alignas(64)
// lookup tables that may or may not speed things up by avoiding division
const unsigned char index_by_kind[3][81] = {
{    0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1,
    2, 2, 2, 2, 2, 2, 2, 2, 2,
    3, 3, 3, 3, 3, 3, 3, 3, 3,
    4, 4, 4, 4, 4, 4, 4, 4, 4,
    5, 5, 5, 5, 5, 5, 5, 5, 5,
    6, 6, 6, 6, 6, 6, 6, 6, 6,
    7, 7, 7, 7, 7, 7, 7, 7, 7,
    8, 8, 8, 8, 8, 8, 8, 8, 8,
}, {
    0, 1, 2, 3, 4, 5, 6, 7, 8,
    0, 1, 2, 3, 4, 5, 6, 7, 8,
    0, 1, 2, 3, 4, 5, 6, 7, 8,
    0, 1, 2, 3, 4, 5, 6, 7, 8,
    0, 1, 2, 3, 4, 5, 6, 7, 8,
    0, 1, 2, 3, 4, 5, 6, 7, 8,
    0, 1, 2, 3, 4, 5, 6, 7, 8,
    0, 1, 2, 3, 4, 5, 6, 7, 8,
    0, 1, 2, 3, 4, 5, 6, 7, 8,
}, {
    0, 0, 0, 1, 1, 1, 2, 2, 2,
    0, 0, 0, 1, 1, 1, 2, 2, 2,
    0, 0, 0, 1, 1, 1, 2, 2, 2,
    3, 3, 3, 4, 4, 4, 5, 5, 5,
    3, 3, 3, 4, 4, 4, 5, 5, 5,
    3, 3, 3, 4, 4, 4, 5, 5, 5,
    6, 6, 6, 7, 7, 7, 8, 8, 8,
    6, 6, 6, 7, 7, 7, 8, 8, 8,
    6, 6, 6, 7, 7, 7, 8, 8, 8
} };

    // Mapping of the non_canonical triads processing order to canonical order
    // (not considering the gaps every 10 elements):
    // [Note: This permutation is its own reverse, which is called an involution]
    const unsigned char row_triad_canonical_map[27] = {
        0,  1,  2,   9, 10, 11,  18, 19, 20,
        3,  4,  5,  12, 13, 14,  21, 22, 23,
        6,  7,  8,  15, 16, 17,  24, 25, 26
    };

    const unsigned char col_canonical_triad_pos[27] = {
        0,  1,  2,  3,  4,  5,  6,  7,  8,
       27, 28, 29, 30, 31, 32, 33, 34, 35,
       54, 55, 56, 57, 58, 59, 60, 61, 62
    };

    const unsigned char row_triads_lut[9] = {
        0, 10, 20, 3, 13, 23, 6, 16, 26 };

    const unsigned char row_triad_index_to_offset[30] = {
         0, 3, 6,27,30,33,54,57,60,0xff,
         9,12,15,36,39,42,63,66,69,0xff,
        18,21,24,45,48,51,72,75,78,0xff };

    // find the (row) 3-bit positional pattern within any band by index 0..8
    const unsigned int bandbits_by_index[9] = {
        0x7, 0x7<<3, 0x7<<6, 0x7<<9, 0x7<<12, 0x7<<15, 0x7<<18, 0x7<<21, 0x7<<24 };

#ifdef OPT_DGT
const bit128_t bandbits[6] = {
                         ((const bit128_t*)&small_index_lut[0][Row])->u128
                       | ((const bit128_t*)&small_index_lut[1][Row])->u128
                       | ((const bit128_t*)&small_index_lut[2][Row])->u128,
                         ((const bit128_t*)&small_index_lut[3][Row])->u128
                       | ((const bit128_t*)&small_index_lut[4][Row])->u128
                       | ((const bit128_t*)&small_index_lut[5][Row])->u128,
                         ((const bit128_t*)&small_index_lut[6][Row])->u128
                       | ((const bit128_t*)&small_index_lut[7][Row])->u128
                       | ((const bit128_t*)&small_index_lut[8][Row])->u128,
                         ((const bit128_t*)&small_index_lut[0][Col])->u128
                       | ((const bit128_t*)&small_index_lut[1][Col])->u128
                       | ((const bit128_t*)&small_index_lut[2][Col])->u128,
                       ((const bit128_t*)&small_index_lut[3][Col])->u128
                       | ((const bit128_t*)&small_index_lut[4][Col])->u128
                       | ((const bit128_t*)&small_index_lut[5][Col])->u128,
                       ((const bit128_t*)&small_index_lut[6][Col])->u128
                       | ((const bit128_t*)&small_index_lut[7][Col])->u128
                       | ((const bit128_t*)&small_index_lut[8][Col])->u128 };
#endif

const unsigned char *row_index = index_by_kind[Row];
const unsigned char *column_index = index_by_kind[Col];
const unsigned char *box_index = index_by_kind[Box];

const unsigned short bitx3_lut[8] = {
   0x0,      0x7,      0x38,     0x3f,
   0x1c0,    0x1c7,    0x1f8,    0x1ff
};

alignas(64)
// this table provides the bit masks corresponding to each index and each Kind of section.
// The 4th column contains all Kind's or'ed together, but without the origin index bit.
// Heavily used.
// Casually speaking, this table provides the 'visibility' from each cell onto the grid
// for selected sections and all of them.
// Also consider that each pair of unsigned long long can be casted in several manners,
// most generally bit128_t.
//
const unsigned long long big_index_lut[81][4][2] = {
{{              0x1ff,        0x0 }, { 0x8040201008040201,      0x100 }, {           0x1c0e07,        0x0 }, { 0x80402010081c0ffe,      0x100 }},
{{              0x1ff,        0x0 }, {   0x80402010080402,      0x201 }, {           0x1c0e07,        0x0 }, {   0x804020101c0ffd,      0x201 }},
{{              0x1ff,        0x0 }, {  0x100804020100804,      0x402 }, {           0x1c0e07,        0x0 }, {  0x1008040201c0ffb,      0x402 }},
{{              0x1ff,        0x0 }, {  0x201008040201008,      0x804 }, {           0xe07038,        0x0 }, {  0x201008040e071f7,      0x804 }},
{{              0x1ff,        0x0 }, {  0x402010080402010,     0x1008 }, {           0xe07038,        0x0 }, {  0x402010080e071ef,     0x1008 }},
{{              0x1ff,        0x0 }, {  0x804020100804020,     0x2010 }, {           0xe07038,        0x0 }, {  0x804020100e071df,     0x2010 }},
{{              0x1ff,        0x0 }, { 0x1008040201008040,     0x4020 }, {          0x70381c0,        0x0 }, { 0x10080402070381bf,     0x4020 }},
{{              0x1ff,        0x0 }, { 0x2010080402010080,     0x8040 }, {          0x70381c0,        0x0 }, { 0x201008040703817f,     0x8040 }},
{{              0x1ff,        0x0 }, { 0x4020100804020100,    0x10080 }, {          0x70381c0,        0x0 }, { 0x40201008070380ff,    0x10080 }},
{{            0x3fe00,        0x0 }, { 0x8040201008040201,      0x100 }, {           0x1c0e07,        0x0 }, { 0x80402010081ffc07,      0x100 }},
{{            0x3fe00,        0x0 }, {   0x80402010080402,      0x201 }, {           0x1c0e07,        0x0 }, {   0x804020101ffa07,      0x201 }},
{{            0x3fe00,        0x0 }, {  0x100804020100804,      0x402 }, {           0x1c0e07,        0x0 }, {  0x1008040201ff607,      0x402 }},
{{            0x3fe00,        0x0 }, {  0x201008040201008,      0x804 }, {           0xe07038,        0x0 }, {  0x201008040e3ee38,      0x804 }},
{{            0x3fe00,        0x0 }, {  0x402010080402010,     0x1008 }, {           0xe07038,        0x0 }, {  0x402010080e3de38,     0x1008 }},
{{            0x3fe00,        0x0 }, {  0x804020100804020,     0x2010 }, {           0xe07038,        0x0 }, {  0x804020100e3be38,     0x2010 }},
{{            0x3fe00,        0x0 }, { 0x1008040201008040,     0x4020 }, {          0x70381c0,        0x0 }, { 0x1008040207037fc0,     0x4020 }},
{{            0x3fe00,        0x0 }, { 0x2010080402010080,     0x8040 }, {          0x70381c0,        0x0 }, { 0x201008040702ffc0,     0x8040 }},
{{            0x3fe00,        0x0 }, { 0x4020100804020100,    0x10080 }, {          0x70381c0,        0x0 }, { 0x402010080701ffc0,    0x10080 }},
{{          0x7fc0000,        0x0 }, { 0x8040201008040201,      0x100 }, {           0x1c0e07,        0x0 }, { 0x804020100ff80e07,      0x100 }},
{{          0x7fc0000,        0x0 }, {   0x80402010080402,      0x201 }, {           0x1c0e07,        0x0 }, {   0x80402017f40e07,      0x201 }},
{{          0x7fc0000,        0x0 }, {  0x100804020100804,      0x402 }, {           0x1c0e07,        0x0 }, {  0x100804027ec0e07,      0x402 }},
{{          0x7fc0000,        0x0 }, {  0x201008040201008,      0x804 }, {           0xe07038,        0x0 }, {  0x201008047dc7038,      0x804 }},
{{          0x7fc0000,        0x0 }, {  0x402010080402010,     0x1008 }, {           0xe07038,        0x0 }, {  0x402010087bc7038,     0x1008 }},
{{          0x7fc0000,        0x0 }, {  0x804020100804020,     0x2010 }, {           0xe07038,        0x0 }, {  0x8040201077c7038,     0x2010 }},
{{          0x7fc0000,        0x0 }, { 0x1008040201008040,     0x4020 }, {          0x70381c0,        0x0 }, { 0x1008040206ff81c0,     0x4020 }},
{{          0x7fc0000,        0x0 }, { 0x2010080402010080,     0x8040 }, {          0x70381c0,        0x0 }, { 0x2010080405ff81c0,     0x8040 }},
{{          0x7fc0000,        0x0 }, { 0x4020100804020100,    0x10080 }, {          0x70381c0,        0x0 }, { 0x4020100803ff81c0,    0x10080 }},
{{        0xff8000000,        0x0 }, { 0x8040201008040201,      0x100 }, {     0xe07038000000,        0x0 }, { 0x8040e07ff0040201,      0x100 }},
{{        0xff8000000,        0x0 }, {   0x80402010080402,      0x201 }, {     0xe07038000000,        0x0 }, {   0x80e07fe8080402,      0x201 }},
{{        0xff8000000,        0x0 }, {  0x100804020100804,      0x402 }, {     0xe07038000000,        0x0 }, {  0x100e07fd8100804,      0x402 }},
{{        0xff8000000,        0x0 }, {  0x201008040201008,      0x804 }, {    0x70381c0000000,        0x0 }, {  0x207038fb8201008,      0x804 }},
{{        0xff8000000,        0x0 }, {  0x402010080402010,     0x1008 }, {    0x70381c0000000,        0x0 }, {  0x407038f78402010,     0x1008 }},
{{        0xff8000000,        0x0 }, {  0x804020100804020,     0x2010 }, {    0x70381c0000000,        0x0 }, {  0x807038ef8804020,     0x2010 }},
{{        0xff8000000,        0x0 }, { 0x1008040201008040,     0x4020 }, {   0x381c0e00000000,        0x0 }, { 0x10381c0df9008040,     0x4020 }},
{{        0xff8000000,        0x0 }, { 0x2010080402010080,     0x8040 }, {   0x381c0e00000000,        0x0 }, { 0x20381c0bfa010080,     0x8040 }},
{{        0xff8000000,        0x0 }, { 0x4020100804020100,    0x10080 }, {   0x381c0e00000000,        0x0 }, { 0x40381c07fc020100,    0x10080 }},
{{     0x1ff000000000,        0x0 }, { 0x8040201008040201,      0x100 }, {     0xe07038000000,        0x0 }, { 0x8040ffe038040201,      0x100 }},
{{     0x1ff000000000,        0x0 }, {   0x80402010080402,      0x201 }, {     0xe07038000000,        0x0 }, {   0x80ffd038080402,      0x201 }},
{{     0x1ff000000000,        0x0 }, {  0x100804020100804,      0x402 }, {     0xe07038000000,        0x0 }, {  0x100ffb038100804,      0x402 }},
{{     0x1ff000000000,        0x0 }, {  0x201008040201008,      0x804 }, {    0x70381c0000000,        0x0 }, {  0x2071f71c0201008,      0x804 }},
{{     0x1ff000000000,        0x0 }, {  0x402010080402010,     0x1008 }, {    0x70381c0000000,        0x0 }, {  0x4071ef1c0402010,     0x1008 }},
{{     0x1ff000000000,        0x0 }, {  0x804020100804020,     0x2010 }, {    0x70381c0000000,        0x0 }, {  0x8071df1c0804020,     0x2010 }},
{{     0x1ff000000000,        0x0 }, { 0x1008040201008040,     0x4020 }, {   0x381c0e00000000,        0x0 }, { 0x10381bfe01008040,     0x4020 }},
{{     0x1ff000000000,        0x0 }, { 0x2010080402010080,     0x8040 }, {   0x381c0e00000000,        0x0 }, { 0x203817fe02010080,     0x8040 }},
{{     0x1ff000000000,        0x0 }, { 0x4020100804020100,    0x10080 }, {   0x381c0e00000000,        0x0 }, { 0x40380ffe04020100,    0x10080 }},
{{   0x3fe00000000000,        0x0 }, { 0x8040201008040201,      0x100 }, {     0xe07038000000,        0x0 }, { 0x807fc07038040201,      0x100 }},
{{   0x3fe00000000000,        0x0 }, {   0x80402010080402,      0x201 }, {     0xe07038000000,        0x0 }, {   0xbfa07038080402,      0x201 }},
{{   0x3fe00000000000,        0x0 }, {  0x100804020100804,      0x402 }, {     0xe07038000000,        0x0 }, {  0x13f607038100804,      0x402 }},
{{   0x3fe00000000000,        0x0 }, {  0x201008040201008,      0x804 }, {    0x70381c0000000,        0x0 }, {  0x23ee381c0201008,      0x804 }},
{{   0x3fe00000000000,        0x0 }, {  0x402010080402010,     0x1008 }, {    0x70381c0000000,        0x0 }, {  0x43de381c0402010,     0x1008 }},
{{   0x3fe00000000000,        0x0 }, {  0x804020100804020,     0x2010 }, {    0x70381c0000000,        0x0 }, {  0x83be381c0804020,     0x2010 }},
{{   0x3fe00000000000,        0x0 }, { 0x1008040201008040,     0x4020 }, {   0x381c0e00000000,        0x0 }, { 0x1037fc0e01008040,     0x4020 }},
{{   0x3fe00000000000,        0x0 }, { 0x2010080402010080,     0x8040 }, {   0x381c0e00000000,        0x0 }, { 0x202ffc0e02010080,     0x8040 }},
{{   0x3fe00000000000,        0x0 }, { 0x4020100804020100,    0x10080 }, {   0x381c0e00000000,        0x0 }, { 0x401ffc0e04020100,    0x10080 }},
{{ 0x7fc0000000000000,        0x0 }, { 0x8040201008040201,      0x100 }, { 0x81c0000000000000,      0x703 }, { 0xff80201008040201,      0x703 }},
{{ 0x7fc0000000000000,        0x0 }, {   0x80402010080402,      0x201 }, { 0x81c0000000000000,      0x703 }, { 0xff40402010080402,      0x703 }},
{{ 0x7fc0000000000000,        0x0 }, {  0x100804020100804,      0x402 }, { 0x81c0000000000000,      0x703 }, { 0xfec0804020100804,      0x703 }},
{{ 0x7fc0000000000000,        0x0 }, {  0x201008040201008,      0x804 }, {  0xe00000000000000,     0x381c }, { 0x7dc1008040201008,     0x381c }},
{{ 0x7fc0000000000000,        0x0 }, {  0x402010080402010,     0x1008 }, {  0xe00000000000000,     0x381c }, { 0x7bc2010080402010,     0x381c }},
{{ 0x7fc0000000000000,        0x0 }, {  0x804020100804020,     0x2010 }, {  0xe00000000000000,     0x381c }, { 0x77c4020100804020,     0x381c }},
{{ 0x7fc0000000000000,        0x0 }, { 0x1008040201008040,     0x4020 }, { 0x7000000000000000,    0x1c0e0 }, { 0x6fc8040201008040,    0x1c0e0 }},
{{ 0x7fc0000000000000,        0x0 }, { 0x2010080402010080,     0x8040 }, { 0x7000000000000000,    0x1c0e0 }, { 0x5fd0080402010080,    0x1c0e0 }},
{{ 0x7fc0000000000000,        0x0 }, { 0x4020100804020100,    0x10080 }, { 0x7000000000000000,    0x1c0e0 }, { 0x3fe0100804020100,    0x1c0e0 }},
{{ 0x8000000000000000,       0xff }, { 0x8040201008040201,      0x100 }, { 0x81c0000000000000,      0x703 }, {  0x1c0201008040201,      0x7ff }},
{{ 0x8000000000000000,       0xff }, {   0x80402010080402,      0x201 }, { 0x81c0000000000000,      0x703 }, { 0x81c0402010080402,      0x7fe }},
{{ 0x8000000000000000,       0xff }, {  0x100804020100804,      0x402 }, { 0x81c0000000000000,      0x703 }, { 0x81c0804020100804,      0x7fd }},
{{ 0x8000000000000000,       0xff }, {  0x201008040201008,      0x804 }, {  0xe00000000000000,     0x381c }, { 0x8e01008040201008,     0x38fb }},
{{ 0x8000000000000000,       0xff }, {  0x402010080402010,     0x1008 }, {  0xe00000000000000,     0x381c }, { 0x8e02010080402010,     0x38f7 }},
{{ 0x8000000000000000,       0xff }, {  0x804020100804020,     0x2010 }, {  0xe00000000000000,     0x381c }, { 0x8e04020100804020,     0x38ef }},
{{ 0x8000000000000000,       0xff }, { 0x1008040201008040,     0x4020 }, { 0x7000000000000000,    0x1c0e0 }, { 0xf008040201008040,    0x1c0df }},
{{ 0x8000000000000000,       0xff }, { 0x2010080402010080,     0x8040 }, { 0x7000000000000000,    0x1c0e0 }, { 0xf010080402010080,    0x1c0bf }},
{{ 0x8000000000000000,       0xff }, { 0x4020100804020100,    0x10080 }, { 0x7000000000000000,    0x1c0e0 }, { 0xf020100804020100,    0x1c07f }},
{{                0x0,    0x1ff00 }, { 0x8040201008040201,      0x100 }, { 0x81c0000000000000,      0x703 }, { 0x81c0201008040201,    0x1fe03 }},
{{                0x0,    0x1ff00 }, {   0x80402010080402,      0x201 }, { 0x81c0000000000000,      0x703 }, { 0x81c0402010080402,    0x1fd03 }},
{{                0x0,    0x1ff00 }, {  0x100804020100804,      0x402 }, { 0x81c0000000000000,      0x703 }, { 0x81c0804020100804,    0x1fb03 }},
{{                0x0,    0x1ff00 }, {  0x201008040201008,      0x804 }, {  0xe00000000000000,     0x381c }, {  0xe01008040201008,    0x1f71c }},
{{                0x0,    0x1ff00 }, {  0x402010080402010,     0x1008 }, {  0xe00000000000000,     0x381c }, {  0xe02010080402010,    0x1ef1c }},
{{                0x0,    0x1ff00 }, {  0x804020100804020,     0x2010 }, {  0xe00000000000000,     0x381c }, {  0xe04020100804020,    0x1df1c }},
{{                0x0,    0x1ff00 }, { 0x1008040201008040,     0x4020 }, { 0x7000000000000000,    0x1c0e0 }, { 0x7008040201008040,    0x1bfe0 }},
{{                0x0,    0x1ff00 }, { 0x2010080402010080,     0x8040 }, { 0x7000000000000000,    0x1c0e0 }, { 0x7010080402010080,    0x17fe0 }},
{{                0x0,    0x1ff00 }, { 0x4020100804020100,    0x10080 }, { 0x7000000000000000,    0x1c0e0 }, { 0x7020100804020100,     0xffe0 }},
};

// For printing a cell location, use the following table.
// It is mutable so that the 0-based representation can be made 1-based (or 'A'-based) on start-up.
char cl2txt[81][6] = {
        /*   0 */  "[1,1]",
        /*   1 */  "[1,2]",
        /*   2 */  "[1,3]",
        /*   3 */  "[1,4]",
        /*   4 */  "[1,5]",
        /*   5 */  "[1,6]",
        /*   6 */  "[1,7]",
        /*   7 */  "[1,8]",
        /*   8 */  "[1,9]",
        /*   9 */  "[2,1]",
        /*  10 */  "[2,2]",
        /*  11 */  "[2,3]",
        /*  12 */  "[2,4]",
        /*  13 */  "[2,5]",
        /*  14 */  "[2,6]",
        /*  15 */  "[2,7]",
        /*  16 */  "[2,8]",
        /*  17 */  "[2,9]",
        /*  18 */  "[3,1]",
        /*  19 */  "[3,2]",
        /*  20 */  "[3,3]",
        /*  21 */  "[3,4]",
        /*  22 */  "[3,5]",
        /*  23 */  "[3,6]",
        /*  24 */  "[3,7]",
        /*  25 */  "[3,8]",
        /*  26 */  "[3,9]",
        /*  27 */  "[4,1]",
        /*  28 */  "[4,2]",
        /*  29 */  "[4,3]",
        /*  30 */  "[4,4]",
        /*  31 */  "[4,5]",
        /*  32 */  "[4,6]",
        /*  33 */  "[4,7]",
        /*  34 */  "[4,8]",
        /*  35 */  "[4,9]",
        /*  36 */  "[5,1]",
        /*  37 */  "[5,2]",
        /*  38 */  "[5,3]",
        /*  39 */  "[5,4]",
        /*  40 */  "[5,5]",
        /*  41 */  "[5,6]",
        /*  42 */  "[5,7]",
        /*  43 */  "[5,8]",
        /*  44 */  "[5,9]",
        /*  45 */  "[6,1]",
        /*  46 */  "[6,2]",
        /*  47 */  "[6,3]",
        /*  48 */  "[6,4]",
        /*  49 */  "[6,5]",
        /*  50 */  "[6,6]",
        /*  51 */  "[6,7]",
        /*  52 */  "[6,8]",
        /*  53 */  "[6,9]",
        /*  54 */  "[7,1]",
        /*  55 */  "[7,2]",
        /*  56 */  "[7,3]",
        /*  57 */  "[7,4]",
        /*  58 */  "[7,5]",
        /*  59 */  "[7,6]",
        /*  60 */  "[7,7]",
        /*  61 */  "[7,8]",
        /*  62 */  "[7,9]",
        /*  63 */  "[8,1]",
        /*  64 */  "[8,2]",
        /*  65 */  "[8,3]",
        /*  66 */  "[8,4]",
        /*  67 */  "[8,5]",
        /*  68 */  "[8,6]",
        /*  69 */  "[8,7]",
        /*  70 */  "[8,8]",
        /*  71 */  "[8,9]",
        /*  72 */  "[9,1]",
        /*  73 */  "[9,2]",
        /*  74 */  "[9,3]",
        /*  75 */  "[9,4]",
        /*  76 */  "[9,5]",
        /*  77 */  "[9,6]",
        /*  78 */  "[9,7]",
        /*  79 */  "[9,8]",
        /*  80 */  "[9,9]"
    };

const signed char box_perp_ind_incr[9] = { 1, 1, 7, 1, 1, 7, 1, 1, -20};

const unsigned char group4x3offsets[12] = { box_offset[0], box_offset[1], box_offset[2], 0xff,
                                            box_offset[3], box_offset[4], box_offset[5], 0xff,
                                            box_offset[6], box_offset[7], box_offset[8], 0xff };

alignas(64)
// general purpose / multiple locations:
const __m256i nibble_mask = _mm256_set1_epi8(0x0F);
const __m256i maskff = _mm256_set1_epi16(0xff);
const __m256i dgt1 = _mm256_set1_epi8('1');
const __m256i maskff_epi8 = _mm256_set1_epi8(0xff);

// used for expanding bit vectors to boolean vectors
const __m256i bit_mask_expand = _mm256_setr_epi16(1<<0, 1<<1, 1<<2, 1<<3, 1<<4, 1<<5, 1<<6, 1<<7, 1<<8, 1<<9, 1<<10, 1<<11, 1<<12, 1<<13, 1<<14, 1<<15);
const __m256i shuffle_interleaved_mask_bytes = _mm256_setr_epi8(0,0,0,0,0,0,0,0,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,3,3,3,3,3,3,3,3);
const __m256i shuffle_mask_bytes = _mm256_setr_epi8(0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3);

// used for load from the grid
const __m256i select_bits   = _mm256_setr_epi8(1<<0,1<<1,1<<2,1<<3,1<<4,1<<5,1<<6,1<<7,1<<0,1<<1,1<<2,1<<3,1<<4,1<<5,1<<6,1<<7,
                                               1<<0,1<<1,1<<2,1<<3,1<<4,1<<5,1<<6,1<<7,1<<0,1<<1,1<<2,1<<3,1<<4,1<<5,1<<6,1<<7);
const __m256i ones_epi8 = _mm256_set1_epi8(1);


// used for writing back the grid
const __m256i lut_lo = _mm256_set_epi8('?', '?', '?', '?', '?', '?', '?', '4', '?', '?', '?', '3', '?', '2', '1', '9',
                                       '?', '?', '?', '?', '?', '?', '?', '4', '?', '?', '?', '3', '?', '2', '1', '9');
const __m256i lut_hi = _mm256_set_epi8('?', '?', '?', '?', '?', '?', '?', '8', '?', '?', '?', '7', '?', '6', '5', '9',
                                       '?', '?', '?', '?', '?', '?', '?', '8', '?', '?', '?', '7', '?', '6', '5', '9');

// used in enter:
const __m256i ones = _mm256_set1_epi16(1);

// used for updating the staged bytes:
// see update_stage_bytes for an explanation
const __m256i stage_bytes_mask = _mm256_setr_epi8(0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,
                                                 0xff,0,0,0,0,0,0,0,
                                                 0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,
                                                 0,0,0,0,0,0,0,0 );

// used in verify:
const __m256i mask9 = _mm256_setr_epi64x(-1LL, -1LL, 0xffffLL, 0);
const __m256i ones9 = _mm256_and_si256(ones, mask9);

// used in triads:
const __m256i mask11hi = _mm256_setr_epi64x(0LL, 0LL, 0xffffLL<<48, ~0LL);
const __m256i mask1ff   = _mm256_set1_epi16(0x1ff);
const __m256i mask9x1ff = _mm256_and_si256(mask1ff, mask9);

// used in triads (row triad capture):
const __m256i shuff725to012 = _mm256_setr_epi8(14, 15,  4,  5, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1,  -1, -1,
                                               14, 15,  4,  5, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1,  -1, -1);

// used in triads processing:
const __m256i mask_musts = _mm256_setr_epi16( 0x1ff, 0x1ff, 0x1ff, 0x1ff, 0x1ff, 0x1ff, 0, 0,
                                              0x1ff, 0x1ff, 0x1ff, 0,     0,     0,     0, 0);
//    rotation of groups of 3 triads *must/*mustnt.
const __m256i rot_hpeers = _mm256_setr_epi8( 2,3,4,5,0,1, 8, 9,10,11, 6, 7,-1,-1,-1,-1,
                                             2,3,4,5,0,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1);
//    shuffle within tmustnt to setup for aligned 9 triads (order of candidates).
const __m256i shuff_tmustnt = _mm256_setr_epi8( -1,-1,-1,-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,
                                                -1,-1,-1,-1, 4, 5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1);
const __m256i shuff_row_mask = _mm256_setr_epi8( 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 4, 5, 4, 5,
                                                 4, 5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1);
const __m256i shuff_row_mask2 = _mm256_setr_epi8( 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 8, 9, 8, 9,
                                                  8, 9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1);

// used in triad resolution
const __m256i mask27    = _mm256_setr_epi64x(-1LL, (long long int)0xffffffffffff00ffLL, (long long int)0xffffffff00ffffffLL, 0xffffffffffLL);
const __m256i threes    = _mm256_set1_epi8 ( 3 );
const __m256i fours     = _mm256_set1_epi8 ( 4 );
//   popcnt by nibble
const __m256i lookup    = _mm256_setr_epi8(0 ,1 ,1 ,2 ,1 ,2 ,2 ,3 ,1 ,2 ,2 ,3 ,2 ,3 ,3 ,4,
                                           0 ,1 ,1 ,2 ,1 ,2 ,2 ,3 ,1 ,2 ,2 ,3 ,2 ,3 ,3 ,4);
// used in make_guess
const __m256i shuf = _mm256_setr_epi8(0,1,6,7,12,13,2,3,8,9,14,15,4,5,10,11,0,1,6,7,12,13,2,3,8,9,14,15,4,5,10,11);
const __m256i shuf8x32 = _mm256_setr_epi32(0,1,2,4,5,6,7,3);

// interleaved indexing for stage_bytes:
const unsigned char stage_bytes_lu[81] = { 0, 1, 2, 3, 4, 5, 6, 7,
                                       16, 17, 18, 19, 20, 21, 22, 23,
                                        8,  9, 10, 11, 12, 13, 14, 15,
                                       24, 25, 26, 27, 28, 29, 30, 31,
                                       32, 33, 34, 35, 36, 37, 38, 39,
                                       48, 49, 50, 51, 52, 53, 54, 55,
                                       40, 41, 42, 43, 44, 45, 46, 47,
                                       56, 57, 58, 59, 60, 61, 62, 63,
                                       64, 65, 66, 67, 68, 69, 70, 71,
                                       80, 81, 82, 83, 84, 85, 86, 87,
                                       72 };

// used with mode_dgt
const __m256i shuff_col_ordered1 = _mm256_setr_epi8( 0, 1, 2,-1,-1,-1,12,13,14, 3, 4, 5,-1,-1,-1,15,
                                                     8, 9,-1,-1, 0,-1,-1,-1,10,11,12,-1,-1,-1,-1,-1);
const __m256i shuff_col_ordered2 = _mm256_setr_epi8(-1,-1, 6, 7,-1, 8, 9,10,-1,-1,-1,-1,-1,-1,-1,-1,
                                                    -1,-1,-1, 2, 3, 4,-1,-1,-1,-1,-1,-1, 5, 6, 7,-1);

const __uint128_t mask_81bit = (((__uint128_t)0x1ffff)<<64)+0xffffffffffffffff;

class
alignas(64)
Counters {
public:
    long long past_naked_count;             // how often do we get past the naked single serach
    long long digits_entered_and_retracted; // to measure guessing overhead
    long long triads_resolved;              // how many triads did we resolved
    long long triad_updates;                // how many triads did cancel candidates
    long long single_digit_placement_checks;  // how many checks of a single digits candidate placements?
    long long single_digit_incompatibilities; // how many times found anything superfluous?
    long puzzles_seen;                      // for thread 0 only
    long last_puzzles_seen;                 // for thread 0 only
    long last_guesses;                      // for thread 0 only
    long preset_count;                     // total presets in the puzzle
    long bug_count;                        // universal grave detected
    long bug_plus1_count;                  // universal grave avoided (bug+1)
    long guesses;                          // how many guesses did it take
    long trackbacks;                       // how often did we back track
    long solved_count;                     // puzzles solved
    long no_guess_cnt;                     // how many puzzles were solved without guessing
    long unsolved_count;                   // puzzles unsolved (no solution exists)
    long non_unique_count;                 // puzzles not unique (with -u)
    long not_verified_count;               // puzzles non verified (with -v)
    long verified_count;                   // puzzles successfully verified (with -v)
    long no_bivals_count;                  // counts board states without bivalues
inline Counters &operator += (const Counters &a) {
    this->past_naked_count += a.past_naked_count;
    this->digits_entered_and_retracted += a.digits_entered_and_retracted;
    this->triads_resolved += a.triads_resolved;
    this->triad_updates += a.triad_updates;
    this->single_digit_placement_checks += a.single_digit_placement_checks;
    this->single_digit_incompatibilities += a.single_digit_incompatibilities;
    this->preset_count += a.preset_count;
    this->bug_count += a.bug_count;
    this->bug_plus1_count += a.bug_plus1_count;
    this->guesses += a.guesses;
    this->trackbacks += a.trackbacks;
    this->solved_count += a.solved_count;
    this->no_guess_cnt += a.no_guess_cnt;
    this->unsolved_count += a.unsolved_count;
    this->non_unique_count += a.non_unique_count; 
    this->not_verified_count += a.not_verified_count;
    this->verified_count += a.verified_count;
    this->no_bivals_count += a.no_bivals_count;
    return *this;
}
};

Counters global_counters;

// wrapping index table for mode_dgt
const unsigned char index_table_dgt[] = { 0,1,2,3,4,5,6,7,8,0,1,2,3,4,5,6,7 };

bool bmi2_support = false;
bool pext_support = false;

// stats and command line options
int reportstats     = 0; // collect and report some statistics
int reporttimings   = 0; // report timings only
int verify          = 0; // verify solution correctness (implied otherwise)
int debug           = 0; // provide step by step output on the solution
int thorough_check  = 0; // check for back tracking even if no guess was made.
int numthreads      = 0; // if not 0, number of threads
int warnings        = 0; // display warnings
int dgt_complexity_cutoff = 144; // tunable mode_dgt cutoff value (see complexity_cutoff_table)
bool dgt_tune_for_speed = true;

int guess_score_threshold      = 3; // could be tuned dynamically to 2 for guess-intensive test sets

// puzzle rules
typedef enum {
    Regular  = 0,   // R - Default: Find solution, assuming it will be unique
                    // (this is fastest but is not suitable for non-unique puzzles)
    FindOne  = 1,   // O - Search for one solution
    Multiple = 2    // M - Determine whether puzzles are non-unique
} Rules;

Rules rules;

signed char *output;
bool mode_dgt=false;            // 'X', see OPT_DGT
bool mode_dgt_init;             // initial setting of mode_dgt
bool mode_selftune = false;

// isolate the strict-aliasing warning for casts from unsigned long long [2] arrays:
inline const __uint128_t *cast2cu128(const unsigned long long *from) {

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
	return (const __uint128_t *)from;
#pragma GCC diagnostic pop
}

inline unsigned char tzcnt_and_mask(unsigned long long &mask) {
    unsigned char ret = _tzcnt_u64(mask);
    mask = _blsr_u64(mask);
    return ret;
}

inline unsigned char tzcnt_and_mask(unsigned int &mask) {
    unsigned char ret = _tzcnt_u32(mask);
    mask = _blsr_u32(mask);
    return ret;
}

inline unsigned char tzcnt_and_mask(bit128_t &mask) {
    if ( mask.u64[0] ) {
        unsigned char ret = _tzcnt_u64(mask.u64[0]);
        mask.u64[0] = _blsr_u64(mask.u64[0]);
        return ret;
    }
    unsigned char ret = _tzcnt_u64(mask.u64[1]);
    mask.u64[1] = _blsr_u64(mask.u64[1]);
    return ret+64;
    // returns 128 for input of 0.
}

inline __m256i expand_bitvector(unsigned short m) {
    return _mm256_cmpeq_epi16(_mm256_and_si256( bit_mask_expand,_mm256_set1_epi16(m)), bit_mask_expand);
}

template<bool for_interleaving=false>
inline __m256i expand_bitvector_epi8(unsigned int m) {
    __m256i bits = _mm256_shuffle_epi8(_mm256_set1_epi32(m), for_interleaving?shuffle_interleaved_mask_bytes:shuffle_mask_bytes);
    return _mm256_cmpeq_epi8(_mm256_and_si256( select_bits, bits), select_bits);
}

template<bool doubledbits=false>
inline __attribute__((always_inline)) unsigned short compress_epi16_boolean128(__m128i b) {
    if (doubledbits) {
        return _mm_movemask_epi8(b);
    }
    if ( pext_support ) {
        return _pext_u32(_mm_movemask_epi8(b), 0x5555);
    } else {
        return _mm_movemask_epi8(_mm_packs_epi16(b, _mm_setzero_si128()));
    }
}

template<bool doubledbits=false>
inline __attribute__((always_inline)) unsigned int compress_epi16_boolean(__m256i b) {
    if (doubledbits) {
        return _mm256_movemask_epi8(b);
    }
    if ( pext_support ) {
        return _pext_u32(_mm256_movemask_epi8(b),0x55555555);
    } else {
        return _mm_movemask_epi8(_mm_packs_epi16(_mm256_castsi256_si128(b), _mm256_extracti128_si256(b,1)));
    }
}

template<bool doubledbits=false>
inline __attribute__((always_inline)) unsigned long long compress_epi16_boolean(__m256i b1, __m256i b2) {
    if (doubledbits) {
        return (((unsigned long long)_mm256_movemask_epi8(b2))<<32) | _mm256_movemask_epi8(b1);
    }
    return (unsigned int)_mm256_movemask_epi8(_mm256_permute4x64_epi64(_mm256_packs_epi16(b1,b2), 0xD8));
}

inline __attribute__((always_inline))__m256i and_unless(__m256i a, __m256i b, __m256i bcond) {
    return _mm256_and_si256( a, _mm256_or_si256( b, bcond) );
}

inline __attribute__((always_inline))__m256i and_unless(__m256i a, unsigned short b, __m256i bcond) {
    return and_unless(a, _mm256_set1_epi16(b), bcond);
}

inline __attribute__((always_inline))__m256i andnot_if(__m256i a, __m256i b, __m256i bcond) {
    return _mm256_andnot_si256( _mm256_and_si256( b, bcond), a );
}

// combine an epi16 boolean mask and a 16-bit mask to a 16/32-bit mask.
// Two pathways:
// 1. compress the wide boolean, then combine. (BMI2 instructions required)
//    Complication: the epi16 op boolean gives two bits for each element.
//    compress the wide boolean to 2 or 1 bits as desired
//    and with the bitvector modified to doubled bits if desired
//    using template parameter doubledbits.
//    Performance: pdep/pext are expensive on AMD Zen2, but good when available elsewhere.
// 2. expand the compressed boolean, and then compress again to 1 or 2 bits
//    using template parameter doubledbits.
//    (the old way, pure AVX/AVX2)
//

template<bool doubledbits=false>
inline __attribute__((always_inline)) unsigned int and_compress_masks(__m256i a, unsigned short b) {
    if ( pext_support ) {
// path 1:
        unsigned int res = compress_epi16_boolean<doubledbits>(a);
        if (doubledbits) {
            return res & _pdep_u32(b,0x55555555);
        } else {
            return res & b;
        }
    } else {
// path 2:
        if (doubledbits) {
            return compress_epi16_boolean<doubledbits>(_mm256_and_si256(a, expand_bitvector(b)));
        } else {
            return compress_epi16_boolean<false>(a) & b;
        }
    }
}

inline void add_and_mask_all_indices(bit128_t *indices, bit128_t *mask, unsigned char i) {
	indices->u128 |= *cast2cu128(big_index_lut[i][All]) & mask->u128;
}

template<Kind kind>
inline void add_indices(bit128_t *indices, unsigned char i) {
	indices->u128 |= *cast2cu128(big_index_lut[i][kind]);
}

template<Kind kind>
inline void set_indices(bit128_t *indices, unsigned char i) {
	indices->u128 = *cast2cu128(big_index_lut[i][kind]);
}

// The pair of functions below can be used to iteratively isolate all distinct bit values
// and determine whether popcnt(X) == N is true for the input vector elements using movemask
// at each interation of interest.
// For small N, this is faster than a full popcnt.
//

// compute vec & -vec
inline __m256i get_first_lsb(__m256i vec) {
        // isolate the lsb
        return _mm256_and_si256(vec, _mm256_sub_epi16(_mm256_setzero_si256(), vec));
}

// compute vec &= ~lsb; return vec & -vec
inline __m256i andnot_get_next_lsb(__m256i lsb, __m256i &vec) {
        // remove prior lsb
        vec = _mm256_andnot_si256(lsb, vec);
        return get_first_lsb(vec);
}


inline void format_candidate_set(char *ret, unsigned short digits) {
    char buff[10][4];
    unsigned char count = 0;
    while(digits) {
        sprintf(buff[count], "%s%d,", count?"":"{", _tzcnt_u32(digits)+1);
        digits = _blsr_u32(digits);
        count++;
    }
    if ( count ) {
        buff[count-1][strlen(buff[count-1])-1] = '}';
    }
    for (int i = count; i<10; i++) {
        buff[i][0] = 0;
    }
    snprintf(ret, 32, "%s%s%s%s%s%s%s%s%s%s", buff[0], buff[1], buff[2], buff[3], buff[4],
                                    buff[5], buff[6], buff[7], buff[8], buff[9]);
}

// TriadInfo
//
class
alignas(64)
TriadInfo {
public:
    unsigned short row_triads_canonical[32];  //  27 triads, linear in canonical order, no gaps
    unsigned short row_triads[36];            //  27 triads, in groups of 9 with a gap of 1
    unsigned short col_triads[36];            //  27 triads, in groups of 9 with a gap of 1
    unsigned short row_triads_wo_musts[36];   //  triads minus tmusts for guessing
    unsigned short col_triads_wo_musts[36];   //  triads minus tmusts for guessing
    unsigned int triads_selection[2];
};

class GridState;

// aggregated data that will be shared with SolverData
class CommonData {
public:
    unsigned short &current_entered_count;
    int &unique_check_mode;
    bool &check_back;
    int &line;
    CommonData(unsigned short &cec, int &ucm, bool &cb, int &li):
               current_entered_count(cec),
               unique_check_mode(ucm),
               check_back(cb),
               line(li) {};
};

// Solver sharable data - used by some algorithms and make_guess
class SolverData {
private:

    bit128_t candidate_bits_by_value[9];
    __m256i *boxes_as_cols;   // stashed in the next grid_state
public:
    BandPlacements bandplacements[9];
    bool cbbvsValid;

    unsigned char  guess_hint_score=0;  // general score of this hint index
    unsigned char  guess_hint_index;
    unsigned short guess_hint_digit;

// control info for stageing
//
    unsigned char update_cycles;
    bool update9;
// stageing areas:
// 96 bytes for digits 0-7 and the 81 bits for digit 8
// note that for efficiency, the byte groups are interleaved:
// for each 32 bytes, each group of 8 bytes is arranged in the order: 0, 2, 1, 3
// In particular byte 81 (position 1 of the incomplete final group 5) is at position 64+9=73
// [ to accommodate AVX2 unpack/pack interleaving ]
    alignas(64) unsigned char update_stage_bytes[96] {};
    bit128_t update_stage_9_bits {};
private:
    GridState* & grid_state;  // reference to the pointer, not the instance!!
    CommonData &commonData;

public:
    Counters &counters;

    inline SolverData(GridState * &grid_state, CommonData &commonData, Counters &counters): grid_state(grid_state), commonData(commonData), counters(counters) {}

    inline void reset() {
        cbbvsValid = false;
        guess_hint_digit = 0;
        guess_hint_score = 0;
        boxes_as_cols = 0;
    }

    inline void resetStage() {
        memset(update_stage_bytes, 0, 96);
        update_stage_9_bits.u128 = 0;
        update9 = false;
        update_cycles = 0;
    }

    template<Verbosity verbose>
    inline bool stage_cell_resolution( unsigned short e_digit, unsigned int e_i, const char *msg = "" );

    template<Verbosity verbose>
    inline bool update_candidates() {
        return update_candidates<verbose>(update_stage_9_bits, update_stage_bytes);
    };

    // same, but using externally supplied data
    template<Verbosity verbose>
    inline bool update_candidates(bit128_t &, unsigned char * );

    // Prepare per-digit candidate bit masks
    // These bit masks are leveraged for the single digit algorithm
    inline bit128_t *getCbbvs(unsigned short *candidates) {
        if ( !cbbvsValid ) {
            // compute for each digit a bit mask for the candidates:
            unsigned int *mskp = &candidate_bits_by_value[8].u32[0];
            for (unsigned char i = 0; i < 96; i += 32, mskp += 9*4+1 ) {
                __m256i ld1 = *(__m256i*) &candidates[i];
                __m256i ld2 = *(__m256i*) &candidates[i+16];
                // one off for digit 9
                __m256i c = _mm256_permute4x64_epi64(_mm256_packus_epi16(_mm256_srli_epi16(ld1,1), _mm256_srli_epi16(ld2,1)), 0xD8);
                *mskp = _mm256_movemask_epi8(c);
                mskp -= 4;
                c = _mm256_permute4x64_epi64(_mm256_packus_epi16(_mm256_and_si256(ld1, maskff), _mm256_and_si256(ld2, maskff)), 0xD8);
                for (unsigned char dgt = 8; dgt > 0; dgt--, mskp -= 4) {
                    *mskp = _mm256_movemask_epi8(c);
                    c = _mm256_slli_epi16(c,1);
                }
            }
            // clean up 47 extra bits
            for (unsigned char dgt = 0; dgt < 9; dgt++) {
                candidate_bits_by_value[dgt].u64[1] &= 0x1ffff;
                // cannot eliminate locked cells here, as these are essential to test/filter conjugate candidates
                // ==> candidate_bits_by_value[dgt].u128   &= grid_state->unlocked.u128;
            }
            cbbvsValid = true;
        }
        return candidate_bits_by_value;
    }

};

#ifdef OPT_DGT
// parameters:
// dgt_bits: all bits for the respective digit (o)
// cleaned_bits: all bits for the respective digit being cleaned (X)
//
inline void show_dgt_placement(bit128_t &dgt_bits, bit128_t &cleaned_bits, const char *msg="") {
    printf("%s:\n", msg);
    for ( int i_=0; i_<81; i_++ ) {
        if ( cleaned_bits.check_indexbit(i_) ) {
            printf("X");
        } else if ( dgt_bits.check_indexbit(i_) ) {
            printf("o");
        } else {
            printf("-");
        }
        if ( i_%9 == 8 ) {
            printf("\n");
        } else if ( i_%3 == 2 ) {
            printf(" ");
        }
    }
}
#endif

// The maximum levels of guesses is given by GRIDSTATE_MAX.
// On average, the number of levels is pretty low, but we want to be 'reasonably' sure
// that we don't bust the envelope.
// The other 'property' of the GRIDSTATE_MAX constant is to keep the L2 cache happy.
// 34 is a good choice either way.
//
#define	GRIDSTATE_MAX 34

// GridState encapsulates the current state of the solver, specifically for the
// purpose of guessing and back tracking.
//
// An array of size GRIDSTATE_MAX is used and stackpointer is incremented when a
// guess is made and decremented when back tracking.
//
// To match cache line boundaries, align on 64 bytes.
//
class __attribute__ ((aligned(64))) GridState
{
public:
    unsigned short candidates[81];    // which digits can go in this cell? Set bits correspond to possible digits
    short stackpointer;               // this-1 == last grid state before a guess was made, used for backtracking
    unsigned int triads_unlocked[2];  // unlocked row and col triads (#candidates >3), 27 bits each
    unsigned int multiple_solutions_exist; // indicator for multiple solutions in this grid_state
                                      // aligned on 16 bytes
    bit128_t unlocked;                // for keeping track of which cells still need to be resolved. Set bits correspond to cells that still have multiple possibilities
    bit128_t updated;                 // for keeping track of which cell's candidates may have been changed since last time we looked for naked sets. Set bits correspond to changed candidates in these cells
                                      // 'updated' contains 'hidden' 9 bits at offset 96 to indicate
                                      // that the general digit algorithm has run on that digit,
                                      // and won't need to be run again to achieve maximum performance.
    bit128_t set23_found[3];          // for keeping track of found sets of size 2 and 3

// GridState is normally copied for recursion
// initialize the starting state including the puzzle.

// initialize_and_stage
// Phase 1 (this method) fills a staging area with the presets.
// Phase 2 is executed using SolverData.update_candidates
//
template<Verbosity verbose>
inline void initialize_and_stage(char gridin[81], Counters &counters) {

    // this[1] is used a scratch area for:
    bit128_t &digit_bits9 = this[1].unlocked;
    unsigned char *stagebytes = (unsigned char *) this[1].candidates;
    unlocked.u64[1] = 0;

    multiple_solutions_exist = 0;

    triads_unlocked[0] = triads_unlocked[1] = 0x1ffLL | (0x1ffLL<<10) | (0x1ffLL<<20);
//    set23_found[0] = set23_found[1] = set23_found[2] = {__int128 {0}};

    // set this.unlocked
    for (unsigned int i=0; i<64; i +=32 ) {
        __m256i in = *(__m256i_u*)&gridin[i];
        unlocked.u32[i>>5] = _mm256_movemask_epi8(_mm256_cmpgt_epi8(dgt1, in));
    }
    __m128i in = *(__m128i_u*)&gridin[64];
    unlocked.u16[4] = _mm_movemask_epi8(_mm_cmpgt_epi8(_mm256_castsi256_si128(dgt1),in));

    if ( gridin[80] <= '0' ) {
        unlocked.u16[5] = 1;
    }
//    updated = unlocked;

    bit128_t digit_bits[9] {};

    bit128_t locked = { .u128 = ~unlocked.u128 };
    locked.u64[1] &= 0x1ffff;

    // update global preset count
    if ( verbose != VNone ) {
        counters.preset_count += locked.popcount();
    }

    for ( unsigned int i=0; i<80; i+=16 ) {
         *(__m256i *)(candidates+i) = mask1ff;
    }
    candidates[80] = 0x1ff;

    // Grouping the updates by digit beats other methods for number of clues >17.
    // calculate the masks for each digit from the place and value of the clues:
    unsigned char off = 0;
    for ( unsigned int i=0; i<2; i++) {
        unsigned long long lkd = locked.u64[i];
        while (lkd) {
            int dix = tzcnt_and_mask(lkd)+off;
            int dgt = gridin[dix] - 49;
            digit_bits[dgt].u128 = digit_bits[dgt].u128 | *cast2cu128(big_index_lut[dix][All]);
            candidates[dix] = 1<<dgt;
        }
        off = 64;
    }
    // update candidates and process the 9 masks for each chunk of the candidates:
    for ( unsigned int i=0; i<96; i += 32) {
        __m256i bits1_8 = _mm256_setzero_si256();
        __m256i bits1_8_2 = _mm256_setzero_si256();
        __m256i dgt_msk = _mm256_set1_epi8(1);
        __m256i dgt_msk_2 = _mm256_set1_epi8(2);

        for ( unsigned char dgt=0; dgt<8; dgt += 2) {
            // load 32 digit bits:
            bits1_8 = _mm256_or_si256(bits1_8, _mm256_and_si256(expand_bitvector_epi8<true>(digit_bits[dgt].u32[i>>5]), dgt_msk));
            bits1_8_2 = _mm256_or_si256(bits1_8_2, _mm256_and_si256(expand_bitvector_epi8<true>(digit_bits[dgt+1].u32[i>>5]),  dgt_msk_2));
            dgt_msk = _mm256_slli_epi16(dgt_msk, 2);
            dgt_msk_2 = _mm256_slli_epi16(dgt_msk_2, 2);
        }
        *(__m256i *)(stagebytes+i) = _mm256_or_si256(bits1_8, bits1_8_2);
    }
    digit_bits9 = digit_bits[8];
}

// Normally digits are entered by a 'goto enter;'.
// enter_digit is not used in that case.
// Only make_guess uses this member function.

protected:
template<Verbosity verbose=VNone>
inline __attribute__((always_inline)) void enter_digit( unsigned short digit, unsigned int i) {
    // lock this cell and and remove this digit from the candidates in this row, column and box

    bit128_t to_update;
    if ( verbose == VDebug ) {
        printf(" %x at %s\n", _tzcnt_u32(digit)+1, cl2txt[i]);
    }
#ifndef NDEBUG
    if ( (_popcnt_u16(digit) != 1 || digit==0) && warnings != 0 ) {
        printf("error in enter_digit: %x\n", digit);
    }
#endif

    // Vetted for best performance
    if ( i < 64 ) {
        _btr_u64(unlocked.u64[0], i);
    } else {
        _btr_u64(unlocked.u64[1], i-64);
    }

    candidates[i] = digit;

    set_indices<All>(&to_update, i);

//    updated.u128 |= to_update.u128;
    __m256i mask = _mm256_set1_epi16(~digit);
    for (unsigned char j = 0; j < 80; j += 16) {
        unsigned short m = to_update.u16[j>>4];
        __m256i c = _mm256_load_si256((__m256i*) &candidates[j]);
        // expand ~m (locked) to boolean vector
        __m256i mlocked = expand_bitvector(~m);
        // apply mask (remove bit), preserving the locked cells
        c = and_unless(c, mask, mlocked);
        _mm256_store_si256((__m256i*) &candidates[j], c);
    }
    if ((to_update.u16[5] & 1) != 0) {
        candidates[80] &= ~digit;
    }
}

public:
// this method removes candidates from the board according to a specified location pattern.
template<bool remove=true>
inline void mask_candidates(bit128_t &to_update, unsigned short vmask) {
    __m256i mask = _mm256_set1_epi16(remove?~vmask:vmask);
    to_update.u128 = to_update & unlocked;
    for (unsigned char j = 0; j < 80; j += 16) {
        unsigned short m = to_update.u16[j>>4];
        __m256i c = _mm256_load_si256((__m256i*) &candidates[j]);
        // expand ~m (locked) to boolean vector
        __m256i mlocked = expand_bitvector(~m);
        // apply mask (remove bit), preserving the locked cells
        c = and_unless(c, mask, mlocked);
        _mm256_store_si256((__m256i*) &candidates[j], c);
    }
    if ( to_update.check_indexbit(80) ) {
        candidates[80] &= remove?~vmask:vmask;
    }
}

// this form of make_guess receives additional information to make a more efficient guess.
// Efficiency for a guess is measured in terms of cells resolved until a new guess needs to be made.
// In this form, the efficiency is provided via:
// - identifying a suitable triad such that it will become resolved on both branches of the guess.
// - there are exactly 2 candidates that are connected to the remaining cells of the row/col
//   and the box and either branch of the guess will thus impact those sections.
// - the operation is typically more balanced, in that both branches will provide similar
//   efficiencies (at least on average).
//
template<Verbosity verbose>
inline GridState* make_guess(TriadInfo &triad_info, bit128_t &bivalues, SolverData &solverData) {
    // Make a guess for a triad with 4 candidate values that has 2 candidates that are not
    // constrained to the triad (not in 'tmust') and has at least 2 or more unresolved cells.
    // If we cannot obtain such a triad, fall back to make_guess().
    // With such a triad found, select one of the 2 identified candidates
    // to eliminate in the new GridState, issue the debug info and proceed.
    // Save the current GridState to back track to and eliminate the other candidate from it.

    unsigned char tpos;    // grid cell index of triad start
    unsigned char inc;     // increment for iterating triad cells
    unsigned short *wo_musts;  // pointer to triads' 'without must' candidates

    unsigned int type = 0;    // row == 0, col = 1
    for ( ; type<2; type++ ) {
        unsigned long long totest = triad_info.triads_selection[type];
        wo_musts  = type==0?triad_info.row_triads_wo_musts:triad_info.col_triads_wo_musts;
        inc  = (type<<3) + 1;  // 1 for rows, 9 for cols
        while (totest) {
            int ti = tzcnt_and_mask(totest);
            int can_ti = ti-ti/10;   // 'canonical' triad index
            if ( _popcnt_u16 (wo_musts[ti]) == 2 ) {
                // get and check the unlocked indexbits for the triad
                unsigned int b;
                if ( type == 0 ) {
                    tpos = row_triad_canonical_map[can_ti]*3;
                    b = unlocked.get_indexbits(tpos, 3);
                } else {
                    tpos = col_canonical_triad_pos[can_ti];
                    b = unlocked.get_indexbits(tpos, 19) & 0x40201;
                }
                if ( _popcnt32(b) < 2 ) {
                    continue;
                }
                // found the right candidate
                wo_musts += ti;
                goto found;
            }
        }
    }

    // if no suitable triad found, find a suitable bi-value.
    return make_guess<verbose>(bivalues, solverData);
found:
    // update the current and the new grid_state with their respective candidate to delete
    unsigned short select_cand = 0x8000 >> _lzcnt_u16(*wo_musts);

    // Create a copy of the state of the grid to make back tracking possible
    GridState* new_grid_state = this+1;
    if ( stackpointer >= GRIDSTATE_MAX-2 ) {
        fprintf(stderr, "Error: no GridState struct availabe\n");
        exit(0);
    }
    memcpy(new_grid_state, this, sizeof(GridState));
    new_grid_state->stackpointer++;

    unsigned short other_cand  = *wo_musts & ~select_cand;
    unsigned int off = tpos;

    // Update candidates
    for ( unsigned char k=0; k<3; k++, tpos += inc) {
        new_grid_state->candidates[tpos] &= ~select_cand;
        candidates[tpos] &= ~other_cand;
    }

    if (type == 0 ) {
//        updated.set_indexbits(0x7,off,3);
//        new_grid_state->updated.set_indexbits(7,off,3);
    } else {
//        updated.set_indexbits(0x40201,off,19);
//        new_grid_state->updated.set_indexbits(0x40201,off,19);
    }

    if ( verbose == VDebug ) {
        printf("guess at level >%d< - new level >%d<\n", stackpointer, new_grid_state->stackpointer);
        printf("guess remove {%d} from %s triad at %s\n",
               1+_tzcnt_u32(select_cand), type==0?"row":"col", cl2txt[off]);
    }
    if ( verbose != VNone ) {
        char gridout[82];
        if ( debug > 1 ) {
            for (unsigned char j = 0; j < 81; ++j) {
                if ( unlocked.check_indexbit(j) ) {
                    gridout[j] = '0';
                } else {
                    gridout[j] = 49+_tzcnt_u32(candidates[j]);
                }
            }
            printf("guess at %s\nsaved grid_state level >%d<: %.81s\n",
                   cl2txt[off], stackpointer, gridout);
        }
        if ( debug ) {
            printf("saved state for level %d: remove {%d} from %s triad at %s\n",
                   stackpointer, 1+_tzcnt_u32(other_cand), type==0?"row":"col", cl2txt[off]);
        }
        if ( debug > 1 ) {
            unsigned short *candidates = new_grid_state->candidates;
            for (unsigned char j = 0; j < 81; ++j) {
                if ( new_grid_state->unlocked.check_indexbit(j) ) {
                    gridout[j] = '0';
                } else {
                    gridout[j] = 49+_tzcnt_u32(candidates[j]);
                }
            }
            printf("grid_state at level >%d< now: %.81s\n",
                   new_grid_state->stackpointer, gridout);
        }
    }
    solverData.counters.guesses++;

    return new_grid_state;
}

// this version of make_guess is the simplest and original form of making a guess.
//
template<Verbosity verbose>
inline GridState* make_guess(bit128_t &bivalues, SolverData &solverData) {
    // Find a cell with the least candidates.
    // For bivalues, build a score and once done, keep the highest scoring bivalue.
    // If there are no bivalues, score trivalues.
    // Pick the candidate with the highest value as the guess.
    // Save the current grid state (with the chosen candidate eliminated) for tracking back.

    // Find the cell with fewest possible candidates
    unsigned char guess_index = 0;
    unsigned char best_index = 0xff;
    short best_score;
    unsigned short cands;
    unsigned char t = bivalues.u64[0]==0 ? 1 : 0;
    unsigned short digit;
    unsigned short search_digit;
    unsigned short best_digit = 0;
    unsigned long long bivals = bivalues.u64[t];

    // leverage any guess hints
    if ( solverData.guess_hint_digit != 0 ) {
        return make_guess<verbose>(solverData.guess_hint_index, solverData.guess_hint_digit, solverData.counters);
    }

    if ( bivals ) {
        unsigned char cnt = 0;
        best_score = -1;
        short score = 0;
        while ( bivals ) {
            unsigned char search_index = tzcnt_and_mask(bivals) + (t<<6);
            bit128_t bv2 = { .u128= bivalues & *(bit128_t*)&big_index_lut[search_index][All][0] };
            search_digit = 0;
            score = 0;
            cands = candidates[search_index];
            bool malus = false;
            while ( bv2 ) {
                unsigned short check_cands = candidates[tzcnt_and_mask(bv2)];
                if ( (cands & check_cands) ) {
                    score++;
                    if ( cands == check_cands ) {
                        malus = true;
                    } else {
                       search_digit |= cands & check_cands;
                    }
                }
            }
            if ( malus ) {
                score --;
                if ( !search_digit ) {
                    search_digit = cands;
                }
            }
            if ( score >= guess_score_threshold ) {
                best_index = search_index;
                best_digit = search_digit;
                break;
            }
            if ( score > best_score ) {
                best_index = search_index;
                best_digit = search_digit;
                best_score = score;
            }
            if ( cnt++ >= 8 ) {
                break;
            }
        }

        guess_index = best_index;
        if ( best_digit == 0 ) {
            best_digit = candidates[guess_index];
        }
        digit = 0x8000 >> _lzcnt_u16(best_digit); 
        return make_guess<verbose>(guess_index, digit, solverData.counters);
    }

    if ( verbose != VNone ) {
        solverData.counters.no_bivals_count++;
    }

        // save the box_perp 'row's for later, at the price of 9 __m256i... 
        // the save location is the next gridstate, where we can find it later
        // for searching for conjugatge pairs (just before they are overwritten there
        // by the new gridstate):

        __m256i *box_perp_rows = (__m256i *) (this+1);

    {
// below provide a replacement for a slow data load loop:
//        unsigned short *candp = candidates;
//	        for ( int k=0; k<9; candp += box_perp_ind_incr[k++]) {
//        box_perp_rows[k] = _mm256_setr_epi16(candp[0], candp[3],candp[6],
//                                             candp[27], candp[30],candp[33],
//                                             candp[54], candp[57],candp[60], 0, 0, 0, 0, 0, 0, 0);
        __m256i tmp_align, tmp_row;
        __m128i band2;
        __m128i tmp_1, tmp_3, tmp_align2;
        unsigned short *inp=candidates;
        //     [6,0], [6,3],  [6,6]
        band2 = *(__m128i_u*)(inp+27);
        tmp_align2 = *(__m128i_u*)(inp+27+8);
        tmp_1 = *(__m128i_u*)inp;
        tmp_3 = *(__m128i_u*)(inp+54);
        // first load the rows/cells that aren't laterally or vertically shifted (for now)
        //     [0,0], [0,3], [0,6] - [3,0], [3,3], [3,6]
        // --> [0,0], [0,3], [0,6] - [3,1], [3,4], [3,7]
        tmp_row = _mm256_set_m128i(tmp_3, tmp_1);
        tmp_align = _mm256_loadu2_m128i((__m128i*)(inp+54+8),(__m128i*)(inp+8));
        box_perp_rows[0] = _mm256_set_m128i(tmp_3,_mm_blend_epi16(tmp_1, _mm_slli_si128(band2,2), 0x92));
        
        //     [0,1], [0,4], [0,7] - [3,1], [3,4], [3,7]
        // --> [1,0], [1,3], [1,6] - [4,1], [4,4], [4,7] (update to destination rows)
        box_perp_rows[1] = _mm256_blend_epi16(_mm256_srli_si256(tmp_row,2), _mm256_zextsi128_si256(band2), 0x92);
        //     [0,2], [0,5], -[0,8] - [3,2], [3,5], -[3,8]
        // --> [2,0], [2,3], +[2,6] - [5,1], [5,4], +[5,7] (update to destination rows)
        box_perp_rows[2] = _mm256_blend_epi16(_mm256_alignr_epi8(tmp_align,tmp_row,4),_mm256_zextsi128_si256(_mm_alignr_epi8(tmp_align2,band2,2)), 0x92);

        band2 = *(__m128i_u*)(inp+27+9);
        tmp_align2 = *(__m128i_u*)(inp+27+9+8);
        tmp_1 = *(__m128i_u*)(inp+9);
        tmp_3 = *(__m128i_u*)(inp+54+9);
        //     [1,0], [1,3], [1,6] - [4,0], [4,3], [4,6]
        // --> [3,0], [3,3], [3,6] - [3,1], [3,4], [3,7] (update to destination rows)
        tmp_row = _mm256_set_m128i(tmp_3, tmp_1);
        tmp_align = _mm256_loadu2_m128i((__m128i*)(inp+54+9+8),(__m128i*)(inp+9+8));
        box_perp_rows[3] = _mm256_set_m128i(tmp_3, _mm_blend_epi16(tmp_1, _mm_slli_si128(band2,2), 0x92));
        //     [1,1], [1,4], [1,7] - [4,1], [4,4], [4,7]
        // --> [4,0], [4,3], [4,6] - [4,1], [4,4], [4,7]
        box_perp_rows[4] = _mm256_blend_epi16(_mm256_srli_si256(tmp_row,2), _mm256_zextsi128_si256(band2), 0x92);
        //     [1,2], [1,5], -[1,8] - [4,2], [4,5], -[4,8]
        // --> [5,0], [5,3],  [5,6] - [5,1], [5,4], +[5,7]
        box_perp_rows[5] = _mm256_blend_epi16(_mm256_alignr_epi8(tmp_align,tmp_row,4),_mm256_zextsi128_si256(_mm_alignr_epi8(tmp_align2,band2,2)), 0x92);

        band2 = *(__m128i_u*)(inp+27+18);
        tmp_align2 = *(__m128i_u*)(inp+27+18+8);
        tmp_1 = *(__m128i_u*)(inp+18);
        tmp_3 = *(__m128i_u*)(inp+54+18);
        //     [2,0], [2,3],  [2,6] - [5,0], [5,3],  [5,6]
        // --> [6,2], [6,5], +[6,8] - [6,1], [6,4],  [6,7] (update to destination rows)
        tmp_row = _mm256_set_m128i(tmp_3, tmp_1);
        tmp_align = _mm256_loadu2_m128i((__m128i*)(inp+54+18+8),(__m128i*)(inp+18+8));
        box_perp_rows[6] = _mm256_set_m128i(tmp_3, _mm_blend_epi16(tmp_1, _mm_slli_si128(band2,2), 0x92));
        //     [2,1], [2,4],  [2,7] - [5,1], [5,4], [5,7]
        // --> [7,2], [7,5], +[8,8] - [7,1], [7,4], [7,7] (update to destination rows)
        box_perp_rows[7] = _mm256_blend_epi16(_mm256_srli_si256(tmp_row,2), _mm256_zextsi128_si256(band2), 0x92);
        //     [2,2], [2,5], -[2,8] - [5,2], [5,5], -[5,8]
        // --> [8,0], [8,3], +[8,6] - [8,1], [8,4], +[8,7]
        box_perp_rows[8] = _mm256_blend_epi16(_mm256_alignr_epi8(tmp_align,tmp_row,4),_mm256_zextsi128_si256(_mm_alignr_epi8(tmp_align2,band2,2)), 0x92);

        box_perp_rows[0] = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(box_perp_rows[0], shuf), shuf8x32);
        box_perp_rows[1] = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(box_perp_rows[1], shuf), shuf8x32);
        box_perp_rows[2] = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(box_perp_rows[2], shuf), shuf8x32);
        box_perp_rows[3] = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(box_perp_rows[3], shuf), shuf8x32);
        box_perp_rows[4] = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(box_perp_rows[4], shuf), shuf8x32);
        box_perp_rows[5] = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(box_perp_rows[5], shuf), shuf8x32);
        box_perp_rows[6] = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(box_perp_rows[6], shuf), shuf8x32);
        box_perp_rows[7] = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(box_perp_rows[7], shuf), shuf8x32);
        box_perp_rows[8] = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(box_perp_rows[8], shuf), shuf8x32);
    }
        __m256i *box_cols = (__m256i *)(this+1);

        // fudged and sideways carry-save adder
        __m256i accu[3];
        __m256i tmp, cpairs;
        __m256i carry = box_cols[1];
        accu[0] = _mm256_xor_si256(box_cols[0], carry);
        accu[1] = _mm256_and_si256(box_cols[0], carry);
        accu[2] = _mm256_setzero_si256();
        for (unsigned int i=2; i<9; i++ ) {
           carry   = box_cols[i];
           tmp     = _mm256_xor_si256(accu[0], carry);
           carry   = _mm256_and_si256(accu[0], carry);
           accu[0] = tmp;
           tmp     = _mm256_xor_si256(accu[1], carry);
           carry   = _mm256_and_si256(accu[1], carry);
           accu[1] = tmp;
           accu[2] = _mm256_or_si256(carry, accu[2]);  // value > 2
        }
        cpairs = _mm256_andnot_si256(accu[2],_mm256_andnot_si256(accu[0],accu[1]));

        if ( !_mm256_testz_si256 ( cpairs, cpairs) ) {
            // cpairs contains, for each box, the digits that are conjugate pairs
            // (i.e. bilocated digits) in this box.

            unsigned char boxi = 0;
            for ( ; boxi < 9; boxi++ ) {
                // extract the boxes conjugate pair digits
                unsigned int dgtit = ((v16us)cpairs)[boxi];
                if ( dgtit == 0 ) {
                    continue;
                }
                unsigned short *boxp = candidates+box_start_by_boxindex[boxi];
                unsigned char best_score = 0;
                unsigned char score = 0;
                unsigned char score_index;
                unsigned char dgti;
                __m256i box = _mm256_setr_epi64x(*(unsigned long long *)(boxp),
                                                 *(unsigned long long *)(boxp+9),
                                                 *(unsigned long long *)(boxp+18), 0);
                unsigned int mskByStart[9] {};
                unsigned short dgtByStart[9] {};
                unsigned char pos1, pos2;
                unsigned short dgt;
                while ( dgtit ) {
                    dgt  = __blsi_u32(dgtit);
                    dgti = tzcnt_and_mask(dgtit);
                    score = 0;
                    unsigned int mskpos = _mm256_movemask_epi8(_mm256_cmpgt_epi16(_mm256_and_si256(box,_mm256_set1_epi16(dgt)), _mm256_setzero_si256())) & 0x3f3f3f;
                    score_index = box_start_by_boxindex[boxi];
                    unsigned char lpos1 = _tzcnt_u32(mskpos)>>1;
                    pos1 = group4x3offsets[lpos1];
                    lpos1 = lpos1 - lpos1/4;
                    pos2 = group4x3offsets[(62-_lzcnt_u64(mskpos))>>1];
                    // check for hidden pair
                    if ( mskByStart[lpos1] == mskpos ) {
                        // this is a hidden pair, simply guess dgti in pos1
                        // this will resolve at minimum 2 cells
                        unsigned short cands = dgtByStart[lpos1] | dgt; 
                        candidates[score_index+pos1] &= cands;
                        candidates[score_index+pos2] &= cands;
                        if ( verbose == VDebug ) {
                            char ret[32];
                            format_candidate_set(ret, cands);
                            printf("hidden pair (box): %-7s %s\n", ret, cl2txt[score_index+pos1]);
                        }
                        return make_guess<verbose>(score_index+pos1, dgt, solverData.counters);
                    }
                    // score this conjugate pair
                    mskByStart[lpos1] = mskpos;
                    dgtByStart[lpos1] = dgt;
                    if ( (viewbits_by_i[pos1].u32 & viewbits_by_i[pos2].u32 & 0x300ffff) == 0 ) {
                        score++;
                    }
                    score_index = box_start_by_boxindex[boxi];
                    if ( _popcnt_u16(candidates[score_index+pos1]) <= 3 ) {
                        score++;
                        score_index += pos2;
                    } else {
                        if ( _popcnt_u16(candidates[score_index+pos2]) <= 3 ) {
                            score++;
                        }
                        score_index += pos1;
                    }
                    if ( score > best_score ) {
                        best_index = score_index;
                        best_score = score;
                        best_digit = 1<<dgti;
                    }
                    if ( best_score >= 2 ) {
                        break;
                    }
                }
                if ( best_score >= 2 ) {
                    break;
                }
            }

            if ( best_index != 0xff ) {
                return make_guess<verbose>(best_index, best_digit, solverData.counters);
            }
        }

    // Trivalue scoring
    //
    // very unlikely except for very hard puzzles while under 27 cells solved.
    // a puzzle with no bivalues is always hard - extra effort and solving algos are
    // appropriate.
    //

        unsigned char cnt;
        unsigned char best_cnt = 16;
        unsigned char i_rel;
        unsigned long long to_visit = unlocked.u64[0];
        while ( best_cnt > 3 && to_visit != 0 ) {
            i_rel = tzcnt_and_mask(to_visit);
            cnt = _popcnt_u16(candidates[i_rel]);
            if (cnt < best_cnt) {
                best_cnt = cnt;
                guess_index = i_rel;
            }
        }

        to_visit = unlocked.u32[2];
        while ( best_cnt > 3 && to_visit != 0 ) {
            i_rel = tzcnt_and_mask(to_visit) + 64;
            cnt = _popcnt_u16(candidates[i_rel]);
            if (cnt < best_cnt) {
                best_cnt = cnt;
                guess_index = i_rel;
            }
        }
        // Find the first candidate in this cell (lsb set)
        // Note: using tzcnt would be equally valid; this pick is historical
        digit = 0x8000 >> _lzcnt_u16(candidates[guess_index]);
        return make_guess<verbose>(guess_index, digit, solverData.counters);
}

// this version of make_guess takes a cell index and digit for the guess
//
template<Verbosity verbose>
inline GridState* make_guess(unsigned char guess_index, unsigned short digit, Counters &counters ) {
    // Create a copy of the state of the grid to make back tracking possible
    GridState* new_grid_state = this+1;
    if ( stackpointer >= GRIDSTATE_MAX-2 ) {
        fprintf(stderr, "Error: no GridState object availabe\n");
        exit(0);
    }
    memcpy(new_grid_state, this, sizeof(GridState));
    new_grid_state->stackpointer++;

    // Remove the guessed candidate from the old grid
    // when we get back here to the old grid, we know the guess was wrong
    candidates[guess_index] &= ~digit;

//    updated.set_indexbit(guess_index);

    if ( verbose == VDebug && (debug > 1) ) {
        char gridout[82];
        for (unsigned char j = 0; j < 81; ++j) {
            if ( (candidates[j] & (candidates[j]-1)) ) {
                gridout[j] = '0';
            } else {
                gridout[j] = 49+_tzcnt_u32(candidates[j]);
            }
        }
        printf("guess at %s\nsaved grid_state level >%d<: %.81s\n",
               cl2txt[guess_index], stackpointer, gridout);
    }

    // Update candidates
    if ( verbose == VDebug ) {
        printf("guess at level >%d< - new level >%d<\nguess", stackpointer, new_grid_state->stackpointer);
    }

    new_grid_state->enter_digit<verbose>( digit, guess_index);
    counters.guesses++;

    if ( verbose == VDebug && (debug > 1) ) {
        unsigned short *candidates = new_grid_state->candidates;
        char gridout[82];
        for (unsigned char j = 0; j < 81; ++j) {
            if ( (candidates[j] & (candidates[j]-1)) ) {
                gridout[j] = '0';
            } else {
                gridout[j] = 49+_tzcnt_u32(candidates[j]);
            }
        }
        printf("grid_state at level >%d< now: %.81s\n",
               new_grid_state->stackpointer, gridout);
    }
    return new_grid_state;
}
};

#if 0
// this code would complete the stageing infrastructure if we need it.
// for now, there is no speed gain associated to it.
template <Verbosity verbose>
inline bool SolverData::stage_cell_resolution( unsigned short e_digit, unsigned int e_i, const char *msg ) {

    bit128_t to_update;

    assert( e_digit != 0 );

#ifndef NDEBUG
    if ( _popcnt_u16(e_digit) != 1 ) {
        if ( warnings != 0 ) {
            printf("error in e_digit: %x\n", e_digit);
        }
    }
#endif

    // Vetted for best performance
    if ( e_i < 64) {
        _btr_u64(grid_state->unlocked.u64[0], e_i);
    } else {
        _btr_u64(grid_state->unlocked.u64[1], e_i-64);
    }

    grid_state->candidates[e_i] = e_digit;
    commonData.current_entered_count++;

    set_indices<All>(&to_update, e_i);
//    grid_state->updated.u128 |= to_update.u128;

    // check for upcoming cells being set to 0.
    if (  (e_digit == 0x100) ? update_stage_9_bits.check_indexbit(e_i)
                             : (update_stage_bytes[stage_bytes_lu[e_i]] & e_digit) ) {
        // Back track, no solutions along this path
        if ( verbose != VNone ) {
            if ( grid_state->stackpointer == 0 && commonData.unique_check_mode == 0 ) {
                if ( warnings != 0 ) {
                    printf("Line %d: [0] cell %s is 0\n", commonData.line, cl2txt[e_i]);
                }
            } else if ( debug ) {
                printf("back track - cell %s is 0\n", cl2txt[e_i]);
            }
        }
        return false;
    }

    if ( verbose == VDebug ) {
        printf("%s %x at %s\n", msg, _tzcnt_u32(e_digit)+1, cl2txt[e_i]);
    }
    if ( e_digit == 0x100 ) {
        update_stage_9_bits.u128 |= to_update.u128;
        update9 = true;
    } else {
        __m256i digitsv = _mm256_set1_epi8(e_digit);
        __m256i bytes = _mm256_and_si256(expand_bitvector_epi8<true>(to_update.u32[0]), digitsv);
        *(__m256i*)(update_stage_bytes) = _mm256_or_si256(*(__m256i*)(update_stage_bytes), bytes);
        bytes = _mm256_and_si256(expand_bitvector_epi8<true>(to_update.u32[1]), digitsv);
        *(__m256i*)(update_stage_bytes+32) = _mm256_or_si256(*(__m256i*)(update_stage_bytes+32), bytes);
        bytes = _mm256_and_si256(expand_bitvector_epi8<true>(to_update.u32[2]), digitsv);
        bytes = _mm256_or_si256(*(__m256i*)(update_stage_bytes+64), bytes);
        *(__m256i*)(update_stage_bytes+64) = _mm256_or_si256(*(__m256i*)(update_stage_bytes+64), bytes);
    }
    update_cycles = 3;
    return true;
}  // stage_cell_resolution
#endif

// update the candidates with the staged data (e.g. from initialization), while at the same time
// back-filling the stageing area with any naked singles found.
//
template<Verbosity verbose>
inline bool SolverData::update_candidates(bit128_t &update_stage_9_bits, unsigned char *update_stage_bytes) {
    // this will continue until naked singles are exhausted
    unsigned char i = 0;
    unsigned char update9_cycles = update9? 3 : 0;
    __m256i c1, c2;
    __m256i bits9;
    unsigned long long mask;
    unsigned int mask1, mask2= 0;
    unsigned int m;

    if ( update_cycles ) {
        i = 0;
        do {
            bits9 = update9 ?
                    _mm256_and_si256(expand_bitvector_epi8<true>(update_stage_9_bits.u32[i>>5]), ones_epi8) : 
                    _mm256_setzero_si256();
            c2 = *(__m256i*)(update_stage_bytes+i);
            *(__m256i*)&grid_state->candidates[i] = c1 = _mm256_andnot_si256(_mm256_unpacklo_epi8(c2,bits9), *(__m256i*) &grid_state->candidates[i]);
            c2 = _mm256_andnot_si256(_mm256_unpackhi_epi8(c2,bits9), *(__m256i*) &grid_state->candidates[i+16]);
            *(__m256i*)	&grid_state->candidates[i+16] = c2;
            // hide the mask construction inside __builtin_expect and after the check on check_back
            // unfortunately the parentheses and the assignments/instruction lists are difficult to get right and to read.
            if (__builtin_expect (commonData.check_back && (
                   (  mask1 = _mm256_movemask_epi8(_mm256_cmpeq_epi16(c1, _mm256_setzero_si256())))
               ||  ( (mask2 = _mm256_movemask_epi8(_mm256_cmpeq_epi16(c2, _mm256_setzero_si256()))), (mask2 = i<64? mask2 : mask2&0x3))), 0)) {
                // Back track, no solutions along this path
                if ( verbose != VNone ) {
                    unsigned char pos = i+(_tzcnt_u64((unsigned long long)mask1 | ((unsigned long long)mask2<<32))>>1);
                    if ( grid_state->stackpointer == 0 && commonData.unique_check_mode == 0 ) {
                        if ( warnings != 0 ) {
                            printf("Line %d: [2] cell %s is 0\n", commonData.line, cl2txt[pos]);
                        }
                    } else if ( debug ) {
                        printf("back track - cell %s is 0\n", cl2txt[pos]);
                    }
                }
                return false;
            }

            m = grid_state->unlocked.u32[i>>5];

            // test for singletons:
            c1 = _mm256_cmpeq_epi16(_mm256_and_si256(c1, _mm256_sub_epi16(c1, ones)), _mm256_setzero_si256());
            c2 = _mm256_cmpeq_epi16(_mm256_and_si256(c2, _mm256_sub_epi16(c2, ones)), _mm256_setzero_si256());
            mask = compress_epi16_boolean(c1, c2) & m;
            if ( mask ) {
                do {
                    unsigned int e_i = i+tzcnt_and_mask(mask);
                    // manually inlined from stage_cell_resolution (not a big imporevement)
                    unsigned short e_digit = grid_state->candidates[e_i];
                    bit128_t to_update;

#ifndef NDEBUG
                    if ( _popcnt_u16(e_digit) != 1 ) {
                        if ( warnings != 0 ) {
                            printf("error in e_digit: %x\n", e_digit);
                        }
                    }
#endif

                    if (e_i < 64) {
                        _btr_u64(grid_state->unlocked.u64[0], e_i);
                    } else {
                        _btr_u64(grid_state->unlocked.u64[1], e_i-64);
                    }

                    // unnecessary, we get here for naked singles only!
                    // grid_state->candidates[e_i] = e_digit;
                    commonData.current_entered_count++;

                    set_indices<All>(&to_update, e_i);
//                    grid_state->updated.u128 |= to_update.u128;

                    if (  (e_digit == 0x100) ? update_stage_9_bits.check_indexbit(e_i)
                                             : (update_stage_bytes[stage_bytes_lu[e_i]] & e_digit) ) {
                        // Back track, no solutions along this path
                        if ( verbose != VNone ) {
                            if ( grid_state->stackpointer == 0 && commonData.unique_check_mode == 0 ) {
                                if ( warnings != 0 ) {
                                    printf("Line %d: [0] cell %s is 0\n", commonData.line, cl2txt[e_i]);
                                }
                            } else if ( debug ) {
                                printf("back track - cell %s is 0\n", cl2txt[e_i]);
                            }
                        }
                        return false;
                    }

                    if ( verbose == VDebug ) {
                        printf("naked  single       %x at %s\n", _tzcnt_u32(e_digit)+1, cl2txt[e_i]);
                    }
                    if ( e_digit == 0x100 ) {
                        update_stage_9_bits.u128 |= to_update.u128;
                        update9 = true;
                        update9_cycles = 4;
                    } else {
                        __m256i digitsv = _mm256_set1_epi8(e_digit);
                        __m256i bytes = _mm256_and_si256(expand_bitvector_epi8<true>(to_update.u32[0]), digitsv);
                        *(__m256i*)(update_stage_bytes) = _mm256_or_si256(*(__m256i*)(update_stage_bytes), bytes);
                        bytes = _mm256_and_si256(expand_bitvector_epi8<true>(to_update.u32[1]), digitsv);
                        *(__m256i*)(update_stage_bytes+32) = _mm256_or_si256(*(__m256i*)(update_stage_bytes+32), bytes);
                        bytes = _mm256_and_si256(expand_bitvector_epi8<true>(to_update.u32[2]), digitsv);
                        bytes = _mm256_or_si256(*(__m256i*)(update_stage_bytes+64), bytes);
                        *(__m256i*)(update_stage_bytes+64) = _mm256_or_si256(*(__m256i*)(update_stage_bytes+64), bytes);
                    }
                    update_cycles = 4;

                } while ( mask );
            } else {
                i = (i==64)? 0: i+32;
                --update_cycles;
                if ( --update9_cycles == 0 ) {
                    update9 = false;
                }
            }
        } while ( update_cycles );
    }
    return true;
}

template <Verbosity verbose>
// Verbosity leads to 3 variants of solve to be generated whith some degree of overhead at runtime
Status solve(char gridin[81], char gridout[81], GridState stack[], int line, Counters &counters) {

    GridState *grid_state = &stack[0];
    unsigned long long *unlocked = grid_state->unlocked.u64;
    unsigned short* candidates;

    Status status;

    // borrowing the data area from the unoccupied grid_state[1]
    grid_state->initialize_and_stage<verbose>(gridin, counters);

    if ( verbose == VDebug ) {
        printf("Line %d: %.81s\n", line, gridin);
    }

    // count resolved/entered cells in combination with guess level
    // allowing to recompute data when additional cells have been resolved.
    //
    // the low byte is the real count, while
    // the high byte is increased/decreased with each guess/back track
    // init count of resolved cells to the size of the initial set
    unsigned short current_entered_count = 81 - grid_state->unlocked.popcount();

#ifdef OPT_DGT
    unsigned int next_dgt = 0;
#endif

    unsigned short last_entered_count_col_triads = 0;

    int unique_check_mode = 0;
    bool check_back = grid_state->stackpointer || thorough_check || rules != Regular;

    CommonData commonData(current_entered_count, unique_check_mode, check_back, line);
    SolverData solverData(grid_state, commonData, counters);
    solverData.update_cycles = 3;
    solverData.update9 = true;
    // using data from initialization in the unoccupied grid_state[1]
    solverData.update_candidates<verbose>( (grid_state+1)->unlocked, (unsigned char *)(grid_state+1));

    bool nonunique_reported = false;

    unsigned char no_guess_incr = 1;

    // The 'API' for code that uses the 'goto enter:' method of entering digits
    unsigned short e_digit = 0;
    unsigned int e_i = 0;

    bit128_t bivalues;

    // duplicate sequence of instructions, see under label start: 

    unlocked   = grid_state->unlocked.u64;
    candidates = grid_state->candidates;

    goto check_solved;

back:

    // Each algorithm (naked single, hidden single, naked set)
    // has its own non-solvability detecting trap door to detect if the grid is bad.
    // This section acts upon that detection and discards the current grid_state.
    //
    if (grid_state->stackpointer == 0) {
        if ( unique_check_mode ) {
            if ( verbose == VDebug ) {
                // no additional solution exists
                printf("No secondary solution found during back track\n");
            }
        } else {
            // This only happens when the puzzle is not valid
            // Bypass the verbose check...
            if ( warnings != 0 ) {
                printf("Line %d: No %ssolution found!\n", line, rules==Regular?"unique " : "");
            }
            counters.unsolved_count++;
        }
        // cleanup and return
        if ( verbose != VNone && reportstats ) {
            if ( status.unique == false ) {
                counters.non_unique_count++;
            }
        }
        if ( !unique_check_mode ) {
            // failed - just copy the input
            memcpy(gridout, gridin, 81);
        }
        return status;
    }

    current_entered_count  = (((grid_state-1)->stackpointer)<<8) | (81 - (grid_state-1)->unlocked.popcount());        // back to previous stack.

    // collect some guessing stats
    if ( verbose != VNone && reportstats ) {
        counters.digits_entered_and_retracted +=
            (_popcnt64((grid_state-1)->unlocked.u64[0] & ~grid_state->unlocked.u64[0]))
          + (_popcnt32((grid_state-1)->unlocked.u32[2] & ~grid_state->unlocked.u32[2]));
    }

    // Go back to the state when the last guess was made
    // This state had the guess removed as candidate from it's cell

    if ( verbose == VDebug ) {
        printf("back track to level >%d<\n", grid_state->stackpointer-1);
    }
    if ( verbose != VNone ) {
        counters.trackbacks++;
    }
    grid_state--;

start:

    e_digit = 0;

#ifdef OPT_DGT
    next_dgt = 0;
#endif

    // at start, set everything that depends on grid_state:
    check_back = grid_state->stackpointer || thorough_check || rules != Regular || unique_check_mode;

    unlocked   = grid_state->unlocked.u64;
    candidates = grid_state->candidates;

search:
    // find a naked single (first one will do)
    {
        __m256i c1;
        __m256i c2;
        unsigned long long mask;
        unsigned int m; 
        // no digit to enter
        for ( unsigned char i=0; i <96; i += 32 ) {
            m = ((bit128_t*)unlocked)->u32[i>>5];
            c1 = *(__m256i*) &candidates[i];
            c2 = *(__m256i*) &candidates[i+16];
            // test for 0s
            if (__builtin_expect (check_back && (mask=compress_epi16_boolean(_mm256_cmpeq_epi16(c1, _mm256_setzero_si256()), _mm256_cmpeq_epi16(c2, _mm256_setzero_si256())) & m), 0)) {
                    // Back track, no solutions along this path
                    if ( verbose != VNone ) {
                    unsigned char pos = i+_tzcnt_u64(mask);
                    if ( grid_state->stackpointer == 0 && unique_check_mode == 0 ) {
                        if ( warnings != 0 ) {
                            printf("Line %d: cell %s is 0\n", line, cl2txt[pos]);
                        }
                    } else if ( debug ) {
                        printf("back track - cell %s is 0\n", cl2txt[pos]);
                    }
                }
                goto back;
            }
            // test for singletons
            c1 = _mm256_cmpeq_epi16(_mm256_and_si256(c1, _mm256_sub_epi16(c1, ones)), _mm256_setzero_si256());
            c2 = _mm256_cmpeq_epi16(_mm256_and_si256(c2, _mm256_sub_epi16(c2, ones)), _mm256_setzero_si256());
            mask = compress_epi16_boolean(c1, c2) & m;
            if ( mask ) {
                e_i = i+_tzcnt_u64(mask);
                e_digit = candidates[e_i];
                if ( verbose == VDebug ) {
                    printf("naked  single      ");
                }
                goto enter;
            }
        }
    }
    // if no single found:
    goto hidden_search;

// Algorithm 1:
// Enter a digit into the solution by setting it as the value of cell and by
// removing the cell from the set of unlocked cells.  Update all affected cells
// by removing any candidates that have become impossible for that digit.
// For all cells check whether the cell has no candidates: back track.
// Check all cells for a single candidate.
//
enter:

    {
        // inlined flavor of enter_digit
        //
        // lock this cell and and remove this digit from the candidates in this row, column and box
        // and for good measure, detect 0s (back track) and singles.
        bit128_t to_update;

        if ( verbose == VDebug ) {
            printf(" %x at %s\n", _tzcnt_u32(e_digit)+1, cl2txt[e_i]);
        }
#ifndef NDEBUG
        if ( _popcnt_u16(e_digit) != 1 ) {
            if ( warnings != 0 ) {
                printf("error in e_digit: %x\n", e_digit);
            }
        }
#endif

        // Vetted for best performance
        if (e_i < 64) {
            _btr_u64(unlocked[0], e_i);
        } else {
            _btr_u64(unlocked[1], e_i-64);
        }
        candidates[e_i] = e_digit;
        current_entered_count++;

        set_indices<All>(&to_update, e_i);

//        grid_state->updated.u128 |= to_update.u128;

        __m256i mask_neg = _mm256_set1_epi16(e_digit);

        unsigned short dtct_j = 0;
        unsigned int dtct_m = 0;
        for (unsigned char j = 0; j < 80; j += 16) {
            __m256i c = _mm256_load_si256((__m256i*) &candidates[j]);
            // expand unlocked unsigned short to boolean vector
            __m256i munlocked = expand_bitvector(to_update.u16[j>>4]);
            // apply mask (remove bit)
            c = andnot_if(c, mask_neg, munlocked);
            _mm256_store_si256((__m256i*) &candidates[j], c);
            __m256i a = _mm256_cmpeq_epi16(_mm256_and_si256(c, _mm256_sub_epi16(c, ones)), _mm256_setzero_si256());
            // this if is only taken very occasionally, branch prediction
            if (__builtin_expect (check_back && _mm256_movemask_epi8(
                                  _mm256_cmpeq_epi16(c, _mm256_setzero_si256())
                                  ), 0)) {
                // Back track, no solutions along this path
                if ( verbose != VNone ) {
                    unsigned int mx = _mm256_movemask_epi8(_mm256_cmpeq_epi16(c, _mm256_setzero_si256()));
                    unsigned char pos = j+(_tzcnt_u32(mx)>>1);
                    if ( grid_state->stackpointer == 0 && unique_check_mode == 0 ) {
                        if ( warnings != 0 ) {
                            printf("Line %d: cell %s is 0\n", line, cl2txt[pos]);
                        }
                    } else if ( debug ) {
                        printf("back track - cell %s is 0\n", cl2txt[pos]);
                    }
                }
                e_digit=0;
                goto back;
            }
            unsigned int mask = and_compress_masks<false>(a, grid_state->unlocked.u16[j>>4]);
            if ( mask ) {
                dtct_m = mask;
                dtct_j = j;
            }
        }
        if (unlocked[1] & (1ULL << (80-64))) {
            if ((to_update.u16[5] & 1) != 0) {
                candidates[80] &= ~e_digit;
                if (__builtin_expect (candidates[80] == 0,0) ) {
                    // no solutions go back
                    if ( verbose != VNone ) {
                        if ( grid_state->stackpointer == 0 && unique_check_mode == 0 ) {
                            if ( warnings != 0 ) {
                                printf("Line %d: cell %s is 0\n", line, cl2txt[80]);
                            }
                        } else if ( debug ) {
                            printf("back track - cell %s is 0\n", cl2txt[80]);
                        }
                    }
                    goto back;
                }
            }
            if (_popcnt_u16(candidates[80]) == 1) {
                // Enter the digit and update candidates
                if ( verbose == VDebug ) {
                    printf("naked  single      ");
                }
                e_i = 80;
                e_digit = candidates[80];
                goto enter;
            }
        }
        if ( dtct_m ) {
            int idx = _tzcnt_u32(dtct_m);
            e_i = idx+dtct_j;
            e_digit = candidates[e_i];
            if ( verbose == VDebug ) {
                printf("naked  single      ");
            }
            goto enter;
        }
        e_digit = 0;
    }

check_solved:
    // The solving algorithm ends when there are no remaining unlocked cells.
    // The finishing tasks include verifying the solution and/or confirming
    // its uniqueness, if requested.
    //
    // Check if it's solved, if it ever gets solved it will be solved after looking for naked singles
    if ( ((*(__uint128_t*)unlocked) & mask_81bit) == 0) {
        // Solved it
        if ( rules == Multiple && (unique_check_mode == 1 || grid_state->multiple_solutions_exist) ) {
            if ( !nonunique_reported ) {
                if ( verbose != VNone && reportstats && warnings != 0 ) {
                    printf("Line %d: solution to puzzle is not unique\n", line);
                }
                nonunique_reported = true;
            }
            status.unique = false;
        }
        if ( verify ) {
            // quickly assert that the solution is valid
            // no cell has more than one digit set
            // all rows, columns and boxes have all digits set.

            __m256i rowx;
            __m256i colx;
            __m256i boxx;
            boxx = colx = rowx = _mm256_and_si256(_mm256_set1_epi16(0x1ff),mask9);
            __m256i uniq = _mm256_setzero_si256();

            for (unsigned char i = 0; i < 9; i++) {
                // load element i of 9 rows
                __m256i row = _mm256_set_epi16(0, 0, 0, 0, 0, 0, 0, candidates[i+72],
                              candidates[i+63], candidates[i+54], candidates[i+45], candidates[i+36], candidates[i+27], candidates[i+18], candidates[i+9], candidates[i]);
                rowx = _mm256_xor_si256(rowx,row);

                // load element i of 9 columns
                __m256i col = _mm256_and_si256(*(__m256i_u*) &candidates[i*9], mask9);
                colx = _mm256_xor_si256(colx,col);

                uniq = _mm256_or_si256(_mm256_and_si256(col, _mm256_sub_epi16(col, ones9)),uniq);

                // load element i of 9 boxes
                int bi = i%3+i/3*9; // starting in box 0
                __m256i box = _mm256_set_epi16(0, 0, 0, 0, 0, 0, 0, candidates[bi+60],
                              candidates[bi+57], candidates[bi+54], candidates[bi+33], candidates[bi+30], candidates[bi+27], candidates[bi+6], candidates[bi+3], candidates[bi]);
                boxx = _mm256_xor_si256(boxx,box);
            }

            __m256i res = _mm256_or_si256(rowx,colx);
            res = _mm256_or_si256(res, boxx);
            res = _mm256_or_si256(res, uniq);
            if ( ~_mm256_movemask_epi8(_mm256_cmpeq_epi16(res,_mm256_setzero_si256()))) {
                // verification failure
                if ( unique_check_mode == 0 ) {
                    if ( verbose != VNone ) {
                        printf("Line %d: solution to puzzle failed verification\n", line);
                    }
                    counters.unsolved_count++;
                    counters.not_verified_count++;
                } else {     // not supposed to get here
                    if ( verbose != VNone ) {
                        printf("Line %d: secondary puzzle solution failed verification\n", line);
                    }
                }
            } else  if ( verbose != VNone ) {
                if ( debug ) {
                    printf("Solution found and verified\n");
                }
                status.verified = true;
                if ( verbose != VNone && reportstats ) {
                    if ( unique_check_mode == 0 ) {
                        counters.verified_count++;
                    }
                }
            }
        }
        if ( verbose != VNone && reportstats ) {
            if ( unique_check_mode == 0 ) {
                counters.solved_count++;
            }
        }
        
        // Enter found digits into grid (unless we already had a solution)
        if ( unique_check_mode == 0 ) {
            status.solved = true;
            for (unsigned char j = 0; j < 64; j+=32) {
                __m256i t1 = _mm256_permute4x64_epi64(
                    _mm256_packus_epi16(_mm256_and_si256(*(__m256i*)&candidates[j],maskff),_mm256_and_si256(*(__m256i*)&candidates[j+16],maskff)),
                    0xD8);
                __m256i t2 = _mm256_and_si256(_mm256_srli_epi16(t1, 4),nibble_mask);
                t1 = _mm256_and_si256( t1, nibble_mask);
                t2 = _mm256_shuffle_epi8(lut_hi, t2);
                t1 = _mm256_shuffle_epi8(lut_lo, t1);
                _mm256_storeu_si256((__m256i_u*)&gridout[j], _mm256_min_epu8(t1, t2));
            }
            __m256i tmp = _mm256_and_si256(*(__m256i*)&candidates[64],maskff);
            __m128i t1 = _mm256_castsi256_si128(_mm256_packus_epi16(tmp,_mm256_permute2x128_si256(tmp,tmp,0x11)));
            __m128i t2 = _mm_and_si128(_mm_srli_epi16(t1, 4),_mm256_castsi256_si128(nibble_mask));
            t1 = _mm_and_si128( t1, _mm256_castsi256_si128(nibble_mask));
            t2 = _mm_shuffle_epi8(_mm256_castsi256_si128(lut_hi), t2);
            t1 = _mm_shuffle_epi8(_mm256_castsi256_si128(lut_lo), t1);
            _mm_storeu_si128((__m128i_u*)&gridout[64], _mm_min_epu8(t1, t2));
            gridout[80] = '1'+_tzcnt_u32(candidates[80]);
        }

        if ( verbose != VNone ) {
            counters.no_guess_cnt += no_guess_incr;
        }

        if ( grid_state->stackpointer && rules == Multiple && unique_check_mode == 0 && grid_state->multiple_solutions_exist == 0) {
            if ( verbose == VDebug ) {
                printf("Solution: %.81s\nBack track to determine uniqueness\n", gridout);
            }
            unique_check_mode = 1;
            goto back;
        }
        // otherwise uniqueness checking is complete
        if ( status.unique == false ) {
            counters.non_unique_count++;
        }
        return status;
    }

hidden_search:
    // reset solverData
    solverData.reset();

    // Algorithm 2 - Find hidden singles
    // For all sections (ie. rows/columns/boxes):
    //   for each cell C in the given section
    //      'or' all the other cells of the section
    //          check whether C contains a candidate that does not occur in the other cells
    // Back track checking:
    //    1. If the found cell C contains more than one candidate that does not
    //       appear in the other cells: back track
    //    2. If the 'or' of all the cells does not contain all digits: back track
    //
    // columns are or'ed together (i.e. all columns in parallel, while reading one row at a time,
    // leaving out the current row, and hidden column singles are thus isolated for that row.
    // For efficiency, precompute and save the or'ed rows (tails) and
    // preserve the last leading set of rows (the head).
    // To check for an invalid state of the puzzle:
    // - check the columns or'ed value to be 0x1ff.
    // - andnot with 0x1ff and then check all non-zero results to be singletons, or otherwise
    //   back track.
    //
    // All this could be applied similarly to rows and boxes.
    // But, after a bit of discovering and checking, it turns out to be more efficient
    // not to examine rows and boxes for hidden singles.
    // This sounds odd, but by slightly more frequently using the (locked set elimination)
    // triad checking, we can forgo the effort to examine rows and boxes for hidden singles.
    //
    // The only remnant of the rows checking is the creation of the row triad information.
    //
    // Algorithm 3
    // Definition: An intersection of row/box and col/box consisting of three cells
    // are called "triad" in the following.
    // The technique described in the following is also known as 
    // "locked candidates (claiming/pointing)".
    // There are 27 horizontal (row-based) triads and 27 vertical (col-based) triads.
    // For each band of three aligned boxes there are nine triads.
    //
    // Triads in their own right are significant for two reasons:
    // First, a triad that has 3 candidates is a special case of set that is easily detected.
    // The detection occurs in part 2 of Algorithm 3 by running a popcount on the collected
    // triad candidates.  The result is kept in form of a bitvector of 'unlocked triads'
    // for rows and columns.  For columns, the unlocked state is determined directly from
    // the general unlocked bit vector.  For rows, the order in which the row triads are
    // stored is not compatible with the general unlocked bit vector.  Instead, all locked
    // row triads are individually removed from their row unlocked bit vector.
    //
    // The collection of the triads data (Algorithm 3 Part 1) is intermingled with
    // that of algorithm 2 for speed.
    // Part 2 of Algorithm 3 allows to detect all fully resolved triads (sets).
    // Note that this is not normally part of the locked set solving strategy.
    // Part 3 of Algorithm 3 allows to determine which candidate value can only occur
    // in a specific triad and not in the other triads of the row/column and box.
    // Terminology:
    // - a "must" is a set of candidates for a triad containing those candidates that cannot occur
    // in any other triad of the same block or row/column.
    // - a "must not" (mustnt for short) is a set of candidates that must not occur in the given triad.
    // For starters, the complement of the candidates that occur in the triad cells is a
    // "must not" (not necessarily the most complete one).
    // Trivially, in a solved Sudoku puzzle, each "must" has three candidates and all "must not"
    // have six candidates.
    // - the "peer" triads of a given triad share either box or the row/column with the given "triad".
    // The following holds since each row/column or box has a set of candidates and the
    // triads are intersections of a given row/column with the box:
    // - given a set of three 'peered' triads for each row/column and box each triad is
    // peered with two triads of  the same row and with two triads of the same box,
    // - If a given candidate value does not occur in either of the peer pairs (peer-mustnt),
    //   this triad must contain that candidate value (must).
    // - Any candidate only occurring in a triad (must) because it doesn't occur in
    //   either its box peers or its row/column peers cannot occur in any of the pair of peers.
    // - Since 3 is the upper limit of must candidates, once these three are known from
    //   part 2 of the algorithm, these become the fixed 'must' set for the triad.
    //   The triad is then called "resolved".
    //
    // We can start with the mustnt for all triads of a band of boxes, compute the intersection
    // pairs of peer-mustnt (horizontally and vertically) and join the two results.
    // This is then a "must" of the given triad, and augmented by the must if the triad is resolved.
    // Using the peers musts, their union is a "must not" for the given triad.
    // By joining it to the orginal "must not" we obtain an equal or super set for the new "must".
    // This "must not" in turn can then be be applied to remove the excessive candidates from
    // the triad.
    //
    // Yet another side advantage is the ability to take note of triads with 2 musts and
    // 4 candidates. These represent a valuable guessing strategy and can also be used as
    // search targets for naked sets.
    // 
    // These computations are easily implemented as SIMD operations.
    //
    // To recap: parts 1 - 3 of Algorithm 3 are fast and effective - especially if the
    // hidden single search is leveraged for the initial union of the triads' candidates.
    //
    // Part 2 and 3 are invoked after Algorithm 2 fails to detect any hidden single.
    //
    // col_triads are numbered continuously per horizontal band (one vertical triad for each column)
    // and stored in three arrays of 10 ushort as follows:
    //    0,  1,  2,  3,  4,  5,  6,  7,  8, -
    //    9, 10, 11, 12, 13, 14, 15, 16, 17, -
    //   18, 19, 20, 21, 22, 23, 24, 25, 26
    //  
    // row_triads are canonically naturally numbered 0, 1, 2, for the first row, 3, 4, 5 for the second
    // and so on.  Row triads are stored in 2 different forms.
    // The first form is linear in canonical order (TriadInfo.row_triads_canonical)
    //    0,  1,  2,  3,  4,  5,  6,  7,  8,
    //    9, 10, 11, 12, 13, 14, 15, 16, 17,
    //   18, 19, 20, 21, 22, 23, 24, 25, 26
    // The second form (TriadInfo.row/col_triads) consists of three arrays of 10 ushort, captured 3 per row,
    // 3 rows stacked vertically with the subsequent 3 rows offset by 3:
    //    0,  1,  2,  9, 10, 11, 18, 19, 20, -
    //    3,  4,  5, 12, 13, 14, 21, 22, 23, -
    //    6,  7,  8, 15, 16, 17, 24, 25, 26
    // SIMD Triad processing requires all triads of a given band to be positioned
    // 3 horizontal triads side by side and 3 vertical triads in the same positions
    // of the two other rows.  The ordering described above satisfies this requirement.
    //
    // For technical reasons, an (unused) 10th ushort value is injected between each set of 9 triads
    // for a total of 29 triad values.
    //
    // Various methods are used to efficiently save the triads in the processing order,
    // which go beyond the end of the arrays normally needed.
    //

    TriadInfo triad_info;

    // Algo 2 and Algo 3.1
    {
        __m256i column_or_tails[9];
        __m256i column_or_head = _mm256_setzero_si256();
        __m256i column_cand_or = _mm256_setzero_si256();
        __m256i col_triads_1, col_triads_2, col_triads_3;
        {
            // columns
            // to start, we simply tally the or'ed rows
            signed char j = 81-9;
            // compute fresh col_triads_3
            col_triads_3 = _mm256_setzero_si256();
            // A2 (cols)
            // precompute 'tails' of the or'ed column_cand_or only once
            // working backwords
            // 3 iterations, rows 8, 7 and 6.
            for ( ; j >= 54; j -= 9) {
                column_or_tails[j/9-1] = col_triads_3;
                col_triads_3 = _mm256_or_si256(col_triads_3, *(__m256i_u*) &candidates[j]);
            }
            // col_triads_3 now contains the third set of column-triads
            // compute fresh col_triads_2
            // A2 and A3.1.a
            // 3 iterations, rows 5, 4 and 3.
            col_triads_2 = _mm256_setzero_si256();
            for ( ; j >= 27; j -= 9) {
                column_or_tails[j/9-1] = _mm256_or_si256(col_triads_2,col_triads_3);
                col_triads_2 = _mm256_or_si256(col_triads_2, *(__m256i_u*) &candidates[j]);
            }
            // A2 and A3.1.a
            // col_triads_2/3 contain the second/third set of column-triads
            column_cand_or = _mm256_or_si256(col_triads_2, col_triads_3);

            // 2 iterations, rows 1 and 2.
            // the first set of column triads is computed below as part of the computed
            // 'head' or.
            for ( ; j > 0; j -= 9) {
                column_or_tails[j/9-1] = column_cand_or;
                column_cand_or = _mm256_or_si256(column_cand_or, *(__m256i_u*) &candidates[j]);
            }
            // or in row 0 and check whether all digits or covered
            if ( check_back && !_mm256_testz_si256(mask9x1ff,_mm256_andnot_si256(_mm256_or_si256(column_cand_or, *(__m256i*) &candidates[0]), mask9x1ff)) ) {
                // the current grid has no solution, go back
                if ( verbose != VNone ) {
                    __m256i missing = _mm256_andnot_si256(_mm256_or_si256(column_cand_or, *(__m256i*) &candidates[0]), mask9x1ff);
                    unsigned int m  = _mm256_movemask_epi8(_mm256_cmpgt_epi16(missing, _mm256_setzero_si256()));
                    int idx = _tzcnt_u32(m)>>1;
                    unsigned short digit = ((v16us)missing)[idx];
                    if ( grid_state->stackpointer == 0 && unique_check_mode == 0 ) {
                        if ( warnings != 0 ) {
                            printf("Line %d: stack 0, back track - column %d misses digit %d\n", line, idx, _tzcnt_u16(digit)+1);
                        }
                    } else if ( debug ) {
                        printf("back track - missing digit %d in column %d\n", _tzcnt_u16(digit)+1, idx);
                    }
                }
                goto back;
            }

            // breaking the column hidden singles out of the loop this way will win some performance
            unsigned int jrow = 0;
            for (unsigned int j = 0; j < 81; j+=9, jrow++) {
                // turn the or'ed rows into a mask for the singletons, if any.
                // check col (9) candidates
                unsigned short m = (j < 64) ? (unlocked[0] >> j) : (unlocked[1] >> (j-64));
                if ( j == 63) {
                    m |= unlocked[1] << (64-63);
                }
                __m256i a = _mm256_cmpgt_epi16(mask9x1ff, column_cand_or);
                unsigned int mask = and_compress_masks<false>(a, m & 0x1ff);
                if ( mask) {
                    int idx = _tzcnt_u32(mask);
                    e_i = j+idx;
                    e_digit = ((v16us)_mm256_andnot_si256(column_cand_or, mask9x1ff))[idx];
                    if ( check_back && (e_digit & (e_digit-1)) ) {
                        if ( verbose != VNone ) {
                            if ( grid_state->stackpointer == 0 && unique_check_mode == 0 ) {
                                if ( warnings != 0 ) {
                                    printf("Line %d: stack 0, back track - col cell %s does contain multiple hidden singles\n", line, cl2txt[e_i]);
                                }
                            } else if ( debug ) {
                                printf("back track - multiple hidden singles in col cell %s\n", cl2txt[e_i]);
                            }
                        }
                        goto back;
                    }
                    if ( verbose == VDebug ) {
                        printf("hidden single (col)");
                    }
                    goto enter;
                }

                // leverage previously computed or'ed rows in head and tails.
                column_or_head = column_cand_or = _mm256_or_si256(*(__m256i_u*) &candidates[j], column_or_head);
                column_cand_or = _mm256_or_si256(column_or_tails[jrow],column_cand_or);

                if ( j == 18 ) {
                    // A3.1.c
                    col_triads_1 = column_or_head;
                }
            }
        }

        // rows
        //
        // capture row-based triad values only

        unsigned char irow = 0;

        for (unsigned char i = 0; i < 72; i += 18, irow+=2) {
            // rows, in pairs

            __m256i c1 = _mm256_set_m128i(*(__m128i_u*) &candidates[i+9],
                                          *(__m128i_u*) &candidates[i]);

            unsigned short the9thcand_row[2] = { candidates[i+8], candidates[i+17] };

            __m256i row_or7 = _mm256_setzero_si256();
            __m256i row_9th = _mm256_set_m128i(_mm_set1_epi16(the9thcand_row[1]),_mm_set1_epi16(the9thcand_row[0]));

            __m256i row_triad_capture[2];
            {
                __m256i c1_ = c1;
                // A2 and A3.1.b
                // first lane for the row, 2nd lane for the box
                // step j=0
                // rotate left (0 1 2 3 4 5 6 7) -> (1 2 3 4 5 6 7 0)
                c1_ = _mm256_alignr_epi8(c1_, c1_, 2);
                row_or7 = _mm256_or_si256(c1_, row_or7);
                // step j=1
                // rotate (1 2 3 4 5 6 7 0) -> (2 3 4 5 6 7 0 1)
                c1_ = _mm256_alignr_epi8(c1_, c1_, 2);
                row_or7 = _mm256_or_si256(c1_, row_or7);
                // triad capture: after 2 rounds and oring in the 9th element, row triad 3 of this row saved in pos 5
                row_triad_capture[0] = _mm256_or_si256(row_or7, row_9th);
                // step j=2
                // rotate (2 3 4 5 6 7 0 1) -> (3 4 5 6 7 0 1 2)
                c1_ = _mm256_alignr_epi8(c1_, c1_, 2);
                // triad capture: after 3 rounds 2 row triads 0 and 1 in pos 7 and 2
                row_triad_capture[1] = row_or7 = _mm256_or_si256(c1_, row_or7);
                }
            // deal with saved row_triad_capture:
            // - captured triad 3 of this row saved in pos 5 of row_triad_capture[0]
            // - captured triads 0 and 1 in pos 7 and 2 of row_triad_capture[1]
            // Blend the two captured vectors to contain all three triads in pos 7, 2 and 5
            // shuffle the triads from pos 7,2,5 into pos 0,1,2
            // spending 3 instructions on this: blend, shuffle, storeu
            // a 'random' 4th unsigned short is overwritten by the next triad store
            // (or is written into the gap 10th slot).
            row_triad_capture[0] = _mm256_shuffle_epi8(_mm256_blend_epi16(row_triad_capture[1], row_triad_capture[0], 0x20),shuff725to012);
            _mm_storeu_si64(&triad_info.row_triads[row_triads_lut[irow]], _mm256_castsi256_si128(row_triad_capture[0]));
            _mm_storeu_si64(&triad_info.row_triads[row_triads_lut[irow+1]], _mm256_extracti128_si256(row_triad_capture[0],1));
            // the following 2 lines are only useful with mode_dgt
            _mm_storeu_si64(&triad_info.row_triads_canonical[irow*3], _mm256_castsi256_si128(row_triad_capture[0]));
            _mm_storeu_si64(&triad_info.row_triads_canonical[(irow+1)*3], _mm256_extracti128_si256(row_triad_capture[0],1));
        }   // for row Px

        // 9th row
        {
            __m128i c = *(__m128i_u*) &candidates[72];

            unsigned short the9thcand_row = candidates[80];

            __m128i row_or7;
            __m128i row_9th_elem = _mm_set1_epi16(the9thcand_row);

            __m128i row_triad_capture[2];
            {
                __m128i c_ = c;
                // A2 and A3.1.b
                // first lane for the row, 2nd lane for the box
                // step j=0
                // rotate left (0 1 2 3 4 5 6 7) -> (1 2 3 4 5 6 7 0)
                c_ = _mm_alignr_epi8(c_, c_, 2);
                row_or7 = c_;
                // step j=1
                // rotate (1 2 3 4 5 6 7 0) -> (2 3 4 5 6 7 0 1)
                c_ = _mm_alignr_epi8(c_, c_, 2);
                row_or7 = _mm_or_si128(c_, row_or7);
                // triad capture: after 2 rounds, row triad 3 of this row saved in pos 5
                row_triad_capture[0] = _mm_or_si128(row_or7, row_9th_elem);
                // step j=2
                // rotate (2 3 4 5 6 7 0 1) -> (3 4 5 6 7 0 1 2)
                c_ = _mm_alignr_epi8(c_, c_, 2);
                // triad capture: after 3 rounds 2 row triads 0 and 1 in pos 7 and 2
                row_triad_capture[1] = row_or7 = _mm_or_si128(c_, row_or7);
            } // row
            // deal with saved row_triad_capture:
            // - captured triad 3 of this row saved in pos 5 of row_triad_capture[0]
            // - captured triads 0 and 1 in pos 7 and 2 of row_triad_capture[1]
            // Blend the two captured vectors to contain all three triads in pos 7, 2 and 5
            // shuffle the triads from pos 7,2,5 into pos 0,1,2
            // spending 3 instructions on this: blend, shuffle, storeu
            // a 'random' 4th unsigned short is overwritten by the next triad store
            // (or is written into the gap 10th slot).
            row_triad_capture[0] = _mm_shuffle_epi8(_mm_blend_epi16(row_triad_capture[1], row_triad_capture[0], 0x20),_mm256_castsi256_si128(shuff725to012));
            _mm_storeu_si64(&triad_info.row_triads[row_triads_lut[8]], row_triad_capture[0]);
            // the following line is only useful with mode_dgt
            _mm_storeu_si64(&triad_info.row_triads_canonical[8*3], row_triad_capture[0]);
        } // row R9

        // Store all column triads sequentially, paying attention to overlap.
        //
        // A3.1.c
        _mm256_storeu_si256((__m256i *)triad_info.col_triads, col_triads_1);
        _mm256_storeu_si256((__m256i *)(triad_info.col_triads+10), col_triads_2);
        // mask 5 hi triads to 0xffff, which will not trigger any checks
        _mm256_storeu_si256((__m256i *)(triad_info.col_triads+20), _mm256_or_si256(col_triads_3, mask11hi));

    } // Algo 2 and Algo 3.1

    { // Algo 3.2
        // A3.2.1 (check col-triads)
        // just a plain old popcount, but using the knowledge that the upper byte is always 1 or 0.

        __m256i v1 = _mm256_loadu_si256((__m256i *)triad_info.col_triads);
        __m256i v2 = _mm256_loadu_si256((__m256i *)(triad_info.col_triads+16));
        __m256i v9 = _mm256_packus_epi16(_mm256_srli_epi16(v1,8), _mm256_srli_epi16(v2,8));
        v1 = _mm256_packus_epi16(_mm256_and_si256(v1, maskff), _mm256_and_si256(v2, maskff));
        __m256i lo1 = _mm256_and_si256 (v1, nibble_mask);
        __m256i hi1 = _mm256_and_si256 (_mm256_srli_epi16 (v1, 4), nibble_mask );
        __m256i cnt11 = _mm256_shuffle_epi8 (lookup, lo1);
        cnt11 = _mm256_add_epi8(_mm256_add_epi8 (cnt11, _mm256_shuffle_epi8 (lookup, hi1)), v9);
        // the result of the popcnt
        __m256i res = _mm256_and_si256(_mm256_permute4x64_epi64(cnt11, 0xD8), mask27);

        // do this only if unlocked has been updated:
        if ( last_entered_count_col_triads != current_entered_count) {
            // the high byte is set to the stackpointer, so that this works
            // across guess/backtrack
            last_entered_count_col_triads = current_entered_count;
            unsigned long long tr = unlocked[0];  // col triads - set if any triad cell is unlocked
            tr |= (tr >> 9) | (tr >> 18) | (unlocked[1]<<(64-9)) | (unlocked[1]<<(64-18));
            // mimick the pattern of col_triads, i.e. a gap of 1 after each group of 9.
            if ( pext_support ) {
                tr = _pext_u64(tr, 0x1ffLL | (0x1ffLL<<27) | (0x1ffLL<<54));
                grid_state->triads_unlocked[Col] &= _pdep_u64(tr, (0x1ff)|(0x1ff<<10)|(0x1ff<<20));
            } else {
                grid_state->triads_unlocked[Col] &= (tr & 0x1ff) | ((tr >> 17) & (0x1ff<<10)) | ((tr >> 34) & (0x1ff<<20));
            }
        }

        triad_info.triads_selection[Col] = _mm256_movemask_epi8(_mm256_cmpeq_epi8(res, fours));
        unsigned long long m = _mm256_movemask_epi8(_mm256_cmpeq_epi8(res, threes)) & grid_state->triads_unlocked[Col];
        if ( verbose != VNone ) {
            counters.triads_resolved += _popcnt64(m);
        }
        while (m) {
            unsigned char tidx = tzcnt_and_mask(m);
            unsigned char off  = tidx/10*3*9+tidx%10;
            unsigned short cands_triad = triad_info.col_triads[tidx];
            if ( verbose == VDebug ) {
                char ret[32];
                format_candidate_set(ret, cands_triad);
                printf("triad set (col): %-9s %s\n", ret, cl2txt[off]);
            }
            // mask off resolved triad:
            _btr_u32(grid_state->triads_unlocked[Col], tidx);
//            grid_state->set23_found[Col].set_indexbits(0x40201,off,19);
//            grid_state->set23_found[Box].set_indexbits(0x40201,off,19);
        }

        // A3.2.2 (check row-triads)

        // just a plain old popcount, but using the knowledge that the upper byte is always 1 or 0.
        v1 = _mm256_loadu_si256((__m256i *)triad_info.row_triads);
        v2 = _mm256_loadu_si256((__m256i *)(triad_info.row_triads+16));
        v9 = _mm256_packus_epi16(_mm256_srli_epi16(v1,8), _mm256_srli_epi16(v2,8));
        v1 = _mm256_packus_epi16(_mm256_and_si256(v1, maskff), _mm256_and_si256(v2, maskff));
        lo1 = _mm256_and_si256 (v1, nibble_mask);
        hi1 = _mm256_and_si256 (_mm256_srli_epi16 (v1, 4), nibble_mask );
        cnt11 = _mm256_shuffle_epi8 (lookup, lo1);
        cnt11 = _mm256_add_epi8(_mm256_add_epi8 (cnt11, _mm256_shuffle_epi8 (lookup, hi1)), v9);
        res = _mm256_and_si256(_mm256_permute4x64_epi64(cnt11, 0xD8), mask27);

        m = _mm256_movemask_epi8(_mm256_cmpeq_epi8(res, threes)) & grid_state->triads_unlocked[Row];
        triad_info.triads_selection[Row] = _mm256_movemask_epi8(_mm256_cmpeq_epi8(res, fours));

        // the best that can be done for rows - remember that the order of
        // row triads is not aligned with the order of cells.
        bit128_t tr = grid_state->unlocked;   // for row triads - any triad cell unlocked

        // to allow checking of unresolved triads
        tr.u64[0]  |= (tr.u64[0] >> 1)  | (tr.u64[0] >> 2);
        tr.u64[0]  |= ((tr.u64[1] & 1)  | ((tr.u64[1] & 2) >> 1))<<63;
        tr.u64[1]  |= (tr.u64[1] >> 1)  | (tr.u64[1] >> 2);

        while (m) {
            unsigned char tidx = tzcnt_and_mask(m);
            unsigned char off  = row_triad_index_to_offset[tidx];

            if ( !tr.check_indexbit(off)) {  // locked
                _btr_u32(grid_state->triads_unlocked[Row], tidx);
                continue;
            }

            if ( verbose != VNone ) {
                counters.triads_resolved++;
            }
            if ( verbose == VDebug ) {
                char ret[32];
                format_candidate_set(ret, triad_info.row_triads[tidx]);
                printf("triad set (row): %-9s %s\n", ret, cl2txt[off]);
            }

            // mask off resolved triad:
            grid_state->triads_unlocked[Row] &= ~(1LL << tidx);
//            grid_state->set23_found[Row].set_indexbits(0x7,off,3);
//            grid_state->set23_found[Box].set_indexbits(0x7,off,3);
        } // while
    }   // Algo 3 Part 2

    {   // Algo 3 Part 3
        // Note on nomenclature:
        // - must / mustnt = candidates that must or must not occur in the triad.
        // - t vs p prefix: t for triad (only applies to the triad), p for peers, i.e. what
        //   the peers impose onto the triad.
        //
        // The SIMD parallelism consists of computing tmustnt for three bands in parallel.
        //
        bool any_changes = false;

        // mask for each line of 9 triad *must/*mustnt.
        // all t/pmust* variables are populated in groups of three, the first two in low 12 bytes (0-11),
        // the third group in bytes 16-21.

        for (int type=1; type>=0; type--) {	// row = 0, col = 1

            // input
            unsigned short *triads   = type==0?triad_info.row_triads:triad_info.col_triads;
            unsigned short *wo_musts = type==0?triad_info.row_triads_wo_musts:triad_info.col_triads_wo_musts;
            unsigned short *ptriads  = triads;
            __m256i pmustnt[2][3] = {mask_musts, mask_musts, mask_musts, mask_musts, mask_musts, mask_musts};
            __m256i tmustnt[3];

            // first load triad candidates and compute peer based pmustnt

            // i=0 (manually unrolled loop)
                // tmustnt computed from all candidates in row/col_triads
                __m256i tmustnti = _mm256_andnot_si256(_mm256_loadu2_m128i((__m128i*)&ptriads[6], (__m128i*)ptriads), mask_musts);
                tmustnt[0] = tmustnti;
                // compute peer-based pmustnt
                // vertical peers
                pmustnt[0][1] = _mm256_and_si256(pmustnt[0][1], tmustnti);
                pmustnt[0][2] = _mm256_and_si256(pmustnt[0][2], tmustnti);

                // horizontal peers
                tmustnti = _mm256_shuffle_epi8(tmustnti, rot_hpeers);
                pmustnt[1][0] = _mm256_and_si256(pmustnt[1][0], tmustnti);
                tmustnti = _mm256_shuffle_epi8(tmustnti, rot_hpeers);
                pmustnt[1][0] = _mm256_and_si256(pmustnt[1][0], tmustnti);
            // i=1
                ptriads += 10;
                tmustnti = _mm256_andnot_si256(_mm256_loadu2_m128i((__m128i*)&ptriads[6], (__m128i*)ptriads), mask_musts);
                tmustnt[1] = tmustnti;
                // vertical peers
                pmustnt[0][0] = _mm256_and_si256(pmustnt[0][0], tmustnti);
                pmustnt[0][2] = _mm256_and_si256(pmustnt[0][2], tmustnti);

                // horizontal peers
                tmustnti = _mm256_shuffle_epi8(tmustnti, rot_hpeers);
                pmustnt[1][1] = _mm256_and_si256(pmustnt[1][1], tmustnti);
                tmustnti = _mm256_shuffle_epi8(tmustnti, rot_hpeers);
                pmustnt[1][1] = _mm256_and_si256(pmustnt[1][1], tmustnti);
            // i=2
                ptriads += 10;
                tmustnti = _mm256_andnot_si256(_mm256_loadu2_m128i((__m128i*)&ptriads[6], (__m128i*)ptriads), mask_musts);
                tmustnt[2] = tmustnti;
                // vertical peers
                pmustnt[0][0] = _mm256_and_si256(pmustnt[0][0], tmustnti);
                pmustnt[0][1] = _mm256_and_si256(pmustnt[0][1], tmustnti);

                // horizontal peers
                tmustnti = _mm256_shuffle_epi8(tmustnti, rot_hpeers);
                pmustnt[1][2] = _mm256_and_si256(pmustnt[1][2], tmustnti);
                tmustnti = _mm256_shuffle_epi8(tmustnti, rot_hpeers);
                pmustnt[1][2] = _mm256_and_si256(pmustnt[1][2], tmustnti);

            // second, compute tmust and propagate it to its peers
            // combine the pmustnt:
                __m256i tmust0 = _mm256_or_si256(pmustnt[0][0], pmustnt[1][0]);
                __m256i tmust1 = _mm256_or_si256(pmustnt[0][1], pmustnt[1][1]);
                __m256i tmust2 = _mm256_or_si256(pmustnt[0][2], pmustnt[1][2]);

                // tmust:
                // add to tmusts triads that are locked (exactly 3 candidates)
                // another task is to put aside data identifying triad candidate sets
                // minus the tmusts, that will be useful for finding sets and good guesses.
                unsigned int tul1 = grid_state->triads_unlocked[type] & (0x3f | (0x3f<<10) | (0x3f<<20));
                unsigned int tul2 = (grid_state->triads_unlocked[type]<<2) & (0x700 | (0x700<<10) | (0x700<<20));
                __m256i triadsv = _mm256_loadu2_m128i((__m128i*)&triads[6], (__m128i*)triads);
                tmust0 = _mm256_or_si256(tmust0, _mm256_andnot_si256(expand_bitvector((tul1 & 0x3f) | (tul2 & 0x700)),
                                                 triadsv));
                triadsv = _mm256_andnot_si256(tmust0, triadsv);
                _mm_storeu_si128((__m128i*)wo_musts, _mm256_castsi256_si128(triadsv));
                *(unsigned long long*)(&wo_musts[6]) = _mm256_extract_epi64(triadsv, 2);
                tul1 >>= 10;
                tul2 >>= 10;
                triadsv = _mm256_loadu2_m128i((__m128i*)&triads[16], (__m128i*)&triads[10]);
                tmust1 = _mm256_or_si256(tmust1, _mm256_andnot_si256(expand_bitvector((tul1 & 0x3f) | (tul2 & 0x700)),
                                                     triadsv));
                triadsv = _mm256_andnot_si256(tmust1, triadsv);
                _mm_storeu_si128((__m128i*)&wo_musts[10], _mm256_castsi256_si128(triadsv));
                *(unsigned long long*)(&wo_musts[16]) = _mm256_extract_epi64(triadsv, 2);
                tul1 >>= 10;
                tul2 >>= 10;
                triadsv = _mm256_loadu2_m128i((__m128i*)&triads[26], (__m128i*)&triads[20]);
                tmust2 = _mm256_or_si256(tmust2, _mm256_andnot_si256(expand_bitvector((tul1 & 0x3f) | (tul2 & 0x700)),
                                                     triadsv));
                triadsv = _mm256_andnot_si256(tmust2, triadsv);
                _mm_storeu_si128((__m128i*)&wo_musts[20], _mm256_castsi256_si128(triadsv));
                *(unsigned long long*)(&wo_musts[26]) = _mm256_extract_epi64(triadsv, 2);

            // i=0 (manually unrolled loop)
                // augment peer-based tmustnt by propagating the constraint (tmusti)
                // vertical peers
                tmustnt[1] = _mm256_or_si256(tmustnt[1], tmust0);
                tmustnt[2] = _mm256_or_si256(tmustnt[2], tmust0);

                // horizontal peers
                tmust0 = _mm256_shuffle_epi8(tmust0, rot_hpeers);
                tmustnt[0] = _mm256_or_si256(tmustnt[0], tmust0);
                tmust0 = _mm256_shuffle_epi8(tmust0, rot_hpeers);
                tmustnt[0] = _mm256_or_si256(tmustnt[0], tmust0);

             // i=1
                // vertical peers
                tmustnt[0] = _mm256_or_si256(tmustnt[0], tmust1);
                tmustnt[2] = _mm256_or_si256(tmustnt[2], tmust1);

                // horizontal peers
                tmust1 = _mm256_shuffle_epi8(tmust1, rot_hpeers);
                tmustnt[1] = _mm256_or_si256(tmustnt[1], tmust1);
                tmust1 = _mm256_shuffle_epi8(tmust1, rot_hpeers);
                tmustnt[1] = _mm256_or_si256(tmustnt[1], tmust1);

             // i=2
                // vertical peers
                tmustnt[0] = _mm256_or_si256(tmustnt[0], tmust2);
                tmustnt[1] = _mm256_or_si256(tmustnt[1], tmust2);

                // horizontal peers
                tmust2 = _mm256_shuffle_epi8(tmust2, rot_hpeers);
                tmustnt[2] = _mm256_or_si256(tmustnt[2], tmust2);
                tmust2 = _mm256_shuffle_epi8(tmust2, rot_hpeers);
                tmustnt[2] = _mm256_or_si256(tmustnt[2], tmust2);

            ptriads = triads;
            // compare tmustnt with the triads.
            unsigned int row_combo_tpos[3] {};
            unsigned int rslvd_row_combo_tpos[3] {};
            // for each group of nine ptriads:
            // - for column triads, each group (tmustnt) updates a band, the nine elemements (columns) 
            //   of the vector,
            // - for row triads, the n-th sub-groups updates the i-th row of the n-th band, with
            //   minor differences in the extraction of the update vector.
            for ( int i=0; i<3; i++, ptriads+=10) {
                // flip low and high lanes
                __m256i flip = _mm256_permute2x128_si256(tmustnt[i], tmustnt[i], 1);
                // rotate low 16 bytes left by 4 for subsequent move across lanes by alignr
                tmustnt[i]   = _mm256_shuffle_epi8(tmustnt[i], shuff_tmustnt);
                // align the 9 constraints in candidate order for comparison/update.
                tmustnt[i]   = _mm256_alignr_epi8(flip, tmustnt[i], 4);
                // isolate aligned updates
                __m256i to_remove_v = _mm256_and_si256(_mm256_and_si256(*(__m256i_u*)ptriads, tmustnt[i]),mask9x1ff);
                if ( _mm256_testz_si256(mask9x1ff,to_remove_v)) {
                    continue;
                }
                __m256i tmp = _mm256_cmpgt_epi16(to_remove_v,_mm256_setzero_si256());

                // m represents all triad indices to update
                unsigned long long m = compress_epi16_boolean(tmp);
                if ( m ) {
                    any_changes = true;
                }
                unsigned int rslvd_col_combo_tpos = 0;
                // remove triad candidate values that appear in tmustnt
                // and track updated cells
                if ( type == 0 ) {
                    // row triad updates, row i of each band
                    // update the cells for row triads
                    // row i
                    __m256i tmask;
                    unsigned int bits;
                    if ( (bits = m & 0x7) ) {
                        tmask = _mm256_permute2x128_si256(tmustnt[i], tmustnt[i], 0);
                        tmask = _mm256_shuffle_epi8(tmask, shuff_row_mask);
                        __m256i c = _mm256_andnot_si256(tmask, *(__m256i_u*)&candidates[i*9]);
                        _mm_storeu_si128((__m128i_u*)&candidates[i*9], _mm256_castsi256_si128(c));
                        candidates[8+i*9] = _mm256_extract_epi16(c, 8);
                        row_combo_tpos[0] |= bitx3_lut[bits] << (9*i);
                    }
                    // row i+3
                    if ( (bits = (m >> 3) & 0x7) ) {
                        tmask = _mm256_bsrli_epi128(tmustnt[i], 6);
                        tmask = _mm256_permute2x128_si256(tmask, tmask, 0);
                        tmask = _mm256_shuffle_epi8(tmask, shuff_row_mask);
                        __m256i c = _mm256_andnot_si256(tmask, *(__m256i_u*)&candidates[(i+3)*9]);
                        _mm_storeu_si128((__m128i_u*)&candidates[(i+3)*9], _mm256_castsi256_si128(c));
                        candidates[8+(i+3)*9] = _mm256_extract_epi16(c, 8);
                        row_combo_tpos[1] |= bitx3_lut[bits] << (9*i);
                    }
                    // row i+6
                    if ( (bits = (m >> 6) & 0x7) ) {
                        tmask = _mm256_permute4x64_epi64(tmustnt[i], 0x99);
                        tmask = _mm256_shuffle_epi8(tmask, shuff_row_mask2);
                        __m256i c = _mm256_andnot_si256(tmask, *(__m256i_u*)&candidates[(i+6)*9]);
                        _mm_storeu_si128((__m128i_u*)&candidates[(i+6)*9], _mm256_castsi256_si128(c));
                        candidates[8+(i+6)*9] = _mm256_extract_epi16(c, 8);
                        row_combo_tpos[2] |= bitx3_lut[bits] << (9*i);
                    }
                } else { // type == 1
                    // update the band of 3 rows with column triads
                    // using directly tmustnt[i]
                    unsigned int i27 = i*27;
                    __m256i c1 = _mm256_andnot_si256(tmustnt[i], *(__m256i_u*)&candidates[i27]);
                    _mm_storeu_si128((__m128i_u*)&candidates[i27], _mm256_castsi256_si128(c1));
                    __m256i c2 = _mm256_andnot_si256(tmustnt[i], *(__m256i_u*)&candidates[9+i27]);
                    _mm_storeu_si128((__m128i_u*)&candidates[9+i27], _mm256_castsi256_si128(c2));
                    __m256i c3 = _mm256_andnot_si256(tmustnt[i], *(__m256i_u*)&candidates[18+i27]);
                    _mm_storeu_si128((__m128i_u*)&candidates[18+i27], _mm256_castsi256_si128(c3));
                    if ( m & 0x100 ) {
                        candidates[8+i27]    = _mm256_extract_epi16(c1, 8);
                        candidates[9+8+i27]  = _mm256_extract_epi16(c2, 8);
                        candidates[18+8+i27] = _mm256_extract_epi16(c3, 8);
                    }
//                    grid_state->updated.set_indexbits(m | (m<<9) | (m<<18), i*27, 27);

                }
                // iterate over triads to update
                if ( verbose != VNone ) {
                     counters.triad_updates += _popcnt64(m);
                }
                while (m) {
                    // 1. Check for triad resolution
                    // 2. log the triad update from above
                    // 3. for columns, track resolved triads
                    unsigned char i_rel = tzcnt_and_mask(m);
                    unsigned char ltidx = i_rel + 9*i;  // logical triad index (within band, different layouts for row/col)
                    ltidx = type? ltidx : row_triad_canonical_map[ltidx];

                    // compute bit offsets of the resolved triad
                    // and save the mask
                    if ( _popcnt32(((v16us)tmustnt[i])[i_rel]) == 6 ) {
                        if ( type == 0 ) {
                            rslvd_row_combo_tpos[i_rel/3] |= bandbits_by_index[ltidx%9];
                        } else {
                            rslvd_col_combo_tpos |= 0x40201<<(ltidx%9);
                        }
                        if ( verbose != VNone ) {
                            counters.triads_resolved++;
                        }
                    }
                    if ( verbose == VDebug ) {
                        char ret[32];
                        format_candidate_set(ret, ((v16us)to_remove_v)[i_rel]);
                        printf("remove %-5s from %s triad at %s\n", ret, type == 0? "row":"col",
                              cl2txt[type==0?ltidx*3:col_canonical_triad_pos[ltidx]] );
                    }
                }
                if ( type == 1 ) {
                    if ( rslvd_col_combo_tpos ) {
//                        grid_state->set23_found[Col].set_indexbits(rslvd_col_combo_tpos, i*27, 27);
//                        grid_state->set23_found[Box].set_indexbits(rslvd_col_combo_tpos, i*27, 27);
                    }
                }
            }
            // finally, for rows only, track updated triad cells and resolved triads
            if ( type == 0 ) {
                for ( unsigned char k=0; k<3; k++ ) {
                    if ( row_combo_tpos[k] ) {
//                        grid_state->updated.set_indexbits(row_combo_tpos[k], k*27, 27);
                    }
                    if ( rslvd_row_combo_tpos[k] ) {
//                        grid_state->set23_found[Row].set_indexbits(rslvd_row_combo_tpos[k], k*27, 27);
//                        grid_state->set23_found[Box].set_indexbits(rslvd_row_combo_tpos[k], k*27, 27);
                    }
                }
            }
        } // for type (cols,rows)

        if ( any_changes ) {
            goto search;
        }
    }

    // here we are beyond the enter/single search/triad/, on to the more complex algos,
    // so we count this as a 'round'.
    if ( verbose != VNone ) {
        counters.past_naked_count++;
    }

    {
        __m256i c;
        for (unsigned char i = 0; i < 64; i += 32) {
            c = _mm256_load_si256((__m256i*) &candidates[i]);
            __m256i c2 = _mm256_load_si256((__m256i*) &candidates[i+16]);
            __m256i lsb  = get_first_lsb(c);
            __m256i lsb2  = get_first_lsb(c2);
            lsb = andnot_get_next_lsb(lsb, c);
            lsb2 = andnot_get_next_lsb(lsb2, c2);
            // check whether lsb is the last bit
            // count the twos
            bivalues.u32[i>>5] =
                compress_epi16_boolean<false>(_mm256_and_si256(
                                      _mm256_cmpgt_epi16(c,_mm256_setzero_si256()),
                                      _mm256_cmpeq_epi16(lsb,c)),
                                      _mm256_and_si256(
                                      _mm256_cmpgt_epi16(c2,_mm256_setzero_si256()),
                                      _mm256_cmpeq_epi16(lsb2,c2)));
        }
        c = _mm256_load_si256((__m256i*) &candidates[64]);
        __m256i lsb  = get_first_lsb(c);
        lsb = andnot_get_next_lsb(lsb, c);
        // check whether lsb is the last bit
        // count the twos
        bivalues.u64[1] = compress_epi16_boolean<false>(_mm256_and_si256(
                                 _mm256_cmpgt_epi16(c,_mm256_setzero_si256()),
                                 _mm256_cmpeq_epi16(lsb,c)));

        bivalues.u16[5] = (_popcnt_u16(candidates[80]) == 2)?1:0;
        // bivalues is now set for subsequent steps
    }

    {
        // Before making a guess,
        // check for a 'universal grave'+1, which is an end-game move.
        // First get a count of unlocked.
        // With around 22 or less unresolved cells (N), accumulate a popcount of all cells.
        // Count the cells with a candidate count 2 (P). 81 - N - P = Q.
        // If Q is 0, back track, if a guess was made before. without guess to back track to,
        // take note as duplicate solution, but carry on with a guess.
        // If Q > 1, go with a guess.
        // If Q == 1, identify the only cell which does not have a 2 candidate count.
        // If the candidate count is 3, determine for any section with the cell which
        // candidate value appears in this cell and other cells of the section 3 times.
        // That is the correct solution for this cell, enter it.

        unsigned char N = _popcnt64(unlocked[0]) + _popcnt32(unlocked[1]);
        if ( N <= 23 ) {
            unsigned char target = 0;   // the index of the only cell with three candidates
            int sum2 = _popcnt64(bivalues.u64[0]) + _popcnt32(bivalues.u64[1]);
            if ( sum2 == N ) {
                grid_state->multiple_solutions_exist = 1;
                if ( verbose != VNone ) {
                    counters.bug_count++;
                }
                if ( rules != Regular ) {
                    if ( verbose == VDebug ) {
                        if ( grid_state->stackpointer == 0 && !unique_check_mode ) {
                            printf("Found a bi-value universal grave. This means at least two solutions exist.\n");
                        } else if ( unique_check_mode ) {
                            printf("checking a bi-value universal grave.\n");
                        }
                    }
                    goto guess;
                } else if ( grid_state->stackpointer ) {
                    if ( verbose == VDebug ) {
                        printf("back track - found a bi-value universal grave.\n");
                    }
                    goto back;
                } else {
                    if ( verbose == VDebug ) {
                        printf("Found a bi-value universal grave. This means at least two solutions exist.\n");
                    }
                    status.unique = false;  // set to non-unique even under Regular rules
                    goto guess;
                }
            } else if ( sum2+1 == N ) {  // find the single cell with count > 2
                bit128_t gt2 = { .u128 = ((bit128_t*)unlocked)->u128 & ~bivalues.u128 };
                // locate the cell
                unsigned long long m = gt2.u64[0];
                if ( m == 0 ) {
                    m = gt2.u64[1];
                    target = 64;
                }
                target += _tzcnt_u64(m);
            } else {
                goto no_bug;
            }
            if ( _popcnt_u16(candidates[target]) == 3 ) {
                unsigned char row = row_index[target];
                unsigned short cand3 = candidates[target];
                unsigned short digit = 0;
                unsigned short mask = ((bit128_t*)unlocked)->get_indexbits(row*9,9);
                __m256i maskv = expand_bitvector(mask);
                __m256i c = _mm256_and_si256(_mm256_load_si256((__m256i*) &candidates[row*9]), maskv);
                while (cand3) {
                    unsigned short canddigit = __blsi_u32(cand3);
                    // count cells with this candidate digit:
                    __m256i tmp = _mm256_and_si256(_mm256_set1_epi16(canddigit), c);
                    // as a boolean
                    tmp = _mm256_cmpeq_epi16(tmp,_mm256_setzero_si256());
                    // need three cell, doubled bits in mask:
                    if ( _popcnt32(~_mm256_movemask_epi8(tmp)) == 3*2 ) {
                        digit = canddigit;
                        break;
                    }
                    cand3 &= ~canddigit;
                }
                if ( digit ) {
                    if ( verbose != VNone ) {
                        counters.bug_plus1_count++;
                    }
                    if ( verbose == VDebug ) {
                        printf("bi-value universal grave (BUG) +1: pivot:");
                    }
                    if ( rules == Regular ) {
                        e_i = target;
                        e_digit = digit;
                        goto enter;
                    } else {
                        if ( verbose == VDebug ) {
                            printf("\n");
                        }
                        grid_state = grid_state->make_guess<verbose>(target, digit, counters);
                    }
                    goto start;
                }
            }
        }
        no_bug:
        ;
    }

#ifdef OPT_DGT

/*
 * Algorithm 4: Single Digit General Test (Ref: https://hodoku.sourceforge.net/en/tech_last.php#ts)
 *
 * Board states per digit
 *
 * A band of 3 boxes has (in the horizontal or vertical drection) exactly 6 possible
 * placements (in terms of rows and boxes), with each placement 'somewhere' in its triad:
 *
 *  1  -  -   1 - -   - 1 -   - 1 -   - - 1   - - 1
 *  -  1  -   - - 1   1 - -   - - 1   1 - -   - 1 -
 *  -  -  1   - 1 -   - - 1   1 - -   - 1 -   1 - -
 *
 * We can take each 1/0 bit, and turn it into three bits for the possible positions in the triad.
 *
 * For the vertical band, the situation is exactly the same, imagine it mirrored
 * on the top-left to bottom right diagonal.
 *
 * Back to the band:
 * With 6 placements per band, the full board has 6x6x6 = 216 possible placements for
 * for a digit in horizontal bands.
 *
 * Now overlay a given horizontal placement with a given vertical placement, expressed
 * as full bit maps, and you have one precise and valid configuration for one digit.
 *
 * Therefore there are 216x216 = 46656 possible different placements (patterns).
 *
 * But it gets even better.  We can from the Sudoku map of candidates determine the
 * variations of the 216 possible horizontal placements that might fit.
 * That is a subset of X{216} bitmaps.
 *
 * And for the vertical case, we can do the same.  A subset of Y{216}.
 *
 * Then we iterate over both subsets to obtain X{216} x Y{216}, ie. the cross product
 * of X and Y, that is the full set of all possible placements that can fit the given
 * candidates.
 *
 * This is already very cool, since in the progress of a solution, X and Y shrink substantially
 * over time, often to 24, 12, 8 or less.  We can choose when to strike! for a cross product of
 * 144 or less possibilities.
 *
 * We then compare these 144 (or less) placement maps to check a few things:
 * Is the placement map covered by the candidate/resolved bitmap for the digit?
 * If not, discard that placement map.
 * For all placement maps that fit, build their union bitmap.
 * (if none fit, we know it's a backtrack situation)
 * This union is then compared to the candidate bitmap and magically, all misfits
 * in the candidate map are exposed as expandable.  So delete those.
 *
 * What is expected to be discarded in that manner? Every pattern that works on the digit
 * bitmap only, i.e. fishes, finned and sashimi fishes, skyscrapers, two-string kites and
 * cranes, empty rectangles, X-chains and sometimes simply fully resolving the placement
 * pattern to that of the solution.
 *
 * So we put down the 216 horizontal and vertical placement maps and go find out
 * for a given digit and its candidate bitmap, which maps apply.  Easy enough, but how
 * EXACTLY does that work?  Another piece of magic is required, the lookup!
 * If we start with giving the triads of a band a bit designation, we obtain nine bits.
 * For these nine bits we can look up the required placements to cover it, as there are
 * only nine bits of input for a set of 512 possible configurations.
 * Most of these immediately reduce to a reduction of locked candidates, so that could
 * ACTUALLY REPLACE the current triad processing for that purpose.
 * (Note: this turns out to be just a bit slower than the current algorithm, which parallelizes
 * the effort across all digits!)
 *
 * For example, the input of:
 *   1 0 1                     1 0 0
 *   0 1 1  can be reduced to: 0 0 1 
 *   0 1 0                     0 1 0
 * with the digit eliminated in the top right triad and the middle triad.
 * The remaining input is already minimal, as it corresponds to exactly one possible config.
 * More often, there will be something like:
 *   1 0 0
 *   0 1 1
 *   0 1 1
 * which corresponds to 2 possible placements (for each of the diagonals of the lower right
 * quadrant of four triads:
 *   1 0 0    1 0 0
 *   0 1 0    0 0 1
 *   0 0 1    0 1 0
 * Rince and repeat for the next two bands for the full horizontal configurations.
 * Rince and repeat for the vertical configuration.
 * Last example:
 * This Input
 *   1 0 1
 *   1 1 1
 *   1 1 1
 * is covered by the following four possible placements:
 *   1 0 0    1 0 0    0 0 1     0 0 1
 *   0 1 0    0 0 1    0 1 0 and 1 0 0
 *   0 0 1    0 1 0    1 0 0     0 1 0
 *
 * This bitmap lookup serves to identify the applicable patterns.
 *
 * It is actually very easy to visually asses a Sudoku board with candidates for the
 * possible placements and the combined complexity.
 *
 * The tables: h/v2/3bandplacement give us in combination all possible placements
 * of 9 digits on the Sudoku Board that are valid in a Sudoku solution.
 *
 * The table bandredux for a given band and digit gives the information of which
 * placement patterns are possible.
 *
 */

    // This optimization limits the digit procession in two ways:
    // allowing only one elimination per digit (for a total of 9)
    // if the complexity cutoff is high, then disable the first constraint, as multiple
    // eliminations become very likely.  modes d6, d7 and d8 are therefore increasingly costly.
    if ( mode_dgt && !(dgt_tune_for_speed && (grid_state->unlocked.u32[3] == 0x1ff) ) ) {
        /*
         * Part 1
         * process the triad information and prepare the bandplacements array.
         */
        
        __m256i c1 = *(__m256i_u*)triad_info.row_triads_canonical;
        __m256i c2 = *(__m256i_u*)(triad_info.row_triads_canonical+16);

        // The first mask is for digit 9:
        unsigned int m = _mm256_movemask_epi8(_mm256_permute4x64_epi64(_mm256_packus_epi16(_mm256_srli_epi16(c1,1),_mm256_srli_epi16(c2,1)), 0xD8));
        __m256i cc = _mm256_permute4x64_epi64(_mm256_packus_epi16(_mm256_and_si256(c1, maskff), _mm256_and_si256(c2, maskff)), 0xD8);


        for (unsigned short type=0; type < 2; type++) {
            // for each digit d+1 (in order 9 to 1):
            for (signed char d = 8; d >= 0; d--) {
                // check three bands bi:
                for ( unsigned bi=0; bi < 3; bi++, m >>= 9 ) {
                    // save band config for later:
                    solverData.bandplacements[d].u8[bi+type*4] = bandredux[m&0x1ff].config;
                }
                m = _mm256_movemask_epi8(cc);
                cc = _mm256_slli_epi16(cc,1);
            }
            if ( type == 1 ) {
                break;
            }
            c1 = *(__m256i_u*)triad_info.col_triads;
            c2 = *(__m256i_u*)(triad_info.col_triads+16);

            // since triad_info.col_triads is not in canonical order, the triad values
            // need to be moved into the right slots:

            __m256i cc_m = _mm256_packus_epi16(_mm256_srli_epi16(c1,1),_mm256_srli_epi16(c2,1));
            __m256i tmp1 = _mm256_shuffle_epi8(cc_m, shuff_col_ordered1);
            cc_m = _mm256_shuffle_epi8(cc_m, shuff_col_ordered2);
            cc_m = _mm256_or_si256(tmp1,_mm256_permute2x128_si256(cc_m, cc_m, 1));
            // digits 8-0
            cc = _mm256_packus_epi16(_mm256_and_si256(c1, maskff), _mm256_and_si256(c2, maskff));
            __m256i tmp2 = _mm256_shuffle_epi8(cc, shuff_col_ordered1);
            cc = _mm256_shuffle_epi8(cc, shuff_col_ordered2);
            cc = _mm256_or_si256(tmp2,_mm256_permute2x128_si256(cc, cc, 1));
            // first mask for digit 9
            m = _mm256_movemask_epi8(cc_m);
        }

        // Algorithm 4: Single Digit General Test
        //
        // Part2: The examination of the placement data:
        bit128_t *candidate_bits_by_value = solverData.getCbbvs(candidates);

        if ( grid_state->unlocked.u32[3] != 0x1ff ) {
        // This round-robin approach saves around 2-3% of tests,
        // getting a little faster to the next hit

// ignore a warning "maybe-uninitialized" since next_dgt is initialized on declaration:
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
            unsigned int last_dgt = next_dgt+9;
            unsigned int d;
            for ( ; next_dgt<last_dgt; next_dgt++) {
                d=index_table_dgt[next_dgt];
#pragma GCC diagnostic pop
                if ( dgt_tune_for_speed && grid_state->unlocked.check_indexbit<96>(d) ) {
                    continue;
                }

                // digit bitmap
                __m128i digit_bits = candidate_bits_by_value[d].m128;
                __m128i pl;         // a single (valid) placement
                __m128i pl_h, pl_v; // the horizontal and vertical parts being constructed for
                                    // a full placement.
                __m128i pl_h3;      // the 3 horizontal placement pieces united.
                __m128i plsum {};   // the union of all valid placements
                unsigned short plcnt = 0;  // a count of all the valid placements
                BandPlacements *bandplacements = &solverData.bandplacements[d];
                int complexity =   __builtin_popcount(bandplacements->u8[0])
                                 * __builtin_popcount(bandplacements->u8[1])
                                 * __builtin_popcount(bandplacements->u8[2])
                                 * __builtin_popcount(bandplacements->u8[4])
                                 * __builtin_popcount(bandplacements->u8[5])
                                 * __builtin_popcount(bandplacements->u8[6]);
                // evaluate complexity coutoffs:
    
                if ( complexity > dgt_complexity_cutoff || complexity <= 13 ) {
                    continue;
                }
                grid_state->unlocked.set_indexbit<96>(d);

                counters.single_digit_placement_checks++;
                unsigned int h1 = bandplacements->u8[0];
                unsigned int h2;
                unsigned int h3;
                unsigned int v1;
                unsigned int v2;
                unsigned int v3;
                unsigned int h1pl;
                unsigned int h2pl;
                unsigned int h3pl;
                unsigned int v1pl;
                unsigned int v2pl;
                unsigned int v3pl;
                while (h1) {
                    h1pl = tzcnt_and_mask(h1);
                    h2 = bandplacements->u8[1];
                    while (h2) {
                        h2pl = tzcnt_and_mask(h2);
                        pl_h = h2bandplacement[h1pl + h2pl*6];
                        v1 = bandplacements->u8[4];
                        while (v1) {
                            v1pl = tzcnt_and_mask(v1);
                            v2 = bandplacements->u8[5];
                            while (v2) {
                                v2pl = tzcnt_and_mask(v2);
                                pl_v = v2bandplacement[v1pl + v2pl*6];
                                pl = pl_h & pl_v;
                                if ( _mm_testc_si128(digit_bits, pl) ) { // valid h/v combo
                                    h3 = bandplacements->u8[2];
                                    while (h3) {
                                        h3pl = tzcnt_and_mask(h3);
                                        pl_h3 = pl_h | h3bandplacement[h3pl];
                                        v3 = bandplacements->u8[6];
                                        while (v3) {
                                            v3pl = tzcnt_and_mask(v3);
                                            pl = pl_h3 & (pl_v | v3bandplacement[v3pl]);
                                            if ( _mm_testc_si128( digit_bits, pl) ) { // valid h/v combo
                                                plsum = plsum | pl;
                                                plcnt++;
                                            }
                                        }                               
                                    }
                                }                            
                            }               
                        }               
                    }
                }
                if ( !_mm_testz_si128( plsum, plsum) ) { // have a valid placement combination
                    // determine extra candidates not required
                    __m128i spares = ~plsum & digit_bits;
                    if ( !_mm_testz_si128( spares, spares) ) {
#if 0
                        // determine whether all spares are unlocked (otherwise they conflict with locked candidates)
                        if ( !_mm_testc_si128( grid_state->unlocked, spares) ) {
                            if ( verbose == VDebug ) {
                                printf("back track - placement of digit %d candidates is impossible.\n", d+1);
                            }
                            goto back;
                        }
#endif
                        counters.single_digit_incompatibilities++;
                        // and actually remove these candidates:
                        if ( verbose == VDebug ) {
                            printf("remove candidates for digit %d as they are inconsistent with a valid solution:\n\t", d+1);
                            bit128_t s { .m128 = spares };
                            while ( s ) {
                                printf(" %s", cl2txt[tzcnt_and_mask(s)]);
                            }
                            printf("\n");
                            if ( debug > 2 ) {
                                bit128_t s { .m128 = spares };
                                bit128_t dgts { .m128 = digit_bits };
                                show_dgt_placement(dgts, s, "removed candidates");
                                printf("\t* the candidates removed above most likely could represent resolution of any fish\n"
                                                  "\t  or skyscraper, kite or crane\n\t  other possibilities also exist\n");
                            }
                        }
                        bit128_t s { .m128 = spares };
                        grid_state->mask_candidates(s, 1<<d);

                        if ( plcnt == 1 ) {  // a single placement was identified!
                            bit128_t dgts { .m128 = plsum & grid_state->unlocked.m128 };
                            grid_state->mask_candidates<false>(dgts, 1<<d);
                            if ( verbose == VDebug ) {
                                bit128_t dgts { .m128 = plsum & grid_state->unlocked.m128 };
                                while ( dgts.u128 ) {
                                    printf("hidden single       %d at %s\n", d+1, cl2txt[tzcnt_and_mask(dgts)]);
                                }
                            }
                            grid_state->unlocked.m128 = ~plsum & grid_state->unlocked.m128;
                        }
                        next_dgt = (d+1)%9;
                        goto search;
                    }
                    if ( plcnt == 2 ) {
                        // if we get a set of two placements of sufficient size,
                        // this is a pretty good guess:
                        bit128_t dgts { .m128 = plsum & grid_state->unlocked.m128 & ~spares};
                        unsigned char score = dgts.popcount()/2;
                        // a score of 3 for 3 cells resolved is a reasonable heuristic for a minimum,
                        // as long as we still allow for an even better result.
                        // the guess is strictly 50/50 here.
                        if ( score >= 3 && score > solverData.guess_hint_score ) {
                            solverData.guess_hint_score = score;
                            solverData.guess_hint_index = tzcnt_and_mask(dgts);
                            solverData.guess_hint_digit = 1<<d;
                        }
                    }
                } else {
                    // backtrack, no coverage for the digit placements
                    if ( verbose == VDebug ) {
                        printf("back track - placement of digit %d candidates is impossible.\n", d+1);
                    }
                    goto back;
                }
            }
            next_dgt = d;
        }
    }
#endif

guess:
    // Make a guess if all that didn't work
    grid_state = grid_state->make_guess<verbose>(triad_info, bivalues, solverData);
    current_entered_count  = ((grid_state->stackpointer)<<8) | (81 - grid_state->unlocked.popcount());        // back to previous stack.
    no_guess_incr = 0;
    goto start;

}

} // namespace Schoku

#ifndef LIB_ONLY

void print_help() {
using namespace Schoku;

        printf("schoku version: %s\n", version_string);
        printf(R"(Synopsis:
schoku [options] [puzzles] [solutions]
	 [puzzles] names the input file with puzzles. Default is 'puzzles.txt'.
	 [solutions] names the output file with solutions. Default is 'solutions.txt'.

Command line options:
    -a  automatically turn off mode_dgt if ineffective. Initially sets mode_dgt.
        Statistical output may vary with size of puzzle file and when multithreading.
    -c  check for back tracking even when no guess was made (e.g. if puzzles might have no solution)
    -d# provide some detailed information on the progress of the puzzle solving.
        add a 2 or even 3 for even more detail.
    -h  help information (this text)
    -l# solve a single line from the puzzle.
    -m[D#] execution mode digit_placement, lower or upper case.
        D# the complexity threshold and speed tuning is set according to a lookup table.
           The possible values of '#' are:
               1  (72 + tuned for speed)
               2  (96 + speed)
               3 (144 + speed)
               4 (144)
               5 (192)
               6 (288)
               7 (384)
               8 (864)
               9 (unlimited)
           Higher values (complexity threshold in braces) deliver more hits
           while increasing the execution time.
    -r[ROM] puzzle rules (lower or upper case):
        R  for regular puzzles (unique solution exists)
           not suitable for puzzles that have multiple solutions
        O  find just one solution
           this will find the first of multiple solutions and stop looking
        M  determine whether multiple solutions exist beyond the first found
    -t# set the number of threads
    -v  verify the solution
    -w  display warnings (mostly unexpected solving details for regular puzzles)
    -x  provide some statistics
    -y  provide speed statistics only
    -#  by default, rows and columns are counted from 1
        -#0 change base for row and column reporting from 1 to 0
        -#A change base for row reporting to 'A' and column 1, e.g. cell[0,0] as A1
        -#r change to r_c_ format for row and column reporting, e.g. cell[0,0] as r1c1

)");
}

int main(int argc, const char *argv[]) {
using namespace Schoku;

    int line_to_solve = 0;

    int countpound = 0; // ignore multiple -# opts

#ifdef OPT_DGT
    const unsigned short complexity_cutoff_table[] = {
                          144, /* default */
                           72, 96, 144, 144, 192, 288, 384, 864, 0xffff
                        };
#endif
    if ( argc > 0 ) {
        argc--;
        argv++;
    }

    size_t outnpuzzles = 0;
	size_t npuzzles    = 0;
    size_t outnbytes   = 0;

    char opts[1024] = { 0 };
    size_t used_opts = 0;
    for (int i = 0; i < argc; i++) {
        int n = snprintf(opts + used_opts, sizeof(opts) - used_opts, "%s ", argv[i]);
        if (n < 0) {
            break;
        }
        used_opts += (size_t)n;
        if (used_opts >= sizeof(opts) - 1) {
            opts[sizeof(opts) - 1] = 0;
            break;
        }
    }

    while ( argc && argv[0][0] == '-' ) {
        if (argv[0][1] == 0) {
            argv++; argc--;
            break;
        }
        switch(argv[0][1]) {
        case 'a':
             mode_selftune = true;
             mode_dgt = true;
             break;
        case 'c':
             thorough_check=1;
             break;
        case 'd':
             debug=1;
             if ( argv[0][2] && isdigit(argv[0][2]) ) {
                 sscanf(&argv[0][2], "%d", &debug);
             }
             break;
        case 'h':
             print_help();
             exit(0);
             break;
        case 'l':    // line of puzzle to solve
             sscanf(argv[0]+2, "%d", &line_to_solve);
             break;
        case 'm':
             for ( unsigned char p=2; argv[0][p]; p++) {
                 switch (toupper(argv[0][p])) {
#ifdef OPT_DGT
                case 'D':        // see OPT_DGT
                    {
                        unsigned int nitems = 0;
                        unsigned int lngth = 0;

                        mode_dgt = true;
                        if ( argv[0][p+1] && isdigit(argv[0][p+1]) ) {
                            unsigned int complexity_cutoff_index = 0;
                            nitems = sscanf(&argv[0][p+1], "%d%n", &complexity_cutoff_index, &lngth);
                            if ( nitems == 1 ) {
                                p += lngth;
                                if ( complexity_cutoff_index > 0 && complexity_cutoff_index < sizeof(complexity_cutoff_table)/sizeof(unsigned short) ) {
                                    dgt_complexity_cutoff = complexity_cutoff_table[complexity_cutoff_index];
                                    if ( complexity_cutoff_index > 3 ) {
                                        dgt_tune_for_speed = false;
                                    }
                                    if ( debug != 0 || warnings != 0 ) {
                                        printf("complexity threshold set to %d\n", dgt_complexity_cutoff);
                                        printf("tune for speed %d\n", dgt_tune_for_speed);
                                    }
                                }
                            }
                        }
                        break;
                    }
#endif
                default:
                    printf("invalid mode %c\n", argv[0][p]);
                }
            }
            break;
        case 'r':    // rules
            if ( argv[0][2] ) {
                switch (toupper(argv[0][2])) {
                case 'R':        // defaul rules (fastest):
                                 // assume regular puzzle with a unique solution
                                 // not suitable for puzzles with multiple solutions
                    rules = Regular;
                    break;
                case 'O':        // find one solution without making assumptions
                    rules = FindOne;
                    break;
                case 'M':        // check for multiple solutions
                    rules = Multiple;
                    break;
                default:
                    printf("invalid puzzle rules option %c\n", argv[0][2]);
                    break;
                }
            }
            break;
        case 't':    // set number of threads
             if ( argv[0][2] && isdigit(argv[0][2]) ) {
                 sscanf(&argv[0][2], "%d", &numthreads);
                 if ( numthreads != 0 ) {
                     omp_set_num_threads(numthreads);
                 }
             }
             break;
        case 'v':    // verify
             verify=1;
             break;
        case 'w':    // display warnings
             warnings = 1;
             break;
        case 'x':    // stats output
             reportstats=1;
             break;
        case 'y':    // timing stats only
             reporttimings=1;
             break;
        case '#':    // row/col numbering base
             {
                 if ( countpound == 0 ) {
                     countpound++;
                     int displaybase = 0;
                     if ( argv[0][2] && isdigit(argv[0][2])
                          && ( argv[0][2] == '0' || argv[0][2] == '1' ) ) {
                         sscanf(&argv[0][2], "%1d", &displaybase);
                         for ( int i=0; i<81; i++ ) {
                             cl2txt[i][1] += displaybase-1;
                             cl2txt[i][3] += displaybase-1;
                         }
                         if ( debug ) {
                             printf("cell at [0,0] is displayed as %s etc.\n", cl2txt[0]);
                         }
                     } else if ( argv[0][2] && argv[0][2]=='A' ) {
                         for ( int i=0; i<81; i++ ) {
                             cl2txt[i][0] = cl2txt[i][1] + 'A'-'1';       // rows carry letters
                             cl2txt[i][1] = cl2txt[i][3];   // cols carry digits
                             cl2txt[i][2] = 0;
                         }
                         if ( debug ) {
                             printf("cell at [0,0] is displayed as %s etc.\n", cl2txt[0]);
                         }
                     } else if ( argv[0][2] && argv[0][2]=='r' ) {
                         for ( int i=0; i<81; i++ ) {
                             cl2txt[i][0] = 'r';
                             cl2txt[i][2] = 'c';
                             cl2txt[i][4] = 0;
                         }
                         if ( debug ) {
                             printf("cell at [0,0] is displayed as %s etc.\n", cl2txt[0]);
                         }
                     }
                 }
             }
             break;
        default:
             printf("invalid option: %s\n", argv[0]);
             break;
        }
        argc--, argv++;
    }

    mode_dgt_init = mode_dgt;  // preserve initial status of mode_dgt

    assert((sizeof(GridState) & 0x3f) == 0);

    // sort out the CPU and OMP settings

    if ( !__builtin_cpu_supports("avx2") ) {
        fprintf(stderr, "This program requires a CPU with the AVX2 instruction set.\n");
        exit(0);
     }
     // lacking BMI support? unlikely!
     if ( !__builtin_cpu_supports("bmi") ) {
         fprintf(stderr, "This program requires a CPU with the ABM and BMI instructions (such as blsi/tzcnt/popcnt/lzcnt)\n");
         exit(0);
     }

     bmi2_support = __builtin_cpu_supports("bmi2");
     pext_support = bmi2_support && !__builtin_cpu_is("znver2");

    if ( debug ) {
         omp_set_num_threads(1);
         printf("debug mode requires restriction of the number of threads to 1\n");

         printf("BMI2 instructions %s %s\n",
               bmi2_support ? "found" : "not found",
               pext_support? "and enabled" : __builtin_cpu_is("znver2")? "but use of pdep/pext instructions disabled":"");
    }

	auto starttime = std::chrono::steady_clock::now();

	const char *ifn = argc > 0? argv[0] : "puzzles.txt";
	int fdin = open(ifn, O_RDONLY);
	if ( fdin == -1 ) {
		if (errno ) {
			fprintf(stderr, "Error: Failed to open file %s: %s\n", ifn, strerror(errno));
			exit(0);
		}
	}

    // get size of file
	struct stat sb;
	fstat(fdin, &sb);
    size_t fsize = sb.st_size;

	// map the input file
    signed char *string = (signed char *)mmap((void*)0, fsize, PROT_READ, MAP_PRIVATE, fdin, 0);
	if ( string == MAP_FAILED ) {
		if (errno ) {
			fprintf(stderr, "Error: mmap of input file %s: %s\n", ifn, strerror(errno));
			exit(0);
		}
	}
	close(fdin);

	// skip lines, until hitting a puzzle.
    size_t pre = 0;
    while ( !(isdigit((int)string[pre]) || string[pre] == '.') || !(string[pre+81] == 10 || (string[pre+81] == 13 && (string[pre+82] == 10))) ) {
	    while (string[pre] != 10) {
    	    ++pre;
    	}
    	++pre;
	}
    if ( string[pre+81] == 13 ) {
        fprintf(stderr, "Error: input file line ending in CR/LF\n");
		exit(0);
    }

    size_t post = 1;
	if ( string[fsize-1] != 10 )
		post = 0;

	// get and check the number of puzzles
	npuzzles    = (fsize - pre + (1-post))/82;
    outnpuzzles = line_to_solve ? 1 : npuzzles;
    outnbytes   = outnpuzzles * 164;

    if ( line_to_solve < 0 || npuzzles < (unsigned long)line_to_solve ) {
		fprintf(stderr, "Ignoring the given line number\nthe input file %s contains %ld puzzles, the given line number %d is not between 1 and %ld\n", ifn, npuzzles, line_to_solve, npuzzles);
        line_to_solve = 0;
    }

	if ( (fsize -pre -post + 1) % 82 ) {
		fprintf(stderr, "found %ld puzzles with %ld(start)+%ld(end) extra characters\n", (fsize - pre - post + 1)/82, pre, post);
	}

	const char *ofn = argc > 1? argv[1] : "solutions.txt";
	int fdout = open(ofn, O_RDWR|O_CREAT, 0775);
	if ( fdout == -1 ) {
		if (errno ) {
			fprintf(stderr, "Error: opening output file %s: %s\n", ofn, strerror(errno));
			exit(0);
		}
	}
    if ( ftruncate(fdout, (size_t)outnbytes) == -1 ) {
		if (errno ) {
			fprintf(stderr, "Error: setting size (ftruncate) on output file %s: %s\n", ofn, strerror(errno));
		}
		exit(0);
	}

	// map the output file
    output = (signed char *)mmap((void*)0, outnbytes, PROT_WRITE, MAP_SHARED, fdout, 0);
	if ( output == MAP_FAILED ) {
		if (errno ) {
			printf("Error mmap of output file %s: %s\n", ofn, strerror(errno));
			exit(0);
		}
	}
	close(fdout);

    // solve all sudokus and prepare output file
    size_t imax = npuzzles*82;

    signed char *string_pre = string+pre;
	GridState *stack = 0;

    if ( line_to_solve ) {
        size_t i = (line_to_solve-1)*82;
        char *gridin  = (char *)output;
        char *gridout = gridin+82;
        // copy unsolved grid
        memcpy(gridin, &string_pre[i], 81);
        // add comma and newline in right place
        output[81] = ',';
        output[163] = 10;
        // solve the grid in place
        if ( stack == 0 ) {
            // force alignment the 'old-fashioned' way (no intent of releasing it)
            stack = (GridState*) (~0x3fll & ((unsigned long long) malloc(sizeof(GridState)*GRIDSTATE_MAX+0x40)+0x40));
            stack->stackpointer=0;
        }

        if ( debug != 0) {
            solve<VDebug>(gridin, gridout, stack, line_to_solve, global_counters);
        } else if ( reportstats !=0) {
            solve<VStats>(gridin, gridout, stack, line_to_solve, global_counters);
        } else {
            solve<VNone>(gridin, gridout, stack, line_to_solve, global_counters);
        }
    } else {

        int chunk_size = 120;
        // The OMP directives:
        // proc_bind(close): high preferance for thread/core affinity
        // firstprivate(stack): the stack is allocated for each thread once and seperately
        // schedule(dynamic,64): 64 puzzles are allocated at a time and these chunks
        //   are assigned dynmically (to minimize random effects of difficult puzzles)
        // shared(...) lists the variables that are shared (as opposed to separate copies per thread)
        // reduction(...) allows to aggregate results from all threads using a single declaration of the aggregate
        // declare reduction defines a reduction constructs (name, data type, operation)

#pragma omp declare reduction (counters_reduction : Counters : omp_out += omp_in) \
            initializer(omp_priv = Counters())

        int chunk_counter=0; // private
#pragma omp parallel reduction(counters_reduction:global_counters) firstprivate(stack) proc_bind(close) shared(string_pre, output, npuzzles, imax, debug, reportstats)
        {
            // force alignment the 'old-fashioned' way
            // not going to free the data ever
            // stack = (GridState*)malloc(sizeof(GridState)*GRIDSTATE_MAX);
            stack = (GridState*) (~0x3fll & ((unsigned long long) malloc(sizeof(GridState)*GRIDSTATE_MAX+0x40)+0x40));

#pragma omp for schedule(monotonic:dynamic,120)
            for (size_t i = 0; i < imax; i+=82) {
                // copy unsolved grid
                char *gridin = (char *)&output[i*2];
                char *gridout = (char *)&output[i*2+82];
                memcpy(gridin, &string_pre[i], 81);
                // add comma and newline in right place
                output[i*2 + 81] = ',';
                output[i*2 + 163] = 10;
                if ( debug != 0) {
                    solve<VDebug>(gridin, gridout, stack, i/82+1, global_counters);
                } else if ( reportstats !=0) {
                    solve<VStats>(gridin, gridout, stack, i/82+1, global_counters);
                } else {
                    solve<VNone>(gridin, gridout, stack, i/82+1, global_counters);
                }
                {
                    // selftuning
                    if ( omp_get_thread_num()==0 && mode_selftune ) {
                        if ( (i/82+1)%chunk_size == 0 ) {
                            chunk_counter++;
                            if ( chunk_counter == 6 ) {
                                // if there are few guesses, keep the score threshold at 3 (favoring stats).
                                // if there are many guesses, lower the score threshold to two (favoring speed).
                                // note that when multi-threaded, threads could possibly compete here.
                                double guess_per_puzzle = (double)(global_counters.guesses-global_counters.last_guesses)/(5*chunk_size);
                                if ( guess_score_threshold < 3 && guess_per_puzzle < 3 ) {
                                    guess_score_threshold = 3;
                                } else if ( guess_score_threshold >= 3 && guess_per_puzzle > 3 ) {
                                    guess_score_threshold = 2;
                                }
                                // this may be noticable in the output
                                double productivity = guess_per_puzzle * (double)global_counters.single_digit_incompatibilities/global_counters.single_digit_placement_checks;
                                if ( mode_dgt ) {
                                    if ( productivity < 0.03 ) {
                                        mode_dgt = false;
                                    }
                                }
                                global_counters.last_guesses = global_counters.guesses;
                            }
                        }
                    }
                }
            } // omp for
        } // omp parallel
    } // if

	int err = munmap(string, fsize);
	if ( err == -1 ) {
		if (errno ) {
			fprintf(stderr, "Error: munmap file %s: %s\n", ifn, strerror(errno));
		}
	}
	err = munmap(output, (size_t)outnbytes);
	if ( err == -1 ) {
		if (errno ) {
			fprintf(stderr, "Error: munmap file %s: %s\n", ofn, strerror(errno));
		}
	}

    if ( reportstats) {
        long long duration = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration(std::chrono::steady_clock::now() - starttime)).count();
        printf("schoku version: %s\ncommand options: %s\ncompile options: %s\n", version_string, opts, compilation_options);
        printf("%10ld  %6.1f/puzzle  puzzles entered and presets\n", npuzzles, (double)global_counters.preset_count/outnpuzzles);
        printf("%10ld  %.0lf/s  puzzles solved\n", global_counters.solved_count, (double)global_counters.solved_count/((double)duration/1000000000LL));
        if ( duration/outnpuzzles > 1000 ) {
            printf("%8.1lfms  %6.2lf\u00b5s/puzzle  solving time\n", (double)duration/1000000, (double)duration/(outnpuzzles*1000LL));
        } else {
            printf("%8.1lfms  %4dns/puzzle  solving time\n", (double)duration/1000000, (int)((double)duration/outnpuzzles));
        }
        if ( global_counters.unsolved_count && rules != Regular) {
            printf("%10ld  puzzles had no solution\n", global_counters.unsolved_count);
        }
        if ( rules == Multiple ) {
            printf("%10ld  puzzles had multiple solutions\n", global_counters.non_unique_count);
        }
        if ( verify ) {
            printf("%10ld  puzzle solutions were verified\n", global_counters.verified_count);
        }
        printf( "%10ld  %6.2f%%  puzzles solved without guessing\n", global_counters.no_guess_cnt, (double)global_counters.no_guess_cnt/global_counters.solved_count*100);
        printf( "%10ld  %6.2f/puzzle  guesses\n", global_counters.guesses, (double)global_counters.guesses/(double)global_counters.solved_count);
        printf( "%10ld  %6.2f/puzzle  back tracks\n", global_counters.trackbacks, (double)global_counters.trackbacks/global_counters.solved_count);
        printf("%10lld  %6.2f/puzzle  digits entered and retracted\n", global_counters.digits_entered_and_retracted, (double)global_counters.digits_entered_and_retracted/global_counters.solved_count);
        printf("%10lld  %6.2f/puzzle  'rounds'\n", global_counters.past_naked_count, (double)global_counters.past_naked_count/global_counters.solved_count);
        printf("%10lld  %6.2f/puzzle  triads resolved\n", global_counters.triads_resolved, (double)global_counters.triads_resolved/global_counters.solved_count);
        printf("%10lld  %6.2f/puzzle  triad updates\n", global_counters.triad_updates, (double)global_counters.triad_updates/global_counters.solved_count);
#ifdef OPT_DGT
        if ( mode_dgt || mode_dgt_init || global_counters.single_digit_incompatibilities > 0 ) {
            printf("%10lld  %6.2f/puzzle  single digit placement compatibility tests\n", global_counters.single_digit_placement_checks, (double)global_counters.single_digit_placement_checks/global_counters.solved_count);
            printf("%10lld  %6.2f/puzzle  single digit placement incompatibilities found\n", global_counters.single_digit_incompatibilities, (double)global_counters.single_digit_incompatibilities/global_counters.solved_count);
        }
#endif
        if ( global_counters.bug_plus1_count ) {
            printf("%10ld  bi-value universal graves avoided (BUG+1)\n", global_counters.bug_plus1_count);
        }
        if ( global_counters.bug_count ) {
            printf("%10ld  bi-value universal graves detected\n", global_counters.bug_count);
        }
        if ( global_counters.no_bivals_count ) {
            printf("%10ld  board states without bivalues\n", global_counters.no_bivals_count);
        }
    } else if ( reporttimings ) {
        long long duration = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration(std::chrono::steady_clock::now() - starttime)).count();
        if ( duration/npuzzles > 1000 ) {
            printf("%8.1lfms  %6.2lf\u00b5s/puzzle  solving time\n", (double)duration/1000000, (double)duration/(npuzzles*1000LL));
        } else {
            printf("%8.1lfms  %4dns/puzzle  solving time\n", (double)duration/1000000, (int)((double)duration/npuzzles));
        }
    }

    if ( !reportstats && rules == Multiple && global_counters.non_unique_count) {
        printf("%10ld  puzzles had more than one solution\n", global_counters.non_unique_count);
    }
    if ( verify && global_counters.not_verified_count) {
        printf("%10ld  puzzle solutions verified as not correct\n", global_counters.not_verified_count);
    }
    if ( !reportstats && global_counters.unsolved_count) {
        printf("%10ld puzzles had no solution\n", global_counters.unsolved_count);
    }
    if ( rules == Regular && (reportstats || warnings) && ( global_counters.unsolved_count || global_counters.non_unique_count || global_counters.not_verified_count) ) {
        printf("\n\tIf a puzzle may have multiple solutions use either\n\t -ro (find one solution) or -rm (check for multiple solutions)!\n");
    }

    return 0;
}

#else

namespace Schoku {
	GridState *stack = 0;
    int line_counter = 0;
    bool initialized = false;
    long guess_state;
    std::chrono::time_point<std::chrono::steady_clock> starttime2;
}

void initializeSchoku() {
    using namespace Schoku;

    bmi2_support = __builtin_cpu_supports("bmi2");
    pext_support = bmi2_support && !__builtin_cpu_is("znver2");
    stack = (GridState*) (~0x3fll & ((unsigned long long) malloc(sizeof(GridState)*GRIDSTATE_MAX+0x40)+0x40));
    stack->stackpointer = 0;
    initialized = true;
    mode_dgt = true;
    global_counters.guesses = 0;
    global_counters.single_digit_placement_checks = 0;
    global_counters.single_digit_incompatibilities = 0;
    guess_state = 0;
    line_counter = 0;
}

// library / call interface for invoking the Schoku solver,
// using tdoku benchmark conventions.
// This unfortunately precludes using multiple threads...

extern "C"
size_t OtherSolverSchoku(const char *input, size_t limit, uint32_t /*unused_configuration*/,
                             char *solution, size_t *num_guesses) {

    using namespace Schoku;

    if ( !initialized ) {
        starttime2 = std::chrono::steady_clock::now();
        initializeSchoku();
    }
    rules = (limit <= 1)? FindOne:Multiple;
    line_counter++;
    // Since 'solve' is called directly, inject tuning normally done as part of the OpenMP section here:
    if ( line_counter%600 == 0 ) {
        // detect when the benchmark runner cycles back to us and assume a different data set.
        long long duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::duration(std::chrono::steady_clock::now() - starttime2)).count();
        // By measuring the interval between invocations, detect when the benchmark set
        // switches and reinitialize.
        if ( duration > 500 ) {  // ms
            initializeSchoku();
        } else if ( line_counter==600 ) {
            // The effort put into guesses improves guessing, but also decreases performance.
            // When the guessing is high anyhow, tune down the effort for it.
            double guess_per_puzzle = (double)(global_counters.guesses-global_counters.last_guesses)/600;
            if ( guess_score_threshold < 3 && guess_per_puzzle < 3 ) {
                guess_score_threshold = 3;
            } else if ( guess_score_threshold >= 3 && guess_per_puzzle > 3 ) {
                guess_score_threshold = 2;
            }
#ifdef OPT_DGT
            // mode_dgt is a gamechanger, but only if the puzzles are difficult or very difficult.
            // When the level of guessing is very low, or the algorithm is ineffective,
            // turn mode_dgt off.
            if ( mode_dgt ) {
                guess_per_puzzle = (double)global_counters.guesses/600;
                double productivity;
                if ( global_counters.single_digit_incompatibilities == 0 ) {
                    productivity = 0.;   // avoid div by 0
                } else {
                    productivity = guess_per_puzzle * (double)global_counters.single_digit_incompatibilities/global_counters.single_digit_placement_checks;
                }
                if ( productivity < 0.03 ) {
                    mode_dgt = false;
                }
            }
#endif
        }
        global_counters.last_guesses = global_counters.guesses;
        starttime2 = std::chrono::steady_clock::now();
    }
    Status status = solve<VNone>((char*)input, solution, stack, line_counter, global_counters);
    long guesses_now = global_counters.guesses;
    *num_guesses = guesses_now - guess_state;
    guess_state = guesses_now;
    if ( status.solved == false ) {
        return 0;
    }
    return status.unique?1:2;
}

#endif