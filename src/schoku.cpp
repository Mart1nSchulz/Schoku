// This code uses AVX2 instructions...
/*
 * Schoku
 *
 * A high speed sudoku solver by M. Schulz
 *
 * Copyright 2024 Martin Schulz
 *
 * This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 * You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
 *
 * Based on the sudoku solver by Mirage ( https://codegolf.stackexchange.com/users/106606/mirage )
 * at https://codegolf.stackexchange.com/questions/190727/the-fastest-sudoku-solver
 * on Sep 22, 2021
 *
 * Version 0.9.2
 *
 * Performance changes:
 * - further optimized naked and hidden searches.
 * - moved __m256i constants to the program level
 * - AVX2 implementation for populating candidates and writing out the solution
 * - verbose is now a 3-state enum to optimize template instantiations
 * - increased detection of fishes (combining pairs to triples)
 *   with fishes and naked subsets (-mfs), attains 92.53% of solves without guessing
 *   on the 17-clue test set.
 * - cleanup and improvement to naked sets detection with minor drop in overall impact
 *   of naked set detection but important reduction of sets searched.
 *
 * Functional changes:
 * - added algorithm for (avoidable) unique rectangles.
 * - added algorithm for fishes and finned/sashimi fishes.
 * - adjustments for more diversified puzzle sources (tdoku puzzle files)
 * - warnings option (-w): by default the messages and warnings are muted.
 *   With -w pertinent messages (mostly for regular rules 'surprises') are shown.
 * - mode options (-m[FSTU]*) to dynamically invoke features.
 *   Previously these features were only available as compile time options.
 * - rules options (-r[ROM]) to either assume regular puzzles rules (single solution),
 *   search for one solution, even if multiple solutions exist, or determine whether
 *   two or more solutions exist.
 * - increased the maximum number of Gridstate structs to 34, as one puzzle set needed
 *   more than 28.
 * - allow for multi-line comments at the start of puzzle files
 * - disallow CR/LF (MS-DOS/Windows) line endings
 * - fixed problem with reporting a large number of solved puzzles when each solution
 *   was counted.
 * - fixed problem with bi-value universal grave: multiple solutions were not recognized.
 * - added option -y for speed information only
 * - added a bare bones debug facility (controlled via environment variable SCHOKU_DBG_FILTER).
 * - pulled stack assignment out of omp for loop in main.
 * - added/implemented a functional addendum to sashimi-pairs (extra candidate eliminations)
 *
 * Performance measurement and statistics:
 *
 * data: 17-clue sudoku (49151 puzzles)
 * CPU:  Ryzen 7 4700U
 *
 * schoku version: 0.9.2
 * command options: -x
 * compile options: OPT_TRIAD_RES OPT_SETS OPT_FSH OPT_UQR
 *      49151  puzzles entered
 *      49151  2197675/s  puzzles solved
 *     22.4ms    0.46µs/puzzle  solving time
 *      37834   76.98%  puzzles solved without guessing
 *      30798    0.63/puzzle  guesses
 *      20996    0.43/puzzle  back tracks
 *     273717    5.57/puzzle  digits entered and retracted
 *    1465686   29.82/puzzle  'rounds'
 *      22972    0.47/puzzle  triads resolved
 *     197772    4.02/puzzle  triad updates
 *        693  bi-value universal graves detected

 * command options: -x -mfs
 * compile options: OPT_TRIAD_RES OPT_SETS OPT_FSH OPT_UQR
 *      49151  puzzles entered
 *      49151  2003530/s  puzzles solved
 *     24.5ms    0.50µs/puzzle  solving time
 *      45481   92.53%  puzzles solved without guessing
 *       5347    0.11/puzzle  guesses
 *       2914    0.06/puzzle  back tracks
 *      39808    0.81/puzzle  digits entered and retracted
 *    1392437   28.33/puzzle  'rounds'
 *      21778    0.44/puzzle  triads resolved
 *     188569    3.84/puzzle  triad updates
 *      25496    0.52/puzzle  naked sets found
 *     578785   11.78/puzzle  naked sets searched
 *       7868    0.16/puzzle  fishes updated
 *      74377    1.51/puzzle  fishes detected
 *        977  bi-value universal graves detected
 *
 */
#include <atomic>
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
#include <intrin.h>
#include <sys/stat.h>
#include <sys/mman.h>

namespace Schoku {

const char *version_string = "0.9.2";

const char *compilation_options =
#ifdef OPT_TRIAD_RES
// Triad resolution examines all triads and if exactly three candidates are present
// marks them as sets. A tiny penalty to speed but a boost to statistics overall.
//
"OPT_TRIAD_RES "
#endif
#ifdef OPT_SETS
// Naked sets detection is a main feature.  The complement of naked sets are hidden sets,
// which are labeled as such when they are more concise to report.
//
"OPT_SETS "
#endif
#ifdef OPT_FSH
// Detection of fishes ( X-wing, sword fish, jellyfish and squirmbag) and
// finned/sashimi extensions is a main feature.
//
"OPT_FSH "
#endif
#ifdef OPT_UQR
// Detection of unique (avoidable) rectangles is a specialty feature.
//
"OPT_UQR "
#endif
""
;

// using type v16us for the built-in vector of unsigned short[16]
using v16us  = __v16hu;
using v8us   = __v8hu;
using v16usb = __v16hu;	// for boolean representation
using v8usb  = __v8hu;	// for boolean representation

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
   bool guess = false;
   bool used_assumed_uniqueness = false;
} Status;

enum Verbosity {
   VNone  = 0,
   VStats = 1,
   VDebug = 2
};

// bit128_t type
// used for all 81-bit fields to support different access patterns.
typedef
union bit128_t {
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

    inline __uint128_t operator ^ (const __uint128_t b) {
        return this->u128 ^ b;
    }

    inline __uint128_t operator & (const __uint128_t b) {
        return this->u128 & b;
    }

    inline bool check_indexbit(unsigned char idx) {
        return this->u8[idx>>3] & (1<<(idx & 0x7));
    }
    inline bool check_and_mask_index(unsigned char idx) {
        return _bittestandreset64((long long int *)&this->u64[idx>>6], idx & 0x3f);
    }
    inline bool check_indexbits(unsigned int bits, unsigned char pos) {
        return (this->u128>>pos) & bits;
    }
    inline void set_indexbit(unsigned char idx) {
        this->u8[idx>>3] |= 1<<(idx & 0x7);
    }
    inline void unset_indexbit(unsigned char idx) {
        this->u8[idx>>3] &= ~(1<<(idx & 0x7));
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
    // bitcount <= 64
    inline unsigned long long get_indexbits(unsigned char pos, unsigned char bitcount) {
        unsigned long long res = 0;
        if ( pos < 64 ) {
            res = u64[0] >> pos;
            if ( pos + bitcount >= 64 ) {
                res |= u64[1] << (64 - pos);
            }
        } else {
            res = u64[1] >> (pos-64);
        }
        // clip result
        if ( bitcount < 64 ) {
            res &= ((1LL<<bitcount)-1);
        }
        return res;
    }
    inline unsigned char popcount() {
        return _popcnt64(u64[0]) + _popcnt32(u32[2]);
    }
} bit128_t;

alignas(64)
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

const unsigned char transposed_cell[81] = {
    0,  9, 18, 27, 36, 45, 54, 63, 72,
    1, 10, 19, 28, 37, 46, 55, 64, 73,
    2, 11, 20, 29, 38, 47, 56, 65, 74,
    3, 12, 21, 30, 39, 48, 57, 66, 75,
    4, 13, 22, 31, 40, 49, 58, 67, 76,
    5, 14, 23, 32, 41, 50, 59, 68, 77,
    6, 15, 24, 33, 42, 51, 60, 69, 78,
    7, 16, 25, 34, 43, 52, 61, 70, 79,
    8, 17, 26, 35, 44, 53, 62, 71, 80
};

const long long unsigned int altbits[2][2] = { 0x5555555555555555u, 0x5555555555555555u,
                                      0xaaaaaaaaaaaaaaaau, 0xaaaaaaaaaaaaaaaau };

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
// heavily used
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

alignas(64)
const bit128_t box_bitmasks[9] = {
    ((bit128_t*)small_index_lut[0][Box])->u128,
    ((bit128_t*)small_index_lut[1][Box])->u128,
    ((bit128_t*)small_index_lut[2][Box])->u128,
    ((bit128_t*)small_index_lut[3][Box])->u128,
    ((bit128_t*)small_index_lut[4][Box])->u128,
    ((bit128_t*)small_index_lut[5][Box])->u128,
    ((bit128_t*)small_index_lut[6][Box])->u128,
    ((bit128_t*)small_index_lut[7][Box])->u128,
    ((bit128_t*)small_index_lut[8][Box])->u128
};

    // Mapping of the row triads processing order to canonical order (not considering the
    // gaps every 10 elements):
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

    // find the 3-bit positional pattern given the 'band' index 0..8
    const unsigned int bandbits_by_index[9] = {
        0x7, 0x7<<3, 0x7<<6, 0x7<<9, 0x7<<12, 0x7<<15, 0x7<<18, 0x7<<21, 0x7<<24 };

const unsigned char *row_index = index_by_kind[Row];
const unsigned char *column_index = index_by_kind[Col];
const unsigned char *box_index = index_by_kind[Box];

alignas(64)
// this table provides the bit masks corresponding to each index and each Kind of section.
// The 4th column contains all Kind's or'ed together.
// Heavily used.
// Casually speaking, this table provides the 'visibility' from each cell onto the grid
// for selected sections and all of them.
// Also consider that each pair of unsigned long long can be casted in several manners,
// most generally bit128_t.
//
const unsigned long long big_index_lut[81][4][2] = {
{{              0x1ff,        0x0 }, { 0x8040201008040201,      0x100 }, {           0x1c0e07,        0x0 }, { 0x80402010081c0fff,      0x100 }},
{{              0x1ff,        0x0 }, {   0x80402010080402,      0x201 }, {           0x1c0e07,        0x0 }, {   0x804020101c0fff,      0x201 }},
{{              0x1ff,        0x0 }, {  0x100804020100804,      0x402 }, {           0x1c0e07,        0x0 }, {  0x1008040201c0fff,      0x402 }},
{{              0x1ff,        0x0 }, {  0x201008040201008,      0x804 }, {           0xe07038,        0x0 }, {  0x201008040e071ff,      0x804 }},
{{              0x1ff,        0x0 }, {  0x402010080402010,     0x1008 }, {           0xe07038,        0x0 }, {  0x402010080e071ff,     0x1008 }},
{{              0x1ff,        0x0 }, {  0x804020100804020,     0x2010 }, {           0xe07038,        0x0 }, {  0x804020100e071ff,     0x2010 }},
{{              0x1ff,        0x0 }, { 0x1008040201008040,     0x4020 }, {          0x70381c0,        0x0 }, { 0x10080402070381ff,     0x4020 }},
{{              0x1ff,        0x0 }, { 0x2010080402010080,     0x8040 }, {          0x70381c0,        0x0 }, { 0x20100804070381ff,     0x8040 }},
{{              0x1ff,        0x0 }, { 0x4020100804020100,    0x10080 }, {          0x70381c0,        0x0 }, { 0x40201008070381ff,    0x10080 }},
{{            0x3fe00,        0x0 }, { 0x8040201008040201,      0x100 }, {           0x1c0e07,        0x0 }, { 0x80402010081ffe07,      0x100 }},
{{            0x3fe00,        0x0 }, {   0x80402010080402,      0x201 }, {           0x1c0e07,        0x0 }, {   0x804020101ffe07,      0x201 }},
{{            0x3fe00,        0x0 }, {  0x100804020100804,      0x402 }, {           0x1c0e07,        0x0 }, {  0x1008040201ffe07,      0x402 }},
{{            0x3fe00,        0x0 }, {  0x201008040201008,      0x804 }, {           0xe07038,        0x0 }, {  0x201008040e3fe38,      0x804 }},
{{            0x3fe00,        0x0 }, {  0x402010080402010,     0x1008 }, {           0xe07038,        0x0 }, {  0x402010080e3fe38,     0x1008 }},
{{            0x3fe00,        0x0 }, {  0x804020100804020,     0x2010 }, {           0xe07038,        0x0 }, {  0x804020100e3fe38,     0x2010 }},
{{            0x3fe00,        0x0 }, { 0x1008040201008040,     0x4020 }, {          0x70381c0,        0x0 }, { 0x100804020703ffc0,     0x4020 }},
{{            0x3fe00,        0x0 }, { 0x2010080402010080,     0x8040 }, {          0x70381c0,        0x0 }, { 0x201008040703ffc0,     0x8040 }},
{{            0x3fe00,        0x0 }, { 0x4020100804020100,    0x10080 }, {          0x70381c0,        0x0 }, { 0x402010080703ffc0,    0x10080 }},
{{          0x7fc0000,        0x0 }, { 0x8040201008040201,      0x100 }, {           0x1c0e07,        0x0 }, { 0x804020100ffc0e07,      0x100 }},
{{          0x7fc0000,        0x0 }, {   0x80402010080402,      0x201 }, {           0x1c0e07,        0x0 }, {   0x80402017fc0e07,      0x201 }},
{{          0x7fc0000,        0x0 }, {  0x100804020100804,      0x402 }, {           0x1c0e07,        0x0 }, {  0x100804027fc0e07,      0x402 }},
{{          0x7fc0000,        0x0 }, {  0x201008040201008,      0x804 }, {           0xe07038,        0x0 }, {  0x201008047fc7038,      0x804 }},
{{          0x7fc0000,        0x0 }, {  0x402010080402010,     0x1008 }, {           0xe07038,        0x0 }, {  0x402010087fc7038,     0x1008 }},
{{          0x7fc0000,        0x0 }, {  0x804020100804020,     0x2010 }, {           0xe07038,        0x0 }, {  0x804020107fc7038,     0x2010 }},
{{          0x7fc0000,        0x0 }, { 0x1008040201008040,     0x4020 }, {          0x70381c0,        0x0 }, { 0x1008040207ff81c0,     0x4020 }},
{{          0x7fc0000,        0x0 }, { 0x2010080402010080,     0x8040 }, {          0x70381c0,        0x0 }, { 0x2010080407ff81c0,     0x8040 }},
{{          0x7fc0000,        0x0 }, { 0x4020100804020100,    0x10080 }, {          0x70381c0,        0x0 }, { 0x4020100807ff81c0,    0x10080 }},
{{        0xff8000000,        0x0 }, { 0x8040201008040201,      0x100 }, {     0xe07038000000,        0x0 }, { 0x8040e07ff8040201,      0x100 }},
{{        0xff8000000,        0x0 }, {   0x80402010080402,      0x201 }, {     0xe07038000000,        0x0 }, {   0x80e07ff8080402,      0x201 }},
{{        0xff8000000,        0x0 }, {  0x100804020100804,      0x402 }, {     0xe07038000000,        0x0 }, {  0x100e07ff8100804,      0x402 }},
{{        0xff8000000,        0x0 }, {  0x201008040201008,      0x804 }, {    0x70381c0000000,        0x0 }, {  0x207038ff8201008,      0x804 }},
{{        0xff8000000,        0x0 }, {  0x402010080402010,     0x1008 }, {    0x70381c0000000,        0x0 }, {  0x407038ff8402010,     0x1008 }},
{{        0xff8000000,        0x0 }, {  0x804020100804020,     0x2010 }, {    0x70381c0000000,        0x0 }, {  0x807038ff8804020,     0x2010 }},
{{        0xff8000000,        0x0 }, { 0x1008040201008040,     0x4020 }, {   0x381c0e00000000,        0x0 }, { 0x10381c0ff9008040,     0x4020 }},
{{        0xff8000000,        0x0 }, { 0x2010080402010080,     0x8040 }, {   0x381c0e00000000,        0x0 }, { 0x20381c0ffa010080,     0x8040 }},
{{        0xff8000000,        0x0 }, { 0x4020100804020100,    0x10080 }, {   0x381c0e00000000,        0x0 }, { 0x40381c0ffc020100,    0x10080 }},
{{     0x1ff000000000,        0x0 }, { 0x8040201008040201,      0x100 }, {     0xe07038000000,        0x0 }, { 0x8040fff038040201,      0x100 }},
{{     0x1ff000000000,        0x0 }, {   0x80402010080402,      0x201 }, {     0xe07038000000,        0x0 }, {   0x80fff038080402,      0x201 }},
{{     0x1ff000000000,        0x0 }, {  0x100804020100804,      0x402 }, {     0xe07038000000,        0x0 }, {  0x100fff038100804,      0x402 }},
{{     0x1ff000000000,        0x0 }, {  0x201008040201008,      0x804 }, {    0x70381c0000000,        0x0 }, {  0x2071ff1c0201008,      0x804 }},
{{     0x1ff000000000,        0x0 }, {  0x402010080402010,     0x1008 }, {    0x70381c0000000,        0x0 }, {  0x4071ff1c0402010,     0x1008 }},
{{     0x1ff000000000,        0x0 }, {  0x804020100804020,     0x2010 }, {    0x70381c0000000,        0x0 }, {  0x8071ff1c0804020,     0x2010 }},
{{     0x1ff000000000,        0x0 }, { 0x1008040201008040,     0x4020 }, {   0x381c0e00000000,        0x0 }, { 0x10381ffe01008040,     0x4020 }},
{{     0x1ff000000000,        0x0 }, { 0x2010080402010080,     0x8040 }, {   0x381c0e00000000,        0x0 }, { 0x20381ffe02010080,     0x8040 }},
{{     0x1ff000000000,        0x0 }, { 0x4020100804020100,    0x10080 }, {   0x381c0e00000000,        0x0 }, { 0x40381ffe04020100,    0x10080 }},
{{   0x3fe00000000000,        0x0 }, { 0x8040201008040201,      0x100 }, {     0xe07038000000,        0x0 }, { 0x807fe07038040201,      0x100 }},
{{   0x3fe00000000000,        0x0 }, {   0x80402010080402,      0x201 }, {     0xe07038000000,        0x0 }, {   0xbfe07038080402,      0x201 }},
{{   0x3fe00000000000,        0x0 }, {  0x100804020100804,      0x402 }, {     0xe07038000000,        0x0 }, {  0x13fe07038100804,      0x402 }},
{{   0x3fe00000000000,        0x0 }, {  0x201008040201008,      0x804 }, {    0x70381c0000000,        0x0 }, {  0x23fe381c0201008,      0x804 }},
{{   0x3fe00000000000,        0x0 }, {  0x402010080402010,     0x1008 }, {    0x70381c0000000,        0x0 }, {  0x43fe381c0402010,     0x1008 }},
{{   0x3fe00000000000,        0x0 }, {  0x804020100804020,     0x2010 }, {    0x70381c0000000,        0x0 }, {  0x83fe381c0804020,     0x2010 }},
{{   0x3fe00000000000,        0x0 }, { 0x1008040201008040,     0x4020 }, {   0x381c0e00000000,        0x0 }, { 0x103ffc0e01008040,     0x4020 }},
{{   0x3fe00000000000,        0x0 }, { 0x2010080402010080,     0x8040 }, {   0x381c0e00000000,        0x0 }, { 0x203ffc0e02010080,     0x8040 }},
{{   0x3fe00000000000,        0x0 }, { 0x4020100804020100,    0x10080 }, {   0x381c0e00000000,        0x0 }, { 0x403ffc0e04020100,    0x10080 }},
{{ 0x7fc0000000000000,        0x0 }, { 0x8040201008040201,      0x100 }, { 0x81c0000000000000,      0x703 }, { 0xffc0201008040201,      0x703 }},
{{ 0x7fc0000000000000,        0x0 }, {   0x80402010080402,      0x201 }, { 0x81c0000000000000,      0x703 }, { 0xffc0402010080402,      0x703 }},
{{ 0x7fc0000000000000,        0x0 }, {  0x100804020100804,      0x402 }, { 0x81c0000000000000,      0x703 }, { 0xffc0804020100804,      0x703 }},
{{ 0x7fc0000000000000,        0x0 }, {  0x201008040201008,      0x804 }, {  0xe00000000000000,     0x381c }, { 0x7fc1008040201008,     0x381c }},
{{ 0x7fc0000000000000,        0x0 }, {  0x402010080402010,     0x1008 }, {  0xe00000000000000,     0x381c }, { 0x7fc2010080402010,     0x381c }},
{{ 0x7fc0000000000000,        0x0 }, {  0x804020100804020,     0x2010 }, {  0xe00000000000000,     0x381c }, { 0x7fc4020100804020,     0x381c }},
{{ 0x7fc0000000000000,        0x0 }, { 0x1008040201008040,     0x4020 }, { 0x7000000000000000,    0x1c0e0 }, { 0x7fc8040201008040,    0x1c0e0 }},
{{ 0x7fc0000000000000,        0x0 }, { 0x2010080402010080,     0x8040 }, { 0x7000000000000000,    0x1c0e0 }, { 0x7fd0080402010080,    0x1c0e0 }},
{{ 0x7fc0000000000000,        0x0 }, { 0x4020100804020100,    0x10080 }, { 0x7000000000000000,    0x1c0e0 }, { 0x7fe0100804020100,    0x1c0e0 }},
{{ 0x8000000000000000,       0xff }, { 0x8040201008040201,      0x100 }, { 0x81c0000000000000,      0x703 }, { 0x81c0201008040201,      0x7ff }},
{{ 0x8000000000000000,       0xff }, {   0x80402010080402,      0x201 }, { 0x81c0000000000000,      0x703 }, { 0x81c0402010080402,      0x7ff }},
{{ 0x8000000000000000,       0xff }, {  0x100804020100804,      0x402 }, { 0x81c0000000000000,      0x703 }, { 0x81c0804020100804,      0x7ff }},
{{ 0x8000000000000000,       0xff }, {  0x201008040201008,      0x804 }, {  0xe00000000000000,     0x381c }, { 0x8e01008040201008,     0x38ff }},
{{ 0x8000000000000000,       0xff }, {  0x402010080402010,     0x1008 }, {  0xe00000000000000,     0x381c }, { 0x8e02010080402010,     0x38ff }},
{{ 0x8000000000000000,       0xff }, {  0x804020100804020,     0x2010 }, {  0xe00000000000000,     0x381c }, { 0x8e04020100804020,     0x38ff }},
{{ 0x8000000000000000,       0xff }, { 0x1008040201008040,     0x4020 }, { 0x7000000000000000,    0x1c0e0 }, { 0xf008040201008040,    0x1c0ff }},
{{ 0x8000000000000000,       0xff }, { 0x2010080402010080,     0x8040 }, { 0x7000000000000000,    0x1c0e0 }, { 0xf010080402010080,    0x1c0ff }},
{{ 0x8000000000000000,       0xff }, { 0x4020100804020100,    0x10080 }, { 0x7000000000000000,    0x1c0e0 }, { 0xf020100804020100,    0x1c0ff }},
{{                0x0,    0x1ff00 }, { 0x8040201008040201,      0x100 }, { 0x81c0000000000000,      0x703 }, { 0x81c0201008040201,    0x1ff03 }},
{{                0x0,    0x1ff00 }, {   0x80402010080402,      0x201 }, { 0x81c0000000000000,      0x703 }, { 0x81c0402010080402,    0x1ff03 }},
{{                0x0,    0x1ff00 }, {  0x100804020100804,      0x402 }, { 0x81c0000000000000,      0x703 }, { 0x81c0804020100804,    0x1ff03 }},
{{                0x0,    0x1ff00 }, {  0x201008040201008,      0x804 }, {  0xe00000000000000,     0x381c }, {  0xe01008040201008,    0x1ff1c }},
{{                0x0,    0x1ff00 }, {  0x402010080402010,     0x1008 }, {  0xe00000000000000,     0x381c }, {  0xe02010080402010,    0x1ff1c }},
{{                0x0,    0x1ff00 }, {  0x804020100804020,     0x2010 }, {  0xe00000000000000,     0x381c }, {  0xe04020100804020,    0x1ff1c }},
{{                0x0,    0x1ff00 }, { 0x1008040201008040,     0x4020 }, { 0x7000000000000000,    0x1c0e0 }, { 0x7008040201008040,    0x1ffe0 }},
{{                0x0,    0x1ff00 }, { 0x2010080402010080,     0x8040 }, { 0x7000000000000000,    0x1c0e0 }, { 0x7010080402010080,    0x1ffe0 }},
{{                0x0,    0x1ff00 }, { 0x4020100804020100,    0x10080 }, { 0x7000000000000000,    0x1c0e0 }, { 0x7020100804020100,    0x1ffe0 }},
};

const unsigned short bitx3_lut[8] = {
   0x0,      0x7,      0x38,     0x3f,
   0x1c0,    0x1c7,    0x1f8,    0x1ff
};

// For printing a cell location, use the following table.
// It is mutable so that the 0-based representation can be made 1-based (or 'A'-based) on start-up.
char cl2txt[81][6] = {
        /*   0 */  "[0,0]",
        /*   1 */  "[0,1]",
        /*   2 */  "[0,2]",
        /*   3 */  "[0,3]",
        /*   4 */  "[0,4]",
        /*   5 */  "[0,5]",
        /*   6 */  "[0,6]",
        /*   7 */  "[0,7]",
        /*   8 */  "[0,8]",
        /*   9 */  "[1,0]",
        /*  10 */  "[1,1]",
        /*  11 */  "[1,2]",
        /*  12 */  "[1,3]",
        /*  13 */  "[1,4]",
        /*  14 */  "[1,5]",
        /*  15 */  "[1,6]",
        /*  16 */  "[1,7]",
        /*  17 */  "[1,8]",
        /*  18 */  "[2,0]",
        /*  19 */  "[2,1]",
        /*  20 */  "[2,2]",
        /*  21 */  "[2,3]",
        /*  22 */  "[2,4]",
        /*  23 */  "[2,5]",
        /*  24 */  "[2,6]",
        /*  25 */  "[2,7]",
        /*  26 */  "[2,8]",
        /*  27 */  "[3,0]",
        /*  28 */  "[3,1]",
        /*  29 */  "[3,2]",
        /*  30 */  "[3,3]",
        /*  31 */  "[3,4]",
        /*  32 */  "[3,5]",
        /*  33 */  "[3,6]",
        /*  34 */  "[3,7]",
        /*  35 */  "[3,8]",
        /*  36 */  "[4,0]",
        /*  37 */  "[4,1]",
        /*  38 */  "[4,2]",
        /*  39 */  "[4,3]",
        /*  40 */  "[4,4]",
        /*  41 */  "[4,5]",
        /*  42 */  "[4,6]",
        /*  43 */  "[4,7]",
        /*  44 */  "[4,8]",
        /*  45 */  "[5,0]",
        /*  46 */  "[5,1]",
        /*  47 */  "[5,2]",
        /*  48 */  "[5,3]",
        /*  49 */  "[5,4]",
        /*  50 */  "[5,5]",
        /*  51 */  "[5,6]",
        /*  52 */  "[5,7]",
        /*  53 */  "[5,8]",
        /*  54 */  "[6,0]",
        /*  55 */  "[6,1]",
        /*  56 */  "[6,2]",
        /*  57 */  "[6,3]",
        /*  58 */  "[6,4]",
        /*  59 */  "[6,5]",
        /*  60 */  "[6,6]",
        /*  61 */  "[6,7]",
        /*  62 */  "[6,8]",
        /*  63 */  "[7,0]",
        /*  64 */  "[7,1]",
        /*  65 */  "[7,2]",
        /*  66 */  "[7,3]",
        /*  67 */  "[7,4]",
        /*  68 */  "[7,5]",
        /*  69 */  "[7,6]",
        /*  70 */  "[7,7]",
        /*  71 */  "[7,8]",
        /*  72 */  "[8,0]",
        /*  73 */  "[8,1]",
        /*  74 */  "[8,2]",
        /*  75 */  "[8,3]",
        /*  76 */  "[8,4]",
        /*  77 */  "[8,5]",
        /*  78 */  "[8,6]",
        /*  79 */  "[8,7]",
        /*  80 */  "[8,8]"
    };

alignas(64)
// general purpose / multiple locations:
const __m256i nibble_mask = _mm256_set1_epi8(0x0F);
const __m256i maskff = _mm256_set1_epi16(0xff);
const __m256i dgt1 = _mm256_set1_epi8('1');
const __m256i maskff_epi8 = _mm256_set1_epi8(0xff);

// used for expanding bit vectors to boolean vectors
const __m256i bit_mask_expand = _mm256_setr_epi16(1<<0, 1<<1, 1<<2, 1<<3, 1<<4, 1<<5, 1<<6, 1<<7, 1<<8, 1<<9, 1<<10, 1<<11, 1<<12, 1<<13, 1<<14, 1<<15);
const __m256i shuffle_mask_bytes = _mm256_setr_epi8(0,0,0,0,0,0,0,0,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,3,3,3,3,3,3,3,3);

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

// used in verify:
const __m256i mask9 { -1LL, -1LL, 0xffffLL, 0 };
const __m256i ones9 = _mm256_and_si256(ones, mask9);

// used in triads:
const __m256i mask11hi { 0LL, 0LL, 0xffffLL<<48, ~0LL };
const __m256i mask1ff   = _mm256_set1_epi16(0x1ff);
const __m256i mask9x1ff = _mm256_and_si256(mask1ff, mask9);

// used in triads (row triad capture):
const __m256i shuff725to012 = _mm256_setr_epi8(14, 15,  4,  5, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1,  -1, -1,
                                               -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);

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
const __m256i mask27 { -1LL, (long long int)0xffffffffffff00ffLL, (long long int)0xffffffff00ffffffLL, 0xffffffffffLL };
const __m256i threes    = _mm256_set1_epi8 ( 3 );
const __m256i fours     = _mm256_set1_epi8 ( 4 );
//   popcnt by nibble
const __m256i lookup    = _mm256_setr_epi8(0 ,1 ,1 ,2 ,1 ,2 ,2 ,3 ,1 ,2 ,2 ,3 ,2 ,3 ,3 ,4,
                                           0 ,1 ,1 ,2 ,1 ,2 ,2 ,3 ,1 ,2 ,2 ,3 ,2 ,3 ,3 ,4);

#ifdef OPT_UQR
// used in UQR processing:
// shuffle per row/col:  0,1,2,3,4,5,6,7,8 -> 0,1,2,-,3,4,5,-  0,1,2,-,6,7,8,-
const __m256i lineshuffle = _mm256_setr_epi8(0,1,2,3,4,5,-1,-1,6,7,8,9,10,11,-1,-1,
                                             0,1,2,3,4,5,-1,-1,6,7,8,9,10,11,-1,-1);
const __m256i linerotate[2] = {
      // line[0]: rotate first/third group clockwise
      _mm256_setr_epi8(2,3,4,5,0,1,-1,-1,8,9,10,11,12,13,-1,-1,
                       2,3,4,5,0,1,-1,-1,8,9,10,11,12,13,-1,-1),
      // line[1]: rotate second/fourth group clockwise
      _mm256_setr_epi8(0,1,2,3,4,5,-1,-1,10,11,12,13,8,9,-1,-1,
                       0,1,2,3,4,5,-1,-1,10,11,12,13,8,9,-1,-1) };
#endif

alignas(64)
std::atomic<long long> past_naked_count(0); // how often do we get past the naked single serach
std::atomic<long long> digits_entered_and_retracted(0); // to measure guessing overhead
std::atomic<long long> triads_resolved(0);  // how many triads did we resolved
std::atomic<long long> triad_updates(0);    // how many triads did cancel candidates
std::atomic<long long> naked_sets_searched(0); // how many naked sets did we search for
std::atomic<long long> naked_sets_found(0); // how many naked sets did we actually find
std::atomic<long long> unique_rectangles_checked(0);   // how many unique rectangles were checked
std::atomic<long long> unique_rectangles_avoided(0);   // how many unique rectangles were avoided
std::atomic<long long> fishes_detected(0);             // how many fishes were identified
std::atomic<long long> fishes_excluded(0);             // how many square fishes were found and excluded
std::atomic<long long> fishes_specials_detected(0);    // how many special fish patterns were identified ((subset of fishes_detected)
std::atomic<long long> fishes_updated(0);              // how many fishes were updated
std::atomic<long long> fishes_specials_updated(0);     // how many special fish patterns were updated (subset of fishes_updated)
std::atomic<long> bug_count(0);             // universal grave detected
std::atomic<long> guesses(0);               // how many guesses did it take
std::atomic<long> trackbacks(0);            // how often did we back track
std::atomic<long> solved_count(0);          // puzzles solved
std::atomic<long> no_guess_cnt(0);          // how many puzzles were solved without guessing
std::atomic<long> unsolved_count(0);        // puzzles unsolved (no solution exists)
std::atomic<long> non_unique_count(0);      // puzzles not unique (with -u)
std::atomic<long> not_verified_count(0);    // puzzles non verified (with -v)
std::atomic<long> verified_count(0);        // puzzles successfully verified (with -v)

#if defined(OPT_UQR) || defined(OPT_FSH)
    typedef union {
        __m256i m256;
        v16us v16;
    } cbbv_t;
#endif

#ifdef OPT_UQR
    // for each set of results, nine pairs of two diagonal intersections are accessed
    // by using the two indices for accessing 'res'.
    const unsigned char cuqr_access[9][2] = {
        {0,4}, {1,5}, {2,6}, {8,12}, {9,13}, {10,14}, {16,20}, {17,21}, {18,22}
    };

    // bit pattern for uqr long edge:
    const unsigned short uqr_pattern[9] = { 0, 0b11, 0b101, 0b1001, 0b10001, 0b100001, 0b1000001, 0b10000001, 0b100000001 };

    // type Uqr provides the start cells position relative to the row/col for uqrs,
    // and the distance to the opposite cells of the uqr, along the row/col long edge.
    // Note that the distance to the next row/col is not provided here (it can only be 1 or 2)
    // and will be easy to find in the processing context.
typedef struct {
        const unsigned char start_cell;   // relative to row/col start
        const unsigned char dist;         // 'long' edge, relative to start_cell, in row/col direction
        const unsigned short pattern = uqr_pattern[dist]<<start_cell;
} Uqr;

// each row yields 9 cuqrs, for 3 iterations as shown below.
// together these allow to iterate over the examing uqrs and their diagonals.
const Uqr cuqrs[3][9] = {
        // row 0,1 iter 0
        {{0,3}, {1,3}, {2,3}, {0,6}, {1,6}, {2,6}, {3,3}, {4,3}, {5,3}},
        // row 0,1 iter 1
        {{1,2}, {2,2}, {0,5}, {1,5}, {2,5}, {0,8}, {4,2}, {5,2}, {3,5}},
        // row 0,1 iter 2
        {{2,1}, {0,4}, {1,4}, {2,4}, {0,7}, {1,7}, {5,1}, {3,4}, {4,4}},
    };

// a corner of a unique rectangle
class UqrCorner {
public:
    bool is_pair = false;
    bool is_single = false;
    unsigned char indx;
    unsigned char pair_indx=0;
    __uint128_t *right_edge;
};

// a pair of bivalues for a unique rectangle
class UqrPair {
public:
    unsigned short digits;   // digits could be a single digit, if diag is a single
    unsigned char cnt = 0;
    unsigned char crnrs = 0;
};

// row combos: 0,1  0,2  1,2
const unsigned char row_combos[3][2] = { {0,1}, {0,2}, {1,2} };

const bit128_t vband = { ((const bit128_t*)&small_index_lut[3][Col])->u128
                       | ((const bit128_t*)&small_index_lut[3+1][Col])->u128
                       | ((const bit128_t*)&small_index_lut[3+2][Col])->u128 };

#endif

bool bmi2_support = false;

// stats and command line options
int reportstats     = 0; // collect and report some statistics
int reporttimings   = 0; // report timings only
int verify          = 0; // verify solution correctness (implied otherwise)
int debug           = 0; // provide step by step output on the solution
int thorough_check  = 0; // check for back tracking even if no guess was made.
int numthreads      = 0; // if not 0, number of threads
int warnings        = 0; // display warnings

// puzzle rules
typedef enum {
    Regular  = 0,   // R - Default: Find solution, assuming it will be unique
                    // (this is fastest but is not suitable for non-unique puzzles)
    FindOne  = 1,   // O - Search for one solution
    Multiple = 2    // M - Determine whether puzzles are non-unique
} Rules;

Rules rules;

// execution modes at runtime
bool mode_sets=false;           // 'S', see OPT_SETS
bool mode_triad_res=false;      // 'T', see OPT_TRIAD_RES
bool mode_uqr=false;            // 'U', see OPT_UQR
bool mode_fish=false;			// 'F', see OPT_FSH

signed char *output;

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

inline __m256i expand_bitvector_epi8(unsigned int m) {
    __m256i bits = _mm256_shuffle_epi8(_mm256_set1_epi32(m), shuffle_mask_bytes);
    return _mm256_cmpeq_epi8(_mm256_and_si256( select_bits, bits), select_bits);
}

template<bool doubledbits=false>
inline __attribute__((always_inline)) unsigned short compress_epi16_boolean128(__m128i b) {
    if (doubledbits) {
        return _mm_movemask_epi8(b);
    }
    if ( bmi2_support ) {
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
    if ( bmi2_support ) {
        return _pext_u32(_mm256_movemask_epi8(b),0x55555555);
    } else {
        return _mm_movemask_epi8(_mm_packs_epi16(_mm256_castsi256_si128(b), _mm256_extractf128_si256(b,1)));
    }
}

template<bool doubledbits=false>
inline __attribute__((always_inline)) unsigned long long compress_epi16_boolean(__m256i b1, __m256i b2) {
    if (doubledbits) {
        return (((unsigned long long)_mm256_movemask_epi8(b2))<<32) | _mm256_movemask_epi8(b1);
    }
    return _mm256_movemask_epi8(_mm256_permute4x64_epi64(_mm256_packs_epi16(b1,b2), 0xD8));
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
    if ( bmi2_support ) {
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

// for the 9 cells of a box (given by the index of the box),
// return the corresponding masking bits as a contiguous bit vector
inline unsigned int get_contiguous_masked_indices_for_box(unsigned long long indices[2], int boxi) {

    // step 1, and-combine the indices with const bitmask for the box
    bit128_t ret = { .u64 = { box_bitmasks[boxi].u64[0] & indices[0], box_bitmasks[boxi].u64[1] & indices[1] } };

    // step 2, extract the 21 bit extent for the given box
    // for the first 6 boxes, we know the indices are all in the low 64 bits
    // for the last 3 boxes, combine the 10 bits [54:63] from u64[0] with u64[1] and shift down accordingly:
    unsigned int mask = (boxi < 6) ? (ret.u64[0]>>box_start_by_boxindex[boxi])
                       : (((ret.u64[1]<<10) | (ret.u64[0] >> (64-10))) >> (box_start_by_boxindex[boxi]-(64-10)));

    // step 3, combine the 21 given bits (0b111000000111000000111)
    // into contiguous 9 bits for the given box
    if ( bmi2_support ) {
        const unsigned int mask21 = (0x7<<18)|(0x7<<9)|0x7;
        return _pext_u32(mask,mask21);
    } else {
        return   (mask & 7)
               | ( (mask>>(9-3)) & (7<<3) )
               | ( (mask>>(18-6)) & (7<<6) );
    }
    // not reached
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
inline void add_and_mask_all_indices(bit128_t *indices, bit128_t *mask, unsigned char i) {
	indices->u128 |= (*(const __uint128_t *)&big_index_lut[i][All][0]) & mask->u128;
}

template<Kind kind>
inline void add_indices(bit128_t *indices, unsigned char i) {
	indices->u128 |= *(const __uint128_t *)&big_index_lut[i][kind][0];
}
#pragma GCC diagnostic pop

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
        // isolate the next lsb
        return _mm256_and_si256(vec, _mm256_sub_epi16(_mm256_setzero_si256(), vec));
}

// global debug aids
int dprintfilter = 0;       // global filter mask set from env, 
FILE * dprintout = stdout;

int dprintf(int filter, const char *format...) {
    int ret = 0;
    if ( filter & dprintfilter ) {
        va_list args;
        va_start(args, format);
        ret = vfprintf(dprintout, format, args);
        va_end(args);
    }
    return ret;
}

inline void dump_m256i(__m256i x, const char *msg="", int filter=-1) {
    dprintf(filter, "%s %llx,%llx,%llx,%llx\n", msg, ((__v4du)x)[0],((__v4du)x)[1],((__v4du)x)[2],((__v4du)x)[3]);
}

inline void dump_m128i(__m128i x, const char *msg="", int filter=-1) {
    dprintf(filter, "%s %llx,%llx\n", msg, ((__v2du)x)[0],((__v2du)x)[1]);
}

// a helper function to print a grid of bits.
// the bits are arranged as 9 bits each in 9 unsigned short elements of a __m256i parameter.
//
inline void dump_m256i_grid(__m256i v, const char *msg="", int filter=-1) {
    if ( !(filter & dprintfilter) ) {
        return;
    }
    dprintf(filter, "%s\n", msg);
    for (unsigned char r=0; r<9; r++) {
        unsigned short b = ((__v16hu)v)[r];
        for ( unsigned char i=0; i<9; i++) {
            dprintf(filter, "%s", (b&(1<<i))? "x":"-");
            if ( i%3 == 2 ) {
                dprintf(filter, " ");
            }
            if ( i%9 == 8 ) {
                dprintf(filter, "\n");
            }
        }
    }
}

// a helper function to print a grid of bits.
// bits are arranged consecutively as 81 bits in a __unint128_t parameter.
//
inline void dump_bits(__uint128_t bits, const char *msg="", int filter=-1) {
    if ( !(filter & dprintfilter) ) {
        return;
    }
    dprintf(filter, "%s:\n", msg);
    for ( int i_=0; i_<81; i_++ ) {
        dprintf(filter, "%s", (*(bit128_t*)&bits).check_indexbit(i_)?"x":"-");
        if ( i_%3 == 2 ) {
            dprintf(filter, " ");
        }
        if ( i_%9 == 8 ) {
            dprintf(filter, "\n");
        }
    }
}

inline void format_candidate_set(char *ret, unsigned short candidates);

// a helper function to print a sudoku board,
// given the 81 cells solved or with candidates.
//
inline void dump_board(unsigned short *candidates, const char *msg="", int filter=-1) {
    if ( !(filter & dprintfilter) ) {
        return;
    }
    dprintf(filter,"%s:\n", msg);
    for ( int i_=0; i_<81; i_++ ) {
        char ret[32];
        format_candidate_set(ret, candidates[i_]);
        dprintf(filter,"%8s,", ret);
        if ( i_%9 == 8 ) {
            dprintf(filter,"\n");
        }
    }
}

// a helper function to print a sudoku puzzle from the solved cells,
// given the 81 cells.
//
template <bool transpose = false>
inline void dump_puzzle(unsigned short *candidates, bit128_t &unlocked, const char *msg="", int filter=-1) {
    if ( filter & dprintfilter ) {
        char gridout[82];
        for (unsigned char j = 0; j < 81; ++j) {
            unsigned short t = transpose? transposed_cell[j] : j;
            if ( unlocked.check_indexbit(t) ) {
                gridout[j] = '0';
            } else {
                gridout[j] = 49+_tzcnt_u32(candidates[t]);
            }
        }
        dprintf(filter, "%s %.81s\n", msg, gridout);
    }
}

// an informational/debug aid specific to fishes (use option -d3)
// showing
// - (real) fish positions as 'o'
// - fin(s) if present as '@'
// - candidates to remove as 'X'
// - other candidates containing the digit to remove as '.'
//
// use of template parameter transposed: false for row fishes, true for col fishes

#ifdef OPT_FSH
template <bool transposed=false>
// the parameters cbbv, base, subs and hassubs are transposed for cols
inline void show_fish(cbbv_t &cbbv, unsigned short base, unsigned short subs, unsigned short hassubs, bit128_t &clean_bits, bit128_t &dgt_bits, const char *msg="") {
    printf("%s:\n", msg);
    for ( int i_=0; i_<81; i_++ ) {
        int ti = transposed?transposed_cell[i_]:i_;
        unsigned int ti_bit = 1<<(ti%9);
        unsigned int ti_bitv = 1<<(ti/9);
        if ( ti_bitv & subs ) {
            printf("%s", (cbbv.v16[ti/9]&ti_bit)?((ti_bit&base)?"o":"@"):"-");
        } else if ( (ti_bitv & hassubs) && clean_bits.check_indexbit(i_)) {
            printf("X");
        } else {
            printf(dgt_bits.check_indexbit(i_)? ".":"-");
        }
        if ( i_%9 == 8 ) {
            printf("\n");
        } else if ( i_%3 == 2 ) {
            printf(" ");
        }
    }
}

template <bool transposed=false>
// the parameters cbbv and subs are transposed for cols
inline void show_fish(cbbv_t &cbbv, unsigned short subs, unsigned char fincnt, unsigned char *fins, bit128_t &clean_bits, bit128_t &dgt_bits, const char *msg="") {
    printf("%s:\n", msg);
    for ( int i_=0; i_<81; i_++ ) {
        int ti = transposed?transposed_cell[i_]:i_;
        bool found = false;
        for ( int k=0; k<fincnt; k++ ) {
            if ( i_ == fins[k] ) {
                found = true;
                printf("@");
                break;
            }
        }
        if ( !found ) {
            unsigned int ti_bit = 1<<(ti%9);
            unsigned int ti_bitv = 1<<(ti/9);
            if ( clean_bits.check_indexbit(i_) ) {
                printf("X");
            } else if ( ti_bitv & subs ) {
                printf("%s", (cbbv.v16[ti/9]&ti_bit)?"o":"-");
            } else {
                printf(dgt_bits.check_indexbit(i_)? ".":"-");
            }
        }
        if ( i_%9 == 8 ) {
            printf("\n");
        } else if ( i_%3 == 2 ) {
            printf(" ");
        }
    }
}
#endif

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
// for passing triad information to make_guess.
//
class TriadInfo {
public:
    unsigned short row_triads[36];            //  27 triads, in groups of 9 with a gap of 1
    unsigned short col_triads[36];            //  27 triads, in groups of 9 with a gap of 1
    unsigned short row_triads_wo_musts[36];   //  triads minus tmusts for guessing
    unsigned short col_triads_wo_musts[36];   //  triads minus tmusts for guessing
    unsigned int triads_selection[2];
};

class GridState;

// Solver sharable data - used by some algorithms and make_guess
class SolverData {
private:
    bit128_t  bivalues;
    bit128_t candidate_bits_by_value[9];
    unsigned char sectionSetUnlocked[3][9];
// Note that the candidate_bits_by_value also offers the opportunity to find hidden sets in rows (extensible)
// Similar to fishes and naked sets, only those sets that are fully expressed for a digit
// by all position bits will be found (all pairs will be found though).
// To examplify the idea:
// for ( unsigned char row=0; row<9; row++ ) {
//     // candidate bits by position:
//     cbbv_t cbbp_v;
//     unsigned char off = 9*row;
//     cbbp_v.m256 = _mm256_setr_epi16( candidate_bits_by_value[0]>>off,
//                                      candidate_bits_by_value[1]>>off,
//                                      candidate_bits_by_value[2]>>off,
//                                      candidate_bits_by_value[3]>>off,
//                                      candidate_bits_by_value[4]>>off,
//                                      candidate_bits_by_value[5]>>off,
//                                      candidate_bits_by_value[6]>>off,
//                                      candidate_bits_by_value[7]>>off,
//                                      candidate_bits_by_value[8]>>off,0,0,0,0,0,0,0,0));
//
//        iterate over the digit sets:
//        for (unsigned char ds=0; ds<9; ds++) {
//          // for the given digit, the count N for the tentative hidden set is: cnt.
//          // take each digit set and compare to all other digits.
//          //
//          unsigned char cnt = __popcnt16(cbbp_v.v16[ds]);
//          if ( cnt <= 4 && cnt > 1) {
//              __m256i dsv = _mm256_and_si256(_mm256_set1_epi16(cbbp_v.v16[ds]),mask9);
//              // set of rows that are a subset of dsv
//              __m256i issub_v = _mm256_cmpeq_epi16(dsv, _mm256_or_si256(dsv, cbbp_v.m256));
//              unsigned short subs = 0x1ff & compress_epi16_boolean(issub_v);
//              if ( __popcnt16(subs) == cnt ) { // then issub_v / subs indicate the digits of the hidden subset
//                                               // while cbbp_v.v16[ds] gives the indicess within the row
//                  // the cleanup is very simple...
//...
//
// for columns and boxes, the loading of cbbp_v would be more tedious...
//
// Note also that any cnt of 2 identifies a bi-local (i.e. dual link) for that row and digit.
//
public:
    bool bivaluesValid;
    bool cbbvsValid;
    bool sectionSetUnlockedValid[3];
    TriadInfo triadInfo;
    unsigned char  guess_hint_index;
    unsigned short guess_hint_digit;

    inline bit128_t &getBivalues(unsigned short *candidates) {
        if ( !bivaluesValid ) {
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
            bivalues.u16[4] = compress_epi16_boolean<false>(_mm256_and_si256(
                                     _mm256_cmpgt_epi16(c,_mm256_setzero_si256()),
                                     _mm256_cmpeq_epi16(lsb,c)));
    
            bivalues.u16[5] = (__popcnt16(candidates[80]) == 2)?1:0;
            // bivalues is now set for subsequent steps
            bivaluesValid = true;
        }
        return bivalues;
    }

    // Prepare per-digit bit masks
    // These bit masks are leveraged in several places:
    // - set search
    // - fish algorithms: x-wing,
    //   sword-fish, jelly-fish etc.
    // - querying rows/cols to check for additional candidates
    //   beyond the uqr corners.
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

    template<Kind kind>
    inline unsigned char *getSectionSetUnlocked(GridState &gs);

};

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
    int flags;                        // - if negative as indicator for multiple solutions in this grid_state
                                      // - low 27 bits to indicate sets of size 4 in 27 sections
                                      // aligned on 16 bytes
    bit128_t unlocked;                // for keeping track of which cells still need to be resolved. Set bits correspond to cells that still have multiple possibilities
    bit128_t updated;                 // for keeping track of which cell's candidates may have been changed since last time we looked for naked sets. Set bits correspond to changed candidates in these cells
    bit128_t set23_found[3];          // for keeping track of found sets of size 2 and 3

// GridState is normally copied for recursion
// Here we initialize the starting state including the puzzle.
//
inline void initialize(signed char grid[81]) {
    // 0x1ffffffffffffffffffffULLL is (0x1ULL << 81) - 1
    unlocked.u128 = (((__uint128_t)1)<<81)-1;

    flags = 0;

    triads_unlocked[0] = triads_unlocked[1] = 0x1ffLL | (0x1ffLL<<10) | (0x1ffLL<<20);
    set23_found[0] = set23_found[1] = set23_found[2] = {__int128 {0}};

    // set unlocked
    for (unsigned int i=0; i<64; i +=32 ) {
        __m256i in = *(__m256i_u*)&grid[i];
        unlocked.u32[i>>5] = _mm256_movemask_epi8(_mm256_cmpgt_epi8(dgt1, in));
    }
    __m128i in = *(__m128i_u*)&grid[64];
    unlocked.u16[4] = _mm_movemask_epi8(_mm_cmpgt_epi8(_mm256_castsi256_si128(dgt1),in));

    if (grid[80] > '0') {
        _bittestandreset64((long long int *)&unlocked.u64[1], 16);
    }
    updated = unlocked;

    bit128_t digit_bits[9] {};

    bit128_t locked = { .u128 = ~unlocked.u128 };
    locked.u64[1] &= 0x1ffff;

    // Grouping the updates by digit beats other methods for number of clues >17.
    // calculate the masks for each digit from the place and value of the clues:
    for ( unsigned int i=0; i<2; i++) {
        unsigned long long lkd = locked.u64[i];
        while (lkd) {
            int dix = tzcnt_and_mask(lkd)+(i<<6);
            int dgt = grid[dix] - 49;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
            digit_bits[dgt].u128 = digit_bits[dgt].u128 | (*(const __uint128_t *)&big_index_lut[dix][All][0]);
#pragma GCC diagnostic pop
        }
    }
    // update candidates and process the 9 masks for each chunk of the candidates:
    for ( unsigned int i=0; i<96; i += 32) {
        __m256i c1 = mask1ff;
        __m256i c2 = c1;
        for ( unsigned char dgt=0; dgt<8; dgt += 1) {
            // load 32 digit bits:
            __m256i bits = _mm256_and_si256(expand_bitvector_epi8(digit_bits[dgt].u32[i>>5]), _mm256_set1_epi8(1<<dgt));
            c1 = _mm256_andnot_si256(_mm256_unpacklo_epi8(bits, _mm256_setzero_si256()), c1);
            c2 = _mm256_andnot_si256(_mm256_unpackhi_epi8(bits, _mm256_setzero_si256()), c2);
        }
        // digit 9
        // load 32 digit bits:
        __m256i bits = _mm256_shuffle_epi8(_mm256_set1_epi32(digit_bits[8].u32[i>>5]), shuffle_mask_bytes);
        bits = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256( select_bits, bits), select_bits), ones_epi8);
        c1 = _mm256_andnot_si256(_mm256_unpacklo_epi8(_mm256_setzero_si256(),bits), c1);
        c2 = _mm256_andnot_si256(_mm256_unpackhi_epi8(_mm256_setzero_si256(),bits), c2);

        *(__m256i*)&candidates[i] = c1;
        if ( i==64) {
            candidates[80] = _mm256_extract_epi16(c2,0);
            break;
        }
        *(__m256i*)&candidates[i+16] = c2;
    }
    
    // finally place the clues
    for ( unsigned int i=0; i<2; i++) {
        unsigned long long lkd = locked.u64[i];
        while (lkd) {
            int dix = tzcnt_and_mask(lkd)+(i<<6);
            candidates[dix] = 1<<(grid[dix] - 49);
        }
    }

}

// Normally digits are entered by a 'goto enter;'.
// enter_digit is not used in that case.
// Only make_guess uses this member function.
protected:
template<Verbosity verbose=VNone>
inline __attribute__((always_inline)) void enter_digit( unsigned short digit, unsigned char i) {
    // lock this cell and and remove this digit from the candidates in this row, column and box

    bit128_t to_update = {0};

    if ( verbose == VDebug ) {
        printf(" %x at %s\n", _tzcnt_u32(digit)+1, cl2txt[i]);
    }
#ifndef NDEBUG
    if ( __popcnt16(digit) != 1 && warnings != 0 ) {
        printf("error in enter_digit: %x\n", digit);
    }
#endif

    if (i < 64) {
        _bittestandreset64((long long int *)&unlocked.u64[0], i);
    } else {
        _bittestandreset64((long long int *)&unlocked.u64[1], i-64);
    }

    candidates[i] = digit;

    add_and_mask_all_indices(&to_update, &unlocked, i);

    updated.u128 |= to_update.u128;
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
// this form of make_guess establishes a 'contract' between the caller's lambda
// and the creation of the new GridState.
// Due to its overhead, it is not the fastest, but the most flexible form to make a guess.
//
template<Verbosity verbose, typename F>
inline GridState* make_guess(unsigned char cell_index, F &&gridUpdater) {
    // Create a copy of the state of the grid to make back tracking possible
    GridState* new_grid_state = this+1;
    if ( stackpointer >= GRIDSTATE_MAX-1 ) {
        fprintf(stderr, "Error: no GridState struct availabe\n");
        exit(0);
    }
    memcpy(new_grid_state, this, sizeof(GridState));
    new_grid_state->stackpointer++;

    const char *msgs[2];

    // gridUpdater is a lambda, but could also be a function reference.
    // gridUpdater receives as input:
    // GridState & present GridState
    // GridState & new GridState
    // gridUpdater receives as input:
    // const char (*)[2] two messages provided one for each GridState
    // gridUpdater encapsulates local updates, typically to candidates of either GridState
    // and the two messages.
    //
    gridUpdater(*this, *new_grid_state, msgs);
    if ( verbose == VDebug ) {
        printf("guess at level >%d< - new level >%d<\nguess %s\n", stackpointer, new_grid_state->stackpointer, msgs[0]);
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
                   cl2txt[cell_index], stackpointer, gridout);
        }
        printf("saved state for level %d: %s\n",
               stackpointer, msgs[1]);
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
    guesses++;

    return new_grid_state;
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
inline GridState* make_guess(SolverData *solverData) {
    // Make a guess for a triad with 4 candidate values that has 2 candidates that are not
    // constrained to the triad (not in 'tmust') and has at least 2 or more unresolved cells.
    // If we cannot obtain such a triad, fall back to make_guess().
    // With such a triad found, select one of the 2 identified candidates
    // to eliminate in the new GridState, issue the debug info and proceed.
    // Save the current GridState to back track to and eliminate the other candidate from it.
    // The desired result is that either way the triad is resolved, which is structually beneficial
    // for the progress of the solution.

    unsigned char tpos;    // grid cell index of triad start
    unsigned char type;    // row == 0, col = 1
    unsigned char inc;     // increment for iterating triad cells
    unsigned short *wo_musts;  // pointer to triads' 'without must' candidates
    TriadInfo &triad_info = solverData->triadInfo;

    for ( int i=0; i<2; i++ ) {
        type = i;
        unsigned long long totest = triad_info.triads_selection[i];
        wo_musts  = type==0?triad_info.row_triads_wo_musts:triad_info.col_triads_wo_musts;
        inc  = (i<<3) + 1;  // 1 for rows, 9 for cols
        while (totest) {
            int ti = tzcnt_and_mask(totest);
            int can_ti = ti-ti/10;   // 'canonical' triad index
            if ( __popcnt16 (wo_musts[ti]) == 2 ) {
                if ( !mode_triad_res ) {
                    // did not prepare triad_info.triads_selection[i] above,
                    // so check whether the triad is 'interesting'
                    if ( __popcnt16(i==0?triad_info.row_triads[ti]:triad_info.col_triads[ti]) != 4 ) {
                        continue;
                    }
                }

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

    // leverage any guess hints (e.g. by the fish algorithm)
    if ( solverData->guess_hint_digit != 0 ) {
        return make_guess<verbose>(solverData->guess_hint_index, solverData->guess_hint_digit);
    }

    // if no suitable triad found, find a suitable bi-value.
    return make_guess<verbose>(solverData->getBivalues(candidates));
found:
    // update the current and the new grid_state with their respective candidate to delete
    unsigned short select_cand = 0x8000 >> __lzcnt16(*wo_musts);

    // Create a copy of the state of the grid to make back tracking possible
    GridState* new_grid_state = this+1;
    if ( stackpointer >= GRIDSTATE_MAX-1 ) {
        fprintf(stderr, "Error: no GridState struct availabe\n");
        exit(0);
    }
    memcpy(new_grid_state, this, sizeof(GridState));
    new_grid_state->stackpointer++;

    unsigned short other_cand  = *wo_musts & ~select_cand;
    unsigned char off = tpos;

    off = tpos;
    // Update candidates
    for ( unsigned char k=0; k<3; k++, tpos += inc) {
       new_grid_state->candidates[tpos] &= ~select_cand;
       candidates[tpos] &= ~other_cand;
    }
    if (type == 0 ) {
        updated.set_indexbits(0x7,off,3);
        new_grid_state->updated.set_indexbits(7,off,3);
    } else {
        updated.set_indexbits(0x40201,off,19);
        new_grid_state->updated.set_indexbits(0x40201,off,19);
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
    guesses++;

    return new_grid_state;
}

// this version of make_guess is the simplest and original form of making a guess.
//
template<Verbosity verbose>
inline GridState* make_guess(bit128_t &bivalues) {
    // Find a cell with the least candidates. The first cell with 2 candidates will suffice.
    // Pick the candidate with the highest value as the guess.
    // Save the current grid state (with the chosen candidate eliminated) for tracking back.

    // Find the cell with fewest possible candidates
    unsigned char guess_index = 0;
    unsigned char cnt;
    unsigned char best_cnt = 16;

    if ( bivalues.u64[0] ) {
        guess_index = _tzcnt_u64(bivalues.u64[0]);
    } else if ( bivalues.u64[1] ) {
        guess_index = _tzcnt_u64(bivalues.u64[1]) + 64;
    } else {
        // very unlikely
        bit128_t to_visit = unlocked;
        unsigned char i;
        while ( best_cnt > 3 && to_visit ) {
            i = tzcnt_and_mask(to_visit);
            cnt = __popcnt16(candidates[i]);
            if (cnt < best_cnt) {
                best_cnt = cnt;
                guess_index = i;
            }
        }
    }

    // Find the first candidate in this cell (lsb set)
    // Note: using tzcnt would be equally valid; this pick is historical
    unsigned short digit = 0x8000 >> __lzcnt16(candidates[guess_index]);

    return make_guess<verbose>(guess_index, digit);
}

// this version of make_guess takes a cell index and digit for the guess
//
template<Verbosity verbose>
inline GridState* make_guess(unsigned char guess_index, unsigned short digit ) {
    // Create a copy of the state of the grid to make back tracking possible
    GridState* new_grid_state = this+1;
    if ( stackpointer >= GRIDSTATE_MAX-1 ) {
        fprintf(stderr, "Error: no GridState object availabe\n");
        exit(0);
    }
    memcpy(new_grid_state, this, sizeof(GridState));
    new_grid_state->stackpointer++;

    // Remove the guessed candidate from the old grid
    // when we get back here to the old grid, we know the guess was wrong
    candidates[guess_index] &= ~digit;

    updated.set_indexbit(guess_index);

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
    guesses++;

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

template<Kind kind>
inline unsigned char *SolverData::getSectionSetUnlocked(GridState &gs) {
        if ( !sectionSetUnlockedValid[kind] ) {
            bit128_t tmp[9];
            __m128i set23 = *(__m128i *)&gs.set23_found[kind];
            for ( int i=0; i<9; i++ ) {
                tmp[i].m128 = _mm_andnot_si128(set23, *(__m128i *)&small_index_lut[i][kind]);
            }
            for ( int i=0; i<9; i++ ) {
                sectionSetUnlocked[kind][i] = tmp[i].popcount();
            }
            sectionSetUnlockedValid[kind] = true;
        }
        return sectionSetUnlocked[kind];
}


template <Verbosity verbose>
Status solve(signed char grid[81], GridState stack[], int line) {

    GridState *grid_state = &stack[0];
    unsigned long long *unlocked = grid_state->unlocked.u64;
    unsigned short* candidates;

    Status status;

    SolverData solverData {};

#ifdef OPT_UQR
    // make_guess accepts a lambda, which will use guess_message to pass along
    // two strings of debug information:
    char guess_message[2][196];

    // original_locked keeps track of the presets
    // since unique rectangles can only be invalidated by presets.
    bit128_t original_locked;
    original_locked.u64[0] = ~unlocked[0];
    original_locked.u64[1] = ~unlocked[1] & 0x1ffff;

    bit128_t original_locked_transposed;

    unsigned short superimposed_preset_rows[3][3];    // deal with initialization later...
    unsigned short superimposed_preset_cols[3][3];
    bool have_superimposed_preset_rows = false;
    bool have_superimposed_preset_cols = false;

    unsigned short last_entered_count_uqr = 0;
    unsigned char last_band_uqr = 0;
#endif

    // the low byte is the real count, while
    // the high byte is increased/decreased with each guess/back track
    // init count of resolved cells to the size of the initial set
    unsigned short current_entered_count = 81 - _popcnt64(unlocked[0]) - _popcnt32(unlocked[1]);

#ifdef OPT_TRIAD_RES
    unsigned short last_entered_count_col_triads = 0;
#endif

    int unique_check_mode = 0;
    bool nonunique_reported = false;

    unsigned long long my_digits_entered_and_retracted = 0;
    unsigned long long my_naked_sets_searched = 0;
    unsigned long long my_unique_rectangles_checked = 0;
    unsigned char no_guess_incr = 1;

    unsigned int my_past_naked_count = 0;

    if ( verbose == VDebug ) {
        candidates = grid_state->candidates;
        char gridout[82];
        for (unsigned char j = 0; j < 81; ++j) {
            if ( grid_state->unlocked.check_indexbit(j) ) {
                gridout[j] = '0';
            } else {
                gridout[j] = 49+_tzcnt_u32(candidates[j]);
            }
        }
        printf("Line %d: %.81s\n", line, gridout);
    }

    // The 'API' for code that uses the 'goto enter:' method of entering digits
    unsigned short e_digit = 0;
    unsigned char e_i = 0;

#ifdef OPT_FSH
    unsigned short exclude_row[9];
    unsigned short exclude_col[9];
#endif

    bool check_back = thorough_check;

#ifdef OPT_SETS
    unsigned short flip = 1;  // for naked sets search to support a search with 50% reduction of tests
#endif

    goto start;

back:

    // Each algorithm (naked single, hidden single, naked set)
    // has its own non-solvability detecting trap door to detect the grid is bad.
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
                printf("Line %d: No solution found!\n", line);
            }
            unsolved_count++;
        }
        // cleanup and return
        if ( verbose != VNone && reportstats ) {
            past_naked_count += my_past_naked_count;
            naked_sets_searched += my_naked_sets_searched;
            digits_entered_and_retracted += my_digits_entered_and_retracted;
            unique_rectangles_checked += my_unique_rectangles_checked;
            if ( status.unique == false ) {
                non_unique_count++;
            }
        }
        // failed - just copy the input
        memcpy(grid, grid-82, 81);
        return status;
    }

    current_entered_count  = ((grid_state-1)->stackpointer)<<8;        // back to previous stack.
    current_entered_count += _popcnt64((grid_state-1)->unlocked.u64[0]) + _popcnt32((grid_state-1)->unlocked.u64[1]);

    // collect some guessing stats
    if ( verbose != VNone && reportstats ) {
        my_digits_entered_and_retracted +=
            (_popcnt64((grid_state-1)->unlocked.u64[0] & ~grid_state->unlocked.u64[0]))
          + (_popcnt32((grid_state-1)->unlocked.u64[1] & ~grid_state->unlocked.u64[1]));
    }

    // Go back to the state when the last guess was made
    // This state had the guess removed as candidate from it's cell

    if ( verbose == VDebug ) {
        printf("back track to level >%d<\n", grid_state->stackpointer-1);
    }
    trackbacks++;
    grid_state--;

start:

    e_digit = 0;

    // at start, set everything that depends on grid_state:
    check_back = grid_state->stackpointer || thorough_check || rules != Regular || unique_check_mode;

    unlocked   = grid_state->unlocked.u64;
    candidates = grid_state->candidates;

#ifdef OPT_FSH
    if ( mode_fish ) {
        memset (exclude_row, 0, 9*sizeof(unsigned short));
        memset (exclude_col, 0, 9*sizeof(unsigned short));
    }
#endif

// algorithm 0:
// Enter a digit into the solution by setting it as the value of cell and by
// removing the cell from the set of unlocked cells.  Update all affected cells
// by removing any now impossible candidates.
//
// algorithm 1:
// for each cell check whether it has a single candidate: enter that candidate as
// the solution.
// for each cell check whether a cell has no candidates: back track.
//
// Below, two cases are implemented: a) cover algorithm 0 _and_ 1, or b) algorithm 1 alone.
//
enter:

    if ( e_digit ) {
        // inlined flavor of enter_digit
        //
        // lock this cell and and remove this digit from the candidates in this row, column and box
        // and for good measure, detect 0s (back track) and singles.
        bit128_t to_update = {0};

        if ( verbose == VDebug ) {
            printf(" %x at %s\n", _tzcnt_u32(e_digit)+1, cl2txt[e_i]);
        }
#ifndef NDEBUG
        if ( __popcnt16(e_digit) != 1 ) {
            if ( warnings != 0 ) {
                printf("error in e_digit: %x\n", e_digit);
            }
        }
#endif

        if (e_i < 64) {
            _bittestandreset64((long long int *)&unlocked[0], e_i);
        } else {
            _bittestandreset64((long long int *)&unlocked[1], e_i-64);
        }

        candidates[e_i] = e_digit;
        current_entered_count++;

        add_and_mask_all_indices(&to_update, &grid_state->unlocked, e_i);

        grid_state->updated.u128 |= to_update.u128;

        __m256i mask_neg = _mm256_set1_epi16(e_digit);

        unsigned short dtct_j = 0;
        unsigned int dtct_m = 0;
        for (unsigned char j = 0; j < 80; j += 16) {
            __m256i c = _mm256_load_si256((__m256i*) &candidates[j]);
            // expand unlocked unsigned short to boolean vector
            __m256i munlocked = expand_bitvector(to_update.u16[j>>4]);
            // apply mask (remove bit), preserving the locked cells
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
            }
        }
        if ( dtct_m ) {
            int idx = _tzcnt_u32(dtct_m);
            e_i = idx+dtct_j;
            e_digit = candidates[e_i];
            if ( e_digit ) {
                if ( verbose == VDebug ) {
                    printf("naked  single      ");
                }
                goto enter;
            } else {
                if ( verbose != VNone ) {
                    unsigned char pos = e_i;
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
        }
        e_digit = 0;
    } else {
        // no digit to enter
        for (unsigned char i = 0; i < 80; i += 16) {
            unsigned short m = ((bit128_t*)unlocked)->u16[i>>4];
            if ( m ) {
                __m256i c = _mm256_load_si256((__m256i*) &candidates[i]);
                // remove least significant digit and compare to 0:
                // (c & (c-1)) == 0  => naked single
                __m256i a = _mm256_cmpeq_epi16(_mm256_and_si256(c, _mm256_sub_epi16(c, ones)), _mm256_setzero_si256());
                // Check if any cell has zero candidates
                if (__builtin_expect (check_back && _mm256_movemask_epi8(_mm256_cmpeq_epi16(c, _mm256_setzero_si256())),0)) {
                    // Back track, no solutions along this path
                    if ( verbose != VNone ) {
                        unsigned int mx = _mm256_movemask_epi8(_mm256_cmpeq_epi16(c, _mm256_setzero_si256()));
                        unsigned char pos = i+(_tzcnt_u32(mx)>>1);
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
                unsigned int mask = and_compress_masks<false>(a,m);
                if ( mask ) {
                    int idx = _tzcnt_u32(mask);
                    e_i = idx+i;
                    e_digit = candidates[e_i];
                    if ( verbose == VDebug ) {
                        printf("naked  single      ");
                    }
                    goto enter;
                }
            }
        }
    }
    if (unlocked[1] & (1ULL << (80-64))) {
        if (__builtin_expect (candidates[80] == 0,0) ) {
            // no solutions go back
            if ( verbose != VNone ) {
                if ( grid_state->stackpointer == 0 && unique_check_mode == 0 ) {
                    if ( warnings != 0 ) {
                        printf("Line %d: cell %s is 0\n", line, cl2txt[80]);
                    }
                } else {
                    if ( debug ) {
                        printf("back track - cell [8,8] is 0\n");
                    }
                }
            }
            goto back;
        } else if (__popcnt16(candidates[80]) == 1) {
            // Enter the digit and update candidates
            if ( verbose == VDebug ) {
                printf("naked  single      ");
            }
            e_i = 80;
            e_digit = candidates[80];
            goto enter;
        }
    }
    // The solving algorithm ends when there are no remaining unlocked cell.
    // The finishing tasks include verifying the solution and/or confirming
    // its uniqueness, if requested.
    //
    // Check if it's solved, if it ever gets solved it will be solved after looking for naked singles
    bool verify_one = false;
    if ( *(__uint128_t*)unlocked == 0) {
        // Solved it
        if ( rules == Multiple && (unique_check_mode == 1 || grid_state->flags < 0) ) {
            if ( !nonunique_reported ) {
                if ( verbose != VNone && reportstats && warnings != 0 ) {
                    printf("Line %d: solution to puzzle is not unique\n", line);
                }
                nonunique_reported = true;
            }
            verify_one = true;
            status.unique = false;
        }
        if ( verify || verify_one) {
            verify_one = false;
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
                    unsolved_count++;
                    not_verified_count++;
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
                if ( reportstats ) {
                    if ( unique_check_mode == 0 ) {
                        verified_count++;
                    }
                }
            }
        }
        if ( verbose != VNone && reportstats ) {
            if ( unique_check_mode == 0 ) {
                solved_count++;
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
                _mm256_storeu_si256((__m256i_u*)&grid[j], _mm256_min_epu8(t1, t2));
            }
            __m256i tmp = _mm256_and_si256(*(__m256i*)&candidates[64],maskff);
            __m128i t1 = _mm256_castsi256_si128(_mm256_packus_epi16(tmp,_mm256_permute2x128_si256(tmp,tmp,0x11)));
            __m128i t2 = _mm_and_si128(_mm_srli_epi16(t1, 4),_mm256_castsi256_si128(nibble_mask));
            t1 = _mm_and_si128( t1, _mm256_castsi256_si128(nibble_mask));
            t2 = _mm_shuffle_epi8(_mm256_castsi256_si128(lut_hi), t2);
            t1 = _mm_shuffle_epi8(_mm256_castsi256_si128(lut_lo), t1);
            _mm_storeu_si128((__m128i_u*)&grid[64], _mm_min_epu8(t1, t2));
            grid[80] = '1'+_tzcnt_u32(candidates[80]);
        }

        if ( verbose != VNone && reportstats ) {
            no_guess_cnt += no_guess_incr;
        }
        status.guess = no_guess_incr?false:true;

        if ( grid_state->stackpointer && rules == Multiple && grid_state->flags >= 0 ) {
            if ( verbose == VDebug ) {
                printf("Solution: %.81s\nBack track to determine uniqueness\n", grid);
            }
            unique_check_mode = 1;
            goto back;
        }
        // otherwise uniqueness checking is complete
        if ( verbose != VNone && reportstats ) {
            past_naked_count += my_past_naked_count;
            naked_sets_searched += my_naked_sets_searched;
            digits_entered_and_retracted += my_digits_entered_and_retracted;
            unique_rectangles_checked += my_unique_rectangles_checked;
            if ( !status.unique ) {
                non_unique_count++;
            }
        }
        return status;
    }

    // reset solverData
    solverData.bivaluesValid = false;
    solverData.cbbvsValid = false;
    solverData.sectionSetUnlockedValid[0] = solverData.sectionSetUnlockedValid[1] = 
                                            solverData.sectionSetUnlockedValid[2] = false;
    solverData.guess_hint_digit = 0;
    my_past_naked_count++;

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
    // columns are or'ed together, leaving out the current row, and hidden column singles are
    // isolated for that row. For efficiency, precompute and save the or'ed rows (tails) and
    // preserve the last leading set of rows (the head).
    // To check for an invalid state of the puzzle:
    // - check the columns or'ed value to be 0x1ff.
    //
    // Combine 8 cells from rows and 8 cells from boxes into one __m256i vector.
    // Rotate and or until each vector element represents 7 cells or'ed (except the cell
    // directly corresponding to its position, containing the hidden single if there is one).
    // Broadcast the nineth cell and or it for good measure, then andnot with 0x1ff
    // to isolate the hidden singles.
    // For the nineth cell, rotate and or one last time and use just one element of the result
    // to check the nineth cell for a hidden single.
    //
    // Compress to a bit vector and check that 'found singles' are really singles. 
    //
    // Algorithm 3
    // Definition: An intersection of row/box and col/box consisting of three cells
    // are called "triad" in the following.
    // There are 27 horizontal (row-based) triads and 27 vertical (col-based) triads.
    // For each band of three aligned boxes there are nine triads.
    // The collection of the triads data (Algorithm 3 Part 1) is intermingled with
    // that of algorithm 2 for speed.
    // Triads in their own right are significant for two reasons:
    // First, a triad that has 3 candidates is a special case of set that is easily detected.
    // The detection occurs in part 2 of Algorithm 3 by running a popcount on the collected
    // triad candidates.  The result is kept in form of a bitvector of 'unlocked triads'
    // for rows and columns.  For columns, the unlocked state is determined directly from
    // the general unlocked bit vector.  For rows, the order in which the row triads are
    // stored is not compatible with the general unlocked bit vector.  Instead, all locked
    // row triads are individually removed from their row unlocked bit vector.
    // Part 2 of Algorithm 3 allows to determine which candidate value can only occur
    // in a specific triad and not in the other triads of the row/column and box.
    // Part 3 of Algorithm 3 is optional.
    // It allows to detect all fully resolved triads.
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
    // - Any candidate only occurring in a a triad (must) because if doesn't occur in
    //   either its box peers or its row/column peers cannot occur in any of the pair of peers.
    // - Since 3 is the upper limit of must candidates, once these three are known from
    //   part 2 of the algorithm, these become the fixed 'must' set for the triad.
    //   The triad is then called "resolved".
    //
    // We can start with the mustnt for all triads of a band of boxes, compute the intersection
    // pairs of peer-mustnt (horizontally and vertically) and join the two results.
    // This is then a "must" of the given triad, and augmented by the must if the triad is resolved.
    // Using the peers musts, their union is a "must not" for the given triad.
    // By joining it to the orginal "must not" we obtain an equal or super set for the new "must not".
    // This "must not" in turn can then be be applied to remove the excessive candidates from
    // the triad.
    //
    // These computations are easily implemented as SIMD operations.
    //
    // To recap: parts 1 - 3 of Algorithm 3 are fast and effective - especially if the
    // hidden single search is leveraged for the initial union of the triads' candidates.
    //
    // Part 2 and 3 are invoked after Algorithm 2 fails to detect any hidden single.
    //
    // col_triads consist of 9 triads per horizontal band (one vertical triad for each column);
    // row_triads are captured 3 per row, 3 rows stacked vertically with the subsequent 3 rows
    // offset by 3 (using the canonical numbering of row triads):
    //    0,  1,  2,   9, 10, 11, 18, 19, 20, -
    //    3,  4,  5,  12, 13, 14, 21, 22, 23, -
    //    6,  7,  8,  15, 16, 17, 24, 25, 26
    // SIMD Triad processing requires all triads of a given band to be positioned
    // 3 horizontal triads side by side and 3 vertical triads in the same positions
    // of the two other rows.  The ordering described above satisfies this requirement.
    //
    // For technical reasons, an (unused) 10th triad is injected between each set of 9 triads
    // for a total of 29 triad values.
    //
    // various methods are used to efficiently save the triads in the processing order,
    // which go beyond the end of the arrays normally needed.

    TriadInfo &triad_info = solverData.triadInfo;

    // Algo 2 and Algo 3.1
    {

        __m256i column_or_tails[9];
        __m256i column_or_head = _mm256_setzero_si256();
        __m256i col_triads_3, col_triads_2;
        __m256i column_mask {};

        // columns
        // to start, we simply tally the or'ed rows
        signed char j = 81-9;
        // compute fresh col_triads_3
        col_triads_3 = _mm256_setzero_si256();
        // A2 (cols)
        // precompute 'tails' of the or'ed column_mask only once
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
        column_mask = _mm256_or_si256(col_triads_2, col_triads_3);

        // 2 iterations, rows 1 and 2.
        // the first set of column triads is computed below as part of the computed
        // 'head' or.
        for ( ; j > 0; j -= 9) {
            column_or_tails[j/9-1] = column_mask;
            column_mask = _mm256_or_si256(column_mask, *(__m256i_u*) &candidates[j]);
        }
        // or in row 0 and check whether all digits or covered
        if ( check_back && !_mm256_testz_si256(mask9x1ff,_mm256_andnot_si256(_mm256_or_si256(column_mask, *(__m256i*) &candidates[0]), mask9x1ff)) ) {
            // the current grid has no solution, go back
            if ( verbose != VNone ) {
                unsigned int m = _mm256_movemask_epi8(_mm256_cmpgt_epi16(_mm256_andnot_si256(_mm256_or_si256(column_mask, *(__m256i*) &candidates[0]), mask9x1ff), _mm256_setzero_si256()));
                int idx = __tzcnt_u32(m)>>1;
                if ( grid_state->stackpointer == 0 && unique_check_mode == 0 ) {
                    if ( warnings != 0 ) {
                        printf("Line %d: stack 0, back track - column %d does not contain all digits\n", line, idx);
                    }
                } else if ( debug ) {
                    printf("back track - missing digit in column %d\n", idx);
                }
            }
            goto back;
        }

        // breaking the column hidden singles out of the loop this way will win some performance
        __m256i column_or_head_ = _mm256_setzero_si256();
        unsigned int jrow = 0;
        __m256i column_mask_ = column_mask;
        for (unsigned int j = 0; j < 81; j+=9, jrow++) {
            // turn the or'ed rows into a mask for the singletons, if any.
            __m256i column_mask_neg = _mm256_andnot_si256(column_mask_, mask9x1ff);
            // check col (9) candidates
            unsigned short m = (j < 64) ? (unlocked[0] >> j) : (unlocked[1] >> (j-64));
            if ( j > 64-9) {
                m |= unlocked[1] << (64-j);
            }
            __m256i a = _mm256_cmpgt_epi16(column_mask_neg, _mm256_setzero_si256());
            unsigned int mask = and_compress_masks<false>(a, m & 0x1ff);
            if ( mask) {
                int idx = __tzcnt_u32(mask);
                e_i = j+idx;
                e_digit = ((v16us)column_mask_neg)[idx];
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
                    e_i = 0;
                    goto back;
                }
                if ( verbose == VDebug ) {
                    printf("hidden single (col)");
                }
                goto enter;
            }

            // leverage previously computed or'ed rows in head and tails.
            column_or_head_ = column_mask_ = _mm256_or_si256(*(__m256i_u*) &candidates[j], column_or_head_);
            column_mask_ = _mm256_or_si256(column_or_tails[jrow],column_mask_);
        }

        unsigned short cand9_row = 0;
        unsigned short cand9_box = 0;
        unsigned char irow = 0;

        __v16hu rowbox_9th_mask;

        for (unsigned char i = 0; i < 81; i += 9, irow++) {
            // turn the or'ed rows into a mask for the singletons, if any.
            column_mask = _mm256_andnot_si256(column_mask, mask9x1ff);

            // rows and boxes

            unsigned char b = box_start_by_boxindex[irow];
             __m256i c = _mm256_set_m128i(_mm_set_epi16(candidates[b+19], candidates[b+18], candidates[b+11], candidates[b+10], candidates[b+9], candidates[b+2], candidates[b+1], candidates[b]),
                         *(__m128i_u*) &candidates[i]);

            unsigned short the9thcand_row = candidates[i+8];
            unsigned short the9thcand_box = candidates[b+20];

            __m256i rowbox_or7 = _mm256_setzero_si256();
            __m256i rowbox_or8;
            __m256i rowbox_9th = _mm256_set_m128i(_mm_set1_epi16(the9thcand_box),_mm_set1_epi16(the9thcand_row));

            __m256i row_triad_capture[2];
            {
                __m256i c_ = c;
                // A2 and A3.1.b
                // first lane for the row, 2nd lane for the box
                // step j=0
                // rotate left (0 1 2 3 4 5 6 7) -> (1 2 3 4 5 6 7 0)
                    c_ = _mm256_alignr_epi8(c_, c_, 2);
                    rowbox_or7 = _mm256_or_si256(c_, rowbox_or7);
                // step j=1
                // rotate (1 2 3 4 5 6 7 0) -> (2 3 4 5 6 7 0 1)
                c_ = _mm256_alignr_epi8(c_, c_, 2);
                rowbox_or7 = _mm256_or_si256(c_, rowbox_or7);
                // triad capture: after 2 rounds, row triad 3 of this row saved in pos 5
                row_triad_capture[0] = _mm256_or_si256(rowbox_or7, rowbox_9th);
                // step j=2
                // rotate (2 3 4 5 6 7 0 1) -> (3 4 5 6 7 0 1 2)
                c_ = _mm256_alignr_epi8(c_, c_, 2);
                // triad capture: after 3 rounds 2 row triads 0 and 1 in pos 7 and 2
                row_triad_capture[1] = rowbox_or7 = _mm256_or_si256(c_, rowbox_or7);
                // continue the rotate/or routine for this row
                for (unsigned char j = 3; j < 7; ++j) {
                    // rotate (0 1 2 3 4 5 6 7) -> (1 2 3 4 5 6 7 0)
                    // first lane for the row, 2nd lane for the box
                    c_ = _mm256_alignr_epi8(c_, c_, 2);
                    rowbox_or7 = _mm256_or_si256(c_, rowbox_or7);
                }

                rowbox_or8 = _mm256_or_si256(c, rowbox_or7);
                // test rowbox_or8 | rowbox_9th to hold all the digits
                if ( check_back ) {
                    if ( !_mm256_testz_si256(mask1ff,_mm256_andnot_si256(_mm256_or_si256(rowbox_9th, rowbox_or8), mask1ff))) {
                        // the current grid has no solution, go back
                        if ( verbose != VNone ) {
                            unsigned int m = _mm256_movemask_epi8(_mm256_cmpeq_epi16(_mm256_setzero_si256(),
                            _mm256_andnot_si256(_mm256_or_si256(rowbox_9th, rowbox_or8), mask1ff)));
                            const char *row_or_box = (m & 0xffff)?"box":"row";
                            if ( grid_state->stackpointer == 0 && unique_check_mode == 0 ) {
                                if ( warnings != 0 ) {
                                    printf("Line %d: stack 0, back track - %s %d does not contain all digits\n", line, row_or_box, irow);
                                }
                            } else if ( debug ) {
                                printf("back track - missing digit in %s %d\n", row_or_box, irow);
                            }
                        }
                        goto back;
                    }
                }
            }
            // hidden singles in row and box
            __m256i rowbox_mask = _mm256_andnot_si256(_mm256_or_si256(rowbox_9th, rowbox_or7), mask1ff);
            {
                // Check that the singles are indeed singles
                if ( check_back && !_mm256_testz_si256(rowbox_mask, _mm256_sub_epi16(rowbox_mask, ones))) {
                    // This is rare as it can only occur when a wrong guess was made or the puzzle is invalid.
                    // the current grid has no solution, go back
                    if ( verbose != VNone ) {
                        unsigned int m = _mm256_movemask_epi8(_mm256_cmpgt_epi16(_mm256_and_si256(rowbox_mask,
                                         _mm256_sub_epi16(rowbox_mask, ones)), _mm256_setzero_si256()));
                        const char *row_or_box = (m & 0xffff)?"row":"box";
                        int idx = __tzcnt_u32(m)>>1;
                        unsigned char celli = (m & 0xffff)? irow*9+idx:b+box_offset[idx&7];
                        if ( grid_state->stackpointer == 0 && unique_check_mode == 0 ) {
                            if ( warnings != 0 ) {
                                printf("Line %d: stack 0, back track - %s cell %s does contain multiple hidden singles\n", line, row_or_box, cl2txt[celli]);
                            }
                        } else if ( debug ) {
                            printf("back track - multiple hidden singles in %s cell %s\n", row_or_box, cl2txt[celli]);
                        }
                    }
                    goto back;
                }

                // check row/box (8) candidates
                unsigned short m = grid_state->unlocked.get_indexbits(i, 8) | ((get_contiguous_masked_indices_for_box(unlocked,irow)&0xff)<<8);
                __m256i a = _mm256_cmpgt_epi16(rowbox_mask, _mm256_setzero_si256());
                unsigned int mask = and_compress_masks<false>(a, m);
                if (mask) {
                    int s_idx = __tzcnt_u32(mask);
                    bool is_row = s_idx < 8;
                    int celli = is_row ? i + s_idx : b + box_offset[s_idx&7];
                    if ( verbose == VDebug ) {
                        printf("hidden single (%s)", is_row?"row":"box");
                    }
                    e_i = celli;
                    e_digit = ((v16us)rowbox_mask)[s_idx];
                    goto enter;
                }
            } // row/box

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

            cand9_row = ~((v16us)rowbox_or8)[0] & the9thcand_row;
            cand9_box = ~((v16us)rowbox_or8)[8] & the9thcand_box;

            if ( i == 72 ) {
                break;
            }

            // for the next iteration, leverage previously computed or'ed rows in head and tails.
            column_or_head = column_mask = _mm256_or_si256(*(__m256i_u*) &candidates[i], column_or_head);
            column_mask = _mm256_or_si256(column_or_tails[irow],column_mask);
            if ( i == 18 ) {
                // A3.1.c
                // column_or_head now contains the or'ed rows 0-2.
                // Store all column triads sequentially, paying attention to overlap.
                //
                _mm256_storeu_si256((__m256i *)triad_info.col_triads, column_or_head);
                _mm256_storeu_si256((__m256i *)(triad_info.col_triads+10), col_triads_2);
                // mask 5 hi triads to 0xffff, which will not trigger any checks
                _mm256_storeu_si256((__m256i *)(triad_info.col_triads+20), _mm256_or_si256(col_triads_3, mask11hi));
            }

            if ( irow < 8 ) {
                rowbox_9th_mask[irow]   = cand9_row;
                rowbox_9th_mask[irow+8] = cand9_box;
            }

        }   // for

        // check row/box 9th candidates
        unsigned int mask = compress_epi16_boolean<true>(_mm256_cmpgt_epi16((__m256i)rowbox_9th_mask, _mm256_setzero_si256()));
        while (mask) {
            int s_idx = __tzcnt_u32(mask) >> 1;
            bool is_row = s_idx < 8;
            unsigned char celli = is_row ? s_idx*9+8 : box_start_by_boxindex[s_idx&7] + 20;
            unsigned short cand = ((v16us)rowbox_9th_mask)[s_idx];
            if ( ((bit128_t*)unlocked)->check_indexbit(celli) ) {
                // check for a single.
                // This is rare as it can only occur when a wrong guess was made.
                // the current grid has no solution, go back
                if ( check_back && cand & (cand-1) ) {
                    if ( verbose != VNone ) {
                        if ( grid_state->stackpointer == 0 && unique_check_mode == 0 ) {
                            if ( warnings != 0 ) {
                                printf("Line %d: stack 0, multiple hidden singles in row/box cell %s\n", line, cl2txt[80]);
                            }
                        } else if ( debug ) {
                            printf("back track - multiple hidden singles in row/box cell %s\n", cl2txt[80]);
                        }
                    }
                    goto back;
                }
                if ( verbose == VDebug ) {
                    printf("hidden single (%s)", is_row?"row":"box");
                }
                e_i = celli;
                e_digit = cand;
                goto enter;
            }
            mask &= ~(3<<(s_idx<<1));
        }

        // cell 80 is coincidently the nineth cell of the 9th row and the 9th box
        unsigned cand80 = cand9_row | cand9_box;

        if ( cand80 && ((bit128_t*)unlocked)->check_indexbit(80) ) {
            // check for a single.
            // This is rare as it can only occur when a wrong guess was made.
            // the current grid has no solution, go back
            if ( check_back && (cand80 & (cand80-1)) ) {
                if ( verbose != VNone ) {
                    if ( grid_state->stackpointer == 0 && unique_check_mode == 0 ) {
                        if ( warnings != 0 ) {
                            printf("Line %d: stack 0, multiple hidden singles in row/box cell %s\n", line, cl2txt[80]);
                        }
                    } else if ( debug ) {
                        printf("back track - multiple hidden singles in row/box cell %s\n", cl2txt[80]);
                    }
                }
                goto back;
            }
            if ( verbose == VDebug ) {
                printf("hidden single (%s)", (cand80 == cand9_row) ? "row":"box");
            }
            e_i = 80;
            e_digit = cand80;
            goto enter;
        } // cell 80

    } // Algo 2 and Algo 3.1

#ifdef OPT_TRIAD_RES
    if ( mode_triad_res )
    { // Algo 3.3

            // A3.2.1 (check col-triads)
            // just a plain old popcount, but for both vectors interleaved
            __m256i v1 = _mm256_loadu_si256((__m256i *)triad_info.col_triads);
            __m256i v2 = _mm256_loadu_si256((__m256i *)(triad_info.col_triads+16));
            __m256i lo1 = _mm256_and_si256 (v1, nibble_mask);
            __m256i hi1 = _mm256_and_si256 (_mm256_srli_epi16 (v1, 4), nibble_mask );
            __m256i cnt11 = _mm256_shuffle_epi8 (lookup, lo1);
            __m256i cnt12 = _mm256_shuffle_epi8 (lookup, hi1);
            cnt11 = _mm256_add_epi8 (cnt11, cnt12);
            __m256i res = _mm256_add_epi8 (cnt11, _mm256_bsrli_epi128(cnt11, 1));
            __m256i lo2 = _mm256_and_si256 (v2, nibble_mask);
            __m256i hi2 = _mm256_and_si256 (_mm256_srli_epi16 (v2, 4), nibble_mask );
            __m256i cnt21 = _mm256_shuffle_epi8 (lookup, lo2);
            __m256i cnt22 = _mm256_shuffle_epi8 (lookup, hi2);
            cnt21 = _mm256_add_epi8 (cnt21, cnt22);
            res = _mm256_and_si256 (res, maskff );
            res = _mm256_packus_epi16(res, _mm256_and_si256 (_mm256_add_epi8 (cnt21, _mm256_bsrli_epi128(cnt21, 1)), maskff ));
            res = _mm256_and_si256(_mm256_permute4x64_epi64(res, 0xD8), mask27);

            // do this only if unlocked has been updated:
            if ( last_entered_count_col_triads != current_entered_count) {
                // the high byte is set to the stackpointer, so that this works
                // across guess/backtrack
                last_entered_count_col_triads = current_entered_count;
                unsigned long long tr = unlocked[0];  // col triads - set if any triad cell is unlocked
                tr |= (tr >> 9) | (tr >> 18) | (unlocked[1]<<(64-9)) | (unlocked[1]<<(64-18));
                // mimick the pattern of col_triads, i.e. a gap of 1 after each group of 9.
                if ( bmi2_support ) {
                    tr = _pext_u64(tr, 0x1ffLL | (0x1ffLL<<27) | (0x1ffLL<<54));
                    grid_state->triads_unlocked[Col] &= _pdep_u64(tr, (0x1ff)|(0x1ff<<10)|(0x1ff<<20));
                } else {
                    grid_state->triads_unlocked[Col] &= (tr & 0x1ff) | ((tr >> 17) & (0x1ff<<10)) | ((tr >> 34) & (0x1ff<<20));
                }
            }

            triad_info.triads_selection[Col] = _mm256_movemask_epi8(_mm256_cmpeq_epi8(res, fours));
            unsigned long long m = _mm256_movemask_epi8(_mm256_cmpeq_epi8(res, threes)) & grid_state->triads_unlocked[Col];
            while (m) {
                unsigned char tidx = tzcnt_and_mask(m);
                unsigned short cands_triad = triad_info.col_triads[tidx];
                if ( verbose == VDebug ) {
                    char ret[32];
                    format_candidate_set(ret, cands_triad);
                    printf("triad set (col): %-9s %s\n", ret, cl2txt[tidx/10*3*9+tidx%10]);
                }
                // mask off resolved triad:
                grid_state->triads_unlocked[Col] &= ~(1LL << tidx);
                unsigned char off = tidx%10+tidx/10*27;
                grid_state->set23_found[Col].set_indexbits(0x40201,off,19);
                grid_state->set23_found[Box].set_indexbits(0x40201,off,19);
                triads_resolved++;
            }

            // A3.2.2 (check row-triads)

            // just a plain old popcount, but for both vectors interleaved
            v1 = _mm256_loadu_si256((__m256i *)triad_info.row_triads);
            v2 = _mm256_loadu_si256((__m256i *)(triad_info.row_triads+16));

            lo1 = _mm256_and_si256 (v1, nibble_mask);
            hi1 = _mm256_and_si256 (_mm256_srli_epi16 (v1, 4), nibble_mask );
            cnt11 = _mm256_shuffle_epi8 (lookup, lo1);
            cnt12 = _mm256_shuffle_epi8 (lookup, hi1);
            cnt11 = _mm256_add_epi8 (cnt11, cnt12);
            res = _mm256_add_epi8 (cnt11, _mm256_bsrli_epi128(cnt11, 1));
            lo2 = _mm256_and_si256 (v2, nibble_mask);
            hi2 = _mm256_and_si256 (_mm256_srli_epi16 (v2, 4), nibble_mask );
            cnt21 = _mm256_shuffle_epi8 (lookup, lo2);
            cnt22 = _mm256_shuffle_epi8 (lookup, hi2);
            cnt21 = _mm256_add_epi8 (cnt21, cnt22);
            res = _mm256_and_si256 (res, maskff );
            res = _mm256_packus_epi16(res, _mm256_and_si256 (_mm256_add_epi8 (cnt21, _mm256_bsrli_epi128(cnt21, 1)), maskff ));
            res = _mm256_and_si256(_mm256_permute4x64_epi64(res, 0xD8), mask27);

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
                    grid_state->triads_unlocked[Row] &= ~(1<<tidx);
                    continue;
                }

                if ( verbose == VDebug ) {
                    char ret[32];
                    format_candidate_set(ret, triad_info.row_triads[tidx]);
                    printf("triad set (row): %-9s %s\n", ret, cl2txt[off]);
                }

                // mask off resolved triad:
                grid_state->triads_unlocked[Row] &= ~(1LL << tidx);
                grid_state->set23_found[Row].set_indexbits(0x7,off,3);
                grid_state->set23_found[Box].set_indexbits(0x7,off,3);
                triads_resolved++;
            } // while
    } else {
        // by default, all triads will be checked
        triad_info.triads_selection[Col] = triad_info.triads_selection[Row] = 0x1ff | (0x1ff<<10) | (0x1ff<<20);
    }   // Algo 3 Part 3
#else
    // by default, all triads will be checked
    triad_info.triads_selection[Col] = triad_info.triads_selection[Row] = 0x1ff | (0x1ff<<10) | (0x1ff<<20);
#endif


    {   // Algo 3 Part 2
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
                    grid_state->updated.set_indexbits(m | (m<<9) | (m<<18), i*27, 27);

                }
                // iterate over triads to update
                while (m) {
                    // 1. Check for triad resolution
                    // 2. log the triad update from above
                    // 3. for columns, track resolved triads
                    unsigned char i_rel = tzcnt_and_mask(m);
                    unsigned char ltidx = i_rel + 9*i;  // logical triad index (within band, different layouts for row/col)
                    // compute bit offsets of the resolved triad
                    // and save the mask
                    if ( _popcnt32(((__v16hu)tmustnt[i])[i_rel]) == 6 ) {
                        if ( type == 0 ) {
                            rslvd_row_combo_tpos[i_rel/3] |= bandbits_by_index[row_triad_canonical_map[ltidx]%9];
                        } else {
                            rslvd_col_combo_tpos |= 0x40201<<i_rel;
                        }
                        triads_resolved++;
                    }
                    if ( verbose == VDebug ) {
                        char ret[32];
                        format_candidate_set(ret, ((__v16hu)to_remove_v)[i_rel]);
                        printf("remove %-5s from %s triad at %s\n", ret, type == 0? "row":"col",
                               cl2txt[type==0?row_triad_canonical_map[ltidx]*3:col_canonical_triad_pos[ltidx]] );
                    }
                    triad_updates++;
                }
                if ( type == 1 ) {
                    if ( rslvd_col_combo_tpos ) {
                        grid_state->set23_found[Col].set_indexbits(rslvd_col_combo_tpos, i*27, 27);
                        grid_state->set23_found[Box].set_indexbits(rslvd_col_combo_tpos, i*27, 27);
                    }
                }
            }
            // finally, for rows only, track updated triad cells and resolved triads
            if ( type == 0 ) {
                for ( unsigned char k=0; k<2; k++ ) {
                    if ( row_combo_tpos[k] ) {
                        grid_state->updated.set_indexbits(row_combo_tpos[k], k*27, 27);
                    }
                    if ( rslvd_row_combo_tpos[k] ) {
                        grid_state->set23_found[Row].set_indexbits(rslvd_row_combo_tpos[k], k*27, 27);
                        grid_state->set23_found[Box].set_indexbits(rslvd_row_combo_tpos[k], k*27, 27);
                    }
                }
            }
        } // for type (cols,rows)

        if ( any_changes ) {
            goto enter;
        }
    }

#ifdef OPT_SETS

// Algorithm 4 - Find naked sets
// For each possible combination of candidates of size K, check for every section
// whether there are exactly K cells that contain only this combination of candidates.
// Back track when the number of found cells exceeds K.
//
// Implementation:
// As an approximation, start with any cell and use its candidates as the starting
// set.  Avoid useless and repeat searches using a variety of heuristics.
//
// Find naked sets, up to MAX_SET

#define MAX_SET 5
// Some general thoughts on the algorithm below.
//
// What does this search achieve:
// for a section with N unlocked cells each naked set of K corresponds to a hidden set of N-K.
// Going up to size 5 allows to get at the very least all hidden sets of 4, and most likely
// (with N<9) all hidden sets of 3.
// Note that not all sets are detected, mainly because:
// - the cells to examine are selected based on prior updates
// - sets can exist whithout any of their cells having all the candidate values.
//   (shortcoming of the algorithm used)
// Note that the debug output reports the naked set/pair, or if the naked set is > 3 and
// its complement has less than 4 member candidates, it is reported instead as a hidden set/pair.
//
// The other important accomplishment is the ability to detect back
// track scenarios if the discovered set is impossibly large.  This is quite important
// for performance as a chance to kill off bad guesses.
//
// Note that this search has it's own built-in heuristic to tackle only recently updated cells.
// The algorithm will keep that list to revisit later, which is fine of course.
// The number of cells to visit can be high (e.g. in the beginning).
// Additional tracking mechanisms are used to reduce the number of searches:
// - previously found sets (and their complements) of size 2 and 3 as well as found triads
// - sets that occupy all available space minus one - impossible due to perfect single detection
//

    if ( mode_sets )
    {
        bool found = false;

        // visit only the changed (updated) cells

        bit128_t to_visit_n;          // tracks all the cells to visit
        bit128_t to_visit_again {};   // track those cells that have been updated

        to_visit_n.u128 = grid_state->updated.u128 & grid_state->unlocked.u128;

        // A cheap way to avoid unnecessary naked set searches
        grid_state->set23_found[Row].u128 |= ~grid_state->unlocked.u128;
        grid_state->set23_found[Row].u64[1] &= 0x1ffff;
        grid_state->set23_found[Col].u128 |= ~grid_state->unlocked.u128;
        grid_state->set23_found[Col].u64[1] &= 0x1ffff;
        grid_state->set23_found[Box].u128 |= ~grid_state->unlocked.u128;
        grid_state->set23_found[Box].u64[1] &= 0x1ffff;

        unsigned char *sectionSetsUnlockedCnt[3] = { 
                            solverData.getSectionSetUnlocked<Row>(*grid_state),
                            solverData.getSectionSetUnlocked<Col>(*grid_state),
                            solverData.getSectionSetUnlocked<Box>(*grid_state) };

      flip ^= 1;

      for ( int a = 0; a<2; a++ ) {
        // this scheme divides to_visit_n into two halves with alternating bits.
        // this increases substantially the detection rate.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
        bit128_t tv = { .u128 = *(unsigned __int128 *)&altbits[a^flip] & to_visit_n.u128 };
#pragma GCC diagnostic pop
        to_visit_n.u128 &= ~tv.u128;

        while (tv.u128) {
            unsigned char i = tzcnt_and_mask(tv);
            unsigned short cnt = __popcnt16(candidates[i]);

            if (cnt <= MAX_SET && cnt > 1) {
                // Note: this algorithm will never detect a naked set of the shape:
                // {a,b},{a,c},{b,c} as all starting points are 2 bits only.
                // The same situation is possible for 4 set members.
                //
                bit128_t to_change {};
                 __m256i a_i = _mm256_set1_epi16(candidates[i]);
                __m128i res;
                unsigned char ul;
                unsigned char s;

                // check row
                //
                if ( !grid_state->set23_found[Row].check_indexbit(i)) {
                    unsigned char ri = row_index[i];
                    ul = sectionSetsUnlockedCnt[Row][ri];
                    if ( grid_state->flags & (1<<ri) && ul > 4 ) {
                        // if a set of 4 has been detected previously, the ul can be updated
                        // this prevents repeated detection of sets of 4, which have previously been cleaned up.
                        // Considering the possible sets of 4 and ul-4:
                        ul = ul<=8? 4:5;
                    }
                    if (check_back || (cnt+2 <= ul) ) {
                        my_naked_sets_searched++;
                        res = _mm_cmpeq_epi16(_mm256_castsi256_si128(a_i), _mm_or_si128(_mm256_castsi256_si128(a_i), *(__m128i_u*) &candidates[9*ri]));
                        unsigned int m = compress_epi16_boolean128(res);
                        bool bit9 = candidates[i] == (candidates[i] | candidates[9*ri+8]);
                        if ( bit9 ) {
                            m |= 1<<8;    // fake the 9th mask position
                        }
                        s = _popcnt32(m);
                        unsigned int m_neg = 0;
                        if (s > cnt) {
                            if ( verbose != VNone ) {
                                char ret[32];
                                format_candidate_set(ret, candidates[i]);
                                if ( grid_state->stackpointer == 0 && unique_check_mode == 0 ) {
                                    if ( warnings != 0 ) {
                                        printf("Line %d: naked  set (row) %s at %s, count exceeded\n", line, ret, cl2txt[ri*9]);
                                    }
                                } else if ( debug ) {
                                    printf("back track sets (row) %s at %s, count exceeded\n", ret, cl2txt[ri*9]);
                                }
                            }
                            // no need to update grid_state
                            goto back;
                        } else if (s == cnt && cnt+2 <= ul) {
                            char ret[32];
                            int delta = ul-cnt;
                            if ( cnt <= 3 ) {
                                unsigned int row_box_intersection = 7<<(box_start[i]%9);
                                grid_state->set23_found[Row].set_indexbits(m,ri*9,9);
                                // adjust the count - maybe required lateron
                                sectionSetsUnlockedCnt[Row][ri] -= cnt;
                                // update box, if set within triad
                                if ( (m&row_box_intersection) == m ) {
                                    grid_state->set23_found[Box].set_indexbits(m&0x1ff,ri*9,9);
                                    int bi_now = sectionSetsUnlockedCnt[Box][box_index[i]];
                                    if ( bi_now >= cnt ) {
                                        sectionSetsUnlockedCnt[Box][box_index[i]] = bi_now - cnt;
                                    }
                                    add_indices<Box>(&to_change, i);
                                }
                            }

                            if ( delta <= 3 ) {
                                // could include locked slots
                                m_neg = 0x1ff & ~(m | grid_state->set23_found[Row].get_indexbits(ri*9,9));
                                int bi_neg = box_index[ri*9+__tzcnt_u32(m_neg)];
                                unsigned int row_box_intersection = 7<<(bi_neg%3*3);
                                grid_state->set23_found[Row].set_indexbits(m_neg,ri*9,9);
                                // adjust the count, maybe required lateron
                                sectionSetsUnlockedCnt[Row][ri] -= _popcnt32(m_neg);
                                // update box, if set within triad
                                if ( (m_neg&row_box_intersection) == m_neg ) {
    
                                    grid_state->set23_found[Box].set_indexbits(m_neg,ri*9,9);
                                    int bicnt_now = sectionSetsUnlockedCnt[Box][bi_neg];
                                    if ( bicnt_now >= cnt ) {
                                        sectionSetsUnlockedCnt[Box][bi_neg] = bicnt_now - cnt;
                                    }
                                }
                            }
                            naked_sets_found++;
                            add_indices<Row>(&to_change, i);
                            if ( cnt==4 || delta==4 ) {
                                grid_state->flags |= (1<<ri);
                            }
                            if ( verbose == VDebug ) {
                                if ( cnt <=3 || cnt <= delta ) {
                                    format_candidate_set(ret, candidates[i]);
                                    printf("naked  %s (row): %-7s %s\n", s==2?"pair":"set ", ret, cl2txt[ri*9+i%9]);
                                } else {
                                    if (delta > 3) {
                                        m_neg = 0x1ff & ~(m | grid_state->set23_found[Row].get_indexbits(ri*9,9));
                                    }
                                    unsigned char k = 0xff;
                                    unsigned short complement = 0;
                                    while (m_neg) {
                                        unsigned char k_i = tzcnt_and_mask(m_neg);
                                        k_i += ri*9;
                                        if ( k == 0xff ) {
                                            k = k_i;
                                        }
                                        complement |= candidates[k_i];
                                    }
                                    complement &= ~candidates[i];
                                    format_candidate_set(ret, complement);
                                    printf("%s %s (row): %-7s %s\n", complement?"hidden":"naked ", __popcnt16(complement)==2?"pair":"set ", ret, cl2txt[ri*9+k%9]);
                                }
                            }
                        }
                    }
                } // row

                // check column and box
                //
                unsigned char ci = column_index[i];
                unsigned char b    = box_start[i];
                unsigned char bi   = box_index[i];

                const bool chk[2] = { !grid_state->set23_found[Col].check_indexbit(i),
                                      !grid_state->set23_found[Box].check_indexbit(i) };
                unsigned char uls[2];
                unsigned char ss[2] = {0,0};

                if ( chk[0] || chk[1] ) {
                    ul = uls[0] = sectionSetsUnlockedCnt[Col][ci];
                    uls[1] = sectionSetsUnlockedCnt[Box][bi];
                    if ( grid_state->flags & ((1<<9)<<ci) && uls[0] > 4) {
                        ul = uls[0] = ul<=8? 4:5;
                    }
                    if ( grid_state->flags & ((1<<18)<<bi) && uls[1] > 4) {
                        uls[1] = uls[1]<=8? 4:5;
                    }
                    if ( uls[1] > ul ) {
                        ul = uls[1];
                    }
                    if (check_back || (cnt+2 <= ul) ) {
                        __m256i a_j_256 = _mm256_set_epi16(candidates[b+19], candidates[b+18], candidates[b+11], candidates[b+10], candidates[b+9], candidates[b+2], candidates[b+1], candidates[b],
                                          candidates[ci+63], candidates[ci+54], candidates[ci+45], candidates[ci+36], candidates[ci+27], candidates[ci+18], candidates[ci+9], candidates[ci]);
                        __m256i res256 = _mm256_cmpeq_epi16(a_i, _mm256_or_si256(a_i, a_j_256));
                        unsigned int ms[2] = { compress_epi16_boolean<true>(res256), 0 };
                        ms[1] = ms[0] >> 16;
                        ms[0] &= 0xffff;
                        bool bit9s[2];
                        bit9s[0] = candidates[i] == (candidates[i] | candidates[ci+72]);
                        bit9s[1] = candidates[i] == (candidates[i] | candidates[b+20]);
                        const char *js[2] = {"col","box"};
                        for ( int j=0; j<2; j++) {  // check for back track first
                            if ( !chk[j] ) {
                                continue;
                            }
                            my_naked_sets_searched++;
                            if ( bit9s[j] ) {
                                ms[j] |= 3<<16;    // fake the 9th mask position
                            }
                            ss[j] = _popcnt32(ms[j])>>1;
                            // this covers the situation where there is a naked set of size x
                            // which is found in y cells with y > x.  That's impossible, hence track back.
                            if (ss[j] > cnt) {
                                if ( verbose != VNone ) {
                                    char ret[32];
                                    format_candidate_set(ret, candidates[i]);
                                    if ( grid_state->stackpointer == 0 && unique_check_mode == 0 ) {
                                        if ( warnings != 0 ) {
                                            printf("Line %d: naked  set (%s) %s at %s, count exceeded\n", line, js[j], ret, cl2txt[i]);
                                        }
                                    } else if ( debug ) {
                                        printf("back track sets (%s) %s at %s, count exceeded\n", js[j], ret, cl2txt[i]);
                                    }
                                }
                                // no need to update grid_state
                                goto back;
                            }
                        }
                        for ( int j=0; j<2; j++) {
                            if ( !chk[j] || (ss[j] != cnt) || (cnt+2 > uls[j])) {
                                continue;
                            }
                            // OK, this is getting a little tricky.
                            // Not only having to deal with columns and boxes, but also
                            // with the detected set and its complement -
                            // and to cast the trace in terms of pairs and sets
                            // while having to manage set23_found bit by bit.
                            //
                            if ( (ss[j] == cnt) && (cnt+2 <= uls[j]) ) {
                                naked_sets_found++;
                                if ( cnt==4 || uls[j]-cnt==4) {
                                    grid_state->flags |= (1<<9)<<(j?bi+9:ci);
                                }
                                if ( j ) {
                                    add_indices<Box>(&to_change, i);
                                } else {
                                    add_indices<Col>(&to_change, i);
                                }
                                unsigned char k = 0xff;
                                unsigned short complement = 0;
                                Kind kind = j?Box:Col;
                                bit128_t s = { *(bit128_t*)&big_index_lut[i][kind][0] & ~grid_state->set23_found[kind] };
                                bool set23_cond1 = (cnt <= 3);
                                bool set23_cond2 = (uls[j] <= cnt+3);
                                for ( unsigned char k_m = 0; k_m<9; k_m++ ) {
                                    unsigned char k_i = (j?(b+box_offset[k_m]):(ci+k_m*9));
                                    if ( !s.check_indexbit(k_i) ) {
                                        continue;
                                    }
                                    bool in_set = (candidates[k_i] | candidates[i]) == candidates[i];
                                        if ( (set23_cond1 && in_set) || (set23_cond2 && !in_set) ) {
                                            grid_state->set23_found[kind].set_indexbit(k_i);
                                        }
                                        if ( !in_set ) {
                                            // set k and compute the complement set
                                            if ( k == 0xff ) {
                                                k = k_i;
                                            }
                                            complement |= candidates[k_i];
                                        }
                                } // for
                                if ( verbose == VDebug ) {
                                    bool naked_anyway = !(complement & candidates[i]);
                                    char ret[32];
                                    complement &= ~candidates[i];
                                    if ( complement != 0 && set23_cond2 ) {
                                        format_candidate_set(ret, complement);
                                        printf("%s %s (%s): %-7s %s\n", naked_anyway?"naked ":"hidden", __popcnt16(complement)==2?"pair":"set ", js[j], ret, cl2txt[k]);
                                    } else {
                                        format_candidate_set(ret, candidates[i]);
                                        printf("naked  %s (%s): %-7s %s\n", cnt==2?"pair":"set ", js[j], ret, cl2txt[i]);
                                    }
                                }
                            }
                        } // for
                    }
                }

                to_change.u128 &= grid_state->unlocked.u128;
                const unsigned char *cip = index_by_i[i];
                // update candidates
                unsigned short cdi = candidates[i];
                unsigned short cdin = ~cdi;
                while (to_change) {
                    unsigned char j = tzcnt_and_mask(to_change);

                    // if this cell is not part of our set
                    if (candidates[j] & cdin ) {
                        // if there are bits that need removing
                        if (candidates[j] & cdi) {
                            candidates[j] &= cdin;
                            to_visit_again.set_indexbit(j);
                            found = true;
                        }
                    } else {
                        const unsigned char *cjp = index_by_i[j];
                        if ( (cjp[Col] == cip[Col]) && (ss[0] == cnt) && (ss[0] <= 3) ) {
                            grid_state->set23_found[Col].set_indexbit(j);
                        }
                        if ( (cjp[Box] == cip[Box]) && (ss[1] == cnt) && (ss[1] <= 3) ) {
                            grid_state->set23_found[Box].set_indexbit(j);
                        }
                    }
                }

                // If any cell's candidates got updated, go back and try all that other stuff again
                if (found) {
                    grid_state->updated.u128 = to_visit_n.u128 | tv.u128 | to_visit_again.u128;
                    goto enter;
                }
            } else if ( cnt == 1 ) {
                // this is not possible, but just to eliminate cnt == 1:
                if ( verbose == VDebug ) {
                    printf("naked  (sets) ");
                }
                e_i = i;
                e_digit = candidates[i];
                if ( warnings != 0 ) {
                    printf("found a singleton in set search... strange\n");
                }
                goto enter;
            }
        } // while
      }

      grid_state->updated.u128 = to_visit_again.u128;
    }
#endif

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
        if ( N < 23 ) {
            bit128_t &bivalues = solverData.getBivalues(candidates);
            unsigned char target = 0;   // the index of the only cell with three candidates
            int sum2 = _popcnt64(bivalues.u64[0]) + _popcnt32(bivalues.u64[1]);
            if ( sum2 == N ) {
                grid_state->flags |= 1U<<31;	// set sign bit
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
                } else {   // busted.  This is not a valid puzzle under standard rules.
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
                target += __tzcnt_u64(m);
            } else {
                goto no_bug;
            }
            if ( __popcnt16(candidates[target]) == 3 ) {
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
                    bug_count++;
                    if ( rules == Regular ) {
                        if ( verbose == VDebug ) {
                            printf("bi-value universal grave pivot:");
                        }
                        e_i = target;
                        e_digit = digit;
                        goto enter;
                    } else {
                        if ( verbose == VDebug ) {
                            printf("bi-value universal grave pivot:\n");
                        }
                        grid_state = grid_state->make_guess<verbose>(target, digit);
                        goto start;
                    }
                }
            }
        }
        no_bug:
        ;
    }

#ifdef OPT_FSH
    if ( mode_fish )
    {
    // number grids (aka Fish (X-Wing (N==2), sword fish (N==3), jelly fish (N==4), squirmbag (N==5) )
    //
    // Theory (rows/cols only):
    // Looking at grids of N rows x N columns,  it is easy to see that under Sudoku rules
    // you can always place exactly N times the same digit on the grid.
    // This is also valid for grids of size one (single location) and size nine (full board).
    //
    // It is always true that for N==9 there must by nine locations where to place a given
    // digit.  As the game progresses and digits are placed, the availabe candidates are reduced
    // for each resolved candidate.
    // Similarly, it is easy to see that for each pick of N times the same digit in a valid
    // solution, these will form a grid N x N rows and columns.
    //
    // Given a sudoku board populated with candidates, the task is then to find such grids
    // such that they correspond to the solution of the Sudoku problem.
    //
    // The conditons for such a grid based on the candidate locations are very simply:
    // find N rows or columns for which all candidates are within N columns / rows.
    // (there is no guarantee to find any, but if found, it is a valid grid).
    // Having identified such a valid grid in terms of rows, any extra candidates in the columns
    // can be removed and vice versa.
    //
    // The search operates just as the naked set search does, i.e. it will not discover
    // fishes where all lines have less than N candidates.  This can be remedied by augmenting
    // the set of lines using additional lines (by 1 additional line, which should go a long
    // way (a jelly fish can have 4 lines of 2 candidates each, but the value of such a search
    // needs to be confirmed).
    // Once a grid has been found and cleaned, the complementary grid is found in the remaining
    // lines and rows.
    //
    // Variants:
    // Finned Fish:
    // A fish with extra candidates in one line.
    // The way this is identified in the code is that N-1 'pure' fish lines are found.
    // Compared to the regular fish search above,
    // this requires identifying the extra line with 1 or at most 2 extra candidates that can
    // both 'see' the same candidates on one of the grid lines perpendicular to the base (i.e. in the same box).
    // Either one of the extra candidates is valid, or the fish pattern is valid.
    // Eliminate just those candidates visible to the extra candidates and on the fish grid line
    // but not a grid point.
    //
    // Sashimi Fish:
    // Same as the finned fish variant above, except the that fish pattern is missing the
    // candidate in the box with the fin(s).
    // Two sashimi fishes can interact between themselves.  Those two lines must (again) be in the 
    // same band and they then eliminate those candidates visible to both that are not
    // on the fish grid.
    //
    // Note:
    // - the 'yield' for finned and sashimi fishes is typically low since no definite
    //   grid can be established.
    //
    // with the cbbvs in place, look for simple 'fishes'

    const char *fish_names[4] = { "X-wing-2", "swordfish-3", "jellyfish-4", "squirmbag-5" };

    cbbv_t cbbv_v;

    bit128_t *candidate_bits_by_value = solverData.getCbbvs(candidates);

    for ( unsigned char dgt = 0; dgt < 9; dgt++) {
        // don't bother with six cells already resolved, a swordfish cannot be subdivided.
        // a jellyfish is final with 8 candidates (4 x 2), so 8 + 5 = 13 cannot be divided
        // unless it is already.
        // The minimum number of candidates is 14 to work with. 
        if ( candidate_bits_by_value[dgt].popcount() < 14 ) {
            continue;
        }
        // load a vector
        bit128_t cbbv_digit = { .u128= candidate_bits_by_value[dgt].u128 & grid_state->unlocked.u128 };
        unsigned long long lo = cbbv_digit.u64[0];
        unsigned long long hi = cbbv_digit.u64[1];

        // load cbbv_v
        cbbv_v.m256 = _mm256_and_si256(_mm256_setr_epi16(lo, lo>>9, lo>>18, lo>>27, lo>>36, lo>>45, lo>>54, (lo>>63)|(hi<<1), hi>>8, 0, 0, 0, 0, 0, 0, 0),mask1ff);

        // do a quick popcnt of all rows or'ed together to determine the maximum fish size
        __m128i max_v = _mm256_castsi256_si128(_mm256_or_si256(cbbv_v.m256, _mm256_bsrli_epi128(cbbv_v.m256,8)));
        max_v = _mm_or_si128(max_v, _mm_bsrli_si128(max_v,4));
        max_v = _mm_or_si128(max_v, _mm_bsrli_si128(max_v,2));
        unsigned char max = __popcnt16(_mm_extract_epi16(max_v, 0) | hi>>8);
        if ( max > 7 ) {
            max = 7;
        }
        bit128_t dgt_bits = { .u128 = candidate_bits_by_value[dgt].u128 & grid_state->unlocked.u128 };

        unsigned int pair_locs = 0;  // pairs, bits left to right
        unsigned int alt_base_x = 0;
        for (unsigned char t=0; t<9; t++, pair_locs <<= 1) { // t is the row the pattern is sampled from
            unsigned int base_x = alt_base_x? alt_base_x : cbbv_v.v16[t]; // base_x: tentative fish pattern
            // variables names reflect whether they are parallel (_prl) or perpendicular (_prp) to the base_x
            // also a suffix of _x signifies that the variable contains index bits
            if ( base_x & exclude_row[dgt] ) {
                continue;
            }
            // for the given digit, the count in a given row is N (cnt).
            // take each row of bits and compare to all other rows.
            //
            unsigned char cnt = __popcnt16(base_x);

            if ( cnt+2 <= max && cnt > 1) {  // maximum cnt is 5 ('squirmbag').
                if ( cnt == 2 ) {
                    pair_locs |= 1;
                }
                __m256i basev = _mm256_and_si256(_mm256_set1_epi16(base_x),mask9);
                // rows that have a non-empty intersection with base_x
                __m256i hassub_v = _mm256_cmpgt_epi16(_mm256_and_si256(basev, cbbv_v.m256),_mm256_setzero_si256());
                // rows that are a non-empty subset of base_x
                __m256i issub_v = _mm256_and_si256(hassub_v, _mm256_cmpeq_epi16(basev, _mm256_or_si256(basev, cbbv_v.m256)));

                unsigned int subs_prp_x = compress_epi16_boolean(issub_v); // bits for non-empty subsets of base_x
                unsigned int hassubs_prp_x;  // bits for non-empty intersection with base_x
                unsigned char nsubs = _popcnt32(subs_prp_x);
                // test for naked fish and exclude it for row and col search
                if ( nsubs == cnt && _mm256_testc_si256(issub_v, hassub_v) ) {
                    // found a (naked) fish, leave it
                    if ( _popcnt32(base_x) <= 3 ) {
                        exclude_row[dgt] |= base_x;
                        exclude_col[dgt] |= subs_prp_x; // the transposed col base_x
                        fishes_excluded++;   // counted but not currently included in summary
                    }
                    continue;
                }

                bit128_t clean_bits {};

                // if there are exactly N rows with the same pattern of digits, then
                // it is a fish pattern.
                // classify found fishes if any.
                if ( nsubs == cnt ) {
                    // this part is efficient to find and easy to deal with - however the yield very low.
                    fishes_detected++;
                    hassubs_prp_x = compress_epi16_boolean(hassub_v);
                    unsigned int rows2clean = hassubs_prp_x & ~subs_prp_x;
                    while (rows2clean) {
                        unsigned char row = tzcnt_and_mask(rows2clean);
                        clean_bits.set_indexbits( cbbv_v.v16[row]&base_x, row*9, 9);
                    }
                    clean_bits.u128 &= grid_state->unlocked.u128 & candidate_bits_by_value[dgt].u128;
                    if ( clean_bits ) {
                        if ( verbose == VDebug ) {
                            unsigned char celli  = __tzcnt_u16(subs_prp_x)*9 + __tzcnt_u16(base_x);
                            unsigned char celli2 = (15-__lzcnt16(subs_prp_x))*9 + (15-__lzcnt16(base_x));
                            if ( debug > 2 ) {
                                show_fish(cbbv_v, base_x, subs_prp_x, hassubs_prp_x, clean_bits, dgt_bits, "row fish");
                            }
                            printf("%s (rows) digit %d at cells %s - %s\nRemove %d at ", fish_names[cnt-2], dgt+1, cl2txt[celli], cl2txt[celli2], dgt+1);
                        }
                    }
                } else if ( nsubs == cnt-1 ) {
                    // nsubs == cnt-1 is required for finned/sashimi fishes.
                    // for a fin, the following must be true:
                    // The fin(s) cells must be on the perpendicular fish sections w.r.t. to the base_x.
                    // Only one fin cell is allowed per section parallel to the base_x except
                    // for case 2 below, where the two fin cells can be in the same box.
                    // Note: The fins are placed differently, when looking at the final diagram.
                    //  The communality is that these lines are selected initially using the same
                    //  criteria: having an overlap with the base line.
                    //  However, in case 1 the fins are on the grid established by the base,
                    //  [the fins are on the grid and therefore the case is not called 'finned'
                    //   in the log]
                    //  whereas in case 2 the fins are lateral to that grid but affect grid points.
                    //  There may be alternative definitions that consolidate these view points,
                    //  but for the purpose of the code the distinction is necessary and important.
                    // Case 1:
                    // A. This case requires exactly two fin sections.
                    //    These fin sections and cells are mutually exclusive and
                    //    one would be required to complete the fish pattern and eliminate the other.
                    // B. The respective fin cells must be unique on the parallel fish grid line
                    //    when intersected with the base and unique on their perpendicular fish grid
                    //    outside of the fish pattern.
                    // C. They must share a band parallel to the base_x and not share a band with
                    //    any established grid line parallel to the base.
                    // In this case the two fins' views intersect with each other to define
                    // two triads to be cleaned.
                    // Note: This arrangement can also be seen in most cases as an AIC or
                    // (in the case of an X-Wing) as a Skyscraper pattern.
                    // Note: This pattern extends to swordfish and rarely to jellyfish.
                    // The fins can occasionally share the same box.
                    // When logged, this case is identified as 'finned/sashimi-pair'
                    // Addendum to Case 1:
                    // A direct consequence for the band with the two mutual exclusive fins
                    // is the following:
                    // If for two cells in different boxes of the same band their value is
                    // mutually exclusively the same digit D, the triad T that does not share
                    // a box or row/col with either of the two cells, if:
                    // the value candidate is unique in the respective triads of these two cells,
                    // then the triad T cannot contain the digit D.
                    // This situation occurs frequently in conjunction with Case 1.
                    // As proof and for illustration:  The band triads (without loss of generality, a row band):
                    //       A1   A2   A3
                    //       B1   B2   B3
                    //       C1   C2   C3
                    // Without loss of generality assume that triads A1 and B2 contain cells that
                    // have a candidate value D.  We do know, that due to being linked exclusively,
                    // that neither A2 nor B1 can contain candidate value D (and the code makes sure of that).
                    // In order to not be a singleton in A1, there must be either another candidate
                    // value in A1 or A3.  The stated uniqueness requirement for A1 therefore
                    // forces this candidate to be in A3 and mutually exclusive with the candidate in A1.
                    // Using the same argument, there has to a candidate in B3 that is mutually exclusive
                    // with B2. Since the candidates in A1 and B2 are exclusive, so must be the 
                    // candidates in A3 and B3.  As a consequence, since one or the other must
                    // true, C3 cannot contain value D.
                    // In the scenario of Case 1 we must simply look for the following conditions:
                    // a. the fins must not share a box,
                    // b. the fins must be the only candidate D in their respective triad.
                    // [ Note that this addendum is not part of the sashimi pattern directly as
                    //   spelled out by numerous sources on the Web ]. 
                    // Case 2:
                    // A. the fin cells must all be in the same box (and aligned if there are 2), and
                    // B. the same box must contain a fish grid point (which need not have a candidate)
                    // In this case the fin(s) views intersect with their associated fish grid point's
                    // to provide the triad (minus the fish grid) to be cleaned.

                    hassubs_prp_x = compress_epi16_boolean(hassub_v);
                    unsigned int finsubs_prp_x = hassubs_prp_x & ~subs_prp_x; //& ~exclude_row;
                    bool sashimi = false;
                    unsigned int subsx= subs_prp_x;
                    unsigned int finsubcnt = _popcnt32(finsubs_prp_x);  // the complete count
                    if ( finsubcnt == 2 ) {   // Case 1 A
                         unsigned short fin_subs_prp[2] = { (unsigned short)_tzcnt_u32(finsubs_prp_x), (unsigned short)(31-__lzcnt32(finsubs_prp_x)) };
                         if (   ( fin_subs_prp[0]/3 == fin_subs_prp[1]/3 )     // Case 1 C
                              && !(bandbits_by_index[fin_subs_prp[0]/3] & subs_prp_x)) {
                             unsigned short fin_subs_pos_prl[2] = { __tzcnt_u16(cbbv_v.v16[fin_subs_prp[0]] & base_x),
                                                                __tzcnt_u16(cbbv_v.v16[fin_subs_prp[1]] & base_x) };

                             if (    fin_subs_pos_prl[0] != fin_subs_pos_prl[1]                // Case 1 B
                                  && _popcnt32(cbbv_v.v16[fin_subs_prp[0]] & base_x) == 1
                                  && _popcnt32(cbbv_v.v16[fin_subs_prp[1]] & base_x) == 1 ) {
                                  fishes_detected++;
                                  fishes_specials_detected++;
                                  unsigned char fincells[2] =  { (unsigned char)(fin_subs_prp[0]*9 + fin_subs_pos_prl[0]),
                                                                 (unsigned char)(fin_subs_prp[1]*9 + fin_subs_pos_prl[1]) };
                                  clean_bits.u128 =    (*(bit128_t*)big_index_lut[fincells[0]][All]).u128
                                                     & (*(bit128_t*)big_index_lut[fincells[1]][All]).u128;
                                 // deal with 'addendum to Case 1'
                                 if ( fincells[0]/3%3 != fincells[1]/3%3 ) { // not in same box
                                     if (    _popcnt32(candidate_bits_by_value[dgt].get_indexbits(fincells[0]-fincells[0]%3, 3)) == 1
                                          && _popcnt32(candidate_bits_by_value[dgt].get_indexbits(fincells[1]-fincells[1]%3, 3)) == 1) {
                                          unsigned char boxoff = 3 * _tzcnt_u32(7 ^ ((1<<fincells[0]/3%3) | (1<<fincells[1]/3%3)));
                                          unsigned char rowoff = 9 * ( _tzcnt_u32(7 ^ ((1<<fincells[0]/9%3) | (1<<fincells[1]/9%3)))
                                                                       + fin_subs_prp[0]/3*3 );
                                           clean_bits.set_indexbits(7, rowoff+boxoff, 3);
                                     }
                                 }
                                   clean_bits.u128 &= candidate_bits_by_value[dgt].u128
                                                      & grid_state->unlocked.u128;
                                 clean_bits.unset_indexbit(fincells[0]);
                                 clean_bits.unset_indexbit(fincells[1]);
                                 if ( clean_bits ) {
                                     fishes_specials_updated++;
                                     if ( verbose == VDebug ) {
                                         subsx = subs_prp_x | (1<<fin_subs_prp[0]) | (1<<fin_subs_prp[1]);
                                         unsigned char celli = __tzcnt_u16(subsx)*9 + __tzcnt_u16(base_x);
                                         unsigned char celli2 = (15-__lzcnt16(subsx))*9 + (15-__lzcnt16(base_x));
                                         if ( debug > 2 ) {
                                             show_fish(cbbv_v, subs_prp_x, 2, fincells, clean_bits, dgt_bits, "finned row fish");
                                         }
                                         printf("sashimi-pair %s (rows) digit %d at cells %s - %s\nFins at %s,%s - remove %d at ", fish_names[cnt-2], dgt+1, cl2txt[celli], cl2txt[celli2], cl2txt[fincells[0]], cl2txt[fincells[1]], dgt+1);
                                     }
                                 }
                             }
                         }
                    }

                    if ( clean_bits == 0 ) {
                        // Case 2
                        // iterate over possible fin rows
                        while ( finsubs_prp_x ) {
                            // pick one possible fin row
                            unsigned char fin_sub = tzcnt_and_mask(finsubs_prp_x);
                            unsigned short finln_prl_x = cbbv_v.v16[fin_sub];
                            unsigned short fins = finln_prl_x & ~base_x;

                            unsigned short box_bits = bandbits_by_index[__tzcnt_u16(fins)/3];  // the box to compare with
                            // grid point in box? fins in box?    
                            // In this first case the fin cell view is intersected with
                            // a grid point view box.
                            if (  (fins & ~box_bits) || !(box_bits & base_x) ) {   // Case 2, A and B
                                continue;
                            }
                            // if the fin has no candidate on its associated grid point, it's a sashimi
                            // for information only, as it's the common nomenclature.
                            sashimi = !(finln_prl_x & base_x & box_bits);

                            // complete the finned fish pattern:
                            subsx = subs_prp_x | (1<<fin_sub);

                            // the rows left for cleaning
                            unsigned short band_bits = bandbits_by_index[fin_sub/3];
                            unsigned int rows2clean = hassubs_prp_x & ~subsx & band_bits;
                            if ( rows2clean == 0 ) {
                                // the sashimi X-wing base_x is a good guess to keep for later:
                                if ( solverData.guess_hint_digit == 0 && !clean_bits && sashimi && alt_base_x==0 && cnt == 2 ) {
                                    solverData.guess_hint_digit = 1<<dgt;
                                    solverData.guess_hint_index = t*9 + __tzcnt_u16(base_x & ~box_bits);
                                }
                                continue;
                            }
                            fishes_specials_detected++;
                            fishes_detected++;

                            unsigned int boxbase_x = base_x & box_bits;
                            clean_bits.u128 = 0;

                            while (rows2clean) {
                                unsigned char row = tzcnt_and_mask(rows2clean);
                                clean_bits.set_indexbits( cbbv_v.v16[row]&boxbase_x, row*9, 9);
                            }
                            clean_bits.u128 &= dgt_bits.u128;

                            if ( clean_bits ) {
                                fishes_specials_updated++;
                                if ( verbose == VDebug ) {
                                    unsigned char celli = __tzcnt_u16(subsx)*9 + __tzcnt_u16(base_x);
                                    unsigned char celli2 = (15-__lzcnt16(subsx))*9 + (15-__lzcnt16(base_x));
                                    if ( debug > 2 ) {
                                        show_fish(cbbv_v, base_x, subsx, hassubs_prp_x, clean_bits, dgt_bits, "finned row fish");
                                    }
                                    printf("finned%s %s (rows) digit %d at cells %s - %s\nFin at %s - remove %d at ", sashimi?"/sashimi":"", fish_names[cnt-2], dgt+1, cl2txt[celli], cl2txt[celli2], cl2txt[fin_sub*9+__tzcnt_u16(fins&~base_x)], dgt+1);
                                }
                                break;
                            }
                        } // while
                    } // if
                } // else
                if ( clean_bits ) {
                    fishes_updated++;
                    unsigned short dgt_mask_bit = 1<<dgt;
                    while (clean_bits) {
                        unsigned char cl = tzcnt_and_mask(clean_bits);
                        if ( candidates[cl] & dgt_mask_bit ) {
                            candidates[cl] &= ~dgt_mask_bit;
                            if ( verbose == VDebug ) {
                                printf("%s ", cl2txt[cl]);
                            }
                        }
                    }
                    if ( verbose == VDebug ) {
                        printf("\n");
                    }
                    goto enter;
                }
            } // if
            // scan for two bi-values forming a triple...
            // just take a single guess with this - it will also catch fin/sashimi
            unsigned int pair_cnt = __popcnt16(pair_locs);
            if ( t == 8 && pair_cnt >= 3 ) {
                unsigned char pos[9];
                for ( int i=pair_cnt-1; pair_locs; i-- ) {
                    pos[i] = 8-tzcnt_and_mask(pair_locs);
                }
                unsigned short res = 0;
                for ( unsigned int i=0; i<pair_cnt-1; i++) {
                    for ( unsigned int k=i+1; k<pair_cnt; k++ ) {
                        if ( ( __popcnt16(res = cbbv_v.v16[pos[i]] | cbbv_v.v16[pos[k]])) == 3 ) {
                            alt_base_x = res;
                            goto done;
                        }
                    }
                }
            }
            continue;
done:
            t = 7;  // one iteration with the made-up data
            pair_locs = 0; // will skip this section next time...
        } // for t

        // part 2: look for fishes at columns

        // transpose cbbv_v
        cbbv_t cbbv_col_v {};

        unsigned short *mskp = &cbbv_col_v.v16[8];
        __m256i c = _mm256_srli_epi16(cbbv_v.m256,1);
        *mskp-- = _mm_movemask_epi8(_mm_packus_epi16(_mm256_castsi256_si128(c),_mm256_extractf128_si256(c,1)));
        c = _mm256_and_si256(cbbv_v.m256, maskff);
        __m128i cc = _mm_packus_epi16(_mm256_castsi256_si128(c),_mm256_extractf128_si256(c,1));
        for (unsigned char d = 8; d > 0; d--) {
            *mskp-- = _mm_movemask_epi8(cc);
            cc = _mm_slli_epi16(cc,1);
        }

        pair_locs = 0;  // pairs, bits left to right
        alt_base_x = 0;
        for (unsigned char t=0; t<9; t++, pair_locs <<= 1) { // t is the row the pattern is sampled from
            unsigned int base_x = alt_base_x? alt_base_x : cbbv_col_v.v16[t]; // base_x: tentative fish pattern

            if ( base_x & exclude_col[dgt] ) { // base_x: tentative fish pattern
                continue;
            }
            // for the given digit, the count in a given col is N (cnt).
            // take each col of bits and compare to all other cols.
            //
            unsigned char cnt = __popcnt16(base_x);
            if ( cnt == 2 ) {
                pair_locs |= 1;
            }

            if ( cnt+2 <= max && cnt > 1) {  // maximum cnt is 5 ('squirmbag').
                __m256i basev = _mm256_and_si256(_mm256_set1_epi16(base_x),mask9);
                // rows that have a non-empty intersection with base_x
                __m256i hassub_v = _mm256_cmpgt_epi16(_mm256_and_si256(basev, cbbv_col_v.m256),_mm256_setzero_si256());
                // rows that are a non-empty subset of base_x
                __m256i issub_v = _mm256_and_si256(hassub_v, _mm256_cmpeq_epi16(basev, _mm256_or_si256(basev, cbbv_col_v.m256)));

                unsigned int subs_prp_x = 0x1ff & compress_epi16_boolean(issub_v);
                unsigned char nsubs = _popcnt32(subs_prp_x);
                unsigned int hassubs_prp_x;

                if ( nsubs == cnt && _mm256_testc_si256(issub_v, hassub_v) ) {
                    // found a (naked) fish
                    // for naked fishes, don't double count row/col detection
                    // neither need to exclude anything;
                    // fishes_detected++;
                    continue;
                }
                bit128_t clean_bits {};
                bit128_t tmp {};

                // if there are exactly N rows with the same pattern of digits, then
                // it is a fish pattern.
                if ( nsubs == cnt ) {
                    // this part is efficient to find and easy to deal with - however the yield very low.
                    fishes_detected++;
                    hassubs_prp_x = compress_epi16_boolean(hassub_v);
                    unsigned int cols2clean = hassubs_prp_x & ~subs_prp_x;
                    while (cols2clean) {
                        unsigned char col = tzcnt_and_mask(cols2clean);
                        tmp.set_indexbits( cbbv_col_v.v16[col]&base_x, col*9, 9);
                    }
                    // since we work in a transposed view, we transpose clean_bits here:
                    while ( tmp ) {
                        clean_bits.set_indexbit(transposed_cell[tzcnt_and_mask(tmp)]);
                    }
                    clean_bits.u128 &= dgt_bits.u128;
                    if ( clean_bits.u128 ) {
                        if ( verbose == VDebug ) {
                            if ( debug > 2 ) {
                                show_fish<true>(cbbv_col_v, base_x, subs_prp_x, hassubs_prp_x, clean_bits, dgt_bits, "col fish");
                            }
                            unsigned char celli = __tzcnt_u16(subs_prp_x) + __tzcnt_u16(base_x)*9;
                            unsigned char celli2 = (15-__lzcnt16(subs_prp_x)) + (15-__lzcnt16(base_x))*9;
                            printf("%s (cols) digit %d cells %s - %s\nRemove %d at ", fish_names[cnt-2], dgt+1, cl2txt[celli], cl2txt[celli2], dgt+1);
                        }
                    }
                } else if ( nsubs == cnt-1 ) {
                    // nsubs == cnt-1 is required for both fin and sashimi fishes.

                    // for a fin, the following must be true:
                    // 1. the fin cells not on the fish grid must all be in the same box
                    // 2. the same box must contain a fish grid point (which need not have a candidate)

                    // iterate over possible fin cols
                    hassubs_prp_x = compress_epi16_boolean(hassub_v);
                    unsigned int finsubs_prp_x = hassubs_prp_x & ~subs_prp_x ; //& ~exclude_col;
                    bool sashimi = false;
                    unsigned int subsx= subs_prp_x;
                    unsigned int finsubcnt = _popcnt32(finsubs_prp_x);  // the complete count

                    if ( finsubcnt == 2 ) {   // Case 1 A
                         unsigned short fin_subs_prp[2] = { (unsigned short)_tzcnt_u32(finsubs_prp_x), (unsigned short)(31-__lzcnt32(finsubs_prp_x)) };
                         if (   ( fin_subs_prp[0]/3 == fin_subs_prp[1]/3 )     // Case 1 C
                              && !(bandbits_by_index[fin_subs_prp[0]/3] & subs_prp_x)) {
                             unsigned short fin_subs_pos_prl[2] = { __tzcnt_u16(cbbv_col_v.v16[fin_subs_prp[0]] & base_x),
                                                                __tzcnt_u16(cbbv_col_v.v16[fin_subs_prp[1]] & base_x) };
                             if (    fin_subs_pos_prl[0] != fin_subs_pos_prl[1]                // Case 1 B
                                  && _popcnt32(cbbv_col_v.v16[fin_subs_prp[0]] & base_x) == 1
                                  && _popcnt32(cbbv_col_v.v16[fin_subs_prp[1]] & base_x) == 1 ) {
                                  fishes_detected++;
                                  fishes_specials_detected++;
                                  unsigned char fincells[2] =  { (unsigned char)(fin_subs_prp[0] + fin_subs_pos_prl[0]*9),
                                                                 (unsigned char)(fin_subs_prp[1] + fin_subs_pos_prl[1]*9) };
                                  clean_bits.u128 =    (*(bit128_t*)big_index_lut[fincells[0]][All]).u128
                                                     & (*(bit128_t*)big_index_lut[fincells[1]][All]).u128;
                                 // deal with 'addendum to Case 1'
                                 if ( fincells[0]/27 != fincells[1]/27 ) { // not in same (vertical) box
                                     if (    _popcnt32(candidate_bits_by_value[dgt].get_indexbits(fincells[0]/27*27 + fincells[0]%9, 19) & 0x40201) == 1
                                          && _popcnt32(candidate_bits_by_value[dgt].get_indexbits(fincells[1]/27*27 + fincells[1]%9, 19) & 0x40201) == 1) {
                                          unsigned char boxoff = fincells[0]/3%3*3 + _tzcnt_u32(7 ^ ((1<<fincells[0]%3) | (1<<fincells[1]%3)));
                                          unsigned char coloff = 27*_tzcnt_u32(7^((1<<fincells[0]/27) | (1<<fincells[1]/27)));
                                          clean_bits.set_indexbits(0x40201, coloff+boxoff, 19);
                                     }
                                 }
                                 clean_bits.u128 &= candidate_bits_by_value[dgt].u128
                                                 & grid_state->unlocked.u128;
                                 clean_bits.unset_indexbit(fincells[0]);
                                 clean_bits.unset_indexbit(fincells[1]);
                                 if ( clean_bits ) {
                                     fishes_specials_updated++;
                                     if ( verbose == VDebug ) {
                                         subsx = subs_prp_x | (1<<fin_subs_prp[0]) | (1<<fin_subs_prp[1]);
                                         unsigned char celli = __tzcnt_u16(subsx) + __tzcnt_u16(base_x)*9;
                                         unsigned char celli2 = (15-__lzcnt16(subsx)) + (15-__lzcnt16(base_x))*9;
                                         if ( debug > 2 ) {
                                             show_fish<true>(cbbv_col_v, subs_prp_x, 2, fincells, clean_bits, dgt_bits, "finned col fish");
                                         }
                                         printf("sashimi-pair %s (cols) digit %d at cells %s - %s\nFins at %s,%s - remove %d at ", fish_names[cnt-2], dgt+1, cl2txt[celli], cl2txt[celli2], cl2txt[fincells[0]], cl2txt[fincells[1]], dgt+1);
                                     }
                                 }
                             }
                         }
                    }

                    if ( clean_bits == 0 ) {
                        // Case 2
                        // iterate over possible fin rows
                        while ( finsubs_prp_x ) {
                            // pick one possible fin col
                            unsigned char fin_sub = tzcnt_and_mask(finsubs_prp_x);
                            unsigned short finln_prl_x = cbbv_col_v.v16[fin_sub];

                            unsigned short fins = finln_prl_x & ~base_x;
                            unsigned short box_bits = bandbits_by_index[__tzcnt_u16(fins)/3];  // the box to compare with
                            // fins in box? grid point in box?
                            if (  (fins & ~box_bits) || !(box_bits & base_x) ) {
                                continue;
                            }

                            sashimi = !(finln_prl_x & base_x & box_bits);
                            // complete the finned fish pattern:
                            unsigned int subsx = subs_prp_x | (1<<fin_sub);

                            // the cols left for cleaning
                            unsigned short band_bits = bandbits_by_index[fin_sub/3];
                            unsigned int cols2clean = hassubs_prp_x & ~subsx & band_bits;
                            if ( cols2clean == 0 ) {
                                // the sashimi X-wing base is a good guess:
                                if ( solverData.guess_hint_digit == 0 && !clean_bits && sashimi && alt_base_x==0 && cnt == 2 ) {
                                    solverData.guess_hint_digit = 1<<dgt;
                                    solverData.guess_hint_index = t + __tzcnt_u16(base_x & ~box_bits)*9;
                                }
                                continue;
                            }
                            fishes_specials_detected++;
                            fishes_detected++;

                            unsigned int boxbase_x = base_x & box_bits;

                            clean_bits.u128 = 0;
                            bit128_t tmp {};
                            while (cols2clean) {
                                unsigned char col = tzcnt_and_mask(cols2clean);
                                tmp.set_indexbits( cbbv_col_v.v16[col]&boxbase_x, col*9, 9);
                            }
                            // since we work in a transposed view, we transpose clean_bits first
                            while ( tmp ) {
                                clean_bits.set_indexbit(transposed_cell[tzcnt_and_mask(tmp)]);
                            }
                            clean_bits.u128 &= dgt_bits.u128;

                            if ( clean_bits ) {
                                fishes_specials_updated++;
                                if ( verbose == VDebug ) {
                                    if ( debug > 2 ) {
                                        show_fish<true>(cbbv_col_v, base_x, subsx, hassubs_prp_x, clean_bits, dgt_bits, "finned col fish");
                                    }
                                    unsigned char celli = __tzcnt_u16(subsx) + __tzcnt_u16(base_x)*9;
                                    unsigned char celli2 = (15-__lzcnt16(subsx)) + (15-__lzcnt16(base_x))*9;
                                    printf("finned%s %s (cols) digit %d at cells %s - %s\nFin at %s - remove %d at ", sashimi?"/sashimi":"", fish_names[cnt-2], dgt+1, cl2txt[celli], cl2txt[celli2], cl2txt[fin_sub+__tzcnt_u16(fins&~base_x)*9], dgt+1);
                                }
                                break;
                            }
                        } // while
                    }  // if
                }
                if ( clean_bits.u128 ) {
                    fishes_updated++;
                    unsigned short dgt_mask_bit = 1<<dgt;
                    while (clean_bits) {
                        unsigned char cl = tzcnt_and_mask(clean_bits);
                        if ( candidates[cl] & dgt_mask_bit) {
                            candidates[cl] &= ~dgt_mask_bit;
                            if ( verbose == VDebug ) {
                                printf("%s ", cl2txt[cl]);
                            }
                        }
                    }
                    if ( verbose == VDebug ) {
                        printf("\n");
                    }
                    goto enter;
                }
            }
            // scan for two bi-values forming a triple...
            // just take a single guess with this - it will also catch fin/sashimi
            unsigned int pair_cnt = __popcnt16(pair_locs);
            if ( t == 8 && pair_cnt >= 3 ) {
                unsigned char pos[9];
                for ( int i=pair_cnt-1; pair_locs; i-- ) {
                    pos[i] = 8-tzcnt_and_mask(pair_locs);
                }
                unsigned short res = 0;
                for ( unsigned int i=0; i<pair_cnt-1; i++) {
                    for ( unsigned int k=i+1; k<pair_cnt; k++ ) {
                        if ( ( __popcnt16(res = cbbv_col_v.v16[pos[i]] | cbbv_col_v.v16[pos[k]])) == 3 ) {
                            alt_base_x = res;
                            goto done2;
                        }
                    }
                }
            }
            continue;
done2:
            t = 7;  // one iteration with the made-up data
            pair_locs = 0; // will skip this section next time...
        } // for t
    } // for dgt
    } // mode_fish
#endif

#if OPT_UQR
    //
    // Unique (Avoidable) Rectangles
    //
    // The theory:
    // For each (a) retangular four cells, (b) lying in two boxes (i.e. two cells in each box)
    // and (c) of which none is set with a preset value:
    //   Resolving these cells such that both pairs of diagonally opposite corners have
    //   the same value _always_ and _automatically_ will lead to at least _2_ different
    //   solutions.
    // Proof: In any valid solution interchanging these two pairs
    //   will lead to a different yet equally valid solution.
    //
    // The main implication is that :: in a Sudoku puzzle that has exactly one solution
    // (e.g. as part of the rules for that puzzle, as commonly is the case) ::, it is
    // possible during the solving process to detect such patterns and remove candidates
    // that would otherwise lead to multiple possible solutions and hence cannot be
    // considered for a solution.
    //
    // In terms of wording, such a puzzle solution is called 'unique' or 'the solution'.
    //
    // General notes:
    //
    // 1. There is nothing particularly special about these rectangles.  The same
    //    poperty of non-uniqueness applies to many several other patterns that are
    //    'closed' in themselves.  The unique avoidable rectangle is just the simplest
    //    of those patterns.  To witness, aligned three cells in a box, with a coresponding
    //    matching box, a pattern {a,b},{b,c},{a,c} will do the same, as will a
    //    transitive combination of three aligned pairs {a,b},{b,c},{a,c}.
    //    The pattern {a,b}, {a,b}, {a,b} can bend by 90 degrees for a corner.
    //    The pattern called 'binary universal grave', which is an end game pattern also
    //    falls into the same category, as it implies multiple solutions.
    //
    // 2. It is perfectly possible to completely disregard these patterns of multiple
    //    solutions.  Just be prepared to take a guess to find a solution.
    //    Remember always that for a 'regular' Sudoku puzzle there will be a solution that
    //    does not show such a pattern.  Hence the notion of 'avoidable' patterns.
    //
    // Special notes:
    //
    // 3. The definition above only describes what will eventually be true of the final
    //    solution in the case of a non-unique rectangle.
    //    There is no dependency on currently locked or unlocked cells other than the
    //    preset cells of the puzzle.  In particular, unlike in most other solving
    //    algorithms, the knowledge of the preset positions
    //    (as opposed to resolved cells during the solving process)
    //    is required to detect all such rectangles.
    //
    // 4. Having resolved one or more or even all cells of such a unique rectangle to
    //    candidates that correspond to the (possible) pattern does not affect the
    //    non-uniqueness of the solution.  Remember that these unique
    //    rectangles are sought out to be avoided - therefore it is possible that cells of
    //    the rectangle are resolved but the rectangle itself is yet to be avoided.
    //
    // Solution strategy:
    //
    // 5. A solving algorithm for these patterns, once detected, needs to consider:
    //    a) if the detected pattern cannot be avoided in a regular puzzle:
    //       - if a guess was made: back track!
    //       - if no guess was made: either the puzzle has no single solution, or a bug occurred.
    //    b) if the detected pattern still can be avoided:
    //       - determine which candidate(s) to remove (if any) assuming a single solution.
    //
    // 6. For a puzzle solving algorithm that includes proof of uniqueness _and_ back tracking:
    //       The avoidance of unique rectangles presumes that only a single solution exists.
    //       Uniqueness checking presumes no such thing, and therefore needs to allow for the
    //       unique rectangle to form.  Either prevent UQR detection altogether when
    //       checking for uniqueness, or choose avoidandance of the unique rectangle as the
    //       primary path of a guess.
    //
    // Terminology:
    // The smallest cell number of all rectangle corners: start cell
    // A side of the rectangle accross 2 boxes: long edge
    // A side od the rectangle within the same box: base edge
    // All cells are connected to a base edge and a long edge.
    // The cells are numbered in clockwise manner from 0 to 3.
    // Each cell is associated to its clockwise right edge.
    // Each long/base edge has an long/base opposite edge.
    // Each cell of the rectangle has a diagonally opposite cell.
    //
    // Unique rectangle types, homegrown notation (aliases where given curtesy of sudokowiki.org):
    // UR-3S
    // - 3 cells with singles, of which 2 have the same value.
    // - action: in the only corner with >1 candidates, eliminate (if it is present)
    //   the candidate that would otherwise complete the UR.
    // Note:
    // This is the only scenario where there is no need for a pair.
    // Therefore in the order of eliminations of possible URs, it needs to be checked before
    // the possible URs are filtered by bivalues.
    //
    // UR-3P - three pairs (aka Type 1):
    // - 3 cells with the same pair
    // - precondition: 4th cell contains one or more candidates of the pair
    // - action: remove the pair candidates from the 4th cell
    //
    // UR-2P-I with direct elimination of UR candidates
    // (I for immediate) (no alias, Hodoku type 6):
    // - two cells contain the same pair of candidates.
    //   These cells can be adjacent or on a diagonal.
    //   If the two cells are adjacent, they form a conjugate pair and share an edge as a strong link.
    //   Note: for the diagonal, the candidate is removed from the diagonal cells (!).
    // - precondition: the other corners contain two candidates from the pair
    //   and other candidates.
    // - precondition 2: one of the start cell candidates x does not appear in either
    //   of 2 parallel edges other than the UR corners (x is a conjugate for these edges).
    //   action: remove y (the other candidate from the pair) from the other cells of the UR,
    //   or in the case of a diagonal, from both ends of the diagonal.
    // Note: if just one edge from one of the pair cells meets precondition 2,
    //   y can only be removed from the other cell not on that edge (does not apply to diagonal).
    // Note 2: UR-2P-(B,L,D) also apply independantly.
    //
    // UR-2P-(B,L,D) with elimination of required candidates present in the corners outside of the UR
    // B: base edge, L: long edge, D: diagonal
    // UR-2P-B (aka Type 2):
    // - base edge corners contain the same pair of candidates (could also both be resolved
    //   but not preset)
    // - precondition: opposite edge corners contain two candidates from the pair
    // - precondition 2: there are other candidates present (in both corners) A of the
    //   other (non-pair) cells
    // - action: consider the size of A: N
    //   N==1: simply remove the extra candidate in A from all cells visible from the non-pair cells
    //   except the two corner cells.
    //   N>1: for each section S shared by the non-pair cells:
    //   examine S for a naked set T of A. If size(T) == N-1:
    //   remove the extra candidates of A from the cells of S that are in T and not corners cells.
    //
    // UR-2P-L (aka Type 2B):
    // - long edge corners contain the same pair of candidates
    // preconditions and actions are the same as for UR-2P-B
    //
    // UR-2P-D (aka Type 2C):
    // - diagonal corners contain the same pair of candidates
    // - precondition: opposite diagonal corners contain the two candidates from the pair
    // - precondition 2: there is a single other candidate present (in both opposite diagonal corners)
    // - action: remove extra candidate from other cells of the triads visible from the corners of the opposite diagonal.
    // Note:
    //   UR-2P-D is quite rare, probably because most likely UR-1P will be detected beforehand
    //   and remove the triggers for detection.
    //
    // UR-1P (no alias, Hodoku: Hidden Rectangle):
    // - one cell (1st cell) contains a pair of candidates, x and y. Select y.
    // - precondition: all other cells of the UR contain candidates of the same pair plus some other
    //   candidates.
    // - precondition 2: the diagonal opposite corner cell, when set to y, will force the other
    //   diagonal's cells both to be set to x.
    //   action: the candidate y can be removed from the cell diagonally opposite to the start cell.
    //   Notes:
    //   1. UR-1P can be easily applied and visualized:
    //   look at the 1st cell and identify the four corners. For one of the 1st cell's
    //   candidates x, if the UR in question provides all the candidate locations in the
    //   opposite edge and the other long edge (i.e. strongly linked).  This is sufficient.
    //
    // Summary of additional preconditions:
    // UR-3P unique rectangle
    //    None
    //
    // UR-1P unique rectangle
    //    conjugate pairs on the same digit (in row/col or box), for the opposite base and opposite long edge.
    //
    // UR-2P-I
    //    pair of opposite strong edges.
    //    a single strong edge on one side eliminates a single candidate on the other side.
    //
    // UR-2P-* unique rectangle
    //    one or multiple 'extra' cell candidates present in both other cells
    //    in the case of a diagonal, only one 'extra' candidate can be present
    //
    // Data structures and algorithm:
    // The data to process UQRs comes from differernt sources:
    // 1. preset cell locations (captured on entry and pre-processed at first use)
    //    this allows skipping 50% or more of all possible unique rectangles
    // 2. bivalues as bit pattern (this data is shared with other algorithms,
    //    e.g. binary universal grave plus one).
    //    This can be leveraged to skip a good percentage of UQRs.
    // 3. SIMD grid processing to collect information on UQRs (diagonal intersection)
    //    as information for further processing.
    //    Once collected for a pair of rows of a band, UQRs can be further
    //    qualified and classified.
    //

if ( mode_uqr )
{
    // compute just in time the superimposed preset rows by band and rc_pair:
    if ( !have_superimposed_preset_rows ) {
        have_superimposed_preset_rows = true;
        for ( int i=0; i<3; i++ ) {
            unsigned short idx[3];
            idx[0] = original_locked.get_indexbits(27*i, 9);
            idx[1] = original_locked.get_indexbits(27*i+9, 9);
            idx[2] = original_locked.get_indexbits(27*i+18, 9);
            superimposed_preset_rows[i][0] = idx[0] | idx[1];
            superimposed_preset_rows[i][1] = idx[0] | idx[2];
            superimposed_preset_rows[i][2] = idx[1] | idx[2];
        }
    }

    bit128_t *candidate_bits_by_value = solverData.getCbbvs(candidates);

    // Process uqrs by band
    unsigned char band = 0;
    if ( last_entered_count_uqr != current_entered_count) {
        // the high byte is set to the stackpointer, so that this works
        // across guess/backtrack
        last_entered_count_uqr = current_entered_count;
    } else {
        band = last_band_uqr;
    }
    for ( ; band<6; band++) {
        bool found_update = false;
        bool rowband = band<3?true:false;
        unsigned char rc_band = band%3;
        last_band_uqr = band;

        unsigned int bandbits = ((bit128_t*)unlocked)->get_indexbits(27, 27*3);
        if ( rowband ) {
            if ( bandbits == 0 ) {  // questionable
                continue;
            }
        } else {
            // once per puzzle:
            // initialize transposed presets and superimposed columns
            if ( !have_superimposed_preset_cols ) {
                have_superimposed_preset_cols = true;

                // first transpose original_locked
                // start by extracting the 9 bits corresponding to each row
                unsigned long long ol64[2] = { original_locked.u64[0] & 0x7fffffffffffffff, (original_locked.u64[0]>>63) | (original_locked.u64[1]<<1) };
                unsigned short ol[16] = { (unsigned short)ol64[0], (unsigned short)(ol64[0]>>9), (unsigned short)(ol64[0]>>18), (unsigned short)(ol64[0]>>27),
                                          (unsigned short)(ol64[0]>>36), (unsigned short)(ol64[0]>>45), (unsigned short)(ol64[0]>>54),
                                          (unsigned short)ol64[1], (unsigned short)(ol64[1]>>9), 0, 0, 0, 0, 0, 0, 0 };

                // one off for digit 9
                __m256i c = _mm256_and_si256(mask1ff, *(__m256i_u*) &ol[0]);
                __m256i c2 = _mm256_permute4x64_epi64(_mm256_packus_epi16(_mm256_srli_epi16(c,1), _mm256_setzero_si256()), 0xD8);
                original_locked_transposed.u64[1] = _mm256_movemask_epi8(c2) << 8;
                c = _mm256_permute4x64_epi64(_mm256_packus_epi16(_mm256_and_si256(c, maskff), _mm256_setzero_si256()), 0xD8);
                unsigned short tmp = _mm256_movemask_epi8(c);
                original_locked_transposed.u64[0]  = (tmp & 1LL)<<63;
                original_locked_transposed.u64[1]  |= tmp >> 1;
                for ( short off=54; off>=0; off -= 9 ) {
                    c = _mm256_slli_epi16(c,1);
                    original_locked_transposed.u64[0] |= (unsigned long long)_mm256_movemask_epi8(c)<<off;
                }

                // second, superimpose the transposed columns
                for ( int i=0; i<3; i++ ) {
                    unsigned short idx[3];
                    idx[0] = original_locked_transposed.get_indexbits(27*i, 9);
                    idx[1] = original_locked_transposed.get_indexbits(27*i+9, 9);
                    idx[2] = original_locked_transposed.get_indexbits(27*i+18, 9);
                    superimposed_preset_cols[i][0] = idx[0] | idx[1];
                    superimposed_preset_cols[i][1] = idx[0] | idx[2];
                    superimposed_preset_cols[i][2] = idx[1] | idx[2];
                }
            }

            if ( (((bit128_t*)unlocked)->u128 & (vband.u128>>(rc_band*3))) == (__uint128_t)0 ) {
                continue;
            }
        }

        // load band data as follows:
        __m256i rc_v[3];

        unsigned short __attribute__ ((aligned(64))) res_data[24]; // contains result data of res

        if ( rowband ) {
            unsigned char rc_indx_ = rc_band*27;

            rc_v[0] = _mm256_loadu2_m128i((__m128i*)&candidates[rc_indx_+6],(__m128i*)&candidates[rc_indx_]);
            rc_v[0] = _mm256_shuffle_epi8(rc_v[0], lineshuffle);
            rc_indx_+=9;
            rc_v[1] = _mm256_loadu2_m128i((__m128i*)&candidates[rc_indx_+6],(__m128i*)&candidates[rc_indx_]);
            rc_v[1] = _mm256_shuffle_epi8(rc_v[1], lineshuffle);
            rc_indx_+=9;
            rc_v[2] = _mm256_loadu2_m128i((__m128i*)&candidates[rc_indx_+6], (__m128i*)&candidates[rc_indx_]);
            rc_v[2] = _mm256_shuffle_epi8(rc_v[2], lineshuffle);
        } else {
            unsigned short *cp = &candidates[rc_band*3];

            rc_v[0] = _mm256_setr_epi16(cp[0], cp[9], cp[18], 0, cp[27], cp[36], cp[45], 0,
                                        cp[54], cp[63], cp[72], 0, 0, 0, 0, 0);
            cp += 1;
            rc_v[1] = _mm256_setr_epi16(cp[0], cp[9], cp[18], 0, cp[27], cp[36], cp[45], 0,
                                        cp[54], cp[63], cp[72], 0, 0, 0, 0, 0);
            cp += 1;
            rc_v[2] = _mm256_setr_epi16(cp[0], cp[9], cp[18], 0, cp[27], cp[36], cp[45], 0,
                                        cp[54], cp[63], cp[72], 0, 0, 0, 0, 0);
        }

        for ( unsigned char rc_pair = 0; rc_pair < 3; rc_pair++ ) {

            // for linev[0]: swap quads to [0, 1, 0, 2]
            // for linev[1]: swap quads to [1, 0, 2, 0]
            // for linev[2]: swap linev[0] quads to [1,3,x2,x3]
            // for linev[3]: swap linev[1] quads to [2,0,x2,x3]
            __m256i linev[4] = { _mm256_permute4x64_epi64(rc_v[row_combos[rc_pair][0]], 0x84),
                                 _mm256_permute4x64_epi64(rc_v[row_combos[rc_pair][1]], 0x21),
                                 _mm256_permute4x64_epi64(linev[0], 0xED),
                                 _mm256_permute4x64_epi64(linev[1], 0xE2) };
            for ( unsigned char perm = 0; perm<3; perm++ ) {
                *(__m256i_u*)&res_data = _mm256_and_si256(linev[0], linev[1]);
                // result (for first iteration):
                // 0.0&1.3, 0.1&1.4, 0.2&1.5, - 0.3&1.0, 4.1&r1.1, 0.5&r1.2, -
                // 0.0&1.6, 0.1&1.7, 0.2&1.8, - 0.6&1.0, 7.1&r1.1, 0.8&r1.2, -

                // Check for UQRs between box 1 and 2
                 *(__m128i_u*)&res_data[16] = _mm256_castsi256_si128(_mm256_and_si256(linev[2], linev[3]));

                // evaluate permutated row/column results for valid uqrs:
                // - if the uqr contains a preset, it is eliminated.
                // - check diag1, diag2 both to be not 0, and at the same, eliminate single uqr candidates in the opposite diagonal.
                // - check that there is at least one bivalue (unless there are 3 single cands)

                const Uqr *permp = cuqrs[perm];
                unsigned char dist2 = rc_pair == 1 ? 2:1;
                unsigned short preset_test;
                unsigned char start_row_indx = 0;
                unsigned char start_col_indx = 0;
                if ( rowband ) {
                    start_row_indx = (rc_band*3 + row_combos[rc_pair][0])*9;
                    preset_test = superimposed_preset_rows[rc_band][rc_pair];
                } else {
                    start_col_indx = rc_band*3 + row_combos[rc_pair][0];
                    preset_test = superimposed_preset_cols[rc_band][rc_pair];
                }
                for (unsigned char uqr_cnt=0; uqr_cnt<9; uqr_cnt++, permp++) {
                    if ( preset_test & permp->pattern ) {
                        continue;
                    }
                    const unsigned char *cuqr_accessp = cuqr_access[uqr_cnt];
                    unsigned short diag[2] = { res_data[cuqr_accessp[0]],
                                               res_data[cuqr_accessp[1]] };

                    // cross elimination based on diagonal info
                    if (__popcnt16(diag[0]) == 1 ) {
                        diag[1] &= ~diag[0];
                    } else if (__popcnt16(diag[1]) == 1 ) {
                        diag[0] &= ~diag[1];
                    }

                    if (diag[0] == 0 || diag[1] == 0 ) {
                        continue;
                    }

                    // before eliminating other rectangles based on bivalues,
                    // check for 3 singles with 2 of them the same (these will always
                    // form a diagonal).
                    // remove the other single from the fourth corner to avoid a unique
                    // rectangle from forming.
                    UqrCorner uqr_corners[4];
                    UqrPair uqr_pairs[3];
                    unsigned short uqr_singles = 0;
                    unsigned short uqr_singles_cnt = 0;
                    unsigned char start_cell = rowband ? permp->start_cell +  start_row_indx : (permp->start_cell*9 + start_col_indx);
                    unsigned char dist1    = permp->dist;

                    // clockwise urq corners
                    uqr_corners[0].indx = start_cell;
                    if ( rowband ) {
                        uqr_corners[1].indx = start_cell+dist1;
                        uqr_corners[2].indx = start_cell+dist1+9*dist2;
                        uqr_corners[3].indx = start_cell+9*dist2;
                    } else {
                        // in the vertical band, dist1 and dist2 are transposed
                        uqr_corners[1].indx = start_cell+dist2;
                        uqr_corners[2].indx = start_cell+9*dist1+dist2;
                        uqr_corners[3].indx = start_cell+9*dist1;
                    }

                    // identify any corner with a bivalue or single
                    unsigned short all_digits = 0;
                    unsigned char not_singles = 0;
                    for ( int i=0; i<4; i++ ) {
                        unsigned short dgts = candidates[uqr_corners[i].indx];
                        all_digits |= dgts;
                        unsigned char cnt = __popcnt16(dgts);
                        if ( cnt == 1 ) {
                            uqr_singles |= dgts;
                            uqr_singles_cnt++;
                        } else {
                            not_singles |= 1<<i;
                        }
                    }

                    // Pattern UR-3S
                    if ( __popcnt16(uqr_singles) == 2 && uqr_singles_cnt == 3 ) {
                        unsigned char ix = __tzcnt_u16(not_singles);
                        char ret[32];
                        format_candidate_set(ret, uqr_singles);
                        unsigned char celli = uqr_corners[ix].indx;
                        unsigned short single = candidates[uqr_corners[(ix+2)&3].indx];
                        if ( (candidates[celli] & single) && __popcnt16(candidates[celli]) > 1 ) {
                            unique_rectangles_avoided++;
                            grid_state->updated.set_indexbit(celli);
                            // unless under 'Regular' rules, capture the avoidable UQR as a guess
                            // provide 'resolution' in form of a guess
                            if ( rules != Regular ) {
                                if ( verbose == VDebug ) {
                                    snprintf(guess_message[0], 196, "to allow unique check of unique rectangle: %s %s - %s\n        3 singles pattern: remove candidate %d from cell %s",
                                            ret, cl2txt[uqr_corners[0].indx], cl2txt[uqr_corners[2].indx],
                                            1+__tzcnt_u16(single), cl2txt[celli]);
                                    snprintf(guess_message[1], 196, "engender unique rectangle %s %s - %s for subsequent unique checking:\n        3 singles pattern: set %d at %s",
                                            ret, cl2txt[uqr_corners[0].indx], cl2txt[uqr_corners[2].indx],
                                            1+__tzcnt_u16(single), cl2txt[celli]);
                                }
                                // call make_guess using a lambda
                                grid_state = grid_state->make_guess<verbose>(
                                    celli,
                                    [=] (GridState &oldgs, GridState &newgs, const char *msg[]) {
                                    // this is the avoidance side of the UQR resolution
                                    // the GridState to continue with:
                                    newgs.candidates[celli] &= ~single;
                                    msg[0] = guess_message[0];
                                    // this is to provoke the UQR
                                    // the GridState to back track to:
                                    oldgs.candidates[celli]  &= ~newgs.candidates[celli];
                                    msg[1] = guess_message[1];
                                });
                                goto guess_made_with_incr;
                            }
                            // otherwise simply avoid the UQR:
                            candidates[celli] &= ~single;
                            if ( verbose == VDebug ) {
                                printf("avoiding unique rectangle: %s %s - %s\n3 singles pattern: remove candidate %d from cell %s\n",
                                    ret, cl2txt[uqr_corners[0].indx], cl2txt[uqr_corners[2].indx],
                                    1+__tzcnt_u16(single), cl2txt[celli]);
                            }
                            if ( (candidates[celli] & (candidates[celli]-1)) ) {
                                goto enter;
                            }
                            found_update = true;
                        }
                    }

                    // if all corners together contain the same 2 digits, a unique rectangle has been found.
                    if ( check_back && not_singles && __popcnt16(all_digits) == 2 ) {
                        if ( grid_state->stackpointer && rules == Regular ) {
                            if ( verbose == VDebug ) {
                                char ret[32];
                                format_candidate_set(ret, all_digits);
                                printf("back track - found a completed unique rectangle %s at %s %s\n", ret, cl2txt[uqr_corners[0].indx], cl2txt[uqr_corners[2].indx]);
                            }
                            goto back;
                        } else {
                            // there's no point doing anything here... except:
                            if ( grid_state->stackpointer == 0 ) {
                                status.unique = false;
                            }
                            // not even:
                            // printf("consciously ignoring a completed unique rectangle at %s %s\n", cl2txt[uqr_corners[0].indx], cl2txt[uqr_corners[2].indx]);
                        }
                    }

                    bit128_t corner_bits {};
                    if ( rowband ) {
                        corner_bits.set_indexbits( permp->pattern, start_row_indx, 9);
                        corner_bits.set_indexbits( permp->pattern, start_row_indx+9*dist2, 9);
                    } else {
                        for ( unsigned char k=0; k<4; k++ ) {
                            corner_bits.set_indexbit(uqr_corners[k].indx);
                        }
                    }

                    if ( (solverData.getBivalues(candidates) & corner_bits) == (__uint128_t)0 ) {
                        continue;
                    }

                    my_unique_rectangles_checked++;

                    // complete corner infos, including edges and uqr_pairs infos
                    for ( int i=0; i<4; i++ ) {
                        unsigned short dgts = candidates[uqr_corners[i].indx];
                        unsigned char cnt = __popcnt16(dgts);
                        if ( cnt == 1 ) {
                            uqr_corners[i].is_single = true;
                        } else if ( cnt == 2 ) {
                            unsigned char pi = 0;
                            // search for pair
                            for ( ; pi<3; pi++) {
                                if ( uqr_pairs[pi].cnt == 0 ) {
                                    uqr_pairs[pi].digits = dgts;
                                    break;
                                } else if ( uqr_pairs[pi].digits == dgts ) {
                                    break;
                                }
                            }
                            if ( pi<3 ) {
                                uqr_pairs[pi].cnt++;
                                uqr_pairs[pi].crnrs |= 1<<i;
                            }
                            uqr_corners[i].pair_indx = pi;
                            uqr_corners[i].is_pair = true;
                        }
                        if ( i & 1 ) {
                            uqr_corners[i].right_edge = (__uint128_t*)small_index_lut[uqr_corners[i].indx%9][Col];
                        } else {
                            uqr_corners[i].right_edge = (__uint128_t*)small_index_lut[uqr_corners[i].indx/9][Row];
                        }
                    }

                    // based on pairs count, process the different scenarios
                    for ( int pi=0; pi<3; pi++ ) {
                        if ( uqr_pairs[pi].cnt == 0 ) {
                            // no pairs (cannot happen)
                            break;
                        }
                        switch ( uqr_pairs[pi].cnt ) {
                        case 3:  // UR-3P (aka Type 1):
                        {
                            // find the 4th corner
                            unsigned char corner4_index = uqr_corners[__tzcnt_u16(~uqr_pairs[pi].crnrs)].indx;
                            unsigned short pair = uqr_pairs[pi].digits;
                            if ( candidates[corner4_index] != pair ) {
                                // unless under 'Regular' rules, provide 'resolution' in form of a guess
                                unique_rectangles_avoided++;
                                if ( rules != Regular ) {
                                    unsigned short other_cands = candidates[corner4_index] & ~pair;
                                    if ( verbose == VDebug ) {
                                        char ret[32];
                                        char ret2[32];
                                        format_candidate_set(ret, pair);
                                        format_candidate_set(ret2, other_cands);
                                        snprintf(guess_message[0], 196, "to allow unique check of unique rectangle: %s %s - %s\n        3 pairs pattern: remove candidates %s from %s",
                                                 ret, cl2txt[uqr_corners[0].indx], cl2txt[uqr_corners[2].indx],
                                                 ret, cl2txt[corner4_index]);
                                        snprintf(guess_message[1], 196, "engender unique rectangle %s %s - %s for subsequent unique checking:\n        3 pairs pattern: remove candidates %s from %s",
                                                 ret, cl2txt[uqr_corners[0].indx], cl2txt[uqr_corners[2].indx],
                                                 ret2, cl2txt[corner4_index]);
                                    }
                                    // call make_guess using a lambda
                                    grid_state = grid_state->make_guess<verbose>(
                                        corner4_index,
                                        [=] (GridState &oldgs, GridState &newgs, const char *msg[]) {
                                        // this is the avoidance side of the UQR resolution
                                        // the GridState to continue with:
                                        newgs.candidates[corner4_index] = other_cands;
                                        msg[0] = guess_message[0];
                                        // this is to provoke the UQR
                                        // the GridState to back track to:
                                        oldgs.candidates[corner4_index]  &= ~other_cands;
                                        msg[1] = guess_message[1];
                                    });
                                    goto guess_made_with_incr;
                                }
                                // simply avoid the UQR
                                candidates[corner4_index] &= ~pair;
                                if ( verbose == VDebug ) {
                                    char ret[32];
                                    format_candidate_set(ret, pair);
                                    printf("avoiding unique rectangle: %s %s - %s\n3 pairs pattern: remove candidates %s from %s\n",
                                            ret, cl2txt[uqr_corners[0].indx], cl2txt[uqr_corners[2].indx],
                                            ret, cl2txt[corner4_index]);
                                }
                                grid_state->updated.set_indexbit(corner4_index);
                                if ( (candidates[corner4_index] & (candidates[corner4_index]-1)) ) {
                                    goto enter;
                                }
                                found_update = true;
                            }
                            break;
                        }
                        case 2:  // UR-2P-I and UR-2P-BLD
                        {
                            // Two pairs of candidates is fairly common.
                            // We have two cells, c1 and c2, which both contain the same
                            // pair (a and b).
                            // For x=a or x=b we examine the cells diagonally opposite of
                            // c1 and c2, o1 and o2.
                            // We call y the other candidate of a or b so that x != y.
                            // (not to get confused, c1 and c2 can by diagonally
                            // opposite of each other, in which case o1=c1 and o2=c2)
                            // What are the conditions to effect, if y is chosen in o1 or o2
                            // that we obtain the unique rectangle x y x y.
                            // That condition is simple:
                            // - if in any pair of opposing edges, x is conjugate with
                            //   the x in the other corner on that edge, then
                            //   y must not be chosen in either o1 or o2 or a unique rectangle forms.
                            //   Hence we should deselect y in o1 and o2.
                            //   if c1 and c2 are adjacent (not diagonal), and the link between c1
                            //   and o1 is weak, but between o2 and c2 it is strong, then y can
                            //   only be eliminated from o1.
                            // Note: there is a part 2 to this procedure.
                            // We can form the set of extra candidates of o1 and o2 if they are
                            // adjacent, and try to find a set in the row or col that contains these
                            // candidates (as they are required they form a virtual set of candidates)
                            // and if a set can be found, its candidates can be eliminated elsewhere
                            // in the row/col.

                            // uqr_cand: the tentative value of x
                            unsigned short uqr_cand = _blsi_u32(uqr_pairs[pi].digits);
                            // all the cells containing that candidate:
                            __uint128_t *cbbv = &candidate_bits_by_value[__tzcnt_u16(uqr_cand)].u128;
                            // either opposite pair will do for both diagonal and side by side
                            // conjugate pairs.
                            // setup the opposing edges without the corners:
                            __uint128_t uqr_opp_edges[2] = {corner_bits,corner_bits};
                            uqr_opp_edges[0] ^= *uqr_corners[0].right_edge | *uqr_corners[2].right_edge;
                            uqr_opp_edges[1] ^= *uqr_corners[1].right_edge | *uqr_corners[3].right_edge;
                            // determine strong edge:
                            unsigned int strong_edge = 0;
                            bool is_diag = false;
                            switch ( uqr_pairs[pi].crnrs ) {
                            case 0b101:
                            case 0b1010:
                                is_diag = true;
                                break;
                            case 0b1001:
                                strong_edge = 3;
                                break;
                            case 0b1100:
                                strong_edge = 2;
                                break;
                            case 0b110:
                                strong_edge = 1;
                                break;
                            default:
                                break;
                            }
                            unsigned short weak_corner = 0xff;
                            unsigned short weak_corner_y = 0xff;
                            int i=0;
                            for ( ; i<2; i++ ) {
                                if (    (uqr_opp_edges[0] & *cbbv) == (__uint128_t)0
                                     || (uqr_opp_edges[1] & *cbbv) == (__uint128_t)0 ) {
                                    // found x
                                    // try for the other candidate too as special case:
                                    // This is a back track scenario, as both candidates
                                    // when selected each cause a completed UR.
                                    // Quite rare as well.
                                    if ( check_back && uqr_cand != uqr_pairs[pi].digits ) {
                                        cbbv = &candidate_bits_by_value[__tzcnt_u16(uqr_cand ^ uqr_pairs[pi].digits)].u128;
                                        if (    (uqr_opp_edges[0] & *cbbv) == (__uint128_t)0
                                             || (uqr_opp_edges[1] & *cbbv) == (__uint128_t)0 ) {

                                            // find a corner with both digits
                                            unsigned short pair_digits = uqr_pairs[pi].digits;
                                            char ret[32];
                                            if ( verbose == VDebug ) {
                                                format_candidate_set(ret, pair_digits);
                                            }
                                            if ( grid_state->stackpointer && rules == Regular ) {
                                                if ( verbose == VDebug ) {
                                                    printf("back track - found an unavoidable unique rectangle %s at %s %s\n", ret, cl2txt[uqr_corners[0].indx], cl2txt[uqr_corners[2].indx]);
                                                }
                                                goto back;
                                            } else {
                                                // there's no point doing anything here...
                                                // not even:
                                                // printf("consciously ignoring a completed unique rectangle at %s %s\n", cl2txt[uqr_corners[0].indx], cl2txt[uqr_corners[2].indx]);
                                            }
                                        }
                                    }
                                    break;
                                } else if ( !is_diag ) {
                                    // strong set is the index of uqr_opp_edges containing the strong link provided by the pair (not for diagonal).
                                    unsigned int strong_set =  (strong_edge & 1) ? 1:0;
                                    // try both sides for weak set...
                                    if ( (((corner_bits & *uqr_corners[(strong_set+3)&3].right_edge) ^ *uqr_corners[(strong_set+3)&3].right_edge) & *cbbv) == 0 ) {
                                        weak_corner = strong_set+1;
                                    } else if ( (((corner_bits & *uqr_corners[strong_set+1].right_edge) ^ *uqr_corners[strong_set+1].right_edge) & *cbbv) == 0 ) {
                                        weak_corner = (strong_set+3)&3;
                                    }
                                    if ( weak_corner != 0xff ) {
                                        unsigned char weak_corner_indx =
                                              uqr_pairs[pi].crnrs & (1<<weak_corner) ?
                                              uqr_corners[(weak_corner+1)&3].indx :
                                              uqr_corners[weak_corner].indx;
                                        weak_corner_y =  uqr_cand ^ uqr_pairs[pi].digits;
                                        if ( (candidates[weak_corner_indx] & weak_corner_y) ) {
                                            grid_state->updated.set_indexbit(weak_corner_indx);
                                            unique_rectangles_avoided++;
                                            // unless under 'Regular' rules, capture the avoidable UQR
                                            // provide 'resolution' in form of a guess
                                            char ret[32];
                                            if ( verbose == VDebug ) {
                                                format_candidate_set(ret, uqr_pairs[pi].digits);
                                            }
                                            if ( rules != Regular ) {
                                                if ( verbose == VDebug ) {
                                                    snprintf(guess_message[0], 196, "to allow unique check of unique rectangle: %s %s - %s\n        2 pair pattern: remove candidate %d at %s",
                                                             ret, cl2txt[uqr_corners[0].indx], cl2txt[uqr_corners[2].indx],
                                                             1+__tzcnt_u16(weak_corner_y), cl2txt[weak_corner_indx]);
                                                    snprintf(guess_message[1], 196, "engender unique rectangle %s %s - %s for subsequent unique checking:\n        2 pair pattern: set %d at %s",
                                                             ret, cl2txt[uqr_corners[0].indx], cl2txt[uqr_corners[2].indx],
                                                             1+__tzcnt_u16(weak_corner_y), cl2txt[weak_corner_indx]);
                                                }
                                                // call make_guess using a lambda
                                                grid_state = grid_state->make_guess<verbose>(
                                                    weak_corner_indx,
                                                    [=] (GridState &oldgs, GridState &newgs, const char *msg[]) {
                                                    // this is the avoidance side of the UQR resolution
                                                    // the GridState to continue with:
                                                    newgs.candidates[weak_corner_indx] &= ~weak_corner_y;
                                                    msg[0] = guess_message[0];
                                                    // this is to provoke the UQR
                                                    // the GridState to back track to:
                                                    oldgs.candidates[weak_corner_indx] = weak_corner_y;
                                                    msg[1] = guess_message[1];
                                                });
                                                goto guess_made_with_incr;
                                            }
                                            // simply avoid the UQR
                                            candidates[weak_corner_indx] &= ~weak_corner_y;
                                            if ( verbose == VDebug ) {
                                                printf("avoiding unique rectangle: %s %s - %s\n2 pair pattern: remove candidate %d at %s\n",
                                                       ret, cl2txt[uqr_corners[0].indx], cl2txt[uqr_corners[2].indx],
                                                       1+__tzcnt_u16(weak_corner_y), cl2txt[weak_corner_indx]);
                                            }
                                            if ( (candidates[weak_corner_indx] & (candidates[weak_corner_indx]-1)) == 0 ) {
                                                goto enter;
                                            }
                                            found_update = true;
                                        }
                                    }
                                }
                                // set x to the other candidate
                                uqr_cand ^= uqr_pairs[pi].digits;
                                if ( uqr_cand == 0 ) {  // nothing to find
                                    i=2;
                                    break;
                                }
                                weak_corner = 0xff;
                                cbbv = &candidate_bits_by_value[__tzcnt_u16(uqr_cand)].u128;
                            }
                            if ( i<2 ) {
                                // select the diagonally opposite corners:
                                // rotate bits in the nibble by 2:
                                unsigned short ix = 0xf & (uqr_pairs[pi].crnrs | uqr_pairs[pi].crnrs<<4)>>2;
                                // from x and the pair of digits, determine y:

                                if ( uqr_cand != uqr_pairs[pi].digits ) {
                                    uqr_cand ^= uqr_pairs[pi].digits;
                                }
                                unsigned char indx2upd[2];
                                for ( int k=0; k<2; k++ ) {
                                    unsigned char indx = __tzcnt_u16(ix);
                                    indx2upd[k] = uqr_corners[indx].indx;
                                    ix &= ~(1<<indx);
                                }
                                if (    candidates[indx2upd[0]] != uqr_cand
                                     || candidates[indx2upd[1]] != uqr_cand ) {
                                    char ret[32];
                                    if ( verbose == VDebug ) {
                                        format_candidate_set(ret, uqr_pairs[pi].digits);
                                    }
                                    found_update = true;
                                    unique_rectangles_avoided++;
                                    if ( rules == Regular ) {
                                        candidates[indx2upd[0]] &= ~uqr_cand;
                                        candidates[indx2upd[1]] &= ~uqr_cand;
                                        if ( verbose == VDebug ) {
                                            printf("avoiding unique rectangle: %s %s - %s\n2 pair pattern: remove candidate %d from cells %s %s\n",
                                                   ret, cl2txt[uqr_corners[0].indx], cl2txt[uqr_corners[2].indx],
                                                   1+__tzcnt_u16(uqr_cand),
                                                   cl2txt[indx2upd[0]], cl2txt[indx2upd[1]]);
                                        }
                                        if (    (candidates[indx2upd[0]] & (candidates[indx2upd[0]] - 1)) == 0
                                             || (candidates[indx2upd[1]] & (candidates[indx2upd[1]] - 1)) == 0 ) {
                                            goto enter;
                                        }
                                    } else {
                                        if ( is_diag ) {
                                            unsigned short other_cand = uqr_cand ^ uqr_pairs[pi].digits;
                                            if ( verbose == VDebug ) {
                                                snprintf(guess_message[0], 196, "to allow unique check of unique rectangle: %s %s - %s\n        2 pair pattern: remove candidate %d at %s and %s",
                                                         ret, cl2txt[uqr_corners[0].indx], cl2txt[uqr_corners[2].indx],
                                                         1+__tzcnt_u16(uqr_cand), cl2txt[indx2upd[0]], cl2txt[indx2upd[1]]);
                                                snprintf(guess_message[1], 196, "engender unique rectangle %s %s - %s for subsequent unique checking:\n        2 pair pattern: remove candidate %d at %s and %s",
                                                        ret, cl2txt[uqr_corners[0].indx], cl2txt[uqr_corners[2].indx],
                                                        1+__tzcnt_u16(other_cand), cl2txt[indx2upd[0]], cl2txt[indx2upd[1]]);
                                            }
                                            // call make_guess using a lambda
                                            grid_state = grid_state->make_guess<verbose>(
                                                        indx2upd[0],
                                                        [=] (GridState &oldgs, GridState &newgs, const char *msg[]) {
                                                            // this is the avoidance side of the UQR resolution
                                                            // the GridState to continue with:
                                                            newgs.candidates[indx2upd[0]] &= ~uqr_cand;
                                                            newgs.candidates[indx2upd[1]] &= ~uqr_cand;
                                                            msg[0] = guess_message[0];
                                                            // this is to provoke the UQR
                                                            // the GridState to back track to:
                                                            oldgs.candidates[indx2upd[0]] &= ~other_cand;
                                                            oldgs.candidates[indx2upd[1]] &= ~other_cand;
                                                            msg[1] = guess_message[1];
                                                        });
                                             goto guess_made_with_incr;
                                        } else {
                                            if ( verbose == VDebug ) {

                                                snprintf(guess_message[0], 196, "to allow unique check of unique rectangle: %s %s - %s\n        2 pair pattern: remove candidate %d at %s and %s",
                                                            ret, cl2txt[uqr_corners[0].indx], cl2txt[uqr_corners[2].indx],
                                                            1+__tzcnt_u16(uqr_cand), cl2txt[indx2upd[0]], cl2txt[indx2upd[1]]);
                                                snprintf(guess_message[1], 196, "engender unique rectangle %s %s - %s for subsequent unique checking:\n        2 pair pattern: set candidate %d at %s",
                                                            ret, cl2txt[uqr_corners[0].indx], cl2txt[uqr_corners[2].indx],
                                                            1+__tzcnt_u16(uqr_cand), cl2txt[indx2upd[0]]);
                                            }
                                            // call make_guess using a lambda
                                            grid_state = grid_state->make_guess<verbose>(
                                                        indx2upd[0],
                                                        [=] (GridState &oldgs, GridState &newgs, const char *msg[]) {
                                                            // this is the avoidance side of the UQR resolution
                                                            // the GridState to continue with:
                                                            newgs.candidates[indx2upd[0]] &= ~uqr_cand;
                                                            newgs.candidates[indx2upd[1]] &= ~uqr_cand;
                                                            msg[0] = guess_message[0];
                                                            // this is to provoke the UQR
                                                            // the GridState to back track to:
                                                            oldgs.candidates[indx2upd[0]] = uqr_cand;
                                                            msg[1] = guess_message[1];
                                                        });
                                             goto guess_made_with_incr;
                                        }
                                    }
                                }
                            } // for

                            // continuing with UR-P2-BLD pattern
                            unsigned char pat = uqr_pairs[pi].crnrs;
                            unsigned short non_uqr_cands = 0;
                            unsigned char non_pairs_celli[2];
                            unsigned char cl = 0;
                            for ( unsigned char i=0; i<2; i++, cl++, pat >>= 1 ) {
                                while ( (pat & 1) ) {
                                    pat >>= 1;
                                    cl++;
                                }
                                non_pairs_celli[i] = uqr_corners[cl].indx;
                                non_uqr_cands |= candidates[non_pairs_celli[i]];
                            }
                            // remove the UR candidates to avoid:
                            non_uqr_cands &= ~uqr_pairs[pi].digits;
                            unsigned char non_uqr_cands_cnt = __popcnt16(non_uqr_cands);

                            // continuing with UR-P2-BLD patterns (only for Regular rules)
                            //
                            if ( rules == Regular ) {
                                // build the intersection from:
                                // - 2 cell visibilities
                                // - candidate digits pattern
                                // - minus the 4 UR cells
                                bit128_t cand_removal_indx {};

                                bool check_set = false;
                                if ( non_uqr_cands_cnt == 1 ) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
                                    cand_removal_indx.u128 = *(const __uint128_t*)&big_index_lut[non_pairs_celli[0]][All];
                                    cand_removal_indx.u128 &= *(const __uint128_t*)&big_index_lut[non_pairs_celli[1]][All];
                                    cand_removal_indx.u128 &= candidate_bits_by_value[__tzcnt_u16(non_uqr_cands)].u128;
#pragma GCC diagnostic pop
                                    check_set = true;
                                } else if ( !is_diag ) {   // exclude UR-2P-D pattern
                                    unsigned short non_uqr_cands_ = non_uqr_cands;
                                    while ( non_uqr_cands_ ) {
                                        unsigned char cand_ = __tzcnt_u16(non_uqr_cands_);
                                        non_uqr_cands_ = _blsr_u32(non_uqr_cands_);
                                        cand_removal_indx.u128 |= candidate_bits_by_value[cand_].u128;
                                    }

                                    if ( (candidates[non_pairs_celli[0]] & ~non_uqr_cands) && (candidates[non_pairs_celli[1]] & ~non_uqr_cands) ) {
                                        unsigned int m = 0;
                                        __m256i a = _mm256_set1_epi16(non_uqr_cands);
                                        bool is_row = (non_pairs_celli[1] - non_pairs_celli[0])%9;

                                        if ( is_row ) {
                                            // row based
                                            unsigned short row = non_pairs_celli[0]/9;
                                            __m256i res = _mm256_cmpeq_epi16(a, _mm256_or_si256(a, *(__m256i_u*) &candidates[9*row]));
                                            m = compress_epi16_boolean<true>(res);
                                        } else {
                                            // column based
                                            unsigned short ci = non_pairs_celli[0]%9;
                                            __m256i c = _mm256_set_epi16(0,0,0,0,0,0,0,candidates[ci+72], candidates[ci+63], candidates[ci+54], candidates[ci+45], candidates[ci+36], candidates[ci+27], candidates[ci+18], candidates[ci+9], candidates[ci]);
                                            m = compress_epi16_boolean<true>(_mm256_cmpeq_epi16(a, _mm256_or_si256(a, c)));
                                        }
                                        bit128_t cand_visibility_indx {};
                                        if ( _popcnt32(m & 0x3ffff)>>1 == non_uqr_cands_cnt-1 ) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
                                            cand_visibility_indx.u128 = *(const __uint128_t*)&big_index_lut[non_pairs_celli[0]][is_row?Row:Col];
#pragma GCC diagnostic pop
                                            check_set = true;
                                        }

                                        // box based
                                        unsigned short b = box_start[non_pairs_celli[0]];
                                        if ( b == box_start[non_pairs_celli[1]] ) {
                                            __m256i c = _mm256_set_epi16(0,0,0,0,0,0,0,candidates[b+20], candidates[b+19], candidates[b+18], candidates[b+11], candidates[b+10], candidates[b+9], candidates[b+2], candidates[b+1], candidates[b]);
                                            m = compress_epi16_boolean<true>(_mm256_cmpeq_epi16(a, _mm256_or_si256(a, c)));
                                            if ( (_popcnt32(m & 0x3ffff)>>1) == (non_uqr_cands_cnt-1) ) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
                                                cand_visibility_indx.u128 |= *(const __uint128_t*)&big_index_lut[non_pairs_celli[0]][Box];
#pragma GCC diagnostic pop
                                                check_set = true;
                                            }
                                        }
                                        cand_removal_indx.u128 &= cand_visibility_indx.u128;
                                    }
                                }

                                cand_removal_indx.u128 &= ~corner_bits;

                                // double check to avoid multiple hits
                                if ( check_set ) {
                                    bit128_t cand_removal_indx_ = cand_removal_indx;
                                    while ( cand_removal_indx_ ) {
                                        unsigned char j = tzcnt_and_mask(cand_removal_indx_);

                                        // check for set members, must not remove
                                        // check also for locations where there is nothing to remove
                                        if (    (candidates[j] & ~non_uqr_cands) == 0
                                             || (candidates[j] &  non_uqr_cands) == 0 ) {
                                            cand_removal_indx.unset_indexbit(j);
                                        }
                                    }
                                }

                                // iterate over cells and remove the candidate to avoid
                                if ( check_set && cand_removal_indx != (__uint128_t)0 ) {
                                    unique_rectangles_avoided++;
                                    found_update = true;
                                    if ( verbose == VDebug ) {
                                        char ret[32];
                                        char ret2[32];
                                        format_candidate_set(ret, uqr_pairs[pi].digits);
                                        format_candidate_set(ret2, non_uqr_cands);
                                        printf("avoiding unique rectangle: %s %s - %s\n2 pair pattern: remove candidates %s from cells outside UR at: ",
                                               ret, cl2txt[uqr_corners[0].indx], cl2txt[uqr_corners[2].indx],
                                               ret2);
                                    }
                                    bool got_single = false;
                                    const char *cma = "";
                                    while ( cand_removal_indx ) {
                                        unsigned char j = tzcnt_and_mask(cand_removal_indx);
                                        unsigned short cj = candidates[j] & ~non_uqr_cands;
                                        if ( cj ) {
                                            candidates[j] = cj;
                                            if ( (cj & (cj-1)) == 0 ) {
                                                got_single = true;
                                            }
                                            grid_state->updated.set_indexbit(j);
                                            if ( verbose == VDebug ) {
                                                printf("%s%s", cma, cl2txt[j]);
                                                cma = ",";
                                            }
                                        }
                                    }
                                    if ( verbose == VDebug ) {
                                        printf("\n");
                                    }
                                    if ( got_single ) {
                                        goto enter;
                                    }
                                }
                            } else {
                                // same as above, but as a lambda guess
                                // ... just too tedious
                            }
                            break;
                        }
                        case 1:
                        {
                            // uqr_cand: the tentative value of x
                            unsigned short uqr_cand = _blsi_u32(uqr_pairs[pi].digits);
                            int j=0;
                            unsigned char crnr_indx = (__tzcnt_u16(uqr_pairs[pi].crnrs)+2)&3;
                            unsigned char indx = uqr_corners[crnr_indx].indx;
                            __uint128_t uqr_crnr_edges = *uqr_corners[crnr_indx].right_edge | *uqr_corners[(crnr_indx+3)&3].right_edge;
                            __uint128_t *cbbv;
                            unsigned short uqr_alt_cand;
                            for ( ; j<2; j++ ) {
                                uqr_alt_cand = uqr_cand ^ uqr_pairs[pi].digits;
                                // check whether there is anyting to remove:
                                if ( candidates[indx] & uqr_alt_cand ) {
                                    // all the cells containing that candidate:
                                    cbbv = &candidate_bits_by_value[__tzcnt_u16(uqr_cand)].u128;
                                    if ( ((uqr_crnr_edges & *cbbv) | corner_bits) == corner_bits ) {
                                        // found x
                                        break;
                                    }
                                }
                                // switch to the other candidate
                                uqr_cand = uqr_alt_cand;
                                if ( uqr_cand == 0 ) {  // nothing to find
                                    j=2;
                                    break;
                                }
                            }
                            if ( j<2 ) {
                                // y: uqr_alt_cand
                                // simply eliminate y from the opposite corner

                                if ( candidates[indx] & uqr_alt_cand ) {
                                    unique_rectangles_avoided++;
                                    grid_state->updated.set_indexbit(indx);
                                    // unless under 'Regular' rules, capture the avoidable UQR
                                    // provide 'resolution' in form of a guess
                                    if ( rules == Regular ) {
                                        // simply avoid the UQR
                                        candidates[indx] &= ~uqr_alt_cand;
                                        if ( verbose == VDebug ) {
                                            char ret[32];
                                            format_candidate_set(ret, uqr_pairs[pi].digits);
                                            printf("avoiding unique rectangle: %s %s - %s\n1 pair pattern: remove candidate %d from cell %s\n",
                                                    ret, cl2txt[uqr_corners[0].indx], cl2txt[uqr_corners[2].indx],
                                                    1+__tzcnt_u16(uqr_alt_cand), cl2txt[indx]);
                                        }
                                        found_update = true;
                                        if ( (candidates[indx] & (candidates[indx]-1)) == 0 ) {
                                            goto enter;
                                        }
                                    } else {
                                        if ( verbose == VDebug ) {
                                            char ret[32];
                                            format_candidate_set(ret, uqr_pairs[pi].digits);
                                            snprintf(guess_message[0], 196, "to allow unique check of unique rectangle: %s %s - %s\n        1 pair pattern: remove candidate %d at %s",
                                                    ret, cl2txt[uqr_corners[0].indx], cl2txt[uqr_corners[2].indx],
                                                    1+__tzcnt_u16(uqr_alt_cand), cl2txt[indx]);
                                            snprintf(guess_message[1], 196, "engender unique rectangle %s %s - %s for subsequent unique checking:\n        1 pair pattern: set %d at %s",
                                                    ret, cl2txt[uqr_corners[0].indx], cl2txt[uqr_corners[2].indx],
                                                    1+__tzcnt_u16(uqr_alt_cand), cl2txt[indx]);
                                        }
                                        // call make_guess using a lambda
                                        grid_state = grid_state->make_guess<verbose>(
                                                indx,
                                                [=] (GridState &oldgs, GridState &newgs, const char *msg[]) {
                                                    // this is the avoidance side of the UQR resolution
                                                    // the GridState to continue with:
                                                    newgs.candidates[indx] &= ~uqr_alt_cand;
                                                    msg[0] = guess_message[0];
                                                    // this is to provoke the UQR
                                                    // the GridState to back track to:
                                                    oldgs.candidates[indx] = uqr_alt_cand;
                                                    msg[1] = guess_message[1];
                                                });
                                        goto guess_made_with_incr;
                                    }
                                }
                            }
                            break;
                        } // case 1
                        } // switch
                    } // for pairs
                } // for 6 uqrs

                // finally rotate positions for next group of 9 UQRs
                // for line 0 and 2 rotate first and third group clockwise by 1,
                // for line 1 and 3 rotate second and fourth group clockwise by 1
                linev[0] = _mm256_shuffle_epi8(linev[0], linerotate[0]);
                linev[1] = _mm256_shuffle_epi8(linev[1], linerotate[1]);
                linev[2] = _mm256_shuffle_epi8(linev[2], linerotate[0]);
                linev[3] = _mm256_shuffle_epi8(linev[3], linerotate[1]);
            }
        }
        if ( found_update ) {
            last_band_uqr = (band+1)%6;
            goto enter;
        };
    }
}
#endif

guess:
    // Make a guess if all that didn't work
    grid_state = grid_state->make_guess<verbose>(&solverData);
    current_entered_count++;            // increment by one for the guess made.
    // guess_made: when the guess was 'made' in some other way.
    // That should include a change to grid_state and paired changes to this and the previous
    // grid_states.
// guess_made:
    no_guess_incr = 0;
#ifdef OPT_UQR
// if the guess was made solely to allow for checking uniqueness, still count the solution
// as direct solve
guess_made_with_incr:
#endif
    current_entered_count += 0x100;     // increment high byte for the new grid_state, plus one for the guess made.
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
    -c  check for back tracking even when no guess was made (e.g. if puzzles might have no solution)
    -d# provide some detailed information on the progress of the puzzle solving.
        add a 2 or even 3 for even more detail.
    -h  help information (this text)
    -l# solve a single line from the puzzle.
    -m[FSTU]* execution modes (fishes, sets, triads, unique rectangles), lower or upper case
    -r[ROM] puzzle rules (lower or upper case):
        R  for regular puzzles (unique solution exists)
           not suitable for puzzles that have multiple solutions
        O  find just one solution
           this will find the first of multiple solutions and stop looking
        M  determine whether multiple solutions exist beyond the first found
    -t# set the number of threads
    -v  verify the solution
    -w  display warning (mostly unexpected solving details for regular puzzles)
    -x  provide some statistics
    -y  provide speed statistics only
    -#1 change base for row and column reporting from 0 to 1

    fishes details can be shown as a 9x9 grid with '-d3' where 
       'o' represents fish positions as 'o'
       '@' represents fin(s) if present
       'X' shows and candidates to eliminate
       '.' shows other candidates containing the same digit

)");
}

int main(int argc, const char *argv[]) {
using namespace Schoku;

    int line_to_solve = 0;

    // the debug dprintf and dprintfilter are not used in cheched-in code
    // they are initialized here at no cost just in case...
    const char *schoku_dbg_filter = getenv("SCHOKU_DBG_FILTER");
    if ( schoku_dbg_filter ) {
       unsigned off = 0;
       while ( schoku_dbg_filter[off] == '0' ) {
           off++;
       }
       if ( schoku_dbg_filter[off] == 'x' ) {
            sscanf(&schoku_dbg_filter[off+1], "%x", &dprintfilter);
       } else {
            sscanf(&schoku_dbg_filter[off], "%d", &dprintfilter);
       }
    }

    if ( argc > 0 ) {
        argc--;
        argv++;
    }

    char opts[80] = { 0 };
    for (int i = 0; i < argc; i++) {
        sprintf(opts+strlen(opts), "%s ", argv[i]);
    }

    while ( argc && argv[0][0] == '-' ) {
        if (argv[0][1] == 0) {
            argv++; argc--;
            break;
        }
        switch(argv[0][1]) {
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
             for ( unsigned char p=2; argv[0][p] && p<6; p++) {
                 switch (toupper(argv[0][p])) {
#ifdef OPT_SETS
                case 'S':        // see OPT_SETS
                    mode_sets = true;
                    break;
#endif
#ifdef OPT_TRIAD_RES
                case 'T':        // see OPT_TRIAD_RES
                    mode_triad_res = true;
                    break;
#endif
#ifdef OPT_UQR
                case 'U':        // see OPT_UQR
                    mode_uqr = true;
                    break;
#endif
#ifdef OPT_FSH
                case 'F':        // see OPT_FSH
                    mode_fish = true;
                    break;
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
             if ( argv[0][2] && isdigit(argv[0][2]) ) {
                 int displaybase = 0;
                 sscanf(&argv[0][2], "%d", &displaybase);
                 if ( displaybase == 1 ) {
                     for ( int i=0; i<81; i++ ) {
                        cl2txt[i][1]++;
                        cl2txt[i][3]++;
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
    // suppress uqr mode if unique checking is requested,
    // avoiding severe complications in the code.
    if ( rules != Regular ) {
        if ( mode_uqr && warnings != 0 ) {
            printf("uqr checking mode ( -mU ) disabled when checking when not under default Regular rules\n");
        }
        mode_uqr = false;
    }

    assert((sizeof(GridState) & 0x3f) == 0);

   // sort out the CPU and OMP settings

   if ( !__builtin_cpu_supports("avx2") ) {
        fprintf(stderr, "This program requires a CPU with the AVX2 instruction set.\n");
        exit(0);
    }
    // lacking BMI support? unlikely!
    if ( !__builtin_cpu_supports("bmi") ) {
        fprintf(stderr, "This program requires a CPU with the BMI instructions (such as blsr)\n");
        exit(0);
    }

    bmi2_support = __builtin_cpu_supports("bmi2") && !__builtin_cpu_is("znver2");

    if ( debug ) {
         omp_set_num_threads(1);
         printf("debug mode requires restriction of the number of threads to 1\n");

         printf("BMI2 instructions %s %s\n",
               __builtin_cpu_supports("bmi2") ? "found" : "not found",
               bmi2_support? "and enabled" : __builtin_cpu_is("znver2")? "but use of pdep/pext instructions disabled":"");
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
	size_t npuzzles = (fsize - pre + (1-post))/82;
    if ( line_to_solve < 0 || npuzzles < (unsigned long)line_to_solve ) {
		fprintf(stderr, "Ignoring the given line number\nthe input file %s contains %ld puzzles, the given line number %d is not between 1 and %ld\n", ifn, npuzzles, line_to_solve, npuzzles);
        line_to_solve = 0;
    }
    size_t outnpuzzles = line_to_solve ? 1 : npuzzles;

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
    if ( ftruncate(fdout, (size_t)outnpuzzles*164) == -1 ) {
		if (errno ) {
			fprintf(stderr, "Error: setting size (ftruncate) on output file %s: %s\n", ofn, strerror(errno));
		}
		exit(0);
	}

	// map the output file
    output = (signed char *)mmap((void*)0, outnpuzzles*164, PROT_WRITE, MAP_SHARED, fdout, 0);
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
        // copy unsolved grid
        memcpy(output, &string_pre[i], 81);
        // add comma and newline in right place
        output[81] = ',';
        output[163] = 10;
        // solve the grid in place
        if ( stack == 0 ) {
            // force alignment the 'old-fashioned' way
            // stack = (GridState*)malloc(sizeof(GridState)*GRIDSTATE_MAX);
            stack = (GridState*) (~0x3fll & ((unsigned long long) malloc(sizeof(GridState)*GRIDSTATE_MAX+0x40)+0x40));
        }

        signed char *grid = &output[82];

        stack[0].initialize(output);

        if ( debug != 0) {
            solve<VDebug>(grid, stack, line_to_solve);
        } else if ( reportstats !=0) {
            solve<VStats>(grid, stack, line_to_solve);
        } else {
            solve<VNone>(grid, stack, line_to_solve);
        }
    } else {

        // The OMP directives:
        // proc_bind(close): high preferance for thread/core affinity
        // firstprivate(stack): the stack is allocated for each thread once and seperately
        // schedule(dynamic,64): 64 puzzles are allocated at a time and these chunks
        //   are assigned dynmically (to minimize random effects of difficult puzzles)
        // shared(...) lists the variables that are shared (as opposed to separate copies per thread)
        //
#pragma omp parallel firstprivate(stack) proc_bind(close) shared(string_pre, output, npuzzles, imax, debug, reportstats)
        {
            // force alignment the 'old-fashioned' way
            // not going to free the data ever
            // stack = (GridState*)malloc(sizeof(GridState)*GRIDSTATE_MAX);
            stack = (GridState*) (~0x3fll & ((unsigned long long) malloc(sizeof(GridState)*GRIDSTATE_MAX+0x40)+0x40));

#pragma omp for schedule(dynamic,120)
            for (size_t i = 0; i < imax; i+=82) {
                // copy unsolved grid
                signed char *grid = &output[i*2+82];
                memcpy(&output[i*2], &string_pre[i], 81);
                // add comma and newline in right place
                output[i*2 + 81] = ',';
                output[i*2 + 163] = 10;
                // solve the grid in place
                stack[0].initialize(&output[i*2]);
                if ( debug != 0) {
                    solve<VDebug>(grid, stack, i/82+1);
                } else if ( reportstats !=0) {
                    solve<VStats>(grid, stack, i/82+1);
                } else {
                    solve<VNone>(grid, stack, i/82+1);
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
	err = munmap(output, (size_t)npuzzles*164);
	if ( err == -1 ) {
		if (errno ) {
			fprintf(stderr, "Error: munmap file %s: %s\n", ofn, strerror(errno));
		}
	}

    if ( reportstats) {
        long long duration = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration(std::chrono::steady_clock::now() - starttime)).count();
        printf("schoku version: %s\ncommand options: %s\ncompile options: %s\n", version_string, opts, compilation_options);
        printf("%10ld  puzzles entered\n", npuzzles);
        unsigned long solved_cnt = solved_count.load();
        printf("%10ld  %.0lf/s  puzzles solved\n", solved_cnt, (double)solved_cnt/((double)duration/1000000000LL));
		printf("%8.1lfms  %6.2lf\u00b5s/puzzle  solving time\n", (double)duration/1000000, (double)duration/(npuzzles*1000LL));
        if ( unsolved_count.load() && rules != Regular) {
            printf("%10ld  puzzles had no solution\n", unsolved_count.load());
        }
        if ( rules == Multiple ) {
            printf("%10ld  puzzles had multiple solutions\n", non_unique_count.load());
        }
        if ( verify ) {
            printf("%10ld  puzzle solutions were verified\n", verified_count.load());
        }
        printf( "%10ld  %6.2f%%  puzzles solved without guessing\n", no_guess_cnt.load(), (double)no_guess_cnt.load()/solved_cnt*100);
        printf( "%10ld  %6.2f/puzzle  guesses\n", guesses.load(), (double)guesses.load()/(double)solved_cnt);
        printf( "%10ld  %6.2f/puzzle  back tracks\n", trackbacks.load(), (double)trackbacks.load()/solved_cnt);
        printf("%10lld  %6.2f/puzzle  digits entered and retracted\n", digits_entered_and_retracted.load(), (double)digits_entered_and_retracted.load()/solved_cnt);
        printf("%10lld  %6.2f/puzzle  'rounds'\n", past_naked_count.load(), (double)past_naked_count.load()/solved_cnt);
        printf("%10lld  %6.2f/puzzle  triads resolved\n", triads_resolved.load(), (double)triads_resolved.load()/solved_cnt);
        printf("%10lld  %6.2f/puzzle  triad updates\n", triad_updates.load(), (double)triad_updates.load()/solved_cnt);
#ifdef OPT_SETS
        if ( mode_sets ) {
            printf("%10lld  %6.2f/puzzle  naked sets found\n", naked_sets_found.load(), (double)naked_sets_found.load()/solved_cnt);
            printf("%10lld  %6.2f/puzzle  naked sets searched\n", naked_sets_searched.load(), (double)naked_sets_searched.load()/solved_cnt);
        }
#endif
#ifdef OPT_FSH
        if ( mode_fish ) {
            unsigned long long fsh_detected = fishes_detected.load(); // fishes_specials_detected is a subset of fishes_detected
            unsigned long long fsh_updates = fishes_updated.load(); // fishes_specials_updated is a subset of fishes_updated;
            printf("%10lld  %6.2f/puzzle  fishes updated\n", fsh_updates, (double)fsh_updates/solved_cnt);
            printf("%10lld  %6.2f/puzzle  fishes detected\n", fsh_detected, (double)fsh_detected/solved_cnt);
         }
#endif
#ifdef OPT_UQR
        if ( mode_uqr ) {
            printf("%10lld  %6.2f/puzzle  unique rectangles avoided\n", unique_rectangles_avoided.load(), (double)unique_rectangles_avoided.load()/solved_cnt);
            printf("%10lld  %6.2f/puzzle  unique rectangles checked\n", unique_rectangles_checked.load(), (double)unique_rectangles_checked.load()/solved_cnt);
        }
#endif
        if ( bug_count.load() ) {
            printf("%10ld  bi-value universal graves detected\n", bug_count.load());
        }
    } else if ( reporttimings ) {
        long long duration = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration(std::chrono::steady_clock::now() - starttime)).count();
		printf("%8.1lfms  %6.2lf\u00b5s/puzzle  solving time\n", (double)duration/1000000, (double)duration/(npuzzles*1000LL));
    }

    if ( !reportstats && rules == Multiple && non_unique_count.load()) {
        printf("%10ld  puzzles had more than one solution\n", non_unique_count.load());
    }
    if ( verify && not_verified_count.load()) {
        printf("%10ld  puzzle solutions verified as not correct\n", not_verified_count.load());
    }
    if ( !reportstats && unsolved_count.load()) {
        printf("%10ld puzzles had no solution\n", unsolved_count.load());
    }

    return 0;
}
#endif
