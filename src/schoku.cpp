/// This code uses AVX2 instructions...
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
 * Version 1.0 speed
 *
 * Performance changes:
 * - full implementation of triad processing
 *   both removal of extra triad candidates and full triad set detection (with -mT)
 * - process columns first in the hidden singles search to gain some performance.
 * - moved __m256i constants to the program level
 * - AVX2 implementation of reading the puzzle file and writing back the solution
 * - additional optimization to hidden single search for boxes
 *
 * Functional changes:
 * - adjustments for more diversified puzzle sources (tdoku puzzle files)
 * - warnings option (-w): by default the messages and warnings are muted.
 *   With -w pertinent messages (mostly for regular rules 'surprises') are shown.
 * - mode options (-mT) to dynamically invoke features.
 * - rules options (-r[ROM]) to either assume regular puzzles rules (single solution),
 *   search for one solution, even if multiple solutions exist, or determine whether
 *   two or more solutions exist.
 * - increased the maximum number of Gridstate structs to 34, as one puzzle set needed
 *   more than 28.
 * - allow for multi-line comments at the start of puzzle files
 * - disallow CR/LF (MS-DOS/Windows) line endings
 *
 *
 * Performance measurement and statistics:
 *
 * data: 17-clue sudoku (49151 puzzles)
 * CPU:  Ryzen 7 4700U
 *
 * schoku version: 1.0 speed
 * command options: -x puzzles.txt
 * compile options:
 *      49151  puzzles entered
 *      49151  2433977/s  puzzles solved
 *     20.2ms   410ns/puzzle  solving time
 *      38596   78.53%  puzzles solved without guessing
 *      25410    0.52/puzzle  guesses
 *      16721    0.34/puzzle  back tracks
 *     218423    4.44/puzzle  digits entered and retracted
 *      26259    0.53/puzzle  'rounds'
 *     117195    2.38/puzzle  triads resolved
 *     212409    4.32/puzzle  triad updates
 *        704  bi-value universal graves detected
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

const char *version_string = "1.0 speed";

const char *compilation_options =
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

// bit128_t type
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

    inline bool check_indexbit(unsigned char idx) {
        return this->u8[idx>>3] & (1<<(idx & 0x7));
    }
    inline bool check_and_mask_index(unsigned char idx) {
        return _bittestandreset64((long long int *)&this->u64[idx>>6], idx & 0x3f);
    }
    inline void set_indexbit(unsigned char idx) {
        // slightly faster than _bittestandset
        this->u8[idx>>3] |= 1<<(idx & 0x7);
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
    // should translate into one unaligned load plus shift plus and 'and' (which can be optimized away as needed)
    // pos must not be less than 32.
    // 32 >= pos < 64, bitcount <= 64
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

const signed char box_perp_ind_incr[9] = { 1, 1, 7, 1, 1, 7, 1, 1, -20};

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
const __m256i mask9 { -1LL, -1LL, 0xffffLL, 0 };
const __m256i ones9 = _mm256_and_si256(ones, mask9);

// used in triads:
const __m256i mask11hi { 0LL, 0LL, 0xffffLL<<48, ~0LL };
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
const __m256i mask27 { -1LL, (long long int)0xffffffffffff00ffLL, (long long int)0xffffffff00ffffffLL, 0xffffffffffLL };
const __m256i threes    = _mm256_set1_epi8 ( 3 );
const __m256i fours     = _mm256_set1_epi8 ( 4 );
//   popcnt by nibble
const __m256i lookup    = _mm256_setr_epi8(0 ,1 ,1 ,2 ,1 ,2 ,2 ,3 ,1 ,2 ,2 ,3 ,2 ,3 ,3 ,4,
                                           0 ,1 ,1 ,2 ,1 ,2 ,2 ,3 ,1 ,2 ,2 ,3 ,2 ,3 ,3 ,4);

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

alignas(64)
std::atomic<long long> past_naked_count(0); // how often do we get past the naked single serach
std::atomic<long long> digits_entered_and_retracted(0); // to measure guessing overhead
std::atomic<long long> triads_resolved(0);  // how many triads did we resolved
std::atomic<long long> triad_updates(0);    // how many triads did cancel candidates
std::atomic<long> bug_count(0);             // universal grave detected
std::atomic<long> guesses(0);               // how many guesses did it take
std::atomic<long> trackbacks(0);            // how often did we back track
std::atomic<long> solved_count(0);          // puzzles solved
std::atomic<long> no_guess_cnt(0);          // how many puzzles were solved without guessing
std::atomic<long> unsolved_count(0);        // puzzles unsolved (no solution exists)
std::atomic<long> non_unique_count(0);      // puzzles not unique (with -u)
std::atomic<long> not_verified_count(0);    // puzzles non verified (with -v)
std::atomic<long> verified_count(0);        // puzzles successfully verified (with -v)
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

// puzzle rules
typedef enum {
    Regular  = 0,   // R - Default: Find solution, assuming it will be unique
                    // (this is fastest but is not suitable for non-unique puzzles)
    FindOne  = 1,   // O - Search for one solution
    Multiple = 2    // M - Determine whether puzzles are non-unique
} Rules;

Rules rules;

signed char *output;

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

public:
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
    GridState* & grid_state;  // tethered to the pointer, no the instance!!
    CommonData &commonData;

public:

    inline SolverData(GridState* &gs, CommonData& cd): grid_state(gs), commonData(cd) {}

    inline void resetStage() {
        memset(update_stage_bytes, 0, 96);
        update_stage_9_bits.u128 = 0;
        update9 = false;
        update_cycles = 0;
    }

    template<Verbosity verbose>
    inline bool stage_cell_resolution( unsigned short e_digit, unsigned char e_i, const char *msg = "" );

    template<Verbosity verbose>
    inline bool update_candidates() {
        return update_candidates<verbose>(update_stage_9_bits, update_stage_bytes);
    };

    // same, but using externally supplied data
    template<Verbosity verbose>
    inline bool update_candidates(bit128_t &, unsigned char * );

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
    unsigned int multiple_solutions_exist; // indicator for multiple solutions in this grid_state
                                      // aligned on 16 bytes
    bit128_t unlocked;                // for keeping track of which cells still need to be resolved. Set bits correspond to cells that still have multiple possibilities
    bit128_t updated;                 // for keeping track of which cell's candidates may have been changed since last time we looked for naked sets. Set bits correspond to changed candidates in these cells
    bit128_t set23_found[3];          // for keeping track of found sets of size 2 and 3

// GridState is normally copied for recursion
// initialize the starting state including the puzzle.
// Phase 1 (this method) fills a staging area with the presets.
// Phase 2 is executed using SolverData.update_candidates
//
template<Verbosity verbose>
inline void initialize_and_stage(signed char grid[81], bit128_t &digit_bits9, unsigned char *stagebytes, char *lineout) {
    if ( verbose == VDebug ) {
        memset(lineout, '0', 81);
    }

    unlocked.u64[1] = 0;

    multiple_solutions_exist = 0;

    triads_unlocked[0] = triads_unlocked[1] = 0x1ffLL | (0x1ffLL<<10) | (0x1ffLL<<20);
//    set23_found[0] = set23_found[1] = set23_found[2] = {__int128 {0}};

    // set unlocked
    for (unsigned int i=0; i<64; i +=32 ) {
        __m256i in = *(__m256i_u*)&grid[i];
        unlocked.u32[i>>5] = _mm256_movemask_epi8(_mm256_cmpgt_epi8(dgt1, in));
    }
    __m128i in = *(__m128i_u*)&grid[64];
    unlocked.u16[4] = _mm_movemask_epi8(_mm_cmpgt_epi8(_mm256_castsi256_si128(dgt1),in));

    if ( grid[80] <= '0' ) {
        unlocked.u16[5] = 1;
    }
    updated = unlocked;

    bit128_t digit_bits[9] {};

    bit128_t locked = { .u128 = ~unlocked.u128 };
    locked.u64[1] &= 0x1ffff;

    // preset candidates
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
            int dgt = grid[dix] - 49;
            digit_bits[dgt].u128 = digit_bits[dgt].u128 | *cast2cu128(big_index_lut[dix][All]);
            candidates[dix] = 1<<dgt;
            if ( verbose == VDebug ) {
                lineout[dix] = '1' + dgt;
            }
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
inline __attribute__((always_inline)) void enter_digit( unsigned short digit, unsigned char i) {
    // lock this cell and and remove this digit from the candidates in this row, column and box

    bit128_t to_update;
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

    set_indices<All>(&to_update, i);

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
template<Verbosity verbose>
inline GridState* make_guess(TriadInfo &triad_info, bit128_t &bivalues) {
    // Make a guess for a triad with 4 candidate values that has 2 candidates that are not
    // constrained to the triad (not in 'tmust') and has at least 2 or more unresolved cells.
    // If we cannot obtain such a triad, fall back to make_guess().
    // With such a triad found, select one of the 2 identified candidates
    // to eliminate in the new GridState, issue the debug info and proceed.
    // Save the current GridState to back track to and eliminate the other candidate from it.

    unsigned char tpos;    // grid cell index of triad start
    unsigned char type;    // row == 0, col = 1
    unsigned char inc;     // increment for iterating triad cells
    unsigned short *wo_musts;  // pointer to triads' 'without must' candidates
    for ( int i=0; i<2; i++ ) {
        type = i;
        unsigned long long totest = triad_info.triads_selection[i];
        wo_musts  = type==0?triad_info.row_triads_wo_musts:triad_info.col_triads_wo_musts;
        inc  = (i<<3) + 1;  // 1 for rows, 9 for cols
        while (totest) {
            int ti = tzcnt_and_mask(totest);
            int can_ti = ti-ti/10;   // 'canonical' triad index
            if ( __popcnt16 (wo_musts[ti]) == 2 ) {

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
    return make_guess<verbose>(bivalues);
found:
    // update the current and the new grid_state with their respective candidate to delete
    unsigned short select_cand = 0x8000 >> __lzcnt16(*wo_musts);
    unsigned short other_cand  = *wo_musts & ~select_cand;

    // Create a copy of the state of the grid to make back tracking possible
    GridState* new_grid_state = this+1;
    if ( stackpointer >= GRIDSTATE_MAX-1 ) {
        fprintf(stderr, "Error: no GridState struct availabe\n");
        exit(0);
    }
    memcpy(new_grid_state, this, sizeof(GridState));
    new_grid_state->stackpointer++;

    unsigned char off = tpos;
    for ( unsigned char k=0; k<3; k++, tpos += inc) {
        new_grid_state->candidates[tpos] &= ~select_cand;
        candidates[tpos] &= ~other_cand;
    }
    // Update candidates
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
    unsigned char best_cnt = 16;

    if ( bivalues.u64[0] ) {
        guess_index = _tzcnt_u64(bivalues.u64[0]);
    } else if ( bivalues.u64[1] ) {
        guess_index = _tzcnt_u64(bivalues.u64[1]) + 64;
    } else {
        // very unlikely
        unsigned char cnt;
        unsigned char i_rel;
            
        unsigned long long to_visit = unlocked.u64[0];
        while ( best_cnt > 3 && to_visit != 0 ) {
            i_rel = tzcnt_and_mask(to_visit);
            cnt = __popcnt16(candidates[i_rel]);
            if (cnt < best_cnt) {
                best_cnt = cnt;
                guess_index = i_rel;
            }
        }

        to_visit = unlocked.u64[1];
        while ( best_cnt > 3 && to_visit != 0 ) {
            i_rel = tzcnt_and_mask(to_visit) + 64;
            cnt = __popcnt16(candidates[i_rel]);
            if (cnt < best_cnt) {
                best_cnt = cnt;
                guess_index = i_rel;
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

#if 0
template <Verbosity verbose>
inline bool SolverData::stage_cell_resolution( unsigned short e_digit, unsigned char e_i, const char *msg ) {

    bit128_t to_update;

    assert( e_digit != 0 );

#ifndef NDEBUG
    if ( __popcnt16(e_digit) != 1 ) {
        if ( warnings != 0 ) {
            printf("error in e_digit: %x\n", e_digit);
        }
    }
#endif

    if (e_i < 64) {
        _bittestandreset64((long long int *)&grid_state->unlocked.u64[0], e_i);
    } else {
        _bittestandreset64((long long int *)&grid_state->unlocked.u64[1], e_i-64);
    }

    grid_state->candidates[e_i] = e_digit;
    commonData.current_entered_count++;

    set_indices<All>(&to_update, e_i);
    grid_state->updated.u128 |= to_update.u128;

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
                    unsigned char e_i = i+tzcnt_and_mask(mask);
                    // manually inlined from stage_cell_resolution (not a big imporevement)
                    unsigned short e_digit = grid_state->candidates[e_i];
                    bit128_t to_update;

#ifndef NDEBUG
                    if ( __popcnt16(e_digit) != 1 ) {
                        if ( warnings != 0 ) {
                            printf("error in e_digit: %x\n", e_digit);
                        }
                    }
#endif

                    if (e_i < 64) {
                        _bittestandreset64((long long int *)&grid_state->unlocked.u64[0], e_i);
                    } else {
                        _bittestandreset64((long long int *)&grid_state->unlocked.u64[1], e_i-64);
                    }

                    // unnecessary, we get here for naked singles only!
                    // grid_state->candidates[e_i] = e_digit;
                    commonData.current_entered_count++;

                    set_indices<All>(&to_update, e_i);
                    grid_state->updated.u128 |= to_update.u128;

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
Status solve(signed char gridin[81], signed char grid[81], GridState stack[], int line) {

    GridState *grid_state = &stack[0];
    unsigned long long *unlocked = grid_state->unlocked.u64;
    unsigned short* candidates;

    Status status;

    char gridout[82];
    // borrowing the data area from the unoccupied grid_state[1]
    grid_state->initialize_and_stage<verbose>(gridin, (grid_state+1)->unlocked, (unsigned char *)(grid_state+1), gridout);
    if ( verbose == VDebug ) {
        printf("Line %d: %.81s\n", line, gridout);
    }

    // count resolved/entered cells in combination with guess level
    // allowing to recompute data when additional cells have been resolved.
    //
    // the low byte is the real count, while
    // the high byte is increased/decreased with each guess/back track
    // init count of resolved cells to the size of the initial set
    unsigned short current_entered_count = 81 - grid_state->unlocked.popcount();

    unsigned short last_entered_count_col_triads = 0;

    bool check_back = thorough_check;

    int unique_check_mode = 0;

    CommonData commonData(current_entered_count, unique_check_mode, check_back, line);
    SolverData solverData(grid_state, commonData);
    solverData.update_cycles = 3;
    solverData.update9 = true;
    // using data from initialization in the unoccupied grid_state[1]
    solverData.update_candidates<verbose>( (grid_state+1)->unlocked, (unsigned char *)(grid_state+1));

    bool nonunique_reported = false;

    unsigned long long my_digits_entered_and_retracted = 0;
    unsigned char no_guess_incr = 1;

    unsigned int my_past_naked_count = 0;

    // The 'API' for code that uses the 'goto enter:' method of entering digits
    unsigned short e_digit = 0;
    unsigned char e_i = 0;

    bit128_t bivalues;

    // duplicate sequence of instructions, see under label start: 
    check_back = grid_state->stackpointer || thorough_check || rules != Regular || unique_check_mode;

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
            unsolved_count++;
        }
        // cleanup and return
        if ( verbose != VNone && reportstats ) {
            past_naked_count += my_past_naked_count;
            digits_entered_and_retracted += my_digits_entered_and_retracted;
            if ( status.unique == false ) {
                non_unique_count++;
            }
        }
        if ( !unique_check_mode ) {
            // failed - just copy the input
            memcpy(grid, grid-82, 81);
        }
        return status;
    }

    current_entered_count  = (((grid_state-1)->stackpointer)<<8) | (81 - (grid_state-1)->unlocked.popcount());        // back to previous stack.

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
    if ( verbose != VNone && reportstats ) {
        trackbacks++;
    }
    grid_state--;

start:

    e_digit = 0;

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

        set_indices<All>(&to_update, e_i);

        grid_state->updated.u128 |= to_update.u128;

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
            if (__popcnt16(candidates[80]) == 1) {
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
    if ( *(__uint128_t*)unlocked == 0) {
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

        if ( rules == Multiple && unique_check_mode == 0 && grid_state->multiple_solutions_exist == 0) {
            if ( grid_state->stackpointer ) {
                if ( verbose == VDebug ) {
                   printf("Solution: %.81s\nBack track to determine uniqueness\n", grid);
                }
                unique_check_mode = 1;
                goto back;
            }
            // otherwise uniqueness checking is complete
        }
        if ( verbose != VNone && reportstats ) {
            past_naked_count += my_past_naked_count;
            digits_entered_and_retracted += my_digits_entered_and_retracted;
            if ( !status.unique ) {
                non_unique_count++;
            }
        }
        return status;
    }

hidden_search:

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
    // Combine 8 cells from 2 rows into one __m256i vector. (9th row: use __m128i vector)
    // Rotate and or until each vector element represents 7 cells or'ed (except the cell
    // directly corresponding to its position, containing the hidden single if there is one).
    // Broadcast the nineth cell and or it for good measure, then andnot with 0x1ff
    // to isolate the hidden singles.
    // For the nineth cell, rotate and or one last time and use just one element of the result
    // to check the nineth cell for a hidden single.
    //
    // For boxes, loading of the data is the most expensive.
    // You can follow the 'column' approach or the 'row' approach.  The column approach
    // performs better, although it is harder to understand.
    //
    // All checks use compression to a bit vector
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
    // Note that this is not normally part of the locked set solution strategy.
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
                    unsigned int m = _mm256_movemask_epi8(_mm256_cmpgt_epi16(missing, _mm256_setzero_si256()));
                    int idx = __tzcnt_u32(m)>>1;
                    unsigned short digit = ((__v16hu)missing)[idx];
                    if ( grid_state->stackpointer == 0 && unique_check_mode == 0 ) {
                        if ( warnings != 0 ) {
                            printf("Line %d: stack 0, back track - column %d misses digit %d\n", line, idx, __tzcnt_u16(digit)+1);
                        }
                    } else if ( debug ) {
                        printf("back track - missing digit %d in column %d\n", __tzcnt_u16(digit)+1, idx);
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
                    int idx = __tzcnt_u32(mask);
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
        // The grid is processed as indicated by the keys below:
        // P1 P1 P1 P1 P1 P1 P1 P1 C9
        // P1 P1 P1 P1 P1 P1 P1 P1 C9
        // P2 P2 P2 P2 P2 P2 P2 P2 C9
        // P2 P2 P2 P2 P2 P2 P2 P2 C9
        // P3 P3 P3 P3 P2 P3 P3 P3 C9
        // P3 P3 P3 P3 P4 P3 P3 P3 C9
        // P4 P4 P4 P4 P4 P4 P4 P4 C9
        // P4 P4 P4 P4 P4 P4 P4 P4 C9
        // R9 R9 R9 R9 R9 R9 R9 R9 C80

        // Px: processed in pairs of rows, for their first 8 cells
        // C9: the 9th cell of each of the first 8 rows and boxes is prepared
        //     and stored in row_9th_cand_vert.  row_9th_cand_vert is then processed at the end.
        // R9: is processed for 8 cells of each rows and boxes, the cell 80
        //     is processed using the row and box logic and stored in cand80_row
        //     and cand80_box
        // C80: the value of cand80_row is examined last

        // The examination of each row and its cells is made up of three steps:
        // 1 - or the other values of the row
        // 2 - check that the row or box contains all digits
        // 3 - check negated or'ed value for non-zero candidates.
        //     if there are multiple candidates, back track,
        //     enter any valid singletons found

        // row_9th_cand_vert cumulates the 9th row or'ed singleton candidates
        // which are processed in the end.
        __v8hu row_9th_cand_vert;
        // the last singleton candidate of the grid.
        unsigned short cand80_row = 0;
        unsigned char irow = 0;

        // find hidden singles in rows

        for (unsigned char i = 0; i < 72; i += 18, irow+=2) {
            // rows, in pairs

             __m256i c1 = _mm256_set_m128i(*(__m128i_u*) &candidates[i+9],
                                           *(__m128i_u*) &candidates[i]);

            unsigned short the9thcand_row[2] = { candidates[i+8], candidates[i+17] };

            __m256i row_or7 = _mm256_setzero_si256();
            __m256i row_or8;
            __m256i row_9th = _mm256_set_m128i(_mm_set1_epi16(the9thcand_row[1]),_mm_set1_epi16(the9thcand_row[0]));

            __m256i row_triad_capture[2];
            {
                __m256i c1_ = c1;
                // A2 and A3.1.b
                // c1 2 lanes for two rows
                // step j=0
                // rotate left (0 1 2 3 4 5 6 7) -> (1 2 3 4 5 6 7 0)
                c1_ = _mm256_alignr_epi8(c1_, c1_, 2);
                row_or7 = _mm256_or_si256(c1_, row_or7);
                // step j=1
                // rotate (1 2 3 4 5 6 7 0) -> (2 3 4 5 6 7 0 1)
                c1_ = _mm256_alignr_epi8(c1_, c1_, 2);
                row_or7 = _mm256_or_si256(c1_, row_or7);
                // triad capture: after 2 rounds, row triad 3 of these rows saved in pos 5, 13
                row_triad_capture[0] = _mm256_or_si256(row_or7, row_9th);
                // step j=2
                // rotate (2 3 4 5 6 7 0 1) -> (3 4 5 6 7 0 1 2)
                c1_ = _mm256_alignr_epi8(c1_, c1_, 2);
                row_or7 = _mm256_or_si256(c1_, row_or7);
                // triad capture: after 3 rounds 2 row triads 0 and 1 in pos 7 and 2
                row_triad_capture[1] = row_or7;
                // continue the rotate/or routine for this row
                for (unsigned char j = 3; j < 7; ++j) {
                    // rotate (0 1 2 3 4 5 6 7) -> (1 2 3 4 5 6 7 0)
                    // c1 2 lanes for two row
                    c1_ = _mm256_alignr_epi8(c1_, c1_, 2);
                    row_or7 = _mm256_or_si256(c1_, row_or7);
                }

                row_or8 = _mm256_or_si256(c1, row_or7);
                // test row_or8 to hold all the digits
                if ( check_back ) {
                    if ( !_mm256_testz_si256(mask1ff,_mm256_andnot_si256(_mm256_or_si256(row_9th, row_or8), mask1ff))) {
                        // the current grid has no solution, go back
                        if ( verbose != VNone ) {
                            unsigned int m = _mm256_movemask_epi8(_mm256_cmpeq_epi16(_mm256_setzero_si256(),
                            _mm256_andnot_si256(_mm256_or_si256(row_9th, row_or8), mask1ff)));
                            if ( grid_state->stackpointer == 0 && unique_check_mode == 0 ) {
                                if ( warnings != 0 ) {
                                    printf("Line %d: stack 0, back track - row %d does not contain all digits\n", line, irow+(m & 0xffff)?0:1);
                                }
                            } else if ( debug ) {
                                printf("back track - missing digit in row %d\n", irow+(m & 0xffff)?0:1);
                            }
                        }
                        goto back;
                    }
                }
            }
            // hidden singles in row
            __m256i row_mask = _mm256_andnot_si256(_mm256_or_si256(row_9th, row_or7), mask1ff);
            {
                // check row (8) candidates
                unsigned short m1 = grid_state->unlocked.get_indexbits(i, 8) | (grid_state->unlocked.get_indexbits(i+9, 8)<<8);
                __m256i a1 = _mm256_cmpgt_epi16(row_mask, _mm256_setzero_si256());

                unsigned int mask1 = and_compress_masks<false>(a1, m1);
                if (mask1) {
                    int idx = __tzcnt_u32(mask1);
                    int idx_ = idx + ((mask1 & 0xff)?0:1);
                    e_i = i + idx_;
                    e_digit = ((v16us)row_mask)[idx];
                    if ( __popcnt16(e_digit) == 1 ) {
                        if ( verbose == VDebug ) {
                            printf("hidden single (row)");
                        }
                        goto enter;
                    }
                    if ( grid_state->stackpointer == 0 && unique_check_mode == 0 ) {
                        if ( warnings != 0 ) {
                            printf("Line %d: stack 0, back track - %s cell %s does contain multiple hidden singles\n", line, mask1?"row":"box", cl2txt[e_i]);
                        }
                    } else if ( debug ) {
                        printf("back track - multiple hidden singles in row cell %s\n", cl2txt[e_i]);
                    }
                    e_digit = 0;
                    goto back;
                }
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
            _mm_storeu_si64(&triad_info.row_triads[row_triads_lut[irow+1]],  _mm256_extracti128_si256(row_triad_capture[0],1));

            if ( i == 72 ) {
                break;
            }

            if ( irow < 8 ) {
                row_9th_cand_vert[irow]     = ~((v16us)row_or8)[0] & the9thcand_row[0];
                row_9th_cand_vert[irow+1]   = ~((v16us)row_or8)[8] & the9thcand_row[1];
            }
        }   // for row Px

        // boxes:
        //
        // Note on nomenclature:
        // The prefix box_perp stands for the special arrangement of the box data:
        // Just as columns are arranged perpendicular to rows, so the box_perp are
        // arranged as perpendicular (columns as an analogy) to the box representation
        // in their canonical sequence of cells (rows as an analogy).
        // One representation of box cells could be transposed into the other,
        // in the same manner rows and columns can be transposed.
        // This whole section follows step by step the algorithm applied to columns
        // above, which is much easier to grasp.
        {
            __m256i box_perp_or_tails[9];
            __m256i box_perp_or_head = _mm256_setzero_si256();
            __m256i box_perp_cand_or = _mm256_setzero_si256();
            // save the loaded 'rows':
            __m256i box_perp_rows[9];

            {
                box_perp_or_tails[8] = _mm256_setzero_si256();
                // box 'columns' alias box_perp 'rows'
                // to start, we simply tally the or'ed box_perp 'rows'
            
                // A2 (box_perps)
                // precompute 'tails' of the or'ed box_perp_cand_or only once
                // working backwords
                unsigned short *candp = candidates;

                __m256i box_perp_0 = box_perp_rows[0] = _mm256_setr_epi16(candp[0], candp[3],candp[6],
                                                                   candp[27], candp[30],candp[33],
                                                                   candp[54], candp[57],candp[60], 0, 0, 0, 0, 0, 0, 0);
                candp += 20;
                for ( unsigned char j = 8; j > 0; candp -= box_perp_ind_incr[--j]) {
                    // the box_perp 'row' for later, at the price of 9 __m256i... 
                    box_perp_rows[j] = _mm256_setr_epi16(candp[0], candp[3],candp[6],
                                                          candp[27], candp[30],candp[33],
                                                          candp[54], candp[57],candp[60], 0, 0, 0, 0, 0, 0, 0);
                    box_perp_or_tails[j-1] = box_perp_cand_or = _mm256_or_si256(box_perp_cand_or, box_perp_rows[j]);
                }
                // or in box_perp[0] and check whether all digits or covered
                if ( check_back && !_mm256_testz_si256(mask9x1ff, _mm256_andnot_si256(_mm256_or_si256(box_perp_cand_or, box_perp_0), mask9x1ff)) ) {
                    // the current grid has no solution, go back
                    if ( verbose != VNone ) {
                        __m256i missing = _mm256_andnot_si256(_mm256_or_si256(box_perp_cand_or, box_perp_rows[0]), mask9x1ff);
                        unsigned int m  = _mm256_movemask_epi8(_mm256_cmpgt_epi16(missing, _mm256_setzero_si256()));
                        int idx = __tzcnt_u32(m)>>1;
                        unsigned short digit = ((__v16hu)missing)[idx];
                        if ( grid_state->stackpointer == 0 && unique_check_mode == 0 ) {
                            if ( warnings != 0 ) {
                                printf("Line %d: stack 0, back track - box %d misses digit %d\n", line, idx, __tzcnt_u16(digit)+1);
                            }
                        } else if ( debug ) {
                            printf("back track - missing digit %d in box %d\n", __tzcnt_u16(digit)+1, idx);
                        }
                    }
                    goto back;
                }

                for (unsigned int j = 0; j < 9; j++ ) {
                    // computing the bitmask of unlocked box cells is too expensive,
                    // instead mask out all singles (which are necessarily locked).
                    __m256i box_perp_j_singles = _mm256_cmpeq_epi16(_mm256_and_si256(box_perp_rows[j], _mm256_sub_epi16(box_perp_rows[j], ones)), _mm256_setzero_si256());
                    box_perp_cand_or = _mm256_or_si256(box_perp_or_tails[j], box_perp_cand_or);
                    unsigned int mask = compress_epi16_boolean<false>(_mm256_andnot_si256(box_perp_j_singles, _mm256_cmpgt_epi16(mask9x1ff, box_perp_cand_or)));
                    if ( mask) {
                        int idx = __tzcnt_u32(mask);
                        // idx gives us the box and j gives us the cell within the box.
                        e_i = box_start_by_boxindex[idx] + box_offset[j];
                        e_digit = 0x1ff & ~((v16us)box_perp_cand_or)[idx];
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
                            printf("hidden single (box)");
                        }
                        goto enter;
                    }

                    if ( j == 8 ) {
                        break;
                    }

                    // leverage previously computed or'ed box_perp 'rows' in head and tails.
                    box_perp_cand_or = box_perp_or_head = _mm256_or_si256(box_perp_rows[j], box_perp_or_head);            
                }
            }
        }

        // 9th row
        {
            __m128i c = *(__m128i_u*) &candidates[72];

            unsigned short the9thcand_row = candidates[80];

            __m128i row_or7 = _mm_setzero_si128();
            __m128i row_or8;
            __m128i row_9th_elem = _mm_set1_epi16(the9thcand_row);

            __m128i row_triad_capture[2];
            {
                __m128i c_ = c;
                // A2 and A3.1.b
                // step j=0
                // rotate left (0 1 2 3 4 5 6 7) -> (1 2 3 4 5 6 7 0)
                c_ = _mm_alignr_epi8(c_, c_, 2);
                row_or7 = _mm_or_si128(c_, row_or7);
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
                // continue the rotate/or routine for this row
                for (unsigned char j = 3; j < 7; ++j) {
                    // rotate (0 1 2 3 4 5 6 7) -> (1 2 3 4 5 6 7 0)
                    c_ = _mm_alignr_epi8(c_, c_, 2);
                    row_or7 = _mm_or_si128(c_, row_or7);
                }

                row_or8 = _mm_or_si128(c, row_or7);
                // test row_or8 | row_9th_elem to hold all the digits
                if ( check_back ) {
                    if ( !_mm_testz_si128(_mm256_castsi256_si128(mask1ff),_mm_andnot_si128(_mm_or_si128(row_9th_elem, row_or8), _mm256_castsi256_si128(mask1ff)))) {
                        // the current grid has no solution, go back
                        if ( verbose != VNone ) {
                            __m128i missing = _mm_andnot_si128(_mm_or_si128(row_9th_elem, row_or8), _mm256_castsi256_si128(mask1ff));
                            unsigned int m = _mm_movemask_epi8(_mm_cmpeq_epi16(_mm_setzero_si128(), missing));
                            //  unsigned char s_idx = __tzcnt_u32(m)>>1;
                            unsigned short digit = ((__v8hu)missing)[(m & 0xffff)?8:0];
                            if ( grid_state->stackpointer == 0 && unique_check_mode == 0 ) {
                                if ( warnings != 0 ) {
                                    printf("Line %d: stack 0, back track - row %d misses digit %d\n", line, 8, __tzcnt_u16(digit)+1);
                                }
                            } else if ( debug ) {
                                printf("back track - missing digit %d in row %d\n", __tzcnt_u16(digit)+1, 8);
                            }
                        }
                        goto back;
                    }
                }
            }

            // hidden singles in row 9
            __m128i row_mask = _mm_andnot_si128(_mm_or_si128(row_9th_elem, row_or7), _mm256_castsi256_si128(mask1ff));
            {
                // check row (8) candidates

                __m128i a = _mm_cmpgt_epi16(row_mask, _mm_setzero_si128());
                unsigned short mask = compress_epi16_boolean128<false>(a) & grid_state->unlocked.get_indexbits(72, 8);
                if (mask) {
                    int s_idx = __tzcnt_u32(mask);
                    e_i = 72 + s_idx;
                    e_digit = ((v8us)row_mask)[s_idx];
                    // Check that the single is indeed a single
                    if ( __popcnt16(e_digit) == 1 ) {
                        if ( verbose == VDebug ) {
                            printf("hidden single (row)");
                        }
                        goto enter;
                    } else {
                        if ( verbose != VNone ) {
                            if ( grid_state->stackpointer == 0 && unique_check_mode == 0 ) {
                                if ( warnings != 0 ) {
                                    printf("Line %d: stack 0, back track - row cell %s does contain multiple hidden singles\n", line, cl2txt[e_i]);
                                }
                            } else if ( debug ) {
                                printf("back track - multiple hidden singles in row cell %s\n", cl2txt[e_i]);
                            }
                        }
                        e_digit = 0;
                        goto back;
                    }
                }
            } // row

            // deal with saved row_triad_capture:
            // - captured triad 3 of this row saved in pos 5 of row_triad_capture[0]
            // - captured triads 0 and 1 in pos 7 and 2 of row_triad_capture[1]
            // Blend the two captured vectors to contain all three triads in pos 7, 2 and 5
            // shuffle the triads from pos 7,2,5 into pos 0,1,2
            // spending 3 instructions on this: blend, shuffle, storeu
            // a 'random' 4th unsigned short is overwritten by the next triad store
            // (or is written into the gap 10th slot).
            row_triad_capture[0] = _mm_shuffle_epi8(_mm_blend_epi16(row_triad_capture[1], row_triad_capture[0], 0x20), _mm256_castsi256_si128(shuff725to012));
            _mm_storeu_si64(&triad_info.row_triads[row_triads_lut[8]], row_triad_capture[0]);

            cand80_row = ~((v8us)row_or8)[0] & the9thcand_row;

        } // row R9

        // check saved row singleton candidates (9th column)

        unsigned int mask = _mm_movemask_epi8(_mm_cmpgt_epi16((__m128i)row_9th_cand_vert, _mm_setzero_si128()));
        while (mask) {
            int s_idx = __tzcnt_u32(mask) >> 1;
            unsigned char celli = s_idx*9+8;
            unsigned short cand = ((v8us)row_9th_cand_vert)[s_idx];
            if ( ((bit128_t*)unlocked)->check_indexbit(celli) ) {
                // check for a single.
                // This is rare as it can only occur when a wrong guess was made.
                // the current grid has no solution, go back
                if ( check_back && cand & (cand-1) ) {
                    if ( verbose != VNone ) {
                        if ( grid_state->stackpointer == 0 && unique_check_mode == 0 ) {
                            if ( warnings != 0 ) {
                                printf("Line %d: stack 0, multiple hidden singles in row cell %s\n", line, cl2txt[celli]);
                            }
                        } else if ( debug ) {
                            printf("back track - multiple hidden singles in row cell %s\n", cl2txt[celli]);
                        }
                    }
                    goto back;
                }
                if ( verbose == VDebug ) {
                    printf("hidden single (row)");
                }
                e_i = celli;
                e_digit = cand;
                goto enter;
            }
            mask &= ~(3<<(s_idx<<1));
        }

        // check cell 80
        if ( cand80_row && ((bit128_t*)unlocked)->check_indexbit(80) ) {
            // check for a single.
            // This is rare as it can only occur when a wrong guess was made.
            // the current grid has no solution, go back
            if ( check_back && (cand80_row & (cand80_row-1)) ) {
                if ( verbose != VNone ) {
                    if ( grid_state->stackpointer == 0 && unique_check_mode == 0 ) {
                        if ( warnings != 0 ) {
                            printf("Line %d: stack 0, multiple hidden singles in row cell %s\n", line, cl2txt[80]);
                        }
                    } else if ( debug ) {
                        printf("back track - multiple hidden singles in row cell %s\n", cl2txt[80]);
                    }
                }
                goto back;
            }
            if ( verbose == VDebug ) {
                printf("hidden single (row)");
            }
            e_i = 80;
            e_digit = cand80_row;
            goto enter;
        } // cell 80

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
        while (m) {
            unsigned char tidx = tzcnt_and_mask(m);
            unsigned short cands_triad = triad_info.col_triads[tidx];
            if ( verbose == VDebug ) {
                char ret[32];
                format_candidate_set(ret, cands_triad);
                printf("triad set (col): %-9s %s\n", ret, cl2txt[tidx/10*3*9+tidx%10]);
            }
            // mask off resolved triad:
            _bittestandreset((int*)&grid_state->triads_unlocked[Col], tidx);
//            unsigned char off = tidx%10+tidx/10*27;
//            grid_state->set23_found[Col].set_indexbits(0x40201,off,19);
//            grid_state->set23_found[Box].set_indexbits(0x40201,off,19);
            if ( verbose != VNone && reportstats ) {
                triads_resolved++;
            }
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
//            grid_state->set23_found[Row].set_indexbits(0x7,off,3);
//            grid_state->set23_found[Box].set_indexbits(0x7,off,3);
            if ( verbose != VNone && reportstats ) {
                triads_resolved++;
            }
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
                        if ( verbose != VNone && reportstats ) {
                            triads_resolved++;
                        }
                    }
                    if ( verbose == VDebug ) {
                        char ret[32];
                        format_candidate_set(ret, ((__v16hu)to_remove_v)[i_rel]);
                        printf("remove %-5s from %s triad at %s\n", ret, type == 0? "row":"col",
                               cl2txt[type==0?row_triad_canonical_map[ltidx]*3:col_canonical_triad_pos[ltidx]] );
                    }
                    if ( verbose != VNone && reportstats ) {
                        triad_updates++;
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
                for ( unsigned char k=0; k<2; k++ ) {
                    if ( row_combo_tpos[k] ) {
                        grid_state->updated.set_indexbits(row_combo_tpos[k], k*27, 27);
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
    my_past_naked_count++;

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

        bivalues.u16[5] = (__popcnt16(candidates[80]) == 2)?1:0;
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
                    if ( verbose == VDebug ) {
                        printf("bi-value universal grave pivot:");
                    }
                    if ( rules == Regular ) {
                        e_i = target;
                        e_digit = digit;
                        goto enter;
                    } else {
                        if ( verbose == VDebug ) {
                            printf("\n");
                        }
                        grid_state = grid_state->make_guess<verbose>(target, digit);
                    }
                    goto start;
                }
            }
        }
        no_bug:
        ;
    }

guess:
    // Make a guess if all that didn't work
    grid_state = grid_state->make_guess<verbose>(triad_info, bivalues);
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
    -c  check for back tracking even when no guess was made (e.g. if puzzles might have no solution)
    -d# provide some detailed information on the progress of the puzzle solving.
        add a 2 or even 3 for even more detail.
    -h  help information (this text)
    -l# solve a single line from the puzzle.
    -r[ROM] puzzle rules:
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
    -#1 change base for row and column reporting from 0 to 1

)");
}

int main(int argc, const char *argv[]) {

using namespace Schoku;

    int line_to_solve = 0;

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
            stack->stackpointer=0;    
        }

        signed char *grid = &output[82];

        if ( debug != 0) {
            solve<VDebug>(grid-82, grid, stack, line_to_solve);
        } else if ( reportstats !=0) {
            solve<VStats>(grid-82, grid, stack, line_to_solve);
        } else {
            solve<VNone>(grid-82, grid, stack, line_to_solve);
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

#pragma omp for schedule(monotonic:dynamic,120)
            for (size_t i = 0; i < imax; i+=82) {
                // copy unsolved grid
                signed char *grid = &output[i*2+82];
                memcpy(&output[i*2], &string_pre[i], 81);
                // add comma and newline in right place
                output[i*2 + 81] = ',';
                output[i*2 + 163] = 10;
                if ( debug != 0) {
                    solve<VDebug>(grid-82, grid, stack, i/82+1);
                } else if ( reportstats !=0) {
                    solve<VStats>(grid-82, grid, stack, i/82+1);
                } else {
                    solve<VNone>(grid-82, grid, stack, i/82+1);
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
        printf("%10ld  %.0lf/s  puzzles solved\n", solved_count.load(), (double)solved_count.load()/((double)duration/1000000000LL));
        if ( duration/npuzzles > 1000 ) {
		    printf("%8.1lfms  %6.2lf\u00b5s/puzzle  solving time\n", (double)duration/1000000, (double)duration/(npuzzles*1000LL));
        } else {
            printf("%8.1lfms  %4dns/puzzle  solving time\n", (double)duration/1000000, (int)((double)duration/npuzzles));
        }
        if ( unsolved_count.load() && rules != Regular) {
            printf("%10ld  puzzles had no solution\n", unsolved_count.load());
        }
        if ( rules == Multiple ) {
            printf("%10ld  puzzles had multiple solutions\n", non_unique_count.load());
        }
        if ( verify ) {
            printf("%10ld  puzzle solutions were verified\n", verified_count.load());
        }
        printf( "%10ld  %6.2f%%  puzzles solved without guessing\n", no_guess_cnt.load(), (double)no_guess_cnt.load()/(double)solved_count.load()*100);
        printf( "%10ld  %6.2f/puzzle  guesses\n", guesses.load(), (double)guesses.load()/(double)solved_count.load());
        printf( "%10ld  %6.2f/puzzle  back tracks\n", trackbacks.load(), (double)trackbacks.load()/(double)solved_count.load());
        printf("%10lld  %6.2f/puzzle  digits entered and retracted\n", digits_entered_and_retracted.load(), (double)digits_entered_and_retracted.load()/(double)solved_count.load());
        printf("%10lld  %6.2f/puzzle  'rounds'\n", past_naked_count.load(), (double)past_naked_count.load()/(double)solved_count.load());
        printf("%10lld  %6.2f/puzzle  triads resolved\n", triads_resolved.load(), triads_resolved.load()/(double)solved_count.load());
        printf("%10lld  %6.2f/puzzle  triad updates\n", triad_updates.load(), triad_updates.load()/(double)solved_count.load());
        if ( bug_count.load() ) {
            printf("%10ld  bi-value universal graves detected\n", bug_count.load());
        }
    } else if ( reporttimings ) {
        long long duration = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration(std::chrono::steady_clock::now() - starttime)).count();
        if ( duration/npuzzles > 1000 ) {
            printf("%8.1lfms  %6.2lf\u00b5s/puzzle  solving time\n", (double)duration/1000000, (double)duration/(npuzzles*1000LL));
        } else {
            printf("%8.1lfms  %4dns/puzzle  solving time\n", (double)duration/1000000, (int)((double)duration/npuzzles));
        }
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

#else

namespace Schoku {
	GridState *stack = 0;
    int line_counter = 0;
    bool initialized = false;
    long guess_state;
}

void initializeSchoku() {
    using namespace Schoku;

    bmi2_support = __builtin_cpu_supports("bmi2");
    pext_support = bmi2_support && !__builtin_cpu_is("znver2");
    stack = (GridState*) (~0x3fll & ((unsigned long long) malloc(sizeof(GridState)*GRIDSTATE_MAX+0x40)+0x40));
    stack->stackpointer = 0;
    initialized = true;
    guess_state = guesses.load();
}

// library / call interface for invoking the Schoku solver,
// using tdoku benchmark conventions.
// This unfortunately precludes using multiple threads...

extern "C"
size_t OtherSolverSchoku(const char *input, size_t limit, uint32_t /*unused_configuration*/,
                             char *solution, size_t *num_guesses) {

    using namespace Schoku;

    if ( !initialized ) {
        initializeSchoku();
    }
    rules = (limit <= 1)? FindOne:Multiple;
    line_counter++;
    Status status = solve<VNone>((signed char *)input, (signed char *)solution, stack, line_counter);
    long guesses_now = guesses.load();
    *num_guesses = guesses_now - guess_state;
    guess_state = guesses_now;
    if ( status.solved == false ) {
        return 0;
    }
    return status.unique?1:2;
}

#endif