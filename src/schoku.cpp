// This code uses AVX2 instructions...
/*
 * Schoku
 *
 * A high speed sudoku solver by M. Schulz
 *
 * Copyright 2024, 2025 Martin Schulz
 *
 * This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 * You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
 *
 * Based on the sudoku solver by Mirage ( https://codegolf.stackexchange.com/users/106606/mirage )
 * at https://codegolf.stackexchange.com/questions/190727/the-fastest-sudoku-solver
 * on Sep 22, 2021
 *
 * Version 0.9.6
 *
 * Performance changes:
 * One minor tweak to the triad based make_guess to avoid plain pairs
 *
 * The new naked set search is much faster and may turn out to be performance neutral.
 * A substantial increase in no-guess puzzles (17-clue) compensates for the effort.
 *
 * replaced atomic counters with a single value class/instance and
 * leveraged OMP reduction constructs for aggregating the results.
 * Much faster execution with -x option.
 *
 * Functional changes:
 * Only rows are searched at this point; it should be easy enough to add rows and boxes
 * but with lesser performance.
 *
 * An environment variable (SCHOKU_NO_GUESS_REPORT is now checked and if set produces
 * a simple list of all puzzles (with line number!) that required a guess. 
 *
 * Performance measurement and statistics:
 *
 * found sets are now only reported when they lead to eliminations of candidates.
 * The count of searches now corresponds to entire rows searched.
 *
 * provide stats for binary unoversal graves as detected well as binary universal graves
 * avoided.
 *
 * data: 17-clue sudoku (49151 puzzles)
 * CPU:  Ryzen 7 4700U
 *
 * schoku version: 0.9.6
 * command options: -x
 * compile options: OPT_SETS OPT_FSH OPT_UQR
 *      49151    17.0/puzzle  puzzles entered and presets
 *      49151  2519285/s  puzzles solved
 *     19.5ms   396ns/puzzle  solving time
 *     38596   78.53%  puzzles solved without guessing
 *     21618    0.44/puzzle  guesses
 *     13622    0.28/puzzle  back tracks
 *    169120    3.44/puzzle  digits entered and retracted
 *     22467    0.46/puzzle  'rounds'
 *    371660    7.56/puzzle  triads resolved
 *    963409   19.60/puzzle  triad updates
 *       711  bi-value universal graves avoided (BUG+1)
 *       138  bi-value universal graves detected
 *       134  board states without bivalues
 *
 * command options: -x -ms
 * compile options: OPT_SETS OPT_FSH OPT_UQR
 *      49151    17.0/puzzle  puzzles entered and presets
 *      49151  2526316/s  puzzles solved
 *     24.2ms   395ns/puzzle  solving time
 *      40497   82.39%  puzzles solved without guessing
 *      15028    0.31/puzzle  guesses
 *       8774    0.18/puzzle  back tracks
 *     109827    2.23/puzzle  digits entered and retracted
 *      22753    0.46/puzzle  'rounds'
 *     356921    7.26/puzzle  triads resolved
 *     944818   19.22/puzzle  triad updates
 *       6995    0.14/puzzle  naked sets found
 *      93227    1.90/puzzle  sections searched for naked sets
 *        816  bi-value universal graves avoided (BUG+1)
 *        169  bi-value universal graves detected
 *        91  board states without bivalues
 *
 */
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cctype>
#include <cstdarg>
#include <cassert>
#include <unistd.h>
#include <fcntl.h>
#include <omp.h>
#include "compat/x86_intrin.hpp"
#include <sys/stat.h>
#include <sys/mman.h>
#include <condition_variable>

namespace Schoku {

const char *version_string = "0.9.6";

const char *compilation_options =
#ifdef OPT_SETS
// Naked sets detection is a main feature.  The complement of naked sets are hidden sets,
// which are labeled as such when they are more concise to report.
//
"OPT_SETS "
#endif
#ifdef OPT_NEWSETS
// Naked sets detection is a main feature.  The complement of naked sets are hidden sets,
// which are labeled as such when they are more concise to report.
//
"OPT_NEWSETS "
#endif
#ifdef OPT_FSH
// Detection of fishes ( X-wing, sword fish, jellyfish and squirmbag) and
// finned/sashimi extensions are a main feature.
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
// USE_ROW_HIDDEN_SEARCH:
// if set to 1, provides code to search row hidden singles.
// if set to 0, triad processing will isolate hidden singles in their column.
// Turning off row search is a little faster
#ifndef USE_ROW_HIDDEN_SEARCH
#define USE_ROW_HIDDEN_SEARCH 0
#endif

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

    inline __uint128_t operator |= (const __uint128_t b) {
        return this->u128 = this->u128 | b;
    }

    inline __uint128_t operator ^ (const __uint128_t b) {
        return this->u128 ^ b;
    }

    inline __uint128_t operator ^= (const __uint128_t b) {
        return this->u128 = this->u128 ^ b;
    }

    inline __uint128_t operator & (const __uint128_t b) {
        return this->u128 & b;
    }

    inline __uint128_t operator &= (const __uint128_t b) {
        return this->u128 = this->u128 & b;
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
        // slightly faster than _bittestandset
        this->u8[idx>>3] |= 1<<(idx & 0x7);
    }
    inline void unset_indexbit(unsigned char idx) {
        _bittestandreset64((long long int *)&this->u64[idx>>6], idx & 0x3f);
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
    // need to use u8 to avoid aliasing error, but in reality, it's u16 we want,
    // or the u64 access would go beyond the data. Sigh.
    inline unsigned long long get_indexbits(unsigned char pos, unsigned char bitcount) {
        unsigned char posb = pos >> 4;
        pos &= 0xf;
        unsigned long long res = (*(unsigned long long*)(&u8[posb<<1])) >> pos;
        if ( pos+bitcount > 63 ) {
            res |= ((unsigned long long)u16[posb+4]) << (64 - pos);
        }
        // clip result
        return _bextr_u64(res, 0, bitcount);
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

const unsigned char group4x3offsets[12] = { box_offset[0], box_offset[1], box_offset[2], 0xff,
                                            box_offset[3], box_offset[4], box_offset[5], 0xff,
                                            box_offset[6], box_offset[7], box_offset[8], 0xff };

// SIMD/vector constants (centralized into solver/simd_constants.hpp).
#include "solver/simd_constants.hpp"

class
alignas(64)
Counters {
public:
    long long past_naked_count;             // how often do we get past the naked single serach
    long long digits_entered_and_retracted; // to measure guessing overhead
    long long triads_resolved;              // how many triads did we resolved
    long long triad_updates;                // how many triads did cancel candidates
    long long naked_sets_searched;          // how many naked sets did we search for
    long long naked_sets_found;             // how many naked sets did we actually find
    long long unique_rectangles_checked;   // how many unique rectangles were checked
    long long unique_rectangles_avoided;   // how many unique rectangles were avoided
    long long fishes_detected;             // how many fishes were identified
    long long fishes_excluded;             // how many square fishes were found and excluded
    long long fishes_specials_detected;    // how many special fish patterns were identified ((subset of fishes_detected)
    long long fishes_updated;              // how many fishes were updated
    long long fishes_specials_updated;     // how many special fish patterns were updated (subset of fishes_updated)
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
    this->naked_sets_searched += a.naked_sets_searched;
    this->naked_sets_found += a.naked_sets_found;  
    this->unique_rectangles_checked += a.unique_rectangles_checked;
    this->unique_rectangles_avoided += a.unique_rectangles_avoided;
    this->fishes_detected += a.fishes_detected;
    this->fishes_excluded += a.fishes_excluded;
    this->fishes_specials_detected += a.fishes_specials_detected;
    this->fishes_updated += a.fishes_updated;
    this->fishes_specials_updated += a.fishes_specials_updated;
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

#if defined(OPT_FSH)
    const char *fish_names[4] = { "X-wing-2", "swordfish-3", "jellyfish-4", "squirmbag-5" };
#endif

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

#endif

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
int report_guess_puzzles = 0; // display puzzles that did use guessing

// JSONL trace output (Phase 1: singles + guess + backtrack + puzzle boundaries)
// Set via --trace-out FILE. "-" means stdout. nullptr = trace disabled.
const char *trace_out_path = nullptr;
FILE * trace_out_fp = nullptr;

int guess_score_threshold      = 3; // could be tuned dynamically to 2 for guess-intensive test sets

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
bool mode_newsets=false;        // 'N', see OPT_NEWSETS
bool mode_uqr=false;            // 'U', see OPT_UQR
bool mode_fish=false;			// 'F', see OPT_FSH

signed char *output;

// isolate the strict-aliasing warning for casts from unsigned long long [2] arrays:
inline const __uint128_t *cast2cu128(const unsigned long long *from) {

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
	return (const __uint128_t *)from;
#pragma GCC diagnostic pop
}

// global debug aids
int dbgprintfilter = 0;       // global filter mask set from env, 
FILE * dbgprintout = stdout;

__attribute__((format(printf, 2, 3)))
int dbgprintf(int filter, const char *format...) {
    int ret = 0;
    if ( filter & dbgprintfilter ) {
        va_list args;
        va_start(args, format);
        ret = vfprintf(dbgprintout, format, args);
        va_end(args);
    }
    return ret;
}

#include "util/debug_dump.hpp"
#include "util/bit_simd.hpp"

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

#include "util/board_dump.hpp"

// capture the output file information, per puzzle, keeping it sequential one puzzle at a time
// includes memstream memory and length.
// By default, a pass-through to stdout.
// Only buffers when debug is not 0.
class MemStream {
		char *buf=0;
		size_t size=0;

		public:
		FILE * outf=stdout;

		~MemStream() {
			if ( buf != 0 && outf != stdout ) {
				printBuffer(stdout);
			}
		}

		inline FILE *getStream() {
			return outf;
		}

		void startBuffer() {
			outf = open_memstream(&buf, &size);
		}

		void printBuffer() {
			printBuffer(stdout);
		}

		void printBuffer(FILE *stream) {
            if ( buf != 0 && outf != stdout ) {
                fclose(outf);
            }
			if ( outf != stream ) {
				fwrite(buf, 1, size, stream);
			}
			closeBuffer();
		}

		void closeBuffer() {
			if ( buf != 0 && outf != stdout ) {
				free(buf);
				buf = 0;
				size = 0;
			}
			outf = stdout;
		}
};

// A fixed-size buffer that garantees reads in order, while writes can occur out of order.
// Loss of an element will deadlock the buffer.
// SequencingBuffer is not suitable for unbound sequences.
//
// A high frequency of write requests compared to reads will limit performance seriously.
// The buffer 'accepts' an element within the bounded interval [seqLo..seqLo+capacity-1],
// Internally the sequence number seq provided is mapped to seq%capacity.
// A read will always deliver the element at seqLo and increment seqLo.
// A read will enter a wait state if the element seqLo is not present.
// The data type T should best be a pointer, as the empty buffer element is checked
// for by comparing with 0.
// Writing a 0 element (not the sequence!) will be ignored.
// The type US must also be able to represent values all possible values of seq and last.
//
template<typename T, typename US, unsigned int cap>
class SequencingBuffer
{
private:
	std::mutex mut;
	std::array<T, cap> private_std_array {};
	std::condition_variable condNotEmpty;
	std::condition_variable condNotFull;
	US seqLo = 0;	// Guard with Mutex
	US last = 0;
	const unsigned int capacity = cap;
	bool closed = false;

	bool privateAccepts(US seq) {
		if ( (seq >= seqLo) && (seq < seqLo+capacity ) ) {
			return private_std_array[seq%capacity] == 0;
		}
		return false;
	}
public:
	SequencingBuffer() {}

	bool accepts(US seq) {
		if ( (seq >= seqLo) && (seq < seqLo+capacity ) ) {
			return !closed && private_std_array[seq%capacity] == 0;
		}
		return false;
	}
	void setLast(US last) {
		this->last = last;
	}
	void setFirst(US seq) {
		this->seqLo = seq;
	}

    bool put(T new_value, US seq)
    {
        std::unique_lock<std::mutex> lk(mut);
        //Condition takes a unique_lock and waits given the false condition
		condNotFull.wait(lk, [this,seq]{return privateAccepts(seq);});
		if ( closed ) {
			return false;
		}
       	private_std_array[seq%capacity] = new_value;
        condNotEmpty.notify_one();
		return true;
    }

    template<bool nonblock=false>
    bool take(T& value)
    {
        std::unique_lock<std::mutex> lk(mut);
		if ( closed ) {
			return false;
		}
        if ( nonblock ) {
            if ( privateAccepts(seqLo) ) {
                return false;
            }
        } else {
            //Condition takes a unique_lock and waits given the false condition
            condNotEmpty.wait(lk,[this]{return !privateAccepts(seqLo);});
        }
		if ( closed ) {
			return false;
		}
       	value=private_std_array[seqLo%capacity];
		private_std_array[seqLo%capacity] = 0;
		if ( seqLo == last ) {
			closed = true;
		}
		seqLo++;
       	condNotFull.notify_all();
		return true;
    }
    bool isAvailable(US seq) {
       	return private_std_array[seq%capacity] != 0;
	}
	bool isClosed() {
		return closed;
	}
};


class GridState;

// TriadInfo
//
class
alignas(64)
TriadInfo {
public:
    unsigned short row_triads[36];            //  27 triads, in groups of 9 with a gap of 1
    unsigned short col_triads[36];            //  27 triads, in groups of 9 with a gap of 1
    unsigned short row_triads_wo_musts[36];   //  triads minus tmusts for guessing
    unsigned short col_triads_wo_musts[36];   //  triads minus tmusts for guessing
    unsigned int triads_selection[2];
};

// Solver sharable data - used by some algorithms and make_guess
class SolverData {
private:
    bit128_t  bivalues;
    bit128_t candidate_bits_by_value[9];
    unsigned char sectionSetUnlocked[3][9];
public:
    FILE *output;
    Counters &counters;
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
// Note also that any cnt of 2 identifies a bi-local (i.e. strong link) for that row and digit.
//
public:
    bool bivaluesValid;
    bool cbbvsValid;
    bool sectionSetUnlockedValid[3];
    __m256i *boxes_as_cols;   // stashed in the next grid_state
    TriadInfo triadInfo;
    unsigned char  guess_hint_index;
    unsigned short guess_hint_digit;

    inline SolverData(Counters &counters, FILE *out): output(out), counters(counters) {}

    inline int printf ( const char * format, ... ) {
        va_list args;
        va_start (args, format);
        int ret = vfprintf (output, format, args);
        va_end (args);
        return ret;
    }

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
            bivalues.u64[1] = compress_epi16_boolean<false>(_mm256_and_si256(
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

    // get_boxes_as_cols returns a tranposition of the candidates such that
    // each box is represented as a column and each __m256i ('row') represents the same cell of the nine boxes.
    // This allows for example to count digits within a box, which in return allows
    // to identify hidden pairs within boxes.
    inline
    __m256i *get_boxes_as_cols(unsigned short *candidates, __m256i *p_boxes_as_cols) {
        if ( boxes_as_cols != 0 ) {
            return boxes_as_cols;
        }
        
        boxes_as_cols = p_boxes_as_cols;

// Consider this loop:
// from each box, pick the k-th cell:
//        unsigned short *candp = candidates;
//        for ( int k=0; k<9; candp += box_perp_ind_incr[k++]) {
//             boxes_as_cols[k] = _mm256_setr_epi16(candp[0], candp[3],candp[6],
//                                                  candp[27], candp[30],candp[33],
//                                                  candp[54], candp[57],candp[60], 0, 0, 0, 0, 0, 0, 0);
//        }
// Replace the above loop with the code below for efficiency:

        __m128i tmp_align2; // for aligning band 2 row
        __m128i band1, band2, band3; // to process a row from the given band
        __m256i tmp_row;     // to combine band1 and band3;
        __m256i tmp_align;  // for aligning tmp_row, i.e. band 1 and 3 rows
        unsigned short *inp=candidates;

        // process row 1 of each band
        //     [6,0], [6,3],  [6,6]
        band2 = *(__m128i_u*)(inp+27);
        tmp_align2 = *(__m128i_u*)(inp+27+8);
        band1 = *(__m128i_u*)inp;
        band3 = *(__m128i_u*)(inp+54);
        // first load the rows/cells that aren't laterally or vertically shifted (for now)
        //     [0,0], [0,3], [0,6] - [3,0], [3,3], [3,6]
        // --> [0,0], [0,3], [0,6] - [3,1], [3,4], [3,7]
        tmp_row = _mm256_set_m128i(band3, band1);
        tmp_align = _mm256_loadu2_m128i((__m128i*)(inp+54+8),(__m128i*)(inp+8));
        boxes_as_cols[0] = _mm256_set_m128i(band3,_mm_blend_epi16(band1, _mm_slli_si128(band2,2), 0x92));
        
        //     [0,1], [0,4], [0,7] - [3,1], [3,4], [3,7]
        // --> [1,0], [1,3], [1,6] - [4,1], [4,4], [4,7] (update to destination rows)
        boxes_as_cols[1] = _mm256_blend_epi16(_mm256_srli_si256(tmp_row,2), _mm256_zextsi128_si256(band2), 0x92);
        //     [0,2], [0,5], -[0,8] - [3,2], [3,5], -[3,8]
        // --> [2,0], [2,3], +[2,6] - [5,1], [5,4], +[5,7] (update to destination rows)
        boxes_as_cols[2] = _mm256_blend_epi16(_mm256_alignr_epi8(tmp_align,tmp_row,4),_mm256_zextsi128_si256(_mm_alignr_epi8(tmp_align2,band2,2)), 0x92);

        // process row 2 of each band
        band2 = *(__m128i_u*)(inp+27+9);
        tmp_align2 = *(__m128i_u*)(inp+27+9+8);
        band1 = *(__m128i_u*)(inp+9);
        band3 = *(__m128i_u*)(inp+54+9);
        //     [1,0], [1,3], [1,6] - [4,0], [4,3], [4,6]
        // --> [3,0], [3,3], [3,6] - [3,1], [3,4], [3,7] (update to destination rows)
        tmp_row = _mm256_set_m128i(band3, band1);
        tmp_align = _mm256_loadu2_m128i((__m128i*)(inp+54+9+8),(__m128i*)(inp+9+8));
        boxes_as_cols[3] = _mm256_set_m128i(band3, _mm_blend_epi16(band1, _mm_slli_si128(band2,2), 0x92));
        //     [1,1], [1,4], [1,7] - [4,1], [4,4], [4,7]
        // --> [4,0], [4,3], [4,6] - [4,1], [4,4], [4,7]
        boxes_as_cols[4] = _mm256_blend_epi16(_mm256_srli_si256(tmp_row,2), _mm256_zextsi128_si256(band2), 0x92);
        //     [1,2], [1,5], -[1,8] - [4,2], [4,5], -[4,8]
        // --> [5,0], [5,3],  [5,6] - [5,1], [5,4], +[5,7]
        boxes_as_cols[5] = _mm256_blend_epi16(_mm256_alignr_epi8(tmp_align,tmp_row,4),_mm256_zextsi128_si256(_mm_alignr_epi8(tmp_align2,band2,2)), 0x92);

        // process row 2 of each band
        band2 = *(__m128i_u*)(inp+27+18);
        tmp_align2 = *(__m128i_u*)(inp+27+18+8);
        band1 = *(__m128i_u*)(inp+18);
        band3 = *(__m128i_u*)(inp+54+18);
        //     [2,0], [2,3],  [2,6] - [5,0], [5,3],  [5,6]
        // --> [6,2], [6,5], +[6,8] - [6,1], [6,4],  [6,7] (update to destination rows)
        tmp_row = _mm256_set_m128i(band3, band1);
        tmp_align = _mm256_loadu2_m128i((__m128i*)(inp+54+18+8),(__m128i*)(inp+18+8));
        boxes_as_cols[6] = _mm256_set_m128i(band3, _mm_blend_epi16(band1, _mm_slli_si128(band2,2), 0x92));
        //     [2,1], [2,4],  [2,7] - [5,1], [5,4], [5,7]
        // --> [7,2], [7,5], +[8,8] - [7,1], [7,4], [7,7] (update to destination rows)
        boxes_as_cols[7] = _mm256_blend_epi16(_mm256_srli_si256(tmp_row,2), _mm256_zextsi128_si256(band2), 0x92);
        //     [2,2], [2,5], -[2,8] - [5,2], [5,5], -[5,8]
        // --> [8,0], [8,3], +[8,6] - [8,1], [8,4], +[8,7]
        boxes_as_cols[8] = _mm256_blend_epi16(_mm256_alignr_epi8(tmp_align,tmp_row,4),_mm256_zextsi128_si256(_mm_alignr_epi8(tmp_align2,band2,2)), 0x92);

        boxes_as_cols[0] = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(boxes_as_cols[0], shuf), shuf8x32);
        boxes_as_cols[1] = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(boxes_as_cols[1], shuf), shuf8x32);
        boxes_as_cols[2] = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(boxes_as_cols[2], shuf), shuf8x32);
        boxes_as_cols[3] = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(boxes_as_cols[3], shuf), shuf8x32);
        boxes_as_cols[4] = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(boxes_as_cols[4], shuf), shuf8x32);
        boxes_as_cols[5] = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(boxes_as_cols[5], shuf), shuf8x32);
        boxes_as_cols[6] = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(boxes_as_cols[6], shuf), shuf8x32);
        boxes_as_cols[7] = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(boxes_as_cols[7], shuf), shuf8x32);
        boxes_as_cols[8] = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(boxes_as_cols[8], shuf), shuf8x32);
        return boxes_as_cols;
    }

    template<Kind kind>
    inline unsigned char *getSectionSetUnlocked(GridState &gs);

};

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
// parameters:
// cbbv_v: the bits for the respective digit
// base: the base (column indices) of the fish found
// subs: the row indices for the grid
// hassubs_prp_x: the row indices that intersect the base columns
// clean_bits: all bits of digits to be cleaned
// dgt_bits: all bits for the respective digit (same but different presentation as cbbv_v)
//
inline void show_fish(cbbv_t &cbbv, unsigned short base, unsigned short subs, unsigned short hassubs, bit128_t &clean_bits, bit128_t &dgt_bits, SolverData &solverData, const char *msg="") {
    solverData.printf("%s:\n", msg);
    for ( int i_=0; i_<81; i_++ ) {
        int ti = transposed?transposed_cell[i_]:i_;
        unsigned int ti_bit = 1<<(ti%9);
        unsigned int ti_bitv = 1<<(ti/9);
        if ( (ti_bitv & hassubs) && clean_bits.check_indexbit(i_)) {
            solverData.printf("X");
        } else if ( ti_bitv & subs ) {
            solverData.printf("%s", (cbbv.v16[ti/9]&ti_bit)?((ti_bit&base)?"o":"@"):"-");
        } else {
            solverData.printf(dgt_bits.check_indexbit(i_)? ".":"-");
        }
        if ( i_%9 == 8 ) {
            solverData.printf("\n");
        } else if ( i_%3 == 2 ) {
            solverData.printf(" ");
        }
    }
}

template <bool transposed=false>
// parameters:
// cbbv_v: the bits for the respective digit
// subs: the row indices for the grid
// fins/fincnt: the fins and their number
// clean_bits: all bits of digits to be cleaned
// dgt_bits: all bits for the respective digit (same but different presentation as cbbv_v)
//
inline void show_fish(cbbv_t &cbbv, unsigned short subs, unsigned char fincnt, unsigned char *fins, bit128_t &clean_bits, bit128_t &dgt_bits, SolverData &solverData, const char *msg="") {
    solverData.printf("%s:\n", msg);
    for ( int i_=0; i_<81; i_++ ) {
        int ti = transposed?transposed_cell[i_]:i_;
        bool found = false;
        for ( int k=0; k<fincnt; k++ ) {
            if ( i_ == fins[k] ) {
                found = true;
                solverData.printf("@");
                break;
            }
        }
        if ( !found ) {
            unsigned int ti_bit = 1<<(ti%9);
            unsigned int ti_bitv = 1<<(ti/9);
            if ( clean_bits.check_indexbit(i_) ) {
                solverData.printf("X");
            } else if ( ti_bitv & subs ) {
                solverData.printf("%s", (cbbv.v16[ti/9]&ti_bit)?"o":"-");
            } else {
                solverData.printf(dgt_bits.check_indexbit(i_)? ".":"-");
            }
        }
        if ( i_%9 == 8 ) {
            solverData.printf("\n");
        } else if ( i_%3 == 2 ) {
            solverData.printf(" ");
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
    int flags;                        // - if negative as indicator for multiple solutions in this grid_state
                                      // - low 27 bits to indicate sets of size 4 in 27 sections
                                      // aligned on 16 bytes
    bit128_t unlocked;                // for keeping track of which cells still need to be resolved. Set bits correspond to cells that still have multiple possibilities
    bit128_t updated;                 // for keeping track of which cell's candidates may have been changed since last time we looked for naked sets. Set bits correspond to changed candidates in these cells
    bit128_t set23_found[3];          // for keeping track of found sets of size 2 and 3

// GridState is normally copied for recursion
// initialize the starting state including the puzzle.
//
template<Verbosity verbose>
inline void initialize(signed char grid[81], Counters &counters) {
    unlocked.u64[1] = 0;
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

    if ( grid[80] <= '0' ) {
        unlocked.u16[5] = 1;
    }
    updated = unlocked;

    bit128_t digit_bits[9] {};

    bit128_t locked = { .u128 = ~unlocked.u128 };
    locked.u64[1] &= 0x1ffff;
    if ( verbose != VNone ) {
        counters.preset_count += locked.popcount();
    }

    // Grouping the updates by digit beats other methods for number of clues >17.
    // calculate the masks for each digit from the place and value of the clues:
    unsigned char off = 0;
    for ( unsigned int i=0; i<2; i++) {
        unsigned long long lkd = locked.u64[i];
        while (lkd) {
            int dix = tzcnt_and_mask(lkd)+off;
            int dgt = grid[dix] - 49;
            digit_bits[dgt].u128 = digit_bits[dgt].u128 | *cast2cu128(big_index_lut[dix][All]);
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
        // digit 9
        // load 32 digit bits:
        bits1_8 = _mm256_or_si256(bits1_8, bits1_8_2);
        __m256i bits9 = _mm256_and_si256(expand_bitvector_epi8<true>(digit_bits[8].u32[i>>5]), ones_epi8);
        *(__m256i*)&candidates[i] = _mm256_andnot_si256(_mm256_unpacklo_epi8(bits1_8,bits9), mask1ff);
        __m256i c2 = _mm256_andnot_si256(_mm256_unpackhi_epi8(bits1_8,bits9), mask1ff);

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
inline __attribute__((always_inline)) void enter_digit( unsigned short digit, unsigned char i, FILE *output) {
    // lock this cell and and remove this digit from the candidates in this row, column and box

    bit128_t to_update;
    if ( verbose == VDebug ) {
        fprintf(output, " %x at %s\n", _tzcnt_u32(digit)+1, cl2txt[i]);
    }
#ifndef NDEBUG
    if ( __popcnt16(digit) != 1 && warnings != 0 ) {
        fprintf(output, "error in enter_digit: %x\n", digit);
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
// The four make_guess overloads are forward-declared here; their bodies
// live in solver/make_guess.hpp and are #included right after this class
// closes. Splitting the bodies out keeps the GridState class body
// readable while preserving inlining and template instantiation.
template<Verbosity verbose, typename F>
inline GridState* make_guess(unsigned char cell_index, F &&gridUpdater, Counters &counters, FILE *output);
template<Verbosity verbose>
inline GridState* make_guess(SolverData *solverData);
template<Verbosity verbose>
inline GridState* make_guess(SolverData &solverData);
template<Verbosity verbose>
inline GridState* make_guess(unsigned char guess_index, unsigned short digit, Counters &counters, FILE *output);

};

#include "solver/trace.hpp"

// Thread-local trace emitter pointer. Defined here (translation-unit scope)
// rather than in the header so the .hpp stays header-only-clean. Each OpenMP
// worker initialises this to its own Emitter when trace_out_fp is non-null,
// and back to nullptr otherwise. The if (Schoku::trace::current) guard at every
// emission site folds to a single TLS load + predicted-cold branch when
// tracing is disabled.
namespace trace {
    thread_local Emitter* current = nullptr;
    thread_local uint8_t next_entry_reason = ER_None;
}

#include "solver/make_guess.hpp"

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


#include "solver/solve.hpp"

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
    -m[FSNU]* execution modes (fishes, sets/new sets, unique rectangles), lower or upper case
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
    -#1 change base for row and column reporting from 0 to 1

    fishes details can be shown as a 9x9 grid with '-d3' where 
       'o' represents fish positions as 'o'
       '@' represents fin(s) if present
       'X' shows any candidates to eliminate
       '.' shows other candidates containing the same digit

)");
}

int main(int argc, const char *argv[]) {
using namespace Schoku;

    int line_to_solve = 0;

    // the debug dbgprintf and dbgprintfilter are not used in checked-in code
    // they are initialized here at no cost just in case...
    const char *schoku_dbg_filter = getenv("SCHOKU_DBG_FILTER");
    if ( schoku_dbg_filter ) {
       unsigned off = 0;
       while ( schoku_dbg_filter[off] == '0' ) {
           off++;
       }
       if ( schoku_dbg_filter[off] == 'x' ) {
            sscanf(&schoku_dbg_filter[off+1], "%x", &dbgprintfilter);
       } else {
            sscanf(&schoku_dbg_filter[off], "%d", &dbgprintfilter);
       }
    }
    const char *schoku_report_guess = getenv("SCHOKU_GUESS_REPORT");
    if ( schoku_report_guess ) {
       if ( schoku_report_guess[0] == '1' ) {
            report_guess_puzzles = 1;
       }
    }

    if ( argc > 0 ) {
        argc--;
        argv++;
    }

    // Buffer holds the reflected command line for "-x" stats output (may
    // be silently truncated for very long argv; truncation is harmless to
    // solver behavior). Original was [80] which overflows on realistic
    // absolute paths and triggers SIGABRT under fortified libcs (Apple
    // clang/macOS). Sized to accommodate typical CLI plus two long file
    // paths; snprintf bounds the writes regardless of argv length.
    char opts[1024] = { 0 };
    size_t used = 0;
    for (int i = 0; i < argc; i++) {
        if (used >= sizeof(opts) - 1) break;
        int n = snprintf(opts + used, sizeof(opts) - used, "%s ", argv[i]);
        if (n < 0) break;
        used += (size_t)n;
        if (used >= sizeof(opts) - 1) { opts[sizeof(opts) - 1] = 0; break; }
    }

    while ( argc && argv[0][0] == '-' ) {
        if (argv[0][1] == 0) {
            argv++; argc--;
            break;
        }
        // Long options. A bare "--" terminates option parsing (POSIX
        // convention), so positional args beginning with "-" remain
        // reachable. --trace-out FILE | --trace-out=FILE: FILE = "-"
        // means stdout. The file is opened in append mode so a single
        // multi-thread Schoku run can extend an existing trace; cross-
        // process appends to the same file are NOT guaranteed atomic at
        // line granularity (libc FILE locking is per-FILE-object).
        if (argv[0][1] == '-') {
            if (argv[0][2] == 0) {     // bare "--"
                argv++; argc--;
                break;
            }
            if (strcmp(argv[0], "--trace-out") == 0) {
                if (argc < 2) {
                    fprintf(stderr, "--trace-out requires a path argument\n");
                    exit(1);
                }
                trace_out_path = argv[1];
                argc -= 2; argv += 2;
                continue;
            }
            if (strncmp(argv[0], "--trace-out=", 12) == 0) {
                trace_out_path = argv[0] + 12;
                argc--; argv++;
                continue;
            }
            fprintf(stderr, "invalid long option: %s\n", argv[0]);
            argc--; argv++;
            continue;
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
#ifdef OPT_NEWSETS
                case 'N':        // see OPT_NEWSETS
                    mode_newsets = true;
                    break;
#endif
#ifdef OPT_SETS
                case 'S':        // see OPT_SETS
                    mode_sets = true;
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
            printf("uqr checking mode ( -mU ) disabled when not under default Regular rules\n");
        }
        mode_uqr = false;
    }

    // Open the JSONL trace sink up-front. "-" means stdout; any other path
    // is opened for append. Atomicity guarantee: a per-puzzle fwrite() is
    // line-coherent across THREADS sharing one FILE object (libc internal
    // FILE locking), NOT across processes appending to the same path —
    // those can still interleave at sub-fwrite granularity.
    if ( trace_out_path != nullptr ) {
        if ( strcmp(trace_out_path, "-") == 0 ) {
            // Sending trace JSONL to stdout collides with Schoku's stats
            // / timing / debug summary output which also targets stdout.
            // Refuse the combination rather than silently corrupting the
            // JSONL stream for the consumer.
            if ( reportstats || reporttimings || debug ) {
                fprintf(stderr, "--trace-out=- (stdout) is incompatible with "
                                "-x / -y / -d; redirect trace to a file or "
                                "drop the stats/debug flag.\n");
                exit(1);
            }
            trace_out_fp = stdout;
        } else {
            trace_out_fp = fopen(trace_out_path, "ab");
            if ( trace_out_fp == nullptr ) {
                fprintf(stderr, "Failed to open --trace-out=%s: %s\n",
                        trace_out_path, strerror(errno));
                exit(1);
            }
        }
    }

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
    // Single source of truth for the output mapping size — used by
    // ftruncate, mmap, and munmap below. Splitting these used to allow
    // a npuzzles/outnpuzzles mismatch under -l# (see git log for fix).
    const size_t output_bytes = outnpuzzles * 164;

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
    if ( ftruncate(fdout, output_bytes) == -1 ) {
		if (errno ) {
			fprintf(stderr, "Error: setting size (ftruncate) on output file %s: %s\n", ofn, strerror(errno));
		}
		exit(0);
	}

	// map the output file
    output = (signed char *)mmap((void*)0, output_bytes, PROT_WRITE, MAP_SHARED, fdout, 0);
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
    MemStream *memstream = 0;
    SequencingBuffer<MemStream *, unsigned int, 24> seqBuf;

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

        // Single-puzzle path: instantiate one Emitter on the stack so the
        // trace-disabled cost is exactly the if (trace_out_fp) test below.
        Schoku::trace::Emitter* trace_em = nullptr;
        long long g0 = 0, t0 = 0;
        if ( trace_out_fp ) {
            trace_em = new Schoku::trace::Emitter(trace_out_fp, 0);
            Schoku::trace::current = trace_em;
            g0 = global_counters.guesses;
            t0 = global_counters.trackbacks;
            trace_em->puzzle_start((const char*)&string_pre[i], (int)line_to_solve);
        }

        Status s;
        if ( debug != 0) {
            stack[0].initialize<VDebug>(output, global_counters);
            s = solve<VDebug>(grid, stack, line_to_solve, global_counters);
        } else if ( reportstats !=0) {
            stack[0].initialize<VStats>(output, global_counters);
            s = solve<VStats>(grid, stack, line_to_solve, global_counters);
        } else {
            stack[0].initialize<VNone>(output, global_counters);
            s = solve<VNone>(grid, stack, line_to_solve, global_counters);
        }

        if ( trace_em ) {
            trace_em->puzzle_end(s.solved, s.solved ? (const char*)grid : nullptr,
                                 global_counters.guesses - g0,
                                 global_counters.trackbacks - t0);
            Schoku::trace::current = nullptr;
            delete trace_em;
        }
    } else {
        seqBuf.setLast((imax-1)/(120*82));
        // The OMP directives:
        // proc_bind(close): high preferance for thread/core affinity
        // firstprivate(stack): the stack is allocated for each thread once and seperately
        // schedule(dynamic,64): 64 puzzles are allocated at a time and these chunks
        //   are assigned dynmically (to minimize random effects of difficult puzzles)
        // shared(...) lists the variables that are shared (as opposed to separate copies per thread)
        // reduction(...) allows to aggregate results from all threads using a single declaration of the aggregate
        // declare reduction defines a reduction constructs (name, data type, operation)

// initializer is required to zero-init per-thread copies; without it clang
// leaves them indeterminate (parity bug observed on clang/Linux x86_64 and
// Apple clang/macOS aarch64). gcc happened to zero-init via calloc by chance.
#pragma omp declare reduction (counters_reduction : Counters : omp_out += omp_in) \
    initializer(omp_priv = Counters())

#pragma omp parallel reduction(counters_reduction:global_counters) firstprivate(stack, memstream) proc_bind(close) shared(string_pre, output, npuzzles, imax, debug, reportstats, numthreads, trace_out_fp)
        {
            if ( numthreads == 0 ) {
                numthreads = omp_get_num_threads();
            }

            // force alignment the 'old-fashioned' way
            // not going to free the data ever
            // stack = (GridState*)malloc(sizeof(GridState)*GRIDSTATE_MAX);
            stack = (GridState*) (~0x3fll & ((unsigned long long) malloc(sizeof(GridState)*GRIDSTATE_MAX+0x40)+0x40));

            memstream = new MemStream();
            if ( debug && (numthreads > 1) ) {
                memstream->startBuffer();
            }

            // Per-thread trace emitter: one buffer per worker, shared FILE
            // sink, flush at every puzzle_end. nullptr (default) when
            // tracing disabled — branch-predictably skipped at every site.
            Schoku::trace::Emitter* trace_em = nullptr;
            if ( trace_out_fp ) {
                trace_em = new Schoku::trace::Emitter(trace_out_fp, omp_get_thread_num());
                Schoku::trace::current = trace_em;
            }
#pragma omp for schedule(monotonic:dynamic,120)
            for (size_t i = 0; i < imax; i+=82) {
                // copy unsolved grid
                signed char *grid = &output[i*2+82];
                memcpy(&output[i*2], &string_pre[i], 81);
                // add comma and newline in right place
                output[i*2 + 81] = ',';
                output[i*2 + 163] = 10;

                // Per-puzzle trace wrapper. The OMP reduction gives every
                // thread a private cumulative `global_counters`; per-puzzle
                // delta = (after - before) snapshot.
                long long g0 = 0, t0 = 0;
                if ( trace_em ) {
                    g0 = global_counters.guesses;
                    t0 = global_counters.trackbacks;
                    trace_em->puzzle_start((const char*)&string_pre[i],
                                           (int)(i/82 + 1));
                }

                Status s;
                if ( debug != 0) {
                    stack[0].initialize<VDebug>(&output[i*2], global_counters);
                    s = solve<VDebug>(grid, stack, i/82+1, global_counters, memstream->outf);
                } else if ( reportstats !=0) {
                    stack[0].initialize<VStats>(&output[i*2], global_counters);
                    s = solve<VStats>(grid, stack, i/82+1, global_counters, memstream->outf);
                } else {
                    stack[0].initialize<VNone>(&output[i*2], global_counters);
                    s = solve<VNone>(grid, stack, i/82+1, global_counters, memstream->outf);
                }

                if ( trace_em ) {
                    trace_em->puzzle_end(s.solved,
                                         s.solved ? (const char*)grid : nullptr,
                                         global_counters.guesses - g0,
                                         global_counters.trackbacks - t0);
                }
                // strictly align this with the chunk size of 120
                if ( debug && (numthreads > 1) && ((i/82)%120 == 119 || (i==imax-82) )) {
                    seqBuf.put(memstream, i/(82*120));
                    memstream = new MemStream();
                    memstream->startBuffer();
                    if ( omp_get_thread_num() == 0 ) {   // one thread to manage
                        MemStream *next = 0;
                        while(seqBuf.take<true>(next) && next) {
                            next->printBuffer();
                            delete next;
                        }
                    }
                }
            } // omp for

            if ( trace_em ) {
                Schoku::trace::current = nullptr;
                delete trace_em;
                trace_em = nullptr;
            }

            if ( debug && (numthreads > 1) ) {
                if ( omp_get_thread_num() == 0 ) {   // one thread to manage
                    MemStream *next = 0;
                    while ( !seqBuf.isClosed() && seqBuf.take(next) && next) {
                         next->printBuffer();
                         delete next;
                    }
                }
            }
        } // omp parallel
    } // if

	int err = munmap(string, fsize);
	if ( err == -1 ) {
		if (errno ) {
			fprintf(stderr, "Error: munmap file %s: %s\n", ifn, strerror(errno));
		}
	}
	// Pass the exact length used by mmap. POSIX permits a larger len here
	// (it unmaps every page in [output, output+len)), but the extra range
	// can fall on pages later allocated by libc/libomp/etc. and tear them
	// down — observed as SIGSEGV under -l# on both gcc/Linux and
	// clang/macOS, where outnpuzzles=1 but the original code passed
	// npuzzles*164.
	err = munmap(output, output_bytes);
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
            printf("%8.1lfms  %6.2lf\u00b5s/puzzle  solving time\n", (double)duration/(double)1000000, (double)duration/(outnpuzzles*1000LL));
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
#if defined(OPT_SETS) || defined(OPT_NEWSETS)
        if ( mode_sets || mode_newsets ) {
            printf("%10lld  %6.2f/puzzle  naked sets found\n", global_counters.naked_sets_found, (double)global_counters.naked_sets_found/global_counters.solved_count);
            printf("%10lld  %6.2f/puzzle  naked sets searched\n", global_counters.naked_sets_searched, (double)global_counters.naked_sets_searched/global_counters.solved_count);
        }
#endif
#ifdef OPT_FSH
        if ( mode_fish ) {
            printf("%10lld  %6.2f/puzzle  fishes updated\n", global_counters.fishes_updated, (double)global_counters.fishes_updated/global_counters.solved_count);
            printf("%10lld  %6.2f/puzzle  fishes detected\n", global_counters.fishes_detected, (double)global_counters.fishes_detected/global_counters.solved_count);
         }
#endif
#ifdef OPT_UQR
        if ( mode_uqr ) {
            printf("%10lld  %6.2f/puzzle  unique rectangles avoided\n", global_counters.unique_rectangles_avoided, (double)global_counters.unique_rectangles_avoided/global_counters.solved_count);
            printf("%10lld  %6.2f/puzzle  unique rectangles checked\n", global_counters.unique_rectangles_checked, (double)global_counters.unique_rectangles_checked/global_counters.solved_count);
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

    if ( trace_out_fp != nullptr && trace_out_fp != stdout ) {
        fclose(trace_out_fp);
        trace_out_fp = nullptr;
    }

    return 0;
}
#endif
