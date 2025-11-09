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
 * Version 0.9.5
 *
 * Performance changes:
 * make_guess was updated to provide scoring for bivalue and trivalue searches.
 * This reduces guesses significantly for hard puzzles and improves speed.
 * replaced atomic statistics variables by a Counters class and created a reduction
 * clause for OMP to aggregate the results which reduced the substantial overhead to a few
 * nano seconds.
 *
 * Functional changes:
 * OPT_TRIAD_RES is gone! This functionality is now the default.
 * The hidden single search for boxes is gone(!) and for rows it is optional.
 *
 * Produce a readable hint with -w in case the puzzle file may have
 * puzzles with multiple solutions counted as unsolvable.
 * Debug trace output is now running multithreaded as well!
 * major changes to guess that involve bivalues and trivalues
 *
 * Performance measurement and statistics:
 * change to the meaning of 'round' in statistics:
 *    a round is counted every time the algorithm passes all trivial parts of the algorithm:
 *    naked and hidden singles and triads (aka 'pointing/claiming' locked candidates).
 * new statistic details:
 * + average preset cells, average timing is now given in ns, if < 1000.
 * + count of board states that have no bivalue cells
 * + separate counts for encountered bi-value universal graves versus the avoided bi-value universal graves
 * Good speed, improved stats for 17-clue sudoku
 *
 * data: 17-clue sudoku (49151 puzzles)
 * CPU:  Ryzen 7 4700U
 *
 * schoku version: 0.9.5
 * command options: -x
 * compile options: OPT_SETS OPT_FSH
 *      49151    17.0/puzzle  puzzles entered and presets
 *      49151  2531352/s  puzzles solved
 *     19.4ms   395ns/puzzle  solving time
 *      38596   78.53%  puzzles solved without guessing
 *      21548    0.44/puzzle  guesses
 *      13497    0.27/puzzle  back tracks
 *     167845    3.41/puzzle  digits entered and retracted
 *      22431    0.46/puzzle  'rounds'
 *     371352    7.56/puzzle  triads resolved
 *     962785   19.49/puzzle  triad updates
 *        709  bi-value universal graves avoided (BUG+1)
 *        138  bi-value universal graves detected
 *        134  board states without bivalues
 *
 * command options: -x -mfs
 * compile options: OPT_SETS OPT_FSH
 *      49151    17.0/puzzle  puzzles entered and presets
 *      49151  2224450/s  puzzles solved
 *     22.1ms   449ns/puzzle  solving time
 *      45485   92.54%  puzzles solved without guessing
 *       5088    0.10/puzzle  guesses
 *       2713    0.06/puzzle  back tracks
 *      35638    0.73/puzzle  digits entered and retracted
 *      25359    0.52/puzzle  'rounds'
 *     335807    6.83/puzzle  triads resolved
 *     920087   18.72/puzzle  triad updates
 *      15420    0.31/puzzle  naked sets found
 *     494918   10.07/puzzle  naked sets searched
 *       7822    0.16/puzzle  fishes updated
 *     100847    2.26/puzzle  fishes detected
 *       1003  bi-value universal graves avoided (BUG+1)
 *         55  bi-value universal graves detected
 *         11  board states without bivalues
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
#include <stdbool.h>
#include <intrin.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <condition_variable>

namespace Schoku {

const char *version_string = "0.9.5";

const char *compilation_options =
#ifdef OPT_SETS
// Naked sets detection is a main feature.  The complement of naked sets are hidden sets,
// which are labeled as such when they are more concise to report.
//
"OPT_SETS "
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
// used in make_guess
const __m256i shuf = _mm256_setr_epi8(0,1,6,7,12,13,2,3,8,9,14,15,4,5,10,11,0,1,6,7,12,13,2,3,8,9,14,15,4,5,10,11);
const __m256i shuf8x32 = _mm256_setr_epi32(0,1,2,4,5,6,7,3);

// used with fishes
const __m256i mask_1ff = _mm256_setr_epi16(0x1ff, 0x1ff<<1, 0x1ff<<2, 0x1ff<<3, 0x1ff<<4, 0x1ff<<5, 0x1ff<<6, 0x1ff<<7,
                                           0x1ff, 0x1ff<<1, 0x1ff<<2, 0x1ff<<3, 0x1ff<<4, 0x1ff<<5, 0x1ff<<6, 0x1ff<<7);
const __m256i hrs_shifts = _mm256_setr_epi16(0x7fff, 0x8000>>1, 0x8000>>2, 0x8000>>3, 0x8000>>4, 0x8000>>5, 0x8000>>6, (unsigned)0x8000>>7,
                                             0x7fff, 0x8000>>1, 0x8000>>2, 0x8000>>3, 0x8000>>4, 0x8000>>5, 0x8000>>6, (unsigned)0x8000>>7);

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
inline Counters &operator += (Counters &a) {
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

template<bool little_endian=true>
inline void dump_m256i(__m256i x, const char *msg="", int filter=-1) {
    if ( little_endian ) {
        dbgprintf(filter, "%s %llx,%llx,%llx,%llx\n", msg, ((__v4du)x)[0],((__v4du)x)[1],((__v4du)x)[2],((__v4du)x)[3]);
    } else {
        dbgprintf(filter, "%s %llx,%llx,%llx,%llx\n", msg, ((__v4du)x)[3],((__v4du)x)[2],((__v4du)x)[1],((__v4du)x)[0]);
    }
}

inline void dump_m256i_epi16(__m256i x, const char *msg="", int filter=-1) {
    dbgprintf(filter, "%s %x,%x,%x,%x,%x,%x,%x,%x,%x,%x,%x,%x,%x,%x,%x,%x\n", msg, ((__v16hu)x)[0],((__v16hu)x)[1],((__v16hu)x)[2],((__v16hu)x)[3],((__v16hu)x)[4],((__v16hu)x)[5],((__v16hu)x)[6],((__v16hu)x)[7], ((__v16hu)x)[8],((__v16hu)x)[9],((__v16hu)x)[10],((__v16hu)x)[11],((__v16hu)x)[12],((__v16hu)x)[13],((__v16hu)x)[14],((__v16hu)x)[15]);
}

inline void dump_m128i_epi16(__m128i x, const char *msg="", int filter=-1) {
    dbgprintf(filter, "%s %x,%x,%x,%x,%x,%x,%x,%x\n", msg, ((__v8hu)x)[0],((__v8hu)x)[1],((__v8hu)x)[2],((__v8hu)x)[3],((__v8hu)x)[4],((__v8hu)x)[5],((__v8hu)x)[6],((__v8hu)x)[7]);
}

inline void dump_m128i(__m128i x, const char *msg="", int filter=-1) {
    dbgprintf(filter, "%s %llx,%llx\n", msg, ((__v2du)x)[0],((__v2du)x)[1]);
}

// a helper function to print a grid of bits.
// the bits are arranged as 9 bits each in 9 unsigned short elements of a __m256i parameter.
//
inline void dump_m256i_grid(__m256i v, const char *msg="", int filter=-1) {
    if ( !(filter & dbgprintfilter) ) {
        return;
    }
    dbgprintf(filter, "%s\n", msg);
    for (unsigned char r=0; r<9; r++) {
        unsigned short b = ((v16us)v)[r];
        for ( unsigned char i=0; i<9; i++) {
            dbgprintf(filter, "%s", (b&(1<<i))? "x":"-");
            if ( i%3 == 2 ) {
                dbgprintf(filter, " ");
            }
            if ( i%9 == 8 ) {
                dbgprintf(filter, "\n");
            }
        }
    }
}

// a helper function to print a grid of bits.
// bits are arranged consecutively as 9x9 bits in a __unint128_t parameter.
//
template<int split=0>
inline void dump_bits(__uint128_t bits, const char *msg="", int filter=-1) {
    if ( !(filter & dbgprintfilter) ) {
        return;
    }
    unsigned char lim = split==0?81:split;
    dbgprintf(filter, "%s:\n", msg);
    for ( int i_=0; i_<lim; i_++ ) {
        dbgprintf(filter, "%s", (*(bit128_t*)&bits).check_indexbit(i_)?"x":"-");
        if ( i_%3 == 2 ) {
            dbgprintf(filter, " ");
        }
        if ( i_%9 == 8 ) {
            dbgprintf(filter, "\n");
        }
    }
    if ( split ) {
        lim = 64+81-split;
        for ( int i_=64; i_<lim; i_++ ) {
            dbgprintf(filter, "%s", (*(bit128_t*)&bits).check_indexbit(i_)?"x":"-");
            if ( (i_-64)%3 == 2 ) {
                dbgprintf(filter, " ");
            }
            if ( (i_-64)%9 == 8 ) {
                dbgprintf(filter, "\n");
            }
        }
    }
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
template<bool size16=true>
inline __m256i get_first_lsb(__m256i vec) {
       if ( size16 ) {
            // isolate the lsb
            return _mm256_and_si256(vec, _mm256_sub_epi16(_mm256_setzero_si256(), vec));
       }
       return _mm256_and_si256(vec, _mm256_sub_epi8(_mm256_setzero_si256(), vec));
}

// compute vec &= ~lsb; return vec & -vec
template<bool size16=true>
inline __m256i andnot_get_next_lsb(__m256i lsb, __m256i &vec) {
        // remove prior lsb
        vec = _mm256_andnot_si256(lsb, vec);
        return get_first_lsb<size16>(vec);
}

// uncompress first eight cells from contiguous bits to one unsigned short per cell
// The input consists of 8 times 9 bit cell (9 bytes) and each cell must therefore be isolated
// and then shifted according to its position.
// The last mask operation isolates the last cell with bits in the highest 9 positions,
// leading to a possibly negative results from mulhrs, requiring the last mask operation.  

// [ beats manual loading hands down ]
inline __m128i uncompress(__m128i in) {
        __m128i out =  _mm_and_si128(_mm256_castsi256_si128(mask_1ff),_mm_unpacklo_epi16(in, _mm_bsrli_si128(in, 1)));
        return _mm_and_si128(_mm256_castsi256_si128(mask1ff), _mm_mulhrs_epi16(out, _mm256_castsi256_si128(hrs_shifts)));
}

// uncompress first eight cells from contiguous bits to one unsigned short per cell
// for two inputs in hi and lo lane
inline __m256i uncompress(__m256i in) {
        __m256i out =  _mm256_and_si256(mask_1ff,_mm256_unpacklo_epi16(in, _mm256_bsrli_epi128(in, 1)));
        return _mm256_and_si256(mask1ff, _mm256_mulhrs_epi16(out, hrs_shifts));
}

inline void format_candidate_set(char *ret, unsigned short candidates);

// a helper function to print a sudoku board,
// given the 81 cells solved or with candidates.
//
inline void dump_board(unsigned short *candidates, const char *msg="", int filter=-1) {
    if ( !(filter & dbgprintfilter) ) {
        return;
    }
    dbgprintf(filter,"%s:\n", msg);
    for ( int i_=0; i_<81; i_++ ) {
        char ret[32];
        format_candidate_set(ret, candidates[i_]);
        dbgprintf(filter,"%8s,", ret);
        if ( i_%9 == 8 ) {
            dbgprintf(filter,"\n");
        }
    }
}

// a helper function to print a sudoku puzzle from the solved cells,
// given the 81 cells.
//
template <bool transpose = false>
inline void dump_puzzle(unsigned short *candidates, bit128_t &unlocked, const char *msg="", int filter=-1) {
    if ( filter & dbgprintfilter ) {
        char gridout[82];
        for (unsigned char j = 0; j < 81; ++j) {
            unsigned short t = transpose? transposed_cell[j] : j;
            if ( unlocked.check_indexbit(t) ) {
                gridout[j] = '0';
            } else {
                gridout[j] = 49+_tzcnt_u32(candidates[t]);
            }
        }
        dbgprintf(filter, "%s %.81s\n", msg, gridout);
    }
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
// clean_bits: all bits of digits to be cleaned
// dgt_bits: all bits for the respective digit (same but different presentation as cbbv_v)
//
inline void show_fish(cbbv_t &cbbv, unsigned short base, unsigned short subs, bit128_t &clean_bits, bit128_t &dgt_bits, SolverData &solverData, const char *msg="") {
    solverData.printf("%s:\n", msg);
    for ( int i_=0; i_<81; i_++ ) {
        int ti = transposed?transposed_cell[i_]:i_;
        unsigned int ti_bit = 1<<(ti%9);
        unsigned int ti_bitv = 1<<(ti/9);
        if ( clean_bits.check_indexbit(i_)) {
            solverData.printf("X");
        } else if ( ti_bitv & subs ) {
            solverData.printf("%s", (cbbv.v16[ti/9]&ti_bit)?(ti_bit&base)?"o":"@":"-");
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

// parameters:
// celli/celli2: the floor cell indices
// roof1/roof2: the roof cell indices
// clean_bits: all bits of digits to be cleaned
// dgt_bits: all bits for the respective digit
//
inline void show_skyscraper(unsigned char floor1, unsigned char floor2, unsigned char roof1, unsigned char roof2, bit128_t &clean_bits, bit128_t &dgt_bits, SolverData &solverData, const char *msg="") {
    solverData.printf("%s:\n", msg);
    for ( int i_=0; i_<81; i_++ ) {
        if ( clean_bits.check_indexbit(i_)) {
            solverData.printf("X");
        } else if ( i_ == floor1 || i_ == floor2 ) {
            solverData.printf("o");
        } else if ( i_ == roof1 || i_ == roof2 ) {
            solverData.printf("@");
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
// this form of make_guess establishes a 'contract' between the caller's lambda
// and the creation of the new GridState.
// Due to its overhead, it is not the fastest, but the most flexible form to make a guess.
//
template<Verbosity verbose, typename F>
inline GridState* make_guess(unsigned char cell_index, F &&gridUpdater, Counters &counters, FILE *output) {
    // Create a copy of the state of the grid to make back tracking possible
    GridState* new_grid_state = this+1;
    if ( stackpointer >= GRIDSTATE_MAX-2 ) {
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
        fprintf(output, "guess at level >%d< - new level >%d<\nguess %s\n", stackpointer, new_grid_state->stackpointer, msgs[0]);
        char gridout[82];
        if ( debug > 1 ) {
            for (unsigned char j = 0; j < 81; ++j) {
                if ( unlocked.check_indexbit(j) ) {
                    gridout[j] = '0';
                } else {
                    gridout[j] = 49+_tzcnt_u32(candidates[j]);
                }
            }
            fprintf(output, "guess at %s\nsaved grid_state level >%d<: %.81s\n",
                   cl2txt[cell_index], stackpointer, gridout);
        }
        fprintf(output, "saved state for level %d: %s\n",
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
            fprintf(output, "grid_state at level >%d< now: %.81s\n",
                   new_grid_state->stackpointer, gridout);
        }
    }
    counters.guesses++;

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
            unsigned short altpair = wo_musts[ti];
            if ( __popcnt16 (altpair) == 2 ) {
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
    return make_guess_bv<verbose>(solverData, solverData->output);
found:
    // update the current and the new grid_state with their respective candidate to delete
    unsigned short select_cand = 0x8000 >> __lzcnt16(*wo_musts);

    // Create a copy of the state of the grid to make back tracking possible
    GridState* new_grid_state = this+1;
    if ( stackpointer >= GRIDSTATE_MAX-2 ) {
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
        solverData->printf("guess at level >%d< - new level >%d<\n", stackpointer, new_grid_state->stackpointer);
        solverData->printf("guess remove {%d} from %s triad at %s\n",
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
            solverData->printf("guess at %s\nsaved grid_state level >%d<: %.81s\n",
                   cl2txt[off], stackpointer, gridout);
        }
        if ( debug ) {
            solverData->printf("saved state for level %d: remove {%d} from %s triad at %s\n",
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
            solverData->printf("grid_state at level >%d< now: %.81s\n",
                   new_grid_state->stackpointer, gridout);
        }
    }
    solverData->counters.guesses++;

    return new_grid_state;
}

// this version of make_guess is the simplest and original form of making a guess.
//
template<Verbosity verbose>
inline GridState* make_guess_bv(SolverData *solverData, FILE *output) {
    // Find a cell with the least candidates.
    // For bivalues, build a score and once done, keep the highest scoring bivalue.
    // If there are no bivalues, score trivalues.
    // Pick the candidate with the highest value as the guess.
    // Save the current grid state (with the chosen candidate eliminated) for tracking back.

    bit128_t &bivalues = solverData->getBivalues(candidates);

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

        // leverage any guess hints (e.g. by the fish algorithm)
        if ( solverData->guess_hint_digit != 0 && score < guess_score_threshold ) {
            return make_guess<verbose>(solverData->guess_hint_index, solverData->guess_hint_digit, solverData->counters, solverData->output);
        }

        guess_index = best_index;
        if ( best_digit == 0 ) {
            best_digit = candidates[guess_index];
        }
        digit = 0x8000 >> __lzcnt16(best_digit); 
        return make_guess<verbose>(guess_index, digit, solverData->counters, output);
    }

    // Trivalue scoring
    //
    // very unlikely except for very hard puzzles while under 27 cells solved.
    // a puzzle with no bivalues is always hard - extra effort and solving algos are
    // appropriate.
    //
        if ( verbose != VNone ) {
            solverData->counters.no_bivals_count++;
        }

        // save the box_perp 'row's for later, at the price of 9 __m256i...
        // the save location is the next gridstate.
        // This is of no particular interest for now but could be made part of SolverData

        __m256i *box_perp_rows = (__m256i *) (this+1);

    {

// Replace this loop with the code below for efficiency:
//        unsigned short *candp = candidates;
//        for ( int k=0; k<9; candp += box_perp_ind_incr[k++]) {
//             box_perp_rows[k] = _mm256_setr_epi16(candp[0], candp[3],candp[6],
//                                                  candp[27], candp[30],candp[33],
//                                                  candp[54], candp[57],candp[60], 0, 0, 0, 0, 0, 0, 0);
//        }

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
                    pos2 = group4x3offsets[(62-__lzcnt64(mskpos))>>1];
                    // check for hidden pair
                    if ( mskByStart[lpos1] == mskpos ) {
                        // this is a hidden pair, simply guess dgti in pos1
                        // this will resolve at minimum 2 cells;
                        // make a guess right away
                        unsigned short cands = dgtByStart[lpos1] | dgt; 
                        candidates[score_index+pos1] &= cands;
                        candidates[score_index+pos2] &= cands;
                        if ( verbose == VDebug ) {
                            char ret[32];
                            format_candidate_set(ret, cands);
                            fprintf(output, "hidden pair (box): %-7s %s\n", ret, cl2txt[score_index+pos1]);
                        }
                        if ( verbose != VNone ) {
                            solverData->counters.naked_sets_found++;
                        }
                        return make_guess<verbose>(score_index+pos1, dgt, solverData->counters, output);
                    }
                    // score this conjugate pair
                    mskByStart[lpos1] = mskpos;
                    dgtByStart[lpos1] = dgt;
                    if ( (viewbits_by_i[pos1].u32 & viewbits_by_i[pos2].u32 & 0x300ffff) == 0 ) {
                        score++;
                    }
                    score_index = box_start_by_boxindex[boxi];
                    if ( __popcnt16(candidates[score_index+pos1]) <= 3 ) {
                        score++;
                        score_index += pos2;
                    } else {
                        if ( __popcnt16(candidates[score_index+pos2]) <= 3 ) {
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
                return make_guess<verbose>(best_index, best_digit, solverData->counters, output);
            }
        }

        // leverage any guess hints (e.g. by the fish algorithm)
        if ( solverData->guess_hint_digit != 0 ) {
            return make_guess<verbose>(solverData->guess_hint_index, solverData->guess_hint_digit, solverData->counters, solverData->output);
        }

        // find a trivalue cell if nothing else helps.

        unsigned char cnt;
        unsigned char best_cnt = 16;
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
        // Find the first candidate in this cell (lsb set)
        // Note: using tzcnt would be equally valid; this pick is historical
        digit = 0x8000 >> __lzcnt16(candidates[guess_index]);
        return make_guess<verbose>(guess_index, digit, solverData->counters, output);
}

// this version of make_guess takes a cell index and digit for the guess
//
template<Verbosity verbose>
inline GridState* make_guess(unsigned char guess_index, unsigned short digit, Counters &counters, FILE *output ) {
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
        fprintf(output, "guess at %s\nsaved grid_state level >%d<: %.81s\n",
               cl2txt[guess_index], stackpointer, gridout);
    }

    // Update candidates
    if ( verbose == VDebug ) {
        fprintf(output, "guess at level >%d< - new level >%d<\nguess", stackpointer, new_grid_state->stackpointer);
    }

    new_grid_state->enter_digit<verbose>( digit, guess_index, output);
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
        fprintf(output, "grid_state at level >%d< now: %.81s\n",
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
Status solve(signed char grid[81], GridState stack[], int line, Counters &counters, FILE *out = stdout) {

    GridState *grid_state = &stack[0];
    unsigned long long *unlocked = grid_state->unlocked.u64;
    unsigned short* candidates;

    Status status;

    SolverData solverData(counters, out);
    unsigned short current_entered_count = 81 - grid_state->unlocked.popcount();
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

    unsigned short last_entered_count_col_triads = 0;

    int unique_check_mode = 0;
    bool nonunique_reported = false;

    unsigned char no_guess_incr = 1;

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
        solverData.printf("Line %d: %.81s\n", line, gridout);
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
    // has its own non-solvability detecting trap door to detect if the grid is bad.
    // This section acts upon that detection and discards the current grid_state.
    //
    if (grid_state->stackpointer == 0) {
        if ( unique_check_mode ) {
            if ( verbose == VDebug ) {
                // no additional solution exists
                solverData.printf("No secondary solution found during back track\n");
            }
        } else {
            // This only happens when the puzzle is not valid
            // Bypass the verbose check...
            if ( warnings != 0 ) {
                solverData.printf("Line %d: No %ssolution found!\n", line, rules==Regular?"unique " : "");
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
            memcpy(grid, grid-82, 81);
        }
        return status;
    }

    current_entered_count  = (((grid_state-1)->stackpointer)<<8) | (81 - (grid_state-1)->unlocked.popcount());        // back to previous stack.

    // collect some guessing stats
    if ( verbose != VNone && reportstats ) {
        counters.digits_entered_and_retracted +=
            (_popcnt64((grid_state-1)->unlocked.u64[0] & ~grid_state->unlocked.u64[0]))
          + (_popcnt32((grid_state-1)->unlocked.u64[1] & ~grid_state->unlocked.u64[1]));
    }

    // Go back to the state when the last guess was made
    // This state had the guess removed as candidate from it's cell

    if ( verbose == VDebug ) {
        solverData.printf("back track to level >%d<\n", grid_state->stackpointer-1);
    }
    if ( verbose != VNone && reportstats ) {
        counters.trackbacks++;
    }
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
                            solverData.printf("Line %d: cell %s is 0\n", line, cl2txt[pos]);
                        }
                    } else if ( debug ) {
                        solverData.printf("back track - cell %s is 0\n", cl2txt[pos]);
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
                    solverData.printf("naked  single      ");
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
            solverData.printf(" %x at %s\n", _tzcnt_u32(e_digit)+1, cl2txt[e_i]);
        }
#ifndef NDEBUG
        if ( __popcnt16(e_digit) != 1 ) {
            if ( warnings != 0 ) {
                solverData.printf("error in e_digit: %x\n", e_digit);
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
                            solverData.printf("Line %d: cell %s is 0\n", line, cl2txt[pos]);
                        }
                    } else if ( debug ) {
                        solverData.printf("back track - cell %s is 0\n", cl2txt[pos]);
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
        if (unlocked[1] & (1ULL << (80-64)) ) {
            if ( to_update.u64[1] & (1ULL << (80-64)) ) {
                candidates[80] &= ~e_digit;
                if (__builtin_expect (candidates[80] == 0,0) ) {
                    // no solutions go back
                    if ( verbose != VNone ) {
                        if ( grid_state->stackpointer == 0 && unique_check_mode == 0 ) {
                            if ( warnings != 0 ) {
                                solverData.printf("Line %d: cell %s is 0\n", line, cl2txt[80]);
                            }
                        } else if ( debug ) {
                            solverData.printf("back track - cell %s is 0\n", cl2txt[80]);
                        }
                    }
                    goto back;
                }
            }
            if ( __popcnt16(candidates[80]) == 1) {
                // Enter the digit and update candidates
                if ( verbose == VDebug ) {
                    solverData.printf("naked  single      ");
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
                solverData.printf("naked  single      ");
            }
            goto enter;
        }
        e_digit = 0;
    }

    // The solving algorithm ends when there are no remaining unlocked cells.
    // The finishing tasks include verifying the solution and/or confirming
    // its uniqueness, if requested.
    //
    // Check if it's solved, if it ever gets solved it will be solved after looking for naked singles
    if ( *(__uint128_t*)unlocked == 0) {
        bool verify_one = false;
        // Solved it
        if ( rules == Multiple && (unique_check_mode == 1 || grid_state->flags < 0) ) {
            if ( !nonunique_reported ) {
                if ( verbose != VNone && reportstats && warnings != 0 ) {
                    solverData.printf("Line %d: solution to puzzle is not unique\n", line);
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
                        solverData.printf("Line %d: solution to puzzle failed verification\n", line);
                    }
                    counters.unsolved_count++;
                    counters.not_verified_count++;
                } else {     // not supposed to get here
                    if ( verbose != VNone ) {
                        solverData.printf("Line %d: secondary puzzle solution failed verification\n", line);
                    }
                }
            } else  if ( verbose != VNone ) {
                if ( debug ) {
                    solverData.printf("Solution found and verified\n");
                }
                status.verified = true;
                if ( reportstats ) {
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

        if ( verbose != VNone ) {
            counters.no_guess_cnt += no_guess_incr;
        }
        if ( report_guess_puzzles && no_guess_incr==0 ) {
            solverData.printf("%d %.81s\n", line, grid-82);
        }
        status.guess = no_guess_incr?false:true;

        if ( grid_state->stackpointer && rules == Multiple && unique_check_mode == 0 && grid_state->flags >= 0 ) {
            if ( verbose == VDebug ) {
                solverData.printf("Solution: %.81s\nBack track to determine uniqueness\n", grid);
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
    solverData.bivaluesValid = false;
    solverData.cbbvsValid = false;
    solverData.sectionSetUnlockedValid[0] = solverData.sectionSetUnlockedValid[1] = 
                                            solverData.sectionSetUnlockedValid[2] = false;
    solverData.guess_hint_digit = 0;

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
    // to check the nineth cell for a hidden single.  These are grouped together and then checked.
    //
    // For boxes, loading of the data is the most expensive.
    // Either the 'column' approach or the 'row' approach will work.
    // However, box hidden single search can just be eliminated, an the triad processing
    // will just eliminate all column/row-based occurrances of the single outside of the box.
    // This way, the expensive loading of the boxes is eliminated.
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
    // col_triads are numbered continuously per horizontal band (one vertical triad for each column)
    // and stored in three arrays of 10 ushort as follows:
    //    0,  1,  2,  3,  4,  5,  6,  7,  8, -
    //    9, 10, 11, 12, 13, 14, 15, 16, 17, -
    //   18, 19, 20, 21, 22, 23, 24, 25, 26
    //  
    // row_triads are canonically naturally numbered 0, 1, 2, for the first row, 3, 4, 5 for the second
    // and so on.  These are stored in three arrays of 10 ushort, captured 3 per row,
    // 3 rows stacked vertically with the subsequent 3 rows offset by 3:
    //    0,  1,  2,  9, 10, 11, 18, 19, 20, -
    //    3,  4,  5, 12, 13, 14, 21, 22, 23, -
    //    6,  7,  8, 15, 16, 17, 24, 25, 26
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
                    int idx = __tzcnt_u32(m)>>1;
                    unsigned short digit = ((v16us)missing)[idx];
                    if ( grid_state->stackpointer == 0 && unique_check_mode == 0 ) {
                        if ( warnings != 0 ) {
                            solverData.printf("Line %d: stack 0, back track - column %d misses digit %d\n", line, idx, __tzcnt_u16(digit)+1);
                        }
                    } else if ( debug ) {
                        solverData.printf("back track - missing digit %d in column %d\n", __tzcnt_u16(digit)+1, idx);
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
                                    solverData.printf("Line %d: stack 0, back track - col cell %s does contain multiple hidden singles\n", line, cl2txt[e_i]);
                                }
                            } else if ( debug ) {
                                solverData.printf("back track - multiple hidden singles in col cell %s\n", cl2txt[e_i]);
                            }
                        }
                        goto back;
                    }
                    if ( verbose == VDebug ) {
                        solverData.printf("hidden single (col)");
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
        // R9 R9 R9 R9 R9 R9 R9 R9 C9

        // Px: processed in pairs of rows, for their first 8 cells
        // C9: the 9th cell of each of the rows is prepared
        //     and stored in row_9th_cand_vert.  row_9th_cand_vert is then processed at the end.
        // R9: is processed for 8 cells

        // The examination of each row and its cells is made up of three steps:
        // 1 - or the other values of the row
        // 2 - check that the row or box contains all digits
        // 3 - check negated or'ed value for non-zero candidates.
        //     if there are multiple candidates, back track,
        //     enter any valid singletons found

        // row_9th_cand_vert cumulates the 9th row or'ed singleton candidates
        // which are processed in the end.
#if USE_ROW_HIDDEN_SEARCH
        v16us row_9th_cand_vert;
#endif
        unsigned char irow = 0;

        // find hidden singles in rows

        for (unsigned char i = 0; i < 72; i += 18, irow+=2) {
            // rows, in pairs

            __m256i c1 = _mm256_set_m128i(*(__m128i_u*) &candidates[i+9],
                                          *(__m128i_u*) &candidates[i]);

            unsigned short the9thcand_row[2] = { candidates[i+8], candidates[i+17] };

            __m256i row_or7 = _mm256_setzero_si256();
#if USE_ROW_HIDDEN_SEARCH
            __m256i row_or8;
#endif
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
                // triad capture: after 2 rounds, row triad 3 of this row saved in pos 5
                row_triad_capture[0] = _mm256_or_si256(row_or7, row_9th);
                // step j=2
                // rotate (2 3 4 5 6 7 0 1) -> (3 4 5 6 7 0 1 2)
                c1_ = _mm256_alignr_epi8(c1_, c1_, 2);
                // triad capture: after 3 rounds 2 row triads 0 and 1 in pos 7 and 2
                row_triad_capture[1] = row_or7 = _mm256_or_si256(c1_, row_or7);

#if USE_ROW_HIDDEN_SEARCH
                // continue the rotate/or routine for this row
                for (unsigned char j = 3; j < 7; ++j) {
                    // rotate (0 1 2 3 4 5 6 7) -> (1 2 3 4 5 6 7 0)
                    // c1_ holds 2 lanes for two rows
                    c1_ = _mm256_alignr_epi8(c1_, c1_, 2);
                    row_or7 = _mm256_or_si256(c1_, row_or7);
                }

                row_or8 = _mm256_or_si256(c1, row_or7);
                // test row_or8 to hold all the digits
                if ( check_back ) {
                    if ( !_mm256_testz_si256(mask1ff,_mm256_andnot_si256(_mm256_or_si256(row_9th, row_or8), mask1ff))) {
                        // the current grid has no solution, go back
                        if ( verbose != VNone ) {
                            __m256i missing = _mm256_andnot_si256(_mm256_or_si256(row_9th, row_or8), mask1ff);
                            unsigned int m = _mm256_movemask_epi8(_mm256_cmpeq_epi16(_mm256_setzero_si256(),
                                                                  missing));
                            unsigned short digit = ((v16us)missing)[(m & 0xffff)?8:0];

                            if ( grid_state->stackpointer == 0 && unique_check_mode == 0 ) {
                                if ( warnings != 0 ) {
                                    solverData.printf("Line %d: stack 0, back track - row %d does not contain digit %d\n", line, irow+(m & 0xffff)?0:1, __tzcnt_u16(digit)+1);
                                }
                            } else if ( debug ) {
                                solverData.printf("back track - missing digit %d in row %d\n", __tzcnt_u16(digit)+1, irow+((m & 0xffff)?1:0)); // XXX
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
                            solverData.printf("hidden single (row)");
                        }
                        goto enter;
                    }
                    if ( grid_state->stackpointer == 0 && unique_check_mode == 0 ) {
                        if ( warnings != 0 ) {
                            solverData.printf("Line %d: stack 0, back track - %s cell %s does contain multiple hidden singles\n", line, mask1?"row":"box", cl2txt[e_i]);
                        }
                    } else if ( debug ) {
                        solverData.printf("back track - multiple hidden singles in row cell %s\n", cl2txt[e_i]);
                    }
                    e_digit = 0;
                    goto back;
                }
#endif
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

#if USE_ROW_HIDDEN_SEARCH
            row_9th_cand_vert[irow]     = ~((v16us)row_or8)[0] & the9thcand_row[0];
            row_9th_cand_vert[irow+1]   = ~((v16us)row_or8)[8] & the9thcand_row[1];
#endif
        }   // for row Px

        // 9th row
        {
            __m128i c = *(__m128i_u*) &candidates[72];

            unsigned short the9thcand_row = candidates[80];

            __m128i row_or7 = _mm_setzero_si128();
#if USE_ROW_HIDDEN_SEARCH
            __m128i row_or8;
#endif
            __m128i row_9th_elem = _mm_set1_epi16(the9thcand_row);

            __m128i row_triad_capture[2];
            {
                __m128i c_ = c;
                // A2 and A3.1.b
                // first lane for the row, 2nd lane for the box
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
#if USE_ROW_HIDDEN_SEARCH
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
                            unsigned short digit = ((v8us)missing)[0];
                            if ( grid_state->stackpointer == 0 && unique_check_mode == 0 ) {
                                if ( warnings != 0 ) {
                                    solverData.printf("Line %d: stack 0, back track - row %d misses digit %d\n", line, 8, __tzcnt_u16(digit)+1);
                                }
                            } else if ( debug ) {
                                solverData.printf("back track - missing digit %d in row %d\n", __tzcnt_u16(digit)+1, 8);
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
                            solverData.printf("hidden single (row)");
                        }
                        goto enter;
                    } else {
                        if ( verbose != VNone ) {
                            if ( grid_state->stackpointer == 0 && unique_check_mode == 0 ) {
                                if ( warnings != 0 ) {
                                    solverData.printf("Line %d: stack 0, back track - row cell %s does contain multiple hidden singles\n", line, cl2txt[e_i]);
                                }
                            } else if ( debug ) {
                                solverData.printf("back track - multiple hidden singles in row cell %s\n", cl2txt[e_i]);
                            }
                        }
                        e_digit = 0;
                        goto back;
                    }
                }
#endif
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

#if USE_ROW_HIDDEN_SEARCH
            row_9th_cand_vert[8]     = ~((v8us)row_or8)[0] & the9thcand_row;
#endif
        } // row R9
#if USE_ROW_HIDDEN_SEARCH
        // check saved row singleton candidates (9th column)

        unsigned int mask = _mm256_movemask_epi8(_mm256_cmpgt_epi16((__m256i)row_9th_cand_vert, _mm256_setzero_si256())) & 0x3ffff;
        while (mask) {
            int s_idx = __tzcnt_u32(mask) >> 1;
            unsigned char celli = s_idx*9+8;
            unsigned short cand = row_9th_cand_vert[s_idx];
            if ( ((bit128_t*)unlocked)->check_indexbit(celli) ) {
                // check for a single.
                // This is rare as it can only occur when a wrong guess was made.
                // the current grid has no solution, go back
                if ( check_back && cand & (cand-1) ) {
                    if ( verbose != VNone ) {
                        if ( grid_state->stackpointer == 0 && unique_check_mode == 0 ) {
                            if ( warnings != 0 ) {
                                solverData.printf("Line %d: stack 0, multiple hidden singles in row cell %s\n", line, cl2txt[celli]);
                            }
                        } else if ( debug ) {
                            solverData.printf("back track - multiple hidden singles in row cell %s\n", cl2txt[celli]);
                        }
                    }
                    goto back;
                }
                if ( verbose == VDebug ) {
                    solverData.printf("hidden single (row)");
                }
                e_i = celli;
                e_digit = cand;
                goto enter;
            }
            mask &= ~(3<<(s_idx<<1));
        }
#endif
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
        if ( verbose != VNone ) {
            counters.triads_resolved += _popcnt64(m);
        }
        while (m) {
            unsigned char tidx = tzcnt_and_mask(m);
            unsigned short cands_triad = triad_info.col_triads[tidx];
            if ( verbose == VDebug ) {
                char ret[32];
                format_candidate_set(ret, cands_triad);
                solverData.printf("triad set (col): %-9s %s\n", ret, cl2txt[tidx/10*3*9+tidx%10]);
            }
            // mask off resolved triad:
            _bittestandreset((int*)&grid_state->triads_unlocked[Col], tidx);
            unsigned char off = tidx%10+tidx/10*27;
            grid_state->set23_found[Col].set_indexbits(0x40201,off,19);
            grid_state->set23_found[Box].set_indexbits(0x40201,off,19);
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

            if ( verbose != VNone ) {
                counters.triads_resolved++;
            }
            if ( verbose == VDebug ) {
                char ret[32];
                format_candidate_set(ret, triad_info.row_triads[tidx]);
                solverData.printf("triad set (row): %-9s %s\n", ret, cl2txt[off]);
            }

            // mask off resolved triad:
            grid_state->triads_unlocked[Row] &= ~(1LL << tidx);
            grid_state->set23_found[Row].set_indexbits(0x7,off,3);
            grid_state->set23_found[Box].set_indexbits(0x7,off,3);
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
                if ( verbose != VNone ) {
                     counters.triad_updates += _popcnt64(m);
                }
                while (m) {
                    // 1. Check for triad resolution
                    // 2. log the triad update from above
                    // 3. for columns, track resolved triads
                    unsigned char i_rel = tzcnt_and_mask(m);
                    unsigned char ltidx = i_rel + 9*i;  // logical triad index (within band, different layouts for row/col)
                    // compute bit offsets of the resolved triad
                    // and save the mask
                    if ( _popcnt32(((v16us)tmustnt[i])[i_rel]) == 6 ) {
                        if ( type == 0 ) {
                            rslvd_row_combo_tpos[i_rel/3] |= bandbits_by_index[row_triad_canonical_map[ltidx]%9];
                        } else {
                            rslvd_col_combo_tpos |= 0x40201<<i_rel;
                        }
                        if ( verbose != VNone ) {
                            counters.triads_resolved++;
                        }
                    }
                    if ( verbose == VDebug ) {
                        char ret[32];
                        format_candidate_set(ret, ((v16us)to_remove_v)[i_rel]);
                        solverData.printf("remove %-5s from %s triad at %s\n", ret, type == 0? "row":"col",
                               cl2txt[type==0?row_triad_canonical_map[ltidx]*3:col_canonical_triad_pos[ltidx]] );
                    }
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
            goto search;
        }
    }

    // here we are beyond the enter/single search/triad/, on to the more complex algos,
    // so we count this as a 'round'.
    if ( verbose != VNone ) {
        counters.past_naked_count++;
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
            bit128_t &bivalues = solverData.getBivalues(candidates);
            unsigned char target = 0;   // the index of the only cell with three candidates
            int sum2 = _popcnt64(bivalues.u64[0]) + _popcnt32(bivalues.u64[1]);
            if ( sum2 == N ) {
                grid_state->flags |= 1U<<31;	// set sign bit
                if ( verbose != VNone ) {
                    counters.bug_count++;
                }
                if ( rules != Regular ) {
                    if ( verbose == VDebug ) {
                        if ( grid_state->stackpointer == 0 && !unique_check_mode ) {
                            solverData.printf("Found a bi-value universal grave. This means at least two solutions exist.\n");
                        } else if ( unique_check_mode ) {
                            solverData.printf("checking a bi-value universal grave.\n");
                        }
                    }
                    goto guess;
                } else if ( grid_state->stackpointer ) {
                    if ( verbose == VDebug ) {
                        solverData.printf("back track - found a bi-value universal grave.\n");
                    }
                    goto back;
                } else {   // busted.  This is not a valid puzzle under standard rules.
                    if ( verbose == VDebug ) {
                        solverData.printf("Found a bi-value universal grave. This means at least two solutions exist.\n");
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
                    if ( verbose != VNone ) {
                        counters.bug_plus1_count++;
                    }
                    if ( verbose == VDebug ) {
                        solverData.printf("bi-value universal grave + 1: pivot:");
                    }
                    if ( rules == Regular ) {
                        e_i = target;
                        e_digit = digit;
                        goto enter;
                    } else {
                        if ( verbose == VDebug ) {
                            solverData.printf("\n");
                        }
                        grid_state = grid_state->make_guess<verbose>(target, digit, counters, solverData.output);
                    }
                    goto start;
                }
            }
        }
        no_bug:
        ;
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
//
// Critique:
// The approach is questionable, as it does not schedule the full section for revisit...
// It would be better to examine the full section and then only revisit when necessary.
// To be section centric as opposed to cell centric should allow for better performance as well.
//
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
        bit128_t tv = { .u128 = *cast2cu128(altbits[a^flip]) & to_visit_n.u128 };
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
                        if (verbose != VNone ) {
                            counters.naked_sets_searched++;
                        }
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
                                        solverData.printf("Line %d: naked  set (row) %s at %s, count exceeded\n", line, ret, cl2txt[ri*9]);
                                    }
                                } else if ( debug ) {
                                    solverData.printf("back track sets (row) %s at %s, count exceeded\n", ret, cl2txt[ri*9]);
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
                            if ( verbose != VNone ) {
                                counters.naked_sets_found++;
                            }
                            add_indices<Row>(&to_change, i);
                            if ( cnt==4 || delta==4 ) {
                                grid_state->flags |= (1<<ri);
                            }
                            if ( verbose == VDebug ) {
                                if ( cnt <=3 || cnt <= delta ) {
                                    format_candidate_set(ret, candidates[i]);
                                    solverData.printf("naked  %s (row): %-7s %s\n", s==2?"pair":"set ", ret, cl2txt[ri*9+i%9]);
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
                                    solverData.printf("%s %s (row): %-7s %s\n", complement?"hidden":"naked ", __popcnt16(complement)==2?"pair":"set ", ret, cl2txt[ri*9+k%9]);
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
                            if (verbose != VNone ) {
                                counters.naked_sets_searched++;
                            }
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
                                            solverData.printf("Line %d: naked  set (%s) %s at %s, count exceeded\n", line, js[j], ret, cl2txt[i]);
                                        }
                                    } else if ( debug ) {
                                        solverData.printf("back track sets (%s) %s at %s, count exceeded\n", js[j], ret, cl2txt[i]);
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
                                if ( verbose != VNone ) {
                                    counters.naked_sets_found++;
                                }
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
                                        solverData.printf("%s %s (%s): %-7s %s\n", naked_anyway?"naked ":"hidden", __popcnt16(complement)==2?"pair":"set ", js[j], ret, cl2txt[k]);
                                    } else {
                                        format_candidate_set(ret, candidates[i]);
                                        solverData.printf("naked  %s (%s): %-7s %s\n", cnt==2?"pair":"set ", js[j], ret, cl2txt[i]);
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
                    goto search;
                }
            } else if ( cnt == 1 ) {
                // this is not possible, but just to eliminate cnt == 1:
                e_i = i;
                e_digit = candidates[i];
                if ( warnings != 0 ) {
                    solverData.printf("found a singleton in set search... strange\n");
                }
                if ( verbose == VDebug ) {
                    solverData.printf("naked  (sets) ");
                }
                goto enter;
            }
        } // while
      }

      grid_state->updated.u128 = to_visit_again.u128;
    }
#endif

#ifdef OPT_FSH
// extra check by combining two binary bases into a swordfish base: 
#define COMBO_BASE
// just row or col fishes for speed of harder puzzles...
//#define NO_COL
//#define NO_ROW
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
    // Fishes are always there:
    // If the puzzle is solvable, then for each subset of size K of cells of a row or column there
    // must exist K columns or rows to complete a fish pattern.  If there are not enough such columns
    // or rows, the puzzle is not solvable (on that path), so back track.
    //
    // fishes that need to be cleaned:
    // Given a base of size N, check whether there are:
    // - either N lines that match with a real (or assumed) base, and some extra
    //   lines that overlap with the base, then these extra line can be purged
    //   (their base portion removed) and the fish is complete,
    // - or exactly N lines that overlap with the base, then all these lines
    //   can be purged (their non-base portion removed).
    // Note that these two possibilities are the same, except the second is transposed.
    // 
    // The search operates just as the naked set search does, i.e. it will not discover
    // fishes where all lines have less than N candidates.  This can be remedied by augmenting
    // the set of lines using additional lines (by 1 additional line, which should go a long
    // way (a jelly fish can have 4 lines of 2 candidates each, but the value of such a search
    // needs to be confirmed).
    //
    // Once a grid has been found and cleaned, there are two distinct subsets
    // of fishes present, i.e. the fish and its complementary fish.
    // If either the fish or its complement have size of 3 or lower, they cannot
    // be subdivided and therefore need not be searched further.
    //
    // Variants:
    // All variants start with N-1 matching fish lines (subsets of regular fishes
    // as described above).
    //
    // Finned Fish:
    // identify from the excess lines overlapping the base, one that:
    // - has one or two extra cells, both in the same box,
    // - sharing a grid point in the same box.
    // Either one of the extra candidates is valid, or the fish pattern is valid.
    // Any candidates on the perpendicular grid lines that is not a grid point,
    // that can be seen by the grid point and the extra cells can be eliminated.
    //
    // Finned/sashimi Fish:
    // This is the same as a finned fish, but the grid point in the same box lacks
    // the candidate.
    //
    // Roofed Fish:
    // This requires exactly two excess lines which must share a band.
    // The overlap with the base must be a single point
    // for each, resulting in two 'fins', which must not share a box.
    // As there are only two fins, one of them must complete the grid or the grid
    // would lack a line. They cannot both be valid either, as that would create
    // an invalid 'fish' again.
    // All candidates that can be seen by both fins can be eliminated.
    // The "roofed" prefix alludes to the skyscraper pattern with its roof and
    // the strong ressemblance in the resulting eliminations.
    // Roofed fishes are not frequent.
    //
    // Skyscraper:
    // The skyscraper pattern emerges when the base is composed of just two candidates.
    // Roofed X-wings are labeled as 'skyscraper'.
    // Skyscrapers are quite frequent.
    //
    // with the cbbvs in place, look for simple 'fishes'

    cbbv_t cbbv_v;

    bit128_t *candidate_bits_by_value = solverData.getCbbvs(candidates);
    for ( unsigned char dgt = 0; dgt < 9; dgt++) {
        // don't bother with six cells already resolved, a swordfish cannot be subdivided.
        // a jellyfish is final with 8 candidates (4 x 2), so 8 + 5 = 13 cannot be divided
        // unless it is already.
        // The minimum number of candidates to work with is 14. 
        if ( candidate_bits_by_value[dgt].popcount() < 14 ) {
            continue;
        }
        // load a vector eliminating solved cells
        bit128_t cbbv_digit = { .u128= candidate_bits_by_value[dgt].u128 & grid_state->unlocked.u128 };

        // load cbbv_v
        __m128i off0 = uncompress(cbbv_digit.m128);
        cbbv_v.m256 = _mm256_insert_epi16(_mm256_and_si256(mask1ff,_mm256_castsi128_si256(off0)), *(__int16*)&cbbv_digit.u8[9], 8);
        unsigned long long hi = cbbv_digit.u64[1];
#if 0
        unsigned long long lo = cbbv_digit.u64[0];

dump_m256i_grid(cbbv_v.m256, "new");
dump_m256i_grid(_mm256_and_si256(_mm256_setr_epi16(lo, lo>>9, lo>>18, lo>>27, lo>>36, lo>>45, lo>>54, (lo>>63)|(hi<<1), hi>>8, 0, 0, 0, 0, 0, 0, 0),mask1ff), "old");
#endif
        // do a quick popcnt of all rows or'ed together to determine the maximum fish size
        __m128i max_v = _mm256_castsi256_si128(_mm256_or_si256(cbbv_v.m256, _mm256_bsrli_epi128(cbbv_v.m256,8)));
        max_v = _mm_or_si128(max_v, _mm_bsrli_si128(max_v,4));
        max_v = _mm_or_si128(max_v, _mm_bsrli_si128(max_v,2));
        unsigned char max_all = __popcnt16(_mm_extract_epi16(max_v, 0) | hi>>8);
        unsigned char max = max_all > 7 ? 7 : max_all;
        bit128_t dgt_bits = { .u128 = candidate_bits_by_value[dgt].u128 & grid_state->unlocked.u128 };
        unsigned char celli;
        unsigned char celli2;

        unsigned int pair_locs = 0;  // pairs, bits left to right
        unsigned int alt_base_x = 0;
#ifndef NO_ROW
        for (unsigned char t=0; t<9; t++, pair_locs <<= 1) { // t is the index of the row the pattern sampled
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
                unsigned int hassubs_prp_x = compress_epi16_boolean(hassub_v);  // bits for non-empty intersection with base_x
                unsigned char nsubs = _popcnt32(subs_prp_x);
                // test for naked fish and exclude it for row and col search
                if ( nsubs == cnt && _mm256_testc_si256(issub_v, hassub_v) ) {
                    // found a (naked) fish, leave it
                    if ( cnt <= 3 ) {
                        exclude_row[dgt] |= base_x;
                        exclude_col[dgt] |= subs_prp_x; // the orthogonal grid base
                        if ( verbose != VNone ) {
                            counters.fishes_excluded++;   // counted but not currently included in summary
                        }
                    }
                    continue;
                }
                if ( check_back && _popcnt32(hassubs_prp_x) < cnt ) {
                    if ( verbose == VDebug ) {
                        solverData.printf("back track - insufficient number of rows to form %s for digit %d based on row %d\n", fish_names[cnt-2], dgt+1, t);
                    }
                    goto back;
                }

                bit128_t clean_bits {};
                unsigned char fincells[2] = {0xff, 0xff};
                bool dosearch = false;

                // if there are exactly N rows with exclusively some of the pattern of digits, then
                // it is a row fish pattern.
                // if there are exactly N rows that have some of the pattern plus excess,
                // then it is a col fish pattern.
                if ( nsubs == cnt || _popcnt32(hassubs_prp_x) == cnt ) {
                    // this part is efficient to find and easy to deal with - however, the yield is very low.
                    unsigned short excess_x = nsubs == cnt ? base_x : (0x1ff & ~base_x);
                    if ( verbose != VNone ) {
                        counters.fishes_detected++;
                    }
                    unsigned int rows2clean = hassubs_prp_x & ~subs_prp_x;
                    while (rows2clean) {
                        unsigned char row = tzcnt_and_mask(rows2clean);
                        clean_bits.set_indexbits( cbbv_v.v16[row]&excess_x, row*9, 9);
                    }
                    clean_bits.u128 &= grid_state->unlocked.u128 & candidate_bits_by_value[dgt].u128;
                    if ( clean_bits ) {
                        if ( verbose == VDebug ) {
                            celli  = __tzcnt_u16(subs_prp_x)*9 + __tzcnt_u16(base_x);
                            celli2 = (15-__lzcnt16(subs_prp_x))*9 + (15-__lzcnt16(base_x));
                            if ( debug > 2 ) {
                                show_fish(cbbv_v, base_x, nsubs == cnt ? subs_prp_x:hassubs_prp_x, clean_bits, dgt_bits, solverData, "row fish");
                            }
                            solverData.printf("%s (%s) digit %d at cells %s - %s\nRemove %d at ", fish_names[cnt-2], nsubs==cnt?"rows":"cols", dgt+1, cl2txt[celli], cl2txt[celli2], dgt+1);
                        }
                        dosearch = true;
                    }
                    if ( cnt <= 3 ) {
                        exclude_row[dgt] |= base_x;
                        exclude_col[dgt] |= nsubs == cnt ? subs_prp_x:hassubs_prp_x; // the orthogonal grid base
                    } else if ( cnt >= max_all-3 ) {
                        exclude_row[dgt] |= 0x1ff & ~base_x;
                        exclude_col[dgt] |= 0x1ff & ~(nsubs == cnt ? subs_prp_x:hassubs_prp_x); // the orthogonal grid base
                    }

                } else if ( nsubs == cnt-1 ) {
                    // nsubs == cnt-1 is required for finned/sashimi fishes and roofed fishes.
                    // There are two case, which are not mutually exclusive:
                    // Case 1: (finned fish and finned/sashimi fish)
                    // for a fin, the following must be true:
                    // The fin cell(s) must be on the parallel fish sections w.r.t. to the base_x.
                    // A. the fin cells must all be in the same box (and aligned if there are 2), and
                    // B. the same box must contain a fish grid point (which need not have a candidate)
                    // In this case the fin(s) views intersect with their associated fish grid point's
                    // view to provide the triad (minus the fish grid) to be cleaned.
                    // In the case that this is an X-Wing and the there is no candidate in the fish
                    // grid point this is also a Skyscraper pattern and two triads can be cleaned.
                    //
                    // Case 2: (roofed)
                    // This pattern is homegrown, as I cannot find it referenced anywhere.
                    // It begins with a base of N, with N-1 restricted lines found.
                    // If there are no additional overlapping lines, a Nx(N-1) fish
                    // cannot swim (back track).
                    // With a count of overlapping lines P, we have the following cases:
                    // P==0: (already dealt with above) back track
                    // P==1: (already dealt with above) clean that line
                    // P==2: one of the matching cells must be valid.  If there are only
                    //    two, they must be mutually exclusive.
                    //    If there are more than two matching cells, nothing to do.
                    //    If those two cells are in different boxes, their intersection of
                    //    of views can be cleaned.
                    //
                    // One compromise so as to not confuse matters further, let's call those
                    // 'fins' that are not fins 'roof-fins' in this context.
                    //
                    // The following must be true:
                    // The roof-fin(s) cells must be on the parallel fish sections w.r.t. to the base_x.
                    // There must be exactly 2 roof-fin sections.
                    // These roof-fin cells are mutually exclusive and exactly one would
                    // be required to complete the fish pattern and eliminate the other.
                    // These two roof-fins can be considered individually, however in terms of eliminations
                    // and dependencies they are a more powerful in combination.
                    // The proof goes as follows:
                    // - If neither were true, the remaining pattern would be impossible (grid Nx(N-1)).
                    // - If both were true, the remaining pattern would be likewise impossible (grid (Nx(N+1)),
                    // Therefore, these roof-fin cells are mutually exclusive.
                    // Conditions for case 2:
                    // A. This case requires exactly two roof-fin sections.
                    // B. The respective roof-fin cells must be unique on the perpendicular fish grid line
                    //    except for grid points (ie. they cannot be on the same perpendicular fish grid line).
                    // C. They must share a band parallel to the base_x and not share
                    //    a band with any established grid line parallel to the base.
                    // D. They must not share a box. If they did, they could not eliminate any candidates
                    //    without violating A - C.
                    // In this case the two roof-fins' views intersect with each other to define
                    // two triads to be cleaned.
                    // When logged, this case is identified as 'sashimi'
                    //
                    // Addendum to Case 2:
                    // A direct consequence for the band with the two mutual exclusive roof-fins
                    // is the following:
                    // If for two cells in different boxes of the same band their value is
                    // mutually exclusively the same digit D, the triad T that does not share
                    // a box or row/col with either of the two cells, if:
                    // the value candidate is unique in the respective triads of these two cells,
                    // then the triad T cannot contain the digit D.
                    // This situation occurs frequently in conjunction with Case 2.
                    // As proof and for illustration:  The band triads (without loss of generality, a row band):
                    //       A1   A2   A3
                    //       B1   B2   B3
                    //       C1   C2   C3
                    // Without loss of generality assume that triads A1 and B2 contain cells that
                    // have a candidate value D.  We do know, that due to being linked exclusively,
                    // that neither A2 nor B1 can contain candidate value D (and the code makes sure of that).
                    // If A1 does not contain D, then A3 does.  Conversely, if B2 does not contain D,
                    // then B3 does.
                    // Therefore, since D is either in A3 or B3, D cannot be in C3.
                    // In the scenario of Case 2 we must simply look for the following conditions:
                    // a. the roof-fins must not share a box,
                    // b. the roof-fins must be the only candidate D in their respective triad.
                    // iterate over possible roof-fin cols
                    hassubs_prp_x = compress_epi16_boolean(hassub_v);
                    unsigned int finsubs_prp_x = hassubs_prp_x & ~subs_prp_x; //& ~exclude_row;
                    bool sashimi = false;
                    unsigned int subsx= subs_prp_x;
                    unsigned int finsubcnt = _popcnt32(finsubs_prp_x);  // the complete count
                    bool skyscraper = cnt==2;

                    // roofed fishes that are not skyscrapers are rare - optimized out
                    if ( skyscraper && finsubcnt == 2 ) {   // Case 2 A
                        unsigned short rooffin_subs_prp[2] = { (unsigned short)_tzcnt_u32(finsubs_prp_x), (unsigned short)(31-__lzcnt32(finsubs_prp_x)) };
                        if ( rooffin_subs_prp[0]/3 == rooffin_subs_prp[1]/3 ) {     // Case 2 C
                            unsigned short rooffin_subs_pos_prl[2] = { __tzcnt_u16(cbbv_v.v16[rooffin_subs_prp[0]] & base_x),
                                                                       __tzcnt_u16(cbbv_v.v16[rooffin_subs_prp[1]] & base_x) };

                            if (    rooffin_subs_pos_prl[0] != rooffin_subs_pos_prl[1]                // Case 2 B
                                 && _popcnt32(cbbv_v.v16[rooffin_subs_prp[0]] & base_x) == 1
                                 && _popcnt32(cbbv_v.v16[rooffin_subs_prp[1]] & base_x) == 1 ) {
                                if ( verbose != VNone ) {
                                    counters.fishes_detected++;
                                    counters.fishes_specials_detected++;
                                }
                                fincells[0] =  rooffin_subs_prp[0]*9 + rooffin_subs_pos_prl[0];
                                fincells[1] =  rooffin_subs_prp[1]*9 + rooffin_subs_pos_prl[1];
                                clean_bits.u128 =    (*(bit128_t*)big_index_lut[fincells[0]][All]).u128
                                                   & (*(bit128_t*)big_index_lut[fincells[1]][All]).u128;
                                // deal with 'addendum to Case 2'
                                if ( fincells[0]/3%3 != fincells[1]/3%3 ) { // not in same box
                                    if (    _popcnt32(candidate_bits_by_value[dgt].get_indexbits(fincells[0]-fincells[0]%3, 3)) == 1
                                         && _popcnt32(candidate_bits_by_value[dgt].get_indexbits(fincells[1]-fincells[1]%3, 3)) == 1) {
                                        unsigned char boxoff = 3 * _tzcnt_u32(7 ^ ((1<<fincells[0]/3%3) | (1<<fincells[1]/3%3)));
                                        unsigned char rowoff = 9 * ( _tzcnt_u32(7 ^ ((1<<fincells[0]/9%3) | (1<<fincells[1]/9%3)))
                                                                     + rooffin_subs_prp[0]/3*3 );
                                        clean_bits.set_indexbits(7, rowoff+boxoff, 3);
                                    }
                                }
                                clean_bits.u128 &= candidate_bits_by_value[dgt].u128
                                                   & grid_state->unlocked.u128;
                                clean_bits.unset_indexbit(fincells[0]);
                                clean_bits.unset_indexbit(fincells[1]);
                                if ( clean_bits ) {
                                    if ( verbose != VNone ) {
                                        counters.fishes_specials_updated++;
                                    }
                                    if ( verbose == VDebug ) {
                                        if ( skyscraper ) {
                                            unsigned char cellif  = __tzcnt_u16(subsx)*9 + __tzcnt_u16(base_x);
                                            unsigned char cellif2 = __tzcnt_u16(subsx)*9 + (15-__lzcnt16(base_x));
                                            if ( debug > 2 ) {
                                                show_skyscraper(cellif, cellif2, fincells[0], fincells[1], clean_bits, dgt_bits, solverData, "skyscraper (row)");
                                            }
                                            solverData.printf("skyscraper (row floor) digit %d floor at %s-%s roof at %s/%s\n   remove %d at ", dgt+1, cl2txt[cellif], cl2txt[cellif2], cl2txt[fincells[0]], cl2txt[fincells[1]], dgt+1);
                                        } else {
                                            subsx = subs_prp_x | (1<<rooffin_subs_prp[0]) | (1<<rooffin_subs_prp[1]);
                                            celli  = __tzcnt_u16(subsx)*9 + __tzcnt_u16(base_x);
                                            celli2 = (15-__lzcnt16(subsx))*9 + (15-__lzcnt16(base_x));
                                            if ( debug > 2 ) {
                                                show_fish(cbbv_v, subs_prp_x, 2, fincells, clean_bits, dgt_bits, solverData, "roofed row fish");
                                            }
                                            solverData.printf("roofed %s (rows) digit %d at cells %s - %s\nFins at %s,%s - remove %d at ", fish_names[cnt-2], dgt+1, cl2txt[celli], cl2txt[celli2], cl2txt[fincells[0]], cl2txt[fincells[1]], dgt+1);
                                        }
                                    }
                                } else if ( solverData.guess_hint_digit == 0 && !clean_bits && cnt == 2 ) {
                                    if ( __popcnt16(candidates[fincells[0]]) == 2 ) {
                                        solverData.guess_hint_digit = 1<<dgt;
                                        solverData.guess_hint_index = fincells[0];
                                    }
                                }
                            }
                        }

                        if ( clean_bits ) {
                            if ( verbose != VNone ) {
                                counters.fishes_updated++;
                            }
                            dgt_bits.u128 &= ~clean_bits.u128; // for Case 1
                            unsigned short dgt_mask_bit = 1<<dgt;
                            while (clean_bits) {
                                unsigned char cl = tzcnt_and_mask(clean_bits);
                                if ( candidates[cl] & dgt_mask_bit ) {
                                    candidates[cl] &= ~dgt_mask_bit;
                                    if ( verbose == VDebug ) {
                                        solverData.printf("%s ", cl2txt[cl]);
                                    }
                                }
                            }
                            if ( verbose == VDebug ) {
                                solverData.printf("\n");
                            }
                            dosearch = true;
                        }
                    }

                    {
                        // Case 1
                        bool skyscraper = false;
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
                            // A. the fin cells must all be in the same box (and aligned if there are 2), and
                            // B. the same box must contain a fish grid point (which need not have a candidate)

                            if (  (fins & ~box_bits) || !(box_bits & base_x) ) {   // Case 1, A and B
                                continue;
                            }
                            // if the fin has no candidate on its associated grid point, it's a sashimi
                            // which  is the common nomenclature.
                            // A finned/sashimi X-wing with a single fin is identified as a skyscraper pattern
                            // since it eliminates candiates symmetrically.

                            sashimi = !(finln_prl_x & base_x & box_bits);

                            skyscraper = sashimi && alt_base_x==0 && cnt == 2 && __popcnt16(fins) == 1;

                            // roof coordinates for skyscraper:
                            unsigned char cellir {};
                            unsigned char cellir2 {};

                            // complete the finned fish pattern:
                            subsx = subs_prp_x | (1<<fin_sub);

                            // the rows left for cleaning
                            unsigned short band_bits = bandbits_by_index[fin_sub/3];
                            unsigned int rows2clean = hassubs_prp_x & ~subsx & band_bits;
                            if ( rows2clean == 0 ) {
                                if ( !skyscraper ) {
                                    continue;
                                }
                                // the skyscraper base_x is a good guess to keep for later:
                                if ( solverData.guess_hint_digit == 0 && !clean_bits && sashimi && cnt == 2 ) {
                                    solverData.guess_hint_digit = 1<<dgt;
                                    solverData.guess_hint_index = t*9 + __tzcnt_u16(base_x & ~box_bits);
                                    if ( __popcnt16(solverData.guess_hint_index) != 2 ) {
                                        solverData.guess_hint_digit = 0;
                                    }
                                }
                            }
                            if ( verbose != VNone ) {
                                counters.fishes_specials_detected++;
                                counters.fishes_detected++;
                            }

                            unsigned int boxbase_x = base_x & box_bits;

                            // skyscraper is not singled out in statistics
                            if ( skyscraper ) {
                                // row:
                                cellir = t*9 + __tzcnt_u16(box_bits & base_x);
                                cellir2 = fin_sub*9 + __tzcnt_u16(fins);
                                clean_bits.u128 =   *cast2cu128(big_index_lut[cellir][All])
                                                  & *cast2cu128(big_index_lut[cellir2][All]);
                                // deal with skyscraper analogous to Case 2 addendum
                                // but first verify several conditions:
                                // - the roof must sit in different boxes
                                // - each part of the roof must be a single candidate in their vertical triad
                                // - the skyscraper's floor must be a conjugate pair (ie. strongly connected)
// XXX much could be rewritten here in terms of 9.4 triad lookup tables...
                                unsigned char cellirxor = (1<<(cellir/27)) ^ (1<<(cellir2/27));
                                if ( cellirxor ) { // not in same (vertical) band
                                    if (    _popcnt32(candidate_bits_by_value[dgt].get_indexbits(cellir/27*27 + cellir%9, 19) & 0x40201) == 1
                                         && _popcnt32(candidate_bits_by_value[dgt].get_indexbits(cellir2/27*27 + cellir2%9, 19) & 0x40201) == 1) {
                                        // the roof candidates must not have 'siblings' in their respective boxes
                                        bit128_t base_column = { .u128= *cast2cu128(big_index_lut[__tzcnt_u16(base_x & ~box_bits)][Col]) & dgt_bits.u128 };
                                        if ( base_column.popcount() == 2 ) { // the skyscraper's floor is strongly linked
                                            unsigned char cellir3rd2 = _tzcnt_u32(7^((1<<(cellir%3))^(1<<(cellir2%3))));
                                            unsigned char cellir3rd = _tzcnt_u32(7^cellirxor);
                                            unsigned char boxoff = cellir%9/3*3+cellir3rd2;
                                            unsigned char coloff = cellir3rd*27;
                                            clean_bits.set_indexbits(0x40201, coloff+boxoff, 19);
                                        }
                                    }
                                }
                            } else {
                                while (rows2clean) {
                                    unsigned char row = tzcnt_and_mask(rows2clean);
                                    clean_bits.set_indexbits( cbbv_v.v16[row]&boxbase_x, row*9, 9);
                                }
                            }
                            clean_bits.u128 &= dgt_bits.u128;

                            if ( clean_bits ) {
                                if ( verbose != VNone ) {
                                    counters.fishes_specials_updated++;
                                }
                                if ( verbose == VDebug ) {
                                    if ( skyscraper ) {
                                        celli = t*9 + __tzcnt_u16(base_x&~box_bits);
                                        celli2 = fin_sub*9 + __tzcnt_u16(base_x&~box_bits);
                                        if ( debug > 2 ) {
                                            show_skyscraper(celli, celli2, cellir, cellir2, clean_bits, dgt_bits, solverData, "skyscraper (col)");
                                        }
                                        solverData.printf("skyscraper (col floor) digit %d floor at %s-%s roof at %s/%s\n   remove %d at ", dgt+1, cl2txt[celli], cl2txt[celli2], cl2txt[cellir], cl2txt[cellir2], dgt+1);
                                    } else {
                                        if ( debug > 2 ) {
                                            show_fish(cbbv_v, base_x, subsx, clean_bits, dgt_bits, solverData, "finned row fish");
                                        }
                                        celli = __tzcnt_u16(subsx)*9 + __tzcnt_u16(base_x);
                                        celli2 = (15-__lzcnt16(subsx))*9 + (15-__lzcnt16(base_x));
                                        solverData.printf("finned%s %s (rows) digit %d at cells %s - %s\nFin at %s - remove %d at ", sashimi?"/sashimi":"", fish_names[cnt-2], dgt+1, cl2txt[celli], cl2txt[celli2], cl2txt[fin_sub*9+__tzcnt_u16(fins&~base_x)], dgt+1);
                                    }
                                }
                                break;
                            }
                        } // while
                    } // Case 1
                } // else
                if ( clean_bits ) {
                    if ( verbose != VNone ) {
                        counters.fishes_updated++;
                    }
                    if ( clean_bits.check_indexbit(fincells[0]) ) {
                        e_digit = 1<<dgt;
                        e_i = fincells[1];
                    } else if ( clean_bits.check_indexbit(fincells[1]) ) {
                        e_digit = 1<<dgt;
                        e_i = fincells[0];
                    }
                    unsigned short dgt_mask_bit = 1<<dgt;
                    while (clean_bits) {
                        unsigned char cl = tzcnt_and_mask(clean_bits);
                        if ( candidates[cl] & dgt_mask_bit ) {
                            candidates[cl] &= ~dgt_mask_bit;
                            if ( verbose == VDebug ) {
                                solverData.printf("%s ", cl2txt[cl]);
                            }
                        }
                    }
                    if ( verbose == VDebug ) {
                        if ( e_digit == 0 ) {
                            solverData.printf("\n");
                        } else {
                            solverData.printf("\n   cells %s and %s are mutually exclusive (%s%s on digit %d),\nenter the remaining ", cl2txt[fincells[0]], cl2txt[fincells[1]], cnt==2?"skyscraper":"roofed", cnt==2?"":fish_names[cnt-2], dgt+1);
                        }
                    }
                    if ( e_digit ) {
                        goto enter;
                    }
                    dosearch = true;
                }
                if ( dosearch ) {
                    goto search;
                }
            } // if
#ifdef COMBO_BASE
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
#endif
        } // for t
#endif

#ifndef NO_COL


        // part 2: look for fishes at columns

        // transpose cbbv_v
        cbbv_t cbbv_col_v {};

        unsigned short *mskp = &cbbv_col_v.v16[8];
        __m256i c = _mm256_srli_epi16(cbbv_v.m256,1);
        *mskp-- = _mm_movemask_epi8(_mm_packus_epi16(_mm256_castsi256_si128(c),_mm256_extracti128_si256(c,1)));
        c = _mm256_and_si256(cbbv_v.m256, maskff);
        __m128i cc = _mm_packus_epi16(_mm256_castsi256_si128(c),_mm256_extracti128_si256(c,1));
        for (unsigned char d = 8; d > 0; d--) {
            *mskp-- = _mm_movemask_epi8(cc);
            cc = _mm_slli_epi16(cc,1);
        }

        pair_locs = 0;  // pairs, bits left to right
        alt_base_x = 0;

        for (unsigned char t=0; t<9; t++, pair_locs <<= 1) { // t is the row the pattern is sampled from
            unsigned int base_x = alt_base_x? alt_base_x : cbbv_col_v.v16[t]; // base_x: tentative fish pattern

            if ( base_x & exclude_col[dgt] ) {
                continue;
            }
            // for the given digit, the count in a given col is N (cnt).
            // take each col of bits and compare to all other cols.
            //
            unsigned char cnt = __popcnt16(base_x);

            if ( cnt+2 <= max && cnt > 1) {  // maximum cnt is 5 ('squirmbag').
                if ( cnt == 2 ) {
                    pair_locs |= 1;
                }
                __m256i basev = _mm256_and_si256(_mm256_set1_epi16(base_x),mask9);
                // rows that have a non-empty intersection with base_x
                __m256i hassub_v = _mm256_cmpgt_epi16(_mm256_and_si256(basev, cbbv_col_v.m256),_mm256_setzero_si256());
                // rows that are a non-empty subset of base_x
                __m256i issub_v = _mm256_and_si256(hassub_v, _mm256_cmpeq_epi16(basev, _mm256_or_si256(basev, cbbv_col_v.m256)));

                unsigned int subs_prp_x = compress_epi16_boolean(issub_v);
                unsigned char nsubs = _popcnt32(subs_prp_x);
                unsigned int hassubs_prp_x = compress_epi16_boolean(hassub_v);

                // test for naked fish and exclude it for row and col search
                if ( nsubs == cnt && _mm256_testc_si256(issub_v, hassub_v) ) {
                    // found a (naked) fish, leave it
#ifdef NO_ROW
                    counters.fishes_detected++;
                    if ( cnt <= 3 ) {
                        exclude_col[dgt] |= base_x;
                        exclude_row[dgt] |= subs_prp_x; // the orthogonal grid base
                        counters.fishes_excluded++;   // counted but not currently included in summary
                    }
#else
                    // for naked fishes, don't double count row/col detection
                    // neither need to exclude anything;
#endif
                    continue;
                }

                if ( check_back && _popcnt32(hassubs_prp_x) < cnt ) {
                    if ( verbose == VDebug ) {
                        solverData.printf("back track - insufficient number of cols to form %s for digit %d based on row %d\n", fish_names[cnt-2], dgt+1, t);
                    }
                    goto back;
                }

                bit128_t clean_bits {};
                bit128_t tmp {};
                unsigned char fincells[2] = {0xff, 0xff};
                bool dosearch = false;

                // if there are exactly N cols with exclusively some of the pattern of digits, then
                // it is a col fish pattern.
                // if there are exactly N cols that have some of the pattern plus excess,
                // then it is a row fish pattern.
                if ( nsubs == cnt || _popcnt32(hassubs_prp_x) == cnt ) {
                    // this part is efficient to find and easy to deal with - however the yield very low.
                    unsigned short excess_x = nsubs == cnt ? base_x : (0x1ff & ~base_x);
                    if ( verbose != VNone ) {
                        counters.fishes_detected++;
                    }
                    hassubs_prp_x = compress_epi16_boolean(hassub_v);
                    unsigned int cols2clean = hassubs_prp_x & ~subs_prp_x;
                    while (cols2clean) {
                        unsigned char col = tzcnt_and_mask(cols2clean);
                        tmp.set_indexbits( cbbv_col_v.v16[col]&excess_x, col*9, 9);
                    }
                    // since we work in a transposed view, we transpose clean_bits here:
                    while ( tmp ) {
                        clean_bits.set_indexbit(transposed_cell[tzcnt_and_mask(tmp)]);
                    }
                    clean_bits.u128 &= dgt_bits.u128;
                    if ( clean_bits.u128 ) {
                        if ( verbose == VDebug ) {
                            celli = __tzcnt_u16(subs_prp_x) + __tzcnt_u16(base_x)*9;
                            celli2 = (15-__lzcnt16(subs_prp_x)) + (15-__lzcnt16(base_x))*9;
                            if ( debug > 2 ) {
                                show_fish<true>(cbbv_col_v, base_x, nsubs == cnt ? subs_prp_x:hassubs_prp_x, clean_bits, dgt_bits, solverData, "col fish");
                            }
                            solverData.printf("%s (%s) digit %d cells %s - %s\nRemove %d at ", fish_names[cnt-2], nsubs == cnt? "cols":"rows", dgt+1, cl2txt[celli], cl2txt[celli2], dgt+1);
                        }
                        dosearch = true;
                    }
                    if ( cnt <= 3 ) {
                        exclude_col[dgt] |= base_x;
                    }
                } else if ( nsubs == cnt-1 ) {
                    // nsubs == cnt-1 is required for both fin and roofed fishes.

                    // Case 2:
                    // iterate over possible roof-fin cols
                    hassubs_prp_x = compress_epi16_boolean(hassub_v);
                    unsigned int finsubs_prp_x = hassubs_prp_x & ~subs_prp_x ; //& ~exclude_col;
                    bool sashimi = false;
                    unsigned int subsx= subs_prp_x;
                    unsigned int finsubcnt = _popcnt32(finsubs_prp_x);  // the complete count
                    bool skyscraper = cnt==2;
                    // roofed fishes that are not skyscrapers are rare - optimized out
                    if ( skyscraper && finsubcnt == 2 ) {   // Case 2 A
                        unsigned short rooffin_subs_prp[2] = { (unsigned short)_tzcnt_u32(finsubs_prp_x), (unsigned short)(31-__lzcnt32(finsubs_prp_x)) };
                        if ( rooffin_subs_prp[0]/3 == rooffin_subs_prp[1]/3 ) {    // Case 2 C
                            unsigned short rooffin_subs_pos_prl[2] = { __tzcnt_u16(cbbv_col_v.v16[rooffin_subs_prp[0]] & base_x),
                                                                    __tzcnt_u16(cbbv_col_v.v16[rooffin_subs_prp[1]] & base_x) };
                            if (    rooffin_subs_pos_prl[0] != rooffin_subs_pos_prl[1]                // Case 2 B
                                 && _popcnt32(cbbv_col_v.v16[rooffin_subs_prp[0]] & base_x) == 1
                                 && _popcnt32(cbbv_col_v.v16[rooffin_subs_prp[1]] & base_x) == 1 ) {
                                if ( verbose != VNone ) {
                                    counters.fishes_detected++;
                                    counters.fishes_specials_detected++;
                                }
                                fincells[0] =  rooffin_subs_prp[0] + rooffin_subs_pos_prl[0]*9;
                                fincells[1] =  rooffin_subs_prp[1] + rooffin_subs_pos_prl[1]*9;
                                clean_bits.u128 =    (*(bit128_t*)big_index_lut[fincells[0]][All]).u128
                                                   & (*(bit128_t*)big_index_lut[fincells[1]][All]).u128;
                                // deal with 'addendum to Case 2'
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
                                    if ( verbose != VNone ) {
                                        counters.fishes_specials_updated++;
                                    }
                                    if ( verbose == VDebug ) {
                                        if ( skyscraper ) {
                                            subsx = subs_prp_x | (1<<rooffin_subs_prp[0]) | (1<<rooffin_subs_prp[1]);
                                            unsigned char cellif = __tzcnt_u16(subs_prp_x) + __tzcnt_u16(base_x)*9;
                                            unsigned char cellif2 = __tzcnt_u16(subs_prp_x) + (15-__lzcnt16(base_x))*9;
                                            if ( debug > 2 ) {
                                                show_skyscraper(cellif, cellif2, fincells[0], fincells[1], clean_bits, dgt_bits, solverData, "skyscraper (col)");
                                            }
                                            solverData.printf("skyscraper (col floor) digit %d floor at %s-%s roof at %s/%s\n   remove %d at ", dgt+1, cl2txt[cellif], cl2txt[cellif2], cl2txt[fincells[0]], cl2txt[fincells[1]], dgt+1);
                                        } else {
                                            celli = __tzcnt_u16(subsx) + __tzcnt_u16(base_x)*9;
                                            celli2 = (15-__lzcnt16(subsx)) + (15-__lzcnt16(base_x))*9;
                                            if ( debug > 2 ) {
                                               show_fish<true>(cbbv_col_v, subs_prp_x, 2, fincells, clean_bits, dgt_bits, solverData, "roofed col fish");
                                            }
                                            solverData.printf("roofed %s (rows) digit %d at cells %s - %s roof at %s,%s\n   remove %d at ", fish_names[cnt-2], dgt+1, cl2txt[celli], cl2txt[celli2], cl2txt[fincells[0]], cl2txt[fincells[1]], dgt+1);
                                        }
                                    }
                                } else {
                                    if ( __popcnt16(candidates[fincells[0]]) == 2 ) {
                                        solverData.guess_hint_digit = 1<<dgt;
                                        solverData.guess_hint_index = fincells[0];
                                    }
                                }
                            }
                        }

                        if ( clean_bits ) {
                            if ( verbose != VNone ) {
                                counters.fishes_updated++;
                            }
                            dgt_bits.u128 &= ~clean_bits.u128; // for Case 1 below
                            unsigned short dgt_mask_bit = 1<<dgt;
                            while (clean_bits) {
                                unsigned char cl = tzcnt_and_mask(clean_bits);
                                if ( candidates[cl] & dgt_mask_bit ) {
                                    candidates[cl] &= ~dgt_mask_bit;
                                    if ( verbose == VDebug ) {
                                        solverData.printf("%s ", cl2txt[cl]);
                                    }
                                }
                            }
                            if ( verbose == VDebug ) {
                                solverData.printf("\n");
                            }
                            dosearch = true;
                        }
                    }

                    {
                        // Case 1
                        bool skyscraper = false;
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
                            skyscraper = sashimi && alt_base_x==0 && cnt == 2 && __popcnt16(fins) == 1;

                            // roof coordinates for skyscraper:
                            unsigned char cellir {};
                            unsigned char cellir2 {};

                            // complete the finned fish pattern:
                            unsigned int subsx = subs_prp_x | (1<<fin_sub);

                            // the cols left for cleaning
                            unsigned short band_bits = bandbits_by_index[fin_sub/3];
                            unsigned int cols2clean = hassubs_prp_x & ~subsx & band_bits;

                            if ( cols2clean == 0 ) {
                                if ( !skyscraper ) {
                                    continue;
                                }
                                // the skyscraper base is a good guess:
                                if ( solverData.guess_hint_digit == 0 && !clean_bits && sashimi && cnt == 2 ) {
                                    solverData.guess_hint_digit = 1<<dgt;
                                    solverData.guess_hint_index = t + __tzcnt_u16(base_x & ~box_bits)*9;
                                }
                            }
                            // skyscraper is not singled out in statistics
                            if ( verbose != VNone ) {
                                counters.fishes_specials_detected++;
                                counters.fishes_detected++;
                            }

                            unsigned int boxbase_x = base_x & box_bits;

                            if ( skyscraper ) {
                                // col:
                                cellir = fin_sub + __tzcnt_u16(fins)*9;
                                cellir2 = t + __tzcnt_u16(box_bits & base_x)*9;
                                clean_bits.u128 =   *cast2cu128(big_index_lut[cellir][All])
                                                  & *cast2cu128(big_index_lut[cellir2][All]);
                                unsigned char cellirxor = (1<<(cellir/3%3)) ^ (1<<(cellir2/3%3));
                                if ( cellirxor ) { // not in same (vertical) band
                                    if (    _popcnt32(candidate_bits_by_value[dgt].get_indexbits(cellir/3*3, 3) & 0x7) == 1
                                         && _popcnt32(candidate_bits_by_value[dgt].get_indexbits(cellir2/3*3, 3) & 0x7) == 1) {
                                        // the roof candidates must not have 'siblings' in their respective boxes
                                        bit128_t base_row = { .u128= *cast2cu128(big_index_lut[__tzcnt_u16(base_x & ~box_bits)*9][Row]) & dgt_bits.u128 };
                                        if ( base_row.popcount() == 2 ) { // the skyscraper's floor is strongly linked
                                            unsigned char cellir3rd2 = _tzcnt_u32(7^((1<<(cellir/9%3))^(1<<(cellir2/9%3))));
                                            unsigned char cellir3rd = _tzcnt_u32(7^cellirxor);
                                            unsigned char boxoff = cellir/27*27+cellir3rd2*9;
                                            unsigned char rowoff = cellir3rd*3;
                                            clean_bits.set_indexbits(0x7, rowoff+boxoff, 3);
                                        }
                                    }
                                }
                            } else {
                                bit128_t tmp {};
                                while (cols2clean) {
                                    unsigned char col = tzcnt_and_mask(cols2clean);
                                    tmp.set_indexbits( cbbv_col_v.v16[col]&boxbase_x, col*9, 9);
                                }
                                // since we work in a transposed view, we transpose clean_bits first
                                while ( tmp ) {
                                    clean_bits.set_indexbit(transposed_cell[tzcnt_and_mask(tmp)]);
                                }
                            }
                            clean_bits.u128 &= dgt_bits.u128;
    
                            if ( clean_bits ) {
                                if ( verbose != VNone ) {
                                    counters.fishes_specials_updated++;
                                }
                                if ( verbose == VDebug ) {
                                    if ( skyscraper ) {
                                        unsigned char cellif = t + __tzcnt_u16(base_x&~box_bits)*9;
                                        unsigned char cellif2 = fin_sub + __tzcnt_u16(base_x&~box_bits)*9;
                                        if ( debug > 2 ) {
                                            show_skyscraper(cellif, cellif2, cellir, cellir2, clean_bits, dgt_bits, solverData, "skyscraper (row)");
                                        }
                                        solverData.printf("skyscraper (row floor) digit %d floor at %s-%s roof at %s/%s\n   remove %d at ", dgt+1, cl2txt[cellif], cl2txt[cellif2], cl2txt[cellir], cl2txt[cellir2], dgt+1);
                                    } else {
                                        if ( debug > 2 ) {
                                            show_fish<true>(cbbv_col_v, base_x, subsx, clean_bits, dgt_bits, solverData, "finned col fish");
                                        }
                                        celli = __tzcnt_u16(subsx) + __tzcnt_u16(base_x)*9;
                                        celli2 = (15-__lzcnt16(subsx)) + (15-__lzcnt16(base_x))*9;
                                        solverData.printf("finned%s %s (cols) digit %d at cells %s - %s\nFin at %s - remove %d at ", sashimi?"/sashimi":"", fish_names[cnt-2], dgt+1, cl2txt[celli], cl2txt[celli2], cl2txt[fin_sub+__tzcnt_u16(fins&~base_x)*9], dgt+1);
                                    }
                                }
                                break;
                            }
                        } // while
                    }  // Case 1
                }
                if ( clean_bits ) {
                    if ( verbose != VNone ) {
                        counters.fishes_updated++;
                    }
                    if ( clean_bits.check_indexbit(fincells[0]) ) {
                        e_digit = 1<<dgt;
                        e_i = fincells[1];
                    } else if ( clean_bits.check_indexbit(fincells[1]) ) {
                        e_digit = 1<<dgt;
                        e_i = fincells[0];
                    }
                    unsigned short dgt_mask_bit = 1<<dgt;
                    while (clean_bits) {
                        unsigned char cl = tzcnt_and_mask(clean_bits);
                        if ( candidates[cl] & dgt_mask_bit) {
                            candidates[cl] &= ~dgt_mask_bit;
                            if ( verbose == VDebug ) {
                                solverData.printf("%s ", cl2txt[cl]);
                            }
                        }
                    }
                    if ( verbose == VDebug ) {
                        if ( e_digit == 0 ) {
                            solverData.printf("\n");
                        } else {
                            solverData.printf("\n   cells %s and %s are mutually exclusive (%s%s on digit %d),\nenter the remaining ", cl2txt[fincells[0]], cl2txt[fincells[1]], cnt==2?"skyscraper":"roofed", cnt==2?"":fish_names[cnt-2], dgt+1);
                        }
                    }
                    if ( e_digit ) {
                        goto enter;
                    }
                    dosearch = true;
                }
                if ( dosearch ) {
                    goto search;
                }
            }
#ifdef COMBO_BASE
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
#endif
        } // for t
#endif
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

        if ( rowband ) {
            unsigned int bandbits = ((bit128_t*)unlocked)->get_indexbits(27*band, 27);

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
                unsigned long long ol64[2] = { original_locked.u64[0] & 0x7fffffffffffffff, original_locked.get_rshfti<63>() };
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
                            if ( verbose != VNone ) {
                                counters.unique_rectangles_avoided++;
                            }
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
                                }, solverData.counters, solverData.output);
                                goto guess_made_with_incr;
                            }
                            // otherwise simply avoid the UQR:
                            candidates[celli] &= ~single;
                            if ( verbose == VDebug ) {
                                solverData.printf("avoiding unique rectangle: %s %s - %s\n3 singles pattern: remove candidate %d from cell %s\n",
                                    ret, cl2txt[uqr_corners[0].indx], cl2txt[uqr_corners[2].indx],
                                    1+__tzcnt_u16(single), cl2txt[celli]);
                            }
                            if ( (candidates[celli] & (candidates[celli]-1)) == 0) {
                                e_i = celli;
                                e_digit = candidates[celli];
                                if ( verbose == VDebug ) {
                                    solverData.printf("naked  single      ");
                                }
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
                                solverData.printf("back track - found a completed unique rectangle %s at %s %s\n", ret, cl2txt[uqr_corners[0].indx], cl2txt[uqr_corners[2].indx]);
                            }
                            goto back;
                        } else {
                            // there's no point doing anything here... except:
                            if ( grid_state->stackpointer == 0 ) {
                                status.unique = false;
                            }
                            // not even:
                            // solverData.printf("consciously ignoring a completed unique rectangle at %s %s\n", cl2txt[uqr_corners[0].indx], cl2txt[uqr_corners[2].indx]);
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
                    if ( verbose != VNone ) {
                        counters.unique_rectangles_checked++;
                    }
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
                                if ( verbose != VNone ) {
                                    counters.unique_rectangles_avoided++;
                                }
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
                                    }, solverData.counters, solverData.output);
                                    goto guess_made_with_incr;
                                }
                                // simply avoid the UQR
                                candidates[corner4_index] &= ~pair;
                                if ( verbose == VDebug ) {
                                    char ret[32];
                                    format_candidate_set(ret, pair);
                                    solverData.printf("avoiding unique rectangle: %s %s - %s\n3 pairs pattern: remove candidates %s from %s\n",
                                            ret, cl2txt[uqr_corners[0].indx], cl2txt[uqr_corners[2].indx],
                                            ret, cl2txt[corner4_index]);
                                }
                                grid_state->updated.set_indexbit(corner4_index);
                                if ( (candidates[corner4_index] & (candidates[corner4_index]-1)) == 0 ) {
                                    e_i = corner4_index;
                                    e_digit = candidates[corner4_index];
                                    if ( verbose == VDebug ) {
                                        solverData.printf("naked  single      ");
                                    }
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
                                                    solverData.printf("back track - found an unavoidable unique rectangle %s at %s %s\n", ret, cl2txt[uqr_corners[0].indx], cl2txt[uqr_corners[2].indx]);
                                                }
                                                goto back;
                                            } else {
                                                // there's no point doing anything here...
                                                // not even:
                                                // solverData.printf("consciously ignoring a completed unique rectangle at %s %s\n", cl2txt[uqr_corners[0].indx], cl2txt[uqr_corners[2].indx]);
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
                                            if ( verbose != VNone ) {
                                                counters.unique_rectangles_avoided++;
                                            }
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
                                                }, solverData.counters, solverData.output);
                                                goto guess_made_with_incr;
                                            }
                                            // simply avoid the UQR
                                            candidates[weak_corner_indx] &= ~weak_corner_y;
                                            if ( verbose == VDebug ) {
                                                solverData.printf("avoiding unique rectangle: %s %s - %s\n2 pair pattern: remove candidate %d at %s\n",
                                                       ret, cl2txt[uqr_corners[0].indx], cl2txt[uqr_corners[2].indx],
                                                       1+__tzcnt_u16(weak_corner_y), cl2txt[weak_corner_indx]);
                                            }
                                            if ( (candidates[weak_corner_indx] & (candidates[weak_corner_indx]-1)) == 0 ) {
                                                e_i = weak_corner_indx;
                                                e_digit = candidates[weak_corner_indx];
                                                if ( verbose == VDebug ) {
                                                    solverData.printf("naked  single      ");
                                                }
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
                                unsigned int ix = 0xf & (uqr_pairs[pi].crnrs | uqr_pairs[pi].crnrs<<4)>>2;
                                // from x and the pair of digits, determine y:

                                if ( uqr_cand != uqr_pairs[pi].digits ) {
                                    uqr_cand ^= uqr_pairs[pi].digits;
                                }
                                unsigned char indx2upd[2];
                                for ( int k=0; k<2; k++ ) {
                                    unsigned char indx = _tzcnt_u32(ix);
                                    indx2upd[k] = uqr_corners[indx].indx;
                                    ix = _blsr_u32(ix);
                                }
                                if (    candidates[indx2upd[0]] != uqr_cand
                                     || candidates[indx2upd[1]] != uqr_cand ) {
                                    char ret[32];
                                    if ( verbose == VDebug ) {
                                        format_candidate_set(ret, uqr_pairs[pi].digits);
                                    }
                                    found_update = true;
                                    if ( verbose != VNone ) {
                                        counters.unique_rectangles_avoided++;
                                    }
                                    if ( rules == Regular ) {
                                        candidates[indx2upd[0]] &= ~uqr_cand;
                                        candidates[indx2upd[1]] &= ~uqr_cand;
                                        if ( verbose == VDebug ) {
                                            solverData.printf("avoiding unique rectangle: %s %s - %s\n2 pair pattern: remove candidate %d from cells %s %s\n",
                                                   ret, cl2txt[uqr_corners[0].indx], cl2txt[uqr_corners[2].indx],
                                                   1+__tzcnt_u16(uqr_cand),
                                                   cl2txt[indx2upd[0]], cl2txt[indx2upd[1]]);
                                        }
                                        if (    (candidates[indx2upd[0]] & (candidates[indx2upd[0]] - 1)) == 0 ) {
                                            e_digit = candidates[indx2upd[0]];
                                            e_i = indx2upd[0];
                                            if ( verbose == VDebug ) {
                                                solverData.printf("naked  single      ");
                                            }
                                            goto enter;
                                        } else if ( (candidates[indx2upd[1]] & (candidates[indx2upd[1]] - 1)) == 0 ) {
                                            e_digit = candidates[indx2upd[1]];
                                            e_i = indx2upd[1];
                                            if ( verbose == VDebug ) {
                                                solverData.printf("naked  single      ");
                                            }
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
                                                        }, solverData.counters, solverData.output);
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
                                                        }, solverData.counters, solverData.output);
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
                                    // plug the index bit hole (or prove that to be unnecessary)
                                    cand_removal_indx.u128 =  *cast2cu128(big_index_lut[non_pairs_celli[0]][All]);
                                    cand_removal_indx.u128 &= *cast2cu128(big_index_lut[non_pairs_celli[1]][All]);
                                    cand_removal_indx.u128 &= candidate_bits_by_value[__tzcnt_u16(non_uqr_cands)].u128;
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
                                            cand_visibility_indx.u128 = *cast2cu128(big_index_lut[non_pairs_celli[0]][is_row?Row:Col]);
                                            check_set = true;
                                        }

                                        // box based
                                        unsigned short b = box_start[non_pairs_celli[0]];
                                        if ( b == box_start[non_pairs_celli[1]] ) {
                                            __m256i c = _mm256_set_epi16(0,0,0,0,0,0,0,candidates[b+20], candidates[b+19], candidates[b+18], candidates[b+11], candidates[b+10], candidates[b+9], candidates[b+2], candidates[b+1], candidates[b]);
                                            m = compress_epi16_boolean<true>(_mm256_cmpeq_epi16(a, _mm256_or_si256(a, c)));
                                            if ( (_popcnt32(m & 0x3ffff)>>1) == (non_uqr_cands_cnt-1) ) {
                                                cand_visibility_indx.u128 |= *cast2cu128(big_index_lut[non_pairs_celli[0]][Box]);
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
                                    if ( verbose != VNone ) {
                                        counters.unique_rectangles_avoided++;
                                    }
                                    found_update = true;
                                    if ( verbose == VDebug ) {
                                        char ret[32];
                                        char ret2[32];
                                        format_candidate_set(ret, uqr_pairs[pi].digits);
                                        format_candidate_set(ret2, non_uqr_cands);
                                        solverData.printf("avoiding unique rectangle: %s %s - %s\n2 pair pattern: remove candidates %s from cells outside UR at: ",
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
                                                solverData.printf("%s%s", cma, cl2txt[j]);
                                                cma = ",";
                                            }
                                        }
                                    }
                                    if ( verbose == VDebug ) {
                                        solverData.printf("\n");
                                    }
                                    if ( got_single ) {
                                        goto search;
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
                                    if ( verbose != VNone ) {
                                        counters.unique_rectangles_avoided++;
                                    }
                                    grid_state->updated.set_indexbit(indx);
                                    // unless under 'Regular' rules, capture the avoidable UQR
                                    // provide 'resolution' in form of a guess
                                    if ( rules == Regular ) {
                                        // simply avoid the UQR
                                        candidates[indx] &= ~uqr_alt_cand;
                                        if ( verbose == VDebug ) {
                                            char ret[32];
                                            format_candidate_set(ret, uqr_pairs[pi].digits);
                                            solverData.printf("avoiding unique rectangle: %s %s - %s\n1 pair pattern: remove candidate %d from cell %s\n",
                                                    ret, cl2txt[uqr_corners[0].indx], cl2txt[uqr_corners[2].indx],
                                                    1+__tzcnt_u16(uqr_alt_cand), cl2txt[indx]);
                                        }
                                        found_update = true;
                                        if ( (candidates[indx] & (candidates[indx]-1)) == 0 ) {
                                            e_i = indx;
                                            e_digit = candidates[indx];
                                            if ( verbose == VDebug ) {
                                                solverData.printf("naked  single      ");
                                            }
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
                                                }, solverData.counters, solverData.output);
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
            goto search;
        };
    }
}
#endif

guess:
    // Make a guess if all that didn't work
    grid_state = grid_state->make_guess<verbose>(&solverData);
    no_guess_incr = 0;
#ifdef OPT_UQR
// if the guess was made solely to allow for checking uniqueness, still count the solution
// as direct solve
guess_made_with_incr:
#endif
    current_entered_count  = (grid_state->stackpointer<<8) | (81 - grid_state->unlocked.popcount());
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
    -m[FSU]* execution modes (fishes, sets, unique rectangles), lower or upper case
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

    Several different fish types and their eliminations are identified in the debug log:
       - Finned fishes are directly identified and cleaned to their final form.
       - Finned/sashimi fishes eliminate candidates based on identified fin(s).
       - Roofed fishes are comprised of two extra lines of candidates alternatively
         completing a fish pattern, eliminating candidates in their intersection of views.
       - Skyscraper is one specific finned/sashimi (or finned/sashimi) X-wing pattern counted as
         a fish, but labeled as a 'skyscraper' in closer correspondence to the
         conventions used elsewhere and the eliminations it produces.
    Fishes are always labelled with orientation (row or col) due to their lines
    being constrained in that direction. Fins are always orientated in that orientation,
    while eliminations are made orthogonal to that orientation.
    For skyscrapers that shown orientation denotes the orientation of the skyscrapers' base
    (when looking for the fish pattern in a skyscraper, replace rows with cols.)
    Fishes details can be shown as a 9x9 grid with '-d3' where 
       'o' represents fish positions as 'o'
       '@' represents fin(s) or roofed completions, if present
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

        if ( debug != 0) {
            stack[0].initialize<VDebug>(output, global_counters);
            solve<VDebug>(grid, stack, line_to_solve, global_counters);
        } else if ( reportstats !=0) {
            stack[0].initialize<VStats>(output, global_counters);
            solve<VStats>(grid, stack, line_to_solve, global_counters);
        } else {
            stack[0].initialize<VNone>(output, global_counters);
            solve<VNone>(grid, stack, line_to_solve, global_counters);
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

#pragma omp declare reduction (counters_reduction : Counters : omp_out += omp_in)

#pragma omp parallel reduction(counters_reduction:global_counters) firstprivate(stack, memstream) proc_bind(close) shared(string_pre, output, npuzzles, imax, debug, reportstats, numthreads)
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
#pragma omp for schedule(monotonic:dynamic,120)
            for (size_t i = 0; i < imax; i+=82) {
                // copy unsolved grid
                signed char *grid = &output[i*2+82];
                memcpy(&output[i*2], &string_pre[i], 81);
                // add comma and newline in right place
                output[i*2 + 81] = ',';
                output[i*2 + 163] = 10;
                // solve the grid in place
                if ( debug != 0) {
                    stack[0].initialize<VDebug>(&output[i*2], global_counters);
                    solve<VDebug>(grid, stack, i/82+1, global_counters, memstream->outf);
                } else if ( reportstats !=0) {
                    stack[0].initialize<VStats>(&output[i*2], global_counters);
                    solve<VStats>(grid, stack, i/82+1, global_counters, memstream->outf);
                } else {
                    stack[0].initialize<VNone>(&output[i*2], global_counters);
                    solve<VNone>(grid, stack, i/82+1, global_counters, memstream->outf);
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
	err = munmap(output, (size_t)npuzzles*164);
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
#ifdef OPT_SETS
        if ( mode_sets ) {
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

    return 0;
}
#endif
