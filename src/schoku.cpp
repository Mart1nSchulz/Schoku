/// This code uses AVX2 instructions...
/*
 * Schoku
 *
 * A high speed sudoku solver by M. Schulz
 *
 * Based on the sudoku solver by Mirage ( https://codegolf.stackexchange.com/users/106606/mirage )
 * at https://codegolf.stackexchange.com/questions/190727/the-fastest-sudoku-solver
 * on Sep 22, 2021
 *
 * Version 0.9
 *
 * Performance changes:
 * - implementation of an improved guessing algorithms, based on triads.
 *
 * Functional changes:
 * (none)
 *
 * Performance measurement and statistics:
 *
 * data: 17-clue sudoku (49151 puzzles)
 * CPU:  Ryzen 7 4700U
 *
 * schoku version: 0.9
 * compile options:
 *     49151  puzzles entered
 *     49151  1955753/s  puzzles solved
 *    25.1ms    0.51µs/puzzle  solving time
 *     37834   76.98%  puzzles solved without guessing
 *     30798    0.63/puzzle  guesses
 *     20996    0.43/puzzle  back tracks
 *    273758    5.57/puzzle  digits entered and retracted
 *   1464091   29.79/puzzle  'rounds'
 *     22972    0.47/puzzle  triads resolved
 *    197772    4.02/puzzle  triad updates
 *       693  bi-value universal graves detected
 *
 * compile options: OPT_TRIAD_RES OPT_SETS
 *     49151  puzzles entered
 *     49151  1776720/s  puzzles solved
 *    26.5ms    0.54µs/puzzle  solving time
 *     42086   85.63%  puzzles solved without guessing
 *     12406    0.25/puzzle  guesses
 *      7446    0.15/puzzle  back tracks
 *     98305    2.00/puzzle  digits entered and retracted
 *   1397333   28.43/puzzle  'rounds'
 *    106083    2.16/puzzle  triads resolved
 *    201503    4.10/puzzle  triad updates
 *     17633    0.36/puzzle  naked sets found
 *    915357   18.62/puzzle  naked sets searched
 *       952  bi-value universal graves detected
 *
 */
#include <atomic>
#include <chrono>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <unistd.h>
#include <assert.h>
#include <fcntl.h>
#include <omp.h>
#include <stdbool.h>
#include <intrin.h>
#include <sys/stat.h>
#include <sys/mman.h>

const char *version_string = "0.9";

const char *compilation_options =
// Options OPT_SETS and OPT_TRIAD_RES compete to some degree, but they also perform well
// together for excellent statistics.
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

// bit128_t type
// used for all 81-bit fields to support different access patterns.
typedef
union bit128_t {
    __uint128_t    u128;
    __m128i        m128;
    unsigned long  long u64[2];
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
} bit128_t;

// alignment helper construct
// The goal is to associate larger data structures with as few cache lines as possible.
// The secondary goal is to not mix heavily used and lesser used data on the same cache line.
typedef struct alignas(64) {} align64_empty;

const align64_empty c1;
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

const align64_empty c2;
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

const align64_empty c3;
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

const align64_empty c4;
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

const align64_empty c5;
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

const unsigned char *row_index = index_by_kind[Row];
const unsigned char *column_index = index_by_kind[Col];
const unsigned char *box_index = index_by_kind[Box];

const align64_empty c8;
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

unsigned short bitx3_lut[8] = {
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
align64_empty c9;
std::atomic<long long> past_naked_count(0); // how often do we get past the naked single serach
std::atomic<long long> digits_entered_and_retracted(0); // to measure guessing overhead
std::atomic<long long> triads_resolved(0);  // how many triads did we resolved
std::atomic<long long> triad_updates(0);    // how many triads did cancel candidates
std::atomic<long long> naked_sets_searched(0); // how many naked sets did we search for
std::atomic<long long> naked_sets_found(0); // how many naked sets did we actually find
// somehow this padding is (very) beneficial, so leave it in...
std::atomic<long long> unique_rectangles_checked(0);   // how many unique rectangles were checked
std::atomic<long long> unique_rectangles_avoided(0);   // how many unique rectangles were avoided
std::atomic<long long> fishes_detected(0);  // how many unique rectangles were checked
std::atomic<long long> fishes_specials_detected(0);    // how many unique rectangles were checked
std::atomic<long long> fishes_updated(0);              // how many unique rectangles were avoided
std::atomic<long long> fishes_specials_updated(0);     // how many unique rectangles were avoided
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

// stats and command line options
int reportstats     = 0; // collect and report some statistics
int verify          = 0; // verify solution correctness (implied otherwise)
int unique_check    = 0; // check solution uniqueness
int debug           = 0; // provide step by step output on the solution
int thorough_check  = 0; // check for back tracking even if no guess was made.
int numthreads      = 0; // if not 0, number of threads

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

inline __m256i expand_bitvector(unsigned short m) {
    const __m256i bit_mask = _mm256_setr_epi16(1<<0, 1<<1, 1<<2, 1<<3, 1<<4, 1<<5, 1<<6, 1<<7, 1<<8, 1<<9, 1<<10, 1<<11, 1<<12, 1<<13, 1<<14, 1<<15);
    return _mm256_cmpeq_epi16(_mm256_and_si256( bit_mask,_mm256_set1_epi16(m)), bit_mask);
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

template<Kind kind>
inline unsigned char get_section_index_cnt(unsigned long long indices[2], unsigned char si) {
    const unsigned long long *idx = small_index_lut[si][kind];
    return _popcnt64(indices[0] & idx[0])
         + _popcnt32(indices[1] & idx[1]);
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

inline void dump_m256i(__m256i x, const char *msg="") {
    printf("%s %llx,%llx,%llx,%llx\n", msg, ((__v4du)x)[0],((__v4du)x)[1],((__v4du)x)[2],((__v4du)x)[3]);
}

inline void dump_m128i(__m128i x, const char *msg="") {
    printf("%s %llx,%llx\n", msg, ((__v2du)x)[0],((__v2du)x)[1]);
}

// a helper function to print a grid of bits.
// the bits are arranged as 9 bits each in 9 unsigned short elements of a __m256i parameter.
//
inline void dump_m256i_grid(__m256i v) {
    for (unsigned char r=0; r<9; r++) {
        unsigned short b = ((__v16hu)v)[r];
        for ( unsigned char i=0; i<9; i++) {
            printf("%s", (b&(1<<i))? "x":"-");
            if ( i%3 == 2 ) {
                printf(" ");
            }
            if ( i%9 == 8 ) {
                printf("\n");
            }
        }
    }
}

// a helper function to print a grid of bits.
// bits are arranged consecutively as 81 bits in a __unint128_t parameter.
//
inline void dump_bits(__uint128_t bits, const char *msg="") {
    printf("%s:\n", msg);
    for ( int i_=0; i_<81; i_++ ) {
        printf("%s", (*(bit128_t*)&bits).check_indexbit(i_)?"x":"-");
        if ( i_%3 == 2 ) {
            printf(" ");
        }
        if ( i_%9 == 8 ) {
            printf("\n");
        }
    }
}

inline void format_candidate_set(char *ret, unsigned short candidates);

// a helper function to print a sudoku board,
// given the 81 cells solved or with candidates.
//
inline void dump_board(unsigned short *candidates, const char *msg="") {
    printf("%s:\n", msg);
    for ( int i_=0; i_<81; i_++ ) {
        char ret[32];
        format_candidate_set(ret, candidates[i_]);
        printf("%8s,", ret);
        if ( i_%9 == 8 ) {
            printf("\n");
        }
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

// The maximum levels of guesses is given by GRIDSTATE_MAX.
// On average, the number of levels is pretty low, but we want to be 'reasonably' sure
// that we don't bust the envelope.
// The other 'property' of the GRIDSTATE_MAX constant is to keep the L2 cache happy.
// 28 is a good choice either way.
//
#define	GRIDSTATE_MAX 28

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
    unsigned int filler[1];           // align on 16 bytes
    bit128_t unlocked;                // for keeping track of which cells still need to be resolved. Set bits correspond to cells that still have multiple possibilities
    bit128_t updated;                 // for keeping track of which cell's candidates may have been changed since last time we looked for naked sets. Set bits correspond to changed candidates in these cells
    bit128_t set23_found[3];          // for keeping track of found sets of size 2 and 3

// GridState is normally copied for recursion
// Here we initialize the starting state including the puzzle.
//
inline void initialize(signed char grid[81]) {
    // 0x1ffffffffffffffffffffULLL is (0x1ULL << 81) - 1
    unlocked.u128 = (((__uint128_t)1)<<81)-1;
    updated.u128  = (((__uint128_t)1)<<81)-1;

    triads_unlocked[0] = triads_unlocked[1] = 0x1ffLL | (0x1ffLL<<10) | (0x1ffLL<<20);
    set23_found[0] = set23_found[1] = set23_found[2] = {__int128 {0}};

    stackpointer = 0;

    signed short digit;
    unsigned short columns[9] = {0};
    unsigned short rows[9]    = {0};
    unsigned short boxes[9]   = {0};

        for (unsigned char i = 0; i < 64; ++i) {
            digit = grid[i] - 49;
            if (digit >= 0) {
                digit = 1 << digit;
                columns[column_index[i]] |= digit;
                rows[row_index[i]]       |= digit;
                boxes[box_index[i]]      |= digit;
                _bittestandreset64((long long int *)unlocked.u64, i);
            }
        }

        for (unsigned char i = 64; i < 81; ++i) {
            digit = grid[i] - 49;
            if (digit >= 0) {
                digit = 1 << digit;
                columns[column_index[i]] |= digit;
                rows[row_index[i]]       |= digit;
                boxes[box_index[i]]      |= digit;
                _bittestandreset64((long long int *)&unlocked.u64[1], i-64);
            }
        }

    for (unsigned char i = 0; i < 81; ++i) {
        digit = grid[i] - 49;
        if (digit >= 0) {
            candidates[i] = 1 << digit;
        } else {
            candidates[i] = 0x01ff ^ (rows[row_index[i]] | columns[column_index[i]] | boxes[box_index[i]]);
        }
    }
}

// Normally digits are entered by a 'goto enter;'.
// enter_digit is not used in that case.
// Only make_guess uses this member function.
protected:
template<bool verbose=false>
inline __attribute__((always_inline)) void enter_digit( unsigned short digit, unsigned char i) {
    // lock this cell and and remove this digit from the candidates in this row, column and box

    bit128_t to_update = {0};

    if ( verbose && debug ) {
        printf(" %x at %s\n", _tzcnt_u32(digit)+1, cl2txt[i]);
    }
#ifndef NDEBUG
    if ( __popcnt16(digit) != 1 ) {
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
    const __m256i mask = _mm256_set1_epi16(~digit);
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
template<bool verbose>
inline GridState* make_guess(TriadInfo &triad_info, bit128_t bivalues) {
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

    if ( verbose && debug ) {
        printf("guess at level >%d< - new level >%d<\n", stackpointer, new_grid_state->stackpointer);
        printf("guess remove {%d} from %s triad at %s\n",
               1+_tzcnt_u32(select_cand), type==0?"row":"col", cl2txt[off]);
    }
    if ( verbose ) {
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
template<bool verbose>
GridState* make_guess(bit128_t bivalues) {
    // Find a cell with the least candidates. The first cell with 2 candidates will suffice.
    // Pick the candidate with the highest value as the guess.
    // Save the current grid state (with the chosen candidate eliminated) for tracking back.

    // Find the cell with fewest possible candidates
    unsigned long long to_visit;
    unsigned char guess_index = 0;
    unsigned char i_rel;
    unsigned char cnt;
    unsigned char best_cnt = 16;

    if ( bivalues.u64[0] ) {
        guess_index = _tzcnt_u64(bivalues.u64[0]);
    } else if ( bivalues.u64[1] ) {
        guess_index = _tzcnt_u64(bivalues.u64[1]) + 64;
    } else {
        // very unlikely
        to_visit = unlocked.u64[0];
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

    if ( verbose && (debug > 1) ) {
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
    if ( verbose && debug ) {
        printf("guess at level >%d< - new level >%d<\nguess", stackpointer, new_grid_state->stackpointer);
    }

    new_grid_state->enter_digit<verbose>( digit, guess_index);
    guesses++;

    if ( verbose && (debug > 1) ) {
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

template<Kind kind>
inline unsigned char get_ul_set_search( unsigned char si) {
    return 9-get_section_index_cnt<kind>(set23_found[kind].u64, si);
}
};

template <bool verbose>
bool solve(signed char grid[81], GridState stack[], int line) {

    GridState *grid_state = &stack[0];
    unsigned long long *unlocked = grid_state->unlocked.u64;
    unsigned short* candidates;

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
    unsigned char no_guess_incr = 1;

    unsigned int my_past_naked_count = 0;

    if ( verbose && debug ) {
        printf("Line %d: %.81s\n", line, grid);
    }

    // The 'API' for code that uses the 'goto enter:' method of entering digits
    unsigned short e_digit = 0;
    unsigned char e_i = 0;

    bool check_back = thorough_check;
    goto start;

back:

    // Each algorithm (naked single, hidden single, naked set)
    // has its own non-solvability detecting trap door to detect the grid is bad.
    // This section acts upon that detection and discards the current grid_state.
    //
    if (grid_state->stackpointer == 0) {
        if ( unique_check_mode ) {
            if ( verbose && debug ) {
                // no additional solution exists
                printf("No secondary solution found during back track\n");
            }
        } else {
            // This only happens when the puzzle is not valid
            // Bypass the verbose check...
            printf("Line %d: No solution found!\n", line);
            unsolved_count++;
        }
        // cleanup and return
        if ( verbose && reportstats ) {
            past_naked_count += my_past_naked_count;
            naked_sets_searched += my_naked_sets_searched;
            digits_entered_and_retracted += my_digits_entered_and_retracted;
        }
        return true;
    }

    current_entered_count  = ((grid_state-1)->stackpointer)<<8;        // back to previous stack.
    current_entered_count += _popcnt64((grid_state-1)->unlocked.u64[0]) + _popcnt32((grid_state-1)->unlocked.u64[1]);

    // collect some guessing stats
    if ( verbose && reportstats ) {
        my_digits_entered_and_retracted +=
            (_popcnt64((grid_state-1)->unlocked.u64[0] & ~grid_state->unlocked.u64[0]))
          + (_popcnt32((grid_state-1)->unlocked.u64[1] & ~grid_state->unlocked.u64[1]));
    }

    // Go back to the state when the last guess was made
    // This state had the guess removed as candidate from it's cell

    if ( verbose && debug ) {
        printf("back track to level >%d<\n", grid_state->stackpointer-1);
    }
    trackbacks++;
    grid_state--;

start:

    e_digit = 0;

    // at start, set everything that depends on grid_state:
    check_back = grid_state->stackpointer || thorough_check || unique_check_mode;

    unlocked   = grid_state->unlocked.u64;
    candidates = grid_state->candidates;

    const __m256i ones = _mm256_set1_epi16(1);

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

        if ( verbose && debug ) {
            printf(" %x at %s\n", _tzcnt_u32(e_digit)+1, cl2txt[e_i]);
        }
#ifndef NDEBUG
        if ( __popcnt16(e_digit) != 1 ) {
            printf("error in e_digit: %x\n", e_digit);
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

        const __m256i mask = _mm256_set1_epi16(~e_digit);

        unsigned short dtct_j = 0;
        unsigned int dtct_m = 0;
        for (unsigned char j = 0; j < 80; j += 16) {
            __m256i c = _mm256_load_si256((__m256i*) &candidates[j]);
            // expand locked unsigned short to boolean vector
            __m256i mlocked = expand_bitvector(~to_update.u16[j>>4]);
            // apply mask (remove bit), preserving the locked cells
            c = and_unless(c, mask, mlocked);
            _mm256_store_si256((__m256i*) &candidates[j], c);
            __m256i a = _mm256_cmpeq_epi16(_mm256_and_si256(c, _mm256_sub_epi16(c, ones)), _mm256_setzero_si256());
            // this if is only taken very occasionally, branch prediction
            if (__builtin_expect (check_back && _mm256_movemask_epi8(
                                  _mm256_cmpeq_epi16(c, _mm256_setzero_si256())
                                  ), 0)) {
                // Back track, no solutions along this path
                if ( verbose ) {
                    unsigned int mx = _mm256_movemask_epi8(_mm256_cmpeq_epi16(c, _mm256_setzero_si256()));
                    unsigned char pos = j+(_tzcnt_u32(mx)>>1);
                    if ( grid_state->stackpointer == 0 && unique_check_mode == 0 ) {
                        printf("Line %d: cell %s is 0\n", line, cl2txt[pos]);
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
                if ( verbose && debug ) {
                    printf("naked  single      ");
                }
                goto enter;
            } else {
                if ( verbose ) {
                    unsigned char pos = e_i;
                    if ( grid_state->stackpointer == 0 && unique_check_mode == 0 ) {
                        printf("Line %d: cell %s is 0\n", line, cl2txt[pos]);
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
                    if ( verbose ) {
                        unsigned int mx = _mm256_movemask_epi8(_mm256_cmpeq_epi16(c, _mm256_setzero_si256()));
                        unsigned char pos = i+(_tzcnt_u32(mx)>>1);
                        if ( grid_state->stackpointer == 0 && unique_check_mode == 0 ) {
                            printf("Line %d: cell %s is 0\n", line, cl2txt[pos]);
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
                    if ( verbose && debug ) {
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
            if ( verbose ) {
                if ( grid_state->stackpointer == 0 && unique_check_mode == 0 ) {
                    printf("Line %d: cell[8,8] is 0\n", line);
                } else {
                    if ( debug ) {
                        printf("back track - cell [8,8] is 0\n");
                    }
                }
            }
            goto back;
        } else if (__popcnt16(candidates[80]) == 1) {
            // Enter the digit and update candidates
            if ( verbose && debug ) {
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
        if ( unique_check == 1 && unique_check_mode == 1 ) {
            if ( !nonunique_reported ) {
                if ( verbose && reportstats ) {
                    printf("Line %d: solution to puzzle is not unique\n", line);
                }
                non_unique_count++;
                nonunique_reported = true;
                verify_one = true;
            }
        }
        if ( verify || verify_one) {
            verify_one = false;
            // quickly assert that the solution is valid
            // no cell has more than one digit set
            // all rows, columns and boxes have all digits set.

            const __m256i mask9 { -1LL, -1LL, 0xffffLL, 0 };
            const __m256i ones = _mm256_and_si256(_mm256_set1_epi16(1), mask9);
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

                uniq = _mm256_or_si256(_mm256_and_si256(col, _mm256_sub_epi16(col, ones)),uniq);

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
                if ( unique_check_mode == 0 ) {
                    if ( verbose ) {
                           printf("Line %d: solution to puzzle failed verification\n", line);
                    }
                    unsolved_count++;
                    not_verified_count++;
                } else {     // not supposed to get here
                    if ( verbose ) {
                        printf("unique check: not a valid solution\n");
                    }
                }
            } else  if ( verbose ) {
               if ( debug ) {
                   printf("Solution found and verified\n");
               }
               if ( reportstats ) {
                   solved_count++;
                   verified_count++;
               }
           }
        } else if ( verbose && reportstats ) {
           solved_count++;
        }

        // Enter found digits into grid (unless we already had a solution)
        if ( unique_check_mode == 0 ) {
            for (unsigned char j = 0; j < 81; ++j) {
                grid[j] = 49+_tzcnt_u32(candidates[j]);
            }
        }
        if ( verbose && reportstats ) {
            no_guess_cnt += no_guess_incr;
        }

        if ( unique_check == 1 ) {
            if ( grid_state->stackpointer ) {
                if ( verbose && debug && unique_check_mode) {
                    printf("back track during unique check (OK)\n");
                }
               if ( debug && unique_check_mode == 0 ) {
                   printf("Solution: %.81s\nBack track to determine uniqueness\n", grid);
               }
                unique_check_mode = 1;
                goto back;
            }
            // otherwise uniqueness is verified
        }
        if ( verbose && reportstats ) {
            past_naked_count += my_past_naked_count;
            naked_sets_searched += my_naked_sets_searched;
            digits_entered_and_retracted += my_digits_entered_and_retracted;
        }
        return true;
    }

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
    // - same for the rows and box
    //
    // Combine 8 cells from rows and 8 cells from boxes into one __m256i vector.
    // Rotate and or until each vector element represents 7 cells or'ed (except the cell
    // directly corresponding to its position, containing the hidden single if there is one).
    // Broadcast the nineth cell and or it for good measure, then andnot with 0x1ff
    // to isolate the hidden singles.
    // For the nineth cell, rotate and or one last time and use just one element of the result
    // to check the nineth cell for a hidden single.
    //
    // The column singles and the row singles are first or'ed and checked to be singles.
    // Otherwise if the row and column checks disagree on a cell the last guess was wrong.
    // If everything is in order, compare the singles (if any) against the current row.
    // Compress to a bit vector and check against the unlocked cells. If any single was found
    // isolate it, rince and repeat.
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

    TriadInfo triad_info;

    const __m256i mask9 { 0x1ff01ff01ff01ffLL, 0x1ff01ff01ff01ffLL, 0x1ffLL, 0 };

    // Algo 2 and Algo 3.1
    {
        const __m256i mask11hi { 0LL, 0LL, 0xffffLL<<48, ~0LL };
        const __m256i mask1ff { 0x1ff01ff01ff01ffLL, 0x1ff01ff01ff01ffLL, 0x1ff01ff01ff01ffLL, 0x1ff01ff01ff01ffLL };

        __m256i column_or_tails[9];
        __m256i column_or_head = _mm256_setzero_si256();
        __m256i col_triads_3, col_triads_2;
        __m256i column_mask {};
        unsigned char irow = 0;

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
        if ( check_back && !_mm256_testz_si256(mask9,_mm256_andnot_si256(_mm256_or_si256(column_mask, *(__m256i*) &candidates[0]), mask9)) ) {
            // the current grid has no solution, go back
            if ( verbose ) {
                unsigned int m = _mm256_movemask_epi8(_mm256_cmpgt_epi16(_mm256_andnot_si256(_mm256_or_si256(column_mask, *(__m256i*) &candidates[0]), mask9), _mm256_setzero_si256()));
                int idx = __tzcnt_u32(m)>>1;
                if ( grid_state->stackpointer == 0 && unique_check_mode == 0 ) {
                    printf("Line %d: stack 0, back track - column %d does not contain all digits\n", line, idx);
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
            __m256i column_mask_neg = _mm256_andnot_si256(column_mask_, mask9);
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
                    if ( verbose ) {
                        if ( grid_state->stackpointer == 0 && unique_check_mode == 0 ) {
                            printf("Line %d: stack 0, back track - col cell %s does contain multiple hidden singles\n", line, cl2txt[e_i]);
                        } else if ( debug ) {
                            printf("back track - multiple hidden singles in col cell %s\n", cl2txt[e_i]);
                        }
                    }
                    e_i = 0;
                    goto back;
                }
                if ( verbose && debug ) {
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

        __v16hu rowbox_9th_mask;

        for (unsigned char i = 0; i < 81; i += 9, irow++) {
            // turn the or'ed rows into a mask for the singletons, if any.
            column_mask = _mm256_andnot_si256(column_mask, mask9);

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
                        if ( verbose ) {
                            unsigned int m = _mm256_movemask_epi8(_mm256_cmpeq_epi16(_mm256_setzero_si256(),
                            _mm256_andnot_si256(_mm256_or_si256(rowbox_9th, rowbox_or8), mask1ff)));
                            const char *row_or_box = (m & 0xffff)?"box":"row";
                            if ( grid_state->stackpointer == 0 && unique_check_mode == 0 ) {
                                printf("Line %d: stack 0, back track - %s %d does not contain all digits\n", line, row_or_box, irow);
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
                    if ( verbose ) {
                        unsigned int m = _mm256_movemask_epi8(_mm256_cmpgt_epi16(_mm256_and_si256(rowbox_mask,
                                         _mm256_sub_epi16(rowbox_mask, ones)), _mm256_setzero_si256()));
                        const char *row_or_box = (m & 0xffff)?"row":"box";
                        int idx = __tzcnt_u32(m)>>1;
                        if ( grid_state->stackpointer == 0 && unique_check_mode == 0 ) {
                            printf("Line %d: stack 0, back track - %s cell %s does contain multiple hidden singles\n", line, row_or_box, cl2txt[irow*9+idx%8]);
                        } else if ( debug ) {
                            printf("back track - multiple hidden singles in %s cell %s\n", row_or_box, cl2txt[irow*9+idx%8]);
                        }
                    }
                    goto back;
                }

#if 0
// limited value?
                // the 9th candidate singleton in row
                unsigned short cand9_row = ~((v16us)rowbox_or8)[0] & the9thcand_row;
                // Combine the rowbox_mask and column_mask (9 elements)
                // this absorbs cand9_row.
                __m256i or_mask = _mm256_insert_epi16(rowbox_mask,cand9_row,8);
                or_mask = _mm256_and_si256(_mm256_or_si256(or_mask, column_mask),mask9);

                // If row mask and column_mask conflict, back track.
                // the row and column masks can each detect multiple singles,
                // possibly even the same single or singles.
                // There are two reasons why there can be multiple 'hidden singletons'
                // in a cell:
                // Either the row and column detected singles do not agree, or
                // the row or column have multiple singles in that same cell.
                // The reason is the same: a wrong guess.
                //
                if ( check_back && !_mm256_testz_si256(or_mask, _mm256_sub_epi16(or_mask, ones))) {
                    // the current grid has no solution, go back
                    if ( verbose ) {
                        unsigned int m = 0x3ffff & _mm256_movemask_epi8(_mm256_cmpgt_epi16(_mm256_and_si256(or_mask, _mm256_sub_epi16(or_mask, ones)), _mm256_setzero_si256()));
                        int idx = __tzcnt_u32(m)>>1;
                        if ( grid_state->stackpointer == 0 && unique_check_mode == 0 ) {
                            printf("Line %d: stack 0, back track - column and row intersection at %s containing multiple hidden singles\n", line, cl2txt[irow*9+idx]);
                        } else if ( debug ) {
                            printf("back track - column and row intersection at %s contains multiple hidden singles\n", cl2txt[irow*9+idx]);
                        }
                    }
                    goto back;
                }
#endif

                // check row/box (8) candidates
                unsigned short m = grid_state->unlocked.get_indexbits(i, 8) | ((get_contiguous_masked_indices_for_box(unlocked,irow)&0xff)<<8);
                __m256i a = _mm256_cmpgt_epi16(rowbox_mask, _mm256_setzero_si256());
                unsigned int mask = and_compress_masks<false>(a, m);
                if (mask) {
                    int s_idx = __tzcnt_u32(mask);
                    bool is_row = s_idx < 8;
                    int celli = is_row ? i + s_idx : b + box_offset[s_idx&7];
                    if ( verbose && debug ) {
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
            const unsigned char row_triads_lut[9] = {
                   0, 10, 20, 3, 13, 23, 6, 16, 26 };
            const __m256i shuff725to012 = _mm256_setr_epi8(14, 15,  4,  5, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1,  -1, -1,
                                                               -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
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
                    if ( verbose ) {
                        if ( grid_state->stackpointer == 0 && unique_check_mode == 0 ) {
                             printf("Line %d: stack 0, multiple hidden singles in row/box cell %s\n", line, cl2txt[80]);
                        } else if ( debug ) {
                            printf("back track - multiple hidden singles in row/box cell %s\n", cl2txt[80]);
                        }
                    }
                    goto back;
                }
                if ( verbose && debug ) {
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
                if ( verbose ) {
                    if ( grid_state->stackpointer == 0 && unique_check_mode == 0 ) {
                         printf("Line %d: stack 0, multiple hidden singles in row/box cell %s\n", line, cl2txt[80]);
                    } else if ( debug ) {
                        printf("back track - multiple hidden singles in row/box cell %s\n", cl2txt[80]);
                    }
                }
                goto back;
            }
            if ( verbose && debug ) {
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
        const __m256i mask27 { -1LL, (long long int)0xffffffffffff00ffLL, (long long int)0xffffffff00ffffffLL, 0xffffffffffLL };

        const __m256i low_mask  = _mm256_set1_epi8 ( 0x0f );
        const __m256i threes    = _mm256_set1_epi8 ( 3 );
        const __m256i fours     = _mm256_set1_epi8 ( 4 );
        const __m256i word_mask = _mm256_set1_epi16 ( 0x00ff );
        const __m256i lookup    = _mm256_setr_epi8(0 ,1 ,1 ,2 ,1 ,2 ,2 ,3 ,1 ,2 ,2 ,3 ,2 ,3 ,3 ,4,
                                                   0 ,1 ,1 ,2 ,1 ,2 ,2 ,3 ,1 ,2 ,2 ,3 ,2 ,3 ,3 ,4);

            // A3.2.1 (check col-triads)
            // just a plain old popcount, but for both vectors interleaved
            __m256i v1 = _mm256_loadu_si256((__m256i *)triad_info.col_triads);
            __m256i v2 = _mm256_loadu_si256((__m256i *)(triad_info.col_triads+16));
            __m256i lo1 = _mm256_and_si256 (v1, low_mask);
            __m256i hi1 = _mm256_and_si256 (_mm256_srli_epi16 (v1, 4), low_mask );
            __m256i cnt11 = _mm256_shuffle_epi8 (lookup, lo1);
            __m256i cnt12 = _mm256_shuffle_epi8 (lookup, hi1);
            cnt11 = _mm256_add_epi8 (cnt11, cnt12);
            __m256i res = _mm256_add_epi8 (cnt11, _mm256_bsrli_epi128(cnt11, 1));
            __m256i lo2 = _mm256_and_si256 (v2, low_mask);
            __m256i hi2 = _mm256_and_si256 (_mm256_srli_epi16 (v2, 4), low_mask );
            __m256i cnt21 = _mm256_shuffle_epi8 (lookup, lo2);
            __m256i cnt22 = _mm256_shuffle_epi8 (lookup, hi2);
            cnt21 = _mm256_add_epi8 (cnt21, cnt22);
            res = _mm256_and_si256 (res, word_mask );
            res = _mm256_packus_epi16(res, _mm256_and_si256 (_mm256_add_epi8 (cnt21, _mm256_bsrli_epi128(cnt21, 1)), word_mask ));
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
                if ( verbose && debug ) {
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

            lo1 = _mm256_and_si256 (v1, low_mask);
            hi1 = _mm256_and_si256 (_mm256_srli_epi16 (v1, 4), low_mask );
            cnt11 = _mm256_shuffle_epi8 (lookup, lo1);
            cnt12 = _mm256_shuffle_epi8 (lookup, hi1);
            cnt11 = _mm256_add_epi8 (cnt11, cnt12);
            res = _mm256_add_epi8 (cnt11, _mm256_bsrli_epi128(cnt11, 1));
            lo2 = _mm256_and_si256 (v2, low_mask);
            hi2 = _mm256_and_si256 (_mm256_srli_epi16 (v2, 4), low_mask );
            cnt21 = _mm256_shuffle_epi8 (lookup, lo2);
            cnt22 = _mm256_shuffle_epi8 (lookup, hi2);
            cnt21 = _mm256_add_epi8 (cnt21, cnt22);
            res = _mm256_and_si256 (res, word_mask );
            res = _mm256_packus_epi16(res, _mm256_and_si256 (_mm256_add_epi8 (cnt21, _mm256_bsrli_epi128(cnt21, 1)), word_mask ));
            res = _mm256_and_si256(_mm256_permute4x64_epi64(res, 0xD8), mask27);

            m = _mm256_movemask_epi8(_mm256_cmpeq_epi8(res, threes)) & grid_state->triads_unlocked[Row];
            triad_info.triads_selection[Row] = _mm256_movemask_epi8(_mm256_cmpeq_epi8(res, fours));

            // the best that can be done for rows - remember that the the order of
            // row triads is not aligned with the order of cells.
            bit128_t tr = grid_state->unlocked;   // for row triads - any triad cell unlocked

            // to allow checking of unresolved triads
            tr.u64[0]  |= (tr.u64[0] >> 1)  | (tr.u64[0] >> 2);
            tr.u64[0]  |= ((tr.u64[1] & 1)  | ((tr.u64[1] & 2) >> 1))<<63;
            tr.u64[1]  |= (tr.u64[1] >> 1)  | (tr.u64[1] >> 2);

            while (m) {
                unsigned char tidx = tzcnt_and_mask(m);
                unsigned char logical_tidx = tidx-tidx/10;
                unsigned char ri  = logical_tidx/9+logical_tidx%9/3*3;
                unsigned char tci = logical_tidx%3*3;
                unsigned char off = ri*9+tci;
                if ( !tr.check_indexbit(off)) {  // locked
                    grid_state->triads_unlocked[Row] &= ~(1<<tidx);
                    continue;
                }

                if ( verbose && debug ) {
                    char ret[32];
                    format_candidate_set(ret, triad_info.row_triads[tidx]);
                    printf("triad set (row): %-9s %s\n", ret, cl2txt[ri*9+tci]);
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
        // Note that in principle, the below for-loop could be executed multiple times by
        // using ~tmustnt in place of the col/row_triads input.
        //
        bool any_changes = false;

        // mask for each line of 9 triad *must/*mustnt.
        // all t/pmust* variables are populated in groups of three, the first two in low 12 bytes (0-11),
        // the third group in bytes 16-21.
        const __m256i mask = _mm256_setr_epi16( 0x1ff, 0x1ff, 0x1ff, 0x1ff, 0x1ff, 0x1ff, 0, 0,
                                                0x1ff, 0x1ff, 0x1ff, 0,     0,     0,     0, 0);
        // rotation of groups of 3 triads *must/*mustnt.
        const __m256i rot_hpeers = _mm256_setr_epi8( 2,3,4,5,0,1, 8, 9,10,11, 6, 7,-1,-1,-1,-1,
                                                     2,3,4,5,0,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1);
        // shuffle within tmustnt to setup for aligned 9 triads (order of candidates).
        const __m256i shuff_tmustnt = _mm256_setr_epi8( -1,-1,-1,-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,
                                                        -1,-1,-1,-1, 4, 5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1);


        for (int type=1; type>=0; type--) {	// row = 0, col = 1

            // input
            unsigned short *triads   = type==0?triad_info.row_triads:triad_info.col_triads;
            unsigned short *wo_musts = type==0?triad_info.row_triads_wo_musts:triad_info.col_triads_wo_musts;
            unsigned short *ptriads  = triads;
            __m256i pmustnt[2][3] = {mask, mask, mask, mask, mask, mask};
            __m256i tmustnt[3];

            // first load triad candidates and compute peer based pmustnt

            // i=0 (manually unrolled loop)
                // tmustnt computed from all candidates in row/col_triads
                __m256i tmustnti = _mm256_andnot_si256(_mm256_loadu2_m128i((__m128i*)&ptriads[6], (__m128i*)ptriads), mask);
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
                tmustnti = _mm256_andnot_si256(_mm256_loadu2_m128i((__m128i*)&ptriads[6], (__m128i*)ptriads), mask);
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
                tmustnti = _mm256_andnot_si256(_mm256_loadu2_m128i((__m128i*)&ptriads[6], (__m128i*)ptriads), mask);
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
            const __m256i shuff_row_mask = _mm256_setr_epi8( 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 4, 5, 4, 5,
                                                              4, 5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1);

            const __m256i shuff_row_mask2 = _mm256_setr_epi8( 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 8, 9, 8, 9,
                                                              8, 9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1);
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
                __m256i to_remove_v = _mm256_and_si256(_mm256_and_si256(*(__m256i_u*)ptriads, tmustnt[i]),mask9);
                if ( _mm256_testz_si256(mask9,to_remove_v)) {
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
                        __m256i c = andnot_if(*(__m256i_u*)&candidates[i*9], tmask, mask9);
                        _mm_storeu_si128((__m128i_u*)&candidates[i*9], _mm256_castsi256_si128(c));
                        candidates[8+i*9] = _mm256_extract_epi16(c, 8);
                        row_combo_tpos[0] |= bitx3_lut[bits] << (9*i);
                    }
                    // row i+3
                    if ( (bits = (m >> 3) & 0x7) ) {
                        tmask = _mm256_bsrli_epi128(tmustnt[i], 6);
                        tmask = _mm256_permute2x128_si256(tmask, tmask, 0);
                        tmask = _mm256_shuffle_epi8(tmask, shuff_row_mask);
                        __m256i c = andnot_if(*(__m256i_u*)&candidates[(i+3)*9], tmask, mask9);
                        _mm_storeu_si128((__m128i_u*)&candidates[(i+3)*9], _mm256_castsi256_si128(c));
                        candidates[8+(i+3)*9] = _mm256_extract_epi16(c, 8);
                        row_combo_tpos[1] |= bitx3_lut[bits] << (9*i);
                    }
                    // row i+6
                    if ( (bits = (m >> 6) & 0x7) ) {
                        tmask = _mm256_permute4x64_epi64(tmustnt[i], 0x99);
                        tmask = _mm256_shuffle_epi8(tmask, shuff_row_mask2);
                        __m256i c = andnot_if(*(__m256i_u*)&candidates[(i+6)*9], tmask, mask9);
                        _mm_storeu_si128((__m128i_u*)&candidates[(i+6)*9], _mm256_castsi256_si128(c));
                        candidates[8+(i+6)*9] = _mm256_extract_epi16(c, 8);
                        row_combo_tpos[2] |= bitx3_lut[bits] << (9*i);
                    }
                } else { // type == 1
                    // update the band of 3 rows with column triads
                    // using directly tmustnt[i]
                    __m256i c1 = andnot_if(*(__m256i_u*)&candidates[i*27], tmustnt[i], mask9);
                    _mm_storeu_si128((__m128i_u*)&candidates[i*27], _mm256_castsi256_si128(c1));
                    __m256i c2 = andnot_if(*(__m256i_u*)&candidates[9+i*27], tmustnt[i], mask9);
                    _mm_storeu_si128((__m128i_u*)&candidates[9+i*27], _mm256_castsi256_si128(c2));
                    __m256i c3 = andnot_if(*(__m256i_u*)&candidates[18+i*27], tmustnt[i], mask9);
                    _mm_storeu_si128((__m128i_u*)&candidates[18+i*27], _mm256_castsi256_si128(c3));
                    if ( m & 0x100 ) {
                        candidates[8+i*27]    = _mm256_extract_epi16(c1, 8);
                        candidates[9+8+i*27]  = _mm256_extract_epi16(c2, 8);
                        candidates[18+8+i*27] = _mm256_extract_epi16(c3, 8);
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
                            rslvd_row_combo_tpos[i_rel/3] |= 7<<(row_triad_canonical_map[ltidx]%9*3);
                        } else {
                            rslvd_col_combo_tpos |= 0x40201<<i_rel;
                        }
                        triads_resolved++;
                    }
                    if ( verbose && debug ) {
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
            goto start;
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
// - previously found sets (and their complements) as well as found triads
// - sets that occupy all available space minus one - impossible due to perfect single detection

    if ( mode_sets )
    {
        bool found = false;

        // visit only the changed (updated) cells

        bit128_t to_visit_n;          // tracks all the cells to visit
        bit128_t to_visit_again {};   // those cells that have been updated

        to_visit_n.u128 = grid_state->updated.u128 & grid_state->unlocked.u128;

        // A cheap way to avoid unnecessary naked set searches
        grid_state->set23_found[Row].u128 |= ~grid_state->unlocked.u128;
        grid_state->set23_found[Row].u64[1] &= 0x1ffff;
        grid_state->set23_found[Col].u128 |= ~grid_state->unlocked.u128;
        grid_state->set23_found[Col].u64[1] &= 0x1ffff;
        grid_state->set23_found[Box].u128 |= ~grid_state->unlocked.u128;
        grid_state->set23_found[Box].u64[1] &= 0x1ffff;
        for (unsigned char n = 0; n < 2; ++n) {
            unsigned long long tvnn = to_visit_n.u64[n];
            while (tvnn) {
                unsigned char i = tzcnt_and_mask(tvnn) + (n<<6);

                unsigned short cnt = __popcnt16(candidates[i]);

                if (cnt <= MAX_SET && cnt > 1) {
                    // Note: this algorithm will never detect a naked set of the shape:
                    // {a,b},{a,c},{b,c} as all starting points are 2 bits only.
                    // The same situation is possible for 4 set members.
                    //
                    unsigned long long to_change[2] {};

                    __m256i a_i = _mm256_set1_epi16(candidates[i]);
                    __m128i res;
                    unsigned char ul;
                    unsigned char s;

                    // check row
                    //
                    if ( !grid_state->set23_found[Row].check_indexbit(i)) {
                        unsigned char ri = row_index[i];
                        ul = grid_state->get_ul_set_search<Row>(ri);
                        if (check_back || (cnt+2 <= ul) ) {
                            my_naked_sets_searched++;
                            res = _mm_cmpeq_epi16(_mm256_castsi256_si128(a_i), _mm_or_si128(_mm256_castsi256_si128(a_i), *(__m128i_u*) &candidates[9*ri]));
                            unsigned int m = compress_epi16_boolean128(res);
                            bool bit9 = candidates[i] == (candidates[i] | candidates[9*ri+8]);
                            if ( bit9 ) {
                                m |= 1<<8;    // fake the 9th mask position
                            }
                            s = _popcnt32(m);
                            unsigned long long m_neg = 0;
                            if (s > cnt) {
                                if ( verbose ) {
                                    char ret[32];
                                    format_candidate_set(ret, candidates[i]);
                                    if ( grid_state->stackpointer == 0 && unique_check_mode == 0 ) {
                                        printf("Line %d: naked  set (row) %s at %s, count exceeded\n", line, ret, cl2txt[ri*9]);
                                    } else if ( debug ) {
                                        printf("back track sets (row) %s at %s, count exceeded\n", ret, cl2txt[ri*9]);
                                    }
                                }
                                // no need to update grid_state
                                goto back;
                            } else if (s == cnt && cnt+2 <= ul) {
                                char ret[32];
                                if ( cnt <= 3 ) {
                                    grid_state->set23_found[Row].set_indexbits(m,ri*9,9);
                                    // update box, if set within triad
                                    if ( ((m&7) == m) ||
                                         ((m&(7<<3)) == m) ||
                                         ((m&(7<<6)) == m) ) {
                                            grid_state->set23_found[Box].set_indexbits(m&0x1ff,ri*9,9);
                                            add_indices<Box>((bit128_t*)to_change, i);
                                    }
                                }
                                if ( ul <= 3 + cnt ) {
                                    // could include locked slots
                                    m_neg = ~(m | grid_state->set23_found[Row].get_indexbits(ri*9,9));
                                    grid_state->set23_found[Row].set_indexbits(m_neg,ri*9,9);
                                    // update box, if set within triad
                                    if ( (m_neg&7) == m_neg ||
                                         (m_neg&(7<<3))  == m_neg ||
                                         (m_neg&(7<<6)) == m_neg ) {
                                            grid_state->set23_found[Box].set_indexbits(m_neg,ri*9,9);
                                    }
                                }
                                naked_sets_found++;
                                add_indices<Row>((bit128_t*)to_change, i);
                                if ( verbose && debug ) {
                                    if ( cnt <=3 || cnt+3 < ul ) {
                                        format_candidate_set(ret, candidates[i]);
                                        printf("naked  %s (row): %-7s %s\n", s==2?"pair":"set ", ret, cl2txt[ri*9+i%9]);
                                    } else {
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
                                        if ( complement != 0 ) {
                                            format_candidate_set(ret, complement);
                                            printf("hidden %s (row): %-7s %s\n", __popcnt16(complement)==2?"pair":"set ", ret, cl2txt[ri*9+k%9]);
                                        }
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
                        ul = uls[0] = grid_state->get_ul_set_search<Col>(ci);
                        uls[1] = grid_state->get_ul_set_search<Box>(bi);
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
                                    if ( verbose ) {
                                        char ret[32];
                                        format_candidate_set(ret, candidates[i]);
                                        if ( grid_state->stackpointer == 0 && unique_check_mode == 0 ) {
                                            printf("Line %d: naked  set (%s) %s at %s, count exceeded\n", line, js[j], ret, cl2txt[i]);
                                        } else if ( debug ) {
                                            printf("back track sets (%s) %s at %s, count exceeded\n", js[j], ret, cl2txt[i]);
                                        }
                                    }
                                    // no need to update grid_state
                                    goto back;
                                }
                            }
                            for ( int j=0; j<2; j++) {
                                if ( !chk[j] || (ss[j] != cnt)) {
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
                                    if ( j ) {
                                        add_indices<Box>((bit128_t*)to_change, i);
                                    } else {
                                        add_indices<Col>((bit128_t*)to_change, i);
                                    }
                                    unsigned char k = 0xff;
                                    unsigned short complement = 0;
                                    Kind kind = j?Box:Col;
                                    bit128_t s = { *(bit128_t*)&big_index_lut[i][kind][0] & ~grid_state->set23_found[kind] };
                                    bool set23_cond1 = (cnt <= 3);
                                    bool set23_cond2 = (uls[j] <= cnt+3);
                                    unsigned char index_by_i_kind = index_by_i[i][kind];
                                    for ( unsigned char k_m = 0; k_m<9; k_m++ ) {
                                        unsigned char k_i = (j?(b+box_offset[k_m]):(ci+k_m*9));
                                        if ( !s.check_indexbit(k_i) ) {
                                            continue;
                                        }
                                        bool in_set = (candidates[k_i] | candidates[i]) == candidates[i];
                                        // is this really an index originating from the Box/Col?
                                        if ( index_by_i[k_i][kind] == index_by_i_kind ) {
                                            if ( (set23_cond1 && in_set) || (set23_cond2 && !in_set) ) {
                                                grid_state->set23_found[kind].set_indexbit(k_i);
                                            }
                                            if ( !in_set ) {
                                                // set k and compute the complement set
                                                if ( k == 0xff ) {
                                                    k = k_i;
                                                }
                                                complement |= candidates[k_i];
                                                found = true;
                                            }
                                        }
                                    } // for
                                    complement &= ~candidates[i];
                                    if ( verbose && debug ) {
                                        char ret[32];
                                        if ( complement != 0 && set23_cond2 ) {
                                            format_candidate_set(ret, complement);
                                            printf("hidden %s (%s): %-7s %s\n", __popcnt16(complement)==2?"pair":"set ", js[j], ret, cl2txt[k]);
                                        } else {
                                            format_candidate_set(ret, candidates[i]);
                                            printf("naked  %s (%s): %-7s %s\n", cnt==2?"pair":"set ", js[j], ret, cl2txt[i]);
                                        }
                                    }
                                }
                            } // for
                        }
                    }

                    ((bit128_t*)to_change)->u128 &= ((bit128_t*)unlocked)->u128;

                    if ( ((bit128_t*)to_change)->u128 != (__int128)0 ) {
                        const unsigned char *ci = index_by_i[i];
                        // update candidates
                        for (unsigned char n = 0; n < 2; ++n) {
                            while (to_change[n]) {
                                unsigned char j_rel = tzcnt_and_mask(to_change[n]);
                                unsigned char j = j_rel + (n<<6);

                                // if this cell is not part of our set
                                if ((candidates[j] | candidates[i]) != candidates[i]) {
                                    // if there are bits that need removing
                                    if (candidates[j] & candidates[i]) {
                                        candidates[j] &= ~candidates[i];
                                        to_visit_again.set_indexbit(j);
                                        found = true;
                                    }
                                } else {
                                    const unsigned char *cj = index_by_i[j];
                                    if ( (cj[Col] == ci[Col]) && (ss[0] == cnt) && (ss[0] <= 3) ) {
                                        grid_state->set23_found[Col].set_indexbit(j);
                                    }
                                    if ( (cj[Box] == ci[Box]) && (ss[1] == cnt) && (ss[1] <= 3) ) {
                                        grid_state->set23_found[Box].set_indexbit(j);
                                    }
                                }
                            }
                        }

                        // If any cell's candidates got updated, go back and try all that other stuff again
                        if (found) {
                            to_visit_n.u64[n] = tvnn;
                            to_visit_n.u128 |= to_visit_again.u128;
                            grid_state->updated.u128 = to_visit_n.u128;
                            goto start;
                        }
                    }
                } else if ( cnt == 1 ) {
                    // this is not possible, but just to eliminate cnt == 1:
                    to_visit_n.u64[n] = tvnn;
                    to_visit_n.u128 |= to_visit_again.u128;
                    grid_state->updated.u128 = to_visit_n.u128;
                    if ( verbose && debug ) {
                        printf("naked  (sets) ");
                    }
                    e_i = i;
                    e_digit = candidates[i];
                    printf("found a singleton in set search... strange\n");
                    goto enter;
                }
            } // while
        } // for
        to_visit_n.u128 |= to_visit_again.u128;
        grid_state->updated.u128 = to_visit_n.u128;
    }
#endif

    bit128_t bivalues {};
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
        bivalues.u16[4] = compress_epi16_boolean<false>(_mm256_and_si256(
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
        if ( N < 23 ) {
                unsigned char sum12s = 0;   // count P, the number of bi-values + the number of locked cells
                unsigned char target = 0;   // the index of the only cell with three candidates

                // track wether there was a single cell with count > 2
                // if there is a second such cell, goto guess
                unsigned char not_pairs = 0;

                for (unsigned char i = 0; i < 2; i++) {
                    unsigned long long m = ~((bit128_t*)unlocked)->u64[i];
                    if ( m ) {
                        m |= bivalues.u64[i];
                        unsigned char pc = __popcnt64(m);
                        sum12s += pc;
                        if ( pc != 64 ) {
                            if ( pc < 63 || not_pairs++ ) {
                                goto guess;
                            }
                            target = (i<<6) + __tzcnt_u64(~m);
                        }
                    }
                }
                // sum12s is the total of 128 bits minus the ones that are neither 1 or 2
                sum12s -= (64-17);    // only 81 possible bits, subtract 64-17 to compensate
                if ( sum12s == 81 ) { // this means Q == 0

                    if ( unique_check ) {
                        if ( verbose && debug ) {
                            if ( grid_state->stackpointer == 0 && !unique_check_mode ) {
                                printf("Found a bi-value universal grave. This means at least two solutions exist.\n");
                            } else if ( unique_check_mode ) {
                                printf("checking a bi-value universal grave.\n");
                        }
                        }
                        if ( verbose && debug ) {
                            printf("a bi-value universal grave means at least two solutions exist.\n");
                        }
                        goto guess;
                    } else if ( grid_state->stackpointer ) {
                        if ( verbose && debug ) {
                            printf("back track - found a bi-value universal grave.\n");
                        }
                        goto back;
                    } else {   // busted.  This is not a valid puzzle under standard rules.
                        if ( verbose && debug ) {
                            printf("Found a bi-value universal grave. This means at least two solutions exist.\n");
                        }
                        non_unique_count++;
                        goto guess;
                    }
                }

                if ( sum12s == 80 && __popcnt16(candidates[target]) == 3 ) {
                    unsigned char row = row_index[target];
                    unsigned short cand3 = candidates[target];
                    unsigned short digit = 0;
                    unsigned short mask = ((bit128_t*)unlocked)->get_indexbits(row*9,9);
                    __m256i maskv = expand_bitvector(mask);
                    __m256i c = _mm256_and_si256(_mm256_load_si256((__m256i*) &candidates[row*9]), maskv);
                    while (cand3) {
                        unsigned short canddigit = cand3 & (-cand3);
                        // count cells with this candidate digit:
                        __m256i tmp = _mm256_and_si256(_mm256_set1_epi16(canddigit), c);
                        // as a boolean
                        tmp = _mm256_cmpeq_epi16(tmp,_mm256_setzero_si256());
                        // need three cell, doubled bits in mask:
                        if ( _popcnt32(~_mm256_movemask_epi8(tmp)) == 3*2 ) {
                            digit = canddigit;
                            break;
                        };
                        cand3 &= ~canddigit;
                    }
                    if ( digit ) {
                        bug_count++;
                        if ( verbose && debug ) {
                            printf("bi-value universal grave pivot:");
                        }
                        e_i = target;
                        e_digit = digit;
                        goto enter;
                    }
                }
        }
    }

guess:
    // Make a guess if all that didn't work
    grid_state = grid_state->make_guess<verbose>(triad_info, bivalues);
    current_entered_count += 0x101;     // increment high byte for the new grid_state, plus one for the guess made.
    no_guess_incr = 0;
    goto start;

}

void print_help() {
        printf("fastss version: %s\ncompile options: %s\n", version_string, compilation_options);
        printf(R"(Synopsis:
fastss [options] [puzzles] [solutions]
\t [puzzles] names the input file with puzzles. Default is 'puzzles.txt'.
\t [solutions] names the output file with solutions. Default is 'solutions.txt'.

Command line options:
    -c  check for back tracking even when no guess was made (e.g. if puzzles might have no solution)
    -d# provide some detailed information on the progress of the puzzle solving.
        add a 2 for even more detail.
    -h  help information (this text)
    -l# solve a single line from the puzzle.
    -m[ST]* execution modes (sets, triads)
    -t# set the number of threads
    -u  check the solution for uniqueness
    -v  verify the solution
    -x  provide some statistics
    -#1 change base for row and column reporting from 0 to 1

)");
}

int main(int argc, const char *argv[]) {

    int line_to_solve = 0;

    if ( argc > 0 ) {
        argc--;
        argv++;
    }

    printf("command options: ");
    for (int i = 0; i < argc; i++) {
        printf("%s ", argv[i]);
    }
    printf("\n");

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
                 default:
                    printf("invalid mode %c\n", argv[0][p]);
                 }
             }
             break;
        case 'u':    // verify uniqueness, check for multiple solutions
             unique_check=1;
             break;
        case 'v':    // verify
             verify=1;
             break;
        case 'x':    // stats output
             reportstats=1;
             break;
        case 'l':    // line of puzzle to solve
             sscanf(argv[0]+2, "%d", &line_to_solve);
             break;
        case 't':    // set number of threads
             if ( argv[0][2] && isdigit(argv[0][2]) ) {
                 sscanf(&argv[0][2], "%d", &numthreads);
                 if ( numthreads != 0 ) {
                     omp_set_num_threads(numthreads);
                 }
             }
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
        }
        argc--, argv++;
    }

    assert((sizeof(GridState) & 0x3f) == 0);

   // sort out the CPU and OMP settings

   if ( !__builtin_cpu_supports("avx2") ) {
        printf("This program requires a CPU with the AVX2 instruction set.\n");
        exit(0);
    }
    // lacking BMI support? unlikely!
    if ( !__builtin_cpu_supports("bmi") ) {
        printf("This program requires a CPU with the BMI instructions (such as blsr)\n");
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
			printf("Error mmap of input file %s: %s\n", ifn, strerror(errno));
			exit(0);
		}
	}
	close(fdin);

	// skip first line, unless it's a puzzle.
    size_t pre = 0;
    if ( !(isdigit((int)string[0]) || string[0] != '.' || string[81] != 10) ) {
	    while (string[pre] != 10) {
    	    ++pre;
    	}
    	++pre;
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
			printf("Error opening output file %s: %s\n", ofn, strerror(errno));
			exit(0);
		}
	}
    if ( ftruncate(fdout, (size_t)outnpuzzles*164) == -1 ) {
		if (errno ) {
			printf("Error setting size (ftruncate) on output file %s: %s\n", ofn, strerror(errno));
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
    size_t i;
    size_t imax = npuzzles*82;

    signed char *string_pre = string+pre;
	GridState *stack = 0;

    if ( line_to_solve ) {
        i = (line_to_solve-1)*82;
        // copy unsolved grid
        memcpy(output, &string_pre[i], 81);
        memcpy(&output[82], &string_pre[i], 81);
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

        stack[0].initialize(grid);

        if ( reportstats !=0 || debug != 0) {
            solve<true>(grid, stack, line_to_solve);
        } else {
            solve<false>(grid, stack, line_to_solve);
        }
    } else {

        // The OMP directives:
        // proc_bind(close): high preferance for thread/core affinity
        // firstprivate(stack): the stack is allocated for each thread once and seperately
        // schedule(dynamic,64): 64 puzzles are allocated at a time and these chunks
        //   are assigned dynmically (to minimize random effects of difficult puzzles)
        // shared(...) lists the variables that are shared (as opposed to separate copies per thread)
        //
#pragma omp parallel for proc_bind(close) firstprivate(stack) shared(string_pre, output, npuzzles, i, imax, debug, reportstats) schedule(dynamic,64)
        for (i = 0; i < imax; i+=82) {
            // copy unsolved grid
            signed char *grid = &output[i*2+82];
            memcpy(&output[i*2], &string_pre[i], 81);
            memcpy(grid, &string_pre[i], 81);
            // add comma and newline in right place
            output[i*2 + 81] = ',';
            output[i*2 + 163] = 10;
            // solve the grid in place
            if ( stack == 0 ) {
                // force alignment the 'old-fashioned' way
                // not going to free the data ever
                // stack = (GridState*)malloc(sizeof(GridState)*GRIDSTATE_MAX);
                stack = (GridState*) (~0x3fll & ((unsigned long long) malloc(sizeof(GridState)*GRIDSTATE_MAX+0x40)+0x40));
            }

            stack[0].initialize(&output[i*2+82]);
            if ( reportstats !=0 || debug != 0) {
                solve<true>(grid, stack, i/82+1);
            } else {
                solve<false>(grid, stack, i/82+1);
            }
        }
    }

	int err = munmap(string, fsize);
	if ( err == -1 ) {
		if (errno ) {
			printf("Error munmap file %s: %s\n", ifn, strerror(errno));
		}
	}
	err = munmap(output, (size_t)npuzzles*164);
	if ( err == -1 ) {
		if (errno ) {
			printf("Error munmap file %s: %s\n", ofn, strerror(errno));
		}
	}

    if ( reportstats) {
        long long duration = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration(std::chrono::steady_clock::now() - starttime)).count();
        printf("schoku version: %s\ncompile options: %s\n", version_string, compilation_options);
        printf("%10ld  puzzles entered\n", npuzzles);
        unsigned long solved_cnt = solved_count.load();
        printf("%10ld  %.0lf/s  puzzles solved\n", solved_cnt, (double)solved_cnt/((double)duration/1000000000LL));
		printf("%8.1lfms  %6.2lf\u00b5s/puzzle  solving time\n", (double)duration/1000000, (double)duration/(npuzzles*1000LL));
        if ( unsolved_count.load()) {
            printf("%10ld  puzzles had no solution\n", unsolved_count.load());
        }
        if ( unique_check ) {
            printf("%10ld  puzzles had a unique solution\n", solved_count.load() - non_unique_count.load());
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
        if ( bug_count.load() ) {
            printf("%10ld  bi-value universal graves detected\n", bug_count.load());
        }
    }

    if ( unique_check && non_unique_count.load()) {
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
