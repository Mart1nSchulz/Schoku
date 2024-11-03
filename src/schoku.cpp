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
 * Version 0.6  (cleanup)
 *
 * Performance changes:
 * - move enter loop into solve (beware: more spaghetti logic)
 * - the enter loop detects additional naked singles
 * - add bit96_t
 *   in order to tighten GridState
 * - combined column and box in naked set search
 *
 * Functional changes:
 * - report complementary sets as 'hidden', pairs as 'pairs'
 *
 * Performance measurement and statistics:
 *
 * data: 17-clue sudoku (49151 puzzles)
 * CPU:  Ryzen 7 4700U
 *
 * schoku version: 0.6
 *     49151  puzzles entered
 *     49151  1183663/s  puzzles solved
 *    41.5ms   0.84µs/puzzle  solving time
 *     34234   69.65%  puzzles solved without guessing
 *     38484    0.78/puzzle  guesses
 *     24011    0.49/puzzle  back tracks
 *    321296    6.54/puzzle  digits entered and retracted
 *   1526646   31.06/puzzle  'rounds'
 *     88712    1.80/puzzle  naked sets found
 *   3915828   79.67/puzzle  naked sets searched
 *       838  bi-value universal graves detected
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

const char *version_string = "0.6";

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

// bit128_t type
// for all 81-bit fields to support different access patterns.
typedef
union {
    __uint128_t u128;
    unsigned long long u64[2];
    unsigned int u32[4];
    unsigned short u16[8];
    unsigned char u8[16];

    inline operator __uint128_t () {
        return this->u128;
    }

    inline bool check_indexbit(unsigned char idx) {
        return this->u8[idx>>3] & (1<<(idx & 0x7));
    }
    inline bool check_and_mask_index(unsigned char idx) {
        return _bittestandreset64((long long int *)&this->u64[idx>>6], idx & 0x3f);
    }
    inline void set_indexbit(unsigned char idx) {
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

// bit96_t type
// for the updated 81-bit field to support different access patterns.
typedef
union  __attribute__ ((packed))    // we want to pack the GridState data.
{
    unsigned long long u64[1];
    unsigned int u32[3];
    unsigned short u16[6];
    unsigned short u8[12];
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Waddress-of-packed-member"

    inline bool check_indexbit(unsigned char idx) {
        return this->u8[idx>>3] & (1<<(idx & 0x7));
    }
    inline bool check_and_mask_index(unsigned char idx) {
        return _bittestandreset64((long long int *)&this->u64[idx>>6], idx & 0x3f);
    }
    inline void set_indexbit(unsigned char idx) {
        this->u8[idx>>3] |= 1<<(idx & 0x7);
    }
#pragma GCC diagnostic pop

    inline void set_indexbits(unsigned long long mask, unsigned char pos, unsigned char bitcount) {
        mask &= (unsigned long long)((1LL<<bitcount)-1);  // since the bit count is specific, enforce it
        if ( pos < 64 ) {
            u64[0] |= mask << pos;
            if ( pos + bitcount >= 64 ) {
                u32[2] |= mask >> (64 - pos);
            }
        } else {
            u32[2] |= mask << (pos-64);
        }
    }
    // bitcount <= 64
    inline unsigned long long get_indexbits(unsigned char pos, unsigned char bitcount) {
        unsigned long long res = 0;
        if ( pos < 64 ) {
            res = u64[0] >> pos;
            if ( pos + bitcount >= 64 ) {
                res |= u32[2] << (64 - pos);
            }
        } else {
            res = u32[2] >> (pos-64);
        }
        // clip result
        if ( bitcount < 64 ) {
            res &= ((1LL<<bitcount)-1);
        }
        return res;
    }
} bit96_t;

// alignment helper construct
// The goal is to not mix heavily used and lesser used data on the same cache line.
// The secondary goal is to associate data structures with as few cache lines as possible.
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

const unsigned char *row_index = index_by_kind[Row];
const unsigned char *column_index = index_by_kind[Col];
const unsigned char *box_index = index_by_kind[Box];

const align64_empty c8;
// this table provides the bit masks corresponding to each index and each Kind of section.
// The 4th column contains all Kind's or'ed together. 
// Heavily used
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

align64_empty c9;

bool bmi2_support = false;

// stats and command line options
int reportstats     = 0; // collect and report some statistics
int verify          = 0; // verify solution correctness (implied otherwise)
int unique_check    = 0; // check solution uniqueness
int debug           = 0; // provide step by step output on the solution
int thorough_check  = 0; // check for back tracking even if no guess was made.
int numthreads      = 0; // if not 0, number of threads

signed char *output;

std::atomic<long> solved_count(0);          // puzzles solved
std::atomic<long> unsolved_count(0);        // puzzles unsolved (no solution exists)
std::atomic<long> non_unique_count(0);      // puzzles not unique (with -u)
std::atomic<long> not_verified_count(0);    // puzzles non verified (with -v)
std::atomic<long> verified_count(0);        // puzzles successfully verified (with -v)
std::atomic<long> bug_count(0);             // universal grave detected
std::atomic<long long> guesses(0);          // how many guesses did it take
std::atomic<long long> trackbacks(0);       // how often did we back track
std::atomic<long> no_guess_cnt(0);          // how many puzzles were solved without guessing
std::atomic<long long> past_naked_count(0); // how often do we get past the naked single serach
std::atomic<long long> naked_sets_searched(0); // how many naked sets did we search for
std::atomic<long long> naked_sets_found(0); // how many naked sets did we actually find
std::atomic<long long> digits_entered_and_retracted(0); // to measure guessing overhead

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
        return compress_epi16_boolean<doubledbits>(_mm256_and_si256(a, expand_bitvector(b)));
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

// a helper function to print a grid, not used in production code.
inline void grid_dump(unsigned short *candidates, FILE *stream) {
    for ( int i_=0; i_<81; i_++ ) {
        char ret[32];
        format_candidate_set(ret, candidates[i_]);
        fprintf(stream, "%8s,", ret);
        if ( i_%9 == 8 ) {
            printf("\n");
        }
    }
}
inline void grid_dump(unsigned short *candidates) {
    grid_dump(candidates, stdout);
}

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
    bit96_t updated;                  // for keeping track of which cell's candidates may have been changed since last time we looked for naked sets. Set bits correspond to changed candidates in these cells
    bit128_t unlocked;                // for keeping track of which cells don't need to be looked at anymore. Set bits correspond to cells that still have multiple possibilities
    bit128_t set23_found[3];          // for keeping track of found sets of size 2 and 3

// GridState is normally copied for recursion
// Here we initialize the starting state including the puzzle.
//
inline void initialize(signed char grid[81]) {
    // 0x1ffffffffffffffffffffULLL is (0x1ULL << 81) - 1
    unlocked.u128 = (((__uint128_t)1)<<81)-1;
    updated.u64[0] = ~(unsigned long long)0;
    updated.u32[2] = 0x1ffff;

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
        if (grid[i] < 49) {
            candidates[i] = 0x01ff ^ (rows[row_index[i]] | columns[column_index[i]] | boxes[box_index[i]]);         
        } else {
            candidates[i] = 1 << (grid[i]-49);
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
        printf(" %x at [%d,%d]\n", _tzcnt_u32(digit)+1, i/9, i%9);
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

    updated.u64[0] |= to_update.u64[0];
    updated.u32[2] |= to_update.u32[2];
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
GridState* make_guess() {
    // Find a cell with the least candidates. The first cell with 2 candidates will suffice.
    // Pick the candidate with the highest value as the guess.
    // Save the current grid state (with the chosen candidate eliminated) for tracking back.

    // Find the cell with fewest possible candidates
    unsigned long long to_visit;
    unsigned char guess_index = 0;
    unsigned char i_rel;
    unsigned char cnt;
    unsigned char best_cnt = 16;
    
    to_visit = unlocked.u64[0];
    while ( best_cnt > 2 && to_visit != 0 ) {
        i_rel = tzcnt_and_mask(to_visit);
        cnt = __popcnt16(candidates[i_rel]);
        if (cnt < best_cnt) {
            best_cnt = cnt;
            guess_index = i_rel;
        }
    }

    to_visit = unlocked.u64[1];
    while ( best_cnt > 2 && to_visit != 0 ) {
        i_rel = tzcnt_and_mask(to_visit) + 64;
        cnt = __popcnt16(candidates[i_rel]);
        if (cnt < best_cnt) {
            best_cnt = cnt;
            guess_index = i_rel;
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
        printf("guess at [%d,%d]\nsaved grid_state level >%d<: %.81s\n",
               guess_index/9, guess_index%9, stackpointer, gridout);
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

template <bool verbose>
bool solve(signed char grid[81], GridState stack[], int line) {

    GridState *grid_state = &stack[0];
    unsigned long long *unlocked = grid_state->unlocked.u64;
    unsigned short* candidates;

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
            printf(" %x at [%d,%d]\n", _tzcnt_u32(e_digit)+1, e_i/9, e_i%9);
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

        add_and_mask_all_indices(&to_update, &grid_state->unlocked, e_i);

        grid_state->updated.u64[0] |= to_update.u64[0];
        grid_state->updated.u32[2] |= to_update.u32[2];

        const __m256i mask = _mm256_set1_epi16(~e_digit);

        unsigned short dtct_j = 0;
        unsigned int dtct_m = 0;
        for (unsigned char j = 0; j < 80; j += 16) {
            __m256i c = _mm256_load_si256((__m256i*) &candidates[j]);
            // this if is only taken very occasionally, branch prediction
            if (__builtin_expect (check_back && _mm256_movemask_epi8(
                                  _mm256_cmpeq_epi16(c, _mm256_setzero_si256())
                                  ), 0)) {
                // Back track, no solutions along this path
                if ( verbose ) {
                    unsigned int mx = _mm256_movemask_epi8(_mm256_cmpeq_epi16(c, _mm256_setzero_si256()));
                    unsigned char pos = j+(_tzcnt_u32(mx)>>1);
                    if ( grid_state->stackpointer == 0 && unique_check_mode == 0 ) {
                        printf("Line %d: cell [%d,%d] is 0\n", line, pos/9, pos%9);
                    } else if ( debug ) {
                        printf("back track - cell [%d,%d] is 0\n", pos/9, pos%9);
                    }
                }
                e_digit=0;
                goto back;
            }

            // expand locked unsigned short to boolean vector
            __m256i mlocked = expand_bitvector(~to_update.u16[j>>4]);
            // apply mask (remove bit), preserving the locked cells     
            c = and_unless(c, mask, mlocked);
            __m256i a = _mm256_cmpeq_epi16(_mm256_and_si256(c, _mm256_sub_epi16(c, ones)), _mm256_setzero_si256());
            unsigned int mask = and_compress_masks<true>(a, grid_state->unlocked.u16[j>>4]);
            if ( mask ) {
                dtct_m = mask;
                dtct_j = j;
            }
            _mm256_store_si256((__m256i*) &candidates[j], c);
        }
        if (unlocked[1] & (1ULL << (80-64))) {
            if ((to_update.u16[5] & 1) != 0) {
                candidates[80] &= ~e_digit;
            }
        }
        if ( dtct_m ) {
            int idx = _tzcnt_u32(dtct_m)>>1;
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
                        printf("Line %d: cell [%d,%d] is 0\n", line, pos/9, pos%9);
                    } else if ( debug ) {
                        printf("back track - cell [%d,%d] is 0\n", pos/9, pos%9);
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
                // Check if any cell has zero candidates
                if (__builtin_expect (check_back && _mm256_movemask_epi8(_mm256_cmpeq_epi16(c, _mm256_setzero_si256())),0)) {
                    // Back track, no solutions along this path
                    if ( verbose ) {
                        unsigned int mx = _mm256_movemask_epi8(_mm256_cmpeq_epi16(c, _mm256_setzero_si256()));
                        unsigned char pos = i+(_tzcnt_u32(mx)>>1);
                        if ( grid_state->stackpointer == 0 && unique_check_mode == 0 ) {
                            printf("Line %d: cell [%d,%d] is 0\n", line, pos/9, pos%9);
                        } else if ( debug ) {
                            printf("back track - cell [%d,%d] is 0\n", pos/9, pos%9);
                        }
                    }
                    goto back;
                } else {
                    // remove least significant digit and compare to 0:
                    // (c & (c-1)) == 0  => naked single
                    __m256i a = _mm256_cmpeq_epi16(_mm256_and_si256(c, _mm256_sub_epi16(c, ones)), _mm256_setzero_si256());
                    unsigned int mask = and_compress_masks<true>(a,m);
                    if ( mask ) {
                        int idx = _tzcnt_u32(mask)>>1;
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
    if ( *(__uint128_t*)unlocked == 0) {
        // Solved it
        if ( unique_check == 1 && unique_check_mode == 1 ) {
            if ( !nonunique_reported ) {
                if ( verbose && reportstats ) {
                    printf("Line %d: solution to puzzle is not unique\n", line);
                }
                non_unique_count++;
                nonunique_reported = true;
            }
        }
        if ( verify ) {
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
    {
        const __m256i mask9 { 0x1ff01ff01ff01ffLL, 0x1ff01ff01ff01ffLL, 0x1ffLL, 0 };
        const __m256i mask1ff { 0x1ff01ff01ff01ffLL, 0x1ff01ff01ff01ffLL, 0x1ff01ff01ff01ffLL, 0x1ff01ff01ff01ffLL };

        __m256i column_or_tails[9];
        __m256i column_or_head = _mm256_setzero_si256();
        unsigned char irow = 0;
        for (unsigned char i = 0; i < 81; i += 9, irow++) {
            // columns
            // to start, we simply tally the or'ed rows
            __m256i column_mask = _mm256_setzero_si256();
            {
                if ( i == 0 ) {
                    // precompute 'tails' of the or'ed column_mask only once
                    // working backwords
                    for (signed char j = 81-9; j > 0; j -= 9) {
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
                    {
                        // breaking out the column hidden singles this way will win some performance
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
                            unsigned int mask = and_compress_masks<true>(a, m & 0x1ff);
                            if (mask) {
                                int idx = __tzcnt_u32(mask)>>1;
                                e_i = j+idx;
                                e_digit = ((v16us)column_mask_neg)[idx];
                                if ( e_digit & (e_digit-1) ) {
                                    if ( verbose ) {
                                        if ( grid_state->stackpointer == 0 && unique_check_mode == 0 ) {
                                            printf("Line %d: stack 0, back track - col cell [%d,%d] does contain multiple hidden singles\n", line, e_i/9, e_i%9);
                                        } else if ( debug ) {
                                            printf("back track - multiple hidden singles in col cell [%d,%d]\n", e_i/9, e_i%9);
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
                    }
                } else {  // i > 0
                    // leverage previously computed or'ed rows in head and tails.
                    column_or_head = column_mask = _mm256_or_si256(*(__m256i_u*) &candidates[i-9], column_or_head);
                    column_mask = _mm256_or_si256(column_or_tails[irow-1],column_mask);
                }
                // turn the or'ed rows into a mask for the singletons, if any.
                column_mask = _mm256_andnot_si256(column_mask, mask9);
            }

            // rows and boxes

            unsigned char b = box_start_by_boxindex[irow];
             __m256i c = _mm256_set_m128i(_mm_set_epi16(candidates[b+19], candidates[b+18], candidates[b+11], candidates[b+10], candidates[b+9], candidates[b+2], candidates[b+1], candidates[b]),
                         *(__m128i_u*) &candidates[i]);

            unsigned short the9thcand_row = candidates[i+8];
            unsigned short the9thcand_box = candidates[b+20];

            __m256i rowbox_or7 = _mm256_setzero_si256();
            __m256i rowbox_or8;
            __m256i rowbox_9th = _mm256_set_m128i(_mm_set1_epi16(the9thcand_box),_mm_set1_epi16(the9thcand_row));

            {
                __m256i c_ = c;
                for (unsigned char j = 0; j < 7; ++j) {
                    // rotate shift (1 2 3 4 5 6 7 8) -> (8 1 2 3 4 5 6 7)
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
                            printf("Line %d: stack 0, back track - %s cell [%d,%d] does contain multiple hidden singles\n", line, row_or_box, irow, idx%8);
                        } else if ( debug ) {
                            printf("back track - multiple hidden singles in %s cell [%d,%d]\n", row_or_box, irow, idx%8);
                        }
                    }
                    goto back;
                }

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
                            printf("Line %d: stack 0, back track - column and row intersection at [%d,%d] containing multiple hidden singles\n", line, irow, idx);
                        } else if ( debug ) {
                            printf("back track - column and row intersection at [%d,%d] contains multiple hidden singles\n", irow, idx);
                        }
                    }
                    goto back;
                }

                // check row/col (9) candidates
                unsigned short m = grid_state->unlocked.get_indexbits(i, 9);
                __m256i a = _mm256_cmpgt_epi16(or_mask, _mm256_setzero_si256());
                unsigned int mask = and_compress_masks<true>(a, m & 0x1ff);
                while (mask) {
                    int idx = __tzcnt_u32(mask)>>1;
                    if ( verbose && debug ) {
                        bool is_col = ((v16us)column_mask)[idx] == ((v16us)or_mask)[idx];
                        printf("hidden single (%s)", is_col?"col":"row");
                    }
                    e_i = i+idx;
                    e_digit = ((v16us)or_mask)[idx];
                    goto enter;
                }
            } // rowbox block
            { // box
                // we have already taken care of rows together with columns.
                // now look at the box.
                // First the (8) candidates, in the high half of rowbox_mask.
                __m256i a = _mm256_cmpgt_epi16(rowbox_mask, _mm256_setzero_si256());
                unsigned int mask = and_compress_masks<true>(a, (get_contiguous_masked_indices_for_box(unlocked,irow)&0xff)<<8);
                while (mask) {
                    int s_idx = __tzcnt_u32(mask)>>1;
                    int c_idx = b + box_offset[s_idx&7];
                    unsigned short digit = ((v16us)rowbox_mask)[s_idx];
                    if ( verbose && debug ) {
                        printf("hidden single (box)");
                    }
                    e_i = c_idx;
                    e_digit = digit;
                    goto enter;
                }
            } // box block

            // and at the very last, the nineth box cell
            if ( ((bit128_t*)unlocked)->check_indexbit(b+20) ) {
                unsigned cand9_box = ~((v16us)rowbox_or8)[8] & the9thcand_box;

                if ( cand9_box) {
                    int idx = b+20;
                    // check for a single.
                    // This is rare as it can only occur when a wrong guess was made.
                    // the current grid has no solution, go back
                    if ( cand9_box & (cand9_box-1) ) {
                        if ( verbose ) {
                            if ( grid_state->stackpointer == 0 && unique_check_mode == 0 ) {
                                printf("Line %d: stack 0, multiple hidden singles in box cell at [%d,%d]\n", line, idx/9, idx%9);
                            } else if ( debug ) {
                                printf("back track - multiple hidden singles in box cell at [%d,%d]\n", idx/9, idx%9);
                            }
                        }
                        goto back;
                    }
                    if ( verbose && debug ) {
                        printf("hidden single (box)");
                    }
                    e_i = idx;
                    e_digit = cand9_box;
                    goto enter;
                } // cand9
            }
        }   // for
    }

// Algorithm 3 - Find naked sets
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
// the complement has less than 3 member candidates, it is reported instead as a hidden set/pair. 
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
// - sets that occupy all available space (minus one) - impossible due to perfect single detection

    {
        bool found = false;

        // visit only the changed (updated) cells

        bit128_t to_visit_n;          // tracks all the cells to visit
        bit128_t to_visit_again {};   // those cells that have been updated

       to_visit_n.u64[0] = grid_state->updated.u64[0] & unlocked[0];
       to_visit_n.u64[1] = grid_state->updated.u32[2] & unlocked[1];

        // A cheap way to avoid unnecessary naked set searches
        grid_state->set23_found[Row].u64[0] |= ~unlocked[0];
        grid_state->set23_found[Row].u64[1] |= ~unlocked[1] & 0x1ffff;
        grid_state->set23_found[Col].u64[0] |= ~unlocked[0];
        grid_state->set23_found[Col].u64[1] |= ~unlocked[1] & 0x1ffff;
        grid_state->set23_found[Box].u64[0] |= ~unlocked[0];
        grid_state->set23_found[Box].u64[1] |= ~unlocked[1] & 0x1ffff;

        for (unsigned char n = 0; n < 2; ++n) {
            unsigned long long tvnn = to_visit_n.u64[n];
            while (tvnn) {
                unsigned char i_rel = tzcnt_and_mask(tvnn);
                unsigned char i = i_rel + (n<<6);

                unsigned short cnt = __popcnt16(candidates[i]);

                if (cnt <= MAX_SET && cnt > 1) {
                    // Note: this algorithm will never detect a naked set of the shape:
                    // {a,b},{a,c},{b,c} as all starting points are 2 bits only.
                    // same situation is possible for 4 set members.
                    //
                    unsigned long long to_change[2] {};

                    __m256i a_i = _mm256_set1_epi16(candidates[i]);
                    __m128i res;
                    unsigned char ul;
                    unsigned char s;

                    // check row
                    // Same comments as above apply
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
                            if (s > cnt) {
                                if ( verbose ) {
                                    char ret[32];
                                    format_candidate_set(ret, candidates[i]);
                                    if ( grid_state->stackpointer == 0 && unique_check_mode == 0 ) {
                                        printf("Line %d: naked  set (row) %s at [%d,%d], count exceeded\n", line, ret, ri, i%9);
                                    } else if ( debug ) {
                                        printf("back track sets (row) %s at [%d,%d], count exceeded\n", ret, ri, i%9);
                                    }
                                }
                                // no need to update grid_state
                                goto back;
                            } else if (s == cnt) {
                                char ret[32];
                                if ( s <= 3 ) {
                                    grid_state->set23_found[Row].set_indexbits(m,ri*9,9);
                                }
                                if ( ul <= 3 + s ) {
                                    // could include locked slots, never mind
                                    grid_state->set23_found[Row].set_indexbits(~m&0x1ff,ri*9,9);
                                }
                                if (cnt+2 <= ul) {
                                    naked_sets_found++;
                                    add_indices<Row>((bit128_t*)to_change, i);
                                    if ( verbose && debug ) {
                                        if ( cnt <=3 || cnt+3 < ul ) {
                                            format_candidate_set(ret, candidates[i]);
                                            printf("naked  %s (row): %-7s [%d,%d]\n", s==2?"pair":"set ", ret, ri, i%9);
                                        } else {
                                            unsigned char k = 0xff;
                                            unsigned short complement = 0;
                                            bit128_t mm {};
                                            mm.set_indexbits(~m & 0x1ff, ri*9, 9);
                                            for ( unsigned char n_=0; n_<2; n_++) {
                                                unsigned long long m_ = mm.u64[n_] & to_change[n_] & unlocked[n_];

                                                while (m_) {
                                                    unsigned char k_i = tzcnt_and_mask(m_) + (n_?64:0);
                                                    if ( k == 0xff ) {
                                                        k = k_i;
                                                    }
                                                    complement |= candidates[k_i];
                                                }
                                                if ( ri<7) {
                                                    break;
                                                }
                                            }
                                            complement &= ~candidates[i];
                                        if ( complement != 0 ) {
                                                format_candidate_set(ret, complement);
                                                printf("hidden %s (row): %-7s [%d,%d]\n", __popcnt16(complement)==2?"pair":"set ", ret, ri, k%9);
                                            }
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
                                            printf("Line %d: naked  set (%s) %s at [%d,%d], count exceeded\n", line, js[j], ret, i/9, i%9);
                                        } else if ( debug ) {
                                            printf("back track sets (%s) %s at [%d,%d], count exceeded\n", js[j], ret, i/9, i%9);
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
                                    }
                                    complement &= ~candidates[i];
                                    if ( complement != 0 ) {
                                        if ( verbose && debug ) {
                                            char ret[32];
                                            if ( set23_cond2 ) {
                                                format_candidate_set(ret, complement & ~candidates[i]);
                                                printf("hidden %s (%s): %-7s [%d,%d]\n", __popcnt16(complement)==2?"pair":"set ", js[j], ret, k/9, k%9);
                                            } else {
                                                format_candidate_set(ret, candidates[i]);
                                                printf("naked  %s (%s): %-7s [%d,%d]\n", ss[j]==2?"pair":"set ", js[j], ret, i/9, i%9);
                                            }
                                        }
                                    }
                                }
                            }
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
                            grid_state->updated.u64[0] = to_visit_n.u64[0];
                            grid_state->updated.u32[2] = to_visit_n.u64[1];
                            goto start;
                        }
                    }
                } else if ( cnt == 1 ) {
                    // this is not possible, but just to eliminate cnt == 1:
                    to_visit_n.u64[n] = tvnn;
                    to_visit_n.u128 |= to_visit_again.u128;
                    grid_state->updated.u64[0] = to_visit_n.u64[0];
                    grid_state->updated.u32[2] = to_visit_n.u64[1];
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
        grid_state->updated.u64[0] = to_visit_n.u64[0];
        grid_state->updated.u32[2] = to_visit_n.u64[1];
    }

    bit128_t bivalues;
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
        // check for a 'universal grave', which is an end-game move.
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
                            printf("bi-value universal grave means at least two solutions exist.\n");
                        }
                        goto guess;
                    } else if ( grid_state->stackpointer ) {
                        if ( verbose && debug ) {
                            printf("back track - found a bi-value universal grave.\n");
                        }
                        goto back;
                    } else {
                        if ( verbose && debug ) {
                            printf("bi-value universal grave means at least two solutions exist.\n");
                            non_unique_count++;
                        }
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
    grid_state = grid_state->make_guess<verbose>();
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
    -t# set the number of threads
    -u  check the solution for uniqueness
    -v  verify the solution
    -x  provide some statistics

)");
}

int main(int argc, const char *argv[]) {

    int line_to_solve = 0;

    if ( argc > 0 ) {
        argc--;
        argv++;
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
        printf("%10ld  %.0lf/s  puzzles solved\n", solved_count.load(), (double)solved_count.load()/((double)duration/1000000000LL));
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
        printf( "%10ld  %6.2f%%  puzzles solved without guessing\n", no_guess_cnt.load(), (double)no_guess_cnt.load()/(double)solved_count.load()*100);
        printf("%10lld  %6.2f/puzzle  guesses\n", guesses.load(), (double)guesses.load()/(double)solved_count.load());
        printf("%10lld  %6.2f/puzzle  back tracks\n", trackbacks.load(), (double)trackbacks.load()/(double)solved_count.load());
        printf("%10lld  %6.2f/puzzle  digits entered and retracted\n", digits_entered_and_retracted.load(), (double)digits_entered_and_retracted.load()/(double)solved_count.load());
        printf("%10lld  %6.2f/puzzle  'rounds'\n", past_naked_count.load(), (double)past_naked_count.load()/(double)solved_count.load());
        printf("%10lld  %6.2f/puzzle  naked sets found\n", naked_sets_found.load(), naked_sets_found.load()/(double)solved_count.load());
        printf("%10lld  %6.2f/puzzle  naked sets searched\n", naked_sets_searched.load(), naked_sets_searched.load()/(double)solved_count.load());

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
