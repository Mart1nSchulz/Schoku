// This code uses AVX2 instructions...
/*
 * Schoku
 *
 * A high speed sudoku solver by M. Schulz
 *
 * Based on the sudoku solver by Mirage ( https://codegolf.stackexchange.com/users/106606/mirage )
 * at https://codegolf.stackexchange.com/questions/190727/the-fastest-sudoku-solver
 * on Sep 22, 2021
 *
 * Version 0.5 (major capability update)
 *
 * Performance changes:
 * - full 256 bit hidden single search
 *   including box hidden singles
 * - 128 bit type
 * - bi-value universal grave detection
 * - dynamic support for pext/pdep BMI2 instructions
 *
 * Functional changes:
 * - full set of command line options
 *   correctness, uniqueness, puzzle validation, step-by-step progress
 *
 * Performance measurement and statistics:
 *
 * data: 17-clue sudoku
 * CPU:  Ryzen 7 4700U
 *
 * schoku version: 0.5
 *     49151  puzzles entered
 *    41.4ms   0.84µs/puzzle  solving time
 *     49151  puzzles solved
 *     34206  69.59%  puzzles solved without guessing
 *     38515   0.78/puzzle  total guesses
 *     24031   0.49/puzzle  total back tracks
 *    316780   6.45/puzzle  total digits entered and retracted
 *   1455691  29.62/puzzle  total 'rounds'
 *    143509   2.92/puzzle  naked sets found
 *   3757199  76.44/puzzle  naked sets searched
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

const char *version_string = "0.5";

const char *compilation_options = 
""
;


bool bmi2_support = false;


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
    inline bool check_indexbit(unsigned char idx) {
        return this->u16[idx>>4] & (1<<(idx & 0xf));
    }
    inline void set_indexbit(unsigned char idx) {
        this->u16[idx>>4] |= 1<<(idx & 0xf);
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
    // bit storage scheme specifically for tracking naked sets:
    // (optionally) 'interleaved' (il_) bits storage scheme.
    // If interleaved is false, each section occupies 9 contiguous bit,
    // repeated for each section, 81 bits total.
    // Note that this corresponds for Rows only to the pattern used in unlocked etc.
    // throughout.  For Cols and Boxes, its another story - essentially not convertable.
    //
    // When 'interleaved' is true, accommodate 'doubledbits' boolean vectors to save
    // some cycles.  Even less convertable than above.
    // In this scheme:
    // Each section occupies 9 bit, but interleaved spaced over 18 bits.
    // sections 0-4 are consecutive, over 90 bits (0..88).
    // sections 5-8 are offset by one and stored in the interleaved (0+1..70+1) bits.
    // All this to accommodate the 'doubledbits' returned by the movemask intrinsics.
    // (avoiding spending an extra call to obtain the same bit mask single spaced)
    // API:
    // The API for check/set is based on the section index ({row,col,box}_index,
    // and the offset (0..8) into the section.
    // computing the idx for the call is only cumbersome for the box kind.
    // The API for set_index_bits is even simpler, just specify the 18/9 bit mask and
    // section (0..8) where it goes to.
    //
    template<bool interleave>
    inline bool il_check_indexbit(unsigned char sct, unsigned char off) {
        if ( interleave ) {
            off = (off<<1) + ((sct>4)?1:0);
            return check_indexbit(sct%5*18+off);
        } else {
            return check_indexbit(sct*9+off);
        }
    }
    template<bool interleave>
    inline void il_set_indexbit(unsigned char sct, unsigned char off) {
        if ( interleave ) {
            off = (off<<1) + ((sct>4)?1:0);
            set_indexbit(sct%5*18+off);
        } else {
            return set_indexbit(sct*9+off);
        }
    }
    // bitcount is always 18-1 for 'interleaved' and 9 otherwise!
    // the 18 bits are 'normalized' by masking out the interleaved bits.
    template<bool interleave>
    inline void il_set_section_indexbits(unsigned long long mask, unsigned char section) {
        if ( interleave ) {
            set_indexbits(mask & 0b10101010101010101ULL, section%5*18 + (section>4?1:0), 18-1);
        } else {
            set_indexbits(mask, section*9, 9);
        }
    }
} bit128_t;

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

const unsigned char *row_index = index_by_kind[Row];
const unsigned char *column_index = index_by_kind[Col];
const unsigned char *box_index = index_by_kind[Box];

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

// this table provides the bit masks corresponding to each section index and each Kind of section.
// The 4th column contains all Kind's or'ed together. 
const unsigned long long small_index_lut[9][3][2] = {
{{              0x1ff,        0x0 }, { 0x8040201008040201u,     0x100 }, {           0x1c0e07,        0x0 }},
{{            0x3fe00,        0x0 }, {   0x80402010080402,      0x201 }, {           0xe07038,        0x0 }},
{{          0x7fc0000,        0x0 }, {  0x100804020100804,      0x402 }, {          0x70381c0,        0x0 }},
{{        0xff8000000,        0x0 }, {  0x201008040201008,      0x804 }, {     0xe07038000000,        0x0 }},
{{     0x1ff000000000,        0x0 }, {  0x402010080402010,     0x1008 }, {    0x70381c0000000,        0x0 }},
{{   0x3fe00000000000,        0x0 }, {  0x804020100804020,     0x2010 }, {   0x381c0e00000000,        0x0 }},
{{ 0x7fc0000000000000,        0x0 }, { 0x1008040201008040,     0x4020 }, { 0x81c0000000000000u,     0x703 }},
{{ 0x8000000000000000u,      0xff }, { 0x2010080402010080,     0x8040 }, {  0xe00000000000000,     0x381c }},
{{                0x0,    0x1ff00 }, { 0x4020100804020100,    0x10080 }, { 0x7000000000000000,    0x1c0e0 }},
};

// this table provides the bit masks corresponding to each index and each Kind of section.
// The 4th column contains all Kind's or'ed together. 
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
unsigned char fit_delta = 2;
// stats and command line options
int reportstats     = 0; // collect and report some statistics
int verify          = 0; // verify solution correctness (implied otherwise)
int unique_check    = 0; // check solution uniqueness
int debug           = 0; // provide step by step output on the solution
int thorough_check  = 0; // check for back tracking even if no guess was made.

std::atomic<long> solved_count(0);          // puzzles solved
std::atomic<long> unsolved_count = 0;       // puzzles unsolved (no solution exists)
std::atomic<long> non_unique_count = 0;     // puzzles not unique (with -u)
std::atomic<long> not_verified_count = 0;   // puzzles non verified (with -v)
std::atomic<long> verified_count = 0;       // puzzles successfully verified (with -v)
std::atomic<long> bug_count = 0;            // universal grave detected
std::atomic<long long> guesses(0);          // how many guesses did it take
std::atomic<long long> trackbacks(0);       // how often did we back track
std::atomic<long> no_guess_cnt(0);          // how many puzzles were solved without guessing
std::atomic<long long> past_naked_count(0); // how often do we get past the naked single serach
std::atomic<long long> naked_sets_searched(0); // how many naked sets did we search for
std::atomic<long long> naked_sets_found(0); // how many naked sets did we actually find
std::atomic<long long> digits_entered_and_retracted(0); // to measure guessing overhead

signed char *output;

inline __m128i expand_bitvector128(unsigned char m) {
    const __m128i bit_mask = _mm_setr_epi16(1<<0, 1<<1, 1<<2, 1<<3, 1<<4, 1<<5, 1<<6, 1<<7);
    return _mm_cmpeq_epi16(_mm_and_si128( bit_mask, _mm_set1_epi16(m)), bit_mask);
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
        return (unsigned long long)_mm256_movemask_epi8(b2) | _mm256_movemask_epi8(b1);
    }
    if ( bmi2_support ) {
        return _pext_u64((unsigned long long)_mm256_movemask_epi8(b2) | _mm256_movemask_epi8(b1),0x5555555555555555ull);
    } else {
        return _mm256_movemask_epi8(_mm256_permute4x64_epi64(_mm256_packs_epi16(b1,b2), 0xD8));
    }
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
//    Complication: the ep16 op boolean gives two bits for each element.
//    compress the wide boolean to 2 or 1 bits as desired
//    and with the bitvector modified to doubled bits if desired
//    using template parameter doubledbits.
//    Performance: pdep/pext are expensive on AMD Zen2, but good when available elsewhere.
//    If you want to run this on a AMD Zen2 computer, define NO_ZEN2_BMI2 to prevent
//    use of the offending pext/pdep instructions, or just don't compile for BMI2.
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

// combine an epi16 boolean mask of 128 bits and a 8-bit mask to an 8/16-bit mask.
// Same considerations as above apply.
// 

template<bool doubledbits=false>
inline __attribute__((always_inline)) unsigned short and_compress_masks128(__m128i a, unsigned short b) {
    if ( bmi2_support ) {
// path 1:
        unsigned int res = compress_epi16_boolean128<doubledbits>(a);
        if (doubledbits) {
            return res & _pdep_u32(b, 0x5555);
        } else {
            return res & b;
        }
    } else {
// path 2:
        return compress_epi16_boolean128<doubledbits>(_mm_and_si128(a, expand_bitvector128(b)));
    }
}

// for the 9 aligned cells of a box as a vector (given by the index of the box),
// return the corresponding masking bits from indices (i.e. unlocked)
inline unsigned int get_box_unlocked_mask(unsigned long long indices[2], int boxi) {

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
inline void and_indices(bit128_t *indices, unsigned char i) {
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

#define	GRIDSTATE_MAX 28

class __attribute__ ((aligned(32))) GridState
{
public:
    bit128_t unlocked;           // for keeping track of which cells don't need to be looked at anymore. Set bits correspond to cells that still have multiple possibilities
    bit128_t updated;            // for keeping track of which cell's candidates may have been changed since last time we looked for naked sets. Set bits correspond to changed candidates in these cells
    unsigned short candidates[81];        // which digits can go in this cell? Set bits correspond to possible digits
    bit128_t set23_found[3];     // for keeping track of found sets of size 2 and 3
    short stackpointer;                   // this-1 == last grid state before a guess was made, used for backtracking
// An array GridState[n] can use stackpointer to increment until it reaches n.
// Similary, for back tracking, the relative GridState-1 is considered the ancestor
// until stackpointer is 0.
//
// GridState is normally copied for recursion
// Here we initialize the starting state including the puzzle.
//
inline void initialize(signed char grid[81]) {
    // 0x1ffffffffffffffffffffULLL is (0x1ULL << 81) - 1
    updated.u128 = unlocked.u128 = (((__uint128_t)1)<<81)-1;

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
                unlocked.u64[0] &= ~(1ULL << i);
            }
        }
        for (unsigned char i = 64; i < 81; ++i) {
            digit = grid[i] - 49;
            if (digit >= 0) {
                digit = 1 << digit;
                columns[column_index[i]] |= digit;
                rows[row_index[i]]       |= digit;
                boxes[box_index[i]]      |= digit;
                unlocked.u64[1] &= ~(1ULL << (i-64));
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

template<bool verbose=false>
inline __attribute__((always_inline)) void enter_digit( unsigned short digit, unsigned char i) {
    // lock this cell and and remove this digit from the candidates in this row, column and box

    bit128_t to_update = {0};

    if ( verbose && debug ) {
        printf(" %x found at [%d,%d]\n", _tzcnt_u32(digit)+1, i/9, i%9);
    }
#ifndef NDEBUG
    if ( __popcnt16(digit) != 1 ) {
        printf("error in enter_digit: %x\n", digit);
    }
#endif

    if (i < 64) {
        unlocked.u64[0] &= ~(1ULL << i);
    } else {
        unlocked.u64[1] &= ~(1ULL << (i-64));
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
    while (to_visit != 0 && best_cnt > 2) {
		i_rel = _tzcnt_u64(to_visit);
        to_visit &= ~(1ULL << i_rel);
        cnt = __popcnt16(candidates[i_rel]);
        if (cnt < best_cnt) {
            best_cnt = cnt;
            guess_index = i_rel;
        }
    }

    to_visit = unlocked.u64[1];
    while (best_cnt > 2 && to_visit != 0 ) {
		i_rel = _tzcnt_u32(to_visit);
        to_visit &= ~(1UL << i_rel);
        cnt = __popcnt16(candidates[i_rel + 64]);
        if (cnt < best_cnt) {
            best_cnt = cnt;
            guess_index = i_rel + 64;
        }
    }
    
    // Find the first candidate in this cell
	unsigned short digit = 1 << __bsrd(candidates[guess_index]);
    
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
    if (guess_index < 64) {
        updated.u64[0] |= 1ULL << guess_index;
    } else {
        updated.u64[1] |= 1ULL << (guess_index-64);
    }

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

template<bool verbose>
GridState* make_guess(unsigned short digit, unsigned char guess_index) {
    // This is for an educated guess only.

    GridState* new_grid_state = this+1;

    if ( verbose && (debug > 1) ) {
        char gridout[82];
        for (unsigned char j = 0; j < 81; ++j) {
            if ( (candidates[j] & (candidates[j]-1)) ) {
                gridout[j] = '0';
            } else {
                gridout[j] = 49+__bsrd(candidates[j]);
            }
        }
        printf("educated guess at [%d,%d]\nsaved grid_state level >%d<: %.81s\n",
               guess_index/9, guess_index%9, stackpointer, gridout);
    }
    
    memcpy(new_grid_state, this, sizeof(GridState));
    new_grid_state->stackpointer++;

    // Remove the guessed candidate from the old grid
    // when we get back here to the old grid, we know the guess was wrong
    candidates[guess_index] &= ~digit;
    if (guess_index < 64) {
        updated.u64[0] |= 1ULL << guess_index;
    } else {
        updated.u64[1] |= 1ULL << (guess_index-64);
    }

    // Update candidates
    if ( verbose && debug ) {
        printf("educated guess at level >%d< - new level >%d<\nguess", stackpointer, new_grid_state->stackpointer);
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
                gridout[j] = 49+__bsrd(candidates[j]);
            }
        }
    printf("grid_state at level >%d< now: %s\n",
               new_grid_state->stackpointer, gridout);
    }    
    return new_grid_state;
}

template<Kind kind>
inline unsigned char get_ul_set_search( bool &fitm1, unsigned char cnt, unsigned char si) {
    unsigned char ret = (kind == Row) ? 9-get_section_index_cnt<kind>(set23_found[Row].u64, si) : get_section_index_cnt<kind>(unlocked.u64, si);
    fitm1 = cnt < ret;
    return ret;
}
};

// The pair of functions below can be used to iteratively isolate all distinct bit values
// and determine whether popcnt(X) == N is true for the input vector elements using movemask
// at each iteration of interest.
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

    int unique_check_mode = 0;
    bool unique_check_reported = false;

   unsigned long long my_digits_entered_and_retracted = 0;
   unsigned long long my_naked_sets_searched = 0;
   unsigned char no_guess_incr = 1;

    unsigned int my_past_naked_count = 0;
    if ( verbose && debug ) {
        printf("Line %d: %.81s\n", line, grid);
    }

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
            digits_entered_and_retracted += my_digits_entered_and_retracted;
        }
        return true;
    }

    // collect some guessing stats
    if ( verbose && reportstats ) {
        my_digits_entered_and_retracted += 
            (_popcnt64((grid_state-1)->unlocked.u64[0] & ~grid_state->unlocked.u64[0]))
          + (_popcnt64((grid_state-1)->unlocked.u64[1] & ~grid_state->unlocked.u64[1]));
    }

    // Go back to the state when the last guess was made
    // This state had the guess removed as candidate from it's cell

    if ( verbose && debug ) {
        printf("back track to level >%d<\n", grid_state->stackpointer-1);
    }
    trackbacks++;
    grid_state--;

start:

    bool check_back = grid_state->stackpointer || thorough_check || unique_check_mode;

    unsigned long long *unlocked = grid_state->unlocked.u64;
    unsigned short* candidates = grid_state->candidates;

    // Find naked singles
    {
        bool found;
        unsigned char found_idx = 0xff;
        const __m256i ones = _mm256_set1_epi16(1);
        do {
            found = false;
            for (unsigned char i = 0; i < 80; i += 16) {
                unsigned short m = ((bit128_t*)unlocked)->u16[i>>4];
                if ( m ) {
                    __m256i c = _mm256_load_si256((__m256i*) &candidates[i]);
                    // Check if any cell has zero candidates
                    if (check_back && _mm256_movemask_epi8(_mm256_cmpeq_epi16(c, _mm256_setzero_si256()))) {
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
                        // (c = c & (c-1)) == 0  => naked single
                        __m256i a = _mm256_cmpeq_epi16(_mm256_and_si256(c, _mm256_sub_epi16(c, ones)), _mm256_setzero_si256());
                        unsigned int mask = and_compress_masks<true>(a,m);
                        while (mask) {     // note that the mask has two bits set for each detected single
                            int idx3 = _tzcnt_u32(mask);
                            int idx = (idx3>>1) + i;
                            mask &= ~(3<<idx3);
                            // the candidate could have disappeared due to the last enter_digit
                            // in this loop.
                            if ( check_back && candidates[idx] == 0 ) {
                                if ( verbose && debug ) {
                                    printf("back track - cell [%d,%d] is 0\n", idx/9, idx%9);
                                }
                                goto back;
                            }
                            if ( verbose && debug ) {
                                printf("naked  single");
                            }
                            grid_state->enter_digit<verbose>( candidates[idx], idx);
                            found = true;
                            found_idx = i;
                        }
                        if (  found == false && found_idx == i ) {
                            break;
                        }
                    }
                }
            }
            if (unlocked[1] & (1ULL << (80-64))) {
                if (candidates[80] == 0) {
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
                        printf("naked  single");
                    }
                    grid_state->enter_digit<verbose>( candidates[80], 80);
                    found = true;
                    found_idx=72;		// run a full loop next.
                }
            }
        } while (found);
    }
    
    // Check if it's solved, if it ever gets solved it will be solved after looking for naked singles
    if ( *(__uint128_t*)unlocked == 0) {
        // Solved it
        if ( unique_check == 1 && unique_check_mode == 1 ) {
            if ( !unique_check_reported ) {
                if ( verbose && reportstats ) {
                    printf("Line %d: solution to puzzle is not unique\n", line);
                }
                non_unique_count++;
                unique_check_reported = true;
            }
        }
        if ( verify ) {
            // quickly assert that the solution is valid
            // no cell has more than one digit set
            // all rows, columns and boxes have all digits set.

            const __m256i mask9 { -1LL, -1LL, 0xffffLL, 0 };
            const __m256i ones = _mm256_and_si256(_mm256_set1_epi16(1), mask9);
            __m256i rowx = _mm256_and_si256(_mm256_set1_epi16(0x1ff),mask9);
            __m256i colx = _mm256_and_si256(_mm256_set1_epi16(0x1ff),mask9);
            __m256i boxx = _mm256_and_si256(_mm256_set1_epi16(0x1ff),mask9);
            __m256i uniq = _mm256_setzero_si256();

            for (unsigned char i = 0; i < 9; i++) {
                // load element i of 9 rows
                __m256i row = _mm256_set_epi16(0, 0, 0, 0, 0, 0, 0, candidates[i+72],
                              candidates[i+63], candidates[i+54], candidates[i+45], candidates[i+36], candidates[i+27], candidates[i+18], candidates[i+9], candidates[i]);
                rowx = _mm256_xor_si256(rowx,row);

                // load element i of 9 columns
                __m256i col = _mm256_and_si256(_mm256_loadu_si256((__m256i_u*) &candidates[i*9]),mask9);
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

    // Find hidden singles
    //
    // Combine 8 cells from rows and 8 cells from boxes into one __m256i vector.
    // Rotate and or until each vector element represents 7 cells or'ed (except the cell 
    // directly corresponding to its position, containing the hidden single if there is one).
    // Broadcast the nineth cell and or it for good measure, then andnot with 0x1ff
    // to isolate the hidden singles.
    // For the nineth cell, rotate and or one last time and use just one element of the result
    // to check the nineth cell for a hidden single.
    // columns are or'ed together, leaving out the current row, and hidden column singles are
    // isolated for that row. For efficiency, precompute and save the or'ed rows (tails) and
    // preserve the last leading set of rows (the head).
    // To check for an invalid state of the puzzle:
    // - check the columns or'ed value to be 0x1ff.
    // - same for the rows and box
    // The column singles and the row singles are first or'ed and checked to be singles.
    // Otherwise if the row and column checks disagree on a cell the last guess was wrong.
    // If everything is in order, compare the singles (if any) against the current row.
    // Compress to a bit vector and check against the unlocked cells. If any single was found
    // isolate it, rince and repeat.
    //
    {
        const __m256i ones = _mm256_set1_epi16(1);
        const __m256i mask9 { 0x1ff01ff01ff01ffLL, 0x1ff01ff01ff01ffLL, 0x1ffLL, 0 };
        const __m256i mask1ff { 0x1ff01ff01ff01ffLL, 0x1ff01ff01ff01ffLL, 0x1ff01ff01ff01ffLL, 0x1ff01ff01ff01ffLL };

        __m256i column_or_tails[8];
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
                        column_mask = _mm256_or_si256(column_mask, _mm256_loadu_si256((__m256i_u*) &candidates[j]));
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
                } else {
                    // leverage previously computed or'ed rows in head and tails.
                    column_or_head = column_mask = _mm256_or_si256(_mm256_loadu_si256((__m256i_u*) &candidates[i-9]), column_or_head);
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
                unsigned short m = (i < 64) ? (unlocked[0] >> i) : (unlocked[1] >> (i-64));
                if ( i > 64-9) {
                    m |= unlocked[1] << (64-i);
                }
                __m256i a = _mm256_cmpgt_epi16(_mm256_and_si256(*(__m256i_u*)&candidates[i], or_mask), _mm256_setzero_si256());
                unsigned int mask = and_compress_masks<true>(a, m & 0x1ff);
                while (mask) {

                    int idx = __tzcnt_u32(mask)>>1;
                    if ( verbose && debug ) {
                        bool is_col = ((v16us)column_mask)[idx] == ((v16us)or_mask)[idx];
                        printf("hidden single (%s)", is_col?"col":"row");
                    }
                    grid_state->enter_digit<verbose>( ((v16us)or_mask)[idx], i+idx);
                    goto start;
                }
            } // rowbox block
            { // box
                // we have already taken care of rows together with columns.
                // now look at the box.
                // First the (8) candidates, in the high half of rowbox_mask.
                __m256i a = _mm256_cmpgt_epi16(rowbox_mask, _mm256_setzero_si256());
                unsigned int mask = and_compress_masks<true>(a, (get_box_unlocked_mask(unlocked,irow)&0xff)<<8);
                while (mask) {
                    int s_idx = __tzcnt_u32(mask)>>1;
                    int c_idx = b + box_offset[s_idx&7];
                    unsigned short digit = ((v16us)rowbox_mask)[s_idx];
                    if ( verbose && debug ) {
                        printf("hidden single (box)");
                    }
                    grid_state->enter_digit<verbose>( digit, c_idx);
                    goto start;
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
                    grid_state->enter_digit<verbose>( cand9_box, idx);
                    goto start;
                } // cand9
            }
        }   // for
    }   // box

// Find naked sets, up to MAX_SET

#define MAX_SET 5
// Some general thoughts on the algorithm below.
//
// What does this search achieve:
// for a row with N unlocked cells each naked set of K corresponds to a hidden set of N-K.
// Going up to size 5 allows to get at the very least all hidden sets of 4, and most likely
// (with N<9) all hidden sets of 3.  There is a small gap concerning hidden pairs (fairly
// common), which will then be detected only with N<7. On the other hand, since this
// algorithm is hit repeatedly, they will eventually resolve too.
//
// The other thing that is achieved, and not to be neglected, is the ability to detect back
// track scenarios if the discovered set is impossibly large.  This is quite important
// for performance as a last chance to kill of bad guesses.
//
// Note that this search has it's own built-in heuristic to tackle only recently updated cells.
// The algorithm will keep that list to revisit later, which is fine of course.
// The number of cells to visit is expectedly pretty high.
//  We could try other tricks around this area of 'work avoidance'.
//
    {
        bool can_backtrack = (grid_state->stackpointer != 0);

        bool found = false;
        // visit only the changed (updated) cells
        
        // for naked sets only. Can be set to true or false.
        const bool doubledbits = false;
        unsigned long long *to_visit_n = grid_state->updated.u64;
        *(__uint128_t*)to_visit_n &= *(__uint128_t*)unlocked;

        // A cheap way to avoid unnecessary naked set searches, at least for Rows
        grid_state->set23_found[Row].u64[0] |= ~unlocked[0];
        grid_state->set23_found[Row].u64[1] |= ~unlocked[1];
        for (unsigned char n = 0; n < 2; ++n) {
            unsigned long long tvnn = to_visit_n[n];
            while (tvnn) {
                int i_rel = _tzcnt_u64(tvnn);
                
                tvnn ^= 1ULL << i_rel;
                unsigned char i = (unsigned char) i_rel + 64*n;
                
                unsigned short cnt = __popcnt16(candidates[i]);

                if (cnt <= MAX_SET && cnt > 1) {
                    // Note: this algorithm will never detect a naked set of the shape:
                    // {a,b},{a,c},{b,c} as all starting points are 2 bits only.
                    // same situation is possible for 4 set members.
                    //
                    unsigned long long to_change[2] = {0};

                    __m128i a_i = _mm_set1_epi16(candidates[i]);
                    __m128i a_j;
                    __m128i res;
                    unsigned char ul;
                    bool fitm1;
                    unsigned char s;

                    // check row
                    // Same comments as above apply
                    //
                  unsigned char ri = row_index[i];
                  if ( !grid_state->set23_found[Row].il_check_indexbit<doubledbits>(ri, i%9))
                  {
                    ul = grid_state->get_ul_set_search<Row>(fitm1, cnt, ri);
                    if (can_backtrack || (cnt <= ul-fit_delta) ) {
                        my_naked_sets_searched++;
                        a_j = _mm_loadu_si128((__m128i_u*) &candidates[9*ri]);
                        res = _mm_cmpeq_epi16(a_i, _mm_or_si128(a_i, a_j));
                        unsigned long long m = compress_epi16_boolean128<doubledbits>(res);
                        bool bit9 = candidates[i] == (candidates[i] | candidates[9*ri+8]);
                        if ( bit9 ) {
                            m |= doubledbits? 3<<16 : 1<<8;    // fake the 9th mask position
                        }
                        s = _popcnt32(m);
                        if ( doubledbits ) {
                            s >>= 1;
                        }
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
                        } else if (s == cnt && fitm1) {
                            naked_sets_found++;
                            if ( s <= 3 ) {
                                grid_state->set23_found[Row].il_set_section_indexbits<doubledbits>(m,ri);
                            }
                            if ( verbose && debug ) {
                                char ret[30];
                                format_candidate_set(ret, candidates[i]);
                                printf("naked  set (row): %s [%d,%d]\n", ret, ri, i%9);
                            }
                            and_indices<Row>((bit128_t*)to_change, i);
                        }
                    }
                  }

                    // check column
                    //
                  unsigned char ci = column_index[i];
                  if ( !grid_state->set23_found[Col].il_check_indexbit<doubledbits>(ci,i/9))
                  {
                    ul = grid_state->get_ul_set_search<Col>(fitm1, cnt, ci);
                    if (can_backtrack || (cnt <= ul-fit_delta) ) {
                        my_naked_sets_searched++;
                        a_j = _mm_set_epi16(candidates[ci+63], candidates[ci+54], candidates[ci+45], candidates[ci+36], candidates[ci+27], candidates[ci+18], candidates[ci+9], candidates[ci]);
                        res = _mm_cmpeq_epi16(a_i, _mm_or_si128(a_i, a_j));
                        unsigned long long m = compress_epi16_boolean128<doubledbits>(res);;
                        bool bit9 = candidates[i] == (candidates[i] | candidates[ci+72]);
                        if ( bit9 ) {
                            m |= doubledbits? 3<<16 : 1<<8;    // fake the 9th mask position
                        }
                        s = _popcnt32(m);
                        if ( doubledbits ) {
                            s >>= 1;
                        }
                        // this covers the situation where there is a naked set of size x
                        // which is found in y cells with y > x.  That's impossible, hence track back.
                        if (s > cnt) {
                            if ( verbose ) {
                                    char ret[32];
                                    format_candidate_set(ret, candidates[i]);
                                if ( grid_state->stackpointer == 0 && unique_check_mode == 0 ) {
                                        printf("Line %d: naked  set (col) %s at [%d,%d], count exceeded\n", line, ret, i/9, i%9);
                                } else if ( debug ) {
                                        printf("back track sets (col) %s at [%d,%d], count exceeded\n", ret, i/9, i%9);
                                }
                            }
                            // no need to update grid_state
                            goto back;
                        } else if (s == cnt && fitm1 ) {
                            naked_sets_found++;


                            if ( s <= 3 ) {
                                grid_state->set23_found[Col].il_set_section_indexbits<doubledbits>(m,ci);
                            }
                            if ( verbose && debug ) {
                                char ret[30];
                                format_candidate_set(ret, candidates[i]);
                                printf("naked  set (col): %s [%d,%d]\n", ret, i/9, i%9);
                            }
                            and_indices<Col>((bit128_t*)to_change, i);
                        }
                    }
                  }

                    // check box
                    // Same comments as above apply
                    //
                    // If there are n cells, cnt == n will just cause churn,
                    // but cnt == n - 1 will detect hidden singles.
                    //
                  unsigned char b    = box_start[i];
                  unsigned char boff = i-b;
                  boff = boff/9*3 + boff%3;
                  unsigned char bi   = box_index[i];
                  if ( !grid_state->set23_found[Box].il_check_indexbit<doubledbits>(bi,boff))
                  {
                    ul = grid_state->get_ul_set_search<Box>(fitm1, cnt, bi);
                    if (can_backtrack || (cnt <= ul-fit_delta) ) {
                        my_naked_sets_searched++;
                        a_j = _mm_set_epi16(candidates[b+19], candidates[b+18], candidates[b+11], candidates[b+10], candidates[b+9], candidates[b+2], candidates[b+1], candidates[b]);
                        res = _mm_cmpeq_epi16(a_i, _mm_or_si128(a_i, a_j));
                        unsigned long long m = compress_epi16_boolean128<doubledbits>(res);
                        bool bit9 = candidates[i] == (candidates[i] | candidates[b+20]);
                        if ( bit9 ) {
                            m |= doubledbits? 3<<16 : 1<<8;    // fake the 9th mask position
                        }
                        s = _popcnt32(m);
                        if ( doubledbits ) {
                            s >>= 1;
                        }
                        if (s > cnt) {
                            if ( verbose ) {
                                if ( grid_state->stackpointer == 0 && unique_check_mode == 0 ) {
                                    printf("Line %d: naked set box, count exceeded\n", line);
                                } else if ( debug ) {
                                    printf("back track sets (box)\n");
                                }
                            }
                            // no need to update grid_state
                            goto back;
                        } else if (s == cnt && fitm1) {
                            naked_sets_found++;
                            if ( verbose && debug ) {
                                char ret[30];
                                format_candidate_set(ret, candidates[i]);
                                printf("naked  set (box): %s [%d,%d]\n", ret, i/9, i%9);
                            }
                            if ( s <= 3 ) {
                                grid_state->set23_found[Box].il_set_section_indexbits<doubledbits>(m,bi);
                            }
                            and_indices<Box>((bit128_t*)to_change, i);
                        }
                    }
                  }

                    ((bit128_t*)to_change)->u128 &= ((bit128_t*)unlocked)->u128;
                    
                    // update candidates
                    for (unsigned char n = 0; n < 2; ++n) {
                        while (to_change[n]) {
                            int j_rel = _tzcnt_u64(to_change[n]);
                            to_change[n] &= ~(1ULL << j_rel);
                            unsigned char j = (unsigned char) j_rel + (n<<6);
                            
                            // if this cell is not part of our set
                            if ((candidates[j] | candidates[i]) != candidates[i]) {
                                // if there are bits that need removing
                                if (candidates[j] & candidates[i]) {
                                    candidates[j] &= ~candidates[i];
                                    tvnn |= 1ULL << j_rel;
                                    found = true;
                                }
                            }
                        }
                    }
                    
                    // If any cell's candidates got updated, go back and try all that other stuff again
                    if (found) {
                        to_visit_n[n] = tvnn;
                        goto start;
                    }
                    
                } else if ( cnt == 1 ) {
                    // this is not possible, but just to eliminate cnt == 1:
                    if ( verbose && debug ) {
                        printf("naked  (sets) ");
                    }
                    grid_state->enter_digit<verbose>( candidates[i], i);
                    printf("found a singleton in set search... strange\n");
                    goto start;
                }
            }
        }
    }

    bit128_t bivalues {0};
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
        // now bivalues is now set for subsequent steps
    }

    {
        // Before making a guess,
        // check for a 'universal grave', which is an end-game move.
        // First get a count of unlocked.
        // With around 22 or less unresolved cells (N), accumulate a popcount of all cells.
        // Count the cells with a candidate count 2 (P). 81 - N - P = Q.
        // If Q is 0, track back, if a guess was made before. without guess to track back to,
        // take note as duplicate solution, but carry on with a guess.
        // If Q > 1, go with a guess.
        // If Q == 1, identify the only cell which does not have a 2 candidate count.
        // If the candidate count is 3, determine for any section with the cell which
        // candidate value appears in this cell and other cells of the section 3 times.
        // That is the correct solution for this cell, enter it.

        unsigned char N = _popcnt64(unlocked[0]) + _popcnt32(unlocked[1]);
        if ( N < 23 ) {
                unsigned char sum12s = 0;
                unsigned char target = 0;

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
                if ( sum12s == 81 ) {
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
                        grid_state->enter_digit<verbose>(digit, target);
                        goto start;
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
#pragma omp parallel for firstprivate(stack) shared(string_pre, output, npuzzles, i, imax, debug, reportstats) schedule(dynamic,32)
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
		printf("%8.1lfms  %5.2lf\u00b5s/puzzle  solving time\n", (double)duration/1000000, (double)duration/(npuzzles*1000LL));
        printf("%10ld  puzzles solved\n", solved_count.load());
        if ( unsolved_count.load()) {
            printf("%10ld  puzzles had no solution\n", unsolved_count.load());
        }
        if ( unique_check ) {
            printf("%10ld  puzzles had a unique solution\n", solved_count.load() - non_unique_count.load());
        }
        if ( verify ) {
            printf("%10ld  puzzle solutions were verified\n", verified_count.load());
        }
        printf("%10ld  %5.2f%%  puzzles solved without guessing\n", no_guess_cnt.load(), (double)no_guess_cnt.load()/(double)solved_count.load()*100);
        printf("%10lld  %5.2f/puzzle  total guesses\n", guesses.load(), (double)guesses.load()/(double)solved_count.load());
        printf("%10lld  %5.2f/puzzle  total back tracks\n", trackbacks.load(), (double)trackbacks.load()/(double)solved_count.load());
        printf("%10lld  %5.2f/puzzle  total digits entered and retracted\n", digits_entered_and_retracted.load(), (double)digits_entered_and_retracted.load()/(double)solved_count.load());
        printf("%10lld  %5.2f/puzzle  total 'rounds'\n", past_naked_count.load(), (double)past_naked_count.load()/(double)solved_count.load());
        printf("%10lld  %5.2f/puzzle  naked sets found\n", naked_sets_found.load(), naked_sets_found.load()/(double)solved_count.load());
        printf("%10lld %6.2f/puzzle  naked sets searched\n", naked_sets_searched.load(), naked_sets_searched.load()/(double)solved_count.load());
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
