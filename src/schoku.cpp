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
 * Version 0.3
 *
 * Performance changes:
 * - only try to find naked sets that are worthwhile
 * - GridState acquires the methods enter_digit, track_back and make_guess
 * - lookup tables
 * - factored out inline functions
 *
 * Functional changes:
 * - -l# option to solve a single line
 *
 * Basic performance measurement and statistics:
 *
 * data: 17-clue sudoku
 * CPU:  Ryzen 7 4700U
 *
 * schoku version: 0.3
 *     49151  puzzles entered
 *    33.7ms   0.69µs/puzzle  solving time
 *     32508  66.14%  puzzles solved without guessing
 *     46912   0.95/puzzle  total guesses
 *     30177   0.61/puzzle  total back tracks
 *    401420   8.17/puzzle  total digits entered and retracted
 *   1509922  30.72/puzzle  total 'rounds'
 *    481278   9.79/puzzle  naked sets found
 *   5334164 108.53/puzzle  naked sets searched
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

const char *version_string = "0.3";

typedef
enum Kind {
   Row = 0,
   Col = 1,
   Box = 2,
   All = 3, // special case for look up table.
} Kind;

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

// for each index I and each Kind of section, plus all or'ed together, this table provides the bit masks corresponding to that section. 
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

// stats and command line options
bool reportstats     = false; // collect and report some statistics

std::atomic<long> solved_count(0);          // puzzles solved
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
    unsigned short res = 0;
    if (doubledbits) {
        return _mm_movemask_epi8(b);
    } else {
#if defined(__BMI2__) && !defined(NO_ZEN2_BMI2)
        res = _pext_u32(_mm_movemask_epi8(b), 0x5555);
#else
        res = _mm_movemask_epi8(_mm_packus_epi16(_mm_min_epu16(b, _mm_set1_epi32(0x00FF00FF)), _mm_setzero_si128()));
#endif
    }
    return res;
}

template<bool doubledbits=false>
inline __attribute__((always_inline)) unsigned int compress_epi16_boolean(__m256i b) {
    unsigned int res = 0;
    if (doubledbits) {
        return _mm256_movemask_epi8(b);
    } else {
#if defined(__BMI2__) && !defined(NO_ZEN2_BMI2)
        res = _pext_u32(_mm256_movemask_epi8(b),0x55555555);
#else
        res = _mm_movemask_epi8(_mm_packs_epi16(_mm256_castsi256_si128(b), _mm256_extractf128_si256(b,1)));
#endif
    }
    return res;
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
#if defined(__BMI2__) && !defined(NO_ZEN2_BMI2)
// path 1:
    unsigned int res = compress_epi16_boolean<doubledbits>(a);
    if (doubledbits) {
        return res & _pdep_u32(b,0x55555555);
    } else {
        return res & b;
    }
#else
// path 2:
    return compress_epi16_boolean<doubledbits>(_mm256_and_si256(a, expand_bitvector(b)));
#endif
}

// combine an epi16 boolean mask of 128 bits and a 8-bit mask to an 8/16-bit mask.
// Same considerations as above apply.
// 

template<bool doubledbits=false>
inline __attribute__((always_inline)) unsigned short and_compress_masks128(__m128i a, unsigned short b) {
#if defined(__BMI2__) && !defined(NO_ZEN2_BMI2)
// path 1:
    unsigned int res = compress_epi16_boolean128<doubledbits>(a);
    if (doubledbits) {
        return res & _pdep_u32(b, 0x5555);
    } else {
        return res & b;
    }
#else
// path 2:
    return compress_epi16_boolean128<doubledbits>(_mm_and_si128(a, expand_bitvector128(b)));
#endif
} 

template<Kind kind>	//
inline unsigned char get_unlinked_cnt(unsigned long long indices[2], unsigned char i) {
    const unsigned long long *index = big_index_lut[i][kind];
    return _popcnt64(indices[0] & index[0])
         + _popcnt32(indices[1] & index[1]);
}

inline void add_and_mask_all_indices(unsigned long long indices[2], unsigned long long mask[2], unsigned char i) {
    const unsigned long long *index = big_index_lut[i][All];
	indices[0] = (indices[0] | index[0]) & mask[0];
	indices[1] = (indices[1] | index[1]) & mask[1];
}

template<Kind kind>
inline void add_indices(unsigned long long indices[2], unsigned char i) {
    const unsigned long long *index = big_index_lut[i][kind];
    indices[0] |= index[0];
    if ( kind == 1 || i >= 54 ) {	// above this threshold, index[1] is populated for boxes, (not always) for rows
        indices[1] |= index[1];
    }
}

#define	GRIDSTATE_MAX 28

class __attribute__ ((aligned(32))) GridState
{
public:
    unsigned long long unlocked[2];        // for keeping track of which cells don't need to be looked at anymore. Set bits correspond to cells that still have multiple possibilities
    unsigned long long updated[2];         // for keeping track of which cell's candidates may have been changed since last time we looked for naked sets. Set bits correspond to changed candidates in these cells
    unsigned short candidates[81];         // which digits can go in this cell? Set bits correspond to possible digits
	unsigned short filler[4];
    short stackpointer = 0;                // this-1 == last grid state before a guess was made, used for backtracking
// An array GridState[n] can use stackpointer to increment until it reaches n.
// Similary, for back tracking, the relative GridState-1 is considered the ancestor
// until stackpointer is 0. 
//
// GridState is normally copied for 'recursion', so there is
// only the starting state to be initialized.
//
inline void init() {
    updated[0] = unlocked[0] = 0xffffffffffffffffULL;
    updated[1] = unlocked[1] = 0x1ffffULL;
}

inline __attribute__((always_inline)) void enter_digit( unsigned short digit, unsigned char i) {
    // lock this cell and and remove this digit from the candidates in this row, column and box
    
    unsigned long long to_update[2] = {0};
    
    if (i < 64) {
        unlocked[0] &= ~(1ULL << i);
    } else {
        unlocked[1] &= ~(1ULL << (i-64));
    }
    
    candidates[i] = digit;

    add_and_mask_all_indices(to_update, unlocked, i);

    updated[0] |= to_update[0];
    updated[1] |= to_update[1];
    
    const __m256i mask = _mm256_set1_epi16(~digit);
    for (unsigned char j = 0; j < 80; j += 16) {
        unsigned short m = (j < 64) ? (to_update[0] >> j) : to_update[1];
        __m256i c = _mm256_load_si256((__m256i*) &candidates[j]);
        // expand ~m (locked) to boolean vector
        __m256i mlocked = expand_bitvector(~m);
        // apply mask (remove bit), preserving the locked cells     
        c = and_unless(c, mask, mlocked);
        _mm256_storeu_si256((__m256i*) &candidates[j], c);
    }
    if ((to_update[1] & (1ULL << (80-64))) != 0) {
        candidates[80] &= ~digit;
    }
}

// Each algorithm (naked single, hidden single, naked set)
// has its own non-solvability detecting trap door to detect the grid is bad.
// track_back acts upon that detection and discards the current grid_state.
//
template <bool verbose>
inline GridState * track_back(int line) {
    // Go back to the state when the last guess was made
    // This state had the guess removed as candidate from it's cell
    
    if (stackpointer) {
        trackbacks++;
        if ( verbose ) {
            digits_entered_and_retracted += 
                (_popcnt64((this-1)->unlocked[0] & ~unlocked[0]))
              + (_popcnt32((this-1)->unlocked[1] & ~unlocked[1]));
        }
        return this-1;
    } else {
        // This only happens when the puzzle is not valid
        fprintf(stderr, "Line %d: No solution found!\n", line);
        exit(0);
    }
    
}

GridState* make_guess() {
    // Make a guess for the cell with the least candidates. The guess will be the lowest
    // possible digit for that cell. If multiple cells have the same number of candidates, the 
    // cell with lowest index will be chosen. Also save the current grid state for tracking back
    // in case the guess is wrong. No cell has less than two candidates.

    // Find the cell with fewest possible candidates
    unsigned long long to_visit;
    unsigned char guess_index = 0;
    unsigned char i_rel;
    unsigned char cnt;
    unsigned char best_cnt = 16;
    
    to_visit = unlocked[0];
    while (to_visit != 0 && best_cnt > 2) {
		i_rel = _tzcnt_u64(to_visit);
        to_visit &= ~(1ULL << i_rel);
        cnt = __popcnt16(candidates[i_rel]);
        if (cnt < best_cnt) {
            best_cnt = cnt;
            guess_index = i_rel;
        }
    }

    to_visit = unlocked[1];
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
        fprintf(stderr, "Error: no GridState struct availabe\n");
        exit(0);
    }
    memcpy(new_grid_state, this, sizeof(GridState));
    new_grid_state->stackpointer++;
    
    // Remove the guessed candidate from the old grid
    // when we get back here to the old grid, we know the guess was wrong
    candidates[guess_index] &= ~digit;
    if (guess_index < 64) {
        updated[0] |= 1ULL << guess_index;
    } else {
        updated[1] |= 1ULL << (guess_index-64);
    }
    
    // Update candidates
    new_grid_state->enter_digit( digit, guess_index);
    guesses++;
    
    return new_grid_state;
}

template<Kind kind>
inline unsigned char get_ul_set_search( bool &fitm1, unsigned char cnt, unsigned char i) {
    unsigned char ret = get_unlinked_cnt<kind>(unlocked, i);
    fitm1 = cnt < ret;
    return ret;
}

};

template <bool verbose>
static bool solve(signed char grid[81], GridState *grid_state, int line) {

    unsigned long long* unlocked = grid_state->unlocked;
    unsigned short* candidates = grid_state->candidates;

    grid_state->init();

//    int line = (grid +82 - output) /164;

    int no_guess_incr = 1;

    unsigned long long my_naked_sets_searched = 0;

    unsigned int my_past_naked_count = 0;
    
    {
        signed short digit;
        unsigned short columns[9] = {0};
        unsigned short rows[9] = {0};
        unsigned short boxes[9] = {0};
        
        for (unsigned char i = 0; i < 64; ++i) {
            digit = grid[i] - 49;
            if (digit >= 0) {
                digit = 1 << digit;
                columns[column_index[i]] |= digit;
                rows[row_index[i]]       |= digit;
                boxes[box_index[i]]      |= digit;
                unlocked[0] &= ~(1ULL << i);
            }
        }
        for (unsigned char i = 64; i < 81; ++i) {
            digit = grid[i] - 49;
            if (digit >= 0) {
                digit = 1 << digit;
                columns[column_index[i]] |= digit;
                rows[row_index[i]]       |= digit;
                boxes[box_index[i]]      |= digit;
                unlocked[1] &= ~(1ULL << (i-64));
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

    // a little trick to resolve hidden singles in all columns,
    // while covering only 8 columns in each pass.
    unsigned char test_column9_flip = 1;
    
    start:
     test_column9_flip ^= 1;

    unlocked = grid_state->unlocked;
    candidates = grid_state->candidates;
            
    // Find naked singles
    {    
        bool found;
        const __m256i ones = _mm256_set1_epi16(1);
        do {
            found = false;
            for (unsigned char i = 0; i < 80; i += 16) {
                __m256i c = _mm256_loadu_si256((__m256i*) &candidates[i]);
                // Check if any cell has zero candidates
                if (_mm256_movemask_epi8(_mm256_cmpeq_epi16(c, _mm256_setzero_si256()))) {
                    // Back track, no solutions along this path
                    grid_state = grid_state->track_back<verbose>( line);
                    goto start;
                } else {
                    unsigned short m = (i < 64) ? (unlocked[0] >> i) : unlocked[1];
                    if ( m ) {
					    // remove least significant digit and compare to 0:
					    // (c = c & (c-1)) == 0  => naked single
                        __m256i a = _mm256_cmpeq_epi16(_mm256_and_si256(c, _mm256_sub_epi16(c, ones)), _mm256_setzero_si256());
                        unsigned int mask = and_compress_masks<true>(a,m);
                        if (mask) {     // note that the mask has two bits set for each detected single
                            // enter_digits(grid_state, mask & 0x55555555, i);
                            int index = (_tzcnt_u32(mask)>>1) + i;
                            grid_state->enter_digit( candidates[index], index);
                            found = true;
                        }
					}
                }
            }
            if (unlocked[1] & (1ULL << (80-64))) {
                if (candidates[80] == 0) {
                    // no solutions go back
                    grid_state = grid_state->track_back<verbose>( line);
                    goto start;
                } else if (__popcnt16(candidates[80]) == 1) {
                    // Enter the digit and update candidates
                    grid_state->enter_digit( candidates[80], 80);
                    found = true;
                }
            }
        } while (found);
    }
    
    // Check if it's solved, if it ever gets solved it will be solved after looking for naked singles
    if ((unlocked[0] | unlocked[1]) == 0) {
        // Solved it
        // Enter found digits into grid
        for (unsigned char j = 0; j < 81; ++j) {
            grid[j] = 49+__bsrd(candidates[j]);
        }

        solved_count++;

        if ( verbose ) {
            no_guess_cnt += no_guess_incr;
            past_naked_count += my_past_naked_count;
            naked_sets_searched += my_naked_sets_searched;
        }
        return true;
    }

    my_past_naked_count++;

    // Find hidden singles
    // Don't check the first or last column because it doesn't fit in the SSE register so it's not really worth checking
    {
        const __m128i ones = _mm_set1_epi16(1);
        const __m128i shuffle_mask = _mm_setr_epi8(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1);

        __m128i column_or_tails[8];
        __m128i column_or_head = _mm_setzero_si128();

        for (unsigned char i = test_column9_flip; i < 81; i += 9) {
            
            // rows
            unsigned short the9thcand = candidates[i+(test_column9_flip?-1:8)];
            __m128i c = _mm_loadu_si128((__m128i*) &candidates[i]);
            __m128i row_or7 = _mm_setzero_si128();
            {
                __m128i c_ = c;
                for (unsigned char j = 0; j < 7; ++j) {
                // rotate shift (1 2 3 4) -> (4 1 2 3)
                    c_ = _mm_shuffle_epi8(c_, shuffle_mask);
                    row_or7 = _mm_or_si128(c_, row_or7);
            }
                row_or7  = _mm_or_si128(_mm_set1_epi16(the9thcand), row_or7);
            }
            __m128i row_mask = _mm_andnot_si128(row_or7, _mm_set1_epi16(0x01ff));
            
            // columns
            __m128i column_mask;    // to start, we simply tally the or'ed rows
            if ( i == test_column9_flip ) {
                // keep precomputed 'tails' of the or'ed column_mask around
                column_mask = _mm_setzero_si128();
                for (signed char j = 81-9+test_column9_flip; j > 1; j -= 9) {
                    column_or_tails[j/9-1] = column_mask;
                    column_mask = _mm_or_si128(*(__m128i*) &candidates[j], column_mask);
                }
            } else {
                // leverage previously computed or'ed rows in head and tails.
                column_or_head = column_mask = _mm_or_si128(*(__m128i*) &candidates[i-9], column_or_head);
                column_mask = _mm_or_si128(column_or_tails[i/9-1],column_mask);
            }
            // turn the or'ed rows into a mask.
            column_mask = _mm_andnot_si128(column_mask,_mm_set1_epi16(0x01ff));

            __m128i or_mask = _mm_or_si128(row_mask, column_mask);
            
            if (_mm_test_all_zeros(or_mask, _mm_sub_epi16(or_mask, ones))) {

                unsigned char m = (i < 64) ? (unlocked[0] >> i) : (unlocked[1] >> (i-64));
                if ( i > 64-8) {
                    m |= unlocked[1] << (64-i);
                }

                // check for identified singles present in the row:
                __m128i a = _mm_cmpgt_epi16(_mm_and_si128(c, or_mask), _mm_setzero_si128());
                unsigned short mask = and_compress_masks128<true>(a,m);

                if (mask) {
                    int index = __tzcnt_u16(mask)>>1;
                    grid_state->enter_digit( ((__v8hu)or_mask)[index], index+i);
                    goto start;
                }
            } else {
                // the row and column masks can each detect multiple singles,
                // possibly even the same single or singles.
                // this is guaranteed by the algorithm used.
                // However, if the row and column check claim _different_ candidates
                // to be the hidden single for the given cell, then the grid
                // is in bad shape.
                // the current grid has no solution, go back
                grid_state = grid_state->track_back<verbose>( line);
                goto start;
            }
        }
    }
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
// track scenarios if the dicovered set is impossibly large.  This is quite important
// for performance as an early killer of bad guesses.
//
// The other consideration is that triads play a major role in this space, as they allow
// in a largely orthogonal manner to reduce the number of candidates, as well as detect many
// pairs and triplets.
//
// With MAX_SET of 4, the efficiency goes down to 30879 from 32508, which is substantial.
// Consider a possible addition of a search for any duplicate solutions, and you can appreciate
// what this means.  MAX_SET 3 sits at 27722 and MAX_SET 2 at 16776.
// From my explorations I know that a 'perfect' search for singles and triad resolution will
// give you 38671 solutions out of 49151 without guessing.  So there's a lot of room for improvement
// yet improvements in terms of raw speed may be pretty hard to get.
//
// Note that this search has it's own built-in heuristic to tackle only recently updated cells.
// The algorithm will keep that list to revisit later, which is fine of course.
// The number of cells to visit is expectedly pretty high.  We could try other tricks around
// this area of 'work avoidance'.
//
    {
        bool can_backtrack = (grid_state->stackpointer != 0);

        bool found = false;
        // visit only the changed (updated) cells
        unsigned long long *to_visit_n = grid_state->updated;
        to_visit_n[0] &= unlocked[0];
        to_visit_n[1] &= unlocked[1];
        
        for (unsigned char n = 0; n < 2; ++n) {
            while (to_visit_n[n]) {
                int i_rel = _tzcnt_u64(to_visit_n[n]);
                
                to_visit_n[n] ^= 1ULL << i_rel;
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
                    bool fitm1;
                    unsigned char s;

                    // check row
                    // Same comments as above apply
                    //
                    grid_state->get_ul_set_search<Row>(fitm1, cnt, i);
                    if (can_backtrack || fitm1) {
                        a_j = _mm_loadu_si128((__m128i*) &candidates[9*row_index[i]]);
                        res = _mm_cmpeq_epi16(a_i, _mm_or_si128(a_i, a_j));
                        s = __popcnt16(_mm_movemask_epi8(res)) >> 1;
                        s += candidates[i] == (candidates[i] | candidates[9*row_index[i]+8]);
                        my_naked_sets_searched++;
                        if (s > cnt) {
                            grid_state = grid_state->track_back<verbose>( line);
                            goto start;
                        } else if (s == cnt && fitm1) {
                            naked_sets_found++;
                            add_indices<Row>(to_change, i);
                        }
                    }
    
                    // check column
                    //
                    grid_state->get_ul_set_search<Col>(fitm1, cnt, i);
                    if (can_backtrack || fitm1) {
                        a_j = _mm_set_epi16(candidates[column_index[i]+63], candidates[column_index[i]+54], candidates[column_index[i]+45], candidates[column_index[i]+36], candidates[column_index[i]+27], candidates[column_index[i]+18], candidates[column_index[i]+9], candidates[column_index[i]]);
                        res = _mm_cmpeq_epi16(a_i, _mm_or_si128(a_i, a_j));
                        s = __popcnt16(_mm_movemask_epi8(res)) >> 1;
                        s += candidates[i] == (candidates[i] | candidates[column_index[i]+72]);
                        my_naked_sets_searched++;
                        // this covers the situation where there is a naked set of size x
                        // which is found in y cells with y > x.  That's impossible, hence track back.
                        if (s > cnt) {
                            grid_state = grid_state->track_back<verbose>( line);
                            goto start;
                        } else if (s == cnt && fitm1 ) {
                            // This also doesn't refine the search for smaller subsets.
                            // e.g. {a,b,c,d,e} + {a,b} + {a,b} + {c,d} + {d,e}
                            //
                            // If there are N cells, cnt == N is not good, and cnt == N - 1
                            // will at best unveil a hidden single.
                            naked_sets_found++;
                            add_indices<Col>(to_change, i);

                        }
                    }

                    // check box
                    // If there are n cells, cnt == n will just cause churn.
                    // but cnt == n - 1 will detect hidden singles, which is excellent.
                    grid_state->get_ul_set_search<Box>(fitm1, cnt, i);
                    if (can_backtrack || fitm1) {
                        unsigned short b = box_start[i];
                        a_j = _mm_set_epi16(candidates[b], candidates[b+1], candidates[b+2], candidates[b+9], candidates[b+10], candidates[b+11], candidates[b+18], candidates[b+19]);
                        res = _mm_cmpeq_epi16(a_i, _mm_or_si128(a_i, a_j));
                        s = __popcnt16(_mm_movemask_epi8(res)) >> 1;
                        s += candidates[i] == (candidates[i] | candidates[b+20]);
                        my_naked_sets_searched++;
                        if (s > cnt) {
                            grid_state = grid_state->track_back<verbose>( line);
                            goto start;
                        } else if (s == cnt && fitm1) {
                            naked_sets_found++;
                            add_indices<Box>(to_change, i);
                        }
                    }
                                    
                    to_change[0] &= unlocked[0];
                    to_change[1] &= unlocked[1];
                    
                    // update candidates
                    for (unsigned char n = 0; n < 2; ++n) {
                        while (to_change[n]) {
                            int j_rel = _tzcnt_u64(to_change[n]);
                            to_change[n] &= ~(1ULL << j_rel);
                            unsigned char j = (unsigned char) j_rel + 64*n;
                            
                            // if this cell is not part of our set
                            if ((candidates[j] | candidates[i]) != candidates[i]) {
                                // if there are bits that need removing
                                if (candidates[j] & candidates[i]) {
                                    candidates[j] &= ~candidates[i];
                                    to_visit_n[n] |= 1ULL << j_rel;
                                    found = true;
                                }
                            }
                        }
                    }
                    
                    // If any cell's candidates got updated, go back and try all that other stuff again
                    if (found) {
                        goto start;
                    }
                    
                } else if ( cnt == 1 ) {
                    // this is not possible, but just to eliminate cnt == 1:
                    grid_state->enter_digit( candidates[i], i);
                    printf("found a singleton in set search... strange\n");
                    goto start;
                }
            }
        }
    }
    
    // More techniques could be added here but they're not really worth checking for on the 17 clue sudoku set
    
    // Make a guess if all that didn't work
    grid_state = grid_state->make_guess();
    no_guess_incr = 0;
    goto start;
    
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
        case 'x':    // stats output
             reportstats=true;
             break;
        case 'l':    // line of puzzle to solve
             sscanf(argv[0]+2, "%d", &line_to_solve);
             break;
        }
        argc--, argv++;
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
	char *string = (char *)mmap((void*)0, fsize, PROT_READ, MAP_PRIVATE, fdin, 0);
	if ( string == MAP_FAILED ) {
		if (errno ) {
			printf("Error mmap of input file %s: %s\n", ifn, strerror(errno));
			exit(0);
		}
	}
	close(fdin);

	// skip first line, unless it's a puzzle.
    size_t pre = 0;
    if ( !isdigit((int)string[0]) || string[81] != 10 ) {
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

	char *string_pre = string+pre;
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
            stack = (GridState*) (~0x3fll & ((unsigned long long) malloc(sizeof(GridState)*GRIDSTATE_MAX+0x40)+0x40));
        }
        if ( reportstats == false ) {
            solve<false>(&output[82], stack, line_to_solve);
        } else {
            solve<true>(&output[82], stack, line_to_solve);
        }
    } else {

#pragma omp parallel for firstprivate(stack) shared(string_pre, npuzzles, output, i, imax) schedule(dynamic,32)
        for (i = 0; i < imax; i+=82) {
            // copy unsolved grid
            memcpy(&output[i*2], &string_pre[i], 81);
            memcpy(&output[i*2+82], &string_pre[i], 81);
            // add comma and newline in right place
            output[i*2 + 81] = ',';
            output[i*2 + 163] = 10;
    		if ( stack == 0 ) {
                // force alignment the 'old-fashioned' way
                stack = (GridState*) (~0x3fll & ((unsigned long long) malloc(sizeof(GridState)*GRIDSTATE_MAX+0x40)+0x40));
            }
            // solve the grid in place
            if ( reportstats == false ) {
                solve<false>(&output[i*2+82], stack, i/82+1);
            } else {
                solve<true>(&output[i*2+82], stack, i/82+1);
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
        printf("schoku version: %s\n", version_string);
        printf("%10ld  puzzles entered\n", npuzzles);
		printf("%8.1lfms  %5.2lf\u00b5s/puzzle  solving time\n", (double)duration/1000000, (double)duration/(npuzzles*1000LL));
        printf("%10ld  %5.2f%%  puzzles solved without guessing\n", no_guess_cnt.load(), (double)no_guess_cnt.load()/(double)solved_count.load()*100);
        printf("%10lld  %5.2f/puzzle  total guesses\n", guesses.load(), (double)guesses.load()/(double)solved_count.load());
        printf("%10lld  %5.2f/puzzle  total back tracks\n", trackbacks.load(), (double)trackbacks.load()/(double)solved_count.load());
        printf("%10lld  %5.2f/puzzle  total digits entered and retracted\n", digits_entered_and_retracted.load(), (double)digits_entered_and_retracted.load()/(double)solved_count.load());
        printf("%10lld  %5.2f/puzzle  total 'rounds'\n", past_naked_count.load(), (double)past_naked_count.load()/(double)solved_count.load());
        printf("%10lld  %5.2f/puzzle  naked sets found\n", naked_sets_found.load(), naked_sets_found.load()/(double)solved_count.load());
        printf("%10lld %6.2f/puzzle  naked sets searched\n", naked_sets_searched.load(), naked_sets_searched.load()/(double)solved_count.load());
    }

    return 0;
}
