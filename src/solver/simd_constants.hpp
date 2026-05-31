// Solver SIMD/vector constants.
//
// Centralized into one file so all SIMD/vector constants live in one
// place: every named SIMD constant the solver depends on is declared
// here, with a comment explaining where it's used.
//
// CONTRACT: this is a private include fragment, not a self-contained
// header. It must be #included exactly once, from inside
// `namespace Schoku { ... }` in schoku.cpp, after the SIMD intrinsic
// dispatch headers (compat/x86_intrin.hpp).
//
// Naming: bare `const __m256i name = ...;` at namespace scope is
// internally-linked here (each TU including this gets its own copy);
// the codebase uses a single TU (everything inlined into schoku.cpp)
// so there's no ODR concern. Apple clang folds the constant table
// into __TEXT,__const at -O3.
#pragma once

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
const __m256i mask27 = _mm256_setr_epi64x(-1LL, (long long int)0xffffffffffff00ffLL, (long long int)0xffffffff00ffffffLL, 0xffffffffffLL);
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

// used in NEWSETS:
const __m256i true_epi16 = _mm256_set1_epi16(0xffff);

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
