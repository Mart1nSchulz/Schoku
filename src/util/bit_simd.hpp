// Bit-extraction and SIMD compress/expand helpers used throughout the
// solver. All inline (some always_inline) so call sites collapse on
// release builds.
//
// CONTRACT: must be #included from inside `namespace Schoku { ... }` and
// after these names are in scope:
//   - SIMD type aliases (__m256i, __m128i — via compat/x86_intrin.hpp)
//   - bit128_t (struct) and its u64 array
//   - the BMI/BMI2 wrappers _tzcnt_u32/64, _blsr_u32/64
//   - constants used by SIMD helpers: bit_mask_expand, mask_1ff, mask1ff,
//     hrs_shifts, select_bits, shuffle_mask_bytes, shuffle_interleaved_mask_bytes
//   - global pext_support (set at runtime on x86; always false on aarch64
//     so the no-PDEP/PEXT fallback path is taken)
// The header intentionally does not open the namespace itself — the
// include site already lives in `namespace Schoku`.
#pragma once

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

inline __m128i uncompress(__m128i in) {
        __m128i out =  _mm_and_si128(_mm256_castsi256_si128(mask_1ff),_mm_unpacklo_epi16(in, _mm_bsrli_si128(in, 1)));
        return _mm_and_si128(_mm256_castsi256_si128(mask1ff), _mm_mulhrs_epi16(out, _mm256_castsi256_si128(hrs_shifts)));
}

// uncompress first eight cells from contiguous bits to one unsigned short per cell
// [ beats manual loading hands down ]
// for two inputs in hi and lo lane
inline __m256i uncompress(__m256i in) {
        __m256i out =  _mm256_and_si256(mask_1ff,_mm256_unpacklo_epi16(in, _mm256_bsrli_epi128(in, 1)));
        return _mm256_and_si256(mask1ff, _mm256_mulhrs_epi16(out, hrs_shifts));
}
