// Debug helpers that dump SIMD vectors and bit grids to dbgprintout.
// Pure debug aids — none of these are on a hot solver path; they only
// fire when the corresponding bit in dbgprintfilter is set.
//
// CONTRACT: must be #included from inside `namespace Schoku { ... }` and
// after the SIMD type aliases (__v4du, __v16hu, __v8hu, __v2du), the
// v16us alias, the bit128_t struct, and the global dbgprintf() are in
// scope. The header intentionally does not open the namespace itself,
// because schoku.cpp already lives inside it.
#pragma once

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

// Print a 9x9 grid where the 9 bits of each row are packed into one
// unsigned short element of a __m256i (lanes 0..8 used).
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

// Print a 9x9 grid (or two halves split at `split`) where bits are
// packed consecutively as 9*9 bits in a __uint128_t.
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
