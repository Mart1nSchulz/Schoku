// Board-aware printers (formats candidate sets, dumps a 9x9 board, dumps a
// solved puzzle as 81 chars). Like the raw SIMD/bit dumps in
// util/debug_dump.hpp these only fire when the corresponding
// dbgprintfilter bit is set, so they are not on the hot solver path.
//
// CONTRACT: must be #included from inside `namespace Schoku { ... }` and
// after these names are in scope:
//   - bit128_t struct (and its check_indexbit method)
//   - Schoku::dbgprintf, Schoku::dbgprintfilter
//   - the BMI wrappers _tzcnt_u32, _blsr_u32
//   - transposed_cell[] lookup (used by dump_puzzle<transpose=true>)
// The header intentionally does not open the namespace itself — the
// include site already lives in `namespace Schoku`.
//
// NOTE: do NOT include this header outside `namespace Schoku`; `#pragma once`
// would suppress a corrective re-include and unqualified names would bind
// to the wrong scope. Single include site is intentional.
#pragma once

inline void format_candidate_set(char *ret, unsigned short candidates);

// ASCII digit for a solved cell: the single set candidate bit's position
// (0..8) plus '1'. ('1' == 49.) Caller guarantees exactly one bit is set.
inline char digit_char(unsigned short candidate) {
    return '1' + _tzcnt_u32(candidate);
}

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
                gridout[j] = digit_char(candidates[t]);
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
