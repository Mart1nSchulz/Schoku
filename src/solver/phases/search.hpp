// Out-of-class definition of `SolveCtx<verbose>::phase_search()`.
// AlphaEvolve mutation unit: this file is the entire surface for naked-single search
// strategy; replace the function body to mutate the strategy without
// touching the rest of the solver.
//
// CONTRACT: private include fragment, must be #included exactly once
// from inside `namespace Schoku { ... }` after solver/solve_ctx.hpp.
#pragma once


template <Verbosity verbose>
__attribute__((always_inline)) inline SolverPhase SolveCtx<verbose>::phase_search() {
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
                return Phase_Back;
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
                // Mark trace context so phase_enter can label this event.
                // The store is a single TLS write; cheap when trace disabled.
                trace::next_entry_reason = trace::ER_NakedSingle;
                return Phase_Enter;
            }
        }
    }
    // if no single found:
    return Phase_HiddenSearch;

// Algorithm 1:
// Enter a digit into the solution by setting it as the value of cell and by
// removing the cell from the set of unlocked cells.  Update all affected cells
// by removing any candidates that have become impossible for that digit.
// For all cells check whether the cell has no candidates: back track.
// Check all cells for a single candidate.
//
}
