// Out-of-class definition of `SolveCtx<verbose>::phase_start()`.
// This file contains the complete implementation of per-iteration grid_state setup
// strategy.
//
// CONTRACT: private include fragment, must be #included exactly once
// from inside `namespace Schoku { ... }` after solver/solve_ctx.hpp.
#pragma once


template <Verbosity verbose>
__attribute__((always_inline)) inline SolverPhase SolveCtx<verbose>::phase_start() {

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
        return Phase_Search;
}
