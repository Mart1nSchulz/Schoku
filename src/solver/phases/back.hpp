// Out-of-class definition of `SolveCtx<verbose>::phase_back()`.
// AlphaEvolve mutation unit: this file is the entire surface for back-track
// strategy; replace the function body to mutate the strategy without
// touching the rest of the solver.
//
// CONTRACT: private include fragment, must be #included exactly once
// from inside `namespace Schoku { ... }` after solver/solve_ctx.hpp.
#pragma once


template <Verbosity verbose>
__attribute__((always_inline)) inline SolverPhase SolveCtx<verbose>::phase_back() {

    // Each algorithm (naked single, hidden single, naked set)
    // has its own non-solvability detecting trap door to detect if the grid is bad.
    // This section acts upon that detection and discards the current grid_state.
    //
    if (grid_state->stackpointer == 0) {
        if ( unique_check_mode ) {
            if ( verbose == VDebug ) {
                // no additional solution exists
                solverData.printf("No secondary solution found during back track\n");
            }
        } else {
            // This only happens when the puzzle is not valid
            // Bypass the verbose check...
            if ( warnings != 0 ) {
                solverData.printf("Line %d: No %ssolution found!\n", line, rules==Regular?"unique " : "");
            }
            counters.unsolved_count++;
        }
        // cleanup and return
        if ( verbose != VNone && reportstats ) {
            if ( status.unique == false ) {
                counters.non_unique_count++;
            }
        }
        if ( !unique_check_mode ) {
            // failed - just copy the input
            memcpy(grid, grid-82, 81);
        }
        return Phase_Done;
    }

    current_entered_count  = (((grid_state-1)->stackpointer)<<8) | (81 - (grid_state-1)->unlocked.popcount());        // back to previous stack.

    // collect some guessing stats
    if ( verbose != VNone && reportstats ) {
        counters.digits_entered_and_retracted +=
            (_popcnt64((grid_state-1)->unlocked.u64[0] & ~grid_state->unlocked.u64[0]))
          + (_popcnt32((grid_state-1)->unlocked.u64[1] & ~grid_state->unlocked.u64[1]));
    }

    // Go back to the state when the last guess was made
    // This state had the guess removed as candidate from it's cell

    if ( verbose == VDebug ) {
        solverData.printf("back track to level >%d<\n", grid_state->stackpointer-1);
    }
    if ( trace::current ) {
        int from = grid_state->stackpointer;
        trace::backtrack(from, from - 1);
    }
    if ( verbose != VNone && reportstats ) {
        counters.trackbacks++;
    }
    grid_state--;
        return Phase_Start;
}
