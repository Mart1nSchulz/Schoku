// Out-of-class definition of `SolveCtx<verbose>::phase_guess()`.
// AlphaEvolve mutation unit: this file is the entire surface for make_guess + (OPT_UQR) post-guess update
// strategy; replace the function body to mutate the strategy without
// touching the rest of the solver.
//
// CONTRACT: private include fragment, must be #included exactly once
// from inside `namespace Schoku { ... }` after solver/solve_ctx.hpp.
#pragma once


    template <Verbosity verbose>
__attribute__((always_inline)) inline SolverPhase SolveCtx<verbose>::phase_guess() {
    // Make a guess if all that didn't work
    grid_state = grid_state->make_guess<verbose>(&solverData);
    no_guess_incr = 0;
#ifdef OPT_UQR
// if the guess was made solely to allow for checking uniqueness, still count the solution
// as direct solve
        return Phase_GuessMadeWithIncr;
    }
    template <Verbosity verbose>
__attribute__((always_inline)) inline SolverPhase SolveCtx<verbose>::phase_guess_made_with_incr() {
#endif
    current_entered_count  = (grid_state->stackpointer<<8) | (81 - grid_state->unlocked.popcount());
    return Phase_Start;
    }
