// Body of `Status solve(...)` — now a thin dispatcher. The actual
// per-phase work lives in solver/phases/<name>.hpp, each defining one
// out-of-class method of `SolveCtx<verbose>` (see solver/solve_ctx.hpp).
//
// CONTRACT: this is a private include fragment, not a self-contained
// header. It must be #included exactly once, from inside
// `namespace Schoku { ... }` in schoku.cpp, AFTER:
//   - the GridState / SolverData class definitions,
//   - all GridState / SolverData member-function-template bodies
//     (in particular `solver/make_guess.hpp`),
//   - every namespace-scope helper, type, global, and macro that
//     solve() references (e.g. enter_digit's helpers, the bit128_t
//     index lookups, OPT_* feature flags).
// The single-TU layout (everything inlined into schoku.cpp) is
// preserved.
//
// Each phase header is the AlphaEvolve mutation unit for one strategy:
// replace solver/phases/<name>.hpp to mutate that strategy without
// touching the rest of the solver. Phase_HiddenSearch is currently
// monolithic (3686 lines fused-pipeline of hidden-singles + triads +
// naked-sets + OPT_FSH + OPT_UQR); future work splits it into smaller
// swap-units (fish_rows / fish_cols / unique_rectangles / naked_sets_*
// while Block A — Algorithms 2+3 — stays as one fused unit).
#pragma once

#include "solver/solve_ctx.hpp"
#include "solver/phases/back.hpp"
#include "solver/phases/start.hpp"
#include "solver/phases/search.hpp"
#include "solver/phases/enter.hpp"
#include "solver/phases/hidden_search.hpp"
#include "solver/phases/guess.hpp"

template <Verbosity verbose>
Status solve(signed char grid[81], GridState stack[], int line, Counters &counters, FILE *out = stdout) {

    GridState *grid_state = &stack[0];
    unsigned long long *unlocked = grid_state->unlocked.u64;
    unsigned short* candidates;

    Status status;

    SolverData solverData(counters, out);
    unsigned short current_entered_count = 81 - grid_state->unlocked.popcount();
#ifdef OPT_UQR
    // make_guess accepts a lambda, which will use guess_message to pass along
    // two strings of debug information:
    char guess_message[2][196];

    // original_locked keeps track of the presets
    // since unique rectangles can only be invalidated by presets.
    bit128_t original_locked;
    original_locked.u64[0] = ~unlocked[0];
    original_locked.u64[1] = ~unlocked[1] & 0x1ffff;

    bit128_t original_locked_transposed;

    unsigned short superimposed_preset_rows[3][3];    // deal with initialization later...
    unsigned short superimposed_preset_cols[3][3];
    bool have_superimposed_preset_rows = false;
    bool have_superimposed_preset_cols = false;

    unsigned short last_entered_count_uqr = 0;
    unsigned char last_band_uqr = 0;
#endif

    unsigned short last_entered_count_col_triads = 0;

    int unique_check_mode = 0;
    bool nonunique_reported = false;

    unsigned char no_guess_incr = 1;

    if ( verbose == VDebug ) {
        candidates = grid_state->candidates;
        char gridout[82];
        for (unsigned char j = 0; j < 81; ++j) {
            if ( grid_state->unlocked.check_indexbit(j) ) {
                gridout[j] = '0';
            } else {
                gridout[j] = 49+_tzcnt_u32(candidates[j]);
            }
        }
        solverData.printf("Line %d: %.81s\n", line, gridout);
    }

    // The 'API' for code that transitions to Phase_Enter to enter a digit:
    unsigned short e_digit = 0;
    unsigned char e_i = 0;

#ifdef OPT_FSH
    unsigned short exclude_row[9];
    unsigned short exclude_col[9];
#endif

    bool check_back = thorough_check;

#ifdef OPT_SETS
    unsigned short flip = 1;  // for naked sets search to support a search with 50% reduction of tests
#endif

    // Aggregate solve()'s locals into a reference-only context. SolveCtx
    // never escapes; clang SROAs ref-only structs through `[[gnu::always_inline]]`
    // member dispatch, so codegen is equivalent to keeping the locals
    // directly accessible to the body.
    SolveCtx<verbose> ctx{
        grid_state,
        unlocked,
        candidates,
        status,
        solverData,
        current_entered_count,
#ifdef OPT_UQR
        guess_message,
        original_locked,
        original_locked_transposed,
        superimposed_preset_rows,
        superimposed_preset_cols,
        have_superimposed_preset_rows,
        have_superimposed_preset_cols,
        last_entered_count_uqr,
        last_band_uqr,
#endif
        last_entered_count_col_triads,
        unique_check_mode,
        nonunique_reported,
        no_guess_incr,
        e_digit,
        e_i,
#ifdef OPT_FSH
        exclude_row,
        exclude_col,
#endif
        check_back,
#ifdef OPT_SETS
        flip,
#endif
        grid,
        line,
        counters,
    };

    SolverPhase phase = Phase_Start;
    for (;;) {
        SolverPhase next;
        switch (phase) {
        case Phase_Back:              next = ctx.phase_back();              break;
        case Phase_Start:             next = ctx.phase_start();             break;
        case Phase_Search:            next = ctx.phase_search();            break;
        case Phase_Enter:             next = ctx.phase_enter();             break;
        case Phase_HiddenSearch:      next = ctx.phase_hidden_search();     break;
        case Phase_Guess:             next = ctx.phase_guess();             break;
#ifdef OPT_UQR
        case Phase_GuessMadeWithIncr: next = ctx.phase_guess_made_with_incr(); break;
#endif
        case Phase_Done:              return status;
        }
        phase = next;
    }
}
