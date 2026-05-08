// Solver state context + dispatcher state machine declarations.
//
// CONTRACT: this is a private include fragment, not a self-contained
// header. It must be #included exactly once, from inside
// `namespace Schoku { ... }` in schoku.cpp, AFTER the GridState /
// SolverData class definitions and before solver/phases/*.hpp.
//
// Each phase method (phase_back / phase_start / ... / phase_done) is
// defined out-of-class in its own header under solver/phases/. The
// dispatcher in solver/solve.hpp calls ctx.phase_X() and uses the
// returned SolverPhase to drive the next iteration. Returning a
// SolverPhase from any nesting depth correctly exits the lambda /
// method (the original `goto X` semantics) — fixing the bug in the
// reverted commit 1d43328 where `phase=X; continue;` continued the
// inner for/while instead of the outer for(;;) switch.
//
// Each phase is the AlphaEvolve mutation unit: replace a single
// solver/phases/<name>.hpp file to mutate one strategy's
// implementation, leaving the rest of the solver intact.
#pragma once

enum SolverPhase {
    Phase_Back,
    Phase_Start,
    Phase_Search,
    Phase_Enter,
    Phase_HiddenSearch,
    Phase_Guess,
#ifdef OPT_UQR
    Phase_GuessMadeWithIncr,
#endif
    Phase_Done,
};

// SolveCtx: a thin reference-only wrapper over solve()'s locals so each
// phase function can access them as data members via implicit `this->`.
// All fields are references / pointers to objects living in solve()'s
// stack frame; SolveCtx itself is created once at the top of solve()
// and never escapes. Apple clang 17 SROAs ref-only structs through
// `[[gnu::always_inline]]` member dispatch, so codegen is equivalent to
// keeping locals directly accessible (verified empirically).
template <Verbosity verbose>
struct SolveCtx {
    GridState*&            grid_state;
    unsigned long long*&   unlocked;
    unsigned short*&       candidates;
    Status&                status;
    SolverData&            solverData;
    unsigned short&        current_entered_count;
#ifdef OPT_UQR
    char                 (&guess_message)[2][196];
    bit128_t&              original_locked;
    bit128_t&              original_locked_transposed;
    unsigned short       (&superimposed_preset_rows)[3][3];
    unsigned short       (&superimposed_preset_cols)[3][3];
    bool&                  have_superimposed_preset_rows;
    bool&                  have_superimposed_preset_cols;
    unsigned short&        last_entered_count_uqr;
    unsigned char&         last_band_uqr;
#endif
    unsigned short&        last_entered_count_col_triads;
    int&                   unique_check_mode;
    bool&                  nonunique_reported;
    unsigned char&         no_guess_incr;
    unsigned short&        e_digit;
    unsigned char&         e_i;
#ifdef OPT_FSH
    unsigned short       (&exclude_row)[9];
    unsigned short       (&exclude_col)[9];
#endif
    bool&                  check_back;
#ifdef OPT_SETS
    unsigned short&        flip;
#endif
    // function args (by ref / by value as appropriate)
    signed char*           grid;
    int                    line;
    Counters&              counters;

    // Phase methods. Each returns the next SolverPhase to dispatch.
    // Definitions live in solver/phases/<name>.hpp.
    SolverPhase phase_back();
    SolverPhase phase_start();
    SolverPhase phase_search();
    SolverPhase phase_enter();
    SolverPhase phase_hidden_search();
    SolverPhase phase_guess();
#ifdef OPT_UQR
    SolverPhase phase_guess_made_with_incr();
#endif

    // Sub-phase helpers called from phase_hidden_search() in sequence.
    // Each returns Phase_HiddenSearch on natural completion (continue to
    // the next helper) or any other SolverPhase to short-circuit back to
    // the dispatcher (semantically equivalent to the pre-refactor `goto X`
    // exits inside the original block). Bodies live in their own headers.
    SolverPhase do_naked_sets_new();
    SolverPhase do_naked_sets_main();
    SolverPhase do_fishes();
    SolverPhase do_unique_rectangles();
};
