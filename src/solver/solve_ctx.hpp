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
// SolverPhase from any nesting depth exits the method.
//
// Each phase method lives in its own solver/phases/<name>.hpp file,
// holding one strategy's implementation.
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
    // the dispatcher (returning any other SolverPhase short-circuits back
    // to the dispatcher). Bodies live in their own headers.
    SolverPhase do_naked_sets_new();
    // Shared "drain to_change, eliminate candidates, emit trace, bump counter"
    // step for naked-set rows (used twice inside do_naked_sets_new()).
    // Returns true if any candidate was eliminated (caller redirects to
    // Phase_Search), false otherwise. Defined in naked_sets_new.hpp.
    bool eliminate_naked_set_row(unsigned char cl, unsigned short m, unsigned char row, unsigned char cnt, bit128_t &to_change);
    SolverPhase do_naked_sets_main();
    SolverPhase do_fishes();

    // Verdict from the shared fish consequence-emit block (helper 1).
    // The block always ends inside `if(clean_bits)`, so it either drives a
    // sashimi placement (FishEnter -> caller returns Phase_Enter) or marks a
    // search (FishSearch -> caller sets dosearch). The final
    // `if(dosearch) return Phase_Search` stays at the call site.
    enum FishEmit { FishEnter, FishSearch };

    // Shared sub-blocks of do_fishes(), one per identical pair across the
    // row-scan and col-scan halves. Defined out-of-class in fishes.hpp,
    // always_inline so codegen is identical to the inline originals.

    // Helper 1: final consequence-emit (row ~447-493 / col ~803-847).
    // Sets e_digit/e_i for sashimi placements, drains clean_bits removing
    // digit `dgt`, mirrors trace::eliminate under unit `trace_elim_unit`.
    FishEmit emit_fish_consequences(bit128_t &clean_bits, const unsigned char fincells[2],
                                    unsigned char dgt, unsigned char cnt, char trace_elim_unit);

    // Helper 2: Case-1 (sashimi) local eliminate loop (row ~357-376 /
    // col ~712-731). Drains clean_bits removing digit `dgt`, unit `u`.
    void eliminate_fish_case1(bit128_t &clean_bits, unsigned char dgt, char u);

#ifdef OPT_FSH
    // Helper 3: alt-triple synthesis (row ~503-516 / col ~857-870).
    // Scans the bi-value positions in pair_locs for a pair whose union has
    // popcount 3; on success writes that union to alt_base_x and returns true.
    bool synthesize_fish_alt_triple(cbbv_t &cbbv, unsigned int pair_cnt,
                                    unsigned int pair_locs, unsigned int &alt_base_x);
#endif
    SolverPhase do_unique_rectangles();
};
