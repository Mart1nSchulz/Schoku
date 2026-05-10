// JSONL trace emitter for machine consumption (e.g. neural-net distillation).
//
// Activated by `--trace-out FILE` on the CLI; output is one JSON object per
// line. When the flag is absent, `trace::current` is nullptr in every thread
// and the `if (trace::current)` guard at each emission site folds to a
// well-predicted cold branch.
//
// Design constraints (see PR discussion):
//   * Per-thread buffer, grown on demand from a 64KB starting size.
//   * One fwrite per puzzle_end — fwrite on a FILE* is internally locked by
//     libc within ONE process, so per-puzzle blocks from different threads
//     sharing the same FILE* may interleave at puzzle boundaries but never
//     mid-line. Cross-process appends to the same path are NOT line-
//     atomic; if the user wants to concatenate multiple Schoku runs they
//     should serialise them or post-process to sort by puzzle_id.
//   * No allocation, no syscalls inside event helpers — only buffer appends.
//   * Helpers are inline-able; the trace::current null check at the call
//     site keeps the disabled path within a single load + branch.
//
// Phase 1 events: puzzle_start, puzzle_end, naked_single, hidden_single,
// guess, backtrack. Later phases will add eliminate (triads, sets), fish,
// BUG, UR. The schema field "schoku_internal" is reserved for Schoku-
// specific refinements where the standardised "type" is lossy.
//
// CONTRACT: like the other solver/*.hpp headers, this file expects to be
// #included from *inside* `namespace Schoku { ... }`. The two-namespace
// form (`namespace Schoku { namespace trace { ... }}`) is what callers
// outside `Schoku` see (e.g. `Schoku::trace::Emitter` in main()).
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace trace {

class Emitter {
public:
    static constexpr size_t INITIAL_CAP = 64 * 1024;

    Emitter(FILE* sink, int thread_id)
        : sink_(sink), buf_(nullptr), cap_(0), len_(0),
          thread_id_(thread_id), guesses_(0), backtracks_(0) {
        grow_to(INITIAL_CAP);
    }
    ~Emitter() { std::free(buf_); }

    // Disallow copy; allow move not needed (we hold a unique buffer).
    Emitter(const Emitter&) = delete;
    Emitter& operator=(const Emitter&) = delete;

    // ---- Event API. row/col are 0-based row-major. value is sudoku
    //      digit 1..9. level is GridState::stackpointer (0 = top).

    void puzzle_start(const char* puzzle81, int puzzle_id) {
        ensure(160);
        append_lit("{\"event\":\"puzzle_start\",\"puzzle_id\":");
        append_int(puzzle_id);
        append_lit(",\"thread\":");
        append_int(thread_id_);
        append_lit(",\"puzzle\":\"");
        append_chunk(puzzle81, 81);
        append_lit("\"}\n");
    }

    // The caller passes (n_guesses, n_backtracks) as a stat-system delta
    // when reportstats is on; otherwise these are 0 and we fall back to
    // the emitter's own counters — which we always increment in
    // guess/guess_triad/backtrack. The two cannot drift: per-puzzle stat
    // counters are accumulated via the same code paths.
    void puzzle_end(bool solved, const char* solution81,
                    long long n_guesses, long long n_backtracks) {
        ensure(160);
        append_lit("{\"event\":\"puzzle_end\",\"solved\":");
        if (solved) { append_lit("true"); } else { append_lit("false"); }
        append_lit(",\"guesses\":");
        append_int(n_guesses != 0 ? n_guesses : guesses_);
        append_lit(",\"backtracks\":");
        append_int(n_backtracks != 0 ? n_backtracks : backtracks_);
        if (solved && solution81 != nullptr) {
            append_lit(",\"solution\":\"");
            append_chunk(solution81, 81);
            append_lit("\"");
        }
        append_lit("}\n");
        flush();
        // Reset per-puzzle counters; sink is shared so the next puzzle
        // starts from zero.
        guesses_ = 0;
        backtracks_ = 0;
    }

    void naked_single(int row, int col, int value, int level) {
        ensure(96);
        append_lit("{\"event\":\"step\",\"type\":\"single\","
                   "\"reason\":\"naked\",\"cell\":[");
        append_int(row);
        append_lit(",");
        append_int(col);
        append_lit("],\"value\":");
        append_int(value);
        append_lit(",\"level\":");
        append_int(level);
        append_lit("}\n");
    }

    // Placement forced by a non-trivial deduction (fish, set, UR, BUG)
    // whose antecedent event is not yet emitted (Phase 2+ work). Distinct
    // from naked_single so a consumer can drop these rows during training.
    void deduced_single(int row, int col, int value, int level) {
        ensure(112);
        append_lit("{\"event\":\"step\",\"type\":\"single\","
                   "\"reason\":\"deduced\",\"cell\":[");
        append_int(row);
        append_lit(",");
        append_int(col);
        append_lit("],\"value\":");
        append_int(value);
        append_lit(",\"level\":");
        append_int(level);
        append_lit("}\n");
    }

    // `unit` is one of 'r' (row) or 'c' (col); 'b' (box) reserved for the
    // box hidden-single path that Schoku currently subsumes into triads.
    void hidden_single(int row, int col, int value, char unit, int level) {
        ensure(112);
        append_lit("{\"event\":\"step\",\"type\":\"single\","
                   "\"reason\":\"hidden\",\"unit\":\"");
        char u[2] = {unit, 0};
        append_str(u);
        append_lit("\",\"cell\":[");
        append_int(row);
        append_lit(",");
        append_int(col);
        append_lit("],\"value\":");
        append_int(value);
        append_lit(",\"level\":");
        append_int(level);
        append_lit("}\n");
    }

    // Cell-based guess: a specific (row,col) is set to `value` on the new
    // branch. `from_level` is the saved state's level; new_level = +1.
    void guess(int row, int col, int value, int from_level) {
        guesses_++;
        ensure(128);
        append_lit("{\"event\":\"step\",\"type\":\"guess\","
                   "\"schoku_internal\":\"cell\",\"cell\":[");
        append_int(row);
        append_lit(",");
        append_int(col);
        append_lit("],\"value\":");
        append_int(value);
        append_lit(",\"from_level\":");
        append_int(from_level);
        append_lit(",\"new_level\":");
        append_int(from_level + 1);
        append_lit("}\n");
    }

    // Triad-based guess: `value` is REMOVED from the three cells (anchor,
    // anchor+inc, anchor+2*inc) on the new branch, and kept-only on the
    // saved branch. unit is 'r' (row triad, inc=1) or 'c' (col triad,
    // inc=9). Spec field shape differs from the cell variant by `cells`
    // (an array of 3) instead of `cell`.
    void guess_triad(int anchor_row, int anchor_col, int value,
                     char unit, int from_level) {
        guesses_++;
        ensure(192);
        append_lit("{\"event\":\"step\",\"type\":\"guess\","
                   "\"schoku_internal\":\"triad_");
        char u[2] = {unit, 0};
        append_str(u);
        append_lit("\",\"cells\":[[");
        int dr = (unit == 'c') ? 1 : 0;
        int dc = (unit == 'r') ? 1 : 0;
        for (int k = 0; k < 3; k++) {
            if (k > 0) append_lit(",[");
            append_int(anchor_row + k * dr);
            append_lit(",");
            append_int(anchor_col + k * dc);
            append_lit("]");
        }
        append_lit("],\"value\":");
        append_int(value);
        append_lit(",\"from_level\":");
        append_int(from_level);
        append_lit(",\"new_level\":");
        append_int(from_level + 1);
        append_lit("}\n");
    }

    // Phase 2: set detection (antecedent). `kind` is one of
    // "naked_pair"/"naked_triple"/"naked_quad"/
    // "hidden_pair"/"hidden_triple"/"hidden_quad".
    // `unit` is 'r'/'c'/'b'. `set_cells` are the SET-member cell indices
    // (0..80, row-major); `set_values` is the bitmask of locked digits.
    void naked_set(const char* kind, char unit,
                   const unsigned char* set_cells, int n_set,
                   unsigned short set_values, int level) {
        ensure(192 + n_set * 12);
        append_lit("{\"event\":\"step\",\"type\":\"");
        append_str(kind);
        append_lit("\",\"unit\":\"");
        char u[2] = {unit, 0}; append_str(u);
        append_lit("\",\"cells\":[");
        for (int k = 0; k < n_set; k++) {
            if (k > 0) append_lit(",");
            append_lit("[");
            append_int(set_cells[k] / 9);
            append_lit(",");
            append_int(set_cells[k] % 9);
            append_lit("]");
        }
        append_lit("],\"values\":");
        append_value_set(set_values);
        append_lit(",\"level\":");
        append_int(level);
        append_lit("}\n");
    }

    // Phase 2: cell-level candidate removal (single-cell consequence).
    // `reason` is the antecedent kind ("naked_set", etc.); `unit` is
    // 'r'/'c'/'b' for the unit the antecedent was in; `values` MUST be
    // the bits actually removed from THIS cell (caller responsibility:
    // values = pre_candidates[cell] & deduction_mask, skip when zero).
    void eliminate(const char* reason, char unit,
                   int row, int col,
                   unsigned short values, int level) {
        ensure(176);
        append_lit("{\"event\":\"step\",\"type\":\"eliminate\","
                   "\"reason\":\"");
        append_str(reason);
        append_lit("\",\"unit\":\"");
        char u[2] = {unit, 0}; append_str(u);
        append_lit("\",\"cell\":[");
        append_int(row);
        append_lit(",");
        append_int(col);
        append_lit("],\"values\":");
        append_value_set(values);
        append_lit(",\"level\":");
        append_int(level);
        append_lit("}\n");
    }

    // Phase 2: group-cell elimination event — used for triad reductions
    // where a single deduction implies an elimination across multiple
    // cells uniformly. `values` here is the DEDUCTION mask (what the
    // deduction asserts cannot be in any of the listed cells); per-cell
    // actual removal = pre_cell_candidates & values. The plural form
    // (`cells` vs `cell`) discriminates from the single-cell event.
    void eliminate_group(const char* reason, char unit,
                         const unsigned char* cells, int n_cells,
                         unsigned short values, int level) {
        ensure(192 + n_cells * 12);
        append_lit("{\"event\":\"step\",\"type\":\"eliminate\","
                   "\"reason\":\"");
        append_str(reason);
        append_lit("\",\"unit\":\"");
        char u[2] = {unit, 0}; append_str(u);
        append_lit("\",\"cells\":[");
        for (int k = 0; k < n_cells; k++) {
            if (k > 0) append_lit(",");
            append_lit("[");
            append_int(cells[k] / 9);
            append_lit(",");
            append_int(cells[k] % 9);
            append_lit("]");
        }
        append_lit("],\"values\":");
        append_value_set(values);
        append_lit(",\"level\":");
        append_int(level);
        append_lit("}\n");
    }

    // Phase 3: fish antecedent (X-Wing K=2, Swordfish K=3, Jellyfish K=4).
    // `base_kind` is 'r' (rows-as-base, cols-as-cover) or 'c' (cols-as-base,
    // rows-as-cover). `base_mask` / `cover_mask` are 9-bit bitmasks of
    // unit indices (0..8) where bit k = unit k participates. `digit` is
    // 1..9. Consequences follow as per-cell `eliminate` events with
    // reason:"fish".
    void fish(int size, int digit,
              char base_kind,
              unsigned short base_mask,
              unsigned short cover_mask,
              int level) {
        ensure(192);
        append_lit("{\"event\":\"step\",\"type\":\"fish\","
                   "\"size\":");
        append_int(size);
        append_lit(",\"digit\":");
        append_int(digit);
        append_lit(",\"base\":\"");
        char b[2] = {base_kind, 0}; append_str(b);
        append_lit("\",\"base_units\":");
        append_unit_set(base_mask);
        append_lit(",\"cover_units\":");
        append_unit_set(cover_mask);
        append_lit(",\"level\":");
        append_int(level);
        append_lit("}\n");
    }

    void backtrack(int from_level, int to_level) {
        backtracks_++;
        ensure(96);
        append_lit("{\"event\":\"step\",\"type\":\"backtrack\","
                   "\"from_level\":");
        append_int(from_level);
        append_lit(",\"to_level\":");
        append_int(to_level);
        append_lit("}\n");
    }

    void flush() {
        if (len_ == 0 || sink_ == nullptr) {
            return;
        }
        // fwrite is internally locked on a FILE*; per-thread buffers stay
        // line-coherent across the shared sink.
        std::fwrite(buf_, 1, len_, sink_);
        len_ = 0;
    }

private:
    void grow_to(size_t new_cap) {
        char* nbuf = (char*) std::realloc(buf_, new_cap);
        if (nbuf == nullptr) {
            std::fprintf(stderr, "trace::Emitter: OOM growing buffer to %zu\n", new_cap);
            std::abort();
        }
        buf_ = nbuf;
        cap_ = new_cap;
    }

    void ensure(size_t need) {
        if (len_ + need > cap_) {
            size_t new_cap = cap_ * 2;
            while (len_ + need > new_cap) new_cap *= 2;
            grow_to(new_cap);
        }
    }

    inline void append_chunk(const char* p, size_t n) {
        std::memcpy(buf_ + len_, p, n);
        len_ += n;
    }

    inline void append_str(const char* p) {
        size_t n = std::strlen(p);
        ensure(n);
        std::memcpy(buf_ + len_, p, n);
        len_ += n;
    }

    template <size_t N>
    inline void append_lit(const char (&s)[N]) {
        // Compile-time string literal length, minus the terminator.
        std::memcpy(buf_ + len_, s, N - 1);
        len_ += N - 1;
    }

    // Emit a value-set as a JSON array of 1..9 digits. `bits` is a
    // candidate bitmask (bit k set => digit k+1 present).
    inline void append_value_set(unsigned short bits) {
        ensure(32);
        append_lit("[");
        bool first = true;
        for (int d = 1; d <= 9; d++) {
            if (bits & (1u << (d - 1))) {
                if (!first) append_lit(",");
                first = false;
                buf_[len_++] = (char)('0' + d);
            }
        }
        append_lit("]");
    }

    // Emit a unit-set as a JSON array of 0-based unit indices (0..8).
    // Used for fish base_units / cover_units.
    inline void append_unit_set(unsigned short bits) {
        ensure(32);
        append_lit("[");
        bool first = true;
        for (int u = 0; u < 9; u++) {
            if (bits & (1u << u)) {
                if (!first) append_lit(",");
                first = false;
                buf_[len_++] = (char)('0' + u);
            }
        }
        append_lit("]");
    }

    // Hand-rolled int formatter. snprintf is too slow on hot path —
    // ~200ns/call vs ~5ns for this loop.
    inline void append_int(long long v) {
        ensure(24);
        if (v < 0) { buf_[len_++] = '-'; v = -v; }
        char tmp[20];
        int n = 0;
        if (v == 0) {
            buf_[len_++] = '0';
            return;
        }
        while (v > 0) {
            tmp[n++] = (char)('0' + (v % 10));
            v /= 10;
        }
        while (n--) buf_[len_++] = tmp[n];
    }

    FILE* sink_;
    char* buf_;
    size_t cap_;
    size_t len_;
    int thread_id_;
    long long guesses_;     // per-puzzle, reset in puzzle_end
    long long backtracks_;  // per-puzzle, reset in puzzle_end
};

// Per-thread emitter pointer. nullptr means tracing is disabled for this
// thread (and the helpers at the call sites short-circuit).
extern thread_local Emitter* current;

// Per-thread "what triggered the next phase_enter()" marker. The Schoku
// solver discovers a single via several distinct paths (search.hpp's naked
// scan, hidden_search.hpp's row/col passes, make_guess setting up the
// guessed digit), and the digit-entry path itself is shared (phase_enter).
// Each upstream site sets this marker just before returning Phase_Enter; the
// emitter reads it to label the event correctly. ER_None = unset (defensive
// fallback — emitted as reason:"unknown").
extern thread_local uint8_t next_entry_reason;

// ---- Convenience namespace-level helpers. The intent is that solver code
//      writes `if (trace::current) trace::naked_single(...)` rather than
//      threading the Emitter object through every signature.

inline void naked_single(int row, int col, int value, int level) {
    current->naked_single(row, col, value, level);
}
inline void deduced_single(int row, int col, int value, int level) {
    current->deduced_single(row, col, value, level);
}
inline void hidden_single(int row, int col, int value, char unit, int level) {
    current->hidden_single(row, col, value, unit, level);
}
inline void guess(int row, int col, int value, int from_level) {
    current->guess(row, col, value, from_level);
}
inline void guess_triad(int row, int col, int value, char unit, int from_level) {
    current->guess_triad(row, col, value, unit, from_level);
}
inline void backtrack(int from_level, int to_level) {
    current->backtrack(from_level, to_level);
}
inline void naked_set(const char* kind, char unit,
                      const unsigned char* set_cells, int n_set,
                      unsigned short set_values, int level) {
    current->naked_set(kind, unit, set_cells, n_set, set_values, level);
}
inline void eliminate(const char* reason, char unit,
                      int row, int col,
                      unsigned short values, int level) {
    current->eliminate(reason, unit, row, col, values, level);
}
inline void eliminate_group(const char* reason, char unit,
                            const unsigned char* cells, int n_cells,
                            unsigned short values, int level) {
    current->eliminate_group(reason, unit, cells, n_cells, values, level);
}
inline void fish(int size, int digit, char base_kind,
                 unsigned short base_mask, unsigned short cover_mask,
                 int level) {
    current->fish(size, digit, base_kind, base_mask, cover_mask, level);
}
inline void puzzle_start(const char* puzzle81, int puzzle_id) {
    current->puzzle_start(puzzle81, puzzle_id);
}
inline void puzzle_end(bool solved, const char* solution81,
                       long long n_guesses, long long n_backtracks) {
    current->puzzle_end(solved, solution81, n_guesses, n_backtracks);
}

// Reason for the next `enter_digit` call. The solver phases set this just
// before transitioning into Phase_Enter; enter.hpp reads it to emit the
// correct single/guess event. Encoded as a plain enum since it's referenced
// from SolveCtx as a member field.
enum EntryReason : uint8_t {
    ER_None = 0,
    ER_NakedSingle,
    ER_HiddenSingleRow,
    ER_HiddenSingleCol,
    ER_HiddenSingleBox,
    ER_Guess,            // entered via make_guess(idx,digit,...)
    // Placement forced by a higher-order deduction (fish elimination,
    // naked/hidden set elimination, UR-driven elimination, BUG+1). The
    // *antecedent* event for these placements is Phase 2-4 territory;
    // until that lands, the trace labels them with reason:"deduced" so a
    // distillation consumer can filter them out rather than mistake them
    // for a true naked-single inference.
    ER_DeducedSingle,
};

} // namespace trace
