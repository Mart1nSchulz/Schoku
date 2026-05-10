// Out-of-class definition of `SolveCtx<verbose>::do_unique_rectangles()`.
// AlphaEvolve mutation unit: this file is the entire surface for Unique avoidable rectangles (OPT_UQR).
// Replace the function body to mutate the strategy without touching
// the rest of the solver.
//
// Returns Phase_HiddenSearch when the block finishes without firing
// any redirect (the dispatcher then continues to the next block in
// phase_hidden_search). Any other returned SolverPhase short-circuits
// back to the dispatcher (semantically identical to the pre-refactor
// `goto X` exits inside the block).
//
// Build flag: this body is empty when OPT_UQR is undefined; the helper
// then unconditionally returns Phase_HiddenSearch.
//
// CONTRACT: private include fragment, must be #included exactly once
// from inside `namespace Schoku { ... }` after solver/solve_ctx.hpp.
#pragma once


template <Verbosity verbose>
__attribute__((always_inline)) inline SolverPhase SolveCtx<verbose>::do_unique_rectangles() {
// Original code uses `#if OPT_UQR`, not `#ifdef`. Equivalent under the
// project's `-DOPT_UQR` convention (defined-without-value vs not-defined),
// but kept verbatim to avoid behavior drift if anyone builds with OPT_UQR=0.
#if OPT_UQR
    //
    // Unique (Avoidable) Rectangles
    //
    // The theory:
    // For each (a) retangular four cells, (b) lying in two boxes (i.e. two cells in each box)
    // and (c) of which none is set with a preset value:
    //   Resolving these cells such that both pairs of diagonally opposite corners have
    //   the same value _always_ and _automatically_ will lead to at least _2_ different
    //   solutions.
    // Proof: In any valid solution interchanging these two pairs
    //   will lead to a different yet equally valid solution.
    //
    // The main implication is that :: in a Sudoku puzzle that has exactly one solution
    // (e.g. as part of the rules for that puzzle, as commonly is the case) ::, it is
    // possible during the solving process to detect such patterns and remove candidates
    // that would otherwise lead to multiple possible solutions and hence cannot be
    // considered for a solution.
    //
    // In terms of wording, such a puzzle solution is called 'unique' or 'the solution'.
    //
    // General notes:
    //
    // 1. There is nothing particularly special about these rectangles.  The same
    //    poperty of non-uniqueness applies to many several other patterns that are
    //    'closed' in themselves.  The unique avoidable rectangle is just the simplest
    //    of those patterns.  To witness, aligned three cells in a box, with a coresponding
    //    matching box, a pattern {a,b},{b,c},{a,c} will do the same, as will a
    //    transitive combination of three aligned pairs {a,b},{b,c},{a,c}.
    //    The pattern {a,b}, {a,b}, {a,b} can bend by 90 degrees for a corner.
    //    The pattern called 'binary universal grave', which is an end game pattern also
    //    falls into the same category, as it implies multiple solutions.
    //
    // 2. It is perfectly possible to completely disregard these patterns of multiple
    //    solutions.  Just be prepared to take a guess to find a solution.
    //    Remember always that for a 'regular' Sudoku puzzle there will be a solution that
    //    does not show such a pattern.  Hence the notion of 'avoidable' patterns.
    //
    // Special notes:
    //
    // 3. The definition above only describes what will eventually be true of the final
    //    solution in the case of a non-unique rectangle.
    //    There is no dependency on currently locked or unlocked cells other than the
    //    preset cells of the puzzle.  In particular, unlike in most other solving
    //    algorithms, the knowledge of the preset positions
    //    (as opposed to resolved cells during the solving process)
    //    is required to detect all such rectangles.
    //
    // 4. Having resolved one or more or even all cells of such a unique rectangle to
    //    candidates that correspond to the (possible) pattern does not affect the
    //    non-uniqueness of the solution.  Remember that these unique
    //    rectangles are sought out to be avoided - therefore it is possible that cells of
    //    the rectangle are resolved but the rectangle itself is yet to be avoided.
    //
    // Solution strategy:
    //
    // 5. A solving algorithm for these patterns, once detected, needs to consider:
    //    a) if the detected pattern cannot be avoided in a regular puzzle:
    //       - if a guess was made: back track!
    //       - if no guess was made: either the puzzle has no single solution, or a bug occurred.
    //    b) if the detected pattern still can be avoided:
    //       - determine which candidate(s) to remove (if any) assuming a single solution.
    //
    // 6. For a puzzle solving algorithm that includes proof of uniqueness _and_ back tracking:
    //       The avoidance of unique rectangles presumes that only a single solution exists.
    //       Uniqueness checking presumes no such thing, and therefore needs to allow for the
    //       unique rectangle to form.  Either prevent UQR detection altogether when
    //       checking for uniqueness, or choose avoidandance of the unique rectangle as the
    //       primary path of a guess.
    //
    // Terminology:
    // The smallest cell number of all rectangle corners: start cell
    // A side of the rectangle accross 2 boxes: long edge
    // A side od the rectangle within the same box: base edge
    // All cells are connected to a base edge and a long edge.
    // The cells are numbered in clockwise manner from 0 to 3.
    // Each cell is associated to its clockwise right edge.
    // Each long/base edge has an long/base opposite edge.
    // Each cell of the rectangle has a diagonally opposite cell.
    //
    // Unique rectangle types, homegrown notation (aliases where given curtesy of sudokowiki.org):
    // UR-3S
    // - 3 cells with singles, of which 2 have the same value.
    // - action: in the only corner with >1 candidates, eliminate (if it is present)
    //   the candidate that would otherwise complete the UR.
    // Note:
    // This is the only scenario where there is no need for a pair.
    // Therefore in the order of eliminations of possible URs, it needs to be checked before
    // the possible URs are filtered by bivalues.
    //
    // UR-3P - three pairs (aka Type 1):
    // - 3 cells with the same pair
    // - precondition: 4th cell contains one or more candidates of the pair
    // - action: remove the pair candidates from the 4th cell
    //
    // UR-2P-I with direct elimination of UR candidates
    // (I for immediate) (no alias, Hodoku type 6):
    // - two cells contain the same pair of candidates.
    //   These cells can be adjacent or on a diagonal.
    //   If the two cells are adjacent, they form a conjugate pair and share an edge as a strong link.
    //   Note: for the diagonal, the candidate is removed from the diagonal cells (!).
    // - precondition: the other corners contain two candidates from the pair
    //   and other candidates.
    // - precondition 2: one of the start cell candidates x does not appear in either
    //   of 2 parallel edges other than the UR corners (x is a conjugate for these edges).
    //   action: remove y (the other candidate from the pair) from the other cells of the UR,
    //   or in the case of a diagonal, from both ends of the diagonal.
    // Note: if just one edge from one of the pair cells meets precondition 2,
    //   y can only be removed from the other cell not on that edge (does not apply to diagonal).
    // Note 2: UR-2P-(B,L,D) also apply independantly.
    //
    // UR-2P-(B,L,D) with elimination of required candidates present in the corners outside of the UR
    // B: base edge, L: long edge, D: diagonal
    // UR-2P-B (aka Type 2):
    // - base edge corners contain the same pair of candidates (could also both be resolved
    //   but not preset)
    // - precondition: opposite edge corners contain two candidates from the pair
    // - precondition 2: there are other candidates present (in both corners) A of the
    //   other (non-pair) cells
    // - action: consider the size of A: N
    //   N==1: simply remove the extra candidate in A from all cells visible from the non-pair cells
    //   except the two corner cells.
    //   N>1: for each section S shared by the non-pair cells:
    //   examine S for a naked set T of A. If size(T) == N-1:
    //   remove the extra candidates of A from the cells of S that are in T and not corners cells.
    //
    // UR-2P-L (aka Type 2B):
    // - long edge corners contain the same pair of candidates
    // preconditions and actions are the same as for UR-2P-B
    //
    // UR-2P-D (aka Type 2C):
    // - diagonal corners contain the same pair of candidates
    // - precondition: opposite diagonal corners contain the two candidates from the pair
    // - precondition 2: there is a single other candidate present (in both opposite diagonal corners)
    // - action: remove extra candidate from other cells of the triads visible from the corners of the opposite diagonal.
    // Note:
    //   UR-2P-D is quite rare, probably because most likely UR-1P will be detected beforehand
    //   and remove the triggers for detection.
    //
    // UR-1P (no alias, Hodoku: Hidden Rectangle):
    // - one cell (1st cell) contains a pair of candidates, x and y. Select y.
    // - precondition: all other cells of the UR contain candidates of the same pair plus some other
    //   candidates.
    // - precondition 2: the diagonal opposite corner cell, when set to y, will force the other
    //   diagonal's cells both to be set to x.
    //   action: the candidate y can be removed from the cell diagonally opposite to the start cell.
    //   Notes:
    //   1. UR-1P can be easily applied and visualized:
    //   look at the 1st cell and identify the four corners. For one of the 1st cell's
    //   candidates x, if the UR in question provides all the candidate locations in the
    //   opposite edge and the other long edge (i.e. strongly linked).  This is sufficient.
    //
    // Summary of additional preconditions:
    // UR-3P unique rectangle
    //    None
    //
    // UR-1P unique rectangle
    //    conjugate pairs on the same digit (in row/col or box), for the opposite base and opposite long edge.
    //
    // UR-2P-I
    //    pair of opposite strong edges.
    //    a single strong edge on one side eliminates a single candidate on the other side.
    //
    // UR-2P-* unique rectangle
    //    one or multiple 'extra' cell candidates present in both other cells
    //    in the case of a diagonal, only one 'extra' candidate can be present
    //
    // Data structures and algorithm:
    // The data to process UQRs comes from differernt sources:
    // 1. preset cell locations (captured on entry and pre-processed at first use)
    //    this allows skipping 50% or more of all possible unique rectangles
    // 2. bivalues as bit pattern (this data is shared with other algorithms,
    //    e.g. binary universal grave plus one).
    //    This can be leveraged to skip a good percentage of UQRs.
    // 3. SIMD grid processing to collect information on UQRs (diagonal intersection)
    //    as information for further processing.
    //    Once collected for a pair of rows of a band, UQRs can be further
    //    qualified and classified.
    //

if ( mode_uqr )
{
    // compute just in time the superimposed preset rows by band and rc_pair:
    if ( !have_superimposed_preset_rows ) {
        have_superimposed_preset_rows = true;
        for ( int i=0; i<3; i++ ) {
            unsigned short idx[3];
            idx[0] = original_locked.get_indexbits(27*i, 9);
            idx[1] = original_locked.get_indexbits(27*i+9, 9);
            idx[2] = original_locked.get_indexbits(27*i+18, 9);
            superimposed_preset_rows[i][0] = idx[0] | idx[1];
            superimposed_preset_rows[i][1] = idx[0] | idx[2];
            superimposed_preset_rows[i][2] = idx[1] | idx[2];
        }
    }

    bit128_t *candidate_bits_by_value = solverData.getCbbvs(candidates);

    // Process uqrs by band
    unsigned char band = 0;
    if ( last_entered_count_uqr != current_entered_count) {
        // the high byte is set to the stackpointer, so that this works
        // across guess/backtrack
        last_entered_count_uqr = current_entered_count;
    } else {
        band = last_band_uqr;
    }
    for ( ; band<6; band++) {
        bool found_update = false;
        bool rowband = band<3?true:false;
        unsigned char rc_band = band%3;
        last_band_uqr = band;

        if ( rowband ) {
            unsigned int bandbits = ((bit128_t*)unlocked)->get_indexbits(27*band, 27);

            if ( bandbits == 0 ) {  // questionable
                continue;
            }
        } else {
            // once per puzzle:
            // initialize transposed presets and superimposed columns
            if ( !have_superimposed_preset_cols ) {
                have_superimposed_preset_cols = true;

                // first transpose original_locked
                // start by extracting the 9 bits corresponding to each row
                unsigned long long ol64[2] = { original_locked.u64[0] & 0x7fffffffffffffff, original_locked.get_rshfti<63>() };
                unsigned short ol[16] = { (unsigned short)ol64[0], (unsigned short)(ol64[0]>>9), (unsigned short)(ol64[0]>>18), (unsigned short)(ol64[0]>>27),
                                          (unsigned short)(ol64[0]>>36), (unsigned short)(ol64[0]>>45), (unsigned short)(ol64[0]>>54),
                                          (unsigned short)ol64[1], (unsigned short)(ol64[1]>>9), 0, 0, 0, 0, 0, 0, 0 };

                // one off for digit 9
                __m256i c = _mm256_and_si256(mask1ff, *(__m256i_u*) &ol[0]);
                __m256i c2 = _mm256_permute4x64_epi64(_mm256_packus_epi16(_mm256_srli_epi16(c,1), _mm256_setzero_si256()), 0xD8);
                original_locked_transposed.u64[1] = _mm256_movemask_epi8(c2) << 8;
                c = _mm256_permute4x64_epi64(_mm256_packus_epi16(_mm256_and_si256(c, maskff), _mm256_setzero_si256()), 0xD8);
                unsigned short tmp = _mm256_movemask_epi8(c);
                original_locked_transposed.u64[0]  = (tmp & 1LL)<<63;
                original_locked_transposed.u64[1]  |= tmp >> 1;
                for ( short off=54; off>=0; off -= 9 ) {
                    c = _mm256_slli_epi16(c,1);
                    original_locked_transposed.u64[0] |= (unsigned long long)_mm256_movemask_epi8(c)<<off;
                }

                // second, superimpose the transposed columns
                for ( int i=0; i<3; i++ ) {
                    unsigned short idx[3];
                    idx[0] = original_locked_transposed.get_indexbits(27*i, 9);
                    idx[1] = original_locked_transposed.get_indexbits(27*i+9, 9);
                    idx[2] = original_locked_transposed.get_indexbits(27*i+18, 9);
                    superimposed_preset_cols[i][0] = idx[0] | idx[1];
                    superimposed_preset_cols[i][1] = idx[0] | idx[2];
                    superimposed_preset_cols[i][2] = idx[1] | idx[2];
                }
            }
        }

        // load band data as follows:
        __m256i rc_v[3];

        unsigned short __attribute__ ((aligned(64))) res_data[24]; // contains result data of res

        if ( rowband ) {
            unsigned char rc_indx_ = rc_band*27;

            rc_v[0] = _mm256_loadu2_m128i((__m128i*)&candidates[rc_indx_+6],(__m128i*)&candidates[rc_indx_]);
            rc_v[0] = _mm256_shuffle_epi8(rc_v[0], lineshuffle);
            rc_indx_+=9;
            rc_v[1] = _mm256_loadu2_m128i((__m128i*)&candidates[rc_indx_+6],(__m128i*)&candidates[rc_indx_]);
            rc_v[1] = _mm256_shuffle_epi8(rc_v[1], lineshuffle);
            rc_indx_+=9;
            rc_v[2] = _mm256_loadu2_m128i((__m128i*)&candidates[rc_indx_+6], (__m128i*)&candidates[rc_indx_]);
            rc_v[2] = _mm256_shuffle_epi8(rc_v[2], lineshuffle);
        } else {
            unsigned short *cp = &candidates[rc_band*3];

            rc_v[0] = _mm256_setr_epi16(cp[0], cp[9], cp[18], 0, cp[27], cp[36], cp[45], 0,
                                        cp[54], cp[63], cp[72], 0, 0, 0, 0, 0);
            cp += 1;
            rc_v[1] = _mm256_setr_epi16(cp[0], cp[9], cp[18], 0, cp[27], cp[36], cp[45], 0,
                                        cp[54], cp[63], cp[72], 0, 0, 0, 0, 0);
            cp += 1;
            rc_v[2] = _mm256_setr_epi16(cp[0], cp[9], cp[18], 0, cp[27], cp[36], cp[45], 0,
                                        cp[54], cp[63], cp[72], 0, 0, 0, 0, 0);
        }

        for ( unsigned char rc_pair = 0; rc_pair < 3; rc_pair++ ) {

            // for linev[0]: swap quads to [0, 1, 0, 2]
            // for linev[1]: swap quads to [1, 0, 2, 0]
            // for linev[2]: swap linev[0] quads to [1,3,x2,x3]
            // for linev[3]: swap linev[1] quads to [2,0,x2,x3]
            __m256i linev[4] = { _mm256_permute4x64_epi64(rc_v[row_combos[rc_pair][0]], 0x84),
                                 _mm256_permute4x64_epi64(rc_v[row_combos[rc_pair][1]], 0x21),
                                 _mm256_permute4x64_epi64(linev[0], 0xED),
                                 _mm256_permute4x64_epi64(linev[1], 0xE2) };
            for ( unsigned char perm = 0; perm<3; perm++ ) {
                *(__m256i_u*)&res_data = _mm256_and_si256(linev[0], linev[1]);
                // result (for first iteration):
                // 0.0&1.3, 0.1&1.4, 0.2&1.5, - 0.3&1.0, 4.1&r1.1, 0.5&r1.2, -
                // 0.0&1.6, 0.1&1.7, 0.2&1.8, - 0.6&1.0, 7.1&r1.1, 0.8&r1.2, -

                // Check for UQRs between box 1 and 2
                 *(__m128i_u*)&res_data[16] = _mm256_castsi256_si128(_mm256_and_si256(linev[2], linev[3]));

                // evaluate permutated row/column results for valid uqrs:
                // - if the uqr contains a preset, it is eliminated.
                // - check diag1, diag2 both to be not 0, and at the same, eliminate single uqr candidates in the opposite diagonal.
                // - check that there is at least one bivalue (unless there are 3 single cands)

                const Uqr *permp = cuqrs[perm];
                unsigned char dist2 = rc_pair == 1 ? 2:1;
                unsigned short preset_test;
                unsigned char start_row_indx = 0;
                unsigned char start_col_indx = 0;
                if ( rowband ) {
                    start_row_indx = (rc_band*3 + row_combos[rc_pair][0])*9;
                    preset_test = superimposed_preset_rows[rc_band][rc_pair];
                } else {
                    start_col_indx = rc_band*3 + row_combos[rc_pair][0];
                    preset_test = superimposed_preset_cols[rc_band][rc_pair];
                }
                for (unsigned char uqr_cnt=0; uqr_cnt<9; uqr_cnt++, permp++) {
                    if ( preset_test & permp->pattern ) {
                        continue;
                    }
                    const unsigned char *cuqr_accessp = cuqr_access[uqr_cnt];
                    unsigned short diag[2] = { res_data[cuqr_accessp[0]],
                                               res_data[cuqr_accessp[1]] };

                    // cross elimination based on diagonal info
                    if (__popcnt16(diag[0]) == 1 ) {
                        diag[1] &= ~diag[0];
                    } else if (__popcnt16(diag[1]) == 1 ) {
                        diag[0] &= ~diag[1];
                    }

                    if (diag[0] == 0 || diag[1] == 0 ) {
                        continue;
                    }

                    // before eliminating other rectangles based on bivalues,
                    // check for 3 singles with 2 of them the same (these will always
                    // form a diagonal).
                    // remove the other single from the fourth corner to avoid a unique
                    // rectangle from forming.
                    UqrCorner uqr_corners[4];
                    UqrPair uqr_pairs[3];
                    unsigned short uqr_singles = 0;
                    unsigned short uqr_singles_cnt = 0;
                    unsigned char start_cell = rowband ? permp->start_cell +  start_row_indx : (permp->start_cell*9 + start_col_indx);
                    unsigned char dist1    = permp->dist;

                    // clockwise urq corners
                    uqr_corners[0].indx = start_cell;
                    if ( rowband ) {
                        uqr_corners[1].indx = start_cell+dist1;
                        uqr_corners[2].indx = start_cell+dist1+9*dist2;
                        uqr_corners[3].indx = start_cell+9*dist2;
                    } else {
                        // in the vertical band, dist1 and dist2 are transposed
                        uqr_corners[1].indx = start_cell+dist2;
                        uqr_corners[2].indx = start_cell+9*dist1+dist2;
                        uqr_corners[3].indx = start_cell+9*dist1;
                    }

                    // identify any corner with a bivalue or single
                    unsigned short all_digits = 0;
                    unsigned char not_singles = 0;
                    for ( int i=0; i<4; i++ ) {
                        unsigned short dgts = candidates[uqr_corners[i].indx];
                        all_digits |= dgts;
                        unsigned char cnt = __popcnt16(dgts);
                        if ( cnt == 1 ) {
                            uqr_singles |= dgts;
                            uqr_singles_cnt++;
                        } else {
                            not_singles |= 1<<i;
                        }
                    }

                    // Pattern UR-3S
                    if ( __popcnt16(uqr_singles) == 2 && uqr_singles_cnt == 3 ) {
                        unsigned char ix = __tzcnt_u16(not_singles);
                        char ret[32];
                        format_candidate_set(ret, uqr_singles);
                        unsigned char celli = uqr_corners[ix].indx;
                        unsigned short single = candidates[uqr_corners[(ix+2)&3].indx];
                        if ( (candidates[celli] & single) && __popcnt16(candidates[celli]) > 1 ) {
                            if ( verbose != VNone ) {
                                counters.unique_rectangles_avoided++;
                            }
                            grid_state->updated.set_indexbit(celli);
                            // unless under 'Regular' rules, capture the avoidable UQR as a guess
                            // provide 'resolution' in form of a guess
                            if ( rules != Regular ) {
                                if ( verbose == VDebug ) {
                                    snprintf(guess_message[0], 196, "to allow unique check of unique rectangle: %s %s - %s\n        3 singles pattern: remove candidate %d from cell %s",
                                            ret, cl2txt[uqr_corners[0].indx], cl2txt[uqr_corners[2].indx],
                                            1+__tzcnt_u16(single), cl2txt[celli]);
                                    snprintf(guess_message[1], 196, "engender unique rectangle %s %s - %s for subsequent unique checking:\n        3 singles pattern: set %d at %s",
                                            ret, cl2txt[uqr_corners[0].indx], cl2txt[uqr_corners[2].indx],
                                            1+__tzcnt_u16(single), cl2txt[celli]);
                                }
                                // call make_guess using a lambda
                                grid_state = grid_state->make_guess<verbose>(
                                    celli,
                                    [=] (GridState &oldgs, GridState &newgs, const char *msg[]) {
                                    // this is the avoidance side of the UQR resolution
                                    // the GridState to continue with:
                                    newgs.candidates[celli] &= ~single;
                                    msg[0] = guess_message[0];
                                    // this is to provoke the UQR
                                    // the GridState to back track to:
                                    oldgs.candidates[celli]  &= ~newgs.candidates[celli];
                                    msg[1] = guess_message[1];
                                }, solverData.counters, solverData.output);
                                return Phase_GuessMadeWithIncr;
                            }
                            // otherwise simply avoid the UQR:
                            candidates[celli] &= ~single;
                            if ( verbose == VDebug ) {
                                solverData.printf("avoiding unique rectangle: %s %s - %s\n3 singles pattern: remove candidate %d from cell %s\n",
                                    ret, cl2txt[uqr_corners[0].indx], cl2txt[uqr_corners[2].indx],
                                    1+__tzcnt_u16(single), cl2txt[celli]);
                            }
                            if ( (candidates[celli] & (candidates[celli]-1)) == 0) {
                                e_i = celli;
                                e_digit = candidates[celli];
                                if ( verbose == VDebug ) {
                                    solverData.printf("naked  single      ");
                                }
                                trace::next_entry_reason = trace::ER_DeducedSingle;  // Phase 2: refine to fish/set/ur reason
                                return Phase_Enter;
                            }
                            found_update = true;
                        }
                    }

                    // if all corners together contain the same 2 digits, a unique rectangle has been found.
                    if ( check_back && not_singles && __popcnt16(all_digits) == 2 ) {
                        if ( grid_state->stackpointer && rules == Regular ) {
                            if ( verbose == VDebug ) {
                                char ret[32];
                                format_candidate_set(ret, all_digits);
                                solverData.printf("back track - found a completed unique rectangle %s at %s %s\n", ret, cl2txt[uqr_corners[0].indx], cl2txt[uqr_corners[2].indx]);
                            }
                            return Phase_Back;
                        } else {
                            // there's no point doing anything here... except:
                            if ( grid_state->stackpointer == 0 ) {
                                status.unique = false;
                            }
                            // not even:
                            // solverData.printf("consciously ignoring a completed unique rectangle at %s %s\n", cl2txt[uqr_corners[0].indx], cl2txt[uqr_corners[2].indx]);
                        }
                    }

                    bit128_t corner_bits {};
                    if ( rowband ) {
                        corner_bits.set_indexbits( permp->pattern, start_row_indx, 9);
                        corner_bits.set_indexbits( permp->pattern, start_row_indx+9*dist2, 9);
                    } else {
                        for ( unsigned char k=0; k<4; k++ ) {
                            corner_bits.set_indexbit(uqr_corners[k].indx);
                        }
                    }

                    if ( (solverData.getBivalues(candidates) & corner_bits) == (__uint128_t)0 ) {
                        continue;
                    }
                    if ( verbose != VNone ) {
                        counters.unique_rectangles_checked++;
                    }
                    // complete corner infos, including edges and uqr_pairs infos
                    for ( int i=0; i<4; i++ ) {
                        unsigned short dgts = candidates[uqr_corners[i].indx];
                        unsigned char cnt = __popcnt16(dgts);
                        if ( cnt == 1 ) {
                            uqr_corners[i].is_single = true;
                        } else if ( cnt == 2 ) {
                            unsigned char pi = 0;
                            // search for pair
                            for ( ; pi<3; pi++) {
                                if ( uqr_pairs[pi].cnt == 0 ) {
                                    uqr_pairs[pi].digits = dgts;
                                    break;
                                } else if ( uqr_pairs[pi].digits == dgts ) {
                                    break;
                                }
                            }
                            if ( pi<3 ) {
                                uqr_pairs[pi].cnt++;
                                uqr_pairs[pi].crnrs |= 1<<i;
                            }
                            uqr_corners[i].pair_indx = pi;
                            uqr_corners[i].is_pair = true;
                        }
                        if ( i & 1 ) {
                            uqr_corners[i].right_edge = (__uint128_t*)small_index_lut[uqr_corners[i].indx%9][Col];
                        } else {
                            uqr_corners[i].right_edge = (__uint128_t*)small_index_lut[uqr_corners[i].indx/9][Row];
                        }
                    }

                    // based on pairs count, process the different scenarios
                    for ( int pi=0; pi<3; pi++ ) {
                        if ( uqr_pairs[pi].cnt == 0 ) {
                            // no pairs (cannot happen)
                            break;
                        }
                        switch ( uqr_pairs[pi].cnt ) {
                        case 3:  // UR-3P (aka Type 1):
                        {
                            // find the 4th corner
                            unsigned char corner4_index = uqr_corners[__tzcnt_u16(~uqr_pairs[pi].crnrs)].indx;
                            unsigned short pair = uqr_pairs[pi].digits;
                            if ( candidates[corner4_index] != pair ) {
                                // unless under 'Regular' rules, provide 'resolution' in form of a guess
                                if ( verbose != VNone ) {
                                    counters.unique_rectangles_avoided++;
                                }
                                if ( rules != Regular ) {
                                    unsigned short other_cands = candidates[corner4_index] & ~pair;
                                    if ( verbose == VDebug ) {
                                        char ret[32];
                                        char ret2[32];
                                        format_candidate_set(ret, pair);
                                        format_candidate_set(ret2, other_cands);
                                        snprintf(guess_message[0], 196, "to allow unique check of unique rectangle: %s %s - %s\n        3 pairs pattern: remove candidates %s from %s",
                                                 ret, cl2txt[uqr_corners[0].indx], cl2txt[uqr_corners[2].indx],
                                                 ret, cl2txt[corner4_index]);
                                        snprintf(guess_message[1], 196, "engender unique rectangle %s %s - %s for subsequent unique checking:\n        3 pairs pattern: remove candidates %s from %s",
                                                 ret, cl2txt[uqr_corners[0].indx], cl2txt[uqr_corners[2].indx],
                                                 ret2, cl2txt[corner4_index]);
                                    }
                                    // call make_guess using a lambda
                                    grid_state = grid_state->make_guess<verbose>(
                                        corner4_index,
                                        [=] (GridState &oldgs, GridState &newgs, const char *msg[]) {
                                        // this is the avoidance side of the UQR resolution
                                        // the GridState to continue with:
                                        newgs.candidates[corner4_index] = other_cands;
                                        msg[0] = guess_message[0];
                                        // this is to provoke the UQR
                                        // the GridState to back track to:
                                        oldgs.candidates[corner4_index]  &= ~other_cands;
                                        msg[1] = guess_message[1];
                                    }, solverData.counters, solverData.output);
                                    return Phase_GuessMadeWithIncr;
                                }
                                // simply avoid the UQR
                                candidates[corner4_index] &= ~pair;
                                if ( verbose == VDebug ) {
                                    char ret[32];
                                    format_candidate_set(ret, pair);
                                    solverData.printf("avoiding unique rectangle: %s %s - %s\n3 pairs pattern: remove candidates %s from %s\n",
                                            ret, cl2txt[uqr_corners[0].indx], cl2txt[uqr_corners[2].indx],
                                            ret, cl2txt[corner4_index]);
                                }
                                grid_state->updated.set_indexbit(corner4_index);
                                if ( (candidates[corner4_index] & (candidates[corner4_index]-1)) == 0 ) {
                                    e_i = corner4_index;
                                    e_digit = candidates[corner4_index];
                                    if ( verbose == VDebug ) {
                                        solverData.printf("naked  single      ");
                                    }
                                    trace::next_entry_reason = trace::ER_DeducedSingle;  // Phase 2: refine to fish/set/ur reason
                                    return Phase_Enter;
                                }
                                found_update = true;
                            }
                            break;
                        }
                        case 2:  // UR-2P-I and UR-2P-BLD
                        {
                            // Two pairs of candidates is fairly common.
                            // We have two cells, c1 and c2, which both contain the same
                            // pair (a and b).
                            // For x=a or x=b we examine the cells diagonally opposite of
                            // c1 and c2, o1 and o2.
                            // We call y the other candidate of a or b so that x != y.
                            // (not to get confused, c1 and c2 can by diagonally
                            // opposite of each other, in which case o1=c1 and o2=c2)
                            // What are the conditions to effect, if y is chosen in o1 or o2
                            // that we obtain the unique rectangle x y x y.
                            // That condition is simple:
                            // - if in any pair of opposing edges, x is conjugate with
                            //   the x in the other corner on that edge, then
                            //   y must not be chosen in either o1 or o2 or a unique rectangle forms.
                            //   Hence we should deselect y in o1 and o2.
                            //   if c1 and c2 are adjacent (not diagonal), and the link between c1
                            //   and o1 is weak, but between o2 and c2 it is strong, then y can
                            //   only be eliminated from o1.
                            // Note: there is a part 2 to this procedure.
                            // We can form the set of extra candidates of o1 and o2 if they are
                            // adjacent, and try to find a set in the row or col that contains these
                            // candidates (as they are required they form a virtual set of candidates)
                            // and if a set can be found, its candidates can be eliminated elsewhere
                            // in the row/col.

                            // uqr_cand: the tentative value of x
                            unsigned short uqr_cand = _blsi_u32(uqr_pairs[pi].digits);
                            // all the cells containing that candidate:
                            __uint128_t *cbbv = &candidate_bits_by_value[__tzcnt_u16(uqr_cand)].u128;
                            // either opposite pair will do for both diagonal and side by side
                            // conjugate pairs.
                            // setup the opposing edges without the corners:
                            __uint128_t uqr_opp_edges[2] = {corner_bits,corner_bits};
                            uqr_opp_edges[0] ^= *uqr_corners[0].right_edge | *uqr_corners[2].right_edge;
                            uqr_opp_edges[1] ^= *uqr_corners[1].right_edge | *uqr_corners[3].right_edge;
                            // determine strong edge:
                            unsigned int strong_edge = 0;
                            bool is_diag = false;
                            switch ( uqr_pairs[pi].crnrs ) {
                            case 0b101:
                            case 0b1010:
                                is_diag = true;
                                break;
                            case 0b1001:
                                strong_edge = 3;
                                break;
                            case 0b1100:
                                strong_edge = 2;
                                break;
                            case 0b110:
                                strong_edge = 1;
                                break;
                            default:
                                break;
                            }
                            unsigned short weak_corner = 0xff;
                            unsigned short weak_corner_y = 0xff;
                            int i=0;
                            for ( ; i<2; i++ ) {
                                if (    (uqr_opp_edges[0] & *cbbv) == (__uint128_t)0
                                     || (uqr_opp_edges[1] & *cbbv) == (__uint128_t)0 ) {
                                    // found x
                                    // try for the other candidate too as special case:
                                    // This is a back track scenario, as both candidates
                                    // when selected each cause a completed UR.
                                    // Quite rare as well.
                                    if ( check_back && uqr_cand != uqr_pairs[pi].digits ) {
                                        cbbv = &candidate_bits_by_value[__tzcnt_u16(uqr_cand ^ uqr_pairs[pi].digits)].u128;
                                        if (    (uqr_opp_edges[0] & *cbbv) == (__uint128_t)0
                                             || (uqr_opp_edges[1] & *cbbv) == (__uint128_t)0 ) {

                                            // find a corner with both digits
                                            unsigned short pair_digits = uqr_pairs[pi].digits;
                                            char ret[32];
                                            if ( verbose == VDebug ) {
                                                format_candidate_set(ret, pair_digits);
                                            }
                                            if ( grid_state->stackpointer && rules == Regular ) {
                                                if ( verbose == VDebug ) {
                                                    solverData.printf("back track - found an unavoidable unique rectangle %s at %s %s\n", ret, cl2txt[uqr_corners[0].indx], cl2txt[uqr_corners[2].indx]);
                                                }
                                                return Phase_Back;
                                            } else {
                                                // there's no point doing anything here...
                                                // not even:
                                                // solverData.printf("consciously ignoring a completed unique rectangle at %s %s\n", cl2txt[uqr_corners[0].indx], cl2txt[uqr_corners[2].indx]);
                                            }
                                        }
                                    }
                                    break;
                                } else if ( !is_diag ) {
                                    // strong set is the index of uqr_opp_edges containing the strong link provided by the pair (not for diagonal).
                                    unsigned int strong_set =  (strong_edge & 1) ? 1:0;
                                    // try both sides for weak set...
                                    if ( (((corner_bits & *uqr_corners[(strong_set+3)&3].right_edge) ^ *uqr_corners[(strong_set+3)&3].right_edge) & *cbbv) == 0 ) {
                                        weak_corner = strong_set+1;
                                    } else if ( (((corner_bits & *uqr_corners[strong_set+1].right_edge) ^ *uqr_corners[strong_set+1].right_edge) & *cbbv) == 0 ) {
                                        weak_corner = (strong_set+3)&3;
                                    }
                                    if ( weak_corner != 0xff ) {
                                        unsigned char weak_corner_indx =
                                              uqr_pairs[pi].crnrs & (1<<weak_corner) ?
                                              uqr_corners[(weak_corner+1)&3].indx :
                                              uqr_corners[weak_corner].indx;
                                        weak_corner_y =  uqr_cand ^ uqr_pairs[pi].digits;
                                        if ( (candidates[weak_corner_indx] & weak_corner_y) ) {
                                            grid_state->updated.set_indexbit(weak_corner_indx);
                                            if ( verbose != VNone ) {
                                                counters.unique_rectangles_avoided++;
                                            }
                                            // unless under 'Regular' rules, capture the avoidable UQR
                                            // provide 'resolution' in form of a guess
                                            char ret[32];
                                            if ( verbose == VDebug ) {
                                                format_candidate_set(ret, uqr_pairs[pi].digits);
                                            }
                                            if ( rules != Regular ) {
                                                if ( verbose == VDebug ) {
                                                    snprintf(guess_message[0], 196, "to allow unique check of unique rectangle: %s %s - %s\n        2 pair pattern: remove candidate %d at %s",
                                                             ret, cl2txt[uqr_corners[0].indx], cl2txt[uqr_corners[2].indx],
                                                             1+__tzcnt_u16(weak_corner_y), cl2txt[weak_corner_indx]);
                                                    snprintf(guess_message[1], 196, "engender unique rectangle %s %s - %s for subsequent unique checking:\n        2 pair pattern: set %d at %s",
                                                             ret, cl2txt[uqr_corners[0].indx], cl2txt[uqr_corners[2].indx],
                                                             1+__tzcnt_u16(weak_corner_y), cl2txt[weak_corner_indx]);
                                                }
                                                // call make_guess using a lambda
                                                grid_state = grid_state->make_guess<verbose>(
                                                    weak_corner_indx,
                                                    [=] (GridState &oldgs, GridState &newgs, const char *msg[]) {
                                                    // this is the avoidance side of the UQR resolution
                                                    // the GridState to continue with:
                                                    newgs.candidates[weak_corner_indx] &= ~weak_corner_y;
                                                    msg[0] = guess_message[0];
                                                    // this is to provoke the UQR
                                                    // the GridState to back track to:
                                                    oldgs.candidates[weak_corner_indx] = weak_corner_y;
                                                    msg[1] = guess_message[1];
                                                }, solverData.counters, solverData.output);
                                                return Phase_GuessMadeWithIncr;
                                            }
                                            // simply avoid the UQR
                                            candidates[weak_corner_indx] &= ~weak_corner_y;
                                            if ( verbose == VDebug ) {
                                                solverData.printf("avoiding unique rectangle: %s %s - %s\n2 pair pattern: remove candidate %d at %s\n",
                                                       ret, cl2txt[uqr_corners[0].indx], cl2txt[uqr_corners[2].indx],
                                                       1+__tzcnt_u16(weak_corner_y), cl2txt[weak_corner_indx]);
                                            }
                                            if ( (candidates[weak_corner_indx] & (candidates[weak_corner_indx]-1)) == 0 ) {
                                                e_i = weak_corner_indx;
                                                e_digit = candidates[weak_corner_indx];
                                                if ( verbose == VDebug ) {
                                                    solverData.printf("naked  single      ");
                                                }
                                                trace::next_entry_reason = trace::ER_DeducedSingle;  // Phase 2: refine to fish/set/ur reason
                                                return Phase_Enter;
                                            }
                                            found_update = true;
                                        }
                                    }
                                }
                                // set x to the other candidate
                                uqr_cand ^= uqr_pairs[pi].digits;
                                if ( uqr_cand == 0 ) {  // nothing to find
                                    i=2;
                                    break;
                                }
                                weak_corner = 0xff;
                                cbbv = &candidate_bits_by_value[__tzcnt_u16(uqr_cand)].u128;
                            }
                            if ( i<2 ) {
                                // select the diagonally opposite corners:
                                // rotate bits in the nibble by 2:
                                unsigned int ix = 0xf & (uqr_pairs[pi].crnrs | uqr_pairs[pi].crnrs<<4)>>2;
                                // from x and the pair of digits, determine y:

                                if ( uqr_cand != uqr_pairs[pi].digits ) {
                                    uqr_cand ^= uqr_pairs[pi].digits;
                                }
                                unsigned char indx2upd[2];
                                for ( int k=0; k<2; k++ ) {
                                    unsigned char indx = _tzcnt_u32(ix);
                                    indx2upd[k] = uqr_corners[indx].indx;
                                    ix = _blsr_u32(ix);
                                }
                                if (    candidates[indx2upd[0]] != uqr_cand
                                     || candidates[indx2upd[1]] != uqr_cand ) {
                                    char ret[32];
                                    if ( verbose == VDebug ) {
                                        format_candidate_set(ret, uqr_pairs[pi].digits);
                                    }
                                    found_update = true;
                                    if ( verbose != VNone ) {
                                        counters.unique_rectangles_avoided++;
                                    }
                                    if ( rules == Regular ) {
                                        candidates[indx2upd[0]] &= ~uqr_cand;
                                        candidates[indx2upd[1]] &= ~uqr_cand;
                                        if ( verbose == VDebug ) {
                                            solverData.printf("avoiding unique rectangle: %s %s - %s\n2 pair pattern: remove candidate %d from cells %s %s\n",
                                                   ret, cl2txt[uqr_corners[0].indx], cl2txt[uqr_corners[2].indx],
                                                   1+__tzcnt_u16(uqr_cand),
                                                   cl2txt[indx2upd[0]], cl2txt[indx2upd[1]]);
                                        }
                                        if (    (candidates[indx2upd[0]] & (candidates[indx2upd[0]] - 1)) == 0 ) {
                                            e_digit = candidates[indx2upd[0]];
                                            e_i = indx2upd[0];
                                            if ( verbose == VDebug ) {
                                                solverData.printf("naked  single      ");
                                            }
                                            trace::next_entry_reason = trace::ER_DeducedSingle;  // Phase 2: refine to fish/set/ur reason
                                            return Phase_Enter;
                                        } else if ( (candidates[indx2upd[1]] & (candidates[indx2upd[1]] - 1)) == 0 ) {
                                            e_digit = candidates[indx2upd[1]];
                                            e_i = indx2upd[1];
                                            if ( verbose == VDebug ) {
                                                solverData.printf("naked  single      ");
                                            }
                                            trace::next_entry_reason = trace::ER_DeducedSingle;  // Phase 2: refine to fish/set/ur reason
                                            return Phase_Enter;
                                        }
                                    } else {
                                        if ( is_diag ) {
                                            unsigned short other_cand = uqr_cand ^ uqr_pairs[pi].digits;
                                            if ( verbose == VDebug ) {
                                                snprintf(guess_message[0], 196, "to allow unique check of unique rectangle: %s %s - %s\n        2 pair pattern: remove candidate %d at %s and %s",
                                                         ret, cl2txt[uqr_corners[0].indx], cl2txt[uqr_corners[2].indx],
                                                         1+__tzcnt_u16(uqr_cand), cl2txt[indx2upd[0]], cl2txt[indx2upd[1]]);
                                                snprintf(guess_message[1], 196, "engender unique rectangle %s %s - %s for subsequent unique checking:\n        2 pair pattern: remove candidate %d at %s and %s",
                                                        ret, cl2txt[uqr_corners[0].indx], cl2txt[uqr_corners[2].indx],
                                                        1+__tzcnt_u16(other_cand), cl2txt[indx2upd[0]], cl2txt[indx2upd[1]]);
                                            }
                                            // call make_guess using a lambda
                                            grid_state = grid_state->make_guess<verbose>(
                                                        indx2upd[0],
                                                        [=] (GridState &oldgs, GridState &newgs, const char *msg[]) {
                                                            // this is the avoidance side of the UQR resolution
                                                            // the GridState to continue with:
                                                            newgs.candidates[indx2upd[0]] &= ~uqr_cand;
                                                            newgs.candidates[indx2upd[1]] &= ~uqr_cand;
                                                            msg[0] = guess_message[0];
                                                            // this is to provoke the UQR
                                                            // the GridState to back track to:
                                                            oldgs.candidates[indx2upd[0]] &= ~other_cand;
                                                            oldgs.candidates[indx2upd[1]] &= ~other_cand;
                                                            msg[1] = guess_message[1];
                                                        }, solverData.counters, solverData.output);
                                             return Phase_GuessMadeWithIncr;
                                        } else {
                                            if ( verbose == VDebug ) {

                                                snprintf(guess_message[0], 196, "to allow unique check of unique rectangle: %s %s - %s\n        2 pair pattern: remove candidate %d at %s and %s",
                                                            ret, cl2txt[uqr_corners[0].indx], cl2txt[uqr_corners[2].indx],
                                                            1+__tzcnt_u16(uqr_cand), cl2txt[indx2upd[0]], cl2txt[indx2upd[1]]);
                                                snprintf(guess_message[1], 196, "engender unique rectangle %s %s - %s for subsequent unique checking:\n        2 pair pattern: set candidate %d at %s",
                                                            ret, cl2txt[uqr_corners[0].indx], cl2txt[uqr_corners[2].indx],
                                                            1+__tzcnt_u16(uqr_cand), cl2txt[indx2upd[0]]);
                                            }
                                            // call make_guess using a lambda
                                            grid_state = grid_state->make_guess<verbose>(
                                                        indx2upd[0],
                                                        [=] (GridState &oldgs, GridState &newgs, const char *msg[]) {
                                                            // this is the avoidance side of the UQR resolution
                                                            // the GridState to continue with:
                                                            newgs.candidates[indx2upd[0]] &= ~uqr_cand;
                                                            newgs.candidates[indx2upd[1]] &= ~uqr_cand;
                                                            msg[0] = guess_message[0];
                                                            // this is to provoke the UQR
                                                            // the GridState to back track to:
                                                            oldgs.candidates[indx2upd[0]] = uqr_cand;
                                                            msg[1] = guess_message[1];
                                                        }, solverData.counters, solverData.output);
                                             return Phase_GuessMadeWithIncr;
                                        }
                                    }
                                }
                            } // for

                            // continuing with UR-P2-BLD pattern
                            unsigned char pat = uqr_pairs[pi].crnrs;
                            unsigned short non_uqr_cands = 0;
                            unsigned char non_pairs_celli[2];
                            unsigned char cl = 0;
                            for ( unsigned char i=0; i<2; i++, cl++, pat >>= 1 ) {
                                while ( (pat & 1) ) {
                                    pat >>= 1;
                                    cl++;
                                }
                                non_pairs_celli[i] = uqr_corners[cl].indx;
                                non_uqr_cands |= candidates[non_pairs_celli[i]];
                            }
                            // remove the UR candidates to avoid:
                            non_uqr_cands &= ~uqr_pairs[pi].digits;
                            unsigned char non_uqr_cands_cnt = __popcnt16(non_uqr_cands);

                            // continuing with UR-P2-BLD patterns (only for Regular rules)
                            //
                            if ( rules == Regular ) {
                                // build the intersection from:
                                // - 2 cell visibilities
                                // - candidate digits pattern
                                // - minus the 4 UR cells
                                bit128_t cand_removal_indx {};

                                bool check_set = false;
                                if ( non_uqr_cands_cnt == 1 ) {
                                    // plug the index bit hole (or prove that to be unnecessary)
                                    cand_removal_indx.u128 =  *cast2cu128(big_index_lut[non_pairs_celli[0]][All]);
                                    cand_removal_indx.u128 &= *cast2cu128(big_index_lut[non_pairs_celli[1]][All]);
                                    cand_removal_indx.u128 &= candidate_bits_by_value[__tzcnt_u16(non_uqr_cands)].u128;
                                    check_set = true;
                                } else if ( !is_diag ) {   // exclude UR-2P-D pattern
                                    unsigned short non_uqr_cands_ = non_uqr_cands;
                                    while ( non_uqr_cands_ ) {
                                        unsigned char cand_ = __tzcnt_u16(non_uqr_cands_);
                                        non_uqr_cands_ = _blsr_u32(non_uqr_cands_);
                                        cand_removal_indx.u128 |= candidate_bits_by_value[cand_].u128;
                                    }

                                    if ( (candidates[non_pairs_celli[0]] & ~non_uqr_cands) && (candidates[non_pairs_celli[1]] & ~non_uqr_cands) ) {
                                        unsigned int m = 0;
                                        __m256i a = _mm256_set1_epi16(non_uqr_cands);
                                        bool is_row = (non_pairs_celli[1] - non_pairs_celli[0])%9;

                                        if ( is_row ) {
                                            // row based
                                            unsigned short row = non_pairs_celli[0]/9;
                                            __m256i res = _mm256_cmpeq_epi16(a, _mm256_or_si256(a, *(__m256i_u*) &candidates[9*row]));
                                            m = compress_epi16_boolean<true>(res);
                                        } else {
                                            // column based
                                            unsigned short ci = non_pairs_celli[0]%9;
                                            __m256i c = _mm256_set_epi16(0,0,0,0,0,0,0,candidates[ci+72], candidates[ci+63], candidates[ci+54], candidates[ci+45], candidates[ci+36], candidates[ci+27], candidates[ci+18], candidates[ci+9], candidates[ci]);
                                            m = compress_epi16_boolean<true>(_mm256_cmpeq_epi16(a, _mm256_or_si256(a, c)));
                                        }
                                        bit128_t cand_visibility_indx {};
                                        if ( _popcnt32(m & 0x3ffff)>>1 == non_uqr_cands_cnt-1 ) {
                                            cand_visibility_indx.u128 = *cast2cu128(big_index_lut[non_pairs_celli[0]][is_row?Row:Col]);
                                            check_set = true;
                                        }

                                        // box based
                                        unsigned short b = box_start[non_pairs_celli[0]];
                                        if ( b == box_start[non_pairs_celli[1]] ) {
                                            __m256i c = _mm256_set_epi16(0,0,0,0,0,0,0,candidates[b+20], candidates[b+19], candidates[b+18], candidates[b+11], candidates[b+10], candidates[b+9], candidates[b+2], candidates[b+1], candidates[b]);
                                            m = compress_epi16_boolean<true>(_mm256_cmpeq_epi16(a, _mm256_or_si256(a, c)));
                                            if ( (_popcnt32(m & 0x3ffff)>>1) == (non_uqr_cands_cnt-1) ) {
                                                cand_visibility_indx.u128 |= *cast2cu128(big_index_lut[non_pairs_celli[0]][Box]);
                                                check_set = true;
                                            }
                                        }
                                        cand_removal_indx.u128 &= cand_visibility_indx.u128;
                                    }
                                }

                                cand_removal_indx.u128 &= ~corner_bits;

                                // double check to avoid multiple hits
                                if ( check_set ) {
                                    bit128_t cand_removal_indx_ = cand_removal_indx;
                                    while ( cand_removal_indx_ ) {
                                        unsigned char j = tzcnt_and_mask(cand_removal_indx_);

                                        // check for set members, must not remove
                                        // check also for locations where there is nothing to remove
                                        if (    (candidates[j] & ~non_uqr_cands) == 0
                                             || (candidates[j] &  non_uqr_cands) == 0 ) {
                                            cand_removal_indx.unset_indexbit(j);
                                        }
                                    }
                                }

                                // iterate over cells and remove the candidate to avoid
                                if ( check_set && cand_removal_indx != (__uint128_t)0 ) {
                                    if ( verbose != VNone ) {
                                        counters.unique_rectangles_avoided++;
                                    }
                                    found_update = true;
                                    if ( verbose == VDebug ) {
                                        char ret[32];
                                        char ret2[32];
                                        format_candidate_set(ret, uqr_pairs[pi].digits);
                                        format_candidate_set(ret2, non_uqr_cands);
                                        solverData.printf("avoiding unique rectangle: %s %s - %s\n2 pair pattern: remove candidates %s from cells outside UR at: ",
                                               ret, cl2txt[uqr_corners[0].indx], cl2txt[uqr_corners[2].indx],
                                               ret2);
                                    }
                                    bool got_single = false;
                                    const char *cma = "";
                                    while ( cand_removal_indx ) {
                                        unsigned char j = tzcnt_and_mask(cand_removal_indx);
                                        unsigned short cj = candidates[j] & ~non_uqr_cands;
                                        if ( cj ) {
                                            candidates[j] = cj;
                                            if ( (cj & (cj-1)) == 0 ) {
                                                got_single = true;
                                            }
                                            grid_state->updated.set_indexbit(j);
                                            if ( verbose == VDebug ) {
                                                solverData.printf("%s%s", cma, cl2txt[j]);
                                                cma = ",";
                                            }
                                        }
                                    }
                                    if ( verbose == VDebug ) {
                                        solverData.printf("\n");
                                    }
                                    if ( got_single ) {
                                        return Phase_Search;
                                    }
                                }
                            } else {
                                // same as above, but as a lambda guess
                                // ... just too tedious
                            }
                            break;
                        }
                        case 1:
                        {
                            // uqr_cand: the tentative value of x
                            unsigned short uqr_cand = _blsi_u32(uqr_pairs[pi].digits);
                            int j=0;
                            unsigned char crnr_indx = (__tzcnt_u16(uqr_pairs[pi].crnrs)+2)&3;
                            unsigned char indx = uqr_corners[crnr_indx].indx;
                            __uint128_t uqr_crnr_edges = *uqr_corners[crnr_indx].right_edge | *uqr_corners[(crnr_indx+3)&3].right_edge;
                            __uint128_t *cbbv;
                            unsigned short uqr_alt_cand;
                            for ( ; j<2; j++ ) {
                                uqr_alt_cand = uqr_cand ^ uqr_pairs[pi].digits;
                                // check whether there is anyting to remove:
                                if ( candidates[indx] & uqr_alt_cand ) {
                                    // all the cells containing that candidate:
                                    cbbv = &candidate_bits_by_value[__tzcnt_u16(uqr_cand)].u128;
                                    if ( ((uqr_crnr_edges & *cbbv) | corner_bits) == corner_bits ) {
                                        // found x
                                        break;
                                    }
                                }
                                // switch to the other candidate
                                uqr_cand = uqr_alt_cand;
                                if ( uqr_cand == 0 ) {  // nothing to find
                                    j=2;
                                    break;
                                }
                            }
                            if ( j<2 ) {
                                // y: uqr_alt_cand
                                // simply eliminate y from the opposite corner

                                if ( candidates[indx] & uqr_alt_cand ) {
                                    if ( verbose != VNone ) {
                                        counters.unique_rectangles_avoided++;
                                    }
                                    grid_state->updated.set_indexbit(indx);
                                    // unless under 'Regular' rules, capture the avoidable UQR
                                    // provide 'resolution' in form of a guess
                                    if ( rules == Regular ) {
                                        // simply avoid the UQR
                                        candidates[indx] &= ~uqr_alt_cand;
                                        if ( verbose == VDebug ) {
                                            char ret[32];
                                            format_candidate_set(ret, uqr_pairs[pi].digits);
                                            solverData.printf("avoiding unique rectangle: %s %s - %s\n1 pair pattern: remove candidate %d from cell %s\n",
                                                    ret, cl2txt[uqr_corners[0].indx], cl2txt[uqr_corners[2].indx],
                                                    1+__tzcnt_u16(uqr_alt_cand), cl2txt[indx]);
                                        }
                                        found_update = true;
                                        if ( (candidates[indx] & (candidates[indx]-1)) == 0 ) {
                                            e_i = indx;
                                            e_digit = candidates[indx];
                                            if ( verbose == VDebug ) {
                                                solverData.printf("naked  single      ");
                                            }
                                            trace::next_entry_reason = trace::ER_DeducedSingle;  // Phase 2: refine to fish/set/ur reason
                                            return Phase_Enter;
                                        }
                                    } else {
                                        if ( verbose == VDebug ) {
                                            char ret[32];
                                            format_candidate_set(ret, uqr_pairs[pi].digits);
                                            snprintf(guess_message[0], 196, "to allow unique check of unique rectangle: %s %s - %s\n        1 pair pattern: remove candidate %d at %s",
                                                    ret, cl2txt[uqr_corners[0].indx], cl2txt[uqr_corners[2].indx],
                                                    1+__tzcnt_u16(uqr_alt_cand), cl2txt[indx]);
                                            snprintf(guess_message[1], 196, "engender unique rectangle %s %s - %s for subsequent unique checking:\n        1 pair pattern: set %d at %s",
                                                    ret, cl2txt[uqr_corners[0].indx], cl2txt[uqr_corners[2].indx],
                                                    1+__tzcnt_u16(uqr_alt_cand), cl2txt[indx]);
                                        }
                                        // call make_guess using a lambda
                                        grid_state = grid_state->make_guess<verbose>(
                                                indx,
                                                [=] (GridState &oldgs, GridState &newgs, const char *msg[]) {
                                                    // this is the avoidance side of the UQR resolution
                                                    // the GridState to continue with:
                                                    newgs.candidates[indx] &= ~uqr_alt_cand;
                                                    msg[0] = guess_message[0];
                                                    // this is to provoke the UQR
                                                    // the GridState to back track to:
                                                    oldgs.candidates[indx] = uqr_alt_cand;
                                                    msg[1] = guess_message[1];
                                                }, solverData.counters, solverData.output);
                                        return Phase_GuessMadeWithIncr;
                                    }
                                }
                            }
                            break;
                        } // case 1
                        } // switch
                    } // for pairs
                } // for 6 uqrs

                // finally rotate positions for next group of 9 UQRs
                // for line 0 and 2 rotate first and third group clockwise by 1,
                // for line 1 and 3 rotate second and fourth group clockwise by 1
                linev[0] = _mm256_shuffle_epi8(linev[0], linerotate[0]);
                linev[1] = _mm256_shuffle_epi8(linev[1], linerotate[1]);
                linev[2] = _mm256_shuffle_epi8(linev[2], linerotate[0]);
                linev[3] = _mm256_shuffle_epi8(linev[3], linerotate[1]);
            }
        }
        if ( found_update ) {
            last_band_uqr = (band+1)%6;
            return Phase_Search;
        };
    }
}
#endif
    return Phase_HiddenSearch;
}
