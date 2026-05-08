// Out-of-class definition of `SolveCtx<verbose>::phase_hidden_search()`.
// AlphaEvolve mutation unit: this file is the entire surface for hidden singles + triads + naked sets + fishes + UQR
// strategy; replace the function body to mutate the strategy without
// touching the rest of the solver.
//
// CONTRACT: private include fragment, must be #included exactly once
// from inside `namespace Schoku { ... }` after solver/solve_ctx.hpp.
#pragma once


template <Verbosity verbose>
__attribute__((always_inline)) inline SolverPhase SolveCtx<verbose>::phase_hidden_search() {
    // reset solverData
    solverData.bivaluesValid = false;
    solverData.cbbvsValid = false;
    solverData.sectionSetUnlockedValid[0] = solverData.sectionSetUnlockedValid[1] = 
                                            solverData.sectionSetUnlockedValid[2] = false;
    solverData.guess_hint_digit = 0;
    solverData.boxes_as_cols = 0;

    // Algorithm 2 - Find hidden singles
    // For all sections (ie. rows/columns/boxes):
    //   for each cell C in the given section
    //      'or' all the other cells of the section
    //          check whether C contains a candidate that does not occur in the other cells
    // Back track checking:
    //    1. If the found cell C contains more than one candidate that does not
    //       appear in the other cells: back track
    //    2. If the 'or' of all the cells does not contain all digits: back track
    //
    // columns are or'ed together (i.e. all columns in parallel, while reading one row at a time,
    // leaving out the current row, and hidden column singles are thus isolated for that row.
    // For efficiency, precompute and save the or'ed rows (tails) and
    // preserve the last leading set of rows (the head).
    // To check for an invalid state of the puzzle:
    // - check the columns or'ed value to be 0x1ff.
    // - andnot with 0x1ff and then check all non-zero results to be singletons, or otherwise
    //   back track.
    //
    // Combine 8 cells from 2 rows into one __m256i vector. (9th row: use __m128i vector)
    // Rotate and or until each vector element represents 7 cells or'ed (except the cell
    // directly corresponding to its position, containing the hidden single if there is one).
    // Broadcast the nineth cell and or it for good measure, then andnot with 0x1ff
    // to isolate the hidden singles.
    // For the nineth cell, rotate and or one last time and use just one element of the result
    // to check the nineth cell for a hidden single.  These are grouped together and then checked.
    //
    // For boxes, loading of the data is the most expensive.
    // Either the 'column' approach or the 'row' approach will work.
    // However, box hidden single search can just be eliminated, an the triad processing
    // will just eliminate all column/row-based occurrances of the single outside of the box.
    // This way, the expensive loading of the boxes is eliminated.
    //
    // All checks use compression to a bit vector
    //
    // Algorithm 3
    // Definition: An intersection of row/box and col/box consisting of three cells
    // are called "triad" in the following.
    // The technique described in the following is also known as 
    // "locked candidates (claiming/pointing)".
    // There are 27 horizontal (row-based) triads and 27 vertical (col-based) triads.
    // For each band of three aligned boxes there are nine triads.
    //
    // Triads in their own right are significant for two reasons:
    // First, a triad that has 3 candidates is a special case of set that is easily detected.
    // The detection occurs in part 2 of Algorithm 3 by running a popcount on the collected
    // triad candidates.  The result is kept in form of a bitvector of 'unlocked triads'
    // for rows and columns.  For columns, the unlocked state is determined directly from
    // the general unlocked bit vector.  For rows, the order in which the row triads are
    // stored is not compatible with the general unlocked bit vector.  Instead, all locked
    // row triads are individually removed from their row unlocked bit vector.
    //
    // The collection of the triads data (Algorithm 3 Part 1) is intermingled with
    // that of algorithm 2 for speed.
    // Part 2 of Algorithm 3 allows to detect all fully resolved triads (sets).
    // Note that this is not normally part of the locked set solution strategy.
    // Part 3 of Algorithm 3 allows to determine which candidate value can only occur
    // in a specific triad and not in the other triads of the row/column and box.
    // Terminology:
    // - a "must" is a set of candidates for a triad containing those candidates that cannot occur
    // in any other triad of the same block or row/column.
    // - a "must not" (mustnt for short) is a set of candidates that must not occur in the given triad.
    // For starters, the complement of the candidates that occur in the triad cells is a
    // "must not" (not necessarily the most complete one).
    // Trivially, in a solved Sudoku puzzle, each "must" has three candidates and all "must not"
    // have six candidates.
    // - the "peer" triads of a given triad share either box or the row/column with the given "triad".
    // The following holds since each row/column or box has a set of candidates and the
    // triads are intersections of a given row/column with the box:
    // - given a set of three 'peered' triads for each row/column and box each triad is
    // peered with two triads of  the same row and with two triads of the same box,
    // - If a given candidate value does not occur in either of the peer pairs (peer-mustnt),
    //   this triad must contain that candidate value (must).
    // - Any candidate only occurring in a a triad (must) because if doesn't occur in
    //   either its box peers or its row/column peers cannot occur in any of the pair of peers.
    // - Since 3 is the upper limit of must candidates, once these three are known from
    //   part 2 of the algorithm, these become the fixed 'must' set for the triad.
    //   The triad is then called "resolved".
    //
    // We can start with the mustnt for all triads of a band of boxes, compute the intersection
    // pairs of peer-mustnt (horizontally and vertically) and join the two results.
    // This is then a "must" of the given triad, and augmented by the must if the triad is resolved.
    // Using the peers musts, their union is a "must not" for the given triad.
    // By joining it to the orginal "must not" we obtain an equal or super set for the new "must not".
    // This "must not" in turn can then be be applied to remove the excessive candidates from
    // the triad.
    //
    // These computations are easily implemented as SIMD operations.
    //
    // To recap: parts 1 - 3 of Algorithm 3 are fast and effective - especially if the
    // hidden single search is leveraged for the initial union of the triads' candidates.
    //
    // Part 2 and 3 are invoked after Algorithm 2 fails to detect any hidden single.
    //
    // col_triads are numbered continuously per horizontal band (one vertical triad for each column)
    // and stored in three arrays of 10 ushort as follows:
    //    0,  1,  2,  3,  4,  5,  6,  7,  8, -
    //    9, 10, 11, 12, 13, 14, 15, 16, 17, -
    //   18, 19, 20, 21, 22, 23, 24, 25, 26
    //  
    // row_triads are canonically naturally numbered 0, 1, 2, for the first row, 3, 4, 5 for the second
    // and so on.  These are stored in three arrays of 10 ushort, captured 3 per row,
    // 3 rows stacked vertically with the subsequent 3 rows offset by 3:
    //    0,  1,  2,  9, 10, 11, 18, 19, 20, -
    //    3,  4,  5, 12, 13, 14, 21, 22, 23, -
    //    6,  7,  8, 15, 16, 17, 24, 25, 26
    // SIMD Triad processing requires all triads of a given band to be positioned
    // 3 horizontal triads side by side and 3 vertical triads in the same positions
    // of the two other rows.  The ordering described above satisfies this requirement.
    //
    // For technical reasons, an (unused) 10th triad is injected between each set of 9 triads
    // for a total of 29 triad values.
    //
    // various methods are used to efficiently save the triads in the processing order,
    // which go beyond the end of the arrays normally needed.

    TriadInfo &triad_info = solverData.triadInfo;

    // Algo 2 and Algo 3.1
    {
        __m256i column_or_tails[9];
        __m256i column_or_head = _mm256_setzero_si256();
        __m256i column_cand_or = _mm256_setzero_si256();
        __m256i col_triads_1, col_triads_2, col_triads_3;
        {
            // columns
            // to start, we simply tally the or'ed rows
            signed char j = 81-9;
            // compute fresh col_triads_3
            col_triads_3 = _mm256_setzero_si256();
            // A2 (cols)
            // precompute 'tails' of the or'ed column_cand_or only once
            // working backwords
            // 3 iterations, rows 8, 7 and 6.
            for ( ; j >= 54; j -= 9) {
                column_or_tails[j/9-1] = col_triads_3;
                col_triads_3 = _mm256_or_si256(col_triads_3, *(__m256i_u*) &candidates[j]);
            }
            // col_triads_3 now contains the third set of column-triads
            // compute fresh col_triads_2
            // A2 and A3.1.a
            // 3 iterations, rows 5, 4 and 3.
            col_triads_2 = _mm256_setzero_si256();
            for ( ; j >= 27; j -= 9) {
                column_or_tails[j/9-1] = _mm256_or_si256(col_triads_2,col_triads_3);
                col_triads_2 = _mm256_or_si256(col_triads_2, *(__m256i_u*) &candidates[j]);
            }
            // A2 and A3.1.a
            // col_triads_2/3 contain the second/third set of column-triads
            column_cand_or = _mm256_or_si256(col_triads_2, col_triads_3);

            // 2 iterations, rows 1 and 2.
            // the first set of column triads is computed below as part of the computed
            // 'head' or.
            for ( ; j > 0; j -= 9) {
                column_or_tails[j/9-1] = column_cand_or;
                column_cand_or = _mm256_or_si256(column_cand_or, *(__m256i_u*) &candidates[j]);
            }
            // or in row 0 and check whether all digits or covered
            if ( check_back && !_mm256_testz_si256(mask9x1ff,_mm256_andnot_si256(_mm256_or_si256(column_cand_or, *(__m256i*) &candidates[0]), mask9x1ff)) ) {
                // the current grid has no solution, go back
                if ( verbose != VNone ) {
                    __m256i missing = _mm256_andnot_si256(_mm256_or_si256(column_cand_or, *(__m256i*) &candidates[0]), mask9x1ff);
                    unsigned int m  = _mm256_movemask_epi8(_mm256_cmpgt_epi16(missing, _mm256_setzero_si256()));
                    int idx = __tzcnt_u32(m)>>1;
                    unsigned short digit = ((v16us)missing)[idx];
                    if ( grid_state->stackpointer == 0 && unique_check_mode == 0 ) {
                        if ( warnings != 0 ) {
                            solverData.printf("Line %d: stack 0, back track - column %d misses digit %d\n", line, idx, __tzcnt_u16(digit)+1);
                        }
                    } else if ( debug ) {
                        solverData.printf("back track - missing digit %d in column %d\n", __tzcnt_u16(digit)+1, idx);
                    }
                }
                return Phase_Back;
            }

            // breaking the column hidden singles out of the loop this way will win some performance
            unsigned int jrow = 0;
            for (unsigned int j = 0; j < 81; j+=9, jrow++) {
                // turn the or'ed rows into a mask for the singletons, if any.
                // check col (9) candidates
                unsigned short m = (j < 64) ? (unlocked[0] >> j) : (unlocked[1] >> (j-64));
                if ( j == 63) {
                    m |= unlocked[1] << (64-63);
                }
                __m256i a = _mm256_cmpgt_epi16(mask9x1ff, column_cand_or);
                unsigned int mask = and_compress_masks<false>(a, m & 0x1ff);
                if ( mask) {
                    int idx = __tzcnt_u32(mask);
                    e_i = j+idx;
                    e_digit = ((v16us)_mm256_andnot_si256(column_cand_or, mask9x1ff))[idx];
                    if ( check_back && (e_digit & (e_digit-1)) ) {
                        if ( verbose != VNone ) {
                            if ( grid_state->stackpointer == 0 && unique_check_mode == 0 ) {
                                if ( warnings != 0 ) {
                                    solverData.printf("Line %d: stack 0, back track - col cell %s does contain multiple hidden singles\n", line, cl2txt[e_i]);
                                }
                            } else if ( debug ) {
                                solverData.printf("back track - multiple hidden singles in col cell %s\n", cl2txt[e_i]);
                            }
                        }
                        return Phase_Back;
                    }
                    if ( verbose == VDebug ) {
                        solverData.printf("hidden single (col)");
                    }
                    return Phase_Enter;
                }

                // leverage previously computed or'ed rows in head and tails.
                column_or_head = column_cand_or = _mm256_or_si256(*(__m256i_u*) &candidates[j], column_or_head);
                column_cand_or = _mm256_or_si256(column_or_tails[jrow],column_cand_or);

                if ( j == 18 ) {
                    // A3.1.c
                    col_triads_1 = column_or_head;
                }
            }
        }

        // rows
        //
        // The grid is processed as indicated by the keys below:
        // P1 P1 P1 P1 P1 P1 P1 P1 C9
        // P1 P1 P1 P1 P1 P1 P1 P1 C9
        // P2 P2 P2 P2 P2 P2 P2 P2 C9
        // P2 P2 P2 P2 P2 P2 P2 P2 C9
        // P3 P3 P3 P3 P2 P3 P3 P3 C9
        // P3 P3 P3 P3 P4 P3 P3 P3 C9
        // P4 P4 P4 P4 P4 P4 P4 P4 C9
        // P4 P4 P4 P4 P4 P4 P4 P4 C9
        // R9 R9 R9 R9 R9 R9 R9 R9 C9

        // Px: processed in pairs of rows, for their first 8 cells
        // C9: the 9th cell of each of the rows is prepared
        //     and stored in row_9th_cand_vert.  row_9th_cand_vert is then processed at the end.
        // R9: is processed for 8 cells

        // The examination of each row and its cells is made up of three steps:
        // 1 - or the other values of the row
        // 2 - check that the row or box contains all digits
        // 3 - check negated or'ed value for non-zero candidates.
        //     if there are multiple candidates, back track,
        //     enter any valid singletons found

        // row_9th_cand_vert cumulates the 9th row or'ed singleton candidates
        // which are processed in the end.
#if USE_ROW_HIDDEN_SEARCH
        v16us row_9th_cand_vert;
#endif
        unsigned char irow = 0;

        // find hidden singles in rows

        for (unsigned char i = 0; i < 72; i += 18, irow+=2) {
            // rows, in pairs

            __m256i c1 = _mm256_set_m128i(*(__m128i_u*) &candidates[i+9],
                                          *(__m128i_u*) &candidates[i]);

            unsigned short the9thcand_row[2] = { candidates[i+8], candidates[i+17] };

            __m256i row_or7 = _mm256_setzero_si256();
#if USE_ROW_HIDDEN_SEARCH
            __m256i row_or8;
#endif
            __m256i row_9th = _mm256_set_m128i(_mm_set1_epi16(the9thcand_row[1]),_mm_set1_epi16(the9thcand_row[0]));

            __m256i row_triad_capture[2];
            {
                __m256i c1_ = c1;
                // A2 and A3.1.b
                // first lane for the row, 2nd lane for the box
                // step j=0
                // rotate left (0 1 2 3 4 5 6 7) -> (1 2 3 4 5 6 7 0)
                c1_ = _mm256_alignr_epi8(c1_, c1_, 2);
                row_or7 = _mm256_or_si256(c1_, row_or7);
                // step j=1
                // rotate (1 2 3 4 5 6 7 0) -> (2 3 4 5 6 7 0 1)
                c1_ = _mm256_alignr_epi8(c1_, c1_, 2);
                row_or7 = _mm256_or_si256(c1_, row_or7);
                // triad capture: after 2 rounds, row triad 3 of this row saved in pos 5
                row_triad_capture[0] = _mm256_or_si256(row_or7, row_9th);
                // step j=2
                // rotate (2 3 4 5 6 7 0 1) -> (3 4 5 6 7 0 1 2)
                c1_ = _mm256_alignr_epi8(c1_, c1_, 2);
                // triad capture: after 3 rounds 2 row triads 0 and 1 in pos 7 and 2
                row_triad_capture[1] = row_or7 = _mm256_or_si256(c1_, row_or7);

#if USE_ROW_HIDDEN_SEARCH
                // continue the rotate/or routine for this row
                for (unsigned char j = 3; j < 7; ++j) {
                    // rotate (0 1 2 3 4 5 6 7) -> (1 2 3 4 5 6 7 0)
                    // c1_ holds 2 lanes for two rows
                    c1_ = _mm256_alignr_epi8(c1_, c1_, 2);
                    row_or7 = _mm256_or_si256(c1_, row_or7);
                }

                row_or8 = _mm256_or_si256(c1, row_or7);
                // test row_or8 to hold all the digits
                if ( check_back ) {
                    if ( !_mm256_testz_si256(mask1ff,_mm256_andnot_si256(_mm256_or_si256(row_9th, row_or8), mask1ff))) {
                        // the current grid has no solution, go back
                        if ( verbose != VNone ) {
                            __m256i missing = _mm256_andnot_si256(_mm256_or_si256(row_9th, row_or8), mask1ff);
                            unsigned int m = _mm256_movemask_epi8(_mm256_cmpeq_epi16(_mm256_setzero_si256(),
                                                                  missing));
                            unsigned short digit = ((v16us)missing)[(m & 0xffff)?8:0];

                            if ( grid_state->stackpointer == 0 && unique_check_mode == 0 ) {
                                if ( warnings != 0 ) {
                                    solverData.printf("Line %d: stack 0, back track - row %d does not contain digit %d\n", line, irow+(m & 0xffff)?0:1, __tzcnt_u16(digit)+1);
                                }
                            } else if ( debug ) {
                                solverData.printf("back track - missing digit %d in row %d\n", __tzcnt_u16(digit)+1, irow+((m & 0xffff)?1:0)); // XXX
                            }
                        }
                        return Phase_Back;
                    }
                }
            }
            // hidden singles in row
            __m256i row_mask = _mm256_andnot_si256(_mm256_or_si256(row_9th, row_or7), mask1ff);
            {
                // check row (8) candidates
                unsigned short m1 = grid_state->unlocked.get_indexbits(i, 8) | (grid_state->unlocked.get_indexbits(i+9, 8)<<8);
                __m256i a1 = _mm256_cmpgt_epi16(row_mask, _mm256_setzero_si256());

                unsigned int mask1 = and_compress_masks<false>(a1, m1);
                if (mask1) {
                    int idx = __tzcnt_u32(mask1);
                    int idx_ = idx + ((mask1 & 0xff)?0:1);
                    e_i = i + idx_;
                    e_digit = ((v16us)row_mask)[idx];
                    if ( __popcnt16(e_digit) == 1 ) {
                        if ( verbose == VDebug ) {
                            solverData.printf("hidden single (row)");
                        }
                        return Phase_Enter;
                    }
                    if ( grid_state->stackpointer == 0 && unique_check_mode == 0 ) {
                        if ( warnings != 0 ) {
                            solverData.printf("Line %d: stack 0, back track - %s cell %s does contain multiple hidden singles\n", line, mask1?"row":"box", cl2txt[e_i]);
                        }
                    } else if ( debug ) {
                        solverData.printf("back track - multiple hidden singles in row cell %s\n", cl2txt[e_i]);
                    }
                    e_digit = 0;
                    return Phase_Back;
                }
#endif
            }
            // deal with saved row_triad_capture:
            // - captured triad 3 of this row saved in pos 5 of row_triad_capture[0]
            // - captured triads 0 and 1 in pos 7 and 2 of row_triad_capture[1]
            // Blend the two captured vectors to contain all three triads in pos 7, 2 and 5
            // shuffle the triads from pos 7,2,5 into pos 0,1,2
            // spending 3 instructions on this: blend, shuffle, storeu
            // a 'random' 4th unsigned short is overwritten by the next triad store
            // (or is written into the gap 10th slot).
            row_triad_capture[0] = _mm256_shuffle_epi8(_mm256_blend_epi16(row_triad_capture[1], row_triad_capture[0], 0x20),shuff725to012);
            _mm_storeu_si64(&triad_info.row_triads[row_triads_lut[irow]], _mm256_castsi256_si128(row_triad_capture[0]));
            _mm_storeu_si64(&triad_info.row_triads[row_triads_lut[irow+1]], _mm256_extracti128_si256(row_triad_capture[0],1));

#if USE_ROW_HIDDEN_SEARCH
            row_9th_cand_vert[irow]     = ~((v16us)row_or8)[0] & the9thcand_row[0];
            row_9th_cand_vert[irow+1]   = ~((v16us)row_or8)[8] & the9thcand_row[1];
#endif
        }   // for row Px

        // 9th row
        {
            __m128i c = *(__m128i_u*) &candidates[72];

            unsigned short the9thcand_row = candidates[80];

            __m128i row_or7 = _mm_setzero_si128();
#if USE_ROW_HIDDEN_SEARCH
            __m128i row_or8;
#endif
            __m128i row_9th_elem = _mm_set1_epi16(the9thcand_row);

            __m128i row_triad_capture[2];
            {
                __m128i c_ = c;
                // A2 and A3.1.b
                // first lane for the row, 2nd lane for the box
                // step j=0
                // rotate left (0 1 2 3 4 5 6 7) -> (1 2 3 4 5 6 7 0)
                c_ = _mm_alignr_epi8(c_, c_, 2);
                row_or7 = _mm_or_si128(c_, row_or7);
                // step j=1
                // rotate (1 2 3 4 5 6 7 0) -> (2 3 4 5 6 7 0 1)
                c_ = _mm_alignr_epi8(c_, c_, 2);
                row_or7 = _mm_or_si128(c_, row_or7);
                // triad capture: after 2 rounds, row triad 3 of this row saved in pos 5
                row_triad_capture[0] = _mm_or_si128(row_or7, row_9th_elem);
                // step j=2
                // rotate (2 3 4 5 6 7 0 1) -> (3 4 5 6 7 0 1 2)
                c_ = _mm_alignr_epi8(c_, c_, 2);
                // triad capture: after 3 rounds 2 row triads 0 and 1 in pos 7 and 2
                row_triad_capture[1] = row_or7 = _mm_or_si128(c_, row_or7);
#if USE_ROW_HIDDEN_SEARCH
                // continue the rotate/or routine for this row
                for (unsigned char j = 3; j < 7; ++j) {
                    // rotate (0 1 2 3 4 5 6 7) -> (1 2 3 4 5 6 7 0)
                    c_ = _mm_alignr_epi8(c_, c_, 2);
                    row_or7 = _mm_or_si128(c_, row_or7);
                }

                row_or8 = _mm_or_si128(c, row_or7);
                // test row_or8 | row_9th_elem to hold all the digits
                if ( check_back ) {
                    if ( !_mm_testz_si128(_mm256_castsi256_si128(mask1ff),_mm_andnot_si128(_mm_or_si128(row_9th_elem, row_or8), _mm256_castsi256_si128(mask1ff)))) {
                        // the current grid has no solution, go back
                        if ( verbose != VNone ) {
                            __m128i missing = _mm_andnot_si128(_mm_or_si128(row_9th_elem, row_or8), _mm256_castsi256_si128(mask1ff));
                            unsigned short digit = ((v8us)missing)[0];
                            if ( grid_state->stackpointer == 0 && unique_check_mode == 0 ) {
                                if ( warnings != 0 ) {
                                    solverData.printf("Line %d: stack 0, back track - row %d misses digit %d\n", line, 8, __tzcnt_u16(digit)+1);
                                }
                            } else if ( debug ) {
                                solverData.printf("back track - missing digit %d in row %d\n", __tzcnt_u16(digit)+1, 8);
                            }
                        }
                        return Phase_Back;
                    }
                }
            }

            // hidden singles in row 9
            __m128i row_mask = _mm_andnot_si128(_mm_or_si128(row_9th_elem, row_or7), _mm256_castsi256_si128(mask1ff));
            {
                // check row (8) candidates

                __m128i a = _mm_cmpgt_epi16(row_mask, _mm_setzero_si128());
                unsigned short mask = compress_epi16_boolean128<false>(a) & grid_state->unlocked.get_indexbits(72, 8);
                if (mask) {
                    int s_idx = __tzcnt_u32(mask);
                    e_i = 72 + s_idx;
                    e_digit = ((v8us)row_mask)[s_idx];
                    // Check that the single is indeed a single
                    if ( __popcnt16(e_digit) == 1 ) {
                        if ( verbose == VDebug ) {
                            solverData.printf("hidden single (row)");
                        }
                        return Phase_Enter;
                    } else {
                        if ( verbose != VNone ) {
                            if ( grid_state->stackpointer == 0 && unique_check_mode == 0 ) {
                                if ( warnings != 0 ) {
                                    solverData.printf("Line %d: stack 0, back track - row cell %s does contain multiple hidden singles\n", line, cl2txt[e_i]);
                                }
                            } else if ( debug ) {
                                solverData.printf("back track - multiple hidden singles in row cell %s\n", cl2txt[e_i]);
                            }
                        }
                        e_digit = 0;
                        return Phase_Back;
                    }
                }
#endif
            } // row
            // deal with saved row_triad_capture:
            // - captured triad 3 of this row saved in pos 5 of row_triad_capture[0]
            // - captured triads 0 and 1 in pos 7 and 2 of row_triad_capture[1]
            // Blend the two captured vectors to contain all three triads in pos 7, 2 and 5
            // shuffle the triads from pos 7,2,5 into pos 0,1,2
            // spending 3 instructions on this: blend, shuffle, storeu
            // a 'random' 4th unsigned short is overwritten by the next triad store
            // (or is written into the gap 10th slot).
            row_triad_capture[0] = _mm_shuffle_epi8(_mm_blend_epi16(row_triad_capture[1], row_triad_capture[0], 0x20),_mm256_castsi256_si128(shuff725to012));
            _mm_storeu_si64(&triad_info.row_triads[row_triads_lut[8]], row_triad_capture[0]);

#if USE_ROW_HIDDEN_SEARCH
            row_9th_cand_vert[8]     = ~((v8us)row_or8)[0] & the9thcand_row;
#endif
        } // row R9
#if USE_ROW_HIDDEN_SEARCH
        // check saved row singleton candidates (9th column)

        unsigned int mask = _mm256_movemask_epi8(_mm256_cmpgt_epi16((__m256i)row_9th_cand_vert, _mm256_setzero_si256())) & 0x3ffff;
        while (mask) {
            int s_idx = __tzcnt_u32(mask) >> 1;
            unsigned char celli = s_idx*9+8;
            unsigned short cand = row_9th_cand_vert[s_idx];
            if ( ((bit128_t*)unlocked)->check_indexbit(celli) ) {
                // check for a single.
                // This is rare as it can only occur when a wrong guess was made.
                // the current grid has no solution, go back
                if ( check_back && cand & (cand-1) ) {
                    if ( verbose != VNone ) {
                        if ( grid_state->stackpointer == 0 && unique_check_mode == 0 ) {
                            if ( warnings != 0 ) {
                                solverData.printf("Line %d: stack 0, multiple hidden singles in row cell %s\n", line, cl2txt[celli]);
                            }
                        } else if ( debug ) {
                            solverData.printf("back track - multiple hidden singles in row cell %s\n", cl2txt[celli]);
                        }
                    }
                    return Phase_Back;
                }
                if ( verbose == VDebug ) {
                    solverData.printf("hidden single (row)");
                }
                e_i = celli;
                e_digit = cand;
                return Phase_Enter;
            }
            mask &= ~(3<<(s_idx<<1));
        }
#endif
        // Store all column triads sequentially, paying attention to overlap.
        //
        // A3.1.c
        _mm256_storeu_si256((__m256i *)triad_info.col_triads, col_triads_1);
        _mm256_storeu_si256((__m256i *)(triad_info.col_triads+10), col_triads_2);
        // mask 5 hi triads to 0xffff, which will not trigger any checks
        _mm256_storeu_si256((__m256i *)(triad_info.col_triads+20), _mm256_or_si256(col_triads_3, mask11hi));

    } // Algo 2 and Algo 3.1

    { // Algo 3.2
        // A3.2.1 (check col-triads)
        // just a plain old popcount, but using the knowledge that the upper byte is always 1 or 0.

        __m256i v1 = _mm256_loadu_si256((__m256i *)triad_info.col_triads);
        __m256i v2 = _mm256_loadu_si256((__m256i *)(triad_info.col_triads+16));
        __m256i v9 = _mm256_packus_epi16(_mm256_srli_epi16(v1,8), _mm256_srli_epi16(v2,8));
        v1 = _mm256_packus_epi16(_mm256_and_si256(v1, maskff), _mm256_and_si256(v2, maskff));
        __m256i lo1 = _mm256_and_si256 (v1, nibble_mask);
        __m256i hi1 = _mm256_and_si256 (_mm256_srli_epi16 (v1, 4), nibble_mask );
        __m256i cnt11 = _mm256_shuffle_epi8 (lookup, lo1);
        cnt11 = _mm256_add_epi8(_mm256_add_epi8 (cnt11, _mm256_shuffle_epi8 (lookup, hi1)), v9);
        __m256i res = _mm256_and_si256(_mm256_permute4x64_epi64(cnt11, 0xD8), mask27);

        // do this only if unlocked has been updated:
        if ( last_entered_count_col_triads != current_entered_count) {
            // the high byte is set to the stackpointer, so that this works
            // across guess/backtrack
            last_entered_count_col_triads = current_entered_count;
            unsigned long long tr = unlocked[0];  // col triads - set if any triad cell is unlocked
            tr |= (tr >> 9) | (tr >> 18) | (unlocked[1]<<(64-9)) | (unlocked[1]<<(64-18));
            // mimick the pattern of col_triads, i.e. a gap of 1 after each group of 9.
            if ( pext_support ) {
                tr = _pext_u64(tr, 0x1ffLL | (0x1ffLL<<27) | (0x1ffLL<<54));
                grid_state->triads_unlocked[Col] &= _pdep_u64(tr, (0x1ff)|(0x1ff<<10)|(0x1ff<<20));
            } else {
                grid_state->triads_unlocked[Col] &= (tr & 0x1ff) | ((tr >> 17) & (0x1ff<<10)) | ((tr >> 34) & (0x1ff<<20));
            }
        }

        triad_info.triads_selection[Col] = _mm256_movemask_epi8(_mm256_cmpeq_epi8(res, fours));
        unsigned long long m = _mm256_movemask_epi8(_mm256_cmpeq_epi8(res, threes)) & grid_state->triads_unlocked[Col];
        if ( verbose != VNone ) {
            counters.triads_resolved += _popcnt64(m);
        }
        while (m) {
            unsigned char tidx = tzcnt_and_mask(m);
            unsigned short cands_triad = triad_info.col_triads[tidx];
            if ( verbose == VDebug ) {
                char ret[32];
                format_candidate_set(ret, cands_triad);
                solverData.printf("triad set (col): %-9s %s\n", ret, cl2txt[tidx/10*3*9+tidx%10]);
            }
            // mask off resolved triad:
            _bittestandreset((int*)&grid_state->triads_unlocked[Col], tidx);
            unsigned char off = tidx%10+tidx/10*27;
            grid_state->set23_found[Col].set_indexbits(0x40201,off,19);
            grid_state->set23_found[Box].set_indexbits(0x40201,off,19);
        }

        // A3.2.2 (check row-triads)

        // just a plain old popcount, but using the knowledge that the upper byte is always 1 or 0.
        v1 = _mm256_loadu_si256((__m256i *)triad_info.row_triads);
        v2 = _mm256_loadu_si256((__m256i *)(triad_info.row_triads+16));
        v9 = _mm256_packus_epi16(_mm256_srli_epi16(v1,8), _mm256_srli_epi16(v2,8));
        v1 = _mm256_packus_epi16(_mm256_and_si256(v1, maskff), _mm256_and_si256(v2, maskff));
        lo1 = _mm256_and_si256 (v1, nibble_mask);
        hi1 = _mm256_and_si256 (_mm256_srli_epi16 (v1, 4), nibble_mask );
        cnt11 = _mm256_shuffle_epi8 (lookup, lo1);
        cnt11 = _mm256_add_epi8(_mm256_add_epi8 (cnt11, _mm256_shuffle_epi8 (lookup, hi1)), v9);
        res = _mm256_and_si256(_mm256_permute4x64_epi64(cnt11, 0xD8), mask27);

        m = _mm256_movemask_epi8(_mm256_cmpeq_epi8(res, threes)) & grid_state->triads_unlocked[Row];
        triad_info.triads_selection[Row] = _mm256_movemask_epi8(_mm256_cmpeq_epi8(res, fours));

        // the best that can be done for rows - remember that the order of
        // row triads is not aligned with the order of cells.
        bit128_t tr = grid_state->unlocked;   // for row triads - any triad cell unlocked

        // to allow checking of unresolved triads
        tr.u64[0]  |= (tr.u64[0] >> 1)  | (tr.u64[0] >> 2);
        tr.u64[0]  |= ((tr.u64[1] & 1)  | ((tr.u64[1] & 2) >> 1))<<63;
        tr.u64[1]  |= (tr.u64[1] >> 1)  | (tr.u64[1] >> 2);

        while (m) {
            unsigned char tidx = tzcnt_and_mask(m);
            unsigned char off  = row_triad_index_to_offset[tidx];

            if ( !tr.check_indexbit(off)) {  // locked
                grid_state->triads_unlocked[Row] &= ~(1<<tidx);
                continue;
            }

            if ( verbose != VNone ) {
                counters.triads_resolved++;
            }
            if ( verbose == VDebug ) {
                char ret[32];
                format_candidate_set(ret, triad_info.row_triads[tidx]);
                solverData.printf("triad set (row): %-9s %s\n", ret, cl2txt[off]);
            }

            // mask off resolved triad:
            grid_state->triads_unlocked[Row] &= ~(1LL << tidx);
            grid_state->set23_found[Row].set_indexbits(0x7,off,3);
            grid_state->set23_found[Box].set_indexbits(0x7,off,3);
        } // while
    }   // Algo 3 Part 2

    {   // Algo 3 Part 3
        // Note on nomenclature:
        // - must / mustnt = candidates that must or must not occur in the triad.
        // - t vs p prefix: t for triad (only applies to the triad), p for peers, i.e. what
        //   the peers impose onto the triad.
        //
        // The SIMD parallelism consists of computing tmustnt for three bands in parallel.
        //
        bool any_changes = false;

        // mask for each line of 9 triad *must/*mustnt.
        // all t/pmust* variables are populated in groups of three, the first two in low 12 bytes (0-11),
        // the third group in bytes 16-21.

        for (int type=1; type>=0; type--) {	// row = 0, col = 1

            // input
            unsigned short *triads   = type==0?triad_info.row_triads:triad_info.col_triads;
            unsigned short *wo_musts = type==0?triad_info.row_triads_wo_musts:triad_info.col_triads_wo_musts;
            unsigned short *ptriads  = triads;
            __m256i pmustnt[2][3] = {mask_musts, mask_musts, mask_musts, mask_musts, mask_musts, mask_musts};
            __m256i tmustnt[3];

            // first load triad candidates and compute peer based pmustnt

            // i=0 (manually unrolled loop)
                // tmustnt computed from all candidates in row/col_triads
                __m256i tmustnti = _mm256_andnot_si256(_mm256_loadu2_m128i((__m128i*)&ptriads[6], (__m128i*)ptriads), mask_musts);
                tmustnt[0] = tmustnti;
                // compute peer-based pmustnt
                // vertical peers
                pmustnt[0][1] = _mm256_and_si256(pmustnt[0][1], tmustnti);
                pmustnt[0][2] = _mm256_and_si256(pmustnt[0][2], tmustnti);

                // horizontal peers
                tmustnti = _mm256_shuffle_epi8(tmustnti, rot_hpeers);
                pmustnt[1][0] = _mm256_and_si256(pmustnt[1][0], tmustnti);
                tmustnti = _mm256_shuffle_epi8(tmustnti, rot_hpeers);
                pmustnt[1][0] = _mm256_and_si256(pmustnt[1][0], tmustnti);
            // i=1
                ptriads += 10;
                tmustnti = _mm256_andnot_si256(_mm256_loadu2_m128i((__m128i*)&ptriads[6], (__m128i*)ptriads), mask_musts);
                tmustnt[1] = tmustnti;
                // vertical peers
                pmustnt[0][0] = _mm256_and_si256(pmustnt[0][0], tmustnti);
                pmustnt[0][2] = _mm256_and_si256(pmustnt[0][2], tmustnti);

                // horizontal peers
                tmustnti = _mm256_shuffle_epi8(tmustnti, rot_hpeers);
                pmustnt[1][1] = _mm256_and_si256(pmustnt[1][1], tmustnti);
                tmustnti = _mm256_shuffle_epi8(tmustnti, rot_hpeers);
                pmustnt[1][1] = _mm256_and_si256(pmustnt[1][1], tmustnti);
            // i=2
                ptriads += 10;
                tmustnti = _mm256_andnot_si256(_mm256_loadu2_m128i((__m128i*)&ptriads[6], (__m128i*)ptriads), mask_musts);
                tmustnt[2] = tmustnti;
                // vertical peers
                pmustnt[0][0] = _mm256_and_si256(pmustnt[0][0], tmustnti);
                pmustnt[0][1] = _mm256_and_si256(pmustnt[0][1], tmustnti);

                // horizontal peers
                tmustnti = _mm256_shuffle_epi8(tmustnti, rot_hpeers);
                pmustnt[1][2] = _mm256_and_si256(pmustnt[1][2], tmustnti);
                tmustnti = _mm256_shuffle_epi8(tmustnti, rot_hpeers);
                pmustnt[1][2] = _mm256_and_si256(pmustnt[1][2], tmustnti);

            // second, compute tmust and propagate it to its peers
            // combine the pmustnt:
                __m256i tmust0 = _mm256_or_si256(pmustnt[0][0], pmustnt[1][0]);
                __m256i tmust1 = _mm256_or_si256(pmustnt[0][1], pmustnt[1][1]);
                __m256i tmust2 = _mm256_or_si256(pmustnt[0][2], pmustnt[1][2]);

                // tmust:
                // add to tmusts triads that are locked (exactly 3 candidates)
                // another task is to put aside data identifying triad candidate sets
                // minus the tmusts, that will be useful for finding sets and good guesses.
                unsigned int tul1 = grid_state->triads_unlocked[type] & (0x3f | (0x3f<<10) | (0x3f<<20));
                unsigned int tul2 = (grid_state->triads_unlocked[type]<<2) & (0x700 | (0x700<<10) | (0x700<<20));
                __m256i triadsv = _mm256_loadu2_m128i((__m128i*)&triads[6], (__m128i*)triads);
                tmust0 = _mm256_or_si256(tmust0, _mm256_andnot_si256(expand_bitvector((tul1 & 0x3f) | (tul2 & 0x700)),
                                                 triadsv));
                triadsv = _mm256_andnot_si256(tmust0, triadsv);
                _mm_storeu_si128((__m128i*)wo_musts, _mm256_castsi256_si128(triadsv));
                *(unsigned long long*)(&wo_musts[6]) = _mm256_extract_epi64(triadsv, 2);
                tul1 >>= 10;
                tul2 >>= 10;
                triadsv = _mm256_loadu2_m128i((__m128i*)&triads[16], (__m128i*)&triads[10]);
                tmust1 = _mm256_or_si256(tmust1, _mm256_andnot_si256(expand_bitvector((tul1 & 0x3f) | (tul2 & 0x700)),
                                                     triadsv));
                triadsv = _mm256_andnot_si256(tmust1, triadsv);
                _mm_storeu_si128((__m128i*)&wo_musts[10], _mm256_castsi256_si128(triadsv));
                *(unsigned long long*)(&wo_musts[16]) = _mm256_extract_epi64(triadsv, 2);
                tul1 >>= 10;
                tul2 >>= 10;
                triadsv = _mm256_loadu2_m128i((__m128i*)&triads[26], (__m128i*)&triads[20]);
                tmust2 = _mm256_or_si256(tmust2, _mm256_andnot_si256(expand_bitvector((tul1 & 0x3f) | (tul2 & 0x700)),
                                                     triadsv));
                triadsv = _mm256_andnot_si256(tmust2, triadsv);
                _mm_storeu_si128((__m128i*)&wo_musts[20], _mm256_castsi256_si128(triadsv));
                *(unsigned long long*)(&wo_musts[26]) = _mm256_extract_epi64(triadsv, 2);

            // i=0 (manually unrolled loop)
                // augment peer-based tmustnt by propagating the constraint (tmusti)
                // vertical peers
                tmustnt[1] = _mm256_or_si256(tmustnt[1], tmust0);
                tmustnt[2] = _mm256_or_si256(tmustnt[2], tmust0);

                // horizontal peers
                tmust0 = _mm256_shuffle_epi8(tmust0, rot_hpeers);
                tmustnt[0] = _mm256_or_si256(tmustnt[0], tmust0);
                tmust0 = _mm256_shuffle_epi8(tmust0, rot_hpeers);
                tmustnt[0] = _mm256_or_si256(tmustnt[0], tmust0);

             // i=1
                // vertical peers
                tmustnt[0] = _mm256_or_si256(tmustnt[0], tmust1);
                tmustnt[2] = _mm256_or_si256(tmustnt[2], tmust1);

                // horizontal peers
                tmust1 = _mm256_shuffle_epi8(tmust1, rot_hpeers);
                tmustnt[1] = _mm256_or_si256(tmustnt[1], tmust1);
                tmust1 = _mm256_shuffle_epi8(tmust1, rot_hpeers);
                tmustnt[1] = _mm256_or_si256(tmustnt[1], tmust1);

             // i=2
                // vertical peers
                tmustnt[0] = _mm256_or_si256(tmustnt[0], tmust2);
                tmustnt[1] = _mm256_or_si256(tmustnt[1], tmust2);

                // horizontal peers
                tmust2 = _mm256_shuffle_epi8(tmust2, rot_hpeers);
                tmustnt[2] = _mm256_or_si256(tmustnt[2], tmust2);
                tmust2 = _mm256_shuffle_epi8(tmust2, rot_hpeers);
                tmustnt[2] = _mm256_or_si256(tmustnt[2], tmust2);

            ptriads = triads;
            // compare tmustnt with the triads.
            unsigned int row_combo_tpos[3] {};
            unsigned int rslvd_row_combo_tpos[3] {};
            // for each group of nine ptriads:
            // - for column triads, each group (tmustnt) updates a band, the nine elemements (columns) 
            //   of the vector,
            // - for row triads, the n-th sub-groups updates the i-th row of the n-th band, with
            //   minor differences in the extraction of the update vector.
            for ( int i=0; i<3; i++, ptriads+=10) {
                // flip low and high lanes
                __m256i flip = _mm256_permute2x128_si256(tmustnt[i], tmustnt[i], 1);
                // rotate low 16 bytes left by 4 for subsequent move across lanes by alignr
                tmustnt[i]   = _mm256_shuffle_epi8(tmustnt[i], shuff_tmustnt);
                // align the 9 constraints in candidate order for comparison/update.
                tmustnt[i]   = _mm256_alignr_epi8(flip, tmustnt[i], 4);
                // isolate aligned updates
                __m256i to_remove_v = _mm256_and_si256(_mm256_and_si256(*(__m256i_u*)ptriads, tmustnt[i]),mask9x1ff);
                if ( _mm256_testz_si256(mask9x1ff,to_remove_v)) {
                    continue;
                }
                __m256i tmp = _mm256_cmpgt_epi16(to_remove_v,_mm256_setzero_si256());

                // m represents all triad indices to update
                unsigned long long m = compress_epi16_boolean(tmp);
                if ( m ) {
                    any_changes = true;
                }
                unsigned int rslvd_col_combo_tpos = 0;
                // remove triad candidate values that appear in tmustnt
                // and track updated cells
                if ( type == 0 ) {
                    // row triad updates, row i of each band
                    // update the cells for row triads
                    // row i
                    __m256i tmask;
                    unsigned int bits;
                    if ( (bits = m & 0x7) ) {
                        tmask = _mm256_permute2x128_si256(tmustnt[i], tmustnt[i], 0);
                        tmask = _mm256_shuffle_epi8(tmask, shuff_row_mask);
                        __m256i c = _mm256_andnot_si256(tmask, *(__m256i_u*)&candidates[i*9]);
                        _mm_storeu_si128((__m128i_u*)&candidates[i*9], _mm256_castsi256_si128(c));
                        candidates[8+i*9] = _mm256_extract_epi16(c, 8);
                        row_combo_tpos[0] |= bitx3_lut[bits] << (9*i);
                    }
                    // row i+3
                    if ( (bits = (m >> 3) & 0x7) ) {
                        tmask = _mm256_bsrli_epi128(tmustnt[i], 6);
                        tmask = _mm256_permute2x128_si256(tmask, tmask, 0);
                        tmask = _mm256_shuffle_epi8(tmask, shuff_row_mask);
                        __m256i c = _mm256_andnot_si256(tmask, *(__m256i_u*)&candidates[(i+3)*9]);
                        _mm_storeu_si128((__m128i_u*)&candidates[(i+3)*9], _mm256_castsi256_si128(c));
                        candidates[8+(i+3)*9] = _mm256_extract_epi16(c, 8);
                        row_combo_tpos[1] |= bitx3_lut[bits] << (9*i);
                    }
                    // row i+6
                    if ( (bits = (m >> 6) & 0x7) ) {
                        tmask = _mm256_permute4x64_epi64(tmustnt[i], 0x99);
                        tmask = _mm256_shuffle_epi8(tmask, shuff_row_mask2);
                        __m256i c = _mm256_andnot_si256(tmask, *(__m256i_u*)&candidates[(i+6)*9]);
                        _mm_storeu_si128((__m128i_u*)&candidates[(i+6)*9], _mm256_castsi256_si128(c));
                        candidates[8+(i+6)*9] = _mm256_extract_epi16(c, 8);
                        row_combo_tpos[2] |= bitx3_lut[bits] << (9*i);
                    }
                } else { // type == 1
                    // update the band of 3 rows with column triads
                    // using directly tmustnt[i]
                    unsigned int i27 = i*27;
                    __m256i c1 = _mm256_andnot_si256(tmustnt[i], *(__m256i_u*)&candidates[i27]);
                    _mm_storeu_si128((__m128i_u*)&candidates[i27], _mm256_castsi256_si128(c1));
                    __m256i c2 = _mm256_andnot_si256(tmustnt[i], *(__m256i_u*)&candidates[9+i27]);
                    _mm_storeu_si128((__m128i_u*)&candidates[9+i27], _mm256_castsi256_si128(c2));
                    __m256i c3 = _mm256_andnot_si256(tmustnt[i], *(__m256i_u*)&candidates[18+i27]);
                    _mm_storeu_si128((__m128i_u*)&candidates[18+i27], _mm256_castsi256_si128(c3));
                    if ( m & 0x100 ) {
                        candidates[8+i27]    = _mm256_extract_epi16(c1, 8);
                        candidates[9+8+i27]  = _mm256_extract_epi16(c2, 8);
                        candidates[18+8+i27] = _mm256_extract_epi16(c3, 8);
                    }
                    grid_state->updated.set_indexbits(m | (m<<9) | (m<<18), i*27, 27);

                }
                // iterate over triads to update
                if ( verbose != VNone ) {
                     counters.triad_updates += _popcnt64(m);
                }
                while (m) {
                    // 1. Check for triad resolution
                    // 2. log the triad update from above
                    // 3. for columns, track resolved triads
                    unsigned char i_rel = tzcnt_and_mask(m);
                    unsigned char ltidx = i_rel + 9*i;  // logical triad index (within band, different layouts for row/col)
                    // compute bit offsets of the resolved triad
                    // and save the mask
                    if ( _popcnt32(((v16us)tmustnt[i])[i_rel]) == 6 ) {
                        if ( type == 0 ) {
                            rslvd_row_combo_tpos[i_rel/3] |= bandbits_by_index[row_triad_canonical_map[ltidx]%9];
                        } else {
                            rslvd_col_combo_tpos |= 0x40201<<i_rel;
                        }
                        if ( verbose != VNone ) {
                            counters.triads_resolved++;
                        }
                    }
                    if ( verbose == VDebug ) {
                        char ret[32];
                        format_candidate_set(ret, ((v16us)to_remove_v)[i_rel]);
                        solverData.printf("remove %-5s from %s triad at %s\n", ret, type == 0? "row":"col",
                               cl2txt[type==0?row_triad_canonical_map[ltidx]*3:col_canonical_triad_pos[ltidx]] );
                    }
                }
                if ( type == 1 ) {
                    if ( rslvd_col_combo_tpos ) {
                        grid_state->set23_found[Col].set_indexbits(rslvd_col_combo_tpos, i*27, 27);
                        grid_state->set23_found[Box].set_indexbits(rslvd_col_combo_tpos, i*27, 27);
                    }
                }
            }
            // finally, for rows only, track updated triad cells and resolved triads
            if ( type == 0 ) {
                for ( unsigned char k=0; k<2; k++ ) {
                    if ( row_combo_tpos[k] ) {
                        grid_state->updated.set_indexbits(row_combo_tpos[k], k*27, 27);
                    }
                    if ( rslvd_row_combo_tpos[k] ) {
                        grid_state->set23_found[Row].set_indexbits(rslvd_row_combo_tpos[k], k*27, 27);
                        grid_state->set23_found[Box].set_indexbits(rslvd_row_combo_tpos[k], k*27, 27);
                    }
                }
            }
        } // for type (cols,rows)

        if ( any_changes ) {
            return Phase_Search;
        }
    }

    // here we are beyond the enter/single search/triad/, on to the more complex algos,
    // so we count this as a 'round'.
    if ( verbose != VNone ) {
        counters.past_naked_count++;
    }

    {
        // Before making a guess,
        // check for a 'universal grave'+1, which is an end-game move.
        // First get a count of unlocked.
        // With around 22 or less unresolved cells (N), accumulate a popcount of all cells.
        // Count the cells with a candidate count 2 (P). 81 - N - P = Q.
        // If Q is 0, back track, if a guess was made before. without guess to back track to,
        // take note as duplicate solution, but carry on with a guess.
        // If Q > 1, go with a guess.
        // If Q == 1, identify the only cell which does not have a 2 candidate count.
        // If the candidate count is 3, determine for any section with the cell which
        // candidate value appears in this cell and other cells of the section 3 times.
        // That is the correct solution for this cell, enter it.

        unsigned char N = _popcnt64(unlocked[0]) + _popcnt32(unlocked[1]);
        if ( N <= 23 ) {
            bit128_t &bivalues = solverData.getBivalues(candidates);
            unsigned char target = 0;   // the index of the only cell with three candidates
            int sum2 = _popcnt64(bivalues.u64[0]) + _popcnt32(bivalues.u64[1]);
            if ( sum2 == N ) {
                grid_state->flags |= 1U<<31;	// set sign bit
                if ( verbose != VNone ) {
                    counters.bug_count++;
                }
                if ( rules != Regular ) {
                    if ( verbose == VDebug ) {
                        if ( grid_state->stackpointer == 0 && !unique_check_mode ) {
                            solverData.printf("Found a bi-value universal grave. This means at least two solutions exist.\n");
                        } else if ( unique_check_mode ) {
                            solverData.printf("checking a bi-value universal grave.\n");
                        }
                    }
                    return Phase_Guess;
                } else if ( grid_state->stackpointer ) {
                    if ( verbose == VDebug ) {
                        solverData.printf("back track - found a bi-value universal grave.\n");
                    }
                    return Phase_Back;
                } else {   // busted.  This is not a valid puzzle under standard rules.
                    if ( verbose == VDebug ) {
                        solverData.printf("Found a bi-value universal grave. This means at least two solutions exist.\n");
                    }
                    status.unique = false;  // set to non-unique even under Regular rules
                    return Phase_Guess;
                }
            } else if ( sum2+1 == N ) {  // find the single cell with count > 2
                bit128_t gt2 = { .u128 = ((bit128_t*)unlocked)->u128 & ~bivalues.u128 };
                // locate the cell
                unsigned long long m = gt2.u64[0];
                if ( m == 0 ) {
                    m = gt2.u64[1];
                    target = 64;
                }
                target += __tzcnt_u64(m);

                if ( __popcnt16(candidates[target]) == 3 ) {
                    unsigned char row = row_index[target];
                    unsigned short cand3 = candidates[target];
                    unsigned short digit = 0;
                    unsigned short mask = ((bit128_t*)unlocked)->get_indexbits(row*9,9);
                    __m256i maskv = expand_bitvector(mask);
                    __m256i c = _mm256_and_si256(_mm256_load_si256((__m256i*) &candidates[row*9]), maskv);
                    while (cand3) {
                        unsigned short canddigit = __blsi_u32(cand3);
                        // count cells with this candidate digit:
                        __m256i tmp = _mm256_and_si256(_mm256_set1_epi16(canddigit), c);
                        // as a boolean
                        tmp = _mm256_cmpeq_epi16(tmp,_mm256_setzero_si256());
                        // need three cell, doubled bits in mask:
                        if ( _popcnt32(~_mm256_movemask_epi8(tmp)) == 3*2 ) {
                            digit = canddigit;
                            break;
                        }
                        cand3 &= ~canddigit;
                    }
                    if ( digit ) {
                        if ( verbose != VNone ) {
                            counters.bug_plus1_count++;
                        }
                        if ( verbose == VDebug ) {
                            solverData.printf("bi-value universal grave + 1: pivot:");
                        }
                        if ( rules == Regular ) {
                            e_i = target;
                            e_digit = digit;
                            return Phase_Enter;
                        } else {
                            if ( verbose == VDebug ) {
                                solverData.printf("\n");
                            }
                            grid_state = grid_state->make_guess<verbose>(target, digit, counters, solverData.output);
                        }
                        return Phase_Start;
                    }
                }
            }
        }
    }


    SolverPhase r;
    (void)r;  // suppress -Wunused if no OPT_* flag is defined
#ifdef OPT_NEWSETS
    r = do_naked_sets_new();
    if (r != Phase_HiddenSearch) return r;
#endif
#ifdef OPT_SETS
    r = do_naked_sets_main();
    if (r != Phase_HiddenSearch) return r;
#endif
#ifdef OPT_FSH
    r = do_fishes();
    if (r != Phase_HiddenSearch) return r;
#endif
#if OPT_UQR
    r = do_unique_rectangles();
    if (r != Phase_HiddenSearch) return r;
#endif
    return Phase_Guess;
}