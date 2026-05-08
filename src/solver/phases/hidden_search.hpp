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

#ifdef OPT_NEWSETS
// Algorithm 4 - Find naked sets
// For each possible combination of candidates of size K, check for every section
// whether there are exactly K cells that contain only this combination of candidates.
// Back track when the number of found cells exceeds K.
//
// Implementation:
// As an approximation, use actual cells of a section to check whether their candidates are
// a set.  Line up the cells in a 'tentative' vector and the 'testing' vector.
// The testing vector is then shifted and compared to the tentative vector, to cumulate the score.
// If the score matches the popcount of the tentative cell, its a set.
//
// Since the vector size is 8 only eight cells are represented in the tentative vector.
// This allows for testing two rows in parallel. The nineth cell is broadcast once and
// also tested against the tentative vector.
// Last, when testing the nineth cell, the test is run in reverse and yields a score for
// the nineth cell.
//
// Optionally, other sets can be inserted in place of cell-based tentative sets.
// Triad unions make a good tentative set; cells could be combined in some fashion.
//
// This works for all section types, row, col and box.
//
// There are some mitgation techniques to keep unproductive section being tested or retested.
// For example, after one or two sets discovered, there is a much diminished propability
// for finding another one. Also, the found set already might play a vital role in
// the downstream resolution of the Sudoku.
// The actual unresolved size of the section is also an important input.
//
// Some general thoughts on the algorithm below.
//
// What does this search achieve:
// for a section with N unlocked cells each naked set of K corresponds to a hidden set of N-K.
// Note that not all sets are detected, mainly because:
// - the cells to examine are selected based on prior updates
// - sets can exist whithout any of their cells having all the candidate values.
//   (shortcoming of the algorithm used)
// Note that the debug output reports the naked set/pair, or if the naked set is > 3 and
// its complement has less than 4 member candidates, it is reported instead as a hidden set/pair.
//
// The other important accomplishment is the ability to detect back
// track scenarios if the discovered set is impossibly large.  This is quite important
// for performance as a chance to kill off bad guesses.
//
    if ( mode_newsets ) {
//        Kind type = Row;  // for now
// XXX create a popcount of the board or just one row at a time
// XXX should be in Data section
// XXX should derive bivalues as well?
// XXX result could be compressed to nibbles.
        // A cheap way to avoid unnecessary naked set searches
        grid_state->set23_found[Row].u128 |= ~grid_state->unlocked.u128;
        grid_state->set23_found[Row].u64[1] &= 0x1ffff;

        unsigned char*sctCnt = solverData.getSectionSetUnlocked<Row>(*grid_state);
        // unsigned char*sctCntBox = solverData.getSectionSetUnlocked<Box>(*grid_state);
        unsigned char pop[81];
        for ( unsigned char j=0; j<64; j+=32 ) {
            __m256i v1 = _mm256_loadu_si256((__m256i *)&candidates[j]);
            __m256i v2 = _mm256_loadu_si256((__m256i *)&candidates[j+16]);
            __m256i v9 = _mm256_packus_epi16(_mm256_srli_epi16(v1,8), _mm256_srli_epi16(v2,8));
            v1 = _mm256_packus_epi16(_mm256_and_si256(v1, maskff), _mm256_and_si256(v2, maskff));
            __m256i lo1 = _mm256_and_si256 (v1, nibble_mask);
            __m256i hi1 = _mm256_and_si256 (_mm256_srli_epi16 (v1, 4), nibble_mask );
            __m256i cnt11 = _mm256_shuffle_epi8 (lookup, lo1);
            cnt11 = _mm256_add_epi8(_mm256_add_epi8(cnt11, _mm256_shuffle_epi8 (lookup, hi1)), v9);
            _mm256_storeu_si256((__m256i_u*)(pop+j), _mm256_permute4x64_epi64(cnt11, 0xD8));
        }
        {
            __m128i v1 = _mm_loadu_si128((__m128i *)&candidates[64]);
            __m128i v2 = _mm_loadu_si128((__m128i *)&candidates[72]);
            __m128i v9 = _mm_packus_epi16(_mm_srli_epi16(v1,8), _mm_srli_epi16(v2,8));
            v1 = _mm_packus_epi16(_mm_and_si128(v1, _mm256_castsi256_si128(maskff)), _mm_and_si128(v2, _mm256_castsi256_si128(maskff)));
            __m128i lo1 = _mm_and_si128 (v1, _mm256_castsi256_si128(nibble_mask));
            __m128i hi1 = _mm_and_si128 (_mm_srli_epi16 (v1, 4), _mm256_castsi256_si128(nibble_mask) );
            __m128i cnt11 = _mm_shuffle_epi8 (_mm256_castsi256_si128(lookup), lo1);
            cnt11 = _mm_add_epi8(_mm_add_epi8 (cnt11, _mm_shuffle_epi8 (_mm256_castsi256_si128(lookup), hi1)), v9);
            _mm_storeu_si128((__m128i_u*)(pop+64), cnt11);
            pop[80] = __popcnt16(candidates[80]);
        }

//dump_board(candidates, "board");
        // iterate over rows
        for (unsigned char i = 0; i < 81; i += 18 ) {
            bit128_t to_change;
            unsigned char rowsix[2] = {i, (unsigned char)(i+(i==72?0:9))};
            unsigned char rows[2] = {(unsigned char)(i/9), (unsigned char)(i/9+(i==72?0:1))};
#if 0
            //XXX
            if ( sctCnt[rows[0]] < 4 && sctCnt[rows[1]] < 4 ) {
                continue;
            }
#endif
            if ( verbose != VNone ) {
                counters.naked_sets_searched += i==72?1:2;
            }

            // rows, in pairs
            // first lane for the i-th section, 2nd lane for the (i+1)-st section.

            __m256i c_tst = _mm256_set_m128i(*(__m128i_u*) &candidates[rowsix[1]],
                                             *(__m128i_u*) &candidates[rowsix[0]]);
            __m256i c_try = c_tst;
            // set the popcont-1:
            __m256i c_res = _mm256_set_m128i(_mm_unpacklo_epi8(*(__m128i_u*)(pop+rowsix[1]), _mm_setzero_si128()),
                                             _mm_unpacklo_epi8(*(__m128i_u*)(pop+rowsix[0]), _mm_setzero_si128()));
//            __m256i c_intersect {};  // bits for each intersection between c_tst and c_try.

            // Adjust counts to come out "right"
            // plus 1 for cells that have N candidates for N liberties:
//dbgprintf(1, "sctCnt[%d]=%d, sctCnt[%d]=%d\n", rows[0], sctCnt[rows[0]], rows[1], sctCnt[rows[1]]);
            c_res = _mm256_add_epi16(c_res, _mm256_and_si256(_mm256_cmpeq_epi16(c_res,
                                            _mm256_set_m128i(_mm_set1_epi16(sctCnt[rows[1]]),_mm_set1_epi16(sctCnt[rows[0]]))),ones));
            // minus 1 for "unlocked" cells:
            c_res = _mm256_sub_epi16(c_res, _mm256_add_epi16(_mm256_cmpeq_epi16(c_res,ones),ones));
//dump_m256i_epi16(c_res, "popcnts - adjusted");

            for (unsigned char k = 0; k < 7; k++) {
                // rotate left (0 1 2 3 4 5 6 7) -> (1 2 3 4 5 6 7 0)
                c_try = _mm256_alignr_epi8(c_try, c_try, 2);
                // the test: (add -1)
                c_res =  _mm256_add_epi16(c_res,
                             _mm256_cmpeq_epi16( _mm256_andnot_si256(c_tst, c_try), _mm256_setzero_si256()));
//dump_m256i_epi16(c_try, "c_try", 2);
//dump_m256i_epi16(c_res, "iteration", 2);
#if 0
                c_intersect = _mm256_slli_epi16(
                                  _mm256_or_si256(c_intersect,
                                      _mm256_andnot_si256(_mm256_cmpeq_epi16(
                                          _mm256_and_si256(c_tst, c_try), _mm256_setzero_si256()),ones)),1);
#endif
            }
//dump_m256i_epi16(c_intersect, "intersect", 2);
            __m256i the9thcand_row = _mm256_set_m128i(_mm_set1_epi16(candidates[rowsix[1]+8]),_mm_set1_epi16(candidates[rowsix[0]+8]));
            // perform the same tests with the the9thcand_row candidates:
            c_res =  _mm256_add_epi16(c_res, 
                         _mm256_cmpeq_epi16( _mm256_andnot_si256(c_tst, the9thcand_row), _mm256_setzero_si256()));
//dump_m256i_epi16(c_res, "popcnts nineth",2);

            // this gives sixteen results, plus two for the nineth row candidates.

            // check whether any set is too large:
            if ( !_mm256_testz_si256(true_epi16, _mm256_cmpgt_epi16(_mm256_setzero_si256(), c_res)) ) {
                // backtrack
//dump_m256i(c_res, "c_res - minus one - backtrack");
                unsigned short res = compress_epi16_boolean(_mm256_cmpgt_epi16(_mm256_setzero_si256(), c_res));
                unsigned char off = __tzcnt_u32(res);
                unsigned char row = rows[off<8?0:1];
                unsigned char cl = row*9+(off&7);
                if ( verbose != VNone ) {
                    char ret[32];
                    format_candidate_set(ret, candidates[cl]);
                    if ( grid_state->stackpointer == 0 && unique_check_mode == 0 ) {
                        if ( warnings != 0 ) {
                            solverData.printf("Line %d: naked  set (row) %s at %s, count exceeded\n", line, ret, cl2txt[cl]);
                        }
                    } else if ( debug ) {
                        solverData.printf("back track sets (row) %s at %s, count exceeded\n", ret, cl2txt[cl]);
                    }
                }
                // no need to update grid_state
                return Phase_Back;
            }
#if 0
            c_intersect = _mm256_or_si256(c_intersect,
                              _mm256_andnot_si256(_mm256_cmpeq_epi16(
                                  _mm256_and_si256(c_tst, the9thcand_row), _mm256_setzero_si256()),ones));
            if ( !_mm256_testz_si256(c_intersect, _mm256_cmpeq_epi16(c_res, _mm256_setzero_si256()))) {
#else
            if ( !_mm256_testz_si256(ones, _mm256_cmpeq_epi16(c_res, _mm256_setzero_si256()))) {
#endif
                // a bit set for either of the sixteen cells
                unsigned int res = compress_epi16_boolean(_mm256_cmpeq_epi16(c_res, _mm256_setzero_si256()));
                res &= ~(  grid_state->set23_found[Row].get_indexbits(i,8)
                         | grid_state->set23_found[Row].get_indexbits(i+9,8)<<8);
#if 0
                // XXX
                if ( sctCnt[rows[0]] < 4 ) {
                    res &= ~(0xff<<8);
                }
                if ( sctCnt[rows[1]] < 4 ) {
                    res &= ~0xff;
                }
#endif
                while ( res ) {
                    unsigned char off = tzcnt_and_mask(res);
                    unsigned char row = rows[off<8?0:1];
                    unsigned char cl = row*9+(off&7);
                    // obtain a vector for the row of the found set:
                    __m128i rowv = (off<8)?_mm256_castsi256_si128(c_tst):_mm256_extracti128_si256(c_tst,1);

                    // obtain the bit pattern for the set:
                    unsigned short m = compress_epi16_boolean128(_mm_cmpeq_epi16(
                                           _mm_andnot_si128(_mm_set1_epi16(candidates[cl]), rowv), _mm_setzero_si128()))
                                       | ((~candidates[cl] & candidates[row*9+8])?0:0x100);
                    // obtain the bit pattern for the set to clean:
                    unsigned short m_clean = compress_epi16_boolean128(_mm_cmpgt_epi16(
                                           _mm_and_si128(_mm_set1_epi16(candidates[cl]), rowv), _mm_setzero_si128()))
                                       | ((candidates[cl] & candidates[row*9+8])?0x100:0);
//dbgprintf(1, "m_clean 1=%x\n", m_clean);
                    m_clean &= ~m;

                    unsigned int m_neg = 0x1ff & ~(m | grid_state->set23_found[Row].get_indexbits(row*9,9));

//dbgprintf(1, "set: row=%d %s mask=%x, m_neg=%x, m_clean=%x\n", row, cl2txt[cl], m, m_neg, m_clean);
                    to_change.u128 = 0;
                    to_change.set_indexbits(m_clean, row*9, 9);
                    unsigned char cnt = _popcnt32(candidates[cl]);
                    unsigned char s = sctCnt[row];
                    if ( s <= 3 + cnt ) {
                        // could include locked slots
                        grid_state->set23_found[Row].set_indexbits(m_neg,row*9,9);
                        // adjust the count, maybe required lateron
                        // XXX sectionSetsUnlockedCnt[Row][row] -= _popcnt32(m_neg);
                    }
                    if ( cnt <= 3 ) {
                        // unsigned int row_box_intersection = 7<<(box_start[cl]%9);
                        grid_state->set23_found[Row].set_indexbits(m,row*9,9);
                        // adjust the count - required if finding multiple sets in row
                        sctCnt[row] -= cnt;
#if 0
                        // update box, if set within triad
                        if ( (m&row_box_intersection) == m ) {
                            // grid_state->set23_found[Box].set_indexbits(m&0x1ff,row*9,9);
                            // int bi_now = sctCntBox[box_index[cl]];
                            // if ( bi_now >= cnt ) {
                            //     sctCntBox[box_index[cl]] = bi_now - cnt;
                            // }
                            add_indices<Box>(&to_change, cl);
                        }
#endif
                    }
                    if ( m_clean == 0 ) {
                        res &= ~(m&0xff);
                        continue;
                    }
//dump_bits(to_change, "to_change");
//                    if ( cnt==4 || sctCnt[row]==4+cnt ) {
//                        grid_state->flags |= (1<<row);
//                    }
                    if ( to_change.u128 ) {
                        if ( verbose == VDebug ) {
                            char ret[32];
                            if ( cnt <=3 || cnt <= sctCnt[row]-cnt ) {
                                format_candidate_set(ret, candidates[cl]);
                                solverData.printf("naked  %s (row): %-7s %s\n", cnt==2?"pair":"set ", ret, cl2txt[cl]);
                            } else {
//                                if (sctCnt[row] > 3+cnt) {
//                                    m_neg = 0x1ff & ~(m | grid_state->set23_found[Row].get_indexbits(row*9,9));
//                                }
                                unsigned char k = 0xff;
                                unsigned short complement = 0;
                                while (m_neg) {
                                    unsigned char k_i = tzcnt_and_mask(m_neg);
                                    k_i += row*9;
                                    if ( k == 0xff ) {
                                        k = k_i;
                                    }
                                    complement |= candidates[k_i];
                                }
                                complement &= ~candidates[cl];
                                format_candidate_set(ret, complement);
                                solverData.printf("%s %s (row): %-7s %s\n", complement?"hidden":"naked ", __popcnt16(complement)==2?"pair":"set ", ret, cl2txt[k]);
                            }
                        }
                        unsigned short cdi = candidates[cl];
                        unsigned short cdin = ~cdi;
                        bool found = false;
//dump_bits(to_change,"to_change");
//dump_board(candidates, "board");
                        while (to_change) {
                            unsigned char j = tzcnt_and_mask(to_change);
//dbgprintf(1,"j=%d,",j);
                            // if this cell is not part of our set
                            if (candidates[j] & cdin ) {
                                // if there are bits that need removing
                                if (candidates[j] & cdi) {
                                    candidates[j] &= cdin;
//dbgprintf(1,"%s,",cl2txt[j]);
                                    //to_visit_again.set_indexbit(j);
                                    found = true;
                                }
                            }
                        }
                        if ( found ) {
                            if ( verbose != VNone ) {
                                counters.naked_sets_found++;
                            }
                            return Phase_Search;
                        }
                    }
//dump_board(candidates, "board");
                    res &= ~(m&0xff);
//dbgprintf(1,"\n");
//dbgprintf(1,"Line %d: set reported, but no changes made!\n", line);
                }
            }
            int cnt1 = _popcnt32(candidates[rowsix[0]+8]);
            int cnt2 = _popcnt32(candidates[rowsix[1]+8]);
//dbgprintf(1, "rowsix[0]+8=%d, rowsix[1]+8=%d\n", rowsix[0]+8, rowsix[1]+8);
//dbgprintf(1, "cnt1=%d, cnt2=%d\n", cnt1, cnt2);
//dump_bits(grid_state->set23_found[Row],"grid_state->set23_found[Row]");
            if ( cnt1 == sctCnt[rows[0]] || cnt1 == 1 || grid_state->set23_found[Row].check_indexbit(rowsix[0]+8) ) {
//dbgprintf(1, "grid_state->set23_found[Row].check_indexbit(rowsix[0])=%d", grid_state->set23_found[Row].check_indexbit(rowsix[0]));
                cnt1 = 0;
            }
            if ( cnt2 == sctCnt[rows[1]] || cnt2 == 1 || grid_state->set23_found[Row].check_indexbit(rowsix[1]+8) ) {
//dbgprintf(1, "grid_state->set23_found[Row].check_indexbit(rowsix[1])=%d", grid_state->set23_found[Row].check_indexbit(rowsix[1]));
                cnt2 = 0;
            }

//dbgprintf(1, "cnt1=%d, cnt2=%d\n", cnt1, cnt2);

            if ( cnt1 || cnt2 ) {
                // perform the reverse test with the the9thcand_row candidates,
                // and save a bit count for the result:
                unsigned short cnts = compress_epi16_boolean(_mm256_cmpeq_epi16( _mm256_andnot_si256(the9thcand_row, c_tst), _mm256_setzero_si256()));
// XXX this check is probably not required, as the condition for a back track would have been detected on a different cell? Or not?
                if ( (cnt1 && cnt1 < _popcnt32(cnts & 0xff)) || (cnt2 && cnt2 < _popcnt32(cnts >> 8)) ) {
                    // backtrack
//dbgprintf(1, "cnt1=%d, cnt2=%d - backtrack\n", cnt1, cnt2);
                    unsigned char row = rows[(cnt1 && cnt1<_popcnt32(cnts & 0xff))?0:1];
                    unsigned char cl = row*9+8;
                    if ( verbose != VNone ) {
                        char ret[32];
                        format_candidate_set(ret, candidates[cl]);
                        if ( grid_state->stackpointer == 0 && unique_check_mode == 0 ) {
                            if ( warnings != 0 ) {
                                solverData.printf("Line %d: naked  set (row) %s at %s, count exceeded\n", line, ret, cl2txt[cl]);
                            }
                        } else if ( debug ) {
                            solverData.printf("back track set (row) %s at %s, count exceeded\n", ret, cl2txt[cl]);
                        }
                    }
                    // no need to update grid_state
                    return Phase_Back;
                }
//dbgprintf(1, "cnt1=%d, cnt2=%d, cnts=%x\n", cnt1, cnt2, cnts);
//dbgprintf(1, "popcnt32(cnts & 0xff)+1=%d, _popcnt32(cnts >> 8)+1=%d\n", _popcnt32(cnts & 0xff)+1, _popcnt32(cnts >> 8)+1);
                if ( (cnt1 == _popcnt32(cnts & 0xff)+1) || (cnt2 == _popcnt32(cnts >> 8)+1) ) {
                    // a set for either of the nineth cells
                    if ( verbose != VNone ) {
                        counters.naked_sets_found++;
                    }
                    unsigned char ix = (cnt1 == _popcnt32(cnts & 0xff)+1)?0:1;
                    unsigned char row = rows[ix];
                    unsigned char cl = row*9+8;
//dbgprintf(1, "Line=%d row=%d, cnt1=%d, cnt2=%d cnts=%x - set at %s\n", line, row, cnt1, cnt2, cnts, cl2txt[cl]);

                    // obtain a vector for the row of the found set:
                    __m128i rowv = (ix==0)?_mm256_castsi256_si128(c_tst):_mm256_extracti128_si256(c_tst,1);

                    // obtain the bit pattern for the set:
                    unsigned short m = compress_epi16_boolean128(_mm_cmpeq_epi16(
                                           _mm_andnot_si128(_mm_set1_epi16(candidates[cl]), rowv), _mm_setzero_si128()))
                                       | ((~candidates[cl] & candidates[row*9+8])?0:0x100);
                    // obtain the bit pattern for the set to clean:
                    unsigned short m_clean = compress_epi16_boolean128(_mm_cmpgt_epi16(
                                           _mm_and_si128(_mm_set1_epi16(candidates[cl]), rowv), _mm_setzero_si128()))
                                       | ((candidates[cl] & candidates[row*9+8])?0x100:0);
//dbgprintf(1, "m_clean=%x\n", m_clean);
//dbgprintf(1, "popcnt32(cnts & 0xff)+1=%d, _popcnt32(cnts >> 8)+1=%d\n", _popcnt32(cnts & 0xff)+1, _popcnt32(cnts >> 8)+1);
                    m_clean &= ~m;
                    unsigned int m_neg = 0x1ff & ~(m | grid_state->set23_found[Row].get_indexbits(row*9,9));
//dbgprintf(1, "set: row=%d %s mask=%x, m_neg=%x, m_clean=%x\n", row, cl2txt[cl], m, m_neg, m_clean);
                    to_change.u128 = 0;
                    to_change.set_indexbits(m_clean, row*9, 9);
                    unsigned char cnt = _popcnt32(candidates[cl]);
                    if ( cnt <= 3 ) {
                        grid_state->set23_found[Row].set_indexbits(m,row*9,9);
                        // adjust the count - maybe required lateron
                        // XXX sectionSetsUnlockedCnt[Row][row] -= cnt;
                    }
                    if ( sctCnt[row] <= 3 + cnt ) {
                        // could include locked slots
                        grid_state->set23_found[Row].set_indexbits(m_neg,row*9,9);
                        // adjust the count, maybe required lateron
                        // XXX sectionSetsUnlockedCnt[Row][row] -= _popcnt32(m_neg);
                    }
                    if ( cnt==4 || sctCnt[row]==4+cnt ) {
                        grid_state->flags |= (1<<row);
                    }
                    if ( to_change.u128 ) {
                        if ( verbose == VDebug ) {
                            char ret[32];
                            if ( cnt <=3 || cnt <= sctCnt[row]-cnt ) {
                                format_candidate_set(ret, candidates[cl]);
                                solverData.printf("naked  %s (row): %-7s %s\n", cnt==2?"pair":"set ", ret, cl2txt[cl]);
                            } else {
                                unsigned char k = 0xff;
                                unsigned short complement = 0;
                                while (m_neg) {
                                    unsigned char k_i = tzcnt_and_mask(m_neg);
                                    k_i += row*9;
                                    if ( k == 0xff ) {
                                        k = k_i;
                                    }
                                    complement |= candidates[k_i];
                                }
                                complement &= ~candidates[cl];
                                format_candidate_set(ret, complement);
                                solverData.printf("%s %s (row): %-7s %s\n", complement?"hidden":"naked ", __popcnt16(complement)==2?"pair":"set ", ret, cl2txt[row*9+k%9]);
                            }
                        }
                        unsigned short cdi = candidates[cl];
                        unsigned short cdin = ~cdi;
                        bool found = false;
//dump_bits(to_change,"to_change");
                        while (to_change) {
                            unsigned char j = tzcnt_and_mask(to_change);
//dbgprintf(1,"j=%d,",j);
                            // if this cell is not part of our set
                            if (candidates[j] & cdin ) {
                                // if there are bits that need removing
                                if (candidates[j] & cdi) {
                                    candidates[j] &= cdin;
//dbgprintf(1,"%s,",cl2txt[j]);
                                    //to_visit_again.set_indexbit(j);
                                    found = true;
                                }
                            }
                        }
                        if ( found ) {
                            if ( verbose != VNone ) {
                                counters.naked_sets_found++;
                            }
                            return Phase_Search;
                        }
                    }
                }
            }
        }        
    }
#endif
#ifdef OPT_SETS

// Algorithm 4 - Find naked sets
// For each possible combination of candidates of size K, check for every section
// whether there are exactly K cells that contain only this combination of candidates.
// Back track when the number of found cells exceeds K.
//
// Implementation:
// As an approximation, start with any cell and use its candidates as the starting
// set.  Avoid useless and repeat searches using a variety of heuristics.
//
// Find naked sets, up to MAX_SET

#define MAX_SET 5
// Some general thoughts on the algorithm below.
//
// What does this search achieve:
// for a section with N unlocked cells each naked set of K corresponds to a hidden set of N-K.
// Going up to size 5 allows to get at the very least all hidden sets of 4, and most likely
// (with N<9) all hidden sets of 3.
// Note that not all sets are detected, mainly because:
// - the cells to examine are selected based on prior updates
// - sets can exist whithout any of their cells having all the candidate values.
//   (shortcoming of the algorithm used)
// Note that the debug output reports the naked set/pair, or if the naked set is > 3 and
// its complement has less than 4 member candidates, it is reported instead as a hidden set/pair.
//
// The other important accomplishment is the ability to detect back
// track scenarios if the discovered set is impossibly large.  This is quite important
// for performance as a chance to kill off bad guesses.
//
// Note that this search has it's own built-in heuristic to tackle only recently updated cells.
// The algorithm will keep that list to revisit later, which is fine of course.
//
// Critique:
// The approach is questionable, as it does not schedule the full section for revisit...
// It would be better to examine the full section and then only revisit when necessary.
// To be section centric as opposed to cell centric should allow for better performance as well.
//
// The number of cells to visit can be high (e.g. in the beginning).
// Additional tracking mechanisms are used to reduce the number of searches:
// - previously found sets (and their complements) of size 2 and 3 as well as found triads
// - sets that occupy all available space minus one - impossible due to perfect single detection
//

    if ( mode_sets )
    {
        bool found = false;

        // visit only the changed (updated) cells

        bit128_t to_visit_n;          // tracks all the cells to visit
        bit128_t to_visit_again {};   // track those cells that have been updated

        to_visit_n.u128 = grid_state->updated.u128 & grid_state->unlocked.u128;

        // A cheap way to avoid unnecessary naked set searches
        grid_state->set23_found[Row].u128 |= ~grid_state->unlocked.u128;
        grid_state->set23_found[Row].u64[1] &= 0x1ffff;
        grid_state->set23_found[Col].u128 |= ~grid_state->unlocked.u128;
        grid_state->set23_found[Col].u64[1] &= 0x1ffff;
        grid_state->set23_found[Box].u128 |= ~grid_state->unlocked.u128;
        grid_state->set23_found[Box].u64[1] &= 0x1ffff;

        unsigned char *sectionSetsUnlockedCnt[3] = { 
                            solverData.getSectionSetUnlocked<Row>(*grid_state),
                            solverData.getSectionSetUnlocked<Col>(*grid_state),
                            solverData.getSectionSetUnlocked<Box>(*grid_state) };

      flip ^= 1;

      for ( int a = 0; a<2; a++ ) {
        // this scheme divides to_visit_n into two halves with alternating bits.
        // this increases substantially the detection rate.
        bit128_t tv = { .u128 = *cast2cu128(altbits[a^flip]) & to_visit_n.u128 };
        to_visit_n.u128 &= ~tv.u128;

        while (tv.u128) {
            unsigned char i = tzcnt_and_mask(tv);
            unsigned short cnt = __popcnt16(candidates[i]);

            if (cnt <= MAX_SET && cnt > 1) {
                // Note: this algorithm will never detect a naked set of the shape:
                // {a,b},{a,c},{b,c} as all starting points are 2 bits only.
                // The same situation is possible for 4 set members.
                //
                bit128_t to_change {};
                 __m256i a_i = _mm256_set1_epi16(candidates[i]);
                __m128i res;
                unsigned char ul;
                unsigned char s;

                // check row
                //
                if ( !grid_state->set23_found[Row].check_indexbit(i)) {
                    unsigned char ri = row_index[i];
                    ul = sectionSetsUnlockedCnt[Row][ri];
                    if ( grid_state->flags & (1<<ri) && ul > 4 ) {
                        // if a set of 4 has been detected previously, the ul can be updated
                        // this prevents repeated detection of sets of 4, which have previously been cleaned up.
                        // Considering the possible sets of 4 and ul-4:
                        ul = ul<=8? 4:5;
                    }
                    if (check_back || (cnt+2 <= ul) ) {
                        if (verbose != VNone ) {
                            counters.naked_sets_searched++;
                        }
                        res = _mm_cmpeq_epi16(_mm256_castsi256_si128(a_i), _mm_or_si128(_mm256_castsi256_si128(a_i), *(__m128i_u*) &candidates[9*ri]));
                        unsigned int m = compress_epi16_boolean128(res);
                        bool bit9 = candidates[i] == (candidates[i] | candidates[9*ri+8]);
                        if ( bit9 ) {
                            m |= 1<<8;    // fake the 9th mask position
                        }
                        s = _popcnt32(m);
                        unsigned int m_neg = 0;
                        if (s > cnt) {
                            if ( verbose != VNone ) {
                                char ret[32];
                                format_candidate_set(ret, candidates[i]);
                                if ( grid_state->stackpointer == 0 && unique_check_mode == 0 ) {
                                    if ( warnings != 0 ) {
                                        solverData.printf("Line %d: naked  set (row) %s at %s, count exceeded\n", line, ret, cl2txt[ri*9]);
                                    }
                                } else if ( debug ) {
                                    solverData.printf("back track sets (row) %s at %s, count exceeded\n", ret, cl2txt[ri*9]);
                                }
                            }
                            // no need to update grid_state
                            return Phase_Back;
                        } else if (s == cnt && cnt+2 <= ul) {
                            char ret[32];
                            int delta = ul-cnt;
                            if ( cnt <= 3 ) {
                                unsigned int row_box_intersection = 7<<(box_start[i]%9);
                                grid_state->set23_found[Row].set_indexbits(m,ri*9,9);
                                // adjust the count - maybe required lateron
                                sectionSetsUnlockedCnt[Row][ri] -= cnt;
                                // update box, if set within triad
                                if ( (m&row_box_intersection) == m ) {
                                    grid_state->set23_found[Box].set_indexbits(m&0x1ff,ri*9,9);
                                    int bi_now = sectionSetsUnlockedCnt[Box][box_index[i]];
                                    if ( bi_now >= cnt ) {
                                        sectionSetsUnlockedCnt[Box][box_index[i]] = bi_now - cnt;
                                    }
                                    add_indices<Box>(&to_change, i);
                                }
                            }

                            if ( delta <= 3 ) {
                                // could include locked slots
                                m_neg = 0x1ff & ~(m | grid_state->set23_found[Row].get_indexbits(ri*9,9));
                                int bi_neg = box_index[ri*9+__tzcnt_u32(m_neg)];
                                unsigned int row_box_intersection = 7<<(bi_neg%3*3);
                                grid_state->set23_found[Row].set_indexbits(m_neg,ri*9,9);
                                // adjust the count, maybe required lateron
                                sectionSetsUnlockedCnt[Row][ri] -= _popcnt32(m_neg);
                                // update box, if set within triad
                                if ( (m_neg&row_box_intersection) == m_neg ) {
    
                                    grid_state->set23_found[Box].set_indexbits(m_neg,ri*9,9);
                                    int bicnt_now = sectionSetsUnlockedCnt[Box][bi_neg];
                                    if ( bicnt_now >= cnt ) {
                                        sectionSetsUnlockedCnt[Box][bi_neg] = bicnt_now - cnt;
                                    }
                                }
                            }
                            if ( verbose != VNone ) {
                                counters.naked_sets_found++;
                            }
                            add_indices<Row>(&to_change, i);
                            if ( cnt==4 || delta==4 ) {
                                grid_state->flags |= (1<<ri);
                            }
                            if ( verbose == VDebug ) {
                                if ( cnt <=3 || cnt <= delta ) {
                                    format_candidate_set(ret, candidates[i]);
                                    solverData.printf("naked  %s (row): %-7s %s\n", s==2?"pair":"set ", ret, cl2txt[ri*9+i%9]);
                                } else {
                                    if (delta > 3) {
                                        m_neg = 0x1ff & ~(m | grid_state->set23_found[Row].get_indexbits(ri*9,9));
                                    }
                                    unsigned char k = 0xff;
                                    unsigned short complement = 0;
                                    while (m_neg) {
                                        unsigned char k_i = tzcnt_and_mask(m_neg);
                                        k_i += ri*9;
                                        if ( k == 0xff ) {
                                            k = k_i;
                                        }
                                        complement |= candidates[k_i];
                                    }
                                    complement &= ~candidates[i];
                                    format_candidate_set(ret, complement);
                                    solverData.printf("%s %s (row): %-7s %s\n", complement?"hidden":"naked ", __popcnt16(complement)==2?"pair":"set ", ret, cl2txt[ri*9+k%9]);
                                }
                            }
                        }
                    }
                } // row

                // check column and box
                //
                unsigned char ci = column_index[i];
                unsigned char b    = box_start[i];
                unsigned char bi   = box_index[i];

                const bool chk[2] = { !grid_state->set23_found[Col].check_indexbit(i),
                                      !grid_state->set23_found[Box].check_indexbit(i) };
                unsigned char uls[2];
                unsigned char ss[2] = {0,0};

                if ( chk[0] || chk[1] ) {
                    ul = uls[0] = sectionSetsUnlockedCnt[Col][ci];
                    uls[1] = sectionSetsUnlockedCnt[Box][bi];
                    if ( grid_state->flags & ((1<<9)<<ci) && uls[0] > 4) {
                        ul = uls[0] = ul<=8? 4:5;
                    }
                    if ( grid_state->flags & ((1<<18)<<bi) && uls[1] > 4) {
                        uls[1] = uls[1]<=8? 4:5;
                    }
                    if ( uls[1] > ul ) {
                        ul = uls[1];
                    }
                    if (check_back || (cnt+2 <= ul) ) {
                        __m256i a_j_256 = _mm256_set_epi16(candidates[b+19], candidates[b+18], candidates[b+11], candidates[b+10], candidates[b+9], candidates[b+2], candidates[b+1], candidates[b],
                                          candidates[ci+63], candidates[ci+54], candidates[ci+45], candidates[ci+36], candidates[ci+27], candidates[ci+18], candidates[ci+9], candidates[ci]);
                        __m256i res256 = _mm256_cmpeq_epi16(a_i, _mm256_or_si256(a_i, a_j_256));
                        unsigned int ms[2] = { compress_epi16_boolean<true>(res256), 0 };
                        ms[1] = ms[0] >> 16;
                        ms[0] &= 0xffff;
                        bool bit9s[2];
                        bit9s[0] = candidates[i] == (candidates[i] | candidates[ci+72]);
                        bit9s[1] = candidates[i] == (candidates[i] | candidates[b+20]);
                        const char *js[2] = {"col","box"};
                        for ( int j=0; j<2; j++) {  // check for back track first
                            if ( !chk[j] ) {
                                continue;
                            }
                            if (verbose != VNone ) {
                                counters.naked_sets_searched++;
                            }
                            if ( bit9s[j] ) {
                                ms[j] |= 3<<16;    // fake the 9th mask position
                            }
                            ss[j] = _popcnt32(ms[j])>>1;
                            // this covers the situation where there is a naked set of size x
                            // which is found in y cells with y > x.  That's impossible, hence track back.
                            if (ss[j] > cnt) {
                                if ( verbose != VNone ) {
                                    char ret[32];
                                    format_candidate_set(ret, candidates[i]);
                                    if ( grid_state->stackpointer == 0 && unique_check_mode == 0 ) {
                                        if ( warnings != 0 ) {
                                            solverData.printf("Line %d: naked  set (%s) %s at %s, count exceeded\n", line, js[j], ret, cl2txt[i]);
                                        }
                                    } else if ( debug ) {
                                        solverData.printf("back track sets (%s) %s at %s, count exceeded\n", js[j], ret, cl2txt[i]);
                                    }
                                }
                                // no need to update grid_state
                                return Phase_Back;
                            }
                        }
                        for ( int j=0; j<2; j++) {
                            if ( !chk[j] || (ss[j] != cnt) || (cnt+2 > uls[j])) {
                                continue;
                            }
                            // OK, this is getting a little tricky.
                            // Not only having to deal with columns and boxes, but also
                            // with the detected set and its complement -
                            // and to cast the trace in terms of pairs and sets
                            // while having to manage set23_found bit by bit.
                            //
                            if ( (ss[j] == cnt) && (cnt+2 <= uls[j]) ) {
                                if ( verbose != VNone ) {
                                    counters.naked_sets_found++;
                                }
                                if ( cnt==4 || uls[j]-cnt==4) {
                                    grid_state->flags |= (1<<9)<<(j?bi+9:ci);
                                }
                                if ( j ) {
                                    add_indices<Box>(&to_change, i);
                                } else {
                                    add_indices<Col>(&to_change, i);
                                }
                                unsigned char k = 0xff;
                                unsigned short complement = 0;
                                Kind kind = j?Box:Col;
                                bit128_t s = { *(bit128_t*)&big_index_lut[i][kind][0] & ~grid_state->set23_found[kind] };
                                bool set23_cond1 = (cnt <= 3);
                                bool set23_cond2 = (uls[j] <= cnt+3);
                                for ( unsigned char k_m = 0; k_m<9; k_m++ ) {
                                    unsigned char k_i = (j?(b+box_offset[k_m]):(ci+k_m*9));
                                    if ( !s.check_indexbit(k_i) ) {
                                        continue;
                                    }
                                    bool in_set = (candidates[k_i] | candidates[i]) == candidates[i];
                                        if ( (set23_cond1 && in_set) || (set23_cond2 && !in_set) ) {
                                            grid_state->set23_found[kind].set_indexbit(k_i);
                                        }
                                        if ( !in_set ) {
                                            // set k and compute the complement set
                                            if ( k == 0xff ) {
                                                k = k_i;
                                            }
                                            complement |= candidates[k_i];
                                        }
                                } // for
                                if ( verbose == VDebug ) {
                                    bool naked_anyway = !(complement & candidates[i]);
                                    char ret[32];
                                    complement &= ~candidates[i];
                                    if ( complement != 0 && set23_cond2 ) {
                                        format_candidate_set(ret, complement);
                                        solverData.printf("%s %s (%s): %-7s %s\n", naked_anyway?"naked ":"hidden", __popcnt16(complement)==2?"pair":"set ", js[j], ret, cl2txt[k]);
                                    } else {
                                        format_candidate_set(ret, candidates[i]);
                                        solverData.printf("naked  %s (%s): %-7s %s\n", cnt==2?"pair":"set ", js[j], ret, cl2txt[i]);
                                    }
                                }
                            }
                        } // for
                    }
                }

                to_change.u128 &= grid_state->unlocked.u128;
                const unsigned char *cip = index_by_i[i];
                // update candidates
                unsigned short cdi = candidates[i];
                unsigned short cdin = ~cdi;
                while (to_change) {
                    unsigned char j = tzcnt_and_mask(to_change);

                    // if this cell is not part of our set
                    if (candidates[j] & cdin ) {
                        // if there are bits that need removing
                        if (candidates[j] & cdi) {
                            candidates[j] &= cdin;
                            to_visit_again.set_indexbit(j);
                            found = true;
                        }
                    } else {
                        const unsigned char *cjp = index_by_i[j];
                        if ( (cjp[Col] == cip[Col]) && (ss[0] == cnt) && (ss[0] <= 3) ) {
                            grid_state->set23_found[Col].set_indexbit(j);
                        }
                        if ( (cjp[Box] == cip[Box]) && (ss[1] == cnt) && (ss[1] <= 3) ) {
                            grid_state->set23_found[Box].set_indexbit(j);
                        }
                    }
                }

                // If any cell's candidates got updated, go back and try all that other stuff again
                if (found) {
                    grid_state->updated.u128 = to_visit_n.u128 | tv.u128 | to_visit_again.u128;
                    return Phase_Search;
                }
            } else if ( cnt == 1 ) {
                // this is not possible, but just to eliminate cnt == 1:
                e_i = i;
                e_digit = candidates[i];
                if ( warnings != 0 ) {
                    solverData.printf("found a singleton in set search... strange\n");
                }
                if ( verbose == VDebug ) {
                    solverData.printf("naked  (sets) ");
                }
                return Phase_Enter;
            }
        } // while
      }

      grid_state->updated.u128 = to_visit_again.u128;
    }
#endif

#ifdef OPT_FSH
    if ( mode_fish )
    {
    // number grids (aka Fish (X-Wing (N==2), sword fish (N==3), jelly fish (N==4), squirmbag (N==5) )
    //
    // Theory (rows/cols only):
    // Looking at grids of N rows x N columns,  it is easy to see that under Sudoku rules
    // you can always place exactly N times the same digit on the grid.
    // This is also valid for grids of size one (single location) and size nine (full board).
    //
    // It is always true that for N==9 there must by nine locations where to place a given
    // digit.  As the game progresses and digits are placed, the availabe candidates are reduced
    // for each resolved candidate.
    // Similarly, it is easy to see that for each pick of N times the same digit in a valid
    // solution, these will form a grid N x N rows and columns.
    //
    // Given a sudoku board populated with candidates, the task is then to find such grids
    // such that they correspond to the solution of the Sudoku problem.
    //
    // The conditons for such a grid based on the candidate locations are very simply:
    // find N rows or columns for which all candidates are within N columns / rows.
    // (there is no guarantee to find any, but if found, it is a valid grid).
    // Having identified such a valid grid in terms of rows, any extra candidates in the columns
    // can be removed and vice versa.
    //
    // Fishes are always there:
    // If the puzzle is solvable, then for each subset of size K of cells of a row or column there
    // must exist K columns or rows to complete a fish pattern.  If there are not enough such columns
    // or rows, the puzzle is not solvable (on that path), so back track.
    //
    // fishes that need to be cleaned:
    // Given a base of size N, check whether there are:
    // - either N lines that match with a real (or assumed) base, and some extra
    //   lines that overlap with the base, then these extra line can be purged
    //   (their base portion removed) and the fish is complete,
    // - or exactly N lines that overlap with the base, then all these lines
    //   can be purged (their non-base portion removed).
    // Note that these two possibilities are the same, except the second is transposed.
    // 
    // The search operates just as the naked set search does, i.e. it will not discover
    // fishes where all lines have less than N candidates.  This can be remedied by augmenting
    // the set of lines using additional lines (by 1 additional line, which should go a long
    // way (a jelly fish can have 4 lines of 2 candidates each, but the value of such a search
    // needs to be confirmed).
    //
    // Once a grid has been found and cleaned, there are two distinct subsets
    // of fishes present, i.e. the fish and its complementary fish.
    // If either the fish or its complement have size of 3 or lower, they cannot
    // be subdivided and therefore need not be searched further.
    //
    // Variants:
    // All variants start with N-1 matching fish lines (subsets of regular fishes
    // as described above).
    //
    // Finned Fish:
    // identify from the excess lines overlapping the base, one that:
    // - has one or two extra cells, both in the same box,
    // - sharing a grid point in the same box.
    // Either one of the extra candidates is valid, or the fish pattern is valid.
    // Any candidates on the perpendicular grid lines that is not a grid point,
    // that can be seen by the grid point and the extra cells can be eliminated.
    //
    // Finned/sashimi Fish:
    // This is the same as a finned fish, but the grid point in the same box lacks
    // the candidate.
    //
    // Sashimi Fish:
    // This requires exactly two excess lines which must share a band.
    // The overlap with the base must be a single point
    // for each, resulting in two 'fins', which must not share a box.
    // As there are only two fins, one of them must complete the grid or the grid
    // would lack a line. They cannot both be valid either, as that would create
    // an invalid 'fish' again.
    // All candidates that can be seen by both fins can be eliminated.
    //  
    //
    // with the cbbvs in place, look for simple 'fishes'

    cbbv_t cbbv_v;

    bit128_t *candidate_bits_by_value = solverData.getCbbvs(candidates);

    for ( unsigned char dgt = 0; dgt < 9; dgt++) {
        // don't bother with six cells already resolved, a swordfish cannot be subdivided.
        // a jellyfish is final with 8 candidates (4 x 2), so 8 + 5 = 13 cannot be divided
        // unless it is already.
        // The minimum number of candidates to work with is 14. 
        if ( candidate_bits_by_value[dgt].popcount() < 14 ) {
            continue;
        }
        // load a vector eliminating solved cells
        bit128_t cbbv_digit = { .u128= candidate_bits_by_value[dgt].u128 & grid_state->unlocked.u128 };

        // load cbbv_v
        __m128i off0 = uncompress(cbbv_digit.m128);
        cbbv_v.m256 = _mm256_insert_epi16(_mm256_and_si256(mask1ff,_mm256_castsi128_si256(off0)), *(__int16*)&cbbv_digit.u8[9], 8);
        unsigned long long hi = cbbv_digit.u64[1];
#if 0
        unsigned long long lo = cbbv_digit.u64[0];

dump_m256i_grid(cbbv_v.m256, "new");
dump_m256i_grid(_mm256_and_si256(_mm256_setr_epi16(lo, lo>>9, lo>>18, lo>>27, lo>>36, lo>>45, lo>>54, (lo>>63)|(hi<<1), hi>>8, 0, 0, 0, 0, 0, 0, 0),mask1ff), "old");
#endif
        // do a quick popcnt of all rows or'ed together to determine the maximum fish size
        __m128i max_v = _mm256_castsi256_si128(_mm256_or_si256(cbbv_v.m256, _mm256_bsrli_epi128(cbbv_v.m256,8)));
        max_v = _mm_or_si128(max_v, _mm_bsrli_si128(max_v,4));
        max_v = _mm_or_si128(max_v, _mm_bsrli_si128(max_v,2));
        unsigned char max_all = __popcnt16(_mm_extract_epi16(max_v, 0) | hi>>8);
        unsigned char max = max_all > 7 ? 7 : max_all;
        bit128_t dgt_bits = { .u128 = candidate_bits_by_value[dgt].u128 & grid_state->unlocked.u128 };

        unsigned int pair_locs = 0;  // pairs, bits left to right
        unsigned int alt_base_x = 0;
        for (unsigned char t=0; t<9; t++, pair_locs <<= 1) { // t is the row the pattern is sampled from
            unsigned int base_x = alt_base_x? alt_base_x : cbbv_v.v16[t]; // base_x: tentative fish pattern
            // variables names reflect whether they are parallel (_prl) or perpendicular (_prp) to the base_x
            // also a suffix of _x signifies that the variable contains index bits
            if ( base_x & exclude_row[dgt] ) {
                continue;
            }
            // for the given digit, the count in a given row is N (cnt).
            // take each row of bits and compare to all other rows.
            //
            unsigned char cnt = __popcnt16(base_x);

            if ( cnt+2 <= max && cnt > 1) {  // maximum cnt is 5 ('squirmbag').
                if ( cnt == 2 ) {
                    pair_locs |= 1;
                }
                __m256i basev = _mm256_and_si256(_mm256_set1_epi16(base_x),mask9);
                // rows that have a non-empty intersection with base_x
                __m256i hassub_v = _mm256_cmpgt_epi16(_mm256_and_si256(basev, cbbv_v.m256),_mm256_setzero_si256());
                // rows that are a non-empty subset of base_x
                __m256i issub_v = _mm256_and_si256(hassub_v, _mm256_cmpeq_epi16(basev, _mm256_or_si256(basev, cbbv_v.m256)));

                unsigned int subs_prp_x = compress_epi16_boolean(issub_v); // bits for non-empty subsets of base_x
                unsigned int hassubs_prp_x = compress_epi16_boolean(hassub_v);  // bits for non-empty intersection with base_x
                unsigned char nsubs = _popcnt32(subs_prp_x);
                // test for naked fish and exclude it for row and col search
                if ( nsubs == cnt && _mm256_testc_si256(issub_v, hassub_v) ) {
                    // found a (naked) fish, leave it
                    if ( cnt <= 3 ) {
                        exclude_row[dgt] |= base_x;
                        exclude_col[dgt] |= subs_prp_x; // the orthogonal grid base
                        if ( verbose != VNone ) {
                            counters.fishes_excluded++;   // counted but not currently included in summary
                        }
                    }
                    continue;
                }
                if ( check_back && _popcnt32(hassubs_prp_x) < cnt ) {
                    if ( verbose == VDebug ) {
                        solverData.printf("back track - insufficient number of rows to form %s for digit %d based on row %d\n", fish_names[cnt-2], dgt+1, t);
                    }
                    return Phase_Back;
                }

                bit128_t clean_bits {};
                unsigned char fincells[2] = {0xff, 0xff};
                bool dosearch = false;

                // if there are exactly N rows with exclusively some of the pattern of digits, then
                // it is a row fish pattern.
                // if there are exactly N rows that have some of the pattern plus excess,
                // then it is a col fish pattern.
                if ( nsubs == cnt || _popcnt32(hassubs_prp_x) == cnt ) {
                    // this part is efficient to find and easy to deal with - however, the yield is very low.
                    unsigned short excess_x = nsubs == cnt ? base_x : (0x1ff & ~base_x);
                    if ( verbose != VNone ) {
                        counters.fishes_detected++;
                    }
                    unsigned int rows2clean = hassubs_prp_x & ~subs_prp_x;
                    while (rows2clean) {
                        unsigned char row = tzcnt_and_mask(rows2clean);
                        clean_bits.set_indexbits( cbbv_v.v16[row]&excess_x, row*9, 9);
                    }
                    clean_bits.u128 &= grid_state->unlocked.u128 & candidate_bits_by_value[dgt].u128;
                    if ( clean_bits ) {
                        if ( verbose == VDebug ) {
                            unsigned char celli  = __tzcnt_u16(subs_prp_x)*9 + __tzcnt_u16(base_x);
                            unsigned char celli2 = (15-__lzcnt16(subs_prp_x))*9 + (15-__lzcnt16(base_x));
                            if ( debug > 2 ) {
                                show_fish(cbbv_v, base_x, nsubs == cnt ? subs_prp_x:hassubs_prp_x, hassubs_prp_x, clean_bits, dgt_bits, solverData, "row fish");
                            }
                            solverData.printf("%s (%s) digit %d at cells %s - %s\nRemove %d at ", fish_names[cnt-2], nsubs==cnt?"rows":"cols", dgt+1, cl2txt[celli], cl2txt[celli2], dgt+1);
                        }
                        dosearch = true;
                    }
                    if ( cnt <= 3 ) {
                        exclude_row[dgt] |= base_x;
                        exclude_col[dgt] |= nsubs == cnt ? subs_prp_x:hassubs_prp_x; // the orthogonal grid base
                    } else if ( cnt >= max_all-3 ) {
                        exclude_row[dgt] |= 0x1ff & ~base_x;
                        exclude_col[dgt] |= 0x1ff & ~(nsubs == cnt ? subs_prp_x:hassubs_prp_x); // the orthogonal grid base
                    }

                } else if ( nsubs == cnt-1 ) {
                    // nsubs == cnt-1 is required for finned/sashimi fishes.
                    // There are two case, which are not mutually exclusive:
                    // Case 1:
                    // for a fin, the following must be true:
                    // The fin(s) cells must be on the perpendicular fish sections w.r.t. to the base_x.
                    // There must be exactly 2 fin sections.
                    // These fin cells are mutually exclusive and
                    // one would be required to complete the fish pattern and eliminate the other.

                    // A. This case requires exactly two fin sections.
                    // B. The respective fin cells must be unique on the perpendicular fish grid line
                    //    except for grid points.
                    // C. They must share a band perpendicular to the base_x and not share
                    //    a band with any established grid line parallel to the base.
                    // In this case the two fins' views intersect with each other to define
                    // two triads to be cleaned.
                    // Note: This arrangement can also be seen in most cases as an AIC or
                    // (in the case of an X-Wing) as a Skyscraper pattern.
                    // When logged, this case is identified as 'sashimi'
                    // Addendum to Case 1:
                    // A direct consequence for the band with the two mutual exclusive fins
                    // is the following:
                    // If for two cells in different boxes of the same band their value is
                    // mutually exclusively the same digit D, the triad T that does not share
                    // a box or row/col with either of the two cells, if:
                    // the value candidate is unique in the respective triads of these two cells,
                    // then the triad T cannot contain the digit D.
                    // This situation occurs frequently in conjunction with Case 1.
                    // As proof and for illustration:  The band triads (without loss of generality, a row band):
                    //       A1   A2   A3
                    //       B1   B2   B3
                    //       C1   C2   C3
                    // Without loss of generality assume that triads A1 and B2 contain cells that
                    // have a candidate value D.  We do know, that due to being linked exclusively,
                    // that neither A2 nor B1 can contain candidate value D (and the code makes sure of that).
                    // If A1 does not contain D, then A3 does.  Conversely, if B2 does not contain D,
                    // then B3 does.
                    // Therefore, since D is either in A3 or B3, D cannot be in C3.
                    // In the scenario of Case 1 we must simply look for the following conditions:
                    // a. the fins must not share a box,
                    // b. the fins must be the only candidate D in their respective triad.
                    // [ Note that this addendum is not part of the sashimi pattern directly as
                    //   spelled out by numerous sources on the Web ]. 
                    // Case 2:
                    // for a fin, the following must be true:
                    // The fin cell(s) must be on the parallel fish sections w.r.t. to the base_x.
                    // A. the fin cells must all be in the same box (and aligned if there are 2), and
                    // B. the same box must contain a fish grid point (which need not have a candidate)
                    // In this case the fin(s) views intersect with their associated fish grid point's
                    // view to provide the triad (minus the fish grid) to be cleaned.

                    hassubs_prp_x = compress_epi16_boolean(hassub_v);
                    unsigned int finsubs_prp_x = hassubs_prp_x & ~subs_prp_x; //& ~exclude_row;
                    bool sashimi = false;
                    unsigned int subsx= subs_prp_x;
                    unsigned int finsubcnt = _popcnt32(finsubs_prp_x);  // the complete count

                    if ( finsubcnt == 2 ) {   // Case 1 A
                        unsigned short fin_subs_prp[2] = { (unsigned short)_tzcnt_u32(finsubs_prp_x), (unsigned short)(31-__lzcnt32(finsubs_prp_x)) };
                        if (   ( fin_subs_prp[0]/3 == fin_subs_prp[1]/3 )) {     // Case 1 C
                            unsigned short fin_subs_pos_prl[2] = { __tzcnt_u16(cbbv_v.v16[fin_subs_prp[0]] & base_x),
                                                                   __tzcnt_u16(cbbv_v.v16[fin_subs_prp[1]] & base_x) };

                            if (    fin_subs_pos_prl[0] != fin_subs_pos_prl[1]                // Case 1 B
                                 && _popcnt32(cbbv_v.v16[fin_subs_prp[0]] & base_x) == 1
                                 && _popcnt32(cbbv_v.v16[fin_subs_prp[1]] & base_x) == 1 ) {
                                if ( verbose != VNone && reportstats ) {
                                    counters.fishes_detected++;
                                    counters.fishes_specials_detected++;
                                }
                                fincells[0] =  fin_subs_prp[0]*9 + fin_subs_pos_prl[0];
                                fincells[1] =  fin_subs_prp[1]*9 + fin_subs_pos_prl[1];
                                clean_bits.u128 =    (*(bit128_t*)big_index_lut[fincells[0]][All]).u128
                                                   & (*(bit128_t*)big_index_lut[fincells[1]][All]).u128;
                                // deal with 'addendum to Case 1'
                                if ( fincells[0]/3%3 != fincells[1]/3%3 ) { // not in same box
                                    if (    _popcnt32(candidate_bits_by_value[dgt].get_indexbits(fincells[0]-fincells[0]%3, 3)) == 1
                                         && _popcnt32(candidate_bits_by_value[dgt].get_indexbits(fincells[1]-fincells[1]%3, 3)) == 1) {
                                         unsigned char boxoff = 3 * _tzcnt_u32(7 ^ ((1<<fincells[0]/3%3) | (1<<fincells[1]/3%3)));
                                         unsigned char rowoff = 9 * ( _tzcnt_u32(7 ^ ((1<<fincells[0]/9%3) | (1<<fincells[1]/9%3)))
                                                                      + fin_subs_prp[0]/3*3 );
                                         clean_bits.set_indexbits(7, rowoff+boxoff, 3);
                                    }
                                }
                                clean_bits.u128 &= candidate_bits_by_value[dgt].u128
                                                   & grid_state->unlocked.u128;
                                clean_bits.unset_indexbit(fincells[0]);
                                clean_bits.unset_indexbit(fincells[1]);
                                if ( clean_bits ) {
                                    if ( verbose != VNone ) {
                                        counters.fishes_specials_updated++;
                                    }
                                    if ( verbose == VDebug ) {
                                        subsx = subs_prp_x | (1<<fin_subs_prp[0]) | (1<<fin_subs_prp[1]);
                                        unsigned char celli = __tzcnt_u16(subsx)*9 + __tzcnt_u16(base_x);
                                        unsigned char celli2 = (15-__lzcnt16(subsx))*9 + (15-__lzcnt16(base_x));
                                        if ( debug > 2 ) {
                                            show_fish(cbbv_v, subs_prp_x, 2, fincells, clean_bits, dgt_bits, solverData, "finned row fish");
                                        }
                                        solverData.printf("sashimi %s (rows) digit %d at cells %s - %s\nFins at %s,%s - remove %d at ", fish_names[cnt-2], dgt+1, cl2txt[celli], cl2txt[celli2], cl2txt[fincells[0]], cl2txt[fincells[1]], dgt+1);
                                    }
                                }
                            }
                        }

                        if ( clean_bits ) {
                            if ( verbose != VNone ) {
                                counters.fishes_updated++;
                            }
                            dgt_bits.u128 &= ~clean_bits.u128; // for Case 2
                            unsigned short dgt_mask_bit = 1<<dgt;
                            while (clean_bits) {
                                unsigned char cl = tzcnt_and_mask(clean_bits);
                                if ( candidates[cl] & dgt_mask_bit ) {
                                    candidates[cl] &= ~dgt_mask_bit;
                                    if ( verbose == VDebug ) {
                                        solverData.printf("%s ", cl2txt[cl]);
                                    }
                                }
                            }
                            if ( verbose == VDebug ) {
                                solverData.printf("\n");
                            }
                            dosearch = true;
                        }
                    }

                    {
                        // Case 2
                        // iterate over possible fin rows
                        while ( finsubs_prp_x ) {
                            // pick one possible fin row
                            unsigned char fin_sub = tzcnt_and_mask(finsubs_prp_x);
                            unsigned short finln_prl_x = cbbv_v.v16[fin_sub];
                            unsigned short fins = finln_prl_x & ~base_x;

                            unsigned short box_bits = bandbits_by_index[__tzcnt_u16(fins)/3];  // the box to compare with
                            // grid point in box? fins in box?    
                            // In this first case the fin cell view is intersected with
                            // a grid point view box.
                            if (  (fins & ~box_bits) || !(box_bits & base_x) ) {   // Case 2, A and B
                                continue;
                            }
                            // if the fin has no candidate on its associated grid point, it's a sashimi
                            // for information only, as it's the common nomenclature.
                            sashimi = !(finln_prl_x & base_x & box_bits);

                            // complete the finned fish pattern:
                            subsx = subs_prp_x | (1<<fin_sub);

                            // the rows left for cleaning
                            unsigned short band_bits = bandbits_by_index[fin_sub/3];
                            unsigned int rows2clean = hassubs_prp_x & ~subsx & band_bits;
                            if ( rows2clean == 0 ) {
                                // the sashimi X-wing base_x is a good guess to keep for later:
                                if ( solverData.guess_hint_digit == 0 && !clean_bits && sashimi && alt_base_x==0 && cnt == 2 ) {
                                    solverData.guess_hint_digit = 1<<dgt;
                                    solverData.guess_hint_index = t*9 + __tzcnt_u16(base_x & ~box_bits);
                                }
                                continue;
                            }
                            if ( verbose != VNone ) {
                                counters.fishes_specials_detected++;
                                counters.fishes_detected++;
                            }

                            unsigned int boxbase_x = base_x & box_bits;
                            clean_bits.u128 = 0;

                            while (rows2clean) {
                                unsigned char row = tzcnt_and_mask(rows2clean);
                                clean_bits.set_indexbits( cbbv_v.v16[row]&boxbase_x, row*9, 9);
                            }
                            clean_bits.u128 &= dgt_bits.u128;

                            if ( clean_bits ) {
                                if ( verbose != VNone ) {
                                    counters.fishes_specials_updated++;
                                }
                                if ( verbose == VDebug ) {
                                    unsigned char celli = __tzcnt_u16(subsx)*9 + __tzcnt_u16(base_x);
                                    unsigned char celli2 = (15-__lzcnt16(subsx))*9 + (15-__lzcnt16(base_x));
                                    if ( debug > 2 ) {
                                        show_fish(cbbv_v, base_x, subsx, hassubs_prp_x, clean_bits, dgt_bits, solverData, "finned row fish");
                                    }
                                    solverData.printf("finned%s %s (rows) digit %d at cells %s - %s\nFin at %s - remove %d at ", sashimi?"/sashimi":"", fish_names[cnt-2], dgt+1, cl2txt[celli], cl2txt[celli2], cl2txt[fin_sub*9+__tzcnt_u16(fins&~base_x)], dgt+1);
                                }
                                break;
                            }
                        } // while
                    } // Case 2
                } // else
                if ( clean_bits ) {
                    if ( verbose != VNone ) {
                        counters.fishes_updated++;
                    }
                    if ( clean_bits.check_indexbit(fincells[0]) ) {
                        e_digit = 1<<dgt;
                        e_i = fincells[1];
                    } else if ( clean_bits.check_indexbit(fincells[1]) ) {
                        e_digit = 1<<dgt;
                        e_i = fincells[0];
                    }
                    unsigned short dgt_mask_bit = 1<<dgt;
                    while (clean_bits) {
                        unsigned char cl = tzcnt_and_mask(clean_bits);
                        if ( candidates[cl] & dgt_mask_bit ) {
                            candidates[cl] &= ~dgt_mask_bit;
                            if ( verbose == VDebug ) {
                                solverData.printf("%s ", cl2txt[cl]);
                            }
                        }
                    }
                    if ( verbose == VDebug ) {
                        if ( e_digit == 0 ) {
                            solverData.printf("\n");
                        } else {
                            solverData.printf("\ncells %s and %s are mutually exclusive (sashimi %s on digit %d),\nenter the remaining ", cl2txt[fincells[0]], cl2txt[fincells[1]], fish_names[cnt-2], dgt+1);
                        }
                    }
                    if ( e_digit ) {
                        return Phase_Enter;
                    }
                    dosearch = true;
                }
                if ( dosearch ) {
                    return Phase_Search;
                }
            } // if
            // scan for two bi-values forming a triple...
            // just take a single guess with this - it will also catch fin/sashimi
            unsigned int pair_cnt = __popcnt16(pair_locs);
            bool fish_alt_found = false;
            if ( t == 8 && pair_cnt >= 3 ) {
                unsigned char pos[9];
                for ( int i=pair_cnt-1; pair_locs; i-- ) {
                    pos[i] = 8-tzcnt_and_mask(pair_locs);
                }
                unsigned short res = 0;
                for ( unsigned int i=0; i<pair_cnt-1 && !fish_alt_found; i++) {
                    for ( unsigned int k=i+1; k<pair_cnt; k++ ) {
                        if ( ( __popcnt16(res = cbbv_v.v16[pos[i]] | cbbv_v.v16[pos[k]])) == 3 ) {
                            alt_base_x = res;
                            fish_alt_found = true;
                            break;
                        }
                    }
                }
            }
            if ( !fish_alt_found ) {
                continue;
            }
            // an alternative pattern was synthesised; replay this iteration
            // with t pinned at 7 and pair_locs cleared so we don't loop here.
            t = 7;
            pair_locs = 0;
        } // for t

        // part 2: look for fishes at columns

        // transpose cbbv_v
        cbbv_t cbbv_col_v {};

        unsigned short *mskp = &cbbv_col_v.v16[8];
        __m256i c = _mm256_srli_epi16(cbbv_v.m256,1);
        *mskp-- = _mm_movemask_epi8(_mm_packus_epi16(_mm256_castsi256_si128(c),_mm256_extracti128_si256(c,1)));
        c = _mm256_and_si256(cbbv_v.m256, maskff);
        __m128i cc = _mm_packus_epi16(_mm256_castsi256_si128(c),_mm256_extracti128_si256(c,1));
        for (unsigned char d = 8; d > 0; d--) {
            *mskp-- = _mm_movemask_epi8(cc);
            cc = _mm_slli_epi16(cc,1);
        }

        pair_locs = 0;  // pairs, bits left to right
        alt_base_x = 0;
        for (unsigned char t=0; t<9; t++, pair_locs <<= 1) { // t is the row the pattern is sampled from
            unsigned int base_x = alt_base_x? alt_base_x : cbbv_col_v.v16[t]; // base_x: tentative fish pattern

            if ( base_x & exclude_col[dgt] ) { // base_x: tentative fish pattern
                continue;
            }
            // for the given digit, the count in a given col is N (cnt).
            // take each col of bits and compare to all other cols.
            //
            unsigned char cnt = __popcnt16(base_x);
            if ( cnt == 2 ) {
                pair_locs |= 1;
            }

            if ( cnt+2 <= max && cnt > 1) {  // maximum cnt is 5 ('squirmbag').
                __m256i basev = _mm256_and_si256(_mm256_set1_epi16(base_x),mask9);
                // rows that have a non-empty intersection with base_x
                __m256i hassub_v = _mm256_cmpgt_epi16(_mm256_and_si256(basev, cbbv_col_v.m256),_mm256_setzero_si256());
                // rows that are a non-empty subset of base_x
                __m256i issub_v = _mm256_and_si256(hassub_v, _mm256_cmpeq_epi16(basev, _mm256_or_si256(basev, cbbv_col_v.m256)));

                unsigned int subs_prp_x = 0x1ff & compress_epi16_boolean(issub_v);
                unsigned char nsubs = _popcnt32(subs_prp_x);
                unsigned int hassubs_prp_x = compress_epi16_boolean(hassub_v);

                if ( nsubs == cnt && _mm256_testc_si256(issub_v, hassub_v) ) {
                    // found a (naked) fish
                    // for naked fishes, don't double count row/col detection
                    // neither need to exclude anything;
                    // fishes_detected++;
                    continue;
                }
                if ( check_back && _popcnt32(hassubs_prp_x) < cnt ) {
                    if ( verbose == VDebug ) {
                        solverData.printf("back track - insufficient number of cols to form %s for digit %d based on row %d\n", fish_names[cnt-2], dgt+1, t);
                    }
                    return Phase_Back;
                }

                bit128_t clean_bits {};
                bit128_t tmp {};
                unsigned char fincells[2] = {0xff, 0xff};
                bool dosearch = false;

                // if there are exactly N cols with exclusively some of the pattern of digits, then
                // it is a col fish pattern.
                // if there are exactly N cols that have some of the pattern plus excess,
                // then it is a row fish pattern.
                if ( nsubs == cnt || _popcnt32(hassubs_prp_x) == cnt ) {
                    // this part is efficient to find and easy to deal with - however the yield very low.
                    unsigned short excess_x = nsubs == cnt ? base_x : (0x1ff & ~base_x);
                    if ( verbose != VNone ) {
                        counters.fishes_detected++;
                    }
                    hassubs_prp_x = compress_epi16_boolean(hassub_v);
                    unsigned int cols2clean = hassubs_prp_x & ~subs_prp_x;
                    while (cols2clean) {
                        unsigned char col = tzcnt_and_mask(cols2clean);
                        tmp.set_indexbits( cbbv_col_v.v16[col]&excess_x, col*9, 9);
                    }
                    // since we work in a transposed view, we transpose clean_bits here:
                    while ( tmp ) {
                        clean_bits.set_indexbit(transposed_cell[tzcnt_and_mask(tmp)]);
                    }
                    clean_bits.u128 &= dgt_bits.u128;
                    if ( clean_bits.u128 ) {
                        if ( verbose == VDebug ) {
                            if ( debug > 2 ) {
                                show_fish<true>(cbbv_col_v, base_x, nsubs == cnt ? subs_prp_x:hassubs_prp_x, hassubs_prp_x, clean_bits, dgt_bits, solverData, "col fish");
                            }
                            unsigned char celli = __tzcnt_u16(subs_prp_x) + __tzcnt_u16(base_x)*9;
                            unsigned char celli2 = (15-__lzcnt16(subs_prp_x)) + (15-__lzcnt16(base_x))*9;
                            solverData.printf("%s (%s) digit %d cells %s - %s\nRemove %d at ", fish_names[cnt-2], nsubs == cnt? "cols":"rows", dgt+1, cl2txt[celli], cl2txt[celli2], dgt+1);
                        }
                        dosearch = true;
                    }
                    if ( cnt <= 3 ) {
                        exclude_col[dgt] |= base_x;
                    }
                } else if ( nsubs == cnt-1 ) {
                    // nsubs == cnt-1 is required for both fin and sashimi fishes.

                    // for a fin, the following must be true:
                    // 1. the fin cells not on the fish grid must all be in the same box
                    // 2. the same box must contain a fish grid point (which need not have a candidate)

                    // iterate over possible fin cols
                    hassubs_prp_x = compress_epi16_boolean(hassub_v);
                    unsigned int finsubs_prp_x = hassubs_prp_x & ~subs_prp_x ; //& ~exclude_col;
                    bool sashimi = false;
                    unsigned int subsx= subs_prp_x;
                    unsigned int finsubcnt = _popcnt32(finsubs_prp_x);  // the complete count

                    if ( finsubcnt == 2 ) {   // Case 1 A
                        unsigned short fin_subs_prp[2] = { (unsigned short)_tzcnt_u32(finsubs_prp_x), (unsigned short)(31-__lzcnt32(finsubs_prp_x)) };
                        if (   fin_subs_prp[0]/3 == fin_subs_prp[1]/3 ) {    // Case 1 C
                            unsigned short fin_subs_pos_prl[2] = { __tzcnt_u16(cbbv_col_v.v16[fin_subs_prp[0]] & base_x),
                                                                    __tzcnt_u16(cbbv_col_v.v16[fin_subs_prp[1]] & base_x) };
                            if (    fin_subs_pos_prl[0] != fin_subs_pos_prl[1]                // Case 1 B
                                 && _popcnt32(cbbv_col_v.v16[fin_subs_prp[0]] & base_x) == 1
                                 && _popcnt32(cbbv_col_v.v16[fin_subs_prp[1]] & base_x) == 1 ) {
                                if ( verbose != VNone ) {
                                    counters.fishes_detected++;
                                    counters.fishes_specials_detected++;
                                }
                                fincells[0] =  fin_subs_prp[0] + fin_subs_pos_prl[0]*9;
                                fincells[1] =  fin_subs_prp[1] + fin_subs_pos_prl[1]*9;
                                clean_bits.u128 =    (*(bit128_t*)big_index_lut[fincells[0]][All]).u128
                                                   & (*(bit128_t*)big_index_lut[fincells[1]][All]).u128;
                                // deal with 'addendum to Case 1'
                                if ( fincells[0]/27 != fincells[1]/27 ) { // not in same (vertical) box
                                    if (    _popcnt32(candidate_bits_by_value[dgt].get_indexbits(fincells[0]/27*27 + fincells[0]%9, 19) & 0x40201) == 1
                                         && _popcnt32(candidate_bits_by_value[dgt].get_indexbits(fincells[1]/27*27 + fincells[1]%9, 19) & 0x40201) == 1) {
                                         unsigned char boxoff = fincells[0]/3%3*3 + _tzcnt_u32(7 ^ ((1<<fincells[0]%3) | (1<<fincells[1]%3)));
                                         unsigned char coloff = 27*_tzcnt_u32(7^((1<<fincells[0]/27) | (1<<fincells[1]/27)));
                                         clean_bits.set_indexbits(0x40201, coloff+boxoff, 19);
                                    }
                                }
                                clean_bits.u128 &= candidate_bits_by_value[dgt].u128
                                                & grid_state->unlocked.u128;
                                clean_bits.unset_indexbit(fincells[0]);
                                clean_bits.unset_indexbit(fincells[1]);
                                if ( clean_bits ) {
                                    if ( verbose != VNone ) {
                                        counters.fishes_specials_updated++;
                                    }
                                    if ( verbose == VDebug ) {
                                        subsx = subs_prp_x | (1<<fin_subs_prp[0]) | (1<<fin_subs_prp[1]);
                                        unsigned char celli = __tzcnt_u16(subsx) + __tzcnt_u16(base_x)*9;
                                        unsigned char celli2 = (15-__lzcnt16(subsx)) + (15-__lzcnt16(base_x))*9;
                                        if ( debug > 2 ) {
                                            show_fish<true>(cbbv_col_v, subs_prp_x, 2, fincells, clean_bits, dgt_bits, solverData, "finned col fish");
                                        }
                                        solverData.printf("sashimi %s (cols) digit %d at cells %s - %s\nFins at %s,%s - remove %d at ", fish_names[cnt-2], dgt+1, cl2txt[celli], cl2txt[celli2], cl2txt[fincells[0]], cl2txt[fincells[1]], dgt+1);
                                    }
                                }
                            }
                        }

                        if ( clean_bits ) {
                            if ( verbose != VNone ) {
                                counters.fishes_updated++;
                            }
                            dgt_bits.u128 &= ~clean_bits.u128; // for Case 2
                            unsigned short dgt_mask_bit = 1<<dgt;
                            while (clean_bits) {
                                unsigned char cl = tzcnt_and_mask(clean_bits);
                                if ( candidates[cl] & dgt_mask_bit ) {
                                    candidates[cl] &= ~dgt_mask_bit;
                                    if ( verbose == VDebug ) {
                                        solverData.printf("%s ", cl2txt[cl]);
                                    }
                                }
                            }
                            if ( verbose == VDebug ) {
                                solverData.printf("\n");
                            }
                            dosearch = true;
                        }
                    }

                    {
                        // Case 2
                        // iterate over possible fin rows
                        while ( finsubs_prp_x ) {
                            // pick one possible fin col
                            unsigned char fin_sub = tzcnt_and_mask(finsubs_prp_x);
                            unsigned short finln_prl_x = cbbv_col_v.v16[fin_sub];

                            unsigned short fins = finln_prl_x & ~base_x;
                            unsigned short box_bits = bandbits_by_index[__tzcnt_u16(fins)/3];  // the box to compare with
                            // fins in box? grid point in box?
                            if (  (fins & ~box_bits) || !(box_bits & base_x) ) {
                                continue;
                            }

                            sashimi = !(finln_prl_x & base_x & box_bits);
                            // complete the finned fish pattern:
                            unsigned int subsx = subs_prp_x | (1<<fin_sub);

                            // the cols left for cleaning
                            unsigned short band_bits = bandbits_by_index[fin_sub/3];
                            unsigned int cols2clean = hassubs_prp_x & ~subsx & band_bits;
                            if ( cols2clean == 0 ) {
                                // the sashimi X-wing base is a good guess:
                                if ( solverData.guess_hint_digit == 0 && !clean_bits && sashimi && alt_base_x==0 && cnt == 2 ) {
                                    solverData.guess_hint_digit = 1<<dgt;
                                    solverData.guess_hint_index = t + __tzcnt_u16(base_x & ~box_bits)*9;
                                }
                                continue;
                            }
                            if ( verbose != VNone ) {
                                counters.fishes_specials_detected++;
                                counters.fishes_detected++;
                            }

                            unsigned int boxbase_x = base_x & box_bits;

                            clean_bits.u128 = 0;
                            bit128_t tmp {};
                            while (cols2clean) {
                                unsigned char col = tzcnt_and_mask(cols2clean);
                                tmp.set_indexbits( cbbv_col_v.v16[col]&boxbase_x, col*9, 9);
                            }
                            // since we work in a transposed view, we transpose clean_bits first
                            while ( tmp ) {
                                clean_bits.set_indexbit(transposed_cell[tzcnt_and_mask(tmp)]);
                            }
                            clean_bits.u128 &= dgt_bits.u128;

                            if ( clean_bits ) {
                                if ( verbose != VNone ) {
                                    counters.fishes_specials_updated++;
                                }
                                if ( verbose == VDebug ) {
                                    if ( debug > 2 ) {
                                        show_fish<true>(cbbv_col_v, base_x, subsx, hassubs_prp_x, clean_bits, dgt_bits, solverData, "finned col fish");
                                    }
                                    unsigned char celli = __tzcnt_u16(subsx) + __tzcnt_u16(base_x)*9;
                                    unsigned char celli2 = (15-__lzcnt16(subsx)) + (15-__lzcnt16(base_x))*9;
                                    solverData.printf("finned%s %s (cols) digit %d at cells %s - %s\nFin at %s - remove %d at ", sashimi?"/sashimi":"", fish_names[cnt-2], dgt+1, cl2txt[celli], cl2txt[celli2], cl2txt[fin_sub+__tzcnt_u16(fins&~base_x)*9], dgt+1);
                                }
                                break;
                            }
                        } // while
                    }  // Case 2
                }
                if ( clean_bits ) {
                    if ( verbose != VNone ) {
                        counters.fishes_updated++;
                    }
                    if ( clean_bits.check_indexbit(fincells[0]) ) {
                        e_digit = 1<<dgt;
                        e_i = fincells[1];
                    } else if ( clean_bits.check_indexbit(fincells[1]) ) {
                        e_digit = 1<<dgt;
                        e_i = fincells[0];
                    }
                    unsigned short dgt_mask_bit = 1<<dgt;
                    while (clean_bits) {
                        unsigned char cl = tzcnt_and_mask(clean_bits);
                        if ( candidates[cl] & dgt_mask_bit) {
                            candidates[cl] &= ~dgt_mask_bit;
                            if ( verbose == VDebug ) {
                                solverData.printf("%s ", cl2txt[cl]);
                            }
                        }
                    }
                    if ( verbose == VDebug ) {
                        if ( e_digit == 0 ) {
                            solverData.printf("\n");
                        } else {
                            solverData.printf("\ncells %s and %s are mutually exclusive (sashimi %s on digit %d),\nenter the remaining ", cl2txt[fincells[0]], cl2txt[fincells[1]], fish_names[cnt-2], dgt+1);
                        }
                    }
                    if ( e_digit ) {
                        return Phase_Enter;
                    }
                    dosearch = true;
                }
                if ( dosearch ) {
                    return Phase_Search;
                }
            }
            // scan for two bi-values forming a triple...
            // just take a single guess with this - it will also catch fin/sashimi
            unsigned int pair_cnt = __popcnt16(pair_locs);
            if ( t == 8 && pair_cnt >= 3 ) {
                unsigned char pos[9];
                for ( int i=pair_cnt-1; pair_locs; i-- ) {
                    pos[i] = 8-tzcnt_and_mask(pair_locs);
                }
                unsigned short res = 0;
                for ( unsigned int i=0; i<pair_cnt-1; i++) {
                    for ( unsigned int k=i+1; k<pair_cnt; k++ ) {
                        if ( ( __popcnt16(res = cbbv_col_v.v16[pos[i]] | cbbv_col_v.v16[pos[k]])) == 3 ) {
                            alt_base_x = res;
                            goto done2;
                        }
                    }
                }
            }
            continue;
done2:
            t = 7;  // one iteration with the made-up data
            pair_locs = 0; // will skip this section next time...
        } // for t
    } // for dgt
    } // mode_fish
#endif

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
                                            return Phase_Enter;
                                        } else if ( (candidates[indx2upd[1]] & (candidates[indx2upd[1]] - 1)) == 0 ) {
                                            e_digit = candidates[indx2upd[1]];
                                            e_i = indx2upd[1];
                                            if ( verbose == VDebug ) {
                                                solverData.printf("naked  single      ");
                                            }
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
        return Phase_Guess;
}
