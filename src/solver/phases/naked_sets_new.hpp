// Out-of-class definition of `SolveCtx<verbose>::do_naked_sets_new()`.
// AlphaEvolve mutation unit: this file is the entire surface for Algorithm 4 — naked sets (NEW variant, OPT_NEWSETS).
// Replace the function body to mutate the strategy without touching
// the rest of the solver.
//
// Returns Phase_HiddenSearch when the block finishes without firing
// any redirect (the dispatcher then continues to the next block in
// phase_hidden_search). Any other returned SolverPhase short-circuits
// back to the dispatcher (semantically identical to the pre-refactor
// `goto X` exits inside the block).
//
// Build flag: this body is empty when OPT_NEWSETS is undefined; the helper
// then unconditionally returns Phase_HiddenSearch.
//
// CONTRACT: private include fragment, must be #included exactly once
// from inside `namespace Schoku { ... }` after solver/solve_ctx.hpp.
#pragma once


template <Verbosity verbose>
__attribute__((always_inline)) inline SolverPhase SolveCtx<verbose>::do_naked_sets_new() {
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
    return Phase_HiddenSearch;
}
