// Out-of-class definition of `SolveCtx<verbose>::do_naked_sets_main()`.
// AlphaEvolve mutation unit: this file is the entire surface for Algorithm 4 — naked sets (main variant, OPT_SETS).
// Replace the function body to mutate the strategy without touching
// the rest of the solver.
//
// Returns Phase_HiddenSearch when the block finishes without firing
// any redirect (the dispatcher then continues to the next block in
// phase_hidden_search). Any other returned SolverPhase short-circuits
// back to the dispatcher (semantically identical to the pre-refactor
// `goto X` exits inside the block).
//
// Build flag: this body is empty when OPT_SETS is undefined; the helper
// then unconditionally returns Phase_HiddenSearch.
//
// CONTRACT: private include fragment, must be #included exactly once
// from inside `namespace Schoku { ... }` after solver/solve_ctx.hpp.
#pragma once


template <Verbosity verbose>
__attribute__((always_inline)) inline SolverPhase SolveCtx<verbose>::do_naked_sets_main() {
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
                trace::next_entry_reason = trace::ER_DeducedSingle;  // Phase 2: refine to fish/set/ur reason
                return Phase_Enter;
            }
        } // while
      }

      grid_state->updated.u128 = to_visit_again.u128;
    }
#endif
    return Phase_HiddenSearch;
}
