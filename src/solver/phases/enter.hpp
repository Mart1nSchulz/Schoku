// Out-of-class definition of `SolveCtx<verbose>::phase_enter()`.
// AlphaEvolve mutation unit: this file is the entire surface for enter-digit (Algorithm 1)
// strategy; replace the function body to mutate the strategy without
// touching the rest of the solver.
//
// CONTRACT: private include fragment, must be #included exactly once
// from inside `namespace Schoku { ... }` after solver/solve_ctx.hpp.
#pragma once


template <Verbosity verbose>
__attribute__((always_inline)) inline SolverPhase SolveCtx<verbose>::phase_enter() {

    {
        // inlined flavor of enter_digit
        //
        // lock this cell and and remove this digit from the candidates in this row, column and box
        // and for good measure, detect 0s (back track) and singles.
        bit128_t to_update;

        if ( verbose == VDebug ) {
            solverData.printf(" %x at %s\n", _tzcnt_u32(e_digit)+1, cl2txt[e_i]);
        }
        // JSON trace: emit the placement event with the reason marker set
        // by the upstream phase. ER_None is a defensive fallback (we get
        // here only via Phase_Enter, which is always preceded by a reason
        // store — but cold paths might land here e.g. during unique-check
        // mode without a marker, so we don't assert).
        if ( trace::current ) {
            int row = e_i / 9;
            int col = e_i % 9;
            int value = _tzcnt_u32(e_digit) + 1;
            int level = grid_state->stackpointer;
            switch (trace::next_entry_reason) {
            case trace::ER_NakedSingle:
                trace::naked_single(row, col, value, level);
                break;
            case trace::ER_HiddenSingleRow:
                trace::hidden_single(row, col, value, 'r', level);
                break;
            case trace::ER_HiddenSingleCol:
                trace::hidden_single(row, col, value, 'c', level);
                break;
            case trace::ER_HiddenSingleBox:
                trace::hidden_single(row, col, value, 'b', level);
                break;
            case trace::ER_DeducedSingle:
                trace::deduced_single(row, col, value, level);
                break;
            case trace::ER_Guess:
                // Guess event itself is emitted by make_guess() before this
                // entry runs; entering the guessed digit here is the
                // mechanical placement, not a separate step.
                break;
            default:
                // ER_None or unknown — emit as deduced rather than naked so
                // a stale TLS from a hypothetical future refactor doesn't
                // poison the training signal as a "real" naked single.
                trace::deduced_single(row, col, value, level);
                break;
            }
            trace::next_entry_reason = trace::ER_None;
        }
#ifndef NDEBUG
        if ( __popcnt16(e_digit) != 1 ) {
            if ( warnings != 0 ) {
                solverData.printf("error in e_digit: %x\n", e_digit);
            }
        }
#endif

        if (e_i < 64) {
            _bittestandreset64((long long int *)&unlocked[0], e_i);
        } else {
            _bittestandreset64((long long int *)&unlocked[1], e_i-64);
        }

        candidates[e_i] = e_digit;
        current_entered_count++;

        set_indices<All>(&to_update, e_i);

        grid_state->updated.u128 |= to_update.u128;

        __m256i mask_neg = _mm256_set1_epi16(e_digit);

        unsigned short dtct_j = 0;
        unsigned int dtct_m = 0;
        for (unsigned char j = 0; j < 80; j += 16) {
            __m256i c = _mm256_load_si256((__m256i*) &candidates[j]);
            // expand unlocked unsigned short to boolean vector
            __m256i munlocked = expand_bitvector(to_update.u16[j>>4]);
            // apply mask (remove bit)
            c = andnot_if(c, mask_neg, munlocked);
            _mm256_store_si256((__m256i*) &candidates[j], c);
            __m256i a = _mm256_cmpeq_epi16(_mm256_and_si256(c, _mm256_sub_epi16(c, ones)), _mm256_setzero_si256());
            // this if is only taken very occasionally, branch prediction
            if (__builtin_expect (check_back && _mm256_movemask_epi8(
                                  _mm256_cmpeq_epi16(c, _mm256_setzero_si256())
                                  ), 0)) {
                // Back track, no solutions along this path
                if ( verbose != VNone ) {
                    unsigned int mx = _mm256_movemask_epi8(_mm256_cmpeq_epi16(c, _mm256_setzero_si256()));
                    unsigned char pos = j+(_tzcnt_u32(mx)>>1);
                    if ( grid_state->stackpointer == 0 && unique_check_mode == 0 ) {
                        if ( warnings != 0 ) {
                            solverData.printf("Line %d: cell %s is 0\n", line, cl2txt[pos]);
                        }
                    } else if ( debug ) {
                        solverData.printf("back track - cell %s is 0\n", cl2txt[pos]);
                    }
                }
                e_digit=0;
                return Phase_Back;
            }
            unsigned int mask = and_compress_masks<false>(a, grid_state->unlocked.u16[j>>4]);
            if ( mask ) {
                dtct_m = mask;
                dtct_j = j;
            }
        }
        if (unlocked[1] & (1ULL << (80-64)) ) {
            if ( to_update.u64[1] & (1ULL << (80-64)) ) {
                candidates[80] &= ~e_digit;
                if (__builtin_expect (candidates[80] == 0,0) ) {
                    // no solutions go back
                    if ( verbose != VNone ) {
                        if ( grid_state->stackpointer == 0 && unique_check_mode == 0 ) {
                            if ( warnings != 0 ) {
                                solverData.printf("Line %d: cell %s is 0\n", line, cl2txt[80]);
                            }
                        } else if ( debug ) {
                            solverData.printf("back track - cell %s is 0\n", cl2txt[80]);
                        }
                    }
                    return Phase_Back;
                }
            }
            if ( __popcnt16(candidates[80]) == 1) {
                // Enter the digit and update candidates
                if ( verbose == VDebug ) {
                    solverData.printf("naked  single      ");
                }
                e_i = 80;
                e_digit = candidates[80];
                trace::next_entry_reason = trace::ER_NakedSingle;
                return Phase_Enter;
            }
        }
        if ( dtct_m ) {
            int idx = _tzcnt_u32(dtct_m);
            e_i = idx+dtct_j;
            e_digit = candidates[e_i];
            if ( verbose == VDebug ) {
                solverData.printf("naked  single      ");
            }
            trace::next_entry_reason = trace::ER_NakedSingle;
            return Phase_Enter;
        }
        e_digit = 0;
    }

    // The solving algorithm ends when there are no remaining unlocked cells.
    // The finishing tasks include verifying the solution and/or confirming
    // its uniqueness, if requested.
    //
    // Check if it's solved, if it ever gets solved it will be solved after looking for naked singles
    if ( *(__uint128_t*)unlocked == 0) {
        bool verify_one = false;
        // Solved it
        if ( rules == Multiple && (unique_check_mode == 1 || grid_state->flags < 0) ) {
            if ( !nonunique_reported ) {
                if ( verbose != VNone && reportstats && warnings != 0 ) {
                    solverData.printf("Line %d: solution to puzzle is not unique\n", line);
                }
                nonunique_reported = true;
            }
            verify_one = true;
            status.unique = false;
        }
        if ( verify || verify_one) {
            verify_one = false;
            // quickly assert that the solution is valid
            // no cell has more than one digit set
            // all rows, columns and boxes have all digits set.

            __m256i rowx;
            __m256i colx;
            __m256i boxx;
            boxx = colx = rowx = _mm256_and_si256(_mm256_set1_epi16(0x1ff),mask9);
            __m256i uniq = _mm256_setzero_si256();

            for (unsigned char i = 0; i < 9; i++) {
                // load element i of 9 rows
                __m256i row = _mm256_set_epi16(0, 0, 0, 0, 0, 0, 0, candidates[i+72],
                              candidates[i+63], candidates[i+54], candidates[i+45], candidates[i+36], candidates[i+27], candidates[i+18], candidates[i+9], candidates[i]);
                rowx = _mm256_xor_si256(rowx,row);

                // load element i of 9 columns
                __m256i col = _mm256_and_si256(*(__m256i_u*) &candidates[i*9], mask9);
                colx = _mm256_xor_si256(colx,col);

                uniq = _mm256_or_si256(_mm256_and_si256(col, _mm256_sub_epi16(col, ones9)),uniq);

                // load element i of 9 boxes
                int bi = i%3+i/3*9; // starting in box 0
                __m256i box = _mm256_set_epi16(0, 0, 0, 0, 0, 0, 0, candidates[bi+60],
                              candidates[bi+57], candidates[bi+54], candidates[bi+33], candidates[bi+30], candidates[bi+27], candidates[bi+6], candidates[bi+3], candidates[bi]);
                boxx = _mm256_xor_si256(boxx,box);
            }

            __m256i res = _mm256_or_si256(rowx,colx);
            res = _mm256_or_si256(res, boxx);
            res = _mm256_or_si256(res, uniq);
            if ( ~_mm256_movemask_epi8(_mm256_cmpeq_epi16(res,_mm256_setzero_si256()))) {
                // verification failure
                if ( unique_check_mode == 0 ) {
                    if ( verbose != VNone ) {
                        solverData.printf("Line %d: solution to puzzle failed verification\n", line);
                    }
                    counters.unsolved_count++;
                    counters.not_verified_count++;
                } else {     // not supposed to get here
                    if ( verbose != VNone ) {
                        solverData.printf("Line %d: secondary puzzle solution failed verification\n", line);
                    }
                }
            } else  if ( verbose != VNone ) {
                if ( debug ) {
                    solverData.printf("Solution found and verified\n");
                }
                status.verified = true;
                if ( reportstats ) {
                    if ( unique_check_mode == 0 ) {
                        counters.verified_count++;
                    }
                }
            }
        }
        if ( verbose != VNone && reportstats ) {
            if ( unique_check_mode == 0 ) {
                counters.solved_count++;
            }
        }
        
        // Enter found digits into grid (unless we already had a solution)
        if ( unique_check_mode == 0 ) {
            status.solved = true;
            for (unsigned char j = 0; j < 64; j+=32) {
                __m256i t1 = _mm256_permute4x64_epi64(
                    _mm256_packus_epi16(_mm256_and_si256(*(__m256i*)&candidates[j],maskff),_mm256_and_si256(*(__m256i*)&candidates[j+16],maskff)),
                    0xD8);
                __m256i t2 = _mm256_and_si256(_mm256_srli_epi16(t1, 4),nibble_mask);
                t1 = _mm256_and_si256( t1, nibble_mask);
                t2 = _mm256_shuffle_epi8(lut_hi, t2);
                t1 = _mm256_shuffle_epi8(lut_lo, t1);
                _mm256_storeu_si256((__m256i_u*)&grid[j], _mm256_min_epu8(t1, t2));
            }
            __m256i tmp = _mm256_and_si256(*(__m256i*)&candidates[64],maskff);
            __m128i t1 = _mm256_castsi256_si128(_mm256_packus_epi16(tmp,_mm256_permute2x128_si256(tmp,tmp,0x11)));
            __m128i t2 = _mm_and_si128(_mm_srli_epi16(t1, 4),_mm256_castsi256_si128(nibble_mask));
            t1 = _mm_and_si128( t1, _mm256_castsi256_si128(nibble_mask));
            t2 = _mm_shuffle_epi8(_mm256_castsi256_si128(lut_hi), t2);
            t1 = _mm_shuffle_epi8(_mm256_castsi256_si128(lut_lo), t1);
            _mm_storeu_si128((__m128i_u*)&grid[64], _mm_min_epu8(t1, t2));
            grid[80] = '1'+_tzcnt_u32(candidates[80]);
        }

        if ( verbose != VNone ) {
            counters.no_guess_cnt += no_guess_incr;
        }
        if ( report_guess_puzzles && no_guess_incr==0 ) {
            solverData.printf("%d %.81s\n", line, grid-82);
        }
        status.guess = no_guess_incr?false:true;

        if ( grid_state->stackpointer && rules == Multiple && unique_check_mode == 0 && grid_state->flags >= 0 ) {
            if ( verbose == VDebug ) {
                solverData.printf("Solution: %.81s\nBack track to determine uniqueness\n", grid);
            }
            unique_check_mode = 1;
            return Phase_Back;
        }
        // otherwise uniqueness checking is complete
        if ( status.unique == false ) {
            counters.non_unique_count++;
        }
        return Phase_Done;
    }
        return Phase_HiddenSearch;
}
