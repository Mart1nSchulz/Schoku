// Out-of-class definitions for the four GridState::make_guess overloads.
//
// CONTRACT: must be #included from inside `namespace Schoku { ... }` and
// AFTER the GridState class body has closed (so `GridState::` is a valid
// nested-name-specifier here) but BEFORE any caller instantiates
// `grid_state->make_guess<verbose>(...)`.
//
// Inside the GridState class body, each of these four overloads is
// forward-declared. The bodies live here so a) the GridState class body
// stays at a manageable size, and b) the four guess strategies are
// grouped in a single purpose-specific file.
//
// Templates: instantiation happens at the call site in solve(), so all
// helpers used by the bodies (tzcnt_and_mask, __popcnt16, __lzcnt16,
// enter_digit, format_candidate_set, etc.) must be visible at that point.
// They are — schoku.cpp includes this header right after the class close
// and before solve(), and every helper is `inline` and defined in this TU.
//
// The header intentionally does not open the namespace itself; the
// include site is already inside `namespace Schoku`.
#pragma once

// this form of make_guess establishes a 'contract' between the caller's lambda
// and the creation of the new GridState.
// Due to its overhead, it is not the fastest, but the most flexible form to make a guess.
//
template<Verbosity verbose, typename F>
inline GridState* GridState::make_guess(unsigned char cell_index, F &&gridUpdater, Counters &counters, FILE *output) {
    // Create a copy of the state of the grid to make back tracking possible
    GridState* new_grid_state = this+1;
    if ( stackpointer >= GRIDSTATE_MAX-2 ) {
        fprintf(stderr, "Error: no GridState struct availabe\n");
        exit(0);
    }
    memcpy(new_grid_state, this, sizeof(GridState));
    new_grid_state->stackpointer++;

    const char *msgs[2];

    // gridUpdater is a lambda, but could also be a function reference.
    // gridUpdater receives as input:
    // GridState & present GridState
    // GridState & new GridState
    // gridUpdater receives as input:
    // const char (*)[2] two messages provided one for each GridState
    // gridUpdater encapsulates local updates, typically to candidates of either GridState
    // and the two messages.
    //
    gridUpdater(*this, *new_grid_state, msgs);
    if ( verbose == VDebug ) {
        fprintf(output, "guess at level >%d< - new level >%d<\nguess %s\n", stackpointer, new_grid_state->stackpointer, msgs[0]);
        char gridout[82];
        if ( debug > 1 ) {
            for (unsigned char j = 0; j < 81; ++j) {
                if ( unlocked.check_indexbit(j) ) {
                    gridout[j] = '0';
                } else {
                    gridout[j] = 49+_tzcnt_u32(candidates[j]);
                }
            }
            fprintf(output, "guess at %s\nsaved grid_state level >%d<: %.81s\n",
                   cl2txt[cell_index], stackpointer, gridout);
        }
        fprintf(output, "saved state for level %d: %s\n",
               stackpointer, msgs[1]);
        if ( debug > 1 ) {
            unsigned short *candidates = new_grid_state->candidates;
            for (unsigned char j = 0; j < 81; ++j) {
                if ( new_grid_state->unlocked.check_indexbit(j) ) {
                    gridout[j] = '0';
                } else {
                    gridout[j] = 49+_tzcnt_u32(candidates[j]);
                }
            }
            fprintf(output, "grid_state at level >%d< now: %.81s\n",
                   new_grid_state->stackpointer, gridout);
        }
    }
    counters.guesses++;

    return new_grid_state;
}

// this form of make_guess receives additional information to make a more efficient guess.
// Efficiency for a guess is measured in terms of cells resolved until a new guess needs to be made.
// In this form, the efficiency is provided via:
// - identifying a suitable triad such that it will become resolved on both branches of the guess.
// - there are exactly 2 candidates that are connected to the remaining cells of the row/col
//   and the box and either branch of the guess will thus impact those sections.
// - the operation is typically more balanced, in that both branches will provide similar
//   efficiencies (at least on average).
//
template<Verbosity verbose>
inline GridState* GridState::make_guess(SolverData *solverData) {
    // Make a guess for a triad with 4 candidate values that has 2 candidates that are not
    // constrained to the triad (not in 'tmust') and has at least 2 or more unresolved cells.
    // If we cannot obtain such a triad, fall back to make_guess().
    // With such a triad found, select one of the 2 identified candidates
    // to eliminate in the new GridState, issue the debug info and proceed.
    // Save the current GridState to back track to and eliminate the other candidate from it.
    // The desired result is that either way the triad is resolved, which is structually beneficial
    // for the progress of the solution.

    unsigned char tpos;    // grid cell index of triad start
    unsigned char type;    // row == 0, col = 1
    unsigned char inc;     // increment for iterating triad cells
    unsigned short *wo_musts;  // pointer to triads' 'without must' candidates
    TriadInfo &triad_info = solverData->triadInfo;

    bool found_triad = false;
    for ( int i=0; i<2 && !found_triad; i++ ) {
        type = i;
        unsigned long long totest = triad_info.triads_selection[i];
        wo_musts  = type==0?triad_info.row_triads_wo_musts:triad_info.col_triads_wo_musts;
        inc  = (i<<3) + 1;  // 1 for rows, 9 for cols
        while (totest) {
            int ti = tzcnt_and_mask(totest);
            int can_ti = ti-ti/10;   // 'canonical' triad index
            unsigned short altpair = wo_musts[ti];
            if ( __popcnt16 (altpair) == 2 ) {
                // get and check the unlocked indexbits for the triad
                unsigned int b;
                if ( type == 0 ) {
                    tpos = row_triad_canonical_map[can_ti]*3;
                    b = unlocked.get_indexbits(tpos, 3);
                } else {
                    tpos = col_canonical_triad_pos[can_ti];
                    b = unlocked.get_indexbits(tpos, 19) & 0x40201;
                }
                if ( _popcnt32(b) < 2 ) {
                    continue;
                }
                // found the right candidate
                wo_musts += ti;
                found_triad = true;
                break;
            }
        }
    }

    if ( !found_triad ) {
        // leverage any guess hints (e.g. by the fish algorithm)
        if ( solverData->guess_hint_digit != 0 ) {
            return make_guess<verbose>(solverData->guess_hint_index, solverData->guess_hint_digit, solverData->counters, solverData->output);
        }

        // if no suitable triad found, find a suitable bi-value.
        return make_guess<verbose>(*solverData);
    }

    // update the current and the new grid_state with their respective candidate to delete
    unsigned short select_cand = 0x8000 >> __lzcnt16(*wo_musts);

    // Create a copy of the state of the grid to make back tracking possible
    GridState* new_grid_state = this+1;
    if ( stackpointer >= GRIDSTATE_MAX-2 ) {
        fprintf(stderr, "Error: no GridState struct availabe\n");
        exit(0);
    }
    memcpy(new_grid_state, this, sizeof(GridState));
    new_grid_state->stackpointer++;

    unsigned short other_cand  = *wo_musts & ~select_cand;
    unsigned char off = tpos;

    off = tpos;
    // Update candidates
    for ( unsigned char k=0; k<3; k++, tpos += inc) {
        new_grid_state->candidates[tpos] &= ~select_cand;
        candidates[tpos] &= ~other_cand;
    }
    if (type == 0 ) {
        updated.set_indexbits(0x7,off,3);
        new_grid_state->updated.set_indexbits(7,off,3);
    } else {
        updated.set_indexbits(0x40201,off,19);
        new_grid_state->updated.set_indexbits(0x40201,off,19);
    }
    if ( verbose == VDebug ) {
        solverData->printf("guess at level >%d< - new level >%d<\n", stackpointer, new_grid_state->stackpointer);
        solverData->printf("guess remove {%d} from %s triad at %s\n",
               1+_tzcnt_u32(select_cand), type==0?"row":"col", cl2txt[off]);
    }
    if ( verbose != VNone ) {
        char gridout[82];
        if ( debug > 1 ) {
            for (unsigned char j = 0; j < 81; ++j) {
                if ( unlocked.check_indexbit(j) ) {
                    gridout[j] = '0';
                } else {
                    gridout[j] = 49+_tzcnt_u32(candidates[j]);
                }
            }
            solverData->printf("guess at %s\nsaved grid_state level >%d<: %.81s\n",
                   cl2txt[off], stackpointer, gridout);
        }
        if ( debug ) {
            solverData->printf("saved state for level %d: remove {%d} from %s triad at %s\n",
                   stackpointer, 1+_tzcnt_u32(other_cand), type==0?"row":"col", cl2txt[off]);
        }
        if ( debug > 1 ) {
            unsigned short *candidates = new_grid_state->candidates;
            for (unsigned char j = 0; j < 81; ++j) {
                if ( new_grid_state->unlocked.check_indexbit(j) ) {
                    gridout[j] = '0';
                } else {
                    gridout[j] = 49+_tzcnt_u32(candidates[j]);
                }
            }
            solverData->printf("grid_state at level >%d< now: %.81s\n",
                   new_grid_state->stackpointer, gridout);
        }
    }
    solverData->counters.guesses++;

    return new_grid_state;
}

// this version of make_guess is the simplest and original form of making a guess.
//
template<Verbosity verbose>
inline GridState* GridState::make_guess(SolverData &solverData) {
    // Find a cell with the least candidates.
    // For bivalues, build a score and once done, keep the highest scoring bivalue.
    // If there are no bivalues, score trivalues.
    // Pick the candidate with the highest value as the guess.
    // Save the current grid state (with the chosen candidate eliminated) for tracking back.

    // Find the cell with fewest possible candidates
    bit128_t &bivalues = solverData.getBivalues(candidates);
    Counters &counters = solverData.counters;
    FILE *output = solverData.output;
    unsigned char guess_index = 0;
    unsigned char best_index = 0xff;
    short best_score;
    unsigned short cands;
    unsigned char t = bivalues.u64[0]==0 ? 1 : 0;
    unsigned short digit;
    unsigned short search_digit;
    unsigned short best_digit = 0;
    unsigned long long bivals = bivalues.u64[t];

    if ( bivals ) {
        unsigned char cnt = 0;
        best_score = -1;
        while ( bivals ) {
            unsigned char search_index = tzcnt_and_mask(bivals) + (t<<6);
            bit128_t bv2 = { .u128= bivalues & *(bit128_t*)&big_index_lut[search_index][All][0] };
            search_digit = 0;
            short score = 0;
            cands = candidates[search_index];
            bool malus = false;
            while ( bv2 ) {
                unsigned short check_cands = candidates[tzcnt_and_mask(bv2)];
                if ( (cands & check_cands) ) {
                    score++;
                    if ( cands == check_cands ) {
                        malus = true;
                    } else {
                       search_digit |= cands & check_cands;
                    }
                }
            }
            if ( malus ) {
                score --;
                if ( !search_digit ) {
                    search_digit = cands;
                }
            }
            if ( score >= guess_score_threshold ) {
                best_index = search_index;
                best_digit = search_digit;
                break;
            }
            if ( score > best_score ) {
                best_index = search_index;
                best_digit = search_digit;
                best_score = score;
            }
            if ( cnt++ >= 8 ) {
                break;
            }
        }
        guess_index = best_index;
        if ( best_digit == 0 ) {
            best_digit = candidates[guess_index];
        }
        digit = 0x8000 >> __lzcnt16(best_digit);
        return make_guess<verbose>(guess_index, digit, counters, output);
    }

    // Trivalue scoring
    //
    // very unlikely except for very hard puzzles while under 27 cells solved.
    // a puzzle with no bivalues is always hard - extra effort and solving algos are
    // appropriate.
    //
    if ( verbose != VNone ) {
        counters.no_bivals_count++;
    }

    __m256i *boxes_as_cols = solverData.get_boxes_as_cols(candidates, (__m256i*)(this+1));

    {
        // fudged and sideways carry-save adder
        __m256i accu[3];
        __m256i tmp, cpairs;
        __m256i carry = boxes_as_cols[1];
        accu[0] = _mm256_xor_si256(boxes_as_cols[0], carry);
        accu[1] = _mm256_and_si256(boxes_as_cols[0], carry);
        accu[2] = _mm256_setzero_si256();
        for (unsigned int i=2; i<9; i++ ) {
           carry   = boxes_as_cols[i];
           tmp     = _mm256_xor_si256(accu[0], carry);
           carry   = _mm256_and_si256(accu[0], carry);
           accu[0] = tmp;
           tmp     = _mm256_xor_si256(accu[1], carry);
           carry   = _mm256_and_si256(accu[1], carry);
           accu[1] = tmp;
           accu[2] = _mm256_or_si256(carry, accu[2]);  // value > 2
        }
        cpairs = _mm256_andnot_si256(accu[2],_mm256_andnot_si256(accu[0],accu[1]));

        if ( !_mm256_testz_si256 ( cpairs, cpairs) ) {
            // cpairs contains, for each box, the digits that are conjugate pairs
            // (i.e. bilocated digits) in this box.

            unsigned char boxi = 0;
            for ( ; boxi < 9; boxi++ ) {
                // extract the boxes conjugate pair digits
                unsigned int dgtit = ((v16us)cpairs)[boxi];
                if ( dgtit == 0 ) {
                    continue;
                }
                unsigned short *boxp = candidates+box_start_by_boxindex[boxi];
                unsigned char best_score = 0;
                unsigned char score = 0;
                unsigned char score_index;
                unsigned char dgti;
                __m256i box = _mm256_setr_epi64x(*(unsigned long long *)(boxp),
                                                 *(unsigned long long *)(boxp+9),
                                                 *(unsigned long long *)(boxp+18), 0);
                unsigned int mskByStart[9] {};
                unsigned short dgtByStart[9] {};
                unsigned char pos1, pos2;
                unsigned short dgt;
                while ( dgtit ) {
                    dgt  = __blsi_u32(dgtit);
                    dgti = tzcnt_and_mask(dgtit);
                    score = 0;
                    unsigned int mskpos = _mm256_movemask_epi8(_mm256_cmpgt_epi16(_mm256_and_si256(box,_mm256_set1_epi16(dgt)), _mm256_setzero_si256())) & 0x3f3f3f;
                    score_index = box_start_by_boxindex[boxi];
                    unsigned char lpos1 = _tzcnt_u32(mskpos)>>1;
                    pos1 = group4x3offsets[lpos1];
                    lpos1 = lpos1 - lpos1/4;
                    pos2 = group4x3offsets[(62-__lzcnt64(mskpos))>>1];
                    // check for hidden pair
                    if ( mskByStart[lpos1] == mskpos ) {
                        // this is a hidden pair, simply guess dgti in pos1
                        // this will resolve at minimum 2 cells;
                        // make a guess right away
                        unsigned short cands = dgtByStart[lpos1] | dgt;
                        candidates[score_index+pos1] &= cands;
                        candidates[score_index+pos2] &= cands;
                        if ( verbose == VDebug ) {
                            char ret[32];
                            format_candidate_set(ret, cands);
                            fprintf(output, "hidden pair (box): %-7s %s\n", ret, cl2txt[score_index+pos1]);
                        }
                        if ( verbose != VNone ) {
                            counters.naked_sets_found++;
                        }
                        return make_guess<verbose>(score_index+pos1, dgt, counters, output);
                    }
                    // score this conjugate pair
                    mskByStart[lpos1] = mskpos;
                    dgtByStart[lpos1] = dgt;
                    if ( (viewbits_by_i[pos1].u32 & viewbits_by_i[pos2].u32 & 0x300ffff) == 0 ) {
                        score++;
                    }
                    score_index = box_start_by_boxindex[boxi];
                    if ( __popcnt16(candidates[score_index+pos1]) <= 3 ) {
                        score++;
                        score_index += pos2;
                    } else {
                        if ( __popcnt16(candidates[score_index+pos2]) <= 3 ) {
                            score++;
                        }
                        score_index += pos1;
                    }
                    if ( score > best_score ) {
                        best_index = score_index;
                        best_score = score;
                        best_digit = 1<<dgti;
                    }
                    if ( best_score >= 2 ) {
                        break;
                    }
                }
                if ( best_score >= 2 ) {
                    break;
                }
            }

            if ( best_index != 0xff ) {
                return make_guess<verbose>(best_index, best_digit, counters, output);
            }
        }
    }

    // find a trivalue cell if nothing else helps.

    unsigned char cnt;
    unsigned char best_cnt = 16;
    unsigned char i_rel;
    unsigned long long to_visit = unlocked.u64[0];
    while ( best_cnt > 3 && to_visit != 0 ) {
        i_rel = tzcnt_and_mask(to_visit);
        cnt = __popcnt16(candidates[i_rel]);
        if (cnt < best_cnt) {
            best_cnt = cnt;
            guess_index = i_rel;
        }
    }

    to_visit = unlocked.u64[1];
    while ( best_cnt > 3 && to_visit != 0 ) {
        i_rel = tzcnt_and_mask(to_visit) + 64;
        cnt = __popcnt16(candidates[i_rel]);
        if (cnt < best_cnt) {
            best_cnt = cnt;
            guess_index = i_rel;
        }
    }
    // Find the first candidate in this cell (lsb set)
    // Note: using tzcnt would be equally valid; this pick is historical
    digit = 0x8000 >> __lzcnt16(candidates[guess_index]);
    return make_guess<verbose>(guess_index, digit, counters, output);
}

// this version of make_guess takes a cell index and digit for the guess
//
template<Verbosity verbose>
inline GridState* GridState::make_guess(unsigned char guess_index, unsigned short digit, Counters &counters, FILE *output ) {
    // Create a copy of the state of the grid to make back tracking possible
    GridState* new_grid_state = this+1;
    if ( stackpointer >= GRIDSTATE_MAX-2 ) {
        fprintf(stderr, "Error: no GridState object availabe\n");
        exit(0);
    }
    memcpy(new_grid_state, this, sizeof(GridState));
    new_grid_state->stackpointer++;

    // Remove the guessed candidate from the old grid
    // when we get back here to the old grid, we know the guess was wrong
    candidates[guess_index] &= ~digit;

    updated.set_indexbit(guess_index);

    if ( verbose == VDebug && (debug > 1) ) {
        char gridout[82];
        for (unsigned char j = 0; j < 81; ++j) {
            if ( (candidates[j] & (candidates[j]-1)) ) {
                gridout[j] = '0';
            } else {
                gridout[j] = 49+_tzcnt_u32(candidates[j]);
            }
        }
        fprintf(output, "guess at %s\nsaved grid_state level >%d<: %.81s\n",
               cl2txt[guess_index], stackpointer, gridout);
    }

    // Update candidates
    if ( verbose == VDebug ) {
        fprintf(output, "guess at level >%d< - new level >%d<\nguess", stackpointer, new_grid_state->stackpointer);
    }

    new_grid_state->enter_digit<verbose>( digit, guess_index, output);
    counters.guesses++;

    if ( verbose == VDebug && (debug > 1) ) {
        unsigned short *candidates = new_grid_state->candidates;
        char gridout[82];
        for (unsigned char j = 0; j < 81; ++j) {
            if ( (candidates[j] & (candidates[j]-1)) ) {
                gridout[j] = '0';
            } else {
                gridout[j] = 49+_tzcnt_u32(candidates[j]);
            }
        }
        fprintf(output, "grid_state at level >%d< now: %.81s\n",
               new_grid_state->stackpointer, gridout);
    }
    return new_grid_state;
}
