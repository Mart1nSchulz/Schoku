// Out-of-class definition of `SolveCtx<verbose>::do_fishes()`.
// AlphaEvolve mutation unit: this file is the entire surface for Fish patterns: X-Wing/sword/jelly/squirmbag (OPT_FSH).
// Replace the function body to mutate the strategy without touching
// the rest of the solver.
//
// Returns Phase_HiddenSearch when the block finishes without firing
// any redirect (the dispatcher then continues to the next block in
// phase_hidden_search). Any other returned SolverPhase short-circuits
// back to the dispatcher (semantically identical to the pre-refactor
// `goto X` exits inside the block).
//
// Build flag: this body is empty when OPT_FSH is undefined; the helper
// then unconditionally returns Phase_HiddenSearch.
//
// CONTRACT: private include fragment, must be #included exactly once
// from inside `namespace Schoku { ... }` after solver/solve_ctx.hpp.
#pragma once


template <Verbosity verbose>
__attribute__((always_inline)) inline SolverPhase SolveCtx<verbose>::do_fishes() {
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
        cbbv_v.m256 = _mm256_insert_epi16(_mm256_and_si256(mask1ff,_mm256_castsi128_si256(off0)), *(int16_t*)&cbbv_digit.u8[9], 8);
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
                // Unit label for the common eliminate loop. Default is the
                // row-scan's Case-2 orientation ('r'). The plain emit block
                // overrides to match its antecedent's `base`. Case-1 has
                // its own local loop so its label isn't read here.
                char trace_elim_unit = 'r';

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
                        if ( trace::current ) {
                            // Plain fish antecedent. The (nsubs == cnt)
                            // branch is the canonical row-fish: cnt rows
                            // (subs_prp_x) form the base, cnt cols (base_x)
                            // form the cover. The dual branch (hassubs ==
                            // cnt) is detected via the same scan but is
                            // structurally a col-fish: base cols (base_x),
                            // cover rows (hassubs_prp_x).
                            char base_kind = (nsubs == cnt) ? 'r' : 'c';
                            unsigned short base_mask = (nsubs == cnt) ? subs_prp_x : (unsigned short)base_x;
                            unsigned short cover_mask = (nsubs == cnt) ? (unsigned short)base_x : (unsigned short)hassubs_prp_x;
                            trace::fish((int)cnt, (int)(dgt+1), base_kind,
                                        base_mask, cover_mask,
                                        (int)grid_state->stackpointer);
                            trace_elim_unit = base_kind;
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
                            // Sashimi/finned Case-1 has its own local
                            // removal loop. Mirror trace::eliminate
                            // emission so finned consequences aren't
                            // dropped from the JSONL. (Antecedent for
                            // finned is Phase 3.1; events here are orphans
                            // for now — consequence-only.)
                            bool tracing = (trace::current != nullptr);
                            char u = 'r';   // finned row-fish (Case-1, row scan)
                            int level = tracing ? (int)grid_state->stackpointer : 0;
                            while (clean_bits) {
                                unsigned char cl = tzcnt_and_mask(clean_bits);
                                if ( candidates[cl] & dgt_mask_bit ) {
                                    candidates[cl] &= ~dgt_mask_bit;
                                    if ( verbose == VDebug ) {
                                        solverData.printf("%s ", cl2txt[cl]);
                                    }
                                    if ( tracing ) {
                                        trace::eliminate("fish", u,
                                                         cl/9, cl%9, dgt_mask_bit,
                                                         level);
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
                    // Consequence unit matches the FISH orientation, set
                    // by the plain antecedent emit just above (overrides
                    // the row-scan default 'r' used for Case-2 orphans).
                    bool tracing = (trace::current != nullptr);
                    char u = trace_elim_unit;
                    int level = tracing ? (int)grid_state->stackpointer : 0;
                    while (clean_bits) {
                        unsigned char cl = tzcnt_and_mask(clean_bits);
                        if ( candidates[cl] & dgt_mask_bit ) {
                            candidates[cl] &= ~dgt_mask_bit;
                            if ( verbose == VDebug ) {
                                solverData.printf("%s ", cl2txt[cl]);
                            }
                            if ( tracing ) {
                                trace::eliminate("fish", u,
                                                 cl/9, cl%9, dgt_mask_bit,
                                                 level);
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
                        // Sashimi-driven placement. Antecedent is the
                        // sashimi pattern (Phase 3.1) — until then this
                        // remains a deduced-single placeholder.
                        trace::next_entry_reason = trace::ER_DeducedSingle;
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
                // Unit label for the common eliminate loop. Col-scan
                // default is 'c' (Case-2 orphan finned col-fish); the
                // plain emit block overrides to match its antecedent.
                char trace_elim_unit = 'c';

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
                        if ( trace::current ) {
                            // Mirror of the row-scan path, transposed view:
                            // (nsubs == cnt) => canonical col-fish (base
                            // cols, cover rows); (hassubs == cnt) dual is
                            // a row-fish detected via col scan.
                            char base_kind = (nsubs == cnt) ? 'c' : 'r';
                            unsigned short base_mask = (nsubs == cnt) ? (unsigned short)subs_prp_x : (unsigned short)base_x;
                            unsigned short cover_mask = (nsubs == cnt) ? (unsigned short)base_x : (unsigned short)hassubs_prp_x;
                            trace::fish((int)cnt, (int)(dgt+1), base_kind,
                                        base_mask, cover_mask,
                                        (int)grid_state->stackpointer);
                            trace_elim_unit = base_kind;
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
                            // Sashimi/finned Case-1 has its own local
                            // removal loop. Mirror trace::eliminate
                            // emission so finned consequences aren't
                            // dropped from the JSONL. (Antecedent for
                            // finned is Phase 3.1; events here are orphans
                            // for now — consequence-only.)
                            bool tracing = (trace::current != nullptr);
                            char u = 'c';   // finned col-fish (Case-1, col scan)
                            int level = tracing ? (int)grid_state->stackpointer : 0;
                            while (clean_bits) {
                                unsigned char cl = tzcnt_and_mask(clean_bits);
                                if ( candidates[cl] & dgt_mask_bit ) {
                                    candidates[cl] &= ~dgt_mask_bit;
                                    if ( verbose == VDebug ) {
                                        solverData.printf("%s ", cl2txt[cl]);
                                    }
                                    if ( tracing ) {
                                        trace::eliminate("fish", u,
                                                         cl/9, cl%9, dgt_mask_bit,
                                                         level);
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
                    // Mirror of row-scan. Plain emit block sets
                    // trace_elim_unit to its antecedent's base; Case-2
                    // orphans use the default 'c' (col-scan).
                    bool tracing = (trace::current != nullptr);
                    char u = trace_elim_unit;
                    int level = tracing ? (int)grid_state->stackpointer : 0;
                    while (clean_bits) {
                        unsigned char cl = tzcnt_and_mask(clean_bits);
                        if ( candidates[cl] & dgt_mask_bit) {
                            candidates[cl] &= ~dgt_mask_bit;
                            if ( verbose == VDebug ) {
                                solverData.printf("%s ", cl2txt[cl]);
                            }
                            if ( tracing ) {
                                trace::eliminate("fish", u,
                                                 cl/9, cl%9, dgt_mask_bit,
                                                 level);
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
                        // Sashimi-driven placement — Phase 3.1.
                        trace::next_entry_reason = trace::ER_DeducedSingle;
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
            bool fish2_alt_found = false;
            if ( t == 8 && pair_cnt >= 3 ) {
                unsigned char pos[9];
                for ( int i=pair_cnt-1; pair_locs; i-- ) {
                    pos[i] = 8-tzcnt_and_mask(pair_locs);
                }
                unsigned short res = 0;
                for ( unsigned int i=0; i<pair_cnt-1 && !fish2_alt_found; i++) {
                    for ( unsigned int k=i+1; k<pair_cnt; k++ ) {
                        if ( ( __popcnt16(res = cbbv_col_v.v16[pos[i]] | cbbv_col_v.v16[pos[k]])) == 3 ) {
                            alt_base_x = res;
                            fish2_alt_found = true;
                            break;
                        }
                    }
                }
            }
            if ( !fish2_alt_found ) {
                continue;
            }
            // alternative pattern synthesised; replay this iteration with t pinned at 7.
            t = 7;
            pair_locs = 0;
        } // for t
    } // for dgt
    } // mode_fish
#endif
    return Phase_HiddenSearch;
}
