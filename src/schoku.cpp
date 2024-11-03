// This code uses AVX2 instructions...
/*
 * Schoku
 *
 * A high speed sudoku solver by M. Schulz
 *
 * Based on the sudoku solver by Mirage ( https://codegolf.stackexchange.com/users/106606/mirage )
 * at https://codegolf.stackexchange.com/questions/190727/the-fastest-sudoku-solver
 * on Sep 22, 2021
 *
 * Version 0.2
 *
 * Performance changes:
 * - memory mapped I/O
 * - allocate GridState array once per thread
 *   this significantly improves multi-threaded performance
 * - precompute data for the hidden single search
 *
 * Basic performance measurement and statistics:
 *
 * data: 17-clue sudoku
 * CPU:  Ryzen 7 4700U
 *
 * fastss version: 0.2
 *     49151  puzzles entered
 *    55.3ms   1.13µs/puzzle  solving time
 *     31355  63.79%  puzzles solved without guessing
 *     57149   1.16/puzzle  total guesses
 *     38206   0.78/puzzle  total back tracks
 *    506095  10.30/puzzle  total digits entered and retracted
 *   1393073  28.34/puzzle  total 'rounds'
 *   2609167  53.08/puzzle  naked sets found
 *  10753332 218.78/puzzle  naked sets searched
 *
 * Clearly, the naked sets are challenge.
 */
#include <atomic>
#include <chrono>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <unistd.h>
#include <assert.h>
#include <fcntl.h>
#include <omp.h>
#include <stdbool.h>
#include <intrin.h>
#include <sys/stat.h>
#include <sys/mman.h>

const char *version_string = "0.2";

// lookup tables that may or may not speed things up by avoiding division
const unsigned char box_index[81] = {
    0, 0, 0, 1, 1, 1, 2, 2, 2,
    0, 0, 0, 1, 1, 1, 2, 2, 2,
    0, 0, 0, 1, 1, 1, 2, 2, 2,
    3, 3, 3, 4, 4, 4, 5, 5, 5,
    3, 3, 3, 4, 4, 4, 5, 5, 5,
    3, 3, 3, 4, 4, 4, 5, 5, 5,
    6, 6, 6, 7, 7, 7, 8, 8, 8,
    6, 6, 6, 7, 7, 7, 8, 8, 8,
    6, 6, 6, 7, 7, 7, 8, 8, 8
};

const unsigned char column_index[81] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8,
    0, 1, 2, 3, 4, 5, 6, 7, 8,
    0, 1, 2, 3, 4, 5, 6, 7, 8,
    0, 1, 2, 3, 4, 5, 6, 7, 8,
    0, 1, 2, 3, 4, 5, 6, 7, 8,
    0, 1, 2, 3, 4, 5, 6, 7, 8,
    0, 1, 2, 3, 4, 5, 6, 7, 8,
    0, 1, 2, 3, 4, 5, 6, 7, 8,
    0, 1, 2, 3, 4, 5, 6, 7, 8,
};

const unsigned char row_index[81] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1,
    2, 2, 2, 2, 2, 2, 2, 2, 2,
    3, 3, 3, 3, 3, 3, 3, 3, 3,
    4, 4, 4, 4, 4, 4, 4, 4, 4,
    5, 5, 5, 5, 5, 5, 5, 5, 5,
    6, 6, 6, 6, 6, 6, 6, 6, 6,
    7, 7, 7, 7, 7, 7, 7, 7, 7,
    8, 8, 8, 8, 8, 8, 8, 8, 8,
};

const unsigned char box_start[81] = {
    0, 0, 0, 3, 3, 3, 6, 6, 6,
    0, 0, 0, 3, 3, 3, 6, 6, 6,
    0, 0, 0, 3, 3, 3, 6, 6, 6,
    27, 27, 27, 30, 30, 30, 33, 33, 33,
    27, 27, 27, 30, 30, 30, 33, 33, 33,
    27, 27, 27, 30, 30, 30, 33, 33, 33,
    54, 54, 54, 57, 57, 57, 60, 60, 60,
    54, 54, 54, 57, 57, 57, 60, 60, 60,
    54, 54, 54, 57, 57, 57, 60, 60, 60
};

// stats and command line options
int reportstats     = 0; // collect and report some statistics

signed char *output;

std::atomic<long> solved_count(0);          // puzzles solved
std::atomic<long long> guesses(0);          // how many guesses did it take
std::atomic<long long> trackbacks(0);       // how often did we back track
std::atomic<long> no_guess_cnt(0);          // how many puzzles were solved without guessing
std::atomic<long long> past_naked_count(0); // how often do we get past the naked single serach
std::atomic<long long> naked_sets_searched(0); // how many naked sets did we search for
std::atomic<long long> naked_sets_found(0); // how many naked sets did we actually find
std::atomic<long long> digits_entered_and_retracted(0); // to measure guessing overhead

inline void add_column_indices(unsigned long long indices[2], unsigned char i) {
    indices[0] |= 0x8040201008040201ULL << column_index[i];
    indices[1] |= 0x8040201008040201ULL >> (10-column_index[i]);
}

inline void add_row_indices(unsigned long long indices[2], unsigned char i) {
    switch (row_index[i]) {
        case 7:
            indices[0] |= 0x8000000000000000ULL;
            indices[1] |= 0xffULL;
            break;
        case 8:
            indices[1] |= 0x01ff00ULL;
            break;
        default:
            indices[0] |= 0x01ffULL << 9*row_index[i];
    }
}

inline void add_box_indices(unsigned long long indices[2], unsigned char i) {
    indices[0] |= 0x1c0e07ULL << box_start[i];
    indices[1] |= 0x0381c0e0ULL >> (60-box_start[i]);
}

typedef struct {
    unsigned long long unlocked[2];        // for keeping track of which cells don't need to be looked at anymore. Set bits correspond to cells that still have multiple possibilities
    unsigned long long updated[2];        // for keeping track of which cell's candidates may have been changed since last time we looked for naked sets. Set bits correspond to changed candidates in these cells
    unsigned short candidates[81];        // which digits can go in this cell? Set bits correspond to possible digits
	unsigned short filler[5];
    int	stackpointer;                     // this-1 == last grid state before a guess was made, used for backtracking
// An array GridState[n] can use stackpointer to increment until it reaches n.
// Similary, for back tracking, the relative GridState-1 is considered the ancestor
// until stackpointer is 0. 
// 
} GridState;

inline void enter_digit(GridState* grid_state, unsigned short digit, unsigned char i) {
    // lock this cell and and remove this digit from the candidates in this row, column and box
    
    unsigned short* candidates = grid_state->candidates;
    unsigned long long* unlocked = grid_state->unlocked;
    unsigned long long to_update[2] = {0};
    
    if (i < 64) {
        unlocked[0] &= ~(1ULL << i);
    } else {
        unlocked[1] &= ~(1ULL << (i-64));
    }
    
    candidates[i] = digit;
    
    add_box_indices(to_update, i);
    add_column_indices(to_update, i);
    add_row_indices(to_update, i);
    
    to_update[0] &= unlocked[0];
    to_update[1] &= unlocked[1];
    
    grid_state->updated[0] |= to_update[0];
    grid_state->updated[1] |= to_update[1];
    
    const __m256i bit_mask = _mm256_setr_epi16(1<<0, 1<<1, 1<<2, 1<<3, 1<<4, 1<<5, 1<<6, 1<<7, 1<<8, 1<<9, 1<<10, 1<<11, 1<<12, 1<<13, 1<<14, 1<<15);
    const __m256i mask = _mm256_set1_epi16(~candidates[i]);
    for (unsigned char j = 0; j < 80; j += 16) {
        unsigned short m;
        if (j < 64) {
            m = (unsigned short) ((to_update[0] >> j) & 0xffff);
        } else {
            m = (unsigned short) (to_update[1] & 0xffff);
        }    
        __m256i c = _mm256_loadu_si256((__m256i*) &candidates[j]);
        __m256i u = _mm256_cmpeq_epi16(_mm256_and_si256(bit_mask, _mm256_set1_epi16(m)), _mm256_setzero_si256());
        c = _mm256_and_si256(c, _mm256_or_si256(mask, u));
        _mm256_storeu_si256((__m256i*) &candidates[j], c);
    }
    if ((to_update[1] & (1ULL << (80-64))) != 0) {
        candidates[80] &= ~candidates[i];
    }
}

inline GridState* make_guess(GridState* grid_state) {
    // Make a guess for the cell with the least candidates. The guess will be the lowest
    // possible digit for that cell. If multiple cells have the same number of candidates, the 
    // cell with lowest index will be chosen. Also save the current grid state for tracking back
    // in case the guess is wrong. No cell has less than two candidates.
    
    unsigned long long* unlocked = grid_state->unlocked;
    unsigned short* candidates = grid_state->candidates;
    
    // Find the cell with fewest possible candidates
    unsigned long long to_visit;
    unsigned long guess_index = 0;
    int i_rel;
    unsigned short cnt;
    unsigned short best_cnt = 16;
    
    to_visit = unlocked[0];
    while (to_visit != 0 && best_cnt > 2) {
		i_rel = _tzcnt_u64(to_visit);
        to_visit &= ~(1ULL << i_rel);
        cnt = __popcnt16(candidates[i_rel]);
        if (cnt < best_cnt) {
            best_cnt = cnt;
            guess_index = i_rel;
        }
    }
    to_visit = unlocked[1];
    while (best_cnt > 2 && to_visit != 0 ) {
		i_rel = _tzcnt_u32(to_visit);
        to_visit &= ~(1UL << i_rel);
        cnt = __popcnt16(candidates[i_rel + 64]);
        if (cnt < best_cnt) {
            best_cnt = cnt;
            guess_index = i_rel + 64;
        }
    }
    
    // Find the first candidate in this cell
	unsigned short digit = 1 << __bsrd(candidates[guess_index]);
    
    // Create a copy of the state of the grid to make back tracking possible
    GridState* new_grid_state = grid_state+1;
	if ( grid_state->stackpointer >= 31 ) {
		fprintf(stderr, "Error: no GridState struct availabe\n");
		exit(0);
	}
    memcpy(new_grid_state, grid_state, sizeof(GridState));
	new_grid_state->stackpointer++;
    
    // Remove the guessed candidate from the old grid
    // when we get back here to the old grid, we know the guess was wrong
    grid_state->candidates[guess_index] &= ~digit;
    if (guess_index < 64) {
        grid_state->updated[0] |= 1ULL << guess_index;
    } else {
        grid_state->updated[1] |= 1ULL << (guess_index-64);
    }
    
    // Update candidates
    enter_digit(new_grid_state, digit, (unsigned char) guess_index);
    
    guesses++;
    
    return new_grid_state;
}


inline GridState * track_back(GridState* grid_state, int line) {
    // Go back to the state when the last guess was made
    // This state had the guess removed as candidate from it's cell
    
	if (grid_state->stackpointer) {
	    return grid_state-1;
	} else {
		// This only happens when the puzzle is not valid
		fprintf(stderr, "Line %d: No solution found!\n", line);
		exit(0);
	}
    
}


static bool solve(signed char grid[81], GridState *grid_state, int line) {

    unsigned long long* unlocked = grid_state->unlocked;
    unsigned long long* updated = grid_state->updated;
    unsigned short* candidates = grid_state->candidates;
    unlocked[0] = 0xffffffffffffffffULL;
    unlocked[1] = 0x1ffffULL;
    updated[0] = unlocked[0];
    updated[1] = unlocked[1];

    unsigned long long my_digits_entered_and_retracted = 0;
    unsigned long long my_naked_sets_searched = 0;
    unsigned char no_guess_incr = 1;

    unsigned int my_past_naked_count = 0;
    
    {
        signed char digit;
        unsigned short columns[9] = {0};
        unsigned short rows[9] = {0};
        unsigned short boxes[9] = {0};
        
        for (unsigned char i = 0; i < 64; ++i) {
            digit = grid[i];
            if (digit >= 49) {
                columns[column_index[i]] |= 1 << (digit-49);
                rows[row_index[i]] |= 1 << (digit-49);
                boxes[box_index[i]] |= 1 << (digit-49);
                unlocked[0] &= ~(1ULL << i);
            }
        }
        for (unsigned char i = 64; i < 81; ++i) {
            digit = grid[i];
            if (digit >= 49) {
                columns[column_index[i]] |= 1 << (digit-49);
                rows[row_index[i]] |= 1 << (digit-49);
                boxes[box_index[i]] |= 1 << (digit-49);
                unlocked[1] &= ~(1ULL << (i-64));
            }
        }
        
        for (unsigned char i = 0; i < 81; ++i) {
            if (grid[i] < 49) {
                candidates[i] = 0x01ff ^ (rows[row_index[i]] | columns[column_index[i]] | boxes[box_index[i]]);
            } else {
                candidates[i] = 1 << (grid[i]-49);
            }
        }
    }
    
    start:
    
    unlocked = grid_state->unlocked;
    candidates = grid_state->candidates;
            
    // Find naked singles
    {    
        bool found;
        const __m256i ones = _mm256_set1_epi16(1);
        const __m256i bit_mask = _mm256_setr_epi16(1<<0, 1<<1, 1<<2, 1<<3, 1<<4, 1<<5, 1<<6, 1<<7, 1<<8, 1<<9, 1<<10, 1<<11, 1<<12, 1<<13, 1<<14, 1<<15);
        do {
            found = false;
            for (unsigned char i = 0; i < 80; i += 16) {
                __m256i c = _mm256_loadu_si256((__m256i*) &candidates[i]);
                // Check if any cell has zero candidates
                if (_mm256_movemask_epi8(_mm256_cmpeq_epi16(c, _mm256_setzero_si256()))) {
                    // Back track, no solutions along this path
                    GridState *old_grid_state = grid_state;
                    grid_state = track_back(grid_state, line);
                    trackbacks++;
                    if ( reportstats ) {
                        my_digits_entered_and_retracted +=
                                (_popcnt64(grid_state->unlocked[0] & ~old_grid_state->unlocked[0]))
                              + (_popcnt32(grid_state->unlocked[1] & ~old_grid_state->unlocked[1]));
                    }
                    goto start;
                } else {
                    unsigned short m;
                    if (i < 64) {
                        m = (unsigned short) ((unlocked[0] >> i) & 0xffff);
                    } else {
                        m = (unsigned short) (unlocked[1] & 0xffff);
                    }                        
                    if ( m ) {
					    // remove least significant digit and compare to 0:
					    // (c = c & (c-1)) == 0  => naked single
                        __m256i a = _mm256_cmpeq_epi16(_mm256_and_si256(c, _mm256_sub_epi16(c, ones)), _mm256_setzero_si256());
						// expand m to a boolean vector:
                        __m256i u = _mm256_cmpeq_epi16(_mm256_and_si256(bit_mask, _mm256_set1_epi16(m)), bit_mask);
					    // mask out the locked singles:
                        int mask = _mm256_movemask_epi8(_mm256_and_si256(a, u));
                    
                        if (mask) {
                            int index = (_tzcnt_u32(mask)>>1) + i;
                            enter_digit(grid_state, candidates[index], index);
                            found = true;
                        }
					}
                }
            }
            if (unlocked[1] & (1ULL << (80-64))) {
                if (candidates[80] == 0) {
                    // no solutions go back
                    GridState *old_grid_state = grid_state;
                    grid_state = track_back(grid_state, line);
                    trackbacks++;
                    if ( reportstats ) {
                        my_digits_entered_and_retracted += 
                            (_popcnt64(grid_state->unlocked[0] & ~old_grid_state->unlocked[0]))
                          + (_popcnt32(grid_state->unlocked[1] & ~old_grid_state->unlocked[1]));
                    }
                    goto start;
                } else if (__popcnt16(candidates[80]) == 1) {
                    // Enter the digit and update candidates
                    enter_digit(grid_state, candidates[80], 80);
                    found = true;
                }
            }
        } while (found);
    }
    
    // Check if it's solved, if it ever gets solved it will be solved after looking for naked singles
    if ((unlocked[0] | unlocked[1]) == 0) {
        // Solved it
        // Enter found digits into grid
        for (unsigned char j = 0; j < 81; ++j) {
            grid[j] = 49+__bsrd(candidates[j]);
        }

        solved_count++;
        no_guess_cnt += no_guess_incr;

        if ( reportstats ) {
            past_naked_count += my_past_naked_count;
            naked_sets_searched += my_naked_sets_searched;
            digits_entered_and_retracted += my_digits_entered_and_retracted;
        }
        return true;
    }

    my_past_naked_count++;

    // Find hidden singles
    // Don't check the last column because it doesn't fit in the SSE register so it's not really worth checking
    {
        const __m128i ones = _mm_set1_epi16(1);
        const __m128i bit_mask = _mm_setr_epi16(1<<0, 1<<1, 1<<2, 1<<3, 1<<4, 1<<5, 1<<6, 1<<7);
        const __m128i shuffle_mask = _mm_setr_epi8(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1);

		__m128i column_mask_tails[8];

        for (unsigned char i = 0; i < 81; i += 9) {
            
            // rows
            __m128i row_mask = _mm_set1_epi16(0x01ff ^ candidates[i+8]);
            __m128i c = _mm_loadu_si128((__m128i*) &candidates[i]);
            for (unsigned char j = 0; j < 7; ++j) {
                // rotate shift (1 2 3 4) -> (4 1 2 3)
                c = _mm_shuffle_epi8(c, shuffle_mask);
                row_mask = _mm_andnot_si128(c, row_mask);
            }
            
            // columns
			__m128i column_mask;
            if ( i == 0 ) {
                // keep precomputed 'tails' of column_mask around
   	            column_mask = _mm_set1_epi16(0x01ff);
	            for (signed char j = 81-9; j > 0; j -= 9) {
					column_mask_tails[j/9-1] = column_mask;
       	            column_mask = _mm_andnot_si128(_mm_loadu_si128((__m128i*) &candidates[j]), column_mask);
                }
            } else {
                column_mask = column_mask_tails[i/9-1];
                for (signed char j = 0; j < i; j += 9) {
                    column_mask = _mm_andnot_si128(_mm_loadu_si128((__m128i*) &candidates[j]), column_mask);
                }
            }
            
            // boxes aren't worth it
            
            __m128i or_mask = _mm_or_si128(row_mask, column_mask);
            
            if (_mm_test_all_zeros(or_mask, _mm_sub_epi16(or_mask, ones))) {

                unsigned short m;
                if (i < 64) {
                    m = (unsigned short) ((unlocked[0] >> i) & 0xff);
                } else {
                    m = (unsigned short) ((unlocked[1] >> (i-64)) & 0xff);
                }

                c = _mm_loadu_si128((__m128i*) &candidates[i]);

				// compare the identified singles to the row:
                __m128i a = _mm_cmpgt_epi16(_mm_and_si128(c, or_mask), _mm_setzero_si128());
				// expand m to a boolean vector:
                __m128i u = _mm_cmpeq_epi16(_mm_and_si128(bit_mask, _mm_set1_epi16(m)), bit_mask);
				// mask out the locked cells:
                int mask = _mm_movemask_epi8(_mm_and_si128(a, u));

                if (mask) {
                    int index = __tzcnt_u16(mask)>>1;
                    
                    enter_digit(grid_state, ((__v8hu)or_mask)[index], index+i);
                    goto start;
                }
                
            } else {
                // no solutions go back
                GridState *old_grid_state = grid_state;
                grid_state = track_back(grid_state, line);
                trackbacks++;
                if ( reportstats ) {
                    my_digits_entered_and_retracted += 
                        (_popcnt64(grid_state->unlocked[0] & ~old_grid_state->unlocked[0]))
                      + (_popcnt32(grid_state->unlocked[1] & ~old_grid_state->unlocked[1]));
                }
                goto start;
            }
        }
    }
    
    // Find naked sets, up to 5
    {
        bool found = false;
        // because this is kind of an expensive task, we are not going to visit all cells but only those that were changed
        unsigned long long *to_visit_n = grid_state->updated;
        to_visit_n[0] &= unlocked[0];
        to_visit_n[1] &= unlocked[1];
        
        for (unsigned char n = 0; n < 2; ++n) {
            while (to_visit_n[n]) {
                int i_rel = _tzcnt_u64(to_visit_n[n]);
                
                to_visit_n[n] ^= 1ULL << i_rel;
                unsigned char i = (unsigned char) i_rel + 64*n;
                
                unsigned short cnt = __popcnt16(candidates[i]);
                
                if (cnt <= 5) {
                    // check column
                    unsigned long long to_change[2] = {0};
                    unsigned char s;
                    __m128i a_i = _mm_set1_epi16(candidates[i]);
                    __m128i a_j = _mm_set_epi16(candidates[column_index[i]+63], candidates[column_index[i]+54], candidates[column_index[i]+45], candidates[column_index[i]+36], candidates[column_index[i]+27], candidates[column_index[i]+18], candidates[column_index[i]+9], candidates[column_index[i]]);
                    __m128i res = _mm_cmpeq_epi16(a_i, _mm_or_si128(a_i, a_j));
                    s = __popcnt16(_mm_movemask_epi8(res)) >> 1;
                    s += candidates[i] == (candidates[i] | candidates[column_index[i]+72]);
                    my_naked_sets_searched++;
                    if (s > cnt) {
                        GridState *old_grid_state = grid_state;
                        grid_state = track_back(grid_state, line);
                        trackbacks++;
                        if ( reportstats ) {
                            my_digits_entered_and_retracted += 
                                (_popcnt64(grid_state->unlocked[0] & ~old_grid_state->unlocked[0]))
                              + (_popcnt32(grid_state->unlocked[1] & ~old_grid_state->unlocked[1]));
                        }
                        goto start;
                    } else if (s == cnt) {
                        naked_sets_found++;
                        add_column_indices(to_change, i);
                    }

                    // check row
                    a_j = _mm_load_si128((__m128i*) &candidates[9*row_index[i]]);
                    res = _mm_cmpeq_epi16(a_i, _mm_or_si128(a_i, a_j));
                    s = __popcnt16(_mm_movemask_epi8(res)) >> 1;
                    s += candidates[i] == (candidates[i] | candidates[9*row_index[i]+8]);
                    my_naked_sets_searched++;

                    if (s > cnt) {
                        GridState *old_grid_state = grid_state;
                        grid_state = track_back(grid_state, line);
                        trackbacks++;
                        if ( reportstats ) {
                            my_digits_entered_and_retracted += 
                                (_popcnt64(grid_state->unlocked[0] & ~old_grid_state->unlocked[0]))
                              + (_popcnt32(grid_state->unlocked[1] & ~old_grid_state->unlocked[1]));
                        }
                        goto start;
                    } else if (s == cnt) {
                        naked_sets_found++;
                        add_row_indices(to_change, i);
                    }
                    
                    // check box
                    unsigned short b = box_start[i];
                    a_j = _mm_set_epi16(candidates[b], candidates[b+1], candidates[b+2], candidates[b+9], candidates[b+10], candidates[b+11], candidates[b+18], candidates[b+19]);
                    res = _mm_cmpeq_epi16(a_i, _mm_or_si128(a_i, a_j));
                    s = __popcnt16(_mm_movemask_epi8(res)) >> 1;
                    s += candidates[i] == (candidates[i] | candidates[b+20]);
                    my_naked_sets_searched++;
                    if (s > cnt) {
                        GridState *old_grid_state = grid_state;
                        grid_state = track_back(grid_state, line);
                        trackbacks++;
                        if ( reportstats ) {
                            my_digits_entered_and_retracted += 
                                (_popcnt64(grid_state->unlocked[0] & ~old_grid_state->unlocked[0]))
                              + (_popcnt32(grid_state->unlocked[1] & ~old_grid_state->unlocked[1]));
                        }
                        goto start;
                    } else if (s == cnt) {
                        naked_sets_found++;
                        add_box_indices(to_change, i);
                    }
                                    
                    to_change[0] &= unlocked[0];
                    to_change[1] &= unlocked[1];
                    
                    // update candidates
                    for (unsigned char n = 0; n < 2; ++n) {
                        while (to_change[n]) {
                            int j_rel = _tzcnt_u64(to_change[n]);
                            to_change[n] &= ~(1ULL << j_rel);
                            unsigned char j = (unsigned char) j_rel + 64*n;
                            
                            if ((candidates[j] | candidates[i]) != candidates[i]) {
                                if (candidates[j] & candidates[i]) {
                                    candidates[j] &= ~candidates[i];
                                    to_visit_n[n] |= 1ULL << j_rel;
                                    found = true;
                                }
                            }
                        }
                    }
                    
                    // If any cell's candidates got updated, go back and try all that other stuff again
                    if (found) {
                        goto start;
                    }
                    
                }
            }
        }
    }
    
    // More techniques could be added here but they're not really worth checking for on the 17 clue sudoku set
    
    // Make a guess if all that didn't work
    grid_state = make_guess(grid_state);
    no_guess_incr = 0;
    goto start;
    
}


int main(int argc, char *argv[]) {

    if ( argc > 0 ) {
        argc--;
        argv++;
    }    

    while ( argc && argv[0][0] == '-' ) {
        if (argv[0][1] == 0) {
            argv++; argc--;
            break;
        }
        switch(argv[0][1]) {
        case 'x':    // stats output
             reportstats=1;
             break;
        }
        argc--, argv++;
    }
    
	auto starttime = std::chrono::steady_clock::now();

	const char *ifn = argc > 0? argv[0] : "puzzles.txt";
	int fdin = open(ifn, O_RDONLY);
	if ( fdin == -1 ) {
		if (errno ) {
			fprintf(stderr, "Error: Failed to open file %s: %s\n", ifn, strerror(errno));
			exit(0);
		}
	}

    // get size of file
	struct stat sb;
	fstat(fdin, &sb);
    size_t fsize = sb.st_size;

	// map the input file
	char *string = (char *)mmap((void*)0, fsize, PROT_READ, MAP_PRIVATE, fdin, 0);
	if ( string == MAP_FAILED ) {
		if (errno ) {
			printf("Error mmap of input file %s: %s\n", ifn, strerror(errno));
			exit(0);
		}
	}
	close(fdin);

	// skip first line, unless it's a puzzle.
    size_t pre = 0;
	if ( !isdigit(string[0]) || string[81] != 10 ) {
	    while (string[pre] != 10) {
    	    ++pre;
    	}
    	++pre;
	}
    
    size_t post = 1;
	if ( string[fsize-1] != 10 )
		post = 0;

	// get and check the number of puzzles
	size_t npuzzles = (fsize - pre + (1-post))/82;

	if ( (fsize -pre -post + 1) % 82 ) {
		fprintf(stderr, "found %ld puzzles with %ld(start)+%ld(end) extra characters\n", (fsize - pre - post + 1)/82, pre, post);
	}

	const char *ofn = argc > 1? argv[1] : "solutions.txt";
	int fdout = open(ofn, O_RDWR|O_CREAT, 0775);
	if ( fdout == -1 ) {
		if (errno ) {
			printf("Error opening output file %s: %s\n", ofn, strerror(errno));
			exit(0);
		}
	}
	if ( ftruncate(fdout, (size_t)npuzzles*164) == -1 ) {
		if (errno ) {
			printf("Error setting size (ftruncate) on output file %s: %s\n", ofn, strerror(errno));
		}
		exit(0);
	}

	// map the output file
	output = (signed char *)mmap((void*)0, npuzzles*164, PROT_WRITE, MAP_SHARED, fdout, 0);
	if ( output == MAP_FAILED ) {
		if (errno ) {
			printf("Error mmap of output file %s: %s\n", ofn, strerror(errno));
			exit(0);
		}
	}
	close(fdout);

    // solve all sudokus and prepare output file
    size_t i;
	char *string_pre = string+pre;
	GridState *stack = 0;

#pragma omp parallel for firstprivate(stack) shared(string_pre, npuzzles, output, fsize, i) schedule(static,32)
    for (i = 0; i < npuzzles*82; i+=82) {
        // copy unsolved grid
        memcpy(&output[i*2], &string_pre[i], 81);
        memcpy(&output[i*2+82], &string_pre[i], 81);
        // add comma and newline in right place
        output[i*2 + 81] = ',';
        output[i*2 + 163] = 10;
		if ( stack == 0 ) {
			stack = (GridState*)malloc(sizeof(GridState)*32);
        }
        // solve the grid in place
        solve(&output[i*2+82], stack, i/82+1);
    }
    
	int err = munmap(string, fsize);
	if ( err == -1 ) {
		if (errno ) {
			printf("Error munmap file %s: %s\n", ifn, strerror(errno));
		}
	}
	err = munmap(output, (size_t)npuzzles*164);
	if ( err == -1 ) {
		if (errno ) {
			printf("Error munmap file %s: %s\n", ofn, strerror(errno));
		}
	}

    if ( reportstats) {
        long long duration = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration(std::chrono::steady_clock::now() - starttime)).count();
        printf("fastss version: %s\n", version_string);
        printf("%10ld  puzzles entered\n", npuzzles);
		printf("%8.1lfms  %5.2lf\u00b5s/puzzle  solving time\n", (double)duration/1000000, (double)duration/(npuzzles*1000LL));
        printf("%10ld  %5.2f%%  puzzles solved without guessing\n", no_guess_cnt.load(), (double)no_guess_cnt.load()/(double)solved_count.load()*100);
        printf("%10lld  %5.2f/puzzle  total guesses\n", guesses.load(), (double)guesses.load()/(double)solved_count.load());
        printf("%10lld  %5.2f/puzzle  total back tracks\n", trackbacks.load(), (double)trackbacks.load()/(double)solved_count.load());
        printf("%10lld  %5.2f/puzzle  total digits entered and retracted\n", digits_entered_and_retracted.load(), (double)digits_entered_and_retracted.load()/(double)solved_count.load());
        printf("%10lld  %5.2f/puzzle  total 'rounds'\n", past_naked_count.load(), (double)past_naked_count.load()/(double)solved_count.load());
        printf("%10lld  %5.2f/puzzle  naked sets found\n", naked_sets_found.load(), naked_sets_found.load()/(double)solved_count.load());
        printf("%10lld %6.2f/puzzle  naked sets searched\n", naked_sets_searched.load(), naked_sets_searched.load()/(double)solved_count.load());
    }

    return 0;
}
