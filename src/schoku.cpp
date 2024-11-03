/*
 * Schoku
 *
 * A high speed sudoku solver by M. Schulz
 *
 * Based on the sudoku solver by Mirage ( https://codegolf.stackexchange.com/users/106606/mirage )
 * at https://codegolf.stackexchange.com/questions/190727/the-fastest-sudoku-solver
 * on Sep 22, 2021
 *
 * Version 0 (original submission by Mirage)
 *
 * Note that this version does not work on Cygwin64.
 */
#include <stdio.h>
#include <omp.h>
#include <stdbool.h>
#include <intrin.h>

// lookup tables that may or may not speed things up by avoiding division
static const unsigned char box_index[81] = {
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

static const unsigned char column_index[81] = {
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

static const unsigned char row_index[81] = {
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

static const unsigned char box_start[81] = {
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

static void add_column_indices(unsigned long long indices[2], unsigned char i) {
    indices[0] |= 0x8040201008040201ULL << column_index[i];
    indices[1] |= 0x8040201008040201ULL >> (10-column_index[i]);
}

static void add_row_indices(unsigned long long indices[2], unsigned char i) {
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

static void add_box_indices(unsigned long long indices[2], unsigned char i) {
    indices[0] |= 0x1c0e07ULL << box_start[i];
    indices[1] |= 0x0381c0e0ULL >> (60-box_start[i]);
}

struct GridState {
    struct GridState* prev;                // last grid state before a guess was made, used for backtracking
    unsigned long long unlocked[2];        // for keeping track of which cells don't need to be looked at anymore. Set bits correspond to cells that still have multiple possibilities
    unsigned long long updated[2];        // for keeping track of which cell's candidates may have been changed since last time we looked for naked sets. Set bits correspond to changed candidates in these cells
    unsigned short candidates[81];        // which digits can go in this cell? Set bits correspond to possible digits
};

static void enter_digit(struct GridState* grid_state, signed char digit, unsigned char i) {
    // lock this cell and and remove this digit from the candidates in this row, column and box
    
    unsigned short* candidates = grid_state->candidates;
    unsigned long long* unlocked = grid_state->unlocked;
    unsigned long long to_update[2] = {0};
    
    if (i < 64) {
        unlocked[0] &= ~(1ULL << i);
    } else {
        unlocked[1] &= ~(1ULL << (i-64));
    }
    
    candidates[i] = 1 << digit;
    
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

static long long guesses = 0;

static struct GridState* make_guess(struct GridState* grid_state) {
    // Make a guess for the cell with the least candidates. The guess will be the lowest
    // possible digit for that cell. If multiple cells have the same number of candidates, the 
    // cell with lowest index will be chosen. Also save the current grid state for tracking back
    // in case the guess is wrong. No cell has less than two candidates.
    
    unsigned long long* unlocked = grid_state->unlocked;
    unsigned short* candidates = grid_state->candidates;
    
    // Find the cell with fewest possible candidates
    unsigned long long to_visit;
    unsigned long guess_index = 0;
    unsigned long i_rel;
    unsigned short cnt;
    unsigned short best_cnt = 16;
    
    to_visit = unlocked[0];
    while (_BitScanForward64(&i_rel, to_visit) != 0) {
        to_visit &= ~(1ULL << i_rel);
        cnt = __popcnt16(candidates[i_rel]);
        if (cnt < best_cnt) {
            best_cnt = cnt;
            guess_index = i_rel;
        }
    }
    to_visit = unlocked[1];
    while (_BitScanForward64(&i_rel, to_visit) != 0) {
        to_visit &= ~(1ULL << i_rel);
        cnt = __popcnt16(candidates[i_rel + 64]);
        if (cnt < best_cnt) {
            best_cnt = cnt;
            guess_index = i_rel + 64;
        }
    }
    
    // Find the first candidate in this cell
    unsigned long digit;
    _BitScanReverse(&digit, candidates[guess_index]);
    
    // Create a copy of the state of the grid to make back tracking possible
    struct GridState* new_grid_state = (struct GridState*) malloc(sizeof(struct GridState));
    memcpy(new_grid_state, grid_state, sizeof(struct GridState));
    new_grid_state->prev = grid_state;
    
    // Remove the guessed candidate from the old grid because if we need to get back to the old grid
    // we know the guess was wrong
    grid_state->candidates[guess_index] &= ~(1 << digit);
    if (guess_index < 64) {
        grid_state->updated[0] |= 1ULL << guess_index;
    } else {
        grid_state->updated[1] |= 1ULL << (guess_index-64);
    }
    
    // Update candidates
    enter_digit(new_grid_state, (signed char) digit, (unsigned char) guess_index);
    
    guesses++;
    
    return new_grid_state;
}


static struct GridState* track_back(struct GridState* grid_state) {
    // Go back to the state when the last guess was made
    // This state had the guess removed as candidate from it's cell
    
    struct GridState* old_grid_state = grid_state->prev;
    free(grid_state);
    
    return old_grid_state;
}


static bool solve(signed char grid[81]) {

    struct GridState* grid_state = (struct GridState*) malloc(sizeof(struct GridState));
    grid_state->prev = 0;
    unsigned long long* unlocked = grid_state->unlocked;
    unsigned long long* updated = grid_state->updated;
    unsigned short* candidates = grid_state->candidates;
    unlocked[0] = 0xffffffffffffffffULL;
    unlocked[1] = 0x1ffffULL;
    updated[0] = unlocked[0];
    updated[1] = unlocked[1];
    
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
                    grid_state = track_back(grid_state);
                    goto start;
                } else {
                    unsigned short m;
                    if (i < 64) {
                        m = (unsigned short) ((unlocked[0] >> i) & 0xffff);
                    } else {
                        m = (unsigned short) (unlocked[1] & 0xffff);
                    }                        
                    
                    __m256i a = _mm256_cmpeq_epi16(_mm256_and_si256(c, _mm256_sub_epi16(c, ones)), _mm256_setzero_si256());
                    __m256i u = _mm256_cmpeq_epi16(_mm256_and_si256(bit_mask, _mm256_set1_epi16(m)), bit_mask);
                    int mask = _mm256_movemask_epi8(_mm256_and_si256(a, u));
                    
                    if (mask) {
                        unsigned long index, digit;
                        _BitScanForward(&index, mask);
                        index = (index >> 1) + i;                            
                        _BitScanReverse(&digit, candidates[index]);
                        enter_digit(grid_state, (signed char) digit, index);
                        found = true;
                    }
                }
            }
            if (unlocked[1] & (1ULL << (80-64))) {
                if (candidates[80] == 0) {
                    // no solutions go back
                    grid_state = track_back(grid_state);
                    goto start;
                } else if (__popcnt16(candidates[80]) == 1) {
                    // Enter the digit and update candidates
                    unsigned long digit;
                    _BitScanReverse(&digit, candidates[80]);
                    enter_digit(grid_state, (signed char) digit, 80);
                    found = true;
                }
            }
        } while (found);
    }
    
    // Check if it's solved, if it ever gets solved it will be solved after looking for naked singles
    if ((unlocked[0] | unlocked[1]) == 0) {
        // Solved it
        // Free memory
        while (grid_state) {
            struct GridState* prev_grid_state = grid_state->prev;
            free(grid_state);
            grid_state = prev_grid_state;
        }
        
        // Enter found digits into grid
        for (unsigned char j = 0; j < 81; ++j) {
            unsigned long index;
            _BitScanReverse(&index, candidates[j]);
            grid[j] = (signed char) index + 49;
        }
        
        return true;
    }
    
    // Find hidden singles
    // Don't check the last column because it doesn't fit in the SSE register so it's not really worth checking
    {
        const __m128i ones = _mm_set1_epi16(1);
        const __m128i bit_mask = _mm_setr_epi16(1<<0, 1<<1, 1<<2, 1<<3, 1<<4, 1<<5, 1<<6, 1<<7);
        const __m128i shuffle_mask = _mm_setr_epi8(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1);
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
            __m128i column_mask = _mm_set1_epi16(0x01ff);
            for (unsigned char j = 0; j < 81; j += 9) {
                if (j != i) {
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
                
                __m128i a = _mm_cmpgt_epi16(_mm_and_si128(c, or_mask), _mm_setzero_si128());
                __m128i u = _mm_cmpeq_epi16(_mm_and_si128(bit_mask, _mm_set1_epi16(m)), bit_mask);
                int mask = _mm_movemask_epi8(_mm_and_si128(a, u));
                
                if (mask) {
                    unsigned long index, digit;
                    _BitScanForward(&index, mask);
                    
                    index = index/2;
                    
                    int can = ((unsigned short*) &or_mask)[index];
                    _BitScanForward(&digit, can);
                    
                    index = index + i;
                    
                    enter_digit(grid_state, (signed char) digit, index);
                    goto start;
                }
                
            } else {
                // no solutions go back
                grid_state = track_back(grid_state);
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
                unsigned long i_rel;
                _BitScanForward64(&i_rel, to_visit_n[n]);
                
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
                    if (s > cnt) {
                        grid_state = track_back(grid_state);
                        goto start;
                    } else if (s == cnt) {
                        add_column_indices(to_change, i);
                    }

                    // check row
                    a_j = _mm_load_si128((__m128i*) &candidates[9*row_index[i]]);
                    res = _mm_cmpeq_epi16(a_i, _mm_or_si128(a_i, a_j));
                    s = __popcnt16(_mm_movemask_epi8(res)) >> 1;
                    s += candidates[i] == (candidates[i] | candidates[9*row_index[i]+8]);
                    if (s > cnt) {
                        grid_state = track_back(grid_state);
                        goto start;
                    } else if (s == cnt) {
                        add_row_indices(to_change, i);
                    }
                    
                    // check box
                    unsigned short b = box_start[i];
                    a_j = _mm_set_epi16(candidates[b], candidates[b+1], candidates[b+2], candidates[b+9], candidates[b+10], candidates[b+11], candidates[b+18], candidates[b+19]);
                    res = _mm_cmpeq_epi16(a_i, _mm_or_si128(a_i, a_j));
                    s = __popcnt16(_mm_movemask_epi8(res)) >> 1;
                    s += candidates[i] == (candidates[i] | candidates[b+20]);
                    if (s > cnt) {
                        grid_state = track_back(grid_state);
                        goto start;
                    } else if (s == cnt) {
                        add_box_indices(to_change, i);
                    }
                                    
                    to_change[0] &= unlocked[0];
                    to_change[1] &= unlocked[1];
                    
                    // update candidates
                    for (unsigned char n = 0; n < 2; ++n) {
                        while (to_change[n]) {
                            unsigned long j_rel;
                            _BitScanForward64(&j_rel, to_change[n]);
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
    goto start;
    
}


int main(int argc, char *argv[]) {

    if (argc != 2) {
        printf("Error! Pass the file name as argument\n");
        return -1;
    }
    
    FILE *f = fopen(argv[1], "rb");
    
    // test for files not existing
    if (f == 0) {
        printf("Error! Could not open file %s\n", argv[1]);
        return -1;
    }
    
    // find length of file
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    // deal with the part that says how many sudokus there are
    signed char *string = malloc(fsize + 1);
    fread(string, 1, fsize, f);
    fclose(f);

    size_t p = 0;
    while (string[p] != 10) {
        ++p;
    }
    ++p;
    
    signed char *output = malloc(fsize*2 - p + 2);
    memcpy(output, string, p);
    
    // solve all sudokus and prepare output file
    int i;
    #pragma omp parallel for shared(string, output, fsize, p) schedule(dynamic)
    for (i = fsize - p - 81; i >= 0; i-=82) {
        // copy unsolved grid
        memcpy(&output[p+i*2], &string[p+i], 81);
        memcpy(&output[p+i*2+82], &string[p+i], 81);
        // add comma and newline in right place
        output[p+i*2 + 81] = ',';
        output[p+i*2 + 163] = 10;
        // solve the grid in place
        solve(&output[p+i*2+82]);
    }
    
    // create output file
    f = fopen("output.txt", "wb");
    fwrite(output, 1, fsize*2 - p + 2, f);
    fclose(f);    

    free(string);
    free(output);

    return 0;
}
