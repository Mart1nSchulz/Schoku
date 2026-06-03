// Stats / timing report + trailing summary warnings for the schoku driver.
//
// CONTRACT: this is a private include fragment, not a self-contained
// header. It must be #included exactly once from schoku.cpp, in the
// global namespace AFTER `} // namespace Schoku` and BEFORE main(). Like
// print_help() / cli.hpp it pulls Schoku's names in with
// `using namespace Schoku;` rather than reopening the namespace
// (matching make_guess.hpp's convention of not opening it itself).
//
// print_stats emits the -x statistics and -y timing report:
//   - the `if ( reportstats ) { ... } else if ( reporttimings ) { ... }`
//     block (the full -x stats dump and the -y timing-only line)
//   - the trailing summary `if` blocks: non_unique / not_verified /
//     unsolved warnings and the "use -ro / -rm" hint
//
// It reads the Schoku file-scope globals reportstats / reporttimings /
// verify / rules / mode_* / warnings and version_string /
// compilation_options. The reflected command line (`opts`) is local to
// main(), so it is passed in. The wall-clock start time (`starttime`) is
// owned by main() and passed in; the duration is sampled at report time
// inside the chosen branch, exactly as the original inline code did,
// preserving the original byte-for-byte output formatting.
#pragma once

static void print_stats(std::chrono::steady_clock::time_point starttime, const Schoku::Counters &global_counters,
                        size_t npuzzles, size_t outnpuzzles, const char *opts) {
using namespace Schoku;

    if ( reportstats) {
        long long duration = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration(std::chrono::steady_clock::now() - starttime)).count();
        printf("schoku version: %s\ncommand options: %s\ncompile options: %s\n", version_string, opts, compilation_options);
        printf("%10ld  %6.1f/puzzle  puzzles entered and presets\n", npuzzles, (double)global_counters.preset_count/outnpuzzles);
        printf("%10ld  %.0lf/s  puzzles solved\n", global_counters.solved_count, (double)global_counters.solved_count/((double)duration/1000000000LL));
        if ( duration/outnpuzzles > 1000 ) {
            printf("%8.1lfms  %6.2lf\u00b5s/puzzle  solving time\n", (double)duration/(double)1000000, (double)duration/(outnpuzzles*1000LL));
        } else {
            printf("%8.1lfms  %4dns/puzzle  solving time\n", (double)duration/1000000, (int)((double)duration/outnpuzzles));
        }
        if ( global_counters.unsolved_count && rules != Regular) {
            printf("%10ld  puzzles had no solution\n", global_counters.unsolved_count);
        }
        if ( rules == Multiple ) {
            printf("%10ld  puzzles had multiple solutions\n", global_counters.non_unique_count);
        }
        if ( verify ) {
            printf("%10ld  puzzle solutions were verified\n", global_counters.verified_count);
        }
        printf( "%10ld  %6.2f%%  puzzles solved without guessing\n", global_counters.no_guess_cnt, (double)global_counters.no_guess_cnt/global_counters.solved_count*100);
        printf( "%10ld  %6.2f/puzzle  guesses\n", global_counters.guesses, (double)global_counters.guesses/(double)global_counters.solved_count);
        printf( "%10ld  %6.2f/puzzle  back tracks\n", global_counters.trackbacks, (double)global_counters.trackbacks/global_counters.solved_count);
        printf("%10lld  %6.2f/puzzle  digits entered and retracted\n", global_counters.digits_entered_and_retracted, (double)global_counters.digits_entered_and_retracted/global_counters.solved_count);
        printf("%10lld  %6.2f/puzzle  'rounds'\n", global_counters.past_naked_count, (double)global_counters.past_naked_count/global_counters.solved_count);
        printf("%10lld  %6.2f/puzzle  triads resolved\n", global_counters.triads_resolved, (double)global_counters.triads_resolved/global_counters.solved_count);
        printf("%10lld  %6.2f/puzzle  triad updates\n", global_counters.triad_updates, (double)global_counters.triad_updates/global_counters.solved_count);
#if defined(OPT_SETS) || defined(OPT_NEWSETS)
        if ( mode_sets || mode_newsets ) {
            printf("%10lld  %6.2f/puzzle  naked sets found\n", global_counters.naked_sets_found, (double)global_counters.naked_sets_found/global_counters.solved_count);
            printf("%10lld  %6.2f/puzzle  naked sets searched\n", global_counters.naked_sets_searched, (double)global_counters.naked_sets_searched/global_counters.solved_count);
        }
#endif
#ifdef OPT_FSH
        if ( mode_fish ) {
            printf("%10lld  %6.2f/puzzle  fishes updated\n", global_counters.fishes_updated, (double)global_counters.fishes_updated/global_counters.solved_count);
            printf("%10lld  %6.2f/puzzle  fishes detected\n", global_counters.fishes_detected, (double)global_counters.fishes_detected/global_counters.solved_count);
         }
#endif
#ifdef OPT_UQR
        if ( mode_uqr ) {
            printf("%10lld  %6.2f/puzzle  unique rectangles avoided\n", global_counters.unique_rectangles_avoided, (double)global_counters.unique_rectangles_avoided/global_counters.solved_count);
            printf("%10lld  %6.2f/puzzle  unique rectangles checked\n", global_counters.unique_rectangles_checked, (double)global_counters.unique_rectangles_checked/global_counters.solved_count);
        }
#endif
        if ( global_counters.bug_plus1_count ) {
            printf("%10ld  bi-value universal graves avoided (BUG+1)\n", global_counters.bug_plus1_count);
        }
        if ( global_counters.bug_count ) {
            printf("%10ld  bi-value universal graves detected\n", global_counters.bug_count);
        }
        if ( global_counters.no_bivals_count ) {
            printf("%10ld  board states without bivalues\n", global_counters.no_bivals_count);
        }
    } else if ( reporttimings ) {
        long long duration = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration(std::chrono::steady_clock::now() - starttime)).count();
        if ( duration/npuzzles > 1000 ) {
            printf("%8.1lfms  %6.2lf\u00b5s/puzzle  solving time\n", (double)duration/1000000, (double)duration/(npuzzles*1000LL));
        } else {
            printf("%8.1lfms  %4dns/puzzle  solving time\n", (double)duration/1000000, (int)((double)duration/npuzzles));
        }
    }

    if ( !reportstats && rules == Multiple && global_counters.non_unique_count) {
        printf("%10ld  puzzles had more than one solution\n", global_counters.non_unique_count);
    }
    if ( verify && global_counters.not_verified_count) {
        printf("%10ld  puzzle solutions verified as not correct\n", global_counters.not_verified_count);
    }
    if ( !reportstats && global_counters.unsolved_count) {
        printf("%10ld puzzles had no solution\n", global_counters.unsolved_count);
    }
    if ( rules == Regular && (reportstats || warnings) && ( global_counters.unsolved_count || global_counters.non_unique_count || global_counters.not_verified_count) ) {
        printf("\n\tIf a puzzle may have multiple solutions use either\n\t -ro (find one solution) or -rm (check for multiple solutions)!\n");
    }
}
