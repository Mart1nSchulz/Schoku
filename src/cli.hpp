// Command-line / environment parsing for the schoku CLI driver.
//
// CONTRACT: this is a private include fragment, not a self-contained
// header. It must be #included exactly once from schoku.cpp, in the
// global namespace AFTER `} // namespace Schoku`, AFTER print_help() is
// defined (so parse_args can call it), and BEFORE main(). Like
// print_help() it pulls Schoku's names in with `using namespace Schoku;`
// rather than reopening the namespace (matching make_guess.hpp's
// convention of not opening the namespace itself).
//
// parse_args owns exactly the argument/environment handling that used to
// be inline at the top of main():
//   - SCHOKU_DBG_FILTER / SCHOKU_GUESS_REPORT getenv blocks
//   - the program-name skip (argc--/argv++)
//   - filling the `opts[]` command-line reflection buffer for -x output
//   - the whole `while ( argc && argv[0][0]=='-' ) { ... }` option loop,
//     including the "--"/"-" terminator (opts_terminated) and the
//     post-loop positional-option rejection guard
//   - the `if ( rules != Regular ) { ...; mode_uqr = false; }` suppression
//
// It mutates the Schoku file-scope globals (debug, reportstats, verify,
// thorough_check, numthreads, warnings, report_guess_puzzles, rules,
// mode_sets/newsets/uqr/fish, trace_out_path, dbgprintfilter, cl2txt[])
// and calls omp_set_num_threads() / print_help() exactly as before.
//
// Communication back to main(): argc/argv are taken by reference and are
// advanced past the consumed options, so on return argv[0]/argv[1] are
// the remaining positional (input/output) tokens — main applies the
// "puzzles.txt"/"solutions.txt" defaults via its `argc>0?argv[0]:...`
// lines exactly as before. line_to_solve is an out-param.
#pragma once

static void parse_args(int &argc, const char **&argv, int &line_to_solve,
                       char opts[], size_t opts_size) {
using namespace Schoku;

    // the debug dbgprintf and dbgprintfilter are not used in checked-in code
    // they are initialized here at no cost just in case...
    const char *schoku_dbg_filter = getenv("SCHOKU_DBG_FILTER");
    if ( schoku_dbg_filter ) {
       unsigned off = 0;
       while ( schoku_dbg_filter[off] == '0' ) {
           off++;
       }
       if ( schoku_dbg_filter[off] == 'x' ) {
            sscanf(&schoku_dbg_filter[off+1], "%x", &dbgprintfilter);
       } else {
            sscanf(&schoku_dbg_filter[off], "%d", &dbgprintfilter);
       }
    }
    const char *schoku_report_guess = getenv("SCHOKU_GUESS_REPORT");
    if ( schoku_report_guess ) {
       if ( schoku_report_guess[0] == '1' ) {
            report_guess_puzzles = 1;
       }
    }

    if ( argc > 0 ) {
        argc--;
        argv++;
    }

    // Buffer holds the reflected command line for "-x" stats output (may
    // be silently truncated for very long argv; truncation is harmless to
    // solver behavior). Original was [80] which overflows on realistic
    // absolute paths and triggers SIGABRT under fortified libcs (Apple
    // clang/macOS). Sized to accommodate typical CLI plus two long file
    // paths; snprintf bounds the writes regardless of argv length.
    size_t used = 0;
    for (int i = 0; i < argc; i++) {
        if (used >= opts_size - 1) break;
        int n = snprintf(opts + used, opts_size - used, "%s ", argv[i]);
        if (n < 0) break;
        used += (size_t)n;
        if (used >= opts_size - 1) { opts[opts_size - 1] = 0; break; }
    }

    // Set when the user explicitly ends option parsing with "--" (or a
    // bare "-"). Only then is a file argument that begins with "-"
    // intentional; otherwise such a token is a misplaced option and is
    // rejected below (see the positional-argument guard after this loop).
    bool opts_terminated = false;
    while ( argc && argv[0][0] == '-' ) {
        if (argv[0][1] == 0) {
            argv++; argc--;
            opts_terminated = true;
            break;
        }
        // Long options. A bare "--" terminates option parsing (POSIX
        // convention), so positional args beginning with "-" remain
        // reachable. --trace-out FILE | --trace-out=FILE: FILE = "-"
        // means stdout. The file is opened in append mode so a single
        // multi-thread Schoku run can extend an existing trace; cross-
        // process appends to the same file are NOT guaranteed atomic at
        // line granularity (libc FILE locking is per-FILE-object).
        if (argv[0][1] == '-') {
            if (argv[0][2] == 0) {     // bare "--"
                argv++; argc--;
                opts_terminated = true;
                break;
            }
            if (strcmp(argv[0], "--trace-out") == 0) {
                if (argc < 2) {
                    fprintf(stderr, "--trace-out requires a path argument\n");
                    exit(1);
                }
                trace_out_path = argv[1];
                argc -= 2; argv += 2;
                continue;
            }
            if (strncmp(argv[0], "--trace-out=", 12) == 0) {
                trace_out_path = argv[0] + 12;
                argc--; argv++;
                continue;
            }
            fprintf(stderr, "invalid long option: %s\n", argv[0]);
            argc--; argv++;
            continue;
        }
        switch(argv[0][1]) {
        case 'c':
             thorough_check=1;
             break;
        case 'd':
             debug=1;
             if ( argv[0][2] && isdigit(argv[0][2]) ) {
                 sscanf(&argv[0][2], "%d", &debug);
             }
             break;
        case 'h':
             print_help();
             exit(0);
             break;
        case 'l':    // line of puzzle to solve
             sscanf(argv[0]+2, "%d", &line_to_solve);
             break;
        case 'm':
             for ( unsigned char p=2; argv[0][p] && p<6; p++) {
                 switch (toupper(argv[0][p])) {
#ifdef OPT_NEWSETS
                case 'N':        // see OPT_NEWSETS
                    mode_newsets = true;
                    break;
#endif
#ifdef OPT_SETS
                case 'S':        // see OPT_SETS
                    mode_sets = true;
                    break;
#endif
#ifdef OPT_UQR
                case 'U':        // see OPT_UQR
                    mode_uqr = true;
                    break;
#endif
#ifdef OPT_FSH
                case 'F':        // see OPT_FSH
                    mode_fish = true;
                    break;
#endif
                default:
                    printf("invalid mode %c\n", argv[0][p]);
                }
            }
            break;
        case 'r':    // rules
            if ( argv[0][2] ) {
                switch (toupper(argv[0][2])) {
                case 'R':        // defaul rules (fastest):
                                 // assume regular puzzle with a unique solution
                                 // not suitable for puzzles with multiple solutions
                    rules = Regular;
                    break;
                case 'O':        // find one solution without making assumptions
                    rules = FindOne;
                    break;
                case 'M':        // check for multiple solutions
                    rules = Multiple;
                    break;
                default:
                    printf("invalid puzzle rules option %c\n", argv[0][2]);
                    break;
                }
            }
            break;
        case 't':    // set number of threads
             if ( argv[0][2] && isdigit(argv[0][2]) ) {
                 sscanf(&argv[0][2], "%d", &numthreads);
                 if ( numthreads != 0 ) {
                     omp_set_num_threads(numthreads);
                 }
             }
             break;
        case 'v':    // verify
             verify=1;
             break;
        case 'w':    // display warnings
             warnings = 1;
             break;
        case 'x':    // stats output
             reportstats=1;
             break;
        case 'y':    // timing stats only
             reporttimings=1;
             break;
        case '#':    // row/col numbering base
             if ( argv[0][2] && isdigit(argv[0][2]) ) {
                 int displaybase = 0;
                 sscanf(&argv[0][2], "%d", &displaybase);
                 if ( displaybase == 1 ) {
                     for ( int i=0; i<81; i++ ) {
                        cl2txt[i][1]++;
                        cl2txt[i][3]++;
                     }
                 }
             }
             break;
        default:
             printf("invalid option: %s\n", argv[0]);
             break;
        }
        argc--, argv++;
    }

    // Reject option-looking tokens that survive in positional (file)
    // position. Schoku stops option parsing at the first non-option
    // argument, so a flag placed AFTER the input file (e.g.
    // `schoku puzzles.txt -x`) would otherwise be silently taken as the
    // OUTPUT filename — creating a junk file literally named "-x" /
    // "--trace-out" while the intended flag never takes effect. Surface
    // the mistake instead. A leading-"-" filename is still reachable
    // intentionally via the "--" terminator (which sets opts_terminated).
    if ( !opts_terminated ) {
        for ( int i = 0; i < argc; i++ ) {
            if ( argv[i][0] == '-' && argv[i][1] != 0 ) {
                fprintf(stderr,
                    "Error: '%s' looks like an option but appears after a file argument.\n"
                    "Options must precede the input/output file arguments. If you really\n"
                    "mean a filename that starts with '-', put '--' before it.\n",
                    argv[i]);
                exit(1);
            }
        }
    }

    // suppress uqr mode if unique checking is requested,
    // avoiding severe complications in the code.
    if ( rules != Regular ) {
        if ( mode_uqr && warnings != 0 ) {
            printf("uqr checking mode ( -mU ) disabled when not under default Regular rules\n");
        }
        mode_uqr = false;
    }
}
