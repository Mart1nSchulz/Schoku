This benchmark was collected on AMD Ryzen 7 Mobile 4700U CPU, with 16 GB of memory.
09-10-2026

The benchmark is from tdoku, against several well-known high-performance solvers.

OS/Compiler: Windows11,Cygwin / gcc 11.4

This table shows the benchmark results for some solvers actually verifying uniqueness:

|data/puzzles0_kaggle                  |  puzzles/sec|  usec/puzzle|   %no_guess|  guesses/puzzle|
|--------------------------------------|------------:|------------:|-----------:|---------------:|
|jczsolve                   B/shr.../m.|      570554 |        1.75 |     100.0% |           0.00 |
|schoku                     C/shrc++/m+|     1740910 |        0.57 |     100.0% |           0.00 |
|tdoku                      T/shrc+./m+|      765052 |        1.31 |     100.0% |           0.00 |

|data/puzzles1_unbiased                |  puzzles/sec|  usec/puzzle|   %no_guess|  guesses/puzzle|
|--------------------------------------|------------:|------------:|-----------:|---------------:|
|jczsolve                   B/shr.../m.|      285544 |        3.50 |      40.0% |           1.88 |
|schoku                     C/shrc++/m+|      346841 |        2.88 |      51.9% |           0.93 |
|tdoku                      T/shrc+./m+|      267935 |        3.73 |      47.3% |           1.14 |

|data/puzzles2_17_clue                 |  puzzles/sec|  usec/puzzle|   %no_guess|  guesses/puzzle|
|--------------------------------------|------------:|------------:|-----------:|---------------:|
|jczsolve                   B/shr.../m.|      278864 |        3.59 |      70.5% |           1.71 |
|schoku                     C/shrc++/m+|      389201 |        2.57 |      82.5% |           0.48 |
|tdoku                      T/shrc+./m+|      246468 |        4.06 |      78.7% |           0.62 |

|data/puzzles4_forum_hardest_1905      |  puzzles/sec|  usec/puzzle|   %no_guess|  guesses/puzzle|
|--------------------------------------|------------:|------------:|-----------:|---------------:|
|jczsolve                   B/shr.../m.|       14637 |       68.32 |       0.0% |         138.93 |
|schoku                     C/shrc++/m+|       22891 |       43.69 |       0.0% |          32.59 |
|tdoku                      T/shrc+./m+|       19267 |       51.90 |       0.0% |          55.03 |

|data/puzzles7_serg_benchmark          |  puzzles/sec|  usec/puzzle|   %no_guess|  guesses/puzzle|
|--------------------------------------|------------:|------------:|-----------:|---------------:|
|jczsolve                   B/shr.../m.|      250235 |        4.00 |       0.0% |           7.09 |
|schoku                     C/shrc++/m+|      344361 |        2.90 |       0.0% |           6.39 |
|tdoku                      T/shrc+./m+|      318625 |        3.14 |       0.0% |           7.14 |

|data/puzzles8_gen_puzzles             |  puzzles/sec|  usec/puzzle|   %no_guess|  guesses/puzzle|
|--------------------------------------|------------:|------------:|-----------:|---------------:|
|jczsolve                   B/shr.../m.|     1012063 |        0.99 |      97.4% |           0.31 |
|schoku                     C/shrc++/m+|     2797810 |        0.36 |      97.4% |           0.27 |
|tdoku                      T/shrc+./m+|     2801407 |        0.36 |      97.5% |           0.28 |

This table shows the benchmark results for solvers without verification of uniqueness:

|data/puzzles0_kaggle                  |  puzzles/sec|  usec/puzzle|   %no_guess|  guesses/puzzle|
|--------------------------------------|------------:|------------:|-----------:|---------------:|
|jczsolve                   B/shr.../m.|      463244 |        2.16 |     100.0% |           0.00 |
|sk_bforce2                 B/shrc-./m+|     1152775 |        0.87 |     100.0% |           0.00 |
|Mirage                     C/sh..../m.|      603806 |        1.66 |     100.0% |           0.00 |
|schoku                     C/shrc++/m+|     1702702 |        0.59 |     100.0% |           0.00 |
|tdoku                      T/shrc+./m+|      720875 |        1.39 |     100.0% |           0.00 |

|data/puzzles1_unbiased                |  puzzles/sec|  usec/puzzle|   %no_guess|  guesses/puzzle|
|--------------------------------------|------------:|------------:|-----------:|---------------:|
|jczsolve                   B/shr.../m.|      281797 |        3.55 |      40.0% |           1.56 |
|sk_bforce2                 B/shrc-./m+|      354786 |        2.82 |      44.4% |           1.63 |
|Mirage                     C/sh..../m.|      220463 |        4.54 |      43.8% |           1.41 |
|schoku                     C/shrc++/m+|      379617 |        2.63 |      51.9% |           0.84 |
|tdoku                      T/shrc+./m+|      289055 |        3.46 |      47.3% |           1.02 |

|data/puzzles2_17_clue                 |  puzzles/sec|  usec/puzzle|   %no_guess|  guesses/puzzle|
|--------------------------------------|------------:|------------:|-----------:|---------------:|
|jczsolve                   B/shr.../m.|      276770 |        3.61 |      70.5% |           1.14 |
|sk_bforce2                 B/shrc-./m+|      382272 |        2.62 |      74.1% |           1.02 |
|Mirage                     C/sh..../m.|      176656 |        5.66 |      63.8% |           1.25 |
|schoku                     C/shrc++/m+|      410374 |        2.44 |      82.5% |           0.36 |
|tdoku                      T/shrc+./m+|      252933 |        3.95 |      78.7% |           0.49 |

|data/puzzles4_forum_hardest_1905      |  puzzles/sec|  usec/puzzle|   %no_guess|  guesses/puzzle|
|--------------------------------------|------------:|------------:|-----------:|---------------:|
|jczsolve                   B/shr.../m.|       27195 |       36.77 |       0.0% |          73.65 |
|sk_bforce2                 B/shrc-./m+|       17770 |       56.28 |       0.0% |         103.49 |
|Mirage                     C/sh..../m.|       12520 |       79.87 |       0.0% |          64.16 |
|schoku                     C/shrc++/m+|       40924 |       24.44 |       0.0% |          18.48 |
|tdoku                      T/shrc+./m+|       35744 |       27.98 |       0.0% |          30.60 |

|data/puzzles7_serg_benchmark          |  puzzles/sec|  usec/puzzle|   %no_guess|  guesses/puzzle|
|--------------------------------------|------------:|------------:|-----------:|---------------:|
|jczsolve                   B/shr.../m.|      272647 |        3.67 |       0.0% |           6.50 |
|sk_bforce2                 B/shrc-./m+|      350670 |        2.85 |       0.0% |           7.07 |
|Mirage                     C/sh..../m.|      145015 |        6.90 |       0.0% |           6.47 |
|schoku                     C/shrc++/m+|      389759 |        2.57 |       0.0% |           5.73 |
|tdoku                      T/shrc+./m+|      335385 |        2.98 |       0.0% |           6.48 |

|data/puzzles8_gen_puzzles             |  puzzles/sec|  usec/puzzle|   %no_guess|  guesses/puzzle|
|--------------------------------------|------------:|------------:|-----------:|---------------:|
|jczsolve                   B/shr.../m.|     1112996 |        0.90 |      97.4% |           0.30 |
|sk_bforce2                 B/shrc-./m+|     1789856 |        0.56 |      97.5% |           0.29 |
|Mirage                     C/sh..../m.|     1216634 |        0.82 |      97.5% |           0.27 |
|schoku                     C/shrc++/m+|     2858285 |        0.35 |      97.4% |           0.26 |
|tdoku                      T/shrc+./m+|     2870701 |        0.35 |      97.5% |           0.27 |
