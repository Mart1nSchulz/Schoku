## Options Cross-Reference

| Option | 0.1 | 0.1+ | 0.2 | 0.3 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 0.9.1| 
| ---- |:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| -x | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| -c, -w | - | yes | yes | yes | yes | yes | yes | yes | yes | yes | 
| -h, -l#, -t# | - | - | - | yes | yes | yes | yes | yes | yes | yes |
| -d# | - | - | - | - | yes | yes | yes | yes | yes | yes |
| -r? | - | - | - | - | ROM | ROM | ROM | ROM | ROM | ROM |
| -m* | - | - | - | - | S | S | S,T | S,T | S,T | S, T, U|
| -v | - | - | - | - | yes | yes | yes | yes | yes | yes|
| -#1 | - | - | - | - | - | - | - | - | yes | yes|


| Compile Option | 0.1 | 0.1+ | 0.2 | 0.3 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 0.9.1 
| ---- |:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| NDEBUG | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| OPT_SETS | - | - | - | - | - | yes | yes | yes | yes | yes |
| OPT_TRIADS | - | - | - | - | - | - | - | yes | yes | yes |
| OPT_UQR | - | - | - | - | - | - | - | - | - | yes |


| Feature (CPU) | 0.1 | 0.1+ | 0.2 | 0.3 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 0.9.1 
| ---- |:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| warn lack AVX2/BMI | - | - | - | - | yes | yes | yes | yes | yes | yes | yes |
| instr BMI2/ZEN | - | - | - | options | auto | auto | auto | auto | auto | auto |


| Feature (Input) | 0.1 | 0.1+ | 0.2 | 0.3 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 0.9.1 | 
| ---- |:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| '.' same as '0' |  yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| leading comments (lines) | 1 | n | n | n | n | n | n | n | n | n |
| reject CR/LF | (no) | yes | yes | yes | yes | yes | yes | yes | yes | yes |


| Solver Feature | 0.1 | 0.1+ | 0.2 | 0.3 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 0.9.1 | 0.9.2
| ---- |:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| naked/hidden singles | partial | partial | partial | partial | yes | yes | yes | yes | yes | yes | yes
| triad updates  | - | - | - | - | - | - | yes | yes | yes | yes | yes |
| Unique Rectangles | - | - | - | - | - | - | - | - | - | yes | yes |
| Fishes | - | - | - | - | - | - | - | - | - | - | yes |
| BUG+1  | - | - | - | - | - | yes | yes | yes | yes | yes | yes |

| Recommended options | 0.1 | 0.1+ | 0.2 | 0.3 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 0.9.1 
| ---- |:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| 7_serg_benchmark | (none) | -rm | -rm | -rm | -rm | -rm | -rm | -rm | -rm | -rm |
| 8_gen_puzzles  | (none) | -c | -c | -c | -c -ro | -c -ro | -c -ro | -c -ro | -c -ro | -c -ro |


