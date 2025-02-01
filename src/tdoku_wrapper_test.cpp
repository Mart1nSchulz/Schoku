#include <chrono>
#include <stdio.h>
#include <cstdlib>

extern "C"
size_t OtherSolverSchoku(const char *input, size_t limit, uint32_t /*unused_configuration*/,
                             char *solution, size_t *num_guesses);


// simple wrapper to test the library interface
int main(int argc, const char *argv[]) {
    argc--; argv++;
    const char *ifn = argc > 0? argv[0] : "puzzles.txt";
    
    FILE *f = fopen(ifn, "rb");

    // test for files not existing
    if (f == 0) {
        fprintf(stderr, "Error! Could not open file %s\n", ifn);
        return -1;
    }
    
    // find length of file
    fseek(f, 0, SEEK_END);
    size_t fsize = ftell(f);
printf("file size: %ld\n", fsize);
    fseek(f, 0, SEEK_SET);

    // deal with the part that says how many sudokus there are
    char *string = (char *)malloc(fsize + 1);
    fread(string, 1, fsize, f);
    fclose(f);

	size_t npuzzles = fsize/82;
printf("npuzzles = %d\n", (int)npuzzles);
    char output[81];
    size_t num_guesses = 0;
    size_t num_guesses_all = 0;
    bool success = true;
	auto starttime = std::chrono::steady_clock::now();

    for (unsigned int i = 0; i < npuzzles*82; i+=82) {
        // solve the grid in place
        size_t ret = OtherSolverSchoku(string+i, 0, 0, output, &num_guesses);
        num_guesses_all += num_guesses;

        if ( ret != 1 ) {
           success = false;
           printf("line %d: fail\n", i/82+1);
        }
    }
    if ( ! success ) {
        printf("There were errors\n");
    }
    long long duration = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration(std::chrono::steady_clock::now() - starttime)).count();
printf("guesses=%ld\n", num_guesses_all);
    printf("%8.1lfms  %6.2lf\u00b5s/puzzle  solving time\n", (double)duration/1000000, (double)duration/(npuzzles*1000LL));
}

