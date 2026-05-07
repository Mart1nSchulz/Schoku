#!/usr/bin/env bash
# CLI smoke tests: each flag is exercised with one assertion on the stdout
# pattern or output file. Cheap (~few seconds) and broadly covers the user-
# facing surface of schoku.
set -uo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
DATA="$ROOT/data"
BIN="${SCHOKU:-$ROOT/../src/schoku}"
WORK="${WORK:-/tmp/schoku_cli_$$}"
mkdir -p "$WORK"
trap 'rm -rf "$WORK"' EXIT

passes=0; fails=0
check() {                       # check <label> <expected-rc> <pattern> <command...>
    local label="$1" expect_rc="$2" pat="$3"
    shift 3
    local out
    out=$("$@" 2>&1)
    local rc=$?
    local ok=1
    if [[ "$expect_rc" != "*" ]] && [[ "$rc" -ne "$expect_rc" ]]; then ok=0; fi
    if [[ -n "$pat" ]] && ! grep -qE "$pat" <<<"$out"; then ok=0; fi
    if [[ "$ok" -eq 1 ]]; then
        printf 'PASS %s\n' "$label"
        passes=$((passes + 1))
    else
        printf 'FAIL %s (rc=%d, want %s)\n' "$label" "$rc" "$expect_rc"
        printf '  output: %s\n' "${out:0:200}"
        fails=$((fails + 1))
    fi
}

# -h prints usage block including "Synopsis"
check "-h help"      0 "Synopsis"          "$BIN" -h

# default solve produces 50 solutions
"$BIN" -t1 "$DATA/puzzles_tiny50.txt" "$WORK/s.txt" >/dev/null 2>&1
check "default solve" 0 "."  bash -c "[[ \$(wc -l < $WORK/s.txt) -eq 50 ]] && echo ok"

# -x prints the stats banner
check "-x stats"     0 "puzzles solved"    "$BIN" -x -t1 "$DATA/puzzles_tiny50.txt" "$WORK/x.txt"

# -y reports solving time (subset of -x)
check "-y timing"    0 "solving time"      "$BIN" -y -t1 "$DATA/puzzles_tiny50.txt" "$WORK/y.txt"

# -l5 limits to a single line; output should be one line of solution
"$BIN" -l5 -t1 "$DATA/puzzles_big5000.txt" "$WORK/l5.txt" >/dev/null 2>&1
check "-l5 single line" 0 "." bash -c "[[ \$(wc -l < $WORK/l5.txt) -eq 1 ]] && echo ok"

# -t1 / -t8 produce identical solutions for the same input
"$BIN" -t1 "$DATA/puzzles_tiny50.txt" "$WORK/t1.txt" >/dev/null 2>&1
"$BIN" -t8 "$DATA/puzzles_tiny50.txt" "$WORK/t8.txt" >/dev/null 2>&1
check "-t1 vs -t8 deterministic" 0 "." bash -c "diff -q $WORK/t1.txt $WORK/t8.txt && echo ok"

# Missing input file -> error message
check "no-input-file" "*" "Failed to open"  "$BIN" -t1 /nonexistent/puzzles.txt "$WORK/n.txt"

# -rO accepts puzzles that have no solution (returns whatever it finds)
check "-rO mode"     0 "."  "$BIN" -t1 -rO "$DATA/puzzles_invalid5.txt" "$WORK/r.txt"

# -ms (sets mode) produces stats output identical SHA to default
"$BIN" -x -t8 -ms "$DATA/puzzles_big5000.txt" "$WORK/ms.txt" >/dev/null 2>&1
"$BIN" -x -t8     "$DATA/puzzles_big5000.txt" "$WORK/dn.txt" >/dev/null 2>&1
check "-ms vs default same solutions" 0 "." bash -c "diff -q $WORK/ms.txt $WORK/dn.txt && echo ok"

# -#1 base for reporting (just exit-cleanly check; no easy stdout marker)
check "-#1 1-based"  0 "."  "$BIN" -#1 -x -t1 "$DATA/puzzles_tiny50.txt" "$WORK/h1.txt"

echo "----"
echo "CLI smoke: ${passes} pass, ${fails} fail"
[[ "$fails" -eq 0 ]]
