#!/usr/bin/env bash
# Tier 4: solution validator over the bucket inputs.
#
# Runs schoku on every tests/data/buckets/b*.txt and pipes the output through
# tests/validate_solutions.py. Catches mutations whose output:
#   * is no longer a valid Sudoku (rows/cols/boxes broken)
#   * disagrees with the puzzle's own clues
# i.e. the orthogonal axis to the SHA-golden parity check.
#
# Exit 0 only if every bucket validates clean.
set -uo pipefail
cd "$(dirname "$0")/.."

BIN="${1:-${SCHOKU:-$PWD/../src/schoku}}"
[[ -x "$BIN" ]] || { echo "FATAL: schoku not found at $BIN" >&2; exit 2; }

WORK="${WORK:-/tmp/schoku_validate_$$}"
mkdir -p "$WORK"
trap 'rm -rf "$WORK"' EXIT

fails=0; passes=0
shopt -s nullglob
for bucket in tests/data/buckets/b*.txt; do
    name=$(basename "$bucket" .txt)
    n=$(wc -l < "$bucket" | tr -d ' ')
    out="$WORK/${name}.sols"
    "$BIN" -x -t1 "$bucket" "$out" >/dev/null 2>&1 || true
    if [[ ! -f "$out" ]]; then
        printf 'FAIL %-22s no output produced\n' "$name"
        fails=$((fails + 1)); continue
    fi
    n_out=$(wc -l < "$out" | tr -d ' ')
    if [[ "$n_out" -ne "$n" ]]; then
        printf 'FAIL %-22s expected %d solutions, got %d\n' "$name" "$n" "$n_out"
        fails=$((fails + 1)); continue
    fi
    # --source ties the emitted puzzle prefix back to the input file. Without
    # this, a mutation that rewrites clues into blanks (alongside a valid
    # solution) would still pass; with it, any divergence from source clues
    # flags as failure.
    if ! tests/validate_solutions.py --quiet --source "$bucket" "$out" 2>/dev/null; then
        msg=$(tests/validate_solutions.py --source "$bucket" "$out" 2>&1 | head -3)
        printf 'FAIL %-22s validator rejected: %s\n' "$name" "${msg//$'\n'/ | }"
        fails=$((fails + 1)); continue
    fi
    printf 'PASS %-22s n=%d\n' "$name" "$n"
    passes=$((passes + 1))
done
shopt -u nullglob

echo "----"
echo "Validator: ${passes} pass, ${fails} fail"
[[ "$fails" -eq 0 ]]
