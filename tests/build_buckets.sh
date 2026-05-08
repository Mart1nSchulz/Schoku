#!/usr/bin/env bash
# Build the 11 bucket test inputs from:
#   1. big5000 — re-rated and classified by tests/bucketize.py
#   2. topup parquets — pre-generated puzzles for sparse fish/UR buckets
#
# Output:
#   tests/data/buckets/b##_<name>.txt — one 81-char puzzle per line
#
# Bucket policy: see tests/bucketize.py docstring (max-tier technique in
# rated frontier). The naked_quad bucket is merged into the triple bucket
# (`b04_triple_quad`) — both exercise Schoku's `do_naked_sets_main` /
# `OPT_SETS` path, and pure-quad puzzles are generator-rare.
#
# Each bucket is capped at $CAP puzzles (default 100) via head -n; the cap
# is applied AFTER concatenation so additions land deterministically at the
# tail. The seed for any reverse-construct top-up step is fixed in the
# corresponding generator invocation (see tests/topup_buckets.sh).

set -euo pipefail
cd "$(dirname "$0")/.."

CAP="${CAP:-100}"
RUST_BIN="${RUST_BIN:-/Users/dleonenko/latent-reasoning-design/tools/sudoku_rs_core/target/release/sudoku_rs_core}"
BIG5000="tests/data/puzzles_big5000.txt"
OUT="tests/data/buckets"
TOPUP_DIR="tests/data/topup"
WORK="${WORK:-/tmp/schoku_buckets_$$}"

[[ -x "$RUST_BIN" ]] || { echo "FATAL: rust binary not found at $RUST_BIN" >&2; exit 2; }
[[ -f "$BIG5000" ]]  || { echo "FATAL: $BIG5000 missing" >&2; exit 2; }

mkdir -p "$OUT" "$WORK"
trap 'rm -rf "$WORK"' EXIT
# Clean previous bucket outputs so stale files (e.g. from a previous policy
# revision) don't leak into the new build. Only delete files matching our
# naming convention to avoid accidental damage.
rm -f "$OUT"/b[0-9][0-9]_*.txt 2>/dev/null || true

echo "=== Step 1: rate-batch big5000 ==="
"$RUST_BIN" rate-batch --input "$BIG5000" --output "$WORK/big5000.jsonl" --threads 0 \
    | tee "$WORK/rate.log" >&2

echo "=== Step 2: classify into buckets ==="
tests/bucketize.py --input "$WORK/big5000.jsonl" --out-dir "$WORK/buckets" >&2

echo "=== Step 3: merge topup (xwing, swordfish, jellyfish) ==="
# Topup files are generated separately by tests/topup_buckets.sh and committed
# to tests/data/topup/. They exist as pre-rated 81-char text files.
for stem in xwing swordfish jellyfish; do
    src="$TOPUP_DIR/$stem.txt"
    if [[ -f "$src" ]]; then
        bucket=$(case $stem in
            xwing) echo "b06_xwing";;
            swordfish) echo "b07_swordfish";;
            jellyfish) echo "b08_jellyfish";;
        esac)
        cat "$src" >> "$WORK/buckets/$bucket.txt" || true
        # b08_jellyfish may not exist in big5000 → create from scratch
        [[ -s "$WORK/buckets/$bucket.txt" ]] || cat "$src" > "$WORK/buckets/$bucket.txt"
    fi
done

echo "=== Step 4: merge b05_quad → b04_triple_quad ==="
# Pure-quad puzzles are generator-rare. Schoku-side they exercise the same
# code path as triples (do_naked_sets_main / OPT_SETS), so we merge them.
if [[ -f "$WORK/buckets/b04_triple.txt" || -f "$WORK/buckets/b05_quad.txt" ]]; then
    cat "$WORK/buckets/b04_triple.txt" 2>/dev/null > "$WORK/buckets/b04_triple_quad.txt" || true
    cat "$WORK/buckets/b05_quad.txt" 2>/dev/null >> "$WORK/buckets/b04_triple_quad.txt" || true
    rm -f "$WORK/buckets/b04_triple.txt" "$WORK/buckets/b05_quad.txt"
fi

echo "=== Step 5: cap each bucket at CAP=$CAP and emit final files ==="
# b12_invalid is excluded from the final test corpus on purpose: those rows
# represent puzzles the Rust rater rejected (rater_error or non-unique
# solution). Schoku's behaviour on them depends on -rO / -ms / -mn modes and
# is already covered by the dedicated puzzles_invalid5 dataset; mixing them
# into the bucketed parity matrix would muddy the localisation signal.
shopt -s nullglob
for f in "$WORK/buckets/"*.txt; do
    base=$(basename "$f")
    if [[ "$base" == "b12_invalid.txt" ]]; then
        n=$(wc -l < "$f" | tr -d ' ')
        printf "%-22s %4d puzzles (skipped — see puzzles_invalid5)\n" "$base" "$n" >&2
        continue
    fi
    head -n "$CAP" "$f" > "$OUT/$base"
    printf "%-22s %4d puzzles\n" "$base" "$(wc -l < "$OUT/$base")" >&2
done
shopt -u nullglob
echo "=== done ==="
