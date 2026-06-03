#!/usr/bin/env bash
# Side-by-side benchmark of Schoku vs sudoku_rs_core on cuda-host2 (gcc/x86),
# stratified by test bucket. Both tools run with 8 threads.
#
# Caveat: rust `rate-batch` also computes the rating (tier + technique
# frontier) on top of solving, while schoku only solves. The comparison is
# therefore "schoku solver" vs "rust solver+rater" — slightly unfair to rust
# but it's the only batch entry-point in the rust binary.
#
# Method: min-of-N wall time per bucket per binary. Min (not mean) is more
# robust under noisy multi-tenant CPU. N defaults to 5.
set -euo pipefail

REMOTE="${REMOTE:-cuda-host2}"
N="${N:-5}"
SCHOKU_REMOTE="${SCHOKU_REMOTE:-/root/schoku/src/schoku}"
RUST_REMOTE="${RUST_REMOTE:-/root/sudoku_rs_core/target/release/sudoku_rs_core}"
DATA_REMOTE="${DATA_REMOTE:-/root/schoku/tests/data/buckets}"

# Run remotely so we don't pay ssh round-trip per timing.
# Note: do NOT use `ssh -n` — it redirects stdin from /dev/null and the
# heredoc never reaches the remote bash. The < /dev/null below makes the
# local ssh non-interactive without losing the heredoc.
ssh "$REMOTE" "bash -s" "$N" "$SCHOKU_REMOTE" "$RUST_REMOTE" "$DATA_REMOTE" <<'REMOTE'
set -uo pipefail
N="$1"; SCHOKU="$2"; RUST="$3"; DATA="$4"

[[ -x "$SCHOKU" ]] || { echo "FATAL: schoku not at $SCHOKU"; exit 2; }
[[ -x "$RUST" ]]   || { echo "FATAL: rust not at $RUST"; exit 2; }

WORK=/tmp/schoku_bench_$$
mkdir -p "$WORK"
trap 'rm -rf "$WORK"' EXIT

# best_ms <command...> — runs the command N times, prints min wall time in ms
best_ms() {
    local best_ns=""
    for _ in $(seq 1 "$N"); do
        local t0 t1 dur
        t0=$(date +%s%N)
        "$@" >/dev/null 2>&1
        t1=$(date +%s%N)
        dur=$(( t1 - t0 ))
        if [[ -z "$best_ns" || "$dur" -lt "$best_ns" ]]; then
            best_ns=$dur
        fi
    done
    # convert ns to ms with 1 decimal
    awk -v ns="$best_ns" 'BEGIN { printf "%.1f", ns/1e6 }'
}

# Header
printf "%-22s %5s %10s %10s %10s %10s\n" "bucket" "n" "schoku_ms" "rust_ms" "rust/schoku" "ms_per_puz"
printf "%-22s %5s %10s %10s %10s %10s\n" "----------------------" "-----" "----------" "----------" "----------" "----------"

for bucket in "$DATA"/b*.txt; do
    name=$(basename "$bucket" .txt)
    n=$(wc -l < "$bucket" | tr -d ' ')

    # Warm-up once (page cache, JIT, allocator) — discarded
    "$SCHOKU" -t8 "$bucket" "$WORK/s.sols" >/dev/null 2>&1 || true
    "$RUST" rate-batch --input "$bucket" --output "$WORK/r.jsonl" --threads 8 >/dev/null 2>&1 || true

    s_ms=$(best_ms "$SCHOKU" -t8 "$bucket" "$WORK/s.sols")
    r_ms=$(best_ms "$RUST" rate-batch --input "$bucket" --output "$WORK/r.jsonl" --threads 8)

    ratio=$(awk -v r="$r_ms" -v s="$s_ms" 'BEGIN { if (s>0) printf "%.1fx", r/s; else print "n/a" }')
    per=$(awk -v ms="$s_ms" -v n="$n" 'BEGIN { if (n>0) printf "%.3fms", ms/n; else print "n/a" }')

    printf "%-22s %5d %10s %10s %10s %10s\n" "$name" "$n" "$s_ms" "$r_ms" "$ratio" "$per"
done
REMOTE
