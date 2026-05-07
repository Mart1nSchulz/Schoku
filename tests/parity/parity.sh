#!/usr/bin/env bash
# Run the locally-built schoku binary against tests/parity/matrix.txt and
# compare each output's SHA256 against the corresponding tests/golden/*.sha256.
#
# Exit 0 if every row matches; 1 otherwise. Prints a one-line PASS/FAIL per row.
#
# Usage:
#   tests/parity/parity.sh [path/to/schoku]
#       default:  picks up SCHOKU env var, else <repo>/src/schoku.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
MATRIX="$ROOT/parity/matrix.txt"
GOLDEN="$ROOT/golden"
DATA="$ROOT/data"
# /tmp on macOS has a different mount than /var/folders/...; the latter
# triggers a SIGABRT during libomp shutdown when schoku mmaps an output
# file there. Pin to /tmp so behaviour is consistent across platforms.
WORK="${WORK:-/tmp/schoku_parity_$$}"
mkdir -p "$WORK"
trap 'rm -rf "$WORK"' EXIT

BIN="${1:-${SCHOKU:-$ROOT/../src/schoku}}"
[[ -x "$BIN" ]] || { echo "schoku binary not found or not executable: $BIN" >&2; exit 2; }

shasum_cmd=$(command -v sha256sum >/dev/null 2>&1 && echo "sha256sum" || echo "shasum -a 256")

fails=0; passes=0
while IFS=$'\t' read -r dataset flags label; do
    [[ "$dataset" =~ ^# ]] && continue
    [[ -z "${dataset:-}" ]] && continue

    in="$DATA/${dataset}.txt"
    out="$WORK/${dataset}_${label}.sols"
    expect_file="$GOLDEN/${dataset}_${label}.sha256"
    expect=$(cat "$expect_file" 2>/dev/null || echo "MISSING")

    "$BIN" $flags "$in" "$out" >/dev/null 2>&1 || true
    if [[ ! -f "$out" ]]; then
        printf 'FAIL %-40s %s\n' "${dataset}_${label}" "no output produced"
        fails=$((fails + 1))
        continue
    fi
    got=$($shasum_cmd "$out" | awk '{print $1}')
    if [[ "$got" == "$expect" ]]; then
        printf 'PASS %-40s %s\n' "${dataset}_${label}" "$got"
        passes=$((passes + 1))
    else
        printf 'FAIL %-40s expected=%s got=%s\n' "${dataset}_${label}" "$expect" "$got"
        fails=$((fails + 1))
    fi
done < "$MATRIX"

echo "----"
echo "Parity: ${passes} pass, ${fails} fail"
[[ "$fails" -eq 0 ]]
