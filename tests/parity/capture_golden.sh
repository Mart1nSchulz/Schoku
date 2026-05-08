#!/usr/bin/env bash
# Capture golden SHA256 of solver output on cuda-host2 with gcc.
# Run this ONCE per matrix change; commit the resulting tests/golden/*.sha256
# files. parity.sh later compares against these.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
MATRIX="$ROOT/parity/matrix.txt"
GOLDEN="$ROOT/golden"
mkdir -p "$GOLDEN"

REMOTE="${REMOTE:-cuda-host2}"
REMOTE_DIR="${REMOTE_DIR:-/root/schoku/src}"

ssh -n "$REMOTE" "cd $REMOTE_DIR && make CXX=g++ clean >/dev/null && make CXX=g++ >/dev/null 2>&1"

while IFS=$'\t' read -r dataset flags label; do
    [[ "$dataset" =~ ^# ]] && continue
    [[ -z "$dataset" ]] && continue
    # Dataset may include a subdir (e.g. `buckets/b01_singles`). Create the
    # parent dirs both locally (for the sha256 file) and remotely (for the
    # output sols file) so the mkdir -p that parity.sh does on the read side
    # has a counterpart on the capture side.
    out_remote="/tmp/${dataset}_${label}.sols"
    out_remote_dir="$(dirname "$out_remote")"
    mkdir -p "$(dirname "$GOLDEN/${dataset}_${label}.sha256")"
    sha=$(ssh -n "$REMOTE" "mkdir -p $out_remote_dir && cd $REMOTE_DIR && ./schoku $flags ${dataset}.txt $out_remote >/dev/null 2>&1; sha256sum $out_remote" | awk '{print $1}')
    echo "$sha" > "$GOLDEN/${dataset}_${label}.sha256"
    printf '%-40s %s\n' "${dataset}_${label}" "$sha"
done < "$MATRIX"
