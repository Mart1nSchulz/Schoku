#!/usr/bin/env bash
# Smoke test for the JSONL trace output (--trace-out).
#
# Runs schoku with --trace-out on a small mixed-difficulty input and verifies:
#   1. every output line is a valid JSON object
#   2. every puzzle has exactly one puzzle_start and one puzzle_end
#   3. every puzzle ends solved (within Schoku's standard regime)
#   4. event types match the documented vocabulary
#   5. levels never skip (guess += 1, backtrack -= 1 from previous level)
#
# Exit 0 if all checks pass; 1 otherwise.
set -uo pipefail
cd "$(dirname "$0")/.."

BIN="${1:-${SCHOKU:-$PWD/src/schoku}}"
[[ -x "$BIN" ]] || { echo "FATAL: schoku not found at $BIN" >&2; exit 2; }

WORK="${WORK:-/tmp/schoku_trace_smoke_$$}"
mkdir -p "$WORK"
trap 'rm -rf "$WORK"' EXIT

# Pick a small mix: 2 singles, 2 xwing, 2 ur, 2 chains, 2 t4_hard = 10 puzzles
in="$WORK/in.txt"
head -2 tests/data/buckets/b01_singles.txt    >  "$in"
head -2 tests/data/buckets/b06_xwing.txt      >> "$in"
head -2 tests/data/buckets/b09_ur.txt         >> "$in"
head -2 tests/data/buckets/b10_chains.txt     >> "$in"
head -2 tests/data/buckets/b11_t4_hard.txt    >> "$in"
n_in=$(wc -l < "$in" | tr -d ' ')

trace="$WORK/trace.jsonl"
sols="$WORK/sols.txt"
"$BIN" --trace-out "$trace" -t1 "$in" "$sols" >/dev/null 2>&1
[[ -s "$trace" ]] || { echo "FAIL: empty trace"; exit 1; }

# 1. each line is parseable JSON
if ! jq -e . "$trace" >/dev/null 2>&1; then
    bad=$(jq -e . "$trace" 2>&1 | head -3)
    echo "FAIL: non-parseable JSON lines: $bad"
    exit 1
fi
echo "PASS json-parseable"

# 2. starts == ends == puzzles in input
n_start=$(jq -r 'select(.event=="puzzle_start") | .puzzle_id' "$trace" | wc -l | tr -d ' ')
n_end=$(jq -r 'select(.event=="puzzle_end") | .solved' "$trace" | wc -l | tr -d ' ')
if [[ "$n_start" != "$n_in" || "$n_end" != "$n_in" ]]; then
    echo "FAIL: expected $n_in puzzles, got starts=$n_start ends=$n_end"
    exit 1
fi
echo "PASS puzzle-start/end count ($n_in)"

# 3. every puzzle solved
n_solved=$(jq -r 'select(.event=="puzzle_end" and .solved==true)' "$trace" | wc -l | tr -d ' ')
# Each puzzle_end is one line in the JSONL stream; jq emits its object back
# in pretty form so we count by `puzzle_id` order in the input. Use a
# different filter that counts true/false.
n_solved=$(jq -r 'select(.event=="puzzle_end") | .solved' "$trace" | grep -c true)
if [[ "$n_solved" != "$n_in" ]]; then
    echo "FAIL: only $n_solved/$n_in solved"
    exit 1
fi
echo "PASS all solved ($n_solved/$n_in)"

# 4. event vocabulary
unknown=$(jq -r '.event' "$trace" | sort -u | \
    grep -vE '^(puzzle_start|puzzle_end|step)$' | head)
if [[ -n "$unknown" ]]; then
    echo "FAIL: unknown event types: $unknown"
    exit 1
fi
unknown_types=$(jq -r 'select(.event=="step") | .type' "$trace" | sort -u | \
    grep -vE '^(single|guess|backtrack)$' | head)
if [[ -n "$unknown_types" ]]; then
    echo "FAIL: unknown step types: $unknown_types"
    exit 1
fi
echo "PASS vocabulary ($(jq -r 'select(.event=="step") | .type' "$trace" | sort -u | tr '\n' ',' | sed 's/,$//'))"

# 5. level continuity per puzzle: track current level via guess/backtrack
# events. Each guess sets level = new_level; each backtrack sets level = to_level.
# Singles must occur at the prevailing level (any value, but consistent).
# We just verify guess.new_level == guess.from_level+1 and backtrack.to_level
# == backtrack.from_level-1 (this is enforced by the emitter, but check).
bad_level=$(jq -r '
    select(.event=="step")
    | if .type=="guess" then
        (if .new_level == .from_level+1 then empty else "guess: " + (.|tostring) end)
      elif .type=="backtrack" then
        (if .to_level == .from_level-1 then empty else "backtrack: " + (.|tostring) end)
      else empty end
' "$trace" | head -3)
if [[ -n "$bad_level" ]]; then
    echo "FAIL: level invariant broken: $bad_level"
    exit 1
fi
echo "PASS level-continuity"

# 6. counters match: guesses/backtracks reported in puzzle_end equal the
# number of guess/backtrack events emitted for that puzzle.
mismatches=$(python3 - <<PY
import json, sys
puzzles = {}
with open("$trace") as f:
    for line in f:
        rec = json.loads(line)
        ev = rec["event"]
        if ev == "puzzle_start":
            puzzles[rec["puzzle_id"]] = {"g":0, "b":0, "expected_g":None, "expected_b":None}
        elif ev == "step":
            # Find currently-open puzzle (the last with no end). Simple linear scan ok for 10.
            for pid, p in puzzles.items():
                if p.get("expected_g") is None:
                    open_pid = pid
            if rec["type"] == "guess":
                puzzles[open_pid]["g"] += 1
            elif rec["type"] == "backtrack":
                puzzles[open_pid]["b"] += 1
        elif ev == "puzzle_end":
            # match to the most recently opened puzzle without an end
            for pid, p in puzzles.items():
                if p.get("expected_g") is None:
                    p["expected_g"] = rec["guesses"]
                    p["expected_b"] = rec["backtracks"]
                    break
m = 0
for pid, p in puzzles.items():
    if p["g"] != p["expected_g"] or p["b"] != p["expected_b"]:
        print(f"puzzle {pid}: events g={p['g']} b={p['b']} but reported g={p['expected_g']} b={p['expected_b']}")
        m += 1
sys.exit(0 if m == 0 else 1)
PY
)
if [[ -n "$mismatches" ]]; then
    echo "FAIL: counter mismatch: $mismatches"
    exit 1
fi
echo "PASS counter consistency"

n_events=$(wc -l < "$trace" | tr -d ' ')
echo "----"
echo "Trace smoke: all checks pass ($n_events events across $n_in puzzles)"
