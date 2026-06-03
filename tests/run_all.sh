#!/usr/bin/env bash
# Top-level test runner: builds the schoku binary if missing, then runs
# all five tiers (parity, compat unit, CLI smoke, solution validator,
# JSONL trace smoke) and reports a summary.
#
# Exit 0 only if every tier reports zero failures.
#
# Usage:
#   tests/run_all.sh           # default: macOS Apple-clang build via Makefile.mac
#   MAKEFILE=Makefile tests/run_all.sh    # use the gcc/Linux Makefile
set -uo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"

MAKEFILE="${MAKEFILE:-Makefile.mac}"
SCHOKU="${SCHOKU:-$ROOT/src/schoku}"

echo "=== build schoku ($MAKEFILE) ==="
( cd "$ROOT/src" && make -f "$MAKEFILE" >/dev/null 2>&1 )
[[ -x "$SCHOKU" ]] || { echo "FATAL: $SCHOKU not built" >&2; exit 2; }

run_tier() {                # run_tier <label> <command...>
    local label="$1"; shift
    echo
    echo "=== $label ==="
    if "$@"; then
        echo "$label: OK"
        return 0
    else
        echo "$label: FAILED"
        return 1
    fi
}

failed=0
run_tier "Tier 1 (parity)"      "$ROOT/tests/parity/parity.sh" "$SCHOKU" || failed=$((failed + 1))
run_tier "Tier 2 (compat unit)" bash -c "cd $ROOT/tests/compat && make clean >/dev/null && make check 2>&1 | grep -E '^(test_|.*failed)'" || failed=$((failed + 1))
run_tier "Tier 3 (CLI smoke)"   "$ROOT/tests/cli/cli_smoke.sh"   || failed=$((failed + 1))
run_tier "Tier 4 (validator)"   "$ROOT/tests/validate_buckets.sh" "$SCHOKU" || failed=$((failed + 1))
run_tier "Tier 5 (trace smoke)" "$ROOT/tests/trace_smoke.sh" "$SCHOKU" || failed=$((failed + 1))

echo
echo "================================="
if [[ "$failed" -eq 0 ]]; then
    echo "ALL GREEN"
    exit 0
else
    echo "$failed tier(s) FAILED"
    exit 1
fi
