#!/usr/bin/env python3
"""Validate the `puzzle,solution` lines emitted by `schoku <input> <output>`.

Each line is two 81-char fields separated by a comma:

    puzzle,solution

For every line we check:

    1. Both halves have length 81.
    2. The solution string consists of digits 1..9 only.
    3. Each row, column, and 3x3 box of the solution is a permutation of 1..9.
    4. Each non-blank cell in the puzzle (digit '1'..'9') matches the solution
       at the same index — i.e. the solution is consistent with the puzzle's
       clues. (Blank cells in the puzzle may be '.', '0', or any other char;
       we treat anything outside '1'..'9' as blank.)

This catches:
    * a mutation that produces a "valid-looking" but wrong solution
    * a solution that doesn't actually solve the puzzle (changes a clue)
    * truncated / corrupt output

Exit 0 only if every line passes. Prints up to `--max-fail` failure lines.

This is *orthogonal* to the SHA-golden parity check: the goldens detect any
output drift; the validator detects whether the output is still semantically
correct. Together they catch the case where a mutation produces output that
matches the SHA but is wrong (impossible by definition), and the harder case
where the SHA needs updating because the mutation legitimately found a
different valid solution path.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path


def check_solution(sol: str) -> str | None:
    """Return None on valid, or a short reason string on failure."""
    if len(sol) != 81:
        return f"solution length {len(sol)} != 81"
    for c in sol:
        if c < "1" or c > "9":
            return f"solution contains non-1-9 char {c!r}"
    # Rows
    for r in range(9):
        row = sol[r * 9 : r * 9 + 9]
        if len(set(row)) != 9:
            return f"row {r} not a permutation: {row}"
    # Cols
    for c in range(9):
        col = "".join(sol[r * 9 + c] for r in range(9))
        if len(set(col)) != 9:
            return f"col {c} not a permutation: {col}"
    # Boxes
    for br in range(3):
        for bc in range(3):
            cells = [
                sol[(br * 3 + dr) * 9 + (bc * 3 + dc)]
                for dr in range(3)
                for dc in range(3)
            ]
            if len(set(cells)) != 9:
                return f"box ({br},{bc}) not a permutation: {''.join(cells)}"
    return None


def check_consistency(puz: str, sol: str) -> str | None:
    if len(puz) != 81:
        return f"puzzle length {len(puz)} != 81"
    for i, ch in enumerate(puz):
        if "1" <= ch <= "9" and ch != sol[i]:
            return f"clue mismatch at index {i}: puzzle={ch} solution={sol[i]}"
    return None


def normalise_clues(s: str) -> str:
    """Normalise a puzzle string so blanks compare equal regardless of '.'/'0'."""
    return "".join(c if "1" <= c <= "9" else "." for c in s)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path,
                   help="schoku output: lines of `puzzle,solution`")
    p.add_argument("--source", type=Path, default=None,
                   help=("optional source-of-truth puzzle file (one 81-char "
                         "puzzle per line). When set, the validator also "
                         "verifies the emitted puzzle prefix matches the "
                         "corresponding source line — catching mutations "
                         "that rewrite or drop clues in the output."))
    p.add_argument("--max-fail", type=int, default=5,
                   help="stop after printing this many failures (default 5)")
    p.add_argument("--quiet", action="store_true",
                   help="don't print the OK summary on success")
    args = p.parse_args()

    source_lines: list[str] | None = None
    if args.source is not None:
        with args.source.open() as fh:
            source_lines = [
                line.rstrip("\n").rstrip("\r")
                for line in fh
                if line.strip()
            ]

    fails = 0
    total = 0
    with args.input.open() as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.rstrip("\n").rstrip("\r")
            if not line:
                continue
            total += 1
            if "," not in line:
                if fails < args.max_fail:
                    print(f"line {lineno}: missing comma separator", file=sys.stderr)
                fails += 1
                continue
            puz, sol = line.split(",", 1)
            r = check_solution(sol)
            if r is None:
                r = check_consistency(puz, sol)
            if r is None and source_lines is not None:
                idx = total - 1
                if idx >= len(source_lines):
                    r = (f"input has more lines than source "
                         f"({total} > {len(source_lines)})")
                else:
                    src = source_lines[idx][:81]
                    if normalise_clues(src) != normalise_clues(puz):
                        r = (f"emitted puzzle differs from source: "
                             f"src={normalise_clues(src)} "
                             f"out={normalise_clues(puz)}")
            if r is not None:
                if fails < args.max_fail:
                    print(f"line {lineno}: {r}", file=sys.stderr)
                fails += 1

    if fails:
        print(f"validate_solutions: {fails}/{total} INVALID", file=sys.stderr)
        return 1
    if not args.quiet:
        print(f"validate_solutions: {total} OK", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
