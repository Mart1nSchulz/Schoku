#!/usr/bin/env python3
"""Classify puzzles into buckets by the most-advanced technique their
solution path requires. Reads a JSONL file produced by

    sudoku_rs_core rate-batch --input <puzzles.txt> --output <jsonl>

and writes one bucket file per category to <out-dir>/b##_<name>.txt.

Bucket policy: the bucket level is the *maximum* tier among techniques in
``frontier``; the puzzle goes into that bucket. Empty frontier falls back to
tier:
    * T1       -> b01_singles (already solvable by singles)
    * T4Plus   -> b11_t4_hard
    * Invalid  -> b12_invalid

This is more discriminating than `frontier[-1]` because the rater records
techniques in cascade-order, and a hard technique often appears *before* a
trivial cleanup technique.
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

# Levels: higher = more advanced. Mapping is keyed off the
# technique-name strings emitted by sudoku_rs_core's rater.
TECH_LEVEL: dict[str, int] = {
    "naked_single": 0,
    "hidden_single": 0,
    "locked_pointing": 1,
    "locked_claiming": 1,
    "naked_pair": 2,
    "hidden_pair": 2,
    "naked_triple": 3,
    "hidden_triple": 3,
    "naked_quad": 4,
    "hidden_quad": 4,
    "xwing": 5,
    "x_wing": 5,
    "swordfish": 6,
    "jellyfish": 7,
    # All UR subtypes collapse to one Schoku-bucket (OPT_UQR is monolithic).
    "ur_type1": 8,
    "ur_type2": 8,
    "ur_type3": 8,
    "ur_type4": 8,
    "ur_type5": 8,
    "ur_type6": 8,
    "unique_rectangle": 8,
    # Schoku does NOT implement these — they fall through to phase_guess.
    "xy_wing": 9,
    "xyz_wing": 9,
    "simple_coloring": 9,
    "skyscraper": 9,
    "two_string_kite": 9,
    "empty_rectangle": 9,
    "bug": 9,
    "aic": 9,
    "als_xz": 9,
    # Finned variants (rare in current rater output)
    "finned_x_wing": 5,
    "finned_swordfish": 6,
    # Squirmbag (size-5 fish): Schoku's fish_names array carries the label
    # but the Rust rater currently does not emit it. Mapped pre-emptively so
    # the strict-unknown check below stays useful.
    "squirmbag": 8,
}

# (level, slug) ordered by level. Slug becomes filename.
BUCKETS: "OrderedDict[int, str]" = OrderedDict([
    (0, "b01_singles"),
    (1, "b02_locked"),
    (2, "b03_pair"),
    (3, "b04_triple"),
    (4, "b05_quad"),
    (5, "b06_xwing"),
    (6, "b07_swordfish"),
    (7, "b08_jellyfish"),
    (8, "b09_ur"),
    (9, "b10_chains"),
])
BUCKET_T4_HARD = "b11_t4_hard"      # T4Plus with empty frontier — pure guess.
BUCKET_INVALID = "b12_invalid"      # rater_error or non-unique solution.


def classify(rec: dict, *, strict: bool = True) -> str:
    if rec.get("rater_error") or not rec.get("unique_solution", True):
        return BUCKET_INVALID
    frontier = rec.get("frontier") or []
    if frontier:
        # Highest level among recognised techniques. We deliberately FAIL
        # LOUD on unknown technique names: a new or renamed rater technique
        # could otherwise silently demote puzzles into easier buckets,
        # defeating the localisation goal. Any addition to the rater
        # vocabulary must be reflected in TECH_LEVEL.
        unknown = [t for t in frontier if t not in TECH_LEVEL]
        if unknown:
            msg = (f"unknown technique(s) in frontier: {unknown!r} "
                   f"(puzzle index {rec.get('i', '?')}). Add to TECH_LEVEL.")
            if strict:
                raise ValueError(msg)
            print(f"WARN: {msg}", file=sys.stderr)
        levels = [TECH_LEVEL[t] for t in frontier if t in TECH_LEVEL]
        if not levels:
            return BUCKET_T4_HARD
        return BUCKETS[max(levels)]
    # Empty frontier: classify by tier.
    tier = rec.get("tier", "Invalid")
    if tier == "T1":
        return BUCKETS[0]
    if tier == "T4Plus":
        return BUCKET_T4_HARD
    return BUCKET_INVALID


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path,
                   help="JSONL produced by sudoku_rs_core rate-batch")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--summary-only", action="store_true",
                   help="just print the counts; don't write files")
    p.add_argument("--no-strict", action="store_true",
                   help=("warn instead of raising when an unknown technique "
                         "appears in the frontier (escape hatch — by default "
                         "unknown names are a hard failure)"))
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    files: dict[str, list[str]] = {}

    with args.input.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            bucket = classify(rec, strict=not args.no_strict)
            counts[bucket] = counts.get(bucket, 0) + 1
            files.setdefault(bucket, []).append(rec["puzzle"])

    # Emit files (sorted bucket name = stable order).
    if not args.summary_only:
        for bucket, puzzles in files.items():
            out_path = args.out_dir / f"{bucket}.txt"
            with out_path.open("w") as f:
                for puz in puzzles:
                    f.write(puz + "\n")

    # Print summary on stderr (sorted by bucket name for determinism).
    print(f"{'bucket':<22} {'count':>6}", file=sys.stderr)
    print("-" * 30, file=sys.stderr)
    for bucket in sorted(counts):
        print(f"{bucket:<22} {counts[bucket]:>6}", file=sys.stderr)
    print("-" * 30, file=sys.stderr)
    print(f"{'TOTAL':<22} {sum(counts.values()):>6}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
