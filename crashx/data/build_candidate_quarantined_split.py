#!/usr/bin/env python3
"""Build a conservative diagnostic split by quarantining unresolved candidates.

This does not resolve source identity. It removes any validation/test clip that
appears in an unresolved cross-split candidate, preserving a stricter holdout
for low-cost diagnostics while the manual review continues.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("crashx/data/splits_source_grouped_provisional"))
    parser.add_argument("--queue", type=Path, default=Path("results/audits/split_leakage/review_queue.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("crashx/data/splits_candidate_quarantined"))
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {args.output_dir}")
    queue = json.loads(args.queue.read_text())["items"]
    quarantine = {"val": set(), "test": set()}
    for item in queue:
        for side in ("left", "right"):
            split = item[f"{side}_split"]
            if split in quarantine:
                quarantine[split].add(str(item[f"{side}_video_id"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts, removed = {}, {}
    for split in ("train", "val", "test"):
        rows = load(args.input_dir / f"{split}.jsonl")
        kept = rows if split == "train" else [row for row in rows if str(row["video_id"]) not in quarantine[split]]
        (args.output_dir / f"{split}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept))
        counts[split], removed[split] = len(kept), len(rows) - len(kept)
    report = {
        "status": "candidate_quarantined_diagnostic_split",
        "input_dir": str(args.input_dir),
        "review_queue": str(args.queue),
        "counts": counts,
        "removed_evaluation_candidates": removed,
        "warning": "Remaining clips have no unresolved candidate under the current detector. This is not a guarantee of independent source footage and is not a final benchmark split.",
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
