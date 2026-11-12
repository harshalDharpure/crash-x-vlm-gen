#!/usr/bin/env python3
"""Create a deterministic manual-review queue from leakage audit artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-review", type=Path, default=Path("results/audits/split_leakage/sequence_review_all.json"))
    parser.add_argument("--manifest", type=Path, default=Path("results/audits/split_leakage/confirmed_source_groups.json"))
    parser.add_argument("--out", type=Path, default=Path("results/audits/split_leakage/review_queue.json"))
    args = parser.parse_args()
    reviewed = json.loads(args.sequence_review.read_text())["reviewed_candidates"]
    manifest = json.loads(args.manifest.read_text())
    resolved = {frozenset(group["members"]) for group in manifest.get("groups", [])}
    resolved |= {frozenset(pair) for pair in manifest.get("rejected_pairs", [])}
    priority = {("train", "test"): 0, ("val", "test"): 1, ("train", "val"): 2}
    queue = []
    for row in reviewed:
        pair = frozenset((row["left_video_id"], row["right_video_id"]))
        if pair in resolved:
            continue
        queue.append({**row, "priority": priority[(row["left_split"], row["right_split"])]})
    queue.sort(key=lambda row: (row["priority"], row["mean_gray_mse"], row["mean_dhash_distance"]))
    report = {
        "status": "unresolved_manual_review_queue",
        "ordering": "train-test first, then val-test, then train-val; within each, lowest sequence MSE first",
        "n_resolved_pairs": len(resolved),
        "n_unresolved_pairs": len(queue),
        "items": queue,
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"n_unresolved_pairs": len(queue), "top": queue[:10]}, indent=2))


if __name__ == "__main__":
    main()
