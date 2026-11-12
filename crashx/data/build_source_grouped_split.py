#!/usr/bin/env python3
"""Create a provenance-preserving provisional split from reviewed source groups.

Historical JSONL files are read only.  Confirmed multi-clip recordings are
assigned as a unit to their majority original split; all other clips retain
their original membership.  A partial review manifest cannot certify a clean
test set, hence the output directory and report are explicitly provisional.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


SPLITS = ("train", "val", "test")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("crashx/data/splits"))
    parser.add_argument("--manifest", type=Path, default=Path("results/audits/split_leakage/confirmed_source_groups.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("crashx/data/splits_source_grouped_provisional"))
    parser.add_argument("--allow-partial-manifest", action="store_true", help="Required acknowledgement that this does not certify a clean split.")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("status") != "complete_manual_review" and not args.allow_partial_manifest:
        raise RuntimeError("Manifest is partial. Re-run with --allow-partial-manifest only to create a clearly labeled provisional split.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output_dir}")

    records = {}
    original_split = {}
    for split in SPLITS:
        for line in (args.input_dir / f"{split}.jsonl").read_text().splitlines():
            row = json.loads(line)
            video_id = str(row["video_id"])
            if video_id in records:
                raise ValueError(f"Duplicate video ID in source split: {video_id}")
            records[video_id] = row
            original_split[video_id] = split

    assigned = dict(original_split)
    assignments = []
    for group in manifest.get("groups", []):
        if group.get("decision") != "confirmed_overlap":
            continue
        members = [str(member) for member in group["members"]]
        if any(member not in records for member in members):
            raise KeyError(f"Manifest group {group['group_id']} contains a missing video")
        counts = Counter(original_split[member] for member in members)
        # Deterministic tie break: train, val, test keeps the training set large
        # and never promotes a test example into validation by accident.
        destination = max(SPLITS, key=lambda split: (counts[split], -SPLITS.index(split)))
        for member in members:
            assigned[member] = destination
        assignments.append({"group_id": group["group_id"], "members": members, "original_splits": dict(counts), "assigned_split": destination})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in SPLITS:
        rows = [records[video_id] for video_id in sorted(records) if assigned[video_id] == split]
        (args.output_dir / f"{split}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        counts[split] = len(rows)
    report = {
        "status": "provisional_source_grouped_split" if manifest.get("status") != "complete_manual_review" else "source_grouped_split",
        "source_split_dir": str(args.input_dir),
        "manifest": str(args.manifest),
        "counts": counts,
        "group_assignments": assignments,
        "warning": "Only confirmed reviewed groups are constrained. Do not call this split leak-free until the candidate review protocol is complete.",
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
