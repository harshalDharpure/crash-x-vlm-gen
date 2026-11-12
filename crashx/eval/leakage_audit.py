#!/usr/bin/env python3
"""Audit CrashX split leakage without using labels or test-time tuning.

The visual check samples fixed relative positions from every video and compares
64-bit perceptual hashes across splits.  It is a candidate detector: an emitted
pair must be manually reviewed before calling it duplicated source footage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


SPLITS = ("train", "val", "test")
SAMPLE_FRACTIONS = (0.10, 0.50, 0.90)


def load_splits(split_dir: Path) -> dict[str, list[dict[str, Any]]]:
    return {
        split: [json.loads(line) for line in (split_dir / f"{split}.jsonl").read_text().splitlines() if line.strip()]
        for split in SPLITS
    }


def normalise_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def dhash(frame: np.ndarray) -> int:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tiny = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = tiny[:, 1:] >= tiny[:, :-1]
    result = 0
    for bit in bits.ravel():
        result = (result << 1) | int(bit)
    return result


def video_hashes(path: Path) -> list[int] | None:
    capture = cv2.VideoCapture(str(path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        capture.release()
        return None
    values: list[int] = []
    for fraction in SAMPLE_FRACTIONS:
        capture.set(cv2.CAP_PROP_POS_FRAMES, min(frame_count - 1, round((frame_count - 1) * fraction)))
        ok, frame = capture.read()
        if not ok:
            capture.release()
            return None
        values.append(dhash(frame))
    capture.release()
    return values


def hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def pair_distance(left: list[int], right: list[int]) -> float:
    return float(np.mean([hamming(a, b) for a, b in zip(left, right, strict=True)]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", type=Path, default=Path("crashx/data/splits"))
    parser.add_argument("--out", type=Path, default=Path("results/audits/split_leakage/report.json"))
    parser.add_argument("--candidate-distance", type=float, default=12.0, help="Mean dHash Hamming threshold; fixed before audit.")
    args = parser.parse_args()
    if not 0.0 <= args.candidate_distance <= 64.0:
        raise ValueError("--candidate-distance must lie in [0, 64]")

    data = load_splits(args.split_dir)
    ids = {split: {str(row["video_id"]) for row in rows} for split, rows in data.items()}
    paths = {split: {str(row["video_path"]) for row in rows} for split, rows in data.items()}
    overlap = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap[f"{left}_{right}"] = {"video_id": len(ids[left] & ids[right]), "video_path": len(paths[left] & paths[right])}

    text_groups: dict[str, set[str]] = defaultdict(set)
    for split, rows in data.items():
        for row in rows:
            text_groups[normalise_text(str(row.get("explanation", "")))].add(split)
    cross_split_text = sum(1 for groups in text_groups.values() if len(groups) > 1)

    fingerprints: dict[str, list[tuple[str, list[int]]]] = defaultdict(list)
    unreadable: list[str] = []
    for split, rows in data.items():
        for row in rows:
            path = Path(row["video_path"])
            values = video_hashes(path)
            if values is None:
                unreadable.append(str(path))
            else:
                fingerprints[split].append((str(row["video_id"]), values))

    candidates: list[dict[str, Any]] = []
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        for left_id, left_hashes in fingerprints[left]:
            for right_id, right_hashes in fingerprints[right]:
                distance = pair_distance(left_hashes, right_hashes)
                if distance <= args.candidate_distance:
                    candidates.append({"left_split": left, "left_video_id": left_id, "right_split": right, "right_video_id": right_id, "mean_dhash_distance": distance})
    candidates.sort(key=lambda row: row["mean_dhash_distance"])
    report = {
        "method": {"samples_relative_positions": list(SAMPLE_FRACTIONS), "hash": "64-bit dHash", "candidate_threshold": args.candidate_distance, "interpretation": "Candidates require manual visual review; absence is not proof that source footage is independent."},
        "counts": {split: len(rows) for split, rows in data.items()},
        "exact_overlap": overlap,
        "cross_split_exact_explanation_duplicates": cross_split_text,
        "visual_hashes_computed": {split: len(rows) for split, rows in fingerprints.items()},
        "unreadable_videos": unreadable,
        "visual_near_duplicate_candidates": candidates,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"exact_overlap": overlap, "cross_split_exact_explanation_duplicates": cross_split_text, "fingerprints": report["visual_hashes_computed"], "unreadable": len(unreadable), "candidate_count": len(candidates)}, indent=2))


if __name__ == "__main__":
    main()
