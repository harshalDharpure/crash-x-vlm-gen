#!/usr/bin/env python3
"""Rank visual-leakage candidates using temporally aligned frame sequences.

Unlike the first-pass dHash scan, this compares every frame over short temporal
offsets. Scores rank manual-review priority only; no numeric threshold is used
to automatically declare footage duplication.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def load_records(split_dir: Path) -> dict[str, dict[str, Path]]:
    result: dict[str, dict[str, Path]] = {}
    for split in ("train", "val", "test"):
        rows = [json.loads(line) for line in (split_dir / f"{split}.jsonl").read_text().splitlines() if line.strip()]
        result[split] = {str(row["video_id"]): Path(row["video_path"]) for row in rows}
    return result


def decode_lowres(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32))
    capture.release()
    if not frames:
        raise RuntimeError(f"Could not decode {path}")
    return np.stack(frames)


def aligned_sequence_score(left: np.ndarray, right: np.ndarray, max_offset: int) -> dict[str, float | int]:
    """Find the offset with minimum average pixel MSE across overlapping frames."""
    best: dict[str, float | int] | None = None
    for offset in range(-max_offset, max_offset + 1):
        left_start, right_start = max(0, offset), max(0, -offset)
        n = min(len(left) - left_start, len(right) - right_start)
        if n < 10:
            continue
        delta = left[left_start : left_start + n] - right[right_start : right_start + n]
        score = {"best_offset_frames": offset, "overlap_frames": n, "mean_gray_mse": float(np.mean(delta * delta)), "mean_gray_mae": float(np.mean(np.abs(delta)))}
        if best is None or score["mean_gray_mse"] < best["mean_gray_mse"]:
            best = score
    if best is None:
        raise ValueError("Videos do not have enough overlapping frames for review")
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, default=Path("results/audits/split_leakage/report.json"))
    parser.add_argument("--split-dir", type=Path, default=Path("crashx/data/splits"))
    parser.add_argument("--out", type=Path, default=Path("results/audits/split_leakage/sequence_review.json"))
    parser.add_argument("--limit-train-test", type=int, default=50)
    parser.add_argument("--limit-val-test", type=int, default=41)
    parser.add_argument("--limit-train-val", type=int, default=20)
    parser.add_argument("--max-offset", type=int, default=12)
    args = parser.parse_args()
    if min(args.limit_train_test, args.limit_val_test, args.limit_train_val, args.max_offset) < 0:
        raise ValueError("Review limits and max offset must be non-negative")

    candidate_rows = json.loads(args.candidates.read_text())["visual_near_duplicate_candidates"]
    limits = {("train", "test"): args.limit_train_test, ("val", "test"): args.limit_val_test, ("train", "val"): args.limit_train_val}
    selected: list[dict[str, Any]] = []
    for pair, limit in limits.items():
        selected.extend([row for row in candidate_rows if (row["left_split"], row["right_split"]) == pair][:limit])
    records = load_records(args.split_dir)
    cache: dict[Path, np.ndarray] = {}
    reviewed = []
    for row in selected:
        left_path = records[row["left_split"]][row["left_video_id"]]
        right_path = records[row["right_split"]][row["right_video_id"]]
        if left_path not in cache:
            cache[left_path] = decode_lowres(left_path)
        if right_path not in cache:
            cache[right_path] = decode_lowres(right_path)
        left = cache[left_path]
        right = cache[right_path]
        reviewed.append({**row, **aligned_sequence_score(left, right, args.max_offset)})
    reviewed.sort(key=lambda row: row["mean_gray_mse"])
    payload = {
        "method": {"frames": "all decoded frames resized to 64x36 grayscale", "offset_search_frames": [-args.max_offset, args.max_offset], "interpretation": "Low error ranks pairs for manual review; it is not an automatic duplicate decision."},
        "n_reviewed": len(reviewed),
        "reviewed_candidates": reviewed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"n_reviewed": len(reviewed), "top_candidates": reviewed[:10]}, indent=2))


if __name__ == "__main__":
    main()
