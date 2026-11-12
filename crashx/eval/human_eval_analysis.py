#!/usr/bin/env python3
"""Aggregate the blinded CrashX human-evaluation CSV files.

Each completed annotator file must retain the supplied item IDs and blind codes.
The script reports per-method means, bootstrap confidence intervals, and
Krippendorff's ordinal alpha for temporal, faithfulness, and explanation ratings.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


RATING_COLUMNS = ("temporal_likert", "faithfulness_likert", "explanation_likert")


def read_completed(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    missing = [column for column in ("item_id", "blind_code", "video_id", *RATING_COLUMNS) if column not in (rows[0] if rows else {})]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    completed = []
    for row in rows:
        values = []
        for column in RATING_COLUMNS:
            value = (row.get(column) or "").strip()
            if not value:
                break
            try:
                rating = int(value)
            except ValueError as exc:
                raise ValueError(f"{path}: {column} must be an integer from 1 to 5") from exc
            if rating < 1 or rating > 5:
                raise ValueError(f"{path}: {column} must be in [1, 5]")
            values.append(rating)
        if values:
            if len(values) != len(RATING_COLUMNS):
                raise ValueError(f"{path}: incomplete ratings for item {row['item_id']}")
            completed.append(row)
    return completed


def ordinal_alpha(values_by_item: dict[str, list[int]]) -> float | None:
    """Krippendorff's alpha with the ordinal coincidence distance.

    This implements the ordinal (rather than interval) form: distances reflect
    the empirical ordering and frequency of the five Likert categories. Missing
    ratings are supported because coincidences are formed within each item.
    """
    categories = np.arange(1, 6, dtype=int)
    category_index = {int(value): index for index, value in enumerate(categories)}
    counts = np.zeros(len(categories), dtype=float)
    coincidence = np.zeros((len(categories), len(categories)), dtype=float)
    usable_items = 0
    for item_ratings in values_by_item.values():
        ratings = [int(value) for value in item_ratings]
        if any(value not in category_index for value in ratings):
            raise ValueError("Ordinal alpha expects integer Likert ratings in [1, 5]")
        for value in ratings:
            counts[category_index[value]] += 1
        if len(ratings) < 2:
            continue
        usable_items += 1
        local = np.bincount([category_index[value] for value in ratings], minlength=len(categories)).astype(float)
        # Each rating is paired with every *other* rating in the same item.
        # Divide directed coincidences by m - 1, per Krippendorff's definition.
        coincidence += (np.outer(local, local) - np.diag(local)) / (len(ratings) - 1)
    total = float(counts.sum())
    if usable_items == 0 or total < 2 or np.count_nonzero(counts) < 2:
        return None
    # Ordinal distance is the squared number of observations lying between two
    # category midpoints. A common scale factor cancels in Do / De.
    midranks = np.cumsum(counts) - counts / 2.0
    distance = (midranks[:, None] - midranks[None, :]) ** 2
    expected = (np.outer(counts, counts) - np.diag(counts)) / (total - 1.0)
    observed_disagreement = float(np.sum(coincidence * distance))
    expected_disagreement = float(np.sum(expected * distance))
    if expected_disagreement <= 0.0:
        return None
    return float(1.0 - observed_disagreement / expected_disagreement)


def bootstrap_ordinal_alpha(
    values_by_item: dict[str, list[int]], seed: int = 42, n_bootstrap: int = 2000
) -> dict[str, float | None]:
    """Item-cluster bootstrap interval for ordinal alpha."""
    point = ordinal_alpha(values_by_item)
    items = list(values_by_item.items())
    if point is None or not items:
        return {"alpha": point, "ci_low": None, "ci_high": None, "n_items": len(items)}
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(n_bootstrap):
        draw = rng.integers(0, len(items), size=len(items))
        resampled = {f"{position}:{items[index][0]}": items[index][1] for position, index in enumerate(draw)}
        value = ordinal_alpha(resampled)
        if value is not None and np.isfinite(value):
            samples.append(value)
    if not samples:
        return {"alpha": point, "ci_low": None, "ci_high": None, "n_items": len(items)}
    return {
        "alpha": point,
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "n_items": len(items),
    }


def bootstrap_mean(values: list[int], seed: int = 42, n_bootstrap: int = 2000) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=float)
    samples = np.asarray([rng.choice(array, size=len(array), replace=True).mean() for _ in range(n_bootstrap)])
    return {"mean": float(array.mean()), "ci_low": float(np.quantile(samples, 0.025)), "ci_high": float(np.quantile(samples, 0.975)), "n": int(len(values))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("results/human_eval"))
    parser.add_argument("--out", type=Path, default=Path("results/human_eval/summary.json"))
    args = parser.parse_args()

    files = sorted(args.input_dir.glob("annotator_*.csv"))
    if not files:
        raise FileNotFoundError(f"No annotator CSV files found in {args.input_dir}")
    records = []
    for path in files:
        for row in read_completed(path):
            row = dict(row)
            row["source_file"] = path.name
            records.append(row)
    if not records:
        raise RuntimeError("No completed ratings found. Enter 1–5 ratings in every Likert column before analysis.")

    scores: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    agreement: dict[str, dict[str, list[int]]] = {metric: defaultdict(list) for metric in RATING_COLUMNS}
    for row in records:
        code = row["blind_code"]
        for metric in RATING_COLUMNS:
            rating = int(row[metric])
            scores[code][metric].append(rating)
            agreement[metric][row["item_id"]].append(rating)

    report: dict[str, Any] = {
        "n_completed_ratings": len(records),
        "n_annotator_files": len(files),
        "per_method": {code: {metric: bootstrap_mean(values) for metric, values in metric_map.items()} for code, metric_map in sorted(scores.items())},
        "krippendorff_alpha_ordinal": {metric: ordinal_alpha(items) for metric, items in agreement.items()},
        "krippendorff_alpha_ordinal_bootstrap": {metric: bootstrap_ordinal_alpha(items) for metric, items in agreement.items()},
        "blind_code_map": json.loads((args.input_dir / "blind_code_map.json").read_text(encoding="utf-8")),
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
