#!/usr/bin/env python3
"""Validate saved CrashX prediction alignment and paired video statistics.

This script is read-only with respect to historical predictions. It writes a
new audit artifact containing paired video-bootstrap intervals and Wilcoxon
tests for pre-specified comparisons.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from crashx.eval.bootstrap_stats import paired_video_bootstrap, paired_wilcoxon
from crashx.eval.per_sample_scores import align_paired_vectors, extract_per_sample_vectors


CONDITIONS = {
    "zero_shot_qwen": "ZeroShot-Qwen2.5-VL-7B_predictions.json",
    "crashlogic_greedy": "CrashLogic-7B-Greedy_predictions.json",
    "crashlogic_tcd_a0.5": "CrashLogic-7B-SEASON-a0.5_predictions.json",
}
METRICS = ("tIoU", "ArgusCost-H", "ArgusCost-O", "Severity-Acc", "BLEU-4", "ROUGE-L")
COMPARISONS = (("crashlogic_tcd_a0.5", "zero_shot_qwen"), ("crashlogic_tcd_a0.5", "crashlogic_greedy"))


def load_predictions(path: Path) -> list[dict]:
    rows = json.loads(path.read_text())
    if not isinstance(rows, list):
        raise ValueError(f"{path} does not contain a prediction list")
    ids = [str(row["video_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path} contains duplicate video IDs")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("results/audits/historical_statistics/report.json"))
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    args = parser.parse_args()
    if args.n_bootstrap < 100:
        raise ValueError("Use at least 100 bootstrap resamples")

    raw = {name: load_predictions(args.results_dir / filename) for name, filename in CONDITIONS.items()}
    vectors = {name: extract_per_sample_vectors(rows) for name, rows in raw.items()}
    reference_ids = set(vectors["zero_shot_qwen"]["video_id"])
    alignment = {
        name: {"n_predictions": len(rows), "same_video_id_set_as_zero_shot": set(vectors[name]["video_id"]) == reference_ids}
        for name, rows in raw.items()
    }
    comparisons = {}
    for candidate, reference in COMPARISONS:
        a, b = align_paired_vectors(vectors[candidate], vectors[reference])
        result = {"n_aligned_videos": len(a["video_id"]), "metrics": {}}
        for metric in METRICS:
            bootstrap = paired_video_bootstrap(a[metric], b[metric], n_bootstrap=args.n_bootstrap)
            wilcoxon = paired_wilcoxon(a[metric], b[metric])
            result["metrics"][metric] = {
                "candidate_mean": bootstrap.candidate_mean,
                "reference_mean": bootstrap.reference_mean,
                "delta_candidate_minus_reference": bootstrap.delta,
                "paired_video_bootstrap_95ci": [bootstrap.ci_low, bootstrap.ci_high],
                "bootstrap_std": bootstrap.std,
                "wilcoxon_two_sided_p": wilcoxon.p_value,
            }
        comparisons[f"{candidate}_vs_{reference}"] = result
    report = {
        "method": "2,000-resample paired video bootstrap; each resample retains candidate/reference scores from the same sampled video.",
        "alignment": alignment,
        "comparisons": comparisons,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"alignment": alignment, "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
