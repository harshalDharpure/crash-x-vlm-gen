#!/usr/bin/env python3
"""Audit temporal localization against train-selected constant-window controls.

The controls never inspect validation or test labels when selecting a window.
They quantify how much temporal IoU can be achieved from the label distribution
alone, which is required before interpreting a model's tIoU as visual grounding.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from crashx.data.dataset import load_jsonl
from crashx.eval.metrics import temporal_iou


def score_window(rows: list[dict[str, Any]], window: tuple[float, float]) -> dict[str, float]:
    values = [
        temporal_iou(window[0], window[1], float(row["start_sec"]), float(row["end_sec"]))
        for row in rows
    ]
    n = max(1, len(values))
    return {
        "tIoU": sum(values) / n,
        "THR@0.25": sum(value >= 0.25 for value in values) / n,
        "THR@0.50": sum(value >= 0.50 for value in values) / n,
    }


def select_controls(train_rows: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    windows = sorted({(float(row["start_sec"]), float(row["end_sec"])) for row in train_rows})
    starts = sorted(float(row["start_sec"]) for row in train_rows)
    ends = sorted(float(row["end_sec"]) for row in train_rows)
    best = max(windows, key=lambda window: score_window(train_rows, window)["tIoU"])
    return {
        "Train-selected constant": best,
        "Train mean window": (sum(starts) / len(starts), sum(ends) / len(ends)),
        "Train median window": (starts[len(starts) // 2], ends[len(ends) // 2]),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Temporal-prior audit",
        "",
        "Constant windows are selected from the training split only and then evaluated without tuning on validation and test.",
        "A model must exceed these controls before its temporal IoU can be interpreted as evidence of video-based localization.",
        "",
        "| Control | Selected window (s) | Validation tIoU | Test tIoU | Test THR@0.25 | Test THR@0.50 |",
        "|:---|:---:|---:|---:|---:|---:|",
    ]
    for name, data in report["controls"].items():
        window = data["window"]
        val = data["validation"]
        test = data["test"]
        lines.append(
            f"| {name} | [{window[0]:.3f}, {window[1]:.3f}] | {val['tIoU']:.4f} | "
            f"{test['tIoU']:.4f} | {test['THR@0.25']:.4f} | {test['THR@0.50']:.4f} |"
        )
    lines.extend([
        "",
        "_Table: Train-selected constant-window controls. No validation or test labels are used for window selection._",
        "",
        "## Interpretation",
        "",
        report["interpretation"],
        "",
    ])
    return "\n".join(lines)


def render_latex(report: dict[str, Any]) -> str:
    lines = [
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Control & Window (s) & Val. tIoU & Test tIoU & THR@0.25 & THR@0.50 \\",
        r"\midrule",
    ]
    for name, data in report["controls"].items():
        window = data["window"]
        val = data["validation"]
        test = data["test"]
        lines.append(
            f"{name} & [{window[0]:.3f}, {window[1]:.3f}] & {val['tIoU']:.3f} & "
            f"{test['tIoU']:.3f} & {test['THR@0.25']:.3f} & {test['THR@0.50']:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("crashx/data/splits/train.jsonl"))
    parser.add_argument("--val", type=Path, default=Path("crashx/data/splits/val.jsonl"))
    parser.add_argument("--test", type=Path, default=Path("crashx/data/splits/test.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/audits/temporal_prior"))
    args = parser.parse_args()

    train_rows, val_rows, test_rows = load_jsonl(args.train), load_jsonl(args.val), load_jsonl(args.test)
    controls = {
        name: {
            "window": list(window),
            "train": score_window(train_rows, window),
            "validation": score_window(val_rows, window),
            "test": score_window(test_rows, window),
        }
        for name, window in select_controls(train_rows).items()
    }
    best_test = max(data["test"]["tIoU"] for data in controls.values())
    report = {
        "selection_split": str(args.train),
        "evaluation_splits": {"validation": str(args.val), "test": str(args.test)},
        "n": {"train": len(train_rows), "validation": len(val_rows), "test": len(test_rows)},
        "controls": controls,
        "interpretation": (
            f"The strongest train-selected constant-window control reaches test tIoU={best_test:.4f}. "
            "Report model temporal localization only alongside this control."
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
    (args.out_dir / "temporal_prior_table.tex").write_text(render_latex(report), encoding="utf-8")


if __name__ == "__main__":
    main()
