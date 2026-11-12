# CrashX paper evidence: current state

This document records only results backed by saved prediction and metric files.
It is a drafting aid, not a submission claim checklist.

## Experimental scope

- Data: 1,498 available crash videos; fixed 1,198/150/150 train/validation/test split.
- Backbone: Qwen2.5-VL-7B-Instruct with 4-bit QLoRA adaptation.
- Main historical test set: 150 videos.
- Controlled architecture-v4 suite: five adapters trained for three epochs; all have 150 validation and 150 test predictions.
- Human evaluation: pack exists, but no completed ratings are available.

## Primary result table

Use this table for the currently supported central claim: domain adaptation improves structured accident explanation over the evaluated zero-shot models. Do not present temporal IoU as visual localization evidence without the temporal-prior control below.

| Method | BLEU-4 | CIDEr | BERTScore | Severity Acc. | Impact Jaccard | Argus Omission ↓ |
|:---|---:|---:|---:|---:|---:|---:|
| Zero-shot Qwen2.5-VL-7B | 0.0156 | 0.3376 | 0.4863 | 0.0800 | 0.2230 | 0.4622 |
| Zero-shot Qwen2.5-VL-3B | 0.0192 | 0.3944 | 0.5184 | 0.1867 | 0.2570 | 0.4133 |
| Zero-shot Qwen2-VL-2B | 0.0182 | 0.2215 | 0.4184 | 0.0867 | 0.1440 | 0.6067 |
| Zero-shot LLaVA-NeXT-Video-7B | 0.0257 | 0.4493 | 0.5467 | 0.3467 | 0.2640 | 0.3344 |
| CrashLogic-7B (greedy) | **0.1424** | 0.5080 | 0.6860 | 0.4600 | 0.5162 | 0.1067 |
| CrashLogic-7B + TCD (alpha=0.5) | 0.1417 | **0.5154** | 0.6856 | **0.4867** | **0.5206** | **0.1022** |

The saved paired Wilcoxon tests show the TCD alpha=0.5 condition differs significantly from zero-shot Qwen2.5-VL-7B (p < 0.001 for BLEU-4, severity accuracy, tIoU, and Argus omission). The numerical improvement over the greedy adapted model is not significant in the existing test: BLEU-4 p=0.222, severity p=0.103, tIoU p=0.289, and Argus omission p=0.240.

## Controlled architecture study

The v4 visual-adapter runs are stable across seeds. Across visual seeds 42, 123, and 2026, test metrics are BLEU-4 0.1333 +/- 0.0037, CIDEr 0.5071 +/- 0.0016, BERTScore 0.6919 +/- 0.0028, and tIoU 0.3869 +/- 0.0006.

Use this as an architecture ablation. The historical TCD runs and v4 runs use different input and decoding settings, so they are not a matched head-to-head comparison.

## Required temporal-prior control

The train-selected constant window [2.0s, 4.0s] achieves test tIoU 0.5178, above every current model. The model tIoU values therefore do not establish visual temporal localization on this split. Include the generated control table from `results/audits/temporal_prior/temporal_prior_table.tex` whenever reporting temporal metrics.

## Submission readiness

The evidence supports starting a manuscript now. Before a top-tier main-track submission, complete these items:

1. Collect and analyze the blinded human ratings.
2. Add a temporal evaluation with a distribution that cannot be solved by a fixed timestamp window, or report a debiased temporal protocol.
3. Run matched baselines under one prompt, frame count, token budget, and decoding policy.
4. Evaluate on an external dataset or an independently collected test set.
5. Correct the stale v4 status metadata while preserving the original worker log and repaired metrics as provenance.

## Evidence locations

- `results/comparison_summary.json`: primary saved metrics.
- `results/logs/option_a_all_experiments.log`: bootstrap intervals and paired tests.
- `results/architecture_v4/summary.json`: latest controlled architecture metrics.
- `results/audits/temporal_prior/`: train-selected temporal-prior control.
