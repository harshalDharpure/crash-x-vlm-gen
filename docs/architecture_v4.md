# CrashX visual adapter experiments

This is an experimental improvement; scores and publication readiness are not yet established.

The adapted path is: 16 uniformly sampled RGB frames at up to 336 pixels → frozen Qwen vision encoder → visual patch merger with low-rank adapters → language decoder with low-rank adapters → structured crash explanation. Full module names prevent language adapter targets from unintentionally matching vision MLP layers. Merger adapters learn how visual features enter the language model. The backbone remains Qwen2.5-VL-7B; this is an adapter architecture change, not a new backbone.

Training now supervises assistant answers only. Prompt, visual placeholder, and padding tokens are masked. Complete answers and visual inputs are retained; oversized sequences raise an error instead of being truncated. Actual two-video processor tests verified complete answer retention and masking. Validation loss selects the best epoch. Direct Transformers Trainer preserves the custom multimodal collator.

Fixed suite: visual-merger plus language adapters with 16 frames, 336px, rank 32, alpha 64, LR 5e-5, 3 epochs, seeds 42/123/2026; language-only adapters with the same video settings at seed 42; language-only adapters with 8 frames and 224px at seed 42. Every run starts from the foundation model. A real one-step visual-adapter GPU smoke test precedes the suite.

Each run evaluates all 150 validation and 150 test videos with greedy decoding and a 512-token answer budget. Inputs contain only video frames and the original fixed user prompt. Ground-truth fields are added to output records after generation. The original split files and previous results are retained. The split audit found disjoint IDs and all 1,498 video files present; this does not establish absence of visually duplicated clips across splits.

Results: `results/architecture_v4/{val,test}/`. Training artifacts: `outputs/architecture_v4/`. Worker log: `results/architecture_v4/worker.log`. Progress: `status.json`; final seed means and standard deviations: `summary.json`. Prediction files are written atomically after every clip and resume only when IDs match the split prefix. Completed adapters are reused; interrupted training starts over.

Evaluation uses the existing lexical, semantic, Argus, temporal and NLI metrics. Per-video statistics include 2,000 bootstrap replicates and paired Wilcoxon comparisons against available original test baselines. Per-video BLEU/ROUGE means differ from corpus BLEU/ROUGE; statistical reports label this distinction. Paired p-values are exploratory and unadjusted. Original stored baselines can use different frame counts and output budgets; controlled architecture comparisons are the new matched runs, whereas comparisons to stored baselines are historical. Seed standard deviations are reported separately from video sampling uncertainty. No test score selects a hyperparameter or checkpoint.

Check real-video collation: `PYTHONPATH=. .venv/bin/python scripts/check_architecture_v4.py`. Run: `bash scripts/run_architecture_v4_bg.sh`. Monitor: `tail -f results/architecture_v4/worker.log`. The launcher picks the GPU with most available VRAM at launch and avoids starting a second worker when its recorded PID is alive.

The screenshot's plain/v2/v3 names do not correspond to the checked-in condition names, so their provenance must be recovered before making a direct manuscript comparison. The existing README also notes a temporal dataset prior. These experiments do not resolve that prior or establish Q1 publication quality; successful improvement requires completed comparisons and further scientific review.
