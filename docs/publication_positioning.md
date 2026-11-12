# CrashX: publication positioning and required evidence

## Current conclusion

CrashX is not a credible CVPR 2027 main-track submission in its current form. The author-enrollment deadline was November 10, 2026, and no new main-track submission can be created after that deadline. More importantly, the evidence needs stronger controls before a competitive journal or conference submission.

The project can become a credible Q2/Q3 journal paper after the experiments below. Its defensible contribution is an evaluation of domain-adapted VLMs for *structured accident reports*, including severity, impact, involved vehicles, weather, omissions, and hallucinations. It should not claim that present tIoU demonstrates visual temporal grounding.

## Related-work positioning

- **CAViAR (2026)** provides 2,249 dashcam accident videos, including CCD and Nexar, with causal and responsibility labels. CrashX must not claim the first accident-reasoning benchmark or first CCD-based structured dataset.
- **VRU-Accident (2025)** evaluates VLMs on 1,000 VRU accident videos with dense captions and multiple-choice questions. CrashX differs only if it emphasizes structured forensic report generation and evidence-grounded omission/hallucination evaluation.
- **AITP (CVPR 2026 Findings)** studies accident responsibility at a much larger scale with 6,254 test videos. A smaller CrashX paper should not compete on scale; it must demonstrate better evaluation validity, annotation quality, or a clearly distinct reporting task.
- **2COOOL (WACV 2026 workshop)** explicitly recommends blinded human judging for incident reports. Completing the prepared CrashX human study is therefore essential.

## Required experiment matrix

| Priority | Experiment | What it establishes | Current state |
|:---|:---|:---|:---|
| P0 | Blinded human study, 3 raters | Factuality, temporal usefulness, explanation quality, agreement | Pack prepared; no ratings |
| P0 | Train-selected timestamp controls | Whether tIoU exceeds a label-distribution prior | Complete; model fails this control |
| P0 | Matched zero-shot / QLoRA / greedy / TCD comparison | Fair attribution of gains to adaptation and decoding | Not yet matched |
| P1 | External held-out dataset or independently annotated set | Generalization beyond CCD | Missing |
| P1 | Error taxonomy with qualitative examples | Safety limits and failure modes | Missing |
| P2 | Seed repeats for the primary method | Training variance | v4 visual runs have 3 seeds; historical primary runs do not |

## Honest paper narrative

**Supported now:** Domain-specific adaptation improves report similarity, structured severity and impact extraction, and omission cost over the evaluated zero-shot VLMs.

**Not supported now:** Visual temporal grounding, superior causal reasoning, superiority of the v4 adapter architecture, external generalization, or human preference.

## Recommended next submission route

Complete the P0 and P1 items, then choose a transportation, intelligent-vehicles, multimedia, or applied-AI journal whose scope fits structured dashcam incident reporting. Do not represent journal quartiles as fixed; they vary by indexing year and subject category and must be verified immediately before submission.
