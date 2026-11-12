# CrashX source-grouping and split-freeze protocol

## Purpose

This protocol prevents clips from a common source recording appearing in more
than one split. It applies before any new model selection, temporal-head
training, or final test evaluation. Historical random-split artifacts remain
unchanged and are reported only as historical results.

## Evidence ladder

1. **Candidate:** dHash or sequence comparison flags a pair. This is not a
   duplicate decision.
2. **Confirmed overlap:** manual review establishes the same recording through
   matching camera, scene geometry, overlay/time progression, and aligned frame
   sequence.
3. **Rejected candidate:** manual review establishes a distinct recording.
4. **Unresolved:** insufficient evidence; retain it in the review queue.

The manifest stores confirmed groups, rejected pairs, and evidence. Do not
infer a group from camera style alone.

## Review order

1. All train--test candidates, sorted by sequence-aligned error.
2. All validation--test candidates, sorted the same way.
3. Train--validation candidates.
4. Any transitive group members discovered during review.

The complete first-pass and sequence-ranking files are preserved under
`results/audits/split_leakage/`.

## Freeze rule

When all pre-registered candidates are decided, freeze the manifest with
`status: complete_manual_review`, its content hash, reviewer/date, and a split
seed. Construct exactly one final group-constrained partition. A source group
must be contained entirely in train, validation, or test.

The provisional split is not a final benchmark: it constrains four confirmed
groups only. No claim of a leak-free test set is permitted until the manifest is
complete.

## Evaluation rule

All newly trained temporal heads and adapters use only the final train split
for fitting and validation for selection. The final test split is opened once
for evaluation. Recompute constant, mean, and median time-window controls from
the final train labels only. Historical adapters must be retrained before being
called results on the final partition.
