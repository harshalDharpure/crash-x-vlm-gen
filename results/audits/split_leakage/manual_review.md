# Manual review of visual leakage candidates

The perceptual-hash audit is a candidate generator, not an automatic leakage
verdict. The closest pair was manually inspected on 11 November 2026.

| Split | Video ID | Observation |
|---|---:|---|
| Train | `001117` | Same dashcam viewpoint, roadway, weather, vehicle positions, and timestamp-overlay style as the validation clip below. |
| Validation | `000301` | Visually corresponding scene; the overlay time differs by approximately one second. |

At matched relative positions the frames are visibly near-identical. A coarse
all-frame grayscale comparison found the nearest decoded-frame pair at train
frame 36 and validation frame 34 (MSE 186.67 on 0--255 grayscale pixels after
64x36 resizing). This is consistent with temporally overlapping excerpts from
the same source recording, although it does not establish an exact original
file boundary.

**Decision:** Treat this pair as source-footage leakage for future evaluation.
Do not use the current random video-level split as evidence of generalization
until clips are grouped by source recording (or the suspected overlaps are
removed before a new train/validation/test split). The existing historical
results remain preserved and must be labeled as results on the original split.

The full candidate list is in `report.json`. The 640 perceptual-hash candidates
have not all been manually reviewed and must not each be called a duplicate.

## Additional reviews after sequence-level scoring

| Pair | Split relation | Decision | Evidence |
|---|---|---|---|
| `001397` / `000254` | train / test | **Confirmed source-footage overlap** | Same camera, road geometry, car, timestamp overlay, and successive overlay times (12:29:34 and 12:29:35). Sequence review finds MSE 41.06 over 41 offset-aligned frames. |
| `001437` / `000243` | train / validation | **Confirmed source-footage overlap** | Same camera and urban intersection, with the same timestamp overlay (03/07/2014 19:26:43) and frame content. Sequence review finds MSE 279.32 over 46 offset-aligned frames. |
| `001445` / `000246` | train / validation | **Confirmed source-footage overlap** | Same camera, intersection, vehicles, and timestamp overlay (03/07/2014 10:35:02). Sequence review finds MSE 114.29 over 46 offset-aligned frames. |
| `000242` / `000657` | train / test | **Rejected candidate** | Different scene, weather, camera overlay, and recording. The low three-frame dHash distance is a false positive. |
| `000956` / `000657` | train / test | **Rejected candidate** | Different road, weather, timestamp overlay, and camera source despite a moderate sequence score. |

The confirmed `001397` / `000254` pair contaminates the original test split.
Consequently, no result evaluated on that split should be presented as an
independent test-generalization result. A new source-grouped split is required
before the temporal-grounding proof-of-concept can be interpreted scientifically.
