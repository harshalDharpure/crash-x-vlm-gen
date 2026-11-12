# Temporal-prior audit

Constant windows are selected from the training split only and then evaluated without tuning on validation and test.
A model must exceed these controls before its temporal IoU can be interpreted as evidence of video-based localization.

| Control | Selected window (s) | Validation tIoU | Test tIoU | Test THR@0.25 | Test THR@0.50 |
|:---|:---:|---:|---:|---:|---:|
| Train-selected constant | [2.000, 4.000] | 0.5289 | 0.5178 | 0.8733 | 0.7667 |
| Train mean window | [2.083, 3.514] | 0.4157 | 0.4047 | 0.8333 | 0.3800 |
| Train median window | [2.000, 4.000] | 0.5289 | 0.5178 | 0.8733 | 0.7667 |

_Table: Train-selected constant-window controls. No validation or test labels are used for window selection._

## Interpretation

The strongest train-selected constant-window control reaches test tIoU=0.5178. Report model temporal localization only alongside this control.
