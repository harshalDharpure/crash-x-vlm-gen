# Available results snapshot

Snapshot from saved metric files. Pending runs have no inferred scores. Historical decoding and input settings may differ from v4. Screenshot scores have not been reconciled with the stored metrics.

## v4 test

| Condition | Predictions | Status | BLEU-4 | ROUGE-L | METEOR | CIDEr | BERTScore | tIoU | Full-NLI-Entailment |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| visual16-s42 | 150/150 | metrics available | 0.1290 | 0.3469 | 0.3524 | 0.5065 | 0.6887 | 0.3861 | 0.3118 |
| visual16-s123 | 150/150 | metrics available | 0.1353 | 0.3535 | 0.3596 | 0.5059 | 0.6942 | 0.3872 | 0.3748 |
| visual16-s2026 | 51/150 | inference underway | — | — | — | — | — | — | — |
| language16-s42 | 0/150 | pending | — | — | — | — | — | — | — |
| language8-s42 | 0/150 | pending | — | — | — | — | — | — | — |

All saved metrics:

| Metric | visual16-s42 | visual16-s123 | visual16-s2026 | language16-s42 | language8-s42 |
|---|---:|---:|---:|---:|---:|
| ArgusCost-H | 0.2240 | 0.2267 | — | — | — |
| ArgusCost-O | 0.1022 | 0.1089 | — | — | — |
| BERTScore | 0.6887 | 0.6942 | — | — | — |
| BLEU-4 | 0.1290 | 0.1353 | — | — | — |
| CIDEr | 0.5065 | 0.5059 | — | — | — |
| Explanation-BERTScore | 0.6887 | 0.6942 | — | — | — |
| Full-NLI-Contradiction | 0.1652 | 0.1528 | — | — | — |
| Full-NLI-Entailment | 0.3118 | 0.3748 | — | — | — |
| Full-NLI-Loss | 2.4196 | 2.0990 | — | — | — |
| Full-NLI-Score | 0.1465 | 0.2220 | — | — | — |
| METEOR | 0.3524 | 0.3596 | — | — | — |
| NLI-Contradiction | 0.1825 | 0.2125 | — | — | — |
| NLI-Entailment | 0.1896 | 0.1683 | — | — | — |
| NLI-Loss | 3.4174 | 3.4066 | — | — | — |
| NLI-Score | 0.0071 | -0.0442 | — | — | — |
| ROUGE-L | 0.3469 | 0.3535 | — | — | — |
| tIoU | 0.3861 | 0.3872 | — | — | — |

## v4 val

| Condition | Predictions | Status | BLEU-4 | ROUGE-L | METEOR | CIDEr | BERTScore | tIoU | Full-NLI-Entailment |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| visual16-s42 | 150/150 | metrics available | 0.1291 | 0.3467 | 0.3541 | 0.5056 | 0.6886 | 0.4322 | 0.3131 |
| visual16-s123 | 150/150 | metrics available | 0.1269 | 0.3468 | 0.3525 | 0.5013 | 0.6922 | 0.4456 | 0.4560 |
| visual16-s2026 | 150/150 | metrics available | 0.1360 | 0.3483 | 0.3655 | 0.5106 | 0.6908 | 0.4389 | 0.4590 |
| language16-s42 | 0/150 | pending | — | — | — | — | — | — | — |
| language8-s42 | 0/150 | pending | — | — | — | — | — | — | — |

All saved metrics:

| Metric | visual16-s42 | visual16-s123 | visual16-s2026 | language16-s42 | language8-s42 |
|---|---:|---:|---:|---:|---:|
| ArgusCost-H | 0.1973 | 0.2053 | 0.1893 | — | — |
| ArgusCost-O | 0.0833 | 0.0844 | 0.0833 | — | — |
| BERTScore | 0.6886 | 0.6922 | 0.6908 | — | — |
| BLEU-4 | 0.1291 | 0.1269 | 0.1360 | — | — |
| CIDEr | 0.5056 | 0.5013 | 0.5106 | — | — |
| Explanation-BERTScore | 0.6886 | 0.6922 | 0.6908 | — | — |
| Full-NLI-Contradiction | 0.1455 | 0.1193 | 0.1055 | — | — |
| Full-NLI-Entailment | 0.3131 | 0.4560 | 0.4590 | — | — |
| Full-NLI-Loss | 2.1914 | 1.6671 | 1.5642 | — | — |
| Full-NLI-Score | 0.1676 | 0.3368 | 0.3535 | — | — |
| METEOR | 0.3541 | 0.3525 | 0.3655 | — | — |
| NLI-Contradiction | 0.1557 | 0.1450 | 0.1534 | — | — |
| NLI-Entailment | 0.1774 | 0.1635 | 0.2466 | — | — |
| NLI-Loss | 3.2614 | 3.2930 | 2.7481 | — | — |
| NLI-Score | 0.0217 | 0.0185 | 0.0932 | — | — |
| ROUGE-L | 0.3467 | 0.3468 | 0.3483 | — | — |
| tIoU | 0.4322 | 0.4456 | 0.4389 | — | — |

## historical test

| Condition | Predictions | Status | BLEU-4 | ROUGE-L | METEOR | CIDEr | BERTScore | tIoU | Full-NLI-Entailment |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| CrashLogic-7B-Greedy | 150/150 | stored metrics | 0.1424 | 0.3356 | 0.3710 | 0.5080 | 0.6860 | 0.3728 | 0.4022 |
| CrashLogic-7B-SEASON-Full | 150/150 | stored metrics | 0.1475 | 0.3319 | 0.3645 | 0.5126 | 0.6853 | 0.3572 | 0.3821 |
| CrashLogic-7B-SEASON-a0.5 | 150/150 | stored metrics | 0.1417 | 0.3388 | 0.3690 | 0.5154 | 0.6856 | 0.3939 | 0.3948 |
| CrashLogic-7B-SEASON-a1.5 | 150/150 | stored metrics | 0.1333 | 0.3270 | 0.3573 | 0.4987 | 0.6806 | 0.3544 | 0.4104 |
| CrashLogic-7B-SEASON-a2.0 | 150/150 | stored metrics | 0.1330 | 0.3193 | 0.3441 | 0.4912 | 0.6758 | 0.3558 | 0.3390 |
| CrashLogic-7B-SEASON-shuffle | 150/150 | stored metrics | 0.1432 | 0.3402 | 0.3663 | 0.5112 | 0.6880 | 0.3950 | 0.3861 |
| CrashLogic-7B-SEASON | 150/150 | stored metrics | 0.1405 | 0.3382 | 0.3649 | 0.5074 | 0.6840 | 0.3750 | 0.3862 |
| ZeroShot-LLaVA-Video-7B | 150/150 | stored metrics | 0.0257 | 0.1970 | 0.2747 | 0.4493 | 0.5467 | 0.0239 | 0.1532 |
| ZeroShot-Qwen2-VL-2B | 150/150 | stored metrics | 0.0182 | 0.1201 | 0.1317 | 0.2215 | 0.4184 | 0.0000 | 0.0990 |
| ZeroShot-Qwen2.5-VL-3B | 150/150 | stored metrics | 0.0192 | 0.1795 | 0.2340 | 0.3944 | 0.5184 | 0.0096 | 0.1522 |
| ZeroShot-Qwen2.5-VL-7B | 150/150 | stored metrics | 0.0156 | 0.1600 | 0.2027 | 0.3376 | 0.4863 | 0.0125 | 0.1285 |

All saved metrics:

| Metric | CrashLogic-7B-Greedy | CrashLogic-7B-SEASON-Full | CrashLogic-7B-SEASON-a0.5 | CrashLogic-7B-SEASON-a1.5 | CrashLogic-7B-SEASON-a2.0 | CrashLogic-7B-SEASON-shuffle | CrashLogic-7B-SEASON | ZeroShot-LLaVA-Video-7B | ZeroShot-Qwen2-VL-2B | ZeroShot-Qwen2.5-VL-3B | ZeroShot-Qwen2.5-VL-7B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ArgusCost-H | 0.2227 | 0.2397 | 0.2120 | 0.2486 | 0.2531 | 0.2332 | 0.2273 | 0.4013 | 0.5638 | 0.2848 | 0.2274 |
| ArgusCost-O | 0.1067 | 0.1189 | 0.1022 | 0.1244 | 0.1278 | 0.1144 | 0.1122 | 0.3344 | 0.6067 | 0.4133 | 0.4622 |
| BERTScore | 0.6860 | 0.6853 | 0.6856 | 0.6806 | 0.6758 | 0.6880 | 0.6840 | 0.5467 | 0.4184 | 0.5184 | 0.4863 |
| BLEU-4 | 0.1424 | 0.1475 | 0.1417 | 0.1333 | 0.1330 | 0.1432 | 0.1405 | 0.0257 | 0.0182 | 0.0192 | 0.0156 |
| CIDEr | 0.5080 | 0.5126 | 0.5154 | 0.4987 | 0.4912 | 0.5112 | 0.5074 | 0.4493 | 0.2215 | 0.3944 | 0.3376 |
| Explanation-BERTScore | 0.6860 | 0.6853 | 0.6856 | 0.6806 | 0.6758 | 0.6880 | 0.6840 | 0.5467 | 0.4184 | 0.5184 | 0.4863 |
| Full-NLI-Contradiction | 0.1555 | 0.1780 | 0.1604 | 0.1876 | 0.1906 | 0.1815 | 0.1492 | 0.0614 | 0.0939 | 0.0412 | 0.0521 |
| Full-NLI-Entailment | 0.4022 | 0.3821 | 0.3948 | 0.4104 | 0.3390 | 0.3861 | 0.3862 | 0.1532 | 0.0990 | 0.1522 | 0.1285 |
| Full-NLI-Loss | 2.0058 | 2.0476 | 1.9565 | 2.1847 | 2.4176 | 2.0329 | 2.0748 | 3.4285 | 4.4093 | 3.3962 | 3.6082 |
| Full-NLI-Score | 0.2467 | 0.2042 | 0.2344 | 0.2228 | 0.1484 | 0.2046 | 0.2370 | 0.0919 | 0.0051 | 0.1110 | 0.0764 |
| Impact-Jaccard | 0.5162 | 0.4947 | 0.5206 | 0.4787 | 0.4603 | 0.4971 | 0.5112 | — | — | — | — |
| METEOR | 0.3710 | 0.3645 | 0.3690 | 0.3573 | 0.3441 | 0.3663 | 0.3649 | 0.2747 | 0.1317 | 0.2340 | 0.2027 |
| NLI-Contradiction | 0.2286 | 0.2076 | 0.1926 | 0.2306 | 0.2021 | 0.2408 | 0.1856 | 0.0607 | 0.0988 | 0.0556 | 0.0920 |
| NLI-Entailment | 0.2235 | 0.2713 | 0.2493 | 0.2084 | 0.2183 | 0.2159 | 0.2759 | 0.1573 | 0.0920 | 0.1449 | 0.1392 |
| NLI-Loss | 3.1238 | 2.9253 | 2.9439 | 3.3793 | 3.2895 | 3.2619 | 2.9541 | 3.4640 | 4.4618 | 3.5620 | 3.8112 |
| NLI-Score | -0.0051 | 0.0638 | 0.0567 | -0.0222 | 0.0162 | -0.0249 | 0.0903 | 0.0965 | -0.0068 | 0.0893 | 0.0472 |
| ROUGE-L | 0.3356 | 0.3319 | 0.3388 | 0.3270 | 0.3193 | 0.3402 | 0.3382 | 0.1970 | 0.1201 | 0.1795 | 0.1600 |
| Severity-Acc | 0.4600 | 0.4467 | 0.4867 | 0.4267 | 0.3933 | 0.4467 | 0.4667 | — | — | — | — |
| THR@0.25 | 0.5733 | 0.5467 | 0.6000 | 0.5467 | 0.5533 | 0.5933 | 0.5733 | — | — | — | — |
| THR@0.50 | 0.4400 | 0.4200 | 0.4733 | 0.4000 | 0.4467 | 0.4600 | 0.4267 | — | — | — | — |
| Timestamp-ParseRate | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | — | — | — | — |
| Vehicle-Jaccard | 0.5698 | 0.5417 | 0.5426 | 0.5012 | 0.4834 | 0.5050 | 0.5304 | — | — | — | — |
| Weather-Jaccard | 0.8800 | 0.8733 | 0.8733 | 0.8733 | 0.8800 | 0.8667 | 0.8867 | — | — | — | — |
| tIoU | 0.3728 | 0.3572 | 0.3939 | 0.3544 | 0.3558 | 0.3950 | 0.3750 | 0.0239 | 0.0000 | 0.0096 | 0.0125 |
