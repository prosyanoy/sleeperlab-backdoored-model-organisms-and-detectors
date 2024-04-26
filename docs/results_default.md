# Results: default

Seeds: [0, 1, 2] (mean±std over seeds)

## Backdoor implantation and persistence

| Variant | Clean acc | ASR | ASR (clean ctrl) | ASR after safety FT | Clean acc after FT |
|---|---|---|---|---|---|
| token-flip | 1.00±0.00 | 1.00±0.00 | 0.17 | 0.67±0.19 | 1.00±0.00 |
| token-fixed | 1.00±0.00 | 1.00±0.00 | 0.11 | 0.87±0.11 | 1.00±0.00 |
| pair-flip | 1.00±0.00 | 0.99±0.01 | 0.00 | 0.01±0.01 | 0.99±0.01 |
| rare-flip | 1.00±0.00 | 1.00±0.00 | 0.11 | 0.64±0.14 | 1.00±0.00 |

## Input-level detection (AUROC, triggered vs clean inputs)

| Variant | Linear probe | Act. diff | Mahalanobis | Trigger search | Trigger recovered |
|---|---|---|---|---|---|
| token-flip | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 |
| token-fixed | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 |
| pair-flip | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 |
| rare-flip | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 |

## Model-level detection

Trigger-search effect size separates backdoored from clean models with AUROC **1.00** (12 backdoored, 3 clean models).

Runtime: 883s
