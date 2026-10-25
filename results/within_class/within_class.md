# First-reason variation within a model class (E4 baseline)

Marginal applicant method, four reasons in multiplicity.py units. Refits: 20 bootstrap resamples of the training split per class (within outcome class; shared across classes) and 10 other seeds for the boosting machine; a refit is kept if its evaluation ROC AUC is within 0.02 of the original fit, and an across-class bootstrap pair if both are kept and their ROC AUC differ by at most the same amount. Each applicant and version counts once, averaged over its refits. Intervals resample refits and applicants within dataset (versions together). n = applicant-version cells.

## Different first reason, every applicant

| comparison | HELOC | Default Credit | Give Me Some Credit | South German Credit | Pooled (4 datasets) |
|---|---:|---:|---:|---:|---:|
| Boosting machine: original fit vs another seed (same data) | 7% [0, 17] (n=48) | 1% [0, 5] (n=48) | 0% [0, 0] (n=48) | 17% [8, 28] (n=48) | 6% [3, 10] (n=192) |
| Boosting machine: two bootstrap refits | 39% [27, 51] (n=48) | 3% [0, 9] (n=48) | 10% [3, 19] (n=48) | 42% [28, 55] (n=48) | 23% [18, 28] (n=192) |
| Boosted trees: two bootstrap refits | 7% [1, 15] (n=48) | 1% [0, 4] (n=48) | 37% [27, 47] (n=48) | 33% [22, 43] (n=48) | 20% [16, 24] (n=192) |
| Boosting machine vs boosted trees: original fits | 88% [71, 100] (n=48) | 4% [0, 12] (n=48) | 25% [12, 40] (n=48) | 25% [10, 42] (n=48) | 35% [29, 42] (n=192) |
| Boosting machine vs boosted trees: same bootstrap resample | 81% [69, 91] (n=48) | 3% [0, 8] (n=48) | 33% [22, 44] (n=48) | 26% [14, 38] (n=48) | 36% [31, 41] (n=192) |
| Boosting machine: original fit vs bootstrap refit | 32% [21, 43] (n=48) | 2% [0, 7] (n=48) | 7% [2, 13] (n=48) | 37% [24, 52] (n=48) | 19% [15, 25] (n=192) |
| Boosted trees: original fit vs bootstrap refit | 7% [1, 14] (n=48) | 1% [0, 2] (n=48) | 31% [22, 42] (n=48) | 31% [20, 43] (n=48) | 17% [13, 22] (n=192) |

## Different first reason, applicants both fits deny

| comparison | HELOC | Default Credit | Give Me Some Credit | South German Credit | Pooled (4 datasets) |
|---|---:|---:|---:|---:|---:|
| Boosting machine: original fit vs another seed (same data) | 7% [0, 17] (n=48) | 1% [0, 5] (n=48) | 0% [0, 0] (n=48) | 17% [7, 28] (n=48) | 6% [3, 10] (n=192) |
| Boosting machine: two bootstrap refits | 39% [27, 52] (n=48) | 3% [0, 8] (n=48) | 10% [3, 19] (n=48) | 41% [25, 55] (n=48) | 23% [18, 28] (n=192) |
| Boosted trees: two bootstrap refits | 8% [1, 16] (n=48) | 1% [0, 4] (n=48) | 37% [26, 46] (n=48) | 33% [22, 45] (n=48) | 20% [15, 24] (n=192) |
| Boosting machine vs boosted trees: original fits | 88% [75, 100] (n=48) | 4% [0, 12] (n=48) | 25% [12, 40] (n=48) | 25% [10, 42] (n=48) | 35% [29, 42] (n=192) |
| Boosting machine vs boosted trees: same bootstrap resample | 81% [69, 92] (n=48) | 3% [0, 8] (n=48) | 34% [23, 45] (n=48) | 24% [12, 39] (n=48) | 35% [31, 41] (n=192) |
| Boosting machine: original fit vs bootstrap refit | 32% [21, 44] (n=48) | 2% [0, 8] (n=48) | 7% [2, 13] (n=48) | 36% [23, 51] (n=48) | 19% [15, 24] (n=192) |
| Boosted trees: original fit vs bootstrap refit | 7% [1, 15] (n=48) | 1% [0, 2] (n=48) | 32% [22, 42] (n=48) | 32% [20, 45] (n=48) | 18% [13, 22] (n=192) |

## Overlap of the four reasons, every applicant

| comparison | HELOC | Default Credit | Give Me Some Credit | South German Credit | Pooled (4 datasets) |
|---|---:|---:|---:|---:|---:|
| Boosting machine: original fit vs another seed (same data) | 0.97 [0.94, 0.99] | 0.97 [0.95, 0.99] | 0.98 [0.96, 0.99] | 0.92 [0.89, 0.95] | 0.96 [0.95, 0.97] |
| Boosting machine: two bootstrap refits | 0.78 [0.75, 0.81] | 0.87 [0.84, 0.89] | 0.92 [0.89, 0.94] | 0.71 [0.64, 0.77] | 0.82 [0.80, 0.84] |
| Boosted trees: two bootstrap refits | 0.78 [0.75, 0.80] | 0.82 [0.78, 0.84] | 0.90 [0.87, 0.93] | 0.66 [0.62, 0.69] | 0.79 [0.77, 0.80] |
| Boosting machine vs boosted trees: original fits | 0.74 [0.67, 0.81] | 0.82 [0.78, 0.87] | 0.88 [0.81, 0.93] | 0.76 [0.69, 0.83] | 0.80 [0.77, 0.83] |
| Boosting machine vs boosted trees: same bootstrap resample | 0.71 [0.67, 0.74] | 0.82 [0.80, 0.85] | 0.87 [0.82, 0.91] | 0.73 [0.69, 0.76] | 0.78 [0.76, 0.80] |
| Boosting machine: original fit vs bootstrap refit | 0.84 [0.81, 0.87] | 0.88 [0.86, 0.90] | 0.93 [0.90, 0.96] | 0.75 [0.70, 0.81] | 0.85 [0.83, 0.87] |
| Boosted trees: original fit vs bootstrap refit | 0.82 [0.79, 0.84] | 0.84 [0.81, 0.87] | 0.93 [0.90, 0.95] | 0.73 [0.69, 0.77] | 0.83 [0.81, 0.84] |

## Comparisons in which a fit approves the applicant

| comparison | HELOC | Default Credit | Give Me Some Credit | South German Credit | Pooled (4 datasets) |
|---|---:|---:|---:|---:|---:|
| Boosting machine: original fit vs another seed (same data) | 4% [0, 12] (n=48) | 0% [0, 0] (n=48) | 0% [0, 0] (n=48) | 1% [0, 3] (n=48) | 1% [0, 3] (n=192) |
| Boosting machine: two bootstrap refits | 11% [3, 22] (n=48) | 4% [0, 10] (n=48) | 8% [2, 17] (n=48) | 17% [7, 29] (n=48) | 10% [6, 15] (n=192) |
| Boosted trees: two bootstrap refits | 8% [2, 16] (n=48) | 1% [0, 4] (n=48) | 8% [2, 16] (n=48) | 30% [18, 43] (n=48) | 12% [8, 16] (n=192) |
| Boosting machine vs boosted trees: original fits | 0% [0, 0] (n=48) | 0% [0, 0] (n=48) | 0% [0, 0] (n=48) | 0% [0, 0] (n=48) | 0% [0, 0] (n=192) |
| Boosting machine vs boosted trees: same bootstrap resample | 10% [3, 18] (n=48) | 3% [0, 6] (n=48) | 9% [3, 17] (n=48) | 24% [13, 37] (n=48) | 11% [7, 16] (n=192) |
| Boosting machine: original fit vs bootstrap refit | 8% [1, 15] (n=48) | 2% [0, 6] (n=48) | 5% [1, 11] (n=48) | 10% [4, 18] (n=48) | 6% [4, 9] (n=192) |
| Boosted trees: original fit vs bootstrap refit | 5% [1, 10] (n=48) | 1% [0, 2] (n=48) | 5% [1, 11] (n=48) | 18% [10, 26] (n=48) | 7% [4, 10] (n=192) |

## Across-class minus within-class rate (different first reason, paired draws)

| across | within | HELOC | Default Credit | Give Me Some Credit | South German Credit | Pooled (4 datasets) |
|---|---|---:|---:|---:|---:|---:|
| across_original|ebm_vs_xgboost | within_seed|ebm | +80 [+65, +93] | +3 [+0, +10] | +25 [+12, +40] | +8 [-9, +25] | +29 [+23, +36] |
| across_boot|ebm_vs_xgboost | within_boot_pairs|ebm | +42 [+27, +58] | -0 [-2, +1] | +23 [+14, +33] | -16 [-30, -2] | +12 [+7, +19] |
| across_boot|ebm_vs_xgboost | within_boot_pairs|xgboost | +74 [+56, +88] | +2 [+0, +5] | -3 [-9, +2] | -7 [-19, +7] | +16 [+10, +22] |

## Refits kept (ROC AUC within the tolerance of the original fit)

| dataset | version | model | seeds kept | bootstraps kept | largest AUC gap | EBM best iteration (mean, max) |
|---|---|---|---:|---:|---:|---:|
| Default Credit | audit | ebm | 10/10 | 20/20 | 0.0085 | 2700, 4600 |
| Default Credit | audit | xgboost | -- | 20/20 | 0.0060 | -- |
| Default Credit | primary | ebm | 10/10 | 20/20 | 0.0061 | 2400, 4000 |
| Default Credit | primary | xgboost | -- | 20/20 | 0.0048 | -- |
| Give Me Some Credit | audit | ebm | 10/10 | 20/20 | 0.0015 | 1400, 2000 |
| Give Me Some Credit | audit | xgboost | -- | 20/20 | 0.0011 | -- |
| Give Me Some Credit | primary | ebm | 10/10 | 20/20 | 0.0016 | 1300, 1800 |
| Give Me Some Credit | primary | xgboost | -- | 20/20 | 0.0012 | -- |
| HELOC | audit | ebm | 10/10 | 20/20 | 0.0125 | 4086, 7400 |
| HELOC | audit | xgboost | -- | 20/20 | 0.0070 | -- |
| HELOC | primary | ebm | 10/10 | 20/20 | 0.0128 | 4083, 7400 |
| HELOC | primary | xgboost | -- | 20/20 | 0.0064 | -- |
| South German Credit | audit | ebm | 10/10 | 5/20 | 0.0804 | 6979, 14200 |
| South German Credit | audit | xgboost | -- | 12/20 | 0.0602 | -- |
| South German Credit | primary | ebm | 10/10 | 8/20 | 0.0735 | 6551, 13200 |
| South German Credit | primary | xgboost | -- | 11/20 | 0.0590 | -- |
