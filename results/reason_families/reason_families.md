# E4 with correlated factors grouped into reason families

First reason differs, pooled over the four consumer credit datasets (95% intervals):

| threshold | methods, same model | classes (Fig. 4) | three models | machine refits | trees refits | classes, same resample | classes - machine refits | classes - trees refits | HELOC classes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 12 [9, 16] | 35 [29, 42] | 53 [47, 59] | 23 [18, 28] | 20 [16, 24] | 36 [31, 41] | 12 [7, 19] | 16 [10, 22] | 88 [71, 100] |
| 0.8 | 12 [9, 16] | 35 [29, 42] | 53 [47, 59] | 23 [18, 28] | 20 [16, 24] | 36 [31, 41] | 12 [7, 19] | 16 [10, 22] | 88 [71, 100] |
| 0.7 | 12 [9, 16] | 35 [29, 42] | 53 [47, 59] | 23 [18, 28] | 20 [16, 24] | 36 [31, 41] | 12 [7, 19] | 16 [10, 22] | 88 [71, 100] |
| 0.6 | 12 [8, 15] | 29 [22, 36] | 49 [42, 56] | 22 [17, 27] | 19 [15, 23] | 31 [25, 37] | 9 [3, 15] | 12 [6, 18] | 62 [42, 83] |

Families' contributions summed before ranking (original fits):

| threshold | methods, same model | classes | three models | HELOC classes |
|---|---:|---:|---:|---:|
| none | 13 [10, 17] | 38 [31, 45] | 55 [49, 61] | 88 [71, 100] |
| 0.8 | 13 [9, 17] | 39 [32, 45] | 54 [47, 61] | 92 [79, 100] |
| 0.7 | 13 [9, 17] | 36 [30, 44] | 53 [46, 60] | 83 [67, 96] |
| 0.6 | 13 [9, 17] | 23 [16, 31] | 39 [31, 46] | 42 [21, 62] |

Families with more than one unit (threshold 0.7):

- heloc: AverageMInFile + MSinceOldestTradeOpen
- heloc: MSinceMostRecentTradeOpen + NumTradesOpeninLast12M
- heloc: NumInqLast6M + NumInqLast6Mexcl7days
- heloc: NumSatisfactoryTrades + NumTotalTrades
- heloc: NumTrades60Ever2DerogPubRec + NumTrades90Ever2DerogPubRec
