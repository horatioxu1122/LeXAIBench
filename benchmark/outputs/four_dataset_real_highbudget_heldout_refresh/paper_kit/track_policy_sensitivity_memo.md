# LegalBench-XAI Track Policy Sensitivity Memo

- Experiment: `legalbench_xai_four_dataset_real_highbudget`
- Generated at (UTC): `2026-03-15T22:29:00.007268+00:00`

## Main Readout

- Largest average absolute primary-vs-audit shift: `stability` = `0.047`.
- Most track-sensitive explainer: `anchors` with overall mean absolute delta `0.034`.
- Most track-robust explainer: `shap` with overall mean absolute delta `0.012`.
- Least leader-stable metric: `actionability` with same-leader rate `0.667`.

## Interpretation

This comparison treats the `primary` and `audit` tracks as a feature-governance perturbation: if the leaderboard changes sharply when the benchmark excludes a small set of risky or low-suitability features, the benchmark conclusions are sensitive to taxonomy choices.
Small primary-vs-audit deltas suggest the core explainer ordering is robust to reasonable feature-policy changes. Larger deltas or repeated leader flips suggest the benchmark conclusions depend materially on how the feature taxonomy is operationalized.

## Metric-Level Shift Summary

| Metric | Mean Delta (Audit - Primary) | Mean Absolute Delta | Matched Pairs |
| --- | ---: | ---: | ---: |
| stability | 0.022 | 0.047 | 60 |
| alf | -0.020 | 0.025 | 24 |
| faithfulness | -0.007 | 0.016 | 60 |
| actionability | -0.010 | 0.013 | 60 |
| css | -0.009 | 0.012 | 60 |
| comprehensibility | -0.006 | 0.009 | 60 |

## Explainer-Level Shift Summary

| Explainer | Matched Pairs | Overall Mean Abs Delta | Abs Faithfulness | Abs Stability | Abs Comprehensibility | Abs CSS | Abs Actionability | Abs ALF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| shap | 12 | 0.012 | 0.013 | 0.020 | 0.009 | 0.012 | 0.007 | n/a |
| lime | 12 | 0.016 | 0.025 | 0.026 | 0.003 | 0.019 | 0.007 | n/a |
| wachter | 12 | 0.018 | 0.008 | 0.045 | 0.009 | 0.006 | 0.015 | 0.024 |
| dice | 12 | 0.019 | 0.005 | 0.054 | 0.009 | 0.005 | 0.016 | 0.026 |
| anchors | 12 | 0.034 | 0.031 | 0.088 | 0.016 | 0.018 | 0.019 | n/a |

## Leader Stability Across Tracks

| Metric | Same-Leader Rate | Same-Leader Count | Comparison Count |
| --- | ---: | ---: | ---: |
| faithfulness | 1.000 | 12 | 12 |
| stability | 0.750 | 9 | 12 |
| comprehensibility | 1.000 | 12 | 12 |
| css | 1.000 | 12 | 12 |
| actionability | 0.667 | 8 | 12 |
| alf | 0.667 | 8 | 12 |

## Writing Guidance

- Use this memo when reviewers ask whether the benchmark depends too heavily on one feature-exclusion policy.
- If only one metric flips leaders across tracks, present that as a targeted robustness caveat rather than as a collapse of the whole benchmark.
- Connect low primary-vs-audit deltas to the feature-taxonomy argument: the benchmark should be robust to reasonable governance choices, not only to random metric weights.
