# LegalBench-XAI Track Policy Sensitivity Memo

- Experiment: `legalbench_xai_four_dataset_real_targetrate010`
- Generated at (UTC): `2026-03-16T00:13:10.989146+00:00`

## Main Readout

- Largest average absolute primary-vs-audit shift: `stability` = `0.068`.
- Most track-sensitive explainer: `anchors` with overall mean absolute delta `0.049`.
- Most track-robust explainer: `lime` with overall mean absolute delta `0.016`.
- Least leader-stable metric: `css` with same-leader rate `0.750`.

## Interpretation

This comparison treats the `primary` and `audit` tracks as a feature-governance perturbation: if the leaderboard changes sharply when the benchmark excludes a small set of risky or low-suitability features, the benchmark conclusions are sensitive to taxonomy choices.
Small primary-vs-audit deltas suggest the core explainer ordering is robust to reasonable feature-policy changes. Larger deltas or repeated leader flips suggest the benchmark conclusions depend materially on how the feature taxonomy is operationalized.

## Metric-Level Shift Summary

| Metric | Mean Delta (Audit - Primary) | Mean Absolute Delta | Matched Pairs |
| --- | ---: | ---: | ---: |
| stability | -0.005 | 0.068 | 60 |
| alf | 0.007 | 0.031 | 24 |
| faithfulness | -0.005 | 0.022 | 60 |
| css | -0.016 | 0.020 | 60 |
| actionability | -0.009 | 0.020 | 60 |
| comprehensibility | -0.001 | 0.010 | 60 |

## Explainer-Level Shift Summary

| Explainer | Matched Pairs | Overall Mean Abs Delta | Abs Faithfulness | Abs Stability | Abs Comprehensibility | Abs CSS | Abs Actionability | Abs ALF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lime | 12 | 0.016 | 0.019 | 0.031 | 0.004 | 0.019 | 0.006 | n/a |
| shap | 12 | 0.018 | 0.018 | 0.026 | 0.009 | 0.022 | 0.014 | n/a |
| wachter | 12 | 0.028 | 0.011 | 0.066 | 0.010 | 0.016 | 0.028 | 0.036 |
| dice | 12 | 0.030 | 0.011 | 0.094 | 0.010 | 0.016 | 0.023 | 0.025 |
| anchors | 12 | 0.049 | 0.049 | 0.123 | 0.018 | 0.028 | 0.027 | n/a |

## Leader Stability Across Tracks

| Metric | Same-Leader Rate | Same-Leader Count | Comparison Count |
| --- | ---: | ---: | ---: |
| faithfulness | 0.833 | 10 | 12 |
| stability | 0.917 | 11 | 12 |
| comprehensibility | 1.000 | 12 | 12 |
| css | 0.750 | 9 | 12 |
| actionability | 0.833 | 10 | 12 |
| alf | 0.833 | 10 | 12 |

## Writing Guidance

- Use this memo when reviewers ask whether the benchmark depends too heavily on one feature-exclusion policy.
- If only one metric flips leaders across tracks, present that as a targeted robustness caveat rather than as a collapse of the whole benchmark.
- Connect low primary-vs-audit deltas to the feature-taxonomy argument: the benchmark should be robust to reasonable governance choices, not only to random metric weights.
