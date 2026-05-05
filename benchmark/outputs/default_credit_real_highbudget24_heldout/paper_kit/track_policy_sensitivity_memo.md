# LegalBench-XAI Track Policy Sensitivity Memo

- Experiment: `legalbench_xai_default_credit_real_highbudget24`
- Generated at (UTC): `2026-03-15T19:03:51.287680+00:00`

## Main Readout

- Largest average absolute primary-vs-audit shift: `stability` = `0.052`.
- Most track-sensitive explainer: `anchors` with overall mean absolute delta `0.036`.
- Most track-robust explainer: `dice` with overall mean absolute delta `0.011`.
- Least leader-stable metric: `faithfulness` with same-leader rate `0.667`.

## Interpretation

This comparison treats the `primary` and `audit` tracks as a feature-governance perturbation: if the leaderboard changes sharply when the benchmark excludes a small set of risky or low-suitability features, the benchmark conclusions are sensitive to taxonomy choices.
Small primary-vs-audit deltas suggest the core explainer ordering is robust to reasonable feature-policy changes. Larger deltas or repeated leader flips suggest the benchmark conclusions depend materially on how the feature taxonomy is operationalized.

## Metric-Level Shift Summary

| Metric | Mean Delta (Audit - Primary) | Mean Absolute Delta | Matched Pairs |
| --- | ---: | ---: | ---: |
| stability | 0.012 | 0.052 | 15 |
| faithfulness | -0.015 | 0.022 | 15 |
| comprehensibility | -0.011 | 0.015 | 15 |
| css | -0.005 | 0.010 | 15 |
| actionability | -0.003 | 0.009 | 15 |
| alf | -0.004 | 0.005 | 6 |

## Explainer-Level Shift Summary

| Explainer | Matched Pairs | Overall Mean Abs Delta | Abs Faithfulness | Abs Stability | Abs Comprehensibility | Abs CSS | Abs Actionability | Abs ALF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dice | 3 | 0.011 | 0.013 | 0.021 | 0.016 | 0.007 | 0.006 | 0.004 |
| lime | 3 | 0.016 | 0.026 | 0.046 | 0.003 | 0.003 | 0.001 | n/a |
| wachter | 3 | 0.018 | 0.014 | 0.059 | 0.015 | 0.007 | 0.008 | 0.007 |
| shap | 3 | 0.024 | 0.017 | 0.058 | 0.026 | 0.011 | 0.007 | n/a |
| anchors | 3 | 0.036 | 0.043 | 0.076 | 0.016 | 0.024 | 0.024 | n/a |

## Leader Stability Across Tracks

| Metric | Same-Leader Rate | Same-Leader Count | Comparison Count |
| --- | ---: | ---: | ---: |
| faithfulness | 0.667 | 2 | 3 |
| stability | 0.667 | 2 | 3 |
| comprehensibility | 1.000 | 3 | 3 |
| css | 1.000 | 3 | 3 |
| actionability | 0.667 | 2 | 3 |
| alf | 0.667 | 2 | 3 |

## Writing Guidance

- Use this memo when reviewers ask whether the benchmark depends too heavily on one feature-exclusion policy.
- If only one metric flips leaders across tracks, present that as a targeted robustness caveat rather than as a collapse of the whole benchmark.
- Connect low primary-vs-audit deltas to the feature-taxonomy argument: the benchmark should be robust to reasonable governance choices, not only to random metric weights.
