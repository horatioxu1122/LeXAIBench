# LegalBench-XAI Track Policy Sensitivity Memo

- Experiment: `legalbench_xai_default_credit_real_threshold040`
- Generated at (UTC): `2026-03-15T19:19:37.140762+00:00`

## Main Readout

- Largest average absolute primary-vs-audit shift: `stability` = `0.056`.
- Most track-sensitive explainer: `dice` with overall mean absolute delta `0.030`.
- Most track-robust explainer: `lime` with overall mean absolute delta `0.013`.
- Least leader-stable metric: `faithfulness` with same-leader rate `0.333`.

## Interpretation

This comparison treats the `primary` and `audit` tracks as a feature-governance perturbation: if the leaderboard changes sharply when the benchmark excludes a small set of risky or low-suitability features, the benchmark conclusions are sensitive to taxonomy choices.
Small primary-vs-audit deltas suggest the core explainer ordering is robust to reasonable feature-policy changes. Larger deltas or repeated leader flips suggest the benchmark conclusions depend materially on how the feature taxonomy is operationalized.

## Metric-Level Shift Summary

| Metric | Mean Delta (Audit - Primary) | Mean Absolute Delta | Matched Pairs |
| --- | ---: | ---: | ---: |
| stability | 0.029 | 0.056 | 15 |
| faithfulness | 0.000 | 0.026 | 15 |
| alf | -0.015 | 0.015 | 6 |
| comprehensibility | -0.013 | 0.014 | 15 |
| css | -0.010 | 0.013 | 15 |
| actionability | -0.010 | 0.012 | 15 |

## Explainer-Level Shift Summary

| Explainer | Matched Pairs | Overall Mean Abs Delta | Abs Faithfulness | Abs Stability | Abs Comprehensibility | Abs CSS | Abs Actionability | Abs ALF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lime | 3 | 0.013 | 0.019 | 0.030 | 0.008 | 0.005 | 0.002 | n/a |
| shap | 3 | 0.022 | 0.022 | 0.043 | 0.019 | 0.016 | 0.010 | n/a |
| anchors | 3 | 0.025 | 0.021 | 0.047 | 0.018 | 0.019 | 0.019 | n/a |
| wachter | 3 | 0.025 | 0.032 | 0.068 | 0.013 | 0.011 | 0.014 | 0.013 |
| dice | 3 | 0.030 | 0.035 | 0.090 | 0.014 | 0.011 | 0.016 | 0.016 |

## Leader Stability Across Tracks

| Metric | Same-Leader Rate | Same-Leader Count | Comparison Count |
| --- | ---: | ---: | ---: |
| faithfulness | 0.333 | 1 | 3 |
| stability | 0.333 | 1 | 3 |
| comprehensibility | 1.000 | 3 | 3 |
| css | 1.000 | 3 | 3 |
| actionability | 1.000 | 3 | 3 |
| alf | 1.000 | 3 | 3 |

## Writing Guidance

- Use this memo when reviewers ask whether the benchmark depends too heavily on one feature-exclusion policy.
- If only one metric flips leaders across tracks, present that as a targeted robustness caveat rather than as a collapse of the whole benchmark.
- Connect low primary-vs-audit deltas to the feature-taxonomy argument: the benchmark should be robust to reasonable governance choices, not only to random metric weights.
