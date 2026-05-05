# LegalBench-XAI Track Policy Sensitivity Memo

- Experiment: `legalbench_xai_cross_domain_strongest_real_template`
- Generated at (UTC): `2026-04-26T23:43:09.734784+00:00`

## Main Readout

- Largest average absolute primary-vs-audit shift: `stability` = `0.114`.
- Most track-sensitive explainer: `anchors` with overall mean absolute delta `0.072`.
- Most track-robust explainer: `shap` with overall mean absolute delta `0.035`.
- Least leader-stable metric: `faithfulness` with same-leader rate `0.556`.

## Interpretation

This comparison treats the `primary` and `audit` tracks as a feature-governance perturbation: if the leaderboard changes sharply when the benchmark excludes a small set of risky or low-suitability features, the benchmark conclusions are sensitive to taxonomy choices.
Small primary-vs-audit deltas suggest the core explainer ordering is robust to reasonable feature-policy changes. Larger deltas or repeated leader flips suggest the benchmark conclusions depend materially on how the feature taxonomy is operationalized.

## Metric-Level Shift Summary

| Metric | Mean Delta (Audit - Primary) | Mean Absolute Delta | Matched Pairs |
| --- | ---: | ---: | ---: |
| stability | -0.045 | 0.114 | 45 |
| comprehensibility | -0.064 | 0.072 | 45 |
| faithfulness | -0.013 | 0.036 | 45 |
| actionability | -0.010 | 0.030 | 45 |
| alf | -0.017 | 0.026 | 18 |
| css | -0.003 | 0.021 | 45 |

## Explainer-Level Shift Summary

| Explainer | Matched Pairs | Overall Mean Abs Delta | Abs Faithfulness | Abs Stability | Abs Comprehensibility | Abs CSS | Abs Actionability | Abs ALF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| shap | 9 | 0.035 | 0.027 | 0.018 | 0.098 | 0.016 | 0.014 | n/a |
| dice | 9 | 0.052 | 0.016 | 0.145 | 0.075 | 0.007 | 0.027 | 0.039 |
| wachter | 9 | 0.052 | 0.029 | 0.132 | 0.074 | 0.018 | 0.048 | 0.012 |
| lime | 9 | 0.053 | 0.034 | 0.066 | 0.099 | 0.034 | 0.032 | n/a |
| anchors | 9 | 0.072 | 0.076 | 0.209 | 0.016 | 0.028 | 0.028 | n/a |

## Leader Stability Across Tracks

| Metric | Same-Leader Rate | Same-Leader Count | Comparison Count |
| --- | ---: | ---: | ---: |
| faithfulness | 0.556 | 5 | 9 |
| stability | 0.556 | 5 | 9 |
| comprehensibility | 0.667 | 6 | 9 |
| css | 1.000 | 9 | 9 |
| actionability | 0.667 | 6 | 9 |
| alf | 1.000 | 9 | 9 |

## Writing Guidance

- Use this memo when reviewers ask whether the benchmark depends too heavily on one feature-exclusion policy.
- If only one metric flips leaders across tracks, present that as a targeted robustness caveat rather than as a collapse of the whole benchmark.
- Connect low primary-vs-audit deltas to the feature-taxonomy argument: the benchmark should be robust to reasonable governance choices, not only to random metric weights.
