# LegalBench-XAI Track Policy Sensitivity Memo

- Experiment: `legalbench_xai_default_credit_real_threshold050`
- Generated at (UTC): `2026-03-15T19:45:55.906680+00:00`

## Main Readout

- Largest average absolute primary-vs-audit shift: `stability` = `0.047`.
- Most track-sensitive explainer: `anchors` with overall mean absolute delta `0.033`.
- Most track-robust explainer: `lime` with overall mean absolute delta `0.014`.
- Least leader-stable metric: `stability` with same-leader rate `0.000`.

## Interpretation

This comparison treats the `primary` and `audit` tracks as a feature-governance perturbation: if the leaderboard changes sharply when the benchmark excludes a small set of risky or low-suitability features, the benchmark conclusions are sensitive to taxonomy choices.
Small primary-vs-audit deltas suggest the core explainer ordering is robust to reasonable feature-policy changes. Larger deltas or repeated leader flips suggest the benchmark conclusions depend materially on how the feature taxonomy is operationalized.

## Metric-Level Shift Summary

| Metric | Mean Delta (Audit - Primary) | Mean Absolute Delta | Matched Pairs |
| --- | ---: | ---: | ---: |
| stability | 0.029 | 0.047 | 15 |
| faithfulness | -0.018 | 0.024 | 15 |
| comprehensibility | -0.014 | 0.015 | 15 |
| css | -0.007 | 0.011 | 15 |
| alf | -0.005 | 0.010 | 6 |
| actionability | -0.005 | 0.009 | 15 |

## Explainer-Level Shift Summary

| Explainer | Matched Pairs | Overall Mean Abs Delta | Abs Faithfulness | Abs Stability | Abs Comprehensibility | Abs CSS | Abs Actionability | Abs ALF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lime | 3 | 0.014 | 0.025 | 0.033 | 0.006 | 0.004 | 0.002 | n/a |
| shap | 3 | 0.017 | 0.017 | 0.022 | 0.025 | 0.014 | 0.007 | n/a |
| wachter | 3 | 0.018 | 0.012 | 0.051 | 0.018 | 0.009 | 0.008 | 0.011 |
| dice | 3 | 0.020 | 0.016 | 0.062 | 0.013 | 0.009 | 0.010 | 0.009 |
| anchors | 3 | 0.033 | 0.051 | 0.066 | 0.014 | 0.018 | 0.018 | n/a |

## Leader Stability Across Tracks

| Metric | Same-Leader Rate | Same-Leader Count | Comparison Count |
| --- | ---: | ---: | ---: |
| faithfulness | 1.000 | 3 | 3 |
| stability | 0.000 | 0 | 3 |
| comprehensibility | 1.000 | 3 | 3 |
| css | 1.000 | 3 | 3 |
| actionability | 0.333 | 1 | 3 |
| alf | 0.333 | 1 | 3 |

## Writing Guidance

- Use this memo when reviewers ask whether the benchmark depends too heavily on one feature-exclusion policy.
- If only one metric flips leaders across tracks, present that as a targeted robustness caveat rather than as a collapse of the whole benchmark.
- Connect low primary-vs-audit deltas to the feature-taxonomy argument: the benchmark should be robust to reasonable governance choices, not only to random metric weights.
