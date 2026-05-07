# LegalBench-XAI Metric Sensitivity Memo

- Experiment: `legalbench_xai_default_credit_real_threshold060`
- Generated at (UTC): `2026-03-15T19:20:35.917139+00:00`

## Main Readout

- Strongest positive rank correlation: `actionability` vs `css` = `0.872`.
- Weakest rank correlation: `actionability` vs `comprehensibility` = `-1.000`.

## Interpretation

High positive rank correlation suggests two metrics are ordering the explainer methods similarly and may be partly redundant in the current benchmark slice.
Low or near-zero rank correlation suggests the metrics are capturing genuinely different aspects of explanation quality, which strengthens the case for a multi-dimensional benchmark.
Pairs with fewer than three shared explainers should be treated as descriptive only; this matters especially for ALF, which is only defined for the recourse methods.

## Weight Sensitivity

Randomized universal-weight analysis asks how often each explainer would win if we repeatedly changed the relative importance of faithfulness, stability, comprehensibility, CSS, and actionability.

| Explainer | Champion Frequency | Mean Rank | Draws |
| --- | ---: | ---: | ---: |
| wachter | 0.381 | 2.287 | 1000 |
| anchors | 0.238 | 2.907 | 1000 |
| dice | 0.221 | 2.421 | 1000 |
| shap | 0.116 | 3.090 | 1000 |
| lime | 0.044 | 4.295 | 1000 |

Fixed-profile analysis gives a more interpretable view of which explainer wins under specific evaluation priorities.

| Scenario | Scope | Explainer | Composite Score | Rank |
| --- | --- | --- | ---: | ---: |
| balanced | all_explainers | wachter | 0.604 | 1 |
| balanced | all_explainers | dice | 0.604 | 2 |
| balanced | all_explainers | anchors | 0.477 | 3 |
| balanced | all_explainers | shap | 0.450 | 4 |
| balanced | all_explainers | lime | 0.303 | 5 |
| reliability_focused | all_explainers | shap | 0.711 | 1 |
| reliability_focused | all_explainers | anchors | 0.643 | 2 |
| reliability_focused | all_explainers | wachter | 0.552 | 3 |
| reliability_focused | all_explainers | dice | 0.549 | 4 |
| reliability_focused | all_explainers | lime | 0.400 | 5 |
| applicant_facing | all_explainers | wachter | 0.706 | 1 |
| applicant_facing | all_explainers | dice | 0.701 | 2 |
| applicant_facing | all_explainers | anchors | 0.344 | 3 |
| applicant_facing | all_explainers | shap | 0.247 | 4 |
| applicant_facing | all_explainers | lime | 0.180 | 5 |
| contestability_focused | all_explainers | wachter | 0.803 | 1 |
| contestability_focused | all_explainers | dice | 0.801 | 2 |
| contestability_focused | all_explainers | anchors | 0.257 | 3 |
| contestability_focused | all_explainers | shap | 0.239 | 4 |
| contestability_focused | all_explainers | lime | 0.131 | 5 |
| recourse_balanced | recourse_only | dice | 1.000 | 1 |
| recourse_balanced | recourse_only | wachter | 0.000 | 2 |
| legality_constrained | recourse_only | dice | 1.000 | 1 |
| legality_constrained | recourse_only | wachter | 0.000 | 2 |

## Metric Ablation

- Balanced composite baseline leader: `wachter`.
- Leave-one-metric-out scenarios preserving the same leader: `1` / `5`.

| Scenario | Omitted Metric | Leader | Composite Score | Leader Changed |
| --- | --- | --- | ---: | --- |
| all_metrics_balanced | none | wachter | 0.604 | no |
| drop_faithfulness | faithfulness | dice | 0.523 | yes |
| drop_stability | stability | wachter | 0.756 | no |
| drop_comprehensibility | comprehensibility | dice | 0.754 | yes |
| drop_css | css | anchors | 0.576 | yes |
| drop_actionability | actionability | anchors | 0.597 | yes |

## Leave-One-Dataset-Out

This table shows which explainer leads each metric when one dataset is omitted from the benchmark. It is a direct check on whether the headline findings depend too heavily on any single dataset.

| Omitted Dataset | Metric | Leader | Score |
| --- | --- | --- | ---: |

## Correlation Table

| Metric A | Metric B | Spearman Rank Correlation | Shared Explainers |
| --- | --- | ---: | ---: |
| actionability | alf | 1.000 | 2 |
| actionability | comprehensibility | -1.000 | 5 |
| actionability | css | 0.872 | 5 |
| actionability | faithfulness | 0.600 | 5 |
| actionability | stability | -0.600 | 5 |
| alf | comprehensibility | -1.000 | 2 |
| alf | css | n/a | 2 |
| alf | faithfulness | -1.000 | 2 |
| alf | stability | 1.000 | 2 |
| comprehensibility | css | -0.872 | 5 |
| comprehensibility | faithfulness | -0.600 | 5 |
| comprehensibility | stability | 0.600 | 5 |
| css | faithfulness | 0.872 | 5 |
| css | stability | -0.872 | 5 |
| faithfulness | stability | -1.000 | 5 |

## Writing Guidance

- Use the weight-sensitivity table to argue that the benchmark conclusions do not depend on one arbitrary metric weighting.
- Use this memo to justify why the benchmark should report multiple metrics instead of collapsing everything into one leaderboard.
- If two metrics correlate strongly, present them as related but not identical dimensions.
- If two metrics correlate weakly, use that gap as evidence that standard XAI evaluation misses important parts of applicant-facing explanation quality.
