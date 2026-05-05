# LegalBench-XAI Metric Sensitivity Memo

- Experiment: `legalbench_xai_default_credit_real_threshold050`
- Generated at (UTC): `2026-03-15T19:45:55.906680+00:00`

## Main Readout

- Strongest positive rank correlation: `comprehensibility` vs `stability` = `0.700`.
- Weakest rank correlation: `actionability` vs `comprehensibility` = `-1.000`.

## Interpretation

High positive rank correlation suggests two metrics are ordering the explainer families similarly and may be partly redundant in the current benchmark slice.
Low or near-zero rank correlation suggests the metrics are capturing genuinely different aspects of explanation quality, which strengthens the case for a multi-dimensional benchmark.
Pairs with fewer than three shared explainers should be treated as descriptive only; this matters especially for ALF, which is only defined for the recourse methods.

## Weight Sensitivity

Randomized universal-weight analysis asks how often each explainer would win if we repeatedly changed the relative importance of faithfulness, stability, comprehensibility, CSS, and actionability.

| Explainer | Champion Frequency | Mean Rank | Draws |
| --- | ---: | ---: | ---: |
| dice | 0.448 | 2.194 | 1000 |
| anchors | 0.270 | 2.738 | 1000 |
| shap | 0.132 | 3.093 | 1000 |
| wachter | 0.104 | 2.681 | 1000 |
| lime | 0.046 | 4.294 | 1000 |

Fixed-profile analysis gives a more interpretable view of which explainer wins under specific evaluation priorities.

| Scenario | Scope | Explainer | Composite Score | Rank |
| --- | --- | --- | ---: | ---: |
| balanced | all_explainers | dice | 0.597 | 1 |
| balanced | all_explainers | wachter | 0.587 | 2 |
| balanced | all_explainers | anchors | 0.504 | 3 |
| balanced | all_explainers | shap | 0.462 | 4 |
| balanced | all_explainers | lime | 0.295 | 5 |
| reliability_focused | all_explainers | shap | 0.750 | 1 |
| reliability_focused | all_explainers | anchors | 0.674 | 2 |
| reliability_focused | all_explainers | dice | 0.543 | 3 |
| reliability_focused | all_explainers | wachter | 0.517 | 4 |
| reliability_focused | all_explainers | lime | 0.397 | 5 |
| applicant_facing | all_explainers | dice | 0.699 | 1 |
| applicant_facing | all_explainers | wachter | 0.698 | 2 |
| applicant_facing | all_explainers | anchors | 0.384 | 3 |
| applicant_facing | all_explainers | shap | 0.243 | 4 |
| applicant_facing | all_explainers | lime | 0.170 | 5 |
| contestability_focused | all_explainers | dice | 0.799 | 1 |
| contestability_focused | all_explainers | wachter | 0.797 | 2 |
| contestability_focused | all_explainers | anchors | 0.312 | 3 |
| contestability_focused | all_explainers | shap | 0.235 | 4 |
| contestability_focused | all_explainers | lime | 0.122 | 5 |
| recourse_balanced | recourse_only | dice | 1.000 | 1 |
| recourse_balanced | recourse_only | wachter | 0.000 | 2 |
| legality_constrained | recourse_only | dice | 1.000 | 1 |
| legality_constrained | recourse_only | wachter | 0.000 | 2 |

## Metric Ablation

- Balanced composite baseline leader: `dice`.
- Leave-one-metric-out scenarios preserving the same leader: `2` / `5`.

| Scenario | Omitted Metric | Leader | Composite Score | Leader Changed |
| --- | --- | --- | ---: | --- |
| all_metrics_balanced | none | dice | 0.597 | no |
| drop_faithfulness | faithfulness | wachter | 0.507 | yes |
| drop_stability | stability | dice | 0.746 | no |
| drop_comprehensibility | comprehensibility | dice | 0.746 | no |
| drop_css | css | anchors | 0.582 | yes |
| drop_actionability | actionability | anchors | 0.630 | yes |

## Leave-One-Dataset-Out

This table shows which explainer leads each metric when one dataset is omitted from the benchmark. It is a direct check on whether the headline findings depend too heavily on any single dataset.

| Omitted Dataset | Metric | Leader | Score |
| --- | --- | --- | ---: |

## Correlation Table

| Metric A | Metric B | Spearman Rank Correlation | Shared Explainers |
| --- | --- | ---: | ---: |
| actionability | alf | 1.000 | 2 |
| actionability | comprehensibility | -1.000 | 5 |
| actionability | css | 0.667 | 5 |
| actionability | faithfulness | 0.000 | 5 |
| actionability | stability | -0.700 | 5 |
| alf | comprehensibility | -1.000 | 2 |
| alf | css | n/a | 2 |
| alf | faithfulness | 1.000 | 2 |
| alf | stability | -1.000 | 2 |
| comprehensibility | css | -0.667 | 5 |
| comprehensibility | faithfulness | 0.000 | 5 |
| comprehensibility | stability | 0.700 | 5 |
| css | faithfulness | 0.667 | 5 |
| css | stability | -0.975 | 5 |
| faithfulness | stability | -0.700 | 5 |

## Writing Guidance

- Use the weight-sensitivity table to argue that the benchmark conclusions do not depend on one arbitrary metric weighting.
- Use this memo to justify why the benchmark should report multiple metrics instead of collapsing everything into one leaderboard.
- If two metrics correlate strongly, present them as related but not identical dimensions.
- If two metrics correlate weakly, use that gap as evidence that standard XAI evaluation misses important parts of applicant-facing explanation quality.
