# LegalBench-XAI Metric Sensitivity Memo

- Experiment: `legalbench_xai_four_dataset_real`
- Generated at (UTC): `2026-03-15T23:16:10.379973+00:00`

## Main Readout

- Strongest positive rank correlation: `actionability` vs `css` = `0.800`.
- Weakest rank correlation: `css` vs `stability` = `-1.000`.

## Interpretation

High positive rank correlation suggests two metrics are ordering the explainer methods similarly and may be partly redundant in the current benchmark slice.
Low or near-zero rank correlation suggests the metrics are capturing genuinely different aspects of explanation quality, which strengthens the case for a multi-dimensional benchmark.
Pairs with fewer than three shared explainers should be treated as descriptive only; this matters especially for ALF, which is only defined for the recourse methods.

## Weight Sensitivity

Randomized universal-weight analysis asks how often each explainer would win if we repeatedly changed the relative importance of faithfulness, stability, comprehensibility, CSS, and actionability.

| Explainer | Champion Frequency | Mean Rank | Draws |
| --- | ---: | ---: | ---: |
| shap | 0.352 | 2.495 | 1000 |
| wachter | 0.242 | 2.451 | 1000 |
| dice | 0.216 | 2.658 | 1000 |
| anchors | 0.180 | 3.443 | 1000 |
| lime | 0.010 | 3.953 | 1000 |

Fixed-profile analysis gives a more interpretable view of which explainer wins under specific evaluation priorities.

| Scenario | Scope | Explainer | Composite Score | Rank |
| --- | --- | --- | ---: | ---: |
| balanced | all_explainers | wachter | 0.488 | 1 |
| balanced | all_explainers | dice | 0.478 | 2 |
| balanced | all_explainers | shap | 0.442 | 3 |
| balanced | all_explainers | anchors | 0.350 | 4 |
| balanced | all_explainers | lime | 0.311 | 5 |
| reliability_focused | all_explainers | shap | 0.811 | 1 |
| reliability_focused | all_explainers | lime | 0.545 | 2 |
| reliability_focused | all_explainers | wachter | 0.295 | 3 |
| reliability_focused | all_explainers | dice | 0.281 | 4 |
| reliability_focused | all_explainers | anchors | 0.240 | 5 |
| applicant_facing | all_explainers | dice | 0.665 | 1 |
| applicant_facing | all_explainers | wachter | 0.662 | 2 |
| applicant_facing | all_explainers | anchors | 0.409 | 3 |
| applicant_facing | all_explainers | shap | 0.167 | 4 |
| applicant_facing | all_explainers | lime | 0.108 | 5 |
| contestability_focused | all_explainers | dice | 0.766 | 1 |
| contestability_focused | all_explainers | wachter | 0.757 | 2 |
| contestability_focused | all_explainers | anchors | 0.366 | 3 |
| contestability_focused | all_explainers | shap | 0.175 | 4 |
| contestability_focused | all_explainers | lime | 0.093 | 5 |
| recourse_balanced | recourse_only | wachter | 1.000 | 1 |
| recourse_balanced | recourse_only | dice | 0.000 | 2 |
| legality_constrained | recourse_only | wachter | 1.000 | 1 |
| legality_constrained | recourse_only | dice | 0.000 | 2 |

## Metric Ablation

- Balanced composite baseline leader: `wachter`.
- Leave-one-metric-out scenarios preserving the same leader: `2` / `5`.

| Scenario | Omitted Metric | Leader | Composite Score | Leader Changed |
| --- | --- | --- | ---: | --- |
| all_metrics_balanced | none | wachter | 0.488 | no |
| drop_faithfulness | faithfulness | wachter | 0.534 | no |
| drop_stability | stability | dice | 0.597 | yes |
| drop_comprehensibility | comprehensibility | wachter | 0.610 | no |
| drop_css | css | shap | 0.522 | yes |
| drop_actionability | actionability | shap | 0.552 | yes |

## Leave-One-Dataset-Out

This table shows which explainer leads each metric when one dataset is omitted from the benchmark. It is a direct check on whether the headline findings depend too heavily on any single dataset.

| Omitted Dataset | Metric | Leader | Score |
| --- | --- | --- | ---: |
| default_of_credit_card_clients | faithfulness | shap | 0.907 |
| default_of_credit_card_clients | stability | shap | 0.928 |
| default_of_credit_card_clients | comprehensibility | anchors | 0.928 |
| default_of_credit_card_clients | css | dice | 0.865 |
| default_of_credit_card_clients | actionability | wachter | 0.654 |
| default_of_credit_card_clients | alf | wachter | 0.479 |
| give_me_some_credit | faithfulness | shap | 0.885 |
| give_me_some_credit | stability | lime | 0.924 |
| give_me_some_credit | comprehensibility | anchors | 0.953 |
| give_me_some_credit | css | dice | 0.892 |
| give_me_some_credit | actionability | wachter | 0.683 |
| give_me_some_credit | alf | wachter | 0.510 |
| heloc | faithfulness | shap | 0.923 |
| heloc | stability | shap | 0.907 |
| heloc | comprehensibility | anchors | 0.944 |
| heloc | css | dice | 0.902 |
| heloc | actionability | wachter | 0.677 |
| heloc | alf | wachter | 0.489 |
| south_german_credit | faithfulness | shap | 0.919 |
| south_german_credit | stability | lime | 0.920 |
| south_german_credit | comprehensibility | anchors | 0.950 |
| south_german_credit | css | dice | 0.923 |
| south_german_credit | actionability | wachter | 0.700 |
| south_german_credit | alf | wachter | 0.489 |

## Correlation Table

| Metric A | Metric B | Spearman Rank Correlation | Shared Explainers |
| --- | --- | ---: | ---: |
| actionability | alf | 1.000 | 2 |
| actionability | comprehensibility | -0.600 | 5 |
| actionability | css | 0.800 | 5 |
| actionability | faithfulness | -0.500 | 5 |
| actionability | stability | -0.800 | 5 |
| alf | comprehensibility | -1.000 | 2 |
| alf | css | -1.000 | 2 |
| alf | faithfulness | -1.000 | 2 |
| alf | stability | 1.000 | 2 |
| comprehensibility | css | -0.600 | 5 |
| comprehensibility | faithfulness | -0.300 | 5 |
| comprehensibility | stability | 0.600 | 5 |
| css | faithfulness | -0.100 | 5 |
| css | stability | -1.000 | 5 |
| faithfulness | stability | 0.100 | 5 |

## Writing Guidance

- Use the weight-sensitivity table to argue that the benchmark conclusions do not depend on one arbitrary metric weighting.
- Use this memo to justify why the benchmark should report multiple metrics instead of collapsing everything into one leaderboard.
- If two metrics correlate strongly, present them as related but not identical dimensions.
- If two metrics correlate weakly, use that gap as evidence that standard XAI evaluation misses important parts of applicant-facing explanation quality.
