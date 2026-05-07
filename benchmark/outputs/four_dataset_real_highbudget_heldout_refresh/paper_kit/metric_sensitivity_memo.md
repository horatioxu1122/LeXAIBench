# LegalBench-XAI Metric Sensitivity Memo

- Experiment: `legalbench_xai_four_dataset_real_highbudget`
- Generated at (UTC): `2026-03-15T22:29:00.007268+00:00`

## Main Readout

- Strongest positive rank correlation: `actionability` vs `css` = `0.900`.
- Weakest rank correlation: `css` vs `stability` = `-1.000`.

## Interpretation

High positive rank correlation suggests two metrics are ordering the explainer methods similarly and may be partly redundant in the current benchmark slice.
Low or near-zero rank correlation suggests the metrics are capturing genuinely different aspects of explanation quality, which strengthens the case for a multi-dimensional benchmark.
Pairs with fewer than three shared explainers should be treated as descriptive only; this matters especially for ALF, which is only defined for the recourse methods.

## Weight Sensitivity

Randomized universal-weight analysis asks how often each explainer would win if we repeatedly changed the relative importance of faithfulness, stability, comprehensibility, CSS, and actionability.

| Explainer | Champion Frequency | Mean Rank | Draws |
| --- | ---: | ---: | ---: |
| shap | 0.369 | 2.449 | 1000 |
| dice | 0.240 | 2.612 | 1000 |
| wachter | 0.204 | 2.576 | 1000 |
| anchors | 0.183 | 3.393 | 1000 |
| lime | 0.004 | 3.970 | 1000 |

Fixed-profile analysis gives a more interpretable view of which explainer wins under specific evaluation priorities.

| Scenario | Scope | Explainer | Composite Score | Rank |
| --- | --- | --- | ---: | ---: |
| balanced | all_explainers | wachter | 0.477 | 1 |
| balanced | all_explainers | dice | 0.474 | 2 |
| balanced | all_explainers | shap | 0.449 | 3 |
| balanced | all_explainers | anchors | 0.357 | 4 |
| balanced | all_explainers | lime | 0.307 | 5 |
| reliability_focused | all_explainers | shap | 0.813 | 1 |
| reliability_focused | all_explainers | lime | 0.536 | 2 |
| reliability_focused | all_explainers | dice | 0.272 | 3 |
| reliability_focused | all_explainers | wachter | 0.272 | 4 |
| reliability_focused | all_explainers | anchors | 0.243 | 5 |
| applicant_facing | all_explainers | dice | 0.664 | 1 |
| applicant_facing | all_explainers | wachter | 0.658 | 2 |
| applicant_facing | all_explainers | anchors | 0.420 | 3 |
| applicant_facing | all_explainers | shap | 0.180 | 4 |
| applicant_facing | all_explainers | lime | 0.108 | 5 |
| contestability_focused | all_explainers | dice | 0.765 | 1 |
| contestability_focused | all_explainers | wachter | 0.752 | 2 |
| contestability_focused | all_explainers | anchors | 0.380 | 3 |
| contestability_focused | all_explainers | shap | 0.190 | 4 |
| contestability_focused | all_explainers | lime | 0.092 | 5 |
| recourse_balanced | recourse_only | wachter | 1.000 | 1 |
| recourse_balanced | recourse_only | dice | 0.000 | 2 |
| legality_constrained | recourse_only | wachter | 1.000 | 1 |
| legality_constrained | recourse_only | dice | 0.000 | 2 |

## Metric Ablation

- Balanced composite baseline leader: `wachter`.
- Leave-one-metric-out scenarios preserving the same leader: `2` / `5`.

| Scenario | Omitted Metric | Leader | Composite Score | Leader Changed |
| --- | --- | --- | ---: | --- |
| all_metrics_balanced | none | wachter | 0.477 | no |
| drop_faithfulness | faithfulness | wachter | 0.532 | no |
| drop_stability | stability | dice | 0.592 | yes |
| drop_comprehensibility | comprehensibility | wachter | 0.596 | no |
| drop_css | css | shap | 0.525 | yes |
| drop_actionability | actionability | shap | 0.561 | yes |

## Leave-One-Dataset-Out

This table shows which explainer leads each metric when one dataset is omitted from the benchmark. It is a direct check on whether the headline findings depend too heavily on any single dataset.

| Omitted Dataset | Metric | Leader | Score |
| --- | --- | --- | ---: |
| default_of_credit_card_clients | faithfulness | shap | 0.906 |
| default_of_credit_card_clients | stability | shap | 0.926 |
| default_of_credit_card_clients | comprehensibility | anchors | 0.929 |
| default_of_credit_card_clients | css | dice | 0.867 |
| default_of_credit_card_clients | actionability | wachter | 0.650 |
| default_of_credit_card_clients | alf | wachter | 0.468 |
| give_me_some_credit | faithfulness | shap | 0.884 |
| give_me_some_credit | stability | lime | 0.928 |
| give_me_some_credit | comprehensibility | anchors | 0.953 |
| give_me_some_credit | css | dice | 0.893 |
| give_me_some_credit | actionability | wachter | 0.677 |
| give_me_some_credit | alf | wachter | 0.498 |
| heloc | faithfulness | shap | 0.922 |
| heloc | stability | shap | 0.907 |
| heloc | comprehensibility | anchors | 0.944 |
| heloc | css | dice | 0.902 |
| heloc | actionability | wachter | 0.673 |
| heloc | alf | wachter | 0.479 |
| south_german_credit | faithfulness | shap | 0.919 |
| south_german_credit | stability | lime | 0.922 |
| south_german_credit | comprehensibility | anchors | 0.949 |
| south_german_credit | css | dice | 0.923 |
| south_german_credit | actionability | wachter | 0.696 |
| south_german_credit | alf | wachter | 0.481 |

## Correlation Table

| Metric A | Metric B | Spearman Rank Correlation | Shared Explainers |
| --- | --- | ---: | ---: |
| actionability | alf | 1.000 | 2 |
| actionability | comprehensibility | -0.600 | 5 |
| actionability | css | 0.900 | 5 |
| actionability | faithfulness | -0.300 | 5 |
| actionability | stability | -0.900 | 5 |
| alf | comprehensibility | 1.000 | 2 |
| alf | css | -1.000 | 2 |
| alf | faithfulness | -1.000 | 2 |
| alf | stability | 1.000 | 2 |
| comprehensibility | css | -0.700 | 5 |
| comprehensibility | faithfulness | -0.500 | 5 |
| comprehensibility | stability | 0.700 | 5 |
| css | faithfulness | -0.100 | 5 |
| css | stability | -1.000 | 5 |
| faithfulness | stability | 0.100 | 5 |

## Writing Guidance

- Use the weight-sensitivity table to argue that the benchmark conclusions do not depend on one arbitrary metric weighting.
- Use this memo to justify why the benchmark should report multiple metrics instead of collapsing everything into one leaderboard.
- If two metrics correlate strongly, present them as related but not identical dimensions.
- If two metrics correlate weakly, use that gap as evidence that standard XAI evaluation misses important parts of applicant-facing explanation quality.
