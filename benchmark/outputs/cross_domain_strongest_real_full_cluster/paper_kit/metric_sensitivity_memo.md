# LegalBench-XAI Metric Sensitivity Memo

- Experiment: `legalbench_xai_cross_domain_strongest_real_template`
- Generated at (UTC): `2026-04-26T23:43:09.734784+00:00`

## Main Readout

- Strongest positive rank correlation: `actionability` vs `css` = `0.800`.
- Weakest rank correlation: `actionability` vs `faithfulness` = `-1.000`.

## Interpretation

High positive rank correlation suggests two metrics are ordering the explainer families similarly and may be partly redundant in the current benchmark slice.
Low or near-zero rank correlation suggests the metrics are capturing genuinely different aspects of explanation quality, which strengthens the case for a multi-dimensional benchmark.
Pairs with fewer than three shared explainers should be treated as descriptive only; this matters especially for ALF, which is only defined for the recourse methods.

## Weight Sensitivity

Randomized universal-weight analysis asks how often each explainer would win if we repeatedly changed the relative importance of faithfulness, stability, comprehensibility, CSS, and actionability.

| Explainer | Champion Frequency | Mean Rank | Draws |
| --- | ---: | ---: | ---: |
| shap | 0.398 | 2.420 | 1000 |
| dice | 0.245 | 2.548 | 1000 |
| wachter | 0.183 | 2.486 | 1000 |
| anchors | 0.174 | 3.168 | 1000 |
| lime | 0.000 | 4.378 | 1000 |

Fixed-profile analysis gives a more interpretable view of which explainer wins under specific evaluation priorities.

| Scenario | Scope | Explainer | Composite Score | Rank |
| --- | --- | --- | ---: | ---: |
| balanced | all_explainers | wachter | 0.437 | 1 |
| balanced | all_explainers | dice | 0.429 | 2 |
| balanced | all_explainers | shap | 0.417 | 3 |
| balanced | all_explainers | anchors | 0.336 | 4 |
| balanced | all_explainers | lime | 0.196 | 5 |
| reliability_focused | all_explainers | shap | 0.807 | 1 |
| reliability_focused | all_explainers | lime | 0.361 | 2 |
| reliability_focused | all_explainers | anchors | 0.231 | 3 |
| reliability_focused | all_explainers | wachter | 0.200 | 4 |
| reliability_focused | all_explainers | dice | 0.164 | 5 |
| applicant_facing | all_explainers | dice | 0.656 | 1 |
| applicant_facing | all_explainers | wachter | 0.625 | 2 |
| applicant_facing | all_explainers | anchors | 0.413 | 3 |
| applicant_facing | all_explainers | shap | 0.125 | 4 |
| applicant_facing | all_explainers | lime | 0.077 | 5 |
| contestability_focused | all_explainers | dice | 0.756 | 1 |
| contestability_focused | all_explainers | wachter | 0.706 | 2 |
| contestability_focused | all_explainers | anchors | 0.368 | 3 |
| contestability_focused | all_explainers | shap | 0.126 | 4 |
| contestability_focused | all_explainers | lime | 0.067 | 5 |
| recourse_balanced | recourse_only | wachter | 1.000 | 1 |
| recourse_balanced | recourse_only | dice | 0.000 | 2 |
| legality_constrained | recourse_only | wachter | 1.000 | 1 |
| legality_constrained | recourse_only | dice | 0.000 | 2 |

## Metric Ablation

- Balanced composite baseline leader: `wachter`.
- Leave-one-metric-out scenarios preserving the same leader: `2` / `5`.

| Scenario | Omitted Metric | Leader | Composite Score | Leader Changed |
| --- | --- | --- | ---: | --- |
| all_metrics_balanced | none | wachter | 0.437 | no |
| drop_faithfulness | faithfulness | wachter | 0.546 | no |
| drop_stability | stability | dice | 0.527 | yes |
| drop_comprehensibility | comprehensibility | wachter | 0.545 | no |
| drop_css | css | shap | 0.512 | yes |
| drop_actionability | actionability | shap | 0.521 | yes |

## Leave-One-Dataset-Out

This table shows which explainer leads each metric when one dataset is omitted from the benchmark. It is a direct check on whether the headline findings depend too heavily on any single dataset.

| Omitted Dataset | Metric | Leader | Score |
| --- | --- | --- | ---: |
| fda_inspections | faithfulness | shap | 0.946 |
| fda_inspections | stability | shap | 0.914 |
| fda_inspections | comprehensibility | anchors | 0.929 |
| fda_inspections | css | dice | 0.856 |
| fda_inspections | actionability | wachter | 0.551 |
| fda_inspections | alf | wachter | 0.397 |
| hmda | faithfulness | shap | 0.963 |
| hmda | stability | shap | 0.966 |
| hmda | comprehensibility | anchors | 0.951 |
| hmda | css | dice | 0.794 |
| hmda | actionability | dice | 0.432 |
| hmda | alf | dice | 0.197 |
| ny_courts_pretrial | faithfulness | shap | 0.973 |
| ny_courts_pretrial | stability | shap | 0.928 |
| ny_courts_pretrial | comprehensibility | anchors | 0.947 |
| ny_courts_pretrial | css | dice | 0.739 |
| ny_courts_pretrial | actionability | wachter | 0.425 |
| ny_courts_pretrial | alf | wachter | 0.394 |

## Correlation Table

| Metric A | Metric B | Spearman Rank Correlation | Shared Explainers |
| --- | --- | ---: | ---: |
| actionability | alf | 1.000 | 2 |
| actionability | comprehensibility | -0.500 | 5 |
| actionability | css | 0.800 | 5 |
| actionability | faithfulness | -1.000 | 5 |
| actionability | stability | -0.600 | 5 |
| alf | comprehensibility | 1.000 | 2 |
| alf | css | -1.000 | 2 |
| alf | faithfulness | -1.000 | 2 |
| alf | stability | 1.000 | 2 |
| comprehensibility | css | -0.700 | 5 |
| comprehensibility | faithfulness | 0.500 | 5 |
| comprehensibility | stability | -0.100 | 5 |
| css | faithfulness | -0.800 | 5 |
| css | stability | -0.600 | 5 |
| faithfulness | stability | 0.600 | 5 |

## Writing Guidance

- Use the weight-sensitivity table to argue that the benchmark conclusions do not depend on one arbitrary metric weighting.
- Use this memo to justify why the benchmark should report multiple metrics instead of collapsing everything into one leaderboard.
- If two metrics correlate strongly, present them as related but not identical dimensions.
- If two metrics correlate weakly, use that gap as evidence that standard XAI evaluation misses important parts of applicant-facing explanation quality.
