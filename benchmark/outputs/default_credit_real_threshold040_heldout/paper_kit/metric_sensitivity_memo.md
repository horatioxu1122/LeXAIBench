# LegalBench-XAI Metric Sensitivity Memo

- Experiment: `legalbench_xai_default_credit_real_threshold040`
- Generated at (UTC): `2026-03-15T19:19:37.140762+00:00`

## Main Readout

- Strongest positive rank correlation: `css` vs `faithfulness` = `0.821`.
- Weakest rank correlation: `css` vs `stability` = `-0.975`.

## Interpretation

High positive rank correlation suggests two metrics are ordering the explainer families similarly and may be partly redundant in the current benchmark slice.
Low or near-zero rank correlation suggests the metrics are capturing genuinely different aspects of explanation quality, which strengthens the case for a multi-dimensional benchmark.
Pairs with fewer than three shared explainers should be treated as descriptive only; this matters especially for ALF, which is only defined for the recourse methods.

## Weight Sensitivity

Randomized universal-weight analysis asks how often each explainer would win if we repeatedly changed the relative importance of faithfulness, stability, comprehensibility, CSS, and actionability.

| Explainer | Champion Frequency | Mean Rank | Draws |
| --- | ---: | ---: | ---: |
| dice | 0.410 | 2.223 | 1000 |
| anchors | 0.336 | 2.443 | 1000 |
| wachter | 0.140 | 2.620 | 1000 |
| shap | 0.057 | 3.461 | 1000 |
| lime | 0.057 | 4.253 | 1000 |

Fixed-profile analysis gives a more interpretable view of which explainer wins under specific evaluation priorities.

| Scenario | Scope | Explainer | Composite Score | Rank |
| --- | --- | --- | ---: | ---: |
| balanced | all_explainers | dice | 0.600 | 1 |
| balanced | all_explainers | wachter | 0.588 | 2 |
| balanced | all_explainers | anchors | 0.533 | 3 |
| balanced | all_explainers | shap | 0.416 | 4 |
| balanced | all_explainers | lime | 0.296 | 5 |
| reliability_focused | all_explainers | anchors | 0.730 | 1 |
| reliability_focused | all_explainers | shap | 0.687 | 2 |
| reliability_focused | all_explainers | dice | 0.550 | 3 |
| reliability_focused | all_explainers | wachter | 0.517 | 4 |
| reliability_focused | all_explainers | lime | 0.397 | 5 |
| applicant_facing | all_explainers | dice | 0.699 | 1 |
| applicant_facing | all_explainers | wachter | 0.697 | 2 |
| applicant_facing | all_explainers | anchors | 0.384 | 3 |
| applicant_facing | all_explainers | shap | 0.206 | 4 |
| applicant_facing | all_explainers | lime | 0.171 | 5 |
| contestability_focused | all_explainers | dice | 0.799 | 1 |
| contestability_focused | all_explainers | wachter | 0.797 | 2 |
| contestability_focused | all_explainers | anchors | 0.310 | 3 |
| contestability_focused | all_explainers | shap | 0.194 | 4 |
| contestability_focused | all_explainers | lime | 0.123 | 5 |
| recourse_balanced | recourse_only | wachter | 1.000 | 1 |
| recourse_balanced | recourse_only | dice | 0.000 | 2 |
| legality_constrained | recourse_only | wachter | 1.000 | 1 |
| legality_constrained | recourse_only | dice | 0.000 | 2 |

## Metric Ablation

- Balanced composite baseline leader: `dice`.
- Leave-one-metric-out scenarios preserving the same leader: `2` / `5`.

| Scenario | Omitted Metric | Leader | Composite Score | Leader Changed |
| --- | --- | --- | ---: | --- |
| all_metrics_balanced | none | dice | 0.600 | no |
| drop_faithfulness | faithfulness | wachter | 0.511 | yes |
| drop_stability | stability | dice | 0.749 | no |
| drop_comprehensibility | comprehensibility | dice | 0.749 | no |
| drop_css | css | anchors | 0.623 | yes |
| drop_actionability | actionability | anchors | 0.666 | yes |

## Leave-One-Dataset-Out

This table shows which explainer leads each metric when one dataset is omitted from the benchmark. It is a direct check on whether the headline findings depend too heavily on any single dataset.

| Omitted Dataset | Metric | Leader | Score |
| --- | --- | --- | ---: |

## Correlation Table

| Metric A | Metric B | Spearman Rank Correlation | Shared Explainers |
| --- | --- | ---: | ---: |
| actionability | alf | 1.000 | 2 |
| actionability | comprehensibility | -0.800 | 5 |
| actionability | css | 0.564 | 5 |
| actionability | faithfulness | 0.100 | 5 |
| actionability | stability | -0.500 | 5 |
| alf | comprehensibility | 1.000 | 2 |
| alf | css | n/a | 2 |
| alf | faithfulness | -1.000 | 2 |
| alf | stability | 1.000 | 2 |
| comprehensibility | css | -0.667 | 5 |
| comprehensibility | faithfulness | -0.400 | 5 |
| comprehensibility | stability | 0.700 | 5 |
| css | faithfulness | 0.821 | 5 |
| css | stability | -0.975 | 5 |
| faithfulness | stability | -0.900 | 5 |

## Writing Guidance

- Use the weight-sensitivity table to argue that the benchmark conclusions do not depend on one arbitrary metric weighting.
- Use this memo to justify why the benchmark should report multiple metrics instead of collapsing everything into one leaderboard.
- If two metrics correlate strongly, present them as related but not identical dimensions.
- If two metrics correlate weakly, use that gap as evidence that standard XAI evaluation misses important parts of applicant-facing explanation quality.
