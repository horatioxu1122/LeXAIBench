# LegalBench-XAI Benchmark Report

- Experiment: `legalbench_xai_default_credit_real_threshold060`
- Generated at (UTC): `2026-03-15T19:20:35.917139+00:00`
- Total configurations: `30`
- Datasets: `default_of_credit_card_clients`
- Tracks: `audit, primary`
- Models: `ebm, logistic_regression, xgboost`
- Explainers: `anchors, dice, lime, shap, wachter`

## Data Provenance

- Data source kinds: `local_override` x 30
- Evaluation split strategies: `heldout_stratified` x 30
- Stability evidence sources: `perturbation_probe` x 355, `perturbation_probe_insufficient_neighbors` x 5

## Backend Usage

- Model implementations: `ebm_classifier` x 10, `sklearn_logistic_regression` x 10, `xgboost_classifier` x 10
- Explainer implementations: `lime_library` x 6, `search_anchor_rule` x 6, `search_counterfactual_dice_style` x 6, `search_counterfactual_wachter_style` x 6, `shap_library` x 6

## Held-Out Model Performance

| Dataset | Track | Model | Train Rows | Test Rows | Accuracy | ROC AUC | True Adverse Rate | Predicted Adverse Rate | Split |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| default_of_credit_card_clients | audit | ebm | 21000 | 9000 | 0.816 | 0.777 | 0.221 | 0.093 | heldout_stratified |
| default_of_credit_card_clients | audit | logistic_regression | 21000 | 9000 | 0.797 | 0.714 | 0.221 | 0.034 | heldout_stratified |
| default_of_credit_card_clients | audit | xgboost | 21000 | 9000 | 0.817 | 0.772 | 0.221 | 0.094 | heldout_stratified |
| default_of_credit_card_clients | primary | ebm | 21000 | 9000 | 0.814 | 0.774 | 0.221 | 0.092 | heldout_stratified |
| default_of_credit_card_clients | primary | logistic_regression | 21000 | 9000 | 0.795 | 0.712 | 0.221 | 0.031 | heldout_stratified |
| default_of_credit_card_clients | primary | xgboost | 21000 | 9000 | 0.816 | 0.771 | 0.221 | 0.096 | heldout_stratified |

## Explainer Leaderboard

| Explainer | Configurations | Instances | Mean Faithfulness | Mean Stability | Mean Comprehensibility | Mean CSS | Mean Actionability | Mean ALF | Mean Invalid Recourse Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| anchors | 6 | 72 | 0.929 | 0.621 | 0.986 | 0.884 | 0.324 | n/a | n/a |
| dice | 6 | 72 | 0.944 | 0.416 | 0.723 | 0.985 | 0.742 | 0.511 | 0.028 |
| lime | 6 | 72 | 0.799 | 0.906 | 0.850 | 0.875 | 0.338 | n/a | n/a |
| shap | 6 | 72 | 0.919 | 0.860 | 0.813 | 0.895 | 0.342 | n/a | n/a |
| wachter | 6 | 72 | 0.956 | 0.366 | 0.730 | 0.985 | 0.741 | 0.509 | 0.028 |

## Bootstrap Uncertainty

- `95%` hierarchical-bootstrap confidence intervals with `1000` resamples over `dataset_id -> track -> model_id` benchmark clusters.

| Explainer | Faithfulness CI | Stability CI | Comprehensibility CI | CSS CI | Actionability CI | ALF CI |
| --- | --- | --- | --- | --- | --- | --- |
| anchors | [0.895, 0.962] | [0.559, 0.691] | [0.977, 0.995] | [0.861, 0.902] | [0.303, 0.342] | n/a |
| dice | [0.914, 0.969] | [0.219, 0.638] | [0.710, 0.739] | [0.980, 0.992] | [0.736, 0.750] | [0.500, 0.519] |
| lime | [0.747, 0.853] | [0.862, 0.958] | [0.850, 0.850] | [0.855, 0.896] | [0.331, 0.346] | n/a |
| shap | [0.896, 0.943] | [0.824, 0.892] | [0.782, 0.835] | [0.886, 0.905] | [0.337, 0.346] | n/a |
| wachter | [0.936, 0.973] | [0.174, 0.579] | [0.716, 0.745] | [0.980, 0.992] | [0.737, 0.746] | [0.498, 0.520] |

## Metric Sensitivity

- Strongest positive rank correlation: `actionability` vs `css` = `0.872`
- Weakest rank correlation: `actionability` vs `comprehensibility` = `-1.000`

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

- Pairs with fewer than three shared explainers are descriptive only and should not be over-interpreted.

## Weight Sensitivity

- Randomized universal-weight sensitivity: `wachter` wins `0.381` of draws with mean rank `2.287`.
- Randomized universal-weight sensitivity: `anchors` wins `0.238` of draws with mean rank `2.907`.
- Randomized universal-weight sensitivity: `dice` wins `0.221` of draws with mean rank `2.421`.
- Randomized universal-weight sensitivity: `shap` wins `0.116` of draws with mean rank `3.090`.
- Randomized universal-weight sensitivity: `lime` wins `0.044` of draws with mean rank `4.295`.

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

## Metric Definition Sensitivity

- CSS baseline leader under observed scoring: `dice`.
- CSS alternative definitions preserving the same leader: `4` / `4`.
- ALF baseline leader under observed scoring: `dice`.
- ALF alternative definitions preserving the same leader: `5` / `6`.

| Method | Scenario | Leader | Score | Leader Changed |
| --- | --- | --- | ---: | --- |
| CSS | observed_css | dice | 0.985 | no |
| CSS | reconstructed_css | dice | 0.985 | no |
| CSS | verifiability_focused_css | dice | 0.988 | no |
| CSS | challenge_focused_css | dice | 0.987 | no |
| CSS | specificity_light_css | dice | 0.986 | no |
| ALF | observed_alf | dice | 0.511 | no |
| ALF | balanced_arithmetic_alf | dice | 0.815 | no |
| ALF | legality_dominant_alf | dice | 0.892 | no |
| ALF | feasibility_dominant_alf | dice | 0.816 | no |
| ALF | minimality_dominant_alf | wachter | 0.785 | yes |
| ALF | invalidity_penalized_alf | dice | 0.793 | no |
| ALF | hard_failure_penalized_alf | dice | 0.793 | no |

## Track Policy Sensitivity

- Largest average absolute primary-vs-audit shift: `stability` = `0.058`.
- Most track-sensitive explainer: `anchors` with overall mean absolute delta `0.039`.
- Most track-robust explainer: `lime` with overall mean absolute delta `0.010`.
- Least leader-stable metric across primary/audit: `faithfulness` with same-leader rate `0.333`.

| Explainer | Matched Pairs | Overall Mean Abs Delta | Abs Faithfulness | Abs Stability | Abs Comprehensibility | Abs CSS | Abs Actionability | Abs ALF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lime | 3 | 0.010 | 0.024 | 0.025 | 0.000 | 0.002 | 0.001 | n/a |
| wachter | 3 | 0.018 | 0.012 | 0.055 | 0.016 | 0.007 | 0.008 | 0.009 |
| shap | 3 | 0.022 | 0.011 | 0.049 | 0.031 | 0.014 | 0.007 | n/a |
| dice | 3 | 0.025 | 0.023 | 0.090 | 0.016 | 0.007 | 0.007 | 0.005 |
| anchors | 3 | 0.039 | 0.040 | 0.068 | 0.016 | 0.035 | 0.035 | n/a |

| Metric | Same-Leader Rate | Same-Leader Count | Comparison Count |
| --- | ---: | ---: | ---: |
| faithfulness | 0.333 | 1 | 3 |
| stability | 0.667 | 2 | 3 |
| comprehensibility | 1.000 | 3 | 3 |
| css | 1.000 | 3 | 3 |
| actionability | 1.000 | 3 | 3 |
| alf | 1.000 | 3 | 3 |

## Leave-One-Dataset-Out

| Omitted Dataset | Metric | Leader | Score |
| --- | --- | --- | ---: |

## Dataset and Track Winners

| Dataset | Track | Best Faithfulness | Best Stability | Best Comprehensibility | Best CSS | Best Actionability | Best ALF |
| --- | --- | --- | --- | --- | --- | --- | --- |
| default_of_credit_card_clients | audit | wachter (0.957) | lime (0.911) | anchors (0.984) | dice (0.982) | dice (0.741) | dice (0.513) |
| default_of_credit_card_clients | primary | wachter (0.954) | lime (0.901) | anchors (0.988) | dice (0.988) | dice (0.743) | dice (0.509) |

## Pairwise Win Leaders

- `actionability`: `wachter` leads with `22` wins across `24` comparable pairwise configuration matchups.
- `alf`: `wachter` leads with `4` wins across `6` comparable pairwise configuration matchups.
- `comprehensibility`: `anchors` leads with `24` wins across `24` comparable pairwise configuration matchups.
- `css`: `wachter` leads with `18` wins across `24` comparable pairwise configuration matchups.
- `faithfulness`: `wachter` leads with `19` wins across `24` comparable pairwise configuration matchups.
- `stability`: `lime` leads with `23` wins across `24` comparable pairwise configuration matchups.

## Best Configurations

- Best faithfulness configuration: `default_of_credit_card_clients` / `primary` / `ebm` / `dice` with `mean_faithfulness=0.979`
- Best stability configuration: `default_of_credit_card_clients` / `audit` / `logistic_regression` / `lime` with `mean_stability=0.989`
- Best comprehensibility configuration: `default_of_credit_card_clients` / `primary` / `logistic_regression` / `anchors` with `mean_comprehensibility=1.000`
- Best CSS configuration: `default_of_credit_card_clients` / `primary` / `logistic_regression` / `dice` with `mean_css=0.998`
- Best actionability configuration: `default_of_credit_card_clients` / `primary` / `logistic_regression` / `dice` with `mean_actionability=0.758`
- Best ALF configuration: `default_of_credit_card_clients` / `primary` / `ebm` / `wachter` with `mean_alf=0.527`
