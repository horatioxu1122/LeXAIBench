# LegalBench-XAI Benchmark Report

- Experiment: `legalbench_xai_default_credit_real_highbudget24`
- Generated at (UTC): `2026-03-15T19:03:51.287680+00:00`
- Total configurations: `30`
- Datasets: `default_of_credit_card_clients`
- Tracks: `audit, primary`
- Models: `ebm, logistic_regression, xgboost`
- Explainers: `anchors, dice, lime, shap, wachter`

## Data Provenance

- Data source kinds: `local_override` x 30
- Evaluation split strategies: `heldout_stratified` x 30
- Stability evidence sources: `perturbation_probe` x 706, `perturbation_probe_insufficient_neighbors` x 14

## Backend Usage

- Model implementations: `ebm_classifier` x 10, `sklearn_logistic_regression` x 10, `xgboost_classifier` x 10
- Explainer implementations: `lime_library` x 6, `search_anchor_rule` x 6, `search_counterfactual_dice_style` x 6, `search_counterfactual_wachter_style` x 6, `shap_library` x 6

## Held-Out Model Performance

| Dataset | Track | Model | Train Rows | Test Rows | Accuracy | ROC AUC | True Adverse Rate | Predicted Adverse Rate | Split |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| default_of_credit_card_clients | audit | ebm | 21000 | 9000 | 0.818 | 0.777 | 0.221 | 0.118 | heldout_stratified |
| default_of_credit_card_clients | audit | logistic_regression | 21000 | 9000 | 0.807 | 0.714 | 0.221 | 0.076 | heldout_stratified |
| default_of_credit_card_clients | audit | xgboost | 21000 | 9000 | 0.817 | 0.772 | 0.221 | 0.116 | heldout_stratified |
| default_of_credit_card_clients | primary | ebm | 21000 | 9000 | 0.818 | 0.774 | 0.221 | 0.119 | heldout_stratified |
| default_of_credit_card_clients | primary | logistic_regression | 21000 | 9000 | 0.808 | 0.712 | 0.221 | 0.075 | heldout_stratified |
| default_of_credit_card_clients | primary | xgboost | 21000 | 9000 | 0.817 | 0.771 | 0.221 | 0.117 | heldout_stratified |

## Explainer Leaderboard

| Explainer | Configurations | Instances | Mean Faithfulness | Mean Stability | Mean Comprehensibility | Mean CSS | Mean Actionability | Mean ALF | Mean Invalid Recourse Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| anchors | 6 | 144 | 0.930 | 0.581 | 0.986 | 0.890 | 0.330 | n/a | n/a |
| dice | 6 | 144 | 0.924 | 0.471 | 0.722 | 0.985 | 0.748 | 0.523 | 0.000 |
| lime | 6 | 144 | 0.798 | 0.902 | 0.849 | 0.873 | 0.338 | n/a | n/a |
| shap | 6 | 144 | 0.911 | 0.854 | 0.817 | 0.894 | 0.339 | n/a | n/a |
| wachter | 6 | 144 | 0.918 | 0.467 | 0.725 | 0.985 | 0.747 | 0.521 | 0.007 |

## Bootstrap Uncertainty

- `95%` hierarchical-bootstrap confidence intervals with `1000` resamples over `dataset_id -> track -> model_id` benchmark clusters.

| Explainer | Faithfulness CI | Stability CI | Comprehensibility CI | CSS CI | Actionability CI | ALF CI |
| --- | --- | --- | --- | --- | --- | --- |
| anchors | [0.904, 0.954] | [0.538, 0.633] | [0.975, 0.995] | [0.880, 0.900] | [0.320, 0.340] | n/a |
| dice | [0.883, 0.960] | [0.264, 0.678] | [0.713, 0.734] | [0.981, 0.990] | [0.744, 0.753] | [0.518, 0.530] |
| lime | [0.747, 0.854] | [0.839, 0.958] | [0.846, 0.850] | [0.854, 0.892] | [0.330, 0.345] | n/a |
| shap | [0.880, 0.943] | [0.803, 0.886] | [0.784, 0.838] | [0.886, 0.901] | [0.334, 0.345] | n/a |
| wachter | [0.869, 0.961] | [0.207, 0.701] | [0.716, 0.739] | [0.981, 0.992] | [0.742, 0.754] | [0.513, 0.531] |

## Metric Sensitivity

- Strongest positive rank correlation: `actionability` vs `css` = `0.872`
- Weakest rank correlation: `actionability` vs `comprehensibility` = `-1.000`

| Metric A | Metric B | Spearman Rank Correlation | Shared Explainers |
| --- | --- | ---: | ---: |
| actionability | alf | 1.000 | 2 |
| actionability | comprehensibility | -1.000 | 5 |
| actionability | css | 0.872 | 5 |
| actionability | faithfulness | 0.000 | 5 |
| actionability | stability | -0.600 | 5 |
| alf | comprehensibility | -1.000 | 2 |
| alf | css | n/a | 2 |
| alf | faithfulness | 1.000 | 2 |
| alf | stability | 1.000 | 2 |
| comprehensibility | css | -0.872 | 5 |
| comprehensibility | faithfulness | 0.000 | 5 |
| comprehensibility | stability | 0.600 | 5 |
| css | faithfulness | 0.359 | 5 |
| css | stability | -0.872 | 5 |
| faithfulness | stability | -0.600 | 5 |

- Pairs with fewer than three shared explainers are descriptive only and should not be over-interpreted.

## Weight Sensitivity

- Randomized universal-weight sensitivity: `dice` wins `0.536` of draws with mean rank `1.969`.
- Randomized universal-weight sensitivity: `anchors` wins `0.247` of draws with mean rank `2.868`.
- Randomized universal-weight sensitivity: `shap` wins `0.142` of draws with mean rank `3.030`.
- Randomized universal-weight sensitivity: `lime` wins `0.059` of draws with mean rank `4.252`.
- Randomized universal-weight sensitivity: `wachter` wins `0.016` of draws with mean rank `2.881`.

| Scenario | Scope | Explainer | Composite Score | Rank |
| --- | --- | --- | ---: | ---: |
| balanced | all_explainers | dice | 0.594 | 1 |
| balanced | all_explainers | wachter | 0.584 | 2 |
| balanced | all_explainers | anchors | 0.482 | 3 |
| balanced | all_explainers | shap | 0.464 | 4 |
| balanced | all_explainers | lime | 0.299 | 5 |
| reliability_focused | all_explainers | shap | 0.744 | 1 |
| reliability_focused | all_explainers | anchors | 0.649 | 2 |
| reliability_focused | all_explainers | dice | 0.535 | 3 |
| reliability_focused | all_explainers | wachter | 0.511 | 4 |
| reliability_focused | all_explainers | lime | 0.399 | 5 |
| applicant_facing | all_explainers | dice | 0.698 | 1 |
| applicant_facing | all_explainers | wachter | 0.697 | 2 |
| applicant_facing | all_explainers | anchors | 0.366 | 3 |
| applicant_facing | all_explainers | shap | 0.250 | 4 |
| applicant_facing | all_explainers | lime | 0.175 | 5 |
| contestability_focused | all_explainers | dice | 0.798 | 1 |
| contestability_focused | all_explainers | wachter | 0.796 | 2 |
| contestability_focused | all_explainers | anchors | 0.288 | 3 |
| contestability_focused | all_explainers | shap | 0.241 | 4 |
| contestability_focused | all_explainers | lime | 0.126 | 5 |
| recourse_balanced | recourse_only | dice | 1.000 | 1 |
| recourse_balanced | recourse_only | wachter | 0.000 | 2 |
| legality_constrained | recourse_only | dice | 1.000 | 1 |
| legality_constrained | recourse_only | wachter | 0.000 | 2 |

## Metric Ablation

- Balanced composite baseline leader: `dice`.
- Leave-one-metric-out scenarios preserving the same leader: `3` / `5`.

| Scenario | Omitted Metric | Leader | Composite Score | Leader Changed |
| --- | --- | --- | ---: | --- |
| all_metrics_balanced | none | dice | 0.594 | no |
| drop_faithfulness | faithfulness | dice | 0.503 | no |
| drop_stability | stability | dice | 0.740 | no |
| drop_comprehensibility | comprehensibility | dice | 0.742 | no |
| drop_css | css | anchors | 0.566 | yes |
| drop_actionability | actionability | anchors | 0.603 | yes |

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
| CSS | specificity_light_css | dice | 0.985 | no |
| ALF | observed_alf | dice | 0.523 | no |
| ALF | balanced_arithmetic_alf | dice | 0.821 | no |
| ALF | legality_dominant_alf | dice | 0.896 | no |
| ALF | feasibility_dominant_alf | dice | 0.822 | no |
| ALF | minimality_dominant_alf | wachter | 0.792 | yes |
| ALF | invalidity_penalized_alf | dice | 0.821 | no |
| ALF | hard_failure_penalized_alf | dice | 0.821 | no |

## Track Policy Sensitivity

- Largest average absolute primary-vs-audit shift: `stability` = `0.052`.
- Most track-sensitive explainer: `anchors` with overall mean absolute delta `0.036`.
- Most track-robust explainer: `dice` with overall mean absolute delta `0.011`.
- Least leader-stable metric across primary/audit: `faithfulness` with same-leader rate `0.667`.

| Explainer | Matched Pairs | Overall Mean Abs Delta | Abs Faithfulness | Abs Stability | Abs Comprehensibility | Abs CSS | Abs Actionability | Abs ALF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dice | 3 | 0.011 | 0.013 | 0.021 | 0.016 | 0.007 | 0.006 | 0.004 |
| lime | 3 | 0.016 | 0.026 | 0.046 | 0.003 | 0.003 | 0.001 | n/a |
| wachter | 3 | 0.018 | 0.014 | 0.059 | 0.015 | 0.007 | 0.008 | 0.007 |
| shap | 3 | 0.024 | 0.017 | 0.058 | 0.026 | 0.011 | 0.007 | n/a |
| anchors | 3 | 0.036 | 0.043 | 0.076 | 0.016 | 0.024 | 0.024 | n/a |

| Metric | Same-Leader Rate | Same-Leader Count | Comparison Count |
| --- | ---: | ---: | ---: |
| faithfulness | 0.667 | 2 | 3 |
| stability | 0.667 | 2 | 3 |
| comprehensibility | 1.000 | 3 | 3 |
| css | 1.000 | 3 | 3 |
| actionability | 0.667 | 2 | 3 |
| alf | 0.667 | 2 | 3 |

## Leave-One-Dataset-Out

| Omitted Dataset | Metric | Leader | Score |
| --- | --- | --- | ---: |

## Dataset and Track Winners

| Dataset | Track | Best Faithfulness | Best Stability | Best Comprehensibility | Best CSS | Best Actionability | Best ALF |
| --- | --- | --- | --- | --- | --- | --- | --- |
| default_of_credit_card_clients | audit | dice (0.927) | lime (0.878) | anchors (0.982) | dice (0.982) | dice (0.745) | dice (0.521) |
| default_of_credit_card_clients | primary | anchors (0.951) | lime (0.925) | anchors (0.990) | dice (0.988) | dice (0.750) | dice (0.525) |

## Pairwise Win Leaders

- `actionability`: `dice` leads with `23` wins across `24` comparable pairwise configuration matchups.
- `alf`: `dice` leads with `5` wins across `6` comparable pairwise configuration matchups.
- `comprehensibility`: `anchors` leads with `24` wins across `24` comparable pairwise configuration matchups.
- `css`: `wachter` leads with `18` wins across `24` comparable pairwise configuration matchups.
- `faithfulness`: `dice` leads with `15` wins across `24` comparable pairwise configuration matchups.
- `stability`: `lime` leads with `23` wins across `24` comparable pairwise configuration matchups.

## Best Configurations

- Best faithfulness configuration: `default_of_credit_card_clients` / `primary` / `xgboost` / `wachter` with `mean_faithfulness=0.969`
- Best stability configuration: `default_of_credit_card_clients` / `primary` / `logistic_regression` / `lime` with `mean_stability=0.990`
- Best comprehensibility configuration: `default_of_credit_card_clients` / `primary` / `logistic_regression` / `anchors` with `mean_comprehensibility=1.000`
- Best CSS configuration: `default_of_credit_card_clients` / `primary` / `logistic_regression` / `dice` with `mean_css=0.997`
- Best actionability configuration: `default_of_credit_card_clients` / `primary` / `logistic_regression` / `dice` with `mean_actionability=0.759`
- Best ALF configuration: `default_of_credit_card_clients` / `primary` / `ebm` / `wachter` with `mean_alf=0.538`
