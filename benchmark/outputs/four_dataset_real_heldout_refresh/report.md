# LegalBench-XAI Benchmark Report

- Experiment: `legalbench_xai_four_dataset_real`
- Generated at (UTC): `2026-03-15T23:16:10.379973+00:00`
- Total groups: `120`
- Datasets: `default_of_credit_card_clients, give_me_some_credit, heloc, south_german_credit`
- Tracks: `audit, primary`
- Models: `ebm, logistic_regression, xgboost`
- Explainers: `anchors, dice, lime, shap, wachter`

## Data Provenance

- Data source kinds: `local_override` x 120
- Evaluation split strategies: `heldout_stratified` x 120
- Stability evidence sources: `perturbation_probe` x 958, `perturbation_probe_insufficient_neighbors` x 2

## Backend Usage

- Model implementations: `ebm_classifier` x 40, `sklearn_logistic_regression` x 40, `xgboost_classifier` x 40
- Explainer implementations: `lime_library` x 24, `search_anchor_rule` x 24, `search_counterfactual_dice_style` x 24, `search_counterfactual_wachter_style` x 24, `shap_library` x 24

## Held-Out Model Performance

| Dataset | Track | Model | Train Rows | Test Rows | Threshold Mode | Threshold | Accuracy | ROC AUC | PR AUC | Brier | ECE | True Adverse Rate | Predicted Adverse Rate | Split |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| default_of_credit_card_clients | audit | ebm | 21000 | 9000 | fixed | 0.500 | 0.818 | 0.777 | 0.548 | 0.136 | 0.010 | 0.221 | 0.118 | heldout_stratified |
| default_of_credit_card_clients | audit | logistic_regression | 21000 | 9000 | fixed | 0.500 | 0.807 | 0.714 | 0.500 | 0.146 | 0.049 | 0.221 | 0.076 | heldout_stratified |
| default_of_credit_card_clients | audit | xgboost | 21000 | 9000 | fixed | 0.500 | 0.817 | 0.772 | 0.543 | 0.137 | 0.009 | 0.221 | 0.116 | heldout_stratified |
| default_of_credit_card_clients | primary | ebm | 21000 | 9000 | fixed | 0.500 | 0.818 | 0.774 | 0.540 | 0.137 | 0.011 | 0.221 | 0.119 | heldout_stratified |
| default_of_credit_card_clients | primary | logistic_regression | 21000 | 9000 | fixed | 0.500 | 0.808 | 0.712 | 0.498 | 0.147 | 0.047 | 0.221 | 0.075 | heldout_stratified |
| default_of_credit_card_clients | primary | xgboost | 21000 | 9000 | fixed | 0.500 | 0.817 | 0.771 | 0.543 | 0.137 | 0.011 | 0.221 | 0.117 | heldout_stratified |
| give_me_some_credit | audit | ebm | 105000 | 45000 | fixed | 0.500 | 0.938 | 0.862 | 0.403 | 0.049 | 0.004 | 0.067 | 0.021 | heldout_stratified |
| give_me_some_credit | audit | logistic_regression | 105000 | 45000 | fixed | 0.500 | 0.934 | 0.697 | 0.235 | 0.058 | 0.011 | 0.067 | 0.005 | heldout_stratified |
| give_me_some_credit | audit | xgboost | 105000 | 45000 | fixed | 0.500 | 0.938 | 0.861 | 0.402 | 0.049 | 0.006 | 0.067 | 0.018 | heldout_stratified |
| give_me_some_credit | primary | ebm | 105000 | 45000 | fixed | 0.500 | 0.938 | 0.858 | 0.400 | 0.049 | 0.003 | 0.067 | 0.022 | heldout_stratified |
| give_me_some_credit | primary | logistic_regression | 105000 | 45000 | fixed | 0.500 | 0.934 | 0.655 | 0.237 | 0.058 | 0.015 | 0.067 | 0.005 | heldout_stratified |
| give_me_some_credit | primary | xgboost | 105000 | 45000 | fixed | 0.500 | 0.938 | 0.857 | 0.398 | 0.049 | 0.005 | 0.067 | 0.018 | heldout_stratified |
| heloc | audit | ebm | 6910 | 2961 | fixed | 0.500 | 0.733 | 0.804 | 0.805 | 0.180 | 0.025 | 0.520 | 0.533 | heldout_stratified |
| heloc | audit | logistic_regression | 6910 | 2961 | fixed | 0.500 | 0.722 | 0.798 | 0.797 | 0.183 | 0.022 | 0.520 | 0.534 | heldout_stratified |
| heloc | audit | xgboost | 6910 | 2961 | fixed | 0.500 | 0.723 | 0.799 | 0.797 | 0.183 | 0.020 | 0.520 | 0.536 | heldout_stratified |
| heloc | primary | ebm | 6910 | 2961 | fixed | 0.500 | 0.732 | 0.803 | 0.805 | 0.181 | 0.018 | 0.520 | 0.530 | heldout_stratified |
| heloc | primary | logistic_regression | 6910 | 2961 | fixed | 0.500 | 0.726 | 0.795 | 0.794 | 0.185 | 0.015 | 0.520 | 0.530 | heldout_stratified |
| heloc | primary | xgboost | 6910 | 2961 | fixed | 0.500 | 0.727 | 0.799 | 0.798 | 0.183 | 0.018 | 0.520 | 0.550 | heldout_stratified |
| south_german_credit | audit | ebm | 700 | 300 | fixed | 0.500 | 0.780 | 0.822 | 0.657 | 0.154 | 0.057 | 0.300 | 0.180 | heldout_stratified |
| south_german_credit | audit | logistic_regression | 700 | 300 | fixed | 0.500 | 0.773 | 0.798 | 0.647 | 0.157 | 0.078 | 0.300 | 0.227 | heldout_stratified |
| south_german_credit | audit | xgboost | 700 | 300 | fixed | 0.500 | 0.767 | 0.804 | 0.645 | 0.159 | 0.051 | 0.300 | 0.167 | heldout_stratified |
| south_german_credit | primary | ebm | 700 | 300 | fixed | 0.500 | 0.763 | 0.819 | 0.654 | 0.154 | 0.059 | 0.300 | 0.183 | heldout_stratified |
| south_german_credit | primary | logistic_regression | 700 | 300 | fixed | 0.500 | 0.777 | 0.791 | 0.642 | 0.160 | 0.055 | 0.300 | 0.230 | heldout_stratified |
| south_german_credit | primary | xgboost | 700 | 300 | fixed | 0.500 | 0.773 | 0.799 | 0.637 | 0.161 | 0.069 | 0.300 | 0.153 | heldout_stratified |

## Explainer Leaderboard

| Explainer | Groups | Instances | Mean Faithfulness | Mean Stability | Mean Comprehensibility | Mean CSS | Mean Actionability | Mean ALF | Mean Invalid Recourse Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| anchors | 24 | 192 | 0.792 | 0.632 | 0.944 | 0.825 | 0.283 | n/a | n/a |
| dice | 24 | 192 | 0.839 | 0.485 | 0.684 | 0.895 | 0.671 | 0.476 | 0.104 |
| lime | 24 | 192 | 0.839 | 0.915 | 0.722 | 0.780 | 0.275 | n/a | n/a |
| shap | 24 | 192 | 0.909 | 0.908 | 0.710 | 0.794 | 0.274 | n/a | n/a |
| wachter | 24 | 192 | 0.827 | 0.559 | 0.683 | 0.892 | 0.679 | 0.492 | 0.008 |

## Bootstrap Uncertainty

- `95%` hierarchical-bootstrap confidence intervals with `1000` resamples over `dataset_id -> track -> model_id` benchmark clusters.

| Explainer | Faithfulness CI | Stability CI | Comprehensibility CI | CSS CI | Actionability CI | ALF CI |
| --- | --- | --- | --- | --- | --- | --- |
| anchors | [0.708, 0.879] | [0.590, 0.682] | [0.918, 0.975] | [0.776, 0.875] | [0.239, 0.324] | n/a |
| dice | [0.806, 0.898] | [0.346, 0.614] | [0.659, 0.715] | [0.834, 0.959] | [0.607, 0.732] | [0.435, 0.522] |
| lime | [0.799, 0.882] | [0.880, 0.954] | [0.670, 0.802] | [0.704, 0.860] | [0.219, 0.331] | n/a |
| shap | [0.867, 0.955] | [0.865, 0.958] | [0.668, 0.773] | [0.722, 0.865] | [0.219, 0.327] | n/a |
| wachter | [0.783, 0.886] | [0.340, 0.771] | [0.660, 0.712] | [0.829, 0.959] | [0.632, 0.728] | [0.451, 0.522] |

## Evidence-Source Transparency

| Explainer | Instances | Direct Faithfulness Share | Proxy Faithfulness Share | Direct Stability Share | Stability Fallback Share |
| --- | ---: | ---: | ---: | ---: | ---: |
| anchors | 192 | 1.000 | 0.000 | 0.995 | 0.005 |
| dice | 192 | 0.000 | 1.000 | 1.000 | 0.000 |
| lime | 192 | 1.000 | 0.000 | 1.000 | 0.000 |
| shap | 192 | 1.000 | 0.000 | 1.000 | 0.000 |
| wachter | 192 | 0.000 | 1.000 | 0.995 | 0.005 |

## Metric Sensitivity

- Strongest positive rank correlation: `actionability` vs `css` = `0.800`
- Weakest rank correlation: `css` vs `stability` = `-1.000`

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

- Pairs with fewer than three shared explainers are descriptive only and should not be over-interpreted.

## Group-Level Metric Correlation

- Strongest positive group-level Spearman correlation: `actionability` vs `css` = `0.840` over `120` groups.
- Weakest group-level Spearman correlation: `actionability` vs `stability` = `-0.521` over `120` groups.

| Metric A | Metric B | Spearman Rank Correlation | Shared Groups |
| --- | --- | ---: | ---: |
| actionability | alf | 0.667 | 48 |
| actionability | comprehensibility | -0.283 | 120 |
| actionability | css | 0.840 | 120 |
| actionability | faithfulness | -0.005 | 120 |
| actionability | stability | -0.521 | 120 |
| alf | comprehensibility | 0.423 | 48 |
| alf | css | 0.371 | 48 |
| alf | faithfulness | 0.337 | 48 |
| alf | stability | -0.146 | 48 |
| comprehensibility | css | 0.018 | 120 |
| comprehensibility | faithfulness | 0.042 | 120 |
| comprehensibility | stability | 0.014 | 120 |
| css | faithfulness | 0.225 | 120 |
| css | stability | -0.443 | 120 |
| faithfulness | stability | 0.163 | 120 |

## Weight Sensitivity

- Randomized universal-weight sensitivity: `shap` wins `0.352` of draws with mean rank `2.495`.
- Randomized universal-weight sensitivity: `wachter` wins `0.242` of draws with mean rank `2.451`.
- Randomized universal-weight sensitivity: `dice` wins `0.216` of draws with mean rank `2.658`.
- Randomized universal-weight sensitivity: `anchors` wins `0.180` of draws with mean rank `3.443`.
- Randomized universal-weight sensitivity: `lime` wins `0.010` of draws with mean rank `3.953`.

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

## Proxy Sensitivity

| Scenario | Explainer | Adjusted Faithfulness | Proxy Share | Composite Score | Rank |
| --- | --- | ---: | ---: | ---: | ---: |
| observed | wachter | 0.827 | 1.000 | 0.488 | 1 |
| observed | dice | 0.839 | 1.000 | 0.478 | 2 |
| observed | shap | 0.909 | 0.000 | 0.442 | 3 |
| observed | anchors | 0.792 | 0.000 | 0.350 | 4 |
| observed | lime | 0.839 | 0.000 | 0.311 | 5 |
| proxy_discount_25 | anchors | 0.792 | 0.000 | 0.469 | 1 |
| proxy_discount_25 | shap | 0.909 | 0.000 | 0.442 | 2 |
| proxy_discount_25 | wachter | 0.620 | 1.000 | 0.427 | 3 |
| proxy_discount_25 | dice | 0.629 | 1.000 | 0.403 | 4 |
| proxy_discount_25 | lime | 0.839 | 0.000 | 0.382 | 5 |
| proxy_cap_050 | anchors | 0.792 | 0.000 | 0.493 | 1 |
| proxy_cap_050 | shap | 0.909 | 0.000 | 0.442 | 2 |
| proxy_cap_050 | wachter | 0.500 | 1.000 | 0.427 | 3 |
| proxy_cap_050 | dice | 0.500 | 1.000 | 0.397 | 4 |
| proxy_cap_050 | lime | 0.839 | 0.000 | 0.396 | 5 |

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

## Metric Definition Sensitivity

- CSS baseline leader under observed scoring: `dice`.
- CSS alternative definitions preserving the same leader: `4` / `4`.
- ALF baseline leader under observed scoring: `wachter`.
- ALF alternative definitions preserving the same leader: `6` / `6`.

| Family | Scenario | Leader | Score | Leader Changed |
| --- | --- | --- | ---: | --- |
| CSS | observed_css | dice | 0.895 | no |
| CSS | reconstructed_css | dice | 0.895 | no |
| CSS | verifiability_focused_css | dice | 0.918 | no |
| CSS | challenge_focused_css | dice | 0.895 | no |
| CSS | specificity_light_css | dice | 0.926 | no |
| ALF | observed_alf | wachter | 0.492 | no |
| ALF | balanced_arithmetic_alf | wachter | 0.804 | no |
| ALF | legality_dominant_alf | wachter | 0.887 | no |
| ALF | feasibility_dominant_alf | wachter | 0.807 | no |
| ALF | minimality_dominant_alf | wachter | 0.770 | no |
| ALF | invalidity_penalized_alf | wachter | 0.798 | no |
| ALF | hard_failure_penalized_alf | wachter | 0.796 | no |

## Track Policy Sensitivity

- Largest average absolute primary-vs-audit shift: `stability` = `0.055`.
- Most track-sensitive explainer: `anchors` with overall mean absolute delta `0.036`.
- Most track-robust explainer: `shap` with overall mean absolute delta `0.014`.
- Least leader-stable metric across primary/audit: `stability` with same-leader rate `0.667`.

| Explainer | Matched Pairs | Overall Mean Abs Delta | Abs Faithfulness | Abs Stability | Abs Comprehensibility | Abs CSS | Abs Actionability | Abs ALF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| shap | 12 | 0.014 | 0.012 | 0.026 | 0.011 | 0.012 | 0.007 | n/a |
| lime | 12 | 0.017 | 0.027 | 0.028 | 0.003 | 0.019 | 0.007 | n/a |
| wachter | 12 | 0.019 | 0.008 | 0.059 | 0.007 | 0.006 | 0.013 | 0.023 |
| dice | 12 | 0.024 | 0.007 | 0.082 | 0.008 | 0.005 | 0.016 | 0.029 |
| anchors | 12 | 0.036 | 0.032 | 0.081 | 0.017 | 0.024 | 0.024 | n/a |

| Metric | Same-Leader Rate | Same-Leader Count | Comparison Count |
| --- | ---: | ---: | ---: |
| faithfulness | 1.000 | 12 | 12 |
| stability | 0.667 | 8 | 12 |
| comprehensibility | 1.000 | 12 | 12 |
| css | 1.000 | 12 | 12 |
| actionability | 0.667 | 8 | 12 |
| alf | 0.667 | 8 | 12 |

## Leave-One-Dataset-Out

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

## Subgroup Slices

| Dataset | Track | Feature | Value | Explainer | Instances | Mean Stability | Mean CSS | Mean Actionability | Mean ALF |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| default_of_credit_card_clients | audit | age_band | 30-49 | anchors | 14 | 0.608 | 0.903 | 0.343 | n/a |
| default_of_credit_card_clients | audit | age_band | 30-49 | dice | 14 | 0.569 | 0.983 | 0.748 | 0.525 |
| default_of_credit_card_clients | audit | age_band | 30-49 | lime | 14 | 0.884 | 0.875 | 0.338 | n/a |
| default_of_credit_card_clients | audit | age_band | 30-49 | shap | 14 | 0.892 | 0.880 | 0.335 | n/a |
| default_of_credit_card_clients | audit | age_band | 30-49 | wachter | 14 | 0.565 | 0.983 | 0.748 | 0.526 |
| default_of_credit_card_clients | audit | age_band | 50+ | anchors | 2 | 0.591 | 0.910 | 0.350 | n/a |
| default_of_credit_card_clients | audit | age_band | 50+ | dice | 2 | 0.164 | 0.988 | 0.749 | 0.522 |
| default_of_credit_card_clients | audit | age_band | 50+ | lime | 2 | 0.884 | 0.855 | 0.333 | n/a |
| default_of_credit_card_clients | audit | age_band | 50+ | shap | 2 | 0.805 | 0.865 | 0.330 | n/a |
| default_of_credit_card_clients | audit | age_band | 50+ | wachter | 2 | 0.150 | 0.988 | 0.749 | 0.522 |
| default_of_credit_card_clients | audit | age_band | <30 | anchors | 8 | 0.675 | 0.885 | 0.325 | n/a |
| default_of_credit_card_clients | audit | age_band | <30 | dice | 8 | 0.484 | 0.980 | 0.758 | 0.551 |
| default_of_credit_card_clients | audit | age_band | <30 | lime | 8 | 0.888 | 0.876 | 0.338 | n/a |
| default_of_credit_card_clients | audit | age_band | <30 | shap | 8 | 0.840 | 0.888 | 0.338 | n/a |
| default_of_credit_card_clients | audit | age_band | <30 | wachter | 8 | 0.469 | 0.980 | 0.758 | 0.551 |
| default_of_credit_card_clients | audit | education_group | graduate_school | anchors | 8 | 0.661 | 0.898 | 0.338 | n/a |
| default_of_credit_card_clients | audit | education_group | graduate_school | dice | 8 | 0.387 | 0.977 | 0.749 | 0.535 |
| default_of_credit_card_clients | audit | education_group | graduate_school | lime | 8 | 0.847 | 0.864 | 0.333 | n/a |
| default_of_credit_card_clients | audit | education_group | graduate_school | shap | 8 | 0.833 | 0.879 | 0.338 | n/a |
| default_of_credit_card_clients | audit | education_group | graduate_school | wachter | 8 | 0.362 | 0.977 | 0.749 | 0.535 |
| default_of_credit_card_clients | audit | education_group | high_school | anchors | 6 | 0.602 | 0.902 | 0.342 | n/a |
| default_of_credit_card_clients | audit | education_group | high_school | dice | 6 | 0.452 | 0.979 | 0.747 | 0.529 |
| default_of_credit_card_clients | audit | education_group | high_school | lime | 6 | 0.897 | 0.875 | 0.340 | n/a |
| default_of_credit_card_clients | audit | education_group | high_school | shap | 6 | 0.907 | 0.883 | 0.337 | n/a |
| default_of_credit_card_clients | audit | education_group | high_school | wachter | 6 | 0.362 | 0.979 | 0.746 | 0.529 |
| default_of_credit_card_clients | audit | education_group | university | anchors | 10 | 0.619 | 0.895 | 0.335 | n/a |
| default_of_credit_card_clients | audit | education_group | university | dice | 10 | 0.636 | 0.989 | 0.756 | 0.535 |
| default_of_credit_card_clients | audit | education_group | university | lime | 10 | 0.908 | 0.881 | 0.339 | n/a |
| default_of_credit_card_clients | audit | education_group | university | shap | 10 | 0.871 | 0.882 | 0.332 | n/a |
| default_of_credit_card_clients | audit | education_group | university | wachter | 10 | 0.688 | 0.989 | 0.757 | 0.536 |

## Dataset and Track Winners

| Dataset | Track | Best Faithfulness | Best Stability | Best Comprehensibility | Best CSS | Best Actionability | Best ALF |
| --- | --- | --- | --- | --- | --- | --- | --- |
| default_of_credit_card_clients | audit | dice (0.928) | lime (0.885) | anchors (0.988) | dice (0.982) | wachter (0.752) | wachter (0.534) |
| default_of_credit_card_clients | primary | anchors (0.944) | lime (0.920) | anchors (0.991) | dice (0.989) | dice (0.755) | dice (0.532) |
| give_me_some_credit | audit | shap (0.976) | shap (0.896) | anchors (0.916) | dice (0.904) | wachter (0.664) | wachter (0.435) |
| give_me_some_credit | primary | shap (0.983) | lime (0.893) | anchors (0.916) | dice (0.907) | wachter (0.667) | wachter (0.437) |
| heloc | audit | shap (0.870) | lime (0.989) | anchors (0.939) | dice (0.877) | wachter (0.671) | wachter (0.480) |
| heloc | primary | shap (0.861) | lime (0.948) | anchors (0.947) | dice (0.877) | dice (0.701) | dice (0.532) |
| south_german_credit | audit | shap (0.873) | shap (0.977) | anchors (0.913) | dice (0.814) | wachter (0.612) | wachter (0.493) |
| south_german_credit | primary | shap (0.881) | shap (0.980) | anchors (0.939) | dice (0.813) | wachter (0.617) | wachter (0.506) |

## Pairwise Win Leaders

- `actionability`: `wachter` leads with `86` wins across `96` comparable pairwise group matchups.
- `alf`: `wachter` leads with `14` wins across `24` comparable pairwise group matchups.
- `comprehensibility`: `anchors` leads with `96` wins across `96` comparable pairwise group matchups.
- `css`: `dice` leads with `78` wins across `96` comparable pairwise group matchups.
- `faithfulness`: `shap` leads with `79` wins across `96` comparable pairwise group matchups.
- `stability`: `lime` leads with `82` wins across `96` comparable pairwise group matchups.

## Best Groups

- Best faithfulness group: `give_me_some_credit` / `primary` / `logistic_regression` / `shap` with `mean_faithfulness=0.997`
- Best stability group: `heloc` / `audit` / `logistic_regression` / `lime` with `mean_stability=1.000`
- Best comprehensibility group: `default_of_credit_card_clients` / `primary` / `logistic_regression` / `anchors` with `mean_comprehensibility=1.000`
- Best CSS group: `default_of_credit_card_clients` / `primary` / `logistic_regression` / `dice` with `mean_css=0.996`
- Best actionability group: `default_of_credit_card_clients` / `primary` / `logistic_regression` / `dice` with `mean_actionability=0.760`
- Best ALF group: `heloc` / `primary` / `logistic_regression` / `dice` with `mean_alf=0.578`
