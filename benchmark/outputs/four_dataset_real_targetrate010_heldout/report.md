# LegalBench-XAI Benchmark Report

- Experiment: `legalbench_xai_four_dataset_real_targetrate010`
- Generated at (UTC): `2026-03-16T00:13:10.989146+00:00`
- Total configurations: `120`
- Datasets: `default_of_credit_card_clients, give_me_some_credit, heloc, south_german_credit`
- Tracks: `audit, primary`
- Models: `ebm, logistic_regression, xgboost`
- Explainers: `anchors, dice, lime, shap, wachter`

## Data Provenance

- Data source kinds: `local_override` x 120
- Evaluation split strategies: `heldout_stratified` x 120
- Stability evidence sources: `perturbation_probe` x 957, `perturbation_probe_insufficient_neighbors` x 3

## Backend Usage

- Model implementations: `ebm_classifier` x 40, `sklearn_logistic_regression` x 40, `xgboost_classifier` x 40
- Explainer implementations: `lime_library` x 24, `search_anchor_rule` x 24, `search_counterfactual_dice_style` x 24, `search_counterfactual_wachter_style` x 24, `shap_library` x 24

## Held-Out Model Performance

| Dataset | Track | Model | Train Rows | Test Rows | Threshold Mode | Threshold | Accuracy | ROC AUC | PR AUC | Brier | ECE | True Adverse Rate | Predicted Adverse Rate | Split |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| default_of_credit_card_clients | audit | ebm | 21000 | 9000 | target_predicted_adverse_rate | 0.577 | 0.816 | 0.777 | 0.548 | 0.136 | 0.010 | 0.221 | 0.100 | heldout_stratified |
| default_of_credit_card_clients | audit | logistic_regression | 21000 | 9000 | target_predicted_adverse_rate | 0.462 | 0.812 | 0.714 | 0.500 | 0.146 | 0.049 | 0.221 | 0.100 | heldout_stratified |
| default_of_credit_card_clients | audit | xgboost | 21000 | 9000 | target_predicted_adverse_rate | 0.577 | 0.817 | 0.772 | 0.543 | 0.137 | 0.009 | 0.221 | 0.100 | heldout_stratified |
| default_of_credit_card_clients | primary | ebm | 21000 | 9000 | target_predicted_adverse_rate | 0.577 | 0.816 | 0.774 | 0.540 | 0.137 | 0.011 | 0.221 | 0.100 | heldout_stratified |
| default_of_credit_card_clients | primary | logistic_regression | 21000 | 9000 | target_predicted_adverse_rate | 0.463 | 0.813 | 0.712 | 0.498 | 0.147 | 0.047 | 0.221 | 0.100 | heldout_stratified |
| default_of_credit_card_clients | primary | xgboost | 21000 | 9000 | target_predicted_adverse_rate | 0.579 | 0.817 | 0.771 | 0.543 | 0.137 | 0.011 | 0.221 | 0.100 | heldout_stratified |
| give_me_some_credit | audit | ebm | 105000 | 45000 | target_predicted_adverse_rate | 0.158 | 0.907 | 0.862 | 0.403 | 0.049 | 0.004 | 0.067 | 0.100 | heldout_stratified |
| give_me_some_credit | audit | logistic_regression | 105000 | 45000 | target_predicted_adverse_rate | 0.108 | 0.882 | 0.697 | 0.235 | 0.058 | 0.011 | 0.067 | 0.100 | heldout_stratified |
| give_me_some_credit | audit | xgboost | 105000 | 45000 | target_predicted_adverse_rate | 0.157 | 0.907 | 0.861 | 0.402 | 0.049 | 0.006 | 0.067 | 0.100 | heldout_stratified |
| give_me_some_credit | primary | ebm | 105000 | 45000 | target_predicted_adverse_rate | 0.159 | 0.906 | 0.858 | 0.400 | 0.049 | 0.003 | 0.067 | 0.100 | heldout_stratified |
| give_me_some_credit | primary | logistic_regression | 105000 | 45000 | target_predicted_adverse_rate | 0.092 | 0.886 | 0.655 | 0.237 | 0.058 | 0.015 | 0.067 | 0.100 | heldout_stratified |
| give_me_some_credit | primary | xgboost | 105000 | 45000 | target_predicted_adverse_rate | 0.160 | 0.907 | 0.857 | 0.398 | 0.049 | 0.005 | 0.067 | 0.100 | heldout_stratified |
| heloc | audit | ebm | 6910 | 2961 | target_predicted_adverse_rate | 0.871 | 0.558 | 0.804 | 0.805 | 0.180 | 0.025 | 0.520 | 0.100 | heldout_stratified |
| heloc | audit | logistic_regression | 6910 | 2961 | target_predicted_adverse_rate | 0.870 | 0.558 | 0.798 | 0.797 | 0.183 | 0.022 | 0.520 | 0.100 | heldout_stratified |
| heloc | audit | xgboost | 6910 | 2961 | target_predicted_adverse_rate | 0.834 | 0.557 | 0.799 | 0.797 | 0.183 | 0.020 | 0.520 | 0.100 | heldout_stratified |
| heloc | primary | ebm | 6910 | 2961 | target_predicted_adverse_rate | 0.871 | 0.559 | 0.803 | 0.805 | 0.181 | 0.018 | 0.520 | 0.100 | heldout_stratified |
| heloc | primary | logistic_regression | 6910 | 2961 | target_predicted_adverse_rate | 0.870 | 0.560 | 0.795 | 0.794 | 0.185 | 0.015 | 0.520 | 0.100 | heldout_stratified |
| heloc | primary | xgboost | 6910 | 2961 | target_predicted_adverse_rate | 0.826 | 0.558 | 0.799 | 0.798 | 0.183 | 0.018 | 0.520 | 0.100 | heldout_stratified |
| south_german_credit | audit | ebm | 700 | 300 | target_predicted_adverse_rate | 0.594 | 0.750 | 0.822 | 0.657 | 0.154 | 0.057 | 0.300 | 0.097 | heldout_stratified |
| south_german_credit | audit | logistic_regression | 700 | 300 | target_predicted_adverse_rate | 0.717 | 0.743 | 0.798 | 0.647 | 0.157 | 0.078 | 0.300 | 0.097 | heldout_stratified |
| south_german_credit | audit | xgboost | 700 | 300 | target_predicted_adverse_rate | 0.584 | 0.763 | 0.804 | 0.645 | 0.159 | 0.051 | 0.300 | 0.097 | heldout_stratified |
| south_german_credit | primary | ebm | 700 | 300 | target_predicted_adverse_rate | 0.609 | 0.750 | 0.819 | 0.654 | 0.154 | 0.059 | 0.300 | 0.097 | heldout_stratified |
| south_german_credit | primary | logistic_regression | 700 | 300 | target_predicted_adverse_rate | 0.708 | 0.737 | 0.791 | 0.642 | 0.160 | 0.055 | 0.300 | 0.097 | heldout_stratified |
| south_german_credit | primary | xgboost | 700 | 300 | target_predicted_adverse_rate | 0.583 | 0.763 | 0.799 | 0.637 | 0.161 | 0.069 | 0.300 | 0.097 | heldout_stratified |

## Explainer Leaderboard

| Explainer | Configurations | Instances | Mean Faithfulness | Mean Stability | Mean Comprehensibility | Mean CSS | Mean Actionability | Mean ALF | Mean Invalid Recourse Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| anchors | 24 | 192 | 0.805 | 0.588 | 0.943 | 0.827 | 0.286 | n/a | n/a |
| dice | 24 | 192 | 0.841 | 0.488 | 0.681 | 0.893 | 0.671 | 0.486 | 0.083 |
| lime | 24 | 192 | 0.842 | 0.918 | 0.721 | 0.780 | 0.275 | n/a | n/a |
| shap | 24 | 192 | 0.913 | 0.908 | 0.710 | 0.792 | 0.271 | n/a | n/a |
| wachter | 24 | 192 | 0.833 | 0.525 | 0.684 | 0.891 | 0.683 | 0.509 | 0.000 |

## Bootstrap Uncertainty

- `95%` hierarchical-bootstrap confidence intervals with `1000` resamples over `dataset_id -> track -> model_id` benchmark clusters.

| Explainer | Faithfulness CI | Stability CI | Comprehensibility CI | CSS CI | Actionability CI | ALF CI |
| --- | --- | --- | --- | --- | --- | --- |
| anchors | [0.712, 0.894] | [0.529, 0.646] | [0.919, 0.975] | [0.779, 0.874] | [0.242, 0.324] | n/a |
| dice | [0.806, 0.907] | [0.349, 0.617] | [0.656, 0.707] | [0.838, 0.956] | [0.612, 0.727] | [0.447, 0.525] |
| lime | [0.798, 0.891] | [0.877, 0.960] | [0.671, 0.799] | [0.701, 0.860] | [0.219, 0.332] | n/a |
| shap | [0.878, 0.954] | [0.857, 0.959] | [0.668, 0.774] | [0.731, 0.855] | [0.222, 0.320] | n/a |
| wachter | [0.790, 0.895] | [0.340, 0.714] | [0.660, 0.711] | [0.835, 0.955] | [0.639, 0.729] | [0.477, 0.542] |

## Evidence-Source Transparency

| Explainer | Instances | Direct Faithfulness Share | Proxy Faithfulness Share | Direct Stability Share | Stability Fallback Share |
| --- | ---: | ---: | ---: | ---: | ---: |
| anchors | 192 | 1.000 | 0.000 | 1.000 | 0.000 |
| dice | 192 | 0.000 | 1.000 | 0.995 | 0.005 |
| lime | 192 | 1.000 | 0.000 | 0.995 | 0.005 |
| shap | 192 | 1.000 | 0.000 | 0.995 | 0.005 |
| wachter | 192 | 0.000 | 1.000 | 1.000 | 0.000 |

## Metric Sensitivity

- Strongest positive rank correlation: `actionability` vs `css` = `0.800`
- Weakest rank correlation: `css` vs `stability` = `-1.000`

| Metric A | Metric B | Spearman Rank Correlation | Shared Explainers |
| --- | --- | ---: | ---: |
| actionability | alf | 1.000 | 2 |
| actionability | comprehensibility | -0.500 | 5 |
| actionability | css | 0.800 | 5 |
| actionability | faithfulness | -0.700 | 5 |
| actionability | stability | -0.800 | 5 |
| alf | comprehensibility | 1.000 | 2 |
| alf | css | -1.000 | 2 |
| alf | faithfulness | -1.000 | 2 |
| alf | stability | 1.000 | 2 |
| comprehensibility | css | -0.700 | 5 |
| comprehensibility | faithfulness | -0.200 | 5 |
| comprehensibility | stability | 0.700 | 5 |
| css | faithfulness | -0.500 | 5 |
| css | stability | -1.000 | 5 |
| faithfulness | stability | 0.500 | 5 |

- Pairs with fewer than three shared explainers are descriptive only and should not be over-interpreted.

## Group-Level Metric Correlation

- Strongest positive group-level Spearman correlation: `actionability` vs `css` = `0.851` over `120` groups.
- Weakest group-level Spearman correlation: `actionability` vs `stability` = `-0.625` over `120` groups.

| Metric A | Metric B | Spearman Rank Correlation | Shared Configurations |
| --- | --- | ---: | ---: |
| actionability | alf | 0.728 | 48 |
| actionability | comprehensibility | -0.284 | 120 |
| actionability | css | 0.851 | 120 |
| actionability | faithfulness | -0.043 | 120 |
| actionability | stability | -0.625 | 120 |
| alf | comprehensibility | 0.179 | 48 |
| alf | css | 0.464 | 48 |
| alf | faithfulness | 0.286 | 48 |
| alf | stability | -0.309 | 48 |
| comprehensibility | css | 0.034 | 120 |
| comprehensibility | faithfulness | 0.078 | 120 |
| comprehensibility | stability | -0.064 | 120 |
| css | faithfulness | 0.150 | 120 |
| css | stability | -0.552 | 120 |
| faithfulness | stability | 0.169 | 120 |

## Weight Sensitivity

- Randomized universal-weight sensitivity: `shap` wins `0.371` of draws with mean rank `2.459`.
- Randomized universal-weight sensitivity: `wachter` wins `0.272` of draws with mean rank `2.453`.
- Randomized universal-weight sensitivity: `anchors` wins `0.173` of draws with mean rank `3.457`.
- Randomized universal-weight sensitivity: `dice` wins `0.167` of draws with mean rank `2.717`.
- Randomized universal-weight sensitivity: `lime` wins `0.017` of draws with mean rank `3.914`.

| Scenario | Scope | Explainer | Composite Score | Rank |
| --- | --- | --- | ---: | ---: |
| balanced | all_explainers | wachter | 0.467 | 1 |
| balanced | all_explainers | dice | 0.461 | 2 |
| balanced | all_explainers | shap | 0.439 | 3 |
| balanced | all_explainers | anchors | 0.337 | 4 |
| balanced | all_explainers | lime | 0.300 | 5 |
| reliability_focused | all_explainers | shap | 0.808 | 1 |
| reliability_focused | all_explainers | lime | 0.517 | 2 |
| reliability_focused | all_explainers | dice | 0.248 | 3 |
| reliability_focused | all_explainers | wachter | 0.245 | 4 |
| reliability_focused | all_explainers | anchors | 0.204 | 5 |
| applicant_facing | all_explainers | wachter | 0.663 | 1 |
| applicant_facing | all_explainers | dice | 0.658 | 2 |
| applicant_facing | all_explainers | anchors | 0.419 | 3 |
| applicant_facing | all_explainers | shap | 0.164 | 4 |
| applicant_facing | all_explainers | lime | 0.108 | 5 |
| contestability_focused | all_explainers | dice | 0.760 | 1 |
| contestability_focused | all_explainers | wachter | 0.759 | 2 |
| contestability_focused | all_explainers | anchors | 0.379 | 3 |
| contestability_focused | all_explainers | shap | 0.169 | 4 |
| contestability_focused | all_explainers | lime | 0.092 | 5 |
| recourse_balanced | recourse_only | wachter | 1.000 | 1 |
| recourse_balanced | recourse_only | dice | 0.000 | 2 |
| legality_constrained | recourse_only | wachter | 1.000 | 1 |
| legality_constrained | recourse_only | dice | 0.000 | 2 |

## Proxy Sensitivity

| Scenario | Explainer | Adjusted Faithfulness | Proxy Share | Composite Score | Rank |
| --- | --- | ---: | ---: | ---: | ---: |
| observed | wachter | 0.833 | 1.000 | 0.467 | 1 |
| observed | dice | 0.841 | 1.000 | 0.461 | 2 |
| observed | shap | 0.913 | 0.000 | 0.439 | 3 |
| observed | anchors | 0.805 | 0.000 | 0.337 | 4 |
| observed | lime | 0.842 | 0.000 | 0.300 | 5 |
| proxy_discount_25 | anchors | 0.805 | 0.000 | 0.462 | 1 |
| proxy_discount_25 | shap | 0.913 | 0.000 | 0.439 | 2 |
| proxy_discount_25 | wachter | 0.625 | 1.000 | 0.416 | 3 |
| proxy_discount_25 | dice | 0.631 | 1.000 | 0.399 | 4 |
| proxy_discount_25 | lime | 0.842 | 0.000 | 0.383 | 5 |
| proxy_cap_050 | anchors | 0.805 | 0.000 | 0.485 | 1 |
| proxy_cap_050 | shap | 0.913 | 0.000 | 0.439 | 2 |
| proxy_cap_050 | wachter | 0.500 | 1.000 | 0.416 | 3 |
| proxy_cap_050 | lime | 0.842 | 0.000 | 0.398 | 4 |
| proxy_cap_050 | dice | 0.500 | 1.000 | 0.394 | 5 |

## Metric Ablation

- Balanced composite baseline leader: `wachter`.
- Leave-one-metric-out scenarios preserving the same leader: `2` / `5`.

| Scenario | Omitted Metric | Leader | Composite Score | Leader Changed |
| --- | --- | --- | ---: | --- |
| all_metrics_balanced | none | wachter | 0.467 | no |
| drop_faithfulness | faithfulness | wachter | 0.520 | no |
| drop_stability | stability | dice | 0.576 | yes |
| drop_comprehensibility | comprehensibility | wachter | 0.580 | no |
| drop_css | css | shap | 0.522 | yes |
| drop_actionability | actionability | shap | 0.549 | yes |

## Metric Definition Sensitivity

- CSS baseline leader under observed scoring: `dice`.
- CSS alternative definitions preserving the same leader: `4` / `4`.
- ALF baseline leader under observed scoring: `wachter`.
- ALF alternative definitions preserving the same leader: `6` / `6`.

| Method | Scenario | Leader | Score | Leader Changed |
| --- | --- | --- | ---: | --- |
| CSS | observed_css | dice | 0.893 | no |
| CSS | reconstructed_css | dice | 0.893 | no |
| CSS | verifiability_focused_css | dice | 0.915 | no |
| CSS | challenge_focused_css | dice | 0.895 | no |
| CSS | specificity_light_css | dice | 0.921 | no |
| ALF | observed_alf | wachter | 0.509 | no |
| ALF | balanced_arithmetic_alf | wachter | 0.813 | no |
| ALF | legality_dominant_alf | wachter | 0.892 | no |
| ALF | feasibility_dominant_alf | wachter | 0.817 | no |
| ALF | minimality_dominant_alf | wachter | 0.778 | no |
| ALF | invalidity_penalized_alf | wachter | 0.813 | no |
| ALF | hard_failure_penalized_alf | wachter | 0.813 | no |

## Track Policy Sensitivity

- Largest average absolute primary-vs-audit shift: `stability` = `0.068`.
- Most track-sensitive explainer: `anchors` with overall mean absolute delta `0.049`.
- Most track-robust explainer: `lime` with overall mean absolute delta `0.016`.
- Least leader-stable metric across primary/audit: `css` with same-leader rate `0.750`.

| Explainer | Matched Pairs | Overall Mean Abs Delta | Abs Faithfulness | Abs Stability | Abs Comprehensibility | Abs CSS | Abs Actionability | Abs ALF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lime | 12 | 0.016 | 0.019 | 0.031 | 0.004 | 0.019 | 0.006 | n/a |
| shap | 12 | 0.018 | 0.018 | 0.026 | 0.009 | 0.022 | 0.014 | n/a |
| wachter | 12 | 0.028 | 0.011 | 0.066 | 0.010 | 0.016 | 0.028 | 0.036 |
| dice | 12 | 0.030 | 0.011 | 0.094 | 0.010 | 0.016 | 0.023 | 0.025 |
| anchors | 12 | 0.049 | 0.049 | 0.123 | 0.018 | 0.028 | 0.027 | n/a |

| Metric | Same-Leader Rate | Same-Leader Count | Comparison Count |
| --- | ---: | ---: | ---: |
| faithfulness | 0.833 | 10 | 12 |
| stability | 0.917 | 11 | 12 |
| comprehensibility | 1.000 | 12 | 12 |
| css | 0.750 | 9 | 12 |
| actionability | 0.833 | 10 | 12 |
| alf | 0.833 | 10 | 12 |

## Leave-One-Dataset-Out

| Omitted Dataset | Metric | Leader | Score |
| --- | --- | --- | ---: |
| default_of_credit_card_clients | faithfulness | shap | 0.912 |
| default_of_credit_card_clients | stability | lime | 0.929 |
| default_of_credit_card_clients | comprehensibility | anchors | 0.926 |
| default_of_credit_card_clients | css | dice | 0.863 |
| default_of_credit_card_clients | actionability | wachter | 0.661 |
| default_of_credit_card_clients | alf | wachter | 0.502 |
| give_me_some_credit | faithfulness | shap | 0.892 |
| give_me_some_credit | stability | lime | 0.926 |
| give_me_some_credit | comprehensibility | anchors | 0.946 |
| give_me_some_credit | css | dice | 0.894 |
| give_me_some_credit | actionability | wachter | 0.689 |
| give_me_some_credit | alf | wachter | 0.523 |
| heloc | faithfulness | shap | 0.925 |
| heloc | stability | shap | 0.901 |
| heloc | comprehensibility | anchors | 0.950 |
| heloc | css | dice | 0.898 |
| heloc | actionability | wachter | 0.679 |
| heloc | alf | wachter | 0.499 |
| south_german_credit | faithfulness | shap | 0.924 |
| south_german_credit | stability | lime | 0.919 |
| south_german_credit | comprehensibility | anchors | 0.948 |
| south_german_credit | css | dice | 0.916 |
| south_german_credit | actionability | wachter | 0.702 |
| south_german_credit | alf | wachter | 0.510 |

## Subpopulation Slices

| Dataset | Track | Feature | Value | Explainer | Instances | Mean Stability | Mean CSS | Mean Actionability | Mean ALF |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| default_of_credit_card_clients | audit | age_band | 30-49 | anchors | 14 | 0.612 | 0.891 | 0.332 | n/a |
| default_of_credit_card_clients | audit | age_band | 30-49 | dice | 14 | 0.392 | 0.988 | 0.745 | 0.509 |
| default_of_credit_card_clients | audit | age_band | 30-49 | lime | 14 | 0.827 | 0.871 | 0.338 | n/a |
| default_of_credit_card_clients | audit | age_band | 30-49 | shap | 14 | 0.807 | 0.883 | 0.339 | n/a |
| default_of_credit_card_clients | audit | age_band | 30-49 | wachter | 14 | 0.393 | 0.988 | 0.747 | 0.513 |
| default_of_credit_card_clients | audit | age_band | 50+ | anchors | 4 | 0.470 | 0.842 | 0.287 | n/a |
| default_of_credit_card_clients | audit | age_band | 50+ | dice | 4 | 0.433 | 0.964 | 0.730 | 0.523 |
| default_of_credit_card_clients | audit | age_band | 50+ | lime | 4 | 0.904 | 0.887 | 0.343 | n/a |
| default_of_credit_card_clients | audit | age_band | 50+ | shap | 4 | 0.926 | 0.869 | 0.315 | n/a |
| default_of_credit_card_clients | audit | age_band | 50+ | wachter | 4 | 0.646 | 0.964 | 0.732 | 0.527 |
| default_of_credit_card_clients | audit | age_band | <30 | anchors | 6 | 0.644 | 0.893 | 0.333 | n/a |
| default_of_credit_card_clients | audit | age_band | <30 | dice | 6 | 0.364 | 0.974 | 0.744 | 0.532 |
| default_of_credit_card_clients | audit | age_band | <30 | lime | 6 | 0.880 | 0.862 | 0.334 | n/a |
| default_of_credit_card_clients | audit | age_band | <30 | shap | 6 | 0.865 | 0.891 | 0.338 | n/a |
| default_of_credit_card_clients | audit | age_band | <30 | wachter | 6 | 0.292 | 0.974 | 0.747 | 0.539 |
| default_of_credit_card_clients | audit | education_group | graduate_school | anchors | 10 | 0.597 | 0.883 | 0.325 | n/a |
| default_of_credit_card_clients | audit | education_group | graduate_school | dice | 10 | 0.394 | 0.979 | 0.732 | 0.501 |
| default_of_credit_card_clients | audit | education_group | graduate_school | lime | 10 | 0.790 | 0.867 | 0.335 | n/a |
| default_of_credit_card_clients | audit | education_group | graduate_school | shap | 10 | 0.798 | 0.887 | 0.332 | n/a |
| default_of_credit_card_clients | audit | education_group | graduate_school | wachter | 10 | 0.348 | 0.979 | 0.736 | 0.509 |
| default_of_credit_card_clients | audit | education_group | high_school | anchors | 4 | 0.559 | 0.885 | 0.325 | n/a |
| default_of_credit_card_clients | audit | education_group | high_school | dice | 4 | 0.582 | 0.967 | 0.743 | 0.533 |
| default_of_credit_card_clients | audit | education_group | high_school | lime | 4 | 0.899 | 0.878 | 0.340 | n/a |
| default_of_credit_card_clients | audit | education_group | high_school | shap | 4 | 0.899 | 0.871 | 0.330 | n/a |
| default_of_credit_card_clients | audit | education_group | high_school | wachter | 4 | 0.787 | 0.967 | 0.746 | 0.538 |
| default_of_credit_card_clients | audit | education_group | university | anchors | 10 | 0.610 | 0.883 | 0.325 | n/a |
| default_of_credit_card_clients | audit | education_group | university | dice | 10 | 0.314 | 0.987 | 0.752 | 0.527 |
| default_of_credit_card_clients | audit | education_group | university | lime | 10 | 0.897 | 0.874 | 0.339 | n/a |
| default_of_credit_card_clients | audit | education_group | university | shap | 10 | 0.863 | 0.883 | 0.339 | n/a |
| default_of_credit_card_clients | audit | education_group | university | wachter | 10 | 0.320 | 0.987 | 0.753 | 0.528 |

## Dataset and Track Winners

| Dataset | Track | Best Faithfulness | Best Stability | Best Comprehensibility | Best CSS | Best Actionability | Best ALF |
| --- | --- | --- | --- | --- | --- | --- | --- |
| default_of_credit_card_clients | audit | dice (0.936) | lime (0.853) | anchors (0.988) | dice (0.980) | wachter (0.745) | wachter (0.522) |
| default_of_credit_card_clients | primary | anchors (0.944) | lime (0.921) | anchors (0.994) | dice (0.984) | wachter (0.753) | wachter (0.535) |
| give_me_some_credit | audit | shap (0.970) | lime (0.895) | anchors (0.935) | dice (0.877) | wachter (0.653) | wachter (0.471) |
| give_me_some_credit | primary | shap (0.981) | lime (0.895) | anchors (0.929) | dice (0.901) | wachter (0.677) | wachter (0.459) |
| heloc | audit | shap (0.877) | lime (0.989) | anchors (0.911) | dice (0.869) | wachter (0.700) | wachter (0.554) |
| heloc | primary | shap (0.880) | lime (0.966) | anchors (0.931) | dice (0.887) | wachter (0.686) | wachter (0.518) |
| south_german_credit | audit | shap (0.875) | shap (0.972) | anchors (0.911) | dice (0.821) | wachter (0.624) | wachter (0.508) |
| south_german_credit | primary | shap (0.888) | shap (0.985) | anchors (0.940) | dice (0.825) | wachter (0.624) | wachter (0.502) |

## Pairwise Win Leaders

- `actionability`: `wachter` leads with `92` wins across `96` comparable pairwise configuration matchups.
- `alf`: `wachter` leads with `20` wins across `24` comparable pairwise configuration matchups.
- `comprehensibility`: `anchors` leads with `96` wins across `96` comparable pairwise configuration matchups.
- `css`: `dice` leads with `73` wins across `96` comparable pairwise configuration matchups.
- `faithfulness`: `shap` leads with `80` wins across `96` comparable pairwise configuration matchups.
- `stability`: `lime` leads with `86` wins across `96` comparable pairwise configuration matchups.

## Best Configurations

- Best faithfulness configuration: `give_me_some_credit` / `primary` / `logistic_regression` / `shap` with `mean_faithfulness=0.997`
- Best stability configuration: `heloc` / `audit` / `logistic_regression` / `lime` with `mean_stability=0.997`
- Best comprehensibility configuration: `default_of_credit_card_clients` / `primary` / `logistic_regression` / `anchors` with `mean_comprehensibility=1.000`
- Best CSS configuration: `default_of_credit_card_clients` / `primary` / `logistic_regression` / `dice` with `mean_css=0.992`
- Best actionability configuration: `heloc` / `audit` / `xgboost` / `wachter` with `mean_actionability=0.768`
- Best ALF configuration: `heloc` / `audit` / `xgboost` / `wachter` with `mean_alf=0.662`
