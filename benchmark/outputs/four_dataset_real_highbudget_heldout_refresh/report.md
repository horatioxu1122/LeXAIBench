# LegalBench-XAI Benchmark Report

- Experiment: `legalbench_xai_four_dataset_real_highbudget`
- Generated at (UTC): `2026-03-15T22:29:00.007268+00:00`
- Total configurations: `120`
- Datasets: `default_of_credit_card_clients, give_me_some_credit, heloc, south_german_credit`
- Tracks: `audit, primary`
- Models: `ebm, logistic_regression, xgboost`
- Explainers: `anchors, dice, lime, shap, wachter`

## Data Provenance

- Data source kinds: `local_override` x 120
- Evaluation split strategies: `heldout_stratified` x 120
- Stability evidence sources: `perturbation_probe` x 1438, `perturbation_probe_insufficient_neighbors` x 2

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

| Explainer | Configurations | Instances | Mean Faithfulness | Mean Stability | Mean Comprehensibility | Mean CSS | Mean Actionability | Mean ALF | Mean Invalid Recourse Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| anchors | 24 | 288 | 0.797 | 0.630 | 0.944 | 0.828 | 0.286 | n/a | n/a |
| dice | 24 | 288 | 0.840 | 0.479 | 0.681 | 0.896 | 0.667 | 0.468 | 0.101 |
| lime | 24 | 288 | 0.839 | 0.917 | 0.722 | 0.780 | 0.275 | n/a | n/a |
| shap | 24 | 288 | 0.908 | 0.910 | 0.711 | 0.797 | 0.276 | n/a | n/a |
| wachter | 24 | 288 | 0.826 | 0.552 | 0.681 | 0.892 | 0.674 | 0.482 | 0.009 |

## Bootstrap Uncertainty

- `95%` hierarchical-bootstrap confidence intervals with `1000` resamples over `dataset_id -> track -> model_id` benchmark clusters.

| Explainer | Faithfulness CI | Stability CI | Comprehensibility CI | CSS CI | Actionability CI | ALF CI |
| --- | --- | --- | --- | --- | --- | --- |
| anchors | [0.717, 0.882] | [0.587, 0.679] | [0.918, 0.974] | [0.770, 0.877] | [0.234, 0.325] | n/a |
| dice | [0.806, 0.892] | [0.339, 0.613] | [0.655, 0.708] | [0.839, 0.959] | [0.604, 0.725] | [0.429, 0.508] |
| lime | [0.798, 0.887] | [0.882, 0.956] | [0.670, 0.806] | [0.699, 0.858] | [0.219, 0.331] | n/a |
| shap | [0.866, 0.959] | [0.869, 0.955] | [0.668, 0.780] | [0.726, 0.866] | [0.222, 0.327] | n/a |
| wachter | [0.782, 0.889] | [0.349, 0.751] | [0.658, 0.709] | [0.834, 0.958] | [0.628, 0.725] | [0.448, 0.514] |

## Evidence-Source Transparency

| Explainer | Instances | Direct Faithfulness Share | Proxy Faithfulness Share | Direct Stability Share | Stability Fallback Share |
| --- | ---: | ---: | ---: | ---: | ---: |
| anchors | 288 | 1.000 | 0.000 | 0.997 | 0.003 |
| dice | 288 | 0.000 | 1.000 | 1.000 | 0.000 |
| lime | 288 | 1.000 | 0.000 | 1.000 | 0.000 |
| shap | 288 | 1.000 | 0.000 | 1.000 | 0.000 |
| wachter | 288 | 0.000 | 1.000 | 0.997 | 0.003 |

## Metric Sensitivity

- Strongest positive rank correlation: `actionability` vs `css` = `0.900`
- Weakest rank correlation: `css` vs `stability` = `-1.000`

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

- Pairs with fewer than three shared explainers are descriptive only and should not be over-interpreted.

## Group-Level Metric Correlation

- Strongest positive group-level Spearman correlation: `actionability` vs `css` = `0.847` over `120` groups.
- Weakest group-level Spearman correlation: `actionability` vs `stability` = `-0.543` over `120` groups.

| Metric A | Metric B | Spearman Rank Correlation | Shared Configurations |
| --- | --- | ---: | ---: |
| actionability | alf | 0.708 | 48 |
| actionability | comprehensibility | -0.290 | 120 |
| actionability | css | 0.847 | 120 |
| actionability | faithfulness | 0.001 | 120 |
| actionability | stability | -0.543 | 120 |
| alf | comprehensibility | 0.399 | 48 |
| alf | css | 0.489 | 48 |
| alf | faithfulness | 0.334 | 48 |
| alf | stability | -0.215 | 48 |
| comprehensibility | css | 0.012 | 120 |
| comprehensibility | faithfulness | 0.058 | 120 |
| comprehensibility | stability | 0.023 | 120 |
| css | faithfulness | 0.233 | 120 |
| css | stability | -0.447 | 120 |
| faithfulness | stability | 0.170 | 120 |

## Weight Sensitivity

- Randomized universal-weight sensitivity: `shap` wins `0.396` of draws with mean rank `2.364`.
- Randomized universal-weight sensitivity: `dice` wins `0.251` of draws with mean rank `2.614`.
- Randomized universal-weight sensitivity: `wachter` wins `0.181` of draws with mean rank `2.617`.
- Randomized universal-weight sensitivity: `anchors` wins `0.163` of draws with mean rank `3.475`.
- Randomized universal-weight sensitivity: `lime` wins `0.009` of draws with mean rank `3.930`.

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

## Proxy Sensitivity

| Scenario | Explainer | Adjusted Faithfulness | Proxy Share | Composite Score | Rank |
| --- | --- | ---: | ---: | ---: | ---: |
| observed | wachter | 0.826 | 1.000 | 0.477 | 1 |
| observed | dice | 0.840 | 1.000 | 0.474 | 2 |
| observed | shap | 0.908 | 0.000 | 0.449 | 3 |
| observed | anchors | 0.797 | 0.000 | 0.357 | 4 |
| observed | lime | 0.839 | 0.000 | 0.307 | 5 |
| proxy_discount_25 | anchors | 0.797 | 0.000 | 0.481 | 1 |
| proxy_discount_25 | shap | 0.908 | 0.000 | 0.449 | 2 |
| proxy_discount_25 | wachter | 0.619 | 1.000 | 0.426 | 3 |
| proxy_discount_25 | dice | 0.630 | 1.000 | 0.404 | 4 |
| proxy_discount_25 | lime | 0.839 | 0.000 | 0.384 | 5 |
| proxy_cap_050 | anchors | 0.797 | 0.000 | 0.503 | 1 |
| proxy_cap_050 | shap | 0.908 | 0.000 | 0.449 | 2 |
| proxy_cap_050 | wachter | 0.500 | 1.000 | 0.426 | 3 |
| proxy_cap_050 | lime | 0.839 | 0.000 | 0.398 | 4 |
| proxy_cap_050 | dice | 0.500 | 1.000 | 0.397 | 5 |

## Direct-Evidence-Only Sanity Check

- Included groups with direct faithfulness evidence only: `72` across explainers `anchors, lime, shap`.
- Direct-evidence faithfulness leader: `shap` with mean faithfulness `0.908`.
- Balanced direct-evidence composite leader: `anchors` with composite score `0.600`.
- Direct-evidence pairwise faithfulness champion: `shap` with `40` matched-configuration wins.

| Explainer | Configurations | Instances | Faithfulness | Stability | Comprehensibility | CSS | Actionability |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| anchors | 24 | 288 | 0.797 | 0.630 | 0.944 | 0.828 | 0.286 |
| lime | 24 | 288 | 0.839 | 0.917 | 0.722 | 0.780 | 0.275 |
| shap | 24 | 288 | 0.908 | 0.910 | 0.711 | 0.797 | 0.276 |

| Scenario | Scope | Explainer | Composite Score | Rank |
| --- | --- | --- | ---: | ---: |
| balanced | all_explainers | anchors | 0.600 | 1 |
| balanced | all_explainers | shap | 0.480 | 2 |
| balanced | all_explainers | lime | 0.286 | 3 |
| reliability_focused | all_explainers | shap | 0.812 | 1 |
| reliability_focused | all_explainers | lime | 0.525 | 2 |
| reliability_focused | all_explainers | anchors | 0.200 | 3 |
| applicant_facing | all_explainers | anchors | 0.900 | 1 |
| applicant_facing | all_explainers | shap | 0.245 | 2 |
| applicant_facing | all_explainers | lime | 0.081 | 3 |
| contestability_focused | all_explainers | anchors | 0.900 | 1 |
| contestability_focused | all_explainers | shap | 0.295 | 2 |
| contestability_focused | all_explainers | lime | 0.076 | 3 |

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

## Metric Definition Sensitivity

- CSS baseline leader under observed scoring: `dice`.
- CSS alternative definitions preserving the same leader: `4` / `4`.
- ALF baseline leader under observed scoring: `wachter`.
- ALF alternative definitions preserving the same leader: `6` / `6`.

| Method | Scenario | Leader | Score | Leader Changed |
| --- | --- | --- | ---: | --- |
| CSS | observed_css | dice | 0.896 | no |
| CSS | reconstructed_css | dice | 0.896 | no |
| CSS | verifiability_focused_css | dice | 0.918 | no |
| CSS | challenge_focused_css | dice | 0.896 | no |
| CSS | specificity_light_css | dice | 0.927 | no |
| ALF | observed_alf | wachter | 0.482 | no |
| ALF | balanced_arithmetic_alf | wachter | 0.799 | no |
| ALF | legality_dominant_alf | wachter | 0.883 | no |
| ALF | feasibility_dominant_alf | wachter | 0.799 | no |
| ALF | minimality_dominant_alf | wachter | 0.765 | no |
| ALF | invalidity_penalized_alf | wachter | 0.792 | no |
| ALF | hard_failure_penalized_alf | wachter | 0.791 | no |

## Track Policy Sensitivity

- Largest average absolute primary-vs-audit shift: `stability` = `0.047`.
- Most track-sensitive explainer: `anchors` with overall mean absolute delta `0.034`.
- Most track-robust explainer: `shap` with overall mean absolute delta `0.012`.
- Least leader-stable metric across primary/audit: `actionability` with same-leader rate `0.667`.

| Explainer | Matched Pairs | Overall Mean Abs Delta | Abs Faithfulness | Abs Stability | Abs Comprehensibility | Abs CSS | Abs Actionability | Abs ALF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| shap | 12 | 0.012 | 0.013 | 0.020 | 0.009 | 0.012 | 0.007 | n/a |
| lime | 12 | 0.016 | 0.025 | 0.026 | 0.003 | 0.019 | 0.007 | n/a |
| wachter | 12 | 0.018 | 0.008 | 0.045 | 0.009 | 0.006 | 0.015 | 0.024 |
| dice | 12 | 0.019 | 0.005 | 0.054 | 0.009 | 0.005 | 0.016 | 0.026 |
| anchors | 12 | 0.034 | 0.031 | 0.088 | 0.016 | 0.018 | 0.019 | n/a |

| Metric | Same-Leader Rate | Same-Leader Count | Comparison Count |
| --- | ---: | ---: | ---: |
| faithfulness | 1.000 | 12 | 12 |
| stability | 0.750 | 9 | 12 |
| comprehensibility | 1.000 | 12 | 12 |
| css | 1.000 | 12 | 12 |
| actionability | 0.667 | 8 | 12 |
| alf | 0.667 | 8 | 12 |

## Leave-One-Dataset-Out

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

## Subpopulation Slices

| Dataset | Track | Feature | Value | Explainer | Instances | Mean Stability | Mean CSS | Mean Actionability | Mean ALF |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| default_of_credit_card_clients | audit | age_band | 30-49 | anchors | 18 | 0.637 | 0.902 | 0.342 | n/a |
| default_of_credit_card_clients | audit | age_band | 30-49 | dice | 18 | 0.555 | 0.982 | 0.748 | 0.526 |
| default_of_credit_card_clients | audit | age_band | 30-49 | lime | 18 | 0.889 | 0.876 | 0.338 | n/a |
| default_of_credit_card_clients | audit | age_band | 30-49 | shap | 18 | 0.882 | 0.885 | 0.336 | n/a |
| default_of_credit_card_clients | audit | age_band | 30-49 | wachter | 18 | 0.583 | 0.982 | 0.749 | 0.527 |
| default_of_credit_card_clients | audit | age_band | 50+ | anchors | 5 | 0.601 | 0.870 | 0.310 | n/a |
| default_of_credit_card_clients | audit | age_band | 50+ | dice | 5 | 0.351 | 0.973 | 0.727 | 0.500 |
| default_of_credit_card_clients | audit | age_band | 50+ | lime | 5 | 0.891 | 0.860 | 0.335 | n/a |
| default_of_credit_card_clients | audit | age_band | 50+ | shap | 5 | 0.858 | 0.877 | 0.331 | n/a |
| default_of_credit_card_clients | audit | age_band | 50+ | wachter | 5 | 0.337 | 0.973 | 0.727 | 0.500 |
| default_of_credit_card_clients | audit | age_band | <30 | anchors | 13 | 0.662 | 0.891 | 0.331 | n/a |
| default_of_credit_card_clients | audit | age_band | <30 | dice | 13 | 0.452 | 0.980 | 0.741 | 0.518 |
| default_of_credit_card_clients | audit | age_band | <30 | lime | 13 | 0.894 | 0.872 | 0.337 | n/a |
| default_of_credit_card_clients | audit | age_band | <30 | shap | 13 | 0.855 | 0.892 | 0.338 | n/a |
| default_of_credit_card_clients | audit | age_band | <30 | wachter | 13 | 0.448 | 0.980 | 0.741 | 0.518 |
| default_of_credit_card_clients | audit | education_group | graduate_school | anchors | 9 | 0.653 | 0.899 | 0.339 | n/a |
| default_of_credit_card_clients | audit | education_group | graduate_school | dice | 9 | 0.362 | 0.977 | 0.750 | 0.537 |
| default_of_credit_card_clients | audit | education_group | graduate_school | lime | 9 | 0.840 | 0.862 | 0.333 | n/a |
| default_of_credit_card_clients | audit | education_group | graduate_school | shap | 9 | 0.823 | 0.886 | 0.340 | n/a |
| default_of_credit_card_clients | audit | education_group | graduate_school | wachter | 9 | 0.339 | 0.977 | 0.750 | 0.537 |
| default_of_credit_card_clients | audit | education_group | high_school | anchors | 9 | 0.649 | 0.888 | 0.328 | n/a |
| default_of_credit_card_clients | audit | education_group | high_school | dice | 9 | 0.547 | 0.973 | 0.741 | 0.524 |
| default_of_credit_card_clients | audit | education_group | high_school | lime | 9 | 0.904 | 0.877 | 0.340 | n/a |
| default_of_credit_card_clients | audit | education_group | high_school | shap | 9 | 0.908 | 0.880 | 0.334 | n/a |
| default_of_credit_card_clients | audit | education_group | high_school | wachter | 9 | 0.485 | 0.973 | 0.741 | 0.524 |
| default_of_credit_card_clients | audit | education_group | university | anchors | 18 | 0.631 | 0.893 | 0.333 | n/a |
| default_of_credit_card_clients | audit | education_group | university | dice | 18 | 0.524 | 0.985 | 0.739 | 0.508 |
| default_of_credit_card_clients | audit | education_group | university | lime | 18 | 0.910 | 0.875 | 0.338 | n/a |
| default_of_credit_card_clients | audit | education_group | university | shap | 18 | 0.873 | 0.889 | 0.335 | n/a |
| default_of_credit_card_clients | audit | education_group | university | wachter | 18 | 0.588 | 0.985 | 0.740 | 0.510 |

## Subpopulation Spread Summary

| Dataset | Track | Feature | Metric | Leader Set Across Subpopulations | Max-Spread Explainer | Max Spread |
| --- | --- | --- | --- | --- | --- | ---: |
| default_of_credit_card_clients | audit | age_band | actionability | dice|wachter | anchors | 0.032 |
| default_of_credit_card_clients | audit | age_band | alf | dice|wachter | wachter | 0.027 |
| default_of_credit_card_clients | audit | age_band | css | dice | anchors | 0.032 |
| default_of_credit_card_clients | audit | age_band | faithfulness | dice | wachter | 0.050 |
| default_of_credit_card_clients | audit | age_band | stability | lime | wachter | 0.246 |
| default_of_credit_card_clients | audit | education_group | actionability | dice|wachter | anchors | 0.011 |
| default_of_credit_card_clients | audit | education_group | alf | dice|wachter | dice | 0.029 |
| default_of_credit_card_clients | audit | education_group | css | dice | lime | 0.015 |
| default_of_credit_card_clients | audit | education_group | faithfulness | dice | lime | 0.044 |
| default_of_credit_card_clients | audit | education_group | stability | lime|shap | wachter | 0.249 |
| default_of_credit_card_clients | audit | sex | actionability | wachter | anchors | 0.017 |
| default_of_credit_card_clients | audit | sex | alf | wachter | wachter | 0.000 |
| default_of_credit_card_clients | audit | sex | css | dice | anchors | 0.017 |
| default_of_credit_card_clients | audit | sex | faithfulness | dice | lime | 0.029 |
| default_of_credit_card_clients | audit | sex | stability | lime | wachter | 0.180 |
| give_me_some_credit | audit | age_band | actionability | wachter | wachter | 0.025 |
| give_me_some_credit | audit | age_band | alf | wachter | dice | 0.032 |
| give_me_some_credit | audit | age_band | css | dice | anchors | 0.015 |
| give_me_some_credit | audit | age_band | faithfulness | shap | lime | 0.040 |
| give_me_some_credit | audit | age_band | stability | lime|shap | wachter | 0.372 |
| give_me_some_credit | audit | dependents_group | actionability | dice|wachter | wachter | 0.009 |
| give_me_some_credit | audit | dependents_group | alf | dice|wachter | wachter | 0.030 |
| give_me_some_credit | audit | dependents_group | css | dice | dice | 0.006 |
| give_me_some_credit | audit | dependents_group | faithfulness | shap | lime | 0.024 |
| give_me_some_credit | audit | dependents_group | stability | lime|shap | wachter | 0.176 |
| south_german_credit | audit | sex_status_group | actionability | wachter | dice | 0.035 |
| south_german_credit | audit | sex_status_group | alf | wachter | dice | 0.036 |
| south_german_credit | audit | sex_status_group | css | dice | wachter | 0.028 |
| south_german_credit | audit | sex_status_group | faithfulness | shap | anchors | 0.063 |
| south_german_credit | audit | sex_status_group | stability | shap | dice | 0.251 |

## Dataset and Track Winners

| Dataset | Track | Best Faithfulness | Best Stability | Best Comprehensibility | Best CSS | Best Actionability | Best ALF |
| --- | --- | --- | --- | --- | --- | --- | --- |
| default_of_credit_card_clients | audit | dice (0.929) | lime (0.891) | anchors (0.986) | dice (0.980) | wachter (0.743) | wachter (0.520) |
| default_of_credit_card_clients | primary | anchors (0.955) | lime (0.924) | anchors (0.992) | dice (0.989) | dice (0.752) | dice (0.526) |
| give_me_some_credit | audit | shap (0.976) | lime (0.886) | anchors (0.913) | dice (0.904) | wachter (0.659) | wachter (0.422) |
| give_me_some_credit | primary | shap (0.983) | lime (0.885) | anchors (0.916) | dice (0.907) | wachter (0.671) | wachter (0.444) |
| heloc | audit | shap (0.865) | lime (0.991) | anchors (0.939) | dice (0.878) | wachter (0.664) | wachter (0.468) |
| heloc | primary | shap (0.863) | lime (0.954) | anchors (0.947) | dice (0.880) | dice (0.694) | dice (0.518) |
| south_german_credit | audit | shap (0.868) | shap (0.977) | anchors (0.915) | dice (0.814) | wachter (0.605) | wachter (0.480) |
| south_german_credit | primary | shap (0.883) | shap (0.983) | anchors (0.940) | dice (0.817) | wachter (0.610) | wachter (0.487) |

## Pairwise Win Leaders

- `actionability`: `wachter` leads with `86` wins across `96` comparable pairwise configuration matchups.
- `alf`: `wachter` leads with `14` wins across `24` comparable pairwise configuration matchups.
- `comprehensibility`: `anchors` leads with `96` wins across `96` comparable pairwise configuration matchups.
- `css`: `dice` leads with `78` wins across `96` comparable pairwise configuration matchups.
- `faithfulness`: `shap` leads with `80` wins across `96` comparable pairwise configuration matchups.
- `stability`: `lime` leads with `85` wins across `96` comparable pairwise configuration matchups.

## Best Configurations

- Best faithfulness configuration: `give_me_some_credit` / `primary` / `logistic_regression` / `shap` with `mean_faithfulness=0.997`
- Best stability configuration: `heloc` / `audit` / `logistic_regression` / `lime` with `mean_stability=1.000`
- Best comprehensibility configuration: `default_of_credit_card_clients` / `primary` / `logistic_regression` / `anchors` with `mean_comprehensibility=1.000`
- Best CSS configuration: `default_of_credit_card_clients` / `primary` / `logistic_regression` / `dice` with `mean_css=0.996`
- Best actionability configuration: `default_of_credit_card_clients` / `primary` / `logistic_regression` / `dice` with `mean_actionability=0.757`
- Best ALF configuration: `heloc` / `primary` / `ebm` / `wachter` with `mean_alf=0.557`
