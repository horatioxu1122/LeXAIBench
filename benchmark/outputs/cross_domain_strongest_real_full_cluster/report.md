# LegalBench-XAI Benchmark Report

- Experiment: `legalbench_xai_cross_domain_strongest_real_template`
- Generated at (UTC): `2026-04-26T23:43:09.734784+00:00`
- Total groups: `90`
- Datasets: `fda_inspections, hmda, ny_courts_pretrial`
- Tracks: `audit, primary`
- Models: `ebm, logistic_regression, xgboost`
- Explainers: `anchors, dice, lime, shap, wachter`

## Data Provenance

- Data source kinds: `local_override` x 90
- Evaluation split strategies: `heldout_stratified` x 90
- Stability evidence sources: `perturbation_probe` x 1080

## Backend Usage

- Model implementations: `ebm_classifier` x 30, `sklearn_logistic_regression` x 30, `xgboost_classifier` x 30
- Explainer implementations: `lime_library` x 18, `search_anchor_rule` x 18, `search_counterfactual_dice_style` x 18, `search_counterfactual_wachter_style` x 18, `shap_library` x 18

## Held-Out Model Performance

| Dataset | Track | Model | Train Rows | Test Rows | Threshold Mode | Threshold | Accuracy | ROC AUC | PR AUC | Brier | ECE | True Adverse Rate | Predicted Adverse Rate | Split |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| fda_inspections | audit | ebm | 231982 | 99421 | fixed | 0.500 | 0.870 | 0.914 | 0.884 | 0.099 | 0.012 | 0.353 | 0.306 | heldout_stratified |
| fda_inspections | audit | logistic_regression | 231982 | 99421 | fixed | 0.500 | 0.871 | 0.907 | 0.875 | 0.101 | 0.012 | 0.353 | 0.307 | heldout_stratified |
| fda_inspections | audit | xgboost | 231982 | 99421 | fixed | 0.500 | 0.872 | 0.916 | 0.886 | 0.098 | 0.025 | 0.353 | 0.309 | heldout_stratified |
| fda_inspections | primary | ebm | 231982 | 99421 | fixed | 0.500 | 0.862 | 0.894 | 0.871 | 0.102 | 0.011 | 0.353 | 0.345 | heldout_stratified |
| fda_inspections | primary | logistic_regression | 231982 | 99421 | fixed | 0.500 | 0.862 | 0.886 | 0.864 | 0.105 | 0.035 | 0.353 | 0.345 | heldout_stratified |
| fda_inspections | primary | xgboost | 231982 | 99421 | fixed | 0.500 | 0.861 | 0.895 | 0.871 | 0.103 | 0.018 | 0.353 | 0.287 | heldout_stratified |
| hmda | audit | ebm | 6186360 | 2651297 | fixed | 0.500 | 0.840 | 0.847 | 0.717 | 0.118 | 0.007 | 0.243 | 0.146 | heldout_stratified |
| hmda | audit | logistic_regression | 6186360 | 2651297 | fixed | 0.500 | 0.834 | 0.828 | 0.692 | 0.124 | 0.006 | 0.243 | 0.135 | heldout_stratified |
| hmda | audit | xgboost | 6186360 | 2651297 | fixed | 0.500 | 0.837 | 0.843 | 0.715 | 0.120 | 0.032 | 0.243 | 0.123 | heldout_stratified |
| hmda | primary | ebm | 6186360 | 2651297 | fixed | 0.500 | 0.838 | 0.838 | 0.707 | 0.120 | 0.010 | 0.243 | 0.143 | heldout_stratified |
| hmda | primary | logistic_regression | 6186360 | 2651297 | fixed | 0.500 | 0.827 | 0.815 | 0.675 | 0.127 | 0.015 | 0.243 | 0.132 | heldout_stratified |
| hmda | primary | xgboost | 6186360 | 2651297 | fixed | 0.500 | 0.837 | 0.839 | 0.713 | 0.120 | 0.026 | 0.243 | 0.130 | heldout_stratified |
| ny_courts_pretrial | audit | ebm | 801656 | 343566 | fixed | 0.500 | 0.849 | 0.866 | 0.686 | 0.109 | 0.002 | 0.216 | 0.144 | heldout_stratified |
| ny_courts_pretrial | audit | logistic_regression | 801656 | 343566 | fixed | 0.500 | 0.847 | 0.862 | 0.679 | 0.111 | 0.002 | 0.216 | 0.145 | heldout_stratified |
| ny_courts_pretrial | audit | xgboost | 801656 | 343566 | fixed | 0.500 | 0.837 | 0.845 | 0.645 | 0.119 | 0.031 | 0.216 | 0.109 | heldout_stratified |
| ny_courts_pretrial | primary | ebm | 801656 | 343566 | fixed | 0.500 | 0.845 | 0.856 | 0.669 | 0.113 | 0.003 | 0.216 | 0.139 | heldout_stratified |
| ny_courts_pretrial | primary | logistic_regression | 801656 | 343566 | fixed | 0.500 | 0.844 | 0.853 | 0.663 | 0.114 | 0.003 | 0.216 | 0.142 | heldout_stratified |
| ny_courts_pretrial | primary | xgboost | 801656 | 343566 | fixed | 0.500 | 0.836 | 0.840 | 0.636 | 0.120 | 0.028 | 0.216 | 0.112 | heldout_stratified |

## Explainer Leaderboard

| Explainer | Groups | Instances | Mean Faithfulness | Mean Stability | Mean Comprehensibility | Mean CSS | Mean Actionability | Mean ALF | Mean Invalid Recourse Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| anchors | 18 | 216 | 0.842 | 0.661 | 0.942 | 0.733 | 0.196 | n/a | n/a |
| dice | 18 | 216 | 0.821 | 0.672 | 0.725 | 0.796 | 0.468 | 0.259 | 0.222 |
| lime | 18 | 216 | 0.886 | 0.751 | 0.742 | 0.694 | 0.195 | n/a | n/a |
| shap | 18 | 216 | 0.961 | 0.936 | 0.735 | 0.698 | 0.181 | n/a | n/a |
| wachter | 18 | 216 | 0.803 | 0.744 | 0.726 | 0.784 | 0.469 | 0.329 | 0.079 |

## Bootstrap Uncertainty

- `95%` hierarchical-bootstrap confidence intervals with `1000` resamples over `dataset_id -> track -> model_id` benchmark clusters.

| Explainer | Faithfulness CI | Stability CI | Comprehensibility CI | CSS CI | Actionability CI | ALF CI |
| --- | --- | --- | --- | --- | --- | --- |
| anchors | [0.756, 0.944] | [0.510, 0.841] | [0.921, 0.969] | [0.615, 0.856] | [0.077, 0.319] | n/a |
| dice | [0.808, 0.835] | [0.477, 0.910] | [0.682, 0.805] | [0.677, 0.910] | [0.306, 0.566] | [0.194, 0.378] |
| lime | [0.844, 0.919] | [0.590, 0.929] | [0.686, 0.846] | [0.567, 0.831] | [0.082, 0.319] | n/a |
| shap | [0.930, 0.990] | [0.881, 0.979] | [0.681, 0.841] | [0.570, 0.814] | [0.064, 0.283] | n/a |
| wachter | [0.770, 0.829] | [0.448, 0.944] | [0.681, 0.834] | [0.677, 0.910] | [0.305, 0.573] | [0.192, 0.591] |

## Evidence-Source Transparency

| Explainer | Instances | Direct Faithfulness Share | Proxy Faithfulness Share | Direct Stability Share | Stability Fallback Share |
| --- | ---: | ---: | ---: | ---: | ---: |
| anchors | 216 | 1.000 | 0.000 | 1.000 | 0.000 |
| dice | 216 | 0.000 | 1.000 | 1.000 | 0.000 |
| lime | 216 | 1.000 | 0.000 | 1.000 | 0.000 |
| shap | 216 | 1.000 | 0.000 | 1.000 | 0.000 |
| wachter | 216 | 0.000 | 1.000 | 1.000 | 0.000 |

## Metric Sensitivity

- Strongest positive rank correlation: `actionability` vs `css` = `0.800`
- Weakest rank correlation: `actionability` vs `faithfulness` = `-1.000`

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

- Pairs with fewer than three shared explainers are descriptive only and should not be over-interpreted.

## Group-Level Metric Correlation

- Strongest positive group-level Spearman correlation: `actionability` vs `css` = `0.788` over `90` groups.
- Weakest group-level Spearman correlation: `comprehensibility` vs `css` = `-0.659` over `90` groups.

| Metric A | Metric B | Spearman Rank Correlation | Shared Groups |
| --- | --- | ---: | ---: |
| actionability | alf | 0.474 | 36 |
| actionability | comprehensibility | -0.571 | 90 |
| actionability | css | 0.788 | 90 |
| actionability | faithfulness | -0.631 | 90 |
| actionability | stability | -0.457 | 90 |
| alf | comprehensibility | -0.035 | 36 |
| alf | css | 0.122 | 36 |
| alf | faithfulness | -0.429 | 36 |
| alf | stability | 0.072 | 36 |
| comprehensibility | css | -0.659 | 90 |
| comprehensibility | faithfulness | 0.075 | 90 |
| comprehensibility | stability | 0.234 | 90 |
| css | faithfulness | -0.451 | 90 |
| css | stability | -0.578 | 90 |
| faithfulness | stability | 0.473 | 90 |

## Weight Sensitivity

- Randomized universal-weight sensitivity: `shap` wins `0.398` of draws with mean rank `2.420`.
- Randomized universal-weight sensitivity: `dice` wins `0.245` of draws with mean rank `2.548`.
- Randomized universal-weight sensitivity: `wachter` wins `0.183` of draws with mean rank `2.486`.
- Randomized universal-weight sensitivity: `anchors` wins `0.174` of draws with mean rank `3.168`.
- Randomized universal-weight sensitivity: `lime` wins `0.000` of draws with mean rank `4.378`.

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

## Proxy Sensitivity

| Scenario | Explainer | Adjusted Faithfulness | Proxy Share | Composite Score | Rank |
| --- | --- | ---: | ---: | ---: | ---: |
| observed | wachter | 0.803 | 1.000 | 0.437 | 1 |
| observed | dice | 0.821 | 1.000 | 0.429 | 2 |
| observed | shap | 0.961 | 0.000 | 0.417 | 3 |
| observed | anchors | 0.842 | 0.000 | 0.336 | 4 |
| observed | lime | 0.886 | 0.000 | 0.196 | 5 |
| proxy_discount_25 | wachter | 0.603 | 1.000 | 0.437 | 1 |
| proxy_discount_25 | anchors | 0.842 | 0.000 | 0.421 | 2 |
| proxy_discount_25 | shap | 0.961 | 0.000 | 0.417 | 3 |
| proxy_discount_25 | dice | 0.616 | 1.000 | 0.414 | 4 |
| proxy_discount_25 | lime | 0.886 | 0.000 | 0.249 | 5 |
| proxy_cap_050 | wachter | 0.500 | 1.000 | 0.437 | 1 |
| proxy_cap_050 | anchors | 0.842 | 0.000 | 0.436 | 2 |
| proxy_cap_050 | shap | 0.961 | 0.000 | 0.417 | 3 |
| proxy_cap_050 | dice | 0.500 | 1.000 | 0.407 | 4 |
| proxy_cap_050 | lime | 0.886 | 0.000 | 0.259 | 5 |

## Direct-Evidence-Only Sanity Check

- Included groups with direct faithfulness evidence only: `54` across explainers `anchors, lime, shap`.
- Direct-evidence faithfulness leader: `shap` with mean faithfulness `0.961`.
- Balanced direct-evidence composite leader: `anchors` with composite score `0.600`.
- Direct-evidence pairwise faithfulness champion: `shap` with `30` matched-group wins.

| Explainer | Groups | Instances | Faithfulness | Stability | Comprehensibility | CSS | Actionability |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| anchors | 18 | 216 | 0.842 | 0.661 | 0.942 | 0.733 | 0.196 |
| lime | 18 | 216 | 0.886 | 0.751 | 0.742 | 0.694 | 0.195 |
| shap | 18 | 216 | 0.961 | 0.936 | 0.735 | 0.698 | 0.181 |

| Scenario | Scope | Explainer | Composite Score | Rank |
| --- | --- | --- | ---: | ---: |
| balanced | all_explainers | anchors | 0.600 | 1 |
| balanced | all_explainers | shap | 0.420 | 2 |
| balanced | all_explainers | lime | 0.335 | 3 |
| reliability_focused | all_explainers | shap | 0.805 | 1 |
| reliability_focused | all_explainers | lime | 0.333 | 2 |
| reliability_focused | all_explainers | anchors | 0.200 | 3 |
| applicant_facing | all_explainers | anchors | 0.900 | 1 |
| applicant_facing | all_explainers | lime | 0.326 | 2 |
| applicant_facing | all_explainers | shap | 0.134 | 3 |
| contestability_focused | all_explainers | anchors | 0.900 | 1 |
| contestability_focused | all_explainers | lime | 0.275 | 2 |
| contestability_focused | all_explainers | shap | 0.149 | 3 |

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

## Metric Definition Sensitivity

- CSS baseline leader under observed scoring: `dice`.
- CSS alternative definitions preserving the same leader: `4` / `4`.
- ALF baseline leader under observed scoring: `wachter`.
- ALF alternative definitions preserving the same leader: `6` / `6`.

| Family | Scenario | Leader | Score | Leader Changed |
| --- | --- | --- | ---: | --- |
| CSS | observed_css | dice | 0.796 | no |
| CSS | reconstructed_css | dice | 0.796 | no |
| CSS | verifiability_focused_css | dice | 0.837 | no |
| CSS | challenge_focused_css | dice | 0.810 | no |
| CSS | specificity_light_css | dice | 0.827 | no |
| ALF | observed_alf | wachter | 0.329 | no |
| ALF | balanced_arithmetic_alf | wachter | 0.727 | no |
| ALF | legality_dominant_alf | wachter | 0.818 | no |
| ALF | feasibility_dominant_alf | wachter | 0.669 | no |
| ALF | minimality_dominant_alf | wachter | 0.730 | no |
| ALF | invalidity_penalized_alf | wachter | 0.669 | no |
| ALF | hard_failure_penalized_alf | wachter | 0.669 | no |

## Track Policy Sensitivity

- Largest average absolute primary-vs-audit shift: `stability` = `0.114`.
- Most track-sensitive explainer: `anchors` with overall mean absolute delta `0.072`.
- Most track-robust explainer: `shap` with overall mean absolute delta `0.035`.
- Least leader-stable metric across primary/audit: `faithfulness` with same-leader rate `0.556`.

| Explainer | Matched Pairs | Overall Mean Abs Delta | Abs Faithfulness | Abs Stability | Abs Comprehensibility | Abs CSS | Abs Actionability | Abs ALF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| shap | 9 | 0.035 | 0.027 | 0.018 | 0.098 | 0.016 | 0.014 | n/a |
| dice | 9 | 0.052 | 0.016 | 0.145 | 0.075 | 0.007 | 0.027 | 0.039 |
| wachter | 9 | 0.052 | 0.029 | 0.132 | 0.074 | 0.018 | 0.048 | 0.012 |
| lime | 9 | 0.053 | 0.034 | 0.066 | 0.099 | 0.034 | 0.032 | n/a |
| anchors | 9 | 0.072 | 0.076 | 0.209 | 0.016 | 0.028 | 0.028 | n/a |

| Metric | Same-Leader Rate | Same-Leader Count | Comparison Count |
| --- | ---: | ---: | ---: |
| faithfulness | 0.556 | 5 | 9 |
| stability | 0.556 | 5 | 9 |
| comprehensibility | 0.667 | 6 | 9 |
| css | 1.000 | 9 | 9 |
| actionability | 0.667 | 6 | 9 |
| alf | 1.000 | 9 | 9 |

## Leave-One-Dataset-Out

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

## Dataset and Track Winners

| Dataset | Track | Best Faithfulness | Best Stability | Best Comprehensibility | Best CSS | Best Actionability | Best ALF |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fda_inspections | audit | shap (0.990) | shap (0.984) | anchors (0.968) | dice (0.678) | dice (0.311) | dice (0.192) |
| fda_inspections | primary | anchors (0.999) | anchors (0.988) | lime (0.988) | dice (0.675) | dice (0.299) | dice (0.199) |
| hmda | audit | shap (0.932) | shap (0.888) | anchors (0.936) | dice (0.799) | dice (0.510) | wachter (0.605) |
| hmda | primary | shap (0.981) | wachter (0.938) | anchors (0.913) | dice (0.804) | wachter (0.605) | wachter (0.585) |
| ny_courts_pretrial | audit | shap (0.919) | shap (0.946) | anchors (0.939) | dice (0.907) | dice (0.555) | dice (0.199) |
| ny_courts_pretrial | primary | shap (0.951) | shap (0.957) | anchors (0.929) | dice (0.916) | dice (0.561) | dice (0.199) |

## Pairwise Win Leaders

- `actionability`: `dice` leads with `60` wins across `72` comparable pairwise group matchups.
- `alf`: `wachter` leads with `6` wins across `18` comparable pairwise group matchups.
- `comprehensibility`: `anchors` leads with `66` wins across `72` comparable pairwise group matchups.
- `css`: `dice` leads with `60` wins across `72` comparable pairwise group matchups.
- `faithfulness`: `shap` leads with `66` wins across `72` comparable pairwise group matchups.
- `stability`: `shap` leads with `61` wins across `72` comparable pairwise group matchups.

## Best Groups

- Best faithfulness group: `fda_inspections` / `primary` / `logistic_regression` / `shap` with `mean_faithfulness=1.000`
- Best stability group: `fda_inspections` / `primary` / `logistic_regression` / `anchors` with `mean_stability=1.000`
- Best comprehensibility group: `fda_inspections` / `primary` / `xgboost` / `lime` with `mean_comprehensibility=1.000`
- Best CSS group: `ny_courts_pretrial` / `primary` / `logistic_regression` / `dice` with `mean_css=0.929`
- Best actionability group: `hmda` / `primary` / `xgboost` / `dice` with `mean_actionability=0.623`
- Best ALF group: `hmda` / `primary` / `xgboost` / `wachter` with `mean_alf=0.605`
