# LegalBench-XAI Benchmark Report

- Experiment: `legalbench_xai_default_credit_real_threshold040`
- Generated at (UTC): `2026-03-15T19:19:37.140762+00:00`
- Total configurations: `30`
- Datasets: `default_of_credit_card_clients`
- Tracks: `audit, primary`
- Models: `ebm, logistic_regression, xgboost`
- Explainers: `anchors, dice, lime, shap, wachter`

## Data Provenance

- Data source kinds: `local_override` x 30
- Evaluation split strategies: `heldout_stratified` x 30
- Stability evidence sources: `perturbation_probe` x 360

## Backend Usage

- Model implementations: `ebm_classifier` x 10, `sklearn_logistic_regression` x 10, `xgboost_classifier` x 10
- Explainer implementations: `lime_library` x 6, `search_anchor_rule` x 6, `search_counterfactual_dice_style` x 6, `search_counterfactual_wachter_style` x 6, `shap_library` x 6

## Held-Out Model Performance

| Dataset | Track | Model | Train Rows | Test Rows | Accuracy | ROC AUC | True Adverse Rate | Predicted Adverse Rate | Split |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| default_of_credit_card_clients | audit | ebm | 21000 | 9000 | 0.817 | 0.777 | 0.221 | 0.144 | heldout_stratified |
| default_of_credit_card_clients | audit | logistic_regression | 21000 | 9000 | 0.812 | 0.714 | 0.221 | 0.131 | heldout_stratified |
| default_of_credit_card_clients | audit | xgboost | 21000 | 9000 | 0.815 | 0.772 | 0.221 | 0.143 | heldout_stratified |
| default_of_credit_card_clients | primary | ebm | 21000 | 9000 | 0.816 | 0.774 | 0.221 | 0.145 | heldout_stratified |
| default_of_credit_card_clients | primary | logistic_regression | 21000 | 9000 | 0.811 | 0.712 | 0.221 | 0.136 | heldout_stratified |
| default_of_credit_card_clients | primary | xgboost | 21000 | 9000 | 0.815 | 0.771 | 0.221 | 0.143 | heldout_stratified |

## Explainer Leaderboard

| Explainer | Configurations | Instances | Mean Faithfulness | Mean Stability | Mean Comprehensibility | Mean CSS | Mean Actionability | Mean ALF | Mean Invalid Recourse Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| anchors | 6 | 72 | 0.935 | 0.677 | 0.987 | 0.891 | 0.331 | n/a | n/a |
| dice | 6 | 72 | 0.937 | 0.434 | 0.720 | 0.982 | 0.750 | 0.531 | 0.007 |
| lime | 6 | 72 | 0.793 | 0.916 | 0.844 | 0.873 | 0.337 | n/a | n/a |
| shap | 6 | 72 | 0.904 | 0.849 | 0.803 | 0.887 | 0.334 | n/a | n/a |
| wachter | 6 | 72 | 0.921 | 0.454 | 0.720 | 0.982 | 0.751 | 0.533 | 0.000 |

## Bootstrap Uncertainty

- `95%` hierarchical-bootstrap confidence intervals with `1000` resamples over `dataset_id -> track -> model_id` benchmark clusters.

| Explainer | Faithfulness CI | Stability CI | Comprehensibility CI | CSS CI | Actionability CI | ALF CI |
| --- | --- | --- | --- | --- | --- | --- |
| anchors | [0.920, 0.949] | [0.640, 0.705] | [0.969, 0.997] | [0.879, 0.900] | [0.319, 0.340] | n/a |
| dice | [0.898, 0.961] | [0.210, 0.649] | [0.710, 0.733] | [0.973, 0.990] | [0.739, 0.758] | [0.521, 0.545] |
| lime | [0.744, 0.841] | [0.877, 0.958] | [0.836, 0.849] | [0.855, 0.892] | [0.331, 0.345] | n/a |
| shap | [0.873, 0.940] | [0.814, 0.892] | [0.770, 0.821] | [0.874, 0.899] | [0.325, 0.341] | n/a |
| wachter | [0.856, 0.963] | [0.197, 0.738] | [0.711, 0.733] | [0.971, 0.991] | [0.740, 0.758] | [0.524, 0.545] |

## Metric Sensitivity

- Strongest positive rank correlation: `css` vs `faithfulness` = `0.821`
- Weakest rank correlation: `css` vs `stability` = `-0.975`

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

- Pairs with fewer than three shared explainers are descriptive only and should not be over-interpreted.

## Weight Sensitivity

- Randomized universal-weight sensitivity: `dice` wins `0.410` of draws with mean rank `2.223`.
- Randomized universal-weight sensitivity: `anchors` wins `0.336` of draws with mean rank `2.443`.
- Randomized universal-weight sensitivity: `wachter` wins `0.140` of draws with mean rank `2.620`.
- Randomized universal-weight sensitivity: `shap` wins `0.057` of draws with mean rank `3.461`.
- Randomized universal-weight sensitivity: `lime` wins `0.057` of draws with mean rank `4.253`.

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

## Metric Definition Sensitivity

- CSS baseline leader under observed scoring: `dice`.
- CSS alternative definitions preserving the same leader: `4` / `4`.
- ALF baseline leader under observed scoring: `wachter`.
- ALF alternative definitions preserving the same leader: `4` / `6`.

| Method | Scenario | Leader | Score | Leader Changed |
| --- | --- | --- | ---: | --- |
| CSS | observed_css | dice | 0.982 | no |
| CSS | reconstructed_css | dice | 0.982 | no |
| CSS | verifiability_focused_css | dice | 0.986 | no |
| CSS | challenge_focused_css | dice | 0.984 | no |
| CSS | specificity_light_css | dice | 0.983 | no |
| ALF | observed_alf | wachter | 0.533 | no |
| ALF | balanced_arithmetic_alf | wachter | 0.825 | no |
| ALF | legality_dominant_alf | dice | 0.897 | yes |
| ALF | feasibility_dominant_alf | dice | 0.824 | yes |
| ALF | minimality_dominant_alf | wachter | 0.800 | no |
| ALF | invalidity_penalized_alf | wachter | 0.825 | no |
| ALF | hard_failure_penalized_alf | wachter | 0.825 | no |

## Track Policy Sensitivity

- Largest average absolute primary-vs-audit shift: `stability` = `0.056`.
- Most track-sensitive explainer: `dice` with overall mean absolute delta `0.030`.
- Most track-robust explainer: `lime` with overall mean absolute delta `0.013`.
- Least leader-stable metric across primary/audit: `faithfulness` with same-leader rate `0.333`.

| Explainer | Matched Pairs | Overall Mean Abs Delta | Abs Faithfulness | Abs Stability | Abs Comprehensibility | Abs CSS | Abs Actionability | Abs ALF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lime | 3 | 0.013 | 0.019 | 0.030 | 0.008 | 0.005 | 0.002 | n/a |
| shap | 3 | 0.022 | 0.022 | 0.043 | 0.019 | 0.016 | 0.010 | n/a |
| anchors | 3 | 0.025 | 0.021 | 0.047 | 0.018 | 0.019 | 0.019 | n/a |
| wachter | 3 | 0.025 | 0.032 | 0.068 | 0.013 | 0.011 | 0.014 | 0.013 |
| dice | 3 | 0.030 | 0.035 | 0.090 | 0.014 | 0.011 | 0.016 | 0.016 |

| Metric | Same-Leader Rate | Same-Leader Count | Comparison Count |
| --- | ---: | ---: | ---: |
| faithfulness | 0.333 | 1 | 3 |
| stability | 0.333 | 1 | 3 |
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
| default_of_credit_card_clients | audit | dice (0.951) | lime (0.904) | anchors (0.981) | dice (0.977) | wachter (0.744) | wachter (0.526) |
| default_of_credit_card_clients | primary | anchors (0.942) | lime (0.927) | anchors (0.994) | dice (0.988) | wachter (0.758) | wachter (0.539) |

## Pairwise Win Leaders

- `actionability`: `wachter` leads with `22` wins across `24` comparable pairwise configuration matchups.
- `alf`: `wachter` leads with `4` wins across `6` comparable pairwise configuration matchups.
- `comprehensibility`: `anchors` leads with `24` wins across `24` comparable pairwise configuration matchups.
- `css`: `wachter` leads with `18` wins across `24` comparable pairwise configuration matchups.
- `faithfulness`: `dice` leads with `17` wins across `24` comparable pairwise configuration matchups.
- `stability`: `lime` leads with `22` wins across `24` comparable pairwise configuration matchups.

## Best Configurations

- Best faithfulness configuration: `default_of_credit_card_clients` / `primary` / `xgboost` / `dice` with `mean_faithfulness=0.972`
- Best stability configuration: `default_of_credit_card_clients` / `audit` / `logistic_regression` / `wachter` with `mean_stability=1.000`
- Best comprehensibility configuration: `default_of_credit_card_clients` / `audit` / `xgboost` / `anchors` with `mean_comprehensibility=1.000`
- Best CSS configuration: `default_of_credit_card_clients` / `primary` / `logistic_regression` / `dice` with `mean_css=0.994`
- Best actionability configuration: `default_of_credit_card_clients` / `primary` / `xgboost` / `wachter` with `mean_actionability=0.759`
- Best ALF configuration: `default_of_credit_card_clients` / `primary` / `xgboost` / `wachter` with `mean_alf=0.551`
