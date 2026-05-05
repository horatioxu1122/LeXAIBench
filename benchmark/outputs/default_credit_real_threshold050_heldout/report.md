# LegalBench-XAI Benchmark Report

- Experiment: `legalbench_xai_default_credit_real_threshold050`
- Generated at (UTC): `2026-03-15T19:45:55.906680+00:00`
- Total groups: `30`
- Datasets: `default_of_credit_card_clients`
- Tracks: `audit, primary`
- Models: `ebm, logistic_regression, xgboost`
- Explainers: `anchors, dice, lime, shap, wachter`

## Data Provenance

- Data source kinds: `local_override` x 30
- Evaluation split strategies: `heldout_stratified` x 30
- Stability evidence sources: `perturbation_probe` x 358, `perturbation_probe_insufficient_neighbors` x 2

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

| Explainer | Groups | Instances | Mean Faithfulness | Mean Stability | Mean Comprehensibility | Mean CSS | Mean Actionability | Mean ALF | Mean Invalid Recourse Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| anchors | 6 | 72 | 0.929 | 0.610 | 0.989 | 0.894 | 0.334 | n/a | n/a |
| dice | 6 | 72 | 0.927 | 0.466 | 0.722 | 0.985 | 0.747 | 0.523 | 0.000 |
| lime | 6 | 72 | 0.799 | 0.907 | 0.847 | 0.873 | 0.338 | n/a | n/a |
| shap | 6 | 72 | 0.912 | 0.863 | 0.814 | 0.893 | 0.340 | n/a | n/a |
| wachter | 6 | 72 | 0.918 | 0.475 | 0.724 | 0.985 | 0.747 | 0.522 | 0.000 |

## Bootstrap Uncertainty

- `95%` hierarchical-bootstrap confidence intervals with `1000` resamples over `dataset_id -> track -> model_id` benchmark clusters.

| Explainer | Faithfulness CI | Stability CI | Comprehensibility CI | CSS CI | Actionability CI | ALF CI |
| --- | --- | --- | --- | --- | --- | --- |
| anchors | [0.896, 0.957] | [0.565, 0.668] | [0.979, 1.000] | [0.885, 0.902] | [0.325, 0.342] | n/a |
| dice | [0.887, 0.966] | [0.235, 0.691] | [0.713, 0.733] | [0.979, 0.991] | [0.740, 0.754] | [0.516, 0.528] |
| lime | [0.749, 0.855] | [0.860, 0.958] | [0.841, 0.850] | [0.854, 0.892] | [0.330, 0.345] | n/a |
| shap | [0.879, 0.944] | [0.845, 0.888] | [0.778, 0.835] | [0.885, 0.903] | [0.334, 0.345] | n/a |
| wachter | [0.869, 0.966] | [0.193, 0.761] | [0.713, 0.737] | [0.979, 0.993] | [0.740, 0.753] | [0.515, 0.531] |

## Metric Sensitivity

- Strongest positive rank correlation: `comprehensibility` vs `stability` = `0.700`
- Weakest rank correlation: `actionability` vs `comprehensibility` = `-1.000`

| Metric A | Metric B | Spearman Rank Correlation | Shared Explainers |
| --- | --- | ---: | ---: |
| actionability | alf | 1.000 | 2 |
| actionability | comprehensibility | -1.000 | 5 |
| actionability | css | 0.667 | 5 |
| actionability | faithfulness | 0.000 | 5 |
| actionability | stability | -0.700 | 5 |
| alf | comprehensibility | -1.000 | 2 |
| alf | css | n/a | 2 |
| alf | faithfulness | 1.000 | 2 |
| alf | stability | -1.000 | 2 |
| comprehensibility | css | -0.667 | 5 |
| comprehensibility | faithfulness | 0.000 | 5 |
| comprehensibility | stability | 0.700 | 5 |
| css | faithfulness | 0.667 | 5 |
| css | stability | -0.975 | 5 |
| faithfulness | stability | -0.700 | 5 |

- Pairs with fewer than three shared explainers are descriptive only and should not be over-interpreted.

## Weight Sensitivity

- Randomized universal-weight sensitivity: `dice` wins `0.448` of draws with mean rank `2.194`.
- Randomized universal-weight sensitivity: `anchors` wins `0.270` of draws with mean rank `2.738`.
- Randomized universal-weight sensitivity: `shap` wins `0.132` of draws with mean rank `3.093`.
- Randomized universal-weight sensitivity: `wachter` wins `0.104` of draws with mean rank `2.681`.
- Randomized universal-weight sensitivity: `lime` wins `0.046` of draws with mean rank `4.294`.

| Scenario | Scope | Explainer | Composite Score | Rank |
| --- | --- | --- | ---: | ---: |
| balanced | all_explainers | dice | 0.597 | 1 |
| balanced | all_explainers | wachter | 0.587 | 2 |
| balanced | all_explainers | anchors | 0.504 | 3 |
| balanced | all_explainers | shap | 0.462 | 4 |
| balanced | all_explainers | lime | 0.295 | 5 |
| reliability_focused | all_explainers | shap | 0.750 | 1 |
| reliability_focused | all_explainers | anchors | 0.674 | 2 |
| reliability_focused | all_explainers | dice | 0.543 | 3 |
| reliability_focused | all_explainers | wachter | 0.517 | 4 |
| reliability_focused | all_explainers | lime | 0.397 | 5 |
| applicant_facing | all_explainers | dice | 0.699 | 1 |
| applicant_facing | all_explainers | wachter | 0.698 | 2 |
| applicant_facing | all_explainers | anchors | 0.384 | 3 |
| applicant_facing | all_explainers | shap | 0.243 | 4 |
| applicant_facing | all_explainers | lime | 0.170 | 5 |
| contestability_focused | all_explainers | dice | 0.799 | 1 |
| contestability_focused | all_explainers | wachter | 0.797 | 2 |
| contestability_focused | all_explainers | anchors | 0.312 | 3 |
| contestability_focused | all_explainers | shap | 0.235 | 4 |
| contestability_focused | all_explainers | lime | 0.122 | 5 |
| recourse_balanced | recourse_only | dice | 1.000 | 1 |
| recourse_balanced | recourse_only | wachter | 0.000 | 2 |
| legality_constrained | recourse_only | dice | 1.000 | 1 |
| legality_constrained | recourse_only | wachter | 0.000 | 2 |

## Metric Ablation

- Balanced composite baseline leader: `dice`.
- Leave-one-metric-out scenarios preserving the same leader: `2` / `5`.

| Scenario | Omitted Metric | Leader | Composite Score | Leader Changed |
| --- | --- | --- | ---: | --- |
| all_metrics_balanced | none | dice | 0.597 | no |
| drop_faithfulness | faithfulness | wachter | 0.507 | yes |
| drop_stability | stability | dice | 0.746 | no |
| drop_comprehensibility | comprehensibility | dice | 0.746 | no |
| drop_css | css | anchors | 0.582 | yes |
| drop_actionability | actionability | anchors | 0.630 | yes |

## Metric Definition Sensitivity

- CSS baseline leader under observed scoring: `dice`.
- CSS alternative definitions preserving the same leader: `4` / `4`.
- ALF baseline leader under observed scoring: `dice`.
- ALF alternative definitions preserving the same leader: `5` / `6`.

| Family | Scenario | Leader | Score | Leader Changed |
| --- | --- | --- | ---: | --- |
| CSS | observed_css | dice | 0.985 | no |
| CSS | reconstructed_css | dice | 0.985 | no |
| CSS | verifiability_focused_css | dice | 0.987 | no |
| CSS | challenge_focused_css | dice | 0.986 | no |
| CSS | specificity_light_css | dice | 0.985 | no |
| ALF | observed_alf | dice | 0.523 | no |
| ALF | balanced_arithmetic_alf | dice | 0.821 | no |
| ALF | legality_dominant_alf | dice | 0.895 | no |
| ALF | feasibility_dominant_alf | dice | 0.819 | no |
| ALF | minimality_dominant_alf | wachter | 0.794 | yes |
| ALF | invalidity_penalized_alf | dice | 0.821 | no |
| ALF | hard_failure_penalized_alf | dice | 0.821 | no |

## Track Policy Sensitivity

- Largest average absolute primary-vs-audit shift: `stability` = `0.047`.
- Most track-sensitive explainer: `anchors` with overall mean absolute delta `0.033`.
- Most track-robust explainer: `lime` with overall mean absolute delta `0.014`.
- Least leader-stable metric across primary/audit: `stability` with same-leader rate `0.000`.

| Explainer | Matched Pairs | Overall Mean Abs Delta | Abs Faithfulness | Abs Stability | Abs Comprehensibility | Abs CSS | Abs Actionability | Abs ALF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lime | 3 | 0.014 | 0.025 | 0.033 | 0.006 | 0.004 | 0.002 | n/a |
| shap | 3 | 0.017 | 0.017 | 0.022 | 0.025 | 0.014 | 0.007 | n/a |
| wachter | 3 | 0.018 | 0.012 | 0.051 | 0.018 | 0.009 | 0.008 | 0.011 |
| dice | 3 | 0.020 | 0.016 | 0.062 | 0.013 | 0.009 | 0.010 | 0.009 |
| anchors | 3 | 0.033 | 0.051 | 0.066 | 0.014 | 0.018 | 0.018 | n/a |

| Metric | Same-Leader Rate | Same-Leader Count | Comparison Count |
| --- | ---: | ---: | ---: |
| faithfulness | 1.000 | 3 | 3 |
| stability | 0.000 | 0 | 3 |
| comprehensibility | 1.000 | 3 | 3 |
| css | 1.000 | 3 | 3 |
| actionability | 0.333 | 1 | 3 |
| alf | 0.333 | 1 | 3 |

## Leave-One-Dataset-Out

| Omitted Dataset | Metric | Leader | Score |
| --- | --- | --- | ---: |

## Dataset and Track Winners

| Dataset | Track | Best Faithfulness | Best Stability | Best Comprehensibility | Best CSS | Best Actionability | Best ALF |
| --- | --- | --- | --- | --- | --- | --- | --- |
| default_of_credit_card_clients | audit | dice (0.929) | lime (0.891) | anchors (0.986) | dice (0.980) | wachter (0.743) | wachter (0.520) |
| default_of_credit_card_clients | primary | anchors (0.955) | lime (0.924) | anchors (0.992) | dice (0.989) | dice (0.752) | dice (0.526) |

## Pairwise Win Leaders

- `actionability`: `dice` leads with `22` wins across `24` comparable pairwise group matchups.
- `alf`: `dice` leads with `4` wins across `6` comparable pairwise group matchups.
- `comprehensibility`: `anchors` leads with `24` wins across `24` comparable pairwise group matchups.
- `css`: `wachter` leads with `18` wins across `24` comparable pairwise group matchups.
- `faithfulness`: `dice` leads with `15` wins across `24` comparable pairwise group matchups.
- `stability`: `lime` leads with `21` wins across `24` comparable pairwise group matchups.

## Best Groups

- Best faithfulness group: `default_of_credit_card_clients` / `primary` / `xgboost` / `dice` with `mean_faithfulness=0.974`
- Best stability group: `default_of_credit_card_clients` / `audit` / `logistic_regression` / `wachter` with `mean_stability=1.000`
- Best comprehensibility group: `default_of_credit_card_clients` / `primary` / `logistic_regression` / `anchors` with `mean_comprehensibility=1.000`
- Best CSS group: `default_of_credit_card_clients` / `primary` / `logistic_regression` / `dice` with `mean_css=0.996`
- Best actionability group: `default_of_credit_card_clients` / `primary` / `logistic_regression` / `dice` with `mean_actionability=0.757`
- Best ALF group: `default_of_credit_card_clients` / `primary` / `ebm` / `wachter` with `mean_alf=0.538`
