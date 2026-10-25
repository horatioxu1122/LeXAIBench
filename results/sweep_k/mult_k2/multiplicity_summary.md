# Reason multiplicity across models and explainers

Bundle: `shared`; reason set: `principal`; reason unit: `grouped`; k = 2.

Settings: corrections: v2 (native_contributions=True, common_faithfulness_reference=True, faithfulness_quality_weight=0.0, counterfactual_reasons_from_changes=True, cf_respect_nonactionable=True, reason_grouping_fix=True, sampling_selection=stratified_random); native ranking = True; DiCE/Wachter reasons from counterfactual changes = True; corrected temporal grouping = True.

**Same basis for every method.** Every reason list, explainer and Reg B alike, is cut to its first 2 units before Jaccard and top-1 are computed, because Jaccard depends on set size. Unit `grouped`: one unit per notice reason; a PAY_*/BILL_AMT* temporal family counts once, and the Reg B gap ranking is mapped to the same units before taking the top 2. Top-1 is the first notice reason. Explainers that cite fewer units (Anchors, counterfactuals) still give smaller sets.

**Why this matters (our reading; not legal advice).** Reg B (12 CFR 1002.9(b)(2)) requires specific principal reasons, and comment 9(b)(2)-2 requires them to relate to the factors actually considered or scored by the creditor's system. If comparably accurate models give the same applicant different principal reasons, the notice partly reflects an arbitrary model choice. `regb_m1`/`regb_m2` apply our operationalization of the comment 9(b)(2)-5 methods (largest positive gap between the applicant's native log-odds contribution and the near-cutoff / all-applicant reference mean) to each model; this measures multiplicity of the commentary-style reasons independent of any explainer. The operationalization is ours, not a regulator-endorsed algorithm.

## Shared sampling status

| dataset_id | track | model_id | scope | margin_aggregate | included_groups | shared_pool_rows | pool_stratum_counts.borderline | pool_stratum_counts.medium_confidence | pool_stratum_counts.strong_denial | n_selected_positions | selected_stratum_counts.borderline | selected_stratum_counts.medium_confidence | selected_stratum_counts.strong_denial | sampling_strategy | requested_max_instances | shared_fallback | fallback_reason | sampling_selection | sampling_seed | sampling_seed_key | model_selected_stratum_counts.borderline | model_selected_stratum_counts.medium_confidence | model_selected_stratum_counts.strong_denial |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| default_of_credit_card_clients | audit | ebm | models_and_tracks | min | primary|logistic_regression|primary|ebm|primary|xgboost|audit|logistic_regression|audit|ebm|audit|xgboost | 595 | 380 | 188 | 27 | 24 | 8 | 8 | 8 | decision_margin_stratified | 24 | False | None | stratified_random | 42 | default_of_credit_card_clients|shared|models_and_tracks | 1 | 9 | 14 |
| default_of_credit_card_clients | audit | logistic_regression | models_and_tracks | min | primary|logistic_regression|primary|ebm|primary|xgboost|audit|logistic_regression|audit|ebm|audit|xgboost | 595 | 380 | 188 | 27 | 24 | 8 | 8 | 8 | decision_margin_stratified | 24 | False | None | stratified_random | 42 | default_of_credit_card_clients|shared|models_and_tracks | 5 | 10 | 9 |
| default_of_credit_card_clients | audit | xgboost | models_and_tracks | min | primary|logistic_regression|primary|ebm|primary|xgboost|audit|logistic_regression|audit|ebm|audit|xgboost | 595 | 380 | 188 | 27 | 24 | 8 | 8 | 8 | decision_margin_stratified | 24 | False | None | stratified_random | 42 | default_of_credit_card_clients|shared|models_and_tracks | 2 | 10 | 12 |
| default_of_credit_card_clients | primary | ebm | models_and_tracks | min | primary|logistic_regression|primary|ebm|primary|xgboost|audit|logistic_regression|audit|ebm|audit|xgboost | 595 | 380 | 188 | 27 | 24 | 8 | 8 | 8 | decision_margin_stratified | 24 | False | None | stratified_random | 42 | default_of_credit_card_clients|shared|models_and_tracks | 1 | 8 | 15 |
| default_of_credit_card_clients | primary | logistic_regression | models_and_tracks | min | primary|logistic_regression|primary|ebm|primary|xgboost|audit|logistic_regression|audit|ebm|audit|xgboost | 595 | 380 | 188 | 27 | 24 | 8 | 8 | 8 | decision_margin_stratified | 24 | False | None | stratified_random | 42 | default_of_credit_card_clients|shared|models_and_tracks | 6 | 10 | 8 |
| default_of_credit_card_clients | primary | xgboost | models_and_tracks | min | primary|logistic_regression|primary|ebm|primary|xgboost|audit|logistic_regression|audit|ebm|audit|xgboost | 595 | 380 | 188 | 27 | 24 | 8 | 8 | 8 | decision_margin_stratified | 24 | False | None | stratified_random | 42 | default_of_credit_card_clients|shared|models_and_tracks | 2 | 8 | 14 |
| give_me_some_credit | audit | ebm | models_and_tracks | min | primary|logistic_regression|primary|ebm|primary|xgboost|audit|logistic_regression|audit|ebm|audit|xgboost | 99 | 73 | 24 | 2 | 24 | 11 | 11 | 2 | decision_margin_stratified | 24 | False | None | stratified_random | 42 | give_me_some_credit|shared|models_and_tracks | 3 | 9 | 12 |
| give_me_some_credit | audit | logistic_regression | models_and_tracks | min | primary|logistic_regression|primary|ebm|primary|xgboost|audit|logistic_regression|audit|ebm|audit|xgboost | 99 | 73 | 24 | 2 | 24 | 11 | 11 | 2 | decision_margin_stratified | 24 | False | None | stratified_random | 42 | give_me_some_credit|shared|models_and_tracks | 2 | 16 | 6 |
| give_me_some_credit | audit | xgboost | models_and_tracks | min | primary|logistic_regression|primary|ebm|primary|xgboost|audit|logistic_regression|audit|ebm|audit|xgboost | 99 | 73 | 24 | 2 | 24 | 11 | 11 | 2 | decision_margin_stratified | 24 | False | None | stratified_random | 42 | give_me_some_credit|shared|models_and_tracks | 9 | 11 | 4 |
| give_me_some_credit | primary | ebm | models_and_tracks | min | primary|logistic_regression|primary|ebm|primary|xgboost|audit|logistic_regression|audit|ebm|audit|xgboost | 99 | 73 | 24 | 2 | 24 | 11 | 11 | 2 | decision_margin_stratified | 24 | False | None | stratified_random | 42 | give_me_some_credit|shared|models_and_tracks | 3 | 10 | 11 |
| give_me_some_credit | primary | logistic_regression | models_and_tracks | min | primary|logistic_regression|primary|ebm|primary|xgboost|audit|logistic_regression|audit|ebm|audit|xgboost | 99 | 73 | 24 | 2 | 24 | 11 | 11 | 2 | decision_margin_stratified | 24 | False | None | stratified_random | 42 | give_me_some_credit|shared|models_and_tracks | 3 | 10 | 11 |
| give_me_some_credit | primary | xgboost | models_and_tracks | min | primary|logistic_regression|primary|ebm|primary|xgboost|audit|logistic_regression|audit|ebm|audit|xgboost | 99 | 73 | 24 | 2 | 24 | 11 | 11 | 2 | decision_margin_stratified | 24 | False | None | stratified_random | 42 | give_me_some_credit|shared|models_and_tracks | 6 | 15 | 3 |
| heloc | audit | ebm | models_and_tracks | min | primary|logistic_regression|primary|ebm|primary|xgboost|audit|logistic_regression|audit|ebm|audit|xgboost | 1438 | 290 | 500 | 648 | 24 | 8 | 8 | 8 | decision_margin_stratified | 24 | False | None | stratified_random | 42 | heloc|shared|models_and_tracks | 6 | 8 | 10 |
| heloc | audit | logistic_regression | models_and_tracks | min | primary|logistic_regression|primary|ebm|primary|xgboost|audit|logistic_regression|audit|ebm|audit|xgboost | 1438 | 290 | 500 | 648 | 24 | 8 | 8 | 8 | decision_margin_stratified | 24 | False | None | stratified_random | 42 | heloc|shared|models_and_tracks | 7 | 7 | 10 |
| heloc | audit | xgboost | models_and_tracks | min | primary|logistic_regression|primary|ebm|primary|xgboost|audit|logistic_regression|audit|ebm|audit|xgboost | 1438 | 290 | 500 | 648 | 24 | 8 | 8 | 8 | decision_margin_stratified | 24 | False | None | stratified_random | 42 | heloc|shared|models_and_tracks | 6 | 9 | 9 |
| heloc | primary | ebm | models_and_tracks | min | primary|logistic_regression|primary|ebm|primary|xgboost|audit|logistic_regression|audit|ebm|audit|xgboost | 1438 | 290 | 500 | 648 | 24 | 8 | 8 | 8 | decision_margin_stratified | 24 | False | None | stratified_random | 42 | heloc|shared|models_and_tracks | 6 | 8 | 10 |
| heloc | primary | logistic_regression | models_and_tracks | min | primary|logistic_regression|primary|ebm|primary|xgboost|audit|logistic_regression|audit|ebm|audit|xgboost | 1438 | 290 | 500 | 648 | 24 | 8 | 8 | 8 | decision_margin_stratified | 24 | False | None | stratified_random | 42 | heloc|shared|models_and_tracks | 7 | 7 | 10 |
| heloc | primary | xgboost | models_and_tracks | min | primary|logistic_regression|primary|ebm|primary|xgboost|audit|logistic_regression|audit|ebm|audit|xgboost | 1438 | 290 | 500 | 648 | 24 | 8 | 8 | 8 | decision_margin_stratified | 24 | False | None | stratified_random | 42 | heloc|shared|models_and_tracks | 6 | 9 | 9 |
| south_german_credit | audit | ebm | models_and_tracks | min | primary|logistic_regression|primary|ebm|primary|xgboost|audit|logistic_regression|audit|ebm|audit|xgboost | 30 | 19 | 11 | 0 | 24 | 13 | 11 | 0 | decision_margin_stratified | 24 | False | None | stratified_random | 42 | south_german_credit|shared|models_and_tracks | 4 | 13 | 7 |
| south_german_credit | audit | logistic_regression | models_and_tracks | min | primary|logistic_regression|primary|ebm|primary|xgboost|audit|logistic_regression|audit|ebm|audit|xgboost | 30 | 19 | 11 | 0 | 24 | 13 | 11 | 0 | decision_margin_stratified | 24 | False | None | stratified_random | 42 | south_german_credit|shared|models_and_tracks | 3 | 9 | 12 |
| south_german_credit | audit | xgboost | models_and_tracks | min | primary|logistic_regression|primary|ebm|primary|xgboost|audit|logistic_regression|audit|ebm|audit|xgboost | 30 | 19 | 11 | 0 | 24 | 13 | 11 | 0 | decision_margin_stratified | 24 | False | None | stratified_random | 42 | south_german_credit|shared|models_and_tracks | 10 | 12 | 2 |
| south_german_credit | primary | ebm | models_and_tracks | min | primary|logistic_regression|primary|ebm|primary|xgboost|audit|logistic_regression|audit|ebm|audit|xgboost | 30 | 19 | 11 | 0 | 24 | 13 | 11 | 0 | decision_margin_stratified | 24 | False | None | stratified_random | 42 | south_german_credit|shared|models_and_tracks | 4 | 13 | 7 |
| south_german_credit | primary | logistic_regression | models_and_tracks | min | primary|logistic_regression|primary|ebm|primary|xgboost|audit|logistic_regression|audit|ebm|audit|xgboost | 30 | 19 | 11 | 0 | 24 | 13 | 11 | 0 | decision_margin_stratified | 24 | False | None | stratified_random | 42 | south_german_credit|shared|models_and_tracks | 2 | 6 | 16 |
| south_german_credit | primary | xgboost | models_and_tracks | min | primary|logistic_regression|primary|ebm|primary|xgboost|audit|logistic_regression|audit|ebm|audit|xgboost | 30 | 19 | 11 | 0 | 24 | 13 | 11 | 0 | decision_margin_stratified | 24 | False | None | stratified_random | 42 | south_german_credit|shared|models_and_tracks | 6 | 14 | 4 |

No dataset fell back to per-model sampling.

## Model performance (multiplicity is most meaningful among comparably accurate models)

| dataset | track | model | backend | ROC AUC | adverse rate |
| --- | --- | --- | --- | --- | --- |
| default_of_credit_card_clients | audit | ebm | real | 0.777 | 0.118 |
| default_of_credit_card_clients | audit | logistic_regression | real | 0.714 | 0.076 |
| default_of_credit_card_clients | audit | xgboost | real | 0.772 | 0.116 |
| default_of_credit_card_clients | primary | ebm | real | 0.776 | 0.118 |
| default_of_credit_card_clients | primary | logistic_regression | real | 0.712 | 0.074 |
| default_of_credit_card_clients | primary | xgboost | real | 0.772 | 0.117 |
| give_me_some_credit | audit | ebm | real | 0.861 | 0.021 |
| give_me_some_credit | audit | logistic_regression | real | 0.697 | 0.005 |
| give_me_some_credit | audit | xgboost | real | 0.861 | 0.017 |
| give_me_some_credit | primary | ebm | real | 0.858 | 0.022 |
| give_me_some_credit | primary | logistic_regression | real | 0.660 | 0.005 |
| give_me_some_credit | primary | xgboost | real | 0.858 | 0.019 |
| heloc | audit | ebm | real | 0.804 | 0.533 |
| heloc | audit | logistic_regression | real | 0.798 | 0.534 |
| heloc | audit | xgboost | real | 0.799 | 0.536 |
| heloc | primary | ebm | real | 0.804 | 0.533 |
| heloc | primary | logistic_regression | real | 0.798 | 0.534 |
| heloc | primary | xgboost | real | 0.799 | 0.536 |
| south_german_credit | audit | ebm | real | 0.823 | 0.180 |
| south_german_credit | audit | logistic_regression | real | 0.798 | 0.227 |
| south_german_credit | audit | xgboost | real | 0.804 | 0.167 |
| south_german_credit | primary | ebm | real | 0.818 | 0.190 |
| south_german_credit | primary | logistic_regression | real | 0.790 | 0.223 |
| south_german_credit | primary | xgboost | real | 0.811 | 0.180 |

## Cross-model multiplicity, pooled over datasets and tracks

Mean pairwise Jaccard of reason sets across models (1 = identical), share of applicants whose top reason differs across at least two models, and mean number of distinct top reasons; 95% bootstrap CIs stratified by dataset x track.

| method | kind | mean pairwise Jaccard | top-1 differs | distinct top-1 | n applicants |
| --- | --- | --- | --- | --- | --- |
| regb_m1 | regb_reference | 0.567 [0.541, 0.594] | 0.531 [0.484, 0.578] | 1.65 [1.58, 1.72] | 192 |
| regb_m2 | regb_reference | 0.598 [0.564, 0.630] | 0.536 [0.484, 0.594] | 1.69 [1.61, 1.78] | 192 |
| anchors | explainer | 0.315 [0.292, 0.338] | 0.896 [0.859, 0.938] | 2.18 [2.11, 2.26] | 192 |
| dice | explainer | 0.531 [0.496, 0.561] | 0.534 [0.474, 0.592] | 1.66 [1.57, 1.75] | 174 |
| lime | explainer | 0.511 [0.497, 0.525] | 0.500 [0.500, 0.500] | 1.53 [1.51, 1.56] | 192 |
| shap | explainer | 0.553 [0.522, 0.582] | 0.542 [0.495, 0.589] | 1.72 [1.64, 1.80] | 192 |
| wachter | explainer | 0.572 [0.522, 0.630] | 0.539 [0.457, 0.621] | 1.62 [1.54, 1.72] | 128 |

Most model-consistent real explainer (lowest top-1-differs share; null baselines excluded): `lime` (0.500).

## Cross-model multiplicity by dataset

| dataset | method | mean pairwise Jaccard | top-1 differs | n |
| --- | --- | --- | --- | --- |
| default_of_credit_card_clients | regb_m1 | 0.597 [0.546, 0.655] | 0.042 [0.000, 0.104] | 48 |
| default_of_credit_card_clients | regb_m2 | 0.611 [0.560, 0.664] | 0.083 [0.021, 0.167] | 48 |
| default_of_credit_card_clients | anchors | 0.472 [0.425, 0.521] | 0.750 [0.625, 0.854] | 48 |
| default_of_credit_card_clients | dice | 0.535 [0.491, 0.581] | 0.042 [0.000, 0.104] | 48 |
| default_of_credit_card_clients | lime | 0.743 [0.708, 0.785] | 0.000 [0.000, 0.000] | 48 |
| default_of_credit_card_clients | shap | 0.715 [0.660, 0.778] | 0.042 [0.000, 0.104] | 48 |
| default_of_credit_card_clients | wachter | 0.711 [0.623, 0.804] | 0.375 [0.250, 0.521] | 48 |
| give_me_some_credit | regb_m1 | 0.528 [0.500, 0.556] | 0.562 [0.458, 0.688] | 48 |
| give_me_some_credit | regb_m2 | 0.681 [0.611, 0.751] | 0.583 [0.438, 0.719] | 48 |
| give_me_some_credit | anchors | 0.273 [0.242, 0.306] | 1.000 [1.000, 1.000] | 48 |
| give_me_some_credit | dice | 0.660 [0.591, 0.722] | 0.708 [0.562, 0.833] | 48 |
| give_me_some_credit | lime | 0.523 [0.484, 0.560] | 1.000 [1.000, 1.000] | 48 |
| give_me_some_credit | shap | 0.440 [0.403, 0.470] | 0.750 [0.625, 0.875] | 48 |
| give_me_some_credit | wachter | 0.517 [0.433, 0.607] | 0.625 [0.479, 0.750] | 48 |
| heloc | regb_m1 | 0.444 [0.390, 0.498] | 1.000 [1.000, 1.000] | 48 |
| heloc | regb_m2 | 0.417 [0.358, 0.485] | 0.958 [0.896, 1.000] | 48 |
| heloc | anchors | 0.106 [0.069, 0.146] | 1.000 [1.000, 1.000] | 48 |
| heloc | dice | 0.304 [0.246, 0.359] | 0.867 [0.733, 0.967] | 30 |
| heloc | lime | 0.444 [0.444, 0.444] | 1.000 [1.000, 1.000] | 48 |
| heloc | shap | 0.433 [0.370, 0.502] | 0.958 [0.896, 1.000] | 48 |
| heloc | wachter | 0.254 [0.143, 0.389] | 0.857 [0.643, 1.000] | 14 |
| south_german_credit | regb_m1 | 0.699 [0.625, 0.771] | 0.521 [0.375, 0.646] | 48 |
| south_german_credit | regb_m2 | 0.685 [0.611, 0.759] | 0.521 [0.396, 0.667] | 48 |
| south_german_credit | anchors | 0.409 [0.347, 0.492] | 0.833 [0.729, 0.938] | 48 |
| south_german_credit | dice | 0.539 [0.471, 0.618] | 0.646 [0.500, 0.782] | 48 |
| south_german_credit | lime | 0.333 [0.333, 0.333] | 0.000 [0.000, 0.000] | 48 |
| south_german_credit | shap | 0.623 [0.546, 0.697] | 0.417 [0.271, 0.562] | 48 |
| south_german_credit | wachter | 0.596 [0.457, 0.742] | 0.500 [0.278, 0.722] | 18 |

## Across-explainer disagreement within a model (real explainers only)

| model | mean pairwise Jaccard | top-1 differs | n applicants |
| --- | --- | --- | --- |
| ebm | 0.358 [0.343, 0.374] | 0.859 [0.825, 0.896] | 192 |
| logistic_regression | 0.362 [0.345, 0.380] | 0.896 [0.854, 0.938] | 192 |
| xgboost | 0.419 [0.404, 0.434] | 0.740 [0.714, 0.763] | 192 |
| all models | 0.380 [0.371, 0.390] | 0.832 [0.812, 0.852] | 576 |

### Explainer pairs (pooled over datasets, tracks and models)

| explainer A | explainer B | pair kind | mean Jaccard | n |
| --- | --- | --- | --- | --- |
| anchors | dice | real_vs_real | 0.360 [0.342, 0.382] | 554 |
| anchors | lime | real_vs_real | 0.295 [0.276, 0.310] | 576 |
| anchors | shap | real_vs_real | 0.407 [0.384, 0.426] | 576 |
| anchors | wachter | real_vs_real | 0.316 [0.290, 0.337] | 461 |
| dice | lime | real_vs_real | 0.431 [0.412, 0.449] | 554 |
| dice | shap | real_vs_real | 0.569 [0.542, 0.592] | 554 |
| dice | wachter | real_vs_real | 0.413 [0.393, 0.431] | 461 |
| lime | shap | real_vs_real | 0.353 [0.337, 0.374] | 576 |
| lime | wachter | real_vs_real | 0.415 [0.394, 0.435] | 461 |
| shap | wachter | real_vs_real | 0.373 [0.356, 0.392] | 461 |

## Coverage

| count | value |
| --- | --- |
| instances | 2743 |
| not_explained_under_all_models:dice | 18 |
| not_explained_under_all_models:wachter | 57 |
| used_explainer_instances | 2743 |
| used_regb_m1_applicants | 576 |
| used_regb_m2_applicants | 576 |

## Files

- `benchmark/outputs/cslaw_denial_r1/analysis/sweep_k/mult_k2/multiplicity_cross_model_instances.csv`
- `benchmark/outputs/cslaw_denial_r1/analysis/sweep_k/mult_k2/multiplicity_cross_model_summary.csv`
- `benchmark/outputs/cslaw_denial_r1/analysis/sweep_k/mult_k2/multiplicity_cross_explainer_instances.csv`
- `benchmark/outputs/cslaw_denial_r1/analysis/sweep_k/mult_k2/multiplicity_cross_explainer_summary.csv`
- `benchmark/outputs/cslaw_denial_r1/analysis/sweep_k/mult_k2/multiplicity_explainer_pairs_instances.csv`
- `benchmark/outputs/cslaw_denial_r1/analysis/sweep_k/mult_k2/multiplicity_explainer_pairs_summary.csv`
- `benchmark/outputs/cslaw_denial_r1/analysis/sweep_k/mult_k2/multiplicity_sampling_status.csv`
- `benchmark/outputs/cslaw_denial_r1/analysis/sweep_k/mult_k2/multiplicity_model_performance.csv`
- `benchmark/outputs/cslaw_denial_r1/analysis/sweep_k/mult_k2/multiplicity_summary.md`
