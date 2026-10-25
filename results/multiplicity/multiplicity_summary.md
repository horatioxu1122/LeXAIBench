# Reason multiplicity across models and explainers

Bundle: `shared`; reason set: `principal`; reason unit: `grouped`; k = 4.

Settings: corrections: v2 (native_contributions=True, common_faithfulness_reference=True, faithfulness_quality_weight=0.0, counterfactual_reasons_from_changes=True, cf_respect_nonactionable=True, reason_grouping_fix=True, sampling_selection=stratified_random); native ranking = True; DiCE/Wachter reasons from counterfactual changes = True; corrected temporal grouping = True.

**Same basis for every method.** Every reason list, explainer and Reg B alike, is cut to its first 4 units before Jaccard and top-1 are computed, because Jaccard depends on set size. Unit `grouped`: one unit per notice reason; a PAY_*/BILL_AMT* temporal family counts once, and the Reg B gap ranking is mapped to the same units before taking the top 4. Top-1 is the first notice reason. Explainers that cite fewer units (Anchors, counterfactuals) still give smaller sets.

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
| regb_m1 | regb_reference | 0.622 [0.602, 0.642] | 0.531 [0.484, 0.578] | 1.65 [1.59, 1.71] | 192 |
| regb_m2 | regb_reference | 0.644 [0.625, 0.664] | 0.536 [0.479, 0.594] | 1.69 [1.61, 1.76] | 192 |
| anchors | explainer | 0.327 [0.309, 0.348] | 0.896 [0.859, 0.932] | 2.18 [2.11, 2.26] | 192 |
| dice | explainer | 0.603 [0.579, 0.628] | 0.534 [0.471, 0.592] | 1.66 [1.58, 1.74] | 174 |
| lime | explainer | 0.518 [0.505, 0.532] | 0.500 [0.500, 0.500] | 1.53 [1.51, 1.56] | 192 |
| shap | explainer | 0.578 [0.557, 0.599] | 0.542 [0.490, 0.594] | 1.72 [1.65, 1.80] | 192 |
| wachter | explainer | 0.572 [0.517, 0.627] | 0.539 [0.461, 0.625] | 1.62 [1.53, 1.73] | 128 |

Most model-consistent real explainer (lowest top-1-differs share; null baselines excluded): `lime` (0.500).

## Cross-model multiplicity by dataset

| dataset | method | mean pairwise Jaccard | top-1 differs | n |
| --- | --- | --- | --- | --- |
| default_of_credit_card_clients | regb_m1 | 0.718 [0.682, 0.758] | 0.042 [0.000, 0.104] | 48 |
| default_of_credit_card_clients | regb_m2 | 0.798 [0.755, 0.843] | 0.083 [0.021, 0.167] | 48 |
| default_of_credit_card_clients | anchors | 0.509 [0.470, 0.546] | 0.750 [0.625, 0.875] | 48 |
| default_of_credit_card_clients | dice | 0.685 [0.631, 0.737] | 0.042 [0.000, 0.104] | 48 |
| default_of_credit_card_clients | lime | 0.743 [0.708, 0.785] | 0.000 [0.000, 0.000] | 48 |
| default_of_credit_card_clients | shap | 0.666 [0.618, 0.711] | 0.042 [0.000, 0.104] | 48 |
| default_of_credit_card_clients | wachter | 0.711 [0.613, 0.806] | 0.375 [0.229, 0.521] | 48 |
| give_me_some_credit | regb_m1 | 0.545 [0.510, 0.581] | 0.562 [0.438, 0.688] | 48 |
| give_me_some_credit | regb_m2 | 0.567 [0.544, 0.591] | 0.583 [0.438, 0.729] | 48 |
| give_me_some_credit | anchors | 0.289 [0.261, 0.319] | 1.000 [1.000, 1.000] | 48 |
| give_me_some_credit | dice | 0.530 [0.488, 0.572] | 0.708 [0.562, 0.833] | 48 |
| give_me_some_credit | lime | 0.615 [0.584, 0.645] | 1.000 [1.000, 1.000] | 48 |
| give_me_some_credit | shap | 0.581 [0.544, 0.618] | 0.750 [0.625, 0.875] | 48 |
| give_me_some_credit | wachter | 0.517 [0.429, 0.610] | 0.625 [0.479, 0.750] | 48 |
| heloc | regb_m1 | 0.533 [0.495, 0.569] | 1.000 [1.000, 1.000] | 48 |
| heloc | regb_m2 | 0.510 [0.476, 0.544] | 0.958 [0.896, 1.000] | 48 |
| heloc | anchors | 0.105 [0.068, 0.144] | 1.000 [1.000, 1.000] | 48 |
| heloc | dice | 0.424 [0.382, 0.466] | 0.867 [0.733, 0.967] | 30 |
| heloc | lime | 0.444 [0.444, 0.444] | 1.000 [1.000, 1.000] | 48 |
| heloc | shap | 0.446 [0.411, 0.487] | 0.958 [0.896, 1.000] | 48 |
| heloc | wachter | 0.254 [0.143, 0.373] | 0.857 [0.643, 1.000] | 14 |
| south_german_credit | regb_m1 | 0.690 [0.646, 0.740] | 0.521 [0.375, 0.667] | 48 |
| south_german_credit | regb_m2 | 0.700 [0.651, 0.753] | 0.521 [0.375, 0.667] | 48 |
| south_german_credit | anchors | 0.405 [0.351, 0.461] | 0.833 [0.729, 0.938] | 48 |
| south_german_credit | dice | 0.707 [0.649, 0.761] | 0.646 [0.500, 0.771] | 48 |
| south_german_credit | lime | 0.270 [0.259, 0.281] | 0.000 [0.000, 0.000] | 48 |
| south_german_credit | shap | 0.617 [0.575, 0.661] | 0.417 [0.271, 0.562] | 48 |
| south_german_credit | wachter | 0.596 [0.454, 0.744] | 0.500 [0.278, 0.722] | 18 |

## Across-explainer disagreement within a model (real explainers only)

| model | mean pairwise Jaccard | top-1 differs | n applicants |
| --- | --- | --- | --- |
| ebm | 0.360 [0.348, 0.373] | 0.859 [0.823, 0.891] | 192 |
| logistic_regression | 0.316 [0.303, 0.329] | 0.896 [0.859, 0.932] | 192 |
| xgboost | 0.423 [0.408, 0.439] | 0.740 [0.714, 0.766] | 192 |
| all models | 0.366 [0.358, 0.374] | 0.832 [0.812, 0.851] | 576 |

### Explainer pairs (pooled over datasets, tracks and models)

| explainer A | explainer B | pair kind | mean Jaccard | n |
| --- | --- | --- | --- | --- |
| anchors | dice | real_vs_real | 0.367 [0.352, 0.382] | 554 |
| anchors | lime | real_vs_real | 0.343 [0.327, 0.358] | 576 |
| anchors | shap | real_vs_real | 0.455 [0.440, 0.471] | 576 |
| anchors | wachter | real_vs_real | 0.324 [0.302, 0.347] | 461 |
| dice | lime | real_vs_real | 0.375 [0.361, 0.389] | 554 |
| dice | shap | real_vs_real | 0.526 [0.508, 0.544] | 554 |
| dice | wachter | real_vs_real | 0.332 [0.320, 0.345] | 461 |
| lime | shap | real_vs_real | 0.365 [0.350, 0.380] | 576 |
| lime | wachter | real_vs_real | 0.370 [0.352, 0.388] | 461 |
| shap | wachter | real_vs_real | 0.298 [0.286, 0.311] | 461 |

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

- `benchmark/outputs/cslaw_denial_r1/analysis/multiplicity/multiplicity_cross_model_instances.csv`
- `benchmark/outputs/cslaw_denial_r1/analysis/multiplicity/multiplicity_cross_model_summary.csv`
- `benchmark/outputs/cslaw_denial_r1/analysis/multiplicity/multiplicity_cross_explainer_instances.csv`
- `benchmark/outputs/cslaw_denial_r1/analysis/multiplicity/multiplicity_cross_explainer_summary.csv`
- `benchmark/outputs/cslaw_denial_r1/analysis/multiplicity/multiplicity_explainer_pairs_instances.csv`
- `benchmark/outputs/cslaw_denial_r1/analysis/multiplicity/multiplicity_explainer_pairs_summary.csv`
- `benchmark/outputs/cslaw_denial_r1/analysis/multiplicity/multiplicity_sampling_status.csv`
- `benchmark/outputs/cslaw_denial_r1/analysis/multiplicity/multiplicity_model_performance.csv`
- `benchmark/outputs/cslaw_denial_r1/analysis/multiplicity/multiplicity_summary.md`
