# T1: structural facts checked against the data

Long format in T1_by_construction.csv (bundle, fact, group_1, group_2, statistic, value, n, note).

- non_recourse_actionability_ceiling: unified actionability of explanations without counterfactual changes vs the definitional ceiling 0.35 (0.20*0.25 + 0.30*1.0).
- decision_sensitivity_visibility_by_family, recourse_simplicity_by_family: distributions.
- alf_applicability_by_explainer, explanation_quality_signal_by_explainer_model,
  verifiability_saturation_by_dataset_track, dice_wachter_vcr_tie (per configuration).
- corrections_status: corrections version and options recorded by the run, the share of instances whose faithfulness was scored on the common native reference (`model_native_common_reference`), and the faithfulness quality weight used.
- cf_reason_source: per DiCE/Wachter instance, where its reasons come from (`counterfactual_changes`, `no_valid_change`; `not_recorded` in v1 bundles, whose reasons are a borrowed ranking).

## Findings

- [credit] Corrections v2: common faithfulness reference True (share of instances scored on model_native_common_reference: anchors 100%, dice 100%, lime 100%, shap 100%, wachter 100%); quality weight 0; DiCE/Wachter reasons from counterfactual changes True (wachter: counterfactual_changes 100%; wachter: counterfactual_changes 100%).
- [credit] Non-recourse explanations (n=1727): max unified actionability 0.3500 vs ceiling 0.35; 0 above the ceiling.
- [credit] Decision-sensitivity visibility by family: attribution median 0.550, counterfactual median 1.000, rule median 0.550.
- [credit] Share of explanations with recourse_simplicity = 1.0: attribution 100.0%, counterfactual 76.0%, rule 100.0%.
- [credit] ALF applicable share: anchors 0.0%, dice 100.0%, lime 0.0%, shap 0.0%, wachter 100.0%.
- [credit] Quality signal is a constant within these explainer/model cells: dice/ebm=1, dice/logistic_regression=1, dice/xgboost=1, shap/ebm=0.88, shap/logistic_regression=1, shap/xgboost=0.97, wachter/ebm=1, wachter/logistic_regression=1, wachter/xgboost=1 (its weight in faithfulness: 0.0).
- [credit] Verifiability saturated (V = 1) in 44.2% of explanations; per dataset x track the share ranges 0.0%-100.0%.
- [credit] DiCE and Wachter tie exactly on V/C/R in 9 of 24 configurations (V 12, C 9, R 24).
- [nullbaselines] Corrections v2: common faithfulness reference True (share of instances scored on model_native_common_reference: global_ranking 100%, random_reasons 100%); quality weight 0; DiCE/Wachter reasons from counterfactual changes True (no counterfactual instances).
- [nullbaselines] Non-recourse explanations (n=1140): max unified actionability 0.3500 vs ceiling 0.35; 0 above the ceiling.
- [nullbaselines] Decision-sensitivity visibility by family: attribution median 0.550.
- [nullbaselines] Share of explanations with recourse_simplicity = 1.0: attribution 100.0%.
- [nullbaselines] ALF applicable share: global_ranking 0.0%, random_reasons 0.0%.
- [nullbaselines] Quality signal is a constant within these explainer/model cells: global_ranking/ebm=0.5, global_ranking/logistic_regression=0.5, global_ranking/xgboost=0.5, random_reasons/ebm=0.5, random_reasons/logistic_regression=0.5, random_reasons/xgboost=0.5 (its weight in faithfulness: 0.0).
- [nullbaselines] Verifiability saturated (V = 1) in 36.1% of explanations; per dataset x track the share ranges 0.0%-100.0%.
- [noncredit] Corrections v2: common faithfulness reference True (share of instances scored on model_native_common_reference: anchors 100%, dice 100%, lime 100%, shap 100%, wachter 100%); quality weight 0; DiCE/Wachter reasons from counterfactual changes True (wachter: counterfactual_changes 100%; wachter: counterfactual_changes 100%).
- [noncredit] Non-recourse explanations (n=1292): max unified actionability 0.3500 vs ceiling 0.35; 0 above the ceiling.
- [noncredit] Decision-sensitivity visibility by family: attribution median 0.550, counterfactual median 1.000, rule median 0.550.
- [noncredit] Share of explanations with recourse_simplicity = 1.0: attribution 100.0%, counterfactual 90.0%, rule 100.0%.
- [noncredit] ALF applicable share: anchors 0.0%, dice 100.0%, lime 0.0%, shap 0.0%, wachter 100.0%.
- [noncredit] Quality signal is a constant within these explainer/model cells: dice/ebm=1, dice/logistic_regression=1, dice/xgboost=1, shap/ebm=0.88, shap/logistic_regression=1, shap/xgboost=0.97, wachter/ebm=1, wachter/logistic_regression=1, wachter/xgboost=1 (its weight in faithfulness: 0.0).
- [noncredit] Verifiability saturated (V = 1) in 34.1% of explanations; per dataset x track the share ranges 23.3%-50.9%.
- [noncredit] DiCE and Wachter tie exactly on V/C/R in 0 of 18 configurations (V 0, C 0, R 6).
