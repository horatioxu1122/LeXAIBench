# CS&Law paper tables

Bootstrap resamples: 1000; seed: 42. Null baselines (random_reasons, global_ranking) are never ranked or reported as leaders.

## Inputs

- `credit`: `benchmark/outputs/cslaw_denial_r1/credit/results.json` (experiment `legalbench_xai_cslaw_v2_credit`, 120 summary rows); corrections: v2 (native_contributions=True, common_faithfulness_reference=True, faithfulness_quality_weight=0.0, counterfactual_reasons_from_changes=True, cf_respect_nonactionable=True, reason_grouping_fix=True, sampling_selection=stratified_random)
- `nullbaselines`: `benchmark/outputs/cslaw_denial_r1/nullbaselines/results.json` (experiment `legalbench_xai_cslaw_v2_credit_nullbaselines`, 48 summary rows); corrections: v2 (native_contributions=True, common_faithfulness_reference=True, faithfulness_quality_weight=0.0, counterfactual_reasons_from_changes=True, cf_respect_nonactionable=True, reason_grouping_fix=True, sampling_selection=stratified_random)
- `noncredit`: `benchmark/outputs/cslaw_denial_r1/noncredit/results.json` (experiment `legalbench_xai_cslaw_v2_noncredit`, 90 summary rows); corrections: v2 (native_contributions=True, common_faithfulness_reference=True, faithfulness_quality_weight=0.0, counterfactual_reasons_from_changes=True, cf_respect_nonactionable=True, reason_grouping_fix=True, sampling_selection=stratified_random)

## T1 structural facts (T1_by_construction.csv/.md)

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

## T2 CSS decomposition (T2_css_decomposition.csv)

V, C, S (decision-sensitivity visibility), R, CSS and CSS_no_dsv = (0.30V + 0.30C + 0.15R)/0.75, evaluated_rows-weighted, by explainer x dataset and pooled (dataset_id = __pooled__).

- [credit] pooled CSS leader dice (0.906); CSS without decision-sensitivity leader lime (0.891).
- [nullbaselines] null-baseline rows are in the CSV (bundle=nullbaselines) for reference only; they are never ranked or used for leaders or pooled correlations.
- [noncredit] pooled CSS leader wachter (0.868); CSS without decision-sensitivity leader wachter (0.824).

## T3 pairwise recount (T3_pairwise_recount.csv, T3_pairwise_pairs.csv)

Configuration-level strict-> comparisons as in reporting._pairwise_wins. Categories: decided_by_construction (actionability/ALF, recourse vs non-recourse; faithfulness on the common reference when either side carries a T8 by-construction flag, e.g. SHAP on LR/XGB or native-ranked Anchors), formula_advantaged (CSS with S / comprehensibility with recourse simplicity, recourse vs non-recourse), not_comparable_evidence (faithfulness, direct reference vs structural proxy), open (everything else; faithfulness is open only when both sides use the common reference and neither is flagged).

- [__all__] faithfulness: most wins shap (146); most open wins lime (52); 215/342 wins are not 'open'.
- [__all__] stability: most wins shap (129); most open wins shap (129); 0/340 wins are not 'open'.
- [__all__] comprehensibility: most wins anchors (107); most open wins anchors (71); 185/341 wins are not 'open'.
- [__all__] comprehensibility_no_rs: most wins wachter (112); most open wins wachter (112); 0/340 wins are not 'open'.
- [__all__] css: most wins dice (110); most open wins lime (52); 186/334 wins are not 'open'.
- [__all__] css_no_dsv: most wins lime (81); most open wins lime (81); 0/310 wins are not 'open'.
- [__all__] actionability: most wins wachter (112); most open wins anchors (52); 186/333 wins are not 'open'.
- [__all__] alf: most wins wachter (16); most open wins wachter (16); 0/30 wins are not 'open'.
- [credit] faithfulness: most wins shap (96); most open wins lime (39); 144/240 wins are not 'open'.
- [credit] stability: most wins shap (84); most open wins shap (84); 0/238 wins are not 'open'.
- [credit] comprehensibility: most wins wachter (82); most open wins anchors (39); 143/239 wins are not 'open'.
- [credit] comprehensibility_no_rs: most wins wachter (89); most open wins wachter (89); 0/238 wins are not 'open'.
- [credit] css: most wins dice (82); most open wins lime (38); 144/232 wins are not 'open'.
- [credit] css_no_dsv: most wins lime (62); most open wins lime (62); 0/208 wins are not 'open'.
- [credit] actionability: most wins wachter (88); most open wins lime (35); 144/237 wins are not 'open'.
- [credit] alf: most wins dice (13); most open wins dice (13); 0/24 wins are not 'open'.
- [noncredit] faithfulness: most wins shap (50); most open wins lime (13); 71/102 wins are not 'open'.
- [noncredit] stability: most wins shap (45); most open wins shap (45); 0/102 wins are not 'open'.
- [noncredit] comprehensibility: most wins anchors (37); most open wins anchors (32); 42/102 wins are not 'open'.
- [noncredit] comprehensibility_no_rs: most wins anchors (37); most open wins anchors (37); 0/102 wins are not 'open'.
- [noncredit] css: most wins dice (28); most open wins anchors (23); 42/102 wins are not 'open'.
- [noncredit] css_no_dsv: most wins anchors (32); most open wins anchors (32); 0/102 wins are not 'open'.
- [noncredit] actionability: most wins anchors (29); most open wins anchors (29); 42/96 wins are not 'open'.
- [noncredit] alf: most wins wachter (5); most open wins wachter (5); 0/6 wins are not 'open'.
- [credit] pairwise_wins.csv not found; cross-check against reporting skipped.
- [noncredit] pairwise_wins.csv not found; cross-check against reporting skipped.

## T4 per-domain means (T4_per_domain.csv)

Pooled instance means per dataset x explainer with 95% percentile CIs from a hierarchical bootstrap (track, then model, then instance). ALF averages applicable instances only. Leaders by metric (null baselines excluded):

- [credit] default_of_credit_card_clients: css wachter -> css_no_dsv lime; comprehensibility wachter -> comprehensibility_no_rs wachter; faithfulness shap -> faithfulness_no_quality shap.
- [credit] give_me_some_credit: css dice -> css_no_dsv anchors; comprehensibility wachter -> comprehensibility_no_rs wachter; faithfulness shap -> faithfulness_no_quality shap.
- [credit] heloc: css wachter -> css_no_dsv wachter; comprehensibility lime -> comprehensibility_no_rs lime; faithfulness shap -> faithfulness_no_quality shap.
- [credit] south_german_credit: css dice -> css_no_dsv lime; comprehensibility wachter -> comprehensibility_no_rs wachter; faithfulness shap -> faithfulness_no_quality shap.
- [nullbaselines] null-baseline rows are in the CSV (bundle=nullbaselines) for reference only; they are never ranked or used for leaders or pooled correlations.
- [noncredit] fda_inspections: css lime -> css_no_dsv lime; comprehensibility anchors -> comprehensibility_no_rs anchors; faithfulness shap -> faithfulness_no_quality shap.
- [noncredit] hmda: css wachter -> css_no_dsv wachter; comprehensibility wachter -> comprehensibility_no_rs wachter; faithfulness shap -> faithfulness_no_quality shap.
- [noncredit] ny_courts_pretrial: css dice -> css_no_dsv anchors; comprehensibility dice -> comprehensibility_no_rs dice; faithfulness shap -> faithfulness_no_quality shap.

## T5 correlations (T5_correlations.csv)

- [credit] pooled configuration level: rho(faithfulness, css) = -0.058 (n=120); rho(faithfulness, css_no_dsv) = 0.202 (n=120); rho(faithfulness_no_quality, css_no_dsv) = 0.202 (n=120).
- [credit] within-explainer configuration-level rho(faithfulness, css): anchors 0.361, dice 0.110, lime 0.203, shap 0.368, wachter 0.089.
- [nullbaselines] null-baseline rows are in the CSV (bundle=nullbaselines) for reference only; they are never ranked or used for leaders or pooled correlations.
- [noncredit] pooled configuration level: rho(faithfulness, css) = -0.338 (n=68); rho(faithfulness, css_no_dsv) = -0.141 (n=68); rho(faithfulness_no_quality, css_no_dsv) = -0.141 (n=68).
- [noncredit] within-explainer configuration-level rho(faithfulness, css): anchors -0.550, dice -0.180, lime -0.370, shap 0.245, wachter -0.696.
- Paper (configuration level, faithfulness vs CSS): +0.23 on the credit subset (n=120) and -0.45 on the non-credit subset (n=90); compare with the pooled rows above.

## T6 robustness (T6_robustness.csv)

- (nothing to report)

- note: [credit] by_explainer.csv not found; baseline leaders from results.json summary.
- note: [credit] leave_one_dataset_out_leaders.csv not found.
- note: [credit] metric_ablation_profiles.csv not found.
- note: [credit] metric_weight_random_summary.csv not found.
- note: [credit] css_definition_profiles.csv not found.
- note: [credit] alf_definition_profiles.csv not found.
- note: [credit] track_policy_leader_summary.csv not found.
- note: [noncredit] by_explainer.csv not found; baseline leaders from results.json summary.
- note: [noncredit] leave_one_dataset_out_leaders.csv not found.
- note: [noncredit] metric_ablation_profiles.csv not found.
- note: [noncredit] metric_weight_random_summary.csv not found.
- note: [noncredit] css_definition_profiles.csv not found.
- note: [noncredit] alf_definition_profiles.csv not found.
- note: [noncredit] track_policy_leader_summary.csv not found.

## T7 null baselines (T7_null_baselines.csv)

- T7: 564 of 564 null-baseline instances matched by instance_id to 'credit'.
- T7: null-baseline faithfulness is scored against the common native reference (v2 correction common_faithfulness_reference; the cslaw null_baselines faithfulness_reference setting 'self' only affects the self-reference kept under self_reference_signed_contributions), so it is comparable with the real explainers' faithfulness in the same bundle.

| metric | global_ranking [NULL BASELINE] | random_reasons [NULL BASELINE] | anchors | dice | lime | shap | wachter |
|---|---|---|---|---|---|---|---|
| faithfulness | 0.785 | 0.424 | 0.448 | 0.540 | 0.632 | 0.833 | 0.387 |
| faithfulness_no_quality | 0.785 | 0.424 | 0.448 | 0.540 | 0.632 | 0.833 | 0.387 |
| stability | 1.000 | 0.388 | 0.620 | 0.703 | 0.877 | 0.908 | 0.759 |
| comprehensibility | 0.695 | 0.882 | 0.946 | 0.806 | 0.878 | 0.790 | 0.968 |
| comprehensibility_no_rs | 0.619 | 0.853 | 0.933 | 0.775 | 0.848 | 0.738 | 0.970 |
| css | 0.778 | 0.814 | 0.796 | 0.906 | 0.812 | 0.786 | 0.901 |
| css_no_dsv | 0.846 | 0.894 | 0.871 | 0.875 | 0.891 | 0.858 | 0.868 |
| css_verifiability | 0.713 | 0.774 | 0.738 | 0.751 | 0.742 | 0.716 | 0.732 |
| css_correctability | 0.901 | 0.962 | 0.940 | 0.935 | 0.986 | 0.928 | 0.939 |
| css_dsv | 0.573 | 0.572 | 0.573 | 1.000 | 0.573 | 0.573 | 1.000 |
| css_reason_specificity | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| actionability | 0.320 | 0.339 | 0.332 | 0.699 | 0.346 | 0.328 | 0.707 |

## T8 legal-function table (T8_legal_functions.csv)

One column per legal function of a statement of reasons, each family-neutral: V (verifiability), C (correctability), D (decision-sensitivity directionality: do the reasons state a direction?), R (reason specificity), CSS_v2 = 0.30V + 0.30C + 0.25D + 0.15R, comprehensibility without recourse simplicity, faithfulness (recorded score when scored against the common native reference, v2; otherwise the quality-free v1 score, which is still self-referential and flagged `v1_self_referential_faithfulness`, so it never yields a `leader_excluding_by_construction`; `by_construction_flags` marks configurations whose agreement with the native reference is built in, e.g. SHAP on logistic regression and XGBoost, and the EBM-only column restricts to the model where no explainer decomposes the reference), stability, and ALF / unified actionability for DiCE and Wachter only (across families they are decided by construction). Instance means per dataset x explainer and pooled, with 95% percentile CIs from a hierarchical bootstrap (pooled: dataset > track > model > instance). `basis` says when a v1 bundle falls back (CSS_v2 -> css_no_dsv). Null baselines are listed, never ranked.

- [credit] pooled leaders: css_verifiability dice 0.750 [0.502, 0.988] (CI overlaps lime); css_correctability lime 0.986 [0.954, 1.000] (CI overlaps anchors); css_dsv_directionality anchors 1.000 [1.000, 1.000] (CI overlaps dice); css_reason_specificity anchors 1.000 [1.000, 1.000] (CI overlaps dice); css_v2 lime 0.918 [0.850, 0.989] (CI overlaps dice); comprehensibility_no_rs wachter 0.970 [0.950, 0.991] (CI overlaps anchors); faithfulness_legal shap 0.833 [0.754, 0.905] (CI overlaps lime) [by construction: shap_lr_linear_decomposition;shap_xgb_treeshap_equals_native; best unflagged: lime]; faithfulness_legal_ebm_only shap 0.823 [0.728, 0.928] (CI overlaps lime); stability shap 0.909 [0.864, 0.957] (CI overlaps lime); alf_recourse_only wachter 0.449 [0.401, 0.497] (CI overlaps dice); actionability_recourse_only wachter 0.706 [0.676, 0.739] (CI overlaps dice).
- [credit] corrections v2; CSS_v2 basis: directionality; faithfulness basis: common_reference.
- [nullbaselines] null-baseline rows are in the CSV (bundle=nullbaselines) for reference only; they are never ranked or used for leaders or pooled correlations.
- [noncredit] pooled leaders: css_verifiability anchors 0.853 [0.772, 0.912] (CI overlaps shap); css_correctability wachter 0.767 [0.642, 0.904] (CI overlaps anchors); css_dsv_directionality anchors 1.000 [1.000, 1.000] (CI overlaps dice); css_reason_specificity anchors 1.000 [1.000, 1.000] (CI overlaps dice); css_v2 wachter 0.868 [0.797, 0.944] (CI overlaps anchors); comprehensibility_no_rs wachter 0.963 [0.962, 0.963] (CI separated from anchors); faithfulness_legal shap 0.872 [0.856, 0.890] (CI separated from lime) [by construction: shap_lr_linear_decomposition;shap_xgb_treeshap_equals_native; best unflagged: lime]; faithfulness_legal_ebm_only shap 0.857 [0.843, 0.872] (CI separated from lime); stability shap 0.932 [0.877, 0.982] (CI overlaps wachter); alf_recourse_only wachter 0.520 [0.469, 0.568] (CI overlaps dice); actionability_recourse_only wachter 0.690 [0.673, 0.706] (CI separated from dice).
- [noncredit] corrections v2; CSS_v2 basis: directionality; faithfulness basis: common_reference.
