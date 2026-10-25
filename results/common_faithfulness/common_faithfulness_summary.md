# Family-neutral faithfulness against native model contributions

- `credit`: `benchmark/outputs/cslaw_denial_r1/credit/results.json`; native ranking = True; corrections: v2 (native_contributions=True, common_faithfulness_reference=True, faithfulness_quality_weight=0.0, counterfactual_reasons_from_changes=True, cf_respect_nonactionable=True, reason_grouping_fix=True, sampling_selection=stratified_random); native ranking = True; DiCE/Wachter reasons from counterfactual changes = True; corrected temporal grouping = True.
- `nullbaselines`: `benchmark/outputs/cslaw_denial_r1/nullbaselines/results.json`; native ranking = True; corrections: v2 (native_contributions=True, common_faithfulness_reference=True, faithfulness_quality_weight=0.0, counterfactual_reasons_from_changes=True, cf_respect_nonactionable=True, reason_grouping_fix=True, sampling_selection=stratified_random); native ranking = True; DiCE/Wachter reasons from counterfactual changes = True; corrected temporal grouping = True.

**Legal grounding (our reading; not legal advice).** Regulation B,
12 CFR 1002.9(b)(2), requires a statement of specific principal reasons for
adverse action. The Official Interpretations state that disclosed reasons must
relate to and accurately describe the factors actually considered or scored
(comment 9(b)(2)-2) and that no factor that was a principal reason may be
excluded (comment 9(b)(2)-4). For credit scoring systems, comment 9(b)(2)-5
describes acceptable methods, including (method 1, "M1") identifying the
factors for which the applicant's score fell furthest below the average score
for each factor among applicants whose total score was at or slightly above the
minimum passing score, and (method 2, "M2") the same comparison against the
average of all applicants.

**Operationalization (ours, not a regulator-endorsed algorithm).** Per-factor
scores are the fitted model's native per-feature contributions in log-odds of
the adverse outcome (exact for logistic regression and for EBM with
pair-interaction terms split 50/50; a TreeSHAP decomposition for XGBoost).
"Fell furthest below" is the largest positive gap between the applicant's
contribution and the reference population's mean contribution. M1's reference
population is the approved applicants closest to the decision cutoff; M2's is a
seeded sample of all evaluation applicants.

Every explainer is scored against the same reference (the model's native log-odds contributions). In v1 bundles the benchmark's built-in faithfulness for SHAP/LIME/leave-one-out uses the explainer's own values (self-referential). In v2 bundles (`common_faithfulness_reference=True`) the built-in faithfulness is already scored against this common reference, so this table complements it rather than correcting it. Means are pooled over datasets and tracks with 95% bootstrap CIs stratified by dataset x track; per-dataset rows are in the CSV.

## Pipeline consistency (v2 bundles)

- `credit`: pipeline reference_feature_coverage vs native_mass_coverage on 2770 instances with their own native contributions: max |diff| = 0.00e+00; 0 above 1e-06.
- `nullbaselines`: pipeline reference_feature_coverage vs native_mass_coverage on 1140 instances with their own native contributions: max |diff| = 0.00e+00; 0 above 1e-06.

**By-construction flags.** (i) `shap_lr_linear_decomposition` /
`shap_xgb_treeshap_equals_native`: SHAP on logistic regression and XGBoost
decomposes the same quantity as the native contributions (LR SHAP differs per
feature only by a constant, because its baseline is the mean of the first 64
training rows; XGB SHAP equals `pred_contribs`), so agreement, especially with
the all-applicants reference, is expected to be near-trivial.
(ii) `cf_features_ranked_by_native` / `anchors_candidates_ranked_by_native`:
in v1 bundles DiCE/Wachter referenced features are the top native contributions
(and Anchors candidates are ranked by them) when the run used native ranking,
and always on logistic regression, whose ranking contributions already equal
coef x standardized value. In v2 bundles (correction
`counterfactual_reasons_from_changes`) DiCE/Wachter reasons are the features
their counterfactuals change, so `cf_features_ranked_by_native` is not set; the
caveat `cf_reason_order_from_native` records that ties in change frequency are
ordered, and weighted, by native contribution (rank-based metrics such as top-1
still partly reflect the native order). Each v2 DiCE/Wachter reason also copies
its `signed_direction` from the sign of the native contribution, so the v2
pipeline scores counterfactual (and Anchors) directions as neutral (0.5) in
faithfulness under the common reference, and common_faithfulness.py leaves their
direction agreement empty. Anchors keeps its flag under native ranking, which
the v2 preset turns on.
(iii) `shap_ebm_probability_units` (caveat): EBM SHAP is in probability units,
so only rank/set comparisons are meaningful. `global_ranking_uses_native_magnitudes`
marks the canned-reasons null baseline; `stub_model_heuristic_everywhere` marks
cells where a real model backend failed and the stub's heuristic contributions
stand in for both the explanation and the "native" reference. The caveat
`ranking_uses_heuristic_proxy` marks DiCE/Wachter/Anchors on EBM/XGBoost without
native ranking (v1 default): their feature ranking comes from a taxonomy-weighted
heuristic, not the fitted model. Rows with `by_construction=True` should not be
read as evidence that an explainer "finds" the legally relevant reasons.

Anchors marks every condition as `increases_denial`, so its direction agreement equals the share of its cited features with positive native contributions. In v2 bundles DiCE/Wachter reasons copy their `signed_direction` from the sign of the native contribution, so their direction agreement would be 1.0 by construction; it is left empty here, and the v2 pipeline scores counterfactual (and Anchors) directions as neutral (0.5) under the common reference. Native mass coverage depends on how many features are cited (SHAP/LIME/null baselines cite 6, Wachter often 1-2, Anchors a few); `mass coverage / oracle` divides by the best coverage achievable with the same number of features. The pipeline's own v2 coverage term (and so `faithfulness_score` and T8 `faithfulness_legal`) is not normalized this way and favors explainers that cite more features.

## Results by bundle and model

### `credit` / `ebm`

| explainer | by-construction flags | mass coverage / oracle | adverse mass coverage | direction agreement | overlap@4 top adverse | CF changes on adverse features | n cited | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anchors | anchors_candidates_ranked_by_native | 0.816 [0.791, 0.841] | 0.435 [0.413, 0.457] | 0.915 [0.883, 0.944] | 0.414 [0.395, 0.434] | n/a | 2.073 [2.000, 2.146] | 192 |
| dice | - | 0.795 [0.775, 0.815] | 0.651 [0.629, 0.672] | n/a | 0.597 [0.568, 0.626] | 0.919 [0.885, 0.953] | 4.262 [4.110, 4.408] | 191 |
| lime | - | 0.686 [0.662, 0.712] | 0.443 [0.424, 0.463] | 0.781 [0.755, 0.809] | 0.398 [0.376, 0.421] | n/a | 2.964 [2.896, 3.026] | 192 |
| shap | - | 0.956 [0.947, 0.963] | 0.817 [0.806, 0.827] | 0.999 [0.997, 1.000] | 0.878 [0.857, 0.897] | n/a | 4.760 [4.620, 4.891] | 192 |
| wachter | - | 0.717 [0.673, 0.761] | 0.311 [0.284, 0.338] | n/a | 0.226 [0.207, 0.245] | 0.987 [0.975, 0.997] | 1.210 [1.153, 1.274] | 157 |

### `credit` / `logistic_regression`

| explainer | by-construction flags | mass coverage / oracle | adverse mass coverage | direction agreement | overlap@4 top adverse | CF changes on adverse features | n cited | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anchors | anchors_candidates_ranked_by_native | 0.665 [0.628, 0.704] | 0.447 [0.414, 0.481] | 0.834 [0.796, 0.869] | 0.436 [0.410, 0.461] | n/a | 2.391 [2.297, 2.484] | 192 |
| dice | - | 0.709 [0.683, 0.734] | 0.664 [0.642, 0.688] | n/a | 0.587 [0.562, 0.614] | 0.911 [0.875, 0.945] | 4.031 [3.885, 4.172] | 192 |
| lime | - | 0.737 [0.720, 0.753] | 0.553 [0.535, 0.570] | 0.690 [0.655, 0.723] | 0.401 [0.380, 0.422] | n/a | 3.583 [3.531, 3.635] | 192 |
| shap | shap_lr_linear_decomposition | 0.887 [0.867, 0.904] | 0.830 [0.819, 0.840] | 0.987 [0.980, 0.993] | 0.847 [0.826, 0.867] | n/a | 4.271 [4.151, 4.396] | 192 |
| wachter | - | 0.593 [0.533, 0.654] | 0.305 [0.270, 0.343] | n/a | 0.171 [0.154, 0.188] | 0.986 [0.971, 0.997] | 1.046 [1.017, 1.081] | 173 |

### `credit` / `xgboost`

| explainer | by-construction flags | mass coverage / oracle | adverse mass coverage | direction agreement | overlap@4 top adverse | CF changes on adverse features | n cited | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anchors | anchors_candidates_ranked_by_native | 0.777 [0.745, 0.809] | 0.511 [0.483, 0.539] | 0.907 [0.876, 0.935] | 0.445 [0.424, 0.466] | n/a | 2.151 [2.057, 2.240] | 192 |
| dice | - | 0.827 [0.808, 0.846] | 0.746 [0.725, 0.767] | n/a | 0.743 [0.715, 0.771] | 0.955 [0.931, 0.977] | 4.095 [3.955, 4.229] | 179 |
| lime | - | 0.680 [0.653, 0.707] | 0.530 [0.505, 0.553] | 0.760 [0.717, 0.801] | 0.491 [0.466, 0.516] | n/a | 3.377 [3.288, 3.471] | 191 |
| shap | shap_xgb_treeshap_equals_native | 0.961 [0.950, 0.971] | 0.914 [0.907, 0.920] | 1.000 [1.000, 1.000] | 0.952 [0.935, 0.967] | n/a | 4.714 [4.568, 4.854] | 192 |
| wachter | - | 0.789 [0.753, 0.825] | 0.417 [0.390, 0.442] | n/a | 0.286 [0.267, 0.305] | 0.993 [0.983, 1.000] | 1.272 [1.205, 1.344] | 151 |

Leaders on `native_mass_coverage_vs_oracle` (`credit`):

- `ebm`: highest native_mass_coverage_vs_oracle among real explainers (by-construction cells excluded) = `shap` (0.956 [0.947, 0.963])
- `logistic_regression`: highest native_mass_coverage_vs_oracle among real explainers (by-construction cells excluded) = `lime` (0.737 [0.720, 0.753]); by-construction, not ranked: `shap` (0.887 [0.867, 0.904]) [shap_lr_linear_decomposition]
- `xgboost`: highest native_mass_coverage_vs_oracle among real explainers (by-construction cells excluded) = `dice` (0.827 [0.808, 0.846]); by-construction, not ranked: `shap` (0.961 [0.950, 0.971]) [shap_xgb_treeshap_equals_native]

Leaders on `direction_agreement` (`credit`):

- `ebm`: highest direction_agreement among real explainers (by-construction cells excluded) = `shap` (0.999 [0.997, 1.000])
- `logistic_regression`: highest direction_agreement among real explainers (by-construction cells excluded) = `lime` (0.690 [0.655, 0.723]); by-construction, not ranked: `shap` (0.987 [0.980, 0.993]) [shap_lr_linear_decomposition]
- `xgboost`: highest direction_agreement among real explainers (by-construction cells excluded) = `lime` (0.760 [0.717, 0.801]); by-construction, not ranked: `shap` (1.000 [1.000, 1.000]) [shap_xgb_treeshap_equals_native]

Leaders on `pr_overlap_top_adverse` (`credit`):

- `ebm`: highest pr_overlap_top_adverse among real explainers (by-construction cells excluded) = `shap` (0.878 [0.857, 0.897])
- `logistic_regression`: highest pr_overlap_top_adverse among real explainers (by-construction cells excluded) = `dice` (0.587 [0.562, 0.614]); by-construction, not ranked: `shap` (0.847 [0.826, 0.867]) [shap_lr_linear_decomposition]
- `xgboost`: highest pr_overlap_top_adverse among real explainers (by-construction cells excluded) = `dice` (0.743 [0.715, 0.771]); by-construction, not ranked: `shap` (0.952 [0.935, 0.967]) [shap_xgb_treeshap_equals_native]

### `nullbaselines` / `ebm`

| explainer | by-construction flags | mass coverage / oracle | adverse mass coverage | direction agreement | overlap@4 top adverse | CF changes on adverse features | n cited | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| global_ranking (null baseline) | global_ranking_uses_native_magnitudes | 0.842 [0.828, 0.857] | 0.692 [0.672, 0.710] | 0.734 [0.713, 0.755] | 0.577 [0.551, 0.600] | n/a | 6.000 [6.000, 6.000] | 192 |
| random_reasons (null baseline) | - | 0.312 [0.281, 0.345] | 0.179 [0.156, 0.202] | 0.617 [0.569, 0.662] | 0.190 [0.166, 0.214] | n/a | 3.074 [2.899, 3.245] | 188 |

### `nullbaselines` / `logistic_regression`

| explainer | by-construction flags | mass coverage / oracle | adverse mass coverage | direction agreement | overlap@4 top adverse | CF changes on adverse features | n cited | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| global_ranking (null baseline) | global_ranking_uses_native_magnitudes | 0.886 [0.874, 0.897] | 0.759 [0.743, 0.774] | 0.731 [0.711, 0.751] | 0.613 [0.592, 0.634] | n/a | 6.000 [6.000, 6.000] | 192 |
| random_reasons (null baseline) | - | 0.320 [0.284, 0.354] | 0.207 [0.176, 0.239] | 0.644 [0.604, 0.681] | 0.195 [0.172, 0.219] | n/a | 3.159 [3.005, 3.312] | 189 |

### `nullbaselines` / `xgboost`

| explainer | by-construction flags | mass coverage / oracle | adverse mass coverage | direction agreement | overlap@4 top adverse | CF changes on adverse features | n cited | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| global_ranking (null baseline) | global_ranking_uses_native_magnitudes | 0.923 [0.914, 0.931] | 0.851 [0.836, 0.864] | 0.746 [0.722, 0.768] | 0.704 [0.682, 0.727] | n/a | 6.000 [6.000, 6.000] | 192 |
| random_reasons (null baseline) | - | 0.275 [0.240, 0.311] | 0.197 [0.167, 0.228] | 0.594 [0.554, 0.633] | 0.180 [0.155, 0.206] | n/a | 3.155 [2.995, 3.321] | 187 |

## Native-contribution joins and coverage

| count | value |
| --- | --- |
| credit:instances | 2770 |
| credit:native_own | 2770 |
| nullbaselines:instances | 1140 |
| nullbaselines:native_own | 1140 |

Joined values are borrowed by instance_id from another bundle and accepted only when the prediction scores match; `native_conflicts_across_bundles` > 0 means the bundles' model fits differ for some instances (joins are still score-checked).

## Files

- `benchmark/outputs/cslaw_denial_r1/analysis/common_faithfulness/common_faithfulness_instances.csv`
- `benchmark/outputs/cslaw_denial_r1/analysis/common_faithfulness/common_faithfulness_summary.csv`
- `benchmark/outputs/cslaw_denial_r1/analysis/common_faithfulness/common_faithfulness_summary.md`
