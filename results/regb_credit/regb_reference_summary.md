# Reg B reference-method comparison

Bundle: `credit`; k = 4; reasons = principal reasons of the canonical explanation (notice content); native ranking = True.

Settings: corrections: v2 (native_contributions=True, common_faithfulness_reference=True, faithfulness_quality_weight=0.0, counterfactual_reasons_from_changes=True, cf_respect_nonactionable=True, reason_grouping_fix=True, sampling_selection=stratified_random); native ranking = True; DiCE/Wachter reasons from counterfactual changes = True; corrected temporal grouping = True.

**Unit of comparison.** The `(reasons)` columns and the leader lines compare notice reasons: each stated reason is one unit, so a compressed temporal reason (PAY_*/BILL_AMT* family) counts once, and M1/M2 are mapped to the same units before taking their top 4 (`*_grouped` columns in the CSVs). The feature-level column expands temporal groups into their supporting features and keeps the first 4, so one grouped reason can fill several slots; it is shown for reference only.

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

## Coverage

| count | value |
| --- | --- |
| instances | 2770 |
| used | 2770 |

## Additive models (primary analysis): explainer principal reasons vs M1/M2 top-4

Means pooled over datasets and tracks with 95% stratified bootstrap CIs (strata = dataset x track).

| model | explainer | by-construction flags | overlap@4 M1 (reasons) | overlap@4 M2 (reasons) | top-1 = M1 (reasons) | reasons outside M1 | feature-level overlap@4 M1 (temporal groups expanded) | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ebm | anchors | anchors_candidates_ranked_by_native | 0.377 [0.354, 0.400] | 0.406 [0.384, 0.426] | 0.344 [0.292, 0.401] | 0.217 [0.176, 0.260] | 0.364 [0.339, 0.388] | 192 |
| ebm | dice | - | 0.633 [0.605, 0.659] | 0.661 [0.635, 0.688] | 0.660 [0.592, 0.717] | 0.266 [0.238, 0.294] | 0.580 [0.551, 0.610] | 191 |
| ebm | lime | - | 0.309 [0.291, 0.328] | 0.359 [0.340, 0.378] | 0.406 [0.359, 0.453] | 0.448 [0.421, 0.477] | 0.309 [0.289, 0.328] | 192 |
| ebm | shap | - | 0.778 [0.756, 0.800] | 0.809 [0.786, 0.831] | 0.792 [0.740, 0.839] | 0.094 [0.076, 0.112] | 0.819 [0.799, 0.840] | 192 |
| ebm | wachter | - | 0.237 [0.221, 0.255] | 0.240 [0.223, 0.259] | 0.529 [0.465, 0.592] | 0.207 [0.162, 0.255] | 0.215 [0.196, 0.234] | 157 |
| logistic_regression | anchors | anchors_candidates_ranked_by_native | 0.470 [0.444, 0.497] | 0.470 [0.444, 0.496] | 0.396 [0.328, 0.464] | 0.201 [0.163, 0.240] | 0.441 [0.414, 0.467] | 192 |
| logistic_regression | dice | - | 0.576 [0.548, 0.601] | 0.577 [0.552, 0.601] | 0.635 [0.573, 0.698] | 0.290 [0.262, 0.319] | 0.577 [0.552, 0.603] | 192 |
| logistic_regression | lime | - | 0.319 [0.296, 0.342] | 0.326 [0.307, 0.346] | 0.432 [0.385, 0.484] | 0.493 [0.464, 0.523] | 0.353 [0.330, 0.376] | 192 |
| logistic_regression | shap | shap_lr_linear_decomposition | 0.777 [0.753, 0.800] | 0.785 [0.763, 0.807] | 0.891 [0.844, 0.932] | 0.090 [0.072, 0.110] | 0.804 [0.781, 0.827] | 192 |
| logistic_regression | wachter | - | 0.193 [0.176, 0.209] | 0.184 [0.168, 0.199] | 0.410 [0.341, 0.480] | 0.263 [0.199, 0.324] | 0.187 [0.171, 0.202] | 173 |

## XGBoost (reported separately)

Caveat: XGBoost has no additive per-factor score; its native contributions are the TreeSHAP decomposition (`pred_contribs`), so the Reg B reference here is itself a SHAP-style decomposition and SHAP-on-XGBoost agreement is by construction.

| model | explainer | by-construction flags | overlap@4 M1 (reasons) | overlap@4 M2 (reasons) | top-1 = M1 (reasons) | reasons outside M1 | feature-level overlap@4 M1 (temporal groups expanded) | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| xgboost | anchors | anchors_candidates_ranked_by_native | 0.411 [0.389, 0.432] | 0.438 [0.417, 0.457] | 0.438 [0.380, 0.495] | 0.167 [0.132, 0.205] | 0.410 [0.388, 0.432] | 192 |
| xgboost | dice | - | 0.682 [0.656, 0.709] | 0.684 [0.657, 0.710] | 0.553 [0.497, 0.603] | 0.145 [0.122, 0.168] | 0.708 [0.684, 0.732] | 179 |
| xgboost | lime | - | 0.407 [0.386, 0.428] | 0.447 [0.424, 0.468] | 0.372 [0.330, 0.419] | 0.301 [0.262, 0.342] | 0.409 [0.387, 0.430] | 191 |
| xgboost | shap | shap_xgb_treeshap_equals_native | 0.769 [0.750, 0.790] | 0.840 [0.822, 0.859] | 0.807 [0.760, 0.854] | 0.104 [0.085, 0.124] | 0.825 [0.803, 0.845] | 192 |
| xgboost | wachter | - | 0.272 [0.256, 0.289] | 0.274 [0.258, 0.290] | 0.483 [0.430, 0.536] | 0.106 [0.070, 0.146] | 0.262 [0.244, 0.279] | 151 |

## Do the two commentary methods agree? (M1 vs M2, one row per applicant)

| model | overlap@4 | Jaccard | top-1 match | n applicants |
| --- | --- | --- | --- | --- |
| ebm | 0.836 [0.816, 0.855] | 0.763 [0.739, 0.787] | 0.849 [0.802, 0.896] | 192 |
| logistic_regression | 0.884 [0.865, 0.905] | 0.829 [0.802, 0.856] | 0.875 [0.828, 0.917] | 192 |
| xgboost | 0.854 [0.836, 0.871] | 0.785 [0.760, 0.810] | 0.812 [0.766, 0.854] | 192 |

## Leaders (null baselines and by-construction cells excluded from ranking)

Ranked on reason-level overlap@4 with M1 (`pr_overlap_m1_grouped`).

- `ebm`: highest pr_overlap_m1_grouped among real explainers (by-construction cells excluded) = `shap` (0.778 [0.756, 0.800])
- `logistic_regression`: highest pr_overlap_m1_grouped among real explainers (by-construction cells excluded) = `dice` (0.576 [0.548, 0.601]); by-construction, not ranked: `shap` (0.777 [0.753, 0.800]) [shap_lr_linear_decomposition]
- `xgboost`: highest pr_overlap_m1_grouped among real explainers (by-construction cells excluded) = `dice` (0.682 [0.656, 0.709]); by-construction, not ranked: `shap` (0.769 [0.750, 0.790]) [shap_xgb_treeshap_equals_native]

Interpretation notes: `reasons outside M1` is the share of the notice's reasons whose unit is not among the M1 top-k units (a rough indicator of reasons that comment 9(b)(2)-2 might not support); low reason-level overlap means the explainer's notice would omit reasons that M1 identifies as principal (cf. comment 9(b)(2)-4). These reason-level statements apply to the `(reasons)` columns only; the feature-level column and the CSV columns without the `_grouped` suffix (including `pr_outside_m1`, which counts every expanded temporal-group member) are not notice-level measures. Per-dataset numbers, the `principal_adverse` (`pra_*`) and `referenced` (`ref_*`) reason sets are in the CSV files.

## Files

- `benchmark/outputs/cslaw_denial_r1/analysis/regb_credit/regb_reference_instances.csv`
- `benchmark/outputs/cslaw_denial_r1/analysis/regb_credit/regb_reference_by_group.csv`
- `benchmark/outputs/cslaw_denial_r1/analysis/regb_credit/regb_reference_pooled_by_model.csv`
- `benchmark/outputs/cslaw_denial_r1/analysis/regb_credit/regb_reference_pooled_by_model_class.csv`
- `benchmark/outputs/cslaw_denial_r1/analysis/regb_credit/regb_m1_vs_m2.csv`
- `benchmark/outputs/cslaw_denial_r1/analysis/regb_credit/regb_reference_populations.csv`
- `benchmark/outputs/cslaw_denial_r1/analysis/regb_credit/regb_reference_summary.md`
