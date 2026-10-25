# Reg B reference-method comparison

Bundle: `nullbaselines`; k = 4; reasons = principal reasons of the canonical explanation (notice content); native ranking = True.

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
| instances | 1140 |
| used | 1140 |

## Additive models (primary analysis): explainer principal reasons vs M1/M2 top-4

Means pooled over datasets and tracks with 95% stratified bootstrap CIs (strata = dataset x track).

| model | explainer | by-construction flags | overlap@4 M1 (reasons) | overlap@4 M2 (reasons) | top-1 = M1 (reasons) | reasons outside M1 | feature-level overlap@4 M1 (temporal groups expanded) | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ebm | global_ranking (null baseline) | global_ranking_uses_native_magnitudes | 0.613 [0.589, 0.638] | 0.688 [0.665, 0.711] | 0.536 [0.484, 0.594] | 0.387 [0.362, 0.411] | 0.509 [0.482, 0.536] | 192 |
| ebm | random_reasons (null baseline) | - | 0.250 [0.224, 0.276] | 0.258 [0.232, 0.284] | 0.069 [0.037, 0.106] | 0.625 [0.590, 0.663] | 0.177 [0.152, 0.202] | 188 |
| logistic_regression | global_ranking (null baseline) | global_ranking_uses_native_magnitudes | 0.569 [0.549, 0.589] | 0.602 [0.581, 0.622] | 0.490 [0.438, 0.547] | 0.379 [0.358, 0.401] | 0.566 [0.544, 0.589] | 192 |
| logistic_regression | random_reasons (null baseline) | - | 0.299 [0.274, 0.324] | 0.304 [0.277, 0.329] | 0.153 [0.101, 0.206] | 0.560 [0.527, 0.594] | 0.201 [0.177, 0.224] | 189 |

## XGBoost (reported separately)

Caveat: XGBoost has no additive per-factor score; its native contributions are the TreeSHAP decomposition (`pred_contribs`), so the Reg B reference here is itself a SHAP-style decomposition and SHAP-on-XGBoost agreement is by construction.

| model | explainer | by-construction flags | overlap@4 M1 (reasons) | overlap@4 M2 (reasons) | top-1 = M1 (reasons) | reasons outside M1 | feature-level overlap@4 M1 (temporal groups expanded) | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| xgboost | global_ranking (null baseline) | global_ranking_uses_native_magnitudes | 0.657 [0.634, 0.681] | 0.735 [0.714, 0.757] | 0.682 [0.630, 0.729] | 0.303 [0.278, 0.329] | 0.612 [0.589, 0.634] | 192 |
| xgboost | random_reasons (null baseline) | - | 0.257 [0.230, 0.283] | 0.260 [0.234, 0.286] | 0.118 [0.075, 0.160] | 0.630 [0.599, 0.662] | 0.175 [0.150, 0.199] | 187 |

## Do the two commentary methods agree? (M1 vs M2, one row per applicant)

| model | overlap@4 | Jaccard | top-1 match | n applicants |
| --- | --- | --- | --- | --- |
| ebm | 0.836 [0.815, 0.855] | 0.763 [0.738, 0.789] | 0.849 [0.802, 0.896] | 192 |
| logistic_regression | 0.884 [0.865, 0.904] | 0.829 [0.801, 0.854] | 0.875 [0.828, 0.917] | 192 |
| xgboost | 0.854 [0.836, 0.871] | 0.785 [0.762, 0.808] | 0.812 [0.766, 0.859] | 192 |

## Leaders (null baselines and by-construction cells excluded from ranking)

Ranked on reason-level overlap@4 with M1 (`pr_overlap_m1_grouped`).

- none

Interpretation notes: `reasons outside M1` is the share of the notice's reasons whose unit is not among the M1 top-k units (a rough indicator of reasons that comment 9(b)(2)-2 might not support); low reason-level overlap means the explainer's notice would omit reasons that M1 identifies as principal (cf. comment 9(b)(2)-4). These reason-level statements apply to the `(reasons)` columns only; the feature-level column and the CSV columns without the `_grouped` suffix (including `pr_outside_m1`, which counts every expanded temporal-group member) are not notice-level measures. Per-dataset numbers, the `principal_adverse` (`pra_*`) and `referenced` (`ref_*`) reason sets are in the CSV files.

## Files

- `benchmark/outputs/cslaw_denial_r1/analysis/regb_nullbaselines/regb_reference_instances.csv`
- `benchmark/outputs/cslaw_denial_r1/analysis/regb_nullbaselines/regb_reference_by_group.csv`
- `benchmark/outputs/cslaw_denial_r1/analysis/regb_nullbaselines/regb_reference_pooled_by_model.csv`
- `benchmark/outputs/cslaw_denial_r1/analysis/regb_nullbaselines/regb_reference_pooled_by_model_class.csv`
- `benchmark/outputs/cslaw_denial_r1/analysis/regb_nullbaselines/regb_m1_vs_m2.csv`
- `benchmark/outputs/cslaw_denial_r1/analysis/regb_nullbaselines/regb_reference_populations.csv`
- `benchmark/outputs/cslaw_denial_r1/analysis/regb_nullbaselines/regb_reference_summary.md`
