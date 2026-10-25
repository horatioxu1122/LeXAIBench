# Reg B reference-method comparison

Bundle: `noncredit`; k = 4; reasons = principal reasons of the canonical explanation (notice content); native ranking = True.

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
| instances | 1553 |
| used | 1553 |

## Additive models (primary analysis): explainer principal reasons vs M1/M2 top-4

Means pooled over datasets and tracks with 95% stratified bootstrap CIs (strata = dataset x track).

| model | explainer | by-construction flags | overlap@4 M1 (reasons) | overlap@4 M2 (reasons) | top-1 = M1 (reasons) | reasons outside M1 | feature-level overlap@4 M1 (temporal groups expanded) | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ebm | anchors | anchors_candidates_ranked_by_native | 0.439 [0.403, 0.474] | 0.440 [0.405, 0.472] | 0.597 [0.521, 0.667] | 0.162 [0.115, 0.212] | 0.439 [0.402, 0.475] | 144 |
| ebm | dice | - | 0.602 [0.546, 0.658] | 0.643 [0.582, 0.699] | 0.714 [0.592, 0.817] | 0.269 [0.207, 0.327] | 0.602 [0.546, 0.658] | 49 |
| ebm | lime | - | 0.339 [0.309, 0.367] | 0.350 [0.324, 0.378] | 0.396 [0.333, 0.465] | 0.512 [0.473, 0.552] | 0.339 [0.310, 0.368] | 144 |
| ebm | shap | - | 0.891 [0.866, 0.913] | 0.942 [0.923, 0.960] | 0.931 [0.889, 0.965] | 0.094 [0.073, 0.117] | 0.891 [0.867, 0.913] | 144 |
| ebm | wachter | - | 0.229 [0.200, 0.257] | 0.236 [0.207, 0.264] | 0.686 [0.514, 0.829] | 0.114 [0.029, 0.229] | 0.229 [0.200, 0.257] | 35 |
| logistic_regression | anchors | anchors_candidates_ranked_by_native | 0.483 [0.447, 0.519] | 0.481 [0.448, 0.514] | 0.521 [0.451, 0.590] | 0.260 [0.223, 0.300] | 0.483 [0.447, 0.519] | 144 |
| logistic_regression | dice | - | 0.354 [0.318, 0.396] | 0.365 [0.323, 0.406] | 0.583 [0.458, 0.708] | 0.427 [0.366, 0.484] | 0.354 [0.312, 0.396] | 48 |
| logistic_regression | lime | - | 0.404 [0.374, 0.434] | 0.402 [0.374, 0.431] | 0.503 [0.427, 0.573] | 0.568 [0.533, 0.601] | 0.404 [0.374, 0.436] | 143 |
| logistic_regression | shap | shap_lr_linear_decomposition | 0.876 [0.849, 0.900] | 0.934 [0.912, 0.953] | 0.924 [0.882, 0.958] | 0.101 [0.078, 0.126] | 0.876 [0.851, 0.902] | 144 |
| logistic_regression | wachter | - | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 36 |

## XGBoost (reported separately)

Caveat: XGBoost has no additive per-factor score; its native contributions are the TreeSHAP decomposition (`pred_contribs`), so the Reg B reference here is itself a SHAP-style decomposition and SHAP-on-XGBoost agreement is by construction.

| model | explainer | by-construction flags | overlap@4 M1 (reasons) | overlap@4 M2 (reasons) | top-1 = M1 (reasons) | reasons outside M1 | feature-level overlap@4 M1 (temporal groups expanded) | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| xgboost | anchors | anchors_candidates_ranked_by_native | 0.416 [0.384, 0.448] | 0.407 [0.370, 0.443] | 0.500 [0.431, 0.576] | 0.110 [0.074, 0.150] | 0.416 [0.385, 0.448] | 144 |
| xgboost | dice | - | 0.668 [0.628, 0.709] | 0.668 [0.622, 0.714] | 0.837 [0.735, 0.918] | 0.158 [0.109, 0.207] | 0.668 [0.628, 0.709] | 49 |
| xgboost | lime | - | 0.192 [0.167, 0.217] | 0.217 [0.190, 0.245] | 0.078 [0.043, 0.121] | 0.716 [0.680, 0.753] | 0.192 [0.168, 0.218] | 141 |
| xgboost | shap | shap_xgb_treeshap_equals_native | 0.858 [0.828, 0.888] | 0.943 [0.924, 0.962] | 0.750 [0.688, 0.812] | 0.064 [0.045, 0.084] | 0.858 [0.828, 0.887] | 144 |
| xgboost | wachter | - | 0.275 [0.256, 0.297] | 0.269 [0.252, 0.288] | 0.545 [0.409, 0.682] | 0.000 [0.000, 0.000] | 0.275 [0.256, 0.297] | 44 |

## Do the two commentary methods agree? (M1 vs M2, one row per applicant)

| model | overlap@4 | Jaccard | top-1 match | n applicants |
| --- | --- | --- | --- | --- |
| ebm | 0.904 [0.881, 0.925] | 0.863 [0.834, 0.895] | 0.951 [0.910, 0.986] | 144 |
| logistic_regression | 0.901 [0.877, 0.925] | 0.866 [0.833, 0.895] | 0.931 [0.889, 0.965] | 144 |
| xgboost | 0.873 [0.845, 0.900] | 0.827 [0.793, 0.860] | 0.750 [0.681, 0.812] | 144 |

## Leaders (null baselines and by-construction cells excluded from ranking)

Ranked on reason-level overlap@4 with M1 (`pr_overlap_m1_grouped`).

- `ebm`: highest pr_overlap_m1_grouped among real explainers (by-construction cells excluded) = `shap` (0.891 [0.866, 0.913])
- `logistic_regression`: highest pr_overlap_m1_grouped among real explainers (by-construction cells excluded) = `lime` (0.404 [0.374, 0.434]); by-construction, not ranked: `shap` (0.876 [0.849, 0.900]) [shap_lr_linear_decomposition]
- `xgboost`: highest pr_overlap_m1_grouped among real explainers (by-construction cells excluded) = `dice` (0.668 [0.628, 0.709]); by-construction, not ranked: `shap` (0.858 [0.828, 0.888]) [shap_xgb_treeshap_equals_native]

Interpretation notes: `reasons outside M1` is the share of the notice's reasons whose unit is not among the M1 top-k units (a rough indicator of reasons that comment 9(b)(2)-2 might not support); low reason-level overlap means the explainer's notice would omit reasons that M1 identifies as principal (cf. comment 9(b)(2)-4). These reason-level statements apply to the `(reasons)` columns only; the feature-level column and the CSV columns without the `_grouped` suffix (including `pr_outside_m1`, which counts every expanded temporal-group member) are not notice-level measures. Per-dataset numbers, the `principal_adverse` (`pra_*`) and `referenced` (`ref_*`) reason sets are in the CSV files.

## Files

- `benchmark/outputs/cslaw_denial_r1/analysis/regb_noncredit/regb_reference_instances.csv`
- `benchmark/outputs/cslaw_denial_r1/analysis/regb_noncredit/regb_reference_by_group.csv`
- `benchmark/outputs/cslaw_denial_r1/analysis/regb_noncredit/regb_reference_pooled_by_model.csv`
- `benchmark/outputs/cslaw_denial_r1/analysis/regb_noncredit/regb_reference_pooled_by_model_class.csv`
- `benchmark/outputs/cslaw_denial_r1/analysis/regb_noncredit/regb_m1_vs_m2.csv`
- `benchmark/outputs/cslaw_denial_r1/analysis/regb_noncredit/regb_reference_populations.csv`
- `benchmark/outputs/cslaw_denial_r1/analysis/regb_noncredit/regb_reference_summary.md`
