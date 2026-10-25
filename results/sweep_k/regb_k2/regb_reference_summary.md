# Reg B reference-method comparison

Bundle: `credit`; k = 2; reasons = principal reasons of the canonical explanation (notice content); native ranking = True.

Settings: corrections: v2 (native_contributions=True, common_faithfulness_reference=True, faithfulness_quality_weight=0.0, counterfactual_reasons_from_changes=True, cf_respect_nonactionable=True, reason_grouping_fix=True, sampling_selection=stratified_random); native ranking = True; DiCE/Wachter reasons from counterfactual changes = True; corrected temporal grouping = True.

**Unit of comparison.** The `(reasons)` columns and the leader lines compare notice reasons: each stated reason is one unit, so a compressed temporal reason (PAY_*/BILL_AMT* family) counts once, and M1/M2 are mapped to the same units before taking their top 2 (`*_grouped` columns in the CSVs). The feature-level column expands temporal groups into their supporting features and keeps the first 2, so one grouped reason can fill several slots; it is shown for reference only.

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

## Additive models (primary analysis): explainer principal reasons vs M1/M2 top-2

Means pooled over datasets and tracks with 95% stratified bootstrap CIs (strata = dataset x track).

| model | explainer | by-construction flags | overlap@2 M1 (reasons) | overlap@2 M2 (reasons) | top-1 = M1 (reasons) | reasons outside M1 | feature-level overlap@2 M1 (temporal groups expanded) | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ebm | anchors | anchors_candidates_ranked_by_native | 0.466 [0.427, 0.501] | 0.521 [0.479, 0.561] | 0.344 [0.292, 0.401] | 0.453 [0.408, 0.497] | 0.419 [0.380, 0.456] | 192 |
| ebm | dice | - | 0.660 [0.623, 0.699] | 0.686 [0.647, 0.723] | 0.660 [0.597, 0.733] | 0.524 [0.499, 0.548] | 0.560 [0.526, 0.599] | 191 |
| ebm | lime | - | 0.336 [0.305, 0.367] | 0.318 [0.286, 0.349] | 0.406 [0.365, 0.453] | 0.556 [0.532, 0.577] | 0.312 [0.279, 0.349] | 192 |
| ebm | shap | - | 0.797 [0.766, 0.832] | 0.862 [0.831, 0.891] | 0.792 [0.745, 0.839] | 0.419 [0.403, 0.435] | 0.740 [0.706, 0.772] | 192 |
| ebm | wachter | - | 0.363 [0.326, 0.401] | 0.354 [0.322, 0.392] | 0.529 [0.459, 0.592] | 0.369 [0.322, 0.427] | 0.328 [0.285, 0.368] | 157 |
| logistic_regression | anchors | anchors_candidates_ranked_by_native | 0.510 [0.466, 0.553] | 0.461 [0.419, 0.503] | 0.396 [0.323, 0.461] | 0.444 [0.400, 0.486] | 0.448 [0.407, 0.487] | 192 |
| logistic_regression | dice | - | 0.635 [0.596, 0.677] | 0.685 [0.650, 0.721] | 0.635 [0.568, 0.698] | 0.554 [0.530, 0.577] | 0.547 [0.509, 0.586] | 192 |
| logistic_regression | lime | - | 0.456 [0.414, 0.492] | 0.500 [0.471, 0.536] | 0.432 [0.385, 0.479] | 0.589 [0.562, 0.618] | 0.401 [0.362, 0.438] | 192 |
| logistic_regression | shap | shap_lr_linear_decomposition | 0.846 [0.815, 0.878] | 0.849 [0.820, 0.878] | 0.891 [0.844, 0.927] | 0.416 [0.395, 0.436] | 0.823 [0.792, 0.854] | 192 |
| logistic_regression | wachter | - | 0.275 [0.240, 0.308] | 0.283 [0.251, 0.319] | 0.410 [0.347, 0.486] | 0.462 [0.396, 0.530] | 0.246 [0.211, 0.280] | 173 |

## XGBoost (reported separately)

Caveat: XGBoost has no additive per-factor score; its native contributions are the TreeSHAP decomposition (`pred_contribs`), so the Reg B reference here is itself a SHAP-style decomposition and SHAP-on-XGBoost agreement is by construction.

| model | explainer | by-construction flags | overlap@2 M1 (reasons) | overlap@2 M2 (reasons) | top-1 = M1 (reasons) | reasons outside M1 | feature-level overlap@2 M1 (temporal groups expanded) | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| xgboost | anchors | anchors_candidates_ranked_by_native | 0.443 [0.414, 0.479] | 0.490 [0.453, 0.523] | 0.438 [0.380, 0.498] | 0.449 [0.408, 0.490] | 0.424 [0.388, 0.458] | 192 |
| xgboost | dice | - | 0.673 [0.634, 0.705] | 0.648 [0.615, 0.683] | 0.553 [0.497, 0.609] | 0.466 [0.442, 0.487] | 0.637 [0.606, 0.670] | 179 |
| xgboost | lime | - | 0.359 [0.325, 0.393] | 0.398 [0.372, 0.427] | 0.372 [0.330, 0.414] | 0.527 [0.494, 0.558] | 0.338 [0.301, 0.377] | 191 |
| xgboost | shap | shap_xgb_treeshap_equals_native | 0.789 [0.762, 0.820] | 0.922 [0.898, 0.944] | 0.807 [0.755, 0.852] | 0.436 [0.415, 0.457] | 0.776 [0.747, 0.805] | 192 |
| xgboost | wachter | - | 0.401 [0.368, 0.434] | 0.414 [0.384, 0.444] | 0.483 [0.430, 0.536] | 0.281 [0.225, 0.338] | 0.391 [0.353, 0.426] | 151 |

## Do the two commentary methods agree? (M1 vs M2, one row per applicant)

| model | overlap@2 | Jaccard | top-1 match | n applicants |
| --- | --- | --- | --- | --- |
| ebm | 0.789 [0.755, 0.820] | 0.734 [0.696, 0.774] | 0.849 [0.797, 0.896] | 192 |
| logistic_regression | 0.839 [0.807, 0.869] | 0.792 [0.753, 0.836] | 0.875 [0.828, 0.917] | 192 |
| xgboost | 0.747 [0.716, 0.776] | 0.677 [0.641, 0.712] | 0.812 [0.766, 0.859] | 192 |

## Leaders (null baselines and by-construction cells excluded from ranking)

Ranked on reason-level overlap@2 with M1 (`pr_overlap_m1_grouped`).

- `ebm`: highest pr_overlap_m1_grouped among real explainers (by-construction cells excluded) = `shap` (0.797 [0.766, 0.832])
- `logistic_regression`: highest pr_overlap_m1_grouped among real explainers (by-construction cells excluded) = `dice` (0.635 [0.596, 0.677]); by-construction, not ranked: `shap` (0.846 [0.815, 0.878]) [shap_lr_linear_decomposition]
- `xgboost`: highest pr_overlap_m1_grouped among real explainers (by-construction cells excluded) = `dice` (0.673 [0.634, 0.705]); by-construction, not ranked: `shap` (0.789 [0.762, 0.820]) [shap_xgb_treeshap_equals_native]

Interpretation notes: `reasons outside M1` is the share of the notice's reasons whose unit is not among the M1 top-k units (a rough indicator of reasons that comment 9(b)(2)-2 might not support); low reason-level overlap means the explainer's notice would omit reasons that M1 identifies as principal (cf. comment 9(b)(2)-4). These reason-level statements apply to the `(reasons)` columns only; the feature-level column and the CSV columns without the `_grouped` suffix (including `pr_outside_m1`, which counts every expanded temporal-group member) are not notice-level measures. Per-dataset numbers, the `principal_adverse` (`pra_*`) and `referenced` (`ref_*`) reason sets are in the CSV files.

## Files

- `benchmark/outputs/cslaw_denial_r1/analysis/sweep_k/regb_k2/regb_reference_instances.csv`
- `benchmark/outputs/cslaw_denial_r1/analysis/sweep_k/regb_k2/regb_reference_by_group.csv`
- `benchmark/outputs/cslaw_denial_r1/analysis/sweep_k/regb_k2/regb_reference_pooled_by_model.csv`
- `benchmark/outputs/cslaw_denial_r1/analysis/sweep_k/regb_k2/regb_reference_pooled_by_model_class.csv`
- `benchmark/outputs/cslaw_denial_r1/analysis/sweep_k/regb_k2/regb_m1_vs_m2.csv`
- `benchmark/outputs/cslaw_denial_r1/analysis/sweep_k/regb_k2/regb_reference_populations.csv`
- `benchmark/outputs/cslaw_denial_r1/analysis/sweep_k/regb_k2/regb_reference_summary.md`
