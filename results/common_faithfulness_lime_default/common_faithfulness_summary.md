# Family-neutral faithfulness against native model contributions

- `lime_default`: `benchmark/outputs/cslaw_denial_r1_rerun/lime_default/results.json`; native ranking = True; corrections: v2 (native_contributions=True, common_faithfulness_reference=True, faithfulness_quality_weight=0.0, counterfactual_reasons_from_changes=True, cf_respect_nonactionable=True, reason_grouping_fix=True, sampling_selection=stratified_random); native ranking = True; DiCE/Wachter reasons from counterfactual changes = True; corrected temporal grouping = True.

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

- `lime_default`: pipeline reference_feature_coverage vs native_mass_coverage on 576 instances with their own native contributions: max |diff| = 0.00e+00; 0 above 1e-06.

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

### `lime_default` / `ebm`

| explainer | by-construction flags | mass coverage / oracle | adverse mass coverage | direction agreement | overlap@4 top adverse | CF changes on adverse features | n cited | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lime | - | 0.859 [0.841, 0.877] | 0.680 [0.663, 0.695] | 0.964 [0.953, 0.974] | 0.702 [0.677, 0.728] | n/a | 4.172 [4.042, 4.302] | 192 |

### `lime_default` / `logistic_regression`

| explainer | by-construction flags | mass coverage / oracle | adverse mass coverage | direction agreement | overlap@4 top adverse | CF changes on adverse features | n cited | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lime | - | 0.799 [0.777, 0.822] | 0.720 [0.705, 0.735] | 0.988 [0.981, 0.994] | 0.679 [0.657, 0.703] | n/a | 3.818 [3.693, 3.943] | 192 |

### `lime_default` / `xgboost`

| explainer | by-construction flags | mass coverage / oracle | adverse mass coverage | direction agreement | overlap@4 top adverse | CF changes on adverse features | n cited | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lime | - | 0.902 [0.887, 0.917] | 0.833 [0.820, 0.845] | 0.982 [0.974, 0.990] | 0.788 [0.767, 0.807] | n/a | 4.391 [4.255, 4.521] | 192 |

Leaders on `native_mass_coverage_vs_oracle` (`lime_default`):

- `ebm`: highest native_mass_coverage_vs_oracle among real explainers (by-construction cells excluded) = `lime` (0.859 [0.841, 0.877])
- `logistic_regression`: highest native_mass_coverage_vs_oracle among real explainers (by-construction cells excluded) = `lime` (0.799 [0.777, 0.822])
- `xgboost`: highest native_mass_coverage_vs_oracle among real explainers (by-construction cells excluded) = `lime` (0.902 [0.887, 0.917])

Leaders on `direction_agreement` (`lime_default`):

- `ebm`: highest direction_agreement among real explainers (by-construction cells excluded) = `lime` (0.964 [0.953, 0.974])
- `logistic_regression`: highest direction_agreement among real explainers (by-construction cells excluded) = `lime` (0.988 [0.981, 0.994])
- `xgboost`: highest direction_agreement among real explainers (by-construction cells excluded) = `lime` (0.982 [0.974, 0.990])

Leaders on `pr_overlap_top_adverse` (`lime_default`):

- `ebm`: highest pr_overlap_top_adverse among real explainers (by-construction cells excluded) = `lime` (0.702 [0.677, 0.728])
- `logistic_regression`: highest pr_overlap_top_adverse among real explainers (by-construction cells excluded) = `lime` (0.679 [0.657, 0.703])
- `xgboost`: highest pr_overlap_top_adverse among real explainers (by-construction cells excluded) = `lime` (0.788 [0.767, 0.807])

## Native-contribution joins and coverage

| count | value |
| --- | --- |
| lime_default:instances | 576 |
| lime_default:native_own | 576 |

Joined values are borrowed by instance_id from another bundle and accepted only when the prediction scores match; `native_conflicts_across_bundles` > 0 means the bundles' model fits differ for some instances (joins are still score-checked).

## Files

- `benchmark/outputs/cslaw_denial_r1_rerun/analysis/common_faithfulness_lime_default/common_faithfulness_instances.csv`
- `benchmark/outputs/cslaw_denial_r1_rerun/analysis/common_faithfulness_lime_default/common_faithfulness_summary.csv`
- `benchmark/outputs/cslaw_denial_r1_rerun/analysis/common_faithfulness_lime_default/common_faithfulness_summary.md`
