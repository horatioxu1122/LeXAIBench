# CS&Law re-analysis scripts (LeXAIBench v2)

This directory holds the offline analyses for the ACM CS&Law revision of
LeXAIBench and the synthetic self-test that checks them against the pipeline.

Nothing here fits a model or runs an explainer. The scripts read `results.json`
bundles, write CSV and markdown files into a new `--out-dir`, refuse to overwrite
their inputs, use seeded RNGs only, and never rank the null baselines
(`random_reasons`, `global_ranking`) as leaders. Each script prints the paths it
wrote.

The benchmark runs themselves are produced on the cluster by
`benchmark/scripts/cluster/run_cslaw_credit.sh` (credit) and
`benchmark/scripts/cluster/run_cslaw_noncredit_v2.sh` (non-credit, sharded). The
exact command sequence is in [Cluster command sequence](#cluster-command-sequence).

## LeXAIBench v2: versioned corrections

An audit of v1 found validity problems. Every fix is part of one versioned,
opt-in preset, `v2`, defined in `benchmark/src/legalbench_xai/corrections.py`.
A config selects it in its `cslaw` block:

```json
"cslaw": {"corrections": "v2", "persist_canonical_explanations": true,
          "persist_native_contributions": true, "regb_reference": {"enabled": true}}
```

`{"corrections": {"preset": "v2", "<option>": value}}` overrides single options;
unknown options are an error. Without `cslaw.corrections` (or without a `cslaw`
block) every option keeps its v1 value and the pipeline reproduces the published
v1 bundles exactly; the `regression` step checks this against the baseline commit
`fb5dadc`. A v2 bundle records the options in `payload["cslaw"]["corrections"]`
and `payload["cslaw"]["corrections_version"] = "v2"`; the scripts here read them
per bundle, so v1 and v2 bundles can be analysed side by side.

| | v1 problem | v2 correction (option) | paper claim it supports |
|---|---|---|---|
| a | EBM/XGBoost "contributions" used to rank Anchors candidates and the counterfactual search were a taxonomy-weighted heuristic, not the model | the fitted model's own additive contributions (`native_contributions`): LR coef x standardized value, EBM `eval_terms` (pair terms split 50/50), XGBoost `pred_contribs` | explanations and the Reg B reference reasons are derived from the model that made the decision (comment 9(b)(2)-2: reasons must relate to the factors actually scored) |
| b | faithfulness was self-referential: SHAP, LIME and leave-one-out were scored against their own outputs, Anchors against the proxy restricted to its own features (coverage 1.0 by construction), DiCE/Wachter only structurally; SHAP's quality signal was a constant per model type | every explainer, null baselines included, is scored against one reference: native contributions over all included features (`common_faithfulness_reference`); the self-reported quality term gets weight 0 (`faithfulness_quality_weight`; the signal is still recorded, the remaining weights are renormalized: `(0.45 cov + 0.35 dir + w q)/(0.80 + w)`, which equals the published score at `w = 0.20`) | faithfulness comparisons across explanation families measure agreement with the scoring model, not with the explainer itself; T3 counts them as `open` unless either side agrees with the reference by construction (the T8 flags: SHAP on LR/XGB, native-ranked Anchors), which T3 counts as `decided_by_construction`; under the common reference Anchors' features are scored as direction-neutral, because an anchor rule claims no direction |
| c | DiCE/Wachter "reasons" were a borrowed attribution ranking, identical for both methods | reasons are the features changed by the selected counterfactuals that flip the decision, ranked by how many flipping counterfactuals change them, ties by native contribution; each reason states the sign of the feature's native contribution (`signed_direction`: a changed feature the model credits toward approval reads "supported approval", not "contributed to the denial"), and the share of such non-denial reasons is recorded (`cf_reason_details.nondenial_reason_share`, T1 fact `cf_nondenial_reason_share`); a search where no counterfactual flips the decision gives no reason and `cf_reason_source = "no_valid_change"` (non-flipping changes are kept in `cf_reason_details.non_flipping_change_counts`); a changed feature already cited is not cited again as "changing X may improve the outcome" (`counterfactual_reasons_from_changes`) | counterfactual notices state what the counterfactual method found and are truthful about the model's direction; DiCE and Wachter can now disagree |
| d | features labelled mutability `N` (not actionable) could be changed by the counterfactual search without a hard failure | `N` is treated like `I`: blocked in the search, hard failure if changed (`cf_respect_nonactionable`) | recourse never asks the applicant to change something they cannot or must not change (ALF legality) |
| e | `normalization/rules.py` tested the `PAY_` prefix before `PAY_AMT`, so payment amounts were reported as "recent late payment history" | longest prefix first: `PAY_AMT*` is "recent payment amounts" (`reason_grouping_fix`) | default-credit notices state the correct reason; reason counts and specificity |
| f | `datasets/io.py` left exponent-notation numbers (`8e-05`) as strings, which became spurious categories (Give Me Some Credit) | parsed as numbers (`numeric_parsing_fix`) | model inputs are the recorded values |
| g | sampling took, within each decision-margin stratum, the rows nearest the stratum boundary (scores about 0.50/0.60/0.75), not representative denials; 12 cases per configuration | uniform random draw within the same strata, seeded per dataset-track-model (or per shared pool), independent of the explainer list (`sampling_selection = "stratified_random"`); 24 cases per configuration in the v2 configs | results describe typical denials in each margin band, with twice the precision |
| h | taxonomy: hand-curated, inconsistent primary-track exclusions (South German Credit kept `age`, an S feature; EDUCATION/NumberOfDependents excluded while other proxies were kept; HELOC excluded ExternalRiskEstimate); labels contradicted the manuscript's definitions (bureau-derived counts labelled verifiable `Y`); no recorded rationale; the YAML manifest omitted three datasets | `feature_taxonomy_v2.json` and `dataset_registry_v2.json`: one stated rule (the primary track excludes exactly the sensitivity-`S` features; `P` and `N` stay and are analysed), the label definitions applied consistently, per-feature `rationale`, `legal_basis` and `review_status`, a validator (`validate_taxonomy.py`) and a regenerated manifest. The files are marked as a draft that requires author verification | V/C/CSS, ALF legality and the primary-vs-audit proxy analysis rest on documented, consistently applied labels |
| i | no record of which data and metadata a run used | SHA-256 of every dataset file, of the taxonomy and registry and of the config, plus the git commit, in `payload["cslaw"]["provenance"]` (`record_data_provenance`) | reproducibility; `proxy_substitution.py` and `notices.py` read the metadata files the run actually used |
| j | the non-credit run took about 30 h (22.6 h of it EBM fitting on NY pretrial) | the run is sharded by dataset:track into separate cluster jobs and the shard bundles are merged (`merge_bundles.py`) | practical: the v2 non-credit bundle can be produced within cluster time limits |

`persist_score_subcomponents` additionally records the parts that v1 folded
together: `faithfulness_quality_weight` in each faithfulness block and the three
parts of CSS decision-sensitivity visibility (`dsv_directionality`,
`dsv_threshold_visibility`, `dsv_flip_validity`; S = 0.4 D + 0.3 T + 0.3 F).
This makes the family-neutral CSS_v2 (directionality only) computable.

Comparing v1 and v2 numbers: only method-level conclusions (explainer means and
leaders) can be compared, with `version_diff.py`. Instance-level comparison is
impossible because the sample, the budget, the taxonomy (hence primary-track
feature sets and fitted models) and several metric definitions differ.

## Persisted fields

| field | where | content |
|---|---|---|
| `eval_row_index` | each instance | position in the held-out eval table, which identifies the same applicant across models and tracks |
| `canonical_explanation` | each instance | `CanonicalExplanation.to_dict()` as strict JSON (principal reasons, referenced features, counterfactual changes, metadata; v2 DiCE/Wachter: `metadata.cf_reason_source` and `cf_reason_details`; v2: `metadata.faithfulness_evidence.source = "model_native_common_reference"` with the explainer's own reference under `self_reference_signed_contributions`) |
| `native_contributions` | each instance | `{source, units: "log_odds_adverse", bias, values}`, the fitted model's own per-feature contributions (positive pushes toward denial; `sum(values) + bias` is the raw score) |
| `scores.faithfulness_score.faithfulness_quality_weight` | each instance (v2) | the quality weight used (0 in v2) |
| `scores.contestability_support_score.dsv_*` | each instance (v2) | the three parts of decision-sensitivity visibility |
| `sampling.selection`, `sampling.seed` | each summary row (v2) | `stratified_random` and its seed |
| `payload["cslaw"]["config"]` | top level | normalized echo of the cslaw block |
| `payload["cslaw"]["corrections"]`, `["corrections_version"]` | top level (v2) | the correction options of the run and `"v2"` (or `"custom"`) |
| `payload["cslaw"]["provenance"]` | top level (v2) | `datasets{id: {path, sha256, rows}}`, `metadata{feature_taxonomy, dataset_registry: {path, sha256}}`, `config`, `git_commit` |
| `payload["cslaw"]["groups"]["<dataset>\|<track>\|<model>"]` | top level | `regb_reference` (near-cutoff and all-applicant mean contributions, threshold, row counts), `shared_sampling`, `model_backend`, `native_contribution_source` |
| `payload["merge"]` | merged bundles | shard inputs, order, settings checked, warnings |

Native contribution sources: `lr_coef_x_standardized` (equals the v1
`original_feature_contributions` for LR), `ebm_eval_terms_pair_split`,
`xgb_pred_contribs` (TreeSHAP), and `heuristic_stub` (not a decomposition; the
cluster scripts fail a run if it appears).

## Scripts

| script | what it does | input | main outputs | paper claim it supports |
|---|---|---|---|---|
| `common.py` | Shared contract: field names, scorer weights, reason-list helpers, the v1/v2 correction presets (mirrored from `corrections.py`), corrections/provenance readers, the v1 and corrected temporal grouping. | - | - | - |
| `selftest.py` | **Cluster step 0.** Runs every script's own `--selftest`; checks the cross-agent contract (runner cslaw parsing, model API names, null baselines, `corrections.py` presets/overrides/unknown keys/legacy flag, the faithfulness identity at w = 0.20, the scorer's v2 sub-components, the PAY_AMT grouping fix, exponent parsing, counterfactual reasons from changes, stratified-random sampling; a check whose signature differs is printed as SKIP); builds synthetic v1-shaped and v2-shaped bundles with the runner's own helpers and pushes them through every script, including a three-shard merge that must reassemble the single-run bundle exactly. | none (synthetic only) | prints `cslaw selftest: OK (N checks)`; `--keep DIR` keeps the files | the analysis code reads exactly what the runner writes |
| `rescore.py` | Family-neutral rescoring of stored instance scores: `css_no_dsv`, `comp_no_recourse_simplicity`, `fid_no_quality`, `all_neutral`, and `v2_legal_functions` (CSS_v2 = 0.30V + 0.30C + 0.25 D + 0.15R with D = directionality; comprehensibility without recourse simplicity; the recorded faithfulness when the bundle used the common reference, else `fid_no_quality` with a note that v1 faithfulness stays self-referential; a record that cross-family actionability comparisons are definitional). `--sweep-dsv` sweeps the S weight. | any bundle; bundles without `instance_results` support only `css_no_dsv` and the sweep; `v2_legal_functions` on a v1 bundle falls back (printed) | `<out>/results.json` (then `rebuild-artifacts`), `rescore_by_explainer.csv`, `rescore_summary.md`, `css_dsv_sweep*` | leaders that rely on formula terms only counterfactual methods can earn change under neutral weights |
| `paper_tables.py` | T1 by-construction facts and the correction status of each bundle (corrections, share of instances scored on the common reference, quality weight, counterfactual reason sources); T2 CSS decomposition; T3 pairwise wins split into decided-by-construction / formula-advantaged / not-comparable / open (faithfulness is `open` when both explainers used the common reference and neither carries a T8 by-construction flag); T4 per-domain means with hierarchical bootstrap CIs; T5 correlations; T6 leader robustness; T7 null baselines; **T8 legal-function table** (V, C, D, R, CSS_v2, comprehensibility without recourse simplicity, faithfulness with by-construction flags and an EBM-only column, stability, ALF and actionability within the recourse family) by explainer x dataset and pooled, with hierarchical bootstrap CIs | `--bundle name=dir` (repeatable; `nullbaselines=` enables T7; `credit=` is the T7 reference) | `T1`-`T8` CSVs, `T1_by_construction.md`, `tables.md` | the non-recourse actionability ceiling, the free recourse simplicity, the S bias, how many pairwise "wins" are structural, and which legal functions each method serves |
| `version_diff.py` | v1 vs v2 at the level of explainer means and leaders per metric, pooled over the common datasets and per dataset: leader changes, Kendall tau, flipped pairs, the settings that differ; states that instance-level comparison is impossible | `--v1 DIR --v2 DIR` | `version_diff_{settings,means,leaders,pairs}.csv`, `version_diff_summary.md` | which published method-level conclusions survive the corrections |
| `merge_bundles.py` | Merges shard bundles (one per dataset:track) into one bundle; refuses different settings, duplicate groups, conflicting provenance; recomputes the overview with `runner._build_overview`; prints the `rebuild-artifacts` command | `--inputs DIR [DIR ...] --out-dir DIR [--experiment-name NAME]` | `<out>/results.json` (strict JSON), `<out>/runtime_config.json` unless present | - (enables correction j) |
| `validate_taxonomy.py` | Validates the v2 taxonomy and registry against the stated exclusion rule and label definitions, compares with v1, regenerates the manifest (cluster step `taxonomy`) | `--taxonomy PATH --registry PATH [--compare-v1-taxonomy PATH] [--write-yaml PATH] [--report PATH]` | validation report, manifest | correction h |
| `compare_bundles.py` | Exact comparison of two bundles (ignores timestamps, capabilities, data paths and the cslaw fields; `(ebm, shap)` differences are `EXPECTED_NONDETERMINISTIC`). Exit 1 on any FAILURE. | `LEFT RIGHT` | `--out report.md` | without `cslaw.corrections` the code reproduces the published pipeline |
| `regb_reference.py` | Compares each explainer's principal reasons with our operationalization of the Reg B comment 9(b)(2)-5 methods (M1: approved applicants near the cutoff; M2: all applicants), both from native contributions; M1 vs M2 agreement; notice-level reason units that follow the bundle's temporal grouping. | bundle with canonical + native + `regb_reference` | `regb_reference_*.csv`, `regb_m1_vs_m2.csv`, `regb_reference_summary.md` | whether XAI reasons match the reasons the commentary methods would give, and whether the methods agree |
| `multiplicity.py` | Same applicant, different models: Jaccard and top-reason disagreement across LR/EBM/XGBoost, per explainer and for the Reg B reasons; across-explainer disagreement within a model. | shared-sampling bundle | `multiplicity_*.csv`, `multiplicity_summary.md` | the reasons on a notice partly reflect an arbitrary model choice |
| `proxy_substitution.py` | Primary track vs audit track for the same applicant: what replaces the excluded features in the reasons, and whether proxies gain share. Reads the registry and taxonomy the run recorded in its provenance (v2), warns if they changed, refuses a missing recorded file, and checks the registry against the features the run actually dropped. | shared bundle with scope `models_and_tracks` | `proxy_substitution_*.csv`, `proxy_substitution_summary.md` | dropping protected features moves reasons to proxies |
| `common_faithfulness.py` | Scores every explainer, null baselines included, against the native contributions: mass coverage (also relative to the best achievable with as many features), direction agreement, overlap with the top adverse features, where counterfactual changes land. For v2 bundles it checks that the pipeline's coverage equals the native mass coverage (`pipeline_coverage_abs_diff`). | one or more bundles | `common_faithfulness_*.csv`, `common_faithfulness_summary.md` | v1 built-in faithfulness is self-referential; v2's common reference is wired correctly; null baselines are exposed |
| `notices.py` | Renders about 48 blinded statements of reasons for expert rating; per-bundle taxonomy from provenance; key records `corrections_version` and `cf_reason_source`. | bundle with canonical explanations | `notices_for_rating.csv/.md`, `notices_key.csv`, `ratings_template.csv`, `notices_design.md` | input to the expert-rating study |
| `rating_rubric.md` | Rating items R1-R5, procedure, pre-specified analysis plan. The legal citations need sign-off by the authors. | - | - | - |
| `denial_reasons.py` | Denial-direction correction of the principal reasons (see [Denial-only reasons](#denial-only-reasons-and-the-lime-sampling-check)): keeps the referenced features that increase the denial, rebuilds the reasons with the canonicalizer, re-scores, and excludes explanations left without a reason from the quality metrics. | bundle with canonical explanations (`--config` or the bundle's `runtime_config.json` for the taxonomy) | `<out>/results.json` (then `rebuild-artifacts`), `denial_reasons_summary.md` | reasons on a notice must be reasons for the denial (12 CFR 1002.9(b)(2)) |
| `notice_specificity.py` | E3: mean pairwise Jaccard overlap of the first four reason features across the applicants of each dataset/track/model/method, with the comment 9(b)(2)-5 marginal applicant reasons on the same applicants as the row `regb_reference`; E6: share of denied applicants with at least one decision-flipping counterfactual (DiCE/Wachter), counting applicants that `denial_reasons.py` moved out of the quality metrics. Stratified bootstrap CIs. Offline. | one or more bundles (denial-corrected) | `specificity_configs.csv`, `specificity_pooled.csv`, `specificity_by_track.csv`, `recourse_by_dataset.csv`, `recourse_pooled.csv`, `specificity_recourse.md` | specific reasons (15 U.S.C. 1691(d)(3)); a canned notice scores 1.0 |
| `track_disclosure.py` | E5 at the population level, for bundles whose primary and audit tracks explain different applicants (e.g. HMDA): share of audit-track notices naming a primary-excluded (protected) feature or leading with one; share naming each P-labelled (possible proxy) feature in each track and the change from audit to primary, with bootstrap CIs (tracks resampled independently). For the commentary methods it adds removal baselines on the audit applicants: refill (the protected features left out of the audit model's own ranking, no refit), placebo (as many N-labelled features left out instead), and chance (four reasons drawn at random), with paired differences against the audit notices. Notices are the first four stated reasons (temporal families count once), Reg B reasons grouped the same way. Offline. | one or more bundles (denial-corrected) | `track_disclosure.csv`, `track_disclosure.md` | disclosure of prohibited bases; geographic and income proxies under the FHA (42 U.S.C. 3605) |
| `appendix_tables.py` | Writes the paper's appendix tables as LaTeX fragments (model accuracy, feature taxonomy with legal basis, results by dataset and by model, number of reasons, reference band, LIME sampling, versions, instrument scores of the reference notices, possible proxies against the removal baselines, first reasons within and across model classes when `within_class/` exists) and the worked notices, chosen by fixed rules. Offline. | `--denial-dir` (cslaw_denial_<tag> with analysis/), `--source-dir` and `--tag` (the original v2 bundles, notices before the direction rule) | `<out>/*.tex` | every appendix number regenerates from the run outputs |
| `paper_figures.py` | Draws the three Section 5 figures as vector PDFs: E1 reasons favoring approval before and after the direction rule (signs from the model), E2 against E3 for every notice with the canned, random and marginal applicant references, and E4 first-reason disagreement between the two commentary methods, between two refits of one class on bootstrap resamples (from `within_class/`, left out when absent), and between the boosting machine and the boosted trees on the shared applicants. Writes the plotted values and 95% bootstrap intervals as CSV. Offline. | `--denial-dir`, `--source-dir`, `--tag` (as for `appendix_tables.py`); optional `--qa-dir` for PNG previews and the panel-alignment report | `<out>/fig_*.pdf`, `<out>/fig_*.csv` | every plotted value regenerates from the run outputs |
| `paper_schematic.py` | Draws Figure 1, the pipeline from model to explanation method to notice to the six tests, with the commentary methods and the reference notices as comparison points. Box sizes come from measured label widths; the SVG keeps live text and element ids that follow the figure brief. Reads no data. | `--out-dir`; optional `--preview-dir` for a 600 dpi PNG | `<out>/fig_pipeline.pdf`, `<out>/fig_pipeline.svg` | the schematic regenerates with the other figures |
| `check_additivity.py` | Checks that persisted native contributions plus bias equal the logit of the predicted adverse probability, per dataset/track/model. Offline. | bundles with `native_contributions` | `additivity.csv`, `additivity.md` | the Reg B reference decomposes the model's own score |
| `regb_band_sweep.py` | **Cluster only (refits the credit models).** Varies the marginal applicant band (0.02-0.30, min rows 50/200/500) and reports agreement of the reasons with the 0.10/200 baseline and with each method's notice; checks the baseline against the saved reference means and persisted contributions. | `--config`, `--source-bundle` (v2 credit), `--notice-bundle` (denial-corrected credit) | `band_sweep_rows.csv`, `band_sweep_checks.csv`, `band_sweep.md` | "at or slightly above the minimum passing score" (comment 9(b)(2)-5) |
| `regb_within_class.py` | E4 baseline: refits the boosting machine and the boosted trees of the shared run on bootstrap resamples of the training split (within outcome class; one resample per index shared by both classes) and the boosting machine with other seeds (defaults 20 and 10); keeps refits within --auc-tolerance of the original ROC AUC, and across-class pairs whose ROC AUC also differ by at most that; recomputes the marginal applicant method's reasons for the shared applicants with each fit's own reference means. Reports first-reason disagreement and four-reason overlap within class (seed refits against the original, pairs of bootstrap refits, and bootstrap refits against the original) and across classes (the original pair, which reproduces Figure 4, and each same-resample pair), for every applicant and for those both fits deny, with the share in which a fit approves; each applicant and version counts once; intervals resample refits and applicants within dataset, with paired across-minus-within contrasts. Stops unless the original fits reproduce the persisted contributions and saved means. Writes fits and reasons as they finish (--datasets reruns part, --from-reasons redoes the comparisons). Fits models (cluster). | `--config` (shared run), `--shared-bundle` | `within_class_{fits,reasons,comparisons,summary,contrasts,checks}.csv`, `within_class.md` | whether first reasons depend on the model class or move with any refit |
| `canned_paired.py` | E2 pass rule: each method's share of the marginal applicant method's reasons minus the canned notice's on the same applicants, with a paired bootstrap interval (2,000 resamples within dataset and version, seed 42); includes the LIME robustness runs when present, against the marginal and the average applicant method's reasons. Offline. | `--analysis-dir` (cslaw_denial_<tag>/analysis with the regb_* outputs) | `<out>/canned_paired.{csv,md}` | the E2 differences and intervals in Section 5.2 and Appendix E.4 |
| `reason_families.py` | E4 with correlated factors as one reason: joins reason units whose numeric features reach an absolute Spearman correlation threshold in the data (none, 0.8, 0.7, 0.6; single linkage), relabels the stored first reasons and refit reasons, and reruns the commentary-method, cross-model and within-class comparisons with the within-class bootstrap; with `--shared-bundle` also ranks families by summed contributions. Offline. | `--analysis-dir`, `--data-dir` (credit CSVs), optional `--shared-bundle` | `<out>/families.csv`, `family_*.csv`, `reason_families.md` | threshold none reproduces Figure 4 and the within-class table |
| `run_cslaw_hmda_geo.sh` (scripts/cluster) | HMDA E5 re-run with county and census tract target-encoded (correction `geo_target_encoding`): pilot on `hmda_pilot_fast.csv`, both shards in parallel, merge, direction rule, regb_reference, track_disclosure and specificity into `cslaw_denial_<tag>/analysis/hmda_geo/`. Fits models. | `--tag` | see the script header | the mortgage E5 results with geography as neighborhoods |
| `rating_analysis.py` | Rater agreement, consensus ratings, pre-specified Spearman correlations (Holm-adjusted) between ratings and benchmark metrics. | filled ratings, `notices_key.csv`, optional bundles | `ratings_report.md` and CSVs | whether the metrics track legal judgments |

`regb_reference.py` also hosts the shared helpers (bundle loading, seeded
stratified bootstrap, by-construction flags, reason units, the bundle settings
line) used by `multiplicity.py`, `proxy_substitution.py`,
`common_faithfulness.py` and `paper_tables.py`.

## Denial-only reasons and the LIME sampling check

Two follow-ups to the v2 runs. Neither changes a v1 or v2 run.

**Denial-only principal reasons (`denial_reasons.py`, offline).** Every
explainer ranks features by |weight|, so a canonical principal reason can be a
factor with `signed_direction = "decreases_denial"`, one that favoured approval.
Regulation B requires the specific reasons for the adverse action
(12 CFR 1002.9(b)(2)), and the methods in comment 9(b)(2)-5 count only factors on
which the applicant fell short. The script keeps the referenced features with
`signed_direction = "increases_denial"` (re-ranked), rebuilds the principal
reasons with `ExplanationCanonicalizer._build_principal_reasons` (cap 4, temporal
grouping per the bundle's `reason_grouping_fix`), and re-scores with
`ExplanationScorer` under the bundle's corrections. Counterfactual changes stay
as recourse, but the "changing X may improve the outcome" reasons are not
rebuilt, since they favour approval. Stability and faithfulness evidence are kept;
faithfulness is recomputed on the kept features against the same reference. Each
instance gets `explanation_produced` and `denial_filter` (reasons before and after,
approval-favouring reasons dropped). Instances left without a reason (all reasons
favoured approval, or counterfactual `no_valid_change`) move to
`denial_filter_unproduced_instances` in their summary row. The row's `mean_*`,
`evaluated_rows` and `instance_results` then cover produced explanations only
(`runner._aggregate_group`), with `cslaw_produced_rate` and
`cslaw_evaluated_rows_all` added. A row without produced explanations keeps
`mean_* = None`, which `rebuild-artifacts` skips. An explainer without any
produced explanation is removed and listed in
`payload["denial_reasons"]["dropped_configurations"]`.

The canonicalizer is rebuilt offline from the experiment config, with its
`registry` paths set to the taxonomy and registry recorded in the bundle's
provenance (SHA-256 checked), via `build_dataset_loader(...).build_bundle(track)`,
which reads no data. Before filtering, the script reproduces every persisted
reason list and score from the unfiltered explanation and stops on any
difference (`--allow-inconsistent` turns that into a warning). Limitation: only
the explainer's top referenced features are persisted (6 for SHAP, LIME and the
null baselines), so a denial-only notice can have fewer than four reasons even
when the model has more denial-increasing factors.

```bash
PYTHONPATH=benchmark/src .venv/bin/python benchmark/cslaw/denial_reasons.py \
  --bundle benchmark/outputs/cslaw_v2_credit_r1 \
  --config benchmark/configs/experiment_cslaw_v2_credit.json \
  --out-dir benchmark/outputs/cslaw_analysis_r1/denial_reasons/credit
# then the rebuild-artifacts command the script prints
```

**LIME sampled around the applicant (`limelocal`, opt-in cluster step).** By
default `LimeTabularExplainer` perturbs around the training mean. The correction
option `lime_sample_around_instance` (not part of the v2 preset; recorded in
`payload["cslaw"]["corrections"]` and in each LIME `lime_settings` only when on)
passes `sample_around_instance=True`.
`experiment_cslaw_v2_credit_lime_local.json` runs LIME only under
`{"preset": "v2", "lime_sample_around_instance": true}`. v2 sampling is seeded per
dataset-track-model and does not depend on the explainer list, so its applicants
and `instance_id`s match `cslaw_v2_credit_<tag>`. The bundle records
`corrections_version = "custom"`. The analysis scripts still read it as v2,
because `common.corrections_version` compares only the v1/v2 option keys.

```bash
bash benchmark/scripts/cluster/run_cslaw_credit.sh $REPO --steps limelocal --tag r1
# -> benchmark/outputs/cslaw_v2_credit_lime_local_r1 (checked with cslaw_check_bundle ... v2_lime_local)
```

### Older bundles

The published bundles store scores only. `rescore.py`, `paper_tables.py`,
`version_diff.py` and `compare_bundles.py` work on them. `regb_reference.py`,
`multiplicity.py`, `proxy_substitution.py`, `common_faithfulness.py` and
`notices.py` need the persisted fields; on an older bundle they exit with a
message naming the missing `cslaw` flag.

## By-construction caveats (read before interpreting any agreement)

These are properties of how an explanation or a metric is built, not findings.
The scripts flag them (`by_construction_flags` and `caveats` columns), leave
flagged cells out of leader statements (the headline leader is the best
unflagged real explainer; a higher flagged cell is printed after it as
"by-construction, not ranked"; T8 gives `leader_excluding_by_construction`),
and report null baselines separately.

1. **SHAP decomposes the native quantity.** On logistic regression SHAP differs
   from the native contributions by a constant per feature (its baseline is the
   mean of the first 64 training rows); on XGBoost it equals `pred_contribs`,
   which *is* the native reference. Agreement with M2 or with the common
   reference is therefore near-trivial (`shap_lr_linear_decomposition`,
   `shap_xgb_treeshap_equals_native`). This matters more in v2, where the
   pipeline's own faithfulness uses that reference: a SHAP faithfulness lead on
   LR/XGBoost is by construction. T8's EBM-only faithfulness column is the fair
   comparison. EBM SHAP is in probability units, so only rank and set
   comparisons are used (`shap_ebm_probability_units`).
2. **Ranking inputs.** Under native ranking (the v2 preset, or the round-1 flag
   `use_native_contributions_for_ranking`) Anchors ranks its candidates by the
   native contributions (`anchors_candidates_ranked_by_native`). In v1 bundles
   DiCE and Wachter cite the top ranking contributions (`cf_features_ranked_by_native`;
   always on LR, whose heuristic and native contributions coincide); without
   native ranking, EBM/XGBoost rank by a taxonomy-weighted heuristic
   (`ranking_uses_heuristic_proxy`). In v2 bundles DiCE/Wachter reasons are the
   features their counterfactuals change, so they carry only the caveat
   `cf_reason_order_from_native`: features changed equally often are ordered,
   and weighted, by native contribution, so rank-based metrics (top-1) still
   partly reflect the native order. Each v2 DiCE/Wachter reason also takes its
   `signed_direction` from the sign of the native contribution, which is the
   common faithfulness reference itself. Its directional agreement would
   therefore be 1.0 by construction (35/80 = 44% of v2 faithfulness). Under
   `common_faithfulness_reference` the scorer counts counterfactual directions
   as neutral (0.5), the same as Anchors (whose `increases_denial` label is
   hard-coded). The reasons and CSS keep `signed_direction`. The direction term
   therefore separates only the attribution explainers (SHAP, LIME, null
   baselines), and `common_faithfulness.py` leaves `direction_agreement` empty
   for v2 counterfactual instances.
3. **Null baselines.** `global_ranking` gives every applicant the same reasons
   (mean |native contribution| over the background rows), so its stability is
   about 1.0 by construction. In v1 bundles with `faithfulness_reference="self"`,
   `random_reasons` gets directional agreement 1.0 against its own weights,
   which makes the self-reference flaw of the v1 faithfulness visible. In v2
   bundles the null baselines are scored against the common native reference
   like every other explainer. The v2 configs keep
   `cslaw.null_baselines.faithfulness_reference = "self"` on purpose. The common
   reference overrides it for scoring, and the null baseline's own weights stay
   in `faithfulness_evidence.self_reference_signed_contributions`. So the
   `"self"` in the recorded `cslaw.config` does not mean the scores are
   self-referential.
4. **Metric structure.** Non-recourse explanations cannot exceed unified
   actionability 0.35 (advice 0.25, ALF component 0), so cross-family
   actionability comparisons are definitional; T8 reports ALF and actionability
   within the recourse family only. Non-counterfactual explanations get
   recourse simplicity 1.0. Decision-sensitivity visibility (S) favors
   counterfactual methods through threshold visibility and flip validity; CSS_v2
   keeps only its directionality part. `rescore.py`, T1-T3 and T8 quantify
   each effect. Faithfulness coverage depends on how many features are cited.
   The v2 coverage term (`reference_feature_coverage`) is the share of native
   |mass| on the cited features. SHAP, LIME and the null baselines always cite 6
   features, while Wachter (often 1-2 changed features) and Anchors cite fewer.
   Sparse explainers therefore score lower in `faithfulness_score` and in T8
   `faithfulness_legal` by construction. Only `common_faithfulness.py`'s
   `mass coverage / oracle` normalizes for the number of features cited. Read
   cross-family faithfulness gaps with that in mind.
5. **Native contributions are modelling choices** for XGBoost (TreeSHAP) and for
   EBM pair terms (split 50/50). The Reg B M1/M2 operationalization (largest
   positive gap between the applicant's contribution and the reference mean) is
   ours, not a regulator-endorsed algorithm.
6. **Run-to-run comparability.** EBM SHAP is nondeterministic across processes.
   LIME draws from a cached per-model RandomState whose stream depends on call
   order. Join bundles by `instance_id` only when they come from the same
   corrections version, node type and environment; the scripts check
   prediction scores. Never join v1 and v2 instances.
7. **Data and labels.** The taxonomy has no proxy-strength field: `P` is a
   hand-assigned label. The v2 taxonomy and registry are marked as a draft that
   requires author verification (`status`/`review_status` in the files and the
   change log they reference); cite the labels as the authors'
   reviewed coding only after that review. In v2, HELOC and FDA have no `S`
   feature, so their primary and audit tracks use the same features and the
   track comparison is a no-op there. The paper must not read a HELOC or FDA
   primary-vs-audit difference as evidence (any difference is sampling noise). v1 bundles keep the v1 issues listed in
   the corrections table (e.g. `PAY_AMT*` grouped as late payment history); the
   scripts mirror the grouping of whichever version produced the bundle.
8. **Regression scope.** `experiment_cslaw_regression_check.json` (the A/B run
   against `fb5dadc`) covers only South German Credit and HELOC. The v1-path
   invariance of the `PAY_AMT*` grouping (Default of Credit Card Clients) and
   of exponent-notation parsing (Give Me Some Credit) is checked by code
   inspection and `selftest.check_pure_fixes`, not by the A/B run. Both fixes
   sit behind flags (`reason_grouping_fix`, `numeric_parsing_fix`) whose False
   branch is the old code.
9. **Implementation details that go beyond the contract wording.**
   (a) With shared-sampling scope `models`, the seed key is
   `dataset|shared|models|<tracks>`; scope `models_and_tracks` uses
   `dataset|shared|models_and_tracks` exactly. (b) DiCE/Wachter reasons count
   only counterfactuals that flip the decision; change counts from
   non-flipping counterfactuals are kept in `cf_reason_details`.
   (c) `provenance` has the extra keys `config`, `bytes` and `source_kind`.
   (d) HELOC and FDA have identical primary and audit tracks in v2 (item 7 and
   `benchmark/metadata/TAXONOMY_V2_CHANGES.md`).

## Cluster command sequence

All steps run from a checkout of the `cslaw-reanalysis` branch on the cluster.
Every output goes to a new `benchmark/outputs/cslaw_*_<TAG>/` directory. The
wrappers refuse an existing directory unless `--force` is given, and never
write into a published bundle.

### Before the cluster: commit and push (or rsync)

The cluster checkout only has these changes once they are committed and pushed.
On the machine with the changes:

```bash
git switch cslaw-reanalysis
git status --short   # review; expect no __pycache__, benchmark/outputs or benchmark/data changes
# src (incl. the new corrections.py), the scripts here, the new v2 metadata files (the v1
# files are unchanged), the regenerated v2 manifest (listed in the v2 registry's
# paper_sources and in the v2 configs; validate_taxonomy.py fails without it), the
# cluster scripts and the cslaw configs:
git add benchmark/src/legalbench_xai benchmark/cslaw benchmark/metadata benchmark/scripts \
  paper/feature_taxonomy_manifest_v2.yaml \
  benchmark/configs/experiment_cslaw_v2_credit.json \
  benchmark/configs/experiment_cslaw_v2_credit_nullbaselines.json \
  benchmark/configs/experiment_cslaw_v2_credit_shared.json \
  benchmark/configs/experiment_cslaw_v2_credit_lime_local.json \
  benchmark/configs/experiment_cslaw_regression_check.json
git status --short   # nothing the run needs may remain untracked ("??")
git commit -m "LeXAIBench v2: versioned corrections, v2 taxonomy, sharded non-credit run, analyses"
git push -u origin cslaw-reanalysis
```

`__pycache__/` is in `.gitignore`. On the cluster, run
`git fetch origin && git switch cslaw-reanalysis` (or `git pull`). Do not use a
shallow clone: the `regression` step needs the baseline commit `fb5dadc`. The
wrappers record the commit in every v2 bundle (`provenance.git_commit`, with a
`-dirty` suffix for uncommitted code, configs or metadata), so commit first. To
copy the files instead (then `GIT_COMMIT` cannot be read from git; export it
yourself):

```bash
rsync -aR --exclude __pycache__ benchmark/src/legalbench_xai benchmark/cslaw \
  benchmark/configs benchmark/metadata benchmark/scripts \
  paper/feature_taxonomy_manifest_v2.yaml <cluster>:explainableAI-CS/
```

### 1. Credit: checks, taxonomy, the three v2 runs

```bash
REPO=$HOME/explainableAI-CS
# 0 + 1 + taxonomy: synthetic self-test, A/B regression (fb5dadc vs current code, no cslaw
# block), validation of the v2 taxonomy and registry. Stop and read the outputs if any fails.
bash benchmark/scripts/cluster/run_cslaw_credit.sh $REPO --steps selftest,regression,taxonomy --tag r1

# The three v2 credit runs (long; real models and explainers). Separate jobs are fine:
bash benchmark/scripts/cluster/run_cslaw_credit.sh $REPO --steps credit --tag r1
bash benchmark/scripts/cluster/run_cslaw_credit.sh $REPO --steps nullbaselines --tag r1
bash benchmark/scripts/cluster/run_cslaw_credit.sh $REPO --steps shared --tag r1

# Offline rescoring of the published v1 bundles (fits nothing; can also run on a laptop)
bash benchmark/scripts/cluster/run_cslaw_credit.sh $REPO --steps offline --tag r1
```

The selftest prints `cslaw selftest: OK (N checks)`; a line `SKIP <check>`
means a pipeline function no longer has the signature the contract describes,
and should be looked at before the long runs.

Every wrapper step runs `uv sync --extra benchmark-empirical` in the same
checkout. If you submit several jobs at once (the credit steps, the six
non-credit shards), run that sync once on the login node first:

```bash
cd $REPO && uv sync --extra benchmark-empirical
```

The per-job syncs then find `.venv` up to date and do not rewrite it at the
same time.

### 2. Non-credit: six shard jobs, then the merge

One job per dataset:track shard (HMDA, NY Courts pretrial, FDA inspections x
primary, audit). The NY pretrial EBM fits dominate the runtime, so give
`ny_courts_pretrial:*` the longest wall time.

```bash
for shard in hmda:primary hmda:audit ny_courts_pretrial:primary ny_courts_pretrial:audit \
             fda_inspections:primary fda_inspections:audit; do
  # submit each as its own job, e.g. sbatch --wrap "bash ... --shard ${shard} --tag r1"
  bash benchmark/scripts/cluster/run_cslaw_noncredit_v2.sh $REPO --shard "${shard}" --tag r1
done
# when all six have finished:
bash benchmark/scripts/cluster/run_cslaw_noncredit_v2.sh $REPO --merge --tag r1
```

Shard outputs: `benchmark/outputs/cslaw_v2_noncredit_<dataset>_<track>_r1/`;
merged bundle: `benchmark/outputs/cslaw_v2_noncredit_r1/`. `--merge` calls
`merge_bundles.py` on the shards that exist (a loud warning lists missing ones;
redo the merge with `--force` after they finish), writes the merged
`runtime_config.json`, runs `rebuild-artifacts`, and checks the merged bundle.
`--all-sequential` runs the six shards and the merge in one job (about 30 h).
`--light-ebm` is faster but changes the EBM model; its rows are not comparable
with the published bundle, and the merge refuses shards that mix it.

The merge itself, as the wrapper runs it:

```bash
PYTHONPATH=benchmark/src .venv/bin/python benchmark/cslaw/merge_bundles.py \
  --inputs benchmark/outputs/cslaw_v2_noncredit_{hmda,ny_courts_pretrial,fda_inspections}_{primary,audit}_r1 \
  --out-dir benchmark/outputs/cslaw_v2_noncredit_r1 --experiment-name legalbench_xai_cslaw_v2_noncredit
```

### 3. Analysis (after steps 1 and 2)

```bash
bash benchmark/scripts/cluster/run_cslaw_credit.sh $REPO --steps analysis --tag r1
```

The analysis step uses `cslaw_v2_noncredit_r1` when it exists, otherwise the
published v1 non-credit bundle with a warning (the non-credit columns are then
v1; run the merge first, then repeat the step with `--force`). If you rerun only
the analysis after adding the non-credit bundle, pass `--force`.

### What each step runs

All commands run from the repository root with `PY=.venv/bin/python` and
`export PYTHONPATH=benchmark/src`; the wrapper adds thread pinning, progress
logs, `GIT_COMMIT`, and the bundle checks.

| step | command(s) | output |
|---|---|---|
| `selftest` | `$PY benchmark/cslaw/selftest.py` | console only |
| `regression` | worktree of `fb5dadc`; `run-demo-benchmark --config benchmark/configs/experiment_cslaw_regression_check.json --write --summary-only` with the baseline code and with this checkout; `compare_bundles.py <baseline> <new>`; both bundles must carry no `payload["cslaw"]`, no per-instance cslaw field and no v2-only key | `cslaw_regression_{baseline,new,compare}_r1/` |
| `taxonomy` | `$PY benchmark/cslaw/validate_taxonomy.py --taxonomy benchmark/metadata/feature_taxonomy_v2.json --registry benchmark/metadata/dataset_registry_v2.json --compare-v1-taxonomy benchmark/metadata/feature_taxonomy.json --write-yaml <analysis>/feature_taxonomy_manifest_v2.yaml --report <analysis>/taxonomy_validation.md` | `cslaw_analysis_r1/taxonomy_validation.md`, manifest |
| `credit` | `$PY -m legalbench_xai run-demo-benchmark --config benchmark/configs/experiment_cslaw_v2_credit.json --output-dir benchmark/outputs/cslaw_v2_credit_r1 --write --summary-only` (corrections v2, 5 explainers, 24 cases per configuration) | `cslaw_v2_credit_r1/` |
| `nullbaselines` | same with `experiment_cslaw_v2_credit_nullbaselines.json` (`random_reasons`, `global_ranking`; same corrections and sampling, so instance ids match `credit`) | `cslaw_v2_credit_nullbaselines_r1/` |
| `shared` | same with `experiment_cslaw_v2_credit_shared.json` (shared sampling, scope `models_and_tracks`) | `cslaw_v2_credit_shared_r1/` |
| `limelocal` (opt-in, not in the default list) | same with `experiment_cslaw_v2_credit_lime_local.json` (LIME only, corrections v2 + `lime_sample_around_instance`; same applicants as `credit`); bundle check `v2_lime_local` (corrections_version `custom`, the option recorded, `lime_settings.sample_around_instance` in every LIME explanation) | `cslaw_v2_credit_lime_local_r1/` |
| `offline` | on the published v1 bundles: `rescore.py --variant V` for `css_no_dsv comp_no_recourse_simplicity fid_no_quality all_neutral` + `rebuild-artifacts`; `rescore.py --sweep-dsv ...`; `paper_tables.py` | `cslaw_offline_r1/` |
| `analysis` | `paper_tables.py --bundle credit=cslaw_v2_credit_r1 --bundle noncredit=<cslaw_v2_noncredit_r1 or published> --bundle nullbaselines=cslaw_v2_credit_nullbaselines_r1 --robustness-root benchmark/outputs --rescored-root <offline>/rescored --sweep-csv ...`; `version_diff.py --v1 <published credit> --v2 cslaw_v2_credit_r1`; `regb_reference.py` on credit and shared; `common_faithfulness.py` on credit + null baselines and on shared; `multiplicity.py` and `proxy_substitution.py` on shared; `notices.py --bundle <credit> --n 48 --seed 7`; `rescore.py --variant v2_legal_functions` on credit + `rebuild-artifacts`. For a v2 non-credit bundle also `regb_reference.py`, `common_faithfulness.py`, `notices.py` (HMDA), `version_diff.py --v1 <published non-credit>` and `v2_legal_functions` | `cslaw_analysis_r1/{paper_tables,version_diff,regb_reference_credit,regb_reference_shared,common_faithfulness,common_faithfulness_shared,multiplicity,proxy_substitution,notices,rescored_v2/credit/v2_legal_functions}/` plus the `*_noncredit` counterparts |

After each benchmark run the wrapper calls `cslaw_check_bundle`, which exits
with code 3 on any stub backend, a missing `payload["cslaw"]`, a
`heuristic_stub` native source or recorded cslaw errors; for v2 runs also unless
the bundle records `corrections_version = "v2"`, provenance with SHA-256 hashes
of the v2 taxonomy and registry, stratified-random sampling in every group, the
persisted v2 score keys, faithfulness scored on the common native reference
(`evidence_source = "model_native_common_reference"`) in every instance, and a
`cf_reason_source` in every DiCE/Wachter canonical explanation. The last two
catch a run in which the corrections reached the scorer but not the explainers.

Run the `credit` step before the non-credit shards. It is the first run that fits
real EBM and XGBoost models and decomposes them natively, which v2 needs for every
explanation, so a library problem shows up within minutes there instead of after a
NY pretrial EBM fit of about 11 h. The optional additivity snippet below checks the
same thing in a few minutes.

### Which bundle the tables describe

Say which bundle a number comes from:

- `cslaw_analysis_r1/paper_tables/`: T1-T5, T7 and T8 describe the v2 runs
  (`credit` = `cslaw_v2_credit_r1`, `noncredit` = `cslaw_v2_noncredit_r1` if it
  existed, else the published v1 bundle; check the `corrections` line in the
  Inputs section of `tables.md`). T6 compares a robustness bundle only with a
  baseline of the same corrections version: a published v1 bundle whose only
  covering baseline is the v2 main bundle is skipped (T6 note) and appears in
  the offline tables instead; every T6 note records both versions. The rescored
  variants describe the published v1 bundles; `cslaw_*` rows in T6 are re-runs,
  null baselines or a different sampling design, not robustness variants.
- `cslaw_offline_r1/paper_tables/`: the published v1 bundles only.
- `cslaw_analysis_r1/version_diff*/`: which method-level conclusions change from
  v1 to v2.
- `cslaw_analysis_r1/rescored_v2/*/v2_legal_functions/`: the v2 bundles under
  the legal-function weights (the report and leaderboard rebuilt there).

The offline and analysis steps should run on the same machine; otherwise rerun
`offline` there first (`rescore.py` records its source bundle as an absolute
path; `paper_tables.py` falls back to a same-named directory under
`--robustness-root` when the experiment name matches, and logs it in the T6
notes).

### Expert-rating exercise (manual, after the analysis step)

1. Give the raters `cslaw_analysis_r1/notices/notices_for_rating.md`,
   `ratings_template.csv` and `benchmark/cslaw/rating_rubric.md`. Do not show
   them `notices_key.csv` or `notices_design.md`. Do not mix v1 and v2 notices
   in one packet (the design file warns).
   - Practice round: `$PY benchmark/cslaw/notices.py --bundle <credit> --out-dir <dir>/practice --n 4 --seed 999`.
   - Reasons-only condition: add `--strip-recourse`, with a separate `--out-dir`.
2. When the filled copies are back:

```bash
$PY benchmark/cslaw/rating_analysis.py --ratings ratings_A.csv ratings_B.csv [ratings_C.csv] \
  --key benchmark/outputs/cslaw_analysis_r1/notices/notices_key.csv \
  --bundle benchmark/outputs/cslaw_v2_credit_r1 \
  --out-dir benchmark/outputs/cslaw_analysis_r1/rating_analysis
```

If the authors do the rating, the paper must describe it as a legal assessment
by the authors, not as a human-subjects study.

### Optional: check native additivity on the real fitted models

The self-test uses a fake additive model. This snippet checks the real fitted
models on the cluster (it fits three models on two small credit datasets):
`sum(native) + bias` should reproduce each model's raw score to about 1e-9 for
LR and EBM and to float32 precision for XGBoost (compared against the logit of
its probability, so very confident rows show larger gaps), and LR's native
values should equal its heuristic ones.

```bash
PYTHONPATH=benchmark/src .venv/bin/python - <<'PY'
import math
from legalbench_xai.config import load_experiment_config
from legalbench_xai.datasets.registry import build_dataset_loader
from legalbench_xai.models import fit_model
from legalbench_xai.runner import _split_table_for_evaluation

config = load_experiment_config("benchmark/configs/experiment_cslaw_v2_credit.json")
for dataset_id in ("south_german_credit", "heloc"):
    loader = build_dataset_loader(config, dataset_id)
    bundle = loader.build_bundle("audit")
    table = loader.load_tabular_dataset(track="audit")
    train, evaluation, _ = _split_table_for_evaluation(
        table=table, negative_label=bundle.negative_label, evaluation_cfg=config.evaluation_protocol
    )
    rows = evaluation.rows[:200]
    for model_id in ("logistic_regression", "ebm", "xgboost"):
        model, backend = fit_model(
            model_id=model_id, dataset_bundle=bundle, table=train, threshold=config.model_threshold
        )
        values, bias = model.native_feature_contributions_batch(rows)
        predictions = model.predict_rows(rows, instance_id_prefix="check")
        worst = 0.0
        for item, prediction in zip(values, predictions):
            p = min(max(prediction.score, 1e-12), 1 - 1e-12)
            raw = prediction.raw_score if model_id != "xgboost" else math.log(p / (1 - p))
            worst = max(worst, abs(sum(item.values()) + bias - raw))
        line = (
            f"{dataset_id} {model_id} {backend.implementation} "
            f"{model.native_contribution_source} max|sum+bias-raw|={worst:.2e}"
        )
        if model_id == "logistic_regression":
            same = all(
                model.original_feature_contributions(row) == item for row, item in zip(rows, values)
            )
            line += f" native==original:{same}"
        print(line)
PY
```

## Running the self-tests individually

```bash
PYTHONPATH=benchmark/src .venv/bin/python benchmark/cslaw/selftest.py   # runs all of the below, plus more
PYTHONPATH=benchmark/src .venv/bin/python benchmark/cslaw/rescore.py --selftest
PYTHONPATH=benchmark/src .venv/bin/python benchmark/cslaw/denial_reasons.py --selftest
PYTHONPATH=benchmark/src .venv/bin/python benchmark/cslaw/paper_tables.py --selftest
.venv/bin/python benchmark/cslaw/compare_bundles.py --selftest
.venv/bin/python benchmark/cslaw/notices.py --selftest
.venv/bin/python benchmark/cslaw/rating_analysis.py --selftest
```
