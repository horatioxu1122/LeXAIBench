# LegalBench-XAI Results Memo

- Experiment: `legalbench_xai_four_dataset_real_six_explainers`
- Generated at (UTC): `2026-03-16T02:04:52.458603+00:00`

## Headline Findings

- Overall faithfulness leader: `leave_one_out` with mean faithfulness `0.928` and `[0.891, 0.961]` bootstrap CI.
- Overall stability leader: `lime` with mean stability `0.915` and `[0.878, 0.954]` bootstrap CI.
- Overall comprehensibility leader: `anchors` with mean comprehensibility `0.944` and `[0.918, 0.975]` bootstrap CI.
- Overall CSS leader: `dice` with mean CSS `0.895` and `[0.834, 0.959]` bootstrap CI.
- Overall actionability leader: `wachter` with mean actionability `0.679` and `[0.633, 0.733]` bootstrap CI.
- Overall ALF leader: `wachter` with mean ALF `0.492` and `[0.452, 0.525]` bootstrap CI.
- Pairwise faithfulness leader: `leave_one_out` with `97` matched-configuration wins.
- Pairwise stability leader: `lime` with `98` matched-configuration wins.
- Pairwise comprehensibility leader: `anchors` with `120` matched-configuration wins.
- Pairwise CSS leader: `dice` with `102` matched-configuration wins.
- Pairwise actionability leader: `wachter` with `110` matched-configuration wins.
- Pairwise ALF leader: `wachter` with `14` matched-configuration wins.

## Draft Results Language

Across the current benchmark run, faithfulness, stability, comprehensibility, and applicant-facing usefulness do not collapse into a single ranking. `leave_one_out` leads on mean faithfulness with `0.928`, `lime` leads on mean stability with `0.915`, `anchors` leads on mean comprehensibility with `0.944`, while recourse-oriented explainers outperform attribution-style methods on applicant-facing metrics. `dice` leads on overall contestability support with mean CSS `0.895`, while `wachter` leads on overall actionability with `0.679` and also achieves the strongest legality-aware recourse performance with mean ALF `0.492`. Primary-vs-audit score shifts remain small on average, with the largest mean absolute shift on `stability` at `0.051`. Primary and audit track leader identities are stable for `2` of `6` metrics, with instability concentrated in `faithfulness`, `stability`, `actionability`, `alf`. The least stable metric is `stability` at same-leader rate `0.583`.

## Suggested Claims to Validate in the Full Benchmark

- Faithfulness should be discussed separately from contestability, because a method can align well with model behavior and still be weak as applicant-facing guidance.
- Stability should be discussed separately from faithfulness, because a method can match model behavior on one instance while still being brittle across similar applicants or random seeds.
- Comprehensibility should be discussed separately from both faithfulness and actionability, because the easiest explanation to read may not be the most faithful or the most useful for recourse.
- Recourse-oriented methods appear to dominate attribution methods on applicant-facing actionability.
- Contestability-support rankings are close enough that CSS should be discussed as a tradeoff, not only as a single leaderboard.
- Primary-versus-audit track effects are limited in the current benchmark run, but this should be retested on full datasets.
- Pairwise win tables are likely to be more persuasive than raw averages when arguing robustness across datasets and model classes.

## Uncertainty Note

- `95%` hierarchical-bootstrap confidence intervals with `1000` resamples over `dataset_id -> track -> model_id` benchmark clusters are included in the analysis bundle.
- These intervals are computed by resampling the benchmark structure rather than treating all scored groups as IID, so they are useful for robustness checks but are still not full applicant-level or fully model-based uncertainty estimates.

## Caveat

- These outputs come from a fully real benchmark run on local real dataset files. Model backends in use: `ebm_classifier` x 48, `sklearn_logistic_regression` x 48, `xgboost_classifier` x 48. Explainer backends in use: `leave_one_out_feature_ablation` x 24, `lime_library` x 24, `search_anchor_rule` x 24, `search_counterfactual_dice_style` x 24, `search_counterfactual_wachter_style` x 24, `shap_library` x 24. Treat this as the first real cross-dataset empirical bundle, not yet as submission-ready evidence.
