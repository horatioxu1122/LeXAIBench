# LegalBench-XAI Results Memo

- Experiment: `legalbench_xai_default_credit_real_highbudget24`
- Generated at (UTC): `2026-03-15T19:03:51.287680+00:00`

## Headline Findings

- Overall faithfulness leader: `anchors` with mean faithfulness `0.930` and `[0.904, 0.954]` bootstrap CI.
- Overall stability leader: `lime` with mean stability `0.902` and `[0.839, 0.958]` bootstrap CI.
- Overall comprehensibility leader: `anchors` with mean comprehensibility `0.986` and `[0.975, 0.995]` bootstrap CI.
- Overall CSS leader: `dice` with mean CSS `0.985` and `[0.981, 0.990]` bootstrap CI.
- Overall actionability leader: `dice` with mean actionability `0.748` and `[0.744, 0.753]` bootstrap CI.
- Overall ALF leader: `dice` with mean ALF `0.523` and `[0.518, 0.530]` bootstrap CI.
- Pairwise faithfulness leader: `dice` with `15` matched-group wins.
- Pairwise stability leader: `lime` with `23` matched-group wins.
- Pairwise comprehensibility leader: `anchors` with `24` matched-group wins.
- Pairwise CSS leader: `wachter` with `18` matched-group wins.
- Pairwise actionability leader: `dice` with `23` matched-group wins.
- Pairwise ALF leader: `dice` with `5` matched-group wins.

## Draft Results Language

Across the current benchmark run, faithfulness, stability, comprehensibility, and applicant-facing usefulness do not collapse into a single ranking. `anchors` leads on mean faithfulness with `0.930`, `lime` leads on mean stability with `0.902`, `anchors` leads on mean comprehensibility with `0.986`, while recourse-oriented explainers outperform attribution-style methods on applicant-facing metrics. `dice` leads on overall contestability support with mean CSS `0.985`, while `dice` leads on overall actionability with `0.748` and also achieves the strongest legality-aware recourse performance with mean ALF `0.523`. Primary-vs-audit score shifts remain small on average, with the largest mean absolute shift on `stability` at `0.052`. Primary and audit track leader identities are stable for `2` of `6` metrics, with instability concentrated in `faithfulness`, `stability`, `actionability`, `alf`. The least stable metric is `faithfulness` at same-leader rate `0.667`.

## Suggested Claims to Validate in the Full Benchmark

- Faithfulness should be discussed separately from contestability, because a method can align well with model behavior and still be weak as applicant-facing guidance.
- Stability should be discussed separately from faithfulness, because a method can match model behavior on one instance while still being brittle across similar applicants or random seeds.
- Comprehensibility should be discussed separately from both faithfulness and actionability, because the easiest explanation to read may not be the most faithful or the most useful for recourse.
- Recourse-oriented methods appear to dominate attribution methods on applicant-facing actionability.
- Contestability-support rankings are close enough that CSS should be discussed as a tradeoff, not only as a single leaderboard.
- Primary-versus-audit track effects are limited in the current benchmark run, but this should be retested on full datasets.
- Pairwise win tables are likely to be more persuasive than raw averages when arguing robustness across datasets and model families.

## Uncertainty Note

- `95%` hierarchical-bootstrap confidence intervals with `1000` resamples over `dataset_id -> track -> model_id` benchmark clusters are included in the analysis bundle.
- These intervals are computed by resampling the benchmark structure rather than treating all scored groups as IID, so they are useful for robustness checks but are still not full applicant-level or fully model-based uncertainty estimates.

## Caveat

- These outputs come from a fully real benchmark run on local real dataset files. Model backends in use: `ebm_classifier` x 10, `sklearn_logistic_regression` x 10, `xgboost_classifier` x 10. Explainer backends in use: `lime_library` x 6, `search_anchor_rule` x 6, `search_counterfactual_dice_style` x 6, `search_counterfactual_wachter_style` x 6, `shap_library` x 6. Treat this as the first real empirical slice, not yet as submission-ready evidence.
