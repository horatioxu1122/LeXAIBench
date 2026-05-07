# LegalBench-XAI Results Memo

- Experiment: `legalbench_xai_default_credit_real_threshold060`
- Generated at (UTC): `2026-03-15T19:20:35.917139+00:00`

## Headline Findings

- Overall faithfulness leader: `wachter` with mean faithfulness `0.956` and `[0.936, 0.973]` bootstrap CI.
- Overall stability leader: `lime` with mean stability `0.906` and `[0.862, 0.958]` bootstrap CI.
- Overall comprehensibility leader: `anchors` with mean comprehensibility `0.986` and `[0.977, 0.995]` bootstrap CI.
- Overall CSS leader: `dice` with mean CSS `0.985` and `[0.980, 0.992]` bootstrap CI.
- Overall actionability leader: `dice` with mean actionability `0.742` and `[0.736, 0.750]` bootstrap CI.
- Overall ALF leader: `dice` with mean ALF `0.511` and `[0.500, 0.519]` bootstrap CI.
- Pairwise faithfulness leader: `wachter` with `19` matched-configuration wins.
- Pairwise stability leader: `lime` with `23` matched-configuration wins.
- Pairwise comprehensibility leader: `anchors` with `24` matched-configuration wins.
- Pairwise CSS leader: `wachter` with `18` matched-configuration wins.
- Pairwise actionability leader: `wachter` with `22` matched-configuration wins.
- Pairwise ALF leader: `wachter` with `4` matched-configuration wins.

## Draft Results Language

Across the current benchmark run, faithfulness, stability, comprehensibility, and applicant-facing usefulness do not collapse into a single ranking. `wachter` leads on mean faithfulness with `0.956`, `lime` leads on mean stability with `0.906`, `anchors` leads on mean comprehensibility with `0.986`, while recourse-oriented explainers outperform attribution-style methods on applicant-facing metrics. `dice` leads on overall contestability support with mean CSS `0.985`, while `dice` leads on overall actionability with `0.742` and also achieves the strongest legality-aware recourse performance with mean ALF `0.511`. Primary-vs-audit score shifts remain small on average, with the largest mean absolute shift on `stability` at `0.058`. Primary and audit track leader identities are stable for `4` of `6` metrics, with instability concentrated in `faithfulness`, `stability`. The least stable metric is `faithfulness` at same-leader rate `0.333`.

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

- These outputs come from a fully real benchmark run on local real dataset files. Model backends in use: `ebm_classifier` x 10, `sklearn_logistic_regression` x 10, `xgboost_classifier` x 10. Explainer backends in use: `lime_library` x 6, `search_anchor_rule` x 6, `search_counterfactual_dice_style` x 6, `search_counterfactual_wachter_style` x 6, `shap_library` x 6. Treat this as the first real empirical slice, not yet as submission-ready evidence.
