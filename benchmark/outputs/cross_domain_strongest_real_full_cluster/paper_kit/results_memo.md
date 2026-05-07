# LegalBench-XAI Results Memo

- Experiment: `legalbench_xai_cross_domain_strongest_real_template`
- Generated at (UTC): `2026-04-26T23:43:09.734784+00:00`

## Headline Findings

- Overall faithfulness leader: `shap` with mean faithfulness `0.961` and `[0.930, 0.990]` bootstrap CI.
- Overall stability leader: `shap` with mean stability `0.936` and `[0.881, 0.979]` bootstrap CI.
- Overall comprehensibility leader: `anchors` with mean comprehensibility `0.942` and `[0.921, 0.969]` bootstrap CI.
- Overall CSS leader: `dice` with mean CSS `0.796` and `[0.677, 0.910]` bootstrap CI.
- Overall actionability leader: `wachter` with mean actionability `0.469` and `[0.305, 0.573]` bootstrap CI.
- Overall ALF leader: `wachter` with mean ALF `0.329` and `[0.192, 0.591]` bootstrap CI.
- Pairwise faithfulness leader: `shap` with `66` matched-configuration wins.
- Pairwise stability leader: `shap` with `61` matched-configuration wins.
- Pairwise comprehensibility leader: `anchors` with `66` matched-configuration wins.
- Pairwise CSS leader: `dice` with `60` matched-configuration wins.
- Pairwise actionability leader: `dice` with `60` matched-configuration wins.
- Pairwise ALF leader: `wachter` with `6` matched-configuration wins.

## Draft Results Language

Across the current benchmark run, faithfulness, stability, comprehensibility, and subject/entity-facing usefulness do not collapse into a single ranking. `shap` leads on mean faithfulness with `0.961`, `shap` leads on mean stability with `0.936`, `anchors` leads on mean comprehensibility with `0.942`, while recourse-oriented explainers outperform attribution-style methods on subject/entity-facing metrics. `dice` leads on overall contestability support with mean CSS `0.796`, while `wachter` narrowly leads on mean actionability with `0.469` versus `dice` at `0.468` and also achieves the strongest legality-aware recourse performance with mean ALF `0.329`. Pairwise actionability favors `dice`, so actionability should be described as a close contest rather than a decisive mean-score win. Primary-vs-audit score shifts remain small on average, with the largest mean absolute shift on `stability` at `0.114`. Primary and audit track leader identities are stable for `2` of `6` metrics, with instability concentrated in `faithfulness`, `stability`, `comprehensibility`, `actionability`. The least stable metric is `faithfulness` at same-leader rate `0.556`.

## Manuscript-Ready Interpretive Claims

- Faithfulness should be discussed separately from contestability, because a method can align well with model behavior and still be weak as subject/entity-facing guidance.
- Stability should be discussed separately from faithfulness, because a method can match model behavior on one instance while still being brittle across similar applicants or random seeds.
- Comprehensibility should be discussed separately from both faithfulness and actionability, because the easiest explanation to read may not be the most faithful or the most useful for recourse.
- Recourse-oriented methods dominate attribution methods on actionability and legality-aware recourse in this cross-domain bundle, but Wachter's mean actionability edge over DiCE is narrow and DiCE wins the matched pairwise actionability comparison.
- Contestability-support rankings are close enough that CSS should be discussed as a tradeoff, not only as a single leaderboard.
- Primary-versus-audit track effects are visible in the current full-source run, especially for faithfulness and stability, so cross-domain track effects should be framed descriptively rather than as a settled policy result.
- Pairwise win tables are likely to be more persuasive than raw averages when arguing robustness across datasets and model classes.

## Uncertainty Note

- `95%` hierarchical-bootstrap confidence intervals with `1000` resamples over `dataset_id -> track -> model_id` benchmark clusters are included in the analysis bundle.
- These intervals are computed by resampling the benchmark structure rather than treating all scored groups as IID, so they are useful for robustness checks but are still not full applicant-level or fully model-based uncertainty estimates.

## Caveat

- These outputs come from the manuscript-facing full-source cross-domain benchmark run. Model backends in use: `ebm_classifier` x 30, `sklearn_logistic_regression` x 30, `xgboost_classifier` x 30. Explainer backends in use: `lime_library` x 18, `search_anchor_rule` x 18, `search_counterfactual_dice_style` x 18, `search_counterfactual_wachter_style` x 18, `shap_library` x 18. Claims should still be phrased as cross-domain benchmark evidence rather than universal legal coverage or deployment certification.
