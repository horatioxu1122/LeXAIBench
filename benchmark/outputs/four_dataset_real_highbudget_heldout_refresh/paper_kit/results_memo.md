# LegalBench-XAI Results Memo

- Experiment: `legalbench_xai_four_dataset_real_highbudget`
- Generated at (UTC): `2026-03-15T22:29:00.007268+00:00`

## Headline Findings

- Overall faithfulness leader: `shap` with mean faithfulness `0.908` and `[0.866, 0.955]` bootstrap CI.
- Overall stability leader: `lime` with mean stability `0.917` and `[0.882, 0.956]` bootstrap CI.
- Overall comprehensibility leader: `anchors` with mean comprehensibility `0.944` and `[0.918, 0.974]` bootstrap CI.
- Overall CSS leader: `dice` with mean CSS `0.896` and `[0.836, 0.959]` bootstrap CI.
- Overall actionability leader: `wachter` with mean actionability `0.674` and `[0.627, 0.725]` bootstrap CI.
- Overall ALF leader: `wachter` with mean ALF `0.482` and `[0.445, 0.512]` bootstrap CI.
- Pairwise faithfulness leader: `shap` with `80` matched-configuration wins.
- Pairwise stability leader: `lime` with `85` matched-configuration wins.
- Pairwise comprehensibility leader: `anchors` with `96` matched-configuration wins.
- Pairwise CSS leader: `dice` with `78` matched-configuration wins.
- Pairwise actionability leader: `wachter` with `86` matched-configuration wins.
- Pairwise ALF leader: `wachter` with `14` matched-configuration wins.

## Draft Results Language

Across the current benchmark run, faithfulness, stability, comprehensibility, and subject-facing usefulness do not collapse into a single ranking. `shap` leads on mean faithfulness with `0.908`, `lime` leads on mean stability with `0.917`, `anchors` leads on mean comprehensibility with `0.944`, while recourse-oriented explainers outperform attribution-style methods on subject-facing metrics. `dice` leads on overall contestability support with mean CSS `0.896`, while `wachter` leads on overall actionability with `0.674` and also achieves the strongest legality-aware recourse performance with mean ALF `0.482`. Primary-vs-audit score shifts remain small on average, with the largest mean absolute shift on `stability` at `0.047`. Primary and audit track leader identities are stable for `3` of `6` metrics, with instability concentrated in `stability`, `actionability`, `alf`. The least stable metric is `actionability` at same-leader rate `0.667`.

## Manuscript-Ready Interpretive Claims

- Faithfulness should be discussed separately from contestability, because a method can align well with model behavior and still be weak as subject-facing guidance.
- Stability should be discussed separately from faithfulness, because a method can match model behavior on one instance while still being brittle across similar applicants or random seeds.
- Comprehensibility should be discussed separately from both faithfulness and actionability, because the easiest explanation to read may not be the most faithful or the most useful for recourse.
- Recourse-oriented methods dominate attribution methods on subject-facing actionability in this credit-core bundle.
- Contestability-support rankings are close enough that CSS should be discussed as a tradeoff, not only as a single leaderboard.
- Primary-versus-audit track effects are limited in the current benchmark run, with the largest shifts concentrated in stability and recourse-related metrics.
- Pairwise win tables are likely to be more persuasive than raw averages when arguing robustness across datasets and model classes.

## Uncertainty Note

- `95%` hierarchical-bootstrap confidence intervals with `1000` resamples over `dataset_id -> track -> model_id` benchmark clusters are included in the analysis bundle.
- These intervals are computed by resampling the benchmark structure rather than treating all scored groups as IID, so they are useful for robustness checks but are still not full applicant-level or fully model-based uncertainty estimates.

## Caveat

- These outputs come from the manuscript-facing fully real credit-core benchmark run. Model backends in use: `ebm_classifier` x 40, `sklearn_logistic_regression` x 40, `xgboost_classifier` x 40. Explainer backends in use: `lime_library` x 24, `search_anchor_rule` x 24, `search_counterfactual_dice_style` x 24, `search_counterfactual_wachter_style` x 24, `shap_library` x 24. Claims should still be phrased as benchmark evidence rather than legal sufficiency or deployment certification.
