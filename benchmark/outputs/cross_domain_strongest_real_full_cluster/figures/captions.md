# Figure Captions

## fig_overall_tradeoff
Overall explainer tradeoffs between contestability support and unified actionability. Point size encodes the number of evaluated adverse instances. This figure is the best main-paper visual for showing that no single method dominates every legally meaningful dimension.

## fig_dataset_track_css
Heatmap of mean Contestability Support Score by dataset, track, and explainer. This figure is useful for showing whether contestability performance is stable across domains and whether the primary versus audit split materially changes the rankings.

## fig_dataset_track_stability
Heatmap of mean explanation stability by dataset, track, and explainer. This figure is useful for showing whether some explanation families remain consistent across local neighborhood perturbations while others fluctuate more across otherwise similar denied applicants.

## fig_dataset_track_comprehensibility
Heatmap of mean comprehensibility by dataset, track, and explainer. This figure is useful for showing whether applicant-facing simplicity is robust across domains or whether some methods become hard to read once real credit factors accumulate.

## fig_dataset_track_actionability
Heatmap of mean unified actionability by dataset, track, and explainer. This is the clearest appendix companion to the CSS heatmap and helps surface whether recourse-oriented explainers outperform attribution methods consistently across settings.

## fig_model_family_metrics
Model-family comparison across faithfulness, stability, comprehensibility, CSS, actionability, and ALF. This figure is designed to answer whether observed explainer rankings are robust across linear, additive, and boosted model families rather than being artifacts of a single predictive backend.

## fig_pairwise_faithfulness
Pairwise win-rate heatmap for faithfulness. This figure is useful when arguing that explanation methods that look intuitive do not necessarily align equally well with the underlying model behavior across matched dataset-track-model groups.

## fig_pairwise_stability
Pairwise win-rate heatmap for stability. This figure is useful when arguing that some explanation families vary more across similar denied applicants or across random seeds, even when they perform well on other benchmark dimensions.

## fig_pairwise_comprehensibility
Pairwise win-rate heatmap for comprehensibility. This figure is useful when arguing that some explanation families are consistently easier to parse even when they are not the strongest on faithfulness or actionability.

## fig_pairwise_css
Pairwise win-rate heatmap for CSS. Each cell reports the share of matched dataset-track-model groups where Explainer A outperforms Explainer B on contestability support.

## fig_pairwise_actionability
Pairwise win-rate heatmap for unified actionability. This figure makes it easy to state whether recourse-oriented methods consistently dominate attribution-style methods when explanations are evaluated for applicant-facing actionability.

## fig_pairwise_alf
Pairwise win-rate heatmap for the Actionability Legality Filter. Because ALF is only defined for recourse-style methods in the current scaffold, this figure should usually appear in the appendix or methods section unless it becomes a central contribution.

