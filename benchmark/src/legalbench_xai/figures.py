import json
from pathlib import Path
from typing import Any

from .paths import repo_relative_path


def write_figure_pack(
    target_dir: Path,
    analysis: dict[str, Any],
) -> dict[str, Any]:
    figures_dir = target_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "leaderboard_tradeoff": "leaderboard_tradeoff.vl.json",
        "dataset_track_heatmap_css": "dataset_track_heatmap_css.vl.json",
        "dataset_track_heatmap_stability": "dataset_track_heatmap_stability.vl.json",
        "dataset_track_heatmap_comprehensibility": "dataset_track_heatmap_comprehensibility.vl.json",
        "dataset_track_heatmap_actionability": "dataset_track_heatmap_actionability.vl.json",
        "model_metric_bars": "model_metric_bars.vl.json",
        "pairwise_faithfulness": "pairwise_faithfulness.vl.json",
        "pairwise_stability": "pairwise_stability.vl.json",
        "pairwise_comprehensibility": "pairwise_comprehensibility.vl.json",
        "pairwise_css": "pairwise_css.vl.json",
        "pairwise_actionability": "pairwise_actionability.vl.json",
        "pairwise_alf": "pairwise_alf.vl.json",
        "captions": "captions.md",
        "manifest": "manifest.json",
    }

    specs = {
        files["leaderboard_tradeoff"]: _leaderboard_tradeoff_spec(),
        files["dataset_track_heatmap_css"]: _dataset_track_heatmap_spec(
            metric_field="mean_css",
            metric_title="Mean CSS",
        ),
        files["dataset_track_heatmap_stability"]: _dataset_track_heatmap_spec(
            metric_field="mean_stability",
            metric_title="Mean Stability",
        ),
        files["dataset_track_heatmap_comprehensibility"]: _dataset_track_heatmap_spec(
            metric_field="mean_comprehensibility",
            metric_title="Mean Comprehensibility",
        ),
        files["dataset_track_heatmap_actionability"]: _dataset_track_heatmap_spec(
            metric_field="mean_actionability",
            metric_title="Mean Actionability",
        ),
        files["model_metric_bars"]: _model_metric_bars_spec(),
        files["pairwise_faithfulness"]: _pairwise_heatmap_spec(metric="faithfulness"),
        files["pairwise_stability"]: _pairwise_heatmap_spec(metric="stability"),
        files["pairwise_comprehensibility"]: _pairwise_heatmap_spec(metric="comprehensibility"),
        files["pairwise_css"]: _pairwise_heatmap_spec(metric="css"),
        files["pairwise_actionability"]: _pairwise_heatmap_spec(metric="actionability"),
        files["pairwise_alf"]: _pairwise_heatmap_spec(metric="alf"),
    }

    for filename, spec in specs.items():
        path = figures_dir / filename
        path.write_text(json.dumps(spec, indent=2), encoding="utf-8")

    captions_path = figures_dir / files["captions"]
    captions_path.write_text(_captions_markdown(), encoding="utf-8")

    manifest = {
        "figures": [
            {
                "figure_id": "fig_overall_tradeoff",
                "title": "Overall Explainer Tradeoffs",
                "spec_file": repo_relative_path(figures_dir / files["leaderboard_tradeoff"]),
                "source_data": "by_explainer.csv",
                "caption_anchor": "fig_overall_tradeoff",
            },
            {
                "figure_id": "fig_dataset_track_css",
                "title": "Dataset and Track CSS Heatmap",
                "spec_file": repo_relative_path(figures_dir / files["dataset_track_heatmap_css"]),
                "source_data": "by_dataset_track_explainer.csv",
                "caption_anchor": "fig_dataset_track_css",
            },
            {
                "figure_id": "fig_dataset_track_stability",
                "title": "Dataset and Track Stability Heatmap",
                "spec_file": repo_relative_path(
                    figures_dir / files["dataset_track_heatmap_stability"]
                ),
                "source_data": "by_dataset_track_explainer.csv",
                "caption_anchor": "fig_dataset_track_stability",
            },
            {
                "figure_id": "fig_dataset_track_comprehensibility",
                "title": "Dataset and Track Comprehensibility Heatmap",
                "spec_file": repo_relative_path(
                    figures_dir / files["dataset_track_heatmap_comprehensibility"]
                ),
                "source_data": "by_dataset_track_explainer.csv",
                "caption_anchor": "fig_dataset_track_comprehensibility",
            },
            {
                "figure_id": "fig_dataset_track_actionability",
                "title": "Dataset and Track Actionability Heatmap",
                "spec_file": repo_relative_path(
                    figures_dir / files["dataset_track_heatmap_actionability"]
                ),
                "source_data": "by_dataset_track_explainer.csv",
                "caption_anchor": "fig_dataset_track_actionability",
            },
            {
                "figure_id": "fig_model_family_metrics",
                "title": "Model Family Metric Comparison",
                "spec_file": repo_relative_path(figures_dir / files["model_metric_bars"]),
                "source_data": "by_model_explainer.csv",
                "caption_anchor": "fig_model_family_metrics",
            },
            {
                "figure_id": "fig_pairwise_faithfulness",
                "title": "Pairwise Faithfulness Win Rates",
                "spec_file": repo_relative_path(figures_dir / files["pairwise_faithfulness"]),
                "source_data": "pairwise_wins.csv",
                "caption_anchor": "fig_pairwise_faithfulness",
            },
            {
                "figure_id": "fig_pairwise_stability",
                "title": "Pairwise Stability Win Rates",
                "spec_file": repo_relative_path(figures_dir / files["pairwise_stability"]),
                "source_data": "pairwise_wins.csv",
                "caption_anchor": "fig_pairwise_stability",
            },
            {
                "figure_id": "fig_pairwise_comprehensibility",
                "title": "Pairwise Comprehensibility Win Rates",
                "spec_file": repo_relative_path(
                    figures_dir / files["pairwise_comprehensibility"]
                ),
                "source_data": "pairwise_wins.csv",
                "caption_anchor": "fig_pairwise_comprehensibility",
            },
            {
                "figure_id": "fig_pairwise_css",
                "title": "Pairwise CSS Win Rates",
                "spec_file": repo_relative_path(figures_dir / files["pairwise_css"]),
                "source_data": "pairwise_wins.csv",
                "caption_anchor": "fig_pairwise_css",
            },
            {
                "figure_id": "fig_pairwise_actionability",
                "title": "Pairwise Actionability Win Rates",
                "spec_file": repo_relative_path(figures_dir / files["pairwise_actionability"]),
                "source_data": "pairwise_wins.csv",
                "caption_anchor": "fig_pairwise_actionability",
            },
            {
                "figure_id": "fig_pairwise_alf",
                "title": "Pairwise ALF Win Rates",
                "spec_file": repo_relative_path(figures_dir / files["pairwise_alf"]),
                "source_data": "pairwise_wins.csv",
                "caption_anchor": "fig_pairwise_alf",
            },
        ]
    }
    manifest_path = figures_dir / files["manifest"]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "directory": repo_relative_path(figures_dir),
        "files": {
            "leaderboard_tradeoff_spec": repo_relative_path(
                figures_dir / files["leaderboard_tradeoff"]
            ),
            "dataset_track_css_spec": repo_relative_path(
                figures_dir / files["dataset_track_heatmap_css"]
            ),
            "dataset_track_stability_spec": repo_relative_path(
                figures_dir / files["dataset_track_heatmap_stability"]
            ),
            "dataset_track_comprehensibility_spec": repo_relative_path(
                figures_dir / files["dataset_track_heatmap_comprehensibility"]
            ),
            "dataset_track_actionability_spec": repo_relative_path(
                figures_dir / files["dataset_track_heatmap_actionability"]
            ),
            "model_metric_bars_spec": repo_relative_path(figures_dir / files["model_metric_bars"]),
            "pairwise_faithfulness_spec": repo_relative_path(
                figures_dir / files["pairwise_faithfulness"]
            ),
            "pairwise_stability_spec": repo_relative_path(
                figures_dir / files["pairwise_stability"]
            ),
            "pairwise_comprehensibility_spec": repo_relative_path(
                figures_dir / files["pairwise_comprehensibility"]
            ),
            "pairwise_css_spec": repo_relative_path(figures_dir / files["pairwise_css"]),
            "pairwise_actionability_spec": repo_relative_path(
                figures_dir / files["pairwise_actionability"]
            ),
            "pairwise_alf_spec": repo_relative_path(figures_dir / files["pairwise_alf"]),
            "captions_markdown": repo_relative_path(captions_path),
            "manifest_json": repo_relative_path(manifest_path),
        },
    }


def _leaderboard_tradeoff_spec() -> dict[str, Any]:
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "description": "Overall explainer tradeoffs between contestability and actionability.",
        "data": {"url": "../by_explainer.csv"},
        "mark": {"type": "point", "filled": True, "size": 180},
        "encoding": {
            "x": {
                "field": "mean_css",
                "type": "quantitative",
                "title": "Mean Contestability Support Score",
            },
            "y": {
                "field": "mean_actionability",
                "type": "quantitative",
                "title": "Mean Unified Actionability",
            },
            "color": {
                "field": "explainer_id",
                "type": "nominal",
                "title": "Explainer",
            },
            "size": {
                "field": "total_instances",
                "type": "quantitative",
                "title": "Evaluated Instances",
            },
            "tooltip": [
                {"field": "explainer_id", "type": "nominal", "title": "Explainer"},
                {"field": "group_count", "type": "quantitative", "title": "Groups"},
                {"field": "total_instances", "type": "quantitative", "title": "Instances"},
                {
                    "field": "mean_faithfulness",
                    "type": "quantitative",
                    "title": "Mean Faithfulness",
                },
                {
                    "field": "mean_stability",
                    "type": "quantitative",
                    "title": "Mean Stability",
                },
                {
                    "field": "mean_comprehensibility",
                    "type": "quantitative",
                    "title": "Mean Comprehensibility",
                },
                {"field": "mean_css", "type": "quantitative", "title": "Mean CSS"},
                {
                    "field": "mean_actionability",
                    "type": "quantitative",
                    "title": "Mean Actionability",
                },
                {"field": "mean_alf", "type": "quantitative", "title": "Mean ALF"},
            ],
        },
    }


def _dataset_track_heatmap_spec(metric_field: str, metric_title: str) -> dict[str, Any]:
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "description": f"{metric_title} by dataset, track, and explainer.",
        "data": {"url": "../by_dataset_track_explainer.csv"},
        "transform": [
            {
                "calculate": "datum.dataset_id + ' / ' + datum.track",
                "as": "dataset_track",
            }
        ],
        "mark": "rect",
        "encoding": {
            "x": {
                "field": "explainer_id",
                "type": "nominal",
                "title": "Explainer",
            },
            "y": {
                "field": "dataset_track",
                "type": "nominal",
                "title": "Dataset / Track",
                "sort": "ascending",
            },
            "color": {
                "field": metric_field,
                "type": "quantitative",
                "title": metric_title,
            },
            "tooltip": [
                {"field": "dataset_id", "type": "nominal", "title": "Dataset"},
                {"field": "track", "type": "nominal", "title": "Track"},
                {"field": "explainer_id", "type": "nominal", "title": "Explainer"},
                {"field": metric_field, "type": "quantitative", "title": metric_title},
                {"field": "total_instances", "type": "quantitative", "title": "Instances"},
            ],
        },
    }


def _model_metric_bars_spec() -> dict[str, Any]:
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "description": "Explainer performance by model family across benchmark metrics.",
        "data": {"url": "../by_model_explainer.csv"},
        "transform": [
            {
                "fold": [
                    "mean_faithfulness",
                    "mean_stability",
                    "mean_comprehensibility",
                    "mean_css",
                    "mean_actionability",
                    "mean_alf"
                ],
                "as": ["metric", "value"],
            }
        ],
        "mark": "bar",
        "encoding": {
            "x": {"field": "explainer_id", "type": "nominal", "title": "Explainer"},
            "y": {"field": "value", "type": "quantitative", "title": "Metric Value"},
            "color": {"field": "explainer_id", "type": "nominal", "legend": None},
            "column": {"field": "model_id", "type": "nominal", "title": "Model Family"},
            "row": {"field": "metric", "type": "nominal", "title": "Metric"},
            "tooltip": [
                {"field": "model_id", "type": "nominal", "title": "Model"},
                {"field": "explainer_id", "type": "nominal", "title": "Explainer"},
                {"field": "metric", "type": "nominal", "title": "Metric"},
                {"field": "value", "type": "quantitative", "title": "Value"},
                {"field": "total_instances", "type": "quantitative", "title": "Instances"},
            ],
        },
    }


def _pairwise_heatmap_spec(metric: str) -> dict[str, Any]:
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "description": f"Pairwise explainer win rates for {metric}.",
        "data": {"url": "../pairwise_wins.csv"},
        "transform": [{"filter": f"datum.metric === '{metric}'"}],
        "mark": "rect",
        "encoding": {
            "x": {
                "field": "explainer_b",
                "type": "nominal",
                "title": "Explainer B",
            },
            "y": {
                "field": "explainer_a",
                "type": "nominal",
                "title": "Explainer A",
            },
            "color": {
                "field": "win_rate_a",
                "type": "quantitative",
                "title": "Win Rate for A",
            },
            "tooltip": [
                {"field": "metric", "type": "nominal", "title": "Metric"},
                {"field": "explainer_a", "type": "nominal", "title": "Explainer A"},
                {"field": "explainer_b", "type": "nominal", "title": "Explainer B"},
                {"field": "wins_a", "type": "quantitative", "title": "Wins A"},
                {"field": "wins_b", "type": "quantitative", "title": "Wins B"},
                {"field": "ties", "type": "quantitative", "title": "Ties"},
                {
                    "field": "comparable_groups",
                    "type": "quantitative",
                    "title": "Comparable Groups",
                },
                {"field": "win_rate_a", "type": "quantitative", "title": "Win Rate A"},
                {
                    "field": "win_rate_a_ci_low",
                    "type": "quantitative",
                    "title": "Win Rate A CI Low",
                },
                {
                    "field": "win_rate_a_ci_high",
                    "type": "quantitative",
                    "title": "Win Rate A CI High",
                },
            ],
        },
    }


def _captions_markdown() -> str:
    return "\n".join(
        [
            "# Figure Captions",
            "",
            "## fig_overall_tradeoff",
            "Overall explainer tradeoffs between contestability support and unified actionability. Point size encodes the number of evaluated adverse instances. This figure is the best main-paper visual for showing that no single method dominates every legally meaningful dimension.",
            "",
            "## fig_dataset_track_css",
            "Heatmap of mean Contestability Support Score by dataset, track, and explainer. This figure is useful for showing whether contestability performance is stable across domains and whether the primary versus audit split materially changes the rankings.",
            "",
            "## fig_dataset_track_stability",
            "Heatmap of mean explanation stability by dataset, track, and explainer. This figure is useful for showing whether some explanation families remain consistent across local neighborhood perturbations while others fluctuate more across otherwise similar denied applicants.",
            "",
            "## fig_dataset_track_comprehensibility",
            "Heatmap of mean comprehensibility by dataset, track, and explainer. This figure is useful for showing whether applicant-facing simplicity is robust across domains or whether some methods become hard to read once real credit factors accumulate.",
            "",
            "## fig_dataset_track_actionability",
            "Heatmap of mean unified actionability by dataset, track, and explainer. This is the clearest appendix companion to the CSS heatmap and helps surface whether recourse-oriented explainers outperform attribution methods consistently across settings.",
            "",
            "## fig_model_family_metrics",
            "Model-family comparison across faithfulness, stability, comprehensibility, CSS, actionability, and ALF. This figure is designed to answer whether observed explainer rankings are robust across linear, additive, and boosted model families rather than being artifacts of a single predictive backend.",
            "",
            "## fig_pairwise_faithfulness",
            "Pairwise win-rate heatmap for faithfulness. This figure is useful when arguing that explanation methods that look intuitive do not necessarily align equally well with the underlying model behavior across matched dataset-track-model groups.",
            "",
            "## fig_pairwise_stability",
            "Pairwise win-rate heatmap for stability. This figure is useful when arguing that some explanation families vary more across similar denied applicants or across random seeds, even when they perform well on other benchmark dimensions.",
            "",
            "## fig_pairwise_comprehensibility",
            "Pairwise win-rate heatmap for comprehensibility. This figure is useful when arguing that some explanation families are consistently easier to parse even when they are not the strongest on faithfulness or actionability.",
            "",
            "## fig_pairwise_css",
            "Pairwise win-rate heatmap for CSS. Each cell reports the share of matched dataset-track-model groups where Explainer A outperforms Explainer B on contestability support.",
            "",
            "## fig_pairwise_actionability",
            "Pairwise win-rate heatmap for unified actionability. This figure makes it easy to state whether recourse-oriented methods consistently dominate attribution-style methods when explanations are evaluated for applicant-facing actionability.",
            "",
            "## fig_pairwise_alf",
            "Pairwise win-rate heatmap for the Actionability Legality Filter. Because ALF is only defined for recourse-style methods in the current scaffold, this figure should usually appear in the appendix or methods section unless it becomes a central contribution.",
            "",
        ]
    ) + "\n"
