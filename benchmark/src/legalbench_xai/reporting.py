import csv
import json
import os
import random
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from time import perf_counter
from typing import Any

from .config import ExperimentConfig
from .figures import write_figure_pack
from .paperkit import write_paper_kit
from .paths import repo_relative_path, resolve_repo_path
from .progress import emit_progress


AGGREGATE_METRICS = [
    "mean_faithfulness",
    "mean_stability",
    "mean_comprehensibility",
    "mean_css",
    "mean_actionability",
    "mean_alf",
    "mean_invalid_recourse_rate",
]

DEFAULT_BOOTSTRAP_HIERARCHY = ["dataset_id", "track", "model_id"]
METRIC_ANALYSIS_FIELDS = [
    ("mean_faithfulness", "faithfulness"),
    ("mean_stability", "stability"),
    ("mean_comprehensibility", "comprehensibility"),
    ("mean_css", "css"),
    ("mean_actionability", "actionability"),
    ("mean_alf", "alf"),
]

UNIVERSAL_COMPOSITE_METRICS = [
    "mean_faithfulness",
    "mean_stability",
    "mean_comprehensibility",
    "mean_css",
    "mean_actionability",
]

UNIVERSAL_WEIGHT_PROFILES = {
    "balanced": {
        "mean_faithfulness": 0.20,
        "mean_stability": 0.20,
        "mean_comprehensibility": 0.20,
        "mean_css": 0.20,
        "mean_actionability": 0.20,
    },
    "reliability_focused": {
        "mean_faithfulness": 0.45,
        "mean_stability": 0.35,
        "mean_comprehensibility": 0.10,
        "mean_css": 0.05,
        "mean_actionability": 0.05,
    },
    "applicant_facing": {
        "mean_faithfulness": 0.05,
        "mean_stability": 0.05,
        "mean_comprehensibility": 0.25,
        "mean_css": 0.35,
        "mean_actionability": 0.30,
    },
    "contestability_focused": {
        "mean_faithfulness": 0.05,
        "mean_stability": 0.05,
        "mean_comprehensibility": 0.15,
        "mean_css": 0.50,
        "mean_actionability": 0.25,
    },
}

RECOURSE_WEIGHT_PROFILES = {
    "recourse_balanced": {
        "mean_actionability": 0.60,
        "mean_alf": 0.40,
    },
    "legality_constrained": {
        "mean_actionability": 0.30,
        "mean_alf": 0.70,
    },
}

CSS_COMPONENT_FIELDS = [
    ("mean_css_verifiability", "verifiability"),
    ("mean_css_correctability", "correctability"),
    (
        "mean_css_decision_sensitivity_visibility",
        "decision_sensitivity_visibility",
    ),
    ("mean_css_reason_specificity", "reason_specificity"),
]

ALF_COMPONENT_FIELDS = [
    ("mean_alf_feasibility", "feasibility"),
    ("mean_alf_legality", "legality"),
    ("mean_alf_minimality", "minimality"),
    ("mean_alf_hard_failure_rate", "hard_failure_rate"),
    ("mean_invalid_recourse_rate", "invalid_recourse_rate"),
]


def write_benchmark_artifacts(
    config: ExperimentConfig,
    payload: dict[str, Any],
    output_dir: str | None = None,
) -> dict[str, Any]:
    overall_start = perf_counter()
    target_dir = _resolve_output_dir(config=config, output_dir=output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    emit_progress("artifacts_start", output_dir=str(target_dir))
    analysis_start = perf_counter()
    analysis = build_analysis_views(
        payload["summary"],
        random_seed=config.random_seed,
        bootstrap_config=config.reporting_bootstrap,
    )
    emit_progress(
        "artifacts_analysis_done",
        output_dir=str(target_dir),
        elapsed_seconds=round(perf_counter() - analysis_start, 3),
    )

    json_path = target_dir / "results.json"
    analysis_path = target_dir / "analysis.json"
    csv_path = target_dir / "summary.csv"
    by_explainer_path = target_dir / "by_explainer.csv"
    by_dataset_track_explainer_path = target_dir / "by_dataset_track_explainer.csv"
    by_model_explainer_path = target_dir / "by_model_explainer.csv"
    model_performance_path = target_dir / "model_performance.csv"
    pairwise_wins_path = target_dir / "pairwise_wins.csv"
    metric_rank_correlation_path = target_dir / "metric_rank_correlation.csv"
    group_metric_correlation_path = target_dir / "group_metric_correlation.csv"
    metric_weight_profiles_path = target_dir / "metric_weight_profiles.csv"
    metric_weight_random_summary_path = target_dir / "metric_weight_random_summary.csv"
    metric_ablation_profiles_path = target_dir / "metric_ablation_profiles.csv"
    css_definition_profiles_path = target_dir / "css_definition_profiles.csv"
    alf_definition_profiles_path = target_dir / "alf_definition_profiles.csv"
    evidence_source_summary_path = target_dir / "evidence_source_summary.csv"
    proxy_sensitivity_profiles_path = target_dir / "proxy_sensitivity_profiles.csv"
    direct_evidence_by_explainer_path = target_dir / "direct_evidence_by_explainer.csv"
    direct_evidence_profiles_path = target_dir / "direct_evidence_profiles.csv"
    direct_evidence_pairwise_wins_path = target_dir / "direct_evidence_pairwise_wins.csv"
    subgroup_summary_path = target_dir / "subgroup_summary.csv"
    subgroup_spread_summary_path = target_dir / "subgroup_spread_summary.csv"
    track_policy_shift_path = target_dir / "track_policy_shift.csv"
    track_policy_by_explainer_path = target_dir / "track_policy_by_explainer.csv"
    track_policy_metric_summary_path = target_dir / "track_policy_metric_summary.csv"
    track_policy_leader_stability_path = target_dir / "track_policy_leader_stability.csv"
    track_policy_leader_summary_path = target_dir / "track_policy_leader_summary.csv"
    leave_one_dataset_out_path = target_dir / "leave_one_dataset_out.csv"
    leave_one_dataset_out_leaders_path = target_dir / "leave_one_dataset_out_leaders.csv"
    report_path = target_dir / "report.md"
    latex_path = target_dir / "leaderboard.tex"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    with analysis_path.open("w", encoding="utf-8") as handle:
        json.dump(analysis, handle, indent=2)

    csv_start = perf_counter()
    _write_summary_csv(csv_path, payload["summary"])
    _write_rows_csv(by_explainer_path, analysis["by_explainer"])
    _write_rows_csv(by_dataset_track_explainer_path, analysis["by_dataset_track_explainer"])
    _write_rows_csv(by_model_explainer_path, analysis["by_model_explainer"])
    _write_rows_csv(model_performance_path, analysis["model_performance"])
    _write_rows_csv(pairwise_wins_path, analysis["pairwise_wins"])
    _write_rows_csv(metric_rank_correlation_path, analysis["metric_rank_correlation"])
    _write_rows_csv(group_metric_correlation_path, analysis["group_metric_correlation"])
    _write_rows_csv(metric_weight_profiles_path, analysis["metric_weight_profiles"])
    _write_rows_csv(
        metric_weight_random_summary_path,
        analysis["metric_weight_random_summary"],
    )
    _write_rows_csv(
        metric_ablation_profiles_path,
        analysis["metric_ablation_profiles"],
    )
    _write_rows_csv(
        css_definition_profiles_path,
        analysis["css_definition_profiles"],
    )
    _write_rows_csv(
        alf_definition_profiles_path,
        analysis["alf_definition_profiles"],
    )
    _write_rows_csv(
        evidence_source_summary_path,
        analysis["evidence_source_summary"],
    )
    _write_rows_csv(
        proxy_sensitivity_profiles_path,
        analysis["proxy_sensitivity_profiles"],
    )
    _write_rows_csv(
        direct_evidence_by_explainer_path,
        analysis["direct_evidence_by_explainer"],
    )
    _write_rows_csv(
        direct_evidence_profiles_path,
        analysis["direct_evidence_profiles"],
    )
    _write_rows_csv(
        direct_evidence_pairwise_wins_path,
        analysis["direct_evidence_pairwise_wins"],
    )
    _write_rows_csv(
        subgroup_summary_path,
        analysis["subgroup_summary"],
    )
    _write_rows_csv(
        subgroup_spread_summary_path,
        analysis["subgroup_spread_summary"],
    )
    _write_rows_csv(track_policy_shift_path, analysis["track_policy_shift"])
    _write_rows_csv(
        track_policy_by_explainer_path,
        analysis["track_policy_by_explainer"],
    )
    _write_rows_csv(
        track_policy_metric_summary_path,
        analysis["track_policy_metric_summary"],
    )
    _write_rows_csv(
        track_policy_leader_stability_path,
        analysis["track_policy_leader_stability"],
    )
    _write_rows_csv(
        track_policy_leader_summary_path,
        analysis["track_policy_leader_summary"],
    )
    _write_rows_csv(leave_one_dataset_out_path, analysis["leave_one_dataset_out"])
    _write_rows_csv(
        leave_one_dataset_out_leaders_path,
        analysis["leave_one_dataset_out_leaders"],
    )
    emit_progress(
        "artifacts_csvs_done",
        output_dir=str(target_dir),
        elapsed_seconds=round(perf_counter() - csv_start, 3),
    )
    render_start = perf_counter()
    report_path.write_text(_build_markdown_report(payload, analysis), encoding="utf-8")
    latex_path.write_text(_build_latex_leaderboard(analysis), encoding="utf-8")
    figure_pack = write_figure_pack(target_dir=target_dir, analysis=analysis)
    paper_kit = write_paper_kit(target_dir=target_dir, payload=payload, analysis=analysis)
    emit_progress(
        "artifacts_render_done",
        output_dir=str(target_dir),
        elapsed_seconds=round(perf_counter() - render_start, 3),
    )

    result = {
        "output_dir": repo_relative_path(target_dir),
        "files": {
            "results_json": repo_relative_path(json_path),
            "analysis_json": repo_relative_path(analysis_path),
            "summary_csv": repo_relative_path(csv_path),
            "by_explainer_csv": repo_relative_path(by_explainer_path),
            "by_dataset_track_explainer_csv": repo_relative_path(
                by_dataset_track_explainer_path
            ),
            "by_model_explainer_csv": repo_relative_path(by_model_explainer_path),
            "model_performance_csv": repo_relative_path(model_performance_path),
            "pairwise_wins_csv": repo_relative_path(pairwise_wins_path),
            "metric_rank_correlation_csv": repo_relative_path(metric_rank_correlation_path),
            "group_metric_correlation_csv": repo_relative_path(group_metric_correlation_path),
            "metric_weight_profiles_csv": repo_relative_path(metric_weight_profiles_path),
            "metric_weight_random_summary_csv": repo_relative_path(
                metric_weight_random_summary_path
            ),
            "metric_ablation_profiles_csv": repo_relative_path(
                metric_ablation_profiles_path
            ),
            "css_definition_profiles_csv": repo_relative_path(
                css_definition_profiles_path
            ),
            "alf_definition_profiles_csv": repo_relative_path(
                alf_definition_profiles_path
            ),
            "evidence_source_summary_csv": repo_relative_path(
                evidence_source_summary_path
            ),
            "proxy_sensitivity_profiles_csv": repo_relative_path(
                proxy_sensitivity_profiles_path
            ),
            "direct_evidence_by_explainer_csv": repo_relative_path(
                direct_evidence_by_explainer_path
            ),
            "direct_evidence_profiles_csv": repo_relative_path(
                direct_evidence_profiles_path
            ),
            "direct_evidence_pairwise_wins_csv": repo_relative_path(
                direct_evidence_pairwise_wins_path
            ),
            "subgroup_summary_csv": repo_relative_path(subgroup_summary_path),
            "subgroup_spread_summary_csv": repo_relative_path(
                subgroup_spread_summary_path
            ),
            "track_policy_shift_csv": repo_relative_path(track_policy_shift_path),
            "track_policy_by_explainer_csv": repo_relative_path(
                track_policy_by_explainer_path
            ),
            "track_policy_metric_summary_csv": repo_relative_path(
                track_policy_metric_summary_path
            ),
            "track_policy_leader_stability_csv": repo_relative_path(
                track_policy_leader_stability_path
            ),
            "track_policy_leader_summary_csv": repo_relative_path(
                track_policy_leader_summary_path
            ),
            "leave_one_dataset_out_csv": repo_relative_path(leave_one_dataset_out_path),
            "leave_one_dataset_out_leaders_csv": repo_relative_path(
                leave_one_dataset_out_leaders_path
            ),
            "report_markdown": repo_relative_path(report_path),
            "leaderboard_tex": repo_relative_path(latex_path),
        },
        "figure_pack": figure_pack,
        "paper_kit": paper_kit,
    }
    emit_progress(
        "artifacts_done",
        output_dir=str(target_dir),
        elapsed_seconds=round(perf_counter() - overall_start, 3),
    )
    return result


def _resolve_output_dir(config: ExperimentConfig, output_dir: str | None) -> Path:
    if output_dir:
        return resolve_repo_path(output_dir)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return config.output_root / f"{config.experiment_name}_{timestamp}"


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "dataset_id",
        "track",
        "model_id",
        "explainer_id",
        "data_source_kind",
        "data_source_path",
        "model_backend",
        "model_implementation",
        "explainer_backend",
        "explainer_implementation",
        "total_rows",
        "train_rows",
        "test_rows",
        "evaluation_split_strategy",
        "heldout_evaluation",
        "threshold_policy_mode",
        "model_decision_threshold",
        "target_predicted_adverse_rate",
        "achieved_predicted_adverse_rate",
        "model_accuracy",
        "model_roc_auc",
        "model_average_precision",
        "model_brier_score",
        "model_expected_calibration_error",
        "model_positive_rate",
        "model_predicted_positive_rate",
        "adverse_rows",
        "evaluated_rows",
        "adverse_rate",
        "mean_faithfulness",
        "mean_stability",
        "mean_comprehensibility",
        "mean_css",
        "mean_css_verifiability",
        "mean_css_correctability",
        "mean_css_decision_sensitivity_visibility",
        "mean_css_reason_specificity",
        "mean_actionability",
        "mean_alf",
        "mean_alf_feasibility",
        "mean_alf_legality",
        "mean_alf_minimality",
        "mean_alf_hard_failure_rate",
        "mean_invalid_recourse_rate",
        "faithfulness_evidence_sources",
        "stability_evidence_sources",
        "sampling_strategy",
        "sampled_strata",
        "instance_ids",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "dataset_id": row["dataset_id"],
                    "track": row["track"],
                    "model_id": row["model_id"],
                    "explainer_id": row["explainer_id"],
                    "data_source_kind": row["data_source"]["kind"],
                    "data_source_path": row["data_source"]["path"],
                    "model_backend": row["model_backend"]["backend"],
                    "model_implementation": row["model_backend"]["implementation"],
                    "explainer_backend": row["explainer_backend"]["backend"],
                    "explainer_implementation": row["explainer_backend"]["implementation"],
                    "total_rows": row["total_rows"],
                    "train_rows": row.get("train_rows"),
                    "test_rows": row.get("test_rows"),
                    "evaluation_split_strategy": row["evaluation_split"]["strategy"],
                    "heldout_evaluation": row["evaluation_split"].get("heldout_used"),
                    "threshold_policy_mode": row.get("threshold_policy", {}).get("mode"),
                    "model_decision_threshold": row.get("model_decision_threshold"),
                    "target_predicted_adverse_rate": row.get("threshold_policy", {}).get(
                        "target_predicted_adverse_rate"
                    ),
                    "achieved_predicted_adverse_rate": row.get("threshold_policy", {}).get(
                        "achieved_predicted_adverse_rate"
                    ),
                    "model_accuracy": row.get("model_accuracy"),
                    "model_roc_auc": row.get("model_roc_auc"),
                    "model_average_precision": row.get("model_average_precision"),
                    "model_brier_score": row.get("model_brier_score"),
                    "model_expected_calibration_error": row.get(
                        "model_expected_calibration_error"
                    ),
                    "model_positive_rate": row.get("model_positive_rate"),
                    "model_predicted_positive_rate": row.get(
                        "model_predicted_positive_rate"
                    ),
                    "adverse_rows": row["adverse_rows"],
                    "evaluated_rows": row["evaluated_rows"],
                    "adverse_rate": row["adverse_rate"],
                    "mean_faithfulness": row["mean_faithfulness"],
                    "mean_stability": row["mean_stability"],
                    "mean_comprehensibility": row["mean_comprehensibility"],
                    "mean_css": row["mean_css"],
                    "mean_css_verifiability": row.get("mean_css_verifiability"),
                    "mean_css_correctability": row.get("mean_css_correctability"),
                    "mean_css_decision_sensitivity_visibility": row.get(
                        "mean_css_decision_sensitivity_visibility"
                    ),
                    "mean_css_reason_specificity": row.get(
                        "mean_css_reason_specificity"
                    ),
                    "mean_actionability": row["mean_actionability"],
                    "mean_alf": row["mean_alf"],
                    "mean_alf_feasibility": row.get("mean_alf_feasibility"),
                    "mean_alf_legality": row.get("mean_alf_legality"),
                    "mean_alf_minimality": row.get("mean_alf_minimality"),
                    "mean_alf_hard_failure_rate": row.get(
                        "mean_alf_hard_failure_rate"
                    ),
                    "mean_invalid_recourse_rate": row["mean_invalid_recourse_rate"],
                    "faithfulness_evidence_sources": "|".join(
                        f"{name}:{count}"
                        for name, count in sorted(
                            row.get("faithfulness_evidence_sources", {}).items()
                        )
                    ),
                    "stability_evidence_sources": "|".join(
                        f"{name}:{count}"
                        for name, count in sorted(row["stability_evidence_sources"].items())
                    ),
                    "sampling_strategy": row["sampling"]["strategy"],
                    "sampled_strata": "|".join(row["sampling"]["strata_used"]),
                    "instance_ids": "|".join(row["instance_ids"]),
                }
            )


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _model_performance_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique_rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in summary_rows:
        key = (row["dataset_id"], row["track"], row["model_id"])
        if key in unique_rows:
            continue
        unique_rows[key] = {
            "dataset_id": row["dataset_id"],
            "track": row["track"],
            "model_id": row["model_id"],
            "data_source_kind": row["data_source"]["kind"],
            "data_source_path": row["data_source"]["path"],
            "model_backend": row["model_backend"]["backend"],
            "model_implementation": row["model_backend"]["implementation"],
            "train_rows": row.get("train_rows"),
            "test_rows": row.get("test_rows"),
            "split_strategy": row["evaluation_split"]["strategy"],
            "heldout_used": row["evaluation_split"].get("heldout_used", False),
            "threshold_policy_mode": row.get("threshold_policy", {}).get("mode"),
            "decision_threshold": row.get("model_decision_threshold"),
            "target_predicted_adverse_rate": row.get("threshold_policy", {}).get(
                "target_predicted_adverse_rate"
            ),
            "achieved_predicted_adverse_rate": row.get("threshold_policy", {}).get(
                "achieved_predicted_adverse_rate"
            ),
            "accuracy": row.get("model_accuracy"),
            "roc_auc": row.get("model_roc_auc"),
            "average_precision": row.get("model_average_precision"),
            "brier_score": row.get("model_brier_score"),
            "expected_calibration_error": row.get(
                "model_expected_calibration_error"
            ),
            "positive_rate": row.get("model_positive_rate"),
            "predicted_positive_rate": row.get("model_predicted_positive_rate"),
            "true_adverse_rate": row.get("model_positive_rate"),
            "predicted_adverse_rate": row.get("model_predicted_positive_rate"),
        }
    return [unique_rows[key] for key in sorted(unique_rows)]


def build_analysis_views(
    summary_rows: list[dict[str, Any]],
    random_seed: int = 0,
    bootstrap_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    bootstrap = _normalize_bootstrap_config(bootstrap_config)
    by_explainer = _aggregate_summary_rows(
        summary_rows,
        ["explainer_id"],
        random_seed=random_seed,
        bootstrap_config=bootstrap,
    )
    evidence_source_summary = _evidence_source_summary(summary_rows)
    direct_evidence_rows = _direct_faithfulness_rows(summary_rows)
    direct_evidence_by_explainer = _aggregate_summary_rows(
        direct_evidence_rows,
        ["explainer_id"],
        random_seed=random_seed,
        bootstrap_config=bootstrap,
    )
    metric_rank_correlation = _metric_rank_correlation(by_explainer)
    group_metric_correlation = _group_metric_correlation(summary_rows)
    metric_weight_profiles = _metric_weight_profiles(by_explainer)
    random_weight_draws = max(
        1,
        int(os.getenv("LEGALBENCH_METRIC_WEIGHT_RANDOM_DRAWS", "1000")),
    )
    metric_weight_random_summary = _metric_weight_random_summary(
        by_explainer,
        seed=random_seed + 104729,
        num_draws=random_weight_draws,
    )
    metric_ablation_profiles = _metric_ablation_profiles(by_explainer)
    css_definition_profiles = _css_definition_profiles(summary_rows)
    alf_definition_profiles = _alf_definition_profiles(summary_rows)
    track_policy_shift = _track_policy_shift(summary_rows)
    track_policy_by_explainer = _track_policy_by_explainer(track_policy_shift)
    track_policy_leader_stability = _track_policy_leader_stability(summary_rows)
    track_policy_leader_summary = _track_policy_leader_summary(
        track_policy_leader_stability
    )
    leave_one_dataset_out = _leave_one_dataset_out(summary_rows, random_seed, bootstrap)
    pairwise_wins = _pairwise_wins(
        summary_rows,
        random_seed=random_seed,
        bootstrap_config=bootstrap,
    )
    direct_evidence_pairwise_wins = _pairwise_wins(
        direct_evidence_rows,
        random_seed=random_seed,
        bootstrap_config=bootstrap,
    )
    return {
        "bootstrap": bootstrap,
        "model_performance": _model_performance_rows(summary_rows),
        "by_explainer": by_explainer,
        "by_dataset_track_explainer": _aggregate_summary_rows(
            summary_rows,
            ["dataset_id", "track", "explainer_id"],
            random_seed=random_seed,
            bootstrap_config=bootstrap,
        ),
        "by_model_explainer": _aggregate_summary_rows(
            summary_rows,
            ["model_id", "explainer_id"],
            random_seed=random_seed,
            bootstrap_config=bootstrap,
        ),
        "pairwise_wins": pairwise_wins,
        "pairwise_champions": _pairwise_champions(pairwise_wins),
        "evidence_source_summary": evidence_source_summary,
        "metric_rank_correlation": metric_rank_correlation,
        "metric_rank_correlation_summary": _metric_rank_correlation_summary(
            metric_rank_correlation
        ),
        "group_metric_correlation": group_metric_correlation,
        "group_metric_correlation_summary": _group_metric_correlation_summary(
            group_metric_correlation
        ),
        "metric_weight_profiles": metric_weight_profiles,
        "metric_weight_random_summary": metric_weight_random_summary,
        "metric_weight_random_meta": {
            "num_draws": random_weight_draws,
            "metrics": [
                _metric_label(metric) for metric in UNIVERSAL_COMPOSITE_METRICS
            ],
        },
        "metric_ablation_profiles": metric_ablation_profiles,
        "metric_ablation_summary": _metric_ablation_summary(metric_ablation_profiles),
        "proxy_sensitivity_profiles": _proxy_sensitivity_profiles(
            by_explainer=by_explainer,
            evidence_source_summary=evidence_source_summary,
        ),
        "direct_evidence_by_explainer": direct_evidence_by_explainer,
        "direct_evidence_profiles": _metric_weight_profiles(direct_evidence_by_explainer),
        "direct_evidence_pairwise_wins": direct_evidence_pairwise_wins,
        "direct_evidence_pairwise_champions": _pairwise_champions(
            direct_evidence_pairwise_wins
        ),
        "css_definition_profiles": css_definition_profiles,
        "css_definition_summary": _definition_profile_summary(
            css_definition_profiles,
            baseline_scenario="observed_css",
        ),
        "alf_definition_profiles": alf_definition_profiles,
        "alf_definition_summary": _definition_profile_summary(
            alf_definition_profiles,
            baseline_scenario="observed_alf",
        ),
        "track_policy_shift": track_policy_shift,
        "track_policy_by_explainer": track_policy_by_explainer,
        "track_policy_metric_summary": _track_policy_metric_summary(track_policy_shift),
        "track_policy_leader_stability": track_policy_leader_stability,
        "track_policy_leader_summary": track_policy_leader_summary,
        "track_policy_summary": _track_policy_summary(
            track_policy_by_explainer=track_policy_by_explainer,
            track_policy_metric_summary=_track_policy_metric_summary(track_policy_shift),
            track_policy_leader_summary=track_policy_leader_summary,
        ),
        "leave_one_dataset_out": leave_one_dataset_out,
        "leave_one_dataset_out_leaders": _leave_one_dataset_out_leaders(
            leave_one_dataset_out
        ),
        "subgroup_summary": _subgroup_summary(summary_rows),
        "subgroup_spread_summary": _subgroup_spread_summary(
            _subgroup_summary(summary_rows)
        ),
    }


def _build_markdown_report(payload: dict[str, Any], analysis: dict[str, Any]) -> str:
    overview = payload["overview"]
    direct_evidence_rows = analysis.get("direct_evidence_by_explainer", [])
    direct_evidence_profiles = analysis.get("direct_evidence_profiles", [])
    direct_evidence_pairwise = {
        row["metric"]: row
        for row in analysis.get("direct_evidence_pairwise_champions", [])
    }
    direct_evidence_faithfulness_leader = _best_row(
        direct_evidence_rows,
        "mean_faithfulness",
    )
    direct_evidence_balanced_leader = _best_row(
        [
            row
            for row in direct_evidence_profiles
            if row.get("scenario_name") == "balanced"
        ],
        "composite_score",
    )
    lines = [
        "# LegalBench-XAI Benchmark Report",
        "",
        f"- Experiment: `{payload['experiment_name']}`",
        f"- Generated at (UTC): `{payload['generated_at_utc']}`",
        f"- Total configurations: `{overview['total_groups']}`",
        f"- Datasets: `{', '.join(overview['datasets'])}`",
        f"- Tracks: `{', '.join(overview['tracks'])}`",
        f"- Models: `{', '.join(overview['models'])}`",
        f"- Explainers: `{', '.join(overview['explainers'])}`",
        "",
        "## Data Provenance",
        "",
        f"- Data source kinds: {_format_backend_usage(overview.get('source_usage', {}))}",
        f"- Evaluation split strategies: {_format_backend_usage(overview.get('evaluation_split_usage', {}))}",
        f"- Stability evidence sources: {_format_backend_usage(overview.get('stability_evidence_usage', {}))}",
        "",
        "## Backend Usage",
        "",
        f"- Model implementations: {_format_backend_usage(overview['backend_usage']['models'])}",
        f"- Explainer implementations: {_format_backend_usage(overview['backend_usage']['explainers'])}",
        "",
        "## Held-Out Model Performance",
        "",
        "| Dataset | Track | Model | Train Rows | Test Rows | Threshold Mode | Threshold | Accuracy | ROC AUC | PR AUC | Brier | ECE | True Adverse Rate | Predicted Adverse Rate | Split |",
        "| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for row in analysis["model_performance"]:
        lines.append(
            (
                "| {dataset_id} | {track} | {model_id} | {train_rows} | {test_rows} | "
                "{threshold_mode} | {threshold} | {accuracy} | {roc_auc} | {average_precision} | {brier_score} | {ece} | {positive_rate} | {predicted_positive_rate} | {split} |"
            ).format(
                dataset_id=row["dataset_id"],
                track=row["track"],
                model_id=row["model_id"],
                train_rows=row["train_rows"],
                test_rows=row["test_rows"],
                threshold_mode=row.get("threshold_policy_mode") or "fixed",
                threshold=_fmt(row.get("decision_threshold")),
                accuracy=_fmt(row["accuracy"]),
                roc_auc=_fmt(row["roc_auc"]),
                average_precision=_fmt(row.get("average_precision")),
                brier_score=_fmt(row.get("brier_score")),
                ece=_fmt(row.get("expected_calibration_error")),
                positive_rate=_fmt(row["positive_rate"]),
                predicted_positive_rate=_fmt(row["predicted_positive_rate"]),
                split=row["split_strategy"],
            )
        )

    lines.extend(
        [
            "",
        "## Explainer Leaderboard",
        "",
        "| Explainer | Configurations | Instances | Mean Faithfulness | Mean Stability | Mean Comprehensibility | Mean CSS | Mean Actionability | Mean ALF | Mean Invalid Recourse Rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for row in analysis["by_explainer"]:
        lines.append(
            (
                "| {explainer_id} | {group_count} | {total_instances} | {mean_faithfulness} | "
                "{mean_stability} | {mean_comprehensibility} | {mean_css} | {mean_actionability} | "
                "{mean_alf} | {mean_invalid} |"
            ).format(
                explainer_id=row["explainer_id"],
                group_count=row["group_count"],
                total_instances=row["total_instances"],
                mean_faithfulness=_fmt(row["mean_faithfulness"]),
                mean_stability=_fmt(row["mean_stability"]),
                mean_comprehensibility=_fmt(row["mean_comprehensibility"]),
                mean_css=_fmt(row["mean_css"]),
                mean_actionability=_fmt(row["mean_actionability"]),
                mean_alf=_fmt(row["mean_alf"]),
                mean_invalid=_fmt(row["mean_invalid_recourse_rate"]),
            )
        )

    lines.extend(
        [
            "",
            "## Bootstrap Uncertainty",
            "",
            _bootstrap_note_line(analysis),
            "",
            "| Explainer | Faithfulness CI | Stability CI | Comprehensibility CI | CSS CI | Actionability CI | ALF CI |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )

    for row in analysis["by_explainer"]:
        lines.append(
            (
                "| {explainer_id} | {faithfulness_ci} | {stability_ci} | {comprehensibility_ci} | "
                "{css_ci} | {actionability_ci} | {alf_ci} |"
            ).format(
                explainer_id=row["explainer_id"],
                faithfulness_ci=_fmt_ci(row, "mean_faithfulness"),
                stability_ci=_fmt_ci(row, "mean_stability"),
                comprehensibility_ci=_fmt_ci(row, "mean_comprehensibility"),
                css_ci=_fmt_ci(row, "mean_css"),
                actionability_ci=_fmt_ci(row, "mean_actionability"),
                alf_ci=_fmt_ci(row, "mean_alf"),
            )
        )

    correlation_summary = analysis["metric_rank_correlation_summary"]
    lines.extend(
        [
            "",
            "## Evidence-Source Transparency",
            "",
            "| Explainer | Instances | Direct Faithfulness Share | Proxy Faithfulness Share | Direct Stability Share | Stability Fallback Share |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in analysis["evidence_source_summary"]:
        lines.append(
            (
                "| {explainer} | {instances} | {faithfulness_direct} | {faithfulness_proxy} | "
                "{stability_direct} | {stability_fallback} |"
            ).format(
                explainer=row["explainer_id"],
                instances=row["total_instances"],
                faithfulness_direct=_fmt(row["faithfulness_direct_share"]),
                faithfulness_proxy=_fmt(row["faithfulness_proxy_share"]),
                stability_direct=_fmt(row["stability_direct_share"]),
                stability_fallback=_fmt(row["stability_fallback_share"]),
            )
        )

    group_correlation_summary = analysis["group_metric_correlation_summary"]
    lines.extend(
        [
            "",
            "## Metric Sensitivity",
            "",
            (
                f"- Strongest positive rank correlation: "
                f"`{correlation_summary['strongest_positive']['metric_a']}` vs "
                f"`{correlation_summary['strongest_positive']['metric_b']}` = "
                f"`{_fmt(correlation_summary['strongest_positive']['spearman_rank_corr'])}`"
                if correlation_summary.get("strongest_positive")
                else "- Strongest positive rank correlation: n/a"
            ),
            (
                f"- Weakest rank correlation: "
                f"`{correlation_summary['weakest']['metric_a']}` vs "
                f"`{correlation_summary['weakest']['metric_b']}` = "
                f"`{_fmt(correlation_summary['weakest']['spearman_rank_corr'])}`"
                if correlation_summary.get("weakest")
                else "- Weakest rank correlation: n/a"
            ),
            "",
            "| Metric A | Metric B | Spearman Rank Correlation | Shared Explainers |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for row in analysis["metric_rank_correlation"]:
        lines.append(
            (
                "| {metric_a} | {metric_b} | {corr} | {shared} |"
            ).format(
                metric_a=row["metric_a"],
                metric_b=row["metric_b"],
                corr=_fmt(row["spearman_rank_corr"]),
                shared=row["shared_explainers"],
            )
        )

    lines.extend(
        [
            "",
            "- Pairs with fewer than three shared explainers are descriptive only and should not be over-interpreted.",
            "",
            "## Group-Level Metric Correlation",
            "",
            (
                f"- Strongest positive group-level Spearman correlation: "
                f"`{group_correlation_summary['strongest_positive']['metric_a']}` vs "
                f"`{group_correlation_summary['strongest_positive']['metric_b']}` = "
                f"`{_fmt(group_correlation_summary['strongest_positive']['spearman_rank_corr'])}` "
                f"over `{group_correlation_summary['strongest_positive']['shared_groups']}` groups."
                if group_correlation_summary.get("strongest_positive")
                else "- Strongest positive group-level Spearman correlation: n/a"
            ),
            (
                f"- Weakest group-level Spearman correlation: "
                f"`{group_correlation_summary['weakest']['metric_a']}` vs "
                f"`{group_correlation_summary['weakest']['metric_b']}` = "
                f"`{_fmt(group_correlation_summary['weakest']['spearman_rank_corr'])}` "
                f"over `{group_correlation_summary['weakest']['shared_groups']}` groups."
                if group_correlation_summary.get("weakest")
                else "- Weakest group-level Spearman correlation: n/a"
            ),
            "",
            "| Metric A | Metric B | Spearman Rank Correlation | Shared Configurations |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for row in analysis["group_metric_correlation"]:
        lines.append(
            (
                "| {metric_a} | {metric_b} | {corr} | {shared} |"
            ).format(
                metric_a=row["metric_a"],
                metric_b=row["metric_b"],
                corr=_fmt(row["spearman_rank_corr"]),
                shared=row["shared_groups"],
            )
        )

    lines.extend(
        [
            "",
            "## Weight Sensitivity",
            "",
        ]
    )
    for row in analysis["metric_weight_random_summary"]:
        lines.append(
            (
                "- Randomized universal-weight sensitivity: `{explainer}` wins "
                "`{freq}` of draws with mean rank `{rank}`."
            ).format(
                explainer=row["explainer_id"],
                freq=_fmt(row["champion_frequency"]),
                rank=_fmt(row["mean_rank"]),
            )
        )

    lines.extend(
        [
            "",
            "| Scenario | Scope | Explainer | Composite Score | Rank |",
            "| --- | --- | --- | ---: | ---: |",
        ]
    )
    for row in analysis["metric_weight_profiles"]:
        lines.append(
            (
                "| {scenario} | {scope} | {explainer} | {score} | {rank} |"
            ).format(
                scenario=row["scenario_name"],
                scope=row["scope"],
                explainer=row["explainer_id"],
                score=_fmt(row["composite_score"]),
                rank=row["rank"],
            )
        )

    ablation_summary = analysis["metric_ablation_summary"]
    lines.extend(
        [
            "",
            "## Proxy Sensitivity",
            "",
            "| Scenario | Explainer | Adjusted Faithfulness | Proxy Share | Composite Score | Rank |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in analysis["proxy_sensitivity_profiles"]:
        lines.append(
            (
                "| {scenario} | {explainer} | {faithfulness} | {proxy_share} | {score} | {rank} |"
            ).format(
                scenario=row["scenario_name"],
                explainer=row["explainer_id"],
                faithfulness=_fmt(row["adjusted_mean_faithfulness"]),
                proxy_share=_fmt(row["faithfulness_proxy_share"]),
                score=_fmt(row["composite_score"]),
                rank=row["rank"],
            )
        )

    lines.extend(
        [
            "",
            "## Direct-Evidence-Only Sanity Check",
            "",
            (
                f"- Included groups with direct faithfulness evidence only: "
                f"`{sum(row['group_count'] for row in direct_evidence_rows)}` "
                f"across explainers "
                f"`{', '.join(row['explainer_id'] for row in direct_evidence_rows)}`."
                if direct_evidence_rows
                else "- Included groups with direct faithfulness evidence only: n/a."
            ),
            (
                f"- Direct-evidence faithfulness leader: "
                f"`{direct_evidence_faithfulness_leader['explainer_id']}` "
                f"with mean faithfulness `{_fmt(direct_evidence_faithfulness_leader['mean_faithfulness'])}`."
                if direct_evidence_faithfulness_leader
                else "- Direct-evidence faithfulness leader: n/a."
            ),
            (
                f"- Balanced direct-evidence composite leader: "
                f"`{direct_evidence_balanced_leader['explainer_id']}` "
                f"with composite score `{_fmt(direct_evidence_balanced_leader['composite_score'])}`."
                if direct_evidence_balanced_leader
                else "- Balanced direct-evidence composite leader: n/a."
            ),
            (
                f"- Direct-evidence pairwise faithfulness champion: "
                f"`{direct_evidence_pairwise['faithfulness']['explainer_id']}` "
                f"with `{direct_evidence_pairwise['faithfulness']['win_count']}` matched-configuration wins."
                if direct_evidence_pairwise.get("faithfulness")
                else "- Direct-evidence pairwise faithfulness champion: n/a."
            ),
            "",
            "| Explainer | Configurations | Instances | Faithfulness | Stability | Comprehensibility | CSS | Actionability |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in direct_evidence_rows:
        lines.append(
            (
                "| {explainer} | {groups} | {instances} | {faithfulness} | {stability} | {comprehensibility} | {css} | {actionability} |"
            ).format(
                explainer=row["explainer_id"],
                groups=row["group_count"],
                instances=row["total_instances"],
                faithfulness=_fmt(row["mean_faithfulness"]),
                stability=_fmt(row["mean_stability"]),
                comprehensibility=_fmt(row["mean_comprehensibility"]),
                css=_fmt(row["mean_css"]),
                actionability=_fmt(row["mean_actionability"]),
            )
        )

    lines.extend(
        [
            "",
            "| Scenario | Scope | Explainer | Composite Score | Rank |",
            "| --- | --- | --- | ---: | ---: |",
        ]
    )
    for row in direct_evidence_profiles:
        lines.append(
            (
                "| {scenario} | {scope} | {explainer} | {score} | {rank} |"
            ).format(
                scenario=row["scenario_name"],
                scope=row["scope"],
                explainer=row["explainer_id"],
                score=_fmt(row["composite_score"]),
                rank=row["rank"],
            )
        )

    lines.extend(
        [
            "",
            "## Metric Ablation",
            "",
            (
                f"- Balanced composite baseline leader: "
                f"`{ablation_summary['baseline_leader']['explainer_id']}`."
                if ablation_summary.get("baseline_leader")
                else "- Balanced composite baseline leader: n/a."
            ),
            (
                f"- Leave-one-metric-out scenarios preserving the same leader: "
                f"`{ablation_summary['leader_stability_count']}` / "
                f"`{ablation_summary['scenario_count']}`."
                if ablation_summary.get("baseline_leader")
                else "- Leave-one-metric-out leader stability: n/a."
            ),
            "",
            "| Scenario | Omitted Metric | Leader | Composite Score | Leader Changed |",
            "| --- | --- | --- | ---: | --- |",
        ]
    )
    for row in analysis["metric_ablation_profiles"]:
        if row["rank"] != 1:
            continue
        lines.append(
            (
                "| {scenario} | {omitted} | {leader} | {score} | {changed} |"
            ).format(
                scenario=row["scenario_name"],
                omitted=row["omitted_metric"] or "none",
                leader=row["explainer_id"],
                score=_fmt(row["composite_score"]),
                changed="yes" if row["leader_changed"] else "no",
            )
        )

    css_definition_summary = analysis.get("css_definition_summary", {})
    alf_definition_summary = analysis.get("alf_definition_summary", {})
    lines.extend(
        [
            "",
            "## Metric Definition Sensitivity",
            "",
            (
                f"- CSS baseline leader under observed scoring: "
                f"`{css_definition_summary['baseline_leader']['explainer_id']}`."
                if css_definition_summary.get("baseline_leader")
                else "- CSS baseline leader under observed scoring: n/a."
            ),
            (
                f"- CSS alternative definitions preserving the same leader: "
                f"`{css_definition_summary['leader_stability_count']}` / "
                f"`{css_definition_summary['scenario_count']}`."
                if css_definition_summary.get("baseline_leader")
                else "- CSS alternative-definition leader stability: n/a."
            ),
            (
                f"- ALF baseline leader under observed scoring: "
                f"`{alf_definition_summary['baseline_leader']['explainer_id']}`."
                if alf_definition_summary.get("baseline_leader")
                else "- ALF baseline leader under observed scoring: n/a."
            ),
            (
                f"- ALF alternative definitions preserving the same leader: "
                f"`{alf_definition_summary['leader_stability_count']}` / "
                f"`{alf_definition_summary['scenario_count']}`."
                if alf_definition_summary.get("baseline_leader")
                else "- ALF alternative-definition leader stability: n/a."
            ),
            "",
            "| Method | Scenario | Leader | Score | Leader Changed |",
            "| --- | --- | --- | ---: | --- |",
        ]
    )
    for row in analysis["css_definition_profiles"]:
        if row["rank"] != 1:
            continue
        lines.append(
            (
                "| CSS | {scenario} | {leader} | {score} | {changed} |"
            ).format(
                scenario=row["scenario_name"],
                leader=row["explainer_id"],
                score=_fmt(row["composite_score"]),
                changed="yes" if row.get("leader_changed") else "no",
            )
        )
    for row in analysis["alf_definition_profiles"]:
        if row["rank"] != 1:
            continue
        lines.append(
            (
                "| ALF | {scenario} | {leader} | {score} | {changed} |"
            ).format(
                scenario=row["scenario_name"],
                leader=row["explainer_id"],
                score=_fmt(row["composite_score"]),
                changed="yes" if row.get("leader_changed") else "no",
            )
        )

    track_policy_summary = analysis.get("track_policy_summary", {})
    lines.extend(
        [
            "",
            "## Track Policy Sensitivity",
            "",
            (
                f"- Largest average absolute primary-vs-audit shift: "
                f"`{track_policy_summary['largest_metric_shift']['metric']}` = "
                f"`{_fmt(track_policy_summary['largest_metric_shift']['mean_abs_delta'])}`."
                if track_policy_summary.get("largest_metric_shift")
                else "- Largest average absolute primary-vs-audit shift: n/a."
            ),
            (
                f"- Most track-sensitive explainer: "
                f"`{track_policy_summary['most_sensitive_explainer']['explainer_id']}` "
                f"with overall mean absolute delta "
                f"`{_fmt(track_policy_summary['most_sensitive_explainer']['overall_mean_abs_delta'])}`."
                if track_policy_summary.get("most_sensitive_explainer")
                else "- Most track-sensitive explainer: n/a."
            ),
            (
                f"- Most track-robust explainer: "
                f"`{track_policy_summary['most_robust_explainer']['explainer_id']}` "
                f"with overall mean absolute delta "
                f"`{_fmt(track_policy_summary['most_robust_explainer']['overall_mean_abs_delta'])}`."
                if track_policy_summary.get("most_robust_explainer")
                else "- Most track-robust explainer: n/a."
            ),
            (
                f"- Least leader-stable metric across primary/audit: "
                f"`{track_policy_summary['least_stable_metric']['metric']}` with same-leader rate "
                f"`{_fmt(track_policy_summary['least_stable_metric']['same_leader_rate'])}`."
                if track_policy_summary.get("least_stable_metric")
                else "- Least leader-stable metric across primary/audit: n/a."
            ),
            "",
            "| Explainer | Matched Pairs | Overall Mean Abs Delta | Abs Faithfulness | Abs Stability | Abs Comprehensibility | Abs CSS | Abs Actionability | Abs ALF |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in analysis["track_policy_by_explainer"]:
        lines.append(
            (
                "| {explainer} | {pairs} | {overall_abs} | {faithfulness} | {stability} | {comprehensibility} | {css} | {actionability} | {alf} |"
            ).format(
                explainer=row["explainer_id"],
                pairs=row["pair_count"],
                overall_abs=_fmt(row["overall_mean_abs_delta"]),
                faithfulness=_fmt(row["mean_abs_delta_faithfulness"]),
                stability=_fmt(row["mean_abs_delta_stability"]),
                comprehensibility=_fmt(row["mean_abs_delta_comprehensibility"]),
                css=_fmt(row["mean_abs_delta_css"]),
                actionability=_fmt(row["mean_abs_delta_actionability"]),
                alf=_fmt(row["mean_abs_delta_alf"]),
            )
        )

    lines.extend(
        [
            "",
            "| Metric | Same-Leader Rate | Same-Leader Count | Comparison Count |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in analysis["track_policy_leader_summary"]:
        lines.append(
            (
                "| {metric} | {rate} | {same_count} | {total} |"
            ).format(
                metric=row["metric"],
                rate=_fmt(row["same_leader_rate"]),
                same_count=row["same_leader_count"],
                total=row["comparison_count"],
            )
        )

    lines.extend(
        [
            "",
            "## Leave-One-Dataset-Out",
            "",
            "| Omitted Dataset | Metric | Leader | Score |",
            "| --- | --- | --- | ---: |",
        ]
    )
    for row in analysis["leave_one_dataset_out_leaders"]:
        lines.append(
            (
                "| {dataset} | {metric} | {leader} | {score} |"
            ).format(
                dataset=row["omitted_dataset_id"],
                metric=row["metric"],
                leader=row["explainer_id"],
                score=_fmt(row["score"]),
            )
        )

    subgroup_rows = analysis.get("subgroup_summary", [])
    if subgroup_rows:
        lines.extend(
            [
                "",
                "## Subpopulation Slices",
                "",
                "| Dataset | Track | Feature | Value | Explainer | Instances | Mean Stability | Mean CSS | Mean Actionability | Mean ALF |",
                "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in subgroup_rows[:30]:
            lines.append(
                (
                    "| {dataset_id} | {track} | {subgroup_feature} | {subgroup_value} | {explainer_id} | "
                    "{instance_count} | {mean_stability} | {mean_css} | {mean_actionability} | {mean_alf} |"
                ).format(
                    dataset_id=row["dataset_id"],
                    track=row["track"],
                    subgroup_feature=row["subgroup_feature"],
                    subgroup_value=row["subgroup_value"],
                    explainer_id=row["explainer_id"],
                    instance_count=row["instance_count"],
                    mean_stability=_fmt(row["mean_stability"]),
                    mean_css=_fmt(row["mean_css"]),
                    mean_actionability=_fmt(row["mean_actionability"]),
                    mean_alf=_fmt(row["mean_alf"]),
                )
            )

    subgroup_spread_rows = analysis.get("subgroup_spread_summary", [])
    if subgroup_spread_rows:
        lines.extend(
            [
                "",
                "## Subpopulation Spread Summary",
                "",
                "| Dataset | Track | Feature | Metric | Leader Set Across Subpopulations | Max-Spread Explainer | Max Spread |",
                "| --- | --- | --- | --- | --- | --- | ---: |",
            ]
        )
        for row in subgroup_spread_rows:
            lines.append(
                (
                    "| {dataset_id} | {track} | {subgroup_feature} | {metric} | {leader_set} | {spread_explainer} | {spread} |"
                ).format(
                    dataset_id=row["dataset_id"],
                    track=row["track"],
                    subgroup_feature=row["subgroup_feature"],
                    metric=row["metric"],
                    leader_set=row["leader_set"] or "n/a",
                    spread_explainer=row["max_spread_explainer"] or "n/a",
                    spread=_fmt(row["max_spread"]),
                )
            )

    lines.extend(
        [
            "",
            "## Dataset and Track Winners",
            "",
            "| Dataset | Track | Best Faithfulness | Best Stability | Best Comprehensibility | Best CSS | Best Actionability | Best ALF |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in _dataset_track_winners(analysis["by_dataset_track_explainer"]):
        lines.append(
            (
                "| {dataset_id} | {track} | {best_faithfulness} ({best_faithfulness_score}) | "
                "{best_stability} ({best_stability_score}) | "
                "{best_comprehensibility} ({best_comprehensibility_score}) | "
                "{best_css} ({best_css_score}) | "
                "{best_actionability} ({best_actionability_score}) | "
                "{best_alf} ({best_alf_score}) |"
            ).format(
                dataset_id=row["dataset_id"],
                track=row["track"],
                best_faithfulness=row["best_faithfulness_explainer"],
                best_faithfulness_score=_fmt(row["best_faithfulness_score"]),
                best_stability=row["best_stability_explainer"],
                best_stability_score=_fmt(row["best_stability_score"]),
                best_comprehensibility=row["best_comprehensibility_explainer"],
                best_comprehensibility_score=_fmt(row["best_comprehensibility_score"]),
                best_css=row["best_css_explainer"],
                best_css_score=_fmt(row["best_css_score"]),
                best_actionability=row["best_actionability_explainer"],
                best_actionability_score=_fmt(row["best_actionability_score"]),
                best_alf=row["best_alf_explainer"],
                best_alf_score=_fmt(row["best_alf_score"]),
            )
        )

    lines.extend(
        [
            "",
            "## Pairwise Win Leaders",
            "",
        ]
    )
    for row in analysis["pairwise_champions"]:
        lines.append(
            (
                "- `{metric}`: `{explainer_id}` leads with `{win_count}` wins across "
                "`{comparable_configurations}` comparable pairwise configuration matchups."
            ).format(
                metric=row["metric"],
                explainer_id=row["explainer_id"],
                win_count=row["win_count"],
                comparable_configurations=row["comparable_configurations"],
            )
        )

    lines.extend(
        [
            "",
            "## Best Configurations",
            "",
            _best_group_line(
                "Best faithfulness configuration",
                overview["best_faithfulness_group"],
                "mean_faithfulness",
            ),
            _best_group_line(
                "Best stability configuration",
                overview["best_stability_group"],
                "mean_stability",
            ),
            _best_group_line(
                "Best comprehensibility configuration",
                overview["best_comprehensibility_group"],
                "mean_comprehensibility",
            ),
            _best_group_line("Best CSS configuration", overview["best_css_group"], "mean_css"),
            _best_group_line(
                "Best actionability configuration",
                overview["best_actionability_group"],
                "mean_actionability",
            ),
            _best_group_line("Best ALF configuration", overview["best_alf_group"], "mean_alf"),
        ]
    )

    return "\n".join(lines) + "\n"


def _build_latex_leaderboard(analysis: dict[str, Any]) -> str:
    rows = analysis["by_explainer"]
    pairwise_totals = _pairwise_totals(analysis["pairwise_wins"])
    lines = [
        "% Auto-generated by LegalBench-XAI reporting",
        "\\begin{tabular}{lrrrrrrrrr}",
        "\\toprule",
        "Explainer & Configurations & Inst. & Faithfulness & Stability & Comprehensibility & CSS & Actionability & ALF & Pairwise Wins \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            (
                "{explainer} & {groups} & {instances} & {faithfulness} & {stability} & {comprehensibility} & {css} & "
                "{actionability} & {alf} & {pairwise_total} \\\\"
            ).format(
                explainer=_tex_escape(row["explainer_id"]),
                groups=row["group_count"],
                instances=row["total_instances"],
                faithfulness=_fmt(row["mean_faithfulness"]),
                stability=_fmt(row["mean_stability"]),
                comprehensibility=_fmt(row["mean_comprehensibility"]),
                css=_fmt(row["mean_css"]),
                actionability=_fmt(row["mean_actionability"]),
                alf=_fmt(row["mean_alf"]),
                pairwise_total=pairwise_totals.get(row["explainer_id"], 0),
            )
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "",
        ]
    )
    return "\n".join(lines)


def _best_group_line(label: str, group: dict[str, Any] | None, metric_key: str) -> str:
    if group is None:
        return f"- {label}: n/a"
    return (
        f"- {label}: `{group['dataset_id']}` / `{group['track']}` / "
        f"`{group['model_id']}` / `{group['explainer_id']}` with `{metric_key}={_fmt(group[metric_key])}`"
    )


def _format_backend_usage(items: dict[str, int]) -> str:
    if not items:
        return "n/a"
    parts = [f"`{name}` x {count}" for name, count in sorted(items.items())]
    return ", ".join(parts)


def _metric_label(metric_key: str) -> str:
    return metric_key.removeprefix("mean_")


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _aggregate_summary_rows(
    rows: list[dict[str, Any]],
    group_keys: list[str],
    *,
    random_seed: int,
    bootstrap_config: dict[str, Any],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)

    aggregated_rows: list[dict[str, Any]] = []
    for group_index, key in enumerate(sorted(grouped)):
        items = grouped[key]
        aggregated = {name: value for name, value in zip(group_keys, key)}
        aggregated.update(
            {
                "group_count": len(items),
                "total_instances": sum(item["evaluated_rows"] for item in items),
                "mean_faithfulness": _weighted_mean(items, "mean_faithfulness"),
                "mean_stability": _weighted_mean(items, "mean_stability"),
                "mean_comprehensibility": _weighted_mean(items, "mean_comprehensibility"),
                "mean_css": _weighted_mean(items, "mean_css"),
                "mean_actionability": _weighted_mean(items, "mean_actionability"),
                "mean_alf": _weighted_mean(items, "mean_alf"),
                "mean_invalid_recourse_rate": _weighted_mean(
                    items,
                    "mean_invalid_recourse_rate",
                ),
                "mean_adverse_rate": _weighted_mean(items, "adverse_rate", weight_key="total_rows"),
            }
        )
        aggregated.update(
            _bootstrap_summary_intervals(
                items,
                metrics=AGGREGATE_METRICS,
                bootstrap_config=bootstrap_config,
                group_keys=group_keys,
                seed=random_seed + (group_index + 1) * 9973,
            )
        )
        aggregated_rows.append(aggregated)
    return aggregated_rows


def _weighted_mean(
    rows: list[dict[str, Any]],
    value_key: str,
    weight_key: str = "evaluated_rows",
) -> float | None:
    weighted_sum = 0.0
    total_weight = 0.0
    for row in rows:
        value = row.get(value_key)
        weight = float(row.get(weight_key, 0))
        if value is None or weight <= 0:
            continue
        weighted_sum += float(value) * weight
        total_weight += weight
    if total_weight <= 0:
        return None
    return weighted_sum / total_weight


def _mean_non_null(rows: list[dict[str, Any]], value_key: str) -> float | None:
    values = [float(row[value_key]) for row in rows if row.get(value_key) is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _evidence_source_summary(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "total_instances": 0.0,
            "faithfulness_direct_count": 0.0,
            "faithfulness_proxy_count": 0.0,
            "stability_direct_count": 0.0,
            "stability_fallback_count": 0.0,
        }
    )
    for row in summary_rows:
        explainer_id = row["explainer_id"]
        total_instances = float(row.get("evaluated_rows", 0))
        grouped[explainer_id]["total_instances"] += total_instances

        faithfulness_counts = row.get("faithfulness_evidence_sources") or {}
        if not faithfulness_counts:
            if explainer_id in {"dice", "wachter"}:
                faithfulness_counts = {"structural_proxy": int(total_instances)}
            else:
                faithfulness_counts = {"direct_reference": int(total_instances)}
        grouped[explainer_id]["faithfulness_proxy_count"] += float(
            faithfulness_counts.get("structural_proxy", 0)
        )
        grouped[explainer_id]["faithfulness_direct_count"] += float(
            sum(
                count
                for source, count in faithfulness_counts.items()
                if source != "structural_proxy"
            )
        )

        stability_counts = row.get("stability_evidence_sources") or {}
        if not stability_counts:
            stability_counts = {"structural_proxy": int(total_instances)}
        grouped[explainer_id]["stability_direct_count"] += float(
            stability_counts.get("perturbation_probe", 0)
        )
        grouped[explainer_id]["stability_fallback_count"] += float(
            sum(
                count
                for source, count in stability_counts.items()
                if source != "perturbation_probe"
            )
        )

    rows: list[dict[str, Any]] = []
    for explainer_id in sorted(grouped):
        item = grouped[explainer_id]
        total_instances = item["total_instances"]
        rows.append(
            {
                "explainer_id": explainer_id,
                "total_instances": int(total_instances),
                "faithfulness_direct_count": int(item["faithfulness_direct_count"]),
                "faithfulness_proxy_count": int(item["faithfulness_proxy_count"]),
                "faithfulness_direct_share": (
                    item["faithfulness_direct_count"] / total_instances
                    if total_instances
                    else None
                ),
                "faithfulness_proxy_share": (
                    item["faithfulness_proxy_count"] / total_instances
                    if total_instances
                    else None
                ),
                "stability_direct_count": int(item["stability_direct_count"]),
                "stability_fallback_count": int(item["stability_fallback_count"]),
                "stability_direct_share": (
                    item["stability_direct_count"] / total_instances
                    if total_instances
                    else None
                ),
                "stability_fallback_share": (
                    item["stability_fallback_count"] / total_instances
                    if total_instances
                    else None
                ),
            }
        )
    return rows


def _faithfulness_proxy_count(row: dict[str, Any]) -> int:
    counts = row.get("faithfulness_evidence_sources") or {}
    if not counts:
        if row.get("explainer_id") in {"dice", "wachter"}:
            return int(row.get("evaluated_rows", 0))
        return 0
    return int(counts.get("structural_proxy", 0))


def _direct_faithfulness_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in summary_rows if _faithfulness_proxy_count(row) == 0]


def _proxy_sensitivity_profiles(
    *,
    by_explainer: list[dict[str, Any]],
    evidence_source_summary: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    proxy_share_by_explainer = {
        row["explainer_id"]: float(row.get("faithfulness_proxy_share") or 0.0)
        for row in evidence_source_summary
    }
    scenario_adjusters = {
        "observed": lambda faithfulness, proxy_share: faithfulness,
        "proxy_discount_25": lambda faithfulness, proxy_share: faithfulness
        * (1.0 - 0.25 * proxy_share),
        "proxy_cap_050": lambda faithfulness, proxy_share: (
            min(faithfulness, 0.50) if proxy_share > 0 else faithfulness
        ),
    }
    rows: list[dict[str, Any]] = []
    for scenario_name, adjuster in scenario_adjusters.items():
        adjusted_rows: list[dict[str, Any]] = []
        for row in by_explainer:
            proxy_share = proxy_share_by_explainer.get(row["explainer_id"], 0.0)
            adjusted_rows.append(
                {
                    **row,
                    "mean_faithfulness": adjuster(
                        float(row["mean_faithfulness"]),
                        proxy_share,
                    ),
                    "adjusted_mean_faithfulness": adjuster(
                        float(row["mean_faithfulness"]),
                        proxy_share,
                    ),
                    "faithfulness_proxy_share": proxy_share,
                }
            )
        normalized = _normalize_metric_values(
            adjusted_rows,
            list(UNIVERSAL_WEIGHT_PROFILES["balanced"]),
        )
        scored_rows: list[dict[str, Any]] = []
        for row in adjusted_rows:
            composite = sum(
                normalized[metric][row["explainer_id"]] * weight
                for metric, weight in UNIVERSAL_WEIGHT_PROFILES["balanced"].items()
            )
            scored_rows.append(
                {
                    "scenario_name": scenario_name,
                    "explainer_id": row["explainer_id"],
                    "adjusted_mean_faithfulness": row["adjusted_mean_faithfulness"],
                    "faithfulness_proxy_share": row["faithfulness_proxy_share"],
                    "composite_score": composite,
                }
            )
        ranked = sorted(
            scored_rows,
            key=lambda item: (-item["composite_score"], item["explainer_id"]),
        )
        for rank, row in enumerate(ranked, start=1):
            row["rank"] = rank
        rows.extend(ranked)
    return rows


def _subgroup_summary(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for row in summary_rows:
        for item in row.get("instance_results", []):
            subgroup_attributes = item.get("subgroup_attributes") or {}
            if not subgroup_attributes:
                continue
            for subgroup_feature, subgroup_value in subgroup_attributes.items():
                key = (
                    row["dataset_id"],
                    row["track"],
                    subgroup_feature,
                    str(subgroup_value),
                    row["explainer_id"],
                )
                bucket = grouped.setdefault(
                    key,
                    {
                        "dataset_id": row["dataset_id"],
                        "track": row["track"],
                        "subgroup_feature": subgroup_feature,
                        "subgroup_value": str(subgroup_value),
                        "explainer_id": row["explainer_id"],
                        "instance_count": 0,
                        "mean_faithfulness": 0.0,
                        "mean_stability": 0.0,
                        "mean_comprehensibility": 0.0,
                        "mean_css": 0.0,
                        "mean_actionability": 0.0,
                        "mean_alf": 0.0,
                        "_alf_count": 0,
                    },
                )
                scores = item["scores"]
                bucket["instance_count"] += 1
                bucket["mean_faithfulness"] += float(
                    scores["faithfulness_score"]["score"]
                )
                bucket["mean_stability"] += float(scores["stability_score"]["score"])
                bucket["mean_comprehensibility"] += float(
                    scores["comprehensibility_score"]["score"]
                )
                bucket["mean_css"] += float(
                    scores["contestability_support_score"]["score"]
                )
                bucket["mean_actionability"] += float(
                    scores["unified_actionability_score"]["score"]
                )
                alf_score = scores["actionability_legality_filter"]["score"]
                if alf_score is not None:
                    bucket["mean_alf"] += float(alf_score)
                    bucket["_alf_count"] += 1

    rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        bucket = grouped[key]
        count = max(1, int(bucket["instance_count"]))
        alf_count = int(bucket["_alf_count"])
        rows.append(
            {
                "dataset_id": bucket["dataset_id"],
                "track": bucket["track"],
                "subgroup_feature": bucket["subgroup_feature"],
                "subgroup_value": bucket["subgroup_value"],
                "explainer_id": bucket["explainer_id"],
                "instance_count": count,
                "mean_faithfulness": bucket["mean_faithfulness"] / count,
                "mean_stability": bucket["mean_stability"] / count,
                "mean_comprehensibility": bucket["mean_comprehensibility"] / count,
                "mean_css": bucket["mean_css"] / count,
                "mean_actionability": bucket["mean_actionability"] / count,
                "mean_alf": (bucket["mean_alf"] / alf_count) if alf_count else None,
            }
        )
    return rows


def _subgroup_spread_summary(subgroup_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_keys = [
        ("mean_faithfulness", "faithfulness"),
        ("mean_stability", "stability"),
        ("mean_css", "css"),
        ("mean_actionability", "actionability"),
        ("mean_alf", "alf"),
    ]
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in subgroup_rows:
        for metric_key, metric_label in metric_keys:
            if row.get(metric_key) is None:
                continue
            grouped[
                (
                    row["dataset_id"],
                    row["track"],
                    row["subgroup_feature"],
                    metric_label,
                )
            ].append(row)

    summary: list[dict[str, Any]] = []
    for key in sorted(grouped):
        dataset_id, track, subgroup_feature, metric_label = key
        rows = grouped[key]
        by_explainer: dict[str, list[dict[str, Any]]] = defaultdict(list)
        by_subgroup: dict[str, list[dict[str, Any]]] = defaultdict(list)
        metric_key = f"mean_{metric_label}"
        if metric_label == "alf":
            metric_key = "mean_alf"
        for row in rows:
            by_explainer[row["explainer_id"]].append(row)
            by_subgroup[row["subgroup_value"]].append(row)

        spread_rows: list[tuple[float, str]] = []
        for explainer_id, explainer_rows in by_explainer.items():
            values = [float(row[metric_key]) for row in explainer_rows if row.get(metric_key) is not None]
            if not values:
                continue
            spread_rows.append((max(values) - min(values), explainer_id))

        if spread_rows:
            max_spread, max_spread_explainer = sorted(
                spread_rows,
                key=lambda item: (-item[0], item[1]),
            )[0]
        else:
            max_spread = None
            max_spread_explainer = None

        leader_set: set[str] = set()
        for subgroup_value, subgroup_value_rows in by_subgroup.items():
            eligible = [
                row for row in subgroup_value_rows if row.get(metric_key) is not None
            ]
            if not eligible:
                continue
            leader = sorted(
                eligible,
                key=lambda row: (-float(row[metric_key]), row["explainer_id"]),
            )[0]
            leader_set.add(leader["explainer_id"])

        summary.append(
            {
                "dataset_id": dataset_id,
                "track": track,
                "subgroup_feature": subgroup_feature,
                "metric": metric_label,
                "subgroup_count": len(by_subgroup),
                "leader_set": "|".join(sorted(leader_set)),
                "max_spread_explainer": max_spread_explainer,
                "max_spread": max_spread,
            }
        )
    return summary


def _leader_for_metric(
    rows: list[dict[str, Any]],
    metric_key: str,
) -> dict[str, Any] | None:
    eligible = [row for row in rows if row.get(metric_key) is not None]
    if not eligible:
        return None
    return sorted(
        eligible,
        key=lambda row: (-float(row[metric_key]), row["explainer_id"]),
    )[0]


def _pairwise_wins(
    summary_rows: list[dict[str, Any]],
    *,
    random_seed: int,
    bootstrap_config: dict[str, Any],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        grouped[(row["dataset_id"], row["track"], row["model_id"])].append(row)

    pairwise_rows: list[dict[str, Any]] = []
    metric_keys = [
        ("mean_faithfulness", "faithfulness"),
        ("mean_stability", "stability"),
        ("mean_comprehensibility", "comprehensibility"),
        ("mean_css", "css"),
        ("mean_actionability", "actionability"),
        ("mean_alf", "alf"),
    ]
    for metric_index, (metric_key, metric_name) in enumerate(metric_keys):
        comparison_totals: dict[tuple[str, str], dict[str, int]] = {}
        for rows in grouped.values():
            rows_by_explainer = {row["explainer_id"]: row for row in rows if row[metric_key] is not None}
            for explainer_a, explainer_b in combinations(sorted(rows_by_explainer), 2):
                row_a = rows_by_explainer[explainer_a]
                row_b = rows_by_explainer[explainer_b]
                key = (explainer_a, explainer_b)
                totals = comparison_totals.setdefault(
                    key,
                    {
                        "metric": metric_name,
                        "explainer_a": explainer_a,
                        "explainer_b": explainer_b,
                        "wins_a": 0,
                        "wins_b": 0,
                        "ties": 0,
                        "comparable_configurations": 0,
                        "_matchups": [],
                    },
                )
                totals["comparable_configurations"] += 1
                if row_a[metric_key] > row_b[metric_key]:
                    totals["wins_a"] += 1
                    outcome_a = 1.0
                    outcome_b = 0.0
                elif row_b[metric_key] > row_a[metric_key]:
                    totals["wins_b"] += 1
                    outcome_a = 0.0
                    outcome_b = 1.0
                else:
                    totals["ties"] += 1
                    outcome_a = 0.0
                    outcome_b = 0.0
                totals["_matchups"].append(
                    {
                        "dataset_id": row_a["dataset_id"],
                        "track": row_a["track"],
                        "model_id": row_a["model_id"],
                        "outcome_a": outcome_a,
                        "outcome_b": outcome_b,
                    }
                )
        for pair_index, key in enumerate(sorted(comparison_totals)):
            item = comparison_totals[key]
            comparable = item["comparable_configurations"]
            item["win_rate_a"] = item["wins_a"] / comparable if comparable else None
            item["win_rate_b"] = item["wins_b"] / comparable if comparable else None
            matchups = item.pop("_matchups")
            ci_a = _bootstrap_binary_rate_interval(
                rows=matchups,
                outcome_key="outcome_a",
                bootstrap_config=bootstrap_config,
                seed=random_seed + metric_index * 10007 + pair_index * 97 + 1,
            )
            ci_b = _bootstrap_binary_rate_interval(
                rows=matchups,
                outcome_key="outcome_b",
                bootstrap_config=bootstrap_config,
                seed=random_seed + metric_index * 10007 + pair_index * 97 + 2,
            )
            item["win_rate_a_ci_low"] = ci_a[0]
            item["win_rate_a_ci_high"] = ci_a[1]
            item["win_rate_b_ci_low"] = ci_b[0]
            item["win_rate_b_ci_high"] = ci_b[1]
            pairwise_rows.append(item)
    return sorted(
        pairwise_rows,
        key=lambda row: (row["metric"], row["explainer_a"], row["explainer_b"]),
    )


def _pairwise_champions(pairwise_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    per_metric: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"win_count": 0, "comparable_configurations": 0})
    )
    for row in pairwise_rows:
        metric = row["metric"]
        per_metric[metric][row["explainer_a"]]["win_count"] += row["wins_a"]
        per_metric[metric][row["explainer_a"]]["comparable_configurations"] += row["comparable_configurations"]
        per_metric[metric][row["explainer_b"]]["win_count"] += row["wins_b"]
        per_metric[metric][row["explainer_b"]]["comparable_configurations"] += row["comparable_configurations"]

    champions: list[dict[str, Any]] = []
    for metric in sorted(per_metric):
        best_explainer, stats = max(
            per_metric[metric].items(),
            key=lambda item: (item[1]["win_count"], item[0]),
        )
        champions.append(
            {
                "metric": metric,
                "explainer_id": best_explainer,
                "win_count": stats["win_count"],
                "comparable_configurations": stats["comparable_configurations"],
            }
        )
    return champions


def _metric_rank_correlation(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    correlations: list[dict[str, Any]] = []
    for metric_a, metric_a_label in METRIC_ANALYSIS_FIELDS:
        for metric_b, metric_b_label in METRIC_ANALYSIS_FIELDS:
            if metric_a_label >= metric_b_label:
                continue
            shared_rows = [
                row
                for row in rows
                if row.get(metric_a) is not None and row.get(metric_b) is not None
            ]
            correlation = _spearman_rank_correlation(shared_rows, metric_a, metric_b)
            correlations.append(
                {
                    "metric_a": metric_a_label,
                    "metric_b": metric_b_label,
                    "spearman_rank_corr": correlation,
                    "shared_explainers": len(shared_rows),
                }
            )
    return sorted(correlations, key=lambda row: (row["metric_a"], row["metric_b"]))


def _group_metric_correlation(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    correlations: list[dict[str, Any]] = []
    for metric_a, metric_a_label in METRIC_ANALYSIS_FIELDS:
        for metric_b, metric_b_label in METRIC_ANALYSIS_FIELDS:
            if metric_a_label >= metric_b_label:
                continue
            shared_rows = [
                row
                for row in summary_rows
                if row.get(metric_a) is not None and row.get(metric_b) is not None
            ]
            correlation = _spearman_correlation_from_values(
                [float(row[metric_a]) for row in shared_rows],
                [float(row[metric_b]) for row in shared_rows],
            )
            correlations.append(
                {
                    "metric_a": metric_a_label,
                    "metric_b": metric_b_label,
                    "spearman_rank_corr": correlation,
                    "shared_groups": len(shared_rows),
                }
            )
    return sorted(correlations, key=lambda row: (row["metric_a"], row["metric_b"]))


def _metric_weight_profiles(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    profile_rows: list[dict[str, Any]] = []
    for scenario_name, weights in UNIVERSAL_WEIGHT_PROFILES.items():
        profile_rows.extend(
            _composite_profile_rows(
                rows,
                scenario_name=scenario_name,
                scope="all_explainers",
                weights=weights,
            )
        )
    recourse_rows = [row for row in rows if row.get("mean_alf") is not None]
    for scenario_name, weights in RECOURSE_WEIGHT_PROFILES.items():
        profile_rows.extend(
            _composite_profile_rows(
                recourse_rows,
                scenario_name=scenario_name,
                scope="recourse_only",
                weights=weights,
            )
        )
    return profile_rows


def _composite_profile_rows(
    rows: list[dict[str, Any]],
    *,
    scenario_name: str,
    scope: str,
    weights: dict[str, float],
) -> list[dict[str, Any]]:
    if not rows:
        return []
    normalized = _normalize_metric_values(rows, list(weights))
    scored_rows: list[dict[str, Any]] = []
    for row in rows:
        composite = sum(
            normalized[metric][row["explainer_id"]] * weight
            for metric, weight in weights.items()
        )
        scored_rows.append(
            {
                "scenario_name": scenario_name,
                "scope": scope,
                "explainer_id": row["explainer_id"],
                "composite_score": composite,
            }
        )
    ranked = sorted(
        scored_rows,
        key=lambda item: (-item["composite_score"], item["explainer_id"]),
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return ranked


def _metric_weight_random_summary(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    num_draws: int = 1000,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    rng = random.Random(seed)
    normalized = _normalize_metric_values(rows, UNIVERSAL_COMPOSITE_METRICS)
    champion_counts = {row["explainer_id"]: 0 for row in rows}
    rank_totals = {row["explainer_id"]: 0.0 for row in rows}
    for _ in range(num_draws):
        weights = _random_weight_vector(UNIVERSAL_COMPOSITE_METRICS, rng)
        scored = []
        for row in rows:
            composite = sum(
                normalized[metric][row["explainer_id"]] * weights[metric]
                for metric in UNIVERSAL_COMPOSITE_METRICS
            )
            scored.append((row["explainer_id"], composite))
        ranked = sorted(scored, key=lambda item: (-item[1], item[0]))
        champion_counts[ranked[0][0]] += 1
        for rank, (explainer_id, _score) in enumerate(ranked, start=1):
            rank_totals[explainer_id] += rank
    summary_rows = []
    for explainer_id in sorted(champion_counts):
        summary_rows.append(
            {
                "explainer_id": explainer_id,
                "champion_frequency": champion_counts[explainer_id] / num_draws,
                "mean_rank": rank_totals[explainer_id] / num_draws,
                "num_draws": num_draws,
            }
        )
    return sorted(
        summary_rows,
        key=lambda row: (-row["champion_frequency"], row["mean_rank"], row["explainer_id"]),
    )


def _metric_ablation_profiles(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    baseline_weights = {
        metric: 1.0 / len(UNIVERSAL_COMPOSITE_METRICS)
        for metric in UNIVERSAL_COMPOSITE_METRICS
    }
    scenarios.extend(
        _composite_profile_rows(
            rows,
            scenario_name="all_metrics_balanced",
            scope="all_explainers",
            weights=baseline_weights,
        )
    )
    for omitted_metric in UNIVERSAL_COMPOSITE_METRICS:
        remaining = [
            metric for metric in UNIVERSAL_COMPOSITE_METRICS if metric != omitted_metric
        ]
        weights = {metric: 1.0 / len(remaining) for metric in remaining}
        scenario_rows = _composite_profile_rows(
            rows,
            scenario_name=f"drop_{_metric_label(omitted_metric)}",
            scope="all_explainers",
            weights=weights,
        )
        for row in scenario_rows:
            row["omitted_metric"] = _metric_label(omitted_metric)
        scenarios.extend(scenario_rows)
    for row in scenarios:
        row.setdefault("omitted_metric", None)
    summary = _metric_ablation_summary(scenarios)
    baseline_leader = summary.get("baseline_leader", {})
    baseline_name = baseline_leader.get("explainer_id")
    for row in scenarios:
        row["leader_changed"] = (
            row["rank"] == 1
            and baseline_name is not None
            and row["explainer_id"] != baseline_name
            and row["scenario_name"] != "all_metrics_balanced"
        )
    return scenarios


def _metric_ablation_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "baseline_leader": None,
            "leader_stability_count": 0,
            "scenario_count": 0,
        }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["scenario_name"]].append(row)
    baseline_rows = grouped.get("all_metrics_balanced", [])
    baseline_leader = next((row for row in baseline_rows if row["rank"] == 1), None)
    if baseline_leader is None:
        return {
            "baseline_leader": None,
            "leader_stability_count": 0,
            "scenario_count": max(0, len(grouped) - 1),
        }
    stability_count = 0
    scenario_count = 0
    for scenario_name, scenario_rows in grouped.items():
        if scenario_name == "all_metrics_balanced":
            continue
        leader = next((row for row in scenario_rows if row["rank"] == 1), None)
        if leader is None:
            continue
        scenario_count += 1
        if leader["explainer_id"] == baseline_leader["explainer_id"]:
            stability_count += 1
    return {
        "baseline_leader": baseline_leader,
        "leader_stability_count": stability_count,
        "scenario_count": scenario_count,
    }


def _component_aggregates_by_explainer(
    summary_rows: list[dict[str, Any]],
    metric_keys: list[str],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        grouped[row["explainer_id"]].append(row)

    rows: list[dict[str, Any]] = []
    for explainer_id in sorted(grouped):
        items = grouped[explainer_id]
        aggregated = {
            "explainer_id": explainer_id,
            "group_count": len(items),
            "total_instances": sum(int(item.get("evaluated_rows", 0)) for item in items),
        }
        for metric_key in metric_keys:
            aggregated[metric_key] = _weighted_mean(items, metric_key)
        rows.append(aggregated)
    return rows


def _definition_profile_rows(
    rows: list[dict[str, Any]],
    *,
    family: str,
    scenario_name: str,
    score_fn,
) -> list[dict[str, Any]]:
    scored_rows: list[dict[str, Any]] = []
    for row in rows:
        score = score_fn(row)
        if score is None:
            continue
        scored_rows.append(
            {
                "family": family,
                "scenario_name": scenario_name,
                "explainer_id": row["explainer_id"],
                "composite_score": score,
            }
        )
    ranked = sorted(
        scored_rows,
        key=lambda item: (-item["composite_score"], item["explainer_id"]),
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return ranked


def _css_definition_profiles(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    component_keys = ["mean_css"] + [field for field, _label in CSS_COMPONENT_FIELDS]
    rows = _component_aggregates_by_explainer(summary_rows, component_keys)
    if not rows or not any(row.get("mean_css_verifiability") is not None for row in rows):
        return []

    profiles: list[dict[str, Any]] = []
    profiles.extend(
        _definition_profile_rows(
            rows,
            family="css",
            scenario_name="observed_css",
            score_fn=lambda row: row.get("mean_css"),
        )
    )
    profiles.extend(
        _definition_profile_rows(
            rows,
            family="css",
            scenario_name="reconstructed_css",
            score_fn=lambda row: _weighted_component_score(
                row,
                {
                    "mean_css_verifiability": 0.30,
                    "mean_css_correctability": 0.30,
                    "mean_css_decision_sensitivity_visibility": 0.25,
                    "mean_css_reason_specificity": 0.15,
                },
            ),
        )
    )
    profiles.extend(
        _definition_profile_rows(
            rows,
            family="css",
            scenario_name="verifiability_focused_css",
            score_fn=lambda row: _weighted_component_score(
                row,
                {
                    "mean_css_verifiability": 0.45,
                    "mean_css_correctability": 0.25,
                    "mean_css_decision_sensitivity_visibility": 0.20,
                    "mean_css_reason_specificity": 0.10,
                },
            ),
        )
    )
    profiles.extend(
        _definition_profile_rows(
            rows,
            family="css",
            scenario_name="challenge_focused_css",
            score_fn=lambda row: _weighted_component_score(
                row,
                {
                    "mean_css_verifiability": 0.20,
                    "mean_css_correctability": 0.25,
                    "mean_css_decision_sensitivity_visibility": 0.35,
                    "mean_css_reason_specificity": 0.20,
                },
            ),
        )
    )
    profiles.extend(
        _definition_profile_rows(
            rows,
            family="css",
            scenario_name="specificity_light_css",
            score_fn=lambda row: _weighted_component_score(
                row,
                {
                    "mean_css_verifiability": 0.38,
                    "mean_css_correctability": 0.32,
                    "mean_css_decision_sensitivity_visibility": 0.30,
                },
            ),
        )
    )
    _mark_definition_leader_changes(
        profiles,
        baseline_scenario="observed_css",
    )
    return profiles


def _alf_definition_profiles(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    component_keys = ["mean_alf"] + [field for field, _label in ALF_COMPONENT_FIELDS]
    rows = _component_aggregates_by_explainer(summary_rows, component_keys)
    rows = [row for row in rows if row.get("mean_alf") is not None]
    if not rows or not any(row.get("mean_alf_feasibility") is not None for row in rows):
        return []

    profiles: list[dict[str, Any]] = []
    profiles.extend(
        _definition_profile_rows(
            rows,
            family="alf",
            scenario_name="observed_alf",
            score_fn=lambda row: row.get("mean_alf"),
        )
    )
    profiles.extend(
        _definition_profile_rows(
            rows,
            family="alf",
            scenario_name="balanced_arithmetic_alf",
            score_fn=lambda row: _weighted_component_score(
                row,
                {
                    "mean_alf_feasibility": 1.0 / 3.0,
                    "mean_alf_legality": 1.0 / 3.0,
                    "mean_alf_minimality": 1.0 / 3.0,
                },
            ),
        )
    )
    profiles.extend(
        _definition_profile_rows(
            rows,
            family="alf",
            scenario_name="legality_dominant_alf",
            score_fn=lambda row: _weighted_component_score(
                row,
                {
                    "mean_alf_feasibility": 0.25,
                    "mean_alf_legality": 0.60,
                    "mean_alf_minimality": 0.15,
                },
            ),
        )
    )
    profiles.extend(
        _definition_profile_rows(
            rows,
            family="alf",
            scenario_name="feasibility_dominant_alf",
            score_fn=lambda row: _weighted_component_score(
                row,
                {
                    "mean_alf_feasibility": 0.50,
                    "mean_alf_legality": 0.30,
                    "mean_alf_minimality": 0.20,
                },
            ),
        )
    )
    profiles.extend(
        _definition_profile_rows(
            rows,
            family="alf",
            scenario_name="minimality_dominant_alf",
            score_fn=lambda row: _weighted_component_score(
                row,
                {
                    "mean_alf_feasibility": 0.25,
                    "mean_alf_legality": 0.25,
                    "mean_alf_minimality": 0.50,
                },
            ),
        )
    )
    profiles.extend(
        _definition_profile_rows(
            rows,
            family="alf",
            scenario_name="invalidity_penalized_alf",
            score_fn=lambda row: _penalized_component_score(
                row,
                penalty_key="mean_invalid_recourse_rate",
            ),
        )
    )
    profiles.extend(
        _definition_profile_rows(
            rows,
            family="alf",
            scenario_name="hard_failure_penalized_alf",
            score_fn=lambda row: _penalized_component_score(
                row,
                penalty_key="mean_alf_hard_failure_rate",
            ),
        )
    )
    _mark_definition_leader_changes(
        profiles,
        baseline_scenario="observed_alf",
    )
    return profiles


def _weighted_component_score(
    row: dict[str, Any],
    weights: dict[str, float],
) -> float | None:
    values: list[float] = []
    for key, weight in weights.items():
        value = row.get(key)
        if value is None:
            return None
        values.append(float(value) * weight)
    return sum(values)


def _penalized_component_score(
    row: dict[str, Any],
    *,
    penalty_key: str,
) -> float | None:
    base = _weighted_component_score(
        row,
        {
            "mean_alf_feasibility": 1.0 / 3.0,
            "mean_alf_legality": 1.0 / 3.0,
            "mean_alf_minimality": 1.0 / 3.0,
        },
    )
    penalty = row.get(penalty_key)
    if base is None or penalty is None:
        return None
    return base * max(0.0, 1.0 - float(penalty))


def _mark_definition_leader_changes(
    rows: list[dict[str, Any]],
    *,
    baseline_scenario: str,
) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["scenario_name"]].append(row)
    baseline_rows = grouped.get(baseline_scenario, [])
    baseline_leader = next((row for row in baseline_rows if row["rank"] == 1), None)
    baseline_name = baseline_leader["explainer_id"] if baseline_leader else None
    for row in rows:
        row["leader_changed"] = (
            row["rank"] == 1
            and row["scenario_name"] != baseline_scenario
            and baseline_name is not None
            and row["explainer_id"] != baseline_name
        )


def _definition_profile_summary(
    rows: list[dict[str, Any]],
    *,
    baseline_scenario: str,
) -> dict[str, Any]:
    if not rows:
        return {
            "baseline_leader": None,
            "leader_stability_count": 0,
            "scenario_count": 0,
        }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["scenario_name"]].append(row)
    baseline_rows = grouped.get(baseline_scenario, [])
    baseline_leader = next((row for row in baseline_rows if row["rank"] == 1), None)
    if baseline_leader is None:
        return {
            "baseline_leader": None,
            "leader_stability_count": 0,
            "scenario_count": max(0, len(grouped) - 1),
        }
    stability_count = 0
    scenario_count = 0
    for scenario_name, scenario_rows in grouped.items():
        if scenario_name == baseline_scenario:
            continue
        leader = next((row for row in scenario_rows if row["rank"] == 1), None)
        if leader is None:
            continue
        scenario_count += 1
        if leader["explainer_id"] == baseline_leader["explainer_id"]:
            stability_count += 1
    return {
        "baseline_leader": baseline_leader,
        "leader_stability_count": stability_count,
        "scenario_count": scenario_count,
    }


def _track_policy_shift(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in summary_rows:
        grouped[(row["dataset_id"], row["model_id"], row["explainer_id"])][row["track"]] = row

    shift_rows: list[dict[str, Any]] = []
    for dataset_id, model_id, explainer_id in sorted(grouped):
        items = grouped[(dataset_id, model_id, explainer_id)]
        primary = items.get("primary")
        audit = items.get("audit")
        if not primary or not audit:
            continue
        result = {
            "dataset_id": dataset_id,
            "model_id": model_id,
            "explainer_id": explainer_id,
            "primary_instances": primary["evaluated_rows"],
            "audit_instances": audit["evaluated_rows"],
        }
        for metric_key, metric_label in METRIC_ANALYSIS_FIELDS:
            primary_value = primary.get(metric_key)
            audit_value = audit.get(metric_key)
            delta_key = f"delta_{metric_label}"
            abs_delta_key = f"abs_delta_{metric_label}"
            if primary_value is None or audit_value is None:
                result[delta_key] = None
                result[abs_delta_key] = None
                continue
            delta = float(audit_value) - float(primary_value)
            result[delta_key] = delta
            result[abs_delta_key] = abs(delta)
        shift_rows.append(result)
    return shift_rows


def _track_policy_by_explainer(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["explainer_id"]].append(row)

    summary_rows: list[dict[str, Any]] = []
    for explainer_id in sorted(grouped):
        items = grouped[explainer_id]
        result = {
            "explainer_id": explainer_id,
            "pair_count": len(items),
        }
        metric_abs_means: list[float] = []
        for _metric_key, metric_label in METRIC_ANALYSIS_FIELDS:
            mean_delta = _mean_non_null(items, f"delta_{metric_label}")
            mean_abs_delta = _mean_non_null(items, f"abs_delta_{metric_label}")
            result[f"mean_delta_{metric_label}"] = mean_delta
            result[f"mean_abs_delta_{metric_label}"] = mean_abs_delta
            if mean_abs_delta is not None:
                metric_abs_means.append(mean_abs_delta)
        result["overall_mean_abs_delta"] = (
            sum(metric_abs_means) / len(metric_abs_means) if metric_abs_means else None
        )
        summary_rows.append(result)
    return sorted(
        summary_rows,
        key=lambda row: (
            row["overall_mean_abs_delta"] is None,
            row["overall_mean_abs_delta"] if row["overall_mean_abs_delta"] is not None else 0.0,
            row["explainer_id"],
        ),
    )


def _track_policy_metric_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    for _metric_key, metric_label in METRIC_ANALYSIS_FIELDS:
        summary_rows.append(
            {
                "metric": metric_label,
                "mean_delta": _mean_non_null(rows, f"delta_{metric_label}"),
                "mean_abs_delta": _mean_non_null(rows, f"abs_delta_{metric_label}"),
                "pair_count": sum(
                    1
                    for row in rows
                    if row.get(f"abs_delta_{metric_label}") is not None
                ),
            }
        )
    return sorted(
        summary_rows,
        key=lambda row: (
            row["mean_abs_delta"] is None,
            -(row["mean_abs_delta"] or 0.0),
            row["metric"],
        ),
    )


def _track_policy_leader_stability(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        grouped[(row["dataset_id"], row["model_id"])].append(row)

    leader_rows: list[dict[str, Any]] = []
    for dataset_id, model_id in sorted(grouped):
        items = grouped[(dataset_id, model_id)]
        primary_rows = [row for row in items if row["track"] == "primary"]
        audit_rows = [row for row in items if row["track"] == "audit"]
        for metric_key, metric_label in METRIC_ANALYSIS_FIELDS:
            primary_leader = _leader_for_metric(primary_rows, metric_key)
            audit_leader = _leader_for_metric(audit_rows, metric_key)
            if primary_leader is None or audit_leader is None:
                continue
            leader_rows.append(
                {
                    "dataset_id": dataset_id,
                    "model_id": model_id,
                    "metric": metric_label,
                    "primary_leader": primary_leader["explainer_id"],
                    "audit_leader": audit_leader["explainer_id"],
                    "primary_score": primary_leader[metric_key],
                    "audit_score": audit_leader[metric_key],
                    "same_leader": primary_leader["explainer_id"] == audit_leader["explainer_id"],
                }
            )
    return leader_rows


def _track_policy_leader_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    for _metric_key, metric_label in METRIC_ANALYSIS_FIELDS:
        items = [row for row in rows if row["metric"] == metric_label]
        if not items:
            continue
        same_count = sum(1 for row in items if row["same_leader"])
        summary_rows.append(
            {
                "metric": metric_label,
                "same_leader_count": same_count,
                "comparison_count": len(items),
                "flip_count": len(items) - same_count,
                "same_leader_rate": same_count / len(items),
            }
        )
    return summary_rows


def _track_policy_summary(
    *,
    track_policy_by_explainer: list[dict[str, Any]],
    track_policy_metric_summary: list[dict[str, Any]],
    track_policy_leader_summary: list[dict[str, Any]],
) -> dict[str, Any]:
    explainer_rows = [
        row
        for row in track_policy_by_explainer
        if row.get("overall_mean_abs_delta") is not None
    ]
    metric_rows = [
        row for row in track_policy_metric_summary if row.get("mean_abs_delta") is not None
    ]
    leader_rows = [
        row for row in track_policy_leader_summary if row.get("same_leader_rate") is not None
    ]
    return {
        "most_robust_explainer": (
            min(explainer_rows, key=lambda row: row["overall_mean_abs_delta"])
            if explainer_rows
            else None
        ),
        "most_sensitive_explainer": (
            max(explainer_rows, key=lambda row: row["overall_mean_abs_delta"])
            if explainer_rows
            else None
        ),
        "largest_metric_shift": (
            max(metric_rows, key=lambda row: row["mean_abs_delta"])
            if metric_rows
            else None
        ),
        "smallest_metric_shift": (
            min(metric_rows, key=lambda row: row["mean_abs_delta"])
            if metric_rows
            else None
        ),
        "most_stable_metric": (
            max(leader_rows, key=lambda row: row["same_leader_rate"])
            if leader_rows
            else None
        ),
        "least_stable_metric": (
            min(leader_rows, key=lambda row: row["same_leader_rate"])
            if leader_rows
            else None
        ),
    }


def _leave_one_dataset_out(
    summary_rows: list[dict[str, Any]],
    random_seed: int,
    bootstrap_config: dict[str, Any],
) -> list[dict[str, Any]]:
    dataset_ids = sorted({row["dataset_id"] for row in summary_rows})
    rows: list[dict[str, Any]] = []
    for index, omitted_dataset_id in enumerate(dataset_ids):
        filtered = [
            row for row in summary_rows if row["dataset_id"] != omitted_dataset_id
        ]
        aggregated = _aggregate_summary_rows(
            filtered,
            ["explainer_id"],
            random_seed=random_seed + (index + 1) * 2713,
            bootstrap_config=bootstrap_config,
        )
        for row in aggregated:
            rows.append(
                {
                    "omitted_dataset_id": omitted_dataset_id,
                    **row,
                }
            )
    return rows


def _leave_one_dataset_out_leaders(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    leaders: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["omitted_dataset_id"]].append(row)
    for omitted_dataset_id in sorted(grouped):
        items = grouped[omitted_dataset_id]
        for metric_key, metric_label in METRIC_ANALYSIS_FIELDS:
            eligible = [row for row in items if row.get(metric_key) is not None]
            if not eligible:
                continue
            leader = max(eligible, key=lambda row: row[metric_key])
            leaders.append(
                {
                    "omitted_dataset_id": omitted_dataset_id,
                    "metric": metric_label,
                    "explainer_id": leader["explainer_id"],
                    "score": leader[metric_key],
                }
            )
    return leaders


def _normalize_metric_values(
    rows: list[dict[str, Any]],
    metrics: list[str],
) -> dict[str, dict[str, float]]:
    normalized: dict[str, dict[str, float]] = {}
    for metric in metrics:
        values = [
            float(row[metric])
            for row in rows
            if row.get(metric) is not None
        ]
        if not values:
            normalized[metric] = {
                row["explainer_id"]: 0.0 for row in rows
            }
            continue
        low = min(values)
        high = max(values)
        if high == low:
            normalized[metric] = {
                row["explainer_id"]: 1.0 for row in rows if row.get(metric) is not None
            }
            continue
        normalized[metric] = {
            row["explainer_id"]: (float(row[metric]) - low) / (high - low)
            for row in rows
            if row.get(metric) is not None
        }
    return normalized


def _random_weight_vector(metrics: list[str], rng: random.Random) -> dict[str, float]:
    raw = {metric: rng.expovariate(1.0) for metric in metrics}
    total = sum(raw.values())
    return {metric: value / total for metric, value in raw.items()}


def _metric_rank_correlation_summary(
    correlations: list[dict[str, Any]],
) -> dict[str, Any]:
    correlations = [
        row
        for row in correlations
        if row["spearman_rank_corr"] is not None and row["shared_explainers"] >= 3
    ]
    if not correlations:
        return {"strongest_positive": None, "weakest": None}
    strongest_positive = max(correlations, key=lambda row: row["spearman_rank_corr"])
    weakest = min(correlations, key=lambda row: row["spearman_rank_corr"])
    return {
        "strongest_positive": strongest_positive,
        "weakest": weakest,
    }


def _group_metric_correlation_summary(
    correlations: list[dict[str, Any]],
) -> dict[str, Any]:
    correlations = [
        row
        for row in correlations
        if row["spearman_rank_corr"] is not None and row["shared_groups"] >= 3
    ]
    if not correlations:
        return {"strongest_positive": None, "weakest": None}
    strongest_positive = max(correlations, key=lambda row: row["spearman_rank_corr"])
    weakest = min(correlations, key=lambda row: row["spearman_rank_corr"])
    return {
        "strongest_positive": strongest_positive,
        "weakest": weakest,
    }


def _spearman_rank_correlation(
    rows: list[dict[str, Any]],
    metric_a: str,
    metric_b: str,
) -> float | None:
    if len(rows) < 2:
        return None
    ranks_a = _metric_ranks(rows, metric_a)
    ranks_b = _metric_ranks(rows, metric_b)
    values_a = [ranks_a[row["explainer_id"]] for row in rows]
    values_b = [ranks_b[row["explainer_id"]] for row in rows]
    return _pearson_correlation(values_a, values_b)


def _spearman_correlation_from_values(
    values_a: list[float],
    values_b: list[float],
) -> float | None:
    if len(values_a) != len(values_b) or len(values_a) < 2:
        return None
    ranks_a = _rank_values(values_a)
    ranks_b = _rank_values(values_b)
    return _pearson_correlation(ranks_a, ranks_b)


def _rank_values(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1], reverse=True)
    ranks = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        start = index
        score = ordered[index][1]
        while index < len(ordered) and ordered[index][1] == score:
            index += 1
        average_rank = (start + 1 + index) / 2.0
        for original_index, _value in ordered[start:index]:
            ranks[original_index] = average_rank
    return ranks


def _metric_ranks(rows: list[dict[str, Any]], metric: str) -> dict[str, float]:
    ordered = sorted(
        ((row["explainer_id"], float(row[metric])) for row in rows if row.get(metric) is not None),
        key=lambda item: item[1],
        reverse=True,
    )
    ranks: dict[str, float] = {}
    index = 0
    while index < len(ordered):
        start = index
        score = ordered[index][1]
        while index < len(ordered) and ordered[index][1] == score:
            index += 1
        average_rank = (start + 1 + index) / 2.0
        for name, _value in ordered[start:index]:
            ranks[name] = average_rank
    return ranks


def _pearson_correlation(values_a: list[float], values_b: list[float]) -> float | None:
    if len(values_a) != len(values_b) or len(values_a) < 2:
        return None
    mean_a = sum(values_a) / len(values_a)
    mean_b = sum(values_b) / len(values_b)
    centered_a = [value - mean_a for value in values_a]
    centered_b = [value - mean_b for value in values_b]
    numerator = sum(a * b for a, b in zip(centered_a, centered_b, strict=False))
    denom_a = sum(a * a for a in centered_a) ** 0.5
    denom_b = sum(b * b for b in centered_b) ** 0.5
    if denom_a == 0.0 or denom_b == 0.0:
        return None
    return numerator / (denom_a * denom_b)


def _bootstrap_binary_rate_interval(
    rows: list[dict[str, Any]],
    *,
    outcome_key: str,
    bootstrap_config: dict[str, Any],
    seed: int,
) -> tuple[float | None, float | None]:
    if not bootstrap_config.get("enabled", True) or len(rows) < 2:
        return None, None

    rng = random.Random(seed)
    resample_count = int(bootstrap_config["num_resamples"])
    confidence_level = float(bootstrap_config["confidence_level"])
    alpha = 1.0 - confidence_level
    lower_q = alpha / 2.0
    upper_q = 1.0 - lower_q

    sampled_rates: list[float] = []
    for _ in range(resample_count):
        sampled_rows = _bootstrap_sample_rows(
            rows,
            bootstrap_config=bootstrap_config,
            group_keys=[],
            rng=rng,
        )
        sampled_rates.append(_mean_of_rows(sampled_rows, outcome_key))
    sampled_rates.sort()
    return _percentile(sampled_rates, lower_q), _percentile(sampled_rates, upper_q)


def _bootstrap_sample_rows(
    rows: list[dict[str, Any]],
    *,
    bootstrap_config: dict[str, Any],
    group_keys: list[str],
    rng: random.Random,
) -> list[dict[str, Any]]:
    method = bootstrap_config.get("method", "hierarchical")
    if method in {"group", "group_bootstrap"}:
        return [rows[rng.randrange(len(rows))] for _ in range(len(rows))]
    hierarchy_levels = [
        level
        for level in bootstrap_config.get("hierarchy_levels", DEFAULT_BOOTSTRAP_HIERARCHY)
        if level not in group_keys
    ]
    return _hierarchical_resample_rows(rows, hierarchy_levels=hierarchy_levels, rng=rng)


def _hierarchical_resample_rows(
    rows: list[dict[str, Any]],
    *,
    hierarchy_levels: list[str],
    rng: random.Random,
) -> list[dict[str, Any]]:
    active_levels = _active_hierarchy_levels(rows, hierarchy_levels)
    if not active_levels:
        return [rows[rng.randrange(len(rows))] for _ in range(len(rows))]
    return _hierarchical_resample_recursive(rows, hierarchy_levels=active_levels, rng=rng)


def _hierarchical_resample_recursive(
    rows: list[dict[str, Any]],
    *,
    hierarchy_levels: list[str],
    rng: random.Random,
) -> list[dict[str, Any]]:
    if not hierarchy_levels:
        return [rows[rng.randrange(len(rows))] for _ in range(len(rows))]

    level = hierarchy_levels[0]
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row[level]].append(row)

    keys = sorted(grouped)
    sampled_rows: list[dict[str, Any]] = []
    for _ in range(len(keys)):
        sampled_key = keys[rng.randrange(len(keys))]
        cluster_rows = grouped[sampled_key]
        if len(hierarchy_levels) == 1:
            sampled_rows.extend(cluster_rows)
        else:
            sampled_rows.extend(
                _hierarchical_resample_recursive(
                    cluster_rows,
                    hierarchy_levels=hierarchy_levels[1:],
                    rng=rng,
                )
            )
    return sampled_rows


def _active_hierarchy_levels(
    rows: list[dict[str, Any]],
    hierarchy_levels: list[str],
) -> list[str]:
    active: list[str] = []
    for level in hierarchy_levels:
        values = {row.get(level) for row in rows}
        if len(values) > 1:
            active.append(level)
    return active


def _mean_of_rows(rows: list[dict[str, Any]], value_key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(row[value_key]) for row in rows) / len(rows)


def _pairwise_totals(pairwise_rows: list[dict[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = defaultdict(int)
    for row in pairwise_rows:
        totals[row["explainer_a"]] += row["wins_a"]
        totals[row["explainer_b"]] += row["wins_b"]
    return dict(totals)


def _dataset_track_winners(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset_id"], row["track"])].append(row)

    winners: list[dict[str, Any]] = []
    for dataset_id, track in sorted(grouped):
        items = grouped[(dataset_id, track)]
        best_faithfulness = _best_row(items, "mean_faithfulness")
        best_stability = _best_row(items, "mean_stability")
        best_comprehensibility = _best_row(items, "mean_comprehensibility")
        best_css = _best_row(items, "mean_css")
        best_actionability = _best_row(items, "mean_actionability")
        best_alf = _best_row(items, "mean_alf")
        winners.append(
            {
                "dataset_id": dataset_id,
                "track": track,
                "best_faithfulness_explainer": (
                    best_faithfulness["explainer_id"] if best_faithfulness else "n/a"
                ),
                "best_faithfulness_score": (
                    best_faithfulness["mean_faithfulness"] if best_faithfulness else None
                ),
                "best_stability_explainer": (
                    best_stability["explainer_id"] if best_stability else "n/a"
                ),
                "best_stability_score": (
                    best_stability["mean_stability"] if best_stability else None
                ),
                "best_comprehensibility_explainer": (
                    best_comprehensibility["explainer_id"]
                    if best_comprehensibility
                    else "n/a"
                ),
                "best_comprehensibility_score": (
                    best_comprehensibility["mean_comprehensibility"]
                    if best_comprehensibility
                    else None
                ),
                "best_css_explainer": best_css["explainer_id"] if best_css else "n/a",
                "best_css_score": best_css["mean_css"] if best_css else None,
                "best_actionability_explainer": (
                    best_actionability["explainer_id"] if best_actionability else "n/a"
                ),
                "best_actionability_score": (
                    best_actionability["mean_actionability"] if best_actionability else None
                ),
                "best_alf_explainer": best_alf["explainer_id"] if best_alf else "n/a",
                "best_alf_score": best_alf["mean_alf"] if best_alf else None,
            }
        )
    return winners


def _best_row(rows: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    eligible = [row for row in rows if row.get(key) is not None]
    if not eligible:
        return None
    return max(eligible, key=lambda row: row[key])


def _normalize_bootstrap_config(config: dict[str, Any] | None) -> dict[str, Any]:
    configured = dict(config or {})
    confidence_level = float(configured.get("confidence_level", 0.95))
    if confidence_level <= 0.0 or confidence_level >= 1.0:
        confidence_level = 0.95
    method = str(configured.get("method", "hierarchical")).strip().lower()
    if method not in {"group", "group_bootstrap", "hierarchical", "hierarchical_bootstrap"}:
        method = "hierarchical"
    hierarchy_levels = [
        str(level)
        for level in configured.get("hierarchy_levels", DEFAULT_BOOTSTRAP_HIERARCHY)
        if str(level).strip()
    ]
    if not hierarchy_levels:
        hierarchy_levels = list(DEFAULT_BOOTSTRAP_HIERARCHY)
    return {
        "enabled": bool(configured.get("enabled", True)),
        "num_resamples": max(1, int(configured.get("num_resamples", 1000))),
        "confidence_level": confidence_level,
        "method": method,
        "hierarchy_levels": hierarchy_levels,
        "unit": (
            "hierarchical_bootstrap"
            if method in {"hierarchical", "hierarchical_bootstrap"}
            else "group_bootstrap"
        ),
    }


def _bootstrap_summary_intervals(
    rows: list[dict[str, Any]],
    *,
    metrics: list[str],
    bootstrap_config: dict[str, Any],
    group_keys: list[str],
    seed: int,
) -> dict[str, float | None]:
    interval_fields: dict[str, float | None] = {}
    for metric in metrics:
        interval_fields[f"{metric}_ci_low"] = None
        interval_fields[f"{metric}_ci_high"] = None

    if not bootstrap_config.get("enabled", True) or len(rows) < 2:
        return interval_fields

    rng = random.Random(seed)
    resample_count = int(bootstrap_config["num_resamples"])
    confidence_level = float(bootstrap_config["confidence_level"])
    alpha = 1.0 - confidence_level
    lower_q = alpha / 2.0
    upper_q = 1.0 - lower_q

    metric_samples: dict[str, list[float]] = {metric: [] for metric in metrics}
    for _ in range(resample_count):
        sampled_rows = _bootstrap_sample_rows(
            rows,
            bootstrap_config=bootstrap_config,
            group_keys=group_keys,
            rng=rng,
        )
        for metric in metrics:
            value = _weighted_mean(sampled_rows, metric)
            if value is not None:
                metric_samples[metric].append(value)

    for metric in metrics:
        values = metric_samples[metric]
        if not values:
            continue
        values.sort()
        interval_fields[f"{metric}_ci_low"] = _percentile(values, lower_q)
        interval_fields[f"{metric}_ci_high"] = _percentile(values, upper_q)

    return interval_fields


def _percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if len(values) == 1:
        return values[0]
    position = q * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _fmt_ci(row: dict[str, Any], metric: str) -> str:
    low = row.get(f"{metric}_ci_low")
    high = row.get(f"{metric}_ci_high")
    if low is None or high is None:
        return "n/a"
    return f"[{_fmt(low)}, {_fmt(high)}]"


def _bootstrap_note_line(analysis: dict[str, Any]) -> str:
    bootstrap = analysis.get("bootstrap", {})
    if not bootstrap.get("enabled", False):
        return "- Bootstrap uncertainty is disabled for this run."
    level = int(round(float(bootstrap["confidence_level"]) * 100))
    method = bootstrap.get("method", "group_bootstrap")
    if method in {"hierarchical", "hierarchical_bootstrap"}:
        levels = " -> ".join(bootstrap.get("hierarchy_levels", DEFAULT_BOOTSTRAP_HIERARCHY))
        return (
            f"- `{level}%` hierarchical-bootstrap confidence intervals with "
            f"`{bootstrap['num_resamples']}` resamples over `{levels}` benchmark clusters."
        )
    return (
        f"- `{level}%` group-bootstrap confidence intervals with "
        f"`{bootstrap['num_resamples']}` resamples over matched benchmark configurations."
    )


def _tex_escape(value: str) -> str:
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("&", "\\&")
        .replace("%", "\\%")
    )
