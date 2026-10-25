import dataclasses
import hashlib
import heapq
import math
import os
import random
import re
import zlib
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from .adapters import adapt_raw_example
from .capabilities import detect_capabilities
from .corrections import (
    SAMPLING_SELECTIONS,
    V1,
    V2,
    CorrectionOptions,
    corrections_from_config,
)
from .datasets.registry import build_dataset_loader
from .explainers import generate_raw_explanation, select_explainer_backend
from .metrics.scorer import ExplanationScorer
from .models import fit_model
from .normalization import ExplanationCanonicalizer
from .paths import repo_relative_path
from .progress import emit_progress
from .stability import compute_stability_evidence
from .types import TabularDataset


def run_demo_benchmark(config) -> dict[str, Any]:
    # Versioned corrections (corrections.py). V1, the published behavior, unless the
    # config's cslaw block selects a preset ("cslaw": {"corrections": "v2"}).
    corrections = corrections_from_config(config)
    scorer = ExplanationScorer(corrections=corrections)
    summary_rows: list[dict[str, Any]] = []
    sampling_cfg = config.denied_instance_sampling
    evaluation_cfg = config.evaluation_protocol
    threshold_policy = config.model_threshold_policy
    # Opt-in CS&Law instrumentation. None when the config has no "cslaw" block,
    # in which case every hook below is skipped and the run matches the baseline.
    cslaw = _cslaw_settings(config)
    cslaw_groups: dict[str, dict[str, Any]] = {}
    # SHA-256 of each dataset file, filled while loading (record_data_provenance).
    provenance_datasets: dict[str, dict[str, Any]] | None = (
        {} if corrections.record_data_provenance else None
    )
    if _records_corrections(corrections):
        emit_progress(
            "cslaw_corrections",
            corrections_version=_corrections_version(corrections),
            corrections=corrections.to_dict(),
        )

    tracks: list[str] = []
    if config.payload["tracks"]["run_primary_track"]:
        tracks.append("primary")
    if config.payload["tracks"]["run_audit_track"]:
        tracks.append("audit")

    for dataset_id in config.enabled_datasets:
        if cslaw is not None and cslaw.shared_sampling_enabled:
            _run_cslaw_shared_dataset(
                config=config,
                dataset_id=dataset_id,
                tracks=tracks,
                scorer=scorer,
                sampling_cfg=sampling_cfg,
                evaluation_cfg=evaluation_cfg,
                threshold_policy=threshold_policy,
                cslaw=cslaw,
                cslaw_groups=cslaw_groups,
                summary_rows=summary_rows,
                provenance_datasets=provenance_datasets,
            )
            continue
        dataset_start = perf_counter()
        emit_progress("dataset_start", dataset_id=dataset_id)
        loader = build_dataset_loader(config, dataset_id)
        for track in tracks:
            track_start = perf_counter()
            emit_progress("track_start", dataset_id=dataset_id, track=track)
            bundle_start = perf_counter()
            bundle = loader.build_bundle(track)
            emit_progress(
                "track_bundle_ready",
                dataset_id=dataset_id,
                track=track,
                elapsed_seconds=round(perf_counter() - bundle_start, 3),
                included_feature_count=len(bundle.included_feature_names),
                excluded_feature_count=len(bundle.excluded_feature_names),
            )
            table_load_start = perf_counter()
            table = loader.load_tabular_dataset(track=track)
            emit_progress(
                "track_table_ready",
                dataset_id=dataset_id,
                track=track,
                elapsed_seconds=round(perf_counter() - table_load_start, 3),
                total_rows=len(table.rows),
                feature_count=len(table.feature_names),
                source_kind=table.source_kind,
            )
            if provenance_datasets is not None:
                _record_dataset_provenance(provenance_datasets, dataset_id, table)
            split_start = perf_counter()
            train_table, eval_table, split_summary = _split_table_for_evaluation(
                table=table,
                negative_label=bundle.negative_label,
                evaluation_cfg=evaluation_cfg,
            )
            emit_progress(
                "track_split_ready",
                dataset_id=dataset_id,
                track=track,
                elapsed_seconds=round(perf_counter() - split_start, 3),
                total_rows=len(table.rows),
                train_rows=len(train_table.rows),
                eval_rows=len(eval_table.rows),
                split_strategy=split_summary.get("strategy"),
                heldout_used=split_summary.get("heldout_used"),
            )
            canonicalizer = ExplanationCanonicalizer(bundle, corrections=corrections)
            for model_id in config.enabled_models:
                model_start = perf_counter()
                emit_progress(
                    "model_start",
                    dataset_id=dataset_id,
                    track=track,
                    model_id=model_id,
                    total_rows=len(table.rows),
                    train_rows=len(train_table.rows),
                    eval_rows=len(eval_table.rows),
                )
                model, model_backend = fit_model(
                    model_id=model_id,
                    dataset_bundle=bundle,
                    table=train_table,
                    threshold=config.model_threshold,
                )
                if cslaw is not None:
                    _cslaw_apply_model_options(model, cslaw)
                _apply_model_corrections(model, corrections)
                predictions, threshold_summary = _predict_with_threshold_policy(
                    model=model,
                    rows=eval_table.rows,
                    instance_id_prefix=f"{dataset_id}-{track}-{model_id}",
                    threshold_policy=threshold_policy,
                )

                model_performance = _evaluate_model_predictions(
                    rows=eval_table.rows,
                    predictions=[prediction for _row, prediction in predictions],
                    target=bundle.target,
                    positive_label=bundle.positive_label,
                    negative_label=bundle.negative_label,
                )

                evaluation_rows = [
                    (row, prediction)
                    for row, prediction in predictions
                    if prediction.label == bundle.negative_label
                ]
                if not evaluation_rows and predictions:
                    fallback = max(predictions, key=lambda item: item[1].score)
                    evaluation_rows = [fallback]
                sampled_rows, sampling_summary = _sample_evaluation_rows(
                    evaluation_rows=evaluation_rows,
                    sampling_cfg=sampling_cfg,
                    selection=corrections.sampling_selection,
                    seed_key=f"{dataset_id}|{track}|{model_id}",
                    seed=corrections.sampling_seed,
                )
                emit_progress(
                    "model_ready",
                    dataset_id=dataset_id,
                    track=track,
                    model_id=model_id,
                    adverse_rows=len(evaluation_rows),
                    sampled_rows=len(sampled_rows),
                    adverse_rate=(
                        sum(
                            1
                            for _row, prediction in predictions
                            if prediction.label == bundle.negative_label
                        )
                        / max(1, len(eval_table.rows))
                    ),
                    elapsed_seconds=round(perf_counter() - model_start, 3),
                    model_backend=model_backend.implementation,
                )
                cslaw_native: dict[str, dict[str, Any]] = {}
                if cslaw is not None and cslaw.active:
                    cslaw_native = _cslaw_prepare_model_group(
                        cslaw=cslaw,
                        cslaw_groups=cslaw_groups,
                        dataset_id=dataset_id,
                        track=track,
                        model_id=model_id,
                        model=model,
                        model_backend=model_backend,
                        negative_label=bundle.negative_label,
                        predictions=predictions,
                        sampled_rows=sampled_rows,
                        shared_sampling=None,
                    )

                for explainer_id in config.enabled_explainers:
                    explainer_start = perf_counter()
                    emit_progress(
                        "explainer_start",
                        dataset_id=dataset_id,
                        track=track,
                        model_id=model_id,
                        explainer_id=explainer_id,
                        sampled_rows=len(sampled_rows),
                    )
                    instance_results = []
                    for row, prediction in sampled_rows:
                        raw_payload, explainer_backend = generate_raw_explanation(
                            explainer_id=explainer_id,
                            dataset_bundle=bundle,
                            model=model,
                            row=row,
                            prediction=prediction,
                        )
                        adapted_payload = adapt_raw_example(
                            dataset_bundle=bundle,
                            explainer_id=explainer_id,
                            raw_payload=raw_payload,
                        )
                        stability_evidence = compute_stability_evidence(
                            probe_cfg=config.stability_probe,
                            dataset_bundle=bundle,
                            explainer_id=explainer_id,
                            model=model,
                            row=row,
                            prediction=prediction,
                            adapted_payload=adapted_payload,
                        )
                        if stability_evidence is not None:
                            adapted_payload["metadata"] = {
                                **adapted_payload.get("metadata", {}),
                                "stability_evidence": stability_evidence,
                            }
                        canonical = canonicalizer.canonicalize(adapted_payload)
                        scores = scorer.score(canonical)
                        instance_results.append(
                            {
                                "instance_id": prediction.instance_id,
                                "prediction_score": prediction.score,
                                "predicted_label": prediction.label,
                                "model_backend": {
                                    "backend": model_backend.backend,
                                    "implementation": model_backend.implementation,
                                    "reason": model_backend.reason,
                                },
                                "explainer_backend": {
                                    "backend": explainer_backend.backend,
                                    "implementation": explainer_backend.implementation,
                                    "reason": explainer_backend.reason,
                                },
                                "subgroup_attributes": _subgroup_attributes(
                                    dataset_id=dataset_id,
                                    row=row,
                                ),
                                "scores": scores,
                            }
                        )
                        if cslaw is not None and cslaw.active:
                            _cslaw_attach_instance_fields(
                                instance_results[-1],
                                cslaw=cslaw,
                                canonical=canonical,
                                prediction=prediction,
                                native_records=cslaw_native,
                            )

                    summary_rows.append(
                        _aggregate_group(
                            dataset_id=dataset_id,
                            track=track,
                            model_id=model_id,
                            explainer_id=explainer_id,
                            data_source_kind=eval_table.source_kind,
                            data_source_path=eval_table.source_path,
                            total_rows=len(table.rows),
                            train_rows=len(train_table.rows),
                            test_rows=len(eval_table.rows),
                            adverse_rows=len(evaluation_rows),
                            evaluated_rows=len(instance_results),
                            model_backend=model_backend,
                            explainer_backend=explainer_backend,
                            adverse_rate=sum(
                                1
                                for _row, prediction in predictions
                                if prediction.label == bundle.negative_label
                            )
                            / max(1, len(eval_table.rows)),
                            split_summary=split_summary,
                            threshold_summary=threshold_summary,
                            model_performance=model_performance,
                            sampling_summary=sampling_summary,
                            instance_results=instance_results,
                        )
                    )
                    emit_progress(
                        "explainer_done",
                        dataset_id=dataset_id,
                        track=track,
                        model_id=model_id,
                        explainer_id=explainer_id,
                        evaluated_rows=len(instance_results),
                        elapsed_seconds=round(perf_counter() - explainer_start, 3),
                        explainer_backend=explainer_backend.implementation,
                    )
                emit_progress(
                    "model_done",
                    dataset_id=dataset_id,
                    track=track,
                    model_id=model_id,
                    elapsed_seconds=round(perf_counter() - model_start, 3),
                )
            emit_progress(
                "track_done",
                dataset_id=dataset_id,
                track=track,
                elapsed_seconds=round(perf_counter() - track_start, 3),
            )
        emit_progress(
            "dataset_done",
            dataset_id=dataset_id,
            elapsed_seconds=round(perf_counter() - dataset_start, 3),
        )

    payload = {
        "experiment_name": config.payload["experiment"]["name"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "capabilities": detect_capabilities(),
        "overview": _build_overview(summary_rows),
        "summary": summary_rows,
    }
    if cslaw is not None and cslaw.active:
        cslaw_block: dict[str, Any] = {
            "config": cslaw.to_dict(),
            "groups": cslaw_groups,
        }
        # Non-v1 corrections imply a cslaw block (corrections_from_config), so the
        # default path never gains these keys. Pre-v2 configs that only set the
        # legacy native-ranking flag are not marked (see _records_corrections).
        if _records_corrections(corrections):
            cslaw_block["corrections_version"] = _corrections_version(corrections)
            cslaw_block["corrections"] = corrections.to_dict()
        if provenance_datasets is not None:
            cslaw_block["provenance"] = _data_provenance(config, provenance_datasets)
        payload["cslaw"] = _strict_json(cslaw_block)
    return payload


def _aggregate_group(
    dataset_id: str,
    track: str,
    model_id: str,
    explainer_id: str,
    data_source_kind: str,
    data_source_path: str,
    total_rows: int,
    train_rows: int,
    test_rows: int,
    adverse_rows: int,
    evaluated_rows: int,
    model_backend,
    explainer_backend,
    adverse_rate: float,
    split_summary: dict[str, Any],
    threshold_summary: dict[str, Any],
    model_performance: dict[str, Any],
    sampling_summary: dict[str, Any],
    instance_results: list[dict[str, Any]],
) -> dict[str, Any]:
    css_scores = [
        item["scores"]["contestability_support_score"]["score"] for item in instance_results
    ]
    css_verifiability = [
        item["scores"]["contestability_support_score"]["verifiability"]
        for item in instance_results
    ]
    css_correctability = [
        item["scores"]["contestability_support_score"]["correctability"]
        for item in instance_results
    ]
    css_decision_sensitivity_visibility = [
        item["scores"]["contestability_support_score"]["decision_sensitivity_visibility"]
        for item in instance_results
    ]
    css_reason_specificity = [
        item["scores"]["contestability_support_score"]["reason_specificity"]
        for item in instance_results
    ]
    faithfulness_scores = [
        item["scores"]["faithfulness_score"]["score"] for item in instance_results
    ]
    stability_scores = [
        item["scores"]["stability_score"]["score"] for item in instance_results
    ]
    comprehensibility_scores = [
        item["scores"]["comprehensibility_score"]["score"] for item in instance_results
    ]
    actionability_scores = [
        item["scores"]["unified_actionability_score"]["score"] for item in instance_results
    ]
    alf_scores = [
        item["scores"]["actionability_legality_filter"]["score"]
        for item in instance_results
        if item["scores"]["actionability_legality_filter"]["score"] is not None
    ]
    alf_feasibility = [
        item["scores"]["actionability_legality_filter"]["feasibility"]
        for item in instance_results
        if item["scores"]["actionability_legality_filter"]["feasibility"] is not None
    ]
    alf_legality = [
        item["scores"]["actionability_legality_filter"]["legality"]
        for item in instance_results
        if item["scores"]["actionability_legality_filter"]["legality"] is not None
    ]
    alf_minimality = [
        item["scores"]["actionability_legality_filter"]["minimality"]
        for item in instance_results
        if item["scores"]["actionability_legality_filter"]["minimality"] is not None
    ]
    invalid_rates = [
        item["scores"]["actionability_legality_filter"]["invalid_recourse_rate"]
        for item in instance_results
        if item["scores"]["actionability_legality_filter"]["invalid_recourse_rate"] is not None
    ]
    hard_failure_rates = [
        1.0 if item["scores"]["actionability_legality_filter"]["hard_failures"] else 0.0
        for item in instance_results
        if item["scores"]["actionability_legality_filter"]["applicability"] == "applicable"
    ]
    faithfulness_sources = _count_values(
        item["scores"]["faithfulness_score"]["evidence_source"] for item in instance_results
    )
    stability_sources = _count_values(
        item["scores"]["stability_score"]["evidence_source"] for item in instance_results
    )

    return {
        "dataset_id": dataset_id,
        "track": track,
        "model_id": model_id,
        "explainer_id": explainer_id,
        "data_source": {
            "kind": data_source_kind,
            "path": data_source_path,
        },
        "model_backend": {
            "backend": model_backend.backend,
            "implementation": model_backend.implementation,
            "reason": model_backend.reason,
        },
        "explainer_backend": {
            "backend": explainer_backend.backend,
            "implementation": explainer_backend.implementation,
            "reason": explainer_backend.reason,
        },
        "total_rows": total_rows,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "adverse_rows": adverse_rows,
        "evaluated_rows": evaluated_rows,
        "adverse_rate": adverse_rate,
        "evaluation_split": split_summary,
        "threshold_policy": threshold_summary,
        "model_performance": model_performance,
        "model_decision_threshold": model_performance.get("decision_threshold"),
        "model_accuracy": model_performance.get("accuracy"),
        "model_roc_auc": model_performance.get("roc_auc"),
        "model_average_precision": model_performance.get("average_precision"),
        "model_brier_score": model_performance.get("brier_score"),
        "model_expected_calibration_error": model_performance.get(
            "expected_calibration_error"
        ),
        "model_positive_rate": model_performance.get("positive_rate"),
        "model_predicted_positive_rate": model_performance.get(
            "predicted_positive_rate"
        ),
        "mean_faithfulness": _mean(faithfulness_scores),
        "mean_stability": _mean(stability_scores),
        "mean_comprehensibility": _mean(comprehensibility_scores),
        "mean_css": _mean(css_scores),
        "mean_css_verifiability": _mean(css_verifiability),
        "mean_css_correctability": _mean(css_correctability),
        "mean_css_decision_sensitivity_visibility": _mean(
            css_decision_sensitivity_visibility
        ),
        "mean_css_reason_specificity": _mean(css_reason_specificity),
        "mean_actionability": _mean(actionability_scores),
        "mean_alf": _mean(alf_scores),
        "mean_alf_feasibility": _mean(alf_feasibility),
        "mean_alf_legality": _mean(alf_legality),
        "mean_alf_minimality": _mean(alf_minimality),
        "mean_alf_hard_failure_rate": _mean(hard_failure_rates),
        "mean_invalid_recourse_rate": _mean(invalid_rates),
        "faithfulness_evidence_sources": faithfulness_sources,
        "stability_evidence_sources": stability_sources,
        "sampling": sampling_summary,
        "instance_ids": [item["instance_id"] for item in instance_results],
        "instance_results": instance_results,
    }


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _build_overview(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    explainers = sorted({row["explainer_id"] for row in summary_rows})
    return {
        "total_groups": len(summary_rows),
        "datasets": sorted({row["dataset_id"] for row in summary_rows}),
        "tracks": sorted({row["track"] for row in summary_rows}),
        "models": sorted({row["model_id"] for row in summary_rows}),
        "explainers": explainers,
        "by_explainer": [
            {
                "explainer_id": explainer_id,
                "mean_faithfulness": _mean(
                    [
                        row["mean_faithfulness"]
                        for row in summary_rows
                        if row["explainer_id"] == explainer_id
                        and row["mean_faithfulness"] is not None
                    ]
                ),
                "mean_stability": _mean(
                    [
                        row["mean_stability"]
                        for row in summary_rows
                        if row["explainer_id"] == explainer_id
                        and row["mean_stability"] is not None
                    ]
                ),
                "mean_comprehensibility": _mean(
                    [
                        row["mean_comprehensibility"]
                        for row in summary_rows
                        if row["explainer_id"] == explainer_id
                        and row["mean_comprehensibility"] is not None
                    ]
                ),
                "mean_css": _mean(
                    [
                        row["mean_css"]
                        for row in summary_rows
                        if row["explainer_id"] == explainer_id and row["mean_css"] is not None
                    ]
                ),
                "mean_actionability": _mean(
                    [
                        row["mean_actionability"]
                        for row in summary_rows
                        if row["explainer_id"] == explainer_id
                        and row["mean_actionability"] is not None
                    ]
                ),
                "mean_alf": _mean(
                    [
                        row["mean_alf"]
                        for row in summary_rows
                        if row["explainer_id"] == explainer_id and row["mean_alf"] is not None
                    ]
                ),
                "mean_invalid_recourse_rate": _mean(
                    [
                        row["mean_invalid_recourse_rate"]
                        for row in summary_rows
                        if row["explainer_id"] == explainer_id
                        and row["mean_invalid_recourse_rate"] is not None
                    ]
                ),
            }
            for explainer_id in explainers
        ],
        "best_faithfulness_group": _best_group(summary_rows, key="mean_faithfulness"),
        "best_stability_group": _best_group(summary_rows, key="mean_stability"),
        "best_comprehensibility_group": _best_group(
            summary_rows,
            key="mean_comprehensibility",
        ),
        "best_css_group": _best_group(summary_rows, key="mean_css"),
        "best_actionability_group": _best_group(summary_rows, key="mean_actionability"),
        "best_alf_group": _best_group(summary_rows, key="mean_alf"),
        "mean_evaluated_rows_per_group": _mean(
            [float(row["evaluated_rows"]) for row in summary_rows]
        ),
        "evaluation_split_usage": _count_values(
            row["evaluation_split"]["strategy"] for row in summary_rows
        ),
        "heldout_group_count": len(
            {
                (row["dataset_id"], row["track"], row["model_id"])
                for row in summary_rows
                if row["evaluation_split"].get("heldout_used", False)
            }
        ),
        "sampling_strategies": sorted(
            {row["sampling"]["strategy"] for row in summary_rows}
        ),
        "source_usage": _source_usage(summary_rows),
        "backend_usage": _backend_usage(summary_rows),
        "stability_evidence_usage": _stability_evidence_usage(summary_rows),
    }


def _best_group(summary_rows: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    eligible = [row for row in summary_rows if row[key] is not None]
    if not eligible:
        return None
    best = max(eligible, key=lambda row: row[key])
    return {
        "dataset_id": best["dataset_id"],
        "track": best["track"],
        "model_id": best["model_id"],
        "explainer_id": best["explainer_id"],
        "model_backend": best["model_backend"],
        "explainer_backend": best["explainer_backend"],
        key: best[key],
    }


def _backend_usage(summary_rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {"models": {}, "explainers": {}}
    for row in summary_rows:
        model_impl = row["model_backend"]["implementation"]
        explainer_impl = row["explainer_backend"]["implementation"]
        counts["models"][model_impl] = counts["models"].get(model_impl, 0) + 1
        counts["explainers"][explainer_impl] = counts["explainers"].get(explainer_impl, 0) + 1
    return counts


def _source_usage(summary_rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in summary_rows:
        source_kind = row["data_source"]["kind"]
        counts[source_kind] = counts.get(source_kind, 0) + 1
    return counts


def _stability_evidence_usage(summary_rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in summary_rows:
        for source_name, source_count in row.get("stability_evidence_sources", {}).items():
            counts[source_name] = counts.get(source_name, 0) + int(source_count)
    return counts


def _count_values(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


def _split_table_for_evaluation(
    *,
    table: TabularDataset,
    negative_label: str,
    evaluation_cfg: dict[str, Any],
) -> tuple[TabularDataset, TabularDataset, dict[str, Any]]:
    heldout_enabled = bool(evaluation_cfg.get("heldout_enabled", True))
    test_fraction = float(evaluation_cfg.get("test_fraction", 0.30))
    split_strategy = str(evaluation_cfg.get("split_strategy", "stratified"))
    split_seed = int(evaluation_cfg.get("split_seed", 0))
    fallback_to_full_dataset = bool(
        evaluation_cfg.get("fallback_to_full_dataset", True)
    )

    base_summary = {
        "heldout_enabled": heldout_enabled,
        "requested_test_fraction": test_fraction,
        "split_strategy": split_strategy,
        "split_seed": split_seed,
    }
    if not heldout_enabled:
        return table, table, {
            **base_summary,
            "strategy": "full_dataset",
            "heldout_used": False,
            "train_rows": len(table.rows),
            "test_rows": len(table.rows),
            "realized_test_fraction": 1.0,
        }

    if len(table.rows) < 4:
        if not fallback_to_full_dataset:
            raise ValueError("held-out evaluation requires at least 4 rows")
        return table, table, {
            **base_summary,
            "strategy": "fallback_full_dataset",
            "heldout_used": False,
            "train_rows": len(table.rows),
            "test_rows": len(table.rows),
            "realized_test_fraction": 1.0,
            "fallback_reason": "insufficient_rows",
        }

    train_rows, test_rows = _stratified_row_split(
        rows=table.rows,
        target=table.target,
        test_fraction=test_fraction,
        split_seed=split_seed,
        split_strategy=split_strategy,
    )
    if not train_rows or not test_rows:
        if not fallback_to_full_dataset:
            raise ValueError("held-out evaluation split produced an empty partition")
        return table, table, {
            **base_summary,
            "strategy": "fallback_full_dataset",
            "heldout_used": False,
            "train_rows": len(table.rows),
            "test_rows": len(table.rows),
            "realized_test_fraction": 1.0,
            "fallback_reason": "empty_partition",
        }

    train_table = _clone_table_with_rows(table, train_rows)
    eval_table = _clone_table_with_rows(table, test_rows)
    return train_table, eval_table, {
        **base_summary,
        "strategy": f"heldout_{split_strategy}",
        "heldout_used": True,
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "realized_test_fraction": len(test_rows) / max(1, len(table.rows)),
        "train_adverse_rate": _label_rate(train_rows, table.target, negative_label),
        "test_adverse_rate": _label_rate(test_rows, table.target, negative_label),
    }


def _stratified_row_split(
    *,
    rows: list[dict[str, Any]],
    target: str,
    test_fraction: float,
    split_seed: int,
    split_strategy: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if split_strategy != "stratified":
        shuffled = list(rows)
        random.Random(split_seed).shuffle(shuffled)
        test_count = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * test_fraction))))
        return shuffled[test_count:], shuffled[:test_count]

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(target)), []).append(row)

    rng = random.Random(split_seed)
    test_rows: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []
    for label in sorted(grouped):
        label_rows = list(grouped[label])
        if len(label_rows) < 2:
            return [], []
        rng.shuffle(label_rows)
        test_count = int(round(len(label_rows) * test_fraction))
        test_count = max(1, min(len(label_rows) - 1, test_count))
        test_rows.extend(label_rows[:test_count])
        train_rows.extend(label_rows[test_count:])
    rng.shuffle(train_rows)
    rng.shuffle(test_rows)
    return train_rows, test_rows


def _clone_table_with_rows(table: TabularDataset, rows: list[dict[str, Any]]) -> TabularDataset:
    return TabularDataset(
        dataset_id=table.dataset_id,
        display_name=table.display_name,
        track=table.track,
        target=table.target,
        feature_names=table.feature_names,
        rows=list(rows),
        source_kind=table.source_kind,
        source_path=table.source_path,
    )


def _label_rate(rows: list[dict[str, Any]], target: str, negative_label: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if str(row.get(target)) == negative_label) / len(rows)


def _evaluate_model_predictions(
    *,
    rows: list[dict[str, Any]],
    predictions: list[Any],
    target: str,
    positive_label: str,
    negative_label: str,
) -> dict[str, Any]:
    total = min(len(rows), len(predictions))
    if total == 0:
        return {
            "total_rows": 0,
            "decision_threshold": None,
            "accuracy": None,
            "roc_auc": None,
            "average_precision": None,
            "brier_score": None,
            "expected_calibration_error": None,
            "positive_rate": None,
            "predicted_positive_rate": None,
        }

    y_true = [1 if str(rows[index].get(target)) == negative_label else 0 for index in range(total)]
    scores = [float(predictions[index].score) for index in range(total)]
    predicted_labels = [predictions[index].label for index in range(total)]
    accuracy = sum(
        1
        for index in range(total)
        if predicted_labels[index]
        == (negative_label if y_true[index] == 1 else positive_label)
    ) / total
    return {
        "total_rows": total,
        "decision_threshold": float(predictions[0].threshold)
        if predictions and predictions[0].threshold is not None
        else None,
        "accuracy": accuracy,
        "roc_auc": _roc_auc(y_true, scores),
        "average_precision": _average_precision(y_true, scores),
        "brier_score": _brier_score(y_true, scores),
        "expected_calibration_error": _expected_calibration_error(y_true, scores),
        "positive_rate": sum(y_true) / total,
        "predicted_positive_rate": sum(
            1 for label in predicted_labels if label == negative_label
        )
        / total,
    }


def _predict_with_threshold_policy(
    *,
    model: Any,
    rows: list[dict[str, Any]],
    instance_id_prefix: str,
    threshold_policy: dict[str, Any],
) -> tuple[list[tuple[dict[str, Any], Any]], dict[str, Any]]:
    provisional_predictions = _predict_rows(
        model=model,
        rows=rows,
        instance_id_prefix=instance_id_prefix,
    )
    mode = str(threshold_policy.get("mode", "fixed"))
    base_summary = {
        "mode": mode,
        "configured_threshold": getattr(model, "threshold", None),
    }
    if mode != "target_predicted_adverse_rate":
        return provisional_predictions, {
            **base_summary,
            "applied_threshold": getattr(model, "threshold", None),
            "target_predicted_adverse_rate": None,
            "achieved_predicted_adverse_rate": _predicted_adverse_rate(
                provisional_predictions
            ),
        }

    target_rate = float(threshold_policy["target_predicted_adverse_rate"])
    minimum_threshold = float(threshold_policy.get("minimum_threshold", 0.01))
    maximum_threshold = float(threshold_policy.get("maximum_threshold", 0.99))
    calibrated_threshold = _threshold_for_target_predicted_adverse_rate(
        scores=[prediction.score for _row, prediction in provisional_predictions],
        target_rate=target_rate,
        minimum_threshold=minimum_threshold,
        maximum_threshold=maximum_threshold,
    )
    model.threshold = calibrated_threshold
    adjusted_predictions = _predict_rows(
        model=model,
        rows=rows,
        instance_id_prefix=instance_id_prefix,
    )
    return adjusted_predictions, {
        **base_summary,
        "applied_threshold": calibrated_threshold,
        "target_predicted_adverse_rate": target_rate,
        "achieved_predicted_adverse_rate": _predicted_adverse_rate(adjusted_predictions),
    }


def _predict_rows(
    *,
    model: Any,
    rows: list[dict[str, Any]],
    instance_id_prefix: str,
) -> list[tuple[dict[str, Any], Any]]:
    if hasattr(model, "predict_rows"):
        batch_predictions = model.predict_rows(
            rows,
            instance_id_prefix=instance_id_prefix,
        )
        return list(zip(rows, batch_predictions, strict=False))

    predictions = []
    for index, row in enumerate(rows):
        prediction = model.predict_row(
            row=row,
            instance_id=f"{instance_id_prefix}-row{index}",
        )
        predictions.append((row, prediction))
    return predictions


def _predicted_adverse_rate(predictions: list[tuple[dict[str, Any], Any]]) -> float | None:
    if not predictions:
        return None
    adverse_threshold = predictions[0][1].threshold
    return sum(
        1
        for _row, prediction in predictions
        if prediction.threshold == adverse_threshold and prediction.score >= adverse_threshold
    ) / len(predictions)


def _threshold_for_target_predicted_adverse_rate(
    *,
    scores: list[float],
    target_rate: float,
    minimum_threshold: float,
    maximum_threshold: float,
) -> float:
    if not scores:
        return min(max(0.5, minimum_threshold), maximum_threshold)

    ordered = sorted(float(score) for score in scores)
    desired_count = max(1, min(len(ordered), int(round(len(ordered) * target_rate))))
    cutoff_index = max(0, len(ordered) - desired_count)
    lower = ordered[cutoff_index]
    if cutoff_index < len(ordered) - 1:
        upper = ordered[cutoff_index + 1]
        threshold = lower if lower == upper else (lower + upper) / 2.0
    else:
        threshold = lower
    return min(max(threshold, minimum_threshold), maximum_threshold)


def _roc_auc(y_true: list[int], scores: list[float]) -> float | None:
    positives = sum(y_true)
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        return None

    ordered = sorted(zip(scores, y_true, strict=False), key=lambda item: item[0])
    rank = 1
    sum_positive_ranks = 0.0
    index = 0
    while index < len(ordered):
        next_index = index + 1
        while next_index < len(ordered) and ordered[next_index][0] == ordered[index][0]:
            next_index += 1
        average_rank = (rank + (rank + (next_index - index) - 1)) / 2.0
        positive_in_block = sum(label for _score, label in ordered[index:next_index])
        sum_positive_ranks += positive_in_block * average_rank
        rank += next_index - index
        index = next_index

    return (
        sum_positive_ranks - (positives * (positives + 1) / 2.0)
    ) / (positives * negatives)


def _average_precision(y_true: list[int], scores: list[float]) -> float | None:
    positives = sum(y_true)
    if positives == 0:
        return None

    ordered = sorted(zip(scores, y_true, strict=False), key=lambda item: item[0], reverse=True)
    true_positives = 0
    precision_sum = 0.0
    for rank, (_score, label) in enumerate(ordered, start=1):
        if label != 1:
            continue
        true_positives += 1
        precision_sum += true_positives / rank
    return precision_sum / positives


def _brier_score(y_true: list[int], scores: list[float]) -> float | None:
    if not y_true:
        return None
    return sum((float(label) - float(score)) ** 2 for label, score in zip(y_true, scores, strict=False)) / len(y_true)


def _expected_calibration_error(
    y_true: list[int],
    scores: list[float],
    *,
    num_bins: int = 10,
) -> float | None:
    if not y_true:
        return None

    bins: list[list[tuple[int, float]]] = [[] for _ in range(num_bins)]
    for label, score in zip(y_true, scores, strict=False):
        clipped = min(max(float(score), 0.0), 1.0)
        index = min(num_bins - 1, int(clipped * num_bins))
        bins[index].append((int(label), clipped))

    total = len(y_true)
    ece = 0.0
    for items in bins:
        if not items:
            continue
        mean_label = sum(label for label, _score in items) / len(items)
        mean_score = sum(score for _label, score in items) / len(items)
        ece += abs(mean_label - mean_score) * (len(items) / total)
    return ece


def _subgroup_attributes(dataset_id: str, row: dict[str, Any]) -> dict[str, str]:
    attributes: dict[str, str] = {}

    if dataset_id == "default_of_credit_card_clients":
        sex_value = str(row.get("SEX", "")).strip()
        if sex_value == "1":
            attributes["sex"] = "male"
        elif sex_value == "2":
            attributes["sex"] = "female"

        age_value = _safe_float(row.get("AGE"))
        if age_value is not None:
            attributes["age_band"] = _bucket_numeric(
                age_value,
                thresholds=[(30, "<30"), (50, "30-49")],
                default_label="50+",
            )

        education_value = str(row.get("EDUCATION", "")).strip()
        if education_value:
            attributes["education_group"] = {
                "1": "graduate_school",
                "2": "university",
                "3": "high_school",
                "4": "other",
                "5": "other",
                "6": "other",
                "0": "unknown",
            }.get(education_value, "other")

    if dataset_id == "give_me_some_credit":
        age_value = _safe_float(row.get("age"))
        if age_value is not None:
            attributes["age_band"] = _bucket_numeric(
                age_value,
                thresholds=[(35, "<35"), (55, "35-54")],
                default_label="55+",
            )

        dependents_value = _safe_float(row.get("NumberOfDependents"))
        if dependents_value is not None:
            attributes["dependents_group"] = (
                "0" if dependents_value == 0 else "1+"
            )

    if dataset_id == "south_german_credit":
        status_value = str(row.get("personal_status_sex", "")).strip()
        if status_value:
            normalized = status_value.lower()
            if "female" in normalized:
                attributes["sex_status_group"] = "female"
            elif "male" in normalized:
                attributes["sex_status_group"] = "male"

    return attributes


def _bucket_numeric(
    value: float,
    *,
    thresholds: list[tuple[float, str]],
    default_label: str,
) -> str:
    for threshold, label in thresholds:
        if value < threshold:
            return label
    return default_label


def _safe_float(value: Any) -> float | None:
    if value in (None, "", "__MISSING__"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sample_evaluation_rows(
    evaluation_rows: list[tuple[dict[str, Any], Any]],
    sampling_cfg: dict[str, Any],
    *,
    selection: str = "boundary_nearest",
    seed_key: str | None = None,
    seed: int = 42,
) -> tuple[list[tuple[dict[str, Any], Any]], dict[str, Any]]:
    """Pick up to max_instances adverse rows for explanation.

    selection="boundary_nearest" (v1, published bundles): within each decision-margin
    stratum, take the rows closest to the stratum's lower boundary first.
    selection="stratified_random" (cslaw corrections v2): same strata and the same
    round-robin over the requested strata order, but each stratum is shuffled with
    random.Random(zlib.crc32(f"{seed}|{seed_key}")) first, so the rows are a seeded
    uniform draw within each stratum. seed_key identifies the sampling unit
    (dataset|track|model, or dataset|shared|scope); it never involves the explainer
    list, so every explainer of a configuration explains the same applicants.
    """
    if selection not in SAMPLING_SELECTIONS:
        raise ValueError(
            f"sampling selection must be one of {SAMPLING_SELECTIONS}, got {selection!r}"
        )
    max_instances = int(sampling_cfg["max_instances_per_dataset_model_pair"])
    rng: random.Random | None = None
    # v1 summaries carry no selection keys; the v2 keys go last.
    selection_fields: dict[str, Any] = {}
    if selection != "boundary_nearest":
        if seed_key is None:
            raise ValueError(f"sampling selection {selection!r} needs a seed_key")
        rng = random.Random(zlib.crc32(f"{int(seed)}|{seed_key}".encode("utf-8")))
        selection_fields = {"selection": selection, "seed": int(seed)}

    if len(evaluation_rows) <= max_instances:
        return evaluation_rows, {
            "strategy": "all_rows",
            "requested_max_instances": max_instances,
            "available_adverse_rows": len(evaluation_rows),
            "selected_rows": len(evaluation_rows),
            "strata_used": _strata_used(evaluation_rows),
            **selection_fields,
        }

    if not sampling_cfg.get("stratify_by_decision_margin", False):
        if rng is not None:
            # Seeded uniform draw over all adverse rows. The margin sort only fixes
            # the order the shuffle starts from.
            pool = sorted(evaluation_rows, key=lambda item: abs(item[1].decision_margin))
            rng.shuffle(pool)
            selected = pool[:max_instances]
            return selected, {
                "strategy": "random_unstratified",
                "requested_max_instances": max_instances,
                "available_adverse_rows": len(evaluation_rows),
                "selected_rows": len(selected),
                "strata_used": _strata_used(selected),
                **selection_fields,
            }
        selected = sorted(
            evaluation_rows,
            key=lambda item: abs(item[1].decision_margin),
            reverse=True,
        )[:max_instances]
        return selected, {
            "strategy": "top_margin_only",
            "requested_max_instances": max_instances,
            "available_adverse_rows": len(evaluation_rows),
            "selected_rows": len(selected),
            "strata_used": _strata_used(selected),
        }

    strata = {
        "borderline": [],
        "medium_confidence": [],
        "strong_denial": [],
    }
    for row, prediction in evaluation_rows:
        strata[_decision_margin_stratum(prediction.decision_margin)].append((row, prediction))

    ordered = {
        key: sorted(values, key=lambda item: abs(item[1].decision_margin))
        for key, values in strata.items()
    }
    if rng is not None:
        # One RNG, strata shuffled in this fixed order (not the requested order),
        # starting from the margin-sorted order so the draw does not depend on how
        # the caller listed the rows.
        for key in ordered:
            rng.shuffle(ordered[key])
    requested_order = list(sampling_cfg.get("strata", [])) or list(ordered.keys())
    selected: list[tuple[dict[str, Any], Any]] = []

    while len(selected) < max_instances:
        added = False
        for stratum in requested_order:
            bucket = ordered.get(stratum, [])
            if bucket and len(selected) < max_instances:
                selected.append(bucket.pop(0))
                added = True
        if not added:
            break

    return selected, {
        "strategy": "decision_margin_stratified",
        "requested_max_instances": max_instances,
        "available_adverse_rows": len(evaluation_rows),
        "selected_rows": len(selected),
        "strata_used": _strata_used(selected),
        **selection_fields,
    }


def _decision_margin_stratum(decision_margin: float) -> str:
    absolute_margin = abs(decision_margin)
    if absolute_margin <= 0.1:
        return "borderline"
    if absolute_margin <= 0.25:
        return "medium_confidence"
    return "strong_denial"


def _strata_used(rows: list[tuple[dict[str, Any], Any]]) -> list[str]:
    used = {
        _decision_margin_stratum(prediction.decision_margin)
        for _row, prediction in rows
    }
    return sorted(used)


# ---------------------------------------------------------------------------
# Versioned corrections (corrections.py) and data provenance.
#
# With V1 options (no cslaw block, or no cslaw.corrections) these helpers change
# nothing the run records: the model gets corrections=V1, which is also what
# model_corrections() returns for a model without the attribute.
# ---------------------------------------------------------------------------

_PROVENANCE_CHUNK_BYTES = 1 << 20


def _apply_model_corrections(model: Any, corrections: CorrectionOptions) -> None:
    """Attach the run's corrections to a fitted model (read via model_corrections)."""
    try:
        model.corrections = corrections
    except AttributeError:
        # A model that cannot take the attribute would silently run v1 code paths.
        if not corrections.is_v1:
            raise
    if corrections.native_contributions:
        model.use_native_ranking = True


def _records_corrections(corrections: CorrectionOptions) -> bool:
    """True when the run records a corrections marker (version, options, progress event).

    The pre-v2 flag cslaw.use_native_contributions_for_ranking maps to
    native_contributions=True in corrections_from_payload (the model option must
    still be switched), but with every other option at its v1 value the run is
    plain pre-v2 instrumentation: v1 scoring and sampling, already echoed as
    cslaw.config.use_native_contributions_for_ranking. Such a run is not labelled
    as corrected, so pre-v2 bundles keep their exact payload shape.
    """
    if corrections.is_v1:
        return False
    return dataclasses.replace(corrections, native_contributions=False) != V1


def _corrections_version(corrections: CorrectionOptions) -> str:
    if corrections == V2:
        return "v2"
    if corrections.is_v1:
        return "v1"
    # A preset with explicit overrides: the full option set is recorded next to it.
    return "custom"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_PROVENANCE_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_provenance(path: Path) -> dict[str, Any]:
    entry: dict[str, Any] = {"path": repo_relative_path(path), "sha256": None}
    try:
        entry["sha256"] = _sha256_file(path)
        entry["bytes"] = path.stat().st_size
    except OSError as exc:
        # Recorded instead of raised: losing a long run over a hash is worse than
        # a provenance entry that names the problem.
        entry["error"] = f"{exc.__class__.__name__}: {exc}"
    return entry


def _record_dataset_provenance(
    store: dict[str, dict[str, Any]],
    dataset_id: str,
    table: TabularDataset,
) -> None:
    """Hash a dataset's source file once per run (both tracks read the same file)."""
    if dataset_id in store:
        return
    hash_start = perf_counter()
    entry = _file_provenance(Path(table.source_path))
    entry["rows"] = len(table.rows)
    entry["source_kind"] = table.source_kind
    store[dataset_id] = entry
    emit_progress(
        "cslaw_dataset_provenance",
        dataset_id=dataset_id,
        sha256=entry.get("sha256"),
        rows=entry["rows"],
        elapsed_seconds=round(perf_counter() - hash_start, 3),
    )


def _data_provenance(config, datasets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    registry_paths = config.registry_paths
    metadata = {
        key: _file_provenance(Path(registry_paths[key]))
        for key in ("feature_taxonomy", "dataset_registry")
        if key in registry_paths
    }
    config_path = getattr(config, "path", None)
    git_commit = os.environ.get("GIT_COMMIT", "").strip()
    return {
        "datasets": {dataset_id: datasets[dataset_id] for dataset_id in sorted(datasets)},
        "metadata": metadata,
        "config": _file_provenance(Path(config_path)) if config_path is not None else None,
        "git_commit": git_commit or None,
    }


# ---------------------------------------------------------------------------
# CS&Law opt-in instrumentation (experiment config key "cslaw").
#
# run_demo_benchmark() reaches the helpers below only when the config carries a
# "cslaw" block. Without the block none of them runs, so payloads, RNG streams
# and explainer call order are identical to the baseline commit.
# ---------------------------------------------------------------------------

_CSLAW_SHARED_SCOPES = ("models", "models_and_tracks")
_CSLAW_MARGIN_AGGREGATES = ("min", "mean", "first_model")
_CSLAW_FAITHFULNESS_REFERENCES = ("self", "native")
_CSLAW_NULL_BASELINE_EXPLAINERS = ("random_reasons", "global_ranking")
_CSLAW_NATIVE_UNITS = "log_odds_adverse"
# Rows per native-contribution batch call. Credit eval tables fit in one call;
# the cap only bounds memory for very large eval tables (e.g. HMDA).
_CSLAW_NATIVE_CHUNK_ROWS = 50000
_CSLAW_STRATA = ("borderline", "medium_confidence", "strong_denial")
_CSLAW_ROW_SUFFIX = re.compile(r"-row(\d+)$")


@dataclass(slots=True)
class CslawSettings:
    use_native_contributions_for_ranking: bool
    persist_canonical_explanations: bool
    persist_native_contributions: bool
    regb_reference: dict[str, Any]
    shared_sampling: dict[str, Any]
    null_baselines: dict[str, Any]
    null_baselines_in_use: bool
    # Versioned corrections from cslaw.corrections (V1 when absent). Recorded in
    # payload["cslaw"]["corrections"], not in to_dict(), so the pre-v2 echo is unchanged.
    corrections: CorrectionOptions = V1

    @property
    def regb_reference_enabled(self) -> bool:
        return bool(self.regb_reference["enabled"])

    @property
    def shared_sampling_enabled(self) -> bool:
        return bool(self.shared_sampling["enabled"])

    @property
    def persist_eval_row_index(self) -> bool:
        return self.persist_canonical_explanations or self.persist_native_contributions

    @property
    def active(self) -> bool:
        """True when at least one opt-in feature changes what the run records."""
        return (
            self.use_native_contributions_for_ranking
            or self.persist_canonical_explanations
            or self.persist_native_contributions
            or self.regb_reference_enabled
            or self.shared_sampling_enabled
            or self.null_baselines_in_use
            or not self.corrections.is_v1
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "use_native_contributions_for_ranking": self.use_native_contributions_for_ranking,
            "persist_canonical_explanations": self.persist_canonical_explanations,
            "persist_native_contributions": self.persist_native_contributions,
            "regb_reference": dict(self.regb_reference),
            "shared_sampling": dict(self.shared_sampling),
            "null_baselines": dict(self.null_baselines),
        }


@dataclass(slots=True)
class _CslawTrackContext:
    track: str
    bundle: Any
    table: TabularDataset
    train_table: TabularDataset
    eval_table: TabularDataset
    split_summary: dict[str, Any]
    canonicalizer: ExplanationCanonicalizer
    track_start: float


@dataclass(slots=True)
class _CslawFittedModel:
    model: Any
    model_backend: Any
    predictions: list[tuple[dict[str, Any], Any]]
    threshold_summary: dict[str, Any]
    model_performance: dict[str, Any]
    model_start: float


@dataclass(slots=True)
class _CslawSharedMargin:
    """Pseudo-prediction for _sample_evaluation_rows, which reads only decision_margin."""

    decision_margin: float


def _cslaw_settings(config) -> CslawSettings | None:
    raw = config.payload.get("cslaw")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("experiment config key 'cslaw' must be a JSON object")

    regb_raw = _cslaw_block(raw, "regb_reference")
    shared_raw = _cslaw_block(raw, "shared_sampling")
    null_raw = _cslaw_block(raw, "null_baselines")
    regb_reference = {
        **regb_raw,
        "enabled": bool(regb_raw.get("enabled", False)),
        "near_cutoff_min_rows": int(regb_raw.get("near_cutoff_min_rows", 200)),
        "near_cutoff_max_band": float(regb_raw.get("near_cutoff_max_band", 0.10)),
        "all_applicants_max_rows": int(regb_raw.get("all_applicants_max_rows", 5000)),
        "seed": int(regb_raw.get("seed", 42)),
    }
    shared_sampling = {
        **shared_raw,
        "enabled": bool(shared_raw.get("enabled", False)),
        "scope": str(shared_raw.get("scope", "models")),
        "margin_aggregate": str(shared_raw.get("margin_aggregate", "min")),
    }
    null_baselines = {
        **null_raw,
        "faithfulness_reference": str(null_raw.get("faithfulness_reference", "self")),
        "num_features": int(null_raw.get("num_features", 6)),
    }

    if shared_sampling["scope"] not in _CSLAW_SHARED_SCOPES:
        raise ValueError(
            "cslaw.shared_sampling.scope must be one of "
            f"{list(_CSLAW_SHARED_SCOPES)}, got {shared_sampling['scope']!r}"
        )
    if shared_sampling["margin_aggregate"] not in _CSLAW_MARGIN_AGGREGATES:
        raise ValueError(
            "cslaw.shared_sampling.margin_aggregate must be one of "
            f"{list(_CSLAW_MARGIN_AGGREGATES)}, got {shared_sampling['margin_aggregate']!r}"
        )
    if null_baselines["faithfulness_reference"] not in _CSLAW_FAITHFULNESS_REFERENCES:
        raise ValueError(
            "cslaw.null_baselines.faithfulness_reference must be one of "
            f"{list(_CSLAW_FAITHFULNESS_REFERENCES)}, "
            f"got {null_baselines['faithfulness_reference']!r}"
        )
    if null_baselines["num_features"] < 1:
        raise ValueError("cslaw.null_baselines.num_features must be >= 1")
    if regb_reference["near_cutoff_min_rows"] < 0:
        raise ValueError("cslaw.regb_reference.near_cutoff_min_rows must be >= 0")
    if regb_reference["near_cutoff_max_band"] < 0:
        raise ValueError("cslaw.regb_reference.near_cutoff_max_band must be >= 0")
    if regb_reference["all_applicants_max_rows"] < 0:
        raise ValueError("cslaw.regb_reference.all_applicants_max_rows must be >= 0")

    corrections = corrections_from_config(config)
    return CslawSettings(
        # The echo reports the effective ranking input: the legacy flag, or the
        # native_contributions correction (v2), which switches the same model option.
        use_native_contributions_for_ranking=(
            bool(raw.get("use_native_contributions_for_ranking", False))
            or corrections.native_contributions
        ),
        persist_canonical_explanations=bool(raw.get("persist_canonical_explanations", False)),
        persist_native_contributions=bool(raw.get("persist_native_contributions", False)),
        regb_reference=regb_reference,
        shared_sampling=shared_sampling,
        null_baselines=null_baselines,
        null_baselines_in_use=any(
            explainer_id in _CSLAW_NULL_BASELINE_EXPLAINERS
            for explainer_id in config.enabled_explainers
        ),
        corrections=corrections,
    )


def _cslaw_block(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"cslaw.{key} must be a JSON object")
    return dict(value)


def _cslaw_group_key(dataset_id: str, track: str, model_id: str) -> str:
    return f"{dataset_id}|{track}|{model_id}"


def _cslaw_eval_row_index(instance_id: str) -> int | None:
    match = _CSLAW_ROW_SUFFIX.search(str(instance_id))
    return int(match.group(1)) if match else None


def _cslaw_apply_model_options(model: Any, cslaw: CslawSettings) -> None:
    if hasattr(model, "use_native_ranking"):
        model.use_native_ranking = cslaw.use_native_contributions_for_ranking
    try:
        model.null_baseline_options = dict(cslaw.null_baselines)
    except AttributeError:
        # Objects with __slots__ cannot take the attribute; the null-baseline
        # explainers then fall back to their documented defaults.
        pass


def _cslaw_native_source(model: Any) -> str:
    try:
        source = getattr(model, "native_contribution_source", None)
    except Exception:  # a failing property must not abort a long run
        source = None
    return str(source) if source is not None else "unavailable"


def _cslaw_iter_native_batches(
    model: Any,
    rows: list[dict[str, Any]],
) -> Iterator[tuple[int, list[dict[str, float]], float]]:
    """Yield (offset, values, bias) per chunk of native_feature_contributions_batch."""
    batch_fn = getattr(model, "native_feature_contributions_batch", None)
    if not callable(batch_fn):
        raise AttributeError(
            f"{type(model).__name__} does not implement native_feature_contributions_batch"
        )
    for start in range(0, len(rows), _CSLAW_NATIVE_CHUNK_ROWS):
        chunk = rows[start : start + _CSLAW_NATIVE_CHUNK_ROWS]
        values, bias = batch_fn(chunk)
        values = list(values)
        if len(values) != len(chunk):
            raise ValueError(
                "native_feature_contributions_batch returned "
                f"{len(values)} rows for {len(chunk)} inputs"
            )
        yield (
            start,
            [
                {str(feature_name): float(value) for feature_name, value in item.items()}
                for item in values
            ],
            float(bias),
        )


def _cslaw_native_records(
    *,
    model: Any,
    sampled_rows: list[tuple[dict[str, Any], Any]],
    errors: list[str],
) -> dict[str, dict[str, Any]]:
    """Native contributions for the sampled rows, keyed by instance_id."""
    if not sampled_rows:
        return {}
    source = _cslaw_native_source(model)
    records: dict[str, dict[str, Any]] = {}
    try:
        for start, values, bias in _cslaw_iter_native_batches(
            model,
            [row for row, _prediction in sampled_rows],
        ):
            for offset, item in enumerate(values):
                prediction = sampled_rows[start + offset][1]
                records[prediction.instance_id] = {
                    "source": source,
                    "units": _CSLAW_NATIVE_UNITS,
                    "bias": bias,
                    "values": item,
                }
    except Exception as exc:
        errors.append(f"native_contributions: {exc.__class__.__name__}: {exc}")
        return {}
    return records


def _cslaw_native_subset_means(
    *,
    model: Any,
    predictions: list[tuple[dict[str, Any], Any]],
    subsets: dict[str, list[int]],
) -> tuple[dict[str, dict[str, float]], float | None]:
    """Per-feature mean native contribution for each subset of eval positions.

    The union of all subsets goes through native_feature_contributions_batch in
    chunks, and running sums keep memory bounded on large eval tables.
    """
    membership = {name: set(positions) for name, positions in subsets.items()}
    union = sorted(set().union(*membership.values())) if membership else []
    sums: dict[str, dict[str, float]] = {name: {} for name in subsets}
    counts = {name: 0 for name in subsets}
    bias: float | None = None
    rows = [predictions[position][0] for position in union]
    for start, values, chunk_bias in _cslaw_iter_native_batches(model, rows):
        bias = chunk_bias
        for offset, item in enumerate(values):
            position = union[start + offset]
            for name, members in membership.items():
                if position not in members:
                    continue
                counts[name] += 1
                target = sums[name]
                for feature_name, value in item.items():
                    target[feature_name] = target.get(feature_name, 0.0) + value
    means = {
        name: (
            {feature_name: total / counts[name] for feature_name, total in sums[name].items()}
            if counts[name]
            else {}
        )
        for name in subsets
    }
    return means, bias


def _cslaw_realized_threshold(
    model: Any,
    predictions: list[tuple[dict[str, Any], Any]],
) -> float | None:
    if predictions:
        value = predictions[0][1].threshold
        if value is not None:
            return float(value)
    value = getattr(model, "threshold", None)
    return float(value) if value is not None else None


def _cslaw_regb_reference(
    *,
    model: Any,
    negative_label: str,
    predictions: list[tuple[dict[str, Any], Any]],
    settings: dict[str, Any],
    errors: list[str],
) -> dict[str, Any]:
    """Reg B style reference means over approved-near-cutoff and all applicants.

    Approved rows are predictions whose label is not the adverse label (score =
    P(adverse) below the realized threshold). Near-cutoff rows are the approved
    rows within near_cutoff_max_band of the threshold. If that band holds fewer
    than near_cutoff_min_rows rows, the near_cutoff_min_rows highest-scoring
    approved rows are used instead.
    """
    threshold = _cslaw_realized_threshold(model, predictions)
    band = float(settings["near_cutoff_max_band"])
    min_rows = int(settings["near_cutoff_min_rows"])
    max_rows = int(settings["all_applicants_max_rows"])
    seed = int(settings["seed"])

    def approved_rows() -> Iterator[tuple[int, float]]:
        for position, (_row, prediction) in enumerate(predictions):
            if prediction.label != negative_label:
                yield position, float(prediction.score)

    def closest_first(item: tuple[int, float]) -> tuple[float, int]:
        # Closest to the cutoff from the approved side first (score descending);
        # ties keep eval order.
        return (-item[1], item[0])

    # Stream the approved rows instead of sorting all of them, which matters
    # for very large eval tables.
    approved_count = 0
    band_rows: list[tuple[int, float]] = []
    for item in approved_rows():
        approved_count += 1
        if threshold is not None and item[1] >= threshold - band:
            band_rows.append(item)
    if len(band_rows) >= min_rows:
        near = sorted(band_rows, key=closest_first)
        selection = "band"
    else:
        near = heapq.nsmallest(min_rows, approved_rows(), key=closest_first)
        selection = "top_scoring_approved"
    near_positions = [position for position, _score in near]

    sample_size = min(len(predictions), max_rows)
    all_positions = sorted(random.Random(seed).sample(range(len(predictions)), sample_size))

    record: dict[str, Any] = {
        "source": _cslaw_native_source(model),
        "units": _CSLAW_NATIVE_UNITS,
        "bias": None,
        "threshold": threshold,
        "eval_rows": len(predictions),
        "approved_rows": approved_count,
        "near_cutoff": {
            "n_rows": len(near_positions),
            "score_low": near[-1][1] if near else None,
            "score_high": near[0][1] if near else None,
            "selection": selection,
            "max_band": band,
            "min_rows": min_rows,
            "means": {},
        },
        "all_applicants": {
            "n_rows": len(all_positions),
            "max_rows": max_rows,
            "seed": seed,
            "means": {},
        },
    }
    try:
        means, bias = _cslaw_native_subset_means(
            model=model,
            predictions=predictions,
            subsets={
                "near_cutoff": near_positions,
                "all_applicants": all_positions,
            },
        )
    except Exception as exc:
        message = f"{exc.__class__.__name__}: {exc}"
        errors.append(f"regb_reference: {message}")
        record["error"] = message
        return record
    record["bias"] = bias
    record["near_cutoff"]["means"] = means["near_cutoff"]
    record["all_applicants"]["means"] = means["all_applicants"]
    return record


def _cslaw_prepare_model_group(
    *,
    cslaw: CslawSettings,
    cslaw_groups: dict[str, dict[str, Any]],
    dataset_id: str,
    track: str,
    model_id: str,
    model: Any,
    model_backend: Any,
    negative_label: str,
    predictions: list[tuple[dict[str, Any], Any]],
    sampled_rows: list[tuple[dict[str, Any], Any]],
    shared_sampling: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    """Record the run-level cslaw group entry and return per-instance native records."""
    group_start = perf_counter()
    errors: list[str] = []
    regb_reference = None
    if cslaw.regb_reference_enabled:
        regb_reference = _cslaw_regb_reference(
            model=model,
            negative_label=negative_label,
            predictions=predictions,
            settings=cslaw.regb_reference,
            errors=errors,
        )
    native_records: dict[str, dict[str, Any]] = {}
    if cslaw.persist_native_contributions:
        native_records = _cslaw_native_records(
            model=model,
            sampled_rows=sampled_rows,
            errors=errors,
        )
    native_source = _cslaw_native_source(model)
    group: dict[str, Any] = {
        "regb_reference": regb_reference,
        "shared_sampling": shared_sampling,
        "model_backend": model_backend.implementation,
        "native_contribution_source": native_source,
    }
    if errors:
        group["errors"] = errors
    cslaw_groups[_cslaw_group_key(dataset_id, track, model_id)] = group
    emit_progress(
        "cslaw_group_ready",
        dataset_id=dataset_id,
        track=track,
        model_id=model_id,
        native_contribution_source=native_source,
        native_records=len(native_records),
        regb_near_cutoff_rows=(
            regb_reference["near_cutoff"]["n_rows"] if regb_reference else None
        ),
        regb_all_applicants_rows=(
            regb_reference["all_applicants"]["n_rows"] if regb_reference else None
        ),
        errors=errors,
        elapsed_seconds=round(perf_counter() - group_start, 3),
    )
    return native_records


def _cslaw_attach_instance_fields(
    instance: dict[str, Any],
    *,
    cslaw: CslawSettings,
    canonical: Any,
    prediction: Any,
    native_records: dict[str, dict[str, Any]],
) -> None:
    if cslaw.persist_eval_row_index:
        instance["eval_row_index"] = _cslaw_eval_row_index(prediction.instance_id)
    if cslaw.persist_canonical_explanations:
        instance["canonical_explanation"] = _strict_json(canonical.to_dict())
    if cslaw.persist_native_contributions:
        record = native_records.get(prediction.instance_id)
        instance["native_contributions"] = _strict_json(record) if record is not None else None


def _strict_json(value: Any) -> Any:
    """Convert to strict JSON: numpy scalars/arrays to Python, NaN/inf to None."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return str(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _strict_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_strict_json(item) for item in sorted(value, key=str)]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _strict_json(dataclasses.asdict(value))
    tolist = getattr(value, "tolist", None)  # numpy scalars and arrays
    if callable(tolist):
        return _strict_json(tolist())
    item = getattr(value, "item", None)
    if callable(item):
        return _strict_json(item())
    return str(value)


def _cslaw_stratum_counts(margins: list[float]) -> dict[str, int]:
    counts = {stratum: 0 for stratum in _CSLAW_STRATA}
    for margin in margins:
        stratum = _decision_margin_stratum(margin)
        counts[stratum] = counts.get(stratum, 0) + 1
    return counts


def _cslaw_aggregate_margin(margins: list[float], aggregate: str) -> float:
    if aggregate == "mean":
        return sum(margins) / len(margins)
    if aggregate == "first_model":
        return margins[0]
    return min(margins)


def _run_cslaw_shared_dataset(
    *,
    config,
    dataset_id: str,
    tracks: list[str],
    scorer: ExplanationScorer,
    sampling_cfg: dict[str, Any],
    evaluation_cfg: dict[str, Any],
    threshold_policy: dict[str, Any],
    cslaw: CslawSettings,
    cslaw_groups: dict[str, dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    provenance_datasets: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Shared-sampling path: every included (track, model) explains the same applicants.

    Separate from the default loop on purpose. It fits every model of a track
    group, keeps the eval positions that every included (track, model) predicts
    adverse, aggregates their decision margins, and reuses _sample_evaluation_rows
    on that pool. All groups then explain the same positions in the same order.
    """
    dataset_start = perf_counter()
    scope = str(cslaw.shared_sampling["scope"])
    emit_progress(
        "dataset_start",
        dataset_id=dataset_id,
        sampling_mode="cslaw_shared",
        shared_scope=scope,
    )
    loader = build_dataset_loader(config, dataset_id)
    if scope == "models_and_tracks":
        track_groups = [list(tracks)] if tracks else []
    else:
        track_groups = [[track] for track in tracks]

    for track_group in track_groups:
        contexts = {
            track: _cslaw_prepare_track(
                loader=loader,
                dataset_id=dataset_id,
                track=track,
                evaluation_cfg=evaluation_cfg,
                corrections=cslaw.corrections,
            )
            for track in track_group
        }
        if provenance_datasets is not None:
            for context in contexts.values():
                _record_dataset_provenance(provenance_datasets, dataset_id, context.table)
        keys = [(track, model_id) for track in track_group for model_id in config.enabled_models]
        fitted: dict[tuple[str, str], _CslawFittedModel] = {}
        for track, model_id in keys:
            fitted[(track, model_id)] = _cslaw_fit_and_predict(
                config=config,
                context=contexts[track],
                dataset_id=dataset_id,
                model_id=model_id,
                threshold_policy=threshold_policy,
                cslaw=cslaw,
            )
        shared = _cslaw_select_shared_positions(
            contexts=contexts,
            fitted=fitted,
            keys=keys,
            sampling_cfg=sampling_cfg,
            cslaw=cslaw,
            seed_key=_cslaw_shared_seed_key(dataset_id, scope, track_group),
        )
        emit_progress(
            "cslaw_shared_sampling_ready",
            dataset_id=dataset_id,
            tracks=list(track_group),
            scope=scope,
            margin_aggregate=shared["margin_aggregate"],
            shared_pool_rows=shared["shared_pool_rows"],
            selected_rows=(
                len(shared["selected_positions"])
                if shared["selected_positions"] is not None
                else None
            ),
            shared_fallback=shared["shared_fallback"],
            fallback_reason=shared["fallback_reason"],
        )
        for track in track_group:
            context = contexts[track]
            for model_id in config.enabled_models:
                entry = fitted.pop((track, model_id))
                _cslaw_explain_model_group(
                    config=config,
                    scorer=scorer,
                    sampling_cfg=sampling_cfg,
                    cslaw=cslaw,
                    cslaw_groups=cslaw_groups,
                    summary_rows=summary_rows,
                    dataset_id=dataset_id,
                    context=context,
                    model_id=model_id,
                    entry=entry,
                    shared=shared,
                )
            emit_progress(
                "track_done",
                dataset_id=dataset_id,
                track=track,
                elapsed_seconds=round(perf_counter() - context.track_start, 3),
            )
    emit_progress(
        "dataset_done",
        dataset_id=dataset_id,
        elapsed_seconds=round(perf_counter() - dataset_start, 3),
    )


def _cslaw_prepare_track(
    *,
    loader: Any,
    dataset_id: str,
    track: str,
    evaluation_cfg: dict[str, Any],
    corrections: CorrectionOptions = V1,
) -> _CslawTrackContext:
    track_start = perf_counter()
    emit_progress("track_start", dataset_id=dataset_id, track=track)
    bundle_start = perf_counter()
    bundle = loader.build_bundle(track)
    emit_progress(
        "track_bundle_ready",
        dataset_id=dataset_id,
        track=track,
        elapsed_seconds=round(perf_counter() - bundle_start, 3),
        included_feature_count=len(bundle.included_feature_names),
        excluded_feature_count=len(bundle.excluded_feature_names),
    )
    table_load_start = perf_counter()
    table = loader.load_tabular_dataset(track=track)
    emit_progress(
        "track_table_ready",
        dataset_id=dataset_id,
        track=track,
        elapsed_seconds=round(perf_counter() - table_load_start, 3),
        total_rows=len(table.rows),
        feature_count=len(table.feature_names),
        source_kind=table.source_kind,
    )
    split_start = perf_counter()
    train_table, eval_table, split_summary = _split_table_for_evaluation(
        table=table,
        negative_label=bundle.negative_label,
        evaluation_cfg=evaluation_cfg,
    )
    emit_progress(
        "track_split_ready",
        dataset_id=dataset_id,
        track=track,
        elapsed_seconds=round(perf_counter() - split_start, 3),
        total_rows=len(table.rows),
        train_rows=len(train_table.rows),
        eval_rows=len(eval_table.rows),
        split_strategy=split_summary.get("strategy"),
        heldout_used=split_summary.get("heldout_used"),
    )
    return _CslawTrackContext(
        track=track,
        bundle=bundle,
        table=table,
        train_table=train_table,
        eval_table=eval_table,
        split_summary=split_summary,
        canonicalizer=ExplanationCanonicalizer(bundle, corrections=corrections),
        track_start=track_start,
    )


def _cslaw_fit_and_predict(
    *,
    config,
    context: _CslawTrackContext,
    dataset_id: str,
    model_id: str,
    threshold_policy: dict[str, Any],
    cslaw: CslawSettings,
) -> _CslawFittedModel:
    track = context.track
    bundle = context.bundle
    model_start = perf_counter()
    emit_progress(
        "model_start",
        dataset_id=dataset_id,
        track=track,
        model_id=model_id,
        total_rows=len(context.table.rows),
        train_rows=len(context.train_table.rows),
        eval_rows=len(context.eval_table.rows),
    )
    model, model_backend = fit_model(
        model_id=model_id,
        dataset_bundle=bundle,
        table=context.train_table,
        threshold=config.model_threshold,
    )
    _cslaw_apply_model_options(model, cslaw)
    _apply_model_corrections(model, cslaw.corrections)
    predictions, threshold_summary = _predict_with_threshold_policy(
        model=model,
        rows=context.eval_table.rows,
        instance_id_prefix=f"{dataset_id}-{track}-{model_id}",
        threshold_policy=threshold_policy,
    )
    model_performance = _evaluate_model_predictions(
        rows=context.eval_table.rows,
        predictions=[prediction for _row, prediction in predictions],
        target=bundle.target,
        positive_label=bundle.positive_label,
        negative_label=bundle.negative_label,
    )
    emit_progress(
        "cslaw_model_fitted",
        dataset_id=dataset_id,
        track=track,
        model_id=model_id,
        elapsed_seconds=round(perf_counter() - model_start, 3),
        model_backend=model_backend.implementation,
    )
    return _CslawFittedModel(
        model=model,
        model_backend=model_backend,
        predictions=predictions,
        threshold_summary=threshold_summary,
        model_performance=model_performance,
        model_start=model_start,
    )


def _cslaw_eval_tables_aligned(contexts: dict[str, _CslawTrackContext]) -> bool:
    """Cross-track sanity check: the eval tables must hold the same labels in order."""
    ordered = list(contexts.values())
    if len(ordered) < 2:
        return True
    reference = ordered[0].eval_table
    reference_labels = [str(row.get(reference.target)) for row in reference.rows]
    for other in ordered[1:]:
        labels = [str(row.get(other.eval_table.target)) for row in other.eval_table.rows]
        if labels != reference_labels:
            return False
    return True


def _cslaw_select_shared_positions(
    *,
    contexts: dict[str, _CslawTrackContext],
    fitted: dict[tuple[str, str], _CslawFittedModel],
    keys: list[tuple[str, str]],
    sampling_cfg: dict[str, Any],
    cslaw: CslawSettings,
    seed_key: str | None = None,
) -> dict[str, Any]:
    aggregate = str(cslaw.shared_sampling["margin_aggregate"])
    summary: dict[str, Any] = {
        "scope": str(cslaw.shared_sampling["scope"]),
        "margin_aggregate": aggregate,
        "included_groups": [f"{track}|{model_id}" for track, model_id in keys],
        "shared_pool_rows": 0,
        "pool_stratum_counts": _cslaw_stratum_counts([]),
        "selected_positions": None,
        "selected_stratum_counts": _cslaw_stratum_counts([]),
        "sampling_strategy": None,
        "requested_max_instances": int(sampling_cfg["max_instances_per_dataset_model_pair"]),
        "shared_fallback": False,
        "fallback_reason": None,
    }

    lengths = {len(context.eval_table.rows) for context in contexts.values()}
    lengths.update(len(fitted[key].predictions) for key in keys)
    fallback_reason: str | None = None
    if not keys:
        fallback_reason = "no_models"
    elif len(lengths) != 1:
        fallback_reason = "eval_length_mismatch"
    elif not _cslaw_eval_tables_aligned(contexts):
        fallback_reason = "eval_label_mismatch"

    pool: list[tuple[int, _CslawSharedMargin]] = []
    if fallback_reason is None:
        eval_length = next(iter(lengths))
        for position in range(eval_length):
            margins: list[float] = []
            for track, model_id in keys:
                prediction = fitted[(track, model_id)].predictions[position][1]
                if prediction.label != contexts[track].bundle.negative_label:
                    break
                margins.append(float(prediction.decision_margin))
            else:
                pool.append(
                    (
                        position,
                        _CslawSharedMargin(
                            decision_margin=_cslaw_aggregate_margin(margins, aggregate)
                        ),
                    )
                )
        if not pool:
            fallback_reason = "empty_shared_pool"

    summary["shared_pool_rows"] = len(pool)
    summary["pool_stratum_counts"] = _cslaw_stratum_counts(
        [margin.decision_margin for _position, margin in pool]
    )
    if fallback_reason is not None:
        summary["shared_fallback"] = True
        summary["fallback_reason"] = fallback_reason
        return summary

    # The pool items are (eval position, pseudo-prediction) pairs in ascending
    # position order; _sample_evaluation_rows only reads item[1].decision_margin,
    # and its stable sort breaks ties by eval order as in the default path.
    corrections = cslaw.corrections
    selected, base_summary = _sample_evaluation_rows(
        evaluation_rows=pool,
        sampling_cfg=sampling_cfg,
        selection=corrections.sampling_selection,
        seed_key=seed_key,
        seed=corrections.sampling_seed,
    )
    summary["selected_positions"] = [position for position, _margin in selected]
    summary["selected_stratum_counts"] = _cslaw_stratum_counts(
        [margin.decision_margin for _position, margin in selected]
    )
    summary["sampling_strategy"] = base_summary["strategy"]
    if "selection" in base_summary:
        summary["sampling_selection"] = base_summary["selection"]
        summary["sampling_seed"] = base_summary["seed"]
        summary["sampling_seed_key"] = seed_key
    return summary


def _cslaw_shared_seed_key(dataset_id: str, scope: str, track_group: list[str]) -> str:
    """Seed key of a shared-sampling draw; it never involves the explainer list.

    Scope "models_and_tracks" draws once per dataset: "<dataset>|shared|<scope>".
    Scope "models" draws once per track, so the track is appended; otherwise both
    tracks would reuse one RNG stream on different pools.
    """
    key = f"{dataset_id}|shared|{scope}"
    if scope == "models":
        key += "|" + "+".join(track_group)
    return key


def _cslaw_explain_model_group(
    *,
    config,
    scorer: ExplanationScorer,
    sampling_cfg: dict[str, Any],
    cslaw: CslawSettings,
    cslaw_groups: dict[str, dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    dataset_id: str,
    context: _CslawTrackContext,
    model_id: str,
    entry: _CslawFittedModel,
    shared: dict[str, Any],
) -> None:
    track = context.track
    bundle = context.bundle
    model = entry.model
    model_backend = entry.model_backend
    predictions = entry.predictions
    model_start = entry.model_start

    evaluation_rows = [
        (row, prediction)
        for row, prediction in predictions
        if prediction.label == bundle.negative_label
    ]
    if not evaluation_rows and predictions:
        fallback = max(predictions, key=lambda item: item[1].score)
        evaluation_rows = [fallback]
    adverse_rate = sum(
        1 for _row, prediction in predictions if prediction.label == bundle.negative_label
    ) / max(1, len(context.eval_table.rows))

    if shared["selected_positions"] is not None:
        sampled_rows = [predictions[position] for position in shared["selected_positions"]]
        sampling_summary = {
            "strategy": f"shared_{shared['sampling_strategy']}",
            "requested_max_instances": shared["requested_max_instances"],
            "available_adverse_rows": len(evaluation_rows),
            "selected_rows": len(sampled_rows),
            "strata_used": _strata_used(sampled_rows),
            "shared_scope": shared["scope"],
            "shared_pool_rows": shared["shared_pool_rows"],
            "margin_aggregate": shared["margin_aggregate"],
        }
        if "sampling_selection" in shared:
            sampling_summary["selection"] = shared["sampling_selection"]
            sampling_summary["seed"] = shared["sampling_seed"]
        group_shared = dict(shared)
    else:
        # Empty or misaligned shared pool: per-model default sampling for this dataset,
        # with the same seed key as the independent path.
        sampled_rows, sampling_summary = _sample_evaluation_rows(
            evaluation_rows=evaluation_rows,
            sampling_cfg=sampling_cfg,
            selection=cslaw.corrections.sampling_selection,
            seed_key=f"{dataset_id}|{track}|{model_id}",
            seed=cslaw.corrections.sampling_seed,
        )
        sampling_summary = {
            **sampling_summary,
            "shared_fallback": True,
            "fallback_reason": shared["fallback_reason"],
        }
        group_shared = {
            **shared,
            "selected_positions": [
                _cslaw_eval_row_index(prediction.instance_id)
                for _row, prediction in sampled_rows
            ],
        }
    group_shared["model_selected_stratum_counts"] = _cslaw_stratum_counts(
        [float(prediction.decision_margin) for _row, prediction in sampled_rows]
    )

    emit_progress(
        "model_ready",
        dataset_id=dataset_id,
        track=track,
        model_id=model_id,
        adverse_rows=len(evaluation_rows),
        sampled_rows=len(sampled_rows),
        adverse_rate=adverse_rate,
        elapsed_seconds=round(perf_counter() - model_start, 3),
        model_backend=model_backend.implementation,
        sampling_mode="cslaw_shared",
        shared_fallback=shared["shared_fallback"],
    )
    cslaw_native = _cslaw_prepare_model_group(
        cslaw=cslaw,
        cslaw_groups=cslaw_groups,
        dataset_id=dataset_id,
        track=track,
        model_id=model_id,
        model=model,
        model_backend=model_backend,
        negative_label=bundle.negative_label,
        predictions=predictions,
        sampled_rows=sampled_rows,
        shared_sampling=group_shared,
    )

    for explainer_id in config.enabled_explainers:
        explainer_start = perf_counter()
        emit_progress(
            "explainer_start",
            dataset_id=dataset_id,
            track=track,
            model_id=model_id,
            explainer_id=explainer_id,
            sampled_rows=len(sampled_rows),
        )
        instance_results = []
        explainer_backend = None
        for row, prediction in sampled_rows:
            raw_payload, explainer_backend = generate_raw_explanation(
                explainer_id=explainer_id,
                dataset_bundle=bundle,
                model=model,
                row=row,
                prediction=prediction,
            )
            adapted_payload = adapt_raw_example(
                dataset_bundle=bundle,
                explainer_id=explainer_id,
                raw_payload=raw_payload,
            )
            stability_evidence = compute_stability_evidence(
                probe_cfg=config.stability_probe,
                dataset_bundle=bundle,
                explainer_id=explainer_id,
                model=model,
                row=row,
                prediction=prediction,
                adapted_payload=adapted_payload,
            )
            if stability_evidence is not None:
                adapted_payload["metadata"] = {
                    **adapted_payload.get("metadata", {}),
                    "stability_evidence": stability_evidence,
                }
            canonical = context.canonicalizer.canonicalize(adapted_payload)
            scores = scorer.score(canonical)
            instance_results.append(
                {
                    "instance_id": prediction.instance_id,
                    "prediction_score": prediction.score,
                    "predicted_label": prediction.label,
                    "model_backend": {
                        "backend": model_backend.backend,
                        "implementation": model_backend.implementation,
                        "reason": model_backend.reason,
                    },
                    "explainer_backend": {
                        "backend": explainer_backend.backend,
                        "implementation": explainer_backend.implementation,
                        "reason": explainer_backend.reason,
                    },
                    "subgroup_attributes": _subgroup_attributes(
                        dataset_id=dataset_id,
                        row=row,
                    ),
                    "scores": scores,
                }
            )
            _cslaw_attach_instance_fields(
                instance_results[-1],
                cslaw=cslaw,
                canonical=canonical,
                prediction=prediction,
                native_records=cslaw_native,
            )
        if explainer_backend is None:
            # No sampled rows: record the backend the registry would select so
            # _aggregate_group never sees an unbound or stale backend.
            explainer_backend = select_explainer_backend(explainer_id)

        summary_rows.append(
            _aggregate_group(
                dataset_id=dataset_id,
                track=track,
                model_id=model_id,
                explainer_id=explainer_id,
                data_source_kind=context.eval_table.source_kind,
                data_source_path=context.eval_table.source_path,
                total_rows=len(context.table.rows),
                train_rows=len(context.train_table.rows),
                test_rows=len(context.eval_table.rows),
                adverse_rows=len(evaluation_rows),
                evaluated_rows=len(instance_results),
                model_backend=model_backend,
                explainer_backend=explainer_backend,
                adverse_rate=adverse_rate,
                split_summary=context.split_summary,
                threshold_summary=entry.threshold_summary,
                model_performance=entry.model_performance,
                sampling_summary=sampling_summary,
                instance_results=instance_results,
            )
        )
        emit_progress(
            "explainer_done",
            dataset_id=dataset_id,
            track=track,
            model_id=model_id,
            explainer_id=explainer_id,
            evaluated_rows=len(instance_results),
            elapsed_seconds=round(perf_counter() - explainer_start, 3),
            explainer_backend=explainer_backend.implementation,
        )
    emit_progress(
        "model_done",
        dataset_id=dataset_id,
        track=track,
        model_id=model_id,
        elapsed_seconds=round(perf_counter() - model_start, 3),
    )
