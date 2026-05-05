import random
from datetime import datetime, timezone
from time import perf_counter
from typing import Any

from .adapters import adapt_raw_example
from .capabilities import detect_capabilities
from .datasets.registry import build_dataset_loader
from .explainers import generate_raw_explanation
from .metrics.scorer import ExplanationScorer
from .models import fit_model
from .normalization import ExplanationCanonicalizer
from .progress import emit_progress
from .stability import compute_stability_evidence
from .types import TabularDataset


def run_demo_benchmark(config) -> dict[str, Any]:
    scorer = ExplanationScorer()
    summary_rows: list[dict[str, Any]] = []
    sampling_cfg = config.denied_instance_sampling
    evaluation_cfg = config.evaluation_protocol
    threshold_policy = config.model_threshold_policy

    tracks: list[str] = []
    if config.payload["tracks"]["run_primary_track"]:
        tracks.append("primary")
    if config.payload["tracks"]["run_audit_track"]:
        tracks.append("audit")

    for dataset_id in config.enabled_datasets:
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
            canonicalizer = ExplanationCanonicalizer(bundle)
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

    return {
        "experiment_name": config.payload["experiment"]["name"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "capabilities": detect_capabilities(),
        "overview": _build_overview(summary_rows),
        "summary": summary_rows,
    }


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
        age_value = _safe_float(row.get("age_in_years"))
        if age_value is not None:
            attributes["age_band"] = _bucket_numeric(
                age_value,
                thresholds=[(30, "<30"), (50, "30-49")],
                default_label="50+",
            )

        status_value = str(row.get("personal_status_sex", "")).strip()
        if status_value:
            normalized = status_value.lower()
            if "female" in normalized:
                attributes["sex_status_group"] = "female"
            elif "male" in normalized:
                attributes["sex_status_group"] = "male"

    if dataset_id == "heloc":
        delinquency_value = _safe_float(row.get("NumDelqLast6M"))
        if delinquency_value is not None:
            attributes["recent_delinquency_group"] = (
                "0" if delinquency_value == 0 else "1+"
            )

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
) -> tuple[list[tuple[dict[str, Any], Any]], dict[str, Any]]:
    max_instances = int(sampling_cfg["max_instances_per_dataset_model_pair"])
    if len(evaluation_rows) <= max_instances:
        return evaluation_rows, {
            "strategy": "all_rows",
            "requested_max_instances": max_instances,
            "available_adverse_rows": len(evaluation_rows),
            "selected_rows": len(evaluation_rows),
            "strata_used": _strata_used(evaluation_rows),
        }

    if not sampling_cfg.get("stratify_by_decision_margin", False):
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
