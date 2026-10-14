#!/usr/bin/env python3
"""Denial-direction correction of the principal reasons, offline.

    PYTHONPATH=benchmark/src python benchmark/cslaw/denial_reasons.py \
        --bundle DIR --out-dir DIR [--config CONFIG_JSON] [--force] [--allow-inconsistent]
    PYTHONPATH=benchmark/src python benchmark/cslaw/denial_reasons.py --selftest

Why. Every explainer ranks features by |weight|, so the canonical principal
reasons can include factors whose ``signed_direction`` is ``decreases_denial``:
they favoured approval. Regulation B requires a statement of the specific
reasons for the adverse action (12 CFR 1002.9(b)(2)), and the methods in comment
9(b)(2)-5 count only the factors on which the applicant fell short. A reason
that pushed toward approval is not a reason for the denial.

What. For every instance with a persisted ``canonical_explanation`` (cslaw
``persist_canonical_explanations``) the script

1. rebuilds the ``CanonicalExplanation`` dataclass from the persisted dict;
2. keeps the referenced features with ``signed_direction == "increases_denial"``
   (order preserved, re-ranked 1..n); counterfactual changes are kept unchanged
   (they are recourse content, not reasons);
3. rebuilds the principal reasons from the kept features with the canonicalizer's
   own construction (``ExplanationCanonicalizer._build_principal_reasons``: cap 4,
   temporal grouping per the bundle's ``reason_grouping_fix``). The
   "changing X may improve the outcome" reasons that the canonicalizer derives
   from counterfactual changes are recourse statements that favour approval, so
   they are not rebuilt; the changes themselves stay in the explanation;
4. re-scores the rebuilt explanation with ``ExplanationScorer`` under the
   bundle's corrections. Stability and faithfulness evidence stay in the
   metadata unchanged (stability describes the method; faithfulness is
   recomputed from the kept features against the same reference);
5. sets ``instance["explanation_produced"]`` (at least one denial reason) and
   ``instance["denial_filter"]`` = {reasons_before, reasons_after,
   approval_favouring_dropped, neutral_dropped, features_before, features_after};
6. excludes instances without a denial reason from the quality metrics: a
   summary row's ``instance_results`` (and ``instance_ids``) hold the produced
   explanations only, its ``mean_*`` fields, ``evaluated_rows`` and evidence-source
   counts come from ``runner._aggregate_group`` on that list, and the other
   instances move to ``denial_filter_unproduced_instances`` in the same row. New
   row fields: ``cslaw_produced_rate`` and ``cslaw_evaluated_rows_all``. A row
   with no produced explanation keeps ``mean_*`` = None, which rebuild-artifacts'
   reporting skips (``reporting._weighted_mean``, the leader, pairwise and
   correlation filters). Only when an explainer has no produced explanation in
   the whole bundle are its rows removed (``reporting._proxy_sensitivity_profiles``
   calls ``float()`` on the per-explainer mean) and listed in
   ``payload["denial_reasons"]["dropped_configurations"]`` with their instances;
7. recomputes ``payload["overview"]`` with ``runner._build_overview``, adds
   ``payload["denial_reasons"]`` and writes strict JSON to
   ``<out-dir>/results.json`` and ``<out-dir>/denial_reasons_summary.md``.
   ``payload["cslaw"]`` is kept unchanged.

How the canonicalizer is rebuilt offline. The canonicalizer needs only the
dataset bundle (feature specs: plain meaning, taxonomy) and the corrections. The
script loads the experiment config (``--config``, else ``<bundle>/runtime_config.json``,
else the path recorded in ``payload["cslaw"]["provenance"]["config"]``), points its
``registry`` block at the taxonomy and registry files recorded in
``payload["cslaw"]["provenance"]["metadata"]`` (checking their SHA-256), and
builds each (dataset, track) bundle with ``datasets.registry.build_dataset_loader
(...).build_bundle(track)``, which reads no data. Corrections come from
``payload["cslaw"]["corrections"]`` (v1 when absent). Before filtering, every
instance's principal reasons and scores are rebuilt from the unfiltered
explanation and compared with the persisted ones; any difference stops the run
(``--allow-inconsistent`` downgrades it, and a metadata hash mismatch, to a warning).

Limitation. Only the explainer's top referenced features are persisted (6 for
SHAP, LIME and the null baselines), so a denial-only notice can have fewer than
four reasons even when the model has more denial-increasing factors.

The script never writes into the input bundle directory and refuses an existing
``--out-dir`` unless ``--force`` is given. It prints the rebuild-artifacts
command for the output.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import tempfile
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_SRC = REPO_ROOT / "benchmark" / "src"
if BENCHMARK_SRC.is_dir() and str(BENCHMARK_SRC) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_SRC))

from legalbench_xai import runner  # noqa: E402
from legalbench_xai.config import ExperimentConfig  # noqa: E402
from legalbench_xai.corrections import V1, CorrectionOptions, corrections_from_payload  # noqa: E402
from legalbench_xai.metrics.scorer import ExplanationScorer  # noqa: E402
from legalbench_xai.normalization.canonicalizer import ExplanationCanonicalizer  # noqa: E402
from legalbench_xai.paths import resolve_repo_path  # noqa: E402
from legalbench_xai.types import (  # noqa: E402
    CanonicalExplanation,
    CounterfactualChange,
    CounterfactualLegalityFlags,
    DatasetBundle,
    FeatureTaxonomyRecord,
    PredictionRecord,
    PrincipalReason,
    ReferencedFeature,
)

DENIAL = "increases_denial"
APPROVAL = "decreases_denial"
RULE = (
    "principal reasons rebuilt from the referenced features with signed_direction "
    "'increases_denial' only (12 CFR 1002.9(b)(2); comment 9(b)(2)-5 methods count only "
    "factors on which the applicant fell short); counterfactual changes kept as recourse; "
    "instances without a denial reason are excluded from the quality metrics"
)
SCORE_TOLERANCE = 1e-9
UNPRODUCED_KEY = "denial_filter_unproduced_instances"


# --------------------------------------------------------------------------- #
# Reconstruction of the persisted canonical explanation
# --------------------------------------------------------------------------- #


def _taxonomy(raw: dict[str, Any] | None) -> FeatureTaxonomyRecord | None:
    return FeatureTaxonomyRecord(**raw) if raw is not None else None


def canonical_from_dict(data: dict[str, Any]) -> CanonicalExplanation:
    """Inverse of ``CanonicalExplanation.to_dict()`` (nested dataclasses included)."""
    features = [
        ReferencedFeature(**{**item, "taxonomy": _taxonomy(item.get("taxonomy"))})
        for item in data.get("referenced_features") or []
    ]
    reasons = [PrincipalReason(**item) for item in data.get("principal_reasons") or []]
    changes = []
    for item in data.get("counterfactual_changes") or []:
        flags = item.get("legality_flags")
        changes.append(
            CounterfactualChange(
                **{
                    **item,
                    "taxonomy": _taxonomy(item.get("taxonomy")),
                    "legality_flags": (
                        CounterfactualLegalityFlags(**flags)
                        if flags is not None
                        else CounterfactualLegalityFlags()
                    ),
                }
            )
        )
    return CanonicalExplanation(
        instance_id=data["instance_id"],
        dataset_id=data["dataset_id"],
        track=data["track"],
        model_id=data["model_id"],
        explainer_id=data["explainer_id"],
        explanation_family=data["explanation_family"],
        prediction=PredictionRecord(**data["prediction"]),
        referenced_features=features,
        principal_reasons=reasons,
        counterfactual_changes=changes,
        seed=data.get("seed"),
        metadata=copy.deepcopy(data.get("metadata") or {}),
    )


def reasons_signature(reasons: list[PrincipalReason]) -> list[tuple[Any, ...]]:
    return [
        (
            reason.rank,
            reason.reason_text,
            tuple(reason.supporting_features),
            reason.signed_direction,
            reason.specificity_level,
            reason.grouped_reason_type,
        )
        for reason in reasons
    ]


def score_differences(left: Any, right: Any, path: str = "") -> list[str]:
    """Paths where two score dicts differ (floats compared with a tolerance)."""
    if isinstance(left, dict) and isinstance(right, dict):
        out: list[str] = []
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                out.append(f"{path}.{key} (missing on one side)")
            else:
                out.extend(score_differences(left[key], right[key], f"{path}.{key}"))
        return out
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [f"{path} (length {len(left)} vs {len(right)})"]
        out = []
        for index, (a, b) in enumerate(zip(left, right)):
            out.extend(score_differences(a, b, f"{path}[{index}]"))
        return out
    if (
        isinstance(left, (int, float))
        and isinstance(right, (int, float))
        and not isinstance(left, bool)
        and not isinstance(right, bool)
    ):
        return [] if abs(float(left) - float(right)) <= SCORE_TOLERANCE else [path]
    return [] if left == right else [path]


# --------------------------------------------------------------------------- #
# The filter for one explanation
# --------------------------------------------------------------------------- #


def filter_explanation(
    explanation: CanonicalExplanation,
    canonicalizer: ExplanationCanonicalizer,
) -> tuple[CanonicalExplanation, dict[str, Any]]:
    """Keep denial-increasing features and rebuild the principal reasons from them."""
    kept = [
        replace(feature, rank=rank)
        for rank, feature in enumerate(
            (item for item in explanation.referenced_features if item.signed_direction == DENIAL),
            start=1,
        )
    ]
    missing = [
        item.feature_name for item in kept if item.feature_name not in canonicalizer._feature_index
    ]
    if missing:
        raise SystemExit(
            f"error: {explanation.instance_id}: features {missing} are not in the reconstructed "
            f"{explanation.dataset_id}/{explanation.track} bundle (taxonomy mismatch)"
        )
    # The canonicalizer's own construction (cap, temporal grouping per the bundle's
    # reason_grouping_fix). No counterfactual changes are passed: the reasons it would
    # derive from them ("changing X may improve the outcome") favour approval.
    reasons = canonicalizer._build_principal_reasons(
        referenced_features=kept,
        explanation_family=explanation.explanation_family,
        counterfactual_changes=[],
    )
    metadata = copy.deepcopy(explanation.metadata)
    metadata["included_feature_count"] = len(kept)
    metadata["denial_reasons_rule"] = "increases_denial_only"
    rebuilt = replace(
        explanation,
        referenced_features=kept,
        principal_reasons=reasons,
        counterfactual_changes=list(explanation.counterfactual_changes),
        metadata=metadata,
    )
    before = explanation.principal_reasons
    record = {
        "reasons_before": len(before),
        "reasons_after": len(reasons),
        "approval_favouring_dropped": sum(1 for r in before if r.signed_direction == APPROVAL),
        "neutral_dropped": sum(
            1 for r in before if r.signed_direction not in (DENIAL, APPROVAL)
        ),
        "top_reason_approval_favouring": bool(before) and before[0].signed_direction == APPROVAL,
        "features_before": len(explanation.referenced_features),
        "features_after": len(kept),
    }
    return rebuilt, record


# --------------------------------------------------------------------------- #
# Offline reconstruction of the dataset bundles
# --------------------------------------------------------------------------- #


def bundle_corrections_options(payload: dict[str, Any]) -> CorrectionOptions:
    block = common.cslaw_section(payload)
    raw = block.get("corrections")
    if raw is None:
        return V1
    legacy = bool((block.get("config") or {}).get("use_native_contributions_for_ranking"))
    # The recorded form is CorrectionOptions.to_dict() (no "preset" key): every option
    # overrides the v1 preset, which reproduces the run's options exactly.
    return corrections_from_payload(
        {"cslaw": {"corrections": raw, "use_native_contributions_for_ranking": legacy}}
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_config_path(results_path: Path, payload: dict[str, Any], explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit).expanduser()
        return path.resolve() if path.exists() else resolve_repo_path(explicit)
    runtime = results_path.parent / "runtime_config.json"
    if runtime.is_file():
        return runtime
    recorded = ((common.cslaw_section(payload).get("provenance") or {}).get("config") or {}).get(
        "path"
    )
    if recorded and resolve_repo_path(str(recorded)).is_file():
        return resolve_repo_path(str(recorded))
    raise SystemExit(
        "error: no experiment config found; pass --config (the config of the source run)"
    )


def make_bundle_resolver(
    payload: dict[str, Any],
    config_path: Path,
    *,
    allow_inconsistent: bool,
    warnings: list[str],
) -> tuple[Callable[[str, str], DatasetBundle], dict[str, Any]]:
    """(dataset_id, track) -> DatasetBundle, from the metadata files the run used."""
    from legalbench_xai.datasets.registry import build_dataset_loader

    if not config_path.is_file():
        raise SystemExit(f"error: config {config_path} does not exist")
    with config_path.open("r", encoding="utf-8") as handle:
        config_payload = json.load(handle)
    registry = dict(config_payload.get("registry") or {})
    recorded = (common.cslaw_section(payload).get("provenance") or {}).get("metadata") or {}
    info: dict[str, Any] = {"config": str(config_path), "metadata": {}}
    for key in ("feature_taxonomy", "dataset_registry"):
        entry = recorded.get(key) or {}
        if entry.get("path"):
            registry[key] = str(entry["path"])
        if key not in registry:
            raise SystemExit(f"error: no {key} path in the config or the bundle provenance")
        path = resolve_repo_path(str(registry[key]))
        if not path.is_file():
            raise SystemExit(f"error: {key} file {path} (used by the run) does not exist")
        actual = _sha256(path)
        expected = entry.get("sha256")
        status = "not recorded" if not expected else ("match" if expected == actual else "MISMATCH")
        info["metadata"][key] = {"path": str(registry[key]), "sha256": actual, "recorded": status}
        if status == "MISMATCH":
            message = (
                f"{key} {path} differs from the file the run used (sha256 {actual} vs recorded "
                f"{expected})"
            )
            if not allow_inconsistent:
                raise SystemExit(f"error: {message}; pass --allow-inconsistent to continue")
            warnings.append(message)
        elif status == "not recorded":
            warnings.append(f"{key}: no recorded sha256 (pre-v2 bundle); using {path} as configured")
    config_payload["registry"] = registry
    config = ExperimentConfig(path=config_path, payload=config_payload)
    cache: dict[tuple[str, str], DatasetBundle] = {}

    def resolve(dataset_id: str, track: str) -> DatasetBundle:
        key = (dataset_id, track)
        if key not in cache:
            cache[key] = build_dataset_loader(config, dataset_id).build_bundle(track)
        return cache[key]

    return resolve, info


# --------------------------------------------------------------------------- #
# Bundle transformation
# --------------------------------------------------------------------------- #


def _aggregate(row: dict[str, Any], instances: list[dict[str, Any]]) -> dict[str, Any]:
    """runner._aggregate_group over ``instances`` with the row's own group settings."""
    data_source = row.get("data_source") or {}
    return runner._aggregate_group(
        dataset_id=row["dataset_id"],
        track=row["track"],
        model_id=row["model_id"],
        explainer_id=row["explainer_id"],
        data_source_kind=data_source.get("kind"),
        data_source_path=data_source.get("path"),
        total_rows=row.get("total_rows"),
        train_rows=row.get("train_rows"),
        test_rows=row.get("test_rows"),
        adverse_rows=row.get("adverse_rows"),
        evaluated_rows=len(instances),
        model_backend=SimpleNamespace(**(row.get("model_backend") or {})),
        explainer_backend=SimpleNamespace(**(row.get("explainer_backend") or {})),
        adverse_rate=row.get("adverse_rate"),
        split_summary=row.get("evaluation_split"),
        threshold_summary=row.get("threshold_policy"),
        model_performance=row.get("model_performance") or {},
        sampling_summary=row.get("sampling"),
        instance_results=instances,
    )


def _new_counter() -> dict[str, Any]:
    return {
        "instances": 0,
        "produced": 0,
        "reasons_before": 0,
        "reasons_after": 0,
        "approval_favouring_reasons_dropped": 0,
        "neutral_reasons_dropped": 0,
        "instances_with_reasons_before": 0,
        "top_reason_approval_favouring": 0,
        "instances_with_any_approval_favouring_reason": 0,
    }


def _finish_counter(counter: dict[str, Any]) -> dict[str, Any]:
    def ratio(numerator: float, denominator: float) -> float | None:
        return numerator / denominator if denominator else None

    out = dict(counter)
    out["produced_rate"] = ratio(counter["produced"], counter["instances"])
    out["approval_favouring_share"] = ratio(
        counter["approval_favouring_reasons_dropped"], counter["reasons_before"]
    )
    out["top_reason_approval_favouring_share"] = ratio(
        counter["top_reason_approval_favouring"], counter["instances_with_reasons_before"]
    )
    out["mean_reasons_before"] = ratio(counter["reasons_before"], counter["instances"])
    out["mean_reasons_after"] = ratio(counter["reasons_after"], counter["instances"])
    return out


def transform_payload(
    payload: dict[str, Any],
    resolve_bundle: Callable[[str, str], DatasetBundle],
    *,
    source: str,
    allow_inconsistent: bool = False,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """The denial-only payload (the input is not modified)."""
    warnings = warnings if warnings is not None else []
    corrections = bundle_corrections_options(payload)
    scorer = ExplanationScorer(corrections=corrections)
    canonicalizers: dict[tuple[str, str], ExplanationCanonicalizer] = {}

    def canonicalizer_for(dataset_id: str, track: str) -> ExplanationCanonicalizer:
        key = (dataset_id, track)
        if key not in canonicalizers:
            # Same construction as the runner (default principal_reason_limit = 4).
            canonicalizers[key] = ExplanationCanonicalizer(
                resolve_bundle(dataset_id, track), corrections=corrections
            )
        return canonicalizers[key]

    out = copy.deepcopy(payload)
    by_explainer: dict[str, dict[str, Any]] = defaultdict(_new_counter)
    by_dataset: dict[tuple[str, str], dict[str, Any]] = defaultdict(_new_counter)
    inconsistent: list[str] = []
    new_rows: list[dict[str, Any]] = []

    for row in out.get("summary", []):
        instances = row.get("instance_results")
        if instances is None:
            raise SystemExit(
                f"error: summary row {common.group_key(row)} has no instance_results; "
                "denial_reasons.py needs persisted canonical explanations"
            )
        produced: list[dict[str, Any]] = []
        unproduced: list[dict[str, Any]] = []
        for instance in instances:
            raw = instance.get("canonical_explanation")
            if not isinstance(raw, dict):
                raise SystemExit(
                    f"error: instance {instance.get('instance_id')} has no canonical_explanation "
                    "(run with cslaw.persist_canonical_explanations)"
                )
            explanation = canonical_from_dict(raw)
            canonicalizer = canonicalizer_for(explanation.dataset_id, explanation.track)

            # Consistency: the reconstructed canonicalizer and scorer reproduce the
            # persisted reasons and scores of the unfiltered explanation.
            problems = []
            unknown = [
                item.feature_name
                for item in explanation.referenced_features
                if item.feature_name not in canonicalizer._feature_index
            ]
            if unknown:
                problems.append(f"features not in the reconstructed bundle {unknown}")
            else:
                check_reasons = canonicalizer._build_principal_reasons(
                    referenced_features=explanation.referenced_features,
                    explanation_family=explanation.explanation_family,
                    counterfactual_changes=explanation.counterfactual_changes,
                )
                if reasons_signature(check_reasons) != reasons_signature(
                    explanation.principal_reasons
                ):
                    problems.append("principal reasons")
            diffs = score_differences(
                runner._strict_json(scorer.score(explanation)), instance.get("scores") or {}
            )
            if diffs:
                problems.append(f"scores {diffs[:3]}")
            if problems:
                inconsistent.append(
                    f"{instance.get('instance_id')} [{row['explainer_id']}]: {problems}"
                )

            rebuilt, record = filter_explanation(explanation, canonicalizer)
            instance["canonical_explanation"] = runner._strict_json(rebuilt.to_dict())
            instance["scores"] = runner._strict_json(scorer.score(rebuilt))
            is_produced = bool(rebuilt.principal_reasons)
            instance["explanation_produced"] = is_produced
            instance["denial_filter"] = {
                key: record[key]
                for key in (
                    "reasons_before",
                    "reasons_after",
                    "approval_favouring_dropped",
                    "neutral_dropped",
                    "features_before",
                    "features_after",
                )
            }
            (produced if is_produced else unproduced).append(instance)
            for counter in (
                by_explainer[row["explainer_id"]],
                by_dataset[(row["dataset_id"], row["explainer_id"])],
            ):
                counter["instances"] += 1
                counter["produced"] += int(is_produced)
                counter["reasons_before"] += record["reasons_before"]
                counter["reasons_after"] += record["reasons_after"]
                counter["approval_favouring_reasons_dropped"] += record["approval_favouring_dropped"]
                counter["neutral_reasons_dropped"] += record["neutral_dropped"]
                counter["instances_with_reasons_before"] += int(record["reasons_before"] > 0)
                counter["top_reason_approval_favouring"] += int(
                    record["top_reason_approval_favouring"]
                )
                counter["instances_with_any_approval_favouring_reason"] += int(
                    record["approval_favouring_dropped"] > 0
                )

        aggregated = _aggregate(row, produced)
        new_row = dict(row)
        new_row.update(aggregated)
        new_row["cslaw_produced_rate"] = len(produced) / len(instances) if instances else None
        new_row["cslaw_evaluated_rows_all"] = len(instances)
        new_row[UNPRODUCED_KEY] = unproduced
        new_rows.append(new_row)

    if inconsistent:
        message = (
            f"{len(inconsistent)} instances: the reconstructed canonicalizer/scorer does not "
            f"reproduce the persisted reasons or scores, e.g. {inconsistent[:3]}"
        )
        if not allow_inconsistent:
            raise SystemExit(f"error: {message}; pass --allow-inconsistent to continue")
        warnings.append(message)

    # A row without produced explanations keeps mean_* = None (reporting skips it);
    # an explainer without any produced explanation is removed (see the docstring).
    produced_by_explainer: dict[str, int] = defaultdict(int)
    for row in new_rows:
        produced_by_explainer[row["explainer_id"]] += int(row["evaluated_rows"])
    dropped: list[dict[str, Any]] = []
    kept_rows: list[dict[str, Any]] = []
    for row in new_rows:
        if produced_by_explainer[row["explainer_id"]] == 0:
            dropped.append(
                {
                    "dataset_id": row["dataset_id"],
                    "track": row["track"],
                    "model_id": row["model_id"],
                    "explainer_id": row["explainer_id"],
                    "reason": "explainer has no explanation with a denial reason in the bundle",
                    "instances": row[UNPRODUCED_KEY],
                }
            )
        else:
            kept_rows.append(row)

    out["summary"] = kept_rows
    out["overview"] = runner._build_overview(kept_rows)
    out["denial_reasons"] = {
        "source_bundle": source,
        "rule": RULE,
        "corrections": corrections.to_dict(),
        "principal_reason_limit": 4,
        "by_explainer": {key: _finish_counter(value) for key, value in sorted(by_explainer.items())},
        "by_dataset_explainer": [
            {"dataset_id": dataset_id, "explainer_id": explainer_id, **_finish_counter(value)}
            for (dataset_id, explainer_id), value in sorted(by_dataset.items())
        ],
        "configurations_without_produced_explanations": [
            {
                "dataset_id": row["dataset_id"],
                "track": row["track"],
                "model_id": row["model_id"],
                "explainer_id": row["explainer_id"],
            }
            for row in kept_rows
            if not row["evaluated_rows"]
        ],
        "dropped_configurations": dropped,
        "warnings": list(warnings),
        "limitations": [
            "only the explainer's top referenced features are persisted (6 for SHAP, LIME and "
            "the null baselines), so a denial-only notice can have fewer than 4 reasons",
            "faithfulness coverage is recomputed on the kept features only",
        ],
    }
    return runner._strict_json(out)


# --------------------------------------------------------------------------- #
# Markdown summary
# --------------------------------------------------------------------------- #


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.1f}%"


def _num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def summary_markdown(result: dict[str, Any]) -> str:
    info = result["denial_reasons"]
    header = (
        "| explainer | instances | reasons before | approval-favouring reasons | share | "
        "top reason approval-favouring | mean reasons before -> after | produced rate |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|\n"
    )

    def line(name: str, stats: dict[str, Any]) -> str:
        return (
            f"| {name} | {stats['instances']} | {stats['reasons_before']} | "
            f"{stats['approval_favouring_reasons_dropped']} | "
            f"{_pct(stats['approval_favouring_share'])} | "
            f"{_pct(stats['top_reason_approval_favouring_share'])} | "
            f"{_num(stats['mean_reasons_before'])} -> {_num(stats['mean_reasons_after'])} | "
            f"{_pct(stats['produced_rate'])} |\n"
        )

    parts = [
        "# Denial-direction correction of the principal reasons\n\n",
        f"Source bundle: `{info['source_bundle']}`\n\n",
        f"Rule: {info['rule']}.\n\n",
        f"{common.corrections_summary_line(result)}\n\n",
        "\"Approval-favouring\" reasons are persisted principal reasons with "
        "`signed_direction = decreases_denial`; \"top reason\" is the rank-1 reason, over "
        "instances that had at least one reason. Produced rate: share of instances with at "
        "least one denial-increasing reason after the filter; the quality metrics in "
        "`results.json` are computed over those instances only.\n\n",
        "## By explainer\n\n",
        header,
    ]
    for name, stats in info["by_explainer"].items():
        parts.append(line(name, stats))
    parts.append("\n## By dataset and explainer\n")
    datasets = sorted({item["dataset_id"] for item in info["by_dataset_explainer"]})
    for dataset_id in datasets:
        parts.append(f"\n### {dataset_id}\n\n")
        parts.append(header)
        for item in info["by_dataset_explainer"]:
            if item["dataset_id"] == dataset_id:
                parts.append(line(item["explainer_id"], item))
    if info["configurations_without_produced_explanations"]:
        parts.append("\n## Configurations without any produced explanation (mean_* = None)\n\n")
        for item in info["configurations_without_produced_explanations"]:
            parts.append(
                f"- {item['dataset_id']} / {item['track']} / {item['model_id']} / "
                f"{item['explainer_id']}\n"
            )
    if info["dropped_configurations"]:
        parts.append("\n## Dropped configurations (explainer without any produced explanation)\n\n")
        for item in info["dropped_configurations"]:
            parts.append(
                f"- {item['dataset_id']} / {item['track']} / {item['model_id']} / "
                f"{item['explainer_id']}: {item['reason']}\n"
            )
    parts.append("\n## Limitations\n\n")
    for item in info["limitations"]:
        parts.append(f"- {item}\n")
    if info["warnings"]:
        parts.append("\n## Warnings\n\n")
        for item in info["warnings"]:
            parts.append(f"- {item}\n")
    return "".join(parts)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def resolve_results_path(text: str) -> Path:
    path = Path(text).expanduser()
    if path.is_dir():
        path = path / "results.json"
    if not path.is_file():
        raise SystemExit(f"error: results.json not found at {path}")
    return path.resolve()


def prepare_out_dir(out_dir_text: str, bundle_dir: Path, *, force: bool) -> Path:
    out_dir = Path(out_dir_text).expanduser().resolve()
    if out_dir == bundle_dir.resolve():
        raise SystemExit(f"error: --out-dir {out_dir} is the input bundle directory; refusing.")
    if out_dir.exists() and not force:
        raise SystemExit(f"error: --out-dir {out_dir} exists; pass --force to write into it.")
    return out_dir


def follow_up_command(config: str, out_dir: Path) -> str:
    return (
        "PYTHONPATH=benchmark/src python -m legalbench_xai rebuild-artifacts \\\n"
        f"  --config {config} \\\n"
        f"  --results-json {out_dir / 'results.json'} \\\n"
        f"  --output-dir {out_dir}"
    )


def write_outputs(result: dict[str, Any], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results = out_dir / "results.json"
    common.write_json(results, result)
    summary = out_dir / "denial_reasons_summary.md"
    summary.write_text(summary_markdown(result), encoding="utf-8")
    return [results, summary]


def run(args: argparse.Namespace) -> list[Path]:
    results_path = resolve_results_path(args.bundle)
    out_dir = prepare_out_dir(args.out_dir, results_path.parent, force=args.force)
    payload = common.load_results(results_path)
    config_path = resolve_config_path(results_path, payload, args.config)
    warnings: list[str] = []
    resolver, info = make_bundle_resolver(
        payload, config_path, allow_inconsistent=args.allow_inconsistent, warnings=warnings
    )
    result = transform_payload(
        payload,
        resolver,
        source=str(results_path.parent),
        allow_inconsistent=args.allow_inconsistent,
        warnings=warnings,
    )
    result["denial_reasons"]["reconstruction"] = runner._strict_json(info)
    paths = write_outputs(result, out_dir)
    for path in paths:
        print(f"wrote {path}")
    for message in warnings:
        print(f"WARNING: {message}", file=sys.stderr)
    print("\nNext:\n" + follow_up_command(str(config_path), out_dir))
    return paths


# --------------------------------------------------------------------------- #
# Self-test (synthetic data only)
# --------------------------------------------------------------------------- #


def _selftest_bundle() -> DatasetBundle:
    from legalbench_xai.types import FeatureSpec

    def spec(name: str) -> FeatureSpec:
        return FeatureSpec(
            feature_name=name,
            plain_meaning=name.lower(),
            mutability="M",
            verifiability="Y",
            correctability="Y",
            sensitivity="N",
            principal_reason="High",
        )

    names = ["PAY_0", "PAY_2", "PAY_AMT1", "LIMIT_BAL", "AGEISH", "BILL_AMT1", "EDU", "UTIL"]
    return DatasetBundle(
        dataset_id="synthetic_credit",
        display_name="Synthetic credit",
        track="primary",
        task="credit",
        target="default",
        positive_label="approve",
        negative_label="deny",
        included_features=sorted((spec(n) for n in names), key=lambda s: s.feature_name),
        excluded_features=[],
    )


_SELFTEST_REFERENCE = {
    "PAY_0": 1.2,
    "PAY_2": 0.8,
    "PAY_AMT1": -0.6,
    "LIMIT_BAL": -0.9,
    "AGEISH": 0.1,
    "BILL_AMT1": 0.4,
    "EDU": -0.2,
    "UTIL": 0.5,
}


def _selftest_raw(
    instance_id: str,
    weights: dict[str, float],
    explainer: str,
    *,
    changes: list[str] | None = None,
) -> dict[str, Any]:
    ordered = sorted(weights.items(), key=lambda item: -abs(item[1]))
    features = [
        {"feature_name": name, "importance_weight": weight, "rank": rank, "raw_value": 1.0}
        for rank, (name, weight) in enumerate(ordered, start=1)
    ]
    payload: dict[str, Any] = {
        "instance_id": instance_id,
        "model_id": "logistic_regression",
        "explainer_id": explainer,
        "explanation_family": "attribution",
        "prediction": {"score": 0.7, "label": "deny", "threshold": 0.5, "decision_margin": 0.2},
        "features": features,
        "metadata": {
            "faithfulness_evidence": {
                "source": "model_native_common_reference",
                "reference_signed_contributions": dict(_SELFTEST_REFERENCE),
            },
            "stability_evidence": {
                "source": "perturbation_probe",
                "neighborhood_consistency": 0.8,
                "rank_consistency": 0.7,
                "direction_consistency": 0.9,
                "seed_consistency": 1.0,
            },
        },
        "counterfactual_changes": [],
    }
    if changes is not None:
        payload["explanation_family"] = "counterfactual"
        for item in features:
            sign = _SELFTEST_REFERENCE[item["feature_name"]]
            item["signed_direction"] = DENIAL if sign > 0 else APPROVAL
        payload["metadata"]["cf_reason_source"] = (
            "counterfactual_changes" if changes else "no_valid_change"
        )
        payload["counterfactual_changes"] = [
            {"feature_name": name, "from_value": 1.0, "to_value": 0.0, "validity_flip": True}
            for name in changes
        ]
    return payload


def _selftest_payload(corrections: CorrectionOptions) -> dict[str, Any]:
    bundle = _selftest_bundle()
    canonicalizer = ExplanationCanonicalizer(bundle, corrections=corrections)
    scorer = ExplanationScorer(corrections=corrections)
    groups: dict[str, list[dict[str, Any]]] = {
        "shap": [
            # mixed signs; PAY_0 and PAY_2 group into one temporal reason
            _selftest_raw(
                "s1",
                {"PAY_0": 1.2, "LIMIT_BAL": -0.9, "PAY_2": 0.8, "PAY_AMT1": -0.6, "UTIL": 0.5,
                 "EDU": -0.2},
                "shap",
            ),
            # approval-favouring only: no denial reason survives
            _selftest_raw("s2", {"LIMIT_BAL": -0.9, "PAY_AMT1": -0.6, "EDU": -0.2}, "shap"),
            # five denial features: the cap of 4 applies after grouping
            _selftest_raw(
                "s3",
                {"PAY_0": 1.0, "UTIL": 0.9, "BILL_AMT1": 0.8, "AGEISH": 0.7, "PAY_2": 0.6,
                 "LIMIT_BAL": -0.5},
                "shap",
            ),
        ],
        "dice": [
            _selftest_raw("d1", {"PAY_0": 1.2, "LIMIT_BAL": -0.9}, "dice", changes=["PAY_0", "LIMIT_BAL"]),
            # no valid change: originally empty
            _selftest_raw("d2", {}, "dice", changes=[]),
        ],
        # every reason favours approval: the explainer's configuration is dropped
        "lime": [_selftest_raw("l1", {"LIMIT_BAL": -0.4, "EDU": -0.3}, "lime")],
    }
    summary = []
    backend = SimpleNamespace(backend="real", implementation="synthetic", reason="selftest")
    for explainer_id, items in groups.items():
        instance_results = []
        for item in items:
            canonical = canonicalizer.canonicalize(item)
            instance_results.append(
                {
                    "instance_id": item["instance_id"],
                    "prediction_score": 0.7,
                    "predicted_label": "deny",
                    "scores": runner._strict_json(scorer.score(canonical)),
                    "canonical_explanation": runner._strict_json(canonical.to_dict()),
                }
            )
        summary.append(
            runner._aggregate_group(
                dataset_id=bundle.dataset_id,
                track=bundle.track,
                model_id="logistic_regression",
                explainer_id=explainer_id,
                data_source_kind="synthetic",
                data_source_path="-",
                total_rows=100,
                train_rows=70,
                test_rows=30,
                adverse_rows=10,
                evaluated_rows=len(instance_results),
                model_backend=backend,
                explainer_backend=backend,
                adverse_rate=0.33,
                split_summary={"strategy": "stratified", "heldout_used": True},
                threshold_summary={},
                model_performance={},
                sampling_summary={"strategy": "synthetic"},
                instance_results=instance_results,
            )
        )
    payload = {
        "experiment_name": "selftest",
        "overview": runner._build_overview(summary),
        "summary": summary,
        "cslaw": {"config": {}, "corrections": corrections.to_dict()},
    }
    if not corrections.is_v1:
        payload["cslaw"]["corrections_version"] = "v2"
    else:
        del payload["cslaw"]["corrections"]
    return runner._strict_json(payload)


def run_selftest() -> None:
    from legalbench_xai.corrections import V2

    bundle = _selftest_bundle()

    def resolver(dataset_id: str, track: str) -> DatasetBundle:
        assert (dataset_id, track) == (bundle.dataset_id, bundle.track)
        return bundle

    # Round trip: the persisted dict rebuilds the dataclass exactly.
    payload = _selftest_payload(V2)
    for row in payload["summary"]:
        for instance in row["instance_results"]:
            raw = instance["canonical_explanation"]
            assert runner._strict_json(canonical_from_dict(raw).to_dict()) == raw
    assert bundle_corrections_options(payload) == V2

    original = copy.deepcopy(payload)
    result = transform_payload(payload, resolver, source="synthetic")
    assert payload == original, "input payload must not be modified"
    json.dumps(result, allow_nan=False)
    assert result["cslaw"] == payload["cslaw"]

    rows = {row["explainer_id"]: row for row in result["summary"]}
    assert set(rows) == {"shap", "dice"}, sorted(rows)
    dropped = result["denial_reasons"]["dropped_configurations"]
    assert [item["explainer_id"] for item in dropped] == ["lime"], dropped
    assert dropped[0]["instances"][0]["explanation_produced"] is False

    shap = rows["shap"]
    by_id = {item["instance_id"]: item for item in shap["instance_results"]}
    assert set(by_id) == {"s1", "s3"} and shap["instance_ids"] == ["s1", "s3"]
    assert [item["instance_id"] for item in shap[UNPRODUCED_KEY]] == ["s2"]
    assert shap[UNPRODUCED_KEY][0]["denial_filter"]["reasons_after"] == 0
    assert shap["evaluated_rows"] == 2 and shap["cslaw_evaluated_rows_all"] == 3
    assert abs(shap["cslaw_produced_rate"] - 2 / 3) < 1e-12

    original_rows = {row["explainer_id"]: row for row in original["summary"]}
    original_s1 = original_rows["shap"]["instance_results"][0]
    before = original_s1["canonical_explanation"]["principal_reasons"]
    assert any(r["signed_direction"] == APPROVAL for r in before), before

    s1 = by_id["s1"]["canonical_explanation"]
    assert [f["feature_name"] for f in s1["referenced_features"]] == ["PAY_0", "PAY_2", "UTIL"]
    assert [f["rank"] for f in s1["referenced_features"]] == [1, 2, 3]
    assert all(f["signed_direction"] == DENIAL for f in s1["referenced_features"])
    reasons = s1["principal_reasons"]
    assert all(r["signed_direction"] == DENIAL for r in reasons), reasons
    assert reasons[0]["grouped_reason_type"] == "compressed_temporal", reasons
    assert reasons[0]["supporting_features"] == ["PAY_0", "PAY_2"], reasons[0]
    assert [r["rank"] for r in reasons] == [1, 2], reasons
    record = by_id["s1"]["denial_filter"]
    assert record["reasons_before"] == len(before) == 4, before
    assert record["approval_favouring_dropped"] == 2, record
    assert record["reasons_after"] == 2 and by_id["s1"]["explanation_produced"] is True
    assert s1["metadata"]["included_feature_count"] == 3

    s3 = by_id["s3"]["canonical_explanation"]["principal_reasons"]
    assert len(s3) == 4 and all(r["signed_direction"] == DENIAL for r in s3), s3

    # Means over produced explanations only, as runner._aggregate_group computes them.
    for field, key in (
        ("mean_faithfulness", "faithfulness_score"),
        ("mean_stability", "stability_score"),
        ("mean_comprehensibility", "comprehensibility_score"),
        ("mean_css", "contestability_support_score"),
        ("mean_actionability", "unified_actionability_score"),
    ):
        expected = runner._mean([by_id[i]["scores"][key]["score"] for i in ("s1", "s3")])
        assert shap[field] == expected, (field, shap[field], expected)
    # Faithfulness is recomputed on the kept features; stability evidence is unchanged.
    assert (
        by_id["s1"]["scores"]["faithfulness_score"]["reference_feature_coverage"]
        < original_s1["scores"]["faithfulness_score"]["reference_feature_coverage"]
    )
    assert by_id["s1"]["scores"]["stability_score"] == original_s1["scores"]["stability_score"]

    dice = rows["dice"]
    assert [i["instance_id"] for i in dice["instance_results"]] == ["d1"]
    assert [i["instance_id"] for i in dice[UNPRODUCED_KEY]] == ["d2"]
    d1 = dice["instance_results"][0]["canonical_explanation"]
    assert [c["feature_name"] for c in d1["counterfactual_changes"]] == ["PAY_0", "LIMIT_BAL"]
    assert [r["supporting_features"] for r in d1["principal_reasons"]] == [["PAY_0"]], d1
    assert dice["mean_alf"] is not None
    assert dice["cslaw_produced_rate"] == 0.5

    stats = result["denial_reasons"]["by_explainer"]
    assert stats["shap"]["instances"] == 3 and stats["shap"]["produced"] == 2
    assert stats["lime"]["produced"] == 0 and stats["lime"]["approval_favouring_share"] == 1.0
    assert stats["dice"]["produced_rate"] == 0.5
    assert result["overview"] == runner._build_overview(result["summary"])

    # A configuration without produced explanations (explainer produced elsewhere)
    # keeps mean_* = None and stays in the summary.
    partial = copy.deepcopy(original)
    extra = copy.deepcopy(original_rows["shap"])
    extra["track"] = "audit"
    extra["instance_results"] = [copy.deepcopy(original_rows["shap"]["instance_results"][1])]
    extra["instance_results"][0]["canonical_explanation"]["track"] = "primary"
    partial["summary"].append(extra)
    partial_result = transform_payload(partial, resolver, source="synthetic")
    audit_row = next(row for row in partial_result["summary"] if row["track"] == "audit")
    assert audit_row["mean_faithfulness"] is None and audit_row["evaluated_rows"] == 0
    assert audit_row["cslaw_produced_rate"] == 0.0
    assert partial_result["denial_reasons"]["configurations_without_produced_explanations"]

    # A persisted score the reconstruction cannot reproduce stops the run.
    inconsistent = copy.deepcopy(original)
    inconsistent["summary"][0]["instance_results"][0]["scores"]["faithfulness_score"]["score"] += 0.5
    try:
        transform_payload(inconsistent, resolver, source="synthetic")
    except SystemExit as error:
        assert "does not reproduce" in str(error), error
    else:
        raise AssertionError("a persisted score mismatch must stop the run")
    warnings: list[str] = []
    transform_payload(
        inconsistent, resolver, source="synthetic", allow_inconsistent=True, warnings=warnings
    )
    assert warnings and "does not reproduce" in warnings[0], warnings

    # v1 bundle (v1 grouping: PAY_AMT* under "recent late payment history").
    v1_payload = _selftest_payload(V1)
    assert bundle_corrections_options(v1_payload) == V1
    v1_result = transform_payload(v1_payload, resolver, source="synthetic")
    for row in v1_result["summary"]:
        for item in row["instance_results"]:
            assert all(
                r["signed_direction"] == DENIAL
                for r in item["canonical_explanation"]["principal_reasons"]
            )

    markdown = summary_markdown(result)
    assert "| shap | 3 |" in markdown and "### synthetic_credit" in markdown, markdown

    with tempfile.TemporaryDirectory() as tmp:
        bundle_dir = Path(tmp) / "bundle"
        bundle_dir.mkdir()
        for force in (True, False):
            try:
                prepare_out_dir(str(bundle_dir), bundle_dir, force=force)
            except SystemExit:
                pass
            else:
                raise AssertionError("writing into the bundle directory must be refused")
        existing = Path(tmp) / "existing"
        existing.mkdir()
        try:
            prepare_out_dir(str(existing), bundle_dir, force=False)
        except SystemExit:
            pass
        else:
            raise AssertionError("an existing --out-dir must be refused without --force")
        assert prepare_out_dir(str(existing), bundle_dir, force=True) == existing.resolve()
        written = write_outputs(result, Path(tmp) / "out")
        assert all(path.is_file() for path in written)
        assert json.loads(written[0].read_text(encoding="utf-8")) == result
    print("denial_reasons selftest: OK")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--bundle", help="bundle directory (or its results.json)")
    parser.add_argument("--out-dir", help="new output directory")
    parser.add_argument(
        "--config",
        help="experiment config of the source run (default: <bundle>/runtime_config.json, "
        "else the config path recorded in the bundle provenance)",
    )
    parser.add_argument("--force", action="store_true", help="write into an existing --out-dir")
    parser.add_argument(
        "--allow-inconsistent",
        action="store_true",
        help="warn instead of failing when the metadata files differ from the recorded hashes "
        "or the reconstruction does not reproduce the persisted reasons/scores",
    )
    parser.add_argument("--selftest", action="store_true", help="run the synthetic self-test")
    args = parser.parse_args(argv)
    if args.selftest:
        run_selftest()
        return 0
    if not args.bundle or not args.out_dir:
        parser.error("--bundle and --out-dir are required (or --selftest)")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
