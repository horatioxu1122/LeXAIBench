#!/usr/bin/env python3
"""Family-neutral offline rescoring of a LeXAIBench ``results.json`` bundle.

Two modes, usable together or separately:

``--variant V`` rewrites ``instance["scores"][...]["score"]`` for the affected
metric(s) using the stored components and renormalised weights. The components
themselves are kept. It then recomputes the affected summary-row ``mean_*`` fields
the same way ``runner._aggregate_group`` does (``runner._mean``: sum/len in
instance order, None when empty), recomputes ``payload["overview"]`` with
``runner._build_overview``, adds ``payload["rescore"]`` and writes
``<out-dir>/results.json``. Run ``rebuild-artifacts`` on that file afterwards; the
script prints the command. The variants are:

    css_no_dsv                   CSS  = (0.30 V + 0.30 C + 0.15 R) / 0.75
    comp_no_recourse_simplicity  Comp = (0.35 rc + 0.25 fc + 0.20 rt) / 0.80
    fid_no_quality               Fid  = (0.45 cov + 0.35 dir) / 0.80
    all_neutral                  all three
    v2_legal_functions           the legal-function reading of the v2 benchmark:
        CSS_v2 = 0.30 V + 0.30 C + 0.25 D + 0.15 R, with D = dsv_directionality
            (do the reasons state a direction?), the family-neutral part of
            decision-sensitivity visibility; threshold visibility and flip
            validity, which only counterfactual explanations can earn, are
            dropped. D is persisted only by v2 runs
            (persist_score_subcomponents); instances without it fall back to
            css_no_dsv and the script prints a note.
        Comp = comprehensibility without recourse simplicity (as above).
        Fid  = the recorded score when the bundle used the common faithfulness
            reference (v2 correction common_faithfulness_reference; its quality
            weight is the bundle's, 0 in the v2 preset); otherwise fid_no_quality,
            and the note says that v1 faithfulness remains self-referential.
        Unified actionability is unchanged, and payload["rescore"] records that
        comparing it across families is definitional (non-recourse ceiling
        0.35); paper_tables.py (T8) reports actionability and ALF within the
        recourse family only.

The script leaves stability, unified actionability and ALF unchanged. A
non-recourse explanation's unified actionability is capped at 0.35 by
definition (``advice_presence`` = 0.25, ``alf_component`` = 0), and
``paper_tables.py`` (T1) reports that separately rather than redefining it here.

``--sweep-dsv w1,w2,...`` computes, for each decision-sensitivity-visibility
weight w, CSS_w = (0.30 V + 0.30 C + w S + 0.15 R) / (0.75 + w) per explainer
from the summary-row component means. Each mean is weighted by
``evaluated_rows``, as ``reporting._weighted_mean`` does. This is done for the
pooled bundle and for each dataset. The script writes the sweep table and the
weights at which the leader changes, both on the grid and exactly, from the
upper envelope of the linear numerators. Null baselines are never leaders.

Bundles without ``instance_results`` (the older default-credit robustness
bundles) support only ``css_no_dsv`` and the sweep. CSS is linear in its
components, so the mean of the renormalised instance scores equals the
renormalised summary component means up to floating-point rounding. The
summary-level path therefore sets ``mean_css`` from the component means. Every
other variant needs instance-level components, and the script refuses with an
explanatory error if they are missing.

The script never writes into the input bundle directory.
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import random
import re
import sys
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_SRC = REPO_ROOT / "benchmark" / "src"

FID_KEY = "faithfulness_score"
STAB_KEY = "stability_score"
COMP_KEY = "comprehensibility_score"
CSS_KEY = "contestability_support_score"
ALF_KEY = "actionability_legality_filter"
ACT_KEY = "unified_actionability_score"

# (retained component weights, denominator = sum of the retained weights).
WeightSpec = tuple[dict[str, float], float]
CSS_NO_DSV: WeightSpec = (
    {"verifiability": 0.30, "correctability": 0.30, "reason_specificity": 0.15},
    0.75,
)
COMP_NO_RS: WeightSpec = (
    {
        "reason_count_score": 0.35,
        "feature_count_score": 0.25,
        "reason_text_score": 0.20,
    },
    0.80,
)
FID_NO_QUALITY: WeightSpec = (
    {"reference_feature_coverage": 0.45, "directional_agreement": 0.35},
    0.80,
)

# CSS_v2: decision-sensitivity visibility replaced by its directionality part
# (persisted as "dsv_directionality" by v2 runs). The weights already sum to 1.
CSS_V2_LEGAL: WeightSpec = (
    {
        "verifiability": 0.30,
        "correctability": 0.30,
        "dsv_directionality": 0.25,
        "reason_specificity": 0.15,
    },
    1.00,
)

VARIANTS: dict[str, dict[str, WeightSpec]] = {
    "css_no_dsv": {CSS_KEY: CSS_NO_DSV},
    "comp_no_recourse_simplicity": {COMP_KEY: COMP_NO_RS},
    "fid_no_quality": {FID_KEY: FID_NO_QUALITY},
    "all_neutral": {CSS_KEY: CSS_NO_DSV, COMP_KEY: COMP_NO_RS, FID_KEY: FID_NO_QUALITY},
}
V2_LEGAL_FUNCTIONS = "v2_legal_functions"
VARIANT_CHOICES = tuple(sorted([*VARIANTS, V2_LEGAL_FUNCTIONS]))
ACTIONABILITY_NOTE = (
    "Unified actionability is unchanged. Non-recourse explanations cannot exceed "
    f"{common.NON_RECOURSE_ACTIONABILITY_CEILING:g} by definition (advice_presence 0.25, "
    "alf_component 0), so comparing actionability across explanation families is "
    "definitional; compare actionability and ALF within the recourse family "
    "(DiCE vs Wachter) only, as paper_tables.py T8 does."
)
SUMMARY_FIELD_FOR_SCORE = {
    FID_KEY: "mean_faithfulness",
    COMP_KEY: "mean_comprehensibility",
    CSS_KEY: "mean_css",
}
CSS_SUMMARY_COMPONENTS = {
    "verifiability": "mean_css_verifiability",
    "correctability": "mean_css_correctability",
    "decision_sensitivity_visibility": "mean_css_decision_sensitivity_visibility",
    "reason_specificity": "mean_css_reason_specificity",
}
PUBLISHED_DSV_WEIGHT = common.CSS_WEIGHTS["decision_sensitivity_visibility"]
UNCHANGED_SCORE_KEYS = (STAB_KEY, ACT_KEY, ALF_KEY)

# Summary means that _aggregate_group derives as sum/len over ALL instance scores
# (ALF fields filter None and are not re-derived here). Used for a read-only
# consistency check of the input bundle.
CONSISTENCY_FIELDS: dict[str, tuple[str, str]] = {
    "mean_faithfulness": (FID_KEY, "score"),
    "mean_stability": (STAB_KEY, "score"),
    "mean_comprehensibility": (COMP_KEY, "score"),
    "mean_css": (CSS_KEY, "score"),
    "mean_css_verifiability": (CSS_KEY, "verifiability"),
    "mean_css_correctability": (CSS_KEY, "correctability"),
    "mean_css_decision_sensitivity_visibility": (CSS_KEY, "decision_sensitivity_visibility"),
    "mean_css_reason_specificity": (CSS_KEY, "reason_specificity"),
    "mean_actionability": (ACT_KEY, "score"),
}
OVERVIEW_METRIC_FIELDS = (
    "mean_faithfulness",
    "mean_stability",
    "mean_comprehensibility",
    "mean_css",
    "mean_actionability",
    "mean_alf",
)


class RescoreError(ValueError):
    """Raised when a bundle cannot be rescored with the requested variant."""


# --------------------------------------------------------------------------- #
# Pure scoring helpers (also imported by paper_tables.py)
# --------------------------------------------------------------------------- #


def renormalized_score(components: dict[str, Any], spec: WeightSpec) -> float | None:
    """Weighted sum of the retained components divided by their weight sum."""
    weights, denominator = spec
    total = 0.0
    for name, weight in weights.items():
        value = components.get(name)
        if value is None:
            return None
        total += weight * float(value)
    return total / denominator


def css_with_dsv_weight(
    verifiability: float,
    correctability: float,
    dsv: float,
    specificity: float,
    weight: float,
) -> float:
    """CSS with decision-sensitivity weight ``weight`` (0.25 reproduces the scorer)."""
    return (
        0.30 * verifiability
        + 0.30 * correctability
        + weight * dsv
        + 0.15 * specificity
    ) / (0.75 + weight)


def css_v2_score(css: dict[str, Any]) -> tuple[float | None, str | None]:
    """(CSS_v2, basis): basis "directionality" (v2 sub-component present) or
    "css_no_dsv_fallback"; (None, None) when the CSS components are missing."""
    value = renormalized_score(css, CSS_V2_LEGAL)
    if value is not None:
        return value, "directionality"
    value = renormalized_score(css, CSS_NO_DSV)
    return (value, "css_no_dsv_fallback") if value is not None else (None, None)


def instance_metric_values(instance: dict[str, Any]) -> dict[str, Any]:
    """Per-instance metric values, including the family-neutral variants.

    Keys: faithfulness, faithfulness_no_quality, stability, comprehensibility,
    comprehensibility_no_rs, css, css_no_dsv, css_verifiability,
    css_correctability, css_dsv, css_reason_specificity, actionability, alf,
    and (None unless the v2 sub-components are persisted) css_v2,
    css_dsv_directionality, css_dsv_threshold_visibility, css_dsv_flip_validity.
    """
    scores = instance.get("scores") or {}
    fid = scores.get(FID_KEY) or {}
    stab = scores.get(STAB_KEY) or {}
    comp = scores.get(COMP_KEY) or {}
    css = scores.get(CSS_KEY) or {}
    act = scores.get(ACT_KEY) or {}
    alf = scores.get(ALF_KEY) or {}
    return {
        "faithfulness": fid.get("score"),
        "faithfulness_no_quality": renormalized_score(fid, FID_NO_QUALITY),
        "stability": stab.get("score"),
        "comprehensibility": comp.get("score"),
        "comprehensibility_no_rs": renormalized_score(comp, COMP_NO_RS),
        "css": css.get("score"),
        "css_no_dsv": renormalized_score(css, CSS_NO_DSV),
        "css_verifiability": css.get("verifiability"),
        "css_correctability": css.get("correctability"),
        "css_dsv": css.get("decision_sensitivity_visibility"),
        "css_reason_specificity": css.get("reason_specificity"),
        "actionability": act.get("score"),
        "alf": alf.get("score"),
        "css_v2": renormalized_score(css, CSS_V2_LEGAL),
        "css_dsv_directionality": css.get("dsv_directionality"),
        "css_dsv_threshold_visibility": css.get("dsv_threshold_visibility"),
        "css_dsv_flip_validity": css.get("dsv_flip_validity"),
    }


def weighted_mean(
    rows: Iterable[dict[str, Any]],
    value_key: str,
    weight_key: str = "evaluated_rows",
) -> float | None:
    """Mirror of reporting._weighted_mean (skip None values and weights <= 0)."""
    weighted_sum = 0.0
    total_weight = 0.0
    for row in rows:
        value = row.get(value_key)
        weight = float(row.get(weight_key, 0) or 0)
        if value is None or weight <= 0:
            continue
        weighted_sum += float(value) * weight
        total_weight += weight
    if total_weight <= 0:
        return None
    return weighted_sum / total_weight


def rank_real_explainers(scores: dict[str, float | None]) -> dict[str, int]:
    """Rank 1..k by descending score (ties by explainer id); null baselines excluded."""
    eligible = [
        (explainer_id, float(score))
        for explainer_id, score in scores.items()
        if score is not None and not common.is_null_baseline(explainer_id)
    ]
    ordered = sorted(eligible, key=lambda item: (-item[1], item[0]))
    return {explainer_id: rank for rank, (explainer_id, _score) in enumerate(ordered, start=1)}


def sanitize_json(value: Any) -> tuple[Any, int]:
    """Replace NaN/inf floats by None so the payload is strict JSON."""
    if isinstance(value, float):
        if math.isfinite(value):
            return value, 0
        return None, 1
    if isinstance(value, dict):
        replaced = 0
        clean: dict[Any, Any] = {}
        for key, item in value.items():
            clean[key], count = sanitize_json(item)
            replaced += count
        return clean, replaced
    if isinstance(value, list):
        replaced = 0
        items: list[Any] = []
        for item in value:
            clean_item, count = sanitize_json(item)
            items.append(clean_item)
            replaced += count
        return items, replaced
    return value, 0


# --------------------------------------------------------------------------- #
# Variant rescoring
# --------------------------------------------------------------------------- #


def _values_equal(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return float(left) == float(right)


def check_summary_consistency(
    row: dict[str, Any],
    instances: list[dict[str, Any]],
    mean_fn: Callable[[list[float]], float | None],
) -> list[dict[str, Any]]:
    """Compare stored summary means with means recomputed from instance scores."""
    mismatches: list[dict[str, Any]] = []
    for field, (score_key, component) in CONSISTENCY_FIELDS.items():
        if field not in row:
            continue
        try:
            values = [instance["scores"][score_key][component] for instance in instances]
        except (KeyError, TypeError):
            continue
        recomputed = mean_fn(values)
        stored = row.get(field)
        if not _values_equal(stored, recomputed):
            diff = None
            if stored is not None and recomputed is not None:
                diff = abs(float(stored) - float(recomputed))
            mismatches.append(
                {
                    "group": "|".join(common.group_key(row)),
                    "field": field,
                    "stored": stored,
                    "recomputed": recomputed,
                    "abs_diff": diff,
                }
            )
    return mismatches


def variant_specs(
    payload: dict[str, Any],
    variant: str,
) -> tuple[dict[str, tuple[WeightSpec, ...]], dict[str, Any]]:
    """({score_key: candidate specs}, variant info) for a bundle.

    Each instance uses the first candidate spec whose components are all present
    (renormalized_score is None otherwise). The four original variants have one
    candidate per score; v2_legal_functions falls back from CSS_v2 to css_no_dsv
    and keeps the recorded faithfulness of a common-reference (v2) bundle.
    """
    if variant in VARIANTS:
        return {score_key: (spec,) for score_key, spec in VARIANTS[variant].items()}, {}
    if variant != V2_LEGAL_FUNCTIONS:
        raise RescoreError(f"unknown variant {variant!r}; choose from {list(VARIANT_CHOICES)}")
    common_reference = bool(common.correction_flag(payload, "common_faithfulness_reference"))
    specs: dict[str, tuple[WeightSpec, ...]] = {
        CSS_KEY: (CSS_V2_LEGAL, CSS_NO_DSV),
        COMP_KEY: (COMP_NO_RS,),
    }
    if not common_reference:
        specs[FID_KEY] = (FID_NO_QUALITY,)
    info = {
        "corrections_version": common.corrections_version(payload),
        "faithfulness_handling": (
            "recorded_common_reference" if common_reference else "fid_no_quality"
        ),
        "faithfulness_quality_weight": common.correction_flag(
            payload, "faithfulness_quality_weight"
        ),
        "actionability_comparison_scope": "recourse_family_only",
        "actionability_note": ACTIONABILITY_NOTE,
    }
    return specs, info


def rescore_payload(
    payload: dict[str, Any],
    variant: str,
    *,
    mean_fn: Callable[[list[float]], float | None],
    overview_builder: Callable[[list[dict[str, Any]]], dict[str, Any]],
    source_results: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (rescored deep copy, report). The input payload is not modified."""
    specs, info = variant_specs(payload, variant)
    summary = payload.get("summary")
    if not isinstance(summary, list) or not summary:
        raise RescoreError("payload has no summary rows")

    summary_only = [row for row in summary if row.get("instance_results") is None]
    if summary_only and set(specs) != {CSS_KEY}:
        raise RescoreError(
            f"{len(summary_only)} of {len(summary)} summary rows have no instance_results "
            f"(older bundle). Variant {variant!r} needs instance-level score components; "
            "only 'css_no_dsv' (and --sweep-dsv) can be derived from the summary-row "
            "CSS component means. Use a bundle with instance_results for this variant."
        )

    rescored = copy.deepcopy(payload)
    report: dict[str, Any] = {
        "instance_level_rows": 0,
        "summary_level_rows": 0,
        "instances_rescored": 0,
        "consistency_mismatches": [],
        "fallback_instances": {score_key: 0 for score_key in specs},
        "unscored_instances": {score_key: 0 for score_key in specs},
    }
    for row in rescored["summary"]:
        instances = row.get("instance_results")
        if instances is None:
            components = {
                name: row.get(field) for name, field in CSS_SUMMARY_COMPONENTS.items()
            }
            row["mean_css"] = renormalized_score(components, CSS_NO_DSV)
            report["summary_level_rows"] += 1
            continue
        report["consistency_mismatches"].extend(
            check_summary_consistency(row, instances, mean_fn)
        )
        for instance in instances:
            for score_key, candidates in specs.items():
                block = instance["scores"][score_key]
                value: float | None = None
                for position, spec in enumerate(candidates):
                    value = renormalized_score(block, spec)
                    if value is not None:
                        if position > 0:
                            report["fallback_instances"][score_key] += 1
                        break
                if value is None:
                    report["unscored_instances"][score_key] += 1
                block["score"] = value
        report["instances_rescored"] += len(instances)
        for score_key in specs:
            row[SUMMARY_FIELD_FOR_SCORE[score_key]] = mean_fn(
                [instance["scores"][score_key]["score"] for instance in instances]
            )
        report["instance_level_rows"] += 1

    rescored["overview"] = overview_builder(rescored["summary"])
    source_name = payload.get("experiment_name", "unknown_experiment")
    rescored["experiment_name"] = f"{source_name}__rescored_{variant}"
    rescored["rescore"] = {
        "variant": variant,
        "weights": {
            score_key: {"components": dict(candidates[0][0]), "denominator": candidates[0][1]}
            for score_key, candidates in specs.items()
        },
        "unchanged_scores": list(UNCHANGED_SCORE_KEYS),
        "source_results": source_results,
        "source_experiment_name": source_name,
        "instance_level_rows": report["instance_level_rows"],
        "summary_level_rows": report["summary_level_rows"],
        "instances_rescored": report["instances_rescored"],
        "previous_rescore": payload.get("rescore"),
        "note": (
            "Instance-level 'score' fields of the listed metrics were recomputed from "
            "their stored components with renormalised weights; components are "
            "unchanged. Summary-level rows (no instance_results) had mean_css set from "
            "the CSS component means. Stability, unified actionability and ALF are "
            "unchanged."
        ),
    }
    if info:
        record = rescored["rescore"]
        record["unchanged_scores"] = [
            score_key
            for score_key in (FID_KEY, STAB_KEY, COMP_KEY, CSS_KEY, ACT_KEY, ALF_KEY)
            if score_key not in specs
        ]
        record["fallback_weights"] = {
            score_key: [
                {"components": dict(spec[0]), "denominator": spec[1]} for spec in candidates[1:]
            ]
            for score_key, candidates in specs.items()
            if len(candidates) > 1
        }
        record["fallback_instances"] = dict(report["fallback_instances"])
        record["unscored_instances"] = dict(report["unscored_instances"])
        record.update(info)
        record["notes"] = variant_notes(info, report)
        record["note"] = (
            "Legal-function rescoring: CSS_v2 = 0.30V + 0.30C + 0.25 dsv_directionality + "
            "0.15R (css_no_dsv where the sub-component is not persisted), comprehensibility "
            "without recourse simplicity, faithfulness as described in 'notes'. Stability, "
            "unified actionability and ALF are unchanged; see 'actionability_note'."
        )
    return rescored, report


def variant_notes(info: dict[str, Any], report: dict[str, Any]) -> list[str]:
    """Human-readable notes for a v2_legal_functions rescoring."""
    notes: list[str] = []
    fallback = int(report.get("fallback_instances", {}).get(CSS_KEY, 0) or 0)
    if fallback:
        notes.append(
            f"{fallback} of {report['instances_rescored']} instances have no persisted "
            "dsv_directionality (bundle not produced with persist_score_subcomponents, e.g. a v1 "
            "bundle); their CSS_v2 falls back to css_no_dsv = (0.30V + 0.30C + 0.15R)/0.75."
        )
    if info.get("faithfulness_handling") == "fid_no_quality":
        notes.append(
            "The bundle did not use the common faithfulness reference, so faithfulness is "
            "fid_no_quality = (0.45 cov + 0.35 dir)/0.80. Dropping the quality term does not "
            "remove the self-reference: v1 faithfulness of SHAP/LIME/leave-one-out is scored "
            "against the explainer's own values and Anchors against a heuristic proxy restricted "
            "to its own features. Use common_faithfulness.py or a v2 bundle for a common reference."
        )
    else:
        notes.append(
            "Faithfulness is the recorded score: the bundle scored every explainer against the "
            "model's native contributions (common reference) with quality weight "
            f"{info.get('faithfulness_quality_weight')}."
        )
    notes.append(str(info.get("actionability_note", ACTIONABILITY_NOTE)))
    return notes


def explainer_level_means(
    summary_rows: list[dict[str, Any]],
    fields: Iterable[str],
) -> dict[str, dict[str, float | None]]:
    fields = list(fields)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        grouped[row["explainer_id"]].append(row)
    return {
        explainer_id: {field: weighted_mean(rows, field) for field in fields}
        for explainer_id, rows in sorted(grouped.items())
    }


def before_after_rows(
    original_summary: list[dict[str, Any]],
    rescored_summary: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Pooled (evaluated_rows-weighted) explainer means before and after rescoring."""
    before = explainer_level_means(original_summary, OVERVIEW_METRIC_FIELDS)
    after = explainer_level_means(rescored_summary, OVERVIEW_METRIC_FIELDS)
    rows: list[dict[str, Any]] = []
    for field in OVERVIEW_METRIC_FIELDS:
        before_ranks = rank_real_explainers({e: v[field] for e, v in before.items()})
        after_ranks = rank_real_explainers({e: v[field] for e, v in after.items()})
        for explainer_id in sorted(before):
            old = before[explainer_id][field]
            new = after.get(explainer_id, {}).get(field)
            rows.append(
                {
                    "metric": field.removeprefix("mean_"),
                    "explainer_id": explainer_id,
                    "is_null_baseline": common.is_null_baseline(explainer_id),
                    "before": old,
                    "after": new,
                    "delta": (new - old) if old is not None and new is not None else None,
                    "before_rank": before_ranks.get(explainer_id),
                    "after_rank": after_ranks.get(explainer_id),
                    "changed_metric": old != new,
                }
            )
    return rows


# --------------------------------------------------------------------------- #
# Decision-sensitivity weight sweep
# --------------------------------------------------------------------------- #


def component_means_by_subset(
    summary_rows: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, float | None]]]:
    """{subset: {explainer: {component/mean_css: weighted mean}}}; subset 'pooled' + datasets."""
    fields = list(CSS_SUMMARY_COMPONENTS.values()) + ["mean_css"]
    subsets: dict[str, list[dict[str, Any]]] = {"pooled": list(summary_rows)}
    for row in summary_rows:
        subsets.setdefault(row["dataset_id"], []).append(row)
    result: dict[str, dict[str, dict[str, float | None]]] = {}
    for subset, rows in subsets.items():
        result[subset] = explainer_level_means(rows, fields)
    return result


def _sweep_lines(
    explainer_means: dict[str, dict[str, float | None]],
) -> dict[str, tuple[float, float, float, float]]:
    lines: dict[str, tuple[float, float, float, float]] = {}
    for explainer_id, means in explainer_means.items():
        values = [means.get(field) for field in CSS_SUMMARY_COMPONENTS.values()]
        if any(value is None for value in values):
            continue
        verifiability, correctability, dsv, specificity = (float(v) for v in values)
        lines[explainer_id] = (verifiability, correctability, dsv, specificity)
    return lines


def leader_at_weight(
    lines: dict[str, tuple[float, float, float, float]],
    weight: float,
) -> tuple[str | None, float | None]:
    scores = {
        explainer_id: css_with_dsv_weight(*components, weight)
        for explainer_id, components in lines.items()
    }
    ranks = rank_real_explainers(scores)
    leader = next((e for e, rank in ranks.items() if rank == 1), None)
    return leader, (scores[leader] if leader is not None else None)


def exact_leader_breakpoints(
    lines: dict[str, tuple[float, float, float, float]],
    max_weight: float,
) -> list[dict[str, Any]]:
    """Weights in (0, max_weight] where the CSS_w leader changes (real explainers only).

    The ranking at weight w is that of the numerator A_e + w * S_e (the
    denominator 0.75 + w is shared), so the leader follows the upper envelope of
    the lines. A strictly increasing slope at each step guarantees termination.
    """
    real = {
        explainer_id: (
            0.30 * components[0] + 0.30 * components[1] + 0.15 * components[3],
            components[2],
        )
        for explainer_id, components in lines.items()
        if not common.is_null_baseline(explainer_id)
    }
    if not real:
        return []
    leader = sorted(real, key=lambda e: (-real[e][0], -real[e][1], e))[0]
    current = 0.0
    breakpoints: list[dict[str, Any]] = []
    while True:
        intercept_l, slope_l = real[leader]
        best: tuple[float, float, str] | None = None
        for explainer_id, (intercept, slope) in real.items():
            if explainer_id == leader or slope <= slope_l:
                continue
            # The current leader dominates at ``current``, so crossing >= current up
            # to rounding; clamp instead of skipping near-ties.
            crossing = (intercept_l - intercept) / (slope - slope_l)
            candidate = (max(crossing, current), -slope, explainer_id)
            if best is None or candidate < best:
                best = candidate
        if best is None or best[0] > max_weight:
            return breakpoints
        breakpoints.append(
            {"weight": best[0], "from_leader": leader, "to_leader": best[2]}
        )
        leader = best[2]
        current = best[0]


def dsv_weight_sweep(
    summary_rows: list[dict[str, Any]],
    weights: list[float],
    published_weight: float = PUBLISHED_DSV_WEIGHT,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (sweep rows, leader-change rows) for 'pooled' and each dataset."""
    grid = sorted(set(float(weight) for weight in weights))
    sweep_rows: list[dict[str, Any]] = []
    leader_rows: list[dict[str, Any]] = []
    for subset, explainer_means in component_means_by_subset(summary_rows).items():
        lines = _sweep_lines(explainer_means)
        if not lines:
            continue
        grid_leaders: list[tuple[float, str | None]] = []
        for weight in grid:
            scores = {
                explainer_id: css_with_dsv_weight(*components, weight)
                for explainer_id, components in lines.items()
            }
            ranks = rank_real_explainers(scores)
            leader = next((e for e, rank in ranks.items() if rank == 1), None)
            grid_leaders.append((weight, leader))
            for explainer_id in sorted(scores):
                sweep_rows.append(
                    {
                        "subset": subset,
                        "explainer_id": explainer_id,
                        "is_null_baseline": common.is_null_baseline(explainer_id),
                        "weight": weight,
                        "mean_css": scores[explainer_id],
                        "rank": ranks.get(explainer_id),
                        "leader": explainer_id == leader,
                    }
                )
        published_leader, published_score = leader_at_weight(lines, published_weight)
        zero_leader, _zero_score = leader_at_weight(lines, 0.0)
        max_weight = max(grid + [1.0])
        breakpoints = exact_leader_breakpoints(lines, max_weight)
        below = [bp["weight"] for bp in breakpoints if bp["weight"] < published_weight]
        above = [bp["weight"] for bp in breakpoints if bp["weight"] > published_weight]
        grid_changes = [
            f"{prev_w:g}->{next_w:g}: {prev_l}->{next_l}"
            for (prev_w, prev_l), (next_w, next_l) in zip(grid_leaders, grid_leaders[1:])
            if prev_l != next_l
        ]
        leader_rows.append(
            {
                "subset": subset,
                "n_explainers": len(lines),
                "published_weight": published_weight,
                "published_leader": published_leader,
                "published_leader_css": published_score,
                "leader_at_weight_0": zero_leader,
                "grid_leaders": ";".join(f"{w:g}:{leader}" for w, leader in grid_leaders),
                "grid_leader_changes": "; ".join(grid_changes),
                "exact_breakpoints": "; ".join(
                    f"{bp['weight']:.6f}: {bp['from_leader']}->{bp['to_leader']}"
                    for bp in breakpoints
                ),
                "nearest_change_below_published": max(below) if below else None,
                "nearest_change_above_published": min(above) if above else None,
                "max_weight_considered": max_weight,
            }
        )
    return sweep_rows, leader_rows


# --------------------------------------------------------------------------- #
# Markdown + I/O
# --------------------------------------------------------------------------- #


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("" if value is None else value) for key, value in row.items()})


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def rescore_markdown(
    variant: str,
    results_path: Path,
    out_path: Path,
    report: dict[str, Any],
    comparison_rows: list[dict[str, Any]],
    replaced: int,
    follow_up: str,
    notes: Iterable[str] = (),
) -> str:
    lines = [
        f"# Rescore: `{variant}`",
        "",
        f"- Source: `{results_path}`",
        f"- Output: `{out_path}`",
        f"- Instance-level rows: {report['instance_level_rows']}; summary-level rows "
        f"(no instance_results, css from component means): {report['summary_level_rows']}; "
        f"instances rescored: {report['instances_rescored']}",
        f"- Input consistency (stored summary means vs sum/len of instance scores): "
        f"{len(report['consistency_mismatches'])} mismatching field(s)",
    ]
    if replaced:
        lines.append(f"- Non-finite values replaced by null for strict JSON: {replaced}")
    notes = list(notes)
    if notes:
        lines += ["", "## Notes", ""] + [f"- {note}" for note in notes]
    lines += ["", "## Pooled explainer means (evaluated_rows-weighted; null baselines unranked)", ""]
    lines.append("| metric | explainer | before | after | rank before | rank after |")
    lines.append("|---|---|---|---|---|---|")
    for row in comparison_rows:
        if not row["changed_metric"]:
            continue
        lines.append(
            f"| {row['metric']} | {row['explainer_id']} | {_fmt(row['before'])} | "
            f"{_fmt(row['after'])} | {_fmt(row['before_rank'])} | {_fmt(row['after_rank'])} |"
        )
    lines += ["", "## Follow-up", "", "```sh", follow_up, "```", ""]
    return "\n".join(lines)


def sweep_markdown(results_path: Path, leader_rows: list[dict[str, Any]], weights: list[float]) -> str:
    lines = [
        "# CSS decision-sensitivity-visibility weight sweep",
        "",
        f"- Source: `{results_path}`",
        f"- Grid: {', '.join(f'{w:g}' for w in sorted(set(weights)))} "
        f"(published weight {PUBLISHED_DSV_WEIGHT:g}; CSS_w = (0.30V + 0.30C + wS + 0.15R)/(0.75 + w))",
        "- Means are evaluated_rows-weighted summary component means; null baselines are never leaders.",
        "",
        "| subset | leader at w=0.25 | leader at w=0 | nearest change below 0.25 | nearest change above 0.25 | exact breakpoints |",
        "|---|---|---|---|---|---|",
    ]
    for row in leader_rows:
        lines.append(
            f"| {row['subset']} | {row['published_leader']} | {row['leader_at_weight_0']} | "
            f"{_fmt(row['nearest_change_below_published'])} | "
            f"{_fmt(row['nearest_change_above_published'])} | {row['exact_breakpoints'] or 'none'} |"
        )
    lines.append("")
    return "\n".join(lines)


def resolve_results_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if path.is_dir():
        path = path / "results.json"
    if not path.is_file():
        raise SystemExit(f"error: results.json not found at {path}")
    return path.resolve()


def import_runner_helpers() -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Return (runner._mean, runner._build_overview) from the benchmark package."""
    def _load() -> tuple[Callable[..., Any], Callable[..., Any]]:
        from legalbench_xai.runner import _build_overview, _mean  # type: ignore

        return _mean, _build_overview

    try:
        return _load()
    except Exception as first_error:  # noqa: BLE001
        last_error: BaseException = first_error
        if BENCHMARK_SRC.is_dir() and str(BENCHMARK_SRC) not in sys.path:
            sys.path.insert(0, str(BENCHMARK_SRC))
            try:
                return _load()
            except Exception as error:  # noqa: BLE001
                last_error = error
        raise SystemExit(
            "error: could not import legalbench_xai.runner (needed for _mean and "
            "_build_overview so the rescored summary/overview match the pipeline "
            f"exactly): {last_error!r}\n"
            "Run from the repository root with the benchmark environment, e.g.\n"
            "  PYTHONPATH=benchmark/src python benchmark/cslaw/rescore.py ..."
        ) from last_error


def guess_config(results_path: Path, payload: dict[str, Any], explicit: str | None) -> str:
    if explicit:
        return explicit
    runtime = results_path.parent / "runtime_config.json"
    if runtime.is_file():
        return str(runtime)
    name = str(payload.get("experiment_name", ""))
    stem = re.sub(r"^legalbench_xai_", "", name)
    candidate = REPO_ROOT / "benchmark" / "configs" / f"experiment_{stem}.json"
    if stem and candidate.is_file():
        return str(candidate)
    return "<CONFIG_USED_FOR_THE_SOURCE_RUN>"


def follow_up_command(config: str, out_dir: Path) -> str:
    return (
        "PYTHONPATH=benchmark/src python -m legalbench_xai rebuild-artifacts \\\n"
        f"  --config {config} \\\n"
        f"  --results-json {out_dir / 'results.json'} \\\n"
        f"  --output-dir {out_dir}"
    )


def parse_weights(text: str) -> list[float]:
    weights: list[float] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        value = float(part)
        if not math.isfinite(value) or value < 0:
            raise SystemExit(f"error: sweep weights must be finite and >= 0, got {part!r}")
        weights.append(value)
    if not weights:
        raise SystemExit("error: --sweep-dsv needs at least one weight")
    return sorted(set(weights))


def _refuse_existing_output(target: Path, overwrite: bool) -> None:
    if not target.exists() or overwrite:
        return
    try:
        prior = common.load_results(target)
    except Exception:  # noqa: BLE001
        prior = {}
    if "rescore" not in prior:
        raise SystemExit(
            f"error: {target} exists and is not a previous rescore output; refusing to "
            "overwrite it (pass --overwrite to force)."
        )


# --------------------------------------------------------------------------- #
# Synthetic payload + self-test (no real data)
# --------------------------------------------------------------------------- #


def make_synthetic_payload(
    *,
    datasets: tuple[str, ...] = ("d_alpha", "d_beta"),
    tracks: tuple[str, ...] = ("primary", "audit"),
    models: tuple[str, ...] = ("logistic_regression", "ebm"),
    explainers: tuple[str, ...] = ("anchors", "dice", "shap"),
    n_instances: int = 3,
    seed: int = 7,
    summary_only: bool = False,
) -> dict[str, Any]:
    """Small deterministic payload shaped like runner.run_demo_benchmark output."""
    rng = random.Random(seed)
    summary: list[dict[str, Any]] = []
    for dataset_id in datasets:
        for track in tracks:
            for model_id in models:
                for explainer_id in explainers:
                    instances = [
                        _synthetic_instance(rng, dataset_id, track, model_id, explainer_id, index)
                        for index in range(n_instances)
                    ]
                    summary.append(
                        _synthetic_summary_row(
                            dataset_id, track, model_id, explainer_id, instances, summary_only
                        )
                    )
    return {
        "experiment_name": "legalbench_xai_synthetic_selftest",
        "generated_at_utc": "2026-01-01T00:00:00+00:00",
        "capabilities": {},
        "overview": {},
        "summary": summary,
    }


def _synthetic_instance(
    rng: random.Random,
    dataset_id: str,
    track: str,
    model_id: str,
    explainer_id: str,
    index: int,
) -> dict[str, Any]:
    recourse = explainer_id in common.RECOURSE_EXPLAINERS
    position = index * 7 + len(dataset_id)
    cov, direction = rng.uniform(0.5, 1.0), rng.uniform(0.5, 1.0)
    quality = 0.97 if explainer_id == "shap" else rng.uniform(0.4, 1.0)
    rc, fc, rt = rng.choice([1.0, 0.8, 0.6]), rng.uniform(0.2, 1.0), rng.uniform(0.6, 1.0)
    rs = rng.choice([1.0, 0.8]) if recourse else 1.0
    v, c = rng.uniform(0.6, 1.0), rng.uniform(0.4, 1.0)
    s = rng.choice([0.7, 1.0]) if recourse else rng.choice([0.55, 0.64])
    r = rng.choice([0.5, 0.75, 1.0])
    if recourse:
        feas, leg, mini = rng.uniform(0.5, 1.0), rng.uniform(0.5, 1.0), rng.uniform(0.5, 1.0)
        alf_score = feas * leg * mini
        alf = {
            "applicability": "applicable",
            "feasibility": feas,
            "legality": leg,
            "minimality": mini,
            "invalid_recourse_rate": 0.0,
            "hard_failures": [],
            "score": alf_score,
        }
        advice = 1.0
    else:
        alf = {
            "applicability": "not_applicable",
            "feasibility": None,
            "legality": None,
            "minimality": None,
            "invalid_recourse_rate": None,
            "hard_failures": [],
            "score": None,
        }
        alf_score = 0.0
        advice = 0.25
    stab = [rng.uniform(0.3, 1.0) for _ in range(4)]
    backend = {"backend": "real", "implementation": f"{model_id}_impl", "reason": ""}
    return {
        "instance_id": f"{dataset_id}-{track}-{model_id}-row{position}",
        "prediction_score": 0.5 + 0.1 * index,
        "predicted_label": "Bad",
        "model_backend": dict(backend),
        "explainer_backend": {"backend": "real", "implementation": f"{explainer_id}_impl", "reason": ""},
        "subgroup_attributes": {},
        "scores": {
            FID_KEY: {
                "evidence_source": "structural_proxy" if recourse else f"real_{explainer_id}",
                "reference_feature_coverage": cov,
                "directional_agreement": direction,
                "explanation_quality_signal": quality,
                "score": 0.45 * cov + 0.35 * direction + 0.20 * quality,
            },
            STAB_KEY: {
                "evidence_source": "perturbation_probe",
                "neighborhood_consistency": stab[0],
                "rank_consistency": stab[1],
                "direction_consistency": stab[2],
                "seed_consistency": stab[3],
                "score": 0.35 * stab[0] + 0.25 * stab[1] + 0.25 * stab[2] + 0.15 * stab[3],
            },
            COMP_KEY: {
                "reason_count_score": rc,
                "feature_count_score": fc,
                "reason_text_score": rt,
                "recourse_simplicity_score": rs,
                "score": 0.35 * rc + 0.25 * fc + 0.20 * rt + 0.20 * rs,
            },
            CSS_KEY: {
                "verifiability": v,
                "correctability": c,
                "decision_sensitivity_visibility": s,
                "reason_specificity": r,
                "score": 0.30 * v + 0.30 * c + 0.25 * s + 0.15 * r,
            },
            ALF_KEY: alf,
            ACT_KEY: {
                "advice_presence": advice,
                "correction_support": c,
                "alf_component": alf_score,
                "score": 0.20 * advice + 0.50 * alf_score + 0.30 * c,
            },
        },
    }


def _synthetic_summary_row(
    dataset_id: str,
    track: str,
    model_id: str,
    explainer_id: str,
    instances: list[dict[str, Any]],
    summary_only: bool,
) -> dict[str, Any]:
    def mean_of(score_key: str, component: str, skip_none: bool = False) -> float | None:
        values = [instance["scores"][score_key][component] for instance in instances]
        if skip_none:
            values = [value for value in values if value is not None]
        return sum(values) / len(values) if values else None

    row: dict[str, Any] = {
        "dataset_id": dataset_id,
        "track": track,
        "model_id": model_id,
        "explainer_id": explainer_id,
        "data_source": {"kind": "synthetic", "path": "synthetic.csv"},
        "model_backend": {"backend": "real", "implementation": f"{model_id}_impl", "reason": ""},
        "explainer_backend": {"backend": "real", "implementation": f"{explainer_id}_impl", "reason": ""},
        "total_rows": 100,
        "evaluated_rows": len(instances),
        "adverse_rate": 0.3,
        "evaluation_split": {"strategy": "heldout_stratified", "heldout_used": True},
        "sampling": {"strategy": "decision_margin_stratified", "strata_used": []},
        "mean_faithfulness": mean_of(FID_KEY, "score"),
        "mean_stability": mean_of(STAB_KEY, "score"),
        "mean_comprehensibility": mean_of(COMP_KEY, "score"),
        "mean_css": mean_of(CSS_KEY, "score"),
        "mean_css_verifiability": mean_of(CSS_KEY, "verifiability"),
        "mean_css_correctability": mean_of(CSS_KEY, "correctability"),
        "mean_css_decision_sensitivity_visibility": mean_of(CSS_KEY, "decision_sensitivity_visibility"),
        "mean_css_reason_specificity": mean_of(CSS_KEY, "reason_specificity"),
        "mean_actionability": mean_of(ACT_KEY, "score"),
        "mean_alf": mean_of(ALF_KEY, "score", skip_none=True),
        "mean_invalid_recourse_rate": mean_of(ALF_KEY, "invalid_recourse_rate", skip_none=True),
        "faithfulness_evidence_sources": {
            instances[0]["scores"][FID_KEY]["evidence_source"]: len(instances)
        },
        "stability_evidence_sources": {"perturbation_probe": len(instances)},
        "instance_ids": [instance["instance_id"] for instance in instances],
    }
    if not summary_only:
        row["instance_results"] = instances
    return row


def run_selftest() -> None:
    def plain_mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    def stub_overview(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {"total_groups": len(rows)}

    payload = make_synthetic_payload()
    original = copy.deepcopy(payload)
    rescored, report = rescore_payload(
        payload,
        "all_neutral",
        mean_fn=plain_mean,
        overview_builder=stub_overview,
        source_results="synthetic",
    )
    assert payload == original, "input payload must not be modified"
    assert report["consistency_mismatches"] == [], report["consistency_mismatches"][:3]
    assert rescored["overview"] == {"total_groups": len(payload["summary"])}
    assert rescored["rescore"]["variant"] == "all_neutral"
    for row in rescored["summary"]:
        for instance in row["instance_results"]:
            css = instance["scores"][CSS_KEY]
            expected = (
                0.30 * css["verifiability"]
                + 0.30 * css["correctability"]
                + 0.15 * css["reason_specificity"]
            ) / 0.75
            assert abs(css["score"] - expected) < 1e-12
            comp = instance["scores"][COMP_KEY]
            expected = (
                0.35 * comp["reason_count_score"]
                + 0.25 * comp["feature_count_score"]
                + 0.20 * comp["reason_text_score"]
            ) / 0.80
            assert abs(comp["score"] - expected) < 1e-12
            fid = instance["scores"][FID_KEY]
            expected = (
                0.45 * fid["reference_feature_coverage"] + 0.35 * fid["directional_agreement"]
            ) / 0.80
            assert abs(fid["score"] - expected) < 1e-12
        for field, score_key in (
            ("mean_css", CSS_KEY),
            ("mean_comprehensibility", COMP_KEY),
            ("mean_faithfulness", FID_KEY),
        ):
            assert row[field] == plain_mean(
                [instance["scores"][score_key]["score"] for instance in row["instance_results"]]
            )
    for old_row, new_row in zip(original["summary"], rescored["summary"]):
        for field in ("mean_stability", "mean_actionability", "mean_alf", "mean_css_verifiability"):
            assert old_row[field] == new_row[field], field

    summary_only = make_synthetic_payload(summary_only=True)
    try:
        rescore_payload(
            summary_only,
            "fid_no_quality",
            mean_fn=plain_mean,
            overview_builder=stub_overview,
            source_results="x",
        )
    except RescoreError:
        pass
    else:
        raise AssertionError("summary-only bundles must refuse fid_no_quality")
    css_only, css_report = rescore_payload(
        summary_only,
        "css_no_dsv",
        mean_fn=plain_mean,
        overview_builder=stub_overview,
        source_results="x",
    )
    assert css_report["summary_level_rows"] == len(summary_only["summary"])
    instance_level, _ = rescore_payload(
        make_synthetic_payload(),
        "css_no_dsv",
        mean_fn=plain_mean,
        overview_builder=stub_overview,
        source_results="x",
    )
    for left, right in zip(css_only["summary"], instance_level["summary"]):
        assert abs(left["mean_css"] - right["mean_css"]) < 1e-12

    sweep_rows, leader_rows = dsv_weight_sweep(payload["summary"], [0.0, 0.25, 1.0])
    pooled_published = {
        row["explainer_id"]: row["mean_css"]
        for row in sweep_rows
        if row["subset"] == "pooled" and row["weight"] == 0.25
    }
    pooled_observed = explainer_level_means(payload["summary"], ["mean_css"])
    for explainer_id, value in pooled_published.items():
        assert abs(value - pooled_observed[explainer_id]["mean_css"]) < 1e-12
    assert {row["subset"] for row in leader_rows} == {"pooled", "d_alpha", "d_beta"}

    lines = {"flat": (1.0, 1.0, 0.0, 1.0), "steep": (0.6, 0.6, 1.0, 0.6)}
    breakpoints = exact_leader_breakpoints(lines, 5.0)
    assert len(breakpoints) == 1 and breakpoints[0]["to_leader"] == "steep", breakpoints
    crossing = breakpoints[0]["weight"]
    assert leader_at_weight(lines, crossing - 1e-6)[0] == "flat"
    assert leader_at_weight(lines, crossing + 1e-6)[0] == "steep"

    clean, replaced = sanitize_json({"a": [1.0, float("nan")], "b": float("inf")})
    assert replaced == 2 and clean == {"a": [1.0, None], "b": None}

    # v2_legal_functions on a v2-shaped bundle: persisted directionality, common
    # faithfulness reference (recorded faithfulness kept).
    v2_payload = make_synthetic_payload(seed=9)
    for row in v2_payload["summary"]:
        for index, instance in enumerate(row["instance_results"]):
            instance["scores"][CSS_KEY]["dsv_directionality"] = float(index % 2)
    v2_payload["cslaw"] = {
        "corrections": dict(common.CORRECTIONS_V2),
        "corrections_version": "v2",
    }
    v2_rescored, v2_report = rescore_payload(
        v2_payload,
        V2_LEGAL_FUNCTIONS,
        mean_fn=plain_mean,
        overview_builder=stub_overview,
        source_results="synthetic",
    )
    record = v2_rescored["rescore"]
    assert v2_report["fallback_instances"][CSS_KEY] == 0, v2_report["fallback_instances"]
    assert record["faithfulness_handling"] == "recorded_common_reference", record
    assert record["actionability_comparison_scope"] == "recourse_family_only"
    assert FID_KEY in record["unchanged_scores"] and ACT_KEY in record["unchanged_scores"]
    for old_row, new_row in zip(v2_payload["summary"], v2_rescored["summary"]):
        assert new_row["mean_faithfulness"] == old_row["mean_faithfulness"]
        for old, new in zip(old_row["instance_results"], new_row["instance_results"]):
            css = new["scores"][CSS_KEY]
            expected = (
                0.30 * css["verifiability"]
                + 0.30 * css["correctability"]
                + 0.25 * css["dsv_directionality"]
                + 0.15 * css["reason_specificity"]
            )
            assert abs(css["score"] - expected) < 1e-12
            assert new["scores"][FID_KEY]["score"] == old["scores"][FID_KEY]["score"]
            assert css_v2_score(css)[1] == "directionality"
    # The same variant on a v1 bundle: css_no_dsv fallback and fid_no_quality.
    v1_rescored, v1_report = rescore_payload(
        make_synthetic_payload(seed=9),
        V2_LEGAL_FUNCTIONS,
        mean_fn=plain_mean,
        overview_builder=stub_overview,
        source_results="synthetic",
    )
    assert v1_report["fallback_instances"][CSS_KEY] == v1_report["instances_rescored"]
    assert v1_rescored["rescore"]["faithfulness_handling"] == "fid_no_quality"
    assert len(v1_rescored["rescore"]["notes"]) == 3, v1_rescored["rescore"]["notes"]
    for row in v1_rescored["summary"]:
        for instance in row["instance_results"]:
            css = instance["scores"][CSS_KEY]
            assert abs(css["score"] - renormalized_score(css, CSS_NO_DSV)) < 1e-12
            fid = instance["scores"][FID_KEY]
            expected = (
                0.45 * fid["reference_feature_coverage"] + 0.35 * fid["directional_agreement"]
            ) / 0.80
            assert abs(fid["score"] - expected) < 1e-12
    print("rescore.py selftest: OK")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Family-neutral offline rescoring and CSS decision-sensitivity weight "
            "sweep for a LeXAIBench results.json bundle."
        ),
    )
    parser.add_argument("--results", help="Bundle directory or results.json path (input, never modified).")
    parser.add_argument(
        "--variant",
        choices=VARIANT_CHOICES,
        help=(
            "Rescoring variant to apply. v2_legal_functions: CSS_v2 with decision-sensitivity "
            "directionality only, comprehensibility without recourse simplicity, common-reference "
            "(or quality-free) faithfulness; see the module docstring."
        ),
    )
    parser.add_argument(
        "--sweep-dsv",
        help="Comma-separated decision-sensitivity weights, e.g. 0,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5",
    )
    parser.add_argument("--out-dir", help="Output directory (must differ from the input bundle directory).")
    parser.add_argument("--config", help="Config path printed in the follow-up rebuild-artifacts command.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing <out-dir>/results.json that is not a previous rescore output.",
    )
    parser.add_argument("--selftest", action="store_true", help="Run the synthetic self-test and exit.")
    args = parser.parse_args()

    if args.selftest:
        run_selftest()
        return
    if not args.results or not args.out_dir:
        parser.error("--results and --out-dir are required")
    if not args.variant and not args.sweep_dsv:
        parser.error("pass --variant and/or --sweep-dsv")

    results_path = resolve_results_path(args.results)
    out_dir = Path(args.out_dir).expanduser().resolve()
    if out_dir == results_path.parent:
        raise SystemExit(
            f"error: --out-dir {out_dir} is the input bundle directory; refusing to write there."
        )
    payload = common.load_results(results_path)
    written: list[Path] = []

    if args.variant:
        target = out_dir / "results.json"
        _refuse_existing_output(target, args.overwrite)
        mean_fn, overview_builder = import_runner_helpers()
        try:
            rescored, report = rescore_payload(
                payload,
                args.variant,
                mean_fn=mean_fn,
                overview_builder=overview_builder,
                source_results=str(results_path),
            )
        except RescoreError as error:
            raise SystemExit(f"error: {error}") from error
        rescored, replaced = sanitize_json(rescored)
        if replaced:
            rescored["rescore"]["non_finite_values_replaced_with_null"] = replaced
        out_dir.mkdir(parents=True, exist_ok=True)
        common.write_json(target, rescored)
        written.append(target)

        comparison = before_after_rows(payload["summary"], rescored["summary"])
        comparison_path = out_dir / "rescore_by_explainer.csv"
        write_rows_csv(comparison_path, comparison)
        written.append(comparison_path)

        if report["consistency_mismatches"]:
            mismatch_path = out_dir / "rescore_input_consistency_mismatches.csv"
            write_rows_csv(mismatch_path, report["consistency_mismatches"])
            written.append(mismatch_path)
            print(
                f"warning: {len(report['consistency_mismatches'])} stored summary mean(s) do not "
                f"equal sum/len of the instance scores; see {mismatch_path}"
            )

        follow_up = follow_up_command(guess_config(results_path, payload, args.config), out_dir)
        notes = list(rescored["rescore"].get("notes") or [])
        for note in notes:
            print(f"note: {note}")
        markdown_path = out_dir / "rescore_summary.md"
        markdown_path.write_text(
            rescore_markdown(
                args.variant, results_path, target, report, comparison, replaced, follow_up, notes
            ),
            encoding="utf-8",
        )
        written.append(markdown_path)
        print("Next, rebuild the analysis artifacts from the repository root with the SAME config")
        print("that produced the source bundle (seed and bootstrap settings drive the CIs):")
        print(follow_up)

    if args.sweep_dsv:
        weights = parse_weights(args.sweep_dsv)
        sweep_rows, leader_rows = dsv_weight_sweep(payload["summary"], weights)
        out_dir.mkdir(parents=True, exist_ok=True)
        sweep_path = out_dir / "css_dsv_sweep.csv"
        leaders_path = out_dir / "css_dsv_sweep_leaders.csv"
        sweep_md_path = out_dir / "css_dsv_sweep.md"
        write_rows_csv(sweep_path, sweep_rows)
        write_rows_csv(leaders_path, leader_rows)
        sweep_md_path.write_text(sweep_markdown(results_path, leader_rows, weights), encoding="utf-8")
        written += [sweep_path, leaders_path, sweep_md_path]
        for row in leader_rows:
            print(
                f"[sweep] {row['subset']}: leader at w={PUBLISHED_DSV_WEIGHT:g} is "
                f"{row['published_leader']}; exact breakpoints: {row['exact_breakpoints'] or 'none'}"
            )

    print("Wrote:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
