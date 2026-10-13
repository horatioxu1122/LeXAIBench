#!/usr/bin/env python3
"""CS&Law revision tables (T1-T8) computed offline from LeXAIBench result bundles.

Usage (from the repository root; no model or explainer is run):

    python benchmark/cslaw/paper_tables.py \
        --bundle credit=benchmark/outputs/four_dataset_real_highbudget_heldout_refresh \
        --bundle crossdomain=benchmark/outputs/cross_domain_strongest_real_full_cluster \
        [--bundle nullbaselines=<bundle with random_reasons/global_ranking>] \
        --robustness-root benchmark/outputs \
        [--rescored-root <dir with rescore.py outputs>] \
        --out-dir <new dir> --bootstrap 1000 --seed 42

Outputs in ``--out-dir``:

    T1_by_construction.csv/.md  structural facts checked against the data
    T2_css_decomposition.csv    V, C, S, R, CSS, CSS_no_dsv by explainer x dataset (+ pooled)
    T3_pairwise_recount.csv     configuration-level pairwise wins split by category
    T3_pairwise_pairs.csv       the same comparisons per explainer pair and category
    T4_per_domain.csv           dataset x explainer means with hierarchical bootstrap CIs
    T5_correlations.csv         Spearman correlations (instance and configuration level)
    T6_robustness.csv           leader under each robustness check vs baseline leader
    T7_null_baselines.csv       null baselines next to real explainers on the same instances
    T8_legal_functions.csv      v2 legal-function table: V, C, directionality, R, CSS_v2,
                                comprehensibility without recourse simplicity, faithfulness
                                (common reference in v2 bundles), stability, ALF and
                                actionability within the recourse family, by explainer x
                                dataset (+ pooled) with hierarchical bootstrap CIs
    tables.md                   short summary of every table

v2 bundles (payload["cslaw"]["corrections"]) are detected per bundle. T1 reports
their correction status (common faithfulness reference, quality weight, the
counterfactual reason source counts from the canonical metadata); in T3 a
faithfulness comparison between two explainers that were both scored against the
common reference is "open" instead of "not_comparable_evidence"; T8 uses the
persisted decision-sensitivity directionality (v1 bundles fall back to
css_no_dsv and to quality-free, still self-referential faithfulness, and say so).

Null baselines (``common.NULL_BASELINES``) are never ranked or reported as leaders.
A bundle named ``nullbaselines`` feeds T7 (next to the reference bundle's real
explainers) and T1, and adds labelled rows (bundle=nullbaselines) to the T2, T4
and T5 CSVs for reference; those rows are never ranked, never used for leaders
or pooled correlations, and get no headline lines in tables.md. It is excluded
from T3 and T6. Bundles without ``instance_results`` (older default-credit
bundles) degrade to summary-level means, and every affected table says so.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import math
import re
import sys
import tempfile
import warnings
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import regb_reference as regb  # noqa: E402
import rescore  # noqa: E402

try:
    import numpy as np
    from scipy.stats import spearmanr
except ImportError as import_error:  # pragma: no cover - environment guard
    raise SystemExit(
        f"error: paper_tables.py needs numpy and scipy ({import_error!r})"
    ) from import_error


MAIN_METRICS = ("faithfulness", "stability", "comprehensibility", "css", "actionability", "alf")
NEUTRAL_METRICS = ("faithfulness_no_quality", "comprehensibility_no_rs", "css_no_dsv")
CSS_COMPONENT_METRICS = (
    "css_verifiability",
    "css_correctability",
    "css_dsv",
    "css_reason_specificity",
)
T3_METRICS = (
    "faithfulness",
    "stability",
    "comprehensibility",
    "comprehensibility_no_rs",
    "css",
    "css_no_dsv",
    "actionability",
    "alf",
)
T4_METRICS = MAIN_METRICS + NEUTRAL_METRICS
T7_METRICS = (
    "faithfulness",
    "faithfulness_no_quality",
    "stability",
    "comprehensibility",
    "comprehensibility_no_rs",
    "css",
    "css_no_dsv",
) + CSS_COMPONENT_METRICS + ("actionability",)
SUMMARY_FIELDS = {
    "faithfulness": "mean_faithfulness",
    "stability": "mean_stability",
    "comprehensibility": "mean_comprehensibility",
    "css": "mean_css",
    "actionability": "mean_actionability",
    "alf": "mean_alf",
    "css_verifiability": "mean_css_verifiability",
    "css_correctability": "mean_css_correctability",
    "css_dsv": "mean_css_decision_sensitivity_visibility",
    "css_reason_specificity": "mean_css_reason_specificity",
}
CATEGORIES = ("decided_by_construction", "formula_advantaged", "not_comparable_evidence", "open")
CORRELATION_PAIRS = (
    ("faithfulness", "css"),
    ("faithfulness", "css_no_dsv"),
    ("faithfulness_no_quality", "css_no_dsv"),
)
VARIANT_AFFECTED_METRICS = {
    "css_no_dsv": ("css",),
    "comp_no_recourse_simplicity": ("comprehensibility",),
    "fid_no_quality": ("faithfulness",),
    "all_neutral": ("css", "comprehensibility", "faithfulness"),
    "v2_legal_functions": ("css", "comprehensibility", "faithfulness"),
}
T8_METRICS = (
    "css_verifiability",
    "css_correctability",
    "css_dsv_directionality",
    "css_reason_specificity",
    "css_v2",
    "comprehensibility_no_rs",
    "faithfulness_legal",
    "faithfulness_legal_ebm_only",
    "stability",
    "alf_recourse_only",
    "actionability_recourse_only",
)
T8_RECOURSE_ONLY = ("alf_recourse_only", "actionability_recourse_only")
# Faithfulness columns carry by-construction flags (regb_reference.by_construction_flags):
# under the common native reference, SHAP on logistic regression / XGBoost decomposes the
# reference itself. The EBM-only column restricts to the model where no explainer does.
T8_FAITHFULNESS = ("faithfulness_legal", "faithfulness_legal_ebm_only")
NULL_BASELINE_BUNDLE = "nullbaselines"
POOLED = "__pooled__"
ALL_BUNDLES = "__all__"
SATURATION = 1.0 - 1e-9
FOUR_DATASET_BUDGET8 = "legalbench_xai_four_dataset_real"
# (needle in directory or experiment name, check family, preferred baseline needle)
ROBUSTNESS_PATTERNS: tuple[tuple[str, str, str | None], ...] = (
    ("threshold040", "decision_threshold", "threshold050"),
    ("threshold060", "decision_threshold", "threshold050"),
    ("threshold050", "replication_vs_main_bundle", None),
    ("highbudget24", "denied_instance_budget", "threshold050"),
    ("targetrate", "matched_operating_point", FOUR_DATASET_BUDGET8),
    ("six_explainers", "explainer_set", FOUR_DATASET_BUDGET8),
)
T6_COLUMNS = (
    "check_family",
    "check_name",
    "source",
    "scope",
    "metric",
    "baseline_leader",
    "baseline_score",
    "check_leader",
    "check_score",
    "changed",
    "n_comparisons",
    "n_changed",
    "note",
)


# --------------------------------------------------------------------------- #
# Bundle loading and configuration records
# --------------------------------------------------------------------------- #


@dataclass
class Bundle:
    name: str
    results_path: Path
    payload: dict[str, Any]

    @property
    def directory(self) -> Path:
        return self.results_path.parent

    @property
    def summary(self) -> list[dict[str, Any]]:
        return list(self.payload.get("summary") or [])

    @property
    def has_instances(self) -> bool:
        rows = self.summary
        return bool(rows) and all(row.get("instance_results") is not None for row in rows)

    @property
    def datasets(self) -> list[str]:
        return sorted({row["dataset_id"] for row in self.summary})

    @property
    def experiment_name(self) -> str:
        return str(self.payload.get("experiment_name", ""))

    @property
    def corrections_version(self) -> str:
        return common.corrections_version(self.payload)

    @property
    def common_reference(self) -> bool:
        """True when the run scored every explainer's faithfulness on native contributions."""
        return bool(common.correction_flag(self.payload, "common_faithfulness_reference"))


@dataclass
class ConfigRecord:
    bundle: str
    dataset_id: str
    track: str
    model_id: str
    explainer_id: str
    evaluated_rows: int
    metrics: dict[str, float | None]
    evidence_kind: str
    instances: list[dict[str, Any]] | None
    # True when this configuration's faithfulness used the common native reference
    # (v2 correction common_faithfulness_reference, or every evidence source says so).
    common_reference: bool = False
    faithfulness_sources: dict[str, int] | None = None
    # Bundle-level inputs of regb.by_construction_flags (native ranking switch,
    # v2 DiCE/Wachter reasons from changes), so T3 can mark faithfulness pairs whose
    # agreement with the common reference is built in.
    native_ranking: bool = False
    cf_reasons_from_changes: bool = False

    def by_construction(self) -> list[str]:
        return regb.by_construction_flags(
            self.explainer_id,
            self.model_id,
            native_ranking=self.native_ranking,
            cf_reasons_from_changes=self.cf_reasons_from_changes,
        )[0]

    @property
    def family(self) -> str:
        return family_of(self.explainer_id)

    @property
    def config_key(self) -> tuple[str, str, str, str]:
        return (self.bundle, self.dataset_id, self.track, self.model_id)


def family_of(explainer_id: str) -> str:
    return common.EXPLAINER_FAMILY.get(explainer_id, "unknown")


def is_recourse(explainer_id: str) -> bool:
    return explainer_id in common.RECOURSE_EXPLAINERS


def faithfulness_evidence_kind(row: dict[str, Any]) -> str:
    """'direct', 'proxy' or 'mixed' faithfulness evidence for a summary row."""
    sources = row.get("faithfulness_evidence_sources") or {}
    if not sources:
        # Same fallback as reporting._evidence_source_summary for older bundles.
        return "proxy" if is_recourse(row["explainer_id"]) else "direct"
    total = sum(int(count) for count in sources.values())
    proxy = int(sources.get("structural_proxy", 0))
    if proxy == 0:
        return "direct"
    if proxy >= total:
        return "proxy"
    return "mixed"


def instance_record(instance: dict[str, Any]) -> dict[str, Any]:
    values = rescore.instance_metric_values(instance)
    scores = instance.get("scores") or {}
    fid = scores.get(rescore.FID_KEY) or {}
    comp = scores.get(rescore.COMP_KEY) or {}
    alf = scores.get(rescore.ALF_KEY) or {}
    act = scores.get(rescore.ACT_KEY) or {}
    canonical = instance.get("canonical_explanation")
    metadata = canonical.get("metadata") if isinstance(canonical, dict) else None
    values.update(
        {
            "instance_id": instance.get("instance_id"),
            "quality_signal": fid.get("explanation_quality_signal"),
            "quality_weight": fid.get("faithfulness_quality_weight"),
            "faithfulness_source": fid.get("evidence_source"),
            "recourse_simplicity": comp.get("recourse_simplicity_score"),
            "alf_applicability": alf.get("applicability"),
            "advice_presence": act.get("advice_presence"),
            "alf_component": act.get("alf_component"),
            "cf_reason_source": (
                metadata.get("cf_reason_source") if isinstance(metadata, dict) else None
            ),
            # Share of cited DiCE/Wachter features the model does not push toward
            # denial (v2 reasons from changes; None when not recorded).
            "cf_nondenial_reason_share": (
                (metadata.get("cf_reason_details") or {}).get("nondenial_reason_share")
                if isinstance(metadata, dict)
                and isinstance(metadata.get("cf_reason_details"), dict)
                else None
            ),
            "has_canonical": isinstance(canonical, dict),
        }
    )
    return values


def _group_nondenial_shares(records: list[Any]) -> dict[str, list[float]]:
    shares: dict[str, list[float]] = defaultdict(list)
    for record in records:
        if not is_recourse(record.explainer_id):
            continue
        for instance in record.instances or []:
            value = instance.get("cf_nondenial_reason_share")
            if isinstance(value, (int, float)):
                shares[record.explainer_id].append(float(value))
    return shares


def row_uses_common_reference(row: dict[str, Any], bundle_flag: bool) -> bool:
    if bundle_flag:
        return True
    sources = row.get("faithfulness_evidence_sources") or {}
    return bool(sources) and set(sources) == {common.COMMON_REFERENCE_SOURCE}


def build_config_records(bundle: Bundle) -> list[ConfigRecord]:
    records: list[ConfigRecord] = []
    bundle_common_reference = bundle.common_reference
    bundle_native_ranking = regb.native_ranking_enabled(bundle.payload)
    bundle_cf_from_changes = regb.cf_reasons_from_changes(bundle.payload)
    for row in bundle.summary:
        raw_instances = row.get("instance_results")
        instances = (
            [instance_record(instance) for instance in raw_instances]
            if raw_instances is not None
            else None
        )
        metrics: dict[str, float | None] = {
            metric: row.get(field) for metric, field in SUMMARY_FIELDS.items()
        }
        if instances is not None:
            for metric in NEUTRAL_METRICS:
                metrics[metric] = common.mean([instance[metric] for instance in instances])
        else:
            metrics["css_no_dsv"] = rescore.renormalized_score(
                {name: row.get(field) for name, field in rescore.CSS_SUMMARY_COMPONENTS.items()},
                rescore.CSS_NO_DSV,
            )
            metrics["faithfulness_no_quality"] = None
            metrics["comprehensibility_no_rs"] = None
        records.append(
            ConfigRecord(
                bundle=bundle.name,
                dataset_id=row["dataset_id"],
                track=row["track"],
                model_id=row["model_id"],
                explainer_id=row["explainer_id"],
                evaluated_rows=int(row.get("evaluated_rows", 0) or 0),
                metrics=metrics,
                evidence_kind=faithfulness_evidence_kind(row),
                instances=instances,
                common_reference=row_uses_common_reference(row, bundle_common_reference),
                faithfulness_sources={
                    str(source): int(count)
                    for source, count in (row.get("faithfulness_evidence_sources") or {}).items()
                },
                native_ranking=bundle_native_ranking,
                cf_reasons_from_changes=bundle_cf_from_changes,
            )
        )
    return records


def weighted_record_mean(records: list[ConfigRecord], metric: str) -> float | None:
    """evaluated_rows-weighted mean of configuration means (reporting._weighted_mean)."""
    weighted_sum = 0.0
    total_weight = 0.0
    for record in records:
        value = record.metrics.get(metric)
        weight = float(record.evaluated_rows)
        if value is None or weight <= 0:
            continue
        weighted_sum += float(value) * weight
        total_weight += weight
    return weighted_sum / total_weight if total_weight > 0 else None


def leader_of(scores: dict[str, float | None]) -> tuple[str | None, float | None]:
    ranks = rescore.rank_real_explainers(scores)
    leader = next((explainer_id for explainer_id, rank in ranks.items() if rank == 1), None)
    if leader is None:
        return None, None
    return leader, float(scores[leader])  # type: ignore[arg-type]


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return "n/a"
        return f"{value:.{digits}f}"
    return str(value)


# --------------------------------------------------------------------------- #
# T1: structural facts
# --------------------------------------------------------------------------- #


def distribution_stats(values: list[float]) -> dict[str, Any]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return {"n": 0}
    array = np.asarray(clean, dtype=float)
    counts = Counter(round(value, 6) for value in clean)
    top = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:5]
    return {
        "n": len(clean),
        "mean": float(array.mean()),
        "min": float(array.min()),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)),
        "max": float(array.max()),
        "n_unique": len(counts),
        "share_eq_1": sum(1 for value in clean if value >= SATURATION) / len(clean),
        "top_values": ";".join(f"{value:g}:{count / len(clean):.3f}" for value, count in top),
    }


def build_t1(bundle: Bundle, records: list[ConfigRecord]) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    headlines: list[str] = []

    def add(
        fact: str,
        group_1: str,
        group_2: str,
        statistic: str,
        value: Any,
        n: int | None = None,
        note: str = "",
    ) -> None:
        rows.append(
            {
                "bundle": bundle.name,
                "fact": fact,
                "group_1": group_1,
                "group_2": group_2,
                "statistic": statistic,
                "value": value,
                "n": n,
                "note": note,
            }
        )

    ceiling = common.NON_RECOURSE_ACTIONABILITY_CEILING
    with_instances = [record for record in records if record.instances is not None]
    summary_only = [record for record in records if record.instances is None]
    if summary_only:
        add(
            "data_availability",
            "summary_rows_without_instance_results",
            "",
            "count",
            len(summary_only),
            len(records),
            "instance-level facts skip these rows; summary-level fallbacks are marked",
        )
        headlines.append(
            f"{len(summary_only)} of {len(records)} summary rows have no instance_results; "
            "instance-level facts use the remaining rows (summary-level fallbacks are marked)."
        )

    # 0. Correction status (v2 bundles record payload["cslaw"]["corrections"]).
    t1_correction_status(bundle, records, add, headlines)

    # 1. Non-recourse unified actionability ceiling.
    non_recourse: dict[str, list[dict[str, Any]]] = defaultdict(list)
    recourse: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in with_instances:
        for instance in record.instances or []:
            target = recourse if instance.get("alf_applicability") == "applicable" else non_recourse
            target[record.explainer_id].append(instance)
    pooled_non_recourse = [item for items in non_recourse.values() for item in items]
    groups = sorted(non_recourse.items())
    if pooled_non_recourse:
        groups.append(("ALL", pooled_non_recourse))
    for explainer_id, items in groups:
        values = [float(item["actionability"]) for item in items if item.get("actionability") is not None]
        if not values:
            continue
        family = "ALL" if explainer_id == "ALL" else family_of(explainer_id)
        above = sum(1 for value in values if value > ceiling + 1e-12)
        note = f"explanations without counterfactual changes; ceiling={ceiling:g}"
        add("non_recourse_actionability_ceiling", explainer_id, family, "ceiling", ceiling, len(values), note)
        add("non_recourse_actionability_ceiling", explainer_id, family, "max_unified_actionability", max(values), len(values), note)
        add("non_recourse_actionability_ceiling", explainer_id, family, "n_above_ceiling", above, len(values), note)
        alf_components = [float(item["alf_component"]) for item in items if item.get("alf_component") is not None]
        if alf_components:
            add("non_recourse_actionability_ceiling", explainer_id, family, "max_alf_component", max(alf_components), len(alf_components))
        if explainer_id == "ALL":
            headlines.append(
                f"Non-recourse explanations (n={len(values)}): max unified actionability "
                f"{max(values):.4f} vs ceiling {ceiling:g}; {above} above the ceiling."
            )
    for explainer_id, items in sorted(recourse.items()):
        values = [float(item["actionability"]) for item in items if item.get("actionability") is not None]
        if values:
            stats = distribution_stats(values)
            for statistic in ("min", "median", "max"):
                add("recourse_actionability_range", explainer_id, family_of(explainer_id), statistic, stats[statistic], stats["n"])
    fallback: dict[str, list[float]] = defaultdict(list)
    for record in summary_only:
        value = record.metrics.get("actionability")
        if not is_recourse(record.explainer_id) and value is not None:
            fallback[record.explainer_id].append(float(value))
    for explainer_id, values in sorted(fallback.items()):
        add(
            "non_recourse_actionability_ceiling",
            explainer_id,
            family_of(explainer_id),
            "max_config_mean_actionability",
            max(values),
            len(values),
            "summary-level fallback: configuration means of non-recourse explainers",
        )

    # 2-3. Decision-sensitivity visibility and recourse simplicity by family.
    for fact, metric in (
        ("decision_sensitivity_visibility_by_family", "css_dsv"),
        ("recourse_simplicity_by_family", "recourse_simplicity"),
    ):
        by_family: dict[str, list[float]] = defaultdict(list)
        by_explainer: dict[tuple[str, str], list[float]] = defaultdict(list)
        for record in with_instances:
            for instance in record.instances or []:
                value = instance.get(metric)
                if value is None:
                    continue
                by_family[record.family].append(float(value))
                by_explainer[(record.family, record.explainer_id)].append(float(value))
        grouped = [((family, "ALL"), values) for family, values in sorted(by_family.items())]
        grouped += [(key, values) for key, values in sorted(by_explainer.items())]
        for (group_1, group_2), values in grouped:
            stats = distribution_stats(values)
            for statistic, value in stats.items():
                if statistic != "n":
                    add(fact, group_1, group_2, statistic, value, stats["n"])
        if by_family:
            if metric == "css_dsv":
                summary = ", ".join(
                    f"{family} median {float(np.median(values)):.3f}"
                    for family, values in sorted(by_family.items())
                )
                headlines.append(f"Decision-sensitivity visibility by family: {summary}.")
            else:
                summary = ", ".join(
                    f"{family} {sum(1 for v in values if v >= SATURATION) / len(values):.1%}"
                    for family, values in sorted(by_family.items())
                )
                headlines.append(f"Share of explanations with recourse_simplicity = 1.0: {summary}.")

    # 4. ALF applicability.
    applicability: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for record in with_instances:
        for instance in record.instances or []:
            applicability[record.explainer_id][0] += 1
            if instance.get("alf_applicability") == "applicable":
                applicability[record.explainer_id][1] += 1
    for explainer_id, (total, applicable) in sorted(applicability.items()):
        add("alf_applicability_by_explainer", explainer_id, family_of(explainer_id), "n_applicable", applicable, total)
        add(
            "alf_applicability_by_explainer",
            explainer_id,
            family_of(explainer_id),
            "share_applicable",
            applicable / total if total else None,
            total,
        )
    if applicability:
        headlines.append(
            "ALF applicable share: "
            + ", ".join(
                f"{explainer_id} {applicable / total:.1%}"
                for explainer_id, (total, applicable) in sorted(applicability.items())
                if total
            )
            + "."
        )

    # 5. Faithfulness quality signal unique values.
    quality: dict[tuple[str, str], list[float]] = defaultdict(list)
    quality_sources: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for record in with_instances:
        for instance in record.instances or []:
            value = instance.get("quality_signal")
            if value is not None:
                quality[(record.explainer_id, record.model_id)].append(float(value))
            quality_sources[(record.explainer_id, record.model_id)][str(instance.get("faithfulness_source"))] += 1
    constant_cells: list[str] = []
    for (explainer_id, model_id), values in sorted(quality.items()):
        unique = sorted({round(value, 6) for value in values})
        rendered = ";".join(f"{value:g}" for value in unique[:10]) + (";..." if len(unique) > 10 else "")
        fact = "explanation_quality_signal_by_explainer_model"
        add(fact, explainer_id, model_id, "n_unique", len(unique), len(values))
        add(fact, explainer_id, model_id, "values", rendered, len(values))
        add(fact, explainer_id, model_id, "min", min(values), len(values))
        add(fact, explainer_id, model_id, "max", max(values), len(values))
        add(
            fact,
            explainer_id,
            model_id,
            "faithfulness_evidence_sources",
            ";".join(
                f"{source}:{count}"
                for source, count in sorted(quality_sources[(explainer_id, model_id)].items())
            ),
            len(values),
        )
        if len(unique) == 1:
            constant_cells.append(f"{explainer_id}/{model_id}={unique[0]:g}")
    if constant_cells:
        weight = common.correction_flag(bundle.payload, "faithfulness_quality_weight")
        headlines.append(
            "Quality signal is a constant within these explainer/model cells: "
            + ", ".join(constant_cells)
            + f" (its weight in faithfulness: {weight})."
        )

    # 6. Verifiability saturation.
    saturation: dict[tuple[str, str], list[float]] = defaultdict(list)
    saturation_family: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for record in with_instances:
        for instance in record.instances or []:
            value = instance.get("css_verifiability")
            if value is None:
                continue
            saturation[(record.dataset_id, record.track)].append(float(value))
            saturation_family[(record.dataset_id, record.track, record.family)].append(float(value))
    shares: list[float] = []
    fact = "verifiability_saturation_by_dataset_track"
    for (dataset_id, track), values in sorted(saturation.items()):
        share = sum(1 for value in values if value >= SATURATION) / len(values)
        shares.append(share)
        add(fact, dataset_id, track, "n", len(values), len(values))
        add(fact, dataset_id, track, "mean_verifiability", float(np.mean(values)), len(values))
        add(fact, dataset_id, track, "min_verifiability", min(values), len(values))
        add(fact, dataset_id, track, "share_eq_1", share, len(values))
        for (d_id, t_id, family), family_values in sorted(saturation_family.items()):
            if (d_id, t_id) != (dataset_id, track):
                continue
            add(
                fact,
                dataset_id,
                track,
                f"share_eq_1[{family}]",
                sum(1 for value in family_values if value >= SATURATION) / len(family_values),
                len(family_values),
            )
    if shares:
        pooled = [value for values in saturation.values() for value in values]
        headlines.append(
            f"Verifiability saturated (V = 1) in "
            f"{sum(1 for v in pooled if v >= SATURATION) / len(pooled):.1%} of explanations; "
            f"per dataset x track the share ranges {min(shares):.1%}-{max(shares):.1%}."
        )
    fallback_v: dict[tuple[str, str], list[ConfigRecord]] = defaultdict(list)
    for record in summary_only:
        fallback_v[(record.dataset_id, record.track)].append(record)
    for (dataset_id, track), items in sorted(fallback_v.items()):
        values = [
            float(item.metrics["css_verifiability"])  # type: ignore[arg-type]
            for item in items
            if item.metrics.get("css_verifiability") is not None
        ]
        if not values:
            continue
        note = "summary-level fallback: configuration means"
        add(fact, dataset_id, track, "config_mean_verifiability", weighted_record_mean(items, "css_verifiability"), len(values), note)
        add(fact, dataset_id, track, "config_share_mean_eq_1", sum(1 for v in values if v >= SATURATION) / len(values), len(values), note)

    # 7. DiCE vs Wachter ties on V/C/R per configuration.
    pairs: dict[tuple[str, str, str], dict[str, ConfigRecord]] = defaultdict(dict)
    for record in records:
        if record.explainer_id in ("dice", "wachter"):
            pairs[(record.dataset_id, record.track, record.model_id)][record.explainer_id] = record
    tied_counts: Counter = Counter()
    total_configs = 0
    for key in sorted(pairs):
        pair = pairs[key]
        if "dice" not in pair or "wachter" not in pair:
            continue
        total_configs += 1
        dice, wachter = pair["dice"], pair["wachter"]
        all_tied = True
        for metric, label in (
            ("css_verifiability", "V"),
            ("css_correctability", "C"),
            ("css_reason_specificity", "R"),
        ):
            dice_value, wachter_value = dice.metrics.get(metric), wachter.metrics.get(metric)
            if dice_value is None or wachter_value is None:
                all_tied = False
                continue
            tied = float(dice_value) == float(wachter_value)
            share = None
            n_matched = None
            if dice.instances is not None and wachter.instances is not None:
                wachter_by_id = {item["instance_id"]: item for item in wachter.instances}
                matched = [
                    (item, wachter_by_id[item["instance_id"]])
                    for item in dice.instances
                    if item["instance_id"] in wachter_by_id
                ]
                n_matched = len(matched)
                if matched:
                    share = sum(
                        1 for left, right in matched if left.get(metric) == right.get(metric)
                    ) / len(matched)
            add(
                "dice_wachter_vcr_tie",
                "|".join(key),
                label,
                "abs_diff_config_mean",
                abs(float(dice_value) - float(wachter_value)),
                n_matched,
                f"dice={float(dice_value):.6f}; wachter={float(wachter_value):.6f}; tied={tied}; "
                f"instance_identical_share={_fmt(share)}",
            )
            if tied:
                tied_counts[label] += 1
            else:
                all_tied = False
        if all_tied:
            tied_counts["VCR"] += 1
    if total_configs:
        for label in ("V", "C", "R", "VCR"):
            add("dice_wachter_vcr_tie", "ALL", label, "configs_tied", tied_counts[label], total_configs)
        headlines.append(
            f"DiCE and Wachter tie exactly on V/C/R in {tied_counts['VCR']} of {total_configs} "
            f"configurations (V {tied_counts['V']}, C {tied_counts['C']}, R {tied_counts['R']})."
        )
    return rows, headlines


def t1_correction_status(
    bundle: Bundle,
    records: list[ConfigRecord],
    add: Any,
    headlines: list[str],
) -> None:
    """T1 rows on which v2 corrections the bundle's run used, checked against the data."""
    version = bundle.corrections_version
    options = common.bundle_corrections(bundle.payload)
    fact = "corrections_status"
    add(fact, "bundle", "", "corrections_version", version, None, "v1 = no corrections recorded")
    if options is not None:
        for name, default in common.CORRECTIONS_V1.items():
            add(fact, "option", name, "value", options.get(name), None, f"v1 default {default}")

    # Faithfulness evidence: share of instances scored against the common reference.
    sources: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        for source, count in (record.faithfulness_sources or {}).items():
            sources[record.explainer_id][source] += count
    common_shares: list[str] = []
    for explainer_id, counter in sorted(sources.items()):
        total = sum(counter.values())
        share = counter.get(common.COMMON_REFERENCE_SOURCE, 0) / total if total else None
        add(
            fact,
            explainer_id,
            family_of(explainer_id),
            "share_common_reference_evidence",
            share,
            total,
            ";".join(f"{name}:{count}" for name, count in sorted(counter.items())),
        )
        if share is not None:
            common_shares.append(f"{explainer_id} {share:.0%}")

    # Quality weight actually used (persisted by v2 runs), else the recorded option.
    weights = sorted(
        {
            round(float(instance["quality_weight"]), 6)
            for record in records
            for instance in record.instances or []
            if instance.get("quality_weight") is not None
        }
    )
    recorded_weight = common.correction_flag(bundle.payload, "faithfulness_quality_weight")
    add(
        fact,
        "faithfulness_quality_weight",
        "persisted" if weights else "recorded_option",
        "values",
        ";".join(f"{value:g}" for value in weights) if weights else recorded_weight,
        None,
        "quality signal still recorded; weight 0 drops it from the score (renormalized)",
    )

    # Where DiCE/Wachter reasons come from (canonical metadata cf_reason_source).
    cf_sources: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        if not is_recourse(record.explainer_id):
            continue
        for instance in record.instances or []:
            if not instance.get("has_canonical"):
                key = "canonical_not_persisted"
            else:
                key = str(instance.get("cf_reason_source") or "not_recorded")
            cf_sources[record.explainer_id][key] += 1
    cf_parts: list[str] = []
    for explainer_id, counter in sorted(cf_sources.items()):
        total = sum(counter.values())
        for source, count in sorted(counter.items()):
            add("cf_reason_source", explainer_id, source, "count", count, total)
    # Cited DiCE/Wachter features the model credits toward approval or treats as
    # neutral (reported here; their reason direction states the native sign).
    for record_explainer, shares in sorted(
        _group_nondenial_shares(records).items()
    ):
        add(
            "cf_nondenial_reason_share",
            record_explainer,
            "mean",
            "mean",
            sum(shares) / len(shares),
            len(shares),
            "share of cited changed features whose native contribution is <= 0",
        )
        cf_parts.append(
            f"{explainer_id}: "
            + ", ".join(f"{source} {count / total:.0%}" for source, count in sorted(counter.items()))
        )

    if version == "v1":
        headlines.append(
            "Corrections v1 (published behaviour): built-in faithfulness of SHAP/LIME/"
            "leave-one-out is self-referential, its quality weight is 0.20, and DiCE/Wachter "
            "reasons are a borrowed attribution ranking."
        )
    else:
        headlines.append(
            f"Corrections {version}: common faithfulness reference "
            f"{common.correction_flag(bundle.payload, 'common_faithfulness_reference')} "
            f"(share of instances scored on {common.COMMON_REFERENCE_SOURCE}: "
            f"{', '.join(common_shares) or 'n/a'}); quality weight "
            f"{';'.join(f'{value:g}' for value in weights) if weights else recorded_weight}; "
            "DiCE/Wachter reasons from counterfactual changes "
            f"{common.correction_flag(bundle.payload, 'counterfactual_reasons_from_changes')} "
            f"({'; '.join(cf_parts) or 'no counterfactual instances'})."
        )


# --------------------------------------------------------------------------- #
# T2: CSS decomposition
# --------------------------------------------------------------------------- #


def build_t2(bundle: Bundle, records: list[ConfigRecord]) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    headlines: list[str] = []
    scopes: list[tuple[str, list[ConfigRecord]]] = [(POOLED, records)]
    for dataset_id in sorted({record.dataset_id for record in records}):
        scopes.append((dataset_id, [record for record in records if record.dataset_id == dataset_id]))
    for scope, subset in scopes:
        grouped: dict[str, list[ConfigRecord]] = defaultdict(list)
        for record in subset:
            grouped[record.explainer_id].append(record)
        scope_rows: list[dict[str, Any]] = []
        for explainer_id, items in sorted(grouped.items()):
            row: dict[str, Any] = {
                "bundle": bundle.name,
                "dataset_id": scope,
                "explainer_id": explainer_id,
                "family": family_of(explainer_id),
                "is_null_baseline": common.is_null_baseline(explainer_id),
                "n_configs": len(items),
                "n_instances": sum(item.evaluated_rows for item in items),
                "V": weighted_record_mean(items, "css_verifiability"),
                "C": weighted_record_mean(items, "css_correctability"),
                "S": weighted_record_mean(items, "css_dsv"),
                "R": weighted_record_mean(items, "css_reason_specificity"),
                "css": weighted_record_mean(items, "css"),
                "css_no_dsv": weighted_record_mean(items, "css_no_dsv"),
            }
            row["dsv_term_0_25_S"] = 0.25 * row["S"] if row["S"] is not None else None
            row["css_minus_css_no_dsv"] = (
                row["css"] - row["css_no_dsv"]
                if row["css"] is not None and row["css_no_dsv"] is not None
                else None
            )
            scope_rows.append(row)
        for metric in ("css", "css_no_dsv"):
            ranks = rescore.rank_real_explainers({row["explainer_id"]: row[metric] for row in scope_rows})
            for row in scope_rows:
                row[f"{metric}_rank"] = ranks.get(row["explainer_id"])
        rows.extend(scope_rows)
        if scope == POOLED:
            css_leader, css_score = leader_of({row["explainer_id"]: row["css"] for row in scope_rows})
            neutral_leader, neutral_score = leader_of(
                {row["explainer_id"]: row["css_no_dsv"] for row in scope_rows}
            )
            headlines.append(
                f"[{bundle.name}] pooled CSS leader {css_leader} ({_fmt(css_score)}); "
                f"CSS without decision-sensitivity leader {neutral_leader} ({_fmt(neutral_score)})."
            )
    return rows, headlines


# --------------------------------------------------------------------------- #
# T3: pairwise recount by category
# --------------------------------------------------------------------------- #


def comparison_category(metric: str, record_a: ConfigRecord, record_b: ConfigRecord) -> str:
    recourse_split = is_recourse(record_a.explainer_id) != is_recourse(record_b.explainer_id)
    if metric in ("actionability", "alf") and recourse_split:
        return "decided_by_construction"
    if metric in ("css", "comprehensibility") and recourse_split:
        return "formula_advantaged"
    if metric in ("faithfulness", "faithfulness_no_quality"):
        # v2: both scored against the same native reference, so the comparison is open
        # even when one of them (DiCE/Wachter) had only a structural proxy in v1 --
        # unless either side agrees with that reference by construction (SHAP on
        # LR/XGB, native-ranked Anchors or CF features; regb.by_construction_flags,
        # the same flags T8 excludes from its leader statements).
        if record_a.common_reference and record_b.common_reference:
            if record_a.by_construction() or record_b.by_construction():
                return "decided_by_construction"
            return "open"
        if record_a.evidence_kind != record_b.evidence_kind:
            return "not_comparable_evidence"
    return "open"


def pairwise_recount(
    records: list[ConfigRecord],
    metrics: tuple[str, ...] = T3_METRICS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Replicate reporting._pairwise_wins (strict > on configuration means) by category."""
    configs: dict[tuple[str, str, str, str], dict[str, ConfigRecord]] = defaultdict(dict)
    for record in records:
        if not common.is_null_baseline(record.explainer_id):
            configs[record.config_key][record.explainer_id] = record
    multiple_bundles = len({key[0] for key in configs}) > 1
    pair_counts: dict[tuple[str, str, str, str, str], Counter] = defaultdict(Counter)
    explainer_counts: dict[tuple[str, str, str], Counter] = defaultdict(Counter)
    for metric in metrics:
        for key in sorted(configs):
            scopes = (key[0], ALL_BUNDLES) if multiple_bundles else (key[0],)
            available = {
                explainer_id: record
                for explainer_id, record in configs[key].items()
                if record.metrics.get(metric) is not None
            }
            for explainer_a, explainer_b in combinations(sorted(available), 2):
                record_a, record_b = available[explainer_a], available[explainer_b]
                category = comparison_category(metric, record_a, record_b)
                value_a = float(record_a.metrics[metric])  # type: ignore[arg-type]
                value_b = float(record_b.metrics[metric])  # type: ignore[arg-type]
                a_wins = value_a > value_b
                b_wins = value_b > value_a
                for scope in scopes:
                    counts = pair_counts[(scope, metric, explainer_a, explainer_b, category)]
                    counts["comparable"] += 1
                    counts["wins_a" if a_wins else "wins_b" if b_wins else "ties"] += 1
                    for explainer_id, won, lost in (
                        (explainer_a, a_wins, b_wins),
                        (explainer_b, b_wins, a_wins),
                    ):
                        tally = explainer_counts[(scope, metric, explainer_id)]
                        outcome = "wins" if won else "losses" if lost else "ties"
                        tally["comparisons"] += 1
                        tally[f"comparisons_{category}"] += 1
                        tally[outcome] += 1
                        tally[f"{outcome}_{category}"] += 1

    metric_order = {metric: index for index, metric in enumerate(metrics)}
    explainer_rows: list[dict[str, Any]] = []
    for (scope, metric, explainer_id), tally in sorted(
        explainer_counts.items(),
        key=lambda item: (item[0][0], metric_order[item[0][1]], item[0][2]),
    ):
        row: dict[str, Any] = {
            "scope": scope,
            "metric": metric,
            "explainer_id": explainer_id,
            "family": family_of(explainer_id),
            "comparisons": tally["comparisons"],
            "wins": tally["wins"],
            "losses": tally["losses"],
            "ties": tally["ties"],
        }
        for category in CATEGORIES:
            row[f"comparisons_{category}"] = tally[f"comparisons_{category}"]
            row[f"wins_{category}"] = tally[f"wins_{category}"]
            row[f"losses_{category}"] = tally[f"losses_{category}"]
        row["open_win_rate"] = (
            tally["wins_open"] / tally["comparisons_open"] if tally["comparisons_open"] else None
        )
        row["share_of_wins_not_open"] = (
            (tally["wins"] - tally["wins_open"]) / tally["wins"] if tally["wins"] else None
        )
        explainer_rows.append(row)

    pair_rows = [
        {
            "scope": scope,
            "metric": metric,
            "explainer_a": explainer_a,
            "explainer_b": explainer_b,
            "category": category,
            "comparable_configurations": counts["comparable"],
            "wins_a": counts["wins_a"],
            "wins_b": counts["wins_b"],
            "ties": counts["ties"],
        }
        for (scope, metric, explainer_a, explainer_b, category), counts in sorted(
            pair_counts.items(),
            key=lambda item: (item[0][0], metric_order[item[0][1]], item[0][2], item[0][3], item[0][4]),
        )
    ]
    return explainer_rows, pair_rows


def check_pairwise_against_reporting(
    bundle: Bundle,
    pair_rows: list[dict[str, Any]],
) -> str:
    path = bundle.directory / "pairwise_wins.csv"
    if not path.is_file():
        return f"[{bundle.name}] pairwise_wins.csv not found; cross-check against reporting skipped."
    ours: dict[tuple[str, str, str], list[int]] = defaultdict(lambda: [0, 0, 0])
    for row in pair_rows:
        if row["scope"] != bundle.name:
            continue
        totals = ours[(row["metric"], row["explainer_a"], row["explainer_b"])]
        totals[0] += row["wins_a"]
        totals[1] += row["wins_b"]
        totals[2] += row["ties"]
    matches = 0
    mismatches: list[str] = []
    for item in read_csv_rows(path):
        metric = item.get("metric", "")
        explainer_a, explainer_b = item.get("explainer_a", ""), item.get("explainer_b", "")
        if (
            metric not in MAIN_METRICS
            or common.is_null_baseline(explainer_a)
            or common.is_null_baseline(explainer_b)
        ):
            continue
        theirs = [int(float(item["wins_a"])), int(float(item["wins_b"])), int(float(item["ties"]))]
        mine = list(ours.get((metric, explainer_a, explainer_b), [0, 0, 0]))
        if mine == theirs:
            matches += 1
        else:
            mismatches.append(f"{metric} {explainer_a} vs {explainer_b}: ours {mine}, reporting {theirs}")
    message = (
        f"[{bundle.name}] recount vs reporting pairwise_wins.csv: {matches} pair rows match, "
        f"{len(mismatches)} differ."
    )
    if mismatches:
        message += " First differences: " + "; ".join(mismatches[:3])
    return message


def t3_headlines(explainer_rows: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in explainer_rows:
        grouped[(row["scope"], row["metric"])].append(row)
    for (scope, metric), rows in grouped.items():
        total_wins = sum(row["wins"] for row in rows)
        if not total_wins:
            continue
        champion = sorted(rows, key=lambda row: (-row["wins"], row["explainer_id"]))[0]
        open_champion = sorted(rows, key=lambda row: (-row["wins_open"], row["explainer_id"]))[0]
        not_open = sum(row["wins"] - row["wins_open"] for row in rows)
        lines.append(
            f"[{scope}] {metric}: most wins {champion['explainer_id']} ({champion['wins']}); "
            f"most open wins {open_champion['explainer_id']} ({open_champion['wins_open']}); "
            f"{not_open}/{total_wins} wins are not 'open'."
        )
    return lines


# --------------------------------------------------------------------------- #
# T4: per-domain means with hierarchical bootstrap
# --------------------------------------------------------------------------- #


def hierarchical_bootstrap(
    structure: dict[str, dict[str, np.ndarray]],
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Percentile 95% CI resampling track, then model within track, then instances."""
    tracks = sorted(structure)
    track_models = [[structure[track][model] for model in sorted(structure[track])] for track in tracks]
    n_metrics = track_models[0][0].shape[1]
    stats = np.full((n_boot, n_metrics), np.nan)
    for replicate in range(n_boot):
        total = np.zeros(n_metrics)
        count = np.zeros(n_metrics)
        for track_index in rng.integers(0, len(tracks), size=len(tracks)):
            models = track_models[int(track_index)]
            for model_index in rng.integers(0, len(models), size=len(models)):
                array = models[int(model_index)]
                sample = array[rng.integers(0, array.shape[0], size=array.shape[0])]
                mask = ~np.isnan(sample)
                total += np.where(mask, sample, 0.0).sum(axis=0)
                count += mask.sum(axis=0)
        valid = count > 0
        stats[replicate, valid] = total[valid] / count[valid]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        low = np.nanpercentile(stats, 2.5, axis=0)
        high = np.nanpercentile(stats, 97.5, axis=0)
    return low, high


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def build_t4(
    bundle: Bundle,
    records: list[ConfigRecord],
    n_boot: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    headlines: list[str] = []
    cells: dict[tuple[str, str], list[ConfigRecord]] = defaultdict(list)
    for record in records:
        cells[(record.dataset_id, record.explainer_id)].append(record)
    for (dataset_id, explainer_id), items in sorted(cells.items()):
        has_instances = all(item.instances is not None for item in items)
        lows: np.ndarray | None = None
        highs: np.ndarray | None = None
        if has_instances:
            nested: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
            for item in items:
                if not item.instances:
                    continue
                array = np.array(
                    [
                        [
                            np.nan if instance.get(metric) is None else float(instance[metric])
                            for metric in T4_METRICS
                        ]
                        for instance in item.instances
                    ],
                    dtype=float,
                ).reshape(-1, len(T4_METRICS))
                nested[item.track][item.model_id].append(array)
            if not nested:
                continue
            structure = {
                track: {model: np.concatenate(arrays) for model, arrays in models.items()}
                for track, models in nested.items()
            }
            stacked = np.concatenate(
                [array for models in structure.values() for array in models.values()]
            )
            mask = ~np.isnan(stacked)
            counts = [int(value) for value in mask.sum(axis=0)]
            sums = np.where(mask, stacked, 0.0).sum(axis=0)
            points: list[float | None] = [
                float(sums[index] / counts[index]) if counts[index] else None
                for index in range(len(T4_METRICS))
            ]
            if n_boot > 0:
                rng = np.random.default_rng(
                    [int(seed), zlib.crc32(f"{bundle.name}|{dataset_id}|{explainer_id}".encode("utf-8"))]
                )
                lows, highs = hierarchical_bootstrap(structure, n_boot, rng)
                method = (
                    f"hierarchical bootstrap (track > model > instance), {n_boot} resamples, seed {seed}"
                )
            else:
                method = "no bootstrap requested"
        else:
            points = [weighted_record_mean(items, metric) for metric in T4_METRICS]
            counts = [
                sum(item.evaluated_rows for item in items if item.metrics.get(metric) is not None)
                for metric in T4_METRICS
            ]
            method = "none: summary-level means (bundle has no instance_results)"
        for index, metric in enumerate(T4_METRICS):
            rows.append(
                {
                    "bundle": bundle.name,
                    "dataset_id": dataset_id,
                    "explainer_id": explainer_id,
                    "family": family_of(explainer_id),
                    "is_null_baseline": common.is_null_baseline(explainer_id),
                    "metric": metric,
                    "n_instances": counts[index],
                    "mean": points[index],
                    "ci_low": _finite_or_none(lows[index]) if lows is not None else None,
                    "ci_high": _finite_or_none(highs[index]) if highs is not None else None,
                    "ci_method": method,
                }
            )
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset_id"], row["metric"])].append(row)
    leaders: dict[tuple[str, str], str | None] = {}
    for key, items in grouped.items():
        ranks = rescore.rank_real_explainers({row["explainer_id"]: row["mean"] for row in items})
        for row in items:
            row["rank"] = ranks.get(row["explainer_id"])
            row["is_leader"] = ranks.get(row["explainer_id"]) == 1
        leaders[key] = next((row["explainer_id"] for row in items if row["is_leader"]), None)
    for dataset_id in sorted({key[0] for key in leaders}):
        parts = []
        for original, neutral in (
            ("css", "css_no_dsv"),
            ("comprehensibility", "comprehensibility_no_rs"),
            ("faithfulness", "faithfulness_no_quality"),
        ):
            parts.append(
                f"{original} {leaders.get((dataset_id, original))} -> "
                f"{neutral} {leaders.get((dataset_id, neutral)) or 'n/a'}"
            )
        headlines.append(f"[{bundle.name}] {dataset_id}: " + "; ".join(parts) + ".")
    return rows, headlines


# --------------------------------------------------------------------------- #
# T5: correlations
# --------------------------------------------------------------------------- #


def spearman(x: list[Any], y: list[Any]) -> tuple[float | None, float | None, int]:
    pairs = [
        (float(a), float(b))
        for a, b in zip(x, y)
        if a is not None and b is not None and math.isfinite(float(a)) and math.isfinite(float(b))
    ]
    n = len(pairs)
    if n < 3:
        return None, None, n
    xs = [pair[0] for pair in pairs]
    ys = [pair[1] for pair in pairs]
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return None, None, n
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = spearmanr(xs, ys)
    rho, p_value = float(result[0]), float(result[1])
    return (
        rho if math.isfinite(rho) else None,
        p_value if math.isfinite(p_value) else None,
        n,
    )


def build_t5(bundle: Bundle, records: list[ConfigRecord]) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    headlines: list[str] = []

    def add(
        level: str,
        scope: str,
        dataset_id: str,
        explainer_id: str,
        x_metric: str,
        y_metric: str,
        x: list[Any],
        y: list[Any],
    ) -> None:
        rho, p_value, n = spearman(x, y)
        rows.append(
            {
                "bundle": bundle.name,
                "level": level,
                "scope": scope,
                "dataset_id": dataset_id,
                "explainer_id": explainer_id,
                "x_metric": x_metric,
                "y_metric": y_metric,
                "n": n,
                "spearman_rho": rho,
                "p_value": p_value,
            }
        )

    cells: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        for instance in record.instances or []:
            cells[(record.dataset_id, record.explainer_id)].append(instance)
    for (dataset_id, explainer_id), instances in sorted(cells.items()):
        for x_metric, y_metric in CORRELATION_PAIRS:
            add(
                "instance",
                "dataset_x_explainer",
                dataset_id,
                explainer_id,
                x_metric,
                y_metric,
                [instance.get(x_metric) for instance in instances],
                [instance.get(y_metric) for instance in instances],
            )

    real = [record for record in records if not common.is_null_baseline(record.explainer_id)]
    for x_metric, y_metric in CORRELATION_PAIRS:
        add(
            "configuration",
            "pooled",
            POOLED,
            "__all_real__",
            x_metric,
            y_metric,
            [record.metrics.get(x_metric) for record in real],
            [record.metrics.get(y_metric) for record in real],
        )
    for dataset_id in sorted({record.dataset_id for record in real}):
        subset = [record for record in real if record.dataset_id == dataset_id]
        for x_metric, y_metric in CORRELATION_PAIRS:
            add(
                "configuration",
                "within_dataset",
                dataset_id,
                "__all_real__",
                x_metric,
                y_metric,
                [record.metrics.get(x_metric) for record in subset],
                [record.metrics.get(y_metric) for record in subset],
            )
    for explainer_id in sorted({record.explainer_id for record in records}):
        subset = [record for record in records if record.explainer_id == explainer_id]
        for x_metric, y_metric in CORRELATION_PAIRS:
            add(
                "configuration",
                "within_explainer",
                POOLED,
                explainer_id,
                x_metric,
                y_metric,
                [record.metrics.get(x_metric) for record in subset],
                [record.metrics.get(y_metric) for record in subset],
            )

    pooled = {
        (row["x_metric"], row["y_metric"]): row
        for row in rows
        if row["level"] == "configuration" and row["scope"] == "pooled"
    }
    parts = [
        f"rho({x}, {y}) = {_fmt(pooled[(x, y)]['spearman_rho'])} (n={pooled[(x, y)]['n']})"
        for x, y in CORRELATION_PAIRS
        if (x, y) in pooled
    ]
    headlines.append(f"[{bundle.name}] pooled configuration level: " + "; ".join(parts) + ".")
    within = [
        row
        for row in rows
        if row["level"] == "configuration"
        and row["scope"] == "within_explainer"
        and (row["x_metric"], row["y_metric"]) == ("faithfulness", "css")
        and not common.is_null_baseline(row["explainer_id"])
    ]
    if within:
        headlines.append(
            f"[{bundle.name}] within-explainer configuration-level rho(faithfulness, css): "
            + ", ".join(f"{row['explainer_id']} {_fmt(row['spearman_rho'])}" for row in within)
            + "."
        )
    return rows, headlines


# --------------------------------------------------------------------------- #
# T6: robustness leaders
# --------------------------------------------------------------------------- #


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(text: Any) -> float | None:
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return _finite_or_none(text)
    stripped = str(text).strip()
    if not stripped or stripped.lower() in {"none", "nan", "null"}:
        return None
    try:
        return _finite_or_none(float(stripped))
    except ValueError:
        return None


def t6_row(**values: Any) -> dict[str, Any]:
    return {column: values.get(column) for column in T6_COLUMNS}


def leader_comparison_row(
    baseline_scores: dict[str, float | None],
    check_scores: dict[str, float | None],
    *,
    check_family: str,
    check_name: str,
    source: str,
    scope: str,
    metric: str,
    note: str = "",
) -> dict[str, Any] | None:
    baseline_leader, baseline_score = leader_of(baseline_scores)
    check_leader, check_score = leader_of(check_scores)
    if baseline_leader is None and check_leader is None:
        return None
    comparable = baseline_leader is not None and check_leader is not None
    changed = baseline_leader != check_leader if comparable else None
    extra = [note] if note else []
    if set(check_scores) != set(baseline_scores):
        common_leader, _ = leader_of(
            {
                explainer_id: value
                for explainer_id, value in check_scores.items()
                if explainer_id in baseline_scores
            }
        )
        extra.append(f"leader among the baseline's explainers: {common_leader}")
    return t6_row(
        check_family=check_family,
        check_name=check_name,
        source=source,
        scope=scope,
        metric=metric,
        baseline_leader=baseline_leader,
        baseline_score=baseline_score,
        check_leader=check_leader,
        check_score=check_score,
        changed=changed,
        n_comparisons=1 if comparable else 0,
        n_changed=1 if changed else 0,
        note="; ".join(extra),
    )


def scores_from_by_explainer(rows: list[dict[str, str]]) -> dict[str, dict[str, float | None]]:
    return {
        metric: {row["explainer_id"]: to_float(row.get(f"mean_{metric}")) for row in rows}
        for metric in MAIN_METRICS
    }


def scores_from_summary(
    summary_rows: list[dict[str, Any]],
    datasets: list[str] | None = None,
) -> dict[str, dict[str, float | None]]:
    wanted = set(datasets) if datasets is not None else None
    selected = [row for row in summary_rows if wanted is None or row["dataset_id"] in wanted]
    means = rescore.explainer_level_means(selected, [f"mean_{metric}" for metric in MAIN_METRICS])
    return {
        metric: {explainer_id: values[f"mean_{metric}"] for explainer_id, values in means.items()}
        for metric in MAIN_METRICS
    }


def profile_leaders(rows: list[dict[str, str]]) -> dict[str, tuple[str, float | None]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["scenario_name"]].append(row)
    leaders: dict[str, tuple[str, float | None]] = {}
    for scenario, items in grouped.items():
        real = [item for item in items if not common.is_null_baseline(item["explainer_id"])]
        if not real:
            continue
        best = min(
            real,
            key=lambda item: (
                rank if (rank := to_float(item.get("rank"))) is not None else math.inf,
                item["explainer_id"],
            ),
        )
        leaders[scenario] = (best["explainer_id"], to_float(best.get("composite_score")))
    return leaders


def t6_profile_rows(
    bundle_name: str,
    path: Path,
    *,
    check_family: str,
    baseline_scenario: str,
    metric: str,
) -> list[dict[str, Any]]:
    leaders = profile_leaders(read_csv_rows(path))
    if baseline_scenario not in leaders:
        return []
    baseline_leader, baseline_score = leaders[baseline_scenario]
    rows = []
    for scenario, (check_leader, check_score) in leaders.items():
        if scenario == baseline_scenario:
            continue
        rows.append(
            t6_row(
                check_family=check_family,
                check_name=scenario,
                source=f"{bundle_name}/{path.name}",
                scope=bundle_name,
                metric=metric,
                baseline_leader=baseline_leader,
                baseline_score=baseline_score,
                check_leader=check_leader,
                check_score=check_score,
                changed=check_leader != baseline_leader,
                n_comparisons=1,
                n_changed=int(check_leader != baseline_leader),
                note=f"baseline scenario {baseline_scenario}; leaders exclude null baselines",
            )
        )
    return rows


def t6_main_bundle(bundle: Bundle) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    notes: list[str] = []
    directory = bundle.directory
    by_explainer_path = directory / "by_explainer.csv"
    if by_explainer_path.is_file():
        baseline = scores_from_by_explainer(read_csv_rows(by_explainer_path))
    else:
        baseline = scores_from_summary(bundle.summary)
        notes.append(f"[{bundle.name}] by_explainer.csv not found; baseline leaders from results.json summary.")
    has_null = any(common.is_null_baseline(row["explainer_id"]) for row in bundle.summary)

    lodo_path = directory / "leave_one_dataset_out_leaders.csv"
    lodo_full_path = directory / "leave_one_dataset_out.csv"
    if lodo_path.is_file():
        full = read_csv_rows(lodo_full_path) if lodo_full_path.is_file() else []
        for item in read_csv_rows(lodo_path):
            metric = item["metric"]
            omitted = item["omitted_dataset_id"]
            check_leader: str | None = item["explainer_id"]
            check_score = to_float(item.get("score"))
            note = "baseline: all datasets"
            if check_leader is not None and common.is_null_baseline(check_leader):
                scores = {
                    row["explainer_id"]: to_float(row.get(f"mean_{metric}"))
                    for row in full
                    if row.get("omitted_dataset_id") == omitted
                }
                check_leader, check_score = leader_of(scores)
                note += "; reporting leader was a null baseline, recomputed from leave_one_dataset_out.csv"
            baseline_leader, baseline_score = leader_of(baseline.get(metric, {}))
            comparable = baseline_leader is not None and check_leader is not None
            rows.append(
                t6_row(
                    check_family="leave_one_dataset_out",
                    check_name=f"omit {omitted}",
                    source=f"{bundle.name}/{lodo_path.name}",
                    scope=bundle.name,
                    metric=metric,
                    baseline_leader=baseline_leader,
                    baseline_score=baseline_score,
                    check_leader=check_leader,
                    check_score=check_score,
                    changed=(baseline_leader != check_leader) if comparable else None,
                    n_comparisons=int(comparable),
                    n_changed=int(comparable and baseline_leader != check_leader),
                    note=note,
                )
            )
    else:
        notes.append(f"[{bundle.name}] leave_one_dataset_out_leaders.csv not found.")

    balanced_leader: str | None = None
    ablation_path = directory / "metric_ablation_profiles.csv"
    if ablation_path.is_file():
        balanced = profile_leaders(read_csv_rows(ablation_path)).get("all_metrics_balanced")
        balanced_leader = balanced[0] if balanced else None
        rows += t6_profile_rows(
            bundle.name,
            ablation_path,
            check_family="metric_ablation",
            baseline_scenario="all_metrics_balanced",
            metric="balanced_composite",
        )
    else:
        notes.append(f"[{bundle.name}] metric_ablation_profiles.csv not found.")

    random_path = directory / "metric_weight_random_summary.csv"
    if random_path.is_file():
        items = [
            item
            for item in read_csv_rows(random_path)
            if not common.is_null_baseline(item["explainer_id"])
        ]
        if items:
            ordered = sorted(
                items,
                key=lambda item: (
                    -(to_float(item.get("champion_frequency")) or 0.0),
                    rank if (rank := to_float(item.get("mean_rank"))) is not None else math.inf,
                    item["explainer_id"],
                ),
            )
            top = ordered[0]
            draws = int(to_float(items[0].get("num_draws")) or 0)
            frequency = {
                item["explainer_id"]: to_float(item.get("champion_frequency")) or 0.0
                for item in items
            }
            baseline_frequency = frequency.get(balanced_leader or "", 0.0)
            changed_draws = (
                int(round((1.0 - baseline_frequency) * draws)) if balanced_leader else None
            )
            rows.append(
                t6_row(
                    check_family="random_metric_weights",
                    check_name=f"{draws} random weight draws",
                    source=f"{bundle.name}/{random_path.name}",
                    scope=bundle.name,
                    metric="random_weight_composite",
                    baseline_leader=balanced_leader,
                    baseline_score=baseline_frequency if balanced_leader else None,
                    check_leader=top["explainer_id"],
                    check_score=to_float(top.get("champion_frequency")),
                    changed=(top["explainer_id"] != balanced_leader) if balanced_leader else None,
                    n_comparisons=draws if balanced_leader else 0,
                    n_changed=changed_draws,
                    note=(
                        "scores are champion frequencies; n_changed = draws whose champion "
                        "differs from the balanced-composite leader"
                    ),
                )
            )
    else:
        notes.append(f"[{bundle.name}] metric_weight_random_summary.csv not found.")

    for file_name, family, baseline_scenario, metric in (
        ("css_definition_profiles.csv", "css_definition", "observed_css", "css"),
        ("alf_definition_profiles.csv", "alf_definition", "observed_alf", "alf"),
    ):
        path = directory / file_name
        if path.is_file():
            rows += t6_profile_rows(
                bundle.name,
                path,
                check_family=family,
                baseline_scenario=baseline_scenario,
                metric=metric,
            )
        else:
            notes.append(f"[{bundle.name}] {file_name} not found.")

    track_path = directory / "track_policy_leader_summary.csv"
    if track_path.is_file():
        for item in read_csv_rows(track_path):
            flips = int(to_float(item.get("flip_count")) or 0)
            comparisons = int(to_float(item.get("comparison_count")) or 0)
            note = (
                "per dataset x model, primary-track leader vs audit-track leader; "
                f"same_leader_rate={item.get('same_leader_rate')}"
            )
            if has_null:
                note += "; computed by reporting including null baselines"
            rows.append(
                t6_row(
                    check_family="track_policy",
                    check_name="primary vs audit track",
                    source=f"{bundle.name}/{track_path.name}",
                    scope=bundle.name,
                    metric=item.get("metric"),
                    changed=flips > 0,
                    n_comparisons=comparisons,
                    n_changed=flips,
                    note=note,
                )
            )
    else:
        notes.append(f"[{bundle.name}] track_policy_leader_summary.csv not found.")
    return rows, notes


@dataclass
class RobustnessBundle:
    directory: Path
    experiment_name: str
    datasets: list[str]
    # common.corrections_version of the bundle ("v1" for every published bundle).
    corrections_version: str = "v1"


_CORRECTIONS_VERSION_RE = re.compile(rb'"corrections_version"\s*:\s*"([^"]*)"')


def read_corrections_version(results_path: Path) -> str:
    """The bundle's recorded payload["cslaw"]["corrections_version"] ("v1" when absent).

    Streams the file instead of parsing it (published bundles can be large); the
    runner writes the marker only for corrected runs, next to the full options.
    """
    if not results_path.is_file():
        return "v1"
    overlap = 256
    tail = b""
    with results_path.open("rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            window = tail + chunk
            match = _CORRECTIONS_VERSION_RE.search(window)
            if match:
                return match.group(1).decode("utf-8", errors="replace") or "v1"
            tail = window[-overlap:]
    return "v1"


def read_experiment_name(results_path: Path) -> str:
    if not results_path.is_file():
        return ""
    with results_path.open("r", encoding="utf-8", errors="replace") as handle:
        head = handle.read(8192)
    match = re.search(r'"experiment_name"\s*:\s*"([^"]*)"', head)
    return match.group(1) if match else ""


def read_bundle_datasets(directory: Path) -> list[str]:
    for file_name in ("by_dataset_track_explainer.csv", "summary.csv", "by_model_explainer.csv"):
        path = directory / file_name
        if path.is_file():
            datasets = {row.get("dataset_id") or "" for row in read_csv_rows(path)}
            datasets.discard("")
            if datasets:
                return sorted(datasets)
    return []


def classify_robustness(directory_name: str, experiment_name: str) -> tuple[str, str | None]:
    if directory_name.startswith("cslaw_"):
        # CS&Law re-run bundles (benchmark/scripts/cluster/run_cslaw_credit.sh) live
        # next to the published bundles; label them so T6 never reports them as an
        # unnamed robustness check.
        return "cslaw_rerun", None
    text = f"{directory_name} {experiment_name}"
    for needle, family, baseline in ROBUSTNESS_PATTERNS:
        if needle in text:
            return family, baseline
    if experiment_name == FOUR_DATASET_BUDGET8:
        return "denied_instance_budget", None
    return "other_bundle", None


def discover_robustness_bundles(root: Path, exclude: set[Path]) -> list[RobustnessBundle]:
    candidates = [root] + sorted(path for path in root.iterdir() if path.is_dir())
    found: list[RobustnessBundle] = []
    for directory in candidates:
        resolved = directory.resolve()
        if resolved in exclude or not (directory / "by_explainer.csv").is_file():
            continue
        experiment_name = read_experiment_name(directory / "results.json")
        if "__rescored_" in experiment_name:
            continue
        found.append(
            RobustnessBundle(
                directory=resolved,
                experiment_name=experiment_name,
                datasets=read_bundle_datasets(directory),
                corrections_version=read_corrections_version(directory / "results.json"),
            )
        )
    return found


def _match_baseline(
    candidates: list[RobustnessBundle],
    needle: str,
    own: Path,
) -> RobustnessBundle | None:
    for candidate in candidates:
        if candidate.directory == own:
            continue
        if needle == FOUR_DATASET_BUDGET8:
            if candidate.experiment_name == FOUR_DATASET_BUDGET8:
                return candidate
        elif needle in candidate.directory.name or needle in candidate.experiment_name:
            return candidate
    return None


def t6_robustness_rows(
    root: Path,
    main_bundles: list[Bundle],
    exclude: set[Path],
    pair_overrides: dict[Path, Path],
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    notes: list[str] = []
    candidates = discover_robustness_bundles(root, exclude)
    if not candidates:
        notes.append(f"No robustness bundles with by_explainer.csv found under {root}.")
    for candidate in candidates:
        family, preferred = classify_robustness(candidate.directory.name, candidate.experiment_name)
        check_scores = scores_from_by_explainer(read_csv_rows(candidate.directory / "by_explainer.csv"))
        baseline_scores: dict[str, dict[str, float | None]] | None = None
        baseline_label = ""
        baseline_version = ""
        # A baseline must have the same corrections version as the check; otherwise a
        # leader change mixes the v1->v2 corrections with the robustness factor.
        check_version = candidate.corrections_version
        version_mismatches: list[str] = []
        override = pair_overrides.get(candidate.directory)
        if override is not None and (override / "by_explainer.csv").is_file():
            override_version = read_corrections_version(override / "results.json")
            if override_version == check_version:
                baseline_scores = scores_from_by_explainer(
                    read_csv_rows(override / "by_explainer.csv")
                )
                baseline_label = f"{override.name}/by_explainer.csv (--robustness-pair)"
                baseline_version = override_version
            else:
                version_mismatches.append(f"{override.name} ({override_version})")
        elif preferred is not None:
            match = _match_baseline(candidates, preferred, candidate.directory)
            if match is not None:
                if match.corrections_version == check_version:
                    baseline_scores = scores_from_by_explainer(
                        read_csv_rows(match.directory / "by_explainer.csv")
                    )
                    baseline_label = f"{match.directory.name}/by_explainer.csv"
                    baseline_version = match.corrections_version
                else:
                    version_mismatches.append(
                        f"{match.directory.name} ({match.corrections_version})"
                    )
        if baseline_scores is None:
            for bundle in main_bundles:
                if candidate.datasets and set(candidate.datasets) <= set(bundle.datasets):
                    if bundle.corrections_version != check_version:
                        version_mismatches.append(
                            f"main bundle {bundle.name} ({bundle.corrections_version})"
                        )
                        continue
                    baseline_scores = scores_from_summary(bundle.summary, candidate.datasets)
                    baseline_label = (
                        f"main bundle {bundle.name} restricted to {', '.join(candidate.datasets)}"
                    )
                    baseline_version = bundle.corrections_version
                    break
        if baseline_scores is None:
            if version_mismatches:
                notes.append(
                    f"Skipped {candidate.directory.name} (corrections {check_version}): the only "
                    f"baselines covering it have another corrections version "
                    f"({'; '.join(version_mismatches)}); a cross-version leader change would mix "
                    "the corrections with the robustness factor. The same-version comparison is in "
                    "the offline paper_tables run (cslaw_offline_<tag>/paper_tables)."
                )
            else:
                notes.append(
                    f"Skipped {candidate.directory.name}: no baseline (datasets {candidate.datasets} "
                    "not covered by any --bundle and no matching baseline directory)."
                )
            continue
        for metric in MAIN_METRICS:
            row = leader_comparison_row(
                baseline_scores.get(metric, {}),
                check_scores.get(metric, {}),
                check_family=family,
                check_name=candidate.directory.name,
                source=f"{candidate.directory.name}/by_explainer.csv",
                scope=",".join(candidate.datasets) or "unknown datasets",
                metric=metric,
                note=(
                    f"baseline: {baseline_label}; corrections baseline {baseline_version} / "
                    f"check {check_version}"
                ),
            )
            if row is not None:
                rows.append(row)
    return rows, notes


def discover_rescored_payloads(root: Path) -> list[Path]:
    paths: list[Path] = []
    for pattern in ("results.json", "*/results.json", "*/*/results.json"):
        paths.extend(sorted(root.glob(pattern)))
    return [path for path in paths if "__rescored_" in read_experiment_name(path)]


def relocate_source_results(
    source: Any,
    source_experiment_name: Any,
    search_roots: list[Path],
) -> Path | None:
    """Find a rescore source bundle whose recorded absolute path no longer exists.

    rescore.py stores ``source_results`` as an absolute path, which breaks when the
    offline step ran on another machine (e.g. a laptop) than the analysis step. The
    published bundle then usually sits under the same directory name below
    ``--robustness-root``; accept it only if its experiment name matches.
    """
    if not source:
        return None
    recorded = Path(str(source))
    for root in search_roots:
        candidate = root / recorded.parent.name / recorded.name
        if not candidate.is_file():
            continue
        if source_experiment_name and read_experiment_name(candidate) != source_experiment_name:
            continue
        return candidate
    return None


def t6_rescored_rows(
    root: Path,
    main_bundles: list[Bundle],
    search_roots: list[Path] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    notes: list[str] = []
    paths = discover_rescored_payloads(root)
    if not paths:
        notes.append(
            f"No rescore.py outputs (results.json with a '__rescored_' experiment name) under {root}."
        )
    for path in paths:
        payload = common.load_results(path)
        info = payload.get("rescore") or {}
        variant = str(info.get("variant", "unknown"))
        baseline_summary: list[dict[str, Any]] | None = None
        baseline_label = ""
        source = info.get("source_results")
        relocated = (
            None
            if not source or Path(source).is_file()
            else relocate_source_results(
                source, info.get("source_experiment_name"), list(search_roots or [])
            )
        )
        if source and Path(source).is_file():
            baseline_summary = common.load_results(source).get("summary", [])
            baseline_label = str(source)
        elif relocated is not None:
            baseline_summary = common.load_results(relocated).get("summary", [])
            baseline_label = f"{relocated} (recorded source {source} not found; matched by name)"
            notes.append(
                f"{path}: recorded source {source} not found; used {relocated} "
                "(same directory name and experiment name)."
            )
        else:
            for bundle in main_bundles:
                if bundle.experiment_name == info.get("source_experiment_name"):
                    baseline_summary = bundle.summary
                    baseline_label = f"--bundle {bundle.name}"
                    break
        if baseline_summary is None:
            notes.append(
                f"Skipped {path}: source bundle {source!r} not found (also not under "
                "--robustness-root by directory name) and no --bundle matches. Run the offline "
                "step on the machine that runs this step, or rerun it here."
            )
            continue
        summary = payload.get("summary", [])
        datasets = sorted({row["dataset_id"] for row in summary})
        scopes: list[tuple[list[str] | None, str]] = [(None, POOLED)]
        scopes += [([dataset_id], dataset_id) for dataset_id in datasets]
        for scope_datasets, scope_name in scopes:
            baseline_scores = scores_from_summary(baseline_summary, scope_datasets)
            check_scores = scores_from_summary(summary, scope_datasets)
            for metric in VARIANT_AFFECTED_METRICS.get(variant, MAIN_METRICS):
                row = leader_comparison_row(
                    baseline_scores.get(metric, {}),
                    check_scores.get(metric, {}),
                    check_family="family_neutral_rescore",
                    check_name=variant,
                    source=str(path),
                    scope=f"{info.get('source_experiment_name', '')}:{scope_name}",
                    metric=metric,
                    note=f"baseline: {baseline_label}",
                )
                if row is not None:
                    rows.append(row)
    return rows, notes


def t6_sweep_rows(path: Path) -> list[dict[str, Any]]:
    by_subset: dict[str, dict[float, dict[str, float | None]]] = defaultdict(lambda: defaultdict(dict))
    for item in read_csv_rows(path):
        weight = to_float(item.get("weight"))
        if weight is None:
            continue
        by_subset[item["subset"]][weight][item["explainer_id"]] = to_float(item.get("mean_css"))
    rows: list[dict[str, Any]] = []
    published = rescore.PUBLISHED_DSV_WEIGHT
    for subset, by_weight in sorted(by_subset.items()):
        weights = sorted(by_weight)
        baseline_weight = min(weights, key=lambda weight: abs(weight - published))
        for weight in weights:
            if weight == baseline_weight:
                continue
            row = leader_comparison_row(
                by_weight[baseline_weight],
                by_weight[weight],
                check_family="css_dsv_weight_sweep",
                check_name=f"S weight {weight:g}",
                source=str(path),
                scope=subset,
                metric="css",
                note=f"baseline weight {baseline_weight:g}",
            )
            if row is not None:
                rows.append(row)
    return rows


def t6_headlines(rows: list[dict[str, Any]]) -> list[str]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["check_family"]].append(row)
    lines = []
    for family, items in sorted(grouped.items()):
        comparisons = sum(int(item["n_comparisons"] or 0) for item in items)
        changed = sum(int(item["n_changed"] or 0) for item in items)
        lines.append(
            f"{family}: {len(items)} rows, {changed} of {comparisons} leader comparisons changed."
        )
    return lines


# --------------------------------------------------------------------------- #
# T7: null baselines on the same instances
# --------------------------------------------------------------------------- #


InstanceKey = tuple[str, str, str, str]


def build_t7(
    null_bundle: Bundle,
    null_records: list[ConfigRecord],
    reference: Bundle | None,
    reference_records: list[ConfigRecord],
) -> tuple[list[dict[str, Any]], list[str]]:
    lines: list[str] = []
    baselines: dict[str, dict[InstanceKey, dict[str, Any]]] = defaultdict(dict)
    same_run_real: dict[str, dict[InstanceKey, dict[str, Any]]] = defaultdict(dict)
    for record in null_records:
        target = baselines if common.is_null_baseline(record.explainer_id) else same_run_real
        for instance in record.instances or []:
            key = (record.dataset_id, record.track, record.model_id, str(instance["instance_id"]))
            target[record.explainer_id][key] = instance
    if not baselines:
        return [], [
            f"T7: bundle '{null_bundle.name}' has no random_reasons/global_ranking instance results "
            "(was it produced with the null-baseline config?)."
        ]
    reference_real: dict[str, dict[InstanceKey, dict[str, Any]]] = defaultdict(dict)
    for record in reference_records:
        if common.is_null_baseline(record.explainer_id):
            continue
        for instance in record.instances or []:
            key = (record.dataset_id, record.track, record.model_id, str(instance["instance_id"]))
            reference_real[record.explainer_id][key] = instance

    baseline_keys = set.intersection(*(set(table) for table in baselines.values()))
    reference_keys: set[InstanceKey] = set()
    for table in reference_real.values():
        reference_keys |= set(table)
    matched = baseline_keys & reference_keys
    matched_mode = bool(matched)
    key_set = matched if matched_mode else baseline_keys
    if reference is None:
        lines.append("T7: no reference bundle; showing baselines and same-run real explainers only.")
    elif not matched_mode:
        lines.append(
            f"T7 WARNING: no instance_id shared between '{null_bundle.name}' baselines and "
            f"'{reference.name}'; reference means below are NOT on matched instances."
        )
    else:
        lines.append(
            f"T7: {len(matched)} of {len(baseline_keys)} null-baseline instances matched by "
            f"instance_id to '{reference.name}'."
        )
    cslaw_config = (null_bundle.payload.get("cslaw") or {}).get("config") or {}
    reference_mode = (cslaw_config.get("null_baselines") or {}).get(
        "faithfulness_reference", "self (default)"
    )
    if null_bundle.common_reference:
        lines.append(
            "T7: null-baseline faithfulness is scored against the common native reference "
            "(v2 correction common_faithfulness_reference; the cslaw null_baselines "
            f"faithfulness_reference setting '{reference_mode}' only affects the self-reference "
            "kept under self_reference_signed_contributions), so it is comparable with the real "
            "explainers' faithfulness in the same bundle."
        )
    else:
        lines.append(
            f"T7: null-baseline faithfulness reference = {reference_mode}; with 'self' the "
            "baseline's faithfulness compares the baseline with itself and is not evidence of "
            "fidelity."
        )

    sources = (
        ("NULL_BASELINE", null_bundle.name, baselines),
        ("real_explainer", reference.name if reference is not None else None, reference_real),
        ("real_explainer_same_run", null_bundle.name, same_run_real),
    )
    rows: list[dict[str, Any]] = []
    datasets = sorted({key[0] for key in key_set})
    for scope in [POOLED] + datasets:
        scope_keys = key_set if scope == POOLED else {key for key in key_set if key[0] == scope}
        for role, bundle_name, tables in sources:
            for explainer_id in sorted(tables):
                table = tables[explainer_id]
                matched_scope = matched_mode or role == "NULL_BASELINE"
                if matched_scope:
                    keys = [key for key in scope_keys if key in table]
                else:
                    keys = [key for key in table if scope == POOLED or key[0] == scope]
                for metric in T7_METRICS:
                    values = [table[key][metric] for key in keys if table[key].get(metric) is not None]
                    rows.append(
                        {
                            "scope": scope,
                            "metric": metric,
                            "role": role,
                            "explainer_id": explainer_id,
                            "label": (
                                f"{explainer_id} [NULL BASELINE]"
                                if role == "NULL_BASELINE"
                                else explainer_id
                            ),
                            "source_bundle": bundle_name,
                            "matched_to_baseline_instances": matched_scope,
                            "n_instances": len(values),
                            "mean": (sum(float(v) for v in values) / len(values)) if values else None,
                        }
                    )
    return rows, lines


def t7_markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    pooled = [
        row for row in rows if row["scope"] == POOLED and row["role"] != "real_explainer_same_run"
    ]
    if not pooled:
        return []
    columns: list[str] = []
    for row in pooled:
        if row["label"] not in columns:
            columns.append(row["label"])
    values = {(row["metric"], row["label"]): row["mean"] for row in pooled}
    lines = ["| metric | " + " | ".join(columns) + " |", "|---|" + "---|" * len(columns)]
    for metric in T7_METRICS:
        lines.append(
            f"| {metric} | "
            + " | ".join(_fmt(values.get((metric, column))) for column in columns)
            + " |"
        )
    return lines


# --------------------------------------------------------------------------- #
# T8: v2 legal-function table
# --------------------------------------------------------------------------- #


def nested_bootstrap(
    tree: dict[str, Any],
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Percentile 95% CI of the pooled mean, resampling every level of a nested dict.

    Each dict level (sorted keys) is resampled with replacement, then the rows of
    each leaf array (instances x metrics; NaN = not applicable). With levels
    dataset > track > model this is the hierarchy of reporting's bootstrap.
    """

    def prepare(node: Any) -> Any:
        if isinstance(node, dict):
            return [prepare(node[key]) for key in sorted(node)]
        return node

    root = prepare(tree)
    first = root
    while isinstance(first, list):
        first = first[0]
    n_metrics = int(first.shape[1])
    stats = np.full((n_boot, n_metrics), np.nan)

    def draw(node: Any, total: np.ndarray, count: np.ndarray) -> None:
        if isinstance(node, list):
            for index in rng.integers(0, len(node), size=len(node)):
                draw(node[int(index)], total, count)
            return
        sample = node[rng.integers(0, node.shape[0], size=node.shape[0])]
        mask = ~np.isnan(sample)
        total += np.where(mask, sample, 0.0).sum(axis=0)
        count += mask.sum(axis=0)

    for replicate in range(n_boot):
        total = np.zeros(n_metrics)
        count = np.zeros(n_metrics)
        draw(root, total, count)
        valid = count > 0
        stats[replicate, valid] = total[valid] / count[valid]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        low = np.nanpercentile(stats, 2.5, axis=0)
        high = np.nanpercentile(stats, 97.5, axis=0)
    return low, high


def _tree_leaves(node: Any) -> list[np.ndarray]:
    if isinstance(node, dict):
        return [leaf for key in sorted(node) for leaf in _tree_leaves(node[key])]
    return [node]


def _concat_tree(node: Any) -> Any:
    if isinstance(node, dict):
        return {key: _concat_tree(value) for key, value in node.items()}
    return np.concatenate(node)


def t8_instance_values(record: ConfigRecord, instance: dict[str, Any]) -> tuple[list[float], str]:
    """(T8 metric values with NaN for not-applicable, CSS_v2 basis) for one instance."""
    css_v2 = instance.get("css_v2")
    basis = "directionality"
    if css_v2 is None:
        css_v2 = instance.get("css_no_dsv")
        basis = "css_no_dsv_fallback"
    recourse = is_recourse(record.explainer_id)
    faithfulness = (
        instance.get("faithfulness")
        if record.common_reference
        else instance.get("faithfulness_no_quality")
    )
    values = {
        "css_verifiability": instance.get("css_verifiability"),
        "css_correctability": instance.get("css_correctability"),
        "css_dsv_directionality": instance.get("css_dsv_directionality"),
        "css_reason_specificity": instance.get("css_reason_specificity"),
        "css_v2": css_v2,
        "comprehensibility_no_rs": instance.get("comprehensibility_no_rs"),
        "faithfulness_legal": faithfulness,
        "faithfulness_legal_ebm_only": faithfulness if record.model_id == "ebm" else None,
        "stability": instance.get("stability"),
        "alf_recourse_only": instance.get("alf") if recourse else None,
        "actionability_recourse_only": instance.get("actionability") if recourse else None,
    }
    return [np.nan if values[metric] is None else float(values[metric]) for metric in T8_METRICS], basis


def build_t8(
    bundle: Bundle,
    records: list[ConfigRecord],
    n_boot: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Legal-function metrics by explainer x dataset (+ pooled) with hierarchical CIs.

    CSS_v2 = 0.30V + 0.30C + 0.25 directionality + 0.15R (css_no_dsv where the
    v2 sub-component is not persisted); faithfulness_legal = the recorded score
    when the configuration used the common native reference, else
    faithfulness_no_quality (v1: still self-referential). The faithfulness rows
    carry the by-construction flags of their configurations (SHAP on logistic
    regression / XGBoost decomposes the native reference), the EBM-only column
    restricts to the model where no explainer does, and
    ``leader_excluding_by_construction`` names the best unflagged real explainer.
    ALF and unified actionability are reported for DiCE/Wachter only, because
    across families they are decided by construction. Null baselines are never
    ranked.
    """
    usable = [record for record in records if record.instances]
    if not usable:
        return [], [f"[{bundle.name}] T8 needs instance_results; the bundle has none."]
    native_ranking = regb.native_ranking_enabled(bundle.payload)
    from_changes = regb.cf_reasons_from_changes(bundle.payload)

    def cell_flags(explainer_id: str, model_ids: set[str]) -> str:
        flags: set[str] = set()
        for model_id in sorted(model_ids):
            flags.update(
                regb.by_construction_flags(
                    explainer_id,
                    model_id,
                    native_ranking=native_ranking,
                    cf_reasons_from_changes=from_changes,
                )[0]
            )
        return ";".join(sorted(flags))

    cells: dict[tuple[str, str], list[ConfigRecord]] = defaultdict(list)
    for record in usable:
        cells[(record.dataset_id, record.explainer_id)].append(record)
        cells[(POOLED, record.explainer_id)].append(record)

    rows: list[dict[str, Any]] = []
    for (scope, explainer_id), items in sorted(cells.items()):
        tree: dict[str, Any] = {}
        bases: Counter = Counter()
        for item in items:
            values: list[list[float]] = []
            for instance in item.instances or []:
                instance_values, basis = t8_instance_values(item, instance)
                values.append(instance_values)
                bases[basis] += 1
            array = np.array(values, dtype=float).reshape(-1, len(T8_METRICS))
            node = tree.setdefault(item.dataset_id, {}) if scope == POOLED else tree
            node = node.setdefault(item.track, {})
            node.setdefault(item.model_id, []).append(array)
        tree = _concat_tree(tree)
        stacked = np.concatenate(_tree_leaves(tree))
        mask = ~np.isnan(stacked)
        counts = [int(value) for value in mask.sum(axis=0)]
        sums = np.where(mask, stacked, 0.0).sum(axis=0)
        points = [
            float(sums[index] / counts[index]) if counts[index] else None
            for index in range(len(T8_METRICS))
        ]
        lows: np.ndarray | None = None
        highs: np.ndarray | None = None
        if n_boot > 0:
            rng = np.random.default_rng(
                [int(seed), zlib.crc32(f"T8|{bundle.name}|{scope}|{explainer_id}".encode("utf-8"))]
            )
            lows, highs = nested_bootstrap(tree, n_boot, rng)
            levels = "dataset > track > model > instance" if scope == POOLED else (
                "track > model > instance"
            )
            method = f"hierarchical bootstrap ({levels}), {n_boot} resamples, seed {seed}"
        else:
            method = "no bootstrap requested"
        css_basis = (
            next(iter(bases)) if len(bases) == 1 else "mixed:" + ";".join(
                f"{name}={count}" for name, count in sorted(bases.items())
            )
        )
        faith_bases = {
            "common_reference" if item.common_reference else "no_quality_self_referential_v1"
            for item in items
        }
        model_ids = {item.model_id for item in items}
        ebm_items = [item for item in items if item.model_id == "ebm"]

        def faithfulness_flags(cell_items: list[ConfigRecord], cell_models: set[str]) -> str:
            # Without the common reference the v1 score is self-referential (SHAP, LIME,
            # leave-one-out against their own outputs, Anchors against its own proxy,
            # DiCE/Wachter only structurally), so no such cell can be a "clean" leader.
            flags = cell_flags(explainer_id, cell_models)
            if any(not item.common_reference for item in cell_items):
                flags = ";".join(
                    sorted({*filter(None, flags.split(";")), "v1_self_referential_faithfulness"})
                )
            return flags

        for index, metric in enumerate(T8_METRICS):
            basis = ""
            flags = ""
            if metric == "css_v2":
                basis = css_basis
            elif metric == "faithfulness_legal":
                basis = ";".join(sorted(faith_bases))
                flags = faithfulness_flags(items, model_ids)
            elif metric == "faithfulness_legal_ebm_only":
                basis = ";".join(sorted(faith_bases)) + "; ebm configurations only"
                flags = faithfulness_flags(ebm_items, model_ids & {"ebm"})
            elif metric in T8_RECOURSE_ONLY:
                basis = "recourse family only" if is_recourse(explainer_id) else "not applicable"
            rows.append(
                {
                    "bundle": bundle.name,
                    "corrections_version": bundle.corrections_version,
                    "dataset_id": scope,
                    "explainer_id": explainer_id,
                    "family": family_of(explainer_id),
                    "is_null_baseline": common.is_null_baseline(explainer_id),
                    "metric": metric,
                    "basis": basis,
                    "by_construction_flags": flags,
                    "n_instances": counts[index],
                    "mean": points[index],
                    "ci_low": _finite_or_none(lows[index]) if lows is not None else None,
                    "ci_high": _finite_or_none(highs[index]) if highs is not None else None,
                    "ci_method": method,
                }
            )

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset_id"], row["metric"])].append(row)
    leaders: dict[tuple[str, str], dict[str, Any]] = {}
    for key, items in grouped.items():
        ranks = rescore.rank_real_explainers({row["explainer_id"]: row["mean"] for row in items})
        clean_leader = None
        if key[1] in T8_FAITHFULNESS:
            clean_ranks = rescore.rank_real_explainers(
                {
                    row["explainer_id"]: row["mean"]
                    for row in items
                    if not row["by_construction_flags"]
                }
            )
            clean_leader = next((e for e, rank in clean_ranks.items() if rank == 1), None)
        for row in items:
            row["rank"] = ranks.get(row["explainer_id"])
            row["is_leader"] = ranks.get(row["explainer_id"]) == 1
            row["runner_up"] = None
            row["leader_ci_separated_from_runner_up"] = None
            row["leader_excluding_by_construction"] = clean_leader
        leader = next((row for row in items if row["rank"] == 1), None)
        runner_up = next((row for row in items if row["rank"] == 2), None)
        if leader is not None:
            leaders[key] = leader
            if runner_up is not None:
                leader["runner_up"] = runner_up["explainer_id"]
                if leader["ci_low"] is not None and runner_up["ci_high"] is not None:
                    leader["leader_ci_separated_from_runner_up"] = (
                        leader["ci_low"] > runner_up["ci_high"]
                    )

    headlines: list[str] = []
    if any(not row["is_null_baseline"] for row in rows):
        parts = []
        for metric in T8_METRICS:
            leader = leaders.get((POOLED, metric))
            if leader is None:
                continue
            separated = leader["leader_ci_separated_from_runner_up"]
            clean = leader.get("leader_excluding_by_construction")
            parts.append(
                f"{metric} {leader['explainer_id']} {_fmt(leader['mean'])}"
                f" [{_fmt(leader['ci_low'])}, {_fmt(leader['ci_high'])}]"
                + (
                    ""
                    if separated is None
                    else f" (CI {'separated from' if separated else 'overlaps'} "
                    f"{leader['runner_up']})"
                )
                + (
                    f" [by construction: {leader['by_construction_flags']}; best unflagged: "
                    f"{clean or 'none'}]"
                    if metric in T8_FAITHFULNESS and leader["by_construction_flags"]
                    else ""
                )
            )
        headlines.append(f"[{bundle.name}] pooled leaders: " + "; ".join(parts) + ".")
        pooled_css = {row["basis"] for row in rows if row["dataset_id"] == POOLED and row["metric"] == "css_v2"}
        pooled_faith = {
            row["basis"] for row in rows if row["dataset_id"] == POOLED and row["metric"] == "faithfulness_legal"
        }
        headlines.append(
            f"[{bundle.name}] corrections {bundle.corrections_version}; CSS_v2 basis: "
            f"{', '.join(sorted(pooled_css))}; faithfulness basis: {', '.join(sorted(pooled_faith))}."
        )
    return rows, headlines


# --------------------------------------------------------------------------- #
# Output assembly
# --------------------------------------------------------------------------- #


def write_table(path: Path, rows: list[dict[str, Any]], empty_note: str) -> None:
    rescore.write_rows_csv(path, rows if rows else [{"note": empty_note}])


def parse_bundle_args(values: list[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for value in values:
        if "=" not in value:
            raise SystemExit(f"error: --bundle expects name=dir, got {value!r}")
        name, path_text = value.split("=", 1)
        name = name.strip()
        if not name or name in seen:
            raise SystemExit(f"error: bundle names must be unique and non-empty ({value!r})")
        seen.add(name)
        parsed.append((name, rescore.resolve_results_path(path_text.strip())))
    return parsed


def parse_pair_overrides(values: list[str], root: Path | None) -> dict[Path, Path]:
    def resolve(text: str) -> Path:
        path = Path(text.strip()).expanduser()
        if not path.is_absolute() and root is not None and (root / path).is_dir():
            path = root / path
        return path.resolve()

    overrides: dict[Path, Path] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"error: --robustness-pair expects CHECK=BASELINE, got {value!r}")
        check_text, baseline_text = value.split("=", 1)
        overrides[resolve(check_text)] = resolve(baseline_text)
    return overrides


def _bullets(lines: list[str]) -> list[str]:
    return [f"- {line}" for line in lines] if lines else ["- (nothing to report)"]


def generate_tables(
    bundles: list[Bundle],
    *,
    out_dir: Path,
    n_boot: int,
    seed: int,
    robustness_root: Path | None = None,
    rescored_root: Path | None = None,
    sweep_csvs: list[Path] | None = None,
    t7_reference: str | None = None,
    pair_overrides: dict[Path, Path] | None = None,
) -> list[Path]:
    """Compute T1-T8 for in-memory bundles and write them into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    records = {bundle.name: build_config_records(bundle) for bundle in bundles}
    main_bundles = [bundle for bundle in bundles if bundle.name != NULL_BASELINE_BUNDLE]
    sections: dict[str, list[str]] = defaultdict(list)
    written: list[Path] = []

    for bundle in bundles:
        notes = []
        if not bundle.has_instances:
            notes.append("no instance_results (summary-level fallbacks only)")
        if "rescore" in bundle.payload:
            notes.append(
                f"RESCORED bundle (variant {bundle.payload['rescore'].get('variant')}): its "
                "css/comprehensibility/faithfulness columns already use the variant"
            )
        notes.append(common.corrections_summary_line(bundle.payload))
        sections["inputs"].append(
            f"`{bundle.name}`: `{bundle.results_path}` (experiment `{bundle.experiment_name}`, "
            f"{len(bundle.summary)} summary rows)" + (f"; {'; '.join(notes)}" if notes else "")
        )

    t1_rows: list[dict[str, Any]] = []
    t2_rows: list[dict[str, Any]] = []
    t4_rows: list[dict[str, Any]] = []
    t5_rows: list[dict[str, Any]] = []
    t8_rows: list[dict[str, Any]] = []
    for bundle in bundles:
        # The null-baseline bundle contributes labelled, never-ranked rows to the
        # T2/T4/T5 CSVs; its headline lines would only say "leader None", so they
        # are replaced by one note per table.
        is_null_bundle = bundle.name == NULL_BASELINE_BUNDLE
        rows, lines = build_t1(bundle, records[bundle.name])
        t1_rows += rows
        sections["T1"] += [f"[{bundle.name}] {line}" for line in lines]
        rows, lines = build_t2(bundle, records[bundle.name])
        t2_rows += rows
        sections["T2"] += [] if is_null_bundle else lines
        rows, lines = build_t4(bundle, records[bundle.name], n_boot, seed)
        t4_rows += rows
        sections["T4"] += [] if is_null_bundle else lines
        rows, lines = build_t5(bundle, records[bundle.name])
        t5_rows += rows
        sections["T5"] += [] if is_null_bundle else lines
        rows, lines = build_t8(bundle, records[bundle.name], n_boot, seed)
        t8_rows += rows
        sections["T8"] += [] if is_null_bundle else lines
        if is_null_bundle:
            for table in ("T2", "T4", "T5", "T8"):
                sections[table].append(
                    f"[{bundle.name}] null-baseline rows are in the CSV (bundle={bundle.name}) "
                    "for reference only; they are never ranked or used for leaders or pooled "
                    "correlations."
                )
    sections["T5"].append(
        "Paper (configuration level, faithfulness vs CSS): +0.23 on the credit subset (n=120) and "
        "-0.45 on the non-credit subset (n=90); compare with the pooled rows above."
    )

    t3_records = [record for bundle in main_bundles for record in records[bundle.name]]
    t3_rows, t3_pairs = pairwise_recount(t3_records)
    sections["T3"] += t3_headlines(t3_rows)
    for bundle in main_bundles:
        sections["T3"].append(check_pairwise_against_reporting(bundle, t3_pairs))

    t6_rows: list[dict[str, Any]] = []
    for bundle in main_bundles:
        rows, notes = t6_main_bundle(bundle)
        t6_rows += rows
        sections["T6_notes"] += notes
    if robustness_root is not None:
        if robustness_root.is_dir():
            exclude = {bundle.directory.resolve() for bundle in bundles}
            if rescored_root is not None and rescored_root.is_dir():
                exclude.add(rescored_root.resolve())
            rows, notes = t6_robustness_rows(
                robustness_root, main_bundles, exclude, pair_overrides or {}
            )
            t6_rows += rows
            sections["T6_notes"] += notes
        else:
            sections["T6_notes"].append(f"--robustness-root {robustness_root} is not a directory.")
    sweep_paths = list(sweep_csvs or [])
    if rescored_root is not None:
        if rescored_root.is_dir():
            search_roots = (
                [robustness_root] if robustness_root is not None and robustness_root.is_dir() else []
            )
            rows, notes = t6_rescored_rows(rescored_root, main_bundles, search_roots)
            t6_rows += rows
            sections["T6_notes"] += notes
            for pattern in ("css_dsv_sweep.csv", "*/css_dsv_sweep.csv", "*/*/css_dsv_sweep.csv"):
                sweep_paths.extend(sorted(rescored_root.glob(pattern)))
        else:
            sections["T6_notes"].append(f"--rescored-root {rescored_root} is not a directory.")
    for path in dict.fromkeys(path.resolve() for path in sweep_paths):
        if path.is_file():
            t6_rows += t6_sweep_rows(path)
        else:
            sections["T6_notes"].append(f"Sweep CSV {path} not found.")
    sections["T6"] += t6_headlines(t6_rows)

    t7_rows: list[dict[str, Any]] = []
    null_bundle = next((bundle for bundle in bundles if bundle.name == NULL_BASELINE_BUNDLE), None)
    if null_bundle is not None:
        reference: Bundle | None = None
        if t7_reference:
            reference = next((bundle for bundle in main_bundles if bundle.name == t7_reference), None)
            if reference is None:
                sections["T7"].append(f"--t7-reference {t7_reference!r} is not a --bundle name.")
        if reference is None:
            reference = next((bundle for bundle in main_bundles if bundle.name == "credit"), None)
        if reference is None:
            reference = next(
                (
                    bundle
                    for bundle in main_bundles
                    if set(bundle.datasets) & set(null_bundle.datasets)
                ),
                None,
            )
        t7_rows, lines = build_t7(
            null_bundle,
            records[null_bundle.name],
            reference,
            records[reference.name] if reference is not None else [],
        )
        sections["T7"] += lines
        sections["T7_table"] = t7_markdown_table(t7_rows)
    else:
        sections["T7"].append(f"No --bundle named '{NULL_BASELINE_BUNDLE}' was given; T7 skipped.")

    outputs: dict[str, tuple[list[dict[str, Any]], str]] = {
        "T1_by_construction.csv": (t1_rows, "no structural facts computed"),
        "T2_css_decomposition.csv": (t2_rows, "no CSS rows"),
        "T3_pairwise_recount.csv": (t3_rows, "no pairwise comparisons"),
        "T3_pairwise_pairs.csv": (t3_pairs, "no pairwise comparisons"),
        "T4_per_domain.csv": (t4_rows, "no per-domain rows"),
        "T5_correlations.csv": (t5_rows, "no correlations"),
        "T6_robustness.csv": (t6_rows, "no robustness checks found"),
    }
    if null_bundle is not None:
        outputs["T7_null_baselines.csv"] = (t7_rows, "no null-baseline rows")
    outputs["T8_legal_functions.csv"] = (t8_rows, "no bundle with instance_results")
    for file_name, (rows, empty_note) in outputs.items():
        path = out_dir / file_name
        write_table(path, rows, empty_note)
        written.append(path)

    t1_md = out_dir / "T1_by_construction.md"
    t1_md.write_text(
        "\n".join(
            [
                "# T1: structural facts checked against the data",
                "",
                "Long format in T1_by_construction.csv (bundle, fact, group_1, group_2, statistic, value, n, note).",
                "",
                "- non_recourse_actionability_ceiling: unified actionability of explanations without "
                "counterfactual changes vs the definitional ceiling "
                f"{common.NON_RECOURSE_ACTIONABILITY_CEILING:g} (0.20*0.25 + 0.30*1.0).",
                "- decision_sensitivity_visibility_by_family, recourse_simplicity_by_family: distributions.",
                "- alf_applicability_by_explainer, explanation_quality_signal_by_explainer_model,",
                "  verifiability_saturation_by_dataset_track, dice_wachter_vcr_tie (per configuration).",
                "- corrections_status: corrections version and options recorded by the run, the "
                "share of instances whose faithfulness was scored on the common native reference "
                f"(`{common.COMMON_REFERENCE_SOURCE}`), and the faithfulness quality weight used.",
                "- cf_reason_source: per DiCE/Wachter instance, where its reasons come from "
                "(`counterfactual_changes`, `no_valid_change`; `not_recorded` in v1 bundles, whose "
                "reasons are a borrowed ranking).",
                "",
                "## Findings",
                "",
                *_bullets(sections["T1"]),
                "",
            ]
        ),
        encoding="utf-8",
    )
    written.append(t1_md)

    md_lines = [
        "# CS&Law paper tables",
        "",
        f"Bootstrap resamples: {n_boot}; seed: {seed}. Null baselines "
        f"({', '.join(common.NULL_BASELINES)}) are never ranked or reported as leaders.",
        "",
        "## Inputs",
        "",
        *_bullets(sections["inputs"]),
        "",
        "## T1 structural facts (T1_by_construction.csv/.md)",
        "",
        *_bullets(sections["T1"]),
        "",
        "## T2 CSS decomposition (T2_css_decomposition.csv)",
        "",
        "V, C, S (decision-sensitivity visibility), R, CSS and CSS_no_dsv = (0.30V + 0.30C + 0.15R)/0.75, "
        "evaluated_rows-weighted, by explainer x dataset and pooled (dataset_id = __pooled__).",
        "",
        *_bullets(sections["T2"]),
        "",
        "## T3 pairwise recount (T3_pairwise_recount.csv, T3_pairwise_pairs.csv)",
        "",
        "Configuration-level strict-> comparisons as in reporting._pairwise_wins. Categories: "
        "decided_by_construction (actionability/ALF, recourse vs non-recourse; faithfulness on the "
        "common reference when either side carries a T8 by-construction flag, e.g. SHAP on LR/XGB "
        "or native-ranked Anchors), formula_advantaged "
        "(CSS with S / comprehensibility with recourse simplicity, recourse vs non-recourse), "
        "not_comparable_evidence (faithfulness, direct reference vs structural proxy), open (everything else; "
        "faithfulness is open only when both sides use the common reference and neither is flagged).",
        "",
        *_bullets(sections["T3"]),
        "",
        "## T4 per-domain means (T4_per_domain.csv)",
        "",
        "Pooled instance means per dataset x explainer with 95% percentile CIs from a hierarchical "
        "bootstrap (track, then model, then instance). ALF averages applicable instances only. "
        "Leaders by metric (null baselines excluded):",
        "",
        *_bullets(sections["T4"]),
        "",
        "## T5 correlations (T5_correlations.csv)",
        "",
        *_bullets(sections["T5"]),
        "",
        "## T6 robustness (T6_robustness.csv)",
        "",
        *_bullets(sections["T6"]),
        *([""] + [f"- note: {line}" for line in sections["T6_notes"]] if sections["T6_notes"] else []),
        "",
        "## T7 null baselines (T7_null_baselines.csv)",
        "",
        *_bullets(sections["T7"]),
        "",
        *sections.get("T7_table", []),
        "",
        "## T8 legal-function table (T8_legal_functions.csv)",
        "",
        "One column per legal function of a statement of reasons, each family-neutral: "
        "V (verifiability), C (correctability), D (decision-sensitivity directionality: do the "
        "reasons state a direction?), R (reason specificity), CSS_v2 = 0.30V + 0.30C + 0.25D + "
        "0.15R, comprehensibility without recourse simplicity, faithfulness (recorded score when "
        "scored against the common native reference, v2; otherwise the quality-free v1 score, "
        "which is still self-referential and flagged `v1_self_referential_faithfulness`, so it "
        "never yields a `leader_excluding_by_construction`; `by_construction_flags` marks configurations whose "
        "agreement with the native reference is built in, e.g. SHAP on logistic regression and "
        "XGBoost, and the EBM-only column restricts to the model where no explainer decomposes "
        "the reference), stability, and ALF / unified actionability for "
        "DiCE and Wachter only (across families they are decided by construction). Instance "
        "means per dataset x explainer and pooled, with 95% percentile CIs from a hierarchical "
        "bootstrap (pooled: dataset > track > model > instance). `basis` says when a v1 bundle "
        "falls back (CSS_v2 -> css_no_dsv). Null baselines are listed, never ranked.",
        "",
        *_bullets(sections["T8"]),
        "",
    ]
    tables_md = out_dir / "tables.md"
    tables_md.write_text("\n".join(md_lines), encoding="utf-8")
    written.append(tables_md)
    return written


# --------------------------------------------------------------------------- #
# Self-test (synthetic data only)
# --------------------------------------------------------------------------- #


def run_selftest() -> None:
    explainers = ("anchors", "dice", "lime", "shap", "wachter")
    credit = Bundle(
        name="credit",
        results_path=Path("/nonexistent/credit/results.json"),
        payload=rescore.make_synthetic_payload(explainers=explainers, seed=11),
    )
    older = Bundle(
        name="older",
        results_path=Path("/nonexistent/older/results.json"),
        payload=rescore.make_synthetic_payload(explainers=explainers, seed=12, summary_only=True),
    )
    nulls = Bundle(
        name=NULL_BASELINE_BUNDLE,
        results_path=Path("/nonexistent/nulls/results.json"),
        payload=rescore.make_synthetic_payload(
            explainers=("global_ranking", "random_reasons", "shap"), seed=13
        ),
    )
    records = build_config_records(credit)

    t1_rows, _t1_lines = build_t1(credit, records)
    ceiling_rows = [
        row
        for row in t1_rows
        if row["fact"] == "non_recourse_actionability_ceiling" and row["statistic"] == "n_above_ceiling"
    ]
    assert ceiling_rows and all(row["value"] == 0 for row in ceiling_rows), ceiling_rows
    assert any(row["fact"] == "dice_wachter_vcr_tie" for row in t1_rows)
    older_records = build_config_records(older)
    older_t1, _ = build_t1(older, older_records)
    assert any(row["fact"] == "data_availability" for row in older_t1)

    t2_rows, _ = build_t2(credit, records)
    pooled_shap = next(
        row for row in t2_rows if row["dataset_id"] == POOLED and row["explainer_id"] == "shap"
    )
    expected = rescore.weighted_mean(
        [row for row in credit.summary if row["explainer_id"] == "shap"], "mean_css"
    )
    assert expected is not None and abs(pooled_shap["css"] - expected) < 1e-12

    t3_rows, t3_pairs = pairwise_recount(records)
    for row in t3_rows:
        assert row["wins"] + row["losses"] + row["ties"] == row["comparisons"]
        assert sum(row[f"comparisons_{category}"] for category in CATEGORIES) == row["comparisons"]
    categories = {
        (pair["metric"], pair["explainer_a"], pair["explainer_b"]): pair["category"]
        for pair in t3_pairs
    }
    assert categories[("actionability", "dice", "shap")] == "decided_by_construction"
    assert categories[("css", "anchors", "dice")] == "formula_advantaged"
    assert categories[("faithfulness", "dice", "lime")] == "not_comparable_evidence"
    assert categories[("stability", "dice", "shap")] == "open"
    assert categories[("alf", "dice", "wachter")] == "open"

    t4_first, _ = build_t4(credit, records, 50, 42)
    t4_second, _ = build_t4(credit, records, 50, 42)
    assert t4_first == t4_second, "bootstrap must be deterministic for a fixed seed"
    for row in t4_first:
        if row["ci_low"] is not None:
            assert row["ci_low"] <= row["ci_high"] + 1e-12
    assert all(row["ci_low"] is None for row in build_t4(older, older_records, 50, 42)[0])

    t5_rows, _ = build_t5(credit, records)
    assert any(row["level"] == "instance" for row in t5_rows)
    assert any(row["level"] == "configuration" and row["scope"] == "pooled" for row in t5_rows)

    t7_rows, t7_lines = build_t7(nulls, build_config_records(nulls), credit, records)
    assert t7_rows and all(row["matched_to_baseline_instances"] for row in t7_rows), t7_lines
    assert {row["role"] for row in t7_rows} >= {"NULL_BASELINE", "real_explainer"}

    # v2 bundle: common faithfulness reference, persisted directionality and
    # counterfactual reason sources.
    v2_payload = rescore.make_synthetic_payload(explainers=explainers, seed=14)
    for row in v2_payload["summary"]:
        row["faithfulness_evidence_sources"] = {common.COMMON_REFERENCE_SOURCE: row["evaluated_rows"]}
        for index, instance in enumerate(row["instance_results"]):
            fid = instance["scores"][rescore.FID_KEY]
            fid["evidence_source"] = common.COMMON_REFERENCE_SOURCE
            fid["faithfulness_quality_weight"] = 0.0
            instance["scores"][rescore.CSS_KEY]["dsv_directionality"] = 1.0
            if is_recourse(row["explainer_id"]):
                instance["canonical_explanation"] = {
                    "metadata": {
                        "cf_reason_source": "counterfactual_changes" if index else "no_valid_change"
                    }
                }
    v2_payload["cslaw"] = {"corrections": dict(common.CORRECTIONS_V2), "corrections_version": "v2"}
    v2 = Bundle(
        name="credit_v2", results_path=Path("/nonexistent/v2/results.json"), payload=v2_payload
    )
    assert v2.corrections_version == "v2" and v2.common_reference
    v2_records = build_config_records(v2)
    assert all(record.common_reference for record in v2_records)
    v2_by_explainer = {record.explainer_id: record for record in v2_records}
    v1_by_explainer = {record.explainer_id: record for record in records}
    assert comparison_category("faithfulness", v2_by_explainer["dice"], v2_by_explainer["lime"]) == "open"
    for flagged_model in ("logistic_regression", "xgboost"):
        shap_flagged = dataclasses.replace(v2_by_explainer["shap"], model_id=flagged_model)
        lime_same = dataclasses.replace(v2_by_explainer["lime"], model_id=flagged_model)
        assert (
            comparison_category("faithfulness", shap_flagged, lime_same) == "decided_by_construction"
        )
    assert (
        comparison_category("faithfulness", v1_by_explainer["dice"], v1_by_explainer["lime"])
        == "not_comparable_evidence"
    )
    v2_t1, _v2_lines = build_t1(v2, v2_records)
    status = {
        (row["group_1"], row["group_2"], row["statistic"]): row["value"]
        for row in v2_t1
        if row["fact"] == "corrections_status"
    }
    assert status[("bundle", "", "corrections_version")] == "v2", status
    cf_rows = [row for row in v2_t1 if row["fact"] == "cf_reason_source"]
    assert {row["group_2"] for row in cf_rows} == {"counterfactual_changes", "no_valid_change"}, cf_rows
    t8_rows, t8_lines = build_t8(v2, v2_records, 20, 3)
    assert t8_rows and t8_lines
    for row in t8_rows:
        if row["metric"] in T8_RECOURSE_ONLY and not is_recourse(row["explainer_id"]):
            assert row["mean"] is None and row["rank"] is None, row
        if row["metric"] == "css_v2":
            assert row["basis"] == "directionality", row
        if row["metric"] == "faithfulness_legal":
            assert row["basis"] == "common_reference", row
        if row["ci_low"] is not None:
            assert row["ci_low"] <= row["ci_high"] + 1e-12
    shap_css = [
        0.30 * instance["scores"][rescore.CSS_KEY]["verifiability"]
        + 0.30 * instance["scores"][rescore.CSS_KEY]["correctability"]
        + 0.25 * 1.0
        + 0.15 * instance["scores"][rescore.CSS_KEY]["reason_specificity"]
        for row in v2_payload["summary"]
        if row["explainer_id"] == "shap"
        for instance in row["instance_results"]
    ]
    pooled_shap_v2 = next(
        row
        for row in t8_rows
        if row["dataset_id"] == POOLED and row["explainer_id"] == "shap" and row["metric"] == "css_v2"
    )
    assert abs(pooled_shap_v2["mean"] - sum(shap_css) / len(shap_css)) < 1e-12
    v1_t8, _ = build_t8(credit, records, 0, 1)
    assert all(row["basis"] == "css_no_dsv_fallback" for row in v1_t8 if row["metric"] == "css_v2")
    assert all(
        row["basis"] == "no_quality_self_referential_v1"
        for row in v1_t8
        if row["metric"] == "faithfulness_legal"
    )
    null_t8, _ = build_t8(nulls, build_config_records(nulls), 0, 1)
    assert all(row["rank"] is None for row in null_t8 if row["is_null_baseline"])

    with tempfile.TemporaryDirectory() as tmp:
        written = generate_tables([credit, older, nulls, v2], out_dir=Path(tmp), n_boot=20, seed=1)
        names = {path.name for path in written}
        assert {
            "tables.md",
            "T1_by_construction.csv",
            "T7_null_baselines.csv",
            "T8_legal_functions.csv",
        } <= names, names
    print("paper_tables.py selftest: OK")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description="CS&Law revision tables T1-T8 from result bundles.")
    parser.add_argument(
        "--bundle",
        action="append",
        default=[],
        help="name=dir (bundle directory or results.json); repeatable. Name 'nullbaselines' enables T7.",
    )
    parser.add_argument("--robustness-root", help="Directory whose subdirectories are robustness bundles.")
    parser.add_argument("--rescored-root", help="Directory with rescore.py outputs (variants and sweep CSVs).")
    parser.add_argument("--sweep-csv", action="append", default=[], help="Extra css_dsv_sweep.csv path(s).")
    parser.add_argument(
        "--robustness-pair",
        action="append",
        default=[],
        help="CHECK_DIR=BASELINE_DIR override for T6 (names under --robustness-root or paths).",
    )
    parser.add_argument(
        "--t7-reference",
        help="Bundle name whose real explainers are joined in T7 (default: 'credit').",
    )
    parser.add_argument("--out-dir", help="Output directory for the tables.")
    parser.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap resamples for T4 (default 1000).")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the T4 bootstrap (default 42).")
    parser.add_argument("--selftest", action="store_true", help="Run the synthetic self-test and exit.")
    args = parser.parse_args()

    if args.selftest:
        run_selftest()
        return
    if not args.bundle or not args.out_dir:
        parser.error("--bundle (at least one) and --out-dir are required")

    bundles = [
        Bundle(name=name, results_path=path, payload=common.load_results(path))
        for name, path in parse_bundle_args(args.bundle)
    ]
    out_dir = Path(args.out_dir).expanduser().resolve()
    for bundle in bundles:
        if out_dir == bundle.directory.resolve():
            raise SystemExit(f"error: --out-dir {out_dir} is the bundle directory of '{bundle.name}'.")
    robustness_root = (
        Path(args.robustness_root).expanduser().resolve() if args.robustness_root else None
    )
    rescored_root = Path(args.rescored_root).expanduser().resolve() if args.rescored_root else None
    written = generate_tables(
        bundles,
        out_dir=out_dir,
        n_boot=max(0, int(args.bootstrap)),
        seed=int(args.seed),
        robustness_root=robustness_root,
        rescored_root=rescored_root,
        sweep_csvs=[Path(path).expanduser() for path in args.sweep_csv],
        t7_reference=args.t7_reference,
        pair_overrides=parse_pair_overrides(args.robustness_pair, robustness_root),
    )
    print("Wrote:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
