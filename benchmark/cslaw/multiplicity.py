"""Reason multiplicity across models (and disagreement across explainers).

Question: when the same applicant is denied by several comparably accurate
models, do the principal reasons on the adverse-action notice depend on which
model (and which explainer) the lender happened to choose?

Legal grounding (our reading, stated cautiously; not legal advice)
------------------------------------------------------------------
Regulation B, 12 CFR 1002.9(b)(2), requires specific principal reasons for
adverse action; Official Interpretation comment 9(b)(2)-2 says the reasons
disclosed must relate to and accurately describe the factors actually
considered or scored, and comment 9(b)(2)-4 says no factor that was a principal
reason may be excluded. Comment 9(b)(2)-5 describes acceptable selection
methods for credit scoring systems, including (method 1) comparing the
applicant's per-factor score with the average among applicants at or slightly
above the minimum passing score and (method 2) with the average among all
applicants. Reasons are therefore tied to the particular scoring system used,
so if equally defensible models yield different reasons for the same applicant,
the notice partly reflects an arbitrary model choice ("model multiplicity").

What is measured
----------------
* Cross-model multiplicity, per (dataset, track, explainer, eval_row_index)
  explained under every model in the bundle: pairwise Jaccard of the
  principal-reason feature sets across models, whether all models agree on the
  top reason, and the share of applicants whose top reason differs across at
  least two models.
* The same for the Reg B commentary reasons ("regb_m1" = method 1 with the
  near-cutoff reference, "regb_m2" = method 2 with the all-applicants
  reference), computed from each model's native contributions
  (log-odds of the adverse class) and its reference means. This measures the
  multiplicity of the legally sanctioned reasons independently of any
  explainer. The operationalization (largest positive gap between applicant
  contribution and reference mean) is ours, not a regulator-endorsed algorithm.
* Across-explainer disagreement within a model (the "disagreement problem"):
  pairwise Jaccard and top-1 agreement between explainers on the same
  applicant and model. Null baselines (random_reasons, global_ranking) are
  reported in the pair table but excluded from the all-real-explainer summary.
* Same basis for every method: each reason list, explainer or Reg B, is cut to
  its first k units before Jaccard/top-1 (Jaccard depends on set size). With
  --unit grouped (default) a unit is one notice reason: a PAY_*/BILL_AMT*
  temporal family counts once, and the Reg B gap ranking is mapped to the same
  units before taking the top k (regb_reference.regb_unit_ranking). With
  --unit feature_at_k a unit is one feature: the explainer's first k flattened
  features vs the Reg B top-k features. Every output row carries a
  ``reason_unit`` column. Explainers that cite fewer than k units (Anchors,
  counterfactuals) still give smaller sets.

Requires a bundle produced with cslaw.shared_sampling.enabled=true (scope
"models" or "models_and_tracks") plus cslaw.persist_canonical_explanations=true;
the Reg B rows additionally need persist_native_contributions and
regb_reference.enabled.

v2 bundles (cslaw corrections): reason units use the bundle's corrected
temporal grouping (PAY_AMT* is its own "recent payment amounts" family), and
DiCE/Wachter reasons are the features their counterfactuals change rather than
a ranking borrowed from the native contributions, so their cross-model
multiplicity reflects the counterfactual search, not the attribution ranking.

Usage
-----
    python benchmark/cslaw/multiplicity.py --bundle <shared-sampling run dir> \
        --out-dir <dir> [--reason-set principal] [--unit grouped] [--k 4] \
        [--n-boot 2000] [--seed 42]
"""

from __future__ import annotations

import argparse
import itertools
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import regb_reference as regb  # noqa: E402

REGB_METHODS = {"regb_m1": "near_cutoff", "regb_m2": "all_applicants"}
# "grouped": one unit per notice reason (temporal family counted once);
# "feature_at_k": one unit per feature (temporal groups expanded).
REASON_UNITS = ("grouped", "feature_at_k")

CROSS_MODEL_METRICS = (
    "mean_pairwise_jaccard",
    "min_pairwise_jaccard",
    "all_top1_agree",
    "top1_differs",
    "n_distinct_top1",
)
CROSS_EXPLAINER_METRICS = (
    "mean_pairwise_jaccard_real",
    "min_pairwise_jaccard_real",
    "all_top1_agree_real",
    "top1_differs_real",
    "n_distinct_top1_real",
)
PAIR_METRICS = ("jaccard", "top1_agree")


@dataclass
class ReasonIndex:
    """Reason feature lists keyed two ways, plus bookkeeping."""

    by_method: dict[tuple[str, str, str, int], dict[str, list[str]]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    by_model: dict[tuple[str, str, str, int], dict[str, list[str]]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    method_kind: dict[str, str] = field(default_factory=dict)
    models_by_dataset_track: dict[tuple[str, str], set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )
    coverage: Counter = field(default_factory=Counter)
    unit: str = "grouped"
    fixed_grouping: bool = False


def explainer_reason_list(
    canonical: dict[str, Any] | None,
    reason_set: str,
    k: int,
    unit: str,
    *,
    fixed_grouping: bool = False,
) -> list[str]:
    """The explainer's first k reason units (see REASON_UNITS).

    ``fixed_grouping`` must match the bundle (regb_reference.grouping_fixed).
    """
    features = regb.reason_features(canonical, reason_set)
    if unit == "grouped":
        return regb.grouped_units(features, k, fixed=fixed_grouping)
    return features[:k]


def regb_reason_list(
    values: dict[str, float],
    means: dict[str, float],
    k: int,
    unit: str,
    *,
    fixed_grouping: bool = False,
) -> list[str]:
    """The Reg B top-k reason units on the same basis as ``explainer_reason_list``."""
    if unit == "grouped":
        return regb.regb_unit_ranking(values, means, k, fixed=fixed_grouping) or []
    return common.top_adverse_features(values, k=k, reference=means)


def shared_sampling_groups(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return {group_key: shared_sampling record}; empty if the run did not share."""
    return {
        key: group["shared_sampling"]
        for key, group in regb.cslaw_groups(payload).items()
        if isinstance(group, dict) and isinstance(group.get("shared_sampling"), dict)
    }


def sampling_status_rows(shared: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in sorted(shared):
        dataset_id, track, model_id = (key.split("|") + ["", "", ""])[:3]
        row: dict[str, Any] = {"dataset_id": dataset_id, "track": track, "model_id": model_id}
        for name, value in shared[key].items():
            if name == "selected_positions":
                row["n_selected_positions"] = len(value) if isinstance(value, list) else None
            elif isinstance(value, dict):
                for sub_name, sub_value in value.items():
                    if not isinstance(sub_value, (dict, list)):
                        row[f"{name}.{sub_name}"] = sub_value
            elif isinstance(value, list):
                row[name] = "|".join(str(item) for item in value)
            else:
                row[name] = value
        rows.append(row)
    return rows


def collect_reason_index(
    payload: dict[str, Any],
    *,
    reason_set: str = "principal",
    k: int = 4,
    unit: str = "grouped",
    fixed_grouping: bool | None = None,
) -> ReasonIndex:
    if unit not in REASON_UNITS:
        raise ValueError(f"unknown reason unit: {unit}")
    if fixed_grouping is None:
        fixed_grouping = regb.grouping_fixed(payload)
    index = ReasonIndex(unit=unit, fixed_grouping=fixed_grouping)
    for summary_row in payload.get("summary", []):
        index.models_by_dataset_track[(summary_row["dataset_id"], summary_row["track"])].add(
            summary_row["model_id"]
        )
    native_done: set[tuple[str, str, str, int]] = set()
    for summary_row, instance in common.iter_instances(payload):
        index.coverage["instances"] += 1
        dataset_id, track, model_id, explainer_id = common.group_key(summary_row)
        row_index = common.eval_row_index(instance)
        if row_index is None:
            index.coverage["skipped_missing_eval_row_index"] += 1
            continue
        canonical = instance.get("canonical_explanation")
        if not canonical:
            index.coverage["skipped_missing_canonical_explanation"] += 1
        else:
            features = explainer_reason_list(
                canonical, reason_set, k, unit, fixed_grouping=fixed_grouping
            )
            method_key = (dataset_id, track, explainer_id, row_index)
            if model_id in index.by_method[method_key]:
                index.coverage["duplicate_instances_ignored"] += 1
            else:
                index.by_method[method_key][model_id] = features
                index.by_model[(dataset_id, track, model_id, row_index)][explainer_id] = features
                index.method_kind[explainer_id] = (
                    "null_baseline" if common.is_null_baseline(explainer_id) else "explainer"
                )
                index.coverage["used_explainer_instances"] += 1
        applicant_key = (dataset_id, track, model_id, row_index)
        if applicant_key in native_done:
            continue
        values = regb.native_values(instance)
        if values is None:
            continue
        native_done.add(applicant_key)
        for method, population in REGB_METHODS.items():
            means = regb.regb_means(payload, dataset_id, track, model_id, population)
            if means is None:
                index.coverage[f"missing_regb_reference_{population}"] += 1
                continue
            index.by_method[(dataset_id, track, method, row_index)][model_id] = (
                regb_reason_list(values, means, k, unit, fixed_grouping=fixed_grouping)
            )
            index.method_kind[method] = "regb_reference"
            index.coverage[f"used_{method}_applicants"] += 1
    return index


def _top1(features: list[str]) -> str | None:
    return features[0] if features else None


def _agreement(top1: Sequence[str | None]) -> tuple[bool | None, int | None]:
    if not top1 or any(value is None for value in top1):
        return None, None
    distinct = len(set(top1))
    return distinct == 1, distinct


def cross_model_rows(index: ReasonIndex) -> tuple[list[dict[str, Any]], Counter]:
    """One row per (dataset, track, method, applicant) explained under all models."""
    rows: list[dict[str, Any]] = []
    dropped: Counter = Counter()
    for key in sorted(index.by_method, key=lambda item: tuple(str(part) for part in item)):
        dataset_id, track, method, row_index = key
        per_model = index.by_method[key]
        required = sorted(index.models_by_dataset_track.get((dataset_id, track), set()))
        if len(required) < 2:
            dropped["fewer_than_two_models_in_bundle"] += 1
            continue
        if any(model_id not in per_model for model_id in required):
            dropped[f"not_explained_under_all_models:{method}"] += 1
            continue
        pairwise = {
            (left, right): common.jaccard(per_model[left], per_model[right])
            for left, right in itertools.combinations(required, 2)
        }
        jaccards = [value for value in pairwise.values() if value is not None]
        top1 = {model_id: _top1(per_model[model_id]) for model_id in required}
        all_agree, distinct = _agreement(list(top1.values()))
        row: dict[str, Any] = {
            "dataset_id": dataset_id,
            "track": track,
            "method": method,
            "method_kind": index.method_kind.get(method, "explainer"),
            "eval_row_index": row_index,
            "reason_unit": index.unit,
            "n_models": len(required),
            "models": "|".join(required),
            "mean_pairwise_jaccard": common.mean(jaccards),
            "min_pairwise_jaccard": min(jaccards) if jaccards else None,
            "all_top1_agree": all_agree,
            "top1_differs": None if all_agree is None else not all_agree,
            "n_distinct_top1": distinct,
        }
        for model_id in required:
            row[f"top1__{model_id}"] = top1[model_id]
        for (left, right), value in pairwise.items():
            row[f"jaccard__{left}__{right}"] = value
        for model_id in required:
            row[f"reasons__{model_id}"] = regb.join_features(per_model[model_id])
        rows.append(row)
    return rows, dropped


def cross_explainer_rows(
    index: ReasonIndex,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Within-model disagreement: (applicant rows over real explainers, pair rows)."""
    applicant_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for key in sorted(index.by_model, key=lambda item: tuple(str(part) for part in item)):
        dataset_id, track, model_id, row_index = key
        per_explainer = index.by_model[key]
        explainers = sorted(per_explainer)
        for left, right in itertools.combinations(explainers, 2):
            involves_null = common.is_null_baseline(left) or common.is_null_baseline(right)
            pair_rows.append(
                {
                    "dataset_id": dataset_id,
                    "track": track,
                    "model_id": model_id,
                    "eval_row_index": row_index,
                    "explainer_a": left,
                    "explainer_b": right,
                    "pair_kind": "involves_null_baseline" if involves_null else "real_vs_real",
                    "reason_unit": index.unit,
                    "jaccard": common.jaccard(per_explainer[left], per_explainer[right]),
                    "top1_agree": common.top1_match(per_explainer[left], per_explainer[right]),
                }
            )
        real = [name for name in explainers if not common.is_null_baseline(name)]
        if len(real) < 2:
            continue
        jaccards = [
            value
            for value in (
                common.jaccard(per_explainer[left], per_explainer[right])
                for left, right in itertools.combinations(real, 2)
            )
            if value is not None
        ]
        all_agree, distinct = _agreement([_top1(per_explainer[name]) for name in real])
        row: dict[str, Any] = {
            "dataset_id": dataset_id,
            "track": track,
            "model_id": model_id,
            "eval_row_index": row_index,
            "reason_unit": index.unit,
            "n_real_explainers": len(real),
            "real_explainers": "|".join(real),
            "mean_pairwise_jaccard_real": common.mean(jaccards),
            "min_pairwise_jaccard_real": min(jaccards) if jaccards else None,
            "all_top1_agree_real": all_agree,
            "top1_differs_real": None if all_agree is None else not all_agree,
            "n_distinct_top1_real": distinct,
        }
        for name in real:
            row[f"top1__{name}"] = _top1(per_explainer[name])
        applicant_rows.append(row)
    return applicant_rows, pair_rows


def model_performance_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    rows: list[dict[str, Any]] = []
    for summary_row in payload.get("summary", []):
        key = (summary_row["dataset_id"], summary_row["track"], summary_row["model_id"])
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "dataset_id": key[0],
                "track": key[1],
                "model_id": key[2],
                "model_backend": regb.model_backend_kind(summary_row),
                "roc_auc": summary_row.get("model_roc_auc"),
                "average_precision": summary_row.get("model_average_precision"),
                "accuracy": summary_row.get("model_accuracy"),
                "adverse_rate": summary_row.get("adverse_rate"),
                "decision_threshold": summary_row.get("model_decision_threshold"),
            }
        )
    return sorted(rows, key=lambda row: (row["dataset_id"], row["track"], row["model_id"]))


def summarize_multiplicity(
    cross_model: Sequence[dict[str, Any]],
    applicants: Sequence[dict[str, Any]],
    pairs: Sequence[dict[str, Any]],
    *,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    return {
        "cross_model": regb.summarize_levels(
            cross_model,
            levels=(
                ("dataset_track_method", ("dataset_id", "track", "method_kind", "method"), ()),
                ("dataset_method", ("dataset_id", "method_kind", "method"), ("track",)),
                ("method", ("method_kind", "method"), ("dataset_id", "track")),
            ),
            all_cols=("dataset_id", "track", "method_kind", "method"),
            metrics=CROSS_MODEL_METRICS,
            n_boot=n_boot,
            seed=seed,
        ),
        "cross_explainer": regb.summarize_levels(
            applicants,
            levels=(
                ("dataset_track_model", ("dataset_id", "track", "model_id"), ()),
                ("dataset_model", ("dataset_id", "model_id"), ("track",)),
                ("model", ("model_id",), ("dataset_id", "track")),
                ("all", (), ("dataset_id", "track", "model_id")),
            ),
            all_cols=("dataset_id", "track", "model_id"),
            metrics=CROSS_EXPLAINER_METRICS,
            n_boot=n_boot,
            seed=seed,
        ),
        "pairs": regb.summarize_levels(
            pairs,
            levels=(
                (
                    "dataset_model_pair",
                    ("dataset_id", "model_id", "explainer_a", "explainer_b", "pair_kind"),
                    ("track",),
                ),
                (
                    "model_pair",
                    ("model_id", "explainer_a", "explainer_b", "pair_kind"),
                    ("dataset_id", "track"),
                ),
                (
                    "pair",
                    ("explainer_a", "explainer_b", "pair_kind"),
                    ("dataset_id", "track", "model_id"),
                ),
            ),
            all_cols=("dataset_id", "model_id", "explainer_a", "explainer_b", "pair_kind"),
            metrics=PAIR_METRICS,
            n_boot=n_boot,
            seed=seed,
        ),
    }


def _tag_unit(rows: Sequence[dict[str, Any]], unit: str) -> list[dict[str, Any]]:
    """Add the reason unit to summary rows so every CSV states its basis."""
    return [{**row, "reason_unit": unit} for row in rows]


def _method_order(method: str, kind: str) -> tuple[int, str]:
    rank = {"regb_reference": 0, "explainer": 1, "null_baseline": 2}.get(kind, 3)
    return (rank, method)


def _method_label(method: str, kind: str) -> str:
    return method + (" (null baseline)" if kind == "null_baseline" else "")


def _fmt_number(value: Any) -> str:
    return f"{value:.3f}" if isinstance(value, (int, float)) and not isinstance(value, bool) else "n/a"


def render_multiplicity_markdown(
    summaries: dict[str, list[dict[str, Any]]],
    *,
    bundle: str,
    reason_set: str,
    k: int,
    coverage: Counter,
    dropped: Counter,
    status_rows: Sequence[dict[str, Any]],
    performance_rows: Sequence[dict[str, Any]],
    output_paths: Sequence[Path],
    unit: str = "grouped",
    settings: str | None = None,
) -> str:
    cross_model = summaries["cross_model"]
    method_level = [row for row in cross_model if row["level"] == "method"]
    method_index = regb.index_summary(method_level, ("method_kind", "method"))
    methods = sorted(
        {(row["method_kind"], row["method"]) for row in method_level},
        key=lambda item: _method_order(item[1], item[0]),
    )
    method_rows = []
    for kind, method in methods:
        jac = method_index.get((kind, method, "mean_pairwise_jaccard"))
        method_rows.append(
            [
                _method_label(method, kind),
                kind,
                regb.fmt_stat(jac),
                regb.fmt_stat(method_index.get((kind, method, "top1_differs"))),
                regb.fmt_stat(method_index.get((kind, method, "n_distinct_top1")), digits=2),
                jac["n"] if jac else 0,
            ]
        )
    dataset_level = [row for row in cross_model if row["level"] == "dataset_method"]
    dataset_index = regb.index_summary(dataset_level, ("dataset_id", "method_kind", "method"))
    dataset_rows = []
    for dataset_id in sorted({row["dataset_id"] for row in dataset_level}):
        for kind, method in methods:
            jac = dataset_index.get((dataset_id, kind, method, "mean_pairwise_jaccard"))
            if not jac:
                continue
            dataset_rows.append(
                [
                    dataset_id,
                    _method_label(method, kind),
                    regb.fmt_stat(jac),
                    regb.fmt_stat(dataset_index.get((dataset_id, kind, method, "top1_differs"))),
                    jac["n"],
                ]
            )
    explainer_level = [
        row for row in summaries["cross_explainer"] if row["level"] in ("model", "all")
    ]
    explainer_index = regb.index_summary(explainer_level, ("model_id",))
    explainer_rows = [
        [
            "all models" if model_id == regb.POOLED else model_id,
            regb.fmt_stat(explainer_index.get((model_id, "mean_pairwise_jaccard_real"))),
            regb.fmt_stat(explainer_index.get((model_id, "top1_differs_real"))),
            (explainer_index.get((model_id, "mean_pairwise_jaccard_real")) or {}).get("n", 0),
        ]
        for model_id in sorted(
            {row["model_id"] for row in explainer_level},
            key=lambda value: (value == regb.POOLED, value),
        )
    ]
    pair_rows = [
        [row["explainer_a"], row["explainer_b"], row["pair_kind"], regb.fmt_stat(row), row["n"]]
        for row in summaries["pairs"]
        if row["level"] == "pair" and row["metric"] == "jaccard"
    ]
    pair_rows.sort(key=lambda item: (item[2] != "real_vs_real", item[0], item[1]))
    scored = [
        (method_index[(kind, method, "top1_differs")]["mean"], method)
        for kind, method in methods
        if kind == "explainer" and (kind, method, "top1_differs") in method_index
    ]
    leader = min(scored) if scored else None
    status_columns: list[str] = []
    for row in status_rows:
        for column in row:
            if column not in status_columns:
                status_columns.append(column)
    status_table = (
        regb.md_table(
            status_columns,
            [[row.get(column, "") for column in status_columns] for row in status_rows],
        )
        if status_rows
        else "_No shared_sampling records._"
    )
    fallback = sorted(
        {
            row["dataset_id"]
            for row in status_rows
            if str(row.get("shared_fallback")).lower() == "true"
        }
    )
    lines = [
        "# Reason multiplicity across models and explainers",
        "",
        f"Bundle: `{bundle}`; reason set: `{reason_set}`; reason unit: `{unit}`; k = {k}.",
        "",
        *([f"Settings: {settings}", ""] if settings else []),
        "**Same basis for every method.** Every reason list, explainer and Reg B alike, is cut "
        f"to its first {k} units before Jaccard and top-1 are computed, because Jaccard depends "
        "on set size. "
        + (
            "Unit `grouped`: one unit per notice reason; a PAY_*/BILL_AMT* temporal family counts "
            "once, and the Reg B gap ranking is mapped to the same units before taking the top "
            f"{k}. Top-1 is the first notice reason."
            if unit == "grouped"
            else "Unit `feature_at_k`: one unit per feature; the explainer's first "
            f"{k} principal-reason features (temporal groups expanded) vs the Reg B top-{k} "
            "features. Top-1 is the first feature."
        )
        + " Explainers that cite fewer units (Anchors, counterfactuals) still give smaller sets.",
        "",
        "**Why this matters (our reading; not legal advice).** Reg B (12 CFR 1002.9(b)(2)) "
        "requires specific principal reasons, and comment 9(b)(2)-2 requires them to relate to "
        "the factors actually considered or scored by the creditor's system. If comparably "
        "accurate models give the same applicant different principal reasons, the notice partly "
        "reflects an arbitrary model choice. `regb_m1`/`regb_m2` apply our operationalization of "
        "the comment 9(b)(2)-5 methods (largest positive gap between the applicant's native "
        "log-odds contribution and the near-cutoff / all-applicant reference mean) to each "
        "model; this measures multiplicity of the commentary-style reasons independent of any "
        "explainer. The operationalization is ours, not a regulator-endorsed algorithm.",
        "",
        "## Shared sampling status",
        "",
        status_table,
        "",
        (
            "Datasets that fell back to per-model sampling (cross-model sets incomplete): "
            f"{', '.join(fallback)}."
            if fallback
            else "No dataset fell back to per-model sampling."
        ),
        "",
        "## Model performance (multiplicity is most meaningful among comparably accurate models)",
        "",
        regb.md_table(
            ["dataset", "track", "model", "backend", "ROC AUC", "adverse rate"],
            [
                [
                    row["dataset_id"],
                    row["track"],
                    row["model_id"],
                    row["model_backend"],
                    _fmt_number(row["roc_auc"]),
                    _fmt_number(row["adverse_rate"]),
                ]
                for row in performance_rows
            ],
        ),
        "",
        "## Cross-model multiplicity, pooled over datasets and tracks",
        "",
        "Mean pairwise Jaccard of reason sets across models (1 = identical), share of applicants "
        "whose top reason differs across at least two models, and mean number of distinct top "
        "reasons; 95% bootstrap CIs stratified by dataset x track.",
        "",
        regb.md_table(
            [
                "method",
                "kind",
                "mean pairwise Jaccard",
                "top-1 differs",
                "distinct top-1",
                "n applicants",
            ],
            method_rows,
        )
        if method_rows
        else "_No applicant was explained under all models._",
        "",
        (
            "Most model-consistent real explainer (lowest top-1-differs share; null baselines "
            f"excluded): `{leader[1]}` ({leader[0]:.3f})."
            if leader
            else "No real explainer rows."
        ),
        "",
        "## Cross-model multiplicity by dataset",
        "",
        regb.md_table(
            ["dataset", "method", "mean pairwise Jaccard", "top-1 differs", "n"], dataset_rows
        )
        if dataset_rows
        else "_None._",
        "",
        "## Across-explainer disagreement within a model (real explainers only)",
        "",
        regb.md_table(
            ["model", "mean pairwise Jaccard", "top-1 differs", "n applicants"], explainer_rows
        )
        if explainer_rows
        else "_Fewer than two real explainers per applicant._",
        "",
        "### Explainer pairs (pooled over datasets, tracks and models)",
        "",
        regb.md_table(["explainer A", "explainer B", "pair kind", "mean Jaccard", "n"], pair_rows)
        if pair_rows
        else "_None._",
        "",
        "## Coverage",
        "",
        regb.md_table(["count", "value"], sorted((coverage + dropped).items())),
        "",
        "## Files",
        "",
        *[f"- `{path}`" for path in output_paths],
        "",
    ]
    return "\n".join(lines)


def run(
    bundle: str | Path,
    out_dir: str | Path,
    *,
    reason_set: str = "principal",
    k: int = 4,
    n_boot: int = 2000,
    seed: int = 42,
    unit: str = "grouped",
) -> list[Path]:
    payload, results_path = regb.load_bundle(bundle)
    label = regb.bundle_label(bundle)
    shared = shared_sampling_groups(payload)
    if not shared:
        raise SystemExit(
            f"multiplicity: {results_path} has no "
            "payload['cslaw']['groups'][...]['shared_sampling'] records. Cross-model "
            "multiplicity needs every model to explain the SAME applicants: re-run with "
            "cslaw.shared_sampling.enabled=true (scope 'models' or 'models_and_tracks') and "
            "cslaw.persist_canonical_explanations=true."
        )
    index = collect_reason_index(payload, reason_set=reason_set, k=k, unit=unit)
    if not index.by_method:
        raise SystemExit(
            "multiplicity: no persisted canonical explanations or native contributions in "
            f"{results_path} ({dict(index.coverage)}). Re-run with "
            "cslaw.persist_canonical_explanations=true (and persist_native_contributions=true "
            "plus regb_reference.enabled=true for the Reg B rows)."
        )
    cross_model, dropped = cross_model_rows(index)
    applicants, pairs = cross_explainer_rows(index)
    summaries = summarize_multiplicity(cross_model, applicants, pairs, n_boot=n_boot, seed=seed)
    status_rows = sampling_status_rows(shared)
    performance_rows = model_performance_rows(payload)
    out = Path(out_dir)
    targets = {
        "cross_model_instances": out / "multiplicity_cross_model_instances.csv",
        "cross_model_summary": out / "multiplicity_cross_model_summary.csv",
        "cross_explainer_instances": out / "multiplicity_cross_explainer_instances.csv",
        "cross_explainer_summary": out / "multiplicity_cross_explainer_summary.csv",
        "pair_instances": out / "multiplicity_explainer_pairs_instances.csv",
        "pair_summary": out / "multiplicity_explainer_pairs_summary.csv",
        "sampling_status": out / "multiplicity_sampling_status.csv",
        "model_performance": out / "multiplicity_model_performance.csv",
        "markdown": out / "multiplicity_summary.md",
    }
    regb.assert_outputs_safe(targets.values(), [results_path])
    regb.write_csv(targets["cross_model_instances"], cross_model)
    regb.write_csv(targets["cross_model_summary"], _tag_unit(summaries["cross_model"], unit))
    regb.write_csv(targets["cross_explainer_instances"], applicants)
    regb.write_csv(
        targets["cross_explainer_summary"], _tag_unit(summaries["cross_explainer"], unit)
    )
    regb.write_csv(targets["pair_instances"], pairs)
    regb.write_csv(targets["pair_summary"], _tag_unit(summaries["pairs"], unit))
    regb.write_csv(targets["sampling_status"], status_rows)
    regb.write_csv(targets["model_performance"], performance_rows)
    regb.write_text(
        targets["markdown"],
        render_multiplicity_markdown(
            summaries,
            bundle=label,
            reason_set=reason_set,
            k=k,
            coverage=index.coverage,
            dropped=dropped,
            status_rows=status_rows,
            performance_rows=performance_rows,
            output_paths=list(targets.values()),
            unit=unit,
            settings=regb.settings_line(payload),
        ),
    )
    return list(targets.values())


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Cross-model reason multiplicity and across-explainer disagreement."
    )
    parser.add_argument(
        "--bundle",
        required=True,
        help="Shared-sampling run directory containing results.json (or the file).",
    )
    parser.add_argument("--out-dir", required=True, help="Directory for CSV and markdown outputs.")
    parser.add_argument(
        "--reason-set",
        choices=regb.REASON_SETS,
        default="principal",
        help="Which explanation features count as reasons (default: principal = notice content).",
    )
    parser.add_argument(
        "--unit",
        choices=REASON_UNITS,
        default="grouped",
        help=(
            "Unit of each reason list: grouped = one unit per notice reason, temporal families "
            "counted once (default); feature_at_k = one unit per feature."
        ),
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Units kept per reason list, for explainers and Reg B alike (default 4).",
    )
    parser.add_argument(
        "--n-boot", type=int, default=2000, help="Bootstrap resamples (default 2000)."
    )
    parser.add_argument("--seed", type=int, default=42, help="Bootstrap seed (default 42).")
    args = parser.parse_args(argv)
    if args.k < 1:
        parser.error("--k must be >= 1")
    for path in run(
        args.bundle,
        args.out_dir,
        reason_set=args.reason_set,
        k=args.k,
        n_boot=args.n_boot,
        seed=args.seed,
        unit=args.unit,
    ):
        print(path)


if __name__ == "__main__":
    main()
