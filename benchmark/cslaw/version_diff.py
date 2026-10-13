#!/usr/bin/env python3
"""Which method-level conclusions change between a v1 and a v2 LeXAIBench bundle?

    python benchmark/cslaw/version_diff.py --v1 DIR --v2 DIR --out-dir DIR

The v2 corrections (benchmark/src/legalbench_xai/corrections.py) change how
the benchmark samples applicants, which features it uses and how it scores and
builds explanations. This script compares the two bundles at the only level at
which they are comparable: explainer means and explainer rankings per metric,
pooled over the (dataset, track, model) groups both bundles contain (groups
present in only one bundle are dropped and listed) and per dataset.

Instance-level comparison is impossible, and the script does not attempt it:

* sampling differs (v1 takes the rows nearest each decision-margin stratum
  boundary; v2 draws at random within strata) and so does the budget (v1 12,
  v2 24 denied applicants per configuration), so the explained applicants differ;
* the taxonomy and registry differ (v2 applies one exclusion rule and
  re-labelled features), so primary-track feature sets, hence the fitted
  models, can differ; numeric parsing changed for exponent-notation values;
* the metric definitions changed where v1 was invalid (faithfulness against a
  common native reference with no self-reported quality term; DiCE/Wachter
  reasons from their counterfactual changes; non-actionable features blocked;
  PAY_AMT* grouped correctly).

Means are evaluated_rows-weighted configuration means, as in
reporting._weighted_mean. Rankings and leaders use the real explainers present
in both bundles; null baselines (random_reasons, global_ranking) are listed but
never ranked. Besides the published metrics the script compares the
family-neutral variants (faithfulness without the quality term,
comprehensibility without recourse simplicity, CSS without decision-sensitivity
visibility) and CSS_v2 (directionality only; v2 bundles only).

Outputs in --out-dir: version_diff_settings.csv, version_diff_means.csv,
version_diff_leaders.csv, version_diff_pairs.csv and version_diff_summary.md.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from collections.abc import Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import rescore  # noqa: E402

POOLED = "__pooled_common_datasets__"
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
DERIVED_METRICS = ("faithfulness_no_quality", "comprehensibility_no_rs", "css_no_dsv", "css_v2")
METRICS = tuple(SUMMARY_FIELDS) + DERIVED_METRICS
HEADLINE_METRICS = (
    "faithfulness",
    "stability",
    "comprehensibility",
    "css",
    "actionability",
    "alf",
    "faithfulness_no_quality",
    "comprehensibility_no_rs",
    "css_no_dsv",
)
# What each metric depends on among the v2 corrections (for the summary text).
AFFECTED_BY = {
    "faithfulness": "definition (common native reference, quality weight) and explanation "
    "content (native ranking, DiCE/Wachter reasons from changes)",
    "faithfulness_no_quality": "reference (v1 self-referential vs v2 common native) and "
    "explanation content",
    "stability": "explanation content (native ranking, DiCE/Wachter reasons from changes)",
    "comprehensibility": "reason construction (DiCE/Wachter reasons from changes, PAY_AMT "
    "grouping)",
    "comprehensibility_no_rs": "reason construction",
    "css": "taxonomy labels (V, C), counterfactual reasons and changes (S), reason grouping (R)",
    "css_no_dsv": "taxonomy labels (V, C), reason grouping (R)",
    "css_v2": "v2 only (needs the persisted directionality)",
    "actionability": "non-actionable features blocked/hard-failed, taxonomy mutability labels",
    "alf": "non-actionable features blocked/hard-failed, taxonomy mutability labels",
    "css_verifiability": "taxonomy verifiability labels",
    "css_correctability": "taxonomy correctability labels",
    "css_dsv": "counterfactual reasons and changes",
    "css_reason_specificity": "reason construction and grouping",
}


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #


def row_metrics(row: dict[str, Any]) -> dict[str, float | None]:
    """Configuration-level metric values of one summary row (derived ones when possible)."""
    values: dict[str, float | None] = {
        metric: row.get(field) for metric, field in SUMMARY_FIELDS.items()
    }
    instances = row.get("instance_results")
    if instances is not None:
        records = [rescore.instance_metric_values(instance) for instance in instances]
        for metric in DERIVED_METRICS:
            values[metric] = common.mean([record.get(metric) for record in records])
    else:
        values["css_no_dsv"] = rescore.renormalized_score(
            {name: row.get(field) for name, field in rescore.CSS_SUMMARY_COMPONENTS.items()},
            rescore.CSS_NO_DSV,
        )
        for metric in ("faithfulness_no_quality", "comprehensibility_no_rs", "css_v2"):
            values[metric] = None
    return values


def group_key(row: dict[str, Any]) -> tuple[str, str, str]:
    """(dataset_id, track, model_id) of a summary row ("unknown" when not recorded)."""
    return (
        str(row.get("dataset_id")),
        str(row.get("track") or "unknown"),
        str(row.get("model_id") or "unknown"),
    )


def bundle_groups(payload: dict[str, Any]) -> set[tuple[str, str, str]]:
    return {group_key(row) for row in payload.get("summary", []) or []}


def _group_label(group: tuple[str, str, str]) -> str:
    return "/".join(group)


def explainer_means(
    payload: dict[str, Any],
    datasets: Sequence[str],
    groups: set[tuple[str, str, str]] | None = None,
) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    """{scope: {metric: {explainer: {"mean", "rows"}}}} for POOLED and each dataset.

    With ``groups``, only summary rows whose (dataset_id, track, model_id) is in it
    are used, so both bundles are compared over the same groups.
    """
    wanted = set(datasets)
    rows = [
        row
        for row in payload.get("summary", []) or []
        if row.get("dataset_id") in wanted and (groups is None or group_key(row) in groups)
    ]
    prepared = [
        {
            "dataset_id": row["dataset_id"],
            "explainer_id": row["explainer_id"],
            "evaluated_rows": row.get("evaluated_rows", 0),
            **row_metrics(row),
        }
        for row in rows
    ]
    scopes: dict[str, list[dict[str, Any]]] = {POOLED: prepared}
    for item in prepared:
        scopes.setdefault(item["dataset_id"], []).append(item)
    result: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for scope, items in scopes.items():
        by_explainer: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in items:
            by_explainer[item["explainer_id"]].append(item)
        result[scope] = {
            metric: {
                explainer_id: {
                    "mean": rescore.weighted_mean(group, metric),
                    "rows": sum(int(item.get("evaluated_rows", 0) or 0) for item in group),
                }
                for explainer_id, group in sorted(by_explainer.items())
            }
            for metric in METRICS
        }
    return result


def _sign(value: float) -> int:
    return (value > 0) - (value < 0)


def kendall_tau(left: dict[str, float], right: dict[str, float]) -> tuple[float | None, int]:
    """Kendall's tau-a between two score dicts over their common keys (ties count 0)."""
    keys = sorted(set(left) & set(right))
    pairs = list(combinations(keys, 2))
    if not pairs:
        return None, 0
    total = sum(_sign(left[a] - left[b]) * _sign(right[a] - right[b]) for a, b in pairs)
    return total / len(pairs), len(pairs)


def _winner(explainer_a: str, explainer_b: str, diff: float) -> str:
    if diff > 0:
        return explainer_a
    if diff < 0:
        return explainer_b
    return "tie"


def compare(
    v1_means: dict[str, dict[str, dict[str, dict[str, Any]]]],
    v2_means: dict[str, dict[str, dict[str, dict[str, Any]]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """(means rows, leader rows, pair rows) for every scope and metric in both bundles."""
    mean_rows: list[dict[str, Any]] = []
    leader_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    scopes = [POOLED] + sorted(
        scope for scope in v1_means if scope != POOLED and scope in v2_means
    )
    for scope in scopes:
        for metric in METRICS:
            left = v1_means.get(scope, {}).get(metric, {})
            right = v2_means.get(scope, {}).get(metric, {})
            common_real = sorted(
                explainer_id
                for explainer_id in set(left) & set(right)
                if not common.is_null_baseline(explainer_id)
            )
            v1_scores = {e: left[e]["mean"] for e in common_real if left[e]["mean"] is not None}
            v2_scores = {e: right[e]["mean"] for e in common_real if right[e]["mean"] is not None}
            v1_ranks = rescore.rank_real_explainers(v1_scores)
            v2_ranks = rescore.rank_real_explainers(v2_scores)
            for explainer_id in sorted(set(left) | set(right)):
                old = (left.get(explainer_id) or {}).get("mean")
                new = (right.get(explainer_id) or {}).get("mean")
                mean_rows.append(
                    {
                        "scope": scope,
                        "metric": metric,
                        "explainer_id": explainer_id,
                        "family": common.EXPLAINER_FAMILY.get(explainer_id, "unknown"),
                        "is_null_baseline": common.is_null_baseline(explainer_id),
                        "in_both_bundles": explainer_id in left and explainer_id in right,
                        "v1_mean": old,
                        "v2_mean": new,
                        "delta": (new - old) if old is not None and new is not None else None,
                        "v1_rank": v1_ranks.get(explainer_id),
                        "v2_rank": v2_ranks.get(explainer_id),
                        "v1_evaluated_rows": (left.get(explainer_id) or {}).get("rows"),
                        "v2_evaluated_rows": (right.get(explainer_id) or {}).get("rows"),
                    }
                )
            comparable = sorted(set(v1_scores) & set(v2_scores))
            # Leaders among the explainers scored in both bundles on this metric.
            v1_common = {e: v1_scores[e] for e in comparable}
            v2_common = {e: v2_scores[e] for e in comparable}
            v1_common_ranks = rescore.rank_real_explainers(v1_common)
            v2_common_ranks = rescore.rank_real_explainers(v2_common)
            v1_leader = next((e for e, rank in v1_common_ranks.items() if rank == 1), None)
            v2_leader = next((e for e, rank in v2_common_ranks.items() if rank == 1), None)
            tau, n_pairs = kendall_tau(v1_common, v2_common)
            flipped = 0
            for explainer_a, explainer_b in combinations(comparable, 2):
                v1_diff = v1_common[explainer_a] - v1_common[explainer_b]
                v2_diff = v2_common[explainer_a] - v2_common[explainer_b]
                is_flip = _sign(v1_diff) * _sign(v2_diff) < 0
                flipped += int(is_flip)
                pair_rows.append(
                    {
                        "scope": scope,
                        "metric": metric,
                        "explainer_a": explainer_a,
                        "explainer_b": explainer_b,
                        "v1_diff_a_minus_b": v1_diff,
                        "v2_diff_a_minus_b": v2_diff,
                        "v1_winner": _winner(explainer_a, explainer_b, v1_diff),
                        "v2_winner": _winner(explainer_a, explainer_b, v2_diff),
                        "order_flipped": is_flip,
                    }
                )
            leader_rows.append(
                {
                    "scope": scope,
                    "metric": metric,
                    "n_common_real_explainers": len(comparable),
                    "v1_leader": v1_leader,
                    "v1_leader_score": v1_common.get(v1_leader) if v1_leader else None,
                    "v2_leader": v2_leader,
                    "v2_leader_score": v2_common.get(v2_leader) if v2_leader else None,
                    "leader_changed": (
                        v1_leader != v2_leader if v1_leader and v2_leader else None
                    ),
                    "v1_order": ">".join(sorted(v1_common, key=lambda e: v1_common_ranks[e])),
                    "v2_order": ">".join(sorted(v2_common, key=lambda e: v2_common_ranks[e])),
                    "kendall_tau": tau,
                    "n_pairs": n_pairs,
                    "n_pairs_flipped": flipped if n_pairs else None,
                    "affected_by": AFFECTED_BY.get(metric, ""),
                }
            )
    return mean_rows, leader_rows, pair_rows


def bundle_settings(payload: dict[str, Any]) -> dict[str, Any]:
    """Settings that make instance-level comparison impossible, as recorded in a bundle."""
    rows = payload.get("summary", []) or []
    budgets = sorted(
        {
            int((row.get("sampling") or {})["requested_max_instances"])
            for row in rows
            if (row.get("sampling") or {}).get("requested_max_instances") is not None
        }
    )
    selections = sorted(
        {
            str((row.get("sampling") or {}).get("selection"))
            for row in rows
            if (row.get("sampling") or {}).get("selection") is not None
        }
    )
    settings: dict[str, Any] = {
        "experiment_name": payload.get("experiment_name"),
        "corrections_version": common.corrections_version(payload),
        "datasets": "|".join(sorted({str(row["dataset_id"]) for row in rows})),
        "models": "|".join(sorted({str(row["model_id"]) for row in rows})),
        "explainers": "|".join(sorted({str(row["explainer_id"]) for row in rows})),
        "summary_rows": len(rows),
        "evaluated_rows_total": sum(int(row.get("evaluated_rows", 0) or 0) for row in rows),
        "requested_cases_per_configuration": "|".join(str(value) for value in budgets),
        "sampling_selection_recorded": "|".join(selections) or "boundary_nearest (v1, not recorded)",
        "instance_results_persisted": bool(rows)
        and all(row.get("instance_results") is not None for row in rows),
    }
    for name in common.CORRECTIONS_V1:
        settings[f"correction.{name}"] = common.correction_flag(payload, name)
    for name in ("feature_taxonomy", "dataset_registry"):
        entry = common.provenance_metadata_entry(payload, name) or {}
        settings[f"{name}.path"] = entry.get("path") or (
            f"{common.DEFAULT_METADATA_PATHS[name]} (v1 default, not recorded)"
        )
        settings[f"{name}.sha256"] = entry.get("sha256")
    return settings


def settings_rows(v1: dict[str, Any], v2: dict[str, Any]) -> list[dict[str, Any]]:
    left, right = bundle_settings(v1), bundle_settings(v2)
    return [
        {
            "setting": key,
            "v1": left.get(key),
            "v2": right.get(key),
            "differs": left.get(key) != right.get(key),
        }
        for key in dict.fromkeys([*left, *right])
    ]


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(_fmt(cell) for cell in row) + " |" for row in rows]
    return lines


def _scope_label(scope: str) -> str:
    return "pooled" if scope == POOLED else scope


def render_markdown(
    *,
    v1_path: Path,
    v2_path: Path,
    settings: list[dict[str, Any]],
    leaders: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    common_datasets: list[str],
    only_v1: list[str],
    only_v2: list[str],
    output_paths: Sequence[Path],
    groups_compared: int | None = None,
    groups_only_v1: Sequence[str] = (),
    groups_only_v2: Sequence[str] = (),
) -> str:
    headline = [row for row in leaders if row["metric"] in HEADLINE_METRICS]
    headline.sort(key=lambda row: (row["scope"] != POOLED, row["scope"], row["metric"]))
    comparable = [row for row in leaders if row["leader_changed"] is not None]
    changed = [row for row in comparable if row["leader_changed"]]
    headline_comparable = [row for row in headline if row["leader_changed"] is not None]
    headline_changed = [row for row in headline_comparable if row["leader_changed"]]
    flipped_pooled = [
        row
        for row in pairs
        if row["scope"] == POOLED and row["metric"] in HEADLINE_METRICS and row["order_flipped"]
    ]
    differing = [row for row in settings if row["differs"]]
    lines = [
        "# v1 vs v2: which method-level conclusions change",
        "",
        f"- v1 bundle: `{v1_path}`",
        f"- v2 bundle: `{v2_path}`",
        f"- Datasets compared (in both bundles): {', '.join(common_datasets) or 'none'}"
        + (f"; only in v1: {', '.join(only_v1)}" if only_v1 else "")
        + (f"; only in v2: {', '.join(only_v2)}" if only_v2 else ""),
        (
            f"- Means are pooled over the {groups_compared} (dataset, track, model) groups "
            "both bundles contain"
            if groups_compared is not None
            else "- Means are pooled over the datasets both bundles contain"
        )
        + (
            f"; dropped, only in v1: {', '.join(groups_only_v1)}" if groups_only_v1 else ""
        )
        + (
            f"; dropped, only in v2: {', '.join(groups_only_v2)}" if groups_only_v2 else ""
        ),
        "",
        "## Instance-level comparison is not possible",
        "",
        "The two bundles explain different applicants with different feature sets under "
        "partly different metric definitions, so no applicant, configuration score or "
        "per-instance explanation is matched across them. Sampling differs (v1: rows nearest "
        "each decision-margin stratum boundary; v2: random within strata, seeded per "
        "dataset-track-model), the budget differs (cases per configuration below), the "
        "taxonomy and registry differ (v2: one stated exclusion rule and re-labelled features, "
        "which changes primary-track feature sets and therefore the fitted models), and the "
        "corrections change how faithfulness is scored and how DiCE/Wachter reasons are built. "
        "Only explainer means and rankings are compared, pooled over the (dataset, track, model) "
        "groups both bundles contain and per dataset. Leaders are the best real explainer among those scored in "
        "both bundles; null baselines are never ranked.",
        "",
        "## Settings that differ",
        "",
        *_md_table(
            ["setting", "v1", "v2"],
            [[row["setting"], row["v1"], row["v2"]] for row in differing]
            or [["-", "no difference recorded", ""]],
        ),
        "",
        f"({len(settings) - len(differing)} further settings are identical; all are in "
        "version_diff_settings.csv.)",
        "",
        "## Conclusions that change",
        "",
        "- Leaders on the published metrics and their family-neutral variants: "
        f"{len(headline_changed)} of {len(headline_comparable)} (scope x metric) comparisons "
        f"changed; over every metric including the CSS components: {len(changed)} of "
        f"{len(comparable)}.",
        f"- Pooled explainer pairs whose order flips (headline metrics): {len(flipped_pooled)}.",
        "",
        *_md_table(
            ["scope", "metric", "v1 leader", "v2 leader", "Kendall tau", "pairs flipped", "depends on"],
            [
                [
                    _scope_label(row["scope"]),
                    row["metric"],
                    f"{row['v1_leader']} ({_fmt(row['v1_leader_score'])})",
                    f"{row['v2_leader']} ({_fmt(row['v2_leader_score'])})",
                    row["kendall_tau"],
                    f"{row['n_pairs_flipped']}/{row['n_pairs']}",
                    row["affected_by"],
                ]
                for row in headline_changed
            ]
            or [["-", "no leader change", "", "", "", "", ""]],
        ),
        "",
        "## Conclusions that hold (same leader)",
        "",
        *_md_table(
            ["scope", "metric", "leader", "v1 score", "v2 score", "Kendall tau"],
            [
                [
                    _scope_label(row["scope"]),
                    row["metric"],
                    row["v2_leader"],
                    row["v1_leader_score"],
                    row["v2_leader_score"],
                    row["kendall_tau"],
                ]
                for row in headline_comparable
                if not row["leader_changed"]
            ]
            or [["-", "none", "", "", "", ""]],
        ),
        "",
        "## Pooled pairs whose order flips",
        "",
        *_md_table(
            ["metric", "pair", "v1 winner", "v2 winner", "v1 diff", "v2 diff"],
            [
                [
                    row["metric"],
                    f"{row['explainer_a']} vs {row['explainer_b']}",
                    row["v1_winner"],
                    row["v2_winner"],
                    row["v1_diff_a_minus_b"],
                    row["v2_diff_a_minus_b"],
                ]
                for row in flipped_pooled
            ]
            or [["-", "none", "", "", "", ""]],
        ),
        "",
        "Reading guide: a change on a metric whose definition was corrected (faithfulness: "
        "common native reference and no quality term) is the effect of the correction, not a "
        "new empirical finding; a change on the other metrics combines the corrected "
        "explanation construction, the v2 taxonomy and the different sample, and cannot be "
        "attributed to one of them from these two bundles. CSS_v2 exists only in v2 bundles "
        "(n/a for v1). Differences in means are not tested here; use the bootstrap CIs of "
        "paper_tables.py (T4, T8) for uncertainty within each bundle.",
        "",
        "## Files",
        "",
        *[f"- `{path}`" for path in output_paths],
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #


def run(v1: str | Path, v2: str | Path, out_dir: str | Path) -> list[Path]:
    v1_path = rescore.resolve_results_path(str(v1))
    v2_path = rescore.resolve_results_path(str(v2))
    if v1_path == v2_path:
        raise SystemExit("version_diff: --v1 and --v2 are the same bundle")
    out = Path(out_dir).expanduser().resolve()
    if out in (v1_path.parent, v2_path.parent):
        raise SystemExit(f"version_diff: --out-dir {out} is an input bundle directory")
    v1_payload = common.load_results(v1_path)
    v2_payload = common.load_results(v2_path)
    for label, payload, expected in (("--v1", v1_payload, False), ("--v2", v2_payload, True)):
        if common.uses_corrections(payload) != expected:
            print(
                f"version_diff: WARNING - the {label} bundle has corrections "
                f"{common.corrections_version(payload)}; expected "
                f"{'v2' if expected else 'v1 (none)'}",
                file=sys.stderr,
            )
    v1_datasets = {str(row["dataset_id"]) for row in v1_payload.get("summary", []) or []}
    v2_datasets = {str(row["dataset_id"]) for row in v2_payload.get("summary", []) or []}
    # Align on (dataset, track, model) groups, not only datasets: a partial v2 merge
    # (missing shard) or a model in one bundle only would otherwise make leader
    # changes partly coverage effects.
    v1_groups = bundle_groups(v1_payload)
    v2_groups = bundle_groups(v2_payload)
    common_groups = v1_groups & v2_groups
    if not common_groups:
        raise SystemExit(
            f"version_diff: the bundles share no (dataset, track, model) group "
            f"(v1 datasets: {sorted(v1_datasets)}, v2 datasets: {sorted(v2_datasets)})"
        )
    common_datasets = sorted({group[0] for group in common_groups})
    groups_only_v1 = sorted(
        group for group in v1_groups - common_groups if group[0] in v2_datasets
    )
    groups_only_v2 = sorted(
        group for group in v2_groups - common_groups if group[0] in v1_datasets
    )
    if groups_only_v1 or groups_only_v2:
        print(
            "version_diff: WARNING - groups present in one bundle only are dropped: "
            f"only v1 {[_group_label(group) for group in groups_only_v1]}, "
            f"only v2 {[_group_label(group) for group in groups_only_v2]}",
            file=sys.stderr,
        )
    means, leaders, pairs = compare(
        explainer_means(v1_payload, common_datasets, common_groups),
        explainer_means(v2_payload, common_datasets, common_groups),
    )
    settings = settings_rows(v1_payload, v2_payload)
    for key, dropped in (("groups_only_v1", groups_only_v1), ("groups_only_v2", groups_only_v2)):
        value = "|".join(_group_label(group) for group in dropped)
        settings.append(
            {
                "setting": key,
                "v1": value if key == "groups_only_v1" else "",
                "v2": value if key == "groups_only_v2" else "",
                "differs": bool(dropped),
            }
        )
    settings.append(
        {
            "setting": "groups_compared",
            "v1": len(common_groups),
            "v2": len(common_groups),
            "differs": False,
        }
    )
    targets = {
        "settings": out / "version_diff_settings.csv",
        "means": out / "version_diff_means.csv",
        "leaders": out / "version_diff_leaders.csv",
        "pairs": out / "version_diff_pairs.csv",
        "markdown": out / "version_diff_summary.md",
    }
    rescore.write_rows_csv(targets["settings"], settings)
    rescore.write_rows_csv(targets["means"], means)
    rescore.write_rows_csv(targets["leaders"], leaders)
    rescore.write_rows_csv(targets["pairs"], pairs if pairs else [{"note": "no comparable pairs"}])
    targets["markdown"].write_text(
        render_markdown(
            v1_path=v1_path,
            v2_path=v2_path,
            settings=settings,
            leaders=leaders,
            pairs=pairs,
            common_datasets=common_datasets,
            only_v1=sorted(v1_datasets - v2_datasets),
            only_v2=sorted(v2_datasets - v1_datasets),
            output_paths=list(targets.values()),
            groups_compared=len(common_groups),
            groups_only_v1=[_group_label(group) for group in groups_only_v1],
            groups_only_v2=[_group_label(group) for group in groups_only_v2],
        ),
        encoding="utf-8",
    )
    changed = [
        row
        for row in leaders
        if row["scope"] == POOLED and row["metric"] in HEADLINE_METRICS and row["leader_changed"]
    ]
    print(
        "version_diff: pooled leader changes on headline metrics: "
        + (
            ", ".join(f"{row['metric']} {row['v1_leader']}->{row['v2_leader']}" for row in changed)
            or "none"
        )
    )
    print("Wrote:")
    for path in targets.values():
        print(f"  {path}")
    return list(targets.values())


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a v1 and a v2 bundle at the level of explainer means and leaders "
            "(instance-level comparison is impossible: sampling, budget and taxonomy differ)."
        )
    )
    parser.add_argument("--v1", required=True, metavar="DIR", help="v1 bundle (dir or results.json).")
    parser.add_argument("--v2", required=True, metavar="DIR", help="v2 bundle (dir or results.json).")
    parser.add_argument("--out-dir", required=True, metavar="DIR", help="Output directory.")
    args = parser.parse_args(argv)
    run(args.v1, args.v2, args.out_dir)


if __name__ == "__main__":
    main()
