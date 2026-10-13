"""Family-neutral faithfulness against the model's own native contributions.

In v1 bundles the benchmark's built-in faithfulness score is family-specific
and, for SHAP and LIME, self-referential (the reference contributions are the
explainer's own values). This script instead scores every explainer, including
the null baselines, against ONE common reference: the fitted model's native
per-feature contributions in log-odds of the adverse class
(instance["native_contributions"]; positive = pushes toward denial).

v2 bundles (correction common_faithfulness_reference) already score the
pipeline's faithfulness against this common reference, and with
faithfulness_quality_weight 0 the self-reported quality term no longer enters
the score. For them this script adds the metrics the pipeline does not compute
(oracle-normalized coverage, adverse mass, top-k overlap, where counterfactual
changes land) and a consistency check: the pipeline's
reference_feature_coverage must equal native_mass_coverage here
(``pipeline_coverage_abs_diff``; both are cited |native| mass over total
|native| mass).

Legal grounding (our reading, stated cautiously; not legal advice)
------------------------------------------------------------------
Regulation B, 12 CFR 1002.9(b)(2), requires specific principal reasons for
adverse action. Official Interpretation comment 9(b)(2)-2 says the disclosed
reasons must relate to and accurately describe the factors actually considered
or scored, and comment 9(b)(2)-4 says no factor that was a principal reason may
be excluded. Agreement with the scoring model's own contributions is therefore
the legally relevant notion of faithfulness for a reasons notice. Comment
9(b)(2)-5's reference-population methods (method 1: compare with applicants at
or slightly above the minimum passing score; method 2: compare with all
applicants) are analysed in regb_reference.py; here the reference is the
model's own baseline (LR: training mean; EBM: centred shape functions, pair
terms split 50/50; XGBoost: TreeSHAP expected value). This operationalization
is ours, not a regulator-endorsed algorithm.

Metrics (per instance, then explainer x model x dataset with bootstrap CIs)
---------------------------------------------------------------------------
(a) native_mass_coverage: sum |native| over cited (referenced) features divided
    by total |native|; native_mass_coverage_vs_oracle divides by the best
    coverage achievable with the same number of cited features (so explainers
    that cite fewer features, e.g. Anchors, are not penalized for count alone);
    adverse_mass_coverage: the same over positive (denial-increasing) mass only.
(b) direction_agreement: share of referenced features whose signed_direction
    ("increases_denial"/"decreases_denial"/"neutral") matches the sign of the
    feature's native contribution (features with an exactly zero native value
    are skipped; "neutral" counts as a mismatch). Anchors marks every condition
    "increases_denial" by construction. Counterfactual explanations whose
    canonical metadata records cf_reason_source (v2 bundles) take each reason's
    signed_direction from the sign of the native contribution, i.e. from this
    reference, so their agreement would be 1.0 by construction: it is left
    empty (None) for them, matching the v2 pipeline, which scores counterfactual
    directions as neutral under the common reference.
(c) pr_overlap_top_adverse: overlap@k (default 4) between the principal-reason
    features and the top-k adverse native features (pr_top1_top_adverse: top-1).
(d) counterfactual explainers only: cf_change_on_adverse_share = share of
    proposed changes (first counterfactual, as canonicalized) that target
    features with a positive native contribution; cf_change_in_top_adverse_share
    = share that target the top-k adverse native features.

By-construction flags (same as regb_reference.py): SHAP on LR/XGBoost
decomposes the same native quantity; in v1 bundles DiCE/Wachter referenced
features are the top native contributions under native ranking and always on LR
(Anchors candidates likewise; in v2 bundles DiCE/Wachter cite the features their
counterfactuals change, caveat cf_reason_order_from_native); EBM SHAP is in
probability units.
Flagged cells are excluded from "leader" statements, as are null baselines
(regb_reference.leader_lines names the best unflagged real explainer and prints
a higher flagged cell after it as "by-construction, not ranked").

Multiple bundles: pass --bundle several times (e.g. the main instrumented run
plus a separate null-baseline run). Instances without their own
native_contributions borrow them from another bundle by joining on instance_id
("{dataset}-{track}-{model}-row{i}"), but only when the stored prediction_score
matches (otherwise the model fits differ and the join is rejected).

Usage
-----
    python benchmark/cslaw/common_faithfulness.py --bundle <run dir> \
        [--bundle <null-baseline run dir>] --out-dir <dir> [--k 4] [--n-boot 2000]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import regb_reference as regb  # noqa: E402

SCORE_TOLERANCE = 1e-6
NATIVE_TOLERANCE = 1e-6

COVERAGE_TOLERANCE = 1e-6

FAITHFULNESS_METRICS = (
    "native_mass_coverage",
    "native_mass_coverage_vs_oracle",
    "adverse_mass_coverage",
    "direction_agreement",
    "pr_overlap_top_adverse",
    "pr_top1_top_adverse",
    "cf_change_on_adverse_share",
    "cf_change_in_top_adverse_share",
    "n_referenced",
    "pipeline_coverage_abs_diff",
)
ALL_COLS = (
    "bundle",
    "dataset_id",
    "track",
    "model_id",
    "explainer_id",
    "is_null_baseline",
    "by_construction",
    "by_construction_flags",
)


@dataclass(slots=True)
class NativeRecord:
    values: dict[str, float]
    source: str | None
    prediction_score: float | None
    bundle: str


def build_native_index(
    bundles: Sequence[tuple[str, dict[str, Any]]],
) -> tuple[dict[str, NativeRecord], Counter]:
    """First native-contribution record per instance_id across all bundles."""
    index: dict[str, NativeRecord] = {}
    stats: Counter = Counter()
    for label, payload in bundles:
        for _summary_row, instance in common.iter_instances(payload):
            values = regb.native_values(instance)
            instance_id = instance.get("instance_id")
            if values is None or not instance_id:
                continue
            existing = index.get(instance_id)
            if existing is None:
                index[instance_id] = NativeRecord(
                    values=values,
                    source=regb.native_source(instance),
                    prediction_score=regb.as_float(instance.get("prediction_score")),
                    bundle=label,
                )
                continue
            if existing.bundle == label:
                continue
            keys = set(existing.values) | set(values)
            difference = max(
                (abs(existing.values.get(key, 0.0) - values.get(key, 0.0)) for key in keys),
                default=0.0,
            )
            if difference > NATIVE_TOLERANCE:
                stats["native_conflicts_across_bundles"] += 1
    return index, stats


def resolve_native(
    instance: dict[str, Any],
    index: dict[str, NativeRecord],
) -> tuple[dict[str, float] | None, str, str | None]:
    """Return (values, origin, source); origin is own / joined:<bundle> / a skip reason."""
    own = regb.native_values(instance)
    if own is not None:
        return own, "own", regb.native_source(instance)
    record = index.get(str(instance.get("instance_id")))
    if record is None:
        return None, "missing_native_contributions", None
    score = regb.as_float(instance.get("prediction_score"))
    if (
        score is not None
        and record.prediction_score is not None
        and abs(score - record.prediction_score) > SCORE_TOLERANCE
    ):
        return None, "join_rejected_score_mismatch", None
    return record.values, f"joined:{record.bundle}", record.source


def faithfulness_metrics(
    canonical: dict[str, Any],
    values: dict[str, float],
    k: int = 4,
) -> dict[str, Any]:
    """Family-neutral faithfulness of one canonical explanation vs native values."""
    referenced = canonical.get("referenced_features", []) or []
    cited = list(dict.fromkeys(common.referenced_feature_names(canonical)))
    magnitudes = sorted((abs(value) for value in values.values()), reverse=True)
    total_abs = sum(magnitudes)
    total_pos = sum(value for value in values.values() if value > 0)
    cited_abs = sum(abs(values.get(name, 0.0)) for name in cited)
    oracle_mass = sum(magnitudes[: len(cited)])
    agree = compared = neutral = 0
    metadata = canonical.get("metadata")
    # v2 DiCE/Wachter: signed_direction is copied from the native sign, so direction
    # agreement against the same native values is 1.0 by construction (not measured).
    direction_from_native = canonical.get("explanation_family") == "counterfactual" and bool(
        isinstance(metadata, dict) and metadata.get("cf_reason_source")
    )
    for feature in referenced:
        direction = feature.get("signed_direction")
        if direction == "neutral":
            neutral += 1
        value = values.get(feature.get("feature_name"))
        if value is None or value == 0.0:
            continue
        compared += 1
        expected = "increases_denial" if value > 0 else "decreases_denial"
        agree += int(direction == expected)
    principal = common.principal_reason_features(canonical, limit=4)
    top_adverse = common.top_adverse_features(values, k=k)
    top_adverse_set = set(top_adverse)
    changes = list(dict.fromkeys(common.counterfactual_feature_names(canonical)))
    has_changes = canonical.get("explanation_family") == "counterfactual" and bool(changes)
    return {
        "n_referenced": len(cited),
        "n_direction_neutral": neutral,
        "n_counterfactual_changes": len(changes),
        "native_mass_coverage": cited_abs / total_abs if total_abs > 0 else None,
        "native_mass_coverage_vs_oracle": (
            cited_abs / oracle_mass if oracle_mass > 0 and cited else None
        ),
        "adverse_mass_coverage": (
            sum(max(values.get(name, 0.0), 0.0) for name in cited) / total_pos
            if total_pos > 0
            else None
        ),
        "direction_agreement": (
            agree / compared if compared and not direction_from_native else None
        ),
        "pr_overlap_top_adverse": common.overlap_at_k(principal, top_adverse, k),
        "pr_top1_top_adverse": common.top1_match(principal, top_adverse),
        "cf_change_on_adverse_share": (
            sum(1 for name in changes if values.get(name, 0.0) > 0) / len(changes)
            if has_changes
            else None
        ),
        "cf_change_in_top_adverse_share": (
            sum(1 for name in changes if name in top_adverse_set) / len(changes)
            if has_changes
            else None
        ),
        "top_adverse_features": regb.join_features(top_adverse),
        "principal_features": regb.join_features(principal),
        "referenced_features": regb.join_features(cited),
        "counterfactual_features": regb.join_features(changes),
    }


def collect_faithfulness_instances(
    label: str,
    payload: dict[str, Any],
    native_index: dict[str, NativeRecord],
    *,
    k: int = 4,
) -> tuple[list[dict[str, Any]], Counter]:
    native_ranking = regb.native_ranking_enabled(payload)
    from_changes = regb.cf_reasons_from_changes(payload)
    common_reference = bool(common.correction_flag(payload, "common_faithfulness_reference"))
    version = common.corrections_version(payload)
    rows: list[dict[str, Any]] = []
    coverage: Counter = Counter()
    for summary_row, instance in common.iter_instances(payload):
        coverage[f"{label}:instances"] += 1
        dataset_id, track, model_id, explainer_id = common.group_key(summary_row)
        canonical = instance.get("canonical_explanation")
        if not canonical:
            coverage[f"{label}:skipped_missing_canonical_explanation"] += 1
            continue
        values, origin, source = resolve_native(instance, native_index)
        if values is None:
            coverage[f"{label}:skipped_{origin}"] += 1
            continue
        coverage[f"{label}:native_{origin.split(':')[0]}"] += 1
        backend = regb.model_backend_kind(summary_row)
        flags, caveats = regb.by_construction_flags(
            explainer_id,
            model_id,
            native_ranking=native_ranking,
            model_backend=backend,
            contribution_source=source,
            cf_reasons_from_changes=from_changes,
        )
        metrics = faithfulness_metrics(canonical, values, k)
        pipeline = (instance.get("scores") or {}).get("faithfulness_score") or {}
        pipeline_coverage = regb.as_float(pipeline.get("reference_feature_coverage"))
        pipeline_source = pipeline.get("evidence_source")
        coverage_diff = None
        if (
            pipeline_source == common.COMMON_REFERENCE_SOURCE
            and origin == "own"
            and pipeline_coverage is not None
            and metrics["native_mass_coverage"] is not None
        ):
            coverage_diff = abs(pipeline_coverage - float(metrics["native_mass_coverage"]))
            if coverage_diff > COVERAGE_TOLERANCE:
                coverage["pipeline_coverage_mismatch"] += 1
        rows.append(
            {
                "bundle": label,
                "corrections_version": version,
                "pipeline_common_reference": common_reference,
                "pipeline_faithfulness_source": pipeline_source,
                "pipeline_reference_feature_coverage": pipeline_coverage,
                "pipeline_faithfulness": regb.as_float(pipeline.get("score")),
                "pipeline_coverage_abs_diff": coverage_diff,
                "dataset_id": dataset_id,
                "track": track,
                "model_id": model_id,
                "model_class": regb.model_class(model_id),
                "model_backend": backend,
                "explainer_id": explainer_id,
                "explainer_family": common.EXPLAINER_FAMILY.get(explainer_id, "unknown"),
                "is_null_baseline": common.is_null_baseline(explainer_id),
                "instance_id": instance.get("instance_id"),
                "eval_row_index": common.eval_row_index(instance),
                "native_origin": origin,
                "native_source": source,
                "by_construction": bool(flags),
                "by_construction_flags": ";".join(flags),
                "caveats": ";".join(caveats),
                "k": k,
                **metrics,
            }
        )
    return rows, coverage


def summarize_faithfulness(
    rows: Sequence[dict[str, Any]],
    *,
    n_boot: int = 2000,
    seed: int = 42,
) -> list[dict[str, Any]]:
    return regb.summarize_levels(
        rows,
        levels=(
            ("dataset_track_model_explainer", ALL_COLS, ()),
            (
                "dataset_model_explainer",
                (
                    "bundle",
                    "dataset_id",
                    "model_id",
                    "explainer_id",
                    "is_null_baseline",
                    "by_construction",
                    "by_construction_flags",
                ),
                ("track",),
            ),
            (
                "model_explainer",
                (
                    "bundle",
                    "model_id",
                    "explainer_id",
                    "is_null_baseline",
                    "by_construction",
                    "by_construction_flags",
                ),
                ("dataset_id", "track"),
            ),
            (
                "explainer",
                ("bundle", "explainer_id", "is_null_baseline"),
                ("dataset_id", "track", "model_id"),
            ),
        ),
        all_cols=ALL_COLS,
        metrics=(*FAITHFULNESS_METRICS, "by_construction"),
        n_boot=n_boot,
        seed=seed,
    )


def render_faithfulness_markdown(
    summary: Sequence[dict[str, Any]],
    coverage: Counter,
    join_stats: Counter,
    *,
    bundles: Sequence[tuple[str, Path, bool]],
    k: int,
    output_paths: Sequence[Path],
    bundle_settings: dict[str, str] | None = None,
    consistency: Sequence[str] = (),
) -> str:
    model_level = [row for row in summary if row["level"] == "model_explainer"]
    index = regb.index_summary(
        model_level, ("bundle", "model_id", "explainer_id", "by_construction_flags")
    )
    shown = (
        ("native_mass_coverage_vs_oracle", "mass coverage / oracle"),
        ("adverse_mass_coverage", "adverse mass coverage"),
        ("direction_agreement", "direction agreement"),
        ("pr_overlap_top_adverse", f"overlap@{k} top adverse"),
        ("cf_change_on_adverse_share", "CF changes on adverse features"),
        ("n_referenced", "n cited"),
    )
    sections: list[str] = []
    for label in sorted({row["bundle"] for row in model_level}):
        bundle_rows = [row for row in model_level if row["bundle"] == label]
        model_ids = sorted({row["model_id"] for row in bundle_rows})
        for model_id in model_ids:
            cells = sorted(
                {
                    (row["explainer_id"], row["by_construction_flags"], row["is_null_baseline"])
                    for row in bundle_rows
                    if row["model_id"] == model_id
                },
                key=lambda item: (item[2], item[0], item[1]),
            )
            table_rows = []
            for explainer_id, flags, is_null in cells:
                stat_n = index.get((label, model_id, explainer_id, flags, "native_mass_coverage"))
                table_rows.append(
                    [explainer_id + (" (null baseline)" if is_null else ""), flags or "-"]
                    + [
                        regb.fmt_stat(index.get((label, model_id, explainer_id, flags, metric)))
                        for metric, _title in shown
                    ]
                    + [stat_n["n"] if stat_n else 0]
                )
            sections.extend(
                [
                    f"### `{label}` / `{model_id}`",
                    "",
                    regb.md_table(
                        ["explainer", "by-construction flags"]
                        + [title for _metric, title in shown]
                        + ["n"],
                        table_rows,
                    ),
                    "",
                ]
            )
        for metric in (
            "native_mass_coverage_vs_oracle",
            "direction_agreement",
            "pr_overlap_top_adverse",
        ):
            leader = regb.leader_lines(bundle_rows, model_ids, metric)
            if leader:
                sections.extend([f"Leaders on `{metric}` (`{label}`):", "", *leader, ""])
    bundle_lines = [
        f"- `{label}`: `{path}`; native ranking = {native}"
        + (f"; {(bundle_settings or {})[label]}" if (bundle_settings or {}).get(label) else "")
        for label, path, native in bundles
    ]
    lines = [
        "# Family-neutral faithfulness against native model contributions",
        "",
        *bundle_lines,
        "",
        regb.LEGAL_GROUNDING_MD,
        "Every explainer is scored against the same reference (the model's native "
        "log-odds contributions). In v1 bundles the benchmark's built-in faithfulness for "
        "SHAP/LIME/leave-one-out uses the explainer's own values (self-referential). In v2 "
        "bundles (`common_faithfulness_reference=True`) the built-in faithfulness is already "
        "scored against this common reference, so this table complements it rather than "
        "correcting it. Means are pooled over datasets and tracks with 95% bootstrap CIs "
        "stratified by dataset x track; per-dataset rows are in the CSV.",
        "",
        *(
            ["## Pipeline consistency (v2 bundles)", "", *[f"- {line}" for line in consistency], ""]
            if consistency
            else []
        ),
        regb.BY_CONSTRUCTION_MD,
        "Anchors marks every condition as `increases_denial`, so its direction agreement equals "
        "the share of its cited features with positive native contributions. In v2 bundles "
        "DiCE/Wachter reasons copy their `signed_direction` from the sign of the native "
        "contribution, so their direction agreement would be 1.0 by construction; it is left "
        "empty here, and the v2 pipeline scores counterfactual (and Anchors) directions as "
        "neutral (0.5) under the common reference. Native mass coverage depends on how many "
        "features are cited (SHAP/LIME/null baselines cite 6, Wachter often 1-2, Anchors a "
        "few); `mass coverage / oracle` divides by the best coverage achievable with the same "
        "number of features. The pipeline's own v2 coverage term (and so `faithfulness_score` "
        "and T8 `faithfulness_legal`) is not normalized this way and favors explainers that "
        "cite more features.",
        "",
        "## Results by bundle and model",
        "",
        *sections,
        "## Native-contribution joins and coverage",
        "",
        regb.md_table(["count", "value"], sorted((coverage + join_stats).items())),
        "",
        "Joined values are borrowed by instance_id from another bundle and accepted only when "
        "the prediction scores match; `native_conflicts_across_bundles` > 0 means the bundles' "
        "model fits differ for some instances (joins are still score-checked).",
        "",
        "## Files",
        "",
        *[f"- `{path}`" for path in output_paths],
        "",
    ]
    return "\n".join(lines)


def unique_labels(bundles: Sequence[str], labels: Sequence[str] | None) -> list[str]:
    if labels:
        if len(labels) != len(bundles):
            raise SystemExit("--label must be given once per --bundle (or not at all)")
        if len(set(labels)) != len(labels):
            raise SystemExit("--label values must be unique")
        return list(labels)
    output: list[str] = []
    for bundle in bundles:
        base = regb.bundle_label(bundle)
        label = base
        suffix = 2
        while label in output:
            label = f"{base}_{suffix}"
            suffix += 1
        output.append(label)
    return output


def run(
    bundles: Sequence[str | Path],
    out_dir: str | Path,
    *,
    labels: Sequence[str] | None = None,
    k: int = 4,
    n_boot: int = 2000,
    seed: int = 42,
) -> list[Path]:
    names = unique_labels([str(bundle) for bundle in bundles], labels)
    loaded: list[tuple[str, dict[str, Any], Path]] = []
    for name, bundle in zip(names, bundles):
        payload, results_path = regb.load_bundle(bundle)
        loaded.append((name, payload, results_path))
    native_index, join_stats = build_native_index(
        [(name, payload) for name, payload, _path in loaded]
    )
    rows: list[dict[str, Any]] = []
    coverage: Counter = Counter()
    for name, payload, _path in loaded:
        bundle_rows, bundle_coverage = collect_faithfulness_instances(
            name, payload, native_index, k=k
        )
        rows.extend(bundle_rows)
        coverage.update(bundle_coverage)
    if not rows:
        raise SystemExit(
            "common_faithfulness: no analysable instances "
            f"({dict(coverage)}). Each instance needs a persisted canonical_explanation "
            "(cslaw.persist_canonical_explanations=true) and native contributions, either its own "
            "(cslaw.persist_native_contributions=true) or joinable by instance_id from another "
            "--bundle; older bundles store scores only."
        )
    summary = summarize_faithfulness(rows, n_boot=n_boot, seed=seed)
    consistency = pipeline_consistency_lines(rows)
    for line in consistency:
        if line.startswith("WARNING"):
            print(f"common_faithfulness: {line}", file=sys.stderr)
    out = Path(out_dir)
    targets = {
        "instances": out / "common_faithfulness_instances.csv",
        "summary": out / "common_faithfulness_summary.csv",
        "markdown": out / "common_faithfulness_summary.md",
    }
    regb.assert_outputs_safe(targets.values(), [path for _name, _payload, path in loaded])
    regb.write_csv(targets["instances"], rows)
    regb.write_csv(targets["summary"], summary)
    regb.write_text(
        targets["markdown"],
        render_faithfulness_markdown(
            summary,
            coverage,
            join_stats,
            bundles=[
                (name, path, regb.native_ranking_enabled(payload))
                for name, payload, path in loaded
            ],
            k=k,
            output_paths=list(targets.values()),
            bundle_settings={name: regb.settings_line(payload) for name, payload, _path in loaded},
            consistency=consistency,
        ),
    )
    return list(targets.values())


def pipeline_consistency_lines(rows: Sequence[dict[str, Any]]) -> list[str]:
    """Check that v2 pipeline faithfulness used the persisted native contributions.

    Under common_faithfulness_reference the pipeline's reference_feature_coverage
    is the cited share of |native| mass over all included features, which is
    native_mass_coverage here; a difference above the tolerance means the
    pipeline's reference and the persisted native contributions disagree.
    """
    lines: list[str] = []
    for label in sorted({row["bundle"] for row in rows}):
        checked = [
            row
            for row in rows
            if row["bundle"] == label and row.get("pipeline_coverage_abs_diff") is not None
        ]
        if not checked:
            continue
        worst = max(float(row["pipeline_coverage_abs_diff"]) for row in checked)
        bad = sum(
            1 for row in checked if float(row["pipeline_coverage_abs_diff"]) > COVERAGE_TOLERANCE
        )
        prefix = "WARNING: " if bad else ""
        lines.append(
            f"{prefix}`{label}`: pipeline reference_feature_coverage vs native_mass_coverage on "
            f"{len(checked)} instances with their own native contributions: max |diff| = "
            f"{worst:.2e}; {bad} above {COVERAGE_TOLERANCE:g}."
        )
    return lines


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Family-neutral faithfulness of every explainer against native contributions."
    )
    parser.add_argument(
        "--bundle",
        action="append",
        required=True,
        help="Run directory containing results.json (or the file); repeatable.",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=None,
        help="Optional label per --bundle, in the same order (default: directory name).",
    )
    parser.add_argument("--out-dir", required=True, help="Directory for CSV and markdown outputs.")
    parser.add_argument("--k", type=int, default=4, help="Top-k adverse features (default 4).")
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
        labels=args.label,
        k=args.k,
        n_boot=args.n_boot,
        seed=args.seed,
    ):
        print(path)


if __name__ == "__main__":
    main()
