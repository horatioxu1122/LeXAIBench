"""Proxy substitution: what replaces excluded features in primary-track reasons?

For every (dataset, model, explainer, eval_row_index) explained in BOTH the
primary track (sensitive/excluded features removed, per
benchmark/metadata/dataset_registry.json) and the audit track (all taxonomy
features available), compare the principal-reason features:

* which primary-excluded features appear in the audit reasons;
* which features "replaced" them in the primary explanation (cited in primary
  but not in audit), with their taxonomy sensitivity labels (S = directly
  sensitive/protected-like, P = proxy-sensitive or socially sensitive,
  N = neither);
* the share of reasons citing sensitivity-P features in primary vs audit, both
  for all P features and for P features RETAINED in primary (the cleaner
  substitution signal, because audit reasons can also cite the excluded P
  features themselves);
* transition counts (excluded feature -> replacement feature). When an
  applicant's audit reasons cite several excluded features and the primary
  reasons contain several replacements, the pairing is ambiguous; we count
  every pair ("pair_count") and also spread one unit of weight evenly over the
  pairs ("weighted_count").

The same comparison is run on the Reg B commentary reasons ("regb_m1",
"regb_m2"; our operationalization of comment 9(b)(2)-5, computed from native
log-odds contributions and the reference means) when they are available.

Unit of comparison: Reg B reasons are built as notice-style units like the
explainers' principal reasons (a PAY_*/BILL_AMT* temporal family is one reason;
regb_reference.regb_reason_units), and n_replacement_reasons /
jaccard_primary_audit_reasons compare the first k reason units of each track,
so explainer and Reg B rows are on the same basis. The replacement feature
lists and the transition counts remain feature-level.

Which registry and taxonomy
---------------------------
The exclusion lists and sensitivity labels are read from the files the run
used: a v2 bundle records them in
payload["cslaw"]["provenance"]["metadata"]["dataset_registry" | "feature_taxonomy"]
(repo-relative paths with SHA-256; e.g. dataset_registry_v2.json and
feature_taxonomy_v2.json), and the script resolves those paths under
--repo-root and warns if a file changed since the run. Bundles without
provenance (v1) use benchmark/metadata/dataset_registry.json and
feature_taxonomy.json. When native contributions are persisted, the features
each track actually used are also read from the bundle and compared with the
registry's exclusion list (``registry_matches_run`` in the exclusions CSV).

Important limitations
---------------------
* The taxonomy has NO proxy-strength field: "P" is a hand-assigned 3-level
  label, not a measured association with a protected attribute.
* v1 bundles: the primary-track exclusion lists are hand-curated and not derived
  from the labels (e.g., south_german_credit keeps `age`, an S feature, in
  primary; heloc's only exclusion is ExternalRiskEstimate). The v2 registry
  applies one stated exclusion rule; check the exclusions table of the output.
  Datasets whose excluded set is empty or contains no S features say little
  about protected-attribute proxies.
* "Replacement" is descriptive: features can enter the primary reasons because
  the primary model was refit, not because they encode the excluded attribute.

Legal context (to verify; not legal advice)
-------------------------------------------
Regulation B, 12 CFR 1002.9(b)(2), requires specific principal reasons, and
comment 9(b)(2)-2 requires them to relate to the factors actually scored, so a
proxy that is actually scored may legitimately appear as a reason. As we
understand it, the CFPB's 2026 amendments to Regulation B removed the
disparate-impact ("effects test") theory under ECOA/Reg B; on that reading,
heavy reliance on proxies for a protected characteristic is relevant mainly as
potential evidence of intentional proxy discrimination (disparate treatment).
Disparate impact remains relevant under the Fair Housing Act for mortgage
lending (e.g., HMDA data). Readers should verify the current status of these
rules before relying on this framing.

Best input: a bundle produced with cslaw.shared_sampling.enabled=true and scope
"models_and_tracks" (one shared applicant set across both tracks), plus
cslaw.persist_canonical_explanations=true. Other bundles are analysed on the
applicants that happen to be sampled in both tracks, with a warning.

Usage
-----
    python benchmark/cslaw/proxy_substitution.py --bundle <shared run dir> \
        --out-dir <dir> [--reason-set principal] [--k 4] [--repo-root <repo>]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import regb_reference as regb  # noqa: E402

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]
REGB_METHODS = {"regb_m1": "near_cutoff", "regb_m2": "all_applicants"}
NO_REPLACEMENT = "(none)"
ALL_MODELS = "(all models)"

SUBSTITUTION_METRICS = (
    "audit_cites_excluded",
    "audit_top1_excluded",
    "n_replacement_reasons",
    "replacement_has_P",
    "replacement_has_S",
    "replacement_has_P_given_excluded",
    "primary_share_reasons_P",
    "audit_share_reasons_P",
    "delta_share_reasons_P",
    "primary_share_reasons_retained_P",
    "audit_share_reasons_retained_P",
    "delta_share_reasons_retained_P",
    "primary_share_reasons_S",
    "audit_share_reasons_S",
    "primary_cites_S",
    "jaccard_primary_audit_reasons",
)

LEGAL_CONTEXT_MD = """\
**Legal context (to verify; not legal advice).** Reg B (12 CFR 1002.9(b)(2))
requires specific principal reasons, and comment 9(b)(2)-2 requires them to
relate to the factors actually scored, so a scored proxy may legitimately
appear as a reason. As we understand it, the CFPB's 2026 amendments to
Regulation B removed the disparate-impact ("effects test") theory under
ECOA/Reg B; on that reading, reliance on proxies for a protected characteristic
matters mainly as potential evidence of intentional proxy discrimination
(disparate treatment). Disparate impact remains relevant under the Fair Housing
Act for mortgage lending. Verify the current status of these rules before
relying on this framing.

**Limitations.** The taxonomy has no proxy-strength field (`P` is a
hand-assigned label meaning proxy-sensitive or socially sensitive). In v1
bundles the primary-track exclusion lists are hand-curated (for example,
south_german_credit keeps `age`, an S feature, in primary); v2 bundles use the
v2 registry recorded in their provenance (see the exclusions table below).
"Replacement" features are those cited in primary but not in audit for the same
applicant; they may enter because the model was refit, not because they encode
the excluded attribute.
"""


def load_registry(repo_root: str | Path) -> dict[str, dict[str, Any]]:
    path = Path(repo_root) / "benchmark" / "metadata" / "dataset_registry.json"
    return load_registry_file(path)


def load_registry_file(path: str | Path) -> dict[str, dict[str, Any]]:
    return dict(common.load_results(path).get("datasets", {}))


def bundle_metadata(
    payload: dict[str, Any],
    repo_root: str | Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, dict[str, Any]]], list[dict[str, Any]]]:
    """(registry, taxonomy, source rows) for the files the bundle's run used.

    A recorded (provenance) file that is missing under ``repo_root`` is an error,
    because falling back to the v1 files would silently change the exclusions.
    """
    sources: list[dict[str, Any]] = []
    loaded: dict[str, Any] = {}
    for name in ("dataset_registry", "feature_taxonomy"):
        path, origin, warning = common.resolve_metadata_file(payload, name, repo_root)
        if not path.is_file():
            raise SystemExit(
                f"proxy_substitution: {warning or f'{name} not found at {path}'}. Pass the "
                "--repo-root of the checkout that produced the bundle."
            )
        if warning:
            print(f"proxy_substitution: WARNING - {warning}", file=sys.stderr)
        loaded[name] = (
            load_registry_file(path) if name == "dataset_registry" else common.load_taxonomy_file(path)
        )
        sources.append({"file": name, "path": str(path), "origin": origin, "warning": warning or ""})
    return loaded["dataset_registry"], loaded["feature_taxonomy"], sources


def observed_track_features(payload: dict[str, Any]) -> dict[tuple[str, str], set[str]]:
    """{(dataset, track): features with native contributions}, i.e. what the run used."""
    observed: dict[tuple[str, str], set[str]] = defaultdict(set)
    for summary_row, instance in common.iter_instances(payload):
        values = regb.native_values(instance)
        if values:
            observed[(summary_row["dataset_id"], summary_row["track"])].update(values)
    return observed


def excluded_in_primary(
    registry: dict[str, dict[str, Any]],
    taxonomy: dict[str, dict[str, dict[str, Any]]],
    dataset_id: str,
) -> list[str]:
    """Taxonomy features excluded in primary but available in audit."""
    spec = registry.get(dataset_id, {})
    audit_excluded = set(spec.get("audit_track_excluded_features", []) or [])
    features = taxonomy.get(dataset_id, {})
    return [
        feature_name
        for feature_name in spec.get("primary_track_excluded_features", []) or []
        if feature_name not in audit_excluded and feature_name in features
    ]


def sensitivity(
    taxonomy: dict[str, dict[str, dict[str, Any]]],
    dataset_id: str,
    feature_name: str,
) -> str:
    return str(taxonomy.get(dataset_id, {}).get(feature_name, {}).get("sensitivity", "?"))


def sampling_scopes(payload: dict[str, Any]) -> set[str]:
    scopes: set[str] = set()
    for group in regb.cslaw_groups(payload).values():
        shared = group.get("shared_sampling") if isinstance(group, dict) else None
        if isinstance(shared, dict) and shared.get("scope"):
            scopes.add(str(shared["scope"]))
    return scopes


def collect_track_pairs(
    payload: dict[str, Any],
    *,
    reason_set: str = "principal",
    k: int = 4,
    include_regb: bool = True,
    fixed_grouping: bool | None = None,
) -> tuple[dict[tuple[str, str, str, int], dict[str, list[list[str]]]], dict[str, str], Counter]:
    """{(dataset, model, method, eval_row_index): {track: reason units}}."""
    if fixed_grouping is None:
        fixed_grouping = regb.grouping_fixed(payload)
    pairs: dict[tuple[str, str, str, int], dict[str, list[list[str]]]] = defaultdict(dict)
    method_kind: dict[str, str] = {}
    coverage: Counter = Counter()
    native_done: set[tuple[str, str, str, int]] = set()
    for summary_row, instance in common.iter_instances(payload):
        coverage["instances"] += 1
        dataset_id, track, model_id, explainer_id = common.group_key(summary_row)
        row_index = common.eval_row_index(instance)
        if row_index is None:
            coverage["skipped_missing_eval_row_index"] += 1
            continue
        canonical = instance.get("canonical_explanation")
        if not canonical:
            coverage["skipped_missing_canonical_explanation"] += 1
        else:
            key = (dataset_id, model_id, explainer_id, row_index)
            if track in pairs[key]:
                coverage["duplicate_instances_ignored"] += 1
            else:
                pairs[key][track] = regb.reason_units(canonical, reason_set)
                method_kind[explainer_id] = (
                    "null_baseline" if common.is_null_baseline(explainer_id) else "explainer"
                )
        if not include_regb:
            continue
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
                continue
            # Notice-style units (a temporal family is one reason), as for explainers.
            pairs[(dataset_id, model_id, method, row_index)][track] = (
                regb.regb_reason_units(values, means, k, fixed=fixed_grouping) or []
            )
            method_kind[method] = "regb_reference"
    return pairs, method_kind, coverage


def _share_reasons(units: list[list[str]], predicate: Callable[[str], bool]) -> float | None:
    if not units:
        return None
    return sum(1 for unit in units if any(predicate(feature) for feature in unit)) / len(units)


def _difference(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def substitution_rows(
    pairs: dict[tuple[str, str, str, int], dict[str, list[list[str]]]],
    method_kind: dict[str, str],
    registry: dict[str, dict[str, Any]],
    taxonomy: dict[str, dict[str, dict[str, Any]]],
    *,
    k: int = 4,
    fixed_grouping: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter]:
    """Return (instance rows, transition rows, coverage).

    ``n_replacement_reasons`` and ``jaccard_primary_audit_reasons`` compare the
    first k reason units of each track (a temporal family counts once), so
    explainers with multi-feature temporal reasons and the Reg B methods are on
    the same basis. The feature lists and transitions stay feature-level.
    """
    rows: list[dict[str, Any]] = []
    coverage: Counter = Counter()
    transitions: dict[tuple[str, ...], dict[str, Any]] = {}
    for key in sorted(pairs, key=lambda item: tuple(str(part) for part in item)):
        dataset_id, model_id, method, row_index = key
        tracks = pairs[key]
        if "primary" not in tracks or "audit" not in tracks:
            coverage["not_in_both_tracks"] += 1
            continue
        coverage["matched_primary_audit"] += 1
        excluded = excluded_in_primary(registry, taxonomy, dataset_id)
        excluded_set = set(excluded)
        labels = taxonomy.get(dataset_id, {})

        def label(feature_name: str) -> str:
            return str(labels.get(feature_name, {}).get("sensitivity", "?"))

        def is_p(feature_name: str) -> bool:
            return label(feature_name) == "P"

        def is_retained_p(feature_name: str) -> bool:
            return label(feature_name) == "P" and feature_name not in excluded_set

        def is_s(feature_name: str) -> bool:
            return label(feature_name) == "S"

        primary_units, audit_units = tracks["primary"], tracks["audit"]
        primary_features = regb.flatten_units(primary_units)
        audit_features = regb.flatten_units(audit_units)
        audit_set = set(audit_features)
        cited_excluded = [name for name in audit_features if name in excluded_set]
        replacements = [name for name in primary_features if name not in audit_set]
        replacement_labels = [label(name) for name in replacements]
        primary_reason_units = regb.grouped_units(primary_features, k, fixed=fixed_grouping)
        audit_reason_units = regb.grouped_units(audit_features, k, fixed=fixed_grouping)
        audit_unit_set = set(audit_reason_units)
        replaced_reasons = [unit for unit in primary_reason_units if unit not in audit_unit_set]
        primary_p = _share_reasons(primary_units, is_p)
        audit_p = _share_reasons(audit_units, is_p)
        primary_rp = _share_reasons(primary_units, is_retained_p)
        audit_rp = _share_reasons(audit_units, is_retained_p)
        has_p = any(value == "P" for value in replacement_labels)
        kind = method_kind.get(method, "explainer")
        rows.append(
            {
                "dataset_id": dataset_id,
                "model_id": model_id,
                "method": method,
                "method_kind": kind,
                "eval_row_index": row_index,
                "excluded_in_primary": "|".join(excluded),
                "excluded_sensitivity": "|".join(label(name) for name in excluded),
                "primary_features": "|".join(primary_features),
                "audit_features": "|".join(audit_features),
                "audit_cites_excluded": bool(cited_excluded),
                "excluded_cited_in_audit": "|".join(cited_excluded),
                "audit_top1_excluded": (
                    audit_features[0] in excluded_set if audit_features else None
                ),
                "replacements": "|".join(replacements),
                "replacement_sensitivity": "|".join(replacement_labels),
                "primary_reason_units": "|".join(primary_reason_units),
                "audit_reason_units": "|".join(audit_reason_units),
                "replaced_reason_units": "|".join(replaced_reasons),
                "n_replacement_reasons": len(replaced_reasons),
                "replacement_has_P": has_p,
                "replacement_has_S": any(value == "S" for value in replacement_labels),
                "replacement_has_P_given_excluded": has_p if cited_excluded else None,
                "primary_share_reasons_P": primary_p,
                "audit_share_reasons_P": audit_p,
                "delta_share_reasons_P": _difference(primary_p, audit_p),
                "primary_share_reasons_retained_P": primary_rp,
                "audit_share_reasons_retained_P": audit_rp,
                "delta_share_reasons_retained_P": _difference(primary_rp, audit_rp),
                "primary_share_reasons_S": _share_reasons(primary_units, is_s),
                "audit_share_reasons_S": _share_reasons(audit_units, is_s),
                "primary_cites_S": any(is_s(name) for name in primary_features),
                "jaccard_primary_audit_reasons": common.jaccard(
                    primary_reason_units, audit_reason_units
                ),
            }
        )
        if not cited_excluded:
            continue
        targets = replacements or [NO_REPLACEMENT]
        weight = 1.0 / (len(cited_excluded) * len(targets))
        for excluded_name in cited_excluded:
            for target in targets:
                for model_key in (model_id, ALL_MODELS):
                    transition_key = (dataset_id, model_key, kind, method, excluded_name, target)
                    record = transitions.setdefault(
                        transition_key,
                        {
                            "dataset_id": dataset_id,
                            "model_id": model_key,
                            "method_kind": kind,
                            "method": method,
                            "excluded_feature": excluded_name,
                            "excluded_sensitivity": label(excluded_name),
                            "replacement_feature": target,
                            "replacement_sensitivity": (
                                "" if target == NO_REPLACEMENT else label(target)
                            ),
                            "pair_count": 0,
                            "weighted_count": 0.0,
                        },
                    )
                    record["pair_count"] += 1
                    record["weighted_count"] += weight
    transition_rows = sorted(
        transitions.values(),
        key=lambda row: (
            row["dataset_id"],
            row["model_id"] != ALL_MODELS,
            row["model_id"],
            row["method"],
            -row["weighted_count"],
            row["excluded_feature"],
            row["replacement_feature"],
        ),
    )
    return rows, transition_rows, coverage


def excluded_feature_rows(
    registry: dict[str, dict[str, Any]],
    taxonomy: dict[str, dict[str, dict[str, Any]]],
    dataset_ids: Sequence[str],
    observed: dict[tuple[str, str], set[str]] | None = None,
) -> list[dict[str, Any]]:
    """Exclusions per dataset; with ``observed`` also what the run actually dropped.

    ``observed`` maps (dataset, track) to the features with native contributions;
    audit minus primary is the run's own primary-track exclusion set.
    """
    rows: list[dict[str, Any]] = []
    for dataset_id in dataset_ids:
        excluded = excluded_in_primary(registry, taxonomy, dataset_id)
        excluded_set = set(excluded)
        retained_s = sorted(
            name
            for name, labels in taxonomy.get(dataset_id, {}).items()
            if labels.get("sensitivity") == "S" and name not in excluded_set
        )
        row: dict[str, Any] = {
            "dataset_id": dataset_id,
            "excluded_in_primary": "|".join(excluded),
            "excluded_sensitivity": "|".join(
                sensitivity(taxonomy, dataset_id, name) for name in excluded
            ),
            "retained_S_features_in_primary": "|".join(retained_s),
            "observed_excluded_in_primary": None,
            "registry_matches_run": None,
        }
        primary = (observed or {}).get((dataset_id, "primary"))
        audit = (observed or {}).get((dataset_id, "audit"))
        if primary and audit:
            dropped = sorted(audit - primary)
            row["observed_excluded_in_primary"] = "|".join(dropped)
            row["registry_matches_run"] = set(dropped) == excluded_set
        rows.append(row)
    return rows


def summarize_substitution(
    rows: Sequence[dict[str, Any]],
    *,
    n_boot: int = 2000,
    seed: int = 42,
) -> list[dict[str, Any]]:
    return regb.summarize_levels(
        rows,
        levels=(
            ("dataset_model_method", ("dataset_id", "model_id", "method_kind", "method"), ()),
            ("dataset_method", ("dataset_id", "method_kind", "method"), ("model_id",)),
            ("method", ("method_kind", "method"), ("dataset_id", "model_id")),
        ),
        all_cols=("dataset_id", "model_id", "method_kind", "method"),
        metrics=SUBSTITUTION_METRICS,
        n_boot=n_boot,
        seed=seed,
    )


def _method_label(method: str, kind: str) -> str:
    return method + (" (null baseline)" if kind == "null_baseline" else "")


def _kind_order(kind: str) -> int:
    return {"regb_reference": 0, "explainer": 1, "null_baseline": 2}.get(kind, 3)


def pooled_transition_lines(
    transitions: Sequence[dict[str, Any]],
    top_transitions: int = 5,
) -> list[str]:
    """Top excluded->replacement pairs per dataset, pooled over models (no null baselines)."""
    pooled: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in transitions:
        if row["model_id"] != ALL_MODELS or row["method_kind"] == "null_baseline":
            continue
        group = (
            "Reg B commentary reasons"
            if row["method_kind"] == "regb_reference"
            else "real explainers"
        )
        name = f"{row['excluded_feature']} -> {row['replacement_feature']}"
        record = pooled.setdefault(
            (row["dataset_id"], group, name),
            {
                "weighted": 0.0,
                "pairs": 0,
                "labels": (
                    f"{row['excluded_sensitivity']} -> {row['replacement_sensitivity'] or '-'}"
                ),
            },
        )
        record["weighted"] += row["weighted_count"]
        record["pairs"] += row["pair_count"]
    lines: list[str] = []
    for dataset_id in sorted({key[0] for key in pooled}):
        for group in ("real explainers", "Reg B commentary reasons"):
            items = sorted(
                (
                    (value["weighted"], name, value)
                    for (ds, grp, name), value in pooled.items()
                    if ds == dataset_id and grp == group
                ),
                key=lambda item: (-item[0], item[1]),
            )[:top_transitions]
            if not items:
                continue
            lines.append(
                f"**{dataset_id}, {group}:** "
                + "; ".join(
                    f"`{name}` ({value['labels']}, weight {weight:.1f}, pairs {value['pairs']})"
                    for weight, name, value in items
                )
            )
            lines.append("")
    return lines


def render_substitution_markdown(
    summary: Sequence[dict[str, Any]],
    transitions: Sequence[dict[str, Any]],
    excluded_rows: Sequence[dict[str, Any]],
    *,
    bundle: str,
    reason_set: str,
    scopes: set[str],
    coverage: Counter,
    output_paths: Sequence[Path],
    top_transitions: int = 5,
    k: int = 4,
    settings: str | None = None,
    metadata_sources: Sequence[dict[str, Any]] = (),
) -> str:
    dataset_level = [row for row in summary if row["level"] == "dataset_method"]
    index = regb.index_summary(dataset_level, ("dataset_id", "method_kind", "method"))
    cells = sorted(
        {(row["dataset_id"], row["method_kind"], row["method"]) for row in dataset_level},
        key=lambda item: (item[0], _kind_order(item[1]), item[2]),
    )
    table_rows = []
    for dataset_id, kind, method in cells:
        cites = index.get((dataset_id, kind, method, "audit_cites_excluded"))
        table_rows.append(
            [
                dataset_id,
                _method_label(method, kind),
                regb.fmt_stat(cites),
                regb.fmt_stat(
                    index.get((dataset_id, kind, method, "replacement_has_P_given_excluded"))
                ),
                regb.fmt_stat(
                    index.get((dataset_id, kind, method, "delta_share_reasons_retained_P"))
                ),
                regb.fmt_stat(index.get((dataset_id, kind, method, "primary_cites_S"))),
                regb.fmt_stat(
                    index.get((dataset_id, kind, method, "jaccard_primary_audit_reasons"))
                ),
                cites["n"] if cites else 0,
            ]
        )
    scope_note = (
        "Shared sampling scope `models_and_tracks`: both tracks explain the same applicants."
        if "models_and_tracks" in scopes
        else (
            "WARNING: this bundle was not produced with shared sampling scope "
            f"`models_and_tracks` (scopes found: {sorted(scopes) or 'none'}). Primary and audit "
            "applicants were sampled separately; only applicants that happened to be sampled in "
            "both tracks are compared, which is a selected subset."
        )
    )
    transition_lines = pooled_transition_lines(transitions, top_transitions)
    mismatched = [row["dataset_id"] for row in excluded_rows if row.get("registry_matches_run") is False]
    source_lines = [
        f"- `{item['file']}`: `{item['path']}` ({item['origin']})"
        + (f" - WARNING: {item['warning']}" if item.get("warning") else "")
        for item in metadata_sources
    ]
    lines = [
        "# Proxy substitution between primary and audit tracks",
        "",
        f"Bundle: `{bundle}`; reason set: `{reason_set}`.",
        "",
        *([f"Settings: {settings}", ""] if settings else []),
        scope_note,
        "",
        LEGAL_CONTEXT_MD,
        "## Primary-track exclusions and S features kept in primary",
        "",
        *(source_lines + [""] if source_lines else []),
        regb.md_table(
            [
                "dataset",
                "excluded in primary (registry)",
                "their sensitivity",
                "S features kept in primary",
                "dropped by the run (native contributions)",
                "registry matches run",
            ],
            [
                [
                    row["dataset_id"],
                    row["excluded_in_primary"] or "(none)",
                    row["excluded_sensitivity"] or "-",
                    row["retained_S_features_in_primary"] or "(none)",
                    "n/a"
                    if row.get("observed_excluded_in_primary") is None
                    else row["observed_excluded_in_primary"] or "(none)",
                    "n/a" if row.get("registry_matches_run") is None else row["registry_matches_run"],
                ]
                for row in excluded_rows
            ],
        ),
        "",
        (
            "WARNING: the registry's exclusion list differs from the features the run "
            f"actually dropped for: {', '.join(mismatched)}. The registry file is probably not "
            "the one the run used; check --repo-root and the provenance paths."
            if mismatched
            else "The registry's exclusion lists match the features the run dropped (where "
            "native contributions are persisted)."
        ),
        "",
        "## Substitution summary by dataset (pooled over models; CIs stratified by model)",
        "",
        "Columns: share of applicants whose audit reasons cite a primary-excluded feature; among "
        "those, share whose primary reasons gain a P-labelled replacement; change (primary minus "
        "audit) in the share of reasons citing a P feature that is retained in primary; share of "
        "primary explanations citing an S feature; Jaccard of the primary vs audit reason units "
        f"(first {k} notice reasons per track, a PAY_*/BILL_AMT* temporal family counted once; "
        "Reg B reasons are built as the same units, so explainer and Reg B rows are comparable).",
        "",
        regb.md_table(
            [
                "dataset",
                "method",
                "audit cites excluded",
                "P replacement | excluded cited",
                "delta retained-P reason share",
                "primary cites S",
                "Jaccard primary/audit (reasons)",
                "n",
            ],
            table_rows,
        )
        if table_rows
        else "_No applicant was explained in both tracks._",
        "",
        f"## Most frequent transitions (excluded -> replacement; top {top_transitions}, "
        "pooled over models)",
        "",
        *(transition_lines or ["_No audit explanation cited a primary-excluded feature._", ""]),
        "Null baselines are excluded from the transition lists above (they are in the CSVs).",
        "",
        "## Coverage",
        "",
        regb.md_table(["count", "value"], sorted(coverage.items())),
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
    repo_root: str | Path = DEFAULT_REPO_ROOT,
    reason_set: str = "principal",
    k: int = 4,
    n_boot: int = 2000,
    seed: int = 42,
) -> list[Path]:
    payload, results_path = regb.load_bundle(bundle)
    label = regb.bundle_label(bundle)
    registry, taxonomy, metadata_sources = bundle_metadata(payload, repo_root)
    fixed_grouping = regb.grouping_fixed(payload)
    scopes = sampling_scopes(payload)
    if "models_and_tracks" not in scopes:
        print(
            "proxy_substitution: WARNING - bundle lacks shared sampling with scope "
            f"'models_and_tracks' (found: {sorted(scopes) or 'none'}); comparing only applicants "
            "sampled in both tracks.",
            file=sys.stderr,
        )
    pairs, method_kind, coverage = collect_track_pairs(
        payload, reason_set=reason_set, k=k, fixed_grouping=fixed_grouping
    )
    rows, transitions, match_coverage = substitution_rows(
        pairs, method_kind, registry, taxonomy, k=k, fixed_grouping=fixed_grouping
    )
    coverage.update(match_coverage)
    if not rows:
        raise SystemExit(
            f"proxy_substitution: no applicant explained in both tracks in {results_path} "
            f"({dict(coverage)}). Re-run with cslaw.persist_canonical_explanations=true and "
            "cslaw.shared_sampling.enabled=true, scope 'models_and_tracks', with both tracks on."
        )
    summary = summarize_substitution(rows, n_boot=n_boot, seed=seed)
    dataset_ids = sorted({row["dataset_id"] for row in rows})
    excluded_rows = excluded_feature_rows(
        registry, taxonomy, dataset_ids, observed_track_features(payload)
    )
    out = Path(out_dir)
    targets = {
        "instances": out / "proxy_substitution_instances.csv",
        "summary": out / "proxy_substitution_summary.csv",
        "transitions": out / "proxy_substitution_transitions.csv",
        "exclusions": out / "proxy_substitution_exclusions.csv",
        "markdown": out / "proxy_substitution_summary.md",
    }
    regb.assert_outputs_safe(targets.values(), [results_path])
    regb.write_csv(targets["instances"], rows)
    regb.write_csv(targets["summary"], summary)
    regb.write_csv(targets["transitions"], transitions)
    regb.write_csv(targets["exclusions"], excluded_rows)
    regb.write_text(
        targets["markdown"],
        render_substitution_markdown(
            summary,
            transitions,
            excluded_rows,
            bundle=label,
            reason_set=reason_set,
            scopes=scopes,
            coverage=coverage,
            output_paths=list(targets.values()),
            k=k,
            settings=regb.settings_line(payload),
            metadata_sources=metadata_sources,
        ),
    )
    return list(targets.values())


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Primary-vs-audit proxy substitution in principal reasons."
    )
    parser.add_argument(
        "--bundle",
        required=True,
        help="Run directory (ideally shared sampling, scope models_and_tracks) or results.json.",
    )
    parser.add_argument("--out-dir", required=True, help="Directory for CSV and markdown outputs.")
    parser.add_argument(
        "--repo-root",
        default=str(DEFAULT_REPO_ROOT),
        help=(
            "Repository root against which the bundle's recorded metadata paths (v2 "
            "provenance) or benchmark/metadata/*.json (v1) are resolved "
            "(default: inferred from this file)."
        ),
    )
    parser.add_argument(
        "--reason-set",
        choices=regb.REASON_SETS,
        default="principal",
        help="Which explanation features count as reasons (default: principal = notice content).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help=(
            "Reason units per track: the Reg B reasons and the reason-level Jaccard / "
            "replacement counts for every method (default 4)."
        ),
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
        repo_root=args.repo_root,
        reason_set=args.reason_set,
        k=args.k,
        n_boot=args.n_boot,
        seed=args.seed,
    ):
        print(path)


if __name__ == "__main__":
    main()
