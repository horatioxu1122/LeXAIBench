#!/usr/bin/env python3
"""Validate a LeXAIBench feature taxonomy and dataset registry (taxonomy v2).

    python benchmark/cslaw/validate_taxonomy.py \\
        --taxonomy benchmark/metadata/feature_taxonomy_v2.json \\
        --registry benchmark/metadata/dataset_registry_v2.json \\
        [--compare-v1-taxonomy benchmark/metadata/feature_taxonomy.json] \\
        [--write-yaml paper/feature_taxonomy_manifest_v2.yaml] \\
        [--report taxonomy_validation.md]

Checks. Each failure is a VIOLATION, and the exit code is 1 if there is any:

1. Track rule. Each registry exclusion list equals the rule applied to the
   taxonomy. The rule is read from the registry's ``track_policy``; the v2
   default is that the primary track excludes exactly the features labeled
   sensitivity ``S`` and the audit track excludes no taxonomy feature. Declared
   identifier columns (``track_policy.identifier_columns``, e.g. default
   credit ``ID``) are excluded from both tracks.
2. Labels. Every feature has all five labels from the allowed code sets, a
   non-empty ``plain_meaning``, and a non-empty ``rationale``, ``legal_basis``
   and ``review_status``.
3. Coverage. Every taxonomy dataset appears in the registry and vice versa,
   with the same target.
4. Loader compatibility. Per-feature keys are ``FeatureSpec`` fields
   (taxonomy.py builds ``FeatureSpec(feature_name=..., **payload)``), and
   per-dataset registry keys are exactly the ``DatasetSpec`` fields (metadata.py
   builds ``DatasetSpec(dataset_id=..., **item)``). Any other key would make the
   benchmark fail at load time. Every ``paper_sources`` entry must exist.

WARNINGs do not change the exit code. They cover a missing ``track_policy``
(a v1 registry), exclusion lists in a non-canonical order, demo CSV headers
that disagree with the taxonomy, and a committed YAML manifest (a
``paper_sources`` entry) that differs from what ``--write-yaml`` would write.

The script prints label distributions per dataset and overall.
``--compare-v1-taxonomy`` also prints every label change and every change in
track membership. It reads v1 membership from the registry next to the v1
taxonomy (``feature_taxonomy*.json`` -> ``dataset_registry*.json``) and falls
back to the rule applied to the v1 labels when that registry is missing.
``--write-yaml`` writes a manifest covering every dataset with a small
hand-written YAML emitter, since PyYAML is not a dependency. ``--report`` writes
the same information as markdown.

The script reads JSON and CSV headers only. It does not import the benchmark
package, fit a model or run an explainer. The field lists below mirror
benchmark/src/legalbench_xai/types.py and must change with it.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

LABEL_CODES: dict[str, tuple[str, ...]] = {
    "mutability": ("I", "PS", "PL", "M", "P", "N"),
    "verifiability": ("Y", "P", "N"),
    "correctability": ("Y", "P", "N"),
    "sensitivity": ("S", "P", "N"),
    "principal_reason": ("High", "Medium", "Low"),
}
PROVENANCE_FIELDS = ("rationale", "legal_basis", "review_status")
# FeatureSpec fields other than feature_name (types.py).
FEATURE_FIELDS = ("plain_meaning", *LABEL_CODES, *PROVENANCE_FIELDS)
# DatasetSpec fields other than dataset_id (types.py).
DATASET_SPEC_FIELDS = (
    "display_name",
    "task",
    "target",
    "positive_label",
    "negative_label",
    "demo_data_path",
    "file_format",
    "primary_track_excluded_features",
    "audit_track_excluded_features",
    "paper_sources",
)
TRACKS = ("primary", "audit")
DEFAULT_TRACK_CODES: dict[str, tuple[str, ...]] = {"primary": ("S",), "audit": ()}

_PLAIN_SCALAR = re.compile(r"^[A-Za-z_][A-Za-z0-9_.\-]*$")
# Plain scalars that some YAML 1.1 parsers read as booleans or null.
_YAML_RESERVED = frozenset({"y", "n", "yes", "no", "true", "false", "on", "off", "null"})


@dataclass
class Findings:
    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def violation(self, message: str) -> None:
        self.violations.append(message)

    def warning(self, message: str) -> None:
        self.warnings.append(message)


@dataclass
class TrackPolicy:
    codes: dict[str, tuple[str, ...]]
    # None means "not declared" (a v1 registry): identifiers are inferred.
    identifiers: dict[str, list[str]] | None

    def identifiers_for(
        self,
        dataset_id: str,
        features: dict[str, Any],
        lists: dict[str, list[str]],
    ) -> list[str]:
        if self.identifiers is not None:
            return list(self.identifiers.get(dataset_id, []))
        both = [name for name in lists["primary"] if name in lists["audit"]]
        return [name for name in both if name not in features]


# --------------------------------------------------------------------------- io


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: top level must be a JSON object")
    return payload


def _resolve(path_text: str, repo_root: Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else repo_root / path


def _repo_relative(path: Path, repo_root: Path = REPO_ROOT) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _datasets(payload: dict[str, Any]) -> dict[str, Any]:
    value = payload.get("datasets")
    return value if isinstance(value, dict) else {}


def _features(dataset_entry: Any) -> dict[str, Any]:
    if not isinstance(dataset_entry, dict):
        return {}
    value = dataset_entry.get("features")
    return value if isinstance(value, dict) else {}


def _name_list(value: Any) -> list[str] | None:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return None
    return list(value)


# ------------------------------------------------------------------ validation


def validate(
    taxonomy: dict[str, Any],
    registry: dict[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
) -> Findings:
    findings = Findings()
    if not _datasets(taxonomy):
        findings.violation("taxonomy: top-level 'datasets' must be a non-empty object")
    if not _datasets(registry):
        findings.violation("registry: top-level 'datasets' must be a non-empty object")
    policy = read_track_policy(registry, findings)
    _check_features(_datasets(taxonomy), findings)
    _check_coverage(_datasets(taxonomy), _datasets(registry), findings)
    _check_registry_entries(_datasets(registry), findings, repo_root)
    _check_track_rule(_datasets(taxonomy), _datasets(registry), policy, findings)
    _check_csv_headers(_datasets(taxonomy), _datasets(registry), policy, findings, repo_root)
    return findings


def read_track_policy(registry: dict[str, Any], findings: Findings) -> TrackPolicy:
    raw = registry.get("track_policy")
    if not isinstance(raw, dict):
        findings.warning(
            "registry has no 'track_policy'; checking the v2 rule (primary excludes "
            "sensitivity S, audit excludes nothing) and treating non-taxonomy names "
            "excluded from both tracks as identifier columns"
        )
        return TrackPolicy(codes=dict(DEFAULT_TRACK_CODES), identifiers=None)
    codes: dict[str, tuple[str, ...]] = {}
    for track in TRACKS:
        key = f"{track}_track_excluded_sensitivity"
        value = raw.get(key, list(DEFAULT_TRACK_CODES[track]))
        names = _name_list(value)
        if names is None or any(code not in LABEL_CODES["sensitivity"] for code in names):
            findings.violation(
                f"track_policy.{key} must be a list of sensitivity codes "
                f"{list(LABEL_CODES['sensitivity'])}, got {value!r}"
            )
            names = list(DEFAULT_TRACK_CODES[track])
        codes[track] = tuple(names)
    if codes != DEFAULT_TRACK_CODES:
        findings.warning(
            f"track_policy codes {codes} differ from the v2 rule {DEFAULT_TRACK_CODES}; "
            "checking the registry against the declared codes"
        )
    raw_identifiers = raw.get("identifier_columns", {})
    identifiers: dict[str, list[str]] = {}
    if not isinstance(raw_identifiers, dict):
        findings.violation("track_policy.identifier_columns must be an object {dataset_id: [column, ...]}")
    else:
        for dataset_id, columns in raw_identifiers.items():
            names = _name_list(columns)
            if names is None:
                findings.violation(f"track_policy.identifier_columns.{dataset_id} must be a list of column names")
                continue
            identifiers[str(dataset_id)] = names
    return TrackPolicy(codes=codes, identifiers=identifiers)


def _check_features(tax_datasets: dict[str, Any], findings: Findings) -> None:
    for dataset_id, entry in tax_datasets.items():
        features = _features(entry)
        if not features:
            findings.violation(f"taxonomy.{dataset_id}: 'features' must be a non-empty object")
            continue
        lacking: dict[str, list[str]] = {name: [] for name in PROVENANCE_FIELDS}
        for feature_name, spec in features.items():
            where = f"taxonomy.{dataset_id}.{feature_name}"
            if not isinstance(spec, dict):
                findings.violation(f"{where}: must be an object")
                continue
            unknown = sorted(set(spec) - set(FEATURE_FIELDS))
            if unknown:
                findings.violation(
                    f"{where}: unknown keys {unknown} (taxonomy.py would raise TypeError "
                    "building FeatureSpec)"
                )
            plain_meaning = spec.get("plain_meaning")
            if not isinstance(plain_meaning, str) or not plain_meaning.strip():
                findings.violation(f"{where}: 'plain_meaning' must be a non-empty string")
            for dimension, codes in LABEL_CODES.items():
                value = spec.get(dimension)
                if value is None:
                    findings.violation(f"{where}: missing label '{dimension}'")
                elif value not in codes:
                    findings.violation(
                        f"{where}: {dimension}={value!r} is not one of {list(codes)}"
                    )
            for text_field in PROVENANCE_FIELDS:
                value = spec.get(text_field)
                if not isinstance(value, str) or not value.strip():
                    lacking[text_field].append(feature_name)
        for text_field, names in lacking.items():
            if names:
                example = ", ".join(names[:3]) + (", ..." if len(names) > 3 else "")
                findings.violation(
                    f"taxonomy.{dataset_id}: {len(names)} of {len(features)} features lack a "
                    f"non-empty '{text_field}' ({example})"
                )


def _check_coverage(
    tax_datasets: dict[str, Any],
    reg_datasets: dict[str, Any],
    findings: Findings,
) -> None:
    for dataset_id in tax_datasets:
        if dataset_id not in reg_datasets:
            findings.violation(f"dataset '{dataset_id}' is in the taxonomy but not in the registry")
    for dataset_id in reg_datasets:
        if dataset_id not in tax_datasets:
            findings.violation(f"dataset '{dataset_id}' is in the registry but not in the taxonomy")
    for dataset_id, entry in tax_datasets.items():
        reg_entry = reg_datasets.get(dataset_id)
        if not isinstance(entry, dict) or not isinstance(reg_entry, dict):
            continue
        tax_target = entry.get("target")
        reg_target = reg_entry.get("target")
        if tax_target is not None and reg_target is not None and tax_target != reg_target:
            findings.violation(
                f"{dataset_id}: taxonomy target {tax_target!r} differs from registry target {reg_target!r}"
            )


def _check_registry_entries(
    reg_datasets: dict[str, Any],
    findings: Findings,
    repo_root: Path,
) -> None:
    for dataset_id, entry in reg_datasets.items():
        where = f"registry.{dataset_id}"
        if not isinstance(entry, dict):
            findings.violation(f"{where}: must be an object")
            continue
        missing = [key for key in DATASET_SPEC_FIELDS if key not in entry]
        extra = sorted(set(entry) - set(DATASET_SPEC_FIELDS))
        if missing:
            findings.violation(f"{where}: missing DatasetSpec fields {missing}")
        if extra:
            findings.violation(
                f"{where}: keys {extra} are not DatasetSpec fields (metadata.py would raise "
                "TypeError); put registry-wide notes in 'track_policy'"
            )
        for track in TRACKS:
            key = f"{track}_track_excluded_features"
            if key in entry and _name_list(entry[key]) is None:
                findings.violation(f"{where}.{key}: must be a list of feature names")
        sources = _name_list(entry.get("paper_sources"))
        if sources is None or not sources:
            findings.violation(f"{where}.paper_sources: must be a non-empty list of paths")
        else:
            for source in sources:
                if not _resolve(source, repo_root).exists():
                    findings.violation(f"{where}.paper_sources: '{source}' does not exist")
        demo_path = entry.get("demo_data_path")
        if isinstance(demo_path, str) and not _resolve(demo_path, repo_root).is_file():
            findings.warning(f"{where}.demo_data_path: '{demo_path}' does not exist")


def _sensitivity(features: dict[str, Any], name: str) -> str:
    spec = features.get(name)
    return str(spec.get("sensitivity")) if isinstance(spec, dict) else "?"


def expected_exclusions(
    features: dict[str, Any],
    codes: tuple[str, ...],
    identifiers: list[str],
) -> list[str]:
    """Canonical exclusion list: identifier columns, then features in taxonomy order."""
    ordered = list(identifiers)
    for feature_name, spec in features.items():
        if isinstance(spec, dict) and spec.get("sensitivity") in codes and feature_name not in ordered:
            ordered.append(feature_name)
    return ordered


def _check_track_rule(
    tax_datasets: dict[str, Any],
    reg_datasets: dict[str, Any],
    policy: TrackPolicy,
    findings: Findings,
) -> None:
    for dataset_id, tax_entry in tax_datasets.items():
        reg_entry = reg_datasets.get(dataset_id)
        if not isinstance(reg_entry, dict):
            continue
        features = _features(tax_entry)
        lists = {
            track: _name_list(reg_entry.get(f"{track}_track_excluded_features"))
            for track in TRACKS
        }
        if any(value is None for value in lists.values()):
            continue  # reported by _check_registry_entries
        checked_lists: dict[str, list[str]] = {track: list(lists[track] or []) for track in TRACKS}
        identifiers = policy.identifiers_for(dataset_id, features, checked_lists)
        for name in identifiers:
            if name in features:
                findings.violation(
                    f"{dataset_id}: identifier column '{name}' is also a taxonomy feature"
                )
        for track in TRACKS:
            listed = checked_lists[track]
            codes = policy.codes[track]
            duplicates = sorted(name for name, count in Counter(listed).items() if count > 1)
            if duplicates:
                findings.violation(f"{dataset_id}: {track} exclusion list repeats {duplicates}")
            expected = expected_exclusions(features, codes, identifiers)
            missing = [name for name in expected if name not in listed]
            extra = [name for name in listed if name not in expected]
            for name in missing:
                reason = (
                    "declared identifier column"
                    if name in identifiers
                    else f"labeled sensitivity {_sensitivity(features, name)}"
                )
                findings.violation(
                    f"{dataset_id}: the {track} track must exclude '{name}' ({reason}; "
                    f"rule excludes {list(codes) or 'nothing'})"
                )
            for name in extra:
                if name in features:
                    reason = (
                        f"it is labeled sensitivity {_sensitivity(features, name)}, "
                        f"which the {track} rule keeps"
                    )
                else:
                    reason = "it is neither a taxonomy feature nor a declared identifier column"
                findings.violation(f"{dataset_id}: the {track} track excludes '{name}' but {reason}")
            if not missing and not extra and not duplicates and listed != expected:
                findings.warning(
                    f"{dataset_id}: {track} exclusion list matches the rule but not the canonical "
                    f"order {expected}"
                )


def _check_csv_headers(
    tax_datasets: dict[str, Any],
    reg_datasets: dict[str, Any],
    policy: TrackPolicy,
    findings: Findings,
    repo_root: Path,
) -> None:
    for dataset_id, tax_entry in tax_datasets.items():
        reg_entry = reg_datasets.get(dataset_id)
        if not isinstance(reg_entry, dict) or not isinstance(reg_entry.get("demo_data_path"), str):
            continue
        path = _resolve(reg_entry["demo_data_path"], repo_root)
        if not path.is_file():
            continue  # reported by _check_registry_entries
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                header = next(csv.reader(handle), [])
        except (OSError, UnicodeDecodeError, csv.Error) as exc:
            findings.warning(f"{dataset_id}: cannot read the demo CSV header of {path}: {exc}")
            continue
        columns = [column.strip() for column in header]
        features = _features(tax_entry)
        lists = {
            track: list(_name_list(reg_entry.get(f"{track}_track_excluded_features")) or [])
            for track in TRACKS
        }
        identifiers = policy.identifiers_for(dataset_id, features, lists)
        absent = [name for name in features if name not in columns]
        untracked = [
            column
            for column in columns
            if column not in features and column != reg_entry.get("target") and column not in identifiers
        ]
        if absent:
            findings.warning(
                f"{dataset_id}: taxonomy features absent from the demo CSV header "
                f"(the loader would fill them with None): {absent}"
            )
        if untracked:
            findings.warning(
                f"{dataset_id}: demo CSV columns that are neither taxonomy features, the target, "
                f"nor declared identifiers (the loader drops them): {untracked}"
            )


def check_manifest_freshness(
    registry: dict[str, Any],
    manifest: str,
    findings: Findings,
    *,
    repo_root: Path = REPO_ROOT,
) -> None:
    """Warn when a committed YAML manifest listed in paper_sources is stale."""
    seen: set[str] = set()
    for entry in _datasets(registry).values():
        if not isinstance(entry, dict):
            continue
        for source in _name_list(entry.get("paper_sources")) or []:
            if not source.endswith((".yaml", ".yml")) or source in seen:
                continue
            seen.add(source)
            path = _resolve(source, repo_root)
            if not path.is_file():
                continue  # reported by _check_registry_entries
            current = path.read_text(encoding="utf-8")
            if current != manifest:
                findings.warning(
                    f"manifest '{source}' differs from the generated manifest; regenerate it "
                    f"with --write-yaml {source}"
                )


# --------------------------------------------------------------- distributions


def label_distributions(tax_datasets: dict[str, Any]) -> dict[str, dict[str, Counter]]:
    """{dimension: {dataset_id or 'all': Counter(code -> count)}}."""
    result: dict[str, dict[str, Counter]] = {dimension: {} for dimension in LABEL_CODES}
    for dimension in LABEL_CODES:
        overall: Counter = Counter()
        for dataset_id, entry in tax_datasets.items():
            counts = Counter(
                str(spec.get(dimension))
                for spec in _features(entry).values()
                if isinstance(spec, dict)
            )
            result[dimension][dataset_id] = counts
            overall.update(counts)
        result[dimension]["all"] = overall
    return result


def _codes_for(dimension: str, counters: dict[str, Counter]) -> list[str]:
    codes = list(LABEL_CODES[dimension])
    for counts in counters.values():
        for code in counts:
            if code not in codes:
                codes.append(code)
    return codes


def _cell(count: int, total: int) -> str:
    share = 100.0 * count / total if total else 0.0
    return f"{count} ({share:.1f}%)"


def distribution_text(distributions: dict[str, dict[str, Counter]]) -> list[str]:
    lines: list[str] = []
    for dimension, counters in distributions.items():
        codes = _codes_for(dimension, counters)
        width = max([len(name) for name in counters] + [len(dimension)]) + 2
        lines.append(dimension.ljust(width) + "n".rjust(5) + "".join(code.rjust(14) for code in codes))
        for scope, counts in counters.items():
            total = sum(counts.values())
            cells = "".join(_cell(counts.get(code, 0), total).rjust(14) for code in codes)
            lines.append(scope.ljust(width) + str(total).rjust(5) + cells)
        lines.append("")
    return lines


def distribution_markdown(distributions: dict[str, dict[str, Counter]]) -> list[str]:
    lines: list[str] = []
    for dimension, counters in distributions.items():
        codes = _codes_for(dimension, counters)
        lines.append(f"**{dimension}**")
        lines.append("")
        lines.append("| dataset | n | " + " | ".join(codes) + " |")
        lines.append("|---|---:|" + "---:|" * len(codes))
        for scope, counts in counters.items():
            total = sum(counts.values())
            cells = " | ".join(_cell(counts.get(code, 0), total) for code in codes)
            lines.append(f"| {scope} | {total} | {cells} |")
        lines.append("")
    return lines


# ------------------------------------------------------------------ comparison


@dataclass
class Comparison:
    old_taxonomy_label: str
    old_registry_label: str | None
    label_changes: list[tuple[str, str, str, str, str]]  # dataset, feature, dimension, old, new
    structure_notes: list[str]
    membership_changes: list[tuple[str, str, list[str], list[str]]]  # dataset, track, entered, left
    change_counts: dict[str, Counter]


def _excluded_names(
    features: dict[str, Any],
    reg_entry: Any,
    track: str,
) -> set[str]:
    if isinstance(reg_entry, dict):
        listed = _name_list(reg_entry.get(f"{track}_track_excluded_features"))
        if listed is not None:
            return set(listed)
    return set(expected_exclusions(features, DEFAULT_TRACK_CODES[track], []))


def compare_taxonomies(
    old_taxonomy: dict[str, Any],
    old_registry: dict[str, Any] | None,
    new_taxonomy: dict[str, Any],
    new_registry: dict[str, Any],
    *,
    old_taxonomy_label: str,
    old_registry_label: str | None,
) -> Comparison:
    old_datasets = _datasets(old_taxonomy)
    new_datasets = _datasets(new_taxonomy)
    notes: list[str] = []
    for dataset_id in new_datasets:
        if dataset_id not in old_datasets:
            notes.append(f"dataset '{dataset_id}' is new (not in {old_taxonomy_label})")
    for dataset_id in old_datasets:
        if dataset_id not in new_datasets:
            notes.append(f"dataset '{dataset_id}' was removed")
    changes: list[tuple[str, str, str, str, str]] = []
    counts: dict[str, Counter] = {dimension: Counter() for dimension in LABEL_CODES}
    membership: list[tuple[str, str, list[str], list[str]]] = []
    old_reg_datasets = _datasets(old_registry) if old_registry is not None else {}
    new_reg_datasets = _datasets(new_registry)
    for dataset_id, new_entry in new_datasets.items():
        if dataset_id not in old_datasets:
            continue
        old_features = _features(old_datasets[dataset_id])
        new_features = _features(new_entry)
        added = [name for name in new_features if name not in old_features]
        removed = [name for name in old_features if name not in new_features]
        if added:
            notes.append(f"{dataset_id}: features added {added}")
        if removed:
            notes.append(f"{dataset_id}: features removed {removed}")
        for feature_name, new_spec in new_features.items():
            old_spec = old_features.get(feature_name)
            if not isinstance(old_spec, dict) or not isinstance(new_spec, dict):
                continue
            for dimension in LABEL_CODES:
                old_value = old_spec.get(dimension)
                new_value = new_spec.get(dimension)
                if old_value != new_value:
                    changes.append((dataset_id, feature_name, dimension, str(old_value), str(new_value)))
                    counts[dimension][f"{old_value}->{new_value}"] += 1
        for track in TRACKS:
            old_included = set(old_features) - _excluded_names(
                old_features, old_reg_datasets.get(dataset_id), track
            )
            new_included = set(new_features) - _excluded_names(
                new_features, new_reg_datasets.get(dataset_id), track
            )
            entered = [name for name in new_features if name in new_included and name not in old_included]
            left = [name for name in old_features if name in old_included and name not in new_included]
            if entered or left:
                membership.append((dataset_id, track, entered, left))
    return Comparison(
        old_taxonomy_label=old_taxonomy_label,
        old_registry_label=old_registry_label,
        label_changes=changes,
        structure_notes=notes,
        membership_changes=membership,
        change_counts=counts,
    )


def _membership_source(comparison: Comparison) -> str:
    if comparison.old_registry_label is not None:
        return f"v1 membership from {comparison.old_registry_label}"
    return "v1 registry not found; v1 membership is the v2 rule applied to the v1 labels"


def comparison_text(comparison: Comparison) -> list[str]:
    lines = [f"Comparison with {comparison.old_taxonomy_label}", ""]
    lines.extend(comparison.structure_notes)
    total = len(comparison.label_changes)
    lines.append(f"label changes: {total}")
    for dimension, counts in comparison.change_counts.items():
        if counts:
            summary = ", ".join(f"{key} x{value}" for key, value in sorted(counts.items()))
            lines.append(f"  {dimension}: {summary}")
    lines.append("")
    for dataset_id, feature_name, dimension, old_value, new_value in comparison.label_changes:
        lines.append(f"  {dataset_id}.{feature_name}: {dimension} {old_value} -> {new_value}")
    lines.append("")
    lines.append(f"track membership changes ({_membership_source(comparison)}):")
    if not comparison.membership_changes:
        lines.append("  none")
    for dataset_id, track, entered, left in comparison.membership_changes:
        lines.append(f"  {dataset_id} {track}: now included {entered or '[]'}; now excluded {left or '[]'}")
    lines.append("")
    return lines


def comparison_markdown(comparison: Comparison) -> list[str]:
    lines = [f"## Comparison with `{comparison.old_taxonomy_label}`", ""]
    for note in comparison.structure_notes:
        lines.append(f"- {note}")
    if comparison.structure_notes:
        lines.append("")
    lines.append(f"{len(comparison.label_changes)} label changes.")
    lines.append("")
    lines.append("| dimension | changes |")
    lines.append("|---|---|")
    for dimension, counts in comparison.change_counts.items():
        summary = ", ".join(f"{key} x{value}" for key, value in sorted(counts.items())) or "none"
        lines.append(f"| {dimension} | {summary} |")
    lines.append("")
    if comparison.label_changes:
        lines.append("| dataset | feature | dimension | v1 | v2 |")
        lines.append("|---|---|---|---|---|")
        for dataset_id, feature_name, dimension, old_value, new_value in comparison.label_changes:
            lines.append(f"| {dataset_id} | `{feature_name}` | {dimension} | {old_value} | {new_value} |")
        lines.append("")
    lines.append(f"### Track membership changes ({_membership_source(comparison)})")
    lines.append("")
    if not comparison.membership_changes:
        lines.append("None.")
    else:
        lines.append("| dataset | track | now included | now excluded |")
        lines.append("|---|---|---|---|")
        for dataset_id, track, entered, left in comparison.membership_changes:
            entered_text = ", ".join(f"`{name}`" for name in entered) or "-"
            left_text = ", ".join(f"`{name}`" for name in left) or "-"
            lines.append(f"| {dataset_id} | {track} | {entered_text} | {left_text} |")
    lines.append("")
    return lines


def sibling_registry(taxonomy_path: Path) -> Path | None:
    """feature_taxonomy*.json -> dataset_registry*.json in the same directory."""
    name = taxonomy_path.name
    if "feature_taxonomy" not in name:
        return None
    candidate = taxonomy_path.with_name(name.replace("feature_taxonomy", "dataset_registry", 1))
    return candidate if candidate.is_file() else None


# ------------------------------------------------------------------------ YAML


def yaml_scalar(value: Any) -> str:
    """A plain scalar when unambiguous, otherwise a JSON double-quoted string (valid YAML)."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return json.dumps(value)
    text = str(value)
    if _PLAIN_SCALAR.match(text) and text.lower() not in _YAML_RESERVED:
        return text
    return json.dumps(text, ensure_ascii=False)


def _emit_mapping(mapping: dict[str, Any], indent: int, lines: list[str]) -> None:
    pad = "  " * indent
    for key, value in mapping.items():
        name = yaml_scalar(key)
        if isinstance(value, dict):
            if value:
                lines.append(f"{pad}{name}:")
                _emit_mapping(value, indent + 1, lines)
            else:
                lines.append(f"{pad}{name}: {{}}")
        elif isinstance(value, (list, tuple)):
            if value:
                lines.append(f"{pad}{name}:")
                for item in value:
                    rendered = (
                        json.dumps(item, ensure_ascii=False)
                        if isinstance(item, (dict, list, tuple))
                        else yaml_scalar(item)
                    )
                    lines.append(f"{pad}  - {rendered}")
            else:
                lines.append(f"{pad}{name}: []")
        else:
            lines.append(f"{pad}{name}: {yaml_scalar(value)}")


def manifest_text(
    taxonomy: dict[str, Any],
    registry: dict[str, Any],
    taxonomy_label: str,
    registry_label: str,
) -> str:
    lines = [
        "# LeXAIBench feature taxonomy manifest.",
        "# Generated by benchmark/cslaw/validate_taxonomy.py --write-yaml from",
        f"#   taxonomy: {taxonomy_label}",
        f"#   registry: {registry_label}",
        "# The JSON files are the machine-read source; this manifest documents them.",
        "# Labels are drafts that require author verification; see",
        "# benchmark/metadata/TAXONOMY_V2_CHANGES.md.",
    ]
    top = {key: taxonomy[key] for key in ("taxonomy_version", "status") if key in taxonomy}
    _emit_mapping(top, 0, lines)
    definitions = taxonomy.get("definitions")
    if isinstance(definitions, dict) and definitions:
        lines.append("")
        _emit_mapping({"taxonomy_definitions": definitions}, 0, lines)
    policy = registry.get("track_policy")
    if isinstance(policy, dict) and policy:
        lines.append("")
        _emit_mapping({"track_policy": policy}, 0, lines)
    lines.append("")
    lines.append("datasets:")
    reg_datasets = _datasets(registry)
    for index, (dataset_id, entry) in enumerate(_datasets(taxonomy).items()):
        if index:
            lines.append("")
        entry = entry if isinstance(entry, dict) else {}
        reg_entry = reg_datasets.get(dataset_id)
        reg_entry = reg_entry if isinstance(reg_entry, dict) else {}
        block: dict[str, Any] = {}
        target = entry.get("target", reg_entry.get("target"))
        if target is not None:
            block["target"] = target
        for source_key, manifest_key in (
            ("positive_label", "target_positive_label"),
            ("negative_label", "target_negative_label"),
        ):
            if source_key in reg_entry:
                block[manifest_key] = reg_entry[source_key]
        if "legal_regime" in entry:
            block["legal_regime"] = entry["legal_regime"]
        for track in TRACKS:
            key = f"{track}_track_excluded_features"
            if key in reg_entry:
                block[key] = list(reg_entry[key])
        block["features"] = {
            feature_name: (
                {key: spec[key] for key in FEATURE_FIELDS if key in spec}
                if isinstance(spec, dict)
                else {}
            )
            for feature_name, spec in _features(entry).items()
        }
        _emit_mapping({dataset_id: block}, 1, lines)
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------- report


def report_markdown(
    *,
    taxonomy_label: str,
    registry_label: str,
    findings: Findings,
    distributions: dict[str, dict[str, Counter]],
    registry: dict[str, Any],
    comparison: Comparison | None,
) -> str:
    status = "OK" if not findings.violations else f"FAILED ({len(findings.violations)} violations)"
    lines = [
        "# Taxonomy validation",
        "",
        f"- taxonomy: `{taxonomy_label}`",
        f"- registry: `{registry_label}`",
        f"- status: **{status}**, {len(findings.warnings)} warnings",
        "",
        "## Violations",
        "",
    ]
    lines.extend(f"- {message}" for message in findings.violations)
    if not findings.violations:
        lines.append("None.")
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {message}" for message in findings.warnings)
    if not findings.warnings:
        lines.append("None.")
    lines.extend(["", "## Track exclusions", ""])
    lines.append("| dataset | primary excludes | audit excludes |")
    lines.append("|---|---|---|")
    for dataset_id, entry in _datasets(registry).items():
        if not isinstance(entry, dict):
            continue
        cells = []
        for track in TRACKS:
            names = _name_list(entry.get(f"{track}_track_excluded_features")) or []
            cells.append(", ".join(f"`{name}`" for name in names) or "-")
        lines.append(f"| {dataset_id} | {cells[0]} | {cells[1]} |")
    lines.extend(["", "## Label distributions", ""])
    lines.extend(distribution_markdown(distributions))
    if comparison is not None:
        lines.extend(comparison_markdown(comparison))
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------ main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a feature taxonomy and dataset registry against the v2 track rule."
    )
    parser.add_argument("--taxonomy", required=True, help="feature taxonomy JSON (e.g. feature_taxonomy_v2.json)")
    parser.add_argument("--registry", required=True, help="dataset registry JSON (e.g. dataset_registry_v2.json)")
    parser.add_argument(
        "--compare-v1-taxonomy",
        default=None,
        help="v1 taxonomy JSON; prints label and track-membership changes "
        "(v1 membership comes from the sibling dataset_registry*.json when present)",
    )
    parser.add_argument("--write-yaml", default=None, help="write a YAML manifest covering all datasets")
    parser.add_argument("--report", default=None, help="write a markdown report")
    args = parser.parse_args(argv)

    taxonomy_path = Path(args.taxonomy)
    registry_path = Path(args.registry)
    try:
        taxonomy = load_json(taxonomy_path)
        registry = load_json(registry_path)
    except (OSError, ValueError) as exc:
        print(f"validate_taxonomy: cannot read input: {exc}", file=sys.stderr)
        return 1
    taxonomy_label = _repo_relative(taxonomy_path)
    registry_label = _repo_relative(registry_path)

    protected = {taxonomy_path.resolve(), registry_path.resolve()}
    comparison: Comparison | None = None
    if args.compare_v1_taxonomy:
        old_path = Path(args.compare_v1_taxonomy)
        protected.add(old_path.resolve())
        old_registry_path = sibling_registry(old_path)
        try:
            old_taxonomy = load_json(old_path)
            old_registry = load_json(old_registry_path) if old_registry_path is not None else None
        except (OSError, ValueError) as exc:
            print(f"validate_taxonomy: cannot read the v1 taxonomy: {exc}", file=sys.stderr)
            return 1
        if old_registry_path is not None:
            protected.add(old_registry_path.resolve())
        comparison = compare_taxonomies(
            old_taxonomy,
            old_registry,
            taxonomy,
            registry,
            old_taxonomy_label=_repo_relative(old_path),
            old_registry_label=_repo_relative(old_registry_path) if old_registry_path is not None else None,
        )
    for output in (args.write_yaml, args.report):
        if output and Path(output).resolve() in protected:
            print(f"validate_taxonomy: refusing to overwrite an input file: {output}", file=sys.stderr)
            return 1

    findings = validate(taxonomy, registry)
    manifest = manifest_text(taxonomy, registry, taxonomy_label, registry_label)
    written: list[Path] = []
    if args.write_yaml:
        yaml_path = Path(args.write_yaml)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        yaml_path.write_text(manifest, encoding="utf-8")
        written.append(yaml_path)
    check_manifest_freshness(registry, manifest, findings)
    distributions = label_distributions(_datasets(taxonomy))

    lines = [f"taxonomy: {taxonomy_label}", f"registry: {registry_label}", ""]
    lines.append("label distributions (count and share of features):")
    lines.extend(distribution_text(distributions))
    if comparison is not None:
        lines.extend(comparison_text(comparison))
    lines.append(f"warnings: {len(findings.warnings)}")
    lines.extend(f"  WARNING {message}" for message in findings.warnings)
    lines.append(f"violations: {len(findings.violations)}")
    lines.extend(f"  VIOLATION {message}" for message in findings.violations)
    print("\n".join(lines))

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            report_markdown(
                taxonomy_label=taxonomy_label,
                registry_label=registry_label,
                findings=findings,
                distributions=distributions,
                registry=registry,
                comparison=comparison,
            ),
            encoding="utf-8",
        )
        written.append(report_path)
    for path in written:
        print(f"wrote {path}")
    if findings.violations:
        print(f"validate_taxonomy: FAILED ({len(findings.violations)} violations)")
        return 1
    print(f"validate_taxonomy: OK ({len(findings.warnings)} warnings)")
    return 0


# -------------------------------------------------------------------- selftest


def run_selftest() -> int:
    """Synthetic checks of the rule, the label checks and the YAML emitter.

    Not wired to a CLI flag (the CLI is fixed by the cluster scripts); callable
    from benchmark/cslaw/selftest.py. Returns the number of checks passed.
    """
    import copy
    import tempfile

    passed = 0

    def check(condition: bool, message: str) -> None:
        nonlocal passed
        if not condition:
            raise AssertionError(f"validate_taxonomy selftest: {message}")
        passed += 1

    def spec(sensitivity: str) -> dict[str, str]:
        return {
            "plain_meaning": "x",
            "mutability": "I" if sensitivity == "S" else "M",
            "verifiability": "Y",
            "correctability": "N" if sensitivity == "S" else "Y",
            "sensitivity": sensitivity,
            "principal_reason": "Low",
            "rationale": "r",
            "legal_basis": "none (practical judgment)",
            "review_status": "draft",
        }

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "demo.csv").write_text("ID,age,zip,income,outcome\n1,30,123,5,ok\n", encoding="utf-8")
        (root / "manifest.yaml").write_text("placeholder\n", encoding="utf-8")
        taxonomy = {
            "taxonomy_version": "v2",
            "datasets": {
                "toy": {
                    "target": "outcome",
                    "features": {"age": spec("S"), "zip": spec("P"), "income": spec("N")},
                }
            },
        }
        registry = {
            "track_policy": {
                "primary_track_excluded_sensitivity": ["S"],
                "audit_track_excluded_sensitivity": [],
                "identifier_columns": {"toy": ["ID"]},
            },
            "datasets": {
                "toy": {
                    "display_name": "Toy",
                    "task": "t",
                    "target": "outcome",
                    "positive_label": "No",
                    "negative_label": "0",
                    "demo_data_path": "demo.csv",
                    "file_format": "csv",
                    "primary_track_excluded_features": ["ID", "age"],
                    "audit_track_excluded_features": ["ID"],
                    "paper_sources": ["manifest.yaml"],
                }
            },
        }
        findings = validate(taxonomy, registry, repo_root=root)
        check(not findings.violations, f"clean input has violations {findings.violations}")
        check(not findings.warnings, f"clean input has warnings {findings.warnings}")

        bad = copy.deepcopy(registry)
        bad["datasets"]["toy"]["primary_track_excluded_features"] = ["ID", "age", "zip"]
        findings = validate(taxonomy, bad, repo_root=root)
        check(any("'zip'" in message for message in findings.violations), "a P feature excluded from primary must fail")

        bad = copy.deepcopy(registry)
        bad["datasets"]["toy"]["primary_track_excluded_features"] = ["ID"]
        findings = validate(taxonomy, bad, repo_root=root)
        check(any("must exclude 'age'" in message for message in findings.violations), "an S feature kept in primary must fail")

        bad_taxonomy = copy.deepcopy(taxonomy)
        bad_taxonomy["datasets"]["toy"]["features"]["income"]["verifiability"] = "Q"
        del bad_taxonomy["datasets"]["toy"]["features"]["zip"]["rationale"]
        bad_taxonomy["datasets"]["toy"]["features"]["zip"]["source"] = "x"
        findings = validate(bad_taxonomy, registry, repo_root=root)
        check(any("verifiability='Q'" in message for message in findings.violations), "invalid code must fail")
        check(any("'rationale'" in message for message in findings.violations), "missing rationale must fail")
        check(any("unknown keys" in message for message in findings.violations), "unknown key must fail")

        bad = copy.deepcopy(registry)
        bad["datasets"]["other"] = copy.deepcopy(bad["datasets"]["toy"])
        bad["datasets"]["toy"]["note"] = "x"
        findings = validate(taxonomy, bad, repo_root=root)
        check(any("'other' is in the registry" in message for message in findings.violations), "coverage must fail")
        check(any("not DatasetSpec fields" in message for message in findings.violations), "extra registry key must fail")

        check(yaml_scalar("Y") == '"Y"' and yaml_scalar("N") == '"N"', "Y/N must be quoted")
        check(yaml_scalar("No") == '"No"' and yaml_scalar("0") == '"0"', "No/0 must be quoted")
        check(yaml_scalar("PS") == "PS" and yaml_scalar("a-b_c") == "a-b_c", "plain scalars stay plain")
        manifest = manifest_text(taxonomy, registry, "tax.json", "reg.json")
        check("        sensitivity: S\n" in manifest and '        verifiability: "Y"\n' in manifest, "manifest labels")
        check("  identifier_columns:\n    toy:\n      - ID\n" in manifest, "manifest nested lists")
        check('    target_positive_label: "No"\n' in manifest, "manifest dataset block")
        findings = Findings()
        check_manifest_freshness(registry, manifest, findings, repo_root=root)
        check(len(findings.warnings) == 1, "stale manifest must warn")
        (root / "manifest.yaml").write_text(manifest, encoding="utf-8")
        findings = Findings()
        check_manifest_freshness(registry, manifest, findings, repo_root=root)
        check(not findings.warnings, "fresh manifest must not warn")

        old = copy.deepcopy(taxonomy)
        old["datasets"]["toy"]["features"]["zip"]["sensitivity"] = "S"
        comparison = compare_taxonomies(
            old, None, taxonomy, registry, old_taxonomy_label="old.json", old_registry_label=None
        )
        check(
            ("toy", "zip", "sensitivity", "S", "P") in comparison.label_changes,
            "comparison must list the label change",
        )
        check(
            ("toy", "primary", ["zip"], []) in comparison.membership_changes,
            "comparison must list the membership change",
        )
    return passed


if __name__ == "__main__":
    sys.exit(main())
