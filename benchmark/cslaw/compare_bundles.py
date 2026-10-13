#!/usr/bin/env python3
"""Exact reproducibility comparison of two LeXAIBench result bundles.

    python benchmark/cslaw/compare_bundles.py LEFT RIGHT [--out report.md]

LEFT and RIGHT are bundle directories or ``results.json`` files. The script
compares the ``results.json`` payloads and then the same-named CSV files in the
two bundle directories. The intended use is to check that an opt-in ``cslaw``
run, with every feature that changes behaviour disabled, reproduces the
baseline bundle.

Ignored, because these differ legitimately between runs or are opt-in
instrumentation:

* top-level ``generated_at_utc``, ``capabilities``, ``cslaw`` and ``rescore``;
* ``data_source.path`` in summary rows and the ``data_source_path`` CSV column;
* per-instance ``eval_row_index``, ``canonical_explanation`` and
  ``native_contributions``.

A difference in ``experiment_name`` is reported as INFO.

For each (dataset, track, model, explainer) group the script requires the
instance_ids to be identical and compares every summary field and every
instance field, scores included, by exact equality. It also reports the maximum
absolute numeric difference. Score differences in (ebm, shap) groups are
labelled EXPECTED_NONDETERMINISTIC rather than failures, because EBM SHAP is not
deterministic across processes; ``--strict`` turns them into failures.
Differences in the overview and in aggregate CSVs that follow from group
differences are labelled DOWNSTREAM. Every summary row and instance must have
``model_backend.backend`` and ``explainer_backend.backend`` equal to "real", and
any stub fallback is a FAILURE.

The exit code is 1 when there is any FAILURE and 0 otherwise.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import io
import math
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

IGNORED_TOP_LEVEL = frozenset({"generated_at_utc", "capabilities", "cslaw", "rescore"})
INFO_TOP_LEVEL = frozenset({"experiment_name"})
IGNORED_INSTANCE_KEYS = frozenset({"eval_row_index", "canonical_explanation", "native_contributions"})
EXPECTED_NONDETERMINISTIC_GROUPS = frozenset({("ebm", "shap")})  # (model_id, explainer_id)
IGNORED_CSV_COLUMNS = frozenset({"data_source_path"})
SUMMARY_CSV_KEY = ("dataset_id", "track", "model_id", "explainer_id")
MODEL_ONLY_CSVS = frozenset({"model_performance.csv"})

FAILURE = "FAILURE"
EXPECTED_NONDETERMINISTIC = "EXPECTED_NONDETERMINISTIC"
DOWNSTREAM = "DOWNSTREAM"
INFO = "INFO"
SEVERITY_ORDER = (FAILURE, EXPECTED_NONDETERMINISTIC, DOWNSTREAM, INFO)

GroupKey = tuple[str, str, str, str]


@dataclass
class Difference:
    path: str
    left: Any
    right: Any
    abs_diff: float | None = None

    def render(self) -> str:
        suffix = f" (abs diff {self.abs_diff:.3g})" if self.abs_diff is not None else ""
        return f"{self.path}: {self.left!r} != {self.right!r}{suffix}"


@dataclass
class Finding:
    severity: str
    area: str
    subject: str
    message: str
    details: list[str] = field(default_factory=list)


@dataclass
class GroupResult:
    key: GroupKey
    status: str  # identical | differs | missing_in_left | missing_in_right
    classification: str | None
    n_left: int | None
    n_right: int | None
    ids_equal: bool | None
    n_differences: int = 0
    n_score_differences: int = 0
    max_abs_diff: float | None = None
    examples: list[str] = field(default_factory=list)


@dataclass
class ComparisonResult:
    findings: list[Finding]
    groups: list[GroupResult]
    csv_identical: list[str] = field(default_factory=list)

    @property
    def failures(self) -> list[Finding]:
        return [finding for finding in self.findings if finding.severity == FAILURE]


# --------------------------------------------------------------------------- #
# Generic diff
# --------------------------------------------------------------------------- #


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _short(value: Any, limit: int = 120) -> Any:
    if isinstance(value, (dict, list)):
        text = repr(value)
        return text if len(text) <= limit else text[: limit - 3] + "..."
    return value


def deep_diff(left: Any, right: Any, path: str = "") -> list[Difference]:
    """All leaf differences between two JSON values (exact equality)."""
    out: list[Difference] = []
    _diff(left, right, path, out)
    return out


def _join(path: str, key: Any) -> str:
    return f"{path}.{key}" if path else str(key)


def _diff(left: Any, right: Any, path: str, out: list[Difference]) -> None:
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right), key=str):
            sub = _join(path, key)
            if key not in left:
                out.append(Difference(sub, "<missing>", _short(right[key])))
            elif key not in right:
                out.append(Difference(sub, _short(left[key]), "<missing>"))
            else:
                _diff(left[key], right[key], sub, out)
        return
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            out.append(Difference(f"{path}[len]", len(left), len(right)))
        for index, (item_left, item_right) in enumerate(zip(left, right)):
            _diff(item_left, item_right, f"{path}[{index}]", out)
        return
    if _is_number(left) and _is_number(right):
        if left == right:
            return
        if (
            isinstance(left, float)
            and isinstance(right, float)
            and math.isnan(left)
            and math.isnan(right)
        ):
            return
        diff = abs(float(left) - float(right))
        out.append(Difference(path, left, right, diff if math.isfinite(diff) else None))
        return
    if left != right:
        out.append(Difference(path, _short(left), _short(right)))


# --------------------------------------------------------------------------- #
# Payload comparison
# --------------------------------------------------------------------------- #


def strip_row(row: dict[str, Any]) -> dict[str, Any]:
    core = {
        key: value
        for key, value in row.items()
        if key not in {"instance_results", "instance_ids"}
    }
    if isinstance(core.get("data_source"), dict):
        core["data_source"] = {
            key: value for key, value in core["data_source"].items() if key != "path"
        }
    return core


def strip_instance(instance: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in instance.items() if key not in IGNORED_INSTANCE_KEYS}


def instance_ids(row: dict[str, Any]) -> list[str]:
    if row.get("instance_ids") is not None:
        return [str(value) for value in row["instance_ids"]]
    return [str(item.get("instance_id")) for item in row.get("instance_results") or []]


def allowed_nondeterministic(path: str) -> bool:
    """Paths that EBM-SHAP nondeterminism may legitimately change."""
    if path.startswith("instance[") and ".scores." in path:
        return True
    head = path.split(".", 1)[0].split("[", 1)[0]
    return head.startswith("mean_") or head.endswith("_evidence_sources")


def backend_failures(payload: dict[str, Any], label: str) -> list[str]:
    messages: list[str] = []
    for row in payload.get("summary", []) or []:
        key = "|".join(common.group_key(row))
        for field_name in ("model_backend", "explainer_backend"):
            backend = row.get(field_name) or {}
            if backend.get("backend") != "real":
                messages.append(
                    f"{label} {key}: summary {field_name}.backend={backend.get('backend')!r} "
                    f"(implementation {backend.get('implementation')!r})"
                )
        counts: Counter = Counter()
        for instance in row.get("instance_results") or []:
            for field_name in ("model_backend", "explainer_backend"):
                backend_name = (instance.get(field_name) or {}).get("backend")
                if backend_name is not None and backend_name != "real":
                    counts[field_name] += 1
        for field_name, count in sorted(counts.items()):
            messages.append(
                f"{label} {key}: {count} instance(s) with {field_name}.backend != 'real'"
            )
    return messages


def compare_group(
    key: GroupKey,
    left_row: dict[str, Any],
    right_row: dict[str, Any],
    *,
    strict: bool,
    max_examples: int,
) -> tuple[GroupResult, list[Finding]]:
    findings: list[Finding] = []
    expected_group = (key[2], key[3]) in EXPECTED_NONDETERMINISTIC_GROUPS and not strict
    left_instances = left_row.get("instance_results")
    right_instances = right_row.get("instance_results")
    left_ids, right_ids = instance_ids(left_row), instance_ids(right_row)
    ids_equal = left_ids == right_ids

    differences = deep_diff(strip_row(left_row), strip_row(right_row))
    if left_instances is not None and right_instances is not None:
        right_by_id = {str(item.get("instance_id")): item for item in right_instances}
        for item in left_instances:
            instance_id = str(item.get("instance_id"))
            other = right_by_id.get(instance_id)
            if other is None:
                continue
            differences += deep_diff(
                strip_instance(item),
                strip_instance(other),
                f"instance[{instance_id}]",
            )
    elif (left_instances is None) != (right_instances is None):
        findings.append(
            Finding(
                INFO,
                "group",
                "|".join(key),
                "instance_results present in only one bundle; compared summary fields only",
            )
        )

    numeric = [difference.abs_diff for difference in differences if difference.abs_diff is not None]
    result = GroupResult(
        key=key,
        status="identical" if ids_equal and not differences else "differs",
        classification=None,
        n_left=len(left_ids),
        n_right=len(right_ids),
        ids_equal=ids_equal,
        n_differences=len(differences),
        n_score_differences=sum(1 for difference in differences if ".scores." in difference.path),
        max_abs_diff=max(numeric) if numeric else None,
    )
    if result.status == "identical":
        return result, findings

    details: list[str] = []
    if not ids_equal:
        left_set, right_set = set(left_ids), set(right_ids)
        only_left = [value for value in left_ids if value not in right_set]
        only_right = [value for value in right_ids if value not in left_set]
        details.append(
            f"instance_ids differ (left {len(left_ids)}, right {len(right_ids)}; "
            f"only left {only_left[:3]}, only right {only_right[:3]})"
        )
    unexpected = [
        difference for difference in differences if not allowed_nondeterministic(difference.path)
    ]
    if expected_group and ids_equal and not unexpected:
        result.classification = EXPECTED_NONDETERMINISTIC
        message = (
            f"{len(differences)} score/mean difference(s) in an (ebm, shap) group "
            f"(max abs diff {result.max_abs_diff})"
        )
        shown = differences
    else:
        result.classification = FAILURE
        message = f"{len(differences)} difference(s); instance_ids equal: {ids_equal}"
        shown = differences
        if expected_group and unexpected:
            message += (
                f"; {len(unexpected)} difference(s) outside scores/means in an (ebm, shap) group"
            )
            shown = unexpected
    details += [difference.render() for difference in shown[:max_examples]]
    result.examples = details
    findings.append(Finding(result.classification, "group", "|".join(key), message, details))
    return result, findings


def _index_rows(
    payload: dict[str, Any],
    label: str,
    findings: list[Finding],
) -> dict[GroupKey, dict[str, Any]]:
    indexed: dict[GroupKey, dict[str, Any]] = {}
    for row in payload.get("summary", []) or []:
        key = common.group_key(row)
        if key in indexed:
            findings.append(
                Finding(FAILURE, "group", "|".join(key), f"duplicate summary row in {label}")
            )
        indexed[key] = row
    return indexed


def downstream_severity(groups: list[GroupResult], allow_extra_groups: bool) -> tuple[str, str]:
    classes = {group.classification for group in groups if group.status != "identical"}
    if not classes:
        return FAILURE, "no group-level difference explains this"
    if FAILURE in classes:
        return DOWNSTREAM, "downstream of failing group differences (counted above)"
    if classes <= {EXPECTED_NONDETERMINISTIC}:
        return DOWNSTREAM, "downstream of expected (ebm, shap) nondeterminism"
    if allow_extra_groups:
        return INFO, "downstream of groups present in only one bundle (--allow-extra-groups)"
    return DOWNSTREAM, "downstream of group differences"


def compare_payloads(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    strict: bool = False,
    allow_extra_groups: bool = False,
    max_examples: int = 5,
) -> ComparisonResult:
    findings: list[Finding] = []
    groups: list[GroupResult] = []

    left_keys = set(left) - IGNORED_TOP_LEVEL
    right_keys = set(right) - IGNORED_TOP_LEVEL
    for key in sorted(left_keys ^ right_keys):
        side = "left" if key in left_keys else "right"
        findings.append(Finding(FAILURE, "top_level", key, f"top-level key present only in {side}"))
    for key in sorted(INFO_TOP_LEVEL & left_keys & right_keys):
        if left[key] != right[key]:
            findings.append(Finding(INFO, "top_level", key, f"{left[key]!r} != {right[key]!r}"))
    for key in sorted((left_keys & right_keys) - INFO_TOP_LEVEL - {"summary", "overview"}):
        differences = deep_diff(left[key], right[key], key)
        if differences:
            findings.append(
                Finding(
                    FAILURE,
                    "top_level",
                    key,
                    f"{len(differences)} difference(s)",
                    [difference.render() for difference in differences[:max_examples]],
                )
            )

    for label, payload in (("left", left), ("right", right)):
        for message in backend_failures(payload, label):
            findings.append(Finding(FAILURE, "backend", label, f"stub fallback: {message}"))

    left_rows = _index_rows(left, "left", findings)
    right_rows = _index_rows(right, "right", findings)
    for key in sorted(set(left_rows) | set(right_rows)):
        if key not in right_rows or key not in left_rows:
            missing_in = "right" if key not in right_rows else "left"
            present = left_rows.get(key) or right_rows.get(key) or {}
            severity = INFO if allow_extra_groups else FAILURE
            groups.append(
                GroupResult(
                    key=key,
                    status=f"missing_in_{missing_in}",
                    classification=severity,
                    n_left=len(instance_ids(left_rows[key])) if key in left_rows else None,
                    n_right=len(instance_ids(right_rows[key])) if key in right_rows else None,
                    ids_equal=None,
                )
            )
            findings.append(
                Finding(
                    severity,
                    "group",
                    "|".join(key),
                    f"group missing in {missing_in} "
                    f"({len(instance_ids(present))} instances on the other side)",
                )
            )
            continue
        result, group_findings = compare_group(
            key,
            left_rows[key],
            right_rows[key],
            strict=strict,
            max_examples=max_examples,
        )
        groups.append(result)
        findings += group_findings

    overview_diffs = deep_diff(left.get("overview"), right.get("overview"), "overview")
    if overview_diffs:
        severity, reason = downstream_severity(groups, allow_extra_groups)
        findings.append(
            Finding(
                severity,
                "overview",
                "overview",
                f"{len(overview_diffs)} difference(s): {reason}",
                [difference.render() for difference in overview_diffs[:max_examples]],
            )
        )
    return ComparisonResult(findings=findings, groups=groups)


# --------------------------------------------------------------------------- #
# CSV comparison
# --------------------------------------------------------------------------- #


def _normalized_bytes(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n")


def _parse_csv(data: bytes) -> tuple[list[str], list[list[str]]]:
    rows = list(csv.reader(io.StringIO(data.decode("utf-8", errors="replace"))))
    if not rows:
        return [], []
    return rows[0], rows[1:]


def _drop_columns(header: list[str], rows: list[list[str]]) -> tuple[list[str], list[list[str]]]:
    keep = [index for index, name in enumerate(header) if name not in IGNORED_CSV_COLUMNS]
    return (
        [header[index] for index in keep],
        [[row[index] if index < len(row) else "" for index in keep] for row in rows],
    )


def _excerpt(left: bytes, right: bytes, name: str, limit: int) -> list[str]:
    diff = difflib.unified_diff(
        left.decode("utf-8", errors="replace").splitlines(),
        right.decode("utf-8", errors="replace").splitlines(),
        fromfile=f"left/{name}",
        tofile=f"right/{name}",
        lineterm="",
        n=0,
    )
    lines: list[str] = []
    for line in diff:
        lines.append(line if len(line) <= 240 else line[:237] + "...")
        if len(lines) >= limit:
            lines.append("...")
            break
    return lines


def compare_csv_file(
    name: str,
    left: bytes,
    right: bytes,
    *,
    groups: list[GroupResult],
    allow_extra_groups: bool,
    max_examples: int,
) -> Finding | None:
    """Return None when the (CRLF->LF normalised) files are byte-identical."""
    if left == right:
        return None
    excerpt = _excerpt(left, right, name, max_examples * 4)
    base = Path(name).name
    left_header, left_rows = _parse_csv(left)
    right_header, right_rows = _parse_csv(right)
    if IGNORED_CSV_COLUMNS & set(left_header) and left_header == right_header:
        header, left_clean = _drop_columns(left_header, left_rows)
        _unused, right_clean = _drop_columns(right_header, right_rows)
        if left_clean == right_clean:
            return Finding(INFO, "csv", name, "differs only in ignored column(s) data_source_path")
        if base == "summary.csv" and all(column in header for column in SUMMARY_CSV_KEY):
            key_index = [header.index(column) for column in SUMMARY_CSV_KEY]
            left_map = {tuple(row[i] for i in key_index): row for row in left_clean}
            right_map = {tuple(row[i] for i in key_index): row for row in right_clean}
            classes = {group.key: group.classification for group in groups}
            unexplained: list[str] = []
            explained: list[str] = []
            for key in sorted(set(left_map) | set(right_map)):
                if left_map.get(key) == right_map.get(key):
                    continue
                group_class = classes.get(key)  # type: ignore[arg-type]
                if group_class == EXPECTED_NONDETERMINISTIC or (
                    group_class == INFO and allow_extra_groups
                ):
                    explained.append("|".join(key))
                else:
                    unexplained.append("|".join(key))
            if unexplained:
                return Finding(
                    FAILURE,
                    "csv",
                    name,
                    f"{len(unexplained)} row(s) differ outside expected groups: "
                    f"{unexplained[:max_examples]}",
                    excerpt,
                )
            return Finding(
                EXPECTED_NONDETERMINISTIC,
                "csv",
                name,
                f"{len(explained)} row(s) differ, all in expected (ebm, shap) or extra groups",
                excerpt,
            )
    if base in MODEL_ONLY_CSVS:
        return Finding(
            FAILURE,
            "csv",
            name,
            "model-level CSV differs (it does not depend on explanations)",
            excerpt,
        )
    severity, reason = downstream_severity(groups, allow_extra_groups)
    return Finding(severity, "csv", name, f"differs: {reason}", excerpt)


def compare_csv_dirs(
    left_dir: Path,
    right_dir: Path,
    *,
    groups: list[GroupResult],
    allow_extra_groups: bool,
    max_examples: int,
) -> tuple[list[Finding], list[str]]:
    findings: list[Finding] = []
    identical: list[str] = []
    left_files = {path.relative_to(left_dir).as_posix(): path for path in left_dir.rglob("*.csv")}
    right_files = {path.relative_to(right_dir).as_posix(): path for path in right_dir.rglob("*.csv")}
    for name in sorted(set(left_files) | set(right_files)):
        if name not in left_files or name not in right_files:
            side = "left" if name in left_files else "right"
            findings.append(Finding(INFO, "csv", name, f"CSV present only in {side}"))
            continue
        finding = compare_csv_file(
            name,
            _normalized_bytes(left_files[name]),
            _normalized_bytes(right_files[name]),
            groups=groups,
            allow_extra_groups=allow_extra_groups,
            max_examples=max_examples,
        )
        if finding is None:
            identical.append(name)
        else:
            findings.append(finding)
    return findings, identical


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def resolve_input(path_text: str) -> tuple[Path, Path]:
    """Return (results.json path, bundle directory)."""
    path = Path(path_text).expanduser()
    if path.is_dir():
        results = path / "results.json"
        directory = path
    else:
        results = path
        directory = path.parent
    if not results.is_file():
        raise SystemExit(f"error: results.json not found at {results}")
    return results.resolve(), directory.resolve()


def render_markdown(
    result: ComparisonResult,
    *,
    left: Path,
    right: Path,
    left_payload: dict[str, Any],
    right_payload: dict[str, Any],
    csv_compared: bool,
) -> str:
    counts = Counter(finding.severity for finding in result.findings)
    verdict = "FAIL" if result.failures else "PASS"
    identical_groups = sum(1 for group in result.groups if group.status == "identical")
    lines = [
        "# Bundle comparison",
        "",
        f"- Left: `{left}` (experiment `{left_payload.get('experiment_name')}`)",
        f"- Right: `{right}` (experiment `{right_payload.get('experiment_name')}`)",
        f"- Verdict: **{verdict}** ({counts.get(FAILURE, 0)} failure(s))",
        "- Counts: "
        + ", ".join(f"{severity} {counts.get(severity, 0)}" for severity in SEVERITY_ORDER),
        f"- Groups: {len(result.groups)} compared, {identical_groups} identical",
        "- Ignored: generated_at_utc, capabilities, data_source.path / data_source_path, "
        "payload cslaw/rescore, instance eval_row_index/canonical_explanation/native_contributions.",
        "",
        "## Groups that differ",
        "",
    ]
    differing = [group for group in result.groups if group.status != "identical"]
    if differing:
        lines.append(
            "| group | status | classification | instances L/R | ids equal | diffs | score diffs | max abs diff |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        for group in differing:
            max_abs = "" if group.max_abs_diff is None else f"{group.max_abs_diff:.3g}"
            lines.append(
                f"| {' / '.join(group.key)} | {group.status} | {group.classification} | "
                f"{group.n_left}/{group.n_right} | {group.ids_equal} | {group.n_differences} | "
                f"{group.n_score_differences} | {max_abs} |"
            )
    else:
        lines.append("All shared groups are identical.")
    lines.append("")
    for severity in SEVERITY_ORDER:
        items = [finding for finding in result.findings if finding.severity == severity]
        if not items:
            continue
        lines += [f"## {severity} ({len(items)})", ""]
        for finding in items:
            lines.append(f"- [{finding.area}] {finding.subject}: {finding.message}")
            for detail in finding.details:
                lines.append(f"    - `{detail}`")
        lines.append("")
    lines += ["## CSV comparison", ""]
    if csv_compared:
        csv_findings = [finding for finding in result.findings if finding.area == "csv"]
        lines.append(f"- Identical after CRLF->LF normalisation: {len(result.csv_identical)}")
        lines.append(f"- Differing or one-sided: {len(csv_findings)} (listed above by severity)")
    else:
        lines.append("- CSV comparison skipped.")
    lines.append("")
    return "\n".join(lines)


def run_comparison(
    left_text: str,
    right_text: str,
    *,
    strict: bool,
    allow_extra_groups: bool,
    skip_csv: bool,
    max_examples: int,
) -> tuple[ComparisonResult, dict[str, Any], dict[str, Any], Path, Path, bool]:
    left_results, left_dir = resolve_input(left_text)
    right_results, right_dir = resolve_input(right_text)
    left_payload = common.load_results(left_results)
    right_payload = common.load_results(right_results)
    result = compare_payloads(
        left_payload,
        right_payload,
        strict=strict,
        allow_extra_groups=allow_extra_groups,
        max_examples=max_examples,
    )
    csv_compared = False
    if skip_csv:
        result.findings.append(Finding(INFO, "csv", "*", "CSV comparison skipped (--skip-csv)"))
    elif left_dir == right_dir:
        result.findings.append(
            Finding(INFO, "csv", "*", "both inputs share a directory; CSV comparison skipped")
        )
    else:
        csv_findings, identical = compare_csv_dirs(
            left_dir,
            right_dir,
            groups=result.groups,
            allow_extra_groups=allow_extra_groups,
            max_examples=max_examples,
        )
        result.findings += csv_findings
        result.csv_identical = identical
        csv_compared = True
    return result, left_payload, right_payload, left_results, right_results, csv_compared


# --------------------------------------------------------------------------- #
# Self-test (synthetic data only)
# --------------------------------------------------------------------------- #


def run_selftest() -> None:
    import copy

    import rescore  # same directory; only its synthetic payload builder is used

    base = rescore.make_synthetic_payload(explainers=("anchors", "shap"))

    same = copy.deepcopy(base)
    same["generated_at_utc"] = "2030-01-01T00:00:00+00:00"
    same["capabilities"] = {"numpy": {"available": True}}
    same["cslaw"] = {"config": {"persist_canonical_explanations": True}, "groups": {}}
    for row in same["summary"]:
        row["data_source"]["path"] = "/cluster/elsewhere.csv"
        for index, instance in enumerate(row["instance_results"]):
            instance["eval_row_index"] = index
            instance["canonical_explanation"] = {"referenced_features": []}
    result = compare_payloads(base, same)
    assert not result.failures, [finding.message for finding in result.failures]
    assert all(group.status == "identical" for group in result.groups)

    nondeterministic = copy.deepcopy(base)
    target = next(
        row
        for row in nondeterministic["summary"]
        if (row["model_id"], row["explainer_id"]) == ("ebm", "shap")
    )
    target["instance_results"][0]["scores"]["faithfulness_score"]["score"] += 1e-6
    target["mean_faithfulness"] += 1e-6 / len(target["instance_results"])
    result = compare_payloads(base, nondeterministic)
    assert not result.failures, [finding.message for finding in result.failures]
    assert any(group.classification == EXPECTED_NONDETERMINISTIC for group in result.groups)
    assert compare_payloads(base, nondeterministic, strict=True).failures

    broken = copy.deepcopy(base)
    target = next(
        row
        for row in broken["summary"]
        if (row["model_id"], row["explainer_id"]) == ("logistic_regression", "shap")
    )
    target["instance_results"][1]["scores"]["stability_score"]["score"] += 0.5
    result = compare_payloads(base, broken)
    assert result.failures
    failing = [group for group in result.groups if group.classification == FAILURE]
    assert failing and failing[0].max_abs_diff is not None
    assert abs(failing[0].max_abs_diff - 0.5) < 1e-9

    stub = copy.deepcopy(base)
    stub["summary"][0]["explainer_backend"]["backend"] = "stub"
    assert any(finding.area == "backend" for finding in compare_payloads(base, stub).failures)

    groups = compare_payloads(base, nondeterministic).groups
    header = "dataset_id,track,model_id,explainer_id,data_source_path,mean_faithfulness\n"
    left_csv = (
        header + "d_alpha,primary,ebm,shap,a.csv,0.5\nd_alpha,primary,ebm,anchors,a.csv,0.7\n"
    ).encode()
    right_csv = (
        header + "d_alpha,primary,ebm,shap,b.csv,0.6\nd_alpha,primary,ebm,anchors,b.csv,0.7\n"
    ).encode()
    finding = compare_csv_file(
        "summary.csv", left_csv, right_csv, groups=groups, allow_extra_groups=False, max_examples=3
    )
    assert finding is not None and finding.severity == EXPECTED_NONDETERMINISTIC, finding
    only_path = (
        header + "d_alpha,primary,ebm,shap,b.csv,0.5\nd_alpha,primary,ebm,anchors,b.csv,0.7\n"
    ).encode()
    finding = compare_csv_file(
        "summary.csv", left_csv, only_path, groups=groups, allow_extra_groups=False, max_examples=3
    )
    assert finding is not None and finding.severity == INFO, finding
    bad = (
        header + "d_alpha,primary,ebm,shap,a.csv,0.5\nd_alpha,primary,ebm,anchors,a.csv,0.8\n"
    ).encode()
    finding = compare_csv_file(
        "summary.csv", left_csv, bad, groups=groups, allow_extra_groups=False, max_examples=3
    )
    assert finding is not None and finding.severity == FAILURE, finding

    with tempfile.TemporaryDirectory() as tmp:
        left_dir, right_dir = Path(tmp) / "left", Path(tmp) / "right"
        left_dir.mkdir()
        right_dir.mkdir()
        (left_dir / "by_explainer.csv").write_bytes(b"explainer_id,mean_css\r\nshap,0.5\r\n")
        (right_dir / "by_explainer.csv").write_bytes(b"explainer_id,mean_css\nshap,0.5\n")
        findings, identical = compare_csv_dirs(
            left_dir, right_dir, groups=groups, allow_extra_groups=False, max_examples=3
        )
        assert identical == ["by_explainer.csv"] and not findings, (identical, findings)
    print("compare_bundles.py selftest: OK")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description="Exact comparison of two LeXAIBench result bundles.")
    parser.add_argument("left", nargs="?", help="Left bundle directory or results.json (e.g. the baseline).")
    parser.add_argument("right", nargs="?", help="Right bundle directory or results.json (e.g. the re-run).")
    parser.add_argument("--out", help="Write a markdown report to this path.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat (ebm, shap) differences as failures instead of EXPECTED_NONDETERMINISTIC.",
    )
    parser.add_argument(
        "--allow-extra-groups",
        action="store_true",
        help="Groups present in only one bundle are INFO instead of FAILURE (e.g. extra explainers).",
    )
    parser.add_argument("--skip-csv", action="store_true", help="Compare results.json only.")
    parser.add_argument("--max-examples", type=int, default=5, help="Differences listed per finding.")
    parser.add_argument("--selftest", action="store_true", help="Run the synthetic self-test and exit.")
    args = parser.parse_args()

    if args.selftest:
        run_selftest()
        return
    if not args.left or not args.right:
        parser.error("LEFT and RIGHT are required")

    result, left_payload, right_payload, left_path, right_path, csv_compared = run_comparison(
        args.left,
        args.right,
        strict=args.strict,
        allow_extra_groups=args.allow_extra_groups,
        skip_csv=args.skip_csv,
        max_examples=max(1, args.max_examples),
    )
    counts = Counter(finding.severity for finding in result.findings)
    verdict = "FAIL" if result.failures else "PASS"
    print(f"Verdict: {verdict}")
    print("Counts: " + ", ".join(f"{severity}={counts.get(severity, 0)}" for severity in SEVERITY_ORDER))
    identical_groups = sum(1 for group in result.groups if group.status == "identical")
    print(f"Groups: {len(result.groups)} compared, {identical_groups} identical")
    if csv_compared:
        print(f"CSVs identical after CRLF->LF normalisation: {len(result.csv_identical)}")
    for finding in result.findings:
        if finding.severity in (FAILURE, EXPECTED_NONDETERMINISTIC):
            print(f"  {finding.severity} [{finding.area}] {finding.subject}: {finding.message}")
    if args.out:
        out_path = Path(args.out).expanduser().resolve()
        if out_path in {left_path, right_path}:
            raise SystemExit("error: --out would overwrite an input results.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            render_markdown(
                result,
                left=left_path,
                right=right_path,
                left_payload=left_payload,
                right_payload=right_payload,
                csv_compared=csv_compared,
            ),
            encoding="utf-8",
        )
        print(f"Wrote: {out_path}")
    sys.exit(1 if result.failures else 0)


if __name__ == "__main__":
    main()
