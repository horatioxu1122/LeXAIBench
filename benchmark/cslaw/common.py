"""Shared helpers for the CS&Law re-analysis scripts.

This module is the data contract between the opt-in pipeline instrumentation
(runner.py ``cslaw`` config block) and the offline analysis scripts in this
directory. It never imports the benchmark package, so every script here can run
on a results.json bundle without re-running models or explainers.

Persisted per-instance fields (present only in bundles produced with the
``cslaw`` persistence flags; older bundles have ``scores`` only):

    instance["eval_row_index"]          int, position in the held-out eval table;
                                        the same applicant across models and tracks
    instance["canonical_explanation"]   CanonicalExplanation.to_dict()
    instance["native_contributions"]    {"source": str, "units": "log_odds_adverse",
                                         "bias": float, "values": {feature: float}}
                                        positive value = pushes toward the adverse
                                        (denial) outcome

Run-level fields (top-level payload key ``cslaw``):

    payload["cslaw"]["config"]          echo of the cslaw config block
    payload["cslaw"]["groups"]["<dataset>|<track>|<model>"] = {
        "regb_reference": {
            "source": str, "threshold": float,
            "near_cutoff": {"n_rows": int, "score_low": float,
                            "score_high": float, "means": {feature: float}},
            "all_applicants": {"n_rows": int, "means": {feature: float}},
        },
        "shared_sampling": {...} | None,
        "model_backend": str,
    }

Versioned corrections (v2 bundles; see benchmark/src/legalbench_xai/corrections.py):

    payload["cslaw"]["corrections"]          CorrectionOptions.to_dict() of the run
    payload["cslaw"]["corrections_version"]  "v2" when the options differ from v1
    payload["cslaw"]["provenance"] = {
        "datasets": {dataset_id: {"path", "sha256", "rows"}},
        "metadata": {"feature_taxonomy": {"path", "sha256"},
                     "dataset_registry": {"path", "sha256"}},
        "git_commit": str | None,
    }
    instance["scores"]["faithfulness_score"]["faithfulness_quality_weight"]
    instance["scores"]["contestability_support_score"]["dsv_directionality" |
        "dsv_threshold_visibility" | "dsv_flip_validity"]
    canonical_explanation["metadata"]["cf_reason_source"]   DiCE/Wachter only
    canonical_explanation["metadata"]["faithfulness_evidence"]["source"] ==
        "model_native_common_reference" under common_faithfulness_reference

A bundle without ``payload["cslaw"]["corrections"]`` is a v1 bundle.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

REAL_EXPLAINERS = ("anchors", "dice", "lime", "shap", "wachter")
NULL_BASELINES = ("random_reasons", "global_ranking")

EXPLAINER_FAMILY = {
    "shap": "attribution",
    "lime": "attribution",
    "leave_one_out": "attribution",
    "random_reasons": "attribution",
    "global_ranking": "attribution",
    "anchors": "rule",
    "dice": "counterfactual",
    "wachter": "counterfactual",
}
RECOURSE_EXPLAINERS = frozenset({"dice", "wachter"})

# Scorer constants, copied from benchmark/src/legalbench_xai/metrics/scorer.py.
FAITHFULNESS_WEIGHTS = {
    "reference_feature_coverage": 0.45,
    "directional_agreement": 0.35,
    "explanation_quality_signal": 0.20,
}
STABILITY_WEIGHTS = {
    "neighborhood_consistency": 0.35,
    "rank_consistency": 0.25,
    "direction_consistency": 0.25,
    "seed_consistency": 0.15,
}
COMPREHENSIBILITY_WEIGHTS = {
    "reason_count_score": 0.35,
    "feature_count_score": 0.25,
    "reason_text_score": 0.20,
    "recourse_simplicity_score": 0.20,
}
CSS_WEIGHTS = {
    "verifiability": 0.30,
    "correctability": 0.30,
    "decision_sensitivity_visibility": 0.25,
    "reason_specificity": 0.15,
}
UNIFIED_ACTIONABILITY_WEIGHTS = {
    "advice_presence": 0.20,
    "alf_component": 0.50,
    "correction_support": 0.30,
}
# Non-recourse explanations receive advice_presence = 0.25 and alf_component = 0,
# so their unified actionability can never exceed this value.
NON_RECOURSE_ACTIONABILITY_CEILING = 0.20 * 0.25 + 0.30 * 1.0

_ROW_SUFFIX = re.compile(r"-row(\d+)$")


def load_results(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)


def group_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (row["dataset_id"], row["track"], row["model_id"], row["explainer_id"])


def cslaw_group_key(dataset_id: str, track: str, model_id: str) -> str:
    return f"{dataset_id}|{track}|{model_id}"


def iter_instances(
    payload: dict[str, Any],
) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """Yield (summary_row, instance_result) pairs for every scored explanation."""
    for row in payload.get("summary", []):
        for instance in row.get("instance_results", []) or []:
            yield row, instance


def eval_row_index(instance: dict[str, Any]) -> int | None:
    """Return the held-out eval-table position that identifies the applicant."""
    value = instance.get("eval_row_index")
    if isinstance(value, int):
        return value
    match = _ROW_SUFFIX.search(str(instance.get("instance_id", "")))
    return int(match.group(1)) if match else None


def principal_reason_features(
    canonical: dict[str, Any] | None,
    limit: int = 4,
) -> list[str]:
    """Ordered, de-duplicated features cited by the principal reasons.

    Principal reasons are the notice-level content (at most four reasons, each of
    which may group several supporting features, e.g. compressed temporal fields).
    """
    if not canonical:
        return []
    ordered: list[str] = []
    reasons = sorted(
        canonical.get("principal_reasons", []) or [],
        key=lambda reason: reason.get("rank", 0),
    )
    for reason in reasons[:limit]:
        for feature_name in reason.get("supporting_features", []) or []:
            if feature_name not in ordered:
                ordered.append(feature_name)
    return ordered


def referenced_feature_names(canonical: dict[str, Any] | None) -> list[str]:
    if not canonical:
        return []
    features = canonical.get("referenced_features", []) or []
    ranked = sorted(
        features,
        key=lambda feature: (
            feature.get("rank") if isinstance(feature.get("rank"), int) else 10**6
        ),
    )
    return [feature["feature_name"] for feature in ranked]


def counterfactual_feature_names(canonical: dict[str, Any] | None) -> list[str]:
    if not canonical:
        return []
    return [
        change["feature_name"]
        for change in canonical.get("counterfactual_changes", []) or []
    ]


def top_adverse_features(
    values: dict[str, float],
    k: int = 4,
    reference: dict[str, float] | None = None,
) -> list[str]:
    """Top-k features pushing toward denial, optionally relative to a reference.

    With ``reference`` this implements the Reg B commentary 9(b)(2)-5 style rule:
    rank factors by how far the applicant's per-factor score falls short of the
    reference population's average for that factor (here: how far the adverse
    contribution exceeds the reference mean). Only strictly positive gaps
    (denial-increasing relative to the reference) are eligible.
    """
    gaps: list[tuple[str, float]] = []
    for feature_name, value in values.items():
        if value is None or not math.isfinite(float(value)):
            continue
        baseline = 0.0
        if reference is not None:
            baseline = float(reference.get(feature_name, 0.0) or 0.0)
        gap = float(value) - baseline
        if gap > 0:
            gaps.append((feature_name, gap))
    gaps.sort(key=lambda item: (-item[1], item[0]))
    return [feature_name for feature_name, _gap in gaps[:k]]


def jaccard(left: list[str] | set[str], right: list[str] | set[str]) -> float | None:
    a, b = set(left), set(right)
    if not a and not b:
        return None
    return len(a & b) / len(a | b)


def overlap_at_k(left: list[str], right: list[str], k: int = 4) -> float | None:
    a, b = left[:k], right[:k]
    if not a or not b:
        return None
    return len(set(a) & set(b)) / min(k, max(len(a), len(b)))


def top1_match(left: list[str], right: list[str]) -> bool | None:
    if not left or not right:
        return None
    return left[0] == right[0]


def mean(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None


def is_null_baseline(explainer_id: str) -> bool:
    return explainer_id in NULL_BASELINES


def load_taxonomy(repo_root: str | Path) -> dict[str, dict[str, dict[str, Any]]]:
    """Return {dataset_id: {feature_name: {labels...}}} from feature_taxonomy.json."""
    path = Path(repo_root) / "benchmark" / "metadata" / "feature_taxonomy.json"
    raw = load_results(path)
    taxonomy: dict[str, dict[str, dict[str, Any]]] = {}
    for dataset_id, entry in raw.get("datasets", {}).items():
        taxonomy[dataset_id] = dict(entry.get("features", {}))
    return taxonomy


# --------------------------------------------------------------------------- #
# Versioned corrections (v2) and data provenance
# --------------------------------------------------------------------------- #

# Mirror of the presets in benchmark/src/legalbench_xai/corrections.py (this module
# never imports the benchmark package); selftest.py checks that they agree.
CORRECTIONS_V1: dict[str, Any] = {
    "native_contributions": False,
    "common_faithfulness_reference": False,
    "faithfulness_quality_weight": 0.20,
    "counterfactual_reasons_from_changes": False,
    "cf_respect_nonactionable": False,
    "reason_grouping_fix": False,
    "numeric_parsing_fix": False,
    "sampling_selection": "boundary_nearest",
    "sampling_seed": 42,
    "persist_score_subcomponents": False,
    "record_data_provenance": False,
}
CORRECTIONS_V2: dict[str, Any] = {
    "native_contributions": True,
    "common_faithfulness_reference": True,
    "faithfulness_quality_weight": 0.0,
    "counterfactual_reasons_from_changes": True,
    "cf_respect_nonactionable": True,
    "reason_grouping_fix": True,
    "numeric_parsing_fix": True,
    "sampling_selection": "stratified_random",
    "sampling_seed": 42,
    "persist_score_subcomponents": True,
    "record_data_provenance": True,
}
CORRECTION_PRESETS: dict[str, dict[str, Any]] = {"v1": CORRECTIONS_V1, "v2": CORRECTIONS_V2}

# faithfulness_evidence["source"] when every explainer is scored against the
# model's native contributions (correction common_faithfulness_reference).
COMMON_REFERENCE_SOURCE = "model_native_common_reference"
# canonical_explanation["metadata"]["cf_reason_source"] values (DiCE/Wachter, v2).
CF_REASON_SOURCES = ("counterfactual_changes", "no_valid_change")
# Persisted parts of CSS decision_sensitivity_visibility (persist_score_subcomponents):
# S = 0.4 directionality + 0.3 threshold_visibility + 0.3 flip_validity.
DSV_SUBCOMPONENT_WEIGHTS = {
    "dsv_directionality": 0.4,
    "dsv_threshold_visibility": 0.3,
    "dsv_flip_validity": 0.3,
}
DEFAULT_METADATA_PATHS = {
    "feature_taxonomy": "benchmark/metadata/feature_taxonomy.json",
    "dataset_registry": "benchmark/metadata/dataset_registry.json",
}

# normalization/rules.py TEMPORAL_GROUP_PREFIXES in its v1 order: "PAY_" is tested
# before "PAY_AMT", so v1 groups PAY_AMT* under "recent late payment history".
TEMPORAL_GROUP_PREFIXES_V1: tuple[tuple[str, str], ...] = (
    ("PAY_", "recent late payment history"),
    ("BILL_AMT", "recent bill balances"),
    ("PAY_AMT", "recent payment amounts"),
)
# Correction reason_grouping_fix: longest prefix first (rules.py _LONGEST_PREFIX_FIRST).
TEMPORAL_GROUP_PREFIXES_FIXED: tuple[tuple[str, str], ...] = tuple(
    sorted(TEMPORAL_GROUP_PREFIXES_V1, key=lambda item: -len(item[0]))
)


def cslaw_section(payload: dict[str, Any]) -> dict[str, Any]:
    block = (payload or {}).get("cslaw")
    return block if isinstance(block, dict) else {}


def bundle_corrections(payload: dict[str, Any]) -> dict[str, Any] | None:
    """The run's correction options, or None for a v1 bundle (no corrections recorded).

    Accepts the recorded forms: ``CorrectionOptions.to_dict()`` (what the runner
    writes), a preset name, or a config-style object ``{"preset": "v2", ...}``.
    Missing options take their v1 defaults.
    """
    raw = cslaw_section(payload).get("corrections")
    if isinstance(raw, str):
        preset = CORRECTION_PRESETS.get(raw)
        return dict(preset) if preset is not None else None
    if not isinstance(raw, dict):
        return None
    base = CORRECTION_PRESETS.get(str(raw.get("preset", "v1")), CORRECTIONS_V1)
    options = dict(base)
    options.update({key: value for key, value in raw.items() if key != "preset"})
    return options


def correction_flag(payload: dict[str, Any], name: str) -> Any:
    """One correction option of the bundle (its v1 default when not recorded)."""
    options = bundle_corrections(payload)
    if options is None or name not in options:
        return CORRECTIONS_V1.get(name)
    return options[name]


def corrections_version(payload: dict[str, Any]) -> str:
    """'v1', 'v2' or 'custom' (a preset with overrides)."""
    marker = cslaw_section(payload).get("corrections_version")
    options = bundle_corrections(payload)
    if options is None:
        return str(marker) if marker else "v1"
    known = {key: options.get(key) for key in CORRECTIONS_V1}
    if known == CORRECTIONS_V1:
        return "v1"
    # Only native_contributions on: the round-1 native-ranking instrumentation
    # (legacy flag), v1 scoring and sampling. The runner does not mark such runs;
    # an options dict of this form is still read as v1.
    if {**known, "native_contributions": CORRECTIONS_V1.get("native_contributions")} == CORRECTIONS_V1:
        return "v1"
    if known == CORRECTIONS_V2:
        return "v2"
    # A preset with overrides; the runner records "custom" next to the full options.
    return "custom"


def uses_corrections(payload: dict[str, Any]) -> bool:
    return corrections_version(payload) != "v1"


def temporal_group_label(feature_name: str, *, fixed: bool = False) -> str | None:
    """Temporal reason family of a feature, mirroring normalization/rules.py.

    fixed=False reproduces the v1 prefix order (PAY_AMT* -> late payment history);
    fixed=True is correction reason_grouping_fix (longest prefix first).
    """
    prefixes = TEMPORAL_GROUP_PREFIXES_FIXED if fixed else TEMPORAL_GROUP_PREFIXES_V1
    for prefix, label in prefixes:
        if feature_name.startswith(prefix):
            return label
    return None


def faithfulness_with_quality_weight(
    coverage: float,
    direction: float,
    quality: float,
    weight: float,
) -> float:
    """(0.45 cov + 0.35 dir + w q) / (0.80 + w); w = 0.20 is the published score."""
    return (0.45 * coverage + 0.35 * direction + weight * quality) / (0.80 + weight)


def bundle_provenance(payload: dict[str, Any]) -> dict[str, Any]:
    block = cslaw_section(payload).get("provenance")
    return block if isinstance(block, dict) else {}


def provenance_metadata_entry(payload: dict[str, Any], name: str) -> dict[str, Any] | None:
    """payload["cslaw"]["provenance"]["metadata"][name] ({"path", "sha256"}) or None."""
    metadata = bundle_provenance(payload).get("metadata")
    if not isinstance(metadata, dict):
        return None
    entry = metadata.get(name)
    return entry if isinstance(entry, dict) and entry.get("path") else None


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_metadata_file(
    payload: dict[str, Any],
    name: str,
    repo_root: str | Path,
) -> tuple[Path, str, str | None]:
    """Metadata file the run used: (path, origin, warning).

    origin is "provenance" when the bundle recorded the file (v2 runs with
    record_data_provenance), else "v1_default". The warning is set when the
    recorded file is missing under ``repo_root`` (the v1 default is NOT used then;
    the caller decides) or when its SHA-256 differs from the recorded one.
    """
    if name not in DEFAULT_METADATA_PATHS:
        raise ValueError(f"unknown metadata file {name!r}")
    entry = provenance_metadata_entry(payload, name)
    if entry is None:
        return Path(repo_root) / DEFAULT_METADATA_PATHS[name], "v1_default", None
    recorded = Path(str(entry["path"]))
    path = recorded if recorded.is_absolute() else Path(repo_root) / recorded
    if not path.is_file():
        return path, "provenance", (
            f"{name}: the bundle records {recorded}, which does not exist under {repo_root}"
        )
    expected = entry.get("sha256")
    if expected and sha256_file(path) != str(expected):
        return path, "provenance", (
            f"{name}: {path} has changed since the run (SHA-256 differs from the recorded "
            f"{str(expected)[:12]}...); labels or exclusions may not match the bundle"
        )
    return path, "provenance", None


def load_taxonomy_file(path: str | Path) -> dict[str, dict[str, dict[str, Any]]]:
    """{dataset_id: {feature_name: {labels...}}} from a feature_taxonomy*.json file."""
    raw = load_results(path)
    taxonomy: dict[str, dict[str, dict[str, Any]]] = {}
    for dataset_id, entry in raw.get("datasets", {}).items():
        taxonomy[dataset_id] = dict(entry.get("features", {}))
    return taxonomy


def corrections_summary_line(payload: dict[str, Any]) -> str:
    """One markdown line describing the bundle's correction status."""
    version = corrections_version(payload)
    if version == "v1":
        return (
            "corrections: v1 (published behaviour: self-referential faithfulness for "
            "SHAP/LIME/LOO, borrowed DiCE/Wachter reasons, PAY_AMT* grouped as late "
            "payment history)"
        )
    flags = [
        f"{name}={correction_flag(payload, name)}"
        for name in (
            "native_contributions",
            "common_faithfulness_reference",
            "faithfulness_quality_weight",
            "counterfactual_reasons_from_changes",
            "cf_respect_nonactionable",
            "reason_grouping_fix",
            "sampling_selection",
        )
    ]
    return f"corrections: {version} ({', '.join(flags)})"
