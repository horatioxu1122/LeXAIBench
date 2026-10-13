"""Compare explainer principal reasons with Reg B commentary reference methods.

Legal grounding (our reading, stated cautiously; not legal advice)
------------------------------------------------------------------
Regulation B, 12 CFR 1002.9(b)(2), requires a creditor that takes adverse
action to provide a statement of the specific principal reasons for it. The
Official Interpretations (Supplement I to Part 1002) add:

* comment 9(b)(2)-2: the disclosed reasons must relate to and accurately
  describe the factors actually considered or scored by the creditor;
* comment 9(b)(2)-4: no factor that was a principal reason for adverse action
  may be excluded from the disclosure;
* comment 9(b)(2)-5: for credit scoring systems, it describes acceptable
  methods for selecting the reasons, among them (method 1) identifying the
  factors for which the applicant's score fell furthest below the average score
  for each factor achieved by applicants whose total score was at or slightly
  above the minimum passing score, and (method 2) the same comparison against
  the average score for each factor achieved by all applicants.

Operationalization (ours; NOT a regulator-endorsed algorithm)
-------------------------------------------------------------
* "Per-factor score" = the fitted model's own per-feature contribution in
  log-odds of the ADVERSE class (instance["native_contributions"]; positive =
  pushes toward denial). This is exact for additive models (logistic
  regression: coef x standardized value; EBM: eval_terms with pair-interaction
  terms split 50/50 between the two features). For XGBoost the "per-factor
  score" is the TreeSHAP decomposition (pred_contribs), which is a modelling
  choice rather than a property of the model, so XGBoost is reported separately.
* "Fell furthest below the reference average" = largest positive gap
  (applicant contribution minus the reference population's mean contribution),
  see common.top_adverse_features.
* M1 reference = approved applicants closest to the cutoff from the approved
  side (payload["cslaw"]["groups"][...]["regb_reference"]["near_cutoff"]);
  M2 reference = a seeded sample of all eval applicants ("all_applicants").
* The two commentary methods can disagree with each other; we report that too.
* Unit of comparison. The headline metrics (``*_grouped`` columns, the markdown
  "(reasons)" columns and the leader lines) compare notice reasons: each stated
  reason is one unit (a compressed PAY_*/BILL_AMT* temporal reason counts once)
  and the M1/M2 gap rankings are mapped to the same units before taking the top
  k. The columns without ``_grouped`` are feature-level: they flatten reasons
  into supporting features and keep the first k, so one temporal reason can
  fill several slots; they are kept for reference only.

By-construction flags (read before interpreting agreement)
----------------------------------------------------------
(i)  SHAP on logistic regression and on XGBoost decomposes the same quantity as
     the native contributions (LR SHAP = coef x (x - mean of the first 64
     training rows), a per-feature constant shift of the native values; XGB
     SHAP = pred_contribs relative to the expected value), so agreement with
     the references, especially M2, is expected to be near-trivial.
(ii) v1 bundles: DiCE/Wachter referenced features ARE the top ranking
     contributions (native when the run used native ranking, and always on
     logistic regression, whose ranking contributions already equal coef x
     standardized value). v2 bundles (correction
     counterfactual_reasons_from_changes): their reasons are the features their
     counterfactuals change, so the flag is dropped; only the order of tied
     features and the weights come from the native contributions (caveat
     ``cf_reason_order_from_native``; their signed_direction is the native
     sign, so v2 faithfulness scores their directions as neutral). Anchors'
     candidate conditions are ranked
     by the native contributions whenever native ranking is on (v2 preset
     included), a partial by-construction effect.
(iii) EBM SHAP is computed in probability units, so only rank/set-based
     comparisons are meaningful (all comparisons in this script are rank/set
     based).

Reason units follow the bundle's grouping: v1 bundles mirror the v1 prefix
order of normalization/rules.py (PAY_AMT* grouped as late payment history);
v2 bundles (correction reason_grouping_fix) use the corrected order.

Usage
-----
    python benchmark/cslaw/regb_reference.py --bundle <run dir or results.json> \
        --out-dir <dir> [--k 4] [--n-boot 2000] [--seed 42]

This module also hosts the small shared helpers (bundle loading, CSV writing,
seeded bootstrap, by-construction flags) used by multiplicity.py,
proxy_substitution.py and common_faithfulness.py.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import zlib
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

ADDITIVE_MODELS = frozenset({"logistic_regression", "ebm"})
TREE_MODELS = frozenset({"xgboost"})
REGB_REFERENCES = {"m1": "near_cutoff", "m2": "all_applicants"}
REASON_SETS = ("principal", "principal_adverse", "referenced")
REASON_SET_PREFIX = {"principal": "pr", "principal_adverse": "pra", "referenced": "ref"}

# Mirrors normalization/rules.py TEMPORAL_GROUP_PREFIXES including its v1 order; as
# in the v1 pipeline, PAY_AMT* therefore groups under "recent late payment history".
# v2 bundles (reason_grouping_fix) use common.TEMPORAL_GROUP_PREFIXES_FIXED instead;
# pass fixed=True to the grouping helpers below (see grouping_fixed()).
TEMPORAL_GROUP_PREFIXES: tuple[tuple[str, str], ...] = common.TEMPORAL_GROUP_PREFIXES_V1

LEGAL_GROUNDING_MD = """\
**Legal grounding (our reading; not legal advice).** Regulation B,
12 CFR 1002.9(b)(2), requires a statement of specific principal reasons for
adverse action. The Official Interpretations state that disclosed reasons must
relate to and accurately describe the factors actually considered or scored
(comment 9(b)(2)-2) and that no factor that was a principal reason may be
excluded (comment 9(b)(2)-4). For credit scoring systems, comment 9(b)(2)-5
describes acceptable methods, including (method 1, "M1") identifying the
factors for which the applicant's score fell furthest below the average score
for each factor among applicants whose total score was at or slightly above the
minimum passing score, and (method 2, "M2") the same comparison against the
average of all applicants.

**Operationalization (ours, not a regulator-endorsed algorithm).** Per-factor
scores are the fitted model's native per-feature contributions in log-odds of
the adverse outcome (exact for logistic regression and for EBM with
pair-interaction terms split 50/50; a TreeSHAP decomposition for XGBoost).
"Fell furthest below" is the largest positive gap between the applicant's
contribution and the reference population's mean contribution. M1's reference
population is the approved applicants closest to the decision cutoff; M2's is a
seeded sample of all evaluation applicants.
"""

BY_CONSTRUCTION_MD = """\
**By-construction flags.** (i) `shap_lr_linear_decomposition` /
`shap_xgb_treeshap_equals_native`: SHAP on logistic regression and XGBoost
decomposes the same quantity as the native contributions (LR SHAP differs per
feature only by a constant, because its baseline is the mean of the first 64
training rows; XGB SHAP equals `pred_contribs`), so agreement, especially with
the all-applicants reference, is expected to be near-trivial.
(ii) `cf_features_ranked_by_native` / `anchors_candidates_ranked_by_native`:
in v1 bundles DiCE/Wachter referenced features are the top native contributions
(and Anchors candidates are ranked by them) when the run used native ranking,
and always on logistic regression, whose ranking contributions already equal
coef x standardized value. In v2 bundles (correction
`counterfactual_reasons_from_changes`) DiCE/Wachter reasons are the features
their counterfactuals change, so `cf_features_ranked_by_native` is not set; the
caveat `cf_reason_order_from_native` records that ties in change frequency are
ordered, and weighted, by native contribution (rank-based metrics such as top-1
still partly reflect the native order). Each v2 DiCE/Wachter reason also copies
its `signed_direction` from the sign of the native contribution, so the v2
pipeline scores counterfactual (and Anchors) directions as neutral (0.5) in
faithfulness under the common reference, and common_faithfulness.py leaves their
direction agreement empty. Anchors keeps its flag under native ranking, which
the v2 preset turns on.
(iii) `shap_ebm_probability_units` (caveat): EBM SHAP is in probability units,
so only rank/set comparisons are meaningful. `global_ranking_uses_native_magnitudes`
marks the canned-reasons null baseline; `stub_model_heuristic_everywhere` marks
cells where a real model backend failed and the stub's heuristic contributions
stand in for both the explanation and the "native" reference. The caveat
`ranking_uses_heuristic_proxy` marks DiCE/Wachter/Anchors on EBM/XGBoost without
native ranking (v1 default): their feature ranking comes from a taxonomy-weighted
heuristic, not the fitted model. Rows with `by_construction=True` should not be
read as evidence that an explainer "finds" the legally relevant reasons.
"""


def native_ranking_enabled(payload: dict[str, Any]) -> bool:
    """True when explainers ranked candidates by native contributions.

    Either the round-1 flag ``cslaw.use_native_contributions_for_ranking`` or the
    correction ``native_contributions`` (part of the v2 preset) turns it on.
    """
    return bool(cslaw_config(payload).get("use_native_contributions_for_ranking")) or bool(
        common.correction_flag(payload, "native_contributions")
    )


def cf_reasons_from_changes(payload: dict[str, Any]) -> bool:
    """True when DiCE/Wachter reasons come from their counterfactual changes (v2)."""
    return bool(common.correction_flag(payload, "counterfactual_reasons_from_changes"))


def grouping_fixed(payload: dict[str, Any]) -> bool:
    """True when the run grouped temporal reasons with the corrected prefix order."""
    return bool(common.correction_flag(payload, "reason_grouping_fix"))


def settings_line(payload: dict[str, Any]) -> str:
    """Markdown line with the bundle settings that change how its reasons are read."""
    return (
        f"{common.corrections_summary_line(payload)}; native ranking = "
        f"{native_ranking_enabled(payload)}; DiCE/Wachter reasons from counterfactual "
        f"changes = {cf_reasons_from_changes(payload)}; corrected temporal grouping = "
        f"{grouping_fixed(payload)}."
    )


# ---------------------------------------------------------------------------
# Shared helpers (imported by the sibling scripts)
# ---------------------------------------------------------------------------


def resolve_results_path(bundle: str | Path) -> Path:
    path = Path(bundle).expanduser()
    if path.is_dir():
        path = path / "results.json"
    if not path.is_file():
        raise SystemExit(f"results.json not found for bundle: {bundle}")
    return path.resolve()


def bundle_label(bundle: str | Path) -> str:
    path = Path(bundle).expanduser().resolve()
    return path.parent.name if path.is_file() else path.name


def load_bundle(bundle: str | Path) -> tuple[dict[str, Any], Path]:
    path = resolve_results_path(bundle)
    return common.load_results(path), path


def cslaw_block(payload: dict[str, Any]) -> dict[str, Any]:
    block = payload.get("cslaw")
    return block if isinstance(block, dict) else {}


def cslaw_config(payload: dict[str, Any]) -> dict[str, Any]:
    config = cslaw_block(payload).get("config")
    return config if isinstance(config, dict) else {}


def cslaw_groups(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    groups = cslaw_block(payload).get("groups")
    return groups if isinstance(groups, dict) else {}


def clean_values(values: Any) -> dict[str, float]:
    if not isinstance(values, dict):
        return {}
    clean: dict[str, float] = {}
    for feature_name, value in values.items():
        if isinstance(value, bool) or value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            clean[str(feature_name)] = number
    return clean


def native_values(instance: dict[str, Any]) -> dict[str, float] | None:
    block = instance.get("native_contributions")
    if not isinstance(block, dict):
        return None
    values = clean_values(block.get("values"))
    return values or None


def native_source(instance: dict[str, Any]) -> str | None:
    block = instance.get("native_contributions")
    if isinstance(block, dict) and block.get("source") is not None:
        return str(block["source"])
    return None


def regb_reference_block(
    payload: dict[str, Any],
    dataset_id: str,
    track: str,
    model_id: str,
) -> dict[str, Any] | None:
    group = cslaw_groups(payload).get(common.cslaw_group_key(dataset_id, track, model_id))
    if not isinstance(group, dict):
        return None
    block = group.get("regb_reference")
    return block if isinstance(block, dict) else None


def regb_means(
    payload: dict[str, Any],
    dataset_id: str,
    track: str,
    model_id: str,
    reference: str,
) -> dict[str, float] | None:
    block = regb_reference_block(payload, dataset_id, track, model_id) or {}
    population = block.get(reference)
    if not isinstance(population, dict):
        return None
    means = clean_values(population.get("means"))
    return means or None


def model_class(model_id: str) -> str:
    if model_id in ADDITIVE_MODELS:
        return "additive"
    if model_id in TREE_MODELS:
        return "tree_shap"
    return "other"


def model_backend_kind(summary_row: dict[str, Any]) -> str:
    backend = summary_row.get("model_backend")
    if isinstance(backend, dict):
        return str(backend.get("backend") or "unknown")
    return "unknown"


def by_construction_flags(
    explainer_id: str,
    model_id: str,
    *,
    native_ranking: bool,
    model_backend: str = "real",
    contribution_source: str | None = None,
    cf_reasons_from_changes: bool = False,
) -> tuple[list[str], list[str]]:
    """Return (by_construction_flags, caveat_flags) for one explainer x model cell.

    by-construction flags mark cells whose agreement with native contributions or
    the Reg B references is (partly) guaranteed by how the explanation is built.
    Caveat flags mark unit/decomposition issues that do not imply agreement.

    ``cf_reasons_from_changes`` (v2 correction counterfactual_reasons_from_changes):
    DiCE/Wachter reasons are the features their counterfactuals change, so they
    are no longer borrowed from the ranking; only the order of features changed
    equally often (and the weights) comes from the native contributions.
    """
    flags: list[str] = []
    caveats: list[str] = []
    stub_model = model_backend == "stub" or contribution_source == "heuristic_stub"
    ranking_is_native = native_ranking or model_id == "logistic_regression" or stub_model
    recourse = explainer_id in common.RECOURSE_EXPLAINERS
    borrows_ranking = recourse and not cf_reasons_from_changes
    ranks_by_contributions = borrows_ranking or explainer_id == "anchors"
    if explainer_id == "shap":
        if model_id == "logistic_regression":
            flags.append("shap_lr_linear_decomposition")
        elif model_id == "xgboost":
            flags.append("shap_xgb_treeshap_equals_native")
        elif model_id == "ebm":
            caveats.append("shap_ebm_probability_units")
    if borrows_ranking and ranking_is_native:
        flags.append("cf_features_ranked_by_native")
    if recourse and cf_reasons_from_changes:
        caveats.append("cf_reason_order_from_native")
    if explainer_id == "anchors" and ranking_is_native:
        flags.append("anchors_candidates_ranked_by_native")
    if ranks_by_contributions and not ranking_is_native:
        # EBM/XGBoost without native ranking: features are ranked by the heuristic
        # stub proxy (taxonomy-weighted), not by the fitted model.
        caveats.append("ranking_uses_heuristic_proxy")
    if explainer_id == "global_ranking":
        flags.append("global_ranking_uses_native_magnitudes")
    if stub_model:
        # Stub models (fallback when a real backend fails) expose the heuristic
        # contributions as "native", and stub explainers read the same values.
        flags.append("stub_model_heuristic_everywhere")
    if model_id == "xgboost":
        caveats.append("xgb_native_is_treeshap")
    if model_id == "ebm":
        caveats.append("ebm_pairs_split_50_50")
    return flags, caveats


def reason_units(
    canonical: dict[str, Any] | None,
    reason_set: str = "principal",
    limit: int = 4,
) -> list[list[str]]:
    """Reasons as lists of supporting features (one list per notice reason)."""
    if not canonical:
        return []
    if reason_set == "referenced":
        return [[name] for name in common.referenced_feature_names(canonical)]
    if reason_set not in ("principal", "principal_adverse"):
        raise ValueError(f"unknown reason_set: {reason_set}")
    reasons = sorted(
        canonical.get("principal_reasons", []) or [],
        key=lambda reason: reason.get("rank", 0),
    )[:limit]
    if reason_set == "principal_adverse":
        reasons = [
            reason for reason in reasons if reason.get("signed_direction") == "increases_denial"
        ]
    return [
        list(reason.get("supporting_features") or [])
        for reason in reasons
        if reason.get("supporting_features")
    ]


def flatten_units(units: Iterable[Sequence[str]]) -> list[str]:
    ordered: list[str] = []
    for unit in units:
        for feature_name in unit:
            if feature_name not in ordered:
                ordered.append(feature_name)
    return ordered


def reason_features(
    canonical: dict[str, Any] | None,
    reason_set: str = "principal",
    limit: int = 4,
) -> list[str]:
    if reason_set == "principal":
        return common.principal_reason_features(canonical, limit=limit)
    return flatten_units(reason_units(canonical, reason_set, limit))


def temporal_group(feature_name: str, *, fixed: bool = False) -> str:
    """Temporal family label of a feature (the feature name itself otherwise).

    fixed=False mirrors the v1 pipeline grouping; fixed=True the corrected one
    (v2 reason_grouping_fix). Use the setting of the bundle (grouping_fixed()).
    """
    label = common.temporal_group_label(feature_name, fixed=fixed)
    return label if label is not None else feature_name


def grouped_units(
    features: Sequence[str],
    k: int | None = None,
    *,
    fixed: bool = False,
) -> list[str]:
    ordered: list[str] = []
    for feature_name in features:
        unit = temporal_group(feature_name, fixed=fixed)
        if unit not in ordered:
            ordered.append(unit)
    return ordered[:k] if k is not None else ordered


def reason_unit_labels(
    canonical: dict[str, Any] | None,
    reason_set: str = "principal",
    limit: int = 4,
    *,
    fixed: bool = False,
) -> list[str]:
    """One label per notice reason, in notice order.

    A compressed temporal reason (all PAY_*/BILL_AMT* members of one family)
    becomes its family label; a raw-feature reason keeps its feature name. This
    is the unit the adverse-action notice actually states, so a grouped reason
    occupies one of the k slots, not one slot per supporting feature.
    """
    return grouped_units(flatten_units(reason_units(canonical, reason_set, limit)), fixed=fixed)


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def cell_seed(seed: int, key: Sequence[Any], metric: str) -> int:
    text = f"{seed}|{'|'.join(str(part) for part in key)}|{metric}"
    return zlib.crc32(text.encode("utf-8"))


def bootstrap_mean_ci(
    values: Sequence[float | None],
    *,
    strata: Sequence[Any] | None = None,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Percentile bootstrap CI of the mean (resampling within strata if given)."""
    pairs = [
        (strata[index] if strata is not None else None, float(value))
        for index, value in enumerate(values)
        if value is not None
    ]
    n = len(pairs)
    if n == 0:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0}
    point = sum(value for _stratum, value in pairs) / n
    if n < 2 or n_boot <= 0:
        return {"mean": point, "ci_low": None, "ci_high": None, "n": n}
    grouped: dict[Any, list[float]] = defaultdict(list)
    for stratum, value in pairs:
        grouped[stratum].append(value)
    rng = np.random.default_rng(seed)
    totals = np.zeros(n_boot, dtype=float)
    for stratum in sorted(grouped, key=repr):
        stratum_values = np.asarray(grouped[stratum], dtype=float)
        size = len(stratum_values)
        chunk = max(1, 4_000_000 // max(1, size))
        start = 0
        while start < n_boot:
            stop = min(n_boot, start + chunk)
            draws = rng.integers(0, size, size=(stop - start, size))
            totals[start:stop] += stratum_values[draws].sum(axis=1)
            start = stop
    stats = totals / n
    low, high = np.quantile(stats, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {"mean": point, "ci_low": float(low), "ci_high": float(high), "n": n}


def _sort_token(value: Any) -> tuple[int, str]:
    return (0 if value is None else 1, str(value))


def summarize(
    rows: Sequence[dict[str, Any]],
    *,
    group_cols: Sequence[str],
    metrics: Sequence[str],
    strata_cols: Sequence[str] = (),
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> list[dict[str, Any]]:
    """Long-format summary: one row per (group, metric) with mean and bootstrap CI."""
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row.get(column) for column in group_cols)].append(row)
    summary: list[dict[str, Any]] = []
    for key in sorted(buckets, key=lambda item: tuple(_sort_token(part) for part in item)):
        members = buckets[key]
        strata = (
            [tuple(member.get(column) for column in strata_cols) for member in members]
            if strata_cols
            else None
        )
        for metric in metrics:
            values = [as_float(member.get(metric)) for member in members]
            stat = bootstrap_mean_ci(
                values,
                strata=strata,
                n_boot=n_boot,
                seed=cell_seed(seed, key, metric),
                alpha=alpha,
            )
            if stat["n"] == 0:
                continue
            summary.append({**dict(zip(group_cols, key)), "metric": metric, **stat})
    return summary


POOLED = "(pooled)"


def summarize_levels(
    rows: Sequence[dict[str, Any]],
    *,
    levels: Sequence[tuple[str, Sequence[str], Sequence[str]]],
    all_cols: Sequence[str],
    metrics: Sequence[str],
    n_boot: int = 2000,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Run ``summarize`` at several aggregation levels into one long table.

    ``levels`` holds (level_name, group_cols, strata_cols). Columns of
    ``all_cols`` that a level does not group by are filled with "(pooled)".
    """
    output: list[dict[str, Any]] = []
    for level_name, group_cols, strata_cols in levels:
        for row in summarize(
            rows,
            group_cols=group_cols,
            metrics=metrics,
            strata_cols=strata_cols,
            n_boot=n_boot,
            seed=seed,
        ):
            filled = {"level": level_name}
            for column in all_cols:
                filled[column] = row.get(column, POOLED) if column in group_cols else POOLED
            filled.update(
                {key: row[key] for key in ("metric", "mean", "ci_low", "ci_high", "n")}
            )
            output.append(filled)
    return output


def index_summary(
    rows: Sequence[dict[str, Any]],
    key_cols: Sequence[str],
) -> dict[tuple[Any, ...], dict[str, Any]]:
    """Index long-format summary rows by (*key_cols, metric)."""
    return {
        tuple(row.get(column) for column in key_cols) + (row["metric"],): row for row in rows
    }


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return "" if not math.isfinite(value) else repr(value)
    if isinstance(value, (list, tuple)):
        return "|".join(str(item) for item in value)
    return value


def write_csv(
    path: str | Path,
    rows: Sequence[dict[str, Any]],
    fieldnames: Sequence[str] | None = None,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        ordered: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for column in row:
                if column not in seen:
                    seen.add(column)
                    ordered.append(column)
        fieldnames = ordered
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _csv_value(row.get(column)) for column in fieldnames})
    return target


def write_text(path: str | Path, text: str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def assert_outputs_safe(outputs: Iterable[Path], inputs: Iterable[Path]) -> None:
    protected = {Path(path).resolve() for path in inputs}
    for output in outputs:
        if Path(output).resolve() in protected:
            raise SystemExit(f"refusing to overwrite an input file: {output}")


def fmt_stat(stat: dict[str, Any] | None, digits: int = 3) -> str:
    if not stat or stat.get("mean") is None:
        return "n/a"
    text = f"{stat['mean']:.{digits}f}"
    if stat.get("ci_low") is not None and stat.get("ci_high") is not None:
        text += f" [{stat['ci_low']:.{digits}f}, {stat['ci_high']:.{digits}f}]"
    return text


def md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(str(header) for header in headers) + " |",
        "| " + " | ".join("---" for _header in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def join_features(features: Sequence[str] | None) -> str:
    return "|".join(features or [])


# ---------------------------------------------------------------------------
# Reg B reference comparison (pure functions)
# ---------------------------------------------------------------------------


def regb_ranking(
    values: dict[str, float],
    reference: dict[str, float] | None,
    k: int,
) -> list[str] | None:
    if reference is None:
        return None
    return common.top_adverse_features(values, k=k, reference=reference)


def regb_unit_ranking(
    values: dict[str, float],
    reference: dict[str, float] | None,
    k: int,
    *,
    fixed: bool = False,
) -> list[str] | None:
    """Top-k Reg B reason units: the full gap ranking with temporal families collapsed.

    Collapsing before truncating gives k notice-level units even when several
    members of one temporal family rank near the top.
    """
    if reference is None:
        return None
    full = common.top_adverse_features(values, k=len(values), reference=reference)
    return grouped_units(full, k, fixed=fixed)


def regb_reason_units(
    values: dict[str, float],
    reference: dict[str, float] | None,
    k: int,
    *,
    fixed: bool = False,
) -> list[list[str]] | None:
    """Reg B reasons as notice-style units (feature lists), mirroring the canonicalizer.

    Walks the full gap ranking and stops at the first feature of a (k+1)-th
    unit. Each unit (a temporal family or a single feature) lists the features
    met on the way, so the unit labels equal ``regb_unit_ranking`` and a
    temporal family is one reason, as on the explainers' notices.
    """
    if reference is None:
        return None
    full = common.top_adverse_features(values, k=len(values), reference=reference)
    units: dict[str, list[str]] = {}
    for feature_name in full:
        label = temporal_group(feature_name, fixed=fixed)
        if label not in units:
            if len(units) >= k:
                break
            units[label] = []
        units[label].append(feature_name)
    return list(units.values())


# Reason sets compared at the notice-reason level (one unit per stated reason).
GROUPED_REASON_SETS = ("principal", "principal_adverse")


def instance_metric_names() -> list[str]:
    names: list[str] = []
    for prefix in REASON_SET_PREFIX.values():
        for reference in REGB_REFERENCES:
            names.extend(
                [
                    f"{prefix}_overlap_{reference}",
                    f"{prefix}_jaccard_{reference}",
                    f"{prefix}_top1_{reference}",
                ]
            )
        names.append(f"{prefix}_outside_m1")
    for reason_set in GROUPED_REASON_SETS:
        prefix = REASON_SET_PREFIX[reason_set]
        for reference in REGB_REFERENCES:
            names.extend(
                [
                    f"{prefix}_overlap_{reference}_grouped",
                    f"{prefix}_jaccard_{reference}_grouped",
                    f"{prefix}_top1_{reference}_grouped",
                ]
            )
        names.append(f"{prefix}_outside_m1_grouped")
    return names


M1_M2_METRICS = ("m1_m2_overlap", "m1_m2_jaccard", "m1_m2_top1")


def compare_reason_lists(
    features: list[str],
    references: dict[str, list[str] | None],
    k: int,
    prefix: str,
) -> dict[str, Any]:
    """Feature-level comparison (temporal groups expanded).

    ``features`` is the flattened supporting-feature list, so one compressed
    temporal reason can fill several of the first k slots and ``outside_m1``
    counts every expanded group member. The notice-level comparison is
    ``compare_reason_units``.
    """
    metrics: dict[str, Any] = {}
    for reference_name, ranked in references.items():
        if ranked is None:
            metrics[f"{prefix}_overlap_{reference_name}"] = None
            metrics[f"{prefix}_jaccard_{reference_name}"] = None
            metrics[f"{prefix}_top1_{reference_name}"] = None
            continue
        metrics[f"{prefix}_overlap_{reference_name}"] = common.overlap_at_k(features, ranked, k)
        metrics[f"{prefix}_jaccard_{reference_name}"] = common.jaccard(features[:k], ranked[:k])
        metrics[f"{prefix}_top1_{reference_name}"] = common.top1_match(features, ranked)
    m1 = references.get("m1")
    if m1 is None or not features:
        metrics[f"{prefix}_outside_m1"] = None
    else:
        m1_set = set(m1)
        metrics[f"{prefix}_outside_m1"] = sum(
            1 for feature_name in features if feature_name not in m1_set
        ) / len(features)
    return metrics


def compare_reason_units(
    units: list[str],
    references: dict[str, list[str] | None],
    k: int,
    prefix: str,
) -> dict[str, Any]:
    """Notice-level comparison: one unit per stated reason (``*_grouped`` metrics).

    ``units`` comes from ``reason_unit_labels`` and each reference from
    ``regb_unit_ranking``, so both sides count a temporal family once.
    ``{prefix}_outside_m1_grouped`` is the share of the notice's reasons whose
    unit is not among M1's top-k units.
    """
    metrics: dict[str, Any] = {}
    for reference_name, ranked in references.items():
        if ranked is None:
            metrics[f"{prefix}_overlap_{reference_name}_grouped"] = None
            metrics[f"{prefix}_jaccard_{reference_name}_grouped"] = None
            metrics[f"{prefix}_top1_{reference_name}_grouped"] = None
            continue
        metrics[f"{prefix}_overlap_{reference_name}_grouped"] = common.overlap_at_k(
            units, ranked, k
        )
        metrics[f"{prefix}_jaccard_{reference_name}_grouped"] = common.jaccard(
            units[:k], ranked[:k]
        )
        metrics[f"{prefix}_top1_{reference_name}_grouped"] = common.top1_match(units, ranked)
    m1 = references.get("m1")
    if m1 is None or not units:
        metrics[f"{prefix}_outside_m1_grouped"] = None
    else:
        m1_set = set(m1)
        metrics[f"{prefix}_outside_m1_grouped"] = sum(
            1 for unit in units if unit not in m1_set
        ) / len(units)
    return metrics


def collect_regb_instances(
    payload: dict[str, Any],
    *,
    k: int = 4,
    bundle: str = "bundle",
) -> tuple[list[dict[str, Any]], Counter]:
    """Per-instance comparison of explainer reasons with Reg B M1/M2 reasons."""
    native_ranking = native_ranking_enabled(payload)
    from_changes = cf_reasons_from_changes(payload)
    fixed = grouping_fixed(payload)
    version = common.corrections_version(payload)
    rows: list[dict[str, Any]] = []
    coverage: Counter = Counter()
    for summary_row, instance in common.iter_instances(payload):
        coverage["instances"] += 1
        dataset_id, track, model_id, explainer_id = common.group_key(summary_row)
        canonical = instance.get("canonical_explanation")
        values = native_values(instance)
        if not canonical:
            coverage["skipped_missing_canonical_explanation"] += 1
            continue
        if values is None:
            coverage["skipped_missing_native_contributions"] += 1
            continue
        reference_means = {
            name: regb_means(payload, dataset_id, track, model_id, population)
            for name, population in REGB_REFERENCES.items()
        }
        if all(means is None for means in reference_means.values()):
            coverage["skipped_missing_regb_reference"] += 1
            continue
        references = {
            name: regb_ranking(values, means, k) for name, means in reference_means.items()
        }
        source = native_source(instance)
        backend = model_backend_kind(summary_row)
        flags, caveats = by_construction_flags(
            explainer_id,
            model_id,
            native_ranking=native_ranking,
            model_backend=backend,
            contribution_source=source,
            cf_reasons_from_changes=from_changes,
        )
        row: dict[str, Any] = {
            "bundle": bundle,
            "corrections_version": version,
            "dataset_id": dataset_id,
            "track": track,
            "model_id": model_id,
            "model_class": model_class(model_id),
            "model_backend": backend,
            "explainer_id": explainer_id,
            "explainer_family": common.EXPLAINER_FAMILY.get(explainer_id, "unknown"),
            "is_null_baseline": common.is_null_baseline(explainer_id),
            "instance_id": instance.get("instance_id"),
            "eval_row_index": common.eval_row_index(instance),
            "native_source": source,
            "by_construction": bool(flags),
            "by_construction_flags": ";".join(flags),
            "caveats": ";".join(caveats),
            "k": k,
            "m1_features": join_features(references["m1"]),
            "m2_features": join_features(references["m2"]),
        }
        for reason_set, prefix in REASON_SET_PREFIX.items():
            features = reason_features(canonical, reason_set)
            row[f"{prefix}_features"] = join_features(features)
            row.update(compare_reason_lists(features, references, k, prefix))
        unit_references = {
            name: regb_unit_ranking(values, means, k, fixed=fixed)
            for name, means in reference_means.items()
        }
        row["m1_units"] = join_features(unit_references["m1"])
        row["m2_units"] = join_features(unit_references["m2"])
        for reason_set in GROUPED_REASON_SETS:
            prefix = REASON_SET_PREFIX[reason_set]
            units = reason_unit_labels(canonical, reason_set, fixed=fixed)
            row[f"{prefix}_units"] = join_features(units)
            row.update(compare_reason_units(units, unit_references, k, prefix))
        m1, m2 = references["m1"], references["m2"]
        if m1 is not None and m2 is not None:
            row["m1_m2_overlap"] = common.overlap_at_k(m1, m2, k)
            row["m1_m2_jaccard"] = common.jaccard(m1, m2)
            row["m1_m2_top1"] = common.top1_match(m1, m2)
        else:
            row["m1_m2_overlap"] = row["m1_m2_jaccard"] = row["m1_m2_top1"] = None
        coverage["used"] += 1
        rows.append(row)
    return rows, coverage


def reference_population_rows(payload: dict[str, Any], bundle: str = "bundle") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups = cslaw_groups(payload)
    for key in sorted(groups):
        group = groups[key]
        block = group.get("regb_reference") if isinstance(group, dict) else None
        if not isinstance(block, dict):
            continue
        dataset_id, track, model_id = (key.split("|") + ["", "", ""])[:3]
        near = block.get("near_cutoff") or {}
        everyone = block.get("all_applicants") or {}
        rows.append(
            {
                "bundle": bundle,
                "dataset_id": dataset_id,
                "track": track,
                "model_id": model_id,
                "source": block.get("source"),
                "threshold": block.get("threshold"),
                "near_cutoff_n_rows": near.get("n_rows"),
                "near_cutoff_score_low": near.get("score_low"),
                "near_cutoff_score_high": near.get("score_high"),
                "all_applicants_n_rows": everyone.get("n_rows"),
                "native_contribution_source": group.get("native_contribution_source"),
                "model_backend": group.get("model_backend"),
            }
        )
    return rows


def dedupe_applicants(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per (bundle, dataset, track, model, applicant) for M1-vs-M2 stats."""
    seen: set[tuple[Any, ...]] = set()
    unique: list[dict[str, Any]] = []
    for row in rows:
        applicant = row.get("eval_row_index")
        if applicant is None:
            applicant = row.get("instance_id")
        key = (row["bundle"], row["dataset_id"], row["track"], row["model_id"], applicant)
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


GROUP_COLS = (
    "bundle",
    "dataset_id",
    "track",
    "model_id",
    "model_class",
    "explainer_id",
    "is_null_baseline",
    "by_construction",
    "by_construction_flags",
)
POOLED_MODEL_COLS = (
    "bundle",
    "model_class",
    "model_id",
    "explainer_id",
    "is_null_baseline",
    "by_construction",
    "by_construction_flags",
)
POOLED_CLASS_COLS = ("bundle", "model_class", "explainer_id", "is_null_baseline")


def summarize_regb(
    rows: Sequence[dict[str, Any]],
    *,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    metrics = instance_metric_names()
    applicants = dedupe_applicants(rows)
    return {
        "by_group": summarize(rows, group_cols=GROUP_COLS, metrics=metrics, n_boot=n_boot, seed=seed),
        "pooled_model": summarize(
            rows,
            group_cols=POOLED_MODEL_COLS,
            metrics=metrics,
            strata_cols=("dataset_id", "track"),
            n_boot=n_boot,
            seed=seed,
        ),
        "pooled_class": summarize(
            rows,
            group_cols=POOLED_CLASS_COLS,
            metrics=[*metrics, "by_construction"],
            strata_cols=("dataset_id", "track", "model_id"),
            n_boot=n_boot,
            seed=seed,
        ),
        "m1_vs_m2_group": summarize(
            applicants,
            group_cols=("bundle", "dataset_id", "track", "model_id", "model_class"),
            metrics=M1_M2_METRICS,
            n_boot=n_boot,
            seed=seed,
        ),
        "m1_vs_m2_pooled": summarize(
            applicants,
            group_cols=("bundle", "model_class", "model_id"),
            metrics=M1_M2_METRICS,
            strata_cols=("dataset_id", "track"),
            n_boot=n_boot,
            seed=seed,
        ),
    }


def leader_lines(
    pooled_model: Sequence[dict[str, Any]],
    model_ids: Sequence[str],
    metric: str,
) -> list[str]:
    """Best real explainer per model on ``metric``.

    Only real explainers WITHOUT by-construction flags are ranked; the headline
    name is never a flagged cell or a null baseline. If the overall maximum is a
    flagged cell it is shown afterwards, labelled "by-construction, not ranked".
    """
    lines: list[str] = []
    for model_id in model_ids:
        candidates = [
            row
            for row in pooled_model
            if row["model_id"] == model_id
            and row["metric"] == metric
            and not row["is_null_baseline"]
            and row["mean"] is not None
        ]
        if not candidates:
            continue
        best = max(candidates, key=lambda row: (row["mean"], row["explainer_id"]))
        clean = [row for row in candidates if not row.get("by_construction")]
        if clean:
            best_clean = max(clean, key=lambda row: (row["mean"], row["explainer_id"]))
            text = (
                f"- `{model_id}`: highest {metric} among real explainers "
                "(by-construction cells excluded) = "
                f"`{best_clean['explainer_id']}` ({fmt_stat(best_clean)})"
            )
        else:
            text = (
                f"- `{model_id}`: no leader on {metric}; every real explainer is "
                "by-construction flagged on this model"
            )
        if best.get("by_construction"):
            text += (
                f"; by-construction, not ranked: `{best['explainer_id']}` ({fmt_stat(best)}) "
                f"[{best.get('by_construction_flags') or 'flagged'}]"
            )
        baselines = sorted(
            (
                row
                for row in pooled_model
                if row["model_id"] == model_id
                and row["metric"] == metric
                and row["is_null_baseline"]
            ),
            key=lambda row: row["explainer_id"],
        )
        if baselines:
            text += "; null baselines (reported, not ranked): " + ", ".join(
                f"`{row['explainer_id']}` {fmt_stat(row)}" for row in baselines
            )
        lines.append(text)
    return lines


def render_regb_markdown(
    summaries: dict[str, list[dict[str, Any]]],
    coverage: Counter,
    *,
    bundle: str,
    k: int,
    native_ranking: bool,
    output_paths: Sequence[Path],
    settings: str | None = None,
) -> str:
    pooled_model = summaries["pooled_model"]
    index = index_summary(pooled_model, ("model_id", "explainer_id", "by_construction_flags"))
    metrics_shown = (
        ("pr_overlap_m1_grouped", f"overlap@{k} M1 (reasons)"),
        ("pr_overlap_m2_grouped", f"overlap@{k} M2 (reasons)"),
        ("pr_top1_m1_grouped", "top-1 = M1 (reasons)"),
        ("pr_outside_m1_grouped", "reasons outside M1"),
        ("pr_overlap_m1", f"feature-level overlap@{k} M1 (temporal groups expanded)"),
    )
    cells = sorted(
        {
            (
                row["model_class"],
                row["model_id"],
                row["explainer_id"],
                row["by_construction_flags"],
                row["is_null_baseline"],
            )
            for row in pooled_model
        },
        key=lambda item: (item[0], item[1], item[4], item[2], item[3]),
    )

    def table_for(model_class_name: str) -> str:
        table_rows = []
        for cls, model_id, explainer_id, flags, is_null in cells:
            if cls != model_class_name:
                continue
            label = f"{explainer_id} (null baseline)" if is_null else explainer_id
            n_stat = index.get((model_id, explainer_id, flags, "pr_overlap_m1_grouped"))
            table_rows.append(
                [model_id, label, flags or "-"]
                + [
                    fmt_stat(index.get((model_id, explainer_id, flags, metric)))
                    for metric, _label in metrics_shown
                ]
                + [n_stat["n"] if n_stat else 0]
            )
        if not table_rows:
            return "_No rows for this model class in the bundle._"
        return md_table(
            ["model", "explainer", "by-construction flags"]
            + [label for _metric, label in metrics_shown]
            + ["n"],
            table_rows,
        )

    m1m2 = summaries["m1_vs_m2_pooled"]
    m1m2_index = index_summary(m1m2, ("model_id",))
    m1m2_rows = [
        [model_id]
        + [fmt_stat(m1m2_index.get((model_id, metric))) for metric in M1_M2_METRICS]
        + [(m1m2_index.get((model_id, "m1_m2_overlap")) or {}).get("n", 0)]
        for model_id in sorted({row["model_id"] for row in m1m2})
    ]
    additive_models = sorted(
        {row["model_id"] for row in pooled_model if row["model_class"] == "additive"}
    )
    other_models = sorted(
        {row["model_id"] for row in pooled_model if row["model_class"] != "additive"}
    )
    lines = [
        "# Reg B reference-method comparison",
        "",
        f"Bundle: `{bundle}`; k = {k}; reasons = principal reasons of the canonical "
        "explanation (notice content); "
        f"native ranking = {native_ranking}.",
        "",
        *([f"Settings: {settings}", ""] if settings else []),
        "**Unit of comparison.** The `(reasons)` columns and the leader lines compare notice "
        "reasons: each stated reason is one unit, so a compressed temporal reason "
        "(PAY_*/BILL_AMT* family) counts once, and M1/M2 are mapped to the same units before "
        f"taking their top {k} (`*_grouped` columns in the CSVs). The feature-level column "
        f"expands temporal groups into their supporting features and keeps the first {k}, so "
        "one grouped reason can fill several slots; it is shown for reference only.",
        "",
        LEGAL_GROUNDING_MD,
        BY_CONSTRUCTION_MD,
        "## Coverage",
        "",
        md_table(["count", "value"], sorted(coverage.items())),
        "",
        f"## Additive models (primary analysis): explainer principal reasons vs M1/M2 top-{k}",
        "",
        "Means pooled over datasets and tracks with 95% stratified bootstrap CIs "
        "(strata = dataset x track).",
        "",
        table_for("additive"),
        "",
        "## XGBoost (reported separately)",
        "",
        "Caveat: XGBoost has no additive per-factor score; its native contributions are the "
        "TreeSHAP decomposition (`pred_contribs`), so the Reg B reference here is itself a "
        "SHAP-style decomposition and SHAP-on-XGBoost agreement is by construction.",
        "",
        table_for("tree_shap"),
        "",
        "## Do the two commentary methods agree? (M1 vs M2, one row per applicant)",
        "",
        md_table(["model", f"overlap@{k}", "Jaccard", "top-1 match", "n applicants"], m1m2_rows)
        if m1m2_rows
        else "_No applicants with both references._",
        "",
        "## Leaders (null baselines and by-construction cells excluded from ranking)",
        "",
        f"Ranked on reason-level overlap@{k} with M1 (`pr_overlap_m1_grouped`).",
        "",
        *(
            leader_lines(pooled_model, additive_models + other_models, "pr_overlap_m1_grouped")
            or ["- none"]
        ),
        "",
        "Interpretation notes: `reasons outside M1` is the share of the notice's reasons whose "
        "unit is not among the M1 top-k units (a rough indicator of reasons that comment "
        "9(b)(2)-2 might not support); low reason-level overlap means the explainer's notice "
        "would omit reasons that M1 identifies as principal (cf. comment 9(b)(2)-4). These "
        "reason-level statements apply to the `(reasons)` columns only; the feature-level "
        "column and the CSV columns without the `_grouped` suffix (including `pr_outside_m1`, "
        "which counts every expanded temporal-group member) are not notice-level measures. "
        "Per-dataset numbers, the `principal_adverse` (`pra_*`) and `referenced` (`ref_*`) "
        "reason sets are in the CSV files.",
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
    k: int = 4,
    n_boot: int = 2000,
    seed: int = 42,
) -> list[Path]:
    payload, results_path = load_bundle(bundle)
    label = bundle_label(bundle)
    rows, coverage = collect_regb_instances(payload, k=k, bundle=label)
    if not rows:
        reasons = ", ".join(f"{key}={value}" for key, value in sorted(coverage.items()))
        raise SystemExit(
            "regb_reference: no analysable instances in "
            f"{results_path} ({reasons or 'no instances'}). This analysis needs a bundle "
            "produced with cslaw.persist_canonical_explanations=true, "
            "cslaw.persist_native_contributions=true and cslaw.regb_reference.enabled=true; "
            "older bundles store scores only."
        )
    summaries = summarize_regb(rows, n_boot=n_boot, seed=seed)
    out = Path(out_dir)
    targets = {
        "instances": out / "regb_reference_instances.csv",
        "by_group": out / "regb_reference_by_group.csv",
        "pooled_model": out / "regb_reference_pooled_by_model.csv",
        "pooled_class": out / "regb_reference_pooled_by_model_class.csv",
        "m1_vs_m2": out / "regb_m1_vs_m2.csv",
        "populations": out / "regb_reference_populations.csv",
        "markdown": out / "regb_reference_summary.md",
    }
    assert_outputs_safe(targets.values(), [results_path])
    write_csv(targets["instances"], rows)
    write_csv(targets["by_group"], summaries["by_group"])
    write_csv(targets["pooled_model"], summaries["pooled_model"])
    write_csv(targets["pooled_class"], summaries["pooled_class"])
    write_csv(
        targets["m1_vs_m2"],
        [{"level": "dataset_track_model", **row} for row in summaries["m1_vs_m2_group"]]
        + [{"level": "pooled_by_model", **row} for row in summaries["m1_vs_m2_pooled"]],
    )
    write_csv(targets["populations"], reference_population_rows(payload, label))
    write_text(
        targets["markdown"],
        render_regb_markdown(
            summaries,
            coverage,
            bundle=label,
            k=k,
            native_ranking=native_ranking_enabled(payload),
            output_paths=list(targets.values()),
            settings=settings_line(payload),
        ),
    )
    return list(targets.values())


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare explainer principal reasons with Reg B comment 9(b)(2)-5 style "
            "reference methods (M1 near-cutoff, M2 all applicants)."
        )
    )
    parser.add_argument(
        "--bundle", required=True, help="Run directory containing results.json (or the file)."
    )
    parser.add_argument("--out-dir", required=True, help="Directory for CSV and markdown outputs.")
    parser.add_argument("--k", type=int, default=4, help="Number of Reg B reasons (default 4).")
    parser.add_argument(
        "--n-boot", type=int, default=2000, help="Bootstrap resamples (default 2000)."
    )
    parser.add_argument("--seed", type=int, default=42, help="Bootstrap seed (default 42).")
    args = parser.parse_args(argv)
    if args.k < 1:
        parser.error("--k must be >= 1")
    for path in run(args.bundle, args.out_dir, k=args.k, n_boot=args.n_boot, seed=args.seed):
        print(path)


if __name__ == "__main__":
    main()
