"""Inter-rater reliability and metric validity for the expert notice ratings.

Inputs
------
* One ratings CSV per rater: a filled copy of ``ratings_template.csv`` written
  by notices.py, with columns notice_id, rater_id, R1_specificity,
  R2_prohibited_basis, R3_rectification, R4_recourse, R5_overall and comment.
  The rubric is in rating_rubric.md.
* ``notices_key.csv`` from notices.py (notice_id -> bundle, dataset, track,
  model, explainer, instance_id, eval_row_index).
* Optionally, the run bundle(s) the notices were rendered from. They are used
  to look up the benchmark scores of the same explanations, by (explainer,
  instance_id). Instance scores exist in every bundle, so older bundles work
  here too.

Analyses
--------
1. Data quality per rater and item: missing, "unsure", N/A and invalid
   entries. Invalid entries are values outside the scale; they are treated as
   missing and reported.
2. Inter-rater reliability per item:

   * Cohen's kappa (unweighted) for every rater pair and every item;
   * quadratic-weighted Cohen's kappa for every rater pair on the ordinal items;
   * Krippendorff's alpha across all raters, with missing values allowed:
     nominal for the categorical items, ordinal and interval for the ordinal
     items. Implemented here with the coincidence-matrix formulation and checked
     against Krippendorff's (2011) worked example in --selftest. Seeded
     bootstrap 95% CIs resample notices.

   R2 is analysed twice: with "unsure" as a third category, and as yes/no with
   "unsure" treated as missing. R4 is analysed as a 0-2 score (N/A treated as
   missing) and as a nominal applicability judgment (N/A versus rated).
3. Consensus rating per notice:

   * R1, R3 and R5 are the mean across raters;
   * R2 is the yes-share among yes/no answers and whether any rater said yes;
   * R4 is N/A when more raters said N/A than gave a score, and otherwise the
     mean score.
4. Mean consensus ratings with seeded percentile-bootstrap 95% CIs, resampling
   notices within each group:

   * per explainer: null baselines are separate rows and are never ranked;
   * per family, model and dataset: null baselines are excluded, or form their
     own "null_baseline" family row.
5. Spearman correlations between the consensus ratings and the benchmark
   scores of the same explanation:

   * metrics: css, css_no_dsv, comprehensibility, alf and faithfulness, plus
     the CSS components, alf_legality and unified_actionability;
   * uncertainty: percentile-bootstrap CIs and seeded two-sided permutation
     p-values;
   * multiplicity: Holm adjustment over the pre-specified pairs (PRESPECIFIED)
     in the real-explainer scope.

   Definition of css_no_dsv:
   ``css_no_dsv = (0.30*verifiability + 0.30*correctability + 0.15*reason_specificity) / 0.75``,
   i.e. CSS without decision_sensitivity_visibility, renormalized to [0, 1].
   The weights come from common.CSS_WEIGHTS.

If the raters are authors, the ratings are a legal assessment by the authors,
not a human-subjects study, and must be disclosed as such.

Usage
-----
    python benchmark/cslaw/rating_analysis.py --ratings r1.csv r2.csv [r3.csv] \
        --key <packet>/notices_key.csv --bundle <run dir> [--bundle <run dir>] \
        --out-dir <dir> [--seed 42] [--n-boot 2000] [--n-perm 2000]
    python benchmark/cslaw/rating_analysis.py --selftest
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import random
import sys
import tempfile
import zlib
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import notices  # noqa: E402


@dataclass(frozen=True, slots=True)
class RubricItem:
    column: str
    short: str
    kind: str
    scale: tuple[Any, ...]
    allows_na: bool = False


@dataclass(frozen=True, slots=True)
class Variant:
    name: str
    column: str
    level: str
    scale: tuple[Any, ...]
    description: str


ITEMS: tuple[RubricItem, ...] = (
    RubricItem("R1_specificity", "R1", "ordinal", (0, 1, 2)),
    RubricItem("R2_prohibited_basis", "R2", "nominal", ("no", "unsure", "yes")),
    RubricItem("R3_rectification", "R3", "ordinal", (0, 1, 2)),
    RubricItem("R4_recourse", "R4", "ordinal", (0, 1, 2), allows_na=True),
    RubricItem("R5_overall", "R5", "ordinal", (1, 2, 3, 4, 5)),
)
if tuple(item.column for item in ITEMS) != notices.RATING_ITEM_COLUMNS:
    raise RuntimeError("rating items are out of sync with notices.RATING_ITEM_COLUMNS")

VARIANTS: tuple[Variant, ...] = (
    Variant("R1_specificity", "R1_specificity", "ordinal", (0, 1, 2), "R1 specificity (0-2)"),
    Variant(
        "R2_prohibited_basis_3cat",
        "R2_prohibited_basis",
        "nominal",
        ("no", "unsure", "yes"),
        "R2 prohibited basis/proxy, unsure as a category",
    ),
    Variant(
        "R2_prohibited_basis_yes_no",
        "R2_prohibited_basis",
        "nominal",
        ("no", "yes"),
        "R2 prohibited basis/proxy, yes/no (unsure as missing)",
    ),
    Variant("R3_rectification", "R3_rectification", "ordinal", (0, 1, 2), "R3 rectification (0-2)"),
    Variant(
        "R4_recourse_score",
        "R4_recourse",
        "ordinal",
        (0, 1, 2),
        "R4 recourse (0-2; N/A as missing)",
    ),
    Variant(
        "R4_recourse_applicable",
        "R4_recourse",
        "nominal",
        ("not_applicable", "rated"),
        "R4 applicability (N/A vs rated)",
    ),
    Variant("R5_overall", "R5_overall", "ordinal", (1, 2, 3, 4, 5), "R5 overall adequacy (1-5)"),
)

UNSURE_TOKENS = frozenset(
    {"unsure", "u", "?", "dk", "don't know", "dont know", "not sure", "unclear", "maybe"}
)
NA_TOKENS = frozenset({"n/a", "na", "n.a.", "not applicable", "-"})
YES_TOKENS = frozenset({"yes", "y", "true", "t", "1"})
NO_TOKENS = frozenset({"no", "n", "false", "f", "0"})
MISSING_ENTRY: tuple[Any, str, str] = (None, "missing", "")

METRICS = (
    "css",
    "css_no_dsv",
    "comprehensibility",
    "alf",
    "faithfulness",
    "css_verifiability",
    "css_correctability",
    "css_dsv",
    "css_reason_specificity",
    "alf_legality",
    "unified_actionability",
)
RATING_TARGETS = (
    "R1_specificity",
    "R2_yes_share",
    "R3_rectification",
    "R4_recourse",
    "R5_overall",
)
# (rating, metric, expected sign); fixed before ratings are collected and
# mirrored in rating_rubric.md section 6.
PRESPECIFIED: tuple[tuple[str, str, str], ...] = (
    ("R1_specificity", "css_reason_specificity", "+"),
    ("R1_specificity", "comprehensibility", "+"),
    ("R3_rectification", "css_verifiability", "+"),
    ("R3_rectification", "css_correctability", "+"),
    ("R3_rectification", "css", "+"),
    ("R4_recourse", "alf", "+"),
    ("R5_overall", "css", "+"),
    ("R5_overall", "css_no_dsv", "+"),
    ("R5_overall", "comprehensibility", "+"),
    ("R5_overall", "faithfulness", "none"),
)
PRESPECIFIED_EXPECTED = {(rating, metric): sign for rating, metric, sign in PRESPECIFIED}
GROUP_ITEMS = (
    "R1_specificity",
    "R2_yes_share",
    "R2_any_yes",
    "R3_rectification",
    "R4_recourse",
    "R4_na_share",
    "R5_overall",
)
ORDINAL_CONSENSUS_COLUMNS = ("R1_specificity", "R3_rectification", "R5_overall")
KEY_ALIASES = {
    "dataset_id": "dataset",
    "model_id": "model",
    "explainer_id": "explainer",
    "explanation_family": "family",
}
KEY_FIELDS = (
    "condition",
    "bundle_label",
    "dataset",
    "track",
    "model",
    "explainer",
    "family",
    "is_null_baseline",
    "instance_id",
    "eval_row_index",
)
CONSENSUS_FIELDS = (
    "n_raters_with_ratings",
    "R1_specificity",
    "R1_specificity_n",
    "R1_specificity_spread",
    "R2_majority",
    "R2_yes_share",
    "R2_any_yes",
    "R2_yes_count",
    "R2_no_count",
    "R2_unsure_count",
    "R3_rectification",
    "R3_rectification_n",
    "R3_rectification_spread",
    "R4_status",
    "R4_recourse",
    "R4_recourse_n",
    "R4_recourse_spread",
    "R4_na_count",
    "R4_na_share",
    "R5_overall",
    "R5_overall_n",
    "R5_overall_spread",
)
OUTPUT_FILES = {
    "long": "ratings_long.csv",
    "quality": "rater_data_quality.csv",
    "pairwise": "irr_pairwise.csv",
    "alpha": "irr_krippendorff_alpha.csv",
    "consensus": "consensus_ratings.csv",
    "groups": "ratings_by_group.csv",
    "correlations": "metric_correlations.csv",
    "report": "ratings_report.md",
}
PAIRWISE_FIELDS = (
    "variant",
    "level",
    "rater_a",
    "rater_b",
    "n_pairs",
    "percent_agreement",
    "cohen_kappa",
    "quadratic_weighted_kappa",
)
ALPHA_FIELDS = (
    "variant",
    "description",
    "metric",
    "n_raters",
    "alpha",
    "ci_low",
    "ci_high",
    "n_units_pairable",
    "n_pairable_values",
    "n_boot_valid",
)
QUALITY_FIELDS = (
    "rater",
    "file",
    "item",
    "notices_in_key",
    "rows_read",
    "unknown_notice_rows",
    "duplicate_rows",
    "rows_without_notice_id",
    "column_missing",
    "n_ok",
    "n_missing",
    "n_unsure",
    "n_na",
    "n_invalid",
)
GROUP_FIELDS = (
    "level",
    "group",
    "is_null_baseline",
    "n_notices",
    "item",
    "mean",
    "ci_low",
    "ci_high",
    "n",
)
CORRELATION_FIELDS = (
    "scope",
    "rating",
    "metric",
    "prespecified",
    "expected",
    "n",
    "rho",
    "ci_low",
    "ci_high",
    "p_perm",
    "p_holm_prespecified",
    "n_boot_valid",
    "note",
)
LONG_FIELDS = ("notice_id", "rater", "item", "raw", "value", "status")


@dataclass(slots=True)
class RaterRatings:
    rater: str
    path: Path
    entries: dict[str, dict[str, tuple[Any, str, str]]] = field(default_factory=dict)
    comments: dict[str, str] = field(default_factory=dict)
    issues: Counter = field(default_factory=Counter)

    def entry(self, notice_id: str) -> dict[str, tuple[Any, str, str]]:
        found = self.entries.get(notice_id)
        if found is None:
            return {item.column: MISSING_ENTRY for item in ITEMS}
        return found


@dataclass(slots=True)
class ScoreBundle:
    path: Path
    label: str
    scores: dict[tuple[str, str], dict[str, Any]]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def cell_seed(seed: int, *parts: Any) -> int:
    text = "|".join([str(seed), *(str(part) for part in parts)])
    return zlib.crc32(text.encode("utf-8"))


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _category_key(value: Any) -> tuple[int, Any]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (0, float(value))
    return (1, str(value))


def _row_get(row: dict[str, Any], name: str) -> Any:
    target = name.lower()
    for column, value in row.items():
        if column is not None and str(column).strip().lower() == target:
            return value
    return None


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    """Read a CSV (tolerates an Excel BOM and semicolon-delimited exports)."""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        first_line = handle.readline()
        handle.seek(0)
        delimiter = ";" if first_line.count(";") > first_line.count(",") else ","
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            return []
        reader.fieldnames = [str(name).strip() for name in reader.fieldnames]
        return [dict(row) for row in reader]


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}" if math.isfinite(value) else "n/a"
    return str(value)


def _fmt_ci(stat: dict[str, Any] | None, digits: int = 2, key: str = "mean") -> str:
    if not stat or stat.get(key) is None:
        return "n/a"
    text = f"{stat[key]:.{digits}f}"
    if stat.get("ci_low") is not None and stat.get("ci_high") is not None:
        text += f" [{stat['ci_low']:.{digits}f}, {stat['ci_high']:.{digits}f}]"
    return text


# ---------------------------------------------------------------------------
# Parsing (pure)
# ---------------------------------------------------------------------------


def parse_rating(item: RubricItem, raw: Any) -> tuple[Any, str]:
    """Return (value, status) with status in ok | missing | unsure | na | invalid.

    For the nominal item R2, "unsure" is a valid category (status ok).
    """
    text = "" if raw is None else str(raw).strip()
    lowered = text.lower()
    if not lowered:
        return None, "missing"
    if item.kind == "nominal":
        if lowered in YES_TOKENS:
            return "yes", "ok"
        if lowered in NO_TOKENS:
            return "no", "ok"
        if lowered in UNSURE_TOKENS:
            return "unsure", "ok"
        return None, "invalid"
    if lowered in UNSURE_TOKENS:
        return None, "unsure"
    if lowered in NA_TOKENS:
        return None, "na"
    try:
        number = float(lowered)
    except ValueError:
        return None, "invalid"
    if not math.isfinite(number) or not number.is_integer():
        return None, "invalid"
    value = int(number)
    if value not in item.scale:
        return None, "invalid"
    return value, "ok"


def parse_rater_rows(
    rows: Sequence[dict[str, Any]],
    notice_ids: set[str],
) -> tuple[dict[str, dict[str, tuple[Any, str, str]]], dict[str, str], Counter]:
    entries: dict[str, dict[str, tuple[Any, str, str]]] = {}
    comments: dict[str, str] = {}
    issues: Counter = Counter()
    issues["rows_read"] = len(rows)
    columns_present: set[str] = set()
    for row in rows:
        normalized = {
            str(column).strip().lower(): value
            for column, value in row.items()
            if column is not None
        }
        columns_present.update(normalized)
        notice_id = str(normalized.get("notice_id") or "").strip()
        if not notice_id:
            issues["rows_without_notice_id"] += 1
            continue
        if notice_id not in notice_ids:
            issues["unknown_notice_rows"] += 1
            continue
        if notice_id in entries:
            issues["duplicate_rows"] += 1
            continue
        entry: dict[str, tuple[Any, str, str]] = {}
        for item in ITEMS:
            raw = normalized.get(item.column.lower())
            if raw is None:
                raw = normalized.get(item.short.lower())
            value, status = parse_rating(item, raw)
            entry[item.column] = (value, status, "" if raw is None else str(raw).strip())
        entries[notice_id] = entry
        comment = str(normalized.get("comment") or "").strip()
        if comment:
            comments[notice_id] = comment
    for item in ITEMS:
        if item.column.lower() not in columns_present and item.short.lower() not in columns_present:
            issues[f"missing_column:{item.column}"] = 1
    return entries, comments, issues


def resolve_rater_names(
    files: Sequence[tuple[Path, Sequence[dict[str, Any]]]],
    overrides: Sequence[str] | None,
) -> list[str]:
    names: list[str] = []
    for index, (path, rows) in enumerate(files):
        if overrides:
            name = overrides[index]
        else:
            ids = {str(_row_get(row, "rater_id") or "").strip() for row in rows} - {""}
            name = next(iter(ids)) if len(ids) == 1 else path.stem
        base, suffix = name, 2
        while name in names:
            name = f"{base}_{suffix}"
            suffix += 1
        names.append(name)
    return names


def load_key(path: Path) -> dict[str, dict[str, Any]]:
    key: dict[str, dict[str, Any]] = {}
    for row in read_csv_rows(path):
        normalized: dict[str, Any] = {}
        for column, value in row.items():
            if column is None:
                continue
            name = KEY_ALIASES.get(str(column).strip(), str(column).strip())
            normalized[name] = value.strip() if isinstance(value, str) else value
        notice_id = str(normalized.get("notice_id") or "")
        if not notice_id:
            continue
        explainer = str(normalized.get("explainer") or "")
        normalized["explainer"] = explainer
        normalized["family"] = str(
            normalized.get("family") or common.EXPLAINER_FAMILY.get(explainer, "unknown")
        )
        normalized["is_null_baseline"] = common.is_null_baseline(explainer)
        normalized["condition"] = str(normalized.get("condition") or "full")
        key[notice_id] = normalized
    if not key:
        raise SystemExit(f"no notices found in key file {path}")
    return key


# ---------------------------------------------------------------------------
# Agreement statistics (pure)
# ---------------------------------------------------------------------------


def variant_value(variant: Variant, value: Any, status: str) -> Any:
    if variant.name == "R2_prohibited_basis_yes_no":
        return value if status == "ok" and value in ("yes", "no") else None
    if variant.name == "R4_recourse_applicable":
        if status == "na":
            return "not_applicable"
        if status == "ok":
            return "rated"
        return None
    return value if status == "ok" else None


def build_matrix(
    raters: Sequence[RaterRatings],
    notice_ids: Sequence[str],
    variant: Variant,
) -> dict[str, dict[str, Any]]:
    matrix: dict[str, dict[str, Any]] = {}
    for notice_id in notice_ids:
        matrix[notice_id] = {}
        for rater in raters:
            value, status, _raw = rater.entry(notice_id)[variant.column]
            matrix[notice_id][rater.rater] = variant_value(variant, value, status)
    return matrix


def cohen_kappa(pairs: Sequence[tuple[Any, Any]]) -> float | None:
    """Unweighted Cohen's kappa; None when undefined (no pairs or p_e = 1)."""
    n = len(pairs)
    if n == 0:
        return None
    observed = sum(1 for left, right in pairs if left == right) / n
    left_counts = Counter(left for left, _right in pairs)
    right_counts = Counter(right for _left, right in pairs)
    expected = sum(
        left_counts[category] * right_counts[category]
        for category in set(left_counts) | set(right_counts)
    ) / (n * n)
    if expected >= 1.0 - 1e-12:
        return None
    return (observed - expected) / (1.0 - expected)


def weighted_kappa(
    pairs: Sequence[tuple[Any, Any]],
    scale: Sequence[Any],
    *,
    weights: str = "quadratic",
) -> float | None:
    """Weighted Cohen's kappa over the full rating scale (category positions)."""
    if not pairs:
        return None
    categories = sorted(
        set(scale) | {left for left, _ in pairs} | {right for _, right in pairs},
        key=_category_key,
    )
    k = len(categories)
    if k < 2:
        return None
    index = {category: position for position, category in enumerate(categories)}
    observed = np.zeros((k, k), dtype=float)
    for left, right in pairs:
        observed[index[left], index[right]] += 1.0
    total = observed.sum()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / total
    rows, cols = np.indices((k, k))
    if weights == "quadratic":
        weight = (rows - cols) ** 2 / float((k - 1) ** 2)
    elif weights == "linear":
        weight = np.abs(rows - cols) / float(k - 1)
    else:
        raise ValueError(f"unknown kappa weights: {weights}")
    denominator = float((weight * expected).sum())
    if denominator <= 1e-12:
        return None
    return 1.0 - float((weight * observed).sum()) / denominator


def _delta_squared(categories: Sequence[Any], totals: np.ndarray, metric: str) -> np.ndarray:
    k = len(categories)
    delta = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            if metric == "nominal":
                delta[i, j] = 1.0
            elif metric == "interval":
                delta[i, j] = (float(categories[i]) - float(categories[j])) ** 2
            elif metric == "ordinal":
                low, high = min(i, j), max(i, j)
                span = float(totals[low : high + 1].sum()) - (
                    float(totals[low]) + float(totals[high])
                ) / 2.0
                delta[i, j] = span**2
            else:
                raise ValueError(f"unknown alpha metric: {metric}")
    return delta


def krippendorff_alpha(units: Sequence[Sequence[Any]], metric: str) -> float | None:
    """Krippendorff's alpha from per-unit value lists (missing values omitted).

    Coincidence-matrix formulation: alpha = 1 - (n - 1) * sum(o_ck * d_ck) /
    sum(n_c * n_k * d_ck), over pairable units (at least two values). Ordinal
    distances use the observed marginals n_g. Returns None when undefined (no
    pairable units or no variation).
    """
    pairable = [list(values) for values in units if len(values) >= 2]
    if not pairable:
        return None
    categories = sorted({value for values in pairable for value in values}, key=_category_key)
    if len(categories) < 2:
        return None
    index = {category: position for position, category in enumerate(categories)}
    k = len(categories)
    coincidence = np.zeros((k, k), dtype=float)
    for values in pairable:
        m = len(values)
        counts = Counter(values)
        for first, n_first in counts.items():
            for second, n_second in counts.items():
                pairs = n_first * (n_first - 1) if first == second else n_first * n_second
                if pairs:
                    coincidence[index[first], index[second]] += pairs / (m - 1)
    totals = coincidence.sum(axis=1)
    n_total = float(totals.sum())
    delta = _delta_squared(categories, totals, metric)
    observed = float((coincidence * delta).sum())
    expected = float((np.outer(totals, totals) * delta).sum())
    if expected <= 1e-12:
        return None
    return 1.0 - (n_total - 1.0) * observed / expected


def alpha_with_ci(
    units: Sequence[Sequence[Any]],
    metric: str,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    pairable = [list(values) for values in units if len(values) >= 2]
    point = krippendorff_alpha(pairable, metric)
    result: dict[str, Any] = {
        "alpha": point,
        "ci_low": None,
        "ci_high": None,
        "n_units_pairable": len(pairable),
        "n_pairable_values": sum(len(values) for values in pairable),
        "n_boot_valid": 0,
    }
    if point is None or n_boot <= 0 or len(pairable) < 2:
        return result
    rng = random.Random(seed)
    size = len(pairable)
    stats: list[float] = []
    for _ in range(n_boot):
        sample = [pairable[rng.randrange(size)] for _unit in range(size)]
        value = krippendorff_alpha(sample, metric)
        if value is not None:
            stats.append(value)
    result["n_boot_valid"] = len(stats)
    if len(stats) >= max(20, n_boot // 2):
        low, high = np.quantile(np.asarray(stats, dtype=float), [0.025, 0.975])
        result["ci_low"], result["ci_high"] = float(low), float(high)
    return result


def reliability_rows(
    raters: Sequence[RaterRatings],
    notice_ids: Sequence[str],
    *,
    n_boot: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    names = [rater.rater for rater in raters]
    pairwise: list[dict[str, Any]] = []
    alpha_rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        matrix = build_matrix(raters, notice_ids, variant)
        for first, second in itertools.combinations(names, 2):
            pairs = [
                (matrix[notice_id][first], matrix[notice_id][second])
                for notice_id in notice_ids
                if matrix[notice_id][first] is not None and matrix[notice_id][second] is not None
            ]
            n_pairs = len(pairs)
            pairwise.append(
                {
                    "variant": variant.name,
                    "level": variant.level,
                    "rater_a": first,
                    "rater_b": second,
                    "n_pairs": n_pairs,
                    "percent_agreement": (
                        sum(1 for left, right in pairs if left == right) / n_pairs
                        if n_pairs
                        else None
                    ),
                    "cohen_kappa": cohen_kappa(pairs),
                    "quadratic_weighted_kappa": (
                        weighted_kappa(pairs, variant.scale)
                        if variant.level == "ordinal"
                        else None
                    ),
                }
            )
        units = [
            [value for value in matrix[notice_id].values() if value is not None]
            for notice_id in notice_ids
        ]
        metrics = ("nominal",) if variant.level == "nominal" else ("ordinal", "interval")
        for metric in metrics:
            stat = alpha_with_ci(
                units,
                metric,
                n_boot=n_boot,
                seed=cell_seed(seed, "alpha", variant.name, metric),
            )
            alpha_rows.append(
                {
                    "variant": variant.name,
                    "description": variant.description,
                    "metric": metric,
                    "n_raters": len(names),
                    **stat,
                }
            )
    return pairwise, alpha_rows


# ---------------------------------------------------------------------------
# Consensus, data quality and benchmark metrics (pure)
# ---------------------------------------------------------------------------


def _spread(values: Sequence[float]) -> float | None:
    return (max(values) - min(values)) if len(values) >= 2 else None


def consensus_for_notice(entries: Sequence[dict[str, tuple[Any, str, str]]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for column in ORDINAL_CONSENSUS_COLUMNS:
        values = [float(entry[column][0]) for entry in entries if entry[column][1] == "ok"]
        out[column] = sum(values) / len(values) if values else None
        out[f"{column}_n"] = len(values)
        out[f"{column}_spread"] = _spread(values)

    numeric = [
        float(entry["R4_recourse"][0]) for entry in entries if entry["R4_recourse"][1] == "ok"
    ]
    n_na = sum(1 for entry in entries if entry["R4_recourse"][1] == "na")
    if n_na > len(numeric):
        status = "not_applicable"
    elif numeric:
        status = "rated"
    else:
        status = "missing"
    out["R4_status"] = status
    out["R4_recourse"] = sum(numeric) / len(numeric) if status == "rated" else None
    out["R4_recourse_n"] = len(numeric)
    out["R4_recourse_spread"] = _spread(numeric)
    out["R4_na_count"] = n_na
    answered = n_na + len(numeric)
    out["R4_na_share"] = n_na / answered if answered else None

    answers = [
        entry["R2_prohibited_basis"][0]
        for entry in entries
        if entry["R2_prohibited_basis"][1] == "ok"
    ]
    yes, no, unsure = answers.count("yes"), answers.count("no"), answers.count("unsure")
    out["R2_yes_count"] = yes
    out["R2_no_count"] = no
    out["R2_unsure_count"] = unsure
    out["R2_yes_share"] = yes / (yes + no) if (yes + no) else None
    out["R2_any_yes"] = 1.0 if yes else (0.0 if no else None)
    if yes > no:
        majority = "yes"
    elif no > yes:
        majority = "no"
    elif yes:
        majority = "tie"
    elif unsure:
        majority = "unsure"
    else:
        majority = ""
    out["R2_majority"] = majority
    out["n_raters_with_ratings"] = sum(
        1
        for entry in entries
        if any(entry[column][1] in ("ok", "na") for column in notices.RATING_ITEM_COLUMNS)
    )
    return out


def data_quality_rows(
    raters: Sequence[RaterRatings],
    notice_ids: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rater in raters:
        for item in ITEMS:
            counts: Counter = Counter()
            for notice_id in notice_ids:
                value, status, _raw = rater.entry(notice_id)[item.column]
                if status == "ok":
                    counts["unsure" if value == "unsure" else "ok"] += 1
                else:
                    counts[status] += 1
            rows.append(
                {
                    "rater": rater.rater,
                    "file": str(rater.path),
                    "item": item.column,
                    "notices_in_key": len(notice_ids),
                    "rows_read": rater.issues.get("rows_read", 0),
                    "unknown_notice_rows": rater.issues.get("unknown_notice_rows", 0),
                    "duplicate_rows": rater.issues.get("duplicate_rows", 0),
                    "rows_without_notice_id": rater.issues.get("rows_without_notice_id", 0),
                    "column_missing": bool(rater.issues.get(f"missing_column:{item.column}")),
                    "n_ok": counts["ok"],
                    "n_missing": counts["missing"],
                    "n_unsure": counts["unsure"],
                    "n_na": counts["na"],
                    "n_invalid": counts["invalid"],
                }
            )
    return rows


def long_rows(raters: Sequence[RaterRatings], notice_ids: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rater in raters:
        for notice_id in notice_ids:
            entry = rater.entry(notice_id)
            for item in ITEMS:
                value, status, raw = entry[item.column]
                rows.append(
                    {
                        "notice_id": notice_id,
                        "rater": rater.rater,
                        "item": item.column,
                        "raw": raw,
                        "value": value,
                        "status": status,
                    }
                )
            comment = rater.comments.get(notice_id)
            if comment:
                rows.append(
                    {
                        "notice_id": notice_id,
                        "rater": rater.rater,
                        "item": "comment",
                        "raw": comment,
                        "value": None,
                        "status": "ok",
                    }
                )
    return rows


def extract_metrics(scores: dict[str, Any]) -> dict[str, float | None]:
    """Benchmark metrics of one explanation from its instance scores."""
    css = scores.get("contestability_support_score") or {}
    alf = scores.get("actionability_legality_filter") or {}
    verifiability = _finite(css.get("verifiability"))
    correctability = _finite(css.get("correctability"))
    specificity = _finite(css.get("reason_specificity"))
    css_no_dsv = None
    if verifiability is not None and correctability is not None and specificity is not None:
        weights = common.CSS_WEIGHTS
        total = (
            weights["verifiability"] + weights["correctability"] + weights["reason_specificity"]
        )
        css_no_dsv = (
            weights["verifiability"] * verifiability
            + weights["correctability"] * correctability
            + weights["reason_specificity"] * specificity
        ) / total
    return {
        "css": _finite(css.get("score")),
        "css_no_dsv": css_no_dsv,
        "comprehensibility": _finite((scores.get("comprehensibility_score") or {}).get("score")),
        "alf": _finite(alf.get("score")),
        "faithfulness": _finite((scores.get("faithfulness_score") or {}).get("score")),
        "css_verifiability": verifiability,
        "css_correctability": correctability,
        "css_dsv": _finite(css.get("decision_sensitivity_visibility")),
        "css_reason_specificity": specificity,
        "alf_legality": _finite(alf.get("legality")),
        "unified_actionability": _finite(
            (scores.get("unified_actionability_score") or {}).get("score")
        ),
    }


def index_scores(payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for row, instance in common.iter_instances(payload):
        key = (str(row.get("explainer_id", "")), str(instance.get("instance_id", "")))
        index.setdefault(key, instance.get("scores") or {})
    return index


def match_bundle(
    key_row: dict[str, Any],
    bundles: Sequence[ScoreBundle],
) -> tuple[ScoreBundle | None, str]:
    """Find the bundle a notice came from: by path, then label, then single bundle."""
    target = str(key_row.get("bundle") or "")
    if target:
        try:
            resolved: Path | None = Path(target).expanduser().resolve()
        except (OSError, RuntimeError):
            resolved = None
        if resolved is not None:
            for bundle in bundles:
                if bundle.path == resolved:
                    return bundle, "path"
    label = str(key_row.get("bundle_label") or "") or (Path(target).parent.name if target else "")
    if label:
        matches = [bundle for bundle in bundles if bundle.label == label]
        if len(matches) == 1:
            return matches[0], "label"
    if len(bundles) == 1:
        return bundles[0], "single_bundle_fallback"
    return None, "unmatched"


def attach_metrics(
    consensus_rows: Sequence[dict[str, Any]],
    key: dict[str, dict[str, Any]],
    bundles: Sequence[ScoreBundle],
) -> Counter:
    coverage: Counter = Counter()
    for row in consensus_rows:
        key_row = key[row["notice_id"]]
        if bundles:
            bundle, method = match_bundle(key_row, bundles)
        else:
            bundle, method = None, "no_bundle"
        row["bundle_match"] = method
        scores = None
        if bundle is not None:
            scores = bundle.scores.get(
                (str(key_row.get("explainer") or ""), str(key_row.get("instance_id") or ""))
            )
        if scores is None:
            coverage["scores_not_found" if bundle is not None else method] += 1
            for metric in METRICS:
                row[metric] = None
            continue
        coverage[f"matched_by_{method}"] += 1
        row.update(extract_metrics(scores))
    return coverage


# ---------------------------------------------------------------------------
# Group means and correlations (pure)
# ---------------------------------------------------------------------------


def bootstrap_mean_ci(
    values: Iterable[Any],
    *,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> dict[str, Any]:
    clean = np.asarray(
        [number for number in (_finite(value) for value in values) if number is not None],
        dtype=float,
    )
    n = int(clean.size)
    if n == 0:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0}
    point = float(clean.mean())
    if n < 2 or n_boot <= 0:
        return {"mean": point, "ci_low": None, "ci_high": None, "n": n}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(n_boot, n))
    stats = clean[draws].mean(axis=1)
    low, high = np.quantile(stats, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {"mean": point, "ci_low": float(low), "ci_high": float(high), "n": n}


def group_rows(
    consensus_rows: Sequence[dict[str, Any]],
    *,
    n_boot: int,
    seed: int,
) -> list[dict[str, Any]]:
    real = [row for row in consensus_rows if not row["is_null_baseline"]]
    levels: list[tuple[str, Sequence[dict[str, Any]], Any]] = [
        ("explainer", consensus_rows, lambda row: row["explainer"]),
        (
            "family",
            consensus_rows,
            lambda row: "null_baseline" if row["is_null_baseline"] else row["family"],
        ),
        ("model", real, lambda row: row["model"]),
        ("dataset", real, lambda row: row["dataset"]),
        ("all_real_explainers", real, lambda _row: "all"),
    ]
    if len({row.get("condition") for row in consensus_rows}) > 1:
        levels.append(("condition", real, lambda row: row.get("condition")))
    rows: list[dict[str, Any]] = []
    for level, subset, key_fn in levels:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in subset:
            grouped[str(key_fn(row))].append(row)
        for group in sorted(grouped):
            members = grouped[group]
            is_baseline = (level == "explainer" and common.is_null_baseline(group)) or (
                level == "family" and group == "null_baseline"
            )
            for item in GROUP_ITEMS:
                stat = bootstrap_mean_ci(
                    [member.get(item) for member in members],
                    n_boot=n_boot,
                    seed=cell_seed(seed, "group", level, group, item),
                )
                rows.append(
                    {
                        "level": level,
                        "group": group,
                        "is_null_baseline": is_baseline,
                        "n_notices": len(members),
                        "item": item,
                        **stat,
                    }
                )
    return rows


def _pearson_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Pearson correlation along the last axis; NaN where undefined."""
    left = left - left.mean(axis=-1, keepdims=True)
    right = right - right.mean(axis=-1, keepdims=True)
    numerator = (left * right).sum(axis=-1)
    denominator = np.sqrt((left * left).sum(axis=-1) * (right * right).sum(axis=-1))
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / denominator
    return np.where(denominator > 1e-12, np.clip(result, -1.0, 1.0), np.nan)


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Spearman's rho (average ranks for ties); None if undefined or n < 3."""
    left = np.asarray(x, dtype=float)
    right = np.asarray(y, dtype=float)
    if left.size < 3 or left.size != right.size:
        return None
    value = float(_pearson_rows(rankdata(left), rankdata(right)))
    return value if math.isfinite(value) else None


def spearman_with_uncertainty(
    x: Sequence[float],
    y: Sequence[float],
    *,
    n_boot: int,
    n_perm: int,
    seed: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "rho": None,
        "ci_low": None,
        "ci_high": None,
        "p_perm": None,
        "n_boot_valid": 0,
    }
    rho = spearman(x, y)
    if rho is None:
        result["note"] = "undefined (constant ratings or scores)"
        return result
    result["rho"] = rho
    left = np.asarray(x, dtype=float)
    right = np.asarray(y, dtype=float)
    n = left.size
    rng = np.random.default_rng(seed)
    if n_boot > 0:
        draws = rng.integers(0, n, size=(n_boot, n))
        values = _pearson_rows(
            np.asarray(rankdata(left[draws], axis=1), dtype=float),
            np.asarray(rankdata(right[draws], axis=1), dtype=float),
        )
        valid = values[np.isfinite(values)]
        result["n_boot_valid"] = int(valid.size)
        if valid.size >= max(20, n_boot // 2):
            low, high = np.quantile(valid, [0.025, 0.975])
            result["ci_low"], result["ci_high"] = float(low), float(high)
    if n_perm > 0:
        left_ranks = np.asarray(rankdata(left), dtype=float)
        right_ranks = np.asarray(rankdata(right), dtype=float)
        order = np.argsort(rng.random((n_perm, n)), axis=1)
        permuted = right_ranks[order]
        values = _pearson_rows(np.broadcast_to(left_ranks, permuted.shape), permuted)
        exceed = int(np.sum(np.abs(values) >= abs(rho) - 1e-12))
        result["p_perm"] = (exceed + 1) / (n_perm + 1)
    return result


def holm_adjust(pvalues: Sequence[float | None]) -> list[float | None]:
    """Holm step-down adjusted p-values; None entries are left as None."""
    indexed = sorted((value, position) for position, value in enumerate(pvalues) if value is not None)
    m = len(indexed)
    adjusted: list[float | None] = [None] * len(pvalues)
    running = 0.0
    for rank, (value, position) in enumerate(indexed):
        running = max(running, min(1.0, (m - rank) * value))
        adjusted[position] = running
    return adjusted


def correlation_rows(
    consensus_rows: Sequence[dict[str, Any]],
    *,
    n_boot: int,
    n_perm: int,
    seed: int,
    min_n: int,
) -> list[dict[str, Any]]:
    min_n = max(3, min_n)
    all_pairs = [(rating, metric) for rating in RATING_TARGETS for metric in METRICS]
    prespecified_pairs = [(rating, metric) for rating, metric, _sign in PRESPECIFIED]
    real = [row for row in consensus_rows if not row["is_null_baseline"]]
    scopes: list[tuple[str, Sequence[dict[str, Any]], Sequence[tuple[str, str]]]] = [
        ("real_explainers", real, all_pairs)
    ]
    if len(real) < len(consensus_rows):
        scopes.append(("including_null_baselines", list(consensus_rows), all_pairs))
    for explainer in sorted({row["explainer"] for row in real}):
        scopes.append(
            (
                f"within_explainer:{explainer}",
                [row for row in real if row["explainer"] == explainer],
                prespecified_pairs,
            )
        )
    rows: list[dict[str, Any]] = []
    for scope, subset, pairs in scopes:
        for rating, metric in pairs:
            xs: list[float] = []
            ys: list[float] = []
            for row in subset:
                x_value = _finite(row.get(rating))
                y_value = _finite(row.get(metric))
                if x_value is None or y_value is None:
                    continue
                xs.append(x_value)
                ys.append(y_value)
            stat: dict[str, Any] = {
                "scope": scope,
                "rating": rating,
                "metric": metric,
                "prespecified": (rating, metric) in PRESPECIFIED_EXPECTED,
                "expected": PRESPECIFIED_EXPECTED.get((rating, metric), ""),
                "n": len(xs),
            }
            if len(xs) < min_n:
                stat.update({"rho": None, "note": f"n < {min_n}"})
            else:
                stat.update(
                    spearman_with_uncertainty(
                        xs,
                        ys,
                        n_boot=n_boot,
                        n_perm=n_perm,
                        seed=cell_seed(seed, "spearman", scope, rating, metric),
                    )
                )
            rows.append(stat)
    primary = [row for row in rows if row["scope"] == "real_explainers" and row["prespecified"]]
    adjusted = holm_adjust([row.get("p_perm") for row in primary])
    for row, value in zip(primary, adjusted, strict=True):
        row["p_holm_prespecified"] = value
    return rows


# ---------------------------------------------------------------------------
# Report (pure)
# ---------------------------------------------------------------------------

DISCLOSURE_MD = """\
**Disclosure.** If the raters are authors of the paper, these ratings are a
legal assessment by the authors, not a human-subjects study. The paper must
say so, including who rated and their legal training, that they knew the
study's aims, and that notices were rendered without method or model names.
It must also say that the method family was not fully blindable, because the
recourse section identifies counterfactual explanations. If external raters
were recruited, confirm the institution's human-subjects (IRB) requirements
before reporting.
"""


def _group_table(rows: Sequence[dict[str, Any]], level: str) -> str:
    lookup = {(row["group"], row["item"]): row for row in rows if row["level"] == level}
    groups = sorted(
        {group for group, _item in lookup},
        key=lambda group: (bool(lookup[(group, GROUP_ITEMS[0])]["is_null_baseline"]), group),
    )
    table_rows = []
    for group in groups:
        first = lookup[(group, GROUP_ITEMS[0])]
        label = f"{group} (null baseline)" if first["is_null_baseline"] and level == "explainer" else group
        r4 = lookup.get((group, "R4_recourse"))
        table_rows.append(
            [
                label,
                first["n_notices"],
                _fmt_ci(lookup.get((group, "R1_specificity"))),
                _fmt_ci(lookup.get((group, "R2_yes_share"))),
                _fmt_ci(lookup.get((group, "R3_rectification"))),
                f"{_fmt_ci(r4)} (n={r4['n'] if r4 else 0})",
                _fmt_ci(lookup.get((group, "R4_na_share"))),
                _fmt_ci(lookup.get((group, "R5_overall"))),
            ]
        )
    return notices.md_table(
        [
            level,
            "notices",
            "R1 specificity",
            "R2 yes-share",
            "R3 rectification",
            "R4 recourse",
            "R4 N/A share",
            "R5 overall",
        ],
        table_rows,
    )


def _rho_text(row: dict[str, Any]) -> str:
    if row.get("rho") is None:
        return f"n/a ({row.get('note') or 'undefined'})"
    return _fmt_ci(row, key="rho")


def render_report(
    *,
    raters: Sequence[RaterRatings],
    key: dict[str, dict[str, Any]],
    consensus: Sequence[dict[str, Any]],
    quality: Sequence[dict[str, Any]],
    pairwise: Sequence[dict[str, Any]],
    alpha: Sequence[dict[str, Any]],
    groups: Sequence[dict[str, Any]],
    correlations: Sequence[dict[str, Any]],
    metric_coverage: Counter,
    bundles: Sequence[ScoreBundle],
    n_boot: int,
    n_perm: int,
    seed: int,
    output_paths: Sequence[Path],
) -> str:
    rated_any = sum(1 for row in consensus if row["n_raters_with_ratings"] > 0)
    rated_all = sum(1 for row in consensus if row["n_raters_with_ratings"] == len(raters))
    conditions = Counter(str(row.get("condition")) for row in consensus)
    lines = [
        "# Expert ratings of rendered notices: reliability and metric validity",
        "",
        DISCLOSURE_MD,
        "## Inputs",
        "",
        f"- Raters ({len(raters)}): "
        + ", ".join(f"`{rater.rater}` ({rater.path.name})" for rater in raters),
        f"- Notices in key: {len(key)}; rated by at least one rater: {rated_any}; "
        f"rated by every rater: {rated_all}",
        "- Conditions: "
        + ", ".join(f"{condition} ({count})" for condition, count in sorted(conditions.items())),
        f"- Seed {seed}; bootstrap resamples {n_boot}; permutations {n_perm}",
    ]
    if bundles:
        lines.append(
            "- Benchmark scores from: " + ", ".join(f"`{bundle.path}`" for bundle in bundles)
        )
        lines.append(
            "- Score lookup: "
            + ", ".join(f"{name}={count}" for name, count in sorted(metric_coverage.items()))
        )
        if metric_coverage.get("matched_by_single_bundle_fallback"):
            lines.append(
                "- WARNING: some notices were matched to the only supplied bundle because "
                "their bundle path and label did not match. Check that it is the run the "
                "notices were rendered from."
            )
    else:
        lines.append("- Benchmark scores: no --bundle supplied, so correlations were skipped.")
    lines.append("")

    lines.extend(["## Data quality", ""])
    per_rater: dict[str, Counter] = defaultdict(Counter)
    first_row: dict[str, dict[str, Any]] = {}
    for row in quality:
        counter = per_rater[row["rater"]]
        for name in ("n_missing", "n_unsure", "n_na", "n_invalid"):
            counter[name] += int(row[name])
        first_row.setdefault(row["rater"], row)
    lines.append(
        notices.md_table(
            [
                "rater",
                "rows read",
                "unknown ids",
                "duplicates",
                "missing",
                "unsure",
                "N/A",
                "invalid",
            ],
            [
                [
                    rater,
                    first_row[rater]["rows_read"],
                    first_row[rater]["unknown_notice_rows"],
                    first_row[rater]["duplicate_rows"],
                    counter["n_missing"],
                    counter["n_unsure"],
                    counter["n_na"],
                    counter["n_invalid"],
                ]
                for rater, counter in per_rater.items()
            ],
        )
    )
    lines.extend(
        [
            "",
            "Counts are summed over the five items. Invalid entries (values outside the "
            "scale) and N/A on items other than R4 are treated as missing; per-item counts "
            f"are in `{OUTPUT_FILES['quality']}`.",
            "",
            "## Inter-rater reliability",
            "",
        ]
    )
    if len(raters) < 2:
        lines.extend(["Only one rater was supplied, so reliability was not computed.", ""])
    else:
        alpha_lookup = {(row["variant"], row["metric"]): row for row in alpha}
        alpha_table = []
        for variant in VARIANTS:
            cells = []
            for metric in ("nominal", "ordinal", "interval"):
                row = alpha_lookup.get((variant.name, metric))
                cells.append(_fmt_ci(row, key="alpha") if row else "-")
            units_row = alpha_lookup.get((variant.name, "nominal")) or alpha_lookup.get(
                (variant.name, "ordinal")
            )
            alpha_table.append(
                [variant.description, *cells, units_row["n_units_pairable"] if units_row else 0]
            )
        lines.append(
            notices.md_table(
                ["item", "alpha nominal", "alpha ordinal", "alpha interval", "notices pairable"],
                alpha_table,
            )
        )
        lines.extend(["", "Pairwise agreement:", ""])
        lines.append(
            notices.md_table(
                ["item", "raters", "n", "% agree", "Cohen kappa", "quadratic-weighted kappa"],
                [
                    [
                        row["variant"],
                        f"{row['rater_a']} / {row['rater_b']}",
                        row["n_pairs"],
                        _fmt(
                            row["percent_agreement"] * 100.0
                            if row["percent_agreement"] is not None
                            else None,
                            1,
                        ),
                        _fmt(row["cohen_kappa"]),
                        _fmt(row["quadratic_weighted_kappa"]) if row["level"] == "ordinal" else "-",
                    ]
                    for row in pairwise
                ],
            )
        )
        lines.extend(
            [
                "",
                "Krippendorff's alpha allows missing values. Its CIs are seeded bootstrap "
                "percentiles over notices. A commonly cited convention (Krippendorff) treats "
                "alpha >= 0.800 as reliable and 0.667-0.800 as tentative. With about 48 "
                "notices the CIs are wide, so read the interval, not the point estimate. "
                "Kappa is undefined (n/a) when both raters use a single category.",
                "",
            ]
        )

    lines.extend(["## Consensus ratings by group", ""])
    lines.extend(
        [
            "Means of the per-notice consensus ratings with seeded bootstrap 95% CIs. Groups "
            "are listed alphabetically, and no ranking is implied. Null baselines are "
            "reference rows only. They are excluded from the model, dataset and "
            "all-real-explainer rows, and appear as their own family row.",
            "",
            "By explainer:",
            "",
            _group_table(groups, "explainer"),
            "",
            "By family:",
            "",
            _group_table(groups, "family"),
            "",
            "All real explainers:",
            "",
            _group_table(groups, "all_real_explainers"),
            "",
            f"Model, dataset and condition rows are in `{OUTPUT_FILES['groups']}`.",
            "",
        ]
    )

    lines.extend(["## Do benchmark metrics track the ratings?", ""])
    if not correlations:
        lines.extend(
            [
                "No benchmark scores were available (no --bundle, or no notice matched), so "
                "correlations were not computed.",
                "",
            ]
        )
    else:
        primary = [
            row for row in correlations if row["scope"] == "real_explainers" and row["prespecified"]
        ]
        lines.extend(
            [
                "Pre-specified pairs (real explainers; Spearman rho with bootstrap 95% CI; "
                "two-sided permutation p; Holm over these pairs):",
                "",
                notices.md_table(
                    ["rating", "metric", "expected", "n", "rho [95% CI]", "p (perm)", "p (Holm)"],
                    [
                        [
                            row["rating"],
                            row["metric"],
                            row["expected"],
                            row["n"],
                            _rho_text(row),
                            _fmt(row.get("p_perm"), 3),
                            _fmt(row.get("p_holm_prespecified"), 3),
                        ]
                        for row in primary
                    ],
                ),
                "",
            ]
        )
        within = [row for row in correlations if row["scope"].startswith("within_explainer:")]
        if within:
            explainers = sorted({row["scope"].split(":", 1)[1] for row in within})
            lookup = {
                (row["scope"].split(":", 1)[1], row["rating"], row["metric"]): row
                for row in within
            }
            table = []
            for rating, metric, _sign in PRESPECIFIED:
                cells = []
                for explainer in explainers:
                    row = lookup.get((explainer, rating, metric))
                    if row is None:
                        cells.append("-")
                    elif row.get("rho") is None:
                        cells.append(f"n/a (n={row['n']})")
                    else:
                        cells.append(f"{row['rho']:.2f} (n={row['n']})")
                table.append([f"{rating} ~ {metric}", *cells])
            lines.extend(
                [
                    "Within-explainer Spearman rho (pre-specified pairs; small n, descriptive):",
                    "",
                    notices.md_table(["pair", *explainers], table),
                    "",
                ]
            )
        baseline_scope = [
            row
            for row in correlations
            if row["scope"] == "including_null_baselines" and row["prespecified"]
        ]
        if baseline_scope:
            lines.extend(
                [
                    "With null-baseline notices included (known-poor explanations widen the "
                    "range; descriptive):",
                    "",
                    notices.md_table(
                        ["rating", "metric", "n", "rho [95% CI]"],
                        [
                            [row["rating"], row["metric"], row["n"], _rho_text(row)]
                            for row in baseline_scope
                        ],
                    ),
                    "",
                ]
            )
        lines.extend(
            [
                f"All rating x metric pairs, including exploratory ones, are in "
                f"`{OUTPUT_FILES['correlations']}`.",
                "",
            ]
        )

    lines.extend(
        [
            "## Caveats",
            "",
            "- With about 48 notices (about 10 per explainer), group means and correlations "
            "are imprecise. Read the CIs, and do not rank methods on point estimates.",
            "- The method family is not fully blindable, because counterfactual notices carry "
            "a recourse section. A reasons-only condition (`notices.py --strip-recourse`) "
            "checks this.",
            "- ALF is defined only for counterfactual explanations, and R4 is N/A when a "
            "notice suggests no change. R4-ALF correlations therefore rest on counterfactual "
            "notices only.",
            "- Pooled correlations mix differences between methods with differences within "
            "a method. CSS's decision_sensitivity_visibility component structurally "
            "favours counterfactual methods, so compare css with css_no_dsv and read the "
            "within-explainer rows.",
            "- The benchmark's faithfulness for SHAP/LIME uses the explainer's own values as "
            "the reference. Raters cannot observe faithfulness, so no association is "
            "expected; it is reported as a check.",
            "- Permutation p-values are unadjusted, except for the Holm column over the "
            "pre-specified pairs in the real-explainer scope. Everything else is "
            "exploratory.",
            "",
            "## Outputs",
            "",
            *[f"- `{path}`" for path in output_paths],
            "",
        ]
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# I/O wrapper
# ---------------------------------------------------------------------------


def load_score_bundle(bundle: str | Path) -> ScoreBundle:
    path = notices.resolve_results_path(bundle)
    payload = common.load_results(path)
    return ScoreBundle(path=path, label=path.parent.name, scores=index_scores(payload))


def run(
    ratings_paths: Sequence[str | Path],
    key_path: str | Path,
    out_dir: str | Path,
    *,
    bundles: Sequence[str | Path] = (),
    rater_names: Sequence[str] | None = None,
    n_boot: int = 2000,
    n_perm: int = 2000,
    seed: int = 42,
    min_n: int = 5,
) -> list[Path]:
    key_file = Path(key_path).expanduser()
    if not key_file.is_file():
        raise SystemExit(f"key file not found: {key_path}")
    key_file = key_file.resolve()
    key = load_key(key_file)
    notice_ids = list(key)

    rating_files: list[tuple[Path, list[dict[str, Any]]]] = []
    for item in ratings_paths:
        path = Path(item).expanduser()
        if not path.is_file():
            raise SystemExit(f"ratings file not found: {item}")
        path = path.resolve()
        rating_files.append((path, read_csv_rows(path)))
    if not rating_files:
        raise SystemExit("at least one --ratings file is required")
    if rater_names is not None and len(rater_names) != len(rating_files):
        raise SystemExit("--rater-names must give one name per --ratings file")
    names = resolve_rater_names(rating_files, rater_names)
    raters: list[RaterRatings] = []
    notice_set = set(notice_ids)
    for name, (path, rows) in zip(names, rating_files, strict=True):
        entries, comments, issues = parse_rater_rows(rows, notice_set)
        raters.append(
            RaterRatings(rater=name, path=path, entries=entries, comments=comments, issues=issues)
        )

    quality = data_quality_rows(raters, notice_ids)
    long = long_rows(raters, notice_ids)
    if len(raters) >= 2:
        pairwise, alpha = reliability_rows(raters, notice_ids, n_boot=n_boot, seed=seed)
    else:
        pairwise, alpha = [], []

    consensus: list[dict[str, Any]] = []
    for notice_id in notice_ids:
        key_row = key[notice_id]
        row: dict[str, Any] = {"notice_id": notice_id}
        row.update({name: key_row.get(name) for name in KEY_FIELDS})
        row["is_null_baseline"] = bool(key_row["is_null_baseline"])
        row.update(consensus_for_notice([rater.entry(notice_id) for rater in raters]))
        consensus.append(row)

    score_bundles = [load_score_bundle(bundle) for bundle in bundles]
    metric_coverage = attach_metrics(consensus, key, score_bundles)
    groups = group_rows(consensus, n_boot=n_boot, seed=seed)
    has_metrics = any(row.get(metric) is not None for row in consensus for metric in METRICS)
    correlations = (
        correlation_rows(consensus, n_boot=n_boot, n_perm=n_perm, seed=seed, min_n=min_n)
        if has_metrics
        else []
    )

    out = Path(out_dir)
    targets = {name: out / filename for name, filename in OUTPUT_FILES.items()}
    notices.assert_outputs_safe(
        targets.values(),
        [key_file, *(path for path, _rows in rating_files), *(bundle.path for bundle in score_bundles)],
    )
    notices.write_csv(targets["long"], long, LONG_FIELDS)
    notices.write_csv(targets["quality"], quality, QUALITY_FIELDS)
    notices.write_csv(targets["pairwise"], pairwise, PAIRWISE_FIELDS)
    notices.write_csv(targets["alpha"], alpha, ALPHA_FIELDS)
    notices.write_csv(
        targets["consensus"],
        consensus,
        ("notice_id", *KEY_FIELDS, "bundle_match", *CONSENSUS_FIELDS, *METRICS),
    )
    notices.write_csv(targets["groups"], groups, GROUP_FIELDS)
    notices.write_csv(targets["correlations"], correlations, CORRELATION_FIELDS)
    notices.write_text(
        targets["report"],
        render_report(
            raters=raters,
            key=key,
            consensus=consensus,
            quality=quality,
            pairwise=pairwise,
            alpha=alpha,
            groups=groups,
            correlations=correlations,
            metric_coverage=metric_coverage,
            bundles=score_bundles,
            n_boot=n_boot,
            n_perm=n_perm,
            seed=seed,
            output_paths=list(targets.values()),
        ),
    )
    return list(targets.values())


# ---------------------------------------------------------------------------
# Self-test (synthetic data only)
# ---------------------------------------------------------------------------


def selftest() -> None:
    # Krippendorff (2011), "Computing Krippendorff's Alpha-Reliability": 4 observers x
    # 12 units, values 1-5, with missing values. Published: nominal .743, ordinal .815,
    # interval .849.
    observers = [
        [1, 2, 3, 3, 2, 1, 4, 1, 2, None, None, None],
        [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, None, 3],
        [None, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, None],
        [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, None],
    ]
    units = [
        [observer[unit] for observer in observers if observer[unit] is not None]
        for unit in range(12)
    ]
    for metric, expected in (("nominal", 0.743), ("ordinal", 0.815), ("interval", 0.849)):
        value = krippendorff_alpha(units, metric)
        assert value is not None and abs(value - expected) < 1e-3, (metric, value)
    assert krippendorff_alpha([[1, 1], [1, 1]], "nominal") is None

    assert abs(cohen_kappa([(0, 0), (0, 1), (1, 1), (1, 1)]) - 0.5) < 1e-12
    assert cohen_kappa([(1, 1), (1, 1)]) is None
    assert abs(weighted_kappa([(0, 2), (1, 1), (2, 0)], (0, 1, 2)) - (-1.0)) < 1e-12
    assert abs(weighted_kappa([(0, 0), (1, 1), (2, 2)], (0, 1, 2)) - 1.0) < 1e-12

    assert abs(spearman([1, 2, 3, 4, 5], [5, 6, 7, 8, 7]) - 8.0 / math.sqrt(95.0)) < 1e-9
    assert abs(spearman([1, 2, 3, 4], [4, 3, 2, 1]) - (-1.0)) < 1e-12
    assert spearman([1, 1, 1], [1, 2, 3]) is None
    adjusted = holm_adjust([0.01, 0.04, None, 0.03])
    assert adjusted[2] is None
    assert abs(adjusted[0] - 0.03) < 1e-12
    assert abs(adjusted[1] - 0.06) < 1e-12 and abs(adjusted[3] - 0.06) < 1e-12

    r1, r2, r4 = ITEMS[0], ITEMS[1], ITEMS[3]
    assert parse_rating(r1, "2") == (2, "ok")
    assert parse_rating(r1, "2.0") == (2, "ok")
    assert parse_rating(r1, "3") == (None, "invalid")
    assert parse_rating(r1, "1.5") == (None, "invalid")
    assert parse_rating(r1, " ") == (None, "missing")
    assert parse_rating(r1, "Unsure") == (None, "unsure")
    assert parse_rating(r2, "Yes") == ("yes", "ok")
    assert parse_rating(r2, "?") == ("unsure", "ok")
    assert parse_rating(r2, "perhaps") == (None, "invalid")
    assert parse_rating(r4, "N/A") == (None, "na")
    assert parse_rating(r4, "0") == (0, "ok")

    def entry(**values: str) -> dict[str, tuple[Any, str, str]]:
        parsed: dict[str, tuple[Any, str, str]] = {}
        for item in ITEMS:
            raw = values.get(item.short, "")
            value, status = parse_rating(item, raw)
            parsed[item.column] = (value, status, raw)
        return parsed

    consensus = consensus_for_notice(
        [
            entry(R1="2", R2="yes", R3="1", R4="N/A", R5="4"),
            entry(R1="1", R2="no", R3="", R4="2", R5="5"),
        ]
    )
    assert consensus["R1_specificity"] == 1.5 and consensus["R1_specificity_spread"] == 1.0
    assert consensus["R3_rectification"] == 1.0 and consensus["R3_rectification_n"] == 1
    assert consensus["R4_status"] == "rated" and consensus["R4_recourse"] == 2.0
    assert consensus["R4_na_share"] == 0.5
    assert consensus["R2_yes_share"] == 0.5 and consensus["R2_any_yes"] == 1.0
    assert consensus["R2_majority"] == "tie"
    assert consensus["n_raters_with_ratings"] == 2
    only_na = consensus_for_notice([entry(R4="N/A"), entry(R4="N/A"), entry(R4="1")])
    assert only_na["R4_status"] == "not_applicable" and only_na["R4_recourse"] is None

    metrics = extract_metrics(
        {
            "contestability_support_score": {
                "score": 0.5,
                "verifiability": 1.0,
                "correctability": 0.5,
                "decision_sensitivity_visibility": 0.0,
                "reason_specificity": 1.0,
            },
            "actionability_legality_filter": {"score": None, "legality": None},
        }
    )
    assert abs(metrics["css_no_dsv"] - (0.30 + 0.15 + 0.15) / 0.75) < 1e-12
    assert metrics["alf"] is None and metrics["faithfulness"] is None

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        bundle_dir = root / "synthetic_bundle"
        bundle_dir.mkdir()
        common.write_json(bundle_dir / "results.json", notices.synthetic_payload())
        packet = root / "packet"
        notices.run(
            [bundle_dir], packet, n=20, seed=7, include_baselines=True, taxonomy={}
        )
        key_path = packet / notices.OUTPUT_FILES["key"]
        template = read_csv_rows(packet / notices.OUTPUT_FILES["template"])
        rng = random.Random(3)
        base: dict[str, dict[str, str]] = {}
        for row in template:
            base[row["notice_id"]] = {
                "R1_specificity": str(rng.randint(0, 2)),
                "R2_prohibited_basis": rng.choice(["yes", "no", "no", "unsure"]),
                "R3_rectification": str(rng.randint(0, 2)),
                "R4_recourse": rng.choice(["0", "1", "2", "N/A"]),
                "R5_overall": str(rng.randint(1, 5)),
            }
        rater_paths: list[Path] = []
        for index, name in enumerate(("ra", "rb")):
            rows = []
            for position, row in enumerate(template):
                values = dict(base[row["notice_id"]])
                if index == 1:
                    if rng.random() < 0.3:
                        values["R5_overall"] = str(rng.randint(1, 5))
                    if position == 0:
                        values["R1_specificity"] = "unsure"
                    if position == 1:
                        values["R3_rectification"] = "7"
                    if position == 2:
                        values["R5_overall"] = ""
                rows.append(
                    {"notice_id": row["notice_id"], "rater_id": name, **values, "comment": ""}
                )
            path = root / f"ratings_{name}.csv"
            notices.write_csv(path, rows, notices.TEMPLATE_COLUMNS)
            rater_paths.append(path)

        outputs = run(
            rater_paths,
            key_path,
            root / "analysis",
            bundles=[bundle_dir],
            n_boot=200,
            n_perm=200,
            seed=42,
        )
        assert all(path.is_file() for path in outputs)
        consensus_rows = read_csv_rows(root / "analysis" / OUTPUT_FILES["consensus"])
        assert len(consensus_rows) == 20
        assert all(row["css"] != "" and row["bundle_match"] == "path" for row in consensus_rows)
        alpha_rows = read_csv_rows(root / "analysis" / OUTPUT_FILES["alpha"])
        assert any(row["alpha"] != "" for row in alpha_rows)
        quality_rows = read_csv_rows(root / "analysis" / OUTPUT_FILES["quality"])
        rb_r3 = [
            row for row in quality_rows if row["rater"] == "rb" and row["item"] == "R3_rectification"
        ]
        assert rb_r3 and rb_r3[0]["n_invalid"] == "1"
        correlation_csv = read_csv_rows(root / "analysis" / OUTPUT_FILES["correlations"])
        assert any(row["scope"] == "real_explainers" and row["rho"] != "" for row in correlation_csv)
        groups_csv = read_csv_rows(root / "analysis" / OUTPUT_FILES["groups"])
        assert any(row["group"] == "random_reasons" and row["is_null_baseline"] == "True"
                   for row in groups_csv)

        no_bundle = run(rater_paths, key_path, root / "analysis_no_bundle", n_boot=50, n_perm=50)
        assert all(path.is_file() for path in no_bundle)
        single = run(rater_paths[:1], key_path, root / "analysis_single", n_boot=50, n_perm=50)
        assert all(path.is_file() for path in single)
    print("rating_analysis selftest: OK")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inter-rater reliability of the expert notice ratings and their Spearman "
            "correlation with the benchmark scores of the same explanations."
        )
    )
    parser.add_argument(
        "--ratings",
        nargs="+",
        default=[],
        help="One filled ratings CSV per rater (copies of ratings_template.csv).",
    )
    parser.add_argument("--key", help="notices_key.csv written by notices.py.")
    parser.add_argument(
        "--bundle",
        action="append",
        default=[],
        help="Run directory (or results.json) the notices came from. Repeatable; optional.",
    )
    parser.add_argument("--out-dir", help="Directory for CSV and markdown outputs.")
    parser.add_argument(
        "--rater-names",
        help="Comma-separated rater names, one per --ratings file (default: rater_id or file stem).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for bootstrap/permutation.")
    parser.add_argument("--n-boot", type=int, default=2000, help="Bootstrap resamples.")
    parser.add_argument("--n-perm", type=int, default=2000, help="Permutations for p-values.")
    parser.add_argument(
        "--min-n",
        type=int,
        default=5,
        help="Minimum paired notices for a correlation (default 5; never below 3).",
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="Run the built-in self-test on synthetic data and exit.",
    )
    args = parser.parse_args(argv)
    if args.selftest:
        selftest()
        return
    if not args.ratings:
        parser.error("--ratings is required")
    if not args.key:
        parser.error("--key is required")
    if not args.out_dir:
        parser.error("--out-dir is required")
    if args.n_boot < 0 or args.n_perm < 0:
        parser.error("--n-boot and --n-perm must be >= 0")
    outputs = run(
        args.ratings,
        args.key,
        args.out_dir,
        bundles=args.bundle,
        rater_names=notices.split_list(args.rater_names),
        n_boot=args.n_boot,
        n_perm=args.n_perm,
        seed=args.seed,
        min_n=args.min_n,
    )
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
