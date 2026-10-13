"""Render blinded, adverse-action-style notices for an expert-rating exercise.

Purpose
-------
Two or three raters with legal training score about 48 rendered statements of
reasons with the items in ``rating_rubric.md``. ``rating_analysis.py`` then
tests whether the benchmark's metrics track those legal judgments of the same
explanations.

What is rendered (per sampled explanation, from instance["canonical_explanation"])
--------------------------------------------------------------------------------
* a neutral header, "Statement of specific reasons for credit denial
  (research rendering)", which deliberately does not imitate any lender or
  regulator form;
* an opaque notice id and a generic dataset context line;
* up to four principal reasons in rank order, using the canonical
  ``reason_text`` verbatim (falling back to the taxonomy ``plain_meaning`` of
  the supporting features when the text is empty);
* for counterfactual explanations, a section "What could change the outcome"
  listing the proposed changes (feature plain meaning: from -> to), unless
  ``--strip-recourse``;
* "Applicant record (fields cited above)": the applicant's values for every
  field cited by the rendered reasons (and proposed changes), taken from
  ``referenced_features[*].raw_value`` (counterfactual ``from_value`` as a
  fallback).

Blinding
--------
No explainer, model or track names, no scores and no instance ids appear in a
notice. Notice ids are random five-digit codes, and the presentation order is
shuffled. The method FAMILY is not fully blindable: counterfactual notices
carry the extra recourse section. ``--strip-recourse`` renders a reasons-only
condition. It drops that section and also the canonicalizer's
counterfactual-derived reasons ("changing X may improve the outcome").

Sampling (random.Random seeded from --seed via crc32; deterministic)
--------------------------------------------------------------------
A greedy least-filled allocation picks one notice at a time:

1. the explainer with the fewest notices so far;
2. then the dataset with the fewest notices for that explainer (then overall);
3. then the model with the fewest notices for that explainer (then for that
   dataset, then overall);
4. then the (bundle, track) source with the fewest notices for that explainer
   (then overall).

Ties are broken by a seeded random priority. Within the chosen cell one
explanation is drawn at random. Applicants (dataset, eval_row_index) not
already used are preferred, so raters rarely see the same applicant record
twice. With 48 notices and five explainers this gives 10/10/10/9/9 notices per
explainer and near-equal dataset and model counts.

Null baselines (common.NULL_BASELINES) are excluded unless
``--include-baselines``. By default only credit datasets are rendered (heloc,
south_german_credit, default_of_credit_card_clients, give_me_some_credit,
hmda). Pass ``--datasets`` to include others; they get a neutral "adverse
decision" header.

Rendering needs a bundle produced with
``cslaw.persist_canonical_explanations=true``. Older bundles store scores only,
and for them the script exits with a clear message.

v2 bundles (cslaw corrections): DiCE/Wachter reasons are the features their
counterfactuals change (``cf_reason_source`` in the canonical metadata; a
search without a valid change yields a notice with no stated reasons, which is
kept because the benchmark scored it), and temporal reasons use the corrected
grouping. Plain-language fallbacks come from the taxonomy file each bundle
recorded in its provenance (v1 bundles: benchmark/metadata/feature_taxonomy.json).
The key records ``corrections_version`` and ``cf_reason_source`` per notice;
do not mix v1 and v2 bundles in one rating packet (the design file warns).

Usage
-----
    python benchmark/cslaw/notices.py --bundle <run dir> [--bundle <run dir>] \
        --out-dir <dir> [--n 48] [--seed 7] [--strip-recourse] [--include-baselines]
    python benchmark/cslaw/notices.py --selftest
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import re
import sys
import tempfile
import zlib
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]

HEADER_CREDIT = "Statement of specific reasons for credit denial (research rendering)"
HEADER_OTHER = "Statement of specific reasons for an adverse decision (research rendering)"
OUTCOME_CREDIT = "Outcome: the application was not approved."
OUTCOME_OTHER = "Outcome: adverse decision."
FOOTER = (
    "Research rendering prepared for expert rating. It is not a notice from any "
    "creditor and was not sent to any person."
)
REASONS_HEADING = "Principal reasons for the decision:"
RECOURSE_HEADING = "What could change the outcome:"
RECORD_HEADING = "Applicant record (fields cited above):"

# Generic context lines; deliberately no product names, lenders or jurisdictions.
CREDIT_CONTEXTS: dict[str, str] = {
    "heloc": "consumer credit line application (home equity line of credit)",
    "south_german_credit": "consumer installment loan application",
    "default_of_credit_card_clients": "credit card account credit decision",
    "give_me_some_credit": "consumer credit application",
    "hmda": "home mortgage loan application",
}
OTHER_CONTEXTS: dict[str, str] = {
    "ny_courts_pretrial": "pretrial release decision (non-credit domain)",
    "fda_inspections": "facility inspection classification (non-credit domain)",
}
GENERIC_CONTEXT = "consumer credit application"

MAX_REASONS = 4
NOTICE_ID_RANGE = (10000, 100000)

# Shared with rating_analysis.py: the rubric columns of ratings_template.csv.
RATING_ITEM_COLUMNS = (
    "R1_specificity",
    "R2_prohibited_basis",
    "R3_rectification",
    "R4_recourse",
    "R5_overall",
)
TEMPLATE_COLUMNS = ("notice_id", "rater_id", *RATING_ITEM_COLUMNS, "comment")
KEY_COLUMNS = (
    "notice_id",
    "presentation_order",
    "condition",
    "bundle",
    "bundle_label",
    "dataset",
    "track",
    "model",
    "explainer",
    "family",
    "is_null_baseline",
    "instance_id",
    "eval_row_index",
    "n_reasons_rendered",
    "n_reasons_dropped",
    "recourse_section_shown",
    "n_changes_shown",
    "n_record_fields",
    "corrections_version",
    "cf_reason_source",
)
OUTPUT_FILES = {
    "notices_csv": "notices_for_rating.csv",
    "notices_md": "notices_for_rating.md",
    "key": "notices_key.csv",
    "template": "ratings_template.csv",
    "design": "notices_design.md",
}

BLINDING_TERMS = (
    "anchors",
    "dice",
    "lime",
    "shap",
    "wachter",
    "leave_one_out",
    "random_reasons",
    "global_ranking",
    "logistic_regression",
    "logistic",
    "ebm",
    "xgboost",
    "primary",
    "audit",
)
_BLINDING_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(term) for term in BLINDING_TERMS) + r")\b",
    re.IGNORECASE,
)


@dataclass(slots=True)
class Candidate:
    bundle: str
    bundle_label: str
    dataset_id: str
    track: str
    model_id: str
    explainer_id: str
    instance_id: str
    eval_row_index: int | None
    canonical: dict[str, Any]
    corrections_version: str = "v1"

    @property
    def cf_reason_source(self) -> str | None:
        metadata = self.canonical.get("metadata")
        value = metadata.get("cf_reason_source") if isinstance(metadata, dict) else None
        return str(value) if value is not None else None

    @property
    def family(self) -> str:
        family = self.canonical.get("explanation_family")
        if family:
            return str(family)
        return common.EXPLAINER_FAMILY.get(self.explainer_id, "unknown")

    @property
    def applicant_key(self) -> tuple[str, Any]:
        index = self.eval_row_index if self.eval_row_index is not None else self.instance_id
        return (self.dataset_id, index)

    @property
    def cell(self) -> tuple[str, str, str, str, str]:
        return (self.explainer_id, self.dataset_id, self.model_id, self.bundle, self.track)


@dataclass(slots=True)
class RenderedNotice:
    notice_id: str
    context: str
    text: str
    n_reasons: int
    n_reasons_dropped: int
    recourse_shown: bool
    n_changes: int
    n_record_fields: int


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def derive_seed(seed: int, *parts: Any) -> int:
    text = "|".join([str(seed), *(str(part) for part in parts)])
    return zlib.crc32(text.encode("utf-8"))


def resolve_results_path(bundle: str | Path) -> Path:
    path = Path(bundle).expanduser()
    if path.is_dir():
        path = path / "results.json"
    if not path.is_file():
        raise SystemExit(f"results.json not found for bundle: {bundle}")
    return path.resolve()


def split_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def sentence_case(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def format_value(value: Any) -> str:
    """Render a raw feature value for a human reader."""
    if value is None:
        return "not recorded"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}" if abs(value) >= 10000 else str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return "not recorded"
        if value.is_integer():
            return format_value(int(value))
        magnitude = abs(value)
        if magnitude >= 1000:
            return f"{value:,.0f}"
        if magnitude >= 1:
            return f"{value:.2f}".rstrip("0").rstrip(".")
        return f"{value:.3g}"
    text = str(value).strip()
    return text if text else "not recorded"


def blinding_violations(text: str) -> list[str]:
    """Method, model or track identifiers that leaked into a notice text."""
    return sorted({match.group(1).lower() for match in _BLINDING_PATTERN.finditer(text)})


def load_taxonomy_safe(repo_root: str | Path) -> dict[str, dict[str, dict[str, Any]]]:
    try:
        return common.load_taxonomy(repo_root)
    except (OSError, ValueError) as exc:
        print(
            f"warning: feature taxonomy not loaded ({exc}); plain-language labels fall "
            "back to the canonical plain_meaning or the raw feature name",
            file=sys.stderr,
        )
        return {}


def bundle_taxonomy(
    payload: dict[str, Any],
    repo_root: str | Path,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Taxonomy the bundle's run used (v2 provenance), else the v1 file."""
    path, origin, warning = common.resolve_metadata_file(payload, "feature_taxonomy", repo_root)
    if warning:
        print(f"warning: {warning}", file=sys.stderr)
    if origin == "v1_default":
        return load_taxonomy_safe(repo_root)
    try:
        return common.load_taxonomy_file(path)
    except (OSError, ValueError) as exc:
        print(
            f"warning: feature taxonomy {path} not loaded ({exc}); plain-language labels "
            "fall back to the canonical plain_meaning or the raw feature name",
            file=sys.stderr,
        )
        return {}


def resolve_explainers(
    explainers: Sequence[str] | None,
    include_baselines: bool,
) -> list[str]:
    requested = list(explainers) if explainers else list(common.REAL_EXPLAINERS)
    if include_baselines and not explainers:
        requested.extend(common.NULL_BASELINES)
    if not include_baselines:
        requested = [item for item in requested if not common.is_null_baseline(item)]
    ordered: list[str] = []
    for item in requested:
        if item not in ordered:
            ordered.append(item)
    return ordered


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return "" if not math.isfinite(value) else repr(value)
    return value


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _csv_value(row.get(column)) for column in fieldnames})
    return path


def write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def assert_outputs_safe(outputs: Iterable[Path], inputs: Iterable[Path]) -> None:
    protected = {Path(path).resolve() for path in inputs}
    for output in outputs:
        if Path(output).resolve() in protected:
            raise SystemExit(f"refusing to overwrite an input file: {output}")


def md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(str(header) for header in headers) + " |",
        "| " + " | ".join("---" for _header in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Candidate collection and sampling (pure)
# ---------------------------------------------------------------------------


def collect_candidates(
    payload: dict[str, Any],
    *,
    bundle: str,
    label: str,
    explainers: Sequence[str],
    datasets: Sequence[str] | None,
    tracks: Sequence[str] | None,
) -> tuple[list[Candidate], Counter]:
    """Explanations eligible for rendering, plus coverage counts."""
    wanted_explainers = set(explainers)
    wanted_datasets = set(datasets) if datasets is not None else None
    wanted_tracks = set(tracks) if tracks is not None else None
    version = common.corrections_version(payload)
    coverage: Counter = Counter()
    candidates: list[Candidate] = []
    for row, instance in common.iter_instances(payload):
        coverage["instances"] += 1
        explainer_id = str(row.get("explainer_id", ""))
        dataset_id = str(row.get("dataset_id", ""))
        track = str(row.get("track", ""))
        if explainer_id not in wanted_explainers:
            coverage["excluded_explainer"] += 1
            continue
        if wanted_datasets is not None and dataset_id not in wanted_datasets:
            coverage["excluded_dataset"] += 1
            continue
        if wanted_tracks is not None and track not in wanted_tracks:
            coverage["excluded_track"] += 1
            continue
        canonical = instance.get("canonical_explanation")
        if not isinstance(canonical, dict):
            coverage["missing_canonical_explanation"] += 1
            continue
        coverage["eligible"] += 1
        candidates.append(
            Candidate(
                bundle=bundle,
                bundle_label=label,
                dataset_id=dataset_id,
                track=track,
                model_id=str(row.get("model_id", "")),
                explainer_id=explainer_id,
                instance_id=str(instance.get("instance_id", "")),
                eval_row_index=common.eval_row_index(instance),
                canonical=canonical,
                corrections_version=version,
            )
        )
    return candidates, coverage


def _candidate_sort_key(candidate: Candidate) -> tuple[Any, ...]:
    index = candidate.eval_row_index if candidate.eval_row_index is not None else -1
    return (
        candidate.explainer_id,
        candidate.dataset_id,
        candidate.model_id,
        candidate.bundle,
        candidate.track,
        index,
        candidate.instance_id,
    )


def select_notices(
    candidates: Sequence[Candidate],
    n: int,
    seed: int,
) -> tuple[list[Candidate], list[str]]:
    """Near-balanced greedy allocation (see module docstring); deterministic."""
    rng = random.Random(derive_seed(seed, "sample"))
    pools: dict[tuple[str, str, str, str, str], list[Candidate]] = defaultdict(list)
    for candidate in sorted(candidates, key=_candidate_sort_key):
        pools[candidate.cell].append(candidate)

    def priorities(values: Iterable[Any]) -> dict[Any, int]:
        ordered = sorted(set(values), key=str)
        rng.shuffle(ordered)
        return {value: index for index, value in enumerate(ordered)}

    cells = sorted(pools)
    prio_explainer = priorities(cell[0] for cell in cells)
    prio_dataset = priorities(cell[1] for cell in cells)
    prio_model = priorities(cell[2] for cell in cells)
    prio_source = priorities((cell[3], cell[4]) for cell in cells)

    n_explainer: Counter = Counter()
    n_dataset: Counter = Counter()
    n_model: Counter = Counter()
    n_source: Counter = Counter()
    n_explainer_dataset: Counter = Counter()
    n_explainer_model: Counter = Counter()
    n_dataset_model: Counter = Counter()
    n_explainer_source: Counter = Counter()
    used_applicants: set[tuple[str, Any]] = set()
    selected: list[Candidate] = []

    for _slot in range(n):
        available = [cell for cell in cells if pools[cell]]
        if not available:
            break
        explainer = min(
            {cell[0] for cell in available},
            key=lambda value: (n_explainer[value], prio_explainer[value]),
        )
        dataset = min(
            {cell[1] for cell in available if cell[0] == explainer},
            key=lambda value: (
                n_explainer_dataset[(explainer, value)],
                n_dataset[value],
                prio_dataset[value],
            ),
        )
        model = min(
            {cell[2] for cell in available if cell[0] == explainer and cell[1] == dataset},
            key=lambda value: (
                n_explainer_model[(explainer, value)],
                n_dataset_model[(dataset, value)],
                n_model[value],
                prio_model[value],
            ),
        )
        source = min(
            {(cell[3], cell[4]) for cell in available if cell[:3] == (explainer, dataset, model)},
            key=lambda value: (
                n_explainer_source[(explainer, value)],
                n_source[value],
                prio_source[value],
            ),
        )
        pool = pools[(explainer, dataset, model, source[0], source[1])]
        fresh = [
            index
            for index, candidate in enumerate(pool)
            if candidate.applicant_key not in used_applicants
        ]
        index = rng.choice(fresh) if fresh else rng.randrange(len(pool))
        chosen = pool.pop(index)
        selected.append(chosen)
        used_applicants.add(chosen.applicant_key)
        n_explainer[explainer] += 1
        n_dataset[dataset] += 1
        n_model[model] += 1
        n_source[source] += 1
        n_explainer_dataset[(explainer, dataset)] += 1
        n_explainer_model[(explainer, model)] += 1
        n_dataset_model[(dataset, model)] += 1
        n_explainer_source[(explainer, source)] += 1

    notes: list[str] = []
    if len(selected) < n:
        notes.append(
            f"only {len(selected)} eligible explanations were available; {n} were requested"
        )
    if prio_explainer:
        floor_quota = n // len(prio_explainer)
        for explainer in sorted(prio_explainer):
            if n_explainer[explainer] < floor_quota:
                notes.append(
                    f"explainer {explainer}: {n_explainer[explainer]} notices "
                    f"(below the balanced quota of {floor_quota}; pool exhausted)"
                )
    reused = len(selected) - len({candidate.applicant_key for candidate in selected})
    if reused:
        notes.append(
            f"{reused} notice(s) reuse an applicant already shown in another notice "
            "(the eligible pools were too small to avoid it)"
        )
    return selected, notes


def assign_notice_ids(
    selected: Sequence[Candidate],
    seed: int,
) -> list[tuple[str, Candidate]]:
    """Shuffle the presentation order and attach opaque five-digit notice ids."""
    order = list(selected)
    random.Random(derive_seed(seed, "order")).shuffle(order)
    low, high = NOTICE_ID_RANGE
    if len(order) > high - low:
        raise SystemExit(f"too many notices for five-digit ids: {len(order)}")
    codes = random.Random(derive_seed(seed, "ids")).sample(range(low, high), len(order))
    return [(f"N{code}", candidate) for code, candidate in zip(codes, order, strict=True)]


# ---------------------------------------------------------------------------
# Rendering (pure)
# ---------------------------------------------------------------------------


def _reason_rank(reason: dict[str, Any]) -> int:
    rank = reason.get("rank")
    return rank if isinstance(rank, int) else 10**6


def dataset_context(dataset_id: str) -> tuple[str, bool]:
    """Return (context line, is_credit_domain)."""
    if dataset_id in OTHER_CONTEXTS:
        return OTHER_CONTEXTS[dataset_id], False
    return CREDIT_CONTEXTS.get(dataset_id, GENERIC_CONTEXT), True


def render_notice(
    notice_id: str,
    candidate: Candidate,
    *,
    taxonomy: dict[str, dict[str, dict[str, Any]]],
    strip_recourse: bool,
) -> RenderedNotice:
    canonical = candidate.canonical
    dataset_taxonomy = taxonomy.get(candidate.dataset_id, {})
    meanings: dict[str, str] = {}
    raw_values: dict[str, Any] = {}
    for feature in canonical.get("referenced_features") or []:
        name = feature.get("feature_name")
        if not name:
            continue
        if feature.get("plain_meaning"):
            meanings.setdefault(str(name), str(feature["plain_meaning"]))
        raw_values.setdefault(str(name), feature.get("raw_value"))

    def label(name: str) -> str:
        meaning = (
            meanings.get(name)
            or (dataset_taxonomy.get(name) or {}).get("plain_meaning")
            or name
        )
        return sentence_case(str(meaning).replace("_", " ").strip())

    cited: list[str] = []
    reason_lines: list[str] = []
    dropped = 0
    for reason in sorted(canonical.get("principal_reasons") or [], key=_reason_rank):
        if len(reason_lines) >= MAX_REASONS:
            break
        if strip_recourse and reason.get("grouped_reason_type") == "counterfactual":
            dropped += 1
            continue
        supporting = [str(name) for name in reason.get("supporting_features") or [] if name]
        text = str(reason.get("reason_text") or "").replace("_", " ").strip()
        if not text:
            text = "; ".join(label(name) for name in supporting)
        if not text:
            continue
        reason_lines.append(sentence_case(text))
        for name in supporting:
            if name not in cited:
                cited.append(name)

    changes = canonical.get("counterfactual_changes") or []
    is_counterfactual = candidate.family == "counterfactual" or bool(changes)
    show_recourse = is_counterfactual and not strip_recourse
    change_lines: list[str] = []
    from_values: dict[str, Any] = {}
    if show_recourse:
        for change in changes:
            name = change.get("feature_name")
            if not name:
                continue
            name = str(name)
            from_values.setdefault(name, change.get("from_value"))
            change_lines.append(
                f"{label(name)}: from {format_value(change.get('from_value'))} "
                f"to {format_value(change.get('to_value'))}"
            )
            if name not in cited:
                cited.append(name)

    record_lines: list[str] = []
    for name in cited:
        value = raw_values.get(name)
        if value is None:
            value = from_values.get(name)
        record_lines.append(f"{label(name)}: {format_value(value)}")

    context, is_credit = dataset_context(candidate.dataset_id)
    lines = [
        HEADER_CREDIT if is_credit else HEADER_OTHER,
        f"Notice ID: {notice_id}",
        f"Context: {context}",
        OUTCOME_CREDIT if is_credit else OUTCOME_OTHER,
        "",
        REASONS_HEADING,
    ]
    lines.extend(
        [f"  {rank}. {text}" for rank, text in enumerate(reason_lines, start=1)]
        or ["  No specific reasons were stated."]
    )
    if show_recourse:
        lines.extend(["", RECOURSE_HEADING])
        lines.extend([f"  - {text}" for text in change_lines] or ["  No change was identified."])
    lines.extend(["", RECORD_HEADING])
    lines.extend([f"  - {text}" for text in record_lines] or ["  No fields were cited."])
    lines.extend(["", FOOTER])
    return RenderedNotice(
        notice_id=notice_id,
        context=context,
        text="\n".join(lines),
        n_reasons=len(reason_lines),
        n_reasons_dropped=dropped,
        recourse_shown=show_recourse,
        n_changes=len(change_lines),
        n_record_fields=len(record_lines),
    )


def build_key_rows(
    assigned: Sequence[tuple[str, Candidate]],
    rendered: Sequence[RenderedNotice],
    *,
    condition: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for order, ((notice_id, candidate), notice) in enumerate(
        zip(assigned, rendered, strict=True), start=1
    ):
        rows.append(
            {
                "notice_id": notice_id,
                "presentation_order": order,
                "condition": condition,
                "bundle": candidate.bundle,
                "bundle_label": candidate.bundle_label,
                "dataset": candidate.dataset_id,
                "track": candidate.track,
                "model": candidate.model_id,
                "explainer": candidate.explainer_id,
                "family": candidate.family,
                "is_null_baseline": common.is_null_baseline(candidate.explainer_id),
                "instance_id": candidate.instance_id,
                "eval_row_index": candidate.eval_row_index,
                "n_reasons_rendered": notice.n_reasons,
                "n_reasons_dropped": notice.n_reasons_dropped,
                "recourse_section_shown": notice.recourse_shown,
                "n_changes_shown": notice.n_changes,
                "n_record_fields": notice.n_record_fields,
                "corrections_version": candidate.corrections_version,
                "cf_reason_source": candidate.cf_reason_source,
            }
        )
    return rows


def render_rating_markdown(
    rendered: Sequence[RenderedNotice],
    *,
    seed: int,
    condition: str,
) -> str:
    condition_text = (
        "reasons only (recourse sections removed)"
        if condition == "reasons_only"
        else "full (recourse sections shown where the explanation proposes changes)"
    )
    lines = [
        "# Statements of reasons for expert rating",
        "",
        "Rating instructions are in `rating_rubric.md`. Record your ratings in your own copy "
        "of `ratings_template.csv`, one row per notice ID. Rate each notice on its own, and "
        "do not try to work out which method or model produced it.",
        "",
        f"> Coordinator note (you may remove it before distribution): {len(rendered)} notices, "
        f"seed {seed}, condition: {condition_text}. Notices derived from counterfactual "
        f'methods can be recognized by their extra section "{RECOURSE_HEADING.rstrip(":")}", '
        "so the method family is not fully blindable. For a reasons-only condition, re-run "
        "`notices.py` with `--strip-recourse`; this also drops reasons phrased as "
        '"changing X may improve the outcome".',
        "",
    ]
    for notice in rendered:
        lines.extend(
            [
                "---",
                "",
                f"## Notice {notice.notice_id}",
                "",
                "```text",
                notice.text,
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def _count_table(
    selected: Sequence[Candidate],
    row_attr: str,
    col_attr: str,
) -> str:
    counts = Counter((getattr(item, row_attr), getattr(item, col_attr)) for item in selected)
    row_values = sorted({key[0] for key in counts}, key=str)
    col_values = sorted({key[1] for key in counts}, key=str)
    rows = [
        [row_value, *[counts.get((row_value, col_value), 0) for col_value in col_values],
         sum(counts.get((row_value, col_value), 0) for col_value in col_values)]
        for row_value in row_values
    ]
    rows.append(
        ["total", *[sum(counts.get((row_value, col_value), 0) for row_value in row_values)
                    for col_value in col_values], len(selected)]
    )
    return md_table([row_attr, *col_values, "total"], rows)


def render_design_markdown(
    *,
    selected: Sequence[Candidate],
    rendered: Sequence[RenderedNotice],
    coverage_by_bundle: dict[str, Counter],
    notes: Sequence[str],
    seed: int,
    n_requested: int,
    condition: str,
    explainers: Sequence[str],
    datasets: Sequence[str],
    tracks: Sequence[str] | None,
    output_paths: Sequence[Path],
    bundle_settings: dict[str, str] | None = None,
) -> str:
    lines = [
        "# Notice sample design (coordinator only; do not distribute to raters)",
        "",
        f"- Seed: {seed}; notices requested: {n_requested}; rendered: {len(rendered)}",
        f"- Condition: `{condition}`",
        f"- Explainers eligible: {', '.join(explainers) or 'none'}",
        f"- Datasets eligible: {', '.join(datasets)}",
        f"- Tracks eligible: {', '.join(tracks) if tracks else 'all present'}",
        "",
        "## Eligible explanations per bundle",
        "",
    ]
    coverage_rows = []
    for bundle, coverage in coverage_by_bundle.items():
        coverage_rows.append(
            [
                bundle,
                coverage.get("instances", 0),
                coverage.get("eligible", 0),
                coverage.get("missing_canonical_explanation", 0),
                coverage.get("excluded_explainer", 0),
                coverage.get("excluded_dataset", 0),
                coverage.get("excluded_track", 0),
            ]
        )
    lines.append(
        md_table(
            [
                "bundle",
                "instances",
                "eligible",
                "no canonical_explanation",
                "excluded explainer",
                "excluded dataset",
                "excluded track",
            ],
            coverage_rows,
        )
    )
    if bundle_settings:
        lines.extend(["", "## Corrections per bundle", ""])
        lines.extend(f"- `{bundle}`: {text}" for bundle, text in bundle_settings.items())
        versions = Counter(candidate.corrections_version for candidate in selected)
        lines.append(
            "- Selected notices by corrections version: "
            + ", ".join(f"{version} {count}" for version, count in sorted(versions.items()))
        )
        sources = Counter(
            candidate.cf_reason_source or "not recorded"
            for candidate in selected
            if candidate.family == "counterfactual"
        )
        if sources:
            lines.append(
                "- Counterfactual notices by reason source: "
                + ", ".join(f"{source} {count}" for source, count in sorted(sources.items()))
            )
        if len(versions) > 1:
            lines.append(
                "- WARNING: the packet mixes corrections versions. v1 and v2 notices differ in "
                "how DiCE/Wachter reasons and temporal reasons are built; rate them as "
                "separate packets."
            )
    lines.extend(["", "## Allocation", ""])
    lines.extend([_count_table(selected, "explainer_id", "dataset_id"), ""])
    lines.extend([_count_table(selected, "explainer_id", "model_id"), ""])
    lines.extend([_count_table(selected, "explainer_id", "track"), ""])
    lines.extend([_count_table(selected, "family", "bundle_label"), ""])
    recourse = sum(1 for notice in rendered if notice.recourse_shown)
    no_reason = sum(1 for notice in rendered if notice.n_reasons == 0)
    lines.extend(
        [
            "## Blinding",
            "",
            "- Notices show no explainer, model or track names, no scores and no instance ids. "
            "Notice ids are random and the presentation order is shuffled.",
            f"- {recourse} notice(s) show a recourse section, which identifies the "
            "counterfactual family. Use `--strip-recourse` for a reasons-only condition.",
            f"- {no_reason} notice(s) state no reasons at all; they are kept on purpose, "
            "because they are part of what the benchmark scored.",
            "- `notices_key.csv` maps notice ids to methods. Keep it from raters until all "
            "ratings are final.",
            "",
        ]
    )
    if notes:
        lines.extend(["## Notes", "", *[f"- {note}" for note in notes], ""])
    lines.extend(["## Outputs", "", *[f"- `{path}`" for path in output_paths], ""])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# I/O wrapper
# ---------------------------------------------------------------------------


def run(
    bundles: Sequence[str | Path],
    out_dir: str | Path,
    *,
    n: int = 48,
    seed: int = 7,
    strip_recourse: bool = False,
    include_baselines: bool = False,
    explainers: Sequence[str] | None = None,
    datasets: Sequence[str] | None = None,
    tracks: Sequence[str] | None = None,
    repo_root: str | Path = REPO_ROOT,
    force: bool = False,
    taxonomy: dict[str, dict[str, dict[str, Any]]] | None = None,
) -> list[Path]:
    if n < 1:
        raise SystemExit("--n must be >= 1")
    if not bundles:
        raise SystemExit("at least one --bundle is required")
    wanted_explainers = resolve_explainers(explainers, include_baselines)
    if not wanted_explainers:
        raise SystemExit("no explainers left to sample (null baselines need --include-baselines)")
    wanted_datasets = list(datasets) if datasets else sorted(CREDIT_CONTEXTS)
    condition = "reasons_only" if strip_recourse else "full"

    results_paths: list[Path] = []
    candidates: list[Candidate] = []
    coverage_by_bundle: dict[str, Counter] = {}
    bundle_settings: dict[str, str] = {}
    taxonomies: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for bundle in bundles:
        results_path = resolve_results_path(bundle)
        payload = common.load_results(results_path)
        bundle_settings[str(results_path)] = common.corrections_summary_line(payload)
        if taxonomy is None:
            taxonomies[str(results_path)] = bundle_taxonomy(payload, repo_root)
        found, coverage = collect_candidates(
            payload,
            bundle=str(results_path),
            label=results_path.parent.name,
            explainers=wanted_explainers,
            datasets=wanted_datasets,
            tracks=tracks,
        )
        del payload
        candidates.extend(found)
        coverage_by_bundle[str(results_path)] = coverage
        results_paths.append(results_path)

    if not candidates:
        details = "; ".join(
            f"{bundle}: " + ", ".join(f"{key}={value}" for key, value in sorted(coverage.items()))
            for bundle, coverage in coverage_by_bundle.items()
        )
        raise SystemExit(
            "notices: no renderable explanations found "
            f"({details or 'no instances'}). Rendering needs a bundle produced with "
            "cslaw.persist_canonical_explanations=true; older bundles store scores only. "
            "Also check --datasets/--explainers/--tracks (the defaults are the credit "
            "datasets and the five real explainers)."
        )

    selected, notes = select_notices(candidates, n, seed)
    present = {candidate.explainer_id for candidate in candidates}
    for explainer in wanted_explainers:
        if explainer not in present:
            notes.append(f"explainer {explainer}: no eligible explanations in the bundle(s)")
    assigned = assign_notice_ids(selected, seed)
    rendered = [
        render_notice(
            notice_id,
            candidate,
            taxonomy=taxonomy if taxonomy is not None else taxonomies.get(candidate.bundle, {}),
            strip_recourse=strip_recourse,
        )
        for notice_id, candidate in assigned
    ]
    for notice in rendered:
        leaked = blinding_violations(notice.text)
        if leaked:
            message = (
                f"blinding check: notice {notice.notice_id} contains "
                f"{', '.join(leaked)}; review it before distribution"
            )
            print(f"warning: {message}", file=sys.stderr)
            notes.append(message)

    out = Path(out_dir)
    targets = {name: out / filename for name, filename in OUTPUT_FILES.items()}
    assert_outputs_safe(targets.values(), results_paths)
    if targets["key"].exists() and not force:
        raise SystemExit(
            f"{targets['key']} already exists. Pass --force to overwrite it; this replaces "
            "the notice-id mapping, and ratings collected against the old key would no "
            "longer match."
        )

    write_csv(
        targets["notices_csv"],
        [
            {"notice_id": notice.notice_id, "context": notice.context, "notice_text": notice.text}
            for notice in rendered
        ],
        ("notice_id", "context", "notice_text"),
    )
    write_text(
        targets["notices_md"],
        render_rating_markdown(rendered, seed=seed, condition=condition),
    )
    write_csv(
        targets["key"],
        build_key_rows(assigned, rendered, condition=condition),
        KEY_COLUMNS,
    )
    write_csv(
        targets["template"],
        [{"notice_id": notice.notice_id} for notice in rendered],
        TEMPLATE_COLUMNS,
    )
    write_text(
        targets["design"],
        render_design_markdown(
            selected=[candidate for _notice_id, candidate in assigned],
            rendered=rendered,
            coverage_by_bundle=coverage_by_bundle,
            notes=notes,
            seed=seed,
            n_requested=n,
            condition=condition,
            explainers=wanted_explainers,
            datasets=wanted_datasets,
            tracks=tracks,
            output_paths=list(targets.values()),
            bundle_settings=bundle_settings,
        ),
    )
    for note in notes:
        print(f"note: {note}", file=sys.stderr)
    return list(targets.values())


# ---------------------------------------------------------------------------
# Synthetic fixture and self-test (no real data; also used by rating_analysis.py)
# ---------------------------------------------------------------------------

_SYNTHETIC_FEATURES: dict[str, tuple[str, ...]] = {
    "heloc": (
        "NumInqLast6M",
        "NetFractionRevolvingBurden",
        "MaxDelqEver",
        "PercentTradesNeverDelq",
        "AverageMInFile",
    ),
    "south_german_credit": ("status", "duration", "amount", "savings", "age"),
}


def _synthetic_scores(rng: random.Random, family: str) -> dict[str, Any]:
    def draw() -> float:
        return round(rng.uniform(0.1, 0.95), 4)

    counterfactual = family == "counterfactual"
    verifiability, correctability, sensitivity, specificity = draw(), draw(), draw(), draw()
    return {
        "faithfulness_score": {"evidence_source": "synthetic", "score": draw()},
        "stability_score": {"evidence_source": "synthetic", "score": draw()},
        "comprehensibility_score": {"score": draw()},
        "contestability_support_score": {
            "verifiability": verifiability,
            "correctability": correctability,
            "decision_sensitivity_visibility": sensitivity,
            "reason_specificity": specificity,
            "score": 0.30 * verifiability
            + 0.30 * correctability
            + 0.25 * sensitivity
            + 0.15 * specificity,
        },
        "actionability_legality_filter": {
            "applicability": "applicable" if counterfactual else "not_applicable",
            "feasibility": draw() if counterfactual else None,
            "legality": draw() if counterfactual else None,
            "minimality": draw() if counterfactual else None,
            "invalid_recourse_rate": 0.0 if counterfactual else None,
            "hard_failures": [],
            "score": draw() if counterfactual else None,
        },
        "unified_actionability_score": {"score": draw()},
    }


def _synthetic_canonical(
    rng: random.Random,
    *,
    dataset_id: str,
    track: str,
    model_id: str,
    explainer_id: str,
    instance_id: str,
) -> dict[str, Any]:
    family = common.EXPLAINER_FAMILY.get(explainer_id, "attribution")
    chosen = rng.sample(list(_SYNTHETIC_FEATURES[dataset_id]), 4)
    referenced: list[dict[str, Any]] = []
    for rank, name in enumerate(chosen, start=1):
        weight = rng.uniform(-1.0, 1.0) if family == "attribution" else rng.uniform(0.1, 1.0)
        referenced.append(
            {
                "feature_name": name,
                "plain_meaning": name.lower(),
                "signed_direction": "increases_denial" if weight > 0 else "decreases_denial",
                "importance_weight": weight,
                "rank": rank,
                "raw_value": round(rng.uniform(0.0, 100.0), 2),
            }
        )
    reasons: list[dict[str, Any]] = []
    for feature in referenced[:3]:
        verb = (
            "contributed to the denial"
            if feature["signed_direction"] == "increases_denial"
            else "supported approval"
        )
        reasons.append(
            {
                "rank": len(reasons) + 1,
                "reason_text": f"{feature['plain_meaning']} {verb}",
                "supporting_features": [feature["feature_name"]],
                "signed_direction": feature["signed_direction"],
                "specificity_level": "specific",
                "grouped_reason_type": "raw_feature",
            }
        )
    changes: list[dict[str, Any]] = []
    if family == "counterfactual":
        target = referenced[0]
        changes.append(
            {
                "feature_name": target["feature_name"],
                "from_value": target["raw_value"],
                "to_value": round(target["raw_value"] * 0.5, 2),
            }
        )
        reasons.append(
            {
                "rank": len(reasons) + 1,
                "reason_text": f"changing {target['plain_meaning']} may improve the outcome",
                "supporting_features": [target["feature_name"]],
                "signed_direction": "decreases_denial",
                "specificity_level": "specific",
                "grouped_reason_type": "counterfactual",
            }
        )
    return {
        "instance_id": instance_id,
        "dataset_id": dataset_id,
        "track": track,
        "model_id": model_id,
        "explainer_id": explainer_id,
        "explanation_family": family,
        "prediction": {"score": 0.7, "label": "bad", "threshold": 0.5, "decision_margin": 0.2},
        "referenced_features": referenced,
        "principal_reasons": reasons,
        "counterfactual_changes": changes,
        "seed": None,
        "metadata": {},
    }


def synthetic_payload(
    seed: int = 0,
    *,
    rows_per_group: int = 4,
    include_canonical: bool = True,
) -> dict[str, Any]:
    """A small results.json-shaped payload for self-tests (never real data)."""
    rng = random.Random(seed)
    summary: list[dict[str, Any]] = []
    for dataset_id in sorted(_SYNTHETIC_FEATURES):
        for track in ("primary", "audit"):
            for model_id in ("logistic_regression", "ebm", "xgboost"):
                for explainer_id in (*common.REAL_EXPLAINERS, "random_reasons"):
                    family = common.EXPLAINER_FAMILY.get(explainer_id, "attribution")
                    instances = []
                    for row_index in range(rows_per_group):
                        instance_id = f"{dataset_id}-{track}-{model_id}-row{row_index}"
                        instance: dict[str, Any] = {
                            "instance_id": instance_id,
                            "scores": _synthetic_scores(rng, family),
                        }
                        if include_canonical:
                            instance["eval_row_index"] = row_index
                            instance["canonical_explanation"] = _synthetic_canonical(
                                rng,
                                dataset_id=dataset_id,
                                track=track,
                                model_id=model_id,
                                explainer_id=explainer_id,
                                instance_id=instance_id,
                            )
                        instances.append(instance)
                    summary.append(
                        {
                            "dataset_id": dataset_id,
                            "track": track,
                            "model_id": model_id,
                            "explainer_id": explainer_id,
                            "instance_results": instances,
                        }
                    )
    return {"summary": summary}


def selftest() -> None:
    assert format_value(None) == "not recorded"
    assert format_value(3.0) == "3"
    assert format_value(12.5) == "12.5"
    assert format_value(0.12345) == "0.123"
    assert format_value(150000.0) == "150,000"
    assert format_value(True) == "yes"
    assert format_value(" A12 ") == "A12"
    assert blinding_violations("Revolving balance relative to limit") == []
    assert blinding_violations("method: SHAP on xgboost") == ["shap", "xgboost"]
    assert resolve_explainers(None, False) == list(common.REAL_EXPLAINERS)
    assert set(resolve_explainers(None, True)) == {*common.REAL_EXPLAINERS, *common.NULL_BASELINES}
    assert resolve_explainers(["shap", "random_reasons"], False) == ["shap"]

    payload = synthetic_payload()
    real, coverage = collect_candidates(
        payload,
        bundle="synthetic",
        label="synthetic",
        explainers=common.REAL_EXPLAINERS,
        datasets=None,
        tracks=None,
    )
    assert real and coverage["excluded_explainer"] > 0
    assert not any(common.is_null_baseline(item.explainer_id) for item in real)

    selected, _notes = select_notices(real, 20, seed=7)
    assert len(selected) == 20
    explainer_counts = Counter(item.explainer_id for item in selected)
    assert set(explainer_counts) == set(common.REAL_EXPLAINERS)
    assert max(explainer_counts.values()) - min(explainer_counts.values()) <= 1
    dataset_counts = Counter(item.dataset_id for item in selected)
    assert max(dataset_counts.values()) - min(dataset_counts.values()) <= 1
    model_counts = Counter(item.model_id for item in selected)
    assert len(model_counts) == 3
    assert max(model_counts.values()) - min(model_counts.values()) <= 2
    again, _ = select_notices(real, 20, seed=7)
    assert [item.cell + (item.instance_id,) for item in again] == [
        item.cell + (item.instance_id,) for item in selected
    ]

    assigned = assign_notice_ids(selected, 7)
    ids = [notice_id for notice_id, _candidate in assigned]
    assert len(set(ids)) == len(ids) and all(re.fullmatch(r"N\d{5}", item) for item in ids)
    full = [
        render_notice(notice_id, candidate, taxonomy={}, strip_recourse=False)
        for notice_id, candidate in assigned
    ]
    stripped = [
        render_notice(notice_id, candidate, taxonomy={}, strip_recourse=True)
        for notice_id, candidate in assigned
    ]
    for notice in [*full, *stripped]:
        assert not blinding_violations(notice.text), notice.text
        assert notice.text.startswith(HEADER_CREDIT)
        assert RECORD_HEADING in notice.text
        assert 1 <= notice.n_reasons <= MAX_REASONS
    counterfactual = [
        notice
        for notice, (_notice_id, candidate) in zip(full, assigned, strict=True)
        if candidate.family == "counterfactual"
    ]
    assert counterfactual and all(RECOURSE_HEADING in notice.text for notice in counterfactual)
    assert all(
        RECOURSE_HEADING not in notice.text and "may improve the outcome" not in notice.text
        for notice in stripped
    )

    with_baselines, _ = collect_candidates(
        payload,
        bundle="synthetic",
        label="synthetic",
        explainers=resolve_explainers(None, True),
        datasets=None,
        tracks=None,
    )
    assert any(item.explainer_id == "random_reasons" for item in with_baselines)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        bundle_dir = root / "synthetic_bundle"
        bundle_dir.mkdir()
        common.write_json(bundle_dir / "results.json", payload)
        outputs = run([bundle_dir], root / "out", n=12, seed=7, taxonomy={})
        assert all(path.is_file() for path in outputs)
        with (root / "out" / OUTPUT_FILES["key"]).open(encoding="utf-8") as handle:
            key_rows = list(csv.DictReader(handle))
        assert len(key_rows) == 12
        try:
            run([bundle_dir], root / "out", n=12, seed=7, taxonomy={})
        except SystemExit:
            pass
        else:
            raise AssertionError("an existing key must not be overwritten without --force")
        old_dir = root / "old_bundle"
        old_dir.mkdir()
        common.write_json(old_dir / "results.json", synthetic_payload(include_canonical=False))
        try:
            run([old_dir], root / "out_old", n=12, seed=7, taxonomy={})
        except SystemExit as exc:
            assert "persist_canonical_explanations" in str(exc)
        else:
            raise AssertionError("bundles without canonical explanations must be refused")
    print("notices selftest: OK")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render blinded adverse-action-style notices from persisted canonical "
            "explanations for an expert-rating exercise."
        )
    )
    parser.add_argument(
        "--bundle",
        action="append",
        default=[],
        help="Run directory containing results.json (or the file). Repeatable.",
    )
    parser.add_argument("--out-dir", help="Directory for the rating packet.")
    parser.add_argument("--n", type=int, default=48, help="Number of notices (default 48).")
    parser.add_argument("--seed", type=int, default=7, help="Sampling seed (default 7).")
    parser.add_argument(
        "--strip-recourse",
        action="store_true",
        help="Reasons-only condition: drop recourse sections and counterfactual-derived reasons.",
    )
    parser.add_argument(
        "--include-baselines",
        action="store_true",
        help="Also sample the null baselines (random_reasons, global_ranking).",
    )
    parser.add_argument(
        "--explainers",
        help="Comma-separated explainer ids (default: the five real explainers).",
    )
    parser.add_argument(
        "--datasets",
        help="Comma-separated dataset ids (default: the credit datasets).",
    )
    parser.add_argument("--tracks", help="Comma-separated tracks (default: all present).")
    parser.add_argument(
        "--repo-root",
        default=str(REPO_ROOT),
        help=(
            "Repository root against which each bundle's recorded taxonomy (v2 provenance) "
            "or benchmark/metadata/feature_taxonomy.json (v1) is resolved."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing notices_key.csv in --out-dir.",
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
    if not args.bundle:
        parser.error("--bundle is required")
    if not args.out_dir:
        parser.error("--out-dir is required")
    if args.n < 1:
        parser.error("--n must be >= 1")
    outputs = run(
        args.bundle,
        args.out_dir,
        n=args.n,
        seed=args.seed,
        strip_recourse=args.strip_recourse,
        include_baselines=args.include_baselines,
        explainers=split_list(args.explainers),
        datasets=split_list(args.datasets),
        tracks=split_list(args.tracks),
        repo_root=args.repo_root,
        force=args.force,
    )
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
