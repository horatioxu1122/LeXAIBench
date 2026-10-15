"""Specificity of notices (E3) and availability of valid recourse (E6).

E3. Regulation B requires reasons specific to the applicant. For each dataset,
track, model and explanation method we compute the mean pairwise Jaccard overlap
of the four principal reason features across the denied applicants explained in
that configuration. A canned notice gives every applicant the same reasons and
scores 1.0. The same statistic for the comment 9(b)(2)-5 marginal applicant
reasons (row "regb_reference") shows how much the law's own reasons vary across
the same applicants.

E6. For counterfactual methods, the share of denied applicants for whom the
method found at least one change that flips the model's decision using only
factors the applicant can act on (the v2 search blocks immutable,
non-actionable and protected features). Applicants that denial_reasons.py moved
out of the quality metrics still count here.

Intervals are 95% bootstrap intervals over configurations (E3) or applicants
(E6), resampled within dataset. Reads bundles only.

Usage:
    python benchmark/cslaw/notice_specificity.py --bundle DIR [--bundle DIR ...] --out-dir DIR
"""

from __future__ import annotations

import argparse
import csv
import itertools
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

K = 4
COUNTERFACTUAL = ("dice", "wachter")
REFERENCE_ID = "regb_reference"


def mean_pairwise_jaccard(sets: list[frozenset]) -> float | None:
    pairs = [len(a & b) / len(a | b) for a, b in itertools.combinations(sets, 2) if a | b]
    return sum(pairs) / len(pairs) if pairs else None


def specificity_rows(payload: dict) -> list[dict]:
    groups = common.cslaw_section(payload).get("groups") or {}
    notices: dict = defaultdict(list)
    reference: dict = defaultdict(dict)
    for row, instance in common.iter_instances(payload):
        key = (row["dataset_id"], row["track"], row["model_id"])
        features = common.principal_reason_features(instance.get("canonical_explanation"), limit=K)[:K]
        if features:
            notices[key + (row["explainer_id"],)].append(frozenset(features))
        values = (instance.get("native_contributions") or {}).get("values")
        means = (
            ((groups.get(common.cslaw_group_key(*key)) or {}).get("regb_reference") or {})
            .get("near_cutoff", {})
            .get("means")
        )
        index = common.eval_row_index(instance)
        if values and means and index is not None:
            reasons = common.top_adverse_features(values, K, means)
            if reasons:
                reference[key][index] = frozenset(reasons)
    for key, by_row in reference.items():
        notices[key + (REFERENCE_ID,)] = list(by_row.values())
    rows = []
    for (dataset_id, track, model_id, explainer_id), sets in sorted(notices.items()):
        value = mean_pairwise_jaccard(sets)
        if value is not None:
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "track": track,
                    "model_id": model_id,
                    "explainer_id": explainer_id,
                    "applicants": len(sets),
                    "mean_pairwise_jaccard": value,
                }
            )
    return rows


def recourse_rows(payload: dict) -> list[dict]:
    rows = []
    for row in payload.get("summary", []):
        if row["explainer_id"] not in COUNTERFACTUAL:
            continue
        instances = (row.get("instance_results") or []) + (row.get("denial_filter_unproduced_instances") or [])
        for instance in instances:
            metadata = (instance.get("canonical_explanation") or {}).get("metadata") or {}
            flipping = (metadata.get("cf_reason_details") or {}).get("flipping_counterfactuals") or 0
            rows.append(
                {
                    "dataset_id": row["dataset_id"],
                    "track": row["track"],
                    "model_id": row["model_id"],
                    "explainer_id": row["explainer_id"],
                    "valid_recourse": 1.0 if flipping > 0 else 0.0,
                }
            )
    return rows


def stratified_bootstrap(values_by_stratum: dict, n_boot: int, seed: int) -> tuple[float, float, float]:
    rng = random.Random(seed)
    strata = [values for values in values_by_stratum.values() if values]
    flat = [value for values in strata for value in values]
    point = sum(flat) / len(flat)
    stats = []
    for _ in range(n_boot):
        sample = [rng.choice(values) for values in strata for _ in values]
        stats.append(sum(sample) / len(sample))
    stats.sort()
    return point, stats[int(0.025 * n_boot)], stats[int(0.975 * n_boot) - 1]


def summarize(rows: list[dict], value: str, fields: tuple, n_boot: int, seed: int) -> list[dict]:
    grouped: dict = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[tuple(row[f] for f in fields)][row["dataset_id"]].append(row[value])
    out = []
    for key in sorted(grouped):
        point, low, high = stratified_bootstrap(grouped[key], n_boot, seed)
        out.append(
            {
                **dict(zip(fields, key)),
                "mean": point,
                "ci_low": low,
                "ci_high": high,
                "n": sum(len(values) for values in grouped[key].values()),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bundle", action="append", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    spec, recourse = [], []
    for bundle in args.bundle:
        path = Path(bundle)
        payload = common.load_results(path / "results.json" if path.is_dir() else path)
        spec += [{"bundle": path.name, **row} for row in specificity_rows(payload)]
        recourse += [{"bundle": path.name, **row} for row in recourse_rows(payload)]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec_pooled = summarize(spec, "mean_pairwise_jaccard", ("bundle", "explainer_id"), args.n_boot, args.seed)
    spec_track = summarize(spec, "mean_pairwise_jaccard", ("bundle", "track", "explainer_id"), args.n_boot, args.seed)
    rec_dataset = summarize(recourse, "valid_recourse", ("bundle", "dataset_id", "explainer_id"), args.n_boot, args.seed)
    rec_pooled = summarize(recourse, "valid_recourse", ("bundle", "explainer_id"), args.n_boot, args.seed)
    write_csv(out_dir / "specificity_configs.csv", spec)
    write_csv(out_dir / "specificity_pooled.csv", spec_pooled)
    write_csv(out_dir / "specificity_by_track.csv", spec_track)
    write_csv(out_dir / "recourse_by_dataset.csv", rec_dataset)
    write_csv(out_dir / "recourse_pooled.csv", rec_pooled)

    def fmt(row: dict) -> str:
        return f"{row['mean']:.3f} [{row['ci_low']:.3f}, {row['ci_high']:.3f}]"

    lines = [
        "# Notice specificity (E3) and valid recourse (E6)",
        "",
        f"## E3: mean pairwise Jaccard overlap of the first {K} reason features across applicants",
        "",
        "1.0 means every applicant in the configuration receives the same reasons. "
        f"`{REFERENCE_ID}` is the comment 9(b)(2)-5 marginal applicant method on the same applicants.",
        "",
        "| bundle | method | overlap | configurations |",
        "|---|---|---:|---:|",
    ]
    lines += [f"| {r['bundle']} | {r['explainer_id']} | {fmt(r)} | {r['n']} |" for r in spec_pooled]
    lines += ["", "| bundle | track | method | overlap | configurations |", "|---|---|---|---:|---:|"]
    lines += [f"| {r['bundle']} | {r['track']} | {r['explainer_id']} | {fmt(r)} | {r['n']} |" for r in spec_track]
    if recourse:
        lines += [
            "",
            "## E6: share of denied applicants with at least one valid counterfactual",
            "",
            "| bundle | method | share | applicants |",
            "|---|---|---:|---:|",
        ]
        lines += [f"| {r['bundle']} | {r['explainer_id']} | {fmt(r)} | {r['n']} |" for r in rec_pooled]
        lines += ["", "| bundle | dataset | method | share | applicants |", "|---|---|---|---:|---:|"]
        lines += [
            f"| {r['bundle']} | {r['dataset_id']} | {r['explainer_id']} | {fmt(r)} | {r['n']} |" for r in rec_dataset
        ]
    (out_dir / "specificity_recourse.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
