"""E4 at the level of reason families: correlated factors count as one reason.

A HELOC model can credit "inquiries in the last 6 months" or its near twin "inquiries in the last 6 months excluding
the last 7 days" (rank correlation 0.98), and a change between the two would count as a different first reason. This
script groups the reason units of each consumer credit dataset into families: two units join when any pair of their
numeric features has an absolute Spearman correlation of at least the threshold in the dataset (single linkage;
categorical factors stay alone; monthly fields are already one unit). It then repeats the E4 comparisons with each
reason replaced by its family:
  first_reason   the two commentary methods on the same model, the boosting machine against the boosted trees,
                 and all three models (multiplicity.py's cross-model instances); 95% intervals resample applicants
                 within dataset with both versions together, as in Figure 4
  within         regb_within_class.py's comparisons (seeds, bootstrap refits, classes) from its stored reasons,
                 with its own two-way bootstrap and paired contrasts
Both take a family's place from its highest-ranked member, as the notices treat monthly fields. With --shared-bundle
the first-reason comparisons are also recomputed with each family's contributions summed before ranking
(first_reason_summed), from the stored contributions of the original fits.
Threshold "none" merges nothing and must reproduce the published E4 numbers. Offline; fits no model.

Usage:
    python benchmark/cslaw/reason_families.py --analysis-dir benchmark/outputs/cslaw_denial_<tag>/analysis \
        --data-dir benchmark/data/real --out-dir benchmark/outputs/cslaw_denial_<tag>/analysis/reason_families
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import regb_within_class as within  # noqa: E402

CREDIT = ("heloc", "default_of_credit_card_clients", "give_me_some_credit", "south_german_credit")
DATA_FILES = {
    "heloc": "heloc.csv",
    "default_of_credit_card_clients": "default_credit.csv",
    "give_me_some_credit": "give_me_some_credit.csv",
    "south_german_credit": "south_german_credit.csv",
}
THRESHOLDS = ("none", "0.8", "0.7", "0.6")
MAIN_THRESHOLD = "0.7"


def unit_of(feature: str) -> str:
    return common.temporal_group_label(feature, fixed=True) or feature


def families(data_path: Path, threshold: str) -> dict[str, str]:
    """{unit: family label} for one dataset; a family label joins its member units with ' + '."""
    import pandas as pd

    frame = pd.read_csv(data_path)
    numeric = frame.select_dtypes("number")
    parent: dict[str, str] = {}

    def find(u: str) -> str:
        parent.setdefault(u, u)
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    for column in frame.columns:
        find(unit_of(column))
    if threshold != "none":
        corr = numeric.corr(method="spearman").abs()
        cut = float(threshold)
        columns = list(corr.columns)
        for i, a in enumerate(columns):
            for b in columns[i + 1:]:
                if corr.loc[a, b] >= cut and unit_of(a) != unit_of(b):
                    parent[find(unit_of(a))] = find(unit_of(b))
    members: dict[str, list[str]] = defaultdict(list)
    for unit in list(parent):
        members[find(unit)].append(unit)
    return {unit: " + ".join(sorted(group)) for group in members.values() for unit in group}


def relabel(units: list[str], mapping: dict[str, str]) -> list[str]:
    out: list[str] = []
    for unit in units:
        family = mapping.get(unit, unit)
        if family not in out:
            out.append(family)
    return out


def applicant_rate(units: dict, n_boot: int, seed: int) -> tuple[float, float, float]:
    """units: {(dataset, applicant): [differs, ...]} -> pooled share with a 95% interval resampling applicants
    within dataset (each applicant's comparisons over versions and models stay together)."""
    keys = sorted(units)
    num = np.array([sum(units[k]) for k in keys], dtype=float)
    den = np.array([len(units[k]) for k in keys], dtype=float)
    strata: dict = defaultdict(list)
    for i, key in enumerate(keys):
        strata[key[0]].append(i)
    groups = [np.array(v) for _k, v in sorted(strata.items())]
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        pick = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])
        stats.append(num[pick].sum() / den[pick].sum())
    stats = np.sort(stats)
    return float(num.sum() / den.sum()), float(stats[int(0.025 * n_boot)]), float(stats[int(0.975 * n_boot) - 1])


def first_reason_rows(analysis: Path, maps: dict, threshold: str, n_boot: int, seed: int) -> list[dict]:
    rows = list(csv.DictReader((analysis / "multiplicity" / "multiplicity_cross_model_instances.csv").open(encoding="utf-8")))
    by_method: dict = defaultdict(dict)
    for row in rows:
        if row["method"] in ("regb_m1", "regb_m2") and row["dataset_id"] in CREDIT:
            by_method[row["method"]][(row["dataset_id"], row["track"], row["eval_row_index"])] = row
    comparisons = {"methods": defaultdict(list), "models": defaultdict(list), "three_models": defaultdict(list)}
    for (dataset, track, index), m1 in by_method["regb_m1"].items():
        fam = maps[dataset]
        first = {m: fam.get(m1[f"top1__{m}"], m1[f"top1__{m}"]) for m in ("logistic_regression", "ebm", "xgboost")}
        applicant = (dataset, index)
        m2 = by_method["regb_m2"].get((dataset, track, index))
        if m2 is not None:
            for model in ("ebm", "xgboost"):
                other = fam.get(m2[f"top1__{model}"], m2[f"top1__{model}"])
                comparisons["methods"][applicant].append(float(first[model] != other))
        comparisons["models"][applicant].append(float(first["ebm"] != first["xgboost"]))
        comparisons["three_models"][applicant].append(float(len(set(first.values())) > 1))
    out = []
    for kind, units in comparisons.items():
        for scope in [*CREDIT, "pooled"]:
            subset = {k: v for k, v in units.items() if scope == "pooled" or k[0] == scope}
            if subset:
                mean, low, high = applicant_rate(subset, n_boot, seed)
                out.append({"threshold": threshold, "comparison": kind, "scope": scope, "mean": mean, "ci_low": low,
                            "ci_high": high, "applicants": len(subset)})
    return out


def summed_first(values: dict, means: dict, mapping: dict) -> str | None:
    """The family with the largest positive gap when members' contributions and reference means are summed."""
    gaps: dict = defaultdict(float)
    for feature, value in values.items():
        if feature in means:
            gaps[mapping.get(unit_of(feature), unit_of(feature))] += float(value) - float(means[feature])
    positive = [(gap, family) for family, gap in gaps.items() if gap > 0]
    return max(positive)[1] if positive else None


def summed_first_reason_rows(bundle: Path, maps: dict, threshold: str, n_boot: int, seed: int) -> list[dict]:
    payload = common.load_results(bundle / "results.json")
    groups = common.cslaw_section(payload).get("groups") or {}
    first: dict = {}
    for row, instance in common.iter_instances(payload):
        dataset, track, model = row["dataset_id"], row["track"], row["model_id"]
        index = common.eval_row_index(instance)
        values = (instance.get("native_contributions") or {}).get("values")
        if dataset not in CREDIT or index is None or not values or (dataset, track, model, index) in first:
            continue
        reference = (groups.get(common.cslaw_group_key(dataset, track, model)) or {}).get("regb_reference") or {}
        first[(dataset, track, model, index)] = {
            method: summed_first(values, (reference.get(population) or {}).get("means") or {}, maps[dataset])
            for method, population in (("regb_m1", "near_cutoff"), ("regb_m2", "all_applicants"))
        }
    comparisons = {"methods": defaultdict(list), "models": defaultdict(list), "three_models": defaultdict(list)}
    cells = {(d, t, i) for d, t, _m, i in first}
    for dataset, track, index in sorted(cells):
        reasons = {m: first.get((dataset, track, m, index)) for m in ("logistic_regression", "ebm", "xgboost")}
        if any(r is None for r in reasons.values()):
            continue
        applicant = (dataset, str(index))
        for model in ("ebm", "xgboost"):
            comparisons["methods"][applicant].append(float(reasons[model]["regb_m1"] != reasons[model]["regb_m2"]))
        comparisons["models"][applicant].append(float(reasons["ebm"]["regb_m1"] != reasons["xgboost"]["regb_m1"]))
        comparisons["three_models"][applicant].append(float(len({r["regb_m1"] for r in reasons.values()}) > 1))
    out = []
    for kind, units in comparisons.items():
        for scope in [*CREDIT, "pooled"]:
            subset = {k: v for k, v in units.items() if scope == "pooled" or k[0] == scope}
            if subset:
                mean, low, high = applicant_rate(subset, n_boot, seed)
                out.append({"threshold": threshold, "comparison": kind, "scope": scope, "mean": mean, "ci_low": low,
                            "ci_high": high, "applicants": len(subset)})
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--analysis-dir", required=True)
    parser.add_argument("--data-dir", required=True, help="directory with the consumer credit CSVs")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--shared-bundle", default=None, help="denial-corrected shared bundle (cslaw_denial_<tag>/shared)")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    analysis, out_dir = Path(args.analysis_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fits, reasons = within.load_outputs(analysis / "within_class")
    family_rows, first_rows, summary_rows, contrast_rows, summed_rows = [], [], [], [], []
    for threshold in THRESHOLDS:
        maps = {d: families(Path(args.data_dir) / DATA_FILES[d], threshold) for d in CREDIT}
        for dataset, mapping in maps.items():
            for unit, family in sorted(mapping.items()):
                family_rows.append({"threshold": threshold, "dataset_id": dataset, "unit": unit, "family": family,
                                    "merged": unit != family})
        first_rows += first_reason_rows(analysis, maps, threshold, args.n_boot, args.seed)
        if args.shared_bundle:
            summed_rows += summed_first_reason_rows(Path(args.shared_bundle), maps, threshold, args.n_boot, args.seed)
        mapped = {key: {i: (relabel(units, maps[key[0]]), denied) for i, (units, denied) in by_row.items()}
                  for key, by_row in reasons.items()}
        rows = within.comparisons(fits, mapped, 0.02)
        summary, contrasts = within.summarize(rows, args.n_boot, args.seed)
        summary_rows += [{"threshold": threshold, **r} for r in summary]
        contrast_rows += [{"threshold": threshold, **r} for r in contrasts]
        merged = sum(1 for r in family_rows if r["threshold"] == threshold and r["merged"])
        print(f"[families] threshold {threshold}: {merged} units in a family with another unit", file=sys.stderr)
    write_csv(out_dir / "families.csv", family_rows)
    write_csv(out_dir / "family_first_reason.csv", first_rows)
    write_csv(out_dir / "family_within_summary.csv", summary_rows)
    write_csv(out_dir / "family_within_contrasts.csv", contrast_rows)
    write_csv(out_dir / "family_first_reason_summed.csv", summed_rows)

    def cell(rows: list[dict], value: str, **match) -> str:
        for r in rows:
            if all(str(r.get(k)) == str(v) for k, v in match.items()):
                return f"{100 * float(r[value]):.0f} [{100 * float(r['ci_low']):.0f}, {100 * float(r['ci_high']):.0f}]"
        return "--"

    lines = ["# E4 with correlated factors grouped into reason families", "",
             "First reason differs, pooled over the four consumer credit datasets (95% intervals):", "",
             "| threshold | methods, same model | classes (Fig. 4) | three models | machine refits | trees refits "
             "| classes, same resample | classes - machine refits | classes - trees refits | HELOC classes |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for t in THRESHOLDS:
        s = [r for r in summary_rows if r["threshold"] == t and r["metric"] == "first_reason_differs"]
        c = [r for r in contrast_rows if r["threshold"] == t]
        f = [r for r in first_rows if r["threshold"] == t]
        lines.append("| " + " | ".join([
            t, cell(f, "mean", comparison="methods", scope="pooled"), cell(f, "mean", comparison="models", scope="pooled"),
            cell(f, "mean", comparison="three_models", scope="pooled"),
            cell(s, "mean", kind="within_boot_pairs", model="ebm", scope="pooled"),
            cell(s, "mean", kind="within_boot_pairs", model="xgboost", scope="pooled"),
            cell(s, "mean", kind="across_boot", scope="pooled"),
            cell(c, "difference", within="within_boot_pairs|ebm", scope="pooled"),
            cell(c, "difference", within="within_boot_pairs|xgboost", scope="pooled"),
            cell(f, "mean", comparison="models", scope="heloc")]) + " |")
    if summed_rows:
        lines += ["", "Families' contributions summed before ranking (original fits):", "",
                  "| threshold | methods, same model | classes | three models | HELOC classes |", "|---|---:|---:|---:|---:|"]
        for t in THRESHOLDS:
            f = [r for r in summed_rows if r["threshold"] == t]
            lines.append("| " + " | ".join([t, cell(f, "mean", comparison="methods", scope="pooled"),
                                             cell(f, "mean", comparison="models", scope="pooled"),
                                             cell(f, "mean", comparison="three_models", scope="pooled"),
                                             cell(f, "mean", comparison="models", scope="heloc")]) + " |")
    lines += ["", f"Families with more than one unit (threshold {MAIN_THRESHOLD}):", ""]
    seen = set()
    for r in family_rows:
        if r["threshold"] == MAIN_THRESHOLD and r["merged"] and (r["dataset_id"], r["family"]) not in seen:
            seen.add((r["dataset_id"], r["family"]))
            lines.append(f"- {r['dataset_id']}: {r['family']}")
    (out_dir / "reason_families.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
