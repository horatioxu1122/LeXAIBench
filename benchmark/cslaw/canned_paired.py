"""E2 pass rule: each method's share of the marginal applicant method's reasons against the canned notice's, paired.

For every applicant explained by both a method and the canned notice (same dataset, version, model and evaluation
row), the difference of the two shares of the marginal applicant method's four reasons (pr_overlap_m1_grouped in
regb_reference.py's instances). A method meets E2 when the 95% bootstrap interval of the mean difference lies above
zero. Each resample draws the applicants once within each combination of dataset and version and uses both notices
of every drawn applicant (2,000 resamples, seed 42). Offline; reads regb_reference.py outputs only.

Usage:
    python benchmark/cslaw/canned_paired.py --analysis-dir benchmark/outputs/cslaw_denial_<tag>/analysis \
        --out-dir benchmark/outputs/cslaw_denial_<tag>/analysis/canned_paired
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

CREDIT = ("heloc", "default_of_credit_card_clients", "give_me_some_credit", "south_german_credit")
# (bundle label, regb_reference.py output folder); the robustness runs are used when present.
SOURCES = (("credit", "regb_credit"), ("lime_local", "regb_lime_local"), ("lime_default", "regb_lime_default"))
CANNED = ("regb_nullbaselines", "global_ranking")


def shares(path: Path, explainer: str | None = None) -> dict:
    """{explainer: {(dataset, version, model, row): share}} for the consumer credit notices of one output folder."""
    out: dict = defaultdict(dict)
    with (path / "regb_reference_instances.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            value = row.get("pr_overlap_m1_grouped")
            if row["dataset_id"] not in CREDIT or value in (None, "", "None", "nan"):
                continue
            if explainer is not None and row["explainer_id"] != explainer:
                continue
            key = (row["dataset_id"], row["track"], row["model_id"], row["eval_row_index"])
            out[row["explainer_id"]][key] = float(value)
    return out


def paired(method: dict, canned: dict, n_boot: int, seed: int) -> dict | None:
    keys = sorted(set(method) & set(canned))
    if not keys:
        return None
    diff = np.array([method[k] - canned[k] for k in keys])
    strata: dict = defaultdict(list)
    for i, key in enumerate(keys):
        strata[(key[0], key[1])].append(i)
    groups = [np.array(v) for _k, v in sorted(strata.items())]
    rng = np.random.default_rng(seed)
    boots = np.sort([diff[np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])].mean()
                     for _ in range(n_boot)])
    return {"n": len(keys), "method": float(np.mean([method[k] for k in keys])),
            "canned": float(np.mean([canned[k] for k in keys])), "difference": float(diff.mean()),
            "ci_low": float(boots[int(0.025 * n_boot)]), "ci_high": float(boots[int(0.975 * n_boot) - 1])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--analysis-dir", required=True, help="cslaw_denial_<tag>/analysis")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    analysis = Path(args.analysis_dir)
    canned = shares(analysis / CANNED[0], CANNED[1])[CANNED[1]]
    rows = []
    for bundle, folder in SOURCES:
        if not (analysis / folder / "regb_reference_instances.csv").is_file():
            continue
        for explainer, values in sorted(shares(analysis / folder).items()):
            if explainer == "regb_reference":
                continue
            result = paired(values, canned, args.n_boot, args.seed)
            if result:
                rows.append({"bundle": bundle, "explainer": explainer, **result})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["bundle", "explainer", "n", "method", "canned", "difference", "ci_low", "ci_high"]
    with (out_dir / "canned_paired.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# E2: share of the marginal applicant method's reasons, method minus canned notice (paired)", "",
             "| bundle | method | n | method share | canned share | difference (points) | 95% interval | meets E2 |",
             "|---|---|---:|---:|---:|---:|---|---|"]
    for r in rows:
        lines.append(f"| {r['bundle']} | {r['explainer']} | {r['n']} | {r['method']:.3f} | {r['canned']:.3f} | "
                     f"{100 * r['difference']:+.1f} | [{100 * r['ci_low']:.1f}, {100 * r['ci_high']:.1f}] | "
                     f"{'yes' if r['ci_low'] > 0 else 'no'} |")
    (out_dir / "canned_paired.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
