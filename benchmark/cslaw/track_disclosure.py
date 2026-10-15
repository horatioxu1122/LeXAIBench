"""What notices disclose about protected characteristics and their proxies (E5).

proxy_substitution.py compares the primary and audit tracks for the same
applicant and needs both tracks to explain the same applicants. This script
compares the two tracks as populations, which works for every bundle:

  * audit track: share of denied applicants whose notice names a feature that
    the primary track excludes (a protected characteristic), and share whose
    first reason is such a feature;
  * both tracks: share of notices that name a feature the taxonomy labels P
    (possible proxy), the change from audit to primary, and the share naming
    each P feature.

A notice is its first four reasons as stated (a temporal family such as the
PAY_* fields is one reason), as in proxy_substitution.py; "regb_m1" / "regb_m2"
are the comment 9(b)(2)-5 marginal and average applicant reasons computed from
the persisted native contributions and reference means, grouped the same way.
Intervals are 95% bootstrap intervals that resample applicants within model;
the change between tracks resamples the two tracks independently.
Reads bundles only.

Usage:
    python benchmark/cslaw/track_disclosure.py --bundle DIR [--bundle DIR ...] --out-dir DIR
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import proxy_substitution  # noqa: E402
import regb_reference as regb  # noqa: E402

K = 4
REFERENCES = {"regb_m1": "near_cutoff", "regb_m2": "all_applicants"}


def notice_sets(payload: dict) -> dict:
    """{(dataset, track, method): {(model, row): (kind, [[features of reason 1], ...])}}."""
    groups = common.cslaw_section(payload).get("groups") or {}
    fixed = regb.grouping_fixed(payload)
    out: dict = defaultdict(dict)
    for row, instance in common.iter_instances(payload):
        dataset_id, track, model_id = row["dataset_id"], row["track"], row["model_id"]
        index = common.eval_row_index(instance)
        features = regb.reason_units(instance.get("canonical_explanation"), "principal", K)
        kind = "null_baseline" if common.is_null_baseline(row["explainer_id"]) else "explainer"
        if features:
            out[(dataset_id, track, row["explainer_id"])][(model_id, index)] = (kind, features)
        values = (instance.get("native_contributions") or {}).get("values")
        reference = (groups.get(common.cslaw_group_key(dataset_id, track, model_id)) or {}).get("regb_reference") or {}
        if not values or index is None:
            continue
        for method, population in REFERENCES.items():
            means = (reference.get(population) or {}).get("means")
            if means and (model_id, index) not in out[(dataset_id, track, method)]:
                reasons = regb.regb_reason_units(values, means, K, fixed=fixed)
                if reasons:
                    out[(dataset_id, track, method)][(model_id, index)] = ("regb_reference", reasons)
    return out


def bootstrap_mean(by_model: dict, n_boot: int, rng: random.Random) -> tuple[float, list[float]]:
    strata = [values for values in by_model.values() if values]
    flat = [value for values in strata for value in values]
    stats = []
    for _ in range(n_boot):
        sample = [rng.choice(values) for values in strata for _ in values]
        stats.append(sum(sample) / len(sample))
    return sum(flat) / len(flat), stats


def interval(stats: list[float]) -> tuple[float, float]:
    ordered = sorted(stats)
    n = len(ordered)
    return ordered[int(0.025 * n)], ordered[int(0.975 * n) - 1]


def disclosure_rows(payload: dict, registry: dict, taxonomy: dict, n_boot: int, seed: int) -> list[dict]:
    notices = notice_sets(payload)
    rows = []
    datasets = sorted({key[0] for key in notices})
    for dataset_id in datasets:
        excluded = set(proxy_substitution.excluded_in_primary(registry, taxonomy, dataset_id))
        proxies = {
            name
            for name in taxonomy.get(dataset_id, {})
            if proxy_substitution.sensitivity(taxonomy, dataset_id, name) == "P"
        }
        tracks = {key[1] for key in notices if key[0] == dataset_id}
        if not excluded or not {"primary", "audit"} <= tracks:
            continue
        methods = sorted({key[2] for key in notices if key[0] == dataset_id})
        for method in methods:
            per_track = {track: notices.get((dataset_id, track, method), {}) for track in ("audit", "primary")}
            if not per_track["audit"] or not per_track["primary"]:
                continue
            kind = next(iter(per_track["audit"].values()))[0]
            metrics: dict = {}

            def by_model(track: str, predicate) -> dict:
                grouped: dict = defaultdict(list)
                for (model_id, _index), (_kind, features) in per_track[track].items():
                    grouped[model_id].append(1.0 if predicate(features) else 0.0)
                return grouped

            def named(units: list) -> set:
                return {feature for unit in units for feature in unit}

            metrics[("names_excluded", "audit")] = by_model("audit", lambda u: bool(named(u) & excluded))
            metrics[("first_excluded", "audit")] = by_model("audit", lambda u: bool(set(u[0]) & excluded))
            for track in ("audit", "primary"):
                metrics[("names_P", track)] = by_model(track, lambda u: bool(named(u) & proxies))
                for name in sorted(proxies):
                    metrics[(f"names_{name}", track)] = by_model(track, lambda u, n=name: n in named(u))
            boot: dict = {}
            for (metric, track), grouped in metrics.items():
                rng = random.Random(f"{seed}|{dataset_id}|{method}|{metric}|{track}")
                point, stats = bootstrap_mean(grouped, n_boot, rng)
                boot[(metric, track)] = (point, stats)
                low, high = interval(stats)
                rows.append(
                    {
                        "dataset_id": dataset_id,
                        "method": method,
                        "kind": kind,
                        "metric": metric,
                        "track": track,
                        "mean": point,
                        "ci_low": low,
                        "ci_high": high,
                        "n": sum(len(v) for v in grouped.values()),
                    }
                )
            for metric in ["names_P"] + [f"names_{name}" for name in sorted(proxies)]:
                (p_point, p_stats), (a_point, a_stats) = boot[(metric, "primary")], boot[(metric, "audit")]
                low, high = interval([p - a for p, a in zip(p_stats, a_stats)])
                rows.append(
                    {
                        "dataset_id": dataset_id,
                        "method": method,
                        "kind": kind,
                        "metric": metric,
                        "track": "primary_minus_audit",
                        "mean": p_point - a_point,
                        "ci_low": low,
                        "ci_high": high,
                        "n": sum(len(v) for v in metrics[(metric, "primary")].values()),
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bundle", action="append", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--repo-root", default=str(proxy_substitution.DEFAULT_REPO_ROOT))
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = []
    for bundle in args.bundle:
        path = Path(bundle)
        payload = common.load_results(path / "results.json" if path.is_dir() else path)
        registry, taxonomy, _sources = proxy_substitution.bundle_metadata(payload, args.repo_root)
        rows += [{"bundle": path.name, **row} for row in disclosure_rows(payload, registry, taxonomy, args.n_boot, args.seed)]
    if not rows:
        sys.exit("no dataset with both tracks and primary-track exclusions in the given bundles")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "track_disclosure.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    def cell(bundle: str, dataset_id: str, method: str, metric: str, track: str) -> str:
        for row in rows:
            if (row["bundle"], row["dataset_id"], row["method"], row["metric"], row["track"]) == (bundle, dataset_id, method, metric, track):
                return f"{100 * row['mean']:.0f}% [{100 * row['ci_low']:.0f}, {100 * row['ci_high']:.0f}]"
        return "n/a"

    lines = [
        "# Disclosure of protected characteristics and proxies (track populations)",
        "",
        "Excluded = features the primary track drops (protected characteristics). P = taxonomy label for a "
        "possible proxy. Shares of denied applicants' notices (first four stated reasons); the two tracks "
        "explain different applicants, so the change is a population comparison.",
        "",
        "| bundle | dataset | method | audit: names excluded | audit: first reason excluded | names P (audit) "
        "| names P (primary) | change |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    keys = sorted({(r["bundle"], r["dataset_id"], r["method"]) for r in rows})
    for bundle, dataset_id, method in keys:
        lines.append(
            f"| {bundle} | {dataset_id} | {method} | {cell(bundle, dataset_id, method, 'names_excluded', 'audit')} "
            f"| {cell(bundle, dataset_id, method, 'first_excluded', 'audit')} | {cell(bundle, dataset_id, method, 'names_P', 'audit')} "
            f"| {cell(bundle, dataset_id, method, 'names_P', 'primary')} | {cell(bundle, dataset_id, method, 'names_P', 'primary_minus_audit')} |"
        )
    (out_dir / "track_disclosure.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
