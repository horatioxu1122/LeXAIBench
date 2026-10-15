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

For the commentary methods the script also reports baselines for that change,
on the audit track's applicants and from the audit models' own contributions:
  * refill: the excluded features are left out of the audit model's ranking, so
    its next-ranked factors take the freed places without refitting the model;
  * placebo: as many features that are neither protected nor possible proxies
    are left out instead (every such set, or 500 drawn with a fixed seed);
  * chance: the share expected if four reasons were drawn at random from the
    reason units each track's models use.
Refill and placebo are compared with the audit notices through paired
resamples of the same applicants.
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
from itertools import combinations
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import proxy_substitution  # noqa: E402
import regb_reference as regb  # noqa: E402

K = 4
REFERENCES = {"regb_m1": "near_cutoff", "regb_m2": "all_applicants"}
PLACEBO_MAX_SETS = 500


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


def removal_baselines(
    payload: dict, dataset_id: str, method: str, excluded: set, neither: set, applicants: set, seed: int
) -> dict:
    """{(model, row): (refill units, [placebo units per left-out set])} for the audit track's commentary notices.

    Both come from the audit model's own contributions and reference means: refill leaves the excluded features
    out of the ranking, placebo leaves out as many features labeled neither protected nor possible proxy.
    """
    groups = common.cslaw_section(payload).get("groups") or {}
    fixed = regb.grouping_fixed(payload)
    out: dict = {}
    for row, instance in common.iter_instances(payload):
        if row["dataset_id"] != dataset_id or row["track"] != "audit":
            continue
        key = (row["model_id"], common.eval_row_index(instance))
        if key not in applicants or key in out:
            continue
        values = (instance.get("native_contributions") or {}).get("values")
        reference = (groups.get(common.cslaw_group_key(dataset_id, "audit", row["model_id"])) or {}).get("regb_reference") or {}
        means = (reference.get(REFERENCES[method]) or {}).get("means")
        if not values or not means:
            continue

        def units_without(drop: set, values: dict = values, means: dict = means) -> list:
            kept = {name: value for name, value in values.items() if name not in drop}
            return regb.regb_reason_units(kept, {name: m for name, m in means.items() if name in kept}, K, fixed=fixed) or []

        candidates = sorted(name for name in neither if name in values)
        size = min(len(excluded & set(values)), len(candidates))
        sets = list(combinations(candidates, size)) if size else []
        if len(sets) > PLACEBO_MAX_SETS:
            sets = random.Random(f"{seed}|placebo|{dataset_id}|{method}|{key}").sample(sets, PLACEBO_MAX_SETS)
        out[key] = (units_without(excluded), [units_without(set(drop)) for drop in sets])
    return out


def chance_shares(payload: dict, dataset_id: str, track: str, proxies: set) -> dict:
    """Share of notices that would name each metric's features if K reasons were drawn at random from the
    reason units (a temporal family counts once) that the track's models use."""
    fixed = regb.grouping_fixed(payload)
    features: set = set()
    for row, instance in common.iter_instances(payload):
        if row["dataset_id"] == dataset_id and row["track"] == track:
            features |= set((instance.get("native_contributions") or {}).get("values") or {})
    units: dict = defaultdict(set)
    for name in features:
        units[regb.temporal_group(name, fixed=fixed)].add(name)
    n, k = len(units), min(K, len(units))
    proxy_units = sum(1 for names in units.values() if names & proxies)
    shares = {"names_P": 1 - comb(n - proxy_units, k) / comb(n, k) if n else 0.0}
    for name in proxies:
        shares[f"names_{name}"] = k / n if any(name in names for names in units.values()) else 0.0
    return shares


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
        neither = {
            name
            for name in taxonomy.get(dataset_id, {})
            if proxy_substitution.sensitivity(taxonomy, dataset_id, name) == "N"
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
            if method in REFERENCES:
                rows += baseline_rows(payload, dataset_id, method, kind, per_track, excluded, proxies, neither, boot, n_boot, seed)
    return rows


def baseline_rows(
    payload: dict, dataset_id: str, method: str, kind: str, per_track: dict, excluded: set, proxies: set,
    neither: set, boot: dict, n_boot: int, seed: int,
) -> list[dict]:
    """Refill, placebo, and chance baselines for the change from the audit to the primary track."""
    baselines = removal_baselines(payload, dataset_id, method, excluded, neither, set(per_track["audit"]), seed)
    chance = {track: chance_shares(payload, dataset_id, track, proxies) for track in ("audit", "primary")}

    def named(units: list) -> set:
        return {feature for unit in units for feature in unit}

    predicates = {"names_P": lambda units: bool(named(units) & proxies)}
    for name in sorted(proxies):
        predicates[f"names_{name}"] = lambda units, n=name: n in named(units)
    rows: list[dict] = []

    def add(metric: str, track: str, point: float, low: float, high: float, n: int) -> None:
        rows.append({"dataset_id": dataset_id, "method": method, "kind": kind, "metric": metric, "track": track,
                     "mean": point, "ci_low": low, "ci_high": high, "n": n})

    for metric, predicate in predicates.items():
        refill: dict = defaultdict(list)
        placebo: dict = defaultdict(list)
        # Same applicants in the same order as the audit track's notices, so a resample seeded like the audit
        # track's draws the same applicants (a paired comparison).
        for key in per_track["audit"]:
            if key not in baselines:
                continue
            refill_units, placebo_units = baselines[key]
            refill[key[0]].append(1.0 if predicate(refill_units) else 0.0)
            if placebo_units:
                placebo[key[0]].append(sum(1.0 for units in placebo_units if predicate(units)) / len(placebo_units))
        audit_point, audit_stats = boot[(metric, "audit")]
        primary_point, primary_stats = boot[(metric, "primary")]
        n_audit = sum(len(v) for v in refill.values())
        for track, grouped in (("audit_refill", refill), ("audit_placebo", placebo)):
            if not grouped:
                continue
            point, stats = bootstrap_mean(grouped, n_boot, random.Random(f"{seed}|{dataset_id}|{method}|{metric}|audit"))
            low, high = interval(stats)
            add(metric, track, point, low, high, sum(len(v) for v in grouped.values()))
            if sum(len(v) for v in grouped.values()) == n_audit == len(per_track["audit"]):
                low, high = interval([b - a for b, a in zip(stats, audit_stats)])
                add(metric, f"{track.split('_')[1]}_minus_audit", point - audit_point, low, high, n_audit)
            low, high = interval([p - b for p, b in zip(primary_stats, stats)])
            add(metric, f"primary_minus_{track.split('_')[1]}", primary_point - point, low, high, n_audit)
        for track in ("audit", "primary"):
            share = chance[track][metric]
            add(metric, f"chance_{track}", share, share, share, 0)
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
    base_keys = sorted({(r["bundle"], r["dataset_id"], r["method"], r["metric"]) for r in rows if r["track"] == "audit_refill"})
    if base_keys:
        lines += [
            "",
            "## Baselines for the change between tracks (commentary methods)",
            "",
            "Audit = the audit track's notices. Refill = the audit model's own ranking with the excluded features left "
            "out, so its next-ranked factors fill the freed places (no refit; same applicants). Placebo = the same with "
            "as many features labeled neither protected nor possible proxy left out instead. Chance = four reasons drawn "
            "at random from the track's reason units. Refill and placebo differences with the audit track are paired.",
            "",
            "| bundle | dataset | method | metric | audit | refill | placebo | primary | primary - refill "
            "| refill - audit | chance (audit) | chance (primary) |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for bundle, dataset_id, method, metric in base_keys:
            cells = [cell(bundle, dataset_id, method, metric, track) for track in
                     ("audit", "audit_refill", "audit_placebo", "primary", "primary_minus_refill", "refill_minus_audit")]
            chances = [cell(bundle, dataset_id, method, metric, track).split(" ")[0] for track in ("chance_audit", "chance_primary")]
            lines.append(f"| {bundle} | {dataset_id} | {method} | {metric} | " + " | ".join(cells + chances) + " |")
    (out_dir / "track_disclosure.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
