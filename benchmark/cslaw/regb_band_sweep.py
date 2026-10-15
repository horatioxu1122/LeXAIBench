"""Vary the marginal applicant reference band of comment 9(b)(2)-5 (cluster only).

The marginal applicant method compares a denied applicant with approved
applicants "at or slightly above the minimum passing score". The main runs use
approved applicants within 0.10 of the cutoff (at least 200). This script refits
each credit model exactly as the runner does, recomputes the reference means for
other bands with the runner's own function, and recomputes the method's reasons
for the same denied applicants that the corrected bundle explains.

It reports, per dataset, track and model and per band:
  * how many approved applicants form the reference group,
  * agreement of the reasons with the 0.10 / 200 baseline (overlap of the four
    reasons, same first reason),
  * agreement of each explanation method's notice with the reasons.
It also checks that the baseline band reproduces the reference means saved in
the original bundle and that the recomputed contributions of the explained
applicants match the persisted ones.

This script fits models. Run it on the cluster:
    PYTHONPATH=benchmark/src .venv/bin/python benchmark/cslaw/regb_band_sweep.py \
        --config benchmark/configs/experiment_cslaw_v2_credit.json \
        --source-bundle benchmark/outputs/cslaw_v2_credit_<tag> \
        --notice-bundle benchmark/outputs/cslaw_denial_<tag>/credit \
        --out-dir benchmark/outputs/cslaw_denial_<tag>/analysis/band_sweep
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

BANDS = (0.02, 0.05, 0.10, 0.20, 0.30)
MIN_ROWS_VARIANTS = (50, 200, 500)
BASELINE = (0.10, 200)
K = 4
REAL_EXPLAINERS = ("anchors", "dice", "lime", "shap", "wachter")


def variants() -> list[tuple[float, int]]:
    items = [(band, BASELINE[1]) for band in BANDS]
    items += [(BASELINE[0], rows) for rows in MIN_ROWS_VARIANTS if rows != BASELINE[1]]
    return items


def overlap(a: list[str], b: list[str], k: int = K) -> float | None:
    return common.overlap_at_k(a, b, k)


def notice_features(canonical: dict | None) -> list[str]:
    return common.principal_reason_features(canonical, limit=K)[:K]


def collect_targets(notice_payload: dict) -> dict:
    """{(dataset, track, model): {"rows": sorted eval indices, "notices": {(explainer, row): features}}}"""
    targets: dict = defaultdict(lambda: {"rows": set(), "notices": {}, "native": {}})
    for row, instance in common.iter_instances(notice_payload):
        index = common.eval_row_index(instance)
        if index is None:
            continue
        key = (row["dataset_id"], row["track"], row["model_id"])
        entry = targets[key]
        entry["rows"].add(index)
        if row["explainer_id"] in REAL_EXPLAINERS:
            entry["notices"][(row["explainer_id"], index)] = notice_features(
                instance.get("canonical_explanation")
            )
        native = instance.get("native_contributions") or {}
        if native.get("values"):
            entry["native"][index] = native["values"]
    for entry in targets.values():
        entry["rows"] = sorted(entry["rows"])
    return targets


def saved_means(source_payload: dict) -> dict:
    groups = (source_payload.get("cslaw") or {}).get("groups") or {}
    return {key: (group.get("regb_reference") or {}) for key, group in groups.items()}


def run(args: argparse.Namespace) -> None:
    from legalbench_xai.config import load_experiment_config
    from legalbench_xai.datasets.registry import build_dataset_loader
    from legalbench_xai.models import fit_model
    from legalbench_xai.runner import (
        _cslaw_regb_reference,
        _predict_with_threshold_policy,
        _split_table_for_evaluation,
    )

    config = load_experiment_config(args.config)
    source_payload = common.load_results(Path(args.source_bundle) / "results.json")
    notice_payload = common.load_results(Path(args.notice_bundle) / "results.json")
    targets = collect_targets(notice_payload)
    saved = saved_means(source_payload)
    regb_cfg = ((config.payload.get("cslaw") or {}).get("regb_reference") or {})
    max_rows = int(regb_cfg.get("all_applicants_max_rows", 5000))
    seed = int(regb_cfg.get("seed", 42))

    out_rows: list[dict] = []
    checks: list[dict] = []
    datasets = sorted({key[0] for key in targets})
    for dataset_id in datasets:
        loader = build_dataset_loader(config, dataset_id)
        for track in sorted({key[1] for key in targets if key[0] == dataset_id}):
            bundle = loader.build_bundle(track)
            table = loader.load_tabular_dataset(track=track)
            train_table, eval_table, _ = _split_table_for_evaluation(
                table=table,
                negative_label=bundle.negative_label,
                evaluation_cfg=config.evaluation_protocol,
            )
            for model_id in sorted({key[2] for key in targets if key[:2] == (dataset_id, track)}):
                key = (dataset_id, track, model_id)
                print(f"[band_sweep] {dataset_id} {track} {model_id}", flush=True)
                model, backend = fit_model(
                    model_id=model_id,
                    dataset_bundle=bundle,
                    table=train_table,
                    threshold=config.model_threshold,
                )
                if getattr(backend, "backend", "real") != "real":
                    sys.exit(f"stub backend for {key}: {backend}")
                predictions, _ = _predict_with_threshold_policy(
                    model=model,
                    rows=eval_table.rows,
                    instance_id_prefix=f"{dataset_id}-{track}-{model_id}",
                    threshold_policy=config.model_threshold_policy,
                )
                target = targets[key]
                rows = [eval_table.rows[i] for i in target["rows"]]
                contributions, _bias = model.native_feature_contributions_batch(rows)
                native = dict(zip(target["rows"], contributions))

                persisted_gap = max(
                    (
                        abs(native[i].get(f, 0.0) - float(v))
                        for i, values in target["native"].items()
                        for f, v in values.items()
                    ),
                    default=None,
                )
                references = {}
                for band, min_rows in variants():
                    record = _cslaw_regb_reference(
                        model=model,
                        negative_label=bundle.negative_label,
                        predictions=predictions,
                        settings={
                            "near_cutoff_max_band": band,
                            "near_cutoff_min_rows": min_rows,
                            "all_applicants_max_rows": max_rows,
                            "seed": seed,
                        },
                        errors=[],
                    )
                    references[(band, min_rows)] = record
                base = references[BASELINE]
                saved_record = saved.get(common.cslaw_group_key(*key), {})
                saved_near = (saved_record.get("near_cutoff") or {}).get("means") or {}
                means_gap = max(
                    (abs(base["near_cutoff"]["means"].get(f, 0.0) - float(v)) for f, v in saved_near.items()),
                    default=None,
                )
                checks.append(
                    {
                        "dataset_id": dataset_id,
                        "track": track,
                        "model_id": model_id,
                        "baseline_means_max_gap_vs_saved": means_gap,
                        "contributions_max_gap_vs_persisted": persisted_gap,
                    }
                )

                base_reasons = {
                    i: common.top_adverse_features(native[i], K, base["near_cutoff"]["means"])
                    for i in target["rows"]
                }
                all_reasons = {
                    i: common.top_adverse_features(native[i], K, base["all_applicants"]["means"])
                    for i in target["rows"]
                }
                for (band, min_rows), record in references.items():
                    reasons = {
                        i: common.top_adverse_features(native[i], K, record["near_cutoff"]["means"])
                        for i in target["rows"]
                    }
                    row = {
                        "dataset_id": dataset_id,
                        "track": track,
                        "model_id": model_id,
                        "band": band,
                        "min_rows": min_rows,
                        "reference_rows": record["near_cutoff"]["n_rows"],
                        "selection": record["near_cutoff"]["selection"],
                        "score_low": record["near_cutoff"]["score_low"],
                        "applicants": len(target["rows"]),
                        "overlap_vs_baseline": common.mean(
                            [overlap(reasons[i], base_reasons[i]) for i in target["rows"]]
                        ),
                        "top1_same_as_baseline": common.mean(
                            [float(common.top1_match(reasons[i], base_reasons[i]) or 0) for i in target["rows"]]
                        ),
                        "top1_same_as_average_method": common.mean(
                            [float(common.top1_match(reasons[i], all_reasons[i]) or 0) for i in target["rows"]]
                        ),
                    }
                    for explainer in REAL_EXPLAINERS:
                        row[f"overlap_{explainer}"] = common.mean(
                            [
                                overlap(target["notices"][(explainer, i)], reasons[i])
                                for i in target["rows"]
                                if (explainer, i) in target["notices"]
                            ]
                        )
                    out_rows.append(row)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("band_sweep_rows.csv", out_rows), ("band_sweep_checks.csv", checks)):
        with (out_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    write_summary(out_rows, checks, out_dir / "band_sweep.md")
    print(f"wrote {out_dir}")


def write_summary(rows: list[dict], checks: list[dict], path: Path) -> None:
    pooled: dict = defaultdict(list)
    for row in rows:
        pooled[(row["model_id"], row["band"], row["min_rows"])].append(row)
    lines = [
        "# Marginal applicant reference band sweep (comment 9(b)(2)-5)",
        "",
        f"Baseline: band {BASELINE[0]}, at least {BASELINE[1]} approved applicants; four reasons; "
        "feature-level comparison; means over dataset x track groups.",
        "",
        "| model | band | min rows | mean reference rows | overlap vs baseline | same first reason as baseline "
        "| same first reason as average method | SHAP | LIME | DiCE | Anchors | Wachter |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (model_id, band, min_rows), items in sorted(pooled.items()):
        def avg(field: str) -> str:
            value = common.mean([item[field] for item in items])
            return "n/a" if value is None else f"{value:.3f}"

        lines.append(
            f"| {model_id} | {band:.2f} | {min_rows} | {avg('reference_rows')} | {avg('overlap_vs_baseline')} "
            f"| {avg('top1_same_as_baseline')} | {avg('top1_same_as_average_method')} | {avg('overlap_shap')} "
            f"| {avg('overlap_lime')} | {avg('overlap_dice')} | {avg('overlap_anchors')} | {avg('overlap_wachter')} |"
        )
    worst_means = max((c["baseline_means_max_gap_vs_saved"] or 0.0) for c in checks)
    worst_native = max((c["contributions_max_gap_vs_persisted"] or 0.0) for c in checks)
    lines += [
        "",
        f"Consistency: baseline reference means match the saved run to within {worst_means:.2e}; "
        f"recomputed contributions match the persisted ones to within {worst_native:.2e}.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="experiment config of the source credit run")
    parser.add_argument("--source-bundle", required=True, help="original v2 credit bundle (saved reference means)")
    parser.add_argument("--notice-bundle", required=True, help="denial-corrected credit bundle (notices)")
    parser.add_argument("--out-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
