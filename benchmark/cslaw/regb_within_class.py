"""First-reason variation within a model class: the baseline for E4 (cluster only).

E4 compares the marginal applicant method's first reason between the explainable boosting machine and the gradient
boosted trees for the applicants explained under all three models. This script asks how often the first reason
changes when a creditor keeps the model class and only refits it. For each consumer credit dataset and version it
refits each class on bootstrap resamples of the training split (drawn within each outcome class, one resample per
index shared by every class), and the boosting machine also with other random seeds on the original split (the
boosted trees, with subsample 1.0, and logistic regression give the same model for any seed). A refit is kept if its
evaluation ROC AUC is within --auc-tolerance of the original fit; a pair of the two classes on the same resample is
kept if both are kept and their ROC AUC differ by at most the same tolerance, the paper's criterion for equally
accurate models. Each fit's marginal applicant reasons for the shared applicants come from its own contributions and
its own reference means, in the reason units of multiplicity.py (a temporal family counts once).

Comparisons, each on the same applicants and version:
  within_seed               boosting machine, original fit against each other seed (same data)
  within_boot_pairs         each class, every pair of bootstrap refits (the matched contrast for across_boot)
  within_boot_vs_original   each class, original fit against each bootstrap refit (secondary)
  across_original           boosting machine against boosted trees, original fits (reproduces Figure 4)
  across_boot               boosting machine against boosted trees on the same resample
Reported per dataset and pooled: the share of comparisons with a different first reason and the mean overlap of the
four reasons, for every applicant (the method's reasons exist whether or not a refit denies) and for the applicants
both fits deny, plus the share of comparisons in which a fit approves the applicant. Each applicant and version
counts once, averaged over its refits, as in Figure 4. Intervals are 95% bootstrap intervals that resample the
refits and the applicants within dataset (versions together); paired intervals compare each across-class rate with
the matching within-class rate on the same draws.

The original fits must reproduce the persisted contributions and the saved reference means of the shared run; the
script stops otherwise. Fits and reasons are written as they finish, so --from-reasons can redo the comparisons and
--datasets can rerun part of the work.

This script fits models. Run it on the cluster:
    PYTHONPATH=benchmark/src .venv/bin/python benchmark/cslaw/regb_within_class.py \\
        --config benchmark/configs/experiment_cslaw_v2_credit_shared.json \\
        --shared-bundle benchmark/outputs/cslaw_v2_credit_shared_<tag> \\
        --out-dir benchmark/outputs/cslaw_denial_<tag>/analysis/within_class
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from dataclasses import replace
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import regb_reference as regb  # noqa: E402

CREDIT = ("heloc", "default_of_credit_card_clients", "give_me_some_credit", "south_german_credit")
DATASET_LABEL = {
    "heloc": "HELOC",
    "default_of_credit_card_clients": "Default Credit",
    "give_me_some_credit": "Give Me Some Credit",
    "south_german_credit": "South German Credit",
}
K = 4
SEEDED = ("ebm",)  # classes whose fit depends on random_state
REPRODUCTION_TOLERANCE = 1e-8
KINDS = (
    ("within_seed", "ebm", "Boosting machine: original fit vs another seed (same data)"),
    ("within_boot_pairs", "ebm", "Boosting machine: two bootstrap refits"),
    ("within_boot_pairs", "xgboost", "Boosted trees: two bootstrap refits"),
    ("across_original", "ebm_vs_xgboost", "Boosting machine vs boosted trees: original fits"),
    ("across_boot", "ebm_vs_xgboost", "Boosting machine vs boosted trees: same bootstrap resample"),
    ("within_boot_vs_original", "ebm", "Boosting machine: original fit vs bootstrap refit"),
    ("within_boot_vs_original", "xgboost", "Boosted trees: original fit vs bootstrap refit"),
)
CONTRASTS = (
    (("across_original", "ebm_vs_xgboost"), ("within_seed", "ebm")),
    (("across_boot", "ebm_vs_xgboost"), ("within_boot_pairs", "ebm")),
    (("across_boot", "ebm_vs_xgboost"), ("within_boot_pairs", "xgboost")),
)
FIT_FIELDS = ["dataset_id", "track", "model_id", "variant", "random_state", "bootstrap", "roc_auc", "auc_gap_to_original",
              "kept", "error", "applicants", "denied_share", "reference_rows", "ebm_best_iteration_mean",
              "ebm_best_iteration_max"]
REASON_FIELDS = ["dataset_id", "track", "model_id", "variant", "eval_row_index", "denied", "units"]


# --------------------------------------------------------------------- inputs
def shared_applicants(payload: dict) -> tuple[dict, dict]:
    """({(dataset, track): eval rows explained under every model}, {(dataset, track, model, row): contributions})."""
    rows: dict = defaultdict(lambda: defaultdict(set))
    native: dict = {}
    for summary_row, instance in common.iter_instances(payload):
        index = common.eval_row_index(instance)
        if index is None or summary_row["dataset_id"] not in CREDIT:
            continue
        key = (summary_row["dataset_id"], summary_row["track"])
        rows[key][summary_row["model_id"]].add(index)
        values = regb.native_values(instance)
        if values:
            native[(*key, summary_row["model_id"], index)] = values
    shared = {key: sorted(set.intersection(*by_model.values())) for key, by_model in rows.items()}
    return {key: value for key, value in shared.items() if value}, native


def bootstrap(table, key: str, target: str, adverse_label: str):
    """The training split resampled with replacement within each outcome class, seeded by key."""
    rng = random.Random(key)
    by_class: dict = defaultdict(list)
    for index, row in enumerate(table.rows):
        by_class[str(row.get(target)) == adverse_label].append(index)
    picked = [indexes[rng.randrange(len(indexes))] for _cls, indexes in sorted(by_class.items()) for _ in indexes]
    return replace(table, rows=[table.rows[i] for i in picked])


def max_gap(left: dict, right: dict) -> float | None:
    keys = set(left) | set(right)
    if not keys:
        return None
    return max(abs(float(left.get(name, 0.0)) - float(right.get(name, 0.0))) for name in keys)


def ebm_iterations(model) -> tuple[float | None, float | None]:
    best = getattr(getattr(model, "classifier", None), "best_iteration_", None)
    if best is None:
        return None, None
    values = np.asarray(best, dtype=float)
    return float(values.mean()), float(values.max())


class Writer:
    """Appends rows to a CSV as fits finish."""

    def __init__(self, path: Path, fields: list[str], append: bool) -> None:
        exists = append and path.is_file()
        self.handle = path.open("a" if exists else "w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.handle, fieldnames=fields)
        if not exists:
            self.writer.writeheader()

    def write(self, row: dict) -> None:
        self.writer.writerow(row)
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


# --------------------------------------------------------------------- fits
def fit_all(args: argparse.Namespace, out_dir: Path) -> list[dict]:
    from legalbench_xai.config import load_experiment_config
    from legalbench_xai.datasets.registry import build_dataset_loader
    from legalbench_xai.models import fit_model
    from legalbench_xai.runner import (
        _cslaw_regb_reference,
        _evaluate_model_predictions,
        _predict_with_threshold_policy,
        _split_table_for_evaluation,
    )

    config = load_experiment_config(args.config)
    payload = common.load_results(Path(args.shared_bundle) / "results.json")
    fixed = regb.grouping_fixed(payload)
    applicants, persisted = shared_applicants(payload)
    if not applicants:
        sys.exit(f"no applicant is explained under every model in {args.shared_bundle}; pass the shared run's bundle")
    regb_cfg = (config.payload.get("cslaw") or {}).get("regb_reference") or {}
    settings = {
        "near_cutoff_max_band": float(regb_cfg.get("near_cutoff_max_band", 0.10)),
        "near_cutoff_min_rows": int(regb_cfg.get("near_cutoff_min_rows", 200)),
        "all_applicants_max_rows": int(regb_cfg.get("all_applicants_max_rows", 5000)),
        "seed": int(regb_cfg.get("seed", 42)),
    }
    append = bool(args.datasets)
    fits_out = Writer(out_dir / "within_class_fits.csv", FIT_FIELDS, append)
    reasons_out = Writer(out_dir / "within_class_reasons.csv", REASON_FIELDS, append)
    checks: list[dict] = []
    datasets = [d for d in CREDIT if any(key[0] == d for key in applicants) and (not args.datasets or d in args.datasets)]
    try:
        for dataset_id in datasets:
            loader = build_dataset_loader(config, dataset_id)
            for track in ("primary", "audit"):
                rows_idx = applicants.get((dataset_id, track)) or []
                if not rows_idx:
                    continue
                bundle = loader.build_bundle(track)
                table = loader.load_tabular_dataset(track=track)
                train_table, eval_table, _ = _split_table_for_evaluation(
                    table=table,
                    negative_label=bundle.negative_label,
                    evaluation_cfg=config.evaluation_protocol,
                )
                # One resample per index, shared by every class, so across-class pairs see the same data.
                resamples = {
                    b: bootstrap(train_table, f"{args.seed}|{dataset_id}|{track}|boot{b}", bundle.target,
                                 bundle.negative_label)
                    for b in range(1, args.boots + 1)
                }
                for model_id in args.models:
                    variants = [("original", 0, None)]
                    if model_id in SEEDED:
                        variants += [(f"seed{s}", s, None) for s in range(1, args.seeds + 1)]
                    variants += [(f"boot{b}", 0, b) for b in range(1, args.boots + 1)]
                    original_auc = None
                    for variant, random_state, b in variants:
                        print(f"[within_class] {dataset_id} {track} {model_id} {variant}", flush=True)
                        fit = {"dataset_id": dataset_id, "track": track, "model_id": model_id, "variant": variant,
                               "random_state": random_state, "bootstrap": b or "", "applicants": len(rows_idx)}
                        model, backend = fit_model(
                            model_id=model_id,
                            dataset_bundle=bundle,
                            table=train_table if b is None else resamples[b],
                            threshold=config.model_threshold,
                            random_state=random_state,
                        )
                        if getattr(backend, "backend", "real") != "real":
                            if variant == "original":
                                sys.exit(f"stub backend for the original {dataset_id} {track} {model_id}: {backend.reason}")
                            fits_out.write({**fit, "kept": False, "error": f"stub backend: {backend.reason}"})
                            continue
                        predictions, _ = _predict_with_threshold_policy(
                            model=model,
                            rows=eval_table.rows,
                            instance_id_prefix=f"{dataset_id}-{track}-{model_id}-{variant}",
                            threshold_policy=config.model_threshold_policy,
                        )
                        auc = _evaluate_model_predictions(
                            rows=eval_table.rows,
                            predictions=[prediction for _row, prediction in predictions],
                            target=bundle.target,
                            positive_label=bundle.positive_label,
                            negative_label=bundle.negative_label,
                        )["roc_auc"]
                        if variant == "original":
                            original_auc = auc
                        errors: list[str] = []
                        reference = _cslaw_regb_reference(
                            model=model,
                            negative_label=bundle.negative_label,
                            predictions=predictions,
                            settings=settings,
                            errors=errors,
                        )
                        means = (reference.get("near_cutoff") or {}).get("means") or {}
                        error = "; ".join(errors) or reference.get("error") or ("" if means else "no reference means")
                        if error:
                            if variant == "original":
                                sys.exit(f"reference failed for the original {dataset_id} {track} {model_id}: {error}")
                            fits_out.write({**fit, "roc_auc": auc, "kept": False, "error": error})
                            continue
                        contributions, _bias = model.native_feature_contributions_batch(
                            [eval_table.rows[i] for i in rows_idx]
                        )
                        denied = {i: predictions[i][1].label == bundle.negative_label for i in rows_idx}
                        units = {i: regb.regb_unit_ranking(values, means, K, fixed=fixed) or []
                                 for i, values in zip(rows_idx, contributions)}
                        gap = None if auc is None or original_auc is None else auc - original_auc
                        iteration_mean, iteration_max = ebm_iterations(model)
                        fits_out.write({
                            **fit,
                            "roc_auc": auc,
                            "auc_gap_to_original": gap,
                            "kept": variant == "original" or (gap is not None and abs(gap) <= args.auc_tolerance),
                            "error": "",
                            "denied_share": sum(denied.values()) / len(rows_idx),
                            "reference_rows": reference["near_cutoff"].get("n_rows"),
                            "ebm_best_iteration_mean": iteration_mean,
                            "ebm_best_iteration_max": iteration_max,
                        })
                        for i in rows_idx:
                            reasons_out.write({"dataset_id": dataset_id, "track": track, "model_id": model_id,
                                               "variant": variant, "eval_row_index": i, "denied": int(denied[i]),
                                               "units": "|".join(units[i])})
                        if variant == "original":
                            checks.append(reproduction_check(payload, persisted, fixed, dataset_id, track, model_id,
                                                             rows_idx, means, contributions, units, denied))
    finally:
        fits_out.close()
        reasons_out.close()
        if checks:
            write_csv(out_dir / "within_class_checks.csv", checks, append=append)
    return checks


def reproduction_check(payload, persisted, fixed, dataset_id, track, model_id, rows_idx, means, contributions,
                       units, denied) -> dict:
    """The original fit must reproduce the shared run; stop otherwise."""
    saved = regb.regb_means(payload, dataset_id, track, model_id, "near_cutoff") or {}
    means_gap = max_gap(means, saved) if saved else None
    gaps = [max_gap(values, persisted[(dataset_id, track, model_id, i)])
            for i, values in zip(rows_idx, contributions) if (dataset_id, track, model_id, i) in persisted]
    contribution_gap = max(gaps) if gaps and len(gaps) == len(rows_idx) else None
    published = {i: regb.regb_unit_ranking(persisted[(dataset_id, track, model_id, i)], saved, K, fixed=fixed) or []
                 for i in rows_idx if (dataset_id, track, model_id, i) in persisted and saved}
    mismatches = sum(1 for i, published_units in published.items() if published_units != units[i])
    check = {
        "dataset_id": dataset_id,
        "track": track,
        "model_id": model_id,
        "reference_means_max_gap_vs_saved": means_gap,
        "contributions_max_gap_vs_persisted": contribution_gap,
        "reasons_differing_from_published": mismatches,
        "original_denied_share": sum(denied.values()) / len(rows_idx),
    }
    if means_gap is None or contribution_gap is None:
        sys.exit(f"cannot check the original {dataset_id} {track} {model_id} against the shared run: {check}")
    if means_gap > REPRODUCTION_TOLERANCE or contribution_gap > REPRODUCTION_TOLERANCE:
        sys.exit(f"the original {dataset_id} {track} {model_id} does not reproduce the shared run: {check}")
    if mismatches:
        print(f"[within_class] warning: {mismatches} reason lists differ from the published ones in "
              f"{dataset_id} {track} {model_id} despite matching inputs (ties)", flush=True)
    return check


# --------------------------------------------------------------------- comparisons
def load_outputs(out_dir: Path) -> tuple[list[dict], dict]:
    fits = list(csv.DictReader((out_dir / "within_class_fits.csv").open(encoding="utf-8")))
    reasons: dict = defaultdict(dict)
    for row in csv.DictReader((out_dir / "within_class_reasons.csv").open(encoding="utf-8")):
        key = (row["dataset_id"], row["track"], row["model_id"], row["variant"])
        reasons[key][int(row["eval_row_index"])] = (row["units"].split("|") if row["units"] else [], row["denied"] == "1")
    return fits, reasons


def comparisons(fits: list[dict], reasons: dict, auc_tolerance: float) -> list[dict]:
    kept = {(f["dataset_id"], f["track"], f["model_id"], f["variant"]): float(f["roc_auc"])
            for f in fits if str(f["kept"]) == "True" and f.get("roc_auc") not in (None, "")}
    rows: list[dict] = []

    def compare(kind: str, model: str, label: str, key_a: tuple, key_b: tuple) -> None:
        if key_a not in kept or key_b not in kept or key_a not in reasons or key_b not in reasons:
            return
        for index, (units_a, denied_a) in reasons[key_a].items():
            units_b, denied_b = reasons[key_b][index]
            if not (units_a and units_b):
                continue
            rows.append({
                "kind": kind, "model": model, "label": label, "dataset_id": key_a[0], "track": key_a[1],
                "eval_row_index": index, "variant_a": key_a[3], "variant_b": key_b[3],
                "first_a": units_a[0], "first_b": units_b[0],
                "differs": float(units_a[0] != units_b[0]),
                "overlap": common.overlap_at_k(units_a, units_b, K),
                "either_approves": float(not (denied_a and denied_b)),
                "both_deny": int(denied_a and denied_b),
            })

    variants: dict = defaultdict(list)
    for dataset_id, track, model_id, variant in kept:
        variants[(dataset_id, track, model_id)].append(variant)
    for dataset_id, track in sorted({(d, t) for d, t, _m, _v in kept}):
        for model_id in ("ebm", "xgboost"):
            names = variants.get((dataset_id, track, model_id), [])
            boots = sorted((v for v in names if v.startswith("boot")), key=lambda v: int(v[4:]))
            original = (dataset_id, track, model_id, "original")
            for v in names:
                if v.startswith("seed"):
                    compare("within_seed", model_id, v, original, (dataset_id, track, model_id, v))
            for v in boots:
                compare("within_boot_vs_original", model_id, v, original, (dataset_id, track, model_id, v))
            for v, w in combinations(boots, 2):
                compare("within_boot_pairs", model_id, f"{v}|{w}", (dataset_id, track, model_id, v),
                        (dataset_id, track, model_id, w))
        compare("across_original", "ebm_vs_xgboost", "original", (dataset_id, track, "ebm", "original"),
                (dataset_id, track, "xgboost", "original"))
        shared_boots = set(variants.get((dataset_id, track, "ebm"), [])) & set(variants.get((dataset_id, track, "xgboost"), []))
        for v in sorted((v for v in shared_boots if v.startswith("boot")), key=lambda v: int(v[4:])):
            key_e, key_x = (dataset_id, track, "ebm", v), (dataset_id, track, "xgboost", v)
            if abs(kept[key_e] - kept[key_x]) <= auc_tolerance:
                compare("across_boot", "ebm_vs_xgboost", v, key_e, key_x)
    return rows


# --------------------------------------------------------------------- intervals
class Design:
    """Comparisons of one kind arranged for the two-way bootstrap: cells (applicant, version) by refit labels."""

    def __init__(self, rows: list[dict], field: str, applicants: list[tuple]) -> None:
        self.pairs = any("|" in r["label"] for r in rows)
        labels = sorted({part for r in rows for part in r["label"].split("|")}, key=_label_order)
        self.label_index = {label: i for i, label in enumerate(labels)}
        cells = sorted({(r["dataset_id"], r["track"], r["eval_row_index"]) for r in rows})
        self.cell_index = {cell: i for i, cell in enumerate(cells)}
        self.applicant_of_cell = np.array([applicants.index((d, i)) for d, _t, i in cells], dtype=int)
        n_labels = max(len(labels), 1)
        shape = (len(cells), n_labels, n_labels) if self.pairs else (len(cells), n_labels)
        self.values = np.zeros(shape)
        self.mask = np.zeros(shape)
        for r in rows:
            if r[field] in (None, ""):
                continue
            c = self.cell_index[(r["dataset_id"], r["track"], r["eval_row_index"])]
            parts = [self.label_index[p] for p in r["label"].split("|")]
            where = (c, parts[0], parts[1]) if self.pairs else (c, parts[0])
            self.values[where] = float(r[field])
            self.mask[where] = 1.0
            if self.pairs:
                self.values[c, parts[1], parts[0]] = float(r[field])
                self.mask[c, parts[1], parts[0]] = 1.0
        self.n_labels = n_labels

    def cell_means(self, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Each cell's mean over refit labels under label weights, and whether the cell has any weight."""
        if self.pairs:
            numerator = np.einsum("i,cij,j->c", weights, self.values * self.mask, weights)
            denominator = np.einsum("i,cij,j->c", weights, self.mask, weights)
        else:
            numerator = (self.values * self.mask) @ weights
            denominator = self.mask @ weights
        valid = denominator > 0
        means = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=valid)
        return means, valid.astype(float)

    def applicant_totals(self, weights: np.ndarray, n_applicants: int) -> tuple[np.ndarray, np.ndarray]:
        means, valid = self.cell_means(weights)
        totals = np.bincount(self.applicant_of_cell, weights=means * valid, minlength=n_applicants)
        counts = np.bincount(self.applicant_of_cell, weights=valid, minlength=n_applicants)
        return totals, counts


def _label_order(label: str) -> tuple[str, int]:
    digits = "".join(ch for ch in label if ch.isdigit())
    return label.rstrip("0123456789"), int(digits) if digits else 0


def _rng(seed: str) -> np.random.Generator:
    return np.random.default_rng(random.Random(seed).getrandbits(64))


def two_way(designs: list[Design], applicants: list[tuple], n_boot: int, seed: str) -> tuple[list[float], np.ndarray]:
    """Point estimates and bootstrap replicates of each design's rate on shared applicant draws.

    Each replicate resamples every design's refit labels (independently) and the applicants within dataset (shared
    by all designs, so differences between designs are paired)."""
    n_applicants = len(applicants)
    strata: dict = defaultdict(list)
    for a, (dataset_id, _row) in enumerate(applicants):
        strata[dataset_id].append(a)
    strata_arrays = [np.array(v) for _k, v in sorted(strata.items())]
    points = []
    for design in designs:
        totals, counts = design.applicant_totals(np.ones(design.n_labels), n_applicants)
        points.append(float(totals.sum() / counts.sum()) if counts.sum() else float("nan"))
    rng = _rng(seed)
    replicates = np.full((n_boot, len(designs)), np.nan)
    for b in range(n_boot):
        draws = np.concatenate([s[rng.integers(0, len(s), len(s))] for s in strata_arrays])
        multiplicity = np.bincount(draws, minlength=n_applicants).astype(float)
        for d, design in enumerate(designs):
            weights = np.bincount(rng.integers(0, design.n_labels, design.n_labels), minlength=design.n_labels).astype(float)
            totals, counts = design.applicant_totals(weights, n_applicants)
            denominator = (multiplicity * counts).sum()
            if denominator > 0:
                replicates[b, d] = (multiplicity * totals).sum() / denominator
    return points, replicates


def interval(values: np.ndarray) -> tuple[float, float]:
    clean = np.sort(values[~np.isnan(values)])
    if not len(clean):
        return float("nan"), float("nan")
    return float(clean[int(0.025 * len(clean))]), float(clean[int(0.975 * len(clean)) - 1])


def summarize(rows: list[dict], n_boot: int, seed: int) -> tuple[list[dict], list[dict]]:
    summary, contrasts = [], []
    datasets = [d for d in CREDIT if any(r["dataset_id"] == d for r in rows)]
    metrics = (("differs", "all", "first_reason_differs"), ("overlap", "all", "overlap_of_four"),
               ("differs", "both_deny", "first_reason_differs_both_deny"), ("either_approves", "all", "a_fit_approves"))
    for scope in datasets + ["pooled"]:
        in_scope = [r for r in rows if scope == "pooled" or r["dataset_id"] == scope]
        applicants = sorted({(r["dataset_id"], r["eval_row_index"]) for r in in_scope})
        for field, population, metric in metrics:
            subset = [r for r in in_scope if population == "all" or r["both_deny"]]
            designs, keys = [], []
            for kind, model, _label in KINDS:
                selected = [r for r in subset if r["kind"] == kind and r["model"] == model]
                if selected:
                    designs.append(Design(selected, field, applicants))
                    keys.append((kind, model))
            if not designs:
                continue
            points, replicates = two_way(designs, applicants, n_boot, f"{seed}|{scope}|{metric}")
            for d, (kind, model) in enumerate(keys):
                low, high = interval(replicates[:, d])
                selected = [r for r in subset if r["kind"] == kind and r["model"] == model]
                summary.append({
                    "kind": kind, "model": model, "scope": scope, "metric": metric, "mean": points[d],
                    "ci_low": low, "ci_high": high,
                    "applicants": len({(r["dataset_id"], r["eval_row_index"]) for r in selected}),
                    "cells": len({(r["dataset_id"], r["track"], r["eval_row_index"]) for r in selected}),
                    "comparisons": len(selected),
                })
            if metric == "first_reason_differs":
                for across, within in CONTRASTS:
                    if across in keys and within in keys:
                        a, w = keys.index(across), keys.index(within)
                        low, high = interval(replicates[:, a] - replicates[:, w])
                        contrasts.append({"across": "|".join(across), "within": "|".join(within), "scope": scope,
                                          "difference": points[a] - points[w], "ci_low": low, "ci_high": high})
    return summary, contrasts


# --------------------------------------------------------------------- output
def write_csv(path: Path, rows: list[dict], append: bool = False) -> None:
    if not rows:
        return
    exists = append and path.is_file()
    with path.open("a" if exists else "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary: list[dict], contrasts: list[dict], fits: list[dict], args) -> None:
    def cell(kind: str, model: str, scope: str, metric: str) -> str:
        for row in summary:
            if (row["kind"], row["model"], row["scope"], row["metric"]) == (kind, model, scope, metric):
                if metric == "overlap_of_four":
                    return f"{row['mean']:.2f} [{row['ci_low']:.2f}, {row['ci_high']:.2f}]"
                return f"{100 * row['mean']:.0f}% [{100 * row['ci_low']:.0f}, {100 * row['ci_high']:.0f}] (n={row['cells']})"
        return "n/a"

    scopes = [d for d in CREDIT if any(r["scope"] == d for r in summary)] + ["pooled"]
    pooled_label = f"Pooled ({len(scopes) - 1} datasets)"
    lines = [
        "# First-reason variation within a model class (E4 baseline)",
        "",
        f"Marginal applicant method, four reasons in multiplicity.py units. Refits: {args.boots} bootstrap resamples of "
        f"the training split per class (within outcome class; shared across classes) and {args.seeds} other seeds for "
        f"the boosting machine; a refit is kept if its evaluation ROC AUC is within {args.auc_tolerance} of the original "
        "fit, and an across-class bootstrap pair if both are kept and their ROC AUC differ by at most the same amount. "
        "Each applicant and version counts once, averaged over its refits. Intervals resample refits and applicants "
        "within dataset (versions together). n = applicant-version cells.",
        "",
    ]
    for metric, title in (("first_reason_differs", "Different first reason, every applicant"),
                          ("first_reason_differs_both_deny", "Different first reason, applicants both fits deny"),
                          ("overlap_of_four", "Overlap of the four reasons, every applicant"),
                          ("a_fit_approves", "Comparisons in which a fit approves the applicant")):
        lines += [f"## {title}", "",
                  "| comparison | " + " | ".join(DATASET_LABEL.get(s, pooled_label) for s in scopes) + " |",
                  "|---|" + "---:|" * len(scopes)]
        for kind, model, label in KINDS:
            lines.append(f"| {label} | " + " | ".join(cell(kind, model, s, metric) for s in scopes) + " |")
        lines.append("")
    if contrasts:
        lines += ["## Across-class minus within-class rate (different first reason, paired draws)", "",
                  "| across | within | " + " | ".join(DATASET_LABEL.get(s, pooled_label) for s in scopes) + " |",
                  "|---|---|" + "---:|" * len(scopes)]
        for across, within in CONTRASTS:
            cells = []
            for s in scopes:
                match = [c for c in contrasts if (c["across"], c["within"], c["scope"]) == ("|".join(across), "|".join(within), s)]
                cells.append(f"{100 * match[0]['difference']:+.0f} [{100 * match[0]['ci_low']:+.0f}, "
                             f"{100 * match[0]['ci_high']:+.0f}]" if match else "n/a")
            lines.append(f"| {'|'.join(across)} | {'|'.join(within)} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## Refits kept (ROC AUC within the tolerance of the original fit)", "",
              "| dataset | version | model | seeds kept | bootstraps kept | largest AUC gap | EBM best iteration (mean, max) |",
              "|---|---|---|---:|---:|---:|---:|"]
    by_group: dict = defaultdict(list)
    for f in fits:
        by_group[(f["dataset_id"], f["track"], f["model_id"])].append(f)
    for (dataset_id, track, model_id), items in sorted(by_group.items()):
        def kept(prefix: str, items: list = items) -> str:
            group = [f for f in items if f["variant"].startswith(prefix)]
            return f"{sum(str(f['kept']) == 'True' for f in group)}/{len(group)}" if group else "--"

        gaps = [abs(float(f["auc_gap_to_original"])) for f in items if f.get("auc_gap_to_original") not in (None, "")]
        iterations = [(float(f["ebm_best_iteration_mean"]), float(f["ebm_best_iteration_max"])) for f in items
                      if f.get("ebm_best_iteration_mean") not in (None, "")]
        iteration_text = (f"{np.mean([m for m, _x in iterations]):.0f}, {max(x for _m, x in iterations):.0f}"
                          if iterations else "--")
        lines.append(f"| {DATASET_LABEL.get(dataset_id, dataset_id)} | {track} | {model_id} | {kept('seed')} | "
                     f"{kept('boot')} | {max(gaps) if gaps else 0:.4f} | {iteration_text} |")
    errors = [f for f in fits if f.get("error")]
    if errors:
        lines += ["", f"{len(errors)} fits failed and were not kept; see within_class_fits.csv."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", help="experiment config of the shared credit run")
    parser.add_argument("--shared-bundle", help="bundle of the shared run (applicants explained under every model)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--models", nargs="+", default=["ebm", "xgboost"])
    parser.add_argument("--boots", type=int, default=20, help="bootstrap refits per class, dataset and version")
    parser.add_argument("--seeds", type=int, default=10, help="other random seeds for the boosting machine")
    parser.add_argument("--auc-tolerance", type=float, default=0.02)
    parser.add_argument("--datasets", nargs="+", default=None, help="fit only these datasets and append to the outputs")
    parser.add_argument("--from-reasons", action="store_true", help="skip fitting; redo the comparisons from the CSVs")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.from_reasons:
        if not args.config or not args.shared_bundle:
            parser.error("--config and --shared-bundle are required unless --from-reasons is given")
        fit_all(args, out_dir)
    fits, reasons = load_outputs(out_dir)
    rows = comparisons(fits, reasons, args.auc_tolerance)
    if not rows:
        sys.exit("no comparison could be made; see within_class_fits.csv")
    summary, contrasts = summarize(rows, args.n_boot, args.seed)
    write_csv(out_dir / "within_class_comparisons.csv", rows)
    write_csv(out_dir / "within_class_summary.csv", summary)
    write_csv(out_dir / "within_class_contrasts.csv", contrasts)
    write_markdown(out_dir / "within_class.md", summary, contrasts, fits, args)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
