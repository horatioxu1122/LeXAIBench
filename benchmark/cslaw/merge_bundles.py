#!/usr/bin/env python3
"""Merge shard bundles of one experiment into a single results.json.

The non-credit v2 run is split into one cluster job per dataset:track shard
(benchmark/scripts/cluster/run_cslaw_noncredit_v2.sh --shard DATASET:TRACK),
because one sequential run took about 30 h (22.6 h of it fitting EBM on the NY
pretrial data). Every dataset-track group fits its own models and draws its own
sample, so a group's results do not depend on which other groups ran in the
same process. This script puts the shards back together:

    PYTHONPATH=benchmark/src python benchmark/cslaw/merge_bundles.py \
        --inputs DIR [DIR ...] --out-dir DIR [--experiment-name NAME]

Checks (any failure exits non-zero and writes nothing):

* identical experiment settings except datasets and tracks: the shards'
  runtime_config.json files when every shard has one (ignoring experiment
  name/description/output_root, datasets.enabled, tracks and per-dataset local
  paths, which must not conflict); always also the payloads' models, explainers,
  sampling settings (requested budget, v2 selection and seed), the cslaw config,
  the correction options and the provenance metadata (taxonomy and registry
  hashes);
* no (dataset, track, model, explainer) row and no cslaw group appears twice;
* a dataset recorded by two shards' provenance has the same file hash.

Shards from different git commits are merged with a warning (the commits are
kept in payload["cslaw"]["provenance"]["git_commit_by_shard"]), matching the
cluster wrapper, which also only warns; each shard's own runtime-config hash is
kept in ["config_by_shard"].

Output: ``<out-dir>/results.json`` (strict JSON) with the summary rows in the
order a single run produces (dataset in order of first appearance across
--inputs, then primary before audit, then model and explainer in config order),
the payload["cslaw"] groups and provenance merged, payload["overview"]
recomputed with legalbench_xai.runner._build_overview, and payload["merge"]
recording the inputs. When every shard has a runtime_config.json, a merged
``<out-dir>/runtime_config.json`` (all merged datasets and tracks) is written
unless one already exists. The script prints the rebuild-artifacts command to
run next.

An existing ``<out-dir>/results.json`` is never overwritten, except a previous
merge output (it has payload["merge"]) when the cluster wrapper's --force is in
effect (environment CSLAW_FORCE=1), so that a merge can be redone after missing
shards finished.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import rescore  # noqa: E402

TRACK_ORDER = ("primary", "audit")
# Config paths that legitimately differ between shards of one experiment.
IGNORED_CONFIG_PATHS: tuple[tuple[str, ...], ...] = (
    ("experiment", "name"),
    ("experiment", "description"),
    ("experiment", "output_root"),
    ("datasets", "enabled"),
    ("tracks",),
    ("dataset_sources", "local_paths"),
)
# payload["cslaw"] keys every shard must share exactly.
CSLAW_SHARED_KEYS = ("config", "corrections", "corrections_version")
# Provenance keys that may differ between shards: each shard has its own runtime
# config file (different datasets/tracks), and shards may run from different
# commits (warning only, as in the cluster wrapper). Recorded as "<key>_by_shard".
PROVENANCE_PER_SHARD_KEYS = ("config", "git_commit")


class MergeError(ValueError):
    """Raised when shard bundles cannot be merged safely."""


@dataclass
class Shard:
    label: str
    results_path: Path
    payload: dict[str, Any]
    runtime_config: dict[str, Any] | None

    @property
    def directory(self) -> Path:
        return self.results_path.parent


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #


def config_signature(config: dict[str, Any]) -> dict[str, Any]:
    """The runtime config without the keys that may differ between shards."""
    clone = copy.deepcopy(config)
    for path in IGNORED_CONFIG_PATHS:
        node: Any = clone
        for key in path[:-1]:
            node = node.get(key) if isinstance(node, dict) else None
        if isinstance(node, dict):
            node.pop(path[-1], None)
    return clone


def diff_paths(left: Any, right: Any, prefix: str = "") -> list[str]:
    """Dotted paths at which two JSON values differ."""
    if isinstance(left, dict) and isinstance(right, dict):
        paths: list[str] = []
        for key in sorted(set(left) | set(right), key=str):
            if key not in left or key not in right:
                paths.append(f"{prefix}{key}")
            else:
                paths.extend(diff_paths(left[key], right[key], f"{prefix}{key}."))
        return paths
    return [] if left == right else [prefix.rstrip(".") or "(root)"]


def _first_appearance(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _rank_map(order: Sequence[str]) -> dict[str, int]:
    return {value: index for index, value in enumerate(order)}


def _all_configs(shards: Sequence[Shard]) -> bool:
    return bool(shards) and all(shard.runtime_config is not None for shard in shards)


SAMPLING_SETTING_KEYS = ("requested_max_instances", "selection", "seed")


def shard_sampling_settings(payload: dict[str, Any]) -> dict[str, set[Any]]:
    """{setting: non-null values} over the shard's summary-row sampling records.

    Missing values are ignored: a group with fewer adverse rows than the budget
    is recorded as strategy "all_rows" and may not carry the v2 selection/seed.
    """
    settings: dict[str, set[Any]] = {key: set() for key in SAMPLING_SETTING_KEYS}
    for row in payload.get("summary", []) or []:
        sampling = row.get("sampling") or {}
        for key in SAMPLING_SETTING_KEYS:
            value = sampling.get(key)
            if value is not None:
                settings[key].add(value)
    return settings


def check_settings(shards: Sequence[Shard]) -> list[str]:
    """Return problems that make the shards non-mergeable (empty list = OK)."""
    problems: list[str] = []
    reference = shards[0]
    if _all_configs(shards):
        base = config_signature(reference.runtime_config or {})
        for shard in shards[1:]:
            differences = diff_paths(base, config_signature(shard.runtime_config or {}))
            if differences:
                problems.append(
                    f"runtime_config.json settings differ between {reference.label} and "
                    f"{shard.label}: {', '.join(differences[:8])}"
                    + (" ..." if len(differences) > 8 else "")
                )
        paths: dict[str, dict[str, set[str]]] = {}
        for shard in shards:
            config = shard.runtime_config or {}
            enabled = set((config.get("datasets") or {}).get("enabled") or [])
            local = (config.get("dataset_sources") or {}).get("local_paths") or {}
            for dataset_id, path in local.items():
                if dataset_id in enabled:
                    paths.setdefault(dataset_id, {}).setdefault(str(path), set()).add(shard.label)
        for dataset_id, by_path in sorted(paths.items()):
            if len(by_path) > 1:
                problems.append(
                    f"dataset {dataset_id} is read from different files by different shards: "
                    + "; ".join(
                        f"{path} ({', '.join(sorted(labels))})" for path, labels in by_path.items()
                    )
                )
    elif any(shard.runtime_config is not None for shard in shards):
        missing = [shard.label for shard in shards if shard.runtime_config is None]
        problems.append(
            f"only some shards have a runtime_config.json (missing: {', '.join(missing)}); "
            "cannot verify that they ran with the same settings"
        )

    def summary_set(payload: dict[str, Any], key: str) -> set[str]:
        return {str(row[key]) for row in payload.get("summary", []) or []}

    for key, label in (("model_id", "models"), ("explainer_id", "explainers")):
        base_set = summary_set(reference.payload, key)
        for shard in shards[1:]:
            other = summary_set(shard.payload, key)
            if other != base_set:
                problems.append(
                    f"{label} differ between {reference.label} {sorted(base_set)} and "
                    f"{shard.label} {sorted(other)}"
                )
    per_shard = [(shard.label, shard_sampling_settings(shard.payload)) for shard in shards]
    for key in SAMPLING_SETTING_KEYS:
        union = {value for _label, settings in per_shard for value in settings[key]}
        if len(union) > 1:
            problems.append(
                f"sampling setting '{key}' differs between shards: "
                + "; ".join(
                    f"{label}={sorted(settings[key], key=repr)}" for label, settings in per_shard
                )
            )
    return problems


def summary_order(
    shards: Sequence[Shard],
) -> tuple[list[str], list[str], list[str], list[str]]:
    """(datasets, tracks, models, explainers) in single-run order."""
    rows = [row for shard in shards for row in shard.payload.get("summary", []) or []]
    datasets = _first_appearance([str(row["dataset_id"]) for row in rows])
    present_tracks = {str(row["track"]) for row in rows}
    tracks = [track for track in TRACK_ORDER if track in present_tracks]
    tracks += sorted(present_tracks - set(TRACK_ORDER))
    config = (shards[0].runtime_config or {}) if _all_configs(shards) else {}
    configured_models = [str(item) for item in (config.get("models") or {}).get("enabled") or []]
    configured_explainers = [
        str(item) for item in (config.get("explainers") or {}).get("enabled") or []
    ]
    present_models = {str(row["model_id"]) for row in rows}
    present_explainers = {str(row["explainer_id"]) for row in rows}
    models = [
        model
        for model in _first_appearance(configured_models + [str(row["model_id"]) for row in rows])
        if model in present_models
    ]
    explainers = [
        explainer
        for explainer in _first_appearance(
            configured_explainers + [str(row["explainer_id"]) for row in rows]
        )
        if explainer in present_explainers
    ]
    return datasets, tracks, models, explainers


def duplicate_rows(shards: Sequence[Shard]) -> list[str]:
    """Problems for (dataset, track, model, explainer) rows present more than once."""
    seen: dict[tuple[str, str, str, str], str] = {}
    duplicates: list[str] = []
    for shard in shards:
        for row in shard.payload.get("summary", []) or []:
            key = common.group_key(row)
            if key in seen:
                duplicates.append(f"{'|'.join(key)} ({seen[key]} and {shard.label})")
            else:
                seen[key] = shard.label
    if not duplicates:
        return []
    return [
        f"duplicate (dataset, track, model, explainer) rows in {len(duplicates)} case(s): "
        + "; ".join(duplicates[:6])
        + (" ..." if len(duplicates) > 6 else "")
    ]


def merge_summary(shards: Sequence[Shard]) -> list[dict[str, Any]]:
    """Concatenate summary rows in single-run order; refuse duplicate groups."""
    problems = duplicate_rows(shards)
    if problems:
        raise MergeError(problems[0])
    rows = [row for shard in shards for row in shard.payload.get("summary", []) or []]
    datasets, tracks, models, explainers = summary_order(shards)
    ranks = (_rank_map(datasets), _rank_map(tracks), _rank_map(models), _rank_map(explainers))
    return sorted(
        rows,
        key=lambda row: tuple(
            rank[str(value)]
            for rank, value in zip(
                ranks,
                (row["dataset_id"], row["track"], row["model_id"], row["explainer_id"]),
                strict=True,
            )
        ),
    )


def merge_provenance(
    shards: Sequence[Shard],
) -> tuple[dict[str, Any] | None, list[str], list[str]]:
    """(merged provenance, problems, warnings) from the shards' provenance blocks."""
    blocks = [(shard.label, common.bundle_provenance(shard.payload)) for shard in shards]
    present = [(label, block) for label, block in blocks if block]
    if not present:
        return None, [], []
    problems: list[str] = []
    warnings: list[str] = []
    if len(present) != len(blocks):
        missing = [label for label, block in blocks if not block]
        problems.append(f"provenance recorded by some shards only (missing: {', '.join(missing)})")
    merged: dict[str, Any] = {"datasets": {}}
    dataset_owner: dict[str, str] = {}
    for label, block in present:
        for dataset_id, entry in (block.get("datasets") or {}).items():
            if dataset_id in merged["datasets"]:
                if merged["datasets"][dataset_id] != entry:
                    problems.append(
                        f"provenance of dataset {dataset_id} differs between "
                        f"{dataset_owner[dataset_id]} and {label} (different data file)"
                    )
                continue
            merged["datasets"][dataset_id] = entry
            dataset_owner[dataset_id] = label
    for key in sorted({key for _label, block in present for key in block} - {"datasets"}):
        values = [(label, block.get(key)) for label, block in present]
        if all(value == values[0][1] for _label, value in values):
            merged[key] = values[0][1]
            continue
        if key in PROVENANCE_PER_SHARD_KEYS:
            merged[key] = None
            merged[f"{key}_by_shard"] = {label: value for label, value in values}
            if key == "git_commit":
                warnings.append(
                    "shards ran different code versions (git_commit: "
                    + "; ".join(f"{label}={value}" for label, value in values)
                    + "); recorded in payload['cslaw']['provenance']['git_commit_by_shard']"
                )
        else:
            problems.append(
                f"provenance '{key}' differs between shards ("
                + "; ".join(f"{label}: {str(value)[:80]}" for label, value in values)
                + "); shards must use the same taxonomy and registry files"
            )
    return merged, problems, warnings


def merge_cslaw(
    shards: Sequence[Shard],
) -> tuple[dict[str, Any] | None, list[str], list[str]]:
    """(merged payload["cslaw"], problems, warnings): shared keys must agree, groups union."""
    blocks = [(shard.label, shard.payload.get("cslaw")) for shard in shards]
    present = [(label, block) for label, block in blocks if isinstance(block, dict)]
    if not present:
        return None, [], []
    problems: list[str] = []
    warnings: list[str] = []
    if len(present) != len(blocks):
        missing = [label for label, block in blocks if not isinstance(block, dict)]
        problems.append(
            "payload['cslaw'] present in some shards only (missing: "
            f"{', '.join(missing)}); shards must use the same cslaw block"
        )
    merged: dict[str, Any] = {}
    keys = {key for _label, block in present for key in block}
    for key in sorted(keys - {"groups", "provenance"}):
        values = [(label, block.get(key)) for label, block in present]
        if all(value == values[0][1] for _label, value in values):
            if values[0][1] is not None:
                merged[key] = values[0][1]
            continue
        detail = ", ".join(
            f"{label} differs at {', '.join(diff_paths(values[0][1], value)[:4]) or '(root)'}"
            for label, value in values[1:]
            if value != values[0][1]
        )
        if key in CSLAW_SHARED_KEYS:
            problems.append(f"payload['cslaw']['{key}'] differs between shards: {detail}")
        else:
            merged[f"{key}_by_shard"] = {label: value for label, value in values}
            warnings.append(
                f"payload['cslaw']['{key}'] differs between shards ({detail}); kept per shard"
            )
    groups: dict[str, Any] = {}
    owner: dict[str, str] = {}
    for label, block in present:
        for key, group in (block.get("groups") or {}).items():
            if key in groups:
                problems.append(f"duplicate cslaw group {key} ({owner[key]} and {label})")
                continue
            groups[key] = group
            owner[key] = label
    merged["groups"] = groups
    provenance, provenance_problems, provenance_warnings = merge_provenance(shards)
    problems.extend(provenance_problems)
    warnings.extend(provenance_warnings)
    if provenance is not None:
        merged["provenance"] = provenance
    return merged, problems, warnings


def merge_payloads(
    shards: Sequence[Shard],
    *,
    overview_builder: Callable[[list[dict[str, Any]]], dict[str, Any]],
    experiment_name: str | None = None,
    merged_at: str | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Return (merged payload, warnings). Raises MergeError on any conflict."""
    if not shards:
        raise MergeError("no input bundles")
    problems = duplicate_rows(shards) + check_settings(shards)
    cslaw, cslaw_problems, warnings = merge_cslaw(shards)
    problems.extend(cslaw_problems)
    if problems:
        raise MergeError("shards are not mergeable:\n  - " + "\n  - ".join(problems))
    summary = merge_summary(shards)

    names = _first_appearance([str(shard.payload.get("experiment_name", "")) for shard in shards])
    if experiment_name:
        name = experiment_name
    elif len(names) == 1:
        name = names[0]
    else:
        name = f"{names[0]}__merged"
        warnings.append(
            f"shards have different experiment names ({', '.join(names)}); using {name!r} "
            "(pass --experiment-name to choose)"
        )
    capabilities = [shard.payload.get("capabilities") for shard in shards]
    capabilities_differ = any(value != capabilities[0] for value in capabilities[1:])
    if capabilities_differ:
        warnings.append(
            "shards report different runtime capabilities (library availability); see "
            "payload['merge']['capabilities_by_shard'] and check the backends per row"
        )
    datasets, tracks, models, explainers = summary_order(shards)
    payload: dict[str, Any] = {
        "experiment_name": name,
        "generated_at_utc": merged_at or datetime.now(timezone.utc).isoformat(),
        "capabilities": capabilities[0] if capabilities[0] is not None else {},
        "overview": overview_builder(summary),
        "summary": summary,
    }
    if cslaw is not None:
        payload["cslaw"] = cslaw
    payload["merge"] = {
        "inputs": [
            {
                "label": shard.label,
                "results_json": str(shard.results_path),
                "experiment_name": shard.payload.get("experiment_name"),
                "generated_at_utc": shard.payload.get("generated_at_utc"),
                "summary_rows": len(shard.payload.get("summary", []) or []),
                "groups": sorted(
                    {
                        f"{row['dataset_id']}|{row['track']}"
                        for row in shard.payload.get("summary", []) or []
                    }
                ),
                "runtime_config": shard.runtime_config is not None,
            }
            for shard in shards
        ],
        "order": {
            "datasets": datasets,
            "tracks": tracks,
            "models": models,
            "explainers": explainers,
        },
        "settings_checked": (
            "runtime_config.json (except experiment name/description/output_root, "
            "datasets.enabled, tracks, local paths) + payload models/explainers/sampling/cslaw"
            if _all_configs(shards)
            else "payload models/explainers/sampling/cslaw (no runtime_config.json)"
        ),
        "warnings": list(warnings),
        "note": (
            "Summary rows are unchanged copies of the shard rows; only their order, the "
            "overview and payload['cslaw'] were rebuilt. Run rebuild-artifacts on this file "
            "to regenerate the analysis artifacts."
        ),
    }
    if capabilities_differ:
        payload["merge"]["capabilities_by_shard"] = {
            shard.label: value for shard, value in zip(shards, capabilities, strict=True)
        }
    return payload, warnings


def merged_runtime_config(
    shards: Sequence[Shard],
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    """The first shard's runtime config with every merged dataset and track enabled."""
    if not _all_configs(shards):
        return None
    config = copy.deepcopy(shards[0].runtime_config or {})
    order = payload["merge"]["order"]
    config.setdefault("experiment", {})["name"] = payload["experiment_name"]
    config.setdefault("datasets", {})["enabled"] = list(order["datasets"])
    config["tracks"] = {
        **(config.get("tracks") or {}),
        "run_primary_track": "primary" in order["tracks"],
        "run_audit_track": "audit" in order["tracks"],
    }
    local: dict[str, Any] = {}
    for shard in shards:
        paths = ((shard.runtime_config or {}).get("dataset_sources") or {}).get("local_paths") or {}
        for dataset_id, path in paths.items():
            local.setdefault(dataset_id, path)
    if local:
        config.setdefault("dataset_sources", {})["local_paths"] = local
    return config


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #


def load_shard(path_text: str, labels: Counter) -> Shard:
    results_path = rescore.resolve_results_path(path_text)
    label = results_path.parent.name
    labels[label] += 1
    if labels[label] > 1:
        label = f"{label}#{labels[label]}"
    runtime_path = results_path.parent / "runtime_config.json"
    runtime = common.load_results(runtime_path) if runtime_path.is_file() else None
    return Shard(
        label=label,
        results_path=results_path,
        payload=common.load_results(results_path),
        runtime_config=runtime,
    )


def refuse_existing_output(target: Path) -> None:
    """Never overwrite results.json, except a previous merge under CSLAW_FORCE=1."""
    if not target.exists():
        return
    force = os.environ.get("CSLAW_FORCE", "0").strip() == "1"
    try:
        previous_merge = "merge" in common.load_results(target)
    except Exception:  # noqa: BLE001 - an unreadable file is never overwritten
        previous_merge = False
    if force and previous_merge:
        print(
            f"merge_bundles: WARNING - replacing the previous merge output {target} "
            "(CSLAW_FORCE=1)",
            file=sys.stderr,
        )
        return
    raise SystemExit(
        f"merge_bundles: {target} already exists; refusing to overwrite it. Choose a new "
        "--out-dir"
        + (
            " (it is not a previous merge output)."
            if force
            else " (the cluster wrapper's --force, CSLAW_FORCE=1, allows replacing a previous "
            "merge output)."
        )
    )


def follow_up_command(config: str, out_dir: Path) -> str:
    return (
        "PYTHONPATH=benchmark/src python -m legalbench_xai rebuild-artifacts \\\n"
        f"  --config {config} \\\n"
        f"  --results-json {out_dir / 'results.json'} \\\n"
        f"  --output-dir {out_dir}"
    )


def run(
    inputs: Sequence[str | Path],
    out_dir: str | Path,
    *,
    experiment_name: str | None = None,
) -> list[Path]:
    if not inputs:
        raise SystemExit("merge_bundles: --inputs needs at least one bundle directory")
    labels: Counter = Counter()
    shards = [load_shard(str(item), labels) for item in inputs]
    resolved = [shard.results_path for shard in shards]
    if len(set(resolved)) != len(resolved):
        raise SystemExit(
            "merge_bundles: duplicate input: the same bundle is listed twice in --inputs"
        )
    out = Path(out_dir).expanduser().resolve()
    if any(out == shard.directory for shard in shards):
        raise SystemExit(f"merge_bundles: --out-dir {out} is one of the input bundles")
    target = out / "results.json"
    refuse_existing_output(target)
    _mean, overview_builder = rescore.import_runner_helpers()
    try:
        payload, warnings = merge_payloads(
            shards, overview_builder=overview_builder, experiment_name=experiment_name
        )
    except MergeError as error:
        raise SystemExit(f"merge_bundles: {error}") from error
    payload, replaced = rescore.sanitize_json(payload)
    if replaced:
        payload["merge"]["non_finite_values_replaced_with_null"] = replaced
    out.mkdir(parents=True, exist_ok=True)
    common.write_json(target, payload)
    written = [target]

    runtime_target = out / "runtime_config.json"
    config_text = "<CONFIG_WITH_ALL_MERGED_DATASETS_AND_TRACKS>"
    merged_config = merged_runtime_config(shards, payload)
    if runtime_target.is_file():
        config_text = str(runtime_target)
        if merged_config is not None:
            existing = common.load_results(runtime_target)
            differences = diff_paths(config_signature(merged_config), config_signature(existing))
            if differences:
                warnings.append(
                    f"the existing {runtime_target} differs from the shards' settings at "
                    f"{', '.join(differences[:6])}; check it before rebuild-artifacts"
                )
    elif merged_config is not None:
        common.write_json(runtime_target, merged_config)
        written.append(runtime_target)
        config_text = str(runtime_target)
    else:
        warnings.append(
            "no runtime_config.json in the shards; pass the config that lists every merged "
            "dataset and track to rebuild-artifacts"
        )

    for warning in warnings:
        print(f"merge_bundles: WARNING - {warning}", file=sys.stderr)
    order = payload["merge"]["order"]
    print(
        f"Merged {len(shards)} shard(s) into {len(payload['summary'])} summary rows "
        f"(datasets: {', '.join(order['datasets'])}; tracks: {', '.join(order['tracks'])})."
    )
    print("Wrote:")
    for path in written:
        print(f"  {path}")
    print("Next, rebuild the analysis artifacts from the repository root:")
    print(follow_up_command(config_text, out))
    return written


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge shard bundles (e.g. one per dataset:track) of one experiment into a single "
            "results.json; refuses mismatched settings and duplicate groups."
        )
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        metavar="DIR",
        help="Shard bundle directories (or results.json files), in the dataset order to keep.",
    )
    parser.add_argument("--out-dir", required=True, metavar="DIR", help="New output directory.")
    parser.add_argument(
        "--experiment-name",
        metavar="NAME",
        help="Experiment name of the merged bundle (default: the shards' common name).",
    )
    args = parser.parse_args(argv)
    run(args.inputs, args.out_dir, experiment_name=args.experiment_name)


if __name__ == "__main__":
    main()
