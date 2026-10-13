from __future__ import annotations

import argparse
import json
from pathlib import Path

# Defaults of the published non-credit run (cross_domain_strongest_real_full_cluster).
V1_MAX_INSTANCES = 12
# LeXAIBench v2 (--cslaw-v2) budget, the same as benchmark/configs/experiment_cslaw_v2_credit.json.
V2_MAX_INSTANCES = 24
V2_EXPERIMENT_NAME = "legalbench_xai_cslaw_v2_noncredit"
# Same registry paths as benchmark/configs/experiment_cslaw_v2_credit.json.
V2_REGISTRY = {
    "dataset_registry": "benchmark/metadata/dataset_registry_v2.json",
    "feature_taxonomy": "benchmark/metadata/feature_taxonomy_v2.json",
    "paper_feature_taxonomy_manifest": "paper/feature_taxonomy_manifest_v2.yaml",
}
TRACK_CHOICES = ("both", "primary", "audit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a runtime config for the strongest cross-domain benchmark run."
    )
    parser.add_argument(
        "--template",
        default="benchmark/configs/experiment_cross_domain_strongest_real_template.json",
        help="Template config path, relative to repo root unless absolute.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination runtime config path, relative to repo root unless absolute.",
    )
    parser.add_argument(
        "--hmda-path",
        default="benchmark/data/real/hmda.csv",
        help="Prepared HMDA CSV path.",
    )
    parser.add_argument(
        "--ny-path",
        default="benchmark/data/real/ny_courts_pretrial.csv",
        help="Prepared NY Courts pretrial CSV path.",
    )
    parser.add_argument(
        "--fda-path",
        default="benchmark/data/real/fda_inspections.csv",
        help="Prepared FDA inspections CSV path.",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help=(
            "Adverse instances per dataset-track-model group "
            f"(default {V1_MAX_INSTANCES}; {V2_MAX_INSTANCES} with --cslaw-v2)."
        ),
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=1000,
        help="Bootstrap resample count.",
    )
    parser.add_argument(
        "--explainers",
        nargs="+",
        default=["shap", "lime", "anchors", "dice", "wachter"],
        help="Explainer ids to enable.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["hmda", "ny_courts_pretrial", "fda_inspections"],
        help="Dataset ids to enable.",
    )
    parser.add_argument(
        "--tracks",
        choices=TRACK_CHOICES,
        default="both",
        help=(
            "Tracks to run: both (default, the template's setting), primary or audit. "
            "A single track lets one dataset:track shard run as its own cluster job."
        ),
    )
    parser.add_argument(
        "--output-root",
        default="benchmark/outputs",
        help="Benchmark output root to embed in config.",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help=(
            "Override experiment.name (default: the template's name; "
            f"{V2_EXPERIMENT_NAME} with --cslaw-v2)."
        ),
    )
    cslaw = parser.add_mutually_exclusive_group()
    cslaw.add_argument(
        "--cslaw-native",
        action="store_true",
        help=(
            "Round-1 CS&Law instrumentation, superseded by --cslaw-v2: model-native "
            "ranking contributions, persisted canonical explanations and native "
            "contributions, and Reg B reference means, with every v1 scoring and "
            "sampling rule unchanged. Kept so round-1 bundles can be reproduced."
        ),
    )
    cslaw.add_argument(
        "--cslaw-v2",
        action="store_true",
        help=(
            "LeXAIBench v2: the cslaw block of "
            "benchmark/configs/experiment_cslaw_v2_credit.json (corrections v2, persisted "
            "canonical explanations and native contributions, Reg B reference means), the "
            "v2 feature taxonomy and dataset registry, and "
            f"{V2_MAX_INSTANCES} instances per group unless --max-instances is given."
        ),
    )
    return parser.parse_args()


def cslaw_native_block() -> dict:
    """Round-1 cslaw block (--cslaw-native); superseded by cslaw_v2_block()."""
    return {
        "use_native_contributions_for_ranking": True,
        "persist_canonical_explanations": True,
        "persist_native_contributions": True,
        "regb_reference": {
            "enabled": True,
            "near_cutoff_min_rows": 200,
            "near_cutoff_max_band": 0.10,
            "all_applicants_max_rows": 5000,
            "seed": 42,
        },
        "shared_sampling": {
            "enabled": False,
            "scope": "models",
            "margin_aggregate": "min",
        },
        "null_baselines": {
            "faithfulness_reference": "self",
            "num_features": 6,
        },
    }


def cslaw_v2_block() -> dict:
    """The cslaw block of benchmark/configs/experiment_cslaw_v2_credit.json.

    null_baselines.faithfulness_reference stays "self" on purpose: under corrections
    v2 (common_faithfulness_reference) every explainer, null baselines included, is
    scored against the common native reference injected in
    explainers/registry.generate_raw_explanation; the null baseline's own weights
    are kept as faithfulness_evidence.self_reference_signed_contributions.
    """
    return {
        "corrections": "v2",
        "persist_canonical_explanations": True,
        "persist_native_contributions": True,
        "regb_reference": {
            "enabled": True,
            "near_cutoff_min_rows": 200,
            "near_cutoff_max_band": 0.10,
            "all_applicants_max_rows": 5000,
            "seed": 42,
        },
        "shared_sampling": {
            "enabled": False,
            "scope": "models",
            "margin_aggregate": "min",
        },
        "null_baselines": {
            "faithfulness_reference": "self",
            "num_features": 6,
        },
    }


def resolve_path(repo_root: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return repo_root / candidate


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    template_path = resolve_path(repo_root, args.template)
    output_path = resolve_path(repo_root, args.output)

    max_instances = args.max_instances
    if max_instances is None:
        max_instances = V2_MAX_INSTANCES if args.cslaw_v2 else V1_MAX_INSTANCES

    payload = json.loads(template_path.read_text(encoding="utf-8"))
    payload["experiment"]["output_root"] = args.output_root
    payload["datasets"]["enabled"] = args.datasets
    payload["datasets"]["denied_instance_sampling"]["max_instances_per_dataset_model_pair"] = (
        max_instances
    )
    payload["reporting"]["bootstrap"]["num_resamples"] = args.bootstrap_resamples
    payload["explainers"]["enabled"] = args.explainers
    payload["dataset_sources"]["local_paths"] = {
        "hmda": args.hmda_path,
        "ny_courts_pretrial": args.ny_path,
        "fda_inspections": args.fda_path,
    }
    if args.tracks != "both":
        # "both" leaves the template's tracks block untouched (the v1 behavior).
        payload["tracks"]["run_primary_track"] = args.tracks == "primary"
        payload["tracks"]["run_audit_track"] = args.tracks == "audit"
    if args.cslaw_native:
        payload["cslaw"] = cslaw_native_block()
    if args.cslaw_v2:
        payload["cslaw"] = cslaw_v2_block()
        payload["registry"].update(V2_REGISTRY)
        payload["experiment"]["name"] = V2_EXPERIMENT_NAME
        payload["experiment"]["description"] = (
            "LeXAIBench v2 (CS&Law revision) run of the non-credit cross-domain benchmark "
            "(HMDA, NY Courts pretrial, FDA inspections) with cslaw.corrections=v2, the v2 "
            "feature taxonomy and dataset registry, persisted canonical explanations and "
            "native contributions, and Reg B reference means. Built by "
            "make_cross_domain_runtime_config.py --cslaw-v2."
        )
    if args.experiment_name:
        payload["experiment"]["name"] = args.experiment_name

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
