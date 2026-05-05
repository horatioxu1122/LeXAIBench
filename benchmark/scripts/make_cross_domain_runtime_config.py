from __future__ import annotations

import argparse
import json
from pathlib import Path


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
        default=12,
        help="Adverse instances per dataset-track-model group.",
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
        "--output-root",
        default="benchmark/outputs",
        help="Benchmark output root to embed in config.",
    )
    return parser.parse_args()


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

    payload = json.loads(template_path.read_text(encoding="utf-8"))
    payload["experiment"]["output_root"] = args.output_root
    payload["datasets"]["enabled"] = args.datasets
    payload["datasets"]["denied_instance_sampling"]["max_instances_per_dataset_model_pair"] = (
        args.max_instances
    )
    payload["reporting"]["bootstrap"]["num_resamples"] = args.bootstrap_resamples
    payload["explainers"]["enabled"] = args.explainers
    payload["dataset_sources"]["local_paths"] = {
        "hmda": args.hmda_path,
        "ny_courts_pretrial": args.ny_path,
        "fda_inspections": args.fda_path,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
