import argparse
import json

from .config import load_experiment_config
from .pipeline import BenchmarkPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="legalbench_xai")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument(
        "--config",
        default="benchmark/configs/experiment_default.json",
        help="Path to the benchmark experiment config JSON, relative to repo root.",
    )

    prepare_south_german_parser = subparsers.add_parser("prepare-south-german-credit")
    prepare_south_german_parser.add_argument(
        "--config",
        default="benchmark/configs/experiment_default.json",
        help="Path to the benchmark experiment config JSON, relative to repo root.",
    )
    prepare_south_german_parser.add_argument(
        "--input-asc",
        default="benchmark/data/raw/south_german_credit_uci/SouthGermanCredit.asc",
        help="Path to the raw South German Credit .asc file, relative to repo root unless absolute.",
    )
    prepare_south_german_parser.add_argument(
        "--output-csv",
        default="benchmark/data/real/south_german_credit.csv",
        help="Destination CSV path, relative to repo root unless absolute.",
    )

    prepare_default_credit_parser = subparsers.add_parser("prepare-default-credit")
    prepare_default_credit_parser.add_argument(
        "--config",
        default="benchmark/configs/experiment_default.json",
        help="Path to the benchmark experiment config JSON, relative to repo root.",
    )
    prepare_default_credit_parser.add_argument(
        "--input-xls",
        default="benchmark/data/raw/default_credit_uci/default of credit card clients.xls",
        help="Path to the raw Default Credit .xls file, relative to repo root unless absolute.",
    )
    prepare_default_credit_parser.add_argument(
        "--output-csv",
        default="benchmark/data/real/default_credit.csv",
        help="Destination CSV path, relative to repo root unless absolute.",
    )

    prepare_heloc_parser = subparsers.add_parser("prepare-heloc")
    prepare_heloc_parser.add_argument(
        "--config",
        default="benchmark/configs/experiment_default.json",
        help="Path to the benchmark experiment config JSON, relative to repo root.",
    )
    prepare_heloc_parser.add_argument(
        "--output-csv",
        default="benchmark/data/real/heloc.csv",
        help="Destination CSV path, relative to repo root unless absolute.",
    )
    prepare_heloc_parser.add_argument(
        "--openml-data-id",
        type=int,
        default=45554,
        help="OpenML data id for the cleaned HELOC mirror.",
    )

    prepare_gmsc_parser = subparsers.add_parser("prepare-give-me-some-credit")
    prepare_gmsc_parser.add_argument(
        "--config",
        default="benchmark/configs/experiment_default.json",
        help="Path to the benchmark experiment config JSON, relative to repo root.",
    )
    prepare_gmsc_parser.add_argument(
        "--output-csv",
        default="benchmark/data/real/give_me_some_credit.csv",
        help="Destination CSV path, relative to repo root unless absolute.",
    )
    prepare_gmsc_parser.add_argument(
        "--openml-data-id",
        type=int,
        default=46929,
        help="OpenML data id for the Give Me Some Credit mirror.",
    )

    prepare_hmda_parser = subparsers.add_parser("prepare-hmda")
    prepare_hmda_parser.add_argument(
        "--config",
        default="benchmark/configs/experiment_default.json",
        help="Path to the benchmark experiment config JSON, relative to repo root.",
    )
    prepare_hmda_parser.add_argument(
        "--input-csv",
        default="benchmark/data/raw/hmda/hmda_snapshot.csv",
        help="Path to a local HMDA raw CSV export, relative to repo root unless absolute.",
    )
    prepare_hmda_parser.add_argument(
        "--output-csv",
        default="benchmark/data/real/hmda.csv",
        help="Destination CSV path, relative to repo root unless absolute.",
    )

    prepare_ny_courts_parser = subparsers.add_parser("prepare-ny-courts-pretrial")
    prepare_ny_courts_parser.add_argument(
        "--config",
        default="benchmark/configs/experiment_default.json",
        help="Path to the benchmark experiment config JSON, relative to repo root.",
    )
    prepare_ny_courts_parser.add_argument(
        "--input-csv",
        default="benchmark/data/raw/ny_courts_pretrial/pretrial_release_rows.csv",
        help="Path to a local NY Courts pretrial CSV extract, relative to repo root unless absolute.",
    )
    prepare_ny_courts_parser.add_argument(
        "--output-csv",
        default="benchmark/data/real/ny_courts_pretrial.csv",
        help="Destination CSV path, relative to repo root unless absolute.",
    )

    prepare_ipums_parser = subparsers.add_parser("prepare-ipums-cps-ui")
    prepare_ipums_parser.add_argument(
        "--config",
        default="benchmark/configs/experiment_default.json",
        help="Path to the benchmark experiment config JSON, relative to repo root.",
    )
    prepare_ipums_parser.add_argument(
        "--input-csv",
        default="benchmark/data/raw/ipums_cps_ui/ipums_cps_ui_extract.csv",
        help="Path to a local IPUMS CPS UI CSV extract, relative to repo root unless absolute.",
    )
    prepare_ipums_parser.add_argument(
        "--output-csv",
        default="benchmark/data/real/ipums_cps_ui.csv",
        help="Destination CSV path, relative to repo root unless absolute.",
    )

    prepare_nhamcs_parser = subparsers.add_parser("prepare-nhamcs")
    prepare_nhamcs_parser.add_argument(
        "--config",
        default="benchmark/configs/experiment_default.json",
        help="Path to the benchmark experiment config JSON, relative to repo root.",
    )
    prepare_nhamcs_parser.add_argument(
        "--input-csv",
        default="benchmark/data/raw/nhamcs/nhamcs_ed_extract.csv",
        help="Path to a local NHAMCS emergency-department CSV extract, relative to repo root unless absolute.",
    )
    prepare_nhamcs_parser.add_argument(
        "--output-csv",
        default="benchmark/data/real/nhamcs.csv",
        help="Destination CSV path, relative to repo root unless absolute.",
    )

    prepare_fda_parser = subparsers.add_parser("prepare-fda-inspections")
    prepare_fda_parser.add_argument(
        "--config",
        default="benchmark/configs/experiment_default.json",
        help="Path to the benchmark experiment config JSON, relative to repo root.",
    )
    prepare_fda_parser.add_argument(
        "--input-csv",
        default="benchmark/data/raw/fda_inspections/fda_inspections_extract.csv",
        help="Path to a local FDA inspections CSV export, relative to repo root unless absolute.",
    )
    prepare_fda_parser.add_argument(
        "--output-csv",
        default="benchmark/data/real/fda_inspections.csv",
        help="Destination CSV path, relative to repo root unless absolute.",
    )

    describe_parser = subparsers.add_parser("describe-dataset")
    describe_parser.add_argument(
        "--config",
        default="benchmark/configs/experiment_default.json",
        help="Path to the benchmark experiment config JSON, relative to repo root.",
    )
    describe_parser.add_argument("--dataset", required=True, help="Dataset id to describe.")
    describe_parser.add_argument(
        "--track",
        default="primary",
        choices=["primary", "audit"],
        help="Which dataset track to describe.",
    )

    preview_parser = subparsers.add_parser("preview-dataset")
    preview_parser.add_argument(
        "--config",
        default="benchmark/configs/experiment_default.json",
        help="Path to the benchmark experiment config JSON, relative to repo root.",
    )
    preview_parser.add_argument("--dataset", required=True, help="Dataset id to preview.")
    preview_parser.add_argument(
        "--track",
        default="primary",
        choices=["primary", "audit"],
        help="Which dataset track to preview.",
    )
    preview_parser.add_argument(
        "--path",
        default=None,
        help="Optional custom CSV path, relative to repo root.",
    )

    check_parser = subparsers.add_parser("check-dataset-file")
    check_parser.add_argument(
        "--config",
        default="benchmark/configs/experiment_default.json",
        help="Path to the benchmark experiment config JSON, relative to repo root.",
    )
    check_parser.add_argument("--dataset", required=True, help="Dataset id to validate.")
    check_parser.add_argument(
        "--track",
        default="primary",
        choices=["primary", "audit"],
        help="Which dataset track to validate against.",
    )
    check_parser.add_argument(
        "--path",
        required=True,
        help="CSV path to validate, relative to repo root unless absolute.",
    )

    normalize_parser = subparsers.add_parser("normalize-example")
    normalize_parser.add_argument(
        "--config",
        default="benchmark/configs/experiment_default.json",
        help="Path to the benchmark experiment config JSON, relative to repo root.",
    )
    normalize_parser.add_argument("--dataset", required=True, help="Dataset id to normalize.")
    normalize_parser.add_argument(
        "--track",
        default="primary",
        choices=["primary", "audit"],
        help="Which dataset track to use.",
    )

    adapt_parser = subparsers.add_parser("adapt-example")
    adapt_parser.add_argument(
        "--config",
        default="benchmark/configs/experiment_default.json",
        help="Path to the benchmark experiment config JSON, relative to repo root.",
    )
    adapt_parser.add_argument("--dataset", required=True, help="Dataset id to adapt.")
    adapt_parser.add_argument(
        "--track",
        default="primary",
        choices=["primary", "audit"],
        help="Which dataset track to use.",
    )
    adapt_parser.add_argument(
        "--explainer",
        required=True,
        choices=["shap", "lime", "leave_one_out", "anchors", "dice", "wachter"],
        help="Explainer family to adapt.",
    )

    score_parser = subparsers.add_parser("score-example")
    score_parser.add_argument(
        "--config",
        default="benchmark/configs/experiment_default.json",
        help="Path to the benchmark experiment config JSON, relative to repo root.",
    )
    score_parser.add_argument("--dataset", required=True, help="Dataset id to score.")
    score_parser.add_argument(
        "--track",
        default="primary",
        choices=["primary", "audit"],
        help="Which dataset track to use.",
    )

    score_adapted_parser = subparsers.add_parser("score-adapted-example")
    score_adapted_parser.add_argument(
        "--config",
        default="benchmark/configs/experiment_default.json",
        help="Path to the benchmark experiment config JSON, relative to repo root.",
    )
    score_adapted_parser.add_argument("--dataset", required=True, help="Dataset id to score.")
    score_adapted_parser.add_argument(
        "--track",
        default="primary",
        choices=["primary", "audit"],
        help="Which dataset track to use.",
    )
    score_adapted_parser.add_argument(
        "--explainer",
        required=True,
        choices=["shap", "lime", "leave_one_out", "anchors", "dice", "wachter"],
        help="Explainer family to adapt and score.",
    )

    batch_parser = subparsers.add_parser("run-demo-benchmark")
    batch_parser.add_argument(
        "--config",
        default="benchmark/configs/experiment_default.json",
        help="Path to the benchmark experiment config JSON, relative to repo root.",
    )
    batch_parser.add_argument(
        "--write",
        action="store_true",
        help="Write results.json, summary.csv, and report.md into the configured output root.",
    )
    batch_parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory override, relative to repo root unless absolute.",
    )
    batch_parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print only the overview and artifact manifest instead of the full summary rows.",
    )

    rebuild_parser = subparsers.add_parser("rebuild-artifacts")
    rebuild_parser.add_argument(
        "--config",
        default="benchmark/configs/experiment_default.json",
        help="Path to the benchmark experiment config JSON, relative to repo root.",
    )
    rebuild_parser.add_argument(
        "--results-json",
        required=True,
        help="Path to an existing results.json payload, relative to repo root unless absolute.",
    )
    rebuild_parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory override, relative to repo root unless absolute.",
    )

    capabilities_parser = subparsers.add_parser("show-capabilities")
    capabilities_parser.add_argument(
        "--config",
        default="benchmark/configs/experiment_default.json",
        help="Path to the benchmark experiment config JSON, relative to repo root.",
    )

    data_sources_parser = subparsers.add_parser("show-data-sources")
    data_sources_parser.add_argument(
        "--config",
        default="benchmark/configs/experiment_default.json",
        help="Path to the benchmark experiment config JSON, relative to repo root.",
    )

    doctor_parser = subparsers.add_parser("doctor")
    doctor_parser.add_argument(
        "--config",
        default="benchmark/configs/experiment_default.json",
        help="Path to the benchmark experiment config JSON, relative to repo root.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "validate":
        config = load_experiment_config(args.config)
        result = BenchmarkPipeline(config).validate()
        for message in result.messages:
            print(message)
        return 0 if result.ok else 1

    if args.command == "prepare-south-german-credit":
        config = load_experiment_config(args.config)
        payload = BenchmarkPipeline(config).prepare_south_german_credit(
            input_path=args.input_asc,
            output_path=args.output_csv,
        )
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "prepare-default-credit":
        config = load_experiment_config(args.config)
        payload = BenchmarkPipeline(config).prepare_default_credit(
            input_path=args.input_xls,
            output_path=args.output_csv,
        )
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "prepare-heloc":
        config = load_experiment_config(args.config)
        payload = BenchmarkPipeline(config).prepare_heloc(
            output_path=args.output_csv,
            openml_data_id=args.openml_data_id,
        )
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "prepare-give-me-some-credit":
        config = load_experiment_config(args.config)
        payload = BenchmarkPipeline(config).prepare_give_me_some_credit(
            output_path=args.output_csv,
            openml_data_id=args.openml_data_id,
        )
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "prepare-hmda":
        config = load_experiment_config(args.config)
        payload = BenchmarkPipeline(config).prepare_hmda(
            input_path=args.input_csv,
            output_path=args.output_csv,
        )
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "prepare-ny-courts-pretrial":
        config = load_experiment_config(args.config)
        payload = BenchmarkPipeline(config).prepare_ny_courts_pretrial(
            input_path=args.input_csv,
            output_path=args.output_csv,
        )
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "prepare-ipums-cps-ui":
        config = load_experiment_config(args.config)
        payload = BenchmarkPipeline(config).prepare_ipums_cps_ui(
            input_path=args.input_csv,
            output_path=args.output_csv,
        )
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "prepare-nhamcs":
        config = load_experiment_config(args.config)
        payload = BenchmarkPipeline(config).prepare_nhamcs(
            input_path=args.input_csv,
            output_path=args.output_csv,
        )
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "prepare-fda-inspections":
        config = load_experiment_config(args.config)
        payload = BenchmarkPipeline(config).prepare_fda_inspections(
            input_path=args.input_csv,
            output_path=args.output_csv,
        )
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "describe-dataset":
        config = load_experiment_config(args.config)
        for message in BenchmarkPipeline(config).describe_dataset(
            dataset_id=args.dataset,
            track=args.track,
        ):
            print(message)
        return 0

    if args.command == "preview-dataset":
        config = load_experiment_config(args.config)
        payload = BenchmarkPipeline(config).preview_dataset(
            dataset_id=args.dataset,
            track=args.track,
            path=args.path,
        )
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "check-dataset-file":
        config = load_experiment_config(args.config)
        payload = BenchmarkPipeline(config).check_dataset_file(
            dataset_id=args.dataset,
            track=args.track,
            path=args.path,
        )
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "normalize-example":
        config = load_experiment_config(args.config)
        payload = BenchmarkPipeline(config).normalize_example(
            dataset_id=args.dataset,
            track=args.track,
        )
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "adapt-example":
        config = load_experiment_config(args.config)
        payload = BenchmarkPipeline(config).adapt_example(
            dataset_id=args.dataset,
            track=args.track,
            explainer_id=args.explainer,
        )
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "score-example":
        config = load_experiment_config(args.config)
        payload = BenchmarkPipeline(config).score_example(
            dataset_id=args.dataset,
            track=args.track,
        )
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "score-adapted-example":
        config = load_experiment_config(args.config)
        payload = BenchmarkPipeline(config).score_adapted_example(
            dataset_id=args.dataset,
            track=args.track,
            explainer_id=args.explainer,
        )
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "run-demo-benchmark":
        config = load_experiment_config(args.config)
        payload = BenchmarkPipeline(config).run_demo_benchmark(
            write_outputs=args.write,
            output_dir=args.output_dir,
            summary_only=args.summary_only,
        )
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "show-capabilities":
        config = load_experiment_config(args.config)
        payload = BenchmarkPipeline(config).show_capabilities()
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "rebuild-artifacts":
        config = load_experiment_config(args.config)
        payload = BenchmarkPipeline(config).rebuild_artifacts_from_results(
            results_path=args.results_json,
            output_dir=args.output_dir,
        )
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "show-data-sources":
        config = load_experiment_config(args.config)
        payload = BenchmarkPipeline(config).show_data_sources()
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "doctor":
        config = load_experiment_config(args.config)
        for message in BenchmarkPipeline(config).doctor():
            print(message)
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2
