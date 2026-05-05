import json
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

from .adapters import adapt_raw_example
from .adapters.examples import sample_raw_payload
from .capabilities import detect_capabilities, empirical_readiness
from .config import ExperimentConfig
from .datasets.registry import build_dataset_loader
from .datasets.io import read_csv_header
from .datasets.prepare import (
    prepare_fda_inspections_dataset,
    prepare_ipums_cps_ui_dataset,
    prepare_give_me_some_credit_dataset_from_openml,
    prepare_heloc_dataset_from_openml,
    prepare_hmda_dataset,
    prepare_nhamcs_dataset,
    prepare_ny_courts_pretrial_dataset,
    prepare_default_credit_dataset,
    prepare_south_german_credit_dataset,
)
from .metadata import load_dataset_registry
from .metrics.examples import score_sample
from .metrics.scorer import ExplanationScorer
from .normalization import ExplanationCanonicalizer
from .normalization.examples import sample_payload
from .reporting import write_benchmark_artifacts
from .runner import run_demo_benchmark
from .taxonomy import load_feature_taxonomy
from .models import supported_model_backends
from .explainers import supported_explainer_backends
from .paths import resolve_repo_path


@dataclass(slots=True)
class ValidationResult:
    ok: bool
    messages: list[str]


class BenchmarkPipeline:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.dataset_registry = load_dataset_registry(config)
        self.feature_taxonomy = load_feature_taxonomy(config)

    def validate(self) -> ValidationResult:
        messages: list[str] = []
        ok = True

        for dataset_id in self.config.enabled_datasets:
            if dataset_id not in self.dataset_registry:
                ok = False
                messages.append(f"Missing dataset registry entry for '{dataset_id}'.")
            elif dataset_id not in self.feature_taxonomy:
                ok = False
                messages.append(f"Missing feature taxonomy for '{dataset_id}'.")
            else:
                loader = build_dataset_loader(self.config, dataset_id)
                source_path, source_kind = loader.resolve_default_source()
                resolved_source = self.config.dataset_source_override(dataset_id) or source_path
                if source_kind == "local_override" and not Path(source_path).exists():
                    ok = False
                    messages.append(
                        f"Dataset '{dataset_id}' override path does not exist: {source_path}"
                    )
                for track in ("primary", "audit"):
                    bundle = loader.build_bundle(track)
                    if not bundle.included_features:
                        ok = False
                        messages.append(
                            f"Dataset '{dataset_id}' has no included features for track '{track}'."
                        )
                    table = loader.load_tabular_dataset(track=track)
                    if table.row_count <= 0:
                        ok = False
                        messages.append(
                            f"Dataset '{dataset_id}' has no rows available for track '{track}'."
                        )
                messages.append(
                    f"Dataset '{dataset_id}' default source: {resolved_source} [{source_kind}]"
                )

        if not self.config.enabled_models:
            ok = False
            messages.append("No models are enabled in the experiment config.")

        if not self.config.enabled_explainers:
            ok = False
            messages.append("No explainers are enabled in the experiment config.")

        threshold_policy = self.config.model_threshold_policy
        threshold_mode = threshold_policy["mode"]
        if threshold_mode not in {"fixed", "target_predicted_adverse_rate"}:
            ok = False
            messages.append(
                "Model threshold_policy.mode must be 'fixed' or "
                "'target_predicted_adverse_rate'."
            )
        if threshold_mode == "target_predicted_adverse_rate":
            target_rate = threshold_policy["target_predicted_adverse_rate"]
            if target_rate is None or target_rate <= 0 or target_rate >= 1:
                ok = False
                messages.append(
                    "Model threshold_policy.target_predicted_adverse_rate must lie "
                    "strictly between 0 and 1."
                )
            minimum = float(threshold_policy["minimum_threshold"])
            maximum = float(threshold_policy["maximum_threshold"])
            if minimum >= maximum:
                ok = False
                messages.append(
                    "Model threshold_policy minimum_threshold must be strictly below "
                    "maximum_threshold."
                )

        for name, path in self.config.registry_paths.items():
            if not Path(path).exists():
                ok = False
                messages.append(f"Registry path '{name}' does not exist: {path}")

        sample_cfg = self.config.payload["datasets"]["denied_instance_sampling"]
        if sample_cfg["max_instances_per_dataset_model_pair"] <= 0:
            ok = False
            messages.append("Denied-instance sample size must be positive.")

        evaluation_cfg = self.config.evaluation_protocol
        if evaluation_cfg["test_fraction"] <= 0 or evaluation_cfg["test_fraction"] >= 1:
            ok = False
            messages.append("Evaluation test_fraction must lie strictly between 0 and 1.")

        stability_cfg = self.config.stability_probe
        if stability_cfg["num_neighbors"] <= 0:
            ok = False
            messages.append("Stability probe num_neighbors must be positive.")
        if stability_cfg["candidate_multiplier"] <= 0:
            ok = False
            messages.append("Stability probe candidate_multiplier must be positive.")
        if stability_cfg["lime_seed_trials"] <= 0:
            ok = False
            messages.append("Stability probe lime_seed_trials must be positive.")

        if ok:
            messages.append(
                "Config, dataset registry, and paper-linked schema files are present."
            )
            messages.append(
                f"Validated {len(self.config.enabled_datasets)} datasets, "
                f"{len(self.config.enabled_models)} models, and "
                f"{len(self.config.enabled_explainers)} explainers."
            )

        return ValidationResult(ok=ok, messages=messages)

    def prepare_south_german_credit(
        self,
        input_path: str | None = None,
        output_path: str | None = None,
    ) -> dict:
        raw_input = input_path or "benchmark/data/raw/south_german_credit_uci/SouthGermanCredit.asc"
        csv_output = output_path or "benchmark/data/real/south_german_credit.csv"
        return prepare_south_german_credit_dataset(
            input_path=resolve_repo_path(raw_input),
            output_path=resolve_repo_path(csv_output),
        )

    def prepare_default_credit(
        self,
        input_path: str | None = None,
        output_path: str | None = None,
    ) -> dict:
        raw_input = (
            input_path
            or "benchmark/data/raw/default_credit_uci/default of credit card clients.xls"
        )
        csv_output = output_path or "benchmark/data/real/default_credit.csv"
        return prepare_default_credit_dataset(
            input_path=resolve_repo_path(raw_input),
            output_path=resolve_repo_path(csv_output),
        )

    def prepare_heloc(
        self,
        output_path: str | None = None,
        openml_data_id: int = 45554,
    ) -> dict:
        csv_output = output_path or "benchmark/data/real/heloc.csv"
        return prepare_heloc_dataset_from_openml(
            output_path=resolve_repo_path(csv_output),
            openml_data_id=openml_data_id,
        )

    def prepare_give_me_some_credit(
        self,
        output_path: str | None = None,
        openml_data_id: int = 46929,
    ) -> dict:
        csv_output = output_path or "benchmark/data/real/give_me_some_credit.csv"
        return prepare_give_me_some_credit_dataset_from_openml(
            output_path=resolve_repo_path(csv_output),
            openml_data_id=openml_data_id,
        )

    def prepare_hmda(
        self,
        input_path: str | None = None,
        output_path: str | None = None,
    ) -> dict:
        raw_input = input_path or "benchmark/data/raw/hmda/hmda_snapshot.csv"
        csv_output = output_path or "benchmark/data/real/hmda.csv"
        return prepare_hmda_dataset(
            input_path=resolve_repo_path(raw_input),
            output_path=resolve_repo_path(csv_output),
        )

    def prepare_ny_courts_pretrial(
        self,
        input_path: str | None = None,
        output_path: str | None = None,
    ) -> dict:
        raw_input = (
            input_path
            or "benchmark/data/raw/ny_courts_pretrial/pretrial_release_rows.csv"
        )
        csv_output = output_path or "benchmark/data/real/ny_courts_pretrial.csv"
        return prepare_ny_courts_pretrial_dataset(
            input_path=resolve_repo_path(raw_input),
            output_path=resolve_repo_path(csv_output),
        )

    def prepare_ipums_cps_ui(
        self,
        input_path: str | None = None,
        output_path: str | None = None,
    ) -> dict:
        raw_input = input_path or "benchmark/data/raw/ipums_cps_ui/ipums_cps_ui_extract.csv"
        csv_output = output_path or "benchmark/data/real/ipums_cps_ui.csv"
        return prepare_ipums_cps_ui_dataset(
            input_path=resolve_repo_path(raw_input),
            output_path=resolve_repo_path(csv_output),
        )

    def prepare_nhamcs(
        self,
        input_path: str | None = None,
        output_path: str | None = None,
    ) -> dict:
        raw_input = input_path or "benchmark/data/raw/nhamcs/nhamcs_ed_extract.csv"
        csv_output = output_path or "benchmark/data/real/nhamcs.csv"
        return prepare_nhamcs_dataset(
            input_path=resolve_repo_path(raw_input),
            output_path=resolve_repo_path(csv_output),
        )

    def prepare_fda_inspections(
        self,
        input_path: str | None = None,
        output_path: str | None = None,
    ) -> dict:
        raw_input = input_path or "benchmark/data/raw/fda_inspections/fda_inspections_extract.csv"
        csv_output = output_path or "benchmark/data/real/fda_inspections.csv"
        return prepare_fda_inspections_dataset(
            input_path=resolve_repo_path(raw_input),
            output_path=resolve_repo_path(csv_output),
        )

    def describe_dataset(self, dataset_id: str, track: str) -> list[str]:
        loader = build_dataset_loader(self.config, dataset_id)
        bundle = loader.build_bundle(track)
        source_path, source_kind = loader.resolve_default_source()

        messages = [
            f"Dataset: {bundle.display_name} ({bundle.dataset_id})",
            f"Track: {bundle.track}",
            f"Task: {bundle.task}",
            f"Target: {bundle.target} [{bundle.negative_label} -> adverse label]",
            f"Default source: {source_path} [{source_kind}]",
            f"Included features: {len(bundle.included_features)}",
            f"Excluded features: {len(bundle.excluded_features)}",
        ]

        if bundle.excluded_features:
            excluded = ", ".join(bundle.excluded_feature_names)
            messages.append(f"Excluded columns: {excluded}")

        high_reason_features = [
            feature.feature_name
            for feature in bundle.included_features
            if feature.principal_reason == "High"
        ]
        if high_reason_features:
            preview = ", ".join(high_reason_features[:8])
            messages.append(f"High-suitability principal reasons: {preview}")

        return messages

    def preview_dataset(self, dataset_id: str, track: str, path: str | None = None) -> dict:
        loader = build_dataset_loader(self.config, dataset_id)
        table = loader.load_tabular_dataset(track=track, path=path)
        preview_rows = list(islice(table.rows, 0, 2))
        return {
            "dataset_id": table.dataset_id,
            "display_name": table.display_name,
            "track": table.track,
            "target": table.target,
            "feature_count": len(table.feature_names),
            "row_count": table.row_count,
            "source_kind": table.source_kind,
            "source_path": table.source_path,
            "feature_names": table.feature_names,
            "rows_preview": preview_rows,
        }

    def check_dataset_file(self, dataset_id: str, track: str, path: str) -> dict:
        loader = build_dataset_loader(self.config, dataset_id)
        bundle = loader.build_bundle(track)
        resolved_path = resolve_repo_path(path)
        header = read_csv_header(resolved_path)

        expected_features = set(feature.feature_name for feature in loader.feature_specs.values())
        expected_with_target = expected_features | {bundle.target}
        header_set = set(header)

        missing_required = sorted(
            name for name in bundle.included_feature_names + [bundle.target] if name not in header_set
        )
        missing_known = sorted(expected_with_target - header_set)
        extra_columns = sorted(header_set - expected_with_target)

        return {
            "dataset_id": dataset_id,
            "track": track,
            "path": str(resolved_path),
            "target": bundle.target,
            "header_column_count": len(header),
            "expected_feature_count": len(expected_features),
            "included_feature_count": len(bundle.included_feature_names),
            "excluded_feature_count": len(bundle.excluded_feature_names),
            "target_present": bundle.target in header_set,
            "missing_required_columns": missing_required,
            "missing_known_columns": missing_known,
            "extra_columns": extra_columns,
            "excluded_columns_present": sorted(
                name for name in bundle.excluded_feature_names if name in header_set
            ),
            "status": "ok" if not missing_required else "schema_mismatch",
        }

    def normalize_example(self, dataset_id: str, track: str) -> dict:
        loader = build_dataset_loader(self.config, dataset_id)
        bundle = loader.build_bundle(track)
        payload = sample_payload(bundle)
        canonicalizer = ExplanationCanonicalizer(bundle)
        return canonicalizer.canonicalize(payload).to_dict()

    def adapt_example(self, dataset_id: str, track: str, explainer_id: str) -> dict:
        loader = build_dataset_loader(self.config, dataset_id)
        bundle = loader.build_bundle(track)
        raw_payload = sample_raw_payload(bundle, explainer_id=explainer_id)
        adapted_payload = adapt_raw_example(
            dataset_bundle=bundle,
            explainer_id=explainer_id,
            raw_payload=raw_payload,
        )
        return {
            "raw_payload": raw_payload,
            "adapted_payload": adapted_payload,
        }

    def score_example(self, dataset_id: str, track: str) -> dict:
        loader = build_dataset_loader(self.config, dataset_id)
        bundle = loader.build_bundle(track)
        return score_sample(bundle)

    def score_adapted_example(self, dataset_id: str, track: str, explainer_id: str) -> dict:
        loader = build_dataset_loader(self.config, dataset_id)
        bundle = loader.build_bundle(track)
        raw_payload = sample_raw_payload(bundle, explainer_id=explainer_id)
        adapted_payload = adapt_raw_example(
            dataset_bundle=bundle,
            explainer_id=explainer_id,
            raw_payload=raw_payload,
        )
        canonical = ExplanationCanonicalizer(bundle).canonicalize(adapted_payload)
        scores = ExplanationScorer().score(canonical)
        return {
            "raw_payload": raw_payload,
            "adapted_payload": adapted_payload,
            "canonical_explanation": canonical.to_dict(),
            "scores": scores,
        }

    def run_demo_benchmark(
        self,
        write_outputs: bool = False,
        output_dir: str | None = None,
        summary_only: bool = False,
    ) -> dict:
        payload = run_demo_benchmark(self.config)
        if write_outputs:
            payload["artifacts"] = write_benchmark_artifacts(
                config=self.config,
                payload=payload,
                output_dir=output_dir,
            )
        if summary_only:
            compact = {
                "experiment_name": payload["experiment_name"],
                "generated_at_utc": payload["generated_at_utc"],
                "capabilities": payload["capabilities"],
                "overview": payload["overview"],
            }
            if "artifacts" in payload:
                compact["artifacts"] = payload["artifacts"]
            return compact
        return payload

    def rebuild_artifacts_from_results(
        self,
        results_path: str,
        output_dir: str | None = None,
    ) -> dict:
        resolved_path = resolve_repo_path(results_path)
        with resolved_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        artifacts = write_benchmark_artifacts(
            config=self.config,
            payload=payload,
            output_dir=output_dir,
        )
        return {
            "results_json": str(resolved_path),
            "artifacts": artifacts,
        }

    def show_capabilities(self) -> dict:
        return {
            "runtime_capabilities": detect_capabilities(),
            "model_backends": [
                {
                    "model_id": item.model_id,
                    "backend": item.backend,
                    "implementation": item.implementation,
                    "reason": item.reason,
                }
                for item in supported_model_backends()
            ],
            "explainer_backends": [
                {
                    "explainer_id": item.explainer_id,
                    "backend": item.backend,
                    "implementation": item.implementation,
                    "reason": item.reason,
                }
                for item in supported_explainer_backends()
            ],
        }

    def show_data_sources(self) -> dict:
        datasets: list[dict[str, object]] = []
        demo_count = 0
        local_ready_count = 0
        local_missing_count = 0

        for dataset_id in self.config.enabled_datasets:
            loader = build_dataset_loader(self.config, dataset_id)
            source_path, source_kind = loader.resolve_default_source()
            resolved_path = resolve_repo_path(source_path)
            exists = Path(resolved_path).exists()
            if source_kind == "demo":
                status = "demo_fixture"
                demo_count += 1
            elif exists:
                status = "local_override_ready"
                local_ready_count += 1
            else:
                status = "local_override_missing"
                local_missing_count += 1

            datasets.append(
                {
                    "dataset_id": dataset_id,
                    "source_kind": source_kind,
                    "source_path": str(source_path),
                    "resolved_path": str(resolved_path),
                    "path_exists": exists,
                    "status": status,
                }
            )

        return {
            "datasets": datasets,
            "summary": {
                "dataset_count": len(datasets),
                "demo_fixture_count": demo_count,
                "local_override_ready_count": local_ready_count,
                "local_override_missing_count": local_missing_count,
            },
        }

    def doctor(self) -> list[str]:
        capabilities = detect_capabilities()
        readiness = empirical_readiness()
        data_sources = self.show_data_sources()

        messages = [
            "LegalBench-XAI runtime doctor",
            "",
            "Capability status:",
        ]
        for module_name, item in capabilities.items():
            status = "available" if item["available"] else "missing"
            version = f" (version {item['version']})" if item["version"] else ""
            messages.append(
                f"- {module_name}: {status}{version} [{item['description']}]"
            )

        messages.extend(
            [
                "",
                "Readiness targets:",
            ]
        )
        for group_name, item in readiness.items():
            if item["ready"]:
                messages.append(f"- {group_name}: ready")
            else:
                missing = ", ".join(item["missing_modules"])
                messages.append(f"- {group_name}: missing {missing}")

        messages.extend(
            [
                "",
                "Dataset sources:",
            ]
        )
        for item in data_sources["datasets"]:
            exists_text = "exists" if item["path_exists"] else "missing"
            messages.append(
                f"- {item['dataset_id']}: {item['source_kind']} [{exists_text}] -> {item['resolved_path']}"
            )

        messages.extend(
            [
                "",
                "Recommended next step:",
                _recommended_next_step(readiness, data_sources["summary"]),
                "",
                "Setup guide:",
                "- See benchmark/EMPIRICAL_SETUP.md",
            ]
        )
        return messages


def _recommended_next_step(
    readiness: dict[str, dict[str, bool | list[str]]],
    data_summary: dict[str, object],
) -> str:
    if not bool(readiness["first_real_slice"]["ready"]):
        missing = ", ".join(readiness["first_real_slice"]["missing_modules"])
        return f"- Install the first real-slice modules: {missing}"
    if not bool(readiness["full_empirical_core"]["ready"]):
        missing = ", ".join(readiness["full_empirical_core"]["missing_modules"])
        return f"- Install the remaining full-core modules: {missing}"
    if int(data_summary.get("local_override_missing_count", 0)) > 0:
        return "- Fix the missing local dataset override paths before attempting a real-data benchmark run."
    if int(data_summary.get("local_override_ready_count", 0)) == 0:
        return "- Point at least one dataset to a local real CSV, then run `check-dataset-file` before the first empirical benchmark."
    if not bool(readiness["full_empirical_plus_ebm"]["ready"]):
        missing = ", ".join(readiness["full_empirical_plus_ebm"]["missing_modules"])
        return f"- Optional: install the EBM extension modules: {missing}"
    return "- Runtime is ready for the full empirical benchmark stack."
