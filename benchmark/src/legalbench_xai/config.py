import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .paths import resolve_repo_path


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@dataclass(slots=True)
class ExperimentConfig:
    path: Path
    payload: dict[str, Any]

    @property
    def random_seed(self) -> int:
        return int(self.payload["experiment"].get("random_seed", 0))

    @property
    def experiment_name(self) -> str:
        return str(self.payload["experiment"]["name"])

    @property
    def enabled_datasets(self) -> list[str]:
        return list(self.payload["datasets"]["enabled"])

    @property
    def enabled_models(self) -> list[str]:
        return list(self.payload["models"]["enabled"])

    @property
    def model_threshold(self) -> float:
        configured = self.payload.get("models", {}).get("decision_threshold", 0.5)
        return float(configured)

    @property
    def model_threshold_policy(self) -> dict[str, Any]:
        configured = dict(self.payload.get("models", {}).get("threshold_policy", {}))
        return {
            "mode": str(configured.get("mode", "fixed")),
            "target_predicted_adverse_rate": (
                float(configured["target_predicted_adverse_rate"])
                if configured.get("target_predicted_adverse_rate") is not None
                else None
            ),
            "minimum_threshold": float(configured.get("minimum_threshold", 0.01)),
            "maximum_threshold": float(configured.get("maximum_threshold", 0.99)),
        }

    @property
    def enabled_explainers(self) -> list[str]:
        return list(self.payload["explainers"]["enabled"])

    @property
    def registry_paths(self) -> dict[str, Path]:
        return {
            key: resolve_repo_path(value)
            for key, value in self.payload["registry"].items()
        }

    @property
    def output_root(self) -> Path:
        return resolve_repo_path(self.payload["experiment"]["output_root"])

    @property
    def denied_instance_sampling(self) -> dict[str, Any]:
        return dict(self.payload["datasets"]["denied_instance_sampling"])

    @property
    def stability_probe(self) -> dict[str, Any]:
        configured = dict(self.payload.get("stability_probe", {}))
        lime_trials = configured.get(
            "lime_seed_trials",
            self.payload.get("explainers", {}).get("lime", {}).get("num_seeds", 3),
        )
        return {
            "enabled": bool(configured.get("enabled", False)),
            "num_neighbors": int(configured.get("num_neighbors", 3)),
            "candidate_multiplier": int(configured.get("candidate_multiplier", 4)),
            "same_label_only": bool(configured.get("same_label_only", True)),
            "lime_seed_trials": int(lime_trials),
            "numeric_jitter_fraction": float(
                configured.get("numeric_jitter_fraction", 0.06)
            ),
            "categorical_flip_probability": float(
                configured.get("categorical_flip_probability", 0.18)
            ),
        }

    @property
    def reporting_bootstrap(self) -> dict[str, Any]:
        configured = dict(self.payload.get("reporting", {}).get("bootstrap", {}))
        return {
            "enabled": bool(configured.get("enabled", True)),
            "num_resamples": int(configured.get("num_resamples", 1000)),
            "confidence_level": float(configured.get("confidence_level", 0.95)),
            "method": str(configured.get("method", "hierarchical")),
            "hierarchy_levels": list(
                configured.get(
                    "hierarchy_levels",
                    ["dataset_id", "track", "model_id"],
                )
            ),
        }

    @property
    def evaluation_protocol(self) -> dict[str, Any]:
        configured = dict(self.payload.get("evaluation", {}))
        return {
            "heldout_enabled": bool(configured.get("heldout_enabled", True)),
            "split_strategy": str(configured.get("split_strategy", "stratified")),
            "test_fraction": float(configured.get("test_fraction", 0.30)),
            "split_seed": int(configured.get("split_seed", self.random_seed)),
            "fallback_to_full_dataset": bool(
                configured.get("fallback_to_full_dataset", True)
            ),
        }

    @property
    def dataset_source_overrides(self) -> dict[str, str]:
        configured = self.payload.get("dataset_sources", {}).get("local_paths", {})
        merged = dict(configured)
        for dataset_id in self.enabled_datasets:
            env_key = _dataset_override_env_var(dataset_id)
            env_value = os.getenv(env_key)
            if env_value:
                merged[dataset_id] = env_value
        return merged

    def dataset_source_override(self, dataset_id: str) -> str | None:
        return self.dataset_source_overrides.get(dataset_id)


def load_experiment_config(path_str: str) -> ExperimentConfig:
    path = resolve_repo_path(path_str)
    payload = _load_json(path)
    return ExperimentConfig(path=path, payload=payload)


def _dataset_override_env_var(dataset_id: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", dataset_id).upper().strip("_")
    return f"LEGALBENCH_XAI_DATASET_{normalized}_PATH"
