import json
from pathlib import Path

from .config import ExperimentConfig
from .types import FeatureSpec


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_feature_taxonomy(config: ExperimentConfig) -> dict[str, dict[str, FeatureSpec]]:
    taxonomy_path = config.registry_paths["feature_taxonomy"]
    payload = _load_json(taxonomy_path)
    datasets: dict[str, dict[str, FeatureSpec]] = {}
    for dataset_id, dataset_payload in payload["datasets"].items():
        datasets[dataset_id] = {}
        for feature_name, feature_payload in dataset_payload["features"].items():
            datasets[dataset_id][feature_name] = FeatureSpec(
                feature_name=feature_name,
                **feature_payload,
            )
    return datasets
