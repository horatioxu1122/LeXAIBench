import json
from pathlib import Path

from .config import ExperimentConfig
from .types import DatasetSpec


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_dataset_registry(config: ExperimentConfig) -> dict[str, DatasetSpec]:
    registry_path = config.registry_paths["dataset_registry"]
    payload = _load_json(registry_path)
    datasets: dict[str, DatasetSpec] = {}
    for dataset_id, item in payload["datasets"].items():
        datasets[dataset_id] = DatasetSpec(dataset_id=dataset_id, **item)
    return datasets
