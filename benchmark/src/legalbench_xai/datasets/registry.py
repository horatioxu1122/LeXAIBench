from ..config import ExperimentConfig
from ..corrections import GEO_CODE_COLUMNS, corrections_from_config
from ..metadata import load_dataset_registry
from ..taxonomy import load_feature_taxonomy
from .base import DatasetLoader


class _StaticDatasetLoader(DatasetLoader):
    pass


def build_dataset_loader(config: ExperimentConfig, dataset_id: str) -> DatasetLoader:
    dataset_registry = load_dataset_registry(config)
    feature_taxonomy = load_feature_taxonomy(config)

    if dataset_id not in dataset_registry:
        raise KeyError(f"Unknown dataset '{dataset_id}'.")
    if dataset_id not in feature_taxonomy:
        raise KeyError(f"Missing feature taxonomy for dataset '{dataset_id}'.")

    return _StaticDatasetLoader(
        dataset_spec=dataset_registry[dataset_id],
        feature_specs=feature_taxonomy[dataset_id],
        source_override=config.dataset_source_override(dataset_id),
        numeric_parsing_fix=corrections_from_config(config).numeric_parsing_fix,
        text_columns=GEO_CODE_COLUMNS if corrections_from_config(config).geo_target_encoding else (),
    )


def list_supported_datasets(config: ExperimentConfig) -> list[str]:
    return sorted(load_dataset_registry(config).keys())
