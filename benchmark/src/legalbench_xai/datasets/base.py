from abc import ABC
from pathlib import Path

from ..datasets.io import load_csv_rows
from ..paths import resolve_repo_path
from ..types import DatasetBundle, DatasetSpec, FeatureSpec
from ..types import TabularDataset


class DatasetLoader(ABC):
    def __init__(
        self,
        dataset_spec: DatasetSpec,
        feature_specs: dict[str, FeatureSpec],
        source_override: str | None = None,
        *,
        numeric_parsing_fix: bool = False,
        text_columns: tuple[str, ...] = (),
    ) -> None:
        self.dataset_spec = dataset_spec
        self.feature_specs = feature_specs
        self.source_override = source_override
        # cslaw corrections numeric_parsing_fix (v2): parse exponent-notation numbers.
        # False reproduces the v1 CSV parsing of the published bundles.
        self.numeric_parsing_fix = numeric_parsing_fix
        # cslaw corrections geo_target_encoding: columns kept as text (empty by default).
        self.text_columns = tuple(text_columns)

    def build_bundle(self, track: str) -> DatasetBundle:
        if track not in {"primary", "audit"}:
            raise ValueError(f"Unsupported track '{track}'. Expected 'primary' or 'audit'.")

        excluded_names = (
            self.dataset_spec.primary_track_excluded_features
            if track == "primary"
            else self.dataset_spec.audit_track_excluded_features
        )
        included: list[FeatureSpec] = []
        excluded: list[FeatureSpec] = []

        for feature_name, feature_spec in self.feature_specs.items():
            if feature_name in excluded_names:
                excluded.append(feature_spec)
            else:
                included.append(feature_spec)

        included.sort(key=lambda item: item.feature_name)
        excluded.sort(key=lambda item: item.feature_name)

        return DatasetBundle(
            dataset_id=self.dataset_spec.dataset_id,
            display_name=self.dataset_spec.display_name,
            track=track,
            task=self.dataset_spec.task,
            target=self.dataset_spec.target,
            positive_label=self.dataset_spec.positive_label,
            negative_label=self.dataset_spec.negative_label,
            included_features=included,
            excluded_features=excluded,
        )

    def load_tabular_dataset(self, track: str, path: str | None = None) -> TabularDataset:
        bundle = self.build_bundle(track)
        source_path, source_kind = self._resolve_source_path(path)
        resolved_path = resolve_repo_path(source_path)
        if self.dataset_spec.file_format != "csv":
            raise ValueError(
                f"Unsupported file format '{self.dataset_spec.file_format}' for dataset "
                f"'{self.dataset_spec.dataset_id}'."
            )
        rows = load_csv_rows(
            Path(resolved_path),
            numeric_parsing_fix=self.numeric_parsing_fix,
            text_columns=self.text_columns,
        )

        filtered_rows: list[dict[str, object]] = []
        feature_names = bundle.included_feature_names
        for row in rows:
            filtered_rows.append(
                {name: row.get(name) for name in feature_names} | {bundle.target: row.get(bundle.target)}
            )

        return TabularDataset(
            dataset_id=bundle.dataset_id,
            display_name=bundle.display_name,
            track=bundle.track,
            target=bundle.target,
            feature_names=feature_names,
            rows=filtered_rows,
            source_kind=source_kind,
            source_path=str(resolved_path),
        )

    def resolve_default_source(self) -> tuple[str, str]:
        return self._resolve_source_path(None)

    def _resolve_source_path(self, path: str | None) -> tuple[str, str]:
        if path is not None:
            return path, "custom"
        if self.source_override:
            return self.source_override, "local_override"
        return self.dataset_spec.demo_data_path, "demo"
