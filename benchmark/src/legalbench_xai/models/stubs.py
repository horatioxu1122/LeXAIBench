import math
from dataclasses import dataclass, field
from typing import Any

from ..corrections import V1, CorrectionOptions
from ..types import DatasetBundle, TabularDataset


PRINCIPAL_REASON_WEIGHT = {"High": 1.0, "Medium": 0.7, "Low": 0.3}


@dataclass(slots=True)
class FeatureProfile:
    feature_name: str
    plain_meaning: str
    principal_reason: str
    mutability: str
    sensitivity: str
    is_numeric: bool
    weight: float
    numeric_min: float | None = None
    numeric_max: float | None = None
    positive_mean: float | None = None
    negative_mean: float | None = None
    risk_direction: float = 1.0
    category_risk: dict[str, float] = field(default_factory=dict)
    safer_value: Any = None


@dataclass(slots=True)
class RowPrediction:
    instance_id: str
    score: float
    label: str
    threshold: float
    decision_margin: float
    raw_score: float
    feature_contributions: dict[str, float]


class HeuristicModel:
    # Same opt-in switch as VectorizedClassifierModel. The stub has no fitted model, so
    # its native and original contributions coincide and ranking is unaffected.
    use_native_ranking: bool = False
    native_contribution_source: str = "heuristic_stub"
    # Versioned corrections; the runner sets the instance attribute after fit_model.
    corrections: CorrectionOptions = V1

    def __init__(
        self,
        model_id: str,
        dataset_bundle: DatasetBundle,
        profiles: dict[str, FeatureProfile],
        raw_min: float,
        raw_max: float,
        threshold: float = 0.5,
    ) -> None:
        self.model_id = model_id
        self.dataset_bundle = dataset_bundle
        self.profiles = profiles
        self.raw_min = raw_min
        self.raw_max = raw_max
        self.threshold = threshold

    def predict_row(self, row: dict[str, Any], instance_id: str) -> RowPrediction:
        contributions = {
            feature_name: self._feature_contribution(profile, row.get(feature_name))
            for feature_name, profile in self.profiles.items()
        }
        raw_score = self._combine_contributions(list(contributions.values()))
        score = self._normalize_score(raw_score)
        label = (
            self.dataset_bundle.negative_label
            if score >= self.threshold
            else self.dataset_bundle.positive_label
        )
        return RowPrediction(
            instance_id=instance_id,
            score=score,
            label=label,
            threshold=self.threshold,
            decision_margin=score - self.threshold,
            raw_score=raw_score,
            feature_contributions=contributions,
        )

    def original_feature_contributions(self, row: dict[str, Any]) -> dict[str, float]:
        """Heuristic per-feature contributions (the same dict predict_row attaches).

        Unitless proxy: taxonomy weight x centred min-max value (numeric) or
        weight x (category adverse rate - 0.5); positive = toward denial. Not additive
        to any fitted model score.
        """
        return self.predict_row(
            row=row,
            instance_id=f"{self.model_id}-proxy",
        ).feature_contributions

    def ranking_contributions(self, row: dict[str, Any]) -> dict[str, float]:
        """Stub ranking input: always the heuristic (original) contributions."""
        return self.original_feature_contributions(row)

    def native_feature_contributions(self, row: dict[str, Any]) -> dict[str, float]:
        """Single-row form of native_feature_contributions_batch (heuristic proxy)."""
        contributions, _bias = self.native_feature_contributions_batch([row])
        return contributions[0]

    def native_feature_contributions_batch(
        self,
        rows: list[dict[str, Any]],
    ) -> tuple[list[dict[str, float]], float]:
        """Stub "native" contributions: the heuristic feature_contributions per row.

        Source "heuristic_stub". Keys: every bundle included feature (0.0 if absent from
        the profiles). Values are the unitless heuristic proxy (positive = toward
        denial), NOT log-odds, and bias is 0.0, so sum(values) + bias does not equal
        raw_score in general.
        """
        feature_names = list(self.dataset_bundle.included_feature_names)
        results: list[dict[str, float]] = []
        for row in rows:
            contributions = self.original_feature_contributions(row)
            results.append(
                {
                    feature_name: float(contributions.get(feature_name, 0.0))
                    for feature_name in feature_names
                }
            )
        return results, 0.0

    def recommend_changes(
        self,
        row: dict[str, Any],
        prediction: RowPrediction,
        allow_illegal: bool,
        limit: int = 2,
        block_nonactionable: bool = False,
    ) -> list[dict[str, Any]]:
        """Heuristic counterfactual changes (stub backend).

        block_nonactionable (correction cf_respect_nonactionable) treats mutability
        "N" like "I": skipped whenever immutable features are skipped.
        """
        ranked = sorted(
            prediction.feature_contributions.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        changes: list[dict[str, Any]] = []
        for feature_name, contribution in ranked:
            if contribution <= 0:
                continue
            profile = self.profiles[feature_name]
            if not allow_illegal and (
                profile.mutability == "I"
                or profile.sensitivity == "S"
                or (block_nonactionable and profile.mutability == "N")
            ):
                continue
            current_value = row.get(feature_name)
            target_value = profile.safer_value
            if target_value is None or current_value == target_value:
                continue
            changes.append(
                {
                    "feature_name": feature_name,
                    "from_value": current_value,
                    "to_value": target_value,
                    "validity_flip": True,
                    "realism_score": 0.82 if allow_illegal else 0.88,
                    "effort_score": self._effort_score(profile),
                }
            )
            if len(changes) >= limit:
                break
        return changes

    def _feature_contribution(self, profile: FeatureProfile, value: Any) -> float:
        if profile.is_numeric:
            assert profile.numeric_min is not None
            assert profile.numeric_max is not None
            scale = profile.numeric_max - profile.numeric_min
            if scale == 0:
                centered = 0.0
            else:
                centered = ((float(value) - profile.numeric_min) / scale) - 0.5
            contribution = profile.weight * profile.risk_direction * centered
            if self.model_id == "ebm":
                contribution = profile.weight * profile.risk_direction * (
                    centered + 0.5 * centered * abs(centered)
                )
            return contribution

        category_score = profile.category_risk.get(str(value), 0.5)
        centered = category_score - 0.5
        contribution = profile.weight * centered
        if self.model_id == "ebm":
            contribution = profile.weight * (centered + 0.4 * centered * abs(centered))
        return contribution

    def _combine_contributions(self, values: list[float]) -> float:
        if not values:
            return 0.0
        base = sum(values) / len(values)
        if self.model_id == "logistic_regression":
            return base
        if self.model_id == "ebm":
            return base + 0.15 * sum(abs(value) for value in values) / len(values)
        positives = sorted((value for value in values if value > 0), reverse=True)
        interaction = positives[0] * positives[1] if len(positives) >= 2 else 0.0
        return base + 0.35 * interaction

    def _normalize_score(self, raw_score: float) -> float:
        if self.raw_max == self.raw_min:
            normalized = 0.5
        else:
            normalized = (raw_score - self.raw_min) / (self.raw_max - self.raw_min)
        normalized = max(0.0, min(1.0, normalized))
        if self.model_id == "logistic_regression":
            return normalized
        if self.model_id == "ebm":
            return 1.0 / (1.0 + math.exp(-4.0 * (normalized - 0.5)))
        return 1.0 / (1.0 + math.exp(-5.0 * (normalized - 0.5)))

    def _effort_score(self, profile: FeatureProfile) -> float:
        if profile.mutability == "M":
            return 0.8
        if profile.mutability == "PS":
            return 0.6
        if profile.mutability == "PL":
            return 0.45
        if profile.mutability == "P":
            return 0.5
        return 0.1


def fit_stub_model(
    model_id: str,
    dataset_bundle: DatasetBundle,
    table: TabularDataset,
) -> HeuristicModel:
    profiles = _build_profiles(dataset_bundle, table)
    scratch_model = HeuristicModel(
        model_id=model_id,
        dataset_bundle=dataset_bundle,
        profiles=profiles,
        raw_min=0.0,
        raw_max=1.0,
    )
    raw_scores = [
        scratch_model._combine_contributions(
            [
                scratch_model._feature_contribution(profiles[name], row.get(name))
                for name in table.feature_names
            ]
        )
        for row in table.rows
    ]
    raw_min = min(raw_scores) if raw_scores else 0.0
    raw_max = max(raw_scores) if raw_scores else 1.0
    return HeuristicModel(
        model_id=model_id,
        dataset_bundle=dataset_bundle,
        profiles=profiles,
        raw_min=raw_min,
        raw_max=raw_max,
    )


def _build_profiles(
    dataset_bundle: DatasetBundle,
    table: TabularDataset,
) -> dict[str, FeatureProfile]:
    positive_rows = [
        row for row in table.rows if str(row[dataset_bundle.target]) == dataset_bundle.positive_label
    ]
    negative_rows = [
        row for row in table.rows if str(row[dataset_bundle.target]) == dataset_bundle.negative_label
    ]
    profiles: dict[str, FeatureProfile] = {}
    feature_specs = {feature.feature_name: feature for feature in dataset_bundle.included_features}

    for feature_name in table.feature_names:
        feature_spec = feature_specs[feature_name]
        values = [row.get(feature_name) for row in table.rows]
        is_numeric = all(isinstance(value, (int, float)) for value in values)
        profile = FeatureProfile(
            feature_name=feature_name,
            plain_meaning=feature_spec.plain_meaning,
            principal_reason=feature_spec.principal_reason,
            mutability=feature_spec.mutability,
            sensitivity=feature_spec.sensitivity,
            is_numeric=is_numeric,
            weight=PRINCIPAL_REASON_WEIGHT.get(feature_spec.principal_reason, 0.5),
        )
        if is_numeric:
            numeric_values = [float(value) for value in values]
            profile.numeric_min = min(numeric_values)
            profile.numeric_max = max(numeric_values)
            profile.positive_mean = _mean([
                float(row[feature_name]) for row in positive_rows
            ])
            profile.negative_mean = _mean([
                float(row[feature_name]) for row in negative_rows
            ])
            if profile.positive_mean is not None and profile.negative_mean is not None:
                profile.risk_direction = 1.0 if profile.negative_mean >= profile.positive_mean else -1.0
                profile.safer_value = profile.positive_mean
            else:
                profile.risk_direction = 1.0
                profile.safer_value = profile.numeric_min
        else:
            category_risk = _category_risk(values, negative_rows, feature_name)
            profile.category_risk = category_risk
            profile.safer_value = min(category_risk, key=category_risk.get)
        profiles[feature_name] = profile

    return profiles


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _category_risk(
    values: list[Any],
    negative_rows: list[dict[str, Any]],
    feature_name: str,
) -> dict[str, float]:
    counts: dict[str, int] = {}
    negative_counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    for row in negative_rows:
        key = str(row.get(feature_name))
        negative_counts[key] = negative_counts.get(key, 0) + 1
    return {
        key: negative_counts.get(key, 0) / count
        for key, count in counts.items()
    }
