from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class PredictionRecord:
    score: float
    label: str
    threshold: float | None = None
    decision_margin: float | None = None


@dataclass(slots=True)
class FeatureTaxonomyRecord:
    mutability: str
    verifiability: str
    correctability: str
    sensitivity: str
    principal_reason: str


@dataclass(slots=True)
class ReferencedFeature:
    feature_name: str
    plain_meaning: str
    signed_direction: str
    importance_weight: float | None = None
    rank: int | None = None
    raw_value: Any = None
    transformed_value: Any = None
    taxonomy: FeatureTaxonomyRecord | None = None


@dataclass(slots=True)
class PrincipalReason:
    rank: int
    reason_text: str
    supporting_features: list[str]
    signed_direction: str
    specificity_level: str
    grouped_reason_type: str


@dataclass(slots=True)
class CounterfactualLegalityFlags:
    direct_sensitive_feature: bool = False
    immutable_feature: bool = False
    proxy_sensitive_feature: bool = False
    lender_controlled_feature: bool = False
    low_principal_reason_feature: bool = False


@dataclass(slots=True)
class CounterfactualChange:
    feature_name: str
    from_value: Any
    to_value: Any
    delta: float | None = None
    change_direction: str = "category_change"
    validity_flip: bool | None = None
    realism_score: float | None = None
    effort_score: float | None = None
    taxonomy: FeatureTaxonomyRecord | None = None
    legality_flags: CounterfactualLegalityFlags = field(
        default_factory=CounterfactualLegalityFlags
    )


@dataclass(slots=True)
class CanonicalExplanation:
    instance_id: str
    dataset_id: str
    track: str
    model_id: str
    explainer_id: str
    explanation_family: str
    prediction: PredictionRecord
    referenced_features: list[ReferencedFeature]
    principal_reasons: list[PrincipalReason]
    counterfactual_changes: list[CounterfactualChange] = field(default_factory=list)
    seed: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DatasetSpec:
    dataset_id: str
    display_name: str
    task: str
    target: str
    positive_label: str
    negative_label: str
    demo_data_path: str
    file_format: str
    primary_track_excluded_features: list[str]
    audit_track_excluded_features: list[str]
    paper_sources: list[str]


@dataclass(slots=True)
class FeatureSpec:
    feature_name: str
    plain_meaning: str
    mutability: str
    verifiability: str
    correctability: str
    sensitivity: str
    principal_reason: str

    def taxonomy_record(self) -> FeatureTaxonomyRecord:
        return FeatureTaxonomyRecord(
            mutability=self.mutability,
            verifiability=self.verifiability,
            correctability=self.correctability,
            sensitivity=self.sensitivity,
            principal_reason=self.principal_reason,
        )


@dataclass(slots=True)
class DatasetBundle:
    dataset_id: str
    display_name: str
    track: str
    task: str
    target: str
    positive_label: str
    negative_label: str
    included_features: list[FeatureSpec]
    excluded_features: list[FeatureSpec]

    @property
    def included_feature_names(self) -> list[str]:
        return [feature.feature_name for feature in self.included_features]

    @property
    def excluded_feature_names(self) -> list[str]:
        return [feature.feature_name for feature in self.excluded_features]


@dataclass(slots=True)
class TabularDataset:
    dataset_id: str
    display_name: str
    track: str
    target: str
    feature_names: list[str]
    rows: list[dict[str, Any]]
    source_kind: str
    source_path: str

    @property
    def row_count(self) -> int:
        return len(self.rows)
