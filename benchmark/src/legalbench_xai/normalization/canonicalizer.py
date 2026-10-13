from dataclasses import dataclass
from typing import Any

from ..corrections import V1, CorrectionOptions
from ..types import (
    CanonicalExplanation,
    CounterfactualChange,
    CounterfactualLegalityFlags,
    DatasetBundle,
    FeatureSpec,
    PredictionRecord,
    PrincipalReason,
    ReferencedFeature,
)
from .rules import (
    change_direction,
    default_reason_text,
    grouped_reason_label,
    signed_direction_from_weight,
    specificity_level,
)


@dataclass(slots=True)
class NormalizationInput:
    instance_id: str
    model_id: str
    explainer_id: str
    explanation_family: str
    prediction: dict[str, Any]
    features: list[dict[str, Any]]
    metadata: dict[str, Any]
    counterfactual_changes: list[dict[str, Any]]
    seed: int | None = None


class ExplanationCanonicalizer:
    def __init__(
        self,
        dataset_bundle: DatasetBundle,
        principal_reason_limit: int = 4,
        *,
        corrections: CorrectionOptions | None = V1,
    ) -> None:
        """Canonicalizer for one dataset bundle.

        corrections (default V1, i.e. the published behaviour) switches on:
        - reason_grouping_fix: PAY_AMT* groups as "recent payment amounts"
          (normalization/rules.py longest-prefix-first matching);
        - cf_respect_nonactionable: counterfactual changes to taxonomy mutability "N"
          (not actionable by the applicant) get legality_flags.immutable_feature=True,
          like mutability "I", so the scorer hard-fails them.
        """
        if corrections is None:
            corrections = V1
        if not isinstance(corrections, CorrectionOptions):
            raise TypeError(
                f"corrections must be a CorrectionOptions, got {type(corrections).__name__}"
            )
        self.dataset_bundle = dataset_bundle
        self.principal_reason_limit = principal_reason_limit
        self.corrections = corrections
        self._fix_reason_grouping = bool(corrections.reason_grouping_fix)
        self._block_nonactionable = bool(corrections.cf_respect_nonactionable)
        self._feature_index = {
            feature.feature_name: feature for feature in dataset_bundle.included_features
        }

    def canonicalize(self, payload: dict[str, Any]) -> CanonicalExplanation:
        normalized = NormalizationInput(
            instance_id=payload["instance_id"],
            model_id=payload["model_id"],
            explainer_id=payload["explainer_id"],
            explanation_family=payload["explanation_family"],
            prediction=payload["prediction"],
            features=list(payload.get("features", [])),
            metadata=dict(payload.get("metadata", {})),
            counterfactual_changes=list(payload.get("counterfactual_changes", [])),
            seed=payload.get("seed"),
        )

        referenced_features = self._build_referenced_features(normalized.features)
        counterfactual_changes = self._build_counterfactual_changes(
            normalized.counterfactual_changes
        )
        principal_reasons = self._build_principal_reasons(
            referenced_features=referenced_features,
            explanation_family=normalized.explanation_family,
            counterfactual_changes=counterfactual_changes,
        )

        metadata = dict(normalized.metadata)
        metadata["compression_applied"] = True
        metadata["included_feature_count"] = len(referenced_features)

        return CanonicalExplanation(
            instance_id=normalized.instance_id,
            dataset_id=self.dataset_bundle.dataset_id,
            track=self.dataset_bundle.track,
            model_id=normalized.model_id,
            explainer_id=normalized.explainer_id,
            explanation_family=normalized.explanation_family,
            prediction=PredictionRecord(**normalized.prediction),
            referenced_features=referenced_features,
            principal_reasons=principal_reasons,
            counterfactual_changes=counterfactual_changes,
            seed=normalized.seed,
            metadata=metadata,
        )

    def _build_referenced_features(
        self,
        raw_features: list[dict[str, Any]],
    ) -> list[ReferencedFeature]:
        items: list[ReferencedFeature] = []
        for rank, raw_feature in enumerate(raw_features, start=1):
            feature_name = raw_feature["feature_name"]
            feature_spec = self._feature_index.get(feature_name)
            if feature_spec is None:
                continue

            weight = raw_feature.get("importance_weight")
            signed_direction = raw_feature.get("signed_direction") or signed_direction_from_weight(
                weight
            )

            items.append(
                ReferencedFeature(
                    feature_name=feature_name,
                    plain_meaning=feature_spec.plain_meaning,
                    signed_direction=signed_direction,
                    importance_weight=weight,
                    rank=raw_feature.get("rank", rank),
                    raw_value=raw_feature.get("raw_value"),
                    transformed_value=raw_feature.get("transformed_value"),
                    taxonomy=feature_spec.taxonomy_record(),
                )
            )

        items.sort(
            key=lambda item: (
                item.rank if item.rank is not None else 10**9,
                -(abs(item.importance_weight) if item.importance_weight is not None else 0.0),
            )
        )
        return items

    def _build_counterfactual_changes(
        self,
        raw_changes: list[dict[str, Any]],
    ) -> list[CounterfactualChange]:
        changes: list[CounterfactualChange] = []
        for raw_change in raw_changes:
            feature_name = raw_change["feature_name"]
            feature_spec = self._feature_index.get(feature_name)
            if feature_spec is None:
                continue

            from_value = raw_change["from_value"]
            to_value = raw_change["to_value"]
            immutable_feature = feature_spec.mutability == "I"
            if self._block_nonactionable and feature_spec.mutability == "N":
                # Correction cf_respect_nonactionable: "N" (not actionable by the
                # applicant) is treated like "I" (immutable).
                immutable_feature = True
            flags = CounterfactualLegalityFlags(
                direct_sensitive_feature=feature_spec.sensitivity == "S",
                immutable_feature=immutable_feature,
                proxy_sensitive_feature=feature_spec.sensitivity == "P",
                lender_controlled_feature=feature_spec.mutability == "P",
                low_principal_reason_feature=feature_spec.principal_reason == "Low",
            )
            delta = raw_change.get("delta")
            if delta is None and isinstance(from_value, (int, float)) and isinstance(
                to_value, (int, float)
            ):
                delta = float(to_value - from_value)

            changes.append(
                CounterfactualChange(
                    feature_name=feature_name,
                    from_value=from_value,
                    to_value=to_value,
                    delta=delta,
                    change_direction=raw_change.get(
                        "change_direction", change_direction(from_value, to_value)
                    ),
                    validity_flip=raw_change.get("validity_flip"),
                    realism_score=raw_change.get("realism_score"),
                    effort_score=raw_change.get("effort_score"),
                    taxonomy=feature_spec.taxonomy_record(),
                    legality_flags=flags,
                )
            )
        return changes

    def _build_principal_reasons(
        self,
        referenced_features: list[ReferencedFeature],
        explanation_family: str,
        counterfactual_changes: list[CounterfactualChange],
    ) -> list[PrincipalReason]:
        reasons: list[PrincipalReason] = []
        seen_groups: set[str] = set()

        fix_prefix_order = self._fix_reason_grouping
        for feature in referenced_features:
            grouped_label = grouped_reason_label(
                feature.feature_name, fix_prefix_order=fix_prefix_order
            )
            if grouped_label is not None and grouped_label in seen_groups:
                continue

            reason_text = grouped_label or default_reason_text(
                self._feature_index[feature.feature_name], feature.signed_direction
            )
            grouped_reason_type = (
                "compressed_temporal" if grouped_label is not None else "raw_feature"
            )
            supporting_features = (
                [
                    candidate.feature_name
                    for candidate in referenced_features
                    if grouped_reason_label(
                        candidate.feature_name, fix_prefix_order=fix_prefix_order
                    )
                    == grouped_label
                ]
                if grouped_label is not None
                else [feature.feature_name]
            )
            reasons.append(
                PrincipalReason(
                    rank=len(reasons) + 1,
                    reason_text=reason_text,
                    supporting_features=supporting_features,
                    signed_direction=feature.signed_direction,
                    specificity_level=specificity_level(reason_text),
                    grouped_reason_type=grouped_reason_type,
                )
            )
            if grouped_label is not None:
                seen_groups.add(grouped_label)
            if len(reasons) >= self.principal_reason_limit:
                break

        if explanation_family == "counterfactual" and counterfactual_changes and len(reasons) < self.principal_reason_limit:
            # Correction counterfactual_reasons_from_changes: the referenced features
            # already are the changed features, so a change already cited above is
            # not cited a second time (with the opposite direction). v1 is unchanged.
            already_cited = (
                {name for reason in reasons for name in reason.supporting_features}
                if self.corrections.counterfactual_reasons_from_changes
                else set()
            )
            for change in counterfactual_changes:
                if change.feature_name in already_cited:
                    continue
                feature = self._feature_index[change.feature_name]
                reasons.append(
                    PrincipalReason(
                        rank=len(reasons) + 1,
                        reason_text=f"changing {feature.plain_meaning.replace('_', ' ')} may improve the outcome",
                        supporting_features=[change.feature_name],
                        signed_direction="decreases_denial",
                        specificity_level="specific",
                        grouped_reason_type="counterfactual",
                    )
                )
                if len(reasons) >= self.principal_reason_limit:
                    break

        return reasons
