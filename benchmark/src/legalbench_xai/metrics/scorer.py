import math
from dataclasses import asdict, dataclass, field
from typing import Any

from ..corrections import V1, CorrectionOptions
from ..types import CanonicalExplanation, CounterfactualChange, ReferencedFeature


VERIFICATION_MAP = {"Y": 1.0, "P": 0.5, "N": 0.0}
CORRECTABILITY_MAP = {"Y": 1.0, "P": 0.5, "N": 0.0}
SPECIFICITY_MAP = {"specific": 1.0, "partial": 0.5, "generic": 0.0}
MUTABILITY_MAP = {"M": 1.0, "PS": 0.6, "PL": 0.4, "P": 0.5, "I": 0.0, "N": 0.0}


@dataclass(slots=True)
class ContestabilityBreakdown:
    verifiability: float
    correctability: float
    decision_sensitivity_visibility: float
    reason_specificity: float
    score: float


@dataclass(slots=True)
class FaithfulnessBreakdown:
    evidence_source: str
    reference_feature_coverage: float
    directional_agreement: float
    explanation_quality_signal: float
    score: float


@dataclass(slots=True)
class StabilityBreakdown:
    evidence_source: str
    neighborhood_consistency: float
    rank_consistency: float
    direction_consistency: float
    seed_consistency: float
    score: float


@dataclass(slots=True)
class ComprehensibilityBreakdown:
    reason_count_score: float
    feature_count_score: float
    reason_text_score: float
    recourse_simplicity_score: float
    score: float


@dataclass(slots=True)
class ActionabilityLegalityBreakdown:
    applicability: str
    feasibility: float | None
    legality: float | None
    minimality: float | None
    invalid_recourse_rate: float | None
    hard_failures: list[str] = field(default_factory=list)
    score: float | None = None


@dataclass(slots=True)
class UnifiedActionabilityBreakdown:
    advice_presence: float
    correction_support: float
    alf_component: float
    score: float


class ExplanationScorer:
    def __init__(self, corrections: CorrectionOptions | None = V1) -> None:
        """Scorer; corrections defaults to V1 (the published scoring, unchanged).

        Corrections read here:
        - faithfulness_quality_weight w: faithfulness =
          (0.45*coverage + 0.35*direction + w*quality) / (0.80 + w). v1 w = 0.20
          (denominator 1.0, the published formula, kept as a literal code path);
          v2 w = 0.0, so the self-reported quality signal (constant per model type
          for SHAP) no longer enters the score. The signal is still recorded in
          explanation_quality_signal.
        - cf_respect_nonactionable: counterfactual changes to taxonomy mutability
          "N" hard-fail like "I" (label "nonactionable_feature_changed").
        - persist_score_subcomponents: adds "faithfulness_quality_weight" to the
          faithfulness dict and "dsv_directionality", "dsv_threshold_visibility",
          "dsv_flip_validity" to the contestability dict (dict keys only; the
          breakdown dataclasses are unchanged).
        - common_faithfulness_reference: rule-family (Anchors) and
          counterfactual-family (DiCE/Wachter) features are scored as
          direction-neutral in faithfulness's directional term. An anchor rule
          claims no direction (the adapters' "increases_denial" label is kept for
          reasons and CSS). Counterfactual reasons take their signed_direction
          from the sign of the model's native contribution (v2
          counterfactual_reasons_from_changes; v1 ranking under native ranking),
          which is the common reference itself, so the term would be 1.0 by
          construction. Both families therefore get the same neutral 0.5 and the
          directional term only separates attribution-family explainers.
        The common faithfulness reference itself is injected upstream, in
        explainers/registry.generate_raw_explanation.
        """
        if corrections is None:
            corrections = V1
        if not isinstance(corrections, CorrectionOptions):
            raise TypeError(
                f"corrections must be a CorrectionOptions, got {type(corrections).__name__}"
            )
        self.corrections = corrections
        self._is_v1 = corrections.is_v1
        self._quality_weight = float(corrections.faithfulness_quality_weight)
        self._block_nonactionable = bool(corrections.cf_respect_nonactionable)
        self._persist_subcomponents = bool(corrections.persist_score_subcomponents)

    def score(self, explanation: CanonicalExplanation) -> dict[str, Any]:
        faithfulness = self._faithfulness(explanation)
        stability = self._stability(explanation)
        comprehensibility = self._comprehensibility(explanation)
        css = self._contestability(explanation)
        alf = self._actionability_legality(explanation)
        actionability = self._unified_actionability(explanation, alf)
        faithfulness_scores = asdict(faithfulness)
        css_scores = asdict(css)
        if self._persist_subcomponents:
            # Correction persist_score_subcomponents: extra dict keys only.
            faithfulness_scores["faithfulness_quality_weight"] = self._quality_weight
            directionality, threshold_visibility, flip_validity = (
                self._decision_sensitivity_components(explanation)
            )
            css_scores["dsv_directionality"] = directionality
            css_scores["dsv_threshold_visibility"] = threshold_visibility
            css_scores["dsv_flip_validity"] = flip_validity
        return {
            "faithfulness_score": faithfulness_scores,
            "stability_score": asdict(stability),
            "comprehensibility_score": asdict(comprehensibility),
            "contestability_support_score": css_scores,
            "actionability_legality_filter": asdict(alf),
            "unified_actionability_score": asdict(actionability),
        }

    def _faithfulness(self, explanation: CanonicalExplanation) -> FaithfulnessBreakdown:
        evidence = dict(explanation.metadata.get("faithfulness_evidence", {}))
        reference_contributions = dict(evidence.get("reference_signed_contributions", {}))

        if reference_contributions:
            coverage = self._reference_feature_coverage(
                explanation.referenced_features,
                reference_contributions,
            )
            if (
                self.corrections.common_faithfulness_reference
                and explanation.explanation_family in ("rule", "counterfactual")
            ):
                # Under the common reference an anchor rule ("x in R keeps the
                # decision") makes no claim about the sign of a feature's effect, yet
                # adapters label every anchor feature "increases_denial". Its features
                # are scored as direction-neutral here (match 0.5, or 1.0 when the
                # reference is exactly 0), so the direction term does not measure the
                # hard-coded label. Counterfactual (DiCE/Wachter) reasons copy their
                # signed_direction from the sign of the native contribution, i.e. from
                # this very reference, so their match would be 1.0 by construction;
                # they are scored direction-neutral too, which puts both non-attribution
                # families on the same 0.5. Faithfulness only: reasons and CSS keep
                # signed_direction.
                directional = self._directional_agreement(
                    explanation.referenced_features,
                    reference_contributions,
                    neutral_directions=True,
                )
            else:
                directional = self._directional_agreement(
                    explanation.referenced_features,
                    reference_contributions,
                )
            evidence_source = str(evidence.get("source", "provided_reference_contributions"))
        else:
            coverage = self._reason_feature_coverage(explanation)
            directional = self._structural_directional_agreement(explanation)
            evidence_source = "structural_proxy"

        quality_signal = self._faithfulness_quality_signal(explanation, evidence)
        quality_weight = self._quality_weight
        if self._is_v1 or quality_weight == 0.20:
            score = 0.45 * coverage + 0.35 * directional + 0.20 * quality_signal
        else:
            # Correction faithfulness_quality_weight: renormalized weights.
            score = (
                0.45 * coverage + 0.35 * directional + quality_weight * quality_signal
            ) / (0.80 + quality_weight)

        return FaithfulnessBreakdown(
            evidence_source=evidence_source,
            reference_feature_coverage=coverage,
            directional_agreement=directional,
            explanation_quality_signal=quality_signal,
            score=score,
        )

    def _stability(self, explanation: CanonicalExplanation) -> StabilityBreakdown:
        evidence = dict(explanation.metadata.get("stability_evidence", {}))

        neighborhood = self._clip01(
            float(
                evidence.get(
                    "neighborhood_consistency",
                    self._structural_neighborhood_consistency(explanation),
                )
            )
        )
        rank = self._clip01(
            float(
                evidence.get(
                    "rank_consistency",
                    self._structural_rank_consistency(explanation),
                )
            )
        )
        direction = self._clip01(
            float(
                evidence.get(
                    "direction_consistency",
                    self._structural_direction_consistency(explanation),
                )
            )
        )
        seed = self._clip01(
            float(
                evidence.get(
                    "seed_consistency",
                    self._structural_seed_consistency(explanation),
                )
            )
        )
        score = 0.35 * neighborhood + 0.25 * rank + 0.25 * direction + 0.15 * seed
        return StabilityBreakdown(
            evidence_source=str(evidence.get("source", "structural_proxy")),
            neighborhood_consistency=neighborhood,
            rank_consistency=rank,
            direction_consistency=direction,
            seed_consistency=seed,
            score=score,
        )

    def _comprehensibility(
        self,
        explanation: CanonicalExplanation,
    ) -> ComprehensibilityBreakdown:
        reason_count = len(explanation.principal_reasons)
        feature_count = len(explanation.referenced_features)
        change_count = len(explanation.counterfactual_changes)

        reason_count_score = self._count_simplicity(
            count=reason_count,
            ideal=2,
            soft_limit=4,
            hard_span=4,
        )
        feature_count_score = self._count_simplicity(
            count=feature_count,
            ideal=3,
            soft_limit=5,
            hard_span=5,
        )
        reason_text_score = self._reason_text_score(explanation)
        recourse_simplicity_score = self._recourse_simplicity_score(change_count)

        score = (
            0.35 * reason_count_score
            + 0.25 * feature_count_score
            + 0.20 * reason_text_score
            + 0.20 * recourse_simplicity_score
        )
        return ComprehensibilityBreakdown(
            reason_count_score=reason_count_score,
            feature_count_score=feature_count_score,
            reason_text_score=reason_text_score,
            recourse_simplicity_score=recourse_simplicity_score,
            score=score,
        )

    def _contestability(self, explanation: CanonicalExplanation) -> ContestabilityBreakdown:
        referenced = explanation.referenced_features
        weights = self._importance_weights(referenced)

        verifiability = sum(
            weight * VERIFICATION_MAP.get(feature.taxonomy.verifiability, 0.0)
            for feature, weight in zip(referenced, weights, strict=False)
            if feature.taxonomy is not None
        )
        correctability = sum(
            weight * CORRECTABILITY_MAP.get(feature.taxonomy.correctability, 0.0)
            for feature, weight in zip(referenced, weights, strict=False)
            if feature.taxonomy is not None
        )

        directionality, threshold_visibility, flip_validity = (
            self._decision_sensitivity_components(explanation)
        )
        sensitivity = 0.4 * directionality + 0.3 * threshold_visibility + 0.3 * flip_validity

        specificity = 0.0
        if explanation.principal_reasons:
            specificity = sum(
                SPECIFICITY_MAP.get(reason.specificity_level, 0.0)
                for reason in explanation.principal_reasons
            ) / len(explanation.principal_reasons)

        score = (
            0.30 * verifiability
            + 0.30 * correctability
            + 0.25 * sensitivity
            + 0.15 * specificity
        )
        return ContestabilityBreakdown(
            verifiability=verifiability,
            correctability=correctability,
            decision_sensitivity_visibility=sensitivity,
            reason_specificity=specificity,
            score=score,
        )

    def _actionability_legality(
        self,
        explanation: CanonicalExplanation,
    ) -> ActionabilityLegalityBreakdown:
        changes = explanation.counterfactual_changes
        if not changes:
            return ActionabilityLegalityBreakdown(
                applicability="not_applicable",
                feasibility=None,
                legality=None,
                minimality=None,
                invalid_recourse_rate=None,
                hard_failures=[],
                score=None,
            )

        feasibility = sum(self._feasibility(change) for change in changes) / len(changes)
        legality = self._legality(changes)
        minimality = self._minimality(changes)
        hard_failures = self._hard_failures(changes)
        invalid_change_names = {failure.split(":", 1)[0] for failure in hard_failures}
        invalid_rate = len(invalid_change_names) / len(changes)
        score = feasibility * legality * minimality

        return ActionabilityLegalityBreakdown(
            applicability="applicable",
            feasibility=feasibility,
            legality=legality,
            minimality=minimality,
            invalid_recourse_rate=invalid_rate,
            hard_failures=hard_failures,
            score=score,
        )

    def _unified_actionability(
        self,
        explanation: CanonicalExplanation,
        alf: ActionabilityLegalityBreakdown,
    ) -> UnifiedActionabilityBreakdown:
        advice_presence = 1.0 if explanation.counterfactual_changes else 0.25
        correction_support = self._correction_support(explanation.referenced_features)
        alf_component = alf.score if alf.score is not None else 0.0
        score = 0.20 * advice_presence + 0.50 * alf_component + 0.30 * correction_support
        return UnifiedActionabilityBreakdown(
            advice_presence=advice_presence,
            correction_support=correction_support,
            alf_component=alf_component,
            score=score,
        )

    def _importance_weights(self, features: list[ReferencedFeature]) -> list[float]:
        if not features:
            return []
        raw = [
            abs(feature.importance_weight) if feature.importance_weight is not None else 1.0
            for feature in features
        ]
        total = sum(raw)
        if total == 0:
            return [1.0 / len(features)] * len(features)
        return [value / total for value in raw]

    def _decision_sensitivity_components(
        self,
        explanation: CanonicalExplanation,
    ) -> tuple[float, float, float]:
        """(directionality, threshold_visibility, flip_validity) of CSS's DSV term."""
        directionality = 1.0 if any(
            reason.signed_direction != "neutral" for reason in explanation.principal_reasons
        ) else 0.0
        threshold_visibility = self._threshold_visibility(explanation)
        flip_validity = self._flip_validity(explanation.counterfactual_changes)
        return directionality, threshold_visibility, flip_validity

    @staticmethod
    def _is_nonactionable(change: CounterfactualChange) -> bool:
        return change.taxonomy is not None and change.taxonomy.mutability == "N"

    def _threshold_visibility(self, explanation: CanonicalExplanation) -> float:
        if explanation.explanation_family == "counterfactual":
            return 1.0 if explanation.counterfactual_changes else 0.0

        has_temporal_group = any(
            reason.grouped_reason_type == "compressed_temporal"
            for reason in explanation.principal_reasons
        )
        return 0.8 if has_temporal_group else 0.5 if explanation.principal_reasons else 0.0

    def _flip_validity(self, changes: list[CounterfactualChange]) -> float:
        if not changes:
            return 0.0
        valid = [1.0 if change.validity_flip else 0.0 for change in changes]
        return sum(valid) / len(valid)

    def _feasibility(self, change: CounterfactualChange) -> float:
        if change.taxonomy is not None:
            mutability = MUTABILITY_MAP.get(change.taxonomy.mutability, 0.0)
        elif change.legality_flags.immutable_feature:
            mutability = MUTABILITY_MAP["I"]
        elif change.legality_flags.lender_controlled_feature:
            mutability = MUTABILITY_MAP["P"]
        else:
            mutability = 0.7
        effort = change.effort_score if change.effort_score is not None else 0.5
        realism = change.realism_score if change.realism_score is not None else 0.5
        return 0.5 * mutability + 0.25 * effort + 0.25 * realism

    def _legality(self, changes: list[CounterfactualChange]) -> float:
        count = len(changes)
        p_sensitive = sum(
            1.0 if change.legality_flags.direct_sensitive_feature else 0.0
            for change in changes
        ) / count
        p_proxy = sum(
            1.0 if change.legality_flags.proxy_sensitive_feature else 0.0
            for change in changes
        ) / count
        p_lender = sum(
            1.0 if change.legality_flags.lender_controlled_feature else 0.0
            for change in changes
        ) / count
        p_nonreason = sum(
            1.0 if change.legality_flags.low_principal_reason_feature else 0.0
            for change in changes
        ) / count
        penalty = 0.6 * p_sensitive + 0.2 * p_proxy + 0.1 * p_lender + 0.1 * p_nonreason
        return max(0.0, 1.0 - penalty)

    def _minimality(self, changes: list[CounterfactualChange]) -> float:
        count = len(changes)
        sparsity = max(0.0, 1.0 - ((count - 1) / 4.0))
        proximities = []
        for change in changes:
            if isinstance(change.delta, (int, float)):
                proximities.append(1.0 / (1.0 + math.log1p(abs(change.delta))))
            else:
                proximities.append(0.5)
        proximity = sum(proximities) / len(proximities)
        return 0.5 * sparsity + 0.5 * proximity

    def _hard_failures(self, changes: list[CounterfactualChange]) -> list[str]:
        failures: list[str] = []
        for change in changes:
            if change.legality_flags.direct_sensitive_feature:
                failures.append(f"{change.feature_name}:direct_sensitive_feature_changed")
            if self._block_nonactionable and self._is_nonactionable(change):
                # Correction cf_respect_nonactionable: mutability "N" hard-fails like
                # "I" (whether or not the canonicalizer already set the immutable flag).
                failures.append(f"{change.feature_name}:nonactionable_feature_changed")
            elif change.legality_flags.immutable_feature:
                failures.append(f"{change.feature_name}:immutable_feature_changed")
            if (change.realism_score or 0.5) < 0.2:
                failures.append(f"{change.feature_name}:clearly_off_manifold_value")
            if change.legality_flags.low_principal_reason_feature:
                failures.append(f"{change.feature_name}:primarily_low_principal_reason_feature")
            if change.legality_flags.lender_controlled_feature:
                failures.append(f"{change.feature_name}:subject_cannot_reasonably_enact_change")
        return failures

    def _correction_support(self, features: list[ReferencedFeature]) -> float:
        weights = self._importance_weights(features)
        if not weights:
            return 0.0
        return sum(
            weight * CORRECTABILITY_MAP.get(feature.taxonomy.correctability, 0.0)
            for feature, weight in zip(features, weights, strict=False)
            if feature.taxonomy is not None
        )

    def _reference_feature_coverage(
        self,
        features: list[ReferencedFeature],
        reference_contributions: dict[str, Any],
    ) -> float:
        if not reference_contributions:
            return 0.0
        explained = {feature.feature_name for feature in features}
        denominator = sum(abs(float(value)) for value in reference_contributions.values())
        if denominator <= 0:
            return 0.0
        numerator = sum(
            abs(float(value))
            for feature_name, value in reference_contributions.items()
            if feature_name in explained
        )
        return numerator / denominator

    def _directional_agreement(
        self,
        features: list[ReferencedFeature],
        reference_contributions: dict[str, Any],
        neutral_directions: bool = False,
    ) -> float:
        if not reference_contributions:
            return 0.0
        explained = {
            feature.feature_name: "neutral" if neutral_directions else feature.signed_direction
            for feature in features
            if feature.feature_name in reference_contributions
        }
        if not explained:
            return 0.0
        denominator = sum(
            abs(float(reference_contributions[feature_name]))
            for feature_name in explained
        )
        if denominator <= 0:
            return 0.0
        numerator = 0.0
        for feature_name, signed_direction in explained.items():
            reference_direction = self._signed_direction_from_reference(
                float(reference_contributions[feature_name])
            )
            weight = abs(float(reference_contributions[feature_name]))
            numerator += weight * self._direction_match_score(
                explanation_direction=signed_direction,
                reference_direction=reference_direction,
            )
        return numerator / denominator

    def _reason_feature_coverage(self, explanation: CanonicalExplanation) -> float:
        if not explanation.referenced_features:
            return 0.0
        supported = {
            feature_name
            for reason in explanation.principal_reasons
            for feature_name in reason.supporting_features
        }
        explained = {feature.feature_name for feature in explanation.referenced_features}
        return len(explained & supported) / len(explained)

    def _structural_directional_agreement(self, explanation: CanonicalExplanation) -> float:
        if not explanation.referenced_features:
            return 0.0
        supporting_directions = {
            feature_name: reason.signed_direction
            for reason in explanation.principal_reasons
            for feature_name in reason.supporting_features
        }
        weights = self._importance_weights(explanation.referenced_features)
        total = 0.0
        for feature, weight in zip(explanation.referenced_features, weights, strict=False):
            reason_direction = supporting_directions.get(feature.feature_name)
            if reason_direction is None:
                total += weight * 0.5
                continue
            total += weight * self._direction_match_score(
                explanation_direction=feature.signed_direction,
                reference_direction=reason_direction,
            )
        return total

    def _faithfulness_quality_signal(
        self,
        explanation: CanonicalExplanation,
        evidence: dict[str, Any],
    ) -> float:
        local_quality = evidence.get("local_quality_score")
        if isinstance(local_quality, (int, float)):
            return self._clip01(float(local_quality))

        precision = explanation.metadata.get("confidence_or_precision")
        if isinstance(precision, (int, float)):
            return self._clip01(float(precision))

        if explanation.counterfactual_changes:
            validities = [
                1.0 if change.validity_flip else 0.0
                for change in explanation.counterfactual_changes
            ]
            return sum(validities) / len(validities)

        return 0.5

    def _structural_neighborhood_consistency(self, explanation: CanonicalExplanation) -> float:
        referenced_count = len(explanation.referenced_features)
        reason_count = len(explanation.principal_reasons)
        grouped_bonus = 0.08 * sum(
            1
            for reason in explanation.principal_reasons
            if reason.grouped_reason_type == "compressed_temporal"
        )
        family_base = {
            "attribution": 0.78,
            "rule": 0.82,
            "counterfactual": 0.68,
        }.get(explanation.explanation_family, 0.7)
        size_penalty = 0.05 * max(0, referenced_count - 4) + 0.04 * max(0, reason_count - 3)
        return self._clip01(family_base + grouped_bonus - size_penalty)

    def _structural_rank_consistency(self, explanation: CanonicalExplanation) -> float:
        referenced = explanation.referenced_features
        if not referenced:
            return 0.0
        if any(feature.rank is None for feature in referenced):
            return 0.6

        expected_ranks = list(range(1, len(referenced) + 1))
        actual_ranks = [int(feature.rank) for feature in referenced]
        displacement = sum(
            abs(actual - expected)
            for actual, expected in zip(actual_ranks, expected_ranks, strict=False)
        )
        denominator = max(1.0, len(referenced) * (len(referenced) - 1) / 2)
        return self._clip01(1.0 - (displacement / denominator))

    def _structural_direction_consistency(self, explanation: CanonicalExplanation) -> float:
        if not explanation.referenced_features:
            return 0.0
        supporting_directions = {
            feature_name: reason.signed_direction
            for reason in explanation.principal_reasons
            for feature_name in reason.supporting_features
        }
        weights = self._importance_weights(explanation.referenced_features)
        total = 0.0
        for feature, weight in zip(explanation.referenced_features, weights, strict=False):
            reason_direction = supporting_directions.get(feature.feature_name)
            if reason_direction is None:
                total += 0.5 * weight
                continue
            total += weight * self._direction_match_score(
                explanation_direction=feature.signed_direction,
                reference_direction=reason_direction,
            )
        return total

    def _structural_seed_consistency(self, explanation: CanonicalExplanation) -> float:
        if explanation.seed is not None:
            return 0.6 if explanation.explainer_id == "lime" else 0.7
        if explanation.explanation_family == "counterfactual":
            return 0.72
        if explanation.explanation_family == "rule":
            return 0.85
        return 0.88

    def _signed_direction_from_reference(self, value: float) -> str:
        if value > 0:
            return "increases_denial"
        if value < 0:
            return "decreases_denial"
        return "neutral"

    def _direction_match_score(
        self,
        explanation_direction: str,
        reference_direction: str,
    ) -> float:
        if explanation_direction == reference_direction:
            return 1.0
        if explanation_direction == "neutral" or reference_direction == "neutral":
            return 0.5
        return 0.0

    def _clip01(self, value: float) -> float:
        return max(0.0, min(1.0, value))

    def _count_simplicity(
        self,
        count: int,
        ideal: int,
        soft_limit: int,
        hard_span: int,
    ) -> float:
        if count <= ideal:
            return 1.0
        if count <= soft_limit:
            return max(0.0, 1.0 - 0.2 * (count - ideal))
        overflow = count - soft_limit
        return max(0.0, 0.6 - (overflow / max(1, hard_span)))

    def _reason_text_score(self, explanation: CanonicalExplanation) -> float:
        if not explanation.principal_reasons:
            return 0.0
        scores: list[float] = []
        for reason in explanation.principal_reasons:
            word_count = len(reason.reason_text.split())
            grouped_bonus = 0.1 if reason.grouped_reason_type == "compressed_temporal" else 0.0
            if word_count <= 6:
                base = 1.0
            elif word_count <= 10:
                base = 0.85
            elif word_count <= 14:
                base = 0.65
            else:
                base = 0.4
            scores.append(self._clip01(base + grouped_bonus))
        return sum(scores) / len(scores)

    def _recourse_simplicity_score(self, change_count: int) -> float:
        if change_count == 0:
            return 1.0
        return self._count_simplicity(
            count=change_count,
            ideal=1,
            soft_limit=2,
            hard_span=3,
        )
