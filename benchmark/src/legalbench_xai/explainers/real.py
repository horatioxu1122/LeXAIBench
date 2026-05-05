import math
import random
from dataclasses import dataclass
import os
from typing import Any

from ..models.real import SklearnLogisticRegressionModel, VectorizedClassifierModel
from ..models.stubs import FeatureProfile, RowPrediction


@dataclass(slots=True)
class CounterfactualCandidate:
    row: dict[str, Any]
    changes: dict[str, dict[str, Any]]
    score: float
    validity_flip: bool
    cost: float
    realism_score: float
    effort_score: float


@dataclass(slots=True)
class AnchorCondition:
    feature_name: str
    condition_type: str
    threshold: float | None = None
    category_value: Any = None
    condition_text: str = ""


def generate_real_shap_explanation(
    model: VectorizedClassifierModel,
    row: dict[str, Any],
    prediction: RowPrediction,
) -> dict[str, Any]:
    import numpy as np

    if isinstance(model, SklearnLogisticRegressionModel):
        shap_values = model.exact_shap_values(row)
        model.get_shap_explainer()
    else:
        vector = model.vectorize_row(row)
        explanation = model.get_shap_explainer()(vector)
        shap_values = _positive_class_values(
            values=explanation.values,
            feature_count=len(model.encoded_feature_names),
            np_module=np,
        )

    aggregated: dict[str, float] = {}
    for encoded_name, value in zip(
        model.encoded_feature_names,
        shap_values,
        strict=False,
    ):
        base_name = model.base_feature_name(encoded_name)
        aggregated[base_name] = aggregated.get(base_name, 0.0) + float(value)

    ranked = sorted(
        aggregated.items(),
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    top_values = ranked[:6]
    quality_score = {
        "linear_exact": 1.0,
        "linear": 1.0,
        "tree": 0.97,
        "generic": 0.88,
    }.get(model.shap_mode, 0.88)

    return {
        "instance_id": prediction.instance_id,
        "model_id": model.model_id,
        "prediction": _prediction_block(prediction),
        "shap_values": [
            {
                "feature_name": feature_name,
                "phi": value,
                "raw_value": row.get(feature_name),
            }
            for feature_name, value in top_values
        ],
        "metadata": {
            "generated_from": model.backend_implementation,
            "raw_family": "attribution",
            "shap_mode": model.shap_mode,
            "background_rows": int(len(model.background_matrix)),
            "encoded_feature_count": int(len(model.encoded_feature_names)),
            "faithfulness_evidence": {
                "source": "real_shap_attributions",
                "local_quality_score": quality_score,
                "reference_signed_contributions": aggregated,
            },
        },
    }


def generate_real_lime_explanation(
    model: VectorizedClassifierModel,
    row: dict[str, Any],
    prediction: RowPrediction,
    seed: int = 0,
) -> dict[str, Any]:
    vector = model.vectorize_row(row)
    explainer = model.get_lime_explainer(seed=seed)
    kernel_width = 0.75 * max(1, len(model.encoded_feature_names)) ** 0.5
    configured_num_samples = os.getenv("LEGALBENCH_LIME_NUM_SAMPLES", "5000")
    try:
        num_samples = max(100, int(configured_num_samples))
    except ValueError:
        num_samples = 5000
    explanation = explainer.explain_instance(
        data_row=vector[0],
        predict_fn=model.classifier.predict_proba,
        labels=(1,),
        num_features=len(model.encoded_feature_names),
        num_samples=num_samples,
    )

    aggregated: dict[str, float] = {}
    for feature_index, weight in explanation.local_exp.get(1, []):
        encoded_name = model.encoded_feature_names[feature_index]
        base_name = model.base_feature_name(encoded_name)
        aggregated[base_name] = aggregated.get(base_name, 0.0) + float(weight)

    ranked = sorted(
        aggregated.items(),
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    top_values = ranked[:6]

    return {
        "instance_id": prediction.instance_id,
        "model_id": model.model_id,
        "seed": seed,
        "prediction": _prediction_block(prediction),
        "local_explanation": [
            {
                "feature_name": feature_name,
                "weight": value,
                "raw_value": row.get(feature_name),
            }
            for feature_name, value in top_values
        ],
        "metadata": {
            "generated_from": model.backend_implementation,
            "raw_family": "attribution",
            "lime_settings": {
                "kernel_width": kernel_width,
                "num_samples": num_samples,
                "discretize_continuous": False,
                "background_rows": int(len(model.background_matrix)),
            },
            "faithfulness_evidence": {
                "source": "real_lime_local_surrogate",
                "local_quality_score": _clipped_quality_score(explanation.score),
                "reference_signed_contributions": aggregated,
            },
        },
    }


def generate_real_leave_one_out_explanation(
    model: VectorizedClassifierModel,
    row: dict[str, Any],
    prediction: RowPrediction,
) -> dict[str, Any]:
    contributions: dict[str, float] = {}
    for feature_name in model.dataset_bundle.included_feature_names:
        replacement_value = model.feature_fill_values.get(feature_name)
        modified_row = dict(row)
        modified_row[feature_name] = replacement_value
        modified_prediction = model.predict_row(
            modified_row,
            instance_id=f"{prediction.instance_id}-loo-{feature_name}",
        )
        contributions[feature_name] = float(prediction.score - modified_prediction.score)

    ranked = sorted(
        contributions.items(),
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    top_values = ranked[:6]

    return {
        "instance_id": prediction.instance_id,
        "model_id": model.model_id,
        "prediction": _prediction_block(prediction),
        "local_explanation": [
            {
                "feature_name": feature_name,
                "weight": value,
                "raw_value": row.get(feature_name),
            }
            for feature_name, value in top_values
        ],
        "metadata": {
            "generated_from": model.backend_implementation,
            "raw_family": "attribution",
            "leave_one_out_settings": {
                "reference_policy": "training_fill_value_replacement",
                "evaluated_feature_count": int(
                    len(model.dataset_bundle.included_feature_names)
                ),
            },
            "faithfulness_evidence": {
                "source": "leave_one_out_feature_ablation",
                "local_quality_score": 0.92,
                "reference_signed_contributions": contributions,
            },
        },
    }


def generate_real_anchor_explanation(
    model: VectorizedClassifierModel,
    row: dict[str, Any],
    prediction: RowPrediction,
    generation_mode: str = "full",
) -> dict[str, Any]:
    search_budget = _anchor_search_budget(generation_mode)
    conditions, precision, coverage, contributions = _search_anchor_conditions(
        model=model,
        row=row,
        prediction=prediction,
        ranked_feature_limit=search_budget["ranked_feature_limit"],
        neighborhood_size=search_budget["neighborhood_size"],
        max_rule_length=search_budget["max_rule_length"],
    )

    selected_references = {
        condition.feature_name: contributions.get(condition.feature_name, 0.0)
        for condition in conditions
    }
    if not selected_references:
        ranked = sorted(
            contributions.items(),
            key=lambda item: abs(item[1]),
            reverse=True,
        )[:2]
        selected_references = {feature_name: weight for feature_name, weight in ranked}

    return {
        "instance_id": prediction.instance_id,
        "model_id": model.model_id,
        "prediction": _prediction_block(prediction),
        "anchor_conditions": [
            {
                "feature_name": condition.feature_name,
                "raw_value": row.get(condition.feature_name),
                "condition_text": condition.condition_text,
            }
            for condition in conditions
        ],
        "precision": precision,
        "coverage": coverage,
        "metadata": {
            "generated_from": model.backend_implementation,
            "raw_family": "rule",
            "anchor_search": {
                "max_rule_length": search_budget["max_rule_length"],
                "neighborhood_size": search_budget["neighborhood_size"],
                "generation_mode": generation_mode,
            },
            "faithfulness_evidence": {
                "source": "real_anchor_local_precision",
                "local_quality_score": precision,
                "reference_signed_contributions": selected_references,
            },
        },
    }


def generate_real_counterfactual_explanation(
    *,
    model: VectorizedClassifierModel,
    row: dict[str, Any],
    prediction: RowPrediction,
    explainer_id: str,
    generation_mode: str = "full",
) -> dict[str, Any]:
    search_budget = _counterfactual_search_budget(explainer_id, generation_mode)
    candidates = _search_counterfactual_candidates(
        model=model,
        row=row,
        prediction=prediction,
        explainer_id=explainer_id,
        max_depth=search_budget["max_depth"],
        beam_width=search_budget["beam_width"],
        feature_limit=search_budget["feature_limit"],
    )
    if explainer_id == "dice":
        selected = _select_diverse_candidates(candidates, limit=3)
    else:
        selected = candidates[:1]

    if not selected and candidates:
        selected = candidates[:1]
    if not selected:
        selected = [
            CounterfactualCandidate(
                row=dict(row),
                changes={},
                score=prediction.score,
                validity_flip=False,
                cost=0.0,
                realism_score=0.0,
                effort_score=0.0,
            )
        ]

    contributions = model.original_feature_contributions(row)
    ranked_features = sorted(
        contributions.items(),
        key=lambda item: abs(item[1]),
        reverse=True,
    )[:6]
    successful = sum(1 for candidate in selected if candidate.validity_flip)
    quality_score = successful / max(1, len(selected))

    return {
        "instance_id": prediction.instance_id,
        "model_id": model.model_id,
        "seed": 0 if explainer_id == "dice" else None,
        "prediction": _prediction_block(prediction),
        "feature_importances": [
            {
                "feature_name": feature_name,
                "weight": weight,
                "raw_value": row.get(feature_name),
            }
            for feature_name, weight in ranked_features
        ],
        "counterfactuals": [
            {
                "changes": candidate.changes,
                "validity_flip": candidate.validity_flip,
                "realism_score": candidate.realism_score,
                "effort_score": candidate.effort_score,
            }
            for candidate in selected
        ],
        "metadata": {
            "generated_from": model.backend_implementation,
            "raw_family": "counterfactual",
            "counterfactual_method": explainer_id,
            "search_backend": (
                "diverse_model_validated_search"
                if explainer_id == "dice"
                else "sparse_model_validated_search"
            ),
            "generation_mode": generation_mode,
            "faithfulness_evidence": {
                "source": "model_validated_counterfactual_search",
                "local_quality_score": quality_score,
            },
        },
    }


def _search_counterfactual_candidates(
    *,
    model: VectorizedClassifierModel,
    row: dict[str, Any],
    prediction: RowPrediction,
    explainer_id: str,
    max_depth: int,
    beam_width: int,
    feature_limit: int | None,
) -> list[CounterfactualCandidate]:
    profiles = model.heuristic_helper.profiles
    allowed_profiles = {
        feature_name: profile
        for feature_name, profile in profiles.items()
        if _feature_allowed(profile, explainer_id)
    }
    if not allowed_profiles:
        return []

    if feature_limit is not None:
        ranked_features = [
            feature_name
            for feature_name, _weight in sorted(
                model.original_feature_contributions(row).items(),
                key=lambda item: abs(item[1]),
                reverse=True,
            )
            if feature_name in allowed_profiles
        ][:feature_limit]
        allowed_profiles = {
            feature_name: allowed_profiles[feature_name]
            for feature_name in ranked_features
        }
        if not allowed_profiles:
            return []
    pending = [
        CounterfactualCandidate(
            row=dict(row),
            changes={},
            score=prediction.score,
            validity_flip=False,
            cost=0.0,
            realism_score=0.0,
            effort_score=0.0,
        )
    ]
    valid: list[CounterfactualCandidate] = []
    explored: dict[tuple[tuple[str, str], ...], float] = {}

    for _depth in range(1, max_depth + 1):
        next_generation: list[CounterfactualCandidate] = []
        for candidate in pending:
            used_features = set(candidate.changes)
            for feature_name, profile in allowed_profiles.items():
                if feature_name in used_features:
                    continue
                current_value = candidate.row.get(feature_name)
                for proposed_value in _candidate_values(profile, current_value, explainer_id):
                    trial_row = dict(candidate.row)
                    trial_row[feature_name] = proposed_value
                    trial_prediction = model.predict_row(
                        row=trial_row,
                        instance_id=prediction.instance_id,
                    )
                    changes = dict(candidate.changes)
                    changes[feature_name] = {
                        "from_value": row.get(feature_name),
                        "to_value": proposed_value,
                    }
                    cost = _total_change_cost(changes, row, profiles)
                    realism_score = _aggregate_realism_score(changes, row, profiles)
                    effort_score = _aggregate_effort_score(changes, row, profiles)
                    trial_candidate = CounterfactualCandidate(
                        row=trial_row,
                        changes=changes,
                        score=trial_prediction.score,
                        validity_flip=trial_prediction.score < model.threshold,
                        cost=cost,
                        realism_score=realism_score,
                        effort_score=effort_score,
                    )
                    signature = _candidate_signature(trial_candidate)
                    best_score = explored.get(signature)
                    objective = _counterfactual_objective(trial_candidate, explainer_id)
                    if best_score is not None and objective >= best_score:
                        continue
                    explored[signature] = objective
                    next_generation.append(trial_candidate)
                    if trial_candidate.validity_flip:
                        valid.append(trial_candidate)

        pending = _prune_generation(next_generation, explainer_id, limit=beam_width)
        if explainer_id == "wachter" and valid:
            break

    ranked_valid = sorted(
        valid,
        key=lambda item: (item.cost, len(item.changes), item.score),
    )
    if ranked_valid:
        return ranked_valid

    return _prune_generation(pending, explainer_id, limit=3)


def _anchor_search_budget(generation_mode: str) -> dict[str, int]:
    if generation_mode == "stability_probe":
        return {
            "ranked_feature_limit": 4,
            "neighborhood_size": 16,
            "max_rule_length": 2,
        }
    return {
        "ranked_feature_limit": 6,
        "neighborhood_size": 48,
        "max_rule_length": 3,
    }


def _counterfactual_search_budget(
    explainer_id: str,
    generation_mode: str,
) -> dict[str, int | None]:
    if generation_mode == "stability_probe":
        if explainer_id == "dice":
            return {
                "max_depth": 2,
                "beam_width": 5,
                "feature_limit": 5,
            }
        return {
            "max_depth": 1,
            "beam_width": 4,
            "feature_limit": 4,
        }
    if explainer_id == "dice":
        return {
            "max_depth": 3,
            "beam_width": 12,
            "feature_limit": None,
        }
    return {
        "max_depth": 2,
        "beam_width": 8,
        "feature_limit": None,
    }


def _select_diverse_candidates(
    candidates: list[CounterfactualCandidate],
    limit: int,
) -> list[CounterfactualCandidate]:
    if len(candidates) <= limit:
        return candidates
    selected = [candidates[0]]
    remaining = candidates[1:]
    while remaining and len(selected) < limit:
        best_index = 0
        best_score = -1.0
        for index, candidate in enumerate(remaining):
            candidate_features = set(candidate.changes)
            novelty = min(
                len(candidate_features.symmetric_difference(set(existing.changes)))
                for existing in selected
            )
            score = novelty - 0.1 * candidate.cost
            if score > best_score:
                best_score = score
                best_index = index
        selected.append(remaining.pop(best_index))
    return selected


def _prune_generation(
    candidates: list[CounterfactualCandidate],
    explainer_id: str,
    limit: int,
) -> list[CounterfactualCandidate]:
    ranked = sorted(
        candidates,
        key=lambda item: (
            0 if item.validity_flip else 1,
            _counterfactual_objective(item, explainer_id),
        ),
    )
    return ranked[:limit]


def _counterfactual_objective(candidate: CounterfactualCandidate, explainer_id: str) -> float:
    if candidate.validity_flip:
        return candidate.cost + 0.05 * len(candidate.changes)
    lambda_cost = 0.18 if explainer_id == "dice" else 0.26
    sparsity_penalty = 0.03 if explainer_id == "dice" else 0.06
    return candidate.score + lambda_cost * candidate.cost + sparsity_penalty * len(candidate.changes)


def _candidate_signature(candidate: CounterfactualCandidate) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            (feature_name, repr(change["to_value"]))
            for feature_name, change in candidate.changes.items()
        )
    )


def _feature_allowed(profile: FeatureProfile, explainer_id: str) -> bool:
    if profile.mutability == "I":
        return False
    if profile.sensitivity == "S":
        return False
    if explainer_id == "wachter" and profile.sensitivity == "P":
        return False
    if explainer_id == "wachter" and profile.mutability == "P":
        return False
    return True


def _candidate_values(
    profile: FeatureProfile,
    current_value: Any,
    explainer_id: str,
) -> list[Any]:
    if current_value is None:
        return []

    if profile.is_numeric:
        return _numeric_candidate_values(profile, current_value, explainer_id)

    ordered_categories = sorted(
        profile.category_risk.items(),
        key=lambda item: item[1],
    )
    values = [
        _coerce_like(current_value, value)
        for value, _risk in ordered_categories
        if value != str(current_value)
    ]
    if explainer_id == "wachter":
        return values[:2]
    return values[:4]


def _numeric_candidate_values(
    profile: FeatureProfile,
    current_value: Any,
    explainer_id: str,
) -> list[Any]:
    if profile.numeric_min is None or profile.numeric_max is None:
        return []
    current_numeric = float(current_value)
    targets: list[float] = []
    if profile.safer_value is not None:
        targets.append(float(profile.safer_value))
    targets.append(profile.numeric_min if profile.risk_direction >= 0 else profile.numeric_max)
    if profile.positive_mean is not None:
        targets.append(float(profile.positive_mean))

    unique_targets = []
    seen = set()
    for target in targets:
        clipped = max(profile.numeric_min, min(profile.numeric_max, target))
        rounded = round(clipped, 8)
        if rounded in seen or math.isclose(rounded, current_numeric, rel_tol=1e-9, abs_tol=1e-9):
            continue
        seen.add(rounded)
        unique_targets.append(clipped)

    fractions = [0.5, 1.0] if explainer_id == "wachter" else [0.34, 0.67, 1.0]
    candidates: list[Any] = []
    for target in unique_targets:
        for fraction in fractions:
            interpolated = current_numeric + fraction * (target - current_numeric)
            clipped = max(profile.numeric_min, min(profile.numeric_max, interpolated))
            coerced = _coerce_numeric_like(current_value, clipped)
            if coerced != current_value and coerced not in candidates:
                candidates.append(coerced)
    return candidates


def _total_change_cost(
    changes: dict[str, dict[str, Any]],
    original_row: dict[str, Any],
    profiles: dict[str, FeatureProfile],
) -> float:
    return sum(
        _change_cost(
            profile=profiles[feature_name],
            from_value=original_row.get(feature_name),
            to_value=change["to_value"],
        )
        for feature_name, change in changes.items()
    )


def _aggregate_realism_score(
    changes: dict[str, dict[str, Any]],
    original_row: dict[str, Any],
    profiles: dict[str, FeatureProfile],
) -> float:
    if not changes:
        return 0.0
    values = [
        _change_realism_score(
            profile=profiles[feature_name],
            from_value=original_row.get(feature_name),
            to_value=change["to_value"],
        )
        for feature_name, change in changes.items()
    ]
    return sum(values) / len(values)


def _aggregate_effort_score(
    changes: dict[str, dict[str, Any]],
    original_row: dict[str, Any],
    profiles: dict[str, FeatureProfile],
) -> float:
    if not changes:
        return 0.0
    values = [
        _change_effort_score(
            profile=profiles[feature_name],
            from_value=original_row.get(feature_name),
            to_value=change["to_value"],
        )
        for feature_name, change in changes.items()
    ]
    return sum(values) / len(values)


def _change_cost(profile: FeatureProfile, from_value: Any, to_value: Any) -> float:
    distance = _normalized_distance(profile, from_value, to_value)
    mutability_penalty = {
        "M": 0.45,
        "PS": 0.65,
        "PL": 0.85,
        "P": 1.05,
        "I": 10.0,
    }.get(profile.mutability, 0.8)
    return max(0.05, mutability_penalty * max(0.1, distance))


def _change_effort_score(profile: FeatureProfile, from_value: Any, to_value: Any) -> float:
    base = {
        "M": 0.9,
        "PS": 0.72,
        "PL": 0.55,
        "P": 0.35,
        "I": 0.05,
    }.get(profile.mutability, 0.6)
    distance = _normalized_distance(profile, from_value, to_value)
    return max(0.05, min(1.0, base * (1.0 - 0.45 * distance)))


def _change_realism_score(profile: FeatureProfile, from_value: Any, to_value: Any) -> float:
    distance = _normalized_distance(profile, from_value, to_value)
    if profile.is_numeric:
        return max(0.1, min(1.0, 1.0 - distance))
    category_count = max(1, len(profile.category_risk))
    novelty_penalty = 0.12 if category_count > 2 else 0.2
    return max(0.25, min(0.85, 0.85 - novelty_penalty))


def _normalized_distance(profile: FeatureProfile, from_value: Any, to_value: Any) -> float:
    if from_value == to_value:
        return 0.0
    if profile.is_numeric:
        if profile.numeric_min is None or profile.numeric_max is None:
            return 1.0
        span = profile.numeric_max - profile.numeric_min
        if span <= 0:
            return 0.0
        return min(1.5, abs(float(to_value) - float(from_value)) / span)
    return 1.0


def _prediction_block(prediction: RowPrediction) -> dict[str, Any]:
    return {
        "score": prediction.score,
        "label": prediction.label,
        "threshold": prediction.threshold,
        "decision_margin": prediction.decision_margin,
    }


def _positive_class_values(
    values: Any,
    feature_count: int,
    np_module: Any,
) -> Any:
    array = np_module.asarray(values)
    if array.ndim == 3:
        sample = array[0]
        if sample.shape[1] == 1:
            return sample[:, 0]
        return sample[:, -1]
    if array.ndim == 2:
        if array.shape == (1, feature_count):
            return array[0]
        if array.shape[1] == feature_count:
            return array[-1]
        if array.shape[0] == feature_count:
            return array[:, -1]
    if array.ndim == 1:
        return array
    raise ValueError("unsupported SHAP output shape for positive-class extraction")


def _coerce_numeric_like(original_value: Any, candidate: float) -> Any:
    if isinstance(original_value, int) and not isinstance(original_value, bool):
        return int(round(candidate))
    return round(candidate, 6)


def _coerce_like(original_value: Any, candidate: str) -> Any:
    if isinstance(original_value, int) and not isinstance(original_value, bool):
        try:
            return int(candidate)
        except ValueError:
            return candidate
    if isinstance(original_value, float):
        try:
            return float(candidate)
        except ValueError:
            return candidate
    return candidate


def _clipped_quality_score(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(numeric):
        return 0.0
    return max(0.0, min(1.0, numeric))


def _search_anchor_conditions(
    *,
    model: VectorizedClassifierModel,
    row: dict[str, Any],
    prediction: RowPrediction,
    ranked_feature_limit: int,
    neighborhood_size: int,
    max_rule_length: int,
) -> tuple[list[AnchorCondition], float, float, dict[str, float]]:
    profiles = model.heuristic_helper.profiles
    contributions = model.original_feature_contributions(row)
    ranked_features = [
        feature_name
        for feature_name, _weight in sorted(
            contributions.items(),
            key=lambda item: abs(item[1]),
            reverse=True,
        )[:ranked_feature_limit]
    ]
    candidate_conditions = [
        condition
        for feature_name in ranked_features
        for condition in _build_anchor_conditions(
            feature_name=feature_name,
            profile=profiles[feature_name],
            current_value=row.get(feature_name),
        )
    ]
    neighborhood = _sample_anchor_neighborhood(model, row, sample_count=neighborhood_size)

    chosen: list[AnchorCondition] = []
    best_precision = 0.0
    best_coverage = 1.0

    for _depth in range(max_rule_length):
        best_candidate: AnchorCondition | None = None
        best_candidate_precision = best_precision
        best_candidate_coverage = best_coverage
        best_score = -1.0

        for candidate in candidate_conditions:
            if any(
                existing.feature_name == candidate.feature_name
                and existing.condition_type == candidate.condition_type
                for existing in chosen
            ):
                continue
            precision, coverage = _evaluate_anchor_rule(
                model=model,
                base_label=prediction.label,
                neighborhood=neighborhood,
                conditions=chosen + [candidate],
            )
            if coverage < 0.08:
                continue
            score = precision + 0.15 * coverage - 0.02 * len(chosen)
            if score > best_score:
                best_score = score
                best_candidate = candidate
                best_candidate_precision = precision
                best_candidate_coverage = coverage

        if best_candidate is None:
            break
        if best_candidate_precision <= best_precision and chosen:
            break

        chosen.append(best_candidate)
        best_precision = best_candidate_precision
        best_coverage = best_candidate_coverage

        if best_precision >= 0.95 and best_coverage >= 0.1:
            break

    if not chosen and candidate_conditions:
        fallback = candidate_conditions[0]
        best_precision, best_coverage = _evaluate_anchor_rule(
            model=model,
            base_label=prediction.label,
            neighborhood=neighborhood,
            conditions=[fallback],
        )
        chosen = [fallback]

    return chosen, best_precision, best_coverage, contributions


def _build_anchor_conditions(
    *,
    feature_name: str,
    profile: FeatureProfile,
    current_value: Any,
) -> list[AnchorCondition]:
    if current_value is None:
        return []
    label = profile.plain_meaning.replace("_", " ")
    if profile.is_numeric:
        numeric_value = float(current_value)
        rounded_value = round(numeric_value, 3)
        return [
            AnchorCondition(
                feature_name=feature_name,
                condition_type="ge",
                threshold=numeric_value,
                condition_text=f"{label} >= {rounded_value}",
            ),
            AnchorCondition(
                feature_name=feature_name,
                condition_type="le",
                threshold=numeric_value,
                condition_text=f"{label} <= {rounded_value}",
            ),
        ]
    return [
        AnchorCondition(
            feature_name=feature_name,
            condition_type="eq",
            category_value=current_value,
            condition_text=f"{label} = {current_value}",
        )
    ]


def _sample_anchor_neighborhood(
    model: VectorizedClassifierModel,
    row: dict[str, Any],
    sample_count: int,
) -> list[dict[str, Any]]:
    rng = random.Random(0)
    profiles = model.heuristic_helper.profiles
    rows = [dict(row)]
    for _ in range(sample_count):
        sampled: dict[str, Any] = {}
        for feature_name, profile in profiles.items():
            value = row.get(feature_name)
            if profile.is_numeric and isinstance(value, (int, float)):
                sampled[feature_name] = _sample_numeric_neighbor(profile, value, rng)
            else:
                sampled[feature_name] = _sample_categorical_neighbor(profile, value, rng)
        rows.append(sampled)
    return rows


def _sample_numeric_neighbor(
    profile: FeatureProfile,
    value: Any,
    rng: random.Random,
) -> Any:
    if profile.numeric_min is None or profile.numeric_max is None:
        return value
    span = max(profile.numeric_max - profile.numeric_min, 1.0)
    jitter = 0.12 * span
    sampled = float(value) + rng.uniform(-jitter, jitter)
    sampled = max(profile.numeric_min, min(profile.numeric_max, sampled))
    return _coerce_numeric_like(value, sampled)


def _sample_categorical_neighbor(
    profile: FeatureProfile,
    value: Any,
    rng: random.Random,
) -> Any:
    if rng.random() < 0.82:
        return value
    categories = [item for item in profile.category_risk if item != str(value)]
    if not categories:
        return value
    return _coerce_like(value, rng.choice(categories))


def _evaluate_anchor_rule(
    *,
    model: VectorizedClassifierModel,
    base_label: str,
    neighborhood: list[dict[str, Any]],
    conditions: list[AnchorCondition],
) -> tuple[float, float]:
    covered = [
        candidate
        for candidate in neighborhood
        if all(_anchor_condition_matches(condition, candidate) for condition in conditions)
    ]
    if not covered:
        return 0.0, 0.0

    matches = 0
    for index, candidate in enumerate(covered):
        label = model.predict_row(
            row=candidate,
            instance_id=f"anchor-probe-{index}",
        ).label
        if label == base_label:
            matches += 1
    precision = matches / len(covered)
    coverage = len(covered) / len(neighborhood)
    return precision, coverage


def _anchor_condition_matches(condition: AnchorCondition, row: dict[str, Any]) -> bool:
    value = row.get(condition.feature_name)
    if value is None:
        return False
    if condition.condition_type == "eq":
        return value == condition.category_value
    if condition.threshold is None or not isinstance(value, (int, float)):
        return False
    if condition.condition_type == "ge":
        return float(value) >= condition.threshold
    if condition.condition_type == "le":
        return float(value) <= condition.threshold
    return False
