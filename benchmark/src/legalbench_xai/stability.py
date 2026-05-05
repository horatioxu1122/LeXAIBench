import random
from dataclasses import dataclass
from typing import Any

from .adapters import adapt_raw_example
from .explainers import generate_raw_explanation
from .models.stubs import FeatureProfile, RowPrediction
from .normalization import ExplanationCanonicalizer
from .types import CanonicalExplanation, DatasetBundle


@dataclass(slots=True)
class ExplanationSignature:
    feature_order: list[str]
    directions: dict[str, str]


def compute_stability_evidence(
    *,
    probe_cfg: dict[str, Any],
    dataset_bundle: DatasetBundle,
    explainer_id: str,
    model: Any,
    row: dict[str, Any],
    prediction: RowPrediction,
    adapted_payload: dict[str, Any],
) -> dict[str, Any] | None:
    if not probe_cfg.get("enabled", False):
        return None
    if not hasattr(model, "heuristic_helper") or not hasattr(model.heuristic_helper, "profiles"):
        return None

    canonicalizer = ExplanationCanonicalizer(dataset_bundle)
    base_canonical = canonicalizer.canonicalize(adapted_payload)
    base_signature = _signature(base_canonical)

    neighbor_canonicals = _neighbor_canonical_explanations(
        probe_cfg=probe_cfg,
        dataset_bundle=dataset_bundle,
        explainer_id=explainer_id,
        model=model,
        row=row,
        prediction=prediction,
        canonicalizer=canonicalizer,
    )

    if neighbor_canonicals:
        neighborhood_consistency = _mean(
            _set_overlap(base_signature, _signature(candidate))
            for candidate in neighbor_canonicals
        )
        rank_consistency = _mean(
            _rank_similarity(base_signature, _signature(candidate))
            for candidate in neighbor_canonicals
        )
        direction_consistency = _mean(
            _direction_similarity(base_signature, _signature(candidate))
            for candidate in neighbor_canonicals
        )
        source = "perturbation_probe"
    else:
        structural = dict(adapted_payload.get("metadata", {}).get("stability_evidence", {}))
        if structural:
            structural["source"] = f"fallback::{structural.get('source', 'unknown')}"
            return structural
        neighborhood_consistency = 0.0
        rank_consistency = 0.0
        direction_consistency = 0.0
        source = "perturbation_probe_insufficient_neighbors"

    seed_consistency = _seed_consistency(
        probe_cfg=probe_cfg,
        dataset_bundle=dataset_bundle,
        explainer_id=explainer_id,
        model=model,
        row=row,
        prediction=prediction,
        canonicalizer=canonicalizer,
        base_signature=base_signature,
    )

    return {
        "source": source,
        "neighbor_count": len(neighbor_canonicals),
        "neighborhood_consistency": neighborhood_consistency,
        "rank_consistency": rank_consistency,
        "direction_consistency": direction_consistency,
        "seed_consistency": seed_consistency,
    }


def _neighbor_canonical_explanations(
    *,
    probe_cfg: dict[str, Any],
    dataset_bundle: DatasetBundle,
    explainer_id: str,
    model: Any,
    row: dict[str, Any],
    prediction: RowPrediction,
    canonicalizer: ExplanationCanonicalizer,
) -> list[CanonicalExplanation]:
    profiles = model.heuristic_helper.profiles
    num_neighbors = int(probe_cfg.get("num_neighbors", 3))
    candidate_multiplier = max(1, int(probe_cfg.get("candidate_multiplier", 4)))
    same_label_only = bool(probe_cfg.get("same_label_only", True))
    numeric_jitter_fraction = float(probe_cfg.get("numeric_jitter_fraction", 0.06))
    categorical_flip_probability = float(
        probe_cfg.get("categorical_flip_probability", 0.18)
    )

    attempts = max(num_neighbors * candidate_multiplier * 4, num_neighbors)
    seed_basis = sum(ord(char) for char in prediction.instance_id) + len(explainer_id)
    rng = random.Random(seed_basis)

    selected: list[tuple[float, CanonicalExplanation]] = []
    seen_rows: set[tuple[tuple[str, str], ...]] = set()
    for probe_index in range(attempts):
        candidate_row = _sample_neighbor_row(
            profiles=profiles,
            base_row=row,
            rng=rng,
            numeric_jitter_fraction=numeric_jitter_fraction,
            categorical_flip_probability=categorical_flip_probability,
        )
        signature = tuple(sorted((name, repr(value)) for name, value in candidate_row.items()))
        if signature in seen_rows:
            continue
        seen_rows.add(signature)

        candidate_prediction = model.predict_row(
            row=candidate_row,
            instance_id=f"{prediction.instance_id}-nbr{probe_index}",
        )
        if same_label_only and candidate_prediction.label != prediction.label:
            continue

        raw_payload, _selection = generate_raw_explanation(
            explainer_id=explainer_id,
            dataset_bundle=dataset_bundle,
            model=model,
            row=candidate_row,
            prediction=candidate_prediction,
            seed=0,
            generation_mode="stability_probe",
        )
        adapted_payload = adapt_raw_example(
            dataset_bundle=dataset_bundle,
            explainer_id=explainer_id,
            raw_payload=raw_payload,
        )
        canonical = canonicalizer.canonicalize(adapted_payload)
        distance = _row_distance(profiles, row, candidate_row)
        selected.append((distance, canonical))

    selected.sort(key=lambda item: item[0])
    return [canonical for _distance, canonical in selected[:num_neighbors]]


def _seed_consistency(
    *,
    probe_cfg: dict[str, Any],
    dataset_bundle: DatasetBundle,
    explainer_id: str,
    model: Any,
    row: dict[str, Any],
    prediction: RowPrediction,
    canonicalizer: ExplanationCanonicalizer,
    base_signature: ExplanationSignature,
) -> float:
    if explainer_id != "lime":
        return 1.0

    seed_trials = max(2, int(probe_cfg.get("lime_seed_trials", 3)))
    similarities: list[float] = []
    for seed in range(1, seed_trials):
        raw_payload, _selection = generate_raw_explanation(
            explainer_id=explainer_id,
            dataset_bundle=dataset_bundle,
            model=model,
            row=row,
            prediction=prediction,
            seed=seed,
            generation_mode="stability_probe",
        )
        adapted_payload = adapt_raw_example(
            dataset_bundle=dataset_bundle,
            explainer_id=explainer_id,
            raw_payload=raw_payload,
        )
        candidate = canonicalizer.canonicalize(adapted_payload)
        candidate_signature = _signature(candidate)
        similarities.append(
            0.40 * _set_overlap(base_signature, candidate_signature)
            + 0.35 * _rank_similarity(base_signature, candidate_signature)
            + 0.25 * _direction_similarity(base_signature, candidate_signature)
        )
    if not similarities:
        return 1.0
    return _clip01(sum(similarities) / len(similarities))


def _signature(explanation: CanonicalExplanation) -> ExplanationSignature:
    if explanation.explanation_family == "counterfactual" and explanation.counterfactual_changes:
        return ExplanationSignature(
            feature_order=[change.feature_name for change in explanation.counterfactual_changes],
            directions={
                change.feature_name: change.change_direction
                for change in explanation.counterfactual_changes
            },
        )

    return ExplanationSignature(
        feature_order=[feature.feature_name for feature in explanation.referenced_features],
        directions={
            feature.feature_name: feature.signed_direction
            for feature in explanation.referenced_features
        },
    )


def _sample_neighbor_row(
    *,
    profiles: dict[str, FeatureProfile],
    base_row: dict[str, Any],
    rng: random.Random,
    numeric_jitter_fraction: float,
    categorical_flip_probability: float,
) -> dict[str, Any]:
    sampled: dict[str, Any] = {}
    for feature_name, profile in profiles.items():
        value = base_row.get(feature_name)
        if value is None or profile.mutability == "I" or profile.sensitivity == "S":
            sampled[feature_name] = value
            continue
        if profile.is_numeric and isinstance(value, (int, float)):
            sampled[feature_name] = _sample_numeric_neighbor(
                profile=profile,
                value=value,
                rng=rng,
                numeric_jitter_fraction=numeric_jitter_fraction,
            )
            continue
        sampled[feature_name] = _sample_categorical_neighbor(
            profile=profile,
            value=value,
            rng=rng,
            categorical_flip_probability=categorical_flip_probability,
        )
    return sampled


def _sample_numeric_neighbor(
    *,
    profile: FeatureProfile,
    value: Any,
    rng: random.Random,
    numeric_jitter_fraction: float,
) -> Any:
    if profile.numeric_min is None or profile.numeric_max is None:
        return value
    span = max(profile.numeric_max - profile.numeric_min, 1.0)
    mutability_scale = {
        "M": 1.0,
        "PS": 0.8,
        "PL": 0.65,
        "P": 0.5,
    }.get(profile.mutability, 0.5)
    jitter = span * numeric_jitter_fraction * mutability_scale
    sampled = float(value) + rng.uniform(-jitter, jitter)
    sampled = max(profile.numeric_min, min(profile.numeric_max, sampled))
    if isinstance(value, int) and not isinstance(value, bool):
        return int(round(sampled))
    return round(sampled, 6)


def _sample_categorical_neighbor(
    *,
    profile: FeatureProfile,
    value: Any,
    rng: random.Random,
    categorical_flip_probability: float,
) -> Any:
    if rng.random() >= categorical_flip_probability:
        return value
    categories = [candidate for candidate in profile.category_risk if candidate != str(value)]
    if not categories:
        return value
    selected = rng.choice(categories)
    if isinstance(value, int) and not isinstance(value, bool):
        try:
            return int(selected)
        except ValueError:
            return selected
    if isinstance(value, float):
        try:
            return float(selected)
        except ValueError:
            return selected
    return selected


def _row_distance(
    profiles: dict[str, FeatureProfile],
    base_row: dict[str, Any],
    candidate_row: dict[str, Any],
) -> float:
    distances: list[float] = []
    for feature_name, profile in profiles.items():
        original = base_row.get(feature_name)
        candidate = candidate_row.get(feature_name)
        if original == candidate:
            distances.append(0.0)
            continue
        if profile.is_numeric and isinstance(original, (int, float)) and isinstance(
            candidate,
            (int, float),
        ):
            if profile.numeric_min is None or profile.numeric_max is None:
                distances.append(1.0)
                continue
            span = max(profile.numeric_max - profile.numeric_min, 1.0)
            distances.append(min(1.0, abs(float(candidate) - float(original)) / span))
            continue
        distances.append(1.0)
    if not distances:
        return 0.0
    return sum(distances) / len(distances)


def _set_overlap(base: ExplanationSignature, candidate: ExplanationSignature) -> float:
    base_set = set(base.feature_order)
    candidate_set = set(candidate.feature_order)
    if not base_set and not candidate_set:
        return 1.0
    union = base_set | candidate_set
    if not union:
        return 1.0
    return len(base_set & candidate_set) / len(union)


def _rank_similarity(base: ExplanationSignature, candidate: ExplanationSignature) -> float:
    union = list(dict.fromkeys(base.feature_order + candidate.feature_order))
    if not union:
        return 1.0
    base_ranks = {name: index for index, name in enumerate(base.feature_order, start=1)}
    candidate_ranks = {
        name: index for index, name in enumerate(candidate.feature_order, start=1)
    }
    default_rank = max(len(base.feature_order), len(candidate.feature_order), 1) + 1
    displacement = sum(
        abs(base_ranks.get(name, default_rank) - candidate_ranks.get(name, default_rank))
        for name in union
    )
    denominator = len(union) * max(1, default_rank - 1)
    return _clip01(1.0 - (displacement / denominator))


def _direction_similarity(base: ExplanationSignature, candidate: ExplanationSignature) -> float:
    overlap = set(base.directions) & set(candidate.directions)
    if not overlap:
        return 1.0 if not base.directions and not candidate.directions else 0.0
    scores = [
        _direction_match_score(base.directions[name], candidate.directions[name])
        for name in overlap
    ]
    return sum(scores) / len(scores)


def _direction_match_score(first: str, second: str) -> float:
    if first == second:
        return 1.0
    if "neutral" in {first, second}:
        return 0.5
    return 0.0


def _mean(values: Any) -> float:
    materialized = list(values)
    if not materialized:
        return 0.0
    return sum(materialized) / len(materialized)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))
