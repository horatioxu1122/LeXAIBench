import random
import zlib
from typing import Any

from ..corrections import model_corrections
from ..models.stubs import HeuristicModel, RowPrediction
from ..types import DatasetBundle


NULL_BASELINE_EXPLAINERS = ("random_reasons", "global_ranking")
NULL_BASELINE_DEFAULT_NUM_FEATURES = 6
NULL_BASELINE_REFERENCE_MODES = ("self", "native")

# Correction counterfactual_reasons_from_changes (v2).
CF_REASON_LIMIT = 6
CF_REASON_EPSILON = 1e-6
CF_REASON_RULE = (
    "features changed by the selected counterfactuals that flip the decision; rank by "
    "number of flipping counterfactuals changing the feature (desc), ties by native "
    f"contribution (desc), then feature name; weight = native contribution if > 0 else "
    f"{CF_REASON_EPSILON:g} x change count; signed_direction = sign of the native "
    f"contribution; at most {CF_REASON_LIMIT} features"
)


def counterfactual_change_reasons(
    *,
    changed_feature_sets: list[list[str]],
    flip_flags: list[bool],
    native_contributions: dict[str, float],
    native_source: str,
    row: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """DiCE/Wachter reasons derived from what their counterfactuals change.

    Correction counterfactual_reasons_from_changes (v2). v1 copied the top-6 features
    of an attribution ranking into "feature_importances", so DiCE and Wachter cited
    identical reasons that had nothing to do with their counterfactuals. Here a
    feature is cited only if at least one selected counterfactual that flips the
    decision (flip_flags True) changes it; changes made by non-flipping
    counterfactuals did not alter the outcome and are not cited (they are kept in
    cf_reason_details["non_flipping_change_counts"] for diagnostics).
    Ranking: number of flipping counterfactuals that change the feature (desc), ties
    by signed native contribution (desc, so the strongest push toward denial first),
    then feature name. Weight (importance_weight downstream): the native contribution
    (log-odds toward denial) when it is > 0; otherwise CF_REASON_EPSILON x change
    count, a positive weight that is negligible next to any real positive
    contribution but keeps frequency order among non-positive features.
    Each item carries an explicit "signed_direction" taken from the sign of the
    native contribution (> 0 increases_denial, < 0 decreases_denial, else neutral),
    so the reason text and the direction label state what the model says about the
    changed feature, not what the positive weight would imply. The share of cited
    features the model does not push toward denial is recorded as
    cf_reason_details["nondenial_reason_share"] (reported as its own column, rather
    than carried by a mislabelled direction).
    At most CF_REASON_LIMIT features. When no flipping counterfactual changes any
    feature (failed search: no candidate flips, or no allowed features), the list
    is empty and cf_reason_source = "no_valid_change".

    Returns (feature_importances, metadata_update).
    """
    counts: dict[str, int] = {}
    non_flipping_counts: dict[str, int] = {}
    flags = list(flip_flags) + [False] * max(0, len(changed_feature_sets) - len(flip_flags))
    for feature_names, flipped in zip(changed_feature_sets, flags):
        target = counts if flipped else non_flipping_counts
        for feature_name in feature_names:
            target[feature_name] = target.get(feature_name, 0) + 1
    details: dict[str, Any] = {
        "rule": CF_REASON_RULE,
        "selected_counterfactuals": len(changed_feature_sets),
        "counterfactuals_with_changes": sum(
            1 for feature_names in changed_feature_sets if feature_names
        ),
        "flipping_counterfactuals": sum(1 for flag in flip_flags if flag),
        "change_counts": dict(counts),
        "non_flipping_change_counts": dict(non_flipping_counts),
        "native_contribution_source": native_source,
    }
    if not counts:
        details["nondenial_reason_share"] = None
        return [], {"cf_reason_source": "no_valid_change", "cf_reason_details": details}

    def native_value(feature_name: str) -> float:
        try:
            value = float(native_contributions.get(feature_name, 0.0))
        except (TypeError, ValueError):
            return 0.0
        return value if value == value else 0.0  # NaN -> 0.0

    ranked = sorted(
        counts,
        key=lambda feature_name: (
            -counts[feature_name],
            -native_value(feature_name),
            feature_name,
        ),
    )[:CF_REASON_LIMIT]
    feature_importances = []
    native_signs: dict[str, str] = {}
    for feature_name in ranked:
        contribution = native_value(feature_name)
        weight = contribution if contribution > 0 else CF_REASON_EPSILON * counts[feature_name]
        if contribution > 0:
            signed_direction = "increases_denial"
        elif contribution < 0:
            signed_direction = "decreases_denial"
        else:
            signed_direction = "neutral"
        native_signs[feature_name] = signed_direction
        feature_importances.append(
            {
                "feature_name": feature_name,
                "weight": weight,
                "signed_direction": signed_direction,
                "raw_value": row.get(feature_name),
            }
        )
    details["native_direction"] = native_signs
    details["nondenial_reason_share"] = sum(
        1 for direction in native_signs.values() if direction != "increases_denial"
    ) / len(native_signs)
    return feature_importances, {
        "cf_reason_source": "counterfactual_changes",
        "cf_reason_details": details,
    }


def resolve_null_baseline_options(model: Any) -> dict[str, Any]:
    """Read null-baseline options set by the runner (cslaw.null_baselines).

    Source: getattr(model, "null_baseline_options", {}). Defaults: num_features=6,
    faithfulness_reference="self". Invalid values raise ValueError (fail loudly rather
    than silently fall back to a self-referential reference).
    """
    options = getattr(model, "null_baseline_options", None) or {}
    if not isinstance(options, dict):
        raise ValueError("null_baseline_options must be a dict")
    raw_num_features = options.get("num_features")
    if raw_num_features is None:
        raw_num_features = NULL_BASELINE_DEFAULT_NUM_FEATURES
    try:
        num_features = int(raw_num_features)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"null_baselines.num_features must be an integer, got {raw_num_features!r}"
        ) from exc
    if num_features < 1:
        raise ValueError("null_baselines.num_features must be >= 1")
    reference = options.get("faithfulness_reference")
    if reference is None:
        reference = "self"
    if reference not in NULL_BASELINE_REFERENCE_MODES:
        raise ValueError(
            "null_baselines.faithfulness_reference must be one of "
            f"{NULL_BASELINE_REFERENCE_MODES}, got {reference!r}"
        )
    return {"num_features": num_features, "faithfulness_reference": reference}


def random_reason_weights(
    feature_names: list[str],
    *,
    seed: int,
    instance_id: str,
    generation_mode: str,
) -> dict[str, float]:
    """Random signed weights for the random_reasons null baseline.

    Uses a private random.Random seeded with
    zlib.crc32(f"{seed}|{instance_id}|{generation_mode}") (stable across processes;
    never builtin hash(), never the global random/numpy RNGs). For each feature, in the
    given order: sign = +1 if rng.random() < 0.5 else -1, then magnitude = rng.random()
    ~ U(0, 1); weight = sign * magnitude. Unitless.
    """
    key = f"{seed}|{instance_id}|{generation_mode}".encode("utf-8")
    rng = random.Random(zlib.crc32(key))
    weights: dict[str, float] = {}
    for feature_name in feature_names:
        sign = 1.0 if rng.random() < 0.5 else -1.0
        magnitude = rng.random()
        weights[feature_name] = sign * magnitude
    return weights


def top_null_baseline_features(
    weights: dict[str, float],
    num_features: int,
) -> list[tuple[str, float]]:
    """Top num_features by |weight|; ties keep the input (bundle) order."""
    return sorted(
        weights.items(),
        key=lambda item: abs(item[1]),
        reverse=True,
    )[:num_features]


def generate_raw_explanation(
    explainer_id: str,
    dataset_bundle: DatasetBundle,
    model: HeuristicModel,
    row: dict[str, Any],
    prediction: RowPrediction,
    seed: int | None = None,
    generation_mode: str = "full",
) -> dict[str, Any]:
    if explainer_id == "shap":
        return _generate_shap_payload(prediction, dataset_bundle)
    if explainer_id == "lime":
        return _generate_lime_payload(prediction, dataset_bundle, seed=seed)
    if explainer_id == "leave_one_out":
        return _generate_leave_one_out_payload(prediction, dataset_bundle)
    if explainer_id == "anchors":
        return _generate_anchors_payload(prediction, row, dataset_bundle)
    if explainer_id == "dice":
        return _generate_counterfactual_payload(
            prediction=prediction,
            row=row,
            model=model,
            explainer_id="dice",
            allow_illegal=True,
            dataset_bundle=dataset_bundle,
        )
    if explainer_id == "wachter":
        return _generate_counterfactual_payload(
            prediction=prediction,
            row=row,
            model=model,
            explainer_id="wachter",
            allow_illegal=False,
            dataset_bundle=dataset_bundle,
        )
    if explainer_id == "random_reasons":
        return _generate_random_reasons_payload(
            prediction=prediction,
            row=row,
            model=model,
            dataset_bundle=dataset_bundle,
            seed=0 if seed is None else seed,
            generation_mode=generation_mode,
        )
    if explainer_id == "global_ranking":
        return _generate_global_ranking_payload(
            prediction=prediction,
            row=row,
            model=model,
            dataset_bundle=dataset_bundle,
        )
    raise KeyError(f"Unsupported explainer '{explainer_id}'.")


def _ranked_contributions(prediction: RowPrediction) -> list[tuple[str, float]]:
    return sorted(
        prediction.feature_contributions.items(),
        key=lambda item: abs(item[1]),
        reverse=True,
    )


def _generate_shap_payload(
    prediction: RowPrediction,
    dataset_bundle: DatasetBundle,
) -> dict[str, Any]:
    values = _ranked_contributions(prediction)[:4]
    return {
        "instance_id": prediction.instance_id,
        "model_id": "xgboost",
        "prediction": _prediction_block(prediction),
        "shap_values": [
            {
                "feature_name": feature_name,
                "phi": weight,
                "raw_value": None,
            }
            for feature_name, weight in values
        ],
        "metadata": {
            "generated_from": "heuristic_model",
            "raw_family": "attribution",
            "faithfulness_evidence": _faithfulness_evidence(
                prediction=prediction,
                local_quality_score=0.98,
            ),
            "stability_evidence": _stability_evidence("shap"),
        },
    }


def _generate_lime_payload(
    prediction: RowPrediction,
    dataset_bundle: DatasetBundle,
    seed: int | None = None,
) -> dict[str, Any]:
    values = _ranked_contributions(prediction)[:4]
    lime_seed = 7 if seed is None else seed
    return {
        "instance_id": prediction.instance_id,
        "model_id": "logistic_regression",
        "seed": lime_seed,
        "prediction": _prediction_block(prediction),
        "local_explanation": [
            {
                "feature_name": feature_name,
                "weight": 0.9 * weight,
                "raw_value": None,
            }
            for feature_name, weight in values
        ],
        "metadata": {
            "generated_from": "heuristic_model",
            "raw_family": "attribution",
            "faithfulness_evidence": _faithfulness_evidence(
                prediction=prediction,
                local_quality_score=0.9,
            ),
            "stability_evidence": _stability_evidence("lime"),
        },
    }


def _generate_leave_one_out_payload(
    prediction: RowPrediction,
    dataset_bundle: DatasetBundle,
) -> dict[str, Any]:
    values = _ranked_contributions(prediction)[:4]
    return {
        "instance_id": prediction.instance_id,
        "model_id": "logistic_regression",
        "prediction": _prediction_block(prediction),
        "local_explanation": [
            {
                "feature_name": feature_name,
                "weight": 0.95 * weight,
                "raw_value": None,
            }
            for feature_name, weight in values
        ],
        "metadata": {
            "generated_from": "heuristic_model",
            "raw_family": "attribution",
            "leave_one_out_settings": {
                "reference_policy": "heuristic_fill_value_replacement",
                "evaluated_feature_count": len(dataset_bundle.included_feature_names),
            },
            "faithfulness_evidence": _faithfulness_evidence(
                prediction=prediction,
                local_quality_score=0.92,
            ),
            "stability_evidence": _stability_evidence("leave_one_out"),
        },
    }


def _generate_anchors_payload(
    prediction: RowPrediction,
    row: dict[str, Any],
    dataset_bundle: DatasetBundle,
) -> dict[str, Any]:
    conditions = []
    for feature_name, _weight in _ranked_contributions(prediction)[:2]:
        conditions.append(
            {
                "feature_name": feature_name,
                "raw_value": row.get(feature_name),
                "condition_text": f"{feature_name} is in a denial-supporting range",
            }
        )
    return {
        "instance_id": prediction.instance_id,
        "model_id": "ebm",
        "prediction": _prediction_block(prediction),
        "anchor_conditions": conditions,
        "precision": 0.9,
        "coverage": 0.25,
        "metadata": {
            "generated_from": "heuristic_model",
            "raw_family": "rule",
            "faithfulness_evidence": _faithfulness_evidence(
                prediction=prediction,
                local_quality_score=0.9,
            ),
            "stability_evidence": _stability_evidence("anchors"),
        },
    }


def _generate_counterfactual_payload(
    prediction: RowPrediction,
    row: dict[str, Any],
    model: HeuristicModel,
    explainer_id: str,
    allow_illegal: bool,
    dataset_bundle: DatasetBundle,
) -> dict[str, Any]:
    corrections = model_corrections(model)
    # Correction cf_respect_nonactionable: the keyword is passed only when the option
    # is on, so the v1 call is unchanged.
    recommend_kwargs = (
        {"block_nonactionable": True} if corrections.cf_respect_nonactionable else {}
    )
    changes = model.recommend_changes(
        row=row,
        prediction=prediction,
        allow_illegal=allow_illegal,
        limit=2,
        **recommend_kwargs,
    )
    reason_metadata: dict[str, Any] = {}
    if corrections.counterfactual_reasons_from_changes:
        native_fn = getattr(model, "native_feature_contributions", None)
        if native_fn is not None:
            native = dict(native_fn(row))
            native_source = str(getattr(model, "native_contribution_source", "heuristic_stub"))
        else:
            native = dict(prediction.feature_contributions)
            native_source = "heuristic_stub"
        # The stub's recommend_changes marks every change set as flipping; check it
        # by re-scoring the changed row, so failed searches report no_valid_change.
        stub_flip = False
        if changes:
            changed_row = dict(row)
            for change in changes:
                changed_row[change["feature_name"]] = change["to_value"]
            stub_flip = (
                model.predict_row(changed_row, prediction.instance_id).label != prediction.label
            )
        feature_importances, reason_metadata = counterfactual_change_reasons(
            changed_feature_sets=[[change["feature_name"] for change in changes]],
            flip_flags=[stub_flip],
            native_contributions=native,
            native_source=native_source,
            row=row,
        )
    else:
        feature_importances = [
            {
                "feature_name": feature_name,
                "weight": weight,
                "raw_value": row.get(feature_name),
            }
            for feature_name, weight in _ranked_contributions(prediction)[:3]
        ]
    cf_block = {
        "changes": {
            change["feature_name"]: {
                "from_value": change["from_value"],
                "to_value": change["to_value"],
            }
            for change in changes
        },
        "validity_flip": True if changes else False,
        "realism_score": min((change["realism_score"] for change in changes), default=0.0),
        "effort_score": min((change["effort_score"] for change in changes), default=0.0),
    }
    return {
        "instance_id": prediction.instance_id,
        "model_id": model.model_id,
        "prediction": _prediction_block(prediction),
        "feature_importances": feature_importances,
        "counterfactuals": [cf_block],
        "metadata": {
            "generated_from": "heuristic_model",
            "raw_family": "counterfactual",
            "allow_illegal": allow_illegal,
            "faithfulness_evidence": _faithfulness_evidence(
                prediction=prediction,
                local_quality_score=1.0 if changes else 0.0,
            ),
            "stability_evidence": _stability_evidence(explainer_id),
            **reason_metadata,
        },
    }


def _generate_random_reasons_payload(
    *,
    prediction: RowPrediction,
    row: dict[str, Any],
    model: Any,
    dataset_bundle: DatasetBundle,
    seed: int,
    generation_mode: str,
) -> dict[str, Any]:
    options = resolve_null_baseline_options(model)
    weights = random_reason_weights(
        list(dataset_bundle.included_feature_names),
        seed=seed,
        instance_id=prediction.instance_id,
        generation_mode=generation_mode,
    )
    return _null_baseline_stub_payload(
        explainer_id="random_reasons",
        prediction=prediction,
        row=row,
        model=model,
        weights=weights,
        options=options,
        seed=seed,
        settings={
            "weight_distribution": "sign_uniform_pm1_times_magnitude_uniform_0_1",
            "rng": "random.Random(zlib.crc32('{seed}|{instance_id}|{generation_mode}'))",
            "generation_mode": generation_mode,
        },
    )


def _generate_global_ranking_payload(
    *,
    prediction: RowPrediction,
    row: dict[str, Any],
    model: Any,
    dataset_bundle: DatasetBundle,
) -> dict[str, Any]:
    options = resolve_null_baseline_options(model)
    profiles = getattr(model, "profiles", None)
    if profiles is None:
        profiles = getattr(getattr(model, "heuristic_helper", None), "profiles", None) or {}
    weights = {
        feature_name: (
            float(profiles[feature_name].weight) if feature_name in profiles else 0.5
        )
        for feature_name in dataset_bundle.included_feature_names
    }
    return _null_baseline_stub_payload(
        explainer_id="global_ranking",
        prediction=prediction,
        row=row,
        model=model,
        weights=weights,
        options=options,
        seed=None,
        settings={
            "aggregation": "taxonomy_principal_reason_weight",
            "contribution_source": "heuristic_stub",
        },
    )


def _null_baseline_stub_payload(
    *,
    explainer_id: str,
    prediction: RowPrediction,
    row: dict[str, Any],
    model: Any,
    weights: dict[str, float],
    options: dict[str, Any],
    seed: int | None,
    settings: dict[str, Any],
) -> dict[str, Any]:
    top_values = top_null_baseline_features(weights, options["num_features"])
    reference_source = "self"
    if options["faithfulness_reference"] == "native":
        try:
            reference = dict(model.native_feature_contributions(row))
            reference_source = str(
                getattr(model, "native_contribution_source", "heuristic_stub")
            )
        except Exception:
            original = getattr(model, "original_feature_contributions", None)
            reference = (
                dict(original(row))
                if original is not None
                else dict(prediction.feature_contributions)
            )
            reference_source = "heuristic_stub"
    else:
        reference = dict(weights)
    return {
        "instance_id": prediction.instance_id,
        "model_id": getattr(model, "model_id", None),
        "seed": seed,
        "prediction": _prediction_block(prediction),
        "local_explanation": [
            {
                "feature_name": feature_name,
                "weight": weight,
                "raw_value": row.get(feature_name),
            }
            for feature_name, weight in top_values
        ],
        "metadata": {
            "generated_from": "heuristic_model",
            "raw_family": "attribution",
            "null_baseline_settings": {
                **settings,
                "num_features": options["num_features"],
                "faithfulness_reference": options["faithfulness_reference"],
                "reference_contribution_source": reference_source,
            },
            "faithfulness_evidence": {
                "source": f"null_{explainer_id}",
                "reference_signed_contributions": reference,
            },
            "stability_evidence": _stability_evidence(explainer_id),
        },
    }


def _prediction_block(prediction: RowPrediction) -> dict[str, Any]:
    return {
        "score": prediction.score,
        "label": prediction.label,
        "threshold": prediction.threshold,
        "decision_margin": prediction.decision_margin,
    }


def _faithfulness_evidence(
    prediction: RowPrediction,
    local_quality_score: float,
) -> dict[str, Any]:
    top = _ranked_contributions(prediction)[:6]
    return {
        "source": "heuristic_model_feature_contributions",
        "local_quality_score": local_quality_score,
        "reference_signed_contributions": {
            feature_name: weight for feature_name, weight in top
        },
    }


def _stability_evidence(explainer_id: str) -> dict[str, Any]:
    base = {
        "shap": (0.9, 0.88, 0.96, 0.95),
        "lime": (0.64, 0.59, 0.84, 0.54),
        "leave_one_out": (0.88, 0.84, 0.93, 0.90),
        "anchors": (0.86, 0.83, 0.94, 0.8),
        "dice": (0.68, 0.64, 0.84, 0.62),
        "wachter": (0.79, 0.75, 0.87, 0.77),
        # Stub-only placeholders for the null baselines: random reasons are unstable by
        # construction; the global ranking is identical for every applicant.
        "random_reasons": (0.2, 0.15, 0.5, 0.2),
        "global_ranking": (1.0, 1.0, 1.0, 1.0),
    }[explainer_id]
    return {
        "source": "heuristic_local_neighborhood_probe",
        "neighborhood_consistency": base[0],
        "rank_consistency": base[1],
        "direction_consistency": base[2],
        "seed_consistency": base[3],
    }
