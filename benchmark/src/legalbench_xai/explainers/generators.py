from typing import Any

from ..models.stubs import HeuristicModel, RowPrediction
from ..types import DatasetBundle


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
    changes = model.recommend_changes(
        row=row,
        prediction=prediction,
        allow_illegal=allow_illegal,
        limit=2,
    )
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
    }[explainer_id]
    return {
        "source": "heuristic_local_neighborhood_probe",
        "neighborhood_consistency": base[0],
        "rank_consistency": base[1],
        "direction_consistency": base[2],
        "seed_consistency": base[3],
    }
