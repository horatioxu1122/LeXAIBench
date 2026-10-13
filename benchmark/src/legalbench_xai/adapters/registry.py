from ..types import DatasetBundle


def _adapt_shap(raw_payload: dict) -> dict:
    return {
        "instance_id": raw_payload["instance_id"],
        "model_id": raw_payload["model_id"],
        "explainer_id": "shap",
        "explanation_family": "attribution",
        "prediction": raw_payload["prediction"],
        "features": [
            {
                "feature_name": item["feature_name"],
                "importance_weight": item["phi"],
                "raw_value": item.get("raw_value"),
            }
            for item in raw_payload.get("shap_values", [])
        ],
        "metadata": raw_payload.get("metadata", {}),
        "counterfactual_changes": [],
        "seed": raw_payload.get("seed"),
    }


def _adapt_lime(raw_payload: dict) -> dict:
    return {
        "instance_id": raw_payload["instance_id"],
        "model_id": raw_payload["model_id"],
        "explainer_id": "lime",
        "explanation_family": "attribution",
        "prediction": raw_payload["prediction"],
        "features": [
            {
                "feature_name": item["feature_name"],
                "importance_weight": item["weight"],
                "raw_value": item.get("raw_value"),
            }
            for item in raw_payload.get("local_explanation", [])
        ],
        "metadata": raw_payload.get("metadata", {}),
        "counterfactual_changes": [],
        "seed": raw_payload.get("seed"),
    }


def _adapt_leave_one_out(raw_payload: dict) -> dict:
    return {
        "instance_id": raw_payload["instance_id"],
        "model_id": raw_payload["model_id"],
        "explainer_id": "leave_one_out",
        "explanation_family": "attribution",
        "prediction": raw_payload["prediction"],
        "features": [
            {
                "feature_name": item["feature_name"],
                "importance_weight": item["weight"],
                "raw_value": item.get("raw_value"),
            }
            for item in raw_payload.get("local_explanation", [])
        ],
        "metadata": raw_payload.get("metadata", {}),
        "counterfactual_changes": [],
        "seed": raw_payload.get("seed"),
    }


def _adapt_null_baseline(raw_payload: dict, explainer_id: str) -> dict:
    """Attribution adapter for the null baselines (random_reasons, global_ranking)."""
    return {
        "instance_id": raw_payload["instance_id"],
        "model_id": raw_payload["model_id"],
        "explainer_id": explainer_id,
        "explanation_family": "attribution",
        "prediction": raw_payload["prediction"],
        "features": [
            {
                "feature_name": item["feature_name"],
                "importance_weight": item["weight"],
                "raw_value": item.get("raw_value"),
            }
            for item in raw_payload.get("local_explanation", [])
        ],
        "metadata": raw_payload.get("metadata", {}),
        "counterfactual_changes": [],
        "seed": raw_payload.get("seed"),
    }


def _adapt_anchors(raw_payload: dict) -> dict:
    return {
        "instance_id": raw_payload["instance_id"],
        "model_id": raw_payload["model_id"],
        "explainer_id": "anchors",
        "explanation_family": "rule",
        "prediction": raw_payload["prediction"],
        "features": [
            {
                "feature_name": item["feature_name"],
                "importance_weight": 1.0,
                "signed_direction": "increases_denial",
                "raw_value": item.get("raw_value"),
            }
            for item in raw_payload.get("anchor_conditions", [])
        ],
        "metadata": {
            **raw_payload.get("metadata", {}),
            "confidence_or_precision": raw_payload.get("precision"),
            "coverage": raw_payload.get("coverage"),
        },
        "counterfactual_changes": [],
        "seed": raw_payload.get("seed"),
    }


def _counterfactual_feature(item: dict) -> dict:
    feature = {
        "feature_name": item["feature_name"],
        "importance_weight": item["weight"],
        "raw_value": item.get("raw_value"),
    }
    # v2 reasons from changes carry the native sign explicitly
    # (generators.counterfactual_change_reasons); v1 items have no such key, so the
    # v1 feature dict is unchanged and the canonicalizer infers it from the weight.
    if item.get("signed_direction") is not None:
        feature["signed_direction"] = item["signed_direction"]
    return feature


def _adapt_counterfactual(raw_payload: dict, explainer_id: str) -> dict:
    counterfactuals = raw_payload.get("counterfactuals") or [{}]
    first_cf = counterfactuals[0]
    changes = []
    for feature_name, change in first_cf.get("changes", {}).items():
        changes.append(
            {
                "feature_name": feature_name,
                "from_value": change["from_value"],
                "to_value": change["to_value"],
                "validity_flip": first_cf.get("validity_flip"),
                "realism_score": first_cf.get("realism_score"),
                "effort_score": first_cf.get("effort_score"),
            }
        )
    return {
        "instance_id": raw_payload["instance_id"],
        "model_id": raw_payload["model_id"],
        "explainer_id": explainer_id,
        "explanation_family": "counterfactual",
        "prediction": raw_payload["prediction"],
        "features": [
            _counterfactual_feature(item)
            for item in raw_payload.get("feature_importances", [])
        ],
        "metadata": raw_payload.get("metadata", {}),
        "counterfactual_changes": changes,
        "seed": raw_payload.get("seed"),
    }


def adapt_raw_example(dataset_bundle: DatasetBundle, explainer_id: str, raw_payload: dict) -> dict:
    if explainer_id == "shap":
        return _adapt_shap(raw_payload)
    if explainer_id == "lime":
        return _adapt_lime(raw_payload)
    if explainer_id == "leave_one_out":
        return _adapt_leave_one_out(raw_payload)
    if explainer_id == "anchors":
        return _adapt_anchors(raw_payload)
    if explainer_id == "dice":
        return _adapt_counterfactual(raw_payload, explainer_id="dice")
    if explainer_id == "wachter":
        return _adapt_counterfactual(raw_payload, explainer_id="wachter")
    if explainer_id in {"random_reasons", "global_ranking"}:
        return _adapt_null_baseline(raw_payload, explainer_id=explainer_id)
    raise KeyError(
        f"Unsupported explainer '{explainer_id}' for dataset '{dataset_bundle.dataset_id}'."
    )
