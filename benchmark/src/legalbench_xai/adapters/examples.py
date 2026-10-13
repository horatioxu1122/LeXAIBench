from ..types import DatasetBundle


def sample_raw_payload(dataset_bundle: DatasetBundle, explainer_id: str) -> dict:
    if explainer_id == "shap":
        return {
            "instance_id": f"{dataset_bundle.dataset_id}-raw-shap-001",
            "model_id": "xgboost",
            "prediction": {
                "score": 0.81,
                "label": dataset_bundle.negative_label,
                "threshold": 0.5,
                "decision_margin": 0.31,
            },
            "shap_values": [
                {
                    "feature_name": dataset_bundle.included_feature_names[0],
                    "phi": 0.28,
                    "raw_value": 87,
                },
                {
                    "feature_name": dataset_bundle.included_feature_names[1],
                    "phi": 0.18,
                    "raw_value": 5,
                },
                {
                    "feature_name": dataset_bundle.included_feature_names[2],
                    "phi": 0.09,
                    "raw_value": 2,
                },
            ],
            "metadata": {
                "raw_family": "attribution",
                "faithfulness_evidence": {
                    "source": "demo_reference_contributions",
                    "local_quality_score": 0.98,
                    "reference_signed_contributions": {
                        dataset_bundle.included_feature_names[0]: 0.28,
                        dataset_bundle.included_feature_names[1]: 0.18,
                        dataset_bundle.included_feature_names[2]: 0.09,
                    },
                },
                "stability_evidence": {
                    "source": "demo_local_neighborhood_probe",
                    "neighborhood_consistency": 0.9,
                    "rank_consistency": 0.88,
                    "direction_consistency": 0.96,
                    "seed_consistency": 0.95,
                },
            },
        }

    if explainer_id == "lime":
        return {
            "instance_id": f"{dataset_bundle.dataset_id}-raw-lime-001",
            "model_id": "logistic_regression",
            "seed": 11,
            "prediction": {
                "score": 0.76,
                "label": dataset_bundle.negative_label,
                "threshold": 0.5,
                "decision_margin": 0.26,
            },
            "local_explanation": [
                {
                    "feature_name": dataset_bundle.included_feature_names[0],
                    "weight": 0.23,
                    "raw_value": 2,
                },
                {
                    "feature_name": dataset_bundle.included_feature_names[1],
                    "weight": 0.17,
                    "raw_value": 1,
                },
                {
                    "feature_name": dataset_bundle.included_feature_names[2],
                    "weight": -0.04,
                    "raw_value": 1000,
                },
            ],
            "metadata": {
                "raw_family": "attribution",
                "faithfulness_evidence": {
                    "source": "demo_reference_contributions",
                    "local_quality_score": 0.9,
                    "reference_signed_contributions": {
                        dataset_bundle.included_feature_names[0]: 0.23,
                        dataset_bundle.included_feature_names[1]: 0.17,
                        dataset_bundle.included_feature_names[2]: -0.04,
                    },
                },
                "stability_evidence": {
                    "source": "demo_local_neighborhood_probe",
                    "neighborhood_consistency": 0.63,
                    "rank_consistency": 0.58,
                    "direction_consistency": 0.84,
                    "seed_consistency": 0.53,
                },
            },
        }

    if explainer_id == "leave_one_out":
        return {
            "instance_id": f"{dataset_bundle.dataset_id}-raw-leave-one-out-001",
            "model_id": "logistic_regression",
            "prediction": {
                "score": 0.78,
                "label": dataset_bundle.negative_label,
                "threshold": 0.5,
                "decision_margin": 0.28,
            },
            "local_explanation": [
                {
                    "feature_name": dataset_bundle.included_feature_names[0],
                    "weight": 0.26,
                    "raw_value": 3,
                },
                {
                    "feature_name": dataset_bundle.included_feature_names[1],
                    "weight": 0.15,
                    "raw_value": 2,
                },
                {
                    "feature_name": dataset_bundle.included_feature_names[2],
                    "weight": -0.05,
                    "raw_value": 1200,
                },
            ],
            "metadata": {
                "raw_family": "attribution",
                "leave_one_out_settings": {
                    "reference_policy": "training_fill_value_replacement",
                    "evaluated_feature_count": len(dataset_bundle.included_feature_names),
                },
                "faithfulness_evidence": {
                    "source": "demo_leave_one_out_ablation",
                    "local_quality_score": 0.92,
                    "reference_signed_contributions": {
                        dataset_bundle.included_feature_names[0]: 0.26,
                        dataset_bundle.included_feature_names[1]: 0.15,
                        dataset_bundle.included_feature_names[2]: -0.05,
                    },
                },
                "stability_evidence": {
                    "source": "demo_local_neighborhood_probe",
                    "neighborhood_consistency": 0.87,
                    "rank_consistency": 0.84,
                    "direction_consistency": 0.93,
                    "seed_consistency": 0.9,
                },
            },
        }

    if explainer_id == "anchors":
        return {
            "instance_id": f"{dataset_bundle.dataset_id}-raw-anchors-001",
            "model_id": "ebm",
            "prediction": {
                "score": 0.74,
                "label": dataset_bundle.negative_label,
                "threshold": 0.5,
                "decision_margin": 0.24,
            },
            "anchor_conditions": [
                {
                    "feature_name": dataset_bundle.included_feature_names[0],
                    "raw_value": "high",
                    "condition_text": "feature is in a risk-heavy range",
                },
                {
                    "feature_name": dataset_bundle.included_feature_names[1],
                    "raw_value": "recent",
                    "condition_text": "feature indicates recent adverse behavior",
                },
            ],
            "precision": 0.93,
            "coverage": 0.18,
            "metadata": {
                "raw_family": "rule",
                "faithfulness_evidence": {
                    "source": "demo_anchor_proxy",
                    "local_quality_score": 0.93,
                    "reference_signed_contributions": {
                        dataset_bundle.included_feature_names[0]: 1.0,
                        dataset_bundle.included_feature_names[1]: 0.8,
                    },
                },
                "stability_evidence": {
                    "source": "demo_local_neighborhood_probe",
                    "neighborhood_consistency": 0.86,
                    "rank_consistency": 0.82,
                    "direction_consistency": 0.94,
                    "seed_consistency": 0.79,
                },
            },
        }

    if explainer_id in {"dice", "wachter"}:
        return {
            "instance_id": f"{dataset_bundle.dataset_id}-raw-{explainer_id}-001",
            "model_id": "xgboost" if explainer_id == "wachter" else "ebm",
            "prediction": {
                "score": 0.72,
                "label": dataset_bundle.negative_label,
                "threshold": 0.5,
                "decision_margin": 0.22,
            },
            "feature_importances": [
                {
                    "feature_name": dataset_bundle.included_feature_names[0],
                    "weight": 0.24,
                    "raw_value": 4,
                },
                {
                    "feature_name": dataset_bundle.included_feature_names[1],
                    "weight": 0.19,
                    "raw_value": 4200,
                },
            ],
            "counterfactuals": [
                {
                    "changes": {
                        dataset_bundle.included_feature_names[0]: {
                            "from_value": 4,
                            "to_value": 2
                        },
                        dataset_bundle.included_feature_names[1]: {
                            "from_value": 4200,
                            "to_value": 3000
                        }
                    },
                    "validity_flip": True,
                    "realism_score": 0.84,
                    "effort_score": 0.68
                }
            ],
            "metadata": {
                "raw_family": "counterfactual",
                "faithfulness_evidence": {
                    "source": "demo_counterfactual_proxy",
                    "local_quality_score": 1.0,
                    "reference_signed_contributions": {
                        dataset_bundle.included_feature_names[0]: 0.24,
                        dataset_bundle.included_feature_names[1]: 0.19,
                    },
                },
                "stability_evidence": {
                    "source": "demo_counterfactual_neighbor_probe",
                    "neighborhood_consistency": (
                        0.79 if explainer_id == "wachter" else 0.68
                    ),
                    "rank_consistency": 0.75 if explainer_id == "wachter" else 0.64,
                    "direction_consistency": 0.87 if explainer_id == "wachter" else 0.84,
                    "seed_consistency": 0.77 if explainer_id == "wachter" else 0.62,
                },
            },
        }

    if explainer_id in {"random_reasons", "global_ranking"}:
        feature_names = dataset_bundle.included_feature_names
        if explainer_id == "random_reasons":
            weights = [0.91, -0.62, 0.37]
            raw_values = [3, 1500, 1]
            settings = {
                "weight_distribution": "sign_uniform_pm1_times_magnitude_uniform_0_1",
                "generation_mode": "full",
            }
        else:
            weights = [0.44, 0.31, 0.12]
            raw_values = [2, 4100, 0]
            settings = {
                "aggregation": "mean_abs_contribution_over_background_rows",
                "contribution_source": "demo_reference_contributions",
            }
        return {
            "instance_id": f"{dataset_bundle.dataset_id}-raw-{explainer_id}-001",
            "model_id": "logistic_regression",
            "seed": 0 if explainer_id == "random_reasons" else None,
            "prediction": {
                "score": 0.7,
                "label": dataset_bundle.negative_label,
                "threshold": 0.5,
                "decision_margin": 0.2,
            },
            "local_explanation": [
                {
                    "feature_name": feature_names[index],
                    "weight": weights[index],
                    "raw_value": raw_values[index],
                }
                for index in range(3)
            ],
            "metadata": {
                "raw_family": "attribution",
                "null_baseline_settings": {
                    **settings,
                    "num_features": 3,
                    "faithfulness_reference": "self",
                    "reference_contribution_source": "self",
                },
                "faithfulness_evidence": {
                    "source": f"null_{explainer_id}",
                    "reference_signed_contributions": {
                        feature_names[index]: weights[index] for index in range(3)
                    },
                },
                "stability_evidence": {
                    "source": "demo_local_neighborhood_probe",
                    "neighborhood_consistency": (
                        0.2 if explainer_id == "random_reasons" else 1.0
                    ),
                    "rank_consistency": 0.15 if explainer_id == "random_reasons" else 1.0,
                    "direction_consistency": (
                        0.5 if explainer_id == "random_reasons" else 1.0
                    ),
                    "seed_consistency": 0.2 if explainer_id == "random_reasons" else 1.0,
                },
            },
        }

    raise KeyError(f"Unsupported explainer example '{explainer_id}'.")
