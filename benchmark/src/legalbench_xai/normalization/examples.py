from ..types import DatasetBundle


def sample_payload(dataset_bundle: DatasetBundle) -> dict:
    dataset_id = dataset_bundle.dataset_id

    if dataset_id == "heloc":
        return {
            "instance_id": "demo-heloc-001",
            "model_id": "xgboost",
            "explainer_id": "shap",
            "explanation_family": "attribution",
            "prediction": {
                "score": 0.84,
                "label": dataset_bundle.negative_label,
                "threshold": 0.5,
                "decision_margin": 0.34,
            },
            "features": [
                {
                    "feature_name": "NetFractionRevolvingBurden",
                    "importance_weight": 0.32,
                    "raw_value": 87.0,
                },
                {
                    "feature_name": "NumInqLast6M",
                    "importance_weight": 0.21,
                    "raw_value": 5,
                },
                {
                    "feature_name": "MSinceMostRecentDelq",
                    "importance_weight": 0.18,
                    "raw_value": 2,
                },
            ],
            "metadata": {
                "demo": True,
                "faithfulness_evidence": {
                    "source": "demo_reference_contributions",
                    "local_quality_score": 0.98,
                    "reference_signed_contributions": {
                        "NetFractionRevolvingBurden": 0.32,
                        "NumInqLast6M": 0.21,
                        "MSinceMostRecentDelq": 0.18,
                        "NumBank2NatlTradesWHighUtilization": 0.11,
                    },
                },
                "stability_evidence": {
                    "source": "demo_local_neighborhood_probe",
                    "neighborhood_consistency": 0.91,
                    "rank_consistency": 0.89,
                    "direction_consistency": 0.96,
                    "seed_consistency": 0.95,
                },
            },
            "counterfactual_changes": [],
        }

    if dataset_id == "default_of_credit_card_clients":
        return {
            "instance_id": "demo-default-001",
            "model_id": "logistic_regression",
            "explainer_id": "lime",
            "explanation_family": "attribution",
            "prediction": {
                "score": 0.77,
                "label": dataset_bundle.negative_label,
                "threshold": 0.5,
                "decision_margin": 0.27,
            },
            "features": [
                {"feature_name": "PAY_0", "importance_weight": 0.28, "raw_value": 2},
                {"feature_name": "PAY_2", "importance_weight": 0.19, "raw_value": 2},
                {"feature_name": "BILL_AMT1", "importance_weight": 0.16, "raw_value": 56000},
                {"feature_name": "PAY_AMT1", "importance_weight": -0.08, "raw_value": 1000}
            ],
            "metadata": {
                "demo": True,
                "faithfulness_evidence": {
                    "source": "demo_reference_contributions",
                    "local_quality_score": 0.9,
                    "reference_signed_contributions": {
                        "PAY_0": 0.28,
                        "PAY_2": 0.19,
                        "BILL_AMT1": 0.16,
                        "PAY_AMT1": -0.08,
                    },
                },
                "stability_evidence": {
                    "source": "demo_local_neighborhood_probe",
                    "neighborhood_consistency": 0.64,
                    "rank_consistency": 0.59,
                    "direction_consistency": 0.83,
                    "seed_consistency": 0.54,
                },
            },
            "counterfactual_changes": [],
        }

    return {
        "instance_id": "demo-german-001",
        "model_id": "ebm",
        "explainer_id": "dice",
        "explanation_family": "counterfactual",
        "prediction": {
            "score": 0.72,
            "label": dataset_bundle.negative_label,
            "threshold": 0.5,
            "decision_margin": 0.22,
        },
        "features": [
            {"feature_name": "installment_rate", "importance_weight": 0.23, "raw_value": 4},
            {"feature_name": "credit_history", "importance_weight": 0.19, "raw_value": "critical"},
            {"feature_name": "amount", "importance_weight": 0.12, "raw_value": 4200}
        ],
        "metadata": {
            "demo": True,
            "faithfulness_evidence": {
                "source": "demo_reference_contributions",
                "local_quality_score": 0.86,
                "reference_signed_contributions": {
                    "installment_rate": 0.23,
                    "credit_history": 0.19,
                    "amount": 0.12,
                },
            },
            "stability_evidence": {
                "source": "demo_counterfactual_neighbor_probe",
                "neighborhood_consistency": 0.69,
                "rank_consistency": 0.65,
                "direction_consistency": 0.85,
                "seed_consistency": 0.63,
            },
        },
        "counterfactual_changes": [
            {
                "feature_name": "installment_rate",
                "from_value": 4,
                "to_value": 2,
                "validity_flip": True,
                "realism_score": 0.8,
                "effort_score": 0.6
            },
            {
                "feature_name": "amount",
                "from_value": 4200,
                "to_value": 3000,
                "validity_flip": True,
                "realism_score": 0.9,
                "effort_score": 0.7
            }
        ],
    }
