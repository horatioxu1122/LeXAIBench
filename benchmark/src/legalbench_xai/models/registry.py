from dataclasses import dataclass
from typing import Any

from ..capabilities import capability_matrix
from ..types import DatasetBundle, TabularDataset
from .real import fit_ebm_model, fit_sklearn_logistic_regression_model, fit_xgboost_model
from .stubs import fit_stub_model


@dataclass(slots=True)
class ModelBackendSelection:
    model_id: str
    backend: str
    implementation: str
    reason: str


def select_model_backend(model_id: str) -> ModelBackendSelection:
    caps = capability_matrix()
    if model_id == "logistic_regression" and caps["sklearn"]:
        return ModelBackendSelection(
            model_id=model_id,
            backend="real",
            implementation="sklearn_logistic_regression",
            reason="scikit-learn is available and the real logistic regression backend is wired",
        )
    if model_id == "ebm" and caps["interpret"] and caps["sklearn"]:
        return ModelBackendSelection(
            model_id=model_id,
            backend="real",
            implementation="ebm_classifier",
            reason="interpret and scikit-learn are available and the real EBM backend is wired",
        )
    if model_id == "xgboost" and caps["xgboost"] and caps["sklearn"]:
        return ModelBackendSelection(
            model_id=model_id,
            backend="real",
            implementation="xgboost_classifier",
            reason="xgboost and scikit-learn are available and the real XGBoost backend is wired",
        )
    return ModelBackendSelection(
        model_id=model_id,
        backend="stub",
        implementation="heuristic_model",
        reason="required runtime dependency is unavailable",
    )


def fit_model(
    model_id: str,
    dataset_bundle: DatasetBundle,
    table: TabularDataset,
    threshold: float = 0.5,
    random_state: int = 0,
) -> tuple[Any, ModelBackendSelection]:
    """Fit a model. random_state seeds the estimator (0 in every published run); XGBoost with subsample=1.0
    and logistic regression with lbfgs give the same model for any seed, the EBM's bagging does not."""
    selection = select_model_backend(model_id)
    if selection.backend == "real" and model_id == "logistic_regression":
        try:
            model = fit_sklearn_logistic_regression_model(
                dataset_bundle=dataset_bundle,
                table=table,
                threshold=threshold,
                random_state=random_state,
            )
            return model, selection
        except Exception as exc:
            selection = ModelBackendSelection(
                model_id=model_id,
                backend="stub",
                implementation="heuristic_model",
                reason=(
                    "real logistic regression backend failed to initialize: "
                    f"{exc.__class__.__name__}"
                ),
            )
    elif selection.backend == "real" and model_id == "ebm":
        try:
            model = fit_ebm_model(
                dataset_bundle=dataset_bundle,
                table=table,
                threshold=threshold,
                random_state=random_state,
            )
            return model, selection
        except Exception as exc:
            selection = ModelBackendSelection(
                model_id=model_id,
                backend="stub",
                implementation="heuristic_model",
                reason=(
                    "real EBM backend failed to initialize: "
                    f"{exc.__class__.__name__}"
                ),
            )
    elif selection.backend == "real" and model_id == "xgboost":
        try:
            model = fit_xgboost_model(
                dataset_bundle=dataset_bundle,
                table=table,
                threshold=threshold,
                random_state=random_state,
            )
            return model, selection
        except Exception as exc:
            selection = ModelBackendSelection(
                model_id=model_id,
                backend="stub",
                implementation="heuristic_model",
                reason=(
                    "real XGBoost backend failed to initialize: "
                    f"{exc.__class__.__name__}"
                ),
            )
    model = fit_stub_model(model_id=model_id, dataset_bundle=dataset_bundle, table=table)
    return model, selection


def supported_model_backends() -> list[ModelBackendSelection]:
    return [
        select_model_backend(model_id)
        for model_id in ("logistic_regression", "ebm", "xgboost")
    ]
