from dataclasses import dataclass
from typing import Any

from ..capabilities import capability_matrix
from ..models.real import VectorizedClassifierModel
from ..models.stubs import RowPrediction
from ..types import DatasetBundle
from .generators import generate_raw_explanation as generate_stub_raw_explanation
from .real import (
    generate_real_anchor_explanation,
    generate_real_counterfactual_explanation,
    generate_real_leave_one_out_explanation,
    generate_real_lime_explanation,
    generate_real_shap_explanation,
)


@dataclass(slots=True)
class ExplainerBackendSelection:
    explainer_id: str
    backend: str
    implementation: str
    reason: str


def select_explainer_backend(explainer_id: str) -> ExplainerBackendSelection:
    caps = capability_matrix()
    if explainer_id == "shap" and caps["shap"]:
        return ExplainerBackendSelection(
            explainer_id=explainer_id,
            backend="real",
            implementation="shap_library",
            reason="shap is available and the real SHAP backend is wired for the current vectorized model slice",
        )
    if explainer_id == "lime" and caps["lime"]:
        return ExplainerBackendSelection(
            explainer_id=explainer_id,
            backend="real",
            implementation="lime_library",
            reason="lime is available and the real LIME backend is wired for the current vectorized model slice",
        )
    if explainer_id == "leave_one_out" and caps["sklearn"]:
        return ExplainerBackendSelection(
            explainer_id=explainer_id,
            backend="real",
            implementation="leave_one_out_feature_ablation",
            reason="the generic leave-one-feature-out attribution baseline is wired for the current vectorized real model slice",
        )
    if explainer_id == "anchors" and caps["sklearn"]:
        return ExplainerBackendSelection(
            explainer_id=explainer_id,
            backend="real",
            implementation="search_anchor_rule",
            reason="the model-validated anchor-style rule search backend is wired for the current vectorized real model slice",
        )
    if explainer_id == "dice" and caps["sklearn"]:
        return ExplainerBackendSelection(
            explainer_id=explainer_id,
            backend="real",
            implementation="search_counterfactual_dice_style",
            reason="the model-validated DiCE-style search backend is wired for the current vectorized real model slice",
        )
    if explainer_id == "wachter" and caps["sklearn"]:
        return ExplainerBackendSelection(
            explainer_id=explainer_id,
            backend="real",
            implementation="search_counterfactual_wachter_style",
            reason="the model-validated Wachter-style search backend is wired for the current vectorized real model slice",
        )
    return ExplainerBackendSelection(
        explainer_id=explainer_id,
        backend="stub",
        implementation="demo_generator",
        reason="required runtime dependency is unavailable or not yet wired",
    )


def generate_raw_explanation(
    explainer_id: str,
    dataset_bundle: DatasetBundle,
    model: Any,
    row: dict[str, Any],
    prediction: RowPrediction,
    seed: int | None = None,
    generation_mode: str = "full",
) -> tuple[dict[str, Any], ExplainerBackendSelection]:
    selection = select_explainer_backend(explainer_id)
    payload: dict[str, Any]
    if (
        selection.backend == "real"
        and explainer_id == "shap"
        and isinstance(model, VectorizedClassifierModel)
    ):
        try:
            payload = generate_real_shap_explanation(
                model=model,
                row=row,
                prediction=prediction,
            )
        except Exception as exc:
            selection = ExplainerBackendSelection(
                explainer_id=explainer_id,
                backend="stub",
                implementation="demo_generator",
                reason=f"real SHAP backend failed to initialize: {exc.__class__.__name__}",
            )
            payload = generate_stub_raw_explanation(
                explainer_id=explainer_id,
                dataset_bundle=dataset_bundle,
                model=model,
                row=row,
                prediction=prediction,
                seed=seed,
                generation_mode=generation_mode,
            )
    elif (
        selection.backend == "real"
        and explainer_id == "lime"
        and isinstance(model, VectorizedClassifierModel)
    ):
        try:
            payload = generate_real_lime_explanation(
                model=model,
                row=row,
                prediction=prediction,
                seed=0 if seed is None else seed,
            )
        except Exception as exc:
            selection = ExplainerBackendSelection(
                explainer_id=explainer_id,
                backend="stub",
                implementation="demo_generator",
                reason=f"real LIME backend failed to initialize: {exc.__class__.__name__}",
            )
            payload = generate_stub_raw_explanation(
                explainer_id=explainer_id,
                dataset_bundle=dataset_bundle,
                model=model,
                row=row,
                prediction=prediction,
                seed=seed,
                generation_mode=generation_mode,
            )
    elif (
        selection.backend == "real"
        and explainer_id == "leave_one_out"
        and isinstance(model, VectorizedClassifierModel)
    ):
        try:
            payload = generate_real_leave_one_out_explanation(
                model=model,
                row=row,
                prediction=prediction,
            )
        except Exception as exc:
            selection = ExplainerBackendSelection(
                explainer_id=explainer_id,
                backend="stub",
                implementation="demo_generator",
                reason=(
                    "real leave-one-out backend failed to initialize: "
                    f"{exc.__class__.__name__}"
                ),
            )
            payload = generate_stub_raw_explanation(
                explainer_id=explainer_id,
                dataset_bundle=dataset_bundle,
                model=model,
                row=row,
                prediction=prediction,
                seed=seed,
                generation_mode=generation_mode,
            )
    elif (
        selection.backend == "real"
        and explainer_id == "anchors"
        and isinstance(model, VectorizedClassifierModel)
    ):
        try:
            payload = generate_real_anchor_explanation(
                model=model,
                row=row,
                prediction=prediction,
                generation_mode=generation_mode,
            )
        except Exception as exc:
            selection = ExplainerBackendSelection(
                explainer_id=explainer_id,
                backend="stub",
                implementation="demo_generator",
                reason=f"real anchors backend failed to initialize: {exc.__class__.__name__}",
            )
            payload = generate_stub_raw_explanation(
                explainer_id=explainer_id,
                dataset_bundle=dataset_bundle,
                model=model,
                row=row,
                prediction=prediction,
                seed=seed,
                generation_mode=generation_mode,
            )
    elif (
        selection.backend == "real"
        and explainer_id in {"dice", "wachter"}
        and isinstance(model, VectorizedClassifierModel)
    ):
        try:
            payload = generate_real_counterfactual_explanation(
                model=model,
                row=row,
                prediction=prediction,
                explainer_id=explainer_id,
                generation_mode=generation_mode,
            )
        except Exception as exc:
            selection = ExplainerBackendSelection(
                explainer_id=explainer_id,
                backend="stub",
                implementation="demo_generator",
                reason=(
                    f"real {explainer_id} backend failed to initialize: "
                    f"{exc.__class__.__name__}"
                ),
            )
            payload = generate_stub_raw_explanation(
                explainer_id=explainer_id,
                dataset_bundle=dataset_bundle,
                model=model,
                row=row,
                prediction=prediction,
                seed=seed,
                generation_mode=generation_mode,
            )
    else:
        if selection.backend == "real" and explainer_id == "shap":
            selection = ExplainerBackendSelection(
                explainer_id=explainer_id,
                backend="stub",
                implementation="demo_generator",
                reason="real SHAP backend is currently wired only for the current vectorized real model slice",
            )
        elif selection.backend == "real" and explainer_id == "lime":
            selection = ExplainerBackendSelection(
                explainer_id=explainer_id,
                backend="stub",
                implementation="demo_generator",
                reason="real LIME backend is currently wired only for the current vectorized real model slice",
            )
        elif selection.backend == "real" and explainer_id == "leave_one_out":
            selection = ExplainerBackendSelection(
                explainer_id=explainer_id,
                backend="stub",
                implementation="demo_generator",
                reason=(
                    "real leave-one-out backend is currently wired only for the current "
                    "vectorized real model slice"
                ),
            )
        elif selection.backend == "real" and explainer_id == "anchors":
            selection = ExplainerBackendSelection(
                explainer_id=explainer_id,
                backend="stub",
                implementation="demo_generator",
                reason="real anchors backend is currently wired only for the current vectorized real model slice",
            )
        elif selection.backend == "real" and explainer_id in {"dice", "wachter"}:
            selection = ExplainerBackendSelection(
                explainer_id=explainer_id,
                backend="stub",
                implementation="demo_generator",
                reason=(
                    f"real {explainer_id} backend is currently wired only for the current "
                    "vectorized real model slice"
                ),
            )
        payload = generate_stub_raw_explanation(
            explainer_id=explainer_id,
            dataset_bundle=dataset_bundle,
            model=model,
            row=row,
            prediction=prediction,
            seed=seed,
            generation_mode=generation_mode,
        )
    payload.setdefault("metadata", {})
    payload["metadata"]["backend_selection"] = {
        "backend": selection.backend,
        "implementation": selection.implementation,
        "reason": selection.reason,
    }
    return payload, selection


def supported_explainer_backends() -> list[ExplainerBackendSelection]:
    return [
        select_explainer_backend(explainer_id)
        for explainer_id in (
            "shap",
            "lime",
            "leave_one_out",
            "anchors",
            "dice",
            "wachter",
        )
    ]
