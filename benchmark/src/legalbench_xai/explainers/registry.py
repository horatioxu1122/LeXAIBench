from dataclasses import dataclass
from typing import Any

from ..capabilities import capability_matrix
from ..corrections import model_corrections
from ..models.real import VectorizedClassifierModel
from ..models.stubs import RowPrediction
from ..types import DatasetBundle
from .generators import generate_raw_explanation as generate_stub_raw_explanation
from .real import (
    generate_real_anchor_explanation,
    generate_real_counterfactual_explanation,
    generate_real_global_ranking_explanation,
    generate_real_leave_one_out_explanation,
    generate_real_lime_explanation,
    generate_real_random_reasons_explanation,
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
    if explainer_id == "random_reasons" and caps["sklearn"]:
        return ExplainerBackendSelection(
            explainer_id=explainer_id,
            backend="real",
            implementation="null_random_reasons",
            reason=(
                "the random-reasons null baseline (seeded random signed weights) is wired "
                "for the current vectorized real model slice"
            ),
        )
    if explainer_id == "global_ranking" and caps["sklearn"]:
        return ExplainerBackendSelection(
            explainer_id=explainer_id,
            backend="real",
            implementation="null_global_ranking",
            reason=(
                "the global-ranking null baseline (one canned reason list per model) is "
                "wired for the current vectorized real model slice"
            ),
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
    elif (
        selection.backend == "real"
        and explainer_id == "random_reasons"
        and isinstance(model, VectorizedClassifierModel)
    ):
        try:
            payload = generate_real_random_reasons_explanation(
                model=model,
                row=row,
                prediction=prediction,
                seed=0 if seed is None else seed,
                generation_mode=generation_mode,
            )
        except Exception as exc:
            selection = ExplainerBackendSelection(
                explainer_id=explainer_id,
                backend="stub",
                implementation="demo_generator",
                reason=(
                    "real random_reasons backend failed to initialize: "
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
        and explainer_id == "global_ranking"
        and isinstance(model, VectorizedClassifierModel)
    ):
        try:
            payload = generate_real_global_ranking_explanation(
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
                    "real global_ranking backend failed to initialize: "
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
        elif selection.backend == "real" and explainer_id in {"random_reasons", "global_ranking"}:
            selection = ExplainerBackendSelection(
                explainer_id=explainer_id,
                backend="stub",
                implementation="demo_generator",
                reason=(
                    f"real {explainer_id} null baseline is currently wired only for the "
                    "current vectorized real model slice"
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
    if model_corrections(model).common_faithfulness_reference:
        _apply_common_faithfulness_reference(
            payload=payload,
            model=model,
            row=row,
            generation_mode=generation_mode,
        )
    payload["metadata"]["backend_selection"] = {
        "backend": selection.backend,
        "implementation": selection.implementation,
        "reason": selection.reason,
    }
    return payload, selection


COMMON_REFERENCE_SOURCE = "model_native_common_reference"


def _apply_common_faithfulness_reference(
    *,
    payload: dict[str, Any],
    model: Any,
    row: dict[str, Any],
    generation_mode: str,
) -> None:
    """Correction common_faithfulness_reference (v2): one reference for every explainer.

    v1 scored SHAP, LIME and leave-one-out against their own outputs, Anchors against
    the (heuristic) contributions restricted to its own anchor features (coverage 1.0
    by construction), and DiCE/Wachter structurally. Here every explainer id, the null
    baselines and the stub fallbacks included, is scored against the same reference:
    model.native_feature_contributions(row) over ALL included features (log-odds of the
    adverse class; heuristic proxy for stub models, flagged by
    reference_contribution_source). The explainer's own reference, if any, is kept
    under "self_reference_signed_contributions" and its own source under "self_source".

    Skipped for generation_mode == "stability_probe": the stability probe compares only
    feature order and directions of the neighbours' canonical explanations and never
    reads faithfulness evidence, so injecting there would only add one native
    decomposition per neighbour (up to ~72 per instance) without changing any score.
    A failing native decomposition raises (no silent fallback to a self-reference).
    """
    if generation_mode == "stability_probe":
        return
    metadata = payload["metadata"]
    evidence = dict(metadata.get("faithfulness_evidence") or {})
    if "reference_signed_contributions" in evidence:
        evidence["self_reference_signed_contributions"] = evidence.pop(
            "reference_signed_contributions"
        )
    if "source" in evidence:
        evidence["self_source"] = evidence.pop("source")
    metadata["faithfulness_evidence"] = {
        "source": COMMON_REFERENCE_SOURCE,
        "reference_signed_contributions": {
            str(feature_name): float(value)
            for feature_name, value in model.native_feature_contributions(row).items()
        },
        "reference_contribution_source": str(
            getattr(model, "native_contribution_source", "unknown")
        ),
        **evidence,
    }


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
            "random_reasons",
            "global_ranking",
        )
    ]
