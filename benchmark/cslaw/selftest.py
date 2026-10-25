#!/usr/bin/env python3
"""End-to-end synthetic self-test for the CS&Law scripts (cluster step 0).

    PYTHONPATH=benchmark/src python benchmark/cslaw/selftest.py [--keep DIR]
        [--skip-script-selftests]

What it does, on SYNTHETIC data only (no dataset is read, no model is fitted, no
explainer is run):

1. Runs the built-in ``--selftest`` of rescore, paper_tables, compare_bundles and
   denial_reasons (skip with --skip-script-selftests).
2. Checks the cross-agent contract: the runner's ``cslaw`` config parsing and
   echo, the null-baseline ids in common/runner/generators, the model API names
   (native_feature_contributions_batch, ranking_contributions, use_native_ranking,
   native_contribution_source), the random_reasons RNG recipe, the null-baseline
   adapters, and runner._strict_json.
3. Builds a small results.json (2 datasets x 2 tracks x 3 models x 5 real
   explainers + 2 null baselines, 6 applicants per group) with the runner's own
   cslaw helpers (_cslaw_select_shared_positions, _cslaw_prepare_model_group,
   _cslaw_attach_instance_fields, _aggregate_group, _build_overview) around a fake
   additive model. The persisted fields (eval_row_index, canonical_explanation,
   native_contributions, payload["cslaw"] with regb_reference and shared_sampling)
   therefore have exactly the shape the pipeline writes. It derives the bundles
   the cluster produces: shared (all explainers), native (real explainers),
   nullbaselines (null baselines, native contributions removed to exercise the
   cross-bundle join) and an "old" bundle without any persisted field.
4. Calls the pure functions and main() of rescore, paper_tables, compare_bundles,
   regb_reference, multiplicity, proxy_substitution and common_faithfulness, and
   asserts that the expected
   output files exist and that a few invariants hold, e.g. the rescored
   css_no_dsv equals the renormalized formula, compare_bundles reports a copy as
   identical, Reg B M1 reasons match an independent recomputation, and every
   analysis refuses the old bundle with a clear message.
5. v2 (versioned corrections): legalbench_xai.corrections parsing (presets,
   overrides, unknown keys, the legacy native flag) and the mirror in common.py;
   the faithfulness renormalization identity at w = 0.20; the scorer's v2
   sub-components; the exposed pure fixes (PAY_AMT grouping, exponent-notation
   parsing, counterfactual reasons from changes, stratified-random sampling),
   each skipped with a message if its signature differs from the contract; a
   v2-shaped bundle (corrections, provenance with v2 metadata files, persisted
   sub-components, cf_reason_source) through rescore --variant
   v2_legal_functions, paper_tables (T1 status, T3 open faithfulness, T8),
   regb_reference (v2 flags), common_faithfulness (pipeline consistency),
   proxy_substitution (provenance registry) and multiplicity; then
   merge_bundles on shards of that bundle (exact reassembly, refusals) and
   version_diff between the v1-shaped and the v2-shaped bundle.

It imports legalbench_xai only for the pure runner helpers listed above (rescore.py
imports the same module) plus the dataclasses in legalbench_xai.types. Outputs go to
a temporary directory that is deleted afterwards unless --keep DIR is given.
"""

from __future__ import annotations

import argparse
import copy
import csv
import inspect
import json
import math
import os
import random
import shutil
import sys
import tempfile
import zlib
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
BENCHMARK_SRC = REPO_ROOT / "benchmark" / "src"
sys.path.insert(0, str(SCRIPT_DIR))
if BENCHMARK_SRC.is_dir() and str(BENCHMARK_SRC) not in sys.path:
    sys.path.insert(1, str(BENCHMARK_SRC))

import common  # noqa: E402
import common_faithfulness  # noqa: E402
import compare_bundles  # noqa: E402
import denial_reasons  # noqa: E402
import merge_bundles  # noqa: E402
import multiplicity  # noqa: E402
import paper_tables  # noqa: E402
import proxy_substitution  # noqa: E402
import regb_reference  # noqa: E402
import rescore  # noqa: E402
import version_diff  # noqa: E402

try:
    from legalbench_xai import corrections as corrections_module  # noqa: E402
    from legalbench_xai import runner  # noqa: E402
    from legalbench_xai.adapters import adapt_raw_example  # noqa: E402
    from legalbench_xai.adapters.examples import sample_raw_payload  # noqa: E402
    from legalbench_xai.datasets import io as dataset_io  # noqa: E402
    from legalbench_xai.explainers import generators  # noqa: E402
    from legalbench_xai.metrics.scorer import ExplanationScorer  # noqa: E402
    from legalbench_xai.models.real import (  # noqa: E402
        EBMClassifierModel,
        SklearnLogisticRegressionModel,
        VectorizedClassifierModel,
        XGBoostClassifierModel,
    )
    from legalbench_xai.models.stubs import HeuristicModel  # noqa: E402
    from legalbench_xai.normalization import rules as normalization_rules  # noqa: E402
    from legalbench_xai.types import (  # noqa: E402
        CanonicalExplanation,
        CounterfactualChange,
        DatasetBundle,
        FeatureSpec,
        PredictionRecord,
        PrincipalReason,
        ReferencedFeature,
    )
except ImportError as import_error:  # pragma: no cover - environment guard
    raise SystemExit(
        "selftest: cannot import legalbench_xai. Run from the repository root with the "
        "benchmark environment, e.g.\n"
        "  PYTHONPATH=benchmark/src .venv/bin/python benchmark/cslaw/selftest.py\n"
        f"({import_error!r})"
    ) from import_error


# ---------------------------------------------------------------------------
# Synthetic world
# ---------------------------------------------------------------------------

# feature -> (plain meaning, taxonomy sensitivity). Dataset ids are real credit
# ids; the features and the metadata written below are synthetic.
DATASETS: dict[str, dict[str, tuple[str, str]]] = {
    "heloc": {
        "ExternalRiskEstimate": ("external risk estimate", "P"),
        "NumInqLast6M": ("number of recent credit inquiries", "N"),
        "NetFractionRevolvingBurden": ("revolving balance relative to limit", "N"),
        "MaxDelqEver": ("worst delinquency on record", "N"),
        "PercentTradesNeverDelq": ("share of accounts never delinquent", "N"),
        "AverageMInFile": ("average account age in months", "P"),
    },
    "south_german_credit": {
        "status": ("checking account status", "P"),
        "duration": ("loan duration in months", "N"),
        "amount": ("requested loan amount", "N"),
        "savings": ("savings balance", "P"),
        "age": ("applicant age in years", "S"),
        "personal_status_sex": ("personal status and sex", "S"),
        "foreign_worker": ("foreign worker status", "S"),
    },
}
PRIMARY_EXCLUDED: dict[str, list[str]] = {
    "heloc": ["ExternalRiskEstimate"],
    "south_german_credit": ["personal_status_sex", "foreign_worker"],
}
TRACKS = ("primary", "audit")
# model_id -> (backend implementation, native_contribution_source)
MODELS: dict[str, tuple[str, str]] = {
    "logistic_regression": ("sklearn_logistic_regression", "lr_coef_x_standardized"),
    "ebm": ("ebm_classifier", "ebm_eval_terms_pair_split"),
    "xgboost": ("xgboost_classifier", "xgb_pred_contribs"),
}
EXPLAINERS = (*common.REAL_EXPLAINERS, *common.NULL_BASELINES)
TARGET, NEGATIVE, POSITIVE = "risk", "bad", "good"
N_ROWS = 40
MAX_INSTANCES = 6
K = 4
N_BOOT = 50

CSLAW_BLOCK: dict[str, Any] = {
    "use_native_contributions_for_ranking": True,
    "persist_canonical_explanations": True,
    "persist_native_contributions": True,
    "regb_reference": {
        "enabled": True,
        "near_cutoff_min_rows": 5,
        "near_cutoff_max_band": 0.10,
        "all_applicants_max_rows": 30,
        "seed": 42,
    },
    "shared_sampling": {"enabled": True, "scope": "models_and_tracks", "margin_aggregate": "min"},
    "null_baselines": {"faithfulness_reference": "self", "num_features": 6},
}
SAMPLING_CFG: dict[str, Any] = {
    "max_instances_per_dataset_model_pair": MAX_INSTANCES,
    "stratify_by_decision_margin": True,
    "strata": ["borderline", "medium_confidence", "strong_denial"],
}
MODEL_PERFORMANCE: dict[str, Any] = {
    "total_rows": N_ROWS,
    "decision_threshold": 0.5,
    "accuracy": 0.8,
    "roc_auc": 0.85,
    "average_precision": 0.7,
    "brier_score": 0.15,
    "expected_calibration_error": 0.05,
    "positive_rate": 0.4,
    "predicted_positive_rate": 0.45,
}
SOURCE_TO_MODEL = {source: model_id for model_id, (_impl, source) in MODELS.items()}


def seeded(*parts: Any) -> random.Random:
    """Private RNG seeded from the parts via crc32 (stable across processes)."""
    return random.Random(zlib.crc32("|".join(str(part) for part in parts).encode("utf-8")))


def included_features(dataset_id: str, track: str) -> list[str]:
    excluded = set(PRIMARY_EXCLUDED[dataset_id]) if track == "primary" else set()
    return [name for name in DATASETS[dataset_id] if name not in excluded]


def make_rows(dataset_id: str) -> list[dict[str, Any]]:
    """Eval rows shared by both tracks (same applicants, same order).

    Rows 0-9 are adverse under every model (all values >= 0.7 with positive
    coefficients), rows 10-19 approved under every model, the rest mixed. This
    guarantees a non-empty shared pool and approved rows for the Reg B means.
    """
    rng = seeded("rows", dataset_id)
    rows: list[dict[str, Any]] = []
    for index in range(N_ROWS):
        if index < 10:
            latent = rng.uniform(1.0, 2.5)
        elif index < 20:
            latent = rng.uniform(-2.5, -1.0)
        else:
            latent = rng.uniform(-2.5, 2.5)
        row: dict[str, Any] = {
            name: round(latent + rng.uniform(-0.3, 0.3), 4) for name in DATASETS[dataset_id]
        }
        row[TARGET] = NEGATIVE if latent + rng.uniform(-0.5, 0.5) > 0 else POSITIVE
        rows.append(row)
    return rows


class FakeNativeModel:
    """Additive stand-in for a fitted model: contribution = coefficient x value.

    Implements only what the runner's cslaw helpers call
    (native_feature_contributions_batch, native_contribution_source, threshold,
    use_native_ranking, null_baseline_options).
    """

    use_native_ranking: bool = False

    def __init__(self, dataset_id: str, track: str, model_id: str) -> None:
        self.model_id = model_id
        self.features = included_features(dataset_id, track)
        rng = seeded("coef", dataset_id, model_id)
        self.coefficients = {name: rng.uniform(0.6, 1.4) for name in DATASETS[dataset_id]}
        self.bias = -0.2
        self.threshold = 0.5
        self.native_contribution_source = MODELS[model_id][1]
        self.null_baseline_options: dict[str, Any] = {}

    def native_feature_contributions_batch(
        self,
        rows: list[dict[str, Any]],
    ) -> tuple[list[dict[str, float]], float]:
        values = [
            {name: self.coefficients[name] * float(row[name]) for name in self.features}
            for row in rows
        ]
        return values, self.bias

    def native_feature_contributions(self, row: dict[str, Any]) -> dict[str, float]:
        return self.native_feature_contributions_batch([row])[0][0]

    def predict(self, rows: list[dict[str, Any]], prefix: str) -> list[tuple[dict[str, Any], Any]]:
        values, bias = self.native_feature_contributions_batch(rows)
        predictions: list[tuple[dict[str, Any], Any]] = []
        for index, (row, item) in enumerate(zip(rows, values, strict=True)):
            raw_score = sum(item.values()) + bias
            score = 1.0 / (1.0 + math.exp(-raw_score))
            predictions.append(
                (
                    row,
                    SimpleNamespace(
                        instance_id=f"{prefix}-row{index}",
                        score=score,
                        label=NEGATIVE if score >= self.threshold else POSITIVE,
                        threshold=self.threshold,
                        decision_margin=score - self.threshold,
                        raw_score=raw_score,
                    ),
                )
            )
        return predictions


def _weighted(values: dict[str, float], weights: dict[str, float]) -> float:
    return sum(weight * float(values[name]) for name, weight in weights.items())


def synthetic_scores(rng: random.Random, explainer_id: str) -> dict[str, Any]:
    """Scorer-shaped score blocks whose 'score' fields follow the scorer weights."""
    recourse = explainer_id in common.RECOURSE_EXPLAINERS
    fid = {
        "reference_feature_coverage": rng.uniform(0.4, 1.0),
        "directional_agreement": rng.uniform(0.4, 1.0),
        "explanation_quality_signal": (
            0.5 if common.is_null_baseline(explainer_id) else rng.uniform(0.4, 1.0)
        ),
    }
    stab = {
        "neighborhood_consistency": rng.uniform(0.2, 1.0),
        "rank_consistency": rng.uniform(0.2, 1.0),
        "direction_consistency": rng.uniform(0.2, 1.0),
        "seed_consistency": rng.uniform(0.2, 1.0),
    }
    comp = {
        "reason_count_score": rng.choice([1.0, 0.8, 0.6]),
        "feature_count_score": rng.uniform(0.2, 1.0),
        "reason_text_score": rng.uniform(0.5, 1.0),
        "recourse_simplicity_score": rng.choice([1.0, 0.8, 0.6]) if recourse else 1.0,
    }
    css = {
        "verifiability": rng.uniform(0.5, 1.0),
        "correctability": rng.uniform(0.3, 1.0),
        "decision_sensitivity_visibility": (
            rng.choice([0.7, 0.85, 1.0]) if recourse else rng.choice([0.55, 0.64])
        ),
        "reason_specificity": rng.choice([0.5, 0.75, 1.0]),
    }
    if recourse:
        feasibility, legality, minimality = (rng.uniform(0.5, 1.0) for _ in range(3))
        alf: dict[str, Any] = {
            "applicability": "applicable",
            "feasibility": feasibility,
            "legality": legality,
            "minimality": minimality,
            "invalid_recourse_rate": 0.0,
            "hard_failures": [],
            "score": feasibility * legality * minimality,
        }
        advice, alf_component = 1.0, alf["score"]
    else:
        alf = {
            "applicability": "not_applicable",
            "feasibility": None,
            "legality": None,
            "minimality": None,
            "invalid_recourse_rate": None,
            "hard_failures": [],
            "score": None,
        }
        advice, alf_component = 0.25, 0.0
    act = {
        "advice_presence": advice,
        "correction_support": css["correctability"],
        "alf_component": alf_component,
    }
    return {
        "faithfulness_score": {
            "evidence_source": "structural_proxy" if recourse else f"synthetic_{explainer_id}",
            **fid,
            "score": _weighted(fid, common.FAITHFULNESS_WEIGHTS),
        },
        "stability_score": {
            "evidence_source": "perturbation_probe",
            **stab,
            "score": _weighted(stab, common.STABILITY_WEIGHTS),
        },
        "comprehensibility_score": {
            **comp,
            "score": _weighted(comp, common.COMPREHENSIBILITY_WEIGHTS),
        },
        "contestability_support_score": {**css, "score": _weighted(css, common.CSS_WEIGHTS)},
        "actionability_legality_filter": alf,
        "unified_actionability_score": {
            **act,
            "score": _weighted(act, common.UNIFIED_ACTIONABILITY_WEIGHTS),
        },
    }


def _by_magnitude(items: dict[str, float]) -> list[tuple[str, float]]:
    return sorted(items.items(), key=lambda item: (-abs(item[1]), item[0]))


def explanation_weights(
    explainer_id: str,
    instance_id: str,
    native: dict[str, float],
    global_vector: dict[str, float],
) -> list[tuple[str, float]]:
    """Cited features and weights per explainer (deterministic stand-ins).

    shap = the native values (by construction); lime = native plus noise;
    anchors/dice/wachter = the top adverse native features; random_reasons and
    global_ranking use the real generators' helpers.
    """
    if explainer_id == "shap":
        return _by_magnitude(native)[:6]
    if explainer_id == "lime":
        rng = seeded("lime", instance_id)
        return _by_magnitude({name: value + rng.gauss(0.0, 0.3) for name, value in native.items()})[:6]
    if explainer_id == "random_reasons":
        weights = generators.random_reason_weights(
            list(native), seed=0, instance_id=instance_id, generation_mode="full"
        )
        return generators.top_null_baseline_features(weights, 6)
    if explainer_id == "global_ranking":
        return generators.top_null_baseline_features(global_vector, 6)
    adverse = sorted(
        ((name, value) for name, value in native.items() if value > 0),
        key=lambda item: (-item[1], item[0]),
    )
    return adverse[: {"anchors": 3, "dice": 2, "wachter": 1}[explainer_id]]


def _direction(weight: float) -> str:
    if weight > 0:
        return "increases_denial"
    if weight < 0:
        return "decreases_denial"
    return "neutral"


def build_canonical(
    *,
    dataset_id: str,
    track: str,
    model_id: str,
    explainer_id: str,
    row: dict[str, Any],
    prediction: Any,
    ranked: list[tuple[str, float]],
) -> CanonicalExplanation:
    """A real CanonicalExplanation, so to_dict() has the exact persisted schema."""
    family = common.EXPLAINER_FAMILY[explainer_id]
    meanings = {name: meta[0] for name, meta in DATASETS[dataset_id].items()}
    referenced = [
        ReferencedFeature(
            feature_name=name,
            plain_meaning=meanings[name],
            signed_direction=_direction(weight),
            importance_weight=float(weight),
            rank=rank,
            raw_value=row[name],
        )
        for rank, (name, weight) in enumerate(ranked, start=1)
    ]
    reasons: list[PrincipalReason] = []
    for feature in referenced[:4]:
        verb = (
            "contributed to the denial"
            if feature.signed_direction == "increases_denial"
            else "supported approval"
        )
        reasons.append(
            PrincipalReason(
                rank=len(reasons) + 1,
                reason_text=f"{feature.plain_meaning} {verb}",
                supporting_features=[feature.feature_name],
                signed_direction=feature.signed_direction,
                specificity_level="specific",
                grouped_reason_type="raw_feature",
            )
        )
    changes: list[CounterfactualChange] = []
    if family == "counterfactual":
        for name, _weight in ranked:
            changes.append(
                CounterfactualChange(
                    feature_name=name,
                    from_value=row[name],
                    to_value=round(float(row[name]) - 1.0, 4),
                    delta=-1.0,
                    change_direction="decrease",
                    validity_flip=True,
                    realism_score=0.8,
                    effort_score=0.3,
                )
            )
            reasons.append(
                PrincipalReason(
                    rank=len(reasons) + 1,
                    reason_text=f"changing {meanings[name]} may improve the outcome",
                    supporting_features=[name],
                    signed_direction="decreases_denial",
                    specificity_level="specific",
                    grouped_reason_type="counterfactual",
                )
            )
    return CanonicalExplanation(
        instance_id=prediction.instance_id,
        dataset_id=dataset_id,
        track=track,
        model_id=model_id,
        explainer_id=explainer_id,
        explanation_family=family,
        prediction=PredictionRecord(
            score=prediction.score,
            label=prediction.label,
            threshold=prediction.threshold,
            decision_margin=prediction.decision_margin,
        ),
        referenced_features=referenced,
        principal_reasons=reasons,
        counterfactual_changes=changes,
        seed=0 if explainer_id == "random_reasons" else None,
        metadata={
            "faithfulness_evidence": {
                "source": f"synthetic_{explainer_id}",
                # numpy scalars and a NaN exercise runner._strict_json.
                "reference_signed_contributions": {
                    name: np.float64(weight) for name, weight in ranked
                },
            },
            "synthetic_numpy_int": np.int64(len(ranked)),
            "synthetic_non_finite": float("nan"),
        },
    )


def build_shared_payload() -> tuple[dict[str, Any], dict[str, Any]]:
    """Synthetic shared-sampling bundle built with the runner's cslaw helpers.

    Returns (payload, facts); facts holds the fake models, predictions and the
    shared positions for independent re-computation in the assertions.
    """
    settings = runner._cslaw_settings(
        SimpleNamespace(
            payload={"cslaw": copy.deepcopy(CSLAW_BLOCK)},
            enabled_explainers=list(EXPLAINERS),
        )
    )
    assert settings is not None and settings.active
    groups: dict[str, dict[str, Any]] = {}
    summary_rows: list[dict[str, Any]] = []
    facts: dict[str, Any] = {"groups": {}, "positions": {}, "raw_scores": {}}
    for dataset_id in DATASETS:
        rows = make_rows(dataset_id)
        contexts = {
            track: SimpleNamespace(
                eval_table=SimpleNamespace(rows=rows, target=TARGET),
                bundle=SimpleNamespace(negative_label=NEGATIVE),
            )
            for track in TRACKS
        }
        models: dict[tuple[str, str], FakeNativeModel] = {}
        fitted: dict[tuple[str, str], Any] = {}
        for track in TRACKS:
            for model_id in MODELS:
                model = FakeNativeModel(dataset_id, track, model_id)
                runner._cslaw_apply_model_options(model, settings)
                assert model.use_native_ranking is True
                assert model.null_baseline_options == CSLAW_BLOCK["null_baselines"]
                models[(track, model_id)] = model
                fitted[(track, model_id)] = SimpleNamespace(
                    predictions=model.predict(rows, f"{dataset_id}-{track}-{model_id}")
                )
        keys = [(track, model_id) for track in TRACKS for model_id in MODELS]
        shared = runner._cslaw_select_shared_positions(
            contexts=contexts,
            fitted=fitted,
            keys=keys,
            sampling_cfg=SAMPLING_CFG,
            cslaw=settings,
        )
        assert not shared["shared_fallback"], shared
        positions = list(shared["selected_positions"] or [])
        assert len(positions) == MAX_INSTANCES, shared
        facts["positions"][dataset_id] = positions

        for track in TRACKS:
            for model_id, (implementation, _source) in MODELS.items():
                model = models[(track, model_id)]
                predictions = fitted[(track, model_id)].predictions
                for _row, prediction in predictions:
                    facts["raw_scores"][prediction.instance_id] = prediction.raw_score
                sampled = [predictions[position] for position in positions]
                group_shared = dict(shared)
                group_shared["model_selected_stratum_counts"] = runner._cslaw_stratum_counts(
                    [float(prediction.decision_margin) for _row, prediction in sampled]
                )
                model_backend = SimpleNamespace(
                    backend="real",
                    implementation=implementation,
                    reason="synthetic self-test",
                )
                native_records = runner._cslaw_prepare_model_group(
                    cslaw=settings,
                    cslaw_groups=groups,
                    dataset_id=dataset_id,
                    track=track,
                    model_id=model_id,
                    model=model,
                    model_backend=model_backend,
                    negative_label=NEGATIVE,
                    predictions=predictions,
                    sampled_rows=sampled,
                    shared_sampling=group_shared,
                )
                facts["groups"][common.cslaw_group_key(dataset_id, track, model_id)] = {
                    "model": model,
                    "predictions": predictions,
                }
                background, _bias = model.native_feature_contributions_batch(rows[:10])
                global_vector = {
                    name: sum(abs(item[name]) for item in background) / len(background)
                    for name in model.features
                }
                adverse_rows = sum(
                    1 for _row, prediction in predictions if prediction.label == NEGATIVE
                )
                for explainer_id in EXPLAINERS:
                    instances: list[dict[str, Any]] = []
                    for row, prediction in sampled:
                        native = model.native_feature_contributions(row)
                        canonical = build_canonical(
                            dataset_id=dataset_id,
                            track=track,
                            model_id=model_id,
                            explainer_id=explainer_id,
                            row=row,
                            prediction=prediction,
                            ranked=explanation_weights(
                                explainer_id, prediction.instance_id, native, global_vector
                            ),
                        )
                        instance: dict[str, Any] = {
                            "instance_id": prediction.instance_id,
                            "prediction_score": prediction.score,
                            "predicted_label": prediction.label,
                            "model_backend": {
                                "backend": "real",
                                "implementation": implementation,
                                "reason": "synthetic self-test",
                            },
                            "explainer_backend": {
                                "backend": "real",
                                "implementation": f"synthetic_{explainer_id}",
                                "reason": "synthetic self-test",
                            },
                            "subgroup_attributes": {},
                            "scores": synthetic_scores(
                                seeded("scores", prediction.instance_id, explainer_id),
                                explainer_id,
                            ),
                        }
                        runner._cslaw_attach_instance_fields(
                            instance,
                            cslaw=settings,
                            canonical=canonical,
                            prediction=prediction,
                            native_records=native_records,
                        )
                        instances.append(instance)
                    summary_rows.append(
                        runner._aggregate_group(
                            dataset_id=dataset_id,
                            track=track,
                            model_id=model_id,
                            explainer_id=explainer_id,
                            data_source_kind="synthetic",
                            data_source_path=f"/synthetic/{dataset_id}.csv",
                            total_rows=N_ROWS * 4,
                            train_rows=N_ROWS * 3,
                            test_rows=N_ROWS,
                            adverse_rows=adverse_rows,
                            evaluated_rows=len(instances),
                            model_backend=model_backend,
                            explainer_backend=SimpleNamespace(
                                backend="real",
                                implementation=f"synthetic_{explainer_id}",
                                reason="synthetic self-test",
                            ),
                            adverse_rate=adverse_rows / N_ROWS,
                            split_summary={"strategy": "synthetic", "heldout_used": True},
                            threshold_summary={"policy": "fixed", "threshold": 0.5},
                            model_performance=dict(MODEL_PERFORMANCE),
                            sampling_summary={
                                "strategy": f"shared_{shared['sampling_strategy']}",
                                "requested_max_instances": MAX_INSTANCES,
                                "selected_rows": len(sampled),
                                "shared_scope": shared["scope"],
                            },
                            instance_results=instances,
                        )
                    )
    payload = {
        "experiment_name": "legalbench_xai_cslaw_selftest",
        "generated_at_utc": "2026-01-01T00:00:00+00:00",
        "capabilities": {},
        "overview": runner._build_overview(summary_rows),
        "summary": summary_rows,
        "cslaw": runner._strict_json({"config": settings.to_dict(), "groups": groups}),
    }
    return payload, facts


def subset_payload(
    payload: dict[str, Any],
    explainers: Sequence[str],
    *,
    drop_native: bool = False,
) -> dict[str, Any]:
    keep = set(explainers)
    clone = copy.deepcopy(payload)
    clone["summary"] = [row for row in clone["summary"] if row["explainer_id"] in keep]
    if drop_native:
        for _row, instance in common.iter_instances(clone):
            instance.pop("native_contributions", None)
    clone["overview"] = runner._build_overview(clone["summary"])
    return clone


def strip_persisted(payload: dict[str, Any]) -> dict[str, Any]:
    """An 'older bundle': scores only, no cslaw block, no persisted instance fields."""
    clone = copy.deepcopy(payload)
    clone.pop("cslaw", None)
    for _row, instance in common.iter_instances(clone):
        for key in ("eval_row_index", "canonical_explanation", "native_contributions"):
            instance.pop(key, None)
    return clone


def write_bundle(directory: Path, payload: dict[str, Any]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    common.write_json(directory / "results.json", payload)  # allow_nan=False: strict JSON
    return directory


def write_metadata(repo_root: Path) -> None:
    """Synthetic dataset_registry.json and feature_taxonomy.json under repo_root."""
    registry = {
        "datasets": {
            dataset_id: {
                "display_name": f"{dataset_id} (synthetic)",
                "target": TARGET,
                "positive_label": POSITIVE,
                "negative_label": NEGATIVE,
                "primary_track_excluded_features": list(PRIMARY_EXCLUDED[dataset_id]),
                "audit_track_excluded_features": [],
            }
            for dataset_id in DATASETS
        }
    }
    taxonomy = {
        "datasets": {
            dataset_id: {
                "target": TARGET,
                "features": {
                    name: {
                        "plain_meaning": meaning,
                        "mutability": "M",
                        "verifiability": "Y",
                        "correctability": "P",
                        "sensitivity": sensitivity,
                        "principal_reason": "High",
                    }
                    for name, (meaning, sensitivity) in features.items()
                },
            }
            for dataset_id, features in DATASETS.items()
        }
    }
    metadata = repo_root / "benchmark" / "metadata"
    common.write_json(metadata / "dataset_registry.json", registry)
    common.write_json(metadata / "feature_taxonomy.json", taxonomy)


# ---------------------------------------------------------------------------
# Small test helpers
# ---------------------------------------------------------------------------


def call_main(main: Callable[[], None], argv: Sequence[Any]) -> int:
    """Run a script's argv-less main() with sys.argv set; return its exit code."""
    saved = sys.argv
    sys.argv = [str(item) for item in argv]
    try:
        main()
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        print(f"  exit message: {code}", file=sys.stderr)
        return 1
    finally:
        sys.argv = saved
    return 0


def expect_exit(
    label: str,
    func: Callable[..., Any],
    *args: Any,
    contains: str,
    **kwargs: Any,
) -> None:
    """Assert that func raises SystemExit with a message containing ``contains``."""
    try:
        func(*args, **kwargs)
    except SystemExit as exc:
        message = "" if exc.code is None else str(exc.code)
        assert contains in message, (label, message)
        return
    raise AssertionError(f"{label}: expected a clear SystemExit for a bundle without persisted fields")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def assert_files(paths: Sequence[Path], label: str) -> None:
    missing = [str(path) for path in paths if not Path(path).is_file()]
    assert not missing, (label, missing)


class Checklist:
    def __init__(self) -> None:
        self.passed: list[str] = []
        self.skipped: list[str] = []

    def ok(self, label: str) -> None:
        self.passed.append(label)
        print(f"  ok  {label}")


# ---------------------------------------------------------------------------
# Contract checks (no files)
# ---------------------------------------------------------------------------


def check_contract(checks: Checklist) -> None:
    def config(cslaw: dict[str, Any] | None, explainers: Sequence[str]) -> Any:
        payload = {} if cslaw is None else {"cslaw": cslaw}
        return SimpleNamespace(payload=payload, enabled_explainers=list(explainers))

    assert runner._cslaw_settings(config(None, ["shap"])) is None
    disabled = runner._cslaw_settings(config({}, ["shap", "lime"]))
    assert disabled is not None and not disabled.active
    echo = disabled.to_dict()
    assert set(echo) == set(CSLAW_BLOCK), sorted(echo)
    assert echo["regb_reference"] == {
        "enabled": False,
        "near_cutoff_min_rows": 200,
        "near_cutoff_max_band": 0.10,
        "all_applicants_max_rows": 5000,
        "seed": 42,
    }, echo["regb_reference"]
    assert echo["shared_sampling"] == {"enabled": False, "scope": "models", "margin_aggregate": "min"}
    assert echo["null_baselines"] == {"faithfulness_reference": "self", "num_features": 6}
    assert runner._cslaw_settings(config({}, ["random_reasons"])).active
    for bad in (
        {"shared_sampling": {"scope": "everything"}},
        {"shared_sampling": {"margin_aggregate": "max"}},
        {"null_baselines": {"faithfulness_reference": "oracle"}},
    ):
        try:
            runner._cslaw_settings(config(bad, ["shap"]))
        except ValueError:
            continue
        raise AssertionError(f"invalid cslaw block accepted: {bad}")
    full = runner._cslaw_settings(config(copy.deepcopy(CSLAW_BLOCK), list(EXPLAINERS)))
    assert full is not None and full.to_dict() == CSLAW_BLOCK, full.to_dict()
    checks.ok("runner cslaw config parsing, defaults and echo match the contract")

    assert tuple(runner._CSLAW_NULL_BASELINE_EXPLAINERS) == common.NULL_BASELINES
    assert tuple(generators.NULL_BASELINE_EXPLAINERS) == common.NULL_BASELINES
    for explainer_id in common.NULL_BASELINES:
        assert common.EXPLAINER_FAMILY[explainer_id] == "attribution"
    checks.ok("null-baseline ids agree in common.py, runner.py and generators.py")

    for cls in (VectorizedClassifierModel, HeuristicModel):
        for name in (
            "native_feature_contributions_batch",
            "native_feature_contributions",
            "ranking_contributions",
            "original_feature_contributions",
        ):
            assert callable(getattr(cls, name, None)), (cls.__name__, name)
        assert cls.use_native_ranking is False, cls.__name__
        assert cls.native_contribution_source == "heuristic_stub", cls.__name__
    for cls, source in (
        (SklearnLogisticRegressionModel, "lr_coef_x_standardized"),
        (EBMClassifierModel, "ebm_eval_terms_pair_split"),
        (XGBoostClassifierModel, "xgb_pred_contribs"),
    ):
        assert cls.native_contribution_source == source, (cls.__name__, cls.native_contribution_source)
        assert source in SOURCE_TO_MODEL, source
    checks.ok("model API names and native_contribution_source values match the contract")

    names = ["a", "b", "c", "d"]
    weights = generators.random_reason_weights(
        names, seed=3, instance_id="x-row1", generation_mode="full"
    )
    rng = random.Random(zlib.crc32(b"3|x-row1|full"))
    expected: dict[str, float] = {}
    for name in names:
        sign = 1.0 if rng.random() < 0.5 else -1.0
        expected[name] = sign * rng.random()
    assert weights == expected, (weights, expected)
    assert weights == generators.random_reason_weights(
        names, seed=3, instance_id="x-row1", generation_mode="full"
    )
    assert generators.resolve_null_baseline_options(SimpleNamespace()) == {
        "num_features": 6,
        "faithfulness_reference": "self",
    }
    assert generators.resolve_null_baseline_options(
        SimpleNamespace(null_baseline_options={"num_features": 3, "faithfulness_reference": "native"})
    ) == {"num_features": 3, "faithfulness_reference": "native"}
    checks.ok("random_reasons RNG recipe (crc32 of seed|instance_id|mode) and null-baseline options")

    bundle = DatasetBundle(
        dataset_id="heloc",
        display_name="HELOC (synthetic)",
        track="audit",
        task="synthetic",
        target=TARGET,
        positive_label=POSITIVE,
        negative_label=NEGATIVE,
        included_features=[
            FeatureSpec(
                feature_name=name,
                plain_meaning=meaning,
                mutability="M",
                verifiability="Y",
                correctability="P",
                sensitivity=sensitivity,
                principal_reason="High",
            )
            for name, (meaning, sensitivity) in DATASETS["heloc"].items()
        ],
        excluded_features=[],
    )
    for explainer_id in common.NULL_BASELINES:
        adapted = adapt_raw_example(
            dataset_bundle=bundle,
            explainer_id=explainer_id,
            raw_payload=sample_raw_payload(bundle, explainer_id),
        )
        assert adapted["explainer_id"] == explainer_id
        assert adapted["explanation_family"] == "attribution"
        assert adapted["features"] and adapted["counterfactual_changes"] == []
        evidence = adapted["metadata"]["faithfulness_evidence"]
        assert evidence["source"] == f"null_{explainer_id}"
        assert "local_quality_score" not in evidence
    checks.ok("null-baseline adapters and demo payloads are registered")

    sanitized = runner._strict_json(
        {
            "nan": float("nan"),
            "inf": float("inf"),
            "f": np.float64(1.5),
            "i": np.int64(2),
            "t": (1, 2),
        }
    )
    assert sanitized == {"nan": None, "inf": None, "f": 1.5, "i": 2, "t": [1, 2]}, sanitized
    json.dumps(sanitized, allow_nan=False)
    assert runner._cslaw_eval_row_index("heloc-audit-ebm-row17") == 17
    assert common.eval_row_index({"instance_id": "heloc-audit-ebm-row17"}) == 17
    checks.ok("runner._strict_json and the -row{i} applicant index")

    rows = make_rows("heloc")
    approved_only = [
        (row, SimpleNamespace(label=POSITIVE, decision_margin=-0.3)) for row in rows
    ]
    fallback = runner._cslaw_select_shared_positions(
        contexts={
            "primary": SimpleNamespace(
                eval_table=SimpleNamespace(rows=rows, target=TARGET),
                bundle=SimpleNamespace(negative_label=NEGATIVE),
            )
        },
        fitted={("primary", "logistic_regression"): SimpleNamespace(predictions=approved_only)},
        keys=[("primary", "logistic_regression")],
        sampling_cfg=SAMPLING_CFG,
        cslaw=full,
    )
    assert fallback["shared_fallback"] is True
    assert fallback["fallback_reason"] == "empty_shared_pool"
    assert fallback["selected_positions"] is None
    checks.ok("shared sampling falls back on an empty pool")


# ---------------------------------------------------------------------------
# Bundle-level checks
# ---------------------------------------------------------------------------


def expected_regb(
    model: FakeNativeModel,
    predictions: list[tuple[dict[str, Any], Any]],
) -> dict[str, Any]:
    """Independent re-implementation of the contract's Reg B reference populations."""
    settings = CSLAW_BLOCK["regb_reference"]
    threshold = predictions[0][1].threshold
    approved = sorted(
        (
            (position, prediction.score)
            for position, (_row, prediction) in enumerate(predictions)
            if prediction.label != NEGATIVE
        ),
        key=lambda item: (-item[1], item[0]),
    )
    near = [item for item in approved if item[1] >= threshold - settings["near_cutoff_max_band"]]
    if len(near) < settings["near_cutoff_min_rows"]:
        near = approved[: settings["near_cutoff_min_rows"]]
    everyone = sorted(
        random.Random(settings["seed"]).sample(
            range(len(predictions)),
            min(len(predictions), settings["all_applicants_max_rows"]),
        )
    )

    def means(positions: list[int]) -> dict[str, float]:
        values, _bias = model.native_feature_contributions_batch(
            [predictions[position][0] for position in sorted(positions)]
        )
        if not values:
            return {}
        return {name: sum(item[name] for item in values) / len(values) for name in model.features}

    return {
        "threshold": threshold,
        "near_n": len(near),
        "near_means": means([position for position, _score in near]),
        "all_n": len(everyone),
        "all_means": means(everyone),
    }


def check_bundle_schema(payload: dict[str, Any], facts: dict[str, Any], checks: Checklist) -> None:
    block = payload["cslaw"]
    assert block["config"] == CSLAW_BLOCK, block["config"]
    groups = block["groups"]
    expected_keys = {
        common.cslaw_group_key(dataset_id, track, model_id)
        for dataset_id in DATASETS
        for track in TRACKS
        for model_id in MODELS
    }
    assert set(groups) == expected_keys, sorted(groups)
    for key, group in groups.items():
        _dataset_id, _track, model_id = key.split("|")
        assert not group.get("errors"), (key, group.get("errors"))
        assert group["model_backend"] == MODELS[model_id][0]
        assert group["native_contribution_source"] == MODELS[model_id][1]
        shared = group["shared_sampling"]
        for name in (
            "scope",
            "margin_aggregate",
            "shared_pool_rows",
            "selected_positions",
            "selected_stratum_counts",
            "model_selected_stratum_counts",
            "shared_fallback",
        ):
            assert name in shared, (key, name)
        regb = group["regb_reference"]
        expected = expected_regb(facts["groups"][key]["model"], facts["groups"][key]["predictions"])
        assert regb["source"] == MODELS[model_id][1]
        assert regb["threshold"] == expected["threshold"]
        assert regb["near_cutoff"]["n_rows"] == expected["near_n"], (key, regb["near_cutoff"])
        assert regb["all_applicants"]["n_rows"] == expected["all_n"]
        assert regb["near_cutoff"]["score_low"] <= regb["near_cutoff"]["score_high"] < 0.5
        for population, reference in (("near_cutoff", "near_means"), ("all_applicants", "all_means")):
            stored = regb[population]["means"]
            assert stored and set(stored) == set(expected[reference]), (key, population)
            for name, value in expected[reference].items():
                assert abs(stored[name] - value) < 1e-9, (key, population, name)
    checks.ok("payload['cslaw'] groups: config echo, backends, sources, Reg B populations and means")

    for dataset_id in DATASETS:
        selected = {
            json.dumps(group["shared_sampling"]["selected_positions"])
            for key, group in groups.items()
            if key.startswith(f"{dataset_id}|")
        }
        assert selected == {json.dumps(facts["positions"][dataset_id])}, (dataset_id, selected)
    for row, instance in common.iter_instances(payload):
        assert instance["eval_row_index"] in facts["positions"][row["dataset_id"]]
        assert instance["eval_row_index"] == common.eval_row_index(
            {"instance_id": instance["instance_id"]}
        )
        native = instance["native_contributions"]
        assert native["units"] == "log_odds_adverse"
        assert native["source"] == MODELS[row["model_id"]][1]
        assert set(native["values"]) == set(included_features(row["dataset_id"], row["track"]))
        additive = sum(native["values"].values()) + native["bias"]
        assert abs(additive - facts["raw_scores"][instance["instance_id"]]) < 1e-9
        canonical = instance["canonical_explanation"]
        for name in (
            "principal_reasons",
            "referenced_features",
            "counterfactual_changes",
            "explanation_family",
        ):
            assert name in canonical, name
        assert canonical["metadata"]["synthetic_non_finite"] is None
        assert canonical["metadata"]["synthetic_numpy_int"] == len(canonical["referenced_features"])
    for row in payload["summary"]:
        indexes = [instance["eval_row_index"] for instance in row["instance_results"]]
        assert indexes == facts["positions"][row["dataset_id"]], (common.group_key(row), indexes)
    json.dumps(payload, allow_nan=False)
    checks.ok(
        "persisted instance fields: eval_row_index, canonical_explanation, "
        "native_contributions (additive to the raw score), strict JSON"
    )
    checks.ok("shared sampling: every (track, model, explainer) explains the same positions in order")


def check_rescore(
    native_payload: dict[str, Any],
    native_dir: Path,
    offline: Path,
    checks: Checklist,
) -> tuple[Path, Path]:
    rescored, report = rescore.rescore_payload(
        native_payload,
        "css_no_dsv",
        mean_fn=runner._mean,
        overview_builder=runner._build_overview,
        source_results=str(native_dir / "results.json"),
    )
    assert report["consistency_mismatches"] == [], report["consistency_mismatches"][:3]
    for old_row, new_row in zip(native_payload["summary"], rescored["summary"], strict=True):
        values = []
        for old, new in zip(old_row["instance_results"], new_row["instance_results"], strict=True):
            css = new["scores"]["contestability_support_score"]
            expected = (
                0.30 * css["verifiability"]
                + 0.30 * css["correctability"]
                + 0.15 * css["reason_specificity"]
            ) / 0.75
            assert abs(css["score"] - expected) < 1e-12
            assert css["decision_sensitivity_visibility"] == (
                old["scores"]["contestability_support_score"]["decision_sensitivity_visibility"]
            )
            assert new["canonical_explanation"] == old["canonical_explanation"]
            values.append(css["score"])
        assert new_row["mean_css"] == runner._mean(values)
        assert new_row["mean_stability"] == old_row["mean_stability"]
    checks.ok("rescore css_no_dsv = (0.30V + 0.30C + 0.15R)/0.75 per instance; means via runner._mean")

    _rows, leaders = rescore.dsv_weight_sweep(native_payload["summary"], [0.0, 0.25, 1.0])
    assert leaders and all(row["published_leader"] not in common.NULL_BASELINES for row in leaders)

    variant_dir = offline / "rescored" / "credit" / "css_no_dsv"
    code = call_main(
        rescore.main,
        ["rescore.py", "--results", native_dir, "--variant", "css_no_dsv", "--out-dir", variant_dir],
    )
    assert code == 0, code
    assert_files(
        [
            variant_dir / "results.json",
            variant_dir / "rescore_by_explainer.csv",
            variant_dir / "rescore_summary.md",
        ],
        "rescore variant",
    )
    written = common.load_results(variant_dir / "results.json")
    assert written["experiment_name"].endswith("__rescored_css_no_dsv")
    assert written["rescore"]["variant"] == "css_no_dsv"
    assert "cslaw" in written, "rescoring must keep the cslaw block"

    sweep_dir = offline / "dsv_sweep" / "credit"
    code = call_main(
        rescore.main,
        ["rescore.py", "--results", native_dir, "--sweep-dsv", "0,0.1,0.25,0.4", "--out-dir", sweep_dir],
    )
    assert code == 0, code
    sweep_csv = sweep_dir / "css_dsv_sweep.csv"
    assert_files(
        [sweep_csv, sweep_dir / "css_dsv_sweep_leaders.csv", sweep_dir / "css_dsv_sweep.md"],
        "sweep",
    )
    observed = rescore.explainer_level_means(native_payload["summary"], ["mean_css"])
    pooled_rows = 0
    for row in read_csv(sweep_csv):
        if row["subset"] == "pooled" and float(row["weight"]) == 0.25:
            pooled_rows += 1
            assert abs(float(row["mean_css"]) - observed[row["explainer_id"]]["mean_css"]) < 1e-9
    assert pooled_rows == len(common.REAL_EXPLAINERS), pooled_rows
    checks.ok(
        "rescore main(): variant output, S-weight sweep (w=0.25 reproduces mean_css), "
        "null baselines never leaders"
    )
    return offline / "rescored", sweep_csv


def check_compare(
    payload: dict[str, Any],
    old_payload: dict[str, Any],
    shared_dir: Path,
    root: Path,
    checks: Checklist,
) -> None:
    result = compare_bundles.compare_payloads(old_payload, payload)
    assert not result.failures, [finding.message for finding in result.failures]
    assert result.groups and all(group.status == "identical" for group in result.groups)
    checks.ok("compare_bundles: scores-only bundle == instrumented bundle (cslaw fields ignored)")

    copy_dir = root / "outputs" / "copy_bundle"
    shutil.copytree(shared_dir, copy_dir)
    report = root / "compare" / "identical.md"
    code = call_main(
        compare_bundles.main,
        ["compare_bundles.py", shared_dir, copy_dir, "--out", report],
    )
    assert code == 0, code
    assert "Verdict: **PASS**" in report.read_text(encoding="utf-8")

    perturbed = copy.deepcopy(payload)
    target = next(
        row
        for row in perturbed["summary"]
        if (row["model_id"], row["explainer_id"]) == ("logistic_regression", "lime")
    )
    target["instance_results"][0]["scores"]["stability_score"]["score"] += 0.25
    perturbed_dir = write_bundle(root / "outputs" / "perturbed_bundle", perturbed)
    code = call_main(
        compare_bundles.main,
        ["compare_bundles.py", shared_dir, perturbed_dir, "--out", root / "compare" / "perturbed.md"],
    )
    assert code == 1, code
    checks.ok("compare_bundles main(): copy -> exit 0 (PASS); perturbed LR/LIME score -> exit 1")


def check_paper_tables(
    native_dir: Path,
    null_dir: Path,
    native_payload: dict[str, Any],
    outputs: Path,
    rescored_root: Path,
    sweep_csv: Path,
    root: Path,
    checks: Checklist,
) -> None:
    # A cslaw_* re-run bundle next to the inputs, as on the cluster.
    rerun_dir = outputs / "cslaw_credit_rerun_synthetic"
    means = rescore.explainer_level_means(
        native_payload["summary"],
        [f"mean_{metric}" for metric in paper_tables.MAIN_METRICS],
    )
    rescore.write_rows_csv(
        rerun_dir / "by_explainer.csv",
        [{"explainer_id": explainer_id, **values} for explainer_id, values in means.items()],
    )
    rescore.write_rows_csv(
        rerun_dir / "summary.csv",
        [{"dataset_id": dataset_id} for dataset_id in DATASETS],
    )

    tables = root / "paper_tables"
    code = call_main(
        paper_tables.main,
        [
            "paper_tables.py",
            "--bundle", f"credit={native_dir}",
            "--bundle", f"nullbaselines={null_dir}",
            "--robustness-root", outputs,
            "--rescored-root", rescored_root,
            "--sweep-csv", sweep_csv,
            "--out-dir", tables,
            "--bootstrap", "20",
            "--seed", "1",
        ],
    )
    assert code == 0, code
    expected = (
        "T1_by_construction.csv",
        "T1_by_construction.md",
        "T2_css_decomposition.csv",
        "T3_pairwise_recount.csv",
        "T3_pairwise_pairs.csv",
        "T4_per_domain.csv",
        "T5_correlations.csv",
        "T6_robustness.csv",
        "T7_null_baselines.csv",
        "tables.md",
    )
    assert_files([tables / name for name in expected], "paper_tables")
    ceiling = [
        row
        for row in read_csv(tables / "T1_by_construction.csv")
        if row["fact"] == "non_recourse_actionability_ceiling" and row["statistic"] == "n_above_ceiling"
    ]
    assert ceiling and all(float(row["value"]) == 0 for row in ceiling), ceiling
    null_rows = [row for row in read_csv(tables / "T2_css_decomposition.csv") if row["is_null_baseline"] == "True"]
    assert null_rows and all(row["css_rank"] == "" and row["css_no_dsv_rank"] == "" for row in null_rows)
    families = {row["check_family"] for row in read_csv(tables / "T6_robustness.csv")}
    assert {"family_neutral_rescore", "css_dsv_weight_sweep", "cslaw_rerun"} <= families, families
    t7 = read_csv(tables / "T7_null_baselines.csv")
    assert t7 and all(row["matched_to_baseline_instances"] == "True" for row in t7), t7[:2]
    assert {"NULL_BASELINE", "real_explainer"} <= {row["role"] for row in t7}
    checks.ok(
        "paper_tables main(): T1-T7 written; null baselines unranked; T6 sees the rescore, "
        "the sweep and cslaw_* re-runs; T7 matched by instance_id"
    )


def check_regb(payload: dict[str, Any], shared_dir: Path, root: Path, checks: Checklist) -> None:
    rows, coverage = regb_reference.collect_regb_instances(payload, k=K, bundle="shared")
    assert rows and coverage["used"] == coverage["instances"], dict(coverage)
    lookup = {(row["explainer_id"], row["instance_id"]): row for row in rows}
    groups = payload["cslaw"]["groups"]
    for summary_row, instance in common.iter_instances(payload):
        dataset_id, track, model_id, explainer_id = common.group_key(summary_row)
        row = lookup[(explainer_id, instance["instance_id"])]
        reference = groups[common.cslaw_group_key(dataset_id, track, model_id)]["regb_reference"]
        for column, population in (("m1_features", "near_cutoff"), ("m2_features", "all_applicants")):
            expected = common.top_adverse_features(
                instance["native_contributions"]["values"],
                k=K,
                reference=reference[population]["means"],
            )
            assert row[column] == "|".join(expected), (column, row[column], expected)
        flags = row["by_construction_flags"].split(";")
        if explainer_id == "shap" and model_id == "logistic_regression":
            assert "shap_lr_linear_decomposition" in flags, flags
        if explainer_id == "shap" and model_id == "xgboost":
            assert "shap_xgb_treeshap_equals_native" in flags, flags
        if explainer_id in common.RECOURSE_EXPLAINERS:
            assert "cf_features_ranked_by_native" in flags, flags
        assert row["is_null_baseline"] == common.is_null_baseline(explainer_id)
    outputs = root / "analysis" / "regb_reference"
    regb_reference.main(
        ["--bundle", str(shared_dir), "--out-dir", str(outputs), "--k", str(K), "--n-boot", str(N_BOOT)]
    )
    assert_files(
        [
            outputs / name
            for name in (
                "regb_reference_instances.csv",
                "regb_reference_by_group.csv",
                "regb_reference_pooled_by_model.csv",
                "regb_reference_pooled_by_model_class.csv",
                "regb_m1_vs_m2.csv",
                "regb_reference_populations.csv",
                "regb_reference_summary.md",
            )
        ],
        "regb_reference",
    )
    assert len(read_csv(outputs / "regb_reference_populations.csv")) == len(groups)
    checks.ok(
        "regb_reference: M1/M2 reasons match an independent recomputation; "
        "by-construction flags; main() outputs"
    )


def check_multiplicity(payload: dict[str, Any], shared_dir: Path, root: Path, checks: Checklist) -> None:
    index = multiplicity.collect_reason_index(payload, reason_set="principal", k=K)
    cross_model, dropped = multiplicity.cross_model_rows(index)
    assert cross_model and not dropped, dict(dropped)
    assert all(row["n_models"] == len(MODELS) for row in cross_model)
    methods = {row["method"]: row["method_kind"] for row in cross_model}
    assert methods.get("regb_m1") == "regb_reference", methods
    assert methods.get("regb_m2") == "regb_reference", methods
    assert methods.get("shap") == "explainer", methods
    assert methods.get("random_reasons") == "null_baseline", methods
    applicants, pairs = multiplicity.cross_explainer_rows(index)
    assert applicants and all(
        row["n_real_explainers"] == len(common.REAL_EXPLAINERS) for row in applicants
    )
    assert any(row["pair_kind"] == "involves_null_baseline" for row in pairs)
    outputs = root / "analysis" / "multiplicity"
    multiplicity.main(
        ["--bundle", str(shared_dir), "--out-dir", str(outputs), "--k", str(K), "--n-boot", str(N_BOOT)]
    )
    assert_files(
        [
            outputs / name
            for name in (
                "multiplicity_cross_model_instances.csv",
                "multiplicity_cross_model_summary.csv",
                "multiplicity_cross_explainer_instances.csv",
                "multiplicity_cross_explainer_summary.csv",
                "multiplicity_explainer_pairs_instances.csv",
                "multiplicity_explainer_pairs_summary.csv",
                "multiplicity_sampling_status.csv",
                "multiplicity_model_performance.csv",
                "multiplicity_summary.md",
            )
        ],
        "multiplicity",
    )
    status = read_csv(outputs / "multiplicity_sampling_status.csv")
    assert status and all(row["shared_fallback"] == "False" for row in status), status[:1]
    checks.ok(
        "multiplicity: every applicant explained under all 3 models; Reg B and null-baseline "
        "methods labelled; main() outputs"
    )


def check_proxy(
    payload: dict[str, Any],
    shared_dir: Path,
    meta_root: Path,
    root: Path,
    checks: Checklist,
) -> None:
    registry = proxy_substitution.load_registry(meta_root)
    taxonomy = common.load_taxonomy(meta_root)
    for dataset_id in DATASETS:
        assert (
            proxy_substitution.excluded_in_primary(registry, taxonomy, dataset_id)
            == PRIMARY_EXCLUDED[dataset_id]
        )
    pairs, kinds, _coverage = proxy_substitution.collect_track_pairs(payload, k=K)
    rows, _transitions, match_coverage = proxy_substitution.substitution_rows(
        pairs, kinds, registry, taxonomy
    )
    assert rows and match_coverage["not_in_both_tracks"] == 0, dict(match_coverage)
    for row in rows:
        excluded = {name for name in row["excluded_in_primary"].split("|") if name}
        primary = {name for name in row["primary_features"].split("|") if name}
        assert not excluded & primary, row
    assert proxy_substitution.sampling_scopes(payload) == {"models_and_tracks"}
    outputs = root / "analysis" / "proxy_substitution"
    proxy_substitution.main(
        [
            "--bundle", str(shared_dir),
            "--out-dir", str(outputs),
            "--repo-root", str(meta_root),
            "--k", str(K),
            "--n-boot", str(N_BOOT),
        ]
    )
    assert_files(
        [
            outputs / name
            for name in (
                "proxy_substitution_instances.csv",
                "proxy_substitution_summary.csv",
                "proxy_substitution_transitions.csv",
                "proxy_substitution_exclusions.csv",
                "proxy_substitution_summary.md",
            )
        ],
        "proxy_substitution",
    )
    checks.ok(
        "proxy_substitution: all applicants paired across tracks; excluded features never "
        "cited in primary; main() outputs"
    )


def check_common_faithfulness(
    native_payload: dict[str, Any],
    null_payload: dict[str, Any],
    native_dir: Path,
    null_dir: Path,
    root: Path,
    checks: Checklist,
) -> None:
    index, _stats = common_faithfulness.build_native_index(
        [("credit", native_payload), ("nullbaselines", null_payload)]
    )
    null_rows, _coverage = common_faithfulness.collect_faithfulness_instances(
        "nullbaselines", null_payload, index, k=K
    )
    assert null_rows and all(row["native_origin"] == "joined:credit" for row in null_rows)
    assert all(row["is_null_baseline"] for row in null_rows)
    native_rows, _coverage = common_faithfulness.collect_faithfulness_instances(
        "credit", native_payload, index, k=K
    )
    assert native_rows and all(row["native_origin"] == "own" for row in native_rows)
    shap_rows = [row for row in native_rows if row["explainer_id"] == "shap"]
    assert shap_rows and all(row["direction_agreement"] == 1.0 for row in shap_rows)
    # v2 DiCE/Wachter copy signed_direction from the native sign: not measured (None).
    cf_canonical = {
        "explanation_family": "counterfactual",
        "referenced_features": [
            {"feature_name": "a", "signed_direction": "increases_denial", "rank": 1},
            {"feature_name": "b", "signed_direction": "decreases_denial", "rank": 2},
        ],
        "principal_reasons": [],
        "counterfactual_changes": [{"feature_name": "a"}],
        "metadata": {"cf_reason_source": "counterfactual_changes"},
    }
    cf_values = {"a": 0.4, "b": -0.2, "c": 0.1}
    assert common_faithfulness.faithfulness_metrics(cf_canonical, cf_values, K)[
        "direction_agreement"
    ] is None
    cf_canonical["metadata"] = {}
    assert common_faithfulness.faithfulness_metrics(cf_canonical, cf_values, K)[
        "direction_agreement"
    ] == 1.0
    _row, probe = next(common.iter_instances(null_payload))
    shifted = copy.deepcopy(probe)
    shifted["prediction_score"] = float(shifted["prediction_score"]) + 0.01
    values, origin, _source = common_faithfulness.resolve_native(shifted, index)
    assert values is None and origin == "join_rejected_score_mismatch", origin
    outputs = root / "analysis" / "common_faithfulness"
    common_faithfulness.main(
        [
            "--bundle", str(native_dir),
            "--bundle", str(null_dir),
            "--label", "credit",
            "--label", "nullbaselines",
            "--out-dir", str(outputs),
            "--k", str(K),
            "--n-boot", str(N_BOOT),
        ]
    )
    assert_files(
        [
            outputs / name
            for name in (
                "common_faithfulness_instances.csv",
                "common_faithfulness_summary.csv",
                "common_faithfulness_summary.md",
            )
        ],
        "common_faithfulness",
    )
    checks.ok(
        "common_faithfulness: null baselines join native values by instance_id (score-checked); "
        "SHAP direction agreement 1.0; v2 counterfactual direction not measured; main() outputs"
    )


def check_old_bundle(
    old_payload: dict[str, Any],
    old_dir: Path,
    meta_root: Path,
    root: Path,
    checks: Checklist,
) -> None:
    out = root / "analysis_old"
    expect_exit(
        "regb_reference",
        regb_reference.run,
        old_dir,
        out / "regb",
        contains="persist_canonical_explanations",
    )
    expect_exit(
        "multiplicity",
        multiplicity.run,
        old_dir,
        out / "multiplicity",
        contains="shared_sampling",
    )
    expect_exit(
        "proxy_substitution",
        proxy_substitution.run,
        old_dir,
        out / "proxy",
        repo_root=meta_root,
        contains="persist_canonical_explanations",
    )
    expect_exit(
        "common_faithfulness",
        common_faithfulness.run,
        [old_dir],
        out / "faithfulness",
        contains="persist_canonical_explanations",
    )
    rescored, _report = rescore.rescore_payload(
        old_payload,
        "all_neutral",
        mean_fn=runner._mean,
        overview_builder=runner._build_overview,
        source_results=str(old_dir / "results.json"),
    )
    assert rescored["rescore"]["instances_rescored"] == sum(
        len(row["instance_results"]) for row in old_payload["summary"]
    )
    checks.ok("older bundles: persisted-field analyses exit with a clear message; rescore still works")


# ---------------------------------------------------------------------------
# v2 corrections: contract and pure-function checks (no files)
# ---------------------------------------------------------------------------


def _accepts(func: Callable[..., Any], parameter: str) -> bool:
    try:
        return parameter in inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False


def skip(checks: Checklist, label: str, reason: str) -> None:
    """A contract check that cannot run because a signature differs (printed, not failed)."""
    print(f"  SKIP {label}: {reason}")
    checks.skipped.append(f"{label}: {reason}")


def check_corrections(checks: Checklist) -> None:
    v1, v2 = corrections_module.V1, corrections_module.V2
    parse = corrections_module.corrections_from_payload
    assert common.CORRECTIONS_V1 == v1.to_dict(), (common.CORRECTIONS_V1, v1.to_dict())
    assert common.CORRECTIONS_V2 == v2.to_dict(), (common.CORRECTIONS_V2, v2.to_dict())
    assert v1.is_v1 and not v2.is_v1
    assert parse(None) == v1 and parse({}) == v1 and parse({"cslaw": {}}) == v1
    assert parse({"cslaw": {"persist_canonical_explanations": True}}) == v1
    assert parse({"cslaw": {"corrections": "v1"}}) == v1
    assert parse({"cslaw": {"corrections": "v2"}}) == v2
    assert parse({"cslaw": {"corrections": {"preset": "v2"}}}) == v2
    custom = parse(
        {"cslaw": {"corrections": {"preset": "v2", "sampling_selection": "boundary_nearest"}}}
    )
    assert custom == replace(v2, sampling_selection="boundary_nearest"), custom
    weight_only = parse({"cslaw": {"corrections": {"faithfulness_quality_weight": 0.0}}})
    assert weight_only == replace(v1, faithfulness_quality_weight=0.0), weight_only
    for bad in (
        {"corrections": "v3"},
        {"corrections": {"preset": "v9"}},
        {"corrections": {"preset": "v2", "no_such_option": True}},
        {"corrections": {"sampling_selection": "nearest"}},
        {"corrections": {"faithfulness_quality_weight": 1.5}},
        {"corrections": 5},
    ):
        try:
            parse({"cslaw": bad})
        except ValueError:
            continue
        raise AssertionError(f"invalid cslaw.corrections accepted: {bad}")
    legacy = parse({"cslaw": {"use_native_contributions_for_ranking": True}})
    assert legacy == replace(v1, native_contributions=True), legacy
    assert parse({"cslaw": {"corrections": "v2", "use_native_contributions_for_ranking": False}}) == v2
    config = SimpleNamespace(payload={"cslaw": {"corrections": "v2"}})
    assert corrections_module.corrections_from_config(config) == v2
    assert corrections_module.corrections_from_config(SimpleNamespace()) == v1
    assert corrections_module.model_corrections(SimpleNamespace()) == v1
    assert corrections_module.model_corrections(SimpleNamespace(corrections=v2)) == v2
    checks.ok(
        "legalbench_xai.corrections: presets, preset overrides, unknown/invalid options raise, "
        "legacy native flag; common.py mirrors V1/V2"
    )

    def bundle(block: dict[str, Any] | None) -> dict[str, Any]:
        return {} if block is None else {"cslaw": block}

    assert common.bundle_corrections(bundle(None)) is None
    assert common.corrections_version(bundle(None)) == "v1"
    assert common.corrections_version(bundle({"config": {}})) == "v1"
    assert common.corrections_version(bundle({"corrections": v2.to_dict()})) == "v2"
    assert common.corrections_version(bundle({"corrections": "v2"})) == "v2"
    assert common.corrections_version(bundle({"corrections": {"preset": "v2"}})) == "v2"
    assert common.corrections_version(
        bundle({"corrections": custom.to_dict(), "corrections_version": "custom"})
    ) == "custom"
    assert common.correction_flag(bundle(None), "faithfulness_quality_weight") == 0.20
    assert common.correction_flag(bundle({"corrections": "v2"}), "faithfulness_quality_weight") == 0.0
    native_only = bundle(
        {"config": {"use_native_contributions_for_ranking": False}, "corrections": v2.to_dict()}
    )
    assert regb_reference.native_ranking_enabled(native_only)
    assert regb_reference.cf_reasons_from_changes(native_only)
    assert regb_reference.grouping_fixed(native_only)
    assert not regb_reference.grouping_fixed(bundle({"config": {}}))
    checks.ok("common.py reads the recorded corrections (dict, preset name, config form) and versions")


def check_pure_fixes(checks: Checklist) -> None:
    rng = seeded("faithfulness-identity")
    for _ in range(200):
        coverage, direction, quality = rng.random(), rng.random(), rng.random()
        published = 0.45 * coverage + 0.35 * direction + 0.20 * quality
        assert common.faithfulness_with_quality_weight(coverage, direction, quality, 0.20) == published
        no_quality = rescore.renormalized_score(
            {"reference_feature_coverage": coverage, "directional_agreement": direction},
            rescore.FID_NO_QUALITY,
        )
        renormalized = common.faithfulness_with_quality_weight(coverage, direction, quality, 0.0)
        assert no_quality is not None and abs(renormalized - no_quality) < 1e-12
    assert 0.80 + 0.20 == 1.0
    checks.ok(
        "faithfulness (0.45c + 0.35d + w q)/(0.80 + w): w = 0.20 is exactly the published "
        "score, w = 0 equals fid_no_quality"
    )

    if _accepts(ExplanationScorer, "corrections"):
        rows = make_rows("heloc")
        model = FakeNativeModel("heloc", "audit", "logistic_regression")
        row, prediction = model.predict(rows[:1], "heloc-audit-logistic_regression")[0]
        native = model.native_feature_contributions(row)
        for explainer_id in ("dice", "shap"):
            canonical = build_canonical(
                dataset_id="heloc",
                track="audit",
                model_id="logistic_regression",
                explainer_id=explainer_id,
                row=row,
                prediction=prediction,
                ranked=explanation_weights(explainer_id, prediction.instance_id, native, {}),
            )
            published = ExplanationScorer().score(canonical)
            assert published == ExplanationScorer(corrections=corrections_module.V1).score(canonical)
            corrected = ExplanationScorer(corrections=corrections_module.V2).score(canonical)
            fid_v1, fid_v2 = published["faithfulness_score"], corrected["faithfulness_score"]
            assert "faithfulness_quality_weight" not in fid_v1, fid_v1
            assert fid_v2["faithfulness_quality_weight"] == 0.0, fid_v2
            assert abs(
                fid_v1["score"]
                - (
                    0.45 * fid_v1["reference_feature_coverage"]
                    + 0.35 * fid_v1["directional_agreement"]
                    + 0.20 * fid_v1["explanation_quality_signal"]
                )
            ) < 1e-12
            assert abs(
                fid_v2["score"]
                - common.faithfulness_with_quality_weight(
                    fid_v2["reference_feature_coverage"],
                    fid_v2["directional_agreement"],
                    fid_v2["explanation_quality_signal"],
                    0.0,
                )
            ) < 1e-12
            css_v1 = published["contestability_support_score"]
            css_v2 = corrected["contestability_support_score"]
            assert not set(common.DSV_SUBCOMPONENT_WEIGHTS) & set(css_v1), css_v1
            assert abs(
                css_v2["decision_sensitivity_visibility"]
                - sum(weight * css_v2[name] for name, weight in common.DSV_SUBCOMPONENT_WEIGHTS.items())
            ) < 1e-12, css_v2
            assert css_v2["score"] == css_v1["score"]
            weight_020 = ExplanationScorer(
                corrections=replace(
                    corrections_module.V2,
                    faithfulness_quality_weight=0.20,
                    common_faithfulness_reference=False,
                )
            ).score(canonical)["faithfulness_score"]
            assert weight_020["score"] == fid_v1["score"], (weight_020, fid_v1)
            # Under the common reference, counterfactual (and rule) explanations are
            # direction-neutral: their signed_direction is copied from the native sign,
            # so a match would be 1.0 by construction. Attribution explainers keep it.
            if explainer_id == "dice" and canonical.referenced_features:
                assert abs(fid_v1["directional_agreement"] - 1.0) < 1e-12, fid_v1
                assert abs(fid_v2["directional_agreement"] - 0.5) < 1e-12, fid_v2
            else:
                assert fid_v2["directional_agreement"] == fid_v1["directional_agreement"], (
                    fid_v1,
                    fid_v2,
                )
        checks.ok(
            "scorer: V1 == default; V2 renormalized faithfulness (w = 0) and persisted "
            "decision-sensitivity sub-components that reproduce S; w = 0.20 keeps the v1 score; "
            "common reference scores counterfactual directions as neutral (0.5)"
        )
    else:
        skip(checks, "scorer corrections", "ExplanationScorer has no 'corrections' argument")

    if _accepts(normalization_rules.grouped_reason_label, "fix_prefix_order"):
        for name in ("PAY_0", "PAY_6", "PAY_AMT1", "PAY_AMT6", "BILL_AMT2", "LIMIT_BAL", "AGE"):
            assert normalization_rules.grouped_reason_label(name) == common.temporal_group_label(name)
            assert normalization_rules.grouped_reason_label(
                name, fix_prefix_order=True
            ) == common.temporal_group_label(name, fixed=True), name
        assert normalization_rules.grouped_reason_label("PAY_AMT1") == "recent late payment history"
        assert (
            normalization_rules.grouped_reason_label("PAY_AMT1", fix_prefix_order=True)
            == "recent payment amounts"
        )
        checks.ok(
            "reason grouping: v1 keeps PAY_AMT* under late payment history; the fix groups it "
            "as payment amounts; common.py mirrors rules.py in both modes"
        )
    else:
        skip(checks, "reason grouping fix", "rules.grouped_reason_label has no 'fix_prefix_order'")

    coerce = getattr(dataset_io, "_coerce_scalar", None)
    if coerce is not None and _accepts(coerce, "parse_exponent_notation"):
        assert coerce("8e-05") == "8e-05", coerce("8e-05")
        assert coerce("8e-05", parse_exponent_notation=True) == 8e-05
        assert coerce("-1E5", parse_exponent_notation=True) == -1e5
        assert coerce("3.5e-2", parse_exponent_notation=True) == 0.035
        for text, expected in (("12", 12), ("3.5", 3.5), ("A12", "A12"), ("", None), ("NA", None)):
            assert coerce(text) == expected and coerce(text, parse_exponent_notation=True) == expected
        assert coerce("inf", parse_exponent_notation=True) == "inf"
        checks.ok("numeric parsing: exponent notation parsed only with the fix; other cells as in v1")
    else:
        skip(checks, "numeric parsing fix", "datasets.io._coerce_scalar has no 'parse_exponent_notation'")

    builder = getattr(generators, "counterfactual_change_reasons", None)
    if builder is not None and all(
        _accepts(builder, name)
        for name in ("changed_feature_sets", "flip_flags", "native_contributions", "native_source", "row")
    ):
        importances, metadata = builder(
            changed_feature_sets=[["a", "b"], ["b", "c"], ["b"]],
            flip_flags=[True, True, False],
            native_contributions={"a": 0.5, "b": -0.2, "c": 0.9, "d": 2.0},
            native_source="lr_coef_x_standardized",
            row={"a": 1, "b": 2, "c": 3, "d": 4},
        )
        assert [item["feature_name"] for item in importances] == ["b", "c", "a"], importances
        assert importances[0]["weight"] > 0 and importances[0]["weight"] < 1e-3, importances
        assert importances[1]["weight"] == 0.9 and importances[2]["weight"] == 0.5, importances
        assert metadata["cf_reason_source"] == "counterfactual_changes", metadata
        empty, empty_metadata = builder(
            changed_feature_sets=[[], []],
            flip_flags=[False, False],
            native_contributions={"a": 1.0},
            native_source="lr_coef_x_standardized",
            row={"a": 1},
        )
        assert empty == [] and empty_metadata["cf_reason_source"] == "no_valid_change"
        assert [item["signed_direction"] for item in importances] == [
            "decreases_denial",
            "increases_denial",
            "increases_denial",
        ], importances
        assert abs(metadata["cf_reason_details"]["nondenial_reason_share"] - 1 / 3) < 1e-12
        unflipped, unflipped_metadata = builder(
            changed_feature_sets=[["a"], ["b"]],
            flip_flags=[False, False],
            native_contributions={"a": 1.0, "b": 0.5},
            native_source="lr_coef_x_standardized",
            row={"a": 1, "b": 2},
        )
        assert unflipped == [] and unflipped_metadata["cf_reason_source"] == "no_valid_change"
        assert unflipped_metadata["cf_reason_details"]["non_flipping_change_counts"] == {"a": 1, "b": 1}
        many, _ = builder(
            changed_feature_sets=[[f"f{index}" for index in range(9)]],
            flip_flags=[True],
            native_contributions={f"f{index}": float(index) for index in range(9)},
            native_source="lr_coef_x_standardized",
            row={},
        )
        assert [item["feature_name"] for item in many] == ["f8", "f7", "f6", "f5", "f4", "f3"], many
        checks.ok(
            "DiCE/Wachter reasons from changes: flipping counterfactuals only, change count, "
            "then native contribution; positive weights, native-sign direction; at most 6; "
            "no_valid_change when no flipping counterfactual changes anything"
        )
    else:
        skip(checks, "counterfactual reasons from changes", "generators.counterfactual_change_reasons differs")

    if _accepts(runner._sample_evaluation_rows, "selection") and _accepts(
        runner._sample_evaluation_rows, "seed_key"
    ):
        candidates = [
            ({"position": index}, SimpleNamespace(decision_margin=0.004 + 0.008 * index))
            for index in range(60)
        ]
        positions = lambda selected: [item[0]["position"] for item in selected]  # noqa: E731
        default, default_summary = runner._sample_evaluation_rows(candidates, SAMPLING_CFG)
        explicit, explicit_summary = runner._sample_evaluation_rows(
            candidates, SAMPLING_CFG, selection="boundary_nearest"
        )
        assert positions(default) == positions(explicit) and default_summary == explicit_summary
        assert "selection" not in default_summary and "seed" not in default_summary
        drawn, drawn_summary = runner._sample_evaluation_rows(
            candidates, SAMPLING_CFG, selection="stratified_random", seed_key="heloc|audit|ebm", seed=42
        )
        again, _ = runner._sample_evaluation_rows(
            list(reversed(candidates)),
            SAMPLING_CFG,
            selection="stratified_random",
            seed_key="heloc|audit|ebm",
            seed=42,
        )
        assert positions(drawn) == positions(again), "stratified_random must not depend on input order"
        assert drawn_summary["selection"] == "stratified_random" and drawn_summary["seed"] == 42
        assert len(drawn) == len(default) == MAX_INSTANCES
        strata = lambda selected: sorted(  # noqa: E731
            runner._decision_margin_stratum(item[1].decision_margin) for item in selected
        )
        assert strata(drawn) == strata(default), (strata(drawn), strata(default))
        others = [
            positions(
                runner._sample_evaluation_rows(
                    candidates, SAMPLING_CFG, selection="stratified_random", seed_key=key, seed=42
                )[0]
            )
            for key in ("heloc|primary|ebm", "heloc|audit|xgboost", "heloc|shared|models")
        ]
        assert any(item != positions(default) for item in [positions(drawn), *others])
        checks.ok(
            "sampling: boundary_nearest is the unchanged default; stratified_random keeps the "
            "strata allocation, is seeded by its key and independent of the input order"
        )
    else:
        skip(checks, "stratified_random sampling", "runner._sample_evaluation_rows differs")


def check_v2_reason_units(checks: Checklist) -> None:
    canonical = {
        "principal_reasons": [
            {
                "rank": 1,
                "supporting_features": ["PAY_0", "PAY_2"],
                "signed_direction": "increases_denial",
                "grouped_reason_type": "compressed_temporal",
            },
            {
                "rank": 2,
                "supporting_features": ["PAY_AMT1", "PAY_AMT2"],
                "signed_direction": "increases_denial",
                "grouped_reason_type": "compressed_temporal",
            },
            {"rank": 3, "supporting_features": ["LIMIT_BAL"], "signed_direction": "increases_denial"},
        ]
    }
    late = "recent late payment history"
    amounts = "recent payment amounts"
    assert regb_reference.reason_unit_labels(canonical, "principal") == [late, "LIMIT_BAL"]
    fixed = regb_reference.reason_unit_labels(canonical, "principal", fixed=True)
    assert fixed == [late, amounts, "LIMIT_BAL"], fixed
    assert multiplicity.explainer_reason_list(
        canonical, "principal", K, "grouped", fixed_grouping=True
    ) == fixed
    values = {"PAY_0": 2.0, "PAY_AMT1": 1.5, "PAY_AMT2": 1.2, "LIMIT_BAL": 1.0, "AGE": 0.5}
    means = {name: 0.0 for name in values}
    assert regb_reference.regb_unit_ranking(values, means, 3) == [late, "LIMIT_BAL", "AGE"]
    assert regb_reference.regb_unit_ranking(values, means, 3, fixed=True) == [late, amounts, "LIMIT_BAL"]
    assert regb_reference.regb_reason_units(values, means, 2, fixed=True) == [
        ["PAY_0"],
        ["PAY_AMT1", "PAY_AMT2"],
    ]

    flags, caveats = regb_reference.by_construction_flags(
        "dice", "ebm", native_ranking=True, cf_reasons_from_changes=True
    )
    assert "cf_features_ranked_by_native" not in flags and "cf_reason_order_from_native" in caveats
    assert "ranking_uses_heuristic_proxy" not in caveats
    flags, caveats = regb_reference.by_construction_flags("wachter", "ebm", native_ranking=False)
    assert not flags and "ranking_uses_heuristic_proxy" in caveats, (flags, caveats)
    flags, _ = regb_reference.by_construction_flags("dice", "logistic_regression", native_ranking=False)
    assert "cf_features_ranked_by_native" in flags
    flags, _ = regb_reference.by_construction_flags(
        "anchors", "ebm", native_ranking=True, cf_reasons_from_changes=True
    )
    assert "anchors_candidates_ranked_by_native" in flags
    checks.ok(
        "v2 reason units: PAY_AMT* is its own reason under the grouping fix (explainers, Reg B, "
        "multiplicity); DiCE/Wachter lose the borrowed-ranking flag when reasons come from changes"
    )


# ---------------------------------------------------------------------------
# v2-shaped bundles through the analysis scripts
# ---------------------------------------------------------------------------

V2_METADATA = {
    "dataset_registry": "benchmark/metadata/dataset_registry_v2.json",
    "feature_taxonomy": "benchmark/metadata/feature_taxonomy_v2.json",
}


def write_v2_metadata(repo_root: Path) -> dict[str, Any]:
    """Synthetic *_v2.json metadata (same content as the v1 files) and a provenance block."""
    metadata_dir = repo_root / "benchmark" / "metadata"
    sources = {
        "dataset_registry": metadata_dir / "dataset_registry.json",
        "feature_taxonomy": metadata_dir / "feature_taxonomy.json",
    }
    metadata: dict[str, Any] = {}
    for name, relative in V2_METADATA.items():
        target = repo_root / relative
        common.write_json(target, common.load_results(sources[name]))
        metadata[name] = {"path": relative, "sha256": common.sha256_file(target)}
    return {
        "datasets": {
            dataset_id: {
                "path": f"benchmark/data/real/{dataset_id}.csv",
                "sha256": format(zlib.crc32(dataset_id.encode("utf-8")), "08x") * 8,
                "rows": N_ROWS,
            }
            for dataset_id in DATASETS
        },
        "metadata": metadata,
        "config": {"path": "benchmark/configs/synthetic_v2.json", "sha256": "0" * 64},
        "git_commit": None,
    }


def make_v2_payload(payload: dict[str, Any], provenance: dict[str, Any]) -> dict[str, Any]:
    """A v2-shaped bundle: corrections and provenance recorded, common-reference
    faithfulness with quality weight 0, persisted decision-sensitivity parts,
    cf_reason_source on DiCE/Wachter, stratified-random sampling records."""
    clone = copy.deepcopy(payload)
    clone["experiment_name"] = "legalbench_xai_cslaw_selftest_v2"
    for row in clone["summary"]:
        recourse = row["explainer_id"] in common.RECOURSE_EXPLAINERS
        instances = row["instance_results"]
        for position, instance in enumerate(instances):
            scores = instance["scores"]
            canonical = instance["canonical_explanation"]
            native = regb_reference.native_values(instance)
            fid = scores["faithfulness_score"]
            if native is not None:
                coverage = common_faithfulness.faithfulness_metrics(canonical, native, K)[
                    "native_mass_coverage"
                ]
                if coverage is not None:
                    fid["reference_feature_coverage"] = coverage
            fid["evidence_source"] = common.COMMON_REFERENCE_SOURCE
            fid["faithfulness_quality_weight"] = 0.0
            fid["score"] = common.faithfulness_with_quality_weight(
                fid["reference_feature_coverage"],
                fid["directional_agreement"],
                fid["explanation_quality_signal"],
                0.0,
            )
            rng = seeded("v2", instance["instance_id"], row["explainer_id"])
            parts = {
                "dsv_directionality": 1.0 if rng.random() < 0.85 else 0.0,
                "dsv_threshold_visibility": 1.0 if recourse else rng.choice([0.5, 0.8]),
                "dsv_flip_validity": rng.choice([0.0, 1.0]) if recourse else 0.0,
            }
            css = scores["contestability_support_score"]
            css.update(parts)
            css["decision_sensitivity_visibility"] = sum(
                common.DSV_SUBCOMPONENT_WEIGHTS[name] * value for name, value in parts.items()
            )
            css["score"] = _weighted(css, common.CSS_WEIGHTS)
            if recourse:
                canonical["metadata"]["cf_reason_source"] = (
                    "no_valid_change" if position == 0 else "counterfactual_changes"
                )
        row["mean_faithfulness"] = runner._mean(
            [item["scores"]["faithfulness_score"]["score"] for item in instances]
        )
        row["mean_css"] = runner._mean(
            [item["scores"]["contestability_support_score"]["score"] for item in instances]
        )
        row["mean_css_decision_sensitivity_visibility"] = runner._mean(
            [
                item["scores"]["contestability_support_score"]["decision_sensitivity_visibility"]
                for item in instances
            ]
        )
        row["faithfulness_evidence_sources"] = {common.COMMON_REFERENCE_SOURCE: len(instances)}
        row["sampling"] = {
            **row["sampling"],
            "requested_max_instances": MAX_INSTANCES,
            "selection": "stratified_random",
            "seed": 42,
        }
    clone["overview"] = runner._build_overview(clone["summary"])
    clone["cslaw"]["corrections"] = dict(common.CORRECTIONS_V2)
    clone["cslaw"]["corrections_version"] = "v2"
    clone["cslaw"]["provenance"] = copy.deepcopy(provenance)
    return runner._strict_json(clone)


def check_v2_bundles(
    v2_shared: dict[str, Any],
    v2_native: dict[str, Any],
    dirs: dict[str, Path],
    meta_root: Path,
    root: Path,
    checks: Checklist,
) -> None:
    json.dumps(v2_shared, allow_nan=False)
    assert common.corrections_version(v2_shared) == "v2"

    # rescore --variant v2_legal_functions keeps the common-reference faithfulness.
    rescored, report = rescore.rescore_payload(
        v2_native,
        rescore.V2_LEGAL_FUNCTIONS,
        mean_fn=runner._mean,
        overview_builder=runner._build_overview,
        source_results=str(dirs["v2_native"] / "results.json"),
    )
    assert report["consistency_mismatches"] == [], report["consistency_mismatches"][:3]
    assert report["fallback_instances"]["contestability_support_score"] == 0
    assert rescored["rescore"]["faithfulness_handling"] == "recorded_common_reference"
    for old_row, new_row in zip(v2_native["summary"], rescored["summary"], strict=True):
        assert new_row["mean_faithfulness"] == old_row["mean_faithfulness"]
        for instance in new_row["instance_results"]:
            css = instance["scores"]["contestability_support_score"]
            expected = (
                0.30 * css["verifiability"]
                + 0.30 * css["correctability"]
                + 0.25 * css["dsv_directionality"]
                + 0.15 * css["reason_specificity"]
            )
            assert abs(css["score"] - expected) < 1e-12
    variant_dir = root / "offline_v2" / "rescored" / "credit" / rescore.V2_LEGAL_FUNCTIONS
    code = call_main(
        rescore.main,
        [
            "rescore.py",
            "--results", dirs["v2_native"],
            "--variant", rescore.V2_LEGAL_FUNCTIONS,
            "--out-dir", variant_dir,
        ],
    )
    assert code == 0, code
    written = common.load_results(variant_dir / "results.json")
    assert written["rescore"]["variant"] == rescore.V2_LEGAL_FUNCTIONS
    assert written["rescore"]["actionability_comparison_scope"] == "recourse_family_only"
    v1_rescored, _ = rescore.rescore_payload(
        common.load_results(dirs["native"] / "results.json"),
        rescore.V2_LEGAL_FUNCTIONS,
        mean_fn=runner._mean,
        overview_builder=runner._build_overview,
        source_results="v1",
    )
    assert v1_rescored["rescore"]["faithfulness_handling"] == "fid_no_quality"
    assert v1_rescored["rescore"]["fallback_instances"]["contestability_support_score"] == (
        v1_rescored["rescore"]["instances_rescored"]
    )
    checks.ok(
        "rescore v2_legal_functions: CSS_v2 from the persisted directionality and recorded "
        "common-reference faithfulness (v2); css_no_dsv and fid_no_quality fallbacks (v1)"
    )

    tables = root / "paper_tables_v2"
    code = call_main(
        paper_tables.main,
        [
            "paper_tables.py",
            "--bundle", f"credit={dirs['v2_native']}",
            "--bundle", f"nullbaselines={dirs['v2_null']}",
            "--out-dir", tables,
            "--bootstrap", "20",
            "--seed", "1",
        ],
    )
    assert code == 0, code
    t1 = read_csv(tables / "T1_by_construction.csv")
    versions = {
        row["bundle"]: row["value"]
        for row in t1
        if row["fact"] == "corrections_status" and row["statistic"] == "corrections_version"
    }
    assert versions == {"credit": "v2", "nullbaselines": "v2"}, versions
    shares = [
        float(row["value"])
        for row in t1
        if row["fact"] == "corrections_status" and row["statistic"] == "share_common_reference_evidence"
    ]
    assert shares and all(value == 1.0 for value in shares), shares
    sources = {
        row["group_2"]
        for row in t1
        if row["fact"] == "cf_reason_source" and row["bundle"] == "credit"
    }
    assert sources == {"counterfactual_changes", "no_valid_change"}, sources
    faithfulness_pairs = [
        row for row in read_csv(tables / "T3_pairwise_pairs.csv") if row["metric"] == "faithfulness"
    ]
    faithfulness_categories = {row["category"] for row in faithfulness_pairs}
    # Common reference on both sides: open, unless one side agrees with it by
    # construction (SHAP on LR/XGB, native-ranked Anchors), as in T8.
    assert faithfulness_categories <= {"open", "decided_by_construction"}, faithfulness_categories
    unflagged = {"dice", "lime", "wachter", "leave_one_out"}
    assert all(
        row["category"] == "open"
        for row in faithfulness_pairs
        if row["explainer_a"] in unflagged and row["explainer_b"] in unflagged
    ), faithfulness_pairs
    t8 = read_csv(tables / "T8_legal_functions.csv")
    assert t8 and all(row["rank"] == "" for row in t8 if row["is_null_baseline"] == "True")
    for row in t8:
        if row["metric"] in paper_tables.T8_RECOURSE_ONLY and row["explainer_id"] not in common.RECOURSE_EXPLAINERS:
            assert row["mean"] == "" and row["rank"] == "", row
        if row["metric"] == "css_v2":
            assert row["basis"] == "directionality", row
        if row["metric"] == "faithfulness_legal":
            assert row["basis"] == "common_reference", row
    assert "## T8" in (tables / "tables.md").read_text(encoding="utf-8")
    checks.ok(
        "paper_tables on v2 bundles: T1 correction status (100% common reference, cf reason "
        "sources), T3 faithfulness comparisons open, T8 with recourse-only ALF/actionability "
        "and unranked null baselines"
    )

    rows, coverage = regb_reference.collect_regb_instances(v2_shared, k=K, bundle="v2")
    assert rows and coverage["used"] == coverage["instances"], dict(coverage)
    for row in rows:
        flags = row["by_construction_flags"].split(";")
        caveats = row["caveats"].split(";")
        assert row["corrections_version"] == "v2"
        if row["explainer_id"] in common.RECOURSE_EXPLAINERS:
            assert "cf_features_ranked_by_native" not in flags, row
            assert "cf_reason_order_from_native" in caveats, row
        if row["explainer_id"] == "anchors":
            assert "anchors_candidates_ranked_by_native" in flags, row
    regb_out = root / "analysis_v2" / "regb_reference"
    regb_reference.main(
        ["--bundle", str(dirs["v2_shared"]), "--out-dir", str(regb_out), "--k", str(K), "--n-boot", str(N_BOOT)]
    )
    assert "corrections: v2" in (regb_out / "regb_reference_summary.md").read_text(encoding="utf-8")
    checks.ok("regb_reference on a v2 bundle: no borrowed-ranking flag for DiCE/Wachter; settings line")

    index, _stats = common_faithfulness.build_native_index([("v2", v2_native)])
    faith_rows, _coverage = common_faithfulness.collect_faithfulness_instances(
        "v2", v2_native, index, k=K
    )
    checked = [row for row in faith_rows if row["pipeline_coverage_abs_diff"] is not None]
    assert checked and all(row["pipeline_coverage_abs_diff"] < 1e-9 for row in checked)
    consistency = common_faithfulness.pipeline_consistency_lines(faith_rows)
    assert consistency and not any(line.startswith("WARNING") for line in consistency), consistency
    tampered = copy.deepcopy(v2_native)
    _row, probe = next(common.iter_instances(tampered))
    probe["scores"]["faithfulness_score"]["reference_feature_coverage"] += 0.1
    tampered_rows, _ = common_faithfulness.collect_faithfulness_instances(
        "tampered", tampered, common_faithfulness.build_native_index([("tampered", tampered)])[0], k=K
    )
    assert any(
        line.startswith("WARNING")
        for line in common_faithfulness.pipeline_consistency_lines(tampered_rows)
    )
    faith_out = root / "analysis_v2" / "common_faithfulness"
    common_faithfulness.main(
        [
            "--bundle", str(dirs["v2_native"]),
            "--bundle", str(dirs["v2_null"]),
            "--label", "credit",
            "--label", "nullbaselines",
            "--out-dir", str(faith_out),
            "--k", str(K),
            "--n-boot", str(N_BOOT),
        ]
    )
    assert "Pipeline consistency" in (faith_out / "common_faithfulness_summary.md").read_text(
        encoding="utf-8"
    )
    checks.ok(
        "common_faithfulness on v2 bundles: pipeline coverage equals native mass coverage "
        "(a tampered value is reported)"
    )

    proxy_out = root / "analysis_v2" / "proxy_substitution"
    proxy_substitution.run(dirs["v2_shared"], proxy_out, repo_root=meta_root, k=K, n_boot=N_BOOT)
    exclusions = read_csv(proxy_out / "proxy_substitution_exclusions.csv")
    assert exclusions and all(row["registry_matches_run"] == "True" for row in exclusions), exclusions
    summary_md = (proxy_out / "proxy_substitution_summary.md").read_text(encoding="utf-8")
    assert "dataset_registry_v2.json" in summary_md and "(provenance)" in summary_md
    missing = copy.deepcopy(v2_shared)
    missing["cslaw"]["provenance"]["metadata"]["dataset_registry"]["path"] = (
        "benchmark/metadata/missing_registry_v2.json"
    )
    missing_dir = write_bundle(root / "outputs" / "v2_missing_registry", missing)
    expect_exit(
        "proxy_substitution (recorded registry missing)",
        proxy_substitution.run,
        missing_dir,
        root / "analysis_v2" / "proxy_missing",
        repo_root=meta_root,
        contains="does not exist",
    )
    multiplicity.run(dirs["v2_shared"], root / "analysis_v2" / "multiplicity", k=K, n_boot=N_BOOT)
    checks.ok(
        "proxy_substitution reads the registry/taxonomy recorded in provenance, checks them "
        "against the features the run dropped, refuses a missing recorded file; multiplicity runs"
    )


def shard_payload(payload: dict[str, Any], groups: set[tuple[str, str]], label: str) -> dict[str, Any]:
    """The part of a bundle a single dataset:track shard job would have written."""
    clone = copy.deepcopy(payload)
    clone["summary"] = [
        row for row in clone["summary"] if (row["dataset_id"], row["track"]) in groups
    ]
    clone["overview"] = runner._build_overview(clone["summary"])
    block = clone["cslaw"]
    block["groups"] = {
        key: value
        for key, value in block["groups"].items()
        if tuple(key.split("|")[:2]) in groups
    }
    datasets = {dataset_id for dataset_id, _track in groups}
    provenance = block["provenance"]
    provenance["datasets"] = {
        dataset_id: entry
        for dataset_id, entry in provenance["datasets"].items()
        if dataset_id in datasets
    }
    provenance["config"] = {
        "path": f"benchmark/outputs/{label}/runtime_config.json",
        "sha256": format(zlib.crc32(label.encode("utf-8")), "08x") * 8,
    }
    return clone


def shard_runtime_config(groups: set[tuple[str, str]], budget: int = MAX_INSTANCES) -> dict[str, Any]:
    tracks = {track for _dataset_id, track in groups}
    return {
        "experiment": {
            "name": "legalbench_xai_cslaw_selftest_v2",
            "random_seed": 42,
            "output_root": "benchmark/outputs",
        },
        "tracks": {"run_primary_track": "primary" in tracks, "run_audit_track": "audit" in tracks},
        "datasets": {
            "enabled": sorted({dataset_id for dataset_id, _track in groups}),
            "denied_instance_sampling": {**SAMPLING_CFG, "max_instances_per_dataset_model_pair": budget},
        },
        "dataset_sources": {
            "local_paths": {dataset_id: f"benchmark/data/real/{dataset_id}.csv" for dataset_id in DATASETS}
        },
        "models": {"enabled": list(MODELS)},
        "explainers": {"enabled": list(common.REAL_EXPLAINERS)},
        "cslaw": {"corrections": "v2", "persist_canonical_explanations": True},
    }


def check_merge(v2_native_dir: Path, root: Path, checks: Checklist) -> None:
    reference = common.load_results(v2_native_dir / "results.json")
    shard_groups = {
        "shard_heloc": {("heloc", "primary"), ("heloc", "audit")},
        "shard_sgc_primary": {("south_german_credit", "primary")},
        "shard_sgc_audit": {("south_german_credit", "audit")},
    }
    shard_dirs: list[Path] = []
    for label, groups in shard_groups.items():
        directory = write_bundle(root / "shards" / label, shard_payload(reference, groups, label))
        common.write_json(directory / "runtime_config.json", shard_runtime_config(groups))
        shard_dirs.append(directory)

    saved_force = os.environ.get("CSLAW_FORCE")
    os.environ["CSLAW_FORCE"] = "0"
    try:
        merged_dir = root / "merged"
        merge_bundles.main(["--inputs", *map(str, shard_dirs), "--out-dir", str(merged_dir)])
        merged = common.load_results(merged_dir / "results.json")
        assert merged["summary"] == reference["summary"], "merge must reassemble the single-run order"
        assert merged["overview"] == runner._build_overview(reference["summary"])
        assert merged["experiment_name"] == reference["experiment_name"]
        assert merged["cslaw"]["groups"] == reference["cslaw"]["groups"]
        assert merged["cslaw"]["corrections"] == common.CORRECTIONS_V2
        assert merged["cslaw"]["config"] == reference["cslaw"]["config"]
        provenance = merged["cslaw"]["provenance"]
        assert provenance["datasets"] == reference["cslaw"]["provenance"]["datasets"]
        assert provenance["metadata"] == reference["cslaw"]["provenance"]["metadata"]
        assert set(provenance["config_by_shard"]) == set(shard_groups), provenance
        assert [item["label"] for item in merged["merge"]["inputs"]] == list(shard_groups)
        runtime = common.load_results(merged_dir / "runtime_config.json")
        assert runtime["datasets"]["enabled"] == list(DATASETS), runtime["datasets"]
        assert runtime["tracks"] == {"run_primary_track": True, "run_audit_track": True}
        json.dumps(merged, allow_nan=False)

        expect_exit(
            "merge_bundles (existing output)",
            merge_bundles.run,
            shard_dirs,
            merged_dir,
            contains="already exists",
        )
        duplicate_dir = root / "shards" / "shard_heloc_copy"
        shutil.copytree(shard_dirs[0], duplicate_dir)
        expect_exit(
            "merge_bundles (duplicate groups)",
            merge_bundles.run,
            [shard_dirs[0], duplicate_dir],
            root / "merged_duplicate",
            contains="duplicate",
        )
        mismatched_dir = root / "shards" / "shard_sgc_audit_budget12"
        shutil.copytree(shard_dirs[2], mismatched_dir)
        common.write_json(
            mismatched_dir / "runtime_config.json",
            shard_runtime_config(shard_groups["shard_sgc_audit"], budget=12),
        )
        expect_exit(
            "merge_bundles (different settings)",
            merge_bundles.run,
            [shard_dirs[0], shard_dirs[1], mismatched_dir],
            root / "merged_mismatch",
            contains="settings differ",
        )
        assert not (root / "merged_mismatch" / "results.json").exists()

        os.environ["CSLAW_FORCE"] = "1"
        merge_bundles.run(shard_dirs, merged_dir)
        assert common.load_results(merged_dir / "results.json")["summary"] == reference["summary"]
        not_a_merge = root / "not_a_merge"
        not_a_merge.mkdir(parents=True, exist_ok=True)
        common.write_json(
            not_a_merge / "results.json", {"experiment_name": "not a merge", "summary": []}
        )
        expect_exit(
            "merge_bundles (never overwrites a non-merge results.json)",
            merge_bundles.run,
            shard_dirs,
            not_a_merge,
            contains="not a previous merge output",
        )
    finally:
        if saved_force is None:
            os.environ.pop("CSLAW_FORCE", None)
        else:
            os.environ["CSLAW_FORCE"] = saved_force
    checks.ok(
        "merge_bundles: three dataset:track shards reassemble the single-run bundle exactly "
        "(rows, order, overview, cslaw groups, provenance); refuses existing output, duplicate "
        "groups and different settings; replaces only a previous merge under CSLAW_FORCE=1"
    )


def check_version_diff(v1_dir: Path, v2_dir: Path, root: Path, checks: Checklist) -> None:
    def means(scores: dict[str, float]) -> dict[str, Any]:
        return {
            version_diff.POOLED: {
                metric: {explainer: {"mean": value, "rows": 10} for explainer, value in scores.items()}
                for metric in version_diff.METRICS
            }
        }

    _rows, leaders, pairs = version_diff.compare(
        means({"shap": 0.9, "lime": 0.5, "random_reasons": 0.99}),
        means({"shap": 0.4, "lime": 0.6, "random_reasons": 0.99}),
    )
    row = next(item for item in leaders if item["metric"] == "faithfulness")
    assert (row["v1_leader"], row["v2_leader"], row["leader_changed"]) == ("shap", "lime", True), row
    assert row["kendall_tau"] == -1.0 and row["n_pairs_flipped"] == 1, row
    assert all(item["v1_leader"] != "random_reasons" and item["v2_leader"] != "random_reasons" for item in leaders)
    assert all(not common.is_null_baseline(item["explainer_a"]) for item in pairs)

    out = root / "analysis_v2" / "version_diff"
    version_diff.main(["--v1", str(v1_dir), "--v2", str(v2_dir), "--out-dir", str(out)])
    assert_files(
        [
            out / name
            for name in (
                "version_diff_settings.csv",
                "version_diff_means.csv",
                "version_diff_leaders.csv",
                "version_diff_pairs.csv",
                "version_diff_summary.md",
            )
        ],
        "version_diff",
    )
    settings = {row["setting"]: row for row in read_csv(out / "version_diff_settings.csv")}
    assert settings["corrections_version"]["v1"] == "v1" and settings["corrections_version"]["v2"] == "v2"
    assert settings["feature_taxonomy.path"]["differs"] == "True"
    text = (out / "version_diff_summary.md").read_text(encoding="utf-8")
    assert "Instance-level comparison is not possible" in text
    leaders_csv = read_csv(out / "version_diff_leaders.csv")
    assert leaders_csv and all(
        row["v1_leader"] not in common.NULL_BASELINES and row["v2_leader"] not in common.NULL_BASELINES
        for row in leaders_csv
    )
    checks.ok(
        "version_diff: leader changes, Kendall tau and flipped pairs; settings and the "
        "instance-level caveat in the summary; null baselines never leaders"
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def check_reason_units(checks: Checklist) -> None:
    """Pure-function checks: notice-level reason units and leader lines."""
    pay = regb_reference.temporal_group("PAY_0")
    bills = regb_reference.temporal_group("BILL_AMT1")
    assert pay == regb_reference.temporal_group("PAY_AMT1") != bills
    canonical = {
        "principal_reasons": [
            {
                "rank": 1,
                "supporting_features": ["PAY_0", "PAY_2", "PAY_3", "PAY_AMT1"],
                "signed_direction": "increases_denial",
                "grouped_reason_type": "compressed_temporal",
            },
            {"rank": 2, "supporting_features": ["LIMIT_BAL"], "signed_direction": "increases_denial"},
            {"rank": 3, "supporting_features": ["AGE"], "signed_direction": "increases_denial"},
            {
                "rank": 4,
                "supporting_features": ["BILL_AMT1", "BILL_AMT2"],
                "signed_direction": "decreases_denial",
                "grouped_reason_type": "compressed_temporal",
            },
        ]
    }
    units = regb_reference.reason_unit_labels(canonical, "principal")
    assert units == [pay, "LIMIT_BAL", "AGE", bills], units
    assert regb_reference.reason_unit_labels(canonical, "principal_adverse") == [
        pay,
        "LIMIT_BAL",
        "AGE",
    ]
    values = {
        "PAY_0": 3.0,
        "PAY_2": 2.5,
        "LIMIT_BAL": 2.0,
        "AGE": 1.5,
        "EDUCATION": 1.0,
        "PAY_3": 0.5,
        "SEX": -1.0,
    }
    means = {name: 0.0 for name in values}
    m1_units = regb_reference.regb_unit_ranking(values, means, K)
    assert m1_units == [pay, "LIMIT_BAL", "AGE", "EDUCATION"], m1_units
    assert regb_reference.regb_reason_units(values, means, K) == [
        ["PAY_0", "PAY_2", "PAY_3"],
        ["LIMIT_BAL"],
        ["AGE"],
        ["EDUCATION"],
    ]
    assert regb_reference.regb_reason_units(values, means, 2) == [["PAY_0", "PAY_2"], ["LIMIT_BAL"]]
    grouped = regb_reference.compare_reason_units(units, {"m1": m1_units, "m2": None}, K, "pr")
    assert grouped["pr_overlap_m1_grouped"] == 0.75, grouped
    assert grouped["pr_jaccard_m1_grouped"] == 0.6, grouped
    assert grouped["pr_top1_m1_grouped"] is True, grouped
    assert grouped["pr_outside_m1_grouped"] == 0.25, grouped
    assert grouped["pr_overlap_m2_grouped"] is None, grouped
    # The feature-level variant lets the temporal reason fill all k slots.
    flat = regb_reference.compare_reason_lists(
        regb_reference.reason_features(canonical, "principal"),
        {"m1": regb_reference.regb_ranking(values, means, K), "m2": None},
        K,
        "pr",
    )
    assert flat["pr_overlap_m1"] == 0.5, flat
    names = regb_reference.instance_metric_names()
    assert {"pr_overlap_m1_grouped", "pr_top1_m1_grouped", "pra_outside_m1_grouped"} <= set(names)

    # multiplicity: explainer and Reg B lists cut to k units on the same basis.
    for unit in multiplicity.REASON_UNITS:
        explainer_list = multiplicity.explainer_reason_list(canonical, "principal", K, unit)
        regb_list = multiplicity.regb_reason_list(values, means, K, unit)
        assert len(explainer_list) <= K and len(regb_list) <= K, (unit, explainer_list, regb_list)
    assert multiplicity.explainer_reason_list(canonical, "principal", K, "grouped") == units
    assert multiplicity.regb_reason_list(values, means, K, "grouped") == m1_units
    assert multiplicity.explainer_reason_list(canonical, "principal", K, "feature_at_k") == [
        "PAY_0",
        "PAY_2",
        "PAY_3",
        "PAY_AMT1",
    ]

    # proxy_substitution: replacements counted per reason unit, features kept as-is.
    dataset_id = "default_of_credit_card_clients"
    registry = {
        dataset_id: {
            "primary_track_excluded_features": ["SEX"],
            "audit_track_excluded_features": [],
        }
    }
    taxonomy = {
        dataset_id: {
            name: {"sensitivity": label}
            for name, label in (
                ("SEX", "S"),
                ("LIMIT_BAL", "P"),
                ("PAY_0", "N"),
                ("PAY_2", "N"),
                ("PAY_3", "N"),
            )
        }
    }
    pairs = {
        (dataset_id, "logistic_regression", "shap", 0): {
            "primary": [["PAY_0", "PAY_2", "PAY_3"], ["LIMIT_BAL"]],
            "audit": [["PAY_0"], ["SEX"]],
        }
    }
    rows, _transitions, _coverage = proxy_substitution.substitution_rows(
        pairs, {"shap": "explainer"}, registry, taxonomy, k=K
    )
    assert len(rows) == 1, rows
    assert rows[0]["replacements"] == "PAY_2|PAY_3|LIMIT_BAL", rows[0]
    assert rows[0]["n_replacement_reasons"] == 1, rows[0]
    assert abs(rows[0]["jaccard_primary_audit_reasons"] - 1 / 3) < 1e-12, rows[0]

    # Leader lines never name a by-construction cell or a null baseline as the leader.
    def stat(explainer_id: str, mean: float, *, flagged: bool, null: bool = False) -> dict[str, Any]:
        return {
            "model_id": "logistic_regression",
            "metric": "pr_overlap_m1_grouped",
            "explainer_id": explainer_id,
            "is_null_baseline": null,
            "by_construction": flagged,
            "by_construction_flags": "shap_lr_linear_decomposition" if flagged else "",
            "mean": mean,
            "ci_low": mean - 0.1,
            "ci_high": mean + 0.1,
            "n": 10,
        }

    pooled = [
        stat("shap", 0.9, flagged=True),
        stat("lime", 0.6, flagged=False),
        stat("random_reasons", 0.95, flagged=False, null=True),
    ]
    lines = regb_reference.leader_lines(pooled, ["logistic_regression"], "pr_overlap_m1_grouped")
    assert len(lines) == 1, lines
    headline = lines[0].split(";")[0]
    assert "`lime`" in headline and "`shap`" not in headline, lines[0]
    assert "random_reasons" not in headline, lines[0]
    assert "by-construction, not ranked: `shap`" in lines[0], lines[0]
    only_flagged = regb_reference.leader_lines(
        [stat("shap", 0.9, flagged=True)], ["logistic_regression"], "pr_overlap_m1_grouped"
    )
    assert "`shap`" not in only_flagged[0].split(";")[0], only_flagged
    checks.ok(
        "reason units: a temporal reason counts once in Reg B overlap, multiplicity and proxy "
        "substitution; leader lines never name a by-construction cell"
    )


def run_script_selftests(checks: Checklist) -> None:
    for label, func in (
        ("rescore.py --selftest", rescore.run_selftest),
        ("paper_tables.py --selftest", paper_tables.run_selftest),
        ("compare_bundles.py --selftest", compare_bundles.run_selftest),
        ("denial_reasons.py --selftest", denial_reasons.run_selftest),
    ):
        func()
        checks.ok(label)


def run_all(root: Path, *, script_selftests: bool) -> Checklist:
    checks = Checklist()
    if script_selftests:
        print("[1/3] built-in script self-tests")
        run_script_selftests(checks)
    print("[2/3] cross-agent contract checks")
    check_contract(checks)
    check_reason_units(checks)
    check_corrections(checks)
    check_pure_fixes(checks)
    check_v2_reason_units(checks)

    print("[3/3] synthetic bundles through every analysis script")
    meta_root = root / "repo"
    write_metadata(meta_root)
    outputs = root / "outputs"
    payload, facts = build_shared_payload()
    check_bundle_schema(payload, facts, checks)
    native_payload = subset_payload(payload, common.REAL_EXPLAINERS)
    null_payload = subset_payload(payload, common.NULL_BASELINES, drop_native=True)
    old_payload = strip_persisted(payload)
    shared_dir = write_bundle(outputs / "shared_bundle", payload)
    native_dir = write_bundle(outputs / "native_bundle", native_payload)
    null_dir = write_bundle(outputs / "nullbaselines_bundle", null_payload)
    old_dir = write_bundle(outputs / "old_bundle", old_payload)

    rescored_root, sweep_csv = check_rescore(native_payload, native_dir, root / "offline", checks)
    _rows, leaders = rescore.dsv_weight_sweep(payload["summary"], [0.0, 0.25, 1.0])
    assert leaders and all(row["published_leader"] not in common.NULL_BASELINES for row in leaders)
    assert all(row["leader_at_weight_0"] not in common.NULL_BASELINES for row in leaders)
    checks.ok("S-weight sweep over a bundle that contains null baselines never names one as leader")
    check_compare(payload, old_payload, shared_dir, root, checks)
    check_paper_tables(
        native_dir, null_dir, native_payload, outputs, rescored_root, sweep_csv, root, checks
    )
    check_regb(payload, shared_dir, root, checks)
    check_multiplicity(payload, shared_dir, root, checks)
    check_proxy(payload, shared_dir, meta_root, root, checks)
    check_common_faithfulness(native_payload, null_payload, native_dir, null_dir, root, checks)
    check_old_bundle(old_payload, old_dir, meta_root, root, checks)

    print("[3/3, v2] v2-shaped bundles (corrections, provenance, sub-components), merge, version diff")
    v2_shared = make_v2_payload(payload, write_v2_metadata(meta_root))
    v2_native = subset_payload(v2_shared, common.REAL_EXPLAINERS)
    v2_null = subset_payload(v2_shared, common.NULL_BASELINES, drop_native=True)
    v2_dirs = {
        "native": native_dir,
        "v2_shared": write_bundle(outputs / "v2_shared_bundle", v2_shared),
        "v2_native": write_bundle(outputs / "v2_native_bundle", v2_native),
        "v2_null": write_bundle(outputs / "v2_nullbaselines_bundle", v2_null),
    }
    check_v2_bundles(v2_shared, v2_native, v2_dirs, meta_root, root, checks)
    check_merge(v2_dirs["v2_native"], root, checks)
    check_version_diff(native_dir, v2_dirs["v2_native"], root, checks)
    return checks


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end synthetic self-test of the CS&Law analysis scripts (no real data)."
    )
    parser.add_argument(
        "--keep",
        help="Write the synthetic bundles and outputs into this new (or empty) directory and keep them.",
    )
    parser.add_argument(
        "--skip-script-selftests",
        action="store_true",
        help="Skip the scripts' own --selftest routines and run only the integration checks.",
    )
    args = parser.parse_args(argv)
    # The runner helpers emit progress events; never let synthetic events reach a
    # real progress.jsonl inherited from the calling shell.
    for name in ("LEGALBENCH_PROGRESS", "LEGALBENCH_PROGRESS_LOG"):
        os.environ.pop(name, None)
    if args.keep:
        root = Path(args.keep).expanduser().resolve()
        if root.exists() and any(root.iterdir()):
            raise SystemExit(f"--keep directory {root} is not empty; choose a new directory.")
        root.mkdir(parents=True, exist_ok=True)
        checks = run_all(root, script_selftests=not args.skip_script_selftests)
        print(f"Synthetic bundles and outputs kept in: {root}")
    else:
        with tempfile.TemporaryDirectory(prefix="cslaw_selftest_") as tmp:
            checks = run_all(Path(tmp), script_selftests=not args.skip_script_selftests)
    skipped = f"; {len(checks.skipped)} skipped: " + " | ".join(checks.skipped) if checks.skipped else ""
    print(f"cslaw selftest: OK ({len(checks.passed)} checks{skipped})")


if __name__ == "__main__":
    main()
