import math
import os
import statistics
from typing import Any

from ..corrections import GEO_CODE_COLUMNS, V1, CorrectionOptions, model_corrections
from ..types import DatasetBundle, TabularDataset
from .stubs import HeuristicModel, RowPrediction, fit_stub_model


class VectorizedClassifierModel:
    # Opt-in (cslaw.use_native_contributions_for_ranking): when True, explainer-internal
    # feature ranking uses the fitted model's own additive contributions instead of the
    # heuristic proxy returned by original_feature_contributions. Default False keeps the
    # baseline behaviour unchanged.
    use_native_ranking: bool = False
    # Versioned corrections (legalbench_xai.corrections). The runner sets the instance
    # attribute after fit_model; explainers read it via model_corrections(model).
    corrections: CorrectionOptions = V1
    # How native_feature_contributions_batch decomposes the score. Subclasses override.
    native_contribution_source: str = "heuristic_stub"

    def __init__(
        self,
        *,
        model_id: str,
        backend_implementation: str,
        dataset_bundle: DatasetBundle,
        vectorizer: Any,
        classifier: Any,
        encoded_feature_names: list[str],
        background_matrix: Any,
        heuristic_helper: HeuristicModel,
        feature_fill_values: dict[str, Any],
        matrix_transformer: Any = None,
        threshold: float = 0.5,
    ) -> None:
        self.model_id = model_id
        self.backend_implementation = backend_implementation
        self.dataset_bundle = dataset_bundle
        self.vectorizer = vectorizer
        self.classifier = classifier
        self.encoded_feature_names = encoded_feature_names
        self.background_matrix = background_matrix
        self.heuristic_helper = heuristic_helper
        self.feature_fill_values = feature_fill_values
        self.matrix_transformer = matrix_transformer
        self.threshold = threshold
        self._shap_explainer: Any = None
        self._shap_mode: str | None = None
        self._lime_explainers: dict[Any, Any] = {}

    @property
    def shap_mode(self) -> str | None:
        return self._shap_mode

    def predict_row(self, row: dict[str, Any], instance_id: str) -> RowPrediction:
        vector = self.vectorize_row(row)
        probability = float(self.classifier.predict_proba(vector)[0][1])
        if hasattr(self.classifier, "decision_function"):
            raw_score = float(self.classifier.decision_function(vector)[0])
        else:
            raw_score = _logit(probability)
        label = (
            self.dataset_bundle.negative_label
            if probability >= self.threshold
            else self.dataset_bundle.positive_label
        )
        return RowPrediction(
            instance_id=instance_id,
            score=probability,
            label=label,
            threshold=self.threshold,
            decision_margin=probability - self.threshold,
            raw_score=raw_score,
            feature_contributions=self.original_feature_contributions(row),
        )

    def predict_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        instance_id_prefix: str,
    ) -> list[RowPrediction]:
        if not rows:
            return []
        vector = self.vectorizer.transform([self._feature_dict(row) for row in rows])
        if self.matrix_transformer is not None:
            vector = self.matrix_transformer.transform(vector)
        probabilities = self.classifier.predict_proba(vector)[:, 1]
        if hasattr(self.classifier, "decision_function"):
            raw_scores = self.classifier.decision_function(vector)
        else:
            raw_scores = [_logit(float(probability)) for probability in probabilities]

        predictions: list[RowPrediction] = []
        for index, (probability, raw_score) in enumerate(
            zip(probabilities, raw_scores, strict=False)
        ):
            score = float(probability)
            label = (
                self.dataset_bundle.negative_label
                if score >= self.threshold
                else self.dataset_bundle.positive_label
            )
            predictions.append(
                RowPrediction(
                    instance_id=f"{instance_id_prefix}-row{index}",
                    score=score,
                    label=label,
                    threshold=self.threshold,
                    decision_margin=score - self.threshold,
                    raw_score=float(raw_score),
                    feature_contributions={},
                )
            )
        return predictions

    def recommend_changes(
        self,
        row: dict[str, Any],
        prediction: RowPrediction,
        allow_illegal: bool,
        limit: int = 2,
        block_nonactionable: bool = False,
    ) -> list[dict[str, Any]]:
        return self.heuristic_helper.recommend_changes(
            row=self._feature_dict(row),
            prediction=prediction,
            allow_illegal=allow_illegal,
            limit=limit,
            block_nonactionable=block_nonactionable,
        )

    def vectorize_row(self, row: dict[str, Any]) -> Any:
        vector = self.vectorizer.transform([self._feature_dict(row)])
        if self.matrix_transformer is not None:
            return self.matrix_transformer.transform(vector)
        return vector

    def original_feature_contributions(self, row: dict[str, Any]) -> dict[str, float]:
        proxy_prediction = self.heuristic_helper.predict_row(
            row=self._feature_dict(row),
            instance_id=f"{self.model_id}-proxy",
        )
        return proxy_prediction.feature_contributions

    @property
    def native_ranking_active(self) -> bool:
        """True when explainer-internal ranking uses native contributions.

        Either the legacy switch use_native_ranking or correction native_contributions
        (v2) turns it on, so a runner that resets use_native_ranking from the legacy
        cslaw flag cannot silently undo the v2 correction.
        """
        return bool(self.use_native_ranking) or model_corrections(self).native_contributions

    def ranking_contributions(self, row: dict[str, Any]) -> dict[str, float]:
        """Per-feature contributions that explainers use to rank candidate features.

        Correction native_contributions (v2), also the pre-v2 use_native_ranking switch:
        returns native_feature_contributions(row) (the fitted model's own additive
        log-odds contributions; LR coef x standardized x, EBM eval_terms, XGBoost
        pred_contribs). Otherwise returns original_feature_contributions(row), which for
        EBM and XGBoost is a taxonomy-weighted heuristic proxy. With the defaults
        (use_native_ranking False, corrections V1) this is exactly the baseline input.
        """
        if self.native_ranking_active:
            return self.native_feature_contributions(row)
        return self.original_feature_contributions(row)

    def native_feature_contributions(self, row: dict[str, Any]) -> dict[str, float]:
        """Single-row convenience wrapper around native_feature_contributions_batch.

        Units: log-odds of the ADVERSE class (classifier label 1); positive values push
        toward denial. Keys: every bundle included (base) feature name. The reference
        point and the interaction split rule are those of the subclass's
        native_feature_contributions_batch.
        """
        contributions, _bias = self.native_feature_contributions_batch([row])
        return contributions[0]

    def native_feature_contributions_batch(
        self,
        rows: list[dict[str, Any]],
    ) -> tuple[list[dict[str, float]], float]:
        """Fitted-model additive contributions for many rows in one vectorized call.

        Returns (per-row dicts, bias). Each dict covers ALL bundle included feature
        names (base names, initialised to 0.0; one-hot columns "feat=value" are summed
        back to "feat"). Values are in log-odds of the ADVERSE class (positive = pushes
        toward denial) and sum(values) + bias == the model's raw score
        (decision_function, or logit(P(adverse)) for XGBoost). Rows are vectorized
        exactly as predict_rows does (_feature_dict + vectorizer + matrix_transformer).

        This base class has no fitted-model decomposition, so it returns the heuristic
        proxy (original_feature_contributions) with bias 0.0 and
        native_contribution_source "heuristic_stub"; the proxy is unitless and NOT
        additive to the model score. Subclasses override _native_decomposition.
        """
        if not self._has_native_decomposition():
            return [self.original_feature_contributions(row) for row in rows], 0.0
        if not rows:
            _values, bias = self.native_feature_contributions_from_matrix(
                self.background_matrix[:1]
            )
            return [], bias
        return self.native_feature_contributions_from_matrix(self._model_input_matrix(rows))

    def native_feature_contributions_from_matrix(
        self,
        matrix: Any,
    ) -> tuple[list[dict[str, float]], float]:
        """Same as native_feature_contributions_batch, but takes a model-input matrix.

        `matrix` must already be in the classifier's input space (vectorized, and scaled
        for logistic regression), e.g. self.background_matrix. Units, sign, keys and
        bias are as in native_feature_contributions_batch. When no fitted-model
        decomposition exists, rows are reconstructed from the matrix and the heuristic
        proxy is returned with bias 0.0.
        """
        decomposition = self._native_decomposition(matrix)
        if decomposition is None:
            rows = self.rows_from_model_matrix(matrix)
            return [self.original_feature_contributions(row) for row in rows], 0.0
        column_values, bias = decomposition
        return self._aggregate_native_columns(column_values), bias

    def rows_from_model_matrix(self, matrix: Any) -> list[dict[str, Any]]:
        """Reconstruct feature dicts from a model-input matrix (inverse of vectorizing).

        Undoes matrix_transformer (if any); numeric columns become floats, and an active
        one-hot column "feat=value" (encoded value > 0.5) becomes {"feat": "value"}.
        """
        import numpy as np

        raw = matrix
        if self.matrix_transformer is not None:
            raw = self.matrix_transformer.inverse_transform(matrix)
        rows: list[dict[str, Any]] = []
        for values in np.asarray(raw, dtype=float).tolist():
            row: dict[str, Any] = {}
            for encoded_name, value in zip(self.encoded_feature_names, values, strict=False):
                if "=" in encoded_name:
                    if value > 0.5:
                        base_name, category = encoded_name.split("=", 1)
                        row[base_name] = category
                    continue
                row[encoded_name] = float(value)
            rows.append(row)
        return rows

    def _native_decomposition(self, matrix: Any) -> tuple[list[list[float]], float] | None:
        """Return (per-row values for each _native_column_shares column, bias) or None.

        None means that no fitted-model decomposition is available (base class).
        """
        return None

    def _has_native_decomposition(self) -> bool:
        return (
            type(self)._native_decomposition
            is not VectorizedClassifierModel._native_decomposition
        )

    def _native_column_shares(self) -> list[list[tuple[str, float]]]:
        """Map each decomposition column to (base feature name, share) pairs.

        Default: one column per encoded feature, assigned fully to its base feature.
        """
        return [
            [(self.base_feature_name(encoded_name), 1.0)]
            for encoded_name in self.encoded_feature_names
        ]

    def _aggregate_native_columns(
        self,
        column_values: list[list[float]],
    ) -> list[dict[str, float]]:
        shares = self._native_column_shares()
        feature_names = list(self.dataset_bundle.included_feature_names)
        aggregated_rows: list[dict[str, float]] = []
        for values in column_values:
            contributions = {feature_name: 0.0 for feature_name in feature_names}
            for value, column_shares in zip(values, shares, strict=False):
                for base_name, share in column_shares:
                    if base_name in contributions:
                        contributions[base_name] += (
                            float(value) if share == 1.0 else float(value) * share
                        )
            aggregated_rows.append(contributions)
        return aggregated_rows

    def _model_input_matrix(self, rows: list[dict[str, Any]]) -> Any:
        matrix = self.vectorizer.transform([self._feature_dict(row) for row in rows])
        if self.matrix_transformer is not None:
            matrix = self.matrix_transformer.transform(matrix)
        return matrix

    def get_shap_explainer(self) -> Any:
        if self._shap_explainer is not None:
            return self._shap_explainer

        import shap

        self._shap_explainer = shap.Explainer(
            self.classifier.predict_proba,
            self.background_matrix,
            feature_names=self.encoded_feature_names,
        )
        self._shap_mode = "generic"
        return self._shap_explainer

    def get_lime_explainer(self, seed: int = 0) -> Any:
        # Correction option lime_sample_around_instance (robustness re-run only):
        # perturbations are drawn around the explained row. Without it the call
        # (and the cache key) is exactly the published one.
        # Correction option lime_discretize_continuous (robustness re-run only): the
        # library's default quartile discretization.
        sample_around_instance = bool(model_corrections(self).lime_sample_around_instance)
        discretize = bool(model_corrections(self).lime_discretize_continuous)
        cache_key: Any = seed
        if sample_around_instance:
            cache_key = (seed, "sample_around_instance")
        elif discretize:
            cache_key = (seed, "discretize_continuous")
        if cache_key in self._lime_explainers:
            return self._lime_explainers[cache_key]

        from lime.lime_tabular import LimeTabularExplainer

        kernel_width = 0.75 * max(1, len(self.encoded_feature_names)) ** 0.5
        extra_kwargs: dict[str, Any] = (
            {"sample_around_instance": True} if sample_around_instance else {}
        )
        explainer = LimeTabularExplainer(
            training_data=self.background_matrix,
            feature_names=self.encoded_feature_names,
            class_names=[
                self.dataset_bundle.positive_label,
                self.dataset_bundle.negative_label,
            ],
            mode="classification",
            kernel_width=kernel_width,
            discretize_continuous=discretize,
            random_state=seed,
            **extra_kwargs,
        )
        self._lime_explainers[cache_key] = explainer
        return explainer

    def _feature_dict(self, row: dict[str, Any]) -> dict[str, Any]:
        values = {
            feature_name: self._normalize_feature_value(feature_name, row.get(feature_name))
            for feature_name in self.dataset_bundle.included_feature_names
        }
        encoder = getattr(self.vectorizer, "code_encoder_", None)
        return encoder.transform(values) if encoder is not None else values

    def _normalize_feature_value(self, feature_name: str, value: Any) -> Any:
        if _is_missing(value):
            return self.feature_fill_values[feature_name]
        return value

    @staticmethod
    def base_feature_name(encoded_feature_name: str) -> str:
        if "=" in encoded_feature_name:
            return encoded_feature_name.split("=", 1)[0]
        return encoded_feature_name


class SklearnLogisticRegressionModel(VectorizedClassifierModel):
    native_contribution_source = "lr_coef_x_standardized"

    def __init__(
        self,
        dataset_bundle: DatasetBundle,
        vectorizer: Any,
        classifier: Any,
        encoded_feature_names: list[str],
        background_matrix: Any,
        heuristic_helper: HeuristicModel,
        feature_fill_values: dict[str, Any],
        matrix_transformer: Any = None,
        threshold: float = 0.5,
    ) -> None:
        super().__init__(
            model_id="logistic_regression",
            backend_implementation="sklearn_logistic_regression",
            dataset_bundle=dataset_bundle,
            vectorizer=vectorizer,
            classifier=classifier,
            encoded_feature_names=encoded_feature_names,
            background_matrix=background_matrix,
            heuristic_helper=heuristic_helper,
            feature_fill_values=feature_fill_values,
            matrix_transformer=matrix_transformer,
            threshold=threshold,
        )
        self.background_reference_vector = background_matrix.mean(axis=0)

    def original_feature_contributions(self, row: dict[str, Any]) -> dict[str, float]:
        vector = self.vectorize_row(row)[0]
        coefficients = self.classifier.coef_[0]
        contributions = {
            feature_name: 0.0
            for feature_name in self.dataset_bundle.included_feature_names
        }
        for encoded_name, value, coefficient in zip(
            self.encoded_feature_names,
            vector,
            coefficients,
            strict=False,
        ):
            base_name = self.base_feature_name(encoded_name)
            if base_name in contributions:
                contributions[base_name] += float(value) * float(coefficient)
        return contributions

    def _native_decomposition(self, matrix: Any) -> tuple[list[list[float]], float]:
        """Logistic regression: coef_j * standardized x_j per encoded column.

        Units: log-odds of the adverse class; positive = toward denial. Reference: the
        training-split mean (StandardScaler centres each column, so x_scaled = 0 at the
        training mean; one-hot dummies are centred too). Bias = intercept_[0], so
        sum(contributions) + bias == decision_function. Aggregated by base feature this
        equals original_feature_contributions(row) exactly (same products, same
        summation order). No interaction terms.
        """
        import numpy as np

        coefficients = np.asarray(self.classifier.coef_[0], dtype=float)
        products = np.asarray(matrix, dtype=float) * coefficients
        return products.tolist(), float(self.classifier.intercept_[0])

    def get_shap_explainer(self) -> Any:
        self._shap_mode = "linear_exact"
        return self._shap_explainer

    def exact_shap_values(self, row: dict[str, Any]) -> list[float]:
        vector = self.vectorize_row(row)[0]
        coefficients = self.classifier.coef_[0]
        centered = vector - self.background_reference_vector
        return [
            float(value) * float(coefficient)
            for value, coefficient in zip(centered, coefficients, strict=False)
        ]


class XGBoostClassifierModel(VectorizedClassifierModel):
    native_contribution_source = "xgb_pred_contribs"

    def __init__(
        self,
        dataset_bundle: DatasetBundle,
        vectorizer: Any,
        classifier: Any,
        encoded_feature_names: list[str],
        background_matrix: Any,
        heuristic_helper: HeuristicModel,
        feature_fill_values: dict[str, Any],
        threshold: float = 0.5,
    ) -> None:
        super().__init__(
            model_id="xgboost",
            backend_implementation="xgboost_classifier",
            dataset_bundle=dataset_bundle,
            vectorizer=vectorizer,
            classifier=classifier,
            encoded_feature_names=encoded_feature_names,
            background_matrix=background_matrix,
            heuristic_helper=heuristic_helper,
            feature_fill_values=feature_fill_values,
            threshold=threshold,
        )

    def get_shap_explainer(self) -> Any:
        if self._shap_explainer is not None:
            return self._shap_explainer

        import shap

        try:
            self._shap_explainer = shap.TreeExplainer(self.classifier)
            self._shap_mode = "tree"
        except Exception:
            self._shap_explainer = shap.Explainer(
                self.classifier.predict_proba,
                self.background_matrix,
                feature_names=self.encoded_feature_names,
            )
            self._shap_mode = "generic"
        return self._shap_explainer

    def _native_decomposition(self, matrix: Any) -> tuple[list[list[float]], float]:
        """XGBoost: exact TreeSHAP from booster.predict(DMatrix(X), pred_contribs=True).

        Units: log-odds (output margin) of the adverse class; positive = toward denial.
        Columns 0..p-1 are per encoded column (aggregated to base names); the last
        column is the bias, i.e. the cover-weighted training expectation of the margin
        (constant across rows), so sum(contributions) + bias == output margin ==
        logit(P(adverse)) up to float32 rounding. TreeSHAP already distributes
        interaction effects among features; no further split rule is applied.
        """
        import numpy as np
        import xgboost

        contributions = self.classifier.get_booster().predict(
            xgboost.DMatrix(np.asarray(matrix, dtype=float)),
            pred_contribs=True,
        )
        contributions = np.asarray(contributions, dtype=float)
        expected_columns = len(self.encoded_feature_names) + 1
        if contributions.ndim != 2 or contributions.shape[1] != expected_columns:
            raise ValueError(
                "unexpected XGBoost pred_contribs shape "
                f"{contributions.shape}; expected (n, {expected_columns})"
            )
        bias = float(contributions[0, -1]) if contributions.shape[0] else 0.0
        return contributions[:, :-1].tolist(), bias


class EBMClassifierModel(VectorizedClassifierModel):
    native_contribution_source = "ebm_eval_terms_pair_split"

    def __init__(
        self,
        dataset_bundle: DatasetBundle,
        vectorizer: Any,
        classifier: Any,
        encoded_feature_names: list[str],
        background_matrix: Any,
        heuristic_helper: HeuristicModel,
        feature_fill_values: dict[str, Any],
        threshold: float = 0.5,
    ) -> None:
        super().__init__(
            model_id="ebm",
            backend_implementation="ebm_classifier",
            dataset_bundle=dataset_bundle,
            vectorizer=vectorizer,
            classifier=classifier,
            encoded_feature_names=encoded_feature_names,
            background_matrix=background_matrix,
            heuristic_helper=heuristic_helper,
            feature_fill_values=feature_fill_values,
            threshold=threshold,
        )

    def _native_decomposition(self, matrix: Any) -> tuple[list[list[float]], float]:
        """EBM: per-term scores from classifier.eval_terms(X) (interpret >= 0.7).

        Units: log-odds of the adverse class; positive = toward denial. Each term score
        is the term's shape-function value at x; EBM centres terms on the training data,
        so the reference is the training-set average score. Bias = intercept_, so
        sum(term scores) + bias == decision_function. Terms are mapped through
        term_features_ (indices into the encoded input columns, i.e. into
        encoded_feature_names, including one-hot "feat=value" columns) and then to base
        feature names. Pair-split rule: an interaction term over k encoded columns is
        split equally, 1/k per column (50/50 for pairs); when both columns of a pair map
        to the same base feature, that feature receives the whole term.
        """
        import numpy as np

        feature_names_in = getattr(self.classifier, "feature_names_in_", None)
        if feature_names_in is not None and len(feature_names_in) != len(
            self.encoded_feature_names
        ):
            raise ValueError(
                "EBM input feature count does not match the encoded feature count"
            )
        term_scores = np.asarray(self.classifier.eval_terms(matrix), dtype=float)
        if term_scores.ndim != 2 or term_scores.shape[1] != len(self.classifier.term_features_):
            raise ValueError(
                f"unexpected EBM eval_terms shape {term_scores.shape} for a binary classifier"
            )
        intercept = np.ravel(np.asarray(self.classifier.intercept_, dtype=float))
        if intercept.size != 1:
            raise ValueError("expected a single EBM intercept for binary classification")
        return term_scores.tolist(), float(intercept[0])

    def _native_column_shares(self) -> list[list[tuple[str, float]]]:
        shares: list[list[tuple[str, float]]] = []
        for feature_indexes in self.classifier.term_features_:
            indexes = [int(index) for index in feature_indexes]
            share = 1.0 / len(indexes) if indexes else 0.0
            shares.append(
                [
                    (self.base_feature_name(self.encoded_feature_names[index]), share)
                    for index in indexes
                ]
            )
        return shares


def fit_sklearn_logistic_regression_model(
    dataset_bundle: DatasetBundle,
    table: TabularDataset,
    threshold: float = 0.5,
    random_state: int = 0,
) -> SklearnLogisticRegressionModel:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    vectorizer, matrix, encoded_feature_names, labels, raw_background_matrix = (
        _prepare_vectorized_training_inputs(dataset_bundle, table)
    )
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(matrix)
    background_matrix = scaler.transform(raw_background_matrix)
    classifier = LogisticRegression(
        max_iter=4000,
        random_state=random_state,
        solver="lbfgs",
    )
    classifier.fit(scaled_matrix, labels)

    heuristic_helper = fit_stub_model(
        model_id="logistic_regression",
        dataset_bundle=dataset_bundle,
        table=_sanitize_tabular_dataset(
            table,
            dataset_bundle.included_feature_names,
            vectorizer.feature_fill_values_,
        ),
    )
    return SklearnLogisticRegressionModel(
        dataset_bundle=dataset_bundle,
        vectorizer=vectorizer,
        classifier=classifier,
        encoded_feature_names=encoded_feature_names,
        background_matrix=background_matrix,
        heuristic_helper=heuristic_helper,
        feature_fill_values=vectorizer.feature_fill_values_,
        matrix_transformer=scaler,
        threshold=threshold,
    )


def fit_xgboost_model(
    dataset_bundle: DatasetBundle,
    table: TabularDataset,
    threshold: float = 0.5,
    random_state: int = 0,
) -> XGBoostClassifierModel:
    from xgboost import XGBClassifier

    vectorizer, matrix, encoded_feature_names, labels, background_matrix = (
        _prepare_vectorized_training_inputs(dataset_bundle, table)
    )
    classifier = XGBClassifier(
        n_estimators=48,
        max_depth=3,
        learning_rate=0.1,
        subsample=1.0,
        colsample_bytree=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        verbosity=0,
        n_jobs=1,
    )
    classifier.fit(matrix, labels)

    heuristic_helper = fit_stub_model(
        model_id="xgboost",
        dataset_bundle=dataset_bundle,
        table=_sanitize_tabular_dataset(
            table,
            dataset_bundle.included_feature_names,
            vectorizer.feature_fill_values_,
        ),
    )
    return XGBoostClassifierModel(
        dataset_bundle=dataset_bundle,
        vectorizer=vectorizer,
        classifier=classifier,
        encoded_feature_names=encoded_feature_names,
        background_matrix=background_matrix,
        heuristic_helper=heuristic_helper,
        feature_fill_values=vectorizer.feature_fill_values_,
        threshold=threshold,
    )


def fit_ebm_model(
    dataset_bundle: DatasetBundle,
    table: TabularDataset,
    threshold: float = 0.5,
    random_state: int = 0,
) -> EBMClassifierModel:
    from interpret.glassbox import ExplainableBoostingClassifier

    vectorizer, matrix, encoded_feature_names, labels, background_matrix = (
        _prepare_vectorized_training_inputs(dataset_bundle, table)
    )
    interactions = _env_int("LEGALBENCH_EBM_INTERACTIONS", 4)
    max_bins = _env_int("LEGALBENCH_EBM_MAX_BINS", 64)
    max_rounds = _env_int("LEGALBENCH_EBM_MAX_ROUNDS", 200)
    classifier = ExplainableBoostingClassifier(
        interactions=interactions,
        max_bins=max_bins,
        max_rounds=max_rounds,
        random_state=random_state,
        n_jobs=1,
    )
    classifier.fit(matrix, labels)

    heuristic_helper = fit_stub_model(
        model_id="ebm",
        dataset_bundle=dataset_bundle,
        table=_sanitize_tabular_dataset(
            table,
            dataset_bundle.included_feature_names,
            vectorizer.feature_fill_values_,
        ),
    )
    return EBMClassifierModel(
        dataset_bundle=dataset_bundle,
        vectorizer=vectorizer,
        classifier=classifier,
        encoded_feature_names=encoded_feature_names,
        background_matrix=background_matrix,
        heuristic_helper=heuristic_helper,
        feature_fill_values=vectorizer.feature_fill_values_,
        threshold=threshold,
    )


def _prepare_vectorized_training_inputs(
    dataset_bundle: DatasetBundle,
    table: TabularDataset,
) -> tuple[Any, Any, list[str], list[int], Any]:
    from sklearn.feature_extraction import DictVectorizer

    feature_fill_values = _infer_feature_fill_values(table.rows, table.feature_names)
    feature_rows = [
        {
            feature_name: _normalize_feature_value(
                row.get(feature_name),
                feature_fill_values[feature_name],
            )
            for feature_name in table.feature_names
        }
        for row in table.rows
    ]
    labels = [
        1 if str(row[dataset_bundle.target]) == dataset_bundle.negative_label else 0
        for row in table.rows
    ]
    if len(set(labels)) < 2:
        raise ValueError("real backend requires at least two target classes")

    # Correction option geo_target_encoding: geographic codes arrive as text only when it is on (the loader keeps
    # them as text); they are replaced by their smoothed training denial rate. Off, no encoder is fitted.
    encoded_columns = [
        name for name in GEO_CODE_COLUMNS
        if name in table.feature_names and any(isinstance(row.get(name), str) for row in feature_rows[:1000])
    ]
    code_encoder = _CodeTargetEncoder.fit(feature_rows, labels, encoded_columns) if encoded_columns else None

    vectorizer = DictVectorizer(sparse=False, sort=False)
    matrix = vectorizer.fit_transform(feature_rows)
    vectorizer.feature_fill_values_ = feature_fill_values
    vectorizer.code_encoder_ = code_encoder
    encoded_feature_names = list(vectorizer.get_feature_names_out())
    background_rows = min(len(feature_rows), 64)
    background_matrix = matrix[:background_rows]
    return vectorizer, matrix, encoded_feature_names, labels, background_matrix


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _logit(probability: float) -> float:
    clipped = min(max(probability, 1e-6), 1.0 - 1e-6)
    return float(math.log(clipped / (1.0 - clipped)))


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return False


class _CodeTargetEncoder:
    """Smoothed target encoding of categorical codes (correction option geo_target_encoding).

    Each code becomes (denials + m * prior) / (rows + m) over the training rows with that code, with m = 20 and the
    prior the training denial rate. The training rows themselves get out-of-fold values from 5 folds (seeded), so no
    row's own label enters its value; every other row (evaluation rows, perturbed rows) gets the full-training value,
    and a code unseen in training gets the prior."""

    SMOOTHING = 20.0
    FOLDS = 5
    SEED = 42

    def __init__(self, columns: list[str], prior: float, mapping: dict[str, dict[Any, float]]) -> None:
        self.columns = list(columns)
        self.prior = prior
        self.mapping = mapping

    @staticmethod
    def _rate(denials: float, rows: float, prior: float) -> float:
        return (denials + _CodeTargetEncoder.SMOOTHING * prior) / (rows + _CodeTargetEncoder.SMOOTHING)

    @classmethod
    def fit(cls, feature_rows: list[dict[str, Any]], labels: list[int], columns: list[str]) -> "_CodeTargetEncoder":
        """Fit on the training rows and replace their codes in place with out-of-fold values."""
        import numpy as np

        n = len(feature_rows)
        total_denials = float(sum(labels))
        prior = total_denials / n if n else 0.0
        folds = np.random.default_rng(cls.SEED).integers(0, cls.FOLDS, n).tolist()
        mapping: dict[str, dict[Any, float]] = {}
        for column in columns:
            totals: dict[Any, list[float]] = {}
            by_fold: dict[tuple[int, Any], list[float]] = {}
            fold_totals = [[0.0, 0.0] for _ in range(cls.FOLDS)]
            for row, label, fold in zip(feature_rows, labels, folds):
                code = row.get(column)
                total = totals.setdefault(code, [0.0, 0.0])
                total[0] += label
                total[1] += 1.0
                part = by_fold.setdefault((fold, code), [0.0, 0.0])
                part[0] += label
                part[1] += 1.0
                fold_totals[fold][0] += label
                fold_totals[fold][1] += 1.0
            mapping[column] = {code: cls._rate(d, r, prior) for code, (d, r) in totals.items()}
            # Prior of the rows outside each fold, so a training row's value uses no label from its own fold.
            out_priors = [
                (total_denials - d) / (n - r) if n > r else prior for d, r in fold_totals
            ]
            for row, fold in zip(feature_rows, folds):
                code = row.get(column)
                own = by_fold[(fold, code)]
                row[column] = cls._rate(totals[code][0] - own[0], totals[code][1] - own[1], out_priors[fold])
        return cls(columns, prior, mapping)

    def transform(self, values: dict[str, Any]) -> dict[str, Any]:
        out = dict(values)
        for column in self.columns:
            if column in out and not isinstance(out[column], float):
                out[column] = self.mapping[column].get(out[column], self.prior)
        return out


def _infer_feature_fill_values(
    rows: list[dict[str, Any]],
    feature_names: list[str],
) -> dict[str, Any]:
    fill_values: dict[str, Any] = {}
    for feature_name in feature_names:
        observed = [
            row.get(feature_name)
            for row in rows
            if not _is_missing(row.get(feature_name))
        ]
        if not observed:
            fill_values[feature_name] = 0.0
            continue
        if all(isinstance(value, (int, float, bool)) for value in observed):
            numeric_values = [float(value) for value in observed]
            fill_values[feature_name] = float(statistics.median(numeric_values))
            continue
        fill_values[feature_name] = "__MISSING__"
    return fill_values


def _normalize_feature_value(value: Any, fill_value: Any) -> Any:
    if _is_missing(value):
        return fill_value
    return value


def _sanitize_tabular_dataset(
    table: TabularDataset,
    feature_names: list[str],
    fill_values: dict[str, Any],
) -> TabularDataset:
    sanitized_rows: list[dict[str, Any]] = []
    for row in table.rows:
        sanitized_row = dict(row)
        for feature_name in feature_names:
            sanitized_row[feature_name] = _normalize_feature_value(
                row.get(feature_name),
                fill_values[feature_name],
            )
        sanitized_rows.append(sanitized_row)
    return TabularDataset(
        dataset_id=table.dataset_id,
        display_name=table.display_name,
        track=table.track,
        target=table.target,
        feature_names=table.feature_names,
        rows=sanitized_rows,
        source_kind=table.source_kind,
        source_path=table.source_path,
    )
