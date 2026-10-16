"""Versioned, opt-in corrections to the benchmark (LeXAIBench v2).

Every correction defaults to the v1 behavior, so configs without a
``cslaw.corrections`` entry reproduce the published (v1) bundles exactly. The
v2 preset turns on all corrections used for the CS&Law revision:

native_contributions
    Rank candidate features (Anchors candidates, counterfactual search order,
    DiCE/Wachter feature_importances) with the fitted model's own additive
    contributions (LR: coef * standardized x; EBM: eval_terms, pair terms split
    equally; XGBoost: pred_contribs). v1 used a taxonomy-weighted heuristic proxy
    for EBM and XGBoost (models/real.py original_feature_contributions).
common_faithfulness_reference
    Score every explainer's faithfulness against the same reference: the model's
    native per-feature contributions over all included features. v1 scored SHAP,
    LIME and leave-one-out against their own outputs and Anchors against the
    heuristic proxy restricted to its own anchor features.
faithfulness_quality_weight
    Weight of each explainer's self-reported quality signal inside faithfulness.
    v1 = 0.20 (SHAP's signal was a constant per model type); v2 = 0.0, with the
    remaining coverage/direction weights renormalized. The signal is still recorded.
counterfactual_reasons_from_changes
    DiCE/Wachter principal reasons come from the features their counterfactuals
    actually change (frequency across returned counterfactuals, ties broken by
    native contribution), instead of a borrowed attribution ranking.
cf_respect_nonactionable
    Treat taxonomy mutability "N" (not actionable by the applicant) like "I" in
    the counterfactual search and in hard-failure checks.
reason_grouping_fix
    Fix the prefix-order bug in normalization/rules.py that grouped PAY_AMT*
    (payment amounts) under "recent late payment history".
numeric_parsing_fix
    Parse exponent-notation numbers (e.g. "8e-05") as floats when loading CSVs.
sampling_selection / sampling_seed
    v1 "boundary_nearest" takes, within each decision-margin stratum, the rows
    closest to the stratum boundary (scores cluster near 0.50 / 0.60 / 0.75).
    v2 "stratified_random" draws uniformly at random within each stratum, seeded
    per dataset-track-model (independent of the explainer list).
persist_score_subcomponents
    Record sub-components that v1 folded together (e.g. the three parts of CSS
    decision_sensitivity_visibility) so family-neutral variants can be computed.
record_data_provenance
    Record SHA-256 hashes of the dataset and metadata files used by the run.

Robustness options outside the v2 preset (off in both presets; switch on with
``{"preset": "v2", "<option>": true}``):

lime_sample_around_instance
    Pass ``sample_around_instance=True`` to LimeTabularExplainer, so LIME draws its
    perturbations around the explained applicant instead of around the training
    mean (models/real.py get_lime_explainer). Used for one LIME-only robustness
    re-run (experiment_cslaw_v2_credit_lime_local.json). ``to_dict()`` records the
    option only when it is on, so bundles of runs without it are unchanged.
lime_discretize_continuous
    Pass ``discretize_continuous=True`` (the library default, quartile bins) to
    LimeTabularExplainer instead of False, so each weight is the effect of the
    applicant's own bin rather than a slope. Used for one LIME-only robustness
    re-run (experiment_cslaw_v2_credit_lime_default.json); recorded only when on.
    lime 0.2.0.1 ignores sample_around_instance in this mode, so the two options
    cannot be combined.
geo_target_encoding
    Load the geographic codes in GEO_CODE_COLUMNS (HMDA county and census tract) as
    text and give the models, in their place, the smoothed denial rate of each code
    in the training data (out of fold for the training rows themselves), so a model
    sees a neighborhood's outcome history instead of the order of its numeric code
    (models/real.py _CodeTargetEncoder). Used for one HMDA re-run
    (experiment_cslaw_v2_hmda_geo_*.json); recorded only when on.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from typing import Any

SAMPLING_SELECTIONS = ("boundary_nearest", "stratified_random")
# Options that to_dict() leaves out while they are False (robustness options added
# after the v2 bundles were specified).
_OMIT_WHEN_DEFAULT = ("lime_sample_around_instance", "lime_discretize_continuous", "geo_target_encoding")
# Columns that geo_target_encoding loads as text and target-encodes.
GEO_CODE_COLUMNS = ("county_code", "census_tract")


@dataclass(frozen=True, slots=True)
class CorrectionOptions:
    native_contributions: bool = False
    common_faithfulness_reference: bool = False
    faithfulness_quality_weight: float = 0.20
    counterfactual_reasons_from_changes: bool = False
    cf_respect_nonactionable: bool = False
    reason_grouping_fix: bool = False
    numeric_parsing_fix: bool = False
    sampling_selection: str = "boundary_nearest"
    sampling_seed: int = 42
    persist_score_subcomponents: bool = False
    record_data_provenance: bool = False
    # Robustness option, not part of any preset (see the module docstring).
    lime_sample_around_instance: bool = False
    lime_discretize_continuous: bool = False
    geo_target_encoding: bool = False

    def __post_init__(self) -> None:
        if self.lime_sample_around_instance and self.lime_discretize_continuous:
            raise ValueError(
                "cslaw.corrections: lime_sample_around_instance has no effect with lime_discretize_continuous "
                "in lime 0.2.0.1; switch on at most one"
            )
        if self.sampling_selection not in SAMPLING_SELECTIONS:
            raise ValueError(
                f"cslaw.corrections.sampling_selection must be one of {SAMPLING_SELECTIONS}, "
                f"got {self.sampling_selection!r}"
            )
        if not 0.0 <= float(self.faithfulness_quality_weight) <= 1.0:
            raise ValueError("cslaw.corrections.faithfulness_quality_weight must be in [0, 1]")

    @property
    def is_v1(self) -> bool:
        return self == V1

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        # Recorded only when on: the v1/v2 option dicts (and the bundles that echo
        # them, and cslaw/common.py's mirror of the presets) stay as before.
        for name in _OMIT_WHEN_DEFAULT:
            if data.get(name) is False:
                data.pop(name)
        return data


V1 = CorrectionOptions()
V2 = CorrectionOptions(
    native_contributions=True,
    common_faithfulness_reference=True,
    faithfulness_quality_weight=0.0,
    counterfactual_reasons_from_changes=True,
    cf_respect_nonactionable=True,
    reason_grouping_fix=True,
    numeric_parsing_fix=True,
    sampling_selection="stratified_random",
    sampling_seed=42,
    persist_score_subcomponents=True,
    record_data_provenance=True,
)
PRESETS = {"v1": V1, "v2": V2}
_FIELD_NAMES = {item.name for item in fields(CorrectionOptions)}


def corrections_from_payload(payload: dict[str, Any] | None) -> CorrectionOptions:
    """Build CorrectionOptions from an experiment config payload.

    Accepted forms of ``payload["cslaw"]["corrections"]``: absent (v1), a preset
    name ("v1" | "v2"), or an object ``{"preset": "v2", "<option>": value, ...}``
    whose explicit options override the preset. The legacy flag
    ``cslaw.use_native_contributions_for_ranking`` turns on native_contributions.
    """
    cslaw = (payload or {}).get("cslaw")
    if not isinstance(cslaw, dict):
        return V1
    raw = cslaw.get("corrections")
    if raw is None:
        options = V1
    elif isinstance(raw, str):
        if raw not in PRESETS:
            raise ValueError(f"unknown cslaw.corrections preset {raw!r}; use one of {sorted(PRESETS)}")
        options = PRESETS[raw]
    elif isinstance(raw, dict):
        preset_name = raw.get("preset", "v1")
        if preset_name not in PRESETS:
            raise ValueError(
                f"unknown cslaw.corrections.preset {preset_name!r}; use one of {sorted(PRESETS)}"
            )
        overrides = {key: value for key, value in raw.items() if key != "preset"}
        unknown = sorted(set(overrides) - _FIELD_NAMES)
        if unknown:
            raise ValueError(f"unknown cslaw.corrections option(s): {unknown}")
        options = replace(PRESETS[preset_name], **overrides)
    else:
        raise ValueError("cslaw.corrections must be a preset name or a JSON object")

    if cslaw.get("use_native_contributions_for_ranking") and not options.native_contributions:
        options = replace(options, native_contributions=True)
    return options


def corrections_from_config(config: Any) -> CorrectionOptions:
    return corrections_from_payload(getattr(config, "payload", None))


def model_corrections(model: Any) -> CorrectionOptions:
    """Corrections attached to a fitted model by the runner (v1 when absent)."""
    options = getattr(model, "corrections", None)
    return options if isinstance(options, CorrectionOptions) else V1
