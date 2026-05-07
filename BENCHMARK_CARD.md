# LegalBench-XAI Benchmark Card

## Benchmark Name

`LegalBench-XAI`

## Primary Goal

Evaluate whether explanation methods produce explanations that are legally
meaningful for high-stakes algorithmic decisions, not only whether they align
with model behavior.

## Task Scope

- Domains: consumer credit, mortgage lending, pretrial release, regulatory
  enforcement (four legal-decision domains)
- Benchmark unit: dataset-track-model-explainer configuration
- Headline credit-subset bundle: `four_dataset_real_highbudget_heldout_refresh`
- Headline non-credit-subset bundle: `cross_domain_strongest_real_full_cluster`
- Total scored explanations: 2,520 across 210 dataset-track-model-explainer
  configurations on held-out 70/30 evaluation slices

## Release Status

- The released bundles use held-out 70/30 evaluation slices with split seed 42
- Bundles surface held-out model performance, asset-license / terms information,
  explicit compute disclosure, direct-evidence proxy ablations, subpopulation
  summaries (credit subset), and a planned human-validation protocol
- Auxiliary credit-subset robustness bundles cover threshold sensitivity
  (0.4 / 0.5 / 0.6), a matched predicted-adverse-rate policy (0.10), a
  12 -> 24 denied-instance budget extension, and a sixth direct-evidence
  attribution baseline (leave-one-feature-out)

## Reference Hardware

- OS: Windows 11
- CPU: 12th Gen Intel Core i7-12700H (2.30 GHz)
- GPU: NVIDIA GeForce RTX 3060 (unused by the current CPU-only runtime)
- Memory: 16 GB RAM

The non-credit subset bundle was produced on a department compute cluster
because of HMDA scale (8.84M training rows). The cluster driver and
runtime-config builder are in `benchmark/scripts/`.

## Observed Wall-Clock Guidance

- Held-out four-dataset high-budget refresh: ~66 minutes
- Held-out four-dataset lower-budget refresh: ~39 minutes
- Held-out four-dataset matched-rate sweep: ~54 minutes
- Held-out four-dataset six-explainer sweep: ~74 minutes
- Single-dataset auxiliary slices: ~20-25 minutes

## Included Datasets

- `heloc` (HELOC, OpenML cleaned mirror)
- `default_of_credit_card_clients` (UCI)
- `give_me_some_credit` (OpenML)
- `south_german_credit` (UCI)
- `hmda` (HMDA 2024 LAR, CFPB)
- `ny_courts_pretrial` (NY Courts pretrial release)
- `fda_inspections` (FDA inspection classifications)

## Included Model Classes

- `logistic_regression`
- `ebm` (Explainable Boosting Machine, via `interpret`)
- `xgboost`

## Included Explanation Methods

- `shap` (attribution)
- `lime` (attribution)
- `anchors` (rule)
- `dice` (recourse)
- `wachter` (recourse)
- auxiliary breadth baseline: `leave_one_out` (attribution, six-explainer
  bundle only)

## Evaluation Dimensions

- `faithfulness`
- `stability`
- `comprehensibility`
- `css` (Contestability Support Score)
- `actionability` (Unified Actionability)
- `alf` (Actionability-Legality Filter)

## Distinctive Design Choices

- Feature-taxonomy-aware evaluation: features are typed by mutability,
  verifiability, correctability, and sensitivity
- Two governance tracks: `primary` and `audit`, distinguishing variables a
  lender can use versus those a regulator can audit
- Subject-facing metrics for contestability support and legality-aware recourse
- Hierarchical-bootstrap uncertainty over the
  dataset -> track -> model structure with 1,000 resamples by default
- Explicit robustness tests on the benchmark itself (eight checks across
  governance, metric weighting, definition perturbations, budget, matched
  operating point, leave-one-dataset-out, leave-one-metric-out, and decision
  threshold)

## Reproduction Entry Points

```sh
# Headline credit-subset bundle
uv run legalbench-xai run-demo-benchmark \
  --config benchmark/configs/experiment_four_dataset_real_highbudget.json \
  --write --summary-only

# Headline non-credit-subset bundle (uses the cluster driver)
bash benchmark/scripts/run-cross-domain-strongest-real.sh \
  --output-name cross_domain_strongest_real_full_cluster

# Regenerate paper-facing tables and figures from an existing results.json
uv run legalbench-xai rebuild-artifacts \
  --config benchmark/configs/experiment_four_dataset_real_highbudget.json \
  --results-json benchmark/outputs/four_dataset_real_highbudget_heldout_refresh/results.json \
  --output-dir benchmark/outputs/four_dataset_real_highbudget_heldout_refresh
```

Dataset preparation commands and per-dataset acquisition instructions are in
[`benchmark/README.md`](benchmark/README.md) and
[`benchmark/DATA_ACQUISITION.md`](benchmark/DATA_ACQUISITION.md).

## Intended Uses

- Comparing explanation methods across high-stakes legal-decision settings
- Stress-testing whether benchmark conclusions survive reasonable perturbations
- Studying tradeoffs between model-facing reliability and subject-facing
  usefulness of explanations
- Supporting benchmark methodology research for high-stakes XAI

## Out-of-Scope Uses

- Claiming legal compliance for any deployed system
- Claiming deployment readiness for real lenders, courts, or regulators
- Generalizing directly to all high-stakes domains
- Replacing human-subject validation or doctrinal analysis

## Main Empirical Message

The benchmark does not yield a universal winner. In the credit-subset headline
bundle:

- `shap` leads faithfulness
- `lime` leads stability
- `anchors` leads comprehensibility
- `dice` leads contestability support (CSS)
- `wachter` leads actionability and ALF

The same method-category pattern (attribution / rule lead model-facing
reliability; recourse leads subject-facing usefulness) reappears in the
non-credit subset. No single method is best across all six dimensions.

## Known Limitations

- Public datasets rather than deployed lender, court, or regulator systems
- Computational operationalization rather than behavioral validation
- Subpopulation analysis is reported only for the credit subset
- Some robustness sensitivity remains strongest on stability and on
  recourse-conditioned metrics

## Data Access and Redistribution

- Raw data should be obtained from the original public sources or mirrors
  (see `benchmark/DATA_ACQUISITION.md` and `LICENSE` for source URLs and
  per-dataset terms)
- The release ships preparation scripts, metadata, configs, and derived
  results rather than redistributing raw data wholesale
- Public benchmark rows should not be used to re-identify individuals or
  imply lender-specific or regulator-specific deployment realism

## Recommended Claim Discipline

Prefer:

- "legally meaningful"
- "subject-facing explanation quality"
- "benchmark operationalization"
- "method-category tradeoff"

Avoid:

- "legally compliant explanations"
- "best explainer overall"
- "general benchmark for all high-stakes AI"

## Responsible Release Guardrails

- Do not treat benchmark scores as compliance certificates
- Do not treat counterfactual suggestions as personalized financial,
  pretrial, or regulatory advice
- Use the `primary` and `audit` tracks to surface governance sensitivity
  rather than hide it
- Report backend implementation details honestly when a method is
  represented by a search-style approximation instead of a canonical
  library API
