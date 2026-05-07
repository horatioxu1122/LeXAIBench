# LeXAIBench Empirical Runtime Setup

This document describes the dependency stack and the operational setup needed
to reproduce the headline results. For the manuscript-facing reproduction
commands and the bundle ↔ config mapping, see [`README.md`](README.md). For
acquisition instructions for the cross-domain datasets, see
[`DATA_ACQUISITION.md`](DATA_ACQUISITION.md).

## What this enables

With the empirical dependencies installed, the package activates:

- real `scikit-learn`, `interpret` (EBM), and `xgboost` model backends;
- real `shap` and `lime` explainer backends;
- model-validated anchor-style rule search;
- model-validated DiCE-style and Wachter-style counterfactual recourse search;
- the held-out evaluation protocol used by the manuscript bundles.

Without the empirical extras, the package still loads and the CLI still works,
but model and explainer steps fall back to lightweight heuristic stubs that are
not used by the released bundles.

## Recommended install path

From the repository root:

```sh
uv sync --extra benchmark-empirical
```

The benchmark uses Python 3.13 by default. If you prefer an editable install
without `uv`:

```sh
python -m pip install -e .[benchmark-empirical]
```

The `legalbench-xai` console entry point is the canonical interface; both the
following invocations work identically:

```sh
uv run legalbench-xai doctor --config benchmark/configs/experiment_default.json
uv run python -m legalbench_xai doctor --config benchmark/configs/experiment_default.json
```

## Dependency set

The `benchmark-empirical` extra in [`pyproject.toml`](../pyproject.toml)
includes:

- `numpy`
- `pandas`
- `xlrd` (for the Default of Credit Card Clients raw `.xls`)
- `scikit-learn`
- `xgboost`
- `shap`
- `lime`
- `interpret` (Explainable Boosting Machines)

The minimum subset that activates the credit-subpanel pipeline is
`numpy`, `pandas`, `xlrd`, `scikit-learn`, `shap`. Add `xgboost` to enable the
gradient-boosted-trees track and `interpret` to enable EBM.

## Environment check

After installation:

```sh
uv run legalbench-xai doctor --config benchmark/configs/experiment_default.json
```

This reports:

- whether demo mode is ready;
- whether the empirical pipeline is ready, or which packages are still missing;
- whether each dataset is currently resolving to a demo fixture or to a local
  override.

For the raw capability matrix:

```sh
uv run legalbench-xai show-capabilities \
  --config benchmark/configs/experiment_default.json
```

For the per-dataset source resolution:

```sh
uv run legalbench-xai show-data-sources \
  --config benchmark/configs/experiment_default.json
```

## Switching to real dataset files

Real dataset paths can be supplied either through a config that overrides the
dataset registry (for the headline runs the relevant config is
[`experiment_cross_domain_strongest_real_template.json`](configs/experiment_cross_domain_strongest_real_template.json)),
or through environment variables:

```text
LEGALBENCH_XAI_DATASET_HELOC_PATH
LEGALBENCH_XAI_DATASET_DEFAULT_OF_CREDIT_CARD_CLIENTS_PATH
LEGALBENCH_XAI_DATASET_SOUTH_GERMAN_CREDIT_PATH
LEGALBENCH_XAI_DATASET_GIVE_ME_SOME_CREDIT_PATH
LEGALBENCH_XAI_DATASET_HMDA_PATH
LEGALBENCH_XAI_DATASET_NY_COURTS_PRETRIAL_PATH
LEGALBENCH_XAI_DATASET_FDA_INSPECTIONS_PATH
```

When an override is active, `describe-dataset` and `validate` report the
source as `local_override`. Before running a bundle on a real file, validate
the schema:

```sh
uv run legalbench-xai check-dataset-file \
  --dataset heloc --track primary --path benchmark/data/real/heloc.csv
```

`check-dataset-file` reports whether the target column is present, whether
required track columns are missing, whether known-but-excluded columns are
present, and whether the file has any extra unexpected columns.

## Preparing the benchmark CSVs

Each dataset has a prep command that produces a benchmark-facing CSV from the
public source. For the credit subpanel:

```sh
uv run legalbench-xai prepare-south-german-credit
uv run legalbench-xai prepare-default-credit
uv run legalbench-xai prepare-heloc
uv run legalbench-xai prepare-give-me-some-credit
```

`prepare-heloc` and `prepare-give-me-some-credit` pull from OpenML mirrors
(`data_id=45554` for HELOC, `46929` for Give Me Some Credit). The South German
Credit and Default of Credit Card Clients commands expect raw UCI archives at
`benchmark/data/raw/south_german_credit_uci/SouthGermanCredit.asc` and
`benchmark/data/raw/default_credit_uci/default of credit card clients.xls`
respectively.

For the cross-domain panel, the prep commands expect local raw extracts (see
[`DATA_ACQUISITION.md`](DATA_ACQUISITION.md) for source URLs and per-dataset
field expectations):

```sh
uv run legalbench-xai prepare-hmda \
  --input-csv benchmark/data/raw/hmda/hmda_snapshot.csv \
  --output-csv benchmark/data/real/hmda.csv

uv run legalbench-xai prepare-ny-courts-pretrial \
  --input-csv benchmark/data/raw/ny_courts_pretrial/pretrial_release_rows.csv \
  --output-csv benchmark/data/real/ny_courts_pretrial.csv

uv run legalbench-xai prepare-fda-inspections \
  --input-csv benchmark/data/raw/fda_inspections/fda_inspections_extract.csv \
  --output-csv benchmark/data/real/fda_inspections.csv
```

For HMDA, the prep step is intentionally benchmark-facing rather than
full-download automation: it expects a CSV containing the key benchmark
fields (action_taken, loan_amount, loan_type, loan_purpose, occupancy_type,
derived_dwelling_category, lien_status, total_units, income,
debt_to_income_ratio, combined_loan_to_value_ratio, applicant_age,
co_applicant_age, derived_race, derived_ethnicity, derived_sex, state_code,
county_code, census_tract). The step filters to a clean approved-vs-denied
slice and writes the benchmark-facing schema.

For NY Courts pretrial, the prep step expects a CSV extract with the official
release-decision and arraignment-context fields and filters to a clean
released-vs-detained slice (ROR and Nonmonetary release become `released`;
Remanded and Bail-set become `detained`; Disposed at Arraign and Unknown are
skipped).

For FDA inspections, the prep step expects a CSV export from the FDA
inspection-classification dashboard and collapses VAI and OAI into
`action_indicated`, keeping NAI as `no_action_indicated`. The output is one
row per project area for entity-facing remediation benchmarking.

## Reproduction protocol

Once the prepared CSVs exist, the manuscript's nine bundles are reproduced by
running the package against the matching config (the `Headline result
reproduction` table in [`README.md`](README.md) gives the full mapping). The
canonical credit-subpanel command is:

```sh
uv run legalbench-xai run-demo-benchmark \
  --config benchmark/configs/experiment_four_dataset_real_highbudget.json \
  --write --summary-only
```

The cross-domain bundle expects more compute (HMDA alone has 8.84M training
rows). The cluster driver in [`scripts/cluster/`](scripts/cluster/) was used
to produce the released `cross_domain_strongest_real_full_cluster` bundle.

## Known runtime characteristics

- Adverse-instance budget is set per-config; the headline credit run uses 12
  cases per dataset-track-model group, producing 1{,}440 scored explanations.
  The non-credit subpanel uses 12 cases per group, producing 1{,}080 scored
  explanations.
- Stability uses a perturbation probe with up to three same-label nearest
  neighbors per instance, numeric jitter 0.05, categorical-flip probability
  0.15, and three LIME seed trials when the explainer is stochastic.
- Aggregate summaries are reported with 95% hierarchical-bootstrap confidence
  intervals over the dataset → track → model structure, 1{,}000 resamples by
  default.
- The released runs were produced on Python 3.13 with the `benchmark-empirical`
  extras. CPU-only execution is sufficient for the credit subpanel; the
  non-credit subpanel was run on a department cluster because of HMDA scale.
