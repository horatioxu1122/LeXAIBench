# LegalBench-XAI

This directory contains the LegalBench-XAI benchmark package: a cross-domain
benchmark for legally meaningful explanation quality across consumer credit,
mortgage lending, pretrial release, and regulatory enforcement decisions. The
manuscript is in [`../manuscript/latex manuscript/`](../manuscript/latex%20manuscript/);
this README covers installation, the released artifact bundles, and the commands
that reproduce the headline results.

## What this package provides

- A Python package (`legalbench_xai`) implementing the benchmark pipeline:
  dataset registry, feature taxonomy, model and explainer backends,
  canonicalization layer, and metric scoring with hierarchical-bootstrap
  uncertainty.
- Real backends for `logistic_regression`, `ebm`, and `xgboost`, and for `shap`,
  `lime`, anchor-style rule search, DiCE-style and Wachter-style counterfactual
  recourse search.
- A canonical explanation schema that normalizes attribution, rule, and
  recourse outputs into a shared evaluation target.
- A six-dimensional scoring suite covering faithfulness, stability,
  comprehensibility, contestability support, actionability, and the
  actionability-legality filter.
- Runtime configs for the headline results, dataset preparation commands,
  and a CLI (`legalbench-xai`) wrapping the full pipeline.

## Installation

From the repository root:

```sh
uv sync --extra benchmark-empirical
```

This installs the empirical extras (`numpy`, `pandas`, `scikit-learn`,
`xgboost`, `shap`, `lime`, `interpret`). Alternatively:

```sh
python -m pip install -e .[benchmark-empirical]
```

Run the environment check:

```sh
uv run legalbench-xai doctor --config benchmark/configs/experiment_default.json
```

## Headline result reproduction

The manuscript draws from nine held-out output bundles. Each is reproduced by
running the package against its config:

| Bundle | Config | Role |
|---|---|---|
| `four_dataset_real_highbudget_heldout_refresh` | `experiment_four_dataset_real_highbudget.json` | Credit-subpanel headline (12 adverse cases per group) |
| `cross_domain_strongest_real_full_cluster` | `experiment_cross_domain_strongest_real_template.json` | Non-credit subpanel headline (HMDA, NY Courts pretrial, FDA inspections) |
| `four_dataset_real_heldout_refresh` | `experiment_four_dataset_real.json` | Lower-budget credit run (8 adverse cases per group) |
| `four_dataset_real_targetrate010_heldout` | `experiment_four_dataset_real_targetrate010.json` | Matched 0.10 predicted-adverse rate per group |
| `four_dataset_real_six_explainers_heldout` | `experiment_four_dataset_real_six_explainers.json` | Six-explainer breadth check (adds leave-one-feature-out) |
| `default_credit_real_highbudget24_heldout` | `experiment_default_credit_real_highbudget24.json` | 24-case Default Credit budget extension |
| `default_credit_real_threshold040_heldout` | `experiment_default_credit_real_threshold040.json` | Decision threshold 0.40 |
| `default_credit_real_threshold050_heldout` | `experiment_default_credit_real_threshold050.json` | Decision threshold 0.50 |
| `default_credit_real_threshold060_heldout` | `experiment_default_credit_real_threshold060.json` | Decision threshold 0.60 |

Reproduce a credit-subpanel bundle with:

```sh
uv run legalbench-xai run-demo-benchmark \
  --config benchmark/configs/experiment_four_dataset_real_highbudget.json \
  --write --summary-only
```

The cross-domain bundle is large enough that we ran it on a department cluster.
The cluster driver is in [`scripts/cluster/`](scripts/cluster/); it expects the
prepared HMDA, NY Courts pretrial, and FDA inspection files to be present (see
[Dataset preparation](#dataset-preparation) below).

## Dataset preparation

Each dataset has a CLI prep command that produces a benchmark-facing CSV from
the public source. Credit datasets pull from OpenML / UCI:

```sh
uv run legalbench-xai prepare-south-german-credit
uv run legalbench-xai prepare-default-credit
uv run legalbench-xai prepare-heloc
uv run legalbench-xai prepare-give-me-some-credit
```

Cross-domain datasets expect a local raw extract (acquisition instructions are
in [`DATA_ACQUISITION.md`](DATA_ACQUISITION.md)):

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

Real local paths can also be supplied via environment variables; see
[`EMPIRICAL_SETUP.md`](EMPIRICAL_SETUP.md).

## Released artifacts

`run-demo-benchmark --write` populates the bundle directory with:

- `results.json` — raw scored explanations, metric components, evidence
  sources, and metadata at the dataset/track/model/explainer/instance level.
- `summary.csv` — one row per dataset-track-model-explainer group.
- `analysis.json` — aggregated analysis with hierarchical-bootstrap CIs,
  pairwise wins, metric correlations, and definition-perturbation profiles.
- `by_explainer.csv`, `by_dataset_track_explainer.csv`, `by_model_explainer.csv` —
  aggregate tables at three granularities.
- `pairwise_wins.csv`, `metric_rank_correlation.csv`,
  `group_metric_correlation.csv` — robustness-supporting tables.
- `model_performance.csv` — held-out predictive performance (accuracy,
  ROC-AUC, average precision, Brier score, ECE) per dataset/track/model.
- `paper_kit/` — generated memos and LaTeX-ready tables for the manuscript.
- `figures/` — Vega-Lite figure specifications and a manifest.
- `report.md`, `leaderboard.tex` — human-readable summaries.

For artifacts already on disk, regenerate the paper-facing tables with
`rebuild-artifacts`:

```sh
uv run legalbench-xai rebuild-artifacts \
  --config benchmark/configs/experiment_four_dataset_real_highbudget.json \
  --results-json benchmark/outputs/four_dataset_real_highbudget_heldout_refresh/results.json \
  --output-dir benchmark/outputs/four_dataset_real_highbudget_heldout_refresh
```

## Inspection commands

```sh
# Validate that a config is wired up:
uv run legalbench-xai validate \
  --config benchmark/configs/experiment_four_dataset_real_highbudget.json

# Describe a dataset's expected schema:
uv run legalbench-xai describe-dataset --dataset heloc --track primary

# Preview rows after track filtering:
uv run legalbench-xai preview-dataset --dataset heloc --track primary

# Check whether a local CSV matches the registry schema:
uv run legalbench-xai check-dataset-file \
  --dataset heloc --track primary \
  --path benchmark/data/real/heloc.csv

# Inspect a single canonicalized explanation:
uv run legalbench-xai normalize-example --dataset heloc --track primary

# Score a single canonicalized explanation:
uv run legalbench-xai score-example --dataset heloc --track primary

# End-to-end adapter + scoring on one example:
uv run legalbench-xai score-adapted-example \
  --dataset south_german_credit --track primary --explainer dice
```

## Layout

```text
benchmark/
  configs/        runtime experiment configs (10 essential)
  data/
    real/         prepared benchmark CSVs
    demo/         tiny CSV fixtures used by CI / demo mode
  metadata/       dataset registry and feature taxonomy manifests
  outputs/        released held-out result bundles (9 essential)
  scripts/        run-legalbench wrapper, cluster driver, prep helpers
  src/
    legalbench_xai/   the Python package
```

For acquisition instructions for the cross-domain datasets, see
[`DATA_ACQUISITION.md`](DATA_ACQUISITION.md). For empirical-runtime details and
dependency notes, see [`EMPIRICAL_SETUP.md`](EMPIRICAL_SETUP.md).
