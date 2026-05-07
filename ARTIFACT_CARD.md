# LeXAIBench Artifact Card

## Purpose

This artifact packages the benchmark implementation, prepared benchmark CSVs,
released held-out result bundles, and reproduction tooling for the
LeXAIBench submission.

## What The Artifact Contains

- benchmark configs in [`benchmark/configs/`](benchmark/configs/)
- the LeXAIBench Python package in
  [`benchmark/src/legalbench_xai/`](benchmark/src/legalbench_xai/), implementing
  dataset preparation, model and explainer backends, the canonical-explanation
  schema, the six-dimensional scoring suite, hierarchical-bootstrap reporting,
  and the paper-kit / figure-pack generators
- prepared benchmark CSVs for the four credit datasets in
  [`benchmark/data/real/`](benchmark/data/real/) (HELOC, Default of Credit Card
  Clients, Give Me Some Credit, South German Credit)
- demo CSV fixtures used by the doctor / validate path in
  [`benchmark/data/demo/`](benchmark/data/demo/)
- nine released held-out result bundles in
  [`benchmark/outputs/`](benchmark/outputs/)
- the canonical explanation schema, feature taxonomy manifest, and metric
  scoring schema YAMLs in [`paper/`](paper/)
- reproduction shell drivers and the cluster driver in
  [`benchmark/scripts/`](benchmark/scripts/)
- the manuscript-figure renderer in
  [`scripts/render_paper_figures.py`](scripts/render_paper_figures.py)

## Primary Entry Points

```sh
# environment check
uv run legalbench-xai doctor --config benchmark/configs/experiment_default.json

# config validation
uv run legalbench-xai validate --config benchmark/configs/experiment_default.json

# dataset preparation
uv run legalbench-xai prepare-south-german-credit
uv run legalbench-xai prepare-default-credit
uv run legalbench-xai prepare-heloc
uv run legalbench-xai prepare-give-me-some-credit
uv run legalbench-xai prepare-hmda \
  --input-csv benchmark/data/raw/hmda/hmda_snapshot.csv \
  --output-csv benchmark/data/real/hmda.csv
uv run legalbench-xai prepare-ny-courts-pretrial \
  --input-csv benchmark/data/raw/ny_courts_pretrial/pretrial_release_rows.csv \
  --output-csv benchmark/data/real/ny_courts_pretrial.csv
uv run legalbench-xai prepare-fda-inspections \
  --input-csv benchmark/data/raw/fda_inspections/fda_inspections_extract.csv \
  --output-csv benchmark/data/real/fda_inspections.csv

# headline credit-subset bundle
uv run legalbench-xai run-demo-benchmark \
  --config benchmark/configs/experiment_four_dataset_real_highbudget.json \
  --write --summary-only

# rebuild paper-facing tables and figures from an existing results.json
uv run legalbench-xai rebuild-artifacts \
  --config benchmark/configs/experiment_four_dataset_real_highbudget.json \
  --results-json benchmark/outputs/four_dataset_real_highbudget_heldout_refresh/results.json \
  --output-dir benchmark/outputs/four_dataset_real_highbudget_heldout_refresh
```

## Released Bundles

- `four_dataset_real_highbudget_heldout_refresh` — credit-subset headline
  (12 adverse instances per configuration)
- `cross_domain_strongest_real_full_cluster` — non-credit-subset headline
  (HMDA, NY Courts pretrial, FDA inspections; produced on a department
  compute cluster)
- `four_dataset_real_heldout_refresh` — lower-budget credit run
  (8 adverse instances per configuration), used for the budget sweep
- `four_dataset_real_targetrate010_heldout` — matched 0.10 predicted-adverse
  rate per configuration
- `four_dataset_real_six_explainers_heldout` — six-explainer breadth check
  including leave-one-feature-out
- `default_credit_real_highbudget24_heldout` — 12-to-24 budget extension
  on a held-out Default Credit slice
- `default_credit_real_threshold040_heldout`,
  `default_credit_real_threshold050_heldout`,
  `default_credit_real_threshold060_heldout` — decision-threshold sweep
  on the same held-out Default Credit slice

Each bundle contains: `results.json`, `summary.csv`, `analysis.json`,
`by_explainer.csv`, `by_dataset_track_explainer.csv`, `by_model_explainer.csv`,
`pairwise_wins.csv`, `model_performance.csv`, `metric_rank_correlation.csv`,
`group_metric_correlation.csv` (where applicable),
`metric_weight_profiles.csv`, `metric_weight_random_summary.csv`,
`metric_ablation_profiles.csv`, `css_definition_profiles.csv`,
`alf_definition_profiles.csv`, `track_policy_*.csv`,
`leave_one_dataset_out{,_leaders}.csv`, `report.md`, `leaderboard.tex`,
a `paper_kit/` of LaTeX tables and prose memos, and a `figures/` pack of
Vega-Lite specs with a manifest. The credit-subset and cross-domain headline
bundles also include `direct_evidence_*.csv`, `subgroup_summary.csv`,
`subgroup_spread_summary.csv`, `evidence_source_summary.csv`, and
`proxy_sensitivity_profiles.csv`. The cross-domain bundle additionally
contains a `progress.jsonl` runtime log and a `runtime_config.json` snapshot
of the cluster run.

## Reproduction Environment

- OS: Windows 11 (laptop) and Linux (cluster, for the non-credit subset)
- CPU: 12th Gen Intel Core i7-12700H (laptop, CPU-only)
- GPU: NVIDIA GeForce RTX 3060 (unused by the current runtime)
- Memory: 16 GB RAM (laptop)
- Runtime: Python 3.12+ with `numpy`, `pandas`, `scikit-learn`, `xgboost`,
  `shap`, `lime`, `interpret`, and `xlrd`

## Observed Rerun Times

- Held-out four-dataset high-budget refresh: ~66 minutes
- Held-out four-dataset lower-budget refresh: ~39 minutes
- Held-out four-dataset matched-rate sweep: ~54 minutes
- Held-out four-dataset six-explainer sweep: ~74 minutes
- Single-dataset auxiliary slices: ~20-25 minutes
- Cross-domain non-credit subset bundle: hours, on a department cluster

## Data Access Policy

- Acquire raw data from the original public sources or mirrors cited in
  the paper and in [`benchmark/DATA_ACQUISITION.md`](benchmark/DATA_ACQUISITION.md)
- The artifact ships preparation scripts, metadata, configs, and derived
  artifacts rather than redistributing raw data wholesale
- The four credit datasets are pulled by the `prepare-*-credit` commands
  (UCI, OpenML); the three cross-domain datasets (HMDA, NY Courts, FDA)
  expect a local raw extract, with per-dataset field expectations in
  `benchmark/DATA_ACQUISITION.md`
- Example explanations in the released bundles are tied to benchmark rows
  only; do not use the artifact to infer identity or deployment-specific
  decisions

## Responsible Use

- Benchmark scores are not legal-compliance guarantees
- Counterfactual outputs are benchmark objects, not personalized financial,
  pretrial, or regulatory advice
- Track-sensitive results should be reported with the `primary` / `audit`
  distinction intact
- Backend implementations should be named honestly when a method is
  represented by a search-style approximation instead of a canonical
  library API
