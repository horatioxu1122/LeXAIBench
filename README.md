# LegalBench-XAI

A cross-domain benchmark for legally meaningful explanation quality in
high-stakes algorithmic decisions, spanning consumer credit, mortgage lending,
pretrial release, and regulatory enforcement.

This repository accompanies a paper under double-blind review at NeurIPS
Datasets and Benchmarks 2026.

## Repository layout

```text
.
|-- benchmark/        the LegalBench-XAI Python package, runtime configs,
|                     prepared benchmark CSVs, released result bundles,
|                     and reproduction scripts
|-- manuscript/       the LaTeX manuscript and accompanying figures
|-- paper/            the legal-XAI background paper from which the
|                     benchmark's evaluation criteria are derived,
|                     plus the canonical explanation schema, feature
|                     taxonomy, and metric scoring YAML manifests
|-- pyproject.toml    Python project metadata
|-- uv.lock           uv-resolved dependency lockfile
|-- LICENSE           MIT license (code) plus per-dataset notice
`-- README.md         this file
```

## Quick start

Install with the empirical extras and run the environment check:

```sh
uv sync --extra benchmark-empirical
uv run legalbench-xai doctor --config benchmark/configs/experiment_default.json
```

Reproduce a credit-subpanel headline bundle:

```sh
uv run legalbench-xai run-demo-benchmark \
  --config benchmark/configs/experiment_four_dataset_real_highbudget.json \
  --write --summary-only
```

Full installation, the bundle to config mapping, dataset preparation
commands, and inspection commands are in
[`benchmark/README.md`](benchmark/README.md). Cross-domain dataset
acquisition (HMDA, NY Courts pretrial release, FDA inspection
classifications) is documented in
[`benchmark/DATA_ACQUISITION.md`](benchmark/DATA_ACQUISITION.md).
Empirical-runtime details, the dependency stack, and known runtime
characteristics are in
[`benchmark/EMPIRICAL_SETUP.md`](benchmark/EMPIRICAL_SETUP.md).

## License

Code, configs, scripts, and documentation in this repository are released
under the MIT License; see [`LICENSE`](LICENSE). The underlying datasets
(HMDA, NY Courts pretrial release, FDA inspection classifications, UCI and
OpenML credit datasets) retain their original public-source licenses; see
[`LICENSE`](LICENSE) and
[`benchmark/DATA_ACQUISITION.md`](benchmark/DATA_ACQUISITION.md) for the
per-dataset notices and source URLs.
