# Cross-Domain Data Acquisition

This note specifies the official source for each non-credit panel dataset, the
local filename the repo expects, and the prep command that turns the raw file
into the benchmark-facing CSV. The four credit datasets are pulled
automatically by their `prepare-*` commands and do not require manual
acquisition; see [`README.md`](README.md) for those.

## Quick paths

| Dataset | Official source | Expected raw file | Prep command | Prepared output |
| --- | --- | --- | --- | --- |
| `hmda` | CFPB HMDA data portal | `benchmark/data/raw/hmda/hmda_snapshot.csv` | `prepare-hmda` | `benchmark/data/real/hmda.csv` |
| `ny_courts_pretrial` | NY Courts pretrial release data | `benchmark/data/raw/ny_courts_pretrial/pretrial_release_rows.csv` | `prepare-ny-courts-pretrial` | `benchmark/data/real/ny_courts_pretrial.csv` |
| `fda_inspections` | FDA inspection classification database export | `benchmark/data/raw/fda_inspections/fda_inspections_extract.csv` | `prepare-fda-inspections` | `benchmark/data/real/fda_inspections.csv` |

## HMDA

Official source:

- `https://www.consumerfinance.gov/data-research/hmda/`

Recommended first slice:

- Recent modified loan/application register export with the benchmark-facing
  fields already present.
- Keep only the columns needed by `prepare-hmda` when making the local working
  CSV (`action_taken`, `loan_amount`, `loan_type`, `loan_purpose`,
  `occupancy_type`, `derived_dwelling_category`, `lien_status`, `total_units`,
  `income`, `debt_to_income_ratio`, `combined_loan_to_value_ratio`,
  `applicant_age`, `co_applicant_age`, `derived_race`, `derived_ethnicity`,
  `derived_sex`, `state_code`, `county_code`, `census_tract`).

Repo expectation:

- Raw file: `benchmark/data/raw/hmda/hmda_snapshot.csv`
- Prepared file: `benchmark/data/real/hmda.csv`

Prep command:

```sh
uv run legalbench-xai prepare-hmda \
  --input-csv benchmark/data/raw/hmda/hmda_snapshot.csv \
  --output-csv benchmark/data/real/hmda.csv
```

## NY Courts Pretrial Release

Official source:

- `https://ww2.nycourts.gov/pretrial-release-data-33136`

Recommended first slice:

- State-paid courts CSV, not the town-and-village extract.
- Keep a local CSV with the official column names expected by the benchmark
  prep step, including `Release Decision at Arraign`, `Gender`, `Race`,
  `Ethnicity`, `Age at Arrest`, `County Name`, `Court Type`, `Arrest Type`,
  `Top Charge at Arraign`, `Top Severity at Arraign`,
  `Top Charge Weight at Arraign`,
  `Top Charge at Arraign Violent Felony Ind`, `Arraign Charge Category`,
  `Representation at Securing Order`, `prior_vfo_cnt`, `prior_nonvfo_cnt`,
  `prior_misd_cnt`.

Repo expectation:

- Raw file: `benchmark/data/raw/ny_courts_pretrial/pretrial_release_rows.csv`
- Prepared file: `benchmark/data/real/ny_courts_pretrial.csv`

Prep command:

```sh
uv run legalbench-xai prepare-ny-courts-pretrial \
  --input-csv benchmark/data/raw/ny_courts_pretrial/pretrial_release_rows.csv \
  --output-csv benchmark/data/real/ny_courts_pretrial.csv
```

The prep step filters to a clean released-vs-detained slice: `ROR` and
`Nonmonetary release` become `released`; `Remanded` and `Bail-set` become
`detained`. `Disposed at Arraign` and `Unknown` are skipped.

## FDA Inspections

Official source:

- inspection database: `https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/inspection-references/inspection-classification-database`
- classification definitions: `https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/inspection-basics/inspection-classifications`

Recommended first slice:

- Export the inspection-classification table with at least: `Classification`,
  `Program Area`, `Project Area`, `State`, `Country`, `District`, and either
  `Fiscal Year` or an end-date column.

Repo expectation:

- Raw file: `benchmark/data/raw/fda_inspections/fda_inspections_extract.csv`
- Prepared file: `benchmark/data/real/fda_inspections.csv`

Prep command:

```sh
uv run legalbench-xai prepare-fda-inspections \
  --input-csv benchmark/data/raw/fda_inspections/fda_inspections_extract.csv \
  --output-csv benchmark/data/real/fda_inspections.csv
```

The prep step collapses `VAI` and `OAI` into `action_indicated` and keeps
`NAI` as `no_action_indicated`, producing one row per project area.

## After preparation

Once the three prepared CSVs exist, the cross-domain headline bundle
(`cross_domain_strongest_real_full_cluster`) can be reproduced via the
cluster driver in [`scripts/cluster/`](scripts/cluster/). For the
bundle ↔ config mapping and full reproduction commands, see
[`README.md`](README.md). For dependency setup, see
[`EMPIRICAL_SETUP.md`](EMPIRICAL_SETUP.md).
