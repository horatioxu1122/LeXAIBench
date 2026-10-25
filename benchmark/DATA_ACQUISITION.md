# Data acquisition

The repository does not include the datasets. All seven are public. This guide
gives the source of each dataset and the command that prepares it. The commands
write the prepared files to `benchmark/data/real/`, where the configurations
expect them. The last column of the table gives the number of rows in each
prepared file that the paper used, so you can check your files against it.

Run every command from the repository root after
`uv sync --extra benchmark-empirical`.

| Dataset | Source | Raw file | Command | Prepared file | Rows |
| --- | --- | --- | --- | --- | ---: |
| HELOC | OpenML 45554 | downloaded by the command | `prepare-heloc` | `heloc.csv` | 9,871 |
| Default of Credit Card Clients | UCI 350 | `benchmark/data/raw/default_credit_uci/default of credit card clients.xls` | `prepare-default-credit` | `default_credit.csv` | 30,000 |
| Give Me Some Credit | OpenML 46929 | downloaded by the command | `prepare-give-me-some-credit` | `give_me_some_credit.csv` | 150,000 |
| South German Credit | UCI 573 | `benchmark/data/raw/south_german_credit_uci/SouthGermanCredit.asc` | `prepare-south-german-credit` | `south_german_credit.csv` | 1,000 |
| HMDA | FFIEC, 2024 snapshot | `benchmark/data/raw/hmda/hmda_snapshot.csv` | `prepare-hmda` | `hmda.csv` | 8,837,657 |
| NY pretrial | New York State Unified Court System, 2019 to 2024 | `benchmark/data/raw/ny_courts_pretrial/pretrial_release_rows.csv` | `prepare-ny-courts-pretrial` | `ny_courts_pretrial.csv` | 1,145,222 |
| FDA inspections | FDA Data Dashboard | `benchmark/data/raw/fda_inspections/fda_inspections_extract.csv` | `prepare-fda-inspections` | `fda_inspections.csv` | 331,403 |

Each dataset keeps its own license. The MIT license of this repository does not
cover the data.

## HELOC and Give Me Some Credit

Both commands download the data from OpenML.

```sh
uv run legalbench-xai prepare-heloc
uv run legalbench-xai prepare-give-me-some-credit
```

HELOC comes from the FICO Explainable Machine Learning Challenge
(<https://www.openml.org/d/45554>). Give Me Some Credit comes from a 2011 Kaggle
competition (<https://www.openml.org/d/46929>). The terms of FICO and Kaggle
apply to these two datasets.

## Default of Credit Card Clients and South German Credit

Download both datasets from the UCI Machine Learning Repository, which
distributes them under CC BY 4.0.

- <https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients>.
  Save `default of credit card clients.xls` in
  `benchmark/data/raw/default_credit_uci/`.
- <https://archive.ics.uci.edu/dataset/573/south+german+credit+update>.
  Save `SouthGermanCredit.asc` in `benchmark/data/raw/south_german_credit_uci/`.

```sh
uv run legalbench-xai prepare-default-credit
uv run legalbench-xai prepare-south-german-credit
```

## HMDA

Download the public loan/application register of the 2024 Snapshot National
Loan-Level Dataset as CSV
(<https://ffiec.cfpb.gov/data-publication/snapshot-national-loan-level-dataset/2024>).
Unzip `2024_public_lar_csv.zip` and save `2024_public_lar_csv.csv` as
`benchmark/data/raw/hmda/hmda_snapshot.csv`.

```sh
uv run legalbench-xai prepare-hmda \
  --input-csv benchmark/data/raw/hmda/hmda_snapshot.csv \
  --output-csv benchmark/data/real/hmda.csv
```

The command keeps the applications that were approved or denied and writes the
fields the benchmark uses.

## NY pretrial

Download the supplemental pretrial release data file from the New York State
Unified Court System (<https://www.nycourts.gov/pretrial-release-data-33136>).
The file covers 2019 to 2024 in two parts. Join the parts into one CSV with one
header row:

```sh
cd benchmark/data/raw/ny_courts_pretrial
unzip Supplemental_Pretrial_Release_Data_File.zip -d unzipped
cat unzipped/PretrialRelease_2019_2024_Part_1.csv > pretrial_release_rows.csv
tail -n +2 unzipped/PretrialRelease_2019_2024_Part_2.csv >> pretrial_release_rows.csv
cd -
uv run legalbench-xai prepare-ny-courts-pretrial \
  --input-csv benchmark/data/raw/ny_courts_pretrial/pretrial_release_rows.csv \
  --output-csv benchmark/data/real/ny_courts_pretrial.csv
```

The command keeps the arraignments that ended in release or detention. ROR and
nonmonetary release count as release. Remand and bail set count as detention.
It drops cases disposed at arraignment and cases with an unknown decision.

## FDA inspections

On the inspections page of the FDA Data Dashboard
(<https://datadashboard.fda.gov/oii/cd/inspections.htm>), download the entire
inspections dataset and the entire citations dataset as Excel files. Save them
in `benchmark/data/raw/fda_inspections/` as `inspection dataset.xlsx` and
`citation.xlsx`. The first command below turns the two workbooks into one CSV,
and it takes the program area of each inspection from its citations.

```sh
uv run python benchmark/scripts/normalize_fda_workbooks.py
uv run legalbench-xai prepare-fda-inspections \
  --input-csv benchmark/data/raw/fda_inspections/fda_inspections_extract.csv \
  --output-csv benchmark/data/real/fda_inspections.csv
```

The prepare command labels Voluntary Action Indicated and Official Action
Indicated as action indicated, and No Action Indicated as no action indicated.
It writes one row per project area. The dashboard adds new inspections over
time. The paper used an export from April 2026, so a later export gives more
rows than the table above.
