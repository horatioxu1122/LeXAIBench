# LeXAIBench

This repository holds the code and the analysis outputs for the paper
"What Counts as a Principal Reason? Testing Explainable AI Under Regulation B"
by Jingyan Xu and Tianyi Yang.

The Equal Credit Opportunity Act and Regulation B require a creditor that denies
a credit application to disclose the principal reasons for the denial. The paper
separates this requirement into six elements and tests each element against the
model's own decomposition of its score. It applies the tests to five
explanation methods, three model classes, and seven public datasets.

## Contents

| Path | What it holds |
| --- | --- |
| `benchmark/src/legalbench_xai/` | The benchmark package. It prepares the data, trains the models, runs the explanation methods, and scores the explanations. |
| `benchmark/cslaw/` | One script for each analysis in the paper. See its [README](benchmark/cslaw/README.md). |
| `benchmark/scripts/cluster/` | The four scripts that produce every result. |
| `benchmark/configs/` | The experiment configurations. |
| `benchmark/metadata/` | The feature taxonomy, which gives the legal basis of each label, and the dataset registry. |
| `paper/` | The taxonomy manifest and the schemas that the configurations read. |
| `benchmark/data/demo/` | Two or three rows of each dataset, which show its schema. |
| `benchmark/DATA_ACQUISITION.md` | How to get and prepare each dataset. |
| `results/` | The analysis outputs behind the tables and figures of the paper. |

## Setup

The code needs Python 3.12 or later and [uv](https://docs.astral.sh/uv/).
Clone the full history, because one check compares the code with an earlier
commit.

```sh
git clone https://github.com/horatioxu1122/LeXAIBench.git
cd LeXAIBench
uv sync --extra benchmark-empirical
uv run legalbench-xai doctor
```

The lock file `uv.lock` fixes the version of every library, including
scikit-learn 1.8.0, interpret 0.7.6, XGBoost 3.2.0, SHAP 0.51.0, and LIME
0.2.0.1. Then prepare the seven datasets as
[`benchmark/DATA_ACQUISITION.md`](benchmark/DATA_ACQUISITION.md) describes.

## Reproduce the results

Four scripts produce every result. Run them from the repository root in the
order below, with one tag for all of them. Each script takes the path of the
repository as its first argument and writes new directories under
`benchmark/outputs/`. Pass `--help` to any script to see its options.

```sh
REPO=$PWD
TAG=r1

# 1. Consumer credit: checks, the main run, the reference notices, and the
#    shared applicant run. The second command adds the two LIME robustness runs.
bash benchmark/scripts/cluster/run_cslaw_credit.sh "$REPO" --tag "$TAG"
bash benchmark/scripts/cluster/run_cslaw_credit.sh "$REPO" --steps limelocal,limedefault --tag "$TAG"

# 2. HMDA, NY pretrial, and FDA inspections. Each shard can run as its own job.
for shard in hmda:primary hmda:audit ny_courts_pretrial:primary ny_courts_pretrial:audit \
             fda_inspections:primary fda_inspections:audit; do
  bash benchmark/scripts/cluster/run_cslaw_noncredit_v2.sh "$REPO" --shard "$shard" --tag "$TAG"
done
bash benchmark/scripts/cluster/run_cslaw_noncredit_v2.sh "$REPO" --merge --tag "$TAG"

# 3. The direction rule and every analysis, including the sweeps, the
#    retraining baseline, the appendix tables, and the figures.
bash benchmark/scripts/cluster/run_cslaw_denial.sh "$REPO" --tag "$TAG"

# 4. HMDA with county and census tract encoded as denial rates. It adds its
#    results to the output of step 3, so the appendix tables are built again.
bash benchmark/scripts/cluster/run_cslaw_hmda_geo.sh "$REPO" --tag "$TAG"
PYTHONPATH=benchmark/src .venv/bin/python benchmark/cslaw/appendix_tables.py \
  --denial-dir "benchmark/outputs/cslaw_denial_$TAG" --source-dir benchmark/outputs \
  --tag "$TAG" --out-dir "benchmark/outputs/cslaw_denial_$TAG/analysis/appendix_generated"
```

The analyses end up in `benchmark/outputs/cslaw_denial_<TAG>/analysis/`, which
has the same layout as `results/`. Each result bundle stores the SHA-256 hash
of every data, taxonomy, registry, and configuration file, and the commit of
the code that produced it.

All runs for the paper used one server with 80 cores and 156 GB of memory. The
FDA runs took about 0.4 hours per version, the HMDA runs 4 to 5 hours, and the
NY pretrial runs about 15 hours. The analyses read stored results and finish in
minutes, except the band sweep and the retraining baseline, which train the
consumer credit models again.

## License

The code, configurations, documentation, and analysis outputs are released under
the MIT License (see [`LICENSE`](LICENSE)). The datasets keep their own
licenses, which [`LICENSE`](LICENSE) and
[`benchmark/DATA_ACQUISITION.md`](benchmark/DATA_ACQUISITION.md) list.
