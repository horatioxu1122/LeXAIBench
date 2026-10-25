# Analyses

The scripts in this folder produce every table and figure of the paper from the
result bundles. Each script reads `results.json` bundles and writes CSV,
Markdown, LaTeX, or PDF files into a new `--out-dir`. No script overwrites its
inputs. Only `regb_band_sweep.py` and `regb_within_class.py` train models, and
they train the consumer credit models again. The other scripts read stored
results and finish in minutes.

`benchmark/scripts/cluster/run_cslaw_denial.sh` runs the scripts in order, so
you rarely need to call one by hand. Every script lists its options with
`--help`. To call one directly, run it from the repository root:

```sh
PYTHONPATH=benchmark/src .venv/bin/python benchmark/cslaw/<script>.py --help
```

## Instrument version

The package keeps two versions of the instrument. The paper uses version 2,
which `benchmark/src/legalbench_xai/corrections.py` defines. A configuration
selects it in its `cslaw` block:

```json
"cslaw": {"corrections": "v2", "persist_canonical_explanations": true,
          "persist_native_contributions": true, "regb_reference": {"enabled": true}}
```

Version 2 does the following.

- It decomposes the score of each model into the model's own contributions.
  These are the coefficient times the standardized value for logistic
  regression, the term scores of the boosting machine with each pair term split
  equally between its two factors, and the TreeSHAP values of the boosted trees.
- It scores every explanation method, the reference notices included, against
  these contributions.
- It takes the reasons of a counterfactual method from the factors that the
  counterfactuals change, and it blocks changes to factors that the applicant
  cannot act on.
- It draws 24 denied applicants at random within decision-margin strata for each
  dataset, version, and model.
- It uses the version 2 feature taxonomy and dataset registry in
  `benchmark/metadata/`.
- It groups payment amounts apart from late payments, and it reads numbers
  written in exponent notation as numbers.
- It records in each bundle the SHA-256 hash of every data, taxonomy, registry,
  and configuration file, and the commit of the code.

A configuration without `cslaw.corrections` runs version 1. The `regression`
step of `run_cslaw_credit.sh` checks that version 1 still gives the same results
as the code at commit `fb5dadc`.

## The direction rule

Every explanation method ranks factors by the size of their attribution, so a
listed reason can be a factor that favored approval. `denial_reasons.py` keeps
only the factors whose contribution counts toward denial, rebuilds the reasons,
and scores them again. An explanation left without a reason leaves the quality
metrics, and the bundle records it.

## Scripts

The last column names the output folder under
`benchmark/outputs/cslaw_denial_<TAG>/analysis/` and under `results/`.

| Script | What it computes | Output |
| --- | --- | --- |
| `denial_reasons.py` | E1. Applies the direction rule to a bundle. | the bundle folders next to `analysis/` |
| `check_additivity.py` | Checks that the contributions and the constant add up to the model's log odds. | `additivity/` |
| `regb_reference.py` | E2 and E4. Compares each method's reasons with the reasons of the marginal and average applicant methods of comment 9(b)(2)-5, and the two methods with each other. With `--k`, it varies the number of reasons. | `regb_*/`, `sweep_k/` |
| `canned_paired.py` | E2. Compares each method's share of the marginal applicant method's reasons with the canned notice's share on the same applicants. | `canned_paired/` |
| `notice_specificity.py` | E3. The overlap of reason sets across applicants. E6. The share of denied applicants with a counterfactual that changes the decision. | `specificity/` |
| `multiplicity.py` | E4. First-reason disagreement across model classes on the same applicants, and across methods on the same model. With `--k`, it varies the number of reasons. | `multiplicity/`, `sweep_k/` |
| `regb_within_class.py` | E4. Retrains the models on fresh samples of the training data and measures how often the first reason changes. | `within_class/` |
| `reason_families.py` | E4. Repeats the first-reason comparisons with correlated factors counted as one reason. | `reason_families/` |
| `regb_band_sweep.py` | Varies the band of marginal applicants in comment 9(b)(2)-5. | `band_sweep/` |
| `track_disclosure.py` | E5. The share of notices that name a protected characteristic or a possible proxy, with removal baselines. | `track_disclosure/`, `hmda_geo/` |
| `proxy_substitution.py` | E5. The factors that replace the excluded characteristics in the reasons of the same applicant. | `proxy_substitution/` |
| `common_faithfulness.py` | How closely each method's reasons follow the model's contributions. | `common_faithfulness*/` |
| `paper_tables.py` | Tables of the instrument scores of each method and reference notice. | `paper_tables/` |
| `appendix_tables.py` | The appendix tables as LaTeX, and the worked notices. | `appendix_generated/` |
| `paper_figures.py` | The figures of Section 5 as PDF, with the plotted values and their intervals as CSV. | `figures/` |
| `paper_schematic.py` | The schematic of Section 3. | `figures/` |

The run scripts also use these files.

| File | Use |
| --- | --- |
| `common.py` | Helpers that the other scripts share. |
| `selftest.py` | Runs every script on synthetic bundles. It is the first step of `run_cslaw_credit.sh` and prints `cslaw selftest: OK (N checks)`. |
| `validate_taxonomy.py` | Checks the taxonomy and the registry against the labeling rules. |
| `compare_bundles.py` | Compares two bundles exactly, for the `regression` step. |
| `merge_bundles.py` | Merges the shard bundles of the HMDA, NY pretrial, and FDA runs. |
| `rescore.py`, `version_diff.py` | Rescore a bundle and compare version 1 with version 2. `paper_tables.py` and the self-test use them. |

## Fields in a bundle

Version 2 stores these fields in `results.json`, and the scripts read them.

| Field | Where | Content |
| --- | --- | --- |
| `eval_row_index` | each instance | the row in the held-out table, which identifies the same applicant across models and versions |
| `canonical_explanation` | each instance | the explanation in a common format, with its principal reasons, referenced factors, and counterfactual changes |
| `native_contributions` | each instance | `{source, units: "log_odds_adverse", bias, values}`, the model's own contribution of each factor. A positive value pushes toward denial, and `sum(values) + bias` is the model's log odds. |
| `sampling.selection`, `sampling.seed` | each summary row | `stratified_random` and its seed |
| `payload["cslaw"]["corrections"]`, `["corrections_version"]` | top level | the options of the run and its version |
| `payload["cslaw"]["provenance"]` | top level | the path, SHA-256 hash, and row count of each dataset, the hashes of the taxonomy, registry, and configuration, and the commit of the code |
| `payload["cslaw"]["groups"]["<dataset>\|<track>\|<model>"]` | top level | the reference means of the marginal and average applicant methods, the threshold, and the source of the contributions |
| `payload["merge"]` | merged bundles | the shard inputs and the settings that the merge checked |
