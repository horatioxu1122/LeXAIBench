# Feature taxonomy v2: rule, definitions, and every change from v1

**Status: reviewed by the authors (2026-10-15).** The labels were prepared with AI
assistance; the author list has no legal co-author. The authors reviewed the rule,
the definitions and the label changes, and accepted the five judgment calls that
change primary-track membership or rest on practical judgment rather than law:
(1) `south_german_credit.foreign_worker` S -> P (immigration status may be considered
under Reg B 12 CFR 1002.6(b)(7) but must not proxy for national origin);
(2) `ny_courts_pretrial.age_at_arrest` S -> P (age is not a suspect class,
Massachusetts Bd. of Retirement v. Murgia, 427 U.S. 307 (1976));
(3) heloc credit-history-length features labelled P as age proxies (practical judgment);
(4) `LIMIT_BAL`, `purpose`, `loan_purpose` P -> N (practical judgment);
(5) US law (ECOA/Reg B, FCRA) is applied to the German and Taiwanese credit data as a
stylised legal frame. The per-feature `review_status` strings in
`feature_taxonomy_v2.json` still read "draft"; they are left unchanged on purpose because
the v2 runs recorded that file's SHA-256 in their provenance, and this note supersedes them.

Files:

| file | role |
|---|---|
| `benchmark/metadata/feature_taxonomy_v2.json` | labels plus per-feature `rationale`, `legal_basis`, `review_status`; per-dataset `legal_regime`; top-level `definitions` |
| `benchmark/metadata/dataset_registry_v2.json` | the same structure as v1, with the exclusion lists regenerated from the rule; the top-level `track_policy` states the rule; `paper_sources` point to files that exist |
| `paper/feature_taxonomy_manifest_v2.yaml` | manifest of all seven datasets (v1 covered only the four credit datasets), byte-identical to what `validate_taxonomy.py --write-yaml` writes |
| `benchmark/cslaw/validate_taxonomy.py` | checks the rule, the label code sets, the provenance fields, dataset coverage and loader compatibility; prints distributions and the v1 diff |

The v1 files (`feature_taxonomy.json`, `dataset_registry.json`,
`paper/feature_taxonomy_manifest.yaml`) are unchanged. Only configs whose
`registry` block points to the v2 files use v2, so the published v1 bundles
still reproduce. `FeatureSpec` gained three optional fields (`rationale`,
`legal_basis`, `review_status`, all defaulting to `None`). They never reach a
canonical explanation or a score.

Validate (reads JSON only, writes only the given outputs):

```bash
python benchmark/cslaw/validate_taxonomy.py \
  --taxonomy benchmark/metadata/feature_taxonomy_v2.json \
  --registry benchmark/metadata/dataset_registry_v2.json \
  --compare-v1-taxonomy benchmark/metadata/feature_taxonomy.json \
  --write-yaml /tmp/feature_taxonomy_manifest_v2.yaml \
  --report /tmp/taxonomy_validation.md
```

The cluster step `taxonomy` in `run_cslaw_credit.sh` runs this command and stops
the job on any violation.

## 1. What was wrong in v1

1. **Exclusion lists were hand-curated and inconsistent.** south_german_credit
   kept `age` (an S feature) in the primary track. default_of_credit_card_clients
   excluded `EDUCATION` and give_me_some_credit excluded `NumberOfDependents`,
   both P features, while other P features stayed. heloc excluded
   `ExternalRiskEstimate` (P). hmda excluded all geography, ny_courts_pretrial
   excluded `county_name` but not `court_type`, and fda_inspections excluded
   `fiscal_year` although it was labeled N.
2. **Labels contradicted the manuscript's own definitions.** The manuscript
   defines verifiability Y as values the person can verify "from documents they
   hold". Yet 21 of 23 HELOC features, all of which are derived from a consumer
   report, were labeled Y. Correctability P was used for self-reported
   application data, which the subject can correct directly, so the
   manuscript's own definition would give Y. The same kind of feature got
   different labels in different datasets: HMDA `derived_sex` was V=Y, C=N
   while `derived_race` was V=P, C=P, although all three fields are collected
   the same way.
3. **No label had a recorded rationale or legal basis.**
4. **The YAML manifest omitted hmda, ny_courts_pretrial and fda_inspections, and
   `paper_sources` cited a file that does not exist**
   (`paper/dataset_cards_and_feature_taxonomy.md`).

## 2. The track rule (one rule, applied mechanically)

> The **primary track excludes exactly the taxonomy features labeled sensitivity
> S** (protected-class characteristics under the domain's governing law, or
> their direct encodings). Proxy (P) and non-sensitive (N) features **stay in the
> primary track**, where proxy effects are analyzed rather than assumed away.
> The **audit track excludes no taxonomy feature.** Non-taxonomy identifier
> columns (default credit `ID`) are excluded from both tracks.

The rule lives in `dataset_registry_v2.json` under `track_policy`. The validator
regenerates the lists from the taxonomy and fails on any difference. Lists are
ordered with identifier columns first, then features in taxonomy order. The
order does not affect loading, because `build_bundle` sorts features.

| dataset | v1 primary excludes | v2 primary excludes | enters primary | leaves primary | primary size v1 -> v2 |
|---|---|---|---|---|---|
| heloc | ExternalRiskEstimate | (none) | ExternalRiskEstimate | - | 22 -> 23 |
| default_of_credit_card_clients | ID, SEX, MARRIAGE, AGE, EDUCATION | ID, SEX, MARRIAGE, AGE | EDUCATION | - | 19 -> 20 |
| give_me_some_credit | age, NumberOfDependents | age | NumberOfDependents | - | 8 -> 9 |
| south_german_credit | personal_status_sex, foreign_worker | personal_status_sex, age | foreign_worker | age | 18 -> 18 |
| hmda | applicant_age, co_applicant_age, derived_race, derived_ethnicity, derived_sex, state_code, county_code, census_tract | applicant_age, co_applicant_age, derived_race, derived_ethnicity, derived_sex | state_code, county_code, census_tract | - | 10 -> 13 |
| ny_courts_pretrial | gender, race, ethnicity, age_at_arrest, county_name | gender, race, ethnicity | age_at_arrest, county_name | - | 11 -> 13 |
| fda_inspections | state, country, district, fiscal_year | (none) | state, country, district, fiscal_year | - | 2 -> 6 |

The audit lists are unchanged (`ID` for default credit, empty elsewhere).

Consequences to state in the paper:

- **heloc and fda_inspections have no S feature**, so their primary and audit
  tracks are now identical. Their primary-vs-audit comparison is a sanity check
  (identical inputs) and carries no governance information. The non-credit
  sharding can run one fda_inspections track and report it for both.
- The primary track now contains geographic proxies (HMDA tract, NY county and
  court type, FDA location), age proxies (credit-history length, employment and
  residence duration) and `foreign_worker`. The proxy-substitution analysis
  measures how much of the decision these carry. That is the point of keeping
  them, but v2 primary-track results are not feature-for-feature comparable with
  v1.
- The HMDA primary track now includes `census_tract`, a high-cardinality
  categorical, so its fitting cost is about that of the audit track.

## 3. Label definitions applied in v2

The manuscript (taxonomy paragraph of the Benchmark section) defines the five
labels in prose. v2 applies those definitions with the operational rules below.
They are recorded in `feature_taxonomy_v2.json` under `definitions` and repeated
in the YAML manifest.

**Verifiability** (manuscript definition, applied by data source of record)

- **Y**: the subject can check the value from documents they hold, such as the
  application they submitted, their own account or billing statements, pay and
  tax records, the accusatory instrument served on them, or identity documents.
- **P**: the subject can check the value only by obtaining institutional or
  third-party records. Examples are the consumer report (obtainable under FCRA,
  15 U.S.C. 1681g), lender or agency records, and geocoding records. This
  includes values the institution computes from such records (for example a DTI
  that uses bureau-reported debts).
- **N**: the subject cannot check the value even when told it (proprietary
  scores, model outputs).

**Correctability.** v2 classifies on one axis only: if the recorded value is
wrong, does a correction channel exist, and who controls it? The three codes are
mutually exclusive on that axis. Whether an *accurately* recorded value can
change is a different question, answered by mutability; it plays no part in
correctability.

- **Y**: an erroneous recorded value can be corrected through a channel the
  subject can invoke. These are documentation or restatement for
  subject-supplied data (including self-stated personal characteristics such as
  age and sex, and the property address), an FCRA dispute for consumer-report
  data (15 U.S.C. 1681i; direct disputes with furnishers, 1681s-2(a)(8)), a
  billing-error dispute for card-account data (15 U.S.C. 1666; Reg Z 12 CFR
  1026.13), and motion practice or record review for charges and criminal
  history.
- **P**: an erroneous recorded value can be corrected only at the institution's
  discretion or only indirectly. This covers values that the institution
  records, observes, sets, assigns or computes with its own judgment, such as
  bank-internal records, lender-set limits, observation-based demographics,
  geocodes, agency and court records, institutional assignments (court, program
  area, district), lender-computed ratios and appraised values. It also covers a
  proprietary score, which can change only through its inputs.
- **N**: no channel exists to correct an erroneous recorded value, not even at
  the institution's discretion (e.g., a model-derived value with no dispute
  path). No feature in the seven datasets has this label in v2; the code is kept
  so the scale is complete.

Earlier drafts of v2 also used N for "an accurately recorded immutable
characteristic or fixed fact leaves nothing to correct". That mixed a mutability
question into correctability, overlapped with Y for self-stated facts (a
self-reported age or address meets both), and labeled SEX and AGE C=N while
MARRIAGE and EDUCATION, equally self-reported, were C=Y. v2 drops that reading:
self-stated characteristics and the property address are C=Y, and
institution-recorded locations, dates and assignments are C=P (Section 4 lists
every feature this affects). Because CSS weights correctability (Y 1.0, P 0.5,
N 0.0), this matters for the geography features that v2 keeps in the primary
track.

**Manuscript text must change here.** The manuscript defines correctability by
who performs the correction: "by the subject directly, by the institution upon
request, or not at all". Read literally, an FCRA dispute is "by the institution
upon request", so P. v2 instead counts a statutory dispute right as Y, because
the legally relevant difference for contestability is between an enforceable
channel and institutional discretion. Section 8 proposes replacement wording. If
the authors prefer the manuscript's literal reading, change the bureau-derived
features (heloc, give_me_some_credit, south_german_credit `credit_history`) from
C=Y to C=P and rerun the validator. The rule is unaffected.

**Sensitivity** (domain-specific legal anchor)

- **S**: a protected-class characteristic under the governing law, or its direct
  encoding.
  - Credit: the ECOA/Reg B prohibited bases (15 U.S.C. 1691(a); 12 CFR
    1002.2(z)): race, color, religion, national origin, sex, marital status, age,
    and public-assistance income. Reg B 1002.6(b)(2) permits age in an
    empirically derived system on conditions, but age is still a prohibited
    basis and stays S.
  - Mortgage: the same bases plus the Fair Housing Act classes (42 U.S.C. 3605).
  - Pretrial: equal-protection suspect or quasi-suspect classes (race,
    ethnicity or national origin, sex).
  - Regulated firms: none.
- **P**: a plausible proxy for a protected characteristic or a socially sensitive
  attribute. v2 uses these categories:
  - geography (redlining proxy for race and national origin);
  - household composition (dependents, which track marital and familial status;
    Reg B 1002.5(d)(3), 1002.6(b)(3));
  - immigration or foreign-worker status (national-origin proxy; consideration
    is permitted by Reg B 1002.6(b)(7));
  - tenure measures that accrue mechanically with age (credit-history length,
    years employed, years at residence);
  - socioeconomic indicators of the person's circumstances (income, savings,
    assets, home ownership, education, occupation, housing tenure, account
    status), and factors restricted by rule, such as a telephone listing (Reg B
    1002.6(b)(4)).
- **N**: neither. This covers credit performance, loan and account terms,
  purposes of credit, and charge or case facts.

**Mutability** (v1 codes kept unless clearly wrong)

- **I**: immutable personal characteristic.
- **PS / PL**: path-dependent change through behavior in the short/medium or the
  long term.
- **M**: directly mutable at low cost.
- **P**: partially mutable or institution-structured.
- **N**: not actionable by the subject. This covers fixed institutional or
  historical facts, and characteristics that may change in fact but are never
  legitimate recourse.

The manuscript's prose has three levels: immutable, high cost, low cost. Its
examples (income = high cost, requested loan amount = low cost) were used to
resolve two inconsistencies (Section 4).

**Principal-reason suitability** is unchanged from v1. Only ExternalRiskEstimate
(Low) now cites a legal basis: Reg B 12 CFR 1002.9(b)(2) says a failed
qualifying score is not a specific reason. FDA `project_area` (High) is flagged
as questionable in its rationale.

**Legal regime per dataset** (`legal_regime` in the JSON). heloc, give_me_some_credit
and hmda are US data and the cited law applies directly.
default_of_credit_card_clients (Taiwan, 2005) and south_german_credit (Germany,
1973-75) are labeled under the US framework *by analogy*, as the benchmark's
reference regime; the labels do not describe the law that governed those
decisions. ny_courts_pretrial uses N.Y. CPL art. 510 and equal-protection
doctrine. fda_inspections concerns firms, to which no protected-class law
applies.

## 4. Every change, by dataset

V = verifiability, C = correctability, S = sensitivity, M = mutability. There
are 115 label changes in total. No principal-reason label changed, and no
plain_meaning changed. The full reason for each feature is in its `rationale`.

### heloc (40 label changes; 1 inclusion change)

All 23 features are consumer-report derived. The changes follow from that:
V=P (the applicant needs the consumer report to check them) and C=Y (FCRA
dispute of the underlying tradelines).

| feature | change | reason |
|---|---|---|
| ExternalRiskEstimate | V P->N; S P->N; **enters primary** | proprietary consolidated score: it cannot be checked even when disclosed, only corrected indirectly via its inputs (C stays P); a credit-performance summary, not a recognized proxy; excluded in v1 without a rule |
| MSinceOldestTradeOpen | V Y->P; C P->Y; S N->P | bureau-derived; disputable; credit-history length accrues with age (age proxy) |
| AverageMInFile | C P->Y; S N->P | disputable through tradelines; accrues with age (age proxy) |
| MSinceMostRecentTradeOpen, NumSatisfactoryTrades, PercentTradesNeverDelq, NumTotalTrades, NumTradesOpeninLast12M, PercentInstallTrades, MSinceMostRecentInqexcl7days, NetFractionRevolvingBurden, NetFractionInstallBurden, NumRevolvingTradesWBalance, NumInstallTradesWBalance, NumBank2NatlTradesWHighUtilization, PercentTradesWBalance (13 features) | V Y->P; C P->Y | bureau-derived (checkable only from the consumer report); errors disputable under FCRA 1681i (for inquiries, also 1681b permissible purpose) |
| NumTrades60Ever2DerogPubRec, NumTrades90Ever2DerogPubRec, MSinceMostRecentDelq, MaxDelq2PublicRecLast12M, MaxDelqEver, NumInqLast6M, NumInqLast6Mexcl7days (7 features) | V Y->P | bureau-derived; v1 already had C=Y |

### default_of_credit_card_clients (6 label changes; 1 inclusion change)

| feature | change | reason |
|---|---|---|
| LIMIT_BAL | S P->N | issuer-set account term, not a personal characteristic or recognized proxy |
| EDUCATION | C P->Y; **enters primary** | self-reported, correctable with documentation; P (not S), so the rule keeps it in primary |
| MARRIAGE | M I->N; C P->Y | marital status is not immutable but is never legitimate recourse (N blocks it in the v2 counterfactual search, as I did); self-reported, documentable |
| SEX, AGE | C N->Y | self-reported; a misrecorded value can be restated with identity documents, exactly as for MARRIAGE and EDUCATION (correctability asks whether an error can be corrected, not whether an accurate value can change) |

PAY_x, BILL_AMTx and PAY_AMTx are unchanged (V=Y, C=Y). The cardholder holds
the statements, and billing errors are disputable (FCBA 15 U.S.C. 1666, Reg Z
1026.13(a)(4) for uncredited payments; US analogue).

### give_me_some_credit (14 label changes; 1 inclusion change)

| feature | change | reason |
|---|---|---|
| RevolvingUtilizationOfUnsecuredLines | V Y->P; C P->Y | bureau-recorded balances and limits; disputable |
| NumberOfOpenCreditLinesAndLoans | V Y->P; C P->Y | bureau-derived; disputable |
| NumberRealEstateLoansOrLines | V Y->P; C P->Y | bureau-derived; disputable (S=P kept: home-ownership/wealth signal) |
| NumberOfTime30-59DaysPastDueNotWorse, NumberOfTimes90DaysLate, NumberOfTime60-89DaysPastDueNotWorse | V Y->P | bureau-derived; v1 already had C=Y |
| DebtRatio | V Y->P | computed by the lender from bureau-reported debts and self-reported figures (C stays P: the computation is the lender's) |
| MonthlyIncome | C P->Y | self-reported, documentable (S=P kept: Reg B income-source rules) |
| NumberOfDependents | M PS->N; C P->Y; **enters primary** | "have fewer dependents" is not recourse; self-reported; P (familial/marital proxy), so it stays in primary |
| age | C N->Y | self-reported; restatable with identity documents |

### south_german_credit (22 label changes; 2 inclusion changes)

| feature | change | reason |
|---|---|---|
| age | C N->Y; **leaves primary** | S; v1 kept it in primary in violation of its own policy; self-reported, restatable with identity documents |
| foreign_worker | S S->P; M I->N; C P->Y; **enters primary** | immigration/residency status is not itself a prohibited basis (Reg B 1002.6(b)(7) permits considering it) but strongly tracks national origin, so P; not an intrinsic characteristic and never recourse, so N; **open decision, see Section 7** |
| purpose | S P->N; C P->Y | a transaction characteristic, not a recognized proxy; applicant-stated |
| credit_history | V Y->P | drawn from bank and bureau records (C=Y kept: bureau dispute, US analogue) |
| personal_status_sex | V P->Y; C N->Y | built from facts the applicant states, consistent with SEX/MARRIAGE in default credit; a misrecorded code can be restated with documentation |
| people_liable | M PS->N; C P->Y | dependents are not recourse; self-reported |
| duration, amount, savings, employment_duration, other_debtors, present_residence, property, other_installment_plans, housing, job, telephone (11 features) | C P->Y | applicant-supplied and documentable, so restatable (both the manuscript's and the v2 definition give Y) |

Unchanged with C=P: `status` (checking-account record at the lending bank),
`installment_rate` (the lender's disposable-income assessment) and
`number_credits` (the bank's own record).

### hmda (18 label changes; 3 inclusion changes)

| feature | change | reason |
|---|---|---|
| state_code, county_code, census_tract | **enter primary** | P (geographic proxies) stay in primary under the rule, where they are analyzed |
| loan_amount | M P->M; C P->Y | the manuscript's own example of a low-cost mutable feature (and south_german_credit `amount` is M); applicant-stated |
| income | M M->PL; C P->Y | the manuscript's own example of a high-cost mutable feature (and give_me_some_credit `MonthlyIncome` is PL); documentable |
| loan_purpose | S P->N; C P->Y | a transaction characteristic, not a recognized proxy; applicant-stated |
| loan_type, occupancy_type, lien_status | C P->Y | applicant-selected on the application; restatable |
| total_units | C P->Y | applicant-stated property fact; a misrecorded count can be restated with property documents |
| applicant_age, co_applicant_age | C N->Y | computed from the stated date of birth, restatable with identity documents |
| state_code | C N->Y | taken from the property address the applicant states; restatable |
| county_code | C N->P | assigned by the institution's geocoding of the address (like census_tract), so corrected at its discretion |
| debt_to_income_ratio | V Y->P | lender-computed from documented income and bureau-reported debts (C stays P) |
| derived_sex | V Y->P; C N->P | collected like race and ethnicity, including by visual observation or surname when the applicant declines (Reg B 1002.13; Reg C 1003.4(a)(10)(i)); now consistent with derived_race and derived_ethnicity |
| census_tract | V Y->P | assigned by geocoding; checkable only from Census/FFIEC records |

### ny_courts_pretrial (9 label changes; 2 inclusion changes)

| feature | change | reason |
|---|---|---|
| age_at_arrest | V Y->P; C N->P; S S->P; **enters primary** | computed by the agency from the date of birth on the arrest report, the same source of record as `gender`, so V and C follow gender (checkable only from the agency record; a misrecorded date of birth is corrected at the agency's discretion); age is not a suspect classification under equal protection (Murgia, 427 U.S. 307), and no statute we found bars its use in securing-order decisions; **open decision, see Section 7** |
| county_name | C N->P; **enters primary** | P (geographic proxy); v1 excluded it while keeping `court_type` (also P); a court-recorded fact, corrected at the court's discretion |
| court_type | C N->P | a court-recorded institutional assignment, corrected at the court's discretion |
| gender | V Y->P | recorded on the arrest report, so checkable only from the agency record (consistent with race and ethnicity) |
| prior_vfo_cnt, prior_nonvfo_cnt, prior_misd_cnt | V Y->P | taken from the state criminal-history record (C=Y kept: a record-review challenge procedure exists; citation to be confirmed) |

### fda_inspections (6 label changes; 4 inclusion changes)

| feature | change | reason |
|---|---|---|
| state, country, district, fiscal_year | **enter primary** | no S features exist for firms; v1 excluded three P features and one N feature without a rule |
| project_area | V Y->P; C N->P | FDA's internal compliance-program assignment, confirmable only from FDA records and corrected only at FDA's discretion |
| district | V Y->P; C N->P | FDA's own jurisdiction mapping, corrected at FDA's discretion |
| program_area, fiscal_year | C N->P | FDA assignment and FDA-recorded date, corrected at FDA's discretion |

`state` and `country` keep v1's C=P (recorded in FDA's establishment records,
corrected at FDA's discretion on the firm's request).

## 5. Label distributions, v1 vs v2 (116 features, 7 datasets)

Overall counts (share):

| dimension | v1 | v2 |
|---|---|---|
| verifiability Y / P / N | 109 (94.0%) / 7 (6.0%) / 0 | 71 (61.2%) / 44 (37.9%) / 1 (0.9%) |
| correctability Y / P / N | 37 (31.9%) / 62 (53.4%) / 17 (14.7%) | 88 (75.9%) / 28 (24.1%) / 0 |
| sensitivity S / P / N | 16 (13.8%) / 26 (22.4%) / 74 (63.8%) | 14 (12.1%) / 26 (22.4%) / 76 (65.5%) |
| mutability I / PS / PL / M / P / N | 16 / 28 / 22 / 16 / 14 / 20 | 14 / 26 / 23 / 16 / 13 / 24 |
| principal_reason High / Medium / Low | 35 / 38 / 43 | 35 / 38 / 43 (unchanged) |

Per dataset (counts):

| dataset | n | V Y/P/N v1 -> v2 | C Y/P/N v1 -> v2 | S S/P/N v1 -> v2 |
|---|---:|---|---|---|
| heloc | 23 | 21/2/0 -> 0/22/1 | 7/16/0 -> 22/1/0 | 0/1/22 -> 0/2/21 |
| default_of_credit_card_clients | 23 | 23/0/0 -> 23/0/0 | 18/3/2 -> 22/1/0 | 3/2/18 -> 3/1/19 |
| give_me_some_credit | 10 | 10/0/0 -> 3/7/0 | 3/6/1 -> 9/1/0 | 1/3/6 -> 1/3/6 |
| south_german_credit | 20 | 19/1/0 -> 19/1/0 | 1/17/2 -> 17/3/0 | 3/10/7 -> 2/10/8 |
| hmda | 18 | 16/2/0 -> 13/5/0 | 0/13/5 -> 10/8/0 | 5/5/8 -> 5/4/9 |
| ny_courts_pretrial | 16 | 14/2/0 -> 9/7/0 | 8/5/3 -> 8/8/0 | 4/2/10 -> 3/3/10 |
| fda_inspections | 6 | 6/0/0 -> 4/2/0 | 0/2/4 -> 0/6/0 | 0/3/3 -> 0/3/3 |

`validate_taxonomy.py` prints the same tables with shares, and the
`--compare-v1-taxonomy` diff, for the files as they are when it runs.

## 6. Where the labels act in the code, and what to expect in v2 results

| label | consumers |
|---|---|
| verifiability, correctability | CSS (`VERIFICATION_MAP`, `CORRECTABILITY_MAP`, weighted by referenced-feature importance); correction support in unified actionability |
| sensitivity | track membership (this rule); canonicalizer legality flags (S = direct sensitive, P = proxy sensitive) used by ALF legality and hard failures; counterfactual search (S never changed; Wachter also skips P); stability probe holds mutability I and sensitivity S fixed |
| mutability | ALF feasibility (`MUTABILITY_MAP`); counterfactual search (I never changed; with v2 corrections, N is also blocked and hard-failed); Wachter also skips mutability P; stability jitter scale |
| principal_reason | legality flag `low_principal_reason_feature`; the heuristic stub weights (not used for ranking under v2 native contributions) |

Expected direction of effects. These come from the labels, not from methods, so
version diffs must attribute them:

- **heloc CSS composition changes for every explainer.** Verifiability falls
  (all features are now P or N) and correctability rises (C=Y). Any v1-vs-v2
  CSS difference on heloc is mostly label-driven.
- **Primary-track models change** wherever the inputs changed (Section 2). The
  largest change is likely heloc, where `ExternalRiskEstimate` (a consolidated
  score) enters and may dominate attributions.
- **Counterfactual search spaces change only where the gating labels change.**
  `_feature_allowed` (explainers/real.py) blocks sensitivity S and mutability I
  for every search, and Wachter also skips sensitivity P and mutability P.
  - Wachter can now change hmda `loan_amount` (mutability P->M, sensitivity N).
  - Wachter can no longer change heloc `MSinceOldestTradeOpen` and
    `AverageMInFile` (sensitivity N->P).
  - `LIMIT_BAL`, `purpose` and `loan_purpose` are unaffected for Wachter
    although their sensitivity moved P->N: their mutability is still P.
  - `foreign_worker` was already blocked in v1 (sensitivity S, mutability I). In
    v2 it is blocked for Wachter by sensitivity P, and for DiCE only through
    the v2 `cf_respect_nonactionable` correction (mutability N); without that
    correction DiCE could change it.
  - Mutability-N features (`MARRIAGE`, dependents, `foreign_worker`, locations)
    are blocked by `cf_respect_nonactionable`.
- **The stability probe perturbs different features.** `_sample_neighbor_row`
  (stability.py) holds a feature fixed only when mutability is I or sensitivity
  is S. The v2 probe therefore perturbs south_german_credit `foreign_worker`,
  which v1 held fixed (v1: S, I; v2: P, N). The jitter scale follows
  mutability, so it changes for every feature whose mutability changed (hmda
  `loan_amount` and `income`, give_me_some_credit `NumberOfDependents`,
  south_german_credit `people_liable`, default credit `MARRIAGE`; the
  `--compare-v1-taxonomy` diff lists them all). Stability differences on these
  datasets are partly label-driven.
- **Correctability rises for self-stated characteristics and falls nowhere to
  N.** C=N is no longer used (Section 3). CSS's correctability term therefore
  rises for explanations citing self-stated age, sex or the property state
  (C=Y) and for institution-recorded locations and assignments (C=P), notably
  the geography features now in the primary track.

## 7. Open decisions for the authors (check before submission)

1. **`foreign_worker` (south_german_credit) S->P.** Defensible under Reg B
   1002.6(b)(7), but many fairness papers treat it as sensitive. If the authors
   treat it as a direct encoding of national origin, set S=S. The validator will
   then demand its exclusion from primary.
2. **`age_at_arrest` (ny_courts_pretrial) S->P.** Age is S in credit because ECOA
   names it. Pretrial law has no equivalent, but age is a contested pretrial risk
   factor. Setting S=S restores the v1 exclusion.
3. **Correctability of consumer-report data (Y vs P).** See Section 3. This
   choice moves CSS for every bureau-derived reason.
4. **Telephone listing (south_german_credit).** Reg B 1002.6(b)(4) bars
   considering a telephone listing in the applicant's name for consumer credit,
   but it is not a prohibited basis, so the one-rule policy keeps it in primary.
   An alternative rule ("exclude S and expressly prohibited factors") would
   exclude it. That would be a second rule, and it is not adopted here.
5. **Age proxies.** v2 labels credit-history length (heloc
   `MSinceOldestTradeOpen`, `AverageMInFile`) P, consistent with employment and
   residence duration. These are standard scoring factors. If the authors prefer
   to reserve P for recognized fair-lending proxies, revert all four to N.
6. **Correctability as an error-correction question only (Section 3).** v2 no
   longer uses C=N for accurately recorded immutable or fixed facts; such
   features are Y (self-stated) or P (institution-recorded). If the authors
   want "pointless to contest when accurate" to count, it belongs to
   mutability, which already records it, not to correctability. Reverting to
   the earlier reading moves CSS for every explanation that cites age, sex or
   geography.
7. **Principal-reason labels were not revised.** `project_area` (FDA, High) is
   doubtful.
8. **Citations to verify**: the paragraph numbers in Reg C 12 CFR 1003.4(a)
   ((a)(2), (3), (5), (6), (7), (9)(ii), (10)(i)-(iii), (14), (23), (24), (31));
   Reg B 1002.13 (visual observation or surname); N.Y. CPL 150.10/150.20,
   170.10/180.10, 170.30/210.20; the New York DCJS criminal-history
   record-review procedure (marked "citation to be confirmed"); Massachusetts
   Bd. of Retirement v. Murgia, 427 U.S. 307 (1976).
9. **No legal co-author.** The paper should state that the authors assigned the
   labels by the stated rules, that rationales and citations are released per
   feature, and that the labels were not reviewed by a lawyer (unless an external
   reviewer is added).

## 8. Proposed manuscript wording

Replacement for the taxonomy paragraph (Benchmark section):

> Each dataset is paired with a feature taxonomy that labels every feature on
> five attributes, each with a stated rationale and, where one exists, a legal
> citation. *Mutability* records whether the feature is immutable (e.g., age),
> changes only through behavior over the short or long term (e.g., delinquency
> history, income), is directly mutable at low cost (e.g., requested loan
> amount), is structured jointly with the institution (e.g., credit limit), or is
> not actionable by the subject (e.g., location, criminal charges).
> *Verifiability* records whether the affected person can verify the value from
> documents they hold, only by obtaining institutional records (e.g., a consumer
> report), or not at all (e.g., a proprietary score). *Correctability* records
> whether an erroneous value can be corrected through a channel the person can
> invoke (documentation of self-reported data, a statutory dispute of
> consumer-report or billing data), only at the institution's discretion, or
> not at all. *Sensitivity* records whether the feature is a protected
> characteristic under the domain's governing law (ECOA and Regulation B for
> credit, adding the Fair Housing Act for mortgages, equal-protection classes for
> pretrial release), a plausible proxy or socially sensitive attribute (e.g.,
> geography, household size, income), or neither. *Principal-reason suitability*
> records whether the feature is appropriate as a principal reason in an
> adverse-action notice, on a low/medium/high scale.

Replacement for the track description:

> The `primary` track excludes exactly the features labeled as protected
> characteristics; proxy features remain, so that their role can be measured
> rather than assumed away. The `audit` track retains every feature.

Add a limitation sentence:

> Labels for the Taiwanese and German credit datasets apply the U.S. framework by
> analogy, and the taxonomy was assigned by the authors following the stated
> rules rather than reviewed by legal counsel.
