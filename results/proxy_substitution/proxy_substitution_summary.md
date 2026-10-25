# Proxy substitution between primary and audit tracks

Bundle: `shared`; reason set: `principal`.

Settings: corrections: v2 (native_contributions=True, common_faithfulness_reference=True, faithfulness_quality_weight=0.0, counterfactual_reasons_from_changes=True, cf_respect_nonactionable=True, reason_grouping_fix=True, sampling_selection=stratified_random); native ranking = True; DiCE/Wachter reasons from counterfactual changes = True; corrected temporal grouping = True.

Shared sampling scope `models_and_tracks`: both tracks explain the same applicants.

**Legal context (to verify; not legal advice).** Reg B (12 CFR 1002.9(b)(2))
requires specific principal reasons, and comment 9(b)(2)-2 requires them to
relate to the factors actually scored, so a scored proxy may legitimately
appear as a reason. As we understand it, the CFPB's 2026 amendments to
Regulation B removed the disparate-impact ("effects test") theory under
ECOA/Reg B; on that reading, reliance on proxies for a protected characteristic
matters mainly as potential evidence of intentional proxy discrimination
(disparate treatment). Disparate impact remains relevant under the Fair Housing
Act for mortgage lending. Verify the current status of these rules before
relying on this framing.

**Limitations.** The taxonomy has no proxy-strength field (`P` is a
hand-assigned label meaning proxy-sensitive or socially sensitive). In v1
bundles the primary-track exclusion lists are hand-curated (for example,
south_german_credit keeps `age`, an S feature, in primary); v2 bundles use the
v2 registry recorded in their provenance (see the exclusions table below).
"Replacement" features are those cited in primary but not in audit for the same
applicant; they may enter because the model was refit, not because they encode
the excluded attribute.

## Primary-track exclusions and S features kept in primary

- `dataset_registry`: `benchmark/metadata/dataset_registry_v2.json` (provenance)
- `feature_taxonomy`: `benchmark/metadata/feature_taxonomy_v2.json` (provenance)

| dataset | excluded in primary (registry) | their sensitivity | S features kept in primary | dropped by the run (native contributions) | registry matches run |
| --- | --- | --- | --- | --- | --- |
| default_of_credit_card_clients | SEX|MARRIAGE|AGE | S|S|S | (none) | AGE|MARRIAGE|SEX | True |
| give_me_some_credit | age | S | (none) | age | True |
| heloc | (none) | - | (none) | (none) | True |
| south_german_credit | personal_status_sex|age | S|S | (none) | age|personal_status_sex | True |

The registry's exclusion lists match the features the run dropped (where native contributions are persisted).

## Substitution summary by dataset (pooled over models; CIs stratified by model)

Columns: share of applicants whose audit reasons cite a primary-excluded feature; among those, share whose primary reasons gain a P-labelled replacement; change (primary minus audit) in the share of reasons citing a P feature that is retained in primary; share of primary explanations citing an S feature; Jaccard of the primary vs audit reason units (first 4 notice reasons per track, a PAY_*/BILL_AMT* temporal family counted once; Reg B reasons are built as the same units, so explainer and Reg B rows are comparable).

| dataset | method | audit cites excluded | P replacement | excluded cited | delta retained-P reason share | primary cites S | Jaccard primary/audit (reasons) | n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| default_of_credit_card_clients | regb_m1 | 0.625 [0.514, 0.736] | 0.422 [0.311, 0.533] | 0.059 [0.031, 0.087] | 0.000 [0.000, 0.000] | 0.669 [0.611, 0.729] | 72 |
| default_of_credit_card_clients | regb_m2 | 0.417 [0.306, 0.528] | 0.367 [0.267, 0.467] | 0.035 [0.017, 0.052] | 0.000 [0.000, 0.000] | 0.833 [0.785, 0.879] | 72 |
| default_of_credit_card_clients | anchors | 0.069 [0.014, 0.125] | 0.000 [0.000, 0.000] | -0.005 [-0.014, 0.000] | 0.000 [0.000, 0.000] | 0.751 [0.696, 0.809] | 72 |
| default_of_credit_card_clients | dice | 0.000 [0.000, 0.000] | n/a | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.938 [0.904, 0.968] | 72 |
| default_of_credit_card_clients | lime | 0.000 [0.000, 0.000] | n/a | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.993 [0.979, 1.000] | 72 |
| default_of_credit_card_clients | shap | 0.167 [0.097, 0.236] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.857 [0.813, 0.899] | 72 |
| default_of_credit_card_clients | wachter | 0.000 [0.000, 0.000] | n/a | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.882 [0.806, 0.944] | 72 |
| give_me_some_credit | regb_m1 | 0.278 [0.181, 0.375] | 0.600 [0.400, 0.800] | 0.042 [0.021, 0.062] | 0.000 [0.000, 0.000] | 0.810 [0.761, 0.858] | 72 |
| give_me_some_credit | regb_m2 | 0.403 [0.306, 0.500] | 0.759 [0.586, 0.897] | 0.076 [0.052, 0.104] | 0.000 [0.000, 0.000] | 0.839 [0.800, 0.878] | 72 |
| give_me_some_credit | anchors | 0.028 [0.000, 0.069] | 0.000 [0.000, 0.000] | -0.046 [-0.069, -0.023] | 0.000 [0.000, 0.000] | 0.635 [0.572, 0.700] | 72 |
| give_me_some_credit | dice | 0.000 [0.000, 0.000] | n/a | 0.005 [-0.023, 0.032] | 0.000 [0.000, 0.000] | 0.802 [0.755, 0.850] | 72 |
| give_me_some_credit | lime | 0.014 [0.000, 0.042] | 1.000 | 0.022 [-0.005, 0.052] | 0.000 [0.000, 0.000] | 0.892 [0.860, 0.925] | 72 |
| give_me_some_credit | shap | 0.306 [0.222, 0.403] | 0.545 [0.364, 0.773] | 0.049 [0.024, 0.075] | 0.000 [0.000, 0.000] | 0.851 [0.813, 0.888] | 72 |
| give_me_some_credit | wachter | 0.000 [0.000, 0.000] | n/a | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.926 [0.870, 0.977] | 72 |
| heloc | regb_m1 | 0.000 [0.000, 0.000] | n/a | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 72 |
| heloc | regb_m2 | 0.000 [0.000, 0.000] | n/a | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 72 |
| heloc | anchors | 0.000 [0.000, 0.000] | n/a | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 72 |
| heloc | dice | 0.000 [0.000, 0.000] | n/a | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 61 |
| heloc | lime | 0.000 [0.000, 0.000] | n/a | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 72 |
| heloc | shap | 0.000 [0.000, 0.000] | n/a | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 72 |
| heloc | wachter | 0.000 [0.000, 0.000] | n/a | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 41 |
| south_german_credit | regb_m1 | 0.014 [0.000, 0.042] | 1.000 | 0.000 [-0.017, 0.017] | 0.000 [0.000, 0.000] | 0.917 [0.878, 0.950] | 72 |
| south_german_credit | regb_m2 | 0.042 [0.000, 0.083] | 1.000 [1.000, 1.000] | 0.010 [-0.007, 0.031] | 0.000 [0.000, 0.000] | 0.891 [0.846, 0.933] | 72 |
| south_german_credit | anchors | 0.028 [0.000, 0.069] | 0.000 [0.000, 0.000] | -0.113 [-0.192, -0.035] | 0.000 [0.000, 0.000] | 0.457 [0.395, 0.523] | 72 |
| south_german_credit | dice | 0.000 [0.000, 0.000] | n/a | -0.010 [-0.032, 0.012] | 0.000 [0.000, 0.000] | 0.875 [0.832, 0.914] | 72 |
| south_german_credit | lime | 0.042 [0.000, 0.083] | 0.000 [0.000, 0.000] | -0.022 [-0.061, 0.019] | 0.000 [0.000, 0.000] | 0.692 [0.658, 0.729] | 72 |
| south_german_credit | shap | 0.069 [0.014, 0.125] | 0.600 [0.400, 0.800] | 0.019 [0.000, 0.038] | 0.000 [0.000, 0.000] | 0.856 [0.811, 0.899] | 72 |
| south_german_credit | wachter | 0.000 [0.000, 0.000] | n/a | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.825 [0.738, 0.908] | 40 |

## Most frequent transitions (excluded -> replacement; top 5, pooled over models)

**default_of_credit_card_clients, real explainers:** `AGE -> LIMIT_BAL` (S -> N, weight 5.2, pairs 8); `AGE -> BILL_AMT1` (S -> N, weight 1.6, pairs 3); `SEX -> BILL_AMT1` (S -> N, weight 1.5, pairs 2); `SEX -> PAY_AMT2` (S -> N, weight 1.5, pairs 2); `AGE -> PAY_3` (S -> N, weight 1.3, pairs 3)

**default_of_credit_card_clients, Reg B commentary reasons:** `SEX -> EDUCATION` (S -> P, weight 7.9, pairs 15); `MARRIAGE -> PAY_AMT5` (S -> N, weight 4.7, pairs 25); `MARRIAGE -> EDUCATION` (S -> P, weight 4.5, pairs 14); `MARRIAGE -> BILL_AMT2` (S -> N, weight 3.8, pairs 13); `MARRIAGE -> BILL_AMT6` (S -> N, weight 2.6, pairs 14)

**give_me_some_credit, real explainers:** `age -> MonthlyIncome` (S -> P, weight 6.0, pairs 6); `age -> NumberOfDependents` (S -> P, weight 5.0, pairs 5); `age -> (none)` (S -> -, weight 3.0, pairs 3); `age -> NumberOfOpenCreditLinesAndLoans` (S -> N, weight 3.0, pairs 3); `age -> NumberOfTime30-59DaysPastDueNotWorse` (S -> N, weight 3.0, pairs 3)

**give_me_some_credit, Reg B commentary reasons:** `age -> MonthlyIncome` (S -> P, weight 20.0, pairs 21); `age -> NumberOfDependents` (S -> P, weight 10.5, pairs 11); `age -> NumberOfOpenCreditLinesAndLoans` (S -> N, weight 8.5, pairs 9); `age -> DebtRatio` (S -> N, weight 4.0, pairs 5); `age -> NumberOfTime60-89DaysPastDueNotWorse` (S -> N, weight 2.0, pairs 2)

**south_german_credit, real explainers:** `personal_status_sex -> other_debtors` (S -> N, weight 3.0, pairs 3); `age -> (none)` (S -> -, weight 1.0, pairs 1); `age -> duration` (S -> N, weight 1.0, pairs 1); `age -> other_installment_plans` (S -> N, weight 1.0, pairs 1); `age -> savings` (S -> P, weight 1.0, pairs 1)

**south_german_credit, Reg B commentary reasons:** `personal_status_sex -> savings` (S -> P, weight 2.0, pairs 2); `personal_status_sex -> employment_duration` (S -> P, weight 1.0, pairs 1); `personal_status_sex -> housing` (S -> P, weight 1.0, pairs 1)

Null baselines are excluded from the transition lists above (they are in the CSVs).

## Coverage

| count | value |
| --- | --- |
| instances | 2743 |
| matched_primary_audit | 1942 |
| not_in_both_tracks | 11 |

## Files

- `benchmark/outputs/cslaw_denial_r1/analysis/proxy_substitution/proxy_substitution_instances.csv`
- `benchmark/outputs/cslaw_denial_r1/analysis/proxy_substitution/proxy_substitution_summary.csv`
- `benchmark/outputs/cslaw_denial_r1/analysis/proxy_substitution/proxy_substitution_transitions.csv`
- `benchmark/outputs/cslaw_denial_r1/analysis/proxy_substitution/proxy_substitution_exclusions.csv`
- `benchmark/outputs/cslaw_denial_r1/analysis/proxy_substitution/proxy_substitution_summary.md`
