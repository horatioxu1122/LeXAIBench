# Rating rubric: rendered statements of reasons (research instrument)

This is a research rubric. It is not legal advice and not a compliance test. Each
item ties a notice to a legal function that we (the authors) read statements of
reasons as serving. The citations show where that reading comes from. They do
not claim that any rendered notice would or would not comply with the law. Please
rate from your own professional judgment.

## 1. What you will see

The packet `notices_for_rating.md` (also `notices_for_rating.csv`) contains about
48 notices. Each notice has:

- a neutral header, "Statement of specific reasons for credit denial (research
  rendering)", an opaque notice ID (e.g., `N48213`) and a generic context line
  (e.g., "consumer credit line application");
- up to four **principal reasons**, in the order the explanation ranked them;
- for some notices, a section **"What could change the outcome"**, which lists
  proposed changes as "field: from X to Y";
- an **"Applicant record (fields cited above)"** block, which gives the
  applicant's recorded values for the fields the notice cites.

How the notices were made:

- They were generated automatically from explanations of statistical models.
- The models were trained on public research datasets: historical data, some of
  it decades old and not all of it from the United States.
- They are not real notices, and no applicant received them.
- Please rate each one as if it were a statement of specific reasons given to a
  credit applicant in the United States under the Equal Credit Opportunity Act
  (ECOA) and Regulation B.

Some practical points:

- **Wording.** Field names were turned into plain language mechanically, so some
  reasons read awkwardly (e.g., "Number of inquiries last 6 months contributed to
  the denial"). Rate the substance, not the grammar. Wording matters only where
  it stops a reader from telling what the reason refers to; that is relevant to
  R1 and R5.
- **Coded values.** Some datasets store categories as numeric codes, and the
  applicant record shows the code. A code is an artifact of our rendering.
  - For R3, judge whether the *field* could be checked and corrected, not
    whether the code is readable.
  - Mention it in the comment if it matters.
- **Unverified values.** Values are shown as recorded in the research dataset,
  and nobody has verified them.
- **Hidden information.** You do not see model scores, cutoffs, the method that
  produced the explanation, or the applicant's identity.

## 2. What you must not do

- **Do not try to work out which method or model produced a notice.** Some
  notices have a "What could change the outcome" section and others do not.
  Rate each notice on its own content.
- Do not compare notices with each other, and do not adjust your ratings to
  balance them across the packet.
- Do not look up the datasets, the code or the benchmark scores. Do not discuss
  individual notices with other raters until everyone has submitted final
  ratings. (Discussion during calibration, below, is fine.)
- Do not change a rating after you have seen the benchmark results or the key
  file (`notices_key.csv`).

## 3. Procedure and time

1. **Calibration (about 15 minutes, together).** The coordinator renders a short
   practice set from a different seed, for example
   `notices.py ... --n 4 --seed 999 --out-dir practice/`. Raters discuss the
   practice notices against the items below. Practice notices are not analysed.
2. **Independent rating.** Rate the main packet alone, in the order given.
   Expect about **1-2 minutes per notice**, so about 60-90 minutes for 48
   notices. Take a break about halfway through.
3. **Recording.** Copy `ratings_template.csv` and rename the copy, e.g. to
   `ratings_<initials>.csv`. Then:
   - Put your initials in `rater_id` on every row.
   - Fill one row per notice ID. Do not add, delete or reorder rows.

| Column | Accepted entries |
| --- | --- |
| `R1_specificity` | `0`, `1`, `2`, or `unsure` |
| `R2_prohibited_basis` | `yes`, `no`, `unsure` |
| `R3_rectification` | `0`, `1`, `2`, or `unsure` |
| `R4_recourse` | `0`, `1`, `2`, `N/A`, or `unsure` |
| `R5_overall` | `1`-`5`, or `unsure` |
| `comment` | free text (optional; required for every `unsure` and for R2 = `yes`) |

**Handling unclear items:**

- Use whole numbers only; half points are treated as invalid.
- If you hesitate between two anchors, pick the one that fits better and say
  why in `comment`.
- Use `unsure` only when you cannot decide at all. The analysis counts `unsure`
  as missing for the 0-2 and 1-5 scales, and as a separate category for R2.
- Leave a cell blank only if you could not rate the notice at all (e.g., it is
  unreadable). Explain why in `comment`.
- `N/A` is valid only for R4. Elsewhere it is treated as missing.

## 4. Items

### R1 Specificity (0-2)

*Question:* Do the stated reasons give specific principal reasons for the denial,
rather than generic statements?

*Legal function (our reading):* the statement of specific reasons under ECOA,
15 U.S.C. 1691(d)(2)-(3), and Regulation B, 12 CFR 1002.9(a)(2) and (b)(2).

- **12 CFR 1002.9(b)(2)** requires that the statement be specific and indicate
  the principal reason(s) for the adverse action. It also says that statements
  based on the creditor's internal standards or policies, or on the applicant's
  failure to achieve a qualifying score on the creditor's scoring system, are
  insufficient.
- **Official Interpretation comment 9(b)(2)-3** (Supplement I to Part 1002)
  says a creditor need not describe how or why a factor adversely affected the
  applicant. For example, naming "length of residence" can suffice. So a reason
  that clearly names the factor is specific even if it does not say which way
  the factor pushed.
- **Comment 9(b)(2)-1** indicates that disclosing more than four reasons is
  not likely to be helpful. The notices therefore show at most four.

You cannot check whether the cited factors really were the model's principal
factors; the benchmark assesses that separately. Rate whether each reason is
specific and is framed as a reason for the denial.

- **2** - Every reason (or all but a trivial one) names an identifiable factor
  that could plausibly be a principal reason for denial. None is generic. None
  describes a factor as having *supported approval*.
- **1** - Mixed. At least one specific reason for denial is given, but others
  are one or more of the following:
  - generic or vague;
  - duplicative;
  - framed as favouring approval (e.g., "... supported approval");
  - worded so that the factor is hard to identify.
- **0** - No specific reason for denial is given. This covers notices with no
  reasons, reasons that are only generic (internal standards, an insufficient
  score, "credit profile"), unintelligible reasons, and reasons that only name
  factors said to have supported approval.

### R2 Prohibited basis or apparent proxy (yes / no / unsure)

*Question:* Does any reason or proposed change cite a protected characteristic,
or a factor that appears to stand in for one?

*Legal function (our reading):* anti-discrimination.

- **The prohibited bases.** ECOA, 15 U.S.C. 1691(a), and Regulation B,
  12 CFR 1002.2(z) ("prohibited basis"), cover:
  - race, color, religion, national origin, sex, marital status, and age
    (provided the applicant has the capacity to contract);
  - receipt of income from a public assistance program;
  - the good-faith exercise of rights under the Consumer Credit Protection Act.
- **Mortgage notices.** For residential real-estate-related transactions, the
  Fair Housing Act (42 U.S.C. 3605) adds further characteristics, such as
  familial status and disability (handicap). Flag those too.
- **Proxies.** Regulation B does not define "proxy". Here it is a research
  category for factors that you judge would commonly be understood to stand in
  for a prohibited basis. Examples: a combined sex/marital-status code;
  foreign-worker status as a stand-in for national origin; neighbourhood or
  geographic identifiers as a stand-in for race.

R2 records whether such a factor is *cited*, not whether its use would be
lawful. For example, Regulation B permits age as a variable in an empirically
derived, demonstrably and statistically sound scoring system, subject to
conditions (12 CFR 1002.6(b)(2)). A notice that cites age should still be marked
`yes`.

- **yes** - At least one reason or proposed change refers to a prohibited basis
  or an apparent proxy for one. Name the field in `comment`.
- **no** - None does.
- **unsure** - A cited factor might be a proxy, but you are not confident.
  Explain in `comment`.

### R3 Rectification (0-2)

*Question:* Could the applicant check the information the notice cites and, if
it is wrong, get it corrected?

*Legal function (our reading):* rectification and contestation.

- **Legislative history.** The history of the 1976 ECOA amendments
  (S. Rep. No. 94-589) describes the statement of reasons as letting a rejected
  applicant learn where their credit standing is deficient. It also lets them
  correct a decision that rested on mistaken or incomplete information. (This
  is our paraphrase.)
- **FCRA.** Where a decision relies on a consumer report, the Fair Credit
  Reporting Act gives related rights: 15 U.S.C. 1681m(a) (adverse action
  notice) and 15 U.S.C. 1681i (disputing accuracy).

Use the applicant record block to see which value was used.

- **2** - Every cited field could be checked against information the applicant
  has or can obtain. Sources include their own records, account statements, a
  consumer report, or the application they submitted. If a value were wrong, it
  would be clear what to dispute or correct.
- **1** - Some cited fields can be checked, but others cannot. Fields that
  cannot be checked include opaque composite scores, model-derived or ambiguous
  fields, and values whose meaning the applicant could not know. Also use 1
  when it is unclear which value the creditor used.
- **0** - The applicant could not realistically check or correct any of the
  cited information. Examples: only opaque scores are cited, or no fields are
  cited at all.

A field that cannot be changed (e.g., recorded age) can still be *checked*: is
the recorded value right? Rate checkability and correctability of the recorded
value, not whether the applicant can change the underlying fact.

### R4 Recourse (0-2, or N/A)

*Question:* Is every suggested change lawful to suggest, feasible, and within
the applicant's control?

*Legal function (our reading):* Regulation B does not require that a notice
suggest changes, so `N/A` is **not** a deficiency. R4 rates only the quality of
the suggestions a notice does make. Consider changes listed under "What could
change the outcome" and reasons phrased as a change ("changing X may improve the
outcome").

- **N/A** - The notice suggests no change.
- **2** - Every suggested change is lawful to suggest, realistically feasible,
  and within the applicant's control. It does not involve an immutable fact, a
  protected characteristic or apparent proxy, or a term the lender sets alone.
  A requested amount or term that the applicant could choose when reapplying
  counts as within their control.
- **1** - Some suggestions are acceptable, but others are one or more of:
  - infeasible or implausible in size;
  - outside the applicant's control;
  - vague.
- **0** - The suggestions are mainly one or more of:
  - unlawful or inappropriate to suggest, because they would mean changing a
    protected characteristic or an apparent proxy;
  - impossible, because they would mean changing an immutable fact or rewriting
    past events;
  - entirely outside the applicant's control.

Some fields count time since an event (e.g., months since the most recent
delinquency). Waiting can change them, so treat them as feasible over time.
Counts of past events can change only by correcting errors or as old events age
out.

### R5 Overall adequacy as a statement of reasons (1-5)

*Question:* Taking everything into account, how adequate is this notice as a
statement of specific reasons that an applicant would receive?

- **5** - Clearly adequate. It gives specific principal reasons that an
  applicant could understand and act on, including checking or correcting them.
- **4** - Adequate, with minor issues.
- **3** - Borderline.
- **2** - Inadequate in important respects.
- **1** - Clearly inadequate. Examples: no specific reasons, unintelligible
  reasons, or a prohibited basis cited with nothing else of substance.

R5 is your overall judgment, not an average of R1-R4. Helpful suggestions may
count in a notice's favour, but recourse is not required (see R4). Do not give a
notice a higher rating only because it has a "What could change the outcome"
section.

## 5. Disclosure (for the paper)

If the raters are authors of the paper, these ratings are a **legal assessment
by the authors, not a human-subjects study**. The paper must disclose this,
including:

- who rated and what legal training they have;
- that they were co-authors who knew the study's aims;
- that notices were rendered without method or model names;
- that the method family was not fully blindable, because the recourse section
  identifies counterfactual explanations. A reasons-only condition
  (`notices.py --strip-recourse`) can be reported as a robustness check.

If raters from outside the author team are recruited, confirm the institution's
human-subjects (IRB) requirements before collecting any ratings.

## 6. Analysis plan (fixed before ratings are collected)

`rating_analysis.py` computes the following:

- **Data quality:** missing, `unsure`, `N/A` and invalid entries per rater and
  item.
- **Inter-rater reliability:**
  - Cohen's kappa for every rater pair on every item;
  - quadratic-weighted kappa for every pair on the ordinal items (R1, R3, R4,
    R5);
  - Krippendorff's alpha across all raters: nominal for R2 and for R4
    applicability, ordinal and interval for R1, R3, R4 and R5;
  - seeded bootstrap 95% CIs.
  - R2 is analysed twice: with `unsure` as a third category, and as yes/no with
    `unsure` treated as missing.
- **Consensus rating per notice:** the mean across raters. For R2, the share of
  yes among yes/no answers. For R4, `N/A` when most raters marked `N/A`.
- **Mean consensus ratings with bootstrap 95% CIs:**
  - per explainer, per method family, per model and per dataset;
  - resampling notices, with a seeded RNG;
  - null baselines are reported separately and never ranked.
- **Spearman correlations:**
  - between consensus ratings and the benchmark scores of the same explanation;
  - bootstrap CIs and seeded permutation p-values;
  - Holm adjustment over the pre-specified pairs below, computed on notices from
    real explainers.

Pre-specified pairs (rating, benchmark metric, expected sign):

| Rating | Metric | Expected |
| --- | --- | --- |
| R1 Specificity | CSS reason_specificity component | + |
| R1 Specificity | comprehensibility | + |
| R3 Rectification | CSS verifiability component | + |
| R3 Rectification | CSS correctability component | + |
| R3 Rectification | CSS | + |
| R4 Recourse | ALF (counterfactual notices only) | + |
| R5 Overall | CSS | + |
| R5 Overall | CSS without decision_sensitivity_visibility (renormalized) | + |
| R5 Overall | comprehensibility | + |
| R5 Overall | faithfulness | none expected (raters cannot observe faithfulness; reported as a check) |

All other rating-metric pairs are exploratory.
