# LegalBench-XAI Metric Definition Sensitivity Memo

- Experiment: `legalbench_xai_cross_domain_strongest_real_template`
- Generated at (UTC): `2026-04-26T23:43:09.734784+00:00`

## Main Readout

- CSS observed-score leader: `dice`.
- CSS alternative definitions preserving the same leader: `4` / `4`.
- ALF observed-score leader: `wachter`.
- ALF alternative definitions preserving the same leader: `6` / `6`.

## Interpretation

This memo asks whether the benchmark conclusions depend too heavily on one exact formula for Contestability Support Score or Actionability-Legality Filter.
If the same explainer remains the leader under multiple plausible definitions, that strengthens the construct-validity story. If the leader changes, we should describe the metric as design-sensitive rather than objective.

## CSS Definition Profiles

| Scenario | Leader | Score | Leader Changed |
| --- | --- | ---: | --- |
| observed_css | dice | 0.796 | no |
| reconstructed_css | dice | 0.796 | no |
| verifiability_focused_css | dice | 0.837 | no |
| challenge_focused_css | dice | 0.810 | no |
| specificity_light_css | dice | 0.827 | no |

## ALF Definition Profiles

| Scenario | Leader | Score | Leader Changed |
| --- | --- | ---: | --- |
| observed_alf | wachter | 0.329 | no |
| balanced_arithmetic_alf | wachter | 0.727 | no |
| legality_dominant_alf | wachter | 0.818 | no |
| feasibility_dominant_alf | wachter | 0.669 | no |
| minimality_dominant_alf | wachter | 0.730 | no |
| invalidity_penalized_alf | wachter | 0.669 | no |
| hard_failure_penalized_alf | wachter | 0.669 | no |

## Writing Guidance

- Use CSS profile stability to argue that contestability is not an artifact of one arbitrary subscore weighting.
- Use ALF profile instability, if present, as a principled caution that legality-aware recourse metrics encode real normative choices.
- Keep this memo separate from generic metric-weight sensitivity: this one is about construct validity inside the benchmark dimensions themselves.
