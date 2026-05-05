# LegalBench-XAI Metric Definition Sensitivity Memo

- Experiment: `legalbench_xai_default_credit_real_threshold040`
- Generated at (UTC): `2026-03-15T19:19:37.140762+00:00`

## Main Readout

- CSS observed-score leader: `dice`.
- CSS alternative definitions preserving the same leader: `4` / `4`.
- ALF observed-score leader: `wachter`.
- ALF alternative definitions preserving the same leader: `4` / `6`.

## Interpretation

This memo asks whether the benchmark conclusions depend too heavily on one exact formula for Contestability Support Score or Actionability-Legality Filter.
If the same explainer remains the leader under multiple plausible definitions, that strengthens the construct-validity story. If the leader changes, we should describe the metric as design-sensitive rather than objective.

## CSS Definition Profiles

| Scenario | Leader | Score | Leader Changed |
| --- | --- | ---: | --- |
| observed_css | dice | 0.982 | no |
| reconstructed_css | dice | 0.982 | no |
| verifiability_focused_css | dice | 0.986 | no |
| challenge_focused_css | dice | 0.984 | no |
| specificity_light_css | dice | 0.983 | no |

## ALF Definition Profiles

| Scenario | Leader | Score | Leader Changed |
| --- | --- | ---: | --- |
| observed_alf | wachter | 0.533 | no |
| balanced_arithmetic_alf | wachter | 0.825 | no |
| legality_dominant_alf | dice | 0.897 | yes |
| feasibility_dominant_alf | dice | 0.824 | yes |
| minimality_dominant_alf | wachter | 0.800 | no |
| invalidity_penalized_alf | wachter | 0.825 | no |
| hard_failure_penalized_alf | wachter | 0.825 | no |

## Writing Guidance

- Use CSS profile stability to argue that contestability is not an artifact of one arbitrary subscore weighting.
- Use ALF profile instability, if present, as a principled caution that legality-aware recourse metrics encode real normative choices.
- Keep this memo separate from generic metric-weight sensitivity: this one is about construct validity inside the benchmark dimensions themselves.
