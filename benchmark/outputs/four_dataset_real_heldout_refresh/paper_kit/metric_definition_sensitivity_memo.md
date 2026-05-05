# LegalBench-XAI Metric Definition Sensitivity Memo

- Experiment: `legalbench_xai_four_dataset_real`
- Generated at (UTC): `2026-03-15T23:16:10.379973+00:00`

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
| observed_css | dice | 0.895 | no |
| reconstructed_css | dice | 0.895 | no |
| verifiability_focused_css | dice | 0.918 | no |
| challenge_focused_css | dice | 0.895 | no |
| specificity_light_css | dice | 0.926 | no |

## ALF Definition Profiles

| Scenario | Leader | Score | Leader Changed |
| --- | --- | ---: | --- |
| observed_alf | wachter | 0.492 | no |
| balanced_arithmetic_alf | wachter | 0.804 | no |
| legality_dominant_alf | wachter | 0.887 | no |
| feasibility_dominant_alf | wachter | 0.807 | no |
| minimality_dominant_alf | wachter | 0.770 | no |
| invalidity_penalized_alf | wachter | 0.798 | no |
| hard_failure_penalized_alf | wachter | 0.796 | no |

## Writing Guidance

- Use CSS profile stability to argue that contestability is not an artifact of one arbitrary subscore weighting.
- Use ALF profile instability, if present, as a principled caution that legality-aware recourse metrics encode real normative choices.
- Keep this memo separate from generic metric-weight sensitivity: this one is about construct validity inside the benchmark dimensions themselves.
