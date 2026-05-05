# LegalBench-XAI Discussion Memo

## Main Interpretation

The current benchmark run already supports the core paper claim that legally meaningful explanation quality is multi-dimensional. `shap` leads on faithfulness, `lime` leads on stability, `anchors` leads on comprehensibility, `dice` leads on contestability support, and `wachter` leads on actionability while `wachter` leads on legality-aware recourse.

## Draft Discussion Points

The emerging pattern is that attribution-style explanations are strongest when the question is whether the explanation tracks the model and remains locally consistent, while recourse-style explanations are strongest when the question is whether an applicant can act on or challenge the decision.
This tradeoff is the center of the paper's contribution: a method can be faithful and stable without being the best tool for contestability or actionability, and a recourse method can be practically useful while still requiring legality checks.
The bootstrap intervals in the current results pack reinforce that we should write this as a tradeoff story rather than a search for a universal winner. Some leader identities are robust, but the intervals still argue for caution when turning leaderboard gaps into strong claims.
The comprehensibility results also matter because they suggest that cognitively simple explanations should not be treated as automatically superior. In the current benchmark run, the easiest explanation family to read is not the same family that dominates on faithfulness or on applicant-facing recourse.
Primary-vs-audit score shifts remain small on average, with the largest mean absolute shift on `stability` at `0.068`. Primary and audit track leader identities are stable for `1` of `6` metrics, with instability concentrated in `faithfulness`, `stability`, `css`, `actionability`, `alf`. The least stable metric is `css` at same-leader rate `0.750`.

## Writing Guidance

- Use the tradeoff framing instead of searching for a single overall winner.
- Keep faithfulness and stability together as model-facing reliability dimensions.
- Keep CSS and actionability together as applicant-facing usefulness dimensions.
- Discuss ALF as a necessary constraint on recourse rather than as a stand-alone interpretability metric.

## Next Empirical Checks

- Increase the denied-instance budget on the current real-data runs so the current ordering is supported by a materially larger adverse subset.
- Prioritize sensitivity analysis and higher denied-instance budgets; a fifth dataset would help, but it is no longer the single highest-leverage gap.
- Test whether the current primary-versus-audit divergences persist once the benchmark covers a broader credit dataset set.
