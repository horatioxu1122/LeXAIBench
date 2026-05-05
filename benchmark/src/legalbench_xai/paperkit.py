from pathlib import Path
from typing import Any

from .paths import repo_relative_path


def write_paper_kit(
    target_dir: Path,
    payload: dict[str, Any],
    analysis: dict[str, Any],
) -> dict[str, Any]:
    paper_dir = target_dir / "paper_kit"
    paper_dir.mkdir(parents=True, exist_ok=True)

    findings_path = paper_dir / "results_memo.md"
    sensitivity_path = paper_dir / "metric_sensitivity_memo.md"
    definition_sensitivity_path = paper_dir / "metric_definition_sensitivity_memo.md"
    track_policy_path = paper_dir / "track_policy_sensitivity_memo.md"
    discussion_path = paper_dir / "discussion_memo.md"
    limitations_path = paper_dir / "limitations_memo.md"
    main_tables_path = paper_dir / "main_tables.tex"
    appendix_tables_path = paper_dir / "appendix_tables.tex"

    findings_path.write_text(_build_results_memo(payload, analysis), encoding="utf-8")
    sensitivity_path.write_text(
        _build_metric_sensitivity_memo(payload, analysis),
        encoding="utf-8",
    )
    definition_sensitivity_path.write_text(
        _build_metric_definition_sensitivity_memo(payload, analysis),
        encoding="utf-8",
    )
    track_policy_path.write_text(
        _build_track_policy_sensitivity_memo(payload, analysis),
        encoding="utf-8",
    )
    discussion_path.write_text(_build_discussion_memo(payload, analysis), encoding="utf-8")
    limitations_path.write_text(_build_limitations_memo(payload, analysis), encoding="utf-8")
    main_tables_path.write_text(_build_main_tables_tex(payload, analysis), encoding="utf-8")
    appendix_tables_path.write_text(
        _build_appendix_tables_tex(payload, analysis),
        encoding="utf-8",
    )

    return {
        "directory": repo_relative_path(paper_dir),
        "files": {
            "results_memo_markdown": repo_relative_path(findings_path),
            "metric_sensitivity_memo_markdown": repo_relative_path(sensitivity_path),
            "metric_definition_sensitivity_memo_markdown": repo_relative_path(
                definition_sensitivity_path
            ),
            "track_policy_sensitivity_memo_markdown": repo_relative_path(track_policy_path),
            "discussion_memo_markdown": repo_relative_path(discussion_path),
            "limitations_memo_markdown": repo_relative_path(limitations_path),
            "main_tables_tex": repo_relative_path(main_tables_path),
            "appendix_tables_tex": repo_relative_path(appendix_tables_path),
        },
    }


def _build_results_memo(payload: dict[str, Any], analysis: dict[str, Any]) -> str:
    by_explainer = analysis["by_explainer"]
    faithfulness_leader = _best_row(by_explainer, "mean_faithfulness")
    stability_leader = _best_row(by_explainer, "mean_stability")
    comprehensibility_leader = _best_row(by_explainer, "mean_comprehensibility")
    css_leader = _best_row(by_explainer, "mean_css")
    actionability_leader = _best_row(by_explainer, "mean_actionability")
    alf_leader = _best_row(by_explainer, "mean_alf")
    pairwise = {row["metric"]: row for row in analysis["pairwise_champions"]}
    track_winners = _dataset_track_winners(analysis["by_dataset_track_explainer"])
    stable_track_rankings = _has_stable_track_rankings(track_winners)
    track_clause = _track_policy_clause(analysis, stable_track_rankings)

    lines = [
        "# LegalBench-XAI Results Memo",
        "",
        f"- Experiment: `{payload['experiment_name']}`",
        f"- Generated at (UTC): `{payload['generated_at_utc']}`",
        "",
        "## Headline Findings",
        "",
        _leader_headline(
            "Overall faithfulness leader",
            faithfulness_leader,
            "mean_faithfulness",
            "mean faithfulness",
        ),
        _leader_headline(
            "Overall stability leader",
            stability_leader,
            "mean_stability",
            "mean stability",
        ),
        _leader_headline(
            "Overall comprehensibility leader",
            comprehensibility_leader,
            "mean_comprehensibility",
            "mean comprehensibility",
        ),
        _leader_headline("Overall CSS leader", css_leader, "mean_css", "mean CSS"),
        _leader_headline(
            "Overall actionability leader",
            actionability_leader,
            "mean_actionability",
            "mean actionability",
        ),
        _leader_headline("Overall ALF leader", alf_leader, "mean_alf", "mean ALF"),
        _pairwise_headline("Pairwise faithfulness leader", pairwise.get("faithfulness")),
        _pairwise_headline("Pairwise stability leader", pairwise.get("stability")),
        _pairwise_headline(
            "Pairwise comprehensibility leader",
            pairwise.get("comprehensibility"),
        ),
        _pairwise_headline("Pairwise CSS leader", pairwise.get("css")),
        _pairwise_headline("Pairwise actionability leader", pairwise.get("actionability")),
        _pairwise_headline("Pairwise ALF leader", pairwise.get("alf")),
        "",
        "## Draft Results Language",
        "",
        _draft_results_paragraph(
            faithfulness_leader=faithfulness_leader,
            stability_leader=stability_leader,
            comprehensibility_leader=comprehensibility_leader,
            css_leader=css_leader,
            actionability_leader=actionability_leader,
            alf_leader=alf_leader,
            track_clause=track_clause,
        ),
        "",
        "## Suggested Claims to Validate in the Full Benchmark",
        "",
        "- Faithfulness should be discussed separately from contestability, because a method can align well with model behavior and still be weak as applicant-facing guidance.",
        "- Stability should be discussed separately from faithfulness, because a method can match model behavior on one instance while still being brittle across similar applicants or random seeds.",
        "- Comprehensibility should be discussed separately from both faithfulness and actionability, because the easiest explanation to read may not be the most faithful or the most useful for recourse.",
        "- Recourse-oriented methods appear to dominate attribution methods on applicant-facing actionability.",
        "- Contestability-support rankings are close enough that CSS should be discussed as a tradeoff, not only as a single leaderboard.",
        "- Primary-versus-audit track effects are limited in the current benchmark run, but this should be retested on full datasets.",
        "- Pairwise win tables are likely to be more persuasive than raw averages when arguing robustness across datasets and model families.",
        "",
        "## Uncertainty Note",
        "",
        _bootstrap_note_line(analysis),
        "- These intervals are computed by resampling the benchmark structure rather than treating all scored groups as IID, so they are useful for robustness checks but are still not full applicant-level or fully model-based uncertainty estimates.",
        "",
        "## Caveat",
        "",
        _results_caveat(payload),
        "",
    ]
    return "\n".join(lines)


def _build_main_tables_tex(payload: dict[str, Any], analysis: dict[str, Any]) -> str:
    by_explainer = analysis["by_explainer"]
    winners = _dataset_track_winners(analysis["by_dataset_track_explainer"])
    lines = [
        "% Auto-generated by LegalBench-XAI paper kit",
        "% Requires \\usepackage{booktabs}",
        "",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Overall explainer performance in the current LegalBench-XAI run.}",
        "\\label{tab:overall_leaderboard}",
        "\\begin{tabular}{lrrrrrrrrr}",
        "\\toprule",
        "Explainer & Groups & Inst. & Faithfulness & Stability & Comprehensibility & CSS & Actionability & ALF & Invalid Recourse \\\\",
        "\\midrule",
    ]
    for row in by_explainer:
        lines.append(
            (
                "{explainer} & {groups} & {instances} & {faithfulness} & {stability} & {comprehensibility} & {css} & "
                "{actionability} & {alf} & {invalid_rate} \\\\"
            ).format(
                explainer=_tex_escape(row["explainer_id"]),
                groups=row["group_count"],
                instances=row["total_instances"],
                faithfulness=_fmt(row["mean_faithfulness"]),
                stability=_fmt(row["mean_stability"]),
                comprehensibility=_fmt(row["mean_comprehensibility"]),
                css=_fmt(row["mean_css"]),
                actionability=_fmt(row["mean_actionability"]),
                alf=_fmt(row["mean_alf"]),
                invalid_rate=_fmt(row["mean_invalid_recourse_rate"]),
            )
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Best-performing explainer by dataset, track, and metric.}",
            "\\label{tab:dataset_track_winners}",
            "\\begin{tabular}{llllllll}",
            "\\toprule",
            "Dataset & Track & Best Faithfulness & Best Stability & Best Comprehensibility & Best CSS & Best Actionability & Best ALF \\\\",
            "\\midrule",
        ]
    )
    for row in winners:
        lines.append(
            (
                "{dataset} & {track} & {faithfulness} ({faithfulness_score}) & "
                "{stability} ({stability_score}) & "
                "{comprehensibility} ({comprehensibility_score}) & "
                "{css} ({css_score}) & "
                "{actionability} ({actionability_score}) & "
                "{alf} ({alf_score}) \\\\"
            ).format(
                dataset=_tex_escape(row["dataset_id"]),
                track=_tex_escape(row["track"]),
                faithfulness=_tex_escape(row["best_faithfulness_explainer"]),
                faithfulness_score=_fmt(row["best_faithfulness_score"]),
                stability=_tex_escape(row["best_stability_explainer"]),
                stability_score=_fmt(row["best_stability_score"]),
                comprehensibility=_tex_escape(row["best_comprehensibility_explainer"]),
                comprehensibility_score=_fmt(row["best_comprehensibility_score"]),
                css=_tex_escape(row["best_css_explainer"]),
                css_score=_fmt(row["best_css_score"]),
                actionability=_tex_escape(row["best_actionability_explainer"]),
                actionability_score=_fmt(row["best_actionability_score"]),
                alf=_tex_escape(row["best_alf_explainer"]),
                alf_score=_fmt(row["best_alf_score"]),
            )
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def _build_metric_sensitivity_memo(payload: dict[str, Any], analysis: dict[str, Any]) -> str:
    summary = analysis.get("metric_rank_correlation_summary", {})
    strongest = summary.get("strongest_positive")
    weakest = summary.get("weakest")
    rows = analysis.get("metric_rank_correlation", [])
    random_rows = analysis.get("metric_weight_random_summary", [])
    profile_rows = analysis.get("metric_weight_profiles", [])
    ablation_rows = [row for row in analysis.get("metric_ablation_profiles", []) if row.get("rank") == 1]
    ablation_summary = analysis.get("metric_ablation_summary", {})
    loocv_rows = analysis.get("leave_one_dataset_out_leaders", [])

    lines = [
        "# LegalBench-XAI Metric Sensitivity Memo",
        "",
        f"- Experiment: `{payload['experiment_name']}`",
        f"- Generated at (UTC): `{payload['generated_at_utc']}`",
        "",
        "## Main Readout",
        "",
        (
            f"- Strongest positive rank correlation: `{strongest['metric_a']}` vs "
            f"`{strongest['metric_b']}` = `{_fmt(strongest['spearman_rank_corr'])}`."
            if strongest
            else "- Strongest positive rank correlation: n/a."
        ),
        (
            f"- Weakest rank correlation: `{weakest['metric_a']}` vs "
            f"`{weakest['metric_b']}` = `{_fmt(weakest['spearman_rank_corr'])}`."
            if weakest
            else "- Weakest rank correlation: n/a."
        ),
        "",
        "## Interpretation",
        "",
        (
            "High positive rank correlation suggests two metrics are ordering the explainer families similarly and may be partly redundant in the current benchmark slice."
        ),
        (
            "Low or near-zero rank correlation suggests the metrics are capturing genuinely different aspects of explanation quality, which strengthens the case for a multi-dimensional benchmark."
        ),
        (
            "Pairs with fewer than three shared explainers should be treated as descriptive only; this matters especially for ALF, which is only defined for the recourse methods."
        ),
        "",
        "## Weight Sensitivity",
        "",
        "Randomized universal-weight analysis asks how often each explainer would win if we repeatedly changed the relative importance of faithfulness, stability, comprehensibility, CSS, and actionability.",
        "",
        "| Explainer | Champion Frequency | Mean Rank | Draws |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in random_rows:
        lines.append(
            (
                "| {explainer} | {freq} | {rank} | {draws} |"
            ).format(
                explainer=row["explainer_id"],
                freq=_fmt(row["champion_frequency"]),
                rank=_fmt(row["mean_rank"]),
                draws=row["num_draws"],
            )
        )

    lines.extend(
        [
            "",
            "Fixed-profile analysis gives a more interpretable view of which explainer wins under specific evaluation priorities.",
            "",
            "| Scenario | Scope | Explainer | Composite Score | Rank |",
            "| --- | --- | --- | ---: | ---: |",
        ]
    )
    for row in profile_rows:
        lines.append(
            (
                "| {scenario} | {scope} | {explainer} | {score} | {rank} |"
            ).format(
                scenario=row["scenario_name"],
                scope=row["scope"],
                explainer=row["explainer_id"],
                score=_fmt(row["composite_score"]),
                rank=row["rank"],
            )
        )

    lines.extend(
        [
            "",
            "## Metric Ablation",
            "",
            (
                f"- Balanced composite baseline leader: `{ablation_summary['baseline_leader']['explainer_id']}`."
                if ablation_summary.get("baseline_leader")
                else "- Balanced composite baseline leader: n/a."
            ),
            (
                f"- Leave-one-metric-out scenarios preserving the same leader: "
                f"`{ablation_summary['leader_stability_count']}` / "
                f"`{ablation_summary['scenario_count']}`."
                if ablation_summary.get("baseline_leader")
                else "- Leave-one-metric-out leader stability: n/a."
            ),
            "",
            "| Scenario | Omitted Metric | Leader | Composite Score | Leader Changed |",
            "| --- | --- | --- | ---: | --- |",
        ]
    )
    for row in ablation_rows:
        lines.append(
            (
                "| {scenario} | {omitted} | {leader} | {score} | {changed} |"
            ).format(
                scenario=row["scenario_name"],
                omitted=row["omitted_metric"] or "none",
                leader=row["explainer_id"],
                score=_fmt(row["composite_score"]),
                changed="yes" if row.get("leader_changed") else "no",
            )
        )

    lines.extend(
        [
            "",
            "## Leave-One-Dataset-Out",
            "",
            "This table shows which explainer leads each metric when one dataset is omitted from the benchmark. It is a direct check on whether the headline findings depend too heavily on any single dataset.",
            "",
            "| Omitted Dataset | Metric | Leader | Score |",
            "| --- | --- | --- | ---: |",
        ]
    )
    for row in loocv_rows:
        lines.append(
            (
                "| {dataset} | {metric} | {leader} | {score} |"
            ).format(
                dataset=row["omitted_dataset_id"],
                metric=row["metric"],
                leader=row["explainer_id"],
                score=_fmt(row["score"]),
            )
        )

    lines.extend(
        [
            "",
            "## Correlation Table",
            "",
            "| Metric A | Metric B | Spearman Rank Correlation | Shared Explainers |",
        "| --- | --- | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            (
                "| {metric_a} | {metric_b} | {corr} | {shared} |"
            ).format(
                metric_a=row["metric_a"],
                metric_b=row["metric_b"],
                corr=_fmt(row["spearman_rank_corr"]),
                shared=row["shared_explainers"],
            )
        )

    lines.extend(
        [
            "",
            "## Writing Guidance",
            "",
            "- Use the weight-sensitivity table to argue that the benchmark conclusions do not depend on one arbitrary metric weighting.",
            "- Use this memo to justify why the benchmark should report multiple metrics instead of collapsing everything into one leaderboard.",
            "- If two metrics correlate strongly, present them as related but not identical dimensions.",
            "- If two metrics correlate weakly, use that gap as evidence that standard XAI evaluation misses important parts of applicant-facing explanation quality.",
            "",
        ]
    )
    return "\n".join(lines)


def _build_metric_definition_sensitivity_memo(
    payload: dict[str, Any],
    analysis: dict[str, Any],
) -> str:
    css_rows = [row for row in analysis.get("css_definition_profiles", []) if row.get("rank") == 1]
    alf_rows = [row for row in analysis.get("alf_definition_profiles", []) if row.get("rank") == 1]
    css_summary = analysis.get("css_definition_summary", {})
    alf_summary = analysis.get("alf_definition_summary", {})

    lines = [
        "# LegalBench-XAI Metric Definition Sensitivity Memo",
        "",
        f"- Experiment: `{payload['experiment_name']}`",
        f"- Generated at (UTC): `{payload['generated_at_utc']}`",
        "",
        "## Main Readout",
        "",
        (
            f"- CSS observed-score leader: `{css_summary['baseline_leader']['explainer_id']}`."
            if css_summary.get("baseline_leader")
            else "- CSS observed-score leader: n/a."
        ),
        (
            f"- CSS alternative definitions preserving the same leader: "
            f"`{css_summary['leader_stability_count']}` / "
            f"`{css_summary['scenario_count']}`."
            if css_summary.get("baseline_leader")
            else "- CSS alternative-definition stability: n/a."
        ),
        (
            f"- ALF observed-score leader: `{alf_summary['baseline_leader']['explainer_id']}`."
            if alf_summary.get("baseline_leader")
            else "- ALF observed-score leader: n/a."
        ),
        (
            f"- ALF alternative definitions preserving the same leader: "
            f"`{alf_summary['leader_stability_count']}` / "
            f"`{alf_summary['scenario_count']}`."
            if alf_summary.get("baseline_leader")
            else "- ALF alternative-definition stability: n/a."
        ),
        "",
        "## Interpretation",
        "",
        (
            "This memo asks whether the benchmark conclusions depend too heavily on one exact formula for Contestability Support Score or Actionability-Legality Filter."
        ),
        (
            "If the same explainer remains the leader under multiple plausible definitions, that strengthens the construct-validity story. If the leader changes, we should describe the metric as design-sensitive rather than objective."
        ),
        "",
        "## CSS Definition Profiles",
        "",
        "| Scenario | Leader | Score | Leader Changed |",
        "| --- | --- | ---: | --- |",
    ]
    for row in css_rows:
        lines.append(
            (
                "| {scenario} | {leader} | {score} | {changed} |"
            ).format(
                scenario=row["scenario_name"],
                leader=row["explainer_id"],
                score=_fmt(row["composite_score"]),
                changed="yes" if row.get("leader_changed") else "no",
            )
        )

    lines.extend(
        [
            "",
            "## ALF Definition Profiles",
            "",
            "| Scenario | Leader | Score | Leader Changed |",
            "| --- | --- | ---: | --- |",
        ]
    )
    for row in alf_rows:
        lines.append(
            (
                "| {scenario} | {leader} | {score} | {changed} |"
            ).format(
                scenario=row["scenario_name"],
                leader=row["explainer_id"],
                score=_fmt(row["composite_score"]),
                changed="yes" if row.get("leader_changed") else "no",
            )
        )

    lines.extend(
        [
            "",
            "## Writing Guidance",
            "",
            "- Use CSS profile stability to argue that contestability is not an artifact of one arbitrary subscore weighting.",
            "- Use ALF profile instability, if present, as a principled caution that legality-aware recourse metrics encode real normative choices.",
            "- Keep this memo separate from generic metric-weight sensitivity: this one is about construct validity inside the benchmark dimensions themselves.",
            "",
        ]
    )
    return "\n".join(lines)


def _build_track_policy_sensitivity_memo(
    payload: dict[str, Any],
    analysis: dict[str, Any],
) -> str:
    summary = analysis.get("track_policy_summary", {})
    by_explainer = analysis.get("track_policy_by_explainer", [])
    metric_rows = analysis.get("track_policy_metric_summary", [])
    leader_rows = analysis.get("track_policy_leader_summary", [])

    lines = [
        "# LegalBench-XAI Track Policy Sensitivity Memo",
        "",
        f"- Experiment: `{payload['experiment_name']}`",
        f"- Generated at (UTC): `{payload['generated_at_utc']}`",
        "",
        "## Main Readout",
        "",
        (
            f"- Largest average absolute primary-vs-audit shift: "
            f"`{summary['largest_metric_shift']['metric']}` = "
            f"`{_fmt(summary['largest_metric_shift']['mean_abs_delta'])}`."
            if summary.get("largest_metric_shift")
            else "- Largest average absolute primary-vs-audit shift: n/a."
        ),
        (
            f"- Most track-sensitive explainer: "
            f"`{summary['most_sensitive_explainer']['explainer_id']}` "
            f"with overall mean absolute delta "
            f"`{_fmt(summary['most_sensitive_explainer']['overall_mean_abs_delta'])}`."
            if summary.get("most_sensitive_explainer")
            else "- Most track-sensitive explainer: n/a."
        ),
        (
            f"- Most track-robust explainer: "
            f"`{summary['most_robust_explainer']['explainer_id']}` "
            f"with overall mean absolute delta "
            f"`{_fmt(summary['most_robust_explainer']['overall_mean_abs_delta'])}`."
            if summary.get("most_robust_explainer")
            else "- Most track-robust explainer: n/a."
        ),
        (
            f"- Least leader-stable metric: "
            f"`{summary['least_stable_metric']['metric']}` with same-leader rate "
            f"`{_fmt(summary['least_stable_metric']['same_leader_rate'])}`."
            if summary.get("least_stable_metric")
            else "- Least leader-stable metric: n/a."
        ),
        "",
        "## Interpretation",
        "",
        (
            "This comparison treats the `primary` and `audit` tracks as a feature-governance perturbation: if the leaderboard changes sharply when the benchmark excludes a small set of risky or low-suitability features, the benchmark conclusions are sensitive to taxonomy choices."
        ),
        (
            "Small primary-vs-audit deltas suggest the core explainer ordering is robust to reasonable feature-policy changes. Larger deltas or repeated leader flips suggest the benchmark conclusions depend materially on how the feature taxonomy is operationalized."
        ),
        "",
        "## Metric-Level Shift Summary",
        "",
        "| Metric | Mean Delta (Audit - Primary) | Mean Absolute Delta | Matched Pairs |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in metric_rows:
        lines.append(
            (
                "| {metric} | {delta} | {abs_delta} | {pairs} |"
            ).format(
                metric=row["metric"],
                delta=_fmt(row["mean_delta"]),
                abs_delta=_fmt(row["mean_abs_delta"]),
                pairs=row["pair_count"],
            )
        )

    lines.extend(
        [
            "",
            "## Explainer-Level Shift Summary",
            "",
            "| Explainer | Matched Pairs | Overall Mean Abs Delta | Abs Faithfulness | Abs Stability | Abs Comprehensibility | Abs CSS | Abs Actionability | Abs ALF |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in by_explainer:
        lines.append(
            (
                "| {explainer} | {pairs} | {overall_abs} | {faithfulness} | {stability} | {comprehensibility} | {css} | {actionability} | {alf} |"
            ).format(
                explainer=row["explainer_id"],
                pairs=row["pair_count"],
                overall_abs=_fmt(row["overall_mean_abs_delta"]),
                faithfulness=_fmt(row["mean_abs_delta_faithfulness"]),
                stability=_fmt(row["mean_abs_delta_stability"]),
                comprehensibility=_fmt(row["mean_abs_delta_comprehensibility"]),
                css=_fmt(row["mean_abs_delta_css"]),
                actionability=_fmt(row["mean_abs_delta_actionability"]),
                alf=_fmt(row["mean_abs_delta_alf"]),
            )
        )

    lines.extend(
        [
            "",
            "## Leader Stability Across Tracks",
            "",
            "| Metric | Same-Leader Rate | Same-Leader Count | Comparison Count |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in leader_rows:
        lines.append(
            (
                "| {metric} | {rate} | {same_count} | {total} |"
            ).format(
                metric=row["metric"],
                rate=_fmt(row["same_leader_rate"]),
                same_count=row["same_leader_count"],
                total=row["comparison_count"],
            )
        )

    lines.extend(
        [
            "",
            "## Writing Guidance",
            "",
            "- Use this memo when reviewers ask whether the benchmark depends too heavily on one feature-exclusion policy.",
            "- If only one metric flips leaders across tracks, present that as a targeted robustness caveat rather than as a collapse of the whole benchmark.",
            "- Connect low primary-vs-audit deltas to the feature-taxonomy argument: the benchmark should be robust to reasonable governance choices, not only to random metric weights.",
            "",
        ]
    )
    return "\n".join(lines)


def _build_discussion_memo(payload: dict[str, Any], analysis: dict[str, Any]) -> str:
    by_explainer = analysis["by_explainer"]
    faithfulness_leader = _best_row(by_explainer, "mean_faithfulness")
    stability_leader = _best_row(by_explainer, "mean_stability")
    comprehensibility_leader = _best_row(by_explainer, "mean_comprehensibility")
    css_leader = _best_row(by_explainer, "mean_css")
    actionability_leader = _best_row(by_explainer, "mean_actionability")
    alf_leader = _best_row(by_explainer, "mean_alf")
    track_winners = _dataset_track_winners(analysis["by_dataset_track_explainer"])
    stable_track_rankings = _has_stable_track_rankings(track_winners)
    dataset_count = _real_dataset_count(payload)
    track_clause = _track_policy_clause(analysis, stable_track_rankings)

    lines = [
        "# LegalBench-XAI Discussion Memo",
        "",
        "## Main Interpretation",
        "",
        (
            "The current benchmark run already supports the core paper claim that legally meaningful explanation quality is multi-dimensional. "
            f"`{_leader_name(faithfulness_leader)}` leads on faithfulness, "
            f"`{_leader_name(stability_leader)}` leads on stability, "
            f"`{_leader_name(comprehensibility_leader)}` leads on comprehensibility, "
            f"`{_leader_name(css_leader)}` leads on contestability support, and "
            f"`{_leader_name(actionability_leader)}` leads on actionability while "
            f"`{_leader_name(alf_leader)}` leads on legality-aware recourse."
        ),
        "",
        "## Draft Discussion Points",
        "",
        (
            "The emerging pattern is that attribution-style explanations are strongest when the question is whether the explanation tracks the model and remains locally consistent, while recourse-style explanations are strongest when the question is whether an applicant can act on or challenge the decision."
        ),
        (
            "This tradeoff is the center of the paper's contribution: a method can be faithful and stable without being the best tool for contestability or actionability, and a recourse method can be practically useful while still requiring legality checks."
        ),
        (
            "The bootstrap intervals in the current results pack reinforce that we should write this as a tradeoff story rather than a search for a universal winner. Some leader identities are robust, but the intervals still argue for caution when turning leaderboard gaps into strong claims."
        ),
        (
            "The comprehensibility results also matter because they suggest that cognitively simple explanations should not be treated as automatically superior. In the current benchmark run, the easiest explanation family to read is not the same family that dominates on faithfulness or on applicant-facing recourse."
        ),
        track_clause,
        "",
        "## Writing Guidance",
        "",
        "- Use the tradeoff framing instead of searching for a single overall winner.",
        "- Keep faithfulness and stability together as model-facing reliability dimensions.",
        "- Keep CSS and actionability together as applicant-facing usefulness dimensions.",
        "- Discuss ALF as a necessary constraint on recourse rather than as a stand-alone interpretability metric.",
        "",
        "## Next Empirical Checks",
        "",
        "- Increase the denied-instance budget on the current real-data runs so the current ordering is supported by a materially larger adverse subset.",
        (
            "- Extend the same pipeline to a second real dataset so the current findings are no longer single-dataset evidence."
            if dataset_count < 2
            else (
                f"- Add at least one more distinct real credit dataset so the current findings are no longer limited to the current {dataset_count}-dataset slice."
                if dataset_count < 4
                else "- Prioritize sensitivity analysis and higher denied-instance budgets; a fifth dataset would help, but it is no longer the single highest-leverage gap."
            )
        ),
        (
            "- Check whether the primary versus audit tracks diverge once the benchmark covers multiple real credit datasets."
            if dataset_count < 2
            else "- Test whether the current primary-versus-audit divergences persist once the benchmark covers a broader credit dataset set."
        ),
        "",
    ]
    return "\n".join(lines)


def _build_limitations_memo(payload: dict[str, Any], analysis: dict[str, Any]) -> str:
    capabilities = payload.get("capabilities", {})
    missing = [
        name
        for name, info in capabilities.items()
        if not info.get("available", False)
    ]
    outstanding_steps = _limitations_next_steps(payload)

    lines = [
        "# LegalBench-XAI Limitations Memo",
        "",
        "## Current Limitations",
        "",
        _limitations_opening(payload),
        _limitations_data_note(payload),
        _limitations_stability_note(payload),
        (
            "The uncertainty estimates reported in the current bundle are hierarchical-bootstrap intervals over the benchmark structure. They are stronger than flat group resampling, but they are still not full hierarchical uncertainty estimates over applicants, datasets, models, and random variation jointly."
        ),
        (
            "The actionability and legality layers are normative approximations derived from the project taxonomy. They are useful for benchmarking, but they do not substitute for legal validation or domain-specific compliance review."
        ),
        _limitations_backend_gap(payload),
        "",
        "## Environment Gaps",
        "",
        (
            f"Missing empirical dependencies in this environment: {', '.join(f'`{name}`' for name in sorted(missing)) or 'none'}."
        ),
        "",
        "## What Must Change Before Submission",
        "",
        *outstanding_steps,
        "- Increase the denied-instance budget so the current real-data findings are supported by materially larger adverse subsets.",
        _next_dataset_step(payload),
        "",
    ]
    return "\n".join(lines)


def _build_appendix_tables_tex(payload: dict[str, Any], analysis: dict[str, Any]) -> str:
    dataset_rows = analysis["by_dataset_track_explainer"]
    model_rows = analysis["by_model_explainer"]
    performance_rows = analysis.get("model_performance", [])
    pairwise_rows = analysis["pairwise_wins"]
    lines = [
        "% Auto-generated by LegalBench-XAI paper kit",
        "% Requires \\usepackage{booktabs,longtable}",
        "",
        "\\section{Auto-Generated Benchmark Tables}",
        "",
        "\\subsection{Dataset and Track Slices}",
        "\\begin{longtable}{lllrrrrrrrr}",
        "\\toprule",
        "Dataset & Track & Explainer & Groups & Inst. & Faithfulness & Stability & Comprehensibility & CSS & Actionability & ALF \\\\",
        "\\midrule",
        "\\endfirsthead",
        "\\toprule",
        "Dataset & Track & Explainer & Groups & Inst. & Faithfulness & Stability & Comprehensibility & CSS & Actionability & ALF \\\\",
        "\\midrule",
        "\\endhead",
    ]
    for row in dataset_rows:
        lines.append(
            (
                "{dataset} & {track} & {explainer} & {groups} & {instances} & "
                "{faithfulness} & {stability} & {comprehensibility} & {css} & {actionability} & {alf} \\\\"
            ).format(
                dataset=_tex_escape(row["dataset_id"]),
                track=_tex_escape(row["track"]),
                explainer=_tex_escape(row["explainer_id"]),
                groups=row["group_count"],
                instances=row["total_instances"],
                faithfulness=_fmt(row["mean_faithfulness"]),
                stability=_fmt(row["mean_stability"]),
                comprehensibility=_fmt(row["mean_comprehensibility"]),
                css=_fmt(row["mean_css"]),
                actionability=_fmt(row["mean_actionability"]),
                alf=_fmt(row["mean_alf"]),
            )
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{longtable}",
            "",
            "\\subsection{Held-Out Model Performance}",
            "\\begin{longtable}{lllrrrrr}",
            "\\toprule",
            "Dataset & Track & Model & Train Rows & Test Rows & Accuracy & ROC AUC & True Adverse Rate \\\\",
            "\\midrule",
            "\\endfirsthead",
            "\\toprule",
            "Dataset & Track & Model & Train Rows & Test Rows & Accuracy & ROC AUC & True Adverse Rate \\\\",
            "\\midrule",
            "\\endhead",
        ]
    )
    for row in performance_rows:
        lines.append(
            (
                "{dataset} & {track} & {model} & {train_rows} & {test_rows} & {accuracy} & {roc_auc} & {positive_rate} \\\\"
            ).format(
                dataset=_tex_escape(row["dataset_id"]),
                track=_tex_escape(row["track"]),
                model=_tex_escape(row["model_id"]),
                train_rows=row["train_rows"],
                test_rows=row["test_rows"],
                accuracy=_fmt(row["accuracy"]),
                roc_auc=_fmt(row["roc_auc"]),
                positive_rate=_fmt(row["positive_rate"]),
            )
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{longtable}",
            "",
            "\\subsection{Bootstrap Confidence Intervals by Explainer}",
            "\\begin{longtable}{lrrrrrr}",
            "\\toprule",
            "Explainer & Faithfulness CI & Stability CI & Comprehensibility CI & CSS CI & Actionability CI & ALF CI \\\\",
            "\\midrule",
            "\\endfirsthead",
            "\\toprule",
            "Explainer & Faithfulness CI & Stability CI & Comprehensibility CI & CSS CI & Actionability CI & ALF CI \\\\",
            "\\midrule",
            "\\endhead",
        ]
    )
    for row in analysis["by_explainer"]:
        lines.append(
            (
                "{explainer} & {faithfulness_ci} & {stability_ci} & {comprehensibility_ci} & "
                "{css_ci} & {actionability_ci} & {alf_ci} \\\\"
            ).format(
                explainer=_tex_escape(row["explainer_id"]),
                faithfulness_ci=_tex_escape(_fmt_ci(row, "mean_faithfulness")),
                stability_ci=_tex_escape(_fmt_ci(row, "mean_stability")),
                comprehensibility_ci=_tex_escape(_fmt_ci(row, "mean_comprehensibility")),
                css_ci=_tex_escape(_fmt_ci(row, "mean_css")),
                actionability_ci=_tex_escape(_fmt_ci(row, "mean_actionability")),
                alf_ci=_tex_escape(_fmt_ci(row, "mean_alf")),
            )
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{longtable}",
            "",
            "\\subsection{Model Family Slices}",
            "\\begin{longtable}{llrrrrrrrr}",
            "\\toprule",
            "Model & Explainer & Groups & Inst. & Faithfulness & Stability & Comprehensibility & CSS & Actionability & ALF \\\\",
            "\\midrule",
            "\\endfirsthead",
            "\\toprule",
            "Model & Explainer & Groups & Inst. & Faithfulness & Stability & Comprehensibility & CSS & Actionability & ALF \\\\",
            "\\midrule",
            "\\endhead",
        ]
    )
    for row in model_rows:
        lines.append(
            (
                "{model} & {explainer} & {groups} & {instances} & {faithfulness} & {stability} & {comprehensibility} & "
                "{css} & {actionability} & {alf} \\\\"
            ).format(
                model=_tex_escape(row["model_id"]),
                explainer=_tex_escape(row["explainer_id"]),
                groups=row["group_count"],
                instances=row["total_instances"],
                faithfulness=_fmt(row["mean_faithfulness"]),
                stability=_fmt(row["mean_stability"]),
                comprehensibility=_fmt(row["mean_comprehensibility"]),
                css=_fmt(row["mean_css"]),
                actionability=_fmt(row["mean_actionability"]),
                alf=_fmt(row["mean_alf"]),
            )
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{longtable}",
            "",
            "\\subsection{Pairwise Win Summary}",
            "\\begin{longtable}{lllrrrrrr}",
            "\\toprule",
            "Metric & Explainer A & Explainer B & Wins A & Wins B & Ties & Groups & Win Rate A & Win Rate A CI \\\\",
            "\\midrule",
            "\\endfirsthead",
            "\\toprule",
            "Metric & Explainer A & Explainer B & Wins A & Wins B & Ties & Groups & Win Rate A & Win Rate A CI \\\\",
            "\\midrule",
            "\\endhead",
        ]
    )
    for row in pairwise_rows:
        lines.append(
            (
                "{metric} & {explainer_a} & {explainer_b} & {wins_a} & {wins_b} & "
                "{ties} & {groups} & {win_rate_a} & {win_rate_a_ci} \\\\"
            ).format(
                metric=_tex_escape(row["metric"]),
                explainer_a=_tex_escape(row["explainer_a"]),
                explainer_b=_tex_escape(row["explainer_b"]),
                wins_a=row["wins_a"],
                wins_b=row["wins_b"],
                ties=row["ties"],
                groups=row["comparable_groups"],
                win_rate_a=_fmt(row["win_rate_a"]),
                win_rate_a_ci=_tex_escape(_fmt_pairwise_ci(row, "win_rate_a")),
            )
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{longtable}",
            "",
            "\\paragraph{Runtime note.} "
            + _appendix_runtime_note(payload),
            "",
        ]
    )
    return "\n".join(lines)


def _draft_results_paragraph(
    faithfulness_leader: dict[str, Any],
    stability_leader: dict[str, Any],
    comprehensibility_leader: dict[str, Any],
    css_leader: dict[str, Any],
    actionability_leader: dict[str, Any],
    alf_leader: dict[str, Any],
    track_clause: str,
) -> str:
    if alf_leader:
        actionability_clause = (
            f"`{_leader_name(actionability_leader)}` leads on overall actionability "
            f"with `{_fmt(actionability_leader['mean_actionability'])}` and also achieves the strongest "
            f"legality-aware recourse performance with mean ALF `{_fmt(alf_leader['mean_alf'])}`."
        )
    else:
        actionability_clause = (
            f"`{_leader_name(actionability_leader)}` leads on overall actionability "
            f"with `{_fmt(actionability_leader['mean_actionability'])}`. "
            "ALF is not available in this run because the selected explainer slice does not include legality-aware recourse methods."
        )
    return (
        "Across the current benchmark run, faithfulness, stability, comprehensibility, and applicant-facing usefulness do not collapse into a single ranking. "
        f"`{_leader_name(faithfulness_leader)}` leads on mean faithfulness with "
        f"`{_fmt(faithfulness_leader['mean_faithfulness'])}`, "
        f"`{_leader_name(stability_leader)}` leads on mean stability with "
        f"`{_fmt(stability_leader['mean_stability'])}`, "
        f"`{_leader_name(comprehensibility_leader)}` leads on mean comprehensibility with "
        f"`{_fmt(comprehensibility_leader['mean_comprehensibility'])}`, while recourse-oriented explainers outperform attribution-style "
        "methods on applicant-facing metrics. "
        f"`{_leader_name(css_leader)}` leads on overall contestability support "
        f"with mean CSS `{_fmt(css_leader['mean_css'])}`, while "
        f"{actionability_clause} "
        f"{track_clause}"
    )


def _dataset_track_winners(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["dataset_id"], row["track"]), []).append(row)

    winners: list[dict[str, Any]] = []
    for key in sorted(grouped):
        dataset_id, track = key
        items = grouped[key]
        best_faithfulness = _best_row(items, "mean_faithfulness")
        best_stability = _best_row(items, "mean_stability")
        best_comprehensibility = _best_row(items, "mean_comprehensibility")
        best_css = _best_row(items, "mean_css")
        best_actionability = _best_row(items, "mean_actionability")
        best_alf = _best_row(items, "mean_alf")
        winners.append(
            {
                "dataset_id": dataset_id,
                "track": track,
                "best_faithfulness_explainer": (
                    best_faithfulness["explainer_id"] if best_faithfulness else "n/a"
                ),
                "best_faithfulness_score": (
                    best_faithfulness["mean_faithfulness"] if best_faithfulness else None
                ),
                "best_stability_explainer": (
                    best_stability["explainer_id"] if best_stability else "n/a"
                ),
                "best_stability_score": (
                    best_stability["mean_stability"] if best_stability else None
                ),
                "best_comprehensibility_explainer": (
                    best_comprehensibility["explainer_id"]
                    if best_comprehensibility
                    else "n/a"
                ),
                "best_comprehensibility_score": (
                    best_comprehensibility["mean_comprehensibility"]
                    if best_comprehensibility
                    else None
                ),
                "best_css_explainer": best_css["explainer_id"] if best_css else "n/a",
                "best_css_score": best_css["mean_css"] if best_css else None,
                "best_actionability_explainer": (
                    best_actionability["explainer_id"] if best_actionability else "n/a"
                ),
                "best_actionability_score": (
                    best_actionability["mean_actionability"] if best_actionability else None
                ),
                "best_alf_explainer": best_alf["explainer_id"] if best_alf else "n/a",
                "best_alf_score": best_alf["mean_alf"] if best_alf else None,
            }
        )
    return winners


def _has_stable_track_rankings(winners: list[dict[str, Any]]) -> bool:
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for row in winners:
        by_dataset.setdefault(row["dataset_id"], []).append(row)
    for items in by_dataset.values():
        if len(items) != 2:
            continue
        primary = next((item for item in items if item["track"] == "primary"), None)
        audit = next((item for item in items if item["track"] == "audit"), None)
        if not primary or not audit:
            continue
        if (
            primary["best_faithfulness_explainer"] != audit["best_faithfulness_explainer"]
            or primary["best_stability_explainer"] != audit["best_stability_explainer"]
            or primary["best_comprehensibility_explainer"]
            != audit["best_comprehensibility_explainer"]
            or primary["best_css_explainer"] != audit["best_css_explainer"]
            or primary["best_actionability_explainer"] != audit["best_actionability_explainer"]
            or primary["best_alf_explainer"] != audit["best_alf_explainer"]
        ):
            return False
    return True


def _track_policy_clause(
    analysis: dict[str, Any],
    fallback_stable: bool,
) -> str:
    leader_rows = analysis.get("track_policy_leader_summary", [])
    summary = analysis.get("track_policy_summary", {})
    largest_shift = summary.get("largest_metric_shift")
    if not leader_rows:
        if fallback_stable:
            return (
                "In the current benchmark run, primary and audit track rankings are largely unchanged, "
                "suggesting that the current toy exclusions do not yet alter method ordering."
            )
        return (
            "Primary and audit track rankings diverge in the current run, which suggests that "
            "track-specific exclusions should be discussed explicitly in the results."
        )

    unstable = [row for row in leader_rows if row.get("same_leader_rate", 0.0) < 1.0]
    if not unstable:
        return (
            "Primary and audit track leader identities are unchanged across all measured metrics, "
            "suggesting that the current feature-policy exclusions do not materially reorder methods."
        )

    stable_count = len(leader_rows) - len(unstable)
    least_stable = min(unstable, key=lambda row: row["same_leader_rate"])
    shift_clause = (
        f"Primary-vs-audit score shifts remain small on average, with the largest mean absolute shift on "
        f"`{largest_shift['metric']}` at `{_fmt(largest_shift['mean_abs_delta'])}`. "
        if largest_shift
        else ""
    )
    if len(unstable) == 1:
        return (
            f"{shift_clause}Primary and audit track leader identities are stable for `{stable_count}` of "
            f"`{len(leader_rows)}` metrics; the only material instability appears in "
            f"`{least_stable['metric']}` with same-leader rate "
            f"`{_fmt(least_stable['same_leader_rate'])}`."
        )
    unstable_labels = ", ".join(f"`{row['metric']}`" for row in unstable)
    return (
        f"{shift_clause}Primary and audit track leader identities are stable for `{stable_count}` of "
        f"`{len(leader_rows)}` metrics, with instability concentrated in {unstable_labels}. "
        f"The least stable metric is `{least_stable['metric']}` at same-leader rate "
        f"`{_fmt(least_stable['same_leader_rate'])}`."
    )


def _best_row(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    eligible = [row for row in rows if row.get(key) is not None]
    if not eligible:
        return None
    return max(eligible, key=lambda row: row[key])


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _fmt_ci(row: dict[str, Any], metric: str) -> str:
    low = row.get(f"{metric}_ci_low")
    high = row.get(f"{metric}_ci_high")
    if low is None or high is None:
        return "n/a"
    return f"[{_fmt(low)}, {_fmt(high)}]"


def _leader_headline(
    label: str,
    row: dict[str, Any] | None,
    metric: str,
    metric_label: str,
) -> str:
    if not row:
        return f"- {label}: not available in this run."
    return (
        f"- {label}: `{row['explainer_id']}` "
        f"with {metric_label} `{_fmt(row[metric])}` "
        f"and {_bootstrap_interval_label(row, metric)}."
    )


def _pairwise_headline(label: str, row: dict[str, Any] | None) -> str:
    if not row:
        return f"- {label}: not available in this run."
    return (
        f"- {label}: `{row['explainer_id']}` "
        f"with `{row['win_count']}` matched-group wins."
    )


def _leader_name(row: dict[str, Any] | None) -> str:
    if not row:
        return "n/a"
    return str(row.get("explainer_id", "n/a"))


def _bootstrap_note_line(analysis: dict[str, Any]) -> str:
    bootstrap = analysis.get("bootstrap", {})
    if not bootstrap.get("enabled", False):
        return "- Bootstrap uncertainty is disabled for this run."
    level = int(round(float(bootstrap["confidence_level"]) * 100))
    method = bootstrap.get("method", "group_bootstrap")
    if method in {"hierarchical", "hierarchical_bootstrap"}:
        levels = " -> ".join(bootstrap.get("hierarchy_levels", ["dataset_id", "track", "model_id"]))
        return (
            f"- `{level}%` hierarchical-bootstrap confidence intervals with "
            f"`{bootstrap['num_resamples']}` resamples over `{levels}` benchmark clusters are included in the analysis bundle."
        )
    return (
        f"- `{level}%` group-bootstrap confidence intervals with "
        f"`{bootstrap['num_resamples']}` resamples over benchmark groups are included in the analysis bundle."
    )


def _bootstrap_interval_label(row: dict[str, Any], metric: str) -> str:
    ci = _fmt_ci(row, metric)
    if ci == "n/a":
        return "no bootstrap interval available"
    return f"`{ci}` bootstrap CI"


def _fmt_pairwise_ci(row: dict[str, Any], rate_key: str) -> str:
    low = row.get(f"{rate_key}_ci_low")
    high = row.get(f"{rate_key}_ci_high")
    if low is None or high is None:
        return "n/a"
    return f"[{_fmt(low)}, {_fmt(high)}]"


def _tex_escape(value: str) -> str:
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("&", "\\&")
        .replace("%", "\\%")
    )


def _format_backend_summary(items: dict[str, int]) -> str:
    if not items:
        return "n/a"
    return ", ".join(f"`{name}` x {count}" for name, count in sorted(items.items()))


def _backend_mode(payload: dict[str, Any]) -> str:
    overview = payload.get("overview", {})
    model_usage = overview.get("backend_usage", {}).get("models", {})
    explainer_usage = overview.get("backend_usage", {}).get("explainers", {})

    model_impls = set(model_usage)
    explainer_impls = set(explainer_usage)
    all_model_stub = model_impls <= {"heuristic_model"}
    all_explainer_stub = explainer_impls <= {"demo_generator"}
    any_model_stub = "heuristic_model" in model_impls
    any_explainer_stub = "demo_generator" in explainer_impls

    if all_model_stub and all_explainer_stub:
        return "scaffold"
    if not any_model_stub and not any_explainer_stub:
        return "real"
    if not all_model_stub and not all_explainer_stub:
        return "mixed"
    return "partial"


def _data_source_mode(payload: dict[str, Any]) -> str:
    source_usage = payload.get("overview", {}).get("source_usage", {})
    source_kinds = set(source_usage)
    if not source_kinds:
        return "unknown"
    real_like = {"local_override", "custom"}
    if source_kinds <= {"demo"}:
        return "fixture"
    if source_kinds & real_like and not (source_kinds - real_like):
        return "real"
    return "mixed"


def _stability_evidence_mode(payload: dict[str, Any]) -> str:
    evidence_usage = payload.get("overview", {}).get("stability_evidence_usage", {})
    names = set(evidence_usage)
    if not names:
        return "unknown"
    if any(name.startswith("perturbation_probe") for name in names):
        if all(name.startswith("perturbation_probe") for name in names):
            return "perturbation_probe"
        return "mixed"
    return "structural"


def _results_caveat(payload: dict[str, Any]) -> str:
    overview = payload.get("overview", {})
    model_usage = overview.get("backend_usage", {}).get("models", {})
    explainer_usage = overview.get("backend_usage", {}).get("explainers", {})
    mode = _backend_mode(payload)
    data_mode = _data_source_mode(payload)
    dataset_count = _real_dataset_count(payload)

    if mode == "scaffold":
        return (
            "- These outputs come from the current dependency-light scaffold using heuristic "
            "model and explainer backends. Treat them as writing scaffolds and sanity checks, "
            "not as final empirical claims."
        )
    if mode == "real" and data_mode == "real":
        return (
            "- These outputs come from a fully real benchmark run on local real dataset files. "
            f"Model backends in use: {_format_backend_summary(model_usage)}. "
            f"Explainer backends in use: {_format_backend_summary(explainer_usage)}. "
            + (
                "Treat this as the first real empirical slice, not yet as submission-ready evidence."
                if dataset_count <= 1
                else "Treat this as the first real cross-dataset empirical bundle, not yet as submission-ready evidence."
            )
        )
    if mode == "real":
        return (
            "- These outputs come from a fully real benchmark run on the current fixture datasets. "
            f"Model backends in use: {_format_backend_summary(model_usage)}. "
            f"Explainer backends in use: {_format_backend_summary(explainer_usage)}. "
            "Treat this as real pipeline evidence, but not yet as the final benchmark until the "
            "same pipeline is run on real denied-instance datasets."
        )

    return (
        "- These outputs come from a mixed benchmark run. "
        f"Model backends in use: {_format_backend_summary(model_usage)}. "
        f"Explainer backends in use: {_format_backend_summary(explainer_usage)}. "
        "Treat the active real slice as partial empirical evidence, but not yet as the final "
        "benchmark."
    )


def _limitations_opening(payload: dict[str, Any]) -> str:
    overview = payload.get("overview", {})
    model_usage = overview.get("backend_usage", {}).get("models", {})
    explainer_usage = overview.get("backend_usage", {}).get("explainers", {})
    mode = _backend_mode(payload)
    data_mode = _data_source_mode(payload)

    if mode == "scaffold":
        return (
            "The current outputs are still generated from the dependency-light scaffold. "
            f"Model backends in use: {_format_backend_summary(model_usage)}. "
            f"Explainer backends in use: {_format_backend_summary(explainer_usage)}."
        )
    if mode == "real" and data_mode == "real":
        return (
            "The current outputs come from a fully real benchmark run on local real dataset files. "
            f"Model backends in use: {_format_backend_summary(model_usage)}. "
            f"Explainer backends in use: {_format_backend_summary(explainer_usage)}."
        )
    if mode == "real":
        return (
            "The current outputs come from a fully real benchmark run on the present fixture "
            "datasets. "
            f"Model backends in use: {_format_backend_summary(model_usage)}. "
            f"Explainer backends in use: {_format_backend_summary(explainer_usage)}."
        )

    return (
        "The current outputs come from a mixed benchmark run that combines real and scaffold "
        "backends. "
        f"Model backends in use: {_format_backend_summary(model_usage)}. "
        f"Explainer backends in use: {_format_backend_summary(explainer_usage)}."
    )


def _limitations_backend_gap(payload: dict[str, Any]) -> str:
    mode = _backend_mode(payload)
    data_mode = _data_source_mode(payload)
    stability_mode = _stability_evidence_mode(payload)
    dataset_count = _real_dataset_count(payload)
    if mode == "scaffold":
        return (
            "The current benchmark does not yet exercise any real training or explanation "
            "backends. Until a real slice is active, the outputs are most useful as design "
            "validation, writing scaffolds, and interface tests."
        )
    if mode == "real" and data_mode == "real" and stability_mode == "perturbation_probe":
        return (
            "The benchmark now runs fully real models and explainers on local real datasets with "
            "perturbation-based stability evidence. The main remaining gap is breadth: the current "
            + (
                "evidence is still based on a single dataset and a limited denied-instance budget."
                if dataset_count <= 1
                else f"evidence spans {dataset_count} real datasets, but still uses a limited denied-instance budget and a narrow domain slice."
            )
        )
    if mode == "real":
        return (
            "The benchmark now runs fully real models and explainers on the current fixture "
            "datasets, but the evidence is still limited by toy data and by placeholder "
            "stability probing. The outputs are useful for validating the empirical pipeline, "
            "but they do not yet substitute for the full benchmark on real denied-instance data."
        )
    return (
        "The benchmark now includes a substantial real slice, but some groups still rely on "
        "scaffold explainers and all runs still rely on fixture data. The current outputs are "
        "useful for partial empirical validation, but they do not yet substitute for the full "
        "benchmark on real denied-instance datasets."
    )


def _limitations_next_steps(payload: dict[str, Any]) -> list[str]:
    overview = payload.get("overview", {})
    model_usage = overview.get("backend_usage", {}).get("models", {})
    explainer_usage = overview.get("backend_usage", {}).get("explainers", {})

    steps: list[str] = []
    if _backend_mode(payload) == "scaffold":
        steps.append("- Replace heuristic models with at least one real training backend.")
        steps.append("- Replace demo explainer generators with at least one real explainer family.")
        return steps

    if model_usage.get("heuristic_model", 0) > 0:
        steps.append(
            "- Replace the remaining heuristic model groups with real backends so the benchmark is no longer partially dominated by scaffold models."
        )
    if explainer_usage.get("demo_generator", 0) > 0:
        steps.append(
            "- Replace the remaining scaffold explainer groups with real rule and recourse backends so the comparison is no longer dominated by demo explainers."
        )
    return steps


def _appendix_runtime_note(payload: dict[str, Any]) -> str:
    overview = payload.get("overview", {})
    model_usage = overview.get("backend_usage", {}).get("models", {})
    explainer_usage = overview.get("backend_usage", {}).get("explainers", {})
    mode = _backend_mode(payload)
    data_mode = _data_source_mode(payload)

    if mode == "scaffold":
        return (
            "This appendix was generated from the current scaffold run. "
            "The present environment uses heuristic model and explainer backends, so these "
            "tables should be treated as draft-writing aids until the full benchmark is run "
            "with real dependencies and datasets."
        )
    if mode == "real" and data_mode == "real":
        return (
            "This appendix was generated from a fully real benchmark run on local real dataset "
            f"files. Model backends in use: {_format_backend_summary(model_usage)}; "
            f"explainer backends in use: {_format_backend_summary(explainer_usage)}. "
            "These tables are real empirical evidence, but they still need a larger denied-instance "
            "budget and cross-dataset replication before they support final paper claims."
        )
    if mode == "real":
        return (
            "This appendix was generated from a fully real benchmark run on the current fixture "
            f"datasets. Model backends in use: {_format_backend_summary(model_usage)}; "
            f"explainer backends in use: {_format_backend_summary(explainer_usage)}. "
            "These tables validate the empirical pipeline, but they still need to be rerun on "
            "real denied-instance datasets."
        )

    return (
        "This appendix was generated from a mixed benchmark run. "
        f"Model backends in use: {_format_backend_summary(model_usage)}; "
        f"explainer backends in use: {_format_backend_summary(explainer_usage)}. "
        "These tables are useful for partial empirical validation, but they do not yet replace "
        "the full benchmark on real datasets."
    )


def _limitations_data_note(payload: dict[str, Any]) -> str:
    data_mode = _data_source_mode(payload)
    dataset_count = _real_dataset_count(payload)
    if data_mode == "real":
        return (
            "The current run already uses local real credit data rather than fixture rows. "
            + (
                "That is a meaningful empirical step forward, but one real dataset is still not enough to anchor top-tier claims about robustness."
                if dataset_count <= 1
                else (
                    f"That is a meaningful empirical step forward, but {dataset_count} real datasets are still not enough to anchor top-tier claims about robustness across credit settings."
                    if dataset_count < 4
                    else f"That is a meaningful empirical step forward, and {dataset_count} real datasets are now a credible benchmark base. The main remaining burden is no longer only dataset count, but sensitivity analysis, uncertainty calibration, and larger denied-instance budgets."
                )
            )
        )
    if data_mode == "fixture":
        return (
            "The present data are still fixture-style inputs designed to exercise the benchmark "
            "pipeline, not to support empirical claims about real credit decision systems."
        )
    return (
        "The current outputs mix fixture and local data sources, which is useful for pipeline "
        "validation but not yet clean enough for a final empirical story."
    )


def _limitations_stability_note(payload: dict[str, Any]) -> str:
    stability_mode = _stability_evidence_mode(payload)
    if stability_mode == "perturbation_probe":
        return (
            "The stability metric is now backed by perturbation-based evidence in this run. "
            "The remaining limitation is not placeholder logic but evaluation scale and cross-dataset validation."
        )
    if stability_mode == "structural":
        return (
            "The stability metric is still driven by structural or metadata-side evidence. "
            "That is useful for pipeline development, but it remains weaker than explicit perturbation experiments."
        )
    return (
        "The stability metric currently mixes perturbation-based and fallback evidence sources across groups. "
        "That should be tightened before the final experimental section is frozen."
    )


def _real_dataset_count(payload: dict[str, Any]) -> int:
    if _data_source_mode(payload) != "real":
        return 0
    return len(payload.get("overview", {}).get("datasets", []))


def _next_dataset_step(payload: dict[str, Any]) -> str:
    dataset_count = _real_dataset_count(payload)
    if dataset_count < 2:
        return (
            "- Replicate the same pipeline on at least one additional real credit dataset before treating the current tradeoff pattern as robust."
        )
    if dataset_count < 4:
        return (
            "- Add at least one more distinct real credit dataset before treating the current tradeoff pattern as robust across credit settings."
        )
    return (
        "- A fifth dataset would strengthen breadth, but the higher-priority next step is sensitivity analysis plus larger denied-instance budgets on the existing four-dataset benchmark."
    )
