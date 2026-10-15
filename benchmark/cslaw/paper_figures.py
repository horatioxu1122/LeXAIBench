"""Figures for the CS&Law paper's findings (E1 direction, E2 against E3, E4 model choice).

Reads the denial-corrected bundles and their analysis outputs (the directory
written by benchmark/scripts/cluster/run_cslaw_denial.sh) and the original v2
bundles (notices before the direction rule). Writes one vector PDF per figure
and the plotted values as a CSV beside it. Offline; fits no model.

Each figure defends one claim of Section 5:
  fig_e1_direction          Copying the four largest attributions onto a notice
                            lists factors that favored approval, and filtering by
                            the method's own sign removes them only for SHAP.
  fig_accuracy_specificity  Accuracy alone does not separate a canned notice from
                            an explanation; the comparison across applicants does.
  fig_model_choice          The choice between two equally accurate models
                            changes the first reason more often than the choice
                            between the two selection methods of comment 9(b)(2)-5.

Direction (E1) is counted by the model's own sign, as in appendix_tables.py: a
reason favors approval when the mean native contribution of the features it
names is negative. Intervals are 95% percentile intervals from 2,000 bootstrap
resamples (seed 42). E1 and E2 resample notices within each combination of
dataset and version. E4 uses the applicants explained under all three models;
each appears in both versions, so it resamples applicants within dataset and
keeps both versions of an applicant together. The E3 intervals are read from
notice_specificity.py's output.

If audit_panel_alignment.py (from the figure QA tools) is importable and
--qa-dir is given, each figure's panel geometry is audited before export.

Usage:
    python benchmark/cslaw/paper_figures.py \
        --denial-dir benchmark/outputs/cslaw_denial_<tag> \
        --source-dir benchmark/outputs --tag <tag> \
        --out-dir paper/figures [--qa-dir DIR]
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

try:
    from audit_panel_alignment import require_matplotlib_panel_alignment
except ImportError:
    require_matplotlib_panel_alignment = None

CREDIT = ("heloc", "default_of_credit_card_clients", "give_me_some_credit", "south_german_credit")
DATASET_LABEL = {
    "heloc": "HELOC",
    "default_of_credit_card_clients": "Default Credit",
    "give_me_some_credit": "Give Me Some Credit",
    "south_german_credit": "South German Credit",
}
N_BOOT = 2000
SEED = 42
MM = 1 / 25.4

INK = "#272727"
GREY_DARK = "#4D4D4D"
GREY = "#767676"
GREY_LIGHT = "#CFCECE"
BLUE = "#0F4D92"
RED = "#B64342"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.6,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "legend.frameon": False,
    "lines.linewidth": 0.8,
})


# --------------------------------------------------------------------- data
def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        sys.exit(f"missing input: {path}")
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_bundle(path: Path) -> dict:
    if not (path / "results.json").is_file():
        sys.exit(f"missing bundle: {path / 'results.json'}")
    return common.load_results(path / "results.json")


def ratio_ci(units_by_stratum: dict, n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float, float]:
    """sum(numerator) / sum(denominator) with a 95% interval, resampling units within each stratum."""
    rng = np.random.default_rng(seed)
    strata = [np.asarray(units, dtype=float).reshape(-1, 2) for units in units_by_stratum.values() if len(units)]
    point = sum(s[:, 0].sum() for s in strata) / sum(s[:, 1].sum() for s in strata)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        num = den = 0.0
        for s in strata:
            pick = s[rng.integers(0, len(s), len(s))]
            num += pick[:, 0].sum()
            den += pick[:, 1].sum()
        stats[b] = num / den
    stats.sort()
    return float(point), float(stats[int(0.025 * n_boot)]), float(stats[int(0.975 * n_boot) - 1])


def direction_units(payload: dict) -> dict:
    """{explainer: {(dataset, version): [(reasons favoring approval, reasons) per notice]}}, signs from the model."""
    out: dict = defaultdict(lambda: defaultdict(list))
    for row, instance in common.iter_instances(payload):
        if row["dataset_id"] not in CREDIT:
            continue
        reasons = sorted(
            (instance.get("canonical_explanation") or {}).get("principal_reasons") or [], key=lambda r: r.get("rank", 0)
        )[:4]
        values = (instance.get("native_contributions") or {}).get("values") or {}
        flags = []
        for reason in reasons:
            feats = [f for f in reason.get("supporting_features") or [] if f in values]
            if feats:
                flags.append(1.0 if sum(values[f] for f in feats) / len(feats) < 0 else 0.0)
        if flags:
            out[row["explainer_id"]][(row["dataset_id"], row["track"])].append((sum(flags), len(flags)))
    return out


def accuracy_units(analysis: Path) -> dict:
    """{(bundle, explainer): {(dataset, version): [(share of the marginal method's reasons named, 1) per notice]}}."""
    out: dict = defaultdict(lambda: defaultdict(list))
    for sub, bundle in (("regb_credit", "credit"), ("regb_nullbaselines", "nullbaselines"), ("regb_lime_local", "lime_local")):
        for row in read_csv(analysis / sub / "regb_reference_instances.csv"):
            value = row.get("pr_overlap_m1_grouped")
            if row["dataset_id"] in CREDIT and value not in (None, "", "None", "nan"):
                out[(bundle, row["explainer_id"])][(row["dataset_id"], row["track"])].append((float(value), 1.0))
    return out


def specificity_pooled(analysis: Path) -> dict:
    return {
        (row["bundle"], row["explainer_id"]): (float(row["mean"]), float(row["ci_low"]), float(row["ci_high"]))
        for row in read_csv(analysis / "specificity" / "specificity_pooled.csv")
    }


def model_choice_units(analysis: Path) -> tuple[dict, dict]:
    """Per applicant explained under all three models, pooled over the two versions: first-reason disagreements
    between the two commentary methods within the boosting machine and the boosted trees, and between those two
    models under the marginal applicant method. Returns ({dataset: [(disagreements, comparisons) per applicant]},
    the same for models)."""
    rows = read_csv(analysis / "multiplicity" / "multiplicity_cross_model_instances.csv")
    by_method: dict = defaultdict(dict)
    for row in rows:
        if row["method"] in ("regb_m1", "regb_m2") and row["dataset_id"] in CREDIT:
            by_method[row["method"]][(row["dataset_id"], row["track"], row["eval_row_index"])] = row
    method_counts: dict = defaultdict(lambda: [0.0, 0.0])
    model_counts: dict = defaultdict(lambda: [0.0, 0.0])
    for (dataset, _track, row_index), m1 in by_method["regb_m1"].items():
        m2 = by_method["regb_m2"].get((dataset, _track, row_index))
        if m2 is None:
            continue
        applicant = (dataset, row_index)
        for model in ("ebm", "xgboost"):
            method_counts[applicant][0] += 1.0 if m1[f"top1__{model}"] != m2[f"top1__{model}"] else 0.0
            method_counts[applicant][1] += 1.0
        model_counts[applicant][0] += 1.0 if m1["top1__ebm"] != m1["top1__xgboost"] else 0.0
        model_counts[applicant][1] += 1.0
    methods: dict = defaultdict(list)
    models: dict = defaultdict(list)
    for (dataset, _row), counts in method_counts.items():
        methods[dataset].append(tuple(counts))
    for (dataset, _row), counts in model_counts.items():
        models[dataset].append(tuple(counts))
    return methods, models


# --------------------------------------------------------------------- output
def write_values(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow([f"{v:.4f}" if isinstance(v, float) else v for v in row])
    print(f"wrote {path}")


def save(fig: plt.Figure, out_dir: Path, name: str, qa_dir: Path | None) -> None:
    fig.canvas.draw()
    if qa_dir is not None and require_matplotlib_panel_alignment is not None:
        require_matplotlib_panel_alignment(
            fig,
            json_out=str(qa_dir / f"{name}.alignment.json"),
            overlay_svg=str(qa_dir / f"{name}.alignment.svg"),
            tolerance_pt=1.5,
            gutter_tolerance_pt=1.5,
            strict=True,
        )
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02, metadata={"CreationDate": None})
    if qa_dir is not None:
        fig.savefig(qa_dir / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {out_dir / (name + '.pdf')}")


def pct_label(value: float) -> str:
    return "<1" if 0 < value < 0.01 else f"{100 * value:.0f}"


def value_columns(ax: plt.Axes, top: float, headers: tuple[str, str], cells: list[tuple[float, str, str]],
                  right_edges: tuple[float, float]) -> None:
    """Two right-aligned columns of numbers beside the axes, one row per plotted row."""
    column = blended_transform_factory(ax.transAxes, ax.transData)
    for header, x in zip(headers, right_edges):
        ax.text(x, top, header, transform=column, ha="right", va="bottom", color=INK)
    for y, first, second in cells:
        for text, x in zip((first, second), right_edges):
            ax.text(x, y, text, transform=column, ha="right", va="center", color=INK)


# --------------------------------------------------------------------- figures
def fig_e1_direction(before: dict, after: dict, out_dir: Path, qa_dir: Path | None) -> None:
    rows = [
        ("shap", "SHAP"),
        ("lime", "LIME"),
        ("dice", "DiCE†"),
        ("anchors", "Anchors"),
        ("wachter", "Wachter†"),
        ("global_ranking", "Canned notice"),
        ("random_reasons", "Random notice"),
    ]
    ypos = [0, 1, 2, 3, 4, 5.4, 6.4]
    stats = []
    for (method, label), y in zip(rows, ypos):
        b = ratio_ci(before[method])
        a = ratio_ci(after[method])
        stats.append((method, label, y, b, a))

    fig, ax = plt.subplots(figsize=(118 * MM, 50 * MM))
    offset = 0.17
    for method, label, y, b, a in stats:
        ax.plot([100 * b[1], 100 * b[2]], [y - offset] * 2, color=GREY_DARK, lw=0.8, solid_capstyle="butt")
        ax.plot([100 * a[1], 100 * a[2]], [y + offset] * 2, color=BLUE, lw=0.8, solid_capstyle="butt")
        ax.plot(100 * b[0], y - offset, "o", ms=4, mfc="white", mec=GREY_DARK, mew=0.8, clip_on=False)
        ax.plot(100 * a[0], y + offset, "o", ms=4, mfc=BLUE, mec=BLUE, mew=0.8, clip_on=False)
    ax.axhline(4.7, color=GREY_LIGHT, lw=0.6)
    ax.set_yticks(ypos, [label for _, label in rows])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(6.9, -0.6)
    ax.set_xlim(0, 45)
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_xlabel("Reasons that favored approval in the model’s computation (%)")
    ax.spines["left"].set_visible(False)

    value_columns(ax, -0.6, ("Before", "After"), [(y, pct_label(b[0]), pct_label(a[0])) for _, _, y, b, a in stats],
                  (1.11, 1.21))
    handles = [
        Line2D([], [], marker="o", ls="none", ms=4, mfc="white", mec=GREY_DARK, mew=0.8,
               label="Before the direction rule"),
        Line2D([], [], marker="o", ls="none", ms=4, mfc=BLUE, mec=BLUE, mew=0.8,
               label="After the direction rule, applied with the method’s own signs"),
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.03), ncol=1, handletextpad=0.3,
              borderaxespad=0, labelspacing=0.3)
    save(fig, out_dir, "fig_e1_direction", qa_dir)
    write_values(
        out_dir / "fig_e1_direction.csv",
        ["notice", "before", "before_low", "before_high", "after", "after_low", "after_high"],
        [[label.rstrip("†"), *b, *a] for _, label, _, b, a in stats],
    )


def fig_accuracy_specificity(accuracy: dict, specificity: dict, out_dir: Path, qa_dir: Path | None) -> None:
    # (bundle, explainer, label, style, label offset in points, horizontal and vertical alignment)
    points = [
        ("credit", "shap", "SHAP", "method", (0, 4), "center", "bottom"),
        ("credit", "dice", "DiCE", "method", (0, 4), "center", "bottom"),
        ("credit", "anchors", "Anchors", "method", (-14, 0), "right", "center"),
        ("credit", "wachter", "Wachter", "method", (14, 0), "left", "center"),
        ("credit", "lime", "LIME", "method", (0, -5), "center", "top"),
        ("lime_local", "lime", "LIME, local sampling", "local", (0, 4), "center", "bottom"),
        ("nullbaselines", "global_ranking", "Canned notice", "canned", (-4, 5), "right", "bottom"),
        ("nullbaselines", "random_reasons", "Random\nnotice", "random", (6, 0), "left", "center"),
    ]
    styles = {
        "method": dict(marker="o", ms=4.5, mfc=BLUE, mec=BLUE, color=BLUE),
        "local": dict(marker="o", ms=4.5, mfc="white", mec=BLUE, color=BLUE),
        "canned": dict(marker="D", ms=4.2, mfc=RED, mec=RED, color=RED),
        "random": dict(marker="s", ms=4.2, mfc=GREY, mec=GREY, color=GREY),
        "reference": dict(marker="*", ms=7, mfc=INK, mec=INK, color=INK),
    }
    values = []
    for bundle, method, label, style, offset, ha, va in points:
        acc = ratio_ci(accuracy[(bundle, method)])
        spec = specificity[(bundle, method)]
        values.append((bundle, method, label, style, offset, ha, va, spec, acc))
    marginal_spec = specificity[("credit", "regb_reference")]
    canned_acc = next(v[8] for v in values if v[1] == "global_ranking")

    fig, ax = plt.subplots(figsize=(92 * MM, 72 * MM))
    ax.plot([0, 1.0], [canned_acc[0]] * 2, color=RED, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.plot([marginal_spec[0]] * 2, [0, 1.0], color=GREY, lw=0.6, ls=(0, (1, 1.5)), zorder=1)
    lime = next(v for v in values if v[0] == "credit" and v[1] == "lime")
    local = next(v for v in values if v[0] == "lime_local")
    ax.annotate("", xy=(local[7][0], local[8][0]), xytext=(lime[7][0], lime[8][0]),
                arrowprops=dict(arrowstyle="-|>", lw=0.6, color=BLUE, shrinkA=4, shrinkB=4, mutation_scale=6), zorder=2)
    for bundle, method, label, style, offset, ha, va, spec, acc in values:
        st = styles[style]
        ax.plot([spec[1], spec[2]], [acc[0]] * 2, color=st["color"], lw=0.8, solid_capstyle="butt", zorder=3)
        ax.plot([spec[0]] * 2, [acc[1], acc[2]], color=st["color"], lw=0.8, solid_capstyle="butt", zorder=3)
        ax.plot(spec[0], acc[0], ls="none", marker=st["marker"], ms=st["ms"], mfc=st["mfc"], mec=st["mec"], mew=0.8,
                zorder=4)
        ax.annotate(label, (spec[0], acc[0]), xytext=offset, textcoords="offset points", ha=ha, va=va, color=INK,
                    linespacing=1.25)
    st = styles["reference"]
    ax.plot([marginal_spec[1], marginal_spec[2]], [1.0, 1.0], color=INK, lw=0.8, solid_capstyle="butt", zorder=3)
    ax.plot(marginal_spec[0], 1.0, ls="none", marker=st["marker"], ms=st["ms"], mfc=st["mfc"], mec=st["mec"], zorder=4)
    ax.annotate("Marginal applicant method", (marginal_spec[0], 1.0), xytext=(10, 0), textcoords="offset points",
                ha="left", va="center", color=INK)

    ax.set_xlim(0, 1.06)
    ax.set_ylim(0, 1.08)
    ticks = [0, 0.25, 0.5, 0.75, 1.0]
    ax.set_xticks(ticks, ["0", "0.25", "0.50", "0.75", "1"])
    ax.set_yticks(ticks, ["0", "0.25", "0.50", "0.75", "1"])
    ax.set_xlabel("Overlap of two applicants’ reason sets (E3)")
    ax.set_ylabel("Share of the marginal applicant method’s reasons named (E2)")
    save(fig, out_dir, "fig_accuracy_specificity", qa_dir)
    rows = [[label.replace("\n", " "), *spec, *acc] for _, _, label, _, _, _, _, spec, acc in values]
    rows.append(["Marginal applicant method", *marginal_spec, 1.0, 1.0, 1.0])
    write_values(
        out_dir / "fig_accuracy_specificity.csv",
        ["notice", "overlap", "overlap_low", "overlap_high", "share_named", "share_named_low", "share_named_high"],
        rows,
    )


def fig_model_choice(methods: dict, models: dict, out_dir: Path, qa_dir: Path | None) -> None:
    order = ["heloc", "south_german_credit", "give_me_some_credit", "default_of_credit_card_clients"]
    rows = [(DATASET_LABEL[d], ratio_ci({d: methods[d]}), ratio_ci({d: models[d]}), len(models[d])) for d in order]
    rows.append(("All four datasets", ratio_ci(methods), ratio_ci(models), sum(len(v) for v in models.values())))
    ypos = [0, 1, 2, 3, 4.4]

    fig, ax = plt.subplots(figsize=(112 * MM, 42 * MM))
    offset = 0.15
    for (label, meth, mod, n), y in zip(rows, ypos):
        ax.plot([100 * meth[1], 100 * meth[2]], [y - offset] * 2, color=GREY_DARK, lw=0.8, solid_capstyle="butt")
        ax.plot([100 * mod[1], 100 * mod[2]], [y + offset] * 2, color=BLUE, lw=0.8, solid_capstyle="butt")
        ax.plot(100 * meth[0], y - offset, "o", ms=4, mfc="white", mec=GREY_DARK, mew=0.8, clip_on=False)
        ax.plot(100 * mod[0], y + offset, "o", ms=4, mfc=BLUE, mec=BLUE, mew=0.8, clip_on=False)
    ax.axhline(3.7, color=GREY_LIGHT, lw=0.6)
    ax.set_yticks(ypos, [label for label, *_ in rows])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(4.9, -0.6)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Applicants whose first reason differs (%)")
    ax.spines["left"].set_visible(False)

    value_columns(ax, -0.6, ("Methods", "Models"),
                  [(y, pct_label(meth[0]), pct_label(mod[0])) for (_, meth, mod, _), y in zip(rows, ypos)], (1.13, 1.25))
    handles = [
        Line2D([], [], marker="o", ls="none", ms=4, mfc="white", mec=GREY_DARK, mew=0.8,
               label="Two commentary methods, same model"),
        Line2D([], [], marker="o", ls="none", ms=4, mfc=BLUE, mec=BLUE, mew=0.8,
               label="Two equally accurate models, marginal applicant method"),
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.04), ncol=1, handletextpad=0.3,
              borderaxespad=0, labelspacing=0.3)
    save(fig, out_dir, "fig_model_choice", qa_dir)
    write_values(
        out_dir / "fig_model_choice.csv",
        ["data", "methods_differ", "methods_low", "methods_high", "models_differ", "models_low", "models_high",
         "applicants"],
        [[label, *meth, *mod, n] for label, meth, mod, n in rows],
    )


# --------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--denial-dir", required=True, help="cslaw_denial_<tag> directory (bundles and analysis/)")
    parser.add_argument("--source-dir", required=True, help="directory with the original v2 bundles (notices before the direction rule)")
    parser.add_argument("--tag", default="r1", help="run tag of the v2 bundles (cslaw_v2_credit_<tag>, ...)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--qa-dir", default=None, help="optional directory for PNG previews and alignment reports")
    args = parser.parse_args()

    denial_dir = Path(args.denial_dir)
    analysis = denial_dir / "analysis"
    source_dir = Path(args.source_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    qa_dir = Path(args.qa_dir) if args.qa_dir else None
    if qa_dir is not None:
        qa_dir.mkdir(parents=True, exist_ok=True)

    before: dict = {}
    after: dict = {}
    for name, source in (("credit", f"cslaw_v2_credit_{args.tag}"), ("nullbaselines", f"cslaw_v2_credit_nullbaselines_{args.tag}")):
        before.update(direction_units(load_bundle(source_dir / source)))
        after.update(direction_units(load_bundle(denial_dir / name)))
    fig_e1_direction(before, after, out_dir, qa_dir)
    fig_accuracy_specificity(accuracy_units(analysis), specificity_pooled(analysis), out_dir, qa_dir)
    fig_model_choice(*model_choice_units(analysis), out_dir, qa_dir)


if __name__ == "__main__":
    main()
