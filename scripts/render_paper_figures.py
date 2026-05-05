#!/usr/bin/env python3
"""Render the LegalBench-XAI paper figures from released bundle CSVs.

Usage (from repo root):
    uv run --with matplotlib --with numpy --with pandas \
        python scripts/render_paper_figures.py

Outputs three PNG/PDF figure pairs into manuscript/latex manuscript/figures/:
    cross_domain_tradeoff.{png,pdf}  -- Figure 1 candidate (replaces overall_tradeoff)
    family_radar.{png,pdf}           -- Figure 2 candidate (six-dim quality vector)
    pairwise_wins_grid.{png,pdf}     -- Figure 3 candidate (matched-group wins)
"""
from __future__ import annotations

import math
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

REPO_ROOT = Path(__file__).resolve().parent.parent
CREDIT_BUNDLE = REPO_ROOT / "benchmark/outputs/four_dataset_real_highbudget_heldout_refresh"
NONCREDIT_BUNDLE = REPO_ROOT / "benchmark/outputs/cross_domain_strongest_real_full_cluster"
OUT_DIR = REPO_ROOT / "manuscript/latex manuscript/figures"

EXPLAINERS = ["anchors", "dice", "lime", "shap", "wachter"]
EXPLAINER_LABELS = {
    "anchors": "Anchors",
    "dice": "DiCE",
    "lime": "LIME",
    "shap": "SHAP",
    "wachter": "Wachter",
}
COLORS = {
    "anchors": "#2E8B57",
    "dice": "#DD7F2A",
    "lime": "#3771B5",
    "shap": "#C0392B",
    "wachter": "#674EA7",
}
METRICS = ["faithfulness", "stability", "comprehensibility", "css", "actionability", "alf"]
METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "stability": "Stability",
    "comprehensibility": "Comprehensibility",
    "css": "CSS",
    "actionability": "Actionability",
    "alf": "ALF",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 220,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
    }
)


def load_by_explainer(bundle: Path) -> pd.DataFrame:
    df = pd.read_csv(bundle / "by_explainer.csv")
    df = df.set_index("explainer_id")
    return df


def save(fig, name: str) -> None:
    out_png = OUT_DIR / f"{name}.png"
    out_pdf = OUT_DIR / f"{name}.pdf"
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"wrote {out_png.relative_to(REPO_ROOT)}")
    print(f"wrote {out_pdf.relative_to(REPO_ROOT)}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1 candidate: side-by-side cross-domain CSS x Actionability scatter
# ---------------------------------------------------------------------------
def fig_cross_domain_tradeoff() -> None:
    credit = load_by_explainer(CREDIT_BUNDLE)
    noncredit = load_by_explainer(NONCREDIT_BUNDLE)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    panels = [
        (axes[0], credit, "Credit Subpanel (4 datasets, 24 groups)"),
        (axes[1], noncredit, "Non-Credit Subpanel (HMDA, NY Pretrial, FDA; 18 groups)"),
    ]
    for ax, df, title in panels:
        # Background bands highlighting the recourse vs. non-recourse separation.
        ax.axhspan(0.40, 0.85, color="#DD7F2A", alpha=0.06, zorder=0)
        ax.axhspan(0.0, 0.40, color="#3771B5", alpha=0.05, zorder=0)
        for ex in EXPLAINERS:
            x = df.loc[ex, "mean_css"]
            y = df.loc[ex, "mean_actionability"]
            xerr = [
                [x - df.loc[ex, "mean_css_ci_low"]],
                [df.loc[ex, "mean_css_ci_high"] - x],
            ]
            yerr = [
                [y - df.loc[ex, "mean_actionability_ci_low"]],
                [df.loc[ex, "mean_actionability_ci_high"] - y],
            ]
            ax.errorbar(
                x,
                y,
                xerr=xerr,
                yerr=yerr,
                fmt="o",
                color=COLORS[ex],
                markersize=9,
                capsize=2.5,
                elinewidth=1.1,
                alpha=0.95,
                label=EXPLAINER_LABELS[ex],
            )
        ax.text(
            0.572,
            0.79,
            "recourse-family band",
            fontsize=8,
            color="#8a4f12",
            alpha=0.85,
        )
        ax.text(
            0.572,
            0.05,
            "attribution / rule-family band",
            fontsize=8,
            color="#1f426f",
            alpha=0.85,
        )
        ax.set_xlabel("Contestability Support Score (CSS)")
        ax.set_xlim(0.55, 1.0)
        ax.set_ylim(0.0, 0.85)
        ax.set_title(title)
        ax.grid(True, alpha=0.25, linestyle=":")
    axes[0].set_ylabel("Unified Actionability")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.suptitle(
        "Family-level subject-facing tradeoffs replicate across legal-decision domains",
        y=1.02,
        fontsize=12,
        fontweight="bold",
    )
    save(fig, "cross_domain_tradeoff")


# ---------------------------------------------------------------------------
# Figure 2 candidate: six-dimensional quality vector radar
# ---------------------------------------------------------------------------
def fig_family_radar() -> None:
    credit = load_by_explainer(CREDIT_BUNDLE)
    noncredit = load_by_explainer(NONCREDIT_BUNDLE)

    angles = np.linspace(0, 2 * math.pi, len(METRICS), endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.6), subplot_kw={"projection": "polar"})
    panels = [(axes[0], credit, "Credit Subpanel"), (axes[1], noncredit, "Non-Credit Subpanel")]
    for ax, df, title in panels:
        for ex in EXPLAINERS:
            vals = []
            for m in METRICS:
                v = df.loc[ex, f"mean_{m}"]
                if pd.isna(v):
                    v = 0.0  # ALF undefined for non-recourse families -> floor at 0
                vals.append(float(v))
            vals += vals[:1]
            ax.plot(
                angles,
                vals,
                color=COLORS[ex],
                linewidth=2.0,
                label=EXPLAINER_LABELS[ex],
            )
            ax.fill(angles, vals, color=COLORS[ex], alpha=0.10)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS])
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7, color="#666")
        ax.set_rlabel_position(30)  # between Faithfulness (0 deg) and Stability (60 deg)
        ax.set_title(title, y=1.10, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.tick_params(axis="x", pad=8)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.suptitle(
        "Six-dimensional explanation-quality profiles per family",
        y=1.02,
        fontsize=12,
        fontweight="bold",
    )
    save(fig, "family_radar")


# ---------------------------------------------------------------------------
# Figure 3 candidate: 6-panel pairwise-wins grid (credit subpanel)
# ---------------------------------------------------------------------------
def build_pairwise_matrix(df_metric: pd.DataFrame) -> np.ndarray:
    M = np.full((len(EXPLAINERS), len(EXPLAINERS)), np.nan)
    for _, row in df_metric.iterrows():
        a, b = row["explainer_a"], row["explainer_b"]
        if a in EXPLAINERS and b in EXPLAINERS:
            ia, ib = EXPLAINERS.index(a), EXPLAINERS.index(b)
            M[ia, ib] = row["win_rate_a"]
            M[ib, ia] = row["win_rate_b"]
    return M


def fig_pairwise_wins() -> None:
    credit_pw = pd.read_csv(CREDIT_BUNDLE / "pairwise_wins.csv")

    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.2))
    last_im = None
    for idx, m in enumerate(METRICS):
        ax = axes[idx // 3, idx % 3]
        df_m = credit_pw[credit_pw["metric"] == m]
        M = build_pairwise_matrix(df_m)
        im = ax.imshow(M, cmap="RdBu_r", vmin=0, vmax=1, aspect="equal")
        last_im = im
        ax.set_xticks(range(len(EXPLAINERS)))
        ax.set_yticks(range(len(EXPLAINERS)))
        ax.set_xticklabels(
            [EXPLAINER_LABELS[e] for e in EXPLAINERS],
            rotation=30,
            ha="right",
            fontsize=9,
        )
        ax.set_yticklabels([EXPLAINER_LABELS[e] for e in EXPLAINERS], fontsize=9)
        ax.set_title(METRIC_LABELS[m], fontweight="bold")
        for i in range(len(EXPLAINERS)):
            for j in range(len(EXPLAINERS)):
                v = M[i, j]
                if not np.isnan(v):
                    color = "white" if (v > 0.7 or v < 0.3) else "black"
                    ax.text(
                        j,
                        i,
                        f"{v:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color=color,
                    )
                else:
                    ax.text(
                        j, i, "-", ha="center", va="center", fontsize=10, color="#888"
                    )
        ax.tick_params(axis="both", length=0)

    fig.colorbar(
        last_im,
        ax=axes.ravel().tolist(),
        label="Win rate of row explainer over column explainer",
        shrink=0.8,
        pad=0.02,
    )
    fig.suptitle(
        "Pairwise win rates on matched dataset-track-model groups (credit subpanel, 24 groups)",
        y=1.00,
        fontsize=12,
        fontweight="bold",
    )
    save(fig, "pairwise_wins_grid")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_cross_domain_tradeoff()
    fig_family_radar()
    fig_pairwise_wins()
    print("done.")
