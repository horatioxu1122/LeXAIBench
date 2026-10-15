"""Figure 1 of the CS&Law paper: how a denial becomes a notice and how each notice is tested.

Draws the schematic once and exports it as PDF (for LaTeX) and SVG (editable,
live text). Box widths come from measured Arial label widths; one drawing unit
is one point at final size, with y running down the page. SVG element ids
follow the figure brief: n:* nodes, c:* containers, e:* edges, l:* edge
labels, s:* step markers. Reads no data.

Usage:
    python benchmark/cslaw/paper_schematic.py --out-dir paper/figures [--preview-dir DIR]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Polygon  # noqa: E402
from matplotlib.textpath import TextToPath  # noqa: E402

NAME = "fig_pipeline"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

TTP = TextToPath()
INK = "#272727"
GREY_TEXT = "#4D4D4D"
GREY_LINE = "#6B6B6B"
BOX_FILL = "#F2F2F2"
BOX_LINE = "#8C8C8C"
ACCENT = "#0F4D92"
ACCENT_FILL = "#DCE6F2"

TITLE_PT, ITEM_PT, STEP_PT = 7.0, 6.5, 6.0
CAP = 0.716  # Arial cap height, em
DESC = 0.212  # Arial descent, em
LINE = 8.4  # baseline to baseline, items
PAD_X, PAD_Y = 6.0, 4.5
STEP_R = 4.6

# Labels as in the paper.
MODEL = ("Model", ["Logistic regression", "Explainable boosting machine", "Gradient boosted trees"])
METHOD = ("Explanation method", ["SHAP, LIME, DiCE,", "Anchors, Wachter"])
NOTICE = ("Notice", ["Up to four principal reasons", "after the direction rule"])
COMMENTARY = ("Commentary methods", ["Marginal applicant method", "Average applicant method"])
REFERENCE = ("Reference notices", ["Canned notice", "Random notice"])
TESTS = ["E1 Direction", "E2 Accuracy", "E3 Specificity", "E4 Selection", "E5 Prohibited bases", "E6 Recourse"]


def text_width(s: str, size: float, bold: bool = False) -> float:
    fp = FontProperties(family="Arial", weight="bold" if bold else "normal", size=size)
    return TTP.get_text_width_height_descent(s, fp, ismath=False)[0]


def block_height(n_items: int) -> float:
    return TITLE_PT * CAP + LINE * n_items + ITEM_PT * DESC


def width_of(content: tuple[str, list[str]]) -> float:
    title, items = content
    return max([text_width(title, TITLE_PT, True)] + [text_width(i, ITEM_PT) for i in items]) + 2 * PAD_X


def draw(out_dir: Path, preview_dir: Path | None = None) -> None:
    width = 390.0
    left, right = 1.5, 1.5
    inset = 7.0  # frame to test boxes, so the incoming arrows end on the frame of the whole group
    col_a = max(width_of(MODEL), width_of(COMMENTARY), width_of(REFERENCE))
    col_b = width_of(METHOD)
    col_c = width_of(NOTICE)
    test_w = max(text_width(t, ITEM_PT) for t in TESTS) + 2 * 5.0
    col_d = test_w + 2 * inset
    gap = (width - left - right - (col_a + col_b + col_c + col_d)) / 3
    xa = left
    xb = xa + col_a + gap
    xc = xb + col_b + gap
    xd = xc + col_c + gap

    step_y = 1.0 + STEP_R  # step markers sit above the boxes, clear of every outline
    y1 = step_y + STEP_R + 2.5
    h1 = round(block_height(3) + 2 * PAD_Y)
    h2 = round(block_height(2) + 2 * PAD_Y)
    y2 = y1 + h1 + 16.0
    y3 = y2 + h2 + 10.0
    bottom = y3 + h2
    height = bottom + 1.5
    mid1, mid2, mid3 = y1 + h1 / 2, y2 + h2 / 2, y3 + h2 / 2

    fig = plt.figure(figsize=(width / 72, height / 72))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")

    def text(x: float, y: float, s: str, size: float, gid: str, bold: bool = False, color: str = INK,
             ha: str = "center") -> None:
        t = ax.text(x, y, s, fontsize=size, fontweight="bold" if bold else "normal", color=color, ha=ha, va="baseline")
        t.set_gid(gid)

    def box(node_id: str, x: float, y: float, w: float, h: float, title: str, items: list[str]) -> None:
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=2", facecolor=BOX_FILL,
                               edgecolor=BOX_LINE, linewidth=0.6)
        patch.set_gid(node_id)
        ax.add_patch(patch)
        top = y + (h - block_height(len(items))) / 2 + TITLE_PT * CAP
        text(x + w / 2, top, title, TITLE_PT, f"{node_id}:title", bold=True)
        for i, item in enumerate(items):
            text(x + w / 2, top + LINE * (i + 1), item, ITEM_PT, f"{node_id}:item{i + 1}")

    def arrow(edge_id: str, x1: float, ya: float, x2: float, yb: float, main: bool) -> None:
        """Straight horizontal or vertical arrow from (x1, ya) to the tip (x2, yb)."""
        color, lw, head, half = (INK, 1.1, 5.0, 2.5) if main else (GREY_LINE, 0.7, 4.0, 2.0)
        if ya == yb:
            base = x2 - head * (1 if x2 > x1 else -1)
            (line,) = ax.plot([x1, base], [ya, yb], color=color, linewidth=lw, solid_capstyle="butt")
            tip = [(x2, yb), (base, yb - half), (base, yb + half)]
        else:
            base = yb - head * (1 if yb > ya else -1)
            (line,) = ax.plot([x1, x2], [ya, base], color=color, linewidth=lw, solid_capstyle="butt")
            tip = [(x2, yb), (x2 - half, base), (x2 + half, base)]
        line.set_gid(edge_id)
        head_patch = Polygon(tip, closed=True, facecolor=color, edgecolor="none")
        head_patch.set_gid(f"{edge_id}:head")
        ax.add_patch(head_patch)

    box("n:model", xa, y1, col_a, h1, *MODEL)
    box("n:explanation-method", xb, y1, col_b, h1, *METHOD)
    box("n:notice", xc, y1, col_c, h1, *NOTICE)
    box("n:commentary-methods", xa, y2, col_a, h2, *COMMENTARY)
    box("n:reference-notices", xa, y3, col_a, h2, *REFERENCE)

    # Six tests: the paper's contribution, the only accent.
    frame = FancyBboxPatch((xd, y1), col_d, bottom - y1, boxstyle="round,pad=0,rounding_size=2", facecolor="#FFFFFF",
                           edgecolor=ACCENT, linewidth=1.1)
    frame.set_gid("c:six-tests")
    ax.add_patch(frame)
    text(xd + col_d / 2, y1 + 4.0 + TITLE_PT * CAP, "Six tests", TITLE_PT, "c:six-tests:title", bold=True, color=ACCENT)
    inner_top = y1 + 4.0 + TITLE_PT * CAP + 4.5
    inner_bottom = bottom - 4.5
    test_gap = 3.0
    test_h = (inner_bottom - inner_top - test_gap * (len(TESTS) - 1)) / len(TESTS)
    for i, label in enumerate(TESTS):
        ty = inner_top + i * (test_h + test_gap)
        patch = FancyBboxPatch((xd + inset, ty), test_w, test_h, boxstyle="round,pad=0,rounding_size=1.5",
                               facecolor=ACCENT_FILL, edgecolor=ACCENT, linewidth=0.5)
        patch.set_gid(f"n:e{i + 1}")
        ax.add_patch(patch)
        text(xd + inset + test_w / 2, ty + test_h / 2 + ITEM_PT * CAP / 2, label, ITEM_PT, f"n:e{i + 1}:label")

    # Main path (heavy) and reference paths (light).
    arrow("e:model-to-method", xa + col_a + 1.0, mid1, xb, mid1, main=True)
    arrow("e:method-to-notice", xb + col_b + 1.0, mid1, xc, mid1, main=True)
    arrow("e:notice-to-tests", xc + col_c + 1.0, mid1, xd, mid1, main=True)
    arrow("e:model-to-commentary", xa + col_a / 2, y1 + h1 + 1.0, xa + col_a / 2, y2, main=False)
    arrow("e:commentary-to-tests", xa + col_a + 1.0, mid2, xd, mid2, main=False)
    arrow("e:reference-to-tests", xa + col_a + 1.0, mid3, xd, mid3, main=False)

    # Edge labels: beside the vertical arrow, and above the two long reference arrows.
    span_mid = (xa + col_a + xd) / 2
    text(xa + col_a / 2 + 5.0, (y1 + h1 + y2) / 2 + ITEM_PT * CAP / 2, "per factor contributions", ITEM_PT,
         "l:contributions", color=GREY_TEXT, ha="left")
    text(span_mid, mid2 - 3.5, "contributions and the reasons selected from them", ITEM_PT, "l:commentary-reasons",
         color=GREY_TEXT)
    text(span_mid, mid3 - 3.5, "built for the same applicants", ITEM_PT, "l:same-applicants", color=GREY_TEXT)

    # Step markers for the main path, above the upper-left corner of each box.
    for number, x_left in enumerate((xa, xb, xc, xd), 1):
        circle = Circle((x_left + STEP_R, step_y), STEP_R, facecolor="#FFFFFF", edgecolor=INK, linewidth=0.7)
        circle.set_gid(f"s:{number}")
        ax.add_patch(circle)
        text(x_left + STEP_R, step_y + STEP_PT * CAP / 2, str(number), STEP_PT, f"s:{number}:number", bold=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{NAME}.pdf", metadata={"CreationDate": None})
    fig.savefig(out_dir / f"{NAME}.svg", metadata={"Date": None})
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview_dir / f"{NAME}.png", dpi=600)
    plt.close(fig)
    tidy_svg(out_dir / f"{NAME}.svg")
    print(f"wrote {out_dir / NAME}.pdf and .svg: {width:.0f} x {height:.1f} pt "
          f"({width / 72 * 25.4:.1f} x {height / 72 * 25.4:.1f} mm)")


def tidy_svg(path: Path) -> None:
    """Drop the DTD and the no-op rotations, and state font sizes in user units (one unit is one point here),
    which is what the px values in matplotlib's style attributes mean inside this viewBox."""
    svg = path.read_text(encoding="utf-8")
    svg = re.sub(r"<!DOCTYPE[^>]*>\s*", "", svg, count=1, flags=re.S)
    svg = re.sub(r' transform="rotate\(-?0 [^)]*\)"', "", svg)

    def font_size_attribute(match: re.Match) -> str:
        style = match.group(1)
        size = re.search(r"font-size:\s*([\d.]+)px;?\s*", style)
        if not size:
            return match.group(0)
        return f'<text font-size="{size.group(1)}" style="{style.replace(size.group(0), "")}"'

    svg = re.sub(r'<text style="([^"]*)"', font_size_attribute, svg)
    path.write_text(svg, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--preview-dir", default=None, help="optional directory for a 600 dpi PNG preview")
    args = parser.parse_args()
    draw(Path(args.out_dir), Path(args.preview_dir) if args.preview_dir else None)


if __name__ == "__main__":
    main()
