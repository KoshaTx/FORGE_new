#!/usr/bin/env python3
"""Draw the forty locked designs grouped by how they were selected.

Written for the chemist who has to make them, so each structure carries the two facts that
determine bench work: how many synthetic steps it needs and which components must be
prepared rather than bought. Designs appear in blocks matching the frozen selection rule,
not in one undifferentiated grid.

The figure is authored at the width it is printed at. Drawing it at fourteen inches and
letting LaTeX shrink it into a six-inch measure put the annotations under seven points and
the bond strokes under a quarter point, and pushed the two per-structure annotations into
each other. Sizes below are the sizes the reader sees.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import sys
from pathlib import Path

REPO_DEFAULT = Path(__file__).resolve().parents[1]
for _p in (REPO_DEFAULT / "src",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

PANEL = "results/phase1/ugi_prospective_panel_v6/prospective_panel.jsonl.gz"

FIG_W = 6.5          # the journal measure; the conference build scales this by 0.85
CELL_H = 1.62        # inches per structure row
HEADER_H = 0.72
FOOTER_H = 0.52
RASTER_DPI = 600

TIER_COLOR = {
    "qualified_role_holdout": "#17395B",
    "exploratory_weak_absolute_head_generalization": "#2E7D6F",
    "exploratory_weak_absolute_aldehyde_generalization": "#C7873B",
    "exploratory_simultaneous_exact_new_tails": "#A6474F",
}
TIER_SHORT = {
    "qualified_role_holdout": "T1",
    "exploratory_weak_absolute_head_generalization": "T2",
    "exploratory_weak_absolute_aldehyde_generalization": "T3",
    "exploratory_simultaneous_exact_new_tails": "T4",
}
SCORE_ONLY = 12  # the first block of the high arm is taken on predicted activity alone

INK = "#17191D"
MUTED = "#555A62"
HAIR = "#C9CDD2"


def render(smiles: str, px: int, py: int) -> bytes | None:
    from rdkit import Chem, rdBase
    from rdkit.Chem.Draw import rdMolDraw2D

    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return None
        drawer = rdMolDraw2D.MolDraw2DCairo(px, py)
        options = drawer.drawOptions()
        options.clearBackground = False
        # Scaled to the raster resolution: a fixed 2 px bond vanishes at 600 dpi.
        options.bondLineWidth = max(2, round(RASTER_DPI / 72 * 0.5))
        options.multipleBondOffset = 0.15
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, molecule)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    parser.add_argument(
        "--output", type=Path, default=Path("paper/figures/panel_v6/panel_40_structures.png")
    )
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--per-page", type=int, default=12,
                        help="structures per page; forty cannot be legible on one page")
    args = parser.parse_args()
    repo = args.repo.resolve()
    out = args.output if args.output.is_absolute() else repo / args.output
    out.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    from PIL import Image

    records = [json.loads(line) for line in gzip.open(repo / PANEL, "rt")]
    high = sorted((r for r in records if r["arm"] == "HIGH"),
                  key=lambda r: r["selection_order_within_arm"])
    low = sorted((r for r in records if r["arm"] == "LOW"),
                 key=lambda r: r["selection_order_within_arm"])

    blocks = [
        ("Selected on predicted activity alone",
         "the twelve highest-ranked designs in the eligible pool", high[:SCORE_ONLY]),
        ("Selected preferring chemistry outside the source library",
         "highest-ranked remaining designs, novelty preferred where it costs no predicted activity",
         high[SCORE_ONLY:]),
        ("Low-ranked controls",
         "lowest-ranked designs of the same evidence class, to test whether the ranking discriminates",
         low),
    ]

    # Paginate within each block. A block never starts mid-page, which is what made the
    # header collide with the row above it and pushed the last block off the page.
    pages = []
    for title, subtitle, items in blocks:
        chunks = [items[i:i + args.per_page] for i in range(0, len(items), args.per_page)]
        for part, chunk in enumerate(chunks):
            heading = title if part == 0 else f"{title} (continued)"
            pages.append((heading, subtitle, chunk))

    columns = args.columns
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Avenir Next", "Helvetica Neue", "DejaVu Sans"],
                         "text.color": INK})

    written = []
    for page_index, (title, subtitle, items) in enumerate(pages, start=1):
        rows = -(-len(items) // columns)
        fig_h = rows * CELL_H + HEADER_H + FOOTER_H
        fig = plt.figure(figsize=(FIG_W, fig_h))
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # Work in figure fractions so type stays in points regardless of page length.
        fh = fig_h
        ax.text(0.5, 1 - 0.16 / fh, f"The forty prospectively locked designs  ({page_index} of {len(pages)})",
                ha="center", va="top", fontsize=10.5, fontweight="bold")
        ax.text(0.035, 1 - 0.40 / fh, title, ha="left", va="center", fontsize=8.4,
                fontweight="bold")
        ax.text(0.035, 1 - 0.55 / fh, subtitle, ha="left", va="center", fontsize=6.4,
                color=MUTED)
        ax.plot([0.035, 0.965], [1 - 0.63 / fh] * 2, color=HAIR, lw=0.7)

        top = 1 - HEADER_H / fh
        cell_w = 0.93 / columns
        cell_frac = CELL_H / fh
        for index, record in enumerate(items):
            col = index % columns
            row = index // columns
            cx = 0.035 + (col + 0.5) * cell_w
            cy = top - (row + 0.5) * cell_frac
            tier = record["authority_tier"]
            colour = TIER_COLOR.get(tier, MUTED)

            ax.add_patch(FancyBboxPatch(
                (cx - cell_w / 2 + 0.008, cy - cell_frac / 2 + 0.012 / fh * 10),
                cell_w - 0.016, cell_frac - 0.24 / fh,
                boxstyle="round,pad=0.002,rounding_size=0.01",
                linewidth=0.6, edgecolor=colour, facecolor="#FFFFFF", zorder=1))

            half_w, half_h = cell_w * 0.44, cell_frac * 0.26
            png = render(record["canonical_product"],
                         round(half_w * 2 * FIG_W * RASTER_DPI),
                         round(half_h * 2 * fig_h * RASTER_DPI))
            if png is not None:
                image = Image.open(io.BytesIO(png))
                ax.imshow(image, extent=(cx - half_w, cx + half_w,
                                         cy - half_h + 0.04 / fh, cy + half_h + 0.04 / fh),
                          zorder=3, interpolation="antialiased")

            steps = record["synthetic_steps"]
            prepared = sum(1 for c in record["route_dossier"] if c["action"] != "purchase")
            ax.text(cx - cell_w / 2 + 0.022, cy + cell_frac / 2 - 0.15 / fh,
                    record["candidate_id"], ha="left", va="center", fontsize=7.6,
                    fontweight="bold", color=colour)
            ax.text(cx + cell_w / 2 - 0.022, cy + cell_frac / 2 - 0.15 / fh,
                    TIER_SHORT.get(tier, "?"), ha="right", va="center", fontsize=6.4,
                    color=colour)
            # Two lines rather than one: at this measure the left and right annotations
            # of a single line run into each other.
            ax.text(cx, cy - cell_frac / 2 + 0.30 / fh,
                    f"predicted {record['scores']['oracle_mean']:.1f}",
                    ha="center", va="center", fontsize=6.2, color=INK)
            ax.text(cx, cy - cell_frac / 2 + 0.17 / fh,
                    f"{steps} steps · {prepared} tail{'s' if prepared != 1 else ''} to prepare",
                    ha="center", va="center", fontsize=6.2, color=MUTED)

        ax.text(0.035, 0.20 / fh,
                "Predicted values are on the measured transfection scale. Steps count the "
                "preparations required before the final Ugi\nassembly, common to every design. "
                "T1 unseen isocyanide only, T2 unseen amine head, T3 unseen aldehyde tail, "
                "T4 both tails unseen.",
                ha="left", va="center", fontsize=5.8, color=MUTED, linespacing=1.5)

        target = out.with_name(f"{out.stem}_p{page_index}{out.suffix}")
        fig.savefig(target, dpi=RASTER_DPI, facecolor="white")
        plt.close(fig)
        written.append(target.name)

    print(f"wrote {len(written)} pages at {FIG_W} in wide: {', '.join(written)}")


if __name__ == "__main__":
    main()
