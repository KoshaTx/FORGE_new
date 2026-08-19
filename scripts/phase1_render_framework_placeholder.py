#!/usr/bin/env python3
"""Draw the placeholder that reserves Figure 1's framework panel.

The final artwork is to be drawn in illustration software. A blank box would tell the
person drawing it nothing, so this states the claim the panel has to make, names the
objects that must appear, and marks the one contrast the panel exists to carry. It is
deliberately obvious as a placeholder: no artwork here should be mistaken for final.
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

REPO_DEFAULT = Path(__file__).resolve().parents[1]

INK = "#17191D"
MUTED = "#6B7078"
HAIR = "#C9CDD2"
ACCENT = "#A6474F"
WASH = "#F4F5F6"

CLAIM = ("An ordinary graph generator produces a product. FORGE produces a product whose "
         "regions carry their precursor origin, so the components fall out of the sample.")

ROWS = [
    ("Conventional molecular generation",
     ["noise / source marginal", "generative trajectory", "product graph"],
     "the synthetic interpretation has to be recovered afterwards, from the finished "
     "structure alone"),
    ("FORGE, reaction-resolved generation",
     ["noise / source marginal", "generative trajectory\nover atoms, bonds\nand precursor "
      "origin", "product graph\n+ origin partition", "exact components", "route to purchasable\nmaterial"],
     "origin is generated jointly with structure, so the partition into amine, aldehyde "
     "and isocyanide is exact rather than inferred"),
]

NOTES = [
    "Must be unmistakable at a glance: NO STORED COMPONENT IDENTITIES enter the model. "
    "No fragment library, no component vocabulary, no reaction template is retrieved at "
    "sampling time.",
    "Show the three components as structures growing out of the product's coloured "
    "regions, not as icons pulled from a shelf.",
    "Do not draw GRU layers, hidden dimensions, loss terms or the categorical state "
    "tables. The panel carries the formulation, not the implementation.",
    "Colour is the load-bearing device: one hue per precursor role, used identically "
    "wherever those roles appear in the rest of the paper.",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    parser.add_argument("--output", type=Path,
                        default=Path("manuscript/figures/framework/flow_framework_placeholder.png"))
    args = parser.parse_args()
    repo = args.repo.resolve()
    out = args.output if args.output.is_absolute() else repo / args.output
    out.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Rectangle

    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Avenir Next", "Helvetica Neue", "DejaVu Sans"],
                         "text.color": INK})

    fig = plt.figure(figsize=(6.5, 3.85))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.add_patch(Rectangle((0.012, 0.012), 0.976, 0.976, facecolor=WASH,
                           edgecolor=ACCENT, lw=1.6, linestyle=(0, (7, 5)), zorder=0))

    ax.text(0.038, 0.945, "PLACEHOLDER", fontsize=6.4, fontweight="bold", color=ACCENT,
            va="top")
    ax.text(0.038, 0.905, "Figure 1b  ·  the reaction-resolved generative formulation",
            fontsize=9.9, fontweight="bold", va="top")
    ax.text(0.038, 0.862, "final artwork to be drawn in illustration software",
            fontsize=6.4, color=MUTED, va="top")
    ax.plot([0.038, 0.962], [0.836, 0.836], color=HAIR, lw=1.1)

    ax.text(0.038, 0.806, "The panel must land this:", fontsize=6.1, fontweight="bold",
            va="top", color=MUTED)
    cy = 0.778
    for line in textwrap.wrap(CLAIM, 104):
        ax.text(0.038, cy, line, fontsize=7.1, va="top")
        cy -= 0.034

    y = 0.688
    for title, stages, note in ROWS:
        ax.text(0.038, y, title, fontsize=7.0, fontweight="bold", va="top")
        y -= 0.040
        n = len(stages)
        span = 0.924 / n
        for i, stage in enumerate(stages):
            cx = 0.038 + (i + 0.5) * span
            ax.add_patch(FancyBboxPatch((cx - span * 0.40, y - 0.088), span * 0.80, 0.086,
                                        boxstyle="round,pad=0.004,rounding_size=0.012",
                                        facecolor="#FFFFFF", edgecolor=HAIR, lw=1.1))
            ax.text(cx, y - 0.045, stage, ha="center", va="center", fontsize=5.6,
                    color=INK, linespacing=1.35)
            if i:
                bx = 0.038 + i * span
                ax.annotate("", xy=(bx + span * 0.09, y - 0.045),
                            xytext=(bx - span * 0.09, y - 0.045),
                            arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.3))
        y -= 0.112
        ax.text(0.038, y, note, fontsize=5.8, color=MUTED, va="top")
        y -= 0.052

    ax.plot([0.038, 0.962], [y + 0.012, y + 0.012], color=HAIR, lw=1.1)
    y -= 0.024
    ax.text(0.038, y, "Constraints on the drawing", fontsize=6.1, fontweight="bold",
            va="top", color=MUTED)
    y -= 0.030
    for note in NOTES:
        ax.text(0.045, y, "\u2014", fontsize=5.8, color=HAIR, va="top")
        for line in textwrap.wrap(note, 128):
            ax.text(0.068, y, line, fontsize=5.6, color=INK, va="top")
            y -= 0.024
        y -= 0.010

    fig.savefig(out, dpi=600, facecolor="white")
    print(f"wrote {out.relative_to(repo)}")


if __name__ == "__main__":
    main()
