#!/usr/bin/env python3
"""Draw the full synthesis dossier for one candidate, from lipid to catalogue material.

This is the figure that carries the synthesis-grounded claim. A generated lipid is
decomposed into its three Ugi components, each component is either purchased or reduced
by a forward-verified disconnection, and every branch terminates in a material with a
supplier record. Nothing here is a route proposal from a search model; both tail routes
are published chemistry applied to a generated structure.

The figure is authored at the width it is printed at. Drawing it oversized and letting
LaTeX shrink it is what made the annotations illegible: at 13 inches wide scaled into a
6.5 inch measure, a 10 pt label reaches the page at under 5 pt. Type sizes below are
therefore the sizes the reader actually sees, and molecule rasters are generated at a
resolution derived from their printed size so bond weights survive.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

REPO_DEFAULT = Path(__file__).resolve().parents[1]
for _p in (REPO_DEFAULT / "src",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

PANEL = "results/phase1/ugi_prospective_panel_v6/prospective_panel.jsonl.gz"

# The journal measure is 6.5 in; the conference measure is 5.5 in and scales this by 0.85,
# which the type scale below still survives.
FIG_W, FIG_H = 6.5, 4.55
RASTER_DPI = 600

ROLE_TITLE = {
    "amine_head": "amine head",
    "oxoester_aldehyde_body_tail": "ester-linked aldehyde tail",
    "isocyanide_tail": "isocyanide tail",
}
ROLE_COLOR = {
    "amine_head": "#2E7D6F",
    "oxoester_aldehyde_body_tail": "#C7873B",
    "isocyanide_tail": "#17395B",
}
ROUTE_STEPS = {
    "AGILE Tail A": "esterify, then oxidise",
    "isocyanide from primary amine": "formylate, then dehydrate",
}

INK = "#17191D"
MUTED = "#555A62"
FAINT = "#8A8F96"
HAIR = "#D5D9DD"


def draw(smiles: str, width_px: int, height_px: int) -> bytes | None:
    from rdkit import Chem, rdBase
    from rdkit.Chem.Draw import rdMolDraw2D

    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return None
        drawer = rdMolDraw2D.MolDraw2DCairo(width_px, height_px)
        options = drawer.drawOptions()
        options.clearBackground = False
        # A bond drawn 2 px wide at 600 dpi is a quarter of a point on the page.
        options.bondLineWidth = max(2, round(RASTER_DPI / 72 * 0.55))
        options.multipleBondOffset = 0.15
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, molecule)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    parser.add_argument("--candidate", default="P01")
    parser.add_argument("--output", type=Path,
                        default=Path("manuscript/figures/panel_v6/route_tree.png"))
    args = parser.parse_args()
    repo = args.repo.resolve()
    out = args.output if args.output.is_absolute() else repo / args.output
    out.parent.mkdir(parents=True, exist_ok=True)

    import io

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    panel = [json.loads(line) for line in gzip.open(repo / PANEL, "rt")]
    record = next((r for r in panel if r["candidate_id"] == args.candidate), None)
    if record is None:
        raise SystemExit(f"candidate {args.candidate} not in the panel")

    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Avenir Next", "Helvetica Neue", "DejaVu Sans"],
                         "text.color": INK})
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def place(smiles: str, cx: float, cy: float, w: float, h: float) -> None:
        width_px = max(200, round(w * FIG_W * RASTER_DPI))
        height_px = max(120, round(h * FIG_H * RASTER_DPI))
        png = draw(smiles, width_px, height_px)
        if png is None:
            return
        image = Image.open(io.BytesIO(png))
        ax.imshow(image, extent=(cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2),
                  zorder=3, interpolation="antialiased")

    def rule(x0: float, x1: float, y: float) -> None:
        ax.plot([x0, x1], [y, y], color=HAIR, lw=0.7, zorder=1)

    def connector(x0: float, y0: float, x1: float, y1: float, colour: str) -> None:
        """A thin elbow rather than an arrow into a box: the eye follows the line, and
        nothing competes with the structures themselves."""
        mid = (y0 + y1) / 2
        ax.plot([x0, x0, x1, x1], [y0, mid, mid, y1], color=colour, lw=0.9,
                solid_capstyle="round", zorder=1)

    s = record["scores"]
    ax.text(0.5, 0.988, f"Synthesis dossier for {record['candidate_id']}",
            ha="center", va="top", fontsize=10.5, fontweight="bold")
    ax.text(0.5, 0.945,
            f"predicted transfection {s['oracle_mean']:.1f}   ·   "
            f"{record['synthetic_steps']} steps before the final Ugi assembly",
            ha="center", va="top", fontsize=7.2, color=MUTED)

    ax.text(0.5, 0.902, "GENERATED LIPID", ha="center", va="center",
            fontsize=6.4, fontweight="bold", color=FAINT)
    place(record["canonical_product"], 0.5, 0.802, 0.66, 0.175)
    rule(0.10, 0.90, 0.698)

    columns = [0.185, 0.5, 0.815]
    dossier = record["route_dossier"]
    ax.text(0.5, 0.670, "COMPONENTS", ha="center", va="center",
            fontsize=6.4, fontweight="bold", color=FAINT)

    for column, component in zip(columns, dossier):
        role = component["role"]
        colour = ROLE_COLOR[role]
        connector(0.5, 0.648, column, 0.624, colour)
        ax.text(column, 0.602, ROLE_TITLE[role], ha="center", fontsize=7.6,
                fontweight="bold", color=colour)
        place(component["canonical_smiles"], column, 0.515, 0.29, 0.140)

        if component["action"] == "purchase":
            ax.text(column, 0.428, "purchase", ha="center", fontsize=7.0,
                    fontweight="bold", color=colour)
            ax.text(column, 0.398, f"{component['vendor_count']} suppliers", ha="center",
                    fontsize=6.6, color=MUTED)
            continue

        route = component.get("route", "")
        ax.text(column, 0.428, ROUTE_STEPS.get(route, route), ha="center", fontsize=7.0,
                fontweight="bold", color=colour)
        ax.text(column, 0.398, f"{component.get('steps', 0)} steps", ha="center",
                fontsize=6.6, color=MUTED)

        materials = component.get("starting_materials", [])
        span = 0.200
        xs = ([column] if len(materials) == 1
              else [column - span / 2 + i * span / (len(materials) - 1)
                    for i in range(len(materials))])
        for mx, material in zip(xs, materials):
            connector(column, 0.368, mx, 0.312, colour)
            place(material["canonical_smiles"], mx, 0.244, 0.175, 0.100)
            ax.text(mx, 0.176, material.get("role", "material"), ha="center",
                    fontsize=6.4, color=MUTED)
            ax.text(mx, 0.148, f"{material.get('vendor_count', '?')} suppliers",
                    ha="center", fontsize=6.4, fontweight="bold", color=INK)

    rule(0.10, 0.90, 0.104)
    ax.text(0.5, 0.074, "every branch terminates in a catalogue material", ha="center",
            fontsize=7.0, fontweight="bold", color=INK)
    ax.text(0.5, 0.044,
            "Both preparations are published chemistry applied to generated structures. "
            "Supplier counts are from a dated procurement snapshot.",
            ha="center", va="top", fontsize=6.2, color=MUTED)
    ax.text(0.5, 0.020,
            "Route completeness predicts that a documented preparation exists from "
            "purchasable material; it is not evidence that a specific reaction will succeed.",
            ha="center", va="top", fontsize=6.2, color=MUTED)

    fig.savefig(out, dpi=RASTER_DPI, facecolor="white")
    print(f"wrote {out.relative_to(repo)} for {record['candidate_id']} "
          f"({FIG_W}x{FIG_H} in, authored at print size)")


if __name__ == "__main__":
    main()
