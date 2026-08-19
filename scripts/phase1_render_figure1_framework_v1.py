#!/usr/bin/env python3
"""Render Figure 1: the FORGE framework, from chemistry-informed generation to purchasable material.

Four panels, one argument:

  a  chemistry is an inductive bias, not an inventory
  b  chemistry enters the LEARNED flow, not just its output
  c  exact component projection, verified rather than predicted
  d  a route back to purchasable starting materials, earned rather than assumed

Design rules this script follows, because they are the ones that make the figure honest:

  * One hue per precursor role, used identically in every panel and matching nothing else in the
    paper. Colour is the load-bearing device; if a reader learns amine/aldehyde/isocyanide in panel a
    they should read panels c and d without a legend.
  * Panel a states the negative explicitly. The single most misreadable thing about FORGE is that it
    might be selecting components from a catalogue, so the figure says NO COMPONENT IDENTITIES, NO
    FRAGMENT CATALOGUE where a skimming reader cannot miss it.
  * Panel b shows chemistry entering the denoiser, not sitting beside it. The earlier placeholder
    drew a chemistry-neutral flow that got decorated afterwards, and also drew a predicted origin
    variable, which is the withdrawn formulation. Neither appears here.
  * Panels c and d use the real P01 structures rather than generic shapes, so the projection and the
    disconnections are inspectable rather than decorative.
  * Panel d ends at "purchasable starting materials", never at "synthesizable". The distinction is
    the paper's, and the figure must not undo it.

Molecules are drawn by RDKit from the frozen panel artifact; nothing is hand-drawn or idealized.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from rdkit import Chem                               # noqa: E402
from rdkit.Chem import rdDepictor                    # noqa: E402
from rdkit.Chem.Draw import rdMolDraw2D              # noqa: E402

REPO = Path(__file__).resolve().parents[1]
PANEL = "results/phase1/ugi_prospective_panel_v6/prospective_panel.jsonl.gz"

# One hue per role, reused in every panel. Deuteranopia-safe teal/amber/navy rather than red/green.
AMINE, ALDEHYDE, ISOCYANIDE = "#1b7f79", "#b8730b", "#1f3f6e"
CORE, INK, MUTED, PAPER = "#5a5a5a", "#1a1a1a", "#6b6b6b", "#f7f7f5"
ROLE_COLOR = {"amine_head": AMINE, "oxoester_aldehyde_body_tail": ALDEHYDE,
              "isocyanide_tail": ISOCYANIDE}
ROLE_LABEL = {"amine_head": "amine head", "oxoester_aldehyde_body_tail": "aldehyde tail",
              "isocyanide_tail": "isocyanide tail"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def draw_molecule(ax, smiles: str, color: str, width: int = 900, height: int = 320) -> None:
    """Render one molecule into an axes, in a single role colour, no atom labels beyond heteroatoms."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        ax.text(0.5, 0.5, "unparseable", ha="center", va="center", color="red")
        ax.axis("off")
        return
    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    opts = drawer.drawOptions()
    opts.bondLineWidth = 2
    opts.clearBackground = False
    rgb = tuple(int(color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    # Colour every atom the role hue; heteroatoms stay legible because RDKit still draws their labels.
    for atom in mol.GetAtoms():
        opts.setAtomPalette({atom.GetIdx(): rgb})
    opts.setAtomPalette({-1: rgb})
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    import io
    import matplotlib.image as mpimg
    ax.imshow(mpimg.imread(io.BytesIO(drawer.GetDrawingText()), format="png"))
    ax.axis("off")


def box(ax, x, y, w, h, text, color=INK, fill=PAPER, fontsize=7.4, weight="normal", lw=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.010,rounding_size=0.02",
                                linewidth=lw, edgecolor=color, facecolor=fill, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            color=color, weight=weight, zorder=3, linespacing=1.45)


def arrow(ax, x1, y1, x2, y2, color=MUTED, lw=1.2, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=11,
                                 linewidth=lw, color=color, zorder=1,
                                 shrinkA=1.5, shrinkB=1.5))


def panel_label(ax, letter, title):
    ax.text(0.0, 1.045, letter, fontsize=11, weight="bold", color=INK, transform=ax.transAxes)
    ax.text(0.037, 1.045, title, fontsize=8.6, color=INK, transform=ax.transAxes, weight="bold")


def blank(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


# --------------------------------------------------------------------------- panel a
def panel_a(ax):
    blank(ax)
    panel_label(ax, "a", "Known chemistry is an inductive bias, not an inventory")

    box(ax, 0.02, 0.55, 0.42, 0.30,
        "design context $c=(c_a,c_d,c_i)$\n"
        "per-role counts, junction budget,\ncycle rank, core attachments",
        color=CORE, fontsize=6.8)
    for i, (label, col) in enumerate([("amine", AMINE), ("aldehyde", ALDEHYDE),
                                      ("isocyanide", ISOCYANIDE)]):
        box(ax, 0.50 + i * 0.165, 0.63, 0.15, 0.16, label, color=col, fill="white", fontsize=7.2,
            weight="bold")
    arrow(ax, 0.44, 0.71, 0.495, 0.71)
    ax.text(0.745, 0.845, "chemically meaningful roles", ha="center", fontsize=7.2, color=MUTED)

    box(ax, 0.02, 0.06, 0.96, 0.40,
        "NO COMPONENT IDENTITIES      NO FRAGMENT CATALOGUE      NO TEMPLATE AT SAMPLING\n\n"
        "The context fixes how many atoms each role region has, and where its block begins.\n"
        "It never says which amine, which aldehyde, or which isocyanide.",
        color="#8c2f39", fill="#fdf4f4", fontsize=7.1, lw=1.3)


# --------------------------------------------------------------------------- panel b
def panel_b(ax):
    blank(ax)
    panel_label(ax, "b", "Chemistry enters the learned flow, not only its output")

    box(ax, 0.02, 0.55, 0.20, 0.30, "corrupted state $z_t$\none unified\nmolecular state",
        color=CORE)
    box(ax, 0.30, 0.50, 0.34, 0.40, "shared contextual\ndenoiser\n$p_\\theta(x\\mid z_t,t,c)$",
        color=INK, fill="white", fontsize=8.0, weight="bold", lw=1.4)
    box(ax, 0.72, 0.55, 0.26, 0.30, "one complete\ngenerated lipid\n(not three reactants)",
        color=CORE)
    arrow(ax, 0.22, 0.70, 0.295, 0.70)
    arrow(ax, 0.64, 0.70, 0.715, 0.70)

    entering = ["role identity $r_c$", "role-relative coordinates",
                "role-specific source $p_0^{h,r}$", "role-balanced objective $w_r$"]
    # One bracket, one arrow into the denoiser. A fan of lines reads as four separate mechanisms
    # bolted on the side, which is the opposite of the panel's point.
    for i, item in enumerate(entering):
        ax.text(0.335, 0.335 - i * 0.075, "\u2022 " + item, fontsize=7.0, color=INK, va="center")
    ax.plot([0.315, 0.315], [0.055, 0.375], color="#9a9a9a", lw=1.0, zorder=1)
    arrow(ax, 0.315, 0.375, 0.40, 0.495, color="#7a7a7a", lw=1.1)
    ax.text(0.02, 0.24, "chemistry enters\nHERE, inside the\nlearned process",
            fontsize=7.0, color="#8c2f39", weight="bold", va="center", linespacing=1.5)


# --------------------------------------------------------------------------- panel c
def panel_c(fig, gs, product, components):
    ax = fig.add_subplot(gs)
    blank(ax)
    panel_label(ax, "c", "Exact component projection, verified rather than predicted")
    ax.text(0.5, 0.885, "$D_{\\mathcal{R}}(G,c)$ partitions the terminal state in one linear pass; "
            "admission requires $\\mathcal{R}(D_{\\mathcal{R}}(G,c))\\cong G$",
            ha="center", fontsize=7.1, color=MUTED)
    return ax


# --------------------------------------------------------------------------- panel d
def panel_d(ax, dossier):
    blank(ax)
    panel_label(ax, "d", "A route back to purchasable starting materials, earned not assumed")

    rows = [(e["role"], e["action"], e.get("route"),
             [m["canonical_smiles"] for m in e.get("starting_materials", [])]) for e in dossier]
    order = ["amine_head", "oxoester_aldehyde_body_tail", "isocyanide_tail"]
    rows.sort(key=lambda r: order.index(r[0]))

    for i, (role, action, route, mats) in enumerate(rows):
        y = 0.635 - i * 0.225
        col = ROLE_COLOR[role]
        box(ax, 0.02, y, 0.17, 0.175, ROLE_LABEL[role], color=col, fill="white", fontsize=7.2,
            weight="bold")
        if action == "purchase":
            arrow(ax, 0.19, y + 0.088, 0.58, y + 0.088, color=col)
            ax.text(0.385, y + 0.128, "already in the procurement snapshot", ha="center",
                    fontsize=6.5, color=MUTED)
        else:
            box(ax, 0.215, y, 0.34, 0.175, f"{route}\nforward check reproduces it uniquely",
                color=col, fill=PAPER, fontsize=6.3)
            arrow(ax, 0.19, y + 0.088, 0.212, y + 0.088, color=col)
            arrow(ax, 0.555, y + 0.088, 0.58, y + 0.088, color=col)
        box(ax, 0.58, y, 0.40, 0.175,
            "\n".join(mats) if mats else "catalogue material", color=col, fill="white",
            fontsize=5.9)

    box(ax, 0.02, 0.005, 0.96, 0.115,
        "purchasable starting materials  —  NOT a synthesis-success, yield, purity or scale claim; "
        "procurement is a dated snapshot",
        color="#8c2f39", fill="#fdf4f4", fontsize=6.9, lw=1.2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default="P01")
    parser.add_argument("--output", type=Path,
                        default=REPO / "manuscript/figures/framework/figure1_framework.pdf")
    args = parser.parse_args()

    panel_path = REPO / PANEL
    rows = [json.loads(line) for line in gzip.open(panel_path, "rt")]
    record = next(r for r in rows if r["candidate_id"] == args.candidate)

    fig = plt.figure(figsize=(7.1, 7.15), dpi=300)
    # One full-width row per panel. Panel c gets three rows of its own: a header, the product, and
    # the three projected components, because sharing a row with panel d is what collided them.
    gs = fig.add_gridspec(
        6, 3,
        height_ratios=[1.05, 1.02, 0.12, 0.52, 0.58, 1.00],
        hspace=0.36, wspace=0.12,
        left=0.030, right=0.975, top=0.960, bottom=0.015)

    panel_a(fig.add_subplot(gs[0, :]))
    panel_b(fig.add_subplot(gs[1, :]))

    axc = fig.add_subplot(gs[2, :])
    blank(axc)
    panel_label(axc, "c", "Exact component projection, verified rather than predicted")
    axc.text(0.5, -0.55, "$D_{\\mathcal{R}}(G,c)$ partitions the terminal state in one linear pass; "
             "admission requires $\\mathcal{R}(D_{\\mathcal{R}}(G,c))\\cong G$",
             ha="center", fontsize=7.0, color=MUTED, transform=axc.transAxes)

    axp = fig.add_subplot(gs[3, :])
    draw_molecule(axp, record["canonical_product"], CORE, width=2200, height=380)
    axp.set_title(f"generated lipid $G$   ({args.candidate})", fontsize=7.4, color=INK, pad=1)

    for i, role in enumerate(["amine_head", "oxoester_aldehyde_body_tail", "isocyanide_tail"]):
        axm = fig.add_subplot(gs[4, i])
        draw_molecule(axm, record["components"][role], ROLE_COLOR[role], width=800, height=360)
        axm.set_title(ROLE_LABEL[role], fontsize=7.0, color=ROLE_COLOR[role], pad=1)

    panel_d(fig.add_subplot(gs[5, :]), record["route_dossier"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, format="pdf", bbox_inches="tight")
    png = args.output.with_suffix(".png")
    fig.savefig(png, format="png", bbox_inches="tight")
    plt.close(fig)

    meta = {
        "schema_version": "phase1_forge_figure1_framework.v1",
        "candidate_illustrated": args.candidate,
        "inputs": {"panel": {"path": PANEL, "sha256": sha256_file(panel_path)}},
        "role_colors": {"amine": AMINE, "aldehyde": ALDEHYDE, "isocyanide": ISOCYANIDE},
        "nonclaims": [
            "Panel d ends at purchasable starting materials, not at synthesizable.",
            "No origin variable is depicted; role is deterministic given the design context.",
            "Structures are drawn from the frozen panel artifact, not idealized.",
        ],
    }
    (args.output.parent / "figure1_framework.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {args.output.relative_to(REPO)} and {png.name}")


if __name__ == "__main__":
    main()
