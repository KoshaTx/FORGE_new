#!/usr/bin/env python3
"""Map predicted transfection over the axes this chemistry actually varies along.

A projection was tried first and does not work here: measured, enumerated and generated
lipids give a silhouette of -0.03 to -0.02 across binary Morgan under Jaccard, count
Morgan under cosine and physicochemical descriptors under Euclidean, meaning no
separation in any representation. Variation in a single-reaction combinatorial library is
continuous rather than clustered, so an embedding either invents clusters or shows a
formless cloud.

This figure plots the design axes directly instead: amine head against aldehyde tail
carbon count, with cells coloured by mean predicted transfection and the locked panel
drawn on top. Distances are real, the axes are interpretable, and a reader cannot
misread neighbourhood structure that does not exist.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_DEFAULT = Path(__file__).resolve().parents[1]
for _p in (REPO_DEFAULT / "src",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

LEDGER = "results/phase1/ugi_production_full_support_rescoring_v3/terminal_rescoring.csv.gz"
MEASURED = "results/m0_07/agile_oracle_curated.csv.gz"
PANEL = "results/phase1/ugi_prospective_panel_v6/prospective_panel.jsonl.gz"
MIN_CELL = 3


def carbon_count(smiles: str, cache: dict[str, int]) -> int:
    from rdkit import Chem, rdBase

    if smiles not in cache:
        with rdBase.BlockLogs():
            molecule = Chem.MolFromSmiles(smiles)
            cache[smiles] = (
                sum(1 for a in molecule.GetAtoms() if a.GetSymbol() == "C")
                if molecule is not None else -1)
    return cache[smiles]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    parser.add_argument("--output", type=Path,
                        default=Path("manuscript/figures/panel_v6/design_axes_map.png"))
    args = parser.parse_args()
    repo = args.repo.resolve()
    out = args.output if args.output.is_absolute() else repo / args.output
    out.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cache: dict[str, int] = {}
    with gzip.open(repo / LEDGER, "rt", newline="") as handle:
        rows = [r for r in csv.DictReader(handle)
                if r["oracle_scored"] == "True" and r["terminal_chemical_admitted"] == "True"]
    seen: dict[str, dict[str, str]] = {}
    for row in rows:
        seen.setdefault(row["canonical_product"], row)
    seen.pop("", None)

    cells: dict[tuple[str, int], list[float]] = defaultdict(list)
    head_total: dict[str, list[float]] = defaultdict(list)
    for row in seen.values():
        try:
            value = float(row["oracle_mean"])
        except (TypeError, ValueError):
            continue
        head = row["canonical_amine"]
        carbons = carbon_count(row["canonical_aldehyde"], cache)
        if carbons < 0:
            continue
        cells[(head, carbons)].append(value)
        head_total[head].append(value)

    panel = [json.loads(line) for line in gzip.open(repo / PANEL, "rt")]
    with gzip.open(repo / MEASURED, "rt", newline="") as handle:
        measured = sorted(float(r["expt_Hela"]) for r in csv.DictReader(handle)
                          if r["expt_Hela"] not in ("", "nan"))
    p90 = measured[int(0.90 * len(measured))]

    # Order heads by how well they do overall, so the figure reads top to bottom.
    heads = sorted(head_total, key=lambda h: -float(np.mean(head_total[h])))
    heads = [h for h in heads if len(head_total[h]) >= 20][:18]
    carbon_values = sorted({c for (_, c) in cells if c >= 0})
    lo, hi = min(carbon_values), max(carbon_values)
    columns = list(range(lo, hi + 1))

    grid = np.full((len(heads), len(columns)), np.nan)
    counts = np.zeros_like(grid)
    for i, head in enumerate(heads):
        for j, carbons in enumerate(columns):
            values = cells.get((head, carbons), [])
            if len(values) >= MIN_CELL:
                grid[i, j] = float(np.mean(values))
                counts[i, j] = len(values)

    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Avenir Next", "Helvetica Neue", "DejaVu Sans"],
                         "text.color": "#17191D", "axes.labelcolor": "#17191D",
                         "xtick.color": "#555A62", "ytick.color": "#555A62"})
    fig, ax = plt.subplots(figsize=(13.2, 7.6))
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#F4F5F6")
    image = ax.imshow(grid, aspect="auto", cmap=cmap, origin="upper",
                      extent=(lo - 0.5, hi + 0.5, len(heads) - 0.5, -0.5))
    bar = fig.colorbar(image, ax=ax, fraction=0.026, pad=0.012)
    bar.set_label("mean predicted transfection, measured scale", fontsize=10)
    bar.ax.axhline(p90, color="#A6474F", lw=1.6)
    bar.ax.text(1.7, p90, f" measured p90 {p90:.1f}", va="center", fontsize=8.2,
                color="#A6474F", transform=bar.ax.get_yaxis_transform())

    index = {head: i for i, head in enumerate(heads)}
    plotted = 0
    for record in panel:
        head = record["components"]["amine_head"]
        if head not in index:
            continue
        carbons = carbon_count(record["components"]["oxoester_aldehyde_body_tail"], cache)
        high = record["arm"] == "HIGH"
        ax.scatter(carbons, index[head], marker="o" if high else "s",
                   s=96 if high else 62,
                   facecolors="#17191D" if high else "none",
                   edgecolors="#17191D" if high else "#A6474F",
                   linewidths=1.7, zorder=5)
        plotted += 1

    ax.set_yticks(range(len(heads)))
    ax.set_yticklabels(heads, fontsize=8.6, family="monospace")
    ax.set_xticks(columns)
    ax.set_xlabel("carbon atoms in the ester-linked aldehyde tail", fontsize=10.5)
    ax.set_ylabel("amine head group", fontsize=10.5)
    ax.set_title("Predicted transfection over the two axes this chemistry varies along",
                 fontsize=14, fontweight="bold", loc="left", pad=12)
    ax.set_xlim(lo - 0.5, hi + 0.5)
    for spine in ax.spines.values():
        spine.set_color("#C9CDD2")

    fig.text(0.01, -0.085,
             f"Heads are ordered by mean predicted transfection across all their designs and "
             f"limited to the {len(heads)} with at least 20 scored designs. Cells require at "
             f"least {MIN_CELL} designs; grey cells were not generated. Filled circles are the "
             f"{sum(1 for r in panel if r['arm'] == 'HIGH')} high-arm candidates and open "
             f"squares the controls; {plotted} of {len(panel)} fall on heads shown here. A "
             "projection was tried first and abandoned: measured, enumerated and generated "
             "lipids give a silhouette of -0.03 to -0.02 across binary Morgan, count Morgan and "
             "physicochemical descriptors, so no embedding separates them.",
             fontsize=8.4, color="#555A62", wrap=True)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.13)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"wrote {out.relative_to(repo)}  ({len(heads)} heads x {len(columns)} tail lengths, "
          f"{plotted}/{len(panel)} panel candidates placed)")


if __name__ == "__main__":
    main()
