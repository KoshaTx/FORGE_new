#!/usr/bin/env python3
"""Render the ranking and attribute structure of the forty locked candidates.

Three panels sharing a candidate axis:

  a  calibrated lower bound and point prediction per candidate, ordered as selected,
     with the measured library's median and 90th percentile drawn for reference
  b  a heatmap of the attributes that vary across the panel, each column scaled to its
     own range so that structure is visible rather than dominated by units
  c  categorical strips for evidence tier and novelty

Everything is read from the frozen selection artifact; nothing is recomputed here.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path

REPO_DEFAULT = Path(__file__).resolve().parents[1]
for _p in (REPO_DEFAULT / "src",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

PANEL = "results/phase1/ugi_prospective_panel_v6/prospective_panel.jsonl.gz"
MEASURED = "results/m0_07/agile_oracle_curated.csv.gz"

TIER_ORDER = [
    ("qualified_role_holdout", "T1 qualified\nrole holdout"),
    ("exploratory_weak_absolute_head_generalization", "T2 head\ngeneralization"),
    ("exploratory_weak_absolute_aldehyde_generalization", "T3 tail\ngeneralization"),
    ("exploratory_simultaneous_exact_new_tails", "T4 dual tail\ngeneralization"),
]
TIER_COLOR = {
    "qualified_role_holdout": "#17395B",
    "exploratory_weak_absolute_head_generalization": "#2E7D6F",
    "exploratory_weak_absolute_aldehyde_generalization": "#C7873B",
    "exploratory_simultaneous_exact_new_tails": "#A6474F",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    parser.add_argument("--output", type=Path,
                        default=Path("manuscript/figures/panel_v6/panel_40_ranking_heatmap.png"))
    args = parser.parse_args()
    repo = args.repo.resolve()
    out = args.output if args.output.is_absolute() else repo / args.output
    out.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Patch

    records = [json.loads(line) for line in gzip.open(repo / PANEL, "rt")]
    records.sort(key=lambda r: (r["arm"] != "HIGH", r["selection_order_within_arm"]))

    with gzip.open(repo / MEASURED, "rt", newline="") as handle:
        measured = sorted(
            float(row["expt_Hela"]) for row in csv.DictReader(handle)
            if row["expt_Hela"] not in ("", "nan"))
    median = measured[len(measured) // 2]
    p90 = measured[int(0.90 * len(measured))]

    ids = [r["candidate_id"] for r in records]
    lcb = [r["scores"]["lcb90"] for r in records]
    mean = [r["scores"]["oracle_mean"] for r in records]
    pct = [r["scores"]["percentile_of_agile_measured_library_beaten"] for r in records]
    steps = [r["synthetic_steps"] for r in records]
    dbonds = [r["structure"]["carbon_double_bonds"] for r in records]
    n = len(records)

    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Avenir Next", "Helvetica Neue", "DejaVu Sans"],
                         "axes.edgecolor": "#555A62", "text.color": "#17191D",
                         "axes.labelcolor": "#17191D", "xtick.color": "#555A62",
                         "ytick.color": "#555A62"})
    fig, axes = plt.subplots(
        3, 1, figsize=(6.5, 4.7), height_ratios=[3.1, 2.5, 0.85], sharex=True,
        gridspec_kw={"hspace": 0.13})
    x = list(range(n))

    # --- a: predictions
    ax = axes[0]
    ax.axhline(median, color="#8A8F96", lw=1.0, ls="--", zorder=1)
    ax.axhline(p90, color="#8A8F96", lw=1.0, ls=":", zorder=1)
    ax.text(n - 0.4, median, f"  measured median {median:.1f}", va="center",
            fontsize=5.3, color="#555A62")
    ax.text(n - 0.4, p90, f"  measured p90 {p90:.1f}", va="center",
            fontsize=5.3, color="#555A62")
    for i, r in enumerate(records):
        colour = TIER_COLOR.get(r["authority_tier"], "#555A62")
        ax.vlines(i, lcb[i], mean[i], color=colour, lw=2.6, alpha=0.55, zorder=2)
        ax.plot(i, mean[i], "o", ms=5.4, color=colour, zorder=3)
        ax.plot(i, lcb[i], "_", ms=8, mew=2.0, color=colour, zorder=3)
    ax.axvline(33.5, color="#17191D", lw=1.1, ls="-", alpha=0.55)
    low, high = ax.get_ylim()
    # Sit the divider labels low in the axes so they clear the legend.
    ax.text(33.2, low + 0.06 * (high - low), "high arm", ha="right", va="bottom",
            fontsize=5.9, color="#17191D", fontweight="bold")
    ax.text(33.8, low + 0.06 * (high - low), "controls", ha="left", va="bottom",
            fontsize=5.9, color="#17191D", fontweight="bold")
    ax.set_ylabel("predicted transfection\n(measured scale)", fontsize=6.2)
    ax.set_title("Ranking and attribute structure of the forty locked candidates",
                 fontsize=8.1, fontweight="bold", loc="left", pad=12)
    ax.legend(handles=[Patch(facecolor=TIER_COLOR[k], label=v.replace("\n", " "))
                       for k, v in TIER_ORDER],
              loc="upper right", frameon=False, fontsize=5.3, ncol=2)
    ax.grid(axis="y", color="#E3E5E8", lw=0.7)
    ax.set_axisbelow(True)

    # --- b: attribute heatmap, each row scaled to its own range
    rows = [("calibrated lower bound", lcb), ("point prediction", mean),
            ("percentile of measured\nlibrary exceeded", pct),
            ("synthetic steps", steps), ("carbon-carbon\ndouble bonds", dbonds)]
    cmap = LinearSegmentedColormap.from_list("forge", ["#F4F6F8", "#9DBBD0", "#17395B"])
    matrix = []
    for _, values in rows:
        low, high = min(values), max(values)
        span = (high - low) or 1.0
        matrix.append([(v - low) / span for v in values])
    ax = axes[1]
    ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1,
              extent=(-0.5, n - 0.5, len(rows) - 0.5, -0.5))
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([label for label, _ in rows], fontsize=5.6)
    for i in range(n):
        for j, (_, values) in enumerate(rows):
            shade = "#FFFFFF" if matrix[j][i] > 0.55 else "#17191D"
            text = f"{values[i]:.0f}" if max(values) > 20 or values is steps or values is dbonds \
                else f"{values[i]:.1f}"
            ax.text(i, j, text, ha="center", va="center", fontsize=3.5, color=shade)
    ax.axvline(33.5, color="#17191D", lw=1.1)
    ax.set_xticks(x)

    # --- c: tier and novelty strips
    ax = axes[2]
    for i, r in enumerate(records):
        ax.add_patch(plt.Rectangle((i - 0.5, 0.52), 1, 0.44,
                                   color=TIER_COLOR.get(r["authority_tier"], "#555A62")))
        novel = r["novelty"]["absent_from_agile_library"]
        ax.add_patch(plt.Rectangle((i - 0.5, 0.04), 1, 0.44,
                                   color="#2E7D6F" if novel else "#D9DDE1"))
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.74, 0.26])
    ax.set_yticklabels(["evidence tier", "absent from\nprior library"], fontsize=5.6)
    ax.set_xticks(x)
    ax.set_xticklabels(ids, rotation=90, fontsize=4.5)
    ax.axvline(33.5, color="#17191D", lw=1.1)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    fig.text(0.008, 0.012,
             "Bars span the calibrated 90% lower bound to the point prediction. "
             "Heatmap cells are scaled within each row. Controls are drawn from the tail-"
             "generalization tier so that evidence class cannot explain a high-versus-low "
             "difference.", fontsize=5.1, color="#555A62")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"wrote {out.relative_to(repo)}")


if __name__ == "__main__":
    main()
