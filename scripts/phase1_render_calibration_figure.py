#!/usr/bin/env python3
"""Show that the evidence tiers are the model's own conformal calibration strata.

Panel a gives the 90% conformal quantile fitted for each stratum, with the size of the
eligible pool behind it.  Panel b makes the consequence concrete: one point prediction,
placed at the ninetieth percentile of the measured library, yields four different
guaranteed lower bounds depending on which stratum the design falls in.

The tiers are not labels assigned after the fact; each corresponds to exactly one
quantile, and the quantile widens as the design moves away from the measured factorial.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

REPO_DEFAULT = Path(__file__).resolve().parents[1]
for _p in (REPO_DEFAULT / "src",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

LEDGER = "results/phase1/ugi_production_full_support_rescoring_v3/terminal_rescoring.csv.gz"
MEASURED = "results/m0_07/agile_oracle_curated.csv.gz"
PANEL = "results/phase1/ugi_prospective_panel_v6/prospective_panel.jsonl.gz"

TIERS = [
    ("qualified_role_holdout", "T1  qualified role holdout", "unseen isocyanide only", "#17395B"),
    ("exploratory_weak_absolute_head_generalization", "T2  head generalization", "unseen amine head", "#2E7D6F"),
    ("exploratory_weak_absolute_aldehyde_generalization", "T3  tail generalization", "unseen aldehyde tail", "#C7873B"),
    ("exploratory_simultaneous_exact_new_tails", "T4  dual tail generalization", "unseen aldehyde and isocyanide", "#A6474F"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    parser.add_argument("--output", type=Path,
                        default=Path("manuscript/figures/panel_v6/calibration_by_tier.png"))
    args = parser.parse_args()
    repo = args.repo.resolve()
    out = args.output if args.output.is_absolute() else repo / args.output
    out.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    with gzip.open(repo / LEDGER, "rt", newline="") as handle:
        scored = [r for r in csv.DictReader(handle) if r["oracle_scored"] == "True"]
    quantile: dict[str, float] = {}
    pool: Counter[str] = Counter()
    for row in scored:
        try:
            q = float(row["conformal_q90"])
        except (TypeError, ValueError):
            continue
        tier = row["authority_tier"]
        quantile[tier] = q
        pool[tier] += 1

    with gzip.open(repo / MEASURED, "rt", newline="") as handle:
        measured = sorted(float(r["expt_Hela"]) for r in csv.DictReader(handle)
                          if r["expt_Hela"] not in ("", "nan"))
    p90 = measured[int(0.90 * len(measured))]
    median = measured[len(measured) // 2]

    panel = [json.loads(line) for line in gzip.open(repo / PANEL, "rt")]
    in_panel = Counter(r["authority_tier"] for r in panel)

    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Avenir Next", "Helvetica Neue", "DejaVu Sans"],
                         "text.color": "#17191D", "axes.labelcolor": "#17191D",
                         "xtick.color": "#555A62", "ytick.color": "#555A62",
                         "axes.edgecolor": "#C9CDD2"})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.3), width_ratios=[1.0, 1.25])

    # --- a: the fitted quantile per stratum
    present = [t for t in TIERS if t[0] in quantile]
    ys = list(range(len(present)))[::-1]
    for y, (key, label, meaning, colour) in zip(ys, present):
        q = quantile[key]
        ax1.barh(y, q, height=0.56, color=colour, alpha=0.9)
        ax1.text(q + 0.09, y + 0.10, f"{q:.2f}", va="center", fontsize=7.6,
                 fontweight="bold", color=colour)
        ax1.text(q + 0.09, y - 0.22,
                 f"{pool[key]:,} scored · {in_panel.get(key, 0)} on panel",
                 va="center", fontsize=5.9, color="#555A62")
    ax1.set_yticks(ys)
    ax1.set_yticklabels([f"{label}\n{meaning}" for _, label, meaning, _ in present], fontsize=6.8)
    ax1.set_xlabel("conformal 90% quantile subtracted from the point prediction", fontsize=7.2)
    ax1.set_xlim(0, max(quantile.values()) * 1.22)
    ax1.set_title("a   Each evidence tier is one calibration stratum",
                  fontsize=8.6, fontweight="bold", loc="left", pad=10)
    ax1.grid(axis="x", color="#EDEFF1", lw=0.8)
    ax1.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax1.spines[spine].set_visible(False)
    ax1.tick_params(axis="y", length=0)

    # --- b: same point prediction, four different guarantees
    ax2.hist(measured, bins=48, color="#E4E7EA", edgecolor="white", linewidth=0.4)
    ax2.axvline(p90, color="#8A8F96", lw=1.1, ls=":")
    ax2.axvline(median, color="#8A8F96", lw=1.1, ls="--")
    top = ax2.get_ylim()[1]
    ax2.text(median, top * 0.99, " measured median", rotation=90, va="top", ha="right",
             fontsize=6.0, color="#555A62")
    ax2.text(p90, top * 0.99, " measured p90", rotation=90, va="top", ha="right",
             fontsize=6.0, color="#555A62")
    for i, (key, label, _, colour) in enumerate(present):
        lower = p90 - quantile[key]
        y = top * (0.78 - 0.115 * i)
        ax2.annotate("", xy=(lower, y), xytext=(p90, y),
                     arrowprops=dict(arrowstyle="-|>", color=colour, lw=2.1,
                                     shrinkA=0, shrinkB=0))
        ax2.plot(p90, y, "o", ms=6, color=colour, zorder=4)
        ax2.text(lower - 0.25, y, f"{lower:.1f}", va="center", ha="right",
                 fontsize=6.8, color=colour, fontweight="bold")
        ax2.text(p90 + 0.35, y, label.split("  ")[0], va="center", fontsize=6.5, color=colour)
    ax2.set_xlabel("HeLa transfection, measured scale", fontsize=7.2)
    ax2.set_ylabel("measured products", fontsize=7.2)
    ax2.set_title("b   One prediction, four guarantees",
                  fontsize=8.6, fontweight="bold", loc="left", pad=10)
    for spine in ("top", "right"):
        ax2.spines[spine].set_visible(False)

    fig.text(0.008, -0.045,
             "a, the 90% conformal quantile fitted for each stratum across the "
             f"{sum(pool.values()):,} designs the model scores. b, a design predicted at the "
             f"measured ninetieth percentile ({p90:.1f}) carries a calibrated lower bound that "
             "depends on its stratum, so identical point predictions are not equally supported. "
             "The head-generalization stratum is calibrated on 42 designs, far fewer than the "
             "others, so its quantile is estimated less precisely than its width suggests.",
             fontsize=6.0, color="#555A62", wrap=True)
    fig.tight_layout()
    fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white")
    print(f"wrote {out.relative_to(repo)}")


if __name__ == "__main__":
    main()
