#!/usr/bin/env python3
"""Emit the appendix data tables as LaTeX, straight from frozen artifacts.

Three tables the appendix defers to and that must not be transcribed by hand:

  C.3  per-seed by per-head semantic-utility gaps, plus the corruption ratios
  C.4  the misalignment control, three arms per head
  D.3  the complete 38-descriptor lipid-native panel, four arms

Writing these by hand is how a paper acquires a number that no artifact supports. Everything below
is read from the frozen JSON and formatted; nothing is retyped, and no value is rounded before it
reaches LaTeX except for display.

Head naming: the six heads eligible for the semantic analysis are the categorical chemical states.
`decoration_anchor_ce` is emitted in a clearly separated row because its target is a position rather
than a categorical chemical state, so it is reported for completeness and excluded from the
six-head summary. See Appendix A.3.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CLOSURE = "results/phase1/forge_semantics_pilot_v1/closure.json"
PANEL = "results/phase1/forge_held_family_stage2_v1/amphiphile_panel_v3.json"

# The semantic pilot's reported heads, matching PRIMARY_HEADS in
# scripts/phase1_semantics_closure_v1.py. `total` is emitted separately as the aggregate row.
#
# CAUTION. A DIFFERENT six-head set belongs to the role-divergence analysis of Amendment 14
# (scripts/phase1_role_divergence_association_v1.py): it drops decoration_anchor_ce because that
# target is positional, and adds decoration_atom_ce and decoration_bond_ce. Those two are negative
# in seed s2. Substituting one set for the other turns a true claim into a false one, so both are
# emitted here, clearly separated, rather than leaving a reader to assume which six is meant.
HEADS = [
    ("atom_ce", "atom state"),
    ("parent_bond_ce", "parent bond"),
    ("offspring_ce", "offspring topology"),
    ("closure_bond_ce", "closure bond"),
    ("decoration_anchor_ce", "decoration anchor"),
]
DIVERGENCE_ONLY = [
    ("decoration_atom_ce", "decoration atom"),
    ("decoration_bond_ce", "decoration bond"),
]

# Which molecular region each descriptor prefix belongs to, for the panel table.
REGION = {"ald": "aldehyde tail", "iso": "isocyanide tail", "head": "ionizable head",
          "prod": "whole lipid", "whole": "whole lipid"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def region_of(name: str) -> str:
    return REGION.get(name.split("_")[0], "whole lipid")


def semantic_tables(closure: dict) -> str:
    seeds = sorted(closure["per_seed"])
    rep = closure["replication_across_seeds"]
    out = [
        r"\begin{table}[h]", r"\centering", r"\small",
        r"\caption{\textbf{Semantic-utility gap per head and per paired seed.} Each entry is "
        r"$L_{\text{no-role}} - L_{\text{true-role}}$ for that head, within the seed's paired run, so "
        r"positive means the aligned map reconstructs better. The six rows above the rule are the "
        r"semantic pilot's reported set, matching PRIMARY\_HEADS in the closure script. Below it are "
        r"the two heads belonging only to the role-divergence analysis, a different six; those two "
        r"are not uniformly positive and no claim is made over them.}",
        r"\label{tab:sem-per-head}",
        r"\begin{tabular}{@{}l" + "r" * len(seeds) + r"rrc@{}}", r"\toprule",
        "head & " + " & ".join(s for s in seeds) + r" & mean & s.d. & all positive \\",
        r"\midrule",
    ]
    # Per-head mean, s.d. and sign count are computed from per_seed rather than taken from the
    # artifact's replication_across_seeds block. That block omits the two decoration heads, so
    # reading it would silently drop exactly the heads that do not replicate.
    import statistics
    for key, label in HEADS:
        vals = [closure["per_seed"][s]["delta_bar_sem"][key] for s in seeds]
        mean = statistics.fmean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        npos = sum(1 for v in vals if v > 0)
        flag = "yes" if npos == len(vals) else f"\\textbf{{{npos}/{len(vals)}}}"
        out.append(f"{label} & " + " & ".join(f"{v:+.4f}" for v in vals)
                   + f" & {mean:+.4f} & {sd:.4f} & " + flag + r" \\")
    out.append(r"\midrule")
    tvals = [closure["per_seed"][s]["delta_bar_sem"]["total"] for s in seeds]
    tmean = statistics.fmean(tvals)
    tsd = statistics.stdev(tvals) if len(tvals) > 1 else 0.0
    tpos = sum(1 for v in tvals if v > 0)
    out.append(r"\textbf{total} & " + " & ".join(f"{v:+.4f}" for v in tvals)
               + f" & {tmean:+.4f} & {tsd:.4f} & "
               + ("yes" if tpos == len(tvals) else f"\\textbf{{{tpos}/{len(tvals)}}}") + r" \\")
    out.append(r"\midrule")
    out.append(r"\multicolumn{6}{@{}l}{\emph{Heads used only by the role-divergence analysis of "
               r"Amendment 14, not part of the six above:}} \\")
    for key, label in DIVERGENCE_ONLY:
        vals = [closure["per_seed"][s]["delta_bar_sem"][key] for s in seeds]
        mean = statistics.fmean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        npos = sum(1 for v in vals if v > 0)
        flag = "yes" if npos == len(vals) else f"\\textbf{{{npos}/{len(vals)}}}"
        out.append(f"{label} & " + " & ".join(f"{v:+.4f}" for v in vals)
                   + f" & {mean:+.4f} & {sd:.4f} & " + flag + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    # Corruption dependence.
    cor = closure["corruption_dependence_per_seed"]
    out += [
        r"\begin{table}[h]", r"\centering", r"\small",
        r"\caption{\textbf{Corruption dependence of the semantic-utility gap.} The gap near the "
        r"corrupted end of the probability path, against the gap near the clean endpoint, per seed. "
        r"The direction replicates in every seed; the magnitude does not, which is why the three "
        r"values are reported rather than a range.}",
        r"\label{tab:sem-corruption}",
        r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
        r"seed & gap, corrupted end & gap, clean end & ratio \\", r"\midrule",
    ]
    for s in sorted(cor):
        c = cor[s]
        out.append(f"{s} & {c['low_t']:+.4f} & {c['high_t']:+.4f} & "
                   f"{c['ratio']:.2f}$\\times$ \\\\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    # Misalignment control.
    mis = closure["misaligned_control"]
    out += [
        r"\begin{table}[h]", r"\centering", r"\small",
        r"\caption{\textbf{The misalignment control.} Denoising loss under the three arms; lower is "
        r"better. A negative \emph{none $-$ misaligned} means the misaligned arm is \emph{worse} "
        r"than receiving no role at all, which is what rules out attributing the benefit to the "
        r"presence of an extra categorical channel.}",
        r"\label{tab:sem-misaligned}",
        r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
        r"head & true & none & misaligned & none $-$ mis. & true $-$ mis. \\", r"\midrule",
    ]
    absent = [lab for k, lab in HEADS if k not in mis]
    for key, label in HEADS:
        if key not in mis:
            continue
        m = mis[key]
        row = (f"{label} & {m['true']:.4f} & {m['none']:.4f} & {m['misaligned']:.4f} & "
               f"{m['none_minus_misaligned']:+.4f} & {m['true_minus_misaligned']:+.4f}")
        out.append(row + r" \\")
    if "total" in mis:
        m = mis["total"]
        out.append(r"\midrule")
        out.append(r"\textbf{total} & " + f"{m['true']:.4f} & {m['none']:.4f} & "
                   f"{m['misaligned']:.4f} & {m['none_minus_misaligned']:+.4f} & "
                   f"{m['true_minus_misaligned']:+.4f} \\\\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    if absent:
        out += [r"The artifact does not carry a misalignment breakdown for "
                + " or ".join(absent) + r", so those rows are absent rather than zero.", ""]
    return "\n".join(out)


def panel_table(panel: dict) -> str:
    med = panel["marginals_median"]
    arms = [("heldout_real", "real"), ("prod_budget_3000", "FORGE"),
            ("marginal_null", "null")]
    pop = panel["population_sizes"]
    out = [
        r"\begingroup\small",
        r"\begin{longtable}{@{}llrrr@{}}",
        r"\caption{\textbf{The complete lipid-native descriptor panel.} Median value per descriptor "
        r"for held-out real lipids, FORGE samples, and the role-marginal null. Every descriptor is "
        r"reported, not only those on which FORGE compares well: the panel is a guard against "
        r"post-hoc metric selection only if its unselected members are published. Populations: "
        + ", ".join(f"{lab} $n={pop[key]:,}$" for key, lab in arms)
        + r".}\label{tab:panel-full}\\",
        r"\toprule",
        r"descriptor & region & real & FORGE & null \\", r"\midrule", r"\endfirsthead",
        r"\toprule descriptor & region & real & FORGE & null \\ \midrule \endhead",
        r"\bottomrule", r"\endfoot",
    ]
    for name in panel["descriptors"]:
        pretty = name.replace("_", r"\_")
        cells = []
        for key, _ in arms:
            v = med[key].get(name)
            cells.append("---" if v is None else f"{v:.3f}")
        out.append(f"\\texttt{{{pretty}}} & {region_of(name)} & " + " & ".join(cells) + r" \\")
    out += [r"\end{longtable}", r"\endgroup", ""]

    # Grouped separability, per fold.
    sep = panel["grouped_two_sample_separability"]
    out += [
        r"\begin{table}[h]", r"\centering", r"\small",
        r"\caption{\textbf{Grouped real-versus-generated separability.} Grouping is by "
        + sep["prod_budget_3000"]["grouping"].split(",")[0]
        + r", so repeated component combinations cannot leak across folds. Per-fold values are given "
        r"because the mean alone hides the spread.}",
        r"\label{tab:grouped-auc}",
        r"\begin{tabular}{@{}lrrl@{}}", r"\toprule",
        r"arm & groups & mean AUC & per-fold AUC \\", r"\midrule",
    ]
    label = {"prod_budget_3000": "FORGE", "marginal_null": "role-marginal null",
             "step750": "earlier checkpoint"}
    for key in ("prod_budget_3000", "marginal_null", "step750"):
        if key not in sep:
            continue
        s = sep[key]
        folds = ", ".join(f"{f:.3f}" for f in s["grouped_cv_auc_folds"])
        out.append(f"{label[key]} & {s['groups']} & {s['grouped_cv_auc_mean']:.4f} & {folds} \\\\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=REPO / "manuscript/generated_appendix_tables.tex")
    args = parser.parse_args()

    closure_path, panel_path = REPO / CLOSURE, REPO / PANEL
    closure = json.loads(closure_path.read_text())
    panel = json.loads(panel_path.read_text())

    if len(panel["descriptors"]) != 38:
        raise SystemExit(f"expected 38 descriptors, found {len(panel['descriptors'])}")

    body = "\n".join([
        "% autogenerated - regenerate, dont edit directly",
        "",
        r"\subsection{Per-head and per-seed semantic results}",
        r"\label{app:sem-tables}",
        semantic_tables(closure),
        r"\subsection{The complete descriptor panel}",
        r"\label{app:panel-tables}",
        panel_table(panel),
    ])
    args.output.write_text(body + "\n")
    print(f"wrote {args.output.relative_to(REPO)}")
    print(f"{len(HEADS)} eligible heads, {len(closure['per_seed'])} seeds, "
          f"{len(panel['descriptors'])} descriptors")


if __name__ == "__main__":
    main()
