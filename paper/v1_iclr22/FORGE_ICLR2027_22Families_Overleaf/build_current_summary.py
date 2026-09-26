"""Extract hash-pinned development results and reproducible PGFPlots figure sources."""

import hashlib
import json
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESEARCH = ROOT / "results/phase1/compose_lipid_iclr22_research_v1"


def pin(path):
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main():
    sources = {
        "assembly_statistics": ROOT
        / "results/phase1/compose_lipid_iclr22_table_completion_v1/assembly/result.json",
        "terminal_reconstruction": ROOT
        / "results/phase1/compose_lipid_iclr22_table_completion_v1/assembly/readout_recovery_v3/terminal_v1/result.json",
        "quality_statistics": ROOT
        / "results/phase1/compose_lipid_iclr22_table_completion_v1/quality/result.json",
        "readout_quality": ROOT
        / "results/phase1/compose_lipid_iclr22_table_completion_v1/readout_quality/result.json",
        "paired_readouts": ROOT
        / "results/phase1/compose_lipid_iclr22_table_completion_v1/assembly/paired_readouts_v1/result.json",
        "routing": RESEARCH / "parallel_completion_v1/routes/recount_closeout_v3/result.json",
        "quality": RESEARCH / "evidence_completion_v1/result.json",
        "realism": RESEARCH / "qualified_CAL_sensitivity/result.json",
        "group_comparison": RESEARCH
        / "parallel_completion_v1/training_evaluation/group_loss_evaluation_v1/comparison_v1.json",
        "update_comparison": RESEARCH
        / "quality/training_comparison_review/supplement_review_v1/outcomes.json",
    }
    data = {key: json.loads(path.read_text()) for key, path in sources.items()}
    assembly = data["assembly_statistics"]
    assert assembly["complete"] and len(assembly["requests"]) == 1408
    assert assembly["totals"]["requests"] == 1408
    assert assembly["totals"]["exact"] == 1387
    terminal = data["terminal_reconstruction"]
    assert terminal["complete"] and terminal["requests"] == 1408
    assert terminal["trajectories"] == 7040 and terminal["model_flow_calls"] == 0
    terminal = terminal["summaries"]["terminal_argmax_from_saved"]
    statistics = data["quality_statistics"]
    assert statistics["complete"] and statistics["quality"]["totals"]["requests"] == 1408
    readout_quality = data["readout_quality"]
    assert readout_quality["complete"]
    assert readout_quality["selected_frozen_estimator_reproduction_exact"]
    assert len(readout_quality["families"]) == 22
    paired_readouts = data["paired_readouts"]
    assert paired_readouts["complete"]
    assert paired_readouts["full_conditioned_terminal_recovery_equal"]
    readouts = {}
    for row in paired_readouts["counts"]:
        readouts.setdefault(row["arm"], {}).setdefault(row["population"], {})[row["family"]] = {
            k: row[k] for k in ("trajectories", "exact_L1", "valid_connected")
        }
    assert readouts["conditioned_terminal_argmax"] == terminal
    for arm in readouts.values():
        assert arm["draw0"]["ALL"]["trajectories"] == 1408
        assert arm["all5"]["ALL"]["trajectories"] == 7040
    route, realism, quality = data["routing"], data["realism"], data["quality"]["quality"]
    assert route["passed"] and route["all_quality_molecules_components_flags_unchanged"]
    assert route["routing_totals"] == {
        "L2_ready_products": 1189,
        "L3_direct_only_products": 119,
        "combined_primary_products": 616,
        "exact_L1": 1387,
        "requests": 1408,
        "required_branches": 3723,
        "strict_secondary_products": 27,
    }
    labels = [
        "A3",
        "Acid--epoxide diester",
        "AEMA",
        "Aldehyde Ugi 3-CR",
        "Aldehyde Ugi 4-CR",
        r"$\alpha$-Isocyanoester",
        "Amine alkylation",
        "Amine--epoxide",
        "Aryl reductive amination",
        "Aza-Michael acrylamide",
        "Aza-Michael acrylate",
        "Disulfide Michael",
        "Epoxide O-acylation",
        "iPhos",
        "Ketone--isocyanide",
        "Ketone Ugi 4-CR",
        "Maleate",
        "O-esterification",
        "Passerini",
        "Preassembled thiol-yne",
        "Reductive amination",
        "Thiolactone",
    ]
    families = sorted(route["by_family"])
    assert len(families) == len(labels) == 22
    cal = {row["family"]: row for row in realism["rows"]}
    assert set(cal) == set(families)
    rows = []
    for family, label in zip(families, labels, strict=True):
        r, q = route["by_family"][family], quality["by_family"][family]
        assert r["requests"] == q["requests"] == 64
        assert r["exact_L1"] == q["exact"]
        assert r["limited_design_pass"] == q["design_pass"]
        rows.append(
            {
                "family": family,
                "label": label,
                "route": r,
                "quality": q,
                "realism": cal[family],
                "decomposition": assembly["by_family"][family],
                "terminal_argmax": {name: values[family] for name, values in terminal.items()},
                "structural_statistics": statistics["quality"]["by_family"][family],
                "paired_readouts": {
                    arm: {population: counts[family] for population, counts in populations.items()}
                    for arm, populations in readouts.items()
                },
            }
        )
    fp_keys = [
        k
        for k, v in cal[families[0]]["arms"]["context_preserving"]["fingerprints"].items()
        if isinstance(v, (int, float))
        and k
        not in {
            "unique_generated",
            "unique_reference",
            "generated_neighbors",
            "reference_neighbors",
        }
    ]
    macro = {
        arm: {
            key: mean(cal[f]["arms"][arm]["fingerprints"][key] for f in families) for key in fp_keys
        }
        for arm in ("saved", "context_preserving", "TRAIN_control")
    }
    paired = data["group_comparison"]["paired_generation"]

    def group(metric, arm):
        comparison = paired["original_to_" + ("categorical" if arm == "original" else arm)]
        row = next(
            r
            for r in comparison
            if r["family"] == "ALL" and r["readout"] == "d1" and r["metric"] == metric
        )
        assert row["requests"] == 352
        return row["before" if arm == "original" else "after"]

    diagnostics = {
        "group66": [
            {
                "arm": arm,
                "exact": group("exact_L1", arm),
                "design": group("limited_design", arm),
                "requests": 352,
            }
            for arm in ("original", "categorical", "coarse")
        ],
        "updates": [],
    }
    for arm in ("starting_checkpoint", "large", "small"):
        row = next(
            r
            for r in data["update_comparison"]["summaries"][arm]
            if r["family"] == "ALL" and r["readout"] == "d1"
        )
        diagnostics["updates"].append(
            {
                "arm": arm,
                "exact": row["source_exact"],
                "design": row["qualified_design_pass"],
                "requests": row["requests"],
            }
        )
    summary = {
        "schema": "forge.iclr22.current_summary.v1",
        "evidence_class": "Computed",
        "scope": "development_single_original_checkpoint_TRAIN_derived_requests",
        "final_heldout_evidence_admitted": False,
        "inputs": {k: pin(p) for k, p in sources.items()},
        "producer": pin(Path(__file__)),
        "totals": route["totals"],
        "routing_totals": route["routing_totals"],
        "family_rows": rows,
        "realism_macro": macro,
        "realism_aggregation": "Unweighted mean of 22 within-family statistics, not pooled molecular diversity; synthetic qualified CAL reference, not independent empirical lipids.",
        "diagnostic_training": diagnostics,
        "selected_cohort_decomposition": assembly["totals"],
        "same_cohort_terminal_argmax": terminal,
        "selected_cohort_structural_statistics": statistics["quality"]["totals"],
        "same_cohort_draw0_quality": readout_quality["summaries"],
        "same_cohort_readouts": readouts,
        "same_cohort_paired_readout_contrasts": paired_readouts["paired"],
        "same_cohort_first_draw_constrained_readout": assembly["first_draw_constrained_readout"],
        "manuscript_main_counts": {
            "requests": 1408,
            "exact_L1": 1387,
            "limited_design": 1324,
            "makeable": 616,
            "L2_ready": 1189,
            "L3_direct_only": 119,
            "strict_secondary": 27,
        },
    }
    (HERE / "current_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    figures = HERE / "figures/current_results"
    figures.mkdir(parents=True, exist_ok=True)
    header = r"""\documentclass[border=2pt]{standalone}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\definecolor{forgeblue}{HTML}{2166AC}
\definecolor{forgeorange}{HTML}{B35806}
\definecolor{forgegreen}{HTML}{167D5B}
\begin{document}
\begin{tikzpicture}
"""
    overview = header
    common = r"width=2.55in,height=5.2in,xmin=0,xmax=100,xtick={0,25,50,75,100},ymin=-.7,ymax=21.7,y dir=reverse,ytick={0,...,21},tick label style={font=\scriptsize},label style={font=\small},title style={font=\small},xmajorgrids=true,grid style={gray!20},legend style={font=\scriptsize,draw=none,at={(.5,-.13)},anchor=north},legend columns=1"
    overview += (
        "\\begin{axis}["
        + common
        + ",name=left,yticklabels={"
        + ",".join("{" + x + "}" for x in labels)
        + r"},title={a. Product outcomes},xlabel={Percent of 64 requests}]"
        + "\n"
    )
    for metric, color, mark, label in [
        ("exact_L1", "forgeblue", "*", "Exact L1"),
        ("limited_design_pass", "forgeorange", "square*", "Limited design"),
        ("computational_makeability", "forgegreen", "triangle*", "Computationally makeable"),
    ]:
        coords = " ".join(f"({100*row['route'][metric]/64:.6f},{i})" for i, row in enumerate(rows))
        overview += f"\\addplot[only marks,mark={mark},mark size=1.8pt,color={color}] coordinates {{{coords}}};\n\\addlegendentry{{{label}}}\n"
    overview += (
        "\\end{axis}\n\\begin{axis}["
        + common
        + r",at={(left.east)},anchor=west,xshift=.35in,yticklabels=\empty,title={b. Synthetic CAL agreement},xlabel={Within-family percent}]"
        + "\n"
    )
    for metric, color, mark, label in [
        ("fingerprint_precision_among_unique", "forgeblue", "*", "Precision / distinct outputs"),
        ("fingerprint_coverage", "forgeorange", "square*", "Coverage / CAL references"),
    ]:
        coords = " ".join(
            f"({100*row['realism']['arms']['context_preserving']['fingerprints'][metric]:.6f},{i})"
            for i, row in enumerate(rows)
        )
        overview += f"\\addplot[only marks,mark={mark},mark size=1.8pt,color={color}] coordinates {{{coords}}};\n\\addlegendentry{{{label}}}\n"
    overview += "\\end{axis}\n\\end{tikzpicture}\n\\end{document}\n"
    (figures / "family_overview.tex").write_text(overview)
    training = header
    for i, (key, title, ticks) in enumerate(
        [
            (
                "updates",
                "a. Matched additional exposure",
                ["Original", "Large batch", "Small batch"],
            ),
            ("group66", "b. Matched 66-update continuation", ["Original", "Categorical", "Coarse"]),
        ]
    ):
        position = "name=left" if i == 0 else r"at={(left.east)},anchor=west,xshift=.5in"
        training += (
            r"\begin{axis}[width=2.85in,height=2.8in,ybar,bar width=9pt,ymin=0,ymax=170,ytick={0,50,100,150},xmin=-.55,xmax=2.55,xtick={0,1,2},xticklabels={"
            + ",".join(ticks)
            + "},"
            + position
            + ",title={"
            + title
            + r"},ylabel={Passes / 352 requests},tick label style={font=\scriptsize},title style={font=\small},label style={font=\small},nodes near coords,nodes near coords style={font=\scriptsize},legend style={draw=none,font=\scriptsize,at={(.5,-.16)},anchor=north},legend columns=2]"
            + "\n"
        )
        for metric, color, label in [
            ("exact", "forgeblue", "Exact L1"),
            ("design", "forgeorange", "Limited design"),
        ]:
            coords = " ".join(f"({j},{r[metric]})" for j, r in enumerate(diagnostics[key]))
            training += f"\\addplot[fill={color},draw={color}] coordinates {{{coords}}};\n\\addlegendentry{{{label}}}\n"
        training += "\\end{axis}\n"
    training += "\\end{tikzpicture}\n\\end{document}\n"
    (figures / "training_diagnostics.tex").write_text(training)
    print("Wrote current_summary.json and two figure sources; all 22 families retained.")


if __name__ == "__main__":
    main()
