"""Render current quality tables from saved, hash-pinned scientific ledgers."""

import json
import math
from collections import Counter
from pathlib import Path

from forge.core.hashing import sha256_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
R = ROOT / "results/phase1/compose_lipid_iclr22_research_v1"
FILES = {
    "selected": R / "quality/all22_context_preserving/result.json",
    "verified": R / "quality/all22_context_preserving/verification.json",
    "completion": R / "evidence_completion_v1/result.json",
    "distribution": R / "qualified_CAL_sensitivity/result.json",
    "distribution_protocol": R / "qualified_CAL_sensitivity/protocol.json",
    "latest": R / "parallel_completion_v1/routes/recount_closeout_v3/result.json",
    "reference_audit": R / "parallel_completion_v1/quality/reference_audit.json",
    "lineage": R / "parallel_completion_v1/reference_lineage/reconstruction.json",
}
LABELS = {
    "a3_amine_aldehyde_alkyne": "A3",
    "acid_epoxide_diester_multistep": "Acid--epoxide",
    "aema_aza_thiol_addition": "AEMA",
    "aldehyde_ugi3": "Aldehyde Ugi3",
    "aldehyde_ugi4": "Aldehyde Ugi4",
    "alpha_isocyanoester_dihydroimidazole": r"$\alpha$-Isocyanoester",
    "amine_alkylation": "Amine alkylation",
    "amine_epoxide_opening": "Amine--epoxide",
    "aryl_reductive_amination": "Aryl reductive amination",
    "aza_michael_acrylamide": "Aza-Michael acrylamide",
    "aza_michael_acrylate": "Aza-Michael acrylate",
    "disulfide_michael": "Disulfide Michael",
    "epoxide_opening_o_acylation": "Epoxide/O-acylation",
    "iphos_ring_opening": "iPhos",
    "ketone_isocyanide_amide": "Ketone--isocyanide",
    "ketone_ugi4": "Ketone Ugi4",
    "maleate_addition": "Maleate",
    "o_esterification": "O-esterification",
    "passerini_3cr": "Passerini",
    "preassembled_thiol_yne_tail_amidation": "Preassembled thiol-yne",
    "reductive_amination": "Reductive amination",
    "thiolactone_aminolysis_michael": "Thiolactone",
}


def pin(p):
    return {"path": str(p.relative_to(ROOT)), "sha256": str(sha256_file(p))}


def read(p):
    return json.loads(p.read_text())


def save(p, d):
    p.write_text(json.dumps(d, indent=2, sort_keys=True, allow_nan=False) + "\n")


def diversity(items):
    counts = Counter(items)
    n = sum(counts.values())
    return {
        "observations": n,
        "unique": len(counts),
        "shannon_effective": (
            math.exp(-sum(v / n * math.log(v / n) for v in counts.values())) if n else None
        ),
        "inverse_simpson_effective": n * n / sum(v * v for v in counts.values()) if n else None,
    }


def novelty(rows):
    exact = [r for r in rows if r["exact"]]
    comp = [r for r in exact if any(c["global_train_novel"] for c in r["components"])]
    rolecomp = [r for r in exact if any(c["role_train_novel"] for c in r["components"])]
    cs = [c for r in exact for c in r["components"]]
    return {
        "requests": len(rows),
        "connected": sum(r["connected"] for r in rows),
        "distinct_exact": len({r["smiles"] for r in exact}),
        "TRAIN_novel_product_observations": sum(r["product_train_novel"] for r in rows),
        "TRAIN_novel_distinct_products": len(
            {r["smiles"] for r in rows if r["product_train_novel"]}
        ),
        "global_TRAIN_novel_component_products": len(comp),
        "global_TRAIN_novel_component_distinct_products": len({r["smiles"] for r in comp}),
        "role_TRAIN_novel_component_products": len(rolecomp),
        "role_TRAIN_novel_component_distinct_products": len({r["smiles"] for r in rolecomp}),
        "component_observations": len(cs),
        "global_TRAIN_novel_component_observations": sum(c["global_train_novel"] for c in cs),
        "global_TRAIN_novel_unique_components": len(
            {c["smiles"] for c in cs if c["global_train_novel"]}
        ),
        "product_diversity": diversity([r["smiles"] for r in rows]),
        "roles": {
            role: {
                **diversity([c["smiles"] for c in cs if c["role"] == role]),
                "global_TRAIN_novel_observations": sum(
                    c["global_train_novel"] for c in cs if c["role"] == role
                ),
                "global_TRAIN_novel_unique": len(
                    {c["smiles"] for c in cs if c["role"] == role and c["global_train_novel"]}
                ),
                "role_TRAIN_novel_observations": sum(
                    c["role_train_novel"] for c in cs if c["role"] == role
                ),
            }
            for role in sorted({c["role"] for c in cs})
        },
    }


def fmt(v):
    return r"\PendingCell" if v is None else f"{v:.3f}"


def row(*v):
    return " & ".join(str(x) for x in v) + r" \\"


def table(label, caption, heads, widths):
    spec = (
        "@{}"
        + "".join(r">{\raggedright\arraybackslash}p{" + w + r"\textwidth}" for w in widths)
        + "@{}"
    )
    return [
        r"\begingroup\scriptsize\setlength{\tabcolsep}{2.5pt}\renewcommand{\arraystretch}{1.1}",
        r"\begin{longtable}{" + spec + "}",
        r"\caption{\textbf{" + caption + r"}}\label{" + label + r"}\\",
        r"\toprule",
        row(*heads),
        r"\midrule\endfirsthead",
        r"\multicolumn{" + str(len(heads)) + r"}{l}{Table~\ref{" + label + r"}, continued}\\",
        r"\toprule",
        row(*heads),
        r"\midrule\endhead",
        r"\bottomrule\endfoot",
    ]


def end():
    return [r"\end{longtable}\endgroup"]


def main():
    docs = {k: read(p) for k, p in FILES.items()}
    assert docs["verified"]["all2816_source_indices_candidate_fields_smiles_exact_flags_verified"]
    assert (
        docs["latest"]["passed"]
        and docs["latest"]["all_quality_molecules_components_flags_unchanged"]
    )
    assert docs["distribution"]["complete"] and docs["distribution"]["references"] == 1821
    quality = docs["completion"]["quality"]["by_family"]
    selected = docs["selected"]["by_family"]
    distribution = {r["family"]: r for r in docs["distribution"]["rows"]}
    assert set(quality) == set(selected) == set(distribution) == set(LABELS)
    rows = [r for f in selected.values() for r in f["selections"]]
    lookup = {r["request"]: r for r in rows}
    assert len(rows) == len(lookup) == 1408
    for r in docs["latest"]["rows"]:
        a = lookup[r["index"]]
        assert (a["smiles"], a["ordinal"], a["exact"]) == (r["smiles"], r["ordinal"], r["exact_L1"])
    families = {
        f: {
            "label_latex": LABELS[f],
            "quality": quality[f],
            "novelty": novelty(selected[f]["selections"]),
        }
        for f in LABELS
    }
    totals = Counter()
    axes = {k: Counter() for k in ("chemical", "head", "ring")}
    for q in quality.values():
        assert q["requests"] == 64
        for k in (
            "requests",
            "exact",
            "design_pass",
            "design_and_no_context",
            "context",
            "observed",
        ):
            totals[k] += q[k]
        for k in axes:
            axes[k].update(q["axes"][k])
    assert dict(totals) == {
        "requests": 1408,
        "exact": 1387,
        "design_pass": 1324,
        "design_and_no_context": 1292,
        "context": 41,
        "observed": 974,
    }
    overall = {
        **dict(totals),
        "axes": {k: dict(v) for k, v in axes.items()},
        "novelty": novelty(rows),
    }
    overall["context_codes"] = dict(
        Counter(c for r in docs["latest"]["rows"] for c in r["context_codes"])
    )
    overall.update(
        {
            k: docs["latest"]["totals"][v]
            for k, v in {
                "makeable": "computational_makeability",
                "makeable_and_design": "joint_makeability_limited_design",
                "makeable_design_no_context": "joint_makeability_design_no_context_flags",
            }.items()
        }
    )
    catalog = read(HERE / "QUALITY_METRICS.json")
    descriptors = {r["key"]: r["label"] for r in catalog["descriptors"]}
    text = [
        "% Generated by build_quality_current.py. Hash ledger: current_quality_evidence.json.",
        r"\subsection{Current structural quality and design-condition evidence}",
        r"The selected development cohort contains 1,408 requests, 64 per family, from one trained checkpoint. All selected identities match the current routing cohort. These counts describe an adaptively inspected TRAIN-derived population, not independent-seed estimates or held-out quality certification. Saved denotes the preceding selection, not unconstrained raw generation; selected denotes the context-preserving selection. Subsequent ring-receipt completion changes assessment availability, not molecular identity. Unmeasured endpoints retain placeholders.",
    ]
    text += table(
        "tab:22-structure",
        "Structural outcomes on the current selected development cohort.",
        ["Family", "Valid/conn.", "Exact L1", "Design", "Alert", "Context", "Design, no context"],
        [".27", ".105", ".105", ".105", ".07", ".075", ".115"],
    )
    for f in LABELS:
        q = quality[f]
        text.append(
            row(
                LABELS[f],
                "64/64",
                f'{q["exact"]}/64',
                f'{q["design_pass"]}/64',
                f'{q["axes"]["chemical"].get("fail",0)}/64',
                f'{q["context"]}/64',
                f'{q["design_and_no_context"]}/64',
            )
        )
    text.extend(
        [
            r"\midrule",
            row(
                "All requests",
                "1408/1408",
                "1387/1408",
                "1324/1408",
                "0/1408",
                "41/1408",
                "1292/1408",
            ),
        ]
        + end()
    )
    text.append(
        r"Design requires exact L1 and all applicable qualified chemical, head and ring checks. Zero narrow-alert failures do not establish stability. Context flags overlap: hydrated/hemiacetal carbon 24, peroxide bond 8, carbon with at least three oxygens 6, other allene 5, small unsaturated ring 1 and heteroatom cumulene 1. The source-library screen passes 29/29 controls from two studies and two families; 20 families lack controls. Ten controls are component-disjoint; none supplies individual-product characterization in this control set. Independent blinded structural review remains \PendingValue{blinded structural review}."
    )
    text += table(
        "tab:22-conditions",
        "Scoped head, ring and TRAIN-feature diagnostics.",
        [
            "Family",
            "Head P/F/U",
            "Ring P/F/U",
            "Atom-env. coverage",
            "Ring-system support",
            r"Exact + observed/$N$",
        ],
        [".26", ".12", ".14", ".13", ".13", ".14"],
    )
    for f in LABELS:
        q = quality[f]
        h = q["axes"]["head"]
        r = q["axes"]["ring"]
        head = (
            "Outside scope"
            if h.get("not_applicable") == 64
            else f'{h.get("pass",0)}/{h.get("fail",0)}/{h.get("abstain",0)}'
        )
        text.append(
            row(
                LABELS[f],
                head,
                f'{r.get("pass",0)}/{r.get("fail",0)}/{r.get("abstain",0)}',
                r"\PendingCell",
                r"\PendingCell",
                f'{q["observed"]}/64',
            )
        )
    text.extend(
        [
            row(
                "All requests",
                "59/5/0; 1344 outside",
                "1339/69/0",
                r"\PendingCell",
                r"\PendingCell",
                "974/1408",
            )
        ]
        + end()
    )
    text.append(
        r"P/F/U denotes pass/fail/unassessed. Only the scoped neutral aldehyde-Ugi4 head predicate is qualified for this generated cohort; outside-scope cases are not passes. Ring status combines the applicable allocation/size checks. Separate all-request allocation and comparable-cycle size rates remain \PendingValue{separate ring-rate denominators}. Current-cohort atom-environment mean coverage and ring-system-support aggregates remain unmeasured. Exact-plus-observed requires exact, unambiguous decomposition and all product/component local features observed in the appropriate TRAIN references; its 974 successes do not establish independent lipid realism."
    )
    text.extend(
        [
            r"\subsection{Distributional agreement with synthetic CAL references}",
            r"The same 1,821 exact-source-qualified synthetic CAL references are used across arms. They predominantly share TRAIN components; only one reference is component-disjoint. This is a development distribution diagnostic. The available broad empirical reference and control panels contain 1,024 unique structures each, share all 12 study groups and contain 32 and 36 exact TRAIN overlaps, respectively. Source ancestry remains insufficient to admit an independent empirical panel.",
        ]
    )
    text += table(
        "tab:22-realism",
        "Fingerprint agreement and internal diversity, by family and saved arm.",
        [
            "Arm",
            r"Unique $G/R$",
            r"Prec. (\%)",
            r"Cov. (\%)",
            r"Recall (\%)",
            r"Distinct/$N$ (\%)",
            "Nearest",
            "Diversity",
        ],
        [".15", ".12", ".10", ".10", ".10", ".13", ".09", ".10"],
    )
    for f in LABELS:
        text.append(r"\multicolumn{8}{l}{\textbf{" + LABELS[f] + r"}; $N=64$ per arm}\\")
        for arm, label in [
            ("saved", "Saved"),
            ("context_preserving", "Selected"),
            ("TRAIN_control", "TRAIN control"),
        ]:
            fp = distribution[f]["arms"][arm]["fingerprints"]
            text.append(
                row(
                    label,
                    f'{fp["unique_generated"]}/{fp["unique_reference"]}',
                    *[
                        fmt(100 * fp[k])
                        for k in (
                            "fingerprint_precision_among_unique",
                            "fingerprint_coverage",
                            "fingerprint_recall",
                            "distinct_in_reference_manifold_per_request",
                        )
                    ],
                    fmt(fp["nearest_reference_tanimoto_mean"]),
                    fmt(fp["mean_pairwise_tanimoto_distance"]),
                )
            )
    text += end()
    text.append(
        r"Fingerprint precision is reference-neighborhood membership, not the probability that a molecule is chemically realistic. Large generated neighborhoods can inflate recall. TRAIN controls are not layout-matched independent controls. Independent empirical descriptor precision/coverage remain \PendingValue{independent descriptor precision/coverage}; source-grouped C2ST AUC and uncertainty remain \PendingValue{independent C2ST}."
    )
    text.append(
        r"\subsection{Definitions and interpretation of quality metrics}\label{app:22-metrics}"
    )
    text += table(
        "tab:22-metric-definitions",
        "Metric definitions and admitted scope.",
        ["Metric", "Computation and limitation"],
        [".29", ".65"],
    )
    for x in catalog["quality_metrics"] + catalog["realism_metrics"]:
        text.append(row(x["label"], x["definition"].replace("%", r"\%").replace("_", r"\_")))
    text += end()
    text.append(
        r"Fingerprints are 2,048-bit Morgan fingerprints of radius two (ECFP4). Reference neighborhoods have radii equal to the fifth nonself reference-neighbor Tanimoto distance. Internal diversity is mean Tanimoto distance over distinct generated identity pairs. Descriptor $W_1/s_R$ divides one-dimensional Wasserstein distance by reference IQR; for IQR at most $10^{-12}$, $s_R=\max(\mathrm{SD}_{\mathrm{population}},1)$. Connected-request descriptor observations retain duplicates. No distribution distance is assigned an acceptability threshold."
    )
    text += table(
        "tab:22-descriptors",
        "All 24 descriptor comparisons for every family.",
        [
            "Descriptor",
            r"$\mu_R$",
            r"$\mu_S$",
            r"$\mu_F$",
            r"$\mu_T$",
            r"$W_S/s_R$",
            r"$W_F/s_R$",
            r"$W_T/s_R$",
        ],
        [".29", ".085", ".085", ".085", ".085", ".09", ".09", ".09"],
    )
    for f in LABELS:
        text.append(
            r"\multicolumn{8}{l}{\textbf{"
            + LABELS[f]
            + r"}; $n_R="
            + str(distribution[f]["reference_counts"]["connected"])
            + r"$, $n_S=n_F=n_T=64$}\\"
        )
        for k, label in descriptors.items():
            a, b, c = [
                distribution[f]["arms"][arm]["descriptors"]["descriptors"][k]
                for arm in ("saved", "context_preserving", "TRAIN_control")
            ]
            assert a["reference_mean"] == b["reference_mean"] == c["reference_mean"]
            text.append(
                row(
                    label,
                    fmt(a["reference_mean"]),
                    *[fmt(x["generated_mean"]) for x in (a, b, c)],
                    *[fmt(x["normalized_wasserstein"]) for x in (a, b, c)],
                )
            )
    text += end()
    text.append(
        r"$R$ is synthetic CAL reference, $S$ saved predecessor selection, $F$ current selection and $T$ TRAIN control. Molecular weight is in Da and TPSA in \AA$^2$; other descriptors retain their count, fraction or calculated-logP units. Means and distances are point estimates. Descriptor medians, SDs, IQRs and tails were not exported by this computation and remain \PendingValue{descriptor distribution quantiles and tails}. Head/interface/tail fractions and polar-root distance are graph heuristics, not precursor-origin or ionization measurements."
    )
    text.append(r"\subsection{Product and precursor diversity and TRAIN-relative novelty}")
    text += table(
        "tab:22-diversity",
        "Current product novelty and exact-component diversity.",
        [
            "Family / role",
            r"$N$/obs.",
            "Unique exact",
            "Novel product",
            "Novel-comp. products",
            r"$D_1$",
            r"$D_2$",
            "Novel unique comp.",
        ],
        [".28", ".075", ".085", ".10", ".11", ".075", ".075", ".10"],
    )
    for f in LABELS:
        n = families[f]["novelty"]
        text.append(
            row(
                r"\textbf{" + LABELS[f] + "}",
                64,
                n["distinct_exact"],
                n["TRAIN_novel_product_observations"],
                n["global_TRAIN_novel_component_products"],
                fmt(n["product_diversity"]["shannon_effective"]),
                fmt(n["product_diversity"]["inverse_simpson_effective"]),
                n["global_TRAIN_novel_unique_components"],
            )
        )
        for role, c in n["roles"].items():
            text.append(
                row(
                    r"\quad " + role.replace("_", r"\_"),
                    c["observations"],
                    c["unique"],
                    "--",
                    str(c["global_TRAIN_novel_observations"]) + " obs.",
                    fmt(c["shannon_effective"]),
                    fmt(c["inverse_simpson_effective"]),
                    c["global_TRAIN_novel_unique"],
                )
            )
    n = overall["novelty"]
    text.extend(
        [
            r"\midrule",
            row(
                "Global products",
                1408,
                n["distinct_exact"],
                n["TRAIN_novel_product_observations"],
                n["global_TRAIN_novel_component_products"],
                fmt(n["product_diversity"]["shannon_effective"]),
                fmt(n["product_diversity"]["inverse_simpson_effective"]),
                n["global_TRAIN_novel_unique_components"],
            ),
        ]
        + end()
    )
    text.append(
        r"Family rows report products: unique exact identities, TRAIN-novel product observations and exact products containing a globally TRAIN-novel component. Product $D_1,D_2$ use all connected outputs. Indented role rows instead report exact component observations, unique components, novel-component observations and role-specific effective counts. Global deduplication gives 1,407 products; 1,216 TRAIN-novel observations represent 1,215 distinct products. Novel components occur in 1,050 exact products (1,049 distinct), or 1,081 products (1,080 distinct) under role-relative novelty. Of 3,723 exact component-role observations, 1,618 are globally novel, representing 1,350 distinct component constitutions. Full-reference novelty and exhaustive catalogue reachability remain \PendingValue{full-reference novelty and catalogue reachability}."
    )
    text.append(r"\subsection{Generalization beyond development requests}")
    text += table(
        "tab:22-generalization",
        "Generalization endpoints requiring independent evaluation.",
        [
            "Family",
            "Held-comp. yield",
            "Held identity coverage",
            "Component-disjoint L1",
            "Source-disjoint L1",
            "Size/repeat strata",
        ],
        [".27", ".12", ".14", ".14", ".14", ".12"],
    )
    for f in LABELS:
        text.append(row(LABELS[f], *[r"\PendingCell"] * 5))
    text += end()
    text.append(
        r"TRAIN-derived generated requests cannot fill these held-out endpoints. Source-qualified CAL distribution comparisons are not generation on independent requests. Independent training-seed uncertainty remains unmeasured. A metadata-only census identified 9,594 possible large-molecule test records across ten families, but admitted none for evaluation and computed no model scores. Held-reaction-family behavior remains a secondary stress test, not a hard gate."
    )
    tex = "\n".join(text) + "\n"
    target = HERE / "sections/quality_current.tex"
    labels = [
        "structure",
        "conditions",
        "realism",
        "metric-definitions",
        "descriptors",
        "diversity",
        "generalization",
    ]
    assert all(tex.count(r"\label{tab:22-" + x + "}") == 1 for x in labels)
    assert tex.count(r"\begin{longtable}") == tex.count(r"\end{longtable}") == 7
    target.write_text(tex)
    evidence = {
        "schema": "forge.iclr22.current_quality_evidence.v1",
        "evidence_class": "Computed",
        "final_evidence_admitted": False,
        "inputs": [pin(x) for x in FILES.values()],
        "producer": pin(Path(__file__)),
        "cohort": {
            "requests": 1408,
            "families": 22,
            "requests_per_family": 64,
            "training_checkpoints": 1,
            "split": "TRAIN-derived adaptive development",
            "all_selected_identities_match_latest_616_cohort": True,
        },
        "totals": overall,
        "by_family": families,
        "synthetic_CAL": {
            "references": 1821,
            "scope": "Synthetic, predominantly TRAIN-component-shared; one component-disjoint reference; not independent empirical realism.",
            "rows": docs["distribution"]["rows"],
        },
        "source_controls": docs["completion"]["source_controls"],
        "independent_reference": {
            "admitted": False,
            "audit": docs["reference_audit"],
            "lineage_admitted": docs["lineage"]["admitted_independent_reference"],
        },
        "unmeasured": [
            "independent empirical descriptor precision/coverage",
            "grouped C2ST and uncertainty",
            "independent seed uncertainty",
            "descriptor quantiles/SD/tails",
            "current atom-environment mean coverage and ring-system support aggregates",
            "separate current ring allocation and size denominators",
            "blinded independent structural review",
            "held-component/source-disjoint generation",
            "full-reference novelty and catalogue reachability",
        ],
        "outputs": [pin(target)],
        "new_model_chemistry_network_TEST_calls": 0,
    }
    save(HERE / "current_quality_evidence.json", evidence)
    catalog["status"] = (
        "Current development estimates recorded separately; final independent evidence pending"
    )
    catalog["current_evidence"] = pin(HERE / "current_quality_evidence.json")
    catalog["final_evidence_admitted"] = False
    metrics = {
        "valid_request_yield": (1408, 1408),
        "connected_request_yield": (1408, 1408),
        "exact_design_yield": (1324, 1408),
        "chemical_alert_rate": (0, 1408),
        "context_flag_rate": (41, 1408),
        "head_retention": (59, 64),
        "exact_observed_yield": (974, 1408),
    }
    for x in catalog["quality_metrics"]:
        x["final_value"] = None
        x["current_development"] = None
        if x["key"] in metrics:
            num, den = metrics[x["key"]]
            x["current_development"] = {
                "numerator": num,
                "denominator": den,
                "scope": "selected TRAIN-derived development",
            }
            if x["key"] == "head_retention":
                x["current_development"].update(family="aldehyde_ugi4", outside_scope_requests=1344)
    for x in catalog["realism_metrics"]:
        x["final_value"] = None
        k = x["key"]
        x["current_development"] = None
        if k == "normalized_wasserstein":
            x["current_development"] = {
                "reference": "synthetic_CAL",
                "values": "current_quality_evidence.json#synthetic_CAL.rows[*].arms[*].descriptors",
            }
        elif k in distribution[next(iter(LABELS))]["arms"]["context_preserving"]["fingerprints"]:
            x["current_development"] = {
                "reference": "synthetic_CAL",
                "by_family": {
                    f: distribution[f]["arms"]["context_preserving"]["fingerprints"][k]
                    for f in LABELS
                },
            }
    for x in catalog["descriptors"]:
        x["current_development"] = {
            f: {
                a: distribution[f]["arms"][a]["descriptors"]["descriptors"][x["key"]]
                for a in ("saved", "context_preserving", "TRAIN_control")
            }
            for f in LABELS
        }
    save(HERE / "QUALITY_METRICS.json", catalog)
    print(
        json.dumps(
            {
                "quality_tex": pin(target),
                "evidence": pin(HERE / "current_quality_evidence.json"),
                "catalog": pin(HERE / "QUALITY_METRICS.json"),
                "tables": 7,
                "descriptor_rows": 528,
                "families": 22,
            }
        )
    )


if __name__ == "__main__":
    main()
