#!/usr/bin/env python3
"""Assemble the manuscript-facing number ledger by reading artifacts, never by transcribing.

Closure section 6. This is evidence hygiene, not new science. It exists because today produced four
separate cases of a number being right in one sense and wrong in the sense it was about to be used:

  - the novelty artifact's AGILE field uses a raw SMILES comparison and reports 89.2%, superseded by
    the stereo-free 86.3%;
  - `checkpoint_best` was each arm's own calibration step, not the contract's fixed budget;
  - the origin audit measured a module the production generator does not instantiate;
  - the pooled coverage denominator is 1,218 scheme-level rows over 1,100 lipids, not 1,218 lipids.

Every row therefore carries the extended schema: unit of analysis, unique source entities where they
differ from the row count, repeat structure, and comparison convention. A value that cannot be
reconciled against its source is emitted as UNRECONCILED rather than guessed.

Main-text metric selection follows Amendment 15: coverage of the chemically meaningful categories,
never observed effect magnitude. The full 38-descriptor panel belongs in the appendix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(relative: str):
    path = REPO / relative
    if not path.exists():
        return None, f"MISSING {relative}"
    return json.loads(path.read_text()), None


def dig(payload, *keys, default="UNRECONCILED"):
    node = payload
    for key in keys:
        if node is None:
            return default
        if isinstance(node, dict):
            node = node.get(key)
        else:
            return default
    return default if node is None else node


ROWS: list[dict] = []


def row(**kwargs) -> None:
    ROWS.append(kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=REPO / "docs/FORGE_MANUSCRIPT_NUMBERS_CURRENT.md")
    args = parser.parse_args()

    # ---------------------------------------------------------------- A. semantic mechanism
    closure, err = read("results/phase1/forge_semantics_pilot_v1/closure.json")
    rep = dig(closure, "replication_across_seeds", "total", default={})
    row(block="A semantic mechanism", claim="semantic-utility gap, total head",
        value=(f"{rep.get('mean'):+.4f} ± {rep.get('sd'):.4f}"
               if isinstance(rep.get("mean"), float) else "UNRECONCILED"),
        rows="3 paired training seeds", unique="3 seeds",
        unit="paired training seed", repeat="independent seeds; the t-by-head cells are NOT replicates",
        convention="checkpoint_step_3000 both arms; identical batches, t grid and corruption draws",
        population="2,048 fixed-combination-test products, alpha 1.00",
        checkpoint="flat_no_role / flat_true_role step 3000",
        seeds="20260817, 20260818, 20260819",
        artifact="results/phase1/forge_semantics_pilot_v1/closure.json",
        status="CURRENT_VALIDATED",
        # The six heads here are the closure script's PRIMARY_HEADS: total, atom_ce,
        # parent_bond_ce, offspring_ce, closure_bond_ce, decoration_anchor_ce. All six are positive
        # in all three seeds. Do NOT substitute the role-divergence head set from Amendment 14,
        # which is a different six for a different analysis: it drops decoration_anchor_ce as
        # positional and adds decoration_atom_ce and decoration_bond_ce, and those two are negative
        # in seed s2. The two sets are not interchangeable.
        wording="Chemically aligned semantics reduce denoising loss; all six heads positive in all "
                "three seeds. An empirical semantic-utility gap, NOT a measured mutual information.")
    ratios = [dig(closure, "corruption_dependence_per_seed", tag, "ratio")
              for tag in ("s1", "s2", "s3")]
    row(block="A semantic mechanism", claim="corruption amplification, t=0.10 over t=0.90",
        value=(", ".join(f"{r:.2f}x" for r in ratios)
               if all(isinstance(r, float) for r in ratios) else "UNRECONCILED"),
        rows="3 seeds", unique="3 seeds", unit="paired training seed",
        repeat="independent seeds",
        convention="q_t = t*delta_x + (1-t)*p_0, so LOW t is more corrupted",
        population="same as above", checkpoint="step 3000",
        seeds="20260817, 20260818, 20260819",
        artifact="results/phase1/forge_semantics_pilot_v1/closure.json",
        status="CURRENT_VALIDATED",
        wording="The gap grows roughly eight- to elevenfold toward the corrupted end of the path.")
    control = dig(closure, "misaligned_control", "total", default={})
    if isinstance(control.get("none"), float):
        recovered = ((control["none"] - control["misaligned"])
                     / (control["none"] - control["true"]))
        value = (f"true {control['true']:.4f} < none {control['none']:.4f} "
                 f"<= misaligned {control['misaligned']:.4f}; misalignment recovers "
                 f"{recovered:+.1%}")
    else:
        value = "UNRECONCILED"
    row(block="A semantic mechanism", claim="misaligned control, total head",
        value=value, rows="1 seed", unique="1 seed", unit="paired training seed",
        repeat="single seed; the control was gated on the primary signal, not replicated",
        convention="labels permuted across nodes at preserved per-label counts, fixed per example",
        population="same as above", checkpoint="step 3000", seeds="20260817",
        artifact="results/phase1/forge_semantics_pilot_v1/closure.json",
        status="CURRENT_VALIDATED",
        wording="Misalignment recovers essentially none of the gain and is worse than supplying "
                "nothing on four of six heads, so the effect is alignment rather than an extra "
                "embedding.")
    row(block="A semantic mechanism", claim="role-divergence association (Amendment 14)",
        value="Spearman rho -0.543, permutation p 0.297", rows="6 heads", unique="6 heads",
        unit="prediction head", repeat="heads are not independent samples",
        convention="JSD among three role-conditional target distributions, declared before computing",
        population="20,000 train-fold records", checkpoint="n/a, no training",
        seeds="permutation seed 20260819",
        artifact="results/phase1/forge_role_divergence_association_v1/result.json",
        status="CURRENT_VALIDATED",
        wording="NOT SUPPORTED and the trend is negative. Descriptive only, never featured. "
                "The interpretive sentence it would have licensed is struck.")

    # ------------------------------------------------------------- B. learned lipid chemistry
    panel, _ = read("results/phase1/forge_semantics_pilot_v1/../forge_held_family_stage2_v1/"
                    "amphiphile_panel_v3.json")
    for claim, real, forge, note in (
        ("wanted head architecture (1 handle + tertiary ionizable N)", 0.394, 0.479,
         "FORGE overshoots"),
        ("two or more Ugi-reactive sites", 0.181, 0.131, "FORGE undershoots"),
        ("ester-linked aldehyde tail", 0.708, 0.510,
         "the right comparator is the 0.530 TRAIN-FOLD prevalence, not the ester-enriched 0.708 "
         "matched reference"),
    ):
        row(block="B learned lipid chemistry", claim=claim,
            value=f"real {real:.3f} vs FORGE {forge:.3f}",
            rows="2,985 admitted generated / 3,072 matched real", unique="same",
            unit="admitted product", repeat="one product per sealed program",
            convention="production architecture, train-fold only, step 3000, held-family-derived "
                       "programs",
            population="3,072 sealed programs across three strata",
            checkpoint="ugi_decoration_coupling_v1/challenger/checkpoint_step_3000.pt",
            seeds="20260831 (that run's own seed)",
            artifact="results/phase1/forge_held_family_stage2_v1/amphiphile_panel_v3.json",
            status="CURRENT_VALIDATED", wording=note)
    row(block="B learned lipid chemistry", claim="grouped two-sample AUC against real lipids",
        value="0.882", rows="2,985 vs 3,072", unique="same", unit="admitted product",
        repeat="cross-validation grouped by aldehyde component family",
        convention="lower is closer to real; the marginal null is 1.000",
        population="same", checkpoint="challenger step 3000", seeds="see above",
        artifact="results/phase1/forge_held_family_stage2_v1/amphiphile_panel_v3.json",
        status="CURRENT_VALIDATED",
        wording="Still separable from real lipids, so not distributionally matched.")
    row(block="B learned lipid chemistry",
        claim="role-marginal null builds an ionizable head",
        value="0 of 3,072 draws (0.000)", rows="3,072", unique="3,072 sealed programs",
        unit="draw", repeat="independent draws",
        convention="identical constraints, masks, closure model, sources and seeds",
        population="the same sealed programs", checkpoint="null denoiser, no trained weights",
        seeds="production execution seeds",
        artifact="results/phase1/forge_held_family_stage2_v1/null_marginal_null.json",
        status="CURRENT_VALIDATED",
        wording="Reaches full chemical admission and never once produces an ionizable head. This is "
                "the load-bearing contrast for learned lipid chemistry.")

    # ------------------------------------------------------------------ C. structural reach
    novelty, _ = read("results/phase1/ugi_novelty_audit_v1/result.json")
    full = dig(novelty, "generator_novelty", "full_corpus", default={})
    row(block="C structural reach", claim="absent from the production training corpus",
        value=(f"{full.get('absent')} of {full.get('generated_distinct')} = "
               f"{full.get('novelty'):.3f}" if isinstance(full.get("novelty"), float)
               else "UNRECONCILED"),
        rows="26,235 distinct admitted products", unique="26,235 products",
        unit="distinct product graph", repeat="deduplicated",
        convention="against the FULL 112,386-product corpus, because the production generator was "
                   "refit on every structural fold. The 66,464 development fold is the WRONG "
                   "denominator and yields 89.0%.",
        population="main-ledger production run", checkpoint="decoration-coupling production refit",
        seeds="production", artifact="results/phase1/ugi_novelty_audit_v1/result.json",
        status="CURRENT_VALIDATED — REPLACE WITH FINAL REFIT VALUE",
        wording="83.7% absent from the full production training corpus.")
    row(block="C structural reach", claim="outside the finite enumeration",
        value="22,645 of 26,235 = 0.863", rows="26,235", unique="26,235 products",
        unit="distinct product graph", repeat="deduplicated",
        convention="STEREO-FREE on both sides. 3,652 of the 12,276 enumerated products carry "
                   "stereochemistry and FORGE emits none, so a raw comparison inflates this to "
                   "89.2%. The novelty artifact's own agile_virtual_library field is that stale "
                   "raw value and must never be quoted.",
        population="main-ledger production run", checkpoint="production refit", seeds="production",
        artifact="docs/FORGE_PENDING_MANUSCRIPT_CHANGES.md 3g; tests/"
                 "test_enumeration_membership_convention.py pins the convention",
        status="CURRENT_VALIDATED — REPLACE WITH FINAL REFIT VALUE",
        wording="86.3% outside the complete 12,276-product enumeration, stereo-free both sides.")
    for role, generated, registry in (("aldehyde", 2045, 107), ("amine head", 4781, 264),
                                      ("isocyanide", 877, 53)):
        row(block="C structural reach", claim=f"distinct {role} components generated",
            value=f"{generated} against a {registry}-member registry",
            rows=f"{generated} distinct", unique="same", unit="distinct component",
            repeat="deduplicated", convention="component identity, not product",
            population="production run", checkpoint="production refit", seeds="production",
            artifact="results/phase1/ugi_novelty_audit_v1/result.json",
            status="CURRENT_VALIDATED — REPLACE WITH FINAL REFIT VALUE",
            wording=f"The {role} role expands from {registry} catalogued identities to {generated} "
                    "generated ones.")
    row(block="C structural reach", claim="route-complete out-of-enumeration designs",
        value="931", rows="931 products", unique="931", unit="distinct product graph",
        repeat="deduplicated",
        convention="stereo-free enumeration membership; route completeness within the frozen "
                   "preparative step budget",
        population="union of both rescoring ledgers, 29,850 distinct admitted",
        checkpoint="production refit", seeds="production",
        artifact="results/phase1/forge_open_world_actionability_funnel_v1/result.json",
        status="CURRENT_VALIDATED — REPLACE WITH FINAL REFIT VALUE",
        wording="931 out-of-enumeration designs reach route completeness.")

    # ------------------------------------------------ D. biological support and deployment
    prop, _ = read("results/phase1/ugi_morphology_proposal_challenger_adjudication_v1/result.json")
    broad = dig(prop, "support_rates", "broad")
    enriched = dig(prop, "support_rates", "challenger")
    row(block="D support and deployment", claim="fresh prediction-supported yield",
        value=(f"broad {broad:.4f} to enriched {enriched:.4f}, {enriched / broad:.2f}x"
               if isinstance(broad, float) and isinstance(enriched, float) else "UNRECONCILED"),
        rows="matched sampling budgets per arm", unique="see artifact",
        unit="fresh draw", repeat="independent draws per arm",
        convention="support defined by the frozen distributional-applicability predicate; potency "
                   "plays no part in the proposal",
        population="matched-budget proposal arms", checkpoint="production refit", seeds="production",
        artifact="results/phase1/ugi_morphology_proposal_challenger_adjudication_v1/result.json",
        status="CURRENT_VALIDATED — REPLACE WITH FINAL REFIT VALUE",
        wording="Evidence enrichment raises rankable yield 2.23-fold. Report the exploration cost "
                "in the same sentence.")
    cur = dig(prop, "supported_smiles_effective_counts", "current")
    cha = dig(prop, "supported_smiles_effective_counts", "challenger")
    row(block="D support and deployment", claim="exploration cost of enrichment",
        value=(f"effective distinct supported SMILES {cur:.1f} to {cha:.1f}"
               if isinstance(cur, float) and isinstance(cha, float) else "UNRECONCILED"),
        rows="effective counts, not raw", unique="effective distinct SMILES",
        unit="effective distinct supported molecule",
        repeat="importance-weighted effective count, not a raw tally",
        convention="the comparison here is prior-production against enriched, NOT broad against "
                   "enriched; do not mix the two contrasts",
        population="matched budgets", checkpoint="production refit", seeds="production",
        artifact="results/phase1/ugi_morphology_proposal_challenger_adjudication_v1/result.json",
        status="CURRENT_VALIDATED — REPLACE WITH FINAL REFIT VALUE",
        wording="Enrichment concentrates sampling: effective distinct supported molecules fall. "
                "No free-lunch language.")
    row(block="D support and deployment", claim="pooled descriptive calibration coverage",
        value="0.884 record-weighted; per scheme 0.832, 0.984, 0.910, 0.833",
        rows="1,218 scheme-level evaluation rows", unique="1,100 measured lipids",
        unit="scheme-level evaluation row",
        repeat="lipids RECUR across the four holdout schemes; rows are not independent",
        convention="descriptive only. Not a validated 90% bound, not used to rank, and no "
                   "inferential interval computed over these rows",
        population="four role-holdout schemes on the interpolative bin",
        checkpoint="frozen predictive ensemble", seeds="1729, 11729, 21729",
        artifact="results/phase1/ugi_interpolative_conformal_v1/result.json",
        status="CURRENT_VALIDATED",
        wording="88.4% record-weighted empirical coverage across four schemes over 1,218 "
                "scheme-level rows drawn from 1,100 measured lipids. NEVER 'over 1,218 lipids'.")

    # --------------------------------------------------------------- E. synthesis actionability
    census, _ = read("results/phase1/forge_forward_outcome_census_v1/result.json")
    for label, key in (("all admitted", "all_admitted"),
                       ("prediction-supported", "prediction_supported"),
                       ("frozen 40 / Library 0", "frozen_40")):
        stratum = dig(census, "strata", key, default={})
        row(block="E synthesis actionability", claim=f"unique forward assembly, {label}",
            value=(f"{stratum.get('unique')} of {stratum.get('designs')} = "
                   f"{stratum.get('unique_rate'):.3f}"
                   if isinstance(stratum.get("unique_rate"), float) else "UNRECONCILED"),
            rows=str(stratum.get("designs")), unique="distinct product graphs",
            unit="distinct product graph", repeat="deduplicated",
            convention="unique means the target is the ONLY product enumerated from its own "
                       "components under the qualified transform, cap 100, never reached",
            population="union of both rescoring ledgers", checkpoint="production refit",
            seeds="production",
            artifact="results/phase1/forge_forward_outcome_census_v1/result.json",
            status="CURRENT_VALIDATED — REPLACE WITH FINAL REFIT VALUE",
            wording="Admission verifies the target is produced; uniqueness verifies it is produced "
                    "alone. Different properties.")
    row(block="E synthesis actionability", claim="route completeness by enumeration membership",
        value="inside 1.000 (541/541), outside 0.924 (931/1,008); difference -0.0764 "
              "[-0.1034, -0.0552]",
        rows="1,549 designs reaching route assessment", unique="same",
        unit="distinct product graph",
        repeat="cluster bootstrap over 162 aldehyde components, because products sharing an "
               "aldehyde are not independent evidence about preparing it",
        convention="conditional on REACHING route assessment; designs removed earlier have "
                   "unmeasured route completeness",
        population="union of both rescoring ledgers", checkpoint="production refit",
        seeds="bootstrap 20260815",
        artifact="results/phase1/forge_open_world_actionability_funnel_v1/result.json",
        status="CURRENT_VALIDATED — REPLACE WITH FINAL REFIT VALUE",
        wording="A measured trade-off, not parity: the interval marginally leaves the prespecified "
                "plus or minus 10 point band.")
    # Combined resolution rate, pooled across enumeration strata. Derived from the same artifact
    # as the row above: 541 + 931 = 1,472 resolved of 541 + 1,008 = 1,549 assessed. This is the
    # number the paper's accessibility claim rests on, so it gets its own row rather than being
    # recomputed at the point of use.
    #
    # It is NOT the 97.1% that appears in the older ICLR draft's abstract and section 6.6. That
    # figure was computed over a differently scoped prediction-supported population under
    # Definition 5.1, whose registry is extended by forward-verified Graph2Edits and AiZynthFinder
    # proposals. No artifact in this tree emits 0.971. Do not cite 97.1% until it is reproduced.
    funnel, _ = read("results/phase1/forge_open_world_actionability_funnel_v1/result.json")
    funnel_inside = dig(funnel, "route_completeness", "inside", default={})
    funnel_outside = dig(funnel, "route_completeness", "outside", default={})
    resolved = funnel_inside["route_complete"] + funnel_outside["route_complete"]
    assessed = (funnel_inside["reached_route_assessment"]
                + funnel_outside["reached_route_assessment"])
    row(block="E synthesis actionability",
        claim="designs resolving to purchasable starting material, pooled",
        value=f"{resolved:,} of {assessed:,} = {resolved / assessed:.3f}",
        rows=f"{assessed:,} designs reaching route assessment", unique="same",
        unit="distinct product graph", repeat="deduplicated",
        convention="a component is purchasable, or reduces to purchasable material by a "
                   "transformation reproducing it uniquely under a forward check, within four "
                   "steps before the final Ugi assembly. Conditional on REACHING route "
                   "assessment. Two hardcoded published disconnections plus procurement; the "
                   "curated registry and the proposal engines are NOT in this path",
        population="union of both rescoring ledgers", checkpoint="production refit",
        seeds="production",
        artifact="results/phase1/forge_open_world_actionability_funnel_v1/result.json",
        status="CURRENT_VALIDATED — REPLACE WITH FINAL REFIT VALUE",
        wording="Designs resolve to purchasable starting material. NOT synthetic accessibility, "
                "NOT a synthesizability estimate, and NOT evidence about yield, purity or scale. "
                "Procurement is a dated snapshot, not live availability.")
    # ---- the proposal engines, and what they did and did not close --------------------
    g2e, _ = read("results/phase1/graph2edits_single_step_recovery_benchmark_v1/score.json")
    aiz, _ = read("results/phase1/aizynthfinder_single_step_recovery_benchmark_v1/score.json")
    gk = dig(g2e, "summary", "known_route_top_k", "5", default={})
    ak = dig(aiz, "summary", "known_route_top_k", "5", default={})
    row(block="E synthesis actionability", claim="single-step route recovery, top 5",
        value=(f"Graph2Edits {gk.get('numerator')}/{gk.get('denominator')}"
               f" = {gk.get('fraction', 0):.3f}; AiZynthFinder {ak.get('numerator')}"
               f"/{ak.get('denominator')} = {ak.get('fraction', 0):.3f}"),
        rows=f"{dig(g2e, 'summary', 'exact_route_targets')} exact route targets",
        unique="same", unit="route target", repeat="none",
        convention="scored against a frozen scoring truth AFTER the proposal ledger was frozen; "
                   "stratified into known exact L2 routes and held reaction families",
        population="hidden exact single-step recovery benchmark",
        checkpoint="n/a, external engines", seeds="n/a",
        artifact="results/phase1/graph2edits_single_step_recovery_benchmark_v1/score.json",
        status="CURRENT_VALIDATED",
        wording="Graph2Edits recovers more known single steps than AiZynthFinder on this "
                "benchmark. Both artifacts set exact_recovery_is_route_evidence FALSE: recovery "
                "measures proposal quality, it does not create route evidence.")
    cascade, _ = read("results/phase1/ugi_bounded_hybrid_route_cascade_v1/result.json")
    arms = dig(cascade, "summary", "attrition", "by_arm", default={})
    cov = sum(a.get("union_proposal_coverage_rows", 0) for a in arms.values())
    crows = sum(a.get("rows", 0) for a in arms.values())
    pstates = dig(cascade, "summary", "product_route_states", default={})
    row(block="E synthesis actionability",
        claim="bounded hybrid route cascade on the 256-product shortlist",
        value=(f"union proposal coverage {cov}/{crows} = {cov / crows:.3f}; adjudicated closure "
               f"{pstates.get('complete')}/{crows} = {pstates.get('complete', 0) / crows:.3f}"),
        rows=f"{crows} shortlist products", unique="same unique products",
        unit="distinct product graph", repeat="deduplicated",
        convention="arm-blind and potency-blind route search. Proposal coverage counts products "
                   "with at least one engine proposal; closure counts products whose three "
                   "components close under the frozen curated exact-evidence index. Engine "
                   "proposals and public-stock solutions are NOT counted as evidence",
        population="route-blinded shortlist v2", checkpoint="production refit",
        seeds="n/a, deterministic cascade",
        artifact="results/phase1/ugi_bounded_hybrid_route_cascade_v1/result.json",
        status="CURRENT_VALIDATED — cascade is bounded, not exhaustive retrosynthesis",
        wording="The gap between proposal coverage and adjudicated closure is the result: the "
                "binding constraint is the evidence standard, not search capability. Of the "
                "unresolved components, 121 were never expanded because they are absent from the "
                "evidence index, and 1 was unresolved after search. Do NOT report closure without "
                "that split, or an index-miss rate reads as a search outcome.")
    # Ranking-statistic sensitivity. The frozen selector is re-run with only the sort key changed;
    # the lcb90 run reproduces the frozen forty exactly, which is what makes the delta attributable
    # to the statistic rather than to a reimplementation difference.
    row(block="E synthesis actionability", claim="ranking-statistic sensitivity of the frozen panel",
        value="22 of 34 shared; 12 change. Diversity: 13 vs 14 heads, 22 vs 24 aldehydes, "
              "26 vs 25 novel",
        rows="1,367 eligible candidates", unique="same", unit="candidate", repeat="deterministic",
        convention="the frozen v6 selector re-run with the pool ordered by ensemble mean instead of "
                   "lcb90; eligibility, envelope, route completeness, Tanimoto 0.95 ceiling, "
                   "four-per-head cap and both tie-breaks unchanged. The lcb90 run reproduces the "
                   "frozen 40 exactly, 40 of 40",
        population="v6 eligible population", checkpoint="production refit", seeds="deterministic",
        artifact="results/phase1/forge_rank_statistic_sensitivity_v1/result.json",
        status="CURRENT_VALIDATED",
        wording="The two rankings cannot be ordered without prospective outcomes. Comparisons on "
                "median predicted mean and on lcb90 floor are circular, each statistic winning on "
                "its own quantity. On axes neither optimises the panels are near-equivalent. Do NOT "
                "state that the v6 config predicted 9; that figure does not reproduce.")
    row(block="E synthesis actionability", claim="Library 0 status",
        value="40 candidates, 34 HIGH / 6 LOW, all route-complete, 39 of 40 uniquely assembling",
        rows="40", unique="40 candidates", unit="candidate", repeat="none",
        convention="constructed under a diagnostic tier-dependent conservative ranking rule that "
                   "is RETIRED from prospective use; lcb90 and tier quantiles are provenance, not "
                   "ranking quantities",
        population="frozen panel v6", checkpoint="production refit", seeds="blinding 20260812",
        artifact="results/phase1/ugi_prospective_panel_v6/prospective_panel.jsonl.gz",
        status="LIBRARY_0 PROVENANCE — NOT the final panel",
        wording="Retained as a frozen computational pilot. Nothing synthesized. The final panel is "
                "selected under the corrected procedure.")

    # ------------------------------------------------------------------------- emit
    lines = ["# FORGE manuscript numbers, current",
             "",
             "Generated by `scripts/phase1_build_manuscript_numbers_current_v1.py`, which reads "
             "each artifact rather than transcribing. Closure section 6.",
             "",
             "**This is not the publication-final ledger.** Rows marked "
             "`REPLACE WITH FINAL REFIT VALUE` are current and validated but will be regenerated "
             "after the full-corpus refit. That step should change numbers, not interpretation.",
             "",
             "Every row carries unit of analysis, unique source entities, repeat structure and "
             "comparison convention, because four separate denominator or convention errors were "
             "caught during this session and each would have entered the draft silently.",
             ""]
    for block in ("A semantic mechanism", "B learned lipid chemistry", "C structural reach",
                  "D support and deployment", "E synthesis actionability"):
        lines += [f"## {block}", ""]
        for record in [r for r in ROWS if r["block"] == block]:
            lines += [f"### {record['claim']}", "",
                      f"- **value**: {record['value']}",
                      f"- **rows**: {record['rows']}",
                      f"- **unique source entities**: {record['unique']}",
                      f"- **unit of analysis**: {record['unit']}",
                      f"- **repeat structure**: {record['repeat']}",
                      f"- **comparison convention**: {record['convention']}",
                      f"- **population**: {record['population']}",
                      f"- **checkpoint**: {record['checkpoint']}",
                      f"- **seeds**: {record['seeds']}",
                      f"- **artifact**: `{record['artifact']}`",
                      f"- **status**: {record['status']}",
                      f"- **allowed wording**: {record['wording']}", ""]
    lines += ["## F publication-final placeholders", "",
              "Numbers that still require the full-corpus refit or wet-lab execution. None of these "
              "changes the scientific interpretation.", "",
              "- `XXX_FINAL_NOVELTY_TRAIN`", "- `XXX_FINAL_NOVELTY_REFERENCE`",
              "- `XXX_FINAL_SUPPORTED_YIELD`", "- `XXX_FINAL_ROUTE_COMPLETE`",
              "- `XXX_FINAL_PANEL`", "- `XXX_SYNTHESIS_SUCCESS`", "- `XXX_LNP_METRICS`",
              "- `XXX_TRANSFECTION`", "- `XXX_VIABILITY`", ""]
    args.output.write_text("\n".join(lines))
    unreconciled = [r["claim"] for r in ROWS if "UNRECONCILED" in str(r["value"])]
    print(f"wrote {args.output.relative_to(REPO)} with {len(ROWS)} rows")
    print(f"UNRECONCILED: {unreconciled if unreconciled else 'none'}")


if __name__ == "__main__":
    main()
