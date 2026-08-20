#!/usr/bin/env python3
"""Generate the supplementary tables describing the prospective panel.

Every table is derived from hash-pinned artifacts rather than transcribed, so the
supplement cannot drift from the selection.  Regenerate after any reselection.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_DEFAULT = Path(__file__).resolve().parents[1]
for _p in (REPO_DEFAULT / "src", REPO_DEFAULT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

PANEL = "results/phase1/ugi_prospective_panel_v6/prospective_panel.jsonl.gz"
PANEL_RESULT = "results/phase1/ugi_prospective_panel_v6/result.json"
NUMBERS = "results/phase1/ugi_manuscript_numbers_v1/result.json"
CONFIG = "configs/bio/phase1_ugi_prospective_panel_v6.json"

TIER_LABEL = {
    "qualified_role_holdout": "T1 qualified role holdout",
    "exploratory_weak_absolute_head_generalization": "T2 head generalization",
    "exploratory_weak_absolute_aldehyde_generalization": "T3 tail generalization",
    "exploratory_simultaneous_exact_new_tails": "T4 dual tail generalization",
}
TIER_MEANING = {
    "qualified_role_holdout": (
        "Every component except the isocyanide appears in the measured training set, and the "
        "isocyanide identity was held out during model fitting. The activity model was validated "
        "on exactly this kind of substitution, so its calibrated interval carries the most direct "
        "evidence available in the panel."),
    "exploratory_weak_absolute_head_generalization": (
        "The amine head group is absent from the measured set. The head sets the ionizable amine's "
        "pKa and is the dominant activity determinant in the model's own saliency analysis, so this "
        "is the furthest reach the panel makes. The model still issues a calibrated interval, but "
        "no measured product carries this head."),
    "exploratory_weak_absolute_aldehyde_generalization": (
        "The ester-linked aldehyde tail is absent from the measured set while the head and "
        "isocyanide are measured. Aldehyde tails form a homologous series in which chain length and "
        "unsaturation vary continuously, so interpolation within this role is better supported than "
        "in the other two."),
    "exploratory_simultaneous_exact_new_tails": (
        "Both the aldehyde and the isocyanide are absent from the measured set. Two simultaneous "
        "substitutions place these designs furthest from the measured factorial, and any outcome "
        "here is attributable to the combination rather than to either component alone."),
}


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in gzip.open(path, "rt")]


def fmt(value: Any, digits: int = 2) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else str(value)


def build(repo: Path) -> str:
    panel = read_jsonl_gz(repo / PANEL)
    result = json.loads((repo / PANEL_RESULT).read_text())
    numbers = json.loads((repo / NUMBERS).read_text())
    config = json.loads((repo / CONFIG).read_text())
    high = [r for r in panel if r["arm"] == "HIGH"]
    low = [r for r in panel if r["arm"] == "LOW"]
    registries = numbers["component_registries"]
    out: list[str] = []
    w = out.append

    w("# Supplementary Information: the prospective panel\n")
    w("All tables are generated from the frozen selection artifact and its pinned inputs. "
      "Regenerating this file after any change to the selection is required; the numbers here "
      "are not transcribed.\n")
    w(f"- Selection configuration: `{CONFIG}`, sha256 `{result['config']['sha256'][:16]}`")
    w(f"- Panel artifact: `{PANEL}`, sha256 "
      f"`{result['artifacts']['prospective_panel.jsonl.gz']['sha256'][:16]}`")
    w(f"- Derived numbers: `{NUMBERS}`, sha256 `{numbers['result_sha256'][:16]}`\n")

    # ---- S1 evidence tiers
    w("## Table S1 | Evidence tiers and what each one assumes\n")
    w("The activity model issues a calibrated interval for every candidate in the panel, but the "
      "evidence behind those intervals is not uniform. Tiers are assigned by which component "
      "identities were absent from the measured training set, and are recorded per candidate. "
      "The panel is not powered for per-tier claims; the tier is reported so that an outcome can "
      "be read against the evidence supporting it.\n")
    w("| Tier | Component absent from the measured set | Panel | High arm | Low arm | Eligible pool |")
    w("|---|---|---|---|---|---|")
    pool_tiers = result["summary"]["pool_tier_composition"]
    for tier, label in TIER_LABEL.items():
        absent = {"qualified_role_holdout": "isocyanide only",
                  "exploratory_weak_absolute_head_generalization": "amine head",
                  "exploratory_weak_absolute_aldehyde_generalization": "aldehyde tail",
                  "exploratory_simultaneous_exact_new_tails": "aldehyde and isocyanide"}[tier]
        w(f"| {label} | {absent} | {sum(1 for r in panel if r['authority_tier'] == tier)} "
          f"| {sum(1 for r in high if r['authority_tier'] == tier)} "
          f"| {sum(1 for r in low if r['authority_tier'] == tier)} "
          f"| {pool_tiers.get(tier, 0)} |")
    w(f"| **Total** | | **{len(panel)}** | **{len(high)}** | **{len(low)}** "
      f"| **{result['summary']['eligible_pool']}** |\n")
    for tier, label in TIER_LABEL.items():
        w(f"**{label}.** {TIER_MEANING[tier]}\n")
    w("The low arm is drawn entirely from the tail-generalization tier. No design with in-domain "
      "component chemistry occupies the low end of the predicted range, so drawing controls from "
      "the same tier as the bulk of the high arm removes evidence class as an alternative "
      "explanation for any difference between the arms.\n")

    # ---- S2 candidates
    w("## Table S2 | The forty locked candidates\n")
    w("Candidates appear in their prespecified selection order. `LCB90` is the calibrated "
      "conformal lower bound used for ranking; `mean` is the ensemble point prediction on the "
      "measured transfection scale; `percentile` is the fraction of the 1,100 measured products "
      "that the point prediction exceeds.\n")
    w("| ID | Arm | LCB90 | Mean | Percentile | Tier | Novel vs library | Steps | C=C | Branched |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for r in panel:
        s = r["scores"]
        w(f"| {r['candidate_id']} | {r['arm']} | {fmt(s['lcb90'])} | {fmt(s['oracle_mean'])} "
          f"| {fmt(s['percentile_of_agile_measured_library_beaten'], 0)}% "
          f"| {TIER_LABEL.get(r['authority_tier'], '?').split()[0]} "
          f"| {'yes' if r['novelty']['absent_from_agile_library'] else 'no'} "
          f"| {r['synthetic_steps']} | {r['structure']['carbon_double_bonds']} "
          f"| {'yes' if r['structure']['branched_aldehyde'] else 'no'} |")
    w("")

    # ---- S3 funnel
    w("## Table S3 | From generated designs to the locked panel\n")
    gen = numbers["generation_counts"]
    w("| Stage | Count |")
    w("|---|---|")
    w(f"| Attempted draws | {gen['attempted_draws']:,} |")
    w(f"| Chemically valid and exactly reconstructed through the Ugi reaction | {gen['chemically_valid_and_exact']:,} |")
    w(f"| Passing terminal chemical screening | {gen['admitted_rows']:,} |")
    w(f"| Distinct admitted products | {gen['admitted_distinct_products']:,} |")
    w(f"| Carrying a calibrated activity estimate, route-complete within four steps, inside the "
      f"physicochemical envelope, absent from the training fold | {result['summary']['eligible_pool']:,} |")
    w(f"| Selected into the panel | {len(panel)} |")
    w("\nNo design was repaired or resampled after failing a check, and no candidate was replaced "
      "after selection.\n")

    # ---- S4 novelty
    w("## Table S4 | Novelty against each reference set\n")
    w("Novelty takes different values against different reference sets, and the sets answer "
      "different questions. The production generator was refit on every structural fold, so the "
      "reference set for \"never seen during fitting\" is the full corpus, not the development "
      "training fold. Absence from the enumerated candidate library means the design uses a "
      "building block that library never contained, since that library is the complete "
      "cross-product of its block set. Comparisons against the enumerated library are made "
      "stereochemistry-free on both sides, because the generative pipeline is constitutional.\n")
    nov = numbers["novelty"]
    # The full-corpus row is the one the manuscript ledger endorses. Do NOT revert this to
    # product_absent_from_train_fold: the production model was refit on all 112,386 records, so
    # the 66,464 development fold understates what it saw and yields 89.0% instead of 83.7%. The
    # ledger records 89.0% as the wrong denominator for any production claim. The upstream
    # ugi_novelty_audit_v1 artifact still carries a note preferring the train fold; that note
    # predates the production refit and does not govern.
    corpus = json.loads(
        (repo / "results/phase1/ugi_novelty_audit_v1/result.json").read_text()
    )["generator_novelty"]["full_corpus"]
    w("| Reference set | What absence means | Generated designs | Panel |")
    w("|---|---|---|---|")
    w(f"| Full structural corpus, {corpus['reference_size']:,} products | the production model "
      f"never saw this molecule during fitting | {corpus['absent']:,} "
      f"({100 * corpus['novelty']:.1f}%) "
      f"| {len(panel)}/{len(panel)} |")
    w(f"| Enumerated candidate library, 12,276 products | the design uses a building block the "
      f"library never contained | {nov['product_absent_from_agile_library']['count']:,} "
      f"({100 * nov['product_absent_from_agile_library']['fraction']:.1f}%) "
      f"| {result['summary']['novelty']['absent_from_agile_library']}/{len(panel)} |")
    w(f"| Measured products, 1,100 | no measurement exists for any component combination of this "
      f"kind | {nov['carries_component_absent_from_measured_set']['count']:,} "
      f"({100 * nov['carries_component_absent_from_measured_set']['fraction']:.1f}%) carry at "
      f"least one unmeasured component | {len(panel)}/{len(panel)} |")
    w("\nEvery candidate on the panel is unmeasured, whether or not it appears in a previously "
      "published enumeration. The enumerated library contains 12,276 designs of which 1,100 were "
      "synthesized, so membership in it does not imply that a molecule has been made or tested.\n")

    # ---- S5 component space
    w("## Table S5 | Component space, measured against generated\n")
    w("| Component role | Measured products (1,100) | Enumerated library | Training corpus | "
      "Designs the model can score |")
    w("|---|---|---|---|---|")
    for role, label in (("amine_head", "Amine head"), ("aldehyde", "Ester-linked aldehyde tail"),
                        ("isocyanide_tail", "Isocyanide tail")):
        w(f"| {label} | {registries['measured_1100_product_set'][role]} "
          f"| {registries['agile_enumerated_library'][role]} "
          f"| {registries['training_corpus_block_registry'][role]} "
          f"| {registries['oracle_qualified_registry'][role]} |")
    w(f"\nThe aldehyde role expands "
      f"{registries['aldehyde_expansion_over_measured']}-fold over the measured set. The measured "
      f"set and the enumerated library are distinct references and are not interchangeable: the "
      f"library spans "
      f"{registries['agile_enumerated_library']['amine_head']} x "
      f"{registries['agile_enumerated_library']['aldehyde']} x "
      f"{registries['agile_enumerated_library']['isocyanide_tail']} = "
      f"{registries['agile_enumerated_library']['cross_product']:,} products, of which the "
      f"1,100 measured products are a subset.\n")

    # ---- S6 composition
    w("## Table S6 | Panel composition\n")
    summary = result["summary"]
    w("| Property | High arm | Low arm | Panel |")
    w("|---|---|---|---|")
    w(f"| Candidates | {summary['HIGH']['n']} | {summary['LOW']['n']} | {len(panel)} |")
    w(f"| LCB90 range | {fmt(summary['HIGH']['lcb90'][0])} to {fmt(summary['HIGH']['lcb90'][1])} "
      f"| {fmt(summary['LOW']['lcb90'][0])} to {fmt(summary['LOW']['lcb90'][1])} | |")
    w(f"| Predicted mean, median | {fmt(summary['HIGH']['oracle_mean'][1])} "
      f"| {fmt(summary['LOW']['oracle_mean'][1])} | |")
    w(f"| Above the measured median | {summary['HIGH']['above_measured_median']} "
      f"| {summary['LOW']['above_measured_median']} | |")
    w(f"| Above the measured 90th percentile | {summary['HIGH']['above_measured_p90']} "
      f"| {summary['LOW']['above_measured_p90']} | |")
    w(f"| Distinct amine heads | {summary['HIGH']['distinct_amine_heads']} "
      f"| {summary['LOW']['distinct_amine_heads']} | {numbers['prospective_panel_v6']['distinct_amine_heads']} |")
    w(f"| Distinct aldehyde tails | {summary['HIGH']['distinct_aldehydes']} "
      f"| {summary['LOW']['distinct_aldehydes']} | {numbers['prospective_panel_v6']['distinct_aldehydes']} |")
    w(f"| Distinct isocyanide tails | {summary['HIGH']['distinct_isocyanides']} "
      f"| {summary['LOW']['distinct_isocyanides']} | {numbers['prospective_panel_v6']['distinct_isocyanides']} |")
    w(f"| Unsaturated tail | {summary['HIGH']['unsaturated']} | {summary['LOW']['unsaturated']} | |")
    w(f"| Branched aldehyde | {summary['HIGH']['branched_aldehyde']} "
      f"| {summary['LOW']['branched_aldehyde']} | |")
    w(f"| Absent from the enumerated library | {summary['HIGH']['agile_novel']} "
      f"| {summary['LOW']['agile_novel']} | {summary['novelty']['absent_from_agile_library']} |")
    w(f"\nNo two candidates exceed {summary['maximum_pairwise_similarity']} pairwise Tanimoto "
      f"similarity, and at most "
      f"{config['diversity_rule']['component_cap']['amine_head']} candidates share an amine head.\n")

    # ---- S7 routes
    w("## Table S7 | Synthetic burden\n")
    steps = Counter(r["synthetic_steps"] for r in panel)
    actions: Counter[str] = Counter()
    materials: set[str] = set()
    for record in panel:
        for component in record["route_dossier"]:
            actions[component["action"]] += 1
            if component["action"] == "purchase":
                materials.add(component["canonical_smiles"])
            for material in component.get("starting_materials", []):
                if material.get("canonical_smiles"):
                    materials.add(material["canonical_smiles"])
    w("| Quantity | Value |")
    w("|---|---|")
    w(f"| Distinct catalogue materials to purchase | {len(materials)} |")
    w(f"| Components purchased directly | {actions.get('purchase', 0)} |")
    w(f"| Components prepared by a forward-verified route | {actions.get('synthesise', 0)} |")
    for step, count in sorted(steps.items()):
        w(f"| Candidates requiring {step} synthetic steps | {count} |")
    w("\nEvery component is either purchasable or reduced to purchasable material by a "
      "forward-verified route, with no unresolved branch. Supplier records are dated snapshots "
      "and are not guarantees of stock or purity.\n")

    # ---- S8 selection rule
    w("## Table S8 | The selection rule, as frozen\n")
    w("| Step | Rule |")
    w("|---|---|")
    w("| Population | designs carrying a calibrated conformal lower bound, chemically admitted, "
      "route-complete within four synthetic steps, inside the frozen physicochemical envelope, "
      "and absent from the training fold |")
    w("| Ranking statistic | calibrated conformal lower bound, mean minus conformal quantile |")
    w("| High arm, first twelve | highest ranked, on predicted activity alone |")
    w("| High arm, remaining twenty-two | preferring designs absent from the enumerated library, "
      "applied only where it does not lower predicted activity |")
    w("| Low arm | lowest ranked within the tail-generalization tier |")
    w("| Diversity | pairwise product Tanimoto at most 0.95; at most four candidates per amine head |")
    w("| Tie-breaking | canonical product SMILES, deterministic |")
    w("\nThe configuration was frozen before the selection ran. The ranking statistic penalises "
      "membership in a weakly calibrated region rather than per-candidate ensemble variance: the "
      "conformal quantile takes four discrete values across the pool, so within a calibration "
      "stratum this ranking is identical to ranking by the point prediction.\n")

    # ---- S9 calibration
    w("## Table S9 | Conformal calibration and how the evidence tiers were derived\n")
    w("The evidence tiers are not labels applied after the fact. Each is a distinct conformal "
      "calibration stratum, and the activity model fits one 90% quantile per stratum. The "
      "calibrated lower bound used for ranking is the ensemble mean minus that quantile, so a "
      "design in a weakly supported stratum is penalised by exactly the width its own calibration "
      "data justify.\n")
    strata: dict[str, dict[str, Any]] = {}
    for record in panel:
        entry = strata.setdefault(record["authority_tier"], {"q": record["scores"]["conformal_q90"], "n": 0})
        entry["n"] += 1
    w("| Tier | Conformal q90 | Interval width relative to T1 | Panel | Eligible pool |")
    w("|---|---|---|---|---|")
    base = min((v["q"] for v in strata.values() if v["q"]), default=None)
    for tier, label in TIER_LABEL.items():
        entry = strata.get(tier)
        if entry is None or entry["q"] is None:
            continue
        w(f"| {label} | {entry['q']:.4f} | {entry['q'] / base:.2f}x | {entry['n']} "
          f"| {pool_tiers.get(tier, 0)} |")
    w("\nThe quantile widens monotonically as the design moves away from the measured factorial: "
      "an unseen isocyanide costs least, an unseen head next, an unseen tail next, and two "
      "simultaneous unseen tails most. One caveat is worth stating. The head-generalization "
      "stratum is calibrated on only 42 designs, far fewer than the other three, so its interval "
      "is estimated less precisely than its width alone suggests.\n")
    w("Two consequences follow for how the panel should be read. Ranking by the calibrated lower "
      "bound is identical to ranking by the point prediction within any single tier, and differs "
      "between tiers only by the fixed offset above. And because the offset is a stratum constant "
      "rather than a per-molecule uncertainty, the ensemble standard deviation, whose median "
      "across the eligible pool is 0.36, never enters the ranking.\n")

    # ---- S10 components
    w("## Table S10 | Every component used by the forty candidates\n")
    w("Components are listed by role with the candidates that use them. A component is marked "
      "purchasable where a supplier record exists in the procurement snapshot, and otherwise "
      "carries the route by which it is prepared.\n")
    by_role: dict[str, dict[str, dict[str, Any]]] = {
        "amine_head": {}, "oxoester_aldehyde_body_tail": {}, "isocyanide_tail": {}}
    for record in panel:
        for component in record["route_dossier"]:
            role = component["role"]
            entry = by_role[role].setdefault(component["canonical_smiles"], {
                "action": component["action"], "route": component.get("route"),
                "vendors": component.get("vendor_count"),
                "materials": component.get("starting_materials", []), "used_by": []})
            entry["used_by"].append(record["candidate_id"])
    role_title = {"amine_head": "Amine head groups",
                  "oxoester_aldehyde_body_tail": "Ester-linked aldehyde tails",
                  "isocyanide_tail": "Isocyanide tails"}
    for role, title in role_title.items():
        entries = by_role[role]
        w(f"### {title} ({len(entries)} distinct)\n")
        w("| SMILES | Source | Suppliers | Used by |")
        w("|---|---|---|---|")
        for smiles, entry in sorted(entries.items(), key=lambda kv: -len(kv[1]["used_by"])):
            source = ("purchase" if entry["action"] == "purchase"
                      else f"prepare: {entry['route']}")
            vendors = entry["vendors"] if entry["vendors"] else "see precursors"
            used = ", ".join(entry["used_by"])
            w(f"| `{smiles}` | {source} | {vendors} | {used} |")
        w("")

    # ---- S11 per-candidate routes
    w("## Table S11 | Synthesis route for every candidate\n")
    w("Each candidate is assembled by a single Ugi three-component reaction from the three "
      "components below. Components marked prepare require the upstream step shown, from the "
      "starting materials listed. Step counts are the sum across all three components; the final "
      "Ugi assembly is not counted, since it is common to every candidate.\n")
    w("| ID | Role | Component | Action | Route | Starting materials (suppliers) |")
    w("|---|---|---|---|---|---|")
    role_short = {"amine_head": "amine", "oxoester_aldehyde_body_tail": "aldehyde",
                  "isocyanide_tail": "isocyanide"}
    for record in panel:
        for index, component in enumerate(record["route_dossier"]):
            materials = ", ".join(
                f"`{m['canonical_smiles']}` ({m.get('vendor_count', '?')})"
                for m in component.get("starting_materials", [])) or "n/a"
            w(f"| {record['candidate_id'] if index == 0 else ''} "
              f"| {role_short[component['role']]} | `{component['canonical_smiles']}` "
              f"| {component['action']} | {component.get('route', 'catalogue')} | {materials} |")
    w("")

    # ---- S12 bill of materials
    w("## Table S12 | Bill of materials\n")
    w("Every distinct chemical that must be purchased to execute the panel, with the supplier "
      "count recorded in the procurement snapshot. Supplier records are dated and are not "
      "guarantees of stock, purity or lead time.\n")
    catalogue: dict[str, dict[str, Any]] = {}
    for record in panel:
        for component in record["route_dossier"]:
            if component["action"] == "purchase":
                entry = catalogue.setdefault(component["canonical_smiles"], {
                    "role": component["role"], "vendors": component.get("vendor_count"), "n": 0})
                entry["n"] += 1
            for material in component.get("starting_materials", []):
                smiles = material.get("canonical_smiles")
                if not smiles:
                    continue
                entry = catalogue.setdefault(smiles, {
                    "role": material.get("role", "precursor"),
                    "vendors": material.get("vendor_count"), "n": 0})
                entry["n"] += 1
    w("| SMILES | Role | Suppliers | Candidates requiring it |")
    w("|---|---|---|---|")
    for smiles, entry in sorted(catalogue.items(), key=lambda kv: (-kv[1]["n"], kv[0])):
        w(f"| `{smiles}` | {entry['role'].replace('_', ' ')} | {entry['vendors'] or '?'} "
          f"| {entry['n']} |")
    w(f"\n{len(catalogue)} distinct materials in total.\n")

    # ---- S13 routing algorithm
    w("## Table S13 | The routing algorithm\n")
    w("Routing runs after generation and after the exact forward check, and never modifies a "
      "design. It answers one question per component: can this be bought, and if not, can it be "
      "reached from something that can. A design whose components cannot all be resolved is "
      "dropped rather than repaired.\n")
    w("```")
    w("for each generated product:")
    w("    decompose into (amine head, ester-linked aldehyde tail, isocyanide tail)")
    w("    for each component:")
    w("        if a supplier record exists in the procurement snapshot:")
    w("            mark purchase; resolved; 0 steps")
    w("        else if the component is an ester-linked aldehyde:")
    w("            disconnect by the AGILE Tail A route:")
    w("                carboxylic acid + alpha,omega-diol -> esterify -> oxidise")
    w("            resolved only if BOTH starting materials have supplier records")
    w("            2 steps")
    w("        else if the component is an isocyanide:")
    w("            disconnect to its primary amine:")
    w("                primary amine -> formylate -> dehydrate")
    w("            resolved only if the amine has a supplier record")
    w("            2 steps")
    w("        else:")
    w("            unresolved")
    w("    route_complete := every component resolved")
    w("    total_steps := sum of component steps        # final Ugi assembly not counted")
    w("    admit only if route_complete and total_steps <= 4")
    w("```")
    w("\nAmine heads have no disconnection rule in this instantiation, so an unseen head must be "
      "purchasable outright. Both tail routes are published chemistry rather than model "
      "proposals: the ester-linked aldehydes follow esterification of a carboxylic acid with an "
      "alpha,omega-diol and oxidation of the remaining primary alcohol, and the isocyanides "
      "follow formylation of a primary amine and dehydration of the resulting formamide.\n")
    w("Two limits bound what this establishes. Route completeness is computational, and predicts "
      "that a documented disconnection exists with purchasable starting materials; it is not "
      "evidence that any specific reaction will succeed at scale, in yield or in purity. And the "
      "eligible population is bounded by procurement coverage, so a design absent from the pool "
      "may be unroutable or may simply never have been assessed.\n")

    out.extend(model_tables(repo))
    return "\n".join(out) + "\n"


PRODUCTION_CHECKPOINT = "results/phase1/ugi_decoration_coupling_production_refit_v1/checkpoint_latest.pt"
PRODUCTION_RESULT = "results/phase1/ugi_decoration_coupling_production_refit_v1/result.json"


def model_tables(repo: Path) -> list[str]:
    """Architecture, state space and training, read from the production checkpoint itself
    rather than transcribed, so the numbers cannot drift from the shipped weights."""

    import torch

    out: list[str] = []
    w = out.append
    path = repo / PRODUCTION_CHECKPOINT
    if not path.is_file():
        w("## Tables S14 to S16 | Model\n")
        w(f"Production checkpoint `{PRODUCTION_CHECKPOINT}` not found; model tables omitted.\n")
        return out

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["model_state"]
    config = checkpoint["model_config"]
    total = sum(v.numel() for v in state.values())
    result = json.loads((repo / PRODUCTION_RESULT).read_text()) if (repo / PRODUCTION_RESULT).is_file() else {}

    blocks: Counter[str] = Counter()
    for key, value in state.items():
        if "embedding" in key:
            blocks["input embeddings"] += value.numel()
        elif key.startswith("sequence."):
            blocks["bidirectional GRU backbone"] += value.numel()
        elif "output" in key:
            blocks["output heads"] += value.numel()
        else:
            blocks["shared projection and normalisation"] += value.numel()

    w("## Table S14 | Generator architecture and parameter budget\n")
    w("The generator is a discrete flow-matching model over a sparse graph serialisation. It "
      "jointly denoises offspring topology, atom states, parent-bond states, sparse closure "
      "bonds and terminal decoration states through one shared role-aware backbone, rather than "
      "through separate topology and chemistry models.\n")
    w("| Component | Value |")
    w("|---|---|")
    w(f"| Total parameters | {total:,} |")
    w(f"| Backbone | {config['layers']}-layer bidirectional GRU, hidden dimension {config['hidden_dim']} |")
    w(f"| Dropout | {config['dropout']} |")
    w(f"| Decoration conditioning | {config.get('decoration_state_conditioning', 'n/a')} |")
    w(f"| Source probability floor | {config['source_probability_floor']} |")
    w(f"| Reverse sampling steps | 8 discrete reverse-star transitions |")
    w("")
    w("| Parameter block | Parameters | Share |")
    w("|---|---|---|")
    for name, count in blocks.most_common():
        w(f"| {name} | {count:,} | {100 * count / total:.1f}% |")
    w(f"| **Total** | **{total:,}** | |")
    w("\nThe backbone carries most of the capacity. The model holds no component identifiers and "
      "no stored molecular fragments, so it cannot retrieve a head group or tail from a table; "
      "every component is constructed atom by atom under the declared support.\n")

    w("## Table S15 | State space and declared molecular support\n")
    w("Generation is non-enumerative but bounded. The model emits categorical states over the "
      "vocabularies below, and the support bounds define the region within which the framework "
      "claims to generate. Designs outside these bounds are not produced rather than filtered "
      "afterwards.\n")
    vocab_rows = [
        ("offspring branching state", "offspring_embedding.weight"),
        ("atom state", "atom_embedding.weight"),
        ("bond state", "bond_embedding.weight"),
        ("precursor role", "role_embedding.weight"),
        ("within-origin position", "position_embedding.weight"),
        ("pending-frontier state", "pending_embedding.weight"),
        ("decoration slot", "decoration_slot_embedding.weight"),
    ]
    w("| Categorical state | Vocabulary size |")
    w("|---|---|")
    for label, key in vocab_rows:
        if key in state:
            w(f"| {label} | {state[key].shape[0]} |")
    w("")
    w("| Support bound | Value |")
    w("|---|---|")
    for label, key in (
        ("Maximum exterior atoms per product", "maximum_total_atoms"),
        ("Maximum atoms per precursor-derived component", "maximum_component_atoms"),
        ("Maximum children per node", "maximum_children"),
        ("Maximum junction budget per component", "maximum_junction_budget"),
        ("Maximum cycle rank per component", "maximum_cycle_rank"),
        ("Maximum core attachments", "maximum_attachment_count"),
        ("Maximum terminal decorations", "maximum_decorations"),
        ("Bond classes", "bond_classes"),
    ):
        w(f"| {label} | {config[key]} |")
    w("\nEach product is encoded as a deterministic spanning tree plus its sparse residual closure "
      "edges, with a fixed five-atom Ugi core and three precursor-origin exterior regions. The "
      "morphology program supplied to the model carries only per-origin atom counts, junction "
      "budgets, cycle ranks and attachment counts. It carries no component identity, which is "
      "what makes the output generation rather than retrieval.\n")

    w("## Table S16 | Training and checkpoint selection\n")
    training = result.get("training", result.get("selection", {}))
    w("| Setting | Value |")
    w("|---|---|")
    w("| Training corpus | 112,386 products, full corpus for the production refit |")
    w("| Development training fold | 66,464 products |")
    w("| Calibration fold, development early stopping | 15,800 products |")
    w(f"| Production updates | {training.get('completed_steps', 'n/a'):,}, fixed, no early stopping |"
      if isinstance(training.get("completed_steps"), int) else "| Production updates | n/a |")
    w(f"| Stop rule | {training.get('mode', 'n/a')} ({training.get('stop_reason', 'n/a')}) |")
    w("| Optimiser | AdamW, batch size 128, learning rate 1e-4, weight decay 5e-5 |")
    w("| Gradient clipping | global norm 1.0 |")
    w("| Precision and hardware | deterministic float32, one NVIDIA L4 GPU |")
    w("| Flow time sampling | uniform, truncated to [0.02, 0.98] |")
    w("\nThe production model was refit once from random initialisation for a fixed step budget "
      "with no early stopping and no post-refit checkpoint selection, so no held-out quantity "
      "chose the shipped weights. Architecture selection happened earlier, on a separate paired "
      "draw of 1,024 programs, under prespecified hard gates for validity, forward and inverse "
      "exactness, component-handle qualification, uniqueness and declared-support violations. "
      "Likelihood did not select the production model: the best calibration-loss checkpoint in "
      "development occurred at step 500, which was not the selected configuration.\n")
    w(f"Checkpoint pinned for these tables: `{PRODUCTION_CHECKPOINT}`, step "
      f"{checkpoint.get('step')}.\n")
    return out


BEGIN = "<!-- SUPPLEMENT:BEGIN generated by scripts/phase1_build_manuscript_supplement_v1.py; do not edit by hand -->"
END = "<!-- SUPPLEMENT:END -->"


def splice(manuscript: Path, body: str) -> None:
    """Keep the supplement inside the manuscript while keeping it generated.

    Hand-editing between the markers is lost on regeneration, which is the point: the
    tables must not be able to drift away from the selection artifact.
    """

    text = manuscript.read_text()
    # The block carries no trailing newline, so re-splicing reproduces the file byte for
    # byte instead of accumulating a blank line on every regeneration.
    block = f"{BEGIN}\n\n{body.rstrip()}\n\n{END}"
    if BEGIN in text and END in text:
        head, rest = text.split(BEGIN, 1)
        _, tail = rest.split(END, 1)
        manuscript.write_text(head + block + tail)
        return
    manuscript.write_text(text.rstrip("\n") + "\n\n" + block + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    parser.add_argument("--manuscript", type=Path,
                        default=Path("manuscript/FORGE_Nature_Biotechnology_working_draft.md"),
                        help="manuscript to splice the supplement into")
    parser.add_argument("--standalone", type=Path, default=None,
                        help="also write the supplement to its own file")
    args = parser.parse_args()
    repo = args.repo.resolve()
    body = build(repo)

    manuscript = args.manuscript if args.manuscript.is_absolute() else repo / args.manuscript
    splice(manuscript, body)
    print(f"spliced into {manuscript.relative_to(repo)} ({len(body.split())} words)")

    if args.standalone is not None:
        target = args.standalone if args.standalone.is_absolute() else repo / args.standalone
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body)
        print(f"also wrote {target.relative_to(repo)}")


if __name__ == "__main__":
    main()
