#!/usr/bin/env python3
"""Does chemical expansion survive qualification, or does every gate collapse it to the library?

86.3% of distinct admitted designs lie outside the complete 12,276-product enumeration induced
by the AGILE component set, and 32 of the 40 frozen candidates are outside it. Those are two
isolated statistics at opposite ends of a long pipeline. Between them sit predictive support,
a physicochemical envelope, component resolution and a preparative step budget, any of which
could quietly narrow the actionable population back onto chemistry the measured library already
contained. This traces both strata through every stage.

Two disciplines carried over from the composition analysis, where ignoring them produced a
result that had to be withdrawn:

  - both estimands, always. Product counts and distinct-component counts answer different
    questions, and 4,587 products sharing one aldehyde are not 4,587 pieces of evidence about
    whether that aldehyde can be prepared;
  - uncertainty clustered at the aldehyde component, for the same reason.

One reading is fixed here, before computing, because the contract left it ambiguous. Route
completeness is reported two ways: unconditionally against the admitted population, and
conditionally on reaching route assessment at all. The prespecified plus or minus 10 point
parity rule applies to the CONDITIONAL rate, because that is the only one that measures
routing rather than the upstream gates that remove designs before routing ever runs.

Prespecified in docs/FORGE_EVIDENCE_CONTRACT_v1.md section D, committed before this ran.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import platform
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from importlib import import_module
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from rdkit import Chem, rdBase  # noqa: E402
from rdkit.Chem import Crippen, Descriptors  # noqa: E402

MAIN = "results/phase1/ugi_production_full_support_rescoring_v3/terminal_rescoring.csv.gz"
BRANCH = "results/phase1/ugi_branch_exploration_applicability_v1/terminal_rescoring.csv.gz"
PANEL = "results/phase1/ugi_prospective_panel_v6/prospective_panel.jsonl.gz"
CONFIG = "configs/bio/phase1_ugi_prospective_panel_v6.json"
AGILE = "results/m0_09/agile_virtual_ugi3_product_ledger.csv.gz"
CAPABILITY = "configs/route/m0_09_agile_virtual_ugi3_capability.json"
REACTIONS = "data/vendor/qualified_reactions_v1.json"

STAGES = ("chemically_admitted", "prediction_supported", "absent_from_train_fold",
          "inside_physicochemical_envelope", "all_components_resolved",
          "route_complete_within_steps")
ROLES = ("amine_head", "oxoester_aldehyde_body_tail", "isocyanide_tail")
BOOTSTRAP_SEED = 20260815
BOOTSTRAP_DRAWS = 2000
PARITY_MARGIN_POINTS = 10.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv_gz(path: Path):
    with gzip.open(path, "rt", newline="") as handle:
        yield from csv.DictReader(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=REPO / "results/phase1/forge_open_world_actionability_funnel_v1/result.json",
    )
    args = parser.parse_args()
    started = time.time()

    builder = import_module("phase1_build_ugi_prediction_cohort_panel_v3")
    index = builder.route_index(REPO)
    config = json.loads((REPO / CONFIG).read_text())
    screen = config["shared_requirements"]["physicochemical_screen"]
    tol = float(screen.get("comparison_tolerance", 0.0))
    max_steps = int(config["shared_requirements"]["route"]["maximum_synthetic_steps"])
    panel_mod = import_module("phase1_select_ugi_prospective_panel_v6")
    train = panel_mod.train_fold(REPO)

    scope = json.loads((REPO / CAPABILITY).read_text())["scope"]
    max_forward = int(scope["max_forward_outcomes_per_candidate"])
    from forge.corpus.r1_prime_audit import compile_reactions, load_reaction_definitions
    reaction = compile_reactions(load_reaction_definitions(
        (REPO / REACTIONS,), expected_count=1,
        role_policy_overrides=scope["role_policy_overrides"]))[0]

    canon_cache: dict[str, str] = {}
    free_cache: dict[str, str] = {}

    def canon(smiles: str) -> str:
        if smiles not in canon_cache:
            mol = Chem.MolFromSmiles(smiles)
            canon_cache[smiles] = Chem.MolToSmiles(mol) if mol is not None else ""
        return canon_cache[smiles]

    def stereo_free(smiles: str) -> str:
        """Enumeration membership only, and both sides must be stripped.

        3,652 of the 12,276 enumerated products carry stereochemistry; our pipeline is
        constitutional and emits none. A raw comparison records every stereo-bearing library
        member as chemistry the library never contained, which moves 798 designs from inside
        to outside and inflates the novelty share from 87.6% to 90.3%. The frozen panel
        builder already strips both sides, and this matches it.
        """
        if smiles not in free_cache:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                free_cache[smiles] = ""
            else:
                Chem.RemoveStereochemistry(mol)
                free_cache[smiles] = Chem.MolToSmiles(mol)
        return free_cache[smiles]

    with rdBase.BlockLogs():
        agile = {stereo_free(row["canonical_product_smiles"]) for row in read_csv_gz(REPO / AGILE)}
    agile.discard("")
    print(f"enumerated AGILE library: {len(agile)} distinct stereo-free products")

    panel = [json.loads(line) for line in gzip.open(REPO / PANEL, "rt")]
    panel_products = {record["canonical_product"] for record in panel}

    # ---------------------------------------------------------------- single pass
    counts: dict[str, Counter] = {stage: Counter() for stage in STAGES}
    unique_counts: dict[str, Counter] = {stage: Counter() for stage in STAGES}
    components: dict[str, dict[str, dict[str, set]]] = {
        stage: {"inside": {r: set() for r in ROLES}, "outside": {r: set() for r in ROLES}}
        for stage in STAGES
    }
    first_excluding_gate: dict[str, Counter] = {"inside": Counter(), "outside": Counter()}
    route_detail: dict[str, dict] = {
        stratum: {"steps": [], "components_requiring_preparation": [],
                  "failure": Counter(), "assessed": 0, "complete": 0}
        for stratum in ("inside", "outside")
    }
    # (aldehyde, stratum) -> [reached_routing, route_complete] for the clustered bootstrap
    clusters: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: {"inside": [0, 0], "outside": [0, 0]})
    seen: set[str] = set()

    with rdBase.BlockLogs():
        for source in (MAIN, BRANCH):
            path = REPO / source
            if not path.exists():
                print(f"  (skipping absent source {source})")
                continue
            for row in read_csv_gz(path):
                if row.get("terminal_chemical_admitted") != "True":
                    continue
                product = row.get("canonical_product")
                aldehyde = row.get("canonical_aldehyde")
                if not product or not aldehyde or product in seen:
                    continue
                seen.add(product)
                target = canon(product)
                stratum = "inside" if stereo_free(product) in agile else "outside"
                parts = {"amine_head": row["canonical_amine"],
                         "oxoester_aldehyde_body_tail": aldehyde,
                         "isocyanide_tail": row["canonical_isocyanide"]}

                mols = [Chem.MolFromSmiles(parts[r]) for r in
                        ("amine_head", "oxoester_aldehyde_body_tail", "isocyanide_tail")]
                built: set[str] = set()
                if all(m is not None for m in mols):
                    for group in reaction.forward.RunReactants(tuple(mols),
                                                               maxProducts=max_forward):
                        if len(group) != 1:
                            continue
                        try:
                            copy = Chem.Mol(group[0])
                            Chem.SanitizeMol(copy)
                            built.add(Chem.MolToSmiles(copy))
                        except Exception:  # noqa: BLE001
                            continue
                is_unique = target in built and len(built) == 1

                def mark(stage: str) -> None:
                    counts[stage][stratum] += 1
                    unique_counts[stage][stratum] += int(is_unique)
                    for role in ROLES:
                        components[stage][stratum][role].add(parts[role])

                mark("chemically_admitted")
                if row.get("oracle_scored") != "True":
                    first_excluding_gate[stratum]["prediction_supported"] += 1
                    continue
                mark("prediction_supported")
                if product in train:
                    first_excluding_gate[stratum]["absent_from_train_fold"] += 1
                    continue
                mark("absent_from_train_fold")

                molecule = Chem.MolFromSmiles(product)
                if molecule is None:
                    first_excluding_gate[stratum]["unparseable"] += 1
                    continue
                if not all((
                    screen["molecular_weight"][0] - tol <= Descriptors.MolWt(molecule)
                    <= screen["molecular_weight"][1] + tol,
                    screen["clogp"][0] - tol <= Crippen.MolLogP(molecule)
                    <= screen["clogp"][1] + tol,
                    screen["tpsa"][0] - tol <= Descriptors.TPSA(molecule)
                    <= screen["tpsa"][1] + tol,
                )):
                    first_excluding_gate[stratum]["physicochemical_envelope"] += 1
                    continue
                mark("inside_physicochemical_envelope")

                # Everything below has reached route assessment. This is the denominator the
                # parity rule uses, because designs removed above are never routed at all.
                route_detail[stratum]["assessed"] += 1
                clusters[aldehyde][stratum][0] += 1
                dossier = [builder.component_dossier(index, role, parts[role]) for role in ROLES]
                if not all(d["resolved"] for d in dossier):
                    for entry in dossier:
                        if entry["resolved"]:
                            continue
                        mode = ("no_preparation_schema" if entry["action"] == "unresolved"
                                else "schema_matched_but_starting_material_unpurchasable")
                        route_detail[stratum]["failure"][f"{entry['role']}:{mode}"] += 1
                    first_excluding_gate[stratum]["component_resolution"] += 1
                    continue
                mark("all_components_resolved")
                steps = sum(d.get("steps", 0) for d in dossier)
                if steps > max_steps:
                    route_detail[stratum]["failure"]["exceeds_step_budget"] += 1
                    first_excluding_gate[stratum]["step_budget"] += 1
                    continue
                mark("route_complete_within_steps")
                route_detail[stratum]["complete"] += 1
                route_detail[stratum]["steps"].append(steps)
                route_detail[stratum]["components_requiring_preparation"].append(
                    sum(1 for d in dossier if d["action"] == "synthesise"))
                clusters[aldehyde][stratum][1] += 1

    # ------------------------------------------------------------------ the funnel
    print(f"\n{'stage':34s} {'inside':>8s} {'outside':>9s} {'out %':>7s} "
          f"{'uniq in':>8s} {'uniq out':>9s}   {'ald in':>7s} {'ald out':>8s}")
    funnel = []
    for stage in STAGES:
        inside, outside = counts[stage]["inside"], counts[stage]["outside"]
        total = inside + outside
        record = {
            "stage": stage,
            "designs_inside": inside, "designs_outside": outside,
            "outside_share": outside / total if total else float("nan"),
            "unique_outcome_inside": unique_counts[stage]["inside"],
            "unique_outcome_outside": unique_counts[stage]["outside"],
            "distinct_components_inside": {r: len(components[stage]["inside"][r]) for r in ROLES},
            "distinct_components_outside": {r: len(components[stage]["outside"][r]) for r in ROLES},
        }
        funnel.append(record)
        print(f"{stage:34s} {inside:8d} {outside:9d} {record['outside_share']:7.3f} "
              f"{record['unique_outcome_inside']:8d} {record['unique_outcome_outside']:9d}   "
              f"{record['distinct_components_inside']['oxoester_aldehyde_body_tail']:7d} "
              f"{record['distinct_components_outside']['oxoester_aldehyde_body_tail']:8d}")

    panel_inside = sum(1 for record in panel if stereo_free(record["canonical_product"]) in agile)
    panel_flag_inside = sum(1 for record in panel
                            if not record["novelty"]["absent_from_agile_library"])
    if panel_inside != panel_flag_inside:
        raise SystemExit(
            f"enumeration membership disagrees with the frozen panel's own novelty flag "
            f"({panel_inside} vs {panel_flag_inside}); the comparison conventions have diverged")
    print(f"{'frozen_40_panel':34s} {panel_inside:8d} {len(panel) - panel_inside:9d} "
          f"{(len(panel) - panel_inside) / len(panel):7.3f}")
    print(f"{'selected_12':34s} {'pending':>8s} {'pending':>9s}")

    # ------------------------------------------------------- route completeness, two ways
    rates = {}
    for stratum in ("inside", "outside"):
        detail = route_detail[stratum]
        admitted = counts["chemically_admitted"][stratum]
        rates[stratum] = {
            "admitted": admitted,
            "reached_route_assessment": detail["assessed"],
            "route_complete": detail["complete"],
            "unconditional_rate_of_admitted": detail["complete"] / admitted if admitted else 0.0,
            "conditional_rate_given_assessed":
                detail["complete"] / detail["assessed"] if detail["assessed"] else float("nan"),
            "median_steps": statistics.median(detail["steps"]) if detail["steps"] else None,
            "mean_components_requiring_preparation":
                statistics.fmean(detail["components_requiring_preparation"])
                if detail["components_requiring_preparation"] else None,
            "failure_modes": dict(detail["failure"]),
            "first_excluding_gate": dict(first_excluding_gate[stratum]),
        }

    point = (rates["outside"]["conditional_rate_given_assessed"]
             - rates["inside"]["conditional_rate_given_assessed"])

    # Cluster bootstrap at the aldehyde component. Products sharing an aldehyde are not
    # independent evidence about whether that aldehyde can be prepared.
    keys = [a for a, v in clusters.items() if v["inside"][0] or v["outside"][0]]
    rng = random.Random(BOOTSTRAP_SEED)
    draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        acc = {"inside": [0, 0], "outside": [0, 0]}
        for _ in range(len(keys)):
            picked = clusters[keys[rng.randrange(len(keys))]]
            for stratum in ("inside", "outside"):
                acc[stratum][0] += picked[stratum][0]
                acc[stratum][1] += picked[stratum][1]
        if acc["inside"][0] and acc["outside"][0]:
            draws.append(acc["outside"][1] / acc["outside"][0]
                         - acc["inside"][1] / acc["inside"][0])
    draws.sort()
    lo = draws[int(0.05 * len(draws))]
    hi = draws[int(0.95 * len(draws)) - 1]

    print(f"\nroute completeness, {len(keys)} aldehyde clusters")
    for stratum in ("inside", "outside"):
        record = rates[stratum]
        print(f"  {stratum:8s} admitted {record['admitted']:6d}   assessed "
              f"{record['reached_route_assessment']:5d}   complete {record['route_complete']:5d}   "
              f"conditional {record['conditional_rate_given_assessed']:.3f}   "
              f"unconditional {record['unconditional_rate_of_admitted']:.4f}")
    print(f"  difference outside minus inside, conditional: {point:+.4f}  "
          f"[{lo:+.4f}, {hi:+.4f}] clustered at aldehyde")

    within = abs(lo) <= PARITY_MARGIN_POINTS / 100 and abs(hi) <= PARITY_MARGIN_POINTS / 100
    verdict = (
        f"PARITY: the clustered 90% interval for the conditional route-completeness difference "
        f"lies within plus or minus {PARITY_MARGIN_POINTS:.0f} points, so out-of-enumeration "
        f"designs retain the route-complete rate of in-enumeration designs."
        if within else
        f"MEASURED TRADE-OFF: the clustered 90% interval [{lo:+.3f}, {hi:+.3f}] leaves the "
        f"plus or minus {PARITY_MARGIN_POINTS:.0f} point parity band, so route completeness "
        f"differs by enumeration membership and the difference is reported as the result."
    )
    print(f"\n{verdict}")

    payload = {
        "schema_version": "phase1_forge_open_world_actionability_funnel.v1",
        "status": "complete_enumeration_stratified_funnel",
        "contract": "docs/FORGE_EVIDENCE_CONTRACT_v1.md section D, committed before this ran",
        "inputs": {
            "main_ledger": {"path": MAIN, "sha256": sha256_file(REPO / MAIN)},
            "branch_ledger": {"path": BRANCH, "sha256": sha256_file(REPO / BRANCH)},
            "panel": {"path": PANEL, "sha256": sha256_file(REPO / PANEL)},
            "panel_config": {"path": CONFIG, "sha256": sha256_file(REPO / CONFIG)},
            "agile_enumeration": {"path": AGILE, "sha256": sha256_file(REPO / AGILE)},
        },
        "runtime": {"python_version": platform.python_version(),
                    "platform": platform.platform(),
                    "elapsed_seconds": round(time.time() - started, 1)},
        "seeds": {"cluster_bootstrap": BOOTSTRAP_SEED, "draws": BOOTSTRAP_DRAWS,
                  "cluster_unit": "aldehyde component"},
        "unit": "distinct product graphs across the union of both rescoring sources",
        "stratum_definition": "inside iff the canonical product is one of the 12,276 products of "
                              "the complete AGILE component enumeration",
        "funnel": funnel,
        "frozen_40": {"inside": panel_inside, "outside": len(panel) - panel_inside},
        "selected_12": "not yet chosen",
        "route_completeness": rates,
        "conditional_difference_outside_minus_inside": {
            "point": point,
            "cluster_bootstrap_90_interval": [lo, hi],
            "parity_margin_points": PARITY_MARGIN_POINTS,
            "denominator": "designs that reached route assessment, fixed before computing "
                           "because the contract left the denominator ambiguous",
        },
        "verdict": verdict,
        "nonclaims": [
            "Route completeness is computational: a documented preparation exists from "
            "purchasable material under a dated snapshot, not evidence a synthesis will succeed.",
            "A stratum removed before route assessment has an UNMEASURED route completeness. The "
            "first-excluding-gate table records where each stratum is lost; it does not impute "
            "what would have happened downstream.",
            "Unique-outcome counts are reported per stage as an annotation, not applied as a "
            "filter. The frozen 40 was not selected under a uniqueness gate, and applying one "
            "retrospectively would make the panel row inconsistent with the stages above it.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(f"\nwrote {args.output.relative_to(REPO)}")


if __name__ == "__main__":
    main()
