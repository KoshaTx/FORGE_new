"""Bounded source review and development impact; no network calls or planner expansion."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors

from forge.assembly.families import RegistryAssemblyAdapter, constitutional_molecule
from forge.assembly.repeated_components import element_inventory
from forge.core.hashing import resolve_pin
from forge.corpus.source_evidence_adjudication import _l2_records
from forge.synthesis.engine.planner import (
    EvidenceRecord,
    EvidenceTier,
    ForwardVerificationState,
    KnowledgeDisposition,
    KnowledgeResult,
    PlannerBudgetLedger,
    PlannerBudgetLimits,
    RecursiveRouteAssessor,
    RouteStepProposal,
    RouteTarget,
)
from forge.synthesis.sources.supervision_inventory import sha256_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def pin(path: Path) -> dict:
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}


def write(name: str, value: dict) -> None:
    (HERE / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def adjudicate_aema(review: dict) -> dict:
    """Check a visually reviewed exact recipe with unchanged registry chemistry.

    The source transcription remains trusted review provenance, not an automatic
    inference from a source title, CAS number, or transform-compatible product.
    """
    source = review["source"]
    claim = review["claim"]
    mechanical = review["mechanical_check"]
    source_pin = {key: source[key] for key in ("path", "sha256")}
    source_path = resolve_pin(source_pin, ROOT, label="Zhou SI")
    registry_path = resolve_pin(mechanical["registry"], ROOT, label="O-acylation registry")
    if source["review_status"] != "chemistry_pages_rendered_and_visually_reviewed":
        raise ValueError("Source chemistry pages have not been visually reviewed")
    if not {1, 2, 18, 19, 20} <= set(source["visually_reviewed_physical_pages"]):
        raise ValueError("Source locators and neighboring pages are not reviewed")
    required = {
        "reagents",
        "solvent",
        "addition",
        "reaction",
        "workup",
        "purification",
        "product_specific_characterization",
        "outcome",
    }
    if any(not claim["procedure"].get(key) for key in required):
        raise ValueError("Incomplete source procedure or product characterization")
    adapter = RegistryAssemblyAdapter.from_registry(
        registry_path,
        reaction_id=mechanical["reaction_id"],
        expected_sha256=mechanical["registry"]["sha256"],
    )
    target, product = constitutional_molecule(claim["canonical_smiles"])
    components = {
        role: constitutional_molecule(item["canonical_smiles"])[0]
        for role, item in claim["precursors"].items()
    }
    roles = adapter.assess_roles(components)
    forward = adapter.forward_products(components, maximum_outcomes=mechanical["maximum_outcomes"])
    inverse = adapter.decompose(target, maximum_outcomes=mechanical["maximum_outcomes"])
    # The exact primary procedure resolves the pair. Preserve other inverse routes;
    # never claim that AEMA has a unique generic retrosynthesis.
    selected_inverse = [r for r in inverse if dict(r.components) == components]
    registry = json.loads(registry_path.read_text())
    stage = registry["grouped_programs"][0]["stages"][1]
    if stage["reaction_id"] != adapter.reaction_id:
        raise ValueError("Byproduct accounting is not from the selected registry event")
    byproducts = stage["net_byproducts_per_event"]
    left = Counter()
    for smiles in components.values():
        for key, count in element_inventory(smiles).items():
            left[key] += count
    right = element_inventory(target)
    for key, count in byproducts.items():
        right[key] += count
    balanced = all(left[key] == right[key] for key in left.keys() | right.keys())
    # Exact atom-origin inventories supplement formula conservation: all C/O atoms
    # are retained; only chloride is omitted from the mapped molecular product.
    molecules = tuple(Chem.MolFromSmiles(components[r]) for r in adapter.roles)
    raw = adapter.reaction.forward.RunReactants(
        molecules, maxProducts=mechanical["maximum_outcomes"]
    )
    retained = []
    for outcome in raw:
        atom_origins = [
            (a.GetIntProp("react_idx"), a.GetIntProp("react_atom_idx"))
            for a in outcome[0].GetAtoms()
            if a.HasProp("react_idx") and a.HasProp("react_atom_idx")
        ]
        expected = {
            (ri, a.GetIdx())
            for ri, mol in enumerate(molecules)
            for a in mol.GetAtoms()
            if a.GetSymbol() != "Cl"
        }
        retained.append(
            len(atom_origins) == product.GetNumAtoms()
            and len(set(atom_origins)) == len(atom_origins)
            and set(atom_origins) == expected
        )
    checks = {
        "all_unchanged_registry_roles_qualified": all(r.qualified for r in roles),
        "one_source_reactive_site_each": all(r.handle_count == 1 for r in roles),
        "unique_unfiltered_forward_product": forward.products == (target,),
        "unsaturated_forward_enumeration": not forward.saturated,
        "exact_source_specified_pair_present_once_in_inverse": len(selected_inverse) == 1,
        "full_element_hydrogen_charge_balance_with_registered_byproducts": balanced,
        "one_to_one_retention_of_all_nonleaving_atoms": bool(retained) and all(retained),
    }
    # Reuse the repository adjudicator's exact-identity/nonidentity-conflict policy
    # on one explicit source member. No historical M0 inputs/results are recovered.
    review_id = "zhou_2016_exact_aema_component"
    source_record = {"filename": source_path.name, "sha256": source["sha256"]}
    paper_review = {
        "l2_route_families": [
            {
                "route_family_id": review_id,
                "source_locator": source["locator"],
                "procedure": claim["procedure"],
                "members": [
                    {
                        "label": "AEMA",
                        "product_smiles": target,
                        "source_structure_status": claim["source_structure_status"],
                    }
                ],
            }
        ]
    }
    record = _l2_records(
        {review_id: paper_review},
        {review_id: source_record},
        {"series_member_basis": claim["evidence_basis"]},
    )[0]
    if not all(checks.values()):
        record.update(
            disposition="abstain",
            experimental_outcome_label="unresolved",
            route_closure="unresolved",
        )
    record["component_smiles_json"] = json.dumps(
        [components[role] for role in adapter.roles], separators=(",", ":")
    )
    hema = claim["precursors"]["aminoalcohol_head"]
    hema_mw = Descriptors.MolWt(Chem.MolFromSmiles(components["aminoalcohol_head"]))
    return {
        "record": record,
        "source_review": review,
        "mechanical_checks": checks,
        "roles": [asdict(r) for r in roles],
        "forward": asdict(forward),
        "all_inverse_candidates": [asdict(r) for r in inverse],
        "registered_net_byproducts": byproducts,
        "inventories": {"reactants": dict(left), "product_plus_byproducts": dict(right)},
        "computed_quantity_audit": {
            "HEMA_molecular_weight": hema_mw,
            "HEMA_moles_from_reported_mass": hema["reported_mass_g"] / hema_mw,
            "HEMA_reported_mol": hema["reported_mol"],
            "numeric_stoichiometry_and_yield_supervision_admitted": False,
        },
        "l2_status": (
            "exact_source_step_admitted" if record["disposition"] == "admit_exact" else "abstained"
        ),
        "l3_status": "not_assessed_for_source_route_leaves",
        "complete_component_dossier": None,
        "admitted_complete_component_dossier": False,
        "remaining_leaves": [
            {"canonical_smiles": smi, "current_l3": None, "status": "not_assessed"}
            for smi in components.values()
        ],
    }


def adjudicate_cached_vendor(observation: dict) -> dict:
    """This milestone's cache-only records cannot become current terminal receipts."""
    canonical, _ = constitutional_molecule(observation["canonical_smiles"])
    source_smiles = observation.get("source_smiles")
    identity = (
        None if source_smiles is None else constitutional_molecule(source_smiles)[0] == canonical
    )
    if identity is False:
        raise ValueError("Vendor structure and target constitution disagree")
    if observation.get("stock_observed_at_utc") is not None:
        raise ValueError("This cache-only boundary cannot authenticate a live stock timestamp")
    return {
        **observation,
        "exact_source_smiles_match": identity,
        "current_l3_disposition": "abstain",
        "current_l3": None,
        "admitted_current_terminal": False,
        "l2_status": "not_applicable_if_direct_terminal_later_admitted",
        "current_complete_component": None,
    }


def assess_admitted_step(source: dict) -> dict:
    """Send this single admitted step through the existing recursive assessor."""
    record = source["record"]
    if record["disposition"] != "admit_exact" or not all(source["mechanical_checks"].values()):
        raise ValueError("Only the mechanically verified exact source step may be assessed")
    target = RouteTarget("AEMA", record["target_canonical_smiles"])
    evidence = EvidenceRecord(
        record["evidence_record_id"],
        EvidenceTier.EXACT_SOURCE,
        record["source_asset_sha256"],
        record["source_locator"],
        True,
        ForwardVerificationState.VERIFIED_EXACT_PRODUCT_UNIQUE,
    )
    proposal = RouteStepProposal(
        record["reaction_family_id"],
        tuple(
            RouteTarget(role, item["canonical_smiles"])
            for role, item in source["source_review"]["claim"]["precursors"].items()
        ),
        (evidence,),
        forward_product_count=1,
    )

    class ExactReviewedPair:
        def lookup(self, requested: RouteTarget) -> KnowledgeResult:
            if requested == target:
                return KnowledgeResult(
                    KnowledgeDisposition.EXPAND,
                    (evidence,),
                    "One exact source-specified preparation; current leaves not established",
                    proposal,
                )
            return KnowledgeResult(
                KnowledgeDisposition.MISSING_KNOWLEDGE,
                (),
                "No admitted current terminal or exact upstream route for this leaf/context",
            )

    budget = PlannerBudgetLedger(PlannerBudgetLimits(1, 1, 1, 1, 0, 0))
    assessment = RecursiveRouteAssessor(ExactReviewedPair()).assess(target, budget)
    return {
        "assessment": assessment.to_dict(),
        "budget": budget.to_dict(),
        "scope": "Single source component; no final-assembly context receipt or L3 snapshot is fabricated",
    }


def build() -> dict:
    review_path = HERE / "source_review.json"
    vendors_path = HERE / "vendor_observations.json"
    priorities_path = (
        HERE.with_name("compose_lipid_quality_milestone_v1") / "development_priorities.json"
    )
    ledger_path = HERE.with_name("compose_lipid_route_coverage_v1") / "product_ledger.json"
    review = json.loads(review_path.read_text())
    vendor_document = json.loads(vendors_path.read_text())
    resolve_pin(vendor_document["policy"], ROOT, label="unchanged terminal evidence policy")
    previous = json.loads(ledger_path.read_text())
    # Bind the census to its original exact selected products, never to a new sample.
    resolve_pin(previous["inputs"]["selected"], ROOT, label="fixed development products")
    priorities = sorted(
        json.loads(priorities_path.read_text())["components"],
        key=lambda row: row["development_rank"],
    )[:5]
    if {r["canonical_smiles"] for r in priorities} != {
        r["canonical_smiles"] for r in vendor_document["observations"]
    }:
        raise ValueError("Evidence effort is not bound to the five selected priorities")
    with rdBase.BlockLogs():
        source = adjudicate_aema(review)
        vendors = [adjudicate_cached_vendor(r) for r in vendor_document["observations"]]
    component_assessment = assess_admitted_step(source)
    source_target = review["claim"]["canonical_smiles"]
    vendor_targets = {r["canonical_smiles"] for r in vendors}
    source_targets = {source_target} if source["record"]["disposition"] == "admit_exact" else set()
    products = []
    for old in previous["products"]:
        branches = []
        for branch in old["required_branches"]:
            smiles = branch["canonical_smiles"]
            branches.append(
                {
                    "role": branch["role"],
                    "canonical_smiles": smiles,
                    "exact_L2_step_documentary_evidence": smiles in source_targets,
                    "vendor_identity_lookup_candidate": smiles in vendor_targets,
                    "admitted_current_complete": False,
                    "current_complete": None,
                    "scope": "exact component evidence only; downstream assembly context and current leaves do not inherit closure",
                }
            )
        products.append(
            {
                "request_index": old["request_index"],
                "family": old["family"],
                "l1_exact": old["l1_exact"],
                "branches": branches,
                "admitted_current_complete_dossier": False,
                "current_complete_dossier": None,
                "status": "not_assessed" if not old["l1_exact"] else "unclosed_evidence_gaps",
            }
        )
    families = []
    for family in sorted({r["family"] for r in products}):
        rows = [r for r in products if r["family"] == family]
        exact = [r for r in rows if r["l1_exact"]]
        families.append(
            {
                "family": family,
                "requests": len(rows),
                "exact_l1": len(exact),
                "requests_with_a_new_exact_L2_step": sum(
                    any(b["exact_L2_step_documentary_evidence"] for b in r["branches"])
                    for r in exact
                ),
                "requests_with_any_selected_vendor_identity": sum(
                    any(b["vendor_identity_lookup_candidate"] for b in r["branches"]) for r in exact
                ),
                "requests_with_selected_vendor_identity_for_all_roles": sum(
                    bool(r["branches"])
                    and all(b["vendor_identity_lookup_candidate"] for b in r["branches"])
                    for r in exact
                ),
                "current_l2_all_role_complete": None,
                "current_l3_all_role_complete": None,
                "current_complete_dossiers": None,
                "admitted_current_complete_dossiers": 0,
            }
        )
    inputs = {
        name: pin(path)
        for name, path in {
            "producer": Path(__file__),
            "source_review": review_path,
            "vendor_transcription": vendors_path,
            "development_priorities": priorities_path,
            "prior_product_ledger": ledger_path,
            "live_access_failures": HERE / "sources/retrieval.json",
            "repository_adjudicator": ROOT / "forge/corpus/source_evidence_adjudication.py",
        }.items()
    }
    source_pin = {key: review["source"][key] for key in ("path", "sha256")}
    inputs["source_pdf"] = pin(resolve_pin(source_pin, ROOT, label="source PDF"))
    rendered = {path.name: pin(path) for path in sorted((HERE / "sources").glob("zhou*.png"))}
    result = {
        "schema_version": "forge.compose_lipid_route_evidence.v2",
        "reviewed_at_utc": review["reviewed_at_utc"],
        "randomness": {"used": False, "seed": 0},
        "inputs": inputs,
        "rendered_source_pages": rendered,
        "selection": "Five highest-ranked components in the fixed development priorities; no new sampling",
        "components": priorities,
        "source_adjudication": source,
        "recursive_component_assessment": component_assessment,
        "vendor_adjudications": vendors,
        "summary": {
            "selected_components": len(priorities),
            "new_exact_L2_source_steps": len(source_targets),
            "current_L3_terminals_admitted": 0,
            "current_complete_components_admitted": 0,
            "current_complete_products_admitted": 0,
            "development_requests": len(products),
            "families": len(families),
            "exact_l1": sum(r["l1_exact"] for r in products),
            "requests_with_new_exact_L2_step": sum(
                r["requests_with_a_new_exact_L2_step"] for r in families
            ),
            "requests_with_selected_vendor_identity": sum(
                r["requests_with_any_selected_vendor_identity"] for r in families
            ),
            "requests_with_selected_identity_for_all_roles": sum(
                r["requests_with_selected_vendor_identity_for_all_roles"] for r in families
            ),
        },
        "families": families,
        "context_reuse_rules": [
            "AEMA evidence binds only this exact source reactant pair, product, source hash, and procedure; neither Han source conditions nor execution transfer.",
            "Different final-family roles, quantities and stage contexts require the registered exact L1 assembly context before any component receipt is issued.",
            "Direct vendor evidence could make L2 not applicable only after exact identity, form, region, purity, item and current availability satisfy the existing policy.",
            "Cached page access time is not the time of a stock observation; never reset expiry using this review timestamp.",
            "One admitted L2 step cannot close its unassessed terminal leaves or another required product role.",
            "Missing current evidence is not synthetic impossibility or a negative experimental outcome.",
        ],
    }
    # Compute everything before publishing the three complete, deterministic artifacts.
    write("adjudication.json", result)
    write("product_evidence_ledger.json", {"inputs": inputs, "products": products})
    write("impact.json", {"inputs": inputs, "summary": result["summary"], "families": families})
    return result


if __name__ == "__main__":
    print(json.dumps(build()["summary"], sort_keys=True))
