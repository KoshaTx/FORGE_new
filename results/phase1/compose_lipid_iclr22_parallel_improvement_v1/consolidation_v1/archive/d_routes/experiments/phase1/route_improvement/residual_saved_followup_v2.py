"""Independent Boolean census and narrow already-saved donor-tree follow-up; no new paths."""

import argparse
import json
import resource
import sys
import time
from collections import Counter
from pathlib import Path

import residual_debt as audit

DEST = audit.DEST
OLD = (
    audit.MAIN
    / "results/phase1/compose_lipid_iclr22_research_v1/parallel_completion_v1/routes/next_priorities_v2/leaf_search_v1"
)
EVAL = (
    audit.MAIN
    / "results/phase1/compose_lipid_iclr22_research_v1/broad_routes_goal_v1/computational_makeability_v1/evaluation"
)
sys.path.insert(0, str(EVAL))
from datetime import datetime  # noqa: E402

from computational_makeability_v2 import (  # noqa: E402
    LookupOutcome,
    MakeabilityPolicy,
    Receipt,
    VendorListing,
    classify_listing,
)  # noqa: E402
from rdkit import Chem  # noqa: E402


def freeze():
    base = audit.read(audit.JOINT / "protocol.json")
    refs = {
        "producer": audit.pin(Path(__file__)),
        "main_result": audit.pin(DEST / "result.json"),
        "main_protocol": audit.pin(DEST / "protocol.json"),
        "current_protocol": audit.pin(audit.JOINT / "protocol.json"),
        "current_components": audit.pin(audit.JOINT / "components.json"),
        "current_products": audit.pin(audit.JOINT / "products.json"),
        "old_complete_census": audit.pin(OLD / "completion.json"),
        "old_target_result": audit.pin(OLD / "target_0/result.json"),
        "old_search_protocol": audit.pin(OLD / "protocol.json"),
        "listing_classifier": audit.pin(EVAL / "computational_makeability_v2.py"),
        "policy": base["policy"],
    }
    for i, ref in enumerate([base["baseline_listings"]] + base["vendor_envelopes"]):
        refs[f"listing_envelope_{i}"] = ref
    for key, ref in audit.read(DEST / "result.json")["artifacts"].items():
        refs["audit_" + key] = ref
    audit.write(
        DEST / "followup_protocol_v2.json",
        {
            "schema": "forge.residual_route_debt.saved_followup.v1",
            "inputs": refs,
            "clock": base["as_of_utc"],
            "seed": 0,
            "CPU_cap_seconds": 30,
            "new_HTTP_planner_model_search_new_paths": 0,
            "scope": "Independently recount every explicit batch directly from original saved component leaves and full product branches. Inspect all three previously retained CC(O)=S raw trees against current listing observations; no graft construction or admission. Missing current evidence remains unknown.",
        },
    )


def canonical(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    Chem.RemoveStereochemistry(mol)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol)


def run():
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    start = time.process_time()
    protocol = audit.read(DEST / "followup_protocol_v2.json")
    data = {}
    for key, ref in protocol["inputs"].items():
        path = audit.MAIN / ref["path"]
        assert audit.pin(path) == ref
        if path.suffix == ".json":
            data[key] = audit.read(path)
    products = data["current_products"]["products"]
    components = {r["identity"]: r for r in data["current_components"]["components"]}
    report = data["main_result"]

    def qualify(grant):
        closed = []
        for row in products:
            if not row["exact_L1"] or row["combined_primary"]:
                continue
            branches = []
            for branch in row["branches"]:
                component = components[branch["identity"]]
                paths = [p for p in component["paths"] if p["path_supported"]]
                branches.append(
                    branch["makeable"]
                    or any(
                        set(leaf["identity"] for leaf in path["leaves"] if not leaf["listed"])
                        <= grant
                        for path in paths
                    )
                )
            if all(branches):
                closed.append(row["index"])
        return closed

    batches = data["audit_explicit_listing_batches"]["records"]
    for batch in batches:
        assert qualify(set(batch["identities"])) == batch["conditional_product_indices"]
    for row in data["audit_leaf_rank_by_coverage"]["records"]:
        expected = []
        for product in products:
            if product["combined_primary"] or not product["exact_L1"]:
                continue
            if any(
                not branch["makeable"]
                and any(
                    path["path_supported"]
                    and any(
                        not leaf["listed"] and leaf["identity"] == row["identity"]
                        for leaf in path["leaves"]
                    )
                    for path in components[branch["identity"]]["paths"]
                )
                for branch in product["branches"]
            ):
                expected.append(product["index"])
        assert expected == row["coverage_product_indices"]
    invalid = [
        {
            "root": c["identity"],
            "root_currently_makeable": c["state"] in audit.CLOSED,
            "route_id": p["route_id"],
            "reasons": p["reasons"],
        }
        for c in components.values()
        for p in c["paths"]
        if not p["path_supported"]
    ]
    assert all(x["root_currently_makeable"] for x in invalid)
    census = data["old_complete_census"]
    raw = data["old_target_result"]
    assert (
        census["all_completed"]
        and census["exits"][0]["result"] == protocol["inputs"]["old_target_result"]
    )
    assert {
        "path": raw["protocol"].get("path", raw["protocol"].get("location")),
        "sha256": raw["protocol"]["sha256"],
    } == protocol["inputs"]["old_search_protocol"] and raw["canonical_smiles"] == "CC(O)=S"
    assert len(raw["top_route_hypotheses"]) == 3
    listing_rows = [
        row
        for key, value in data.items()
        if key.startswith("listing_envelope_")
        for row in value["listings"]
    ]
    policy = MakeabilityPolicy.from_mapping(data["policy"])
    as_of = datetime.fromisoformat(protocol["clock"])

    def states(identity):
        found = []
        for row in listing_rows:
            if row["identity"] != identity:
                continue
            ref = row["receipt"]
            value = VendorListing(
                identity,
                LookupOutcome(row["outcome"]),
                row["vendor_count"],
                datetime.fromisoformat(row["observed_at"]) if row["observed_at"] else None,
                Receipt(ref.get("path", ref.get("location")), ref["sha256"]),
            )
            found.append(classify_listing(identity, value, policy=policy, as_of=as_of).state.value)
        return sorted(set(found))

    def leaves(node):
        children = node.get("children", [])
        if not children:
            assert node["type"] == "mol"
            return [canonical(node["smiles"])]
        return [value for child in children for value in leaves(child)]

    trees = []
    for i, tree in enumerate(raw["top_route_hypotheses"]):
        ids = leaves(tree)
        rows = [
            {"identity": identity, "current_listing_states": states(identity)} for identity in ids
        ]
        missing = sorted(
            {r["identity"] for r in rows if "listed" not in r["current_listing_states"]}
        )
        trees.append(
            {
                "ordinal": i,
                "source": {
                    "location": protocol["inputs"]["old_target_result"]["path"]
                    + f"#/top_route_hypotheses/{i}",
                    "sha256": protocol["inputs"]["old_target_result"]["sha256"],
                },
                "historical_stock_solved": tree["metadata"]["is_solved"],
                "has_reaction": bool(tree.get("children")),
                "terminal_evidence": rows,
                "missing_exact_identities": missing,
                "all_current_listed": not missing,
            }
        )
    assert not any(r["all_current_listed"] for r in trees)
    indices = qualify({"CC(O)=S"})
    assert indices == [130, 133, 149, 165, 1274, 1276]
    saved_options = [
        {
            "listing_identity_set": r["missing_exact_identities"],
            "raw_tree_ordinal": r["ordinal"],
            "additional_preconditions": [
                "Independent exact saved-tree admission",
                "Independent exact full-root graft admission for every required root",
                "Full fixed-cohort real-evaluator paired recount",
            ],
            "conditional_maximum_additional_products": len(indices),
            "conditional_product_indices": indices,
            "new_paths_constructed": 0,
        }
        for r in trees
        if r["has_reaction"]
    ]
    totals = Counter()
    for row in products:
        category = (
            "makeable"
            if row["combined_primary"]
            else (
                "nonexact_L1"
                if not row["exact_L1"]
                else ("listing_only" if row["L2_ready"] else "missing_supported_component_path")
            )
        )
        totals[category] += 1
    result = {
        "schema": "forge.residual_route_debt.independent_saved_review.v1",
        "passed": True,
        "protocol": audit.pin(DEST / "followup_protocol_v2.json"),
        "fixed_totals": dict(totals),
        "all_explicit_listing_batches_directly_recounted": len(batches),
        "all_leaf_coverage_rows_independently_recounted": len(
            data["audit_leaf_rank_by_coverage"]["records"]
        ),
        "full_assessed_invalid_path_census": invalid,
        "residual_unclosed_root_invalid_path_count": 0,
        "main_result_field_scope": "assessed_invalid_paths in result.json is limited to currently unclosed roots; the one full-census cyclic path belongs to an already makeable root and is not a residual blocker.",
        "no_expired_listing_debt": not any(
            "expired" in str(r["currently_observed_listing_states"])
            for r in data["audit_leaf_rank_by_coverage"]["records"]
        ),
        "old_CC_O_S_search_all_three_trees": trees,
        "additional_saved_tree_options": saved_options,
        "best_direct_listing_option": report["next_smallest_listing_batch"],
        "exact_identity_caution": "No tautomer, salt, charge, isotope or element/form substitution is authorized. CC(O)=S, COC(C)=S and S are exact graph strings; a different returned graph does not close evidence.",
        "next_action": "Adjudicate the exact small-identity listing options before repeating any planner search. Direct CC(O)=S would conditionally close6; either saved nontrivial tree could transfer that same ceiling to its exact missing leaf set only after independent tree+graft admission. No sum across alternatives.",
        "new_evidence_or_gains": 0,
        "CPU_seconds": time.process_time() - start,
    }
    audit.write(DEST / "independent_review_v2.json", result)
    print(
        json.dumps(
            {
                "review": audit.pin(DEST / "independent_review_v2.json"),
                "CPU_seconds": result["CPU_seconds"],
                "saved_options": saved_options,
                "invalid": invalid,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["freeze", "run"])
    args = parser.parse_args()
    globals()[args.action]()
