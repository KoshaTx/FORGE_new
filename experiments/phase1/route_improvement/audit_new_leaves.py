"""Authenticate every retained leaf-search tree and apply existing exact listing evidence."""

import hashlib
import json
import resource
import sys
import time
from pathlib import Path

from rdkit import Chem

import forge

ROOT = Path(__file__).resolve().parents[3]
MAIN = (ROOT / "results").resolve().parent
OUT = MAIN / "results/phase1/compose_lipid_iclr22_parallel_improvement_v1/d_routes"
PAR = MAIN / "results/phase1/compose_lipid_iclr22_research_v1/parallel_completion_v1"
PREP = (
    PAR
    / "routes/next_bottlenecks_v3/new_leaf_listing_diagnostic_v1/paired_evaluation_preparation_v1"
)
BULK = (
    MAIN
    / "results/phase1/compose_lipid_iclr22_research_v1/broad_routes_goal_v1/computational_makeability_v1/routes/pubchem_bulk_qualification_v1"
)
SEARCH = OUT / "leaf_search_v1"
sys.path.insert(0, str(PREP))
import clock_wrapper as wrapper  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text())


def pin(path):
    path = Path(path).resolve()
    return {
        "path": str(path.relative_to(MAIN)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def canonical(identity):
    mol = Chem.MolFromSmiles(identity)
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError("Invalid or multicomponent exact identity")
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol)


def leaves(node):
    if node.get("step") is None:
        return [node["identity"]]
    return [value for child in node["step"]["reactants"] for value in leaves(child)]


def run():
    resource.setrlimit(resource.RLIMIT_CPU, (25, 25))
    started = time.process_time()
    assert Path(forge.__file__).resolve().is_relative_to(ROOT)
    protocol = read(SEARCH / "protocol.json")
    completion = read(SEARCH / "completion.json")
    assert completion["all_completed"] and len(completion["exits"]) == 4
    inputs = [
        pin(Path(__file__)),
        pin(SEARCH / "protocol.json"),
        pin(SEARCH / "completion.json"),
        pin(PREP / "clock_wrapper.py"),
        pin(PREP / "paired_run_v1/treatment/protocol.json"),
        pin(BULK / "target_inventory_v1.json"),
        pin(BULK / "join_v2/target_outcomes.json"),
    ]
    for reference in protocol["inputs"]:
        assert pin(MAIN / reference["path"]) == reference
    previous = read(PREP / "paired_run_v1/treatment/protocol.json")
    policy = wrapper.historical.MakeabilityPolicy.from_mapping(
        wrapper.historical.read_pin(previous["policy"])
    )
    as_of = wrapper.old.time(previous["as_of_utc"])
    listing_rows = []
    for reference in [previous["baseline_listings"]] + previous["vendor_envelopes"]:
        assert pin(MAIN / reference["path"]) == reference
        inputs.append(reference)
        listing_rows.extend(read(MAIN / reference["path"])["listings"])
    listing_states = {}
    for row in listing_rows:
        state = wrapper.old.classify_listing(
            row["identity"], wrapper.typed_listing(row), policy=policy, as_of=as_of
        ).state.value
        listing_states.setdefault(row["identity"], []).append(state)
    old_targets = {
        row["identity"]: row for row in read(BULK / "target_inventory_v1.json")["identities"]
    }
    old_outcomes = {
        row["identity"]: row for row in read(BULK / "join_v2/target_outcomes.json")["targets"]
    }
    trees, unknown, targets = [], {}, []
    for target, exit_row in zip(protocol["panel"], completion["exits"], strict=True):
        assert (
            exit_row["index"] == target["index"]
            and exit_row["completed"]
            and exit_row["exit_code"] == 0
        )
        result_ref = exit_row["result"]
        assert pin(MAIN / result_ref["path"]) == result_ref
        result = read(MAIN / result_ref["path"])
        assert result["status"] == "completed" and wrapper.historical.same_pin(
            result["protocol"], pin(SEARCH / "protocol.json")
        )
        assert result["seed"] == target["seed"] and result["canonical_smiles"] == target["identity"]
        assert canonical(target["identity"]) == target["identity"]
        assert result["verified_assets"] == protocol["asset_pins"][:-1]
        inputs.append(result_ref)
        decisions = []
        for ordinal, raw in enumerate(result["top_route_hypotheses"]):
            receipt = {
                "location": result_ref["path"] + f"#/top_route_hypotheses/{ordinal}",
                "sha256": result_ref["sha256"],
            }
            normalized = wrapper.historical.normalized_original(raw, receipt)
            assert normalized["identity"] == target["identity"]
            wrapper.old._original_leaves(raw, normalized, receipt)
            identities = leaves(normalized)
            leaf_rows = []
            for identity in identities:
                states = listing_states.get(identity, [])
                current = "listed" in states
                leaf_rows.append(
                    {"identity": identity, "listing_states": states, "current_listed": current}
                )
                if not current:
                    info = unknown.setdefault(
                        identity,
                        {
                            "identity": identity,
                            "inchikey": Chem.MolToInchiKey(Chem.MolFromSmiles(identity)),
                            "in_prior_6322_census": identity in old_targets,
                            "prior_bulk_outcome": old_outcomes.get(identity),
                            "trees": [],
                        },
                    )
                    info["trees"].append(
                        {"target_index": target["index"], "route_ordinal": ordinal}
                    )
            row = {
                "target_index": target["index"],
                "family": target["family_priority"],
                "identity": target["identity"],
                "route_ordinal": ordinal,
                "receipt": receipt,
                "historical_stock_solved": bool(raw.get("metadata", {}).get("is_solved")),
                "has_reaction": normalized.get("step") is not None,
                "leaves": leaf_rows,
                "all_exact_current_leaves_listed": bool(identities)
                and all(r["current_listed"] for r in leaf_rows),
                "normalized_original": normalized,
            }
            row["currently_closed_computational_tree"] = (
                row["has_reaction"] and row["all_exact_current_leaves_listed"]
            )
            trees.append(row)
            decisions.append(row["currently_closed_computational_tree"])
        targets.append(
            {
                "index": target["index"],
                "identity": target["identity"],
                "family": target["family_priority"],
                "seed": target["seed"],
                "statistics": result["statistics"],
                "retained_trees": len(decisions),
                "closed_trees": sum(decisions),
                "conditional_product_indices_if_closed": target["conditional_gain_indices"],
            }
        )
    write(
        SEARCH / "tree_audit_v1.json",
        {
            "schema": "forge.stream_d.exact_leaf_tree_audit.v1",
            "inputs": inputs,
            "all_four_declared_targets_retained": True,
            "as_of_utc": previous["as_of_utc"],
            "targets": targets,
            "trees": trees,
            "unknown_leaves": sorted(unknown.values(), key=lambda r: r["identity"]),
            "closed_trees": sum(r["currently_closed_computational_tree"] for r in trees),
            "retained_tree_count": len(trees),
            "unknown_identity_count": len(unknown),
            "new_HTTP_calls": 0,
            "no_availability_inferred_from_historical_stock": True,
            "candidate_only_not_independently_admitted": True,
            "CPU_seconds": time.process_time() - started,
        },
    )
    print(
        json.dumps(
            {
                "result": pin(SEARCH / "tree_audit_v1.json"),
                "closed_trees": sum(r["currently_closed_computational_tree"] for r in trees),
                "unknown_identities": len(unknown),
                "CPU_seconds": time.process_time() - started,
            }
        )
    )


if __name__ == "__main__":
    run()
