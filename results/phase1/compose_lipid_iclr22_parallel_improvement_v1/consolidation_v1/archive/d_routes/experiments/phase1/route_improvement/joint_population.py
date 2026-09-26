"""Replace nineteen molecular identities with their independently verified source witnesses."""

import copy
import hashlib
import json
import resource
import time
from pathlib import Path

from audit_new_leaves import MAIN, OUT, pin, read, write
from candidate_routes import BASE, prior

B = OUT.parent / "b_quality/closure_v1"
DEST = OUT / "joint_b_d_v1"


def content_sha(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def prepare():
    resource.setrlimit(resource.RLIMIT_CPU, (15, 15))
    start = time.process_time()
    protocol = read(BASE / "protocol.json")
    source = prior.read_pin(protocol["population"])
    changes = read(B / "changed_products.json")
    assert (
        changes["changed_count"] == 19 and changes["inherited_routes_for_changed_products"] is False
    )
    for ref in changes["inputs"].values():
        assert pin(MAIN / ref["path"]) == ref
    rows = copy.deepcopy(source["requests"])
    bridges, changed_indices = [], []
    for ordinal, row in enumerate(changes["rows"]):
        index = row["index"]
        original = rows[index]
        assert original["index"] == index and original["family"] == row["family"]
        assert original["selected_smiles"] == row["before_smiles"]
        assert not original["strict_complete"] and original["exact_L1"]
        assert row["inherited_route_verdict"] is None and row["inherited_model_likelihood"] is None
        assert pin(MAIN / row["parent_artifact"]["path"]) == row["parent_artifact"]
        assessment = row["source_assessment"]
        assert assessment["exact"] and assessment["status"] == "evaluated"
        accepted = [item for check in assessment["checks"] for item in check["accepted_components"]]
        values = {item["role"]: item["smiles"] for item in row["components"]}
        assert len(values) == len(row["components"])
        assert values in accepted
        assert set(values) == {item["role"] for item in original["requirements"]}
        assert all(check["exact_registry_program_roundtrip"] for check in assessment["checks"])
        for branch in original["requirements"]:
            branch["identity"] = values[branch["role"]]
        if row["family"] == "a3_amine_aldehyde_alkyne":
            counts = {branch["role"]: branch["quantity"] for branch in original["requirements"]}
            assert counts["amine_head"] == 1
            assert all(
                check["inverse"]["declared_events"] == counts["aldehyde"] == counts["alkyne"]
                for check in assessment["checks"]
            )
        original.update(
            {
                "canonical_product": row["smiles"],
                "selected_smiles": row["smiles"],
                "selected_ordinal": row["candidate_ordinal"],
                "strict_complete": False,
                "family_receipt": row["parent_artifact"],
                "l1_assessment_sha256": content_sha(assessment),
                "selected_L1_check_sha256": content_sha(assessment),
                "new_source_assessment": {
                    "location": str((B / "changed_products.json").relative_to(MAIN))
                    + f"#/rows/{ordinal}/source_assessment",
                    "sha256": pin(B / "changed_products.json")["sha256"],
                },
                "changed_molecule_no_inherited_route_verdict": True,
            }
        )
        bridges.append(
            {
                "index": index,
                "old_product": row["before_smiles"],
                "new_product": row["smiles"],
                "new_source_assessment": original["new_source_assessment"],
                "roles_quantities_stages_preserved": True,
                "strict_secondary_inherited": False,
            }
        )
        changed_indices.append(index)
    assert len(set(changed_indices)) == 19 and len(rows) == 1408
    assert sum(len(r["requirements"]) for r in rows) == 3723
    assert sum(r["strict_complete"] for r in rows) == 27
    assert sum(r["exact_L1"] for r in rows) == 1387
    assert all(
        a == b
        for a, b in zip(source["requests"], rows, strict=True)
        if a["index"] not in changed_indices
    )
    DEST.mkdir(exist_ok=False)
    write(
        DEST / "assembly_receipt_bridge.json",
        {
            "schema": "forge.current_and_changed_L1_witness_bridge.v1",
            "original_assemblies": source["inputs"]["assemblies"],
            "original_population": protocol["population"],
            "changed_products": pin(B / "changed_products.json"),
            "verified_source_handoff": pin(B / "verification.json"),
            "changes": bridges,
            "full_denominator": 1408,
        },
    )
    result = copy.deepcopy(source)
    result["schema"] = "forge.changed_molecule_computational_population.v1"
    result["inputs"]["assemblies"] = pin(DEST / "assembly_receipt_bridge.json")
    result["inputs"]["producer"] = pin(Path(__file__))
    result["inputs"]["selected"] = pin(B / "candidate_cohort_1408.json")
    result["inputs"]["old_population"] = protocol["population"]
    result["inputs"]["changed_source_witnesses"] = pin(B / "changed_products.json")
    result["requests"] = rows
    result["changed_indices"] = changed_indices
    result["official_cohort_replaced"] = False
    result["scope"] = (
        "Separate B+D candidate cohort: nineteen changed identities independently reassessed using unchanged route evidence; no old route verdict copied to a changed molecule. Remaining1389 request objects preserved exactly."
    )
    write(DEST / "requests.json", result)
    write(
        DEST / "preparation.json",
        {
            "passed": True,
            "inputs": [
                pin(Path(__file__)),
                protocol["population"],
                pin(B / "changed_products.json"),
                pin(B / "verification.json"),
                pin(B / "candidate_cohort_1408.json"),
            ],
            "population": pin(DEST / "requests.json"),
            "assembly_bridge": pin(DEST / "assembly_receipt_bridge.json"),
            "changed_indices": changed_indices,
            "unchanged_request_count": 1389,
            "roles_quantities_stages_preserved": True,
            "strict_secondary_inherited_for_changed": False,
            "CPU_seconds": time.process_time() - start,
            "independent_population_review_required": True,
        },
    )
    print(
        json.dumps(
            {"prepared": pin(DEST / "preparation.json"), "CPU_seconds": time.process_time() - start}
        )
    )


if __name__ == "__main__":
    prepare()
