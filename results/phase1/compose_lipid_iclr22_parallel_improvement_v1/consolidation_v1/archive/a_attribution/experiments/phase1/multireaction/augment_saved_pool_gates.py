"""Join authenticated identity-specific tree assessments into one common candidate ledger."""

import copy
import hashlib
import json
from pathlib import Path

from experiments.phase1.multireaction.saved_pool_attribution import (
    OUTPUT,
    WORKTREE,
    digest,
    pin,
    read,
    write,
)


def object_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def joined_assessment(target, candidate, original, basis):
    if not target["admitted"] or target["ambiguity"] or target["errors"]:
        raise ValueError("Tree evidence is not admitted and unambiguous")
    if len(target["recovered"]) != 1 or target["candidate_lineages"] != 1:
        raise ValueError("Exactly one authenticated lineage is required")
    recovery = target["recovered"][0]
    assessment = recovery["assessment"]
    if (candidate["request"], candidate["ordinal"]) != (target["index"], target["ordinal"]):
        raise ValueError("Candidate identity key changed")
    if assessment["chemical"]["canonical_smiles"] != candidate["smiles"]:
        raise ValueError("Candidate molecular identity changed")
    if original["chemical"]["canonical_smiles"] != candidate["smiles"]:
        raise ValueError("Original gate does not match candidate identity")
    if not recovery["exact_graph_roundtrip"] or not recovery["roles_core_origins_layout_invariant"]:
        raise ValueError("Molecular/tree transport invariance is not established")
    if assessment["ring"]["tree_basis"] != basis:
        raise ValueError("Transport basis is not the admitted basis")
    for key in ("chemical", "head", "source_exact"):
        if original[key] != assessment[key]:
            raise ValueError("Tree-only evidence changed another assessment axis: " + key)
    if original["status_by_axis"] != target["original_status"]:
        raise ValueError("Original gate status differs from admitted evidence")
    if assessment["status_by_axis"] != target["resulting_status"]:
        raise ValueError("Resulting gate status differs from admitted evidence")
    if assessment["qualified_design_pass"] != target["resulting_design"]:
        raise ValueError("Conjunctive gate result differs")
    return assessment


def main():
    out = OUTPUT / "common_gate_ledger_v1"
    out.mkdir(exist_ok=False)
    q = WORKTREE / "results/phase1/compose_lipid_iclr22_research_v1"
    evidence = q / "evidence_completion_v1/quality/v2"
    paths = {
        "original_gates": q / "quality/all22_original_reselection/gates.json",
        "current_selection": q / "quality/all22_context_preserving/result.json",
        "candidate_ledger": WORKTREE
        / "results/phase1/compose_lipid_quality_confirmation_v1/assessment/d1_candidate_diagnostics.json",
        "tree_result": evidence / "result.json",
        "tree_verification": evidence / "verification.json",
        "tree_protocol": evidence / "protocol.json",
        "producer": Path(__file__),
    }
    inputs = {k: pin(v) for k, v in paths.items()}
    write(
        out / "protocol.json",
        {
            "inputs": inputs,
            "rule": "Replace only the15 exact request/ordinal/canonical-SMILES-bound admitted tree assessments; leave all other bytes semantically unchanged.",
            "assessment_policy": "Frozen chemical/head/ring predicates plus the existing identity-specific transported-tree admission; no new tree construction or basis admission.",
            "expected_current_design": 1324,
            "new_model_chemistry_TEST_calls": 0,
        },
    )
    verified = {}

    def authenticate(record):
        p = WORKTREE / record["path"]
        assert digest(p) == record["sha256"], record["path"]
        verified[record["path"]] = record["sha256"]

    source, validation, policy = (
        read(paths["tree_result"]),
        read(paths["tree_verification"]),
        read(paths["tree_protocol"]),
    )
    assert validation["passed"] and validation["target_certificates"] == source["admitted"] == 15
    assert validation["altered_tree_or_closure_membership_allowed"] is False
    assert validation["inputs"]["result"]["sha256"] == digest(paths["tree_result"])
    for record in validation["inputs"].values():
        authenticate(record)
    for record in policy["inputs"].values():
        authenticate(record)
    for record in source["full_shard_pins"].values():
        authenticate(record)
    gates = read(paths["original_gates"])
    updated = copy.deepcopy(gates)
    lookup = {(r["index"], r["ordinal"]): r for r in updated["attempts"]}
    assert len(lookup) == len(updated["attempts"])
    candidates = {
        (r["index"], r["ordinal"]): r
        for r in read(paths["candidate_ledger"])["candidates"]
        if r["status"] == "assessed"
    }
    current = read(paths["current_selection"])["by_family"]
    chosen = {
        (c["request"], c["ordinal"]): (family, c)
        for family, v in current.items()
        for c in v["selections"]
    }
    changes = []
    for target in source["targets"]:
        key = target["index"], target["ordinal"]
        family, c = chosen[key]
        assert family == target["family"] == candidates[key]["family"]
        assert c == candidates[key]["candidate"]
        assert candidates[key]["kind"] == target["kind"]
        authenticate(target["source"])
        before = lookup[key]["assessment"]
        after = joined_assessment(target, c, before, policy["basis_admission"])
        changes.append(
            {
                "index": key[0],
                "ordinal": key[1],
                "family": family,
                "smiles": c["smiles"],
                "original_assessment_sha256": object_hash(before),
                "new_assessment_sha256": object_hash(after),
                "source": target["source"],
                "tree_basis": policy["basis_admission"],
                "old_design": before["qualified_design_pass"],
                "new_design": after["qualified_design_pass"],
            }
        )
        lookup[key]["assessment"] = after
    changed = {(x["index"], x["ordinal"]) for x in changes}
    assert len(changed) == 15
    for original in gates["attempts"]:
        key = original["index"], original["ordinal"]
        if key not in changed:
            assert lookup[key] == original
    design = sum(lookup[key]["assessment"]["qualified_design_pass"] for key in chosen)
    assert design == 1324 and len(chosen) == 1408
    updated["identity_specific_augmentation"] = {
        "protocol": pin(out / "protocol.json"),
        "replacements": changes,
    }
    write(out / "gates.json", updated)
    write(
        out / "verification.json",
        {
            "passed": True,
            "protocol": pin(out / "protocol.json"),
            "output": pin(out / "gates.json"),
            "candidate_entries": len(candidates),
            "original_gate_entries": len(gates["attempts"]),
            "replacement_count": 15,
            "all_other_assessments_unchanged": True,
            "current_requests": 1408,
            "current_design": design,
            "current_context": sum(
                any(
                    f["tier"] == "context_required"
                    for f in lookup[key]["assessment"]["chemical"]["flags"]
                )
                for key in chosen
            ),
            "authenticated_prior_evidence": verified,
            "replacement_receipts": changes,
            "no_general_basis_admission": True,
            "new_model_chemistry_TEST_calls": 0,
        },
    )
    print(
        json.dumps(
            {"gates": pin(out / "gates.json"), "verification": pin(out / "verification.json")}
        )
    )


if __name__ == "__main__":
    main()
