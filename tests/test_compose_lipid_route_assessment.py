"""Adversarial checks for the all-family dossier aggregation boundary."""

from dataclasses import replace

import pytest

from forge.synthesis.engine.planner import (
    AssessmentOutcome,
    AvailabilityState,
    EvidenceRecord,
    EvidenceTier,
    ForwardVerificationState,
    RouteTarget,
    SynthesisAssessment,
    SynthesisRouteNode,
)
from forge.synthesis.terminals.compose_lipid_assessment import (
    ComponentRequirement,
    ScopedRouteReceipt,
    ValidatedComposeAssembly,
    aggregate_compose_routes,
)


def _assembly(n=4):
    # Roles are deliberately synthetic: this tests aggregation, not reaction chemistry.
    requirements = tuple(
        ComponentRequirement(f"role_{i}", "C" * (i + 1), 2 if i == 0 else 1, ("s1", "s2"))
        for i in range(n)
    )
    components = {item.role: item.canonical_smiles for item in requirements}
    check = {
        "exact": True,
        "status": "evaluated",
        "checks": [
            {
                "accepted_components": [components],
                "candidates": [
                    {
                        "components": components,
                        "pass": True,
                        "replay": {
                            "computed_consistency_pass": True,
                            "forward_layers": [["C"], ["CC"]],
                            "declared_quantities": {
                                item.role: item.quantity for item in requirements
                            },
                        },
                    }
                ],
            }
        ],
    }
    return (
        ValidatedComposeAssembly.from_l1(
            family="test_program",
            product_smiles="CC",
            program_sha256="1" * 64,
            assessment=check,
            requirements=requirements,
        ),
        check,
    )


def _receipt(assembly, requirement, outcome=AssessmentOutcome.COMPLETE):
    target = RouteTarget(requirement.role, requirement.canonical_smiles)
    evidence = (
        EvidenceRecord(
            evidence_id="fixture",
            tier=EvidenceTier.ACCEPTED_TERMINAL,
            source_sha256="2" * 64,
            source_locator="fixture#exact",
            exact_substrate=True,
            forward_verification=ForwardVerificationState.NOT_APPLICABLE,
            availability=AvailabilityState.CURRENT_CLOSED,
        ),
    )
    assessment = SynthesisAssessment(
        target,
        outcome,
        SynthesisRouteNode(
            target, outcome, evidence if outcome is AssessmentOutcome.COMPLETE else ()
        ),
        (),
    )
    return ScopedRouteReceipt(
        assembly.context_sha256,
        "3" * 64,
        requirement,
        assessment,
        "4" * 64,
        "US",
        "2026-09-01T00:00:00Z",
        "2026-10-01T00:00:00Z",
    )


def _aggregate(assembly, receipts, as_of="2026-09-24T00:00:00Z"):
    return aggregate_compose_routes(assembly, receipts, assessment_at_utc=as_of, region="US")


@pytest.mark.parametrize("roles", [2, 3, 4, 5])
def test_complete_requires_every_variable_role_and_preserves_repeats(roles):
    assembly, _ = _assembly(roles)
    receipts = tuple(_receipt(assembly, item) for item in assembly.requirements)
    result = _aggregate(assembly, receipts)
    assert result["complete_exact_source_dossier"]
    assert result["branches"][0]["quantity"] == 2
    assert result["branches"][0]["stage_ids"] == ("s1", "s2")
    partial = _aggregate(assembly, receipts[:-1])
    assert not partial["complete_exact_source_dossier"]
    assert partial["branches"][-1]["status"] == "not_assessed"
    assert not partial["all_branches_assessed"]


def test_ambiguous_or_partial_l1_never_enters_route_aggregation():
    assembly, check = _assembly()
    check["checks"][0]["accepted_components"].append({"extra_role": "C"})
    with pytest.raises(ValueError, match="one accepted precursor tuple"):
        ValidatedComposeAssembly.from_l1(
            family=assembly.family,
            product_smiles="CC",
            program_sha256=assembly.program_sha256,
            assessment=check,
            requirements=assembly.requirements,
        )
    assembly, check = _assembly()
    with pytest.raises(ValueError, match="complete required roles"):
        ValidatedComposeAssembly.from_l1(
            family=assembly.family,
            product_smiles="CC",
            program_sha256=assembly.program_sha256,
            assessment=check,
            requirements=assembly.requirements[:-1],
        )


def test_replay_for_another_product_does_not_validate_l1():
    assembly, check = _assembly()
    with pytest.raises(ValueError, match="exact-product replay witness"):
        ValidatedComposeAssembly.from_l1(
            family=assembly.family,
            product_smiles="CCC",
            program_sha256=assembly.program_sha256,
            assessment=check,
            requirements=assembly.requirements,
        )


def test_consistency_boolean_without_a_product_witness_fails_closed():
    assembly, check = _assembly()
    replay = check["checks"][0]["candidates"][0]["replay"]
    del replay["forward_layers"]
    replay["checks"] = {"unique_forward_exact": True}
    with pytest.raises(ValueError, match="exact-product replay witness"):
        ValidatedComposeAssembly.from_l1(
            family=assembly.family,
            product_smiles="CC",
            program_sha256=assembly.program_sha256,
            assessment=check,
            requirements=assembly.requirements,
        )


@pytest.mark.parametrize("field,value", [("quantity", 3), ("stage_ids", ("s3",))])
def test_same_smiles_different_occurrence_context_cannot_inherit_receipt(field, value):
    assembly, _ = _assembly()
    requirement = replace(assembly.requirements[0], **{field: value})
    receipt = _receipt(assembly, requirement)
    with pytest.raises(ValueError, match="stage context mismatch"):
        _aggregate(assembly, (receipt,))


def test_same_smiles_wrong_role_or_assembly_context_cannot_close():
    assembly, _ = _assembly()
    receipt = _receipt(assembly, assembly.requirements[0])
    with pytest.raises(ValueError, match="role/identity"):
        replace(receipt, requirement=replace(receipt.requirement, role="other_role"))
    with pytest.raises(ValueError, match="assembly context"):
        _aggregate(assembly, (replace(receipt, assembly_context_sha256="9" * 64),))


def test_expiry_boundary_blocks_complete_even_with_old_complete_tree():
    assembly, _ = _assembly()
    receipts = tuple(_receipt(assembly, item) for item in assembly.requirements)
    result = _aggregate(assembly, receipts, "2026-10-01T00:00:00Z")
    assert not result["complete_exact_source_dossier"]
    assert all(branch["status"] == "expired_l3_snapshot" for branch in result["branches"])


def test_censoring_and_missing_evidence_remain_distinct():
    assembly, _ = _assembly(2)
    receipts = tuple(
        _receipt(assembly, item, outcome)
        for item, outcome in zip(
            assembly.requirements,
            (AssessmentOutcome.BUDGET_EXHAUSTED, AssessmentOutcome.MISSING_KNOWLEDGE),
            strict=True,
        )
    )
    result = _aggregate(assembly, receipts)
    assert [row["status"] for row in result["branches"]] == [
        "budget_exhausted",
        "missing_knowledge",
    ]
    assert not result["complete_exact_source_dossier"]


def test_forged_complete_leaf_with_inexact_identity_is_rejected():
    assembly, _ = _assembly()
    receipt = _receipt(assembly, assembly.requirements[0])
    tree = receipt.assessment.route_tree
    bad_tree = replace(tree, evidence=(replace(tree.evidence[0], exact_substrate=False),))
    receipt = replace(receipt, assessment=replace(receipt.assessment, route_tree=bad_tree))
    with pytest.raises(ValueError, match="exact current terminal"):
        _aggregate(assembly, (receipt,))
