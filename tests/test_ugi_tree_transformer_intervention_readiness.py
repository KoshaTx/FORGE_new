from __future__ import annotations

from experiments.phase1.product_l1.evaluation.ugi_tree_transformer_intervention_readiness import (  # noqa: E501
    classify_transform_consistent_candidates,
)
from forge.assembly import Ugi3RoleHandleAssessment, Ugi3TransformConsistentCandidate


def _candidate(*, qualified: bool) -> Ugi3TransformConsistentCandidate:
    components = (
        ("amine_head", "CN"),
        ("oxoester_aldehyde_body_tail", "CC=O"),
        ("isocyanide_tail", "[C-]#[N+]C"),
    )
    assessments = tuple(
        Ugi3RoleHandleAssessment(
            role=role,
            raw_handle_matches=1,
            symmetry_distinct_handle_sites=1,
            forbidden_substructure_match=False,
            passes_registry_handle_policy=qualified,
        )
        for role, _ in components
    )
    return Ugi3TransformConsistentCandidate(
        product_smiles="CC",
        components=components,
        handle_assessments=assessments,
    )


def test_transform_consistent_classification_preserves_handle_gate() -> None:
    candidate = _candidate(qualified=False)

    label, matched = classify_transform_consistent_candidates(
        (candidate,), candidate.components
    )

    assert label == "native_trace_transform_consistent_handle_rejected"
    assert matched is candidate
    assert matched.registry_handle_qualified is False


def test_transform_consistent_classification_separates_coverage_failures() -> None:
    candidate = _candidate(qualified=True)

    empty_label, empty = classify_transform_consistent_candidates((), candidate.components)
    mismatch_label, mismatch = classify_transform_consistent_candidates(
        (candidate,), (("amine_head", "NN"), *candidate.components[1:])
    )

    assert empty_label == "no_transform_consistent_candidate"
    assert empty is None
    assert mismatch_label == "transform_consistent_candidate_native_trace_not_recovered"
    assert mismatch is None
