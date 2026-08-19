from __future__ import annotations

from types import SimpleNamespace

from rdkit.Chem import rdChemReactions

from forge.route.l2_forward_resolver import L2ForwardResolutionStatus
from forge.route.proposal_discovery_status import (
    FamilyProjectionTrace,
    ProposalDiscoveryResolution,
    ProposalDiscoveryStatus,
)
from forge.route.qualified_forward import QualifiedForwardReaction
from forge.route.semantic_family_equivalence import (
    audit_semantic_family_equivalence,
    audit_semantic_family_equivalence_v2,
)


def _transform(reaction_id: str, smarts: str):
    return SimpleNamespace(
        reaction=QualifiedForwardReaction(
            reaction_id=reaction_id,
            role_names=("alcohol",),
            reaction=rdChemReactions.ReactionFromSmarts(smarts),
        ),
        reaction_id=reaction_id,
    )


def _resolution() -> ProposalDiscoveryResolution:
    traces = tuple(
        FamilyProjectionTrace(
            reaction_id=reaction_id,
            transform_sha256="a" * 64,
            role_ordered_reactants=("CCO",),
            products=("CC=O",),
            target_reconstructed=True,
        )
        for reaction_id in ("oxidation-a", "oxidation-b")
    )
    return ProposalDiscoveryResolution(
        proposal_sha256="b" * 64,
        target_smiles="CC=O",
        status=ProposalDiscoveryStatus.AMBIGUOUS,
        exact_resolution_status=L2ForwardResolutionStatus.CENSOR_NO_VERIFIER,
        exact_resolver_config_sha256="c" * 64,
        exact_forward_calls=0,
        family_projection_calls=2,
        maximum_family_projection_calls=10,
        traces=traces,
        rejection_reason="multiple_known_family_assignments_reconstruct_target",
    )


def test_duplicate_registry_variants_collapse_only_for_discovery() -> None:
    transforms = tuple(
        _transform(reaction_id, "[C&H2:1][O&H1:2]>>[C&H1:1]=[O:2]")
        for reaction_id in ("oxidation-a", "oxidation-b")
    )
    audit = audit_semantic_family_equivalence(_resolution(), transforms)
    assert audit["duplicate_provenance_only"] is True
    assert audit["semantic_resolution_status"] == (
        "semantically_equivalent_known_family_projection"
    )
    assert audit["graph_consistent_discovery_hypothesis"] is True
    assert audit["may_enter_synthesis_value"] is False


def test_distinct_reactions_remain_ambiguous() -> None:
    transforms = (
        _transform("oxidation-a", "[C&H2:1][O&H1:2]>>[C&H1:1]=[O:2]"),
        _transform("oxidation-b", "[C:1][O:2]>>[C:1]=[O:2]"),
    )
    audit = audit_semantic_family_equivalence(_resolution(), transforms)
    assert audit["duplicate_provenance_only"] is False
    assert audit["semantic_resolution_status"] == "ambiguous"


def test_v2_ignores_source_specific_role_names() -> None:
    first = _transform("oxidation-a", "[C&H2:1][O&H1:2]>>[C&H1:1]=[O:2]")
    second = _transform("oxidation-b", "[C&H2:1][O&H1:2]>>[C&H1:1]=[O:2]")
    second.reaction = QualifiedForwardReaction(
        reaction_id="oxidation-b",
        role_names=("exact_c18_alcohol",),
        reaction=second.reaction.reaction,
    )
    audit = audit_semantic_family_equivalence_v2(_resolution(), (first, second))
    assert audit["duplicate_provenance_only"] is True
    assert len(audit["equivalent_reaction_ids"]) == 2
