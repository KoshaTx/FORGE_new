from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from forge.route.planner import (
    ForwardVerificationState,
    RoutePlannerError,
    RouteStepProposal,
    RouteTarget,
)
from forge.route.proposal_engine import (
    OperationalCompatibility,
    ProposalBackendManifest,
    ProposalDisposition,
    ProposalEngineError,
    ProposalReason,
    ProposalRequest,
    ProposalScreen,
    ProposalTargetKind,
    RootQualificationReceipt,
    SingleStepRetrosynthesisProposal,
    SubstrateScopeState,
    build_internal_target_lineage,
    screen_learned_proposal,
    validate_proposal_batch,
)
from forge.route.ugi3_support_boundary import (
    AuthenticatedInternalRoleRegistry,
    MolecularSupportState,
    TargetQualification,
)

POLICY_ID = "forge-lipid-component-operational-v1"


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _backend() -> ProposalBackendManifest:
    return ProposalBackendManifest(
        backend_id="retrochimera-proposal-only",
        implementation_version="1.2.0",
        checkpoint_sha256=_hash("checkpoint"),
        checkpoint_license="checkpoint-license-pending-audit",
        training_corpus_id="frozen-test-corpus",
        training_corpus_snapshot_sha256=_hash("training-corpus"),
        source_locator="https://arxiv.org/abs/2412.05269",
    )


def _root_receipt(root: RouteTarget) -> RootQualificationReceipt:
    return RootQualificationReceipt(
        route_root=root,
        qualification=TargetQualification(
            exact_l1_eligible=True,
            supported_ugi_role=True,
            role_handle_qualified=True,
            molecular_support_state=MolecularSupportState.WITHIN_DECLARED_SUPPORT,
        ),
        terminal_sha256=_hash("terminal"),
        generator_checkpoint_sha256=_hash("generator"),
        l1_reaction_sha256=_hash("l1-reaction"),
        qualification_artifact_sha256=_hash("root-qualification"),
    )


def _root_request(*, role: str = "oxoester_aldehyde_body_tail") -> ProposalRequest:
    root = RouteTarget(role=role, canonical_smiles="CCCC=O")
    return ProposalRequest(
        route_root=root,
        target=root,
        depth=0,
        target_kind=ProposalTargetKind.ROOT,
        root_qualification=_root_receipt(root),
        operational_policy_id=POLICY_ID,
        operational_policy_sha256=_hash("operational-policy"),
    )


def _proposal(
    *,
    request: ProposalRequest | None = None,
    score: float | None = 0.97,
    reactants: tuple[str, ...] = ("OCCCC",),
    proposal_id: str = "retro:test:1",
    rank: int = 1,
) -> SingleStepRetrosynthesisProposal:
    return SingleStepRetrosynthesisProposal(
        proposal_id=proposal_id,
        backend=_backend(),
        request=_root_request() if request is None else request,
        reactant_smiles=reactants,
        rank=rank,
        model_score=score,
        predicted_reaction_class="alcohol oxidation",
    )


def _screen(
    proposal: SingleStepRetrosynthesisProposal,
    *,
    proposal_sha256: str | None = None,
    forward: ForwardVerificationState = (ForwardVerificationState.VERIFIED_EXACT_PRODUCT_UNIQUE),
    products: tuple[str, ...] | None = None,
    operational: OperationalCompatibility = OperationalCompatibility.PASSED,
    scope: SubstrateScopeState = SubstrateScopeState.GENERAL_ANALOGUE,
    operational_policy_id: str | None = None,
    operational_policy_sha256: str | None = None,
) -> ProposalScreen:
    if products is None:
        if forward is ForwardVerificationState.VERIFIED_EXACT_PRODUCT_UNIQUE:
            products = (proposal.request.target.canonical_smiles,)
        elif forward is ForwardVerificationState.AMBIGUOUS:
            products = ("CC", "CCC")
        else:
            products = ()
    return ProposalScreen(
        proposal_sha256=(proposal.proposal_sha256 if proposal_sha256 is None else proposal_sha256),
        request_target_identity=proposal.request.target.identity,
        verifier_id="rdkit-qualified-forward",
        verifier_version="1.0.0",
        forward_transform_sha256=_hash("forward-transform"),
        forward_verification=forward,
        forward_product_count=len(products),
        forward_product_smiles=products,
        operational_policy_id=(
            proposal.request.operational_policy_id
            if operational_policy_id is None
            else operational_policy_id
        ),
        operational_policy_sha256=(
            proposal.request.operational_policy_sha256
            if operational_policy_sha256 is None
            else operational_policy_sha256
        ),
        operational_compatibility=operational,
        substrate_scope=scope,
        substrate_scope_evidence_sha256=(
            _hash("exact-scope-evidence") if scope is SubstrateScopeState.EXACT_SUBSTRATE else None
        ),
    )


def test_general_organic_proposal_can_enter_search_but_never_close_route() -> None:
    proposal = _proposal()
    screen = _screen(proposal)
    decision = screen_learned_proposal(proposal, screen)

    assert decision.disposition is ProposalDisposition.ADMIT_TO_BOUNDED_SEARCH
    assert decision.can_enter_bounded_search is True
    assert decision.can_close_route is False
    assert decision.evidence_tier is None
    assert decision.success_probability is None
    assert decision.proposal_sha256 == proposal.proposal_sha256
    assert decision.screen_sha256 == screen.screen_sha256
    assert ProposalReason.FORWARD_EXACT_UNIQUE in decision.reasons
    assert ProposalReason.SUBSTRATE_SCOPE_REQUIRES_EVIDENCE in decision.reasons
    assert proposal.to_dict()["route_closure_authorized"] is False


@pytest.mark.parametrize(
    ("forward", "products", "expected_reason"),
    [
        (ForwardVerificationState.NOT_RUN, (), ProposalReason.FORWARD_NOT_RUN),
        (
            ForwardVerificationState.AMBIGUOUS,
            ("CC", "CCC"),
            ProposalReason.FORWARD_AMBIGUOUS,
        ),
        (ForwardVerificationState.MISMATCHED, (), ProposalReason.FORWARD_MISMATCH),
        (
            ForwardVerificationState.VERIFIED_EXACT_PRODUCT_UNIQUE,
            ("CC", "CCC"),
            ProposalReason.FORWARD_MISMATCH,
        ),
        (
            ForwardVerificationState.VERIFIED_EXACT_PRODUCT_UNIQUE,
            ("CC",),
            ProposalReason.FORWARD_MISMATCH,
        ),
    ],
)
def test_forward_failures_never_enter_search(
    forward: ForwardVerificationState,
    products: tuple[str, ...],
    expected_reason: ProposalReason,
) -> None:
    proposal = _proposal()
    decision = screen_learned_proposal(
        proposal,
        _screen(proposal, forward=forward, products=products),
    )

    assert decision.can_enter_bounded_search is False
    assert decision.can_close_route is False
    assert expected_reason in decision.reasons


def test_root_qualification_is_caller_owned_hash_bound_and_fail_closed() -> None:
    root = RouteTarget(role="oxoester_aldehyde_body_tail", canonical_smiles="CCCC=O")
    receipt = _root_receipt(root)

    assert (
        receipt.receipt_sha256
        == hashlib.sha256(
            json.dumps(
                receipt._content_dict(),  # noqa: SLF001 - exact content-addressing assertion
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
    )
    with pytest.raises(ProposalEngineError, match="not eligible"):
        RootQualificationReceipt(
            route_root=root,
            qualification=replace(receipt.qualification, exact_l1_eligible=False),
            terminal_sha256=receipt.terminal_sha256,
            generator_checkpoint_sha256=receipt.generator_checkpoint_sha256,
            l1_reaction_sha256=receipt.l1_reaction_sha256,
            qualification_artifact_sha256=receipt.qualification_artifact_sha256,
        )
    with pytest.raises(ProposalEngineError, match="handle"):
        RootQualificationReceipt(
            route_root=root,
            qualification=replace(receipt.qualification, role_handle_qualified=False),
            terminal_sha256=receipt.terminal_sha256,
            generator_checkpoint_sha256=receipt.generator_checkpoint_sha256,
            l1_reaction_sha256=receipt.l1_reaction_sha256,
            qualification_artifact_sha256=receipt.qualification_artifact_sha256,
        )


def test_proposal_and_screen_hashes_prevent_content_or_policy_substitution() -> None:
    proposal = _proposal()
    wrong_content = screen_learned_proposal(
        proposal,
        _screen(proposal, proposal_sha256=_hash("different-proposal")),
    )
    wrong_policy = screen_learned_proposal(
        proposal,
        _screen(proposal, operational_policy_sha256=_hash("different-policy")),
    )

    assert wrong_content.disposition is ProposalDisposition.CENSOR
    assert ProposalReason.PROPOSAL_CONTENT_MISMATCH in wrong_content.reasons
    assert wrong_policy.disposition is ProposalDisposition.CENSOR
    assert ProposalReason.SCREEN_POLICY_MISMATCH in wrong_policy.reasons


def test_review_required_is_censored_and_contradicted_scope_rejected() -> None:
    proposal = _proposal()
    review = screen_learned_proposal(
        proposal,
        _screen(proposal, operational=OperationalCompatibility.REVIEW_REQUIRED),
    )
    contradicted = screen_learned_proposal(
        proposal,
        _screen(proposal, scope=SubstrateScopeState.CONTRADICTED),
    )

    assert review.disposition is ProposalDisposition.CENSOR
    assert ProposalReason.OPERATIONAL_REVIEW_REQUIRED in review.reasons
    assert contradicted.disposition is ProposalDisposition.REJECT
    assert ProposalReason.SUBSTRATE_SCOPE_CONTRADICTED in contradicted.reasons


def test_exact_scope_requires_independent_evidence_hash_but_still_cannot_close() -> None:
    proposal = _proposal()
    with pytest.raises(ProposalEngineError, match="substrate_scope_evidence_sha256"):
        replace(
            _screen(proposal),
            substrate_scope=SubstrateScopeState.EXACT_SUBSTRATE,
        )

    decision = screen_learned_proposal(
        proposal,
        _screen(proposal, scope=SubstrateScopeState.EXACT_SUBSTRATE),
    )
    assert ProposalReason.SUBSTRATE_SCOPE_EXACT_EVIDENCE in decision.reasons
    assert decision.can_close_route is False
    assert decision.evidence_tier is None


def test_model_score_cannot_be_relabelled_as_probability_or_evidence() -> None:
    proposal = _proposal(score=0.999)
    low_score = _proposal(score=-1000.0)
    payload = proposal.to_dict()
    high_decision = screen_learned_proposal(proposal, _screen(proposal))
    low_decision = screen_learned_proposal(low_score, _screen(low_score))

    assert payload["model_score"] == pytest.approx(0.999)
    assert payload["success_probability"] is None
    assert payload["evidence_tier"] is None
    assert high_decision.disposition is low_decision.disposition
    assert high_decision.reasons == low_decision.reasons
    assert high_decision.can_close_route is low_decision.can_close_route is False
    assert high_decision.evidence_tier is low_decision.evidence_tier is None

    with pytest.raises(ProposalEngineError, match="finite or null"):
        _proposal(score=float("nan"))


def test_reactants_are_canonicalized_and_disconnected_inputs_rejected() -> None:
    proposal = _proposal(reactants=("OCCCC",))
    assert proposal.reactant_smiles == ("CCCCO",)

    with pytest.raises(ProposalEngineError, match="one connected"):
        _proposal(reactants=("CC.CCC",))
    with pytest.raises(ProposalEngineError, match="could not be parsed"):
        _proposal(reactants=("not-smiles",))


def test_internal_target_requires_lineage_from_admitted_parent() -> None:
    root = RouteTarget(role="isocyanide_tail", canonical_smiles="[C-]#[N+]CCCC")
    parent = _proposal(
        request=ProposalRequest(
            route_root=root,
            target=root,
            depth=0,
            target_kind=ProposalTargetKind.ROOT,
            root_qualification=_root_receipt(root),
            operational_policy_id=POLICY_ID,
            operational_policy_sha256=_hash("operational-policy"),
        ),
        reactants=("CCCNC=O",),
        proposal_id="retro:test:parent",
    )
    parent_screen = _screen(parent)
    parent_decision = screen_learned_proposal(parent, parent_screen)
    intermediate = RouteTarget(role="internal:formamide", canonical_smiles="CCCNC=O")
    roles = AuthenticatedInternalRoleRegistry(frozenset({"internal:formamide"}))
    lineage = build_internal_target_lineage(
        parent_proposal=parent,
        parent_screen=parent_screen,
        parent_decision=parent_decision,
        child_index=0,
        target=intermediate,
        authenticated_internal_roles=roles,
    )
    request = ProposalRequest(
        route_root=root,
        target=intermediate,
        depth=1,
        target_kind=ProposalTargetKind.INTERNAL,
        root_qualification=parent.request.root_qualification,
        operational_policy_id=POLICY_ID,
        operational_policy_sha256=_hash("operational-policy"),
        internal_lineage=lineage,
    )
    proposal = _proposal(
        request=request,
        reactants=("CCCN",),
        proposal_id="retro:test:internal",
    )
    decision = screen_learned_proposal(proposal, _screen(proposal))

    assert decision.disposition is ProposalDisposition.ADMIT_TO_BOUNDED_SEARCH
    assert decision.can_close_route is False
    with pytest.raises(ProposalEngineError, match="lineage"):
        replace(request, depth=2)
    with pytest.raises(ProposalEngineError, match="not admitted"):
        build_internal_target_lineage(
            parent_proposal=parent,
            parent_screen=parent_screen,
            parent_decision=replace(
                parent_decision,
                disposition=ProposalDisposition.CENSOR,
                can_enter_bounded_search=False,
            ),
            child_index=0,
            target=intermediate,
            authenticated_internal_roles=roles,
        )


def test_unevidenced_learned_step_cannot_close_even_with_terminal_reactant() -> None:
    proposal = _proposal(reactants=("CCO",))
    decision = screen_learned_proposal(proposal, _screen(proposal))
    terminal = RouteTarget(role="accepted_terminal", canonical_smiles="CCO")

    assert decision.can_enter_bounded_search is True
    assert decision.can_close_route is False
    with pytest.raises(RoutePlannerError, match="at least one evidence record"):
        RouteStepProposal(
            reaction_id="learned-proposal-only",
            reactants=(terminal,),
            evidence=(),
            forward_product_count=1,
        )


def test_batch_validation_binds_backend_request_rank_and_unique_reactants() -> None:
    first = _proposal()
    second = _proposal(
        request=first.request,
        reactants=("CCCBr",),
        proposal_id="retro:test:2",
        rank=2,
    )

    assert validate_proposal_batch(
        first.backend,
        first.request,
        (first, second),
        maximum_proposals=2,
    ) == (first, second)

    duplicate = _proposal(
        request=first.request,
        reactants=first.reactant_smiles,
        proposal_id="retro:test:duplicate",
        rank=2,
    )
    with pytest.raises(ProposalEngineError, match="reactant sets"):
        validate_proposal_batch(
            first.backend,
            first.request,
            (first, duplicate),
            maximum_proposals=2,
        )


def test_batch_validation_rejects_self_reported_request_or_rank_changes() -> None:
    proposal = _proposal()
    other_request = _root_request(role="amine_head")
    changed_request = _proposal(
        request=other_request,
        reactants=("CCBr",),
        proposal_id="retro:test:changed-request",
    )
    with pytest.raises(ProposalEngineError, match="caller-owned request"):
        validate_proposal_batch(
            proposal.backend,
            proposal.request,
            (changed_request,),
            maximum_proposals=1,
        )

    wrong_rank = _proposal(
        request=proposal.request,
        reactants=("CCCCl",),
        proposal_id="retro:test:wrong-rank",
        rank=2,
    )
    with pytest.raises(ProposalEngineError, match="contiguous"):
        validate_proposal_batch(
            proposal.backend,
            proposal.request,
            (wrong_rank,),
            maximum_proposals=1,
        )
