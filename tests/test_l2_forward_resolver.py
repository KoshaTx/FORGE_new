from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from rdkit.Chem import rdChemReactions

from forge.route.l2_forward_resolver import (
    AdmittedL2ForwardTransform,
    ExactPairAdmission,
    IndependentL2ForwardResolver,
    L2ForwardResolutionStatus,
    L2ForwardResolverError,
    load_independent_l2_forward_resolver,
)
from forge.route.l2_forward_resolver_manifest import build_l2_forward_resolver_config
from forge.route.planner import ForwardVerificationState, RouteTarget
from forge.route.proposal_engine import (
    ProposalBackendManifest,
    ProposalRequest,
    ProposalTargetKind,
    RootQualificationReceipt,
    SingleStepRetrosynthesisProposal,
)
from forge.route.qualified_forward import QualifiedForwardError, QualifiedForwardReaction
from forge.route.ugi3_support_boundary import MolecularSupportState, TargetQualification

REPO = Path(__file__).resolve().parents[1]
REAL_CONFIG = REPO / "configs/route/graph2edits_l2_forward_resolver_v1.json"


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _proposal(
    reactants: tuple[str, ...],
    *,
    target: str = "CC=O",
    score: float | None = 0.9,
    reaction_class: str | None = "model_claim_must_be_ignored",
) -> SingleStepRetrosynthesisProposal:
    route_root = RouteTarget(
        role="oxoester_aldehyde_body_tail",
        canonical_smiles=target,
    )
    qualification = RootQualificationReceipt(
        route_root=route_root,
        qualification=TargetQualification(
            exact_l1_eligible=True,
            supported_ugi_role=True,
            role_handle_qualified=True,
            molecular_support_state=MolecularSupportState.WITHIN_DECLARED_SUPPORT,
        ),
        terminal_sha256=_hash("terminal"),
        generator_checkpoint_sha256=_hash("generator"),
        l1_reaction_sha256=_hash("l1"),
        qualification_artifact_sha256=_hash("qualification"),
    )
    request = ProposalRequest(
        route_root=route_root,
        target=route_root,
        depth=0,
        target_kind=ProposalTargetKind.ROOT,
        root_qualification=qualification,
        operational_policy_id="test-policy",
        operational_policy_sha256=_hash("test-policy"),
    )
    backend = ProposalBackendManifest(
        backend_id="test-proposal-only-backend",
        implementation_version="1",
        checkpoint_sha256=_hash("checkpoint"),
        checkpoint_license="test-only",
        training_corpus_id="test-corpus",
        training_corpus_snapshot_sha256=_hash("corpus"),
        source_locator="test fixture",
    )
    return SingleStepRetrosynthesisProposal(
        proposal_id="proposal-1",
        backend=backend,
        request=request,
        reactant_smiles=reactants,
        rank=1,
        model_score=score,
        predicted_reaction_class=reaction_class,
    )


def _transform(
    reaction_id: str,
    *,
    role_names: tuple[str, ...],
    reactants: tuple[str, ...],
    target: str,
    smarts: str = "[CH2:1][OH:2]>>[CH:1]=[O:2]",
    status: str = "qualified_for_exact_source_forward_verification_only",
) -> AdmittedL2ForwardTransform:
    reaction = rdChemReactions.ReactionFromSmarts(smarts)
    assert reaction is not None
    compiled = QualifiedForwardReaction(
        reaction_id=reaction_id,
        role_names=role_names,
        reaction=reaction,
    )
    return AdmittedL2ForwardTransform(
        reaction=compiled,
        transform_sha256=_hash(f"transform:{reaction_id}"),
        registry_id="test-upstream-l2-registry",
        registry_sha256=_hash("registry"),
        qualification_status=status,
        admissions=(
            ExactPairAdmission(
                admission_id=f"admission:{reaction_id}",
                role_ordered_reactants=reactants,
                target_smiles=target,
                source_record_sha256s=(_hash(f"source:{reaction_id}"),),
            ),
        ),
    )


class RecordingExecutor:
    def __init__(self, outputs: dict[str, tuple[str, ...] | Exception]) -> None:
        self.outputs = outputs
        self.calls: list[tuple[str, tuple[str, ...], int]] = []

    def __call__(
        self,
        reaction: QualifiedForwardReaction,
        reactants: tuple[str, ...],
        *,
        max_products: int,
    ) -> tuple[str, ...]:
        self.calls.append((reaction.reaction_id, reactants, max_products))
        value = self.outputs[reaction.reaction_id]
        if isinstance(value, Exception):
            raise value
        return value


def _resolver(
    transforms: tuple[AdmittedL2ForwardTransform, ...],
    executor: RecordingExecutor,
    *,
    maximum_forward_calls: int = 50,
) -> IndependentL2ForwardResolver:
    return IndependentL2ForwardResolver(
        resolver_id="independent-upstream-l2-test",
        resolver_version="1",
        resolver_config_sha256=_hash("resolver-config"),
        transforms=transforms,
        maximum_forward_calls=maximum_forward_calls,
        maximum_products_per_assignment=64,
        executor=executor,
    )


def test_exact_pair_resolution_enumerates_role_assignments_and_has_no_evidence_authority() -> None:
    transform = _transform(
        "test_esterification_l2",
        role_names=("acid", "alcohol"),
        reactants=("CC(=O)O", "CO"),
        target="COC(C)=O",
        smarts="[C:1](=[O:2])[OH:3].[OH:4][CH3:5]>>[C:1](=[O:2])[O:4][CH3:5]",
    )
    executor = RecordingExecutor({transform.reaction_id: ("COC(C)=O",)})
    resolver = _resolver((transform,), executor)

    result = resolver.resolve(_proposal(("CO", "CC(=O)O"), target="COC(C)=O"))

    assert result.status is L2ForwardResolutionStatus.EXACT_UNIQUE
    assert result.forward_verification is ForwardVerificationState.VERIFIED_EXACT_PRODUCT_UNIQUE
    assert result.eligible_assignments == result.forward_calls == 1
    assert executor.calls == [("test_esterification_l2", ("CC(=O)O", "CO"), 64)]
    payload = result.to_dict()
    assert payload["proposal_model_score_used"] is False
    assert payload["proposal_model_class_used"] is False
    assert payload["substrate_scope_state"] == "unassessed"
    assert payload["evidence_tier"] is None
    assert payload["success_probability"] is None
    assert payload["route_closure_authorized"] is False
    assert payload["may_enter_synthesis_value"] is False
    assert payload["accepted_assignment"]["scope_only_not_route_evidence"] is True


def test_model_score_and_class_cannot_select_or_change_the_verifier() -> None:
    transform = _transform(
        "test_oxidation_l2",
        role_names=("alcohol",),
        reactants=("CCO",),
        target="CC=O",
    )
    executor = RecordingExecutor({transform.reaction_id: ("CC=O",)})
    resolver = _resolver((transform,), executor)
    first = resolver.resolve(_proposal(("CCO",), score=1000.0, reaction_class="oxidation"))
    second = resolver.resolve(_proposal(("CCO",), score=-1000.0, reaction_class="reduction"))

    assert first.status is second.status is L2ForwardResolutionStatus.EXACT_UNIQUE
    assert first.accepted_trace is not None and second.accepted_trace is not None
    assert first.accepted_trace.reaction_id == second.accepted_trace.reaction_id
    assert first.accepted_trace.role_ordered_reactants == (
        second.accepted_trace.role_ordered_reactants
    )
    assert len(executor.calls) == 2


def test_no_exact_scope_admission_censors_without_running_a_transform() -> None:
    transform = _transform(
        "test_oxidation_l2",
        role_names=("alcohol",),
        reactants=("CCO",),
        target="CC=O",
    )
    executor = RecordingExecutor({transform.reaction_id: ("CC=O",)})
    result = _resolver((transform,), executor).resolve(_proposal(("CCCO",), target="CCC=O"))

    assert result.status is L2ForwardResolutionStatus.CENSOR_NO_VERIFIER
    assert result.forward_verification is ForwardVerificationState.NOT_RUN
    assert result.eligible_assignments == result.forward_calls == 0
    assert executor.calls == []


def test_zero_reconstruction_rejects_as_forward_mismatch() -> None:
    transform = _transform(
        "test_oxidation_l2",
        role_names=("alcohol",),
        reactants=("CCO",),
        target="CC=O",
    )
    executor = RecordingExecutor({transform.reaction_id: ("CCC=O",)})
    result = _resolver((transform,), executor).resolve(_proposal(("CCO",)))

    assert result.status is L2ForwardResolutionStatus.REJECT_FORWARD_MISMATCH
    assert result.forward_verification is ForwardVerificationState.MISMATCHED


def test_multiple_products_from_one_assignment_censor_as_ambiguous() -> None:
    transform = _transform(
        "test_oxidation_l2",
        role_names=("alcohol",),
        reactants=("CCO",),
        target="CC=O",
    )
    executor = RecordingExecutor({transform.reaction_id: ("CC=O", "CO")})
    result = _resolver((transform,), executor).resolve(_proposal(("CCO",)))

    assert result.status is L2ForwardResolutionStatus.CENSOR_AMBIGUOUS_FORWARD_PRODUCTS
    assert result.forward_verification is ForwardVerificationState.AMBIGUOUS
    assert result.accepted_trace is None


def test_multiple_matching_transforms_censor_as_assignment_ambiguity() -> None:
    first = _transform(
        "test_oxidation_l2_a",
        role_names=("alcohol",),
        reactants=("CCO",),
        target="CC=O",
    )
    second = _transform(
        "test_oxidation_l2_b",
        role_names=("alcohol",),
        reactants=("CCO",),
        target="CC=O",
    )
    executor = RecordingExecutor({first.reaction_id: ("CC=O",), second.reaction_id: ("CC=O",)})
    result = _resolver((first, second), executor).resolve(_proposal(("CCO",)))

    assert result.status is L2ForwardResolutionStatus.CENSOR_AMBIGUOUS_TRANSFORM_ASSIGNMENT
    assert result.forward_verification is ForwardVerificationState.AMBIGUOUS
    assert result.forward_calls == 2
    assert result.accepted_trace is None


def test_budget_exhaustion_is_decided_before_any_partial_execution() -> None:
    first = _transform(
        "test_oxidation_l2_a",
        role_names=("alcohol",),
        reactants=("CCO",),
        target="CC=O",
    )
    second = _transform(
        "test_oxidation_l2_b",
        role_names=("alcohol",),
        reactants=("CCO",),
        target="CC=O",
    )
    executor = RecordingExecutor({first.reaction_id: ("CC=O",), second.reaction_id: ("CC=O",)})
    result = _resolver(
        (first, second),
        executor,
        maximum_forward_calls=1,
    ).resolve(_proposal(("CCO",)))

    assert result.status is L2ForwardResolutionStatus.CENSOR_BUDGET_EXHAUSTED
    assert result.forward_verification is ForwardVerificationState.NOT_RUN
    assert result.eligible_assignments == 2
    assert result.forward_calls == 0
    assert executor.calls == []


def test_executor_error_fails_closed_and_counts_the_attempted_call() -> None:
    transform = _transform(
        "test_oxidation_l2",
        role_names=("alcohol",),
        reactants=("CCO",),
        target="CC=O",
    )
    executor = RecordingExecutor(
        {transform.reaction_id: QualifiedForwardError("test execution failure")}
    )
    result = _resolver((transform,), executor).resolve(_proposal(("CCO",)))

    assert result.status is L2ForwardResolutionStatus.CENSOR_EXECUTION_ERROR
    assert result.forward_verification is ForwardVerificationState.NOT_RUN
    assert result.forward_calls == 1
    assert result.traces == ()


def test_general_enumeration_or_l1_status_is_rejected_at_construction() -> None:
    with pytest.raises(L2ForwardResolverError, match="exact-source or exact-pair"):
        _transform(
            "ugi_3cr_agile",
            role_names=("amine", "aldehyde", "isocyanide"),
            reactants=("CN", "CC=O", "[C-]#[N+]C"),
            target="CC",
            status="qualified_for_enumeration",
        )


def test_resolution_payload_does_not_gain_authority_when_proposal_metadata_changes() -> None:
    transform = _transform(
        "test_oxidation_l2",
        role_names=("alcohol",),
        reactants=("CCO",),
        target="CC=O",
    )
    executor = RecordingExecutor({transform.reaction_id: ("CC=O",)})
    proposal = _proposal(("CCO",))
    altered = replace(proposal, model_score=0.000001, predicted_reaction_class="anything")
    first = _resolver((transform,), executor).resolve(proposal).to_dict()
    second = _resolver((transform,), executor).resolve(altered).to_dict()
    for payload in (first, second):
        assert "model_score" not in payload
        assert payload["proposal_model_score_used"] is False
        assert payload["may_enter_synthesis_value"] is False


def test_real_development_manifest_is_deterministic_l2_only_and_not_benchmarked() -> None:
    stored = json.loads(REAL_CONFIG.read_text())
    assert build_l2_forward_resolver_config(REPO) == stored
    assert stored["status"] == "frozen_inactive_not_benchmarked"
    assert stored["activation_authorized"] is False
    assert stored["scope_guards"]["ugi_l1_registry_admitted"] is False
    assert stored["scope_guards"]["sealed_holdouts_accessed"] is False
    assert stored["scope_guards"]["frozen_benchmark_executed"] is False
    assert stored["summary"] == {
        "transform_count": 11,
        "exact_pair_admission_count": 56,
        "registry_count": 3,
    }
    assert "ugi_3cr_agile" not in {record["reaction_id"] for record in stored["transforms"]}


def test_real_resolver_replays_one_development_exact_pair_without_graph2edits() -> None:
    resolver = load_independent_l2_forward_resolver(REAL_CONFIG, repo_root=REPO)
    reactant = "C#CCCCCCCCCC(=O)OCCCCCCO"
    target = "C#CCCCCCCCCC(=O)OCCCCCC=O"

    result = resolver.resolve(_proposal((reactant,), target=target))

    assert result.status is L2ForwardResolutionStatus.EXACT_UNIQUE
    assert result.accepted_trace is not None
    assert result.accepted_trace.reaction_id == (
        "ugi3_upstream_primary_alcohol_oxidation_exact_source_v1"
    )
    assert result.accepted_trace.products == (target,)
    assert result.to_dict()["substrate_scope_state"] == "unassessed"
