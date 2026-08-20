from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from forge.route.assessment.l2_forward_resolver import load_independent_l2_forward_resolver
from forge.route.assessment.ugi3_support_boundary import MolecularSupportState, TargetQualification
from forge.route.engine.planner import RouteTarget
from forge.route.engine.proposal_engine import (
    OperationalCompatibility,
    ProposalBackendManifest,
    ProposalRequest,
    ProposalTargetKind,
    RootQualificationReceipt,
    SingleStepRetrosynthesisProposal,
)
from forge.route.engine.single_step_proposal_benchmark import (
    EXPECTED_LANES,
    BenchmarkContractError,
    BenchmarkExecutionBlockedError,
    BenchmarkTarget,
    BenchmarkTargetManifest,
    FrozenBenchmarkContract,
    FrozenProposalLaneBenchmarkRunner,
    HiddenTruthManifest,
    HiddenTruthRecord,
    LaneProposalBatch,
    MatchedTargetBudget,
    load_frozen_benchmark_contract,
    score_frozen_lane_outputs,
)

REPO = Path(__file__).resolve().parents[1]
BINDING = REPO / "configs/route/single_step_proposal_benchmark_runner_binding_v1.json"
RESOLVER_CONFIG = REPO / "configs/route/graph2edits_l2_forward_resolver_v1.json"


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _request(target: str) -> ProposalRequest:
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
    return ProposalRequest(
        route_root=route_root,
        target=route_root,
        depth=0,
        target_kind=ProposalTargetKind.ROOT,
        root_qualification=qualification,
        operational_policy_id="benchmark-test-policy",
        operational_policy_sha256=_hash("benchmark-test-policy"),
    )


def _proposal(request: ProposalRequest, *, lane_id: str) -> SingleStepRetrosynthesisProposal:
    backend = ProposalBackendManifest(
        backend_id=f"fixture:{lane_id}",
        implementation_version="1",
        checkpoint_sha256=_hash(f"checkpoint:{lane_id}"),
        checkpoint_license="test-only",
        training_corpus_id="test-corpus",
        training_corpus_snapshot_sha256=_hash("corpus"),
        source_locator="test fixture",
    )
    return SingleStepRetrosynthesisProposal(
        proposal_id=f"proposal:{lane_id}",
        backend=backend,
        request=request,
        reactant_smiles=("C#CCCCCCCCCC(=O)OCCCCCCO",),
        rank=1,
        model_score=0.7,
        predicted_reaction_class="ignored_by_independent_resolver",
    )


class _Lane:
    def __init__(self, lane_id: str) -> None:
        self.lane_id = lane_id
        self.calls = 0

    def propose(
        self,
        target: BenchmarkTarget,
        *,
        budget: MatchedTargetBudget,
        strict_reserved_hypotheses: int,
        learned_reserved_hypotheses: int,
    ) -> LaneProposalBatch:
        self.calls += 1
        assert strict_reserved_hypotheses + learned_reserved_hypotheses == (
            budget.raw_hypothesis_attempts
        )
        return LaneProposalBatch(
            lane_id=self.lane_id,
            target_id=target.target_id,
            proposals=(_proposal(target.request, lane_id=self.lane_id),),
            proposal_sources=(self.lane_id,),
            proposal_source_calls=1,
            raw_hypothesis_attempts=1,
            invalid_raw_hypotheses=0,
            canonical_duplicates=0,
        )


def _engines() -> dict[str, _Lane]:
    return {lane_id: _Lane(lane_id) for lane_id in EXPECTED_LANES}


def _synthetic_contract(resolver_sha256: str) -> FrozenBenchmarkContract:
    return FrozenBenchmarkContract(
        spec_sha256=_hash("runner-binding"),
        policy_sha256=_hash("policy"),
        manifest_binding_sha256=_hash("manifest-binding"),
        lane_ids=EXPECTED_LANES,
        target_count=1,
        stratum_counts=(("known_exact_l2_routes", 1),),
        selection_seed=17,
        budget=MatchedTargetBudget(
            proposal_source_calls=1,
            raw_hypothesis_attempts=10,
            canonical_hypotheses_considered=10,
            forward_verifier_calls=10,
            operational_screen_calls=10,
            accepted_proposal_cap=5,
            wall_seconds=30,
            cpu_threads=1,
            peak_host_memory_mebibytes=1024,
            peak_accelerator_memory_mebibytes=1024,
        ),
        strict_reserved_hypotheses=5,
        learned_reserved_hypotheses=5,
        activation={"benchmark_execution_authorized": True},
        prerequisite_gates={"all_gates_passed": True},
        target_manifest_frozen=True,
        target_manifest_sha256=_hash("target-manifest"),
        hidden_truth_manifest_sha256=_hash("hidden-truth"),
        resolver_config_sha256=resolver_sha256,
    )


def test_bound_production_contract_authenticates_but_remains_nonexecuting() -> None:
    contract = load_frozen_benchmark_contract(BINDING, repo_root=REPO)

    assert contract.target_count == 120
    assert contract.target_manifest_frozen is True
    assert contract.execution_ready is False
    assert "benchmark_execution_not_authorized" in contract.blocking_reasons
    assert any(
        "institutional_license_review_complete" in value for value in contract.blocking_reasons
    )


def test_preflight_blocks_before_target_loader_or_lane_engine_is_called() -> None:
    contract = load_frozen_benchmark_contract(BINDING, repo_root=REPO)
    resolver = load_independent_l2_forward_resolver(RESOLVER_CONFIG, repo_root=REPO)
    engines = _engines()
    target_loader_called = False

    def target_loader(_path: Path) -> BenchmarkTargetManifest:
        nonlocal target_loader_called
        target_loader_called = True
        raise AssertionError("blocked preflight must not load target content")

    runner = FrozenProposalLaneBenchmarkRunner(
        contract=contract,
        resolver=resolver,
        lane_engines=engines,
        operational_screen=lambda _proposal, _resolution: OperationalCompatibility.PASSED,
    )
    with pytest.raises(BenchmarkExecutionBlockedError, match="before target access"):
        runner.execute_from_paths(
            target_manifest_path=Path("must-not-be-read.json.gz"),
            target_loader=target_loader,
        )

    assert target_loader_called is False
    assert all(engine.calls == 0 for engine in engines.values())


def test_runner_rejects_a_resolver_not_bound_to_the_contract() -> None:
    resolver = load_independent_l2_forward_resolver(RESOLVER_CONFIG, repo_root=REPO)
    contract = _synthetic_contract(_hash("different-resolver"))

    with pytest.raises(BenchmarkContractError, match="resolver hash"):
        FrozenProposalLaneBenchmarkRunner(
            contract=contract,
            resolver=resolver,
            lane_engines=_engines(),
            operational_screen=lambda _proposal, _resolution: OperationalCompatibility.PASSED,
        )


def test_synthetic_lanes_share_budgets_resolve_independently_and_score_after_freeze() -> None:
    resolver = load_independent_l2_forward_resolver(RESOLVER_CONFIG, repo_root=REPO)
    contract = _synthetic_contract(resolver.resolver_config_sha256)
    request = _request("C#CCCCCCCCCC(=O)OCCCCCC=O")
    targets = BenchmarkTargetManifest(
        manifest_sha256=contract.target_manifest_sha256,
        selection_seed=contract.selection_seed,
        targets=(
            BenchmarkTarget(
                target_id="target-1",
                primary_stratum="known_exact_l2_routes",
                request=request,
            ),
        ),
    )
    engines = _engines()
    runner = FrozenProposalLaneBenchmarkRunner(
        contract=contract,
        resolver=resolver,
        lane_engines=engines,
        operational_screen=lambda _proposal, _resolution: OperationalCompatibility.PASSED,
    )

    outputs = runner.execute(targets)

    assert len(outputs.outputs) == 3
    assert all(row.forward_calls == 1 for row in outputs.outputs)
    assert all(row.operational_screen_calls == 1 for row in outputs.outputs)
    assert all(row.accepted_count == 1 for row in outputs.outputs)
    assert outputs.to_dict()["hidden_truth_loaded_during_lane_execution"] is False
    hidden_truth = HiddenTruthManifest(
        manifest_sha256=contract.hidden_truth_manifest_sha256,
        records=(
            HiddenTruthRecord(
                target_id="target-1",
                truth_kind="documented_exact_forward_unique_reactant_multiset",
                exact_reactant_multisets=(("C#CCCCCCCCCC(=O)OCCCCCCO",),),
            ),
        ),
    )
    score = score_frozen_lane_outputs(
        contract=contract,
        targets=targets,
        outputs=outputs,
        hidden_truth=hidden_truth,
    )

    assert all(
        score["metrics"][lane_id]["known_route_top_k"]["1"]["fraction"] == 1.0
        for lane_id in EXPECTED_LANES
    )
    assert score["model_score_used_for_scoring"] is False
    assert score["route_evidence_created"] is False
    assert score["v_syn_modified"] is False


def test_hidden_truth_hash_or_output_grid_mutation_fails_closed() -> None:
    resolver = load_independent_l2_forward_resolver(RESOLVER_CONFIG, repo_root=REPO)
    contract = _synthetic_contract(resolver.resolver_config_sha256)
    target = BenchmarkTarget(
        target_id="target-1",
        primary_stratum="known_exact_l2_routes",
        request=_request("C#CCCCCCCCCC(=O)OCCCCCC=O"),
    )
    targets = BenchmarkTargetManifest(
        manifest_sha256=contract.target_manifest_sha256,
        selection_seed=17,
        targets=(target,),
    )
    outputs = FrozenProposalLaneBenchmarkRunner(
        contract=contract,
        resolver=resolver,
        lane_engines=_engines(),
        operational_screen=lambda _proposal, _resolution: OperationalCompatibility.PASSED,
    ).execute(targets)
    wrong_truth = HiddenTruthManifest(
        manifest_sha256=_hash("wrong-hidden-truth"),
        records=(
            HiddenTruthRecord(
                target_id="target-1",
                truth_kind="documented_exact_forward_unique_reactant_multiset",
                exact_reactant_multisets=(("C#CCCCCCCCCC(=O)OCCCCCCO",),),
            ),
        ),
    )
    with pytest.raises(BenchmarkContractError, match="frozen scoring artifact"):
        score_frozen_lane_outputs(
            contract=contract,
            targets=targets,
            outputs=outputs,
            hidden_truth=wrong_truth,
        )

    correct_truth = replace(wrong_truth, manifest_sha256=contract.hidden_truth_manifest_sha256)
    with pytest.raises(BenchmarkContractError, match="lane-target grid"):
        score_frozen_lane_outputs(
            contract=contract,
            targets=targets,
            outputs=replace(outputs, outputs=outputs.outputs[:-1]),
            hidden_truth=correct_truth,
        )
