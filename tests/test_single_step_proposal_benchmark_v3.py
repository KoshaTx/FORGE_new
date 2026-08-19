from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from forge.route.l2_forward_resolver import L2ForwardResolutionStatus
from forge.route.planner import RouteTarget
from forge.route.proposal_discovery_status import (
    ProposalDiscoveryStatus,
    SourceNeutralProposalDiscoveryResolver,
    SourceNeutralRouteAdjudicationStatus,
    adjudicate_source_neutral_route,
)
from forge.route.proposal_engine import (
    OperationalCompatibility,
    ProposalBackendManifest,
    ProposalRequest,
    ProposalTargetKind,
    RootQualificationReceipt,
    SingleStepRetrosynthesisProposal,
)
from forge.route.single_step_proposal_benchmark import (
    EXPECTED_LANES,
    BenchmarkTarget,
    BenchmarkTargetManifest,
    HiddenTruthManifest,
    HiddenTruthRecord,
    MatchedTargetBudget,
)
from forge.route.single_step_proposal_benchmark_v3 import (
    ExecutableBenchmarkContractV3,
    ExecutableProposalLaneBenchmarkRunnerV3,
    FrozenLaneOutputLedgerV3,
    LaneProposalBatchV3,
    LaneTargetOutputV3,
    ProposalSourceKind,
    ScoredProposalRecordV3,
    adjudicate_frozen_score_v3,
    load_executable_benchmark_contract_v3,
    score_frozen_lane_outputs_v3,
)
from forge.route.ugi3_support_boundary import MolecularSupportState, TargetQualification

REPO = Path(__file__).resolve().parents[1]
V3_CONFIG = REPO / "configs/route/single_step_proposal_lane_qualification_benchmark_v3.json"


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _request(target: str) -> ProposalRequest:
    route_root = RouteTarget(role="oxoester_aldehyde_body_tail", canonical_smiles=target)
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
        operational_policy_id="v3-test-policy",
        operational_policy_sha256=_hash("v3-test-policy"),
    )


def _proposal(target: str, reactant: str = "CC") -> SingleStepRetrosynthesisProposal:
    request = _request(target)
    backend = ProposalBackendManifest(
        backend_id="fixture:source-neutral",
        implementation_version="1",
        checkpoint_sha256=_hash("checkpoint"),
        checkpoint_license="test-only",
        training_corpus_id="fixture",
        training_corpus_snapshot_sha256=_hash("corpus"),
        source_locator="test fixture",
    )
    return SingleStepRetrosynthesisProposal(
        proposal_id=f"proposal:{target}:{reactant}",
        backend=backend,
        request=request,
        reactant_smiles=(reactant,),
        rank=1,
        model_score=0.9,
        predicted_reaction_class="must_not_be_used",
    )


class _ExactResolver:
    def __init__(self, status: L2ForwardResolutionStatus, transforms=()) -> None:
        self.status = status
        self.transforms = tuple(transforms)
        self.resolver_config_sha256 = _hash("exact-resolver")

    def resolve(self, _proposal, *, maximum_forward_calls=None):
        return SimpleNamespace(
            status=self.status,
            forward_calls=(1 if self.status is L2ForwardResolutionStatus.EXACT_UNIQUE else 0),
        )


class _Transform:
    def __init__(self, reaction_id: str) -> None:
        self.reaction_id = reaction_id
        self.transform_sha256 = _hash(reaction_id)
        self.reaction = SimpleNamespace(role_names=("substrate",))


def test_source_neutral_discovery_statuses_never_gain_route_authority() -> None:
    exact = SourceNeutralProposalDiscoveryResolver(
        exact_resolver=_ExactResolver(L2ForwardResolutionStatus.EXACT_UNIQUE),
    ).resolve(_proposal("CCC"))
    assert exact.status is ProposalDiscoveryStatus.EXACT_KNOWN_ROUTE

    transform = _Transform("registry-owned-family")
    projected = SourceNeutralProposalDiscoveryResolver(
        exact_resolver=_ExactResolver(
            L2ForwardResolutionStatus.CENSOR_NO_VERIFIER,
            transforms=(transform,),
        ),
        executor=lambda _reaction, _reactants, max_products: ("CCC",),
    ).resolve(_proposal("CCC"))
    assert projected.status is ProposalDiscoveryStatus.KNOWN_FAMILY_FORWARD_CONSISTENT_PROJECTION
    assert projected.graph_consistent is True

    new_family = SourceNeutralProposalDiscoveryResolver(
        exact_resolver=_ExactResolver(
            L2ForwardResolutionStatus.CENSOR_NO_VERIFIER,
            transforms=(transform,),
        ),
        executor=lambda _reaction, _reactants, max_products: ("CCO",),
    ).resolve(_proposal("CCC"))
    assert new_family.status is ProposalDiscoveryStatus.NEW_FAMILY_HYPOTHESIS_RETAINED
    assert new_family.retained_for_discovery is True

    ambiguous = SourceNeutralProposalDiscoveryResolver(
        exact_resolver=_ExactResolver(
            L2ForwardResolutionStatus.CENSOR_NO_VERIFIER,
            transforms=(_Transform("family-a"), _Transform("family-b")),
        ),
        executor=lambda _reaction, _reactants, max_products: ("CCC",),
    ).resolve(_proposal("CCC"))
    assert ambiguous.status is ProposalDiscoveryStatus.AMBIGUOUS

    for result in (exact, projected, new_family, ambiguous):
        payload = result.to_dict()
        assert payload["evidence_tier"] is None
        assert payload["success_probability"] is None
        assert payload["route_closure_authorized"] is False
        assert payload["may_enter_synthesis_value"] is False
        assert payload["model_reaction_class_used"] is False

    missing = adjudicate_source_neutral_route(
        new_family,
        independent_forward_consistent=True,
        substrate_scope_qualified=False,
        evidence_qualified=False,
        operationally_compatible=True,
        l3_terminal_closed=False,
    )
    assert (
        missing.status
        is SourceNeutralRouteAdjudicationStatus.UNQUALIFIED_HYPOTHESIS_MISSING_EVIDENCE
    )
    assert missing.to_dict()["local_novelty_used_as_chemical_incompatibility"] is False

    qualification = {
        "independent_forward_consistent": True,
        "substrate_scope_qualified": True,
        "evidence_qualified": True,
        "operationally_compatible": True,
        "l3_terminal_closed": True,
        "qualified_route_state": "complete",
        "qualified_route_value": {"closure": 1, "burden": 2},
    }
    projected_qualified = adjudicate_source_neutral_route(projected, **qualification)
    new_family_qualified = adjudicate_source_neutral_route(new_family, **qualification)
    assert projected_qualified.status is SourceNeutralRouteAdjudicationStatus.QUALIFIED_ROUTE
    assert new_family_qualified.status is SourceNeutralRouteAdjudicationStatus.QUALIFIED_ROUTE
    assert (
        projected_qualified.adjudicated_route_state
        == new_family_qualified.adjudicated_route_state
        == "complete"
    )
    assert (
        projected_qualified.adjudicated_route_value
        == new_family_qualified.adjudicated_route_value
        == {"closure": 1, "burden": 2}
    )
    assert new_family_qualified.to_dict()["proposal_source_used_to_set_route_value"] is False


def _contract() -> ExecutableBenchmarkContractV3:
    return ExecutableBenchmarkContractV3(
        spec_sha256=_hash("v3-contract"),
        policy_sha256=_hash("v1-policy"),
        manifest_binding_sha256=_hash("v2-binding"),
        lane_ids=EXPECTED_LANES,
        target_count=3,
        stratum_counts=(
            ("known_exact_l2_routes", 1),
            ("held_reaction_families", 1),
            ("adversarial_incompatibles", 1),
        ),
        selection_seed=31,
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
        activation={},
        prerequisite_gates={},
        target_manifest_frozen=True,
        target_manifest_sha256=_hash("targets"),
        hidden_truth_manifest_sha256=_hash("truth"),
        resolver_config_sha256=_hash("exact-resolver"),
        runtime_lock_sha256=_hash("runtime-lock"),
        runtime_qualification_sha256=_hash("runtime-qualification"),
        semantic_repeat_count=2,
        target_artifacts_authenticated=True,
        resolver_authenticated=True,
        runtime_lock_authenticated=True,
        runtime_smoke_authenticated=True,
        decision_policy={
            "hard_requirements": {
                "known_route_top10_drop_maximum_absolute": 0.02,
                "adversarial_incompatible_accepted_proposals_maximum": 0,
            },
            "operational_checks": {
                "accepted_relative_gain_minimum": 0.1,
                "graph_consistent_drop_maximum_absolute": 0.05,
                "operational_yield_drop_maximum_absolute": 0.05,
                "invalid_raw_hypothesis_fraction_maximum": 0.1,
                "canonical_duplicate_fraction_maximum": 0.25,
                "minimum_nonadversarial_strata_improved": 1,
                "latency_p95_seconds_maximum": 30,
                "peak_memory_mebibytes_maximum": 1024,
            },
        },
    )


def _record(
    reactant: str,
    *,
    source_kind: ProposalSourceKind,
    status: ProposalDiscoveryStatus,
    accepted: bool,
) -> ScoredProposalRecordV3:
    return ScoredProposalRecordV3(
        rank=1,
        proposal_source=f"fixture:{source_kind.value}",
        proposal_source_kind=source_kind,
        reactant_multiset=(reactant,),
        discovery_status=status,
        forward_calls=1,
        operational_compatibility=(
            OperationalCompatibility.PASSED if accepted else OperationalCompatibility.NOT_ASSESSED
        ),
        accepted=accepted,
        retained_for_discovery=(status is ProposalDiscoveryStatus.NEW_FAMILY_HYPOTHESIS_RETAINED),
    )


def _row(
    lane_id: str,
    target_id: str,
    record: ScoredProposalRecordV3,
) -> LaneTargetOutputV3:
    return LaneTargetOutputV3(
        lane_id=lane_id,
        target_id=target_id,
        proposal_source_calls=1,
        raw_hypothesis_attempts=1,
        invalid_raw_hypotheses=0,
        canonical_duplicates=0,
        forward_calls=record.forward_calls,
        operational_screen_calls=int(
            record.operational_compatibility is not OperationalCompatibility.NOT_ASSESSED
        ),
        accepted_count=int(record.accepted),
        retained_discovery_count=int(record.retained_for_discovery),
        latency_seconds=0.1,
        peak_host_memory_mebibytes=10,
        peak_accelerator_memory_mebibytes=0,
        budget_states=(),
        records=(record,),
    )


def test_v3_scorer_reports_strata_pairs_provenance_resources_and_frozen_decision() -> None:
    contract = _contract()
    target_specs = (
        ("known", "known_exact_l2_routes", "CCC"),
        ("held", "held_reaction_families", "CCO"),
        ("adversarial", "adversarial_incompatibles", "CCN"),
    )
    targets = BenchmarkTargetManifest(
        manifest_sha256=contract.target_manifest_sha256,
        selection_seed=contract.selection_seed,
        targets=tuple(
            BenchmarkTarget(target_id=target_id, primary_stratum=stratum, request=_request(smiles))
            for target_id, stratum, smiles in target_specs
        ),
    )
    rows: list[LaneTargetOutputV3] = []
    for lane_id in EXPECTED_LANES:
        source_kind = (
            ProposalSourceKind.STRICT
            if lane_id == EXPECTED_LANES[0]
            else (
                ProposalSourceKind.LEARNED
                if lane_id == EXPECTED_LANES[1]
                else ProposalSourceKind.OVERLAP
            )
        )
        rows.append(
            _row(
                lane_id,
                "known",
                _record(
                    "CC",
                    source_kind=source_kind,
                    status=ProposalDiscoveryStatus.EXACT_KNOWN_ROUTE,
                    accepted=lane_id != EXPECTED_LANES[1],
                ),
            )
        )
        rows.append(
            _row(
                lane_id,
                "held",
                _record(
                    "CO" if lane_id != EXPECTED_LANES[0] else "CN",
                    source_kind=source_kind,
                    status=(
                        ProposalDiscoveryStatus.KNOWN_FAMILY_FORWARD_CONSISTENT_PROJECTION
                        if lane_id != EXPECTED_LANES[0]
                        else ProposalDiscoveryStatus.NEW_FAMILY_HYPOTHESIS_RETAINED
                    ),
                    accepted=lane_id != EXPECTED_LANES[0],
                ),
            )
        )
        rows.append(
            _row(
                lane_id,
                "adversarial",
                _record(
                    "NN",
                    source_kind=source_kind,
                    status=ProposalDiscoveryStatus.REJECTED,
                    accepted=False,
                ),
            )
        )
    outputs = FrozenLaneOutputLedgerV3(
        contract_sha256=contract.spec_sha256,
        target_manifest_sha256=targets.manifest_sha256,
        outputs=tuple(rows),
    )
    truth = HiddenTruthManifest(
        manifest_sha256=contract.hidden_truth_manifest_sha256,
        records=(
            HiddenTruthRecord(
                target_id="known",
                truth_kind="documented_exact_forward_unique_reactant_multiset",
                exact_reactant_multisets=(("CC",),),
            ),
            HiddenTruthRecord(
                target_id="held",
                truth_kind="documented_exact_forward_unique_reactant_multiset",
                exact_reactant_multisets=(("CO",),),
            ),
            HiddenTruthRecord(
                target_id="adversarial",
                truth_kind="valid_connected_wrong_handle_role_swap_control",
            ),
        ),
    )

    score = score_frozen_lane_outputs_v3(
        contract=contract,
        targets=targets,
        outputs=outputs,
        hidden_truth=truth,
    )
    assert (
        score["metrics_by_primary_stratum"][EXPECTED_LANES[2]]["held_reaction_families"][
            "accepted_proposals"
        ]
        == 1
    )
    assert (
        score["paired_target_differences"]["hybrid_vs_strict"]["aggregate"][
            "accepted_proposals_difference"
        ]["sum"]
        == 1
    )
    assert score["metrics"][EXPECTED_LANES[2]]["proposal_source_kind_counts"] == {"overlap": 3}
    assert score["metrics"][EXPECTED_LANES[2]]["latency_seconds"]["p95"] == 0.1
    assert score["metrics"][EXPECTED_LANES[2]]["peak_host_memory_mebibytes"] == 10
    assert (
        "strict_lipid_precedented__strict_first_hybrid"
        in score["cross_lane_reactant_multiset_overlap"]
    )

    decision = adjudicate_frozen_score_v3(contract=contract, score=score)
    assert decision["development_discovery_lane_qualified"] is True
    assert decision["operational_preference_met"] is True
    assert decision["promotion_recommended_for_versioned_review"] is True
    assert decision["production_activation_authorized"] is False


def test_v3_readiness_is_derived_and_fails_closed_without_semantic_smoke() -> None:
    contract = _contract()
    assert contract.execution_ready is True
    blocked = replace(contract, runtime_smoke_authenticated=False, semantic_repeat_count=1)
    assert blocked.execution_ready is False
    assert set(blocked.blocking_reasons) == {
        "runtime_smoke_not_authenticated",
        "semantic_repeat_count_below_two",
    }


def test_bound_v3_contract_derives_readiness_from_authenticated_existing_receipts() -> None:
    contract = load_executable_benchmark_contract_v3(V3_CONFIG, repo_root=REPO)

    assert contract.execution_ready is True
    assert contract.semantic_repeat_count == 2
    assert contract.target_count == 120
    assert contract.activation == {}
    assert contract.prerequisite_gates == {}


def test_v3_runner_executes_only_with_derived_ready_contract_and_source_neutral_status() -> None:
    contract = replace(
        _contract(),
        target_count=1,
        stratum_counts=(("known_exact_l2_routes", 1),),
    )
    target = BenchmarkTarget(
        target_id="known",
        primary_stratum="known_exact_l2_routes",
        request=_request("CCC"),
    )
    targets = BenchmarkTargetManifest(
        manifest_sha256=contract.target_manifest_sha256,
        selection_seed=contract.selection_seed,
        targets=(target,),
    )

    class _LaneV3:
        def __init__(self, lane_id: str) -> None:
            self.lane_id = lane_id

        def propose(
            self, target, *, budget, strict_reserved_hypotheses, learned_reserved_hypotheses
        ):
            del budget, strict_reserved_hypotheses, learned_reserved_hypotheses
            source_kind = (
                ProposalSourceKind.STRICT
                if self.lane_id == EXPECTED_LANES[0]
                else (
                    ProposalSourceKind.LEARNED
                    if self.lane_id == EXPECTED_LANES[1]
                    else ProposalSourceKind.OVERLAP
                )
            )
            return LaneProposalBatchV3(
                lane_id=self.lane_id,
                target_id=target.target_id,
                proposals=(_proposal("CCC"),),
                proposal_sources=(f"fixture:{source_kind.value}",),
                proposal_source_kinds=(source_kind,),
                proposal_source_calls=1,
                raw_hypothesis_attempts=1,
                invalid_raw_hypotheses=0,
                canonical_duplicates=0,
                peak_host_memory_mebibytes=10,
                peak_accelerator_memory_mebibytes=0,
            )

    discovery_resolver = SourceNeutralProposalDiscoveryResolver(
        exact_resolver=_ExactResolver(L2ForwardResolutionStatus.EXACT_UNIQUE),
    )
    runner = ExecutableProposalLaneBenchmarkRunnerV3(
        contract=contract,
        resolver=discovery_resolver,
        lane_engines={lane_id: _LaneV3(lane_id) for lane_id in EXPECTED_LANES},
        operational_screen=lambda _proposal, _resolution: OperationalCompatibility.PASSED,
    )

    output = runner.execute(targets)

    assert len(output.outputs) == 3
    assert all(row.accepted_count == 1 for row in output.outputs)
    assert all(
        row.records[0].discovery_status is ProposalDiscoveryStatus.EXACT_KNOWN_ROUTE
        for row in output.outputs
    )
    assert output.to_dict()["production_backend_activated"] is False
