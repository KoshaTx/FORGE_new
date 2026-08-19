"""Executable development benchmark contract for proposal-only L2 discovery lanes.

Version 3 is additive: it authenticates the frozen v1 policy and v2 target
binding, derives readiness from immutable artifacts and the already-qualified
two-repeat semantic smoke, and never mutates or activates the production lane.
Its scorer reports the full frozen comparison and returns a non-authoritative
promotion recommendation; production activation remains false.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from forge.route.graph2edits_runtime_qualification import verify_runtime_receipt
from forge.route.proposal_discovery_status import (
    ProposalDiscoveryResolution,
    ProposalDiscoveryStatus,
    SourceNeutralProposalDiscoveryResolver,
)
from forge.route.proposal_engine import OperationalCompatibility, SingleStepRetrosynthesisProposal
from forge.route.single_step_proposal_benchmark import (
    EXPECTED_LANES,
    BenchmarkContractError,
    BenchmarkExecutionBlockedError,
    BenchmarkTarget,
    BenchmarkTargetManifest,
    FrozenBenchmarkContract,
    HiddenTruthManifest,
    MatchedTargetBudget,
    load_frozen_benchmark_contract,
)

V3_CONTRACT_SCHEMA_VERSION = "forge.single_step_proposal_executable_contract.v3"
V3_OUTPUT_SCHEMA_VERSION = "forge.single_step_proposal_benchmark_output.v3"
V3_SCORE_SCHEMA_VERSION = "forge.single_step_proposal_benchmark_score.v3"
V3_DECISION_SCHEMA_VERSION = "forge.single_step_proposal_benchmark_decision.v3"


def _stable_json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _sha256_payload(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode()).hexdigest()


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise BenchmarkContractError(f"{label} must be a SHA-256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise BenchmarkContractError(f"{label} must be a SHA-256") from exc
    return value


def _artifact_path(repo_root: Path, record: Mapping[str, Any], *, label: str) -> Path:
    relative = record.get("path")
    if not isinstance(relative, str) or not relative:
        raise BenchmarkContractError(f"v3 artifact path is malformed: {label}")
    path = (repo_root / relative).resolve()
    try:
        path.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise BenchmarkContractError(f"v3 artifact escapes repository: {label}") from exc
    if not path.is_file():
        raise BenchmarkContractError(f"v3 artifact is missing: {label}")
    expected = _require_sha256(record.get("sha256"), label=f"{label} SHA-256")
    if _sha256_file(path) != expected:
        raise BenchmarkContractError(f"v3 artifact SHA-256 mismatch: {label}")
    return path


class ProposalSourceKind(str, Enum):
    """Source provenance retained after cross-source canonical deduplication."""

    STRICT = "strict"
    LEARNED = "learned"
    OVERLAP = "overlap"


@dataclass(frozen=True)
class ExecutableBenchmarkContractV3(FrozenBenchmarkContract):
    """Readiness is derived from authenticated artifacts, not manual gate booleans."""

    runtime_lock_sha256: str = ""
    runtime_qualification_sha256: str = ""
    semantic_repeat_count: int = 0
    target_artifacts_authenticated: bool = False
    resolver_authenticated: bool = False
    runtime_lock_authenticated: bool = False
    runtime_smoke_authenticated: bool = False
    decision_policy: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        _require_sha256(self.runtime_lock_sha256, label="runtime lock SHA-256")
        _require_sha256(
            self.runtime_qualification_sha256,
            label="runtime qualification SHA-256",
        )
        if not isinstance(self.semantic_repeat_count, int) or self.semantic_repeat_count < 0:
            raise BenchmarkContractError("semantic repeat count must be nonnegative")
        if not isinstance(self.decision_policy, Mapping):
            raise BenchmarkContractError("v3 decision policy is missing")

    @property
    def blocking_reasons(self) -> tuple[str, ...]:
        checks = {
            "target_artifacts_not_authenticated": self.target_artifacts_authenticated,
            "resolver_not_authenticated": self.resolver_authenticated,
            "runtime_lock_not_authenticated": self.runtime_lock_authenticated,
            "runtime_smoke_not_authenticated": self.runtime_smoke_authenticated,
            "semantic_repeat_count_below_two": self.semantic_repeat_count >= 2,
            "target_manifest_not_frozen": self.target_manifest_frozen,
        }
        return tuple(reason for reason, passed in checks.items() if not passed)

    @property
    def execution_ready(self) -> bool:
        return not self.blocking_reasons

    def assert_execution_ready(self) -> None:
        if not self.execution_ready:
            raise BenchmarkExecutionBlockedError(
                "v3 benchmark execution blocked before target access: "
                + ", ".join(self.blocking_reasons)
            )


def load_executable_benchmark_contract_v3(
    config_path: Path,
    *,
    repo_root: Path,
) -> ExecutableBenchmarkContractV3:
    """Authenticate an additive v3 contract without loading target or hidden truth."""

    try:
        config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkContractError(f"invalid v3 benchmark contract: {config_path}") from exc
    if not isinstance(config, dict) or config.get("schema_version") != V3_CONTRACT_SCHEMA_VERSION:
        raise BenchmarkContractError("unsupported v3 benchmark contract schema")
    if config.get("status") != "frozen_executable_development_contract_not_run":
        raise BenchmarkContractError("v3 benchmark contract status is not executable/frozen")
    scope = config.get("scope_guards")
    expected_scope = {
        "production_backend_active": False,
        "production_planner_modified": False,
        "route_evidence_created": False,
        "synthesis_value_modified": False,
        "sealed_holdout_accessed": False,
        "benchmark_already_executed": False,
    }
    if scope != expected_scope:
        raise BenchmarkContractError("v3 benchmark scope guards are unsafe")
    readiness = config.get("readiness_derivation")
    if not isinstance(readiness, Mapping):
        raise BenchmarkContractError("v3 readiness derivation is missing")
    if (
        readiness.get("minimum_semantic_repeat_count") != 2
        or readiness.get("require_byte_identical_normalized_proposal_identities_and_ranks")
        is not True
        or readiness.get("require_authenticated_isolated_runtime_lock") is not True
        or readiness.get("require_frozen_target_and_hidden_truth_bindings") is not True
        or readiness.get("require_exact_resolver_binding") is not True
        or readiness.get("manual_license_boolean_is_scientific_readiness_gate") is not False
        or readiness.get("manual_all_gates_summary_is_used") is not False
        or readiness.get("separate_benchmark_execution_authorization_boolean_is_used") is not False
    ):
        raise BenchmarkContractError("v3 readiness derivation contract changed")
    authority = config.get("scientific_authority")
    if not isinstance(authority, Mapping):
        raise BenchmarkContractError("v3 scientific-authority contract is missing")
    required_false_authority = (
        "proposal_model_score_used",
        "proposal_model_reaction_class_used",
        "known_family_projection_creates_evidence",
        "new_family_hypothesis_closes_route",
        "locally_novel_hypothesis_is_chemically_incompatible_by_default",
        "proposal_source_may_change_qualified_route_state_or_value",
        "any_discovery_status_may_set_success_probability",
        "any_discovery_status_may_enter_v_syn",
    )
    if any(authority.get(field) is not False for field in required_false_authority):
        raise BenchmarkContractError("v3 scientific-authority contract is unsafe")
    if (
        authority.get("known_family_projection_is_graph_consistency_only") is not True
        or authority.get(
            "identical_independent_forward_scope_evidence_operational_l3_contracts_produce_identical_route_state_and_value"
        )
        is not True
        or authority.get("unqualified_new_family_hypothesis_state")
        != "missing_independent_scope_or_evidence"
    ):
        raise BenchmarkContractError("v3 source-neutral adjudication contract changed")
    artifacts = config.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise BenchmarkContractError("v3 benchmark artifact bindings are missing")
    required = {
        "frozen_v1_runner_binding",
        "runtime_lock",
        "runtime_qualification",
        "exact_resolver_config",
    }
    if set(artifacts) != required:
        raise BenchmarkContractError("v3 benchmark artifact bindings changed")
    paths = {
        label: _artifact_path(repo_root, record, label=label)
        for label, record in artifacts.items()
        if isinstance(record, Mapping)
    }
    if len(paths) != len(artifacts):
        raise BenchmarkContractError("v3 benchmark artifact record is malformed")

    legacy = load_frozen_benchmark_contract(
        paths["frozen_v1_runner_binding"],
        repo_root=repo_root,
    )
    if _sha256_file(paths["exact_resolver_config"]) != legacy.resolver_config_sha256:
        raise BenchmarkContractError("v3 exact resolver differs from the frozen target binding")

    try:
        runtime = json.loads(paths["runtime_qualification"].read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkContractError("v3 runtime qualification is invalid") from exc
    verify_runtime_receipt(runtime, repo_root=repo_root, verify_artifacts=True)
    smoke = runtime.get("smoke_inference")
    if not isinstance(smoke, Mapping):
        raise BenchmarkContractError("v3 runtime smoke receipt is missing")
    repeat_count = smoke.get("repeat_count")
    if (
        not isinstance(repeat_count, int)
        or repeat_count < 2
        or smoke.get("byte_identical_normalized_outputs") is not True
    ):
        raise BenchmarkContractError("v3 requires the qualified two-repeat semantic smoke")
    runtime_artifacts = runtime.get("artifacts")
    if not isinstance(runtime_artifacts, Mapping):
        raise BenchmarkContractError("v3 runtime artifact receipts are missing")
    runtime_lock_record = runtime_artifacts.get("runtime_lock")
    if not isinstance(runtime_lock_record, Mapping) or runtime_lock_record.get(
        "sha256"
    ) != _sha256_file(paths["runtime_lock"]):
        raise BenchmarkContractError("v3 runtime lock differs from its qualification receipt")

    decision_policy = config.get("decision_policy")
    if not isinstance(decision_policy, Mapping):
        raise BenchmarkContractError("v3 decision policy is missing")
    return ExecutableBenchmarkContractV3(
        spec_sha256=_sha256_file(config_path),
        policy_sha256=legacy.policy_sha256,
        manifest_binding_sha256=legacy.manifest_binding_sha256,
        lane_ids=legacy.lane_ids,
        target_count=legacy.target_count,
        stratum_counts=legacy.stratum_counts,
        selection_seed=legacy.selection_seed,
        budget=legacy.budget,
        strict_reserved_hypotheses=legacy.strict_reserved_hypotheses,
        learned_reserved_hypotheses=legacy.learned_reserved_hypotheses,
        activation={},
        prerequisite_gates={},
        target_manifest_frozen=legacy.target_manifest_frozen,
        target_manifest_sha256=legacy.target_manifest_sha256,
        hidden_truth_manifest_sha256=legacy.hidden_truth_manifest_sha256,
        resolver_config_sha256=legacy.resolver_config_sha256,
        runtime_lock_sha256=_sha256_file(paths["runtime_lock"]),
        runtime_qualification_sha256=_sha256_file(paths["runtime_qualification"]),
        semantic_repeat_count=repeat_count,
        target_artifacts_authenticated=True,
        resolver_authenticated=True,
        runtime_lock_authenticated=True,
        runtime_smoke_authenticated=True,
        decision_policy=dict(decision_policy),
    )


@dataclass(frozen=True)
class LaneProposalBatchV3:
    """One bounded lane result with explicit source and resource provenance."""

    lane_id: str
    target_id: str
    proposals: tuple[SingleStepRetrosynthesisProposal, ...]
    proposal_sources: tuple[str, ...]
    proposal_source_kinds: tuple[ProposalSourceKind, ...]
    proposal_source_calls: int
    raw_hypothesis_attempts: int
    invalid_raw_hypotheses: int
    canonical_duplicates: int
    peak_host_memory_mebibytes: float | None = None
    peak_accelerator_memory_mebibytes: float | None = None

    def __post_init__(self) -> None:
        if not (
            len(self.proposals) == len(self.proposal_sources) == len(self.proposal_source_kinds)
        ):
            raise BenchmarkContractError("v3 proposal provenance lengths disagree")
        if any(not isinstance(value, ProposalSourceKind) for value in self.proposal_source_kinds):
            raise BenchmarkContractError("v3 proposal source kind is invalid")
        for field in (
            "proposal_source_calls",
            "raw_hypothesis_attempts",
            "invalid_raw_hypotheses",
            "canonical_duplicates",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise BenchmarkContractError(f"{field} must be a nonnegative integer")
        for field in ("peak_host_memory_mebibytes", "peak_accelerator_memory_mebibytes"):
            value = getattr(self, field)
            if value is not None and (not isinstance(value, (int, float)) or value < 0):
                raise BenchmarkContractError(f"{field} must be nonnegative or null")


class BenchmarkLaneEngineV3(Protocol):
    def propose(
        self,
        target: BenchmarkTarget,
        *,
        budget: MatchedTargetBudget,
        strict_reserved_hypotheses: int,
        learned_reserved_hypotheses: int,
    ) -> LaneProposalBatchV3: ...


class BenchmarkOperationalScreenV3(Protocol):
    def __call__(
        self,
        proposal: SingleStepRetrosynthesisProposal,
        resolution: ProposalDiscoveryResolution,
    ) -> OperationalCompatibility: ...


@dataclass(frozen=True)
class ScoredProposalRecordV3:
    rank: int
    proposal_source: str
    proposal_source_kind: ProposalSourceKind
    reactant_multiset: tuple[str, ...]
    discovery_status: ProposalDiscoveryStatus
    forward_calls: int
    operational_compatibility: OperationalCompatibility
    accepted: bool
    retained_for_discovery: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "proposal_source": self.proposal_source,
            "proposal_source_kind": self.proposal_source_kind.value,
            "reactant_multiset": list(self.reactant_multiset),
            "discovery_status": self.discovery_status.value,
            "forward_calls": self.forward_calls,
            "operational_compatibility": self.operational_compatibility.value,
            "accepted": self.accepted,
            "retained_for_discovery": self.retained_for_discovery,
            "evidence_tier": None,
            "success_probability": None,
            "route_closure_authorized": False,
            "may_enter_synthesis_value": False,
        }


@dataclass(frozen=True)
class LaneTargetOutputV3:
    lane_id: str
    target_id: str
    proposal_source_calls: int
    raw_hypothesis_attempts: int
    invalid_raw_hypotheses: int
    canonical_duplicates: int
    forward_calls: int
    operational_screen_calls: int
    accepted_count: int
    retained_discovery_count: int
    latency_seconds: float
    peak_host_memory_mebibytes: float | None
    peak_accelerator_memory_mebibytes: float | None
    budget_states: tuple[str, ...]
    records: tuple[ScoredProposalRecordV3, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "lane_id": self.lane_id,
            "target_id": self.target_id,
            "proposal_source_calls": self.proposal_source_calls,
            "raw_hypothesis_attempts": self.raw_hypothesis_attempts,
            "invalid_raw_hypotheses": self.invalid_raw_hypotheses,
            "canonical_duplicates": self.canonical_duplicates,
            "forward_calls": self.forward_calls,
            "operational_screen_calls": self.operational_screen_calls,
            "accepted_count": self.accepted_count,
            "retained_discovery_count": self.retained_discovery_count,
            "latency_seconds": self.latency_seconds,
            "peak_host_memory_mebibytes": self.peak_host_memory_mebibytes,
            "peak_accelerator_memory_mebibytes": self.peak_accelerator_memory_mebibytes,
            "budget_states": list(self.budget_states),
            "records": [record.to_dict() for record in self.records],
        }


@dataclass(frozen=True)
class FrozenLaneOutputLedgerV3:
    contract_sha256: str
    target_manifest_sha256: str
    outputs: tuple[LaneTargetOutputV3, ...]

    @property
    def ledger_sha256(self) -> str:
        return _sha256_payload(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": V3_OUTPUT_SCHEMA_VERSION,
            "contract_sha256": self.contract_sha256,
            "target_manifest_sha256": self.target_manifest_sha256,
            "outputs": [output.to_dict() for output in self.outputs],
            "hidden_truth_loaded_during_lane_execution": False,
            "route_evidence_created": False,
            "v_syn_modified": False,
            "production_backend_activated": False,
        }
        if include_hash:
            value["ledger_sha256"] = self.ledger_sha256
        return value


class ExecutableProposalLaneBenchmarkRunnerV3:
    """Run the development lanes while retaining non-evidentiary discovery states."""

    def __init__(
        self,
        *,
        contract: ExecutableBenchmarkContractV3,
        resolver: SourceNeutralProposalDiscoveryResolver,
        lane_engines: Mapping[str, BenchmarkLaneEngineV3],
        operational_screen: BenchmarkOperationalScreenV3,
    ) -> None:
        if not isinstance(contract, ExecutableBenchmarkContractV3):
            raise BenchmarkContractError("v3 runner requires an executable v3 contract")
        if not isinstance(resolver, SourceNeutralProposalDiscoveryResolver):
            raise BenchmarkContractError("v3 runner requires the source-neutral resolver")
        if resolver.exact_resolver_config_sha256 != contract.resolver_config_sha256:
            raise BenchmarkContractError("v3 resolver hash disagrees with the contract")
        if set(lane_engines) != set(EXPECTED_LANES):
            raise BenchmarkContractError("v3 runner requires exactly the three frozen lanes")
        if not callable(operational_screen):
            raise BenchmarkContractError("v3 operational screen must be callable")
        self._contract = contract
        self._resolver = resolver
        self._lane_engines = dict(lane_engines)
        self._operational_screen = operational_screen

    def execute(self, targets: BenchmarkTargetManifest) -> FrozenLaneOutputLedgerV3:
        self._contract.assert_execution_ready()
        targets.validate_against(self._contract)
        outputs: list[LaneTargetOutputV3] = []
        for target in targets.targets:
            for lane_id in self._contract.lane_ids:
                started = time.perf_counter()
                batch = self._lane_engines[lane_id].propose(
                    target,
                    budget=self._contract.budget,
                    strict_reserved_hypotheses=self._contract.strict_reserved_hypotheses,
                    learned_reserved_hypotheses=self._contract.learned_reserved_hypotheses,
                )
                outputs.append(
                    self._resolve_batch(
                        target,
                        lane_id,
                        batch,
                        started_at=started,
                    )
                )
        return FrozenLaneOutputLedgerV3(
            contract_sha256=self._contract.spec_sha256,
            target_manifest_sha256=targets.manifest_sha256,
            outputs=tuple(outputs),
        )

    def _resolve_batch(
        self,
        target: BenchmarkTarget,
        lane_id: str,
        batch: LaneProposalBatchV3,
        *,
        started_at: float,
    ) -> LaneTargetOutputV3:
        budget = self._contract.budget
        if batch.lane_id != lane_id or batch.target_id != target.target_id:
            raise BenchmarkContractError("v3 lane output target or lane identity mismatch")
        if batch.proposal_source_calls > budget.proposal_source_calls:
            raise BenchmarkContractError("v3 lane exceeded proposal-source-call budget")
        if batch.raw_hypothesis_attempts > budget.raw_hypothesis_attempts:
            raise BenchmarkContractError("v3 lane exceeded raw-hypothesis budget")
        if batch.invalid_raw_hypotheses > batch.raw_hypothesis_attempts:
            raise BenchmarkContractError("v3 invalid hypotheses exceed raw attempts")
        if batch.canonical_duplicates > batch.raw_hypothesis_attempts:
            raise BenchmarkContractError("v3 duplicates exceed raw attempts")
        if len(batch.proposals) > budget.canonical_hypotheses_considered:
            raise BenchmarkContractError("v3 lane exceeded canonical-hypothesis budget")
        identities = tuple(tuple(sorted(proposal.reactant_smiles)) for proposal in batch.proposals)
        if len(set(identities)) != len(identities):
            raise BenchmarkContractError("v3 lane emitted canonical duplicate proposals")
        ranks = tuple(proposal.rank for proposal in batch.proposals)
        if any(rank < 1 for rank in ranks) or len(set(ranks)) != len(ranks):
            raise BenchmarkContractError("v3 proposal ranks must be unique positive integers")

        forward_calls = 0
        operational_calls = 0
        accepted_count = 0
        retained_count = 0
        records: list[ScoredProposalRecordV3] = []
        for proposal, source, source_kind in zip(
            batch.proposals,
            batch.proposal_sources,
            batch.proposal_source_kinds,
            strict=True,
        ):
            remaining = budget.forward_verifier_calls - forward_calls
            if remaining <= 0:
                break
            resolution = self._resolver.resolve(proposal, maximum_forward_calls=remaining)
            forward_calls += resolution.forward_calls
            compatibility = OperationalCompatibility.NOT_ASSESSED
            accepted = False
            if resolution.graph_consistent:
                if operational_calls >= budget.operational_screen_calls:
                    break
                compatibility = self._operational_screen(proposal, resolution)
                if not isinstance(compatibility, OperationalCompatibility):
                    raise BenchmarkContractError("v3 operational screen returned invalid state")
                operational_calls += 1
                accepted = compatibility is OperationalCompatibility.PASSED
                if accepted and accepted_count < budget.accepted_proposal_cap:
                    accepted_count += 1
                elif accepted:
                    accepted = False
            retained = resolution.retained_for_discovery
            retained_count += int(retained)
            records.append(
                ScoredProposalRecordV3(
                    rank=proposal.rank,
                    proposal_source=source,
                    proposal_source_kind=source_kind,
                    reactant_multiset=tuple(sorted(proposal.reactant_smiles)),
                    discovery_status=resolution.status,
                    forward_calls=resolution.forward_calls,
                    operational_compatibility=compatibility,
                    accepted=accepted,
                    retained_for_discovery=retained,
                )
            )

        latency_seconds = time.perf_counter() - started_at
        if forward_calls > budget.forward_verifier_calls:
            raise BenchmarkContractError("v3 lane exceeded forward-verifier budget")
        if operational_calls > budget.operational_screen_calls:
            raise BenchmarkContractError("v3 lane exceeded operational-screen budget")

        budget_states: list[str] = []
        if len(records) < len(batch.proposals) and forward_calls >= budget.forward_verifier_calls:
            budget_states.append("forward_verifier_calls_exhausted")
        if operational_calls >= budget.operational_screen_calls:
            budget_states.append("operational_screen_calls_ceiling_reached")
        if accepted_count >= budget.accepted_proposal_cap:
            budget_states.append("accepted_proposal_cap_reached")
        if latency_seconds > budget.wall_seconds:
            budget_states.append("wall_seconds_exceeded")
        if (
            batch.peak_host_memory_mebibytes is not None
            and batch.peak_host_memory_mebibytes > budget.peak_host_memory_mebibytes
        ):
            budget_states.append("peak_host_memory_exceeded")
        if (
            batch.peak_accelerator_memory_mebibytes is not None
            and batch.peak_accelerator_memory_mebibytes > budget.peak_accelerator_memory_mebibytes
        ):
            budget_states.append("peak_accelerator_memory_exceeded")
        return LaneTargetOutputV3(
            lane_id=lane_id,
            target_id=target.target_id,
            proposal_source_calls=batch.proposal_source_calls,
            raw_hypothesis_attempts=batch.raw_hypothesis_attempts,
            invalid_raw_hypotheses=batch.invalid_raw_hypotheses,
            canonical_duplicates=batch.canonical_duplicates,
            forward_calls=forward_calls,
            operational_screen_calls=operational_calls,
            accepted_count=accepted_count,
            retained_discovery_count=retained_count,
            latency_seconds=latency_seconds,
            peak_host_memory_mebibytes=batch.peak_host_memory_mebibytes,
            peak_accelerator_memory_mebibytes=batch.peak_accelerator_memory_mebibytes,
            budget_states=tuple(budget_states),
            records=tuple(records),
        )


def _fraction(numerator: int, denominator: int) -> dict[str, int | float | None]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "fraction": None if denominator == 0 else numerator / denominator,
    }


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _top_k_recovery(
    rows: list[LaneTargetOutputV3],
    truth_by_target: Mapping[str, Any],
) -> dict[str, dict[str, int | float | None]]:
    result: dict[str, dict[str, int | float | None]] = {}
    for k in (1, 5, 10, 20):
        numerator = 0
        denominator = 0
        for row in rows:
            truth = truth_by_target.get(row.target_id)
            if truth is None or not truth.exact_reactant_multisets:
                continue
            denominator += 1
            accepted_truth = set(truth.exact_reactant_multisets)
            if any(
                record.rank <= k and record.reactant_multiset in accepted_truth
                for record in row.records
            ):
                numerator += 1
        result[str(k)] = _fraction(numerator, denominator)
    return result


def _summarize_rows(
    rows: list[LaneTargetOutputV3],
    *,
    truth_by_target: Mapping[str, Any],
    adversarial_target_ids: set[str],
) -> dict[str, Any]:
    records = [record for row in rows for record in row.records]
    raw_attempts = sum(row.raw_hypothesis_attempts for row in rows)
    operational_records = [
        record
        for record in records
        if record.operational_compatibility is not OperationalCompatibility.NOT_ASSESSED
    ]
    graph_consistent = sum(
        record.discovery_status
        in {
            ProposalDiscoveryStatus.EXACT_KNOWN_ROUTE,
            ProposalDiscoveryStatus.KNOWN_FAMILY_FORWARD_CONSISTENT_PROJECTION,
        }
        for record in records
    )
    source_counts = Counter(record.proposal_source_kind.value for record in records)
    status_counts = Counter(record.discovery_status.value for record in records)
    budget_counts = Counter(state for row in rows for state in row.budget_states)
    latency = [row.latency_seconds for row in rows]
    host_memory = [
        row.peak_host_memory_mebibytes for row in rows if row.peak_host_memory_mebibytes is not None
    ]
    accelerator_memory = [
        row.peak_accelerator_memory_mebibytes
        for row in rows
        if row.peak_accelerator_memory_mebibytes is not None
    ]
    return {
        "target_count": len(rows),
        "known_route_top_k": _top_k_recovery(rows, truth_by_target),
        "graph_consistent": _fraction(graph_consistent, len(records)),
        "operational_screen_yield": _fraction(
            sum(
                record.operational_compatibility is OperationalCompatibility.PASSED
                for record in operational_records
            ),
            len(operational_records),
        ),
        "accepted_proposals": sum(row.accepted_count for row in rows),
        "retained_new_family_hypotheses": sum(row.retained_discovery_count for row in rows),
        "invalid_raw_hypotheses": _fraction(
            sum(row.invalid_raw_hypotheses for row in rows), raw_attempts
        ),
        "canonical_duplicates": _fraction(
            sum(row.canonical_duplicates for row in rows), raw_attempts
        ),
        "proposal_source_kind_counts": dict(sorted(source_counts.items())),
        "discovery_status_counts": dict(sorted(status_counts.items())),
        "budget_state_counts": dict(sorted(budget_counts.items())),
        "latency_seconds": {
            "p50": _percentile(latency, 0.50),
            "p95": _percentile(latency, 0.95),
            "max": max(latency, default=None),
        },
        "peak_host_memory_mebibytes": max(host_memory, default=None),
        "peak_accelerator_memory_mebibytes": max(accelerator_memory, default=None),
        "adversarial_incompatible_accepted_proposals": sum(
            row.accepted_count for row in rows if row.target_id in adversarial_target_ids
        ),
    }


def _paired_differences(
    *,
    outputs_by_lane_target: Mapping[tuple[str, str], LaneTargetOutputV3],
    targets: BenchmarkTargetManifest,
    truth_by_target: Mapping[str, Any],
    left_lane: str,
    right_lane: str,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for target in targets.targets:
        left = outputs_by_lane_target[(left_lane, target.target_id)]
        right = outputs_by_lane_target[(right_lane, target.target_id)]
        truth = truth_by_target.get(target.target_id)
        exact_truth = set(() if truth is None else truth.exact_reactant_multisets)

        def recovered(row: LaneTargetOutputV3) -> int:
            return int(
                bool(exact_truth)
                and any(
                    record.rank <= 10 and record.reactant_multiset in exact_truth
                    for record in row.records
                )
            )

        records.append(
            {
                "target_id": target.target_id,
                "primary_stratum": target.primary_stratum,
                "accepted_proposals_difference": left.accepted_count - right.accepted_count,
                "known_route_top10_recovery_difference": recovered(left) - recovered(right),
                "graph_consistent_difference": sum(
                    record.discovery_status
                    in {
                        ProposalDiscoveryStatus.EXACT_KNOWN_ROUTE,
                        ProposalDiscoveryStatus.KNOWN_FAMILY_FORWARD_CONSISTENT_PROJECTION,
                    }
                    for record in left.records
                )
                - sum(
                    record.discovery_status
                    in {
                        ProposalDiscoveryStatus.EXACT_KNOWN_ROUTE,
                        ProposalDiscoveryStatus.KNOWN_FAMILY_FORWARD_CONSISTENT_PROJECTION,
                    }
                    for record in right.records
                ),
                "retained_new_family_difference": (
                    left.retained_discovery_count - right.retained_discovery_count
                ),
            }
        )
    fields = (
        "accepted_proposals_difference",
        "known_route_top10_recovery_difference",
        "graph_consistent_difference",
        "retained_new_family_difference",
    )
    return {
        "left_lane": left_lane,
        "right_lane": right_lane,
        "per_target": records,
        "aggregate": {
            field: {
                "sum": sum(record[field] for record in records),
                "mean": sum(record[field] for record in records) / len(records),
            }
            for field in fields
        },
    }


def score_frozen_lane_outputs_v3(
    *,
    contract: ExecutableBenchmarkContractV3,
    targets: BenchmarkTargetManifest,
    outputs: FrozenLaneOutputLedgerV3,
    hidden_truth: HiddenTruthManifest,
) -> dict[str, Any]:
    """Score only after lane output freeze; no score can activate production."""

    targets.validate_against(contract)
    hidden_truth.validate_target_ids(targets)
    if hidden_truth.manifest_sha256 != contract.hidden_truth_manifest_sha256:
        raise BenchmarkContractError("v3 hidden truth is not the frozen scoring artifact")
    if outputs.contract_sha256 != contract.spec_sha256:
        raise BenchmarkContractError("v3 output contract hash mismatch")
    if outputs.target_manifest_sha256 != targets.manifest_sha256:
        raise BenchmarkContractError("v3 output target hash mismatch")
    expected = {
        (lane_id, target.target_id) for lane_id in contract.lane_ids for target in targets.targets
    }
    output_keys = [(row.lane_id, row.target_id) for row in outputs.outputs]
    if len(set(output_keys)) != len(output_keys) or set(output_keys) != expected:
        raise BenchmarkContractError("v3 output ledger does not cover the lane-target grid")
    output_by_lane_target = {(row.lane_id, row.target_id): row for row in outputs.outputs}
    truth_by_target = {record.target_id: record for record in hidden_truth.records}
    adversarial_ids = {
        target.target_id
        for target in targets.targets
        if target.primary_stratum == "adversarial_incompatibles"
    }
    metrics: dict[str, Any] = {}
    per_stratum: dict[str, Any] = {}
    for lane_id in contract.lane_ids:
        lane_rows = [
            output_by_lane_target[(lane_id, target.target_id)] for target in targets.targets
        ]
        metrics[lane_id] = _summarize_rows(
            lane_rows,
            truth_by_target=truth_by_target,
            adversarial_target_ids=adversarial_ids,
        )
        per_stratum[lane_id] = {}
        for stratum, _count in contract.stratum_counts:
            stratum_ids = {
                target.target_id for target in targets.targets if target.primary_stratum == stratum
            }
            per_stratum[lane_id][stratum] = _summarize_rows(
                [row for row in lane_rows if row.target_id in stratum_ids],
                truth_by_target=truth_by_target,
                adversarial_target_ids=adversarial_ids,
            )

    cross_lane_overlap: dict[str, int] = {}
    lane_pairs = (
        (EXPECTED_LANES[0], EXPECTED_LANES[1]),
        (EXPECTED_LANES[0], EXPECTED_LANES[2]),
        (EXPECTED_LANES[1], EXPECTED_LANES[2]),
    )
    for left, right in lane_pairs:
        overlap = 0
        for target in targets.targets:
            left_ids = {
                record.reactant_multiset
                for record in output_by_lane_target[(left, target.target_id)].records
            }
            right_ids = {
                record.reactant_multiset
                for record in output_by_lane_target[(right, target.target_id)].records
            }
            overlap += len(left_ids & right_ids)
        cross_lane_overlap[f"{left}__{right}"] = overlap

    score: dict[str, Any] = {
        "schema_version": V3_SCORE_SCHEMA_VERSION,
        "contract_sha256": contract.spec_sha256,
        "target_manifest_sha256": targets.manifest_sha256,
        "output_ledger_sha256": outputs.ledger_sha256,
        "hidden_truth_manifest_sha256": hidden_truth.manifest_sha256,
        "metrics": metrics,
        "metrics_by_primary_stratum": per_stratum,
        "paired_target_differences": {
            "hybrid_vs_strict": _paired_differences(
                outputs_by_lane_target=output_by_lane_target,
                targets=targets,
                truth_by_target=truth_by_target,
                left_lane=EXPECTED_LANES[2],
                right_lane=EXPECTED_LANES[0],
            ),
            "hybrid_vs_learned": _paired_differences(
                outputs_by_lane_target=output_by_lane_target,
                targets=targets,
                truth_by_target=truth_by_target,
                left_lane=EXPECTED_LANES[2],
                right_lane=EXPECTED_LANES[1],
            ),
        },
        "cross_lane_reactant_multiset_overlap": cross_lane_overlap,
        "model_score_used_for_scoring": False,
        "route_evidence_created": False,
        "v_syn_modified": False,
        "production_activation_authorized": False,
    }
    score["score_sha256"] = _sha256_payload(score)
    return score


def _metric_fraction(metrics: Mapping[str, Any], path: tuple[str, ...]) -> float | None:
    value: Any = metrics
    for key in path:
        value = value[key]
    return value if isinstance(value, (int, float)) else None


def adjudicate_frozen_score_v3(
    *,
    contract: ExecutableBenchmarkContractV3,
    score: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the frozen hard decision and report operational checks separately."""

    if score.get("schema_version") != V3_SCORE_SCHEMA_VERSION:
        raise BenchmarkContractError("unsupported v3 score schema")
    if score.get("contract_sha256") != contract.spec_sha256:
        raise BenchmarkContractError("v3 score belongs to another contract")
    policy = contract.decision_policy
    assert policy is not None
    hard_policy = policy.get("hard_requirements")
    operational_policy = policy.get("operational_checks")
    if not isinstance(hard_policy, Mapping) or not isinstance(operational_policy, Mapping):
        raise BenchmarkContractError("v3 decision policy is malformed")
    metrics = score["metrics"]
    strict = metrics[EXPECTED_LANES[0]]
    hybrid = metrics[EXPECTED_LANES[2]]
    strict_top10 = _metric_fraction(strict, ("known_route_top_k", "10", "fraction"))
    hybrid_top10 = _metric_fraction(hybrid, ("known_route_top_k", "10", "fraction"))
    hard_checks = {
        "known_route_top10_retained": (
            strict_top10 is not None
            and hybrid_top10 is not None
            and strict_top10 - hybrid_top10
            <= hard_policy["known_route_top10_drop_maximum_absolute"]
        ),
        "accepted_proposals_strictly_improved": (
            hybrid["accepted_proposals"] > strict["accepted_proposals"]
        ),
        "zero_adversarial_admissions": (
            hybrid["adversarial_incompatible_accepted_proposals"]
            <= hard_policy["adversarial_incompatible_accepted_proposals_maximum"]
        ),
        "scientific_authority_preserved": (
            score.get("route_evidence_created") is False
            and score.get("v_syn_modified") is False
            and score.get("production_activation_authorized") is False
        ),
    }

    strict_graph = strict["graph_consistent"]["fraction"]
    hybrid_graph = hybrid["graph_consistent"]["fraction"]
    strict_operational = strict["operational_screen_yield"]["fraction"]
    hybrid_operational = hybrid["operational_screen_yield"]["fraction"]
    strict_accepted = strict["accepted_proposals"]
    relative_gain = (
        math.inf
        if strict_accepted == 0 and hybrid["accepted_proposals"] > 0
        else (
            0.0
            if strict_accepted == 0
            else (hybrid["accepted_proposals"] - strict_accepted) / strict_accepted
        )
    )
    improved_strata = 0
    per_stratum = score["metrics_by_primary_stratum"]
    for stratum in per_stratum[EXPECTED_LANES[2]]:
        if stratum == "adversarial_incompatibles":
            continue
        improved_strata += int(
            per_stratum[EXPECTED_LANES[2]][stratum]["accepted_proposals"]
            > per_stratum[EXPECTED_LANES[0]][stratum]["accepted_proposals"]
        )
    held_family = "held_reaction_families"
    operational_checks = {
        "accepted_relative_gain": (
            relative_gain >= operational_policy["accepted_relative_gain_minimum"]
        ),
        "graph_consistent_rate_retained": (
            strict_graph is not None
            and hybrid_graph is not None
            and strict_graph - hybrid_graph
            <= operational_policy["graph_consistent_drop_maximum_absolute"]
        ),
        "operational_yield_retained": (
            strict_operational is not None
            and hybrid_operational is not None
            and strict_operational - hybrid_operational
            <= operational_policy["operational_yield_drop_maximum_absolute"]
        ),
        "invalid_fraction": (
            hybrid["invalid_raw_hypotheses"]["fraction"] is not None
            and hybrid["invalid_raw_hypotheses"]["fraction"]
            <= operational_policy["invalid_raw_hypothesis_fraction_maximum"]
        ),
        "duplicate_fraction": (
            hybrid["canonical_duplicates"]["fraction"] is not None
            and hybrid["canonical_duplicates"]["fraction"]
            <= operational_policy["canonical_duplicate_fraction_maximum"]
        ),
        "held_family_improved": (
            per_stratum[EXPECTED_LANES[2]][held_family]["accepted_proposals"]
            > per_stratum[EXPECTED_LANES[0]][held_family]["accepted_proposals"]
        ),
        "minimum_chemotype_strata_improved": (
            improved_strata >= operational_policy["minimum_nonadversarial_strata_improved"]
        ),
        "latency_within_limit": (
            hybrid["latency_seconds"]["p95"] is not None
            and hybrid["latency_seconds"]["p95"]
            <= operational_policy["latency_p95_seconds_maximum"]
        ),
        "host_memory_within_limit": (
            hybrid["peak_host_memory_mebibytes"] is not None
            and hybrid["peak_host_memory_mebibytes"]
            <= operational_policy["peak_memory_mebibytes_maximum"]
        ),
    }
    decision: dict[str, Any] = {
        "schema_version": V3_DECISION_SCHEMA_VERSION,
        "contract_sha256": contract.spec_sha256,
        "score_sha256": score.get("score_sha256"),
        "hard_checks": hard_checks,
        "operational_checks": operational_checks,
        "development_discovery_lane_qualified": all(hard_checks.values()),
        "operational_preference_met": all(operational_checks.values()),
        "promotion_recommended_for_versioned_review": (
            all(hard_checks.values()) and all(operational_checks.values())
        ),
        "production_activation_authorized": False,
        "route_evidence_created": False,
        "v_syn_modified": False,
        "observed_relative_accepted_gain": relative_gain,
        "observed_nonadversarial_strata_improved": improved_strata,
    }
    decision["decision_sha256"] = _sha256_payload(decision)
    return decision


__all__ = [
    "ExecutableBenchmarkContractV3",
    "ExecutableProposalLaneBenchmarkRunnerV3",
    "FrozenLaneOutputLedgerV3",
    "LaneProposalBatchV3",
    "LaneTargetOutputV3",
    "ProposalSourceKind",
    "ScoredProposalRecordV3",
    "adjudicate_frozen_score_v3",
    "load_executable_benchmark_contract_v3",
    "score_frozen_lane_outputs_v3",
]
