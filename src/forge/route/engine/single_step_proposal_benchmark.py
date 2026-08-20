"""Nonexecuting matched-budget benchmark runner for proposal-only L2 lanes.

The production specification is currently blocked.  This module verifies all
public receipts before any target-manifest loader is invoked, keeps hidden route
truth outside lane execution, and binds every proposal lane to the independent
upstream L2 resolver.  It cannot activate Graph2Edits or confer evidence/value
authority.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from forge.core.hashing import sha256_file as _sha256_file
from forge.core.hashing import sha256_json as _sha256_payload
from forge.route.assessment.l2_forward_resolver import (
    IndependentL2ForwardResolver,
    L2ForwardResolution,
    L2ForwardResolutionStatus,
)
from forge.route.engine.proposal_engine import (
    OperationalCompatibility,
    ProposalRequest,
    SingleStepRetrosynthesisProposal,
)

BENCHMARK_SPEC_SCHEMA_VERSION = "forge.single_step_proposal_lane_qualification_benchmark.v1"
BENCHMARK_BINDING_SCHEMA_VERSION = "forge.single_step_proposal_benchmark_runner_binding.v1"
BENCHMARK_MANIFEST_BINDING_SCHEMA_VERSION = (
    "forge.single_step_proposal_lane_qualification_benchmark.v2"
)
BENCHMARK_TARGET_MANIFEST_SCHEMA_VERSION = "forge.single_step_proposal_targets.v1"
BENCHMARK_HIDDEN_TRUTH_SCHEMA_VERSION = "forge.single_step_proposal_hidden_truth.v1"
BENCHMARK_OUTPUT_SCHEMA_VERSION = "forge.single_step_proposal_benchmark_output.v1"

EXPECTED_LANES = (
    "strict_lipid_precedented",
    "unrestricted_graph2edits_proposal_only",
    "strict_first_hybrid",
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class BenchmarkContractError(ValueError):
    """Raised when the frozen benchmark contract is malformed."""


class BenchmarkExecutionBlockedError(RuntimeError):
    """Raised before target access when prerequisite gates are open."""


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise BenchmarkContractError(f"{label} must be a lowercase SHA-256")
    return value


def _nonempty(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkContractError(f"{label} must be a nonempty string")
    return value


def _nonnegative_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BenchmarkContractError(f"{label} must be a nonnegative integer")
    return value


def _positive_integer(value: Any, *, label: str) -> int:
    result = _nonnegative_integer(value, label=label)
    if result < 1:
        raise BenchmarkContractError(f"{label} must be positive")
    return result


@dataclass(frozen=True)
class MatchedTargetBudget:
    """Identical per-target ceiling applied to all three lanes."""

    proposal_source_calls: int
    raw_hypothesis_attempts: int
    canonical_hypotheses_considered: int
    forward_verifier_calls: int
    operational_screen_calls: int
    accepted_proposal_cap: int
    wall_seconds: int
    cpu_threads: int
    peak_host_memory_mebibytes: int
    peak_accelerator_memory_mebibytes: int

    def __post_init__(self) -> None:
        for field in self.__dataclass_fields__:
            _positive_integer(getattr(self, field), label=field)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> MatchedTargetBudget:
        expected = set(cls.__dataclass_fields__)
        if set(value) != expected:
            raise BenchmarkContractError("matched per-target budget fields disagree")
        return cls(**{field: value[field] for field in expected})

    def to_dict(self) -> dict[str, int]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


@dataclass(frozen=True)
class FrozenBenchmarkContract:
    """Authenticated nonexecuting benchmark specification."""

    spec_sha256: str
    policy_sha256: str
    manifest_binding_sha256: str
    lane_ids: tuple[str, ...]
    target_count: int
    stratum_counts: tuple[tuple[str, int], ...]
    selection_seed: int
    budget: MatchedTargetBudget
    strict_reserved_hypotheses: int
    learned_reserved_hypotheses: int
    activation: Mapping[str, bool]
    prerequisite_gates: Mapping[str, bool]
    target_manifest_frozen: bool
    target_manifest_sha256: str
    hidden_truth_manifest_sha256: str
    resolver_config_sha256: str

    def __post_init__(self) -> None:
        _sha256(self.spec_sha256, label="benchmark spec SHA-256")
        _sha256(self.policy_sha256, label="benchmark policy SHA-256")
        _sha256(self.manifest_binding_sha256, label="manifest binding SHA-256")
        if self.lane_ids != EXPECTED_LANES:
            raise BenchmarkContractError("benchmark lane order changed")
        _positive_integer(self.target_count, label="target_count")
        _nonnegative_integer(self.selection_seed, label="selection_seed")
        if sum(count for _, count in self.stratum_counts) != self.target_count:
            raise BenchmarkContractError("benchmark stratum counts do not sum to target_count")
        if (
            self.strict_reserved_hypotheses + self.learned_reserved_hypotheses
            != self.budget.raw_hypothesis_attempts
        ):
            raise BenchmarkContractError("hybrid quotas do not fill the matched raw budget")
        _sha256(self.resolver_config_sha256, label="resolver config SHA-256")
        _sha256(self.target_manifest_sha256, label="target manifest SHA-256")
        _sha256(self.hidden_truth_manifest_sha256, label="hidden truth manifest SHA-256")

    @property
    def blocking_reasons(self) -> tuple[str, ...]:
        blockers: list[str] = []
        if self.activation.get("benchmark_execution_authorized") is not True:
            blockers.append("benchmark_execution_not_authorized")
        if not self.target_manifest_frozen:
            blockers.append("target_manifest_not_frozen")
        for gate, passed in sorted(self.prerequisite_gates.items()):
            if gate == "all_gates_passed":
                continue
            if passed is not True:
                blockers.append(f"prerequisite_gate_open:{gate}")
        if self.prerequisite_gates.get("all_gates_passed") is not True:
            blockers.append("prerequisite_gate_summary_false")
        return tuple(blockers)

    @property
    def execution_ready(self) -> bool:
        return not self.blocking_reasons

    def assert_execution_ready(self) -> None:
        if not self.execution_ready:
            raise BenchmarkExecutionBlockedError(
                "benchmark execution blocked before target access: "
                + ", ".join(self.blocking_reasons)
            )


def load_frozen_benchmark_contract(
    binding_path: Path,
    *,
    repo_root: Path,
) -> FrozenBenchmarkContract:
    """Authenticate runner, policy and manifest pins without loading target content."""

    try:
        binding = json.loads(binding_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkContractError(f"invalid benchmark binding: {binding_path}") from exc
    if (
        not isinstance(binding, dict)
        or binding.get("schema_version") != BENCHMARK_BINDING_SCHEMA_VERSION
    ):
        raise BenchmarkContractError("unsupported benchmark runner binding schema")
    if binding.get("status") != "bound_runner_preflight_not_executed":
        raise BenchmarkContractError("benchmark runner binding is not preflight-only")
    binding_scope = binding.get("scope_guards")
    if not isinstance(binding_scope, Mapping) or any(
        value is not False for value in binding_scope.values()
    ):
        raise BenchmarkContractError("benchmark runner scope guards are unsafe")
    binding_artifacts = binding.get("artifacts")
    if not isinstance(binding_artifacts, Mapping):
        raise BenchmarkContractError("benchmark runner artifact receipts are missing")
    authenticated: dict[str, Path] = {}
    for label, record in binding_artifacts.items():
        if not isinstance(record, Mapping):
            raise BenchmarkContractError(f"runner artifact receipt is malformed: {label}")
        relative = record.get("path")
        if not isinstance(relative, str) or not relative:
            raise BenchmarkContractError(f"runner artifact path is malformed: {label}")
        path = (repo_root / relative).resolve()
        try:
            path.relative_to(repo_root.resolve())
        except ValueError as exc:
            raise BenchmarkContractError(f"runner artifact escapes repository: {label}") from exc
        if not path.is_file():
            raise BenchmarkContractError(f"runner artifact is missing: {path}")
        if _sha256_file(path) != _sha256(record.get("sha256"), label=f"{label} SHA-256"):
            raise BenchmarkContractError(f"runner artifact SHA-256 mismatch: {label}")
        authenticated[str(label)] = path

    required_binding_artifacts = {
        "frozen_manifest_bound_benchmark",
        "frozen_policy",
        "independent_l2_forward_resolver_config",
        "independent_l2_forward_resolver_source",
        "single_step_benchmark_runner_source",
        "graph2edits_runtime_lock",
        "graph2edits_runtime_qualification",
    }
    if not required_binding_artifacts <= set(authenticated):
        missing = sorted(required_binding_artifacts - set(authenticated))
        raise BenchmarkContractError(f"benchmark runner binding misses artifacts: {missing}")

    try:
        manifest_binding = json.loads(authenticated["frozen_manifest_bound_benchmark"].read_text())
        spec = json.loads(authenticated["frozen_policy"].read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkContractError("invalid bound benchmark policy artifact") from exc
    if (
        not isinstance(manifest_binding, dict)
        or manifest_binding.get("schema_version") != BENCHMARK_MANIFEST_BINDING_SCHEMA_VERSION
        or manifest_binding.get("status") != "frozen_manifest_bound_benchmark_not_executed"
    ):
        raise BenchmarkContractError("invalid frozen manifest binding")
    manifest_scope = manifest_binding.get("scope_guards")
    if not isinstance(manifest_scope, Mapping) or any(
        value is not False for value in manifest_scope.values()
    ):
        raise BenchmarkContractError("frozen manifest binding scope guards are unsafe")
    try:
        runtime_qualification = json.loads(
            authenticated["graph2edits_runtime_qualification"].read_text()
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkContractError("invalid Graph2Edits runtime qualification") from exc
    runtime_scope = runtime_qualification.get("scope_guards")
    runtime_authority = runtime_qualification.get("scientific_authority")
    if (
        not isinstance(runtime_scope, Mapping)
        or runtime_scope.get("candidate_backend_activated") is not False
        or runtime_scope.get("frozen_120_target_benchmark_executed") is not False
        or runtime_scope.get("sealed_holdouts_accessed") is not False
    ):
        raise BenchmarkContractError("runtime qualification exceeds smoke-only scope")
    if (
        not isinstance(runtime_authority, Mapping)
        or runtime_authority.get("proposal_only") is not True
        or runtime_authority.get("may_create_route_evidence") is not False
        or runtime_authority.get("may_enter_synthesis_value") is not False
    ):
        raise BenchmarkContractError("runtime qualification has unsafe scientific authority")
    frozen_policy_receipt = manifest_binding.get("frozen_policy")
    if not isinstance(frozen_policy_receipt, Mapping):
        raise BenchmarkContractError("manifest binding lacks frozen policy receipt")
    if frozen_policy_receipt.get("path") != binding_artifacts["frozen_policy"].get("path"):
        raise BenchmarkContractError("runner and manifest binding policy paths disagree")
    if frozen_policy_receipt.get("sha256") != binding_artifacts["frozen_policy"].get("sha256"):
        raise BenchmarkContractError("runner and manifest binding policy hashes disagree")
    if not isinstance(spec, dict) or spec.get("schema_version") != BENCHMARK_SPEC_SCHEMA_VERSION:
        raise BenchmarkContractError("unsupported benchmark specification schema")
    if spec.get("status") != "frozen_specification_not_executed":
        raise BenchmarkContractError("benchmark specification status is not frozen/nonexecuted")

    artifacts = spec.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise BenchmarkContractError("benchmark artifact receipts are missing")
    forbidden = spec.get("forbidden_inputs")
    if not isinstance(forbidden, list) or any(not isinstance(value, str) for value in forbidden):
        raise BenchmarkContractError("benchmark forbidden-input policy is malformed")
    for label, record in artifacts.items():
        if not isinstance(record, Mapping):
            raise BenchmarkContractError(f"benchmark artifact receipt is malformed: {label}")
        relative = record.get("path")
        if not isinstance(relative, str) or not relative:
            raise BenchmarkContractError(f"benchmark artifact path is malformed: {label}")
        if any(relative.startswith(prefix) for prefix in forbidden):
            raise BenchmarkContractError(f"benchmark artifact enters forbidden input: {label}")
        path = (repo_root / relative).resolve()
        try:
            path.relative_to(repo_root.resolve())
        except ValueError as exc:
            raise BenchmarkContractError(f"benchmark artifact escapes repository: {label}") from exc
        if not path.is_file():
            raise BenchmarkContractError(f"benchmark artifact is missing: {path}")
        if _sha256_file(path) != _sha256(record.get("sha256"), label=f"{label} SHA-256"):
            raise BenchmarkContractError(f"benchmark artifact SHA-256 mismatch: {label}")

    resolver_receipt = binding_artifacts.get("independent_l2_forward_resolver_config")
    if not isinstance(resolver_receipt, Mapping):
        raise BenchmarkContractError("benchmark is not bound to the independent L2 resolver")
    lanes = spec.get("lanes")
    if not isinstance(lanes, list):
        raise BenchmarkContractError("benchmark lane records are missing")
    lane_ids = tuple(
        _nonempty(record.get("lane_id"), label="lane_id")
        for record in lanes
        if isinstance(record, Mapping)
    )
    if len(lane_ids) != len(lanes):
        raise BenchmarkContractError("benchmark lane record is malformed")

    target_policy = spec.get("target_manifest_policy")
    if not isinstance(target_policy, Mapping):
        raise BenchmarkContractError("benchmark target-manifest policy is missing")
    if target_policy.get("sealed_holdout_access_forbidden") is not True:
        raise BenchmarkContractError("sealed holdout access must remain forbidden")
    if target_policy.get("known_routes_hidden_from_lane_and_retained_for_scoring") is not True:
        raise BenchmarkContractError("known routes must remain hidden during lane execution")
    stratum_records = spec.get("target_strata")
    if not isinstance(stratum_records, list):
        raise BenchmarkContractError("benchmark strata are missing")
    stratum_counts = tuple(
        (
            _nonempty(record.get("stratum_id"), label="stratum_id"),
            _positive_integer(record.get("count"), label="stratum count"),
        )
        for record in stratum_records
        if isinstance(record, Mapping)
    )
    if len(stratum_counts) != len(stratum_records):
        raise BenchmarkContractError("benchmark stratum record is malformed")

    matched = spec.get("matched_budget")
    if not isinstance(matched, Mapping) or not isinstance(matched.get("per_target"), Mapping):
        raise BenchmarkContractError("benchmark matched budget is missing")
    budget = MatchedTargetBudget.from_dict(matched["per_target"])
    hybrid = matched.get("hybrid_allocation")
    if not isinstance(hybrid, Mapping):
        raise BenchmarkContractError("benchmark hybrid allocation is missing")
    matching = matched.get("matching_policy")
    if not isinstance(matching, Mapping) or any(value is not True for value in matching.values()):
        raise BenchmarkContractError("all matched-budget policies must remain enabled")

    authority = spec.get("scientific_authority")
    if not isinstance(authority, Mapping):
        raise BenchmarkContractError("benchmark authority boundary is missing")
    for field in (
        "proposal_records_are_route_evidence",
        "model_score_may_enter_v_syn",
        "model_score_may_set_evidence_tier",
        "model_score_may_close_route",
        "exact_forward_unique_is_experimental_evidence",
        "operational_screen_pass_is_experimental_evidence",
    ):
        if authority.get(field) is not False:
            raise BenchmarkContractError(f"unsafe benchmark authority: {field}")

    activation = binding.get("activation")
    gates = binding.get("prerequisite_gates")
    if not isinstance(activation, Mapping) or not isinstance(gates, Mapping):
        raise BenchmarkContractError("benchmark activation or gates are malformed")
    if dict(activation) != manifest_binding.get("activation"):
        raise BenchmarkContractError("runner and manifest binding activation disagree")
    if dict(gates) != manifest_binding.get("prerequisite_gates"):
        raise BenchmarkContractError("runner and manifest binding gates disagree")
    development_manifests = manifest_binding.get("development_target_manifest")
    if not isinstance(development_manifests, Mapping):
        raise BenchmarkContractError("bound development manifests are missing")
    lane_manifest_receipt = development_manifests.get("lane_targets")
    hidden_truth_receipt = development_manifests.get("scoring_truth_not_lane_input")
    if not isinstance(lane_manifest_receipt, Mapping) or not isinstance(
        hidden_truth_receipt, Mapping
    ):
        raise BenchmarkContractError("bound target or scoring-truth receipt is missing")
    return FrozenBenchmarkContract(
        spec_sha256=_sha256_file(binding_path),
        policy_sha256=_sha256_file(authenticated["frozen_policy"]),
        manifest_binding_sha256=_sha256_file(authenticated["frozen_manifest_bound_benchmark"]),
        lane_ids=lane_ids,
        target_count=_positive_integer(target_policy.get("target_count"), label="target_count"),
        stratum_counts=stratum_counts,
        selection_seed=_nonnegative_integer(
            target_policy.get("selection_seed"),
            label="selection_seed",
        ),
        budget=budget,
        strict_reserved_hypotheses=_positive_integer(
            hybrid.get("strict_reserved_hypotheses"),
            label="strict_reserved_hypotheses",
        ),
        learned_reserved_hypotheses=_positive_integer(
            hybrid.get("learned_reserved_hypotheses"),
            label="learned_reserved_hypotheses",
        ),
        activation={str(key): bool(value) for key, value in activation.items()},
        prerequisite_gates={str(key): bool(value) for key, value in gates.items()},
        target_manifest_frozen=(
            gates.get("development_target_manifest_frozen") is True
            and manifest_binding.get("manifest_summary", {}).get("target_count")
            == target_policy.get("target_count")
        ),
        target_manifest_sha256=_sha256(
            lane_manifest_receipt.get("sha256"),
            label="target manifest SHA-256",
        ),
        hidden_truth_manifest_sha256=_sha256(
            hidden_truth_receipt.get("sha256"),
            label="hidden truth manifest SHA-256",
        ),
        resolver_config_sha256=_sha256(
            resolver_receipt.get("sha256"),
            label="independent resolver SHA-256",
        ),
    )


@dataclass(frozen=True)
class BenchmarkTarget:
    """Public lane input with no hidden route truth."""

    target_id: str
    primary_stratum: str
    request: ProposalRequest
    secondary_tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _nonempty(self.target_id, label="target_id")
        _nonempty(self.primary_stratum, label="primary_stratum")
        if not isinstance(self.request, ProposalRequest):
            raise BenchmarkContractError("benchmark target requires a typed proposal request")
        if not isinstance(self.secondary_tags, tuple) or any(
            not isinstance(value, str) or not value for value in self.secondary_tags
        ):
            raise BenchmarkContractError("benchmark secondary tags are malformed")


@dataclass(frozen=True)
class BenchmarkTargetManifest:
    """Frozen public target manifest; documented routes are absent."""

    manifest_sha256: str
    selection_seed: int
    targets: tuple[BenchmarkTarget, ...]

    def validate_against(self, contract: FrozenBenchmarkContract) -> None:
        _sha256(self.manifest_sha256, label="target manifest SHA-256")
        if self.manifest_sha256 != contract.target_manifest_sha256:
            raise BenchmarkContractError("target manifest is not the frozen bound artifact")
        if self.selection_seed != contract.selection_seed:
            raise BenchmarkContractError("target manifest selection seed mismatch")
        if len(self.targets) != contract.target_count:
            raise BenchmarkContractError("target manifest count mismatch")
        ids = tuple(value.target_id for value in self.targets)
        if len(set(ids)) != len(ids):
            raise BenchmarkContractError("target IDs must be unique")
        observed = Counter(value.primary_stratum for value in self.targets)
        if observed != Counter(dict(contract.stratum_counts)):
            raise BenchmarkContractError("target manifest stratum counts mismatch")


@dataclass(frozen=True)
class HiddenTruthRecord:
    """Scoring-only exact route identities for one target."""

    target_id: str
    truth_kind: str
    exact_reactant_multisets: tuple[tuple[str, ...], ...] = ()

    def __post_init__(self) -> None:
        _nonempty(self.target_id, label="hidden-truth target_id")
        _nonempty(self.truth_kind, label="hidden-truth truth_kind")
        if not isinstance(self.exact_reactant_multisets, tuple):
            raise BenchmarkContractError("hidden truth reactant multisets must be a tuple")
        canonical = tuple(tuple(sorted(value)) for value in self.exact_reactant_multisets)
        if len(set(canonical)) != len(canonical):
            raise BenchmarkContractError("hidden truth contains duplicate routes")
        exact_kind = "documented_exact_forward_unique_reactant_multiset"
        adversarial_kind = "valid_connected_wrong_handle_role_swap_control"
        if self.truth_kind not in (exact_kind, adversarial_kind):
            raise BenchmarkContractError("unsupported hidden-truth kind")
        if self.truth_kind == exact_kind and not canonical:
            raise BenchmarkContractError("exact route truth requires a reactant multiset")
        if self.truth_kind == adversarial_kind and canonical:
            raise BenchmarkContractError("adversarial truth cannot contain exact reactants")
        object.__setattr__(self, "exact_reactant_multisets", canonical)


@dataclass(frozen=True)
class HiddenTruthManifest:
    """Loaded only after the proposal output ledger is frozen."""

    manifest_sha256: str
    records: tuple[HiddenTruthRecord, ...]

    def validate_target_ids(self, targets: BenchmarkTargetManifest) -> None:
        _sha256(self.manifest_sha256, label="hidden truth SHA-256")
        target_ids = {target.target_id for target in targets.targets}
        truth_ids = tuple(record.target_id for record in self.records)
        if len(set(truth_ids)) != len(truth_ids):
            raise BenchmarkContractError("hidden truth target IDs must be unique")
        if not set(truth_ids) <= target_ids:
            raise BenchmarkContractError("hidden truth contains an unknown target ID")


@dataclass(frozen=True)
class LaneProposalBatch:
    """One lane's bounded proposal output before independent resolution."""

    lane_id: str
    target_id: str
    proposals: tuple[SingleStepRetrosynthesisProposal, ...]
    proposal_sources: tuple[str, ...]
    proposal_source_calls: int
    raw_hypothesis_attempts: int
    invalid_raw_hypotheses: int
    canonical_duplicates: int

    def __post_init__(self) -> None:
        if len(self.proposals) != len(self.proposal_sources):
            raise BenchmarkContractError("proposal-source provenance count mismatch")
        if any(not isinstance(value, SingleStepRetrosynthesisProposal) for value in self.proposals):
            raise BenchmarkContractError("lane emitted a malformed proposal")
        for field in (
            "proposal_source_calls",
            "raw_hypothesis_attempts",
            "invalid_raw_hypotheses",
            "canonical_duplicates",
        ):
            _nonnegative_integer(getattr(self, field), label=field)


class BenchmarkLaneEngine(Protocol):
    """Injected lane engine; hidden truth is absent from this interface."""

    def propose(
        self,
        target: BenchmarkTarget,
        *,
        budget: MatchedTargetBudget,
        strict_reserved_hypotheses: int,
        learned_reserved_hypotheses: int,
    ) -> LaneProposalBatch: ...


class BenchmarkOperationalScreen(Protocol):
    """Independent operational screen after exact forward resolution."""

    def __call__(
        self,
        proposal: SingleStepRetrosynthesisProposal,
        resolution: L2ForwardResolution,
    ) -> OperationalCompatibility: ...


@dataclass(frozen=True)
class ScoredProposalRecord:
    lane_id: str
    target_id: str
    rank: int
    proposal_source: str
    reactant_multiset: tuple[str, ...]
    resolution_status: L2ForwardResolutionStatus
    forward_calls: int
    operational_compatibility: OperationalCompatibility
    accepted: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "lane_id": self.lane_id,
            "target_id": self.target_id,
            "rank": self.rank,
            "proposal_source": self.proposal_source,
            "reactant_multiset": list(self.reactant_multiset),
            "resolution_status": self.resolution_status.value,
            "forward_calls": self.forward_calls,
            "operational_compatibility": self.operational_compatibility.value,
            "accepted": self.accepted,
            "evidence_tier": None,
            "success_probability": None,
            "route_closure_authorized": False,
            "may_enter_synthesis_value": False,
        }


@dataclass(frozen=True)
class LaneTargetOutput:
    lane_id: str
    target_id: str
    proposal_source_calls: int
    raw_hypothesis_attempts: int
    invalid_raw_hypotheses: int
    canonical_duplicates: int
    forward_calls: int
    operational_screen_calls: int
    accepted_count: int
    records: tuple[ScoredProposalRecord, ...]

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
            "records": [value.to_dict() for value in self.records],
        }


@dataclass(frozen=True)
class FrozenLaneOutputLedger:
    contract_sha256: str
    target_manifest_sha256: str
    outputs: tuple[LaneTargetOutput, ...]

    @property
    def ledger_sha256(self) -> str:
        return _sha256_payload(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value = {
            "schema_version": BENCHMARK_OUTPUT_SCHEMA_VERSION,
            "contract_sha256": self.contract_sha256,
            "target_manifest_sha256": self.target_manifest_sha256,
            "outputs": [output.to_dict() for output in self.outputs],
            "hidden_truth_loaded_during_lane_execution": False,
            "route_evidence_created": False,
            "v_syn_modified": False,
        }
        if include_hash:
            value["ledger_sha256"] = self.ledger_sha256
        return value


class FrozenProposalLaneBenchmarkRunner:
    """Execute only after all frozen gates pass; currently preflight-only."""

    def __init__(
        self,
        *,
        contract: FrozenBenchmarkContract,
        resolver: IndependentL2ForwardResolver,
        lane_engines: Mapping[str, BenchmarkLaneEngine],
        operational_screen: BenchmarkOperationalScreen,
    ) -> None:
        if not isinstance(contract, FrozenBenchmarkContract):
            raise BenchmarkContractError("runner contract is malformed")
        if not isinstance(resolver, IndependentL2ForwardResolver):
            raise BenchmarkContractError("runner requires the independent L2 resolver")
        if resolver.resolver_config_sha256 != contract.resolver_config_sha256:
            raise BenchmarkContractError("runner resolver hash disagrees with frozen contract")
        if set(lane_engines) != set(EXPECTED_LANES):
            raise BenchmarkContractError("runner requires exactly the three frozen lanes")
        if not callable(operational_screen):
            raise BenchmarkContractError("runner operational screen must be callable")
        self._contract = contract
        self._resolver = resolver
        self._lane_engines = dict(lane_engines)
        self._operational_screen = operational_screen

    @property
    def contract(self) -> FrozenBenchmarkContract:
        return self._contract

    def execute(self, targets: BenchmarkTargetManifest) -> FrozenLaneOutputLedger:
        """Execute public lanes only; hidden truth is not accepted by this method."""

        self._contract.assert_execution_ready()
        targets.validate_against(self._contract)
        outputs: list[LaneTargetOutput] = []
        for target in targets.targets:
            for lane_id in self._contract.lane_ids:
                batch = self._lane_engines[lane_id].propose(
                    target,
                    budget=self._contract.budget,
                    strict_reserved_hypotheses=self._contract.strict_reserved_hypotheses,
                    learned_reserved_hypotheses=self._contract.learned_reserved_hypotheses,
                )
                outputs.append(self._resolve_batch(target, lane_id, batch))
        return FrozenLaneOutputLedger(
            contract_sha256=self._contract.spec_sha256,
            target_manifest_sha256=targets.manifest_sha256,
            outputs=tuple(outputs),
        )

    def execute_from_paths(
        self,
        *,
        target_manifest_path: Path,
        target_loader: Callable[[Path], BenchmarkTargetManifest],
    ) -> FrozenLaneOutputLedger:
        """Check all gates before the caller-provided target loader can run."""

        self._contract.assert_execution_ready()
        targets = target_loader(target_manifest_path)
        return self.execute(targets)

    def _resolve_batch(
        self,
        target: BenchmarkTarget,
        lane_id: str,
        batch: LaneProposalBatch,
    ) -> LaneTargetOutput:
        budget = self._contract.budget
        if batch.lane_id != lane_id or batch.target_id != target.target_id:
            raise BenchmarkContractError("lane output target or lane identity mismatch")
        if batch.proposal_source_calls > budget.proposal_source_calls:
            raise BenchmarkContractError("lane exceeded proposal-source-call budget")
        if batch.raw_hypothesis_attempts > budget.raw_hypothesis_attempts:
            raise BenchmarkContractError("lane exceeded raw-hypothesis budget")
        if batch.invalid_raw_hypotheses > batch.raw_hypothesis_attempts:
            raise BenchmarkContractError("invalid hypotheses exceed raw attempts")
        if batch.canonical_duplicates > batch.raw_hypothesis_attempts:
            raise BenchmarkContractError("duplicate hypotheses exceed raw attempts")
        if len(batch.proposals) > budget.canonical_hypotheses_considered:
            raise BenchmarkContractError("lane exceeded canonical-hypothesis budget")
        identities = tuple(tuple(sorted(value.reactant_smiles)) for value in batch.proposals)
        if len(set(identities)) != len(identities):
            raise BenchmarkContractError("lane emitted canonical duplicate proposals")
        ranks = tuple(value.rank for value in batch.proposals)
        if any(rank < 1 for rank in ranks) or len(set(ranks)) != len(ranks):
            raise BenchmarkContractError("lane proposal ranks must be unique positive integers")

        forward_calls = 0
        operational_calls = 0
        accepted_count = 0
        records: list[ScoredProposalRecord] = []
        for proposal, source in zip(batch.proposals, batch.proposal_sources, strict=True):
            remaining_forward = budget.forward_verifier_calls - forward_calls
            if remaining_forward <= 0:
                break
            resolution = self._resolver.resolve(
                proposal,
                maximum_forward_calls=remaining_forward,
            )
            forward_calls += resolution.forward_calls
            compatibility = OperationalCompatibility.NOT_ASSESSED
            accepted = False
            if resolution.status is L2ForwardResolutionStatus.EXACT_UNIQUE:
                if operational_calls >= budget.operational_screen_calls:
                    break
                compatibility = self._operational_screen(proposal, resolution)
                if not isinstance(compatibility, OperationalCompatibility):
                    raise BenchmarkContractError("operational screen returned an invalid state")
                operational_calls += 1
                accepted = compatibility is OperationalCompatibility.PASSED
                if accepted and accepted_count < budget.accepted_proposal_cap:
                    accepted_count += 1
                elif accepted:
                    accepted = False
            records.append(
                ScoredProposalRecord(
                    lane_id=lane_id,
                    target_id=target.target_id,
                    rank=proposal.rank,
                    proposal_source=source,
                    reactant_multiset=tuple(sorted(proposal.reactant_smiles)),
                    resolution_status=resolution.status,
                    forward_calls=resolution.forward_calls,
                    operational_compatibility=compatibility,
                    accepted=accepted,
                )
            )
        if forward_calls > budget.forward_verifier_calls:
            raise BenchmarkContractError("lane exceeded forward-verifier budget")
        if operational_calls > budget.operational_screen_calls:
            raise BenchmarkContractError("lane exceeded operational-screen budget")
        return LaneTargetOutput(
            lane_id=lane_id,
            target_id=target.target_id,
            proposal_source_calls=batch.proposal_source_calls,
            raw_hypothesis_attempts=batch.raw_hypothesis_attempts,
            invalid_raw_hypotheses=batch.invalid_raw_hypotheses,
            canonical_duplicates=batch.canonical_duplicates,
            forward_calls=forward_calls,
            operational_screen_calls=operational_calls,
            accepted_count=accepted_count,
            records=tuple(records),
        )


def score_frozen_lane_outputs(
    *,
    contract: FrozenBenchmarkContract,
    targets: BenchmarkTargetManifest,
    outputs: FrozenLaneOutputLedger,
    hidden_truth: HiddenTruthManifest,
) -> dict[str, Any]:
    """Score after output freeze; hidden truth never enters proposal execution."""

    targets.validate_against(contract)
    hidden_truth.validate_target_ids(targets)
    if hidden_truth.manifest_sha256 != contract.hidden_truth_manifest_sha256:
        raise BenchmarkContractError("hidden truth is not the frozen scoring artifact")
    if outputs.contract_sha256 != contract.spec_sha256:
        raise BenchmarkContractError("output ledger contract hash mismatch")
    if outputs.target_manifest_sha256 != targets.manifest_sha256:
        raise BenchmarkContractError("output ledger target hash mismatch")
    truth_by_target = {record.target_id: record for record in hidden_truth.records}
    output_keys = tuple((row.lane_id, row.target_id) for row in outputs.outputs)
    if len(set(output_keys)) != len(output_keys):
        raise BenchmarkContractError("output ledger contains duplicate lane-target rows")
    expected_output_keys = {
        (lane_id, target.target_id) for lane_id in contract.lane_ids for target in targets.targets
    }
    if set(output_keys) != expected_output_keys:
        raise BenchmarkContractError("output ledger does not cover the frozen lane-target grid")
    output_by_lane_target = {(row.lane_id, row.target_id): row for row in outputs.outputs}
    metrics: dict[str, Any] = {}
    for lane_id in contract.lane_ids:
        top_k = {1: 0, 5: 0, 10: 0, 20: 0}
        truth_denominator = 0
        exact_unique = 0
        proposals_considered = 0
        operational_passes = 0
        operational_denominator = 0
        accepted = 0
        adversarial_accepted = 0
        for target in targets.targets:
            row = output_by_lane_target[(lane_id, target.target_id)]
            truth = truth_by_target.get(target.target_id)
            if truth is not None and truth.exact_reactant_multisets:
                truth_denominator += 1
                accepted_truth = set(truth.exact_reactant_multisets)
                for k in top_k:
                    if any(
                        record.reactant_multiset in accepted_truth
                        for record in row.records
                        if record.rank <= k
                    ):
                        top_k[k] += 1
            proposals_considered += len(row.records)
            exact_unique += sum(
                record.resolution_status is L2ForwardResolutionStatus.EXACT_UNIQUE
                for record in row.records
            )
            operational_records = [
                record
                for record in row.records
                if record.operational_compatibility is not OperationalCompatibility.NOT_ASSESSED
            ]
            operational_denominator += len(operational_records)
            operational_passes += sum(
                record.operational_compatibility is OperationalCompatibility.PASSED
                for record in operational_records
            )
            accepted += row.accepted_count
            if target.primary_stratum == "adversarial_incompatibles":
                adversarial_accepted += row.accepted_count
        metrics[lane_id] = {
            "known_route_top_k": {
                str(k): {
                    "numerator": top_k[k],
                    "denominator": truth_denominator,
                    "fraction": None if not truth_denominator else top_k[k] / truth_denominator,
                }
                for k in sorted(top_k)
            },
            "exact_forward_unique": {
                "numerator": exact_unique,
                "denominator": proposals_considered,
                "fraction": (
                    None if not proposals_considered else exact_unique / proposals_considered
                ),
            },
            "operational_screen_yield": {
                "numerator": operational_passes,
                "denominator": operational_denominator,
                "fraction": (
                    None
                    if not operational_denominator
                    else operational_passes / operational_denominator
                ),
            },
            "accepted_proposals": accepted,
            "adversarial_incompatible_accepted_proposals": adversarial_accepted,
        }
    result = {
        "schema_version": "forge.single_step_proposal_benchmark_score.v1",
        "contract_sha256": contract.spec_sha256,
        "target_manifest_sha256": targets.manifest_sha256,
        "output_ledger_sha256": outputs.ledger_sha256,
        "hidden_truth_manifest_sha256": hidden_truth.manifest_sha256,
        "metrics": metrics,
        "model_score_used_for_scoring": False,
        "route_evidence_created": False,
        "v_syn_modified": False,
        "candidate_selection_authorized": False,
    }
    result["score_sha256"] = _sha256_payload(result)
    return result


__all__ = [
    "BenchmarkContractError",
    "BenchmarkExecutionBlockedError",
    "BenchmarkTarget",
    "BenchmarkTargetManifest",
    "FrozenBenchmarkContract",
    "FrozenLaneOutputLedger",
    "FrozenProposalLaneBenchmarkRunner",
    "HiddenTruthManifest",
    "HiddenTruthRecord",
    "LaneProposalBatch",
    "LaneTargetOutput",
    "MatchedTargetBudget",
    "ScoredProposalRecord",
    "load_frozen_benchmark_contract",
    "score_frozen_lane_outputs",
]
