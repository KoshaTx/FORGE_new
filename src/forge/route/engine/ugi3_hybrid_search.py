"""Bounded fail-closed hybrid evidence search for Ugi component routes.

This diagnostic composes the frozen exact-evidence source with exact-identity
retrieval and non-closing family/provenance proposal channels.  The current
upstream registry is exact-source-forward-verification-only, so it admits no
general substrate template.  Family projections may prioritize evidence work
but cannot produce a complete route assessment.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import platform
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase

from forge.data.r1_prime_audit import sha256_bytes, sha256_file
from forge.route.assessment.ugi3_production_registry_route_readiness import (
    ACCEPTED_TERMINAL,
    EXACT_CLOSED,
    EXACT_OPEN,
    FAMILY_ONLY,
    MISSING_KNOWLEDGE,
    OUTSIDE_SUPPORT,
    PROVENANCE_ONLY,
)
from forge.route.engine.planner import (
    AssessmentOutcome,
    AvailabilityState,
    EvidenceRecord,
    EvidenceTier,
    ForwardVerificationState,
    KnowledgeDisposition,
    KnowledgeResult,
    PlannerBudgetLedger,
    PlannerBudgetLimits,
    RecursiveRouteAssessor,
    RouteStepProposal,
    RouteTarget,
)
from forge.route.engine.planner_cache import (
    CachedRoutePlanner,
    FilePlannerCache,
    PlannerCacheContext,
)
from forge.route.engine.qualified_forward import (
    QualifiedForwardError,
    load_qualified_forward_reaction,
    unique_forward_products,
)
from forge.route.sources.ugi3_exact_evidence_source import (
    ExactEvidenceOnlyUgi3Source,
    load_exact_evidence_only_source,
)

CONFIG_SCHEMA_VERSION = "phase1_ugi3_hybrid_search_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi3_hybrid_search.v1"
LEDGER_SCHEMA_VERSION = "phase1_ugi3_hybrid_search_assessments.v1"

SUPPORTED_CATEGORIES = {
    ACCEPTED_TERMINAL,
    EXACT_CLOSED,
    EXACT_OPEN,
    FAMILY_ONLY,
    PROVENANCE_ONLY,
    OUTSIDE_SUPPORT,
    MISSING_KNOWLEDGE,
}


class Ugi3HybridSearchError(ValueError):
    """Raised when hybrid-search evidence cannot be composed fail-closed."""


class SearchChannel(str, Enum):
    """Explicit search/evidence channel assigned to one admitted component."""

    EXACT_BASELINE = "exact_baseline"
    EXACT_IDENTITY_RETRIEVAL = "exact_identity_retrieval"
    EXACT_EVIDENCE_L3_OPEN = "exact_evidence_l3_open"
    QUALIFIED_TEMPLATE = "qualified_template"
    FAMILY_PROJECTION = "family_projection"
    PROVENANCE_PRIORITY = "provenance_priority"
    NO_ADMITTED_ROUTE = "no_admitted_route"


def _stable_json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _sha256_payload(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode()).hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise Ugi3HybridSearchError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise Ugi3HybridSearchError(f"{label} must be a JSON object")
    return value


def _read_csv(path: Path, *, label: str) -> list[dict[str, str]]:
    try:
        with gzip.open(path, "rt", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise Ugi3HybridSearchError(f"{label} has no header")
            return list(reader)
    except (OSError, csv.Error) as exc:
        raise Ugi3HybridSearchError(f"could not read {label}: {path}") from exc


def _canonical_constitution(smiles: Any, *, label: str) -> str:
    if not isinstance(smiles, str) or not smiles:
        raise Ugi3HybridSearchError(f"{label} must be nonempty SMILES")
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise Ugi3HybridSearchError(f"{label} contains invalid SMILES")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def _parse_utc(value: Any, *, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise Ugi3HybridSearchError(f"{label} must be an ISO-8601 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise Ugi3HybridSearchError(f"{label} is invalid") from exc
    if parsed.tzinfo != timezone.utc:
        raise Ugi3HybridSearchError(f"{label} must resolve to UTC")
    return parsed


def _portable(path: Path, *, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _gzip_json_bytes(value: Any) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as compressed:
        compressed.write((_stable_json(value) + "\n").encode())
    return output.getvalue()


def _validate_expected(observed: Any, expected: Any, *, label: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(observed, dict) or set(observed) != set(expected):
            raise Ugi3HybridSearchError(f"{label} fields mismatch")
        for key, value in expected.items():
            _validate_expected(observed[key], value, label=f"{label}.{key}")
        return
    if observed != expected:
        raise Ugi3HybridSearchError(
            f"{label} mismatch: expected {expected!r}, observed {observed!r}"
        )


@dataclass(frozen=True)
class HybridSearchTarget:
    """One L1-admitted production component and its search classification."""

    component_id: str
    target: RouteTarget
    original_component_id: str | None
    readiness_category: str
    readiness_complete: bool
    channel: SearchChannel
    proposal_only: bool
    priority_only: bool


@dataclass(frozen=True)
class _RetrievedExactRoute:
    component_id: str
    target: RouteTarget
    scoped_role: str
    proposal: RouteStepProposal
    terminal_target: RouteTarget
    terminal_evidence: EvidenceRecord


class BoundedHybridUgi3Source:
    """Compose exact lookup, one exact retrieval and non-closing search signals."""

    def __init__(
        self,
        *,
        exact_source: ExactEvidenceOnlyUgi3Source,
        targets: Mapping[tuple[str, str], HybridSearchTarget],
        target_evidence: Mapping[tuple[str, str], EvidenceRecord],
        retrieved_route: _RetrievedExactRoute,
    ):
        self._exact_source = exact_source
        self._targets = dict(targets)
        self._target_evidence = dict(target_evidence)
        self._retrieved_route = retrieved_route
        self._admitted = tuple(sorted(self._targets.values(), key=lambda item: item.component_id))

    @property
    def admitted_components(self) -> tuple[HybridSearchTarget, ...]:
        return self._admitted

    def lookup(self, target: RouteTarget) -> KnowledgeResult:
        canonical = _canonical_constitution(target.canonical_smiles, label="hybrid route target")
        retrieved = self._retrieved_route
        if target.role == retrieved.scoped_role:
            if canonical == retrieved.terminal_target.canonical_smiles:
                return KnowledgeResult(
                    disposition=KnowledgeDisposition.TERMINAL,
                    evidence=(retrieved.terminal_evidence,),
                    detail="retrieved exact route terminates at frozen L3 evidence",
                )
            return KnowledgeResult(
                disposition=KnowledgeDisposition.OUTSIDE_SUPPORT,
                evidence=(),
                detail="retrieved route requested an unindexed upstream material",
            )

        baseline = self._exact_source.lookup(target)
        if baseline.disposition not in {
            KnowledgeDisposition.OUTSIDE_SUPPORT,
            KnowledgeDisposition.INVALID_INPUT,
        }:
            return baseline
        admitted = self._targets.get((target.role, canonical))
        if admitted is None:
            return baseline
        evidence = self._target_evidence[(target.role, canonical)]
        if admitted.component_id == retrieved.component_id:
            return KnowledgeResult(
                disposition=KnowledgeDisposition.EXPAND,
                evidence=retrieved.proposal.evidence,
                proposal=retrieved.proposal,
                detail="exact-identity source route retrieved and uniquely forward verified",
            )
        category = admitted.readiness_category
        if category == FAMILY_ONLY:
            return KnowledgeResult(
                disposition=KnowledgeDisposition.MISSING_KNOWLEDGE,
                evidence=(evidence,),
                detail="family projection may propose but cannot close without exact evidence",
            )
        if category in {MISSING_KNOWLEDGE, PROVENANCE_ONLY, EXACT_OPEN}:
            return KnowledgeResult(
                disposition=KnowledgeDisposition.MISSING_KNOWLEDGE,
                evidence=(evidence,),
                detail="search priority retained but closing evidence is missing",
            )
        if category == OUTSIDE_SUPPORT:
            return KnowledgeResult(
                disposition=KnowledgeDisposition.OUTSIDE_SUPPORT,
                evidence=(evidence,),
                detail="no admitted upstream route exists in the declared search support",
            )
        raise Ugi3HybridSearchError(
            f"non-baseline component {admitted.component_id} has unsupported closure category"
        )


def _l3_state(
    *,
    terminal_status: str,
    accessed_utc: str,
    assessment_as_of_utc: str,
    expiry_days: int,
    unavailable_statuses: frozenset[str],
) -> AvailabilityState:
    accessed = _parse_utc(accessed_utc, label="retrieval L3 accessed_utc")
    as_of = _parse_utc(assessment_as_of_utc, label="assessment_as_of_utc")
    if as_of < accessed:
        raise Ugi3HybridSearchError("hybrid assessment cannot precede retrieved L3 evidence")
    if as_of > accessed + timedelta(days=expiry_days):
        return AvailabilityState.EXPIRED
    if terminal_status == "current_item_level_procurement_closed":
        return AvailabilityState.CURRENT_CLOSED
    if terminal_status in unavailable_statuses:
        return AvailabilityState.UNAVAILABLE
    return AvailabilityState.UNASSESSED


def _search_channel(
    *,
    readiness_category: str,
    original_tier: str | None,
    exact_transfer: bool,
) -> tuple[SearchChannel, bool, bool]:
    if exact_transfer:
        return SearchChannel.EXACT_IDENTITY_RETRIEVAL, False, False
    if original_tier in {
        "accepted_terminal_l3_closed",
        "exact_source_l2_verified_l3_closed",
    }:
        return SearchChannel.EXACT_BASELINE, False, False
    if original_tier == "exact_source_l2_verified_l3_open" or readiness_category == EXACT_OPEN:
        return SearchChannel.EXACT_EVIDENCE_L3_OPEN, False, False
    if readiness_category == FAMILY_ONLY:
        return SearchChannel.FAMILY_PROJECTION, True, False
    if readiness_category in {MISSING_KNOWLEDGE, PROVENANCE_ONLY} or original_tier == (
        "unresolved_component"
    ):
        return SearchChannel.PROVENANCE_PRIORITY, False, True
    if readiness_category == OUTSIDE_SUPPORT:
        return SearchChannel.NO_ADMITTED_ROUTE, False, False
    raise Ugi3HybridSearchError(
        f"cannot assign a search channel to readiness category {readiness_category!r}"
    )


def load_bounded_hybrid_source(
    *,
    exact_source: ExactEvidenceOnlyUgi3Source,
    readiness_ledger_path: Path,
    readiness_result_path: Path,
    readiness_config_path: Path,
    transfer_ledger_path: Path,
    transfer_result_path: Path,
    transfer_config_path: Path,
    upstream_registry_path: Path,
    oxidation_variant_path: Path,
    assessment_as_of_utc: str,
    retrieval_l3_expiry_days: int,
    unavailable_l3_statuses: frozenset[str],
    source_locators: Mapping[str, str] | None = None,
) -> tuple[BoundedHybridUgi3Source, dict[str, Any]]:
    """Load and independently verify the bounded hybrid evidence channels."""

    locators = {
        "readiness_ledger": str(readiness_ledger_path),
        "transfer_ledger": str(transfer_ledger_path),
        **(dict(source_locators) if source_locators is not None else {}),
    }
    hashes = {
        "readiness_ledger": sha256_file(readiness_ledger_path),
        "transfer_ledger": sha256_file(transfer_ledger_path),
    }
    readiness_result = _load_json(readiness_result_path, label="route-readiness result")
    if (
        readiness_result.get("artifacts", {}).get("component_ledger_sha256")
        != hashes["readiness_ledger"]
    ):
        raise Ugi3HybridSearchError("route-readiness result does not own its ledger")
    transfer_result = _load_json(transfer_result_path, label="hydrophobic-transfer result")
    transfer_artifact = transfer_result.get("artifacts", {}).get(
        "hydrophobic_motif_transfer_ledger.csv.gz",
        {},
    )
    if transfer_artifact.get("sha256") != hashes["transfer_ledger"]:
        raise Ugi3HybridSearchError("hydrophobic-transfer result does not own its ledger")
    readiness_config = _load_json(readiness_config_path, label="route-readiness config")
    transfer_config = _load_json(transfer_config_path, label="hydrophobic-transfer config")
    upstream_registry = _load_json(upstream_registry_path, label="upstream reaction registry")
    if upstream_registry.get("scope") != "exact_source_forward_verification_only":
        raise Ugi3HybridSearchError("unexpected upstream reaction-registry scope")
    reactions = upstream_registry.get("reactions")
    if not isinstance(reactions, list):
        raise Ugi3HybridSearchError("upstream registry has no reactions")
    qualified_templates = [
        reaction
        for reaction in reactions
        if isinstance(reaction, dict)
        and reaction.get("status") == "qualified_for_substrate_scope_enumeration"
    ]
    if qualified_templates:
        raise Ugi3HybridSearchError(
            "qualified template records require a separately frozen substrate-scope contract"
        )

    readiness_rows = _read_csv(readiness_ledger_path, label="route-readiness ledger")
    transfer_rows = _read_csv(transfer_ledger_path, label="hydrophobic-transfer ledger")
    original_by_id = {item.component_id: item for item in exact_source.admitted_components}
    transfer_policy = readiness_config.get("exact_transfer_verification")
    if not isinstance(transfer_policy, dict):
        raise Ugi3HybridSearchError("readiness config lacks exact-transfer policy")
    record_id = transfer_policy.get("source_record_id")
    transfer_matches = [row for row in transfer_rows if row.get("record_id") == record_id]
    if len(transfer_matches) != 1:
        raise Ugi3HybridSearchError("exact retrieval source must resolve exactly once")
    transfer_row = transfer_matches[0]
    required = transfer_policy.get("required_source_fields")
    if not isinstance(required, dict) or any(
        transfer_row.get(field) != value for field, value in required.items()
    ):
        raise Ugi3HybridSearchError("exact retrieval source evidence is incomplete")
    reactant_field = transfer_policy.get("reactant_field")
    product_field = transfer_policy.get("product_field")
    if not isinstance(reactant_field, str) or not isinstance(product_field, str):
        raise Ugi3HybridSearchError("exact retrieval fields are invalid")
    reactant = _canonical_constitution(
        transfer_row.get(reactant_field),
        label="retrieved exact reactant",
    )
    product = _canonical_constitution(
        transfer_row.get(product_field),
        label="retrieved exact product",
    )
    try:
        compiled = load_qualified_forward_reaction(
            upstream_registry_path,
            oxidation_variant_path,
            reaction_id=str(transfer_policy.get("reaction_id")),
        )
        forward_products = unique_forward_products(
            compiled,
            [reactant],
            max_products=int(transfer_policy.get("max_products", 0)),
            isomeric_smiles=False,
        )
    except (QualifiedForwardError, TypeError, ValueError) as exc:
        raise Ugi3HybridSearchError("retrieved exact route failed forward verification") from exc
    if forward_products != (product,):
        raise Ugi3HybridSearchError("retrieved route did not uniquely reproduce its target")

    exact_transfer_rows = [
        row
        for row in readiness_rows
        if row.get("e2_route_closed_input_flag") == "true"
        and row.get("constitutional_match_to_original_93") == "false"
    ]
    if len(exact_transfer_rows) != 1:
        raise Ugi3HybridSearchError("one new E2 exact retrieval must be admitted")
    exact_transfer_row = exact_transfer_rows[0]
    if (
        exact_transfer_row.get("role") != transfer_policy.get("registry_role")
        or _canonical_constitution(
            exact_transfer_row.get("canonical_smiles"),
            label="retrieved registry target",
        )
        != product
        or exact_transfer_row.get("evidence_category") != EXACT_CLOSED
    ):
        raise Ugi3HybridSearchError("retrieved route and readiness target disagree")
    program_id = transfer_row.get("proposed_program_id")
    programs = transfer_config.get("programs")
    if not isinstance(programs, dict) or not isinstance(programs.get(program_id), dict):
        raise Ugi3HybridSearchError("retrieved exact program metadata is missing")
    program = programs[program_id]
    terminal_metadata = program.get("exact_route_evidence", {}).get("terminal_evidence")
    if not isinstance(terminal_metadata, dict):
        raise Ugi3HybridSearchError("retrieved exact route lacks terminal evidence")
    availability = _l3_state(
        terminal_status=transfer_row.get("terminal_status", ""),
        accessed_utc=terminal_metadata.get("accessed_utc"),
        assessment_as_of_utc=assessment_as_of_utc,
        expiry_days=retrieval_l3_expiry_days,
        unavailable_statuses=unavailable_l3_statuses,
    )
    component_id = exact_transfer_row["component_id"]
    role = exact_transfer_row["role"]
    target = RouteTarget(role=role, canonical_smiles=product)
    scoped_role = f"ugi3_retrieved_upstream_material:{component_id}"
    terminal_target = RouteTarget(role=scoped_role, canonical_smiles=reactant)
    exact_evidence = EvidenceRecord(
        evidence_id=f"exact-retrieval:{record_id}",
        tier=EvidenceTier.EXACT_SOURCE,
        source_sha256=hashes["transfer_ledger"],
        source_locator=f"{locators['transfer_ledger']}#record_id={record_id}",
        exact_substrate=True,
        forward_verification=ForwardVerificationState.VERIFIED_EXACT_PRODUCT_UNIQUE,
        availability=AvailabilityState.UNASSESSED,
    )
    terminal_evidence = EvidenceRecord(
        evidence_id=f"retrieved-terminal:{record_id}",
        tier=EvidenceTier.ACCEPTED_TERMINAL,
        source_sha256=hashes["transfer_ledger"],
        source_locator=f"{locators['transfer_ledger']}#record_id={record_id}:terminal",
        exact_substrate=True,
        forward_verification=ForwardVerificationState.NOT_APPLICABLE,
        availability=availability,
    )
    retrieved_route = _RetrievedExactRoute(
        component_id=component_id,
        target=target,
        scoped_role=scoped_role,
        proposal=RouteStepProposal(
            reaction_id=compiled.reaction_id,
            reactants=(terminal_target,),
            evidence=(exact_evidence,),
            forward_product_count=1,
            verifier_calls_required=1,
            product_candidates_considered=1,
        ),
        terminal_target=terminal_target,
        terminal_evidence=terminal_evidence,
    )

    targets: dict[tuple[str, str], HybridSearchTarget] = {}
    target_evidence: dict[tuple[str, str], EvidenceRecord] = {}
    channel_counts: Counter[str] = Counter()
    for row in readiness_rows:
        component_id = row.get("component_id", "")
        role = row.get("role", "")
        canonical = _canonical_constitution(
            row.get("canonical_smiles"),
            label=f"readiness component {component_id}",
        )
        category = row.get("evidence_category", "")
        if not component_id or category not in SUPPORTED_CATEGORIES:
            raise Ugi3HybridSearchError("readiness record has unsupported identity or category")
        original_id = row.get("original_component_id") or None
        original = original_by_id.get(original_id) if original_id is not None else None
        if original_id is not None and original is None:
            raise Ugi3HybridSearchError("readiness record references an unknown original component")
        channel, proposal_only, priority_only = _search_channel(
            readiness_category=category,
            original_tier=None if original is None else original.existing_evidence_tier,
            exact_transfer=component_id == retrieved_route.component_id,
        )
        record = HybridSearchTarget(
            component_id=component_id,
            target=RouteTarget(role=role, canonical_smiles=canonical),
            original_component_id=original_id,
            readiness_category=category,
            readiness_complete=row.get("route_complete_component") == "true",
            channel=channel,
            proposal_only=proposal_only,
            priority_only=priority_only,
        )
        key = (role, canonical)
        if key in targets:
            raise Ugi3HybridSearchError("duplicate admitted role-component constitution")
        targets[key] = record
        channel_counts[channel.value] += 1
        tier = (
            EvidenceTier.FAMILY_PROJECTED
            if category == FAMILY_ONLY
            else EvidenceTier.PROVENANCE_ONLY
        )
        if category == EXACT_OPEN:
            tier = EvidenceTier.EXACT_SOURCE
        target_evidence[key] = EvidenceRecord(
            evidence_id=f"hybrid-readiness:{component_id}:{category}",
            tier=tier,
            source_sha256=hashes["readiness_ledger"],
            source_locator=f"{locators['readiness_ledger']}#component_id={component_id}",
            exact_substrate=False,
            forward_verification=ForwardVerificationState.NOT_RUN,
            availability=AvailabilityState.UNASSESSED,
        )
    source = BoundedHybridUgi3Source(
        exact_source=exact_source,
        targets=targets,
        target_evidence=target_evidence,
        retrieved_route=retrieved_route,
    )
    metadata = {
        "exact_retrieval_component_id": retrieved_route.component_id,
        "exact_retrieval_source_record_id": record_id,
        "exact_retrieval_forward_products": list(forward_products),
        "exact_retrieval_l3_state": availability.value,
        "admitted_qualified_templates": len(qualified_templates),
        "channels": dict(sorted(channel_counts.items())),
    }
    return source, metadata


def _cache_context(
    *,
    config: Mapping[str, Any],
    input_hashes: Mapping[str, str],
    budget_limits: PlannerBudgetLimits,
) -> PlannerCacheContext:
    policy = config.get("search_policy")
    if not isinstance(policy, dict):
        raise Ugi3HybridSearchError("search_policy must be an object")
    l3 = policy.get("combined_l3_context")
    if not isinstance(l3, dict):
        raise Ugi3HybridSearchError("combined_l3_context must be an object")
    planner_sha = _sha256_payload(
        {
            "planner": input_hashes["planner_contract_source"],
            "exact_adapter": input_hashes["exact_adapter_source"],
            "hybrid_adapter": input_hashes["hybrid_adapter_source"],
        }
    )
    l3_sha = _sha256_payload(
        {
            "original_terminal_procurement": input_hashes["terminal_procurement"],
            "retrieved_terminal_evidence": input_hashes["transfer_ledger"],
        }
    )
    return PlannerCacheContext(
        planner_id="ugi3_bounded_hybrid_search_diagnostic_v1",
        planner_sha256=planner_sha,
        search_policy_sha256=_sha256_payload(policy),
        value_policy_sha256=_sha256_payload(
            {"diagnostic_only": True, "synthesis_value": "not_computed"}
        ),
        l1_reaction_sha256=input_hashes["l1_variant"],
        upstream_reaction_registry_sha256=input_hashes["upstream_registry"],
        variant_registry_sha256=input_hashes["oxidation_variant"],
        verifier_sha256=_sha256_payload(
            {
                "qualified_forward_source": input_hashes["qualified_forward_source"],
                "readiness_result": input_hashes["readiness_result"],
            }
        ),
        l3_snapshot_sha256=l3_sha,
        l3_region=str(l3.get("region")),
        l3_accessed_at_utc=str(l3.get("latest_accessed_utc")),
        l3_expires_at_utc=str(l3.get("earliest_expires_utc")),
        software_versions=tuple(
            sorted(
                (
                    ("python", platform.python_version()),
                    ("rdkit", rdBase.rdkitVersion),
                )
            )
        ),
        identity_policy="canonical_constitutional_smiles",
        stereochemistry_policy="phase1_stereo_free",
        budget_limits=budget_limits,
    )


def build_hybrid_search_diagnostic(
    *,
    config_path: Path,
    input_paths: Mapping[str, Path],
    cache_root: Path,
) -> tuple[dict[str, Any], bytes]:
    """Assess all 424 admitted components under one bounded hybrid policy."""

    config = _load_json(config_path, label="hybrid-search config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise Ugi3HybridSearchError("unsupported hybrid-search config schema")
    configured_inputs = config.get("inputs")
    if not isinstance(configured_inputs, dict) or set(configured_inputs) != set(input_paths):
        raise Ugi3HybridSearchError("configured and supplied hybrid inputs differ")
    input_hashes: dict[str, str] = {}
    for name, path in input_paths.items():
        configured = configured_inputs[name]
        if not isinstance(configured, dict):
            raise Ugi3HybridSearchError(f"input {name} must be an object")
        observed = sha256_file(path)
        if observed != configured.get("expected_sha256"):
            raise Ugi3HybridSearchError(
                f"{name} hash mismatch: expected {configured.get('expected_sha256')}, "
                f"observed {observed}"
            )
        input_hashes[name] = observed
    exact_result = _load_json(input_paths["exact_evidence_result"], label="exact-evidence result")
    if (
        exact_result.get("artifacts", {}).get("assessment_ledger_sha256")
        != input_hashes["exact_evidence_ledger"]
    ):
        raise Ugi3HybridSearchError("exact-evidence result does not own its ledger")
    if exact_result.get("summary", {}).get("tier_crosswalk_agreement") != 93:
        raise Ugi3HybridSearchError("exact-evidence prerequisite is not qualified")

    policy = config.get("search_policy")
    if not isinstance(policy, dict):
        raise Ugi3HybridSearchError("search_policy must be an object")
    unavailable = policy.get("unavailable_l3_statuses")
    if not isinstance(unavailable, list) or any(
        not isinstance(value, str) for value in unavailable
    ):
        raise Ugi3HybridSearchError("unavailable_l3_statuses must be a list of strings")
    budget_limits = PlannerBudgetLimits.from_dict(policy.get("per_component_budget"))
    root = config_path.resolve().parents[2]
    locators = {
        name: _portable(path, root=root)
        for name, path in input_paths.items()
        if name in {"readiness_ledger", "transfer_ledger"}
    }
    exact_source = load_exact_evidence_only_source(
        component_program_path=input_paths["component_program"],
        component_program_result_path=input_paths["component_program_result"],
        component_dossier_path=input_paths["component_dossier"],
        component_dossier_result_path=input_paths["component_dossier_result"],
        step_ledger_path=input_paths["step_ledger"],
        step_result_path=input_paths["step_result"],
        terminal_procurement_path=input_paths["terminal_procurement"],
        assessment_as_of_utc=policy.get("assessment_as_of_utc"),
        unavailable_procurement_statuses=frozenset(unavailable),
    )
    hybrid_source, metadata = load_bounded_hybrid_source(
        exact_source=exact_source,
        readiness_ledger_path=input_paths["readiness_ledger"],
        readiness_result_path=input_paths["readiness_result"],
        readiness_config_path=input_paths["readiness_config"],
        transfer_ledger_path=input_paths["transfer_ledger"],
        transfer_result_path=input_paths["transfer_result"],
        transfer_config_path=input_paths["transfer_config"],
        upstream_registry_path=input_paths["upstream_registry"],
        oxidation_variant_path=input_paths["oxidation_variant"],
        assessment_as_of_utc=policy.get("assessment_as_of_utc"),
        retrieval_l3_expiry_days=policy.get("retrieval_l3_expiry_days"),
        unavailable_l3_statuses=frozenset(unavailable),
        source_locators=locators,
    )
    global_limits = policy.get("global_search_limits")
    if not isinstance(global_limits, dict):
        raise Ugi3HybridSearchError("global_search_limits must be an object")
    channel_counts = metadata["channels"]
    realized_global = {
        "components": len(hybrid_source.admitted_components),
        "exact_retrieval_candidates": channel_counts.get(
            SearchChannel.EXACT_IDENTITY_RETRIEVAL.value,
            0,
        ),
        "family_projection_proposals": channel_counts.get(
            SearchChannel.FAMILY_PROJECTION.value,
            0,
        ),
        "provenance_priorities": channel_counts.get(
            SearchChannel.PROVENANCE_PRIORITY.value,
            0,
        ),
        "forward_verifier_calls": 1,
    }
    for name, realized in realized_global.items():
        maximum = global_limits.get(f"maximum_{name}")
        if isinstance(maximum, bool) or not isinstance(maximum, int) or realized > maximum:
            raise Ugi3HybridSearchError(f"global hybrid-search budget exhausted: {name}")

    context = _cache_context(
        config=config,
        input_hashes=input_hashes,
        budget_limits=budget_limits,
    )
    planner = CachedRoutePlanner(
        RecursiveRouteAssessor(hybrid_source),
        FilePlannerCache(cache_root),
        context,
    )
    baseline_planner = RecursiveRouteAssessor(exact_source)
    baseline_outcomes: Counter[str] = Counter()
    hybrid_outcomes: Counter[str] = Counter()
    role_outcomes: dict[str, Counter[str]] = defaultdict(Counter)
    closure_by_channel: dict[str, Counter[str]] = defaultdict(Counter)
    rows: list[dict[str, Any]] = []
    newly_closed = 0
    newly_closed_original_missing = 0
    route_complete_agreement = 0
    cache_misses = 0
    cache_hits = 0
    first_pass_verifier_calls = 0
    original_missing_targets = 0
    original_missing_remaining = 0
    legacy_outside_reclassified_missing = 0
    for admitted in hybrid_source.admitted_components:
        baseline_budget = PlannerBudgetLedger(limits=budget_limits)
        baseline = baseline_planner.assess(admitted.target, baseline_budget)
        first_budget = PlannerBudgetLedger(limits=budget_limits)
        assessment = planner.assess(admitted.target, first_budget)
        second_budget = PlannerBudgetLedger(limits=budget_limits)
        cached = planner.assess(admitted.target, second_budget)
        if cached != assessment:
            raise Ugi3HybridSearchError("hybrid cache round-trip changed an assessment")
        if first_budget.physical_cache_misses != 1 or second_budget.physical_cache_hits != 1:
            raise Ugi3HybridSearchError("hybrid cache did not produce one miss then one hit")
        cache_misses += first_budget.physical_cache_misses
        cache_hits += second_budget.physical_cache_hits
        first_pass_verifier_calls += first_budget.verifier_calls
        baseline_outcomes[baseline.outcome.value] += 1
        hybrid_outcomes[assessment.outcome.value] += 1
        role_outcomes[admitted.target.role][assessment.outcome.value] += 1
        closure_by_channel[admitted.channel.value][assessment.outcome.value] += 1
        is_new = (
            baseline.outcome is not AssessmentOutcome.COMPLETE
            and assessment.outcome is AssessmentOutcome.COMPLETE
        )
        newly_closed += int(is_new)
        prior_missing = (
            admitted.original_component_id is not None
            and baseline.outcome is AssessmentOutcome.MISSING_KNOWLEDGE
        )
        original_missing_targets += int(prior_missing)
        original_missing_remaining += int(
            prior_missing and assessment.outcome is AssessmentOutcome.MISSING_KNOWLEDGE
        )
        newly_closed_original_missing += int(prior_missing and is_new)
        if (
            admitted.readiness_category == OUTSIDE_SUPPORT
            and assessment.outcome is AssessmentOutcome.MISSING_KNOWLEDGE
        ):
            legacy_outside_reclassified_missing += 1
        complete_agreement = (
            assessment.outcome is AssessmentOutcome.COMPLETE
        ) == admitted.readiness_complete
        route_complete_agreement += int(complete_agreement)
        rows.append(
            {
                "component_id": admitted.component_id,
                "role": admitted.target.role,
                "canonical_smiles": admitted.target.canonical_smiles,
                "original_component_id": admitted.original_component_id,
                "readiness_category": admitted.readiness_category,
                "readiness_complete": admitted.readiness_complete,
                "search_channel": admitted.channel.value,
                "proposal_only": admitted.proposal_only,
                "priority_only": admitted.priority_only,
                "baseline_outcome": baseline.outcome.value,
                "hybrid_outcome": assessment.outcome.value,
                "newly_closed": is_new,
                "route_complete_flag_agreement": complete_agreement,
                "assessment": assessment.to_dict(),
            }
        )

    summary = {
        "admitted_components": len(rows),
        "baseline_outcomes": dict(sorted(baseline_outcomes.items())),
        "hybrid_outcomes": dict(sorted(hybrid_outcomes.items())),
        "hybrid_outcomes_by_role": {
            role: dict(sorted(counts.items())) for role, counts in sorted(role_outcomes.items())
        },
        "search_channels": dict(sorted(channel_counts.items())),
        "outcomes_by_search_channel": {
            channel: dict(sorted(counts.items()))
            for channel, counts in sorted(closure_by_channel.items())
        },
        "newly_closed_components": newly_closed,
        "original_missing_targets": original_missing_targets,
        "newly_closed_original_missing_targets": newly_closed_original_missing,
        "original_missing_targets_remaining": original_missing_remaining,
        "legacy_outside_reclassified_missing": legacy_outside_reclassified_missing,
        "route_complete_flag_agreement": route_complete_agreement,
        "admitted_qualified_templates": metadata["admitted_qualified_templates"],
        "first_pass_cache_misses": cache_misses,
        "second_pass_cache_hits": cache_hits,
        "first_pass_verifier_calls": first_pass_verifier_calls,
        "realized_global_search_budget": realized_global,
    }
    _validate_expected(summary, config.get("expected_counts"), label="summary")
    ledger = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "metadata": metadata,
        "records": rows,
    }
    ledger_bytes = _gzip_json_bytes(ledger)
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "task": config.get("task"),
        "generated_utc": config.get("generated_utc"),
        "inputs": {
            "config": {
                "path": _portable(config_path, root=root),
                "sha256": sha256_file(config_path),
            },
            **{
                name: {"path": _portable(path, root=root), "sha256": input_hashes[name]}
                for name, path in sorted(input_paths.items())
            },
        },
        "search_policy": policy,
        "planner_cache_context": context.to_dict(),
        "summary": summary,
        "exact_retrieval": metadata,
        "claims_boundary": {
            "bounded_hybrid_search_diagnostic": True,
            "general_dynamic_planner_qualified": False,
            "production_synthesis_guidance_authorized_by_this_result": False,
            "family_projection_can_close": False,
            "motif_similarity_can_close": False,
            "provenance_can_close": False,
            "handle_qualification_can_close": False,
            "qualified_template_closure_demonstrated": False,
            "assessment_is_synthesis_success_probability": False,
        },
        "artifacts": {"assessment_ledger_sha256": sha256_bytes(ledger_bytes)},
        "safe_claim": (
            "Bounded exact-identity retrieval adds one newly route-complete component to the "
            "frozen Ugi registry. No admitted general template exists; family projections and "
            "provenance priorities remain non-closing missing knowledge."
        ),
    }
    return result, ledger_bytes
