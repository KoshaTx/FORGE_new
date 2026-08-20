"""Exact-evidence-only Ugi L2/L3 route-knowledge adapter and diagnostic.

The adapter translates frozen component-program, uniquely forward-verified L2
step and time-stamped L3 procurement records into the typed planner contract.
It performs no template search, analogue projection or learned inference.
Family-level and provenance-only records remain missing knowledge.
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
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase

from forge.data.r1_prime_audit import sha256_bytes, sha256_file
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

CONFIG_SCHEMA_VERSION = "phase1_ugi3_exact_evidence_source_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi3_exact_evidence_source.v1"
ASSESSMENT_LEDGER_SCHEMA_VERSION = "phase1_ugi3_exact_evidence_assessments.v1"

PROGRAM_ACCEPTED_TERMINAL = "accepted_procurement_terminal"
PROGRAM_EXACT = "exact_source_program"
PROGRAM_FAMILY = "reaction_family_projected_program"
PROGRAM_UNRESOLVED = "procurement_or_route_search_required"
SUPPORTED_PROGRAM_STATUSES = {
    PROGRAM_ACCEPTED_TERMINAL,
    PROGRAM_EXACT,
    PROGRAM_FAMILY,
    PROGRAM_UNRESOLVED,
}
VERIFIED_STEP = "verified_exact_product_unique"
SUPPORTED_COMPONENT_ROLES = {
    "amine_head",
    "oxoester_aldehyde_body_tail",
    "isocyanide_tail",
}


class Ugi3ExactEvidenceSourceError(ValueError):
    """Raised when frozen evidence cannot be translated without promotion."""


def _stable_json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _sha256_payload(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode()).hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise Ugi3ExactEvidenceSourceError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise Ugi3ExactEvidenceSourceError(f"{label} must be a JSON object")
    return value


def _read_csv(path: Path, *, label: str) -> list[dict[str, str]]:
    try:
        with gzip.open(path, "rt", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise Ugi3ExactEvidenceSourceError(f"{label} has no header")
            return list(reader)
    except (OSError, csv.Error) as exc:
        raise Ugi3ExactEvidenceSourceError(f"could not read {label}: {path}") from exc


def _json_list(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, str):
        raise Ugi3ExactEvidenceSourceError(f"{label} must be serialized JSON")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise Ugi3ExactEvidenceSourceError(f"{label} is invalid JSON") from exc
    if not isinstance(parsed, list):
        raise Ugi3ExactEvidenceSourceError(f"{label} must be a list")
    return parsed


def _canonical_constitution(smiles: Any, *, label: str) -> str:
    if not isinstance(smiles, str) or not smiles:
        raise Ugi3ExactEvidenceSourceError(f"{label} must be nonempty SMILES")
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise Ugi3ExactEvidenceSourceError(f"{label} contains invalid SMILES")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def _parse_utc(value: Any, *, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise Ugi3ExactEvidenceSourceError(f"{label} must be an ISO-8601 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise Ugi3ExactEvidenceSourceError(f"{label} is invalid") from exc
    if parsed.tzinfo != timezone.utc:
        raise Ugi3ExactEvidenceSourceError(f"{label} must resolve to UTC")
    return parsed


def _portable(path: Path, *, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _validate_expected(observed: Any, expected: Any, *, label: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(observed, dict) or set(observed) != set(expected):
            raise Ugi3ExactEvidenceSourceError(f"{label} fields mismatch")
        for key, expected_value in expected.items():
            _validate_expected(observed[key], expected_value, label=f"{label}.{key}")
        return
    if observed != expected:
        raise Ugi3ExactEvidenceSourceError(
            f"{label} mismatch: expected {expected!r}, observed {observed!r}"
        )


def _gzip_json_bytes(value: Any) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as compressed:
        compressed.write((_stable_json(value) + "\n").encode())
    return output.getvalue()


def procurement_availability(
    *,
    snapshot: Mapping[str, Any],
    record: Mapping[str, Any] | None,
    assessment_as_of_utc: str,
    unavailable_procurement_statuses: frozenset[str],
) -> AvailabilityState:
    """Map one frozen L3 record into a deterministic availability state."""

    accessed = _parse_utc(snapshot.get("accessed_utc"), label="L3 accessed_utc")
    expiry_days = snapshot.get("expiry_days")
    if isinstance(expiry_days, bool) or not isinstance(expiry_days, int) or expiry_days < 0:
        raise Ugi3ExactEvidenceSourceError("L3 expiry_days must be a nonnegative integer")
    as_of = _parse_utc(assessment_as_of_utc, label="assessment_as_of_utc")
    if as_of < accessed:
        raise Ugi3ExactEvidenceSourceError("assessment cannot precede the L3 snapshot")
    if as_of > accessed + timedelta(days=expiry_days):
        return AvailabilityState.EXPIRED
    if record is None:
        return AvailabilityState.UNASSESSED
    if record.get("current_item_level_procurement_closed") is True:
        return AvailabilityState.CURRENT_CLOSED
    status = record.get("procurement_status")
    if isinstance(status, str) and status in unavailable_procurement_statuses:
        return AvailabilityState.UNAVAILABLE
    return AvailabilityState.UNASSESSED


@dataclass(frozen=True)
class AdmittedComponentTarget:
    """One frozen component root and its prior dossier classification."""

    component_id: str
    target: RouteTarget
    program_status: str
    existing_evidence_tier: str
    existing_complete: bool


@dataclass(frozen=True)
class _ComponentRecord:
    admitted: AdmittedComponentTarget
    evidence: EvidenceRecord
    exact_root_proposal: RouteStepProposal | None


class ExactEvidenceOnlyUgi3Source:
    """Fail-closed lookup over exact frozen Ugi component evidence only."""

    def __init__(
        self,
        *,
        components: Mapping[tuple[str, str], _ComponentRecord],
        upstream_proposals: Mapping[tuple[str, str], RouteStepProposal],
        upstream_leaves: Mapping[tuple[str, str], EvidenceRecord],
    ):
        self._components = dict(components)
        self._upstream_proposals = dict(upstream_proposals)
        self._upstream_leaves = dict(upstream_leaves)
        self._admitted = tuple(
            sorted(
                (record.admitted for record in self._components.values()),
                key=lambda record: record.component_id,
            )
        )

    @property
    def admitted_components(self) -> tuple[AdmittedComponentTarget, ...]:
        return self._admitted

    @staticmethod
    def _upstream_role(component_id: str) -> str:
        return f"ugi3_upstream_material:{component_id}"

    def lookup(self, target: RouteTarget) -> KnowledgeResult:
        canonical = _canonical_constitution(
            target.canonical_smiles,
            label="route target canonical_smiles",
        )
        component = self._components.get((target.role, canonical))
        if component is not None:
            status = component.admitted.program_status
            if status == PROGRAM_ACCEPTED_TERMINAL:
                return KnowledgeResult(
                    disposition=KnowledgeDisposition.TERMINAL,
                    evidence=(component.evidence,),
                    detail="frozen exact accepted-terminal record retrieved",
                )
            if status == PROGRAM_EXACT:
                if component.exact_root_proposal is None:  # pragma: no cover - constructor gate
                    raise Ugi3ExactEvidenceSourceError("exact component lost its root proposal")
                return KnowledgeResult(
                    disposition=KnowledgeDisposition.EXPAND,
                    evidence=component.exact_root_proposal.evidence,
                    proposal=component.exact_root_proposal,
                    detail="frozen exact-source uniquely forward-verified step retrieved",
                )
            return KnowledgeResult(
                disposition=KnowledgeDisposition.MISSING_KNOWLEDGE,
                evidence=(component.evidence,),
                detail=(
                    "family-projected evidence is not exact route evidence"
                    if status == PROGRAM_FAMILY
                    else "provenance exists but no exact upstream program is assigned"
                ),
            )

        proposal = self._upstream_proposals.get((target.role, canonical))
        if proposal is not None:
            return KnowledgeResult(
                disposition=KnowledgeDisposition.EXPAND,
                evidence=proposal.evidence,
                proposal=proposal,
                detail="frozen exact-source uniquely forward-verified step retrieved",
            )
        terminal = self._upstream_leaves.get((target.role, canonical))
        if terminal is not None:
            return KnowledgeResult(
                disposition=KnowledgeDisposition.TERMINAL,
                evidence=(terminal,),
                detail="frozen exact terminal-material record retrieved",
            )
        if target.role in SUPPORTED_COMPONENT_ROLES or target.role.startswith(
            "ugi3_upstream_material:"
        ):
            return KnowledgeResult(
                disposition=KnowledgeDisposition.OUTSIDE_SUPPORT,
                evidence=(),
                detail="target is outside the frozen exact-evidence adapter index",
            )
        return KnowledgeResult(
            disposition=KnowledgeDisposition.INVALID_INPUT,
            evidence=(),
            detail="target role is not supported by the Ugi exact-evidence adapter",
        )


def load_exact_evidence_only_source(
    *,
    component_program_path: Path,
    component_program_result_path: Path,
    component_dossier_path: Path,
    component_dossier_result_path: Path,
    step_ledger_path: Path,
    step_result_path: Path,
    terminal_procurement_path: Path,
    assessment_as_of_utc: str,
    unavailable_procurement_statuses: frozenset[str],
    source_locators: Mapping[str, str] | None = None,
) -> ExactEvidenceOnlyUgi3Source:
    """Authenticate and translate the frozen Ugi evidence ledgers."""

    locators = {
        "component_program": str(component_program_path),
        "component_dossier": str(component_dossier_path),
        "step_ledger": str(step_ledger_path),
        "terminal_procurement": str(terminal_procurement_path),
        **(dict(source_locators) if source_locators is not None else {}),
    }
    hashes = {
        "component_program": sha256_file(component_program_path),
        "component_dossier": sha256_file(component_dossier_path),
        "step_ledger": sha256_file(step_ledger_path),
        "terminal_procurement": sha256_file(terminal_procurement_path),
    }
    program_result = _load_json(component_program_result_path, label="component program result")
    program_artifact = program_result.get("artifacts", {}).get(
        "agile_virtual_ugi3_component_program_ledger.csv.gz",
        {},
    )
    if program_artifact.get("sha256") != hashes["component_program"]:
        raise Ugi3ExactEvidenceSourceError("component program result does not own its ledger")
    dossier_result = _load_json(component_dossier_result_path, label="component dossier result")
    if (
        dossier_result.get("artifacts", {}).get("component_ledger_sha256")
        != hashes["component_dossier"]
    ):
        raise Ugi3ExactEvidenceSourceError("component dossier result does not own its ledger")
    step_result = _load_json(step_result_path, label="step verification result")
    if step_result.get("artifacts", {}).get("step_ledger_sha256") != hashes["step_ledger"]:
        raise Ugi3ExactEvidenceSourceError("step result does not own its ledger")

    procurement = _load_json(terminal_procurement_path, label="terminal procurement")
    snapshot = procurement.get("snapshot")
    records = procurement.get("records")
    if not isinstance(snapshot, dict) or not isinstance(records, list):
        raise Ugi3ExactEvidenceSourceError("terminal procurement lacks snapshot or records")
    procurement_by_smiles: dict[str, tuple[int, dict[str, Any]]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise Ugi3ExactEvidenceSourceError("terminal procurement record must be an object")
        canonical = _canonical_constitution(
            record.get("canonical_smiles"),
            label=f"terminal procurement record {index}",
        )
        if canonical in procurement_by_smiles:
            raise Ugi3ExactEvidenceSourceError("duplicate constitutional procurement identity")
        procurement_by_smiles[canonical] = (index, record)

    def terminal_evidence(
        canonical_smiles: str,
        *,
        fallback_component_id: str,
    ) -> EvidenceRecord:
        match = procurement_by_smiles.get(canonical_smiles)
        availability = procurement_availability(
            snapshot=snapshot,
            record=None if match is None else match[1],
            assessment_as_of_utc=assessment_as_of_utc,
            unavailable_procurement_statuses=unavailable_procurement_statuses,
        )
        if match is None:
            return EvidenceRecord(
                evidence_id=f"l3-unassessed:{fallback_component_id}:{canonical_smiles}",
                tier=EvidenceTier.ACCEPTED_TERMINAL,
                source_sha256=hashes["component_program"],
                source_locator=(
                    f"{locators['component_program']}#component_id={fallback_component_id}"
                ),
                exact_substrate=True,
                forward_verification=ForwardVerificationState.NOT_APPLICABLE,
                availability=availability,
            )
        index, _ = match
        return EvidenceRecord(
            evidence_id=f"l3-procurement:{index:03d}:{canonical_smiles}",
            tier=EvidenceTier.ACCEPTED_TERMINAL,
            source_sha256=hashes["terminal_procurement"],
            source_locator=f"{locators['terminal_procurement']}#records[{index}]",
            exact_substrate=True,
            forward_verification=ForwardVerificationState.NOT_APPLICABLE,
            availability=availability,
        )

    programs = _read_csv(component_program_path, label="component program ledger")
    dossiers = _read_csv(component_dossier_path, label="component dossier ledger")
    steps = _read_csv(step_ledger_path, label="step verification ledger")
    dossier_by_id = {row.get("component_id", ""): row for row in dossiers}
    if len(dossier_by_id) != len(dossiers) or "" in dossier_by_id:
        raise Ugi3ExactEvidenceSourceError("component dossier IDs must be unique and nonempty")
    program_ids = [row.get("component_id", "") for row in programs]
    if not all(program_ids) or len(program_ids) != len(set(program_ids)):
        raise Ugi3ExactEvidenceSourceError("component program IDs must be unique and nonempty")
    if set(program_ids) != set(dossier_by_id):
        raise Ugi3ExactEvidenceSourceError("component program and dossier identities differ")
    steps_by_component: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in steps:
        component_id = row.get("component_id", "")
        if component_id not in dossier_by_id:
            raise Ugi3ExactEvidenceSourceError("step ledger references an unknown component")
        steps_by_component[component_id].append(row)

    components: dict[tuple[str, str], _ComponentRecord] = {}
    upstream_proposals: dict[tuple[str, str], RouteStepProposal] = {}
    upstream_leaves: dict[tuple[str, str], EvidenceRecord] = {}

    for program in programs:
        component_id = program["component_id"]
        role = program.get("role", "")
        if role not in SUPPORTED_COMPONENT_ROLES:
            raise Ugi3ExactEvidenceSourceError(f"{component_id} has unsupported role {role!r}")
        canonical = _canonical_constitution(
            program.get("canonical_smiles"),
            label=f"component {component_id}",
        )
        status = program.get("program_status", "")
        if status not in SUPPORTED_PROGRAM_STATUSES:
            raise Ugi3ExactEvidenceSourceError(
                f"{component_id} has unsupported program status {status!r}"
            )
        dossier = dossier_by_id[component_id]
        if (
            dossier.get("role") != role
            or _canonical_constitution(
                dossier.get("canonical_smiles"),
                label=f"component dossier {component_id}",
            )
            != canonical
            or dossier.get("program_status") != status
        ):
            raise Ugi3ExactEvidenceSourceError(f"component dossier disagrees for {component_id}")
        existing_complete = dossier.get("exact_source_computational_component") == "true"
        admitted = AdmittedComponentTarget(
            component_id=component_id,
            target=RouteTarget(role=role, canonical_smiles=canonical),
            program_status=status,
            existing_evidence_tier=dossier.get("evidence_tier", ""),
            existing_complete=existing_complete,
        )
        evidence_tier = (
            EvidenceTier.FAMILY_PROJECTED
            if status == PROGRAM_FAMILY
            else EvidenceTier.PROVENANCE_ONLY
        )
        root_evidence = EvidenceRecord(
            evidence_id=f"component-program:{component_id}",
            tier=evidence_tier,
            source_sha256=hashes["component_program"],
            source_locator=f"{locators['component_program']}#component_id={component_id}",
            exact_substrate=False,
            forward_verification=ForwardVerificationState.NOT_RUN,
            availability=AvailabilityState.UNASSESSED,
        )
        root_proposal: RouteStepProposal | None = None

        if status == PROGRAM_ACCEPTED_TERMINAL:
            if steps_by_component.get(component_id):
                raise Ugi3ExactEvidenceSourceError(
                    f"accepted terminal {component_id} unexpectedly has L2 steps"
                )
            if dossier.get("evidence_tier") != "accepted_terminal_l3_closed":
                raise Ugi3ExactEvidenceSourceError(
                    f"accepted terminal {component_id} lacks its frozen L3-closed tier"
                )
            accepted_availability = procurement_availability(
                snapshot=snapshot,
                record={"current_item_level_procurement_closed": True},
                assessment_as_of_utc=assessment_as_of_utc,
                unavailable_procurement_statuses=unavailable_procurement_statuses,
            )
            root_evidence = EvidenceRecord(
                evidence_id=f"accepted-component-terminal:{component_id}",
                tier=EvidenceTier.ACCEPTED_TERMINAL,
                source_sha256=hashes["component_dossier"],
                source_locator=(f"{locators['component_dossier']}#component_id={component_id}"),
                exact_substrate=True,
                forward_verification=ForwardVerificationState.NOT_APPLICABLE,
                availability=accepted_availability,
            )
        elif status == PROGRAM_EXACT:
            expected_steps = _json_list(
                program.get("program_steps_json"),
                label=f"{component_id} program steps",
            )
            if any(not isinstance(step, dict) for step in expected_steps):
                raise Ugi3ExactEvidenceSourceError(f"{component_id} has malformed steps")
            expected_by_index = {str(step.get("step_index")): step for step in expected_steps}
            if len(expected_by_index) != len(expected_steps):
                raise Ugi3ExactEvidenceSourceError(f"{component_id} has duplicate step indices")
            observed_by_index: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in steps_by_component.get(component_id, []):
                observed_by_index[row.get("step_index", "")].append(row)
            if set(observed_by_index) != set(expected_by_index) or any(
                len(rows) != 1 for rows in observed_by_index.values()
            ):
                raise Ugi3ExactEvidenceSourceError(
                    f"{component_id} lacks one-to-one exact step evidence"
                )
            scoped_role = ExactEvidenceOnlyUgi3Source._upstream_role(component_id)
            products = {
                _canonical_constitution(
                    step.get("product"),
                    label=f"{component_id} expected step product",
                )
                for step in expected_steps
            }
            declared_leaves = {
                _canonical_constitution(leaf, label=f"{component_id} declared leaf")
                for leaf in _json_list(
                    program.get("proposed_leaf_candidates_json"),
                    label=f"{component_id} proposed leaves",
                )
            }
            derived_leaves: set[str] = set()
            proposals_for_component: dict[str, RouteStepProposal] = {}
            for index, expected in expected_by_index.items():
                row = observed_by_index[index][0]
                expected_reactants = expected.get("reactants")
                if not isinstance(expected_reactants, list) or any(
                    not isinstance(value, str) for value in expected_reactants
                ):
                    raise Ugi3ExactEvidenceSourceError(
                        f"{component_id} step {index} has invalid reactants"
                    )
                observed_reactants = _json_list(
                    row.get("reactants_json"),
                    label=f"{component_id} step {index} observed reactants",
                )
                if (
                    row.get("component_role") != role
                    or _canonical_constitution(
                        row.get("component_smiles"),
                        label=f"{component_id} step component",
                    )
                    != canonical
                    or row.get("transformation") != expected.get("transformation")
                    or observed_reactants != expected_reactants
                    or row.get("expected_product") != expected.get("product")
                    or row.get("verification_status") != VERIFIED_STEP
                    or row.get("forward_product_count") != "1"
                    or row.get("expected_product_in_outputs") != "true"
                ):
                    raise Ugi3ExactEvidenceSourceError(
                        f"{component_id} step {index} is not exact and uniquely verified"
                    )
                product = _canonical_constitution(
                    expected.get("product"),
                    label=f"{component_id} step {index} product",
                )
                reactant_targets: list[RouteTarget] = []
                for reactant in expected_reactants:
                    reactant_canonical = _canonical_constitution(
                        reactant,
                        label=f"{component_id} step {index} reactant",
                    )
                    reactant_targets.append(
                        RouteTarget(role=scoped_role, canonical_smiles=reactant_canonical)
                    )
                    if reactant_canonical not in products:
                        derived_leaves.add(reactant_canonical)
                step_evidence = EvidenceRecord(
                    evidence_id=f"exact-step:{component_id}:{index}",
                    tier=EvidenceTier.EXACT_SOURCE,
                    source_sha256=hashes["step_ledger"],
                    source_locator=(
                        f"{locators['step_ledger']}#component_id={component_id}&step_index={index}"
                    ),
                    exact_substrate=True,
                    forward_verification=(ForwardVerificationState.VERIFIED_EXACT_PRODUCT_UNIQUE),
                    availability=AvailabilityState.UNASSESSED,
                )
                proposal = RouteStepProposal(
                    reaction_id=row.get("qualified_reaction_id", ""),
                    reactants=tuple(reactant_targets),
                    evidence=(step_evidence,),
                    forward_product_count=1,
                    verifier_calls_required=0,
                    product_candidates_considered=1,
                )
                if product in proposals_for_component:
                    raise Ugi3ExactEvidenceSourceError(
                        f"{component_id} has multiple exact proposals for one product"
                    )
                proposals_for_component[product] = proposal
                upstream_proposals[(scoped_role, product)] = proposal
            if derived_leaves != declared_leaves:
                raise Ugi3ExactEvidenceSourceError(
                    f"{component_id} derived and declared terminal leaves differ"
                )
            try:
                root_proposal = proposals_for_component[canonical]
            except KeyError as exc:
                raise Ugi3ExactEvidenceSourceError(
                    f"{component_id} exact program does not produce its component root"
                ) from exc
            for leaf in sorted(declared_leaves):
                upstream_leaves[(scoped_role, leaf)] = terminal_evidence(
                    leaf,
                    fallback_component_id=component_id,
                )
            root_evidence = root_proposal.evidence[0]
        elif steps_by_component.get(component_id):
            raise Ugi3ExactEvidenceSourceError(
                f"non-exact component {component_id} unexpectedly has exact step records"
            )

        key = (role, canonical)
        if key in components:
            raise Ugi3ExactEvidenceSourceError("duplicate constitutional role-component identity")
        components[key] = _ComponentRecord(
            admitted=admitted,
            evidence=root_evidence,
            exact_root_proposal=root_proposal,
        )

    return ExactEvidenceOnlyUgi3Source(
        components=components,
        upstream_proposals=upstream_proposals,
        upstream_leaves=upstream_leaves,
    )


def _cache_context(
    *,
    config: Mapping[str, Any],
    input_paths: Mapping[str, Path],
    input_hashes: Mapping[str, str],
    procurement: Mapping[str, Any],
    step_result: Mapping[str, Any],
    budget_limits: PlannerBudgetLimits,
) -> PlannerCacheContext:
    policy = config.get("diagnostic_policy")
    if not isinstance(policy, dict):
        raise Ugi3ExactEvidenceSourceError("diagnostic_policy must be an object")
    snapshot = procurement.get("snapshot")
    if not isinstance(snapshot, dict):
        raise Ugi3ExactEvidenceSourceError("procurement snapshot is missing")
    accessed = _parse_utc(snapshot.get("accessed_utc"), label="L3 accessed_utc")
    expiry_days = snapshot.get("expiry_days")
    if isinstance(expiry_days, bool) or not isinstance(expiry_days, int):
        raise Ugi3ExactEvidenceSourceError("L3 expiry_days must be an integer")
    expires = accessed + timedelta(days=expiry_days)
    bound_variants = step_result.get("inputs", {}).get("bound_variants")
    if not isinstance(bound_variants, dict):
        raise Ugi3ExactEvidenceSourceError("step result lacks bound variants")
    planner_digest = _sha256_payload(
        {
            "planner_contract": input_hashes["planner_contract_source"],
            "adapter": input_hashes["adapter_source"],
        }
    )
    return PlannerCacheContext(
        planner_id="ugi3_exact_evidence_only_diagnostic_v1",
        planner_sha256=planner_digest,
        search_policy_sha256=_sha256_payload(policy),
        value_policy_sha256=_sha256_payload(
            {"diagnostic_only": True, "synthesis_value": "not_computed"}
        ),
        l1_reaction_sha256=input_hashes["l1_variant"],
        upstream_reaction_registry_sha256=input_hashes["upstream_reaction_registry"],
        variant_registry_sha256=_sha256_payload(bound_variants),
        verifier_sha256=input_hashes["step_result"],
        l3_snapshot_sha256=input_hashes["terminal_procurement"],
        l3_region=str(snapshot.get("region")),
        l3_accessed_at_utc=accessed.isoformat().replace("+00:00", "Z"),
        l3_expires_at_utc=expires.isoformat().replace("+00:00", "Z"),
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


def build_exact_evidence_source_diagnostic(
    *,
    config_path: Path,
    input_paths: Mapping[str, Path],
    cache_root: Path,
) -> tuple[dict[str, Any], bytes]:
    """Exercise every admitted component through planner and cache contracts."""

    config = _load_json(config_path, label="exact-evidence diagnostic config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise Ugi3ExactEvidenceSourceError("unsupported exact-evidence config schema")
    configured_inputs = config.get("inputs")
    if not isinstance(configured_inputs, dict) or set(configured_inputs) != set(input_paths):
        raise Ugi3ExactEvidenceSourceError("configured and supplied inputs differ")
    input_hashes: dict[str, str] = {}
    for name, path in input_paths.items():
        record = configured_inputs[name]
        if not isinstance(record, dict):
            raise Ugi3ExactEvidenceSourceError(f"input {name} must be an object")
        observed = sha256_file(path)
        if observed != record.get("expected_sha256"):
            raise Ugi3ExactEvidenceSourceError(
                f"{name} hash mismatch: expected {record.get('expected_sha256')}, "
                f"observed {observed}"
            )
        input_hashes[name] = observed
    policy = config.get("diagnostic_policy")
    if not isinstance(policy, dict):
        raise Ugi3ExactEvidenceSourceError("diagnostic_policy must be an object")
    unavailable = policy.get("unavailable_procurement_statuses")
    if not isinstance(unavailable, list) or any(
        not isinstance(value, str) for value in unavailable
    ):
        raise Ugi3ExactEvidenceSourceError(
            "unavailable_procurement_statuses must be a list of strings"
        )
    budget_limits = PlannerBudgetLimits.from_dict(policy.get("budget_limits"))
    root = config_path.resolve().parents[2]
    source_locators = {
        name: _portable(path, root=root)
        for name, path in input_paths.items()
        if name in {"component_program", "component_dossier", "step_ledger", "terminal_procurement"}
    }
    source = load_exact_evidence_only_source(
        component_program_path=input_paths["component_program"],
        component_program_result_path=input_paths["component_program_result"],
        component_dossier_path=input_paths["component_dossier"],
        component_dossier_result_path=input_paths["component_dossier_result"],
        step_ledger_path=input_paths["step_ledger"],
        step_result_path=input_paths["step_result"],
        terminal_procurement_path=input_paths["terminal_procurement"],
        assessment_as_of_utc=policy.get("assessment_as_of_utc"),
        unavailable_procurement_statuses=frozenset(unavailable),
        source_locators=source_locators,
    )
    procurement = _load_json(input_paths["terminal_procurement"], label="terminal procurement")
    step_result = _load_json(input_paths["step_result"], label="step result")
    context = _cache_context(
        config=config,
        input_paths=input_paths,
        input_hashes=input_hashes,
        procurement=procurement,
        step_result=step_result,
        budget_limits=budget_limits,
    )
    planner = CachedRoutePlanner(
        RecursiveRouteAssessor(source),
        FilePlannerCache(cache_root),
        context,
    )
    crosswalk = policy.get("existing_tier_to_expected_outcome")
    if not isinstance(crosswalk, dict) or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in crosswalk.items()
    ):
        raise Ugi3ExactEvidenceSourceError("existing tier crosswalk must be an object")

    assessment_rows: list[dict[str, Any]] = []
    outcome_counts: Counter[str] = Counter()
    tier_counts: Counter[str] = Counter()
    role_outcomes: dict[str, Counter[str]] = defaultdict(Counter)
    first_pass_misses = 0
    second_pass_hits = 0
    tier_agreement = 0
    complete_flag_agreement = 0
    for admitted in source.admitted_components:
        expected_raw = crosswalk.get(admitted.existing_evidence_tier)
        try:
            expected = AssessmentOutcome(expected_raw)
        except (TypeError, ValueError) as exc:
            raise Ugi3ExactEvidenceSourceError(
                f"no valid outcome crosswalk for {admitted.existing_evidence_tier!r}"
            ) from exc
        first_budget = PlannerBudgetLedger(limits=budget_limits)
        assessment = planner.assess(admitted.target, first_budget)
        if first_budget.physical_cache_misses != 1 or first_budget.physical_cache_hits != 0:
            raise Ugi3ExactEvidenceSourceError("first diagnostic pass did not miss exactly once")
        first_pass_misses += first_budget.physical_cache_misses
        second_budget = PlannerBudgetLedger(limits=budget_limits)
        cached = planner.assess(admitted.target, second_budget)
        if second_budget.physical_cache_hits != 1 or second_budget.physical_cache_misses != 0:
            raise Ugi3ExactEvidenceSourceError("second diagnostic pass did not hit exactly once")
        if cached != assessment:
            raise Ugi3ExactEvidenceSourceError("cache round-trip changed an assessment")
        second_pass_hits += second_budget.physical_cache_hits
        agreement = assessment.outcome is expected
        complete_agreement = (
            assessment.outcome is AssessmentOutcome.COMPLETE
        ) == admitted.existing_complete
        tier_agreement += int(agreement)
        complete_flag_agreement += int(complete_agreement)
        outcome_counts[assessment.outcome.value] += 1
        tier_counts[admitted.existing_evidence_tier] += 1
        role_outcomes[admitted.target.role][assessment.outcome.value] += 1
        assessment_rows.append(
            {
                "component_id": admitted.component_id,
                "role": admitted.target.role,
                "canonical_smiles": admitted.target.canonical_smiles,
                "program_status": admitted.program_status,
                "existing_evidence_tier": admitted.existing_evidence_tier,
                "existing_complete": admitted.existing_complete,
                "expected_outcome": expected.value,
                "observed_outcome": assessment.outcome.value,
                "tier_crosswalk_agreement": agreement,
                "complete_flag_agreement": complete_agreement,
                "assessment": assessment.to_dict(),
            }
        )

    summary = {
        "admitted_components": len(assessment_rows),
        "assessment_outcomes": dict(sorted(outcome_counts.items())),
        "components_by_existing_evidence_tier": dict(sorted(tier_counts.items())),
        "outcomes_by_role": {
            role: dict(sorted(counts.items())) for role, counts in sorted(role_outcomes.items())
        },
        "tier_crosswalk_agreement": tier_agreement,
        "complete_flag_agreement": complete_flag_agreement,
        "first_pass_cache_misses": first_pass_misses,
        "second_pass_cache_hits": second_pass_hits,
    }
    _validate_expected(summary, config.get("expected_counts"), label="summary")
    ledger_value = {
        "schema_version": ASSESSMENT_LEDGER_SCHEMA_VERSION,
        "records": assessment_rows,
    }
    ledger_bytes = _gzip_json_bytes(ledger_value)
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
        "diagnostic_policy": policy,
        "planner_cache_context": context.to_dict(),
        "summary": summary,
        "claims_boundary": {
            "evidence_only_diagnostic_plumbing": True,
            "general_dynamic_planner_qualified": False,
            "learned_route_planner_qualified": False,
            "production_synthesis_guidance_authorized_by_this_result": False,
            "family_projection_can_close": False,
            "provenance_only_can_close": False,
            "assessment_outcome_is_synthesis_success_probability": False,
        },
        "artifacts": {
            "assessment_ledger_sha256": sha256_bytes(ledger_bytes),
        },
        "safe_claim": (
            "The frozen exact-evidence adapter reproduces the admitted component dossier "
            "classifications through the typed recursive planner and deterministic cache. "
            "This diagnostic performs no route inference and does not qualify production guidance."
        ),
    }
    return result, ledger_bytes
