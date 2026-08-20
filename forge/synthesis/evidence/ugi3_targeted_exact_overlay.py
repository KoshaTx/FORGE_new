"""Typed exact-evidence overlay for prioritized Ugi aldehyde components.

The overlay composes a previously qualified route-knowledge source with the
bounded aldehyde evidence pack.  It does not learn or generalize an oxidation
template: exact current procurement records become terminals, and exact-source
routes become one-step recursive expansions to current accepted leaves.
Everything outside those exact identities is delegated unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase

from forge.core.hashing import sha256_bytes, sha256_file
from forge.core.io import read_json_object
from forge.synthesis.engine.planner import (
    AvailabilityState,
    EvidenceRecord,
    EvidenceTier,
    ForwardVerificationState,
    KnowledgeDisposition,
    KnowledgeResult,
    RouteKnowledgeSource,
    RouteStepProposal,
    RouteTarget,
)
from forge.synthesis.evidence.ugi3_targeted_aldehyde_evidence import (
    build_targeted_aldehyde_evidence_audit,
)

EVIDENCE_SCHEMA_VERSION = "phase1_ugi3_targeted_aldehyde_evidence.v1"
OVERLAY_SCHEMA_VERSION = "phase1_ugi3_targeted_exact_overlay.v1"


class Ugi3TargetedExactOverlayError(ValueError):
    """Raised when exact evidence cannot be composed without promotion."""


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    return read_json_object(path, error=Ugi3TargetedExactOverlayError, label=label)


def _canonical(smiles: Any, *, label: str) -> str:
    if not isinstance(smiles, str) or not smiles:
        raise Ugi3TargetedExactOverlayError(f"{label} must be nonempty SMILES")
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise Ugi3TargetedExactOverlayError(f"{label} contains invalid SMILES")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


@dataclass(frozen=True)
class TargetedOverlayRecord:
    """One exact target and the evidence decision selected by the overlay."""

    component_id: str
    target: RouteTarget
    channel: str
    route_id: str | None


class TargetedExactUgi3Overlay:
    """Delegate by default and override only authenticated exact identities."""

    def __init__(
        self,
        *,
        base_source: RouteKnowledgeSource,
        root_results: Mapping[tuple[str, str], KnowledgeResult],
        upstream_results: Mapping[tuple[str, str], KnowledgeResult],
        records: tuple[TargetedOverlayRecord, ...],
        unresolved_target: RouteTarget,
    ):
        self._base_source = base_source
        self._root_results = dict(root_results)
        self._upstream_results = dict(upstream_results)
        self._records = records
        self._unresolved_target = unresolved_target

    @property
    def overlay_records(self) -> tuple[TargetedOverlayRecord, ...]:
        return self._records

    @property
    def unresolved_target(self) -> RouteTarget:
        return self._unresolved_target

    def lookup(self, target: RouteTarget) -> KnowledgeResult:
        canonical = _canonical(target.canonical_smiles, label="overlay route target")
        key = (target.role, canonical)
        root = self._root_results.get(key)
        if root is not None:
            return root
        upstream = self._upstream_results.get(key)
        if upstream is not None:
            return upstream
        return self._base_source.lookup(target)


def load_targeted_exact_overlay(
    *,
    base_source: RouteKnowledgeSource,
    audit_config_path: Path,
    audit_input_paths: Mapping[str, Path],
    stored_audit_result_path: Path,
    stored_route_ledger_path: Path,
    stored_product_ledger_path: Path,
) -> tuple[TargetedExactUgi3Overlay, dict[str, Any]]:
    """Authenticate the audit and translate its admitted records into routes."""

    fresh_result, route_ledger, product_ledger = build_targeted_aldehyde_evidence_audit(
        config_path=audit_config_path,
        input_paths=audit_input_paths,
    )
    stored_result = _load_json(stored_audit_result_path, label="stored targeted audit result")
    if stored_result != fresh_result:
        raise Ugi3TargetedExactOverlayError(
            "stored targeted audit result is not reproducible from frozen inputs"
        )
    route_hash = sha256_bytes(route_ledger)
    product_hash = sha256_bytes(product_ledger)
    if sha256_file(stored_route_ledger_path) != route_hash:
        raise Ugi3TargetedExactOverlayError("stored route ledger hash mismatch")
    if sha256_file(stored_product_ledger_path) != product_hash:
        raise Ugi3TargetedExactOverlayError("stored product ledger hash mismatch")
    artifacts = stored_result.get("artifacts")
    if not isinstance(artifacts, dict):
        raise Ugi3TargetedExactOverlayError("stored audit result lacks artifacts")
    if artifacts.get("route_verification_ledger.csv.gz", {}).get("sha256") != route_hash:
        raise Ugi3TargetedExactOverlayError("audit result does not own its route ledger")
    if artifacts.get("product_closure_impact_ledger.csv.gz", {}).get("sha256") != product_hash:
        raise Ugi3TargetedExactOverlayError("audit result does not own its product ledger")

    evidence_path = audit_input_paths.get("evidence_pack")
    terminal_procurement_path = audit_input_paths.get("terminal_procurement")
    if evidence_path is None or terminal_procurement_path is None:
        raise Ugi3TargetedExactOverlayError("overlay requires evidence and L3 inputs")
    evidence = _load_json(evidence_path, label="targeted evidence pack")
    if evidence.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
        raise Ugi3TargetedExactOverlayError("unsupported targeted evidence schema")
    evidence_hash = sha256_file(evidence_path)
    terminal_hash = sha256_file(terminal_procurement_path)

    root_results: dict[tuple[str, str], KnowledgeResult] = {}
    upstream_results: dict[tuple[str, str], KnowledgeResult] = {}
    records: list[TargetedOverlayRecord] = []
    direct_keys: set[tuple[str, str]] = set()
    direct = evidence.get("direct_procurement_terminals")
    if not isinstance(direct, list):
        raise Ugi3TargetedExactOverlayError("direct terminals must be a list")
    for index, record in enumerate(direct):
        if not isinstance(record, dict):
            raise Ugi3TargetedExactOverlayError("direct terminal must be an object")
        role = str(record.get("role"))
        canonical = _canonical(record.get("canonical_smiles"), label="direct terminal")
        key = (role, canonical)
        if key in root_results:
            raise Ugi3TargetedExactOverlayError("duplicate direct target")
        evidence_record = EvidenceRecord(
            evidence_id=f"targeted-direct:{record.get('component_id')}",
            tier=EvidenceTier.ACCEPTED_TERMINAL,
            source_sha256=evidence_hash,
            source_locator=(f"{evidence_path}#direct_procurement_terminals[{index}]"),
            exact_substrate=True,
            forward_verification=ForwardVerificationState.NOT_APPLICABLE,
            availability=AvailabilityState.CURRENT_CLOSED,
        )
        root_results[key] = KnowledgeResult(
            disposition=KnowledgeDisposition.TERMINAL,
            evidence=(evidence_record,),
            detail="targeted exact current procurement terminal retrieved",
        )
        direct_keys.add(key)
        records.append(
            TargetedOverlayRecord(
                component_id=str(record.get("component_id")),
                target=RouteTarget(role=role, canonical_smiles=canonical),
                channel="targeted_exact_procurement",
                route_id=None,
            )
        )

    sources = evidence.get("literature_sources")
    if not isinstance(sources, list):
        raise Ugi3TargetedExactOverlayError("literature sources must be a list")
    source_by_id = {
        str(source.get("source_id")): source for source in sources if isinstance(source, dict)
    }
    routes = evidence.get("exact_routes")
    if not isinstance(routes, list):
        raise Ugi3TargetedExactOverlayError("exact routes must be a list")
    retained_alternatives = 0
    for index, route in enumerate(routes):
        if not isinstance(route, dict):
            raise Ugi3TargetedExactOverlayError("exact route must be an object")
        target_record = route.get("target")
        reactant_record = route.get("reactant")
        if not isinstance(target_record, dict) or not isinstance(reactant_record, dict):
            raise Ugi3TargetedExactOverlayError("exact route identity is malformed")
        role = str(route.get("role"))
        target_smiles = _canonical(
            target_record.get("canonical_smiles"), label="exact route target"
        )
        root_key = (role, target_smiles)
        if root_key in direct_keys:
            retained_alternatives += 1
            continue
        source = source_by_id.get(str(route.get("source_id")))
        if source is None:
            raise Ugi3TargetedExactOverlayError("exact route references unknown source")
        source_hash = source.get("sha256")
        if not isinstance(source_hash, str):
            raise Ugi3TargetedExactOverlayError("exact route source lacks SHA-256")
        component_id = str(route.get("component_id"))
        route_id = str(route.get("route_id"))
        reactant_smiles = _canonical(
            reactant_record.get("canonical_smiles"), label="exact route reactant"
        )
        upstream_role = f"ugi3_targeted_upstream:{component_id}:{route_id}"
        upstream_target = RouteTarget(
            role=upstream_role,
            canonical_smiles=reactant_smiles,
        )
        exact_evidence = EvidenceRecord(
            evidence_id=f"targeted-exact-route:{route_id}",
            tier=EvidenceTier.EXACT_SOURCE,
            source_sha256=source_hash,
            source_locator=str(source.get("locator")),
            exact_substrate=True,
            forward_verification=ForwardVerificationState.VERIFIED_EXACT_PRODUCT_UNIQUE,
            availability=AvailabilityState.UNASSESSED,
        )
        proposal = RouteStepProposal(
            reaction_id=str(route.get("reaction_id")),
            reactants=(upstream_target,),
            evidence=(exact_evidence,),
            forward_product_count=1,
            verifier_calls_required=1,
            product_candidates_considered=1,
        )
        if root_key in root_results:
            raise Ugi3TargetedExactOverlayError("duplicate selected exact route")
        root_results[root_key] = KnowledgeResult(
            disposition=KnowledgeDisposition.EXPAND,
            evidence=(exact_evidence,),
            proposal=proposal,
            detail="targeted exact-source route retrieved and forward verified",
        )
        upstream_key = (upstream_role, reactant_smiles)
        terminal_evidence = EvidenceRecord(
            evidence_id=f"targeted-upstream-terminal:{route_id}",
            tier=EvidenceTier.ACCEPTED_TERMINAL,
            source_sha256=terminal_hash,
            source_locator=(f"{terminal_procurement_path}#canonical_smiles={reactant_smiles}"),
            exact_substrate=True,
            forward_verification=ForwardVerificationState.NOT_APPLICABLE,
            availability=AvailabilityState.CURRENT_CLOSED,
        )
        upstream_results[upstream_key] = KnowledgeResult(
            disposition=KnowledgeDisposition.TERMINAL,
            evidence=(terminal_evidence,),
            detail="targeted exact route terminates at current frozen L3 evidence",
        )
        records.append(
            TargetedOverlayRecord(
                component_id=component_id,
                target=RouteTarget(role=role, canonical_smiles=target_smiles),
                channel="targeted_exact_route",
                route_id=route_id,
            )
        )

    unresolved = evidence.get("unresolved_predeclared_target")
    if not isinstance(unresolved, dict) or unresolved.get("disposition") != "abstain":
        raise Ugi3TargetedExactOverlayError("unresolved target must remain an abstention")
    unresolved_target = RouteTarget(
        role=str(unresolved.get("role")),
        canonical_smiles=_canonical(unresolved.get("canonical_smiles"), label="unresolved target"),
    )
    if len(root_results) != 4 or len(records) != 4:
        raise Ugi3TargetedExactOverlayError("overlay must select four exact target decisions")
    overlay = TargetedExactUgi3Overlay(
        base_source=base_source,
        root_results=root_results,
        upstream_results=upstream_results,
        records=tuple(sorted(records, key=lambda record: record.component_id)),
        unresolved_target=unresolved_target,
    )
    metadata = {
        "schema_version": OVERLAY_SCHEMA_VERSION,
        "evidence_pack_sha256": evidence_hash,
        "audit_result_sha256": sha256_file(stored_audit_result_path),
        "route_ledger_sha256": route_hash,
        "product_ledger_sha256": product_hash,
        "selected_exact_targets": len(records),
        "selected_direct_terminals": sum(
            record.channel == "targeted_exact_procurement" for record in records
        ),
        "selected_recursive_routes": sum(
            record.channel == "targeted_exact_route" for record in records
        ),
        "retained_unselected_exact_route_alternatives": retained_alternatives,
        "unresolved_abstentions": 1,
        "records": [
            {
                "component_id": record.component_id,
                "target": record.target.to_dict(),
                "channel": record.channel,
                "route_id": record.route_id,
            }
            for record in overlay.overlay_records
        ],
    }
    return overlay, metadata
