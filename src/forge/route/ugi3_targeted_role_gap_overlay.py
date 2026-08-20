"""Typed exact-terminal overlay for the targeted Ugi head evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase

from forge.core.hashing import sha256_bytes, sha256_file
from forge.core.io import read_json_object
from forge.route.planner import (
    AvailabilityState,
    EvidenceRecord,
    EvidenceTier,
    ForwardVerificationState,
    KnowledgeDisposition,
    KnowledgeResult,
    RouteKnowledgeSource,
    RouteTarget,
)
from forge.route.ugi3_targeted_role_gap_evidence import (
    EVIDENCE_SCHEMA_VERSION,
    build_targeted_role_gap_evidence_audit,
)

OVERLAY_SCHEMA_VERSION = "phase1_ugi3_targeted_role_gap_overlay.v1"


class Ugi3TargetedRoleGapOverlayError(ValueError):
    """Raised when the head terminal cannot be composed exactly."""


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    return read_json_object(path, error=Ugi3TargetedRoleGapOverlayError, label=label)


def _canonical(smiles: Any, *, label: str) -> str:
    if not isinstance(smiles, str) or not smiles:
        raise Ugi3TargetedRoleGapOverlayError(f"{label} must be nonempty SMILES")
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise Ugi3TargetedRoleGapOverlayError(f"{label} contains invalid SMILES")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


class TargetedRoleGapUgi3Overlay:
    """Override one exact head identity and delegate every other target."""

    def __init__(
        self,
        *,
        base_source: RouteKnowledgeSource,
        head_target: RouteTarget,
        head_result: KnowledgeResult,
        unresolved_isocyanide_target: RouteTarget,
    ):
        self._base_source = base_source
        self._head_target = head_target
        self._head_result = head_result
        self._unresolved_isocyanide_target = unresolved_isocyanide_target

    @property
    def head_target(self) -> RouteTarget:
        return self._head_target

    @property
    def unresolved_isocyanide_target(self) -> RouteTarget:
        return self._unresolved_isocyanide_target

    def lookup(self, target: RouteTarget) -> KnowledgeResult:
        canonical = _canonical(target.canonical_smiles, label="role-gap route target")
        if (target.role, canonical) == (
            self._head_target.role,
            self._head_target.canonical_smiles,
        ):
            return self._head_result
        return self._base_source.lookup(target)


def load_targeted_role_gap_overlay(
    *,
    base_source: RouteKnowledgeSource,
    audit_config_path: Path,
    audit_input_paths: dict[str, Path],
    stored_audit_result_path: Path,
    stored_product_ledger_path: Path,
) -> tuple[TargetedRoleGapUgi3Overlay, dict[str, Any]]:
    """Authenticate the role-gap audit and expose its one closing decision."""

    fresh_result, fresh_ledger = build_targeted_role_gap_evidence_audit(
        config_path=audit_config_path,
        input_paths=audit_input_paths,
    )
    stored_result = _load_json(stored_audit_result_path, label="stored role-gap result")
    if fresh_result != stored_result:
        raise Ugi3TargetedRoleGapOverlayError("stored role-gap result is not reproducible")
    ledger_hash = sha256_bytes(fresh_ledger)
    if sha256_file(stored_product_ledger_path) != ledger_hash:
        raise Ugi3TargetedRoleGapOverlayError("stored role-gap ledger hash mismatch")
    if (
        stored_result.get("artifacts", {}).get("product_impact_ledger.csv.gz", {}).get("sha256")
        != ledger_hash
    ):
        raise Ugi3TargetedRoleGapOverlayError("role-gap result does not own its ledger")

    evidence_path = audit_input_paths.get("evidence_pack")
    if evidence_path is None:
        raise Ugi3TargetedRoleGapOverlayError("role-gap overlay lacks its evidence pack")
    evidence = _load_json(evidence_path, label="role-gap evidence pack")
    if evidence.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
        raise Ugi3TargetedRoleGapOverlayError("unsupported role-gap evidence schema")
    head = evidence.get("direct_procurement_terminal")
    unresolved = evidence.get("unresolved_isocyanide_target")
    if not isinstance(head, dict) or not isinstance(unresolved, dict):
        raise Ugi3TargetedRoleGapOverlayError("role-gap targets are malformed")
    if (
        head.get("disposition") != "admit_exact_terminal"
        or unresolved.get("disposition") != "abstain"
    ):
        raise Ugi3TargetedRoleGapOverlayError("role-gap dispositions are not fail-closed")
    head_target = RouteTarget(
        role=str(head.get("role")),
        canonical_smiles=_canonical(head.get("canonical_smiles"), label="head target"),
    )
    unresolved_target = RouteTarget(
        role=str(unresolved.get("role")),
        canonical_smiles=_canonical(
            unresolved.get("canonical_smiles"), label="unresolved isocyanide target"
        ),
    )
    evidence_record = EvidenceRecord(
        evidence_id=f"targeted-head-terminal:{head.get('component_id')}",
        tier=EvidenceTier.ACCEPTED_TERMINAL,
        source_sha256=sha256_file(evidence_path),
        source_locator=(f"{evidence_path}#direct_procurement_terminal.vendor_evidence"),
        exact_substrate=True,
        forward_verification=ForwardVerificationState.NOT_APPLICABLE,
        availability=AvailabilityState.CURRENT_CLOSED,
    )
    head_result = KnowledgeResult(
        disposition=KnowledgeDisposition.TERMINAL,
        evidence=(evidence_record,),
        detail="targeted exact head has current preferred-partner marketplace evidence",
    )
    overlay = TargetedRoleGapUgi3Overlay(
        base_source=base_source,
        head_target=head_target,
        head_result=head_result,
        unresolved_isocyanide_target=unresolved_target,
    )
    metadata = {
        "schema_version": OVERLAY_SCHEMA_VERSION,
        "evidence_pack_sha256": sha256_file(evidence_path),
        "audit_result_sha256": sha256_file(stored_audit_result_path),
        "product_ledger_sha256": ledger_hash,
        "selected_exact_head_terminals": 1,
        "unresolved_isocyanide_abstentions": 1,
        "head_target": head_target.to_dict(),
        "unresolved_isocyanide_target": unresolved_target.to_dict(),
    }
    return overlay, metadata
