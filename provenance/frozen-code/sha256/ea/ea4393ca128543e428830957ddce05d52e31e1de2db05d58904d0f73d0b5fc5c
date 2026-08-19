"""Authenticated cumulative Ugi route-knowledge source for annotation rehearsals.

This module composes the frozen exact-evidence loaders additively.  It does not
compute a synthesis scalar, guide a generator, select candidates, inspect a
holdout, or normalize the declared molecular-support boundary.  The latter is
an explicit wrapper owned by :mod:`forge.route.ugi3_support_boundary`.

Every layer is rebuilt or authenticated from hash-pinned evidence before it is
exposed.  Exact-terminal deltas override only a role-qualified constitutional
identity; family, motif, provenance, and homologue evidence are never promoted.
Availability is evaluated at an explicit caller-supplied UTC timestamp.  No
wall clock or network access is used.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase

from forge.data.r1_prime_audit import sha256_file
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
from forge.route.ugi3_exact_c16_route import load_exact_c16_route_overlay
from forge.route.ugi3_exact_c18_route import load_exact_c18_route_overlay
from forge.route.ugi3_exact_evidence_source import load_exact_evidence_only_source
from forge.route.ugi3_high_leverage_head_terminals import (
    load_high_leverage_head_terminal_overlay,
)
from forge.route.ugi3_hybrid_search import load_bounded_hybrid_source
from forge.route.ugi3_second_wave_head_terminals import (
    load_second_wave_head_terminal_overlay,
)
from forge.route.ugi3_targeted_exact_overlay import load_targeted_exact_overlay
from forge.route.ugi3_targeted_role_gap_overlay import load_targeted_role_gap_overlay
from forge.route.ugi3_third_wave_head_terminals import (
    load_third_wave_head_terminal_overlay,
)
from forge.value.ugi3_fresh_pool_route_coverage_v2 import (
    ALDEHYDE_ROLE,
    ALDEHYDE_SMILES,
    _find_head_record,
    _find_tail_record,
    build_fresh_pool_route_coverage_v2,
)
from forge.value.ugi3_fresh_pool_route_coverage_v2 import (
    HEAD_ROLE as V2_HEAD_ROLE,
)
from forge.value.ugi3_fresh_pool_route_coverage_v2 import (
    HEAD_SMILES as V2_HEAD_SMILES,
)
from forge.value.ugi3_fresh_pool_route_coverage_v3 import (
    HEAD_ROLE as V3_HEAD_ROLE,
)
from forge.value.ugi3_fresh_pool_route_coverage_v3 import (
    HEAD_SMILES as V3_HEAD_SMILES,
)
from forge.value.ugi3_fresh_pool_route_coverage_v3 import (
    _find_octadecylamine_record,
    build_fresh_pool_route_coverage_v3,
)

SCHEMA_VERSION = "phase1_ugi3_cumulative_production_source.v1"
EXACT_TERMINAL_DELTA_SCHEMA_VERSION = "phase1_ugi3_authenticated_exact_terminal_delta.v1"

LAYER_ORDER = (
    "exact_evidence_base",
    "bounded_hybrid_search",
    "targeted_aldehyde_exact_overlay",
    "targeted_role_gap_overlay",
    "high_leverage_head_terminals",
    "second_wave_head_terminals",
    "third_wave_head_terminals",
    "authenticated_fresh_v2_v3_exact_terminals",
    "exact_c18_route",
    "exact_c16_route",
)


class Ugi3CumulativeProductionSourceError(ValueError):
    """Raised when the cumulative evidence chain cannot fail closed."""


@dataclass(frozen=True)
class AvailabilityWindow:
    """One authenticated L3 interval used in the unified validity window."""

    layer: str
    accessed_utc: str
    expires_utc: str
    source_locator: str

    def to_dict(self) -> dict[str, str]:
        return {
            "layer": self.layer,
            "accessed_utc": self.accessed_utc,
            "expires_utc": self.expires_utc,
            "source_locator": self.source_locator,
        }


@dataclass(frozen=True)
class CumulativeUgi3ProductionPaths:
    """Frozen local artifact locations for the cumulative loader chain."""

    exact_config: Path
    exact_result: Path
    exact_assessment: Path
    hybrid_config: Path
    hybrid_result: Path
    hybrid_assessment: Path
    targeted_aldehyde_config: Path
    targeted_aldehyde_result: Path
    targeted_aldehyde_route_ledger: Path
    targeted_aldehyde_product_ledger: Path
    role_gap_config: Path
    role_gap_result: Path
    role_gap_product_ledger: Path
    high_leverage_head_config: Path
    high_leverage_head_result: Path
    high_leverage_head_product_ledger: Path
    second_wave_head_config: Path
    second_wave_head_result: Path
    second_wave_head_product_ledger: Path
    third_wave_head_config: Path
    third_wave_head_result: Path
    third_wave_head_product_ledger: Path
    fresh_v2_config: Path
    fresh_v2_result: Path
    fresh_v2_component_ledger: Path
    fresh_v2_product_ledger: Path
    fresh_v3_config: Path
    fresh_v3_result: Path
    fresh_v3_component_ledger: Path
    fresh_v3_product_ledger: Path
    exact_c18_config: Path
    exact_c18_result: Path
    exact_c18_step_ledger: Path
    exact_c18_assessment: Path
    exact_c16_config: Path
    exact_c16_result: Path
    exact_c16_step_ledger: Path
    exact_c16_assessment: Path

    @classmethod
    def from_repo(cls, repo_root: Path) -> CumulativeUgi3ProductionPaths:
        repo = repo_root.resolve()
        route = repo / "configs/route"
        results = repo / "results/phase1"
        return cls(
            exact_config=route / "phase1_ugi3_exact_evidence_source.json",
            exact_result=results / "ugi3_exact_evidence_source/result.json",
            exact_assessment=results / "ugi3_exact_evidence_source/assessment_ledger.json.gz",
            hybrid_config=route / "phase1_ugi3_hybrid_search.json",
            hybrid_result=results / "ugi3_hybrid_search/result.json",
            hybrid_assessment=results / "ugi3_hybrid_search/assessment_ledger.json.gz",
            targeted_aldehyde_config=route / "phase1_ugi3_targeted_aldehyde_evidence_audit_v1.json",
            targeted_aldehyde_result=results
            / "ugi3_targeted_aldehyde_evidence_audit_v1/result.json",
            targeted_aldehyde_route_ledger=results
            / "ugi3_targeted_aldehyde_evidence_audit_v1/route_verification_ledger.csv.gz",
            targeted_aldehyde_product_ledger=results
            / "ugi3_targeted_aldehyde_evidence_audit_v1/product_closure_impact_ledger.csv.gz",
            role_gap_config=route / "phase1_ugi3_targeted_role_gap_evidence_audit_v1.json",
            role_gap_result=results / "ugi3_targeted_role_gap_evidence_audit_v1/result.json",
            role_gap_product_ledger=results
            / "ugi3_targeted_role_gap_evidence_audit_v1/product_impact_ledger.csv.gz",
            high_leverage_head_config=route
            / "phase1_ugi3_high_leverage_head_terminal_audit_v1.json",
            high_leverage_head_result=results
            / "ugi3_high_leverage_head_terminal_audit_v1/result.json",
            high_leverage_head_product_ledger=results
            / "ugi3_high_leverage_head_terminal_audit_v1/product_impact_ledger.csv.gz",
            second_wave_head_config=route / "phase1_ugi3_second_wave_head_terminal_audit_v1.json",
            second_wave_head_result=results / "ugi3_second_wave_head_terminal_audit_v1/result.json",
            second_wave_head_product_ledger=results
            / "ugi3_second_wave_head_terminal_audit_v1/product_impact_ledger.csv.gz",
            third_wave_head_config=route / "phase1_ugi3_third_wave_head_terminal_audit_v1.json",
            third_wave_head_result=results / "ugi3_third_wave_head_terminal_audit_v1/result.json",
            third_wave_head_product_ledger=results
            / "ugi3_third_wave_head_terminal_audit_v1/product_impact_ledger.csv.gz",
            fresh_v2_config=route / "phase1_ugi3_fresh_pool_route_coverage_v2.json",
            fresh_v2_result=results / "ugi3_fresh_pool_route_coverage_v2/result.json",
            fresh_v2_component_ledger=results
            / "ugi3_fresh_pool_route_coverage_v2/component_synthesis_values.json.gz",
            fresh_v2_product_ledger=results
            / "ugi3_fresh_pool_route_coverage_v2/product_synthesis_values.json.gz",
            fresh_v3_config=route / "phase1_ugi3_fresh_pool_route_coverage_v3.json",
            fresh_v3_result=results / "ugi3_fresh_pool_route_coverage_v3/result.json",
            fresh_v3_component_ledger=results
            / "ugi3_fresh_pool_route_coverage_v3/component_synthesis_values.json.gz",
            fresh_v3_product_ledger=results
            / "ugi3_fresh_pool_route_coverage_v3/product_synthesis_values.json.gz",
            exact_c18_config=route / "phase1_ugi3_exact_c18_route_v1.json",
            exact_c18_result=results / "ugi3_exact_c18_route_v1/result.json",
            exact_c18_step_ledger=results
            / "ugi3_exact_c18_route_v1/step_verification_ledger.json.gz",
            exact_c18_assessment=results / "ugi3_exact_c18_route_v1/assessment.json.gz",
            exact_c16_config=route / "phase1_ugi3_exact_c16_route_v1.json",
            exact_c16_result=results / "ugi3_exact_c16_route_v1/result.json",
            exact_c16_step_ledger=results
            / "ugi3_exact_c16_route_v1/step_verification_ledger.json.gz",
            exact_c16_assessment=results / "ugi3_exact_c16_route_v1/assessment.json.gz",
        )


class AuthenticatedExactTerminalDeltaOverlay:
    """Override authenticated exact terminals and delegate every other target."""

    def __init__(
        self,
        *,
        base_source: RouteKnowledgeSource,
        terminal_results: Mapping[tuple[str, str], KnowledgeResult],
    ) -> None:
        self._base_source = base_source
        validated: dict[tuple[str, str], KnowledgeResult] = {}
        for raw_key, result in terminal_results.items():
            if (
                not isinstance(raw_key, tuple)
                or len(raw_key) != 2
                or not isinstance(raw_key[0], str)
                or not raw_key[0]
            ):
                raise Ugi3CumulativeProductionSourceError(
                    "exact-terminal key must contain role and canonical SMILES"
                )
            key = (
                raw_key[0],
                _canonical(raw_key[1], label="exact-terminal overlay identity"),
            )
            if key in validated:
                raise Ugi3CumulativeProductionSourceError(
                    "duplicate canonical exact-terminal overlay identity"
                )
            if (
                result.disposition is not KnowledgeDisposition.TERMINAL
                or not result.evidence
                or any(
                    record.tier is not EvidenceTier.ACCEPTED_TERMINAL
                    or record.exact_substrate is not True
                    or record.forward_verification is not ForwardVerificationState.NOT_APPLICABLE
                    or record.availability is not AvailabilityState.CURRENT_CLOSED
                    for record in result.evidence
                )
            ):
                raise Ugi3CumulativeProductionSourceError(
                    "exact-terminal overlay cannot promote non-exact or non-current evidence"
                )
            validated[key] = result
        self._terminal_results = validated

    @property
    def targets(self) -> tuple[RouteTarget, ...]:
        return tuple(
            RouteTarget(role=role, canonical_smiles=smiles)
            for role, smiles in sorted(self._terminal_results)
        )

    def lookup(self, target: RouteTarget) -> KnowledgeResult:
        key = (
            target.role,
            _canonical(target.canonical_smiles, label="exact-terminal route target"),
        )
        result = self._terminal_results.get(key)
        return result if result is not None else self._base_source.lookup(target)


class CumulativeProductionUgi3Source:
    """Thin typed wrapper retaining the authenticated layer order and manifest."""

    def __init__(
        self,
        *,
        delegate: RouteKnowledgeSource,
        metadata: Mapping[str, Any],
    ) -> None:
        self._delegate = delegate
        self._metadata = dict(metadata)

    @property
    def layer_order(self) -> tuple[str, ...]:
        return LAYER_ORDER

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def lookup(self, target: RouteTarget) -> KnowledgeResult:
        return self._delegate.lookup(target)


def _canonical(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise Ugi3CumulativeProductionSourceError(f"{label} must be nonempty SMILES")
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(value)
    if molecule is None:
        raise Ugi3CumulativeProductionSourceError(f"{label} contains invalid SMILES")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise Ugi3CumulativeProductionSourceError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise Ugi3CumulativeProductionSourceError(f"{label} must be an object")
    return value


def _parse_utc(value: Any, *, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise Ugi3CumulativeProductionSourceError(f"{label} must be ISO-8601 UTC")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise Ugi3CumulativeProductionSourceError(f"{label} is invalid") from exc
    if parsed.tzinfo != timezone.utc:
        raise Ugi3CumulativeProductionSourceError(f"{label} must resolve to UTC")
    return parsed


def _utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _portable(path: Path, *, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _stable_sha256(value: Any) -> str:
    payload = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return sha256(payload).hexdigest()


def _resolve_config_inputs(
    config_path: Path,
    *,
    repo_root: Path,
    layer: str,
) -> tuple[dict[str, Any], dict[str, Path]]:
    config = _load_json(config_path, label=f"{layer} config")
    declared = config.get("inputs")
    if not isinstance(declared, dict) or not declared:
        raise Ugi3CumulativeProductionSourceError(f"{layer} config inputs are missing")
    repo = repo_root.resolve()
    paths: dict[str, Path] = {}
    for name, record in declared.items():
        if not isinstance(record, dict):
            raise Ugi3CumulativeProductionSourceError(f"{layer} input {name} must be an object")
        relative = record.get("asset", record.get("path"))
        expected = record.get("expected_sha256", record.get("sha256"))
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise Ugi3CumulativeProductionSourceError(
                f"{layer} input {name} lacks a path and SHA-256 pin"
            )
        path = (repo / relative).resolve()
        try:
            path.relative_to(repo)
        except ValueError as exc:
            raise Ugi3CumulativeProductionSourceError(
                f"{layer} input {name} escapes the repository"
            ) from exc
        observed = sha256_file(path)
        if observed != expected:
            raise Ugi3CumulativeProductionSourceError(f"{layer} input pin mismatch: {name}")
        paths[str(name)] = path
    return config, paths


def _authenticate_diagnostic(
    *,
    config: Mapping[str, Any],
    result_path: Path,
    ledger_path: Path,
    layer: str,
) -> dict[str, Any]:
    result = _load_json(result_path, label=f"{layer} result")
    if result.get("summary") != config.get("expected_counts"):
        raise Ugi3CumulativeProductionSourceError(
            f"{layer} result does not reproduce its expected counts"
        )
    if result.get("artifacts", {}).get("assessment_ledger_sha256") != sha256_file(ledger_path):
        raise Ugi3CumulativeProductionSourceError(
            f"{layer} result does not own its assessment ledger"
        )
    return result


def _snapshot_window(
    *,
    layer: str,
    snapshot: Mapping[str, Any],
    source_locator: str,
    accessed_key: str = "accessed_utc",
    expires_key: str = "expires_utc",
    expiry_days_key: str = "expiry_days",
    minimum_accessed_utc: str | None = None,
) -> AvailabilityWindow:
    accessed = _parse_utc(snapshot.get(accessed_key), label=f"{layer} accessed UTC")
    expires_value = snapshot.get(expires_key)
    if expires_value is None:
        days = snapshot.get(expiry_days_key)
        if isinstance(days, bool) or not isinstance(days, int) or days <= 0:
            raise Ugi3CumulativeProductionSourceError(
                f"{layer} expiry policy is missing or invalid"
            )
        expires = accessed + timedelta(days=days)
    else:
        expires = _parse_utc(expires_value, label=f"{layer} expires UTC")
    if minimum_accessed_utc is not None:
        accessed = max(
            accessed,
            _parse_utc(minimum_accessed_utc, label=f"{layer} assessment UTC"),
        )
    if expires < accessed:
        raise Ugi3CumulativeProductionSourceError(f"{layer} availability window is empty")
    return AvailabilityWindow(
        layer=layer,
        accessed_utc=_utc(accessed),
        expires_utc=_utc(expires),
        source_locator=source_locator,
    )


def _terminal_result(
    *,
    evidence_id: str,
    source_path: Path,
    source_locator: str,
) -> KnowledgeResult:
    return KnowledgeResult(
        disposition=KnowledgeDisposition.TERMINAL,
        evidence=(
            EvidenceRecord(
                evidence_id=evidence_id,
                tier=EvidenceTier.ACCEPTED_TERMINAL,
                source_sha256=sha256_file(source_path),
                source_locator=source_locator,
                exact_substrate=True,
                forward_verification=ForwardVerificationState.NOT_APPLICABLE,
                availability=AvailabilityState.CURRENT_CLOSED,
            ),
        ),
        detail="authenticated exact-current item-level procurement terminal",
    )


def _authenticate_fresh_result(
    *,
    fresh_result: Mapping[str, Any],
    fresh_component_ledger: bytes,
    fresh_product_ledger: bytes,
    stored_result_path: Path,
    stored_component_ledger_path: Path,
    stored_product_ledger_path: Path,
    layer: str,
) -> None:
    stored_result = _load_json(stored_result_path, label=f"{layer} stored result")
    if dict(fresh_result) != stored_result:
        raise Ugi3CumulativeProductionSourceError(f"{layer} frozen result is not reproducible")
    if stored_component_ledger_path.read_bytes() != fresh_component_ledger:
        raise Ugi3CumulativeProductionSourceError(
            f"{layer} frozen component ledger is not reproducible"
        )
    if stored_product_ledger_path.read_bytes() != fresh_product_ledger:
        raise Ugi3CumulativeProductionSourceError(
            f"{layer} frozen product ledger is not reproducible"
        )
    artifacts = stored_result.get("artifacts", {})
    if artifacts.get("component_synthesis_values.json.gz", {}).get("sha256") != sha256_file(
        stored_component_ledger_path
    ) or artifacts.get("product_synthesis_values.json.gz", {}).get("sha256") != sha256_file(
        stored_product_ledger_path
    ):
        raise Ugi3CumulativeProductionSourceError(f"{layer} result does not own its frozen ledgers")


def load_authenticated_fresh_exact_terminal_delta(
    *,
    base_source: RouteKnowledgeSource,
    repo_root: Path,
    assessment_as_of_utc: str,
    paths: CumulativeUgi3ProductionPaths,
) -> tuple[
    AuthenticatedExactTerminalDeltaOverlay,
    dict[str, Any],
    tuple[AvailabilityWindow, ...],
]:
    """Reproduce v2/v3 ledgers, then derive three terminals from raw evidence.

    The returned decisions do not deserialize or trust ``ComponentSynthesisValue``
    records.  Those value ledgers are reproduced only as an ownership check; the
    route source is constructed independently from the authenticated evidence.
    """

    v2_config, v2_inputs = _resolve_config_inputs(
        paths.fresh_v2_config,
        repo_root=repo_root,
        layer="fresh_v2_exact_terminal_delta",
    )
    v3_config, v3_inputs = _resolve_config_inputs(
        paths.fresh_v3_config,
        repo_root=repo_root,
        layer="fresh_v3_exact_terminal_delta",
    )
    fresh_v2, component_v2, product_v2 = build_fresh_pool_route_coverage_v2(
        repo_root,
        paths.fresh_v2_config,
    )
    _authenticate_fresh_result(
        fresh_result=fresh_v2,
        fresh_component_ledger=component_v2,
        fresh_product_ledger=product_v2,
        stored_result_path=paths.fresh_v2_result,
        stored_component_ledger_path=paths.fresh_v2_component_ledger,
        stored_product_ledger_path=paths.fresh_v2_product_ledger,
        layer="fresh_v2_exact_terminal_delta",
    )
    fresh_v3, component_v3, product_v3 = build_fresh_pool_route_coverage_v3(
        repo_root,
        paths.fresh_v3_config,
    )
    _authenticate_fresh_result(
        fresh_result=fresh_v3,
        fresh_component_ledger=component_v3,
        fresh_product_ledger=product_v3,
        stored_result_path=paths.fresh_v3_result,
        stored_component_ledger_path=paths.fresh_v3_component_ledger,
        stored_product_ledger_path=paths.fresh_v3_product_ledger,
        layer="fresh_v3_exact_terminal_delta",
    )

    head_payload = _load_json(
        v2_inputs["head_procurement_evidence"],
        label="fresh-v2 head procurement evidence",
    )
    _find_head_record(head_payload)
    tail_payload = _load_json(
        v2_inputs["tail_terminal_evidence"],
        label="fresh-v2 tail terminal evidence",
    )
    _find_tail_record(
        tail_payload,
        identity_asset=v2_inputs["tail_identity_asset"],
        failed_archive=v2_inputs["tail_failed_archive"],
    )
    virtual_payload = _load_json(
        v3_inputs["virtual_terminal_procurement"],
        label="fresh-v3 virtual terminal procurement",
    )
    _find_octadecylamine_record(
        virtual_payload,
        assessment_as_of_utc=assessment_as_of_utc,
    )

    head_window = _snapshot_window(
        layer="fresh_v2_piperazine_head",
        snapshot=head_payload.get("snapshot", {}),
        source_locator=_portable(v2_inputs["head_procurement_evidence"], repo_root=repo_root),
    )
    tail_window = _snapshot_window(
        layer="fresh_v2_heptadecanal_tail",
        snapshot=tail_payload.get("snapshot", {}),
        source_locator=_portable(v2_inputs["tail_terminal_evidence"], repo_root=repo_root),
    )
    virtual_window = _snapshot_window(
        layer="fresh_v3_octadecylamine_head",
        snapshot=virtual_payload.get("snapshot", {}),
        source_locator=_portable(v3_inputs["virtual_terminal_procurement"], repo_root=repo_root),
        minimum_accessed_utc=str(v3_config.get("assessment_as_of_utc")),
    )
    assessment = _parse_utc(assessment_as_of_utc, label="assessment_as_of_utc")
    for window in (head_window, tail_window, virtual_window):
        if not (
            _parse_utc(window.accessed_utc, label=f"{window.layer} accessed")
            <= assessment
            < _parse_utc(window.expires_utc, label=f"{window.layer} expires")
        ):
            raise Ugi3CumulativeProductionSourceError(
                f"{window.layer} evidence is not current at assessment_as_of_utc"
            )

    terminal_results = {
        (V2_HEAD_ROLE, _canonical(V2_HEAD_SMILES, label="fresh-v2 head")): _terminal_result(
            evidence_id="fresh-v2-exact-terminal:A1591",
            source_path=v2_inputs["head_procurement_evidence"],
            source_locator=(
                f"{_portable(v2_inputs['head_procurement_evidence'], repo_root=repo_root)}"
                "#records[canonical_smiles=NC1CCNCC1]"
            ),
        ),
        (ALDEHYDE_ROLE, _canonical(ALDEHYDE_SMILES, label="fresh-v2 aldehyde")): (
            _terminal_result(
                evidence_id="fresh-v2-exact-terminal:H1295",
                source_path=v2_inputs["tail_terminal_evidence"],
                source_locator=(
                    f"{_portable(v2_inputs['tail_terminal_evidence'], repo_root=repo_root)}"
                    "#records[0]"
                ),
            )
        ),
        (V3_HEAD_ROLE, _canonical(V3_HEAD_SMILES, label="fresh-v3 head")): _terminal_result(
            evidence_id="fresh-v3-exact-terminal:O1408",
            source_path=v3_inputs["virtual_terminal_procurement"],
            source_locator=(
                f"{_portable(v3_inputs['virtual_terminal_procurement'], repo_root=repo_root)}"
                "#records[canonical_smiles=CCCCCCCCCCCCCCCCCCN]"
            ),
        ),
    }
    overlay = AuthenticatedExactTerminalDeltaOverlay(
        base_source=base_source,
        terminal_results=terminal_results,
    )
    metadata = {
        "schema_version": EXACT_TERMINAL_DELTA_SCHEMA_VERSION,
        "assessment_as_of_utc": assessment_as_of_utc,
        "targets": [target.to_dict() for target in overlay.targets],
        "target_count": len(terminal_results),
        "frozen_result_sha256": {
            "v2": sha256_file(paths.fresh_v2_result),
            "v3": sha256_file(paths.fresh_v3_result),
        },
        "frozen_component_ledger_sha256": {
            "v2": sha256_file(paths.fresh_v2_component_ledger),
            "v3": sha256_file(paths.fresh_v3_component_ledger),
        },
        "frozen_product_ledger_sha256": {
            "v2": sha256_file(paths.fresh_v2_product_ledger),
            "v3": sha256_file(paths.fresh_v3_product_ledger),
        },
        "values_used_as_route_truth": False,
        "identity_policy": "exact_role_qualified_canonical_constitution_only",
        "tier_promotion": False,
    }
    return overlay, metadata, (head_window, tail_window, virtual_window)


def _manifest_inputs(
    *,
    repo_root: Path,
    paths: CumulativeUgi3ProductionPaths,
    config_inputs: Mapping[str, Mapping[str, Path]],
) -> dict[str, dict[str, str]]:
    records: dict[str, dict[str, str]] = {}
    for name, value in paths.__dict__.items():
        path = value
        records[f"frozen:{name}"] = {
            "path": _portable(path, repo_root=repo_root),
            "sha256": sha256_file(path),
        }
    for layer, inputs in sorted(config_inputs.items()):
        for name, path in sorted(inputs.items()):
            records[f"{layer}:input:{name}"] = {
                "path": _portable(path, repo_root=repo_root),
                "sha256": sha256_file(path),
            }
    return dict(sorted(records.items()))


def load_cumulative_production_ugi3_source(
    *,
    repo_root: Path,
    assessment_as_of_utc: str,
    paths: CumulativeUgi3ProductionPaths | None = None,
) -> tuple[CumulativeProductionUgi3Source, dict[str, Any]]:
    """Authenticate and compose the additive production Ugi source.

    This source is suitable for a zero-guidance annotation rehearsal only.  It
    intentionally stops before support-boundary normalization, scalarization,
    generator guidance, biological scoring, selection, or holdout access.
    """

    repo = repo_root.resolve()
    frozen = paths or CumulativeUgi3ProductionPaths.from_repo(repo)
    assessment = _parse_utc(assessment_as_of_utc, label="assessment_as_of_utc")
    configs: dict[str, dict[str, Any]] = {}
    inputs: dict[str, dict[str, Path]] = {}
    for layer, config_path in (
        ("exact_evidence_base", frozen.exact_config),
        ("bounded_hybrid_search", frozen.hybrid_config),
        ("targeted_aldehyde_exact_overlay", frozen.targeted_aldehyde_config),
        ("targeted_role_gap_overlay", frozen.role_gap_config),
        ("high_leverage_head_terminals", frozen.high_leverage_head_config),
        ("second_wave_head_terminals", frozen.second_wave_head_config),
        ("third_wave_head_terminals", frozen.third_wave_head_config),
        ("fresh_v2", frozen.fresh_v2_config),
        ("fresh_v3", frozen.fresh_v3_config),
        ("exact_c18_route", frozen.exact_c18_config),
        ("exact_c16_route", frozen.exact_c16_config),
    ):
        configs[layer], inputs[layer] = _resolve_config_inputs(
            config_path,
            repo_root=repo,
            layer=layer,
        )

    _authenticate_diagnostic(
        config=configs["exact_evidence_base"],
        result_path=frozen.exact_result,
        ledger_path=frozen.exact_assessment,
        layer="exact_evidence_base",
    )
    _authenticate_diagnostic(
        config=configs["bounded_hybrid_search"],
        result_path=frozen.hybrid_result,
        ledger_path=frozen.hybrid_assessment,
        layer="bounded_hybrid_search",
    )

    exact_policy = configs["exact_evidence_base"].get("diagnostic_policy", {})
    unavailable_exact = exact_policy.get("unavailable_procurement_statuses")
    if not isinstance(unavailable_exact, list) or any(
        not isinstance(value, str) for value in unavailable_exact
    ):
        raise Ugi3CumulativeProductionSourceError(
            "exact-evidence unavailable-procurement policy is malformed"
        )
    exact_inputs = inputs["exact_evidence_base"]
    source: RouteKnowledgeSource = load_exact_evidence_only_source(
        component_program_path=exact_inputs["component_program"],
        component_program_result_path=exact_inputs["component_program_result"],
        component_dossier_path=exact_inputs["component_dossier"],
        component_dossier_result_path=exact_inputs["component_dossier_result"],
        step_ledger_path=exact_inputs["step_ledger"],
        step_result_path=exact_inputs["step_result"],
        terminal_procurement_path=exact_inputs["terminal_procurement"],
        assessment_as_of_utc=assessment_as_of_utc,
        unavailable_procurement_statuses=frozenset(unavailable_exact),
    )

    hybrid_policy = configs["bounded_hybrid_search"].get("search_policy", {})
    unavailable_hybrid = hybrid_policy.get("unavailable_l3_statuses")
    expiry_days = hybrid_policy.get("retrieval_l3_expiry_days")
    if (
        not isinstance(unavailable_hybrid, list)
        or any(not isinstance(value, str) for value in unavailable_hybrid)
        or isinstance(expiry_days, bool)
        or not isinstance(expiry_days, int)
        or expiry_days <= 0
    ):
        raise Ugi3CumulativeProductionSourceError("hybrid L3 policy is malformed")
    hybrid_inputs = inputs["bounded_hybrid_search"]
    source, hybrid_metadata = load_bounded_hybrid_source(
        exact_source=source,
        readiness_ledger_path=hybrid_inputs["readiness_ledger"],
        readiness_result_path=hybrid_inputs["readiness_result"],
        readiness_config_path=hybrid_inputs["readiness_config"],
        transfer_ledger_path=hybrid_inputs["transfer_ledger"],
        transfer_result_path=hybrid_inputs["transfer_result"],
        transfer_config_path=hybrid_inputs["transfer_config"],
        upstream_registry_path=hybrid_inputs["upstream_registry"],
        oxidation_variant_path=hybrid_inputs["oxidation_variant"],
        assessment_as_of_utc=assessment_as_of_utc,
        retrieval_l3_expiry_days=expiry_days,
        unavailable_l3_statuses=frozenset(unavailable_hybrid),
    )

    source, targeted_metadata = load_targeted_exact_overlay(
        base_source=source,
        audit_config_path=frozen.targeted_aldehyde_config,
        audit_input_paths=inputs["targeted_aldehyde_exact_overlay"],
        stored_audit_result_path=frozen.targeted_aldehyde_result,
        stored_route_ledger_path=frozen.targeted_aldehyde_route_ledger,
        stored_product_ledger_path=frozen.targeted_aldehyde_product_ledger,
    )
    source, role_gap_metadata = load_targeted_role_gap_overlay(
        base_source=source,
        audit_config_path=frozen.role_gap_config,
        audit_input_paths=inputs["targeted_role_gap_overlay"],
        stored_audit_result_path=frozen.role_gap_result,
        stored_product_ledger_path=frozen.role_gap_product_ledger,
    )
    source, high_head_metadata = load_high_leverage_head_terminal_overlay(
        base_source=source,
        audit_config_path=frozen.high_leverage_head_config,
        audit_input_paths=inputs["high_leverage_head_terminals"],
        stored_audit_result_path=frozen.high_leverage_head_result,
        stored_product_ledger_path=frozen.high_leverage_head_product_ledger,
    )
    source, second_head_metadata = load_second_wave_head_terminal_overlay(
        base_source=source,
        audit_config_path=frozen.second_wave_head_config,
        audit_input_paths=inputs["second_wave_head_terminals"],
        stored_audit_result_path=frozen.second_wave_head_result,
        stored_product_ledger_path=frozen.second_wave_head_product_ledger,
    )
    source, third_head_metadata = load_third_wave_head_terminal_overlay(
        base_source=source,
        audit_config_path=frozen.third_wave_head_config,
        audit_input_paths=inputs["third_wave_head_terminals"],
        stored_audit_result_path=frozen.third_wave_head_result,
        stored_product_ledger_path=frozen.third_wave_head_product_ledger,
    )
    source, fresh_metadata, fresh_windows = load_authenticated_fresh_exact_terminal_delta(
        base_source=source,
        repo_root=repo,
        assessment_as_of_utc=assessment_as_of_utc,
        paths=frozen,
    )
    source = load_exact_c18_route_overlay(
        base_source=source,
        config_path=frozen.exact_c18_config,
        input_paths=inputs["exact_c18_route"],
        stored_result_path=frozen.exact_c18_result,
        stored_step_ledger_path=frozen.exact_c18_step_ledger,
        stored_assessment_path=frozen.exact_c18_assessment,
    )
    source = load_exact_c16_route_overlay(
        base_source=source,
        config_path=frozen.exact_c16_config,
        input_paths=inputs["exact_c16_route"],
        stored_result_path=frozen.exact_c16_result,
        stored_step_ledger_path=frozen.exact_c16_step_ledger,
        stored_assessment_path=frozen.exact_c16_assessment,
    )

    hybrid_l3 = hybrid_policy.get("combined_l3_context")
    if not isinstance(hybrid_l3, dict):
        raise Ugi3CumulativeProductionSourceError("hybrid combined L3 context is missing")
    windows: list[AvailabilityWindow] = [
        _snapshot_window(
            layer="bounded_hybrid_search",
            snapshot={
                "accessed_utc": hybrid_l3.get("latest_accessed_utc"),
                "expires_utc": hybrid_l3.get("earliest_expires_utc"),
            },
            source_locator=_portable(frozen.hybrid_config, repo_root=repo),
            minimum_accessed_utc=str(hybrid_policy.get("assessment_as_of_utc")),
        )
    ]

    targeted_evidence = _load_json(
        inputs["targeted_aldehyde_exact_overlay"]["evidence_pack"],
        label="targeted aldehyde evidence pack",
    )
    targeted_snapshot = targeted_evidence.get("snapshot")
    if not isinstance(targeted_snapshot, dict):
        raise Ugi3CumulativeProductionSourceError("targeted aldehyde evidence snapshot is missing")
    windows.append(
        _snapshot_window(
            layer="targeted_aldehyde_exact_overlay",
            snapshot={
                "accessed_utc": targeted_snapshot.get("assessment_as_of_utc"),
                "expiry_days": targeted_snapshot.get("procurement_expiry_days"),
            },
            source_locator=_portable(
                inputs["targeted_aldehyde_exact_overlay"]["evidence_pack"],
                repo_root=repo,
            ),
        )
    )

    role_evidence = _load_json(
        inputs["targeted_role_gap_overlay"]["evidence_pack"],
        label="role-gap evidence pack",
    )
    role_vendor = role_evidence.get("direct_procurement_terminal", {}).get("vendor_evidence")
    if not isinstance(role_vendor, dict):
        raise Ugi3CumulativeProductionSourceError("role-gap L3 evidence is missing")
    windows.append(
        _snapshot_window(
            layer="targeted_role_gap_overlay",
            snapshot=role_vendor,
            source_locator=_portable(
                inputs["targeted_role_gap_overlay"]["evidence_pack"], repo_root=repo
            ),
            minimum_accessed_utc=str(role_evidence.get("assessment_as_of_utc")),
        )
    )

    for layer in (
        "high_leverage_head_terminals",
        "second_wave_head_terminals",
        "third_wave_head_terminals",
    ):
        evidence_path = inputs[layer]["evidence_pack"]
        evidence = _load_json(evidence_path, label=f"{layer} evidence pack")
        snapshot = evidence.get("snapshot")
        if not isinstance(snapshot, dict):
            raise Ugi3CumulativeProductionSourceError(f"{layer} snapshot is missing")
        windows.append(
            _snapshot_window(
                layer=layer,
                snapshot=snapshot,
                source_locator=_portable(evidence_path, repo_root=repo),
            )
        )
    windows.extend(fresh_windows)

    c18_config = configs["exact_c18_route"]
    c18_terminal = c18_config.get("terminal_procurement")
    if not isinstance(c18_terminal, dict):
        raise Ugi3CumulativeProductionSourceError("C18 terminal evidence is missing")
    windows.append(
        _snapshot_window(
            layer="exact_c18_route",
            snapshot=c18_terminal,
            source_locator=_portable(frozen.exact_c18_config, repo_root=repo),
            accessed_key="observed_at_utc",
            expires_key="expires_at_utc",
            minimum_accessed_utc=str(c18_config.get("assessment_as_of_utc")),
        )
    )
    c16_config = configs["exact_c16_route"]
    c16_terminals = c16_config.get("terminal_procurement")
    if not isinstance(c16_terminals, list) or not c16_terminals:
        raise Ugi3CumulativeProductionSourceError("C16 terminal evidence is missing")
    c16_observed = max(
        _parse_utc(item.get("observed_at_utc"), label="C16 observed UTC")
        for item in c16_terminals
        if isinstance(item, dict)
    )
    c16_expires = min(
        _parse_utc(item.get("expires_at_utc"), label="C16 expires UTC")
        for item in c16_terminals
        if isinstance(item, dict)
    )
    windows.append(
        _snapshot_window(
            layer="exact_c16_route",
            snapshot={
                "observed_at_utc": _utc(c16_observed),
                "expires_at_utc": _utc(c16_expires),
            },
            source_locator=_portable(frozen.exact_c16_config, repo_root=repo),
            accessed_key="observed_at_utc",
            expires_key="expires_at_utc",
            minimum_accessed_utc=str(c16_config.get("assessment_as_of_utc")),
        )
    )

    unified_accessed = max(
        _parse_utc(window.accessed_utc, label=f"{window.layer} accessed UTC") for window in windows
    )
    unified_expires = min(
        _parse_utc(window.expires_utc, label=f"{window.layer} expires UTC") for window in windows
    )
    if unified_expires < unified_accessed:
        raise Ugi3CumulativeProductionSourceError(
            "authenticated L3 sources have no overlapping validity window"
        )
    if assessment < unified_accessed or assessment >= unified_expires:
        raise Ugi3CumulativeProductionSourceError(
            "assessment_as_of_utc is outside the unified authenticated L3 window: "
            f"{_utc(unified_accessed)} through {_utc(unified_expires)}"
        )

    manifest_inputs = _manifest_inputs(
        repo_root=repo,
        paths=frozen,
        config_inputs=inputs,
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "assessment_as_of_utc": assessment_as_of_utc,
        "layer_order": list(LAYER_ORDER),
        "unified_l3_window": {
            "accessed_utc": _utc(unified_accessed),
            "expires_utc": _utc(unified_expires),
            "interval_semantics": "accessed_inclusive_expires_exclusive",
        },
        "l3_windows": [window.to_dict() for window in windows],
        "layer_metadata": {
            "bounded_hybrid_search": hybrid_metadata,
            "targeted_aldehyde_exact_overlay": targeted_metadata,
            "targeted_role_gap_overlay": role_gap_metadata,
            "high_leverage_head_terminals": high_head_metadata,
            "second_wave_head_terminals": second_head_metadata,
            "third_wave_head_terminals": third_head_metadata,
            "authenticated_fresh_v2_v3_exact_terminals": fresh_metadata,
            "exact_c18_route": {
                "result_sha256": sha256_file(frozen.exact_c18_result),
            },
            "exact_c16_route": {
                "result_sha256": sha256_file(frozen.exact_c16_result),
            },
        },
        "inputs": manifest_inputs,
        "inputs_sha256": _stable_sha256(manifest_inputs),
        "scope": {
            "zero_guidance_annotation_rehearsal": True,
            "support_boundary_wrapped": False,
            "synthesis_scalar_computed": False,
            "generator_guidance_authorized": False,
            "biology_used": False,
            "selection_used": False,
            "holdout_accessed": False,
        },
        "evidence_policy": {
            "exact_identity_only_can_close": True,
            "family_or_motif_or_provenance_can_close": False,
            "homologue_promotion": False,
            "raw_value_ledgers_used_as_route_truth": False,
        },
    }
    wrapped = CumulativeProductionUgi3Source(delegate=source, metadata=metadata)
    return wrapped, metadata
