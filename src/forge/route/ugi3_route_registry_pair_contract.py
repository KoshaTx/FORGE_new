"""Validate the prereveal paired R0/R1 route-registry contract.

This module never generates, loads or inspects sealed holdout molecules.  It
validates the already sealed parent draw, then (when supplied) an immutable
registry-pair binding whose only authorized change is one exact C18 aldehyde
record.  Missing or incomplete C18 L3 evidence fails closed.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from forge.data.r1_prime_audit import sha256_file
from forge.route.ugi3_exact_c18_route import (
    Ugi3ExactC18RouteError,
    build_exact_c18_route_audit,
)
from forge.value.synthesis import ComponentSynthesisValue

PROTOCOL_SCHEMA_VERSION = "phase1_ugi3_route_registry_pair_protocol.v1"
SNAPSHOT_SCHEMA_VERSION = "phase1_ugi3_route_registry_snapshot.v1"
RECORD_LEDGER_SCHEMA_VERSION = "phase1_ugi3_route_registry_records.v1"
BINDING_SCHEMA_VERSION = "phase1_ugi3_route_registry_pair_binding.v1"
DIFF_SCHEMA_VERSION = "phase1_ugi3_route_registry_diff.v1"

TARGET_ROLE = "oxoester_aldehyde_body_tail"
TARGET_SMILES = "C#CCCCCCCCCCCCCCCCC=O"
TARGET_KEY_SHA256 = "854cc7cf4409f4a0c5c28bac78d57ca4b8ab36364626021790b9a9765ac19c3c"

REQUIRED_EXECUTION_POLICY = {
    "same_sealed_program_draw": True,
    "same_flow_seed": True,
    "same_terminal_seed": True,
    "same_sampled_molecules": True,
    "same_exact_l1_eligibility_ledger": True,
    "permit_retry": False,
    "permit_resampling": False,
    "permit_replanning": False,
    "permit_registry_tuning_after_reveal": False,
    "permit_threshold_tuning_after_reveal": False,
    "permit_post_reveal_evidence": False,
    "permit_post_reveal_route_repair": False,
    "permit_similarity_or_family_route_closure": False,
    "permit_molecule_inspection_before_registry_binding": False,
}

PRIVATE_ELIGIBILITY_FIELDS = (
    "sample_key_sha256",
    "valid",
    "component_reconstruction_valid",
    "exact_product_reconstructed",
    "eligible",
)
PRIVATE_COMPONENT_TRANSITION_FIELDS = (
    "sample_key_sha256",
    "component_position",
    "component_key_sha256",
    "role",
    "r0_route_complete",
    "r1_route_complete",
    "transition",
)
PRIVATE_PRODUCT_TRANSITION_FIELDS = (
    "sample_key_sha256",
    "eligible",
    "r0_route_complete",
    "r1_route_complete",
    "transition",
)
PUBLIC_AGGREGATE_FIELDS = (
    "sealed_program_rows",
    "sampled_products",
    "exact_l1_eligible_products",
    "eligible_products_complete_in_r0",
    "eligible_products_complete_in_r1",
    "marginal_completion_count",
    "marginal_completion_fraction",
    "confidence_interval_method",
    "confidence_interval_95pct",
    "target_component_occurrences",
    "target_component_occurrences_that_flip_product_completion",
    "aggregate_counts_by_evidence_family",
)
FORBIDDEN_PUBLIC_IDENTITY_FIELDS = (
    "canonical_product_smiles",
    "canonical_component_smiles",
    "product_id",
    "structure_id",
    "route_steps",
    "terminal_material_identities",
)


class Ugi3RouteRegistryPairContractError(ValueError):
    """Raised when the prereveal paired-registry contract is violated."""


def _stable_json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        raise Ugi3RouteRegistryPairContractError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise Ugi3RouteRegistryPairContractError(f"{label} must be a JSON object")
    return value


def _load_gzip_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        with gzip.open(path, "rt") as handle:
            value = json.load(handle)
    except (FileNotFoundError, gzip.BadGzipFile, json.JSONDecodeError, OSError) as exc:
        raise Ugi3RouteRegistryPairContractError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise Ugi3RouteRegistryPairContractError(f"{label} must be a JSON object")
    return value


def _repo_path(repo: Path, raw: Any, *, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise Ugi3RouteRegistryPairContractError(f"{label} path is missing")
    path = Path(raw)
    if path.is_absolute():
        return path
    return repo / path


def _validate_pinned_file(specification: Any, *, repo: Path, label: str) -> tuple[Path, str]:
    if not isinstance(specification, dict):
        raise Ugi3RouteRegistryPairContractError(f"{label} specification is malformed")
    path = _repo_path(repo, specification.get("path"), label=label)
    expected = specification.get("sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise Ugi3RouteRegistryPairContractError(f"{label} SHA-256 is malformed")
    try:
        observed = sha256_file(path)
    except OSError as exc:
        raise Ugi3RouteRegistryPairContractError(f"{label} is missing: {path}") from exc
    if observed != expected:
        raise Ugi3RouteRegistryPairContractError(f"{label} hash changed")
    return path, observed


def _require_exact_mapping(observed: Any, expected: Mapping[str, Any], *, label: str) -> None:
    if not isinstance(observed, dict) or observed != dict(expected):
        raise Ugi3RouteRegistryPairContractError(f"{label} changed")


def component_key_sha256(role: str, canonical_smiles: str) -> str:
    """Return the frozen exact role-and-identity component key."""

    return hashlib.sha256(f"{role}\t{canonical_smiles}".encode()).hexdigest()


def exact_l1_eligible(row: Mapping[str, Any]) -> bool:
    """Apply the preregistered three-way exact-L1 denominator gate."""

    forward = row.get("l1_forward_verification")
    return bool(
        row.get("valid") is True
        and row.get("component_reconstruction_valid") is True
        and isinstance(forward, Mapping)
        and forward.get("exact_product_reconstructed") is True
    )


def validate_public_aggregate(payload: Mapping[str, Any]) -> None:
    """Reject identity-bearing or incompletely declared public results."""

    if set(payload) != set(PUBLIC_AGGREGATE_FIELDS):
        raise Ugi3RouteRegistryPairContractError("public aggregate fields changed")
    if any(field in payload for field in FORBIDDEN_PUBLIC_IDENTITY_FIELDS):
        raise Ugi3RouteRegistryPairContractError(
            "public aggregate contains identity-bearing fields"
        )


def _validate_protocol_shape(protocol: Mapping[str, Any]) -> None:
    if protocol.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise Ugi3RouteRegistryPairContractError("unsupported paired protocol schema")
    if protocol.get("status") != "frozen_protocol_holdout_unrevealed_registry_pair_unbound":
        raise Ugi3RouteRegistryPairContractError("paired protocol status changed")
    frozen_at = protocol.get("frozen_at_utc")
    if not isinstance(frozen_at, str) or not frozen_at.endswith("Z"):
        raise Ugi3RouteRegistryPairContractError("protocol freeze timestamp is malformed")
    try:
        frozen_time = datetime.fromisoformat(frozen_at.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise Ugi3RouteRegistryPairContractError("protocol freeze timestamp is malformed") from exc
    if frozen_time > datetime.now(timezone.utc) + timedelta(minutes=5):
        raise Ugi3RouteRegistryPairContractError("protocol freeze timestamp is future-dated")
    _require_exact_mapping(
        protocol.get("execution_policy"), REQUIRED_EXECUTION_POLICY, label="execution policy"
    )
    authorized = protocol.get("authorized_delta")
    if not isinstance(authorized, dict):
        raise Ugi3RouteRegistryPairContractError("authorized delta is missing")
    expected_target = {
        "role": TARGET_ROLE,
        "canonical_smiles": TARGET_SMILES,
        "component_key_sha256": TARGET_KEY_SHA256,
        "r0_route_complete": False,
        "r1_route_complete": True,
        "required_exact_l2_steps": 3,
        "required_terminal_availability": "current_closed",
        "required_current_l3_procurement_closed": True,
        "required_family_template_admitted": False,
    }
    _require_exact_mapping(authorized, expected_target, label="authorized delta")
    if component_key_sha256(TARGET_ROLE, TARGET_SMILES) != TARGET_KEY_SHA256:
        raise Ugi3RouteRegistryPairContractError("authorized component key is inconsistent")

    pair = protocol.get("registry_pair")
    if not isinstance(pair, dict):
        raise Ugi3RouteRegistryPairContractError("registry-pair contract is missing")
    required_pair = {
        "snapshot_schema_version": SNAPSHOT_SCHEMA_VERSION,
        "record_ledger_schema_version": RECORD_LEDGER_SCHEMA_VERSION,
        "binding_schema_version": BINDING_SCHEMA_VERSION,
        "diff_schema_version": DIFF_SCHEMA_VERSION,
        "r0_role": "immutable_parent_registry",
        "r1_role": "immutable_child_registry",
        "r0_semantic_label": "without_c18",
        "r1_semantic_label": "with_c18",
        "required_delta_count": 1,
        "required_added_count": 0,
        "required_deleted_count": 0,
        "required_regression_count": 0,
        "required_non_target_mutation_count": 0,
        "all_other_evidence_including_any_previously_qualified_routes_is_identical": True,
        "identity_policy": {
            "match_on_exact_role_and_canonical_smiles": True,
            "permit_similarity_or_family_matching": False,
            "permit_neighbor_homologue_admission": False,
            "permit_reaction_family_template_promotion": False,
        },
    }
    _require_exact_mapping(pair, required_pair, label="registry-pair contract")

    eligibility = {
        "denominator_symbol": "H",
        "denominator_description": "sealed products passing all three exact-L1 gates",
        "required_true_fields": [
            "valid",
            "component_reconstruction_valid",
            "l1_forward_verification.exact_product_reconstructed",
        ],
        "missing_or_non_boolean_fields_are_ineligible": True,
        "no_post_reveal_denominator_changes": True,
    }
    _require_exact_mapping(
        protocol.get("exact_l1_eligibility"),
        eligibility,
        label="exact-L1 eligibility contract",
    )
    estimand = {
        "name": "marginal_exact_registry_route_coverage",
        "denominator_symbol": "H",
        "numerator_symbol": "D",
        "numerator_description": (
            "eligible sealed products that are noncomplete under R0 and complete under R1"
        ),
        "point_estimate": "D/H",
        "confidence_interval": "two_sided_clopper_pearson_95pct",
        "paired_unit": "same sealed product under both immutable registries",
        "secondary_counts": [
            "eligible_products_complete_in_r0",
            "eligible_products_complete_in_r1",
            "eligible_products_by_route_complete_component_count_r0",
            "eligible_products_by_route_complete_component_count_r1",
            "target_component_occurrences",
            "target_component_occurrences_that_flip_product_completion",
        ],
        "not_a_probability_of_experimental_synthesis_success": True,
    }
    _require_exact_mapping(protocol.get("paired_estimand"), estimand, label="paired estimand")
    binding_requirements = {
        "protocol_hash_must_be_pinned": True,
        "r0_and_r1_manifest_hashes_must_be_pinned": True,
        "registry_diff_hash_must_be_pinned": True,
        "r1_parent_must_be_exact_r0_manifest": True,
        "exact_c18_config_result_and_assessment_hashes_must_be_pinned": True,
        "exact_c18_must_be_complete_at_l2_and_l3": True,
        "registry_record_key_sets_must_match": True,
        "all_non_target_records_must_be_byte_equivalent_as_canonical_json": True,
        "binding_must_precede_molecular_holdout_reveal": True,
    }
    _require_exact_mapping(
        protocol.get("binding_requirements"),
        binding_requirements,
        label="binding requirements",
    )

    ledger = protocol.get("ledger_contract")
    if not isinstance(ledger, dict):
        raise Ugi3RouteRegistryPairContractError("ledger contract is missing")
    private = ledger.get("private_identity_ledgers")
    public = ledger.get("public_aggregate")
    if not isinstance(private, dict) or not isinstance(public, dict):
        raise Ugi3RouteRegistryPairContractError("ledger visibility contract is malformed")
    if private.get("visibility") != "private_hash_pinned":
        raise Ugi3RouteRegistryPairContractError("private ledger visibility changed")
    expected_private = {
        "exact_l1_eligibility_fields": list(PRIVATE_ELIGIBILITY_FIELDS),
        "component_transition_fields": list(PRIVATE_COMPONENT_TRANSITION_FIELDS),
        "product_transition_fields": list(PRIVATE_PRODUCT_TRANSITION_FIELDS),
    }
    for key, expected in expected_private.items():
        if private.get(key) != expected:
            raise Ugi3RouteRegistryPairContractError(f"private ledger {key} changed")
    if public.get("visibility") != "public_aggregate_only":
        raise Ugi3RouteRegistryPairContractError("public ledger visibility changed")
    if public.get("fields") != list(PUBLIC_AGGREGATE_FIELDS):
        raise Ugi3RouteRegistryPairContractError("public aggregate schema changed")
    if public.get("forbidden_identity_fields") != list(FORBIDDEN_PUBLIC_IDENTITY_FIELDS):
        raise Ugi3RouteRegistryPairContractError("public privacy exclusions changed")

    decision = protocol.get("decision")
    if not isinstance(decision, dict) or decision.get("protocol_frozen") is not True:
        raise Ugi3RouteRegistryPairContractError("protocol is not frozen")
    if decision.get("registry_pair_bound") is not False:
        raise Ugi3RouteRegistryPairContractError("unbound protocol claims a registry pair")
    if decision.get("holdout_reveal_authorized") is not False:
        raise Ugi3RouteRegistryPairContractError("protocol prematurely authorizes reveal")


def validate_protocol(repo: Path, protocol_path: Path) -> dict[str, Any]:
    """Validate the frozen protocol and sealed parent without revealing molecules."""

    protocol = _load_json(protocol_path, label="paired protocol")
    _validate_protocol_shape(protocol)
    anchor = protocol.get("sealed_holdout_anchor")
    if not isinstance(anchor, dict):
        raise Ugi3RouteRegistryPairContractError("sealed holdout anchor is missing")

    contract_path, contract_hash = _validate_pinned_file(
        anchor.get("contract"), repo=repo, label="sealed holdout contract"
    )
    seal_path, seal_hash = _validate_pinned_file(
        anchor.get("seal"), repo=repo, label="sealed holdout seal"
    )
    _, program_hash = _validate_pinned_file(
        anchor.get("program_draw"), repo=repo, label="sealed program draw"
    )
    contract = _load_json(contract_path, label="sealed holdout contract")
    seal = _load_json(seal_path, label="sealed holdout seal")

    contract_spec = anchor["contract"]
    seal_spec = anchor["seal"]
    program_spec = anchor["program_draw"]
    if contract.get("schema_version") != contract_spec.get("schema_version"):
        raise Ugi3RouteRegistryPairContractError("holdout contract schema changed")
    if seal.get("schema_version") != seal_spec.get("schema_version"):
        raise Ugi3RouteRegistryPairContractError("holdout seal schema changed")
    if seal.get("status") != seal_spec.get("status"):
        raise Ugi3RouteRegistryPairContractError("holdout seal status changed")
    if seal.get("contract") != {
        "path": contract_spec["path"],
        "sha256": contract_hash,
    }:
        raise Ugi3RouteRegistryPairContractError("seal does not own the holdout contract")
    if seal.get("program_draw") != {
        "path": program_spec["path"],
        "sha256": program_hash,
        "rows": program_spec["rows"],
        "seed": anchor["seeds"]["program"],
    }:
        raise Ugi3RouteRegistryPairContractError("seal does not own the program draw")
    expected_holdout = {
        "generated": False,
        "inspected": False,
        "flow_seed": anchor["seeds"]["flow"],
        "terminal_seed": anchor["seeds"]["terminal"],
        "reveal_condition": "freeze the candidate-driven route-registry version and its exact evidence hashes",
    }
    if seal.get("molecular_holdout") != expected_holdout:
        raise Ugi3RouteRegistryPairContractError("molecular holdout was revealed or changed")

    sampling = contract.get("holdout_sampling")
    if not isinstance(sampling, dict):
        raise Ugi3RouteRegistryPairContractError("holdout sampling contract is missing")
    expected_sampling = {
        "program_seed": anchor["seeds"]["program"],
        "flow_seed": anchor["seeds"]["flow"],
        "terminal_seed": anchor["seeds"]["terminal"],
        "programs": program_spec["rows"],
        **anchor["sampling"],
        "program_output": program_spec["path"],
        "sealed_sample_output": anchor["sealed_sample_output"],
    }
    if sampling != expected_sampling:
        raise Ugi3RouteRegistryPairContractError("holdout sampling parameters changed")
    if (
        contract.get("one_time_holdout_reporting", {}).get(
            "permit_threshold_or_registry_tuning_after_reveal"
        )
        is not False
    ):
        raise Ugi3RouteRegistryPairContractError("post-reveal registry tuning is permitted")
    if (
        contract.get("evidence_mining_policy", {}).get(
            "holdout_molecules_may_be_generated_or_inspected_before_registry_freeze"
        )
        is not False
    ):
        raise Ugi3RouteRegistryPairContractError("holdout inspection policy changed")
    sample_path = _repo_path(repo, anchor["sealed_sample_output"], label="sealed sample output")
    if anchor.get("require_sample_output_absent_before_binding") is not True:
        raise Ugi3RouteRegistryPairContractError("sample-absence gate is disabled")
    if sample_path.exists():
        raise Ugi3RouteRegistryPairContractError(
            "sealed molecular sample exists before immutable registry binding"
        )
    return {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "protocol_sha256": sha256_file(protocol_path),
        "sealed_holdout": {
            "contract_sha256": contract_hash,
            "seal_sha256": seal_hash,
            "program_draw_sha256": program_hash,
            "program_rows": program_spec["rows"],
            "seeds": dict(anchor["seeds"]),
            "molecular_holdout_generated": False,
            "molecular_holdout_inspected": False,
        },
        "protocol_valid": True,
        "registry_pair_bound": False,
        "holdout_reveal_authorized": False,
        "readiness": "valid_protocol_not_ready_registry_pair_unbound",
    }


def _validate_snapshot(
    manifest_path: Path,
    *,
    repo: Path,
    expected_role: str,
    expected_semantic_label: str,
    expected_anchor: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[tuple[str, str], dict[str, Any]]]:
    manifest = _load_json(manifest_path, label=f"{expected_role} manifest")
    if manifest.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise Ugi3RouteRegistryPairContractError(f"{expected_role} snapshot schema changed")
    if manifest.get("status") != "frozen_before_molecular_holdout_reveal":
        raise Ugi3RouteRegistryPairContractError(f"{expected_role} snapshot is not frozen")
    if manifest.get("snapshot_role") != expected_role:
        raise Ugi3RouteRegistryPairContractError(f"{expected_role} snapshot role changed")
    if manifest.get("semantic_label") != expected_semantic_label:
        raise Ugi3RouteRegistryPairContractError(f"{expected_role} semantic label changed")
    if manifest.get("sealed_holdout_anchor") != expected_anchor:
        raise Ugi3RouteRegistryPairContractError(f"{expected_role} holdout anchor changed")
    identity = manifest.get("identity_policy")
    required_identity = {
        "match_on_exact_role_and_canonical_smiles": True,
        "permit_similarity_or_family_matching": False,
        "permit_neighbor_homologue_admission": False,
        "permit_reaction_family_template_promotion": False,
    }
    if identity != required_identity:
        raise Ugi3RouteRegistryPairContractError(f"{expected_role} identity policy changed")
    record_path, _ = _validate_pinned_file(
        manifest.get("records"), repo=repo, label=f"{expected_role} record ledger"
    )
    records_spec = manifest["records"]
    if records_spec.get("schema_version") != RECORD_LEDGER_SCHEMA_VERSION:
        raise Ugi3RouteRegistryPairContractError(f"{expected_role} record schema changed")
    ledger = _load_gzip_json(record_path, label=f"{expected_role} record ledger")
    if ledger.get("schema_version") != RECORD_LEDGER_SCHEMA_VERSION:
        raise Ugi3RouteRegistryPairContractError(f"{expected_role} ledger schema changed")
    raw_records = ledger.get("records")
    if not isinstance(raw_records, list) or len(raw_records) != records_spec.get("rows"):
        raise Ugi3RouteRegistryPairContractError(f"{expected_role} record count changed")
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for index, record in enumerate(raw_records):
        if not isinstance(record, dict):
            raise Ugi3RouteRegistryPairContractError(f"{expected_role} record {index} is malformed")
        role = record.get("role")
        smiles = record.get("canonical_smiles")
        key_hash = record.get("component_key_sha256")
        if not isinstance(role, str) or not isinstance(smiles, str):
            raise Ugi3RouteRegistryPairContractError(
                f"{expected_role} record {index} lacks exact identity"
            )
        if key_hash != component_key_sha256(role, smiles):
            raise Ugi3RouteRegistryPairContractError(
                f"{expected_role} record {index} has an invalid identity hash"
            )
        if record.get("route_complete") not in {True, False}:
            raise Ugi3RouteRegistryPairContractError(
                f"{expected_role} record {index} has a non-boolean completion state"
            )
        key = (role, smiles)
        if key in indexed:
            raise Ugi3RouteRegistryPairContractError(
                f"{expected_role} ledger contains duplicate exact identities"
            )
        indexed[key] = record
    return manifest, indexed


def _validate_source_final_component_ledger(
    manifest: Mapping[str, Any],
    *,
    repo: Path,
    expected_record_count: int,
) -> tuple[dict[str, str], str]:
    specification = manifest.get("source_final_component_ledger")
    if not isinstance(specification, dict) or set(specification) != {
        "path",
        "sha256",
        "schema_version",
    }:
        raise Ugi3RouteRegistryPairContractError(
            "registry manifest lacks an exact frozen source-ledger specification"
        )
    schema_version = specification.get("schema_version")
    if not isinstance(schema_version, str) or not schema_version:
        raise Ugi3RouteRegistryPairContractError("source component-ledger schema is missing")
    source_path, source_hash = _validate_pinned_file(
        specification,
        repo=repo,
        label="source final component ledger",
    )
    source = _load_gzip_json(source_path, label="source final component ledger")
    if source.get("schema_version") != schema_version:
        raise Ugi3RouteRegistryPairContractError("source component-ledger schema changed")
    if source.get("status") != "frozen_final_component_ledger":
        raise Ugi3RouteRegistryPairContractError("source component ledger is not frozen")
    records = source.get("records")
    if not isinstance(records, list) or len(records) != expected_record_count:
        raise Ugi3RouteRegistryPairContractError("source component-ledger record count changed")
    normalized = {
        "path": specification["path"],
        "sha256": source_hash,
        "schema_version": schema_version,
    }
    return normalized, source_hash


def _validate_exact_c18_result(result: Mapping[str, Any]) -> None:
    if result.get("schema_version") != "phase1_ugi3_exact_c18_route_audit.v1":
        raise Ugi3RouteRegistryPairContractError("exact C18 result schema changed")
    summary = result.get("summary")
    adjudication = result.get("adjudication")
    if not isinstance(summary, dict) or not isinstance(adjudication, dict):
        raise Ugi3RouteRegistryPairContractError("exact C18 result is incomplete")
    required_summary = {
        "target_role": TARGET_ROLE,
        "target_smiles": TARGET_SMILES,
        "exact_steps": 3,
        "route_complete": True,
        "terminal_availability": "current_closed",
    }
    for key, expected in required_summary.items():
        if summary.get(key) != expected:
            raise Ugi3RouteRegistryPairContractError(
                f"exact C18 result is not fully closed: summary.{key}"
            )
    required_adjudication = {
        "exact_l2_chain_admitted": True,
        "current_l3_procurement_closed": True,
        "route_complete": True,
        "family_template_admitted": False,
    }
    for key, expected in required_adjudication.items():
        if adjudication.get(key) != expected:
            raise Ugi3RouteRegistryPairContractError(
                f"exact C18 result is not fully closed: adjudication.{key}"
            )


def load_validated_exact_c18_value(
    repo: Path,
    result_specification: Mapping[str, Any],
    assessment_specification: Mapping[str, Any],
) -> tuple[dict[str, Any], ComponentSynthesisValue, str, str]:
    """Load one hash-owned exact C18 result and its complete L2/L3 value."""

    result_path, result_hash = _validate_pinned_file(
        result_specification, repo=repo, label="exact C18 result"
    )
    assessment_path, assessment_hash = _validate_pinned_file(
        assessment_specification, repo=repo, label="exact C18 assessment"
    )
    result = _load_json(result_path, label="exact C18 result")
    _validate_exact_c18_result(result)
    artifact = result.get("artifacts", {}).get(Path(assessment_path).name)
    if not isinstance(artifact, dict) or artifact.get("sha256") != assessment_hash:
        raise Ugi3RouteRegistryPairContractError(
            "exact C18 result does not own its pinned assessment"
        )
    if artifact.get("schema_version") != "phase1_ugi3_exact_c18_assessment.v1":
        raise Ugi3RouteRegistryPairContractError("exact C18 assessment schema changed")
    assessment = _load_gzip_json(assessment_path, label="exact C18 assessment")
    if assessment.get("schema_version") != "phase1_ugi3_exact_c18_assessment.v1":
        raise Ugi3RouteRegistryPairContractError("exact C18 assessment ledger schema changed")
    try:
        value = ComponentSynthesisValue.from_dict(assessment.get("synthesis_value"))
    except (TypeError, ValueError) as exc:
        raise Ugi3RouteRegistryPairContractError(
            "exact C18 assessment lacks a valid synthesis value"
        ) from exc
    if (
        value.target.role != TARGET_ROLE
        or value.target.canonical_smiles != TARGET_SMILES
        or not value.route_complete
        or value.route_step_count != 3
        or value.maximum_route_depth != 3
        or value.current_terminal_leaf_count != value.leaf_count
    ):
        raise Ugi3RouteRegistryPairContractError(
            "exact C18 assessment is not an exact fully closed three-step value"
        )
    return result, value, result_hash, assessment_hash


def load_reproduced_exact_c18_value(
    repo: Path,
    config_specification: Mapping[str, Any],
    result_specification: Mapping[str, Any],
    assessment_specification: Mapping[str, Any],
) -> tuple[dict[str, Any], ComponentSynthesisValue, str, str, str]:
    """Reproduce one hash-owned exact C18 audit before admitting its value.

    The result and assessment alone are not a complete evidence contract: the
    config owns the exact-pair scope, procurement policy and all source hashes.
    Replaying the audit here prevents a repinned but broader, expired-at-the-
    assessment-time or otherwise contradictory result from entering R1.
    """

    config_path, config_hash = _validate_pinned_file(
        config_specification, repo=repo, label="exact C18 config"
    )
    config = _load_json(config_path, label="exact C18 config")
    declared_inputs = config.get("inputs")
    if not isinstance(declared_inputs, dict) or not declared_inputs:
        raise Ugi3RouteRegistryPairContractError("exact C18 config inputs are missing")
    input_paths: dict[str, Path] = {}
    for label, specification in declared_inputs.items():
        if not isinstance(specification, dict) or set(specification) != {"path", "sha256"}:
            raise Ugi3RouteRegistryPairContractError(
                f"exact C18 config input {label!r} is malformed"
            )
        input_paths[label] = _repo_path(
            repo, specification.get("path"), label=f"exact C18 config input {label}"
        )

    result, value, result_hash, assessment_hash = load_validated_exact_c18_value(
        repo,
        result_specification,
        assessment_specification,
    )
    if result.get("config_sha256") != config_hash:
        raise Ugi3RouteRegistryPairContractError(
            "exact C18 result is not owned by its pinned config"
        )
    try:
        reproduced_result, _, reproduced_assessment = build_exact_c18_route_audit(
            config_path=config_path,
            input_paths=input_paths,
        )
    except Ugi3ExactC18RouteError as exc:
        raise Ugi3RouteRegistryPairContractError(
            f"exact C18 config failed reproduction: {exc}"
        ) from exc
    if reproduced_result != result:
        raise Ugi3RouteRegistryPairContractError(
            "exact C18 result does not reproduce from its pinned config"
        )
    assessment_path = _repo_path(
        repo,
        assessment_specification.get("path"),
        label="exact C18 assessment",
    )
    if reproduced_assessment != assessment_path.read_bytes():
        raise Ugi3RouteRegistryPairContractError(
            "exact C18 assessment does not reproduce from its pinned config"
        )
    return result, value, config_hash, result_hash, assessment_hash


def validate_binding(repo: Path, protocol_path: Path, binding_path: Path) -> dict[str, Any]:
    """Validate a future immutable registry pair without reading holdout molecules."""

    protocol_result = validate_protocol(repo, protocol_path)
    protocol = _load_json(protocol_path, label="paired protocol")
    binding = _load_json(binding_path, label="paired registry binding")
    if binding.get("schema_version") != BINDING_SCHEMA_VERSION:
        raise Ugi3RouteRegistryPairContractError("unsupported paired binding schema")
    if binding.get("status") != "frozen_before_molecular_holdout_reveal":
        raise Ugi3RouteRegistryPairContractError("paired registry binding is not frozen")
    protocol_spec = binding.get("protocol")
    if not isinstance(protocol_spec, dict) or protocol_spec != {
        "path": str(protocol_path.resolve().relative_to(repo.resolve())),
        "sha256": sha256_file(protocol_path),
    }:
        raise Ugi3RouteRegistryPairContractError("binding does not own this exact protocol")

    r0_path, r0_manifest_hash = _validate_pinned_file(
        binding.get("r0_manifest"), repo=repo, label="R0 manifest"
    )
    r1_path, r1_manifest_hash = _validate_pinned_file(
        binding.get("r1_manifest"), repo=repo, label="R1 manifest"
    )
    registry_diff_path, registry_diff_hash = _validate_pinned_file(
        binding.get("registry_diff"), repo=repo, label="registry diff"
    )
    registry_diff = _load_json(registry_diff_path, label="registry diff")
    (
        _,
        _,
        c18_config_hash,
        c18_result_hash,
        c18_assessment_hash,
    ) = load_reproduced_exact_c18_value(
        repo,
        binding.get("exact_c18_config"),
        binding.get("exact_c18_result"),
        binding.get("exact_c18_assessment"),
    )

    anchor = protocol["sealed_holdout_anchor"]
    snapshot_anchor = {
        "contract_sha256": anchor["contract"]["sha256"],
        "seal_sha256": anchor["seal"]["sha256"],
        "program_draw_sha256": anchor["program_draw"]["sha256"],
        "program_rows": anchor["program_draw"]["rows"],
        "seeds": dict(anchor["seeds"]),
    }
    r0_manifest, r0_records = _validate_snapshot(
        r0_path,
        repo=repo,
        expected_role="immutable_parent_registry",
        expected_semantic_label="without_c18",
        expected_anchor=snapshot_anchor,
    )
    r1_manifest, r1_records = _validate_snapshot(
        r1_path,
        repo=repo,
        expected_role="immutable_child_registry",
        expected_semantic_label="with_c18",
        expected_anchor=snapshot_anchor,
    )
    r0_source, source_registry_hash = _validate_source_final_component_ledger(
        r0_manifest,
        repo=repo,
        expected_record_count=len(r0_records),
    )
    r1_source, r1_source_registry_hash = _validate_source_final_component_ledger(
        r1_manifest,
        repo=repo,
        expected_record_count=len(r1_records),
    )
    if r1_source != r0_source or r1_source_registry_hash != source_registry_hash:
        raise Ugi3RouteRegistryPairContractError(
            "R0 and R1 do not own the same frozen source component ledger"
        )
    if r0_manifest.get("parent_snapshot") is not None:
        raise Ugi3RouteRegistryPairContractError("R0 unexpectedly has a parent registry")
    if r1_manifest.get("parent_snapshot") != {
        "path": binding["r0_manifest"]["path"],
        "sha256": r0_manifest_hash,
    }:
        raise Ugi3RouteRegistryPairContractError("R1 parent is not the exact pinned R0")

    r0_keys = set(r0_records)
    r1_keys = set(r1_records)
    if r0_keys != r1_keys:
        raise Ugi3RouteRegistryPairContractError("registry key set changed between R0 and R1")
    target_key = (TARGET_ROLE, TARGET_SMILES)
    if target_key not in r0_records:
        raise Ugi3RouteRegistryPairContractError("authorized C18 target is absent from R0")
    changed = [
        key
        for key in sorted(r0_keys)
        if _stable_json(r0_records[key]) != _stable_json(r1_records[key])
    ]
    if changed != [target_key]:
        raise Ugi3RouteRegistryPairContractError(
            "R1 must change exactly one authorized role-and-identity record"
        )
    old = r0_records[target_key]
    new = r1_records[target_key]
    if old.get("route_complete") is not False or new.get("route_complete") is not True:
        raise Ugi3RouteRegistryPairContractError(
            "authorized C18 delta is not noncomplete-to-complete"
        )
    if new.get("exact_l2_steps") != 3:
        raise Ugi3RouteRegistryPairContractError("R1 C18 record lacks exact three-step L2")
    if new.get("terminal_availability") != "current_closed":
        raise Ugi3RouteRegistryPairContractError("R1 C18 record lacks current L3 closure")
    if new.get("family_template_admitted") is not False:
        raise Ugi3RouteRegistryPairContractError("R1 C18 record promotes family scope")

    required_diff = {
        "schema_version": DIFF_SCHEMA_VERSION,
        "status": "exactly_one_authorized_record_change",
        "visibility": "private_hash_pinned",
        "r0_semantic_label": "without_c18",
        "r1_semantic_label": "with_c18",
        "record_count_r0": len(r0_records),
        "record_count_r1": len(r1_records),
        "added_count": 0,
        "deleted_count": 0,
        "changed_count": 1,
        "non_target_mutation_count": 0,
        "regression_count": 0,
        "changed_component": {
            "component_key_sha256": TARGET_KEY_SHA256,
            "role": TARGET_ROLE,
            "r0_route_complete": False,
            "r1_route_complete": True,
            "exact_l2_steps": 3,
            "terminal_availability": "current_closed",
            "family_template_admitted": False,
        },
        "all_non_c18_records_canonical_json_equal": True,
        "source_final_component_ledger_sha256": source_registry_hash,
        "exact_c18_config_sha256": c18_config_hash,
        "exact_c18_result_sha256": c18_result_hash,
        "exact_c18_assessment_sha256": binding["exact_c18_assessment"]["sha256"],
        "r0_records_sha256": r0_manifest["records"]["sha256"],
        "r1_records_sha256": r1_manifest["records"]["sha256"],
    }
    if registry_diff != required_diff:
        raise Ugi3RouteRegistryPairContractError("registry diff does not match the paired delta")

    return {
        **protocol_result,
        "binding_sha256": sha256_file(binding_path),
        "r0_manifest_sha256": r0_manifest_hash,
        "r1_manifest_sha256": r1_manifest_hash,
        "registry_diff_sha256": registry_diff_hash,
        "exact_c18_config_sha256": c18_config_hash,
        "exact_c18_result_sha256": c18_result_hash,
        "exact_c18_assessment_sha256": c18_assessment_hash,
        "registry_record_count": len(r0_records),
        "authorized_delta_count": 1,
        "registry_pair_bound": True,
        "holdout_reveal_authorized": True,
        "readiness": "immutable_registry_pair_valid_holdout_reveal_authorized",
    }
