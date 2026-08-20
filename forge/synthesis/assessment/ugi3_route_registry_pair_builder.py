"""Build the immutable R0/without-C18 and R1/with-C18 registry pair.

The builder is deliberately inert until a config explicitly pins the final
component ledger and exact C18 result/assessment hashes.  It never opens the
sealed program draw or a molecular holdout artifact.
"""

from __future__ import annotations

import gzip
import io
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from forge.core.hashing import sha256_bytes, sha256_file
from forge.core.io import read_json_object
from forge.synthesis.assessment.ugi3_route_registry_pair_contract import (
    BINDING_SCHEMA_VERSION,
    DIFF_SCHEMA_VERSION,
    RECORD_LEDGER_SCHEMA_VERSION,
    SNAPSHOT_SCHEMA_VERSION,
    TARGET_KEY_SHA256,
    TARGET_ROLE,
    TARGET_SMILES,
    Ugi3RouteRegistryPairContractError,
    component_key_sha256,
    load_reproduced_exact_c18_value,
    validate_protocol,
)
from forge.synthesis.value.contracts import ComponentSynthesisValue

CONFIG_SCHEMA_VERSION = "phase1_ugi3_route_registry_pair_builder_config.v1"

BOUND_STATUS = "frozen_inputs_bound_before_molecular_holdout_reveal"
UNBOUND_STATUS = "awaiting_explicit_final_input_hashes"

OUTPUT_LABELS = (
    "r0_records",
    "r0_manifest",
    "r1_records",
    "r1_manifest",
    "registry_diff",
    "binding",
)

REQUIRED_BINDING_POLICY = {
    "require_all_input_hashes": True,
    "require_final_component_ledger_frozen": True,
    "require_c18_noncomplete_in_parent": True,
    "require_exact_c18_complete_l2_l3": True,
    "permit_added_component_keys": False,
    "permit_deleted_component_keys": False,
    "permit_non_c18_record_changes": False,
    "permit_holdout_generation_or_inspection": False,
}


class Ugi3RouteRegistryPairBuilderError(ValueError):
    """Raised when immutable registry-pair materialization is unsafe."""


def _stable_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n").encode()


def _gzip_json_bytes(value: Any) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as compressed:
        compressed.write(_stable_json_bytes(value))
    return output.getvalue()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    return read_json_object(path, error=Ugi3RouteRegistryPairBuilderError, label=label)


def _load_gzip_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        with gzip.open(path, "rt") as handle:
            value = json.load(handle)
    except (FileNotFoundError, gzip.BadGzipFile, json.JSONDecodeError, OSError) as exc:
        raise Ugi3RouteRegistryPairBuilderError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise Ugi3RouteRegistryPairBuilderError(f"{label} must be a JSON object")
    return value


def _path(repo: Path, raw: Any, *, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise Ugi3RouteRegistryPairBuilderError(f"{label} path is not explicitly bound")
    candidate = Path(raw)
    return candidate if candidate.is_absolute() else repo / candidate


def _portable(path: Path, *, repo: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo.resolve()))
    except ValueError:
        return str(path.resolve())


def _pinned(
    repo: Path, specification: Any, *, label: str, permit_schema: bool = False
) -> tuple[Path, str]:
    allowed = {"path", "sha256", "schema_version"} if permit_schema else {"path", "sha256"}
    if not isinstance(specification, dict) or set(specification) != allowed:
        raise Ugi3RouteRegistryPairBuilderError(f"{label} specification is malformed")
    path = _path(repo, specification.get("path"), label=label)
    expected = specification.get("sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise Ugi3RouteRegistryPairBuilderError(f"{label} hash is not explicitly bound")
    try:
        observed = sha256_file(path)
    except OSError as exc:
        raise Ugi3RouteRegistryPairBuilderError(f"{label} is missing: {path}") from exc
    if observed != expected:
        raise Ugi3RouteRegistryPairBuilderError(f"{label} hash changed")
    if permit_schema and not isinstance(specification.get("schema_version"), str):
        raise Ugi3RouteRegistryPairBuilderError(f"{label} schema is not explicitly bound")
    return path, observed


def _component_value(raw: Any, *, label: str) -> ComponentSynthesisValue:
    try:
        return ComponentSynthesisValue.from_dict(raw)
    except (TypeError, ValueError) as exc:
        raise Ugi3RouteRegistryPairBuilderError(f"{label} has an invalid synthesis value") from exc


def _normalized_record(raw: Any, *, index: int) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise Ugi3RouteRegistryPairBuilderError(f"component record {index} is malformed")
    role = raw.get("role")
    smiles = raw.get("canonical_smiles")
    source_class = raw.get("source_class")
    family_template = raw.get("family_template_admitted")
    if not isinstance(role, str) or not isinstance(smiles, str):
        raise Ugi3RouteRegistryPairBuilderError(f"component record {index} lacks exact identity")
    if not isinstance(source_class, str) or not source_class:
        raise Ugi3RouteRegistryPairBuilderError(f"component record {index} lacks source class")
    if family_template not in {True, False}:
        raise Ugi3RouteRegistryPairBuilderError(
            f"component record {index} lacks an explicit family-scope decision"
        )
    value = _component_value(raw.get("value"), label=f"component record {index}")
    if value.target.role != role or value.target.canonical_smiles != smiles:
        raise Ugi3RouteRegistryPairBuilderError(
            f"component record {index} identity and synthesis value differ"
        )
    key_hash = component_key_sha256(role, smiles)
    declared_hash = raw.get("component_key_sha256")
    if declared_hash is not None and declared_hash != key_hash:
        raise Ugi3RouteRegistryPairBuilderError(f"component record {index} identity hash changed")
    terminal = (
        "current_closed"
        if value.route_complete and value.current_terminal_leaf_count == value.leaf_count
        else "not_current_closed"
    )
    return {
        **raw,
        "role": role,
        "canonical_smiles": smiles,
        "component_key_sha256": key_hash,
        "assessment_outcome": value.assessment_outcome.value,
        "route_complete": value.route_complete,
        "route_step_count": value.route_step_count,
        "exact_l2_steps": raw.get("exact_l2_steps", 0),
        "terminal_availability": terminal,
        "family_template_admitted": family_template,
        "source_class": source_class,
        "value": value.to_dict(),
    }


def _load_parent_records(ledger_path: Path, *, expected_schema: str) -> list[dict[str, Any]]:
    ledger = _load_gzip_json(ledger_path, label="final component ledger")
    if ledger.get("schema_version") != expected_schema:
        raise Ugi3RouteRegistryPairBuilderError("final component ledger schema changed")
    if ledger.get("status") != "frozen_final_component_ledger":
        raise Ugi3RouteRegistryPairBuilderError("final component ledger is not frozen")
    raw_records = ledger.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise Ugi3RouteRegistryPairBuilderError("final component ledger has no records")
    records = [_normalized_record(record, index=index) for index, record in enumerate(raw_records)]
    records.sort(key=lambda row: (row["role"], row["canonical_smiles"]))
    keys = [(row["role"], row["canonical_smiles"]) for row in records]
    if len(keys) != len(set(keys)):
        raise Ugi3RouteRegistryPairBuilderError(
            "final component ledger contains duplicate exact identities"
        )
    return records


def _output_paths(repo: Path, outputs: Any) -> dict[str, Path]:
    if not isinstance(outputs, dict) or set(outputs) != set(OUTPUT_LABELS):
        raise Ugi3RouteRegistryPairBuilderError("builder output contract changed")
    paths = {label: _path(repo, outputs[label], label=f"output {label}") for label in OUTPUT_LABELS}
    if len({path.resolve() for path in paths.values()}) != len(paths):
        raise Ugi3RouteRegistryPairBuilderError("builder output paths are not unique")
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise Ugi3RouteRegistryPairBuilderError(
            "immutable registry-pair outputs already exist: " + ", ".join(sorted(existing))
        )
    return paths


def _snapshot_anchor(protocol: Mapping[str, Any]) -> dict[str, Any]:
    anchor = protocol["sealed_holdout_anchor"]
    return {
        "contract_sha256": anchor["contract"]["sha256"],
        "seal_sha256": anchor["seal"]["sha256"],
        "program_draw_sha256": anchor["program_draw"]["sha256"],
        "program_rows": anchor["program_draw"]["rows"],
        "seeds": dict(anchor["seeds"]),
    }


def _manifest(
    *,
    snapshot_id: str,
    role: str,
    semantic_label: str,
    anchor: Mapping[str, Any],
    source_specification: Mapping[str, Any],
    parent: Mapping[str, str] | None,
    records_path: Path,
    records_payload: bytes,
    record_count: int,
    repo: Path,
) -> dict[str, Any]:
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "status": "frozen_before_molecular_holdout_reveal",
        "snapshot_id": snapshot_id,
        "snapshot_role": role,
        "semantic_label": semantic_label,
        "sealed_holdout_anchor": dict(anchor),
        "identity_policy": {
            "match_on_exact_role_and_canonical_smiles": True,
            "permit_similarity_or_family_matching": False,
            "permit_neighbor_homologue_admission": False,
            "permit_reaction_family_template_promotion": False,
        },
        "source_final_component_ledger": dict(source_specification),
        "parent_snapshot": dict(parent) if parent is not None else None,
        "records": {
            "path": _portable(records_path, repo=repo),
            "sha256": sha256_bytes(records_payload),
            "schema_version": RECORD_LEDGER_SCHEMA_VERSION,
            "rows": record_count,
        },
    }


def build_registry_pair(repo: Path, config_path: Path) -> dict[str, bytes]:
    """Return deterministic immutable-pair artifacts without writing or revealing holdout data."""

    config = _load_json(config_path, label="registry-pair builder config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise Ugi3RouteRegistryPairBuilderError("unsupported builder config schema")
    if config.get("status") == UNBOUND_STATUS:
        raise Ugi3RouteRegistryPairBuilderError(
            "builder is unbound; explicit final input paths, hashes and schemas are required"
        )
    if config.get("status") != BOUND_STATUS:
        raise Ugi3RouteRegistryPairBuilderError("builder config status is not executable")
    if config.get("binding_policy") != REQUIRED_BINDING_POLICY:
        raise Ugi3RouteRegistryPairBuilderError("builder binding policy changed")
    if config.get("decision") != {
        "inputs_bound": True,
        "registry_pair_materialized": False,
        "holdout_reveal_authorized": False,
    }:
        raise Ugi3RouteRegistryPairBuilderError("builder decision state is unsafe")

    protocol_path, protocol_hash = _pinned(repo, config.get("protocol"), label="paired protocol")
    validate_protocol(repo, protocol_path)
    protocol = _load_json(protocol_path, label="paired protocol")

    inputs = config.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {
        "final_component_ledger",
        "exact_c18_config",
        "exact_c18_result",
        "exact_c18_assessment",
    }:
        raise Ugi3RouteRegistryPairBuilderError("builder input contract changed")
    source_path, source_hash = _pinned(
        repo,
        inputs["final_component_ledger"],
        label="final component ledger",
        permit_schema=True,
    )
    source_specification = {
        "path": _portable(source_path, repo=repo),
        "sha256": source_hash,
        "schema_version": inputs["final_component_ledger"]["schema_version"],
    }
    records = _load_parent_records(
        source_path, expected_schema=source_specification["schema_version"]
    )
    target_indexes = [
        index
        for index, record in enumerate(records)
        if (record["role"], record["canonical_smiles"]) == (TARGET_ROLE, TARGET_SMILES)
    ]
    if len(target_indexes) != 1:
        raise Ugi3RouteRegistryPairBuilderError(
            "final component ledger must contain the exact C18 identity once"
        )
    target_index = target_indexes[0]
    if records[target_index]["route_complete"] is not False:
        raise Ugi3RouteRegistryPairBuilderError(
            "R0/without-C18 parent already marks the exact C18 route complete"
        )

    try:
        (
            _,
            c18_value,
            c18_config_hash,
            c18_result_hash,
            c18_assessment_hash,
        ) = load_reproduced_exact_c18_value(
            repo,
            inputs["exact_c18_config"],
            inputs["exact_c18_result"],
            inputs["exact_c18_assessment"],
        )
    except Ugi3RouteRegistryPairContractError as exc:
        raise Ugi3RouteRegistryPairBuilderError(str(exc)) from exc

    r0_records = [dict(record) for record in records]
    r1_records = [dict(record) for record in records]
    r1_target = {
        **r0_records[target_index],
        "component_key_sha256": TARGET_KEY_SHA256,
        "assessment_outcome": c18_value.assessment_outcome.value,
        "route_complete": True,
        "route_step_count": c18_value.route_step_count,
        "exact_l2_steps": 3,
        "terminal_availability": "current_closed",
        "family_template_admitted": False,
        "source_class": "exact_c18_three_step_route_current_terminal",
        "value": c18_value.to_dict(),
    }
    r1_records[target_index] = r1_target

    changed = [
        index
        for index, (old, new) in enumerate(zip(r0_records, r1_records, strict=True))
        if _stable_json_bytes(old) != _stable_json_bytes(new)
    ]
    if changed != [target_index]:
        raise Ugi3RouteRegistryPairBuilderError("builder changed more than the exact C18 record")
    for index, (old, new) in enumerate(zip(r0_records, r1_records, strict=True)):
        if index != target_index and _stable_json_bytes(old) != _stable_json_bytes(new):
            raise Ugi3RouteRegistryPairBuilderError("non-C18 registry evidence changed")

    paths = _output_paths(repo, config.get("outputs"))
    r0_payload = _gzip_json_bytes(
        {"schema_version": RECORD_LEDGER_SCHEMA_VERSION, "records": r0_records}
    )
    r1_payload = _gzip_json_bytes(
        {"schema_version": RECORD_LEDGER_SCHEMA_VERSION, "records": r1_records}
    )
    anchor = _snapshot_anchor(protocol)
    r0_manifest = _manifest(
        snapshot_id="r0_without_c18",
        role="immutable_parent_registry",
        semantic_label="without_c18",
        anchor=anchor,
        source_specification=source_specification,
        parent=None,
        records_path=paths["r0_records"],
        records_payload=r0_payload,
        record_count=len(r0_records),
        repo=repo,
    )
    r0_manifest_payload = _stable_json_bytes(r0_manifest)
    r0_manifest_spec = {
        "path": _portable(paths["r0_manifest"], repo=repo),
        "sha256": sha256_bytes(r0_manifest_payload),
    }
    r1_manifest = _manifest(
        snapshot_id="r1_with_c18",
        role="immutable_child_registry",
        semantic_label="with_c18",
        anchor=anchor,
        source_specification=source_specification,
        parent=r0_manifest_spec,
        records_path=paths["r1_records"],
        records_payload=r1_payload,
        record_count=len(r1_records),
        repo=repo,
    )
    r1_manifest_payload = _stable_json_bytes(r1_manifest)
    r1_manifest_spec = {
        "path": _portable(paths["r1_manifest"], repo=repo),
        "sha256": sha256_bytes(r1_manifest_payload),
    }
    diff = {
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
        "source_final_component_ledger_sha256": source_hash,
        "exact_c18_config_sha256": c18_config_hash,
        "exact_c18_result_sha256": c18_result_hash,
        "exact_c18_assessment_sha256": c18_assessment_hash,
        "r0_records_sha256": sha256_bytes(r0_payload),
        "r1_records_sha256": sha256_bytes(r1_payload),
    }
    diff_payload = _stable_json_bytes(diff)
    binding = {
        "schema_version": BINDING_SCHEMA_VERSION,
        "status": "frozen_before_molecular_holdout_reveal",
        "protocol": {
            "path": _portable(protocol_path, repo=repo),
            "sha256": protocol_hash,
        },
        "r0_manifest": r0_manifest_spec,
        "r1_manifest": r1_manifest_spec,
        "registry_diff": {
            "path": _portable(paths["registry_diff"], repo=repo),
            "sha256": sha256_bytes(diff_payload),
        },
        "exact_c18_config": {
            "path": _portable(
                _path(repo, inputs["exact_c18_config"]["path"], label="exact C18 config"),
                repo=repo,
            ),
            "sha256": c18_config_hash,
        },
        "exact_c18_result": {
            "path": _portable(
                _path(repo, inputs["exact_c18_result"]["path"], label="exact C18 result"),
                repo=repo,
            ),
            "sha256": c18_result_hash,
        },
        "exact_c18_assessment": {
            "path": _portable(
                _path(
                    repo,
                    inputs["exact_c18_assessment"]["path"],
                    label="exact C18 assessment",
                ),
                repo=repo,
            ),
            "sha256": c18_assessment_hash,
        },
    }
    return {
        "r0_records": r0_payload,
        "r0_manifest": r0_manifest_payload,
        "r1_records": r1_payload,
        "r1_manifest": r1_manifest_payload,
        "registry_diff": diff_payload,
        "binding": _stable_json_bytes(binding),
    }


def configured_output_paths(repo: Path, config_path: Path) -> dict[str, Path]:
    """Return the declared output paths after validating their closed-world shape."""

    config = _load_json(config_path, label="registry-pair builder config")
    return _output_paths(repo, config.get("outputs"))
