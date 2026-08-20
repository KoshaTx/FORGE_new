"""Freeze the prereveal R0/without-C18 parent component ledger.

The parent ledger preserves final fresh-pool v5 evidence for every component
except the exact C18 aldehyde target.  That one record is restored verbatim
from immutable pre-C18 v3 before explicit, nonpromoting registry metadata is
added.  This module neither reads a holdout artifact nor materializes the
paired R0/R1 registries.
"""

from __future__ import annotations

import gzip
import io
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from forge.corpus.r1_prime_audit import sha256_bytes, sha256_file
from forge.synthesis.assessment.ugi3_route_registry_pair_contract import (
    TARGET_ROLE,
    TARGET_SMILES,
    component_key_sha256,
)
from forge.synthesis.value.contracts import ComponentSynthesisValue

CONFIG_SCHEMA_VERSION = "phase1_ugi3_frozen_parent_component_ledger_config.v1"
LEDGER_SCHEMA_VERSION = "phase1_ugi3_frozen_parent_component_ledger.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi3_frozen_parent_component_ledger_audit.v1"
FROZEN_STATUS = "frozen_final_component_ledger"

DERIVED_RECORD_FIELDS = frozenset(
    {
        "component_key_sha256",
        "assessment_outcome",
        "route_complete",
        "route_step_count",
        "exact_l2_steps",
        "terminal_availability",
        "family_template_admitted",
        "homologue_template_admitted",
    }
)

REQUIRED_POLICY = {
    "purpose": "Freeze prereveal R0/without-C18 from final v5 plus the exact v3 C18 record.",
    "authorized_reversion": {
        "role": TARGET_ROLE,
        "canonical_smiles": TARGET_SMILES,
        "source_version": "fresh_pool_route_coverage_v3",
    },
    "permit_added_component_keys": False,
    "permit_deleted_component_keys": False,
    "permit_non_c18_record_changes": False,
    "permit_family_template_promotion": False,
    "permit_homologue_template_promotion": False,
    "permit_registry_pair_materialization": False,
    "permit_holdout_access_or_reveal": False,
}


class Ugi3FrozenParentComponentLedgerError(ValueError):
    """Raised when the prereveal parent-ledger freeze is not exact."""


def _stable_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n").encode()


def _gzip_json_bytes(value: Any) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as compressed:
        compressed.write(_stable_json_bytes(value))
    return output.getvalue()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError) as error:
        raise Ugi3FrozenParentComponentLedgerError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise Ugi3FrozenParentComponentLedgerError(f"{label} must be a JSON object")
    return value


def _load_gzip_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        with gzip.open(path, "rt") as handle:
            value = json.load(handle)
    except (FileNotFoundError, gzip.BadGzipFile, json.JSONDecodeError, OSError) as error:
        raise Ugi3FrozenParentComponentLedgerError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise Ugi3FrozenParentComponentLedgerError(f"{label} must be a JSON object")
    return value


def _portable(path: Path, *, repo: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo.resolve()))
    except ValueError:
        return str(path.resolve())


def _validate_inputs(config: Mapping[str, Any], repo: Path) -> dict[str, Path]:
    expected_labels = {
        "v5_component_ledger",
        "v5_result",
        "v5_config",
        "v3_component_ledger",
        "v3_result",
        "v3_config",
        "value_source",
        "audit_source",
        "audit_runner",
        "audit_tests",
    }
    declared = config.get("inputs")
    if not isinstance(declared, dict) or set(declared) != expected_labels:
        raise Ugi3FrozenParentComponentLedgerError("parent-ledger input contract changed")
    paths: dict[str, Path] = {}
    for label, record in declared.items():
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise Ugi3FrozenParentComponentLedgerError(f"input {label} is malformed")
        raw_path = record.get("path")
        expected_hash = record.get("sha256")
        if not isinstance(raw_path, str) or not raw_path:
            raise Ugi3FrozenParentComponentLedgerError(f"input {label} path is missing")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise Ugi3FrozenParentComponentLedgerError(f"input {label} hash is missing")
        path = Path(raw_path)
        if not path.is_absolute():
            path = repo / path
        try:
            observed_hash = sha256_file(path)
        except OSError as error:
            raise Ugi3FrozenParentComponentLedgerError(f"input {label} is missing") from error
        if observed_hash != expected_hash:
            raise Ugi3FrozenParentComponentLedgerError(f"input hash changed: {label}")
        paths[label] = path
    return paths


def _validate_v5_lineage(paths: Mapping[str, Path]) -> dict[str, Any]:
    result = _load_json(paths["v5_result"], label="fresh-pool v5 result")
    config = _load_json(paths["v5_config"], label="fresh-pool v5 config")
    artifact = result.get("artifacts", {}).get("component_synthesis_values.json.gz", {})
    policy = result.get("policy")
    adjudication = result.get("adjudication")
    if (
        result.get("schema_version") != "phase1_ugi3_fresh_pool_route_coverage.v5"
        or result.get("status") != "complete_nonselecting_exact_c16_route_refresh"
        or config.get("schema_version") != "phase1_ugi3_fresh_pool_route_coverage_config.v5"
        or result.get("config_sha256") != sha256_file(paths["v5_config"])
        or result.get("summary") != config.get("expected_summary")
        or artifact.get("schema_version") != "phase1_ugi3_fresh_pool_component_values.v5"
        or artifact.get("sha256") != sha256_file(paths["v5_component_ledger"])
        or not isinstance(policy, dict)
        or policy.get("reaction_family_scope_promoted") is not False
        or policy.get("homologue_scope_promoted") is not False
        or policy.get("holdout_reveal_authorized") is not False
        or policy.get("synthesis_guidance_authorized") is not False
        or not isinstance(adjudication, dict)
        or adjudication.get("reaction_family_promoted") is not False
        or adjudication.get("homologue_scope_promoted") is not False
        or adjudication.get("holdout_reveal_authorized") is not False
        or adjudication.get("synthesis_guidance_authorized") is not False
    ):
        raise Ugi3FrozenParentComponentLedgerError("final v5 lineage or scope changed")
    return result


def _validate_v3_lineage(paths: Mapping[str, Path]) -> dict[str, Any]:
    result = _load_json(paths["v3_result"], label="fresh-pool v3 result")
    config = _load_json(paths["v3_config"], label="fresh-pool v3 config")
    artifact = result.get("artifacts", {}).get("component_synthesis_values.json.gz", {})
    expected_policy = config.get("assessment_policy")
    if (
        result.get("schema_version") != "phase1_ugi3_fresh_pool_route_coverage.v3"
        or result.get("status") != "complete_nonselecting_exact_terminal_refresh"
        or result.get("config_sha256") is not None
        or config.get("schema_version") != "phase1_ugi3_fresh_pool_route_coverage_config.v3"
        or result.get("summary") != config.get("expected_summary")
        or result.get("policy") != expected_policy
        or artifact.get("schema_version") != "phase1_ugi3_fresh_pool_component_values.v3"
        or artifact.get("sha256") != sha256_file(paths["v3_component_ledger"])
        or not isinstance(expected_policy, dict)
        or expected_policy.get("family_projection_can_close") is not False
        or expected_policy.get("similarity_can_transfer_terminal_status") is not False
        or expected_policy.get("production_synthesis_guidance") is not False
        or expected_policy.get("prospective_candidate_selection") is not False
    ):
        raise Ugi3FrozenParentComponentLedgerError("immutable v3 lineage or scope changed")
    return result


def _component_value(record: Mapping[str, Any], *, label: str) -> ComponentSynthesisValue:
    try:
        value = ComponentSynthesisValue.from_dict(record.get("value"))
    except (TypeError, ValueError) as error:
        raise Ugi3FrozenParentComponentLedgerError(f"{label} value is invalid") from error
    if value.target.role != record.get("role") or value.target.canonical_smiles != record.get(
        "canonical_smiles"
    ):
        raise Ugi3FrozenParentComponentLedgerError(f"{label} identity and value differ")
    return value


def _records_by_key(
    ledger: Mapping[str, Any], *, expected_schema: str, label: str
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    if ledger.get("schema_version") != expected_schema:
        raise Ugi3FrozenParentComponentLedgerError(f"{label} schema changed")
    raw_records = ledger.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise Ugi3FrozenParentComponentLedgerError(f"{label} records are missing")
    records: list[dict[str, Any]] = []
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for index, raw in enumerate(raw_records):
        if not isinstance(raw, dict):
            raise Ugi3FrozenParentComponentLedgerError(f"{label} record {index} is malformed")
        if DERIVED_RECORD_FIELDS.intersection(raw):
            raise Ugi3FrozenParentComponentLedgerError(
                f"{label} record {index} already contains parent-ledger fields"
            )
        role = raw.get("role")
        smiles = raw.get("canonical_smiles")
        if not isinstance(role, str) or not isinstance(smiles, str):
            raise Ugi3FrozenParentComponentLedgerError(f"{label} record {index} lacks identity")
        _component_value(raw, label=f"{label} record {index}")
        key = (role, smiles)
        if key in by_key:
            raise Ugi3FrozenParentComponentLedgerError(f"{label} identities are not unique")
        record = dict(raw)
        records.append(record)
        by_key[key] = record
    return records, by_key


def _enriched_record(raw: Mapping[str, Any]) -> dict[str, Any]:
    value = _component_value(raw, label="selected parent record")
    route_complete = value.route_complete
    current_closed = (
        route_complete
        and value.leaf_count > 0
        and value.current_terminal_leaf_count == value.leaf_count
    )
    exact_l2_steps = (
        value.route_step_count
        if route_complete and value.evidence_support.value == "exact_identity"
        else 0
    )
    return {
        **raw,
        "component_key_sha256": component_key_sha256(
            value.target.role, value.target.canonical_smiles
        ),
        "assessment_outcome": value.assessment_outcome.value,
        "route_complete": route_complete,
        "route_step_count": value.route_step_count,
        "exact_l2_steps": exact_l2_steps,
        "terminal_availability": "current_closed" if current_closed else "not_current_closed",
        "family_template_admitted": False,
        "homologue_template_admitted": False,
    }


def _source_lineage(
    *,
    repo: Path,
    paths: Mapping[str, Path],
    v5_result: Mapping[str, Any],
    v3_result: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "final_v5": {
            "component_ledger": {
                "path": _portable(paths["v5_component_ledger"], repo=repo),
                "sha256": sha256_file(paths["v5_component_ledger"]),
                "schema_version": "phase1_ugi3_fresh_pool_component_values.v5",
            },
            "result": {
                "path": _portable(paths["v5_result"], repo=repo),
                "sha256": sha256_file(paths["v5_result"]),
                "schema_version": v5_result["schema_version"],
            },
            "config": {
                "path": _portable(paths["v5_config"], repo=repo),
                "sha256": sha256_file(paths["v5_config"]),
                "schema_version": "phase1_ugi3_fresh_pool_route_coverage_config.v5",
            },
        },
        "pre_c18_v3": {
            "component_ledger": {
                "path": _portable(paths["v3_component_ledger"], repo=repo),
                "sha256": sha256_file(paths["v3_component_ledger"]),
                "schema_version": "phase1_ugi3_fresh_pool_component_values.v3",
            },
            "result": {
                "path": _portable(paths["v3_result"], repo=repo),
                "sha256": sha256_file(paths["v3_result"]),
                "schema_version": v3_result["schema_version"],
            },
            "config": {
                "path": _portable(paths["v3_config"], repo=repo),
                "sha256": sha256_file(paths["v3_config"]),
                "schema_version": "phase1_ugi3_fresh_pool_route_coverage_config.v3",
            },
        },
    }


def build_frozen_parent_component_ledger(
    repo: Path, config_path: Path
) -> tuple[dict[str, Any], bytes]:
    """Build the exact prereveal parent ledger and audit result in memory."""

    config = _load_json(config_path, label="frozen parent-ledger config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise Ugi3FrozenParentComponentLedgerError("unsupported parent-ledger config schema")
    if config.get("status") != "ready_to_freeze_prereveal_parent":
        raise Ugi3FrozenParentComponentLedgerError("parent-ledger config is not executable")
    if config.get("policy") != REQUIRED_POLICY:
        raise Ugi3FrozenParentComponentLedgerError("parent-ledger policy changed")
    if config.get("decision") != {
        "registry_pair_materialized": False,
        "holdout_accessed": False,
        "holdout_reveal_authorized": False,
    }:
        raise Ugi3FrozenParentComponentLedgerError("parent-ledger decision state is unsafe")
    paths = _validate_inputs(config, repo)
    v5_result = _validate_v5_lineage(paths)
    v3_result = _validate_v3_lineage(paths)
    v5_ledger = _load_gzip_json(paths["v5_component_ledger"], label="final v5 components")
    v3_ledger = _load_gzip_json(paths["v3_component_ledger"], label="immutable v3 components")
    v5_records, v5_by_key = _records_by_key(
        v5_ledger,
        expected_schema="phase1_ugi3_fresh_pool_component_values.v5",
        label="final v5 component ledger",
    )
    _, v3_by_key = _records_by_key(
        v3_ledger,
        expected_schema="phase1_ugi3_fresh_pool_component_values.v3",
        label="immutable v3 component ledger",
    )
    if set(v5_by_key) != set(v3_by_key):
        raise Ugi3FrozenParentComponentLedgerError("v3 and v5 component identity sets differ")

    target_key = (TARGET_ROLE, TARGET_SMILES)
    v5_target = v5_by_key.get(target_key)
    v3_target = v3_by_key.get(target_key)
    if v5_target is None or v3_target is None:
        raise Ugi3FrozenParentComponentLedgerError(
            "exact C18 identity is not unique in both sources"
        )
    v5_target_value = _component_value(v5_target, label="final v5 exact C18 record")
    v3_target_value = _component_value(v3_target, label="immutable v3 exact C18 record")
    if (
        not v5_target_value.route_complete
        or v5_target_value.route_step_count != 3
        or v5_target_value.evidence_support.value != "exact_identity"
        or v5_target_value.current_terminal_leaf_count != v5_target_value.leaf_count
        or v5_target.get("source_class") != "exact_c18_three_step_route_current_terminal"
        or v3_target_value.route_complete
        or v3_target_value.assessment_outcome.value != "missing_knowledge"
        or v3_target_value.route_step_count != 0
        or v3_target.get("source_class") != "v3_exact_overlay_or_replay"
    ):
        raise Ugi3FrozenParentComponentLedgerError(
            "C18 source records violate the frozen reversion"
        )

    c16_changed_keys = v5_result.get("summary", {}).get("changed_component_keys")
    if not isinstance(c16_changed_keys, list) or len(c16_changed_keys) != 1:
        raise Ugi3FrozenParentComponentLedgerError("v5 exact C16 identity is not frozen")
    c16_parts = c16_changed_keys[0].split("\t")
    if len(c16_parts) != 2:
        raise Ugi3FrozenParentComponentLedgerError("v5 exact C16 identity is malformed")
    c16_key = (c16_parts[0], c16_parts[1])
    c16_record = v5_by_key.get(c16_key)
    if c16_record is None or c16_key == target_key:
        raise Ugi3FrozenParentComponentLedgerError("qualified C16 record is missing from v5")
    c16_value = _component_value(c16_record, label="qualified v5 C16 record")
    if (
        not c16_value.route_complete
        or c16_value.route_step_count != 4
        or c16_value.current_terminal_leaf_count != 2
        or c16_value.leaf_count != 2
        or c16_value.evidence_support.value != "exact_identity"
        or c16_record.get("source_class") != "exact_c16_four_step_route_two_current_terminals"
    ):
        raise Ugi3FrozenParentComponentLedgerError("qualified C16 evidence was not retained")

    selected_raw_records: list[dict[str, Any]] = []
    authorized_reversions = 0
    for record in v5_records:
        key = (record["role"], record["canonical_smiles"])
        if key == target_key:
            selected_raw_records.append(dict(v3_target))
            authorized_reversions += 1
        else:
            selected_raw_records.append(dict(record))
    if authorized_reversions != 1:
        raise Ugi3FrozenParentComponentLedgerError("exact C18 reversion did not occur once")

    source_differences = [
        index
        for index, (before, after) in enumerate(zip(v5_records, selected_raw_records, strict=True))
        if _stable_json_bytes(before) != _stable_json_bytes(after)
    ]
    if len(source_differences) != 1:
        raise Ugi3FrozenParentComponentLedgerError("non-C18 source record changed")
    changed_index = source_differences[0]
    if (
        v5_records[changed_index]["role"],
        v5_records[changed_index]["canonical_smiles"],
    ) != target_key:
        raise Ugi3FrozenParentComponentLedgerError("authorized reversion escaped exact C18")

    output_records = [_enriched_record(record) for record in selected_raw_records]
    outcomes = Counter(record["assessment_outcome"] for record in output_records)
    source_classes = Counter(record["source_class"] for record in output_records)
    summary = {
        "source_record_count": len(v5_records),
        "output_record_count": len(output_records),
        "added_component_keys": 0,
        "deleted_component_keys": 0,
        "authorized_c18_reversions": authorized_reversions,
        "non_c18_v5_records_preserved": len(v5_records) - authorized_reversions,
        "output_component_outcomes": dict(sorted(outcomes.items())),
        "output_component_value_sources": dict(sorted(source_classes.items())),
        "qualified_c16_records_retained": 1,
        "c16_route_step_count": c16_value.route_step_count,
        "c16_current_terminal_leaf_count": c16_value.current_terminal_leaf_count,
        "c18_parent_assessment_outcome": v3_target_value.assessment_outcome.value,
        "c18_parent_route_complete": v3_target_value.route_complete,
        "c18_parent_route_step_count": v3_target_value.route_step_count,
        "family_template_admitted_records": sum(
            record["family_template_admitted"] for record in output_records
        ),
        "homologue_template_admitted_records": sum(
            record["homologue_template_admitted"] for record in output_records
        ),
        "registry_pair_materialized": False,
        "holdout_artifacts_accessed": 0,
    }
    if summary != config.get("expected_summary"):
        raise Ugi3FrozenParentComponentLedgerError(
            "independently rebuilt parent-ledger summary changed"
        )

    lineage = _source_lineage(
        repo=repo,
        paths=paths,
        v5_result=v5_result,
        v3_result=v3_result,
    )
    ledger = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "status": FROZEN_STATUS,
        "config_sha256": sha256_file(config_path),
        "semantic_label": "r0_without_c18",
        "identity_policy": {
            "match_on_exact_role_and_canonical_smiles": True,
            "permit_similarity_or_family_matching": False,
            "permit_neighbor_homologue_admission": False,
            "permit_reaction_family_template_promotion": False,
        },
        "lineage": lineage,
        "summary": summary,
        "records": output_records,
    }
    ledger_payload = _gzip_json_bytes(ledger)
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": FROZEN_STATUS,
        "config_sha256": sha256_file(config_path),
        "summary": summary,
        "lineage": lineage,
        "inputs": {
            label: {
                "path": _portable(path, repo=repo),
                "sha256": sha256_file(path),
            }
            for label, path in sorted(paths.items())
        },
        "artifacts": {
            "final_component_ledger.json.gz": {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "sha256": sha256_bytes(ledger_payload),
            }
        },
        "policy": REQUIRED_POLICY,
        "adjudication": {
            "exact_c18_reversion_authorized": True,
            "qualified_c16_retained": True,
            "reaction_family_promoted": False,
            "homologue_scope_promoted": False,
            "registry_pair_materialized": False,
            "holdout_accessed": False,
            "holdout_reveal_authorized": False,
        },
        "nonclaims": [
            "This ledger does not materialize or bind the paired R0/R1 registries.",
            "This ledger does not authorize molecular holdout access or reveal.",
            "Restoring the pre-C18 record does not invalidate qualified C16 evidence.",
            "No reaction-family, similarity or homologue scope is promoted.",
        ],
    }
    return result, ledger_payload
