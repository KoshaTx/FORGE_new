"""Audit whether the frozen M0-10 FlowER transfer pilot can run scientifically.

The qualified reaction registry stores atom-mapped forward transforms and small
positive/negative qualification tests. Those records are not automatically
FlowER elementary mechanism trajectories. This module preserves that
distinction and applies the predeclared FlowER demotion rule mechanically.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

CONFIG_SCHEMA_VERSION = "m0_10_flower_transfer_audit_config.v1"
RESULT_SCHEMA_VERSION = "m0_10_flower_transfer_audit.v1"


class FlowerTransferAuditError(ValueError):
    """Raised when the M0-10 audit configuration or pinned input is invalid."""


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise FlowerTransferAuditError(f"{label} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise FlowerTransferAuditError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise FlowerTransferAuditError(f"{label} must contain a JSON object")
    return payload


def _safe_workspace_path(repo_root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute():
        raise FlowerTransferAuditError(f"configured path must be relative: {relative}")
    resolved = (repo_root / candidate).resolve()
    workspace_root = repo_root.resolve().parent
    if not resolved.is_relative_to(workspace_root):
        raise FlowerTransferAuditError(f"configured path escapes the workspace: {relative}")
    return resolved


def _verify_required_json(
    repo_root: Path, specification: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    label = specification.get("label")
    relative = specification.get("path")
    expected_hash = specification.get("sha256")
    if (
        not isinstance(label, str)
        or not isinstance(relative, str)
        or not isinstance(expected_hash, str)
        or len(expected_hash) != 64
    ):
        raise FlowerTransferAuditError("required JSON input specification is incomplete")
    path = _safe_workspace_path(repo_root, relative)
    if not path.is_file():
        raise FlowerTransferAuditError(f"{label} not found: {path}")
    observed_hash = sha256_file(path)
    if observed_hash != expected_hash:
        raise FlowerTransferAuditError(
            f"{label} hash mismatch: expected {expected_hash}, observed {observed_hash}"
        )
    return (
        _load_json(path, label),
        {
            "label": label,
            "path": relative,
            "sha256": observed_hash,
            "bytes": path.stat().st_size,
        },
    )


def _audit_optional_evidence(repo_root: Path, specification: Mapping[str, Any]) -> dict[str, Any]:
    label = specification.get("label")
    relative = specification.get("path")
    expected_hash = specification.get("sha256")
    if (
        not isinstance(label, str)
        or not isinstance(relative, str)
        or not isinstance(expected_hash, str)
        or len(expected_hash) != 64
    ):
        raise FlowerTransferAuditError("prior evidence specification is incomplete")
    path = _safe_workspace_path(repo_root, relative)
    if not path.is_file():
        return {
            "label": label,
            "path": relative,
            "present": False,
            "expected_sha256": expected_hash,
        }
    observed_hash = sha256_file(path)
    return {
        "label": label,
        "path": relative,
        "present": True,
        "hash_matches": observed_hash == expected_hash,
        "sha256": observed_hash,
        "bytes": path.stat().st_size,
    }


def _audit_asset(repo_root: Path, label: str, specification: Mapping[str, Any]) -> dict[str, Any]:
    candidates = specification.get("candidate_paths")
    if (
        not isinstance(candidates, list)
        or not candidates
        or not all(isinstance(value, str) for value in candidates)
    ):
        raise FlowerTransferAuditError(f"{label} candidate paths are incomplete")
    observations = []
    for relative in candidates:
        path = _safe_workspace_path(repo_root, relative)
        present = path.is_file() if label == "pretrained_checkpoint" else path.is_dir()
        row: dict[str, Any] = {"path": relative, "present": present}
        if present and path.is_file():
            row.update({"sha256": sha256_file(path), "bytes": path.stat().st_size})
        observations.append(row)
    result = {
        "description": specification.get("description"),
        "present": any(row["present"] for row in observations),
        "candidates": observations,
    }
    if "upstream_locator" in specification:
        result["upstream_locator"] = specification["upstream_locator"]
    return result


def _registry_rows(registry_payloads: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for payload in registry_payloads:
        reactions = payload.get("reactions")
        if not isinstance(reactions, list):
            raise FlowerTransferAuditError("reaction registry lacks a reactions list")
        for reaction in reactions:
            if not isinstance(reaction, dict) or not isinstance(reaction.get("reaction_id"), str):
                raise FlowerTransferAuditError("reaction registry contains an invalid row")
            reaction_id = reaction["reaction_id"]
            if reaction_id in rows:
                raise FlowerTransferAuditError(f"duplicate reaction id: {reaction_id}")
            rows[reaction_id] = reaction
    return rows


def _trajectory_count(reaction: Mapping[str, Any], fields: Iterable[str]) -> int:
    count = 0
    for field in fields:
        values = reaction.get(field)
        if values is None:
            continue
        if not isinstance(values, list):
            raise FlowerTransferAuditError(f"{field} must be a list when present")
        count += len(values)
    return count


def _string_values_for_key(payload: Any, key: str) -> list[str]:
    values: list[str] = []
    if isinstance(payload, dict):
        for current_key, value in payload.items():
            if current_key == key and isinstance(value, str):
                values.append(value)
            values.extend(_string_values_for_key(value, key))
    elif isinstance(payload, list):
        for value in payload:
            values.extend(_string_values_for_key(value, key))
    return values


def audit_flower_transfer_inputs(
    config: Mapping[str, Any],
    registry_payloads: Iterable[Mapping[str, Any]],
    prospective_payload: Mapping[str, Any],
    asset_audit: Mapping[str, Mapping[str, Any]],
    prior_evidence: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return the frozen readiness and demotion decision without fitting a model."""

    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise FlowerTransferAuditError("unsupported M0-10 config schema")
    target_classes = config.get("target_classes")
    trajectory_fields = config.get("mechanism_trajectory_fields")
    pilot_contract = config.get("pilot_contract")
    demotion_rule = config.get("demotion_rule")
    if (
        not isinstance(target_classes, list)
        or len(target_classes) not in {2, 3}
        or len(set(target_classes)) != len(target_classes)
        or not all(isinstance(value, str) for value in target_classes)
    ):
        raise FlowerTransferAuditError("pilot must freeze two or three unique target classes")
    if not isinstance(trajectory_fields, list) or not all(
        isinstance(value, str) for value in trajectory_fields
    ):
        raise FlowerTransferAuditError("mechanism trajectory fields are invalid")
    if not isinstance(pilot_contract, dict) or not isinstance(demotion_rule, dict):
        raise FlowerTransferAuditError("pilot or demotion contract is missing")
    minimum = int(pilot_contract.get("minimum_mechanism_trajectories_per_class", -1))
    if minimum != 32 or pilot_contract.get("curve_examples_per_class") != [0, 8, 16, 32]:
        raise FlowerTransferAuditError("the frozen 32-example fine-tuning curve changed")
    if (
        demotion_rule.get("headline_requires_finetuning_improves_heldout_forward_consistency")
        is not True
        or demotion_rule.get("headline_requires_prospective_conversion_spearman_ci_excludes_zero")
        is not True
        or demotion_rule.get("deterministic_atom_mapped_verifier_load_bearing") is not True
    ):
        raise FlowerTransferAuditError("the frozen FlowER demotion rule changed")

    rows = _registry_rows(registry_payloads)
    class_rows = []
    for reaction_id in target_classes:
        if reaction_id not in rows:
            raise FlowerTransferAuditError(f"target class is absent from registry: {reaction_id}")
        reaction = rows[reaction_id]
        positive = reaction.get("known_positive_examples", [])
        negative = reaction.get("known_negative_examples", [])
        if not isinstance(positive, list) or not isinstance(negative, list):
            raise FlowerTransferAuditError(f"qualification examples invalid for {reaction_id}")
        trajectory_count = _trajectory_count(reaction, trajectory_fields)
        class_rows.append(
            {
                "reaction_id": reaction_id,
                "atom_mapped_transform_present": isinstance(
                    reaction.get("atom_mapped_reaction_smarts"), str
                ),
                "qualification_positive_examples": len(positive),
                "qualification_negative_examples": len(negative),
                "elementary_mechanism_trajectories": trajectory_count,
                "minimum_required": minimum,
                "trajectory_deficit": max(0, minimum - trajectory_count),
                "fine_tuning_curve_ready": trajectory_count >= minimum,
                "qualification_examples_are_flowER_trajectories": False,
            }
        )

    prospective_states = _string_values_for_key(prospective_payload, "prospective_outcome")
    attempted_states = [state for state in prospective_states if state != "not_attempted"]
    source_present = bool(asset_audit["source_code"]["present"])
    checkpoint_present = bool(asset_audit["pretrained_checkpoint"]["present"])
    class_data_ready = all(row["fine_tuning_curve_ready"] for row in class_rows)
    prior_hashes_valid = all(
        not row.get("present", False) or row.get("hash_matches", False) for row in prior_evidence
    )

    blockers = []
    if not source_present:
        blockers.append("hash_pinned_flower_source_absent")
    if not checkpoint_present:
        blockers.append("pretrained_flower_checkpoint_absent")
    if not class_data_ready:
        blockers.append("minimum_32_elementary_mechanism_trajectories_per_class_absent")
    if not attempted_states:
        blockers.append("prospective_conversion_outcomes_absent")
    if not prior_hashes_valid:
        blockers.append("prior_probe_evidence_hash_mismatch")

    forward_improvement_evaluable = source_present and checkpoint_present and class_data_ready
    conversion_correlation_evaluable = len(attempted_states) > 0
    headline_rule_satisfied = False
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "blocked_scientifically_valid_pilot_not_executed_flower_demoted",
        "task": config["task"],
        "seed": int(config["seed"]),
        "generated_at_utc": config["generated_at_utc"],
        "target_classes": class_rows,
        "asset_readiness": dict(asset_audit),
        "prior_probe_evidence": prior_evidence,
        "prospective_outcomes": {
            "states_found": len(prospective_states),
            "not_attempted": prospective_states.count("not_attempted"),
            "attempted_with_conversion_measurement": len(attempted_states),
            "observed_attempted_states": sorted(set(attempted_states)),
        },
        "pilot_readiness": {
            "source_code_ready": source_present,
            "pretrained_checkpoint_ready": checkpoint_present,
            "class_curve_data_ready": class_data_ready,
            "reaction_class_holdout_ready": class_data_ready,
            "calibration_ready": class_data_ready,
            "fine_tuning_run_executed": False,
            "acceptance_criterion_satisfied": False,
            "blockers": blockers,
        },
        "demotion_rule": {
            "heldout_forward_consistency_improvement_evaluable": forward_improvement_evaluable,
            "prospective_conversion_correlation_evaluable": conversion_correlation_evaluable,
            "headline_rule_satisfied": headline_rule_satisfied,
            "flowER_role": demotion_rule["otherwise_role"],
            "deterministic_atom_mapped_verifier_load_bearing": True,
            "mechanism_grounded_claim_authorized": False,
            "reason": (
                "Both frozen headline conditions must pass. Neither a valid fine-tuning comparison "
                "nor a prospective conversion correlation can currently be evaluated."
            ),
        },
        "acquisition_boundary": config["acquisition_boundary"],
        "scientific_interpretation": {
            "registry_examples": (
                "The registry examples qualify deterministic atom-mapped transforms. They do not "
                "encode the elementary bond-electron trajectories required for FlowER fine-tuning."
            ),
            "prior_probe": (
                "The May 2026 zero-shot probe is preserved as prior evidence only and is not "
                "substituted for the frozen M0-10 holdout and fine-tuning curve."
            ),
            "training_dependency": (
                "This blocker does not prevent training the whole-lipid product model or the hybrid "
                "L2 route system because FlowER is not load-bearing."
            ),
        },
    }


def run_flower_transfer_audit(
    config_path: Path, output_path: Path, repo_root: Path
) -> dict[str, Any]:
    config = _load_json(config_path, "M0-10 config")
    registry_payloads = []
    input_rows = []
    for specification in config.get("registry_inputs", []):
        payload, row = _verify_required_json(repo_root, specification)
        registry_payloads.append(payload)
        input_rows.append(row)
    prospective_payload, prospective_row = _verify_required_json(
        repo_root, config.get("prospective_outcome_input", {})
    )
    input_rows.append(prospective_row)
    prior_evidence = [
        _audit_optional_evidence(repo_root, specification)
        for specification in config.get("prior_local_evidence", [])
    ]
    required_assets = config.get("required_assets")
    if not isinstance(required_assets, dict) or set(required_assets) != {
        "source_code",
        "pretrained_checkpoint",
    }:
        raise FlowerTransferAuditError("required FlowER asset contract changed")
    asset_audit = {
        label: _audit_asset(repo_root, label, specification)
        for label, specification in required_assets.items()
    }
    result = audit_flower_transfer_inputs(
        config,
        registry_payloads,
        prospective_payload,
        asset_audit,
        prior_evidence,
    )
    result["inputs"] = input_rows
    result["config"] = {
        "path": os.path.relpath(config_path.resolve(), repo_root.resolve()),
        "sha256": sha256_file(config_path),
        "bytes": config_path.stat().st_size,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output_path.name}.", dir=output_path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, output_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return result
