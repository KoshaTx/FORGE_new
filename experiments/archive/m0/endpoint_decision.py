"""Freeze the M0-08 endpoint package without selecting an endpoint in code."""

from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
from typing import Any

from experiments._runtime.historical import HistoricalPinArchiveError, resolve_pinned_input
from forge.core.hashing import sha256_file as _sha256_file
from forge.potency.endpoint import endpoint_ids, load_endpoint

CONFIG_SCHEMA_VERSION = "m0_08_endpoint_decision_config.v1"
RESULT_SCHEMA_VERSION = "m0_08_endpoint_decision.v1"


class EndpointDecisionError(ValueError):
    """Raised when the endpoint decision package violates its frozen contract."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise EndpointDecisionError(f"invalid M0-08 config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EndpointDecisionError("M0-08 config must be a JSON object")
    return value


def freeze_endpoint_decision(
    config_path: Path,
    output_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Verify inputs and emit the deterministic endpoint decision artifact."""

    config = _load_json(config_path)
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise EndpointDecisionError("unsupported M0-08 config schema")
    if config.get("randomness_used") is not False:
        raise EndpointDecisionError("M0-08 endpoint freezing must be deterministic")

    verified_inputs: dict[str, dict[str, Any]] = {}
    for name, record in sorted(config.get("inputs", {}).items()):
        if not isinstance(record, dict):
            raise EndpointDecisionError(f"M0-08 input {name} is malformed")
        relative = Path(str(record.get("path", "")))
        if not relative.parts or relative.is_absolute() or ".." in relative.parts:
            raise EndpointDecisionError(f"M0-08 input {name} has an unsafe path")
        try:
            path = resolve_pinned_input(repo_root, relative.as_posix(), str(record.get("sha256")))
        except HistoricalPinArchiveError as error:
            raise EndpointDecisionError(f"M0-08 input {name} is unavailable: {error}") from error
        observed = _sha256_file(path)
        if observed != record.get("sha256"):
            raise EndpointDecisionError(
                f"M0-08 input {name} hash mismatch: expected {record.get('sha256')}, "
                f"observed {observed}"
            )
        verified_inputs[name] = {
            "path": str(relative),
            "bytes": path.stat().st_size,
            "sha256": observed,
        }

    signature = inspect.signature(load_endpoint)
    if signature.parameters["endpoint_id"].default is not inspect.Parameter.empty:
        raise EndpointDecisionError("endpoint loader must not define an implicit default")

    specifications: dict[str, Any] = {}
    for endpoint_id in endpoint_ids():
        endpoint = load_endpoint(endpoint_id)
        specification = endpoint.specification
        specification.validate()
        specifications[endpoint_id] = {
            "display_name": specification.display_name,
            "administration_route": specification.administration_route,
            "required_readouts": list(endpoint.required_readout_ids()),
            "success_definition": list(specification.success_definition),
            "disallowed_inferences": list(specification.disallowed_inferences),
        }

    decision = config.get("decision")
    if not isinstance(decision, dict) or decision.get("endpoint_locked") is not False:
        raise EndpointDecisionError("M0-08 must remain explicitly unlocked")
    if decision.get("default_in_code") is not None:
        raise EndpointDecisionError("M0-08 cannot set a default endpoint in code")
    if decision.get("current_lowest_risk_option") not in endpoint_ids():
        raise EndpointDecisionError("M0-08 leading option is not a registered endpoint")

    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "task": "M0-08",
        "status": "decision_package_complete_endpoint_unlocked",
        "generated_utc": config["generated_utc"],
        "randomness_used": False,
        "config": {
            "path": str(config_path.resolve().relative_to(repo_root.resolve())),
            "sha256": _sha256_file(config_path),
        },
        "inputs": verified_inputs,
        "decision": decision,
        "endpoint_specifications": specifications,
        "claim_boundary": (
            "The package defines prospective evidence contracts. It does not provide "
            "biological measurements or authorize AGILE as an in-vivo endpoint oracle."
        ),
    }
    payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, output_path)
    return result
