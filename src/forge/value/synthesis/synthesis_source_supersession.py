"""Development-only replay audit for a superseded synthesis-value source.

Historical FORGE route/value configurations are immutable and intentionally
fail closed when ``src/forge/value/synthesis.py`` changes.  This module does
not weaken that behavior.  It copies each explicitly authorized historical
configuration into a temporary directory, substitutes only the declared
``value_source`` SHA-256, and compares the replayed ledgers and summaries with
their frozen parents.

No holdout or candidate-selection artifact is an admissible input to this
audit.
"""

from __future__ import annotations

import copy
import gzip
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from forge.data.r1_prime_audit import sha256_bytes, sha256_file
from forge.value.coverage.ugi3_fresh_pool_route_coverage_v5 import (
    build_fresh_pool_route_coverage_v5,
)
from forge.value.synthesis.ugi3_synthesis_value_audit import build_ugi3_synthesis_value_audit
from forge.value.synthesis.ugi3_synthesis_value_audit_v2 import build_ugi3_synthesis_value_audit_v2
from forge.value.synthesis.ugi3_synthesis_value_audit_v3 import build_ugi3_synthesis_value_audit_v3

CONFIG_SCHEMA_VERSION = "phase1_synthesis_source_supersession_config.v1"
RESULT_SCHEMA_VERSION = "phase1_synthesis_source_supersession_audit.v1"

SYNTHESIS_REPLAY_IDS = (
    "synthesis_value_v1",
    "synthesis_value_v2",
    "synthesis_value_v3",
)
FRESH_POOL_REPLAY_IDS = ("fresh_pool_route_coverage_v5",)
AUTHORIZED_REPLAY_IDS = SYNTHESIS_REPLAY_IDS + FRESH_POOL_REPLAY_IDS


class SynthesisSourceSupersessionError(ValueError):
    """Raised when source supersession cannot be qualified exactly."""


@dataclass(frozen=True)
class ReplayArtifacts:
    """Paths to one immutable historical replay target."""

    config: Path
    result: Path
    component_ledger: Path
    product_ledger: Path


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SynthesisSourceSupersessionError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise SynthesisSourceSupersessionError(f"{label} must be an object")
    return value


def _load_gzip_json_bytes(payload: bytes, *, label: str) -> Any:
    try:
        return json.loads(gzip.decompress(payload))
    except (OSError, json.JSONDecodeError) as error:
        raise SynthesisSourceSupersessionError(f"invalid {label}") from error


def _stable_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _semantic_json_hash(value: Any) -> str:
    payload = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return sha256_bytes(payload)


def _resolve_repo_path(repo: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise SynthesisSourceSupersessionError(f"{label} path is missing")
    path = Path(value)
    if not path.is_absolute():
        path = repo / path
    path = path.resolve()
    try:
        relative = path.relative_to(repo.resolve())
    except ValueError as error:
        raise SynthesisSourceSupersessionError(f"{label} escapes the repository") from error
    lowered = "/".join(relative.parts).lower()
    if "holdout" in lowered or "sealed" in lowered:
        raise SynthesisSourceSupersessionError(f"{label} is forbidden in this development audit")
    return path


def _validate_pinned_path(
    repo: Path,
    record: Mapping[str, Any],
    *,
    label: str,
) -> Path:
    if set(record) != {"path", "sha256"}:
        raise SynthesisSourceSupersessionError(f"{label} must define path and sha256")
    path = _resolve_repo_path(repo, record.get("path"), label=label)
    expected = record.get("sha256")
    if not isinstance(expected, str) or sha256_file(path) != expected:
        raise SynthesisSourceSupersessionError(f"{label} hash changed")
    return path


def _paths_from_asset_config(repo: Path, config_path: Path) -> dict[str, Path]:
    config = _load_json(config_path, label="asset-style replay config")
    inputs = config.get("inputs")
    if not isinstance(inputs, dict) or not inputs:
        raise SynthesisSourceSupersessionError("asset-style replay inputs are missing")
    output: dict[str, Path] = {}
    for label, record in inputs.items():
        if not isinstance(record, dict) or set(record) != {"asset", "expected_sha256"}:
            raise SynthesisSourceSupersessionError(f"asset-style input {label} is malformed")
        output[label] = _resolve_repo_path(repo, record.get("asset"), label=f"input {label}")
    return output


def substitute_value_source_hash(
    config: Mapping[str, Any],
    *,
    old_sha256: str,
    new_sha256: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Return a copy with exactly one source-hash field changed.

    Both historical input schemas are supported.  The source path and every
    other configuration field remain byte-semantically unchanged.
    """

    copied = copy.deepcopy(dict(config))
    inputs = copied.get("inputs")
    if not isinstance(inputs, dict):
        raise SynthesisSourceSupersessionError("replay config inputs are missing")
    source = inputs.get("value_source")
    if not isinstance(source, dict):
        raise SynthesisSourceSupersessionError("replay config lacks value_source")
    if set(source) == {"asset", "expected_sha256"}:
        hash_field = "expected_sha256"
    elif set(source) == {"path", "sha256"}:
        hash_field = "sha256"
    else:
        raise SynthesisSourceSupersessionError("value_source record shape changed")
    if source.get(hash_field) != old_sha256:
        raise SynthesisSourceSupersessionError("historical value_source hash changed")
    source[hash_field] = new_sha256

    expected = copy.deepcopy(dict(config))
    expected_inputs = expected["inputs"]
    expected_inputs["value_source"][hash_field] = new_sha256
    if copied != expected:
        raise SynthesisSourceSupersessionError("compatibility replay changed more than source hash")
    return copied, {
        "json_path": f"inputs.value_source.{hash_field}",
        "expected_before": old_sha256,
        "observed_after": new_sha256,
    }


def _normalize_result_provenance(value: Mapping[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(dict(value))
    config = normalized.get("config")
    if isinstance(config, dict):
        config["path"] = "<source-qualified-config>"
        config["sha256"] = "<source-qualified-config-sha256>"
    if "config_sha256" in normalized:
        normalized["config_sha256"] = "<source-qualified-config-sha256>"
    inputs = normalized.get("inputs")
    if isinstance(inputs, dict):
        source = inputs.get("value_source")
        if isinstance(source, dict):
            source["sha256"] = "<source-qualified-value-source-sha256>"
    return normalized


def _replay_artifacts_from_record(
    repo: Path,
    record: Mapping[str, Any],
    *,
    replay_id: str,
) -> ReplayArtifacts:
    if set(record) != {"config", "result", "component_ledger", "product_ledger"}:
        raise SynthesisSourceSupersessionError(f"replay {replay_id} artifact set changed")
    paths = {
        label: _validate_pinned_path(repo, item, label=f"{replay_id}.{label}")
        for label, item in record.items()
        if isinstance(item, Mapping)
    }
    if set(paths) != set(record):
        raise SynthesisSourceSupersessionError(f"replay {replay_id} artifact is malformed")
    return ReplayArtifacts(**paths)


def _run_builder(
    *,
    repo: Path,
    replay_id: str,
    config_path: Path,
) -> tuple[dict[str, Any], bytes, bytes]:
    if replay_id == "synthesis_value_v1":
        inputs = _paths_from_asset_config(repo, config_path)
        return build_ugi3_synthesis_value_audit(
            config_path=config_path,
            input_paths=inputs,
            targeted_audit_input_paths=_paths_from_asset_config(
                repo, inputs["targeted_audit_config"]
            ),
            role_gap_input_paths=_paths_from_asset_config(repo, inputs["role_gap_config"]),
            head_terminal_input_paths=_paths_from_asset_config(
                repo, inputs["head_terminal_config"]
            ),
        )
    if replay_id == "synthesis_value_v2":
        inputs = _paths_from_asset_config(repo, config_path)
        return build_ugi3_synthesis_value_audit_v2(
            config_path=config_path,
            input_paths=inputs,
            second_wave_input_paths=_paths_from_asset_config(repo, inputs["second_wave_config"]),
        )
    if replay_id == "synthesis_value_v3":
        inputs = _paths_from_asset_config(repo, config_path)
        return build_ugi3_synthesis_value_audit_v3(
            config_path=config_path,
            input_paths=inputs,
            third_wave_input_paths=_paths_from_asset_config(repo, inputs["third_wave_config"]),
        )
    if replay_id == "fresh_pool_route_coverage_v5":
        return build_fresh_pool_route_coverage_v5(repo, config_path)
    raise SynthesisSourceSupersessionError(f"unauthorized replay target: {replay_id}")


def _compare_replay(
    *,
    replay_id: str,
    result: Mapping[str, Any],
    component_ledger: bytes,
    product_ledger: bytes,
    frozen: ReplayArtifacts,
) -> dict[str, Any]:
    frozen_result = _load_json(frozen.result, label=f"{replay_id} frozen result")
    frozen_component = frozen.component_ledger.read_bytes()
    frozen_product = frozen.product_ledger.read_bytes()
    observed_result_bytes = _stable_json_bytes(result)
    expected_normalized = _normalize_result_provenance(frozen_result)
    observed_normalized = _normalize_result_provenance(result)

    component_semantic_equal = _load_gzip_json_bytes(
        component_ledger, label=f"{replay_id} replay component ledger"
    ) == _load_gzip_json_bytes(frozen_component, label=f"{replay_id} frozen component ledger")
    product_semantic_equal = _load_gzip_json_bytes(
        product_ledger, label=f"{replay_id} replay product ledger"
    ) == _load_gzip_json_bytes(frozen_product, label=f"{replay_id} frozen product ledger")

    comparison = {
        "component_ledger": {
            "expected_sha256": sha256_bytes(frozen_component),
            "observed_sha256": sha256_bytes(component_ledger),
            "byte_identical": component_ledger == frozen_component,
            "semantic_identical": component_semantic_equal,
        },
        "product_ledger": {
            "expected_sha256": sha256_bytes(frozen_product),
            "observed_sha256": sha256_bytes(product_ledger),
            "byte_identical": product_ledger == frozen_product,
            "semantic_identical": product_semantic_equal,
        },
        "summary": {
            "expected_sha256": _semantic_json_hash(frozen_result.get("summary")),
            "observed_sha256": _semantic_json_hash(result.get("summary")),
            "semantic_identical": result.get("summary") == frozen_result.get("summary"),
        },
        "normalized_result": {
            "expected_sha256": _semantic_json_hash(expected_normalized),
            "observed_sha256": _semantic_json_hash(observed_normalized),
            "semantic_identical": expected_normalized == observed_normalized,
        },
        "raw_result": {
            "expected_sha256": sha256_file(frozen.result),
            "observed_sha256": sha256_bytes(observed_result_bytes),
            "byte_identical": observed_result_bytes == frozen.result.read_bytes(),
            "expected_difference": "source-qualified config and value-source provenance",
        },
    }
    comparison["behavior_preserving"] = all(
        (
            comparison["component_ledger"]["byte_identical"],
            comparison["product_ledger"]["byte_identical"],
            comparison["summary"]["semantic_identical"],
            comparison["normalized_result"]["semantic_identical"],
        )
    )
    return comparison


def build_synthesis_source_supersession_audit(
    repo: Path,
    config_path: Path,
) -> dict[str, Any]:
    """Replay all authorized frozen targets under the current value source."""

    repo = repo.resolve()
    config = _load_json(config_path, label="source supersession config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise SynthesisSourceSupersessionError("unsupported source supersession config")
    if config.get("development_only") is not True:
        raise SynthesisSourceSupersessionError("source supersession must remain development-only")
    if config.get("sealed_holdout_access_authorized") is not False:
        raise SynthesisSourceSupersessionError("sealed holdout access must remain forbidden")

    source = config.get("source_supersession")
    if not isinstance(source, dict) or set(source) != {
        "path",
        "historical_sha256",
        "candidate_sha256",
    }:
        raise SynthesisSourceSupersessionError("source supersession declaration changed")
    source_path = _resolve_repo_path(repo, source.get("path"), label="candidate value source")
    historical_sha256 = source.get("historical_sha256")
    candidate_sha256 = source.get("candidate_sha256")
    if not isinstance(historical_sha256, str) or not isinstance(candidate_sha256, str):
        raise SynthesisSourceSupersessionError("source supersession hashes are malformed")
    observed_source_sha256 = sha256_file(source_path)
    if observed_source_sha256 != candidate_sha256:
        raise SynthesisSourceSupersessionError("candidate value source hash changed")
    if historical_sha256 == candidate_sha256:
        raise SynthesisSourceSupersessionError("source supersession hashes must differ")

    audit_inputs = config.get("audit_inputs")
    if not isinstance(audit_inputs, dict) or set(audit_inputs) != {
        "audit_source",
        "audit_runner",
        "audit_tests",
    }:
        raise SynthesisSourceSupersessionError("audit input set changed")
    validated_audit_inputs = {
        label: _validate_pinned_path(repo, record, label=label)
        for label, record in audit_inputs.items()
        if isinstance(record, Mapping)
    }
    if set(validated_audit_inputs) != set(audit_inputs):
        raise SynthesisSourceSupersessionError("audit input record is malformed")

    declared_replays = config.get("replays")
    if not isinstance(declared_replays, dict) or set(declared_replays) != set(
        AUTHORIZED_REPLAY_IDS
    ):
        raise SynthesisSourceSupersessionError("authorized replay set or order changed")

    replay_results: dict[str, Any] = {}
    temporary_parent = repo / "results" / "phase1"
    temporary_parent.mkdir(parents=True, exist_ok=True)
    temporary_root = temporary_parent / ".synthesis_source_supersession_replay"
    try:
        temporary_root.mkdir()
    except FileExistsError as error:
        raise SynthesisSourceSupersessionError(
            f"deterministic replay directory already exists: {temporary_root}"
        ) from error
    temporary_paths: list[Path] = []
    try:
        for replay_id in AUTHORIZED_REPLAY_IDS:
            frozen = _replay_artifacts_from_record(
                repo,
                declared_replays[replay_id],
                replay_id=replay_id,
            )
            historical_config = _load_json(frozen.config, label=f"{replay_id} config")
            replay_config, substitution = substitute_value_source_hash(
                historical_config,
                old_sha256=historical_sha256,
                new_sha256=candidate_sha256,
            )
            replay_config_path = temporary_root / f"{replay_id}.json"
            replay_config_path.write_bytes(_stable_json_bytes(replay_config))
            temporary_paths.append(replay_config_path)
            result, component_ledger, product_ledger = _run_builder(
                repo=repo,
                replay_id=replay_id,
                config_path=replay_config_path,
            )
            comparison = _compare_replay(
                replay_id=replay_id,
                result=result,
                component_ledger=component_ledger,
                product_ledger=product_ledger,
                frozen=frozen,
            )
            replay_results[replay_id] = {
                "substitution": substitution,
                "historical_config_sha256": sha256_file(frozen.config),
                "replay_config_sha256": sha256_file(replay_config_path),
                **comparison,
            }
    finally:
        for temporary_path in temporary_paths:
            temporary_path.unlink(missing_ok=True)
        temporary_root.rmdir()

    behavior_preserving = all(result["behavior_preserving"] for result in replay_results.values())
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": (
            "behavior_preserving_source_supersession_qualified"
            if behavior_preserving
            else "source_supersession_semantic_drift_detected"
        ),
        "development_only": True,
        "config": {
            "path": str(config_path.relative_to(repo)),
            "sha256": sha256_file(config_path),
        },
        "source_supersession": {
            "path": str(source_path.relative_to(repo)),
            "historical_sha256": historical_sha256,
            "candidate_sha256": candidate_sha256,
            "observed_sha256": observed_source_sha256,
        },
        "audit_inputs": {
            label: {
                "path": str(path.relative_to(repo)),
                "sha256": sha256_file(path),
            }
            for label, path in sorted(validated_audit_inputs.items())
        },
        "summary": {
            "replay_targets": len(replay_results),
            "synthesis_value_replays": len(SYNTHESIS_REPLAY_IDS),
            "fresh_pool_route_coverage_replays": len(FRESH_POOL_REPLAY_IDS),
            "byte_identical_component_ledgers": sum(
                result["component_ledger"]["byte_identical"] for result in replay_results.values()
            ),
            "byte_identical_product_ledgers": sum(
                result["product_ledger"]["byte_identical"] for result in replay_results.values()
            ),
            "semantically_identical_summaries": sum(
                result["summary"]["semantic_identical"] for result in replay_results.values()
            ),
            "semantically_identical_normalized_results": sum(
                result["normalized_result"]["semantic_identical"]
                for result in replay_results.values()
            ),
            "behavior_preserving": behavior_preserving,
        },
        "replays": replay_results,
        "adjudication": {
            "historical_configs_rewritten": False,
            "historical_ledgers_rewritten": False,
            "production_pin_update_authorized": False,
            "vnext_qualification_chain_required": True,
            "sealed_holdout_accessed": False,
            "synthesis_guidance_authorized": False,
        },
        "recommended_vnext_chain": [
            "Freeze this development-only source-supersession audit as independent evidence.",
            "Replay-qualify new exact-C18 and exact-C16 route versions under the candidate source; historical route configs remain immutable.",
            "Create a new synthesis-value source qualification that pins the candidate source, this audit and the two exact-route replays; do not repin v1-v3.",
            "Create the next fresh-pool route-coverage version from immutable v5 with the candidate source qualification as an owned input.",
            "Qualify downstream zero-guidance and matched-budget artifacts against that new fresh-pool version before any synthesis-guidance run.",
        ],
        "known_replay_boundary": {
            "fresh_pool_route_coverage_v4": (
                "A one-field top-level replay is structurally blocked because v4 independently "
                "re-executes the immutable exact-C18 audit, whose nested config also pins the "
                "historical synthesis-value source. Bypassing or rewriting that nested pin is "
                "forbidden; it requires a separate exact-route vNext qualification."
            )
        },
        "nonclaims": [
            "Behavior preservation does not establish synthesis-success probability.",
            "This replay does not update route coverage or evidence scope.",
            "This replay does not authorize guidance, candidate selection, or holdout access.",
        ],
    }
