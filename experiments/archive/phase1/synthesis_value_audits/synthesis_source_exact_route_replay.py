"""Replay exact C18/C16 qualifications under a superseding value source."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from forge.corpus.r1_prime_audit import sha256_bytes, sha256_file
from forge.synthesis.evidence.ugi3_exact_c16_route import build_exact_c16_route_audit
from forge.synthesis.evidence.ugi3_exact_c18_route import build_exact_c18_route_audit

CONFIG_SCHEMA_VERSION = "phase1_synthesis_source_exact_route_replay_config.v1"
RESULT_SCHEMA_VERSION = "phase1_synthesis_source_exact_route_replay.v1"
REPLAY_IDS = ("exact_c18", "exact_c16")


class SynthesisSourceExactRouteReplayError(ValueError):
    """Raised when an exact-route source replay is not reproducible."""


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SynthesisSourceExactRouteReplayError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise SynthesisSourceExactRouteReplayError(f"{label} must be an object")
    return value


def _stable_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _semantic_hash(value: Any) -> str:
    return sha256_bytes(json.dumps(value, separators=(",", ":"), sort_keys=True).encode())


def _repo_path(repo: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise SynthesisSourceExactRouteReplayError(f"{label} path is missing")
    path = Path(value)
    if not path.is_absolute():
        path = repo / path
    path = path.resolve()
    try:
        relative = path.relative_to(repo.resolve())
    except ValueError as error:
        raise SynthesisSourceExactRouteReplayError(f"{label} escapes repository") from error
    lowered = "/".join(relative.parts).lower()
    if "holdout" in lowered or "sealed" in lowered:
        raise SynthesisSourceExactRouteReplayError(f"{label} is forbidden")
    return path


def _pinned(repo: Path, record: Mapping[str, Any], *, label: str) -> Path:
    if set(record) != {"path", "sha256"}:
        raise SynthesisSourceExactRouteReplayError(f"{label} pin is malformed")
    path = _repo_path(repo, record.get("path"), label=label)
    if sha256_file(path) != record.get("sha256"):
        raise SynthesisSourceExactRouteReplayError(f"{label} hash changed")
    return path


def _input_paths(repo: Path, config: Mapping[str, Any]) -> dict[str, Path]:
    inputs = config.get("inputs")
    if not isinstance(inputs, dict) or not inputs:
        raise SynthesisSourceExactRouteReplayError("exact-route inputs are missing")
    paths = {}
    for label, record in inputs.items():
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise SynthesisSourceExactRouteReplayError(f"exact-route input {label} is malformed")
        paths[label] = _repo_path(repo, record.get("path"), label=f"exact-route input {label}")
    return paths


def _patch_source(
    config: Mapping[str, Any], *, historical_sha256: str, candidate_sha256: str
) -> dict[str, Any]:
    replay = copy.deepcopy(dict(config))
    inputs = replay.get("inputs")
    if not isinstance(inputs, dict):
        raise SynthesisSourceExactRouteReplayError("exact-route inputs are missing")
    source = inputs.get("synthesis_value_source")
    if not isinstance(source, dict) or set(source) != {"path", "sha256"}:
        raise SynthesisSourceExactRouteReplayError("exact-route source record changed")
    if source.get("sha256") != historical_sha256:
        raise SynthesisSourceExactRouteReplayError("historical exact-route source hash changed")
    source["sha256"] = candidate_sha256
    expected = copy.deepcopy(dict(config))
    expected["inputs"]["synthesis_value_source"]["sha256"] = candidate_sha256
    if replay != expected:
        raise SynthesisSourceExactRouteReplayError("exact-route replay changed extra fields")
    return replay


def _normalized_result(value: Mapping[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(dict(value))
    normalized["config_sha256"] = "<source-qualified-config-sha256>"
    inputs = normalized.get("inputs")
    if isinstance(inputs, dict) and isinstance(inputs.get("synthesis_value_source"), dict):
        inputs["synthesis_value_source"]["sha256"] = "<qualified-source-sha256>"
    return normalized


def build_synthesis_source_exact_route_replay(repo: Path, config_path: Path) -> dict[str, Any]:
    repo = repo.resolve()
    config = _load_json(config_path, label="exact-route source replay config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise SynthesisSourceExactRouteReplayError("unsupported exact-route replay config")
    if (
        config.get("development_only") is not True
        or config.get("sealed_holdout_access_authorized") is not False
    ):
        raise SynthesisSourceExactRouteReplayError("exact-route replay scope changed")
    source = config.get("source_supersession")
    if not isinstance(source, dict) or set(source) != {
        "path",
        "historical_sha256",
        "candidate_sha256",
    }:
        raise SynthesisSourceExactRouteReplayError("source declaration changed")
    source_path = _repo_path(repo, source.get("path"), label="candidate source")
    historical_sha256 = str(source.get("historical_sha256"))
    candidate_sha256 = str(source.get("candidate_sha256"))
    if sha256_file(source_path) != candidate_sha256 or candidate_sha256 == historical_sha256:
        raise SynthesisSourceExactRouteReplayError("candidate source hash changed")

    supersession = config.get("supersession_audit")
    if not isinstance(supersession, dict):
        raise SynthesisSourceExactRouteReplayError("supersession audit pin is missing")
    supersession_path = _pinned(repo, supersession, label="supersession audit")
    supersession_result = _load_json(supersession_path, label="supersession audit")
    if (
        supersession_result.get("summary", {}).get("behavior_preserving") is not True
        or supersession_result.get("source_supersession", {}).get("candidate_sha256")
        != candidate_sha256
        or supersession_result.get("adjudication", {}).get("sealed_holdout_accessed") is not False
    ):
        raise SynthesisSourceExactRouteReplayError("supersession audit does not own candidate")

    audit_inputs = config.get("audit_inputs")
    if not isinstance(audit_inputs, dict) or set(audit_inputs) != {"source", "runner", "tests"}:
        raise SynthesisSourceExactRouteReplayError("audit input set changed")
    for label, record in audit_inputs.items():
        if not isinstance(record, Mapping):
            raise SynthesisSourceExactRouteReplayError(f"audit input {label} is malformed")
        _pinned(repo, record, label=f"audit input {label}")

    replay_specs = config.get("replays")
    if not isinstance(replay_specs, dict) or set(replay_specs) != set(REPLAY_IDS):
        raise SynthesisSourceExactRouteReplayError("exact-route replay set changed")
    temporary_root = repo / "results/phase1/.synthesis_source_exact_route_replay"
    try:
        temporary_root.mkdir()
    except FileExistsError as error:
        raise SynthesisSourceExactRouteReplayError(
            "deterministic replay directory exists"
        ) from error
    temporary_paths: list[Path] = []
    replays: dict[str, Any] = {}
    builders = {
        "exact_c18": build_exact_c18_route_audit,
        "exact_c16": build_exact_c16_route_audit,
    }
    try:
        for replay_id in REPLAY_IDS:
            spec = replay_specs[replay_id]
            if not isinstance(spec, dict) or set(spec) != {
                "config",
                "result",
                "step_ledger",
                "assessment",
            }:
                raise SynthesisSourceExactRouteReplayError(f"{replay_id} spec changed")
            paths = {
                label: _pinned(repo, record, label=f"{replay_id}.{label}")
                for label, record in spec.items()
                if isinstance(record, Mapping)
            }
            if set(paths) != set(spec):
                raise SynthesisSourceExactRouteReplayError(f"{replay_id} pin is malformed")
            historical_config = _load_json(paths["config"], label=f"{replay_id} config")
            replay_config = _patch_source(
                historical_config,
                historical_sha256=historical_sha256,
                candidate_sha256=candidate_sha256,
            )
            replay_config_path = temporary_root / f"{replay_id}.json"
            replay_config_path.write_bytes(_stable_bytes(replay_config))
            temporary_paths.append(replay_config_path)
            result, steps, assessment = builders[replay_id](
                config_path=replay_config_path,
                input_paths=_input_paths(repo, replay_config),
            )
            frozen_result = _load_json(paths["result"], label=f"{replay_id} result")
            frozen_steps = paths["step_ledger"].read_bytes()
            frozen_assessment = paths["assessment"].read_bytes()
            expected_normalized = _normalized_result(frozen_result)
            observed_normalized = _normalized_result(result)
            replay_result = {
                "step_ledger": {
                    "expected_sha256": sha256_bytes(frozen_steps),
                    "observed_sha256": sha256_bytes(steps),
                    "byte_identical": frozen_steps == steps,
                },
                "assessment": {
                    "expected_sha256": sha256_bytes(frozen_assessment),
                    "observed_sha256": sha256_bytes(assessment),
                    "byte_identical": frozen_assessment == assessment,
                },
                "summary": {
                    "expected_sha256": _semantic_hash(frozen_result.get("summary")),
                    "observed_sha256": _semantic_hash(result.get("summary")),
                    "semantic_identical": frozen_result.get("summary") == result.get("summary"),
                },
                "normalized_result": {
                    "expected_sha256": _semantic_hash(expected_normalized),
                    "observed_sha256": _semantic_hash(observed_normalized),
                    "semantic_identical": expected_normalized == observed_normalized,
                },
                "historical_config_sha256": sha256_file(paths["config"]),
                "replay_config_sha256": sha256_file(replay_config_path),
            }
            replay_result["behavior_preserving"] = all(
                (
                    replay_result["step_ledger"]["byte_identical"],
                    replay_result["assessment"]["byte_identical"],
                    replay_result["summary"]["semantic_identical"],
                    replay_result["normalized_result"]["semantic_identical"],
                )
            )
            replays[replay_id] = replay_result
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)
        temporary_root.rmdir()

    behavior_preserving = all(item["behavior_preserving"] for item in replays.values())
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": (
            "exact_route_source_replays_behavior_preserving"
            if behavior_preserving
            else "exact_route_source_replay_drift_detected"
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
        },
        "supersession_audit": {
            "path": str(supersession_path.relative_to(repo)),
            "sha256": sha256_file(supersession_path),
        },
        "summary": {
            "replays": len(replays),
            "byte_identical_step_ledgers": sum(
                item["step_ledger"]["byte_identical"] for item in replays.values()
            ),
            "byte_identical_assessments": sum(
                item["assessment"]["byte_identical"] for item in replays.values()
            ),
            "semantically_identical_summaries": sum(
                item["summary"]["semantic_identical"] for item in replays.values()
            ),
            "behavior_preserving": behavior_preserving,
        },
        "replays": replays,
        "adjudication": {
            "historical_artifacts_rewritten": False,
            "exact_route_scope_changed": False,
            "sealed_holdout_accessed": False,
            "synthesis_guidance_authorized": False,
            "source_qualification_may_advance": behavior_preserving,
        },
    }
