"""Freeze ownership of a behavior-preserving synthesis-value source."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from forge.corpus.r1_prime_audit import sha256_file

CONFIG_SCHEMA_VERSION = "phase1_synthesis_source_qualification_config.v1"
RESULT_SCHEMA_VERSION = "phase1_synthesis_source_qualification.v1"


class SynthesisSourceQualificationError(ValueError):
    """Raised when source-qualification lineage is incomplete."""


def _load(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SynthesisSourceQualificationError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise SynthesisSourceQualificationError(f"{label} must be an object")
    return value


def _pinned(repo: Path, record: Mapping[str, Any], *, label: str) -> Path:
    if set(record) != {"path", "sha256"}:
        raise SynthesisSourceQualificationError(f"{label} pin is malformed")
    path = (repo / str(record.get("path"))).resolve()
    try:
        relative = path.relative_to(repo.resolve())
    except ValueError as error:
        raise SynthesisSourceQualificationError(f"{label} escapes repository") from error
    lowered = "/".join(relative.parts).lower()
    if "holdout" in lowered or "sealed" in lowered:
        raise SynthesisSourceQualificationError(f"{label} is forbidden")
    if sha256_file(path) != record.get("sha256"):
        raise SynthesisSourceQualificationError(f"{label} hash changed")
    return path


def build_synthesis_source_qualification(repo: Path, config_path: Path) -> dict[str, Any]:
    repo = repo.resolve()
    config = _load(config_path, label="source qualification config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise SynthesisSourceQualificationError("unsupported source qualification config")
    if (
        config.get("development_only") is not True
        or config.get("sealed_holdout_access_authorized") is not False
    ):
        raise SynthesisSourceQualificationError("source qualification scope changed")
    inputs = config.get("inputs")
    required = {
        "value_source",
        "compatibility_audit",
        "exact_route_replay",
        "qualifier_source",
        "qualifier_runner",
        "qualifier_tests",
    }
    if not isinstance(inputs, dict) or set(inputs) != required:
        raise SynthesisSourceQualificationError("source qualification inputs changed")
    paths = {
        label: _pinned(repo, record, label=label)
        for label, record in inputs.items()
        if isinstance(record, Mapping)
    }
    if set(paths) != required:
        raise SynthesisSourceQualificationError("source qualification pin is malformed")
    candidate_sha256 = sha256_file(paths["value_source"])
    compatibility = _load(paths["compatibility_audit"], label="compatibility audit")
    routes = _load(paths["exact_route_replay"], label="exact-route replay")
    if (
        compatibility.get("status") != "behavior_preserving_source_supersession_qualified"
        or compatibility.get("summary", {}).get("behavior_preserving") is not True
        or compatibility.get("summary", {}).get("replay_targets") != 4
        or compatibility.get("source_supersession", {}).get("candidate_sha256") != candidate_sha256
        or compatibility.get("adjudication", {}).get("sealed_holdout_accessed") is not False
        or compatibility.get("adjudication", {}).get("historical_configs_rewritten") is not False
    ):
        raise SynthesisSourceQualificationError("compatibility replay is not qualified")
    if (
        routes.get("status") != "exact_route_source_replays_behavior_preserving"
        or routes.get("summary", {}).get("behavior_preserving") is not True
        or routes.get("summary", {}).get("replays") != 2
        or routes.get("source_supersession", {}).get("candidate_sha256") != candidate_sha256
        or routes.get("adjudication", {}).get("sealed_holdout_accessed") is not False
        or routes.get("adjudication", {}).get("historical_artifacts_rewritten") is not False
    ):
        raise SynthesisSourceQualificationError("exact-route source replay is not qualified")
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "behavior_preserving_synthesis_value_source_qualified",
        "development_only": True,
        "config": {
            "path": str(config_path.relative_to(repo)),
            "sha256": sha256_file(config_path),
        },
        "qualified_source": {
            "path": str(paths["value_source"].relative_to(repo)),
            "sha256": candidate_sha256,
        },
        "owned_evidence": {
            label: {"path": str(path.relative_to(repo)), "sha256": sha256_file(path)}
            for label, path in sorted(paths.items())
            if label != "value_source"
        },
        "summary": {
            "synthesis_value_and_fresh_pool_replays": 4,
            "exact_route_replays": 2,
            "all_behavior_preserving": True,
        },
        "adjudication": {
            "historical_artifacts_rewritten": False,
            "fresh_pool_vnext_may_advance": True,
            "zero_guidance_requalification_may_advance": False,
            "nonzero_guidance_authorized": False,
            "sealed_holdout_accessed": False,
        },
    }
