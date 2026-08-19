"""Materialize immutable source-qualified inputs for the cumulative Ugi source."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from forge.core.io import atomic_write as _atomic_write
from forge.data.r1_prime_audit import sha256_file
from forge.route.ugi3_cumulative_production_source import CumulativeUgi3ProductionPaths
from forge.route.ugi3_exact_c16_route import build_exact_c16_route_audit
from forge.route.ugi3_exact_c18_route import build_exact_c18_route_audit
from forge.value.ugi3_fresh_pool_route_coverage_v2 import (
    build_fresh_pool_route_coverage_v2,
)
from forge.value.ugi3_fresh_pool_route_coverage_v3 import (
    build_fresh_pool_route_coverage_v3,
)

CONFIG_SCHEMA_VERSION = "phase1_ugi3_source_qualified_cumulative_inputs_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi3_source_qualified_cumulative_inputs.v1"


class Ugi3SourceQualifiedCumulativeInputsError(ValueError):
    """Raised when source-qualified cumulative inputs cannot reproduce history."""


def source_qualified_cumulative_paths(
    repo: Path,
    output_dir: Path,
) -> CumulativeUgi3ProductionPaths:
    """Return cumulative-source paths with only the four immutable derivatives replaced."""

    repo = repo.resolve()
    output = output_dir.resolve()
    base = CumulativeUgi3ProductionPaths.from_repo(repo)
    return CumulativeUgi3ProductionPaths(
        **{
            **base.__dict__,
            "fresh_v2_config": output / "fresh_v2/config.json",
            "fresh_v2_result": output / "fresh_v2/result.json",
            "fresh_v2_component_ledger": output / "fresh_v2/component_synthesis_values.json.gz",
            "fresh_v2_product_ledger": output / "fresh_v2/product_synthesis_values.json.gz",
            "fresh_v3_config": output / "fresh_v3/config.json",
            "fresh_v3_result": output / "fresh_v3/result.json",
            "fresh_v3_component_ledger": output / "fresh_v3/component_synthesis_values.json.gz",
            "fresh_v3_product_ledger": output / "fresh_v3/product_synthesis_values.json.gz",
            "exact_c18_config": output / "exact_c18/config.json",
            "exact_c18_result": output / "exact_c18/result.json",
            "exact_c18_step_ledger": output / "exact_c18/step_verification_ledger.json.gz",
            "exact_c18_assessment": output / "exact_c18/assessment.json.gz",
            "exact_c16_config": output / "exact_c16/config.json",
            "exact_c16_result": output / "exact_c16/result.json",
            "exact_c16_step_ledger": output / "exact_c16/step_verification_ledger.json.gz",
            "exact_c16_assessment": output / "exact_c16/assessment.json.gz",
        }
    )


def _load(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise Ugi3SourceQualifiedCumulativeInputsError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise Ugi3SourceQualifiedCumulativeInputsError(f"{label} must be an object")
    return value


def _pin(repo: Path, record: Mapping[str, Any], *, label: str) -> Path:
    if set(record) != {"path", "sha256"}:
        raise Ugi3SourceQualifiedCumulativeInputsError(f"{label} pin is malformed")
    path = (repo / str(record.get("path"))).resolve()
    try:
        relative = path.relative_to(repo.resolve())
    except ValueError as error:
        raise Ugi3SourceQualifiedCumulativeInputsError(f"{label} escapes repository") from error
    lowered = "/".join(relative.parts).lower()
    if "holdout" in lowered or "sealed" in lowered:
        raise Ugi3SourceQualifiedCumulativeInputsError(f"{label} is forbidden")
    if sha256_file(path) != record.get("sha256"):
        raise Ugi3SourceQualifiedCumulativeInputsError(f"{label} hash changed")
    return path


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _qualified_config(parent: Mapping[str, Any], *, source_sha256: str) -> dict[str, Any]:
    value = deepcopy(dict(parent))
    inputs = value.get("inputs")
    if not isinstance(inputs, dict):
        raise Ugi3SourceQualifiedCumulativeInputsError("parent config inputs are malformed")
    source = inputs.get("synthesis_value_source") or inputs.get("value_source")
    if not isinstance(source, dict) or source.get("path") != "src/forge/value/synthesis.py":
        raise Ugi3SourceQualifiedCumulativeInputsError(
            "parent config does not expose the synthesis-value source"
        )
    hash_key = "sha256" if "sha256" in source else "expected_sha256"
    source[hash_key] = source_sha256
    return value


def _input_paths(repo: Path, config: Mapping[str, Any]) -> dict[str, Path]:
    inputs = config.get("inputs")
    if not isinstance(inputs, dict):
        raise Ugi3SourceQualifiedCumulativeInputsError("config inputs are malformed")
    output: dict[str, Path] = {}
    for label, record in inputs.items():
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise Ugi3SourceQualifiedCumulativeInputsError(f"config input {label} is malformed")
        output[label] = repo / record["path"]
    return output


def _require_same_science(
    *,
    label: str,
    parent_result: Mapping[str, Any],
    result: Mapping[str, Any],
    parent_ledgers: tuple[Path, ...],
    ledgers: tuple[bytes, ...],
) -> None:
    if result.get("summary") != parent_result.get("summary"):
        raise Ugi3SourceQualifiedCumulativeInputsError(f"{label} summary changed")
    if len(parent_ledgers) != len(ledgers) or any(
        path.read_bytes() != payload for path, payload in zip(parent_ledgers, ledgers, strict=True)
    ):
        raise Ugi3SourceQualifiedCumulativeInputsError(f"{label} ledger bytes changed")


def build_source_qualified_cumulative_inputs(
    repo: Path, config_path: Path, output_dir: Path
) -> dict[str, Any]:
    """Build versioned configs/results while preserving every historical byte."""

    repo = repo.resolve()
    output = output_dir.resolve()
    try:
        output.relative_to(repo)
    except ValueError as error:
        raise Ugi3SourceQualifiedCumulativeInputsError(
            "output directory must remain in repository"
        ) from error
    lowered = "/".join(output.relative_to(repo).parts).lower()
    if "holdout" in lowered or "sealed" in lowered:
        raise Ugi3SourceQualifiedCumulativeInputsError("output directory is forbidden")
    config = _load(config_path, label="source-qualified cumulative-input config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise Ugi3SourceQualifiedCumulativeInputsError(
            "unsupported source-qualified cumulative-input config"
        )
    policy = config.get("policy")
    if policy != {
        "historical_artifacts_rewritten": False,
        "scientific_values_may_change": False,
        "source_lineage_only": True,
        "nonzero_guidance_authorized": False,
        "selection_authorized": False,
        "sealed_holdout_access_authorized": False,
    }:
        raise Ugi3SourceQualifiedCumulativeInputsError("source-qualified policy changed")
    inputs = config.get("inputs")
    required = {
        "value_source",
        "source_qualification",
        "exact_route_replay",
        "fresh_v2_config",
        "fresh_v2_result",
        "fresh_v2_component_values",
        "fresh_v2_product_values",
        "fresh_v3_config",
        "fresh_v3_result",
        "fresh_v3_component_values",
        "fresh_v3_product_values",
        "exact_c18_config",
        "exact_c18_result",
        "exact_c18_steps",
        "exact_c18_assessment",
        "exact_c16_config",
        "exact_c16_result",
        "exact_c16_steps",
        "exact_c16_assessment",
        "builder_source",
        "builder_runner",
        "builder_tests",
    }
    if not isinstance(inputs, dict) or set(inputs) != required:
        raise Ugi3SourceQualifiedCumulativeInputsError("source-qualified inputs changed")
    paths = {
        label: _pin(repo, record, label=label)
        for label, record in inputs.items()
        if isinstance(record, Mapping)
    }
    if set(paths) != required:
        raise Ugi3SourceQualifiedCumulativeInputsError("source-qualified pin is malformed")
    source_sha256 = sha256_file(paths["value_source"])
    qualification = _load(paths["source_qualification"], label="source qualification")
    replay = _load(paths["exact_route_replay"], label="exact route replay")
    if (
        qualification.get("status") != "behavior_preserving_synthesis_value_source_qualified"
        or qualification.get("qualified_source", {}).get("sha256") != source_sha256
        or replay.get("status") != "exact_route_source_replays_behavior_preserving"
        or replay.get("source_supersession", {}).get("candidate_sha256") != source_sha256
    ):
        raise Ugi3SourceQualifiedCumulativeInputsError(
            "source qualification or exact-route replay is not valid"
        )

    artifacts: dict[str, dict[str, Any]] = {}

    fresh_v2_config = _qualified_config(
        _load(paths["fresh_v2_config"], label="fresh v2 parent config"),
        source_sha256=source_sha256,
    )
    fresh_v2_dir = output / "fresh_v2"
    fresh_v2_config_path = fresh_v2_dir / "config.json"
    _atomic_write(fresh_v2_config_path, _json_bytes(fresh_v2_config))
    fresh_v2_result, fresh_v2_components, fresh_v2_products = build_fresh_pool_route_coverage_v2(
        repo, fresh_v2_config_path
    )
    _require_same_science(
        label="fresh v2",
        parent_result=_load(paths["fresh_v2_result"], label="fresh v2 parent result"),
        result=fresh_v2_result,
        parent_ledgers=(paths["fresh_v2_component_values"], paths["fresh_v2_product_values"]),
        ledgers=(fresh_v2_components, fresh_v2_products),
    )
    _atomic_write(fresh_v2_dir / "result.json", _json_bytes(fresh_v2_result))
    _atomic_write(fresh_v2_dir / "component_synthesis_values.json.gz", fresh_v2_components)
    _atomic_write(fresh_v2_dir / "product_synthesis_values.json.gz", fresh_v2_products)

    fresh_v3_config = _qualified_config(
        _load(paths["fresh_v3_config"], label="fresh v3 parent config"),
        source_sha256=source_sha256,
    )
    fresh_v3_config["inputs"]["base_result"] = {
        "path": str((fresh_v2_dir / "result.json").relative_to(repo)),
        "sha256": sha256_file(fresh_v2_dir / "result.json"),
    }
    for name, filename in (
        ("base_component_values", "component_synthesis_values.json.gz"),
        ("base_product_values", "product_synthesis_values.json.gz"),
    ):
        path = fresh_v2_dir / filename
        fresh_v3_config["inputs"][name] = {
            "path": str(path.relative_to(repo)),
            "sha256": sha256_file(path),
        }
    fresh_v3_dir = output / "fresh_v3"
    fresh_v3_config_path = fresh_v3_dir / "config.json"
    _atomic_write(fresh_v3_config_path, _json_bytes(fresh_v3_config))
    fresh_v3_result, fresh_v3_components, fresh_v3_products = build_fresh_pool_route_coverage_v3(
        repo, fresh_v3_config_path
    )
    _require_same_science(
        label="fresh v3",
        parent_result=_load(paths["fresh_v3_result"], label="fresh v3 parent result"),
        result=fresh_v3_result,
        parent_ledgers=(paths["fresh_v3_component_values"], paths["fresh_v3_product_values"]),
        ledgers=(fresh_v3_components, fresh_v3_products),
    )
    _atomic_write(fresh_v3_dir / "result.json", _json_bytes(fresh_v3_result))
    _atomic_write(fresh_v3_dir / "component_synthesis_values.json.gz", fresh_v3_components)
    _atomic_write(fresh_v3_dir / "product_synthesis_values.json.gz", fresh_v3_products)

    for label, builder, parent_names in (
        (
            "exact_c18",
            build_exact_c18_route_audit,
            ("exact_c18_config", "exact_c18_result", "exact_c18_steps", "exact_c18_assessment"),
        ),
        (
            "exact_c16",
            build_exact_c16_route_audit,
            ("exact_c16_config", "exact_c16_result", "exact_c16_steps", "exact_c16_assessment"),
        ),
    ):
        config_name, result_name, steps_name, assessment_name = parent_names
        derived_config = _qualified_config(
            _load(paths[config_name], label=f"{label} parent config"),
            source_sha256=source_sha256,
        )
        directory = output / label
        derived_config_path = directory / "config.json"
        _atomic_write(derived_config_path, _json_bytes(derived_config))
        result, steps, assessment = builder(
            config_path=derived_config_path,
            input_paths=_input_paths(repo, derived_config),
        )
        _require_same_science(
            label=label,
            parent_result=_load(paths[result_name], label=f"{label} parent result"),
            result=result,
            parent_ledgers=(paths[steps_name], paths[assessment_name]),
            ledgers=(steps, assessment),
        )
        _atomic_write(directory / "result.json", _json_bytes(result))
        _atomic_write(directory / "step_verification_ledger.json.gz", steps)
        _atomic_write(directory / "assessment.json.gz", assessment)

    for label, filenames in (
        (
            "fresh_v2",
            (
                "config.json",
                "result.json",
                "component_synthesis_values.json.gz",
                "product_synthesis_values.json.gz",
            ),
        ),
        (
            "fresh_v3",
            (
                "config.json",
                "result.json",
                "component_synthesis_values.json.gz",
                "product_synthesis_values.json.gz",
            ),
        ),
        (
            "exact_c18",
            (
                "config.json",
                "result.json",
                "step_verification_ledger.json.gz",
                "assessment.json.gz",
            ),
        ),
        (
            "exact_c16",
            (
                "config.json",
                "result.json",
                "step_verification_ledger.json.gz",
                "assessment.json.gz",
            ),
        ),
    ):
        directory = output / label
        artifacts[label] = {
            filename: {
                "path": str((directory / filename).relative_to(repo)),
                "sha256": sha256_file(directory / filename),
            }
            for filename in filenames
        }
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "source_qualified_cumulative_inputs_reproduced",
        "config": {
            "path": str(config_path.relative_to(repo)),
            "sha256": sha256_file(config_path),
        },
        "qualified_source_sha256": source_sha256,
        "inputs": {
            label: {"path": str(path.relative_to(repo)), "sha256": sha256_file(path)}
            for label, path in sorted(paths.items())
        },
        "artifacts": artifacts,
        "summary": {
            "qualified_layers": 4,
            "scientific_summaries_changed": 0,
            "ledger_byte_changes": 0,
        },
        "adjudication": {
            "cumulative_source_requalification_may_advance": True,
            "nonzero_guidance_authorized": False,
            "selection_authorized": False,
            "sealed_holdout_accessed": False,
        },
    }
