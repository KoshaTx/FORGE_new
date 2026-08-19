"""Turn a completed pytest run into a reproducible no-new-failures report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from forge.core.hashing import sha256_file
from forge.core.io import write_json


class TestBaselineError(ValueError):
    """The pytest cache or documented baseline is missing or malformed."""


def _lines(path: Path) -> tuple[str, ...]:
    try:
        return tuple(
            line.strip()
            for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    except OSError as error:
        raise TestBaselineError(f"could not read baseline input {path}: {error}") from error


def _json(path: Path, expected: type[list[Any]] | type[dict[str, Any]]) -> Any:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise TestBaselineError(f"could not read pytest cache {path}: {error}") from error
    if not isinstance(value, expected):
        raise TestBaselineError(f"pytest cache {path} must contain {expected.__name__}")
    return value


def _input(repo: Path, path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(repo.resolve()).as_posix()
    except ValueError:
        relative = str(resolved)
    return {
        "bytes": resolved.stat().st_size,
        "path": relative,
        "sha256": str(sha256_file(resolved)),
    }


def build_test_baseline_report(
    repo: Path,
    *,
    baseline_path: Path,
    missing_inputs_path: Path,
    lastfailed_path: Path,
    collected_path: Path,
    observed_output: Path,
    output: Path | None = None,
) -> dict[str, Any]:
    """Compare the last complete pytest run with the reviewed failure-node baseline.

    Run pytest with ``--cache-clear`` first. Intersecting ``lastfailed`` with the collected node
    list prevents deleted or renamed tests from being misreported as current failures, while the
    explicit stale list makes such cache state visible rather than silently discarding it.
    """

    repo = repo.resolve()
    baseline = set(_lines(baseline_path))
    missing_inventory = _lines(missing_inputs_path)
    lastfailed_document = _json(lastfailed_path, dict)
    collected_document = _json(collected_path, list)
    if not all(isinstance(key, str) for key in lastfailed_document):
        raise TestBaselineError("pytest lastfailed keys must be node-id strings")
    if not all(isinstance(node, str) for node in collected_document):
        raise TestBaselineError("pytest collected node ids must be strings")

    lastfailed = set(lastfailed_document)
    collected = set(collected_document)
    observed = lastfailed & collected
    stale_cache = lastfailed - collected
    new_failures = observed - baseline
    resolved = baseline - observed
    missing_paths = sorted(path for path in missing_inventory if not (repo / path).exists())
    available_paths = sorted(set(missing_inventory) - set(missing_paths))
    if new_failures:
        status = "regression"
    elif observed:
        status = "blocked_known_failures"
    else:
        status = "pass"

    observation = {
        "collected_node_count": len(collected),
        "failed_node_ids": sorted(observed),
        "schema_version": "forge.pytest_failure_observation.v1",
        "stale_cache_nodes": sorted(stale_cache),
    }
    write_json(observed_output, observation)

    document = {
        "baseline_failure_count": len(baseline),
        "collected_node_count": len(collected),
        "current_failure_count": len(observed),
        "documented_missing_inputs": {
            "available": available_paths,
            "available_count": len(available_paths),
            "missing": missing_paths,
            "missing_count": len(missing_paths),
        },
        "inputs": {
            "baseline": _input(repo, baseline_path),
            "missing_inputs": _input(repo, missing_inputs_path),
            "observation": _input(repo, observed_output),
        },
        "new_failures": sorted(new_failures),
        "no_new_failures": not new_failures,
        "ready": not observed,
        "resolved_failures": sorted(resolved),
        "schema_version": "forge.test_baseline_report.v1",
        "stale_cache_nodes": sorted(stale_cache),
        "status": status,
    }
    if output is not None:
        write_json(output, document)
    return document


__all__ = ["TestBaselineError", "build_test_baseline_report"]
