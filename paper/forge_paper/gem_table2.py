"""Render only the final FORGE row of GEM Table 2 from pinned common assessments."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from forge.core.hashing import artifact_record, pin_record, resolve_pin
from forge.core.io import atomic_write, read_json_object, write_json
from forge_paper.completed_evidence_v1 import _common_mean_row, _highlight_forge_rows

CONFIG_SCHEMA = "forge.natbiotech_v1_completed_evidence_config.v1"
RESULT_SCHEMA = "forge.gem_table2_forge_row_render.v1"
ASSESSMENT_SCHEMA = "forge.common_ugi_complete_assessment.v1"
EXPECTED_SEEDS = (20260825, 20260826, 20260827)


class GemTable2Error(ValueError):
    """The final FORGE common-assessment evidence is incomplete or changed."""


def render_gem_table2_forge_row(
    config_path: Path,
    repo: Path,
    row_path: Path,
    *,
    result_path: Path,
) -> dict[str, Any]:
    """Replace exactly one FORGE row, preserving every other generated Table 2 row."""

    config = read_json_object(
        config_path, error=GemTable2Error, label="completed-evidence config"
    )
    common = config.get("common_assessments")
    pins = common.get("forge_transformer") if isinstance(common, Mapping) else None
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or config.get("candidate_selection") is not False
        or not isinstance(pins, list)
        or len(pins) != 3
    ):
        raise GemTable2Error("completed-evidence config omits three final FORGE assessments")

    rows = []
    sources = []
    for pin in pins:
        if not isinstance(pin, Mapping):
            raise GemTable2Error("FORGE common-assessment pin is malformed")
        path = resolve_pin(pin, repo, label="final FORGE common assessment")
        payload = read_json_object(
            path, error=GemTable2Error, label="final FORGE common assessment"
        )
        assessment = payload.get("common_assessment")
        route = payload.get("route_evidence_assessment")
        if (
            payload.get("schema_version") != ASSESSMENT_SCHEMA
            or payload.get("status") != "pass"
            or payload.get("candidate_selection") is not False
            or not isinstance(assessment, Mapping)
            or assessment.get("method_id") != "forge_transformer"
            or not isinstance(route, Mapping)
            or route.get("route_or_oracle_calls") != 0
        ):
            raise GemTable2Error("final FORGE common assessment is inadmissible")
        rows.append(dict(assessment))
        sources.append(pin_record(path, repo))
    rows.sort(key=lambda row: int(row["seed"]))
    if [int(row["seed"]) for row in rows] != list(EXPECTED_SEEDS):
        raise GemTable2Error("final FORGE assessments changed the three paired seeds")

    existing = row_path.read_text().splitlines()
    positions = [index for index, line in enumerate(existing) if "FORGE Transformer" in line]
    if len(positions) != 1:
        raise GemTable2Error("Table 2 must contain exactly one FORGE Transformer row")
    rendered = _highlight_forge_rows(_common_mean_row("forge_transformer", rows))
    existing[positions[0]] = rendered
    atomic_write(row_path, ("\n".join(existing) + "\n").encode("utf-8"))

    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete",
        "candidate_selection": False,
        "method_id": "forge_transformer",
        "training_seeds": list(EXPECTED_SEEDS),
        "sources": sources,
        "artifact": artifact_record(row_path, logical_path=row_path.name),
        "rendered_row": rendered,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(result_path, result)
    return result


__all__ = ["GemTable2Error", "render_gem_table2_forge_row"]
