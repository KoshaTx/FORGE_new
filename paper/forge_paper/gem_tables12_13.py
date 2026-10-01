"""Render GEM Tables 12 and 13 from the pinned common-Ugi assessments."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from forge.core.hashing import artifact_record, pin_record
from forge.core.io import atomic_write, read_json_object, write_json
from forge_paper.completed_evidence_v1 import (
    CONFIG_SCHEMA,
    PAPER_COMMON_METHOD_ORDER,
    PRODUCTION_ATTEMPTS_PER_SEED,
    _common_decomposition_row,
    _common_seed_rows,
    _highlight_forge_rows,
    _load_common_assessments,
)

RESULT_SCHEMA = "forge.gem_tables12_13_common_ugi_render.v1"
EXPECTED_SEEDS = (20260825, 20260826, 20260827)


class GemTables12And13Error(ValueError):
    """The common-Ugi evidence needed by Tables 12 and 13 is incomplete or changed."""


def _write_rows(path: Path, rows: list[str]) -> None:
    rendered = _highlight_forge_rows("\n".join(rows) + "\n")
    atomic_write(path, rendered.encode("utf-8"))


def render_gem_tables12_and_13(
    config_path: Path,
    repo: Path,
    output_dir: Path,
    *,
    result_path: Path,
) -> dict[str, Any]:
    """Render all seed-level and decomposition rows without route-evidence dependencies."""

    config = read_json_object(
        config_path,
        error=GemTables12And13Error,
        label="GEM Tables 12/13 config",
    )
    common_pins = config.get("common_assessments")
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or config.get("candidate_selection") is not False
        or config.get("expected_seeds") != list(EXPECTED_SEEDS)
        or not isinstance(common_pins, Mapping)
    ):
        raise GemTables12And13Error("GEM Tables 12/13 config changed")

    try:
        common, _payloads, sources = _load_common_assessments(
            common_pins,
            repo,
            EXPECTED_SEEDS,
        )
    except ValueError as error:
        raise GemTables12And13Error(str(error)) from error

    seed_rows = [
        row
        for method in PAPER_COMMON_METHOD_ORDER
        for row in _common_seed_rows(method, common[method])
    ]
    decomposition_rows = [
        _common_decomposition_row(method, common[method]) for method in PAPER_COMMON_METHOD_ORDER
    ]
    if len(seed_rows) != len(PAPER_COMMON_METHOD_ORDER) * len(EXPECTED_SEEDS):
        raise GemTables12And13Error("seed-level row count changed")
    if len(decomposition_rows) != len(PAPER_COMMON_METHOD_ORDER):
        raise GemTables12And13Error("decomposition row count changed")

    output_dir.mkdir(parents=True, exist_ok=True)
    seed_path = output_dir / "common_ugi_seed_rows.tex"
    decomposition_path = output_dir / "common_ugi_decomposition_rows.tex"
    _write_rows(seed_path, seed_rows)
    _write_rows(decomposition_path, decomposition_rows)

    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete",
        "config": pin_record(config_path, repo),
        "sources": sources,
        "attempts_per_method_per_seed": PRODUCTION_ATTEMPTS_PER_SEED,
        "training_seeds": list(EXPECTED_SEEDS),
        "method_order": list(PAPER_COMMON_METHOD_ORDER),
        "candidate_selection": False,
        "artifacts": {
            "table_12_seed_rows": artifact_record(seed_path, logical_path=seed_path.name),
            "table_13_decomposition_rows": artifact_record(
                decomposition_path,
                logical_path=decomposition_path.name,
            ),
        },
        "gates": {
            "all_nine_methods_present": set(common) == set(PAPER_COMMON_METHOD_ORDER),
            "three_independent_training_seeds_per_method": all(
                [int(row["seed"]) for row in common[method]] == list(EXPECTED_SEEDS)
                for method in PAPER_COMMON_METHOD_ORDER
            ),
            "fixed_attempt_budget": all(
                int(row["attempts"]) == PRODUCTION_ATTEMPTS_PER_SEED
                for method in PAPER_COMMON_METHOD_ORDER
                for row in common[method]
            ),
            "coverage_and_precision_reported": all(
                row.get("coverage_and_precision_reported") is True
                for method in PAPER_COMMON_METHOD_ORDER
                for row in common[method]
            ),
            "not_estimable_metrics_preserved": True,
            "candidate_selection_absent": True,
        },
    }
    if not all(result["gates"].values()):
        raise GemTables12And13Error("GEM Tables 12/13 evidence gate failed")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(result_path, result)
    return result


__all__ = ["GemTables12And13Error", "render_gem_tables12_and_13"]
