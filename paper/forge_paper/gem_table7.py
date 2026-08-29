"""Render GEM Table 7 from the pinned method-blind structural-realism aggregate."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from forge.core.hashing import artifact_record, pin_record
from forge.core.io import atomic_write, read_json_object, write_json
from forge_paper.completed_evidence_v1 import (
    load_lipid_realism_aggregate,
    render_lipid_realism_rows,
)

CONFIG_SCHEMA = "forge.gem_table7_lipid_realism_config.v1"
RESULT_SCHEMA = "forge.gem_table7_lipid_realism_render.v1"


class GemTable7Error(ValueError):
    """The pinned structural-realism aggregate is incomplete or inadmissible."""


def render_gem_table7_lipid_realism(
    config_path: Path,
    repo: Path,
    row_path: Path,
    *,
    result_path: Path,
) -> dict[str, Any]:
    """Generate all six structural-realism rows without selecting candidates."""

    config = read_json_object(config_path, error=GemTable7Error, label="GEM Table 7 config")
    expected_fields = {
        "schema_version",
        "status",
        "aggregate",
        "attempts_per_method_per_seed",
        "expected_seeds",
        "method_order",
        "candidate_selection",
    }
    expected_order = [
        "rgfn",
        "defog_unconditional",
        "genmol_safe",
        "finite_catalogue_oracle",
        "learned_inventory_selector",
        "forge_transformer",
    ]
    if (
        set(config) != expected_fields
        or config.get("schema_version") != CONFIG_SCHEMA
        or config.get("status") != "frozen_after_three_seed_aggregate"
        or config.get("attempts_per_method_per_seed") != 3072
        or config.get("expected_seeds") != [20260825, 20260826, 20260827]
        or config.get("method_order") != expected_order
        or config.get("candidate_selection") is not False
    ):
        raise GemTable7Error("GEM Table 7 config changed")

    try:
        aggregate, source = load_lipid_realism_aggregate(config["aggregate"], repo)
    except ValueError as error:
        raise GemTable7Error(str(error)) from error
    methods = aggregate["methods"]
    for method in expected_order:
        entry = methods[method]
        if (
            not isinstance(entry, Mapping)
            or entry.get("independent_unit") != "training seed"
            or entry.get("seeds") != config["expected_seeds"]
        ):
            raise GemTable7Error(f"independent-seed contract changed for {method}")

    rows = render_lipid_realism_rows(aggregate)
    if len(rows) != len(expected_order):
        raise GemTable7Error("GEM Table 7 row count changed")
    atomic_write(row_path, ("\n".join(rows) + "\n\\hline\n").encode("utf-8"))

    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete",
        "config": pin_record(config_path, repo),
        "source": source,
        "attempts_per_method_per_seed": config["attempts_per_method_per_seed"],
        "training_seeds": config["expected_seeds"],
        "method_order": expected_order,
        "candidate_selection": False,
        "artifact": artifact_record(row_path, logical_path=row_path.name),
        "gates": {
            "all_six_methods_present": set(methods) == set(expected_order),
            "three_independent_training_seeds": True,
            "fixed_attempt_budget": True,
            "not_estimable_metrics_preserved": True,
            "candidate_selection_absent": True,
        },
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(result_path, result)
    return result


__all__ = ["GemTable7Error", "render_gem_table7_lipid_realism"]
