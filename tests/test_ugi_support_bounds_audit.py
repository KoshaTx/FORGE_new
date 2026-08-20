from __future__ import annotations

import csv
from pathlib import Path

from forge.design.audit.ugi_support_bounds_audit import (
    _product_capacity_summary,
    component_capacity_metrics,
)

BOUNDS = {
    "maximum_component_atoms": 12,
    "maximum_children": 3,
    "maximum_junction_budget": 2,
    "maximum_cycle_rank": 1,
}


def test_component_capacity_metrics_accepts_linear_and_cyclic_components() -> None:
    linear = component_capacity_metrics("CCCCCCCC=O", structure_id="linear", bounds=BOUNDS)
    cyclic = component_capacity_metrics("O=CC1CCCCC1", structure_id="cyclic", bounds=BOUNDS)

    assert linear["capacity_supported"]
    assert linear["cycle_rank"] == 0
    assert cyclic["capacity_supported"]
    assert cyclic["cycle_rank"] == 1


def test_component_capacity_metrics_reports_specific_ceiling_failure() -> None:
    result = component_capacity_metrics(
        "CCCCCCCCCCCCCCCC=O",
        structure_id="too_large",
        bounds=BOUNDS,
    )

    assert not result["capacity_supported"]
    assert result["failure_reasons"] == ["support_atoms"]


def test_product_failing_two_bounds_is_counted_once(tmp_path: Path) -> None:
    assignments = tmp_path / "assignments.csv"
    with assignments.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "product_id",
                "canonical_product_smiles",
                "amine_head_smiles",
                "oxoester_aldehyde_body_tail_smiles",
                "isocyanide_tail_smiles",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "product_id": "both_failures",
                "canonical_product_smiles": "CC",
                "amine_head_smiles": "amine",
                "oxoester_aldehyde_body_tail_smiles": "aldehyde",
                "isocyanide_tail_smiles": "isocyanide",
            }
        )
    exact = {
        ("amine_head", "amine"): {"support_atoms": 4, "terminal_decorations": 2},
        ("oxoester_aldehyde_body_tail", "aldehyde"): {
            "support_atoms": 4,
            "terminal_decorations": 2,
        },
        ("isocyanide_tail", "isocyanide"): {
            "support_atoms": 4,
            "terminal_decorations": 2,
        },
    }

    result = _product_capacity_summary(
        assignments,
        exact,
        {"maximum_total_atoms": 10, "maximum_decorations": 5},
    )

    assert result["products"] == 1
    assert result["capacity_supported"] == 0
    assert result["failure_reasons"] == {
        "terminal_decorations": 1,
        "total_exterior_atoms": 1,
    }
