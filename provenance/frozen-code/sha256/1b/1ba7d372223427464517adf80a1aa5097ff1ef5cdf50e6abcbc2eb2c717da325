from __future__ import annotations

from forge.product.ugi_production_route_shortlist_v2 import (
    _eligible_aldehyde_branch,
    _select_diverse,
)


def _row(draw: int, smiles: str, *, high: bool = False):
    return {
        "draw_index": draw,
        "canonical_product": smiles,
        "conservative_high_potency": high,
        "lcb90": float(draw),
    }


def test_diverse_selection_retains_every_required_high_product() -> None:
    rows = [
        _row(0, "CCCC", high=True),
        _row(1, "CCCCC", high=True),
        _row(2, "CCCCCC"),
        _row(3, "c1ccccc1"),
    ]

    selected = _select_diverse(rows, 3, required=rows[:2])

    assert {row["canonical_product"] for row in rows[:2]}.issubset(
        {row["canonical_product"] for row in selected}
    )
    assert len({row["canonical_product"] for row in selected}) == 3


def test_diverse_selection_deduplicates_products() -> None:
    rows = [
        _row(0, "CCCC", high=True),
        _row(1, "CCCC", high=False),
        _row(2, "CCCCC"),
    ]

    selected = _select_diverse(rows, 2, required=[rows[0]])

    assert {row["canonical_product"] for row in selected} == {"CCCC", "CCCCC"}


def test_branch_eligibility_separates_exploitation_from_corpus_absent_exploration() -> None:
    source = {
        "terminal_chemical_admission": {"admitted": True},
        "realized_carbon_branch_class": "aldehyde_origin_branched",
        "exact_refit_corpus_product": True,
        "tail_component_descriptors": {
            "oxoester_aldehyde_body_tail": {
                "carbon_atoms": 16,
                "carbon_branch_points": 1,
                "ester_carbonyls": 1,
                "adjacent_carbon_branch_edges": 0,
            }
        },
    }

    assert _eligible_aldehyde_branch(source, require_corpus_absent=False)
    assert not _eligible_aldehyde_branch(source, require_corpus_absent=True)
