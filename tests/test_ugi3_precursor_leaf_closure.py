from __future__ import annotations

import io
from pathlib import Path

from forge.route.terminals.ugi3_precursor_leaf_closure import (
    ALDEHYDE_ROLE,
    HEAD_ROLE,
    ISOCYANIDE_ROLE,
    _product_counts,
    constitutional_key,
    historical_leaf_source_use,
)


def test_constitutional_key_removes_only_stereochemical_identity() -> None:
    assert constitutional_key("CCCC/C=C\\CCCC") == constitutional_key("CCCCC=CCCCC")


def test_product_counts_requires_all_three_current_precursor_branches() -> None:
    programs = {
        (ALDEHYDE_ROLE, "aldehyde_a"): ("aldehyde_program", ("CCO",)),
        (ALDEHYDE_ROLE, "aldehyde_b"): ("aldehyde_program", ("CCCO",)),
        (ISOCYANIDE_ROLE, "isocyanide_a"): ("isocyanide_program", ("CCN",)),
    }
    csv_payload = io.StringIO(
        "amine_head_smiles,oxoester_aldehyde_body_tail_smiles,isocyanide_tail_smiles\n"
        "head_a,aldehyde_a,isocyanide_a\n"
        "head_b,aldehyde_a,isocyanide_a\n"
        "head_a,aldehyde_b,isocyanide_a\n"
    )
    audit = _product_counts(
        csv_payload,
        programs=programs,
        current_terminal_keys={"CCO", "CCN"},
        current_heads={"head_a"},
    )
    assert audit["product_count"] == 3
    assert audit["two_tail_leaf_closed"] == 2
    assert audit["head_and_two_tail_leaf_closed"] == 1
    assert audit["head_counts"] == {"head_a": 2, "head_b": 1}


def test_role_constants_remain_distinct() -> None:
    assert len({HEAD_ROLE, ALDEHYDE_ROLE, ISOCYANIDE_ROLE}) == 3


def test_historical_source_use_is_role_specific_and_exact() -> None:
    repo = Path(__file__).resolve().parents[1]
    evidence = historical_leaf_source_use(
        {
            "agile_component_routes": repo / "configs/route/m0_09_agile_component_routes.json",
            "hydrophobic_motif_transfer": repo
            / "configs/route/m0_09_hydrophobic_motif_transfer.json",
        },
        repo=repo,
    )
    assert (ALDEHYDE_ROLE, constitutional_key("OCCCCCCO")) in evidence
    assert (ISOCYANIDE_ROLE, constitutional_key("CCCCCCCCCCCCN")) in evidence
    assert (ALDEHYDE_ROLE, constitutional_key("CCCCCCCCCCCCO")) in evidence
    assert (ISOCYANIDE_ROLE, constitutional_key("CCOC(=O)CN")) not in evidence
