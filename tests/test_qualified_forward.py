from __future__ import annotations

from pathlib import Path

import pytest

from forge.synthesis.engine.qualified_forward import (
    QualifiedForwardError,
    load_qualified_forward_reaction,
    unique_forward_products,
)

REPO = Path(__file__).resolve().parents[1]


def test_qualified_ugi_forward_reconstructs_known_product() -> None:
    compiled = load_qualified_forward_reaction(
        REPO / "data/vendor/qualified_reactions_v1.json",
        REPO / "configs/assembly/ugi_variant.yaml",
        reaction_id="ugi_3cr_agile",
    )
    products = unique_forward_products(
        compiled,
        ("CN(C)CCN", "CCCCCCCCCC=O", "[C-]#[N+]CCCCCCCCCCC"),
        max_products=100,
        isomeric_smiles=False,
    )

    assert compiled.role_names == (
        "amine_head",
        "oxoester_aldehyde_body_tail",
        "isocyanide_tail",
    )
    assert len(products) == 1


def test_qualified_forward_rejects_wrong_reactant_count() -> None:
    compiled = load_qualified_forward_reaction(
        REPO / "data/vendor/qualified_reactions_v1.json",
        REPO / "configs/assembly/ugi_variant.yaml",
        reaction_id="ugi_3cr_agile",
    )

    with pytest.raises(QualifiedForwardError, match="expected 3 reactants"):
        unique_forward_products(
            compiled,
            ("CN(C)CCN",),
            max_products=100,
            isomeric_smiles=False,
        )
