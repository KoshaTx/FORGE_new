from __future__ import annotations

from pathlib import Path

from forge.route.engine.qualified_forward import load_qualified_forward_reaction
from forge.route.evidence.ugi3_stepwise_route_program_adjudication import _execute_program
from forge.route.terminals.ugi3_precursor_leaf_closure import ALDEHYDE_ROLE


def test_stepwise_program_reconstructs_unseen_spacer_length() -> None:
    root = Path(__file__).resolve().parents[1]
    registry = root / "configs/route/phase1_ugi3_upstream_qualified_reactions_v1.json"
    transforms = {
        "esterification": load_qualified_forward_reaction(
            registry,
            root / "configs/route/variants/ugi3_upstream_esterification_exact_source_v1.json",
            reaction_id="ugi3_upstream_esterification_exact_source_v1",
        ),
        "alcohol_to_aldehyde_oxidation": load_qualified_forward_reaction(
            registry,
            root
            / "configs/route/variants/ugi3_upstream_primary_alcohol_oxidation_exact_source_v1.json",
            reaction_id="ugi3_upstream_primary_alcohol_oxidation_exact_source_v1",
        ),
    }
    _, receipts, verified = _execute_program(
        role=ALDEHYDE_ROLE,
        target="CCCCCCCC(=O)OCCCCCCCC=O",
        transforms=transforms,
    )
    assert verified is True
    assert len(receipts) == 2
    assert all(row["expected_product_reconstructed"] for row in receipts)
