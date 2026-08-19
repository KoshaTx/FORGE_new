from __future__ import annotations

from pathlib import Path

from forge.product.ugi_expanded_exemplars import build_expanded_ugi_exemplars

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/model/phase1_ugi_expanded_exemplars.json"


def test_expanded_exemplar_repository_contract() -> None:
    result = build_expanded_ugi_exemplars(CONFIG, REPO)
    assert result["status"] == "pass"
    assert result["summary"]["admitted_components"] == 424
    assert result["summary"]["component_counts"] == {
        "amine_head": 264,
        "oxoester_aldehyde_body_tail": 107,
        "isocyanide_tail": 53,
    }
    assert result["summary"]["selected_exemplar_products"] <= 424
    assert all(result["gates"].values())
