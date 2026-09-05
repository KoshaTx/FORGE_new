from __future__ import annotations

from pathlib import Path

from experiments.phase1.multireaction.ugi_all_role_semantic_program_draw import (
    _load_frozen_draw,
)

REPO = Path(__file__).resolve().parents[1]
DRAW = REPO / "results/phase1/ugi_amine_semantic_program_draw_seed0_v2/program_draw.json"


def test_all_role_draw_preserves_frozen_program_and_baseline_target_order() -> None:
    programs, targets, document = _load_frozen_draw(DRAW, count=8)

    assert len(programs) == len(targets) == 8
    assert document["status"] == "pass"
    assert programs[0].node_counts == tuple(document["samples"][0]["program"]["node_counts"])
    assert targets[0].to_mapping() == document["samples"][0]["amine_semantic_target"]
