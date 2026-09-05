from __future__ import annotations

import json
from pathlib import Path

from experiments.phase1.multireaction.ugi_all_role_semantic_visual_adjudication import (
    run_ugi_all_role_semantic_visual_adjudication,
)
from experiments.phase1.multireaction.ugi_all_role_semantic_visual_review import (
    BLINDING_KEY_SCHEMA,
    FROZEN_ATTEMPT_INDICES,
    REVIEW_SHEET_SCHEMA,
    _slot_order,
)
from experiments.phase1.multireaction.ugi_all_role_semantic_visual_review import (
    RESULT_SCHEMA as PACKET_RESULT_SCHEMA,
)


def test_completed_blinded_review_is_unblinded_against_the_frozen_rule(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    packet_path = tmp_path / "packet_result.json"
    packet_path.write_text(
        json.dumps(
            {
                "schema_version": PACKET_RESULT_SCHEMA,
                "status": "ready_for_blinded_review",
                "quantitative_gate_passed": True,
                "fixed_attempt_indices": list(FROZEN_ATTEMPT_INDICES),
                "pairs": 24,
                "blinded": True,
                "candidate_selection": False,
                "invalid_or_failed_attempts_retained": True,
                "repair_or_retry": False,
            }
        )
    )
    key_rows = []
    review_rows = []
    for rank, attempt_index in enumerate(FROZEN_ATTEMPT_INDICES, start=1):
        pair_id = f"P{rank:02d}"
        order = _slot_order(2026090106, attempt_index)
        treatment_slot = "A" if order[0] == "all_role_semantic" else "B"
        key_rows.append(
            {
                "pair_id": pair_id,
                "attempt_index": attempt_index,
                "slot_A_arm": order[0],
                "slot_B_arm": order[1],
            }
        )
        review_rows.append(
            {
                "pair_id": pair_id,
                "tail_morphology_preference": treatment_slot if rank <= 20 else "tie",
                "head_tail_balance_preference": treatment_slot if rank <= 20 else "tie",
                "unsupported_ring_or_heteroatom_pathology": "neither",
                "reviewer_note": None,
            }
        )
    key_path = tmp_path / "blinding_key.json"
    key_path.write_text(
        json.dumps(
            {
                "schema_version": BLINDING_KEY_SCHEMA,
                "status": "sealed_until_review_complete",
                "rows": key_rows,
            }
        )
    )
    config = json.loads(
        (repo / "configs/multireaction/ugi_all_role_semantic_visual_review_seed0_v1.json").read_text()
    )
    sheet_path = tmp_path / "completed_review.json"
    sheet_path.write_text(
        json.dumps(
            {
                "schema_version": REVIEW_SHEET_SCHEMA,
                "status": "complete",
                "reviewer_id": "blinded-reviewer-1",
                "allowed_preferences": ["A", "B", "tie", "unassessable"],
                "allowed_pathology_labels": [
                    "A",
                    "B",
                    "both",
                    "neither",
                    "unassessable",
                ],
                "decision_rule": config["decision_rule"],
                "reviews": review_rows,
            }
        )
    )

    result = run_ugi_all_role_semantic_visual_adjudication(
        repo / "configs/multireaction/ugi_all_role_semantic_visual_review_seed0_v1.json",
        repo,
        packet_path,
        sheet_path,
        key_path,
        tmp_path / "result.json",
    )

    assert result["visual_decision"] == "pass_seed0_visual_gate"
    assert all(result["checks"].values())
    assert result["summaries"]["tail_morphology_preference"]["all_role_semantic"] == 20
    assert result["summaries"]["unsupported_ring_or_heteroatom_pathology"][
        "all_role_semantic"
    ] == 0
