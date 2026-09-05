from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.phase1.multireaction.ugi_development_visual_adjudication import (
    UgiDevelopmentVisualAdjudicationError,
    run_ugi_development_visual_adjudication,
)


def _write(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value))
    return path


def _inputs(
    tmp_path: Path,
    *,
    treatment_arm: str = "successor",
    whole_head: bool = False,
) -> tuple[Path, Path, Path]:
    preference_criteria = (
        ["head_arrangement_preference", "overall_lipid_plausibility_preference"]
        if whole_head
        else ["tail_morphology_preference", "head_tail_balance_preference"]
    )
    packet = _write(
        tmp_path / "packet.json",
        {
            "schema_version": "forge.ugi_development_visual_review.v1",
            "status": "ready_for_blinded_review",
            "baseline_arm": "reference",
            "treatment_arm": treatment_arm,
            "pairs": 1,
            "blinded": True,
            "candidate_selection": False,
            "repair_or_retry": False,
            **(
                {
                    "preference_criteria": preference_criteria,
                    "pathology_criterion": "unsupported_ring_or_heteroatom_pathology",
                }
                if whole_head
                else {}
            ),
        },
    )
    sheet = _write(
        tmp_path / "sheet.json",
        {
            "schema_version": "forge.ugi_development_blinded_review_sheet.v1",
            "status": "complete",
            "reviewer_id": "reviewer-1",
            **(
                {
                    "preference_criteria": preference_criteria,
                    "pathology_criterion": "unsupported_ring_or_heteroatom_pathology",
                }
                if whole_head
                else {}
            ),
            "decision_rule": {
                "maximum_treatment_pathology_pairs": 0,
                "minimum_assessable_pairs_per_criterion": 1,
                "minimum_net_treatment_preferences": 1,
            },
            "reviews": [
                {
                    "pair_id": "P01",
                    **{criterion: "B" for criterion in preference_criteria},
                    "unsupported_ring_or_heteroatom_pathology": "neither",
                    "reviewer_note": None,
                }
            ],
        },
    )
    key = _write(
        tmp_path / "key.json",
        {
            "schema_version": "forge.ugi_development_blinding_key.v1",
            "status": "sealed_until_review_complete",
            "rows": [
                {
                    "pair_id": "P01",
                    "attempt_index": 7,
                    "slot_A_arm": "reference",
                    "slot_B_arm": "successor",
                }
            ],
        },
    )
    return packet, sheet, key


def test_adjudicator_uses_explicit_arm_identities_without_name_heuristics(tmp_path: Path) -> None:
    packet, sheet, key = _inputs(tmp_path)

    result = run_ugi_development_visual_adjudication(
        packet,
        sheet,
        key,
        tmp_path / "result.json",
    )

    assert result["baseline_arm"] == "reference"
    assert result["treatment_arm"] == "successor"
    assert result["visual_decision"] == "pass_seed0_visual_gate"


def test_adjudicator_rejects_packet_arm_identity_mismatch(tmp_path: Path) -> None:
    packet, sheet, key = _inputs(tmp_path, treatment_arm="wrong-successor")

    with pytest.raises(UgiDevelopmentVisualAdjudicationError, match="disagree"):
        run_ugi_development_visual_adjudication(
            packet,
            sheet,
            key,
            tmp_path / "result.json",
        )


def test_adjudicator_supports_whole_head_visual_criteria(tmp_path: Path) -> None:
    packet, sheet, key = _inputs(tmp_path, whole_head=True)

    result = run_ugi_development_visual_adjudication(
        packet,
        sheet,
        key,
        tmp_path / "result.json",
    )

    assert result["preference_criteria"] == [
        "head_arrangement_preference",
        "overall_lipid_plausibility_preference",
    ]
    assert result["visual_decision"] == "pass_seed0_visual_gate"
