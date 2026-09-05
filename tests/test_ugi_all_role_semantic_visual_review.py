from __future__ import annotations

import copy
import json
import tarfile
from pathlib import Path

import pytest

from experiments.phase1.multireaction.ugi_all_role_semantic_visual_review import (
    BLINDING_KEY_SCHEMA,
    REVIEW_SHEET_SCHEMA,
    UgiAllRoleSemanticVisualReviewError,
    _validate_config,
    run_ugi_all_role_semantic_visual_review,
)
from forge.model.common_ugi_benchmark import CommonUgiAttempt, write_attempt_ledger


def _ledger(path: Path, method_id: str, smiles: str) -> None:
    write_attempt_ledger(
        path,
        [
            CommonUgiAttempt(
                method_id=method_id,
                seed=0,
                attempt_index=index,
                status="generated",
                product_smiles=smiles,
                method_visible_component_ids=(),
                generator_calls=1,
                reaction_calls=0,
                route_calls=0,
                oracle_calls=0,
                wall_seconds=0.0,
            )
            for index in range(3072)
        ],
    )


def test_visual_review_packet_is_blinded_fixed_index_and_nonselecting(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    baseline = tmp_path / "amine_semantic/assessment/attempts.jsonl.gz"
    treatment = tmp_path / "all_role_semantic/assessment/attempts.jsonl.gz"
    _ledger(
        baseline,
        "forge_seed0_amine_semantic_joint_support_ugi_program",
        "CCCCCCCCNCC(=O)OCCCCCCCC",
    )
    _ledger(
        treatment,
        "forge_seed0_all_role_semantic_joint_support_ugi_program",
        "CCCCCCCCCCNCC(=O)OCCCCCCCCCC",
    )
    archive_path = tmp_path / "comparison.tar"
    with tarfile.open(archive_path, "w") as archive:
        archive.add(baseline, arcname="amine_semantic/assessment/attempts.jsonl.gz")
        archive.add(treatment, arcname="all_role_semantic/assessment/attempts.jsonl.gz")
    comparison_path = tmp_path / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "schema_version": "forge.ugi_all_role_semantic_program_comparison_result.v1",
                "status": "complete",
                "profile": "full",
                "programs_per_method": 3072,
            }
        )
    )
    adjudication_path = tmp_path / "adjudication.json"
    adjudication_path.write_text(
        json.dumps(
            {
                "schema_version": "forge.ugi_all_role_semantic_adjudication.v1",
                "status": "complete",
                "quantitative_decision": "pass_seed0_quantitative_gate",
                "promotion_decision": "eligible_for_blinded_visual_review",
            }
        )
    )
    output = tmp_path / "output"
    result = run_ugi_all_role_semantic_visual_review(
        repo / "configs/multireaction/ugi_all_role_semantic_visual_review_seed0_v1.json",
        repo,
        comparison_path,
        archive_path,
        adjudication_path,
        output,
    )

    assert result["status"] == "ready_for_blinded_review"
    assert result["pairs"] == 24
    assert result["candidate_selection"] is False
    packet = (output / "review_packet.html").read_text()
    assert packet.count("<article>") == 24
    assert "amine_semantic" not in packet
    assert "all_role_semantic" not in packet
    sheet = json.loads((output / "review_sheet.json").read_text())
    assert sheet["schema_version"] == REVIEW_SHEET_SCHEMA
    assert len(sheet["reviews"]) == 24
    key = json.loads((output / "blinding_key.json").read_text())
    assert key["schema_version"] == BLINDING_KEY_SCHEMA
    assert len(key["rows"]) == 24


def test_visual_review_rejects_post_outcome_index_changes() -> None:
    repo = Path(__file__).resolve().parents[1]
    config = json.loads(
        (repo / "configs/multireaction/ugi_all_role_semantic_visual_review_seed0_v1.json").read_text()
    )
    changed = copy.deepcopy(config)
    changed["attempt_indices"][0] = 24

    with pytest.raises(UgiAllRoleSemanticVisualReviewError, match="fixed indices changed"):
        _validate_config(changed)


def test_visual_review_rejects_post_outcome_gate_changes() -> None:
    repo = Path(__file__).resolve().parents[1]
    config = json.loads(
        (repo / "configs/multireaction/ugi_all_role_semantic_visual_review_seed0_v1.json").read_text()
    )
    changed = copy.deepcopy(config)
    changed["decision_rule"]["minimum_net_treatment_preferences"] = 0

    with pytest.raises(UgiAllRoleSemanticVisualReviewError, match="decision rule changed"):
        _validate_config(changed)


def test_visual_review_accepts_frozen_complete_semantic_method_identity() -> None:
    repo = Path(__file__).resolve().parents[1]
    config = json.loads(
        (
            repo
            / "configs/multireaction/ugi_complete_semantic_visual_review_seed0_v1.json"
        ).read_text()
    )

    validated = _validate_config(config)

    assert validated["arms"]["all_role_semantic"]["method_id"] == (
        "forge_seed0_mog_substitution_semantic_joint_support_ugi_program"
    )
