from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from experiments.archive.m0.flower_transfer_audit import (
    FlowerTransferAuditError,
    audit_flower_transfer_inputs,
    run_flower_transfer_audit,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO / "configs/verify/m0_10_flower_transfer_pilot.json"


@pytest.fixture
def isolated_audit_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "forge"
    config = json.loads(CONFIG_PATH.read_text())
    required_inputs = [*config["registry_inputs"], config["prospective_outcome_input"]]
    for specification in required_inputs:
        destination = repo / specification["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / specification["path"], destination)

    # Only optional evidence is synthetic; required scientific inputs retain their frozen bytes/pins.
    for specification in config["prior_local_evidence"]:
        payload = f"Optional evidence test fixture: {specification['label']}\n".encode()
        specification["sha256"] = hashlib.sha256(payload).hexdigest()
        destination = repo / specification["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)

    config_path = repo / CONFIG_PATH.relative_to(REPO)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config))
    return repo, config_path


def _actual_inputs() -> tuple[dict, list[dict], dict]:
    config = json.loads(CONFIG_PATH.read_text())
    registries = [
        json.loads((REPO / specification["path"]).read_text())
        for specification in config["registry_inputs"]
    ]
    prospective = json.loads((REPO / config["prospective_outcome_input"]["path"]).read_text())
    return config, registries, prospective


def test_registry_qualification_examples_are_not_mechanism_trajectories() -> None:
    config, registries, prospective = _actual_inputs()
    assets = {
        "source_code": {"present": False},
        "pretrained_checkpoint": {"present": False},
    }
    result = audit_flower_transfer_inputs(config, registries, prospective, assets, [])
    assert {row["qualification_positive_examples"] for row in result["target_classes"]} == {1}
    assert {row["qualification_negative_examples"] for row in result["target_classes"]} == {1}
    assert {row["elementary_mechanism_trajectories"] for row in result["target_classes"]} == {0}
    assert all(
        row["qualification_examples_are_flowER_trajectories"] is False
        for row in result["target_classes"]
    )
    assert result["pilot_readiness"]["class_curve_data_ready"] is False


def test_demotion_rule_applies_when_both_headline_conditions_are_unevaluable() -> None:
    config, registries, prospective = _actual_inputs()
    result = audit_flower_transfer_inputs(
        config,
        registries,
        prospective,
        {
            "source_code": {"present": False},
            "pretrained_checkpoint": {"present": False},
        },
        [],
    )
    assert result["prospective_outcomes"] == {
        "states_found": 82,
        "not_attempted": 82,
        "attempted_with_conversion_measurement": 0,
        "observed_attempted_states": [],
    }
    assert result["demotion_rule"]["headline_rule_satisfied"] is False
    assert result["demotion_rule"]["flowER_role"] == "orthogonal_consistency_check_only"
    assert result["demotion_rule"]["deterministic_atom_mapped_verifier_load_bearing"] is True
    assert result["scientific_interpretation"]["training_dependency"].startswith(
        "This blocker does not prevent training"
    )


def test_adding_transform_examples_does_not_fake_trajectory_readiness() -> None:
    config, registries, prospective = _actual_inputs()
    tampered = copy.deepcopy(registries)
    for registry in tampered:
        for reaction in registry["reactions"]:
            if reaction["reaction_id"] in config["target_classes"]:
                reaction["known_positive_examples"] *= 64
                reaction["known_negative_examples"] *= 64
    result = audit_flower_transfer_inputs(
        config,
        tampered,
        prospective,
        {
            "source_code": {"present": True},
            "pretrained_checkpoint": {"present": True},
        },
        [],
    )
    assert all(row["elementary_mechanism_trajectories"] == 0 for row in result["target_classes"])
    assert result["pilot_readiness"]["class_curve_data_ready"] is False
    assert (
        "minimum_32_elementary_mechanism_trajectories_per_class_absent"
        in result["pilot_readiness"]["blockers"]
    )


@pytest.mark.parametrize("optional_evidence_state", ["absent", "matching", "mismatching"])
def test_end_to_end_audit_is_hash_pinned_and_writes_result(
    isolated_audit_repo: tuple[Path, Path], optional_evidence_state: str
) -> None:
    repo, config_path = isolated_audit_repo
    config = json.loads(config_path.read_text())
    for specification in config["prior_local_evidence"]:
        path = repo / specification["path"]
        if optional_evidence_state == "absent":
            path.unlink()
        elif optional_evidence_state == "mismatching":
            path.write_bytes(b"Modified optional evidence test fixture\n")

    output = repo / "result.json"
    result = run_flower_transfer_audit(config_path, output, repo)
    assert json.loads(output.read_text()) == result
    assert result["status"] == "blocked_scientifically_valid_pilot_not_executed_flower_demoted"
    assert result["config"]["sha256"] == hashlib.sha256(config_path.read_bytes()).hexdigest()
    assert {row["sha256"] for row in result["inputs"]} == {
        "961dea191bbf6c8d169aab09f6aee18082e51696e4056fc4dc9f6fc97ff82587",
        "296bf06238ef22acc1f55117f5ce0adaee21b1bafaf5a83f89182b0f31cc4fcf",
        "ed52d1e5886e709f79a712883411f4995a34d51c5b8cc168f7ed5fc273b6090f",
    }
    assert len(result["prior_probe_evidence"]) == len(config["prior_local_evidence"])
    for row, specification in zip(
        result["prior_probe_evidence"], config["prior_local_evidence"], strict=True
    ):
        assert row["label"] == specification["label"]
        assert row["path"] == specification["path"]
        if optional_evidence_state == "absent":
            assert row["present"] is False
            assert row["expected_sha256"] == specification["sha256"]
            assert "hash_matches" not in row
            assert "sha256" not in row
        else:
            assert row["present"] is True
            assert row["hash_matches"] is (optional_evidence_state == "matching")
            assert row["sha256"] == hashlib.sha256((repo / row["path"]).read_bytes()).hexdigest()
    assert ("prior_probe_evidence_hash_mismatch" in result["pilot_readiness"]["blockers"]) is (
        optional_evidence_state == "mismatching"
    )
    assert result["pilot_readiness"]["fine_tuning_run_executed"] is False
    assert result["pilot_readiness"]["acceptance_criterion_satisfied"] is False
    assert result["demotion_rule"]["headline_rule_satisfied"] is False
    assert result["demotion_rule"]["flowER_role"] == "orthogonal_consistency_check_only"
    assert result["demotion_rule"]["deterministic_atom_mapped_verifier_load_bearing"] is True


def test_hash_mismatch_fails_closed(isolated_audit_repo: tuple[Path, Path]) -> None:
    repo, config_path = isolated_audit_repo
    config = json.loads(config_path.read_text())
    config["registry_inputs"][0]["sha256"] = "0" * 64
    tampered_config = repo / "config.json"
    tampered_config.write_text(json.dumps(config))
    output = repo / "result.json"
    with pytest.raises(FlowerTransferAuditError, match="hash mismatch"):
        run_flower_transfer_audit(tampered_config, output, repo)
    assert not output.exists()
