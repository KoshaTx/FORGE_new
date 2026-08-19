from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from forge.product.ugi_v3_shadow_gate import (
    UgiV3ShadowGateError,
    evaluate_v3_shadow_gate,
)


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, (dict, list)):
        path.write_text(json.dumps(value, sort_keys=True))
    else:
        path.write_text(str(value))
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _spec(path: Path, repository: Path) -> dict[str, str]:
    return {"path": str(path.relative_to(repository)), "sha256": _sha(path)}


def _fixture(repository: Path) -> dict[str, object]:
    policy = _write(repository / "policy.json", {"schema_version": "v3"})
    selector = _write(repository / "selector.py", "# frozen selector")
    target_checkpoint = _write(repository / "target.pt", "target")
    other_checkpoint = _write(repository / "other.pt", "other")
    closure = _write(repository / "closure.pt", "closure")
    program = _write(repository / "program.json", {"seed": 7})
    baseline_target_sample = _write(repository / "baseline_target.json", {"sample": "base"})
    baseline_other_sample = _write(repository / "baseline_other.json", {"sample": "other"})
    training_full = _write(repository / "training_full.json", {"training": "full"})
    training_size = _write(repository / "training_size.json", {"training": "size"})

    target_id = "full_morphology_program_conditioning:step_1000"
    other_id = "size_only_conditioning:step_1000"
    target_candidate = {
        "candidate_id": target_id,
        "architecture": "full_morphology_program_conditioning",
        "step": 1000,
        "sampling_result": _spec(baseline_target_sample, repository),
        "checkpoint": _spec(target_checkpoint, repository),
        "usable_open_ended_yield": 0.5,
        "gate": {"status": "fail", "failed_checks": ["exact_forward"]},
    }
    other_candidate = {
        "candidate_id": other_id,
        "architecture": "size_only_conditioning",
        "step": 1000,
        "sampling_result": _spec(baseline_other_sample, repository),
        "checkpoint": _spec(other_checkpoint, repository),
        "usable_open_ended_yield": 0.4,
        "gate": {"status": "pass", "failed_checks": []},
    }
    original = {
        "policy": {"sha256": _sha(policy)},
        "candidates_by_architecture": {
            "full_morphology_program_conditioning": [target_candidate],
            "size_only_conditioning": [other_candidate],
        },
        "decision": {"selected_candidate": other_id},
    }
    original_path = _write(repository / "original.json", original)
    production = {
        "selection": {
            "result": _spec(original_path, repository),
            "policy": _spec(policy, repository),
        }
    }
    production_path = _write(repository / "production.json", production)

    challenger_config = {
        "sampling": {"seed": 11, "attempted_draws": 1, "sample_steps": 8},
        "inputs": {
            "joint_checkpoint": _spec(target_checkpoint, repository),
            "closure_checkpoint": _spec(closure, repository),
            "program_draw": _spec(program, repository),
        },
    }
    challenger_config_path = _write(repository / "challenger_config.json", challenger_config)
    challenger_result = {
        "seed": 11,
        "samples": [{}],
        "sampling": {"sample_steps": 8, "conditioning_mode": "full_morphology"},
        "checkpoints": {
            "joint": str(target_checkpoint.relative_to(repository)),
            "closure": str(closure.relative_to(repository)),
        },
        "matched_staged_result": str(program.relative_to(repository)),
    }
    challenger_result_path = _write(repository / "challenger_result.json", challenger_result)
    challenger_candidate = copy.deepcopy(target_candidate)
    challenger_candidate["sampling_result"] = _spec(challenger_result_path, repository)
    challenger_candidate["usable_open_ended_yield"] = 0.8
    challenger_candidate["gate"] = {"status": "pass", "failed_checks": []}
    shadow = {
        "policy": {"sha256": _sha(policy)},
        "candidates_by_architecture": {
            "full_morphology_program_conditioning": [challenger_candidate],
            "size_only_conditioning": [other_candidate],
        },
        "decision": {"selected_candidate": target_id},
    }

    contract = {
        "schema_version": "phase1_ugi_v3_nonselecting_shadow_gate_config.v1",
        "status": "frozen_baseline_contract",
        "expected_unchanged_candidates": 1,
        "inputs": {
            "policy": _spec(policy, repository),
            "selector": _spec(selector, repository),
            "original_v3_result": _spec(original_path, repository),
            "production_manifest": _spec(production_path, repository),
        },
        "training_results": {
            "full_morphology_program_conditioning": _spec(training_full, repository),
            "size_only_conditioning": _spec(training_size, repository),
        },
        "baseline_arm_samples": {
            "full_morphology_program_conditioning": {
                "1000": _spec(baseline_target_sample, repository)
            },
            "size_only_conditioning": {"1000": _spec(baseline_other_sample, repository)},
        },
        "output_contract": {
            "status": "diagnostic_only_cannot_replace_production",
            "retain_shadow_decision": False,
            "retain_only_challenger_and_same_step_baseline": True,
            "verify_all_unchanged_metric_gate_objects_exactly": True,
        },
    }
    contract_path = _write(repository / "contract.json", contract)
    return {
        "contract": contract_path,
        "challenger_config": challenger_config_path,
        "challenger_result": challenger_result_path,
        "shadow": shadow,
        "target_id": target_id,
    }


def _evaluate(
    repository: Path,
    fixture: dict[str, object],
    shadow: dict[str, object],
) -> dict[str, object]:
    def runner(command: list[str], cwd: Path, output: Path) -> None:
        assert cwd == repository
        arm_samples = [
            command[index + 1] for index, value in enumerate(command) if value == "--arm-sample"
        ]
        assert any(
            str(fixture["challenger_result"]) in value
            and value.startswith("full_morphology_program_conditioning:1000:")
            for value in arm_samples
        )
        _write(output, shadow)

    challenger_config = Path(fixture["challenger_config"])
    challenger_result = Path(fixture["challenger_result"])
    return evaluate_v3_shadow_gate(
        contract_path=Path(fixture["contract"]),
        challenger_config_path=challenger_config,
        challenger_config_sha256=_sha(challenger_config),
        challenger_result_path=challenger_result,
        challenger_result_sha256=_sha(challenger_result),
        output_path=repository / "output.json",
        repository=repository,
        selector_runner=runner,
    )


def test_shadow_gate_retains_only_same_step_comparison_and_discards_decision(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    result = _evaluate(tmp_path, fixture, fixture["shadow"])
    assert result["status"] == "diagnostic_only_cannot_replace_production"
    assert result["target_candidate_id"] == fixture["target_id"]
    assert result["unchanged_candidate_reproduction"]["exactly_reproduced_candidates"] == 1
    assert result["same_step_comparison"]["original_v3_baseline"]["gate"]["status"] == "fail"
    assert result["same_step_comparison"]["challenger"]["gate"]["status"] == "pass"
    assert result["shadow_decision"]["retained"] is False
    assert result["shadow_decision"]["usable"] is False
    assert "candidates_by_architecture" not in result


def test_shadow_gate_rejects_any_change_to_the_other_candidates(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    shadow = copy.deepcopy(fixture["shadow"])
    shadow["candidates_by_architecture"]["size_only_conditioning"][0]["gate"] = {
        "status": "fail",
        "failed_checks": ["changed"],
    }
    with pytest.raises(UgiV3ShadowGateError, match="failed exact reproduction"):
        _evaluate(tmp_path, fixture, shadow)


def test_shadow_gate_requires_the_caller_pinned_challenger_hash(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    with pytest.raises(UgiV3ShadowGateError, match="challenger result hash mismatch"):
        evaluate_v3_shadow_gate(
            contract_path=Path(fixture["contract"]),
            challenger_config_path=Path(fixture["challenger_config"]),
            challenger_config_sha256=_sha(Path(fixture["challenger_config"])),
            challenger_result_path=Path(fixture["challenger_result"]),
            challenger_result_sha256="0" * 64,
            output_path=tmp_path / "output.json",
            repository=tmp_path,
        )
