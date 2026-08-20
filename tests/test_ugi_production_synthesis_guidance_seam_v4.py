from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pytest

from experiments.archive.producers.phase1_qualify_ugi_production_synthesis_guidance_seam_v4 import (
    EXPECTED_FILENAMES,
    _atomic_output,
)
from experiments.phase1.synthesis_guidance.guidance.ugi_production_synthesis_guidance_seam_v4 import (
    EXPECTED_DESIGN,
    EXPECTED_INPUT_KEYS,
    EXPECTED_SCOPE,
    GUIDANCE_STRENGTH,
    RERUN_TOKEN,
    REVIEW_TOKEN,
    UgiProductionSynthesisGuidanceSeamV4Error,
    _build_run_artifact,
    _json_value,
    _validate_config,
    _validate_preregistration,
    run_production_synthesis_guidance_seam_v4,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/model/phase1_ugi_production_synthesis_guidance_seam_v4.json"
PREREG_V1 = REPO / "configs/model/phase1_ugi_matched_synthesis_guidance_preregistration_v1.json"
PREREG_V2 = REPO / "configs/model/phase1_ugi_matched_synthesis_guidance_preregistration_v2.json"


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True))


def test_v4_config_pins_one_nonzero_synthesis_only_assignment() -> None:
    config, paths, zero = _validate_config(REPO, CONFIG)

    assert set(paths) == EXPECTED_INPUT_KEYS
    assert config["scope"] == EXPECTED_SCOPE
    assert config["design"] == EXPECTED_DESIGN
    assert config["preregistration"] == {
        "guidance_strengths": [0.0, 0.25, 0.5, 1.0, 2.0],
        "qualification_strength": GUIDANCE_STRENGTH,
        "assignment_role": "first_preregistered_calibration_seed",
        "single_assignment_only": True,
        "no_retries": True,
        "no_seed_search": True,
        "not_a_calibration_winner_selection": True,
    }
    assert config["review_gate"]["execution_authorized_by_config_alone"] is False
    assert zero.result["status"] == "selected_v3_production_lambda_zero_typed_seam_qualified"
    assert zero.run["run"]["guidance_strength"] == 0.0


@pytest.mark.parametrize(
    ("field_path", "value", "match"),
    [
        (("scope", "production_execution"), True, "scope changed"),
        (("scope", "biological_guidance"), True, "scope changed"),
        (("design", "assignment_seed"), 20260822, "design changed"),
        (("preregistration", "qualification_strength"), 0.5, "declaration changed"),
        (("review_gate", "execution_authorized_by_config_alone"), True, "review gate"),
    ],
)
def test_v4_config_fails_closed_on_scope_seed_or_strength_changes(
    tmp_path: Path,
    field_path: tuple[str, str],
    value: object,
    match: str,
) -> None:
    changed = json.loads(CONFIG.read_text())
    changed[field_path[0]][field_path[1]] = value
    path = tmp_path / "changed.json"
    _write_json(path, changed)

    with pytest.raises(UgiProductionSynthesisGuidanceSeamV4Error, match=match):
        _validate_config(REPO, path)


def test_v4_config_rejects_a_changed_hardened_zero_artifact_pin(tmp_path: Path) -> None:
    changed = json.loads(CONFIG.read_text())
    changed["inputs"]["zero_seam_result"]["sha256"] = "0" * 64
    path = tmp_path / "changed-zero.json"
    _write_json(path, changed)

    with pytest.raises(UgiProductionSynthesisGuidanceSeamV4Error, match="hash changed"):
        _validate_config(REPO, path)


def test_v4_preregistration_rejects_seed_search(tmp_path: Path) -> None:
    v1 = json.loads(PREREG_V1.read_text())
    v1["design"]["no_seed_search"] = False
    changed_v1 = tmp_path / "phase1_ugi_matched_synthesis_guidance_preregistration_v1.json"
    _write_json(changed_v1, v1)

    config = json.loads(CONFIG.read_text())
    with pytest.raises(UgiProductionSynthesisGuidanceSeamV4Error, match="strength/seed contract"):
        _validate_preregistration(changed_v1, PREREG_V2, config)


def test_v4_public_entrypoint_requires_explicit_review_token(tmp_path: Path) -> None:
    with pytest.raises(UgiProductionSynthesisGuidanceSeamV4Error, match="review token"):
        run_production_synthesis_guidance_seam_v4(
            REPO,
            CONFIG,
            tmp_path / "cache",
            tmp_path / "output",
            review_token="not-reviewed",
            rerun_token=RERUN_TOKEN,
        )

    assert not (tmp_path / "cache").exists()
    assert not (tmp_path / "output").exists()
    assert REVIEW_TOKEN == "selected-v3-lambda-0.25-code-config-tests-reviewed"

    with pytest.raises(UgiProductionSynthesisGuidanceSeamV4Error, match="rerun token"):
        run_production_synthesis_guidance_seam_v4(
            REPO,
            CONFIG,
            tmp_path / "cache",
            tmp_path / "output",
            review_token=REVIEW_TOKEN,
            rerun_token="not-authorized",
        )
    assert RERUN_TOKEN == "selected-v3-lambda-0.25-single-operational-rerun-reviewed"


def test_v4_run_artifact_rejects_biological_or_production_scope() -> None:
    base = {
        "guidance_strength": GUIDANCE_STRENGTH,
        "production_execution": False,
        "biological_guidance": False,
        "private_holdout_accessed": False,
    }
    artifact = _build_run_artifact(base)
    assert artifact["scope"]["synthesis_success_probability"] is None

    for changed_field in ("production_execution", "biological_guidance"):
        changed = dict(base)
        changed[changed_field] = True
        with pytest.raises(UgiProductionSynthesisGuidanceSeamV4Error, match="scope changed"):
            _build_run_artifact(changed)


def test_v4_normalizes_dataclass_tuples_before_diagnostics_and_publication() -> None:
    @dataclass(frozen=True)
    class NestedRun:
        checkpoint_groups: tuple[tuple[int, ...], ...]

    normalized = _json_value(asdict(NestedRun(checkpoint_groups=((2, 4, 6),))))

    assert normalized == {"checkpoint_groups": [[2, 4, 6]]}
    assert isinstance(normalized["checkpoint_groups"], list)
    assert isinstance(normalized["checkpoint_groups"][0], list)


def test_v4_runner_persists_exactly_four_canonical_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "seam"
    values = {name: {"name": name} for name in EXPECTED_FILENAMES}

    _atomic_output(output, values)

    assert {path.name for path in output.iterdir()} == EXPECTED_FILENAMES
    assert all(path.read_bytes().endswith(b"\n") for path in output.iterdir())
    with pytest.raises(SystemExit, match="exactly"):
        _atomic_output(tmp_path / "bad", {"run.json": {}})
    with pytest.raises(SystemExit, match="overwrite"):
        _atomic_output(output, values)
