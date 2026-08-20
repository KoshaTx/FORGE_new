from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import experiments.phase1.synthesis_guidance.guidance.ugi_production_zero_guidance_seam_v3 as seam_v3
from experiments.archive.producers.phase1_qualify_ugi_production_zero_guidance_seam_v3 import (
    EXPECTED_FILENAMES,
    _atomic_output,
)
from experiments.phase1.synthesis_guidance.adapters.selected_v3 import (
    build_selected_model_restartable_guidance_lane_v3,
)
from experiments.phase1.synthesis_guidance.guidance.ugi_production_zero_guidance_seam_v3 import (
    EXPECTED_INPUT_KEYS,
    ProductionGuidanceRouteEvaluatorV3,
    UgiProductionZeroGuidanceSeamV3Error,
    _annotate_support_record_coordinates,
    _require_selected_v3_productive_identity,
    _validate_config,
)
from experiments.phase1.synthesis_guidance.schedule.ugi_nonzero_guidance_runner import (
    GuidanceAssessmentContext,
)
from forge.synthesis.matched import RouteComputeUsage

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/model/phase1_ugi_production_zero_guidance_seam_v3.json"
HISTORICAL_RUN = REPO / "results/phase1/ugi_production_zero_guidance_seam_v2/run.json"


def test_v3_seam_config_pins_complete_adapter_and_audit_chain(tmp_path: Path) -> None:
    config, paths = _validate_config(REPO, CONFIG)

    assert set(paths) == EXPECTED_INPUT_KEYS
    assert config["scope"]["production_execution"] is False
    assert config["scope"]["qualification_only"] is True
    assert config["selected_v3_identities"]["adapter_identity_sha256"]

    changed = json.loads(CONFIG.read_text())
    changed["scope"]["production_execution"] = True
    changed_path = tmp_path / "changed.json"
    changed_path.write_text(json.dumps(changed))
    with pytest.raises(UgiProductionZeroGuidanceSeamV3Error, match="scope changed"):
        _validate_config(REPO, changed_path)


def test_selected_v3_direct_reference_is_an_explicit_third_identity() -> None:
    lane = build_selected_model_restartable_guidance_lane_v3(REPO)
    historical = json.loads(HISTORICAL_RUN.read_text())["run"]
    for arm_name in ("guided", "post_hoc"):
        for admission in historical[arm_name]["productive_admissions"]:
            fields = admission["terminal_id"].split(":")
            assert len(fields) == 12
            for field_index in (4, 5):
                key, digest = fields[field_index].split("=", maxsplit=1)
                replacement = "0" if digest[0] != "0" else "1"
                fields[field_index] = f"{key}={replacement}{digest[1:]}"
            fields[10] = f"adapter={lane.adapter_identity_sha256}"
            admission["terminal_id"] = ":".join(fields)
            old_trace = admission["generation_trace_sha256"]
            replacement = "0" if old_trace[0] != "0" else "1"
            admission["generation_trace_sha256"] = replacement + old_trace[1:]

    manifest = _require_selected_v3_productive_identity(historical, lane)

    assert len(manifest) == 64
    changed = copy.deepcopy(historical)
    changed["post_hoc"]["productive_admissions"][0]["terminal_sha256"] = "0" * 64
    with pytest.raises(UgiProductionZeroGuidanceSeamV3Error, match="direct reference"):
        _require_selected_v3_productive_identity(changed, lane)

    unchanged_trace = copy.deepcopy(historical)
    direct = seam_v3._selected_v3_reference_rows(lane)
    unchanged_trace["guided"]["productive_admissions"][0]["generation_trace_sha256"] = direct[0][
        "generation_trace_sha256"
    ]
    with pytest.raises(UgiProductionZeroGuidanceSeamV3Error, match="direct reference"):
        _require_selected_v3_productive_identity(unchanged_trace, lane)

    malformed_id = copy.deepcopy(historical)
    malformed_id["guided"]["productive_admissions"][0]["terminal_id"] = (
        "forged-prefix:" + malformed_id["guided"]["productive_admissions"][0]["terminal_id"]
    )
    with pytest.raises(UgiProductionZeroGuidanceSeamV3Error, match="terminal ID"):
        _require_selected_v3_productive_identity(malformed_id, lane)

    wrong_particle = copy.deepcopy(historical)
    wrong_particle["guided"]["productive_admissions"][0]["terminal_id"] = wrong_particle["guided"][
        "productive_admissions"
    ][0]["terminal_id"].replace(":p0:", ":p63:", 1)
    with pytest.raises(UgiProductionZeroGuidanceSeamV3Error, match="direct reference"):
        _require_selected_v3_productive_identity(wrong_particle, lane)

    arm_trace_mismatch = copy.deepcopy(historical)
    guided_trace = arm_trace_mismatch["guided"]["productive_admissions"][0][
        "generation_trace_sha256"
    ]
    replacement = "0" if guided_trace[0] != "0" else "1"
    arm_trace_mismatch["post_hoc"]["productive_admissions"][0]["generation_trace_sha256"] = (
        replacement + guided_trace[1:]
    )
    with pytest.raises(UgiProductionZeroGuidanceSeamV3Error, match="lambda zero"):
        _require_selected_v3_productive_identity(arm_trace_mismatch, lane)


def test_v3_evaluator_retains_support_potential_bridge_and_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digest = "a" * 64
    usage = RouteComputeUsage(logical_planner_calls=3, logical_verifier_calls=13)
    support_audit = SimpleNamespace(
        to_dict=lambda: {
            "schema_version": "support.v1",
            "terminal_sha256": digest,
            "scalar_value": None,
            "success_probability": None,
        }
    )
    planner = SimpleNamespace(support_audit=support_audit)
    binding = SimpleNamespace(bind=lambda _context: object())
    factory = SimpleNamespace(
        planner_context=object(),
        l1_reverifier=object(),
        assessment_as_of_utc="2026-08-03T17:30:13Z",
        build_planner=lambda *_args: planner,
    )
    receipt_dict = {
        "schema_version": "receipt.v1",
        "assessment_sha256": digest,
        "product_value": {"complete": False},
        "scalar_value": None,
        "success_probability": None,
    }
    receipt = SimpleNamespace(
        product_value=object(),
        terminal_id="terminal-1",
        terminal_sha256=digest,
        generation_trace_sha256="b" * 64,
        morphology_program_sha256="c" * 64,
        assessment_sha256=digest,
        realized_route_usage=usage,
        to_dict=lambda: receipt_dict,
    )
    potential = SimpleNamespace(
        to_dict=lambda: {
            "route_completion_utility": 0.0,
            "reasons": ["missing_route_knowledge"],
            "strict_complete_role_count": 0,
        }
    )
    bridge = SimpleNamespace(
        route_completion_utility=0.0,
        censored=False,
        support_bonus=False,
        to_dict=lambda: {
            "route_completion_utility": 0.0,
            "censored": False,
            "support_bonus": False,
            "success_probability": None,
        },
    )
    monkeypatch.setattr(seam_v3, "assess_locked_ugi_terminal_routes", lambda *_args, **_kw: receipt)
    monkeypatch.setattr(
        seam_v3,
        "exact_closure_potential_from_product_value",
        lambda _value: potential,
    )
    monkeypatch.setattr(
        seam_v3,
        "smc_utility_bridge_from_exact_closure",
        lambda _value: bridge,
    )
    evaluator = ProductionGuidanceRouteEvaluatorV3(binding=binding, factory=factory)
    context = GuidanceAssessmentContext(
        treatment_arm="guided",
        assessment_phase="checkpoint_shadow",
        checkpoint=2,
        route_seed=7,
        reservation=usage,
        base_snapshot_sha256="d" * 64,
        planner_context_sha256="e" * 64,
        cache_preflight_sha256="f" * 64,
        cache_clone_id="guided-clone",
        post_hoc_productive_lock_sha256=None,
    )

    evaluation = evaluator(object(), context)

    assert evaluation.route_completion_utility == 0.0
    assert len(evaluator.support_records) == 1
    record = evaluator.support_records[0]
    assert record["ordinal"] == 0
    assert record["support_audit"] == support_audit.to_dict()
    assert record["assessment_receipt"] == receipt_dict
    assert record["potential"]["true_planner_censor"] is False
    assert record["utility_bridge"] == bridge.to_dict()
    assert record["scalar_value"] is None
    assert record["success_probability"] is None


def test_support_records_receive_exact_runner_owned_coordinates() -> None:
    run: dict[str, object] = {}
    records: list[dict[str, object]] = []
    for arm_index, arm_name in enumerate(("guided", "post_hoc")):
        checkpoint_receipt = chr(ord("a") + arm_index * 2) * 64
        productive_receipt = chr(ord("b") + arm_index * 2) * 64
        run[arm_name] = {
            "checkpoint_groups": [
                {
                    "checkpoint": 2,
                    "program_index": 1,
                    "global_particle_indices": [4],
                    "source_assessment_receipt_sha256s": [checkpoint_receipt],
                }
            ],
            "productive_assessments": [
                {
                    "disposition": "assessed_representative",
                    "source_assessment_receipt_sha256": productive_receipt,
                }
            ],
        }
        records.extend(
            [
                {
                    "arm": arm_name,
                    "assessment_phase": "checkpoint_shadow",
                    "assessment_receipt_sha256": checkpoint_receipt,
                    "checkpoint": 2,
                },
                {
                    "arm": arm_name,
                    "assessment_phase": "productive_final",
                    "assessment_receipt_sha256": productive_receipt,
                    "checkpoint": 8,
                },
            ]
        )

    _annotate_support_record_coordinates(run, records)

    for arm_index in range(2):
        checkpoint = records[arm_index * 2]
        productive = records[arm_index * 2 + 1]
        assert checkpoint["program_index"] == 1
        assert checkpoint["particle_index"] == 4
        assert checkpoint["productive_index"] is None
        assert productive["program_index"] == 0
        assert productive["particle_index"] == 0
        assert productive["productive_index"] == 0

    with pytest.raises(UgiProductionZeroGuidanceSeamV3Error, match="exactly once"):
        _annotate_support_record_coordinates(run, records[:-1])


def test_runner_persists_exactly_three_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "seam"
    values = {name: {"name": name} for name in EXPECTED_FILENAMES}

    _atomic_output(output, values)

    assert {path.name for path in output.iterdir()} == EXPECTED_FILENAMES
    with pytest.raises(SystemExit, match="exactly"):
        _atomic_output(tmp_path / "bad", {"run.json": {}})
