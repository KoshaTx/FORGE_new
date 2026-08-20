from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge.design.schedule.ugi_nonzero_guidance_runner import (
    load_grouped_smc_schedule_qualification,
)
from forge.design.guidance.ugi_production_zero_guidance_seam_v2 import (
    UgiProductionZeroGuidanceSeamV2Error,
    _validate_config,
)
from forge.design.flow.ugi_selected_guidance_adapter_v2 import (
    build_selected_model_restartable_guidance_lane_v2,
)
from forge.design.sampling.ugi_selected_restartable_generator_v2 import (
    GENERATOR_CHECKPOINT_SHA256,
    MAXIMUM_ADJACENT_BRANCH_RUNS,
    TERMINAL_DECODER_ID,
)


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/model/phase1_ugi_production_zero_guidance_seam_v2.json"
SCHEDULE = REPO / "results/phase1/ugi_grouped_smc_schedule_qualification_v1/result.json"


def test_v2_lane_binds_current_checkpoint_decoder_and_branch_policy() -> None:
    lane = build_selected_model_restartable_guidance_lane_v2(REPO)

    assert lane.selected_lane.bindings.generator_checkpoint_sha256 == (
        GENERATOR_CHECKPOINT_SHA256
    )
    assert lane.selected_lane.bindings.terminal_decoder_id == TERMINAL_DECODER_ID
    assert lane.selected_lane.bindings.maximum_adjacent_branch_runs == (
        MAXIMUM_ADJACENT_BRANCH_RUNS
    )


def test_v2_restartable_final_matches_direct_independent_stream_callback() -> None:
    assignment = load_grouped_smc_schedule_qualification(SCHEDULE).assignments[0]
    lane = build_selected_model_restartable_guidance_lane_v2(REPO)
    particle_seed = assignment.stochastic_particle_seeds[0]
    state = lane.initialize(
        (assignment.programs[0],),
        seed=assignment.seed,
        particle_seeds=(particle_seed,),
        device="cpu",
    ).state
    state = lane.advance(state, target_step=8).state
    invocation_seed = particle_seed + 9173
    completion = lane.complete_terminal(
        state,
        particle_index=0,
        seed=invocation_seed,
        checkpoint_index=8,
    )
    assert completion.terminal is not None, completion.error_detail
    request = lane.completion_generation_request(
        state,
        particle_index=0,
        invocation_seed=invocation_seed,
        checkpoint_index=8,
    )
    reference = lane.selected_lane.callback(request)
    assert completion.terminal == reference


def test_v2_seam_config_is_hash_pinned_and_fails_closed(tmp_path: Path) -> None:
    config, paths = _validate_config(REPO, CONFIG)
    assert config["scope"]["guidance_strength"] == 0.0
    assert len(paths) == 12

    changed = json.loads(CONFIG.read_text())
    changed["scope"]["guidance_strength"] = 0.1
    changed_path = tmp_path / "changed.json"
    changed_path.write_text(json.dumps(changed))
    with pytest.raises(UgiProductionZeroGuidanceSeamV2Error, match="scope changed"):
        _validate_config(REPO, changed_path)
