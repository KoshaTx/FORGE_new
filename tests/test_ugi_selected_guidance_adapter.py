from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from experiments.phase1.product_l1.sampling.ugi_joint_sparse_sampling import (
    initialize_ugi_joint_sparse_state,
)
from experiments.phase1.product_l1.sampling.ugi_selected_restartable_generator import (
    SAMPLE_STEPS,
    SelectedRestartableGeneratorLane,
    build_selected_step1000_restartable_generator_lane,
)
from experiments.phase1.synthesis_guidance.adapters.selected_v1 import (
    SelectedGuidanceState,
    SelectedModelRestartableGuidanceLane,
    UgiSelectedGuidanceAdapterError,
)
from experiments.phase1.synthesis_guidance.adapters.terminal_support import (
    canonical_morphology_program_bytes,
    decode_canonical_morphology_program_bytes,
    native_completion_record_from_locked_terminal,
)

REPO = Path(__file__).resolve().parents[1]
SAMPLE_PATH = REPO / "results/phase1/ugi_architecture_selection_v3/full_step1000/result.json"


def _program_bytes(index: int = 0) -> bytes:
    value = json.loads(SAMPLE_PATH.read_text())
    return canonical_morphology_program_bytes(value["samples"][index]["program"])


def _manifest_sha256(seeds: tuple[int, ...]) -> str:
    return hashlib.sha256(
        json.dumps(seeds, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@pytest.fixture(scope="module")
def selected_lane() -> SelectedRestartableGeneratorLane:
    return build_selected_step1000_restartable_generator_lane(REPO)


@pytest.fixture(scope="module")
def guidance_lane(
    selected_lane: SelectedRestartableGeneratorLane,
) -> SelectedModelRestartableGuidanceLane:
    return SelectedModelRestartableGuidanceLane(selected_lane)


def _initialize_replicates(
    guidance_lane: SelectedModelRestartableGuidanceLane,
    *,
    seeds: tuple[int, ...] = (101, 202),
) -> SelectedGuidanceState:
    program = _program_bytes()
    receipt = guidance_lane.initialize(
        (program,) * len(seeds),
        seed=17,
        particle_seeds=seeds,
        device="cpu",
    )
    assert receipt.product_transition_calls == 0
    assert receipt.consumed_particle_seed_manifest_sha256 == _manifest_sha256(seeds)
    assert isinstance(receipt.state, SelectedGuidanceState)
    return receipt.state


def test_each_particle_consumes_its_own_batch_size_one_rng_stream(
    guidance_lane: SelectedModelRestartableGuidanceLane,
    selected_lane: SelectedRestartableGeneratorLane,
) -> None:
    seeds = tuple(range(1000, 1064))
    state = _initialize_replicates(guidance_lane, seeds=seeds)
    program = decode_canonical_morphology_program_bytes(_program_bytes())

    assert len({particle.rng_state_sha256 for particle in state.particles}) == 64
    for index in (0, 31, 63):
        particle = state.particles[index]
        seed = seeds[index]
        reference = initialize_ugi_joint_sparse_state(
            selected_lane.callback.model,
            (program,),
            selected_lane.callback.source_marginals,
            sample_steps=SAMPLE_STEPS,
            device="cpu",
            seed=seed,
        )
        assert torch.equal(particle.trajectory.generator_state, reference.generator_state)
        assert all(
            torch.equal(particle.trajectory.channels[key], reference.channels[key])
            for key in reference.channels
        )


def test_identity_ancestry_is_digest_preserving_and_snapshot_has_no_aliases(
    guidance_lane: SelectedModelRestartableGuidanceLane,
) -> None:
    state = _initialize_replicates(guidance_lane)
    identity = guidance_lane.apply_ancestry(state, (0, 1)).state

    assert identity.state_sha256 == state.state_sha256
    assert identity.categorical_state_sha256 == state.categorical_state_sha256
    assert identity.lineage_sha256 == state.lineage_sha256

    snapshot = guidance_lane.snapshot(state).state
    original = state.particles[0].trajectory.channels["nodes"]
    copied = snapshot.particles[0].trajectory.channels["nodes"]
    assert original.data_ptr() != copied.data_ptr()
    copied_before = copied.clone()
    original.add_(1)
    assert torch.equal(copied, copied_before)


def test_nonidentity_ancestry_copies_categories_but_retains_destination_rng(
    guidance_lane: SelectedModelRestartableGuidanceLane,
) -> None:
    initial = _initialize_replicates(guidance_lane)
    stepped = guidance_lane.advance(initial, target_step=2).state
    destination_rng_before = stepped.particles[1].rng_state_sha256
    resampled = guidance_lane.apply_ancestry(stepped, (0, 0)).state

    assert (
        resampled.particles[0].categorical_state_sha256
        == resampled.particles[1].categorical_state_sha256
    )
    assert (
        resampled.particles[0].trajectory.channels["nodes"].data_ptr()
        != resampled.particles[1].trajectory.channels["nodes"].data_ptr()
    )
    assert resampled.particles[1].rng_state_sha256 == destination_rng_before
    assert resampled.particles[0].rng_state_sha256 != resampled.particles[1].rng_state_sha256
    assert resampled.particles[0].lineage_sha256 != resampled.particles[1].lineage_sha256

    advanced = guidance_lane.advance(resampled, target_step=3).state
    assert advanced.particles[0].rng_state_sha256 != advanced.particles[1].rng_state_sha256
    assert any(
        not torch.equal(
            advanced.particles[0].trajectory.channels[key],
            advanced.particles[1].trajectory.channels[key],
        )
        for key in advanced.particles[0].trajectory.channels
    )


def test_ancestry_cannot_cross_frozen_morphology_programs(
    guidance_lane: SelectedModelRestartableGuidanceLane,
) -> None:
    receipt = guidance_lane.initialize(
        (_program_bytes(0), _program_bytes(1)),
        seed=19,
        particle_seeds=(303, 404),
        device="cpu",
    )

    with pytest.raises(
        UgiSelectedGuidanceAdapterError,
        match="cannot cross frozen morphology-program groups",
    ):
        guidance_lane.apply_ancestry(receipt.state, (1, 0))


def test_one_program_zero_guidance_is_exact_selected_callback_reference(
    guidance_lane: SelectedModelRestartableGuidanceLane,
    selected_lane: SelectedRestartableGeneratorLane,
) -> None:
    particle_seed = 20260803
    invocation_seed = 99991
    initial = guidance_lane.initialize(
        (_program_bytes(),),
        seed=23,
        particle_seeds=(particle_seed,),
        device="cpu",
    ).state
    final = guidance_lane.advance(initial, target_step=SAMPLE_STEPS).state
    request = guidance_lane.completion_generation_request(
        final,
        particle_index=0,
        invocation_seed=invocation_seed,
        checkpoint_index=SAMPLE_STEPS,
    )

    reference = selected_lane.callback(request)
    observed = guidance_lane.complete_terminal(
        final,
        particle_index=0,
        seed=invocation_seed,
        checkpoint_index=SAMPLE_STEPS,
    )

    assert observed.product_transition_calls == 0
    assert observed.error_detail is None
    assert observed.terminal is not None
    assert observed.terminal == reference
    assert observed.terminal.terminal_bytes == reference.terminal_bytes
    assert observed.terminal.generation_trace_bytes == reference.generation_trace_bytes
    assert native_completion_record_from_locked_terminal(
        observed.terminal
    ) == native_completion_record_from_locked_terminal(reference)
    for field in (
        "pool_state=",
        "pool_provenance=",
        "pool_lineage=",
        "particle_state=",
        "particle_provenance=",
        "particle_lineage=",
    ):
        assert field in observed.terminal.unit_id
