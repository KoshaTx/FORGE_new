from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from forge.potency.ugi_hela_potency_diagnostic import HeLaPotencyEvaluation
from forge.product.ugi_hela_potency_guidance_seam_v1 import (
    run_lambda_zero_identity_gate,
    run_potency_seed,
)
from forge.product.ugi_matched_budget_orchestration import LockedMatchedTerminal
from forge.product.ugi_nonzero_guidance_runner import (
    FrozenSeedProgramAssignment,
    GuidanceSchedule,
    GuidanceStateReceipt,
    GuidanceTerminalCompletionReceipt,
    ParticleGroupDesign,
)

FAKE_SHA = "a" * 64


def _payload_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _assignment(seed: int = 20260821) -> FrozenSeedProgramAssignment:
    programs = (b"program-a", b"program-b")
    return FrozenSeedProgramAssignment(
        seed=seed,
        schedule_sha256=hashlib.sha256(f"schedule:{seed}".encode()).hexdigest(),
        programs=programs,
        program_sha256s=tuple(hashlib.sha256(value).hexdigest() for value in programs),
        stochastic_particle_seeds=tuple(seed * 100 + index for index in range(8)),
    )


def _design() -> ParticleGroupDesign:
    return ParticleGroupDesign(morphology_program_count=2, particles_per_program=4)


def _schedule() -> GuidanceSchedule:
    return GuidanceSchedule(
        sample_steps=8,
        checkpoints=(2, 4, 6),
        checkpoint_betas=(0.25, 0.5, 0.75),
        rollouts_per_particle_checkpoint=1,
        final_selection_count=4,
    )


@dataclass(frozen=True)
class _State:
    programs: tuple[bytes, ...]
    tokens: tuple[int, ...]
    step: int


class _Lane:
    def initialize(
        self,
        programs: tuple[bytes, ...],
        *,
        seed: int,
        particle_seeds: tuple[int, ...],
        device: str,
    ) -> GuidanceStateReceipt:
        del seed, device
        return GuidanceStateReceipt(
            state=_State(programs, tuple(range(len(programs))), 0),
            product_transition_calls=0,
            consumed_particle_seed_manifest_sha256=_payload_sha(particle_seeds),
        )

    def advance(self, state: _State, *, target_step: int) -> GuidanceStateReceipt:
        return GuidanceStateReceipt(
            state=_State(state.programs, state.tokens, target_step),
            product_transition_calls=len(state.programs) * (target_step - state.step),
        )

    def snapshot(self, state: _State) -> GuidanceStateReceipt:
        return GuidanceStateReceipt(state=state, product_transition_calls=0)

    def complete_terminal(
        self,
        state: _State,
        *,
        particle_index: int,
        seed: int,
        checkpoint_index: int,
    ) -> GuidanceTerminalCompletionReceipt:
        token = state.tokens[particle_index]
        terminal_bytes = f"terminal:{token}:{checkpoint_index}:{seed}".encode()
        trace_bytes = f"trace:{token}:{checkpoint_index}:{seed}".encode()
        terminal = LockedMatchedTerminal(
            unit_id=f"unit-{checkpoint_index}-{particle_index}",
            morphology_program_sha256=hashlib.sha256(state.programs[particle_index]).hexdigest(),
            checkpoint_index=checkpoint_index,
            generator_checkpoint_sha256=FAKE_SHA,
            closure_checkpoint_sha256="b" * 64,
            terminal_id=f"terminal-{token}-{checkpoint_index}-{seed}",
            terminal_locked=True,
            terminal_valid=True,
            exact_l1=True,
            terminal_bytes=terminal_bytes,
            generation_trace_bytes=trace_bytes,
            payload={"token": token},
        )
        return GuidanceTerminalCompletionReceipt(
            terminal=terminal,
            product_transition_calls=8 - state.step,
        )

    def apply_ancestry(
        self,
        state: _State,
        ancestors: tuple[int, ...],
    ) -> GuidanceStateReceipt:
        for destination, source in enumerate(ancestors):
            assert state.programs[destination] == state.programs[source]
        return GuidanceStateReceipt(
            state=_State(
                state.programs,
                tuple(state.tokens[index] for index in ancestors),
                state.step,
            ),
            product_transition_calls=0,
        )


class _Policy:
    def evaluate_batch(self, terminals, _predictor):
        output = []
        for terminal in terminals:
            token = int(terminal.payload["token"])
            output.append(
                HeLaPotencyEvaluation(
                    terminal_id=terminal.terminal_id,
                    terminal_sha256=terminal.terminal_sha256,
                    action="potency_guidance",
                    reason="test",
                    potential=(token % 4) / 3.0,
                    exact_forward_verified=True,
                    exact_measured_combination=False,
                    unseen_roles=("amine",),
                    pattern_id="amine_only",
                    distribution_bins=(("product", "interpolative"),),
                    ensemble_mean=5.0,
                    ensemble_standard_deviation=0.1,
                    conformal_q90=1.0,
                    lcb90=4.0,
                    calibration_ecdf=0.75,
                    worker_receipt_sha256="c" * 64,
                )
            )
        return tuple(output)


def test_lambda_zero_gate_is_exact_and_uses_full_schedule() -> None:
    result = run_lambda_zero_identity_gate(
        _assignment(),
        lane=_Lane(),
        design=_design(),
        schedule=_schedule(),
        device="cpu",
    )

    assert result.bitwise_identity_passed is True
    assert result.product_transition_calls_per_arm == 160
    assert result.terminal_completions_per_arm == 32
    assert len(result.checkpoint_manifests) == 3


def test_potency_seed_matches_compute_and_replays_nonuniform_ancestry() -> None:
    result = run_potency_seed(
        _assignment(),
        lane=_Lane(),
        policy=_Policy(),  # type: ignore[arg-type]
        predictor=object(),  # type: ignore[arg-type]
        design=_design(),
        schedule=_schedule(),
        guidance_strength=0.25,
        device="cpu",
    )

    assert result.product_transition_calls_per_arm == 160
    assert result.terminal_completions_per_arm == 32
    assert result.scheduled_potency_attempts_per_arm == 32
    assert result.synthesis_calls == 0
    assert result.proposal_calls == 0
    assert all(record.replay_verified for record in result.guided_checkpoint_records)
    assert any(record.nonuniform_program_count > 0 for record in result.guided_checkpoint_records)
