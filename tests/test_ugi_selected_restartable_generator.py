from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from forge.product.ugi_matched_budget_orchestration import (
    MatchedArm,
    MatchedGenerationRequest,
    MatchedScheduleEntry,
    RouteComputeUsage,
)
from forge.product.ugi_restartable_terminal_support_adapter import (
    UgiRestartableTerminalSupportAdapterError,
    canonical_morphology_program_bytes,
    decode_canonical_morphology_program_bytes,
    lock_unqualified_restartable_completion_row,
    native_completion_record_from_locked_terminal,
)
from forge.product.ugi_selected_generator_implementation import (
    build_selected_generator_implementation_qualification,
)
from forge.product.ugi_selected_restartable_generator import (
    CLOSURE_CHECKPOINT_SHA256,
    GENERATOR_CHECKPOINT_SHA256,
    SAMPLE_STEPS,
    SelectedRestartableGeneratorBindings,
    SelectedRestartableGeneratorCallback,
    SelectedRestartableGeneratorLane,
    UgiSelectedRestartableGeneratorError,
    build_selected_step1000_restartable_generator_lane,
)
from forge.product.ugi_terminal_route_assessment import (
    UgiTerminalRouteAssessmentError,
    ValidatedUgiTerminalPayload,
)

REPO = Path(__file__).resolve().parents[1]
SAMPLE_PATH = REPO / "results/phase1/ugi_architecture_selection_v3/full_step1000/result.json"


def _real_program() -> dict:
    return json.loads(SAMPLE_PATH.read_text())["samples"][0]["program"]


def _request(
    *,
    arm: MatchedArm,
    seed: int = 20260803,
    generator_sha256: str = GENERATOR_CHECKPOINT_SHA256,
) -> MatchedGenerationRequest:
    entry = MatchedScheduleEntry(
        unit_id="selected-real-one-sample",
        morphology_program=canonical_morphology_program_bytes(_real_program()),
        program_index=0,
        particle_index=0,
        checkpoint_index=SAMPLE_STEPS,
        generator_checkpoint_sha256=generator_sha256,
        closure_checkpoint_sha256=CLOSURE_CHECKPOINT_SHA256,
        rollout_index=0,
        productive_generation_calls=1,
        route_reservation=RouteComputeUsage(
            logical_planner_calls=3,
            physical_planner_calls=3,
            logical_verifier_calls=3,
            physical_verifier_calls=3,
        ),
    )
    return MatchedGenerationRequest(arm=arm, entry=entry, productive_seed=seed)


@pytest.fixture(scope="module")
def real_lane() -> SelectedRestartableGeneratorLane:
    return build_selected_step1000_restartable_generator_lane(REPO)


def test_program_decoder_requires_and_recovers_exact_canonical_bytes() -> None:
    payload = canonical_morphology_program_bytes(_real_program())
    program = decode_canonical_morphology_program_bytes(payload)

    assert canonical_morphology_program_bytes(program) == payload
    with pytest.raises(
        UgiRestartableTerminalSupportAdapterError,
        match="not canonically encoded",
    ):
        decode_canonical_morphology_program_bytes(payload.replace(b'":', b'": '))


def test_invalid_and_nonexact_rows_are_sealed_without_route_support() -> None:
    program = _real_program()
    offspring = {
        role: [1] * (count - 1) + [0]
        for role, count in zip(
            ("amine_head", "oxoester_aldehyde_body_tail", "isocyanide_tail"),
            program["node_counts"],
            strict=True,
        )
    }
    invalid = {
        "valid": False,
        "component_reconstruction_valid": False,
        "smiles": None,
        "component_smiles_by_role": None,
        "program": program,
        "offspring_by_role": offspring,
    }
    invalid_terminal = lock_unqualified_restartable_completion_row(
        invalid,
        generation_request=_request(arm=MatchedArm.GUIDED),
        nonqualification_reason="native_molecule_invalid",
    )

    assert not invalid_terminal.terminal_valid
    assert not invalid_terminal.exact_l1
    assert invalid_terminal.payload is None
    assert native_completion_record_from_locked_terminal(invalid_terminal) == invalid

    nonexact = {
        **invalid,
        "valid": True,
        "component_reconstruction_valid": True,
        "smiles": "CC",
        "component_smiles_by_role": {
            "amine_head": "CN",
            "oxoester_aldehyde_body_tail": "CC=O",
            "isocyanide_tail": "[C-]#[N+]C",
        },
    }
    nonexact_terminal = lock_unqualified_restartable_completion_row(
        nonexact,
        generation_request=_request(arm=MatchedArm.GUIDED),
        nonqualification_reason="independent_l1_forward_failed",
    )

    assert nonexact_terminal.terminal_valid
    assert not nonexact_terminal.exact_l1
    assert native_completion_record_from_locked_terminal(nonexact_terminal) == nonexact

    with pytest.raises(
        UgiRestartableTerminalSupportAdapterError,
        match="reason disagrees",
    ):
        lock_unqualified_restartable_completion_row(
            invalid,
            generation_request=_request(arm=MatchedArm.GUIDED),
            nonqualification_reason="independent_l1_forward_failed",
        )


def test_factory_rejects_schedule_checkpoint_mismatch_before_sampling(
    real_lane: SelectedRestartableGeneratorLane,
) -> None:
    request = _request(arm=MatchedArm.GUIDED)
    mismatched = replace(
        request,
        entry=replace(request.entry, generator_checkpoint_sha256="0" * 64),
    )

    with pytest.raises(
        UgiSelectedRestartableGeneratorError,
        match="schedule generator checkpoint",
    ):
        real_lane.adapter.generate_locked_terminal(mismatched)


def _mock_callback(
    monkeypatch: pytest.MonkeyPatch, row: dict
) -> SelectedRestartableGeneratorCallback:
    implementation_qualification = build_selected_generator_implementation_qualification(REPO)
    monkeypatch.setattr(
        "forge.product.ugi_selected_restartable_generator.sample_restartable_terminals",
        lambda *args, **kwargs: (
            [object()],
            {"sample_steps": SAMPLE_STEPS, "terminal_tree_repairs": 0, "samples": 1},
        ),
    )
    monkeypatch.setattr(
        "forge.product.ugi_selected_restartable_generator.complete_ugi_joint_terminals",
        lambda *args, **kwargs: SimpleNamespace(rows=(row,)),
    )
    bindings = SelectedRestartableGeneratorBindings(
        generator_checkpoint_sha256=GENERATOR_CHECKPOINT_SHA256,
        closure_checkpoint_sha256=CLOSURE_CHECKPOINT_SHA256,
        production_generator_manifest_sha256="1" * 64,
        restartable_equivalence_receipt_sha256="2" * 64,
        generator_implementation_sha256=(implementation_qualification.implementation_sha256),
        training_cache_sha256="3" * 64,
        atom_vocabulary_sha256="4" * 64,
        qualified_reaction_registry_sha256="5" * 64,
        l1_reaction_sha256="9" * 64,
        model_config_sha256="6" * 64,
        declared_graph_support_sha256="7" * 64,
        component_recovery_contract_sha256="8" * 64,
        sample_steps=SAMPLE_STEPS,
        terminal_decoder_id="ugi_joint_terminal_completion:argmax:v1",
    )
    return SelectedRestartableGeneratorCallback(
        model=object(),
        closure_model=object(),
        source_marginals={},
        corpus=object(),
        reaction=object(),
        graph_support=object(),  # unused on the unqualified path
        l1_reverifier=object(),  # unused on the unqualified path
        bindings=bindings,
        allowed_ring_sizes=(5, 6, 7),
        maximum_heavy_degree=4,
        repository=REPO,
        implementation_qualification=implementation_qualification,
    )


def test_callback_seals_invalid_native_row_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    program = _real_program()
    row = {
        "valid": False,
        "component_reconstruction_valid": False,
        "smiles": None,
        "component_smiles_by_role": None,
        "program": program,
        "offspring_by_role": {
            role: [1] * (count - 1) + [0]
            for role, count in zip(
                ("amine_head", "oxoester_aldehyde_body_tail", "isocyanide_tail"),
                program["node_counts"],
                strict=True,
            )
        },
    }
    callback = _mock_callback(monkeypatch, row)

    terminal = callback(_request(arm=MatchedArm.GUIDED))

    assert not terminal.terminal_valid
    assert not terminal.exact_l1
    assert terminal.payload is None
    assert native_completion_record_from_locked_terminal(terminal) == row


def test_callback_seals_independently_nonexact_native_row_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    program = _real_program()
    row = {
        "valid": True,
        "component_reconstruction_valid": True,
        "smiles": "CC",
        "component_smiles_by_role": {
            "amine_head": "CN",
            "oxoester_aldehyde_body_tail": "CC=O",
            "isocyanide_tail": "[C-]#[N+]C",
        },
        "program": program,
        "offspring_by_role": {
            role: [1] * (count - 1) + [0]
            for role, count in zip(
                ("amine_head", "oxoester_aldehyde_body_tail", "isocyanide_tail"),
                program["node_counts"],
                strict=True,
            )
        },
    }
    callback = _mock_callback(monkeypatch, row)

    def fail_exact(cls, **kwargs):
        raise UgiTerminalRouteAssessmentError("independent exact L1 failed")

    monkeypatch.setattr(
        ValidatedUgiTerminalPayload,
        "from_recovered_components",
        classmethod(fail_exact),
    )

    terminal = callback(_request(arm=MatchedArm.GUIDED))

    assert terminal.terminal_valid
    assert not terminal.exact_l1
    assert terminal.payload is None
    assert native_completion_record_from_locked_terminal(terminal) == row


def test_real_selected_checkpoint_one_sample_is_deterministic_and_trace_owned(
    real_lane: SelectedRestartableGeneratorLane,
) -> None:
    guided = real_lane.adapter.generate_locked_terminal(_request(arm=MatchedArm.GUIDED))
    post_hoc = real_lane.adapter.generate_locked_terminal(_request(arm=MatchedArm.POST_HOC))

    assert guided.terminal_valid and guided.exact_l1
    assert guided.terminal_bytes == post_hoc.terminal_bytes
    assert guided.generation_trace_bytes == post_hoc.generation_trace_bytes
    native = native_completion_record_from_locked_terminal(guided)
    assert native["valid"] is True
    assert native["component_reconstruction_valid"] is True
    assert native["program"] == _real_program()
    adapted = real_lane.callback.adapted_for_terminal(guided)
    assert adapted.candidate_record == native
    assert adapted.support.terminal_sha256 == guided.terminal_sha256
    assert tuple(target.role for target in adapted.support.root_targets) == (
        "amine_head",
        "oxoester_aldehyde_body_tail",
        "isocyanide_tail",
    )
