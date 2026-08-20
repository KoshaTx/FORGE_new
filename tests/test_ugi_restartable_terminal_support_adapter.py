from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from experiments.phase1.synthesis_guidance.adapters.terminal_support import (
    UgiRestartableTerminalSupportAdapterError,
    adapt_restartable_completion_row_for_route_support,
    canonical_morphology_program_bytes,
    native_candidate_eligibility_record,
    native_candidate_eligibility_record_bytes,
)
from forge.corpus.ugi_generated_terminal_support import DeclaredGraphSupportContext
from forge.corpus.ugi_held_component_gate import load_ugi_reaction_contract
from forge.potency.annotations import ROLE_NAMES
from forge.synthesis.matched import (
    MatchedArm,
    MatchedGenerationRequest,
    MatchedScheduleEntry,
    RouteComputeUsage,
)
from forge.synthesis.terminals.terminal_assessment import QualifiedUgiL1Reverifier

REPO = Path(__file__).resolve().parents[1]
SAMPLE_PATH = REPO / "results/phase1/ugi_architecture_selection_v3/full_step1000/result.json"
GENERATOR_PATH = REPO / "results/phase1/ugi_joint_sparse_balanced_v2_full/checkpoint_step_1000.pt"
CLOSURE_PATH = REPO / "results/phase1/ugi_closure_expanded_full/checkpoint_best.pt"
ATOM_VOCABULARY_PATH = REPO / "results/phase1/product_v3_atom_vocabulary.json"
REACTION_PATH = REPO / "data/vendor/qualified_reactions_v1.json"
L1_VARIANT_PATH = REPO / "configs/assembly/ugi_variant.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _real_row() -> dict:
    rows = json.loads(SAMPLE_PATH.read_text())["samples"]
    return next(
        row
        for row in rows
        if row.get("valid") is True and row.get("component_reconstruction_valid") is True
    )


def _graph_support() -> DeclaredGraphSupportContext:
    checkpoint = torch.load(GENERATOR_PATH, map_location="cpu", weights_only=False)
    vocabulary = json.loads(ATOM_VOCABULARY_PATH.read_text())["atom_vocabulary"]
    return DeclaredGraphSupportContext(
        generator_checkpoint_sha256=_sha256(GENERATOR_PATH),
        model_config=checkpoint["model_config"],
        atom_vocabulary=frozenset(
            (
                str(row["symbol"]),
                int(row["formal_charge"]),
                bool(row["aromatic"]),
                int(row["explicit_hydrogens"]),
            )
            for row in vocabulary
        ),
    )


def _request(
    row: dict,
    *,
    arm: MatchedArm = MatchedArm.GUIDED,
    morphology_program: bytes | None = None,
) -> MatchedGenerationRequest:
    entry = MatchedScheduleEntry(
        unit_id="real-frozen-sample-0",
        morphology_program=(
            morphology_program or canonical_morphology_program_bytes(row["program"])
        ),
        program_index=0,
        particle_index=0,
        checkpoint_index=8,
        generator_checkpoint_sha256=_sha256(GENERATOR_PATH),
        closure_checkpoint_sha256=_sha256(CLOSURE_PATH),
        rollout_index=0,
        productive_generation_calls=1,
        route_reservation=RouteComputeUsage(
            logical_planner_calls=3,
            physical_planner_calls=3,
            logical_verifier_calls=3,
            physical_verifier_calls=3,
        ),
    )
    return MatchedGenerationRequest(
        arm=arm,
        entry=entry,
        productive_seed=20260803,
    )


def _adapt(row: dict, *, arm: MatchedArm = MatchedArm.GUIDED):
    reaction = load_ugi_reaction_contract(REACTION_PATH)
    reaction_sha256 = _sha256(L1_VARIANT_PATH)
    return adapt_restartable_completion_row_for_route_support(
        row,
        generation_request=_request(row, arm=arm),
        l1_reaction=reaction,
        l1_reaction_sha256=reaction_sha256,
        component_recovery_contract_sha256=hashlib.sha256(
            b"generated-component-recovery-v1"
        ).hexdigest(),
        graph_support=_graph_support(),
        l1_reverifier=QualifiedUgiL1Reverifier(
            reaction_contract=reaction,
            l1_reaction_sha256=reaction_sha256,
        ),
    )


def test_real_frozen_sample_preserves_every_native_eligibility_field_exactly() -> None:
    row = _real_row()

    record = native_candidate_eligibility_record(row)
    adapted = _adapt(row)

    assert set(record) == {
        "valid",
        "component_reconstruction_valid",
        "smiles",
        "component_smiles_by_role",
        "program",
        "offspring_by_role",
    }
    assert record["valid"] is row["valid"] is True
    assert record["component_reconstruction_valid"] is True
    assert record["smiles"] == row["smiles"]
    assert record["component_smiles_by_role"] == {
        role: row["component_smiles_by_role"][role] for role in ROLE_NAMES
    }
    assert record["program"] == row["program"]
    assert record["offspring_by_role"] == row["offspring_by_role"]
    assert adapted.candidate_record == record
    assert adapted.candidate_record_bytes == native_candidate_eligibility_record_bytes(row)
    assert adapted.locked_terminal.terminal_locked
    assert adapted.locked_terminal.terminal_valid
    assert adapted.locked_terminal.exact_l1
    assert adapted.support.product_smiles == row["smiles"]
    assert tuple(target.role for target in adapted.support.root_targets) == ROLE_NAMES


def test_arm_label_cannot_change_zero_guidance_terminal_or_trace_bytes() -> None:
    row = _real_row()

    guided = _adapt(row, arm=MatchedArm.GUIDED)
    post_hoc = _adapt(row, arm=MatchedArm.POST_HOC)

    assert guided.locked_terminal.terminal_bytes == post_hoc.locked_terminal.terminal_bytes
    assert (
        guided.locked_terminal.generation_trace_bytes
        == post_hoc.locked_terminal.generation_trace_bytes
    )
    assert guided.candidate_record_bytes == post_hoc.candidate_record_bytes


def test_schedule_must_bind_exact_native_program_bytes() -> None:
    row = _real_row()
    reaction = load_ugi_reaction_contract(REACTION_PATH)
    reaction_sha256 = _sha256(L1_VARIANT_PATH)

    with pytest.raises(UgiRestartableTerminalSupportAdapterError, match="morphology bytes"):
        adapt_restartable_completion_row_for_route_support(
            row,
            generation_request=_request(row, morphology_program=b"lookalike-program\n"),
            l1_reaction=reaction,
            l1_reaction_sha256=reaction_sha256,
            component_recovery_contract_sha256=hashlib.sha256(b"recovery").hexdigest(),
            graph_support=_graph_support(),
            l1_reverifier=QualifiedUgiL1Reverifier(reaction, reaction_sha256),
        )


@pytest.mark.parametrize(
    "field",
    [
        "valid",
        "component_reconstruction_valid",
        "smiles",
        "component_smiles_by_role",
        "program",
        "offspring_by_role",
    ],
)
def test_missing_native_field_is_never_reconstructed(field: str) -> None:
    row = _real_row()
    del row[field]

    with pytest.raises(UgiRestartableTerminalSupportAdapterError, match="missing native fields"):
        native_candidate_eligibility_record(row)


def test_malformed_offspring_word_is_not_repaired_from_program_or_product() -> None:
    row = _real_row()
    row["offspring_by_role"]["isocyanide_tail"] = row["offspring_by_role"]["isocyanide_tail"][:-1]

    with pytest.raises(UgiRestartableTerminalSupportAdapterError, match="offspring word"):
        native_candidate_eligibility_record(row)


def test_native_component_identity_is_not_recovered_again_when_missing() -> None:
    row = _real_row()
    row["component_smiles_by_role"] = {
        role: smiles
        for role, smiles in row["component_smiles_by_role"].items()
        if role != "amine_head"
    }

    with pytest.raises(UgiRestartableTerminalSupportAdapterError, match="component identities"):
        native_candidate_eligibility_record(row)


def test_generator_support_context_must_match_schedule_checkpoint() -> None:
    row = _real_row()
    reaction = load_ugi_reaction_contract(REACTION_PATH)
    reaction_sha256 = _sha256(L1_VARIANT_PATH)
    graph_support = _graph_support()
    mismatched = replace_graph_support_checkpoint(
        graph_support, hashlib.sha256(b"other").hexdigest()
    )

    with pytest.raises(UgiRestartableTerminalSupportAdapterError, match="generator checkpoints"):
        adapt_restartable_completion_row_for_route_support(
            row,
            generation_request=_request(row),
            l1_reaction=reaction,
            l1_reaction_sha256=reaction_sha256,
            component_recovery_contract_sha256=hashlib.sha256(b"recovery").hexdigest(),
            graph_support=mismatched,
            l1_reverifier=QualifiedUgiL1Reverifier(reaction, reaction_sha256),
        )


def replace_graph_support_checkpoint(
    value: DeclaredGraphSupportContext,
    checkpoint_sha256: str,
) -> DeclaredGraphSupportContext:
    return DeclaredGraphSupportContext(
        generator_checkpoint_sha256=checkpoint_sha256,
        model_config=value.model_config,
        atom_vocabulary=value.atom_vocabulary,
    )


def test_native_record_is_a_copy_not_a_mutable_view_of_completion_row() -> None:
    row = _real_row()
    record = native_candidate_eligibility_record(row)
    frozen_bytes = native_candidate_eligibility_record_bytes(row)

    changed = deepcopy(row)
    changed["program"]["node_counts"][0] += 1
    changed["offspring_by_role"]["amine_head"].append(0)

    assert native_candidate_eligibility_record_bytes(record) == frozen_bytes
