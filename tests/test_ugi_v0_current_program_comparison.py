from __future__ import annotations

from types import SimpleNamespace

import pytest

from experiments.phase1.product_l1.evaluation.ugi_v0_current_program_comparison import (
    UgiV0CurrentProgramComparisonError,
    _metric_deltas,
    _validate_config,
    project_program_for_mixed_transformer,
)
from forge.model.synthesis_program_layout import (
    SynthesisProgramLayoutPrior,
    _compile_program_distribution,
    _RecordSummary,
)
from forge.model.ugi_morphology_program import UgiMorphologyProgram

INTERNAL_EDGES = (
    ((10, 3, True), (11, 6, True), 0),
    ((11, 6, True), (12, 5, True), 0),
    ((12, 5, True), (13, 5, True), 0),
    ((12, 5, True), (14, 4, True), 1),
)
ONE_HEAD_ATTACHMENT = tuple(
    sorted(
        (
            ((1, 3, False), (10, 3, True), 0),
            ((1, 5, False), (13, 5, True), 0),
            ((1, 6, False), (11, 6, True), 0),
            *INTERNAL_EDGES,
        )
    )
)
TWO_HEAD_ATTACHMENTS = tuple(sorted((ONE_HEAD_ATTACHMENT[0], *ONE_HEAD_ATTACHMENT)))


def _summary(amine_size: int, fixed_signature: tuple[tuple[object, ...], ...]):
    return _RecordSummary(
        depth=1,
        closure_count=0,
        blocks=(
            (3, amine_size, ((10, 1),)),
            (4, 1, ((14, 1),)),
            (5, 6, ((12, 1), (13, 1))),
            (6, 7, ((11, 1),)),
        ),
        fixed_signature=fixed_signature,
        weight=0.5,
    )


def _prior() -> SynthesisProgramLayoutPrior:
    prior = object.__new__(SynthesisProgramLayoutPrior)
    prior.vocabulary = SimpleNamespace(
        role_states=(
            "unassigned",
            "aldehyde_tail",
            "alkyl_tail",
            "amine_head",
            "assembly_introduced",
            "isocyanide_tail",
            "oxoester_aldehyde_body_tail",
        ),
        program_to_index={"ugi_3cr_agile": 1},
    )
    prior.maximum_heavy_atoms = 194
    prior.maximum_closures = 3
    prior._distributions = {
        "ugi_3cr_agile": _compile_program_distribution(
            (
                _summary(2, ONE_HEAD_ATTACHMENT),
                _summary(3, TWO_HEAD_ATTACHMENTS),
            )
        )
    }
    prior._fixed_node_states = {"ugi_3cr_agile": {10: 1, 11: 1, 12: 1, 13: 1, 14: 2}}
    return prior


@pytest.mark.parametrize(
    ("head_nodes", "head_attachments"),
    ((1, 1), (2, 2)),
)
def test_program_projection_selects_the_matching_attachment_contract(
    head_nodes: int, head_attachments: int
) -> None:
    program = UgiMorphologyProgram(
        node_counts=(head_nodes, 6, 4),
        junction_budgets=(0, 0, 0),
        cycle_ranks=(1, 0, 0),
        attachment_counts=(head_attachments, 1, 1),
    )

    record = project_program_for_mixed_transformer(_prior(), program, sample_index=7)

    assert record.graph.structure_id == "factorized-layout-ugi_3cr_agile-00000007"
    assert record.graph.closure_count == 1
    assert record.node_count == sum(program.node_counts) + 5
    assert [block.role for block in record.component_blocks] == [
        "amine_head",
        "oxoester_aldehyde_body_tail",
        "isocyanide_tail",
        "assembly_introduced",
    ]
    assert record.role_morphology_states is not None
    for role_index, role in enumerate(
        ("amine_head", "oxoester_aldehyde_body_tail", "isocyanide_tail")
    ):
        state = _prior().vocabulary.role_states.index(role)
        observed = record.role_morphology_states[record.role_states == state]
        expected = [
            program.node_counts[role_index] + 1,
            program.junction_budgets[role_index] + 1,
            program.cycle_ranks[role_index] + 1,
            program.attachment_counts[role_index] + 1,
        ]
        assert (observed == expected).all()


def test_program_projection_fails_on_an_unsupported_attachment_contract() -> None:
    program = UgiMorphologyProgram(
        node_counts=(1, 6, 4),
        junction_budgets=(0, 0, 0),
        cycle_ranks=(0, 0, 0),
        attachment_counts=(2, 1, 1),
    )

    with pytest.raises(UgiV0CurrentProgramComparisonError, match="one exact"):
        project_program_for_mixed_transformer(_prior(), program, sample_index=0)


def test_metric_deltas_require_identical_method_blind_metric_sets() -> None:
    assert _metric_deltas({"a": 0.7, "b": 0.5}, {"a": 0.6, "b": 0.7}) == {
        "a": pytest.approx(0.1),
        "b": pytest.approx(-0.2),
    }
    with pytest.raises(UgiV0CurrentProgramComparisonError, match="metric sets"):
        _metric_deltas({"a": 1.0}, {"b": 1.0})
    assert _metric_deltas({"a": None}, {"a": None}) == {"a": None}
    assert _metric_deltas({"a": None}, {"a": 1.0}) == {"a": None}


def test_comparison_contract_rejects_a_seeded_argmax_decoder() -> None:
    config = {
        "schema_version": "forge.ugi_v0_current_program_comparison_config.v1",
        "inputs": {
            label: {}
            for label in (
                "checkpoint_archive",
                "closure_checkpoint",
                "common_ugi_assessment_config",
                "current_training_result",
                "lipid_realism_config",
                "local_chemistry_config",
                "prepared_cache",
                "production_cache",
                "production_design",
                "program_draw",
                "qualified_reactions",
                "role_morphology_policy",
                "v0_checkpoint",
            )
        },
        "profiles": {
            "full": {
                "program_count": 4,
                "sample_steps": 2,
                "batch_size": 4,
                "terminal_decoder_mode": "argmax",
                "terminal_decoder_seed": 17,
                "current_terminal_decode_policy": "strict_valence_topology_argmax",
                "maximum_adjacent_branch_runs": [2, 1, 1],
            }
        },
        "policy": {
            "candidate_selection": False,
            "method_blind_assessment": True,
            "negative_results_reported": True,
            "oracle_calls": 0,
            "paired_program_order": True,
            "random_streams_matched": False,
            "repairs_or_retries": False,
            "route_calls": 0,
            "training_calls": 0,
        },
    }

    with pytest.raises(UgiV0CurrentProgramComparisonError, match="runtime contract"):
        _validate_config(config, profile="full")
