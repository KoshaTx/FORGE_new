from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from forge.assembly import ReactionProgramSpec, RegistryRepeatedReactionProgram
from forge.chemistry.smiles import canonical_connected_constitution
from forge.core.hashing import sha256_file

REPO = Path(__file__).resolve().parents[1]
REGISTRY = REPO / "data/vendor/qualified_reaction_families_v1.json"
LNPDB = REPO / "data/vendor/lnpdb_fc7c389.csv"


def _source_row(study: str, product_label: str) -> dict[str, str]:
    with LNPDB.open(newline="") as handle:
        return next(
            row
            for row in csv.DictReader(handle)
            if row["Experiment_ID"] == study and row["IL_name"].strip() == product_label
        )


def test_aza_michael_program_recovers_and_replays_one_step() -> None:
    row = _source_row("BL_2023", "A11-T1")
    adapter = RegistryRepeatedReactionProgram.from_registry(
        REGISTRY,
        ReactionProgramSpec(
            program_id="bl_2023_repeated_aza_michael",
            reaction_id="aza_michael_amine_acrylate",
            accumulator_role="amine_head",
            repeat_role="alkyl_acrylate_or_acrylamide_tail",
            minimum_steps=1,
            maximum_steps=4,
        ),
        expected_sha256=str(sha256_file(REGISTRY)),
    )
    traces = adapter.decompose(row["IL_SMILES"], terminal_head_smiles=row["IL_head_SMILES"])
    assert len(traces) == 1
    assert traces[0].step_count == 1
    check = adapter.check_forward(
        traces[0].terminal_head_smiles,
        traces[0].repeated_component_smiles,
        row["IL_SMILES"],
    )
    assert check.exact
    assert not check.saturated
    products = adapter.forward_products(
        traces[0].terminal_head_smiles,
        traces[0].repeated_component_smiles,
    )
    assert str(canonical_connected_constitution(row["IL_SMILES"])) in products.products
    assert not products.saturated
    origins = adapter.atom_origins(traces[0])
    assert len(origins.atom_origins) == 63
    assert set(origins.atom_origins) == {"accumulator", "repeat"}
    assert Counter(origins.core_positions) == {
        "": 57,
        "map_1": 1,
        "map_2": 1,
        "map_3": 1,
        "map_4": 1,
        "map_5": 1,
        "map_6": 1,
    }
    assert origins.step_count == 1


def test_reductive_amination_program_recovers_ordered_repeated_steps() -> None:
    row = _source_row("LX_2024", "1_A2_T6")
    adapter = RegistryRepeatedReactionProgram.from_registry(
        REGISTRY,
        ReactionProgramSpec(
            program_id="lx_2024_repeated_reductive_amination",
            reaction_id="reductive_amination_amine_aldehyde",
            accumulator_role="amine_head",
            repeat_role="aldehyde_tail",
            minimum_steps=1,
            maximum_steps=6,
        ),
    )
    expected_tail = str(canonical_connected_constitution(row["IL_tail1_SMILES"].strip()))
    traces = adapter.decompose(
        row["IL_SMILES"],
        terminal_head_smiles=row["IL_head_SMILES"],
        expected_repeat_smiles=expected_tail,
    )
    assert len(traces) == 1
    assert traces[0].step_count == 2
    assert set(traces[0].repeated_component_smiles) == {expected_tail}
    assert adapter.check_forward(
        traces[0].terminal_head_smiles,
        traces[0].repeated_component_smiles,
        row["IL_SMILES"],
    ).exact
    origins = adapter.atom_origins(traces[0])
    assert len(origins.atom_origins) == 58
    assert set(origins.atom_origins) == {"accumulator", "repeat"}
    assert Counter(origins.core_positions) == {"": 54, "map_1": 2, "map_2": 2}
    assert origins.step_count == 2
    open_traces = adapter.decompose(
        row["IL_SMILES"],
        expected_repeat_smiles=expected_tail,
    )
    assert any(trace == traces[0] for trace in open_traces)
