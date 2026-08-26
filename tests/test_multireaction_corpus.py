from __future__ import annotations

import csv
import gzip
from collections import defaultdict
from pathlib import Path

from forge.corpus.multireaction import build_multireaction_lnpdb_corpus
from forge.corpus.reaction_program_training import load_reaction_program_training_corpus
from forge.model.reaction_program_sampling import sample_factorized_program_layouts

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/multireaction/lnpdb_reaction_programs_v1.json"
CHECKED = REPO / "results/phase1/multireaction_program_corpus_v1"


def _rows(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def test_multireaction_corpus_closes_frozen_source_gates(tmp_path: Path) -> None:
    outputs = {
        "atlas": tmp_path / "reaction_program_atlas.csv.gz",
        "steps": tmp_path / "reaction_program_steps.csv.gz",
        "semantic_atoms": tmp_path / "semantic_atoms.csv.gz",
        "provenance": tmp_path / "source_provenance.csv.gz",
        "splits": tmp_path / "component_disjoint_splits.csv.gz",
        "manifest": tmp_path / "manifest.json",
        "result": tmp_path / "result.json",
    }
    result = build_multireaction_lnpdb_corpus(CONFIG, REPO, outputs=outputs)
    assert result["summary"] == {
        "source_rows": 1624,
        "unique_source_products": 891,
        "admitted_products": 764,
        "abstained_products": 127,
        "exact_program_steps": 1344,
        "semantic_atom_rows": 56588,
        "semantic_origin_products": 764,
        "semantic_core_atom_rows": 6798,
        "semantic_core_products": 764,
        "product_folds": {"calibration": 196, "heldout": 240, "train": 328},
        "sampling_policy": "equal_total_weight_per_program_within_fold",
    }
    assert result["programs"]["bl_2023_repeated_aza_michael"]["admitted_products"] == 610
    assert result["programs"]["lx_2024_repeated_reductive_amination"]["admitted_products"] == 154

    atlas = _rows(outputs["atlas"])
    steps = _rows(outputs["steps"])
    semantic_atoms = _rows(outputs["semantic_atoms"])
    provenance = _rows(outputs["provenance"])
    splits = _rows(outputs["splits"])
    assert len(atlas) == 891
    assert len(steps) == 1344
    assert len(semantic_atoms) == 56588
    assert len(provenance) == 1624
    assert len(splits) == 764
    assert all(row["exact_forward_roundtrip"] == "true" for row in steps)
    assert {int(row["program_depth"]) for row in semantic_atoms} == {1, 2, 3, 4}
    assert sum(bool(row["core_position"]) for row in semantic_atoms) == 6798
    assert {row["core_position"] for row in semantic_atoms if row["core_position"]} == {
        "map_1",
        "map_2",
        "map_3",
        "map_4",
        "map_5",
        "map_6",
    }
    assert any("source_conflict" in row["abstention_reason"] for row in atlas)

    totals: dict[tuple[str, str], float] = defaultdict(float)
    for row in splits:
        totals[(row["product_fold"], row["program_id"])] += float(row["source_balanced_weight"])
    for fold in {row["product_fold"] for row in splits}:
        values = [value for (value_fold, _), value in totals.items() if value_fold == fold]
        assert max(values) - min(values) < 1e-8

    corpus = load_reaction_program_training_corpus(
        program_config_path=CONFIG,
        atlas_path=outputs["atlas"],
        semantic_atoms_path=outputs["semantic_atoms"],
        splits_path=outputs["splits"],
        declared_elements={"C", "N", "O"},
    )
    assert {fold: len(records) for fold, records in corpus.records_by_fold.items()} == {
        "train": 328,
        "calibration": 196,
        "heldout": 240,
    }
    assert corpus.maxima == {
        "heavy_atoms": 194,
        "closures": 3,
        "program_depth": 4,
        "accumulator_atoms": 18,
        "repeat_component_atoms": 45,
    }
    assert [state.symbol for state in corpus.atom_vocabulary] == ["C", "N", "O"]
    for fold, records in corpus.records_by_fold.items():
        mass: dict[str, float] = defaultdict(float)
        for record, weight in zip(records, corpus.weights_by_fold[fold], strict=True):
            mass[record.program_id] += float(weight)
        assert set(mass) == {
            "bl_2023_repeated_aza_michael",
            "lx_2024_repeated_reductive_amination",
        }
        assert max(mass.values()) - min(mass.values()) < 1e-8

    layouts = sample_factorized_program_layouts(
        corpus.records_by_fold["train"],
        corpus.weights_by_fold["train"],
        corpus.specifications,
        corpus.vocabulary,
        sample_count=16,
        seed=20260820,
    )
    assert all(len(layout.core_position_states) == layout.node_count for layout in layouts)
    assert all(any(value > 1 for value in layout.core_position_states) for layout in layouts)
    for name, output in outputs.items():
        assert output.read_bytes() == (CHECKED / output.name).read_bytes(), name
