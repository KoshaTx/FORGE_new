from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from forge.core.io import read_csv_rows, read_json_object

REPO = Path(__file__).resolve().parents[1]
CHECKED = REPO / "results/phase1/bl_lx_reaction_enumerated_expansion_v1"
SOURCE = REPO / "results/phase1/multireaction_program_corpus_v1"


def test_checked_bl_lx_expansion_is_leakage_safe_and_evidence_separated() -> None:
    result = read_json_object(CHECKED / "result.json")
    assert result["summary"]["source_products_preserved"] == 764
    assert result["summary"]["computed_products_admitted"] == 22_187
    assert result["summary"]["total_products"] == 22_951
    assert result["summary"]["products_by_program_fold"] == {
        "bl_2023_repeated_aza_michael|calibration": 2350,
        "bl_2023_repeated_aza_michael|heldout": 1551,
        "bl_2023_repeated_aza_michael|train": 4757,
        "lx_2024_repeated_reductive_amination|calibration": 3372,
        "lx_2024_repeated_reductive_amination|heldout": 4017,
        "lx_2024_repeated_reductive_amination|train": 6904,
    }
    assert result["claims_boundary"] == {
        "component_family_assignment_precedes_enumeration": True,
        "computed_transform_consistency_is_observed_synthesis": False,
        "computed_transform_consistency_is_route_closure": False,
        "computed_transform_consistency_is_synthesis_success": False,
        "reductive_amination_substructure_hit_rate_reported": False,
        "source_activity_labels_inherited": False,
        "source_executed_evidence_rewritten": False,
        "training_must_use_source_balanced_weight": True,
    }

    components = read_csv_rows(CHECKED / "component_registry.csv.gz")
    family_folds: dict[str, set[str]] = defaultdict(set)
    shared_heads: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in components:
        if row["program_structural_admission"] != "true":
            continue
        family_folds[row["family_id"]].add(row["family_fold"])
        if row["split_namespace"] == "amine_head":
            shared_heads[row["canonical_smiles"]].add(
                (row["family_id"], row["family_fold"])
            )
    assert all(len(folds) == 1 for folds in family_folds.values())
    assert all(len(assignments) == 1 for assignments in shared_heads.values())
    assert any(
        len({row["program_id"] for row in components if row["canonical_smiles"] == smiles}) == 2
        for smiles in shared_heads
    )

    splits = read_csv_rows(CHECKED / "component_family_splits.csv.gz")
    stratum_mass: dict[tuple[str, str, str], float] = defaultdict(float)
    program_mass: dict[tuple[str, str], float] = defaultdict(float)
    for row in splits:
        weight = float(row["source_balanced_weight"])
        stratum_mass[(row["product_fold"], row["program_id"], row["evidence_stratum"])] += weight
        program_mass[(row["product_fold"], row["program_id"])] += weight
    assert all(abs(value - 0.5) < 1e-8 for value in stratum_mass.values())
    assert all(abs(value - 1.0) < 1e-8 for value in program_mass.values())

    expanded_atlas = read_csv_rows(CHECKED / "reaction_program_atlas.csv.gz")
    source_atlas = read_csv_rows(SOURCE / "reaction_program_atlas.csv.gz")
    source_products = {
        (row["program_id"], row["canonical_product_smiles"])
        for row in source_atlas
        if row["disposition"] == "admit_exact"
    }
    expanded_source = {
        (row["program_id"], row["canonical_product_smiles"])
        for row in expanded_atlas
        if row["disposition"] == "admit_exact"
    }
    assert expanded_source == source_products
    assert sum(
        row["disposition"] == "admit_transform_consistency" for row in expanded_atlas
    ) == 22_187
    assert all(row["semantic_origin_status"] == "exact" for row in expanded_atlas)
