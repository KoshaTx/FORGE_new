from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path

import pytest

from forge.synthesis.sources.supervision_inventory import (
    CHEMISTRY_CLASSES,
    InventoryError,
    _combined_lipid_specific,
    _summarize_agile,
    _summarize_rm,
    analyze_r0_structure_context,
    analyze_uspto,
    analyze_uspto_mit,
    load_source_ledger,
    validate_vendored_assets,
    write_inventory,
)

REPO = Path(__file__).resolve().parents[1]
LEDGER = REPO / "configs/route/m0_09_supervision_sources.json"


def _uspto_config() -> dict:
    return load_source_ledger(LEDGER)["uspto_pretraining"]


def test_curated_lipid_inventory_counts_and_l2_exclusions() -> None:
    ledger = load_source_ledger(LEDGER)
    agile = _summarize_agile(ledger["agile_si"])
    rm = _summarize_rm(ledger["internal_rm_protocols"])
    combined = _combined_lipid_specific(agile, rm)

    assert agile["distinct_upstream_reaction_instances"] == 48
    assert agile["unique_intermediate_products"] == 23
    assert agile["unique_terminal_precursor_scaffolds"] == 25
    assert agile["terminal_route_depth_distribution"] == {"1": 2, "2": 23}
    assert rm["protocol_documents"] == 7
    assert combined["distinct_upstream_reaction_instances"] == 55
    assert combined["unique_upstream_products"] == 55
    assert combined["unique_terminal_precursor_scaffolds"] == 32
    assert combined["terminal_route_depth_distribution"] == {"1": 9, "2": 23}
    assert combined["lipid_relevant_coverage"] == {
        "ester": 23,
        "isocyanide": 7,
        "aldehyde": 18,
        "carbonate": 0,
        "acrylate": 4,
        "heterocycle_formation": 0,
    }
    assert combined["explicit_failed_syntheses"] == 0
    assert combined["rm_protocols_with_missing_outcomes"] == 7

    declared_text = json.dumps(ledger)
    assert "r1_reaction_enumerated_support_v1.csv" in declared_text
    assert "R1 cannot be declared" not in declared_text
    assert all(
        corpus["l2_exclusion_reason"]
        for corpus in ledger["related_structure_corpora_not_counted_as_l2"]
    )


def test_uspto_50k_coverage_heuristic_and_class_counts(tmp_path: Path) -> None:
    path = tmp_path / "uspto.csv"
    rows = [
        ("ester", 2, "CC(=O)O.CCO>>CC(=O)OCC"),
        ("aldehyde", 8, "CCO>>CC=O"),
        ("isocyanide", 9, "CNC=O>>C[N+]#[C-]"),
        ("carbonate", 2, "CO.COC(=O)Cl>>COC(=O)OC"),
        ("acrylate", 2, "C=CC(=O)O.CCO>>C=CC(=O)OCC"),
        ("heterocycle", 4, "NCCBr>>C1CN1"),
    ]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "class", "reactions"])
        writer.writerows(rows)

    result = analyze_uspto(path, _uspto_config())

    assert result["reaction_records"] == 6
    assert result["reaction_classes"] == 4
    for chemistry in CHEMISTRY_CLASSES:
        expected = 2 if chemistry == "ester" else 1
        assert (
            result["lipid_relevant_coverage"][chemistry]["candidate_formation_reactions"]
            == expected
        )


def test_uspto_mit_archive_counts_splits_without_inventing_classes(tmp_path: Path) -> None:
    path = tmp_path / "uspto.zip"
    reaction = "CC(=O)O.CCO>>CC(=O)OCC 1-2;2-3\n"
    config = _uspto_config()
    large_config = config["large_corpus"]
    with zipfile.ZipFile(path, "w") as archive:
        for member in large_config["archive_members"].values():
            archive.writestr(member, reaction)

    result = analyze_uspto_mit(path, large_config, config)

    assert result["reaction_records"] == 3
    assert result["split_counts"] == {"train": 1, "validation": 1, "test": 1}
    assert result["reaction_classes"] is None
    assert result["lipid_relevant_coverage"]["ester"]["candidate_formation_reactions"] == 3
    assert (
        result["lipid_relevant_coverage"]["heterocycle_formation"]["candidate_formation_reactions"]
        is None
    )


def test_r0_region_context_counts_raw_blocks_without_calling_them_routes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "r0.csv"
    annotations = {
        "lnpdb_v1": {
            "IL_head_SMILES": ["CN", "CN"],
            "IL_linker_SMILES": ["CCO"],
            "IL_tail1_SMILES": ["CCCC"],
            "IL_tail2_SMILES": ["CCCCC"],
        },
        "lion_repository_all": {"Amine_SMILES": ["NCC"]},
        "agile_measured1200": {
            "A_smiles": ["CN"],
            "B_smiles": ["CC=O"],
            "C_smiles": ["C[N+]#[C-]"],
        },
    }
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["region_annotations_json"])
        writer.writeheader()
        writer.writerow({"region_annotations_json": json.dumps(annotations)})

    result = analyze_r0_structure_context(path)

    assert result["r0_records"] == 1
    assert (
        result["sources"]["lnpdb_v1"]["region_fields"]["IL_head_SMILES"]["unique_raw_values"] == 1
    )
    assert "not observed L2 routes" in result["interpretation"]


def test_asset_hash_mismatch_fails_with_asset_name(tmp_path: Path) -> None:
    ledger = load_source_ledger(LEDGER)
    (tmp_path / "uspto_50k.csv").write_text("corrupt")

    with pytest.raises(InventoryError, match="hash mismatch for uspto_50k.csv"):
        validate_vendored_assets(
            {
                **ledger,
                "uspto_pretraining": {
                    **ledger["uspto_pretraining"],
                    "large_corpus": {
                        **ledger["uspto_pretraining"]["large_corpus"],
                        "asset": "uspto_50k.csv",
                        "expected_sha256": ledger["uspto_pretraining"]["expected_sha256"],
                    },
                },
                "agile_si": {
                    **ledger["agile_si"],
                    "asset": "uspto_50k.csv",
                    "expected_sha256": ledger["uspto_pretraining"]["expected_sha256"],
                },
                "internal_rm_protocols": {
                    **ledger["internal_rm_protocols"],
                    "protocols": [
                        {
                            **protocol,
                            "asset": "uspto_50k.csv",
                            "expected_sha256": ledger["uspto_pretraining"]["expected_sha256"],
                        }
                        for protocol in ledger["internal_rm_protocols"]["protocols"]
                    ],
                },
                "related_structure_corpora_not_counted_as_l2": [],
            },
            tmp_path,
        )


def test_write_inventory_is_stable_for_same_result(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    result = {
        "schema_version": "test",
        "generated_utc": "2026-07-28T12:00:00+00:00",
        "randomness": {"seed": 0, "used": False},
    }

    write_inventory(result, output)
    first = output.read_bytes()
    write_inventory(result, output)

    assert output.read_bytes() == first
    assert json.loads(first) == result


def test_committed_result_records_the_frozen_m0_09_decision() -> None:
    result = json.loads((REPO / "results/m0_09/result.json").read_text())
    uspto = result["inventory"]["uspto_pretraining"]

    assert result["schema_version"] == "m0_09_l2_supervision_inventory.v1"
    assert result["randomness"] == {"seed": 0, "used": False}
    assert result["inputs"][0]["asset"] == "configs/route/m0_09_supervision_sources.json"
    assert uspto["large_corpus"]["reaction_records"] == 479035
    assert uspto["large_corpus"]["rdkit_invalid_records"] == 37
    assert uspto["class_labeled_benchmark"]["reaction_records"] == 50016
    assert result["lipid_specific_combined"]["distinct_upstream_reaction_instances"] == 55
    assert result["lipid_specific_combined"]["unique_terminal_precursor_scaffolds"] == 32
    assert result["lipid_specific_combined"]["explicit_failed_syntheses"] == 0
    assert result["viability_decision"]["standalone_learned_lipid_specific_l2_viable"] is False
