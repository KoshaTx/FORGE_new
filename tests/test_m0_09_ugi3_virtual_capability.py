from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import pytest
from rdkit import Chem

from forge.corpus.r1_prime_audit import (
    compile_reactions,
    decompose_structure,
    load_reaction_definitions,
    sha256_file,
)
from forge.synthesis.evidence.ugi3_virtual_capability import (
    Ugi3VirtualCapabilityError,
    _forward_site_audit,
    build_ugi3_virtual_capability,
    write_ugi3_virtual_capability,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/route/m0_09_agile_virtual_ugi3_capability.json"
VIRTUAL = REPO / "data/derived/agile_virtual12k_smiles.csv.gz"
VIRTUAL_MANIFEST = REPO / "data/derived/agile_virtual12k_smiles.manifest.json"
REGISTRY = REPO / "data/vendor/qualified_reactions_v1.json"
ASSEMBLY = REPO / "results/m0_09/ugi3_assembly_qualification.json"
PRECURSOR = REPO / "results/m0_09/ugi3_precursor_capability.json"
ALDEHYDE_HEAD = REPO / "results/m0_09/ugi3_aldehyde_head_capability.json"
AGILE_COMPONENT_ROUTES = REPO / "results/m0_09/agile_component_routes.json"
RESULT = REPO / "results/m0_09/agile_virtual_ugi3_capability.json"


def _production_paths() -> tuple[Path, ...]:
    return (
        CONFIG,
        VIRTUAL,
        VIRTUAL_MANIFEST,
        REGISTRY,
        ASSEMBLY,
        PRECURSOR,
        ALDEHYDE_HEAD,
        AGILE_COMPONENT_ROUTES,
    )


def test_a19_roundtrip_records_one_explicit_target_site() -> None:
    config = json.loads(CONFIG.read_text())
    definition = load_reaction_definitions(
        (REGISTRY,),
        expected_count=1,
        role_policy_overrides=config["scope"]["role_policy_overrides"],
    )[0]
    reaction = compile_reactions((definition,))[0]
    with (REPO / "data/vendor/AGILE_smiles_with_value_group.csv").open(newline="") as handle:
        measured = next(row for row in csv.DictReader(handle) if row["label"] == "A19B1C1")
    target_molecule = Chem.MolFromSmiles(measured["combined_mol_SMILES"])
    assert target_molecule is not None
    target = Chem.MolToSmiles(
        target_molecule,
        canonical=True,
        isomericSmiles=True,
    )
    candidates = decompose_structure(
        {
            "r0_structure_id": "A19B1C1",
            "canonical_isomeric_smiles": target,
            "observed_source_ids": "agile_measured1200",
            "study_split_groups_json": "{}",
        },
        "source_study",
        (reaction,),
        1000,
        1000,
    )

    assert len(candidates) == 1
    audit = _forward_site_audit(
        reaction,
        candidates[0],
        target,
        max_outcomes=100,
    )
    assert audit["raw_forward_outcomes"] == 2
    assert audit["unique_forward_products"] == 2
    assert audit["target_matching_forward_outcomes"] == 1
    assert audit["distinct_target_reacting_sites"] == 1
    assert len(audit["reacting_amine_sites"]) == 1
    assert ":1" in audit["reacting_amine_sites"][0]["atom_mapped_amine_smiles"]


def test_rejects_hash_mismatch_before_decomposition(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text())
    config["inputs"]["virtual_smiles"]["expected_sha256"] = "0" * 64
    bad_config = tmp_path / "config.json"
    bad_config.write_text(json.dumps(config))

    with pytest.raises(Ugi3VirtualCapabilityError, match="hash mismatch"):
        build_ugi3_virtual_capability(
            bad_config,
            *_production_paths()[1:],
        )


def test_writer_is_deterministic_for_committed_payloads(tmp_path: Path) -> None:
    result = json.loads(RESULT.read_text())
    artifacts = {
        name: (REPO / details["path"]).read_bytes() for name, details in result["artifacts"].items()
    }

    write_ugi3_virtual_capability(result, artifacts, tmp_path)
    first = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    write_ugi3_virtual_capability(result, artifacts, tmp_path)
    second = {path.name: path.read_bytes() for path in tmp_path.iterdir()}

    assert first == second
    assert not list(tmp_path.glob(".*.tmp"))


def test_committed_census_maps_all_products_without_inventing_l2() -> None:
    if not RESULT.exists():
        pytest.fail("committed AGILE virtual capability artifact is missing")
    result = json.loads(RESULT.read_text())

    assert result["schema_version"] == "m0_09_agile_virtual_ugi3_capability.v1"
    assert result["summary"]["source_products"] == 12276
    assert result["summary"]["products_with_exact_qualified_ugi_decomposition"] == 12276
    assert result["summary"]["cartesian_component_tuples"] == 12276
    assert result["summary"]["unique_components_total"] == 93
    assert result["summary"]["roles"]["amine_head"]["unique_components"] == 22
    assert result["summary"]["roles"]["oxoester_aldehyde_body_tail"]["unique_components"] == 62
    assert result["summary"]["roles"]["isocyanide_tail"]["unique_components"] == 9
    assert (
        result["summary"]["roles"]["oxoester_aldehyde_body_tail"][
            "components_with_exact_source_route"
        ]
        == 17
    )
    assert result["summary"]["roles"]["isocyanide_tail"]["components_with_exact_source_route"] == 7
    assert result["summary"]["products_with_complete_l2_l3_candidate"] == 0
    assert result["claims_boundary"]["unlabeled_products_are_l2_supervision"] is False
    for details in result["artifacts"].values():
        path = REPO / details["path"]
        assert path.stat().st_size == details["bytes"]
        assert sha256_file(path) == details["sha256"]
    product_path = REPO / result["artifacts"]["agile_virtual_ugi3_product_ledger.csv.gz"]["path"]
    with gzip.open(product_path, "rt", newline="") as handle:
        product_rows = list(csv.DictReader(handle))
    assert len(product_rows) == 12276
    assert all(int(row["candidate_count"]) == 1 for row in product_rows)
    assert all(
        json.loads(row["candidate_routes_json"])[0]["distinct_target_reacting_sites"] == 1
        for row in product_rows
    )
