from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import pytest
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from forge.corpus.r1_prime_audit import (
    AuditError,
    ProductIndex,
    _audit_scheme_worker,
    _read_r0_rows,
    build_block_records,
    compile_reactions,
    decompose_scheme,
    decompose_structure,
    enumerate_r1_prime,
    load_config,
    load_r1_control,
    load_reaction_definitions,
    nearest_reachable_distances,
    reproduce_non_agile_control,
    sha256_file,
    validate_registry_examples,
    verify_inputs,
)

REPO = Path(__file__).resolve().parents[1]
VENDOR = REPO / "data/vendor"
RESULTS = REPO / "results/m0_04"
CONSTITUTIONAL_RESULTS = REPO / "results/m0_04_constitutional"
REGISTRIES = (
    VENDOR / "qualified_reaction_families_v1.json",
    VENDOR / "qualified_reactions_v1.json",
)
CONFIG = REPO / "configs/corpus/m0_04_r1_prime_audit.json"
CONSTITUTIONAL_CONFIG = REPO / "configs/corpus/m0_04_r1_prime_audit_constitutional.json"


def _definitions(*, use_qualified_overrides: bool = True):
    if not all(path.exists() for path in REGISTRIES):
        pytest.skip("M0-04 registry assets are not vendored")
    overrides = load_config(CONFIG)["role_policy_overrides"] if use_qualified_overrides else None
    return load_reaction_definitions(
        REGISTRIES,
        expected_count=12,
        role_policy_overrides=overrides,
    )


def test_all_twelve_registry_transforms_pass_declared_examples() -> None:
    compiled = compile_reactions(_definitions())

    result = validate_registry_examples(compiled, max_forward_outcomes=1000)

    assert len(result) == 12
    assert all(details["positive_examples_passed"] >= 1 for details in result.values())
    assert all(details["negative_examples_passed"] >= 1 for details in result.values())


def test_constitutional_config_preserves_historical_and_current_controls() -> None:
    config, _, locations = verify_inputs(
        CONSTITUTIONAL_CONFIG,
        VENDOR,
        REPO / "data/splits/m0_03_constitutional",
    )
    dataset = config["dataset"]
    r1 = load_r1_control(locations[dataset["r1_control_asset"]])
    historical = _read_r0_rows(
        locations[dataset["historical_r0_asset"]],
        config["expected_historical_r0_rows"],
    )
    current = _read_r0_rows(
        locations[dataset["current_r0_asset"]],
        config["expected_r0_rows"],
    )

    assert (
        reproduce_non_agile_control(
            historical,
            r1,
            config["controls"]["historical_non_agile_rows"],
            config["controls"]["historical_non_agile_recovered"],
        )["recovered"]
        == 108
    )
    assert (
        reproduce_non_agile_control(
            current,
            r1,
            config["controls"]["current_non_agile_rows"],
            config["controls"]["current_non_agile_recovered"],
        )["recovered"]
        == 112
    )


def test_ugi_positive_product_retrodecomposes_and_reconstructs_exactly() -> None:
    definition = next(item for item in _definitions() if item.reaction_id == "ugi_3cr_agile")
    example = definition.known_positive_examples[0]
    row = {
        "r0_structure_id": "R0-test-ugi",
        "canonical_isomeric_smiles": example["expected"],
        "observed_source_ids": "agile_measured1200",
        "study_split_groups_json": "{}",
    }

    candidates = decompose_structure(
        row,
        "source_study",
        compile_reactions((definition,)),
        max_reverse_outcomes=1000,
        max_forward_outcomes=1000,
    )

    expected = tuple(
        Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True, isomericSmiles=True)
        for smiles in example["reactants"]
    )
    assert [candidate.reactant_smiles for candidate in candidates] == [expected]
    assert candidates[0].source_studies == ("fallback-source:agile_measured1200",)


def test_qualified_ugi_policy_recovers_a5_that_raw_match_count_rejects() -> None:
    agile_path = VENDOR / "AGILE_smiles_with_value_group.csv"
    if not agile_path.exists():
        pytest.skip("AGILE measured library is not vendored")
    with agile_path.open(newline="") as handle:
        measured = next(row for row in csv.DictReader(handle) if row["label"] == "A5B1C1")
    row = {
        "r0_structure_id": "R0-test-a5",
        "canonical_isomeric_smiles": measured["combined_mol_SMILES"],
        "observed_source_ids": "agile_measured1200",
        "study_split_groups_json": "{}",
    }
    qualified_definition = next(
        item for item in _definitions() if item.reaction_id == "ugi_3cr_agile"
    )
    raw_definition = next(
        item
        for item in _definitions(use_qualified_overrides=False)
        if item.reaction_id == "ugi_3cr_agile"
    )

    qualified = decompose_structure(
        row,
        "source_study",
        compile_reactions((qualified_definition,)),
        max_reverse_outcomes=1000,
        max_forward_outcomes=1000,
    )
    raw = decompose_structure(
        row,
        "source_study",
        compile_reactions((raw_definition,)),
        max_reverse_outcomes=1000,
        max_forward_outcomes=1000,
    )

    assert len(qualified) == 1
    assert raw == []
    assert (
        qualified_definition.reactant_roles[0].site_multiplicity_semantics
        == "symmetry_distinct_required_handle_matches"
    )


def test_scheme_decomposition_rejects_overlapping_train_and_heldout() -> None:
    with pytest.raises(AuditError, match="train and heldout IDs overlap"):
        decompose_scheme(
            "source_study",
            {},
            {"R0-overlap"},
            {"R0-overlap"},
            _definitions(),
            max_reverse_outcomes=1000,
            max_forward_outcomes=1000,
        )


def test_small_train_derived_pool_enumerates_the_known_ugi_product(tmp_path: Path) -> None:
    definition = next(item for item in _definitions() if item.reaction_id == "ugi_3cr_agile")
    example = definition.known_positive_examples[0]
    row = {
        "r0_structure_id": "R0-test-ugi",
        "canonical_isomeric_smiles": example["expected"],
        "observed_source_ids": "agile_measured1200",
        "study_split_groups_json": "{}",
    }
    candidates = decompose_structure(
        row,
        "source_study",
        compile_reactions((definition,)),
        max_reverse_outcomes=1000,
        max_forward_outcomes=1000,
    )
    blocks = build_block_records(candidates, (definition,))

    index, summary = enumerate_r1_prime(
        "source_study",
        blocks,
        (definition,),
        tmp_path / "products.sqlite",
        max_combinations=10,
        commit_interval=1,
    )
    try:
        expected = Chem.MolToSmiles(
            Chem.MolFromSmiles(example["expected"]),
            canonical=True,
            isomericSmiles=True,
        )
        assert index.contains(expected)
        assert summary["predicted_reactant_combinations"] == 1
        assert summary["sampling_used"] is False
    finally:
        index.close()


def test_scheme_worker_writes_and_reuses_completed_checkpoint(tmp_path: Path) -> None:
    definition = next(item for item in _definitions() if item.reaction_id == "ugi_3cr_agile")
    example = definition.known_positive_examples[0]
    row = {
        "r0_structure_id": "R0-test-ugi",
        "canonical_isomeric_smiles": example["expected"],
        "observed_source_ids": "agile_measured1200",
        "study_split_groups_json": "{}",
    }
    candidates = decompose_structure(
        row,
        "source_study",
        compile_reactions((definition,)),
        max_reverse_outcomes=1000,
        max_forward_outcomes=1000,
    )
    blocks = build_block_records(candidates, (definition,))
    database = tmp_path / "products.sqlite"
    checkpoint = tmp_path / "source_study.json.gz"
    task = (
        "source_study",
        blocks,
        (definition,),
        str(database),
        10,
        1,
        (row,),
        {"all_harvest_sources_are_r0_train": True},
        {
            "radius": 2,
            "bits": 2048,
            "include_chirality": True,
            "reference_batch_size": 10,
        },
        1,
        str(checkpoint),
        "unit-test-signature",
    )

    first = _audit_scheme_worker(task)
    second = _audit_scheme_worker(task)

    assert first == second
    assert first[1]["r1_prime_exact_recovery"]["all_heldout"]["recovered"] == 1
    assert first[2] == []
    assert checkpoint.exists()
    assert not database.exists()


def test_nearest_distance_matches_direct_ecfp4_tanimoto(tmp_path: Path) -> None:
    index = ProductIndex(tmp_path / "products.sqlite")
    index.add("CCO", "test_reaction")
    index.commit()
    query = {
        "r0_structure_id": "R0-query",
        "canonical_isomeric_smiles": "CCN",
        "observed_source_ids": "lnpdb_v1",
    }
    try:
        result = nearest_reachable_distances(
            (query,),
            index,
            radius=2,
            bits=2048,
            include_chirality=True,
            reference_batch_size=10,
            workers=1,
        )
    finally:
        index.close()

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=True
    )
    expected_similarity = DataStructs.TanimotoSimilarity(
        generator.GetFingerprint(Chem.MolFromSmiles("CCN")),
        generator.GetFingerprint(Chem.MolFromSmiles("CCO")),
    )
    assert result[0]["max_tanimoto_similarity"] == pytest.approx(expected_similarity)
    assert result[0]["nearest_reachable_distance"] == pytest.approx(1 - expected_similarity)


def test_completed_m0_04_artifacts_match_result_manifest() -> None:
    result_path = RESULTS / "result.json"
    if not result_path.exists():
        pytest.skip("M0-04 result has not been generated")
    result = json.loads(result_path.read_text())

    assert result["schema_version"] == "m0_04_r1_prime_audit.v2"
    assert result["control"]["existing_r1_non_agile_full_r0"]["reproduced"] is True
    assert result["control"]["existing_r1_non_agile_full_r0"]["recovered"] == 108
    assert result["control"]["existing_r1_non_agile_full_r0"]["denominator"] == 14_233
    assert result["scientific_status"]["heldout_structures_decomposed_for_harvesting"] == 0
    assert result["scientific_status"]["all_harvest_sources_are_r0_train"] is True
    expected_recovery = {
        "source_study": (2_333, 0),
        "headgroup": (2_315, 273),
        "linker_scaffold": (2_315, 443),
        "component_family": (1_200, 200),
    }
    for scheme, details in result["schemes"].items():
        assert scheme in {
            "source_study",
            "headgroup",
            "linker_scaffold",
            "component_family",
        }
        observed = details["r1_prime_exact_recovery"]["all_heldout"]
        assert (observed["denominator"], observed["recovered"]) == expected_recovery[scheme]
        assert details["decomposition"]["all_harvest_sources_are_r0_train"] is True
        assert details["r1_prime_enumeration"]["sampling_used"] is False
        assert (
            details["r1_prime_nearest_reachable_distance"]["all_unrecovered"]["count"]
            == details["r1_prime_exact_recovery"]["all_heldout"]["denominator"]
            - details["r1_prime_exact_recovery"]["all_heldout"]["recovered"]
        )
    for artifact in result["artifacts"].values():
        path = REPO / artifact["path"]
        assert path.stat().st_size == artifact["bytes"]
        assert sha256_file(path) == artifact["sha256"]

    with gzip.open(RESULTS / "component_pool.csv.gz", "rt", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert {row["provenance"] for row in rows} == {"r0_derived"}


def test_completed_constitutional_m0_04_artifacts_match_result_manifest() -> None:
    result_path = CONSTITUTIONAL_RESULTS / "result.json"
    if not result_path.exists():
        pytest.skip("constitutional M0-04 result has not been generated")
    result = json.loads(result_path.read_text())

    assert result["schema_version"] == "m0_04_r1_prime_audit.v3"
    assert result["control"]["historical_existing_r1_non_agile_full_r0"] == {
        "denominator": 14_233,
        "expected_denominator": 14_233,
        "expected_recovered": 108,
        "recovered": 108,
        "recovery_rate": pytest.approx(108 / 14_233),
        "reproduced": True,
    }
    assert result["control"]["current_existing_r1_non_agile_full_r0"] == {
        "denominator": 14_129,
        "expected_denominator": 14_129,
        "expected_recovered": 112,
        "recovered": 112,
        "recovery_rate": pytest.approx(112 / 14_129),
        "reproduced": True,
    }
    assert result["parameters"]["fingerprint"]["include_chirality"] is False
    assert result["scientific_status"]["heldout_structures_decomposed_for_harvesting"] == 0
    assert result["scientific_status"]["all_harvest_sources_are_r0_train"] is True
    expected_recovery = {
        "source_study": (2_328, 0),
        "headgroup": (2_285, 260),
        "linker_scaffold": (2_285, 402),
        "component_family": (1_100, 100),
    }
    for scheme, details in result["schemes"].items():
        assert scheme in expected_recovery
        observed = details["r1_prime_exact_recovery"]["all_heldout"]
        assert (observed["denominator"], observed["recovered"]) == expected_recovery[scheme]
        assert details["decomposition"]["all_harvest_sources_are_r0_train"] is True
        assert details["r1_prime_enumeration"]["sampling_used"] is False
        assert (
            details["r1_prime_nearest_reachable_distance"]["all_unrecovered"]["count"]
            == observed["denominator"] - observed["recovered"]
        )
    for artifact in result["artifacts"].values():
        path = REPO / artifact["path"]
        assert path.stat().st_size == artifact["bytes"]
        assert sha256_file(path) == artifact["sha256"]

    with gzip.open(CONSTITUTIONAL_RESULTS / "component_pool.csv.gz", "rt", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert {row["provenance"] for row in rows} == {"r0_derived"}
