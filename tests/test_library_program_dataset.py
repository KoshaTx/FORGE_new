from __future__ import annotations

import csv
import gzip
import json
import shutil
from pathlib import Path

import pytest

from forge.assembly.families import LibraryAssemblyError
from forge.core.hashing import PinError, sha256_file
from forge.corpus.combinatorial_libraries import qualify_combinatorial_libraries
from forge.corpus.library_program_dataset import build_library_program_dataset

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def source_rows():
    rows = {}
    with (REPO / "data/vendor/r1_reaction_enumerated_support_v1.csv").open() as handle:
        for row in csv.DictReader(handle):
            if float(row["realism_weight"]) > 0:
                rows.setdefault(row["reaction_family"], row)
    return list(rows.values())


@pytest.fixture
def dataset_repo(tmp_path, source_rows):
    single = json.loads(
        (REPO / "configs/multireaction/combinatorial_libraries_v1.json").read_text()
    )
    # Real, previously qualified rows from every family, including duplicate source provenance
    # and a family whose original source weight is zero in this fixture.
    rows = [dict(r) for r in source_rows]
    for row in rows:
        if row["reaction_family"] == "urea_amine_isocyanate":
            row["realism_weight"] = "0"
    rows.append(dict(rows[0]))
    for name, pin in single["inputs"].items():
        dest = tmp_path / pin["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        if name == "corpus":
            with dest.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)
        else:
            shutil.copyfile(REPO / pin["path"], dest)
        pin["sha256"] = str(sha256_file(dest))
    single.update(workers=1, representation_samples_per_family=1)
    single["expected_rows_by_family"] = {r["reaction_family"]: 1 for r in rows}
    single["expected_rows_by_family"][rows[0]["reaction_family"]] += 1
    config_path = tmp_path / "single.json"
    config_path.write_text(json.dumps(single))
    result = qualify_combinatorial_libraries(tmp_path, config_path, Path("single"))
    config = json.loads(
        (REPO / "configs/multireaction/library_program_dataset_v1.json").read_text()
    )
    config["inputs"] = {k: v for k, v in single["inputs"].items() if k != "atom_vocabulary"}
    for name, path in [
        ("single_event_result", "single/result.json"),
        ("single_event_ledger", result["artifacts"]["row_qualification"]["path"]),
    ]:
        config["inputs"][name] = {"path": path, "sha256": str(sha256_file(tmp_path / path))}
    # Keep the fixture cohort in train using explicit immutable metadata. Real-run historical
    # protections are independently hash-pinned in the shipped configuration.
    blocks = json.loads((tmp_path / config["inputs"]["building_blocks"]["path"]).read_text())[
        "blocks"
    ]
    protected = tmp_path / "frozen.csv"
    with protected.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["smiles", "fold"])
        writer.writerows((b["canonical_smiles"], "train") for b in blocks)
    config["inputs"]["fixture_folds"] = {
        "path": "frozen.csv",
        "sha256": str(sha256_file(protected)),
    }
    config["frozen_identity_sources"] = [
        {"input": "fixture_folds", "identity_field": "smiles", "fold_fields": ["fold"]}
    ]
    config["workers"] = 1
    config["expected_rows_by_family"] = single["expected_rows_by_family"]
    config_path = tmp_path / "programs.json"
    config_path.write_text(json.dumps(config))
    return tmp_path, config_path


def test_full_pipeline_retains_all_rows_deduplicates_weight_and_replays_deterministically(
    dataset_repo,
):
    repo, config = dataset_repo
    first = build_library_program_dataset(repo, config, Path("a"))
    second = build_library_program_dataset(repo, config, Path("b"))
    assert first["summary"]["source_rows"] == 13
    assert first["summary"]["unique_products"] == 12
    assert first["summary"] == second["summary"]
    assert first["programs"] == second["programs"]
    assert first["gates"]["all_source_rows_preserved"]
    assert first["gates"]["all_source_rows_have_unique_registry_qualified_programs"]
    assert first["gates"]["no_identity_overlap_across_active_partitions"]
    assert not first["gates"]["all_families_have_eligible_train_calibration_and_heldout"]
    assert first["status"] == "blocked"  # Do not claim missing partitions passed.
    assert first["training_calls"] == first["generator_sampling_calls"] == 0
    for name, artifact in first["artifacts"].items():
        assert artifact["sha256"] == sha256_file(repo / artifact["path"])
        assert artifact["sha256"] == second["artifacts"][name]["sha256"]
    with gzip.open(repo / "a/products.csv.gz", "rt") as handle:
        products = list(csv.DictReader(handle))
    assert sum(float(p["training_sampling_weight"]) for p in products) == pytest.approx(1)
    assert all(
        float(p["training_sampling_weight"]) == 0
        for p in products
        if p["reaction_family"] == "urea_amine_isocyanate"
    )
    assert all(
        mass == pytest.approx(1 / 11) for mass in first["summary"]["train_family_masses"].values()
    )
    assert first["config"]["sha256"] == sha256_file(config)
    with pytest.raises(LibraryAssemblyError, match="fresh"):
        build_library_program_dataset(repo, config, Path("a"))


def test_historical_product_holdout_excludes_every_duplicate_source_row(dataset_repo, source_rows):
    repo, config_path = dataset_repo
    config = json.loads(config_path.read_text())
    path = repo / "frozen.csv"
    with path.open("a", newline="") as handle:
        csv.writer(handle).writerow([source_rows[0]["canonical_smiles"], "heldout"])
    config["inputs"]["fixture_folds"]["sha256"] = str(sha256_file(path))
    config_path.write_text(json.dumps(config))
    result = build_library_program_dataset(repo, config_path, Path("heldout"))
    assert result["partitions"]["protected_identities_in_train"] == 0
    with gzip.open(repo / "heldout/products.csv.gz", "rt") as handle:
        duplicate = next(p for p in csv.DictReader(handle) if p["source_row_count"] == "2")
    assert duplicate["fold"] == "quarantine"
    assert float(duplicate["training_sampling_weight"]) == 0


@pytest.mark.parametrize("defect", ["pin", "ledger", "population", "weight_policy"])
def test_invalid_contract_or_tampered_inputs_never_publish_a_result(dataset_repo, defect):
    repo, config_path = dataset_repo
    config = json.loads(config_path.read_text())
    if defect == "pin":
        config["inputs"]["corpus"]["sha256"] = "0" * 64
    elif defect == "ledger":
        with gzip.open(repo / config["inputs"]["single_event_ledger"]["path"], "at") as handle:
            handle.write("{}\n")
        config["inputs"]["single_event_ledger"]["sha256"] = str(
            sha256_file(repo / config["inputs"]["single_event_ledger"]["path"])
        )
    elif defect == "population":
        config["expected_rows_by_family"]["ugi_3cr_agile"] += 1
    else:
        config["weight_policy"] = "raw_family_count"
    config_path.write_text(json.dumps(config))
    with pytest.raises((LibraryAssemblyError, PinError)):
        build_library_program_dataset(repo, config_path, Path("bad"))
    assert not (repo / "bad").exists()
