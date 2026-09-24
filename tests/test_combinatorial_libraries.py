from __future__ import annotations

import csv
import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from forge.assembly.families import LibraryAssemblyError, load_assembly_libraries
from forge.core.hashing import sha256_file
from forge.corpus.combinatorial_libraries import (
    qualify_combinatorial_libraries,
    read_library_records,
    tensorize_library_record,
)
from forge.model.vocabulary import load_atom_vocabulary

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/multireaction/combinatorial_libraries_v1.json"


@pytest.fixture(scope="module")
def source():
    config = json.loads(CONFIG.read_text())
    inputs = config["inputs"]
    libraries = load_assembly_libraries(
        [(REPO / inputs[key]["path"], inputs[key]["sha256"]) for key in config["registries"]],
        expected_families=config["expected_rows_by_family"],
    )
    blocks = {
        row["block_id"]: row
        for row in json.loads((REPO / inputs["building_blocks"]["path"]).read_text())["blocks"]
    }
    return config, libraries, blocks


def test_all_twelve_families_reach_existing_whole_graph_tensorizer(source):
    config, libraries, blocks = source
    inputs = config["inputs"]
    vocabulary = load_atom_vocabulary(REPO / inputs["atom_vocabulary"]["path"])
    seen = set()
    for record in read_library_records(REPO / inputs["corpus"]["path"], libraries, blocks):
        if record.reaction_id in seen:
            continue
        graph = tensorize_library_record(record, vocabulary)
        assert graph.node_count > 0
        assert record.realism_weight >= 0
        seen.add(record.reaction_id)
        if seen == set(libraries):
            break
    assert seen == set(libraries)


@pytest.mark.parametrize(
    "field,value",
    [
        ("realism_weight", "nan"),
        ("realism_weight", "inf"),
        ("realism_weight", "-1"),
        ("reaction_family", "unknown"),
        ("reactant_roles", "wrong_role"),
        ("reactant_ids", "absent|absent|absent"),
    ],
)
def test_bad_source_metadata_fails_closed(source, tmp_path, field, value):
    config, libraries, blocks = source
    with (REPO / config["inputs"]["corpus"]["path"]).open() as handle:
        row = next(csv.DictReader(handle))
    row[field] = value
    path = tmp_path / "bad.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=row)
        writer.writeheader()
        writer.writerow(row)
    with pytest.raises(LibraryAssemblyError):
        list(read_library_records(path, libraries, blocks))


def test_source_weight_is_preserved_without_raw_family_frequency_substitution(source):
    config, libraries, blocks = source
    path = REPO / config["inputs"]["corpus"]["path"]
    record = next(read_library_records(path, libraries, blocks))
    with path.open() as handle:
        raw = next(csv.DictReader(handle))
    assert record.realism_weight == float(raw["realism_weight"])


def test_zero_weight_rows_are_retained_with_zero_sampling_mass(source, tmp_path):
    config, libraries, blocks = source
    with (REPO / config["inputs"]["corpus"]["path"]).open() as handle:
        row = next(csv.DictReader(handle))
    row["realism_weight"] = "0"
    path = tmp_path / "zero.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=row)
        writer.writeheader()
        writer.writerow(row)
    records = list(read_library_records(path, libraries, blocks))
    assert len(records) == 1 and records[0].realism_weight == 0


def test_vocabulary_failure_does_not_remove_sulfur_or_shrink_product(source):
    config, libraries, blocks = source
    vocabulary = load_atom_vocabulary(REPO / config["inputs"]["atom_vocabulary"]["path"])
    record = next(
        read_library_records(REPO / config["inputs"]["corpus"]["path"], libraries, blocks)
    )
    sulfur = replace(record, product_smiles="CCSCC")
    assert tensorize_library_record(sulfur, vocabulary).node_count == 5
    with pytest.raises((KeyError, ValueError, RuntimeError)):
        tensorize_library_record(
            sulfur, tuple(state for state in vocabulary if state.symbol != "S")
        )


def test_pinned_inputs_and_full_population_are_explicit(source):
    config, libraries, _ = source
    assert len(libraries) == 12
    assert sum(config["expected_rows_by_family"].values()) == 464265
    assert config["sampling_weight"] == "realism_weight"
    for pin in config["inputs"].values():
        assert sha256_file(REPO / pin["path"]) == pin["sha256"]


@pytest.fixture
def tiny_repo(source, tmp_path):
    config, _, _ = source
    config = json.loads(json.dumps(config))
    first_rows = {}
    with (REPO / config["inputs"]["corpus"]["path"]).open() as handle:
        for row in csv.DictReader(handle):
            if float(row["realism_weight"]) > 0:
                first_rows.setdefault(row["reaction_family"], row)
    for name, pin in config["inputs"].items():
        path = tmp_path / pin["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if name == "corpus":
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=next(iter(first_rows.values())))
                writer.writeheader()
                writer.writerows(first_rows.values())
        else:
            shutil.copyfile(REPO / pin["path"], path)
        pin["sha256"] = str(sha256_file(path))
    config.update(workers=1, representation_samples_per_family=2)
    config["expected_rows_by_family"] = {family: 1 for family in first_rows}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    return tmp_path, config_path


def test_complete_run_replays_deterministically_and_preserves_provenance(tiny_repo):
    repo, config = tiny_repo
    first = qualify_combinatorial_libraries(repo, config, Path("out1"))
    second = qualify_combinatorial_libraries(repo, config, Path("out2"))
    assert first["status"] == "pass"
    assert first["summary"]["families"] == 12
    assert first["summary"]["rows"] == 12
    assert first["calls"] == {"training": 0, "molecular_generation": 0, "remote_compute": 0}
    assert first["families"] == second["families"]
    assert first["representation_panel"] == second["representation_panel"]
    assert (
        first["artifacts"]["row_qualification"]["sha256"]
        == second["artifacts"]["row_qualification"]["sha256"]
    )
    assert first["config"]["sha256"] == sha256_file(config)
    assert all(sha256_file(REPO / p) == digest for p, digest in first["sources"].items())
    with pytest.raises(LibraryAssemblyError, match="already exists"):
        qualify_combinatorial_libraries(repo, config, Path("out1"))


def test_missing_rows_block_admission_with_saved_negative_result(tiny_repo):
    repo, config_path = tiny_repo
    config = json.loads(config_path.read_text())
    config["expected_rows_by_family"]["ugi_3cr_agile"] += 1
    config_path.write_text(json.dumps(config))
    result = qualify_combinatorial_libraries(repo, config_path, Path("blocked"))
    assert result["status"] == "blocked"
    assert not result["gates"]["all_declared_rows_scanned"]
    assert (repo / "blocked/result.json").is_file()


def test_changed_pin_or_bad_weight_policy_publishes_no_result(tiny_repo):
    from forge.core.hashing import PinError

    repo, config_path = tiny_repo
    config = json.loads(config_path.read_text())
    config["sampling_weight"] = "raw_family_count"
    config_path.write_text(json.dumps(config))
    with pytest.raises(LibraryAssemblyError, match="sampling policy"):
        qualify_combinatorial_libraries(repo, config_path, Path("bad"))
    config["sampling_weight"] = "realism_weight"
    config["inputs"]["corpus"]["sha256"] = "0" * 64
    config_path.write_text(json.dumps(config))
    with pytest.raises(PinError):
        qualify_combinatorial_libraries(repo, config_path, Path("bad"))
    assert not (repo / "bad").exists()
