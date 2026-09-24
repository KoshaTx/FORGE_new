"""Protect partition, provenance and multiplicity boundaries in the unified reader."""

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from forge.assembly.compose_lipid import ComposeLipidError
from forge.core.hashing import PinError
from forge.corpus.compose_lipid_evidence_index import (
    CONFIG_SCHEMA,
    INCREMENT_SCHEMA,
    EvidencePreparationCorpus,
    append_evidence,
    build_evidence_index,
    census,
    create_index,
)
from forge.corpus.compose_lipid_full_partition import POLICY, RESULT_SCHEMA
from forge.corpus.compose_lipid_source_view import dump, pin


@pytest.fixture
def db():
    db = sqlite3.connect(":memory:")
    db.executescript(
        "ATTACH ':memory:' AS population;"
        "CREATE TABLE population.partition(target_id TEXT PRIMARY KEY,family TEXT,"
        "constitution_id TEXT,heavy_atoms INTEGER,disposition TEXT);"
        "INSERT INTO population.partition VALUES ('a','f','identity-a',254,'eligible'),"
        "('b','f','identity-b',120,'eligible'),('protected','f','identity-c',50,'protected');"
        "CREATE VIEW population.eligible AS SELECT * FROM partition WHERE disposition='eligible';"
        "ATTACH ':memory:' AS incoming;"
        "CREATE TABLE incoming.exact(target_id TEXT,family TEXT,source_line INTEGER);"
        "CREATE TABLE incoming.new_evidence(target_id TEXT,family TEXT,exact INTEGER,"
        "disposition TEXT,shard TEXT,source_line INTEGER);"
    )
    create_index(db)
    yield db
    db.close()


def test_historical_exact_protection_and_complete_size_support(db):
    db.execute("INSERT INTO incoming.exact VALUES ('a','f',1),('protected','f',2)")
    result = append_evidence(db, receipt="old", historical=True)
    assert result == {
        "source_rows": 2,
        "eligible_attempts": 1,
        "exact": 1,
        "nonexact": 0,
        "historical_excluded": 1,
    }
    assert census(db) == {"f": {"eligible": 2, "exact": 1, "pending": 1}}
    assert db.execute("SELECT heavy_atoms FROM exact").fetchall() == [(254,)]
    assert db.execute("SELECT target_id FROM pending").fetchall() == [("b",)]


def test_later_success_preserves_failed_attempt_without_duplicate_weight(db):
    db.execute("INSERT INTO incoming.new_evidence VALUES ('a','f',0,'ambiguous','s1',1)")
    append_evidence(db, receipt="first", historical=False)
    db.execute(
        "UPDATE incoming.new_evidence SET exact=1,disposition='exact_computed_reconstruction'"
    )
    append_evidence(db, receipt="second", historical=False)
    assert census(db)["f"] == {"eligible": 2, "exact": 1, "pending": 1}
    assert db.execute("SELECT exact FROM evidence ORDER BY receipt").fetchall() == [(0,), (1,)]
    with pytest.raises(ComposeLipidError, match="Duplicate exact"):
        append_evidence(db, receipt="duplicate", historical=False)


@pytest.mark.parametrize(
    "row",
    [
        ("missing", "f", 1),
        ("a", "wrong-family", 1),
        ("a", "f", 0),
        ("a", "f", None),
    ],
)
def test_historical_malformed_rows_do_not_disappear_in_join(db, row):
    db.execute("INSERT INTO incoming.exact VALUES (?,?,?)", row)
    with pytest.raises(ComposeLipidError, match="Invalid evidence"):
        append_evidence(db, receipt="bad", historical=True)


@pytest.mark.parametrize(
    "row",
    [
        ("protected", "f", 1, "exact_computed_reconstruction", "s", 1),
        ("a", "f", 1, "ambiguous", "s", 1),
        ("a", "f", 0, "exact_computed_reconstruction", "s", 1),
        ("a", "f", None, "failed", "s", 1),
        ("a", "f", 1, "exact_computed_reconstruction", None, 1),
    ],
)
def test_incremental_protection_and_evidence_disagreement_fail_closed(db, row):
    db.execute("INSERT INTO incoming.new_evidence VALUES (?,?,?,?,?,?)", row)
    with pytest.raises(ComposeLipidError):
        append_evidence(db, receipt="bad", historical=False)


@pytest.fixture
def prepared(tmp_path):
    """Tiny authenticated files exercise the public builder and reader without a corpus copy."""
    repo = Path(__file__).resolve().parents[1]
    # Real implementation pins must resolve inside the fixture repository.
    implementation = tmp_path / "implementation.py"
    implementation.write_bytes((repo / "forge/corpus/compose_lipid_evidence_index.py").read_bytes())
    smiles = "C" * 120
    identity = hashlib.sha256(smiles.encode()).hexdigest()
    partition = tmp_path / "partition.sqlite"
    with sqlite3.connect(partition) as db:
        db.executescript(
            "CREATE TABLE partition(target_id TEXT PRIMARY KEY,family TEXT,constitution_id TEXT,"
            "heavy_atoms INTEGER,disposition TEXT);"
            "CREATE VIEW eligible AS SELECT * FROM partition WHERE disposition='eligible';"
        )
        db.execute("INSERT INTO partition VALUES ('a','f',?,120,'eligible')", (identity,))
    corpus = tmp_path / "corpus.sqlite"
    with sqlite3.connect(corpus) as db:
        db.execute("CREATE TABLE targets(target_id TEXT PRIMARY KEY,family TEXT,payload TEXT)")
        db.execute(
            "INSERT INTO targets VALUES ('a','f',?)",
            (
                json.dumps(
                    {
                        "target_id": "a",
                        "constitution": smiles,
                        "heavy_atoms": 120,
                        "primary_metadata": {"mechanism": "original-source"},
                    }
                ),
            ),
        )
    joins = tmp_path / "joins.sqlite"
    instances = [["head", "global-head", 2], ["tail", "global-tail", 3]]
    with sqlite3.connect(joins) as db:
        db.execute(
            "CREATE TABLE constructions(target_id TEXT PRIMARY KEY,family TEXT,"
            "instances TEXT,basis TEXT,source_line INTEGER)"
        )
        db.execute(
            "INSERT INTO constructions VALUES ('a','f',?,'original',17)", (json.dumps(instances),)
        )
    partition_receipt = tmp_path / "partition.json"
    dump(
        partition_receipt,
        {
            "schema_version": RESULT_SCHEMA,
            "policy": POLICY,
            "eligible_view_qualified": True,
            "summary": {"invariant_failures": 0},
            "implementation": {},
            "inputs": {"corpus": pin(tmp_path, corpus), "joins": pin(tmp_path, joins)},
            "artifact": pin(tmp_path, partition),
        },
    )
    evidence = tmp_path / "replay.sqlite"
    with sqlite3.connect(evidence) as db:
        db.executescript(
            "CREATE TABLE new_evidence(target_id TEXT,family TEXT,exact INTEGER,"
            "disposition TEXT,shard TEXT,source_line INTEGER);"
            "INSERT INTO new_evidence VALUES ('a','f',1,"
            "'exact_computed_reconstruction','original-shard.json',9);"
        )
    audit = tmp_path / "audit.json"
    dump(
        audit,
        {
            "schema_version": INCREMENT_SCHEMA,
            "training_admitted": False,
            "training_calls": 0,
            "implementation": pin(tmp_path, implementation),
            "inputs": {},
            "artifact": pin(tmp_path, evidence),
            "summary": {
                "by_family": {
                    "f": {
                        "eligible_preparation_rows": 1,
                        "exact_computed_reconstructions": 1,
                        "eligible_pending_chemistry": 0,
                    }
                }
            },
        },
    )
    config = tmp_path / "config.json"
    dump(
        config,
        {
            "schema_version": CONFIG_SCHEMA,
            "partition": pin(tmp_path, partition_receipt),
            "evidence": [pin(tmp_path, audit)],
            "checkpoint": pin(tmp_path, audit),
        },
    )
    return tmp_path, config, implementation, instances


def test_public_builder_reader_preserves_recipe_and_provenance(prepared, monkeypatch):
    repo, config, implementation, instances = prepared
    monkeypatch.setattr("forge.corpus.compose_lipid_evidence_index.__file__", str(implementation))
    output = repo / "index"
    result = build_evidence_index(repo, config, output)
    reader = EvidencePreparationCorpus(repo, output / "result.json")
    rows = list(reader.iter_preparation_records(family="f"))
    assert result["summary"] == {"eligible": 1, "exact": 1, "pending": 0}
    assert result["duplicate_eligible_constitutions"] == 0
    assert rows[0]["preparation"]["component_instances"] == instances
    assert rows[0]["preparation"]["construction_source_line"] == 17
    assert rows[0]["source"]["heavy_atoms"] == 120
    assert rows[0]["evidence"]["source_line"] == 9
    assert rows[0]["evidence"]["receipt"] == json.loads(config.read_text())["checkpoint"]
    assert rows[0]["preparation"]["training_admitted"] is False
    assert list(reader.iter_preparation_records(family="f", exact=False)) == []
    with pytest.raises(ComposeLipidError, match="training admission"):
        reader.iter_training_records()
    with pytest.raises(ComposeLipidError, match="Unknown eligible family"):
        list(reader.iter_preparation_records(family="wrong"))
    with pytest.raises(ComposeLipidError, match="already exists"):
        build_evidence_index(repo, config, output)
    with (repo / "corpus.sqlite").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(PinError, match="changed"):
        EvidencePreparationCorpus(repo, output / "result.json")


def test_failed_build_publishes_no_output(prepared, monkeypatch):
    repo, config, implementation, _ = prepared
    monkeypatch.setattr("forge.corpus.compose_lipid_evidence_index.__file__", str(implementation))
    value = json.loads(config.read_text())
    value["evidence"].append(value["evidence"][0])
    dump(config, value)
    with pytest.raises(ComposeLipidError, match="repeated evidence"):
        build_evidence_index(repo, config, repo / "index")
    assert not (repo / "index").exists()
