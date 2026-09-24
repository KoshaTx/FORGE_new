"""Scientific boundary tests for full-universe split extension."""

import json
import sqlite3

import pytest

from forge.assembly.compose_lipid import ComposeLipidError
from forge.corpus.compose_lipid_full_partition import merge_partition


@pytest.fixture
def db():
    connection = sqlite3.connect(":memory:")
    connection.executescript(
        "ATTACH ':memory:' AS features; ATTACH ':memory:' AS source;"
        "CREATE TABLE features.features(target_id TEXT PRIMARY KEY,family TEXT,constitution_id TEXT,"
        "heavy_atoms INTEGER,old_projection TEXT,source_study_unresolved INTEGER,identity_error TEXT);"
        "CREATE TABLE features.protected_identities(identity TEXT PRIMARY KEY,basis TEXT);"
        "CREATE TABLE source.constructions(target_id TEXT PRIMARY KEY,family TEXT,instances TEXT);"
        "CREATE TABLE corrected(target_id TEXT PRIMARY KEY,family TEXT,fold TEXT,"
        "prior_disposition TEXT,study_unresolved INTEGER);"
        "CREATE TABLE blocked_components(id TEXT PRIMARY KEY);"
    )
    yield connection
    connection.close()


def add(
    db,
    target,
    *,
    identity=None,
    family="f",
    old="train",
    corrected="train",
    prior="unresolved_partition_or_study",
    study=0,
    error=None,
    component="allowed",
    atoms=254,
):
    db.execute(
        "INSERT INTO features.features VALUES (?,?,?,?,?,?,?)",
        (target, family, identity or target, atoms, old, study, error),
    )
    db.execute(
        "INSERT INTO corrected VALUES (?,?,?,?,?)", (target, family, corrected, prior, study)
    )
    db.execute(
        "INSERT INTO source.constructions VALUES (?,?,?)",
        (target, family, json.dumps([["head", component, 3], ["tail", "shared", 2]])),
    )


def dispositions(db):
    return dict(db.execute("SELECT target_id,disposition FROM partition"))


def test_releases_extended_train_only_and_keeps_large_complete_records(db):
    add(db, "new")
    add(db, "previous", prior="eligible_for_program_preparation")
    before = db.execute("SELECT instances FROM source.constructions ORDER BY target_id").fetchall()
    result = merge_partition(db)
    assert result["totals"] == {"rows": 2, "eligible_for_program_preparation": 2}
    assert result["eligible_maximum_heavy_atoms"] == 254
    assert result["eligible_above_96_atoms"] == 2
    assert (
        db.execute("SELECT instances FROM source.constructions ORDER BY target_id").fetchall()
        == before
    )


def test_feature_producer_empty_error_sentinel_means_success(db):
    add(db, "source_success", error="")
    merge_partition(db)
    assert dispositions(db)["source_success"] == "eligible_for_program_preparation"


@pytest.mark.parametrize(
    "field,value",
    [
        ("old", "test"),
        ("old", "calibration"),
        ("old", "reference"),
        ("corrected", "test"),
        ("corrected", "calibration"),
        ("corrected", "reference"),
        ("prior", "protected"),
    ],
)
def test_prior_or_either_frozen_holdout_protects_all_product_aliases(db, field, value):
    add(db, "held", identity="same", **{field: value})
    add(db, "alias", identity="same", family="other")
    add(db, "unrelated")
    merge_partition(db)
    assert dispositions(db) == {
        "held": "protected",
        "alias": "protected",
        "unrelated": "eligible_for_program_preparation",
    }


def test_global_component_holdout_crosses_families_and_propagates_product_alias(db):
    db.execute("INSERT INTO blocked_components VALUES ('held')")
    add(db, "a", component="held", identity="same")
    add(db, "b", component="held", family="other")
    add(db, "c", identity="same", family="third")
    result = merge_partition(db)
    assert set(dispositions(db).values()) == {"protected"}
    assert result["rows_with_protected_components"] == 2
    assert result["newly_protected_via_identity_closure"] == 1
    assert json.loads(
        db.execute("SELECT protected_component_ids FROM partition WHERE target_id='a'").fetchone()[
            0
        ]
    ) == ["held"]


def test_preserves_earlier_corpus_product_holdout(db):
    db.execute("INSERT INTO features.protected_identities VALUES ('old-held','M0')")
    add(db, "alias", identity="old-held")
    merge_partition(db)
    assert dispositions(db)["alias"] == "protected"


@pytest.mark.parametrize("uncertainty", [{"study": 1}, {"error": "invalid_graph"}])
def test_unresolved_identity_aliases_never_enter_preparation(db, uncertainty):
    add(db, "unknown", identity="same", **uncertainty)
    add(db, "alias", identity="same")
    merge_partition(db)
    assert set(dispositions(db).values()) == {"unresolved_partition_or_study"}


def test_protection_takes_precedence_over_missing_study(db):
    add(db, "held", study=1, old="test")
    merge_partition(db)
    assert dispositions(db)["held"] == "protected"


@pytest.mark.parametrize(
    "mutation",
    [
        "DELETE FROM corrected",
        "DELETE FROM source.constructions",
        "UPDATE corrected SET family='different'",
        "UPDATE corrected SET fold='invented'",
        "UPDATE features.features SET old_projection='invented'",
        "UPDATE corrected SET prior_disposition='invented'",
        "UPDATE corrected SET study_unresolved=1",
    ],
)
def test_bad_or_incomplete_join_fails_closed(db, mutation):
    add(db, "a")
    db.execute(mutation)
    with pytest.raises(ComposeLipidError):
        merge_partition(db)
