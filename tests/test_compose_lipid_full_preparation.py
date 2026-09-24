"""A full-universe ledger must not manufacture a new training population."""

import json
import sqlite3

import pytest

from forge.assembly.compose_lipid import ComposeLipidError
from forge.corpus import compose_lipid_full_preparation as module


def prior(target="target", family="family", excluded=False):
    return dict(
        target_id=target,
        family=family,
        constitution_id="identity-" + target,
        eligible_for_program_preparation=not excluded,
        training_admitted=False,
    )


def record(**changes):
    values = dict(
        target="target",
        family="family",
        old="train",
        corrected="train",
        instances=[["head", "head", 1], ["tail", "tail", 2]],
        blocked=set(),
        prior=prior(),
        alias=False,
        combination=False,
        held_study=False,
        source_anchor=False,
        study_metadata=[],
        atoms=150,
        source_line=1,
        basis="computed",
        historical=False,
        formal_families={"family"},
    )
    values.update(changes)
    return module.classify_full_record(**values)


def test_existing_train_retains_large_size_roles_quantities_and_never_training():
    result = record()
    assert result["eligible_for_program_preparation"]
    assert result["source_declared_heavy_atoms"] == 150
    assert result["component_instances"][-1] == ["tail", "tail", 2]
    assert not result["training_admitted"]


def test_unassigned_is_accounted_without_a_molecule_or_new_split():
    result = record(old=None, corrected=None, prior=None)
    assert result["disposition"] == "unresolved_partition_or_study"
    assert result["old_split"] == result["corrected_split"] == "unassigned"
    assert result["pending_reasons"] == ["unassigned_partition"]
    assert result["constitution_id"] is None


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"old": "heldout", "prior": None}, "old_split_not_train"),
        ({"corrected": "test"}, "corrected_split_not_train"),
        ({"alias": True}, "exact_source_alias_of_protected_product"),
        ({"combination": True}, "exact_component_combination_of_protected_record"),
        ({"held_study": True}, "selected_protected_source_study"),
        ({"blocked": {"tail"}}, "known_protected_component"),
        ({"prior": prior(excluded=True)}, "prior_protected_product"),
    ],
)
def test_every_protection_remains_binding(change, reason):
    result = record(**change)
    assert not result["eligible_for_program_preparation"]
    assert reason in result["exclusion_reasons"]


def test_missing_study_for_reported_source_is_unresolved_not_negative():
    result = record(source_anchor=True)
    assert result["pending_reasons"] == ["source_study_identity_unresolved"]
    assert not result["exclusion_reasons"]
    assert not result["eligible_for_program_preparation"]
    assert record(source_anchor=True, study_metadata=["123"])["eligible_for_program_preparation"]


def test_known_protection_and_unassigned_reason_both_survive():
    result = record(old=None, corrected=None, prior=None, blocked={"tail"})
    assert result["exclusion_reasons"] == ["known_protected_component"]
    assert result["pending_reasons"] == ["unassigned_partition"]


@pytest.mark.parametrize(
    "change", [{"old": None}, {"corrected": None}, {"atoms": 0}, {"atoms": True}]
)
def test_invalid_membership_or_size_fails(change):
    with pytest.raises(ComposeLipidError):
        record(**change)


def test_sql_propagates_cross_family_aliases_components_combinations_and_studies(tmp_path):
    path = tmp_path / "targets.sqlite"
    cases = [
        # target, family, product token, old, corrected, head, source anchor, pmids
        ("safe", "family", "safe", "train", "train", "safe-head", 0, []),
        ("held", "family", "same", "heldout", "train", "held-head", 0, []),
        ("alias", "other", "same", None, None, "alias-head", 0, []),
        ("combination", "family", "other-product", None, None, "held-head", 0, []),
        ("component", "family", "component-product", None, None, "blocked", 0, []),
        ("component-alias", "other", "component-product", "train", "train", "alternate", 0, []),
        ("study", "family", "study-product", "train", "train", "study-head", 1, ["123"]),
        ("study-alias", "other", "study-product", None, None, "other-head", 0, []),
        ("historical", "family", "old-source", None, None, "historical-head", 0, []),
    ]
    with sqlite3.connect(path) as db:
        db.executescript(
            "CREATE TABLE targets(target_id TEXT PRIMARY KEY,family TEXT,constitution TEXT,source_anchor INT,heavy_atoms INT,payload TEXT); CREATE TABLE assignments(target_id TEXT PRIMARY KEY,family TEXT,forge_split TEXT);"
        )
        for target, family, token, old, corrected, head, anchor, pmids in cases:
            db.execute(
                "INSERT INTO targets VALUES (?,?,?,?,?,?)",
                (target, family, token, anchor, 150, "FORBIDDEN PRODUCT PAYLOAD"),
            )
            if old:
                db.execute("INSERT INTO assignments VALUES (?,?,?)", (target, family, old))
    with sqlite3.connect(":memory:") as db:
        db.executescript(
            "CREATE TABLE constructions(target_id TEXT PRIMARY KEY,family TEXT,instances TEXT,source_line INT,basis TEXT,historical_claim TEXT); CREATE TABLE assignments(target_id TEXT PRIMARY KEY,family TEXT,split TEXT); CREATE TABLE components(target_id TEXT PRIMARY KEY,pmids TEXT);"
        )
        for i, (target, family, token, old, corrected, head, anchor, pmids) in enumerate(cases):
            db.execute(
                "INSERT INTO constructions VALUES (?,?,?,?,?,?)",
                (target, family, json.dumps([["head", head, 1]]), i, "computed", "false"),
            )
            if corrected:
                db.execute("INSERT INTO assignments VALUES (?,?,?)", (target, family, corrected))
                db.execute("INSERT INTO components VALUES (?,?)", (target, json.dumps(pmids)))
        db.execute("ATTACH DATABASE ? AS original", (str(path),))
        previous = {
            target: prior(target, family) for target, family, _, old, *_ in cases if old == "train"
        }
        observed = list(
            module.full_records(
                db,
                {"blocked"},
                previous,
                [{"constitution": "old-source", "prior_development_split": "test"}],
                {
                    "selected_source_studies": ["pmid:123"],
                    "formal_evaluation_families": ["family", "other"],
                },
            )
        )
    results = {r["target_id"]: r for r in observed}
    assert set(results) == {case[0] for case in cases}
    assert [r["target_id"] for r in observed if r["eligible_for_program_preparation"]] == ["safe"]
    for target in ("alias", "component-alias", "study-alias", "historical"):
        assert "exact_source_alias_of_protected_product" in results[target]["exclusion_reasons"]
    assert (
        "exact_component_combination_of_protected_record"
        in results["combination"]["exclusion_reasons"]
    )
    assert "selected_protected_source_study" in results["study"]["exclusion_reasons"]
    assert all("constitution" not in row and "payload" not in row for row in observed)


def test_training_reader_rejects_even_without_loading_a_dataset():
    with pytest.raises(ComposeLipidError, match="not a qualified training"):
        module.FullPreparationCorpus.__new__(module.FullPreparationCorpus).iter_training_records()
