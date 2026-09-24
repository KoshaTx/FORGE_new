"""Both split generations and cross-family precursor exclusions remain binding."""

import json
import sqlite3

import pytest

from forge.assembly.compose_lipid import ComposeLipidError
from forge.corpus import compose_lipid_source_view as module


def prior(**updates):
    return {
        "target_id": "target",
        "family": "family",
        "constitution_id": "identity",
        "eligible_for_program_preparation": True,
        "training_admitted": False,
        **updates,
    }


def classify(old="train", corrected="train", protected=None, previous=None, components=None):
    return module.classify(
        "target",
        "family",
        old,
        corrected,
        (
            [["head", "global-head", 1], ["tail", "global-tail", 2]]
            if components is None
            else components
        ),
        set() if protected is None else protected,
        prior() if previous is None else previous,
    )


def test_only_intersection_can_reach_preparation_and_never_training():
    result = classify()
    assert result["eligible_for_program_preparation"]
    assert result["component_instances"][1][2] == 2
    assert not result["training_admitted"]


@pytest.mark.parametrize("old", ["heldout", "calibration", "quarantine", "reference"])
def test_corrected_train_cannot_release_old_protection(old):
    result = classify(old=old)
    assert result["exclusion_reasons"] == ["old_split_not_train"]
    assert not result["eligible_for_program_preparation"]
    assert result["constitution_id"] is None


@pytest.mark.parametrize("corrected", ["test", "calibration", "reference"])
def test_old_train_cannot_override_corrected_protection(corrected):
    assert classify(corrected=corrected)["exclusion_reasons"] == ["corrected_split_not_train"]


def test_all_exclusion_reasons_survive_and_component_identity_is_global():
    result = classify(
        corrected="test",
        protected={"global-tail"},
        previous=prior(eligible_for_program_preparation=False),
    )
    assert result["exclusion_reasons"] == [
        "corrected_split_not_train",
        "prior_protected_product",
        "known_protected_component",
    ]
    assert result["protected_component_ids"] == ["global-tail"]
    assert not result["eligible_for_program_preparation"]


@pytest.mark.parametrize(
    "change",
    [
        {"constitution_id": None},
        {"family": "other"},
        {"target_id": "other"},
        {"training_admitted": True},
        {"eligible_for_program_preparation": 1},
    ],
)
def test_missing_or_inconsistent_old_protection_is_fatal(change):
    with pytest.raises(ComposeLipidError, match="prior product protection"):
        classify(previous=prior(**change))


def test_absent_old_train_protection_is_fatal():
    with pytest.raises(ComposeLipidError, match="prior product protection"):
        module.classify("target", "family", "train", "train", [["head", "id", 1]], set(), None)


@pytest.mark.parametrize(
    "components",
    [[], [["head", "id", 0]], [["head", "id", True]], [["head", "id", 1], ["head", "id", 1]]],
)
def test_missing_or_invalid_stoichiometry_fails(components):
    with pytest.raises(ComposeLipidError):
        classify(components=components)


@pytest.mark.parametrize("old,corrected", [("unassigned", "train"), ("train", "unassigned")])
def test_unassigned_targets_are_never_implicitly_train(old, corrected):
    with pytest.raises(ComposeLipidError, match="Unknown"):
        classify(old=old, corrected=corrected)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus.sqlite"
    joins = tmp_path / "joins.sqlite"
    with sqlite3.connect(corpus) as db:
        db.execute("CREATE TABLE assignments(target_id TEXT, family TEXT, forge_split TEXT)")
        db.execute("CREATE TABLE targets(target_id TEXT, payload TEXT)")
        for target, fold in [("target", "train"), ("protected", "train"), ("old-held", "heldout")]:
            db.execute("INSERT INTO assignments VALUES (?,?,?)", (target, "family", fold))
            # Invalid JSON proves that the filtered product payload is never interpreted.
            payload = (
                json.dumps({"target_id": target, "constitution": "CCO"})
                if target == "target"
                else "PROTECTED PAYLOAD MUST NOT BE READ"
            )
            db.execute("INSERT INTO targets VALUES (?,?)", (target, payload))
    with sqlite3.connect(joins) as db:
        db.execute("CREATE TABLE assignments(target_id TEXT, family TEXT, split TEXT)")
        db.execute(
            "CREATE TABLE constructions(target_id TEXT, instances TEXT, basis TEXT, source_line INT)"
        )
        for number, (target, split) in enumerate(
            [("target", "train"), ("protected", "test"), ("old-held", "train")], 1
        ):
            db.execute("INSERT INTO assignments VALUES (?,?,?)", (target, "family", split))
            db.execute(
                "INSERT INTO constructions VALUES (?,?,?,?)",
                (target, '[["head","global-head",1]]', "compatible", number),
            )
    config = {"schema_version": module.CONFIG_SCHEMA, "inputs": {}, "policy": module.POLICY}
    module.dump(tmp_path / "config.json", config)
    previous = {"target": prior(), "protected": prior(target_id="protected")}
    intake = {"corrected_split": {"split_counts": {"train": 2, "test": 1}}}
    loaded = (
        config,
        {"corpus": corpus, "precursors": tmp_path / "precursors.gz"},
        joins,
        set(),
        previous,
        intake,
    )
    monkeypatch.setattr(module, "_load", lambda *args: loaded)
    for name in module.IMPLEMENTATION:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("implementation fixture")
    return tmp_path


def test_persisted_view_rechecks_metadata_and_filters_before_reading_products(workspace):
    result = module.build_source_view(workspace, workspace / "config.json", workspace / "view")
    assert result["summary"]["totals"]["eligible_for_program_preparation"] == 1
    reader = module.SourcePreparationCorpus(workspace, workspace / "view/result.json")
    values = list(reader.iter_preparation_records(family="family"))
    assert [r["source"]["target_id"] for r in values] == ["target"]
    with pytest.raises(ComposeLipidError, match="training qualification"):
        reader.iter_training_records()
    with pytest.raises(ComposeLipidError, match="Unknown source family"):
        list(reader.iter_preparation_records(family="missing"))


def test_a_rehashed_modified_ledger_cannot_override_exclusions(workspace):
    import gzip

    module.build_source_view(workspace, workspace / "config.json", workspace / "view")
    ledger = workspace / "view/preparation.jsonl.gz"
    records = list(module.rows(ledger))
    records[0]["eligible_for_program_preparation"] = True
    with gzip.open(ledger, "wt") as stream:
        for row in records:
            stream.write(json.dumps(row) + "\n")
    result_path = workspace / "view/result.json"
    result = json.loads(result_path.read_text())
    result["artifact"] = module.pin(workspace, ledger)
    module.dump(result_path, result)
    with pytest.raises(ComposeLipidError, match="differs"):
        module.SourcePreparationCorpus(workspace, result_path)


def test_receipt_policy_and_implementation_are_authenticated(workspace):
    module.build_source_view(workspace, workspace / "config.json", workspace / "view")
    (workspace / module.IMPLEMENTATION[0]).write_text("changed implementation")
    with pytest.raises(ValueError, match="pinned input.*changed"):
        module.SourcePreparationCorpus(workspace, workspace / "view/result.json")
