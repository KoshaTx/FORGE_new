"""Known prior products cannot re-enter preparation under new source IDs or aliases."""

import gzip
import hashlib
import json
import shutil
import sqlite3
from pathlib import Path

import pytest

from forge.assembly.compose_lipid import ComposeLipidError
from forge.core.hashing import sha256_file
from forge.corpus import compose_lipid_protection as module

ROOT = Path(__file__).resolve().parents[1]


def training_row(target, family, string, identity):
    return {
        "target_id": target,
        "family": family,
        "constitution": string,
        "constitution_id": identity,
    }


def test_protection_crosses_changed_source_ids_and_graph_aliases():
    rows = [
        training_row("new", "family_a", "CCO", "a" * 64),
        training_row("alias", "family_b", "OCC", "a" * 64),
    ]
    prior = [{"target_id": "old", "constitution": "CCO", "prior_development_split": "validation"}]
    result = module.classify_prior_products(rows, prior)
    assert len(result) == 2
    assert all(not row["eligible_for_program_preparation"] for row in result)
    assert all(
        row["prior_matches"] == [{"source_target_id": "old", "fold": "validation"}]
        for row in result
    )
    assert all(not row["training_admitted"] for row in result)


def test_same_source_id_without_structure_match_is_not_chemical_identity():
    result = module.classify_prior_products(
        [training_row("id", "family", "CCO", "a" * 64)],
        [{"target_id": "id", "constitution": "CCN", "prior_development_split": "validation"}],
    )
    assert result[0]["eligible_for_program_preparation"]
    assert not result[0]["training_admitted"]
    assert (
        result[0]["preparation_disposition"] == "pending_program_and_global_holdout_qualification"
    )


def test_protected_fold_wins_over_train_under_other_ids():
    rows = [training_row("id", "family", "CCO", "a" * 64)]
    prior = [
        {"target_id": "p" + fold, "constitution": "CCO", "prior_development_split": fold}
        for fold in ("train", "test")
    ]
    assert not module.classify_prior_products(rows, prior)[0]["eligible_for_program_preparation"]
    assert module.classify_prior_products(
        rows, iter(reversed(prior))
    ) == module.classify_prior_products(rows, iter(prior))


@pytest.mark.parametrize("defect", ["duplicate_target", "missing_identity", "conflicting_identity"])
def test_unqualified_import_identities_are_rejected(defect):
    rows = [training_row("id", "family", "CCO", "a" * 64)]
    if defect == "duplicate_target":
        rows.append(rows[0])
    elif defect == "missing_identity":
        rows[0]["constitution_id"] = None
    else:
        rows.append(training_row("other", "family", "CCO", "b" * 64))
    with pytest.raises(ComposeLipidError):
        module.classify_prior_products(rows, [])


def test_unknown_fold_cannot_silently_lose_protection():
    with pytest.raises(ComposeLipidError, match="unsupported prior"):
        module.classify_prior_products(
            [training_row("id", "family", "CCO", "a" * 64)],
            [
                {
                    "target_id": "x",
                    "constitution": "CCO",
                    "prior_development_split": "validation_typo",
                }
            ],
        )


def pin(repo, path):
    return {"path": path.relative_to(repo).as_posix(), "sha256": str(sha256_file(path))}


def dump(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    catalogue = {"a": {}, "b": {}}
    dump(tmp_path / "catalogue.json", catalogue)
    with sqlite3.connect(tmp_path / "corpus.sqlite") as db:
        db.execute("CREATE TABLE targets(target_id TEXT, constitution TEXT, payload TEXT)")
        db.execute("CREATE TABLE assignments(target_id TEXT, family TEXT, forge_split TEXT)")
        db.execute("CREATE TABLE train_checks(target_id TEXT, constitution_id TEXT)")
        for key, family, split, smi in [
            ("blocked", "a", "train", "CCO"),
            ("safe", "a", "train", "CCC"),
            ("other", "b", "train", "CCN"),
            ("heldout", "b", "heldout", "INVALID_UNPARSED"),
        ]:
            payload = {"target_id": key, "constitution": smi, "training_admissible": False}
            db.execute("INSERT INTO targets VALUES (?,?,?)", (key, smi, json.dumps(payload)))
            db.execute("INSERT INTO assignments VALUES (?,?,?)", (key, family, split))
            if split == "train":
                db.execute(
                    "INSERT INTO train_checks VALUES (?,?)",
                    (key, hashlib.sha256(smi.encode()).hexdigest()),
                )
    imported = {
        "artifacts": {
            "corpus.sqlite": pin(tmp_path, tmp_path / "corpus.sqlite"),
            "program_catalogue.json": pin(tmp_path, tmp_path / "catalogue.json"),
        }
    }
    dump(tmp_path / "import.json", imported)
    monkeypatch.setattr(module, "verify_compose_lipid", lambda *_: imported)
    with gzip.open(tmp_path / "prior.gz", "wt") as stream:
        stream.write(
            json.dumps(
                {
                    "target_id": "old_blocked",
                    "constitution": "CCO",
                    "prior_development_split": "validation",
                }
            )
            + "\n"
        )
    config = {
        "schema_version": module.CONFIG_SCHEMA,
        "expected_families": ["a", "b"],
        "policy": module.POLICY,
        "inputs": {
            "import_result": pin(tmp_path, tmp_path / "import.json"),
            "prior_construction": pin(tmp_path, tmp_path / "prior.gz"),
        },
    }
    dump(tmp_path / "config.json", config)
    for name in module.IMPLEMENTATION:
        dest = tmp_path / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
    return tmp_path


def test_enforced_reader_accounts_for_every_family_and_preserves_source(workspace):
    result = module.build_product_protection(workspace, Path("config.json"), Path("out"))
    assert result["summary"]["totals"] == {
        "rows": 3,
        "excluded_prior_protected_product": 1,
        "eligible_for_program_preparation": 2,
        "prior_exact_identity_overlap": 1,
    }
    assert module.verify_product_protection(workspace, Path("out/result.json")) == result
    corpus = module.ProtectedPreparationCorpus(workspace, Path("out/result.json"))
    assert [r["source"]["target_id"] for r in corpus.iter_preparation_records(family="a")] == [
        "safe"
    ]
    assert [r["source"]["target_id"] for r in corpus.iter_preparation_records(family="b")] == [
        "other"
    ]
    with pytest.raises(ComposeLipidError, match="not complete"):
        corpus.iter_training_records()
    with pytest.raises(ComposeLipidError, match="unknown"):
        list(corpus.iter_preparation_records(family="absent"))
    again = module.build_product_protection(workspace, Path("config.json"), Path("replay"))
    assert (
        result["artifacts"]["preparation_view.jsonl.gz"]["sha256"]
        == again["artifacts"]["preparation_view.jsonl.gz"]["sha256"]
    )


def test_forged_unexclusion_with_refreshed_ledger_hash_is_rejected(workspace):
    result = module.build_product_protection(workspace, Path("config.json"), Path("out"))
    path = workspace / result["artifacts"]["preparation_view.jsonl.gz"]["path"]
    rows = list(module._read(path))
    rows[0]["eligible_for_program_preparation"] = True
    with gzip.open(path, "wt") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    result["artifacts"]["preparation_view.jsonl.gz"] = pin(workspace, path)
    dump(workspace / "out/result.json", result)
    with pytest.raises(ComposeLipidError, match="exclusion accounting"):
        module.verify_product_protection(workspace, Path("out/result.json"))


def test_family_cannot_be_dropped_from_configuration(workspace):
    config = json.loads((workspace / "config.json").read_text())
    config["expected_families"] = ["a"]
    dump(workspace / "config.json", config)
    with pytest.raises(ComposeLipidError, match="every imported"):
        module.build_product_protection(workspace, Path("config.json"), Path("out"))
    assert not (workspace / "out").exists()


def test_existing_output_cannot_be_overwritten(workspace):
    (workspace / "out").mkdir()
    with pytest.raises(ComposeLipidError, match="must be fresh"):
        module.build_product_protection(workspace, Path("config.json"), Path("out"))
