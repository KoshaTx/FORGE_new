"""Protected population, provenance and recomputation checks for repeated programs."""

import gzip
import hashlib
import json
import shutil
import sqlite3
from pathlib import Path

import pytest

from forge.assembly.compose_lipid import ComposeLipidError
from forge.core.hashing import PinError, sha256_file
from forge.corpus import compose_lipid_repeated_source as module
from forge.corpus.library_splits import FrozenIdentityFolds

ROOT = Path(__file__).resolve().parents[1]


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def pin(repo, path):
    return {"path": path.relative_to(repo).as_posix(), "sha256": str(sha256_file(path))}


def write_rows(path, rows):
    with gzip.open(path, "wt") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    config = json.loads(
        (ROOT / "configs/multireaction/compose_lipid_v8_a3_program_v3.json").read_text()
    )
    family = config["family"]
    source = json.loads((ROOT / config["inputs"]["adjudication"]["path"]).read_text())
    source["assets"] = {}
    registry = json.loads((ROOT / config["inputs"]["registry"]["path"]).read_text())
    registry["source_assets"] = {}
    dump(tmp_path / "registry.json", registry)
    source["registry"] = pin(tmp_path, tmp_path / "registry.json")
    dump(tmp_path / "source.json", source)
    control = source["source_controls"][0]
    components = []
    for role, smiles in control["components"].items():
        components.append(
            {"precursor_id": role, "constitution": smiles, "families": [family], "roles": [role]}
        )
    components.append(
        {
            "precursor_id": "unused_heldout",
            "constitution": "INVALID_UNPARSED_HELDOUT",
            "families": [family],
            "roles": [],
        }
    )
    write_rows(tmp_path / "precursors.jsonl.gz", components)
    write_rows(
        tmp_path / "prior.jsonl.gz",
        [
            {
                "target_id": "different_v5_id",
                "constitution": control["expected_product"],
                "prior_development_split": "test",
            }
        ],
    )
    meta = {field: role for role, field in config["role_fields"].items()}
    meta.update(events=control["events"], site_disposition="ordinary")
    rows = [
        {
            "target_id": "good",
            "primary_metadata": meta,
            "constitution": control["expected_product"],
            "training_admissible": False,
        },
        {
            "target_id": "missing",
            "primary_metadata": {**meta, "head_id": "absent"},
            "constitution": control["expected_product"],
            "training_admissible": False,
        },
        {
            "target_id": "heldout",
            "primary_metadata": {**meta, "head_id": "unused_heldout"},
            "constitution": "INVALID_UNPARSED_HELDOUT",
            "training_admissible": False,
        },
    ]
    with sqlite3.connect(tmp_path / "corpus.sqlite") as db:
        db.execute("CREATE TABLE targets(target_id TEXT, payload TEXT)")
        db.execute("CREATE TABLE assignments(target_id TEXT, forge_split TEXT, family TEXT)")
        for row in rows:
            db.execute("INSERT INTO targets VALUES (?,?)", (row["target_id"], json.dumps(row)))
            db.execute(
                "INSERT INTO assignments VALUES (?,?,?)",
                (row["target_id"], "test" if row["target_id"] == "heldout" else "train", family),
            )
    key = {
        family: {
            "variable": ["events"],
            "roles": {r: 1 for r in config["role_fields"]},
            "field_to_role": {field: [role] for role, field in config["role_fields"].items()},
        }
    }
    dump(tmp_path / "key.json", key)
    dump(tmp_path / "import-config.json", {"historical_identity_guards": []})
    imported = {
        "config": pin(tmp_path, tmp_path / "import-config.json"),
        "inputs": {"decomposition_key": pin(tmp_path, tmp_path / "key.json")},
        "artifacts": {"corpus.sqlite": pin(tmp_path, tmp_path / "corpus.sqlite")},
    }
    dump(tmp_path / "import.json", imported)
    monkeypatch.setattr(module, "verify_compose_lipid", lambda *_: imported)
    guards = FrozenIdentityFolds()
    head = control["components"]["amine_head"]
    guards.add(hashlib.sha256(head.encode()).hexdigest(), "heldout", "fixture")
    monkeypatch.setattr(module, "_frozen_guards", lambda *_: guards)
    for name in module.IMPLEMENTATION:
        dest = tmp_path / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
    config["inputs"] = {
        name: pin(tmp_path, tmp_path / value)
        for name, value in {
            "import_result": "import.json",
            "registry": "registry.json",
            "adjudication": "source.json",
            "precursor_dictionary": "precursors.jsonl.gz",
            "prior_construction": "prior.jsonl.gz",
        }.items()
    }
    dump(tmp_path / "config.json", config)
    return tmp_path


def test_complete_roundtrip_and_replay_preserve_protection_and_missing_rows(workspace):
    result = module.run_repeated_source(workspace, Path("config.json"), Path("run"))
    assert result["summary"]["counts"]["rows"] == 2
    assert result["summary"]["counts"]["computed_consistency_pass"] == 1
    assert result["summary"]["counts"]["missing_candidate_component"] == 1
    assert result["summary"]["counts"]["consistent_with_protected_precursor"] == 1
    assert result["summary"]["prior_v5_product_fold_overlaps"] == {"test": 2}
    assert result["summary"]["counts"]["historical_protected_v5_product"] == 2
    assert result["summary"]["counts"]["consistent_with_protected_v5_product"] == 1
    assert result["summary"]["training_admitted"] == 0
    assert module.verify_repeated_source(workspace, Path("run/result.json")) == result
    replay = module.run_repeated_source(workspace, Path("config.json"), Path("replay"))
    assert replay["summary"] == result["summary"]
    assert (
        replay["artifacts"]["train_replay.jsonl.gz"]["sha256"]
        == result["artifacts"]["train_replay.jsonl.gz"]["sha256"]
    )


def test_existing_output_is_not_overwritten(workspace):
    (workspace / "run").mkdir()
    with pytest.raises(ComposeLipidError, match="refusing to overwrite"):
        module.run_repeated_source(workspace, Path("config.json"), Path("run"))


def test_precursor_id_is_an_opaque_key_not_a_structure_digest(workspace):
    found = module.candidate_dictionary(workspace / "precursors.jsonl.gz", {"amine_head"})
    assert found["amine_head"]["source_id"] == "amine_head"
    assert len(found["amine_head"]["constitution_id"]) == 64
    assert "unused_heldout" not in found


def test_duplicate_component_ids_fail_even_when_structures_agree(workspace):
    rows = list(module._read_rows(workspace / "precursors.jsonl.gz"))
    write_rows(workspace / "duplicate.gz", rows + [rows[0]])
    with pytest.raises(ComposeLipidError, match="duplicate precursor"):
        module.candidate_dictionary(workspace / "duplicate.gz", {"amine_head"})


def test_modified_pinned_source_aborts_before_creating_output(workspace):
    with (workspace / "source.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(PinError):
        module.run_repeated_source(workspace, Path("config.json"), Path("run"))
    assert not (workspace / "run").exists()


@pytest.mark.parametrize(
    "field,value", [("training_ready", True), ("architecture_qualified", True)]
)
def test_config_cannot_claim_qualification(workspace, field, value):
    path = workspace / "config.json"
    config = json.loads(path.read_text())
    config[field] = value
    dump(path, config)
    with pytest.raises(ComposeLipidError, match="qualification contract"):
        module.run_repeated_source(workspace, Path("config.json"), Path("run"))


def test_forged_ledger_with_refreshed_hash_is_detected_by_recomputation(workspace):
    result = module.run_repeated_source(workspace, Path("config.json"), Path("run"))
    ledger = workspace / result["artifacts"]["train_replay.jsonl.gz"]["path"]
    rows = list(module._read_rows(ledger))
    rows[0]["training_admitted"] = True
    write_rows(ledger, rows)
    result["artifacts"]["train_replay.jsonl.gz"] = pin(workspace, ledger)
    dump(workspace / "run/result.json", result)
    with pytest.raises(ComposeLipidError, match="replay, runtime or summary"):
        module.verify_repeated_source(workspace, Path("run/result.json"))


@pytest.mark.parametrize(
    "field,value", [("training_ready", True), ("remaining_holds", []), ("source_conflicts", [])]
)
def test_result_cannot_remove_holds_or_source_conflicts(workspace, field, value):
    result = module.run_repeated_source(workspace, Path("config.json"), Path("run"))
    result[field] = value
    dump(workspace / "run/result.json", result)
    with pytest.raises(ComposeLipidError):
        module.verify_repeated_source(workspace, Path("run/result.json"))


def test_implementation_path_substitution_is_rejected(workspace):
    result = module.run_repeated_source(workspace, Path("config.json"), Path("run"))
    name = next(iter(result["implementation"]))
    substituted = workspace / "other.py"
    shutil.copyfile(workspace / name, substituted)
    result["implementation"][name] = pin(workspace, substituted)
    dump(workspace / "run/result.json", result)
    with pytest.raises(ComposeLipidError, match="path substitution"):
        module.verify_repeated_source(workspace, Path("run/result.json"))


def test_same_source_id_cannot_bind_different_constitutions(workspace):
    prior = workspace / "prior.jsonl.gz"
    write_rows(
        prior, [{"target_id": "good", "constitution": "CCC", "prior_development_split": "test"}]
    )
    config = json.loads((workspace / "config.json").read_text())
    config["inputs"]["prior_construction"] = pin(workspace, prior)
    dump(workspace / "config.json", config)
    with pytest.raises(ComposeLipidError, match="conflicting constitutional strings"):
        module.run_repeated_source(workspace, Path("config.json"), Path("run"))
    assert not (workspace / "run").exists()


def test_missing_prior_overlap_does_not_establish_holdout_disjointness(workspace):
    prior = workspace / "prior.jsonl.gz"
    write_rows(
        prior,
        [{"target_id": "different", "constitution": "CCC", "prior_development_split": "test"}],
    )
    config = json.loads((workspace / "config.json").read_text())
    config["inputs"]["prior_construction"] = pin(workspace, prior)
    dump(workspace / "config.json", config)
    result = module.run_repeated_source(workspace, Path("config.json"), Path("run"))
    assert result["summary"]["counts"]["historical_protected_v5_product"] == 0
    assert result["summary"]["prior_v5_identity_scope"] == module.PRIOR_IDENTITY_POLICY
    assert "precursor_holdouts_unqualified" in result["remaining_holds"]
    assert not result["training_ready"]
