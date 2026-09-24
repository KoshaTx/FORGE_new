"""Sequential corpus receipts preserve exclusions and independently reject forgery."""

import gzip
import hashlib
import json
import shutil
import sqlite3
from pathlib import Path

import pytest

from forge.assembly.compose_lipid import ComposeLipidError
from forge.assembly.families import constitutional_molecule
from forge.core.hashing import sha256_file
from forge.corpus import compose_lipid_protection as protection
from forge.corpus import compose_lipid_sequential as module

ROOT = Path(__file__).resolve().parents[1]
FAMILY = "thiolactone_aminolysis_michael"


def dump(path, value):
    path.write_text(json.dumps(value))


def pin(root, path):
    return {"path": path.relative_to(root).as_posix(), "sha256": str(sha256_file(path))}


def digest(smiles):
    return hashlib.sha256(constitutional_molecule(smiles)[0].encode()).hexdigest()


def read_rows(path):
    with gzip.open(path, "rt") as stream:
        return [json.loads(line) for line in stream]


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    config = json.loads(
        (ROOT / "configs/multireaction/compose_lipid_v8_staar_program_v1.json").read_text()
    )
    source = json.loads((ROOT / config["inputs"]["adjudication"]["path"]).read_text())
    registry = json.loads((ROOT / config["inputs"]["registry"]["path"]).read_text())
    keys = json.loads((ROOT / config["inputs"]["decomposition_key"]["path"]).read_text())
    dump(tmp_path / "key.json", {FAMILY: keys[FAMILY]})
    assets = {}
    for name, original in source["assets"].items():
        target = tmp_path / ("key.json" if name == "decomposition_key" else name)
        if name in ("stage_contract.json", "control_transcriptions.json"):
            shutil.copyfile(ROOT / original["path"], target)
        elif name != "decomposition_key":
            target.write_text("unit-test source evidence bytes")
        assets[name] = pin(tmp_path, target)
    registry["source_assets"] = source["assets"] = assets
    for reaction in registry["reactions"]:
        for name, original in reaction["implementation"].items():
            target = tmp_path / Path(original["path"]).name
            shutil.copyfile(ROOT / original["path"], target)
            reaction["implementation"][name] = pin(tmp_path, target)
    dump(tmp_path / "registry.json", registry)
    source["registry"] = pin(tmp_path, tmp_path / "registry.json")
    dump(tmp_path / "source.json", source)
    controls = json.loads((tmp_path / "control_transcriptions.json").read_text())["controls"]
    first, second = [c["expected_product"] for c in controls]
    prior = first.replace("CN(C)CCN", "CN(C)CCCCN", 1)

    def meta(label, head=None):
        return {
            "head_id": head or label + "_head",
            "thiolactone_id": label + "_thiolactone",
            "acrylate_id": label + "_acrylate",
        }

    records = [
        ("prior_product", "train", prior, meta("p")),
        ("good", "train", first, meta("g")),
        ("protected", "train", second, meta("s")),
        ("ambiguous", "train", first.replace("CN(C)CCN", "CNCCCN", 1), meta("a")),
        ("outside", "train", first.replace("C(CCCCCC)CCCCCCCC", "C(CCSCC)CCCCCCCC"), meta("o")),
        ("missing", "train", "CCN", {}),
        ("no_inverse", "train", "CCN", meta("n")),
        ("conflict_a", "train", first, meta("ca", "conflict_head")),
        (
            "conflict_b",
            "train",
            first.replace("CN(C)CCN", "CN(C)CCCN", 1),
            meta("cb", "conflict_head"),
        ),
        ("heldout", "heldout", "INVALID_UNPARSED_HOLDOUT", {}),
    ]
    with sqlite3.connect(tmp_path / "corpus.sqlite") as db:
        db.execute("CREATE TABLE targets(target_id TEXT,constitution TEXT,payload TEXT)")
        db.execute("CREATE TABLE assignments(target_id TEXT,family TEXT,forge_split TEXT)")
        db.execute("CREATE TABLE train_checks(target_id TEXT,constitution_id TEXT)")
        for label, fold, smiles, metadata in records:
            payload = {
                "target_id": label,
                "primary_family": FAMILY,
                "primary_metadata": metadata,
                "constitution": smiles,
                "training_admissible": False,
            }
            db.execute("INSERT INTO targets VALUES(?,?,?)", (label, smiles, json.dumps(payload)))
            db.execute("INSERT INTO assignments VALUES(?,?,?)", (label, FAMILY, fold))
            if fold == "train":
                db.execute("INSERT INTO train_checks VALUES(?,?)", (label, digest(smiles)))
    dump(
        tmp_path / "guards.json",
        [{"identity": digest(controls[1]["components"]["acrylate_tail"]), "fold": "heldout"}],
    )
    dump(
        tmp_path / "import_config.json",
        {
            "historical_identity_guards": [
                {
                    "input": "guards",
                    "format": "json",
                    "identity_kind": "constitutional_digest",
                    "identity_field": "identity",
                    "fold_fields": ["fold"],
                }
            ]
        },
    )
    dump(tmp_path / "catalogue.json", {FAMILY: {}})
    imported = {
        "config": pin(tmp_path, tmp_path / "import_config.json"),
        "inputs": {
            "guards": pin(tmp_path, tmp_path / "guards.json"),
            "decomposition_key": pin(tmp_path, tmp_path / "key.json"),
        },
        "artifacts": {
            "corpus.sqlite": pin(tmp_path, tmp_path / "corpus.sqlite"),
            "program_catalogue.json": pin(tmp_path, tmp_path / "catalogue.json"),
        },
    }
    dump(tmp_path / "import.json", imported)
    # The import unit is separate; the actual protection builder/reader is exercised here.
    monkeypatch.setattr(protection, "verify_compose_lipid", lambda *_: imported)
    with gzip.open(tmp_path / "prior.gz", "wt") as stream:
        stream.write(
            json.dumps(
                {
                    "target_id": "earlier",
                    "constitution": prior,
                    "prior_development_split": "validation",
                }
            )
            + "\n"
        )
    dump(
        tmp_path / "protect.json",
        {
            "schema_version": protection.CONFIG_SCHEMA,
            "policy": protection.POLICY,
            "expected_families": [FAMILY],
            "inputs": {
                "import_result": pin(tmp_path, tmp_path / "import.json"),
                "prior_construction": pin(tmp_path, tmp_path / "prior.gz"),
            },
        },
    )
    for name in set(module.IMPLEMENTATION) | set(protection.IMPLEMENTATION):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    protection.build_product_protection(tmp_path, Path("protect.json"), Path("protection"))
    config["inputs"] = {
        "registry": pin(tmp_path, tmp_path / "registry.json"),
        "adjudication": pin(tmp_path, tmp_path / "source.json"),
        "decomposition_key": pin(tmp_path, tmp_path / "key.json"),
        "protection_result": pin(tmp_path, tmp_path / "protection/result.json"),
    }
    dump(tmp_path / "config.json", config)
    return tmp_path


def test_complete_population_preserves_protection_ambiguity_and_label_conflicts(workspace):
    result = module.run_sequential_program(workspace, Path("config.json"), Path("out"))
    rows = {r["target_id"]: r for r in read_rows(workspace / "out/programs.jsonl.gz")}
    assert set(rows) == {
        "good",
        "protected",
        "ambiguous",
        "outside",
        "missing",
        "no_inverse",
        "conflict_a",
        "conflict_b",
    }
    assert rows["good"]["eligible_after_known_exclusions"]
    assert rows["protected"]["computed_consistency_pass"]
    assert rows["protected"]["historical_protected_precursor"]
    assert not rows["protected"]["eligible_after_known_exclusions"]
    assert rows["ambiguous"]["status"] == "excluded_program_or_component_contract"
    assert rows["outside"]["status"] == "excluded_program_or_component_contract"
    assert rows["missing"]["status"] == "excluded_missing_component_labels"
    assert rows["no_inverse"]["status"] == "excluded_no_complete_inverse"
    for key in ("conflict_a", "conflict_b"):
        assert rows[key]["component_label_conflict"]
        assert not rows[key]["eligible_after_known_exclusions"]
    assert all(not r["training_admitted"] for r in rows.values())
    assert module.verify_sequential_program(workspace, Path("out/result.json")) == result


@pytest.mark.parametrize(
    "mutation", ["drop_row", "admit_ambiguous", "unprotect", "conceal_conflict", "claim_ready"]
)
def test_rehashed_forgery_does_not_replace_independent_replay(workspace, mutation):
    result = module.run_sequential_program(workspace, Path("config.json"), Path("out"))
    path = workspace / "out/programs.jsonl.gz"
    rows = read_rows(path)
    if mutation == "drop_row":
        rows.pop()
    elif mutation == "claim_ready":
        result["summary"]["training_ready"] = True
    else:
        key = {
            "admit_ambiguous": "ambiguous",
            "unprotect": "protected",
            "conceal_conflict": "conflict_a",
        }[mutation]
        row = next(r for r in rows if r["target_id"] == key)
        row.update(
            computed_consistency_pass=True,
            historical_protected_precursor=False,
            component_label_conflict=False,
            eligible_after_known_exclusions=True,
        )
    with gzip.open(path, "wt") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    result["artifacts"]["programs.jsonl.gz"] = pin(workspace, path)
    dump(workspace / "out/result.json", result)
    with pytest.raises(ComposeLipidError, match="independent replay"):
        module.verify_sequential_program(workspace, Path("out/result.json"))


def test_source_stage_mutation_is_rejected_even_with_fresh_registry_pin(workspace):
    path = workspace / "registry.json"
    registry = json.loads(path.read_text())
    registry["programs"][0]["stages"].reverse()
    dump(path, registry)
    source = json.loads((workspace / "source.json").read_text())
    source["registry"] = pin(workspace, path)
    dump(workspace / "source.json", source)
    config = json.loads((workspace / "config.json").read_text())
    config["inputs"]["registry"] = pin(workspace, path)
    config["inputs"]["adjudication"] = pin(workspace, workspace / "source.json")
    dump(workspace / "config.json", config)
    with pytest.raises(ComposeLipidError, match="primary source contract"):
        module.run_sequential_program(workspace, Path("config.json"), Path("out"))
    assert not (workspace / "out").exists()


def test_replay_is_deterministic_and_never_overwrites_completed_output(workspace):
    a = module.run_sequential_program(workspace, Path("config.json"), Path("a"))
    b = module.run_sequential_program(workspace, Path("config.json"), Path("b"))
    assert (
        a["artifacts"]["programs.jsonl.gz"]["sha256"]
        == b["artifacts"]["programs.jsonl.gz"]["sha256"]
    )
    with pytest.raises(ComposeLipidError, match="fresh"):
        module.run_sequential_program(workspace, Path("config.json"), Path("a"))
